"""
Dataset generator — converts video into G-buffer training pairs.

Features:
  - Multi-resolution: generates samples at different render→display scales
  - Game-style degradation: makes real video color look more like game renders
  - Temporal pairs: consecutive frames for training temporal models
  - Depth via Depth Anything V2
  - Optical flow via RAFT (torchvision)
  - Halton jitter (identical to DLSS/FSR)

Usage:
  python generate_dataset.py --videos data/videos --output data/dataset
  python generate_dataset.py --videos data/videos --output data/dataset --game-style medium
  python generate_dataset.py --videos data/videos --output data/dataset --multi-res
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from game_stylize import apply_game_style


# ============================================================================
# Jitter — Halton sequence, identical to DLSS/FSR
# ============================================================================

def halton(index: int, base: int) -> float:
    result, f, i = 0.0, 1.0, index
    while i > 0:
        f /= base
        result += f * (i % base)
        i //= base
    return result


def get_jitter(frame_idx: int) -> tuple[float, float]:
    return halton(frame_idx + 1, 2) - 0.5, halton(frame_idx + 1, 3) - 0.5


# ============================================================================
# Depth estimation
# ============================================================================

class DepthEstimator:
    def __init__(self, device: torch.device, model_size: str = "small"):
        from transformers import pipeline
        model_name = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
        print(f"Loading depth model: {model_name}")
        self.pipe = pipeline("depth-estimation", model=model_name, device=device)

    @torch.no_grad()
    def predict(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        from PIL import Image
        pil_img = Image.fromarray(image_rgb_uint8)
        result = self.pipe(pil_img)
        depth = np.array(result["depth"], dtype=np.float32)
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
        return 1.0 - depth  # reversed-Z: near=1, far=0


# ============================================================================
# Optical flow
# ============================================================================

class FlowEstimator:
    def __init__(self, device: torch.device):
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        print("Loading RAFT-Small optical flow model")
        self.weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=self.weights).to(device).eval()
        self.device = device
        self.transforms = self.weights.transforms()

    @torch.no_grad()
    def predict(self, prev_rgb: np.ndarray, curr_rgb: np.ndarray, max_size: int = 720) -> np.ndarray:
        """Returns optical flow. Downscales to max_size if needed to fit in VRAM."""
        h, w = prev_rgb.shape[:2]
        scale = 1.0
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            # Make divisible by 8 for RAFT
            new_h = (new_h // 8) * 8
            new_w = (new_w // 8) * 8
            prev_rgb = cv2.resize(prev_rgb, (new_w, new_h))
            curr_rgb = cv2.resize(curr_rgb, (new_w, new_h))

        prev_t = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        curr_t = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        prev_t, curr_t = self.transforms(prev_t, curr_t)
        flow_list = self.model(prev_t, curr_t)
        flow = flow_list[-1][0].cpu().permute(1, 2, 0).numpy()

        # Scale flow back to original resolution
        if scale != 1.0:
            flow = cv2.resize(flow, (w, h))
            flow /= scale  # flow values scale with resolution

        return flow


# ============================================================================
# Multi-resolution configs — matching DLSS quality presets
# ============================================================================

RESOLUTION_PRESETS = {
    # name: (render_h, render_w) for 1080p display
    "ultra_performance": (360, 640),    # 3.0× scale
    "performance":       (540, 960),    # 2.0× scale
    "balanced":          (635, 1129),   # ~1.7× scale
    "quality":           (720, 1280),   # 1.5× scale
}


# ============================================================================
# Frame pair generation
# ============================================================================

def process_frame_pair(
    prev_frame: np.ndarray | None,
    curr_frame: np.ndarray,
    frame_idx: int,
    depth_est: DepthEstimator,
    flow_est: FlowEstimator,
    render_size: tuple[int, int],
    display_size: tuple[int, int],
    game_style: str = "random",
    rng: np.random.Generator | None = None,
) -> dict:
    display_h, display_w = display_size
    render_h, render_w = render_size
    scale_x = display_w / render_w
    scale_y = display_h / render_h

    # Ground truth — HD frame as float [0, 1]
    ground_truth = curr_frame.astype(np.float32) / 255.0
    if ground_truth.shape[:2] != (display_h, display_w):
        ground_truth = cv2.resize(ground_truth, (display_w, display_h), interpolation=cv2.INTER_LANCZOS4)

    # Jitter
    jx, jy = get_jitter(frame_idx)

    # Color — downsample with jitter + game stylization
    M = np.float32([[1, 0, -jx * scale_x], [0, 1, -jy * scale_y]])
    shifted = cv2.warpAffine(ground_truth, M, (display_w, display_h), borderMode=cv2.BORDER_REFLECT)
    color = cv2.resize(shifted, (render_w, render_h), interpolation=cv2.INTER_AREA)

    # Apply game-style degradation to the COLOR input
    color = apply_game_style(color, style=game_style, rng=rng)

    # Depth
    depth_hd = depth_est.predict(curr_frame)
    if depth_hd.shape[:2] != (display_h, display_w):
        depth_hd = cv2.resize(depth_hd, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth_hd, (render_w, render_h), interpolation=cv2.INTER_LINEAR)

    # Motion vectors
    if prev_frame is not None:
        prev_r = cv2.resize(prev_frame, (display_w, display_h), interpolation=cv2.INTER_LANCZOS4)
        curr_r = cv2.resize(curr_frame, (display_w, display_h), interpolation=cv2.INTER_LANCZOS4)
        flow_hd = flow_est.predict(prev_r, curr_r)
        mv = cv2.resize(flow_hd, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
        mv[..., 0] /= scale_x
        mv[..., 1] /= scale_y
    else:
        mv = np.zeros((render_h, render_w, 2), dtype=np.float32)

    return {
        "color": torch.from_numpy(color).permute(2, 0, 1).half(),
        "depth": torch.from_numpy(depth).unsqueeze(0).float(),
        "motion_vectors": torch.from_numpy(mv).permute(2, 0, 1).half(),
        "jitter": torch.tensor([jx, jy], dtype=torch.float32),
        "mv_scale": torch.tensor([render_w, render_h], dtype=torch.float32),
        "ground_truth": torch.from_numpy(ground_truth).permute(2, 0, 1).half(),
        "render_size": torch.tensor([render_h, render_w], dtype=torch.int32),
        "display_size": torch.tensor([display_h, display_w], dtype=torch.int32),
        "is_real": torch.tensor(1, dtype=torch.bool),  # True = real video, False = synthetic
    }


# ============================================================================
# Video extraction
# ============================================================================

def extract_frames(video_path: Path, max_frames: int = 0, skip: int = 0) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {video_path.name}: {total} frames @ {fps:.1f} FPS")

    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if skip and idx % (skip + 1) != 0:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return frames


# ============================================================================
# Main
# ============================================================================

def generate_dataset(
    video_paths: list[Path],
    output_dir: Path,
    display_size: tuple[int, int] = (1080, 1920),
    resolution_presets: list[str] | None = None,
    max_frames_per_video: int = 3000,
    skip_frames: int = 1,
    game_style: str = "random",
    device: str = "cuda",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    if resolution_presets is None:
        resolution_presets = ["performance"]

    presets = {k: v for k, v in RESOLUTION_PRESETS.items() if k in resolution_presets}
    print(f"Device: {dev}")
    print(f"Display: {display_size[1]}×{display_size[0]}")
    print(f"Render presets: {presets}")
    print(f"Game style: {game_style}")

    depth_est = DepthEstimator(dev, model_size="small")
    flow_est = FlowEstimator(dev)
    rng = np.random.default_rng(42)

    sample_idx = 0

    for video_path in video_paths:
        print(f"\n{'='*60}\nProcessing: {video_path.name}\n{'='*60}")
        frames = extract_frames(video_path, max_frames=max_frames_per_video, skip=skip_frames)

        for preset_name, render_size in presets.items():
            print(f"\n  Preset: {preset_name} ({render_size[1]}×{render_size[0]})")
            prev_frame = None

            for i, frame in enumerate(tqdm(frames, desc=f"  {preset_name}")):
                sample = process_frame_pair(
                    prev_frame=prev_frame,
                    curr_frame=frame,
                    frame_idx=i,
                    depth_est=depth_est,
                    flow_est=flow_est,
                    render_size=render_size,
                    display_size=display_size,
                    game_style=game_style,
                    rng=rng,
                )

                torch.save(sample, output_dir / f"sample_{sample_idx:06d}.pt")
                sample_idx += 1
                prev_frame = frame

    print(f"\n{'='*60}\nDataset complete: {sample_idx} samples in {output_dir}\n{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate G-buffer training dataset from video")
    parser.add_argument("--videos", type=Path, default=Path("data/videos"))
    parser.add_argument("--output", type=Path, default=Path("data/dataset"))
    parser.add_argument("--display-h", type=int, default=1080)
    parser.add_argument("--display-w", type=int, default=1920)
    parser.add_argument("--max-frames", type=int, default=3000)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--game-style", default="random",
                        choices=["none", "light", "medium", "heavy", "random"],
                        help="Game-style degradation filter for color input")
    parser.add_argument("--multi-res", action="store_true",
                        help="Generate at all DLSS presets (performance/balanced/quality/ultra)")
    parser.add_argument("--presets", nargs="+", default=["performance"],
                        choices=list(RESOLUTION_PRESETS.keys()),
                        help="Which resolution presets to generate")
    args = parser.parse_args()

    if args.multi_res:
        args.presets = list(RESOLUTION_PRESETS.keys())

    from video_sources import list_local_videos
    videos = list_local_videos(args.videos)
    if not videos:
        print("No videos found! Run: python video_sources.py --output data/videos")
        exit(1)

    generate_dataset(
        video_paths=videos,
        output_dir=args.output,
        display_size=(args.display_h, args.display_w),
        resolution_presets=args.presets,
        max_frames_per_video=args.max_frames,
        skip_frames=args.skip,
        game_style=args.game_style,
        device=args.device,
    )
