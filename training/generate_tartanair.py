"""
TartanAir V2 → Prism dataset converter.

Reads TartanAir zip files (image + depth + flow) and converts them
to the same .pt format as generate_dataset.py, with is_real=False.

TartanAir provides PERFECT ground truth:
  - image: RGB PNG (640x480 typically)
  - depth: float32 .npy (distance in meters)
  - flow: float32 .npy [H, W, 2] (optical flow in pixels)

We convert to match the game's G-buffer format:
  - color: downsampled + jittered (render res)
  - depth: normalized to [0,1] reversed-Z
  - motion_vectors: flow in pixel space at render res
  - ground_truth: original image at display res

Usage:
  python generate_tartanair.py --input I:/prism-dataset/raw/tartanair --output I:/prism-dataset/dataset
  python generate_tartanair.py --input I:/prism-dataset/raw/tartanair --output I:/prism-dataset/dataset --max-per-env 500
"""

import argparse
import zipfile
import io
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


# ============================================================================
# Jitter — same Halton sequence as games and our real video pipeline
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
# Game-style degradation (reuse from game_stylize.py)
# ============================================================================

from game_stylize import apply_game_style


# ============================================================================
# TartanAir zip reader
# ============================================================================

class TartanAirEnvReader:
    """Reads image/depth/flow from TartanAir zip files for one environment."""

    def __init__(self, env_dir: Path):
        self.env_dir = env_dir
        self.env_name = env_dir.name

        # Find zip files
        self.image_zip = self._find_zip("image_lcam_front.zip")
        self.depth_zip = self._find_zip("depth_lcam_front.zip")
        self.flow_zip = self._find_zip("flow_lcam_front.zip")

        if not self.image_zip:
            raise FileNotFoundError(f"No image zip in {env_dir}")

    def _find_zip(self, name: str) -> Path | None:
        # Check Data_easy/ subfolder
        p = self.env_dir / "Data_easy" / name
        if p.exists():
            return p
        # Check direct
        p = self.env_dir / name
        if p.exists():
            return p
        return None

    def read_frames(self, max_frames: int = 0) -> list[dict]:
        """Read all frames from zips. Returns list of {image, depth, flow} dicts."""
        frames = []

        with zipfile.ZipFile(self.image_zip, "r") as img_z:
            # Get sorted PNG filenames
            img_files = sorted([f for f in img_z.namelist() if f.endswith(".png")])

            depth_z = zipfile.ZipFile(self.depth_zip, "r") if self.depth_zip else None
            flow_z = zipfile.ZipFile(self.flow_zip, "r") if self.flow_zip else None

            depth_files = sorted([f for f in depth_z.namelist() if f.endswith(".npy")]) if depth_z else []
            flow_files = sorted([f for f in flow_z.namelist() if f.endswith(".npy")]) if flow_z else []

            for i, img_name in enumerate(img_files):
                if max_frames and i >= max_frames:
                    break

                # Read image
                img_data = img_z.read(img_name)
                img_arr = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                if img_arr is None:
                    continue
                img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

                frame = {"image": img_rgb}

                # Read depth
                if depth_z and i < len(depth_files):
                    depth_data = depth_z.read(depth_files[i])
                    depth_arr = np.load(io.BytesIO(depth_data))
                    frame["depth"] = depth_arr

                # Read flow (flow is between frame i and i+1)
                if flow_z and i < len(flow_files):
                    flow_data = flow_z.read(flow_files[i])
                    flow_arr = np.load(io.BytesIO(flow_data))
                    frame["flow"] = flow_arr

                frames.append(frame)

            if depth_z:
                depth_z.close()
            if flow_z:
                flow_z.close()

        return frames


# ============================================================================
# Convert frames to dataset samples
# ============================================================================

def convert_frame(
    frame: dict,
    prev_frame: dict | None,
    frame_idx: int,
    render_size: tuple[int, int],
    display_size: tuple[int, int],
    game_style: str = "random",
    rng: np.random.Generator | None = None,
) -> dict:
    display_h, display_w = display_size
    render_h, render_w = render_size
    scale_x = display_w / render_w
    scale_y = display_h / render_h

    img = frame["image"]  # uint8 RGB

    # Ground truth — resize to display res, normalize to [0,1]
    ground_truth = cv2.resize(img, (display_w, display_h), interpolation=cv2.INTER_LANCZOS4)
    ground_truth = ground_truth.astype(np.float32) / 255.0

    # Jitter
    jx, jy = get_jitter(frame_idx)

    # Color — downsample with jitter + game stylization
    M = np.float32([[1, 0, -jx * scale_x], [0, 1, -jy * scale_y]])
    shifted = cv2.warpAffine(ground_truth, M, (display_w, display_h), borderMode=cv2.BORDER_REFLECT)
    color = cv2.resize(shifted, (render_w, render_h), interpolation=cv2.INTER_AREA)
    color = apply_game_style(color, style=game_style, rng=rng)

    # Depth — TartanAir gives distance in meters, convert to [0,1] reversed-Z
    if "depth" in frame:
        depth_raw = frame["depth"].astype(np.float32)
        # Handle multi-channel depth (sometimes TartanAir has extra dims)
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        # Normalize: clip far plane, normalize to [0,1], reverse
        far_plane = np.percentile(depth_raw[depth_raw < 1e4], 99) if (depth_raw < 1e4).any() else 100.0
        depth_norm = np.clip(depth_raw / far_plane, 0, 1)
        depth_norm = 1.0 - depth_norm  # reversed-Z: near=1, far=0
        depth = cv2.resize(depth_norm, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
    else:
        depth = np.zeros((render_h, render_w), dtype=np.float32)

    # Motion vectors — TartanAir flow is [H, W, 2] in pixels at source res
    if "flow" in frame:
        flow_raw = frame["flow"].astype(np.float32)
        if flow_raw.ndim == 3 and flow_raw.shape[2] >= 2:
            flow_raw = flow_raw[:, :, :2]
        # Resize flow to render res and scale values
        src_h, src_w = flow_raw.shape[:2]
        mv = cv2.resize(flow_raw, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
        mv[:, :, 0] *= render_w / src_w
        mv[:, :, 1] *= render_h / src_h
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
        "is_real": torch.tensor(0, dtype=torch.bool),  # SYNTHETIC — discriminator won't use as "real"
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Convert TartanAir V2 to Prism dataset format")
    parser.add_argument("--input", type=Path, default=Path("I:/prism-dataset/raw/tartanair"))
    parser.add_argument("--output", type=Path, default=Path("I:/prism-dataset/dataset"))
    parser.add_argument("--render-h", type=int, default=540)
    parser.add_argument("--render-w", type=int, default=960)
    parser.add_argument("--display-h", type=int, default=1080)
    parser.add_argument("--display-w", type=int, default=1920)
    parser.add_argument("--max-per-env", type=int, default=500,
                        help="Max frames to sample per environment (0=all)")
    parser.add_argument("--game-style", default="heavy",
                        choices=["none", "light", "medium", "heavy", "random"],
                        help="Game-style degradation (heavy recommended for synthetic)")
    parser.add_argument("--multi-res", action="store_true",
                        help="Also generate at 720p and 360p render res")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Starting sample index (for appending to existing dataset)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    render_sizes = [(args.render_h, args.render_w)]
    if args.multi_res:
        render_sizes += [(360, 640), (720, 1280)]

    display_size = (args.display_h, args.display_w)

    # Find all environments
    env_dirs = sorted([d for d in args.input.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"Found {len(env_dirs)} environments in {args.input}")
    print(f"Render sizes: {render_sizes}")
    print(f"Display size: {display_size}")
    print(f"Max per env: {args.max_per_env}")
    print(f"Game style: {args.game_style}")
    print()

    sample_idx = args.start_idx
    total_envs_ok = 0

    for env_dir in env_dirs:
        # Check if this env has the Data_easy subfolder with zips
        data_dir = env_dir / "Data_easy"
        if not data_dir.exists():
            data_dir = env_dir
        img_zip = data_dir / "image_lcam_front.zip"
        if not img_zip.exists():
            print(f"  [{env_dir.name}] No image zip, skipping")
            continue

        print(f"[{env_dir.name}]")
        try:
            reader = TartanAirEnvReader(env_dir)
            frames = reader.read_frames(max_frames=args.max_per_env)
            print(f"  Read {len(frames)} frames")
        except Exception as e:
            print(f"  Error reading: {e}")
            continue

        if not frames:
            continue

        total_envs_ok += 1

        for render_size in render_sizes:
            desc = f"  {env_dir.name} {render_size[1]}x{render_size[0]}"
            prev_frame = None
            for i, frame in enumerate(tqdm(frames, desc=desc)):
                sample = convert_frame(
                    frame=frame,
                    prev_frame=prev_frame,
                    frame_idx=i,
                    render_size=render_size,
                    display_size=display_size,
                    game_style=args.game_style,
                    rng=rng,
                )
                torch.save(sample, args.output / f"sample_{sample_idx:06d}.pt")
                sample_idx += 1
                prev_frame = frame

    print(f"\n{'='*60}")
    print(f"TartanAir conversion complete")
    print(f"  Environments processed: {total_envs_ok}")
    print(f"  Samples generated: {sample_idx - args.start_idx}")
    print(f"  Output: {args.output}")
    print(f"  Sample index range: {args.start_idx} - {sample_idx - 1}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
