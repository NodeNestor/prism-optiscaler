"""
FAST TartanAir processor — reads extracted raw files instead of zips.
Should be 100+ samples/sec on NVMe.

Usage:
  python generate_tartanair_fast.py --input E:/tartanair-extracted --output I:/prism-dataset/dataset-tartanair-v2
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random


def halton(index, base):
    result, f, i = 0.0, 1.0, index
    while i > 0:
        f /= base
        result += f * (i % base)
        i //= base
    return result


def process_env(env_dir: Path, output_dir: Path, start_idx: int,
                max_frames: int = 2000, crop_size: int = 128) -> int:
    """Process one extracted environment. Returns number of samples written."""
    img_dir = env_dir / "image_lcam_front"
    depth_dir = env_dir / "depth_lcam_front"
    flow_dir = env_dir / "flow_lcam_front"

    if not img_dir.exists():
        return 0

    # Find all image files (could be nested in subdirectories from zip extraction)
    img_files = sorted(img_dir.rglob("*.png"))
    if not img_files:
        return 0

    depth_files = sorted(depth_dir.rglob("*.npy")) if depth_dir.exists() else []
    flow_files = sorted(flow_dir.rglob("*.npy")) if flow_dir.exists() else []

    n = min(len(img_files), max_frames)
    idx = start_idx
    scales = [2, 3]  # both 2x and 3x

    for i in range(n):
        # Read image
        img = cv2.imread(str(img_files[i]))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        # Read depth
        if i < len(depth_files):
            depth = np.load(str(depth_files[i])).astype(np.float32)
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            far = np.percentile(depth[depth < 1e4], 99) if (depth < 1e4).any() else 100.0
            depth = np.clip(depth / far, 0, 1)
            depth = 1.0 - depth  # reversed-Z
        else:
            depth = np.zeros((img_h, img_w), dtype=np.float32)

        # Read flow
        if i < len(flow_files):
            flow = np.load(str(flow_files[i])).astype(np.float32)
            if flow.ndim == 3 and flow.shape[2] >= 2:
                flow = flow[:, :, :2]
            else:
                flow = np.zeros((img_h, img_w, 2), dtype=np.float32)
        else:
            flow = np.zeros((img_h, img_w, 2), dtype=np.float32)

        # Generate at each scale
        for scale in scales:
            display_h, display_w = 1080, 1920
            render_h = display_h // scale
            render_w = display_w // scale

            # Jitter
            jx = halton(i + 1, 2) - 0.5
            jy = halton(i + 1, 3) - 0.5

            # Resize everything
            gt = cv2.resize(img, (display_w, display_h), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255.0
            color = cv2.resize(img, (render_w, render_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            d = cv2.resize(depth, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
            mv = cv2.resize(flow, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
            mv[:, :, 0] *= render_w / img_w
            mv[:, :, 1] *= render_h / img_h

            # Crop
            cr = min(crop_size, render_h, render_w)
            cd = cr * scale
            if render_h > cr and render_w > cr:
                y = random.randint(0, render_h - cr)
                x = random.randint(0, render_w - cr)
                dy = int(y * display_h / render_h)
                dx = int(x * display_w / render_w)
                dh = int(cr * display_h / render_h)
                dw = int(cr * display_w / render_w)
            else:
                y, x, dy, dx = 0, 0, 0, 0
                dh, dw = display_h, display_w

            gt_crop = cv2.resize(gt[dy:dy+dh, dx:dx+dw], (cd, cd), interpolation=cv2.INTER_LINEAR)

            sample = {
                "color": torch.from_numpy(color[y:y+cr, x:x+cr]).permute(2, 0, 1).half(),
                "depth": torch.from_numpy(d[y:y+cr, x:x+cr]).unsqueeze(0).half(),
                "motion_vectors": torch.from_numpy(mv[y:y+cr, x:x+cr]).permute(2, 0, 1).half(),
                "ground_truth": torch.from_numpy(gt_crop).permute(2, 0, 1).half(),
                "is_real": torch.tensor(False, dtype=torch.bool),
            }

            torch.save(sample, output_dir / f"sample_{idx:06d}.pt")
            idx += 1

    return idx - start_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("E:/tartanair-extracted"))
    parser.add_argument("--output", type=Path, default=Path("I:/prism-dataset/dataset-tartanair-v2"))
    parser.add_argument("--max-per-env", type=int, default=2000)
    parser.add_argument("--crop", type=int, default=128)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    envs = sorted([d for d in args.input.iterdir() if d.is_dir()])
    print(f"Found {len(envs)} extracted environments")

    total = 0
    for env in envs:
        n = process_env(env, args.output, start_idx=total,
                        max_frames=args.max_per_env, crop_size=args.crop)
        if n > 0:
            print(f"  {env.name}: {n} samples ({n//2} frames x 2 scales)")
        total += n

    print(f"\nDone: {total} total samples in {args.output}")


if __name__ == "__main__":
    main()
