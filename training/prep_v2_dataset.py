"""
Prepare the v2 training dataset — merge all sources, pre-crop to fast drive.

Merges:
  - Original pexels-nature (is_real=True)
  - New multi-source real video (is_real=True)
  - TartanAir synthetic (is_real=False)

Pre-crops everything to 128x128 with ORIGINAL scales (2x + 3x).
Copies to E: drive for fast NVMe training.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import random

def crop_sample(data: dict, crop_size: int = 128) -> dict:
    """Crop to fixed render-res size, preserve original scale for ground truth."""
    _, rH, rW = data["color"].shape
    _, dH, dW = data["ground_truth"].shape
    scale = dH / rH

    cr = min(crop_size, rH, rW)
    cd = int(cr * scale)

    if rH > cr and rW > cr:
        y = random.randint(0, rH - cr)
        x = random.randint(0, rW - cr)
        dy = int(y * dH / rH)
        dx = int(x * dW / rW)
        dh = int(cr * dH / rH)
        dw = int(cr * dW / rW)
    else:
        y, x, dy, dx = 0, 0, 0, 0
        dh, dw = dH, dW

    result = {
        "color": data["color"][:, y:y+cr, x:x+cr].clone().half(),
        "depth": data["depth"][:, y:y+cr, x:x+cr].clone().half(),
        "motion_vectors": data["motion_vectors"][:, y:y+cr, x:x+cr].clone().half(),
        "ground_truth": F.interpolate(
            data["ground_truth"][:, dy:dy+dh, dx:dx+dw].unsqueeze(0).float(),
            size=(cd, cd), mode="bilinear", align_corners=False
        ).squeeze(0).half().clone(),
    }
    if "is_real" in data:
        result["is_real"] = data["is_real"]
    return result


def main():
    sources = [
        ("I:/prism-dataset/dataset", "original pexels-nature"),
        ("I:/prism-dataset/dataset-v2", "new multi-source"),
        ("I:/prism-dataset/dataset-tartanair", "TartanAir synthetic"),
    ]

    output = Path("E:/prism-dataset-v2")
    output.mkdir(parents=True, exist_ok=True)

    all_files = []
    for src_dir, name in sources:
        src = Path(src_dir)
        files = sorted(src.glob("sample_*.pt"))
        print(f"{name}: {len(files)} samples from {src}")
        all_files.extend(files)

    print(f"\nTotal: {len(all_files)} samples")
    print("Shuffling...")
    random.seed(42)
    random.shuffle(all_files)

    print(f"Pre-cropping to {output} with mixed scales...")
    valid = 0
    real_count = 0
    synth_count = 0
    scale_counts = {}

    for i, path in enumerate(tqdm(all_files)):
        try:
            data = torch.load(path, weights_only=True)
            cropped = crop_sample(data, crop_size=128)

            # Track stats
            is_real = cropped.get("is_real", torch.tensor(True)).item()
            if is_real:
                real_count += 1
            else:
                synth_count += 1

            scale = round(cropped["ground_truth"].shape[1] / cropped["color"].shape[1], 1)
            scale_counts[scale] = scale_counts.get(scale, 0) + 1

            torch.save(cropped, output / f"sample_{valid:06d}.pt")
            valid += 1
        except Exception:
            continue

        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(all_files)} processed, {valid} valid")

    print(f"\n{'='*60}")
    print(f"V2 Dataset Ready!")
    print(f"  Total samples: {valid}")
    print(f"  Real video: {real_count} ({real_count/valid*100:.0f}%)")
    print(f"  Synthetic: {synth_count} ({synth_count/valid*100:.0f}%)")
    print(f"  Scale distribution:")
    for s in sorted(scale_counts):
        print(f"    {s}x: {scale_counts[s]} ({scale_counts[s]/valid*100:.0f}%)")
    print(f"  Output: {output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
