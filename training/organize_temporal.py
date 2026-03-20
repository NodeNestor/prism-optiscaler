"""
Organize dataset for temporal training.

Takes the merged dataset and reorganizes it so consecutive samples
form valid temporal sequences (same crop position, consecutive frames).

For TartanAir (5 crops per frame):
  - Groups every 5 samples (same source frame)
  - Takes crop position from the first crop
  - Applies SAME crop position to consecutive source frames
  - Creates sequences of [frame_n, frame_n+1, frame_n+2] with consistent crops

For real video:
  - Already processed as consecutive frames from video
  - Groups by source video, applies consistent crop position

Output format adds:
  - sequence_id: int (unique per sequence)
  - frame_idx: int (0, 1, 2... within sequence)

Usage:
  python organize_temporal.py --input E:/prism-v2-merged --output E:/prism-v2-temporal --seq-len 3
  python organize_temporal.py --input E:/prism-v2-merged --output E:/prism-v2-temporal --test  # small test
"""

import argparse
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading


def load_safe(path):
    try:
        return torch.load(path, weights_only=True)
    except:
        return None


def crop_at_position(data, y, x, crop_size=128):
    """Apply a specific crop position (not random)."""
    _, rH, rW = data["color"].shape
    _, dH, dW = data["ground_truth"].shape
    scale = dH / rH
    cr = min(crop_size, rH, rW)
    cd = int(cr * scale)

    # Clamp position to valid range
    y = min(y, rH - cr) if rH > cr else 0
    x = min(x, rW - cr) if rW > cr else 0
    dy = int(y * dH / rH)
    dx = int(x * dW / rW)
    dh = int(cr * dH / rH)
    dw = int(cr * dW / rW)

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


def organize(input_dir: Path, output_dir: Path, seq_len: int = 3,
             crop_size: int = 128, crops_per_sequence: int = 3):
    """
    Reorganize dataset for temporal training.

    Strategy:
    - Load full-size samples from I: drive sources
    - Group consecutive frames
    - For each group, pick random crop positions
    - Apply SAME position to all frames in the sequence
    - Save with sequence_id and frame_idx
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("sample_*.pt"))
    print(f"Input: {len(files)} samples from {input_dir}")

    # We need to work with the FULL SIZE samples for temporal
    # because the merged crops have random positions per frame
    # Check if these are pre-cropped (small) or full-size
    sample = load_safe(files[0])
    if sample is None:
        print("Can't load first sample!")
        return

    _, rH, rW = sample["color"].shape
    is_precropped = rH <= 128

    if is_precropped:
        print(f"Samples are pre-cropped ({rH}x{rW}) — creating pseudo-temporal sequences")
        print("Note: crop positions won't match across frames, but ConvGRU will still")
        print("learn general temporal behavior. Fine-tune with real data later.")
        _organize_precropped(files, output_dir, seq_len)
    else:
        print(f"Samples are full-size ({rH}x{rW}) — creating proper temporal sequences")
        _organize_fullsize(files, output_dir, seq_len, crop_size, crops_per_sequence)


def _organize_precropped(files, output_dir, seq_len):
    """
    For pre-cropped data: group consecutive files into sequences.
    Not perfect (different crop positions) but ConvGRU still learns.
    """
    idx = 0
    seq_id = 0

    # Group TartanAir: every 5 files = same source frame
    # So consecutive source frames are at i*5, (i+1)*5, (i+2)*5
    # For real video: consecutive files are consecutive frames

    # Detect TartanAir vs real video boundary
    ta_count = 0
    for f in files:
        s = load_safe(f)
        if s and not s.get("is_real", torch.tensor(True)).item():
            ta_count += 1
        else:
            break

    print(f"TartanAir samples: {ta_count}")
    print(f"Real video samples: {len(files) - ta_count}")

    # TartanAir: 5 crops per frame, take 1 crop from each of seq_len consecutive frames
    if ta_count > 0:
        frames_count = ta_count // 5
        print(f"TartanAir frames: {frames_count}")

        for frame_start in tqdm(range(0, frames_count - seq_len, seq_len), desc="TartanAir sequences"):
            # For each crop position (use crop 0 from each frame)
            for crop_offset in range(min(3, 5)):  # 3 sequences per frame group
                valid = True
                seq_samples = []
                for t in range(seq_len):
                    file_idx = (frame_start + t) * 5 + crop_offset
                    if file_idx >= ta_count:
                        valid = False
                        break
                    s = load_safe(files[file_idx])
                    if s is None:
                        valid = False
                        break
                    s["sequence_id"] = torch.tensor(seq_id, dtype=torch.int32)
                    s["frame_idx"] = torch.tensor(t, dtype=torch.int32)
                    seq_samples.append(s)

                if valid and len(seq_samples) == seq_len:
                    for s in seq_samples:
                        torch.save(s, output_dir / f"sample_{idx:06d}.pt")
                        idx += 1
                    seq_id += 1

    # Real video: consecutive files are consecutive frames
    real_start = ta_count
    real_files = files[real_start:]
    print(f"Processing {len(real_files)} real video samples...")

    for i in tqdm(range(0, len(real_files) - seq_len, seq_len), desc="Real sequences"):
        valid = True
        seq_samples = []
        for t in range(seq_len):
            s = load_safe(real_files[i + t])
            if s is None:
                valid = False
                break
            s["sequence_id"] = torch.tensor(seq_id, dtype=torch.int32)
            s["frame_idx"] = torch.tensor(t, dtype=torch.int32)
            seq_samples.append(s)

        if valid and len(seq_samples) == seq_len:
            for s in seq_samples:
                torch.save(s, output_dir / f"sample_{idx:06d}.pt")
                idx += 1
            seq_id += 1

    print(f"\nDone: {idx} samples in {seq_id} sequences of {seq_len} frames")
    print(f"Output: {output_dir}")


def _organize_fullsize(files, output_dir, seq_len, crop_size, crops_per_seq):
    """
    For full-size data: proper temporal with consistent crop positions.
    """
    idx = 0
    seq_id = 0

    for i in tqdm(range(0, len(files) - seq_len, seq_len), desc="Full-size sequences"):
        # Load seq_len consecutive full-size frames
        frames = []
        valid = True
        for t in range(seq_len):
            s = load_safe(files[i + t])
            if s is None:
                valid = False
                break
            frames.append(s)

        if not valid:
            continue

        _, rH, rW = frames[0]["color"].shape
        cr = min(crop_size, rH, rW)

        # Generate multiple sequences with different crop positions
        for _ in range(crops_per_seq):
            y = random.randint(0, max(0, rH - cr))
            x = random.randint(0, max(0, rW - cr))

            for t, frame in enumerate(frames):
                cropped = crop_at_position(frame, y, x, crop_size)
                cropped["sequence_id"] = torch.tensor(seq_id, dtype=torch.int32)
                cropped["frame_idx"] = torch.tensor(t, dtype=torch.int32)
                torch.save(cropped, output_dir / f"sample_{idx:06d}.pt")
                idx += 1

            seq_id += 1

    print(f"\nDone: {idx} samples in {seq_id} sequences of {seq_len} frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("E:/prism-v2-merged"))
    parser.add_argument("--output", type=Path, default=Path("E:/prism-v2-temporal"))
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--test", action="store_true", help="Test with first 100 samples only")
    args = parser.parse_args()

    if args.test:
        print("=== TEST MODE (first 100 samples) ===")
        test_dir = args.output.parent / "temporal-test"
        test_dir.mkdir(parents=True, exist_ok=True)

        # Copy first 100 samples to test dir
        src_files = sorted(args.input.glob("sample_*.pt"))[:100]
        if not src_files:
            print(f"No samples in {args.input}")
            return

        import shutil
        test_input = test_dir / "input"
        test_input.mkdir(exist_ok=True)
        for f in src_files:
            shutil.copy2(f, test_input / f.name)

        test_output = test_dir / "output"
        organize(test_input, test_output, seq_len=args.seq_len, crop_size=args.crop)

        # Verify
        out_files = sorted(test_output.glob("sample_*.pt"))
        if out_files:
            s = torch.load(out_files[0], weights_only=True)
            print(f"\nVerification:")
            print(f"  sequence_id: {s.get('sequence_id', 'MISSING')}")
            print(f"  frame_idx: {s.get('frame_idx', 'MISSING')}")
            print(f"  is_real: {s.get('is_real', 'MISSING')}")
            print(f"  color: {list(s['color'].shape)}")
            print(f"  Total output samples: {len(out_files)}")
    else:
        organize(args.input, args.output, seq_len=args.seq_len, crop_size=args.crop)


if __name__ == "__main__":
    main()
