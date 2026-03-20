"""
Save a training run — archive checkpoints, config, losses, and samples.

Usage:
  python save_run.py --name v1-gan-2x --checkpoint E:/prism-dataset-fast/checkpoints/checkpoint_ep50.pth
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Run name (e.g. v1-gan-2x)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=Path("E:/prism-training.log"))
    parser.add_argument("--output", type=Path, default=Path("E:/prism-runs"))
    args = parser.parse_args()

    run_dir = args.output / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy checkpoint
    shutil.copy2(args.checkpoint, run_dir / "checkpoint.pth")

    # Copy generator weights only
    gen_path = args.checkpoint.parent / "prism_generator_latest.pth"
    if gen_path.exists():
        shutil.copy2(gen_path, run_dir / "generator.pth")

    # Extract and save losses
    if args.log.exists():
        losses = []
        with open(args.log) as f:
            for line in f:
                if "L1=" in line:
                    losses.append(line.strip())
        with open(run_dir / "losses.txt", "w") as f:
            for i, l in enumerate(losses, 1):
                f.write(f"Epoch {i}: {l}\n")

    # Save run info
    info = {
        "name": args.name,
        "date": datetime.now().isoformat(),
        "epochs": len([l for l in open(args.log) if "L1=" in l]) if args.log.exists() else 0,
        "model": "balanced (0.81M params)",
        "optimizer": "adamw",
        "batch_size": 16,
        "crop_size": 128,
        "seq_len": 1,
        "precision": "bf16",
        "gpu": "RTX 5060 Ti",
        "dataset": "79K pexels-nature samples (2x only)",
        "notes": "First training run. GAN with hinge loss. No temporal. "
                 "Color grading toward photorealism visible. 3x path untrained.",
    }
    with open(run_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Copy all checkpoints
    ckpt_dir = args.checkpoint.parent
    for ckpt in ckpt_dir.glob("checkpoint_ep*.pth"):
        shutil.copy2(ckpt, run_dir / ckpt.name)

    print(f"Run saved to {run_dir}")
    print(f"  Checkpoints: {len(list(run_dir.glob('checkpoint_ep*.pth')))}")
    print(f"  Generator: {run_dir / 'generator.pth'}")
    print(f"  Losses: {run_dir / 'losses.txt'}")
    print(f"  Info: {run_dir / 'info.json'}")


if __name__ == "__main__":
    main()
