"""
Export trained Prism model to ONNX for deployment in OptiScaler.

Creates a model folder with:
  - model.onnx        (the neural network)
  - config.json        (metadata for the inference engine)

Usage:
  python export_onnx.py --checkpoint path/to/checkpoint.pth --output models/prism-v2-balanced
  python export_onnx.py --checkpoint path/to/generator.pth --model balanced --name "Prism v2"
"""

import argparse
import json
import torch
from pathlib import Path
from model import PrismDecoder, PRESETS


def export(
    checkpoint_path: Path,
    output_dir: Path,
    model_name: str = "balanced",
    display_name: str = "Prism Model",
    version: str = "1.0",
    render_h: int = 540,
    render_w: int = 960,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    cfg = PRESETS[model_name]
    model = PrismDecoder(cfg)

    # Load weights (handle both full checkpoint and generator-only)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "generator" in state:
        model.load_state_dict(state["generator"])
        epoch = state.get("epoch", "?")
    else:
        model.load_state_dict(state)
        epoch = "?"

    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_name} ({params/1e6:.2f}M params) from {checkpoint_path}")

    # Create dummy inputs
    color = torch.randn(1, 3, render_h, render_w)
    depth = torch.randn(1, 1, render_h, render_w)
    mv = torch.randn(1, 2, render_h, render_w)

    # Export ONNX
    onnx_path = output_dir / "model.onnx"
    print(f"Exporting to {onnx_path}...")

    torch.onnx.export(
        model,
        (color, depth, mv),
        str(onnx_path),
        opset_version=17,
        input_names=["color", "depth", "motion_vectors"],
        output_names=["output", "hidden"],
        dynamic_axes={
            "color": {0: "batch", 2: "height", 3: "width"},
            "depth": {0: "batch", 2: "height", 3: "width"},
            "motion_vectors": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
            "hidden": {0: "batch", 2: "height", 3: "width"},
        },
    )

    # Verify
    onnx_size = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Exported: {onnx_size:.1f} MB")

    # Save config
    config = {
        "name": display_name,
        "version": version,
        "description": f"{model_name} model, {params/1e6:.2f}M params, epoch {epoch}",
        "model_preset": model_name,
        "parameters": params,
        "input_channels": cfg.input_channels,
        "has_temporal": cfg.temporal != "none",
        "hidden_channels": cfg.ch,
        "supported_scales": cfg.upsample_paths,
        "default_scale": 2,
        "trained_epoch": epoch if isinstance(epoch, int) else 0,
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Config saved: {config_path}")
    print(f"\nModel folder ready: {output_dir}")
    print(f"  model.onnx:  {onnx_size:.1f} MB")
    print(f"  config.json: {config['name']} v{config['version']}")
    print(f"\nDrop this folder into OptiScaler's models/ directory to use it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Prism model to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=list(PRESETS.keys()), default="balanced")
    parser.add_argument("--name", default="Prism Model")
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--render-h", type=int, default=540)
    parser.add_argument("--render-w", type=int, default=960)
    args = parser.parse_args()

    export(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        model_name=args.model,
        display_name=args.name,
        version=args.version,
        render_h=args.render_h,
        render_w=args.render_w,
    )
