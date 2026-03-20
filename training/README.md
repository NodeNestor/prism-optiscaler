# Prism Training Pipeline

Train a tiny neural decoder: G-buffer → photorealistic image.

## Quick Start

```bash
cd training
pip install -r requirements.txt

# 1. Download training videos (Blender open movies — CC licensed, cinematic)
python video_sources.py --output data/videos

# 2. Generate G-buffer dataset from videos
python generate_dataset.py --videos data/videos --output data/dataset

# 3. Train
python train.py --data data/dataset --model balanced --epochs 100 --batch 4

# 4. Export to ONNX (for TensorRT integration in OptiScaler)
python train.py --data data/dataset --export-onnx
```

## Architecture

The game engine is our encoder — it compresses the 3D scene into a G-buffer:
```
color(3) + depth(1) + motion_vectors(2) = 6 channels at render res
```

We only train the decoder:
```
G-buffer (540p) → [PrismDecoder ~2M params] → Photorealistic RGB (1080p)
```

Inspired by TAESD (1.2M params decodes Stable Diffusion latents instantly).

## Model Variants

| Variant | Params | Target TRT FP16 | Notes |
|---------|--------|-----------------|-------|
| fast | ~1.5M | 2ms | DSC + 48ch + 3 blocks |
| balanced | ~3M | 4ms | DSC + 64ch + 4 blocks |
| quality | ~5M | 6ms | Standard conv + 64ch + 6 blocks |

## Video Sources

Training data comes from real-life cinematic video. The pipeline extracts:
- **Depth** via Depth Anything V2
- **Motion vectors** via RAFT optical flow
- **Jittered low-res color** via Halton sequence + downsample

### Included sources (CC licensed)
- Sintel — fantasy (dragons, snow, caves)
- Tears of Steel — sci-fi (urban, robots)
- Spring — nature/fantasy (forests, meadows)
- Cosmos Laundromat — surreal landscapes

### Add your own
Drop any .mp4/.mkv/.mov files in `data/videos/` and re-run dataset generation.
