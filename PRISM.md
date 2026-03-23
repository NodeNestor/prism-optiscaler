# Prism — Neural Upscaling + Frame Extrapolation for OptiScaler

Fork of [OptiScaler](https://github.com/cdozdil/OptiScaler) with custom upscaling and frame generation backends.

## Components

### Prism Neural Upscaler
Custom neural network upscaler that replaces DLSS/FSR/XeSS with a GAN-trained model. Takes G-buffer inputs (color + depth + motion vectors) and produces photorealistic output at target resolution.

- **Training**: PyTorch GAN pipeline with temporal stability (ConvGRU), trained on Blender open movies
- **Inference**: Vulkan compute with cooperative vectors, ONNX export for TensorRT/CUDA
- **Integration**: Registered as `"prism"` upscaler backend in OptiScaler (DX11/DX12/Vulkan)

### FGExtrap — Frame Extrapolation
Zero-latency frame generation via depth-layered reprojection. No AI, pure shader math.

- **5 depth layers** (HUD / Sky / Far / Mid / Near) reprojected independently
- **Mouse input as ground truth** — reads raw mouse deltas via RawInput, no prediction needed
- **Adaptive cadence** — measures real FPS, inserts N synthetic frames to hit target FPS
- **Two gap fill modes** — extend (fast) and Fourier (quality)
- **Back-to-front compositing** — background naturally fills foreground disocclusion gaps

```
Game Render → OptiScaler upscaler pass → [REAL FRAME] → Present
                                              ↓
                                     Store color + depth + MVs
                                              ↓
                                     Classify into 5 depth layers
                                              ↓
                                     Per-layer reprojection (mouse + MVs)
                                              ↓
                                     Gap fill → Composite
                                              ↓
                                     [SYNTHETIC FRAME] → Present
```

## Repository Structure

```
prism-optiscaler/
├── OptiScaler/              # OptiScaler core (modified fork)
│   └── framegen/extrap/     # FGExtrap frame extrapolation backend
├── prism-inference/         # [submodule] Vulkan inference engine
├── prism-training/          # [submodule] PyTorch training pipeline
├── external/                # Vendor SDKs (DLSS, FSR, XeSS, etc.)
├── OptiScaler.sln           # Visual Studio solution
└── OptiScaler.ini           # Runtime configuration
```

## Quick Start

```bash
git clone --recursive https://github.com/NodeNestor/prism-optiscaler.git
```

Open `OptiScaler.sln` in Visual Studio 2026, build Release x64.

### Enable FGExtrap

In `OptiScaler.ini`:
```ini
[FrameGen]
FGInput=upscaler
FGOutput=fgextrap

[FGExtrap]
TargetFPS=120
MouseSensitivity=1.0
AutoCalibrate=true
GapFillMode=0          ; 0=extend, 1=fourier
DebugLayers=false      ; true = color-tinted layer visualization
```

## Related Repositories

- [NodeNestor/prism-inference](https://github.com/NodeNestor/prism-inference) — Vulkan compute inference engine with cooperative vectors
- [NodeNestor/prism-training](https://github.com/NodeNestor/prism-training) — PyTorch training pipeline (GAN + temporal stability)
- [cdozdil/OptiScaler](https://github.com/cdozdil/OptiScaler) — Upstream OptiScaler
