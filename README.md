# Prism OptiScaler

Neural upscaling + zero-latency frame extrapolation for games. Built on [OptiScaler](https://github.com/cdozdil/OptiScaler).

> For upstream OptiScaler documentation (installation, supported games, config reference, wiki), see the [original repo](https://github.com/cdozdil/OptiScaler).

## What this adds

### Prism Neural Upscaler
Custom neural network upscaler that replaces DLSS/FSR/XeSS with a GAN-trained model. Takes G-buffer inputs (color + depth + motion vectors) and produces photorealistic output at target resolution.

- **Training** — PyTorch GAN pipeline with temporal stability (ConvGRU), trained on Blender open movies
- **Inference** — Vulkan compute with cooperative vectors, ONNX export for TensorRT/CUDA
- **Integration** — registered as `"prism"` upscaler backend in OptiScaler (DX11/DX12/Vulkan)

### FGExtrap — Frame Extrapolation
Zero-latency frame generation via depth-layered reprojection. No AI, pure shader math.

- **5 depth layers** (HUD / Sky / Far / Mid / Near) reprojected independently — like anime cel layers, background fills in behind foreground naturally
- **Mouse input as ground truth** — reads raw mouse deltas via RawInput, exact camera position, no prediction
- **Adaptive cadence** — measures real FPS, inserts N synthetic frames to hit target (60 to 120, 60 to 180, etc.)
- **Two gap fill modes** — extend (fast, default) and Fourier (quality, for fast motion)
- **Back-to-front compositing** — disocclusion handled by layer overlap, gap fill only needed at screen edges

```
Game Render -> OptiScaler upscaler pass -> [REAL FRAME] -> Present
                                                |
                                    Store color + depth + MVs
                                                |
                                    Classify into 5 depth layers
                                                |
                                    Per-layer reprojection (mouse + MVs)
                                                |
                                    Gap fill -> Composite
                                                |
                                    [SYNTHETIC FRAME] -> Present
```

**Why extrapolation over interpolation:**
- Zero added latency (no waiting for next real frame)
- Camera position is exact from mouse input (not predicted)
- Only world object motion is extrapolated, and for tiny time deltas

## Repository Structure

```
prism-optiscaler/
├── OptiScaler/              # OptiScaler core (modified fork)
│   ├── framegen/extrap/     # FGExtrap frame extrapolation backend
│   ├── upscalers/prism/     # Prism neural upscaler integration
│   └── ...                  # All standard OptiScaler code
├── prism-inference/         # [submodule] Vulkan inference engine
├── prism-training/          # [submodule] PyTorch training pipeline
├── external/                # Vendor SDKs (DLSS, FSR, XeSS, etc.)
├── OptiScaler.sln           # Visual Studio solution
└── OptiScaler.ini           # Runtime configuration
```

## Build

```bash
git clone --recursive https://github.com/NodeNestor/prism-optiscaler.git
```

Open `OptiScaler.sln` in Visual Studio, build Release x64. Output: `x64/Release/OptiScaler.dll`

## Configuration

### FGExtrap

In `OptiScaler.ini`:
```ini
[FrameGen]
FGInput=upscaler
FGOutput=fgextrap

[FGExtrap]
TargetFPS=120              ; target framerate (adaptive cadence)
MouseSensitivity=1.0       ; mouse delta to camera angle factor
AutoCalibrate=true         ; auto-derive sensitivity from motion vectors
DepthScale=1.0             ; parallax strength
FOV=90                     ; horizontal degrees (auto-detected if game provides it)
GapFillMode=0              ; 0=extend (fast), 1=fourier (quality)
DebugLayers=false          ; true = color-tinted layer visualization

; Depth layer thresholds (normalized 0-1)
HUDThreshold=0.001
NearThreshold=0.05
FarThreshold=0.80
SkyThreshold=0.99
```

### Prism Upscaler

```ini
[Upscalers]
Dx12Upscaler=prism
```

## Related Repositories

| Repo | Description |
|---|---|
| [NodeNestor/prism-inference](https://github.com/NodeNestor/prism-inference) | Vulkan compute inference engine with cooperative vectors |
| [NodeNestor/prism-training](https://github.com/NodeNestor/prism-training) | PyTorch training pipeline (GAN + temporal stability) |
| [cdozdil/OptiScaler](https://github.com/cdozdil/OptiScaler) | Upstream OptiScaler (base project) |

## Credits

Built on [OptiScaler](https://github.com/cdozdil/OptiScaler) by cdozdil and contributors. Uses AMD FidelityFX SDK, Intel XeSS SDK, NVIDIA DLSS SDK.
