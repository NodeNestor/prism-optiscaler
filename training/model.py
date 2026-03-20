"""
Prism model — modular G-buffer decoder for photorealistic neural rendering.

Everything is configurable via ModelConfig. Easy to experiment with:
  - Channel width, block count, block type (DSC vs standard)
  - Temporal mode (none, EMA, GRU)
  - Upsample paths (2x, 3x, or both)
  - Refine depth at display res
  - Input channels (with/without warped prev)

Architecture:
  Game engine = encoder (compresses 3D scene into G-buffer)
  This model = decoder (G-buffer -> photorealistic image)

Inspired by TAESD, NSRD, FRVSR, SPADE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


# ============================================================================
# Config — one place to change everything
# ============================================================================

@dataclass
class ModelConfig:
    # Architecture
    ch: int = 64                        # Base channel width (48=fast, 64=balanced, 96=quality)
    n_render_blocks: int = 4            # ResBlocks at render res (where the magic happens)
    n_refine_blocks: int = 1            # ResBlocks at display res (cleanup after upsample)
    n_refine_blocks_3x: int = 2         # Extra refine for 3x path (needs more at high scale)
    use_dsc: bool = True                # Depthwise separable convs (8x fewer FLOPs, same quality)

    # Temporal
    temporal: str = "gru"               # "none", "ema", "gru"
    ema_alpha: float = 0.85             # EMA blend factor (only if temporal="ema")

    # Upsampling
    upsample_paths: list = field(default_factory=lambda: [2, 3])  # Which PixelShuffle scales to build
    max_scale: float = 3.0              # Max supported upscale factor

    # Input
    use_warped_prev: bool = True        # Concat warped previous output as input (adds 3ch)
    input_channels: int = 0             # Auto-calculated: color(3)+depth(1)+mv(2)+warped_prev(3)=9

    def __post_init__(self):
        self.input_channels = 6 + (3 if self.use_warped_prev else 0)


# ============================================================================
# Building blocks
# ============================================================================

def conv(n_in: int, n_out: int, k: int = 3, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, k, padding=k // 2, **kwargs)


class Block(nn.Module):
    """Standard residual block: Conv->ReLU->Conv->ReLU->Conv + skip."""
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in, n_out), nn.ReLU(inplace=True),
            conv(n_out, n_out), nn.ReLU(inplace=True),
            conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x) + self.skip(x), inplace=True)


class DSCBlock(nn.Module):
    """Depthwise separable residual block — ~8x fewer FLOPs."""
    def __init__(self, ch: int):
        super().__init__()
        self.dw1 = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw1 = nn.Conv2d(ch, ch, 1)
        self.dw2 = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw2 = nn.Conv2d(ch, ch, 1)
        self.dw3 = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw3 = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.pw1(self.dw1(x)), inplace=True)
        h = F.relu(self.pw2(self.dw2(h)), inplace=True)
        h = self.pw3(self.dw3(h))
        return F.relu(h + x, inplace=True)


def make_block(ch: int, use_dsc: bool) -> nn.Module:
    return DSCBlock(ch) if use_dsc else Block(ch, ch)


# ============================================================================
# Motion vector warping
# ============================================================================

def warp(x: torch.Tensor, mv: torch.Tensor) -> torch.Tensor:
    """Warp tensor x using motion vectors mv (pixel space at x's resolution)."""
    B, _, H, W = x.shape
    if mv.shape[2:] != (H, W):
        scale_x = W / mv.shape[3]
        scale_y = H / mv.shape[2]
        mv = F.interpolate(mv, (H, W), mode="bilinear", align_corners=False)
        mv = mv * torch.tensor([scale_x, scale_y], device=mv.device).view(1, 2, 1, 1)

    gy, gx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(-1, 1, W, device=x.device),
        indexing="ij",
    )
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    mv_norm = mv.permute(0, 2, 3, 1).clone()
    mv_norm[..., 0] /= W / 2
    mv_norm[..., 1] /= H / 2

    return F.grid_sample(x, grid - mv_norm, mode="bilinear",
                         padding_mode="border", align_corners=True)


# ============================================================================
# Temporal modules
# ============================================================================

class TemporalNone(nn.Module):
    """No temporal — just pass through."""
    def forward(self, current, prev_hidden, mv):
        return current


class TemporalEMA(nn.Module):
    """Exponential moving average with motion-compensated warping. Near-zero cost."""
    def __init__(self, alpha: float = 0.85):
        super().__init__()
        self.alpha = alpha

    def forward(self, current: torch.Tensor, prev_hidden: torch.Tensor | None,
                mv: torch.Tensor | None) -> torch.Tensor:
        if prev_hidden is None:
            return current
        warped = warp(prev_hidden, mv) if mv is not None else prev_hidden
        return self.alpha * warped + (1 - self.alpha) * current


class TemporalGRU(nn.Module):
    """Convolutional GRU — 2 gates, motion-compensated warping."""
    def __init__(self, ch: int):
        super().__init__()
        self.gate_conv = conv(ch * 2, ch * 2)
        self.cand_conv = conv(ch * 2, ch)

    def forward(self, current: torch.Tensor, prev_hidden: torch.Tensor | None,
                mv: torch.Tensor | None) -> torch.Tensor:
        if prev_hidden is None:
            return current
        if mv is not None:
            prev_hidden = warp(prev_hidden, mv)

        combined = torch.cat([current, prev_hidden], dim=1)
        gates = torch.sigmoid(self.gate_conv(combined))
        reset, update = gates.chunk(2, dim=1)
        candidate = torch.tanh(self.cand_conv(torch.cat([current, reset * prev_hidden], dim=1)))
        return update * prev_hidden + (1 - update) * candidate


def make_temporal(mode: str, ch: int, alpha: float = 0.85) -> nn.Module:
    if mode == "none":
        return TemporalNone()
    elif mode == "ema":
        return TemporalEMA(alpha)
    elif mode == "gru":
        return TemporalGRU(ch)
    raise ValueError(f"Unknown temporal mode: {mode}")


# ============================================================================
# Adaptive upsampler — picks the right PixelShuffle path at runtime
# ============================================================================

class AdaptiveUpsample(nn.Module):
    """
    Multiple PixelShuffle paths, picks the best one at runtime.

    Built paths match DLSS presets:
      scale <= 1.0 -> no upsample (DLAA mode)
      scale <= 2.0 -> PixelShuffle(2) + resize to exact target
      scale <= 3.0 -> PixelShuffle(3) + resize to exact target

    Each path has its own learned conv (different channel expansion)
    but only ONE runs per frame. Zero wasted compute.
    """
    def __init__(self, ch: int, paths: list[int], use_dsc: bool = True):
        super().__init__()
        self.paths = sorted(paths)  # e.g. [2, 3]

        # One learned conv + shuffle per path
        self.up_convs = nn.ModuleDict()
        self.up_shuffles = nn.ModuleDict()
        for s in self.paths:
            self.up_convs[str(s)] = conv(ch, ch * s * s)
            self.up_shuffles[str(s)] = nn.PixelShuffle(s)

    def forward(self, x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        _, _, h, w = x.shape
        scale = max(target_h / h, target_w / w)

        if scale <= 1.0:
            # DLAA mode — no upsample, just resize to target
            if x.shape[2] != target_h or x.shape[3] != target_w:
                return F.interpolate(x, (target_h, target_w), mode="bilinear", align_corners=False)
            return x

        # Pick smallest PixelShuffle path that covers the target scale
        chosen = self.paths[-1]  # default to largest
        for s in self.paths:
            if s >= scale:
                chosen = s
                break

        # Apply learned conv + PixelShuffle
        up = self.up_convs[str(chosen)](x)
        up = F.relu(self.up_shuffles[str(chosen)](up), inplace=True)

        # Resize to exact target (only if PixelShuffle overshot)
        if up.shape[2] != target_h or up.shape[3] != target_w:
            up = F.interpolate(up, (target_h, target_w), mode="bilinear", align_corners=False)

        return up


# ============================================================================
# Generator — the full decoder
# ============================================================================

class PrismDecoder(nn.Module):
    """
    G-buffer decoder: (color + depth + mv) -> photorealistic RGB.

    ALL processing at render res. PixelShuffle directly to RGB at the end.
    No convolutions at display resolution = 3-5x faster than old architecture.

    Pipeline:
      1. Input conv (9ch -> ch) at render res
      2. N DSC ResBlocks at render res (all the "thinking")
      3. Temporal ConvGRU at render res
      4. Conv to subpixel channels (ch -> 3*scale^2) at render res
      5. PixelShuffle to 3ch RGB at display res
      6. Sigmoid — done. No display-res convolutions!

    Speed (PyTorch FP16, RTX 5060 Ti, 540p->1080p):
      ch=32, 4 blocks: 10ms (100 FPS)
      ch=48, 4 blocks: 18ms (56 FPS)
      ch=64, 4 blocks: 24ms (42 FPS)
    """

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        ch = cfg.ch

        # Input projection
        self.input_conv = nn.Sequential(conv(cfg.input_channels, ch), nn.ReLU(inplace=True))

        # ALL heavy processing at render res
        self.render_blocks = nn.Sequential(
            *[make_block(ch, cfg.use_dsc) for _ in range(cfg.n_render_blocks)]
        )

        # Temporal recurrence at render res
        self.temporal = make_temporal(cfg.temporal, ch, cfg.ema_alpha)

        # Direct PixelShuffle to RGB — NO display-res convolutions
        # conv(ch -> 3*scale^2) at render res, then shuffle to 3ch at display res
        self.to_subpixel_2x = conv(ch, 3 * 4)   # 3 * 2^2 = 12ch -> shuffle -> 3ch at 2x
        self.to_subpixel_3x = conv(ch, 3 * 9)   # 3 * 3^2 = 27ch -> shuffle -> 3ch at 3x
        self.shuffle_2x = nn.PixelShuffle(2)
        self.shuffle_3x = nn.PixelShuffle(3)

    def forward(
        self,
        color: torch.Tensor,
        depth: torch.Tensor,
        motion_vectors: torch.Tensor,
        prev_output: torch.Tensor | None = None,
        prev_hidden: torch.Tensor | None = None,
        target_h: int = 0,
        target_w: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, rH, rW = color.shape

        if target_h <= 0:
            target_h = rH * 2
        if target_w <= 0:
            target_w = rW * 2

        scale = max(target_h / rH, target_w / rW)

        # Build input
        inputs = [color, depth.to(color.dtype), motion_vectors]
        if self.cfg.use_warped_prev:
            if prev_output is not None:
                prev_down = F.interpolate(prev_output, (rH, rW), mode="bilinear", align_corners=False)
                inputs.append(warp(prev_down, motion_vectors))
            else:
                inputs.append(torch.zeros(B, 3, rH, rW, device=color.device, dtype=color.dtype))

        x = torch.cat(inputs, dim=1)

        # Render-res processing (all the heavy lifting happens here)
        x = self.input_conv(x)
        x = self.render_blocks(x)

        # Temporal
        hidden = self.temporal(x, prev_hidden, motion_vectors)

        # Direct PixelShuffle to RGB — ZERO display-res convolutions
        if scale <= 2.0:
            output = torch.sigmoid(self.shuffle_2x(self.to_subpixel_2x(hidden)))
        else:
            output = torch.sigmoid(self.shuffle_3x(self.to_subpixel_3x(hidden)))

        # Resize to exact target if PixelShuffle overshot
        if output.shape[2] != target_h or output.shape[3] != target_w:
            output = F.interpolate(output, (target_h, target_w), mode="bilinear", align_corners=False)

        return output, hidden.detach()


# ============================================================================
# Discriminator — only used during training
# ============================================================================

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        ch = ndf
        for i in range(1, n_layers):
            ch_next = min(ndf * 2 ** i, ndf * 8)
            layers += [nn.Conv2d(ch, ch_next, 4, 2, 1), nn.LeakyReLU(0.2, True)]
            ch = ch_next
        layers.append(nn.Conv2d(ch, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.disc1 = PatchDiscriminator(in_ch)
        self.disc2 = PatchDiscriminator(in_ch)
        self.down = nn.AvgPool2d(3, 2, 1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return [self.disc1(x), self.disc2(self.down(x))]


# ============================================================================
# Losses
# ============================================================================

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16]])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        x, y = pred, target
        for block in self.blocks:
            x, y = block(x), block(y)
            loss = loss + F.l1_loss(x, y)
        return loss


class HingeLoss:
    @staticmethod
    def d_loss(real_preds: list, fake_preds: list) -> torch.Tensor:
        return sum(F.relu(1 - r).mean() + F.relu(1 + f).mean()
                   for r, f in zip(real_preds, fake_preds)) / len(real_preds)

    @staticmethod
    def g_loss(fake_preds: list) -> torch.Tensor:
        return sum(-f.mean() for f in fake_preds) / len(fake_preds)


# ============================================================================
# Preset configs
# ============================================================================

PRESETS = {
    # New fast architecture: ALL processing at render res, direct PixelShuffle to RGB
    # Speed measured in PyTorch FP16 on RTX 5060 Ti at 540p->1080p
    "ultra": ModelConfig(        # ~5ms, 200 FPS — minimal enhancement
        ch=32, n_render_blocks=2, use_dsc=True, temporal="ema",
    ),
    "fast": ModelConfig(         # ~10ms, 100 FPS — good quality
        ch=32, n_render_blocks=4, use_dsc=True, temporal="ema",
    ),
    "balanced": ModelConfig(     # ~18ms, 56 FPS — best quality at 60fps
        ch=48, n_render_blocks=4, use_dsc=True, temporal="gru",
    ),
    "quality": ModelConfig(      # ~24ms, 42 FPS — best quality
        ch=64, n_render_blocks=4, use_dsc=True, temporal="gru",
    ),
    "extreme": ModelConfig(      # ~35ms, 29 FPS — maximum capacity
        ch=64, n_render_blocks=6, use_dsc=False, temporal="gru",
    ),
}

def prism_fast() -> PrismDecoder:
    return PrismDecoder(PRESETS["fast"])

def prism_balanced() -> PrismDecoder:
    return PrismDecoder(PRESETS["balanced"])

def prism_quality() -> PrismDecoder:
    return PrismDecoder(PRESETS["quality"])

def prism_extreme() -> PrismDecoder:
    return PrismDecoder(PRESETS["extreme"])

def prism_custom(**kwargs) -> PrismDecoder:
    """Build with any config overrides: prism_custom(ch=80, n_render_blocks=5)"""
    return PrismDecoder(ModelConfig(**kwargs))


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # Use TINY resolutions for testing — same scale factors, no RAM spike
    # Real res would be 540->1080 etc, we use 30->60 etc
    dlss_presets = [
        (60,  80,  60,  80,  "DLAA 1.0x"),
        (40,  72,  60,  108, "Quality 1.5x"),
        (36,  64,  60,  108, "Balanced 1.7x"),
        (30,  54,  60,  108, "Performance 2.0x"),
        (20,  36,  60,  108, "Ultra Perf 3.0x"),
    ]

    for name, cfg in PRESETS.items():
        model = PrismDecoder(cfg)
        params = sum(p.numel() for p in model.parameters())
        print(f"\n=== {name} ({params/1e6:.2f}M params) ===")
        print(f"    cfg: ch={cfg.ch} render_blocks={cfg.n_render_blocks} "
              f"refine={cfg.n_refine_blocks}/{cfg.n_refine_blocks_3x} "
              f"temporal={cfg.temporal} dsc={cfg.use_dsc}")

        for rH, rW, tH, tW, label in dlss_presets:
            out, hidden = model(
                torch.randn(1, 3, rH, rW),
                torch.randn(1, 1, rH, rW),
                torch.randn(1, 2, rH, rW),
                target_h=tH, target_w=tW,
            )
            # Temporal test
            out2, _ = model(
                torch.randn(1, 3, rH, rW),
                torch.randn(1, 1, rH, rW),
                torch.randn(1, 2, rH, rW),
                prev_output=out.detach(),
                prev_hidden=hidden,
                target_h=tH, target_w=tW,
            )
            scale = f"{tH/rH:.1f}x"
            print(f"    {rH}x{rW} -> {out.shape[2]}x{out.shape[3]} ({scale}) {label} | temporal OK")
