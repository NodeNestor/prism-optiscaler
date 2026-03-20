"""
GPU-side game-style augmentation — applied on-the-fly during training.
No disk I/O, no CPU bottleneck. Just tensor ops on GPU.

Applied to the COLOR input only (not ground truth).
Random strength per batch = infinite variety.
"""

import torch
import torch.nn.functional as F


def quantize_gpu(x: torch.Tensor, levels: int = 32) -> torch.Tensor:
    """Reduce color precision — simulates low texture bit depth."""
    return (x * levels).round() / levels


def flatten_lighting_gpu(x: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
    """Reduce lighting contrast — games have flatter ambient."""
    gray = x.mean(dim=1, keepdim=True)
    mean_lum = gray.mean(dim=(2, 3), keepdim=True)
    return x + strength * (mean_lum - gray)


def sharpen_gpu(x: torch.Tensor, amount: float = 0.5) -> torch.Tensor:
    """Over-sharpen — game renders have unnaturally sharp edges."""
    blur = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), 3, stride=1)
    return x + amount * (x - blur)


def add_banding_gpu(x: torch.Tensor, bands: int = 48) -> torch.Tensor:
    """Color banding in gradients."""
    return (x * bands).floor() / bands


def gamma_shift_gpu(x: torch.Tensor, gamma: float = 1.1) -> torch.Tensor:
    """Slight gamma shift — different tone mapping."""
    return x.clamp(1e-6, 1.0).pow(gamma)


def smooth_skin_gpu(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Bilateral-like smoothing — makes textures look CG/plastic."""
    # Approximate with gaussian blur
    k = int(sigma * 4) | 1  # odd kernel
    if k < 3:
        k = 3
    pad = k // 2
    # Depthwise gaussian blur
    channels = x.shape[1]
    kernel_1d = torch.exp(-torch.arange(k, device=x.device, dtype=x.dtype).sub(pad).pow(2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_h = kernel_1d.view(1, 1, k, 1).expand(channels, 1, k, 1)
    kernel_w = kernel_1d.view(1, 1, 1, k).expand(channels, 1, 1, k)
    blurred = F.conv2d(F.pad(x, (0, 0, pad, pad), mode='reflect'), kernel_h, groups=channels)
    blurred = F.conv2d(F.pad(blurred, (pad, pad, 0, 0), mode='reflect'), kernel_w, groups=channels)
    return blurred


def game_augment_gpu(
    color: torch.Tensor,
    strength: float = 0.5,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Apply random game-style degradation to color tensor on GPU.

    Args:
        color: [B, 3, H, W] float tensor in [0, 1]
        strength: 0.0 = no augment, 1.0 = maximum game-ification
        rng: torch random generator for reproducibility

    Returns:
        Augmented color tensor, same shape
    """
    if strength <= 0:
        return color

    def rand():
        return torch.rand(1, generator=rng).item()

    x = color

    # Each augmentation applied with probability proportional to strength
    if rand() < strength:
        x = flatten_lighting_gpu(x, strength=rand() * 0.4 * strength)

    if rand() < strength:
        x = sharpen_gpu(x, amount=rand() * 0.6 * strength)

    if rand() < strength * 0.7:
        levels = int(16 + (1 - strength) * 48)
        x = quantize_gpu(x, levels=levels)

    if rand() < strength * 0.5:
        bands = int(32 + (1 - strength) * 64)
        x = add_banding_gpu(x, bands=bands)

    if rand() < strength * 0.4:
        x = gamma_shift_gpu(x, gamma=1.0 + rand() * 0.2 * strength)

    if rand() < strength * 0.3:
        x = smooth_skin_gpu(x, sigma=0.5 + rand() * strength)

    return x.clamp(0, 1)


if __name__ == "__main__":
    # Quick test
    x = torch.randn(1, 3, 128, 128).clamp(0, 1)
    for s in [0.0, 0.3, 0.5, 0.8, 1.0]:
        out = game_augment_gpu(x, strength=s)
        print(f"strength={s}: min={out.min():.3f} max={out.max():.3f} mean={out.mean():.3f}")
    print("GPU augmentation OK!")
