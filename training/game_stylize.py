"""
Game-style degradation filters — make real video look more like game renders.

The idea: real video has film grain, lens blur, natural lighting.
Game renders have: flat lighting, sharp edges, banding, posterization,
lower material variety, uniform specularity.

By degrading real video to look game-like BEFORE feeding it as the "input",
we train the model to bridge the game→real domain gap.

Filters are applied to the LOW-RES COLOR input only.
The GROUND TRUTH stays as the original high-quality video frame.
"""

import cv2
import numpy as np


def quantize_colors(img: np.ndarray, levels: int = 32) -> np.ndarray:
    """Reduce color precision — simulates lower texture bit depth."""
    return (np.round(img * levels) / levels).clip(0, 1).astype(np.float32)


def flatten_lighting(img: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Reduce lighting contrast — game engines have flatter ambient."""
    gray = np.mean(img, axis=2, keepdims=True)
    mean_lum = gray.mean()
    flattened = img + strength * (mean_lum - gray)
    return flattened.clip(0, 1).astype(np.float32)


def sharpen_edges(img: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """Over-sharpen — game renders have unnaturally sharp edges."""
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    sharp = img + amount * (img - blur)
    return sharp.clip(0, 1).astype(np.float32)


def add_banding(img: np.ndarray, bands: int = 64) -> np.ndarray:
    """Add color banding — common in game gradients (sky, fog)."""
    return (np.floor(img * bands) / bands).clip(0, 1).astype(np.float32)


def reduce_specular(img: np.ndarray, threshold: float = 0.85, reduction: float = 0.5) -> np.ndarray:
    """Tone down specular highlights — games often have uniform specularity."""
    bright = np.max(img, axis=2, keepdims=True)
    mask = (bright > threshold).astype(np.float32)
    dampened = img * (1.0 - mask * reduction)
    return dampened.clip(0, 1).astype(np.float32)


def add_slight_blur(img: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Slight blur — simulates TAA ghosting / lower texture resolution."""
    return cv2.GaussianBlur(img, (0, 0), sigma).astype(np.float32)


def gamma_shift(img: np.ndarray, gamma: float = 1.1) -> np.ndarray:
    """Slight gamma shift — games often have different tone mapping."""
    return np.power(img.clip(1e-6, 1.0), gamma).astype(np.float32)


# ============================================================================
# Preset pipelines
# ============================================================================

def apply_game_style(img: np.ndarray, style: str = "medium", rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Apply game-style degradation to an image.

    Args:
        img: float32 [H, W, 3] in [0, 1]
        style: "none", "light", "medium", "heavy", "random"
        rng: numpy random generator for reproducibility

    Returns:
        Degraded image, same shape/dtype
    """
    if style == "none":
        return img

    if rng is None:
        rng = np.random.default_rng()

    if style == "random":
        style = rng.choice(["none", "light", "medium", "heavy"])

    out = img.copy()

    if style == "light":
        out = flatten_lighting(out, strength=rng.uniform(0.1, 0.2))
        out = sharpen_edges(out, amount=rng.uniform(0.2, 0.4))
        if rng.random() < 0.3:
            out = quantize_colors(out, levels=rng.integers(48, 64))

    elif style == "medium":
        out = flatten_lighting(out, strength=rng.uniform(0.2, 0.4))
        out = sharpen_edges(out, amount=rng.uniform(0.3, 0.6))
        out = quantize_colors(out, levels=rng.integers(32, 48))
        out = gamma_shift(out, gamma=rng.uniform(1.0, 1.2))
        if rng.random() < 0.5:
            out = add_banding(out, bands=rng.integers(48, 96))
        if rng.random() < 0.3:
            out = reduce_specular(out, threshold=rng.uniform(0.8, 0.9))

    elif style == "heavy":
        out = flatten_lighting(out, strength=rng.uniform(0.3, 0.5))
        out = sharpen_edges(out, amount=rng.uniform(0.5, 0.8))
        out = quantize_colors(out, levels=rng.integers(16, 32))
        out = add_banding(out, bands=rng.integers(32, 64))
        out = gamma_shift(out, gamma=rng.uniform(1.1, 1.3))
        out = reduce_specular(out, threshold=0.8, reduction=rng.uniform(0.3, 0.6))
        out = add_slight_blur(out, sigma=rng.uniform(0.3, 0.7))

    return out.clip(0, 1).astype(np.float32)


if __name__ == "__main__":
    # Visual test — apply all styles to a gradient test image
    H, W = 256, 512
    y = np.linspace(0, 1, H).reshape(-1, 1, 1) * np.ones((1, W, 3))
    x = np.linspace(0, 1, W).reshape(1, -1, 1) * np.ones((H, 1, 3))
    test_img = ((y + x) / 2).astype(np.float32)

    for style in ["none", "light", "medium", "heavy"]:
        result = apply_game_style(test_img, style)
        cv2.imwrite(f"test_style_{style}.png", (result * 255).astype(np.uint8)[..., ::-1])
        print(f"{style}: min={result.min():.3f} max={result.max():.3f} mean={result.mean():.3f}")
