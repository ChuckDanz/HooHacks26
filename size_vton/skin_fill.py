"""
SkinFill — fills the garment region with the person's own skin tone.

No extra model. No extra inference. Pure CV:
  1. Detect exposed skin pixels (HSV range) outside the garment mask.
  2. Sample mean skin color + texture grain from those real pixels.
  3. Synthesize a skin patch: mean color + matched noise.
  4. Gaussian-blend the patch into the garment region so edges are smooth.

The result is fed to CatVTON as the person image for TIGHT style,
so the hem-reveal zone shows skin rather than the original shirt.
"""

import cv2
import numpy as np
from PIL import Image


# HSV skin-tone ranges — covers light to dark complexions.
# Hue 0-25 (orange-red), Saturation 20-170, Value 60-255.
_SKIN_H_LO, _SKIN_H_HI = 0,  25
_SKIN_S_LO, _SKIN_S_HI = 20, 170
_SKIN_V_LO              = 60


def _detect_skin(rgb: np.ndarray) -> np.ndarray:
    """Return boolean mask (H, W) of skin-tone pixels in an RGB image."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int32)
    return (
        (hsv[:, :, 0] >= _SKIN_H_LO) & (hsv[:, :, 0] <= _SKIN_H_HI) &
        (hsv[:, :, 1] >= _SKIN_S_LO) & (hsv[:, :, 1] <= _SKIN_S_HI) &
        (hsv[:, :, 2] >= _SKIN_V_LO)
    )


def fill_with_skin(
    person_image: Image.Image,
    garment_mask: Image.Image,
    blend_radius: int = 51,
    noise_std: float = 6.0,
) -> Image.Image:
    """
    Replace the garment region with the person's own skin tone.

    Args:
        person_image:  PIL RGB, model-resized (768×1024).
        garment_mask:  PIL grayscale (L), 255=garment region, 0=keep.
                       Pass the FULL base mask (not the TIGHT-eroded version).
        blend_radius:  Gaussian kernel size for boundary feathering (must be odd).
        noise_std:     Std-dev of per-pixel noise added for natural skin texture.

    Returns:
        PIL RGB with garment region filled by the person's own skin tone,
        boundary-blended so edges are smooth.
    """
    person_np = np.array(person_image).astype(np.float32)
    H, W = person_np.shape[:2]

    mask_np  = np.array(garment_mask.convert("L"))
    garment  = (mask_np > 127)   # True where garment is
    exposed  = ~garment           # True where original photo is kept

    # ── 1. Find exposed skin pixels ───────────────────────────────────────────
    person_u8   = np.array(person_image)
    skin_pixels = _detect_skin(person_u8) & exposed

    if skin_pixels.sum() < 50:
        # Fallback: look only in the top third of the image (face/neck)
        top_band = np.zeros((H, W), bool)
        top_band[:H // 3, :] = True
        skin_pixels = _detect_skin(person_u8) & top_band

    if skin_pixels.sum() < 10:
        # Last resort: neutral mid-tone skin
        mean_skin = np.array([195, 155, 125], dtype=np.float32)
    else:
        mean_skin = person_u8[skin_pixels].mean(axis=0).astype(np.float32)

    print(f"  [SkinFill] Sampled skin from {skin_pixels.sum()} pixels, "
          f"mean RGB=({mean_skin[0]:.0f},{mean_skin[1]:.0f},{mean_skin[2]:.0f})")

    # ── 2. Build a skin-tone fill with matched noise ──────────────────────────
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, noise_std, (H, W, 3)).astype(np.float32)
    skin_fill = np.clip(mean_skin + noise, 0, 255)

    # ── 3. Gaussian-blend skin fill into garment region ───────────────────────
    if blend_radius % 2 == 0:
        blend_radius += 1

    # Blur the binary mask → smooth alpha channel
    alpha = cv2.GaussianBlur(
        garment.astype(np.float32),
        (blend_radius, blend_radius),
        0,
    )[:, :, np.newaxis]   # (H, W, 1) for broadcasting

    blended = skin_fill * alpha + person_np * (1.0 - alpha)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)
