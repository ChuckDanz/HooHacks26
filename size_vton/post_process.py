"""
Post-processing effects for virtual try-on results.

    from post_process import apply_all, make_split_image
    result = apply_all(result, garment_img, garment_mask)
    split  = make_split_image(original_person, result)
"""

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter


# ── Film grain ────────────────────────────────────────────────────────────────

def add_film_grain(img: Image.Image, strength: float = 5.0) -> Image.Image:
    """
    Luminance-weighted film grain — more grain in midtones, less at
    pure black/white, matching real camera sensor noise.
    """
    arr = np.array(img.convert("RGB")).astype(np.float32)
    lum = arr.mean(axis=2, keepdims=True) / 255.0
    weight = 4.0 * lum * (1.0 - lum)          # peaks at 0.5 lum
    noise  = np.random.normal(0, strength, arr.shape)
    arr   += noise * weight
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ── Fabric texture shading ────────────────────────────────────────────────────

def fabric_shading(
    result:   Image.Image,
    garment:  Image.Image,
    mask:     Image.Image,
    strength: float = 0.35,
    sigma:    float = 1.5,
) -> Image.Image:
    """
    Overlay high-frequency texture detail from the garment onto the result
    in the masked region. Simulates fabric weave / wrinkle micro-shading
    without needing 3-D geometry.

    strength: 0.0 = no effect, 1.0 = full garment HF overlay
    sigma:    Gaussian blur radius used to extract the low-frequency base
    """
    r_np = np.array(result.convert("RGB")).astype(np.float32)
    g_np = np.array(garment.resize(result.size, Image.LANCZOS)).astype(np.float32)
    m_np = np.array(mask.convert("L").resize(result.size, Image.LANCZOS)).astype(np.float32) / 255.0
    m_np = m_np[:, :, np.newaxis]

    # High-pass = original − blurred  (isolates texture/edge detail)
    blurred = gaussian_filter(g_np, sigma=[sigma, sigma, 0])
    hf      = g_np - blurred                   # signed, centred at 0

    r_np += hf * m_np * strength
    return Image.fromarray(np.clip(r_np, 0, 255).astype(np.uint8))


# ── Logo / print sharpness ────────────────────────────────────────────────────

def preserve_logo_sharpness(
    result:   Image.Image,
    garment:  Image.Image,
    mask:     Image.Image,
    strength: float = 0.5,
    sigma:    float = 0.8,
) -> Image.Image:
    """
    Blend the sharp high-frequency detail from the source garment back onto
    the result inside the garment mask. Recovers logos, prints, and text
    that VAE encode/decode softens.

    Stronger than fabric_shading — use a tighter sigma to hit fine detail.
    """
    r_np = np.array(result.convert("RGB")).astype(np.float32)
    g_np = np.array(garment.resize(result.size, Image.LANCZOS)).astype(np.float32)
    m_np = np.array(mask.convert("L").resize(result.size, Image.LANCZOS)).astype(np.float32) / 255.0
    m_np = m_np[:, :, np.newaxis]

    blurred = gaussian_filter(g_np, sigma=[sigma, sigma, 0])
    hf      = g_np - blurred

    # Only apply where garment has high contrast (likely a logo/print)
    hf_mag  = np.abs(hf).mean(axis=2, keepdims=True)
    detail_weight = np.clip(hf_mag / 30.0, 0, 1)   # ramps up over 30-unit contrast

    r_np += hf * m_np * detail_weight * strength
    return Image.fromarray(np.clip(r_np, 0, 255).astype(np.uint8))


# ── Convenience wrapper ───────────────────────────────────────────────────────

def apply_all(
    result:   Image.Image,
    garment:  Image.Image,
    mask:     Image.Image,
    grain:    float = 5.0,
    shading:  float = 0.35,
    logo:     float = 0.5,
) -> Image.Image:
    """Apply fabric shading → logo preservation → film grain in one call."""
    result = fabric_shading(result, garment, mask, strength=shading)
    result = preserve_logo_sharpness(result, garment, mask, strength=logo)
    result = add_film_grain(result, strength=grain)
    return result


# ── Before / after split ──────────────────────────────────────────────────────

def make_split_image(
    before: Image.Image,
    after:  Image.Image,
    split:  float = 0.5,
    line_width: int = 3,
    line_color: tuple = (255, 255, 255),
) -> Image.Image:
    """
    Side-by-side split: left portion = before, right = after.
    split: fraction of width to show from 'before' (default 0.5)
    """
    W, H   = after.size
    before = before.resize((W, H), Image.LANCZOS)
    after  = after.resize((W, H), Image.LANCZOS)

    split_x = int(W * split)
    canvas  = Image.new("RGB", (W, H))

    # Left half from before, right half from after
    canvas.paste(before.crop((0, 0, split_x, H)), (0, 0))
    canvas.paste(after.crop((split_x, 0, W, H)),  (split_x, 0))

    # Draw dividing line
    arr = np.array(canvas)
    arr[:, split_x : split_x + line_width] = line_color
    return Image.fromarray(arr)
