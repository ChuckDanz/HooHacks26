import cv2
import numpy as np
from PIL import Image
from enum import Enum


class SizeStyle(Enum):
    TIGHT     = "tight"      # Erode mask — skin preserved where mask shrinks
    FITTED    = "fitted"     # No manipulation — model default
    LOOSE     = "loose"      # Dilate mask — garment covers more body area
    OVERSIZED = "oversized"  # Maximum dilate
    JNCO      = "jnco"       # Extreme horizontal dilation — wide-leg / JNCO silhouette


# Tunable parameters — adjust these experimentally on your test images
SIZE_PARAMS = {
    SizeStyle.TIGHT: {
        "erode_kernel":   7,   # px — general inward squeeze
        "erode_iter":     1,
        "height_crop":  0.20,  # crop 20% off the bottom
        "top_crop":     0.04,  # crop 4% off the top
        # Sleeve-specific: wide horizontal kernel squeezes sides aggressively
        # without shrinking the vertical much.  upper_body only.
        "sleeve_kernel": 30,   # px wide — amount trimmed per side
        "sleeve_iter":    2,   # passes
    },
    SizeStyle.FITTED: {
        "erode_kernel":   0,
        "erode_iter":     0,
        "height_crop":  0.0,
    },
    SizeStyle.LOOSE: {
        "dilate_kernel": 20,
        "dilate_iter":    2,
        "height_add":   0.05,  # extend mask this fraction downward
    },
    SizeStyle.OVERSIZED: {
        "dilate_kernel": 25,
        "dilate_iter":    3,
        "height_add":   0.10,
    },
    SizeStyle.JNCO: {
        # Very wide horizontal kernel blows the legs outward aggressively.
        # Tall kernel is small so we don't add much vertical height.
        "h_kernel":    90,   # px — lateral expansion per side
        "h_iter":       3,   # passes → each pass adds ~90px per side
        "v_kernel":    12,   # px — mild downward extension
        "v_iter":       1,
        "height_add":  0.08, # 8% extra length at the hem
    },
}

# How much to scale the garment image to match the mask manipulation.
# TIGHT: scale UP — more fabric compressed into a smaller mask region = tighter look.
# LOOSE: scale UP — more fabric spread over a larger mask region = baggier look.
GARMENT_SCALE = {
    SizeStyle.TIGHT:     1.05,
    SizeStyle.FITTED:    1.00,
    SizeStyle.LOOSE:     1.18,   # larger fabric image → CatVTON renders it looser
    SizeStyle.OVERSIZED: 1.28,
    SizeStyle.JNCO:      1.00,   # non-uniform scale handled separately via JNCO_SCALE
}

# Non-uniform garment rescale for JNCO — stretch wide, barely taller.
JNCO_SCALE = {
    "width":  1.8,   # 80% wider than the standard pants garment image
    "height": 1.05,  # just slightly longer
}


class MaskGenerator:
    """
    Manipulates CatVTON's inpainting masks to simulate garment fit.

    Core mechanic (from SV-VTON):
      - Smaller mask → less body repainted → original skin pixels preserved
      - Larger mask  → more body repainted → garment fills more space

    The midriff-reveal on TIGHT upper_body works because:
      1. Hem is cropped upward (mask bottom removed)
      2. That region stays 0 (keep original pixels)
      3. CatVTON never touches that area → real skin shows through
    """

    def generate_size_mask(
        self,
        base_mask: Image.Image,
        size_style: SizeStyle,
        garment_category: str = "upper_body",
    ) -> np.ndarray:
        """
        Args:
            base_mask: PIL Image from AutoMasker, grayscale, 0=keep, 255=inpaint
            size_style: SizeStyle enum
            garment_category: "upper_body" | "lower_body" | "dress"
        Returns:
            numpy array (H, W) uint8, 0=keep, 255=inpaint
        """
        mask = np.array(base_mask.convert("L"))
        binary = (mask > 127).astype(np.uint8)  # 0 or 1

        # Remove small isolated noise blobs before any morphological ops.
        # The AutoMasker sometimes produces stray pixels outside the main
        # body region — dilation blows these up into visible squares.
        binary = self._keep_largest_component(binary)

        params = SIZE_PARAMS[size_style]

        if size_style == SizeStyle.FITTED:
            return (binary * 255).astype(np.uint8)

        if size_style == SizeStyle.TIGHT:
            # Measure original mask bounds BEFORE erosion so crops are
            # relative to the full garment height, not the already-shrunk region.
            orig_rows = np.where(binary.sum(axis=1) > 0)[0]
            if len(orig_rows) > 0:
                orig_top    = orig_rows[0]
                orig_bottom = orig_rows[-1]
                orig_height = orig_bottom - orig_top
            else:
                orig_top = orig_bottom = orig_height = 0

            # Erode mask inward (lateral squeeze + some vertical shrink)
            k = params["erode_kernel"]
            kernel = np.ones((k, k), np.uint8)
            eroded = cv2.erode(binary, kernel, iterations=params["erode_iter"])

            # Raise hem (upper_body) / minimal crop (lower_body — don't shorten pants)
            crop_frac = 0.05 if garment_category == "lower_body" else params["height_crop"]
            if crop_frac > 0 and orig_height > 0:
                crop_px = int(orig_height * crop_frac)
                eroded[orig_bottom - crop_px:, :] = 0

            # Lower neckline (upper_body only — top_crop has no useful meaning for pants)
            top_frac = 0.0 if garment_category == "lower_body" else params.get("top_crop", 0.0)
            if top_frac > 0 and orig_height > 0:
                top_px = int(orig_height * top_frac)
                eroded[:orig_top + top_px, :] = 0

            # Sleeve trim — upper_body only: wide horizontal kernel that
            # squeezes the left/right sides (sleeves) without shrinking height.
            sk = params.get("sleeve_kernel", 0)
            si = params.get("sleeve_iter", 1)
            if sk > 0 and garment_category == "upper_body":
                sleeve_kernel = np.ones((3, sk), np.uint8)   # 3 rows tall, sk cols wide
                eroded = cv2.erode(eroded, sleeve_kernel, iterations=si)

            # For lower_body tight: erode more horizontally (slim leg effect)
            if garment_category == "lower_body":
                h_kernel = np.ones((3, k * 2), np.uint8)
                eroded = cv2.erode(eroded, h_kernel, iterations=1)

            return (eroded * 255).astype(np.uint8)

        if size_style in (SizeStyle.LOOSE, SizeStyle.OVERSIZED):
            # Dilate mask outward
            k = params["dilate_kernel"]
            kernel = np.ones((k, k), np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=params["dilate_iter"])

            # Prevent dilation from creating edge artifacts at image borders.
            # When the mask is near the bottom/top/sides, dilation fills the
            # image edge with a full-width pixel bar — clamp those regions back.
            border = k * params["dilate_iter"]
            dilated[:border, :]  = np.minimum(dilated[:border, :],  binary[:border, :])
            dilated[-border:, :] = np.minimum(dilated[-border:, :], binary[-border:, :])
            dilated[:, :border]  = np.minimum(dilated[:, :border],  binary[:, :border])
            dilated[:, -border:] = np.minimum(dilated[:, -border:], binary[:, -border:])

            # Extend hem downward
            add_frac = params["height_add"]
            if add_frac > 0:
                H = dilated.shape[0]
                rows_with_mask = np.where(dilated.sum(axis=1) > 0)[0]
                if len(rows_with_mask) > 0:
                    top_of_mask = rows_with_mask[0]
                    bottom_of_mask = rows_with_mask[-1]
                    mask_height = bottom_of_mask - top_of_mask
                    add_px = int(mask_height * add_frac)
                    new_bottom = min(H, bottom_of_mask + add_px)
                    # Fill a rectangle from old bottom to new bottom
                    # using the width of the mask at that row
                    avg_width = int(dilated[max(0, bottom_of_mask - 5):bottom_of_mask + 1].sum(axis=1).mean())
                    center_x = dilated.shape[1] // 2
                    x0 = max(0, center_x - avg_width // 2)
                    x1 = min(dilated.shape[1], center_x + avg_width // 2)
                    dilated[bottom_of_mask:new_bottom, x0:x1] = 1

            return (dilated * 255).astype(np.uint8)

        if size_style == SizeStyle.JNCO:
            p = params
            # 1. Massive horizontal dilation — wide-leg silhouette
            h_kernel = np.ones((5, p["h_kernel"]), np.uint8)   # tall=5 keeps vertical stable
            wide = cv2.dilate(binary, h_kernel, iterations=p["h_iter"])

            # 2. Mild vertical dilation — slightly longer hem
            v_kernel = np.ones((p["v_kernel"], 3), np.uint8)
            wide = cv2.dilate(wide, v_kernel, iterations=p["v_iter"])

            # 3. Clamp top — don't let the lateral blowout creep into the torso.
            #    Find the original top of the pants mask and zero everything above it.
            orig_rows = np.where(binary.sum(axis=1) > 0)[0]
            if len(orig_rows) > 0:
                wide[:orig_rows[0], :] = 0

            # 4. Extend hem downward (same logic as LOOSE)
            add_frac = p["height_add"]
            H = wide.shape[0]
            rows_with_mask = np.where(wide.sum(axis=1) > 0)[0]
            if add_frac > 0 and len(rows_with_mask) > 0:
                top_of_mask    = rows_with_mask[0]
                bottom_of_mask = rows_with_mask[-1]
                mask_height    = bottom_of_mask - top_of_mask
                add_px         = int(mask_height * add_frac)
                new_bottom     = min(H, bottom_of_mask + add_px)
                avg_width = int(wide[max(0, bottom_of_mask - 5):bottom_of_mask + 1].sum(axis=1).mean())
                center_x  = wide.shape[1] // 2
                x0 = max(0, center_x - avg_width // 2)
                x1 = min(wide.shape[1], center_x + avg_width // 2)
                wide[bottom_of_mask:new_bottom, x0:x1] = 1

            return (wide * 255).astype(np.uint8)

        return (binary * 255).astype(np.uint8)

    @staticmethod
    def _keep_largest_component(binary: np.ndarray) -> np.ndarray:
        """Keep only the largest connected white region. Removes noise artifacts."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        if num_labels <= 1:
            return binary
        # Label 0 is background — find largest non-background component
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels == largest).astype(np.uint8)

    def adjust_garment_proportions(
        self, garment_img: Image.Image, size_style: SizeStyle
    ) -> Image.Image:
        """
        Scale garment image to give CatVTON a better prior for fit.
        TIGHT: scale down (smaller garment → model renders it tighter)
        LOOSE: scale up  (larger garment → model renders it looser)
        JNCO:  non-uniform stretch — 1.8× wide, 1.05× tall
        Returns image at same canvas size, centered on white background.
        """
        w, h = garment_img.size

        if size_style == SizeStyle.JNCO:
            new_w = int(w * JNCO_SCALE["width"])
            new_h = int(h * JNCO_SCALE["height"])
        else:
            scale = GARMENT_SCALE[size_style]
            if scale == 1.0:
                return garment_img
            new_w, new_h = int(w * scale), int(h * scale)

        resized = garment_img.resize((new_w, new_h), Image.LANCZOS)

        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas
