"""
Sam2GarmentMasker — uses SAM2.1-large with point prompts for garment segmentation.

FG points are derived from the SCHP mask centroid.
BG points exclude head, edges, and irrelevant body parts.
Falls back to SCHP blob if SAM2 returns no useful mask.

Same call interface as AutoMasker:
    masker(person_img_pil, cloth_type) -> {"mask": PIL Image (L, 0/255)}
"""

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes

from sam2.sam2_image_predictor import SAM2ImagePredictor

_MIN_IOU_FALLBACK = 0.10
_SAM2_MODEL      = "facebook/sam2.1-hiera-large"

# Background point templates (normalized x, y — 0=left/top, 1=right/bottom).
# Applied for upper/overall garments to exclude head and image edges.
_BG_UPPER = [
    (0.50, 0.07),   # top center (head crown)
    (0.50, 0.14),   # mid head
    (0.20, 0.11),   # left temple
    (0.80, 0.11),   # right temple
    (0.50, 0.20),   # lower face / chin
    (0.02, 0.30),   # far left edge
    (0.98, 0.30),   # far right edge
    (0.02, 0.55),   # mid left edge
    (0.98, 0.55),   # mid right edge
]

# For lower body: exclude torso + head, keep legs
_BG_LOWER = [
    (0.50, 0.07),
    (0.50, 0.14),
    (0.50, 0.30),   # torso center
    (0.02, 0.50),
    (0.98, 0.50),
]

_BG_TEMPLATES = {
    "upper":   _BG_UPPER,
    "lower":   _BG_LOWER,
    "overall": _BG_UPPER,
}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _schp_fg_points(mask_np: np.ndarray, cloth_type: str):
    """
    Derive 1-3 foreground points from the SCHP garment blob.
    Returns list of (x, y) pixel coordinates.
    """
    H, W = mask_np.shape
    ys, xs = np.where(mask_np)
    if len(xs) == 0:
        return [(W * 0.5, H * 0.45)]   # safe fallback

    pts = []
    # Upper-half centroid (avoids hem dragging point toward belt)
    upper_mask = mask_np[:H // 2]
    uy, ux = np.where(upper_mask)
    if len(ux) > 0:
        pts.append((float(ux.mean()), float(uy.mean())))

    # Full centroid
    pts.append((float(xs.mean()), float(ys.mean())))

    # For lower body, also add lower-half centroid to capture legs
    if cloth_type == "lower":
        lower_mask = mask_np[H // 2:]
        ly, lx = np.where(lower_mask)
        if len(lx) > 0:
            pts.append((float(lx.mean()), float(ly.mean()) + H // 2))

    return pts


class Sam2GarmentMasker:
    """
    Wraps AutoMasker + SAM2.1-large.

    AutoMasker provides the SCHP blob used to:
      - derive foreground point prompts for SAM2
      - validate the returned mask (IoU check)
      - serve as fallback if SAM2 fails
    """

    def __init__(self, base_masker, device: str = "cuda"):
        self.base_masker = base_masker
        self.device      = device
        print(f"Loading SAM2.1-large ({_SAM2_MODEL})...")
        self.predictor = SAM2ImagePredictor.from_pretrained(
            _SAM2_MODEL, device=device
        )
        print("SAM2.1 ready.")

    def __call__(self, person_image: Image.Image, cloth_type: str) -> dict:
        # 1. AutoMasker → SCHP blob (used as reference, not final output)
        base_result = self.base_masker(person_image, cloth_type)
        base_np     = (np.array(base_result["mask"]) > 127)

        H, W = base_np.shape
        ct   = cloth_type if cloth_type in _BG_TEMPLATES else "upper"

        # 2. Build point prompts
        fg_pts  = _schp_fg_points(base_np, ct)
        bg_tmpl = _BG_TEMPLATES[ct]
        bg_pts  = [(x * W, y * H) for x, y in bg_tmpl]

        coords = np.array(fg_pts + bg_pts, dtype=np.float32)
        labels = np.array(
            [1] * len(fg_pts) + [0] * len(bg_pts), dtype=np.int32
        )

        # 3. SAM2 inference
        image_np = np.array(person_image)
        with torch.inference_mode():
            self.predictor.set_image(image_np)
            masks, _, _ = self.predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=True,
            )
        # masks: (N, H, W) bool

        # 4. Pick best mask by IoU with SCHP blob
        best_mask, best_iou = None, 0.0
        for i in range(masks.shape[0]):
            m   = masks[i].astype(bool)
            iou = _iou(m, base_np)
            if iou > best_iou:
                best_iou, best_mask = iou, m

        # 5. Fallback
        if best_mask is None or best_iou < _MIN_IOU_FALLBACK:
            print(f"  [SAM2] No good mask (best IoU={best_iou:.2f}) → SCHP fallback.")
            return base_result

        # 6. Fill interior holes + hard head-exclusion clip
        sam2_mask = binary_fill_holes(best_mask).astype(np.uint8)
        sam2_mask[:int(H * 0.22), :] = 0

        area_ratio = sam2_mask.sum() / max(base_np.sum(), 1)
        print(f"  [SAM2] IoU={best_iou:.2f}  area_ratio={area_ratio:.2f} ✓")

        # Pass through all keys from base_result (densepose, schp_lip, schp_atr)
        # so downstream code can use them (e.g. for background replacement).
        result = dict(base_result)
        result["mask"] = Image.fromarray((sam2_mask * 255).astype(np.uint8), mode="L")
        return result


# Alias so existing code using Sam3GarmentMasker still works
Sam3GarmentMasker = Sam2GarmentMasker
