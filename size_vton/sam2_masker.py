"""
Sam2GarmentMasker — uses SAM2.1-large with point prompts for torso segmentation.

FG points densely cover the entire torso region so SAM2 segments the full trunk,
not just where the SCHP blob happens to land.
BG points exclude head, legs, arms, and image edges.
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

# ── Foreground point templates (normalized x, y) ─────────────────────────────
# Dense grid across the full torso: shoulders → waist.
_FG_UPPER = [
    (0.50, 0.30),   # upper chest center
    (0.32, 0.33),   # left chest
    (0.68, 0.33),   # right chest
    (0.50, 0.42),   # mid chest
    (0.30, 0.45),   # left mid torso
    (0.70, 0.45),   # right mid torso
    (0.50, 0.52),   # belly center
    (0.32, 0.55),   # left waist
    (0.68, 0.55),   # right waist
    (0.20, 0.38),   # left shoulder
    (0.80, 0.38),   # right shoulder
]

# Lower body: cover thighs and legs
_FG_LOWER = [
    (0.35, 0.60),   # left upper thigh
    (0.65, 0.60),   # right upper thigh
    (0.35, 0.72),   # left mid thigh
    (0.65, 0.72),   # right mid thigh
    (0.35, 0.83),   # left knee
    (0.65, 0.83),   # right knee
    (0.50, 0.65),   # crotch/center
]

# Overall (dress): union of torso + legs
_FG_OVERALL = _FG_UPPER + _FG_LOWER

# ── Background point templates ────────────────────────────────────────────────
_BG_UPPER = [
    (0.50, 0.06),   # head crown
    (0.50, 0.13),   # mid head
    (0.20, 0.10),   # left temple
    (0.80, 0.10),   # right temple
    (0.50, 0.21),   # chin / lower face
    (0.02, 0.35),   # far left edge
    (0.98, 0.35),   # far right edge
    (0.02, 0.55),   # mid left edge
    (0.98, 0.55),   # mid right edge
    (0.35, 0.75),   # left leg
    (0.65, 0.75),   # right leg
    (0.50, 0.85),   # lower body
]

_BG_LOWER = [
    (0.50, 0.06),   # head crown
    (0.50, 0.13),   # mid head
    (0.50, 0.28),   # torso center
    (0.30, 0.35),   # left torso
    (0.70, 0.35),   # right torso
    (0.02, 0.55),   # far left edge
    (0.98, 0.55),   # far right edge
]

_BG_OVERALL = [
    (0.50, 0.06),
    (0.50, 0.13),
    (0.20, 0.10),
    (0.80, 0.10),
    (0.50, 0.21),
    (0.02, 0.35),
    (0.98, 0.35),
    (0.02, 0.65),
    (0.98, 0.65),
]

_FG_TEMPLATES = {
    "upper":   _FG_UPPER,
    "lower":   _FG_LOWER,
    "overall": _FG_OVERALL,
}
_BG_TEMPLATES = {
    "upper":   _BG_UPPER,
    "lower":   _BG_LOWER,
    "overall": _BG_OVERALL,
}

# ── Person silhouette point templates (full-body segmentation) ────────────────
_FG_PERSON = [
    (0.50, 0.08),   # top of head
    (0.50, 0.18),   # face
    (0.50, 0.32),   # neck / upper chest
    (0.50, 0.50),   # mid torso
    (0.38, 0.68),   # left upper leg
    (0.62, 0.68),   # right upper leg
    (0.38, 0.85),   # left lower leg
    (0.62, 0.85),   # right lower leg
]
_BG_PERSON = [
    (0.02, 0.02), (0.50, 0.02), (0.98, 0.02),
    (0.02, 0.50), (0.98, 0.50),
    (0.02, 0.98), (0.50, 0.98), (0.98, 0.98),
]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


class Sam2GarmentMasker:
    """
    Wraps AutoMasker + SAM2.1-large.

    SAM2 is prompted with a dense hardcoded grid of FG points covering the
    full torso (shoulders → waist) and BG points excluding head/legs/edges.
    AutoMasker's SCHP blob is used only for IoU validation and fallback.
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
        ct   = cloth_type if cloth_type in _FG_TEMPLATES else "upper"

        # 2. Build point prompts — dense torso FG grid + exclusion BG points
        fg_pts = [(x * W, y * H) for x, y in _FG_TEMPLATES[ct]]
        bg_pts = [(x * W, y * H) for x, y in _BG_TEMPLATES[ct]]

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


    def segment_person(self, person_image: Image.Image) -> Image.Image:
        """
        Segment the full person body using SAM2 point prompts.

        Returns a PIL L image (255 = person, 0 = background).
        Reuses the same SAM2 predictor already loaded for garment masking.
        """
        image_np = np.array(person_image)
        H, W = image_np.shape[:2]

        fg_pts = [(x * W, y * H) for x, y in _FG_PERSON]
        bg_pts = [(x * W, y * H) for x, y in _BG_PERSON]
        coords = np.array(fg_pts + bg_pts, dtype=np.float32)
        labels = np.array(
            [1] * len(fg_pts) + [0] * len(bg_pts), dtype=np.int32
        )

        with torch.inference_mode():
            self.predictor.set_image(image_np)
            masks, _, _ = self.predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=True,
            )

        # Pick the largest mask — should be the full body
        best_mask = masks[max(range(len(masks)), key=lambda i: masks[i].sum())]

        # Fill interior holes (armpit gaps, etc.) without using convex hull
        filled = binary_fill_holes(best_mask).astype(np.uint8) * 255
        print(f"  [SAM2 person] segmented {filled.sum() // 255} px")
        return Image.fromarray(filled, mode="L")


# Alias so existing code using Sam3GarmentMasker still works
Sam3GarmentMasker = Sam2GarmentMasker
