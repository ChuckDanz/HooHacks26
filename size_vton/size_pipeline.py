"""
SizeVariablePipeline — wraps CatVTON with mask-based fit simulation.

Usage:
    from size_vton.size_pipeline import SizeVariablePipeline
    from size_vton.mask_generator import SizeStyle

    pipeline = SizeVariablePipeline(catvton_pipeline, masker)
    result = pipeline.run(person_img, garment_img, SizeStyle.TIGHT)

    # Skin fill: replace shirt with person's own skin tone before TIGHT pass
    result = pipeline.run(person_img, garment_img, SizeStyle.TIGHT,
                          skin_fill=True)
"""

import sys
import os
import torch
import numpy as np
from PIL import Image

from mask_generator import MaskGenerator, SizeStyle
from skin_fill import fill_with_skin
from sam2_masker import Sam2GarmentMasker

# CatVTON utils — imported from the cloned repo
CATVTON_DIR = os.path.join(os.path.dirname(__file__), "..", "CatVTON")
sys.path.insert(0, CATVTON_DIR)
from utils import resize_and_crop, resize_and_padding

WIDTH, HEIGHT = 768, 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SizeVariablePipeline:
    """
    Full size-variable try-on pipeline.

    Args:
        catvton_pipeline: loaded CatVTONPipeline instance
        masker: loaded AutoMasker instance
    """

    def __init__(self, catvton_pipeline, masker, use_sam2: bool = False):
        self.pipeline = catvton_pipeline
        self.masker   = Sam2GarmentMasker(masker, device=DEVICE) if use_sam2 else masker
        self.mask_gen = MaskGenerator()

    def run(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        size_style: SizeStyle = SizeStyle.FITTED,
        garment_category: str = "upper_body",
        num_inference_steps: int = 20,
        guidance_scale: float = 2.5,
        seed: int = 42,
        debug: bool = False,
        skin_fill: bool = False,
        use_raw_mask: bool = False,
    ) -> dict:
        """
        Args:
            person_image:    PIL RGB
            garment_image:   PIL RGB, should already have background removed
            size_style:      SizeStyle enum
            garment_category:"upper_body" | "lower_body" | "dress"
            debug:           if True, also returns intermediate mask + scaled garment
            skin_fill:       if True and size_style is TIGHT, replace the existing
                             garment with the person's own sampled skin tone before
                             CatVTON inference so the hem-reveal zone shows skin.

        Returns dict:
            result_image   — PIL Image, try-on result
            size_style     — str
            skin_filled    — PIL Image (debug, only when skin_fill=True + TIGHT)
            mask_used      — PIL Image (debug)
            garment_scaled — PIL Image (debug)
        """
        # Map garment_category → CatVTON cloth_type
        cloth_type_map = {
            "upper_body": "upper",
            "lower_body": "lower",
            "dress":      "overall",
        }
        cloth_type = cloth_type_map.get(garment_category, "upper")

        # 1. Resize inputs to model dimensions
        person_resized  = resize_and_crop(person_image, (WIDTH, HEIGHT))
        garment_resized = resize_and_padding(garment_image, (WIDTH, HEIGHT))

        # 2. Generate base mask via AutoMasker
        base_mask = self.masker(person_resized, cloth_type)["mask"]

        # 3. (Optional) Skin fill for TIGHT style.
        #    Replace the garment region with the person's own sampled skin tone
        #    so the hem-reveal zone shows skin rather than the original shirt.
        skin_filled = None
        if skin_fill and size_style == SizeStyle.TIGHT:
            print("  [SkinFill] Filling garment region with person's skin tone...")
            skin_filled = fill_with_skin(person_resized, base_mask)
            person_resized = skin_filled

        # 4–6. Mask manipulation + garment scaling
        if use_raw_mask:
            # Default mode: pass AutoMasker mask straight through, no manipulation
            smooth_mask    = base_mask.convert("L")
            scaled_garment = garment_resized
        else:
            size_mask_np = self.mask_gen.generate_size_mask(
                base_mask, size_style, garment_category
            )
            smooth_mask    = Image.fromarray(size_mask_np).convert("L")
            scaled_garment = self.mask_gen.adjust_garment_proportions(
                garment_resized, size_style
            )

        # 7. CatVTON inference
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        result = self.pipeline(
            image=person_resized,
            condition_image=scaled_garment,
            mask=smooth_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=HEIGHT,
            width=WIDTH,
            generator=generator,
        )[0]

        output = {
            "result_image":    result,
            "size_style":      size_style.value,
        }
        if debug:
            output["mask_used"]      = smooth_mask
            output["garment_scaled"] = scaled_garment
            output["base_mask"]      = base_mask
            if skin_filled is not None:
                output["skin_filled"] = skin_filled

        return output
