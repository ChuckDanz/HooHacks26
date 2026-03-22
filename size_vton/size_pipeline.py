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
        self.pipeline  = catvton_pipeline
        self.masker    = Sam2GarmentMasker(masker, device=DEVICE) if use_sam2 else masker
        self.mask_gen  = MaskGenerator()
        # Pix2Pix pipeline takes no mask parameter
        self.is_p2p    = type(catvton_pipeline).__name__ == "CatVTONPix2PixPipeline"

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
        original_person_image: Image.Image = None,
        person_mask: Image.Image = None,
        result_masker=None,
        upscaler=None,
    ) -> dict:
        """
        Args:
            person_image:          PIL RGB (may have cleaned white background for inference)
            garment_image:         PIL RGB, should already have background removed
            size_style:            SizeStyle enum
            garment_category:      "upper_body" | "lower_body" | "dress"
            debug:                 if True, also returns intermediate mask + scaled garment
            skin_fill:             if True and size_style is TIGHT, replace the existing
                                   garment with the person's own sampled skin tone before
                                   CatVTON inference so the hem-reveal zone shows skin.
            original_person_image: if provided, composites the try-on result back onto
                                   this image (restores original background).
            person_mask:           fallback PIL grayscale silhouette (e.g. rembg alpha).
                                   Used only when result_masker is not provided.
            result_masker:         SAM2 masker (or similar) with a segment_person() method.
                                   When provided, segments the CatVTON result itself for
                                   compositing — matches the new rendered silhouette exactly,
                                   handling cases where the garment changes the body outline.

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

        # 1. Resize inputs to model dimensions.
        #    Upscale garment 2× first (LANCZOS) so the subsequent downscale to
        #    768×1024 preserves fine detail (logos, texture, stitching).
        person_resized = resize_and_crop(person_image, (WIDTH, HEIGHT))
        gw, gh = garment_image.size
        garment_up     = garment_image.resize((gw * 2, gh * 2), Image.LANCZOS)
        garment_resized = resize_and_padding(garment_up, (WIDTH, HEIGHT))

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
        torch.cuda.empty_cache()
        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        call_kwargs = dict(
            image=person_resized,
            condition_image=scaled_garment,
            num_inference_steps=num_inference_steps,
        )
        if not self.is_p2p:
            call_kwargs["mask"] = smooth_mask
        result = self.pipeline(
            **call_kwargs,
            guidance_scale=guidance_scale,
            height=HEIGHT,
            width=WIDTH,
            generator=generator,
        )[0]

        # 8. Real-ESRGAN sharpening (optional).
        #    Upscale 4× then resize back to 768×1024 so downstream compositing
        #    works at the same resolution while recovering VAE-decode blur.
        if upscaler is not None:
            print("  [ESRGAN] Upscaling result...")
            result = upscaler.upscale(result, out_size=(WIDTH, HEIGHT))

        # 9. Composite back onto original background.
        #
        #    Paste the CatVTON result onto the original (or bgfill-inpainted)
        #    background using the person silhouette mask.  The mask is
        #    threshold → erode → feathered to avoid blending the white
        #    CatVTON background into the final image at edge pixels.
        if original_person_image is not None:
            orig_resized = resize_and_crop(original_person_image, (WIDTH, HEIGHT))
            if result_masker is not None:
                # Segment the CatVTON result directly — the result has a white bg so
                # SAM2 produces a clean hard binary mask that matches the new rendered
                # silhouette (not the original person shape, which may differ e.g. for
                # a tight garment replacing a bulky one).
                sil_mask = result_masker.segment_person(result)
                orig_resized.paste(result, mask=sil_mask)
            elif person_mask is not None:
                # Fallback: use the pre-computed silhouette (e.g. rembg alpha)
                sil_arr = np.array(
                    person_mask.resize((WIDTH, HEIGHT), Image.LANCZOS).convert("L")
                )
                orig_resized.paste(result, mask=Image.fromarray(sil_arr))
            else:
                # Last resort: paste via garment inpainting mask only
                composite_mask = smooth_mask.convert("L").point(lambda p: 255 if p > 127 else 0)
                orig_resized.paste(result, mask=composite_mask)
            result = orig_resized

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
