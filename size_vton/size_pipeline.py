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

def _face_chin_y(person_img: Image.Image) -> int:
    """
    Returns the y-coordinate below which the garment mask is allowed.
    Fixed at 22% of image height — covers the face/neck region for
    typical fashion photos regardless of pose or face angle.
    """
    return int(person_img.height * 0.22)


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
        #    ESRGAN the garment first (if available) so the VAE encoder gets
        #    sharper texture detail. Fall back to 2× LANCZOS without upscaler.
        person_resized = resize_and_crop(person_image, (WIDTH, HEIGHT))
        if upscaler is not None:
            print("  [ESRGAN] Upscaling garment for conditioning...")
            garment_resized = upscaler.upscale(garment_image, out_size=(WIDTH, HEIGHT))
        else:
            gw, gh = garment_image.size
            garment_up = garment_image.resize((gw * 2, gh * 2), Image.LANCZOS)
            garment_resized = resize_and_padding(garment_up, (WIDTH, HEIGHT))

        # 2. Generate base mask via AutoMasker
        base_mask = self.masker(person_resized, cloth_type)["mask"]

        # 2b. Cap mask at top 22% — garment never bleeds onto face/neck
        chin_y = _face_chin_y(person_resized)
        mask_np = np.array(base_mask.convert("L"))
        mask_np[:chin_y, :] = 0
        base_mask = Image.fromarray(mask_np).convert(base_mask.mode)
        print(f"  [FaceCap] Mask zeroed above y={chin_y}.")

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

        # 8a. Restore original pixels anywhere diffusion shouldn't have touched:
        #     (a) Face region — everything above chin_y
        #     (b) White speckle artifacts — near-white pixels outside the garment mask
        result_np  = np.array(result)
        orig_np    = np.array(person_resized)
        mask_bool  = np.array(smooth_mask.convert("L")) > 127

        # (a) Hard-restore face
        result_np[:chin_y, :] = orig_np[:chin_y, :]

        # (b) Kill white artifacts outside the mask (arms, skin, background)
        is_white    = (result_np[:, :, 0] > 230) & (result_np[:, :, 1] > 230) & (result_np[:, :, 2] > 230)
        is_artifact = is_white & ~mask_bool
        result_np[is_artifact] = orig_np[is_artifact]

        result = Image.fromarray(result_np)

        # 8b. Composite back onto original background.
        if original_person_image is not None:
            orig_resized = resize_and_crop(original_person_image, (WIDTH, HEIGHT))
            orig_bg_np   = np.array(orig_resized)
            result_np2   = np.array(result)

            # Pixel-diff composite: where diffusion changed a pixel → use result,
            # where it didn't → use original background. No masks needed.
            diff    = np.abs(result_np2.astype(np.int32) - orig_np.astype(np.int32)).max(axis=2)
            changed = diff > 25  # pixels the diffusion model actually modified
            composite = np.where(changed[:, :, np.newaxis], result_np2, orig_bg_np)
            result  = Image.fromarray(composite.astype(np.uint8))

        output = {
            "result_image":    result,
            "size_style":      size_style.value,
        }
        if debug:
            output["mask_used"]      = smooth_mask
            output["garment_scaled"] = scaled_garment
            output["base_mask"]      = base_mask
            output["person_resized"] = person_resized
            if skin_filled is not None:
                output["skin_filled"] = skin_filled

        return output
