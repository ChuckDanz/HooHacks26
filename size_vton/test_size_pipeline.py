"""
Test script: runs TIGHT / FITTED / LOOSE on the same person+garment pair.
Saves individual results + a side-by-side comparison grid.

Usage:
    python size_vton/test_size_pipeline.py \
        --person test_cases/person.jpg \
        --garment test_cases/LateRegTshirt.jpeg \
        --category upper_body

Tune SIZE_PARAMS in mask_generator.py until the masks look right,
then run full inference.
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "CatVTON"))

import torch
from PIL import Image
from huggingface_hub import snapshot_download

from model.pipeline import CatVTONPipeline, CatVTONPix2PixPipeline
from model.cloth_masker import AutoMasker
from utils import init_weight_dtype

from mask_generator import MaskGenerator, SizeStyle
from boundary_smoother import BoundarySmoother
from size_pipeline import SizeVariablePipeline
from fit_utils import add_fit_badge, make_comparison_grid

# ── Config ────────────────────────────────────────────────────────────────────

ATTN_CKPT      = "zhengchong/CatVTON"
BASE_CKPT      = "runwayml/stable-diffusion-inpainting"
BASE_CKPT_P2P  = "timbrooks/instruct-pix2pix"
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "output", "size_test")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PREC     = "bf16"
STEPS          = 30
GUIDANCE_SCALE = 2.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_models():
    print("Loading CatVTON (inpainting)...")
    repo_path = snapshot_download(repo_id=ATTN_CKPT)
    weight_dtype = init_weight_dtype(MIXED_PREC)
    pipeline = CatVTONPipeline(
        base_ckpt=BASE_CKPT,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=weight_dtype,
        use_tf32=True,
        device=DEVICE,
        skip_safety_check=True,
    )
    masker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device=DEVICE,
    )
    return pipeline, masker


def load_models_p2p():
    print("Loading CatVTON-Pix2Pix (in-the-wild, mask-free)...")
    # Pix2Pix attn weights are in the same zhengchong/CatVTON repo under mix-48k-1024/
    repo_path    = snapshot_download(repo_id=ATTN_CKPT)
    weight_dtype = init_weight_dtype(MIXED_PREC)
    pipeline = CatVTONPix2PixPipeline(
        base_ckpt=BASE_CKPT_P2P,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix-48k-1024",
        weight_dtype=weight_dtype,
        use_tf32=True,
        device=DEVICE,
        skip_safety_check=True,
    )
    masker = AutoMasker(
        densepose_ckpt=os.path.join(repo_path, "DensePose"),
        schp_ckpt=os.path.join(repo_path, "SCHP"),
        device=DEVICE,
    )
    return pipeline, masker


def preview_masks(person_img, masker, category, size_styles):
    """Save mask previews without running inference — fast sanity check."""
    from mask_generator import MaskGenerator
    from boundary_smoother import BoundarySmoother
    from utils import resize_and_crop

    person_resized = resize_and_crop(person_img, (768, 1024))
    cloth_type = {"upper_body": "upper", "lower_body": "lower", "dress": "overall"}[category]
    base_mask = masker(person_resized, cloth_type)["mask"]
    base_mask.save(os.path.join(OUTPUT_DIR, "mask_base.png"))
    print(f"  Base mask → output/size_test/mask_base.png")

    mg = MaskGenerator()
    sm = BoundarySmoother()
    for style in size_styles:
        np_mask = mg.generate_size_mask(base_mask, style, category)
        if style == SizeStyle.TIGHT:
            np_mask = sm.smooth_hem_only(np_mask)
        else:
            np_mask = sm.smooth_boundary(np_mask)
        Image.fromarray(np_mask).save(
            os.path.join(OUTPUT_DIR, f"mask_{style.value}.png")
        )
        print(f"  {style.value} mask → output/size_test/mask_{style.value}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person",   required=True)
    parser.add_argument("--garment",  required=True)
    parser.add_argument("--category", default="upper_body",
                        choices=["upper_body", "lower_body", "dress"])
    parser.add_argument("--masks_only", action="store_true",
                        help="Preview masks without running inference (fast)")
    parser.add_argument("--skin_strip", action="store_true",
                        help="Two-pass: strip existing shirt with SD inpainting before TIGHT inference")
    parser.add_argument("--steps",    type=int,   default=STEPS)
    parser.add_argument("--guidance", type=float, default=GUIDANCE_SCALE,
                        help="CFG guidance scale (default 2.5; raise to 4-6 for stronger color transfer)")
    parser.add_argument("--sam2", action="store_true",
                        help="Use SAM2 for clean garment mask instead of SCHP blob")
    parser.add_argument("--sam2b", action="store_true",
                        help="Use SAM2 to segment person from background before inference "
                             "and composite back onto original background after (no rembg needed for person)")
    parser.add_argument("--default", action="store_true",
                        help="Also run a pure CatVTON default pass (no size manipulation) for comparison")
    parser.add_argument("--clean", action="store_true",
                        help="Remove garment background and replace with white (uses rembg)")
    parser.add_argument("--bgfill", action="store_true",
                        help="Inpaint the background where the person was (LaMa) before compositing "
                             "so the final background has no person-shaped hole. Requires --clean.")
    parser.add_argument("--p2p", action="store_true",
                        help="Use CatVTON-Pix2Pix (in-the-wild, mask-free) instead of inpainting pipeline")
    parser.add_argument("--esrgan", action="store_true",
                        help="Apply Real-ESRGAN x4 sharpening to each result (recovers VAE-decode blur)")
    parser.add_argument("--jnco", action="store_true",
                        help="Add JNCO wide-leg style to the inference run (forces --category lower_body)")
    parser.add_argument("--post", action="store_true",
                        help="Apply post-processing: fabric shading, logo sharpness, film grain, split preview")
    parser.add_argument("--overlay", action="store_true",
                        help="Save mask overlay debug images: person + base mask (red) + final mask (green)")
    parser.add_argument("--diffmask", action="store_true",
                        help="Use pixel-difference map instead of silhouette mask for background compositing")
    parser.add_argument("--diffvis", action="store_true",
                        help="Save amplified difference heatmap between original and diffusion output")
    args = parser.parse_args()

    from PIL import ImageOps
    person_img  = ImageOps.exif_transpose(Image.open(args.person)).convert("RGB")
    garment_img = ImageOps.exif_transpose(Image.open(args.garment)).convert("RGB")

    # Keep the original person image for final compositing (background restore)
    original_person_img  = person_img
    person_silhouette    = None   # silhouette mask used for post-inference compositing

    if args.clean:
        import gc
        from rembg import new_session, remove as rembg_remove

        session = new_session()

        print("Cleaning garment background...")
        rgba = rembg_remove(garment_img, session=session)
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white.paste(rgba, mask=rgba.split()[3])
        garment_img = white.convert("RGB")
        garment_img.save(os.path.join(OUTPUT_DIR, "garment_clean.jpg"))
        print(f"  Saved cleaned garment → {os.path.join(OUTPUT_DIR, 'garment_clean.jpg')}")

        if not args.sam2b:
            # rembg handles person bg removal when --sam2b is not set
            print("Cleaning person background with rembg (original bg restored in output)...")
            rgba = rembg_remove(person_img, session=session)
            person_silhouette = rgba.split()[3]
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            white.paste(rgba, mask=person_silhouette)
            person_img = white.convert("RGB")
            person_img.save(os.path.join(OUTPUT_DIR, "person_clean.jpg"))
            print(f"  Saved cleaned person → {os.path.join(OUTPUT_DIR, 'person_clean.jpg')}")

        del session, rgba, white
        gc.collect()
        torch.cuda.empty_cache()
        print("  rembg memory freed.")

    # --bgfill: use LaMa to inpaint background where the person stood,
    # so the final composite has a complete background (no person-shaped hole).
    # Must run after rembg (we need person_silhouette) and before loading CatVTON.
    if args.bgfill:
        if person_silhouette is None:
            print("WARNING: --bgfill requires --clean (need person silhouette). Skipping bgfill.")
        else:
            import cv2
            import numpy as np
            print("Inpainting background with cv2.inpaint...")
            person_np = np.array(original_person_img)
            # Binarize mask: rembg alpha is soft, cv2.inpaint needs hard 0/255
            mask_np = np.array(person_silhouette.convert("L"))
            mask_bin = (mask_np > 10).astype(np.uint8) * 255
            # Dilate slightly so edge fringe pixels are also filled
            mask_bin = cv2.dilate(mask_bin, np.ones((5, 5), np.uint8), iterations=2)
            inpainted = cv2.inpaint(person_np, mask_bin, inpaintRadius=20, flags=cv2.INPAINT_TELEA)
            bg_filled = Image.fromarray(inpainted)
            bg_filled.save(os.path.join(OUTPUT_DIR, "bg_inpaint.jpg"))
            print(f"  Inpainted background → {os.path.join(OUTPUT_DIR, 'bg_inpaint.jpg')}")
            original_person_img = bg_filled

    pipeline, masker = load_models_p2p() if args.p2p else load_models()

    upscaler = None
    if args.esrgan:
        from esrgan_upscaler import RealESRGANUpscaler
        upscaler = RealESRGANUpscaler(device=DEVICE)

    # Load SAM2 once — reused for garment masking (--sam2) and/or bg segmentation (--sam2b)
    sam2_wrapper = None
    if args.sam2 or args.sam2b:
        from sam2_masker import Sam2GarmentMasker
        sam2_wrapper = Sam2GarmentMasker(masker, device=DEVICE)

    # --sam2b: SAM2 segments the CatVTON result for compositing (not the original person).
    # This matches the new rendered silhouette exactly, handling garment outline changes.
    # Without --clean, also generates the white-bg inference image.
    if args.sam2b and not args.clean:
        print("Segmenting person from background with SAM2...")
        person_silhouette = sam2_wrapper.segment_person(person_img)
        white = Image.new("RGB", person_img.size, (255, 255, 255))
        white.paste(person_img, mask=person_silhouette)
        person_img = white
        person_img.save(os.path.join(OUTPUT_DIR, "person_sam2b_clean.jpg"))
        print(f"  Saved SAM2 person → {os.path.join(OUTPUT_DIR, 'person_sam2b_clean.jpg')}")

    # --sam2: use SAM2-enhanced masker for garment inpainting mask
    if args.sam2:
        masker = sam2_wrapper

    styles = [SizeStyle.TIGHT, SizeStyle.FITTED, SizeStyle.LOOSE]
    if args.jnco:
        styles.append(SizeStyle.JNCO)
        args.category = "lower_body"   # JNCO is always lower body

    # ── Mask preview (no GPU inference) ──
    print("\n── Mask previews ────────────────────────────────────")
    preview_masks(person_img, masker, args.category, styles)

    if args.masks_only:
        print("--masks_only set, skipping inference.")
        return

    # ── Full inference ────────────────────────────────────
    # Pass use_sam2=False — masker is already wrapped above if needed
    size_pipeline = SizeVariablePipeline(pipeline, masker, use_sam2=False)

    results = {}

    print("\n── Inference ────────────────────────────────────────")

    # ── Default pass (raw AutoMasker mask, no size manipulation) ──────────────
    if args.default:
        print("Running default (CatVTON baseline)...")
        start = time.time()
        out = size_pipeline.run(
            person_img, garment_img,
            size_style=SizeStyle.FITTED,
            garment_category=args.category,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            debug=True,
            use_raw_mask=True,
            original_person_image=original_person_img if (args.clean or args.sam2b) else None,
            upscaler=upscaler,
        )
        elapsed = time.time() - start
        result_with_badge = add_fit_badge(out["result_image"], "default")
        out_path = os.path.join(OUTPUT_DIR, "result_default.jpg")
        result_with_badge.save(out_path)
        results["default"] = result_with_badge
        print(f"  default: {elapsed:.1f}s → {out_path}")

    for style in styles:
        print(f"Running {style.value}...")
        start = time.time()
        out = size_pipeline.run(
            person_img, garment_img,
            size_style=style,
            garment_category=args.category,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            debug=True,
            skin_fill=args.skin_strip,
            original_person_image=original_person_img if (args.clean or args.sam2b) else None,
            upscaler=upscaler,
        )
        elapsed = time.time() - start

        # Save skin-fill intermediate (TIGHT + --skin_strip only)
        if "skin_filled" in out:
            fill_path = os.path.join(OUTPUT_DIR, "skin_filled.jpg")
            out["skin_filled"].save(fill_path)
            print(f"  Skin-fill intermediate → {fill_path}")

        # Mask overlay debug (--overlay)
        if args.overlay and "base_mask" in out and "mask_used" in out:
            import numpy as np
            person_arr = np.array(out.get("skin_filled") or person_img.resize(
                (out["base_mask"].width, out["base_mask"].height), Image.LANCZOS
            )).copy()
            base_arr   = np.array(out["base_mask"].convert("L"))
            final_arr  = np.array(out["mask_used"].convert("L"))
            # Red = base mask, Green = final mask used for inference
            person_arr[base_arr  > 127, 0] = 220
            person_arr[base_arr  > 127, 1] = 30
            person_arr[base_arr  > 127, 2] = 30
            person_arr[final_arr > 127, 0] = 30
            person_arr[final_arr > 127, 1] = 220
            person_arr[final_arr > 127, 2] = 30
            overlay_img  = Image.fromarray(person_arr)
            overlay_path = os.path.join(OUTPUT_DIR, f"overlay_{style.value}.jpg")
            overlay_img.save(overlay_path)
            print(f"  Mask overlay → {overlay_path}  (red=base, green=final)")

        # Diff visualisation (--diffvis)
        if args.diffvis and "person_resized" in out:
            import numpy as np
            orig_f   = np.array(out["person_resized"]).astype(np.float32)
            result_f = np.array(out["result_image"]).astype(np.float32)
            diff     = np.abs(result_f - orig_f).max(axis=2)          # max channel diff
            diff_amp = np.clip(diff * 4.0, 0, 255).astype(np.uint8)   # 4× amplify so subtle changes show
            # Save greyscale diff and false-colour heatmap side by side
            grey    = Image.fromarray(diff_amp).convert("RGB")
            heat_np = np.zeros((*diff_amp.shape, 3), dtype=np.uint8)
            heat_np[:, :, 0] = diff_amp                                # red channel = magnitude
            heat_np[:, :, 1] = np.clip(255 - diff_amp * 2, 0, 255)    # green fades out
            heatmap = Image.fromarray(heat_np)
            canvas  = Image.new("RGB", (grey.width * 2, grey.height))
            canvas.paste(grey,    (0, 0))
            canvas.paste(heatmap, (grey.width, 0))
            diffvis_path = os.path.join(OUTPUT_DIR, f"diffvis_{style.value}.jpg")
            canvas.save(diffvis_path)
            print(f"  Diff vis → {diffvis_path}  (left=greyscale ×4, right=heatmap)")

        # Post-processing (--post)
        result_image = out["result_image"]
        if args.post:
            from post_process import apply_all, make_split_image
            garment_mask = out.get("mask_used") or out.get("base_mask")
            result_image = apply_all(result_image, garment_img, garment_mask)
            split = make_split_image(original_person_img, result_image)
            split_path = os.path.join(OUTPUT_DIR, f"split_{style.value}.jpg")
            split.save(split_path)
            print(f"  Split preview → {split_path}")

        # Add fit badge
        result_with_badge = add_fit_badge(result_image, style.value)
        out_path = os.path.join(OUTPUT_DIR, f"result_{style.value}.jpg")
        result_with_badge.save(out_path)
        results[style.value] = result_with_badge
        print(f"  {style.value}: {elapsed:.1f}s → {out_path}")

    # ── Comparison grid ───────────────────────────────────
    print("\nBuilding comparison grid...")
    grid = make_comparison_grid(results, person_img, garment_img)
    grid_path = os.path.join(OUTPUT_DIR, "size_comparison.jpg")
    grid.save(grid_path)
    print(f"Comparison grid → {grid_path}")


if __name__ == "__main__":
    main()
