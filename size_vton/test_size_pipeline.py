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

from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from utils import init_weight_dtype

from mask_generator import MaskGenerator, SizeStyle
from boundary_smoother import BoundarySmoother
from size_pipeline import SizeVariablePipeline
from fit_utils import add_fit_badge, make_comparison_grid

# ── Config ────────────────────────────────────────────────────────────────────

ATTN_CKPT      = "zhengchong/CatVTON"
BASE_CKPT      = "runwayml/stable-diffusion-inpainting"
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "output", "size_test")
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PREC     = "bf16"
STEPS          = 30
GUIDANCE_SCALE = 2.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_models():
    print("Loading CatVTON...")
    repo_path = snapshot_download(repo_id=ATTN_CKPT)
    weight_dtype = init_weight_dtype(MIXED_PREC)
    pipeline = CatVTONPipeline(
        base_ckpt=BASE_CKPT,
        attn_ckpt=repo_path,
        attn_ckpt_version="mix",
        weight_dtype=weight_dtype,
        use_tf32=True,
        device=DEVICE,
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
    parser.add_argument("--default", action="store_true",
                        help="Also run a pure CatVTON default pass (no size manipulation) for comparison")
    parser.add_argument("--clean", action="store_true",
                        help="Remove garment background and replace with white (uses rembg)")
    args = parser.parse_args()

    person_img  = Image.open(args.person).convert("RGB")
    garment_img = Image.open(args.garment).convert("RGB")

    if args.clean:
        from rembg import remove as rembg_remove
        print("Cleaning garment background...")
        rgba = rembg_remove(garment_img)
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white.paste(rgba, mask=rgba.split()[3])
        garment_img = white.convert("RGB")
        clean_path = os.path.join(OUTPUT_DIR, "garment_clean.jpg")
        garment_img.save(clean_path)
        print(f"  Saved cleaned garment → {clean_path}")

    pipeline, masker = load_models()

    # Wrap masker with SAM2 now so both preview and inference use it
    if args.sam2:
        from sam2_masker import Sam2GarmentMasker
        masker = Sam2GarmentMasker(masker, device=DEVICE)

    styles = [SizeStyle.TIGHT, SizeStyle.FITTED, SizeStyle.LOOSE]

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
        )
        elapsed = time.time() - start

        # Save skin-fill intermediate (TIGHT + --skin_strip only)
        if "skin_filled" in out:
            fill_path = os.path.join(OUTPUT_DIR, "skin_filled.jpg")
            out["skin_filled"].save(fill_path)
            print(f"  Skin-fill intermediate → {fill_path}")

        # Add fit badge
        result_with_badge = add_fit_badge(out["result_image"], style.value)
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
