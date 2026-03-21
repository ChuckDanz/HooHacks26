"""
CatVTON end-to-end test pipeline.

Setup before running:
    git clone https://github.com/Zheng-Chong/CatVTON
    pip install -r CatVTON/requirements.txt
    pip install rembg opencv-python

Usage:
    python test_pipeline.py --person test_cases/person.jpg --garment test_cases/garment.jpg
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2
from PIL import Image

# CatVTON is cloned locally, not pip-installed
CATVTON_DIR = os.path.join(os.path.dirname(__file__), "CatVTON")
if not os.path.exists(CATVTON_DIR):
    raise RuntimeError(
        "CatVTON repo not found. Run:\n"
        "  git clone https://github.com/Zheng-Chong/CatVTON"
    )
sys.path.insert(0, CATVTON_DIR)

import torch
from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from utils import resize_and_crop, resize_and_padding, init_weight_dtype

# ── Config ────────────────────────────────────────────────────────────────────

ATTN_CKPT   = "zhengchong/CatVTON"   # HuggingFace: attention weights
BASE_CKPT   = "runwayml/stable-diffusion-inpainting"
OUTPUT_DIR  = "output"
WIDTH, HEIGHT = 768, 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PRECISION = "bf16"  # RTX 5070 Ti supports bf16

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("test_cases", exist_ok=True)


# ── Quality gates ─────────────────────────────────────────────────────────────

def check_lighting(img: Image.Image) -> str:
    gray = np.array(img.convert("L"))
    mean, std = gray.mean(), gray.std()
    if mean < 60:   return "too_dark"
    if mean > 200:  return "too_bright"
    if std > 90:    return "harsh_shadows"
    return "ok"


def is_complex_pattern(img: Image.Image) -> bool:
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() > 800


def check_face(img: Image.Image) -> bool:
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0


# ── Garment preprocessing ─────────────────────────────────────────────────────

def clean_garment(img: Image.Image) -> Image.Image:
    """Strip background and paste garment onto white."""
    from rembg import remove
    rgba = remove(img)
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    white.paste(rgba, mask=rgba.split()[3])
    return white.convert("RGB")


# ── Pipeline loader ───────────────────────────────────────────────────────────

def load_pipeline():
    print(f"Loading CatVTON pipeline on {DEVICE} ({MIXED_PRECISION})...")
    weight_dtype = init_weight_dtype(MIXED_PRECISION)
    pipeline = CatVTONPipeline(
        base_ckpt=BASE_CKPT,
        attn_ckpt=ATTN_CKPT,
        attn_ckpt_version="mix",
        weight_dtype=weight_dtype,
        use_tf32=True,
        device=DEVICE,
    )
    masker = AutoMasker(
        densepose_ckpt=os.path.join(CATVTON_DIR, "resource/ckpt/densepose"),
        schp_ckpt=os.path.join(CATVTON_DIR, "resource/ckpt/schp"),
        device=DEVICE,
    )
    print("Pipeline ready.")
    return pipeline, masker


# ── Single inference pass ─────────────────────────────────────────────────────

def run_tryon(
    pipeline,
    masker,
    person_img: Image.Image,
    garment_img: Image.Image,
    cloth_type: str = "upper",   # "upper" | "lower" | "overall"
    num_inference_steps: int = 20,
    guidance_scale: float = 2.5,
    seed: int = 42,
) -> Image.Image:

    person_img  = resize_and_crop(person_img,  (WIDTH, HEIGHT))
    garment_img = resize_and_padding(garment_img, (WIDTH, HEIGHT))

    mask = masker(person_img, cloth_type)["mask"]

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipeline(
        image=person_img,
        condition_image=garment_img,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
    )[0]

    return result


# ── Timing sweep ──────────────────────────────────────────────────────────────

def benchmark(pipeline, masker, person_img, garment_img):
    print("\n── Speed benchmark ──────────────────────────────────")
    for steps in [10, 15, 20, 25, 30]:
        start = time.time()
        result = run_tryon(pipeline, masker, person_img, garment_img,
                           num_inference_steps=steps)
        elapsed = time.time() - start
        out_path = os.path.join(OUTPUT_DIR, f"benchmark_{steps}steps.jpg")
        result.save(out_path)
        print(f"  {steps:>2} steps → {elapsed:.1f}s  ({out_path})")
    print("─────────────────────────────────────────────────────\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person",    required=True, help="Path to person image")
    parser.add_argument("--garment",   required=True, help="Path to garment image")
    parser.add_argument("--cloth_type", default="upper",
                        choices=["upper", "lower", "overall"])
    parser.add_argument("--steps",     type=int, default=20)
    parser.add_argument("--benchmark", action="store_true",
                        help="Run timing sweep across step counts")
    args = parser.parse_args()

    person_img  = Image.open(args.person).convert("RGB")
    garment_img = Image.open(args.garment).convert("RGB")

    # ── Quality gates
    lighting = check_lighting(person_img)
    if lighting != "ok":
        print(f"[WARNING] Lighting issue detected: {lighting}")

    if not check_face(person_img):
        print("[WARNING] No face detected with high confidence — identity preservation may be poor")

    if is_complex_pattern(garment_img):
        print("[WARNING] Complex pattern detected on garment — texture may not reproduce perfectly")

    # ── Garment cleanup
    print("Removing garment background...")
    garment_img = clean_garment(garment_img)
    garment_img.save(os.path.join(OUTPUT_DIR, "garment_clean.jpg"))
    print("  Saved: output/garment_clean.jpg")

    # ── Load model
    pipeline, masker = load_pipeline()

    # ── Benchmark sweep or single pass
    if args.benchmark:
        benchmark(pipeline, masker, person_img, garment_img)
    else:
        print(f"Running try-on ({args.steps} steps)...")
        start = time.time()
        result = run_tryon(
            pipeline, masker, person_img, garment_img,
            cloth_type=args.cloth_type,
            num_inference_steps=args.steps,
        )
        elapsed = time.time() - start
        out_path = os.path.join(OUTPUT_DIR, "result.jpg")
        result.save(out_path)
        print(f"Done in {elapsed:.1f}s → {out_path}")


if __name__ == "__main__":
    main()
