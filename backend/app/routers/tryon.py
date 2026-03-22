"""
/api/tryon  — runs the CatVTON size pipeline and returns the fitted result image.

The inference script lives outside Docker, so this endpoint shells out to:
    python test_size_pipeline.py --person <tmp> --garment <tmp>
        --clean --bgfill --sam2b --category <cat> --steps 50

Env vars (set when running uvicorn locally):
    PIPELINE_DIR   absolute path to the size_vton/ directory
    PYTHON_BIN     python executable inside the venv  (default: python)
    OUTPUT_DIR     where the pipeline writes result_fitted.jpg
                   (default: <PIPELINE_DIR>/../output/size_test)
"""

import os
import uuid
import shutil
import asyncio
import tempfile
import httpx

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
from app import db

router = APIRouter(prefix="/api", tags=["tryon"])

# ── Config from env ───────────────────────────────────────────────────────────

PIPELINE_DIR = os.getenv("PIPELINE_DIR", "")
PYTHON_BIN   = os.getenv("PYTHON_BIN",  "python")
_OUTPUT_DIR  = os.getenv(
    "OUTPUT_DIR",
    os.path.join(PIPELINE_DIR, "..", "output", "size_test") if PIPELINE_DIR else "",
)
TIMEOUT_SEC      = 300
VALID_CATEGORIES = {"upper_body", "lower_body", "dress"}

# Spoof a browser User-Agent so retailer CDNs don't 403 server-side requests
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Referer": "https://www.google.com/",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _output_dir() -> str:
    if _OUTPUT_DIR:
        return os.path.abspath(_OUTPUT_DIR)
    if PIPELINE_DIR:
        return os.path.abspath(os.path.join(PIPELINE_DIR, "..", "output", "size_test"))
    raise RuntimeError("PIPELINE_DIR env var not set — check start_api.bat")


async def _download_garment(url: str) -> str:
    """Download garment image with browser headers, return temp file path."""
    async with httpx.AsyncClient(timeout=30, follow_redirects=True, headers=_HEADERS) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    suffix = ".webp" if "webp" in content_type else ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(resp.content)
    tmp.close()
    return tmp.name


async def _run_pipeline(person_path: str, garment_path: str, category: str) -> str:
    if not PIPELINE_DIR:
        raise RuntimeError("PIPELINE_DIR is not set — check start_api.bat")

    script = os.path.join(PIPELINE_DIR, "test_size_pipeline.py")
    cmd = [
        PYTHON_BIN, script,
        "--person",   person_path,
        "--garment",  garment_path,
        "--category", category,
        "--clean", "--bgfill", "--sam2b",
        "--steps", "50",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=PIPELINE_DIR,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_SEC)
    except asyncio.TimeoutError:
        proc.kill()
        raise RuntimeError(f"Pipeline timed out after {TIMEOUT_SEC}s")

    log = stdout.decode(errors="replace") if stdout else ""

    if proc.returncode != 0:
        tail = "\n".join(log.splitlines()[-40:])
        raise RuntimeError(f"Pipeline exited {proc.returncode}:\n{tail}")

    result_path = os.path.join(_output_dir(), "result_fitted.jpg")
    if not os.path.exists(result_path):
        raise RuntimeError(f"Pipeline succeeded but result_fitted.jpg not found at {result_path}")
    return result_path


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/tryon")
async def run_tryon(
    garment_url:  str            = Form(...),
    session_id:   str            = Form(""),
    category:     str            = Form("upper_body"),
    garment_name: str            = Form(""),
    person:       Optional[UploadFile] = File(default=None),
):
    if category not in VALID_CATEGORIES:
        raise HTTPException(400, f"category must be one of {VALID_CATEGORIES}")

    person_tmp       = None
    person_tmp_owned = False  # only True for anonymous (no session) uploads
    garment_tmp      = None

    try:
        # ── Resolve person image ──────────────────────────────────────────────
        if person and person.filename:
            person_bytes = await person.read()
            suffix = os.path.splitext(person.filename)[1] or ".jpg"
            if session_id:
                person_tmp = db.save_person_image(session_id, person_bytes, suffix)
                # Persistent cache — do NOT delete after use
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(person_bytes)
                    person_tmp = f.name
                person_tmp_owned = True
        elif session_id:
            person_tmp = db.get_person_image(session_id)
            if person_tmp is None:
                raise HTTPException(400, "No person photo found for this session — please upload one first.")
            # Persistent cache path — do NOT delete it
        else:
            raise HTTPException(400, "person photo is required.")

        # ── Download garment ──────────────────────────────────────────────────
        garment_tmp = await _download_garment(garment_url)

        # ── Run pipeline ──────────────────────────────────────────────────────
        result_path = await _run_pipeline(person_tmp, garment_tmp, category)

        response_path = os.path.join(
            tempfile.gettempdir(),
            f"fitcheck_result_{uuid.uuid4().hex}.jpg",
        )
        shutil.copy2(result_path, response_path)

        return FileResponse(response_path, media_type="image/jpeg", filename="tryon_result.jpg")

    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Garment image download failed ({e.response.status_code}): {garment_url}")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Garment image download error: {e}")
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        import traceback, sys
        tb = traceback.format_exc()
        print(f"[tryon] Unhandled exception:\n{tb}", flush=True, file=sys.stderr)
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    finally:
        # Only delete garment temp + anonymous person temp; never delete cached person
        if person_tmp_owned and person_tmp and os.path.exists(person_tmp):
            try:
                os.unlink(person_tmp)
            except OSError:
                pass
        if garment_tmp and os.path.exists(garment_tmp):
            try:
                os.unlink(garment_tmp)
            except OSError:
                pass


@router.get("/tryon/person")
async def has_person_cached(session_id: str):
    """Check if a person photo is cached for this session."""
    cached = db.get_person_image(session_id) if session_id else None
    return {"cached": cached is not None}


@router.post("/person")
async def upload_person(
    session_id: str = Form(...),
    person: UploadFile = File(...),
):
    """Cache the user's person photo for this session (no try-on, just save)."""
    person_bytes = await person.read()
    suffix = os.path.splitext(person.filename or "photo.jpg")[1] or ".jpg"
    path = db.save_person_image(session_id, person_bytes, suffix)
    return {"cached": True, "path": os.path.basename(path)}
