from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routers import queue, tryon
import os

app = FastAPI(title="HooHacks Virtual Try-On API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # extension can run from any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(queue.router)
app.include_router(tryon.router)

# Serve the try-on SPA + demo garment images
_static = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static), name="static")

_garments_dir = os.path.join(os.path.dirname(__file__), "..", "garments")
os.makedirs(_garments_dir, exist_ok=True)
app.mount("/garments", StaticFiles(directory=_garments_dir), name="garments")


@app.get("/tryon-page")
def tryon_page():
    return FileResponse(os.path.join(_static, "tryon.html"))


@app.get("/health")
def health():
    return {"status": "ok"}
