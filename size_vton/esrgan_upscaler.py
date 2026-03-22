"""
Real-ESRGAN x4plus upscaler via spandrel.
Downloads model weights on first use to ~/.cache/realesrgan/.

Usage:
    from esrgan_upscaler import RealESRGANUpscaler
    upscaler = RealESRGANUpscaler(device="cuda")
    sharp = upscaler.upscale(pil_image)           # 4x larger
    sharp = upscaler.upscale(pil_image, out_size=(768, 1024))  # 4x then resize back
"""

import os
import torch
import numpy as np
from PIL import Image

_MODEL_URL  = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
_CACHE_DIR  = os.path.join(os.path.expanduser("~"), ".cache", "realesrgan")
_MODEL_PATH = os.path.join(_CACHE_DIR, "RealESRGAN_x4plus.pth")

# Tile size for inference — reduce if you hit OOM on the 4x output
_TILE = 512
_TILE_PAD = 10


class RealESRGANUpscaler:
    """
    Thin wrapper around spandrel + RealESRGAN_x4plus.pth.
    Runs tiled inference to avoid OOM on large images.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        if not os.path.exists(_MODEL_PATH):
            print(f"Downloading Real-ESRGAN weights → {_MODEL_PATH}")
            torch.hub.download_url_to_file(_MODEL_URL, _MODEL_PATH, progress=True)
        import spandrel
        self.model = spandrel.ModelLoader(device=self.device).load_from_file(_MODEL_PATH)
        self.model.eval()
        print(f"  Real-ESRGAN ready ({self.model.__class__.__name__}, scale=×4).")

    @torch.inference_mode()
    def upscale(
        self,
        img: Image.Image,
        out_size: tuple[int, int] | None = None,
    ) -> Image.Image:
        """
        Upscale img by 4×.

        Args:
            img:      PIL RGB image
            out_size: if given, resize the 4× result back to (W, H) with LANCZOS.
                      Useful to keep file size down while retaining recovered detail.
        Returns:
            PIL RGB image, 4× the input size (or out_size if provided)
        """
        img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        # (H, W, C) → (1, C, H, W)
        tensor = (
            torch.from_numpy(img_np)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        out = self._tiled_infer(tensor)

        out_np = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        result = Image.fromarray((out_np * 255).astype(np.uint8))

        if out_size is not None:
            result = result.resize(out_size, Image.LANCZOS)

        return result

    def _tiled_infer(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run model in tiles to avoid OOM on large inputs."""
        _, C, H, W = tensor.shape
        scale = 4

        out = torch.zeros(1, C, H * scale, W * scale, device=self.device)

        for y in range(0, H, _TILE):
            for x in range(0, W, _TILE):
                # Tile with padding
                y0 = max(0, y - _TILE_PAD)
                y1 = min(H, y + _TILE + _TILE_PAD)
                x0 = max(0, x - _TILE_PAD)
                x1 = min(W, x + _TILE + _TILE_PAD)

                tile = tensor[:, :, y0:y1, x0:x1]
                tile_out = self.model(tile)

                # Crop padding from output
                pad_y0 = (y - y0) * scale
                pad_y1 = tile_out.shape[2] - (y1 - min(H, y + _TILE)) * scale
                pad_x0 = (x - x0) * scale
                pad_x1 = tile_out.shape[3] - (x1 - min(W, x + _TILE)) * scale

                out[
                    :, :,
                    y * scale : min(H, y + _TILE) * scale,
                    x * scale : min(W, x + _TILE) * scale,
                ] = tile_out[:, :, pad_y0:pad_y1, pad_x0:pad_x1]

        return out
