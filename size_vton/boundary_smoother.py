import cv2
import numpy as np
from PIL import Image


class BoundarySmoother:
    """
    Softens mask edges so CatVTON blends into boundaries naturally.
    Approximates SV-VTON's RMGS (Refined Mask Generation Stage)
    without U²-Net — uses Gaussian feathering instead.

    Critical for realistic hem edges on TIGHT style (midriff reveal).
    A hard mask edge produces a visible pixel seam; a feathered edge
    gives CatVTON room to blend the garment hem into skin naturally.
    """

    def smooth_boundary(
        self,
        mask: np.ndarray,
        blur_radius: int = 21,
        feather_strength: float = 0.4,
    ) -> np.ndarray:
        """
        Args:
            mask: uint8 numpy array (H, W), values 0 or 255
            blur_radius: must be odd. Larger = softer edge (try 15-31)
            feather_strength: 0=hard edge, 1=fully blurred edge
        Returns:
            uint8 numpy array (H, W), same shape, with soft edges
        """
        if blur_radius % 2 == 0:
            blur_radius += 1

        binary = (mask > 127).astype(np.float32)

        # Blur the whole mask to get a smooth gradient at edges
        blurred = cv2.GaussianBlur(binary, (blur_radius, blur_radius), 0)

        # Find the confident interior (well inside the mask)
        interior = cv2.erode(
            (binary * 255).astype(np.uint8),
            np.ones((5, 5), np.uint8),
            iterations=3,
        ).astype(np.float32) / 255.0

        # Interior stays solid 1.0; boundary region gets feathered
        result = np.where(
            interior > 0.5,
            1.0,
            blurred * feather_strength + interior * (1.0 - feather_strength),
        )

        return (result * 255).astype(np.uint8)

    def smooth_hem_only(
        self,
        mask: np.ndarray,
        hem_blur_radius: int = 31,
        band_px: int = 40,
    ) -> np.ndarray:
        """
        Apply extra smoothing specifically at the hem (bottom edge of mask).
        Leaves the rest of the mask hard. Use this for TIGHT style where
        the hem-reveal is the money shot.

        Args:
            mask: uint8 (H, W)
            hem_blur_radius: blur strength at hem line
            band_px: how many pixels above/below hem to apply smoothing
        Returns:
            uint8 (H, W)
        """
        if hem_blur_radius % 2 == 0:
            hem_blur_radius += 1

        result = mask.copy().astype(np.float32)
        hem_row = self._detect_hem_row(mask)

        if hem_row is None:
            return mask

        # Region around hem
        top = max(0, hem_row - band_px)
        bot = min(mask.shape[0], hem_row + band_px)

        band = result[top:bot].astype(np.float32) / 255.0
        blurred_band = cv2.GaussianBlur(band, (hem_blur_radius, 1), 0)
        result[top:bot] = blurred_band * 255.0

        return result.astype(np.uint8)

    def _detect_hem_row(self, mask: np.ndarray) -> int | None:
        """Find the y-coordinate of the bottom edge of the mask."""
        row_coverage = (mask > 127).sum(axis=1)
        active_rows = np.where(row_coverage > mask.shape[1] * 0.1)[0]
        if len(active_rows) == 0:
            return None
        return int(active_rows[-1])
