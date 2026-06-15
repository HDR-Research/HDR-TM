from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .colorspace import convert_rgb_to_srgb


class HDRMonitorDriver:
    """Cross-platform software replacement for the MATLAB HDR display driver."""

    def __init__(self, display_max: float = 1000.0, preview_exposure: float = 1.0) -> None:
        self.display_max = float(display_max)
        self.preview_exposure = float(preview_exposure)
        self.last_frame: np.ndarray | None = None

    def encode(self, image: np.ndarray) -> np.ndarray:
        linear = np.clip(np.asarray(image, dtype=np.float64) * self.preview_exposure / max(self.display_max, 1e-12), 0, 1)
        return np.clip(convert_rgb_to_srgb(linear), 0, 1)

    def display(self, image: np.ndarray, window_name: str = "HDR preview", wait: int = 1) -> np.ndarray:
        self.last_frame = self.encode(image)
        cv2.imshow(window_name, cv2.cvtColor(np.float32(self.last_frame), cv2.COLOR_RGB2BGR))
        cv2.waitKey(wait)
        return self.last_frame

    def write(self, image: np.ndarray, filename: str | Path) -> Path:
        output = self.encode(image)
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(np.rint(output * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        self.last_frame = output
        return path
