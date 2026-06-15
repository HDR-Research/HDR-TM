from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from .core import remove_specials
from .filters import bilateral_filter
from .io import hdrimwrite
from .video import VideoStream, ldrv_get_frame


def find_hdr_ldr_crf(hdr: np.ndarray, ldr: np.ndarray) -> np.ndarray:
    hdr, ldr = np.asarray(hdr, dtype=float), np.asarray(ldr, dtype=float)
    if hdr.shape != ldr.shape:
        raise ValueError("HDR and LDR images must have equal shapes")
    if np.max(ldr) > 1:
        ldr = ldr / 255
    channels = 1 if hdr.ndim == 2 else hdr.shape[2]
    response = np.empty((256, channels))
    grid = np.linspace(0, 1, 256)
    for channel in range(channels):
        x = ldr if channels == 1 else ldr[..., channel]
        y = hdr if channels == 1 else hdr[..., channel]
        y = y / max(float(np.mean(y)), 1e-12)
        mask = (x >= 16 / 255) & (x <= 240 / 255)
        coefficients = np.polyfit(x[mask], y[mask], 3) if np.count_nonzero(mask) >= 4 else np.array([0, 0, 1, 0])
        values = np.maximum(np.polyval(coefficients, grid), 0)
        response[:, channel] = values / max(float(np.max(values)), 1e-12)
    return response


def find_hdr_ldr_scale(hdr: np.ndarray, ldr: np.ndarray) -> float:
    hdr, ldr = np.asarray(hdr, dtype=float), np.asarray(ldr, dtype=float)
    if hdr.shape != ldr.shape:
        raise ValueError("HDR and LDR images must have equal shapes")
    mask = (ldr >= (32 / 255) ** 2.2) & (ldr <= (230 / 255) ** 2.2)
    return float(np.mean(hdr[mask] / ldr[mask])) if np.any(mask) else 1.0


def banterle_enhance_ldr_frame(first: np.ndarray, second: np.ndarray, background_hdr: np.ndarray, blend_mode: str = "linear") -> np.ndarray:
    first, second, background = map(lambda value: np.asarray(value, dtype=float), (first, second, background_hdr))
    mask = np.min(first, axis=2) > 0.7
    difference = cv2.resize(np.abs(np.min(second, axis=2) - np.min(first, axis=2)), None, fx=0.125, fy=0.125)
    difference = cv2.resize(ndimage.gaussian_filter(difference, 0.4), (first.shape[1], first.shape[0]))
    mask &= difference <= 0.1
    structure = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, structure, iterations=2).astype(float)
    mask = bilateral_filter(mask, np.min(first, axis=2), 0, 1, 64, 0.05)
    mask = ndimage.gaussian_filter(mask, 0.4)
    small = cv2.resize(background, None, fx=0.125, fy=0.125)
    reference = cv2.resize(ndimage.gaussian_filter(small, (0.8, 0.8, 0)), (first.shape[1], first.shape[0]))
    ref = background * mask[..., None] + reference * (1 - mask[..., None])
    if blend_mode not in {"linear", "poisson"}:
        raise ValueError("blend_mode must be linear or poisson")
    output = np.exp2(np.log2(ref + 1) * mask[..., None] + np.log2(first + 1) * (1 - mask[..., None])) - 1
    return remove_specials(np.maximum(output, 0))


def create_hdrv_from_image(image: np.ndarray, output_directory: str | Path | None = None, rows: int | None = None, columns: int | None = None, frames: int = 96, start: tuple[float, float] | None = None, end: tuple[float, float] | None = None) -> np.ndarray:
    source = np.asarray(image)
    rows = round(source.shape[0] / 8) if rows is None else rows
    columns = round(source.shape[1] / 8) if columns is None else columns
    start = (columns / 2, rows / 2) if start is None else start
    end = (source.shape[1] - columns / 2 - 1, source.shape[0] - rows / 2 - 1) if end is None else end
    result = []
    for index in range(frames):
        t = index / max(frames - 1, 1)
        x, y = np.round((1 - t) * np.asarray(start) + t * np.asarray(end)).astype(int)
        x0, y0 = np.clip(x - columns // 2, 0, source.shape[1] - columns), np.clip(y - rows // 2, 0, source.shape[0] - rows)
        frame = source[y0 : y0 + rows, x0 : x0 + columns].copy()
        result.append(frame)
        if output_directory is not None:
            hdrimwrite(frame, Path(output_directory) / f"frame_{index + 1:010d}.exr")
    return np.stack(result, axis=3)


def banterle_enhance_ldr_video(video: np.ndarray | VideoStream, background_hdr: np.ndarray, output: str | Path | None = None, crf: np.ndarray | None = None, scale_factor: float | None = None) -> np.ndarray:
    frames = [ldrv_get_frame(video, i)[0] for i in range(video.total_frames)] if isinstance(video, VideoStream) else [np.asarray(video)[..., i] for i in range(np.asarray(video).shape[3])]
    crf = find_hdr_ldr_crf(background_hdr, frames[0]) if crf is None else crf
    grid = np.arange(256)
    linear = [np.stack([np.interp(np.clip(frame[..., c] * 255, 0, 255), grid, crf[:, c]) for c in range(frame.shape[2])], axis=2) for frame in frames]
    scale_factor = find_hdr_ldr_scale(background_hdr, linear[0]) if scale_factor is None else scale_factor
    background = background_hdr / max(scale_factor, 1e-12)
    result = [banterle_enhance_ldr_frame(linear[i], linear[i + 1], background) for i in range(len(linear) - 1)]
    if output is not None:
        path = Path(output)
        for index, frame in enumerate(result, 1):
            hdrimwrite(frame, path.with_name(f"{path.stem}_{index:010d}{path.suffix}"))
    return np.stack(result, axis=3)


BanterleEnhanceLDRVideo = banterle_enhance_ldr_video
CreateHDRvFromImage = create_hdrv_from_image
BanterleEnhanceLDRFrame = banterle_enhance_ldr_frame
findHDRLDRCRF = find_hdr_ldr_crf
findHDRLDRScale = find_hdr_ldr_scale
