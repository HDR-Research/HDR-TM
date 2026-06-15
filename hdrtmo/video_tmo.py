from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .analysis import log_mean
from .colorspace import convert_rgb_to_srgb, luminance
from .core import matlab_percentile, remove_specials
from .operators import color_correction, reinhard_tmo
from .tmo import drago_tmo, gamma_tmo
from .tools import false_color
from .video import LazyVideoWriter, VideoStream, hdrv_get_frame


def _frames(source: np.ndarray | VideoStream) -> tuple[list[np.ndarray], float]:
    if isinstance(source, VideoStream):
        return [hdrv_get_frame(source, index)[0] for index in range(source.total_frames)], source.frame_rate
    array = np.asarray(source)
    if array.ndim == 3:
        return [array], 24.0
    if array.ndim != 4:
        raise ValueError("Video must have shape HxWxC or HxWxCxN")
    return [array[..., index] for index in range(array.shape[3])], 24.0


def _encode(frame: np.ndarray, gamma: float) -> np.ndarray:
    frame = np.maximum(remove_specials(frame), 0)
    return np.clip(convert_rgb_to_srgb(frame) if gamma < 0 else gamma_tmo(frame, gamma), 0, 1)


def _finish(frames: list[np.ndarray], output: str | Path | None, frame_rate: float) -> np.ndarray:
    stack = np.stack(frames, axis=3)
    if output is None:
        return stack
    path = Path(output)
    if path.suffix.lower() in {".avi", ".mp4"}:
        writer = LazyVideoWriter(path, frame_rate)
        for frame in frames:
            writer.write(frame)
        writer.close()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames, 1):
            filename = path.with_name(f"{path.stem}{index:010d}{path.suffix or '.png'}")
            cv2.imwrite(str(filename), cv2.cvtColor(np.rint(np.clip(frame, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return stack


def static_tmov(source: np.ndarray | VideoStream, output: str | Path | None = None, operator: Callable[[np.ndarray], np.ndarray] = drago_tmo, gamma: float = 2.2, quality: int = 95, profile: str = "mp4v") -> np.ndarray:
    del quality, profile
    frames, rate = _frames(source)
    return _finish([_encode(operator(np.maximum(remove_specials(frame), 0)), gamma) for frame in frames], output, rate)


def gamma_exposure_tmov(source: np.ndarray | VideoStream, output: str | Path | None = None, gamma: float = 2.2, fstop: float = 0.0, quality: int = 95, profile: str = "mp4v") -> np.ndarray:
    del quality, profile
    frames, rate = _frames(source)
    result = [_encode(np.maximum(remove_specials(frame), 0) * 2**fstop, gamma) for frame in frames]
    return _finish(result, output, rate)


def false_color_tmov(source: np.ndarray | VideoStream, output: str | Path | None = None, compression: str = "log10", luminance_range: tuple[float, float] = (0, 64), quality: int = 95, profile: str = "mp4v") -> np.ndarray:
    del quality, profile
    frames, rate = _frames(source)
    result = [false_color(np.maximum(remove_specials(frame), 0), compression, False, luminance_range)[0] for frame in frames]
    return _finish(result, output, rate)


def scene_key(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    return 0.18 * np.power(2.0, 2 * (np.asarray(b) - np.asarray(a)) / np.maximum(np.asarray(a) + np.asarray(b), 1e-12))


def reinhard_tmov(source: np.ndarray | VideoStream, output: str | Path | None = None, alpha: float = 0.5, white: float = 1e6, gamma: float = -2.2, quality: int = 95, profile: str = "mp4v") -> np.ndarray:
    del quality, profile
    frames, rate = _frames(source)
    result, previous = [], None
    for frame in frames:
        frame = np.maximum(remove_specials(frame), 0)
        current = log_mean(luminance(frame))
        adaptation = current if previous is None else 0.5 * (previous + current)
        mapped = reinhard_tmo(frame, alpha, white)[0]
        mapped = color_correction(_encode(mapped * current / max(adaptation, 1e-12), gamma), 0.9)
        result.append(np.clip(mapped, 0, 1))
        previous = adaptation
    return _finish(result, output, rate)


def kiser_tmov(source: np.ndarray | VideoStream, output: str | Path | None = None, alpha: float = 0.98, clamp_levels: bool = False, white: float = 1e6, gamma: float = -2.2, quality: int = 95, profile: str = "mp4v") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del quality, profile
    frames, rate = _frames(source)
    result, maxima, filtered = [], [], []
    a_prev = b_prev = key_prev = maximum_prev = None
    for frame in frames:
        frame = np.maximum(remove_specials(frame), 0)
        lum = luminance(frame)
        if clamp_levels:
            low, high = matlab_percentile(lum, 0.001), matlab_percentile(lum, 0.999)
            frame = np.clip(frame, low, high) - low
            lum = luminance(frame)
        average, maximum, minimum = log_mean(lum), float(np.max(lum)), float(np.min(lum))
        a, b = maximum - average, average - minimum
        if a_prev is None:
            a_prev, b_prev, key_prev, maximum_prev = a, b, float(scene_key(a, b)), maximum
        an, bn = (1 - alpha) * a_prev + alpha * a, (1 - alpha) * b_prev + alpha * b
        key = float((1 - alpha) * key_prev + alpha * scene_key(an, bn))
        result.append(_encode(reinhard_tmo(frame, key, white)[0], gamma))
        maximum_prev = 0.5 * (maximum_prev + maximum)
        maxima.append(maximum)
        filtered.append(maximum_prev)
        a_prev, b_prev, key_prev = a, b, key
    return _finish(result, output, rate), np.asarray(maxima), np.asarray(filtered)


def ramsey_tmov(source: np.ndarray | VideoStream, output: str | Path | None = None, alpha: float = 0.18, white: float = 1e10, gamma: float = -1.0, quality: int = 95, profile: str = "mp4v") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del quality, profile
    frames, rate = _frames(source)
    highlights = np.asarray([matlab_percentile(luminance(np.maximum(frame, 0)), 0.99) for frame in frames])
    adaptation, result = np.empty_like(highlights), []
    for index, frame in enumerate(frames):
        start = index
        while start > 0 and index - start < 60 and ((0.9 * highlights[index] < highlights[start] < 1.1 * highlights[index]) or index - start < 4):
            start -= 1
        adaptation[index] = np.exp(np.mean(np.log(np.maximum(highlights[start : index + 1], 1e-12))))
        mapped = reinhard_tmo(np.maximum(remove_specials(frame), 0), alpha, white)[0]
        result.append(_encode(mapped * highlights[index] / max(adaptation[index], 1e-12), gamma))
    return _finish(result, output, rate), highlights, adaptation


FalseColorTMOv = false_color_tmov
GammaExposureTMOv = gamma_exposure_tmov
KiserTMOv = kiser_tmov
SceneKey = scene_key
RamseyTMOv = ramsey_tmov
ReinhardTMOv = reinhard_tmov
StaticTMOv = static_tmov
