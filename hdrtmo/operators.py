from __future__ import annotations

import numpy as np

from .io import remove_specials


LUMINANCE_WEIGHTS = {
    "rec709": (0.2126, 0.7152, 0.0722),
    "rec2020": (0.2627, 0.6780, 0.0593),
}


def luminance(image: np.ndarray, primaries: str = "rec709") -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError("Image must be HxW or HxWxC")
    if image.shape[2] == 1:
        return image[:, :, 0]
    if image.shape[2] == 3:
        if primaries not in LUMINANCE_WEIGHTS:
            raise ValueError("primaries must be 'rec709' or 'rec2020'")
        red, green, blue = LUMINANCE_WEIGHTS[primaries]
        return red * image[:, :, 0] + green * image[:, :, 1] + blue * image[:, :, 2]
    return np.mean(image, axis=2)


def log_mean(image: np.ndarray, delta: float = 1e-6) -> float:
    return float(np.exp(np.mean(np.log(image + delta))))


def _matlab_percentile(image: np.ndarray, percentile: float) -> float:
    values = np.sort(np.asarray(image).reshape(-1))
    percentile = float(np.clip(percentile, 0.0, 1.0))
    matlab_index = max(int(np.floor(values.size * percentile + 0.5)), 1)
    return float(values[matlab_index - 1])


def _reinhard_alpha(luma: np.ndarray, delta: float = 1e-6) -> float:
    low = _matlab_percentile(luma, 0.01)
    high = _matlab_percentile(luma, 0.99)
    log_low = np.log2(low + delta)
    log_high = np.log2(high + delta)
    log_average = np.log2(log_mean(luma) + delta)
    span = log_high - log_low
    if abs(span) < np.finfo(np.float64).eps:
        return 0.18
    return float(0.18 * 4.0 ** ((2.0 * log_average - log_low - log_high) / span))


def _reinhard_white_point(luma: np.ndarray) -> float:
    positive = luma[luma > 0.0]
    if positive.size == 0:
        return 1.0
    log_low = np.log2(np.min(positive) + 1e-6)
    log_high = np.log2(np.max(positive) + 1e-6)
    return float(1.5 * 2.0 ** (log_high - log_low - 5.0))


def change_luminance(
    image: np.ndarray,
    old_luma: np.ndarray,
    new_luma: np.ndarray,
) -> np.ndarray:
    if image.ndim == 2:
        return remove_specials(np.asarray(new_luma, dtype=np.float64))
    ratio = np.divide(
        new_luma,
        old_luma,
        out=np.zeros_like(new_luma, dtype=np.float64),
        where=old_luma != 0.0,
    )
    return remove_specials(image * ratio[:, :, np.newaxis])


def reinhard_tmo(
    image: np.ndarray,
    alpha: float = 0.0,
    white_point: float = 0.0,
    primaries: str = "rec709",
) -> tuple[np.ndarray, float, float]:
    """Global Reinhard tone mapping, equivalent to ReinhardTMO(..., 'global')."""
    image = np.asarray(image, dtype=np.float64)
    if np.any(image < 0.0):
        raise ValueError("Reinhard TMO requires non-negative image values")

    luma = luminance(image, primaries)
    alpha = alpha if alpha > 0.0 else _reinhard_alpha(luma)
    white_point = (
        white_point if white_point > 0.0 else _reinhard_white_point(luma)
    )
    average = log_mean(luma)
    if average <= 0.0:
        return np.zeros_like(image), alpha, white_point

    scaled = alpha * luma / average
    display_luma = scaled * (1.0 + scaled / white_point**2) / (1.0 + scaled)
    return change_luminance(image, luma, display_luma), alpha, white_point


def gamma_tmo(
    image: np.ndarray,
    gamma: float = 2.2,
    fstop: float = 0.0,
) -> np.ndarray:
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if np.any(image < 0.0):
        raise ValueError("Gamma TMO requires non-negative image values")
    exposure = 2.0**fstop
    return np.clip(np.power(exposure * image, 1.0 / gamma), 0.0, 1.0)


def color_correction(
    image: np.ndarray,
    saturation: float = 0.5,
    primaries: str = "rec709",
) -> np.ndarray:
    if saturation < 0.0:
        saturation = 0.5
    image = np.asarray(image, dtype=np.float64)
    luma = luminance(image, primaries)
    ratio = np.divide(
        image,
        luma[:, :, np.newaxis],
        out=np.zeros_like(image),
        where=luma[:, :, np.newaxis] != 0.0,
    )
    return remove_specials(np.power(np.maximum(ratio, 0.0), saturation) * luma[:, :, None])


GammaTMO = gamma_tmo
ReinhardTMO = reinhard_tmo
ColorCorrection = color_correction
