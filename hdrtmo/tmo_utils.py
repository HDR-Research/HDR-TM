from __future__ import annotations

import numpy as np
from scipy import ndimage

from .analysis import histogram_hdr, log_mean
from .colorspace import luminance
from .core import matlab_percentile, remove_specials


def change_luminance(
    image: np.ndarray,
    old_luminance: np.ndarray,
    new_luminance: np.ndarray,
    epsilon: bool = False,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    old = np.asarray(old_luminance, dtype=np.float64) + (1e-6 if epsilon else 0.0)
    new = np.asarray(new_luminance, dtype=np.float64)
    if image.ndim == 2:
        return remove_specials(new)
    if new.ndim == 3:
        new = new if new.shape[2] == image.shape[2] else luminance(new)
    if new.ndim == 2:
        new = new[..., None]
    return remove_specials(
        np.divide(
            image * new,
            old[..., None],
            out=np.zeros_like(image),
            where=old[..., None] != 0,
        )
    )


def reinhard_alpha(luma: np.ndarray, delta: float = 1e-6) -> float:
    minimum = matlab_percentile(luma, 0.01)
    maximum = matlab_percentile(luma, 0.99)
    log_min = np.log2(minimum + delta)
    log_max = np.log2(maximum + delta)
    span = log_max - log_min
    if abs(span) <= np.finfo(float).eps:
        return 0.18
    average = np.log2(log_mean(luma, delta) + delta)
    return float(0.18 * 4 ** ((2 * average - log_min - log_max) / span))


def reinhard_white_point(luma: np.ndarray) -> float:
    positive = np.asarray(luma)[np.asarray(luma) > 0]
    if not positive.size:
        return 1.0
    log_min = np.log2(np.min(positive) + 1e-6)
    log_max = np.log2(np.max(positive) + 1e-6)
    return float(1.5 * 2 ** (log_max - log_min - 5))


def gamma_drago(
    image: np.ndarray,
    gamma: float = 2.2,
    slope: float = 4.5,
    start: float = 0.018,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    return np.clip(
        np.where(
            image <= start,
            image * slope,
            np.power(image, 0.9 / gamma) * 1.099 - 0.099,
        ),
        0.0,
        1.0,
    )


def stevens_csf(values: np.ndarray | float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(
        values <= 100,
        1.855 + 0.4 * np.log10(values + 2.3e-5),
        2.655,
    )


def tp_ferwerda(value: float) -> float:
    t = np.log10(value)
    if t <= -2.6:
        result = -0.72
    elif t >= 1.9:
        result = t - 1.255
    else:
        result = (0.249 * t + 0.65) ** 2.7 - 0.72
    return float(10**result)


def ts_ferwerda(value: float) -> float:
    t = np.log10(value)
    if t <= -3.94:
        result = -2.86
    elif t >= -1.44:
        result = t - 0.395
    else:
        result = (0.405 * t + 1.6) ** 2.18 - 2.86
    return float(10**result)


def ferwerda_k(world_adaptation: float) -> float:
    return float(np.clip((1 - (world_adaptation / 2 - 0.01) / (10 - 0.01)) ** 2, 0, 1))


def walraven_valeton_k(
    world_adaptation: np.ndarray | float,
    sigma: float = 100.0,
) -> np.ndarray:
    return np.maximum((sigma - np.asarray(world_adaptation) / 4) / (sigma + world_adaptation), 0)


def exposure_histogram_sampling(
    image: np.ndarray,
    bit_depth: int = 8,
    overlap: float = 0.0,
) -> np.ndarray:
    bit_depth = 8 if bit_depth < 1 else bit_depth
    half = round(bit_depth / 2)
    histogram, bounds, _, _ = histogram_hdr(image, 4096, "log2")
    step = (bounds[1] - bounds[0]) / 4096
    if step <= 0:
        return np.array([0.0])
    overlap = 0.0 if overlap > half else overlap
    removal = max(bit_depth / step + overlap, 2)
    radius = round(removal / 2)
    stops = []
    while np.sum(histogram) > 0:
        best_total, best_index, best_bounds = -1.0, -1, (0, 0)
        for index in range(4096):
            low = max(0, index - radius)
            high = min(index + radius + 1, 4096)
            total = np.sum(histogram[low:high])
            if total > best_total:
                best_total, best_index, best_bounds = total, index, (low, high)
        histogram[best_bounds[0] : best_bounds[1]] = 0
        stops.append(-((best_index + 1) * step + bounds[0]) - half)
    return np.asarray(stops)


logMean = log_mean
ChangeLuminance = change_luminance
ReinhardAlpha = reinhard_alpha
ReinhardWhitePoint = reinhard_white_point
GammaDrago = gamma_drago
StevensCSF = stevens_csf
TpFerwerda = tp_ferwerda
TsFerwerda = ts_ferwerda
Ferwerda_k = ferwerda_k
WalravenValeton_k = walraven_valeton_k
ExposureHistogramSampling = exposure_histogram_sampling


def mertens_contrast(luma: np.ndarray) -> np.ndarray:
    kernel = np.asarray(((0, 1, 0), (1, -4, 1), (0, 1, 0)))
    return np.abs(ndimage.convolve(np.asarray(luma, dtype=np.float64), kernel, mode="nearest"))


def mertens_saturation(image: np.ndarray) -> np.ndarray:
    source = np.asarray(image, dtype=np.float64)
    return np.ones(source.shape[:2]) if source.ndim == 2 or source.shape[2] == 1 else np.std(source, axis=2)


def mertens_well_exposedness(
    image: np.ndarray,
    mean: float = 0.5,
    sigma: float = 0.2,
) -> np.ndarray:
    source = np.asarray(image, dtype=np.float64)
    if source.ndim == 2:
        source = source[..., None]
    return np.prod(np.exp(-((source - mean) ** 2) / (2 * sigma**2)), axis=2)


MertensContrast = mertens_contrast
MertensSaturation = mertens_saturation
MertensWellExposedness = mertens_well_exposedness
