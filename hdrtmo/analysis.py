from __future__ import annotations

import numpy as np

from .colorspace import luminance
from .core import matlab_percentile


def log_mean(image: np.ndarray, delta: float = 1e-6) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(image) + delta))))


def image_key(image: np.ndarray, robust: float = 0.01) -> tuple[float, float]:
    luma = luminance(image)
    average = log_mean(luma)
    positive = luma[luma > 0]
    if robust > 0 and positive.size:
        minimum = matlab_percentile(positive, robust)
        maximum = matlab_percentile(positive, 1 - robust)
    elif positive.size:
        minimum, maximum = float(np.min(luma)), float(np.max(luma))
    else:
        minimum = maximum = 1.0
    log_min, log_max = np.log(minimum), np.log(maximum)
    span = log_max - log_min
    key = (np.log(average) - log_min) / span if span > 0 and np.isfinite(span) else -1.0
    return float(key), average


def dynamic_range(
    image: np.ndarray,
    robust: float = 0.0,
    kind: str = "Classic",
) -> np.ndarray | float:
    luma = luminance(image)
    if robust >= 0.5:
        robust = 0.01
    minimum = matlab_percentile(luma, robust) if robust > 0 else float(np.min(luma))
    maximum = matlab_percentile(luma, 1 - robust) if robust > 0 else float(np.max(luma))
    if minimum <= 0:
        positive = luma[luma > 0]
        if not positive.size:
            return np.array([np.inf, np.inf, np.inf])
        minimum = float(np.min(positive))
    if kind == "Classic":
        ratio = maximum / minimum
        return np.array([ratio, np.log2(ratio), np.log10(ratio)])
    if kind == "Michelson":
        return (maximum - minimum) / (maximum + minimum)
    if kind == "Weber":
        return (maximum - minimum) / minimum
    raise ValueError("kind must be Classic, Michelson, or Weber")


def get_fstops(image: np.ndarray, percentile: float = 0.001) -> tuple[int, int]:
    if not 0 <= percentile <= 1:
        percentile = 0.001
    luma = luminance(image)
    minimum = matlab_percentile(luma, percentile)
    maximum = matlab_percentile(luma, 1 - percentile)
    return round(np.log2(minimum + 1e-6)), round(np.log2(maximum + 1e-6))


def over_under_exposed(
    image: np.ndarray,
    under_threshold: float = 7 / 255,
    over_threshold: float = 248 / 255,
) -> tuple[np.ndarray, float, float]:
    values = luminance(image)
    mask = np.zeros(values.shape)
    mask[values >= over_threshold] = 1
    mask[values <= under_threshold] = -1
    return mask, float(np.mean(mask > 0.5)), float(np.mean(mask < -0.5))


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Gray-world white balance requires RGB input")
    averages = np.mean(image, axis=(0, 1))
    return image / averages


def absolute_calibration(image: np.ndarray) -> tuple[np.ndarray, float]:
    luma = luminance(image)
    key, _ = image_key(image, 0.001)
    maximum = matlab_percentile(luma, 0.999)
    factor = 1e4 * key / maximum if maximum > 0 and key > 0 else 1.0
    return image * factor, float(factor)


def light_sources_detection(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    luma = luminance(image)
    maximum = matlab_percentile(luma, 0.95)
    minimum = matlab_percentile(luma, 0.05)
    delta = maximum - minimum
    key, _ = image_key(image, 0.05)
    threshold = (0.6 + 0.4 * (1 - key)) * delta + minimum
    mask = (luma >= threshold).astype(np.float64)
    low = max(minimum, threshold - 0.1 * delta)
    high = min(maximum, threshold + 0.1 * delta)
    if high == low:
        return mask, np.zeros_like(luma)
    scale = (luma - threshold + 0.1 * delta) / (high - low)
    smooth = 1 - 3 * scale**2 + 2 * scale**3
    smooth[luma < low] = 1
    smooth[luma > high] = 0
    return mask, smooth


def image_num_pixels(image: np.ndarray) -> int:
    return int(image.shape[0] * image.shape[1])


def histogram_hdr(
    image: np.ndarray,
    bins: int = 256,
    log_type: str = "log10",
    bounds: tuple[float, float] | None = None,
    normalized: int = 0,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, tuple[float, float], float, np.ndarray]:
    original = luminance(image).reshape(-1)
    values = original.copy()
    transforms = {
        "linear": lambda value: value,
        "log2": lambda value: np.log2(value + epsilon),
        "loge": lambda value: np.log(value + epsilon),
        "log10": lambda value: np.log10(value + epsilon),
    }
    if log_type not in transforms:
        raise ValueError("log_type must be linear, log2, loge, or log10")
    values = transforms[log_type](values)
    minimum, maximum = bounds or (float(np.min(values)), float(np.max(values)))
    step = (maximum - minimum) / (bins - 1)
    histogram = np.zeros(bins)
    weighted_average = 0.0
    total = 0
    for index in range(bins):
        selected = (values >= step * index + minimum) & (
            values < step * (index + 1) + minimum
        )
        count = int(np.sum(selected))
        if count:
            histogram[index] = count
            weighted_average += matlab_percentile(original[selected], 0.5) * count
            total += count
    if normalized:
        norm = np.max(histogram) if normalized == 2 else np.sum(histogram)
        if norm > 0:
            histogram /= norm
    average = weighted_average / total if total else np.nan
    x = (np.arange(1, bins + 1) / bins) * (maximum - minimum) + minimum
    return histogram, (minimum, maximum), float(average), x


imKey = image_key
imDynamicRange = dynamic_range
getFstops = get_fstops
getOverUnderExposedParts = over_under_exposed
computeWhiteBalanceGrayWorld = gray_world_white_balance
AkyuzAbsoluteCalibration = absolute_calibration
AkyuzLightSourcesDetection = light_sources_detection
imNumPixels = image_num_pixels
HistogramHDR = histogram_hdr
MaxQuart = matlab_percentile
