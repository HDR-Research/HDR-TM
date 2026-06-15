from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from .analysis import image_key, image_num_pixels, log_mean
from .colorspace import luminance
from .core import box_filter, check_in_unit_interval, remove_specials
from .filters import bilateral_filter
from .tmo_utils import change_luminance


def _linearize(image: np.ndarray, gamma_removal: float) -> np.ndarray:
    check_in_unit_interval(image)
    return image**gamma_removal if gamma_removal > 0 else image.copy()


def akyuz_eo(
    image: np.ndarray,
    max_output: float,
    gamma: float,
    gamma_removal: float,
) -> np.ndarray:
    if max_output <= 0 or gamma <= 0:
        raise ValueError("max_output and gamma must be positive")
    image = _linearize(image, gamma_removal)
    luma = luminance(image)
    minimum, maximum = np.min(luma), np.max(luma)
    expanded = max_output * ((luma - minimum) / (maximum - minimum)) ** gamma
    return change_luminance(image, luma, expanded)


def landis_eo(
    image: np.ndarray,
    alpha: float,
    threshold: float,
    max_output: float,
    gamma_removal: float,
) -> np.ndarray:
    if max_output < 0:
        raise ValueError("max_output must be non-negative")
    image = _linearize(image, gamma_removal)
    alpha = 2.0 if alpha <= 0 else alpha
    luma = luminance(image)
    threshold = np.mean(luma) if threshold <= 0 else threshold
    maximum = np.max(luma)
    expanded = luma.copy()
    selected = luma >= threshold
    denominator = maximum - threshold
    weights = (
        ((luma[selected] - threshold) / denominator) ** alpha
        if denominator > 0
        else np.ones(np.sum(selected))
    )
    expanded[selected] = (
        luma[selected] * (1 - weights) + max_output * luma[selected] * weights
    )
    return change_luminance(image, luma, expanded)


def masia_eo(
    image: np.ndarray,
    max_output: float,
    noise_removal: bool = True,
    multi_regression: bool = False,
    gamma_removal: float = 2.2,
) -> tuple[np.ndarray, bool]:
    if max_output < 0:
        raise ValueError("max_output must be non-negative")
    image = _linearize(image, gamma_removal)
    luma = luminance(image)
    key, average = image_key(image)
    if not multi_regression:
        gamma = key * 10.44 - 6.282
    else:
        overexposed = np.mean(luma * 255 >= 254) * 100
        gamma = 2.4379 + 0.2319 * np.log(average) - 1.1228 * key + 0.0085 * overexposed
    if noise_removal:
        base = bilateral_filter(luma, None, 0, 1, 32, 0.01)
        detail = remove_specials(luma / base)
        expanded = detail * base**gamma
    else:
        expanded = luma**gamma
    return change_luminance(image, luma, expanded * max_output), bool(gamma <= 0)


def huo_eo(
    image: np.ndarray,
    scale: float = 1.6,
    theta: float = 1e-5,
    gamma_removal: float = 2.2,
) -> np.ndarray:
    image = _linearize(image, gamma_removal)
    luma = luminance(image)
    average = np.mean(luma)
    mapped = 10 ** (-scale) * luma / (average * (1 - luma + theta))
    local = bilateral_filter(luma, None, 0, 1, 16, 3 / 255)
    expansion = remove_specials(mapped / local)
    return image * (expansion[..., None] if image.ndim == 3 else expansion)


def huo_physiological_eo(
    image: np.ndarray,
    max_output: float,
    exponent: float = 0.86,
    gamma_removal: float = 2.2,
) -> np.ndarray:
    if max_output < 0:
        raise ValueError("max_output must be non-negative")
    image = _linearize(image, gamma_removal)
    luma = luminance(image)
    maximum = np.max(luma)
    sigma_l = log_mean(luma)
    first = bilateral_filter(luma, None, 0, 1, 16, 0.3)
    second = bilateral_filter(first, None, 0, 1, 10, 0.1)
    sigma = max_output * sigma_l
    high = max_output * second
    expanded = (luma / maximum) * (high**exponent + sigma**exponent) ** (1 / exponent)
    return change_luminance(image, luma, expanded)


def kuo_expand_map(luma: np.ndarray, gamma_removal: float = -1.0) -> np.ndarray:
    kernel_size = max(int(np.ceil(0.1 * max(luma.shape))), 1)
    filtered = box_filter(luma, kernel_size)
    mask = ndimage.binary_erosion(luma >= np.max(filtered))
    temporary = luma * mask
    sigma_range = (100 / 255) ** gamma_removal if gamma_removal > 0 else 100 / 255
    return bilateral_filter(
        temporary,
        luma,
        0,
        1,
        kernel_size / 5,
        sigma_range,
    )


def kuo_eo(
    image: np.ndarray,
    max_output: float,
    gamma_removal: float,
) -> np.ndarray:
    if max_output < 0:
        raise ValueError("max_output must be non-negative")
    image = _linearize(image, gamma_removal)
    luma = luminance(image)
    expanded = luma * max_output / (30 * (1 - luma) + luma)
    expansion_map = kuo_expand_map(luma, gamma_removal)
    filtered = cv2.blur(expanded.astype(np.float64), (5, 5))
    expanded = filtered * expansion_map + (1 - expansion_map) * expanded
    return change_luminance(image, luma, expanded)


def kovaleski_oliveira_eo(
    image: np.ndarray,
    content_type: str,
    sigma_spatial: float = 150,
    sigma_range: float = 25 / 255,
    display_min: float = 0.3,
    display_max: float = 1200,
    gamma_removal: float = 2.2,
) -> tuple[np.ndarray, np.ndarray]:
    if display_min < 0 or display_max < 0:
        raise ValueError("Display luminance values must be non-negative")
    threshold = 254 / 255 if content_type == "image" else 230 / 255
    image = _linearize(image, gamma_removal)
    if gamma_removal > 0:
        threshold **= gamma_removal
        sigma_range **= gamma_removal
    luma = luminance(image)
    clipped = (np.max(image, axis=2) > threshold).astype(np.float64)
    expansion_map = bilateral_filter(
        clipped,
        luma,
        0,
        1,
        sigma_spatial,
        sigma_range,
    )
    expansion_map = (expansion_map * 3 + 1) / 4
    expanded = (luma * (display_max - display_min) + display_min) * expansion_map
    return change_luminance(image, luma, expanded), expansion_map


AkyuzEO = akyuz_eo
LandisEO = landis_eo
MasiaEO = masia_eo
HuoEO = huo_eo
HuoPhysEO = huo_physiological_eo
KuoExpandMap = kuo_expand_map
KuoEO = kuo_eo
KovaleskiOliveiraEO = kovaleski_oliveira_eo

