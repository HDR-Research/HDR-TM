from __future__ import annotations

import numpy as np

from .analysis import log_mean
from .colorspace import luminance
from .core import matlab_percentile, normalize_image, remove_specials
from .tmo_utils import (
    change_luminance,
    exposure_histogram_sampling,
    reinhard_alpha,
    stevens_csf,
    tp_ferwerda,
    ts_ferwerda,
    walraven_valeton_k,
)


def gamma_tmo(image: np.ndarray, gamma: float = 2.2, fstop: float = 0.0) -> np.ndarray:
    if gamma <= 0 or np.any(image < 0):
        raise ValueError("Gamma must be positive and image must be non-negative")
    return np.clip(np.power((2**fstop) * image, 1 / gamma), 0, 1)


def normalize_tmo(image: np.ndarray, robust: bool = True) -> np.ndarray:
    luma = luminance(image)
    normalized, _, _ = (
        normalize_image(luma)
        if robust
        else normalize_image(luma, float(np.min(luma)), float(np.max(luma)))
    )
    return np.clip(change_luminance(image, luma, normalized), 0, 1)


def exponential_tmo(image: np.ndarray, appearance: float = 1.0) -> np.ndarray:
    appearance = 1.0 if appearance <= 0 else appearance
    luma = luminance(image)
    average = log_mean(luma)
    display = 1 - np.exp(-appearance * luma / average)
    return change_luminance(image, luma, display)


def logarithmic_tmo(
    image: np.ndarray,
    q: float = 1.0,
    k: float = 1.0,
) -> np.ndarray:
    q, k = max(q, 1), max(k, 1)
    luma = luminance(image)
    display = np.log10(1 + luma * q) / np.log10(1 + np.max(luma) * k)
    return change_luminance(image, luma, display)


def schlick_tmo(
    image: np.ndarray,
    mode: str = "manual",
    p: float = 200.0,
    bit_depth: int = 8,
    lowest_display: float = 1.0,
    k: float = 0.5,
) -> np.ndarray:
    luma = luminance(image)
    positive = luma[luma > 0]
    if not positive.size:
        return np.zeros_like(image)
    minimum, maximum = np.min(positive), np.max(positive)
    if mode == "manual":
        p = max(p, 1)
    elif mode in {"automatic", "nonuniform"}:
        p = lowest_display * maximum / (2**bit_depth * minimum)
        if mode == "nonuniform":
            p = p * (1 - k + k * luma / np.sqrt(maximum * minimum))
    else:
        raise ValueError("mode must be manual, automatic, or nonuniform")
    display = p * luma / ((p - 1) * luma + maximum)
    return change_luminance(image, luma, display)


def ward_global_tmo(
    image: np.ndarray,
    display_max: float = 100.0,
    world_adaptation: np.ndarray | float | None = None,
) -> np.ndarray:
    display_max = 100.0 if display_max <= 0 else display_max
    luma = luminance(image)
    adaptation = log_mean(luma) if world_adaptation is None else world_adaptation
    if np.min(adaptation) < 0:
        adaptation = log_mean(luma)
    scale = ((1.219 + (display_max / 2) ** 0.4) / (1.219 + adaptation**0.4)) ** 2.5
    return np.clip(image * scale / display_max, 0, 1)


def drago_tmo(
    image: np.ndarray,
    display_max: float = 100.0,
    bias: float = 0.85,
    world_adaptation: float | None = None,
) -> np.ndarray:
    luma = luminance(image)
    adaptation = log_mean(luma) if world_adaptation is None or world_adaptation < 0 else world_adaptation
    adaptation /= (1 + bias - 0.85) ** 5
    scaled = luma / adaptation
    scaled_max = np.max(luma) / adaptation
    c1 = np.log(bias) / np.log(0.5)
    p1 = (display_max / 100) / np.log10(1 + scaled_max)
    p2 = np.log(1 + scaled) / np.log(2 + 8 * (scaled / scaled_max) ** c1)
    return change_luminance(image, luma, p1 * p2)


def tumblin_tmo(
    image: np.ndarray,
    display_adaptation: float = 20.0,
    display_max: float = 100.0,
    display_contrast: float = 100.0,
    world_adaptation: float | None = None,
) -> np.ndarray:
    luma = luminance(image)
    world = log_mean(luma) if world_adaptation is None else world_adaptation
    gamma_world = stevens_csf(world)
    gamma_display = stevens_csf(display_adaptation)
    gamma_ratio = gamma_world / (1.855 + 0.4 * np.log(display_adaptation))
    multiplier = display_contrast ** ((gamma_ratio - 1) / 2)
    ratio = np.divide(luma, world, out=np.zeros_like(luma, dtype=np.float64), where=np.asarray(world) != 0)
    response = np.zeros_like(ratio)
    np.power(ratio, gamma_world / gamma_display, out=response, where=ratio > 0)
    display = display_adaptation * multiplier * response
    return change_luminance(image, luma, display / display_max)


def reinhard_robust_tmo(
    image: np.ndarray,
    alpha: float | None = None,
    world_adaptation: float | None = None,
) -> tuple[np.ndarray, float, float]:
    luma = luminance(image)
    clamped = np.clip(
        luma,
        matlab_percentile(luma, 0.01),
        matlab_percentile(luma, 0.99),
    )
    world = log_mean(clamped, 1e-5) if world_adaptation is None else world_adaptation
    alpha = reinhard_alpha(clamped, 1e-5) if alpha is None else alpha
    scaled = alpha * clamped / world
    display = scaled / (1 + scaled)
    return change_luminance(image, luma, display), float(alpha), float(world)


def reinhard_devlin_tmo(
    image: np.ndarray,
    contrast: float | None = None,
    intensity: float = 0.0,
    light_adaptation: float = 1.0,
    chromatic_adaptation: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    luma = luminance(image)
    average = log_mean(luma)
    if contrast is None:
        maximum = np.log2(matlab_percentile(luma, 0.99) + 1e-9)
        minimum = np.log2(matlab_percentile(luma, 0.01) + 1e-9)
        span = maximum - minimum
        key = (maximum - np.log2(average + 1e-9)) / span if span else 0.5
        contrast = 0.3 + 0.7 * key**1.4
    contrast = np.clip(contrast, 0.3, 1)
    intensity_factor = np.exp(-np.clip(intensity, -8, 8))
    light_adaptation = np.clip(light_adaptation, 0, 1)
    chromatic_adaptation = np.clip(chromatic_adaptation, 0, 1)
    source = image if image.ndim == 3 else image[..., None]
    output = np.zeros_like(source, dtype=np.float64)
    for channel in range(source.shape[2]):
        channel_values = source[..., channel]
        channel_average = np.mean(channel_values)
        local = chromatic_adaptation * channel_values + (1 - chromatic_adaptation) * luma
        global_value = chromatic_adaptation * channel_average + (1 - chromatic_adaptation) * average
        adaptation = light_adaptation * local + (1 - light_adaptation) * global_value
        denominator = channel_values + (intensity_factor * adaptation) ** contrast
        output[..., channel] = np.divide(
            channel_values,
            denominator,
            out=np.zeros_like(channel_values),
            where=denominator != 0,
        )
    if normalize:
        output_luma = luminance(output if image.ndim == 3 else output[..., 0])
        minimum, maximum = np.min(output_luma), np.max(output_luma)
        if maximum > minimum:
            output = np.clip((output - minimum) / (maximum - minimum), 0, 1)
    output = remove_specials(output)
    return output if image.ndim == 3 else output[..., 0]


def ferwerda_tmo(
    image: np.ndarray,
    display_max: float = 100.0,
    display_adaptation: float | None = None,
    world_adaptation: float | None = None,
) -> np.ndarray:
    display_adaptation = display_max / 2 if display_adaptation is None else display_adaptation
    luma = luminance(image)
    world = np.max(luma) / 2 if world_adaptation is None else world_adaptation
    cone_scale = tp_ferwerda(display_adaptation) / tp_ferwerda(world)
    rod_scale = ts_ferwerda(display_adaptation) / ts_ferwerda(world)
    k = walraven_valeton_k(world)
    source = image if image.ndim == 3 else image[..., None]
    weights = np.array([1.05, 0.97, 1.27]) if source.shape[2] == 3 else np.ones(source.shape[2])
    output = np.empty_like(source, dtype=np.float64)
    for channel in range(source.shape[2]):
        output[..., channel] = cone_scale * source[..., channel] + weights[channel] * rod_scale * k * luma
    output = np.clip(output / display_max, 0, 1)
    return output if image.ndim == 3 else output[..., 0]


def best_exposure_tmo(
    image: np.ndarray,
    method: str = "histogram",
) -> tuple[np.ndarray, float]:
    if method == "histogram":
        stops = exposure_histogram_sampling(image, 8, 0)
        exposure = 2 ** stops[0]
    elif method == "mean":
        mean = np.mean(luminance(image))
        exposure = 1 / (4 * mean) if mean > 0 else 1.0
    else:
        raise ValueError("method must be histogram or mean")
    return np.clip(image * exposure, 0, 1), float(exposure)


def select_overexposed_tmo(
    image: np.ndarray,
    percent: float = 5.0,
) -> tuple[np.ndarray, float]:
    target = percent / 100
    gamma_inverse = 1 / 2.2
    positive = np.asarray(image, dtype=np.float64)
    positive = positive[positive > 0]
    if not positive.size:
        return np.zeros_like(image), 1.0
    threshold_linear = 0.95 ** (1 / gamma_inverse)
    quantile = float(np.quantile(positive, np.clip(1 - target, 0, 1)))
    exposure = threshold_linear / max(quantile, 1e-12)
    return np.clip(image * exposure, 0, 1), exposure


GammaTMO = gamma_tmo
NormalizeTMO = normalize_tmo
ExponentialTMO = exponential_tmo
LogarithmicTMO = logarithmic_tmo
SchlickTMO = schlick_tmo
WardGlobalTMO = ward_global_tmo
DragoTMO = drago_tmo
TumblinTMO = tumblin_tmo
ReinhardRobustTMO = reinhard_robust_tmo
ReinhardDevlinTMO = reinhard_devlin_tmo
FerwerdaTMO = ferwerda_tmo
BestExposureTMO = best_exposure_tmo
SelectOverexposedTMO = select_overexposed_tmo
