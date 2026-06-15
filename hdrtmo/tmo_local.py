from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import brentq

from .analysis import log_mean
from .colorspace import (
    convert_ipt_to_ich,
    convert_rgb_to_xyz,
    convert_xyz_to_ipt,
    luminance,
    scotopic_luminance,
)
from .core import matlab_percentile, remove_specials
from .filters import bilateral_filter
from .pyramids import (
    gaussian_pyramid,
    laplacian_pyramid,
    pyramid_add,
    pyramid_blend,
    pyramid_multiply,
    reconstruct_pyramid,
)
from .stacks import create_ldr_stack_from_hdr
from .tmo import drago_tmo, ferwerda_tmo, tumblin_tmo, ward_global_tmo
from .tmo_local_utils import (
    ashikhmin_filtering,
    bleaching_parameters,
    chiu_glare,
    create_segments,
    histogram_ceiling,
    krawczyk_image_partition,
    krawczyk_kmeans,
    krawczyk_max_distance,
    lischinski_minimization,
    saturation_parameters,
    sigmoid_color_response,
    sigmoid_response,
    tvi_ashikhmin,
    yee_pattanaik_luminance_adaptation,
    ciecam02_chromatic_adaptation,
    ciecam02_f_l,
    kuang_gamma,
    kuang_normalized_gamma,
    stevenson_detail_enhancement,
)
from .tmo_utils import (
    change_luminance,
    mertens_contrast,
    mertens_saturation,
    mertens_well_exposedness,
    reinhard_alpha,
)
from .utilities import bilateral_separation


def ashikhmin_tmo(image: np.ndarray, display_max: float = 100.0, local: bool = True) -> np.ndarray:
    source = np.asarray(image, dtype=np.float64)
    lum = luminance(source)
    base, detail = ashikhmin_filtering(lum) if local else (lum, np.ones_like(lum))
    low, high = matlab_percentile(base, 0.0005), matlab_percentile(base, 0.9995)
    denominator = tvi_ashikhmin(high) - tvi_ashikhmin(low)
    display = (display_max / 100.0) * (tvi_ashikhmin(base) - tvi_ashikhmin(low)) / max(float(denominator), 1e-12)
    return change_luminance(source, lum, np.maximum(display * detail, 0))


def chiu_tmo(
    image: np.ndarray,
    scale: float = 8.0,
    sigma: float | None = None,
    clamping_iterations: int = 500,
    glare_options: tuple[float, float, int] = (0.8, 8.0, 121),
) -> np.ndarray:
    source = np.asarray(image, dtype=np.float64)
    lum = luminance(source)
    sigma = round(16 * max(lum.shape) / 1024) + 1 if sigma is None or sigma <= 0 else sigma
    filtered = max(scale, 1e-12) * ndimage.gaussian_filter(lum, sigma / 5, mode="nearest")
    scaling = np.divide(1.0, filtered, out=np.zeros_like(lum), where=filtered != 0)
    if clamping_iterations > 0:
        inverse_lum = np.divide(1.0, lum, out=np.zeros_like(lum), where=lum != 0)
        scaling = np.minimum(scaling, inverse_lum)
        kernel = np.array([[0.080, 0.113, 0.080], [0.113, 0.227, 0.113], [0.080, 0.113, 0.080]])
        for _ in range(clamping_iterations):
            scaling = ndimage.convolve(scaling, kernel, mode="nearest")
    return change_luminance(source, lum, chiu_glare(lum * scaling, glare_options))


def durand_tmo(image: np.ndarray, target_contrast: float = 5.0, filter_type: str = "approx_importance") -> np.ndarray:
    del filter_type
    source = np.asarray(image, dtype=np.float64)
    lum = luminance(source)
    base, detail = bilateral_separation(lum, max(lum.shape) * 0.02, 0.4, "log10")
    log_base = np.log10(base + 1e-6)
    factor = np.log10(target_contrast) / max(float(np.ptp(log_base)), 1e-12)
    display = np.maximum(10 ** (log_base * factor + np.log10(np.maximum(detail, 1e-12)) - factor * np.max(log_base)) - 1e-6, 0)
    return change_luminance(source, lum, display)


def kim_kautz_consistent_tmo(
    image: np.ndarray,
    display_max: float = 300.0,
    display_min: float = 0.3,
    c1: float = 3.0,
    c2: float = 0.5,
) -> np.ndarray:
    source = np.asarray(image, dtype=np.float64)
    lum = luminance(source)
    log_l = np.log(lum + 1e-6)
    mean = np.mean(log_l)
    span = max(float(np.ptp(log_l)), 1e-12)
    k1 = (np.log(display_max) - np.log(display_min)) / span
    sigma = span / c1
    weight = np.exp(-((log_l - mean) ** 2) / max(2 * sigma**2, 1e-12))
    display = np.exp(c2 * ((1 - k1) * weight + k1) * (log_l - mean) + mean)
    low, high = matlab_percentile(display, 0.01), matlab_percentile(display, 0.99)
    display = np.clip((display - low) / max(high - low, 1e-12), 0, 1)
    return remove_specials(change_luminance(source, lum, display))


def lischinski_tmo(image: np.ndarray, alpha: float | None = None) -> np.ndarray:
    source = np.asarray(image, dtype=np.float64)
    lum = luminance(source)
    alpha = reinhard_alpha(lum) if alpha is None else alpha
    epsilon = 1e-4
    low, high = np.log2(np.min(lum) + epsilon), np.log2(np.max(lum) + epsilon)
    fstop = np.zeros_like(lum)
    average = log_mean(lum)
    for zone in range(1, max(1, int(np.ceil(high - low))) + 1):
        mask = (lum >= 2 ** (zone - 1 + low)) & (lum < 2 ** (zone + low))
        if np.any(mask):
            representative = matlab_percentile(lum[mask], 0.75)
            scaled = alpha * representative / average
            fstop[mask] = np.log2((scaled / (scaled + 1.0)) / representative)
    smooth, _ = lischinski_minimization(np.log2(lum + epsilon), fstop, np.full_like(lum, 0.07))
    return source * np.power(2.0, smooth)[..., None] if source.ndim == 3 else source * np.power(2.0, smooth)


def pattanaik_tmo(image: np.ndarray, rod_goal: float = 80.0, cone_goal: float = 80.0) -> np.ndarray:
    source = np.asarray(image, dtype=np.float64)
    color = np.repeat(source[..., None], 3, axis=2) if source.ndim == 2 else source
    xyz = convert_rgb_to_xyz(color)
    cone_l = xyz[..., 1]
    rod_l = scotopic_luminance(xyz)
    b_cone, b_rod = bleaching_parameters(cone_goal, rod_goal)
    sigma_cone, sigma_rod = saturation_parameters(cone_goal, rod_goal)
    cone = sigmoid_response(cone_l, 0.73, sigma_cone, b_cone)
    rod = sigmoid_response(rod_l, 0.73, sigma_rod, b_rod)
    exponent = sigmoid_color_response(cone_l, 0.73, sigma_cone, b_cone)
    chroma = remove_specials(np.power(np.maximum(color / np.maximum(cone_l[..., None], 1e-12), 0), exponent[..., None]))
    output = chroma * (cone + rod)[..., None]
    return output[..., 0] if source.ndim == 2 else output


def kuang_tmo(
    image: np.ndarray,
    calibration: str = "unknown",
    p: float = 0.75,
    surround: str = "average",
    bilateral_type: str = "approx_importance",
) -> np.ndarray:
    del bilateral_type
    source = np.asarray(image, dtype=np.float64)
    p = float(np.clip(p, 0.6, 0.85))
    if calibration == "unknown":
        maximum = float(np.max(luminance(source)))
        source = source / maximum * 2e4 if maximum > 0 else source
    xyz = convert_rgb_to_xyz(source)
    base, detail = bilateral_separation(xyz, min(source.shape[:2]) * 0.02, 0.35, "log10")
    white = ndimage.gaussian_filter(xyz, (max(source.shape[:2]) / 10, max(source.shape[:2]) / 10, 0), mode="nearest")
    adapted = ciecam02_chromatic_adaptation(base, np.mean(white, axis=(0, 1)))
    y_white = np.maximum(white[..., 1], 1e-12)
    f_l = ciecam02_f_l(0.2 * y_white)
    compressed = np.power(np.maximum(f_l[..., None] * adapted / y_white[..., None], 0), p)
    compressed = 400 * compressed / (27.13 + compressed) + 0.1
    reconstructed = compressed * stevenson_detail_enhancement(np.maximum(detail, 0), f_l[..., None])
    ipt = convert_xyz_to_ipt(reconstructed)
    ich = convert_ipt_to_ich(ipt)
    chroma = ich[..., 1]
    scale = np.power(f_l + 1, 0.2) * (1.29 * chroma**2 - 0.27 * chroma + 0.42) / np.maximum(chroma**2 - 0.31 * chroma + 0.42, 1e-12)
    ich[..., 1] *= scale
    ipt = convert_ipt_to_ich(ich, True)
    ipt[..., 0] = kuang_normalized_gamma(ipt[..., 0], kuang_gamma(surround))
    output = convert_rgb_to_xyz(convert_xyz_to_ipt(ipt, True), True)
    low, high = matlab_percentile(output, 0.01), matlab_percentile(output, 0.99)
    return np.clip((output - low) / max(high - low, 1e-12), 0, 1)


def ward_hist_adj_tmo(
    image: np.ndarray,
    bins: int = 100,
    display_min: float = 1.0,
    display_max: float = 100.0,
    plot_histogram: bool = False,
    downsampling: bool = False,
) -> np.ndarray:
    del plot_histogram
    source = np.asarray(image, dtype=np.float64)
    lum = luminance(source)
    sample = lum + 1e-6
    if downsampling and min(lum.shape) > 64:
        scale = 64 / min(lum.shape)
        sample = cv2.resize(sample, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    log_sample = np.log(sample)
    low, high = float(np.min(log_sample)), float(np.max(log_sample))
    hist, edges = np.histogram(log_sample, bins=max(1, bins), range=(low, high))
    delta = (high - low) / max(bins, 1)
    hist = histogram_ceiling(hist, delta / max(np.log(display_max + 1e-6) - np.log(display_min + 1e-6), 1e-12))
    cumulative = np.cumsum(hist) / max(float(np.sum(hist)), 1e-12)
    centers = (edges[:-1] + edges[1:]) / 2
    probability = np.interp(np.log(np.minimum(lum + 1e-6, np.exp(high))), centers, cumulative)
    display = np.exp(np.log(display_min + 1e-6) + (np.log(display_max + 1e-6) - np.log(display_min + 1e-6)) * probability)
    display = (display - display_min) / max(display_max - display_min, 1e-12)
    return change_luminance(source, lum, display)


def krawczyk_tmo(image: np.ndarray) -> np.ndarray:
    source = np.asarray(image, dtype=np.float64)
    lum = luminance(source)
    log_l = np.log10(lum + 1e-6)
    hist, edges = np.histogram(log_l, bins=256)
    bound = np.array([edges[0], edges[-1]])
    centers, totals = krawczyk_kmeans(bound, hist)
    framework, _, centers = krawczyk_image_partition(centers, log_l, bound, totals)
    sigma = max(krawczyk_max_distance(centers, bound), 1e-12)
    probabilities, normalization = [], np.zeros_like(lum)
    articulation = []
    for index, center in enumerate(centers):
        mask = framework == index + 1
        values = log_l[mask]
        a = 1 - np.exp(-(float(np.ptp(values)) if values.size else 0) ** 2 / (2 * 0.33**2))
        probability = np.exp(-((center - log_l) ** 2) / (2 * sigma**2))
        probability = bilateral_filter(probability, sigma_spatial=max(min(lum.shape) / 2, 1), sigma_range=0.4)
        probabilities.append(probability)
        articulation.append(a)
        normalization += probability * a
    mapped = log_l.copy()
    for index, probability in enumerate(probabilities):
        mask = framework == index + 1
        if np.any(mask):
            normalized = remove_specials(probability * articulation[index] / normalization)
            mapped -= matlab_percentile(log_l[mask], 0.95) * normalized
    display = np.power(10.0, np.clip(mapped, -2, 0) + 2) / 100
    return change_luminance(source, lum, display)


def _normalize_stack(stack: np.ndarray) -> np.ndarray:
    stack = np.asarray(stack)
    if stack.dtype == np.uint8:
        return stack.astype(np.float64) / 255
    if stack.dtype == np.uint16:
        return stack.astype(np.float64) / 65535
    return stack.astype(np.float64)


def _stack_from_inputs(image: np.ndarray | None, image_stack: np.ndarray | None, sampling: str) -> np.ndarray:
    if image is not None and np.asarray(image).size:
        stack, _ = create_ldr_stack_from_hdr(np.asarray(image), sampling_mode=sampling)
        return stack
    if image_stack is None:
        raise ValueError("image or image_stack is required")
    return _normalize_stack(image_stack)


def mertens_tmo(
    image: np.ndarray | None = None,
    image_stack: np.ndarray | None = None,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    warning: bool = True,
) -> np.ndarray:
    del warning
    stack = _stack_from_inputs(image, image_stack, "zone")
    well_exposed, saturation, contrast = weights
    weight_maps = np.ones((*stack.shape[:2], stack.shape[3]), dtype=np.float64)
    for index in range(stack.shape[3]):
        frame = stack[..., index]
        if well_exposed > 0:
            weight_maps[..., index] *= mertens_well_exposedness(frame) ** well_exposed
        if saturation > 0:
            weight_maps[..., index] *= mertens_saturation(frame) ** saturation
        if contrast > 0:
            weight_maps[..., index] *= mertens_contrast(np.mean(frame, axis=2)) ** contrast
        weight_maps[..., index] += 1e-12
    weight_maps /= np.sum(weight_maps, axis=2, keepdims=True)
    output = np.zeros(stack.shape[:3], dtype=np.float64)
    for channel in range(stack.shape[2]):
        accumulated = None
        for index in range(stack.shape[3]):
            weighted = pyramid_multiply(
                laplacian_pyramid(stack[..., channel, index]),
                gaussian_pyramid(weight_maps[..., index]),
            )
            accumulated = weighted if accumulated is None else pyramid_add(accumulated, weighted)
        output[..., channel] = reconstruct_pyramid(accumulated)
    low, high = float(np.min(output)), float(np.max(output))
    return np.clip((output - low) / max(high - low, 1e-12), 0, 1)


def raman_tmo(image: np.ndarray | None = None, image_stack: np.ndarray | None = None) -> np.ndarray:
    stack = _stack_from_inputs(image, image_stack, "uniform")
    sigma_s = min(stack.shape[:2])
    sigma_r = max(float(np.ptp(stack)) / 10, 1e-12)
    weights = np.empty((*stack.shape[:2], stack.shape[3]))
    for index in range(stack.shape[3]):
        lum = luminance(stack[..., index])
        weights[..., index] = 70 / 255 + np.abs(lum - bilateral_filter(lum, sigma_spatial=sigma_s, sigma_range=sigma_r))
    weights /= np.maximum(np.sum(weights, axis=2, keepdims=True), 1e-12)
    return np.clip(np.sum(stack * weights[:, :, None, :], axis=3), 0, 1)


def bruce_expo_blend_tmo(
    image: np.ndarray | None = None,
    image_stack: np.ndarray | None = None,
    radius: int = 29,
    beta: float = 6.0,
) -> np.ndarray:
    stack = _stack_from_inputs(image, image_stack, "uniform")
    entropy = np.empty((*stack.shape[:2], stack.shape[3]))
    window = 2 * radius + 1
    for index in range(stack.shape[3]):
        gray = luminance(np.log1p(stack[..., index]))
        mean = ndimage.uniform_filter(gray, window, mode="nearest")
        variance = np.maximum(ndimage.uniform_filter(gray**2, window, mode="nearest") - mean**2, 0)
        entropy[..., index] = np.log1p(variance)
    normalized = entropy / np.maximum(np.sum(entropy, axis=2, keepdims=True), 1e-12)
    weights = np.exp(beta * normalized)
    weights /= np.maximum(np.sum(weights, axis=2, keepdims=True), 1e-12)
    output = np.exp(np.sum(np.log1p(stack) * weights[:, :, None, :], axis=3))
    return np.clip((output - np.min(output)) / max(float(np.ptp(output)), 1e-12), 0, 1)


def banterle_tmo(
    image: np.ndarray,
    segments: np.ndarray | None = None,
    rescale: bool = True,
) -> tuple[np.ndarray, np.ndarray, str]:
    source = np.asarray(image, dtype=np.float64)
    if rescale:
        maximum = matlab_percentile(luminance(source), 0.999)
        source = np.clip(source / maximum * 3000, 0.015, 3000) if maximum > 0 else source
    segments = create_segments(source) if segments is None else np.round(segments)
    mask = np.zeros(segments.shape)
    for zone, operator in zip([-2, -1, 0, 1, 2, 3, 4], [0, 0, 1, 0, 1, 0, 0]):
        mask[segments == zone] = operator
    drago = drago_tmo(source)
    lum = luminance(source)
    scaled = 0.5 * lum / max(log_mean(lum), 1e-12)
    reinhard = change_luminance(source, lum, scaled / (1 + scaled))
    if np.all(mask == 0):
        return drago, segments, "DragoTMO"
    if np.all(mask == 1):
        return reinhard, segments, "ReinhardTMO"
    return np.clip(pyramid_blend(np.power(reinhard, 1 / 2.2), np.power(drago, 1 / 2.2), mask), 0, 1) ** 2.2, segments, "BanterleTMO"


def van_hateren_tmo(image: np.ndarray, pupil_area: float = 10.0, warning: bool = True) -> np.ndarray:
    del warning
    source = np.asarray(image, dtype=np.float64)
    lum = luminance(source)
    a_c, c_beta, k_beta = 9e-2, 2.8e-3, 1.6e-4
    maximum = brentq(lambda x: a_c * x**5 + x**4 - 1 / c_beta, 0, 100)
    root_upper = maximum * (1 + 1e-10)
    values = lum * max(pupil_area, 1e-12)
    ios = np.array(
        [
            brentq(
                lambda x, value=v: a_c * x**5 + x**4 - 1 / (c_beta + k_beta * value),
                0,
                root_upper,
            )
            for v in values.ravel()
        ]
    ).reshape(values.shape)
    return change_luminance(source, lum, np.clip(1 - ios / maximum, 0, 1))


def yp_ferwerda_tmo(image: np.ndarray, display_max: float = 100.0, display_adaptation: float | None = None, max_layers: int = 32) -> tuple[np.ndarray, np.ndarray]:
    adaptation = yee_pattanaik_luminance_adaptation(image, max_layers)
    display_adaptation = display_max / 2 if display_adaptation is None else display_adaptation

    def sensitivity(value: np.ndarray, cone: bool) -> np.ndarray:
        log_value = np.log10(np.maximum(value, 1e-12))
        exponent = np.empty_like(log_value)
        if cone:
            low, high = log_value <= -2.6, log_value >= 1.9
            middle = ~(low | high)
            exponent[low] = -0.72
            exponent[high] = log_value[high] - 1.255
            exponent[middle] = (0.249 * log_value[middle] + 0.65) ** 2.7 - 0.72
        else:
            low, high = log_value <= -3.94, log_value >= -1.44
            middle = ~(low | high)
            exponent[low] = -2.86
            exponent[high] = log_value[high] - 0.395
            exponent[middle] = (0.405 * log_value[middle] + 1.6) ** 2.18 - 2.86
        return np.power(10.0, exponent)

    lum = luminance(image)
    cone_scale = sensitivity(np.asarray(display_adaptation), True) / sensitivity(adaptation, True)
    rod_scale = sensitivity(np.asarray(display_adaptation), False) / sensitivity(adaptation, False)
    k = np.maximum((100 - adaptation / 4) / (100 + adaptation), 0)
    source = np.asarray(image, dtype=np.float64)
    if source.ndim == 2:
        output = cone_scale * source + rod_scale * k * lum
    else:
        weights = np.array([1.05, 0.97, 1.27])
        output = cone_scale[..., None] * source + weights * (rod_scale * k * lum)[..., None]
    return np.clip(output / display_max, 0, 1), adaptation


def yp_tumblin_tmo(image: np.ndarray, display_adaptation: float = 20.0, display_max: float = 100.0, display_contrast: float = 100.0, max_layers: int = 32) -> tuple[np.ndarray, np.ndarray]:
    adaptation = yee_pattanaik_luminance_adaptation(image, max_layers)
    return tumblin_tmo(image, display_adaptation, display_max, display_contrast, adaptation), adaptation


def yp_ward_global_tmo(image: np.ndarray, display_max: float = 100.0, max_layers: int = 32) -> tuple[np.ndarray, np.ndarray]:
    adaptation = yee_pattanaik_luminance_adaptation(image, max_layers)
    scale = ((1.219 + (display_max / 2) ** 0.4) / (1.219 + adaptation**0.4)) ** 2.5
    source = np.asarray(image, dtype=np.float64)
    output = source * (scale[..., None] if source.ndim == 3 else scale) / display_max
    return np.clip(output, 0, 1), adaptation


AshikhminTMO = ashikhmin_tmo
BanterleTMO = banterle_tmo
BruceExpoBlendTMO = bruce_expo_blend_tmo
ChiuTMO = chiu_tmo
DurandTMO = durand_tmo
KimKautzConsistentTMO = kim_kautz_consistent_tmo
KrawczykTMO = krawczyk_tmo
LischinskiTMO = lischinski_tmo
MertensTMO = mertens_tmo
PattanaikTMO = pattanaik_tmo
KuangTMO = kuang_tmo
RamanTMO = raman_tmo
VanHaterenTMO = van_hateren_tmo
WardHistAdjTMO = ward_hist_adj_tmo
YPFerwerdaTMO = yp_ferwerda_tmo
YPTumblinTMO = yp_tumblin_tmo
YPWardGlobalTMO = yp_ward_global_tmo
