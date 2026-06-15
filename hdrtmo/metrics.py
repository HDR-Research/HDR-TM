from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy import ndimage, special, stats

from .colorspace import (
    convert_ipt_to_ich,
    convert_rgb_to_xyz,
    convert_xyz_to_cielab,
    convert_xyz_to_ipt,
    luminance,
)
from .core import normalize_image, same_image


def check_domains(
    reference: np.ndarray,
    distorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    if not same_image(reference, distorted):
        raise ValueError("Images must have the same shape")
    if reference.dtype != distorted.dtype:
        raise ValueError("Images must have the same dtype")
    domain = str(reference.dtype)
    if reference.dtype == np.uint8:
        maximum = 255.0
    elif reference.dtype == np.uint16:
        maximum = 65535.0
    elif np.issubdtype(reference.dtype, np.floating):
        maximum = float(np.max(reference))
    else:
        raise ValueError(f"Unsupported image dtype: {reference.dtype}")
    return reference.astype(np.float64), distorted.astype(np.float64), domain, maximum


def change_comparison_domain(
    image: np.ndarray | float,
    domain: str = "lin",
) -> tuple[np.ndarray, bool]:
    values = np.asarray(image, dtype=np.float64)
    epsilon = 1e-6
    if domain in {"", "lin"}:
        return values, True
    if domain == "log2":
        return np.log2(values + epsilon), False
    if domain == "log":
        return np.log(values + epsilon), False
    if domain == "log10":
        return np.log10(values + epsilon), False
    raise NotImplementedError("PU08/PU21 encoding will be added with the TMQI utility layer")


def mse(
    reference: np.ndarray,
    distorted: np.ndarray,
    negative_check: bool = True,
    comparison_domain: str = "lin",
    mask: np.ndarray | None = None,
) -> float:
    reference, distorted, _, _ = check_domains(reference, distorted)
    if negative_check and (np.any(reference < 0) or np.any(distorted < 0)):
        raise ValueError("Images contain negative values")
    reference, _ = change_comparison_domain(reference, comparison_domain)
    distorted, _ = change_comparison_domain(distorted, comparison_domain)
    squared = (reference - distorted) ** 2
    if mask is None or np.max(mask) <= 0:
        return float(np.mean(squared))
    normalized_mask = np.asarray(mask, dtype=np.float64) / np.max(mask)
    if normalized_mask.shape[:2] != reference.shape[:2]:
        raise ValueError("Mask and image resolutions differ")
    selected = normalized_mask > 0
    if normalized_mask.ndim == 2 and reference.ndim == 3:
        selected = np.repeat(selected[..., None], reference.shape[2], axis=2)
    return float(np.mean(squared[selected]))


def rmse(
    reference: np.ndarray,
    distorted: np.ndarray,
    comparison_domain: str = "lin",
) -> float:
    return float(np.sqrt(mse(reference, distorted, comparison_domain=comparison_domain)))


def psnr(
    reference: np.ndarray,
    distorted: np.ndarray,
    comparison_domain: str = "lin",
    max_value: float | None = None,
    mask: np.ndarray | None = None,
) -> float:
    reference_float, distorted_float, _, domain_max = check_domains(reference, distorted)
    maximum = domain_max if max_value is None else max_value
    reference_domain, _ = change_comparison_domain(reference_float, comparison_domain)
    distorted_domain, _ = change_comparison_domain(distorted_float, comparison_domain)
    maximum_domain, negative_check = change_comparison_domain(maximum, comparison_domain)
    error = mse(
        reference_domain,
        distorted_domain,
        negative_check=negative_check,
        comparison_domain="lin",
        mask=mask,
    )
    return 1000.0 if error == 0 else float(20 * np.log10(abs(float(maximum_domain)) / np.sqrt(error)))


def maximum_error(reference: np.ndarray, distorted: np.ndarray) -> float:
    reference, distorted, _, _ = check_domains(reference, distorted)
    return float(np.max(np.abs(reference - distorted)))


def mean_absolute_error(reference: np.ndarray, distorted: np.ndarray) -> float:
    reference, distorted, _, _ = check_domains(reference, distorted)
    return float(np.mean(np.abs(reference - distorted)))


def relative_error(reference: np.ndarray, distorted: np.ndarray) -> float:
    reference, distorted, _, _ = check_domains(reference, distorted)
    positive = reference > 0
    if not np.any(positive):
        raise ValueError("Reference image contains only zero values")
    return float(np.mean(np.abs(reference - distorted)[positive] / reference[positive]))


def display_referred(
    image: np.ndarray,
    display_min: float = 0.02,
    display_max: float = 1400.0,
    robust: bool = True,
    scaling: bool = True,
) -> np.ndarray:
    if scaling:
        maximum = (
            np.sort(luminance(image).reshape(-1))[round(luminance(image).size * 0.999) - 1]
            if robust
            else np.max(luminance(image))
        )
        output = image * display_max / maximum
    else:
        output, _, _ = normalize_image(image)
        output = output * (display_max - display_min) + display_min
    return np.clip(output, display_min, display_max)


def dist_cielab(distorted: np.ndarray, reference: np.ndarray) -> float:
    distorted_lab = convert_xyz_to_cielab(convert_rgb_to_xyz(distorted))
    reference_lab = convert_xyz_to_cielab(convert_rgb_to_xyz(reference))
    return float(np.sqrt(np.mean((distorted_lab - reference_lab) ** 2)))


def dist_hue(
    distorted: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, np.ndarray]:
    if not same_image(distorted, reference):
        raise ValueError("Images must have the same shape")
    reference = reference / np.max(reference)
    distorted = distorted / np.max(distorted)
    reference_ich = convert_ipt_to_ich(convert_xyz_to_ipt(convert_rgb_to_xyz(reference)))
    distorted_ich = convert_ipt_to_ich(convert_xyz_to_ipt(convert_rgb_to_xyz(distorted)))
    delta = distorted_ich[..., 2] - reference_ich[..., 2]
    delta = np.abs(delta + np.pi)
    over = delta > np.pi
    delta[over] = np.pi - (delta[over] - np.pi)
    delta = (np.pi - delta) / np.pi
    return float(np.mean(np.abs(delta))), delta


def tmqi_beta_function(a: float, b: float) -> float:
    return float(special.beta(a, b))


def tmqi_beta_pdf(values: np.ndarray | float, a: float, b: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Beta PDF values must be in [0, 1]")
    return np.power(values, a - 1) * np.power(1 - values, b - 1) / special.beta(a, b)


def tmqi_normal_pdf(
    values: np.ndarray | float,
    mean: float = 0,
    sigma: float = 1,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.exp(-((values - mean) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def tmqi_statistical_naturalness(ldr_luma: np.ndarray) -> float:
    luma = np.asarray(ldr_luma, dtype=np.float64)
    if np.max(luma) <= 1:
        luma = luma * 255
    brightness = float(np.mean(luma))
    block_deviations = []
    for y in range(0, luma.shape[0], 11):
        for x in range(0, luma.shape[1], 11):
            block_deviations.append(float(np.std(luma[y : y + 11, x : x + 11])))
    contrast = float(np.mean(block_deviations))
    a, b = 4.4, 10.1
    mode = (a - 1) / (a + b - 2)
    contrast_probability = float(
        tmqi_beta_pdf(np.clip(contrast / 64.29, 0, 1), a, b)
        / tmqi_beta_pdf(mode, a, b)
    )
    brightness_probability = float(
        tmqi_normal_pdf(brightness, 115.94, 27.99)
        / tmqi_normal_pdf(115.94, 115.94, 27.99)
    )
    return float(np.clip(contrast_probability * brightness_probability, 0, 1))


def tmqi_local_structural_fidelity(
    hdr_luma: np.ndarray,
    ldr_luma: np.ndarray,
    window: np.ndarray,
    spatial_frequency: float,
) -> tuple[float, np.ndarray]:
    kernel = np.asarray(window, dtype=np.float64)
    kernel /= np.sum(kernel)
    first = np.asarray(hdr_luma, dtype=np.float64)
    second = np.asarray(ldr_luma, dtype=np.float64)
    mean_first = ndimage.convolve(first, kernel, mode="constant")
    mean_second = ndimage.convolve(second, kernel, mode="constant")
    variance_first = np.maximum(ndimage.convolve(first**2, kernel, mode="constant") - mean_first**2, 0)
    variance_second = np.maximum(ndimage.convolve(second**2, kernel, mode="constant") - mean_second**2, 0)
    sigma_first, sigma_second = np.sqrt(variance_first), np.sqrt(variance_second)
    covariance = ndimage.convolve(first * second, kernel, mode="constant") - mean_first * mean_second
    csf = 100 * 2.6 * (0.0192 + 0.114 * spatial_frequency) * np.exp(
        -(0.114 * spatial_frequency) ** 1.1
    )
    threshold = 128 / (1.4 * csf)
    deviation = threshold / 3
    first_probability = stats.norm.cdf(sigma_first, threshold, deviation)
    second_probability = stats.norm.cdf(sigma_second, threshold, deviation)
    map_value = (
        (2 * first_probability * second_probability + 0.01)
        / (first_probability**2 + second_probability**2 + 0.01)
        * (covariance + 10)
        / (sigma_first * sigma_second + 10)
    )
    return float(np.mean(map_value)), map_value


def tmqi_structural_fidelity(
    hdr_luma: np.ndarray,
    ldr_luma: np.ndarray,
    levels: int,
    weights: np.ndarray,
    window: np.ndarray,
) -> tuple[float, np.ndarray, list[np.ndarray]]:
    first = np.asarray(hdr_luma, dtype=np.float64)
    second = np.asarray(ldr_luma, dtype=np.float64)
    local_scores = []
    maps = []
    frequency = 32.0
    for _ in range(levels):
        frequency /= 2
        score, score_map = tmqi_local_structural_fidelity(first, second, window, frequency)
        local_scores.append(score)
        maps.append(score_map)
        first = ndimage.uniform_filter(first, size=2, mode="reflect")[::2, ::2]
        second = ndimage.uniform_filter(second, size=2, mode="reflect")[::2, ::2]
    local = np.asarray(local_scores)
    return float(np.prod(np.maximum(local, 0) ** np.asarray(weights))), local, maps


def tmqi(
    hdr_image: np.ndarray,
    ldr_image: np.ndarray,
    window: np.ndarray | None = None,
) -> tuple[float, float, float, list[np.ndarray], np.ndarray]:
    if not same_image(hdr_image, ldr_image):
        raise ValueError("HDR and LDR images must have the same shape")
    if min(hdr_image.shape[:2]) < 11:
        raise ValueError("Images must be at least 11x11")
    if window is None:
        axis = np.arange(11) - 5
        kernel = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * 1.5**2))
        window = kernel / np.sum(kernel)
    hdr_luma = luminance(hdr_image)
    minimum, maximum = float(np.min(hdr_luma)), float(np.max(hdr_luma))
    hdr_encoded = np.rint((2**32 - 1) * (hdr_luma - minimum) / max(maximum - minimum, 1e-12))
    ldr_luma = luminance(np.asarray(ldr_image, dtype=np.float64))
    if np.max(ldr_luma) <= 1:
        ldr_luma *= 255
    weights = np.asarray((0.0448, 0.2856, 0.3001, 0.2363, 0.1333))
    structural, local, maps = tmqi_structural_fidelity(hdr_encoded, ldr_luma, 5, weights, window)
    naturalness = tmqi_statistical_naturalness(ldr_luma)
    quality = 0.8012 * structural**0.3046 + (1 - 0.8012) * naturalness**0.7088
    return float(quality), structural, naturalness, maps, local


def quant8(image: np.ndarray, exposure: float, inverse_gamma: float = 1 / 2.2) -> np.ndarray:
    return np.rint(255 * np.clip((np.asarray(image) * exposure) ** inverse_gamma, 0, 1))


def multiple_exposure_psnr(
    reference: np.ndarray,
    distorted: np.ndarray,
    exposure_min: int | None = None,
    exposure_max: int | None = None,
) -> tuple[float, int, int]:
    if not same_image(reference, distorted) or np.any(reference < 0) or np.any(distorted < 0):
        raise ValueError("Images must have equal shapes and non-negative values")
    if exposure_min is None or exposure_max is None:
        positive = np.concatenate((luminance(reference)[luminance(reference) > 0], luminance(distorted)[luminance(distorted) > 0]))
        if positive.size == 0:
            raise ValueError("Images contain no positive luminance")
        exposure_min = -round(np.log2(np.max(positive)))
        exposure_max = -round(np.log2(np.min(positive)))
    selected, errors = [], []
    for stop in range(exposure_min, exposure_max + 1):
        encoded_reference = quant8(reference, 2**stop)
        mean_value = np.mean(encoded_reference) / 255
        if 0.1 < mean_value < 0.9:
            selected.append(stop)
            errors.append(np.mean((encoded_reference - quant8(distorted, 2**stop)) ** 2))
    if not selected:
        return -1.0, exposure_max, exposure_min
    error = float(np.mean(errors))
    score = 1000.0 if error == 0 else float(20 * np.log10(255 / np.sqrt(error)))
    return score, max(selected), min(selected)


MSE = mse
RMSE = rmse
PSNR = psnr
MaximumError = maximum_error
MeanAbsoluteError = mean_absolute_error
RelativeError = relative_error
checkDomains = check_domains
getDisplayReferred = display_referred
distCIELAB = dist_cielab
distHue = dist_hue
TMQI_beta_function = tmqi_beta_function
TMQI_betapdf = tmqi_beta_pdf
TMQI_normpdf = tmqi_normal_pdf
TMQI_StatisticalNaturalness = tmqi_statistical_naturalness
TMQI_LocalStructuralFidelity = tmqi_local_structural_fidelity
TMQI_StructuralFidelity = tmqi_structural_fidelity
TMQI = tmqi
quant8 = quant8
mPSNR = multiple_exposure_psnr
changeComparisonDomain = change_comparison_domain
# Perceptually uniform encoding used by HDR-VDP-2.
def hdrvdp_joint_rod_cone_sens(
    adaptation_luminance: np.ndarray,
    csf_parameters: np.ndarray = np.array([30.162, 4.0627, 1.6596, 0.2712]),
) -> np.ndarray:
    p1, sensitivity_drop, transition_slope, low_slope = np.asarray(csf_parameters, dtype=float)
    luminance_value = np.maximum(np.asarray(adaptation_luminance, dtype=float), 1e-12)
    return p1 * ((sensitivity_drop / luminance_value) ** transition_slope + 1) ** (-low_slope)


def build_jndspace_from_s(log_luminance: np.ndarray, sensitivity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_luminance = np.asarray(log_luminance, dtype=float)
    linear = np.power(10.0, log_luminance)
    derivative = np.asarray(sensitivity, dtype=float) * np.log(10.0)
    return log_luminance, cumulative_trapezoid(derivative, log_luminance, initial=0)


def pu2_encode(luminance_value: np.ndarray) -> np.ndarray:
    log_lut = np.linspace(-5, 10, 2**12)
    sensitivity = hdrvdp_joint_rod_cone_sens(np.power(10.0, log_lut))
    _, perceptual_lut = build_jndspace_from_s(log_lut, sensitivity)
    log_value = np.log10(np.clip(np.asarray(luminance_value, dtype=float), 1e-5, 1e10))
    encoded = np.interp(log_value, log_lut, perceptual_lut)
    return 255 * (encoded - 31.9270) / (149.9244 - 31.9270)


pu2_encode = pu2_encode
hdrvdp_joint_rod_cone_sens = hdrvdp_joint_rod_cone_sens
build_jndspace_from_S = build_jndspace_from_s
