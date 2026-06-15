from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import minimize

from .analysis import image_key, log_mean
from .colorspace import convert_rgb_to_srgb, luminance
from .core import matlab_percentile, remove_specials


def tabled_function(image: np.ndarray, table: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    table = np.asarray(table, dtype=np.float64)
    source = image if image.ndim == 3 else image[..., None]
    if table.ndim == 1:
        table = table[:, None]
    if table.shape[1] == 1 and source.shape[2] > 1:
        table = np.repeat(table, source.shape[2], axis=1)
    if table.shape[1] != source.shape[2]:
        raise ValueError("LUT channel count does not match image")
    indices = np.rint(source * (table.shape[0] - 1)).astype(int)
    indices = np.clip(indices, 0, table.shape[0] - 1)
    output = np.empty_like(source)
    for channel in range(source.shape[2]):
        output[..., channel] = table[indices[..., channel], channel]
    return output if image.ndim == 3 else output[..., 0]


def remove_crf(
    image: np.ndarray,
    linearization: str = "gamma",
    function: np.ndarray | float = 2.2,
) -> np.ndarray:
    if linearization == "gamma":
        return np.asarray(image) ** function
    if linearization.lower() == "srgb":
        return convert_rgb_to_srgb(image, inverse=True)
    if linearization == "LUT":
        return tabled_function(image, np.asarray(function))
    if linearization == "poly":
        coefficients = np.asarray(function)
        source = image if image.ndim == 3 else image[..., None]
        output = np.empty_like(source, dtype=np.float64)
        for channel in range(source.shape[2]):
            output[..., channel] = np.polyval(coefficients[:, channel], source[..., channel])
        return output if image.ndim == 3 else output[..., 0]
    return np.asarray(image, dtype=np.float64).copy()


def apply_crf(
    image: np.ndarray,
    linearization: str = "gamma",
    function: np.ndarray | float = 2.2,
) -> np.ndarray:
    if linearization == "gamma":
        return np.asarray(image) ** (1.0 / float(function))
    if linearization.lower() == "srgb":
        return convert_rgb_to_srgb(image)
    if linearization == "poly":
        coefficients = np.asarray(function)
        channels = coefficients.shape[1]
        x = np.linspace(0, 1, 256)
        inverse_table = np.empty((256, channels))
        for channel in range(channels):
            y = np.polyval(coefficients[:, channel], x)
            order = np.argsort(y)
            inverse_table[:, channel] = np.interp(x, y[order], x[order])
        return tabled_function(image, inverse_table)
    if linearization == "LUT":
        return tabled_function(image, np.asarray(function))
    return np.asarray(image, dtype=np.float64).copy()


def weight_function(
    image: np.ndarray,
    kind: str,
    mean_weight: bool = False,
    bounds: tuple[float, float] = (0.0, 1.0),
    polynomial: np.ndarray | None = None,
) -> np.ndarray:
    source = np.asarray(image, dtype=np.float64).copy()
    if source.ndim == 3 and mean_weight:
        source[:] = np.mean(source, axis=2, keepdims=True)
    if kind == "Deb97_p05":
        kind, bounds = "Deb97", (0.05, 0.95)
    if kind == "all":
        weight = np.ones_like(source)
    elif kind == "identity":
        weight = source
    elif kind == "reverse":
        weight = 1 - source
    elif kind == "box":
        weight = ((source >= bounds[0]) & (source <= bounds[1])).astype(float)
    elif kind == "Robertson":
        shift = np.exp(-4)
        weight = (np.exp(-16 * (source - 0.5) ** 2) - shift) / (1 - shift)
    elif kind == "hat":
        weight = 1 - (2 * source - 1) ** 12
    elif kind == "poly":
        if polynomial is None:
            raise ValueError("Polynomial coefficients are required")
        polynomial = np.asarray(polynomial)
        weight = np.empty_like(source)
        for channel in range(source.shape[2]):
            derivative = np.polyder(polynomial[:, channel])
            weight[..., channel] = np.polyval(polynomial[:, channel], source[..., channel]) / np.polyval(
                derivative, source[..., channel]
            )
    elif kind == "Deb97":
        minimum, maximum = bounds
        midpoint = (minimum + maximum) / 2
        weight = np.where(source <= midpoint, source - minimum, maximum - source)
        if maximum > minimum:
            weight /= midpoint
    else:
        raise ValueError(f"Unsupported weight type: {kind}")
    return np.clip(weight, 0, 1)


def re_expose(
    source: np.ndarray,
    source_exposure: float,
    target_exposure: float,
    linearization: str = "gamma",
    function: np.ndarray | float = 2.2,
) -> np.ndarray:
    linear = remove_crf(source, linearization, function)
    return np.clip(((linear * target_exposure) / source_exposure) ** (1 / 2.2), 0, 1)


def estimate_average_luminance(
    exposure_time: float,
    aperture: float = 1.0,
    iso: float = 1.0,
    calibration: float = 12.5,
) -> tuple[float, float]:
    calibration = float(np.clip(calibration, 10.6, 13.4))
    value = calibration * aperture**2 / (iso * exposure_time)
    return value, 1.0 / value


def simulate_spatial_exposure(image: np.ndarray, fstops: np.ndarray) -> np.ndarray:
    exposures = np.exp2(np.asarray(fstops).reshape(-1))
    if exposures.size < 4:
        raise ValueError("Four exposures are required")
    exposures = exposures[:4]
    output = np.zeros_like(image, dtype=np.float64)
    output[0::2, 0::2, ...] = exposures[0] * image[0::2, 0::2, ...]
    output[0::2, 1::2, ...] = exposures[1] * image[0::2, 1::2, ...]
    output[1::2, 0::2, ...] = exposures[2] * image[1::2, 0::2, ...]
    output[1::2, 1::2, ...] = exposures[3] * image[1::2, 1::2, ...]
    return np.clip(output ** (1 / 2.2), 0, 1)


def akyuz_tau(image: np.ndarray) -> np.ndarray:
    first, second, width = 200 / 255, 250 / 255, 50 / 255
    tau = np.ones_like(image, dtype=np.float64)
    tau[image >= second] = 0
    h = 1 - (second - image) / width
    smooth = 1 - 3 * h**2 + 2 * h**3
    selected = (image >= first) & (image <= second)
    tau[selected] = smooth[selected]
    return tau


def calibrate_hdr(image: np.ndarray, robust: bool = False) -> tuple[np.ndarray, float]:
    luma = luminance(image)
    average = np.mean(np.log10(luma + 1e-5))
    if robust:
        minimum = matlab_percentile(luma, 0.05)
        maximum = matlab_percentile(luma, 0.95)
    else:
        positive = luma[luma > 0]
        minimum, maximum = np.min(positive), np.max(positive)
    key = (average - np.log10(minimum)) / (np.log10(maximum) - np.log10(minimum))
    factor = 1e4 * key / maximum
    threshold = minimum + (0.6 + 0.4 * (1 - key)) * (maximum - minimum)
    return image * factor, float(threshold)


def find_chromaticity_scale(measured: np.ndarray, image: np.ndarray) -> np.ndarray:
    measured = np.asarray(measured, dtype=np.float64)
    image = np.asarray(image, dtype=np.float64)
    if measured.size != image.size or measured.size == 0:
        raise ValueError("Colors must have equal non-zero channel counts")

    def residual(scale: np.ndarray) -> float:
        corrected = image * scale
        corrected /= np.linalg.norm(corrected)
        target = measured / np.linalg.norm(measured)
        return float(np.sum((corrected - target) ** 2))

    return minimize(residual, np.ones(measured.size), method="Nelder-Mead").x


def gsolve(
    samples: np.ndarray,
    log_exposures: np.ndarray,
    smoothing: float,
    weights: np.ndarray,
) -> np.ndarray:
    samples = np.asarray(samples)
    log_exposures = np.asarray(log_exposures, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if samples.ndim != 2 or samples.shape[1] != log_exposures.size:
        raise ValueError("Samples must have shape sample_count x exposure_count")
    if weights.size < 3:
        raise ValueError("At least three response weights are required")

    levels = weights.size
    valid_count = int(np.count_nonzero((samples >= 0) & (samples < levels)))
    matrix = np.zeros((valid_count + levels - 1, levels + samples.shape[0]))
    target = np.zeros(matrix.shape[0])
    row = 0
    for sample_index in range(samples.shape[0]):
        for exposure_index in range(samples.shape[1]):
            value = int(np.rint(samples[sample_index, exposure_index]))
            if value < 0 or value >= levels:
                continue
            weight = weights[value]
            matrix[row, value] = weight
            matrix[row, levels + sample_index] = -weight
            target[row] = weight * log_exposures[exposure_index]
            row += 1

    matrix[row, levels // 2] = 1
    row += 1
    for value in range(1, levels - 1):
        weight = smoothing * weights[value]
        matrix[row, value - 1 : value + 2] = (weight, -2 * weight, weight)
        row += 1
    solution, *_ = np.linalg.lstsq(matrix[:row], target[:row], rcond=None)
    return solution[:levels]


def debevec_crf(
    stack: np.ndarray,
    exposures: np.ndarray,
    samples: int = 256,
    sampling_strategy: str = "Grossberg",
    smoothing: float = 128,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    from .stacks import stack_subsampling

    stack = np.asarray(stack)
    exposures = np.asarray(exposures, dtype=np.float64).reshape(-1)
    if stack.ndim != 4 or stack.shape[3] != exposures.size:
        raise ValueError("Stack must have shape HxWxCxN matching exposures")
    if np.any(exposures <= 0):
        raise ValueError("Exposures must be positive")

    sampled = stack_subsampling(stack, exposures, samples, sampling_strategy)
    if sampled.shape[0] == 0:
        raise ValueError("The sampling strategy did not return any valid samples")
    weights = weight_function(np.linspace(0, 1, 256), "Deb97")
    response = np.empty((256, stack.shape[2]), dtype=np.float64)
    for channel in range(stack.shape[2]):
        response[:, channel] = np.exp(
            gsolve(sampled[:, :, channel], np.log(exposures), smoothing, weights)
        )

    if response.shape[1] > 1:
        midpoint = response[128, :]
        scale = find_chromaticity_scale(np.full(response.shape[1], 0.5), midpoint)
        response *= scale[None, :]
    maximum = np.max(response, axis=0)
    if normalize:
        response /= max(float(np.max(maximum)), np.finfo(np.float64).eps)
    return response, maximum


def compute_glare_image(
    image: np.ndarray,
    psf: np.ndarray,
    hot_pixels: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    del psf  # Kept for MATLAB API compatibility; the polynomial defines the PSF.
    image = np.asarray(image, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    positions = np.asarray(hot_pixels)
    if image.ndim != 3 or coefficients.size != 4:
        raise ValueError("Expected an HxWxC image and four PSF coefficients")
    if positions.size == 0:
        return np.zeros_like(image)
    if positions.shape[0] != 2 and positions.shape[1] == 2:
        positions = positions.T
    if positions.shape[0] != 2:
        raise ValueError("Hot-pixel coordinates must have shape 2xN or Nx2")

    height, width, _ = image.shape
    yy, xx = np.mgrid[:height, :width]
    glare = np.zeros_like(image)
    for x_position, y_position in positions.T:
        x = int(np.rint(x_position))
        y = int(np.rint(y_position))
        if not (0 <= x < width and 0 <= y < height):
            continue
        radius = np.maximum(np.hypot(xx - x, yy - y), 1)
        value = (
            coefficients[0]
            + coefficients[1] / radius
            + coefficients[2] / radius**2
            + coefficients[3] / radius**3
        )
        glare += value[..., None] * image[y, x, :]

    excessive = glare > image
    if np.any(excessive):
        ratios = np.divide(
            image[excessive],
            glare[excessive],
            out=np.ones(np.count_nonzero(excessive)),
            where=glare[excessive] != 0,
        )
        glare *= max(float(np.min(ratios)), 0.0)
    return remove_specials(glare)


def estimate_psf(
    image: np.ndarray,
    working_width: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3 or working_width < 8:
        raise ValueError("Expected an HxWxC image and working_width >= 8")
    height, width = image.shape[:2]
    working_height = max(int(round(height * working_width / width)), 1)
    gray = cv2.resize(
        luminance(image),
        (working_width, working_height),
        interpolation=cv2.INTER_LINEAR,
    )
    positive = gray[gray > 0]
    if positive.size == 0:
        raise ValueError("PSF cannot be estimated from a black image")
    threshold = min(1000 * float(np.min(positive)), matlab_percentile(gray, 0.1))
    hot_y, hot_x = np.nonzero(gray > threshold)
    dark_y, dark_x = np.nonzero(gray <= threshold)
    if hot_x.size == 0 or dark_x.size < 4:
        raise ValueError("PSF estimation requires both bright and dark pixels")

    hot_values = gray[hot_y, hot_x]
    matrix = np.zeros((dark_x.size, 4), dtype=np.float64)
    matrix[:, 0] = np.sum(hot_values)
    chunk_size = 4096
    for start in range(0, dark_x.size, chunk_size):
        end = min(start + chunk_size, dark_x.size)
        dx = dark_x[start:end, None] - hot_x[None, :]
        dy = dark_y[start:end, None] - hot_y[None, :]
        radius_squared = dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2
        valid = radius_squared >= 9
        radius = np.sqrt(np.maximum(radius_squared, 1))
        weighted = hot_values[None, :] * valid
        matrix[start:end, 1] = np.sum(weighted / radius, axis=1)
        matrix[start:end, 2] = np.sum(weighted / radius_squared.clip(min=1), axis=1)
        matrix[start:end, 3] = np.sum(weighted / (radius_squared.clip(min=1) * radius), axis=1)
    coefficients, *_ = np.linalg.lstsq(matrix, gray[dark_y, dark_x], rcond=None)

    kernel_y, kernel_x = np.mgrid[-16:17, -16:17]
    radius = np.maximum(np.hypot(kernel_x, kernel_y), 1)
    psf = (
        coefficients[0]
        + coefficients[1] / radius
        + coefficients[2] / radius**2
        + coefficients[3] / radius**3
    )
    return psf, coefficients, np.vstack((hot_x, hot_y))


def remove_glare(
    image: np.ndarray,
    working_width: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float64)
    psf, coefficients, hot_pixels = estimate_psf(image, working_width)
    working_height = max(int(round(image.shape[0] * working_width / image.shape[1])), 1)
    working = cv2.resize(image, (working_width, working_height), interpolation=cv2.INTER_LANCZOS4)
    glare = compute_glare_image(working, psf, hot_pixels, coefficients)
    glare = cv2.resize(glare, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return image - glare, glare, psf


def build_hdr(
    stack: np.ndarray,
    exposures: np.ndarray,
    linearization: str = "gamma",
    function: np.ndarray | float | None = None,
    weight_type: str = "all",
    merge_type: str = "log",
    mean_weight: bool = False,
) -> tuple[np.ndarray, np.ndarray | float | None]:
    stack = np.asarray(stack)
    exposures = np.asarray(exposures, dtype=np.float64).reshape(-1)
    if stack.ndim != 4 or stack.shape[3] != exposures.size:
        raise ValueError("Stack must have shape HxWxCxN matching exposures")
    if np.unique(exposures).size != exposures.size or np.any(exposures <= 0):
        raise ValueError("Exposures must be positive and unique")
    if linearization == "LUT" and function is None:
        function, _ = debevec_crf(stack, exposures)
    if linearization == "gamma" and (function is None or float(function) <= 0):
        function = 2.2
    scale = np.iinfo(stack.dtype).max if np.issubdtype(stack.dtype, np.integer) else 1.0
    output = np.zeros(stack.shape[:3], dtype=np.float64)
    total_weight = np.zeros_like(output)
    shortest = int(np.argmin(exposures))
    longest = int(np.argmax(exposures))
    delta = 1 / 65535
    for index, exposure in enumerate(exposures):
        frame = np.clip(stack[..., index].astype(np.float64) / scale, 0, 1)
        current_weight = weight_type
        if index == shortest and np.any(frame > 0.9):
            current_weight = "identity"
        if index == longest and np.any(frame < 0.1):
            current_weight = "reverse"
        weight = weight_function(frame, current_weight, mean_weight)
        linear = remove_crf(frame, linearization, function)
        if merge_type == "linear":
            output += weight * linear / exposure
            total_weight += weight
        elif merge_type == "log":
            output += weight * (np.log(linear + delta) - np.log(exposure))
            total_weight += weight
        elif merge_type == "w_time_sq":
            output += weight * linear * exposure
            total_weight += weight * exposure**2
        else:
            raise ValueError("merge_type must be linear, log, or w_time_sq")
    merged = np.divide(output, total_weight, out=np.zeros_like(output), where=total_weight > 0)
    if merge_type == "log":
        merged = np.exp(merged)
    middle = stack.shape[3] // 2
    median = np.clip(stack[..., middle].astype(np.float64) / scale, 0, 1)
    for dark, frame_index in ((False, shortest), (True, longest)):
        frame = np.clip(stack[..., frame_index].astype(np.float64) / scale, 0, 1)
        fallback = remove_crf(frame, linearization, function) / exposures[frame_index]
        zero = np.max(total_weight <= 1e-4, axis=2)
        mask = zero & ((np.mean(median, axis=2) < 0.5) if dark else (np.mean(median, axis=2) > 0.5))
        merged[mask] = fallback[mask]
    return remove_specials(merged), function


tabledFunction = tabled_function
RemoveCRF = remove_crf
ApplyCRF = apply_crf
WeightFunction = weight_function
reExpose = re_expose
EstimateAverageLuminance = estimate_average_luminance
SimulateSpatialExposure = simulate_spatial_exposure
AkyuzTau = akyuz_tau
CalibrateHDR = calibrate_hdr
FindChromaticyScale = find_chromaticity_scale
gsolve = gsolve
DebevecCRF = debevec_crf
ComputeGlareImage = compute_glare_image
EstimatePSF = estimate_psf
RemoveGlare = remove_glare
BuildHDR = build_hdr
