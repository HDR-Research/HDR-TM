from __future__ import annotations

import numpy as np

from .colorspace import convert_rgb_to_srgb
from .generation import akyuz_tau, find_chromaticity_scale, tabled_function, weight_function
from .stacks import stack_subsampling


def _normalized_stack(stack: np.ndarray) -> np.ndarray:
    source = np.asarray(stack)
    if np.issubdtype(source.dtype, np.integer):
        return source.astype(np.float64) / np.iinfo(source.dtype).max
    return np.clip(source.astype(np.float64), 0, 1)


def Normalization(response: np.ndarray) -> np.ndarray:
    output = np.asarray(response, dtype=np.float64).copy()
    for channel in range(output.shape[1]):
        positive = np.flatnonzero(output[:, channel] > 0)
        if positive.size:
            midpoint = positive[0] + round((positive[-1] - positive[0]) / 2)
            scale = output[midpoint, channel]
            if scale > 0:
                output[:, channel] /= scale
    return output


def Update_X(
    stack: np.ndarray,
    exposures: np.ndarray,
    response: np.ndarray,
) -> np.ndarray:
    source = _normalized_stack(stack)
    exposures = np.asarray(exposures, dtype=np.float64)
    numerator = np.zeros(source.shape[:3])
    denominator = np.zeros_like(numerator)
    for index, exposure in enumerate(exposures):
        frame = source[..., index]
        weight = weight_function(frame, "Robertson")
        linear = tabled_function(frame, response)
        numerator += weight * linear * exposure
        denominator += weight * exposure**2
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def Update_lin_fun(
    radiance: np.ndarray,
    stack: np.ndarray,
    exposures: np.ndarray,
    response: np.ndarray,
) -> np.ndarray:
    quantized = np.rint(_normalized_stack(stack) * 255).astype(np.uint8)
    output = np.zeros_like(response)
    for channel in range(quantized.shape[2]):
        previous = 0.0
        for level in range(256):
            total, count = 0.0, 0
            for index, exposure in enumerate(exposures):
                selected = quantized[..., channel, index] == level
                values = radiance[..., channel][selected]
                values = values[values > 0]
                total += float(exposure * np.sum(values))
                count += values.size
            value = total / count if count else previous
            output[level, channel] = value
            previous = value
        output[:, channel] = np.maximum.accumulate(output[:, channel])
    return output


def robertson_crf(
    stack: np.ndarray,
    exposures: np.ndarray,
    max_iterations: int = 15,
    error_threshold: float = 1e-5,
    normalize: bool = False,
) -> tuple[np.ndarray, float]:
    source = _normalized_stack(stack)
    exposures = np.asarray(exposures, dtype=np.float64).reshape(-1)
    if source.ndim != 4 or source.shape[3] != exposures.size or np.any(exposures <= 0):
        raise ValueError("Stack and positive exposures are incompatible")
    response = np.repeat(np.linspace(0, 1, 256)[:, None], source.shape[2], axis=1)
    for _ in range(max(max_iterations, 1)):
        previous = response.copy()
        response = Normalization(response)
        radiance = Update_X(source, exposures, response)
        response = Update_lin_fun(radiance, source, exposures, response)
        if np.mean((previous - response) ** 2) < error_threshold:
            break
    maximum = float(np.max(response))
    if normalize and maximum > 0:
        response = np.clip(response / maximum, 0, 1)
    return response, maximum


def mann_picard_crf(
    stack: np.ndarray,
    exposures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = _normalized_stack(stack)
    exposures = np.asarray(exposures, dtype=np.float64)
    order = np.argsort(exposures)
    source, exposures = source[..., order], exposures[order]
    scores = [
        np.count_nonzero((source[..., index] > 0.05) & (source[..., index] < 0.95))
        for index in range(1, source.shape[3])
    ]
    first = max(int(np.argmax(scores)), 0) if scores else 0
    second = min(first + 1, source.shape[3] - 1)
    ratio = exposures[second] / exposures[first]
    parameters = np.empty((2, source.shape[2]))
    table = np.empty((256, source.shape[2]))
    x_table = np.linspace(0, 1, 256)
    for channel in range(source.shape[2]):
        x = source[..., channel, first].reshape(-1)
        y = source[..., channel, second].reshape(-1)
        valid = (y < 0.95) & (x > 0) & (y > 0)
        slope, intercept = np.polyfit(x[valid], y[valid], 1)
        gamma = np.log(max(slope, 1e-12)) / np.log(ratio)
        alpha = intercept / (1 - ratio**gamma)
        offset, exponent = -alpha, 1 / gamma
        parameters[:, channel] = (offset, exponent)
        table[:, channel] = np.maximum(x_table + offset, 0) ** exponent
        table[:, channel] = np.maximum.accumulate(table[:, channel])
    return table, parameters


def MN_d(
    first: np.ndarray,
    second: np.ndarray,
    ratio: float,
    power: int,
) -> np.ndarray:
    valid = (first > 0) & (second > 0)
    return first[valid] ** power - ratio * second[valid] ** power


def _mitsunaga_fit(
    samples: np.ndarray,
    exposures: np.ndarray,
    degree: int,
    full: bool,
    max_iterations: int = -1,
) -> tuple[np.ndarray, float]:
    del max_iterations
    sample_count, frame_count, channels = samples.shape
    coefficients = np.empty((degree + 1, channels))
    error = 0.0
    pairs = (
        [(a, b) for a in range(frame_count) for b in range(a + 1, frame_count)]
        if full
        else [(index, index + 1) for index in range(frame_count - 1)]
    )
    for channel in range(channels):
        rows = []
        for first, second in pairs:
            ratio = exposures[first] / exposures[second]
            x, y = samples[:, first, channel], samples[:, second, channel]
            valid = (x > 0) & (y > 0) & (x < 1) & (y < 1)
            for xv, yv in zip(x[valid], y[valid]):
                rows.append([xv**power - ratio * yv**power for power in range(degree + 1)])
        matrix = np.asarray(rows)
        if matrix.shape[0] < degree + 1:
            raise ValueError("Not enough valid samples for polynomial CRF fitting")
        augmented = np.vstack((matrix, np.ones((1, degree + 1))))
        target = np.zeros(augmented.shape[0])
        target[-1] = 1
        ascending, *_ = np.linalg.lstsq(augmented, target, rcond=None)
        coefficients[:, channel] = ascending[::-1]
        error += float(np.sum((matrix @ ascending) ** 2))
    return coefficients, error


def mitsunaga_nayar_crf_classic(
    samples: np.ndarray,
    exposures: np.ndarray,
    degree: int,
    max_iterations: int = -1,
) -> tuple[np.ndarray, float]:
    return _mitsunaga_fit(samples, np.asarray(exposures), degree, False, max_iterations)


def mitsunaga_nayar_crf_full(
    samples: np.ndarray,
    exposures: np.ndarray,
    degree: int,
    max_iterations: int = -1,
) -> tuple[np.ndarray, float]:
    return _mitsunaga_fit(samples, np.asarray(exposures), degree, True, max_iterations)


def mitsunaga_nayar_crf(
    stack: np.ndarray,
    exposures: np.ndarray,
    degree: int = -3,
    samples: int = 256,
    sampling_strategy: str = "RegularSpatial",
    full: bool = False,
    max_iterations: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    source = _normalized_stack(stack)
    exposures = np.asarray(exposures, dtype=np.float64)
    order = np.argsort(exposures)
    source, exposures = source[..., order], exposures[order]
    sampled = stack_subsampling(source, exposures, samples, sampling_strategy, 0.01) / 255
    fitter = mitsunaga_nayar_crf_full if full else mitsunaga_nayar_crf_classic
    degrees = range(1, 7) if degree <= 0 else (degree,)
    candidates = [(*fitter(sampled, exposures, value, max_iterations), value) for value in degrees]
    polynomial, _, _ = min(candidates, key=lambda result: result[1])
    x = np.linspace(0, 1, 256)
    table = np.stack([np.polyval(polynomial[:, channel], x) for channel in range(source.shape[2])], axis=1)
    table = np.maximum.accumulate(np.clip(table, 0, None), axis=0)
    midpoint = table[128]
    table *= find_chromaticity_scale(np.full(source.shape[2], 0.5), midpoint)[None, :]
    return table, polynomial


def RAWCRFn(
    raw_image: np.ndarray,
    jpeg_image: np.ndarray,
    degree: int,
    thresholds: tuple[float, float] = (0.05, 0.95),
) -> np.ndarray:
    raw, jpeg = np.asarray(raw_image, dtype=np.float64), np.asarray(jpeg_image, dtype=np.float64)
    if raw.shape != jpeg.shape:
        raise ValueError("RAW and JPEG images must have equal shapes")
    source_raw = raw[..., None] if raw.ndim == 2 else raw
    source_jpeg = jpeg[..., None] if jpeg.ndim == 2 else jpeg
    polynomial = np.empty((degree + 1, source_raw.shape[2]))
    for channel in range(source_raw.shape[2]):
        x, y = source_jpeg[..., channel].reshape(-1), source_raw[..., channel].reshape(-1)
        valid = (x > thresholds[0]) & (x < thresholds[1]) & np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(valid) < degree + 1:
            raise ValueError("Not enough data for RAW CRF estimation")
        polynomial[:, channel] = np.polyfit(x[valid], y[valid], degree)
    return polynomial


def raw_crf(
    raw_image: np.ndarray,
    jpeg_image: np.ndarray,
    degree: int = -1,
    outlier_threshold: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    degrees = range(1, 7) if degree < 0 else (degree,)
    candidates = []
    for value in degrees:
        polynomial = RAWCRFn(raw_image, jpeg_image, value, (outlier_threshold, 1 - outlier_threshold))
        predicted = np.stack(
            [np.polyval(polynomial[:, channel], jpeg_image[..., channel]) for channel in range(jpeg_image.shape[2])],
            axis=2,
        )
        candidates.append((float(np.mean((predicted - raw_image) ** 2)), polynomial))
    _, polynomial = min(candidates, key=lambda candidate: candidate[0])
    x = np.linspace(0, 1, 256)
    table = np.stack([np.polyval(polynomial[:, channel], x) for channel in range(polynomial.shape[1])], axis=1)
    return np.maximum.accumulate(np.clip(table, 0, None), axis=0), polynomial


def akyuz_ldr_stack_denoise(
    stack: np.ndarray,
    exposures: np.ndarray,
    linearization: str = "linear",
    function: np.ndarray | float = 2.2,
) -> np.ndarray:
    source = _normalized_stack(stack)
    exposures = np.asarray(exposures, dtype=np.float64)
    output = source.copy()
    target = max(round(source.shape[3] / 3), 1)
    for index in range(source.shape[3]):
        numerator = np.zeros(source.shape[:3])
        denominator = np.zeros_like(numerator)
        for offset in range(target):
            frame_index = index + offset
            if frame_index >= source.shape[3]:
                break
            frame = source[..., frame_index]
            weight = np.ones_like(frame) if offset == 0 else akyuz_tau(frame)
            if linearization == "gamma2.2":
                linear = frame**2.2
            elif linearization.lower() == "srgb":
                linear = convert_rgb_to_srgb(frame, inverse=True)
            elif linearization == "tabledDeb97":
                linear = tabled_function(frame, np.asarray(function))
            else:
                linear = frame
            numerator += weight * linear
            denominator += weight * exposures[frame_index]
        radiance = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        output[..., index] = np.clip(radiance * exposures[index], 0, 1)
    return output


RobertsonCRF = robertson_crf
MannPicardCRF = mann_picard_crf
MitsunagaNayarCRF = mitsunaga_nayar_crf
MitsunagaNayarCRFClassic = mitsunaga_nayar_crf_classic
MitsunagaNayarCRFFull = mitsunaga_nayar_crf_full
RAWCRF = raw_crf
AkyuzLDRStackDenoise = akyuz_ldr_stack_denoise
