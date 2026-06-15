from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage


def as_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in {2, 3}:
        raise ValueError("Image must have shape HxW or HxWxC")
    return array


def check_three_color(image: np.ndarray) -> None:
    image = as_image(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("A three-channel color image is required")


def check_one_or_three_color(image: np.ndarray) -> None:
    image = as_image(image)
    channels = 1 if image.ndim == 2 else image.shape[2]
    if channels not in {1, 3}:
        raise ValueError("A one- or three-channel image is required")


def check_nonnegative(image: np.ndarray) -> None:
    if np.any(np.asarray(image) < 0):
        raise ValueError("Image contains negative values")


def check_in_unit_interval(image: np.ndarray) -> None:
    if np.any(np.asarray(image) < 0) or np.any(np.asarray(image) > 1):
        raise ValueError("Image values must be in [0, 1]")


def clamp_image(image: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(image, lower, upper)


def remove_specials(image: np.ndarray, value: float = 0.0) -> np.ndarray:
    return np.nan_to_num(
        np.asarray(image),
        copy=True,
        nan=value,
        posinf=value,
        neginf=value,
    )


def matlab_percentile(image: np.ndarray, percentile: float) -> float:
    values = np.sort(np.asarray(image).reshape(-1))
    percentile = float(np.clip(percentile, 0.0, 1.0))
    index = max(int(np.floor(values.size * percentile + 0.5)), 1)
    return float(values[index - 1])


def normalize_image(
    image: np.ndarray,
    image_min: float | None = None,
    image_max: float | None = None,
) -> tuple[np.ndarray, float, float]:
    image = np.asarray(image)
    image_min = matlab_percentile(image, 0.01) if image_min is None else image_min
    image_max = matlab_percentile(image, 0.999) if image_max is None else image_max
    if image_min < 0:
        image_min = float(np.min(image))
    if image_max < 0:
        image_max = float(np.max(image))
    delta = image_max - image_min
    output = np.clip((image - image_min) / delta, 0.0, 1.0) if delta > 0 else image.copy()
    return output, float(image_min), float(image_max)


def normalize_from_anything(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    if np.issubdtype(image.dtype, np.floating) and np.max(image) > 1.0:
        return image / np.max(image)
    return image.copy()


def image_shift(
    image: np.ndarray,
    shift_vector: tuple[int, int] = (0, 0),
) -> np.ndarray:
    dx, dy = map(int, shift_vector)
    pad = ((max(dy, 0), max(-dy, 0)), (max(dx, 0), max(-dx, 0)))
    if image.ndim == 3:
        pad += ((0, 0),)
    padded = np.pad(image, pad, mode="edge")
    y0 = max(-dy, 0)
    x0 = max(-dx, 0)
    return padded[y0 : y0 + image.shape[0], x0 : x0 + image.shape[1]]


def image_shift_wrap(image: np.ndarray, dx: int = 0) -> np.ndarray:
    return np.roll(image, int(dx), axis=1)


def image_flip(image: np.ndarray) -> np.ndarray:
    return np.flip(image, axis=1).copy()


def one_to_many(image: np.ndarray, channels: int) -> np.ndarray:
    if np.asarray(image).ndim != 2:
        raise ValueError("Input must be a single-channel image")
    return np.repeat(np.asarray(image)[:, :, None], channels, axis=2)


def same_image(image1: np.ndarray, image2: np.ndarray) -> bool:
    return np.asarray(image1).shape == np.asarray(image2).shape


def similar_image(image1: np.ndarray, image2: np.ndarray) -> bool:
    return np.asarray(image1).shape[:2] == np.asarray(image2).shape[:2]


def same_height(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    height = image1.shape[0]
    width = round(image2.shape[1] * height / image2.shape[0])
    return cv2.resize(image2, (width, height), interpolation=cv2.INTER_LINEAR)


def gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    sigma_axes = (sigma, sigma, 0) if image.ndim == 3 else sigma
    return ndimage.gaussian_filter(image, sigma_axes, mode="nearest")


def box_filter(image: np.ndarray, size: int) -> np.ndarray:
    size_axes = (size, size, 1) if image.ndim == 3 else size
    return ndimage.uniform_filter(image, size=size_axes, mode="nearest")


def compute_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.gradient(np.asarray(image), axis=(1, 0))


def compute_divergence(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return np.gradient(dx, axis=1) + np.gradient(dy, axis=0)


def file_extension(filename: str | Path) -> str:
    return Path(filename).suffix.lstrip(".")


def remove_extension(filename: str | Path) -> str:
    return str(Path(filename).with_suffix(""))


# MATLAB-compatible aliases.
ClampImg = clamp_image
RemoveSpecials = remove_specials
MaxQuart = matlab_percentile
normalizeImg = normalize_image
normalizeFromAnything = normalize_from_anything
imShift = image_shift
imShiftWrap = image_shift_wrap
imFlip = image_flip
imOneToMany = one_to_many
isSameImage = same_image
isSimilarImage = similar_image
imSameHeight = same_height
filterGaussian = gaussian_filter
filterBox = box_filter
computeGradients = compute_gradients
computeDivergence = compute_divergence
fileExtension = file_extension
getExt = file_extension
RemoveExt = remove_extension
check3Color = check_three_color
check13Color = check_one_or_three_color
checkNegative = check_nonnegative
checkIn01 = check_in_unit_interval
