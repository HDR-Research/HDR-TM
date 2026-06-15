from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from scipy import ndimage

from .core import same_image, similar_image


BURT_ADELSON_KERNEL = np.outer([1, 4, 6, 4, 1], [1, 4, 6, 4, 1]) / 256.0


@dataclass
class Pyramid:
    base: np.ndarray
    details: list[np.ndarray]

    @property
    def list(self) -> list[dict[str, np.ndarray]]:
        return [{"detail": detail} for detail in self.details]


def _filter(image: np.ndarray) -> np.ndarray:
    return ndimage.convolve(image, BURT_ADELSON_KERNEL, mode="reflect")


def _resize_nearest(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def gaussian_aux(image: np.ndarray) -> np.ndarray:
    return _filter(image)[::2, ::2, ...]


def gaussian_pyramid(image: np.ndarray, max_levels: int = -1) -> Pyramid:
    levels_log2 = int(np.floor(np.log2(min(image.shape[:2]))))
    levels = levels_log2 if max_levels < 0 else min(levels_log2, max_levels)
    current = np.asarray(image, dtype=np.float64)
    details = []
    for _ in range(levels):
        details.append(current)
        current = gaussian_aux(current)
    return Pyramid(current, details)


def laplacian_aux(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = gaussian_aux(image)
    expanded = _filter(_resize_nearest(lower, image.shape[:2]))
    return lower, image - expanded


def laplacian_pyramid(image: np.ndarray, max_levels: int = -1) -> Pyramid:
    levels_log2 = int(np.floor(np.log2(min(image.shape[:2]))))
    levels = levels_log2 if max_levels < 0 else min(levels_log2, max_levels)
    current = np.asarray(image, dtype=np.float64)
    details = []
    for _ in range(levels):
        current, detail = laplacian_aux(current)
        details.append(detail)
    return Pyramid(current, details)


def reconstruct_pyramid(pyramid: Pyramid) -> np.ndarray:
    image = pyramid.base
    for detail in reversed(pyramid.details):
        image = _filter(_resize_nearest(image, detail.shape[:2])) + detail
    return image


def pyramid_add(first: Pyramid, second: Pyramid) -> Pyramid:
    if len(first.details) != len(second.details):
        raise ValueError("Pyramids have different numbers of levels")
    return Pyramid(
        first.base + second.base,
        [a + b for a, b in zip(first.details, second.details)],
    )


def pyramid_multiply(first: Pyramid, second: Pyramid) -> Pyramid:
    if len(first.details) != len(second.details):
        raise ValueError("Pyramids have different numbers of levels")
    return Pyramid(
        first.base * second.base,
        [a * b for a, b in zip(first.details, second.details)],
    )


def pyramid_gaussian_blur(pyramid: Pyramid, kernel_size: int) -> Pyramid:
    sigma = max(kernel_size / 5.0, 1e-12)
    return Pyramid(
        ndimage.gaussian_filter(pyramid.base, sigma, mode="nearest"),
        [
            ndimage.gaussian_filter(detail, sigma, mode="nearest")
            for detail in pyramid.details
        ],
    )


def pyramid_image_channels(
    image: np.ndarray,
    function: Callable[..., Pyramid],
    max_levels: int = -1,
) -> list[Pyramid]:
    source = image if image.ndim == 3 else image[..., None]
    if max_levels > 0:
        return [function(source[..., channel], max_levels) for channel in range(source.shape[2])]
    return [function(source[..., channel]) for channel in range(source.shape[2])]


def pyramid_list_unary(
    pyramids: list[Pyramid],
    function: Callable[[Pyramid], Pyramid],
) -> list[Pyramid]:
    return [function(pyramid) for pyramid in pyramids]


def pyramid_list_binary(
    first: list[Pyramid],
    second: list[Pyramid],
    function: Callable[[Pyramid, Pyramid], Pyramid],
) -> list[Pyramid]:
    if len(first) != len(second):
        raise ValueError("Pyramid lists have different lengths")
    return [function(a, b) for a, b in zip(first, second)]


def pyramid_list_scalar_binary(
    pyramids: list[Pyramid],
    pyramid: Pyramid,
    function: Callable[[Pyramid, Pyramid], Pyramid],
) -> list[Pyramid]:
    return [function(item, pyramid) for item in pyramids]


def empty_pyramid(rows: int, columns: int) -> Pyramid:
    return laplacian_pyramid(np.zeros((rows, columns)))


def pyramid_blend(
    first: np.ndarray,
    second: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    if not same_image(first, second) or not similar_image(first, weight):
        raise ValueError("Input images have incompatible shapes")
    source_first = first if first.ndim == 3 else first[..., None]
    source_second = second if second.ndim == 3 else second[..., None]
    weight_first = gaussian_pyramid(weight)
    weight_second = gaussian_pyramid(1 - weight)
    output = np.empty_like(source_first, dtype=np.float64)
    for channel in range(source_first.shape[2]):
        first_weighted = pyramid_multiply(
            laplacian_pyramid(source_first[..., channel]),
            weight_first,
        )
        second_weighted = pyramid_multiply(
            laplacian_pyramid(source_second[..., channel]),
            weight_second,
        )
        output[..., channel] = reconstruct_pyramid(
            pyramid_add(first_weighted, second_weighted)
        )
    return output if first.ndim == 3 else output[..., 0]


def pyramid_blend_hdr(
    first: np.ndarray,
    second: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    inverse_gamma = 1 / 2.2
    blended = pyramid_blend(first**inverse_gamma, second**inverse_gamma, weight)
    return np.maximum(blended, 0) ** 2.2


pyrGaussGenAux = gaussian_aux
pyrGaussGen = gaussian_pyramid
pyrLapGenAux = laplacian_aux
pyrLapGen = laplacian_pyramid
pyrVal = reconstruct_pyramid
pyrAdd = pyramid_add
pyrMul = pyramid_multiply
pyrGaussianBlur = pyramid_gaussian_blur
pyrImg3 = pyramid_image_channels
pyrLst1OP = pyramid_list_unary
pyrLst2OP = pyramid_list_binary
pyrLstS2OP = pyramid_list_scalar_binary
pyrEmptyGen = empty_pyramid
pyrBlend = pyramid_blend
pyrBlendHDR = pyramid_blend_hdr

