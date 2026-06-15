from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from .alignment import ward_compute_threshold
from .io import ldrimread
from .pyramids import gaussian_pyramid, laplacian_pyramid, pyramid_add, pyramid_multiply, reconstruct_pyramid
from .tmo_utils import mertens_contrast, mertens_saturation, mertens_well_exposedness


def gallo_reference_image(
    stack: np.ndarray | None,
    directory: str | Path = "",
    extension: str = "",
) -> int:
    if stack is not None and np.asarray(stack).size:
        source = np.asarray(stack)
        frames = [source[..., index] for index in range(source.shape[3])]
    else:
        paths = sorted(Path(directory).glob(f"*.{extension.lstrip('.')}"))
        if not paths:
            raise ValueError("The image stack is empty")
        frames = [ldrimread(path) for path in paths]
    scores = []
    for frame in frames:
        scores.append(
            np.count_nonzero(
                (np.max(frame, axis=2) >= 248 / 255)
                | (np.min(frame, axis=2) <= 7 / 255)
            )
        )
    return int(np.argmin(scores))


def _disk(radius: int) -> np.ndarray:
    radius = max(int(radius), 0)
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def pece_kautz_move_mask(
    stack: np.ndarray,
    iterations: int = 1,
    erosion_size: int = 3,
    dilation_size: int = 17,
    ward_percentile: float = 0.5,
) -> tuple[np.ndarray, int]:
    source = np.asarray(stack)
    if source.ndim != 4 or source.shape[3] < 2:
        raise ValueError("An HxWxCxN stack with at least two frames is required")
    threshold_sum = np.zeros(source.shape[:2], dtype=np.int32)
    for index in range(source.shape[3]):
        threshold, _ = ward_compute_threshold(source[..., index], ward_percentile)
        threshold_sum += threshold
    moving = (threshold_sum > 0) & (threshold_sum < source.shape[3])
    for _ in range(max(iterations, 0)):
        moving = ndimage.binary_dilation(moving, structure=_disk(dilation_size))
        moving = ndimage.binary_erosion(moving, structure=_disk(erosion_size))
    labels, count = ndimage.label(moving, ndimage.generate_binary_structure(2, 1))
    labels = labels.astype(np.int32)
    labels[~moving] = -1
    return labels, int(count)


def pece_kautz_merge(
    stack: np.ndarray,
    weights: np.ndarray | tuple[float, float, float] = (1, 1, 1),
    iterations: int = 1,
    erosion_size: int = 3,
    dilation_size: int = 17,
    ward_percentile: float = 0.5,
) -> np.ndarray:
    source = np.asarray(stack)
    if np.issubdtype(source.dtype, np.integer):
        source = source.astype(np.float64) / np.iinfo(source.dtype).max
    else:
        source = source.astype(np.float64)
    if source.ndim != 4:
        raise ValueError("Stack must have shape HxWxCxN")
    exposed_weight, saturation_weight, contrast_weight = np.asarray(weights, dtype=float)
    frame_weights = np.empty(source.shape[:2] + (source.shape[3],))
    for index in range(source.shape[3]):
        frame = source[..., index]
        weight = np.ones(source.shape[:2])
        if exposed_weight > 0:
            weight *= mertens_well_exposedness(frame) ** exposed_weight
        if saturation_weight > 0:
            weight *= mertens_saturation(frame) ** saturation_weight
        if contrast_weight > 0:
            weight *= mertens_contrast(np.mean(frame, axis=2)) ** contrast_weight
        frame_weights[..., index] = weight + 1e-12

    movement, count = pece_kautz_move_mask(
        source,
        iterations,
        erosion_size,
        dilation_size,
        ward_percentile,
    )
    for region in range(1, count + 1):
        selected = movement == region
        best = int(np.argmax(np.sum(frame_weights[selected], axis=0)))
        frame_weights[selected, :] = 0
        frame_weights[selected, best] = 1
    frame_weights /= np.maximum(np.sum(frame_weights, axis=2, keepdims=True), 1e-12)

    output = np.empty(source.shape[:3])
    for channel in range(source.shape[2]):
        combined = None
        for index in range(source.shape[3]):
            weighted = pyramid_multiply(
                laplacian_pyramid(source[..., channel, index]),
                gaussian_pyramid(frame_weights[..., index]),
            )
            combined = weighted if combined is None else pyramid_add(combined, weighted)
        output[..., channel] = reconstruct_pyramid(combined)
    maximum = float(np.max(output))
    return np.clip(output / maximum if maximum > 0 else output, 0, 1)


GalloReferenceImage = gallo_reference_image
PeceKautzMoveMask = pece_kautz_move_mask
PeceKautzMerge = pece_kautz_merge
