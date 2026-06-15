from __future__ import annotations

import numpy as np
import cv2

from .core import image_shift


def bilateral_filter_full(
    image: np.ndarray,
    edges: np.ndarray | None,
    sigma_spatial: float,
    sigma_range: float,
) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    edges = image if edges is None else np.asarray(edges, dtype=np.float64)
    channels = 1 if image.ndim == 2 else image.shape[2]
    output = np.zeros_like(image)
    weight_sum = np.zeros(image.shape[:2])
    radius = max(round(max(sigma_spatial * 5, 1) / 2), 1)
    spatial_denominator = 2 * sigma_spatial**2
    range_denominator = 2 * sigma_range**2
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = image_shift(image, (dx, dy))
            shifted_edges = image_shift(edges, (dx, dy))
            difference = shifted_edges - edges
            range_distance = (
                np.sum(difference**2, axis=2)
                if difference.ndim == 3
                else difference**2
            )
            weight = np.exp(
                -(dx**2 + dy**2) / spatial_denominator
                - range_distance / range_denominator
            )
            output += shifted * (weight[..., None] if channels > 1 else weight)
            weight_sum += weight
    if channels > 1:
        return np.divide(
            output,
            weight_sum[..., None],
            out=image.copy(),
            where=weight_sum[..., None] > 1e-9,
        )
    return np.divide(output, weight_sum, out=image.copy(), where=weight_sum > 1e-9)


def bilateral_filter(
    data: np.ndarray,
    edge: np.ndarray | None = None,
    edge_min: float | None = None,
    edge_max: float | None = None,
    sigma_spatial: float | None = None,
    sigma_range: float | None = None,
    sampling_spatial: float | None = None,
    sampling_range: float | None = None,
) -> np.ndarray:
    del sampling_spatial, sampling_range
    edge = data if edge is None else edge
    edge_min = float(np.min(edge)) if edge_min is None else edge_min
    edge_max = float(np.max(edge)) if edge_max is None else edge_max
    sigma_spatial = min(data.shape[:2]) / 16 if sigma_spatial is None else sigma_spatial
    sigma_range = (edge_max - edge_min) / 10 if sigma_range is None else sigma_range
    if edge is data or np.shares_memory(np.asarray(edge), np.asarray(data)):
        source = np.asarray(data, dtype=np.float32)
        return cv2.bilateralFilter(
            source,
            d=0,
            sigmaColor=max(float(sigma_range), 1e-12),
            sigmaSpace=max(float(sigma_spatial), 1e-12),
            borderType=cv2.BORDER_REPLICATE,
        ).astype(np.float64)
    return bilateral_filter_full(data, edge, sigma_spatial, max(sigma_range, 1e-12))


bilateralFilter = bilateral_filter
bilateralFilterFull = bilateral_filter_full
bilateralFilterS = bilateral_filter_full
bilateralFilterSI = bilateral_filter_full
