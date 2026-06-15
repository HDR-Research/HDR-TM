from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy import ndimage

from .colorspace import convert_rgb_to_xyz, convert_xyz_to_cielab
from .core import remove_specials
from .filters import bilateral_filter, bilateral_filter_full


def bilateral_noise_removal(
    image: np.ndarray,
    sigma_spatial: float = 4,
    sigma_range: float = 0.01,
) -> np.ndarray:
    source = np.asarray(image, dtype=np.float64)
    sigma_spatial = max(sigma_spatial, 4)
    sigma_range = max(sigma_range, 0.01)
    working = (
        convert_xyz_to_cielab(convert_rgb_to_xyz(source))
        if source.ndim == 3 and source.shape[2] == 3
        else source
    )
    channels = working[..., None] if working.ndim == 2 else working
    filtered = np.empty_like(channels)
    for channel in range(channels.shape[2]):
        plane = channels[..., channel]
        minimum, maximum = float(np.min(plane)), float(np.max(plane))
        delta = maximum - minimum
        if delta <= 0:
            filtered[..., channel] = plane
        else:
            normalized = (plane - minimum) / delta
            filtered[..., channel] = bilateral_filter(
                normalized,
                sigma_spatial=sigma_spatial,
                sigma_range=sigma_range,
            ) * delta + minimum
    output = filtered[..., 0] if source.ndim == 2 else filtered
    return convert_rgb_to_xyz(convert_xyz_to_cielab(output, inverse=True), inverse=True) if working is not source else output


def bilateral_separation(
    image: np.ndarray,
    sigma_spatial: float = -1,
    sigma_range: float = -1,
    domain: str = "log10",
    filter_type: str = "approx_importance",
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(image, dtype=np.float64)
    sigma_spatial = max(source.shape[:2]) * 0.02 if sigma_spatial <= 0 else sigma_spatial
    sigma_range = 0.4 if sigma_range <= 0 else sigma_range
    epsilon = 1e-6
    transforms = {
        "sigmoid": (lambda x: x / (x + 1), lambda x: np.divide(x, 1 - x, out=np.zeros_like(x), where=x < 1)),
        "log2": (lambda x: np.log2(x + epsilon), lambda x: np.exp2(x) - epsilon),
        "loge": (lambda x: np.log(x + epsilon), lambda x: np.exp(x) - epsilon),
        "log10": (lambda x: np.log10(x + epsilon), lambda x: 10**x - epsilon),
        "linear": (lambda x: x, lambda x: x),
    }
    if domain not in transforms:
        raise ValueError("Unsupported bilateral domain")
    transformed, inverse = transforms[domain]
    working = transformed(source)
    function = bilateral_filter_full if filter_type == "full" else bilateral_filter
    base_domain = function(
        working,
        None,
        sigma_spatial=sigma_spatial,
        sigma_range=sigma_range,
    )
    base = np.clip(inverse(base_domain), 0, None)
    detail = np.divide(source, base, out=np.zeros_like(source), where=base > 0)
    return remove_specials(base), remove_specials(detail)


def bitblit(
    image: np.ndarray,
    sprite: np.ndarray,
    positions: np.ndarray,
    modulation: np.ndarray | None = None,
    normalize: bool = False,
) -> np.ndarray:
    output = np.asarray(image, dtype=np.float64).copy()
    sprite = np.asarray(sprite, dtype=np.float64)
    if output.ndim != sprite.ndim or output.shape[2:] != sprite.shape[2:]:
        raise ValueError("image and sprite must have matching channels")
    positions = np.asarray(positions, dtype=int).reshape(-1, 2)
    modulation = np.ones(len(positions)) if modulation is None else np.asarray(modulation).reshape(-1)
    counter = np.zeros(output.shape[:2])
    half_height, half_width = sprite.shape[0] // 2, sprite.shape[1] // 2
    for (x, y), scale in zip(positions, modulation):
        left, top = x - half_width, y - half_height
        x0, y0 = max(left, 0), max(top, 0)
        x1, y1 = min(left + sprite.shape[1], output.shape[1]), min(top + sprite.shape[0], output.shape[0])
        if x0 >= x1 or y0 >= y1:
            continue
        sx0, sy0 = x0 - left, y0 - top
        output[y0:y1, x0:x1] += scale * sprite[sy0 : sy0 + y1 - y0, sx0 : sx0 + x1 - x0]
        counter[y0:y1, x0:x1] += 1
    if normalize:
        output /= np.maximum(counter, 1)[..., None] if output.ndim == 3 else np.maximum(counter, 1)
    return output


def compute_connected_components(
    image: np.ndarray,
    mode: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(image)
    structure = ndimage.generate_binary_structure(2, 1 if mode == 4 else 2)
    output = np.zeros(source.shape, dtype=np.int32)
    labels = np.unique(source)
    shifts = [0]
    offset = 0
    for value in labels:
        components, count = ndimage.label(source == value, structure)
        selected = components > 0
        output[selected] = components[selected] + offset
        offset += count
        shifts.append(offset)
    return output, labels, np.asarray(shifts)


def filter_firefly(image: np.ndarray, max_iterations: int = 16) -> np.ndarray:
    output = np.asarray(image, dtype=np.float64).copy()
    for _ in range(max_iterations):
        invalid = ~np.isfinite(output)
        if not np.any(invalid):
            return output
        channels = output[..., None] if output.ndim == 2 else output
        invalid_channels = invalid[..., None] if invalid.ndim == 2 else invalid
        for channel in range(channels.shape[2]):
            valid_plane = np.nan_to_num(channels[..., channel], nan=0, posinf=0, neginf=0)
            filtered = ndimage.median_filter(valid_plane, size=3, mode="nearest")
            channels[..., channel][invalid_channels[..., channel]] = filtered[invalid_channels[..., channel]]
        output = channels[..., 0] if output.ndim == 2 else channels
    return remove_specials(output)


def filter_gaussian_window(
    image: np.ndarray,
    window: float,
    scaling_factor: float = 1,
) -> np.ndarray:
    source = np.asarray(image, dtype=np.float64)
    if scaling_factor > 1:
        small = cv2.resize(source, None, fx=1 / scaling_factor, fy=1 / scaling_factor, interpolation=cv2.INTER_LINEAR)
        filtered = ndimage.gaussian_filter(small, sigma=(window / scaling_factor / 5, window / scaling_factor / 5, 0))
        return cv2.resize(filtered, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
    sigma = max(window / 5, 1e-12)
    return ndimage.gaussian_filter(source, sigma=(sigma, sigma, 0) if source.ndim == 3 else sigma, mode="nearest")


def find_name_in_list(items: list[Any], name: str) -> int:
    for index, item in enumerate(items):
        value = item.get("name") if isinstance(item, dict) else getattr(item, "name", item)
        if value == name:
            return index
    return -1


def find_neighbors(
    label: int,
    rows: np.ndarray,
    columns: np.ndarray,
    total: int,
    label_image: np.ndarray,
) -> np.ndarray:
    neighbors: set[int] = set()
    for y, x in zip(np.asarray(rows)[:total], np.asarray(columns)[:total]):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                yy, xx = int(y + dy), int(x + dx)
                value = label if not (0 <= yy < label_image.shape[0] and 0 <= xx < label_image.shape[1]) else int(label_image[yy, xx])
                if value != label:
                    neighbors.add(value)
    return np.asarray(sorted(neighbors))


def normalize_coordinates(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    center = np.mean(points, axis=0)
    centered = points - center
    scale = np.mean(np.linalg.norm(centered, axis=1)) / np.sqrt(2)
    if scale <= 0:
        raise ValueError("Cannot normalize coincident points")
    matrix = np.asarray(((1 / scale, 0, -center[0] / scale), (0, 1 / scale, -center[1] / scale), (0, 0, 1)))
    return centered / scale, matrix


def estimate_homography(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_normalized, first_matrix = normalize_coordinates(first)
    second_normalized, second_matrix = normalize_coordinates(second)
    if first_normalized.shape != second_normalized.shape or first_normalized.shape[0] < 4:
        raise ValueError("At least four matching point pairs are required")
    matrix = []
    for (x, y), (u, v) in zip(first_normalized, second_normalized):
        matrix.extend(((x, y, 1, 0, 0, 0, -u * x, -u * y, -u), (0, 0, 0, x, y, 1, -v * x, -v * y, -v)))
    _, _, vh = np.linalg.svd(np.asarray(matrix))
    normalized_h = vh[-1].reshape(3, 3)
    homography = np.linalg.inv(second_matrix) @ normalized_h @ first_matrix
    return homography / homography[2, 2]


def image_warp(image: np.ndarray, offset_map: np.ndarray, absolute: bool = False) -> np.ndarray:
    source = np.asarray(image)
    x, y = np.meshgrid(np.arange(source.shape[1]), np.arange(source.shape[0]))
    map_x = offset_map[..., 0] if absolute else x - offset_map[..., 0]
    map_y = offset_map[..., 1] if absolute else y - offset_map[..., 1]
    return cv2.remap(source, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def blend_poisson(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    iterations: int = 500,
) -> np.ndarray:
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or mask.shape != first.shape[:2]:
        raise ValueError("Images and mask have incompatible shapes")
    selected = np.asarray(mask, dtype=bool)
    output = second.copy()
    laplacian = ndimage.laplace(first, mode="nearest")
    for _ in range(iterations):
        previous = output.copy()
        neighbors = (
            np.roll(output, 1, 0)
            + np.roll(output, -1, 0)
            + np.roll(output, 1, 1)
            + np.roll(output, -1, 1)
            - laplacian
        ) / 4
        output[selected] = neighbors[selected]
        if np.max(np.abs(output - previous)) < 1e-7:
            break
    return output


def is_octave() -> bool:
    return False


bilateralNoiseRemoval = bilateral_noise_removal
bilateralSeparation = bilateral_separation
bitblit = bitblit
blendPoisson = blend_poisson
computeConnectedComponents = compute_connected_components
estimateHomography = estimate_homography
normalizeCoordinates = normalize_coordinates
filterFirefly = filter_firefly
filterGaussianWindow = filter_gaussian_window
findNameInList = find_name_in_list
findNeighbors = find_neighbors
imWarp = image_warp
isOctave = is_octave
