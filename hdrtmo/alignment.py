from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .core import image_shift, matlab_percentile, normalize_from_anything
from .io import ldrimread, write_image


def ward_compute_threshold(
    image: np.ndarray,
    percentile: float = 0.5,
    tolerance: float = 4 / 256,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(image, dtype=np.float64)
    gray = (
        (54 * source[..., 0] + 183 * source[..., 1] + 19 * source[..., 2]) / 256
        if source.ndim == 3
        else source
    )
    median = matlab_percentile(gray, percentile)
    threshold = gray > median
    exclusion = (gray < median - tolerance) | (gray > median + tolerance)
    return threshold, exclusion


def _shift_zero(image: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    dx, dy = map(int, shift)
    matrix = np.float32(((1, 0, dx), (0, 1, dy)))
    return cv2.warpAffine(
        np.asarray(image).astype(np.uint8),
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)


def ward_get_exposure_shift(
    first: np.ndarray,
    second: np.ndarray,
    percentile: float = 0.5,
    shift_bits: int = 6,
) -> np.ndarray:
    if first.shape[:2] != second.shape[:2]:
        raise ValueError("Images must have matching dimensions")
    levels = max(
        min(int(shift_bits), int(np.floor(np.log2(min(first.shape[:2]) / 8)))),
        0,
    )
    shift = np.zeros(2, dtype=int)
    for level in range(levels, -1, -1):
        scale = 2**level
        size = (
            max(int(round(first.shape[1] / scale)), 1),
            max(int(round(first.shape[0] / scale)), 1),
        )
        first_small = cv2.resize(first, size, interpolation=cv2.INTER_AREA)
        second_small = cv2.resize(second, size, interpolation=cv2.INTER_AREA)
        first_threshold, first_exclusion = ward_compute_threshold(first_small, percentile)
        second_threshold, second_exclusion = ward_compute_threshold(second_small, percentile)
        if level != levels:
            shift *= 2
        best = shift.copy()
        minimum = np.inf
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                candidate = shift + (dx, dy)
                threshold = _shift_zero(second_threshold, tuple(candidate))
                exclusion = _shift_zero(second_exclusion, tuple(candidate))
                difference = np.logical_xor(first_threshold, threshold) & first_exclusion & exclusion
                error = int(np.count_nonzero(difference))
                if error < minimum or (
                    error == minimum
                    and np.sum((candidate - shift) ** 2) < np.sum((best - shift) ** 2)
                ):
                    minimum = error
                    best = candidate
        shift = best
    return shift


def ward_simple_rotation_aux(
    first: np.ndarray,
    second: np.ndarray,
    rectangle: tuple[int, int, int, int],
) -> tuple[float, bool]:
    height, width = first.shape[:2]
    y0, y1, x0, x1 = rectangle
    y0, x0 = max(y0, 0), max(x0, 0)
    y1, x1 = min(y1, height), min(x1, width)
    first_shift = ward_get_exposure_shift(first[y0:y1, x0:x1], second[y0:y1, x0:x1])
    mirror = (height - y1, height - y0, width - x1, width - x0)
    my0, my1, mx0, mx1 = mirror
    second_shift = ward_get_exposure_shift(first[my0:my1, mx0:mx1], second[my0:my1, mx0:mx1])
    dx, dy = mx0 - x0, my0 - y0
    denominator = dx * dx + dy * dy
    if denominator <= 0:
        return 0.0, False
    rotated_x = dx + 0.5 * (second_shift[0] - first_shift[0])
    rotated_y = dy + 0.5 * (second_shift[1] - first_shift[1])
    divergence = abs(np.sqrt((rotated_x**2 + rotated_y**2) / denominator) - 1)
    angle = np.degrees(np.arctan2(rotated_y, rotated_x) - np.arctan2(dy, dx))
    return (float(angle), True) if divergence <= 0.005 else (0.0, False)


def ward_simple_rotation(first: np.ndarray, second: np.ndarray) -> tuple[float, bool]:
    height, width = first.shape[:2]
    block_height, block_width = round(height / 3), round(width / 4)
    angles = []
    for row in range(3):
        angle, valid = ward_simple_rotation_aux(
            first,
            second,
            (row * block_height, (row + 1) * block_height, 0, block_width),
        )
        if valid:
            angles.append(angle)
    if not angles:
        return 0.0, False
    threshold = np.degrees(0.07)
    if any(angle > threshold for angle in angles) and any(angle < -threshold for angle in angles):
        return 0.0, False
    return float(np.mean(angles)), True


def ward_image_alignment(
    reference: np.ndarray,
    image: np.ndarray,
    rotation: bool = True,
    percentile: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    shift = ward_get_exposure_shift(reference, image, percentile)
    aligned = image_shift(image, tuple(shift))
    information = np.zeros((3, 2), dtype=np.float64)
    information[0] = shift
    if rotation:
        angle, valid = ward_simple_rotation(reference, aligned)
        information[1] = (angle, float(valid))
        if valid and abs(angle) > 1e-9:
            matrix = cv2.getRotationMatrix2D((aligned.shape[1] / 2, aligned.shape[0] / 2), angle, 1)
            aligned = cv2.warpAffine(
                aligned,
                matrix,
                (aligned.shape[1], aligned.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REPLICATE,
            )
            final_shift = ward_get_exposure_shift(reference, aligned, percentile)
            aligned = image_shift(aligned, tuple(final_shift))
            information[2] = final_shift
    return aligned, information


def _load_stack(
    stack: np.ndarray | None,
    directory: str | Path,
    extension: str,
) -> tuple[np.ndarray, list[Path]]:
    if stack is not None and np.asarray(stack).size:
        return normalize_from_anything(stack), []
    paths = sorted(Path(directory).glob(f"*.{extension.lstrip('.')}"))
    if not paths:
        raise ValueError("The image stack is empty")
    return np.stack([ldrimread(path, double=False) for path in paths], axis=-1), paths


def _target_index(target: int | str | None, paths: list[Path], count: int) -> int:
    if target is None:
        return count // 2
    if isinstance(target, str):
        names = [path.name for path in paths]
        if target not in names:
            raise ValueError(f"Target image not found: {target}")
        return names.index(target)
    index = int(target)
    if not 0 <= index < count:
        raise ValueError("Target exposure index is out of range")
    return index


def ward_alignment(
    stack: np.ndarray | None,
    return_stack: bool = True,
    directory: str | Path = "",
    extension: str = "",
    target_exposure: int | str | None = None,
) -> np.ndarray | None:
    source, paths = _load_stack(stack, directory, extension)
    target = _target_index(target_exposure, paths, source.shape[3])
    output = np.empty_like(source)
    output[..., target] = source[..., target]
    for index in range(source.shape[3]):
        if index == target:
            continue
        output[..., index], _ = ward_image_alignment(source[..., target], source[..., index])
        if paths:
            write_image(paths[index].with_name(f"{paths[index].stem}_shifted{paths[index].suffix}"), output[..., index])
    return output if return_stack else None


def sift_image_alignment(
    reference: np.ndarray,
    image: np.ndarray,
    max_iterations: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.clip(np.asarray(reference, dtype=np.float64), 0, None)
    second = np.clip(np.asarray(image, dtype=np.float64), 0, None)
    first_gray = cv2.cvtColor(first.astype(np.float32), cv2.COLOR_RGB2GRAY) if first.ndim == 3 else first.astype(np.float32)
    second_gray = cv2.cvtColor(second.astype(np.float32), cv2.COLOR_RGB2GRAY) if second.ndim == 3 else second.astype(np.float32)
    if np.mean(first_gray) <= 0 or np.mean(second_gray) <= 0:
        raise ValueError("Images must contain positive values")
    first_gray /= max(float(np.max(first_gray)), 1e-12)
    second_gray /= max(float(np.max(second_gray)), 1e-12)
    detector = cv2.SIFT_create()
    keypoints_first, descriptors_first = detector.detectAndCompute((first_gray * 255).astype(np.uint8), None)
    keypoints_second, descriptors_second = detector.detectAndCompute((second_gray * 255).astype(np.uint8), None)
    if descriptors_first is None or descriptors_second is None:
        raise ValueError("Not enough SIFT features for alignment")
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_second, descriptors_first, k=2)
    good = [first_match for first_match, second_match in matches if first_match.distance < 0.75 * second_match.distance]
    if len(good) < 4:
        raise ValueError("Not enough matching SIFT features")
    source_points = np.float32([keypoints_second[match.queryIdx].pt for match in good])
    target_points = np.float32([keypoints_first[match.trainIdx].pt for match in good])
    homography, _ = cv2.findHomography(
        source_points,
        target_points,
        cv2.RANSAC,
        2.0,
        maxIters=max(max_iterations, 1),
    )
    if homography is None:
        raise ValueError("Homography estimation failed")
    aligned = cv2.warpPerspective(
        image,
        homography,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return aligned, homography


def residual(homography: np.ndarray, source: np.ndarray, target: np.ndarray) -> float:
    matrix = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    points = np.column_stack((source, np.ones(len(source))))
    transformed = (matrix @ points.T).T
    transformed = transformed[:, :2] / transformed[:, 2:]
    return float(np.sum((np.asarray(target) - transformed) ** 2))


def sift_alignment(
    stack: np.ndarray | None,
    return_stack: bool = True,
    directory: str | Path = "",
    extension: str = "",
    target_exposure: int | str | None = None,
) -> np.ndarray | None:
    source, paths = _load_stack(stack, directory, extension)
    target = _target_index(target_exposure, paths, source.shape[3])
    output = np.empty_like(source)
    output[..., target] = source[..., target]
    for index in range(source.shape[3]):
        if index == target:
            continue
        output[..., index], _ = sift_image_alignment(source[..., target], source[..., index])
        if paths:
            write_image(paths[index].with_name(f"{paths[index].stem}_aligned{paths[index].suffix}"), output[..., index])
    return output if return_stack else None


WardComputeThreshold = ward_compute_threshold
WardGetExpShift = ward_get_exposure_shift
WardSimpleRot = ward_simple_rotation
WardSimpleRotAux = ward_simple_rotation_aux
WardImageAlignment = ward_image_alignment
WardAlignment = ward_alignment
SiftImageAlignment = sift_image_alignment
SiftAlignment = sift_alignment
