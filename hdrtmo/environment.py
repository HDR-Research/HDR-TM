from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .colorspace import luminance
from .core import remove_specials
from .tmo import normalize_tmo


def _normalize(directions: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(directions, axis=2, keepdims=True)
    return np.divide(directions, norm, out=np.zeros_like(directions), where=norm > 0)


def angular_to_direction(rows: int, columns: int) -> np.ndarray:
    x, y = np.meshgrid(np.arange(columns) / columns, np.arange(rows) / rows)
    theta = np.arctan2(1 - 2 * y, 2 * x - 1)
    phi = np.pi * np.hypot(2 * x - 1, 2 * y - 1)
    sin_phi = np.sin(phi)
    return np.stack(
        (np.cos(theta) * sin_phi, np.sin(theta) * sin_phi, -np.cos(phi)),
        axis=2,
    )


def angular_mask(rows: int, columns: int) -> np.ndarray:
    x, y = np.meshgrid(
        (np.arange(columns) + 1) / columns * 2 - 1,
        (np.arange(rows) + 1) / rows * 2 - 1,
    )
    return np.repeat((np.hypot(x, y) <= 1)[..., None], 3, axis=2).astype(float)


def ll_to_direction(rows: int, columns: int) -> np.ndarray:
    x, y = np.meshgrid(np.arange(columns), np.arange(rows))
    phi = np.pi * ((x / columns) * 2 - 1) - np.pi / 2
    theta = np.pi * y / rows
    sin_theta = np.sin(theta)
    return np.stack(
        (np.cos(phi) * sin_theta, np.cos(theta), np.sin(phi) * sin_theta),
        axis=2,
    )


def direction_to_ll(
    directions: np.ndarray,
    rows: int,
    columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    directions = _normalize(np.asarray(directions, dtype=np.float64))
    x = (1 + np.arctan2(directions[..., 0], -directions[..., 2]) / np.pi) * columns / 2
    y = np.arccos(np.clip(directions[..., 1], -1, 1)) * rows / np.pi
    return remove_specials(x), remove_specials(y)


def direction_to_angular(
    directions: np.ndarray,
    rows: int,
    columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    directions = _normalize(np.asarray(directions, dtype=np.float64))
    planar = np.hypot(directions[..., 0], directions[..., 1])
    radius = np.divide(
        np.arccos(np.clip(-directions[..., 2], -1, 1)),
        2 * np.pi * planar,
        out=np.zeros_like(planar),
        where=planar > 0,
    )
    return (
        remove_specials((0.5 + radius * directions[..., 0]) * columns),
        remove_specials((0.5 - radius * directions[..., 1]) * rows),
    )


def direction_to_spherical(
    directions: np.ndarray,
    rows: int,
    columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    directions = _normalize(np.asarray(directions, dtype=np.float64))
    planar = np.hypot(directions[..., 0], directions[..., 1])
    radius = np.divide(
        np.sin(0.5 * np.arccos(np.clip(-directions[..., 2], -1, 1))),
        2 * planar,
        out=np.zeros_like(planar),
        where=planar > 0,
    )
    x = np.rint((0.5 + radius * directions[..., 0]) * columns)
    y = np.rint((0.5 - radius * directions[..., 1]) * rows)
    return remove_specials(x), remove_specials(y)


def cube_map_to_direction(rows: int, columns: int) -> np.ndarray:
    tile = int(round(max(rows / 4, columns / 3)))
    directions = np.zeros((rows, columns, 3), dtype=np.float64)
    u, v = np.meshgrid((np.arange(tile) + 1) / tile * 2 - 1, (np.arange(tile) + 1) / tile * 2 - 1)
    base = _normalize(np.stack((u, np.ones_like(u), v), axis=2))
    placements = (
        (slice(0, tile), slice(tile, 2 * tile), (1, 2, -3)),
        (slice(2 * tile, 3 * tile), slice(tile, 2 * tile), (1, -2, 3)),
        (slice(tile, 2 * tile), slice(tile, 2 * tile), (1, -3, -2)),
        (slice(3 * tile, 4 * tile), slice(tile, 2 * tile), (1, 3, 2)),
        (slice(tile, 2 * tile), slice(0, tile), (-2, -3, -1)),
        (slice(tile, 2 * tile), slice(2 * tile, 3 * tile), (2, -3, 1)),
    )
    for row_slice, column_slice, mapping in placements:
        if row_slice.stop > rows or column_slice.stop > columns:
            continue
        for channel, source in enumerate(mapping):
            directions[row_slice, column_slice, channel] = np.sign(source) * base[..., abs(source) - 1]
    return directions


def direction_to_cube_map(
    directions: np.ndarray,
    rows: int,
    columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    d = _normalize(np.asarray(directions, dtype=np.float64))
    tile = (rows / 4 + columns / 3) / 2
    x = np.zeros(d.shape[:2])
    y = np.zeros(d.shape[:2])
    dominant = np.argmax(np.abs(d), axis=2)
    signs = np.sign(d)

    faces = (
        ((dominant == 1) & (signs[..., 1] > 0), 1, 0, 2, 1.5, 0.5, 0.5, 0.5),
        ((dominant == 0) & (signs[..., 0] > 0), 0, 2, 1, 1.5, 0.5, 1.5, 0.5),
        ((dominant == 2) & (signs[..., 2] > 0), 2, 0, 1, 1.5, 0.5, 2.5, -0.5),
        ((dominant == 1) & (signs[..., 1] < 0), 1, 0, 2, 1.5, -0.5, 3.5, -0.5),
        ((dominant == 0) & (signs[..., 0] < 0), 0, 2, 1, 0.5, 0.5, 1.5, -0.5),
        ((dominant == 2) & (signs[..., 2] < 0), 2, 0, 1, 2.5, -0.5, 1.5, -0.5),
    )
    for mask, primary, a, b, x0, xa, y0, yb in faces:
        denominator = d[..., primary]
        x[mask] = x0 + xa * d[..., a][mask] / denominator[mask]
        y[mask] = y0 + yb * d[..., b][mask] / denominator[mask]
    return remove_specials(x * tile), remove_specials(np.flipud(y * tile))


def cross_mask(rows: int, columns: int) -> np.ndarray:
    tile = int(round(max(rows / 4, columns / 3)))
    mask = np.zeros((rows, columns, 3), dtype=float)
    for row, column in ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)):
        mask[row * tile : min((row + 1) * tile, rows), column * tile : min((column + 1) * tile, columns)] = 1
    return mask


def change_mapping(image: np.ndarray, mapping_in: str, mapping_out: str) -> np.ndarray:
    aliases = {"ll": "longitudelatitude", "longitudelatitude": "longitudelatitude", "angular": "angular", "cubemap": "cubemap"}
    source_name = aliases.get(mapping_in.lower())
    target_name = aliases.get(mapping_out.lower())
    if source_name is None or target_name is None:
        raise ValueError("Mapping must be Angular, LongitudeLatitude/LL, or CubeMap")
    if source_name == target_name:
        return np.asarray(image).copy()

    height, width = image.shape[:2]
    size = max(height, width) // 2
    if target_name == "longitudelatitude":
        directions = ll_to_direction(size, size * 2)
    elif target_name == "angular":
        directions = angular_to_direction(size, size)
    else:
        directions = cube_map_to_direction(size * 4, size * 3)
    converters = {
        "longitudelatitude": direction_to_ll,
        "angular": direction_to_angular,
        "cubemap": direction_to_cube_map,
    }
    map_x, map_y = converters[source_name](directions, height, width)
    border = cv2.BORDER_WRAP if source_name == "longitudelatitude" else cv2.BORDER_CONSTANT
    output = cv2.remap(
        np.asarray(image, dtype=np.float64),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=border,
    )
    if target_name == "cubemap":
        output *= cross_mask(*output.shape[:2])
    elif target_name == "angular":
        output *= angular_mask(*output.shape[:2])
    return np.clip(remove_specials(output), 0, None)


def ll_descriptor(image: np.ndarray, normalize: bool = False) -> np.ndarray:
    descriptor = np.sum(luminance(image), axis=0)
    maximum = float(np.max(descriptor))
    return descriptor / maximum if normalize and maximum > 0 else descriptor


def rotate_y_ll(image: np.ndarray, angle: float) -> np.ndarray:
    return np.roll(image, int(round(angle * image.shape[1] / 360)), axis=1)


def align_ll_panoramas(
    image: np.ndarray,
    reference: np.ndarray,
    visualize: bool = False,
) -> tuple[np.ndarray, int, float]:
    del visualize
    if image.shape != reference.shape:
        raise ValueError("Panoramas must have equal shapes")
    first = ll_descriptor(normalize_tmo(image), True)
    second = ll_descriptor(normalize_tmo(reference), True)
    errors = np.asarray([np.sum((np.roll(first, shift) - second) ** 2) for shift in range(image.shape[1])])
    rotation = int(np.argmin(errors))
    return np.roll(image, rotation, axis=1), rotation, float(errors[rotation])


def cross_cutter(
    image: np.ndarray,
    name: str | Path = "output_cube_map",
    image_format: str = "hdr",
) -> dict[str, np.ndarray]:
    tile = int(round(image.shape[1] / 3))
    faces = {
        "POS_Y": np.rot90(image[:tile, tile : 2 * tile], 3),
        "POS_X": image[tile : 2 * tile, tile : 2 * tile],
        "NEG_Y": np.rot90(image[2 * tile : 3 * tile, tile : 2 * tile], 1),
        "NEG_X": np.flip(image[3 * tile : 4 * tile, tile : 2 * tile], axis=(0, 1)),
        "POS_Z": image[tile : 2 * tile, :tile],
        "NEG_Z": image[tile : 2 * tile, 2 * tile : 3 * tile],
    }
    prefix = Path(name)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for face_name, face in faces.items():
        encoded = cv2.cvtColor(face.astype(np.float32), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(prefix.parent / f"{prefix.name}_{face_name}.{image_format}"), encoded):
            raise ValueError(f"Cannot write cubemap face {face_name}")
    return faces


Angular2Direction = angular_to_direction
AngularMask = angular_mask
ChangeMapping = change_mapping
CrossCutter = cross_cutter
CrossMask = cross_mask
CubeMap2Direction = cube_map_to_direction
Direction2Angular = direction_to_angular
Direction2CubeMap = direction_to_cube_map
Direction2LL = direction_to_ll
Direction2Spherical = direction_to_spherical
LL2Direction = ll_to_direction
LLDescriptor = ll_descriptor
RotateYLL = rotate_y_ll
AlignLLPanoramas = align_ll_panoramas
