from __future__ import annotations

import numpy as np

from .colorspace import convert_rgb_to_xyz
from .core import check_three_color, remove_specials


def float_to_rgbe(image: np.ndarray) -> np.ndarray:
    check_three_color(image)
    maximum = np.max(image, axis=2)
    low = maximum < 1e-32
    _, exponent = np.frexp(maximum)
    encoded_exponent = exponent + 128
    encoded_exponent[low] = 0
    scale = np.exp2(encoded_exponent - 128.0)
    output = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    for channel in range(3):
        values = np.floor(image[..., channel] * 256.0 / scale)
        values[low] = 0
        output[..., channel] = np.clip(values, 0, 255).astype(np.uint8)
    output[..., 3] = np.clip(encoded_exponent, 0, 255).astype(np.uint8)
    return output


def rgbe_to_float(rgbe: np.ndarray) -> np.ndarray:
    if rgbe.ndim != 3 or rgbe.shape[2] != 4:
        raise ValueError("RGBE input must have four channels")
    exponent = rgbe[..., 3].astype(np.float64) - 128.0 - 8.0
    scale = np.exp2(exponent)
    scale[rgbe[..., 3] == 0] = 0
    return (rgbe[..., :3].astype(np.float64) + 0.5) * scale[..., None]


def float_to_logluv(image: np.ndarray) -> np.ndarray:
    xyz = convert_rgb_to_xyz(image)
    output = np.zeros_like(image, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        output[..., 0] = np.clip(np.floor(256 * (np.log2(xyz[..., 1]) + 64)), 0, 65535)
        norm = np.sum(xyz, axis=2)
        x = xyz[..., 0] / norm
        y = xyz[..., 1] / norm
        uv_norm = -2 * x + 12 * y + 3
        output[..., 1] = np.clip(np.floor(410 * (4 * x / uv_norm)), 0, 255)
        output[..., 2] = np.clip(np.floor(410 * (9 * y / uv_norm)), 0, 255)
    return remove_specials(output)


def logluv_to_float(logluv: np.ndarray) -> np.ndarray:
    check_three_color(logluv)
    xyz = np.zeros_like(logluv, dtype=np.float64)
    xyz[..., 1] = np.exp2((logluv[..., 0] + 0.5) / 256.0 - 64)
    up = (logluv[..., 1] + 0.5) / 410.0
    vp = (logluv[..., 2] + 0.5) / 410.0
    norm = 6 * up - 16 * vp + 12
    x = 9 * up / norm
    y = 4 * vp / norm
    z = 1 - x - y
    factor = remove_specials(xyz[..., 1] / y)
    xyz[..., 0] = x * factor
    xyz[..., 2] = z * factor
    return convert_rgb_to_xyz(xyz, inverse=True)


float2RGBE = float_to_rgbe
RGBE2float = rgbe_to_float
float2LogLuv = float_to_logluv
LogLuv2float = logluv_to_float

