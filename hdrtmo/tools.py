from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from .colorspace import convert_rgb_to_srgb, convert_rgb_to_xyz, convert_xyz_to_cielab, luminance
from .environment import direction_to_ll, ll_to_direction
from .generation import build_hdr, debevec_crf
from .io import hdrimwrite
from .metrics import dist_hue
from .stacks import read_ldr_stack, read_ldr_stack_info


def false_color(
    image: np.ndarray,
    compression: str = "log",
    visualize: bool = False,
    luminance_range: tuple[float, float] | None = None,
    figure: int = 1,
    title: str = "False color visualization",
    linear_labels: bool = False,
    unit: str = "Lux",
) -> tuple[np.ndarray, float]:
    del visualize, figure, title, linear_labels, unit
    values = luminance(image) if np.asarray(image).ndim == 3 else np.asarray(image)
    minimum, maximum = (
        (float(np.min(values)), float(np.max(values)))
        if luminance_range is None
        else tuple(map(float, luminance_range))
    )
    original_maximum = maximum
    epsilon = 1e-6
    transforms = {
        "log": lambda x: np.log(x + epsilon),
        "log2": lambda x: np.log2(x + epsilon),
        "log10": lambda x: np.log10(x + epsilon),
        "sigmoid": lambda x: (x / (x + 1)) ** (1 / 2.2),
        "lin": lambda x: x,
    }
    transform = transforms.get(compression, transforms["lin"])
    mapped = transform(np.clip(values, 0, None))
    low, high = float(transform(np.asarray(minimum))), float(transform(np.asarray(maximum)))
    normalized = np.clip((mapped - low) / max(high - low, epsilon), 0, 1)
    colored = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float64) / 255, original_maximum


def hdr_image_crop(
    image: np.ndarray,
    rectangle: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if rectangle is None:
        rectangle = (0, 0, image.shape[1], image.shape[0])
    x, y, width, height = map(int, rectangle)
    x, y = max(x, 0), max(y, 0)
    width, height = min(width, image.shape[1] - x), min(height, image.shape[0] - y)
    return np.asarray(image)[y : y + height, x : x + width].copy(), (x, y, width, height)


def image_white_balance(
    image: np.ndarray,
    white: np.ndarray | str = "gray_world",
) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None]:
    source = np.asarray(image, dtype=np.float64)
    position = None
    if isinstance(white, str):
        if white == "gray_world":
            color = np.mean(source, axis=(0, 1))
        elif white == "gray_world_center":
            y, x = source.shape[0] // 2, source.shape[1] // 2
            radius = max(round(max(source.shape[:2]) * 0.05), 1)
            color = np.mean(source[max(y - radius, 0) : y + radius + 1, max(x - radius, 0) : x + radius + 1], axis=(0, 1))
            position = (x, y)
        else:
            raise ValueError("white must be gray_world, gray_world_center, or an RGB color")
    else:
        color = np.asarray(white, dtype=np.float64).reshape(-1)
    scale = np.mean(color) / np.maximum(color, 1e-12)
    return source * scale, color, position


def image_color_calibration(
    source_image: np.ndarray,
    target_image: np.ndarray,
    source_colors: np.ndarray | None = None,
    target_colors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if source_colors is None or target_colors is None:
        if source_image.shape != target_image.shape:
            raise ValueError("Images must match when explicit color samples are not supplied")
        source_colors = source_image.reshape(-1, 3)[:: max(source_image.size // 30000, 1)]
        target_colors = target_image.reshape(-1, 3)[:: max(target_image.size // 30000, 1)]
    matrix, *_ = np.linalg.lstsq(np.asarray(source_colors), np.asarray(target_colors), rcond=None)
    output = np.asarray(source_image) @ matrix
    return output, matrix.T


def automatic_exposure(
    image: np.ndarray,
    display_gamma: float = 2.2,
    point: tuple[int, int] | None = None,
    radius: int = 7,
) -> tuple[np.ndarray, float]:
    del display_gamma
    luma = luminance(image)
    if point is None:
        average = float(np.mean(luma))
    else:
        x, y = point
        average = float(np.mean(luma[max(y - radius, 0) : y + radius + 1, max(x - radius, 0) : x + radius + 1]))
    exposure = 0.25 / max(average, 1e-6)
    return np.asarray(image) * exposure, exposure


def get_theta_phi(x: float, y: float, rows: int, columns: int) -> tuple[float, float]:
    return np.pi * y / rows, np.pi * ((x / columns) * 2 - 1) - np.pi / 2


def get_matrix_for_vector_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    u = np.asarray(source, dtype=np.float64) / np.linalg.norm(source)
    v = np.asarray(target, dtype=np.float64) / np.linalg.norm(target)
    cross = np.cross(u, v)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(np.dot(u, v), -1, 1))
    if sine < 1e-12:
        return np.eye(3)
    axis = cross / sine
    skew = np.asarray(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)))
    return np.eye(3) + sine * skew + (1 - cosine) * (skew @ skew)


def rotate_map(directions: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(directions) @ np.asarray(matrix).T


def rotate_ll_gui(
    image: np.ndarray,
    source_point: tuple[float, float],
    target_point: tuple[float, float],
) -> np.ndarray:
    if np.allclose(source_point, target_point):
        return np.asarray(image).copy()
    rows, columns = image.shape[:2]
    theta, phi = get_theta_phi(*source_point, rows, columns)
    theta_target, phi_target = get_theta_phi(*target_point, rows, columns)
    source = np.asarray((np.cos(phi) * np.sin(theta), np.cos(theta), np.sin(phi) * np.sin(theta)))
    target = np.asarray((np.cos(phi_target) * np.sin(theta_target), np.cos(theta_target), np.sin(phi_target) * np.sin(theta_target)))
    directions = rotate_map(ll_to_direction(rows, columns), get_matrix_for_vector_rotation(source, target))
    map_x, map_y = direction_to_ll(directions, rows, columns)
    return cv2.remap(np.asarray(image), map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)


def compute_crf_from_path(
    path: str | Path,
    output_path: str | Path,
    extension: str = "jpg",
) -> np.ndarray:
    stack, _ = read_ldr_stack(path, extension, normalize=True)
    exposures = read_ldr_stack_info(path, extension)
    response, _ = debevec_crf(stack, exposures)
    destination = Path(output_path)
    destination.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination / "crf.txt", response, fmt="%.6f")
    return response


def build_hdr_from_path(
    image_path: str | Path,
    crf_path: str | Path | None,
    output_path: str | Path,
    extension: str = "jpg",
) -> np.ndarray:
    stack, _ = read_ldr_stack(image_path, extension, normalize=True)
    exposures = read_ldr_stack_info(image_path, extension)
    if crf_path is None:
        response, _ = debevec_crf(stack, exposures)
    elif Path(crf_path).is_dir():
        response = compute_crf_from_path(crf_path, Path(output_path).parent, extension)
    else:
        response = np.loadtxt(crf_path)
    image, _ = build_hdr(stack, exposures, "LUT", response, "Deb97", "log")
    hdrimwrite(image, output_path)
    return image


def plot_channels(input_image: np.ndarray, output_image: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if input_image.shape[2] != output_image.shape[2]:
        raise ValueError("Images have different channel counts")
    curves = []
    for channel in range(input_image.shape[2]):
        x = cv2.resize(input_image[..., channel], (16, 16)).reshape(-1)
        y = cv2.resize(output_image[..., channel], (16, 16)).reshape(-1)
        order = np.argsort(x)
        curves.append((x[order], np.convolve(y[order], np.ones(16) / 16, mode="same")))
    return curves


def plot_colors(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(image).reshape(-1, 3)
    return points, points.copy()


def plot_gamut(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(image).reshape(-1, 3)
    linear = convert_rgb_to_srgb(np.asarray(image), inverse=True)
    lab = convert_xyz_to_cielab(convert_rgb_to_xyz(linear)).reshape(-1, 3)
    points = lab[:, (2, 1, 0)]
    hull = ConvexHull(points)
    return points, hull.simplices, rgb


def plot_line_column_row(
    image: np.ndarray,
    column: bool,
    channel: int,
    coordinate: tuple[int, int],
) -> np.ndarray:
    plane = image[..., channel % image.shape[2]] if channel >= 0 else luminance(image)
    x, y = coordinate
    return plane[:, x] if column else plane[y, :]


def show_hue_diff(distorted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    _, difference = dist_hue(distorted, reference)
    return false_color(difference, "lin", luminance_range=(0, 1))[0]


AExposureGUI = automatic_exposure
FalseColor = false_color
RotateLLGUI = rotate_ll_gui
getThetaPhi = get_theta_phi
getMatrixForVectorRotation = get_matrix_for_vector_rotation
RotateMap = rotate_map
buildHDRFromPath = build_hdr_from_path
computeCRFFromPath = compute_crf_from_path
hdrimCrop = hdr_image_crop
imColorCalibration = image_color_calibration
imWhiteBalance = image_white_balance
plotChannels = plot_channels
plotColors = plot_colors
plotGamut = plot_gamut
plotLineColRow = plot_line_column_row
plotsRGB = plot_gamut
showHueDiff = show_hue_diff
