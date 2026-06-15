from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from .colorspace import luminance
from .generation import apply_crf
from .io import read_raw, read_raw_info
from .tmo_utils import exposure_histogram_sampling


def sort_stack(
    stack: np.ndarray,
    exposures: np.ndarray,
    sorting: str = "ascend",
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(exposures)
    if sorting == "descend":
        order = order[::-1]
    elif sorting != "ascend":
        raise ValueError("sorting must be ascend or descend")
    return stack[..., order], np.asarray(exposures)[order]


def check_monotonicity(sort_index: np.ndarray, values: np.ndarray) -> int:
    ordered = np.asarray(values)[np.asarray(sort_index)]
    return int(np.sum([np.all(ordered[:-1, channel] > ordered[1:, channel]) for channel in range(ordered.shape[1])]))


def read_ldr_stack(
    directory: str | Path,
    extension: str,
    normalize: bool = False,
    to_float: bool = True,
) -> tuple[np.ndarray, float]:
    paths = sorted(Path(directory).glob(f"*.{extension.lstrip('.')}"))
    if not paths:
        raise ValueError("The stack is empty")
    frames = []
    norm = 1.0
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise ValueError(f"Cannot read {path}")
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if np.issubdtype(frame.dtype, np.integer):
            norm = float(np.iinfo(frame.dtype).max)
        frames.append(frame.astype(np.float32) if to_float else frame)
    stack = np.stack(frames, axis=-1)
    return (stack / norm if normalize else stack), norm


def write_ldr_stack(
    stack: np.ndarray,
    name: str | Path,
    extension: str,
) -> None:
    prefix = Path(name)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for index in range(stack.shape[3]):
        frame = stack[..., index]
        path = prefix.parent / f"{prefix.name}_{100001 + index}.{extension}"
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.rint(np.clip(frame, 0, 1) * 255).astype(np.uint8)
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), frame):
            raise ValueError(f"Cannot write {path}")


def read_ldr_stack_info(directory: str | Path, extension: str) -> np.ndarray:
    paths = sorted(Path(directory).glob(f"*.{extension.lstrip('.')}"))
    exposures = []
    for path in paths:
        with Image.open(path) as image:
            exif = image.getexif()
            exposure_time = float(exif.get(33434, 1.0))
            aperture = float(exif.get(33437, 1.0))
            iso = float(exif.get(34855, 1.0))
        exposures.append(iso * exposure_time / (12.5 * aperture**2))
    return np.asarray(exposures)


def compute_stack_histogram(stack: np.ndarray) -> np.ndarray:
    channels, count = stack.shape[2], stack.shape[3]
    output = np.zeros((256, channels, count))
    for index in range(count):
        for channel in range(channels):
            values = stack[..., channel, index]
            if np.issubdtype(values.dtype, np.floating):
                values = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
            elif values.dtype == np.uint16:
                values = np.rint(values / 255).astype(np.uint8)
            output[:, channel, index] = np.bincount(values.reshape(-1), minlength=256)
    return output


def grossberg_sampling(histogram_stack: np.ndarray, samples: int = 256) -> np.ndarray:
    samples = 256 if samples < 1 else samples
    cdf = np.cumsum(histogram_stack, axis=0)
    cdf /= np.maximum(cdf[-1:, ...], 1)
    quantiles = np.linspace(0, 1, samples)
    output = np.empty((samples, histogram_stack.shape[2], histogram_stack.shape[1]))
    for i, quantile in enumerate(quantiles):
        for channel in range(histogram_stack.shape[1]):
            for frame in range(histogram_stack.shape[2]):
                output[i, frame, channel] = np.argmin(np.abs(cdf[:, channel, frame] - quantile))
    return output


def spatial_sampling(
    stack: np.ndarray,
    sort_index: np.ndarray,
    samples: int,
    kind: str,
) -> np.ndarray:
    height, width, channels, count = stack.shape
    if kind == "RandomSpatial":
        rng = np.random.default_rng()
        xs = rng.integers(0, width, samples)
        ys = rng.integers(0, height, samples)
    elif kind == "RegularSpatial":
        factor = round(np.sqrt(samples) + 1)
        xs_grid, ys_grid = np.meshgrid(
            np.arange(0, width, max(int(np.ceil(width / factor)), 1)),
            np.arange(0, height, max(int(np.ceil(height / factor)), 1)),
        )
        xs, ys = xs_grid.reshape(-1), ys_grid.reshape(-1)
    else:
        raise ValueError("kind must be RandomSpatial or RegularSpatial")
    selected = []
    for x, y in zip(xs, ys):
        values = stack[y, x, :, :].T
        if check_monotonicity(sort_index, values) > 0:
            selected.append(values)
    return np.stack(selected) if selected else np.empty((0, count, channels))


def stack_subsampling(
    stack: np.ndarray,
    exposures: np.ndarray,
    samples: int = 256,
    strategy: str = "Grossberg",
    outliers_percentage: float = 0,
) -> np.ndarray:
    sort_index = np.argsort(exposures)[::-1]
    if strategy == "Grossberg":
        output = grossberg_sampling(compute_stack_histogram(stack), samples)
    else:
        output = np.rint(spatial_sampling(stack, sort_index, samples, strategy) * 255)
    if outliers_percentage > 0:
        output[(output < outliers_percentage * 255) | (output > (1 - outliers_percentage) * 255)] = -1
    return output


def create_ldr_stack_from_hdr(
    image: np.ndarray,
    fstop_distance: np.ndarray | float = 1,
    sampling_mode: str = "histogram",
    linearization: str = "gamma",
    function: np.ndarray | float = 2.2,
) -> tuple[np.ndarray, np.ndarray]:
    luma = luminance(image)
    if sampling_mode == "histogram":
        exposures = np.exp2(exposure_histogram_sampling(image, 8, float(fstop_distance)))
    elif sampling_mode == "selected":
        exposures = np.exp2(np.asarray(fstop_distance))
    elif sampling_mode == "uniform":
        positive = luma[luma > 0]
        minimum, maximum = np.min(positive), np.max(positive)
        low = -(np.floor(np.log2(maximum + 1e-6) + 1))
        high = -(np.ceil(np.log2(minimum + 1e-6) + 1) + 8)
        if high < low:
            high = -np.ceil(np.log2(minimum + 1e-6) + 1)
        exposures = np.exp2(np.arange(low, high + float(fstop_distance), float(fstop_distance)))
    elif sampling_mode == "zone":
        zones = ndimage.median_filter(np.floor(np.log2(luma + 2**-20)), size=5)
        zone_means = np.asarray([np.mean(luma[zones == zone]) for zone in np.unique(zones)])
        exposures = 1 / (2 * np.maximum(zone_means, np.finfo(float).eps))
    else:
        raise ValueError("Unsupported sampling mode")
    frames = [np.clip(apply_crf(image * exposure, linearization, function), 0, 1) for exposure in exposures]
    return np.stack(frames, axis=-1), exposures


def read_raw_stack(
    directory: str | Path,
    extension: str,
    saturation_level: int = 2**12 - 1,
) -> np.ndarray:
    paths = sorted(Path(directory).glob(f"*.{extension.lstrip('.')}"))
    if not paths:
        raise ValueError("The RAW stack is empty")
    saturation = saturation_level
    if saturation < 0:
        levels = [read_raw(path, -1)[2] for path in paths]
        saturation = min(levels)
    frames = [read_raw(path, saturation)[0].astype(np.float32) / 65535 for path in paths]
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError("RAW stack frames have different dimensions")
    return np.stack(frames, axis=-1)


def read_raw_stack_info(directory: str | Path, extension: str) -> np.ndarray:
    paths = sorted(Path(directory).glob(f"*.{extension.lstrip('.')}"))
    exposures = []
    for path in paths:
        metadata = read_raw_info(path)
        exposure_time = float(metadata.get("ExposureTime", 1) or 1)
        aperture = float(metadata.get("FNumber", 1) or 1)
        iso = float(metadata.get("ISOSpeedRatings", 1) or 1)
        exposures.append(iso * exposure_time / (12.5 * aperture**2))
    return np.asarray(exposures)


SortStack = sort_stack
checkMonotonicity = check_monotonicity
ReadLDRStack = read_ldr_stack
ReadLDRStackInfo = read_ldr_stack_info
WriteLDRStack = write_ldr_stack
ComputeLDRStackHistogram = compute_stack_histogram
ReadLDRStackHistogram = compute_stack_histogram
GrossbergSampling = grossberg_sampling
SpatialSampling = spatial_sampling
LDRStackSubSampling = stack_subsampling
CreateLDRStackFromHDR = create_ldr_stack_from_hdr
ReadRAWStack = read_raw_stack
ReadRAWStackInfo = read_raw_stack_info
