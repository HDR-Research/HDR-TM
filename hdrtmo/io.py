from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from PIL import Image

try:
    import pyexr
except ImportError:
    pyexr = None

try:
    import rawpy
except ImportError:
    rawpy = None


READER_PRESETS = {
    "hdtv1k_pq_rec2020": {
        "png_mode": "normalized",
        "transfer": "pq",
        "primaries": "rec2020",
        "linear_scale": 1.0,
    },
    "hdrps_linear100_rec709": {
        "png_mode": "normalized",
        "transfer": "linear_times_100",
        "primaries": "rec709",
        "linear_scale": 1.0,
    },
    "lvzhdr_linear100_rec709": {
        "png_mode": "normalized",
        "transfer": "linear_times_100",
        "primaries": "rec709",
        "linear_scale": 1.0,
    },
    "hdrps_linear_rec709": {
        "png_mode": "normalized",
        "transfer": "linear",
        "primaries": "rec709",
        "linear_scale": 1.0,
    },
    "lvzhdr_linear_rec709": {
        "png_mode": "normalized",
        "transfer": "linear",
        "primaries": "rec709",
        "linear_scale": 1.0,
    },
}


@dataclass(frozen=True)
class ReaderConfig:
    preset: str = "hdtv1k_pq_rec2020"
    png_mode: str = "normalized"
    transfer: str = "pq"
    primaries: str = "rec2020"
    linear_scale: float = 1.0
    remove_specials_value: float = 0.0

    def __post_init__(self) -> None:
        if self.preset != "custom" and self.preset not in READER_PRESETS:
            choices = ", ".join([*READER_PRESETS, "custom"])
            raise ValueError(f"reader.preset must be one of: {choices}")
        if self.png_mode not in {"raw", "normalized"}:
            raise ValueError("reader.png_mode must be 'raw' or 'normalized'")
        if self.transfer not in {"linear", "linear_times_100", "pq"}:
            raise ValueError(
                "reader.transfer must be 'linear', 'linear_times_100', or 'pq'"
            )
        if self.primaries not in {"rec709", "rec2020"}:
            raise ValueError("reader.primaries must be 'rec709' or 'rec2020'")
        if self.linear_scale <= 0.0:
            raise ValueError("reader.linear_scale must be positive")

    @property
    def resolved(self) -> dict[str, str | float]:
        if self.preset == "custom":
            return {
                "png_mode": self.png_mode,
                "transfer": self.transfer,
                "primaries": self.primaries,
                "linear_scale": self.linear_scale,
            }
        return READER_PRESETS[self.preset]


def remove_specials(image: np.ndarray, value: float = 0.0) -> np.ndarray:
    return np.nan_to_num(
        image,
        copy=False,
        nan=value,
        posinf=value,
        neginf=value,
    )


def pq_to_linear(signal: np.ndarray) -> np.ndarray:
    """Decode an ST 2084/PQ signal to absolute luminance in cd/m^2."""
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    signal = np.clip(np.asarray(signal, dtype=np.float64), 0.0, 1.0)
    powered = np.power(signal, 1.0 / m2)
    numerator = np.maximum(powered - c1, 0.0)
    denominator = c2 - c3 * powered
    ratio = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    return 10000.0 * np.power(ratio, 1.0 / m1)


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as pfm:
        header = pfm.readline().decode("ascii").rstrip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"{path} is not a PFM file")
        color = header == "PF"
        width, height = map(int, pfm.readline().decode("ascii").split())
        scale = float(pfm.readline().decode("ascii").strip())
        dtype = "<f4" if scale < 0 else ">f4"
        channels = 3 if color else 1
        data = np.fromfile(pfm, dtype=dtype)

    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f"Invalid PFM payload: expected {expected} values")
    shape = (height, width, channels) if color else (height, width)
    return np.flipud(data.reshape(shape)).astype(np.float64)


def read_pfm(
    filename: str | Path,
    photoshop_compatibility: bool = False,
) -> np.ndarray:
    image = _read_pfm(Path(filename))
    return np.flipud(image) if photoshop_compatibility else image


def write_pfm(
    image: np.ndarray,
    filename: str | Path,
    endian_mode: str = "l",
    photoshop_compatibility: bool = False,
) -> bool:
    if endian_mode not in {"l", "b"}:
        raise ValueError("endian_mode must be 'l' or 'b'")
    source = np.asarray(image)
    if source.ndim not in {2, 3} or (source.ndim == 3 and source.shape[2] not in {1, 3}):
        raise ValueError("PFM requires one or three channels")
    if source.ndim == 3 and source.shape[2] == 1:
        source = source[..., 0]
    if photoshop_compatibility:
        source = np.flipud(source)
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype("<f4" if endian_mode == "l" else ">f4")
    scale = -1.0 if endian_mode == "l" else 1.0
    header = "PF" if source.ndim == 3 else "Pf"
    with path.open("wb") as output:
        output.write(f"{header}\n{source.shape[1]} {source.shape[0]}\n{scale:.6f}\n".encode("ascii"))
        np.flipud(source).astype(dtype).tofile(output)
    return True


def read_rgbe(filename: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(filename)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read RGBE image: {path}")
    image = _to_rgb(image).astype(np.float64)
    metadata: dict[str, Any] = {"loaded": True, "exposure": 1.0, "gamma": 1.0}
    with path.open("rb") as source:
        for raw_line in source:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if not line:
                break
            if line.startswith("EXPOSURE="):
                metadata["exposure"] = float(line.split("=", 1)[1])
            elif line.startswith("GAMMA="):
                metadata["gamma"] = float(line.split("=", 1)[1])
    return image, metadata


def write_rgbe(
    image: np.ndarray,
    filename: str | Path,
    metadata: dict[str, Any] | None = None,
) -> bool:
    del metadata  # OpenCV writes standards-compliant RLE RGBE without custom header fields.
    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("RGBE encoding requires three color channels")
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), cv2.cvtColor(source, cv2.COLOR_RGB2BGR)))


def write_rgbe_line(buffer_line: np.ndarray, output: Any) -> None:
    values = np.asarray(buffer_line, dtype=np.uint8).reshape(-1)
    index = 0
    while index < values.size:
        run = 1
        while index + run < values.size and run < 127 and values[index + run] == values[index]:
            run += 1
        if run >= 4:
            output.write(bytes((128 + run, int(values[index]))))
            index += run
            continue
        start = index
        index += run
        while index < values.size and index - start < 128:
            next_run = 1
            while index + next_run < values.size and next_run < 4 and values[index + next_run] == values[index]:
                next_run += 1
            if next_run >= 4:
                break
            index += next_run
        output.write(bytes((index - start,)))
        output.write(values[start:index].tobytes())


def ldrimread(filename: str | Path, double: bool = True) -> np.ndarray:
    image = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read image: {filename}")
    image = _to_rgb(image)
    scale = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0
    dtype = np.float64 if double else np.float32
    return image.astype(dtype) / scale


def read_crfs(filename: str | Path) -> tuple[np.ndarray, np.ndarray]:
    tokens = Path(filename).read_text(encoding="utf-8", errors="ignore").split()
    irradiance, brightness = [], []
    index = 0
    while index + 4 <= len(tokens):
        index += 2  # camera model and graph identifier
        index += 2  # irradiance labels
        try:
            irradiance.append([float(value) for value in tokens[index : index + 1024]])
        except ValueError as error:
            raise ValueError("Invalid CRF irradiance block") from error
        index += 1024
        index += 2  # brightness labels
        try:
            brightness.append([float(value) for value in tokens[index : index + 1024]])
        except ValueError as error:
            raise ValueError("Invalid CRF brightness block") from error
        index += 1024
    return np.asarray(irradiance), np.asarray(brightness)


def read_hdr(
    filename: str | Path,
    config: ReaderConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = config or ReaderConfig()
    resolved = config.resolved
    path = Path(filename)
    extension = path.suffix.lower()

    if extension == ".pfm":
        image = _read_pfm(path)
    elif extension == ".exr" and pyexr is not None:
        image = pyexr.read(str(path)).astype(np.float32)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        image = _to_rgb(image)

    source_dtype = image.dtype
    image = image.astype(np.float64, copy=False)
    metadata: dict[str, Any] = {
        "loaded": True,
        "format": extension.lstrip("."),
        "source_dtype": str(source_dtype),
        "preset": config.preset,
        "transfer": resolved["transfer"],
        "primaries": resolved["primaries"],
    }

    if extension == ".png":
        metadata["png_mode"] = resolved["png_mode"]
        if resolved["png_mode"] == "normalized" and np.issubdtype(
            source_dtype, np.integer
        ):
            image /= np.iinfo(source_dtype).max
    elif extension not in {".hdr", ".rgbe", ".pic", ".pfm"}:
        if np.issubdtype(source_dtype, np.integer):
            image /= np.iinfo(source_dtype).max
            metadata["normalized_ldr"] = True

    image = np.clip(image, 0.0, None)
    if resolved["transfer"] == "pq":
        image = pq_to_linear(image)
    elif resolved["transfer"] == "linear_times_100":
        image *= 100.0
    image *= float(resolved["linear_scale"])

    image = remove_specials(image, config.remove_specials_value)
    return image, metadata


def write_image(
    filename: str | Path,
    image: np.ndarray,
    bit_depth: int = 8,
) -> None:
    if bit_depth not in {8, 16}:
        raise ValueError("output.bit_depth must be 8 or 16")

    maximum = 255 if bit_depth == 8 else 65535
    dtype = np.uint8 if bit_depth == 8 else np.uint16
    encoded = np.rint(np.clip(image, 0.0, 1.0) * maximum).astype(dtype)
    if encoded.ndim == 3 and encoded.shape[2] == 3:
        encoded = cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR)

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), encoded):
        raise ValueError(f"Cannot write image: {path}")


def hdrimread(
    filename: str | Path,
    config: ReaderConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    extension = Path(filename).suffix.lower()
    if extension in {".hdr", ".rgbe", ".pic"}:
        return read_rgbe(filename)
    return read_hdr(filename, config)


def hdrimwrite(
    image: np.ndarray,
    filename: str | Path,
    metadata: dict[str, Any] | None = None,
) -> bool:
    extension = Path(filename).suffix.lower()
    if extension in {".hdr", ".rgbe", ".pic"}:
        return write_rgbe(image, filename, metadata)
    if extension == ".pfm":
        return write_pfm(image, filename)
    if extension == ".exr":
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        source = np.asarray(image, dtype=np.float32)
        if pyexr is not None:
            pyexr.write(str(path), source)
            return True
        return bool(cv2.imwrite(str(path), cv2.cvtColor(source, cv2.COLOR_RGB2BGR)))
    bit_depth = int((metadata or {}).get("bit_depth", 8))
    write_image(filename, image, bit_depth)
    return True


def get_raw_saturation_level(image_or_filename: np.ndarray | str | Path) -> int:
    if isinstance(image_or_filename, (str, Path)):
        image = cv2.imread(str(image_or_filename), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read RAW-derived image: {image_or_filename}")
    else:
        image = np.asarray(image_or_filename)
    values = image.reshape(-1)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Saturation estimation requires integer RAW values")
    levels = np.iinfo(values.dtype).max + 1
    histogram = np.bincount(values.astype(np.int64), minlength=levels)
    laplacian = np.convolve(histogram, (-1, 2, -1), mode="same")
    minimum = min(2**12, levels - 1)
    return max(int(np.argmax(laplacian[minimum:]) + minimum), minimum)


def read_raw_info(filename: str | Path) -> dict[str, float | int]:
    path = Path(filename)
    if rawpy is not None and path.suffix.lower() not in {".tif", ".tiff", ".png"}:
        with rawpy.imread(str(path)) as raw:
            metadata = raw.metadata
            sizes = raw.sizes
            return {
                "FNumber": float(metadata.aperture or 1),
                "ISOSpeedRatings": float(metadata.iso_speed or 1),
                "FocalLength": float(metadata.focal_len or 1),
                "ExposureTime": float(metadata.shutter or 1),
                "Width": int(sizes.width),
                "Height": int(sizes.height),
                "NumberOfSamples": 3,
            }
    with Image.open(path) as image:
        exif = image.getexif()
        return {
            "FNumber": float(exif.get(33437, 1) or 1),
            "ISOSpeedRatings": float(exif.get(34855, 1) or 1),
            "FocalLength": float(exif.get(37386, 1) or 1),
            "ExposureTime": float(exif.get(33434, 1) or 1),
            "Width": int(image.width),
            "Height": int(image.height),
            "NumberOfSamples": len(image.getbands()),
        }


def read_raw(
    filename: str | Path,
    saturation_level: int = 2**12 - 1,
) -> tuple[np.ndarray, dict[str, float | int], int]:
    path = Path(filename)
    if rawpy is not None and path.suffix.lower() not in {".tif", ".tiff", ".png"}:
        with rawpy.imread(str(path)) as raw:
            saturation = (
                int(min(raw.camera_white_level_per_channel))
                if saturation_level < 0 and raw.camera_white_level_per_channel
                else int(saturation_level)
            )
            image = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=True,
                output_bps=16,
                user_sat=saturation,
            )
        return image, read_raw_info(path), saturation
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        requirement = " Install rawpy for camera RAW formats." if rawpy is None else ""
        raise ValueError(f"Cannot read RAW image: {path}.{requirement}")
    image = _to_rgb(image)
    saturation = get_raw_saturation_level(image) if saturation_level < 0 else int(saturation_level)
    return image, read_raw_info(path), saturation


def rr_contains(string: str, value: str) -> bool:
    return value in string


read_pfm = read_pfm
write_pfm = write_pfm
read_rgbe = read_rgbe
write_rgbe = write_rgbe
write_rgbe_line = write_rgbe_line
hdrimread = hdrimread
hdrimwrite = hdrimwrite
ldrimread = ldrimread
read_crfs = read_crfs
getRAWSaturationLevel = get_raw_saturation_level
read_raw = read_raw
read_raw_info = read_raw_info
