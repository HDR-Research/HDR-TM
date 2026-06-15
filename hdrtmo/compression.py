from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .operators import gamma_tmo, reinhard_tmo


def _write_jp2(path: Path, image: np.ndarray, ratio: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source = image
    if source.ndim == 3:
        source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
    compression = int(np.clip(round(1000 / max(ratio, 1)), 1, 1000))
    if not cv2.imwrite(
        str(path),
        source,
        [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, compression],
    ):
        raise ValueError(f"Cannot write JPEG2000 image: {path}")


def _read_jp2(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read JPEG2000 image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image


def hdr_jpeg2000_encode(
    image: np.ndarray,
    filename: str | Path = "test_hdrjpeg2000.jp2",
    compression_ratio: float = 2,
    bit_depth: int = 16,
) -> None:
    if not 1 <= bit_depth <= 16:
        raise ValueError("bit_depth must be in [1, 16]")
    source = np.clip(np.asarray(image, dtype=np.float64), 0, None)
    working = source[..., None] if source.ndim == 2 else source
    log_image = np.log(working + 1e-6)
    minimum = np.min(log_image, axis=(0, 1))
    maximum = np.max(log_image, axis=(0, 1))
    delta = maximum - minimum
    normalized = np.divide(
        log_image - minimum,
        delta,
        out=np.zeros_like(log_image),
        where=delta > 0,
    )
    quantized = np.rint(normalized * (2**bit_depth - 1)).astype(np.uint16)
    if source.ndim == 2:
        quantized = quantized[..., 0]
    path = Path(filename)
    _write_jp2(path, quantized, compression_ratio)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(
            {
                "codec": "HDRJPEG2000",
                "bit_depth": bit_depth,
                "minimum": minimum.tolist(),
                "maximum": maximum.tolist(),
            }
        ),
        encoding="ascii",
    )


def hdr_jpeg2000_decode(filename: str | Path) -> np.ndarray:
    path = Path(filename)
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not metadata_path.exists():
        raise ValueError(f"Missing HDR JPEG2000 metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    quantized = _read_jp2(path).astype(np.float64)
    source = quantized[..., None] if quantized.ndim == 2 else quantized
    normalized = source / (2 ** int(metadata["bit_depth"]) - 1)
    minimum = np.asarray(metadata["minimum"])
    maximum = np.asarray(metadata["maximum"])
    output = np.exp(normalized * (maximum - minimum) + minimum) - 1e-6
    output = np.clip(output, 0, None)
    return output[..., 0] if quantized.ndim == 2 else output


def boschetti_encode(
    image: np.ndarray,
    name: str | Path = "bosc_enc",
    rate_e: float = 15,
    rate_rgb: float = 15,
    bit_depth: int = 16,
    tone_mapper: Callable[[np.ndarray], np.ndarray | tuple] | None = None,
) -> None:
    if not 1 <= bit_depth <= 16:
        raise ValueError("bit_depth must be in [1, 16]")
    source = np.clip(np.asarray(image, dtype=np.float64), 0, None)
    tone_mapper = reinhard_tmo if tone_mapper is None else tone_mapper
    mapped = tone_mapper(source)
    mapped = mapped[0] if isinstance(mapped, tuple) else mapped
    mapped = gamma_tmo(np.clip(mapped, 0, None), 2.2)
    maximum_value = 2**bit_depth - 1
    mapped = np.rint(mapped * maximum_value) / maximum_value
    exponent = np.mean(np.log2(source / (mapped + 1 / maximum_value) + 1e-4), axis=2)
    minimum, maximum = float(np.min(exponent)), float(np.max(exponent))
    normalized_exponent = (
        (exponent - minimum) / (maximum - minimum)
        if maximum > minimum
        else np.zeros_like(exponent)
    )
    prefix = Path(name)
    exponent_path = prefix.parent / f"{prefix.name}_bos_E.jp2"
    rgb_path = prefix.parent / f"{prefix.name}_bos_RGB.jp2"
    _write_jp2(exponent_path, np.rint(normalized_exponent * maximum_value).astype(np.uint16), rate_e)
    decoded_exponent = _read_jp2(exponent_path).astype(np.float64) / maximum_value
    decoded_exponent = decoded_exponent * (maximum - minimum) + minimum
    rgb = np.divide(source, np.exp2(decoded_exponent)[..., None])
    _write_jp2(rgb_path, np.rint(np.clip(rgb, 0, 1) * maximum_value).astype(np.uint16), rate_rgb)
    rgb_path.with_suffix(rgb_path.suffix + ".json").write_text(
        json.dumps(
            {
                "codec": "Boschetti",
                "bit_depth": bit_depth,
                "minimum_exponent": minimum,
                "maximum_exponent": maximum,
            }
        ),
        encoding="ascii",
    )


def boschetti_decode(name: str | Path) -> np.ndarray:
    prefix = Path(name)
    exponent_path = prefix.parent / f"{prefix.name}_bos_E.jp2"
    rgb_path = prefix.parent / f"{prefix.name}_bos_RGB.jp2"
    metadata_path = rgb_path.with_suffix(rgb_path.suffix + ".json")
    if not metadata_path.exists():
        raise ValueError(f"Missing Boschetti metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    maximum_value = 2 ** int(metadata["bit_depth"]) - 1
    exponent = _read_jp2(exponent_path).astype(np.float64) / maximum_value
    exponent = (
        exponent
        * (metadata["maximum_exponent"] - metadata["minimum_exponent"])
        + metadata["minimum_exponent"]
    )
    rgb = _read_jp2(rgb_path).astype(np.float64) / maximum_value
    return rgb * np.exp2(exponent)[..., None]


BoschettiEnc = boschetti_encode
BoschettiDec = boschetti_decode
HDRJPEG2000Enc = hdr_jpeg2000_encode
HDRJPEG2000Dec = hdr_jpeg2000_decode
