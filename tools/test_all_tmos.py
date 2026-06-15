from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from hdrtmo.colorspace import convert_rgb2020_to_rgb709, luminance
from hdrtmo.io import ReaderConfig, read_hdr, write_image
from hdrtmo.operators import reinhard_tmo
from hdrtmo.tmo import (
    best_exposure_tmo,
    drago_tmo,
    exponential_tmo,
    ferwerda_tmo,
    gamma_tmo,
    logarithmic_tmo,
    normalize_tmo,
    reinhard_devlin_tmo,
    reinhard_robust_tmo,
    schlick_tmo,
    select_overexposed_tmo,
    tumblin_tmo,
    ward_global_tmo,
)
from hdrtmo.tmo_utils import change_luminance, gamma_drago
from hdrtmo.tools import automatic_exposure
from hdrtmo.tmo_local import (
    ashikhmin_tmo,
    banterle_tmo,
    bruce_expo_blend_tmo,
    chiu_tmo,
    durand_tmo,
    kim_kautz_consistent_tmo,
    krawczyk_tmo,
    kuang_tmo,
    lischinski_tmo,
    mertens_tmo,
    pattanaik_tmo,
    raman_tmo,
    van_hateren_tmo,
    ward_hist_adj_tmo,
    yp_ferwerda_tmo,
    yp_tumblin_tmo,
    yp_ward_global_tmo,
)


@dataclass(frozen=True)
class Operator:
    name: str
    function: Callable[[np.ndarray], np.ndarray | tuple]
    output_encoding: str = "linear"
    range_policy: str = "no final RGB clamp"


@dataclass
class Result:
    image: str
    operator: str
    status: str
    seconds: float
    input_shape: str
    output_shape: str = ""
    output_min: float | None = None
    output_max: float | None = None
    luminance_min: float | None = None
    luminance_max: float | None = None
    finite_fraction: float | None = None
    input_domain: str = "linear Rec.709/sRGB primaries"
    output_encoding: str = ""
    range_policy: str = ""
    output_file: str = ""
    error: str = ""


OPERATORS = (
    Operator("AshikhminTMO", ashikhmin_tmo),
    Operator("BanterleTMO", banterle_tmo),
    Operator("BestExposureTMO", best_exposure_tmo, range_policy="explicit [0,1] RGB clamp"),
    Operator("BruceExpoBlendTMO", bruce_expo_blend_tmo, "gamma encoded", "global min/max normalization"),
    Operator("ChiuTMO", chiu_tmo),
    Operator("DragoTMO", drago_tmo, "linear; use GammaDrago"),
    Operator("DurandTMO", durand_tmo),
    Operator("ExponentialTMO", exponential_tmo),
    Operator("FerwerdaTMO", ferwerda_tmo, range_policy="explicit [0,1] RGB clamp"),
    Operator(
        "GammaTMO",
        lambda image: gamma_tmo(image, fstop=np.log2(automatic_exposure(image)[1])),
        "gamma encoded",
        "automatic exposure then [0,1] clamp; no normalization",
    ),
    Operator("KimKautzConsistentTMO", kim_kautz_consistent_tmo, range_policy="robust luminance normalization"),
    Operator("KrawczykTMO", krawczyk_tmo),
    Operator("KuangTMO", kuang_tmo, range_policy="robust global RGB normalization and clamp"),
    Operator("LischinskiTMO", lischinski_tmo),
    Operator("LogarithmicTMO", logarithmic_tmo, range_policy="maximum luminance normalization; no RGB clamp"),
    Operator("MertensTMO", mertens_tmo, "gamma encoded", "global min/max normalization and clamp"),
    Operator("NormalizeTMO", normalize_tmo, range_policy="robust luminance normalization and RGB clamp"),
    Operator("PattanaikTMO", pattanaik_tmo, "linear; restore source chroma and display with gamma 1.6"),
    Operator("RamanTMO", raman_tmo, "gamma encoded", "explicit [0,1] RGB clamp"),
    Operator(
        "ReinhardDevlinTMO",
        lambda image: reinhard_devlin_tmo(image, intensity=-2.0),
        "linear; display with gamma 1.6",
        "normalized output; intensity=-2 for absolute-luminance HDR preview",
    ),
    Operator("ReinhardRobustTMO", reinhard_robust_tmo),
    Operator("ReinhardTMO", reinhard_tmo),
    Operator("SchlickTMO", schlick_tmo, range_policy="mapped luminance in [0,1]; gamut clip before display"),
    Operator("SelectOverexposedTMO", select_overexposed_tmo, range_policy="selected exposure then [0,1] RGB clamp"),
    Operator("TumblinTMO", tumblin_tmo),
    Operator("VanHaterenTMO", van_hateren_tmo, "perceptual display response; no gamma required"),
    Operator("WardGlobalTMO", ward_global_tmo, range_policy="explicit [0,1] RGB clamp"),
    Operator("WardHistAdjTMO", ward_hist_adj_tmo, range_policy="display luminance normalization; no final RGB clamp"),
    Operator("YPFerwerdaTMO", yp_ferwerda_tmo, range_policy="explicit [0,1] RGB clamp"),
    Operator("YPTumblinTMO", yp_tumblin_tmo),
    Operator("YPWardGlobalTMO", yp_ward_global_tmo, range_policy="explicit [0,1] RGB clamp"),
)


def reader_for(path: Path) -> ReaderConfig:
    transfer = "pq" if path.suffix.lower() == ".png" else "linear_times_100"
    return ReaderConfig(
        preset="custom",
        png_mode="normalized",
        transfer=transfer,
        primaries="rec2020",
    )


def resize_for_test(image: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0 or max(image.shape[:2]) <= max_side:
        return image
    scale = max_side / max(image.shape[:2])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def primary_output(value: np.ndarray | tuple) -> np.ndarray:
    return np.asarray(value[0] if isinstance(value, tuple) else value, dtype=np.float64)


def gamut_compress(image: np.ndarray) -> np.ndarray:
    source = np.maximum(np.asarray(image, dtype=np.float64), 0)
    maximum = np.max(source, axis=2, keepdims=True)
    return source / np.maximum(maximum, 1.0)


def preview(output: np.ndarray, output_encoding: str, source: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
    if output_encoding == "gamma encoded" or output_encoding.startswith("perceptual"):
        return np.clip(clean, 0.0, 1.0)
    if output_encoding == "linear; use GammaDrago":
        return gamma_drago(gamut_compress(clean))
    if output_encoding == "linear; restore source chroma and display with gamma 1.6":
        restored = change_luminance(source, luminance(source), luminance(clean))
        return gamma_tmo(gamut_compress(restored), 1.6)
    if output_encoding == "linear; display with gamma 1.6":
        return gamma_tmo(gamut_compress(clean), 1.6)
    return gamma_tmo(gamut_compress(clean), 2.2)


def run(input_directory: Path, output_directory: Path, max_side: int) -> list[Result]:
    inputs = sorted(
        path for path in input_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".exr", ".hdr", ".png"}
    )
    if not inputs:
        raise ValueError(f"No EXR, HDR, or PNG images found in {input_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    input_metadata: list[dict[str, object]] = []

    for path in inputs:
        image, metadata = read_hdr(path, reader_for(path))
        rec709_image = np.maximum(convert_rgb2020_to_rgb709(image, clip=False), 0.0)
        test_image = resize_for_test(rec709_image, max_side)
        input_metadata.append(
            {
                "file": str(path),
                "original_shape": list(image.shape),
                "test_shape": list(test_image.shape),
                "minimum": float(np.min(image)),
                "maximum": float(np.max(image)),
                "mean": float(np.mean(image)),
                "tmo_primaries": "rec709/sRGB",
                "gamut_conversion": "linear Rec.2020 to linear Rec.709; negative out-of-gamut values clamped to zero",
                **metadata,
            }
        )
        image_output = output_directory / path.stem
        image_output.mkdir(parents=True, exist_ok=True)

        for operator in OPERATORS:
            started = time.perf_counter()
            result = Result(
                image=path.name,
                operator=operator.name,
                status="failed",
                seconds=0.0,
                input_shape="x".join(map(str, test_image.shape)),
                output_encoding=operator.output_encoding,
                range_policy=operator.range_policy,
            )
            try:
                mapped = primary_output(operator.function(test_image.copy()))
                result.seconds = time.perf_counter() - started
                result.output_shape = "x".join(map(str, mapped.shape))
                result.output_min = float(np.nanmin(mapped))
                result.output_max = float(np.nanmax(mapped))
                mapped_luminance = luminance(mapped)
                result.luminance_min = float(np.nanmin(mapped_luminance))
                result.luminance_max = float(np.nanmax(mapped_luminance))
                result.finite_fraction = float(np.mean(np.isfinite(mapped)))
                if mapped.shape != test_image.shape:
                    raise ValueError(f"Unexpected output shape {mapped.shape}")
                if result.finite_fraction < 1.0:
                    raise ValueError(f"Non-finite output fraction: {1 - result.finite_fraction:.6g}")
                destination = image_output / f"{operator.name}.png"
                write_image(destination, preview(mapped, operator.output_encoding, test_image), 8)
                result.output_file = str(destination)
                result.status = "passed"
            except Exception as error:
                result.seconds = time.perf_counter() - started
                result.error = f"{type(error).__name__}: {error}"
                (image_output / f"{operator.name}.traceback.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            results.append(result)
            print(
                f"[{result.status.upper():6}] {path.name:16} "
                f"{operator.name:28} {result.seconds:8.3f}s"
                + (f"  {result.error}" if result.error else "")
            )

    (output_directory / "inputs.json").write_text(
        json.dumps(input_metadata, indent=2), encoding="utf-8"
    )
    with (output_directory / "results.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=Result.__dataclass_fields__.keys())
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    summary = {
        "operators": len(OPERATORS),
        "images": len(inputs),
        "runs": len(results),
        "passed": sum(result.status == "passed" for result in results),
        "failed": sum(result.status == "failed" for result in results),
        "max_side": max_side,
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every single-image TMO on configured HDR inputs")
    parser.add_argument("--input", type=Path, default=Path("hdrimage"))
    parser.add_argument("--output", type=Path, default=Path("tmo_test_results"))
    parser.add_argument(
        "--max-side",
        type=int,
        default=256,
        help="Resize longest side for testing; use 0 for original resolution",
    )
    arguments = parser.parse_args()
    results = run(arguments.input, arguments.output, arguments.max_side)
    failures = [result for result in results if result.status != "passed"]
    print(f"Completed {len(results)} runs: {len(results) - len(failures)} passed, {len(failures)} failed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
