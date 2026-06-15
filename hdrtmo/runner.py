from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .colorspace import convert_rgb2020_to_rgb709, convert_rgb709_to_rgb2020, luminance
from .io import ReaderConfig, hdrimwrite, read_hdr, write_image
from .operators import reinhard_tmo
from .tmo import (
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
from .tmo_local import (
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
from .tmo_utils import change_luminance, gamma_drago
from .tools import automatic_exposure


@dataclass(frozen=True)
class TMOSpec:
    name: str
    function: Callable[..., np.ndarray | tuple]
    output_encoding: str = "linear"
    absolute_luminance: bool = False
    parity: str = "formula"
    notes: str = ""


TMO_SPECS = (
    TMOSpec("AshikhminTMO", ashikhmin_tmo),
    TMOSpec("BanterleTMO", banterle_tmo, absolute_luminance=True),
    TMOSpec("BestExposureTMO", best_exposure_tmo),
    TMOSpec("BruceExpoBlendTMO", bruce_expo_blend_tmo, "gamma", parity="approximate", notes="Local entropy uses a numerical approximation."),
    TMOSpec("ChiuTMO", chiu_tmo),
    TMOSpec("DragoTMO", drago_tmo, "drago"),
    TMOSpec("DurandTMO", durand_tmo),
    TMOSpec("ExponentialTMO", exponential_tmo),
    TMOSpec("FerwerdaTMO", ferwerda_tmo, absolute_luminance=True),
    TMOSpec("GammaTMO", gamma_tmo, "gamma"),
    TMOSpec("KimKautzConsistentTMO", kim_kautz_consistent_tmo),
    TMOSpec("KrawczykTMO", krawczyk_tmo, parity="approximate", notes="Bilateral filtering uses the Python filter backend."),
    TMOSpec("KuangTMO", kuang_tmo, absolute_luminance=True, parity="approximate", notes="CIECAM local white estimation uses SciPy filtering."),
    TMOSpec("LischinskiTMO", lischinski_tmo),
    TMOSpec("LogarithmicTMO", logarithmic_tmo),
    TMOSpec("MertensTMO", mertens_tmo, "gamma"),
    TMOSpec("NormalizeTMO", normalize_tmo),
    TMOSpec("PattanaikTMO", pattanaik_tmo, absolute_luminance=True),
    TMOSpec("RamanTMO", raman_tmo, "gamma", parity="approximate", notes="Bilateral filtering uses the Python filter backend."),
    TMOSpec("ReinhardDevlinTMO", reinhard_devlin_tmo, "gamma16"),
    TMOSpec("ReinhardRobustTMO", reinhard_robust_tmo),
    TMOSpec("ReinhardTMO", reinhard_tmo),
    TMOSpec("SchlickTMO", schlick_tmo),
    TMOSpec("SelectOverexposedTMO", select_overexposed_tmo),
    TMOSpec("TumblinTMO", tumblin_tmo, absolute_luminance=True),
    TMOSpec("VanHaterenTMO", van_hateren_tmo, "perceptual", absolute_luminance=True),
    TMOSpec("WardGlobalTMO", ward_global_tmo, absolute_luminance=True),
    TMOSpec("WardHistAdjTMO", ward_hist_adj_tmo, absolute_luminance=True),
    TMOSpec("YPFerwerdaTMO", yp_ferwerda_tmo, absolute_luminance=True),
    TMOSpec("YPTumblinTMO", yp_tumblin_tmo, absolute_luminance=True),
    TMOSpec("YPWardGlobalTMO", yp_ward_global_tmo, absolute_luminance=True),
)


def _key(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum()).removesuffix("tmo")


TMO_REGISTRY = {_key(spec.name): spec for spec in TMO_SPECS}


def get_tmo(name: str) -> TMOSpec:
    try:
        return TMO_REGISTRY[_key(name)]
    except KeyError as error:
        raise ValueError(f"Unknown TMO '{name}'. Use --list-algorithms.") from error


def primary_output(value: np.ndarray | tuple) -> np.ndarray:
    return np.asarray(value[0] if isinstance(value, tuple) else value, dtype=np.float64)


def resolve_reader(
    path: str | Path,
    transfer: str,
    input_primaries: str,
    png_mode: str = "normalized",
    linear_scale: float = 1.0,
) -> ReaderConfig:
    extension = Path(path).suffix.lower()
    resolved_transfer = ("pq" if extension == ".png" else "linear_times_100") if transfer == "auto" else transfer
    return ReaderConfig(
        preset="custom",
        png_mode=png_mode,
        transfer=resolved_transfer,
        primaries=input_primaries,
        linear_scale=linear_scale,
    )


def convert_working_primaries(image: np.ndarray, input_primaries: str, working_primaries: str) -> np.ndarray:
    if input_primaries == working_primaries:
        return np.asarray(image, dtype=np.float64)
    if input_primaries == "rec2020" and working_primaries == "rec709":
        return np.maximum(convert_rgb2020_to_rgb709(image, clip=False), 0)
    if input_primaries == "rec709" and working_primaries == "rec2020":
        return np.maximum(convert_rgb709_to_rgb2020(image, clip=False), 0)
    raise ValueError(f"Unsupported primaries conversion: {input_primaries} -> {working_primaries}")


def gamut_compress(image: np.ndarray) -> np.ndarray:
    source = np.maximum(np.asarray(image, dtype=np.float64), 0)
    if source.ndim != 3:
        return np.clip(source, 0, 1)
    maximum = np.max(source, axis=2, keepdims=True)
    return source / np.maximum(maximum, 1.0)


def encode_sdr(
    mapped: np.ndarray,
    spec: TMOSpec,
    source: np.ndarray,
    gamma: float = 2.2,
    restore_pattanaik_chroma: bool = True,
) -> np.ndarray:
    clean = np.nan_to_num(mapped, nan=0.0, posinf=1.0, neginf=0.0)
    if spec.name == "PattanaikTMO" and restore_pattanaik_chroma:
        clean = change_luminance(source, luminance(source), luminance(clean))
    clean = gamut_compress(clean)
    if spec.output_encoding in {"gamma", "perceptual"}:
        return np.clip(clean, 0, 1)
    if spec.output_encoding == "drago":
        return gamma_drago(clean)
    if spec.output_encoding == "gamma16":
        return gamma_tmo(clean, 1.6)
    return gamma_tmo(clean, gamma)


def run_tmo(
    image: np.ndarray,
    algorithm: str,
    parameters: dict[str, Any] | None = None,
    auto_expose_gamma: bool = False,
) -> tuple[np.ndarray, TMOSpec]:
    spec = get_tmo(algorithm)
    parameters = dict(parameters or {})
    if spec.name == "GammaTMO" and auto_expose_gamma and "fstop" not in parameters:
        _, exposure = automatic_exposure(image)
        parameters["fstop"] = float(np.log2(exposure))
    return primary_output(spec.function(np.asarray(image).copy(), **parameters)), spec


def write_result(
    path: str | Path,
    mapped: np.ndarray,
    spec: TMOSpec,
    source: np.ndarray,
    output_mode: str = "sdr",
    gamma: float = 2.2,
    bit_depth: int = 8,
) -> None:
    if output_mode == "raw":
        if Path(path).suffix.lower() not in {".exr", ".hdr", ".pfm"}:
            raise ValueError("Raw linear output requires .exr, .hdr, or .pfm")
        hdrimwrite(mapped, path)
        return
    write_image(path, encode_sdr(mapped, spec, source, gamma), bit_depth)
