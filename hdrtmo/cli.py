from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from .config import load_config
from .io import read_hdr
from .runner import (
    TMO_SPECS,
    convert_working_primaries,
    resolve_reader,
    run_tmo,
    write_result,
)


SUPPORTED_INPUTS = {".exr", ".hdr", ".rgbe", ".pic", ".pfm", ".png"}


def parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_parameters(items: list[str]) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Parameter must be key=value: {item}")
        key, value = item.split("=", 1)
        parameters[key.replace("-", "_")] = parse_value(value)
    return parameters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MATLAB-compatible HDR Toolbox tone mappers")
    parser.add_argument("input", nargs="?", type=Path, help="Input image or folder")
    parser.add_argument("output", nargs="?", type=Path, help="Output image or folder")
    parser.add_argument("-a", "--algorithm", action="append", default=[], help="TMO name; repeat or use comma-separated names")
    parser.add_argument("--all", action="store_true", help="Run all 31 single-image TMOs")
    parser.add_argument("--list-algorithms", action="store_true")
    parser.add_argument("--config", type=Path, help="Optional legacy TOML configuration")
    parser.add_argument("--param", action="append", default=[], help="Algorithm parameter as key=value")
    parser.add_argument("--transfer", choices=("auto", "pq", "linear", "linear_times_100"), default="auto")
    parser.add_argument("--input-primaries", choices=("rec709", "rec2020"), default="rec2020")
    parser.add_argument("--working-primaries", choices=("rec709", "rec2020"), default="rec709")
    parser.add_argument("--png-mode", choices=("raw", "normalized"), default="normalized")
    parser.add_argument("--linear-scale", type=float, default=1.0)
    parser.add_argument("--output-mode", choices=("sdr", "raw"), default="sdr")
    parser.add_argument("--gamma", type=float, default=2.2, help="Display gamma for linear TMO outputs")
    parser.add_argument("--bit-depth", type=int, choices=(8, 16), default=8)
    parser.add_argument("--auto-expose-gamma", action="store_true", help="Automatically choose GammaTMO f-stop")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-side", type=int, default=0, help="Resize longest side before TMO; 0 keeps original resolution")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def selected_algorithms(arguments: argparse.Namespace, configured_name: str | None = None) -> list[str]:
    if arguments.all:
        return [spec.name for spec in TMO_SPECS]
    values = [name.strip() for item in arguments.algorithm for name in item.split(",") if name.strip()]
    return values or [configured_name or "ReinhardTMO"]


def input_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(item for item in iterator if item.is_file() and item.suffix.lower() in SUPPORTED_INPUTS)


def output_path(base: Path, source: Path, algorithm: str, multiple: bool, output_mode: str) -> Path:
    if not multiple and base.suffix:
        return base
    extension = ".exr" if output_mode == "raw" else ".png"
    output_dir = base.parent / base.stem if base.suffix else base
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{source.stem}_{algorithm}{extension}"


def resize_image(image, max_side: int):
    if max_side <= 0 or max(image.shape[:2]) <= max_side:
        return image
    scale = max_side / max(image.shape[:2])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.list_algorithms:
        for spec in TMO_SPECS:
            print(f"{spec.name:28} output={spec.output_encoding:10} absolute={str(spec.absolute_luminance):5} parity={spec.parity}")
        return
    if arguments.input is None or arguments.output is None:
        parser.error("input and output are required unless --list-algorithms is used")

    configured = load_config(arguments.config) if arguments.config else None
    algorithms = selected_algorithms(arguments, configured.operator.name if configured else None)
    files = input_files(arguments.input, arguments.recursive)
    if not files:
        raise ValueError(f"No supported input images found: {arguments.input}")
    parameters = parse_parameters(arguments.param)
    if parameters and len(algorithms) != 1:
        parser.error("--param 只能与单个算法一起使用；多个算法请分别运行")
    if configured and not parameters:
        configured_key = configured.operator.name.lower()
        if configured_key in {"reinhard", "reinhardtmo"}:
            parameters = {
                "alpha": configured.operator.alpha,
                "white_point": configured.operator.white_point,
            }
        elif configured_key in {"gamma", "gammatmo"}:
            parameters = {
                "gamma": configured.operator.gamma,
                "fstop": configured.operator.fstop,
            }
    multiple = len(files) > 1 or len(algorithms) > 1 or arguments.input.is_dir()

    for source_path in files:
        reader = configured.reader if configured else resolve_reader(
            source_path, arguments.transfer, arguments.input_primaries, arguments.png_mode, arguments.linear_scale
        )
        input_primaries = str(reader.resolved["primaries"]) if configured else arguments.input_primaries
        image, metadata = read_hdr(source_path, reader)
        working = convert_working_primaries(image, input_primaries, arguments.working_primaries)
        working = resize_image(working, arguments.max_side)
        for algorithm in algorithms:
            mapped, spec = run_tmo(working, algorithm, parameters, arguments.auto_expose_gamma)
            destination = output_path(arguments.output, source_path, spec.name, multiple, arguments.output_mode)
            if destination.exists() and not arguments.overwrite:
                raise FileExistsError(f"Output exists; use --overwrite: {destination}")
            output_gamma = configured.operator.gamma if configured else arguments.gamma
            bit_depth = configured.output.bit_depth if configured else arguments.bit_depth
            write_result(destination, mapped, spec, working, arguments.output_mode, output_gamma, bit_depth)
            print(
                f"{source_path} -> {destination} | {spec.name} | "
                f"transfer={metadata['transfer']} | {input_primaries}->{arguments.working_primaries}"
            )


if __name__ == "__main__":
    main()
