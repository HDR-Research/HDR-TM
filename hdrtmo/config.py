from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from .io import ReaderConfig


@dataclass(frozen=True)
class OperatorConfig:
    name: str = "reinhard"
    alpha: float = 0.0
    white_point: float = 0.0
    gamma: float = 2.2
    fstop: float = 0.0


@dataclass(frozen=True)
class ColorConfig:
    enabled: bool = True
    saturation: float = 0.5


@dataclass(frozen=True)
class OutputConfig:
    bit_depth: int = 8


@dataclass(frozen=True)
class AppConfig:
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    operator: OperatorConfig = field(default_factory=OperatorConfig)
    color: ColorConfig = field(default_factory=ColorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section [{name}] must be a table")
    return value


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("rb") as config_file:
        data = tomllib.load(config_file)

    return AppConfig(
        reader=ReaderConfig(**_section(data, "reader")),
        operator=OperatorConfig(**_section(data, "operator")),
        color=ColorConfig(**_section(data, "color")),
        output=OutputConfig(**_section(data, "output")),
    )

