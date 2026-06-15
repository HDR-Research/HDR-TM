from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .colorspace import luminance


@dataclass(frozen=True)
class Distribution1D:
    pdf: np.ndarray
    cdf: np.ndarray
    maximum: float


@dataclass(frozen=True)
class Light:
    color: np.ndarray
    x: float
    y: float
    x_bound: tuple[int, int]
    y_bound: tuple[int, int]


@dataclass(frozen=True)
class Sample:
    direction: np.ndarray
    x: float
    y: float
    color: np.ndarray
    pdf: float


def create_1d_distribution(values: np.ndarray) -> Distribution1D:
    values = np.clip(np.asarray(values, dtype=np.float64).reshape(-1), 0, None)
    total = float(np.sum(values))
    pdf = values / total if total > 0 else np.zeros_like(values)
    raw_cdf = np.cumsum(values)
    maximum = float(raw_cdf[-1]) if raw_cdf.size else 0.0
    cdf = raw_cdf / maximum if maximum > 0 else raw_cdf
    return Distribution1D(pdf, cdf, maximum)


def sample_1d_distribution(distribution: Distribution1D, value: float) -> tuple[int, float]:
    if distribution.cdf.size == 0:
        raise ValueError("Cannot sample an empty distribution")
    index = int(np.argmin(np.abs(distribution.cdf - np.clip(value, 0, 1))))
    return index, float(distribution.pdf[index])


def create_light(
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    luma: np.ndarray,
    image: np.ndarray,
) -> Light | None:
    region_luma = np.asarray(luma)[y_min : y_max + 1, x_min : x_max + 1]
    total_luma = float(np.sum(region_luma))
    if region_luma.size == 0 or total_luma <= 0:
        return None
    region = np.asarray(image)[y_min : y_max + 1, x_min : x_max + 1]
    x, y = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
    return Light(
        np.sum(region, axis=(0, 1)),
        float(np.sum(region_luma * x) / (total_luma * luma.shape[1])),
        float(np.sum(region_luma * y) / (total_luma * luma.shape[0])),
        (x_min, x_max),
        (y_min, y_max),
    )


def fall_off(rows: int, columns: int) -> np.ndarray:
    y = np.arange(1, rows + 1)[:, None]
    return np.repeat(np.cos((0.5 - y / rows) * np.pi), columns, axis=1)


def fall_off_environment_map(image: np.ndarray) -> np.ndarray:
    return np.asarray(image) * fall_off(*image.shape[:2])[..., None]


def generate_light_map(
    lights: list[Light],
    width: int = 512,
    height: int = 256,
) -> np.ndarray:
    if not lights:
        raise ValueError("lights cannot be empty")
    output = np.zeros((height, width, lights[0].color.size))
    for light in lights:
        x = int(np.clip(np.rint(light.x * width), 0, width - 1))
        y = int(np.clip(np.rint(light.y * height), 0, height - 1))
        output[y, x] += light.color
    return output


def polar_vec3(theta: float, phi: float) -> np.ndarray:
    sin_theta = np.sin(theta)
    return np.asarray((np.cos(phi) * sin_theta, np.cos(theta), np.sin(phi) * sin_theta))


def variance_region(
    image: np.ndarray,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> float:
    region = np.asarray(image)[y_min : y_max + 1, x_min : x_max + 1]
    weights = np.sum(region, axis=2) if region.ndim == 3 else region
    total = float(np.sum(weights))
    if total <= 0:
        return 0.0
    x, y = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
    centroid_x = np.sum(weights * x) / total
    centroid_y = np.sum(weights * y) / total
    return float(np.sqrt(np.sum(weights * ((x - centroid_x) ** 2 + (y - centroid_y) ** 2))))


def uniform_sampling(
    image: np.ndarray,
    light_count: int = -1,
    compensate_falloff: bool = False,
) -> tuple[np.ndarray, list[Light]]:
    source = fall_off_environment_map(image) if compensate_falloff else np.asarray(image)
    luma = luminance(source)
    rows, columns = luma.shape
    if light_count < 0:
        light_count = int(2 ** round(np.log2(min(rows, columns)) + 2))
    side = max(int(round(np.sqrt(light_count))), 1)
    cell_width, cell_height = columns // side, rows // side
    if cell_width < 2 or cell_height < 2:
        raise ValueError("Requested too many lights for this map")
    lights = []
    for row in range(side):
        for column in range(side):
            light = create_light(
                column * cell_width,
                min((column + 1) * cell_width, columns) - 1,
                row * cell_height,
                min((row + 1) * cell_height, rows) - 1,
                luma,
                source,
            )
            if light is not None:
                lights.append(light)
    return generate_light_map(lights, columns, rows), lights


def importance_sampling(
    image: np.ndarray,
    compensate_falloff: bool = False,
    sample_count: int = 1024,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[Sample]]:
    source = fall_off_environment_map(image) if compensate_falloff else np.asarray(image)
    luma = luminance(source)
    rows, columns = luma.shape
    column_distributions = [create_1d_distribution(luma[:, column]) for column in range(columns)]
    column_distribution = create_1d_distribution(
        np.asarray([distribution.maximum for distribution in column_distributions])
    )
    rng = np.random.default_rng() if rng is None else rng
    output = np.zeros_like(luma)
    samples = []
    for _ in range(sample_count):
        x, pdf_x = sample_1d_distribution(column_distribution, float(rng.random()))
        y, pdf_y = sample_1d_distribution(column_distributions[x], float(rng.random()))
        phi, theta = 2 * np.pi * x / columns, np.pi * y / rows
        solid_angle = 2 * np.pi**2 * abs(np.sin(theta))
        pdf = pdf_x * pdf_y / solid_angle if solid_angle > 0 else 0.0
        samples.append(Sample(polar_vec3(theta, phi), x / columns, y / rows, source[y, x].copy(), pdf))
        output[y, x] += 1
    return output, samples


def evaluation_sh(
    coefficients: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
) -> np.ndarray:
    sh = np.asarray(coefficients, dtype=np.float64)
    constants = (0.429043, 0.511664, 0.743125, 0.886227, 0.247708)
    output = np.empty(dx.shape + (sh.shape[0],))
    for channel in range(sh.shape[0]):
        value = sh[channel]
        output[..., channel] = (
            constants[0] * value[8] * (dx**2 - dy**2)
            + constants[2] * value[6] * dz**2
            + constants[3] * value[0]
            - constants[4] * value[6]
            + 2 * constants[0] * (value[4] * dx * dy + value[7] * dx * dz + value[5] * dy * dz)
            + 2 * constants[1] * (value[3] * dx + value[1] * dy + value[2] * dz)
        )
    return output


def diffuse_convolution_sh(
    image: np.ndarray,
    compensate_falloff: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    source = fall_off_environment_map(image) if compensate_falloff else np.asarray(image, dtype=np.float64)
    rows, columns, channels = source.shape
    x, y = np.meshgrid(np.arange(1, columns + 1), np.arange(1, rows + 1))
    phi, theta = 2 * np.pi * x / columns, np.pi * y / rows
    sin_theta = np.sin(theta)
    dx, dy, dz = np.cos(phi) * sin_theta, np.cos(theta), np.sin(phi) * sin_theta
    weighted = source * sin_theta[..., None]
    basis = (
        np.full_like(dx, 0.282095),
        dy * 0.488603,
        dz * 0.488603,
        dx * 0.488603,
        dx * dy * 1.092548,
        dy * dz * 1.092548,
        dx * dz * 1.092548,
        (3 * dz**2 - 1) * 0.315392,
        (dx**2 - dy**2) * 0.546274,
    )
    coefficients = np.stack(
        [[np.mean(weighted[..., channel] * term) for term in basis] for channel in range(channels)]
    ) * (2 * np.pi**2)
    return evaluation_sh(coefficients, dx, dy, dz), coefficients


def median_cut_aux(
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    iterations: int,
    luma: np.ndarray,
    image: np.ndarray,
    lights: list[Light] | None = None,
) -> list[Light]:
    lights = [] if lights is None else lights
    width, height = x_max - x_min, y_max - y_min
    if max(width, height) > 1 and iterations > 0:
        region = luma[y_min : y_max + 1, x_min : x_max + 1]
        total = float(np.sum(region))
        if width >= height and width > 0:
            energy = np.cumsum(np.sum(region, axis=0))
            pivot = x_min + min(int(np.searchsorted(energy, total / 2)), width - 1)
            median_cut_aux(x_min, pivot, y_min, y_max, iterations - 1, luma, image, lights)
            median_cut_aux(pivot + 1, x_max, y_min, y_max, iterations - 1, luma, image, lights)
        elif height > 0:
            energy = np.cumsum(np.sum(region, axis=1))
            pivot = y_min + min(int(np.searchsorted(energy, total / 2)), height - 1)
            median_cut_aux(x_min, x_max, y_min, pivot, iterations - 1, luma, image, lights)
            median_cut_aux(x_min, x_max, pivot + 1, y_max, iterations - 1, luma, image, lights)
    else:
        light = create_light(x_min, x_max, y_min, y_max, luma, image)
        if light is not None:
            lights.append(light)
    return lights


def median_cut(
    image: np.ndarray,
    light_count: int = -1,
    compensate_falloff: bool = False,
) -> tuple[np.ndarray, list[Light]]:
    source = fall_off_environment_map(image) if compensate_falloff else np.asarray(image)
    luma = luminance(source)
    if light_count < 0:
        light_count = int(2 ** round(np.log2(min(luma.shape)) + 2))
    levels = max(int(round(np.log2(max(light_count, 1)))), 0)
    lights = median_cut_aux(
        0,
        luma.shape[1] - 1,
        0,
        luma.shape[0] - 1,
        levels,
        luma,
        source,
    )
    return generate_light_map(lights, luma.shape[1], luma.shape[0]), lights


def variance_minimization_sampling_aux(
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    iteration: int,
    levels: int,
    luma: np.ndarray,
    image: np.ndarray,
    lights: list[Light] | None = None,
) -> list[Light]:
    lights = [] if lights is None else lights
    if x_max - x_min > 2 and y_max - y_min > 2 and iteration < levels:
        candidates: list[tuple[float, str, int]] = []
        for pivot in range(x_min, x_max):
            candidates.append(
                (
                    max(
                        variance_region(luma, x_min, pivot, y_min, y_max),
                        variance_region(luma, pivot + 1, x_max, y_min, y_max),
                    ),
                    "x",
                    pivot,
                )
            )
        for pivot in range(y_min, y_max):
            candidates.append(
                (
                    max(
                        variance_region(luma, x_min, x_max, y_min, pivot),
                        variance_region(luma, x_min, x_max, pivot + 1, y_max),
                    ),
                    "y",
                    pivot,
                )
            )
        _, axis, pivot = min(candidates, key=lambda candidate: candidate[0])
        if axis == "x":
            variance_minimization_sampling_aux(
                x_min, pivot, y_min, y_max, iteration + 1, levels, luma, image, lights
            )
            variance_minimization_sampling_aux(
                pivot + 1, x_max, y_min, y_max, iteration + 1, levels, luma, image, lights
            )
        else:
            variance_minimization_sampling_aux(
                x_min, x_max, y_min, pivot, iteration + 1, levels, luma, image, lights
            )
            variance_minimization_sampling_aux(
                x_min, x_max, pivot + 1, y_max, iteration + 1, levels, luma, image, lights
            )
    else:
        light = create_light(x_min, x_max, y_min, y_max, luma, image)
        if light is not None:
            lights.append(light)
    return lights


def variance_minimization_sampling(
    image: np.ndarray,
    light_count: int = -1,
    compensate_falloff: bool = False,
) -> tuple[np.ndarray, list[Light]]:
    source = fall_off_environment_map(image) if compensate_falloff else np.asarray(image)
    luma = luminance(source)
    if light_count < 0:
        light_count = int(2 ** round(np.log2(min(luma.shape)) + 2))
    levels = max(int(round(np.log2(max(light_count, 1)))), 0)
    lights = variance_minimization_sampling_aux(
        0,
        luma.shape[1] - 1,
        0,
        luma.shape[0] - 1,
        0,
        levels,
        luma,
        source,
    )
    return generate_light_map(lights, luma.shape[1], luma.shape[0]), lights


def export_lights(lights: list[Light], filename: str | Path) -> None:
    lines = [f"Num: {len(lights)}", ""]
    for light in lights:
        direction = polar_vec3((0.5 - light.y) * np.pi, light.x * 2 * np.pi)
        lines.append("Dir: " + " ".join(f"{value:g}" for value in direction))
        lines.append("Col: " + " ".join(f"{value:g}" for value in light.color))
        lines.append("")
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


Create1DDistribution = create_1d_distribution
Sampling1DDistribution = sample_1d_distribution
CreateLight = create_light
FallOff = fall_off
FallOffEnvMap = fall_off_environment_map
GenerateLightMap = generate_light_map
PolarVec3 = polar_vec3
VarianceRegion = variance_region
UniformSampling = uniform_sampling
ImportanceSampling = importance_sampling
EvaluationSH = evaluation_sh
DiffuseConvolutionSH = diffuse_convolution_sh
MedianCutAux = median_cut_aux
MedianCut = median_cut
VarianceMinimizationSamplingAux = variance_minimization_sampling_aux
VarianceMinimizationSampling = variance_minimization_sampling
ExportLights = export_lights
