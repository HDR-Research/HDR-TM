from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .generation import build_hdr
from .io import ReaderConfig, hdrimread, hdrimwrite, ldrimread, write_image
from .operators import gamma_tmo, reinhard_tmo
from .stacks import create_ldr_stack_from_hdr, read_ldr_stack, read_ldr_stack_info
from .video import VideoStream, get_open_video_writer, hdrv_get_frame, hdrvclose, hdrvopen, ldrv_get_frame, ldrvclose, ldrvread


def _files(directory: str | Path, extension: str) -> list[Path]:
    return sorted(Path(directory).glob(f"*.{extension.lstrip('.')}"))


def convert_hdr_to_hdr(
    input_format: str,
    output_format: str,
    directory: str | Path = ".",
    output_directory: str | Path | None = None,
    reader_config: ReaderConfig | None = None,
) -> list[Path]:
    destination = Path(directory if output_directory is None else output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for path in _files(directory, input_format):
        image, _ = hdrimread(path, reader_config)
        output = destination / f"{path.stem}.{output_format.lstrip('.')}"
        hdrimwrite(image, output)
        outputs.append(output)
    return outputs


def convert_hdr_to_ldr(
    input_format: str,
    output_format: str,
    tone_mapper: Callable[[np.ndarray], np.ndarray | tuple] | None = None,
    gamma: float = 2.2,
    directory: str | Path = ".",
    output_directory: str | Path | None = None,
    reader_config: ReaderConfig | None = None,
    bit_depth: int = 8,
) -> list[Path]:
    destination = Path(directory if output_directory is None else output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    tone_mapper = reinhard_tmo if tone_mapper is None else tone_mapper
    outputs = []
    for path in _files(directory, input_format):
        image, _ = hdrimread(path, reader_config)
        mapped = tone_mapper(image)
        mapped = mapped[0] if isinstance(mapped, tuple) else mapped
        mapped = gamma_tmo(np.clip(mapped, 0, None), gamma)
        output = destination / f"{path.stem}.{output_format.lstrip('.')}"
        write_image(output, mapped, bit_depth)
        outputs.append(output)
    return outputs


def convert_hdr_to_stack(
    input_format: str,
    output_format: str,
    histogram_sampling: bool = False,
    gamma: float = 2.2,
    directory: str | Path = ".",
    output_directory: str | Path | None = None,
    reader_config: ReaderConfig | None = None,
) -> list[Path]:
    destination = Path(directory if output_directory is None else output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    mode = "histogram" if histogram_sampling else "uniform"
    for path in _files(directory, input_format):
        image, _ = hdrimread(path, reader_config)
        stack, exposures = create_ldr_stack_from_hdr(
            image,
            sampling_mode=mode,
            linearization="gamma",
            function=gamma,
        )
        for index in range(stack.shape[3]):
            mean = float(np.mean(np.rint(stack[..., index] * 255) / 255))
            if histogram_sampling or 0.1 < mean < 0.9:
                output = destination / f"{path.stem}_fstop_{index + 1}.{output_format.lstrip('.')}"
                write_image(output, stack[..., index])
                outputs.append(output)
        np.savetxt(destination / f"{path.stem}_exposures.txt", exposures)
    return outputs


def convert_ldr_to_ldr(
    input_format: str,
    output_format: str,
    directory: str | Path = ".",
    output_directory: str | Path | None = None,
    bit_depth: int = 8,
) -> list[Path]:
    destination = Path(directory if output_directory is None else output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for path in _files(directory, input_format):
        output = destination / f"{path.stem}.{output_format.lstrip('.')}"
        write_image(output, ldrimread(path), bit_depth)
        outputs.append(output)
    return outputs


def convert_hdr_video_to_ldr_video(
    video: VideoStream,
    output: str | Path,
    fstops: np.ndarray,
    gamma: float = 2.2,
    quality: int = 95,
    profile: str = "mp4v",
) -> list[Path]:
    fstops = np.asarray(fstops, dtype=float).reshape(-1)
    if fstops.size == 0:
        raise ValueError("fstops cannot be empty")
    output = Path(output)
    image_output = output.suffix.lower() not in {".avi", ".mp4"}
    written = []
    writer = None if image_output else get_open_video_writer(output, video.frame_rate, profile, quality)
    if image_output:
        output.mkdir(parents=True, exist_ok=True)
    video = hdrvopen(video)
    try:
        for index in range(video.total_frames):
            frame, video = hdrv_get_frame(video, index)
            encoded = gamma_tmo(np.clip(frame * 2 ** fstops[index % fstops.size], 0, None), gamma)
            if writer is not None:
                writer.write(encoded)
            else:
                path = output / f"frame_{index:010d}.png"
                write_image(path, encoded)
                written.append(path)
    finally:
        hdrvclose(video)
        if writer is not None:
            writer.close()
    return [output] if writer is not None else written


def convert_ldr_video_to_images(
    filename: str | Path,
    output_name: str | Path,
    image_format: str = "png",
) -> list[Path]:
    video = ldrvread(filename)
    prefix = Path(output_name)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    try:
        for index in range(video.total_frames):
            frame, video = ldrv_get_frame(video, index)
            output = prefix.parent / f"{prefix.name}_{index + 1:07d}.{image_format.lstrip('.')}"
            write_image(output, frame)
            outputs.append(output)
    finally:
        ldrvclose(video)
    return outputs


def build_hdr_from_many_folders(
    stack_root: str | Path,
    output_directory: str | Path,
    extension: str = "png",
    exposures: np.ndarray | None = None,
    linearization: str = "gamma",
    function: np.ndarray | float | None = 2.2,
) -> list[Path]:
    root = Path(stack_root)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir() and path != destination):
        stack, normalization = read_ldr_stack(folder, extension, normalize=True)
        del normalization
        frame_exposures = read_ldr_stack_info(folder, extension) if exposures is None else np.asarray(exposures)
        image, _ = build_hdr(
            stack,
            frame_exposures,
            linearization=linearization,
            function=function,
            weight_type="Deb97",
            merge_type="log",
        )
        output = destination / f"{folder.name}.hdr"
        hdrimwrite(image, output)
        outputs.append(output)
    return outputs


ConvHDRtoHDR = convert_hdr_to_hdr
ConvHDRtoLDR = convert_hdr_to_ldr
ConvHDRtoStack = convert_hdr_to_stack
ConvLDRtoLDR = convert_ldr_to_ldr
ConvHDRvtoLDRv = convert_hdr_video_to_ldr_video
ConvLDRvtoLDRi = convert_ldr_video_to_images
buildHDRFromPathManyFolders = build_hdr_from_many_folders
