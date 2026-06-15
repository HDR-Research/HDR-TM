from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .analysis import log_mean
from .colorspace import luminance
from .core import matlab_percentile, remove_specials
from .io import ReaderConfig, hdrimread, ldrimread


@dataclass
class VideoStream:
    type: str
    path: Path
    files: list[Path]
    total_frames: int
    frame_rate: float = 24.0
    frame_counter: int = 0
    stream_open: bool = True
    permission: str = "r"
    capture: cv2.VideoCapture | None = None
    reader_config: ReaderConfig | None = None


class LazyVideoWriter:
    def __init__(
        self,
        filename: str | Path,
        frame_rate: float,
        profile: str = "mp4v",
        quality: int = 95,
    ) -> None:
        self.filename = str(filename)
        self.frame_rate = float(frame_rate)
        self.profile = profile
        self.quality = quality
        self.writer: cv2.VideoWriter | None = None

    def write(self, frame: np.ndarray) -> None:
        source = np.asarray(frame)
        if np.issubdtype(source.dtype, np.floating):
            source = np.rint(np.clip(source, 0, 1) * 255).astype(np.uint8)
        if source.ndim == 2:
            source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        else:
            source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        if self.writer is None:
            code = self.profile[:4].ljust(4, "v")
            self.writer = cv2.VideoWriter(
                self.filename,
                cv2.VideoWriter_fourcc(*code),
                self.frame_rate,
                (source.shape[1], source.shape[0]),
            )
            if not self.writer.isOpened():
                raise ValueError(f"Cannot open video writer: {self.filename}")
        self.writer.write(source)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    release = close


def check_video_resolution(rows: int, columns: int) -> bool:
    resolutions = {
        (144, 176), (128, 160), (240, 320), (240, 352), (288, 352),
        (480, 640), (480, 720), (576, 704), (576, 720), (720, 1280),
        (1080, 1920),
    }
    return (int(rows), int(columns)) in resolutions


def get_open_video_writer(
    filename: str | Path,
    frame_rate: float,
    profile: str = "mp4v",
    quality: int = 95,
) -> LazyVideoWriter:
    return LazyVideoWriter(filename, frame_rate, profile, quality)


def _directory_stream(
    directory: Path,
    choices: tuple[tuple[str, tuple[str, ...]], ...],
    config: ReaderConfig | None = None,
) -> VideoStream:
    for stream_type, extensions in choices:
        files = []
        for extension in extensions:
            files.extend(directory.glob(f"*.{extension}"))
        files = sorted(files)
        if files:
            return VideoStream(stream_type, directory, files, len(files), reader_config=config)
    raise ValueError(f"No supported frames found in {directory}")


def hdrvread(
    filename: str | Path,
    config: ReaderConfig | None = None,
) -> VideoStream:
    path = Path(filename)
    if not path.is_dir():
        raise ValueError("HDR video input must be a directory of HDR frames")
    return _directory_stream(
        path,
        (
            ("TYPE_HDR_RGBE", ("hdr", "rgbe", "pic")),
            ("TYPE_HDR_PFM", ("pfm",)),
            ("TYPE_HDR_EXR", ("exr",)),
            ("TYPE_HDR_JPEG_2000", ("jp2",)),
        ),
        config,
    )


def ldrvread(filename: str | Path) -> VideoStream:
    path = Path(filename)
    if path.is_dir():
        return _directory_stream(
            path,
            (
                ("TYPE_LDR_PNG", ("png",)),
                ("TYPE_LDR_JPEG", ("jpg", "jpeg")),
                ("TYPE_LDR_JPEG_2000", ("jp2",)),
            ),
        )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    return VideoStream(
        "TYPE_LDR_VIDEO",
        path,
        [],
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        float(capture.get(cv2.CAP_PROP_FPS) or 24),
        capture=capture,
    )


def hdrvopen(video: VideoStream) -> VideoStream:
    video.stream_open = True
    return video


def ldrvopen(video: VideoStream) -> VideoStream:
    if video.capture is not None and not video.capture.isOpened():
        video.capture.open(str(video.path))
    video.stream_open = True
    return video


def _frame_index(video: VideoStream, frame_counter: int | None) -> int:
    if video.total_frames < 1:
        raise ValueError("Video has no frames")
    index = video.frame_counter if frame_counter is None else int(frame_counter)
    return int(np.clip(index, 0, video.total_frames - 1))


def hdrv_get_frame(
    video: VideoStream,
    frame_counter: int | None = None,
) -> tuple[np.ndarray, VideoStream]:
    index = _frame_index(video, frame_counter)
    frame, _ = hdrimread(video.files[index], video.reader_config)
    video.frame_counter = (index + 1) % video.total_frames
    return frame, video


def ldrv_get_frame(
    video: VideoStream,
    frame_counter: int | None = None,
) -> tuple[np.ndarray, VideoStream]:
    index = _frame_index(video, frame_counter)
    if video.capture is None:
        frame = ldrimread(video.files[index], double=False)
    else:
        video.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, raw = video.capture.read()
        if not success:
            raise ValueError(f"Cannot read frame {index}")
        frame = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    video.frame_counter = (index + 1) % video.total_frames
    return frame, video


def hdrvclose(video: VideoStream) -> VideoStream:
    video.stream_open = False
    if video.capture is not None:
        video.capture.release()
    return video


def ldrvclose(video: VideoStream) -> VideoStream:
    return hdrvclose(video)


def _analyze(
    video: VideoStream,
    frame_reader: Callable[[VideoStream, int], tuple[np.ndarray, VideoStream]],
    percentile: float,
    crop: tuple[int, int] | None,
    histogram: bool,
    ldr: bool,
) -> tuple[VideoStream, np.ndarray, np.ndarray]:
    percentile = float(np.clip(percentile, 0.500001, 0.99))
    bins = 256 if ldr else 4096
    statistics = np.zeros((video.total_frames, 7))
    histograms = np.zeros((video.total_frames, bins)) if histogram else np.empty((video.total_frames, 0))
    for index in range(video.total_frames):
        frame, video = frame_reader(video, index)
        if crop and crop[0] > 0 and crop[1] > 0:
            frame = frame[crop[0] : -crop[0], crop[1] : -crop[1]]
        frame = remove_specials(frame)
        luma = luminance(frame**2.2 if ldr else frame)
        values = luma[luma >= 0]
        if values.size:
            statistics[index] = (
                np.min(values),
                np.max(values),
                matlab_percentile(values, 1 - percentile),
                matlab_percentile(values, percentile),
                np.mean(values),
                matlab_percentile(values, 0.5),
                log_mean(values),
            )
            if histogram:
                histograms[index], _ = np.histogram(
                    np.log10(np.maximum(values, 1e-6)),
                    bins=bins,
                    range=(-6, 6),
                )
    return hdrvclose(video), statistics, histograms


def hdrv_analysis(
    video: VideoStream,
    percentile: float = 0.99,
    crop_rect: tuple[int, int] | None = None,
    histogram: bool = False,
) -> tuple[VideoStream, np.ndarray, np.ndarray]:
    return _analyze(hdrvopen(video), hdrv_get_frame, percentile, crop_rect, histogram, False)


def ldrv_analysis(
    video: VideoStream,
    percentile: float = 0.99,
    crop_rect: tuple[int, int] | None = None,
    histogram: bool = False,
) -> tuple[VideoStream, np.ndarray, np.ndarray]:
    return _analyze(ldrvopen(video), ldrv_get_frame, percentile, crop_rect, histogram, True)


checkVideoResolution = check_video_resolution
getAnOpenVideoWriter = get_open_video_writer
hdrvopen = hdrvopen
hdrvGetFrame = hdrv_get_frame
hdrvread = hdrvread
hdrvclose = hdrvclose
ldrvopen = ldrvopen
ldrvGetFrame = ldrv_get_frame
ldrvread = ldrvread
ldrvclose = ldrvclose
hdrvAnalysis = hdrv_analysis
ldrvAnalysis = ldrv_analysis
