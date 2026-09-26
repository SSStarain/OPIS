"""Frame loading and sampling utilities."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image, ImageSequence

from multimem_bench.vision.config import VisionConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


class FrameSourceError(RuntimeError):
    pass


def _media_binary(name: str) -> str:
    """Resolve ffmpeg tools from PATH or beside the active Python binary."""
    resolved = shutil.which(name)
    if resolved:
        return resolved
    bundled = Path(sys.executable).resolve().with_name(name)
    if bundled.is_file():
        return str(bundled)
    raise FrameSourceError(f"{name} is not available on PATH or beside {sys.executable}")


@dataclass(frozen=True)
class VideoTimeline:
    frame_timestamps: tuple[float, ...]
    average_fps: float | None
    duration: float | None


def list_frame_paths(path: str | Path) -> list[Path]:
    src = Path(path)
    if src.is_dir():
        return sorted(
            p for p in src.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    if src.is_file() and src.suffix.lower() in IMAGE_EXTENSIONS:
        return [src]
    return []


def load_frame_map(
    path: str | Path,
    *,
    frame_indices: Iterable[int] | None = None,
) -> dict[int, Image.Image]:
    """Load image frames from a directory, still image, or animated image."""

    src = Path(path)
    frame_paths = list_frame_paths(src)
    if frame_paths:
        indices = (
            [int(index) for index in frame_indices]
            if frame_indices is not None
            else list(range(len(frame_paths)))
        )
        if len(indices) != len(frame_paths):
            raise FrameSourceError(
                f"extracted frame count mismatch: expected {len(indices)}, "
                f"found {len(frame_paths)} in {src}"
            )
        return {
            frame_index: Image.open(frame_path).convert("RGB")
            for frame_index, frame_path in zip(indices, frame_paths)
        }
    if src.is_file() and src.suffix.lower() == ".gif":
        image = Image.open(src)
        return {
            idx: frame.convert("RGB").copy()
            for idx, frame in enumerate(ImageSequence.Iterator(image))
        }
    raise FrameSourceError(
        f"cannot load frames from {src}. Provide a frame directory or install a "
        "decoder and extract frames before running observation building."
    )


def extract_video_frames_with_ffmpeg(
    video_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    frame_indices: Iterable[int] | None = None,
) -> Path:
    ffmpeg = _media_binary("ffmpeg")
    out = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="multimem_frames_"))
    out.mkdir(parents=True, exist_ok=True)
    pattern = out / "%06d.png"
    command = [ffmpeg, "-y", "-i", str(video_path)]
    indices = None if frame_indices is None else [int(index) for index in frame_indices]
    if indices is not None:
        if not indices:
            return out
        expression = "+".join(f"eq(n\\,{index})" for index in indices)
        command.extend(["-vf", f"select={expression}", "-fps_mode", "vfr"])
    command.append(str(pattern))
    subprocess.run(command, check=True, capture_output=True, text=True)
    return out


def probe_video_timeline(video_path: str | Path) -> VideoTimeline:
    ffprobe = _media_binary("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,duration,nb_frames:frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        data = json.loads(completed.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise FrameSourceError(f"failed to read video timeline from {video_path}: {exc}") from exc

    streams = data.get("streams") or []
    if not streams:
        raise FrameSourceError(f"no video stream found in {video_path}")
    stream = streams[0]
    average_fps = _parse_frame_rate(stream.get("avg_frame_rate"))
    if average_fps is None:
        average_fps = _parse_frame_rate(stream.get("r_frame_rate"))
    timestamps = [
        float(frame["best_effort_timestamp_time"])
        for frame in data.get("frames", [])
        if frame.get("best_effort_timestamp_time") is not None
    ]
    if not timestamps:
        num_frames = _optional_int(stream.get("nb_frames"))
        if num_frames and average_fps:
            timestamps = [index / average_fps for index in range(num_frames)]
    if not timestamps:
        raise FrameSourceError(f"no frame timestamps found in {video_path}")
    start = timestamps[0]
    relative_timestamps = tuple(max(0.0, timestamp - start) for timestamp in timestamps)
    duration = _optional_float(stream.get("duration"))
    if duration is None and relative_timestamps:
        frame_duration = 1.0 / average_fps if average_fps else 0.0
        duration = relative_timestamps[-1] + frame_duration
    return VideoTimeline(
        frame_timestamps=relative_timestamps,
        average_fps=average_fps,
        duration=duration,
    )


def timestamp_sample_indices(
    timeline: VideoTimeline,
    target_fps: float,
    *,
    max_frames: int | None = None,
    start_time: float = 0.0,
) -> list[int]:
    if target_fps <= 0:
        raise ValueError("video_sample_fps must be greater than 0")
    if start_time < 0:
        raise ValueError("start_time must be non-negative")
    if max_frames is not None and int(max_frames) <= 0:
        return []
    timestamps = timeline.frame_timestamps
    if not timestamps:
        return []
    interval = 1.0 / float(target_fps)
    last_timestamp = timestamps[-1]
    if start_time > last_timestamp + 1e-9:
        return []
    target_count = int(math.floor((last_timestamp - start_time) / interval + 1e-9)) + 1
    selected: list[int] = []
    first_valid_index = bisect_left(timestamps, start_time)
    for target_index in range(target_count):
        target = start_time + target_index * interval
        right = bisect_left(timestamps, target)
        candidates = [
            index
            for index in (right - 1, right)
            if first_valid_index <= index < len(timestamps)
        ]
        if not candidates:
            break
        # Keep the original nearest-frame behavior, but never allow a frame
        # before the configured evaluation interval.
        frame_index = min(
            candidates,
            key=lambda index: (abs(timestamps[index] - target), index),
        )
        if not selected or selected[-1] != frame_index:
            selected.append(frame_index)
        if max_frames is not None and len(selected) >= max(0, int(max_frames)):
            break
    return selected


def ensure_frame_map(
    video_or_frames: str | Path,
    config: VisionConfig,
    *,
    extracted_frames_dir: str | Path | None = None,
    timeline: VideoTimeline | None = None,
) -> dict[int, Image.Image]:
    src = Path(video_or_frames)
    try:
        frame_map = load_frame_map(src)
        if config.video_sample_fps is None:
            return frame_map
        if timeline is None:
            raise FrameSourceError(
                "video_sample_fps requires an encoded source video timeline; "
                "use video_sample_stride for a standalone frame directory"
            )
        selected = timestamp_sample_indices(
            timeline,
            config.video_sample_fps,
            max_frames=config.video_max_frames,
            start_time=config.video_skip_initial_seconds,
        )
        missing = [index for index in selected if index not in frame_map]
        if missing:
            raise FrameSourceError(
                f"frame directory is missing sampled source indices: {missing[:10]}"
            )
        return {index: frame_map[index] for index in selected}
    except FrameSourceError:
        if src.suffix.lower() not in VIDEO_EXTENSIONS:
            raise
        if timeline is None and config.video_sample_fps is not None:
            timeline = probe_video_timeline(src)
        selected = None
        if config.video_sample_fps is not None:
            selected = timestamp_sample_indices(
                timeline,
                config.video_sample_fps,
                max_frames=config.video_max_frames,
                start_time=config.video_skip_initial_seconds,
            )
        frames_dir = extract_video_frames_with_ffmpeg(
            src,
            extracted_frames_dir,
            frame_indices=selected,
        )
        return load_frame_map(frames_dir, frame_indices=selected)


def sample_indices(
    available_frames: int | Iterable[int],
    config: VisionConfig,
    *,
    timeline: VideoTimeline | None = None,
) -> list[int]:
    if isinstance(available_frames, int):
        if available_frames <= 0:
            return []
        available = list(range(available_frames))
    else:
        available = sorted({int(index) for index in available_frames})
    if not available:
        return []
    skip_seconds = float(config.video_skip_initial_seconds)
    if skip_seconds < 0:
        raise ValueError("video_skip_initial_seconds must be non-negative")
    if skip_seconds > 0:
        if timeline is None:
            raise FrameSourceError(
                "video_skip_initial_seconds requires an encoded video timeline "
                "or an explicit value of 0 for a frame directory/precomputed artifact"
            )
        available = [
            index
            for index in available
            if 0 <= index < len(timeline.frame_timestamps)
            and timeline.frame_timestamps[index] >= skip_seconds - 1e-9
        ]
        if not available:
            return []
    if config.video_sample_fps is not None:
        indices = available
    else:
        stride = max(1, int(config.video_sample_stride))
        indices = available[::stride]
    if config.video_max_frames is not None:
        indices = indices[: max(0, int(config.video_max_frames))]
    return indices


def selected_frame_map(
    frame_map: dict[int, Image.Image],
    indices: list[int],
) -> dict[int, Image.Image]:
    return {idx: frame_map[idx] for idx in indices if idx in frame_map}


def iter_frame_items(frame_map: dict[int, Image.Image]) -> Iterator[tuple[int, Image.Image]]:
    for idx in sorted(frame_map):
        yield idx, frame_map[idx]


def _parse_frame_rate(value: object) -> float | None:
    if value in (None, "", "0/0"):
        return None
    try:
        rate = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
