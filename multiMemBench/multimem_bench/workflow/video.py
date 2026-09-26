"""Generated-video structural validation through ffprobe."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Callable


class VideoValidationError(RuntimeError):
    """A generated file is not a usable encoded video."""


def validate_aspect_ratio(reference_size: tuple[int, int], video_size: tuple[int, int]) -> None:
    """Allow encoder rounding by two pixels, but reject changed image framing."""
    rw, rh = reference_size
    vw, vh = video_size
    if min(rw, rh, vw, vh) <= 0:
        raise VideoValidationError("image dimensions must be positive")
    reference_ratio = rw / rh
    video_ratio = vw / vh
    # Video providers quantize dimensions to codec/model block sizes (for
    # example 1280x720 -> 832x480). Compare ratios, not absolute pixels.
    if abs(video_ratio - reference_ratio) / reference_ratio > 0.05:
        raise VideoValidationError(
            f"aspect_ratio_mismatch: reference={rw}x{rh}, video={vw}x{vh}; "
            "evaluation stopped to avoid comparing cropped or stretched canvases"
        )


def validate_duration(expected: float, actual: float) -> None:
    """Reject local generators that silently cap or change requested duration."""
    tolerance = max(0.5, expected * 0.05)
    if abs(actual - expected) > tolerance:
        raise VideoValidationError(
            f"duration_mismatch: requested={expected:g}s, actual={actual:g}s"
        )


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    byte_size: int
    codec: str
    width: int
    height: int
    duration_seconds: float
    frame_rate: float
    frame_count: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_video(
    path: str | Path,
    *,
    ffprobe: str = "ffprobe",
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> VideoMetadata:
    video = Path(path).expanduser().resolve()
    if not video.is_file() or video.stat().st_size == 0:
        raise VideoValidationError(f"video is missing or empty: {video}")
    completed = runner(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        raise VideoValidationError(
            f"ffprobe failed with exit code {completed.returncode}: {detail}"
        )
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise VideoValidationError("ffprobe returned invalid JSON") from exc
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list):
        raise VideoValidationError("ffprobe result has no streams list")
    stream = next(
        (
            item
            for item in streams
            if isinstance(item, dict) and item.get("codec_type") == "video"
        ),
        None,
    )
    if stream is None:
        raise VideoValidationError("generated file has no video stream")
    width = _positive_int(stream.get("width"), "video width")
    height = _positive_int(stream.get("height"), "video height")
    format_info = payload.get("format")
    if not isinstance(format_info, dict):
        format_info = {}
    duration = _positive_float(
        stream.get("duration", format_info.get("duration")), "video duration"
    )
    frame_rate = _parse_rate(
        stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    )
    frame_count_value = stream.get("nb_frames")
    frame_count: int | None
    try:
        frame_count = int(frame_count_value) if frame_count_value is not None else None
    except (TypeError, ValueError):
        frame_count = None
    if frame_count is None and frame_rate > 0:
        frame_count = max(1, round(duration * frame_rate))
    return VideoMetadata(
        path=str(video),
        byte_size=video.stat().st_size,
        codec=str(stream.get("codec_name") or "unknown"),
        width=width,
        height=height,
        duration_seconds=duration,
        frame_rate=frame_rate,
        frame_count=frame_count,
    )


def _positive_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise VideoValidationError(f"{label} is invalid") from exc
    if parsed <= 0:
        raise VideoValidationError(f"{label} must be positive")
    return parsed


def _positive_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise VideoValidationError(f"{label} is invalid") from exc
    if parsed <= 0:
        raise VideoValidationError(f"{label} must be positive")
    return parsed


def _parse_rate(value: Any) -> float:
    if not isinstance(value, str) or value in {"", "0/0"}:
        raise VideoValidationError("video frame rate is invalid")
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            rate = float(numerator) / float(denominator)
        else:
            rate = float(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise VideoValidationError("video frame rate is invalid") from exc
    if rate <= 0:
        raise VideoValidationError("video frame rate must be positive")
    return rate
