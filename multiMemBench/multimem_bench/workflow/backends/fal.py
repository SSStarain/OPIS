"""fal.ai image-to-video backend."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable
import urllib.request
from .base import SubmissionUnknownError
from ..schema import GenerationRequest, GenerationResult
from ..video_models import get_api_video_model_profile, load_video_models_config

class FalBackend:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def generate(self, request: GenerationRequest, *, resume_state: dict[str, Any] | None = None,
                 on_state: Callable[[dict[str, Any]], None] | None = None) -> GenerationResult:
        if not request.prompt or not request.image_url:
            raise ValueError("fal generation requires prompt and public image URL")
        try:
            import fal_client
        except ImportError as exc:
            raise ValueError("fal-client is required: pip install fal-client") from exc
        profile = _profile(request.model)
        policy = load_video_models_config().get("policy", {})
        duration = _select_duration(
            round(request.duration),
            profile["durations"],
            policy.get("duration_preference"),
        )
        resolution = _select_resolution(
            request.options.get("resolution") or request.size,
            profile["resolutions"],
            policy.get("preferred_resolution", 480),
            policy.get("resolution_preference"),
        )
        image_field = profile.get("image_field", "image_url")
        arguments = {"prompt": request.prompt, image_field: request.image_url,
                     "duration": _duration_value(request.model, duration),
                     "resolution": resolution}
        aspect_ratio = profile.get("aspect_ratio")
        if isinstance(aspect_ratio, list) and aspect_ratio:
            arguments["aspect_ratio"] = _select_aspect_ratio(
                request.image_path, [str(value) for value in aspect_ratio]
            )
        elif aspect_ratio and policy.get("aspect_ratio", "auto_if_supported") == "auto_if_supported":
            arguments["aspect_ratio"] = "auto"
        if "generate_audio" in profile or "generate_audio" in policy:
            arguments["generate_audio"] = profile.get(
                "generate_audio", policy.get("generate_audio", False)
            )
        if "prompt_expansion_mode" in profile: arguments["prompt_expansion_mode"] = "balanced"
        if request.seed is not None: arguments["seed"] = request.seed
        client = fal_client.SyncClient(key=self.api_key)
        result = client.subscribe(request.model, arguments=arguments, with_logs=False)
        video = result.get("video") if isinstance(result, dict) else None
        url = video.get("url") if isinstance(video, dict) else None
        if not url: raise SubmissionUnknownError("fal returned no video URL")
        output = Path(request.output_path); output.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, output)
        return GenerationResult(video_path=str(output.resolve()), model_id=request.model,
                                effective_parameters={k: v for k, v in arguments.items()
                                                      if k not in {image_field, "prompt"}})

def _profile(model: str) -> dict[str, Any]:
    try:
        return get_api_video_model_profile(model)
    except ValueError as exc:
        raise ValueError(f"fal {exc}") from exc

def resolve_preferences(model: str, duration: float | None, size: str | None) -> tuple[int, str]:
    profile = _profile(model)
    policy = load_video_models_config().get("policy", {})
    if duration is None:
        duration = next((d for d in policy["duration_preference"] if d in profile["durations"]), None)
    if duration is None or duration not in profile["durations"]:
        raise ValueError(f"unsupported fal duration for {model}: {duration}")
    return int(duration), _select_resolution(
        size,
        profile["resolutions"],
        policy.get("preferred_resolution", 480),
        policy.get("resolution_preference"),
    )

def _select_duration(
    requested: int,
    supported: list[int],
    preference: list[int] | None = None,
) -> int:
    if requested in supported: return requested
    for value in preference or (10, 12):
        if value in supported: return value
    return min(supported, key=lambda x: abs(x-requested))

def _select_resolution(
    requested: str | None,
    supported: list[str],
    preferred: int = 480,
    preference: list[str] | None = None,
) -> str:
    if requested:
        match = next((value for value in supported if value.lower() == requested.lower()), None)
        if match is None:
            raise ValueError(f"unsupported fal resolution {requested!r}; choose from {supported}")
        return match
    if preference:
        supported_by_name = {value.lower(): value for value in supported}
        for value in preference:
            match = supported_by_name.get(str(value).lower())
            if match is not None:
                return match
    def pixels(value: str) -> int:
        normalized = value.lower()
        if normalized.endswith("k"):
            return int(float(normalized.removesuffix("k")) * 1000)
        return int(normalized.removesuffix("p"))
    exact = [value for value in supported if pixels(value) == preferred]
    lower = [value for value in supported if pixels(value) < preferred]
    higher = [value for value in supported if pixels(value) > preferred]
    # Prefer the requested tier, then the closest lower tier (480p -> 360p),
    # then the closest higher tier (480p -> 720p).
    return (exact or ([max(lower, key=pixels)] if lower else []) or
            ([min(higher, key=pixels)] if higher else []) or
            [min(supported, key=pixels)])[0]


def _select_aspect_ratio(image_path: str | None, supported: list[str]) -> str:
    """Choose the supported orientation closest to the source image."""
    if not supported:
        raise ValueError("fal aspect ratio profile has no supported values")
    if not image_path:
        return supported[0]
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
    except (OSError, ValueError):
        return supported[0]
    source_ratio = width / max(height, 1)
    parsed: list[tuple[str, float]] = []
    for value in supported:
        try:
            numerator, denominator = value.split(":", 1)
            parsed.append((value, float(numerator) / float(denominator)))
        except (ValueError, ZeroDivisionError):
            continue
    if not parsed:
        raise ValueError(f"fal aspect ratio profile is invalid: {supported}")
    return min(parsed, key=lambda item: abs(item[1] - source_ratio))[0]

def _duration_value(model: str, value: int) -> str | int:
    return str(value) if "seedance" in model or "kling" in model else value
