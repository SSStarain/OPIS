"""Shared capability profiles for local and API video generation models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VIDEO_MODELS_PATH = Path(__file__).resolve().parents[2] / "configs" / "video_models.json"


def load_video_models_config() -> dict[str, Any]:
    with VIDEO_MODELS_PATH.open("r", encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict):
        raise ValueError(f"video model config must be a JSON object: {VIDEO_MODELS_PATH}")
    return data


def get_video_model_profile(model: str, section: str) -> dict[str, Any]:
    data = load_video_models_config()
    profiles = data.get(section)
    if not isinstance(profiles, dict) or not isinstance(profiles.get(model), dict):
        raise ValueError(f"video model has no configured {section} profile: {model}")
    return dict(profiles[model])


def get_local_video_model_profile(model: str) -> dict[str, Any]:
    return get_video_model_profile(model, "local_models")


def get_api_video_model_profile(model: str) -> dict[str, Any]:
    return get_video_model_profile(model, "models")


def get_video_observation_overrides(model: str) -> dict[str, Any]:
    data = load_video_models_config()
    overrides = data.get("observation_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("video model observation_overrides must be an object")
    model_overrides = overrides.get(model, {})
    if not isinstance(model_overrides, dict):
        raise ValueError(f"video observation override must be an object: {model}")
    return dict(model_overrides)


def resolve_local_defaults(profile: dict[str, Any]) -> tuple[float, str]:
    """Resolve local defaults from the global benchmark preference order."""
    data = load_video_models_config()
    policy = data.get("policy", {})
    durations = [float(value) for value in profile.get("durations", [])]
    sizes = [str(value) for value in profile.get("resolutions", [])]
    labels = [str(value).lower() for value in profile.get("resolution_labels", [])]
    if not durations or not sizes or len(labels) != len(sizes):
        raise ValueError("local video profile has invalid duration/resolution capabilities")

    duration = next(
        (float(value) for value in policy.get("duration_preference", [])
         if float(value) in durations),
        durations[0],
    )
    preferred_labels = [
        str(value).lower() for value in policy.get("resolution_preference", [])
    ]
    size = next(
        (sizes[labels.index(label)] for label in preferred_labels if label in labels),
        sizes[0],
    )
    return duration, size
