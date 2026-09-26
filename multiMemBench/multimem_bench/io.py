"""I/O helpers for benchmark artifacts and results."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

from PIL import Image

from multimem_bench.schema import ReferenceScene, VideoObservation


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, ensure_ascii=False, indent=2)
    return out


def write_jsonl(path: str | Path, rows: list[Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_to_jsonable(row), ensure_ascii=False) + "\n")
    return out


def _resolve_artifact_path(raw_path: str | Path | None, artifact_path: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    return path if path.is_absolute() else artifact_path.parent / path


def _load_mask(path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    try:
        return Image.open(path).convert("L")
    except (OSError, ValueError):
        return None


def load_reference_scene(path: str | Path) -> ReferenceScene:
    artifact_path = Path(path)
    scene = ReferenceScene.from_dict(load_json(artifact_path))
    scene.artifact_dir = str(artifact_path.parent)
    for obj in scene.objects.values():
        loaded_mask = _load_mask(_resolve_artifact_path(obj.mask_path, artifact_path))
        object.__setattr__(
            obj,
            "mask",
            loaded_mask if loaded_mask is not None else obj.mask,
        )
    return scene


def load_video_observation(path: str | Path) -> VideoObservation:
    artifact_path = Path(path)
    observation = VideoObservation.from_dict(load_json(artifact_path))
    observation.artifact_dir = str(artifact_path.parent)
    for window in observation.windows:
        for obj in window.objects:
            loaded_mask = _load_mask(_resolve_artifact_path(obj.mask_path, artifact_path))
            object.__setattr__(
                obj,
                "mask",
                loaded_mask if loaded_mask is not None else obj.mask,
            )
    return observation
