"""Protocol records shared by V2 association and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from multimem_bench.schema import ReferenceScene


INSTANCE_KINDS = {"static", "rigid", "articulated", "deformable", "unknown"}
EVALUATION_STATES = {
    "scored",
    "not_observable",
    "generation_failure",
    "evaluator_failure",
}


@dataclass(frozen=True)
class ReferenceInstance:
    instance_id: str
    category: str
    kind: str = "unknown"
    ambiguity_group: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    visible_fraction: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    parts: dict[str, Any] = field(default_factory=dict)
    keypoints: dict[str, tuple[float, ...]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in INSTANCE_KINDS:
            raise ValueError(f"unsupported instance kind: {self.kind}")


@dataclass
class ReferenceAnnotation:
    scene_id: str
    instances: dict[str, ReferenceInstance]
    relationships: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_reference_scene(cls, scene: ReferenceScene) -> "ReferenceAnnotation":
        instances: dict[str, ReferenceInstance] = {}
        for object_id, obj in scene.objects.items():
            metadata = dict(obj.metadata)
            kind = str(metadata.get("instance_kind", metadata.get("kind", "unknown")))
            if kind not in INSTANCE_KINDS:
                kind = "unknown"
            raw_keypoints = metadata.get("keypoints", {})
            keypoints = {
                str(name): tuple(float(value) for value in coords)
                for name, coords in raw_keypoints.items()
                if isinstance(coords, (list, tuple)) and len(coords) >= 2
            }
            visible = metadata.get("visible_fraction")
            instances[object_id] = ReferenceInstance(
                instance_id=object_id,
                category=obj.category,
                kind=kind,
                ambiguity_group=_optional_text(metadata.get("ambiguity_group")),
                bbox=obj.bbox,
                visible_fraction=float(visible) if visible is not None else None,
                attributes=dict(obj.attributes),
                embedding=obj.embedding,
                parts=dict(metadata.get("parts", {})),
                keypoints=keypoints,
                metadata=metadata,
            )
        return cls(
            scene_id=scene.scene_id,
            instances=instances,
            relationships=list(scene.metadata.get("relationships", [])),
            metadata=dict(scene.metadata),
        )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
