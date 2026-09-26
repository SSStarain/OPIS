"""JSON schemas represented as light dataclasses.

This prototype assumes reference-image preprocessing and generated-video
observation extraction have already happened. The evaluator consumes two
artifacts:

- reference_scene.json: objects and absolute geometry.
- video_observation.json: per-frame/window observed instances and tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math


KINEMATIC_CLASSES = frozenset({"rigid", "articulated", "deformable", "unknown"})
MOTION_EXPECTATIONS = frozenset({"world_static", "protocol_frozen", "independent_motion", "unknown"})
EVALUATION_TRACKS = frozenset({"static_geometry", "dynamic_identity", "excluded"})


@dataclass(frozen=True)
class MobilityAnnotation:
    """Versioned object mobility contract used by V3 dispatch.

    ``unknown`` is deliberately conservative: it remains available to Core-2D
    diagnostics but is not eligible for a rigid geometry headline score.
    """

    kinematic_class: str = "unknown"
    motion_expectation: str = "unknown"
    evaluation_track: str = "excluded"
    geometry_eligible: bool = False
    confidence: float | None = None
    evidence_reason: str = ""
    provenance: str = "legacy_default"
    annotator: str | None = None
    annotation_version: str = "mobility_annotation_v1"
    human_review_required: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MobilityAnnotation":
        raw = dict(data or {})
        kinematic = str(raw.get("kinematic_class", "unknown"))
        motion = str(raw.get("motion_expectation", "unknown"))
        track = str(raw.get("evaluation_track", "excluded"))
        if kinematic not in KINEMATIC_CLASSES:
            kinematic = "unknown"
        if motion not in MOTION_EXPECTATIONS:
            motion = "unknown"
        if track not in EVALUATION_TRACKS:
            track = "excluded"
        geometry_eligible = bool(raw.get("geometry_eligible", False))
        static_contract = (
            kinematic == "rigid"
            and motion in {"world_static", "protocol_frozen"}
            and track == "static_geometry"
        )
        if not static_contract:
            geometry_eligible = False
            if kinematic == "unknown":
                track = "excluded"
            elif (kinematic in {"articulated", "deformable"} or motion in {"independent_motion", "unknown"}) and track != "excluded":
                track = "dynamic_identity"
        confidence = raw.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None:
            confidence = min(1.0, max(0.0, confidence)) if math.isfinite(confidence) else None
        return cls(
            kinematic_class=kinematic,
            motion_expectation=motion,
            evaluation_track=track,
            geometry_eligible=geometry_eligible,
            confidence=confidence,
            evidence_reason=str(raw.get("evidence_reason", "")),
            provenance=str(raw.get("provenance", "legacy_default")),
            annotator=(str(raw["annotator"]) if raw.get("annotator") is not None else None),
            annotation_version=str(raw.get("annotation_version", "mobility_annotation_v1")),
            human_review_required=bool(raw.get("human_review_required", True)),
        )

    @classmethod
    def from_object_metadata(cls, metadata: dict[str, Any] | None) -> "MobilityAnnotation":
        raw = dict(metadata or {})
        if isinstance(raw.get("mobility"), dict):
            return cls.from_dict(raw["mobility"])
        legacy = str(raw.get("instance_kind", raw.get("kind", "unknown")))
        if legacy in {"static", "rigid"}:
            return cls(
                kinematic_class="rigid",
                motion_expectation="world_static",
                evaluation_track="static_geometry",
                geometry_eligible=True,
                confidence=0.5,
                evidence_reason="legacy_instance_kind",
                provenance="legacy_metadata",
                human_review_required=True,
            )
        if legacy in {"articulated", "deformable"}:
            return cls(
                kinematic_class=legacy,
                motion_expectation="unknown",
                evaluation_track="dynamic_identity",
                geometry_eligible=False,
                confidence=0.5,
                evidence_reason="legacy_instance_kind",
                provenance="legacy_metadata",
                human_review_required=True,
            )
        return cls()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kinematic_class": self.kinematic_class,
            "motion_expectation": self.motion_expectation,
            "evaluation_track": self.evaluation_track,
            "geometry_eligible": self.geometry_eligible,
            "confidence": self.confidence,
            "evidence_reason": self.evidence_reason,
            "provenance": self.provenance,
            "annotator": self.annotator,
            "annotation_version": self.annotation_version,
            "human_review_required": self.human_review_required,
        }


@dataclass(frozen=True)
class ObjectSignature:
    object_id: str
    category: str
    bbox: tuple[float, float, float, float] | None = None
    center: tuple[float, float] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    size_signature: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    mask: Any | None = field(default=None, repr=False, compare=False)
    mask_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectSignature":
        bbox_raw = data.get("bbox")
        bbox = None
        if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
            bbox = (
                float(bbox_raw[0]),
                float(bbox_raw[1]),
                float(bbox_raw[2]),
                float(bbox_raw[3]),
            )
        center_raw = data.get("center")
        center = None
        if isinstance(center_raw, list) and len(center_raw) == 2:
            center = (float(center_raw[0]), float(center_raw[1]))
        elif bbox is not None:
            center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        metadata = dict(data.get("metadata", {}))
        return cls(
            object_id=str(data["object_id"]),
            category=str(data.get("category", "object")),
            bbox=bbox,
            center=center,
            attributes=dict(data.get("attributes", {})),
            embedding=data.get("embedding"),
            size_signature=dict(data.get("size_signature", {})),
            metadata=metadata,
            mask=data.get("mask"),
            mask_path=data.get("mask_path") or metadata.get("mask_path"),
        )

    @property
    def width(self) -> float | None:
        if self.bbox is None:
            return None
        return max(1e-6, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float | None:
        if self.bbox is None:
            return None
        return max(1e-6, self.bbox[3] - self.bbox[1])

    @property
    def mobility(self) -> MobilityAnnotation:
        return MobilityAnnotation.from_object_metadata(self.metadata)


@dataclass
class ReferenceScene:
    scene_id: str
    objects: dict[str, ObjectSignature]
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_dir: str | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceScene":
        objects = {
            obj.object_id: obj
            for obj in (ObjectSignature.from_dict(raw) for raw in data.get("objects", []))
        }
        return cls(
            scene_id=str(data.get("scene_id", "reference_scene")),
            objects=objects,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class ObservedObject:
    observed_id: str
    bbox: tuple[float, float, float, float]
    category: str = "object"
    track_id: str | None = None
    confidence: float = 1.0
    attributes: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    mask: Any | None = field(default=None, repr=False, compare=False)
    mask_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservedObject":
        bbox = data.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"observed object {data.get('observed_id')} missing bbox")
        metadata = dict(data.get("metadata", {}))
        return cls(
            observed_id=str(data["observed_id"]),
            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            category=str(data.get("category", "object")),
            track_id=data.get("track_id"),
            confidence=float(data.get("confidence", 1.0)),
            attributes=dict(data.get("attributes", {})),
            embedding=data.get("embedding"),
            metadata=metadata,
            mask=data.get("mask"),
            mask_path=data.get("mask_path") or metadata.get("mask_path"),
        )

    @property
    def width(self) -> float:
        return max(1e-6, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(1e-6, self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


@dataclass
class ObservedWindow:
    window_id: str
    frame_index: int | None
    frame_size: tuple[int, int]
    objects: list[ObservedObject]
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservedWindow":
        frame_size = data.get("frame_size", [0, 0])
        return cls(
            window_id=str(data.get("window_id", data.get("frame_index", "window"))),
            frame_index=(
                int(data["frame_index"])
                if data.get("frame_index") is not None
                else None
            ),
            frame_size=(int(frame_size[0]), int(frame_size[1])),
            objects=[ObservedObject.from_dict(raw) for raw in data.get("objects", [])],
            timestamp=(
                float(data["timestamp"])
                if data.get("timestamp") is not None
                else None
            ),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class VideoObservation:
    video_id: str
    windows: list[ObservedWindow]
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_dir: str | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoObservation":
        return cls(
            video_id=str(data.get("video_id", "generated_video")),
            windows=[ObservedWindow.from_dict(raw) for raw in data.get("windows", [])],
            metadata=dict(data.get("metadata", {})),
        )
