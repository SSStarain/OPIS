"""Shared vision pipeline datatypes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multimem_bench.vision.mask_utils import bbox_center


@dataclass
class SegmentInstance:
    """A segmented object instance before conversion to benchmark schema."""

    segment_id: str
    bbox: tuple[float, float, float, float]
    category: str = "object"
    confidence: float = 1.0
    frame_index: int | None = None
    timestamp: float | None = None
    track_id: str | None = None
    source_prompt: str | None = None
    mask: Any | None = field(default=None, repr=False, compare=False)
    mask_path: str | None = None
    crop_path: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        return bbox_center(self.bbox)

    @property
    def width(self) -> float:
        return max(1e-6, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(1e-6, self.bbox[3] - self.bbox[1])

    def to_reference_object(self, object_id: str) -> dict[str, Any]:
        metadata = dict(self.metadata)
        if self.mask_path:
            metadata["mask_path"] = self.mask_path
        if self.crop_path:
            metadata["crop_path"] = self.crop_path
        if self.source_prompt:
            metadata["source_prompt"] = self.source_prompt
        metadata["segment_id"] = self.segment_id
        return {
            "object_id": object_id,
            "category": self.category,
            "bbox": list(self.bbox),
            "center": list(self.center),
            "attributes": dict(self.attributes),
            "embedding": self.embedding,
            "size_signature": {
                "width": self.width,
                "height": self.height,
                "area": self.width * self.height,
                "aspect_ratio": self.width / max(self.height, 1e-6),
            },
            "metadata": metadata,
        }

    def to_observed_object(self, observed_id: str) -> dict[str, Any]:
        metadata = dict(self.metadata)
        if self.mask_path:
            metadata["mask_path"] = self.mask_path
        if self.crop_path:
            metadata["crop_path"] = self.crop_path
        if self.source_prompt:
            metadata["source_prompt"] = self.source_prompt
        metadata["segment_id"] = self.segment_id
        return {
            "observed_id": observed_id,
            "track_id": self.track_id,
            "category": self.category,
            "bbox": list(self.bbox),
            "confidence": float(self.confidence),
            "attributes": dict(self.attributes),
            "embedding": self.embedding,
            "metadata": metadata,
        }


def load_segment_instances(path: str | Path) -> list[SegmentInstance]:
    """Load precomputed segments from a permissive JSON format.

    Accepted top-level keys are `segments`, `objects`, or `instances`; a raw
    list is accepted as well. Each item may contain a `bbox`, `box`, or `xyxy`.
    """

    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("segments") or data.get("objects") or data.get("instances") or []
    else:
        rows = []

    instances: list[SegmentInstance] = []
    for idx, row in enumerate(rows):
        bbox = row.get("bbox") or row.get("box") or row.get("xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"segment row {idx} missing bbox/box/xyxy")
        segment_id = str(row.get("segment_id") or row.get("id") or f"seg_{idx:04d}")
        frame_index = row.get("frame_index")
        instances.append(
            SegmentInstance(
                segment_id=segment_id,
                bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                category=str(row.get("category") or row.get("label") or "object"),
                confidence=float(row.get("confidence", row.get("score", 1.0))),
                frame_index=int(frame_index) if frame_index is not None else None,
                timestamp=(
                    float(row["timestamp"])
                    if row.get("timestamp") is not None
                    else None
                ),
                track_id=(
                    str(row["track_id"])
                    if row.get("track_id") is not None
                    else None
                ),
                source_prompt=row.get("source_prompt") or row.get("prompt"),
                mask_path=row.get("mask_path"),
                crop_path=row.get("crop_path"),
                attributes=dict(row.get("attributes", {})),
                embedding=row.get("embedding"),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return instances
