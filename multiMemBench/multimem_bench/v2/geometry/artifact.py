"""Versioned JSON manifest plus compressed NumPy geometry arrays."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from multimem_bench.io import load_json, write_json


_ARRAY_FIELDS = (
    "frame_indices",
    "extrinsics",
    "intrinsics",
    "world_points",
    "point_confidence",
    "depth",
    "depth_confidence",
    "source_sizes",
    "image_transforms",
    "query_points",
    "tracks",
    "track_visibility",
    "track_confidence",
    "camera_confidence",
)


@dataclass
class GeometryBundle:
    status: str
    backend: str
    frame_indices: np.ndarray | None = None
    extrinsics: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    world_points: np.ndarray | None = None
    point_confidence: np.ndarray | None = None
    depth: np.ndarray | None = None
    depth_confidence: np.ndarray | None = None
    source_sizes: np.ndarray | None = None
    model_size: tuple[int, int] | None = None
    image_transforms: np.ndarray | None = None
    query_points: np.ndarray | None = None
    query_reference_ids: tuple[str, ...] = ()
    tracks: np.ndarray | None = None
    track_visibility: np.ndarray | None = None
    track_confidence: np.ndarray | None = None
    camera_confidence: np.ndarray | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def unavailable(
        cls,
        *,
        backend: str,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> "GeometryBundle":
        return cls(
            status="unavailable",
            backend=backend,
            error=error,
            metadata=dict(metadata or {}),
        )

    def validate(self) -> None:
        if self.status not in {"available", "unavailable"}:
            raise ValueError("geometry status must be available or unavailable")
        if self.status == "unavailable":
            return
        missing = [name for name in _ARRAY_FIELDS if getattr(self, name) is None]
        if missing:
            raise ValueError(f"available geometry missing arrays: {', '.join(missing)}")
        assert self.frame_indices is not None
        frame_count = int(self.frame_indices.shape[0])
        for name in (
            "extrinsics",
            "intrinsics",
            "world_points",
            "point_confidence",
            "depth",
            "depth_confidence",
            "source_sizes",
            "image_transforms",
            "tracks",
            "track_visibility",
            "track_confidence",
            "camera_confidence",
        ):
            array = getattr(self, name)
            if array is None or array.shape[0] != frame_count:
                raise ValueError(f"{name} frame axis does not match frame_indices")
        assert self.query_points is not None
        query_count = int(self.query_points.shape[0])
        if len(self.query_reference_ids) != query_count:
            raise ValueError("query_reference_ids does not match query_points")
        assert self.tracks is not None
        if self.tracks.shape[1] != query_count:
            raise ValueError("tracks query axis does not match query_points")
        if self.world_points is not None and self.world_points.shape[-1] != 3:
            raise ValueError("world_points must end with XYZ coordinates")
        if self.model_size is None or min(self.model_size) <= 0:
            raise ValueError("model_size is required for available geometry")

    def save(self, directory: str | Path) -> dict[str, Path]:
        self.validate()
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "geometry_manifest.json"
        manifest: dict[str, Any] = {
            "schema_version": 2,
            "status": self.status,
            "backend": self.backend,
            "error": self.error,
            "model_size": list(self.model_size) if self.model_size else None,
            "query_reference_ids": list(self.query_reference_ids),
            "metadata": self.metadata,
        }
        paths = {"manifest": manifest_path}
        if self.status == "available":
            arrays_path = root / "geometry_arrays.npz"
            arrays = {name: getattr(self, name) for name in _ARRAY_FIELDS}
            np.savez_compressed(arrays_path, **arrays)
            manifest["arrays_file"] = arrays_path.name
            paths["arrays"] = arrays_path
        else:
            stale_arrays = root / "geometry_arrays.npz"
            if stale_arrays.exists():
                stale_arrays.unlink()
        write_json(manifest_path, manifest)
        return paths

    @classmethod
    def load(cls, manifest_path: str | Path) -> "GeometryBundle":
        path = Path(manifest_path)
        data = load_json(path)
        kwargs: dict[str, Any] = {
            "status": str(data.get("status", "unavailable")),
            "backend": str(data.get("backend", "unknown")),
            "error": data.get("error"),
            "model_size": (
                tuple(int(value) for value in data["model_size"])
                if data.get("model_size")
                else None
            ),
            "query_reference_ids": tuple(
                str(value) for value in data.get("query_reference_ids", [])
            ),
            "metadata": dict(data.get("metadata", {})),
        }
        arrays_file = data.get("arrays_file")
        if arrays_file:
            arrays_path = Path(str(arrays_file))
            if not arrays_path.is_absolute():
                arrays_path = path.parent / arrays_path
            with np.load(arrays_path, allow_pickle=False) as arrays:
                for name in _ARRAY_FIELDS:
                    kwargs[name] = np.asarray(arrays[name])
        bundle = cls(**kwargs)
        bundle.validate()
        return bundle
