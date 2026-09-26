"""Backend contract for feed-forward geometry models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np
from PIL import Image


class GeometryBackendUnavailable(RuntimeError):
    """Raised when an optional geometry backend cannot be loaded."""


@dataclass
class GeometryPrediction:
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    world_points: np.ndarray
    point_confidence: np.ndarray
    depth: np.ndarray
    depth_confidence: np.ndarray
    tracks: np.ndarray
    track_visibility: np.ndarray
    track_confidence: np.ndarray
    camera_confidence: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, frame_count: int, query_count: int) -> None:
        frame_arrays = {
            "extrinsics": self.extrinsics,
            "intrinsics": self.intrinsics,
            "world_points": self.world_points,
            "point_confidence": self.point_confidence,
            "depth": self.depth,
            "depth_confidence": self.depth_confidence,
            "tracks": self.tracks,
            "track_visibility": self.track_visibility,
            "track_confidence": self.track_confidence,
            "camera_confidence": self.camera_confidence,
        }
        for name, array in frame_arrays.items():
            if np.asarray(array).shape[0] != frame_count:
                raise ValueError(f"geometry prediction {name} has the wrong frame axis")
        if self.world_points.shape[-1] != 3:
            raise ValueError("geometry prediction world_points must end in XYZ")
        if self.tracks.shape[1:] != (query_count, 2):
            raise ValueError("geometry prediction tracks have the wrong query shape")
        if self.track_visibility.shape[1] != query_count:
            raise ValueError("track_visibility has the wrong query axis")
        if self.track_confidence.shape[1] != query_count:
            raise ValueError("track_confidence has the wrong query axis")


class GeometryBackend(Protocol):
    name: str

    def predict(
        self,
        images: Sequence[Image.Image],
        query_points: np.ndarray,
    ) -> GeometryPrediction:
        ...
