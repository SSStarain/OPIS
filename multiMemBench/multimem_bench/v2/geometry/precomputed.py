"""Injectable precomputed geometry backend for offline runs and tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from multimem_bench.v2.geometry.base import GeometryPrediction


class PrecomputedGeometryBackend:
    name = "precomputed"

    def __init__(self, prediction: GeometryPrediction) -> None:
        self.prediction = prediction

    def predict(
        self,
        images: Sequence[Image.Image],
        query_points: np.ndarray,
    ) -> GeometryPrediction:
        self.prediction.validate(frame_count=len(images), query_count=len(query_points))
        return self.prediction
