"""Configuration knobs for the observation-first evaluator."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class EvaluationConfig:
    """Top-level evaluator configuration.

    The defaults are intentionally conservative for small local windows. Real
    benchmark runs should tune these values on a detector/tracker validation set.
    """

    top_k_anchor_pairs: int = 5
    max_expected_objects: int | None = None
    ambiguity_epsilon: float = 0.03
    min_anchor_similarity: float = 0.20
    slot_mask_iou_threshold: float = 0.10
    slot_expected_coverage_threshold: float = 0.45
    slot_observed_coverage_threshold: float = 0.45
    slot_center_error_ratio_threshold: float = 0.50
    slot_shape_iou_threshold: float = 0.50
    appearance_similarity_threshold: float = 0.55
    min_projected_visible_area_px: float = 16.0
    min_projected_visible_ratio: float = 0.02
    track_reappearance_min_gap_windows: int = 1
    require_masks: bool = True
    zoom_relative_tolerance: float = 0.15
    zoom_dropoff_relative: float = 0.50
    empty_window_policy: str = "previous_view_missing"
    include_attribute_similarity: bool = False
    include_category_similarity: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EvaluationConfig":
        if not data:
            return cls()
        kwargs = dict(data)
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in kwargs.items() if key in allowed})
