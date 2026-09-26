from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
import math


@dataclass
class V3EvaluationConfig:
    protocol_version: str = "3.0"
    geometry_mode: str = "input_grounded"
    monocular_model_id: str = "Ruicheng/moge-2-vitl"
    monocular_device: str = "cuda"
    monocular_resolution_level: int = 5
    reference_query_count: int = 128
    reference_mask_margin_px: int = 3
    min_background_matches: int = 12
    min_background_coverage: float = 0.02
    max_depth_alignment_error: float = 0.15
    min_depth_alignment_samples: int = 8
    max_depth_scale_ratio: float = 4.0
    matcher: str = "mast3r"
    matcher_model_id: str = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    matcher_device: str = "cuda"
    matcher_image_size: int = 512
    min_matches: int = 12
    min_inlier_ratio: float = 0.45
    max_reprojection_error_px: float = 8.0
    max_normalized_reprojection_error: float = 0.05
    min_reference_coverage: float = 0.10
    ratio_test: float = 0.80
    ransac_iterations: int = 1000
    confidence: float = 0.999
    max_instances: int = 256
    min_identity_similarity: float = 0.35
    min_embedding_cosine_similarity: float = 0.05
    ambiguity_margin_threshold: float = 0.05
    min_observable_area_px: float = 16.0
    min_visibility_camera_confidence: float = 0.65
    min_visibility_depth_confidence: float = 0.65
    min_visibility_depth_coverage: float = 0.70
    min_visible_fraction: float = 0.30
    min_occluded_fraction: float = 0.70
    occlusion_depth_tolerance: float = 0.05
    allow_unlabeled_geometry: bool = False
    allow_opencv_fallback: bool = False
    min_geometry_coverage: float | None = None
    min_geometry_frames: int = 3
    presence_weight: float = 0.2
    identity_weight: float = 0.4
    geometry_weight: float = 0.4
    provisional: bool = True
    min_motion_displacement_fraction: float = 0.01
    min_motion_frame_fraction: float = 0.50

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "V3EvaluationConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(data or {}) - allowed)
        if unknown:
            raise ValueError(f"unknown V3 config fields: {', '.join(unknown)}")
        config = cls(**dict(data or {}))
        config.validate()
        return config

    def validate(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite")
        weights = (self.presence_weight, self.identity_weight, self.geometry_weight)
        if any(value < 0 for value in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
            raise ValueError("Full-IGM weights must be non-negative and sum to one")
        if self.provisional is not True:
            raise ValueError("provisional must remain true until calibration is frozen")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("RANSAC confidence must be strictly between zero and one")
        if self.geometry_mode not in {"input_grounded", "pairwise_diagnostic"}:
            raise ValueError("geometry_mode must be input_grounded or pairwise_diagnostic")
        local_model = Path(self.monocular_model_id)
        if self.monocular_model_id != "Ruicheng/moge-2-vitl" and not (local_model.is_file() and local_model.suffix == ".pt"):
            raise ValueError("monocular_model_id must be Ruicheng/moge-2-vitl or an existing .pt checkpoint")
        if not 0 <= self.monocular_resolution_level <= 9 or self.reference_query_count < 1 or self.reference_mask_margin_px < 0:
            raise ValueError("invalid monocular reference sampling parameters")
        if self.min_background_matches < 1 or self.min_depth_alignment_samples < 1:
            raise ValueError("geometry sample thresholds must be positive")
        if not 0.0 < self.min_background_coverage <= 1.0:
            raise ValueError("min_background_coverage must be in (0, 1]")
        if self.max_depth_alignment_error <= 0 or self.max_depth_scale_ratio <= 1:
            raise ValueError("invalid depth alignment thresholds")
        if self.matcher not in {"auto", "mast3r", "opencv"}:
            raise ValueError("matcher must be auto, mast3r, or opencv")
        if self.min_matches < 4:
            raise ValueError("min_matches must be at least 4")
        for name in (
            "min_inlier_ratio",
            "min_motion_displacement_fraction",
            "min_motion_frame_fraction",
            "confidence",
            "min_reference_coverage",
            "ambiguity_margin_threshold",
            "min_embedding_cosine_similarity",
            "min_identity_similarity",
            "min_visibility_camera_confidence",
            "min_visibility_depth_confidence",
            "min_visibility_depth_coverage",
            "min_visible_fraction",
            "min_occluded_fraction",
            "occlusion_depth_tolerance",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.max_reprojection_error_px <= 0 or self.max_normalized_reprojection_error <= 0 or self.ransac_iterations < 1:
            raise ValueError("invalid RANSAC parameters")
        if self.min_observable_area_px <= 0:
            raise ValueError("min_observable_area_px must be positive")
        if self.min_geometry_frames < 1:
            raise ValueError("min_geometry_frames must be at least 1")
        if self.min_geometry_coverage is not None and not 0.0 < self.min_geometry_coverage <= 1.0:
            raise ValueError("min_geometry_coverage must be null or in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
