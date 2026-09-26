"""Configuration for the MultiMemBench V2 evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


_TRACKS = {"static_3d"}


@dataclass
class V2EvaluationConfig:
    protocol_version: str = "2.0"
    track: str = "static_3d"
    max_instances: int = 256

    embedding_weight: float = 0.75
    attribute_weight: float = 0.10
    category_weight: float = 0.10
    track_continuity_weight: float = 0.05
    min_identity_similarity: float = 0.35
    min_identity_margin: float = 0.03
    ambiguity_epsilon: float = 0.03
    duplicate_similarity_threshold: float = 0.80

    min_visible_fraction: float = 0.25
    min_mask_area_px: float = 1500.0
    min_correspondences: int = 32
    min_spatial_coverage: float = 0.20
    min_inlier_ratio: float = 0.60
    max_reprojection_error_ratio: float = 0.02
    min_valid_depth_ratio: float = 0.70
    min_baseline_ratio: float = 0.01
    min_geometry_confidence: float = 0.65
    min_valid_geometry_frames: int = 3
    static_coverage_eligibility: float = 0.50
    static_observability_eligibility: float = 0.50
    require_cycle_consistency: bool = False

    min_camera_confidence: float = 0.50
    min_projected_area_px: float = 1500.0
    min_occlusion_depth_confidence: float = 0.50
    min_occlusion_depth_coverage: float = 0.50
    occlusion_depth_tolerance_ratio: float = 0.05
    min_direct_occlusion_fraction: float = 0.50
    min_temporal_occlusion_fraction: float = 0.25
    max_temporal_occlusion_gap: int = 16

    ransac_iterations: int = 128
    rigid_inlier_threshold_ratio: float = 0.05
    geometry_error_scale: float = 0.10
    topology_error_scale: float = 0.15
    topology_noise_floor: float = 0.05
    topology_min_edge_ratio: float = 0.03
    max_points_per_instance: int = 512
    queries_per_instance: int = 64
    background_queries: int = 512
    random_seed: int = 0
    geometry_backend: str = "vggt"
    geometry_model_id: str = "facebook/VGGT-1B"
    geometry_device: str = "cuda"
    geometry_dtype: str = "bfloat16"
    geometry_image_size: int = 518
    geometry_strict: bool = False
    geometry_checkpoint: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "V2EvaluationConfig":
        raw = data or {}
        allowed = {item.name for item in fields(cls)}
        config = cls(**{key: value for key, value in raw.items() if key in allowed})
        config.validate()
        return config

    def validate(self) -> None:
        if self.track not in _TRACKS:
            raise ValueError(f"track must be one of {sorted(_TRACKS)}, got {self.track!r}")
        for name in (
            "min_identity_similarity",
            "min_identity_margin",
            "duplicate_similarity_threshold",
            "min_visible_fraction",
            "min_spatial_coverage",
            "min_inlier_ratio",
            "max_reprojection_error_ratio",
            "min_valid_depth_ratio",
            "min_geometry_confidence",
            "static_coverage_eligibility",
            "static_observability_eligibility",
            "min_camera_confidence",
            "min_occlusion_depth_confidence",
            "min_occlusion_depth_coverage",
            "occlusion_depth_tolerance_ratio",
            "min_direct_occlusion_fraction",
            "min_temporal_occlusion_fraction",
            "topology_noise_floor",
            "topology_min_edge_ratio",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_correspondences < 1 or self.min_valid_geometry_frames < 1:
            raise ValueError("correspondence and frame thresholds must be positive")
        if self.min_projected_area_px < 0.0:
            raise ValueError("min_projected_area_px must be non-negative")
        if self.max_temporal_occlusion_gap < 0:
            raise ValueError("max_temporal_occlusion_gap must be non-negative")
        if self.topology_error_scale <= 0.0:
            raise ValueError("topology_error_scale must be positive")
        if isinstance(self.max_instances, bool) or not isinstance(self.max_instances, int) or self.max_instances < 1:
            raise ValueError("max_instances must be a positive integer")
        if self.geometry_image_size < 14:
            raise ValueError("geometry_image_size must be at least 14")
        backend = self.geometry_backend.lower()
        if backend == "vggt" and self.geometry_image_size % 14 != 0:
            raise ValueError("VGGT geometry_image_size must be divisible by 14")
        if backend in {"vggt_omega", "omega"} and self.geometry_image_size % 16 != 0:
            raise ValueError("VGGT-Omega geometry_image_size must be divisible by 16")
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
