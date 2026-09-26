"""Auditable observability gates for conditional 3D scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from multimem_bench.v2.config import V2EvaluationConfig
from multimem_bench.v2.math_utils import clamp01


@dataclass
class GateEvidence:
    association_confidence: float = 0.0
    visible_fraction: float = 0.0
    mask_area_px: float = 0.0
    correspondence_count: int = 0
    spatial_coverage: float = 0.0
    track_confidence: float = 0.0
    valid_depth_ratio: float = 0.0
    depth_confidence: float = 0.0
    pose_inlier_ratio: float = 0.0
    reprojection_error_ratio: float = math.inf
    camera_confidence: float = 0.0
    baseline_ratio: float = 0.0
    cycle_consistency: float | None = None
    generation_failure: bool = False
    failure_reason: str | None = None
    evaluator_error: str | None = None


@dataclass(frozen=True)
class GateDecision:
    state: str
    scored: bool
    confidence: float
    components: dict[str, float]
    reasons: tuple[str, ...] = field(default_factory=tuple)


def evaluate_gate(
    evidence: GateEvidence,
    config: V2EvaluationConfig | None = None,
) -> GateDecision:
    cfg = config or V2EvaluationConfig()
    if evidence.generation_failure:
        return GateDecision(
            state="generation_failure",
            scored=False,
            confidence=0.0,
            components={},
            reasons=(evidence.failure_reason or "generation_failure",),
        )
    if evidence.evaluator_error:
        return GateDecision(
            state="evaluator_failure",
            scored=False,
            confidence=0.0,
            components={},
            reasons=(evidence.evaluator_error,),
        )

    reasons: list[str] = []
    if evidence.visible_fraction < cfg.min_visible_fraction:
        reasons.append("insufficient_visibility")
    if evidence.mask_area_px < cfg.min_mask_area_px:
        reasons.append("instance_too_small")
    if evidence.correspondence_count < cfg.min_correspondences:
        reasons.append("insufficient_correspondences")
    if evidence.spatial_coverage < cfg.min_spatial_coverage:
        reasons.append("insufficient_spatial_coverage")
    if evidence.valid_depth_ratio < cfg.min_valid_depth_ratio:
        reasons.append("insufficient_valid_depth")
    if evidence.pose_inlier_ratio < cfg.min_inlier_ratio:
        reasons.append("unstable_pose_fit")
    if evidence.reprojection_error_ratio > cfg.max_reprojection_error_ratio:
        reasons.append("high_reprojection_error")
    if evidence.baseline_ratio < cfg.min_baseline_ratio:
        reasons.append("insufficient_baseline")
    if cfg.require_cycle_consistency and evidence.cycle_consistency is None:
        reasons.append("cycle_consistency_unavailable")

    visibility = math.sqrt(
        _threshold_score(evidence.visible_fraction, cfg.min_visible_fraction)
        * _threshold_score(evidence.mask_area_px, cfg.min_mask_area_px)
    )
    depth = math.sqrt(
        _threshold_score(evidence.valid_depth_ratio, cfg.min_valid_depth_ratio)
        * clamp01(evidence.depth_confidence)
    )
    reprojection = math.exp(
        -max(0.0, evidence.reprojection_error_ratio)
        / max(cfg.max_reprojection_error_ratio, 1e-12)
    )
    components = {
        "association": clamp01(evidence.association_confidence),
        "visibility": clamp01(visibility),
        "track": clamp01(evidence.track_confidence),
        "depth": clamp01(depth),
        "pose": clamp01(
            math.sqrt(clamp01(evidence.pose_inlier_ratio) * clamp01(reprojection))
        ),
        "camera": clamp01(evidence.camera_confidence),
        "baseline": _threshold_score(evidence.baseline_ratio, cfg.min_baseline_ratio),
    }
    if evidence.cycle_consistency is not None:
        components["cycle"] = clamp01(evidence.cycle_consistency)
    confidence = _geometric_mean(components.values())
    if confidence < cfg.min_geometry_confidence:
        reasons.append("low_combined_confidence")

    if reasons:
        return GateDecision(
            state="not_observable",
            scored=False,
            confidence=confidence,
            components=components,
            reasons=tuple(dict.fromkeys(reasons)),
        )
    return GateDecision(
        state="scored",
        scored=True,
        confidence=confidence,
        components=components,
    )


def _threshold_score(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 1.0
    return clamp01(float(value) / threshold)


def _geometric_mean(values) -> float:
    items = [clamp01(value) for value in values]
    if not items or any(value <= 0.0 for value in items):
        return 0.0
    return math.exp(sum(math.log(value) for value in items) / len(items))
