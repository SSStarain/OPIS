from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TYPE_CHECKING
from pathlib import Path
import json
import numpy as np

from multimem_bench.schema import ReferenceScene, VideoObservation
from multimem_bench.similarity import cosine_similarity
from .config import V3EvaluationConfig
from .geometry import V3GeometryArtifact
from .matching import V3MatchResult, solve_partial_matching
from .metrics import aggregate_core_2d, aggregate_score_tracks, aggregate_structure_quality
from .observability import classify_visibility

if TYPE_CHECKING:
    from .structure import V3StructureArtifact

METRIC_CONTRACT_VERSION = "3.11-strict-frame-presence-provisional"
# Backward-compatible import name for callers that only need the evaluator
# contract version; new code should use METRIC_CONTRACT_VERSION.
STRUCTURE_SCORE_VERSION = METRIC_CONTRACT_VERSION


@dataclass
class V3FrameResult:
    frame_index: int | None
    reference_id: str
    observed_id: str | None
    state: str
    identity: float | None
    presence: float | None
    visible_surface_geometry: float | None
    scale_consistency: float | None
    local_topology: float | None
    correspondence_coverage: float | None
    inlier_ratio: float | None
    reprojection_error_px: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    input_structure: float | None = None
    structure_coverage: float | None = None


@dataclass
class V3EvaluationResult:
    protocol_version: str
    scene_id: str
    video_id: str
    metrics: dict[str, Any]
    frames: list[V3FrameResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_v3(scene: ReferenceScene, observation: VideoObservation, geometry: V3GeometryArtifact,
                config: V3EvaluationConfig | None = None, *,
                structure: V3StructureArtifact | None = None) -> V3EvaluationResult:
    cfg = config or V3EvaluationConfig()
    cfg.validate()
    if len({w.window_id for w in observation.windows}) != len(observation.windows):
        raise ValueError("window_id must be unique")
    if structure is not None:
        structure.validate_context(scene, observation, cfg, geometry=geometry)
    rows: list[V3FrameResult] = []
    matching_frames = []
    for ordinal, window in enumerate(observation.windows):
        frame = geometry.frames.get(window.frame_index) if window.frame_index is not None else None
        tracks = {}
        required = {}
        initial = {}
        for ref_id, reference in scene.objects.items():
            explicit = isinstance(reference.metadata.get("mobility"), dict)
            mobility = reference.mobility
            tracks[ref_id] = mobility.evaluation_track if explicit else ("legacy" if cfg.allow_unlabeled_geometry else "unlabeled")
            if not explicit and mobility.kinematic_class in {"articulated", "deformable"}:
                tracks[ref_id] = "dynamic_identity"
            required[ref_id] = tracks[ref_id] in {"legacy", "static_geometry"} and (not explicit or mobility.geometry_eligible)
            initial[ref_id] = classify_visibility(reference=reference, frame_geometry=frame, observed=None,
                config=cfg, geometry_required=required[ref_id], frame_size=window.frame_size)
        candidates = {key: ref for key, ref in scene.objects.items()
                      if initial[key].state not in {"outside_view", "occluded", "too_small"}}
        matching = solve_partial_matching(candidates, window.objects, cfg)
        observed_by_id = {item.observed_id: item for item in window.objects}
        frame_rows = []
        for ref_id, reference in scene.objects.items():
            observed = observed_by_id.get(matching.pairs.get(ref_id))
            visibility = classify_visibility(reference=reference, frame_geometry=frame, observed=observed,
                config=cfg, geometry_required=required[ref_id], frame_size=window.frame_size)
            diagnostics = {
                **visibility.diagnostics, **_match_diagnostics(matching, ref_id),
                "window_id": window.window_id, "window_ordinal": ordinal,
                "evaluation_track": tracks[ref_id], "geometry_candidate": required[ref_id],
                "structure_candidate": tracks[ref_id] == "dynamic_identity",
                "structure_eligible": False,
                "geometry_eligible": False, "presence_eligible": visibility.presence_eligible,
                "visibility_state": visibility.state, "visibility_confidence": visibility.confidence,
                "reason": visibility.reason, "ambiguity_group": reference.metadata.get("ambiguity_group"),
                "observed_track_id": observed.track_id if observed else None,
            }
            row = V3FrameResult(window.frame_index, ref_id, observed.observed_id if observed else None,
                "evaluator_failure", None, None, None, None, None, None, None, None, diagnostics)
            frame_rows.append(row)
            if not visibility.presence_eligible:
                row.state = "evaluator_failure" if visibility.state in {"evaluator_failure", "unreliable_camera_pose"} else "not_observable"
                continue
            if observed is None:
                row.state, row.identity, row.presence = "generation_failure", 0., 0.
                diagnostics["reason"] = "expected_visible_missing"
                continue
            row.identity, row.presence = _identity(reference.embedding, observed.embedding), 1.
            if not required[ref_id]:
                row.state = "dynamic_identity" if tracks[ref_id] == "dynamic_identity" else "identity_only"
                if tracks[ref_id] == "dynamic_identity" and structure is not None:
                    if ref_id in matching.ambiguous_reference_ids:
                        evidence = {"reason": "ambiguous_association"}
                    else:
                        evidence = structure.score(
                            window.window_id, ref_id, observed.observed_id)
                        row.input_structure = evidence["score"]
                        row.structure_coverage = evidence["coverage"]
                    diagnostics.update(structure_eligible=row.input_structure is not None,
                                       structure_evidence=evidence)
                continue
            if not visibility.geometry_eligible or frame is None or frame.status != "scored":
                diagnostics["reason"] = "geometry_unavailable_for_visible_instance"
                continue
            selected, coverage = _instance_correspondences(frame, reference.bbox, observed.bbox)
            row.correspondence_coverage = coverage
            if int(selected.sum()) < 4:
                diagnostics["reason"] = "insufficient_instance_correspondences"
                continue
            row.inlier_ratio = float(np.asarray(frame.inlier_mask, dtype=bool)[selected].mean())
            raw = _reprojection_error(frame, selected)
            normalized = _normalised_reprojection_error(frame, observed.bbox, selected)
            if not np.isfinite(normalized):
                diagnostics["reason"] = "invalid_or_unknown_residual_units"
                continue
            row.state = "scored"
            row.reprojection_error_px = raw
            row.visible_surface_geometry = float(np.exp(-normalized / cfg.max_normalized_reprojection_error))
            diagnostics.update(geometry_eligible=True, reason="projective_consistency_measured",
                geometry_level=frame.diagnostics.get("geometry_level", "projective_2d" if frame.homography is not None else "unknown"),
                raw_reprojection_error_px=raw, normalized_reprojection_error=normalized,
                error_space="source_pixels", residual_support="all_instance_candidate_correspondences")
        # An unknown missing reference could explain an unmatched detection. Keep it
        # diagnostic until every reference has usable visibility/association evidence.
        fp_eligible = bool(frame_rows) and all(r.presence is not None or
            r.diagnostics["visibility_state"] in {"outside_view", "occluded"} for r in frame_rows)
        unmatched_count = len(matching.unmatched_observed_ids) if fp_eligible else 0
        for row in frame_rows:
            row.diagnostics.update(
                unmatched_observed_count=unmatched_count,
                unmatched_observed_candidate_count=len(matching.unmatched_observed_ids),
                false_positive_evaluable=fp_eligible,
            )
        matching_frames.append({"window_id": window.window_id, "frame_index": window.frame_index,
            "unmatched_observed_ids": list(matching.unmatched_observed_ids), "false_positive_evaluable": fp_eligible})
        rows.extend(frame_rows)
    metrics = _aggregate(rows, cfg)
    if structure is not None:
        metrics.update(aggregate_structure_quality(rows))
    valid_frames = len({_geometry_window(r) for r in rows if _geometry_eligible(r)})
    motion_diagnostics = geometry.metadata.get("motion_diagnostics", {})
    motion_diagnostics = motion_diagnostics if isinstance(motion_diagnostics, dict) else {}
    motion_valid = motion_diagnostics.get("motion_valid")
    motion_valid = motion_valid if isinstance(motion_valid, bool) else None
    tracks = aggregate_score_tracks(metrics, valid_geometry_frames=valid_frames,
                                    motion_valid=motion_valid, config=cfg)
    for name in ("full_igm_diagnostic_score", "full_igm_score", "headline_score"):
        metrics[name] = tracks[name]
    eligible, reason = tracks["headline_eligible"], tracks["headline_eligibility_reason"]
    metadata = {
        "metric_contract_version": METRIC_CONTRACT_VERSION, "thresholds_provisional": True,
        "identity_calibration": {"policy": "nonnegative_raw_cosine_v1", "baseline": 0.5,
                                 "formula": "max(0, 2 * shifted_cosine - 1)",
                                 "scope": "per_identity_row_before_aggregation",
                                 "human_calibrated": False},
        "reference_only_ground_truth": True, "fixed_reference_geometry": bool(geometry.metadata.get("fixed_reference_geometry")),
        "reference_geometry_is_estimated": True,
        "reference_fingerprint": geometry.metadata.get("reference_fingerprint"),
        "pointmap_conditioning": "input_only" if geometry.metadata.get("fixed_reference_geometry") else "reference_and_current",
        "independent_pairwise_geometry": True,
        "joint_vggt_used": False, "matching_strategy": "global_partial_hungarian",
        "headline_eligible": eligible, "headline_eligibility_reason": reason,
        "motion_valid": motion_valid, "motion_diagnostics": motion_diagnostics,
        "full_igm_diagnostic_eligible": tracks["full_igm_diagnostic_eligible"],
        "full_igm_diagnostic_reason": tracks["full_igm_diagnostic_reason"],
        "full_igm_score_mode": tracks["full_igm_score_mode"],
        "effective_score_weights": tracks["effective_score_weights"],
        "score_gates_enabled": tracks["score_gates_enabled"],
        "score_weights": {"presence": cfg.presence_weight, "identity": cfg.identity_weight,
                          "geometry": cfg.geometry_weight, "provisional": True},
        "full_igm_geometry_component": "v3_reference_projective_score",
        "structure_enabled": structure is not None,
        "full_igm_third_component": ("v3_structural_quality_score" if structure is not None else
                                     "v3_reference_projective_score"),
        "coverage_breakdown": {
            "n_rows": len(rows),
            "n_presence_evaluable": sum(r.presence is not None for r in rows),
            "n_identity_evaluable": sum(r.identity is not None and r.presence == 1. for r in rows),
            "n_geometry_candidates": sum(bool(r.diagnostics.get("geometry_candidate")) for r in rows),
            "n_geometry_valid": sum(_geometry_eligible(r) for r in rows),
            "n_structure_candidates": sum(bool(r.diagnostics.get("structure_candidate")) for r in rows),
            "n_structure_valid": sum(r.input_structure is not None for r in rows),
            "states": {state: sum(r.state == state for r in rows) for state in sorted({r.state for r in rows})},
        },
        "aggregation": {"case_id": scene.scene_id, "n_cases": 1, "bootstrap_unit": "case",
                        "missingness": {name: {"n_valid": int(metrics[name] is not None),
                                               "n_missing": int(metrics[name] is None)}
                                        for name in ("core_2d_score", "full_igm_score", "full_igm_diagnostic_score")}},
        "valid_geometry_frames": valid_frames, "min_geometry_coverage": cfg.min_geometry_coverage,
        "min_geometry_frames": cfg.min_geometry_frames, "matching_frames": matching_frames,
        "geometry_coverage_denominator": "all_requested_static_instance_windows",
        "geometry_quality_denominator": "valid_instance_scores_only",
        "identity_swap_policy": "explicit_track_diagnostic_only_not_in_core_2d",
        "deprecated_metric_aliases": {"v3_pairwise_geometry_score": "v3_reference_projective_score"},
        "track_metrics": {track: aggregate_core_2d([r for r in rows if r.diagnostics["evaluation_track"] == track])
                          for track in sorted({r.diagnostics["evaluation_track"] for r in rows})},
    }
    if len(metadata["track_metrics"]) > 1 and any(f["unmatched_observed_ids"] for f in matching_frames):
        for track_name in metadata["track_metrics"]:
            metadata["track_metrics"][track_name] = aggregate_core_2d(
                [r for r in rows if r.diagnostics["evaluation_track"] == track_name],
                fp_attributable=False)
            metadata["track_metrics"][track_name]["presence_false_positive"] = None
        metadata["track_false_positive_policy"] = "unassigned_extras_not_attributable_to_individual_tracks"
    if structure is not None:
        metadata["structure_protocol"] = {
            "reference": "initial_image_only", "comparison": "independent_initial_current_pair",
            "evidence_type": structure.metadata["evidence_type"], "human_calibrated": False,
            "pose_protocol": structure.metadata.get("pose_protocol"),
            "fallback_protocol": structure.metadata.get("fallback_protocol"),
            "pose_routing_version": structure.metadata.get("pose_routing_version"),
            "routes": structure.metadata.get("routes"),
            "producer_errors": structure.metadata.get("errors", []),
            "quality_aggregation": "per_instance_beta_prior_shrinkage_then_macro",
            "raw_quality_aggregation": "available_windows_then_observed_instance_macro",
            "calibration": "prior_mean_0.5_strength_2.0",
            "coverage_denominator": "all_requested_dynamic_instance_windows",
            "unknown_policy": "exclude_from_raw_quality_prior_only_in_shrunk_score",
            "third_component": "static_geometry_or_dynamic_input_structure_per_instance",
        }
        evidence_rows = [r.diagnostics.get("structure_evidence", {}) or {} for r in rows]
        metadata["structure_protocol"]["backend_counts"] = {
            backend: sum(e.get("backend") == backend and e.get("score") is not None for e in evidence_rows)
            for backend in ("pose", "vlm", "vlm_fallback")
        }
        metadata["structure_protocol"]["fallback_attempted_rows"] = sum(bool(e.get("fallback_attempted")) for e in evidence_rows)
        metadata["structure_protocol"]["fallback_unscored_rows"] = sum(
            bool(e.get("fallback_attempted")) and e.get("score") is None for e in evidence_rows)
        for track_name, track_metrics in metadata["track_metrics"].items():
            track_rows = [r for r in rows if r.diagnostics["evaluation_track"] == track_name]
            track_metrics.update(aggregate_structure_quality(track_rows))
            track_metrics.update(aggregate_score_tracks(
                track_metrics, valid_geometry_frames=len({_geometry_window(r) for r in track_rows
                                                          if _geometry_eligible(r)}),
                motion_valid=motion_valid, config=cfg))
    return V3EvaluationResult(cfg.protocol_version, scene.scene_id, observation.video_id, metrics, rows, metadata)


def write_v3_result(result: V3EvaluationResult, output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary, per_frame = root / "summary_v3.json", root / "per_frame_metrics_v3.jsonl"
    summary_text = json.dumps(result.to_dict(), indent=2, allow_nan=False)
    frame_text = "".join(json.dumps(asdict(row), allow_nan=False) + "\n" for row in result.frames)
    summary.write_text(summary_text, encoding="utf-8")
    per_frame.write_text(frame_text, encoding="utf-8")
    return {"summary": summary, "per_frame": per_frame}


def _match_diagnostics(result: V3MatchResult, reference_id: str) -> dict[str, Any]:
    return {"match_score": result.scores.get((reference_id, result.pairs.get(reference_id))),
            "match_margin": result.margins.get(reference_id),
            "ambiguous_match": reference_id in result.ambiguous_reference_ids}


def _identity(reference: list[float] | None, observed: list[float] | None) -> float | None:
    """Remove the shifted-cosine baseline for scoring, leaving matching unchanged."""
    if not reference or not observed or len(reference) != len(observed):
        return None
    if not np.isfinite(reference).all() or not np.isfinite(observed).all():
        return None
    value = cosine_similarity(reference, observed)
    return max(0.0, 2.0 * float(value) - 1.0) if value is not None and np.isfinite(value) else None


def _instance_correspondences(frame: Any, reference_bbox: Any, observed_bbox: Any) -> tuple[np.ndarray, float]:
    if len(frame.reference_pixels) == 0:
        return np.zeros(0, dtype=bool), 0.
    selected = _inside_bbox(frame.reference_pixels, reference_bbox) & _inside_bbox(frame.current_pixels, observed_bbox)
    # Spatial support, not the RANSAC inlier ratio. No inlier-selection bias in residuals.
    coverage = 0.
    if selected.sum() >= 3 and reference_bbox is not None:
        import cv2
        hull = cv2.convexHull(np.asarray(frame.reference_pixels[selected], dtype=np.float32))
        x0, y0, x1, y1 = reference_bbox
        coverage = min(1., float(cv2.contourArea(hull)) / max((x1-x0)*(y1-y0), 1e-6))
    return selected, coverage


def _inside_bbox(points: np.ndarray, bbox: Any) -> np.ndarray:
    if bbox is None:
        return np.ones(len(points), dtype=bool)
    x0, y0, x1, y1 = (float(value) for value in bbox)
    return (points[:,0] >= x0) & (points[:,0] <= x1) & (points[:,1] >= y0) & (points[:,1] <= y1)


def _reprojection_error(frame: Any, inliers: np.ndarray | None = None) -> float:
    selected = np.asarray(frame.inlier_mask if inliers is None else inliers, dtype=bool)
    if frame.reprojection_errors is not None:
        if getattr(frame, "error_space", "unknown") != "source_pixels":
            return float("inf")
        error = np.asarray(frame.reprojection_errors, dtype=float)
    elif frame.matcher == "opencv" and frame.homography is not None and len(frame.reference_pixels):
        import cv2
        projected = cv2.perspectiveTransform(frame.reference_pixels.reshape(-1,1,2).astype(np.float32), frame.homography).reshape(-1,2)
        error = np.linalg.norm(projected-frame.current_pixels, axis=1)
    else:
        return float("inf")
    if error.shape != selected.shape or not selected.any() or not np.isfinite(error[selected]).all():
        return float("inf")
    return float(np.median(error[selected]))


def _normalised_reprojection_error(frame: Any, bbox: Any, inliers: np.ndarray | None = None) -> float:
    if bbox is None:
        return float("inf")
    x0, y0, x1, y1 = (float(value) for value in bbox)
    diagonal = float(np.hypot(x1-x0, y1-y0))
    if x1 <= x0 or y1 <= y0 or not np.isfinite(diagonal):
        return float("inf")
    return _reprojection_error(frame, inliers) / diagonal


def _aggregate(rows: list[V3FrameResult], cfg: V3EvaluationConfig) -> dict[str, float | None]:
    def mean(name: str, diagnostic: bool = False) -> float | None:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            value = row.diagnostics.get(name) if diagnostic else getattr(row, name)
            if _geometry_eligible(row) and value is not None and np.isfinite(value):
                grouped.setdefault(row.reference_id, []).append(float(value))
        return float(np.mean([np.mean(values) for values in grouped.values()])) if grouped else None
    scored = [row for row in rows if _geometry_eligible(row)]
    candidates = [row for row in rows if row.diagnostics.get("geometry_candidate")]
    geometry = mean("visible_surface_geometry")
    return {**aggregate_core_2d(rows), "v3_reference_projective_score": geometry,
        "v3_pairwise_geometry_score": geometry, "v3_correspondence_coverage": mean("correspondence_coverage"),
        "v3_inlier_ratio": mean("inlier_ratio"), "v3_mean_reprojection_error_px": mean("reprojection_error_px"),
        "v3_normalized_reprojection_error": mean("normalized_reprojection_error", True),
        "v3_geometry_coverage": len(scored)/len(candidates) if candidates else None,
        "v3_geometry_candidate_rows": float(len(candidates)), "full_igm_score": None, "headline_score": None}


def _headline_eligibility(rows: list[V3FrameResult], metrics: dict[str, float | None],
                          cfg: V3EvaluationConfig) -> tuple[bool, str, int]:
    valid = len({r.diagnostics.get("window_id", r.frame_index) for r in rows if _geometry_eligible(r)})
    coverage = metrics.get("v3_geometry_coverage")
    if cfg.min_geometry_coverage is not None and (coverage is None or coverage < cfg.min_geometry_coverage):
        return False, "geometry_coverage_below_threshold", valid
    if valid < cfg.min_geometry_frames:
        return False, "valid_geometry_frames_below_threshold", valid
    return False, "protocol_not_calibrated", valid


def _geometry_eligible(row: V3FrameResult) -> bool:
    return row.state == "scored" and row.diagnostics.get("geometry_eligible", True) is not False


def _geometry_window(row: V3FrameResult) -> Any:
    return row.diagnostics.get("window_id", row.frame_index)
