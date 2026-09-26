"""Pure-initial-frame Static-3D evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np

from multimem_bench.io import write_json, write_jsonl
from multimem_bench.schema import ObservedObject, ReferenceScene, VideoObservation
from multimem_bench.similarity import cosine_similarity
from multimem_bench.vision.mask_utils import mask_area, mask_to_bool_array
from multimem_bench.v2.association import (
    AssociationBundle,
    AssociationMatch,
    FrameAssociation,
    associate_instances,
)
from multimem_bench.v2.confidence import GateDecision, GateEvidence, evaluate_gate
from multimem_bench.v2.config import V2EvaluationConfig
from multimem_bench.v2.geometry.artifact import GeometryBundle
from multimem_bench.v2.math_utils import clamp01
from multimem_bench.v2.metrics import geometric_mean, macro_instance_mean, mean
from multimem_bench.v2.normalization import (
    CanonicalizationResult,
    canonicalize_instance,
    score_articulated_keypoints,
)
from multimem_bench.v2.observability import (
    ObservabilityMap,
    ReferenceObservability,
    assess_static_observability,
    association_candidates,
    refine_temporal_occlusions,
)
from multimem_bench.v2.schema import ReferenceAnnotation, ReferenceInstance


@dataclass
class InstanceFrameResult:
    track: str
    window_id: str
    frame_index: int | None
    reference_id: str
    observed_id: str | None
    state: str
    reasons: tuple[str, ...]
    confidence: float
    metrics: dict[str, float | None]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackReport:
    track: str
    headline_score: float | None
    eligible: bool
    geometry_coverage: float
    geometry_candidate_rows: int
    baseline_excluded_rows: int
    valid_geometry_frames: int
    observability_coverage: float
    observable_rows: int
    total_reference_rows: int
    metrics: dict[str, float | None]
    error_counts: dict[str, int]
    state_counts: dict[str, int]
    abstention_reasons: dict[str, int]
    per_instance: dict[str, dict[str, float | None]]
    confidence: dict[str, float]


@dataclass
class V2EvaluationResult:
    scene_id: str
    video_id: str
    reports: dict[str, TrackReport]
    associations: dict[str, AssociationBundle]
    instance_frames: list[InstanceFrameResult]
    state_counts: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "scene_id": self.scene_id,
            "video_id": self.video_id,
            "reports": {name: asdict(report) for name, report in self.reports.items()},
            "state_counts": dict(self.state_counts),
            "metadata": dict(self.metadata),
        }


@dataclass
class _GeometryComparison:
    gate: GateDecision
    canonical: CanonicalizationResult | None
    reference_points: np.ndarray | None
    current_points: np.ndarray | None
    diagnostics: dict[str, Any]


def evaluate_v2(
    scene: ReferenceScene,
    observation: VideoObservation,
    geometry: GeometryBundle,
    config: V2EvaluationConfig | None = None,
) -> V2EvaluationResult:
    cfg = config or V2EvaluationConfig()
    cfg.validate()
    if len(scene.objects) > cfg.max_instances:
        raise ValueError(
            f"reference contains {len(scene.objects)} instances, exceeding "
            f"max_instances={cfg.max_instances}"
        )
    geometry.validate()
    annotation = ReferenceAnnotation.from_reference_scene(scene)
    tracks = [cfg.track]
    associations: dict[str, AssociationBundle] = {}
    all_rows: list[InstanceFrameResult] = []
    reports: dict[str, TrackReport] = {}

    for track in tracks:
        observability = assess_static_observability(
            scene, observation, geometry, cfg
        )
        association = associate_instances(
            scene,
            observation,
            cfg,
            expected_reference_ids=association_candidates(observability),
        )
        observability = refine_temporal_occlusions(
            observability, association, cfg
        )
        associations[track] = association
        rows, event_counts = _evaluate_track(
            track,
            scene,
            annotation,
            observation,
            geometry,
            association,
            observability,
            cfg,
        )
        all_rows.extend(rows)
        reports[track] = _aggregate_track(
            track,
            rows,
            association,
            event_counts,
            cfg,
        )

    state_counts = _count_values(row.state for row in all_rows)
    return V2EvaluationResult(
        scene_id=scene.scene_id,
        video_id=observation.video_id,
        reports=reports,
        associations=associations,
        instance_frames=all_rows,
        state_counts=state_counts,
        metadata={
            "geometry_backend": geometry.backend,
            "geometry_status": geometry.status,
            "geometry_error": geometry.error,
            "initial_frame_only_ground_truth": True,
            "cross_track_overall_score": False,
        },
    )


def write_v2_result(
    result: V2EvaluationResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary = write_json(root / "summary_v2.json", result.to_dict())
    associations = write_json(
        root / "associations_v2.json",
        {name: bundle.to_dict() for name, bundle in result.associations.items()},
    )
    per_instance = write_jsonl(
        root / "per_instance_metrics_v2.jsonl",
        [asdict(row) for row in result.instance_frames],
    )
    return {
        "summary": summary,
        "associations": associations,
        "per_instance": per_instance,
    }


def _evaluate_track(
    track: str,
    scene: ReferenceScene,
    annotation: ReferenceAnnotation,
    observation: VideoObservation,
    geometry: GeometryBundle,
    association: AssociationBundle,
    observability: ObservabilityMap,
    config: V2EvaluationConfig,
) -> tuple[list[InstanceFrameResult], dict[str, int]]:
    windows = {window.window_id: window for window in observation.windows}
    rows: list[InstanceFrameResult] = []
    event_counts = {
        "missing": 0,
        "duplicate": 0,
        "hallucination": 0,
        "replacement": 0,
        "merge": 0,
        "unmatched_observed": 0,
    }
    for frame in association.frames:
        window = windows[frame.window_id]
        frame_observability = observability[frame.window_id]
        observed_by_id = {item.observed_id: item for item in window.objects}
        matches_by_ref = {item.reference_id: item for item in frame.matches}
        judged_missing = [
            reference_id
            for reference_id in frame.missing_reference_ids
            if frame_observability[reference_id].state == "observable"
        ]
        frame_events, missing_reasons = _classify_frame_events(
            track,
            scene,
            window,
            frame,
            geometry,
            config,
            missing_reference_ids=judged_missing,
        )
        for name, count in frame_events.items():
            event_counts[name] += count
        for reference_id in frame.excluded_reference_ids:
            assessment = frame_observability[reference_id]
            rows.append(
                _empty_row(
                    track,
                    frame,
                    reference_id,
                    state=assessment.state,
                    reason=assessment.reason or "outside_reference_view",
                    assessment=assessment,
                )
            )
        for reference_id in frame.missing_reference_ids:
            assessment = frame_observability[reference_id]
            if assessment.state != "observable":
                rows.append(
                    _empty_row(
                        track,
                        frame,
                        reference_id,
                        state=assessment.state,
                        reason=assessment.reason or "not_observable",
                        assessment=assessment,
                    )
                )
                continue
            failure_reason = missing_reasons.get(reference_id, "missing_instance")
            rows.append(
                InstanceFrameResult(
                    track=track,
                    window_id=frame.window_id,
                    frame_index=frame.frame_index,
                    reference_id=reference_id,
                    observed_id=None,
                    state="generation_failure",
                    reasons=(failure_reason,),
                    confidence=0.0,
                    metrics=_missing_metrics(
                        annotation,
                        reference_id,
                    ),
                    diagnostics=_observability_diagnostics(assessment),
                )
            )
        for reference_id, match in matches_by_ref.items():
            observed = observed_by_id[match.observed_id]
            identity = _identity_score(match)
            assessment = frame_observability[reference_id]
            if not assessment.camera_reliable:
                geometry_reason = assessment.reason or geometry.error or "geometry_unavailable"
                rows.append(
                    InstanceFrameResult(
                        track=track,
                        window_id=frame.window_id,
                        frame_index=frame.frame_index,
                        reference_id=reference_id,
                        observed_id=observed.observed_id,
                        state="evaluator_failure",
                        reasons=(geometry_reason,),
                        confidence=0.0,
                        metrics={
                            "presence": 1.0,
                            "identity": identity,
                            "set_identity": 1.0 if match.identity_state == "set_matched" else None,
                            "geometry": None,
                            "scale": None,
                            "topology": None,
                            "part_structure": _part_structure_score(
                                annotation, reference_id, observed
                            ),
                        },
                        diagnostics={
                            "identity_state": match.identity_state,
                            "association_score": match.score,
                            "association_margin": match.margin,
                            **_observability_diagnostics(assessment),
                        },
                    )
                )
                continue
            if assessment.state == "not_observable":
                rows.append(
                    InstanceFrameResult(
                        track=track,
                        window_id=frame.window_id,
                        frame_index=frame.frame_index,
                        reference_id=reference_id,
                        observed_id=observed.observed_id,
                        state="not_observable",
                        reasons=(assessment.reason or "not_observable",),
                        confidence=assessment.confidence,
                        metrics={
                            "presence": 1.0,
                            "identity": identity,
                            "set_identity": (
                                1.0 if match.identity_state == "set_matched" else None
                            ),
                            "geometry": None,
                            "scale": None,
                            "topology": None,
                            "part_structure": _part_structure_score(
                                annotation, reference_id, observed
                            ),
                        },
                        diagnostics={
                            "identity_state": match.identity_state,
                            "association_score": match.score,
                            "association_margin": match.margin,
                            **_observability_diagnostics(assessment),
                        },
                    )
                )
                continue

            comparison = _compare_geometry(
                scene,
                geometry,
                frame,
                reference_id,
                observed,
                match,
                "static",
                config,
            )
            canonical = comparison.canonical
            rows.append(
                InstanceFrameResult(
                    track=track,
                    window_id=frame.window_id,
                    frame_index=frame.frame_index,
                    reference_id=reference_id,
                    observed_id=observed.observed_id,
                    state=comparison.gate.state,
                    reasons=comparison.gate.reasons,
                    confidence=comparison.gate.confidence,
                    metrics={
                        "presence": 1.0,
                        "identity": identity,
                        "set_identity": 1.0 if match.identity_state == "set_matched" else None,
                        "geometry": (
                            canonical.geometry_score
                            if comparison.gate.scored and canonical is not None
                            else None
                        ),
                        "scale": (
                            canonical.scale_score
                            if comparison.gate.scored and canonical is not None
                            else None
                        ),
                        "topology": (
                            canonical.topology_score
                            if comparison.gate.scored and canonical is not None
                            else None
                        ),
                        "part_structure": _part_structure_score(
                            annotation, reference_id, observed
                        ),
                    },
                    diagnostics={
                        "identity_state": match.identity_state,
                        "association_score": match.score,
                        "association_margin": match.margin,
                        **_observability_diagnostics(assessment),
                        "gate_components": comparison.gate.components,
                        "canonicalization_mode": canonical.mode if canonical else None,
                        **comparison.diagnostics,
                    },
                )
            )
    return rows, event_counts


def _compare_geometry(
    scene: ReferenceScene,
    geometry: GeometryBundle,
    frame: FrameAssociation,
    reference_id: str,
    observed: ObservedObject,
    match: AssociationMatch,
    kind: str,
    config: V2EvaluationConfig,
) -> _GeometryComparison:
    try:
        view_index = _geometry_view_index(geometry, frame.frame_index)
        query_indices = np.flatnonzero(
            np.asarray(geometry.query_reference_ids) == reference_id
        )
        if view_index is None:
            raise ValueError("sampled frame is absent from geometry bundle")
        if len(query_indices) == 0:
            raise ValueError("reference instance has no geometry queries")
        assert geometry.query_points is not None
        assert geometry.tracks is not None
        assert geometry.track_visibility is not None
        assert geometry.track_confidence is not None
        assert geometry.world_points is not None
        assert geometry.point_confidence is not None
        assert geometry.depth_confidence is not None
        assert geometry.image_transforms is not None
        assert geometry.camera_confidence is not None

        reference_pixels = geometry.query_points[query_indices]
        current_pixels = geometry.tracks[view_index, query_indices]
        visibility = geometry.track_visibility[view_index, query_indices]
        track_confidence = geometry.track_confidence[view_index, query_indices]
        inside = np.asarray(
            [
                _inside_observed(point, observed, geometry.image_transforms[view_index])
                for point in current_pixels
            ],
            dtype=bool,
        )
        reference_points = _sample_map(
            geometry.world_points[0], reference_pixels
        )
        current_points = _sample_map(
            geometry.world_points[view_index], current_pixels
        )
        reference_confidence = _sample_map(
            geometry.point_confidence[0], reference_pixels
        )
        current_confidence = _sample_map(
            geometry.point_confidence[view_index], current_pixels
        )
        depth_confidence = _sample_map(
            geometry.depth_confidence[view_index], current_pixels
        )
        finite = np.isfinite(reference_points).all(axis=1) & np.isfinite(current_points).all(axis=1)
        valid = (
            inside
            & finite
            & (visibility >= 0.5)
            & (track_confidence > 0.0)
            & (reference_confidence > 0.0)
            & (current_confidence > 0.0)
        )
        weights = np.minimum.reduce(
            [
                visibility[valid],
                track_confidence[valid],
                reference_confidence[valid],
                current_confidence[valid],
            ]
        )
        ref_valid = reference_points[valid]
        cur_valid = current_points[valid]
        canonical = canonicalize_instance(
            ref_valid,
            cur_valid,
            kind=kind,
            config=config,
            weights=weights,
        )
        reprojection = _reprojection_error_ratio(
            cur_valid,
            current_pixels[valid],
            geometry,
            view_index,
            observed,
        )
        transformed_pixels = np.asarray(
            [
                _model_to_source(point, geometry.image_transforms[view_index])
                for point in current_pixels[valid]
            ]
        )
        area = _observed_area(observed)
        spatial_coverage = _convex_hull_area(transformed_pixels) / max(area, 1.0)
        baseline = max(
            _camera_baseline_ratio(geometry, view_index, ref_valid),
            _track_displacement_ratio(reference_pixels[valid], current_pixels[valid], geometry.model_size),
        )
        evidence = GateEvidence(
            association_confidence=match.score,
            visible_fraction=float(valid.sum() / max(len(query_indices), 1)),
            mask_area_px=area,
            correspondence_count=int(valid.sum()),
            spatial_coverage=clamp01(spatial_coverage),
            track_confidence=float(np.mean(track_confidence[valid])) if valid.any() else 0.0,
            valid_depth_ratio=float(valid.sum() / max(len(query_indices), 1)),
            depth_confidence=float(np.mean(depth_confidence[valid])) if valid.any() else 0.0,
            pose_inlier_ratio=canonical.inlier_ratio if canonical.valid else 0.0,
            reprojection_error_ratio=reprojection,
            camera_confidence=float(geometry.camera_confidence[view_index]),
            baseline_ratio=baseline,
            cycle_consistency=None,
        )
        gate = evaluate_gate(evidence, config)
        return _GeometryComparison(
            gate=gate,
            canonical=canonical,
            reference_points=ref_valid,
            current_points=cur_valid,
            diagnostics={
                "correspondence_count": int(valid.sum()),
                "visible_fraction": evidence.visible_fraction,
                "spatial_coverage": evidence.spatial_coverage,
                "reprojection_error_ratio": reprojection,
                "baseline_ratio": baseline,
                "reference_centroid": ref_valid.mean(axis=0).tolist() if len(ref_valid) else None,
                "current_centroid": cur_valid.mean(axis=0).tolist() if len(cur_valid) else None,
                "scale_ratio": canonical.scale_ratio,
                "rigid_inlier_ratio": canonical.inlier_ratio,
            },
        )
    except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
        gate = evaluate_gate(GateEvidence(evaluator_error=str(exc)), config)
        return _GeometryComparison(
            gate=gate,
            canonical=None,
            reference_points=None,
            current_points=None,
            diagnostics={"geometry_error": str(exc)},
        )


def _aggregate_track(
    track: str,
    rows: list[InstanceFrameResult],
    association: AssociationBundle,
    event_counts: dict[str, int],
    config: V2EvaluationConfig,
) -> TrackReport:
    metric_names = (
        "presence",
        "identity",
        "set_identity",
        "geometry",
        "scale",
        "topology",
        "part_structure",
    )
    macro = {
        name: macro_instance_mean(
            (row.reference_id, row.metrics.get(name))
            for row in rows
        )
        for name in metric_names
    }
    coverage_rows, baseline_excluded = _partition_geometry_coverage_rows(rows)
    scored_geometry = [row for row in coverage_rows if row.state == "scored"]
    geometry_coverage = len(scored_geometry) / max(len(coverage_rows), 1)
    valid_frames = len({row.window_id for row in scored_geometry})
    total_reference_rows = len(rows)
    observable_rows = sum(
        1 for row in rows if row.metrics.get("presence") is not None
    )
    observability_coverage = observable_rows / max(total_reference_rows, 1)
    layout_score = _relative_layout_score(rows)
    identity_states = {"matched", "set_matched", "ambiguous"}
    ambiguity_total = sum(
        1
        for row in rows
        if row.diagnostics.get("identity_state") in identity_states
    )
    ambiguity_count = sum(
        1 for row in rows if row.diagnostics.get("identity_state") == "ambiguous"
    )
    metrics = {
        "presence_score": macro["presence"],
        "identity_score": macro["identity"],
        "set_identity_score": macro["set_identity"],
        "identity_ambiguity_rate": ambiguity_count / max(ambiguity_total, 1),
        "visible_surface_geometry_score": macro["geometry"],
        "relative_layout_score": layout_score,
        "scale_consistency_score": macro["scale"],
        "local_topology_score": macro["topology"],
        "part_structure_score": macro["part_structure"],
    }
    eligible = (
        geometry_coverage >= config.static_coverage_eligibility
        and observability_coverage >= config.static_observability_eligibility
        and valid_frames >= config.min_valid_geometry_frames
    )
    headline_values = (
        metrics["presence_score"],
        metrics["identity_score"],
        metrics["visible_surface_geometry_score"],
        metrics["scale_consistency_score"],
        metrics["local_topology_score"],
        metrics["relative_layout_score"],
    )
    per_instance = {
        reference_id: {
            name: mean(
                row.metrics.get(name)
                for row in rows
                if row.reference_id == reference_id
            )
            for name in metric_names
        }
        for reference_id in sorted({row.reference_id for row in rows})
    }
    state_counts = _count_values(row.state for row in rows)
    reasons = _count_values(reason for row in rows for reason in row.reasons)
    error_counts = {
        **event_counts,
        "identity_ambiguous": ambiguity_count,
        "id_switch": _id_switch_count(association),
    }
    return TrackReport(
        track=track,
        headline_score=geometric_mean(headline_values),
        eligible=True,
        geometry_coverage=geometry_coverage,
        geometry_candidate_rows=len(coverage_rows),
        baseline_excluded_rows=len(baseline_excluded),
        valid_geometry_frames=valid_frames,
        observability_coverage=observability_coverage,
        observable_rows=observable_rows,
        total_reference_rows=total_reference_rows,
        metrics=metrics,
        error_counts=error_counts,
        state_counts=state_counts,
        abstention_reasons=reasons,
        per_instance=per_instance,
        confidence={
            "geometry_coverage": geometry_coverage,
            "observability_coverage": observability_coverage,
            "valid_geometry_frame_ratio": valid_frames / max(len({row.window_id for row in rows}), 1),
            "overall": geometry_coverage * observability_coverage,
            "strict_eligibility": float(eligible),
        },
    )


_BASELINE_EXCLUSION_REASONS = frozenset(
    {
        "insufficient_baseline",
        "high_reprojection_error",
        "low_combined_confidence",
    }
)


def _partition_geometry_coverage_rows(
    rows: list[InstanceFrameResult],
) -> tuple[list[InstanceFrameResult], list[InstanceFrameResult]]:
    candidates: list[InstanceFrameResult] = []
    baseline_excluded: list[InstanceFrameResult] = []
    for row in rows:
        reasons = set(row.reasons)
        if row.state == "evaluator_failure":
            continue
        if reasons & {
            "outside_reference_view",
            "projected_instance_too_small",
            "occluded_instance",
        }:
            continue
        if (
            row.state == "not_observable"
            and "insufficient_baseline" in reasons
            and reasons <= _BASELINE_EXCLUSION_REASONS
        ):
            baseline_excluded.append(row)
            continue
        candidates.append(row)
    return candidates, baseline_excluded


def _identity_score(match: AssociationMatch) -> float:
    if match.identity_state == "set_matched":
        return 1.0
    if match.identity_state == "ambiguous":
        return 0.5 * match.score
    return match.score


def _classify_frame_events(
    track: str,
    scene: ReferenceScene,
    window,
    frame: FrameAssociation,
    geometry: GeometryBundle,
    config: V2EvaluationConfig,
    *,
    missing_reference_ids: list[str] | None = None,
) -> tuple[dict[str, int], dict[str, str]]:
    missing_ids = (
        frame.missing_reference_ids
        if missing_reference_ids is None
        else missing_reference_ids
    )
    counts = {
        "missing": len(missing_ids),
        "duplicate": 0,
        "hallucination": 0,
        "replacement": 0,
        "merge": 0,
        "unmatched_observed": len(frame.unmatched_observed_ids),
    }
    reasons: dict[str, str] = {}
    observed_by_id = {item.observed_id: item for item in window.objects}
    matched_observed = {item.observed_id for item in frame.matches}
    consumed_unmatched: set[str] = set()
    view_index = _geometry_view_index(geometry, frame.frame_index)
    can_use_tracks = (
        geometry.status == "available"
        and view_index is not None
        and geometry.tracks is not None
        and geometry.track_visibility is not None
        and geometry.image_transforms is not None
    )
    query_ids = np.asarray(geometry.query_reference_ids)
    for reference_id in missing_ids:
        reason = "missing_instance"
        if can_use_tracks:
            indices = np.flatnonzero(query_ids == reference_id)
            if len(indices):
                assert view_index is not None
                assert geometry.tracks is not None
                assert geometry.track_visibility is not None
                assert geometry.image_transforms is not None
                pixels = geometry.tracks[view_index, indices]
                visible = geometry.track_visibility[view_index, indices] >= 0.5
                best_observed_id = None
                best_coverage = 0.0
                for observed in window.objects:
                    inside = np.asarray(
                        [
                            _inside_observed(
                                point,
                                observed,
                                geometry.image_transforms[view_index],
                            )
                            for point in pixels
                        ],
                        dtype=bool,
                    )
                    coverage = float((inside & visible).sum() / max(int(visible.sum()), 1))
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_observed_id = observed.observed_id
                if best_observed_id is not None and best_coverage >= config.min_visible_fraction:
                    if best_observed_id in matched_observed:
                        counts["merge"] += 1
                        reason = "merged_instance"
                    else:
                        counts["replacement"] += 1
                        consumed_unmatched.add(best_observed_id)
                        reason = "replacement_instance"
        reasons[reference_id] = reason

    for observed_id in frame.unmatched_observed_ids:
        if observed_id in consumed_unmatched:
            continue
        observed = observed_by_id[observed_id]
        best_similarity = max(
            (
                cosine_similarity(observed.embedding, reference.embedding) or 0.0
                for reference in scene.objects.values()
            ),
            default=0.0,
        )
        if best_similarity >= config.duplicate_similarity_threshold:
            counts["duplicate"] += 1
        else:
            counts["hallucination"] += 1
    return counts, reasons


def _part_structure_score(
    annotation: ReferenceAnnotation,
    reference_id: str,
    observed: ObservedObject,
) -> float | None:
    reference = annotation.instances[reference_id]
    current_raw = observed.metadata.get("keypoints", {})
    current = {
        str(name): tuple(float(value) for value in coords)
        for name, coords in current_raw.items()
        if isinstance(coords, (list, tuple)) and len(coords) >= 2
    }
    bones_raw = reference.metadata.get("bones", [])
    bones = tuple(
        (str(item[0]), str(item[1]))
        for item in bones_raw
        if isinstance(item, (list, tuple)) and len(item) == 2
    )
    if not bones:
        return None
    return score_articulated_keypoints(reference.keypoints, current, bones=bones)


def _missing_metrics(
    annotation: ReferenceAnnotation,
    reference_id: str,
) -> dict[str, float | None]:
    reference = annotation.instances[reference_id]
    return {
        "presence": 0.0,
        "identity": 0.0,
        "set_identity": 0.0 if reference.ambiguity_group is not None else None,
        "geometry": 0.0,
        "scale": 0.0,
        "topology": 0.0,
        "part_structure": 0.0 if _has_reference_bones(reference) else None,
    }


def _has_reference_bones(reference: ReferenceInstance) -> bool:
    return any(
        isinstance(item, (list, tuple))
        and len(item) == 2
        and str(item[0]) in reference.keypoints
        and str(item[1]) in reference.keypoints
        for item in reference.metadata.get("bones", [])
    )


def _empty_row(
    track: str,
    frame: FrameAssociation,
    reference_id: str,
    *,
    state: str,
    reason: str,
    assessment: ReferenceObservability | None = None,
) -> InstanceFrameResult:
    return InstanceFrameResult(
        track=track,
        window_id=frame.window_id,
        frame_index=frame.frame_index,
        reference_id=reference_id,
        observed_id=None,
        state=state,
        reasons=(reason,),
        confidence=assessment.confidence if assessment is not None else 0.0,
        metrics={
            "presence": None,
            "identity": None,
            "set_identity": None,
            "geometry": None,
            "scale": None,
            "topology": None,
            "part_structure": None,
        },
        diagnostics=(
            _observability_diagnostics(assessment)
            if assessment is not None
            else {}
        ),
    )


def _observability_diagnostics(
    assessment: ReferenceObservability,
) -> dict[str, Any]:
    return {
        "observability_state": assessment.state,
        "observability_reason": assessment.reason,
        "observability_confidence": assessment.confidence,
        **assessment.diagnostics,
    }


def _geometry_view_index(geometry: GeometryBundle, frame_index: int | None) -> int | None:
    if frame_index is None or geometry.frame_indices is None:
        return None
    matches = np.flatnonzero(geometry.frame_indices == frame_index)
    return int(matches[0]) if len(matches) else None


def _sample_map(array: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    height, width = values.shape[:2]
    xy = np.rint(np.asarray(points)).astype(np.int64)
    xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
    return np.asarray(values[xy[:, 1], xy[:, 0]])


def _inside_observed(
    model_point: np.ndarray,
    observed: ObservedObject,
    transform: np.ndarray,
) -> bool:
    x, y = _model_to_source(model_point, transform)
    x0, y0, x1, y1 = observed.bbox
    epsilon = 1e-4
    if not (
        x0 - epsilon <= x <= x1 + epsilon
        and y0 - epsilon <= y <= y1 + epsilon
    ):
        return False
    if observed.mask is None:
        return True
    mask = mask_to_bool_array(observed.mask)
    ix = min(max(int(round(x)), 0), mask.shape[1] - 1)
    iy = min(max(int(round(y)), 0), mask.shape[0] - 1)
    return bool(mask[iy, ix])


def _model_to_source(point: np.ndarray, transform: np.ndarray) -> tuple[float, float]:
    scale_x, scale_y, offset_x, offset_y = [float(value) for value in transform]
    return (
        (float(point[0]) - offset_x) / max(scale_x, 1e-12),
        (float(point[1]) - offset_y) / max(scale_y, 1e-12),
    )


def _observed_area(observed: ObservedObject) -> float:
    if observed.mask is not None:
        try:
            return float(mask_area(observed.mask))
        except (TypeError, ValueError):
            pass
    return observed.width * observed.height


def _project(
    points: np.ndarray,
    geometry: GeometryBundle,
    view_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    assert geometry.extrinsics is not None
    assert geometry.intrinsics is not None
    rotation = geometry.extrinsics[view_index, :3, :3]
    translation = geometry.extrinsics[view_index, :3, 3]
    camera = points @ rotation.T + translation
    pixels_h = camera @ geometry.intrinsics[view_index].T
    positive = pixels_h[:, 2] > 1e-9
    pixels = pixels_h[:, :2] / np.maximum(pixels_h[:, 2:3], 1e-9)
    return pixels, positive


def _reprojection_error_ratio(
    points: np.ndarray,
    pixels: np.ndarray,
    geometry: GeometryBundle,
    view_index: int,
    observed: ObservedObject,
) -> float:
    if len(points) == 0:
        return math.inf
    projected, positive = _project(points, geometry, view_index)
    if not positive.any():
        return math.inf
    errors = np.linalg.norm(projected[positive] - pixels[positive], axis=1)
    assert geometry.image_transforms is not None
    scale_x, scale_y = geometry.image_transforms[view_index, :2]
    diagonal = math.hypot(observed.width * scale_x, observed.height * scale_y)
    return float(np.median(errors)) / max(diagonal, 1e-9)


def _camera_baseline_ratio(
    geometry: GeometryBundle,
    view_index: int,
    reference_points: np.ndarray,
) -> float:
    if len(reference_points) == 0 or geometry.extrinsics is None:
        return 0.0
    centers = []
    for index in (0, view_index):
        rotation = geometry.extrinsics[index, :3, :3]
        translation = geometry.extrinsics[index, :3, 3]
        centers.append(-rotation.T @ translation)
    baseline = float(np.linalg.norm(centers[1] - centers[0]))
    scene_scale = float(np.median(np.linalg.norm(reference_points - centers[0], axis=1)))
    return baseline / max(scene_scale, 1e-9)


def _track_displacement_ratio(
    reference_pixels: np.ndarray,
    current_pixels: np.ndarray,
    model_size: tuple[int, int] | None,
) -> float:
    if len(reference_pixels) == 0 or model_size is None:
        return 0.0
    displacement = np.linalg.norm(current_pixels - reference_pixels, axis=1)
    return float(np.median(displacement)) / max(math.hypot(*model_size), 1e-9)


def _convex_hull_area(points: np.ndarray) -> float:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) < 3:
        return 0.0

    def cross(origin, first, second) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return 0.5 * abs(
        sum(
            hull[index][0] * hull[(index + 1) % len(hull)][1]
            - hull[(index + 1) % len(hull)][0] * hull[index][1]
            for index in range(len(hull))
        )
    )


def _relative_layout_score(rows: list[InstanceFrameResult]) -> float | None:
    by_window: dict[str, list[InstanceFrameResult]] = {}
    for row in rows:
        if row.state == "scored":
            by_window.setdefault(row.window_id, []).append(row)
    scores: list[float] = []
    for window_rows in by_window.values():
        for first_index, first in enumerate(window_rows):
            ref_a = first.diagnostics.get("reference_centroid")
            cur_a = first.diagnostics.get("current_centroid")
            if ref_a is None or cur_a is None:
                continue
            for second in window_rows[first_index + 1 :]:
                ref_b = second.diagnostics.get("reference_centroid")
                cur_b = second.diagnostics.get("current_centroid")
                if ref_b is None or cur_b is None:
                    continue
                ref_distance = float(np.linalg.norm(np.asarray(ref_a) - np.asarray(ref_b)))
                cur_distance = float(np.linalg.norm(np.asarray(cur_a) - np.asarray(cur_b)))
                if ref_distance > 1e-9 and cur_distance > 1e-9:
                    scores.append(math.exp(-abs(math.log(cur_distance / ref_distance))))
    return mean(scores)


def _id_switch_count(association: AssociationBundle) -> int:
    previous: dict[str, str] = {}
    count = 0
    for frame in association.frames:
        for match in frame.matches:
            if match.track_id is None:
                continue
            old = previous.get(match.reference_id)
            if old is not None and old != match.track_id:
                count += 1
            previous[match.reference_id] = match.track_id
    return count


def _count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts
