"""Conservative per-instance observability for Static-3D evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from multimem_bench.schema import ReferenceScene, VideoObservation
from multimem_bench.v2.association import AssociationBundle, AssociationMatch
from multimem_bench.v2.config import V2EvaluationConfig
from multimem_bench.v2.geometry.artifact import GeometryBundle
from multimem_bench.v2.math_utils import clamp01


@dataclass(frozen=True)
class ReferenceObservability:
    state: str
    reason: str | None
    confidence: float
    camera_reliable: bool
    association_candidate: bool = True
    diagnostics: dict[str, Any] = field(default_factory=dict)


ObservabilityMap = dict[str, dict[str, ReferenceObservability]]


def assess_static_observability(
    scene: ReferenceScene,
    observation: VideoObservation,
    geometry: GeometryBundle,
    config: V2EvaluationConfig,
) -> ObservabilityMap:
    """Assess whether every reference instance can be judged in each window."""

    if geometry.status != "available":
        reason = geometry.error or "geometry_unavailable"
        return {
            window.window_id: {
                reference_id: _failure(reason, camera_reliable=False)
                for reference_id in scene.objects
            }
            for window in observation.windows
        }

    query_ids = np.asarray(geometry.query_reference_ids)
    results: ObservabilityMap = {}
    for window in observation.windows:
        view_index = _geometry_view_index(geometry, window.frame_index)
        results[window.window_id] = {
            reference_id: _assess_reference(
                geometry,
                view_index,
                np.flatnonzero(query_ids == reference_id),
                config,
            )
            for reference_id in scene.objects
        }
    return results


def association_candidates(
    observability: ObservabilityMap,
) -> dict[str, set[str]]:
    """Return references that may legitimately participate in association."""

    return {
        window_id: {
            reference_id
            for reference_id, assessment in references.items()
            if assessment.association_candidate
        }
        for window_id, references in observability.items()
    }


def refine_temporal_occlusions(
    observability: ObservabilityMap,
    association: AssociationBundle,
    config: V2EvaluationConfig,
) -> ObservabilityMap:
    """Confirm partial occlusions bracketed by compatible identity matches."""

    refined = {window_id: dict(values) for window_id, values in observability.items()}
    references = sorted(
        {
            reference_id
            for values in observability.values()
            for reference_id in values
        }
    )
    for reference_id in references:
        matched = [
            (index, match)
            for index, frame in enumerate(association.frames)
            for match in frame.matches
            if match.reference_id == reference_id
        ]
        for (left_index, left), (right_index, right) in zip(matched, matched[1:]):
            gap = right_index - left_index - 1
            if gap <= 0 or gap > config.max_temporal_occlusion_gap:
                continue
            if not _compatible_reappearance(left, right, config):
                continue
            for frame in association.frames[left_index + 1 : right_index]:
                if reference_id not in frame.missing_reference_ids:
                    continue
                assessment = refined[frame.window_id][reference_id]
                occluded_fraction = float(
                    assessment.diagnostics.get("occluded_fraction", 0.0)
                )
                if assessment.state == "evaluator_failure":
                    continue
                if occluded_fraction < config.min_temporal_occlusion_fraction:
                    continue
                diagnostics = {
                    **assessment.diagnostics,
                    "temporal_reappearance_confirmed": True,
                    "temporal_gap_windows": gap,
                }
                refined[frame.window_id][reference_id] = replace(
                    assessment,
                    state="not_observable",
                    reason="occluded_instance",
                    diagnostics=diagnostics,
                )
    return refined


def _assess_reference(
    geometry: GeometryBundle,
    view_index: int | None,
    query_indices: np.ndarray,
    config: V2EvaluationConfig,
) -> ReferenceObservability:
    if view_index is None:
        return _failure("geometry_frame_unavailable", camera_reliable=False)
    camera_status = _camera_status(geometry, view_index)
    camera_confidence = _camera_confidence(geometry, view_index)
    camera_diagnostics = {
        "camera_refinement_status": camera_status,
        "camera_confidence": camera_confidence,
    }
    if camera_status is not None and camera_status not in {"reference", "refined"}:
        return _failure(
            "unreliable_camera_pose",
            camera_reliable=False,
            diagnostics=camera_diagnostics,
        )
    if camera_confidence < config.min_camera_confidence:
        return _failure(
            "unreliable_camera_pose",
            camera_reliable=False,
            diagnostics=camera_diagnostics,
        )
    if len(query_indices) == 0:
        return _failure(
            "missing_reference_geometry",
            camera_reliable=True,
            diagnostics=camera_diagnostics,
        )
    required = (
        geometry.query_points,
        geometry.world_points,
        geometry.extrinsics,
        geometry.intrinsics,
        geometry.depth,
        geometry.depth_confidence,
        geometry.image_transforms,
        geometry.model_size,
    )
    if any(value is None for value in required):
        return _failure(
            "incomplete_geometry_artifact",
            camera_reliable=True,
            diagnostics=camera_diagnostics,
        )

    assert geometry.query_points is not None
    assert geometry.world_points is not None
    assert geometry.extrinsics is not None
    assert geometry.intrinsics is not None
    assert geometry.depth is not None
    assert geometry.depth_confidence is not None
    assert geometry.image_transforms is not None
    assert geometry.model_size is not None
    reference_pixels = geometry.query_points[query_indices]
    reference_points = _sample_map(geometry.world_points[0], reference_pixels)
    finite_reference = np.isfinite(reference_points).all(axis=1)
    if not finite_reference.any():
        return _failure(
            "invalid_reference_geometry",
            camera_reliable=True,
            diagnostics=camera_diagnostics,
        )

    projected, positive, projected_depth = _project(
        reference_points,
        geometry.extrinsics[view_index],
        geometry.intrinsics[view_index],
    )
    width, height = geometry.model_size
    inside = (
        finite_reference
        & positive
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    in_frame_fraction = float(inside.sum() / max(len(query_indices), 1))
    diagnostics = {
        **camera_diagnostics,
        "in_frame_fraction": in_frame_fraction,
    }
    if in_frame_fraction < config.min_visible_fraction:
        return ReferenceObservability(
            state="not_observable",
            reason="outside_reference_view",
            confidence=camera_confidence,
            camera_reliable=True,
            association_candidate=False,
            diagnostics=diagnostics,
        )

    source_points = _model_to_source(
        projected[inside], geometry.image_transforms[view_index]
    )
    projected_area = _convex_hull_area(source_points)
    diagnostics["projected_area_px"] = projected_area
    if projected_area < config.min_projected_area_px:
        return ReferenceObservability(
            state="not_observable",
            reason="projected_instance_too_small",
            confidence=camera_confidence,
            camera_reliable=True,
            diagnostics=diagnostics,
        )

    frame_depth = _sample_map(geometry.depth[view_index], projected)
    depth_confidence = _sample_map(geometry.depth_confidence[view_index], projected)
    valid_depth = (
        inside
        & np.isfinite(frame_depth)
        & (frame_depth > 0.0)
        & np.isfinite(projected_depth)
        & (projected_depth > 0.0)
        & np.isfinite(depth_confidence)
        & (depth_confidence >= config.min_occlusion_depth_confidence)
    )
    depth_coverage = float(valid_depth.sum() / max(int(inside.sum()), 1))
    diagnostics["occlusion_depth_coverage"] = depth_coverage
    if depth_coverage < config.min_occlusion_depth_coverage:
        return _failure(
            "insufficient_occlusion_depth",
            camera_reliable=True,
            confidence=camera_confidence * depth_coverage,
            diagnostics=diagnostics,
        )

    tolerance = (
        np.abs(projected_depth) * config.occlusion_depth_tolerance_ratio
    )
    occluded = valid_depth & (frame_depth < projected_depth - tolerance)
    unoccluded = valid_depth & ~occluded
    occluded_fraction = float(occluded.sum() / max(int(valid_depth.sum()), 1))
    unoccluded_fraction = float(unoccluded.sum() / max(len(query_indices), 1))
    diagnostics.update(
        {
            "occluded_fraction": occluded_fraction,
            "unoccluded_fraction": unoccluded_fraction,
        }
    )
    confidence = clamp01(camera_confidence * depth_coverage)
    if (
        unoccluded_fraction < config.min_visible_fraction
        and occluded_fraction >= config.min_direct_occlusion_fraction
    ):
        return ReferenceObservability(
            state="not_observable",
            reason="occluded_instance",
            confidence=confidence,
            camera_reliable=True,
            diagnostics=diagnostics,
        )
    if unoccluded_fraction < config.min_visible_fraction:
        return _failure(
            "insufficient_unoccluded_evidence",
            camera_reliable=True,
            confidence=confidence,
            diagnostics=diagnostics,
        )
    return ReferenceObservability(
        state="observable",
        reason=None,
        confidence=confidence,
        camera_reliable=True,
        diagnostics=diagnostics,
    )


def _failure(
    reason: str,
    *,
    camera_reliable: bool,
    confidence: float = 0.0,
    diagnostics: dict[str, Any] | None = None,
) -> ReferenceObservability:
    return ReferenceObservability(
        state="evaluator_failure",
        reason=reason,
        confidence=clamp01(confidence),
        camera_reliable=camera_reliable,
        diagnostics=dict(diagnostics or {}),
    )


def _camera_status(geometry: GeometryBundle, view_index: int) -> str | None:
    reports = geometry.metadata.get("camera_refinement", [])
    if not isinstance(reports, list):
        return None
    for report in reports:
        if not isinstance(report, dict):
            continue
        if report.get("view_index") == view_index:
            status = report.get("status")
            return str(status) if status is not None else None
    return None


def _camera_confidence(geometry: GeometryBundle, view_index: int) -> float:
    if geometry.camera_confidence is None:
        return 0.0
    try:
        return clamp01(float(geometry.camera_confidence[view_index]))
    except (IndexError, TypeError, ValueError):
        return 0.0


def _geometry_view_index(
    geometry: GeometryBundle, frame_index: int | None
) -> int | None:
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
    sampled = np.asarray(values[xy[:, 1], xy[:, 0]])
    if values.ndim == 3 and values.shape[-1] == 1:
        return sampled[..., 0]
    return sampled


def _project(
    points: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    pixels_h = camera @ intrinsic.T
    positive = pixels_h[:, 2] > 1e-9
    pixels = pixels_h[:, :2] / np.maximum(pixels_h[:, 2:3], 1e-9)
    return pixels, positive, camera[:, 2]


def _model_to_source(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    scale_x, scale_y, offset_x, offset_y = [float(value) for value in transform]
    return np.stack(
        [
            (points[:, 0] - offset_x) / max(scale_x, 1e-12),
            (points[:, 1] - offset_y) / max(scale_y, 1e-12),
        ],
        axis=1,
    )


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


def _compatible_reappearance(
    left: AssociationMatch,
    right: AssociationMatch,
    config: V2EvaluationConfig,
) -> bool:
    if left.track_id is not None and right.track_id is not None:
        return left.track_id == right.track_id
    return min(left.score, right.score) >= config.min_identity_similarity
