"""Per-object visibility: direct observations or audited input-only projections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from multimem_bench.schema import ObjectSignature, ObservedObject
from .config import V3EvaluationConfig


@dataclass(frozen=True)
class V3VisibilityDecision:
    state: str
    presence_eligible: bool
    geometry_eligible: bool
    confidence: float
    reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def classify_visibility(*, reference: ObjectSignature, frame_geometry: Any | None,
                        observed: ObservedObject | None, config: V3EvaluationConfig,
                        geometry_required: bool = True,
                        frame_size: tuple[int, int] | None = None) -> V3VisibilityDecision:
    evidence = (getattr(frame_geometry, "instance_visibility", {}) or {}).get(reference.object_id)
    projected = None
    # A static reference projection does not predict independently moving objects.
    if evidence and geometry_required:
        projected = _projected_visibility(evidence, config, frame_size)
        if projected.state in {"outside_view", "occluded", "too_small"}:
            return projected
    if observed is not None:
        bbox = np.asarray(observed.bbox, dtype=float)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            return _decision("evaluator_failure", False, False, 0., "invalid_observed_bbox")
        x0, y0, x1, y1 = bbox
        if frame_size is not None:
            width, height = frame_size
            x0, x1 = np.clip([x0, x1], 0, width)
            y0, y1 = np.clip([y0, y1], 0, height)
        area = float(max(0., x1-x0) * max(0., y1-y0))
        if area < config.min_observable_area_px:
            return _decision("too_small", False, False, 1., "observed_instance_too_small", {"observed_area_px": area})
        geometry_ok = geometry_required and getattr(frame_geometry, "status", None) == "scored"
        if projected is not None and not projected.geometry_eligible:
            geometry_ok = False
        return _decision("expected_visible", True, geometry_ok, 1., "direct_observation",
                         {"observed_area_px": area, "visibility_evidence": "direct_observation",
                          "projection_state": projected.state if projected else None})
    if projected is not None:
        return projected
    return _decision("evaluator_failure", False, False, 0., "missing_instance_visibility_evidence")


def _projected_visibility(evidence: dict[str, Any], cfg: V3EvaluationConfig,
                          frame_size: tuple[int, int] | None) -> V3VisibilityDecision:
    def fail(reason: str, state: str = "evaluator_failure") -> V3VisibilityDecision:
        return _decision(state, False, False, 0., reason)

    if evidence.get("reference_source") != "input_only":
        return fail("reference_projection_not_input_only")
    confidence = _number(evidence.get("camera_confidence"))
    if confidence is None or confidence < cfg.min_visibility_camera_confidence:
        return fail("unreliable_camera_pose", "unreliable_camera_pose")
    if frame_size is None or len(frame_size) != 2 or min(frame_size) <= 0:
        return fail("missing_frame_bounds")
    try:
        if "projected_pixels" in evidence:
            pixels = np.asarray(evidence["projected_pixels"], dtype=float)
            depth = np.asarray(evidence["projected_depth"], dtype=float)
        else:
            points = np.asarray(evidence["reference_points_3d"], dtype=float)
            transform = np.asarray(evidence["world_to_camera"], dtype=float)
            intrinsics = np.asarray(evidence["intrinsics"], dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4,4) or intrinsics.shape != (3,3):
                return fail("invalid_projection_shape")
            camera = points @ transform[:3,:3].T + transform[:3,3]
            depth = camera[:,2]
            homogeneous = camera @ intrinsics.T
            pixels = homogeneous[:,:2] / np.where(depth != 0, depth, np.nan)[:,None]
        if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) < 4 or depth.shape != (len(pixels),):
            return fail("insufficient_projection_queries")
        if not np.isfinite(pixels).all() or not np.isfinite(depth).all():
            return fail("invalid_projection_values")
        width, height = frame_size
        inside = (depth > 0) & (pixels[:,0] >= 0) & (pixels[:,0] < width) & (pixels[:,1] >= 0) & (pixels[:,1] < height)
        fraction = float(inside.mean())
        diagnostic = {"visibility_evidence": "input_only_projection", "projected_inside_fraction": fraction}
        if fraction < cfg.min_visible_fraction:
            return _decision("outside_view", False, False, confidence, "projected_outside_view", diagnostic)
        import cv2
        area = float(cv2.contourArea(cv2.convexHull(pixels[inside].astype(np.float32))))
        diagnostic["projected_area_px"] = area
        if area < cfg.min_observable_area_px:
            return _decision("too_small", False, False, confidence, "projected_instance_too_small", diagnostic)
        current = np.asarray(evidence["current_depth"], dtype=float)
        depth_conf = np.asarray(evidence["depth_confidence"], dtype=float)
        if current.shape != depth.shape or depth_conf.shape != depth.shape:
            return fail("invalid_depth_query_shape")
        valid = inside & np.isfinite(current) & (current > 0) & np.isfinite(depth_conf) & (depth_conf >= cfg.min_visibility_depth_confidence)
        if valid.sum() / inside.sum() < cfg.min_visibility_depth_coverage:
            return fail("insufficient_depth_evidence")
        occluded = valid & (current < depth * (1. - cfg.occlusion_depth_tolerance))
        occ_fraction = float(occluded.sum() / valid.sum())
        visible_fraction = float((valid & ~occluded).sum() / len(pixels))
        diagnostic.update(occluded_fraction=occ_fraction, visible_fraction=visible_fraction)
        if occ_fraction >= cfg.min_occluded_fraction and visible_fraction < cfg.min_visible_fraction:
            return _decision("occluded", False, False, confidence, "depth_occluded", diagnostic)
        if visible_fraction < cfg.min_visible_fraction:
            return fail("insufficient_unoccluded_queries")
        if evidence.get("baseline_sufficient") is False:
            return _decision("insufficient_baseline", True, False, confidence, "insufficient_baseline", diagnostic)
        return _decision("expected_visible", True, True, confidence, "projected_expected_visible", diagnostic)
    except (KeyError, TypeError, ValueError, ImportError):
        return fail("invalid_or_missing_projection_evidence")


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _decision(state: str, presence: bool, geometry: bool, confidence: float,
              reason: str, diagnostics: dict[str, Any] | None = None) -> V3VisibilityDecision:
    return V3VisibilityDecision(state, presence, geometry, min(1., max(0., confidence)), reason, diagnostics or {})
