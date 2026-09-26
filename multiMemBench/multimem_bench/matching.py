"""Anchor-local slot auditing with projected masks.

The evaluator is observation-first, but it no longer searches for the cheapest
post-hoc explanation for all visible objects. Anchor matching is still based on
appearance. Once an anchor is selected, reference objects define fixed spatial
slots by projecting their masks into the generated frame. Observed SAM3 masks
then either occupy those slots or become hallucinations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np
from PIL import Image

from multimem_bench.camera import (
    AnchorCameraEstimate,
    estimate_anchor_camera,
)
from multimem_bench.config import EvaluationConfig
from multimem_bench.schema import (
    ObjectSignature,
    ObservedObject,
    ObservedWindow,
    ReferenceScene,
)
from multimem_bench.similarity import appearance_similarity
from multimem_bench.vision.mask_utils import (
    mask_area,
    mask_centroid,
    mask_from_bbox,
    mask_to_bool_array,
)


@dataclass
class ObjectMatch:
    observed_id: str
    reference_id: str
    track_id: str | None
    role: str
    cost: float
    scores: dict[str, float | None] = field(default_factory=dict)
    components: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnmatchedObserved:
    observed_id: str
    track_id: str | None
    kind: str
    max_slot_iou: float | None = None
    max_expected_coverage: float | None = None
    max_observed_coverage: float | None = None


@dataclass
class SlotAudit:
    reference_id: str
    occupied_observed_ids: list[str] = field(default_factory=list)
    occupant_track_ids: list[str | None] = field(default_factory=list)
    primary_observed_id: str | None = None
    primary_track_id: str | None = None
    role: str = "slot"
    errors: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float | None] = field(default_factory=dict)
    components: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalExplanation:
    window_id: str
    valid: bool
    anchor_observed_id: str | None = None
    anchor_reference_id: str | None = None
    anchor_similarity: float | None = None
    expected_reference_ids: list[str] = field(default_factory=list)
    slot_audits: list[SlotAudit] = field(default_factory=list)
    matches: list[ObjectMatch] = field(default_factory=list)
    missing_reference_ids: list[str] = field(default_factory=list)
    unmatched_observed: list[UnmatchedObserved] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    camera_scores: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnchorCandidate:
    observed: ObservedObject
    reference: ObjectSignature
    similarity: float
    components: dict[str, float | None]


@dataclass(frozen=True)
class ProjectedGeometry:
    valid: bool
    expected_center: tuple[float, float] | None = None
    expected_bbox: tuple[float, float, float, float] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SlotProjection:
    reference_id: str
    reference: ObjectSignature
    projected_geometry: ProjectedGeometry
    mask: np.ndarray = field(repr=False, compare=False)
    mask_area: float
    mask_centroid: tuple[float, float] | None
    mask_bbox: tuple[int, int, int, int] | None
    projected_full_area: float
    visible_area: float
    visible_ratio: float | None
    mask_source: str


@dataclass(frozen=True)
class MaskOverlap:
    observed: ObservedObject
    mask_iou: float
    expected_coverage: float
    observed_coverage: float
    intersection_area: float
    center_error_ratio: float | None
    qualifies: bool
    forced: bool = False

    def to_components(self) -> dict[str, float | bool]:
        return {
            "mask_iou": self.mask_iou,
            "expected_coverage": self.expected_coverage,
            "observed_coverage": self.observed_coverage,
            "intersection_area": self.intersection_area,
            "center_error_ratio": self.center_error_ratio,
            "slot_gate_pass": self.qualifies,
            "forced_anchor_slot": self.forced,
        }


@dataclass(frozen=True)
class MaskStats:
    mask: np.ndarray = field(repr=False, compare=False)
    area: float
    centroid: tuple[float, float] | None
    bbox: tuple[int, int, int, int] | None
    source: str


def top_anchor_candidates(
    scene: ReferenceScene,
    window: ObservedWindow,
    config: EvaluationConfig,
) -> list[AnchorCandidate]:
    candidates: list[AnchorCandidate] = []
    for observed in window.objects:
        for reference in scene.objects.values():
            score, components = appearance_similarity(
                observed,
                reference,
                include_attribute_similarity=config.include_attribute_similarity,
                include_category_similarity=config.include_category_similarity,
            )
            if score >= config.min_anchor_similarity:
                candidates.append(AnchorCandidate(observed, reference, score, components))
    candidates.sort(key=lambda c: c.similarity, reverse=True)
    return candidates[: max(1, config.top_k_anchor_pairs)]


def _project_reference_geometry(
    observed_anchor: ObservedObject,
    reference_anchor: ObjectSignature,
    reference: ObjectSignature,
    camera: AnchorCameraEstimate,
    observed_anchor_stats: MaskStats | None = None,
    reference_anchor_stats: MaskStats | None = None,
    reference_stats: MaskStats | None = None,
) -> ProjectedGeometry:
    if not camera.valid or camera.actual_zoom is None:
        return ProjectedGeometry(valid=False, error="missing valid camera estimate")
    reference_anchor_center = _reference_center(reference_anchor, reference_anchor_stats)
    observed_anchor_center = _observed_center(observed_anchor, observed_anchor_stats)
    reference_center = _reference_center(reference, reference_stats)
    if reference_anchor_center is None:
        return ProjectedGeometry(valid=False, error="reference anchor missing center")
    if observed_anchor_center is None:
        return ProjectedGeometry(valid=False, error="observed anchor missing center")
    if reference_center is None:
        return ProjectedGeometry(valid=False, error="reference object missing center")

    anchor_x, anchor_y = observed_anchor_center
    ref_anchor_x, ref_anchor_y = reference_anchor_center
    zoom = camera.actual_zoom
    expected_center = (
        anchor_x + (reference_center[0] - ref_anchor_x) * zoom,
        anchor_y + (reference_center[1] - ref_anchor_y) * zoom,
    )

    expected_bbox = None
    if reference.bbox is not None:
        expected_bbox = (
            anchor_x + (reference.bbox[0] - ref_anchor_x) * zoom,
            anchor_y + (reference.bbox[1] - ref_anchor_y) * zoom,
            anchor_x + (reference.bbox[2] - ref_anchor_x) * zoom,
            anchor_y + (reference.bbox[3] - ref_anchor_y) * zoom,
        )
    return ProjectedGeometry(
        valid=True,
        expected_center=expected_center,
        expected_bbox=expected_bbox,
    )


def _mask_stats(mask: Any, source: str) -> MaskStats:
    return _mask_stats_from_array(mask_to_bool_array(mask), source)


def _mask_stats_from_array(mask: np.ndarray, source: str) -> MaskStats:
    arr = mask.astype(bool, copy=False)
    ys, xs = np.nonzero(arr)
    if len(xs) == 0:
        return MaskStats(mask=arr, area=0.0, centroid=None, bbox=None, source=source)
    return MaskStats(
        mask=arr,
        area=float(len(xs)),
        centroid=(float(xs.mean()), float(ys.mean())),
        bbox=(int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
        source=source,
    )


def _reference_center(
    reference: ObjectSignature,
    stats: MaskStats | None = None,
) -> tuple[float, float] | None:
    if stats is not None and stats.centroid is not None:
        return stats.centroid
    if reference.mask is not None:
        try:
            center = mask_centroid(reference.mask)
        except (TypeError, ValueError):
            center = None
        if center is not None:
            return center
    return reference.center


def _observed_center(
    observed: ObservedObject,
    stats: MaskStats | None = None,
) -> tuple[float, float] | None:
    if stats is not None and stats.centroid is not None:
        return stats.centroid
    if observed.mask is not None:
        try:
            center = mask_centroid(observed.mask)
        except (TypeError, ValueError):
            center = None
        if center is not None:
            return center
    return observed.center


def _reference_image_size(scene: ReferenceScene) -> tuple[int, int] | None:
    raw = scene.metadata.get("image_size") or scene.metadata.get("processed_image_size")
    if isinstance(raw, list) and len(raw) == 2:
        return int(raw[0]), int(raw[1])
    for obj in scene.objects.values():
        if obj.mask is not None:
            arr = mask_to_bool_array(obj.mask)
            return int(arr.shape[1]), int(arr.shape[0])
    return None


def _mask_from_reference(
    scene: ReferenceScene,
    reference: ObjectSignature,
    config: EvaluationConfig,
) -> tuple[np.ndarray | None, str]:
    if reference.mask is not None:
        return mask_to_bool_array(reference.mask), "artifact_mask"
    if config.require_masks:
        return None, "missing_required_mask"
    image_size = _reference_image_size(scene)
    if reference.bbox is None or image_size is None:
        return None, "missing_mask_and_bbox"
    return mask_from_bbox(reference.bbox, image_size), "bbox_fallback_mask"


def _mask_from_observed(
    window: ObservedWindow,
    observed: ObservedObject,
    config: EvaluationConfig,
) -> tuple[np.ndarray | None, str]:
    frame_size = window.frame_size
    if observed.mask is not None:
        return _normalize_mask_to_size(observed.mask, frame_size), "artifact_mask"
    if config.require_masks:
        return None, "missing_required_mask"
    return mask_from_bbox(observed.bbox, frame_size), "bbox_fallback_mask"


def _reference_mask_cache(
    scene: ReferenceScene,
    config: EvaluationConfig,
) -> dict[str, MaskStats | None]:
    out: dict[str, MaskStats | None] = {}
    for ref_id, reference in scene.objects.items():
        mask, source = _mask_from_reference(scene, reference, config)
        out[ref_id] = _mask_stats(mask, source) if mask is not None else None
    return out


def _observed_mask_cache(
    window: ObservedWindow,
    config: EvaluationConfig,
) -> dict[str, MaskStats | None]:
    out: dict[str, MaskStats | None] = {}
    for observed in window.objects:
        mask, source = _mask_from_observed(window, observed, config)
        out[observed.observed_id] = _mask_stats(mask, source) if mask is not None else None
    return out


def _normalize_mask_to_size(mask: Any, image_size: tuple[int, int]) -> np.ndarray:
    arr = mask_to_bool_array(mask)
    width, height = image_size
    if arr.shape == (height, width):
        return arr
    image = Image.fromarray(arr.astype(np.uint8) * 255, mode="L")
    image = image.resize((width, height), _nearest_resampling())
    return mask_to_bool_array(image)


def _nearest_resampling() -> Any:
    return getattr(getattr(Image, "Resampling", Image), "NEAREST")


def _affine_transform_kind() -> Any:
    return getattr(getattr(Image, "Transform", Image), "AFFINE")


def _project_reference_mask(
    reference_mask: np.ndarray,
    observed_anchor_center: tuple[float, float] | None,
    reference_anchor_center: tuple[float, float] | None,
    camera: AnchorCameraEstimate,
    frame_size: tuple[int, int],
) -> np.ndarray:
    if (
        reference_anchor_center is None
        or observed_anchor_center is None
        or camera.actual_zoom is None
    ):
        return np.zeros((frame_size[1], frame_size[0]), dtype=bool)
    zoom = max(camera.actual_zoom, 1e-6)
    anchor_x, anchor_y = observed_anchor_center
    ref_anchor_x, ref_anchor_y = reference_anchor_center
    inverse = (
        1.0 / zoom,
        0.0,
        ref_anchor_x - anchor_x / zoom,
        0.0,
        1.0 / zoom,
        ref_anchor_y - anchor_y / zoom,
    )
    image = Image.fromarray(reference_mask.astype(np.uint8) * 255, mode="L")
    projected = image.transform(
        frame_size,
        _affine_transform_kind(),
        inverse,
        resample=_nearest_resampling(),
        fillcolor=0,
    )
    return mask_to_bool_array(projected)


def _slot_projections_for_anchor(
    scene: ReferenceScene,
    window: ObservedWindow,
    anchor_observed: ObservedObject,
    anchor_reference: ObjectSignature,
    config: EvaluationConfig,
    requested_zoom: float | None,
    reference_masks: dict[str, MaskStats | None],
    observed_masks: dict[str, MaskStats | None],
) -> tuple[list[SlotProjection], AnchorCameraEstimate, dict[str, Any]]:
    camera = estimate_anchor_camera(
        scene,
        window,
        anchor_observed,
        anchor_reference,
        config,
        requested_zoom=requested_zoom,
    )
    diagnostics: dict[str, Any] = {
        "skipped_reference_masks": [],
        "reference_mask_sources": {},
    }
    if not camera.valid:
        return [], camera, diagnostics

    projections: list[SlotProjection] = []
    reference_anchor_stats = reference_masks.get(anchor_reference.object_id)
    observed_anchor_stats = observed_masks.get(anchor_observed.observed_id)
    reference_anchor_center = _reference_center(anchor_reference, reference_anchor_stats)
    observed_anchor_center = _observed_center(anchor_observed, observed_anchor_stats)
    for ref_id, reference in scene.objects.items():
        reference_stats = reference_masks.get(ref_id)
        mask_source = reference_stats.source if reference_stats is not None else "missing_required_mask"
        diagnostics["reference_mask_sources"][ref_id] = mask_source
        if reference_stats is None:
            diagnostics["skipped_reference_masks"].append(ref_id)
            continue
        projected = _project_reference_geometry(
            anchor_observed,
            anchor_reference,
            reference,
            camera,
            observed_anchor_stats,
            reference_anchor_stats,
            reference_stats,
        )
        projected_mask = _project_reference_mask(
            reference_stats.mask,
            observed_anchor_center,
            reference_anchor_center,
            camera,
            window.frame_size,
        )
        projected_stats = _mask_stats_from_array(projected_mask, "projected_reference_mask")
        visible_area = projected_stats.area
        full_area = float(reference_stats.area * (camera.actual_zoom or 1.0) ** 2)
        visible_ratio = visible_area / full_area if full_area > 0 else None
        visible = (
            visible_area >= config.min_projected_visible_area_px
            and (visible_ratio is None or visible_ratio >= config.min_projected_visible_ratio)
        )
        if ref_id != anchor_reference.object_id and not visible:
            continue
        projections.append(
            SlotProjection(
                reference_id=ref_id,
                reference=reference,
                projected_geometry=projected,
                mask=projected_mask,
                mask_area=projected_stats.area,
                mask_centroid=projected_stats.centroid,
                mask_bbox=projected_stats.bbox,
                projected_full_area=full_area,
                visible_area=visible_area,
                visible_ratio=visible_ratio,
                mask_source=mask_source,
            )
        )

    projections.sort(key=lambda slot: (_distance_from_anchor(anchor_reference, slot.reference), slot.reference_id))
    if config.max_expected_objects is not None:
        return projections[: max(1, config.max_expected_objects)], camera, diagnostics
    return projections, camera, diagnostics


def _distance_from_anchor(anchor: ObjectSignature, obj: ObjectSignature) -> float:
    if anchor.center is None or obj.center is None:
        return 1e9
    dx = obj.center[0] - anchor.center[0]
    dy = obj.center[1] - anchor.center[1]
    return math.sqrt(dx * dx + dy * dy)


def _mask_overlap(
    slot: SlotProjection,
    observed_stats: MaskStats,
) -> tuple[float, float, float, float, float | None]:
    slot_mask = slot.mask
    observed_mask = observed_stats.mask
    if slot_mask.shape != observed_mask.shape:
        return 0.0, 0.0, 0.0, 0.0, None
    overlap_bbox = _bbox_intersection(slot.mask_bbox, observed_stats.bbox)
    if overlap_bbox is None:
        return 0.0, 0.0, 0.0, 0.0, None
    x0, y0, x1, y1 = overlap_bbox
    inter = float(np.logical_and(
        slot_mask[y0:y1, x0:x1],
        observed_mask[y0:y1, x0:x1],
    ).sum())
    slot_area = slot.mask_area
    obs_area = observed_stats.area
    union = slot_area + obs_area - inter
    iou = 0.0 if union <= 0 else inter / union
    expected_coverage = 0.0 if slot_area <= 0 else inter / slot_area
    observed_coverage = 0.0 if obs_area <= 0 else inter / obs_area
    center_error_ratio = None
    if slot.mask_centroid is not None and observed_stats.centroid is not None:
        dist = math.dist(slot.mask_centroid, observed_stats.centroid)
        center_error_ratio = dist / max(math.sqrt(slot_area), 1.0)
    return iou, expected_coverage, observed_coverage, inter, center_error_ratio


def _bbox_intersection(
    first: tuple[int, int, int, int] | None,
    second: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if first is None or second is None:
        return None
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _overlap_qualifies(
    iou: float,
    expected_coverage: float,
    observed_coverage: float,
    config: EvaluationConfig,
) -> bool:
    if iou >= config.slot_mask_iou_threshold:
        return True
    return (
        expected_coverage >= config.slot_expected_coverage_threshold
        and observed_coverage >= config.slot_observed_coverage_threshold
    )


def _audit_slots(
    scene: ReferenceScene,
    window: ObservedWindow,
    anchor_candidate: AnchorCandidate,
    slots: list[SlotProjection],
    config: EvaluationConfig,
    observed_masks: dict[str, MaskStats | None],
) -> tuple[list[SlotAudit], list[ObjectMatch], list[str], list[UnmatchedObserved], dict[str, Any]]:
    del scene
    diagnostics: dict[str, Any] = {
        "observed_mask_sources": {
            observed_id: stats.source if stats is not None else "missing_required_mask"
            for observed_id, stats in observed_masks.items()
        },
        "missing_observed_masks": [
            observed_id
            for observed_id, stats in observed_masks.items()
            if stats is None
        ],
    }

    overlaps_by_slot: dict[str, list[MaskOverlap]] = {}
    max_overlap_by_observed: dict[str, MaskOverlap] = {}
    observed_to_slots: dict[str, list[str]] = {}
    anchor_obs_id = anchor_candidate.observed.observed_id
    anchor_ref_id = anchor_candidate.reference.object_id

    for slot in slots:
        overlaps: list[MaskOverlap] = []
        for observed in window.objects:
            if observed.observed_id == anchor_obs_id and slot.reference_id != anchor_ref_id:
                continue
            observed_stats = observed_masks[observed.observed_id]
            if observed_stats is None:
                continue
            iou, exp_cov, obs_cov, inter, center_error_ratio = _mask_overlap(
                slot,
                observed_stats,
            )
            forced = observed.observed_id == anchor_obs_id and slot.reference_id == anchor_ref_id
            qualifies = forced or _overlap_qualifies(iou, exp_cov, obs_cov, config)
            overlap = MaskOverlap(
                observed=observed,
                mask_iou=iou,
                expected_coverage=exp_cov,
                observed_coverage=obs_cov,
                intersection_area=inter,
                center_error_ratio=center_error_ratio,
                qualifies=qualifies,
                forced=forced,
            )
            current = max_overlap_by_observed.get(observed.observed_id)
            if current is None or iou > current.mask_iou:
                max_overlap_by_observed[observed.observed_id] = overlap
            if not qualifies:
                continue
            overlaps.append(overlap)
            observed_to_slots.setdefault(observed.observed_id, []).append(slot.reference_id)
        overlaps.sort(
            key=lambda item: (
                item.mask_iou,
                item.expected_coverage,
                item.observed_coverage,
                item.observed.confidence,
            ),
            reverse=True,
        )
        overlaps_by_slot[slot.reference_id] = overlaps

    slot_audits: list[SlotAudit] = []
    matches: list[ObjectMatch] = []
    missing_ids: list[str] = []
    for slot in slots:
        overlaps = overlaps_by_slot.get(slot.reference_id, [])
        primary = overlaps[0] if overlaps else None
        occupied_ids = [item.observed.observed_id for item in overlaps]
        track_ids = [item.observed.track_id for item in overlaps]
        appearance_score = None
        appearance_components: dict[str, float | None] | None = None
        category_error = False
        if primary is not None:
            appearance_score, appearance_components = appearance_similarity(
                primary.observed,
                slot.reference,
                include_attribute_similarity=config.include_attribute_similarity,
                include_category_similarity=config.include_category_similarity,
            )
            category_error = (
                bool(primary.observed.category)
                and bool(slot.reference.category)
                and primary.observed.category.lower() != slot.reference.category.lower()
            )

        missing = primary is None
        duplicate = len(overlaps) > 1
        replacement = (
            primary is not None
            and appearance_score is not None
            and appearance_score < config.appearance_similarity_threshold
        )
        position_error = (
            primary is not None
            and primary.center_error_ratio is not None
            and primary.center_error_ratio > config.slot_center_error_ratio_threshold
        )
        shape_error = (
            primary is not None
            and (
                primary.mask_iou < config.slot_shape_iou_threshold
                or primary.expected_coverage < config.slot_expected_coverage_threshold
                or primary.observed_coverage < config.slot_observed_coverage_threshold
            )
        )
        merge_error = (
            primary is not None
            and len(observed_to_slots.get(primary.observed.observed_id, [])) > 1
        )
        if missing:
            missing_ids.append(slot.reference_id)

        components: dict[str, Any] = {
            "projected_geometry": slot.projected_geometry.to_dict(),
            "projected_visible_area": slot.visible_area,
            "projected_full_area": slot.projected_full_area,
            "projected_visible_ratio": slot.visible_ratio,
            "reference_mask_source": slot.mask_source,
            "candidate_overlaps": [
                {
                    "observed_id": item.observed.observed_id,
                    "track_id": item.observed.track_id,
                    **item.to_components(),
                }
                for item in overlaps
            ],
        }
        if appearance_components is not None:
            components["appearance_components"] = appearance_components

        metrics = {
            "primary_mask_iou": primary.mask_iou if primary is not None else None,
            "primary_expected_coverage": primary.expected_coverage if primary is not None else None,
            "primary_observed_coverage": primary.observed_coverage if primary is not None else None,
            "primary_center_error_ratio": primary.center_error_ratio if primary is not None else None,
            "appearance_similarity": appearance_score,
        }
        errors = {
            "missing": missing,
            "duplicate": duplicate,
            "replacement": replacement,
            "appearance_error": replacement,
            "category_error": category_error,
            "position_error": position_error,
            "shape_error": shape_error,
            "merge_error": merge_error,
        }
        role = "anchor" if slot.reference_id == anchor_ref_id else "slot"
        slot_audits.append(
            SlotAudit(
                reference_id=slot.reference_id,
                occupied_observed_ids=occupied_ids,
                occupant_track_ids=track_ids,
                primary_observed_id=primary.observed.observed_id if primary is not None else None,
                primary_track_id=primary.observed.track_id if primary is not None else None,
                role=role,
                errors=errors,
                metrics=metrics,
                components=components,
            )
        )
        if primary is not None:
            matches.append(
                ObjectMatch(
                    observed_id=primary.observed.observed_id,
                    reference_id=slot.reference_id,
                    track_id=primary.observed.track_id,
                    role="anchor" if role == "anchor" else "slot_primary",
                    cost=1.0 - primary.mask_iou,
                    scores={
                        "mask_iou": primary.mask_iou,
                        "expected_coverage": primary.expected_coverage,
                        "observed_coverage": primary.observed_coverage,
                        "appearance_similarity": appearance_score,
                    },
                    components=components,
                )
            )

    occupied_observed = set(observed_to_slots)
    unmatched: list[UnmatchedObserved] = []
    for observed in window.objects:
        if observed.observed_id in occupied_observed:
            continue
        best = max_overlap_by_observed.get(observed.observed_id)
        unmatched.append(
            UnmatchedObserved(
                observed_id=observed.observed_id,
                track_id=observed.track_id,
                kind="hallucinated",
                max_slot_iou=best.mask_iou if best is not None else None,
                max_expected_coverage=best.expected_coverage if best is not None else None,
                max_observed_coverage=best.observed_coverage if best is not None else None,
            )
        )

    return slot_audits, matches, missing_ids, unmatched, diagnostics


def evaluate_anchor_candidate(
    scene: ReferenceScene,
    window: ObservedWindow,
    candidate: AnchorCandidate,
    config: EvaluationConfig,
    requested_zoom: float | None = None,
    reference_masks: dict[str, MaskStats | None] | None = None,
    observed_masks: dict[str, MaskStats | None] | None = None,
) -> LocalExplanation:
    reference_mask_cache = reference_masks or _reference_mask_cache(scene, config)
    observed_mask_cache = observed_masks or _observed_mask_cache(window, config)
    slots, camera, projection_diagnostics = _slot_projections_for_anchor(
        scene,
        window,
        candidate.observed,
        candidate.reference,
        config,
        requested_zoom,
        reference_mask_cache,
        observed_mask_cache,
    )
    if not camera.valid:
        return LocalExplanation(
            window_id=window.window_id,
            valid=False,
            anchor_observed_id=candidate.observed.observed_id,
            anchor_reference_id=candidate.reference.object_id,
            anchor_similarity=candidate.similarity,
            camera_scores=_camera_scores(camera),
            diagnostics={
                "anchor_components": candidate.components,
                "camera_estimate": camera.to_dict(),
                **projection_diagnostics,
            },
            error=camera.error or "invalid anchor camera estimate",
        )
    if not slots:
        return LocalExplanation(
            window_id=window.window_id,
            valid=False,
            anchor_observed_id=candidate.observed.observed_id,
            anchor_reference_id=candidate.reference.object_id,
            anchor_similarity=candidate.similarity,
            camera_scores=_camera_scores(camera),
            diagnostics={
                "anchor_components": candidate.components,
                "camera_estimate": camera.to_dict(),
                **projection_diagnostics,
            },
            error="no projected reference slots",
        )

    slot_audits, matches, missing_ids, unmatched_observed, audit_diagnostics = _audit_slots(
        scene,
        window,
        candidate,
        slots,
        config,
        observed_mask_cache,
    )
    scores = _score_slot_audits(slot_audits, window, unmatched_observed)
    expected_ids = [slot.reference_id for slot in slots]
    return LocalExplanation(
        window_id=window.window_id,
        valid=True,
        anchor_observed_id=candidate.observed.observed_id,
        anchor_reference_id=candidate.reference.object_id,
        anchor_similarity=candidate.similarity,
        expected_reference_ids=expected_ids,
        slot_audits=slot_audits,
        matches=matches,
        missing_reference_ids=missing_ids,
        unmatched_observed=unmatched_observed,
        scores=scores,
        camera_scores=_camera_scores(camera),
        diagnostics={
            "anchor_components": candidate.components,
            "camera_estimate": camera.to_dict(),
            "num_anchor_candidates_considered": None,
            **projection_diagnostics,
            **audit_diagnostics,
        },
    )


def _camera_scores(camera: AnchorCameraEstimate) -> dict[str, Any]:
    return {
        "actual_zoom": camera.actual_zoom,
        "requested_zoom": camera.requested_zoom,
        "zoom_relative_error": camera.zoom_relative_error,
        "zoom_compliance_score": camera.zoom_compliance_score,
        "zoom_pass": camera.zoom_pass,
        "zoom_anisotropy": camera.zoom_anisotropy,
    }


def _score_slot_audits(
    slot_audits: list[SlotAudit],
    window: ObservedWindow,
    unmatched_observed: list[UnmatchedObserved],
) -> dict[str, float]:
    expected_count = len(slot_audits)
    observed_count = len(window.objects)
    occupied = [slot for slot in slot_audits if not slot.errors.get("missing", False)]
    occupied_count = len(occupied)

    missing_count = sum(1 for slot in slot_audits if slot.errors.get("missing", False))
    duplicate_slot_count = sum(1 for slot in slot_audits if slot.errors.get("duplicate", False))
    duplicate_extra_count = sum(max(0, len(slot.occupied_observed_ids) - 1) for slot in slot_audits)
    replacement_count = sum(1 for slot in occupied if slot.errors.get("replacement", False))
    appearance_error_count = sum(1 for slot in occupied if slot.errors.get("appearance_error", False))
    category_error_count = sum(1 for slot in occupied if slot.errors.get("category_error", False))
    position_error_count = sum(1 for slot in occupied if slot.errors.get("position_error", False))
    shape_error_count = sum(1 for slot in occupied if slot.errors.get("shape_error", False))
    merge_slot_count = sum(1 for slot in occupied if slot.errors.get("merge_error", False))
    hallucinated_count = len(unmatched_observed)
    erroneous_slot_count = sum(
        1
        for slot in slot_audits
        if any(slot.errors.get(key, False) for key in (
            "missing",
            "duplicate",
            "replacement",
            "position_error",
            "shape_error",
            "merge_error",
        ))
    )

    presence_rate = occupied_count / max(expected_count, 1)
    missing_rate = missing_count / max(expected_count, 1)
    hallucination_rate = hallucinated_count / max(observed_count, 1)
    local_precision = occupied_count / max(observed_count, 1)
    precision_recall_score = 0.5 * (local_precision + presence_rate)

    return {
        "expected_count": float(expected_count),
        "observed_count": float(observed_count),
        "occupied_slot_count": float(occupied_count),
        "missing_count": float(missing_count),
        "hallucinated_count": float(hallucinated_count),
        "duplicate_slot_count": float(duplicate_slot_count),
        "duplicate_extra_count": float(duplicate_extra_count),
        "replacement_count": float(replacement_count),
        "appearance_error_count": float(appearance_error_count),
        "category_error_count": float(category_error_count),
        "position_error_count": float(position_error_count),
        "shape_error_count": float(shape_error_count),
        "merge_slot_count": float(merge_slot_count),
        "erroneous_slot_count": float(erroneous_slot_count),
        "presence_rate": presence_rate,
        "missing_rate": missing_rate,
        "hallucination_rate": hallucination_rate,
        "hallucination_per_expected": hallucinated_count / max(expected_count, 1),
        "duplicate_rate": duplicate_extra_count / max(expected_count, 1),
        "replacement_rate": replacement_count / max(occupied_count, 1),
        "appearance_error_rate": appearance_error_count / max(occupied_count, 1),
        "category_error_rate": category_error_count / max(occupied_count, 1),
        "position_error_rate": position_error_count / max(occupied_count, 1),
        "shape_error_rate": shape_error_count / max(occupied_count, 1),
        "merge_rate": merge_slot_count / max(expected_count, 1),
        "object_memory_error_rate": erroneous_slot_count / max(expected_count, 1),
        "precision_recall_score": precision_recall_score,
    }


def missing_only_explanation(
    window: ObservedWindow,
    expected_reference_ids: list[str],
    config: EvaluationConfig,
    *,
    requested_zoom: float | None,
    source_window: ObservedWindow,
) -> LocalExplanation:
    del config
    slot_audits = [
        SlotAudit(
            reference_id=ref_id,
            errors={
                "missing": True,
                "duplicate": False,
                "replacement": False,
                "appearance_error": False,
                "category_error": False,
                "position_error": False,
                "shape_error": False,
                "merge_error": False,
            },
        )
        for ref_id in expected_reference_ids
    ]
    scores = _score_slot_audits(slot_audits, window, unmatched_observed=[])
    frame_gap = None
    if window.frame_index is not None and source_window.frame_index is not None:
        frame_gap = window.frame_index - source_window.frame_index
    timestamp_gap = None
    if window.timestamp is not None and source_window.timestamp is not None:
        timestamp_gap = window.timestamp - source_window.timestamp
    return LocalExplanation(
        window_id=window.window_id,
        valid=True,
        expected_reference_ids=list(expected_reference_ids),
        slot_audits=slot_audits,
        missing_reference_ids=list(expected_reference_ids),
        scores=scores,
        camera_scores={
            "actual_zoom": None,
            "requested_zoom": requested_zoom,
            "zoom_relative_error": None,
            "zoom_compliance_score": None,
            "zoom_pass": None,
            "zoom_anisotropy": None,
        },
        diagnostics={
            "empty_window_policy": "previous_view_missing",
            "reference_context_source_window_id": source_window.window_id,
            "reference_context_source_frame_index": source_window.frame_index,
            "reference_context_frame_gap": frame_gap,
            "reference_context_timestamp_gap": timestamp_gap,
        },
    )


def evaluate_window(
    scene: ReferenceScene,
    window: ObservedWindow,
    config: EvaluationConfig,
    requested_zoom: float | None = None,
) -> tuple[LocalExplanation, list[LocalExplanation]]:
    if not window.objects:
        invalid = LocalExplanation(
            window_id=window.window_id,
            valid=False,
            error="no observed objects in window",
        )
        return invalid, [invalid]

    candidates = top_anchor_candidates(scene, window, config)
    if not candidates:
        invalid = LocalExplanation(
            window_id=window.window_id,
            valid=False,
            error="no anchor candidate above min_anchor_similarity",
        )
        return invalid, [invalid]

    reference_masks = _reference_mask_cache(scene, config)
    observed_masks = _observed_mask_cache(window, config)
    explanations = [
        evaluate_anchor_candidate(
            scene,
            window,
            candidate,
            config,
            requested_zoom=requested_zoom,
            reference_masks=reference_masks,
            observed_masks=observed_masks,
        )
        for candidate in candidates
    ]

    valid = [item for item in explanations if item.valid]
    best = valid[0] if valid else explanations[0]
    best.diagnostics["num_anchor_candidates_considered"] = len(explanations)
    if len(candidates) > 1:
        best_similarity = candidates[0].similarity
        second_similarity = candidates[1].similarity
        best.diagnostics["anchor_margin"] = best_similarity - second_similarity
        best.diagnostics["anchor_ambiguity_count"] = sum(
            1
            for item in candidates
            if best_similarity - item.similarity <= config.ambiguity_epsilon
        )
    else:
        best.diagnostics["anchor_margin"] = 1.0
        best.diagnostics["anchor_ambiguity_count"] = 1
    best.diagnostics["anchor_selection_policy"] = "top_appearance_similarity_evaluable"
    return best, explanations
