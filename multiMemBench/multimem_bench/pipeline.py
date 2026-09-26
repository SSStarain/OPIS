"""Evaluation orchestration, slot error aggregation, and track metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multimem_bench.camera import requested_zoom_from_metadata
from multimem_bench.config import EvaluationConfig
from multimem_bench.io import write_json, write_jsonl
from multimem_bench.matching import (
    LocalExplanation,
    SlotAudit,
    evaluate_window,
    missing_only_explanation,
)
from multimem_bench.schema import (
    ObservedWindow,
    ReferenceScene,
    VideoObservation,
)


@dataclass
class EvaluationRunResult:
    video_id: str
    scene_id: str
    window_results: list[LocalExplanation]
    anchor_hypotheses: list[LocalExplanation]
    summary: dict[str, Any] = field(default_factory=dict)


def evaluate_video_observations(
    scene: ReferenceScene,
    observation: VideoObservation,
    config: EvaluationConfig | None = None,
) -> EvaluationRunResult:
    cfg = config or EvaluationConfig()
    requested_zoom = requested_zoom_from_metadata(scene, observation)
    selected: list[LocalExplanation] = []
    all_hypotheses: list[LocalExplanation] = []
    previous_context: tuple[ObservedWindow, LocalExplanation] | None = None
    for window in observation.windows:
        if (
            not window.objects
            and cfg.empty_window_policy == "previous_view_missing"
            and previous_context is not None
        ):
            source_window, source_explanation = previous_context
            best = missing_only_explanation(
                window,
                source_explanation.expected_reference_ids,
                cfg,
                requested_zoom=requested_zoom,
                source_window=source_window,
            )
            hypotheses = [best]
        else:
            best, hypotheses = evaluate_window(
                scene,
                window,
                cfg,
                requested_zoom=requested_zoom,
            )
        selected.append(best)
        all_hypotheses.extend(hypotheses)
        if best.valid and best.anchor_reference_id is not None:
            previous_context = (window, best)

    summary = summarize_results(
        scene.scene_id,
        observation.video_id,
        selected,
        scene=scene,
        config=cfg,
    )
    return EvaluationRunResult(
        video_id=observation.video_id,
        scene_id=scene.scene_id,
        window_results=selected,
        anchor_hypotheses=all_hypotheses,
        summary=summary,
    )


_COUNT_KEYS = (
    "expected_count",
    "observed_count",
    "occupied_slot_count",
    "missing_count",
    "hallucinated_count",
    "duplicate_slot_count",
    "duplicate_extra_count",
    "replacement_count",
    "appearance_error_count",
    "category_error_count",
    "position_error_count",
    "shape_error_count",
    "merge_slot_count",
    "erroneous_slot_count",
)

_RATE_KEYS = (
    "presence_rate",
    "missing_rate",
    "hallucination_rate",
    "hallucination_per_expected",
    "duplicate_rate",
    "replacement_rate",
    "appearance_error_rate",
    "category_error_rate",
    "position_error_rate",
    "shape_error_rate",
    "merge_rate",
    "object_memory_error_rate",
)


def summarize_results(
    scene_id: str,
    video_id: str,
    windows: list[LocalExplanation],
    *,
    scene: ReferenceScene | None = None,
    config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or EvaluationConfig()
    valid = [w for w in windows if w.valid]
    window_mean_error_rates = {
        key: _mean([float(w.scores[key]) for w in valid if key in w.scores])
        for key in _RATE_KEYS
    }
    counts = {
        key: float(sum(float(w.scores.get(key, 0.0)) for w in valid))
        for key in _COUNT_KEYS
    }
    micro_rates = _micro_error_rates(counts)
    precision_recall_score = _micro_precision_recall_score(counts)
    continuous_metrics = _continuous_slot_metrics(valid)
    temporal = _temporal_metrics(valid, scene=scene, config=cfg)
    camera_scores = _camera_score_summary(valid)

    ambiguity_vals = [
        float(w.diagnostics.get("anchor_ambiguity_count", 0))
        for w in valid
    ]
    margin_vals = [
        float(w.diagnostics.get("anchor_margin", 0.0))
        for w in valid
    ]
    key_metrics = {
        "object_memory_error_rate": micro_rates["object_memory_error_rate"],
        "missing_rate": micro_rates["missing_rate"],
        "hallucination_rate": micro_rates["hallucination_rate"],
        "position_error_rate": micro_rates["position_error_rate"],
        "shape_error_rate": micro_rates["shape_error_rate"],
        "mean_primary_mask_iou": continuous_metrics["mean_primary_mask_iou"],
        "mean_expected_coverage": continuous_metrics["mean_expected_coverage"],
        "mean_observed_coverage": continuous_metrics["mean_observed_coverage"],
        "mean_center_error_ratio": continuous_metrics["mean_center_error_ratio"],
        "precision_recall_score": precision_recall_score,
        "track_match_stability": temporal["track_match_stability"],
        "id_switch_rate": temporal["id_switch_rate"],
        "track_fragmentation_rate": temporal["track_fragmentation_rate"],
        "reappearance_identity_failure_rate": temporal[
            "reappearance_identity_failure_rate"
        ],
        "unexpected_disappearance_rate": temporal["unexpected_disappearance_rate"],
        "temporal_appearance_drift": temporal["temporal_appearance_drift"],
        "zoom_compliance_score": camera_scores["zoom_compliance_score"],
        "zoom_pass_rate": camera_scores["zoom_pass_rate"],
        "requested_zoom": camera_scores["requested_zoom"],
        "mean_actual_zoom": camera_scores["mean_actual_zoom"],
        "mean_zoom_relative_error": camera_scores["mean_zoom_relative_error"],
    }
    return {
        "scene_id": scene_id,
        "video_id": video_id,
        "num_windows": len(windows),
        "num_valid_windows": len(valid),
        "key_metrics": key_metrics,
        "error_counts": counts,
        "error_rates": micro_rates,
        "window_mean_error_rates": window_mean_error_rates,
        "continuous_metrics": continuous_metrics,
        "camera_scores": camera_scores,
        "temporal_error_rates": temporal,
        "anchor_ambiguity": {
            "mean_ambiguity_count": (
                sum(ambiguity_vals) / len(ambiguity_vals)
                if ambiguity_vals
                else None
            ),
            "mean_anchor_margin": (
                sum(margin_vals) / len(margin_vals)
                if margin_vals
                else None
            ),
        },
        "invalid_windows": [
            {"window_id": w.window_id, "error": w.error}
            for w in windows
            if not w.valid
        ],
        "statistics_note": (
            "key_metrics is the compact benchmark report. The remaining groups "
            "contain complete count, rate, continuous, camera, temporal, and "
            "diagnostic data."
        ),
    }


def _micro_error_rates(counts: dict[str, float]) -> dict[str, float]:
    expected = max(counts.get("expected_count", 0.0), 1.0)
    observed = max(counts.get("observed_count", 0.0), 1.0)
    occupied = max(counts.get("occupied_slot_count", 0.0), 1.0)
    return {
        "presence_rate": counts.get("occupied_slot_count", 0.0) / expected,
        "missing_rate": counts.get("missing_count", 0.0) / expected,
        "hallucination_rate": counts.get("hallucinated_count", 0.0) / observed,
        "hallucination_per_expected": counts.get("hallucinated_count", 0.0) / expected,
        "duplicate_rate": counts.get("duplicate_extra_count", 0.0) / expected,
        "replacement_rate": counts.get("replacement_count", 0.0) / occupied,
        "appearance_error_rate": counts.get("appearance_error_count", 0.0) / occupied,
        "category_error_rate": counts.get("category_error_count", 0.0) / occupied,
        "position_error_rate": counts.get("position_error_count", 0.0) / occupied,
        "shape_error_rate": counts.get("shape_error_count", 0.0) / occupied,
        "merge_rate": counts.get("merge_slot_count", 0.0) / expected,
        "object_memory_error_rate": counts.get("erroneous_slot_count", 0.0) / expected,
    }


def _micro_precision_recall_score(counts: dict[str, float]) -> float:
    expected = max(counts.get("expected_count", 0.0), 1.0)
    observed = max(counts.get("observed_count", 0.0), 1.0)
    occupied = counts.get("occupied_slot_count", 0.0)
    precision = occupied / observed
    recall = occupied / expected
    return 0.5 * (precision + recall)


def _continuous_slot_metrics(windows: list[LocalExplanation]) -> dict[str, float | None]:
    values: dict[str, list[float]] = {
        "primary_mask_iou": [],
        "expected_coverage": [],
        "observed_coverage": [],
        "center_error_ratio": [],
        "appearance_similarity": [],
    }
    metric_to_key = {
        "primary_mask_iou": "primary_mask_iou",
        "expected_coverage": "primary_expected_coverage",
        "observed_coverage": "primary_observed_coverage",
        "center_error_ratio": "primary_center_error_ratio",
        "appearance_similarity": "appearance_similarity",
    }
    for window in windows:
        for slot in window.slot_audits:
            for output_key, metric_key in metric_to_key.items():
                value = slot.metrics.get(metric_key)
                if value is not None:
                    values[output_key].append(float(value))

    result: dict[str, float | None] = {}
    for key, items in values.items():
        result[f"mean_{key}"] = _mean(items) if items else None
        result[f"median_{key}"] = _quantile(items, 0.5)
        result[f"p10_{key}"] = _quantile(items, 0.1)
        result[f"p90_{key}"] = _quantile(items, 0.9)
    return result


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _camera_score_summary(windows: list[LocalExplanation]) -> dict[str, Any]:
    zoom_scores = [
        float(w.camera_scores["zoom_compliance_score"])
        for w in windows
        if w.camera_scores.get("zoom_compliance_score") is not None
    ]
    actual_zoom = [
        float(w.camera_scores["actual_zoom"])
        for w in windows
        if w.camera_scores.get("actual_zoom") is not None
    ]
    rel_errors = [
        float(w.camera_scores["zoom_relative_error"])
        for w in windows
        if w.camera_scores.get("zoom_relative_error") is not None
    ]
    zoom_pass = [
        bool(w.camera_scores["zoom_pass"])
        for w in windows
        if w.camera_scores.get("zoom_pass") is not None
    ]
    requested_values = [
        float(w.camera_scores["requested_zoom"])
        for w in windows
        if w.camera_scores.get("requested_zoom") is not None
    ]
    return {
        "zoom_compliance_score": _mean_or_none(zoom_scores),
        "zoom_pass_rate": (
            sum(1.0 for item in zoom_pass if item) / len(zoom_pass)
            if zoom_pass
            else None
        ),
        "requested_zoom": requested_values[0] if requested_values else None,
        "mean_actual_zoom": _mean_or_none(actual_zoom),
        "mean_zoom_relative_error": _mean_or_none(rel_errors),
        "num_zoom_scored_windows": len(zoom_scores),
        "note": "Camera trajectory scores are reported separately and are not included in object error rates.",
    }


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _temporal_metrics(
    windows: list[LocalExplanation],
    *,
    scene: ReferenceScene | None,
    config: EvaluationConfig,
) -> dict[str, Any]:
    if not windows:
        return {
            "track_match_stability": None,
            "id_switch_count": 0,
            "id_switch_rate": None,
            "fragmented_reference_count": 0,
            "track_fragmentation_rate": None,
            "reappearance_events": 0,
            "reappearance_identity_failure_count": 0,
            "reappearance_identity_failure_rate": None,
            "unexpected_disappearance_count": 0,
            "unexpected_disappearance_rate": None,
            "visibility_opportunities_after_first_appearance": 0,
            "temporal_appearance_drift": None,
            "details": {},
        }

    track_events: dict[str, list[dict[str, Any]]] = {}
    reference_tracks: dict[str, set[str]] = {}
    track_appearance: dict[str, list[float]] = {}
    slot_by_window: list[dict[str, SlotAudit]] = []
    ambiguous_track_windows: list[dict[str, Any]] = []
    for window_index, explanation in enumerate(windows):
        slot_map = {slot.reference_id: slot for slot in explanation.slot_audits}
        slot_by_window.append(slot_map)
        assignments_by_track: dict[str, list[SlotAudit]] = {}
        for slot in explanation.slot_audits:
            if not slot.primary_track_id:
                continue
            assignments_by_track.setdefault(slot.primary_track_id, []).append(slot)
        for track_id, assignments in assignments_by_track.items():
            reference_ids = sorted({slot.reference_id for slot in assignments})
            if len(reference_ids) > 1:
                # A single observed mask merged multiple reference slots in this
                # frame. That is a spatial merge error, not a temporal ID switch.
                ambiguous_track_windows.append({
                    "window_index": window_index,
                    "window_id": explanation.window_id,
                    "track_id": track_id,
                    "reference_ids": reference_ids,
                })
                continue
            slot = max(
                assignments,
                key=lambda item: float(item.metrics.get("primary_mask_iou") or 0.0),
            )
            appearance = slot.metrics.get("appearance_similarity")
            event = {
                "window_index": window_index,
                "window_id": explanation.window_id,
                "reference_id": slot.reference_id,
                "track_id": track_id,
                "mask_iou": slot.metrics.get("primary_mask_iou"),
                "appearance_similarity": appearance,
            }
            track_events.setdefault(track_id, []).append(event)
            reference_tracks.setdefault(slot.reference_id, set()).add(track_id)
            if appearance is not None:
                track_appearance.setdefault(track_id, []).append(float(appearance))

    switch_count = 0
    transition_count = 0
    stability_values: list[float] = []
    stability_details: dict[str, Any] = {}
    for track_id, events in track_events.items():
        events.sort(key=lambda item: item["window_index"])
        refs = [str(item["reference_id"]) for item in events]
        counts: dict[str, int] = {}
        for ref_id in refs:
            counts[ref_id] = counts.get(ref_id, 0) + 1
        majority_ref, majority_count = max(counts.items(), key=lambda item: item[1])
        stability = majority_count / len(refs)
        stability_values.append(stability)
        local_switches = 0
        for first, second in zip(events, events[1:]):
            transition_count += 1
            if first["reference_id"] != second["reference_id"]:
                switch_count += 1
                local_switches += 1
        stability_details[track_id] = {
            "num_assignments": len(events),
            "majority_reference_id": majority_ref,
            "stability": stability,
            "id_switch_count": local_switches,
            "assignments": events,
        }

    fragmented = {
        ref_id: sorted(track_ids)
        for ref_id, track_ids in reference_tracks.items()
        if len(track_ids) > 1
    }
    reference_count_with_tracks = len(reference_tracks)
    fragmentation_rate = (
        len(fragmented) / reference_count_with_tracks
        if reference_count_with_tracks
        else None
    )

    reappearance_events = 0
    reappearance_failures = 0
    unexpected_disappearances = 0
    visibility_opportunities = 0
    reappearance_details: list[dict[str, Any]] = []
    all_ref_ids = sorted({
        ref_id
        for slot_map in slot_by_window
        for ref_id in slot_map
    })
    for ref_id in all_ref_ids:
        previous_track: str | None = None
        had_occupied = False
        gap_windows = 0
        gap_has_visible_missing = False
        for window_index, slot_map in enumerate(slot_by_window):
            slot = slot_map.get(ref_id)
            if slot is None:
                if had_occupied:
                    gap_windows += 1
                continue
            occupied = slot.primary_observed_id is not None
            if had_occupied:
                visibility_opportunities += 1
            if occupied:
                if (
                    had_occupied
                    and gap_windows >= config.track_reappearance_min_gap_windows
                ):
                    reappearance_events += 1
                    current_track = slot.primary_track_id
                    failure = (
                        previous_track is not None
                        and current_track is not None
                        and current_track != previous_track
                    )
                    if failure:
                        reappearance_failures += 1
                    reappearance_details.append({
                        "reference_id": ref_id,
                        "window_index": window_index,
                        "previous_track_id": previous_track,
                        "reappearance_track_id": current_track,
                        "gap_windows": gap_windows,
                        "identity_failure": failure,
                    })
                had_occupied = True
                previous_track = slot.primary_track_id or previous_track
                gap_windows = 0
                gap_has_visible_missing = False
            elif had_occupied:
                gap_windows += 1
                if not gap_has_visible_missing:
                    unexpected_disappearances += 1
                    gap_has_visible_missing = True

    drift_values: list[float] = []
    drift_details: dict[str, Any] = {}
    for track_id, values in track_appearance.items():
        if len(values) >= 2:
            pairwise_changes = [
                abs(second - first)
                for first, second in zip(values, values[1:])
            ]
            drift = _mean(pairwise_changes)
            drift_values.append(drift)
            drift_details[track_id] = {
                "num_consecutive_comparisons": len(pairwise_changes),
                "mean_absolute_similarity_change": drift,
                "min_similarity_to_reference": min(values),
                "max_similarity_to_reference": max(values),
                "drift": drift,
            }

    return {
        "track_match_stability": _mean(stability_values) if stability_values else None,
        "id_switch_count": switch_count,
        "id_switch_rate": (
            switch_count / transition_count
            if transition_count
            else None
        ),
        "track_transition_count": transition_count,
        "fragmented_reference_count": len(fragmented),
        "track_fragmentation_rate": fragmentation_rate,
        "reappearance_events": reappearance_events,
        "reappearance_identity_failure_count": reappearance_failures,
        "reappearance_identity_failure_rate": (
            reappearance_failures / reappearance_events
            if reappearance_events
            else None
        ),
        "unexpected_disappearance_count": unexpected_disappearances,
        "unexpected_disappearance_rate": (
            unexpected_disappearances / visibility_opportunities
            if visibility_opportunities
            else None
        ),
        "visibility_opportunities_after_first_appearance": visibility_opportunities,
        "temporal_appearance_drift": _mean(drift_values) if drift_values else None,
        "ambiguous_track_window_count": len(ambiguous_track_windows),
        "details": {
            "tracks": stability_details,
            "fragmented_references": fragmented,
            "reappearances": reappearance_details,
            "embedding_drift": drift_details,
            "ambiguous_track_windows": ambiguous_track_windows,
            "scene_available": scene is not None,
        },
    }
def write_run_result(result: EvaluationRunResult, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": str(write_json(out / "summary.json", result.summary)),
        "windows": str(write_jsonl(
            out / "window_results.jsonl",
            [w.to_dict() for w in result.window_results],
        )),
        "anchor_hypotheses": str(write_jsonl(
            out / "anchor_hypotheses.jsonl",
            [w.to_dict() for w in result.anchor_hypotheses],
        )),
    }
    return paths
