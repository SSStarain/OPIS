"""Auditable Core-2D object-memory metrics for V3."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import math
from typing import Any, Sequence


@dataclass(frozen=True)
class V3AggregationConfig:
    min_geometry_coverage: float | None = None
    min_geometry_frames: int = 3
    presence_weight: float = 0.2
    identity_weight: float = 0.4
    geometry_weight: float = 0.4

    def validate(self) -> None:
        weights = (self.presence_weight, self.identity_weight, self.geometry_weight)
        if any(float(value) < 0 for value in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
            raise ValueError("Core-2D/Full-IGM weights must be non-negative and sum to one")
        if (self.min_geometry_coverage is not None and not 0.0 < self.min_geometry_coverage <= 1.0) or self.min_geometry_frames < 1:
            raise ValueError("invalid geometry eligibility thresholds")


def aggregate_core_2d(rows: Sequence[Any], config: V3AggregationConfig | None = None, *,
                      fp_attributable: bool = True) -> dict[str, Any]:
    """Aggregate one case using strict frame Presence and identity evidence.

    Rows with ``presence=None`` are non-observable and excluded. A row with
    ``state=generation_failure`` and presence zero is an eligible false
    negative. Unmatched observations are counted once per frame from the
    matching diagnostics, rather than once per reference row.
    ``presence_score`` is the strict frame Presence score: an evaluable frame
    is one only when no confirmed reference is missing and no confirmed extra
    observation remains after one-to-one association. ``presence_f1`` is kept
    as a softer diagnostic that retains known TP/FN/FP and explicit penalties
    for unverified missing references and candidate extras. Strict F1 and
    precision remain null when any FP window is unknown. An empty denominator
    returns an explicitly labeled zero fallback.
    """

    cfg = config or V3AggregationConfig()
    cfg.validate()
    eligible_rows = [row for row in rows if row.presence is not None]
    true_positive = sum(float(row.presence or 0.0) > 0.0 for row in eligible_rows)
    false_negative = sum(
        row.state == "generation_failure" and float(row.presence or 0.0) <= 0.0
        for row in eligible_rows
    )
    window_evaluable = {}
    for row in rows:
        window = _window(row)
        window_evaluable[window] = window_evaluable.get(window, True) and fp_attributable and bool(
            row.diagnostics.get("false_positive_evaluable", True))
    fp_rows = [row for row in rows if window_evaluable[_window(row)]]
    false_positive = _unmatched_observations(fp_rows)
    unverified_missing = sum(
        row.presence is None
        and row.diagnostics.get("visibility_state", row.diagnostics.get("reason"))
        not in {"occluded", "outside_view"}
        and row.diagnostics.get("reason") not in {"occluded", "outside_view"}
        for row in rows
    )
    unverified_extra = 0
    if fp_attributable:
        unknown_rows = [row for row in rows if not window_evaluable[_window(row)]]
        if unknown_rows:
            # The evaluator preserves candidate extras even when attribution is
            # unavailable. Old artifacts only have the ordinary confirmed-FP
            # count, which must not be reinterpreted as an unverified extra.
            by_window: dict[Any, int] = {}
            for row in unknown_rows:
                candidate_count = row.diagnostics.get("unmatched_observed_candidate_count")
                if candidate_count is not None:
                    by_window[_window(row)] = max(
                        by_window.get(_window(row), 0), int(candidate_count or 0))
            unverified_extra = sum(by_window.values())
    presence_recall = _ratio(
        true_positive,
        true_positive + false_negative + unverified_missing,
    )
    presence_precision = _ratio(
        true_positive,
        true_positive + false_positive + unverified_extra,
    )
    presence_f1 = _ratio(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative
        + unverified_missing + unverified_extra,
    )
    known_presence_f1 = _ratio(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
    )
    fp_coverage = _ratio(sum(window_evaluable.values()), len(window_evaluable))
    strict_f1 = presence_f1
    # Retain known TP/FN/FP. Unknown extras do not veto the relaxed score;
    # strict diagnostics still distinguish missing evidence from zero FP.
    if any(not value for value in window_evaluable.values()):
        presence_precision = strict_f1 = None
    has_presence_evidence = any(row.presence is not None for row in rows)
    score_status = ("no_evidence_zero_fallback" if not has_presence_evidence else
                    "partial_fp_evidence" if strict_f1 is None else "complete_fp_evidence")
    presence_f1 = presence_f1 if presence_f1 is not None else 0.0

    identity_rows = [
        row for row in rows
        if row.identity is not None and math.isfinite(row.identity) and row.presence == 1.0
    ]
    identity_by_instance: dict[str, list[float]] = {}
    for row in identity_rows:
        identity_by_instance.setdefault(row.reference_id, []).append(float(row.identity))
    identity_score = _mean([
        float(_mean(values)) for values in identity_by_instance.values() if values
    ])
    identity_status = "observed" if identity_score is not None else "no_evidence_zero_fallback"
    identity_score = identity_score if identity_score is not None else 0.0
    set_windows: dict[tuple[str, Any], list[float]] = {}
    for row in identity_rows:
        group = row.diagnostics.get("ambiguity_group")
        if group:
            set_windows.setdefault((str(group), _window(row)), []).append(float(row.identity))
    set_groups: dict[str, list[float]] = {}
    for (group, _), values in set_windows.items():
        set_groups.setdefault(group, []).append(float(_mean(values)))
    set_identity = _mean([float(_mean(values)) for values in set_groups.values()])
    recall_by_instance: dict[str, list[float]] = {}
    for row in eligible_rows:
        recall_by_instance.setdefault(row.reference_id, []).append(float(row.presence))
    ambiguity_rate = _ratio(
        sum(bool(row.diagnostics.get("ambiguous_match")) for row in identity_rows),
        len(identity_rows),
    )
    swap_rate = _identity_swap_rate(identity_rows)
    unambiguous_rows = [row for row in identity_rows if not row.diagnostics.get("ambiguous_match")]
    unambiguous_by_instance: dict[str, list[float]] = {}
    for row in unambiguous_rows:
        unambiguous_by_instance.setdefault(row.reference_id, []).append(float(row.identity))
    identity_unambiguous = _mean([
        float(_mean(values)) for values in unambiguous_by_instance.values() if values
    ])
    strict_frame_macro = _strict_presence_frame_macro(rows, fp_attributable=fp_attributable)
    strict_presence = strict_frame_macro if strict_frame_macro is not None else 0.0
    core_2d = None
    if identity_score is not None:
        core_2d = 100.0 * math.sqrt(_unit(strict_presence) * _unit(identity_score))

    frame_f1 = _frame_macro_f1(fp_rows)
    return {
        "presence_true_positive": true_positive,
        "presence_false_negative": false_negative,
        "presence_false_positive": false_positive,
        "presence_unverified_missing": unverified_missing,
        "presence_unverified_extra": unverified_extra,
        "presence_recall": presence_recall,
        "presence_precision": presence_precision,
        "presence_f1": presence_f1,
        "presence_score": strict_presence,
        "presence_score_known_evidence": known_presence_f1 if known_presence_f1 is not None else 0.0,
        "presence_f1_strict": strict_f1,
        "presence_score_policy": "strict_frame_presence_v1",
        "presence_score_status": score_status,
        "presence_f1_frame_macro": frame_f1,
        "presence_strict_frame_macro": strict_frame_macro,
        "presence_evidence_coverage": _ratio(len(eligible_rows),len(rows)),
        "false_positive_evidence_coverage": fp_coverage,
        "presence_recall_instance_macro": _mean([float(_mean(values)) for values in recall_by_instance.values()]),
        "identity_score": identity_score,
        "identity_score_unambiguous": identity_unambiguous,
        "identity_unambiguous_rows": len(unambiguous_rows),
        "identity_unambiguous_instances": len(unambiguous_by_instance),
        "identity_score_status": identity_status,
        "set_identity_score": set_identity,
        "identity_evidence_coverage": _ratio(len(identity_rows), sum(row.presence == 1.0 for row in rows)),
        "identity_ambiguity_rate": ambiguity_rate,
        "identity_swap_rate": swap_rate,
        "core_2d_score": core_2d,
        # Compatibility fields retained for one result-version migration.
        "v3_presence_score": strict_presence,
        "v3_identity_score": identity_score,
    }


def _unmatched_observations(rows: Sequence[Any]) -> int:
    seen_frames: set[Any] = set()
    total = 0
    for row in rows:
        frame = _window(row)
        if frame in seen_frames:
            continue
        seen_frames.add(frame)
        total += int(
            row.diagnostics.get("unmatched_observed_count", 0) or 0
        )
    return total


def _strict_presence_frame_macro(rows: Sequence[Any], *, fp_attributable: bool) -> float | None:
    """Score each evaluable frame as 1 only when no reference is missing and no extra is confirmed."""
    grouped: dict[Any, list[Any]] = {}
    for row in rows:
        grouped.setdefault(_window(row), []).append(row)
    values: list[float] = []
    for frame_rows in grouped.values():
        eligible = [row for row in frame_rows if row.presence is not None]
        extra_evaluable = fp_attributable and all(
            bool(row.diagnostics.get("false_positive_evaluable", True))
            for row in frame_rows
        )
        extra_count = max(
            (int(row.diagnostics.get("unmatched_observed_count", 0) or 0)
             for row in frame_rows),
            default=0,
        )
        extra = extra_evaluable and extra_count > 0
        if not eligible:
            if extra:
                values.append(0.0)
            continue
        missing = any(float(row.presence or 0.0) <= 0.0 for row in eligible)
        values.append(0.0 if missing or extra else 1.0)
    return _mean(values)


def _frame_macro_f1(rows: Sequence[Any]) -> float | None:
    grouped: dict[Any, list[Any]] = {}
    for row in rows:
        grouped.setdefault(_window(row), []).append(row)
    values: list[float] = []
    for frame_rows in grouped.values():
        eligible = [row for row in frame_rows if row.presence is not None]
        tp = sum(float(row.presence or 0.0) > 0.0 for row in eligible)
        fn = sum(row.state == "generation_failure" and float(row.presence or 0.0) <= 0.0 for row in eligible)
        fp = int(next((row.diagnostics.get("unmatched_observed_count", 0) for row in frame_rows if row.diagnostics.get("unmatched_observed_count") is not None), 0) or 0)
        score = _ratio(2 * tp, 2 * tp + fp + fn)
        if score is not None:
            values.append(score)
    return _mean(values)


def _identity_swap_rate(rows: Sequence[Any]) -> float | None:
    by_observed: dict[str, list[tuple[Any, str]]] = {}
    for row in rows:
        track_id = row.diagnostics.get("observed_track_id")
        if not row.observed_id or not track_id:
            continue
        identity = "group:" + str(row.diagnostics["ambiguity_group"]) if row.diagnostics.get("ambiguity_group") else row.reference_id
        by_observed.setdefault(str(track_id), []).append((row.diagnostics.get("window_ordinal", row.frame_index), identity))
    transitions = 0
    swaps = 0
    for values in by_observed.values():
        values.sort(key=lambda item: (-1 if item[0] is None else item[0]))
        for (_, previous), (_, current) in zip(values, values[1:]):
            transitions += 1
            swaps += previous != current
    return _ratio(swaps, transitions)


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _window(row: Any) -> Any:
    return row.diagnostics.get("window_id", row.frame_index)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall <= 0:
        return None
    return float(2.0 * precision * recall / (precision + recall))


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def aggregate_structure_quality(rows: Sequence[Any]) -> dict[str, Any]:
    """Report raw, prior-shrunk, and legacy coverage-adjusted evidence.

    Candidate instances include unobservable rows. Raw quality averages valid
    windows, then observed instances. The headline quality uses a neutral
    prior (mean 0.5, strength 2) per instance; the old coverage product remains
    available as a diagnostic. Missing rows stay unknown in per-row diagnostics.
    """
    dynamic = [row for row in rows if row.diagnostics.get("structure_candidate")]
    candidates = [row for row in rows if row.diagnostics.get("geometry_candidate")
                  or row.diagnostics.get("structure_candidate")]
    structure_by_instance: dict[str, list[float]] = {}
    quality_by_instance: dict[str, list[float]] = {}
    structure_requested = Counter(row.reference_id for row in dynamic)
    quality_requested = Counter(row.reference_id for row in candidates)
    n_structure_valid = n_quality_valid = 0
    for row in candidates:
        is_dynamic = bool(row.diagnostics.get("structure_candidate"))
        value = (row.input_structure if is_dynamic else row.visible_surface_geometry
                 if row.state == "scored" and row.diagnostics.get("geometry_eligible", True) else None)
        if value is None:
            continue
        if not math.isfinite(value) or not 0. <= value <= 1.:
            raise ValueError("structure/geometry quality must be finite and in [0, 1]")
        quality_by_instance.setdefault(row.reference_id, []).append(float(value))
        n_quality_valid += 1
        if is_dynamic:
            structure_by_instance.setdefault(row.reference_id, []).append(float(value))
            n_structure_valid += 1

    def macro(grouped: dict[str, list[float]]) -> float | None:
        return _mean([float(_mean(values)) for values in grouped.values()])

    def coverage_adjusted(grouped: dict[str, list[float]], requested: Counter) -> float | None:
        return _mean([sum(grouped.get(ref_id, [])) / count
                      for ref_id, count in requested.items()])

    prior_mean = 0.5
    prior_strength = 2.0

    def shrunk(grouped: dict[str, list[float]], requested: Counter) -> float | None:
        if not requested:
            return None
        values = [
            (sum(grouped.get(ref_id, [])) + prior_strength * prior_mean)
            / (len(grouped.get(ref_id, [])) + prior_strength)
            for ref_id in requested
        ]
        return _mean(values)

    quality_prior_only = sum(not quality_by_instance.get(ref_id) for ref_id in quality_requested)
    structure_prior_only = sum(not structure_by_instance.get(ref_id) for ref_id in structure_requested)

    return {
        "v3_input_structure_score_raw": macro(structure_by_instance),
        "v3_input_structure_score": shrunk(structure_by_instance, structure_requested),
        "v3_input_structure_score_coverage_adjusted": coverage_adjusted(structure_by_instance, structure_requested),
        "v3_structure_coverage": _ratio(n_structure_valid, len(dynamic)),
        "v3_structure_candidate_rows": len(dynamic),
        "v3_structure_valid_rows": n_structure_valid,
        "v3_structural_quality_score_raw": macro(quality_by_instance),
        "v3_structural_quality_score": shrunk(quality_by_instance, quality_requested),
        "v3_structural_quality_score_coverage_adjusted": coverage_adjusted(quality_by_instance, quality_requested),
        "v3_structural_quality_coverage": _ratio(n_quality_valid, len(candidates)),
        "v3_quality_prior_mean": prior_mean,
        "v3_quality_prior_strength": prior_strength,
        "v3_quality_prior_only_instances": quality_prior_only,
        "v3_structure_prior_only_instances": structure_prior_only,
    }


def aggregate_score_tracks(
    metrics: dict[str, Any], *, valid_geometry_frames: int,
    motion_valid: bool | None, config: Any,
) -> dict[str, Any]:
    """Ungated score with explicit fallback when the third component is absent."""
    config.validate()
    presence = metrics.get("presence_score", metrics.get("presence_f1"))
    identity = metrics.get("identity_score")
    structure_enabled = "v3_structural_quality_score" in metrics
    third_name = "structural_quality" if structure_enabled else "geometry"
    geometry = metrics.get("v3_structural_quality_score" if structure_enabled else "v3_reference_projective_score")
    values = [0.0 if presence is None else presence, 0.0 if identity is None else identity]
    weights = (.2, .8, 0.) if geometry is None else (
        config.presence_weight, config.identity_weight, config.geometry_weight)
    if geometry is not None:
        values.append(geometry)
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("quality components must be finite and in [0, 1]")
    candidate = 100.0 * math.prod(value ** weight for value, weight in zip(values, weights))
    return {
        "full_igm_diagnostic_score": candidate,
        "full_igm_score": candidate, "headline_score": candidate,
        "headline_eligible": True, "headline_eligibility_reason": None,
        "full_igm_diagnostic_eligible": True,
        "full_igm_diagnostic_reason": None, "provisional": True,
        "full_igm_score_mode": ("presence_identity_fallback" if geometry is None else
                                "presence_identity_structural_quality" if structure_enabled else
                                "presence_identity_geometry"),
        "effective_score_weights": dict(zip(("presence", "identity", third_name), weights)),
        "score_gates_enabled": False,
    }


def aggregate_hierarchy(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Macro-average cases → subcategories → domains, within one model/run.

    Each input is one independent case, with case_id, domain, subcategory and
    metrics. Duplicate case IDs are rejected so repeat frames/runs cannot
    silently increase a case's weight. Missing scores remain null; counts always
    refer to original cases, including at upper levels. Bootstrap resampling
    must use cases and recompute this hierarchy, never resample frame rows.
    """
    metric_names = ("core_2d_score", "full_igm_diagnostic_score", "full_igm_score")
    seen = set()
    for case in cases:
        for name in ("case_id", "domain", "subcategory"):
            if not isinstance(case.get(name), str) or not case[name].strip():
                raise ValueError(f"{name} must be a non-empty string")
        if case['case_id'] in seen:
            raise ValueError("duplicate case_id: aggregate repeated observations within each case first")
        seen.add(case['case_id'])
        if not isinstance(case.get('metrics'), dict):
            raise ValueError("case metrics must be a mapping")
        for name in metric_names:
            value = case['metrics'].get(name)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite or null")

    def summarize(raw, children):
        return {
            'metrics': {name: _mean([child['metrics'][name] for child in children
                                    if child['metrics'].get(name) is not None]) for name in metric_names},
            'n_cases': len(raw), 'bootstrap_unit': 'case',
            'missingness': {name: {
                'n_valid': sum(case['metrics'].get(name) is not None for case in raw),
                'n_missing': sum(case['metrics'].get(name) is None for case in raw),
            } for name in metric_names},
            'n_children': len(children),
            'missing_children': {name: sum(child['metrics'].get(name) is None for child in children)
                                 for name in metric_names},
        }

    case_scores = [{**{key: case[key] for key in ('case_id', 'domain', 'subcategory')},
                    **summarize([case], [case])} for case in cases]
    subcategories = []
    domains = []
    for domain in sorted({case['domain'] for case in cases}):
        domain_raw = [case for case in cases if case['domain'] == domain]
        domain_children = []
        for subcategory in sorted({case['subcategory'] for case in domain_raw}):
            raw = [case for case in domain_raw if case['subcategory'] == subcategory]
            summary = dict(domain=domain, subcategory=subcategory, **summarize(raw, raw))
            subcategories.append(summary)
            domain_children.append(summary)
        domains.append(dict(domain=domain, **summarize(domain_raw, domain_children)))
    return {'cases': case_scores, 'subcategories': subcategories, 'domains': domains,
            'overall': summarize(cases, domains), 'provisional': True,
            'aggregation_order': ['case', 'subcategory', 'domain', 'overall']}
