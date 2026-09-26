"""Global partial one-to-one matching for V3 object observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import math

from multimem_bench.schema import ObjectSignature, ObservedObject
from multimem_bench.similarity import appearance_similarity, cosine_similarity
from multimem_bench.v2.math_utils import linear_sum_assignment

from .config import V3EvaluationConfig


@dataclass(frozen=True)
class V3MatchResult:
    pairs: dict[str, str]
    unmatched_reference_ids: tuple[str, ...]
    unmatched_observed_ids: tuple[str, ...]
    scores: dict[tuple[str, str], float]
    margins: dict[str, float]
    ambiguous_reference_ids: tuple[str, ...]


def solve_partial_matching(
    references: Mapping[str, ObjectSignature],
    observations: Sequence[ObservedObject],
    config: V3EvaluationConfig,
) -> V3MatchResult:
    config.validate()
    reference_items = sorted(references.items())
    observed_items = sorted(observations, key=lambda item: item.observed_id)
    if any(key != ref.object_id for key, ref in reference_items):
        raise ValueError("reference mapping keys must equal object_id")
    if len({item.observed_id for item in observed_items}) != len(observed_items):
        raise ValueError("observed_id must be unique within a window")
    if not reference_items:
        return V3MatchResult({}, (), tuple(item.observed_id for item in observed_items), {}, {}, ())
    if not observed_items:
        return V3MatchResult({}, tuple(item[0] for item in reference_items), (), {}, {}, ())

    scores: dict[tuple[str, str], float] = {}
    matrix: list[list[float]] = []
    null_cost = 1.0 - float(config.min_identity_similarity)
    for reference_id, reference in reference_items:
        row_scores = []
        for observed in observed_items:
            if any(vector is not None and any(not math.isfinite(v) for v in vector)
                   for vector in (reference.embedding, observed.embedding)):
                scores[(reference_id, observed.observed_id)] = 0.0
                row_scores.append(1.0)
                continue
            score, evidence = appearance_similarity(
                observed,
                reference,
                include_attribute_similarity=True,
                include_category_similarity=True,
            )
            embedding_cosine = _raw_cosine(reference.embedding, observed.embedding)
            if embedding_cosine is not None and embedding_cosine < config.min_embedding_cosine_similarity:
                score = 0.0
            elif embedding_cosine is None and evidence.get("attribute_similarity") is None:
                # Category alone is insufficient to claim an identity match.
                score = 0.0
            score = float(score) if math.isfinite(score) else 0.0
            scores[(reference_id, observed.observed_id)] = score
            row_scores.append(1.0 - score)
        row_scores.extend([null_cost] * len(reference_items))
        matrix.append(row_scores)

    assignment = linear_sum_assignment(matrix)
    pairs: dict[str, str] = {}
    assigned_observed: set[str] = set()
    for row, column in assignment:
        reference_id = reference_items[row][0]
        if column >= len(observed_items):
            continue
        observed_id = observed_items[column].observed_id
        score = scores[(reference_id, observed_id)]
        if score < config.min_identity_similarity:
            continue
        pairs[reference_id] = observed_id
        assigned_observed.add(observed_id)

    margins: dict[str, float] = {}
    ambiguous: list[str] = []
    for reference_id, _ in reference_items:
        ranked = sorted(
            (scores[(reference_id, item.observed_id)] for item in observed_items),
            reverse=True,
        )
        assigned = pairs.get(reference_id)
        chosen = scores[(reference_id, assigned)] if assigned else config.min_identity_similarity
        alternatives = [scores[(reference_id, item.observed_id)] for item in observed_items if item.observed_id != assigned]
        margin = float(chosen - max(alternatives + [config.min_identity_similarity]))
        margins[reference_id] = margin
        if margin < config.ambiguity_margin_threshold:
            ambiguous.append(reference_id)

    unmatched_references = tuple(item[0] for item in reference_items if item[0] not in pairs)
    unmatched_observed = tuple(item.observed_id for item in observed_items if item.observed_id not in assigned_observed)
    return V3MatchResult(
        pairs=pairs,
        unmatched_reference_ids=unmatched_references,
        unmatched_observed_ids=unmatched_observed,
        scores=scores,
        margins=margins,
        ambiguous_reference_ids=tuple(ambiguous),
    )


def _raw_cosine(first: list[float] | None, second: list[float] | None) -> float | None:
    if not first or not second or len(first) != len(second):
        return None
    value = cosine_similarity(first, second)
    if value is None:
        return None
    return float(2.0 * value - 1.0)
