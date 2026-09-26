"""Appearance and attribute similarity helpers."""

from __future__ import annotations

import math
from typing import Any

from multimem_bench.schema import ObjectSignature, ObservedObject


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def cosine_similarity(first: list[float] | None, second: list[float] | None) -> float | None:
    if not first or not second or len(first) != len(second):
        return None
    dot = sum(float(a) * float(b) for a, b in zip(first, second))
    na = math.sqrt(sum(float(a) * float(a) for a in first))
    nb = math.sqrt(sum(float(b) * float(b) for b in second))
    if na <= 1e-12 or nb <= 1e-12:
        return None
    return clamp01((dot / (na * nb) + 1.0) / 2.0)


def attribute_similarity(
    observed: dict[str, Any],
    reference: dict[str, Any],
    *,
    keys: tuple[str, ...] = ("color", "shape", "texture", "mark"),
) -> float | None:
    scores: list[float] = []
    for key in keys:
        if key not in observed or key not in reference:
            continue
        obs_val = observed[key]
        ref_val = reference[key]
        if isinstance(obs_val, (int, float)) and isinstance(ref_val, (int, float)):
            denom = max(abs(float(ref_val)), 1.0)
            scores.append(clamp01(1.0 - abs(float(obs_val) - float(ref_val)) / denom))
        else:
            scores.append(1.0 if str(obs_val).lower() == str(ref_val).lower() else 0.0)
    if not scores:
        return None
    return sum(scores) / len(scores)


def category_similarity(observed: ObservedObject, reference: ObjectSignature) -> float:
    if not observed.category or not reference.category:
        return 0.5
    return 1.0 if observed.category.lower() == reference.category.lower() else 0.0


def appearance_similarity(
    observed: ObservedObject,
    reference: ObjectSignature,
    *,
    include_attribute_similarity: bool = False,
    include_category_similarity: bool = False,
) -> tuple[float, dict[str, float | None]]:
    """Return appearance similarity, using embedding alone by default."""

    emb = cosine_similarity(observed.embedding, reference.embedding)
    attr = (
        attribute_similarity(observed.attributes, reference.attributes)
        if include_attribute_similarity
        else None
    )
    cat = category_similarity(observed, reference) if include_category_similarity else None

    weighted_sum = 0.0
    weight_total = 0.0
    if emb is not None:
        weighted_sum += 0.65 * emb
        weight_total += 0.65
    if attr is not None:
        weighted_sum += 0.25 * attr
        weight_total += 0.25
    if cat is not None:
        weighted_sum += 0.10 * cat
        weight_total += 0.10

    score = weighted_sum / weight_total if weight_total else 0.0
    return clamp01(score), {
        "embedding_similarity": emb,
        "attribute_similarity": attr,
        "category_similarity": cat,
    }
