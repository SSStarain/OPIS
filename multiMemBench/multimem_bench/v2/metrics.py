"""Aggregation helpers for macro instance-level benchmark reports."""

from __future__ import annotations

import math
from typing import Iterable


def mean(values: Iterable[float | None]) -> float | None:
    items = [float(value) for value in values if value is not None]
    return sum(items) / len(items) if items else None


def geometric_mean(values: Iterable[float | None]) -> float | None:
    items = [max(0.0, min(1.0, float(value))) for value in values if value is not None]
    if not items:
        return None
    if any(value <= 0.0 for value in items):
        return 0.0
    return math.exp(sum(math.log(value) for value in items) / len(items))


def macro_instance_mean(
    rows: Iterable[tuple[str, float | None]],
) -> float | None:
    grouped: dict[str, list[float]] = {}
    for instance_id, value in rows:
        if value is not None:
            grouped.setdefault(instance_id, []).append(float(value))
    return mean(mean(values) for values in grouped.values())
