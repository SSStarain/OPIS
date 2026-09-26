"""Small dependency-free numerical algorithms used by the V2 evaluator."""

from __future__ import annotations

import math
from typing import Sequence


def linear_sum_assignment(cost_matrix: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Return the minimum-cost rectangular assignment using Hungarian potentials."""

    costs = [[float(value) for value in row] for row in cost_matrix]
    if not costs:
        return []
    width = len(costs[0])
    if width == 0:
        return []
    if any(len(row) != width for row in costs):
        raise ValueError("cost matrix must be rectangular")

    transposed = len(costs) > width
    if transposed:
        costs = [list(row) for row in zip(*costs)]

    rows = len(costs)
    cols = len(costs[0])
    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    p = [0] * (cols + 1)
    way = [0] * (cols + 1)

    for row_index in range(1, rows + 1):
        p[0] = row_index
        min_values = [math.inf] * (cols + 1)
        used = [False] * (cols + 1)
        col0 = 0
        while True:
            used[col0] = True
            active_row = p[col0]
            delta = math.inf
            col1 = 0
            for col in range(1, cols + 1):
                if used[col]:
                    continue
                current = costs[active_row - 1][col - 1] - u[active_row] - v[col]
                if current < min_values[col]:
                    min_values[col] = current
                    way[col] = col0
                if min_values[col] < delta:
                    delta = min_values[col]
                    col1 = col
            for col in range(cols + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    min_values[col] -= delta
            col0 = col1
            if p[col0] == 0:
                break
        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break

    assignment = [(p[col] - 1, col - 1) for col in range(1, cols + 1) if p[col]]
    if transposed:
        assignment = [(col, row) for row, col in assignment]
    return sorted(assignment)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
