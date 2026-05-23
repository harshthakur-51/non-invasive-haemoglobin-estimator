"""Simple linear ridge model used for an interpretable public starter."""

from __future__ import annotations

import math


def _solve(system: list[list[float]], values: list[float]) -> list[float]:
    rows = [row[:] + [value] for row, value in zip(system, values)]
    for pivot in range(len(rows)):
        best = max(range(pivot, len(rows)), key=lambda index: abs(rows[index][pivot]))
        rows[pivot], rows[best] = rows[best], rows[pivot]
        factor = rows[pivot][pivot]
        rows[pivot] = [value / factor for value in rows[pivot]]
        for index in range(len(rows)):
            if index == pivot:
                continue
            scale = rows[index][pivot]
            rows[index] = [value - scale * anchor for value, anchor in zip(rows[index], rows[pivot])]
    return [row[-1] for row in rows]


def fit(rows: list[list[float]], targets: list[float], penalty: float = 0.03) -> list[float]:
    inputs = [[1.0] + row for row in rows]
    width = len(inputs[0])
    gram = [[0.0] * width for _ in range(width)]
    projection = [0.0] * width
    for row, target in zip(inputs, targets):
        for i in range(width):
            projection[i] += row[i] * target
            for j in range(width):
                gram[i][j] += row[i] * row[j]
    for index in range(1, width):
        gram[index][index] += penalty
    return _solve(gram, projection)


def predict(weights: list[float], rows: list[list[float]]) -> list[float]:
    return [weights[0] + sum(a * b for a, b in zip(row, weights[1:])) for row in rows]


def mae(actual: list[float], predicted: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(actual, predicted)) / len(actual)


def rmse(actual: list[float], predicted: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(actual, predicted)) / len(actual))
