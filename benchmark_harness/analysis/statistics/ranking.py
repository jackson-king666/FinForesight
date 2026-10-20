"""System-ranking helpers with explicit tie handling."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import sqrt
from statistics import mean

from benchmark_harness.analysis.contracts import AnalysisError


def _validate_pair(x: Sequence[float], y: Sequence[float]) -> None:
    if len(x) != len(y) or len(x) < 2:
        raise AnalysisError("correlation requires equally sized samples with at least two values.")


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    _validate_pair(x, y)
    x_mean, y_mean = mean(x), mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    x_dev = sqrt(sum((a - x_mean) ** 2 for a in x))
    y_dev = sqrt(sum((b - y_mean) ** 2 for b in y))
    return numerator / (x_dev * y_dev) if x_dev and y_dev else None


def _rank(values: Sequence[float]) -> list[float]:
    positions: dict[float, list[int]] = defaultdict(list)
    for index, value in enumerate(sorted(values)):
        positions[value].append(index)
    return [mean(positions[value]) for value in values]


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    _validate_pair(x, y)
    return pearson_correlation(_rank(x), _rank(y))


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float | None:
    _validate_pair(x, y)
    concordant = discordant = comparable = 0
    for index in range(len(x)):
        for other in range(index + 1, len(x)):
            dx = x[index] - x[other]
            dy = y[index] - y[other]
            if dx == 0 or dy == 0:
                continue
            comparable += 1
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / comparable if comparable else None


def pairwise_win_matrix(scores: Mapping[str, float]) -> dict[str, dict[str, int]]:
    systems = sorted(scores)
    return {
        left: {
            right: (
                1 if scores[left] > scores[right] else 0 if scores[left] == scores[right] else -1
            )
            for right in systems
        }
        for left in systems
    }
