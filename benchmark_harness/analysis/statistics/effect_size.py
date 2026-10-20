"""Paired effect-size calculations."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from statistics import mean

from benchmark_harness.analysis.contracts import AnalysisError


def paired_standardized_effect(
    values_a: Sequence[float], values_b: Sequence[float]
) -> float | None:
    if len(values_a) != len(values_b) or not values_a:
        raise AnalysisError("paired effect size requires equally sized non-empty samples.")
    differences = [float(a) - float(b) for a, b in zip(values_a, values_b, strict=True)]
    center = mean(differences)
    if len(differences) < 2:
        return None
    variance = sum((value - center) ** 2 for value in differences) / (len(differences) - 1)
    deviation = sqrt(variance)
    return center / deviation if deviation else None
