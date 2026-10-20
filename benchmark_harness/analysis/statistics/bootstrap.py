"""Seeded paired bootstrap without model/search dependencies."""

from __future__ import annotations

import random
from collections.abc import Sequence
from statistics import mean

from benchmark_harness.analysis.contracts import AnalysisError


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AnalysisError("cannot compute a quantile of an empty sample.")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def paired_bootstrap(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Return the distribution of mean(a-b) with deterministic resampling."""

    if len(values_a) != len(values_b) or not values_a:
        raise AnalysisError("paired bootstrap requires equally sized non-empty samples.")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise AnalysisError("bootstrap resamples must be a positive integer.")
    differences = [float(a) - float(b) for a, b in zip(values_a, values_b, strict=True)]
    rng = random.Random(seed)
    draws = [
        mean(differences[rng.randrange(len(differences))] for _ in differences)
        for _ in range(resamples)
    ]
    observed = mean(differences)
    return {
        "n": len(differences),
        "resamples": resamples,
        "seed": seed,
        "mean_difference": observed,
        "ci_lower": _quantile(draws, 0.025),
        "ci_upper": _quantile(draws, 0.975),
        "probability_positive": sum(draw > 0 for draw in draws) / resamples,
    }
