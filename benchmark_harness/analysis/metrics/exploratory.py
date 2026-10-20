"""Metric-catalog v2 utility functions for the non-ranking tranche."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from statistics import mean
from typing import Any


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("value is not finite")
    return float(value)


def _spearman(predicted: list[Any], actual: list[Any]) -> float:
    if not predicted or set(predicted) != set(actual) or len(predicted) != len(actual):
        raise ValueError("ranking lists must contain the same unique items")
    actual_rank = {value: index for index, value in enumerate(actual)}
    predicted_rank = {value: index for index, value in enumerate(predicted)}
    n = len(predicted)
    if n < 2:
        return 0.0
    d2 = sum((predicted_rank[value] - actual_rank[value]) ** 2 for value in predicted)
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def exploratory_task_utility(
    category: str,
    prediction: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    *,
    tolerance: float = 0.002,
) -> tuple[float | None, str | None]:
    """Compute the frozen per-task utility and an explicit missing reason.

    The evaluator remains host-owned; this helper accepts only a prediction
    object and a ground-truth object supplied by the analysis process.
    """

    try:
        category = str(category).upper()
        if category == "A1":
            cutoff = _number(ground_truth["cutoff_price"])
            actual = ground_truth["actual_prices"]
            predicted = prediction.get("prices", prediction.get("predicted_prices"))
            if not isinstance(predicted, Mapping):
                raise ValueError("A1 prediction must contain prices")
            errors = []
            for window in ("30", "90", "180", "365"):
                errors.append(
                    abs(_number(predicted[window]) - _number(actual[window])) / abs(cutoff)
                )
            return _clip(1.0 - mean(errors)), None
        if category == "A2":
            predicted = prediction.get("ranking", prediction.get("predicted_ranking"))
            actual = ground_truth["actual_ranking"]
            if not isinstance(predicted, list) or not isinstance(actual, list):
                raise ValueError("A2 prediction must contain ranking")
            return _spearman(predicted, actual), None
        if category == "B":
            probability = prediction.get("p_up", prediction.get("probability_up"))
            if probability is None:
                direction = prediction.get("direction")
                probability = 1.0 if direction == "up" else 0.0 if direction == "down" else None
            if probability is None:
                raise ValueError("B prediction must contain p_up or direction")
            probability = _number(probability)
            if not 0.0 <= probability <= 1.0:
                raise ValueError("B probability must be between 0 and 1")
            actual = _number(ground_truth["actual_up"])
            return _clip(1.0 - ((probability - actual) ** 2) / 0.25), None
        if category == "C":
            predicted = prediction.get("value", prediction.get("predicted_value"))
            actual = _number(ground_truth["future_value"])
            if predicted is None or actual == 0:
                raise ValueError("C prediction is missing or has zero denominator")
            return _clip(1.0 - abs(_number(predicted) - actual) / abs(actual)), None
        if category == "D":
            expected = ground_truth.get("logic_direction")
            observed = prediction.get("logic_direction", prediction.get("direction"))
            if expected is None or observed is None:
                raise ValueError("D control prediction is missing")
            return (1.0 if str(expected) == str(observed) else -1.0), None
        if category == "E":
            expected = _number(ground_truth["correct_answer"])
            observed = _number(prediction.get("answer"))
            return (1.0 if abs(observed - expected) <= tolerance else -1.0), None
        raise ValueError(f"unsupported category {category!r}")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return None, str(exc)


def aggregate_category_utility(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Average non-missing utilities with explicit category coverage."""

    values: dict[str, list[float]] = defaultdict(list)
    missing: dict[str, int] = defaultdict(int)
    for row in rows:
        category = str(row.get("category", "")).upper()
        value = row.get("utility")
        if isinstance(value, int | float) and not isinstance(value, bool):
            values[category].append(float(value))
        else:
            missing[category] += 1
    category_means = {category: mean(items) for category, items in sorted(values.items()) if items}
    weights = {category: 0.25 for category in ("A1", "A2", "B", "C")}
    weighted = [
        weights[category] * category_means[category]
        for category in weights
        if category in category_means
    ]
    return {
        "category_means": category_means,
        "missing_counts": dict(sorted(missing.items())),
        "predictive_utility": mean(weighted) if len(weighted) == 4 else None,
        "predictive_categories_present": sorted(category_means.keys()),
        "category_weights": weights,
    }


def c2_status(
    system_ids: Iterable[str], *, has_t2: bool = False, resolved_t3: bool = False
) -> dict[str, Any]:
    """Return the non-ranking C2 gate required by the exploratory tranche."""

    ids = tuple(dict.fromkeys(str(value) for value in system_ids))
    reasons = []
    if len(ids) < 5:
        reasons.append("system_count_lt_5")
    if not has_t2:
        reasons.append("no_T2")
    if not resolved_t3:
        reasons.append("no_resolved_T3")
    return {
        "status": "exploratory_only",
        "ranking_eligible": False,
        "system_count": len(ids),
        "reasons": reasons,
        "correlations": None,
    }


def outcome_cluster_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed_namespace: str = "20260817",
    resamples: int = 10_000,
) -> dict[str, Any]:
    """Compute a paired cluster bootstrap interval without score selection."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        cluster = str(row.get("cluster_key", row.get("task_id", "")))
        value = row.get("difference", row.get("utility"))
        if cluster and isinstance(value, int | float) and not isinstance(value, bool):
            grouped[cluster].append(float(value))
    clusters = sorted(grouped)
    observed = mean(mean(grouped[key]) for key in clusters) if clusters else None
    if not clusters:
        return {
            "estimate": None,
            "lower": None,
            "upper": None,
            "clusters": 0,
            "resamples": resamples,
        }
    seed = int(hashlib.sha256(f"{seed_namespace}|cluster-bootstrap".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        samples.append(mean(mean(grouped[key]) for key in sampled))
    samples.sort()
    low = samples[max(0, int(0.025 * len(samples)) - 1)]
    high = samples[min(len(samples) - 1, int(0.975 * len(samples)))]
    return {
        "estimate": observed,
        "lower": low,
        "upper": high,
        "clusters": len(clusters),
        "resamples": resamples,
        "seed": seed,
    }


__all__ = [
    "aggregate_category_utility",
    "c2_status",
    "exploratory_task_utility",
    "outcome_cluster_bootstrap",
]
