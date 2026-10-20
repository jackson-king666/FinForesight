"""Seeded cluster bootstrap with common draws across paired inputs.

Self-contained analysis-side counterpart of the experiment/v3 paired event
cluster bootstrap: whole clusters (event/entity or calendar blocks) are
resampled with a derived seed so identical inputs give identical outputs.
The paired variant draws one cluster-index vector per resample and applies
it to both arms, keeping each pair under shared draws; because the
statistic is the mean of per-cluster mean differences, it is computed
directly on per-cluster difference means.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

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


def _derive_seed(seed: int | str) -> int:
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    if isinstance(seed, str) and seed.strip():
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)
    raise AnalysisError("bootstrap seed must be an integer or a non-empty string.")


def _check_resampling(resamples: int, confidence: float) -> None:
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise AnalysisError("bootstrap resamples must be a positive integer.")
    if not 0.0 < confidence < 1.0:
        raise AnalysisError("bootstrap confidence must be between 0 and 1.")


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise AnalysisError(f"{name} must be a finite number.")
    return float(value)


def _cluster_key(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnalysisError(f"{name} must be a non-empty string.")
    return value.strip()


def _as_pair(item: Any, index: int) -> tuple[Any, Any]:
    if isinstance(item, tuple | list) and len(item) == 2:
        return item[0], item[1]
    raise AnalysisError(f"cluster value {index} must be a (cluster_key, value) pair.")


def make_cluster_key(*parts: str) -> str:
    """Join non-empty string parts into one canonical cluster key."""

    if not parts:
        raise AnalysisError("cluster key requires at least one part.")
    return "|".join(
        _cluster_key(part, f"cluster key part {index}") for index, part in enumerate(parts)
    )


def _resample_draws(
    cluster_means: Mapping[str, float],
    clusters: Sequence[str],
    *,
    resamples: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    return [
        mean(cluster_means[clusters[rng.randrange(len(clusters))]] for _ in clusters)
        for _ in range(resamples)
    ]


def cluster_bootstrap(
    cluster_values: Mapping[str, float] | Sequence[tuple[str, float]],
    *,
    resamples: int = 10_000,
    seed: int | str = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap the mean of per-cluster means with seeded cluster resampling.

    ``cluster_values`` is either a mapping of cluster key to value or a
    sequence of ``(cluster_key, value)`` pairs; repeated keys are averaged
    within the cluster first.  String seeds derive an integer from the first
    16 hex digits of the UTF-8 SHA-256 digest.  Returns the estimate,
    percentile interval, cluster count, resample count, derived integer
    seed, and confidence.
    """

    _check_resampling(resamples, confidence)
    if isinstance(cluster_values, Mapping):
        pairs: list[tuple[Any, Any]] = list(cluster_values.items())
    elif isinstance(cluster_values, Sequence) and not isinstance(cluster_values, str | bytes):
        pairs = [_as_pair(item, index) for index, item in enumerate(cluster_values)]
    else:
        raise AnalysisError("cluster_values must be a mapping or a sequence of pairs.")
    grouped: dict[str, list[float]] = defaultdict(list)
    for index, (key, value) in enumerate(pairs):
        grouped[_cluster_key(key, f"cluster value {index} key")].append(
            _finite(value, f"cluster value {index} value")
        )
    clusters = sorted(grouped)
    if not clusters:
        raise AnalysisError("cluster bootstrap requires at least one cluster.")
    cluster_means = {cluster: mean(grouped[cluster]) for cluster in clusters}
    derived_seed = _derive_seed(seed)
    draws = _resample_draws(cluster_means, clusters, resamples=resamples, seed=derived_seed)
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": mean(cluster_means[cluster] for cluster in clusters),
        "ci_lower": _quantile(draws, tail),
        "ci_upper": _quantile(draws, 1.0 - tail),
        "cluster_count": len(clusters),
        "resamples": resamples,
        "seed": derived_seed,
        "confidence": confidence,
    }


def paired_cluster_bootstrap(
    values_a: Sequence[float],
    values_b: Sequence[float],
    cluster_keys: Sequence[str],
    *,
    resamples: int = 10_000,
    seed: int | str = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap per-cluster mean paired differences under shared draws.

    Rows are paired positionally across ``values_a``/``values_b`` and grouped
    by ``cluster_keys``; one cluster-index vector is drawn per resample and
    applied to both arms, so each matched pair is always resampled together.
    Returns the ``cluster_bootstrap`` fields plus ``pair_count``.
    """

    _check_resampling(resamples, confidence)
    if len(values_a) != len(values_b) or len(values_a) != len(cluster_keys):
        raise AnalysisError("paired cluster bootstrap requires equally sized inputs.")
    if len(cluster_keys) == 0:
        raise AnalysisError("paired cluster bootstrap requires at least one paired row.")
    differences: list[tuple[str, float]] = []
    for index, (left, right, key) in enumerate(zip(values_a, values_b, cluster_keys, strict=True)):
        difference = _finite(left, f"values_a[{index}]") - _finite(right, f"values_b[{index}]")
        differences.append((key, difference))
    result = cluster_bootstrap(differences, resamples=resamples, seed=seed, confidence=confidence)
    result["pair_count"] = len(differences)
    return result


cluster_bootstrap_estimates = cluster_bootstrap

__all__ = [
    "cluster_bootstrap",
    "cluster_bootstrap_estimates",
    "make_cluster_key",
    "paired_cluster_bootstrap",
]
