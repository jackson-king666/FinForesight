"""Participant-independent metric normalization."""

from __future__ import annotations

from typing import Any

from benchmark_harness.analysis.contracts import AnalysisError
from benchmark_harness.analysis.metrics.catalog import MetricSpec


def normalize_metric(value: float | int | None, spec: MetricSpec) -> float | None:
    """Return a fixed-normalizer value, preserving missing values as missing."""

    if value is None:
        return None
    normalizer = float(spec.normalizer["value"])
    normalized = float(value) / normalizer
    if spec.direction == "minimize":
        normalized = 1.0 - normalized
    if spec.cap is not None:
        normalized = max(-float(spec.cap), min(float(spec.cap), normalized))
    return normalized


def metric_observation(row: dict[str, Any], spec: MetricSpec) -> tuple[float | None, str | None]:
    try:
        raw = spec.raw_value(row)
    except AnalysisError:
        raise
    if raw is None:
        return None, f"missing metric: {spec.name}"
    return normalize_metric(raw, spec), None
