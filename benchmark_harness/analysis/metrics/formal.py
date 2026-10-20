"""Formal-utility v1 mapping from frozen evaluator metrics to the v4 U scale.

Pure, versioned, analysis-only: these functions consume the metric dicts
emitted by ``reference_evaluation/engine.py`` (already materialized by the
host) and never call the frozen parser or evaluator.  Every per-category
utility returns ``(value, reason)``: ``value`` is ``None`` with a short
snake_case reason whenever the frozen input cannot support the formal
endpoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from benchmark_harness.analysis.contracts import AnalysisError

FORMAL_UTILITY_VERSION = "formal-utility/v1"

_RELATIVE_ERROR_MODE = "relative_error"
_ABSOLUTE_BASE_ZERO_MODE = "absolute_error_base_zero"


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{name} must be a mapping of frozen evaluator output.")
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def a1_formal_utility(frozen_metrics: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Map frozen A1 metrics to ``1 - 2*min(ape_30, 1)``.

    Only the 30-day window participates; pending 90/180/365-day windows never
    block the endpoint.  The window error is accepted only in the engine's
    ``relative_error`` mode; the base-zero absolute mode is not a percentage
    error.  ``monotonic_valid`` and ``range_hit`` stay raw views, not U.
    """

    metrics = _require_mapping(frozen_metrics, "frozen_metrics")
    by_window = metrics.get("by_window")
    if by_window is None:
        return None, "a1_window_30_unavailable"
    window = _require_mapping(by_window, "frozen_metrics.by_window").get("30")
    if window is None:
        return None, "a1_window_30_unavailable"
    entry = _require_mapping(window, "frozen_metrics.by_window.30")
    error = entry.get("target_price_error_abs")
    if error is None:
        return None, "a1_window_30_unavailable"
    error_value = _finite(error)
    if error_value is None:
        return None, "a1_error_not_finite"
    if error_value < 0.0:
        return None, "a1_error_negative"
    mode = entry.get("target_price_error_mode")
    if mode != _RELATIVE_ERROR_MODE:
        if mode == _ABSOLUTE_BASE_ZERO_MODE:
            return None, "a1_error_mode_absolute_base_zero"
        return None, "a1_error_mode_unknown"
    return 1.0 - 2.0 * min(error_value, 1.0), None


def a2_formal_utility(frozen_metrics: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Pass through frozen ``spearman_rho`` when the constituent set matches.

    The A2 variant (F/T/H) is carried by the caller, not by this mapping.
    """

    metrics = _require_mapping(frozen_metrics, "frozen_metrics")
    if metrics.get("exact_set_match") is not True:
        return None, "a2_constituent_set_mismatch"
    rho = _finite(metrics.get("spearman_rho"))
    if rho is None:
        return None, "a2_spearman_unavailable"
    if not -1.0 <= rho <= 1.0:
        return None, "a2_spearman_out_of_range"
    return rho, None


def b_probability(parsed: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Copy an explicit finite ``probability_up`` in [0, 1] from parsed B output.

    A missing probability is never synthesized from ``direction``.
    """

    parsed_mapping = _require_mapping(parsed, "parsed")
    value = parsed_mapping.get("probability_up")
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None, "b_probability_missing"
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return None, "b_probability_out_of_range"
    return probability, None


def b_formal_utility(frozen_metrics: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Map frozen B metrics to ``1 - 2*brier_score``."""

    metrics = _require_mapping(frozen_metrics, "frozen_metrics")
    brier = _finite(metrics.get("brier_score"))
    if brier is None:
        return None, "b_brier_unavailable"
    if not 0.0 <= brier <= 1.0:
        return None, "b_brier_out_of_range"
    return 1.0 - 2.0 * brier, None


def brier_skill_score_unclipped(brier_score: float) -> float:
    """Return ``1 - BS/0.25`` against the 0.25 baseline; negatives allowed."""

    brier = _finite(brier_score)
    if brier is None:
        raise AnalysisError("brier_score must be a finite number.")
    if not 0.0 <= brier <= 1.0:
        raise AnalysisError("brier_score must lie in [0, 1].")
    return 1.0 - brier / 0.25


def c_formal_utility(frozen_metrics: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Map frozen C metrics to ``1 - 2*min(mape, 1)``.

    A ``None`` MAPE from the frozen zero-denominator rule stays unavailable.
    """

    metrics = _require_mapping(frozen_metrics, "frozen_metrics")
    mape = _finite(metrics.get("mape"))
    if mape is None:
        return None, "c_mape_unavailable"
    if mape < 0.0:
        return None, "c_mape_negative"
    return 1.0 - 2.0 * min(mape, 1.0), None


def e_formal_utility(frozen_metrics: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Map frozen ``exact_match_0_2pct`` to +1/-1.

    The tolerance decision stays with the frozen engine (relative error
    < 0.002; zero-answer absolute error < 1e-9); this mapping never
    recomputes it.
    """

    metrics = _require_mapping(frozen_metrics, "frozen_metrics")
    match = metrics.get("exact_match_0_2pct")
    if not isinstance(match, bool):
        return None, "e_exact_match_unavailable"
    return (1.0 if match else -1.0), None


def operational_utility(raw_utility: float | None, failure_reason: str | None) -> float | None:
    """Compute the operational utility field without touching the raw value.

    Ordinary failures (invalid output, timeout, model failure) charge -1.0 in
    the operational field only; a present raw utility always wins, and a
    non-failure missing value stays ``None``.
    """

    if raw_utility is not None:
        value = _finite(raw_utility)
        if value is None:
            raise AnalysisError("raw_utility must be a finite number when present.")
        return value
    if failure_reason is not None:
        return -1.0
    return None


__all__ = [
    "FORMAL_UTILITY_VERSION",
    "a1_formal_utility",
    "a2_formal_utility",
    "b_formal_utility",
    "b_probability",
    "brier_skill_score_unclipped",
    "c_formal_utility",
    "e_formal_utility",
    "operational_utility",
]
