"""T2-to-T3 system-ranking proxy with the exploratory gate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmark_harness.analysis.statistics.ranking import (
    kendall_tau,
    pearson_correlation,
    spearman_correlation,
)


def c2_proxy(
    t2_scores: Mapping[str, float],
    t3_scores: Mapping[str, float],
    *,
    resolved_t3_systems: set[str] | None = None,
    bootstrap: Mapping[str, Any] | None = None,
    minimum_systems: int = 5,
) -> dict[str, Any]:
    """Compare system-level T2/T3 scores without treating tasks as systems."""

    systems = sorted(set(t2_scores) & set(t3_scores))
    if resolved_t3_systems is not None:
        systems = [system for system in systems if system in resolved_t3_systems]
    t2 = [float(t2_scores[system]) for system in systems]
    t3 = [float(t3_scores[system]) for system in systems]
    status = "confirmatory" if len(systems) >= minimum_systems else "exploratory_only"
    result: dict[str, Any] = {
        "status": status,
        "matched_systems": systems,
        "matched_n": len(systems),
        "spearman": spearman_correlation(t2, t3) if len(systems) >= 2 else None,
        "pearson": pearson_correlation(t2, t3) if len(systems) >= 2 else None,
        "kendall": kendall_tau(t2, t3) if len(systems) >= 2 else None,
        "bootstrap": dict(bootstrap or {}),
    }
    if len(systems) < minimum_systems:
        result["gate_reason"] = f"requires at least {minimum_systems} paper-eligible systems"
    return result
