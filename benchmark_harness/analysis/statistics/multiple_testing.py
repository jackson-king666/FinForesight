"""Holm-Bonferroni family-wise error adjustment."""

from __future__ import annotations

from collections.abc import Mapping

from benchmark_harness.analysis.contracts import AnalysisError


def holm_bonferroni(p_values: Mapping[str, float]) -> dict[str, float]:
    if any(
        not isinstance(value, int | float) or isinstance(value, bool) or not 0 <= value <= 1
        for value in p_values.values()
    ):
        raise AnalysisError("p-values must be numbers in [0, 1].")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * float(p_value)))
        adjusted[name] = running
    return {name: adjusted[name] for name in sorted(adjusted)}
