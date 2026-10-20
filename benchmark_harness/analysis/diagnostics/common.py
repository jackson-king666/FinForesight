"""Shared diagnostic counting over observable row metadata."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def summarize_diagnostics(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        diagnostics = row.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            continue
        for dimension in ("retrieval", "control_flow", "calculation", "format_reliability"):
            value = diagnostics.get(dimension)
            if isinstance(value, int) and value > 0:
                counts[dimension]["count"] += value
    return {
        dimension: {name: value for name, value in sorted(values.items())}
        for dimension, values in sorted(counts.items())
    }
