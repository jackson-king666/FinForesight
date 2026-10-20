"""Fixed-order task/trial/category aggregation for paper analysis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from statistics import mean
from typing import Any

from benchmark_harness.analysis.contracts import AnalysisRow, coerce_analysis_rows
from benchmark_harness.analysis.metrics.catalog import MetricCatalog
from benchmark_harness.analysis.metrics.normalize import metric_observation


def _average(values: Iterable[float]) -> float | None:
    values = list(values)
    return mean(values) if values else None


def aggregate_metrics(
    rows: Iterable[AnalysisRow | dict[str, Any]],
    catalog: MetricCatalog,
    *,
    metric_name: str,
) -> dict[str, Any]:
    """Aggregate trial means, then category macro means, with exclusions explicit."""

    spec = catalog.metric(metric_name)
    normalized_rows = coerce_analysis_rows(rows)
    observations: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in normalized_rows:
        if not row.comparable:
            exclusions.append(
                {"task_id": row.task_id, "reason": row.failure_reason or "non_comparable"}
            )
            continue
        if not row.analysis_eligible:
            exclusions.append({"task_id": row.task_id, "reason": "temporal_or_control_exclusion"})
            continue
        if row.category in {"D", "E"}:
            exclusions.append({"task_id": row.task_id, "reason": "control_category_exclusion"})
            continue
        value, reason = metric_observation(row.to_dict(), spec)
        if reason is not None or value is None:
            exclusions.append({"task_id": row.task_id, "reason": reason or "missing_metric"})
            continue
        observations.append(
            {
                "system_id": row.system_id,
                "task_id": row.task_id,
                "category": row.category,
                "temporal_band": row.temporal_band,
                "retrieval_regime": row.retrieval_regime,
                "trial": row.trial,
                "value": value,
                "ranking_eligible": row.ranking_eligible,
            }
        )

    trial_groups: dict[tuple[str, str, str, str | None, str], list[float]] = defaultdict(list)
    for observation in observations:
        trial_groups[
            (
                observation["system_id"],
                observation["task_id"],
                observation["retrieval_regime"],
                observation["temporal_band"],
                observation["category"],
            )
        ].append(observation["value"])
    task_means = [
        {
            "system_id": key[0],
            "task_id": key[1],
            "retrieval_regime": key[2],
            "temporal_band": key[3],
            "category": key[4],
            "value": mean(values),
        }
        for key, values in sorted(trial_groups.items())
    ]

    category_groups: dict[tuple[str, str, str | None, str], list[float]] = defaultdict(list)
    for row in task_means:
        category_groups[
            (
                row["system_id"],
                row["retrieval_regime"],
                row["temporal_band"],
                row["category"],
            )
        ].append(row["value"])
    category_means = [
        {
            "system_id": key[0],
            "retrieval_regime": key[1],
            "temporal_band": key[2],
            "category": key[3],
            "value": mean(values),
        }
        for key, values in sorted(
            category_groups.items(),
            key=lambda item: (item[0][0], item[0][1], item[0][2] or "", item[0][3]),
        )
    ]
    system_groups: dict[tuple[str, str, str | None], list[float]] = defaultdict(list)
    for row in category_means:
        system_groups[(row["system_id"], row["retrieval_regime"], row["temporal_band"])].append(
            row["value"]
        )
    system_values: dict[tuple[str, str, str | None], float] = {
        key: mean(values) for key, values in sorted(system_groups.items())
    }
    systems: dict[str, dict[str, Any]] = {}
    for (system_id, regime, band), value in system_values.items():
        entry = systems.setdefault(
            system_id, {"system_id": system_id, "by_regime": {}, "by_band": {}}
        )
        entry["by_regime"].setdefault(regime, {})[band or "UNASSIGNED"] = value
        entry["by_band"].setdefault(band or "UNASSIGNED", {})[regime] = value
    return {
        "schema_version": "analysis-aggregate/v1",
        "metric": metric_name,
        "catalog_sha256": catalog.hash,
        "attempted_rows": len(normalized_rows),
        "eligible_observations": len(observations),
        "exclusions": sorted(exclusions, key=lambda item: (item["task_id"], item["reason"])),
        "category_means": category_means,
        "systems": [systems[key] for key in sorted(systems)],
        "task_means": task_means,
    }
