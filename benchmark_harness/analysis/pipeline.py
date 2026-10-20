"""Offline C1/C2/C3 analysis over immutable, already-scored rows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from statistics import mean
from typing import Any

from benchmark_harness.analysis.contracts import AnalysisError, AnalysisRow, coerce_analysis_rows
from benchmark_harness.analysis.metrics.catalog import MetricCatalog, MetricSpec
from benchmark_harness.analysis.metrics.normalize import metric_observation
from benchmark_harness.analysis.statistics.bootstrap import paired_bootstrap
from benchmark_harness.analysis.statistics.effect_size import paired_standardized_effect
from benchmark_harness.analysis.statistics.multiple_testing import holm_bonferroni
from benchmark_harness.analysis.statistics.proxy import c2_proxy
from benchmark_harness.analysis.statistics.ranking import pairwise_win_matrix


def _metric_value(row: AnalysisRow, spec: MetricSpec) -> float | None:
    value, reason = metric_observation(row.to_dict(), spec)
    if reason is not None:
        return None
    return value


def _eligible(row: AnalysisRow, spec: MetricSpec) -> tuple[float | None, str | None]:
    if not row.comparable:
        return None, row.failure_reason or "non_comparable"
    if not row.analysis_eligible:
        return None, "temporal_or_control_exclusion"
    if row.category in {"D", "E"}:
        return None, "control_category_exclusion"
    value = _metric_value(row, spec)
    if value is None:
        return None, f"missing metric: {spec.name}"
    return value, None


def _c1(
    rows: tuple[AnalysisRow, ...], spec: MetricSpec, *, seed: int, resamples: int
) -> dict[str, Any]:
    groups: dict[tuple[str, int, str, str | None, str | None], dict[str, float]] = defaultdict(dict)
    for row in rows:
        value, _ = _eligible(row, spec)
        if value is None or row.temporal_band not in {"T1", "T2", "T3"}:
            continue
        if row.retrieval_regime not in {
            "closed_book",
            "cutoff_controlled",
            "unrestricted_current_web",
        }:
            continue
        key = (row.task_id, row.trial, row.system_id, row.backbone_id, row.temporal_band)
        if row.retrieval_regime in groups[key]:
            raise AnalysisError(f"duplicate C1 cell for {key!r} and {row.retrieval_regime!r}.")
        groups[key][row.retrieval_regime] = value

    def contrast_rows(
        pairs: list[tuple[float, float, tuple[str, int, str, str | None, str | None]]],
        label: str,
        offset: int,
    ) -> dict[str, Any]:
        if not pairs:
            return {
                "contrast": label,
                "matched_n": 0,
                "mean_difference": None,
                "ci_lower": None,
                "ci_upper": None,
                "probability_positive": None,
                "effect_size": None,
                "p_raw": None,
                "temporal_bands": [],
            }
        left_values = [pair[0] for pair in pairs]
        right_values = [pair[1] for pair in pairs]
        bootstrap = paired_bootstrap(
            left_values,
            right_values,
            resamples=resamples,
            seed=seed + offset,
        )
        row = {
            "contrast": label,
            "matched_n": len(pairs),
            "mean_difference": bootstrap["mean_difference"],
            "ci_lower": bootstrap["ci_lower"],
            "ci_upper": bootstrap["ci_upper"],
            "probability_positive": bootstrap["probability_positive"],
            "effect_size": paired_standardized_effect(left_values, right_values),
            "p_raw": min(
                1.0,
                2 * min(bootstrap["probability_positive"], 1 - bootstrap["probability_positive"]),
            ),
            "temporal_bands": sorted({pair[2][-1] for pair in pairs}),
        }
        return row

    contrasts: list[dict[str, Any]] = []
    band_contrasts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for left, right, label in (
        ("cutoff_controlled", "closed_book", "controlled_minus_closed"),
        ("unrestricted_current_web", "cutoff_controlled", "unrestricted_minus_controlled"),
    ):
        pairs = [
            (values[left], values[right], key)
            for key, values in sorted(groups.items())
            if left in values and right in values
        ]
        contrast = contrast_rows(pairs, label, offset=len(contrasts))
        if contrast["p_raw"] is None:
            contrast.pop("p_raw", None)
        contrasts.append(contrast)
        for band in sorted({pair[2][-1] for pair in pairs if pair[2][-1] is not None}):
            band_pairs = [pair for pair in pairs if pair[2][-1] == band]
            band_contrasts[band].append(
                contrast_rows(band_pairs, label, offset=len(contrasts) + len(band_contrasts))
            )
    p_values = {row["contrast"]: row["p_raw"] for row in contrasts if row.get("p_raw") is not None}
    adjusted = holm_bonferroni(p_values)
    for row in contrasts:
        row["p_holm"] = adjusted.get(row["contrast"])
    for band_rows in band_contrasts.values():
        band_p_values = {
            row["contrast"]: row["p_raw"] for row in band_rows if row.get("p_raw") is not None
        }
        band_adjusted = holm_bonferroni(band_p_values)
        for row in band_rows:
            row["p_holm"] = band_adjusted.get(row["contrast"])
    return {
        "schema_version": "c1/v1",
        "status": "ok"
        if any(row["matched_n"] for row in contrasts)
        else "insufficient_matched_cells",
        "contrasts": contrasts,
        "band_contrasts": dict(sorted(band_contrasts.items())),
        "a2_t1_t2_slice": _a2_t1_t2_slice(rows, spec),
    }


def _a2_t1_t2_slice(rows: tuple[AnalysisRow, ...], spec: MetricSpec) -> dict[str, Any]:
    """Descriptive A2-only T1-vs-T2 performance slice.

    This slice is explicitly observational and unpaired: T1 and T2 rows come
    from disjoint task sets, so task/time composition confounds the contrast.
    It is exploratory evidence at most and is never a leakage proof.
    """

    band_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    task_counts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.category != "A2" or row.temporal_band not in {"T1", "T2"}:
            continue
        value, _ = _eligible(row, spec)
        if value is None:
            continue
        band_values[(row.temporal_band, row.task_id)].append(value)
        task_counts[row.temporal_band].add(row.task_id)
    task_means: dict[str, list[float]] = {"T1": [], "T2": []}
    for (band, _task_id), values in band_values.items():
        task_means[band].append(mean(values))
    for band in task_means:
        task_means[band].sort()
    t1_mean = mean(task_means["T1"]) if task_means["T1"] else None
    t2_mean = mean(task_means["T2"]) if task_means["T2"] else None
    return {
        "label": "a2_t1_vs_t2_observational",
        "paired": False,
        "confounded_by_task_composition": True,
        "claim_eligible": False,
        "t1_task_count": len(task_counts["T1"]),
        "t2_task_count": len(task_counts["T2"]),
        "t1_mean_utility": t1_mean,
        "t2_mean_utility": t2_mean,
        "t2_minus_t1": (t2_mean - t1_mean) if t1_mean is not None and t2_mean is not None else None,
    }


def _band_scores(
    rows: tuple[AnalysisRow, ...], spec: MetricSpec
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, set[str]]]]:
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    tasks: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if (
            row.temporal_band not in {"T2", "T3"}
            or row.retrieval_regime == "unrestricted_current_web"
        ):
            continue
        value, _ = _eligible(row, spec)
        if value is None:
            continue
        values[(row.system_id, row.temporal_band, row.task_id)].append(value)
        tasks[row.system_id][row.temporal_band].add(row.task_id)
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for (system_id, band, _task_id), task_values in sorted(values.items()):
        scores[system_id].setdefault(band, []).append(mean(task_values))
    # The type is intentionally converted after the task-level means so each
    # task/category has equal weight before the system-level proxy.
    return (
        {
            system: {band: mean(task_values) for band, task_values in bands.items()}
            for system, bands in scores.items()
        },
        {system: dict(bands) for system, bands in tasks.items()},
    )


def _c2(rows: tuple[AnalysisRow, ...], spec: MetricSpec) -> dict[str, Any]:
    scores, tasks = _band_scores(rows, spec)
    t2 = {system: values["T2"] for system, values in scores.items() if "T2" in values}
    t3 = {system: values["T3"] for system, values in scores.items() if "T3" in values}
    matched = sorted(set(t2) & set(t3))
    coverage_equal = bool(matched) and all(
        tasks[system].get("T2", set()) == tasks[matched[0]].get("T2", set())
        and tasks[system].get("T3", set()) == tasks[matched[0]].get("T3", set())
        for system in matched
    )
    result = c2_proxy(t2, t3, resolved_t3_systems=set(matched))
    if not coverage_equal:
        result["status"] = "exploratory_only"
        result["gate_reason"] = "matched T2/T3 task coverage is not identical"
    result["schema_version"] = "c2/v1"
    result["coverage_equal"] = coverage_equal
    result["t2_scores"] = {system: t2[system] for system in sorted(t2)}
    result["t3_scores"] = {system: t3[system] for system in sorted(t3)}
    return result


def _c3(rows: tuple[AnalysisRow, ...], spec: MetricSpec) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    ranking_flags: dict[str, list[bool]] = defaultdict(list)
    diagnostic_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        value, reason = _eligible(row, spec)
        if value is not None and row.retrieval_regime != "unrestricted_current_web":
            values[row.system_id].append(value)
            ranking_flags[row.system_id].append(row.ranking_eligible)
        if reason is not None:
            diagnostic_counts["reliability"][reason] += 1
        for name in ("retrieval", "control_flow", "calculation", "format_reliability"):
            count = row.diagnostics.get(name, 0)
            if isinstance(count, int) and count > 0:
                diagnostic_counts[name]["count"] += count
    scores = {system: mean(items) for system, items in sorted(values.items()) if items}
    return {
        "schema_version": "c3/v1",
        "status": "ok" if scores else "insufficient_analysis_rows",
        "system_scores": scores,
        "pairwise_win_matrix": pairwise_win_matrix(scores),
        "ranking_eligible_systems": sorted(
            system for system in scores if ranking_flags[system] and all(ranking_flags[system])
        ),
        "diagnostics": {
            name: {key: value for key, value in sorted(counts.items())}
            for name, counts in sorted(diagnostic_counts.items())
        },
    }


def analyze_rows(
    rows: Iterable[AnalysisRow | dict[str, Any]],
    catalog: MetricCatalog,
    *,
    metric_name: str | None = None,
    seed: int = 0,
    bootstrap_resamples: int = 10_000,
    identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic C1/C2/C3 objects without network or model calls."""

    normalized = coerce_analysis_rows(rows)
    seen: set[tuple[str, str, int, str]] = set()
    for row in normalized:
        key = (row.system_id, row.task_id, row.trial, row.retrieval_regime)
        if key in seen:
            raise AnalysisError(f"duplicate analysis cell {key!r}.")
        seen.add(key)
    selected = metric_name or next(
        (metric.name for metric in catalog.metrics if metric.primary), None
    )
    if selected is None:
        selected = catalog.metrics[0].name
    spec = catalog.metric(selected)
    exclusions = [
        {
            "system_id": row.system_id,
            "task_id": row.task_id,
            "trial": row.trial,
            "retrieval_regime": row.retrieval_regime,
            "reason": reason,
        }
        for row in normalized
        for _, reason in [_eligible(row, spec)]
        if reason is not None
    ]
    return {
        "schema_version": "analysis/v1",
        "metric": selected,
        "metric_catalog_sha256": catalog.hash,
        "identity": dict(identity or {}),
        "attempted_rows": len(normalized),
        "exclusions": sorted(
            exclusions,
            key=lambda item: (
                item["system_id"],
                item["task_id"],
                item["trial"],
                item["retrieval_regime"],
                item["reason"],
            ),
        ),
        "c1": _c1(normalized, spec, seed=seed, resamples=bootstrap_resamples),
        "c2": _c2(normalized, spec),
        "c3": _c3(normalized, spec),
    }
