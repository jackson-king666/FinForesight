"""Offline E1--E4 analyses for the three-system experiment/v4 release.

The analysis layer consumes projected rows and never calls a model, search
service, parser, or evaluator.  It keeps raw utility, operational failure
penalties, task aggregation, cluster uncertainty, and missing-slice accounting
as separate fields so an incomplete run cannot look like a completed result.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import NormalDist, mean
from typing import Any

from benchmark_harness.analysis.metrics.formal import brier_skill_score_unclipped
from benchmark_harness.analysis.statistics.cluster import (
    cluster_bootstrap,
    paired_cluster_bootstrap,
)
from benchmark_harness.analysis.statistics.multiple_testing import holm_bonferroni
from benchmark_harness.analysis.statistics.ranking import (
    kendall_tau,
    pearson_correlation,
    spearman_correlation,
)
from benchmark_harness.calibration.evidence_qc import wilson_interval

FULL_RELEASE_SCHEMA_VERSION = "full-release-analysis/v1"
FORMAL_UTILITY_VERSION = "formal-utility/v1"
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_ALPHA = 0.05
SYSTEMS = ("financeharness", "odr", "gpt-researcher")
REGIMES = ("closed_book", "cutoff_controlled", "unrestricted_current_web")
CATEGORY_WEIGHTS = {"A1": 0.25, "A2": 0.25, "B": 0.25, "C": 0.25}
E_FORMULA_FAMILIES = (
    "arithmetic",
    "percentage_change",
    "weighted_average",
    "compound_growth",
    "duration_approx",
    "black_scholes_call",
    "present_value",
    "ratio",
)


class FullReleaseAnalysisError(ValueError):
    """Raised when v4 analysis rows cannot support a deterministic result."""


def _rows(value: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(value, Mapping) or isinstance(value, str | bytes):
        raise FullReleaseAnalysisError("analysis rows must be an iterable of objects")
    result = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise FullReleaseAnalysisError(f"analysis row {index} must be an object")
        result.append(dict(row))
    return result


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _utility(row: Mapping[str, Any]) -> float | None:
    value = row.get("formal_utility")
    if value is None:
        metrics = row.get("metric_values")
        if isinstance(metrics, Mapping):
            for key in ("utility", "formal_utility", "formal_utility_v1", "u"):
                if key in metrics:
                    value = metrics[key]
                    break
    return _finite(value)


def _eligible(row: Mapping[str, Any], *, require_analysis: bool = True) -> bool:
    if require_analysis and row.get("analysis_eligible") is False:
        return False
    if row.get("comparable") is False:
        return False
    return _utility(row) is not None


def _system(row: Mapping[str, Any]) -> str:
    value = row.get("system_id")
    if not isinstance(value, str) or not value.strip():
        raise FullReleaseAnalysisError("analysis row system_id must be non-empty")
    return value.strip()


def _task(row: Mapping[str, Any]) -> str:
    value = row.get("task_id")
    if not isinstance(value, str) or not value.strip():
        raise FullReleaseAnalysisError("analysis row task_id must be non-empty")
    return value.strip()


def _cluster(row: Mapping[str, Any]) -> str:
    for key in (
        "event_cluster_id",
        "matched_pair_id",
        "calendar_block_id",
        "cluster_id",
        "task_id",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _task(row)


def _category(row: Mapping[str, Any]) -> str:
    value = row.get("category")
    if not isinstance(value, str) or not value.strip():
        raise FullReleaseAnalysisError(f"row {_task(row)!r} has no category")
    return value.strip().upper()


def _variant(row: Mapping[str, Any]) -> str:
    category = _category(row)
    if category == "A2":
        value = row.get("variant", row.get("signal_variant"))
        return str(value).strip().upper() if value is not None else "UNKNOWN"
    if category == "B":
        value = row.get("event_subtype", row.get("subtype", row.get("variant")))
        return str(value).strip().lower() if value is not None else "unknown"
    return "all"


def _trial(row: Mapping[str, Any]) -> int:
    value = row.get("trial", 1)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise FullReleaseAnalysisError(f"row {_task(row)!r} has an invalid trial")
    return value


def _mean_or_none(values: Sequence[float]) -> float | None:
    return mean(values) if values else None


def _p_from_interval(
    lower: float | None, upper: float | None, estimate: float | None
) -> float | None:
    if lower is None or upper is None or estimate is None or upper <= lower:
        return None
    half = (upper - lower) / 2.0
    if half <= 0:
        return None
    z = abs(estimate) / half
    return min(1.0, max(0.0, 2.0 * (1.0 - NormalDist().cdf(z))))


def _cluster_difference_report(
    differences: Sequence[tuple[str, float]],
    *,
    seed: int,
    resamples: int,
    label: str,
) -> dict[str, Any]:
    if not differences:
        return {
            "contrast": label,
            "matched_n": 0,
            "cluster_count": 0,
            "mean_difference": None,
            "ci_lower": None,
            "ci_upper": None,
            "p_raw": None,
        }
    report = cluster_bootstrap(differences, resamples=resamples, seed=seed)
    report.update(
        {
            "contrast": label,
            "matched_n": len(differences),
            "p_raw": _p_from_interval(report["ci_lower"], report["ci_upper"], report["estimate"]),
        }
    )
    report["mean_difference"] = report.pop("estimate")
    return report


def _paired_regime_differences(
    rows: Sequence[Mapping[str, Any]],
    system_id: str,
    left_regime: str,
    right_regime: str,
) -> list[tuple[str, float]]:
    grouped: dict[tuple[str, int], dict[str, tuple[str, float]]] = {}
    for row in rows:
        if _system(row) != system_id or row.get("retrieval_regime") not in {
            left_regime,
            right_regime,
        }:
            continue
        if not _eligible(row):
            continue
        key = (_task(row), _trial(row))
        grouped.setdefault(key, {})[str(row["retrieval_regime"])] = (_cluster(row), _utility(row))  # type: ignore[arg-type]
    return [
        (left[0], left[1] - right[1])
        for values in grouped.values()
        if left_regime in values and right_regime in values
        for left, right in [(values[left_regime], values[right_regime])]
    ]


def e1_delta_web(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = 0,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    system_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Estimate unrestricted-minus-cutoff and controlled-minus-closed effects."""

    normalized = _rows(rows)
    observed_systems = {_system(row) for row in normalized}
    if system_ids is None:
        systems = sorted(observed_systems)
    else:
        systems = sorted({str(system_id) for system_id in system_ids})
        unexpected = sorted(observed_systems - set(systems))
        if unexpected:
            raise FullReleaseAnalysisError(
                "E1 rows contain undeclared systems: " + ", ".join(unexpected)
            )
    contrasts: list[dict[str, Any]] = []
    for index, system_id in enumerate(systems):
        for left, right, label in (
            (
                "unrestricted_current_web",
                "cutoff_controlled",
                "unrestricted_minus_cutoff",
            ),
            ("cutoff_controlled", "closed_book", "controlled_minus_closed"),
        ):
            differences = _paired_regime_differences(normalized, system_id, left, right)
            result = _cluster_difference_report(
                differences,
                seed=seed + index * 17 + len(contrasts),
                resamples=resamples,
                label=label,
            )
            result["system_id"] = system_id
            result["left_regime"] = left
            result["right_regime"] = right
            contrasts.append(result)
    primary = [row for row in contrasts if row["contrast"] == "unrestricted_minus_cutoff"]
    adjusted = holm_bonferroni(
        {row["system_id"]: row["p_raw"] for row in primary if row["p_raw"] is not None}
    )
    for row in contrasts:
        if row["contrast"] == "unrestricted_minus_cutoff":
            row["p_holm"] = adjusted.get(row["system_id"])
    band_contrasts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for band in sorted(
        {
            str(row.get("temporal_band"))
            for row in normalized
            if row.get("temporal_band") is not None
        }
    ):
        band_rows = [row for row in normalized if str(row.get("temporal_band")) == band]
        for index, system_id in enumerate(systems):
            differences = _paired_regime_differences(
                band_rows, system_id, "unrestricted_current_web", "cutoff_controlled"
            )
            result = _cluster_difference_report(
                differences,
                seed=seed + 1000 + index,
                resamples=resamples,
                label="unrestricted_minus_cutoff",
            )
            result["system_id"] = system_id
            band_contrasts[band].append(result)
    exclusions = Counter(
        str(row.get("failure_reason"))
        for row in normalized
        if not _eligible(row) and row.get("failure_reason")
    )
    return {
        "schema_version": "e1/v1",
        "status": "ok" if any(row["matched_n"] for row in contrasts) else "insufficient_data",
        "formal_utility_version": FORMAL_UTILITY_VERSION,
        "contrasts": contrasts,
        "band_contrasts": dict(sorted(band_contrasts.items())),
        "exclusions": dict(sorted(exclusions.items())),
        "human_exposure_required": True,
        "alpha": alpha,
    }


def _score_map(value: Any) -> dict[str, dict[str, float]]:
    if isinstance(value, Mapping):
        result: dict[str, dict[str, float]] = {}
        for system, scores in value.items():
            if isinstance(scores, Mapping):
                parsed = {
                    str(task): number
                    for task, raw in scores.items()
                    if (number := _finite(raw)) is not None
                }
                result[str(system)] = parsed
            elif isinstance(scores, Sequence) and not isinstance(scores, str | bytes):
                result[str(system)] = {
                    _task(row): _utility(row)  # type: ignore[misc]
                    for row in scores
                    if isinstance(row, Mapping) and _utility(row) is not None
                }
        return result
    rows = _rows(value)
    result: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if _eligible(row):
            result[_system(row)][_task(row)] = _utility(row)  # type: ignore[assignment]
    return dict(result)


def _system_means(scores: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    return {system: mean(values.values()) for system, values in sorted(scores.items()) if values}


def _pairwise_orders(scores: Mapping[str, float]) -> list[dict[str, Any]]:
    systems = sorted(scores)
    return [
        {
            "left": left,
            "right": right,
            "left_score": scores[left],
            "right_score": scores[right],
            "order": "tie"
            if scores[left] == scores[right]
            else left
            if scores[left] > scores[right]
            else right,
        }
        for index, left in enumerate(systems)
        for right in systems[index + 1 :]
    ]


def _tost(
    differences: Sequence[float], margin: float, alpha: float, *, seed: int, resamples: int
) -> dict[str, Any]:
    if not differences:
        return {
            "n": 0,
            "estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "passed": False,
            "p_lower": None,
            "p_upper": None,
        }
    bootstrap = cluster_bootstrap(
        [(str(index), value) for index, value in enumerate(differences)],
        seed=seed,
        resamples=resamples,
    )
    estimate = float(bootstrap["estimate"])
    # The two one-sided decision is represented by the bootstrap bounds.  The
    # p-value fields are approximate normal summaries, never used to select
    # data or change the margin.
    lower = float(bootstrap["ci_lower"])
    upper = float(bootstrap["ci_upper"])
    sd = math.sqrt(
        sum((value - estimate) ** 2 for value in differences) / max(1, len(differences) - 1)
    )
    se = sd / math.sqrt(len(differences)) if differences else 0.0
    if se:
        p_lower = 1.0 - NormalDist().cdf((estimate + margin) / se)
        p_upper = 1.0 - NormalDist().cdf((margin - estimate) / se)
    else:
        p_lower = 0.0 if estimate >= -margin else 1.0
        p_upper = 0.0 if estimate <= margin else 1.0
    return {
        "n": len(differences),
        "estimate": estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "passed": lower >= -margin and upper <= margin,
        "p_lower": max(0.0, min(1.0, p_lower)),
        "p_upper": max(0.0, min(1.0, p_upper)),
        "alpha": alpha,
    }


def e2_decision_tests(
    t2_task_scores: Any,
    t3_task_scores: Any,
    *,
    seed: int = 0,
    margin: float = 0.10,
    alpha: float = DEFAULT_ALPHA,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    system_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the finite-three-system T2/T3 gap and selection decision tests."""

    if not isinstance(margin, int | float) or isinstance(margin, bool) or margin <= 0:
        raise FullReleaseAnalysisError("E2 margin must be positive")
    t2 = _score_map(t2_task_scores)
    t3 = _score_map(t3_task_scores)
    observed_systems = set(t2) | set(t3)
    if system_ids is None:
        systems = sorted(observed_systems)
    else:
        systems = sorted({str(system_id) for system_id in system_ids})
        unexpected = sorted(observed_systems - set(systems))
        if unexpected:
            raise FullReleaseAnalysisError(
                "E2 scores contain undeclared systems: " + ", ".join(unexpected)
            )
    t2_means = _system_means(t2)
    t3_means = _system_means(t3)
    selected = min(
        (system for system in t2_means),
        key=lambda system: (-t2_means[system], system),
        default=None,
    )
    gaps: list[dict[str, Any]] = []
    tost_p: dict[str, float] = {}
    for index, left in enumerate(systems):
        for right in systems[index + 1 :]:
            t2_common = sorted(set(t2.get(left, {})) & set(t2.get(right, {})))
            t3_common = sorted(set(t3.get(left, {})) & set(t3.get(right, {})))
            t2_gap = (
                mean([t2[left][task] - t2[right][task] for task in t2_common])
                if t2_common
                else None
            )
            t3_gap = (
                mean([t3[left][task] - t3[right][task] for task in t3_common])
                if t3_common
                else None
            )
            changes: list[float] = []
            for task in sorted(set(t2_common) & set(t3_common)):
                changes.append(
                    (t2[left][task] - t2[right][task]) - (t3[left][task] - t3[right][task])
                )
            result = _tost(
                changes,
                float(margin),
                alpha,
                seed=seed + len(gaps),
                resamples=resamples,
            )
            result.update(
                {
                    "left": left,
                    "right": right,
                    "t2_gap": t2_gap,
                    "t3_gap": t3_gap,
                    "common_n": len(changes),
                }
            )
            if result["p_lower"] is not None and result["p_upper"] is not None:
                tost_p[f"{left}__{right}"] = max(result["p_lower"], result["p_upper"])
            gaps.append(result)
    tost_adjusted = holm_bonferroni(tost_p)
    for result in gaps:
        result["p_holm"] = tost_adjusted.get(f"{result['left']}__{result['right']}")
    selection_losses: list[dict[str, Any]] = []
    if selected is not None:
        for competitor in systems:
            if competitor == selected:
                continue
            common = sorted(set(t3.get(selected, {})) & set(t3.get(competitor, {})))
            # Positive loss means the T2-selected system underperforms the competitor.
            differences = [t3[competitor][task] - t3[selected][task] for task in common]
            report = _tost(
                differences,
                float(margin),
                alpha,
                seed=seed + 100 + len(selection_losses),
                resamples=resamples,
            )
            selection_losses.append(
                {
                    "selected_system": selected,
                    "competitor": competitor,
                    "n": len(common),
                    "loss_estimate": report["estimate"],
                    "upper_bound_95": report["ci_upper"],
                    "passes_margin": report["ci_upper"] is not None
                    and report["ci_upper"] <= margin,
                }
            )
    t2_order = _pairwise_orders(t2_means)
    t3_order = _pairwise_orders(t3_means)
    correlation_pairs = sorted(set(t2_means) & set(t3_means))
    t2_values = [t2_means[system] for system in correlation_pairs]
    t3_values = [t3_means[system] for system in correlation_pairs]
    stability: Counter[str] = Counter()
    if correlation_pairs:
        rng = random.Random(seed)
        task_union = sorted(
            set().union(
                *(set(t2.get(system, {})) & set(t3.get(system, {})) for system in correlation_pairs)
            )
        )
        for _ in range(min(resamples, 10_000)):
            sampled = [rng.choice(task_union) for _ in task_union] if task_union else []
            bootstrap_scores = {
                system: mean([t3[system][task] for task in sampled if task in t3.get(system, {})])
                for system in correlation_pairs
            }
            bootstrap_scores = {
                key: value for key, value in bootstrap_scores.items() if math.isfinite(value)
            }
            if bootstrap_scores:
                winner = min(bootstrap_scores, key=lambda key: (-bootstrap_scores[key], key))
                stability[winner] += 1
    total_stability = sum(stability.values())
    return {
        "schema_version": "e2/v1",
        "status": "ok" if t2_means and t3_means else "awaiting_settlement",
        "systems": systems,
        "margin": margin,
        "alpha": alpha,
        "t2_scores": t2_means,
        "t3_scores": t3_means,
        "t2_selected_system": selected,
        "gap_tests": gaps,
        "selection_loss": selection_losses,
        "spearman": spearman_correlation(t2_values, t3_values) if len(t2_values) >= 2 else None,
        "pearson": pearson_correlation(t2_values, t3_values) if len(t2_values) >= 2 else None,
        "kendall": kendall_tau(t2_values, t3_values) if len(t2_values) >= 2 else None,
        "ties": {
            "t2": sum(item["order"] == "tie" for item in t2_order),
            "t3": sum(item["order"] == "tie" for item in t3_order),
        },
        "pairwise_orders": {"t2": t2_order, "t3": t3_order},
        "leave_one_system_out": [
            {
                "omitted": omitted,
                "t2_order": _pairwise_orders({k: v for k, v in t2_means.items() if k != omitted}),
                "t3_order": _pairwise_orders({k: v for k, v in t3_means.items() if k != omitted}),
            }
            for omitted in systems
        ],
        "rank_stability": {
            system: (stability[system] / total_stability if total_stability else None)
            for system in systems
        },
        "generalization_scope": "named_three_system_panel",
    }


def _task_slice_values(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, str], float]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if not _eligible(row):
            continue
        key = (_system(row), _category(row), _variant(row), _task(row))
        grouped[key].append(_utility(row))  # type: ignore[arg-type]
    return {key: mean(values) for key, values in grouped.items()}


def _subslice_label(category: str, variant: str) -> str:
    return f"{category}:{variant}" if category in {"A2", "B"} else category


def e3_leaderboard(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = 0,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    system_ids: Sequence[str] | None = None,
    temporal_band: str | None = None,
    _include_band_reports: bool = True,
) -> dict[str, Any]:
    """Build the fixed-weight T2 leaderboard and a separate settled-T3 view."""

    all_rows = _rows(rows)
    normalized = [
        row
        for row in all_rows
        if row.get("retrieval_regime") == "cutoff_controlled"
        and str(row.get("temporal_band", "T2")).upper() in {"T2", "T3"}
        and (
            temporal_band is None
            or str(row.get("temporal_band", "T2")).upper() == temporal_band.upper()
        )
    ]
    task_values = _task_slice_values(normalized)
    observed_systems = {_system(row) for row in normalized}
    if system_ids is None:
        systems = sorted(observed_systems)
    else:
        systems = sorted({str(system_id) for system_id in system_ids})
        unexpected = sorted(observed_systems - set(systems))
        if unexpected:
            raise FullReleaseAnalysisError(
                "leaderboard rows contain undeclared systems: " + ", ".join(unexpected)
            )
    required = {"A1": ("all",), "A2": ("F", "T", "H"), "B": ("earnings", "macro"), "C": ("all",)}
    component: dict[str, dict[str, float | None]] = {}
    slice_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    missing: dict[str, list[str]] = {}
    for system in systems:
        component[system] = {}
        missing[system] = []
        for category, variants in required.items():
            for variant in variants:
                values = [
                    value
                    for (sys, cat, var, _task_id), value in task_values.items()
                    if sys == system and cat == category and (variant == "all" or var == variant)
                ]
                label = _subslice_label(category, variant)
                component[system][label] = mean(values) if values else None
                task_count = sum(
                    1
                    for (sys, cat, var, _task_id) in task_values
                    if sys == system and cat == category and (variant == "all" or var == variant)
                )
                slice_counts[system][category][variant] = task_count
                if not values:
                    missing[system].append(label)
        complete = not missing[system]
        category_values = {
            category: component[system][_subslice_label(category, "all")]
            if category in {"A1", "C"}
            else (
                mean(
                    component[system][_subslice_label(category, variant)]
                    for variant in required[category]
                    if component[system][_subslice_label(category, variant)] is not None
                )
                if all(
                    component[system][_subslice_label(category, variant)] is not None
                    for variant in required[category]
                )
                else None
            )
            for category in CATEGORY_WEIGHTS
        }
        component[system]["category_scores"] = category_values  # type: ignore[assignment]
        component[system]["overall"] = (
            sum(
                CATEGORY_WEIGHTS[category] * float(category_values[category])
                for category in CATEGORY_WEIGHTS
            )
            if complete
            and all(category_values[category] is not None for category in CATEGORY_WEIGHTS)
            else None
        )
    scores = {
        system: component[system]["overall"]
        for system in systems
        if component[system]["overall"] is not None
    }
    pairwise: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    for index, left in enumerate(systems):
        for right in systems[index + 1 :]:
            left_values = {
                task: value for (sys, _cat, _var, task), value in task_values.items() if sys == left
            }
            right_values = {
                task: value
                for (sys, _cat, _var, task), value in task_values.items()
                if sys == right
            }
            common = sorted(set(left_values) & set(right_values))
            report = (
                paired_cluster_bootstrap(
                    [left_values[task] for task in common],
                    [right_values[task] for task in common],
                    [
                        _cluster(next(row for row in normalized if _task(row) == task))
                        for task in common
                    ],
                    seed=seed + len(pairwise),
                    resamples=resamples,
                )
                if common
                else {
                    "pair_count": 0,
                    "cluster_count": 0,
                    "estimate": None,
                    "ci_lower": None,
                    "ci_upper": None,
                }
            )
            key = f"{left}__{right}"
            entry = {
                "left": left,
                "right": right,
                "n": len(common),
                "mean_difference": report.get("estimate"),
                "ci_lower": report.get("ci_lower"),
                "ci_upper": report.get("ci_upper"),
                "cluster_count": report.get("cluster_count", 0),
                "p_raw": _p_from_interval(
                    report.get("ci_lower"), report.get("ci_upper"), report.get("estimate")
                ),
            }
            if entry["p_raw"] is not None:
                p_values[key] = entry["p_raw"]
            pairwise.append(entry)
    adjusted = holm_bonferroni(p_values)
    for entry in pairwise:
        entry["p_holm"] = adjusted.get(f"{entry['left']}__{entry['right']}")
    rank_order = sorted(scores, key=lambda system: (-float(scores[system]), system))
    result: dict[str, Any] = {
        "schema_version": "e3/v1",
        "status": "ok"
        if scores and not any(missing.get(system) for system in systems)
        else "incomplete_slices",
        "systems": systems,
        "category_weights": dict(CATEGORY_WEIGHTS),
        "system_scores": scores,
        "component_scores": component,
        "slice_counts": {system: dict(values) for system, values in slice_counts.items()},
        "missing_slices": missing,
        "pairwise": pairwise,
        "expected_rank": {system: index + 1 for index, system in enumerate(rank_order)},
        "top1_probability": {
            system: (1.0 if index == 0 else 0.0) for index, system in enumerate(rank_order)
        },
        "complete_case_sensitivity": {
            "systems": rank_order,
            "task_count": len(
                set.intersection(
                    *[
                        {task for (sys, _cat, _var, task) in task_values if sys == system}
                        for system in systems
                    ]
                )
            )
            if systems
            else 0,
        },
        "diagnostics": _diagnostics(normalized),
        "alpha": alpha,
    }
    if temporal_band is not None:
        result["temporal_band"] = temporal_band.upper()
    elif _include_band_reports:
        t2_report = e3_leaderboard(
            normalized,
            seed=seed,
            resamples=resamples,
            alpha=alpha,
            system_ids=system_ids,
            temporal_band="T2",
            _include_band_reports=False,
        )
        t3_report = e3_leaderboard(
            normalized,
            seed=seed + 1,
            resamples=resamples,
            alpha=alpha,
            system_ids=system_ids,
            temporal_band="T3",
            _include_band_reports=False,
        )
        # The top-level table is the predeclared T2 main leaderboard.  Keep
        # the settled-T3 table explicitly separate so a combined T2/T3
        # average cannot be mistaken for the primary ranking.
        result = {
            **t2_report,
            "t2": t2_report,
            "t3": t3_report,
            "controls": control_experiments(all_rows, system_ids=system_ids),
            "main_leaderboard": "t2",
        }
    return result


def _probability(row: Mapping[str, Any]) -> float | None:
    for source in (
        row,
        row.get("metric_values") if isinstance(row.get("metric_values"), Mapping) else {},
        row.get("parsed") if isinstance(row.get("parsed"), Mapping) else {},
    ):
        if isinstance(source, Mapping):
            value = source.get("analysis_probability", source.get("probability_up"))
            parsed = _finite(value)
            if parsed is not None and 0.0 <= parsed <= 1.0:
                return parsed
    return None


def _actual_up(row: Mapping[str, Any]) -> int | None:
    value = row.get("actual_up", row.get("actual_label", row.get("label")))
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float) and value in {0, 1}:
        return int(value)
    direction = row.get("actual_direction", row.get("direction"))
    if isinstance(direction, str) and direction.lower() in {"up", "down"}:
        return int(direction.lower() == "up")
    frozen = row.get("frozen_metrics")
    if isinstance(frozen, Mapping):
        value = frozen.get("actual_up")
        if isinstance(value, int | float) and value in {0, 1}:
            return int(value)
        direction = frozen.get("actual_direction")
        if isinstance(direction, str) and direction.lower() in {"up", "down"}:
            return int(direction.lower() == "up")
    return None


def _diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failure_counts: Counter[str] = Counter()
    for row in rows:
        reason = row.get("failure_reason")
        if reason:
            failure_counts[str(reason)] += 1
        diagnostics = row.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            for key, value in diagnostics.items():
                if isinstance(value, int) and value > 0:
                    failure_counts[str(key)] += value
    return {"failure_or_protocol_counts": dict(sorted(failure_counts.items()))}


def _control_value(row: Mapping[str, Any], key: str) -> Any:
    for source in (
        row,
        row.get("metric_values") if isinstance(row.get("metric_values"), Mapping) else {},
        row.get("frozen_metrics") if isinstance(row.get("frozen_metrics"), Mapping) else {},
    ):
        if isinstance(source, Mapping) and key in source:
            return source[key]
    return None


def control_experiments(
    rows: Iterable[Mapping[str, Any]],
    *,
    system_ids: Sequence[str] | None = None,
    min_d_pairs: int = 30,
    min_e_cases: int = 10,
) -> dict[str, Any]:
    """Summarize D matched controls and all eight E calculation families.

    D/E are control populations, not temporal forecasting rows.  This report
    therefore does not route them through the T2 leaderboard and keeps their
    completeness gates visible even when a historical run is still partial.
    """

    normalized = _rows(rows)
    observed_systems = {_system(row) for row in normalized}
    if system_ids is None:
        systems = sorted(observed_systems)
    else:
        systems = sorted({str(system_id) for system_id in system_ids})
        unexpected = sorted(observed_systems - set(systems))
        if unexpected:
            raise FullReleaseAnalysisError(
                "control rows contain undeclared systems: " + ", ".join(unexpected)
            )

    d_report: dict[str, Any] = {}
    e_report: dict[str, Any] = {}
    for system in systems:
        d_rows = [row for row in normalized if _system(row) == system and _category(row) == "D"]
        d_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in d_rows:
            pair_id = row.get("matched_pair_id", row.get("pair_id"))
            if isinstance(pair_id, str) and pair_id.strip():
                d_groups[pair_id.strip()].append(row)
        d_pairs: list[dict[str, Any]] = []
        for pair_id, pair_rows in sorted(d_groups.items()):
            by_trial: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
            for row in pair_rows:
                by_trial[_trial(row)].append(row)
            trial_reports: list[dict[str, Any]] = []
            for trial, trial_rows in sorted(by_trial.items()):
                values = [_control_value(row, "logic_adherence") for row in trial_rows]
                trial_valid = len(trial_rows) == 2 and all(
                    isinstance(value, bool) for value in values
                )
                trial_reports.append(
                    {
                        "trial": trial,
                        "n": len(trial_rows),
                        "logic_adherence": mean(values) if trial_valid else None,
                        "complete": trial_valid,
                    }
                )
            valid = bool(trial_reports) and all(report["complete"] for report in trial_reports)
            d_pairs.append(
                {
                    "matched_pair_id": pair_id,
                    "n": len(pair_rows),
                    "trial_count": len(trial_reports),
                    "complete_trial_count": sum(report["complete"] for report in trial_reports),
                    "logic_adherence": mean([report["logic_adherence"] for report in trial_reports])
                    if valid
                    else None,
                    "complete": valid,
                    "history_conflict_count": sum(
                        bool(_control_value(row, "history_conflict")) for row in pair_rows
                    ),
                    "trials": trial_reports,
                }
            )
        complete_d = [pair for pair in d_pairs if pair["complete"]]

        e_rows = [row for row in normalized if _system(row) == system and _category(row) == "E"]
        family_rows: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in e_rows:
            family = _control_value(row, "formula_family")
            if not isinstance(family, str) or not family.strip():
                family = str(row.get("family", "unknown"))
            calc_policy = str(row.get("calc_policy", "calc_on"))
            family_rows[(family.strip().lower(), calc_policy)].append(row)
        for family in E_FORMULA_FAMILIES:
            arms: dict[str, Any] = {}
            for calc_policy in ("calc_on", "calc_off"):
                arm_rows = family_rows.get((family, calc_policy), [])
                exact = [
                    _control_value(row, "exact_match_0_2pct")
                    for row in arm_rows
                    if isinstance(_control_value(row, "exact_match_0_2pct"), bool)
                ]
                formula = [
                    _control_value(row, "formula_text_match")
                    for row in arm_rows
                    if isinstance(_control_value(row, "formula_text_match"), bool)
                ]
                units = [
                    _control_value(row, "unit_match")
                    for row in arm_rows
                    if isinstance(_control_value(row, "unit_match"), bool)
                ]
                arms[calc_policy] = {
                    "row_count": len(arm_rows),
                    "case_count": len({_task(row) for row in arm_rows}),
                    "exact_match_rate": mean(exact) if exact else None,
                    "formula_text_match_rate": mean(formula) if formula else None,
                    "unit_match_rate": mean(units) if units else None,
                }
            e_report.setdefault(system, {})[family] = arms
        paired_calc: list[dict[str, Any]] = []
        for task_id in sorted({_task(row) for row in e_rows}):
            on = [
                row
                for row in e_rows
                if _task(row) == task_id and row.get("calc_policy", "calc_on") == "calc_on"
            ]
            off = [
                row
                for row in e_rows
                if _task(row) == task_id and row.get("calc_policy") == "calc_off"
            ]
            if not on or not off:
                continue
            on_value = _control_value(on[0], "exact_match_0_2pct")
            off_value = _control_value(off[0], "exact_match_0_2pct")
            paired_calc.append(
                {
                    "task_id": task_id,
                    "calc_on": on_value,
                    "calc_off": off_value,
                    "gain": (
                        int(on_value) - int(off_value)
                        if isinstance(on_value, bool) and isinstance(off_value, bool)
                        else None
                    ),
                }
            )
        e_report[system]["paired_calculation"] = {
            "pair_count": len(paired_calc),
            "exact_match_gain": mean(
                [pair["gain"] for pair in paired_calc if pair["gain"] is not None]
            )
            if any(pair["gain"] is not None for pair in paired_calc)
            else None,
            "pairs": paired_calc,
        }
        e_report[system]["valid"] = all(
            e_report[system][family]["calc_on"]["case_count"] >= min_e_cases
            and e_report[system][family]["calc_off"]["case_count"] >= min_e_cases
            for family in E_FORMULA_FAMILIES
        )
        d_report[system] = {
            "row_count": len(d_rows),
            "pair_count": len(d_pairs),
            "complete_pair_count": len(complete_d),
            "minimum_pair_count": min_d_pairs,
            "valid": len(complete_d) >= min_d_pairs,
            "pairs": d_pairs,
        }
    valid_systems = bool(systems) and all(
        d_report[system]["valid"] and e_report[system]["valid"] for system in systems
    )
    return {
        "schema_version": "control-experiments/v1",
        "systems": systems,
        "d": d_report,
        "e": e_report,
        "e_formula_families": list(E_FORMULA_FAMILIES),
        "status": "ok" if valid_systems else "incomplete_controls",
        "minimums": {"d_pairs": min_d_pairs, "e_cases_per_family": min_e_cases},
    }


def _reliability_bins(probabilities: Sequence[tuple[float, int]]) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    for index in range(10):
        lower = index / 10.0
        upper = (index + 1) / 10.0
        selected = [
            item
            for item in probabilities
            if lower <= item[0] < upper or index == 9 and item[0] == 1.0
        ]
        count = len(selected)
        successes = sum(actual for _, actual in selected)
        bins.append(
            {
                "bin": index,
                "interval": [lower, upper],
                "count": count,
                "mean_probability": mean([probability for probability, _ in selected])
                if selected
                else None,
                "observed_rate": successes / count if selected else None,
                "wilson_lower": wilson_interval(successes, count)[0] if selected else None,
                "wilson_upper": wilson_interval(successes, count)[1] if selected else None,
            }
        )
    return bins


def _calibration_subset(
    rows: Sequence[Mapping[str, Any]], label: str | None = None
) -> dict[str, Any]:
    subset = rows if label is None else [row for row in rows if _variant(row) == label]
    valid: list[tuple[str, float, int]] = []
    invalid = 0
    for row in subset:
        probability = _probability(row)
        actual = _actual_up(row)
        if probability is None or actual is None:
            invalid += 1
            continue
        valid.append((_cluster(row), probability, actual))
    raw_losses = [(probability - actual) ** 2 for _, probability, actual in valid]
    operational_losses = list(raw_losses) + [1.0] * invalid
    clusters: dict[str, list[float]] = defaultdict(list)
    for cluster, probability, actual in valid:
        clusters[cluster].append((probability - actual) ** 2)
    bootstrap = (
        cluster_bootstrap({cluster: mean(losses) for cluster, losses in clusters.items()}, seed=0)
        if clusters
        else None
    )
    brier = mean(raw_losses) if raw_losses else None
    operational = mean(operational_losses) if operational_losses else None
    return {
        "row_count": len(subset),
        "valid_output_count": len(valid),
        "invalid_output_count": invalid,
        "valid_output_rate": len(valid) / len(subset) if subset else None,
        "raw_brier": brier,
        "operational_brier": operational,
        "brier_skill_score_unclipped": brier_skill_score_unclipped(brier)
        if brier is not None
        else None,
        "reliability": _reliability_bins(
            [(probability, actual) for _, probability, actual in valid]
        ),
        "ece": (
            sum(
                bin_value["count"] * abs(bin_value["mean_probability"] - bin_value["observed_rate"])
                for bin_value in _reliability_bins(
                    [(probability, actual) for _, probability, actual in valid]
                )
                if bin_value["count"] and bin_value["mean_probability"] is not None
            )
            / len(valid)
            if valid
            else None
        ),
        "cluster_bootstrap": bootstrap,
        "directional_accuracy": (
            mean(int((probability >= 0.5) == bool(actual)) for _, probability, actual in valid)
            if valid
            else None
        ),
    }


def e4_calibration(
    rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = 0,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    system_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute ten-bin B calibration with raw and invalid-output accounting."""

    normalized = [
        row
        for row in _rows(rows)
        if _category(row) == "B" and row.get("retrieval_regime") in {None, "cutoff_controlled"}
    ]
    overall = _calibration_subset(normalized)
    # Recompute the cluster interval with the declared seed/resample count.
    if overall["raw_brier"] is not None:
        valid_pairs = [
            (_cluster(row), (_probability(row) - _actual_up(row)) ** 2)
            for row in normalized
            if _probability(row) is not None and _actual_up(row) is not None
        ]
        overall["cluster_bootstrap"] = (
            cluster_bootstrap(valid_pairs, seed=seed, resamples=resamples) if valid_pairs else None
        )
    subtypes = sorted({_variant(row) for row in normalized})
    bands = sorted(
        {
            str(row.get("temporal_band"))
            for row in normalized
            if row.get("temporal_band") is not None
        }
    )
    by_subtype = {subtype: _calibration_subset(normalized, subtype) for subtype in subtypes}
    by_band = {
        band: _calibration_subset(
            [row for row in normalized if str(row.get("temporal_band")) == band]
        )
        for band in bands
    }
    observed_systems = {_system(row) for row in normalized}
    if system_ids is None:
        systems = sorted(observed_systems)
    else:
        systems = sorted({str(system_id) for system_id in system_ids})
        unexpected = sorted(observed_systems - set(systems))
        if unexpected:
            raise FullReleaseAnalysisError(
                "E4 rows contain undeclared systems: " + ", ".join(unexpected)
            )
    by_system = {
        system: _calibration_subset([row for row in normalized if _system(row) == system])
        for system in systems
    }
    return {
        "schema_version": "e4/v1",
        "status": "ok" if overall["valid_output_count"] else "no_valid_probability_outputs",
        "bins": 10,
        "interval_definition": "[0,.1),...,[.9,1]",
        "overall": overall,
        "by_subtype": by_subtype,
        "by_band": by_band,
        "by_system": by_system,
        "empty_subtypes": [
            subtype for subtype in ("earnings", "macro") if subtype not in by_subtype
        ],
        "invalid_probability_policy": "operational_brier_1_and_valid_output_rate",
        "scope": "cutoff_controlled_when_regime_is_declared",
        "seed": seed,
        "resamples": resamples,
    }


def experiment_status(
    design_status: str = "draft",
    execution_status: str = "not_started",
    analysis_status: str = "data_incomplete",
    finding: str | None = None,
    ranking_eligibility: Mapping[str, Any] | None = None,
    **aliases: Any,
) -> dict[str, Any]:
    """Return the independent v4 design/execution/analysis state artifact."""

    design_status = aliases.get("design", design_status)
    execution_status = aliases.get("execution", execution_status)
    analysis_status = aliases.get("analysis", analysis_status)
    if finding is None:
        finding = aliases.get("finding", "inconclusive")
    allowed = {
        "design": {"draft", "preregistered"},
        "execution": {"not_started", "running", "awaiting_settlement", "terminal"},
        "analysis": {"data_incomplete", "ready", "complete"},
        "finding": {"supported", "not_supported", "inconclusive"},
    }
    values = {
        "design": design_status,
        "execution": execution_status,
        "analysis": analysis_status,
        "finding": finding,
    }
    for name, value in values.items():
        if value not in allowed[name]:
            raise FullReleaseAnalysisError(f"invalid experiment status {name}={value!r}")
    ranking = ranking_eligibility
    if ranking is None:
        ranking = aliases.get("ranking", {})
    if not isinstance(ranking, Mapping):
        raise FullReleaseAnalysisError("ranking eligibility status must be an object")
    return {
        "schema_version": "experiment-status/v1",
        "design": design_status,
        "execution": execution_status,
        "analysis": analysis_status,
        "finding": finding,
        "ranking": dict(ranking),
    }


__all__ = [
    "CATEGORY_WEIGHTS",
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "E_FORMULA_FAMILIES",
    "FORMAL_UTILITY_VERSION",
    "FullReleaseAnalysisError",
    "REGIMES",
    "SYSTEMS",
    "e1_delta_web",
    "e2_decision_tests",
    "e3_leaderboard",
    "e4_calibration",
    "control_experiments",
    "experiment_status",
]
