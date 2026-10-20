"""Phase 2 offline analysis, lineage, and exploratory-gate coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_harness.analysis.contracts import AnalysisError
from benchmark_harness.analysis.ingest import ingest_rows
from benchmark_harness.analysis.lineage import (
    LineageError,
    build_artifact_manifest,
    write_artifact_manifest,
)
from benchmark_harness.analysis.metrics.aggregate import aggregate_metrics
from benchmark_harness.analysis.metrics.catalog import MetricCatalog, MetricSpec
from benchmark_harness.analysis.pipeline import analyze_rows
from benchmark_harness.analysis.reporting.figures_data import write_figure_data
from benchmark_harness.analysis.reporting.tables import write_tables
from benchmark_harness.analysis.statistics.bootstrap import paired_bootstrap
from benchmark_harness.analysis.statistics.multiple_testing import holm_bonferroni


def _catalog() -> MetricCatalog:
    return MetricCatalog(
        schema_version="metric-catalog/v1",
        metrics=(
            MetricSpec(
                name="score",
                source="metrics.score",
                direction="maximize",
                primary=True,
                normalizer={"kind": "fixed", "value": 1.0},
            ),
        ),
    )


def _row(
    system_id: str,
    task_id: str,
    *,
    band: str,
    regime: str,
    score: float,
    category: str = "A1",
    trial: int = 1,
    comparable: bool = True,
    eligible: bool = True,
    ranking_eligible: bool = False,
) -> dict[str, object]:
    return {
        "system_id": system_id,
        "task_id": task_id,
        "category": category,
        "trial": trial,
        "retrieval_regime": regime,
        "temporal_band": band,
        "metric_values": {"score": score},
        "comparable": comparable,
        "analysis_eligible": eligible,
        "ranking_eligible": ranking_eligible,
        "backbone_id": "provider/model@sha",
    }


def _fixture_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, system in enumerate(("system-a", "system-b", "system-c")):
        score = 0.4 + index * 0.1
        rows.extend(
            [
                _row(system, "shared-t2", band="T2", regime="closed_book", score=score),
                _row(system, "shared-t2", band="T2", regime="cutoff_controlled", score=score + 0.1),
                _row(
                    system, "shared-t3", band="T3", regime="cutoff_controlled", score=score + 0.05
                ),
                _row(
                    system,
                    "c1-task",
                    band="T1",
                    regime="closed_book",
                    score=score,
                ),
                _row(
                    system,
                    "c1-task",
                    band="T1",
                    regime="cutoff_controlled",
                    score=score + 0.1,
                ),
                _row(
                    system,
                    "c1-task",
                    band="T1",
                    regime="unrestricted_current_web",
                    score=score + 0.2,
                ),
            ]
        )
    rows.append(
        _row(
            "system-a",
            "control-d",
            band=None,
            regime="closed_book",
            score=0.99,
            category="D",
        )
    )
    rows.append(
        _row(
            "system-a",
            "failed-task",
            band="T2",
            regime="closed_book",
            score=0.0,
            comparable=False,
            eligible=False,
        )
    )
    return rows


def test_analysis_pipeline_is_deterministic_and_marks_c2_exploratory() -> None:
    rows = _fixture_rows()
    result_a = analyze_rows(rows, _catalog(), seed=17, bootstrap_resamples=50)
    result_b = analyze_rows(list(reversed(rows)), _catalog(), seed=17, bootstrap_resamples=50)

    assert json.dumps(result_a, sort_keys=True) == json.dumps(result_b, sort_keys=True)
    assert result_a["c1"]["status"] == "ok"
    assert result_a["c1"]["contrasts"][0]["matched_n"] == 6
    assert result_a["c2"]["status"] == "exploratory_only"
    assert result_a["c2"]["matched_n"] == 3
    assert "system-a" not in result_a["c3"]["ranking_eligible_systems"]
    assert any(item["reason"] == "control_category_exclusion" for item in result_a["exclusions"])
    assert any(item["reason"] == "non_comparable" for item in result_a["exclusions"])


def test_aggregate_keeps_missing_and_control_rows_out_of_zero_scores() -> None:
    result = aggregate_metrics(_fixture_rows(), _catalog(), metric_name="score")
    assert result["attempted_rows"] == len(_fixture_rows())
    assert result["eligible_observations"] == 18
    assert all(item["value"] != 0 for item in result["task_means"])
    assert any(item["reason"] == "control_category_exclusion" for item in result["exclusions"])


def test_aggregate_macro_averages_categories_before_system_scores() -> None:
    rows = [
        _row("system-a", "a1", band="T2", regime="cutoff_controlled", score=1.0),
        _row("system-a", "a2", band="T2", regime="cutoff_controlled", score=1.0),
        _row("system-a", "a3", band="T2", regime="cutoff_controlled", score=1.0),
        _row("system-a", "c1", band="T2", regime="cutoff_controlled", score=0.0, category="C"),
    ]

    result = aggregate_metrics(rows, _catalog(), metric_name="score")

    assert result["category_means"] == [
        {
            "system_id": "system-a",
            "retrieval_regime": "cutoff_controlled",
            "temporal_band": "T2",
            "category": "A1",
            "value": 1.0,
        },
        {
            "system_id": "system-a",
            "retrieval_regime": "cutoff_controlled",
            "temporal_band": "T2",
            "category": "C",
            "value": 0.0,
        },
    ]
    assert result["systems"][0]["by_regime"]["cutoff_controlled"]["T2"] == pytest.approx(0.5)


def test_statistics_are_seeded_and_holm_adjusted() -> None:
    first = paired_bootstrap([1, 2, 3], [0, 1, 1], resamples=100, seed=3)
    second = paired_bootstrap([1, 2, 3], [0, 1, 1], resamples=100, seed=3)
    assert first == second
    assert first["mean_difference"] == pytest.approx(4 / 3)
    assert holm_bonferroni({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}


def test_lineage_manifest_is_required_and_tamper_evident(tmp_path: Path) -> None:
    rows_path = tmp_path / "analysis_rows.jsonl"
    rows_path.write_text(json.dumps(_fixture_rows()[0]) + "\n", encoding="utf-8")
    manifest = build_artifact_manifest(tmp_path, lineage={"plan_hash": "plan-v1"})
    write_artifact_manifest(tmp_path, manifest)
    rows, loaded = ingest_rows(tmp_path, expected_lineage={"plan_hash": "plan-v1"})
    assert len(rows) == 1
    assert loaded.lineage["plan_hash"] == "plan-v1"

    rows_path.write_text(rows_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(LineageError, match="hash drift"):
        ingest_rows(tmp_path)


def test_analysis_reports_write_separate_deterministic_outputs(tmp_path: Path) -> None:
    result = analyze_rows(_fixture_rows(), _catalog(), seed=17, bootstrap_resamples=10)
    paths = write_tables(tmp_path, result)
    figure_path = write_figure_data(tmp_path, result)
    assert {path.name for path in paths} == {
        "c1_contrasts.csv",
        "c2_system_scores.csv",
        "c3_system_scores.csv",
    }
    assert figure_path.name == "figures-data.json"
    assert "exploratory_only" in figure_path.read_text(encoding="utf-8")


def test_duplicate_analysis_cells_fail_closed() -> None:
    rows = _fixture_rows()
    with pytest.raises(AnalysisError, match="duplicate analysis cell"):
        analyze_rows(rows + [rows[0]], _catalog())
