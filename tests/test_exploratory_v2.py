"""Focused v2 artifact, planning, executor, and metric checks."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmark_harness.analysis.metrics.exploratory import (
    c2_status,
    exploratory_task_utility,
)
from benchmark_harness.experiment import (
    AnalysisRequest,
    AuditReport,
    ExecutionDependencies,
    ExperimentExecutionRequest,
    ManifestError,
    analyze_experiment,
    audit_experiment,
    build_plan,
    load_manifest,
    report_analysis,
    run_experiment,
    validate_manifest_artifacts,
)
from benchmark_harness.systems.fakes import FakeDeepResearchSystem

ROOT = Path(__file__).parents[1]


def test_exploratory_plan_has_reviewed_cell_counts() -> None:
    manifest = load_manifest(ROOT / "configs/exploratory-v2.yaml")
    assert len(build_plan(manifest, selection="smoke").cells) == 180
    assert len(build_plan(manifest, selection="dev").cells) == 1_404
    assert len(build_plan(manifest, selection="test").cells) == 4_698
    assert len(build_plan(manifest, selection="all").cells) == 6_102


def test_fake_e2e_is_binding_aware_and_append_only(tmp_path: Path) -> None:
    manifest = load_manifest(ROOT / "configs/exploratory-v2.yaml")
    plan = build_plan(manifest, selection="smoke")
    task_ids = {cell.task_id for cell in plan.cells}
    rows = tuple(
        {
            "task_id": task_id,
            "prompt": f"Hermetic smoke fixture for {task_id}",
            "seed": {"cutoff_date": "2025-06-06"},
        }
        for task_id in sorted(task_ids)
    )

    def factory(system, regime):
        return FakeDeepResearchSystem(system.descriptor(regime), score=0.5)

    summary = run_experiment(
        ExperimentExecutionRequest(manifest, plan, tmp_path, selection="smoke"),
        dependencies=ExecutionDependencies(system_factory=factory, task_loader=lambda: rows),
    )
    assert summary.status == "complete"
    assert summary.total_cells == summary.completed_cells == 180
    assert summary.attempts_written == 180
    analysis = analyze_experiment(
        AnalysisRequest(
            manifest=manifest,
            plan=plan,
            input_root=tmp_path,
            output_dir=tmp_path / "analysis",
            selection="smoke",
        )
    )
    assert analysis.c2_status == "exploratory_only"
    audit = audit_experiment(
        ExperimentExecutionRequest(manifest, plan, tmp_path, selection="smoke")
    )
    assert isinstance(audit, AuditReport)
    assert audit.status == "valid"
    assert audit.checked_attempts == 180
    c2 = json.loads((tmp_path / "analysis" / "c2.json").read_text())
    assert c2["reasons"] == ["system_count_lt_5", "no_T2", "no_resolved_T3"]
    assert c2["correlations"] is None
    assert report_analysis(tmp_path / "analysis").schema_version == "artifact-manifest/v1"


def test_v2_utility_and_nonranking_c2_gate() -> None:
    utility, reason = exploratory_task_utility(
        "B",
        {"p_up": 1.0},
        {"actual_up": 1.0},
    )
    assert utility == 1.0
    assert reason is None
    c2 = c2_status(("financeharness", "odr", "gpt-researcher"))
    assert c2["status"] == "exploratory_only"
    assert c2["ranking_eligible"] is False
    assert set(c2["reasons"]) == {"system_count_lt_5", "no_T2", "no_resolved_T3"}


def test_v2_validate_fails_closed_on_temporal_hash_drift() -> None:
    manifest = load_manifest(ROOT / "configs/exploratory-v2.yaml")
    drifted = replace(manifest, temporal_index_sha256="0" * 64)
    with pytest.raises(ManifestError, match="temporal index hash mismatch"):
        validate_manifest_artifacts(drifted)
