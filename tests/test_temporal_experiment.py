"""Phase 1 temporal and deterministic experiment-plan coverage."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from benchmark_harness.experiment import (
    AttemptRecord,
    ManifestError,
    append_attempt,
    build_plan,
    load_manifest,
    next_attempt_number,
    read_attempts,
    resume_bindings,
)
from benchmark_harness.experiment.eligibility import evaluate_system_eligibility
from benchmark_harness.systems import BudgetSpec
from benchmark_harness.temporal import (
    TemporalApplicability,
    TemporalIndexRow,
    TemporalStatus,
    assign_temporal_status,
    load_cutoff_registry,
    load_temporal_index,
)
from benchmark_harness.temporal.policy import TemporalLeakageError, assert_no_temporal_leakage


def _row(
    task_id: str,
    *,
    origin: str | None,
    outcome: str | None,
    applicability: TemporalApplicability = TemporalApplicability.FORECAST,
) -> TemporalIndexRow:
    return TemporalIndexRow(
        task_id=task_id,
        temporal_applicability=applicability,
        forecast_origin=date.fromisoformat(origin) if origin else None,
        outcome_available_at=date.fromisoformat(outcome) if outcome else None,
        source_time_band="T2",
        evidence=({"source": "fixture"},),
        review_status="verified",
    )


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (
            _row("t1", origin="2020-01-01", outcome="2023-01-01"),
            TemporalStatus.T1_EXPOSURE_PRONE,
        ),
        (
            _row("t2", origin="2025-02-01", outcome="2025-06-01"),
            TemporalStatus.T2_PASTCAST,
        ),
        (
            _row("t3-pending", origin="2026-01-01", outcome=None),
            TemporalStatus.T3_LIVE_PENDING,
        ),
        (
            _row("t3-resolved", origin="2026-01-01", outcome="2026-03-01"),
            TemporalStatus.T3_LIVE_RESOLVED,
        ),
        (
            _row("quarantine", origin="2024-06-15", outcome="2025-12-15"),
            TemporalStatus.TEMPORAL_QUARANTINE,
        ),
        (
            _row(
                "d", origin=None, outcome=None, applicability=TemporalApplicability.COUNTERFACTUAL
            ),
            TemporalStatus.COUNTERFACTUAL_CONTROL,
        ),
        (
            _row("e", origin=None, outcome=None, applicability=TemporalApplicability.ATEMPORAL),
            TemporalStatus.ATEMPORAL_CONTROL,
        ),
    ],
)
def test_temporal_boundary_rules_are_pure(row: TemporalIndexRow, expected: TemporalStatus) -> None:
    assignment = assign_temporal_status(
        row,
        "2024-01-01",
        "2025-12-01",
        analysis_as_of="2026-04-01",
        guard_days=30,
    )
    assert assignment.status is expected
    assert assignment.source_time_band == "T2"
    assert assignment.input_hashes
    assert_no_temporal_leakage({"task_id": row.task_id, "cutoff_date": "2024-01-01"})


def test_temporal_index_is_unique_reviewed_and_coverage_checked(tmp_path: Path) -> None:
    path = tmp_path / "task-temporal-index.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "temporal_applicability": "forecast",
                    "forecast_origin": "2025-01-01",
                    "outcome_available_at": "2025-02-01",
                    "source_time_band": "T2",
                    "evidence": [],
                    "review_status": "verified",
                }
            )
            for task_id in ("A1-00001", "A1-00002")
        )
        + "\n",
        encoding="utf-8",
    )

    index = load_temporal_index(path, expected_task_ids=("A1-00001", "A1-00002"))
    assert index.row_for("A1-00001").identity_hash
    assert len(index.sha256) == 64
    with pytest.raises(ValueError, match="coverage mismatch"):
        load_temporal_index(path, expected_task_ids=("A1-00001", "missing"))


def test_cutoff_registry_is_versioned_and_does_not_supply_defaults(tmp_path: Path) -> None:
    path = tmp_path / "backbone-cutoffs.v1.yaml"
    path.write_text(
        "schema_version: backbone-cutoffs/v1\n"
        "models:\n"
        "  - backbone_id: provider/model@sha\n"
        "    training_data_cutoff: 2024-06-01\n"
        "    precision: day\n"
        "    source: user_supplied\n"
        "    declared_at: '2026-08-17T00:00:00Z'\n"
        "    notes: fixture\n",
        encoding="utf-8",
    )
    registry = load_cutoff_registry(path)
    assert registry.for_backbone("provider/model@sha").analysis_usable is True
    assert len(registry.sha256) == 64


def _manifest_text() -> str:
    return """schema_version: experiment/v1
experiment_id: phase1-fixture
experiment_as_of: 2025-12-01
claim_scope: exploratory
dataset:
  repo_id: sselaine27/benchmark-research
  revision: e941b978af5f8a38fbeb4670ee2d77671a5b551e
temporal:
  index_path: data/temporal/task-temporal-index.v1.jsonl
  index_sha256: '0000000000000000000000000000000000000000000000000000000000000000'
  cutoff_registry: configs/backbone-cutoffs.v1.yaml
  guard_days: 30
systems:
  - system_id: system-a
    adapter: fixture
    config: configs/fixture.toml
    framework_id: framework-a
    framework_revision: framework-sha
    strategy_id: strategy-a
    strategy_revision: strategy-v1
    backbone_id: provider/model@sha
    backbone_training_cutoff: 2024-01-01
    cutoff_source: user_supplied
    adapter_id: fixture-adapter
    adapter_revision: adapter-v1
    profile: deep_research
    license_status: MIT
  - system_id: system-b
    adapter: fixture
    config: configs/fixture.toml
    framework_id: framework-b
    framework_revision: framework-sha
    strategy_id: strategy-b
    strategy_revision: strategy-v1
    backbone_id: provider/model@sha
    backbone_training_cutoff: 2024-01-01
    cutoff_source: user_supplied
    adapter_id: fixture-adapter
    adapter_revision: adapter-v1
    profile: deep_research
    license_status: MIT
retrieval_regimes: [closed_book, cutoff_controlled, unrestricted_current_web]
trials:
  count: 2
  seed: 7
  order: blocked_randomized
analysis:
  metric_catalog: configs/analysis/metric-catalog.v1.json
  bootstrap_resamples: 100
  alpha: 0.05
"""


def test_manifest_plan_and_resume_are_deterministic_and_score_independent(tmp_path: Path) -> None:
    manifest_path = tmp_path / "experiment.yaml"
    manifest_path.write_text(_manifest_text(), encoding="utf-8")
    manifest = load_manifest(manifest_path)
    plan_a = build_plan(manifest, ["T-2", "T-1"])
    plan_b = build_plan(manifest, ["T-1", "T-2"])

    assert plan_a.plan_hash == plan_b.plan_hash
    assert len(plan_a.cells) == 2 * 2 * 3 * 2
    assert len(plan_a.matched_cell_ids) == len(plan_a.cells)
    assert [cell.execution_order for cell in plan_a.cells] == list(range(len(plan_a.cells)))

    budget = BudgetSpec(4, 3, 2, 5, 1000, 2000, 60.0)
    bindings = plan_a.bindings(manifest, {"system-a": budget, "system-b": budget})
    assert all(binding.attempt_id.endswith("attempt-1") for binding in bindings)

    records = (AttemptRecord(plan_a.cells[0].cell_id, 1, "failed", reason="timeout"),)
    resumed = resume_bindings(plan_a, manifest, {"system-a": budget, "system-b": budget}, records)
    assert resumed[0].attempt_id.endswith("attempt-2")
    assert all(binding.attempt_id.endswith("attempt-1") for binding in resumed[1:])

    ledger = tmp_path / "attempts.jsonl"
    append_attempt(ledger, records[0])
    assert read_attempts(ledger) == records
    assert next_attempt_number(records, plan_a.cells[0].cell_id) == 2
    with pytest.raises(ValueError, match="already exists"):
        append_attempt(ledger, records[0])

    decision = evaluate_system_eligibility(manifest.systems[0])
    assert decision.execution_eligible is True
    assert decision.ranking_eligible is False


def test_attempt_ledger_accepts_historical_and_new_sub_reason_rows(tmp_path: Path) -> None:
    ledger = tmp_path / "attempts.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "cell_id": "historical",
                "attempt": 1,
                "status": "failed",
                "artifact_manifest_sha256": None,
                "reason": "worker_error:RuntimeError",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    historical = read_attempts(ledger)
    assert historical[0].sub_reason is None

    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "cell_id": "legacy-alias",
                    "attempt": 1,
                    "status": "failed",
                    "artifact_manifest_sha256": None,
                    "reason": "worker_error:RuntimeError",
                    "failure_subreason": "odr_search_tool_call_invalid",
                }
            )
            + "\n"
        )
    assert read_attempts(ledger)[1].sub_reason == "odr_search_tool_call_invalid"

    append_attempt(
        ledger,
        AttemptRecord(
            "new",
            1,
            "failed",
            reason="worker_error:RuntimeError",
            sub_reason="odr_search_tool_call_invalid",
        ),
    )

    rows = read_attempts(ledger)
    assert rows[2].sub_reason == "odr_search_tool_call_invalid"


def test_manifest_rejects_missing_backbone_identity(tmp_path: Path) -> None:
    text = _manifest_text().replace("    backbone_id: provider/model@sha\n", "", 1)
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ManifestError, match="missing required identity"):
        load_manifest(path)


def test_temporal_system_payload_rejects_analysis_only_fields() -> None:
    with pytest.raises(TemporalLeakageError, match="time_band"):
        assert_no_temporal_leakage({"tool": {"time_band": "T2"}})
