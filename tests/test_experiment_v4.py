"""Focused contract checks for the additive experiment/v4 release."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from benchmark_harness.analysis.full_release import (
    E_FORMULA_FAMILIES,
    SYSTEMS,
    control_experiments,
    e3_leaderboard,
    e4_calibration,
)
from benchmark_harness.contracts import AdapterResult, canonical_input_hash
from benchmark_harness.experiment import (
    ExecutionDependencies,
    ExperimentExecutionRequest,
    resume_experiment,
    run_experiment,
)
from benchmark_harness.experiment.data_inventory import (
    V4_EXPECTED_RECORD_COUNT,
    build_disposition_ledger,
    compute_inventory,
    validate_b_event_chain,
    validate_d_pairs,
)
from benchmark_harness.experiment.executor import ExecutionError, assert_calc_policy
from benchmark_harness.experiment.manifest import ManifestError, load_manifest
from benchmark_harness.experiment.planner import PlanCell, build_plan
from benchmark_harness.experiment.projector import _frozen_scoring, _parse_projection_response
from benchmark_harness.experiment.v4_api import analyze_full_release
from benchmark_harness.systems.contracts import BudgetSpec, PreflightResult, TaskCapability
from benchmark_harness.systems.fakes import FakeDeepResearchSystem


def test_v4_example_declares_the_fixed_three_system_scope() -> None:
    manifest = load_manifest("configs/full-release-v4.example.yaml")
    assert manifest.schema_version == "experiment/v4"
    assert manifest.experiment_id == "hua40-full-v1"
    assert tuple(system.system_id for system in manifest.systems) == SYSTEMS
    assert manifest.retrieval_regimes == (
        "closed_book",
        "cutoff_controlled",
        "unrestricted_current_web",
    )
    assert manifest.trial_count == 3
    assert manifest.to_dict()["systems"][0]["gates"]["license_resolved"] is False


def test_v4_rejects_two_systems_before_planning() -> None:
    manifest = load_manifest("configs/full-release-v4.example.yaml")
    with pytest.raises(ManifestError, match="exactly financeharness"):
        replace(manifest, systems=manifest.systems[:2])


def test_v4_cohort_selection_and_e_calculation_arms_are_deterministic() -> None:
    manifest = load_manifest("configs/full-release-v4.example.yaml")
    cohort = [
        {"task_id": "A1-0001", "category": "A1", "split": "dev", "smoke": True},
        {"task_id": "E-0001", "category": "E", "split": "test", "smoke": False},
    ]
    manifest = replace(manifest, v4_metadata={**manifest.v4_metadata, "cohort": cohort})
    dev = build_plan(manifest, selection="dev")
    assert len(dev.cells) == 3 * 3 * 3
    assert {cell.calc_policy for cell in dev.cells} == {"calc_on"}
    test = build_plan(manifest, selection="test")
    assert len(test.cells) == 3 * 1 * 2 * 3
    assert {cell.calc_policy for cell in test.cells} == {"calc_on", "calc_off"}
    assert len({cell.cell_id for cell in test.cells}) == len(test.cells)


def test_inventory_is_score_blind_and_records_b_disposition() -> None:
    records = [
        {
            "task_id": "A1-0001",
            "category": "A1",
            "paper_band": "T1",
            "official_temporal_eligible": True,
            "review_status": "reviewed",
            "outcome_available_at_source": "observed_2026-01-01",
            "status": "ready",
        },
        {"task_id": "B-0001", "category": "B", "paper_band": "T2"},
    ]
    inventory = compute_inventory(records, as_of="2026-09-07")
    assert inventory["expected_record_count"] == V4_EXPECTED_RECORD_COUNT
    assert inventory["record_count"] == 2
    assert "prediction" not in inventory
    assert inventory["gap_masks"]["review_status"]["missing_count"] == 1
    ok, reasons = validate_b_event_chain(records[1])
    assert not ok
    assert "missing_event_identity" in reasons
    ledger = build_disposition_ledger(records)
    assert len(ledger.entries) == 2
    assert ledger.to_dict()["ledger_sha256"] == ledger.ledger_sha256


def _leaderboard_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    slices = (
        ("A1", "all"),
        ("A2", "F"),
        ("A2", "T"),
        ("A2", "H"),
        ("B", "earnings"),
        ("B", "macro"),
        ("C", "all"),
    )
    for system_index, system in enumerate(SYSTEMS):
        for slice_index, (category, variant) in enumerate(slices):
            rows.append(
                {
                    "system_id": system,
                    "task_id": f"{category}-{variant}-{system_index}",
                    "category": category,
                    "variant": variant,
                    "retrieval_regime": "cutoff_controlled",
                    "temporal_band": "T2",
                    "formal_utility": 0.2 * system_index + 0.01 * slice_index,
                    "comparable": True,
                    "analysis_eligible": True,
                    "event_cluster_id": f"cluster-{slice_index}",
                    "metric_values": {"analysis_probability": 0.6, "actual_up": 1},
                    "actual_up": 1,
                }
            )
    return rows


def test_leaderboard_and_calibration_preserve_t2_t3_and_invalid_probability_views() -> None:
    rows = _leaderboard_rows()
    leaderboard = e3_leaderboard(rows, system_ids=SYSTEMS, resamples=25)
    assert leaderboard["main_leaderboard"] == "t2"
    assert leaderboard["t2"]["status"] == "ok"
    assert leaderboard["t3"]["status"] == "incomplete_slices"
    assert set(leaderboard["t2"]["system_scores"]) == set(SYSTEMS)
    calibration = e4_calibration(
        [
            *rows,
            {
                "system_id": "financeharness",
                "task_id": "B-invalid",
                "category": "B",
                "variant": "earnings",
                "retrieval_regime": "cutoff_controlled",
                "actual_up": 1,
                "analysis_probability": 1.5,
            },
        ],
        system_ids=SYSTEMS,
        resamples=25,
    )
    assert set(calibration["by_system"]) == set(SYSTEMS)
    assert calibration["by_system"]["financeharness"]["invalid_output_count"] == 1


def test_pit_c_value_usd_millions_alias_is_projection_only() -> None:
    parsed = _parse_projection_response("C", '{"value_usd_millions": 86.31}')
    assert parsed["format_valid"] is True
    assert parsed["predicted_value"] == pytest.approx(86.31)

    utility, frozen, reason = _frozen_scoring(
        "C-PIT-FIXTURE",
        {"raw_text": '{"value_usd_millions": 86.31}'},
        {
            "ground_truth": {"value_usd_millions": 86.31},
            "seed": {"metric_name": "operating_revenue"},
        },
        category="C",
    )
    assert reason is None
    assert utility["exploratory_task_utility_v1"] == pytest.approx(1.0)
    assert frozen["future_value"] == pytest.approx(86.31)


def test_control_report_keeps_d_pairs_and_all_e_families_out_of_t2() -> None:
    rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        rows.extend(
            [
                {
                    "system_id": system,
                    "task_id": f"D-positive-{system}",
                    "category": "D",
                    "matched_pair_id": f"pair-{system}",
                    "metric_values": {"logic_adherence": True},
                },
                {
                    "system_id": system,
                    "task_id": f"D-negative-{system}",
                    "category": "D",
                    "matched_pair_id": f"pair-{system}",
                    "metric_values": {"logic_adherence": False},
                },
            ]
        )
        for family in E_FORMULA_FAMILIES:
            for calc_policy in ("calc_on", "calc_off"):
                rows.append(
                    {
                        "system_id": system,
                        "task_id": f"E-{family}-{system}",
                        "category": "E",
                        "formula_family": family,
                        "calc_policy": calc_policy,
                        "frozen_metrics": {
                            "exact_match_0_2pct": calc_policy == "calc_on",
                            "formula_text_match": True,
                            "unit_match": True,
                        },
                    }
                )
    report = control_experiments(rows, system_ids=SYSTEMS, min_d_pairs=1, min_e_cases=1)
    assert report["status"] == "ok"
    assert report["d"]["financeharness"]["complete_pair_count"] == 1
    assert set(report["e"]["financeharness"]) >= set(E_FORMULA_FAMILIES)


class _FrozenProjectionFixtureSystem:
    """Record-free executor fixture with supplemental B/D/E JSON outputs."""

    def __init__(self, descriptor: Any, outputs: dict[str, str]) -> None:
        self.descriptor = descriptor
        self.outputs = outputs

    def preflight(self, binding: Any) -> PreflightResult:
        return PreflightResult(
            True,
            provenance={"system_id": self.descriptor.system_id, "fixture": True},
        )

    def run_task(
        self, task: Any, context: Any, capability: Any, binding: Any = None
    ) -> AdapterResult:
        raw_text = self.outputs[task.task_id]
        return AdapterResult(
            raw_text=raw_text,
            usage={"input_tokens": 1, "output_tokens": 1},
            latency_seconds=0.0,
            canonical_input_hash=canonical_input_hash(task.prompt),
            effective_input_hash=canonical_input_hash(task.prompt),
            effective_messages=tuple(task.messages),
            effective_request={"messages": task.messages, "fixture": True},
            model_call_count=1,
            tool_calls=(),
            network_events=(),
            trace_events=tuple(context.events),
            trace_artifact_ref="fixture-trace.jsonl",
        )

    def close(self) -> None:
        return None


def test_supplemental_bde_projection_uses_frozen_metrics(tmp_path: Path) -> None:
    """B probability and D/E controls survive projection from raw sealed output."""

    manifest = load_manifest("configs/full-release-v4.example.yaml")
    temporal_index = Path("data/temporal/task-temporal-index.v1.jsonl").resolve()
    manifest = replace(
        manifest,
        temporal_index_path=str(temporal_index),
        temporal_index_sha256=hashlib.sha256(temporal_index.read_bytes()).hexdigest(),
    )
    t3_registry = tmp_path / "t3-registry.jsonl"
    t3_registry.write_text("", encoding="utf-8")
    manifest = replace(
        manifest,
        v4_metadata={
            **manifest.v4_metadata,
            "t3": {
                "path": str(t3_registry),
                "sha256": hashlib.sha256(t3_registry.read_bytes()).hexdigest(),
                "schema": "t3-registry/v1",
            },
        },
    )
    task_rows = [
        {
            "task_id": "B-REPL-EARNINGS-FIXTURE",
            "category": "B",
            "variant": "earnings",
            "prompt": "Return B JSON.",
            "ground_truth": {"actual_direction": "up", "actual_up": 1.0},
        },
        {
            "task_id": "D-REPL-PAIR-FIXTURE-UP",
            "category": "D",
            "matched_pair_id": "D-REPL-PAIR-FIXTURE",
            "history_conflict": True,
            "stipulated_direction": "up",
            "prompt": "Return D JSON.",
            "ground_truth": {"logic_direction": "positive"},
        },
        {
            "task_id": "D-REPL-PAIR-FIXTURE-DOWN",
            "category": "D",
            "matched_pair_id": "D-REPL-PAIR-FIXTURE",
            "history_conflict": True,
            "stipulated_direction": "down",
            "prompt": "Return D JSON.",
            "ground_truth": {"logic_direction": "positive"},
        },
        {
            "task_id": "E-REPL-ARITHMETIC-FIXTURE",
            "category": "E",
            "formula_family": "arithmetic",
            "prompt": "Return E JSON.",
            "ground_truth": {
                "answer_unit": "units",
                "correct_answer": 116.5,
                "correct_formula": "Arithmetic",
            },
        },
    ]
    cohort = [
        {
            "task_id": row["task_id"],
            "category": row["category"],
            "variant": row.get("variant"),
            "matched_pair_id": row.get("matched_pair_id"),
            "paper_band": "T2",
        }
        for row in task_rows
    ]
    manifest = replace(manifest, v4_metadata={**manifest.v4_metadata, "cohort": cohort})
    plan = build_plan(manifest, task_ids=[row["task_id"] for row in task_rows])
    outputs = {
        "B-REPL-EARNINGS-FIXTURE": ('Source [1]\n{"direction":"up","probability_up":0.75}'),
        "D-REPL-PAIR-FIXTURE-UP": '{"direction":"positive","reasoning":"fixture"}',
        "D-REPL-PAIR-FIXTURE-DOWN": '{"direction":"positive","reasoning":"fixture"}',
        "E-REPL-ARITHMETIC-FIXTURE": (
            '{"answer":116.5,"formula_used":"Arithmetic","unit":"units"}'
        ),
    }
    records = {row["task_id"]: row for row in task_rows}

    def factory(system: Any, regime: str) -> _FrozenProjectionFixtureSystem:
        return _FrozenProjectionFixtureSystem(system.descriptor(regime), outputs)

    request = ExperimentExecutionRequest(
        manifest=manifest,
        plan=plan,
        results_root=tmp_path,
        selection="all",
        max_workers=3,
        max_attempts=1,
    )
    summary = run_experiment(
        request,
        dependencies=ExecutionDependencies(system_factory=factory, task_loader=lambda: task_rows),
    )
    assert summary.status == "complete"

    root = tmp_path / manifest.experiment_id / plan.plan_id
    rows = [json.loads(line) for line in (root / "analysis_rows.jsonl").read_text().splitlines()]
    b = next(row for row in rows if row["category"] == "B")
    d = next(row for row in rows if row["task_id"].endswith("-UP"))
    e = next(row for row in rows if row["category"] == "E")

    assert b["frozen_metrics"]["probability_up"] == 0.75
    assert b["metric_values"]["analysis_probability"] == 0.75
    assert b["formal_utility"] == pytest.approx(1.0 - 2.0 * (0.75 - 1.0) ** 2)
    assert d["frozen_metrics"]["logic_adherence"] is True
    assert d["history_conflict"] is True
    assert d["stipulated_direction"] == "up"
    assert e["formula_family"] == "arithmetic"
    assert e["frozen_metrics"]["exact_match_0_2pct"] is True
    assert e["formal_utility"] == 1.0

    artifacts = analyze_full_release(
        manifest,
        plan,
        root,
        tmp_path / "analysis",
        records=records,
    )
    assert artifacts.status["execution"] == "terminal"
    assert artifacts.status["analysis"] == "data_incomplete"


def test_d_controls_pair_three_trials_as_one_complete_pair() -> None:
    rows = []
    for trial in (1, 2, 3):
        rows.extend(
            [
                {
                    "system_id": "financeharness",
                    "task_id": f"D-positive-{trial}",
                    "category": "D",
                    "trial": trial,
                    "matched_pair_id": "pair-1",
                    "frozen_metrics": {"logic_adherence": True},
                },
                {
                    "system_id": "financeharness",
                    "task_id": f"D-negative-{trial}",
                    "category": "D",
                    "trial": trial,
                    "matched_pair_id": "pair-1",
                    "frozen_metrics": {"logic_adherence": False},
                },
            ]
        )

    report = control_experiments(
        rows,
        system_ids=("financeharness",),
        min_d_pairs=1,
        min_e_cases=1,
    )

    pair = report["d"]["financeharness"]["pairs"][0]
    assert pair["n"] == 6
    assert pair["trial_count"] == 3
    assert pair["complete_trial_count"] == 3
    assert pair["complete"] is True
    assert report["d"]["financeharness"]["complete_pair_count"] == 1


def _complete_b_event_record() -> dict[str, object]:
    return {
        "event_id": "event-1",
        "release_timestamp": "2026-01-02T15:00:00+00:00",
        "first_party_url": "https://issuer.example/event-1",
        "event_snapshot_sha256": "a" * 64,
        "pre_price": 100.0,
        "post_price": 105.0,
        "benchmark_pre_price": 200.0,
        "benchmark_post_price": 202.0,
        "price_unit": "USD/share",
        "benchmark_unit": "USD/index",
        "pre_timestamp": "2026-01-02T14:00:00+00:00",
        "post_timestamp": "2026-01-02T16:00:00+00:00",
        "benchmark_pre_timestamp": "2026-01-02T14:00:00+00:00",
        "benchmark_post_timestamp": "2026-01-02T16:00:00+00:00",
        "pre_url": "https://prices.example/pre",
        "post_url": "https://prices.example/post",
        "benchmark_pre_url": "https://prices.example/benchmark-pre",
        "benchmark_post_url": "https://prices.example/benchmark-post",
        "pre_sha256": "b" * 64,
        "post_sha256": "c" * 64,
        "benchmark_pre_sha256": "d" * 64,
        "benchmark_post_sha256": "e" * 64,
    }


def test_b_event_chain_rejects_a_missing_benchmark_price() -> None:
    record = _complete_b_event_record()
    assert validate_b_event_chain(record) == (True, ())
    missing = dict(record)
    missing.pop("benchmark_post_price")

    valid, reasons = validate_b_event_chain(missing)

    assert not valid
    assert "missing_benchmark_post_price" in reasons


def test_cross_market_t1_t2_pair_is_rejected() -> None:
    rows = [
        {
            "task_id": "D-pair-t1",
            "category": "D",
            "time_band": "T1",
            "matched_pair_id": "pair-1",
            "stipulated_direction": "up",
            "market": "NYSE:ABC",
            "context_hash": "context-1",
            "history_conflict": False,
        },
        {
            "task_id": "D-pair-t2",
            "category": "D",
            "time_band": "T2",
            "matched_pair_id": "pair-1",
            "stipulated_direction": "down",
            "market": "NASDAQ:ABC",
            "context_hash": "context-1",
            "history_conflict": False,
        },
    ]

    report = validate_d_pairs(rows, min_pairs=1)

    assert report["valid_pair_count"] == 0
    assert report["mismatches"]["pair-1"] == ["entity_or_market_mismatch"]


def _task_capability(*, allow_calculation: bool) -> TaskCapability:
    return TaskCapability(
        service_url=None,
        service_token=None,
        model_broker_capability=None,
        budget=BudgetSpec(0, 0, 0, 0, 0, 0, 1.0),
        allow_calculation=allow_calculation,
        allow_citations=True,
        require_structured_output=True,
        run_id="run-1",
        trial_id="trial-1",
        trace_mode="metadata_only",
    )


def test_runtime_calc_capability_leakage_fails_closed() -> None:
    cell = PlanCell(
        cell_id="cell-calc-off",
        system_id="financeharness",
        task_id="E-0001",
        retrieval_regime="closed_book",
        trial=1,
        execution_order=0,
        calc_policy="calc_off",
    )

    with pytest.raises(ExecutionError, match="calc capability leakage"):
        assert_calc_policy(cell, _task_capability(allow_calculation=True))

    assert_calc_policy(cell, _task_capability(allow_calculation=False))


def test_e4_retains_an_all_failure_system() -> None:
    rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        for index in range(2):
            row: dict[str, object] = {
                "system_id": system,
                "task_id": f"B-{system}-{index}",
                "category": "B",
                "variant": "earnings",
                "retrieval_regime": "cutoff_controlled",
                "event_cluster_id": f"event-{system}-{index}",
                "actual_up": 1,
            }
            if system != "financeharness":
                row["analysis_probability"] = 1.0
            rows.append(row)

    report = e4_calibration(rows, system_ids=SYSTEMS, resamples=5)

    failed = report["by_system"]["financeharness"]
    assert report["status"] == "ok"
    assert failed["valid_output_count"] == 0
    assert failed["invalid_output_count"] == 2
    assert failed["valid_output_rate"] == 0.0
    assert failed["raw_brier"] is None
    assert failed["operational_brier"] == 1.0


def test_v4_plan_is_unchanged_by_score_based_data_replacement() -> None:
    manifest = load_manifest("configs/full-release-v4.example.yaml")
    cohort = {"task_id": "A1-0001", "category": "A1", "split": "dev", "smoke": True}
    low_score = replace(
        manifest,
        v4_metadata={**manifest.v4_metadata, "cohort": [{**cohort, "score": -1.0}]},
    )
    high_score = replace(
        manifest,
        v4_metadata={**manifest.v4_metadata, "cohort": [{**cohort, "score": 1.0}]},
    )

    low_plan = build_plan(low_score, selection="all")
    high_plan = build_plan(high_score, selection="all")

    assert low_plan.cells == high_plan.cells
    assert [cell.task_id for cell in low_plan.cells] == [cell.task_id for cell in high_plan.cells]


def test_v4_resume_ignores_replaced_score_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "benchmark_harness.experiment.projector.project_execution_artifacts",
        lambda *args, **kwargs: None,
    )
    manifest = load_manifest("configs/full-release-v4.example.yaml")
    manifest = replace(
        manifest,
        v4_metadata={
            **manifest.v4_metadata,
            "cohort": [{"task_id": "A1-0001", "category": "A1", "split": "dev", "smoke": True}],
        },
    )
    plan = build_plan(manifest, selection="smoke")
    task_row = {
        "task_id": "A1-0001",
        "prompt": "Hermetic resume fixture",
        "seed": {"cutoff_date": "2026-01-01"},
        "score": 0.0,
    }

    def factory(system, regime):
        return FakeDeepResearchSystem(system.descriptor(regime), score=0.5)

    dependencies = ExecutionDependencies(
        system_factory=factory,
        task_loader=lambda: (task_row,),
    )
    request = ExperimentExecutionRequest(
        manifest,
        plan,
        tmp_path,
        selection="smoke",
        max_workers=3,
    )
    first = run_experiment(request, dependencies=dependencies)
    task_row["score"] = 1.0
    resumed = resume_experiment(request, dependencies=dependencies)

    assert first.status == "complete"
    assert resumed.status == "complete"
    assert resumed.attempts_written == 0
    assert resumed.completed_cells == len(plan.cells)


def test_v4_rejects_an_extra_system_population() -> None:
    manifest = load_manifest("configs/full-release-v4.example.yaml")
    extra = replace(manifest.systems[0], system_id="extra-system")

    with pytest.raises(ManifestError, match="exactly financeharness"):
        replace(manifest, systems=(*manifest.systems, extra))


def _full_release_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    slices = (
        ("A1", "all"),
        ("A2", "F"),
        ("A2", "T"),
        ("A2", "H"),
        ("B", "earnings"),
        ("B", "macro"),
        ("C", "all"),
    )
    for system_index, system in enumerate(SYSTEMS):
        for slice_index, (category, variant) in enumerate(slices):
            task_id = f"{category}-{variant}"
            common = {
                "system_id": system,
                "task_id": task_id,
                "category": category,
                "variant": variant,
                "temporal_band": "T2",
                "trial": 1,
                "comparable": True,
                "analysis_eligible": True,
                "event_cluster_id": task_id,
                "actual_up": 1,
                "analysis_probability": 0.8,
            }
            for regime, utility in (
                ("closed_book", 0.10),
                ("cutoff_controlled", 0.20),
                ("unrestricted_current_web", 0.40),
            ):
                rows.append(
                    {
                        **common,
                        "retrieval_regime": regime,
                        "formal_utility": utility + 0.01 * system_index,
                        "metric_values": {"analysis_probability": 0.8, "actual_up": 1},
                    }
                )
        for temporal_band, utility in (("T2", 0.55), ("T3", 0.56)):
            rows.append(
                {
                    "system_id": system,
                    "task_id": "E2-proxy",
                    "category": "A1",
                    "variant": "all",
                    "retrieval_regime": "cutoff_controlled",
                    "temporal_band": temporal_band,
                    "trial": 1,
                    "formal_utility": utility + 0.01 * system_index,
                    "comparable": True,
                    "analysis_eligible": True,
                    "event_cluster_id": "E2-proxy",
                }
            )
        for pair_index in range(30):
            pair_id = f"pair-{system}-{pair_index}"
            rows.extend(
                [
                    {
                        "system_id": system,
                        "task_id": f"D-{system}-{pair_index}-positive",
                        "category": "D",
                        "matched_pair_id": pair_id,
                        "metric_values": {"logic_adherence": True},
                    },
                    {
                        "system_id": system,
                        "task_id": f"D-{system}-{pair_index}-negative",
                        "category": "D",
                        "matched_pair_id": pair_id,
                        "metric_values": {"logic_adherence": False},
                    },
                ]
            )
        for family in E_FORMULA_FAMILIES:
            for case_index in range(10):
                for calc_policy in ("calc_on", "calc_off"):
                    rows.append(
                        {
                            "system_id": system,
                            "task_id": f"E-{system}-{family}-{case_index}",
                            "category": "E",
                            "formula_family": family,
                            "calc_policy": calc_policy,
                            "frozen_metrics": {
                                "exact_match_0_2pct": calc_policy == "calc_on",
                                "formula_text_match": True,
                                "unit_match": True,
                            },
                        }
                    )

    return rows


def test_v4_finding_is_derived_from_completed_e1_to_e4(tmp_path: Path) -> None:
    manifest = load_manifest("configs/full-release-v4.example.yaml")
    manifest = replace(
        manifest,
        bootstrap_resamples=5,
        v4_metadata={
            **manifest.v4_metadata,
            "cohort": [{"task_id": "A1-0001", "category": "A1"}],
        },
    )
    plan = build_plan(manifest)
    root = tmp_path / "results"
    root.mkdir()
    (root / "analysis_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _full_release_rows()),
        encoding="utf-8",
    )

    artifacts = analyze_full_release(manifest, plan, root, tmp_path / "analysis")

    assert artifacts.status["analysis"] == "complete"
    assert artifacts.status["finding"] == "supported"
