"""GPT-4.1 live-pack coverage: geometry, quality gating, fail-closed live runtime."""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from benchmark_harness.analysis.metrics.exploratory import c2_status
from benchmark_harness.experiment import (
    AnalysisRequest,
    ExecutionDependencies,
    ExperimentExecutionRequest,
    ManifestError,
    analyze_experiment,
    build_plan,
    load_manifest,
    run_experiment,
    validate_manifest_artifacts,
)
from benchmark_harness.experiment.live_model import (
    LiveRuntimeError,
    derive_model_seed,
    resolve_live_model_profile,
    validate_live_runtime,
)
from benchmark_harness.experiment.projector import _frozen_scoring
from benchmark_harness.systems.fakes import FakeDeepResearchSystem
from benchmark_harness.temporal import assign_temporal_status, load_temporal_index

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "configs/exploratory-gpt41-20260817-v1.yaml"
GPT41_BACKBONE = "openai:gpt-4.1-2025-04-14"


def _manifest():
    return load_manifest(MANIFEST)


def _band_counts(selection: str = "all") -> Counter:
    manifest = _manifest()
    plan = build_plan(manifest, selection=selection)
    temporal = load_temporal_index(manifest.source_path.parent / manifest.temporal_index_path)
    cutoff = date(2024, 6, 1)
    counter: Counter = Counter()
    for cell in plan.cells:
        assignment = assign_temporal_status(
            temporal.row_for(cell.task_id),
            cutoff,
            manifest.experiment_as_of,
            analysis_as_of=manifest.analysis_as_of or manifest.experiment_as_of,
            guard_days=manifest.guard_days,
        )
        counter[(assignment.paper_band, assignment.analysis_eligible)] += 1
    return counter


def test_gpt41_manifest_validates_with_exact_snapshot_identity() -> None:
    manifest = _manifest()
    artifacts = validate_manifest_artifacts(manifest)
    assert artifacts["temporal_index_rows"] == 339
    assert artifacts["cutoff_registry_sha256"] == manifest.v2_metadata["cutoff_registry_sha256"]
    assert {system.backbone_id for system in manifest.systems} == {GPT41_BACKBONE}
    assert {system.backbone_training_cutoff for system in manifest.systems} == {date(2024, 6, 1)}
    from benchmark_harness.temporal import load_cutoff_registry

    registry = load_cutoff_registry(manifest.source_path.parent / manifest.cutoff_registry_path)
    cutoff = registry.for_backbone(GPT41_BACKBONE)
    assert cutoff.training_data_cutoff == date(2024, 6, 1)
    assert cutoff.precision == "day"
    model = manifest.v2_metadata["model"]
    assert model["api_model_name"] == "gpt-4.1-2025-04-14"
    assert model["mode"] == "openai_compatible"
    assert "reasoning_effort" not in model


def test_gpt41_plan_has_reviewed_5886_cell_geometry() -> None:
    manifest = _manifest()
    assert len(build_plan(manifest, selection="all").cells) == 5_886
    assert len(build_plan(manifest, selection="dev").cells) == 1_350
    assert len(build_plan(manifest, selection="test").cells) == 4_536
    cells = build_plan(manifest, selection="all").cells
    assert len({cell.task_id for cell in cells}) == 339
    controls = [
        cell for cell in cells if cell.task_id.startswith("D-") or cell.task_id.startswith("E-")
    ]
    assert controls and all(cell.retrieval_regime == "closed_book" for cell in controls)
    assert sum(1 for cell in cells if cell.retrieval_regime == "cutoff_controlled") == 2_835


def test_gpt41_band_geometry_matches_pm_table() -> None:
    bands = _band_counts("all")
    assert bands[(None, False)] == 5_886 - 216 - 684
    assert bands[("T1", True)] == 12 * 3 * 2 * 3
    eligible_t2 = sum(
        count for (band, eligible), count in bands.items() if band == "T2" and eligible
    )
    raw_t2 = sum(count for (band, _eligible), count in bands.items() if band == "T2")
    assert raw_t2 == 38 * 3 * 2 * 3
    assert eligible_t2 == 36 * 3 * 2 * 3
    # Contaminated A2-H tasks stay T2 by date but are quality-excluded.
    assert bands[("T2", False)] == 2 * 3 * 2 * 3


def test_quality_audit_is_anded_with_band_eligibility() -> None:
    manifest = _manifest()
    temporal = load_temporal_index(manifest.source_path.parent / manifest.temporal_index_path)
    contaminated = temporal.row_for("A2H-00001")
    assignment = assign_temporal_status(
        contaminated,
        date(2024, 6, 1),
        manifest.experiment_as_of,
        analysis_as_of=manifest.analysis_as_of or manifest.experiment_as_of,
        guard_days=manifest.guard_days,
    )
    assert assignment.paper_band == "T2"
    assert assignment.band_eligible is True
    assert assignment.analysis_eligible is False
    assert assignment.quality_audit_reasons == ("post_origin_seed_contamination",)
    clean = temporal.row_for("A2F-00001")
    clean_assignment = assign_temporal_status(
        clean,
        date(2024, 6, 1),
        manifest.experiment_as_of,
        analysis_as_of=manifest.analysis_as_of or manifest.experiment_as_of,
        guard_days=manifest.guard_days,
    )
    assert clean_assignment.analysis_eligible is True


def test_c2_reasons_derive_from_actual_rows() -> None:
    gate = c2_status(("financeharness", "odr", "gpt-researcher"), has_t2=True, resolved_t3=False)
    assert gate["status"] == "exploratory_only"
    assert gate["reasons"] == ["system_count_lt_5", "no_resolved_T3"]
    assert gate["correlations"] is None


def test_live_model_resolution_fails_closed_without_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    monkeypatch.delenv("BENCHMARK_MODEL_API_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_MODEL_NAME", raising=False)
    with pytest.raises(LiveRuntimeError, match="BENCHMARK_MODEL_API_KEY"):
        resolve_live_model_profile(manifest, cell_id="cell-x")
    monkeypatch.setenv("BENCHMARK_MODEL_API_KEY", "sk-test")
    with pytest.raises(LiveRuntimeError, match="alias is forbidden"):
        monkeypatch.setenv("BENCHMARK_MODEL_NAME", "gpt-4.1")
        resolve_live_model_profile(manifest, cell_id="cell-x")
    monkeypatch.delenv("BENCHMARK_MODEL_NAME", raising=False)
    with pytest.raises(LiveRuntimeError, match="https provider URL"):
        monkeypatch.setenv("BENCHMARK_MODEL_ENDPOINT", "mock://model")
        resolve_live_model_profile(manifest, cell_id="cell-x")
    monkeypatch.delenv("BENCHMARK_MODEL_ENDPOINT", raising=False)
    profile = resolve_live_model_profile(manifest, cell_id="cell-x")
    assert profile.mode == "openai_compatible"
    assert profile.model_name == "gpt-4.1-2025-04-14"
    assert profile.public_dict()["endpoint"].startswith("https://")
    assert "sk-test" not in json.dumps(profile.public_dict())
    assert profile.seed == derive_model_seed(manifest.seed, "cell-x")
    with pytest.raises(LiveRuntimeError, match="BENCHMARK_FINANCE_SOURCE"):
        validate_live_runtime(
            manifest,
            tuple(build_plan(manifest, selection="smoke").cells),
            search_service_check=False,
        )


def test_frozen_scoring_materializes_utility_without_zero_fill() -> None:
    record = {"ground_truth": {"future_value": 100.0}}
    metrics, frozen, reason = _frozen_scoring(
        "C-00001", {"raw_text": '{"predicted_value": 105}'}, record
    )
    assert reason is None
    assert metrics["exploratory_task_utility_v1"] == pytest.approx(0.95)
    assert isinstance(frozen, dict) and frozen
    _, _, invalid = _frozen_scoring("C-00001", {"raw_text": '{"nope": 1}'}, record)
    assert invalid == "parse_invalid"
    _, _, missing = _frozen_scoring("C-00001", {"raw_text": '{"predicted_value": 5}'}, None)
    assert missing == "dataset_record_unavailable_for_frozen_scoring"


def test_threaded_executor_terminalizes_every_smoke_cell(tmp_path: Path) -> None:
    manifest = _manifest()
    plan = build_plan(manifest, selection="smoke")
    assert len(plan.cells) == 162
    task_ids = {cell.task_id for cell in plan.cells}
    rows = tuple(
        {
            "task_id": task_id,
            "prompt": f"Hermetic gpt41 smoke fixture for {task_id}",
            "seed": {"cutoff_date": "2024-06-01"},
        }
        for task_id in sorted(task_ids)
    )

    def factory(system, regime, _search_handler=None):
        return FakeDeepResearchSystem(system.descriptor(regime), score=0.5)

    summary = run_experiment(
        ExperimentExecutionRequest(
            manifest, plan, tmp_path, selection="smoke", max_workers=3, max_attempts=3
        ),
        dependencies=ExecutionDependencies(
            system_factory=factory,
            task_loader=lambda: rows,
            service_identity_factory=lambda cell: {
                "retrieval_regime": cell.retrieval_regime,
                "service_cohort_sha256": "d" * 64,
                "instance_id": "service-instance",
            },
        ),
    )
    assert summary.status == "complete"
    assert summary.total_cells == summary.completed_cells == 162
    assert summary.pending_cells == 0
    summary_path = tmp_path / manifest.experiment_id / plan.plan_id / "execution-summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(payload["terminal_cells"]) == {cell.cell_id for cell in plan.cells}
    assert set(payload["terminal_cells"].values()) == {"succeeded"}
    ledger = (tmp_path / manifest.experiment_id / plan.plan_id / "attempts.jsonl").read_text()
    assert len(ledger.splitlines()) == 162
    first_cell = plan.cells[0]
    first_artifact = (
        tmp_path
        / manifest.experiment_id
        / plan.plan_id
        / "cells"
        / first_cell.cell_id
        / "attempt-1.json"
    )
    first_payload = json.loads(first_artifact.read_text(encoding="utf-8"))
    assert first_payload["service_identity"]["service_cohort_sha256"] == "d" * 64
    assert first_payload["service_identity"]["instance_id"] == "service-instance"


def test_cutoff_controlled_cells_receive_task_scoped_sessions(tmp_path: Path) -> None:
    import hashlib
    from types import SimpleNamespace

    manifest = _manifest()
    plan = build_plan(manifest, selection="smoke")
    task_ids = {cell.task_id for cell in plan.cells}
    rows = tuple(
        {
            "task_id": task_id,
            "prompt": f"Hermetic gpt41 session fixture for {task_id}",
            "seed": {"cutoff_date": "2024-06-01"},
        }
        for task_id in sorted(task_ids)
    )
    opened: list[object] = []

    class FakeSearchSession:
        def __init__(self, task_id: str, cutoff_date: str, principal: str) -> None:
            self.descriptor = SimpleNamespace(
                session_id="sess-" + task_id,
                task_id=task_id,
                profile="deep_research",
                baseline=principal,
                cutoff_date=cutoff_date,
                policy_version="deep_research/v2",
                task_token="capability-token-1234567890abcdef",
            )
            self._receipts: list[object] = []
            self.close_result = None
            self.closed_status = None

        @property
        def receipts(self):
            return tuple(self._receipts)

        def search_web(self, query: str):
            self._receipts.append(
                SimpleNamespace(receipt_sha256=hashlib.sha256(query.encode()).hexdigest())
            )
            return [{"url": "https://example.com/a", "title": "t", "content": "c"}]

        def close(self, *, status: str = "completed"):
            self.closed_status = status
            self.close_result = {"status": status}

    def session_factory(binding, task, system_id):
        if system_id == "financeharness":
            return None
        if binding.retrieval_regime != "cutoff_controlled":
            return None
        session = FakeSearchSession(task.task_id, task.cutoff_date or "2024-06-01", system_id)
        opened.append(session)
        return session

    def factory(system, regime, search_handler=None):
        return FakeDeepResearchSystem(system.descriptor(regime), score=0.5)

    expected_sessions = sum(
        1
        for cell in plan.cells
        if cell.retrieval_regime == "cutoff_controlled" and cell.system_id != "financeharness"
    )
    summary = run_experiment(
        ExperimentExecutionRequest(
            manifest, plan, tmp_path, selection="smoke", max_workers=3, max_attempts=3
        ),
        dependencies=ExecutionDependencies(
            system_factory=factory,
            task_loader=lambda: rows,
            search_session_factory=session_factory,
        ),
    )
    assert summary.status == "complete"
    assert len(opened) == expected_sessions
    assert all(session.closed_status == "completed" for session in opened)
    controlled = next(
        cell
        for cell in plan.cells
        if cell.retrieval_regime == "cutoff_controlled" and cell.system_id == "odr"
    )
    artifact_path = (
        tmp_path
        / manifest.experiment_id
        / plan.plan_id
        / "cells"
        / controlled.cell_id
        / "attempt-1.json"
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["search_session"]["session_id"] == "sess-" + controlled.task_id
    assert payload["search_session"]["close_status"] == "completed"
    assert payload["search_session"]["profile"] == "deep_research"
    assert payload["search_session"]["search_principal"] == "odr"
    assert payload["model_provenance"]["mode"] == "mock"


def test_search_principal_drift_rejects_cell_before_execution(tmp_path: Path) -> None:
    from types import SimpleNamespace

    manifest = _manifest()
    plan = build_plan(manifest, selection="smoke")
    task_ids = {cell.task_id for cell in plan.cells}
    rows = tuple(
        {
            "task_id": task_id,
            "prompt": f"Hermetic gpt41 drift fixture for {task_id}",
            "seed": {"cutoff_date": "2024-06-01"},
        }
        for task_id in sorted(task_ids)
    )

    class WrongPrincipalSession:
        def __init__(self, task_id: str, cutoff_date: str) -> None:
            self.descriptor = SimpleNamespace(
                session_id="sess-wrong",
                task_id=task_id,
                profile="deep_research",
                baseline="financeharness",
                cutoff_date=cutoff_date,
                policy_version="deep_research/v2",
                task_token="capability-token-1234567890abcdef",
            )
            self.close_result = None
            self.closed_status = None

        @property
        def receipts(self):
            return ()

        def close(self, *, status: str = "completed"):
            self.closed_status = status
            self.close_result = {"status": status}

    def session_factory(binding, task, system_id):
        if system_id == "financeharness" or binding.retrieval_regime != "cutoff_controlled":
            return None
        return WrongPrincipalSession(task.task_id, task.cutoff_date or "2024-06-01")

    def factory(system, regime, _search_handler=None):
        return FakeDeepResearchSystem(system.descriptor(regime), score=0.5)

    summary = run_experiment(
        ExperimentExecutionRequest(
            manifest, plan, tmp_path, selection="smoke", max_workers=3, max_attempts=1
        ),
        dependencies=ExecutionDependencies(
            system_factory=factory,
            task_loader=lambda: rows,
            search_session_factory=session_factory,
        ),
    )
    ledger = (tmp_path / manifest.experiment_id / plan.plan_id / "attempts.jsonl").read_text(
        encoding="utf-8"
    )
    assert summary.status == "exhausted"
    assert "search principal mismatch" in ledger
    drifted = next(
        cell
        for cell in plan.cells
        if cell.retrieval_regime == "cutoff_controlled" and cell.system_id == "odr"
    )
    drifted_rows = [
        json.loads(line)
        for line in ledger.splitlines()
        if json.loads(line)["cell_id"] == drifted.cell_id
    ]
    assert drifted_rows and drifted_rows[0]["status"] == "failed"
    assert "search principal mismatch" in (drifted_rows[0]["reason"] or "")


def test_live_preflight_rejects_stale_service_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark_harness.experiment.live_model import _verify_service_identity
    from benchmark_harness.search_service.api import SEARCH_SERVICE_PROTOCOL
    from benchmark_harness.search_service.identities import service_manifest

    for name in (
        "SEARCH_FILTER_MODEL_ENDPOINT",
        "SEARCH_FILTER_MODEL_NAME",
        "SEARCH_FILTER_MODEL_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    local = service_manifest()
    matching = {
        "protocol": SEARCH_SERVICE_PROTOCOL,
        "api_contract_sha256": local.api_contract_sha256,
        "implementation_manifest_sha256": local.implementation_manifest_sha256,
        "capability_sha256": local.capability_sha256,
        "service_cohort_sha256": local.service_cohort_sha256,
    }
    _verify_service_identity(matching)
    stale = {
        **matching,
        "implementation_manifest_sha256": "0" * 64,
    }
    with pytest.raises(LiveRuntimeError, match="identity mismatch"):
        _verify_service_identity(stale)
    incomplete = dict(matching)
    incomplete.pop("service_cohort_sha256")
    with pytest.raises(LiveRuntimeError, match="missing identity hashes"):
        _verify_service_identity(incomplete)


def test_live_service_identity_binds_filter_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark_harness.experiment.live_model import _verify_service_identity
    from benchmark_harness.search_service.api import SEARCH_SERVICE_PROTOCOL
    from benchmark_harness.search_service.config import SearchServiceConfig
    from benchmark_harness.search_service.identities import (
        filter_model_identity,
        service_manifest,
    )
    from benchmark_harness.search_service.server import _filter_profile

    monkeypatch.setenv("SEARCH_FILTER_MODEL_ENDPOINT", "https://filter.example/v1")
    monkeypatch.setenv("SEARCH_FILTER_MODEL_NAME", "filter-model")
    monkeypatch.setenv("SEARCH_FILTER_MODEL_API_KEY", "filter-secret")
    profile = _filter_profile(SearchServiceConfig())
    assert profile is not None
    filter_model = filter_model_identity(profile.public_dict())
    local = service_manifest(filter_model=filter_model)
    identity = _verify_service_identity(
        {
            "protocol": SEARCH_SERVICE_PROTOCOL,
            **local.as_dict(),
            "instance_id": "instance-1",
        }
    )
    assert identity["filter_model"]["max_tokens"] == 2048
    assert identity["filter_model"]["temperature"] == 0.0
    assert "filter-secret" not in json.dumps(identity)


def test_analyze_record_fallback_fails_closed_or_reuses_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmark_harness.dataset_sources.huggingface import HuggingFaceDatasetSource

    manifest = _manifest()
    plan = build_plan(manifest, selection="smoke")
    task_ids = {cell.task_id for cell in plan.cells}
    rows = tuple(
        {
            "task_id": task_id,
            "prompt": f"Hermetic gpt41 analyze fixture for {task_id}",
            "seed": {"cutoff_date": "2024-06-01"},
        }
        for task_id in sorted(task_ids)
    )

    def factory(system, regime, _search_handler=None):
        return FakeDeepResearchSystem(system.descriptor(regime), score=0.5)

    def run_fake(root: Path) -> None:
        run_experiment(
            ExperimentExecutionRequest(
                manifest, plan, root, selection="smoke", max_workers=3, max_attempts=3
            ),
            dependencies=ExecutionDependencies(system_factory=factory, task_loader=lambda: rows),
        )

    def _boom(_self: object) -> None:
        raise RuntimeError("no hub cache")

    monkeypatch.setattr(HuggingFaceDatasetSource, "load_records", _boom)

    # Records unavailable and no existing projection -> fail closed.
    root_a = tmp_path / "a"
    run_fake(root_a)
    exec_root_a = root_a / manifest.experiment_id / plan.plan_id
    (exec_root_a / "analysis_rows.jsonl").unlink()
    (exec_root_a / "artifact_manifest.json").unlink()
    with pytest.raises(ManifestError, match="fails closed"):
        analyze_experiment(
            AnalysisRequest(
                manifest=manifest,
                plan=plan,
                input_root=root_a,
                output_dir=tmp_path / "out-a",
                selection="smoke",
            )
        )

    # Records unavailable but an existing projection is present -> reuse it.
    root_b = tmp_path / "b"
    run_fake(root_b)
    artifacts = analyze_experiment(
        AnalysisRequest(
            manifest=manifest,
            plan=plan,
            input_root=root_b,
            output_dir=tmp_path / "out-b",
            selection="smoke",
        )
    )
    assert artifacts.c2_status == "exploratory_only"
