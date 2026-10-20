"""Coverage for the isolated dated GPT-4.1 mini exploratory pack."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.experiment import (
    AnalysisRequest,
    ExecutionContractError,
    ExecutionDependencies,
    ExperimentExecutionRequest,
    analyze_experiment,
    audit_experiment,
    build_execution_contract,
    build_plan,
    load_manifest,
    load_plan,
    resume_experiment,
    run_experiment,
    validate_manifest_artifacts,
)
from benchmark_harness.experiment.live_model import (
    API_KEY_ENV,
    ENDPOINT_ENV,
    NAME_ENV,
    LiveRuntimeError,
    resolve_live_model_profile,
    validate_live_runtime,
)
from benchmark_harness.external_harnesses.financeharness.adapter import (
    DeepModelBroker,
    FinanceHarnessBoundaryError,
)
from benchmark_harness.systems.contracts import BudgetSpec
from benchmark_harness.systems.external_stdio import BudgetLedger
from benchmark_harness.systems.fakes import FakeDeepResearchSystem
from benchmark_harness.systems.model_broker import HostModelBroker, HostModelBrokerError
from benchmark_harness.temporal import (
    assign_temporal_status,
    load_cutoff_registry,
    load_temporal_index,
)

ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "configs/exploratory-gpt41mini-20260817-v1.yaml"
PLAN_PATH = ROOT / "configs/exploratory-plan-gpt41mini-20260817-v1.json"
PREVIOUS_PACK_PATH = ROOT / "configs/exploratory-gpt41-20260817-v1.yaml"
PREVIOUS_STATUS_PATH = ROOT / "configs/exploratory-gpt41-20260817-v1.status.yaml"
MINI_WIRE_MODEL = "gpt-4.1-mini"
MINI_SNAPSHOT_MODEL = "gpt-4.1-mini-2025-04-14"
MINI_MODEL = MINI_WIRE_MODEL
MINI_BACKBONE = "openai:gpt-4.1-mini-2025-04-14"
VALID_ENDPOINT = "https://mini-resource.services.ai.azure.com/openai/v1"


def _manifest():
    return load_manifest(MANIFEST_PATH)


def _set_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(API_KEY_ENV, "test-provider-key")
    monkeypatch.setenv(NAME_ENV, MINI_MODEL)
    monkeypatch.setenv(ENDPOINT_ENV, VALID_ENDPOINT)
    for env_name, directory in (
        ("BENCHMARK_FINANCE_SOURCE", "finance"),
        ("BENCHMARK_ODR_SOURCE", "odr"),
        ("BENCHMARK_GPT_SOURCE", "gpt"),
    ):
        source = tmp_path / directory
        source.mkdir()
        monkeypatch.setenv(env_name, str(source))
    token_file = tmp_path / "search-admin.token"
    token_file.write_text("test-search-token\n", encoding="utf-8")
    monkeypatch.setenv("BENCHMARK_SEARCH_SERVICE_ADMIN_TOKEN_FILE", str(token_file))


def _profile() -> ResolvedModelProfile:
    return ResolvedModelProfile(
        mode="mock",
        endpoint="mock://local",
        model_name=MINI_WIRE_MODEL,
        api_key=None,
        temperature=0.0,
        max_tokens=128,
        timeout_seconds=1.0,
        retries=0,
        seed=None,
        snapshot_model_name=MINI_SNAPSHOT_MODEL,
        backbone_id=MINI_BACKBONE,
        wire_name_binding="user_authorized_azure_wire_mapping/v1",
    )


def _context(profile: ResolvedModelProfile):
    from benchmark_harness.contracts import RunContext

    return RunContext(
        run_id="mini-pack-test",
        framework="test",
        adapter_version="test",
        model=profile,
        trace_content_mode="hash_only",
    )


def _response(model: str) -> dict[str, object]:
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": "answer"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def test_mini_manifest_uses_exact_snapshot_and_immutable_v06_inputs() -> None:
    manifest = _manifest()
    artifacts = validate_manifest_artifacts(manifest)
    assert manifest.experiment_id == "exploratory-339-gpt41mini-20260817-v1"
    assert manifest.dataset_revision == "e941b978af5f8a38fbeb4670ee2d77671a5b551e"
    assert manifest.v2_metadata["dataset"]["record_count"] == 339
    assert artifacts["temporal_index_rows"] == 339
    assert {system.backbone_id for system in manifest.systems} == {MINI_BACKBONE}
    assert {system.backbone_training_cutoff for system in manifest.systems} == {date(2024, 6, 1)}
    model = manifest.v2_metadata["model"]
    assert model["provider"] == "azure_openai"
    assert model["api_model_name"] == MINI_WIRE_MODEL
    assert model["snapshot_model_name"] == MINI_SNAPSHOT_MODEL
    assert model["api_model_name_semantics"] == "azure_request_model"
    assert model["wire_name_binding"] == "user_authorized_azure_wire_mapping/v1"
    assert model["backbone_id"] == MINI_BACKBONE
    assert model["endpoint"].endswith("/openai/v1")
    assert manifest.v2_metadata["policy"]["ranking_eligible"] is False
    assert manifest.v2_metadata["policy"]["score_selection"] == "forbidden"
    provenance = manifest.v2_metadata["provenance"]
    assert provenance["backbone_id"] == MINI_BACKBONE
    assert provenance["snapshot_model_name"] == MINI_SNAPSHOT_MODEL
    assert provenance["wire_model_name"] == MINI_WIRE_MODEL
    assert provenance["wire_name_binding"] == "user_authorized_azure_wire_mapping/v1"
    registry = load_cutoff_registry(manifest.source_path.parent / manifest.cutoff_registry_path)
    cutoff = registry.for_backbone(MINI_BACKBONE)
    assert cutoff.training_data_cutoff == date(2024, 6, 1)
    assert cutoff.source.endswith("gpt-4.1-mini")


def test_previous_gpt41_pack_is_retained_but_marked_abandoned() -> None:
    assert PREVIOUS_PACK_PATH.is_file()
    status = PREVIOUS_STATUS_PATH.read_text(encoding="utf-8")
    assert "experiment_id: exploratory-339-gpt41-20260817-v1" in status
    assert "status: abandoned" in status
    assert "superseded_by: exploratory-339-gpt41mini-20260817-v1" in status


def test_mini_plan_file_and_smoke_selection_have_reviewed_geometry() -> None:
    manifest = _manifest()
    plan = load_plan(PLAN_PATH, expected_manifest_hash=manifest.manifest_hash)
    rebuilt = build_plan(manifest, selection="all")
    assert plan.to_dict() == rebuilt.to_dict()
    assert plan.plan_id == "exploratory-339-gpt41mini-20260817-v1-9b81a7d7893e"
    assert plan.plan_hash == "9b81a7d7893e23e0a03c66321bcb4e1097eeb31d1a2bf3ef7378b3bb4def2d2f"
    assert plan.plan_hash != "a3b9f575c1f1abccec4180c940938130e604fe263b0b8db6d5e23363e2a279d0"
    assert len(plan.cells) == 5_886
    assert len({cell.cell_id for cell in plan.cells}) == 5_886
    assert len([cell for cell in plan.cells if cell.smoke]) == 162
    assert len(build_plan(manifest, selection="smoke").cells) == 162
    assert (
        hashlib.sha256(("\n".join(cell.cell_id for cell in plan.cells) + "\n").encode()).hexdigest()
        == "1857a170b2b94aa32c84c381d0f22ca5b169a6c52d6b402c717e2bb75ed5a4f7"
    )
    smoke_cells = build_plan(manifest, selection="smoke").cells
    assert (
        hashlib.sha256(
            ("\n".join(cell.cell_id for cell in smoke_cells) + "\n").encode()
        ).hexdigest()
        == "225a4c1f4eb1dd65af2424a112d425d7722f33b6482c50d41a666174fd9d47dc"
    )
    assert (
        len([cell for cell in plan.cells if cell.retrieval_regime == "cutoff_controlled"]) == 2_835
    )
    controls = [
        cell
        for cell in plan.cells
        if cell.task_id.startswith("D-") or cell.task_id.startswith("E-")
    ]
    assert len(controls) == 216
    assert all(cell.retrieval_regime == "closed_book" for cell in controls)


def test_mini_temporal_geometry_matches_the_v06_cutoff() -> None:
    manifest = _manifest()
    plan = build_plan(manifest, selection="all")
    temporal = load_temporal_index(manifest.source_path.parent / manifest.temporal_index_path)
    registry = load_cutoff_registry(manifest.source_path.parent / manifest.cutoff_registry_path)
    cutoff = registry.for_backbone(MINI_BACKBONE).training_data_cutoff
    bands: Counter[tuple[str | None, bool]] = Counter()
    for cell in plan.cells:
        assignment = assign_temporal_status(
            temporal.row_for(cell.task_id),
            cutoff,
            manifest.experiment_as_of,
            analysis_as_of=manifest.analysis_as_of or manifest.experiment_as_of,
            guard_days=manifest.guard_days,
        )
        bands[(assignment.paper_band, assignment.analysis_eligible)] += 1
    assert bands[None, False] == 4_986
    assert bands["T1", True] == 12 * 3 * 2 * 3
    assert bands["T2", True] == 36 * 3 * 2 * 3
    assert bands["T2", False] == 2 * 3 * 2 * 3


def test_mini_live_preflight_requires_exact_name_key_and_azure_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    monkeypatch.setenv(NAME_ENV, MINI_WIRE_MODEL)
    monkeypatch.setenv(ENDPOINT_ENV, VALID_ENDPOINT)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(LiveRuntimeError, match=API_KEY_ENV):
        resolve_live_model_profile(manifest, cell_id="preflight")

    monkeypatch.setenv(API_KEY_ENV, "test-provider-key")
    monkeypatch.delenv(NAME_ENV, raising=False)
    with pytest.raises(LiveRuntimeError, match=NAME_ENV):
        resolve_live_model_profile(manifest, cell_id="preflight")

    monkeypatch.setenv(NAME_ENV, MINI_WIRE_MODEL)
    profile = resolve_live_model_profile(manifest, cell_id="preflight")
    assert profile.model_name == MINI_WIRE_MODEL
    assert profile.snapshot_model_name == MINI_SNAPSHOT_MODEL
    assert profile.backbone_id == MINI_BACKBONE
    assert profile.wire_name_binding == "user_authorized_azure_wire_mapping/v1"


@pytest.mark.parametrize(
    "wire_name", (MINI_SNAPSHOT_MODEL, "gpt-4.1", "gpt-4.1-2025-04-14", "mini")
)
def test_mini_live_preflight_rejects_non_wire_names(
    monkeypatch: pytest.MonkeyPatch, wire_name: str
) -> None:
    manifest = _manifest()
    monkeypatch.setenv(API_KEY_ENV, "test-provider-key")
    monkeypatch.setenv(NAME_ENV, wire_name)
    monkeypatch.setenv(ENDPOINT_ENV, VALID_ENDPOINT)
    with pytest.raises(LiveRuntimeError, match="exactly the Azure wire name"):
        resolve_live_model_profile(manifest, cell_id="preflight")


def test_mini_run_rejects_invalid_wire_before_creating_result_leaf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _manifest()
    plan = build_plan(manifest, selection="smoke")
    monkeypatch.setenv(API_KEY_ENV, "test-provider-key")
    monkeypatch.setenv(NAME_ENV, MINI_SNAPSHOT_MODEL)
    monkeypatch.setenv(ENDPOINT_ENV, VALID_ENDPOINT)
    with pytest.raises(LiveRuntimeError, match="exactly the Azure wire name"):
        run_experiment(
            ExperimentExecutionRequest(manifest, plan, tmp_path, selection="smoke"),
            dependencies=ExecutionDependencies(
                system_factory=lambda *_args: pytest.fail("system factory must not run"),
                live_validation=lambda _cells: pytest.fail("live validation must not run"),
            ),
        )
    assert not (tmp_path / manifest.experiment_id / plan.plan_id).exists()


@pytest.mark.parametrize(
    ("endpoint", "message"),
    (
        ("https://api.openai.com/v1", "Azure services.ai.azure.com host"),
        ("https://mini-resource.services.ai.azure.com/api/projects/x", "path must end"),
        ("http://mini-resource.services.ai.azure.com/openai/v1", "HTTPS URL"),
    ),
)
def test_mini_live_preflight_rejects_non_azure_endpoint_shapes(
    monkeypatch: pytest.MonkeyPatch, endpoint: str, message: str
) -> None:
    manifest = _manifest()
    monkeypatch.setenv(API_KEY_ENV, "test-provider-key")
    monkeypatch.setenv(NAME_ENV, MINI_WIRE_MODEL)
    monkeypatch.setenv(ENDPOINT_ENV, endpoint)
    with pytest.raises(LiveRuntimeError, match=message):
        resolve_live_model_profile(manifest, cell_id="preflight")


def test_mini_live_preflight_provenance_is_sanitized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _manifest()
    _set_runtime(monkeypatch, tmp_path)
    provenance = validate_live_runtime(
        manifest,
        tuple(build_plan(manifest, selection="smoke").cells),
        search_service_check=False,
    )
    encoded = json.dumps(provenance, sort_keys=True)
    assert provenance["model_name"] == MINI_WIRE_MODEL
    assert provenance["wire_model_name"] == MINI_WIRE_MODEL
    assert provenance["snapshot_model_name"] == MINI_SNAPSHOT_MODEL
    assert provenance["backbone_id"] == MINI_BACKBONE
    assert provenance["wire_name_binding"] == "user_authorized_azure_wire_mapping/v1"
    assert provenance["provider_response_model_names"] == []
    assert provenance["system_fingerprints"] == []
    assert provenance["endpoint"] == VALID_ENDPOINT
    assert "test-provider-key" not in encoded
    assert "test-search-token" not in encoded


def test_mini_smoke_fixture_terminalizes_exactly_162_cells(tmp_path: Path) -> None:
    manifest = _manifest()
    plan = build_plan(manifest, selection="smoke")
    task_ids = {cell.task_id for cell in plan.cells}
    rows = tuple(
        {
            "task_id": task_id,
            "prompt": f"Hermetic dated mini smoke fixture for {task_id}",
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
            model_profile=_profile(),
        ),
    )
    assert summary.status == "complete"
    assert summary.total_cells == summary.completed_cells == 162
    assert summary.pending_cells == summary.failed_cells == summary.exhausted_cells == 0
    root = tmp_path / manifest.experiment_id / plan.plan_id
    payload = json.loads((root / "execution-summary.json").read_text(encoding="utf-8"))
    assert set(payload["terminal_cells"]) == {cell.cell_id for cell in plan.cells}
    assert set(payload["terminal_cells"].values()) == {"succeeded"}
    assert len((root / "attempts.jsonl").read_text(encoding="utf-8").splitlines()) == 162
    assert MINI_WIRE_MODEL in (root / "cells" / plan.cells[0].cell_id / "attempt-1.json").read_text(
        encoding="utf-8"
    )
    contract = json.loads((root / "execution-contract.json").read_text(encoding="utf-8"))
    assert contract == build_execution_contract(manifest, plan)
    assert payload["execution_contract"] == contract
    assert payload["model_provenance"]["wire_model_name"] == MINI_WIRE_MODEL
    assert payload["model_provenance"]["snapshot_model_name"] == MINI_SNAPSHOT_MODEL
    attempt = json.loads(
        (root / "cells" / plan.cells[0].cell_id / "attempt-1.json").read_text(encoding="utf-8")
    )
    assert attempt["model_provenance"]["wire_model_name"] == MINI_WIRE_MODEL
    assert attempt["model_provenance"]["snapshot_model_name"] == MINI_SNAPSHOT_MODEL

    analysis = analyze_experiment(
        AnalysisRequest(
            manifest=manifest,
            plan=plan,
            input_root=tmp_path,
            output_dir=tmp_path / "analysis",
            selection="smoke",
        )
    )
    assert analysis.identity["wire_model_name"] == MINI_WIRE_MODEL
    assert analysis.identity["snapshot_model_name"] == MINI_SNAPSHOT_MODEL
    assert analysis.identity["backbone_id"] == MINI_BACKBONE
    assert analysis.identity["wire_name_binding"] == "user_authorized_azure_wire_mapping/v1"
    lineage = json.loads((tmp_path / "analysis" / "lineage.json").read_text(encoding="utf-8"))
    assert lineage["execution_contract"] == contract
    assert lineage["raw_artifact_lineage"]["execution_contract"] == contract
    assert (
        audit_experiment(
            ExperimentExecutionRequest(manifest, plan, tmp_path, selection="smoke")
        ).status
        == "valid"
    )


def test_model_brokers_send_wire_name_and_verify_dual_provider_identity() -> None:
    profile = _profile()
    context = _context(profile)
    deep = DeepModelBroker(
        profile,
        context,
        max_calls=1,
        responder=lambda _request: {**_response(MINI_WIRE_MODEL), "system_fingerprint": "fp-mini"},
    )
    deep.request({"messages": [{"role": "user", "content": "q"}]})
    assert deep.requests[0]["model"] == MINI_WIRE_MODEL
    assert deep.provider_model_names == [MINI_WIRE_MODEL]
    assert deep.system_fingerprints == ["fp-mini"]
    assert deep.requests[0]["model"] == MINI_WIRE_MODEL

    dated = DeepModelBroker(
        profile,
        context,
        max_calls=1,
        responder=lambda _request: _response(MINI_SNAPSHOT_MODEL),
    )
    dated.request({"messages": [{"role": "user", "content": "q"}]})
    assert dated.provider_model_names == [MINI_SNAPSHOT_MODEL]

    drifted_name = "gpt-4.1"
    with pytest.raises(FinanceHarnessBoundaryError, match="model_identity_mismatch"):
        DeepModelBroker(
            profile,
            context,
            max_calls=1,
            responder=lambda _request: _response(drifted_name),
        ).request({"messages": [{"role": "user", "content": "q"}]})

    budget = BudgetSpec(1, 0, 0, 0, 100, 100, 30.0)
    host = HostModelBroker(
        profile,
        context,
        BudgetLedger(budget),
        ("research",),
        responder=lambda _request: _response(MINI_SNAPSHOT_MODEL),
    )
    host.request({"role": "research", "messages": [{"role": "user", "content": "q"}]})
    host_summary = host.request_summary()
    assert host_summary["provider_response_model_names"] == [MINI_SNAPSHOT_MODEL]
    assert host_summary["model_provenance"]["wire_model_name"] == MINI_WIRE_MODEL
    assert host_summary["model_provenance"]["snapshot_model_name"] == MINI_SNAPSHOT_MODEL
    with pytest.raises(HostModelBrokerError, match="model_identity_mismatch"):
        HostModelBroker(
            profile,
            context,
            BudgetLedger(budget),
            ("research",),
            responder=lambda _request: _response(drifted_name),
        ).request({"role": "research", "messages": [{"role": "user", "content": "q"}]})


@pytest.mark.parametrize(
    "response",
    (
        {"choices": [{"message": {"role": "assistant", "content": "answer"}}]},
        {**_response(MINI_WIRE_MODEL), "model": None},
        {**_response(MINI_WIRE_MODEL), "model": 41},
        {**_response(MINI_WIRE_MODEL), "model": "gpt-4.1-mini "},
        {**_response(MINI_WIRE_MODEL), "model": "gpt-4.1"},
    ),
)
def test_mini_deep_broker_requires_exact_provider_response_identity(
    response: dict[str, object],
) -> None:
    profile = _profile()
    context = _context(profile)
    with pytest.raises(FinanceHarnessBoundaryError, match="model_identity_mismatch"):
        DeepModelBroker(
            profile,
            context,
            max_calls=1,
            responder=lambda _request: response,
        ).request({"messages": [{"role": "user", "content": "q"}]})


def test_mini_execution_contract_is_required_before_resume_side_effects(tmp_path: Path) -> None:
    manifest = _manifest()
    plan = build_plan(manifest, selection="smoke")
    root = tmp_path / manifest.experiment_id / plan.plan_id
    root.mkdir(parents=True)
    (root / "attempts.jsonl").write_text("stale-ledger\n", encoding="utf-8")
    calls: list[str] = []

    def factory(*_args):
        calls.append("factory")
        raise AssertionError("system factory must not run before contract validation")

    def load_tasks():
        calls.append("tasks")
        raise AssertionError("task loader must not run before contract validation")

    with pytest.raises(ExecutionContractError, match="missing for non-empty"):
        resume_experiment(
            ExperimentExecutionRequest(manifest, plan, tmp_path, selection="smoke"),
            dependencies=ExecutionDependencies(system_factory=factory, task_loader=load_tasks),
        )
    assert calls == []

    (root / "attempts.jsonl").unlink()
    contract = build_execution_contract(manifest, plan)
    contract["wire_model_name"] = MINI_SNAPSHOT_MODEL
    (root / "execution-contract.json").write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ExecutionContractError, match="mismatch"):
        resume_experiment(
            ExperimentExecutionRequest(manifest, plan, tmp_path, selection="smoke"),
            dependencies=ExecutionDependencies(system_factory=factory, task_loader=load_tasks),
        )
    assert calls == []
