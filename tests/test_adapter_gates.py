"""Phase 3B/3C adapter gates and Phase 4 offline policy wiring."""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from benchmark_harness.config import ExperimentConfig, ModelConfig
from benchmark_harness.contracts import RunContext, TaskInput
from benchmark_harness.systems import (
    AdapterBoundaryError,
    BudgetSpec,
    ExperimentBinding,
    TaskCapability,
    V3Envelope,
    build_retrieval_policy,
)
from benchmark_harness.systems.adapters import (
    GPT_RESEARCHER_LICENSE_SHA256,
    GPT_RESEARCHER_PROVENANCE,
    ODR_LICENSE_SHA256,
    ODR_PROVENANCE,
    GptResearcherAdapter,
    OdrAdapter,
    build_gpt_researcher_descriptor,
    build_odr_descriptor,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _budget(**overrides: int | float) -> BudgetSpec:
    values: dict[str, int | float] = {
        "max_model_calls": 2,
        "max_search_calls": 2,
        "max_visit_calls": 0,
        "max_other_tool_calls": 0,
        "max_input_tokens": 1000,
        "max_output_tokens": 1000,
        "wall_clock_timeout_seconds": 5.0,
    }
    values.update(overrides)
    return BudgetSpec(**values)


def _descriptor(kind: str, *, retrieval_regime: str = "cutoff_controlled"):
    kwargs = {
        "system_id": f"{kind}-fixture",
        "backbone_id": "fixture/model@sha",
        "backbone_training_cutoff": date(2024, 1, 1),
        "cutoff_source": "fixture-only",
        "retrieval_regime": retrieval_regime,
    }
    return (
        build_odr_descriptor(**kwargs)
        if kind == "odr"
        else build_gpt_researcher_descriptor(**kwargs)
    )


def _binding(
    descriptor,
    *,
    budget: BudgetSpec | None = None,
    clock: date | None = date(2025, 1, 1),
):
    selected = budget or _budget()
    return ExperimentBinding(
        experiment_id="adapter-fixture-experiment",
        plan_id="adapter-fixture-plan",
        plan_hash="adapter-fixture-plan-hash",
        cell_id="adapter-cell",
        attempt_id="adapter-cell:attempt-1",
        descriptor_hash=descriptor.descriptor_hash,
        task_id="fixture-task",
        retrieval_regime=descriptor.retrieval_regime,
        trial=1,
        budget=selected,
        experiment_as_of=clock,
    )


def _capability(budget: BudgetSpec) -> TaskCapability:
    return TaskCapability(
        service_url=None,
        service_token="fixture-search-token",
        model_broker_capability="fixture-model-token",
        budget=budget,
        allow_calculation=True,
        allow_citations=True,
        require_structured_output=False,
        run_id="adapter-run",
        trial_id="adapter-trial",
        trace_mode="hash_only",
    )


def _source_verifier(kind: str):
    provenance = ODR_PROVENANCE if kind == "odr" else GPT_RESEARCHER_PROVENANCE
    license_hash = ODR_LICENSE_SHA256 if kind == "odr" else GPT_RESEARCHER_LICENSE_SHA256

    def verify(_source_dir, declared):
        assert declared == provenance
        return {**declared.to_dict(), "license_files": {"LICENSE": license_hash}}

    return verify


class WorkerScript:
    def __init__(
        self,
        kind: str,
        *,
        hello_overrides: dict[str, Any] | None = None,
        malformed_model: bool = False,
        malformed_search: bool = False,
        task_error_sub_reason: str | None = None,
        slow: bool = False,
    ) -> None:
        self.kind = kind
        self.hello_overrides = hello_overrides or {}
        self.malformed_model = malformed_model
        self.malformed_search = malformed_search
        self.task_error_sub_reason = task_error_sub_reason
        self.slow = slow
        self.step = 0
        self.task_start: V3Envelope | None = None
        self.search_response: V3Envelope | None = None

    @property
    def adapter_id(self) -> str:
        return "odr-typed-host-v3" if self.kind == "odr" else "gpt-researcher-typed-host-v3"

    def _hello(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol": "external_research_stdio/v3",
            "adapter_id": self.adapter_id,
            "child_network": "forbidden",
            "mcp_enabled": False,
            "mcp_config": None,
            "native_browser": False,
            "public_network": False,
            "listener": False,
            "ranking_eligible": False,
            "injected_search": "search_filter_http/v1",
        }
        if self.kind == "odr":
            payload.update(
                {
                    "search_api": "none",
                    "allow_clarification": False,
                    "native_search": False,
                    "pinned_factory": "open_deep_research.utils.get_all_tools",
                    "pinned_graph": "open_deep_research.deep_researcher.deep_researcher",
                }
            )
        else:
            payload.update(
                {
                    "retriever": "custom",
                    "retriever_env": "custom",
                    "fallback_retriever": False,
                    "scraper": "disabled",
                    "embeddings": "disabled",
                    "source_curation": "disabled",
                    "telemetry": False,
                    "clock_injection": "declared_experiment_clock/v1",
                    "license_status": "CONFLICTING",
                    "pinned_deep_path": "gpt_researcher.skills.deep_research.DeepResearchSkill",
                    "pinned_agent": "gpt_researcher.agent.GPTResearcher",
                }
            )
        payload.update(self.hello_overrides)
        return payload

    def _envelope(self, message: V3Envelope, message_type: str, seq: int, payload: dict[str, Any]):
        return V3Envelope(
            type=message_type,
            request_id=f"worker-{seq}",
            task_id=message.task_id,
            seq=seq,
            binding_hash=message.binding_hash,
            payload=payload,
        )

    def __call__(self, message: V3Envelope | None) -> V3Envelope:
        if message is not None and message.type == "task.start":
            self.task_start = message
            return self._envelope(message, "hello", 0, self._hello())
        if self.slow:
            time.sleep(0.02)
        if message is None and self.step == 0:
            assert self.task_start is not None
            self.step = 1
            model_payload: dict[str, Any] = {
                "role": "research" if self.kind == "odr" else "fast",
                "messages": [{"role": "user", "content": "fixture question"}],
            }
            if self.malformed_model:
                model_payload = {"messages": "malformed"}
            return self._envelope(self.task_start, "model.request", 1, model_payload)
        if message is not None and message.type == "model.response":
            request = message
            if self.task_error_sub_reason is not None:
                return self._envelope(
                    request,
                    "task.error",
                    2,
                    {
                        "error": "worker_error:RuntimeError",
                        "sub_reason": self.task_error_sub_reason,
                    },
                )
            search_payload: dict[str, Any] = {"query": "fixture search", "max_results": 1}
            if self.malformed_search:
                search_payload["url"] = "https://must-not-be-fetched.example"
            return self._envelope(request, "search.request", 2, search_payload)
        if message is not None and message.type == "search.response":
            self.search_response = message
            return self._envelope(
                message,
                "task.result",
                3,
                {"prediction": "fixture answer", "usage": {"fixture": 1}},
            )
        raise RuntimeError(f"unexpected worker input: {message}")


def _adapter(
    kind: str,
    script: WorkerScript,
    *,
    configuration=None,
    budget=None,
    clock=date(2025, 1, 1),
    model_responder=None,
):
    descriptor = _descriptor(kind)
    policy = build_retrieval_policy(descriptor.retrieval_regime)
    cls = OdrAdapter if kind == "odr" else GptResearcherAdapter
    adapter = cls(
        descriptor=descriptor,
        retrieval_policy=policy,
        source_verifier=_source_verifier(kind),
        worker_exchange=script,
        search_handler=lambda _query, _token, _index: {
            "results": [{"url": "https://fixture.example", "raw_content": "x" * 120}]
        },
        search_token="parent-only-token",
        model_responder=model_responder,
        configuration=configuration,
        repo_root=REPOSITORY_ROOT,
    )
    selected_budget = budget or _budget()
    return adapter, _binding(descriptor, budget=selected_budget, clock=clock), selected_budget


def _run(adapter, binding, budget):
    profile = ExperimentConfig(model=ModelConfig(mode="mock")).resolve_model(require_ready=True)
    assert profile is not None
    context = RunContext(
        run_id="adapter-run",
        framework=adapter.descriptor.framework_id,
        adapter_version=adapter.adapter_revision,
        model=profile,
        trace_content_mode="hash_only",
    )
    return adapter.run_task(
        TaskInput("fixture-task", "fixture question", cutoff_date="2025-01-01"),
        context,
        _capability(budget),
        binding,
    )


@pytest.mark.parametrize("kind", ["odr", "gpt"])
def test_adapter_gate_reports_are_ready_integration_only_and_non_ranking(kind: str) -> None:
    script = WorkerScript(kind)
    adapter, binding, budget = _adapter(kind, script)
    preflight = adapter.preflight(binding)

    assert preflight.ready is True
    report = adapter.last_gate_report
    assert report is not None
    assert report.status == "integration_only"
    assert report.ranking_eligible is False
    assert report.to_dict()["ranking_eligible"] is False
    assert report.phase == ("3B" if kind == "odr" else "3C")
    result = _run(adapter, binding, budget)
    assert result.raw_text == "fixture answer"
    assert result.model_call_count == 1
    assert len(result.tool_calls) == 1
    assert result.effective_request["ranking_eligible"] is False
    assert result.effective_request["search_capability"]["capability_token"] is None
    assert "parent-only-token" not in json.dumps(result.effective_request)


def test_odr_gate_requires_none_search_api_no_mcp_and_four_roles() -> None:
    script = WorkerScript("odr")
    adapter, binding, _budget_value = _adapter("odr", script)
    report = adapter.preflight(binding)

    assert report.ready
    assert report.provenance["provenance"]["revision"] == ODR_PROVENANCE.revision
    checks = report.provenance["checks"]
    assert checks["odr_search_api"]
    assert checks["odr_mcp_config"]
    assert checks["odr_allow_clarification"]
    assert checks["odr_typed_injection"]
    assert adapter.model_roles == ("summarization", "research", "compression", "final_report")


def test_gpt_gate_records_benchmark_lock_and_conflicting_license_nonranking() -> None:
    script = WorkerScript("gpt")
    adapter, binding, _budget_value = _adapter("gpt", script)
    report = adapter.preflight(binding)

    assert report.ready
    evidence = report.provenance["provenance"]
    assert evidence["benchmark_lock"]["sha256"] == (
        "bad03a459c52521301ad9445d27ed6b83c146c0d481eee57a0a974e2ccea6e3a"
    )
    assert evidence["benchmark_lock"]["python_version"] == "3.12"
    assert evidence["benchmark_lock"]["runtime"] == "3.12.13"
    assert evidence["benchmark_lock"]["resolver"] == "uv"
    assert evidence["benchmark_lock"]["sync"] == "verify_only"
    assert evidence["license_status"] == "CONFLICTING"
    assert evidence["license_disposition"] == "adapter_allowed_ranking_blocked"
    checks = report.provenance["checks"]
    assert checks["gpt_retriever"]
    assert checks["gpt_scraper"]
    assert checks["gpt_clock_injection"]
    assert checks["gpt_conflicting_license_nonranking"]


@pytest.mark.parametrize(
    ("kind", "flag"),
    [
        ("odr", "native_search"),
        ("odr", "mcp_config"),
        ("odr", "allow_clarification"),
        ("odr", "listener"),
        ("odr", "public_network"),
        ("gpt", "browser"),
        ("gpt", "telemetry"),
        ("gpt", "public_network"),
    ],
)
def test_worker_forbidden_paths_fail_closed(kind: str, flag: str) -> None:
    value: Any = "forbidden-config" if flag == "mcp_config" else True
    script = WorkerScript(kind, hello_overrides={flag: value})
    adapter, binding, budget = _adapter(kind, script)

    with pytest.raises(AdapterBoundaryError):
        _run(adapter, binding, budget)


def test_missing_declared_clock_fails_preflight() -> None:
    script = WorkerScript("gpt")
    adapter, binding, _budget_value = _adapter("gpt", script, clock=None)

    result = adapter.preflight(binding)

    assert result.ready is False
    assert "declared_experiment_clock_missing" in result.reasons


@pytest.mark.parametrize(
    "configuration",
    [
        {"retriever": "tavily"},
        {"fallback_retriever": True},
        {"scraper": "browser"},
        {"telemetry": True},
        {"clock_injection": "ambient_machine_time"},
    ],
)
def test_gpt_alternate_retriever_scraper_clock_and_telemetry_fail_closed(configuration) -> None:
    script = WorkerScript("gpt")
    adapter, binding, _budget_value = _adapter("gpt", script, configuration=configuration)

    result = adapter.preflight(binding)

    assert result.ready is False
    assert result.reasons


def test_gpt_short_raw_content_is_explicit_empty_without_url_fetch() -> None:
    script = WorkerScript("gpt")
    descriptor = _descriptor("gpt")
    budget = _budget()
    adapter = GptResearcherAdapter(
        descriptor=descriptor,
        source_verifier=_source_verifier("gpt"),
        worker_exchange=script,
        search_handler=lambda _query, _token, _index: {
            "results": [{"url": "https://fixture.example", "raw_content": "short"}]
        },
        repo_root=REPOSITORY_ROOT,
    )
    binding = _binding(descriptor, budget=budget)

    _run(adapter, binding, budget)

    assert script.search_response is not None
    results = script.search_response.payload["results"]
    assert results[0]["raw_content"] == ""
    assert results[0]["content_status"] == "empty_or_short"


def test_malformed_search_protocol_and_model_payload_fail_closed() -> None:
    search_script = WorkerScript("odr", malformed_search=True)
    search_adapter, search_binding, search_budget = _adapter("odr", search_script)
    with pytest.raises(AdapterBoundaryError, match="unsupported fields"):
        _run(search_adapter, search_binding, search_budget)

    model_script = WorkerScript("odr", malformed_model=True)
    model_adapter, model_binding, model_budget = _adapter("odr", model_script)
    with pytest.raises(AdapterBoundaryError, match="role_missing|messages_missing"):
        _run(model_adapter, model_binding, model_budget)


def test_worker_failure_subreason_survives_and_is_validated_at_host_boundary() -> None:
    script = WorkerScript("odr", task_error_sub_reason="odr_search_tool_call_invalid")
    adapter, binding, budget = _adapter("odr", script)

    with pytest.raises(AdapterBoundaryError, match="worker_error:RuntimeError") as caught:
        _run(adapter, binding, budget)

    assert caught.value.diagnostic is not None
    assert caught.value.diagnostic["sub_reason"] == "odr_search_tool_call_invalid"
    assert caught.value.diagnostic["broker_summary"]["call_count"] == 1

    invalid_script = WorkerScript("odr", task_error_sub_reason="provider secret leaked")
    invalid_adapter, invalid_binding, invalid_budget = _adapter("odr", invalid_script)
    with pytest.raises(AdapterBoundaryError, match="worker_error_subreason_invalid"):
        _run(invalid_adapter, invalid_binding, invalid_budget)


def test_terminal_worker_envelope_is_binding_and_sequence_checked() -> None:
    script = WorkerScript("odr")

    def drifted_exchange(message):
        response = script(message)
        if response.type == "task.result":
            return replace(response, binding_hash="wrong-binding")
        return response

    descriptor = _descriptor("odr")
    budget = _budget()
    adapter = OdrAdapter(
        descriptor=descriptor,
        source_verifier=_source_verifier("odr"),
        worker_exchange=drifted_exchange,
        search_handler=lambda _query, _token, _index: {
            "results": [{"url": "https://fixture.example", "raw_content": "x" * 120}]
        },
        search_token="parent-only-token",
        repo_root=REPOSITORY_ROOT,
    )
    with pytest.raises(AdapterBoundaryError, match="worker_binding_mismatch"):
        _run(adapter, _binding(descriptor, budget=budget), budget)


def test_missing_model_usage_fails_closed_at_host_budget_seam() -> None:
    script = WorkerScript("odr")
    adapter, binding, budget = _adapter(
        "odr",
        script,
        model_responder=lambda _request: {
            "choices": [{"message": {"role": "assistant", "content": "answer"}}]
        },
    )
    with pytest.raises(AdapterBoundaryError, match="model_usage_missing"):
        _run(adapter, binding, budget)


def test_budget_excess_and_timeout_fail_closed() -> None:
    budget = _budget(max_model_calls=0)
    budget_script = WorkerScript("odr")
    budget_adapter, budget_binding, _ = _adapter("odr", budget_script, budget=budget)
    with pytest.raises(AdapterBoundaryError, match="budget exceeded"):
        _run(budget_adapter, budget_binding, budget)

    timeout_budget = _budget(wall_clock_timeout_seconds=0.001)
    timeout_script = WorkerScript("odr", slow=True)
    timeout_adapter, timeout_binding, _ = _adapter("odr", timeout_script, budget=timeout_budget)
    with pytest.raises(AdapterBoundaryError, match="worker_timeout"):
        _run(timeout_adapter, timeout_binding, timeout_budget)


def test_phase4_retrieval_regimes_are_explicit_offline_and_non_ranking() -> None:
    closed = build_retrieval_policy("closed_book")
    cutoff = build_retrieval_policy("cutoff_controlled")
    leakage = build_retrieval_policy("unrestricted_current_web")

    assert closed.ranking_eligible is False
    assert cutoff.ranking_eligible is False
    assert leakage.ranking_eligible is False
    closed.validate_capability(search_handler_present=False)
    cutoff.validate_capability(search_handler_present=True)
    with pytest.raises(ValueError, match="requires a host-owned search"):
        cutoff.validate_capability(search_handler_present=False)
    with pytest.raises(ValueError, match="explicit authorization"):
        leakage.assert_live_execution_allowed()


def _real_worker_adapter(kind: str, source: str, *, search_content: str = "C" * 200):
    descriptor = _descriptor(kind)
    budget = _budget(
        max_model_calls=8 if kind == "odr" else 4,
        max_search_calls=2,
        wall_clock_timeout_seconds=20.0,
    )
    cls = OdrAdapter if kind == "odr" else GptResearcherAdapter
    seen_requests: list[dict[str, Any]] = []

    def model_responder(request: dict[str, Any]) -> dict[str, Any]:
        seen_requests.append(request)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": f"real-{kind}-answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }

    adapter = cls(
        descriptor=descriptor,
        source_dir=source,
        search_handler=lambda _query, _token, _index: {
            "results": [{"url": f"https://fixture.example/{kind}", "raw_content": search_content}]
        },
        search_token="parent-only-token",
        model_responder=model_responder,
        repo_root=REPOSITORY_ROOT,
    )
    return adapter, _binding(descriptor, budget=budget), budget, seen_requests


@pytest.mark.parametrize(
    ("kind", "expected_model_calls"),
    [("odr", 7), ("gpt", 4)],
)
def test_pinned_worker_stdio_one_task_smoke(kind: str, expected_model_calls: int) -> None:
    source = os.environ.get(
        "BENCHMARK_ODR_SOURCE" if kind == "odr" else "BENCHMARK_GPT_SOURCE"
    ) or str(REPOSITORY_ROOT / "tests" / "fixtures" / f"pinned_{kind}")
    adapter, binding, budget, _seen_requests = _real_worker_adapter(kind, source)
    try:
        preflight = adapter.preflight(binding)
        assert preflight.ready, preflight.reasons
        result = _run(adapter, binding, budget)
    finally:
        adapter.close()
    assert isinstance(result.raw_text, str) and result.raw_text
    assert result.model_call_count == expected_model_calls
    assert len(result.tool_calls) == (1 if kind == "odr" else 2)
    assert result.effective_request["ranking_eligible"] is False
    assert result.usage["pinned_factory" if kind == "odr" else "pinned_deep_path"] is True
    assert result.usage["worker_search_tools"] == 1
    if kind == "gpt":
        assert result.usage["scraper_guard"] is True
        assert result.usage["clock_paths"]


def test_pinned_gpt_worker_clock_scraper_and_credential_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = os.environ.get("BENCHMARK_GPT_SOURCE") or str(
        REPOSITORY_ROOT / "tests" / "fixtures" / "pinned_gpt"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-worker")
    adapter, binding, budget, seen_requests = _real_worker_adapter(
        "gpt", source, search_content="short"
    )
    try:
        result = _run(adapter, binding, budget)
    finally:
        adapter.close()
    serialized_requests = json.dumps(seen_requests, sort_keys=True)
    assert serialized_requests.count("2025-01-01") >= 3
    assert "'raw_content': 'short'" not in serialized_requests
    assert "must-not-enter-worker" not in serialized_requests
    assert "parent-only-token" not in json.dumps(result.effective_request)


def test_pinned_gpt_lock_drift_is_rejected_before_worker_start(
    tmp_path: Path,
) -> None:
    source = os.environ.get("BENCHMARK_GPT_SOURCE") or str(
        REPOSITORY_ROOT / "tests" / "fixtures" / "pinned_gpt"
    )
    lock_root = tmp_path / "configs" / "external" / "gpt-researcher"
    lock_root.mkdir(parents=True)
    for filename in ("uv.lock", "pyproject.toml", "LOCK_METADATA.json"):
        shutil.copy2(
            REPOSITORY_ROOT / "configs" / "external" / "gpt-researcher" / filename,
            lock_root,
        )
    (lock_root / "uv.lock").write_text(
        (lock_root / "uv.lock").read_text(encoding="utf-8") + "\n# drift\n",
        encoding="utf-8",
    )
    descriptor = _descriptor("gpt")
    adapter = GptResearcherAdapter(
        descriptor=descriptor,
        source_dir=source,
        source_verifier=None,
        worker_command=[],
        repo_root=tmp_path,
        benchmark_lock_path=lock_root / "uv.lock",
    )
    report = adapter.preflight(_binding(descriptor))
    assert report.ready is False
    assert any("declared file hash" in reason for reason in report.reasons)
