"""Hermetic tests for the ODR structured-output retry contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.contracts import RunContext, TaskInput
from benchmark_harness.experiment import (
    ExecutionDependencies,
    ExperimentExecutionRequest,
    build_plan,
    load_manifest,
    resume_experiment,
    run_experiment,
)
from benchmark_harness.systems.adapters.odr import (
    ODR_LICENSE_SHA256,
    OdrAdapter,
    build_odr_descriptor,
)
from benchmark_harness.systems.contracts import (
    BudgetSpec,
    ExperimentBinding,
    PreflightResult,
    TaskCapability,
)
from benchmark_harness.systems.external_stdio import V3Envelope
from benchmark_harness.systems.regimes import build_retrieval_policy
from benchmark_harness.systems.runtime import AdapterBoundaryError
from benchmark_harness.systems.worker_diagnostics import (
    WorkerDiagnosticError,
    validate_worker_diagnostic,
)
from benchmark_harness.systems.workers.odr import (
    ODR_STRUCTURED_RETRY_CORRECTION,
    ODRStructuredOutputError,
    _PinnedODRGraphModel,
)


class ResearchQuestion(BaseModel):
    research_brief: str


class _ScriptedBroker:
    def __init__(self, responses: list[str | BaseException]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    def request_model(
        self,
        *,
        role: str,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.requests.append(
            {"role": role, "messages": messages, "response_format": response_format}
        )
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return {"choices": [{"message": {"role": "assistant", "content": response}}]}


class _TaskErrorWorker:
    """Emit one model call, then a typed ODR task failure."""

    def __init__(self, diagnostic: dict[str, Any]) -> None:
        self.diagnostic = diagnostic
        self.start: V3Envelope | None = None
        self.step = 0

    def __call__(self, message: V3Envelope | None) -> V3Envelope:
        if message is not None and message.type == "task.start":
            self.start = message
            return V3Envelope(
                type="hello",
                request_id="hello",
                task_id=message.task_id,
                seq=0,
                binding_hash=message.binding_hash,
                payload={
                    "protocol": "external_research_stdio/v3",
                    "adapter_id": "odr-typed-host-v3",
                    "child_network": "forbidden",
                    "mcp_enabled": False,
                    "mcp_config": None,
                    "search_api": "none",
                    "allow_clarification": False,
                    "native_search": False,
                    "native_browser": False,
                    "listener": False,
                    "public_network": False,
                    "telemetry": False,
                    "ranking_eligible": False,
                    "injected_search": "search_filter_http/v1",
                    "pinned_factory": "open_deep_research.utils.get_all_tools",
                    "pinned_graph": "open_deep_research.deep_researcher.deep_researcher",
                },
            )
        assert self.start is not None
        if message is None and self.step == 0:
            self.step = 1
            return V3Envelope(
                type="model.request",
                request_id="model-request",
                task_id=self.start.task_id,
                seq=1,
                binding_hash=self.start.binding_hash,
                payload={
                    "role": "summarization",
                    "messages": [{"role": "user", "content": "q"}],
                },
            )
        assert message is not None and message.type == "model.response"
        return V3Envelope(
            type="task.error",
            request_id="task-error",
            task_id=self.start.task_id,
            seq=2,
            binding_hash=self.start.binding_hash,
            payload={"error": "worker_error:RuntimeError", "diagnostic": self.diagnostic},
        )


def _odr_diagnostic() -> dict[str, Any]:
    return {
        "schema_version": "worker_error/v1",
        "component": "odr",
        "stage": "write_research_brief",
        "code": "odr_structured_model_response_invalid",
        "schema_name": "ResearchQuestion",
        "structured_attempt": 1,
        "broker_call_ordinal": 1,
        "response_bytes": 3,
        "response_sha256": "a" * 64,
    }


def _structured_model(broker: _ScriptedBroker) -> Any:
    return (
        _PinnedODRGraphModel(broker, task_prompt="task-bound prompt")
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=3)
    )


def test_invalid_structured_output_retries_once_then_returns_valid_value() -> None:
    broker = _ScriptedBroker(
        [
            "not-json-and-must-not-be-echoed",
            json.dumps({"research_brief": "validated brief"}),
        ]
    )

    result = asyncio.run(_structured_model(broker).ainvoke([HumanMessage(content="upstream")]))

    assert result.research_brief == "validated brief"
    assert len(broker.requests) == 2
    assert all(request["role"] == "summarization" for request in broker.requests)
    assert all(request["response_format"] == {"type": "json_object"} for request in broker.requests)
    retry_messages = broker.requests[1]["messages"]
    assert retry_messages[-1]["content"] == ODR_STRUCTURED_RETRY_CORRECTION
    assert (
        sum(message["content"] == ODR_STRUCTURED_RETRY_CORRECTION for message in retry_messages)
        == 1
    )
    assert all(
        "not-json-and-must-not-be-echoed" not in json.dumps(message) for message in retry_messages
    )


def test_three_invalid_structured_outputs_fail_with_typed_secret_free_diagnostic() -> None:
    invalid = "private prompt and provider response must not be persisted"
    broker = _ScriptedBroker([invalid, "{}", '{"wrong": 1}'])

    with pytest.raises(
        ODRStructuredOutputError,
        match="odr_structured_model_response_invalid",
    ) as caught:
        asyncio.run(_structured_model(broker).ainvoke([HumanMessage(content="upstream")]))

    diagnostic = caught.value.diagnostic
    assert set(diagnostic) == {
        "schema_version",
        "component",
        "stage",
        "code",
        "schema_name",
        "structured_attempt",
        "broker_call_ordinal",
        "response_bytes",
        "response_sha256",
    }
    assert diagnostic["schema_version"] == "worker_error/v1"
    assert diagnostic["component"] == "odr"
    assert diagnostic["stage"] == "write_research_brief"
    assert diagnostic["code"] == "odr_structured_model_response_invalid"
    assert diagnostic["schema_name"] == "ResearchQuestion"
    assert diagnostic["structured_attempt"] == 3
    assert diagnostic["broker_call_ordinal"] == 3
    assert diagnostic["response_bytes"] == len(b'{"wrong": 1}')
    assert diagnostic["response_sha256"] == hashlib.sha256(b'{"wrong": 1}').hexdigest()
    assert invalid not in json.dumps(diagnostic)
    assert len(broker.requests) == 3
    assert all(
        sum(
            message["content"] == ODR_STRUCTURED_RETRY_CORRECTION for message in request["messages"]
        )
        == (0 if index == 0 else 1)
        for index, request in enumerate(broker.requests)
    )


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"stop_after_attempt": 0}, {"stop_after_attempt": 4}, {"stop_after_attempt": None}],
)
def test_odr_retry_bound_rejects_missing_or_unbounded_values(kwargs: dict[str, Any]) -> None:
    model = _PinnedODRGraphModel(_ScriptedBroker([]))
    with pytest.raises(RuntimeError, match="odr_structured_retry_bound_invalid"):
        model.with_retry(**kwargs)


def test_provider_failure_is_not_retried_or_reclassified_as_schema_failure() -> None:
    broker = _ScriptedBroker([RuntimeError("transport must stay outside the typed code")])
    model = _structured_model(broker)

    with pytest.raises(RuntimeError, match="transport must stay outside"):
        asyncio.run(model.ainvoke([HumanMessage(content="upstream")]))
    assert len(broker.requests) == 1


def test_host_diagnostic_validator_rejects_unknown_keys_codes_and_oversized_values() -> None:
    valid = {
        "schema_version": "worker_error/v1",
        "component": "odr",
        "stage": "write_research_brief",
        "code": "odr_structured_model_response_invalid",
        "schema_name": "ResearchQuestion",
        "structured_attempt": 3,
        "broker_call_ordinal": 3,
        "response_bytes": 10,
        "response_sha256": "a" * 64,
    }
    assert validate_worker_diagnostic(valid) == valid
    for field, value in (
        ("unexpected", True),
        ("code", "provider_response_body"),
        ("stage", "model_response"),
        ("response_bytes", 8 * 1024 * 1024 + 1),
        ("response_sha256", "not-a-hash"),
    ):
        malformed = dict(valid)
        if field == "unexpected":
            malformed[field] = value
        else:
            malformed[field] = value
        with pytest.raises(WorkerDiagnosticError):
            validate_worker_diagnostic(malformed)


def test_host_attaches_validated_diagnostic_and_safe_partial_broker_summary() -> None:
    descriptor = build_odr_descriptor(
        system_id="odr-diagnostic-fixture",
        backbone_id="fixture/model@sha",
        backbone_training_cutoff=date(2024, 1, 1),
        cutoff_source="fixture-only",
        retrieval_regime="cutoff_controlled",
    )
    budget = BudgetSpec(4, 2, 0, 0, 1000, 1000, 30.0)
    binding = ExperimentBinding(
        experiment_id="odr-diagnostic-fixture-experiment",
        plan_id="odr-diagnostic-fixture-plan",
        plan_hash="odr-diagnostic-fixture-hash",
        cell_id="odr-diagnostic-fixture-cell",
        attempt_id="odr-diagnostic-fixture-cell:attempt-1",
        descriptor_hash=descriptor.descriptor_hash,
        task_id="A1-00001",
        retrieval_regime="cutoff_controlled",
        trial=1,
        budget=budget,
        experiment_as_of=date(2025, 1, 1),
    )
    profile = ResolvedModelProfile(
        mode="mock",
        endpoint="mock://local",
        model_name="fixture-model",
        api_key=None,
        temperature=0.0,
        max_tokens=32,
        timeout_seconds=5.0,
        retries=0,
        seed=7,
    )
    adapter = OdrAdapter(
        descriptor=descriptor,
        retrieval_policy=build_retrieval_policy("cutoff_controlled"),
        source_verifier=lambda _source, declared: {
            **declared.to_dict(),
            "license_files": {"LICENSE": ODR_LICENSE_SHA256},
        },
        worker_exchange=_TaskErrorWorker(_odr_diagnostic()),
        search_handler=lambda _query, _token, _index: {"results": []},
        search_token="fixture-search-token",
    )
    context = RunContext(
        run_id="odr-diagnostic-fixture-run",
        framework=descriptor.framework_id,
        adapter_version=descriptor.adapter_revision,
        model=profile,
        trace_content_mode="hash_only",
    )
    capability = TaskCapability(
        service_url=None,
        service_token="fixture-search-token",
        model_broker_capability="fixture-model-token",
        budget=budget,
        allow_calculation=True,
        allow_citations=True,
        require_structured_output=True,
        run_id="odr-diagnostic-fixture-run",
        trial_id="odr-diagnostic-fixture-trial",
        trace_mode="hash_only",
    )
    try:
        with pytest.raises(AdapterBoundaryError, match="worker_error:RuntimeError") as caught:
            adapter.run_task(
                TaskInput("A1-00001", "task-specific prompt", cutoff_date="2024-01-01"),
                context,
                capability,
                binding,
            )
    finally:
        adapter.close()

    diagnostic = caught.value.diagnostic
    assert diagnostic is not None
    assert diagnostic["code"] == "odr_structured_model_response_invalid"
    broker_summary = diagnostic["broker_summary"]
    assert broker_summary["call_count"] == 1
    assert broker_summary["roles"] == {"summarization": "fixture-model"}
    assert broker_summary["provider_response_model_names"] == []
    assert broker_summary["provider_model_names"] == []
    assert broker_summary["system_fingerprints"] == []
    assert isinstance(broker_summary["requests_sha256"], str)
    serialized = json.dumps(diagnostic, sort_keys=True)
    assert "task-specific prompt" not in serialized
    assert "fixture-search-token" not in serialized


class _FailingSystem:
    def __init__(self, descriptor: Any) -> None:
        self.descriptor = descriptor

    def preflight(self, _binding: ExperimentBinding) -> PreflightResult:
        return PreflightResult(ready=True)

    def run_task(self, *_args: Any) -> Any:
        raise RuntimeError("deliberate hermetic failure")

    def close(self) -> None:
        return None


def test_resume_artifact_binding_identifies_the_actual_attempt(tmp_path: Path) -> None:
    repository_root = Path(__file__).parents[1]
    manifest = load_manifest(repository_root / "configs/exploratory-gpt41mini-20260817-v1.yaml")
    task_id = "A1-CN_A-20260130-00018"
    plan = build_plan(manifest, task_ids=[task_id], selection="smoke")
    rows = (
        {
            "task_id": task_id,
            "prompt": "task-specific prompt",
            "seed": {"cutoff_date": "2026-01-30"},
        },
    )
    dependencies = ExecutionDependencies(
        system_factory=lambda system, regime, *_args: _FailingSystem(system.descriptor(regime)),
        task_loader=lambda: rows,
    )
    request = ExperimentExecutionRequest(
        manifest,
        plan,
        tmp_path,
        selection="smoke",
        max_workers=2,
        max_attempts=1,
    )
    first = run_experiment(request, dependencies=dependencies)
    assert first.failed_cells == len(plan.cells)

    resumed = resume_experiment(
        ExperimentExecutionRequest(
            manifest,
            plan,
            tmp_path,
            selection="smoke",
            max_workers=2,
            max_attempts=2,
        ),
        dependencies=dependencies,
    )
    assert resumed.failed_cells == len(plan.cells)
    root = tmp_path / manifest.experiment_id / plan.plan_id
    artifact = json.loads(
        (root / "cells" / plan.cells[0].cell_id / "attempt-2.json").read_text(encoding="utf-8")
    )
    assert artifact["binding"]["attempt_id"] == f"{plan.cells[0].cell_id}:attempt-2"
