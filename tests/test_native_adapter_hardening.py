"""Regression coverage for native-adapter egress and telemetry hardening."""

from __future__ import annotations

import json

import httpx
import pytest

from benchmark_harness.adapters.langgraph import (
    AmbientLangSmithTracingError,
    _chat_model_kwargs,
    _ensure_no_ambient_langsmith_tracing,
    _LangGraphNativeTransport,
    _stream_event_summary,
)
from benchmark_harness.adapters.langgraph import (
    _configured_timeout as langgraph_timeout,
)
from benchmark_harness.adapters.pydantic_ai import (
    _build_uninstrumented_agent,
)
from benchmark_harness.adapters.pydantic_ai import (
    _configured_timeout as pydantic_ai_timeout,
)
from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.contracts import RunContext, TaskInput


def _profile() -> ResolvedModelProfile:
    return ResolvedModelProfile(
        mode="openai_compatible",
        endpoint="https://models.example.test/v1",
        model_name="shared-test-model",
        api_key="test-only-key",
        temperature=0.0,
        max_tokens=128,
        timeout_seconds=7.5,
        retries=0,
        seed=3,
    )


def _context(task_id: str) -> RunContext:
    return RunContext(
        run_id="native-adapter-hardening",
        framework="langgraph",
        adapter_version="test-adapter",
        model=_profile(),
        trace_content_mode="full",
    ).for_task(task_id)


@pytest.mark.parametrize(
    "environment_name",
    [
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING",
        "LANGCHAIN_TRACING_V2",
    ],
)
def test_langgraph_rejects_any_enabled_ambient_tracing_environment(
    monkeypatch: pytest.MonkeyPatch, environment_name: str
) -> None:
    monkeypatch.setenv(environment_name, "true")

    with pytest.raises(AmbientLangSmithTracingError, match="ambient_langsmith_tracing_enabled"):
        _ensure_no_ambient_langsmith_tracing()


def test_langgraph_allows_explicitly_disabled_ambient_tracing_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "0")

    _ensure_no_ambient_langsmith_tracing()


def test_langgraph_egress_gate_returns_zero_call_result_before_graph_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = TaskInput(task_id="A1-00001", prompt="Do not contact a model.")
    monkeypatch.setenv("LANGSMITH_TRACING", "on")

    response = _LangGraphNativeTransport().complete(task, task.messages, _context(task.task_id))

    assert response.error == "AmbientLangSmithTracingError"
    assert response.network_events == ()
    assert response.model_call_count == 0
    assert response.request_parameters["egress_guard"] == "ambient_langsmith_tracing_rejected"


def test_langgraph_stream_summary_keeps_control_flow_evidence_without_response_content() -> None:
    secret_response = "the model response must never reach trace artifacts"
    secret_error = "provider error containing private response text"
    event = (
        "debug",
        {
            "type": "task_result",
            "payload": {
                "name": "model",
                "path": ("__pregel_pull", "model"),
                "id": "task-42",
                "triggers": ("start:model",),
                "result": {"messages": [{"content": secret_response}]},
                "error": RuntimeError(secret_error),
            },
        },
    )

    summary = _stream_event_summary(event)
    serialized = json.dumps(summary, sort_keys=True)

    assert summary["stream_mode"] == "debug"
    assert summary["event_type"] == "task_result"
    assert summary["node"] == "model"
    assert summary["path"] == ["__pregel_pull", "model"]
    assert "task_hash" in summary
    assert "trigger_hash" in summary
    assert summary["result_type"] == "dict"
    assert "result_hash" in summary
    assert summary["error_type"] == "RuntimeError"
    assert "error_hash" in summary
    assert secret_response not in serialized
    assert secret_error not in serialized


def test_native_adapter_http_clients_use_configured_timeouts_and_chat_completions() -> None:
    context = _context("C-00001")
    pydantic_timeout = pydantic_ai_timeout(context.model.timeout_seconds)
    graph_timeout = langgraph_timeout(context.model.timeout_seconds)
    client = httpx.Client(timeout=graph_timeout)
    try:
        kwargs = _chat_model_kwargs(context, client)
    finally:
        client.close()

    for timeout in (pydantic_timeout, graph_timeout):
        assert timeout.connect == context.model.timeout_seconds
        assert timeout.read == context.model.timeout_seconds
        assert timeout.write == context.model.timeout_seconds
        assert timeout.pool == context.model.timeout_seconds
    assert kwargs["use_responses_api"] is False
    assert kwargs["disable_streaming"] is True
    assert kwargs["callbacks"] == []
    assert kwargs["timeout"] == context.model.timeout_seconds
    assert kwargs["max_retries"] == 0


def test_pydantic_ai_agent_instrumentation_is_disabled_per_instance() -> None:
    class FakeAgent:
        def __init__(self, model: object, **kwargs: object) -> None:
            self.model = model
            self.kwargs = kwargs
            self.instrument: bool | None = True

    agent = _build_uninstrumented_agent(FakeAgent, object())

    assert agent.instrument is False
    assert agent.kwargs == {
        "retries": 0,
        "system_prompt": (),
        "tools": (),
        "toolsets": (),
    }
