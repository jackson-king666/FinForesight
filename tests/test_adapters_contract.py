"""Contract tests shared by the PydanticAI and LangGraph strict adapters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import fields, replace

import pytest

from benchmark_harness.adapters.langgraph import LangGraphAdapter
from benchmark_harness.adapters.pydantic_ai import PydanticAIAdapter
from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.contracts import (
    NetworkEvent,
    RunContext,
    StrictTransport,
    TaskInput,
    ToolCall,
    TransportResponse,
    canonical_messages,
)
from benchmark_harness.trace.policy_guard import StrictPolicyGuard

AdapterFactory = Callable[[StrictTransport], object]


class RecordingTransport:
    """Mock provider that records its narrow adapter input boundary."""

    def __init__(self, response: TransportResponse) -> None:
        self.response = response
        self.calls: list[tuple[TaskInput, list[dict[str, str]], RunContext]] = []

    def complete(
        self,
        task: TaskInput,
        messages: Sequence[dict[str, str]],
        context: RunContext,
    ) -> TransportResponse:
        self.calls.append((task, list(messages), context))
        return self.response


class SequencedTransport:
    """Return or raise one configured outcome per visible retry attempt."""

    def __init__(self, outcomes: Sequence[TransportResponse | Exception]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def complete(
        self,
        task: TaskInput,
        messages: Sequence[dict[str, str]],
        context: RunContext,
    ) -> TransportResponse:
        del task, messages, context
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _profile() -> ResolvedModelProfile:
    return ResolvedModelProfile(
        mode="openai_compatible",
        endpoint="https://models.example.test/v1",
        model_name="shared-test-model",
        api_key="test-only-key",
        temperature=0.0,
        max_tokens=128,
        timeout_seconds=10.0,
        retries=0,
        seed=3,
    )


def _context(framework: str, adapter_version: str, task_id: str) -> RunContext:
    return RunContext(
        run_id="adapter-contract",
        framework=framework,
        adapter_version=adapter_version,
        model=_profile(),
        trace_content_mode="full",
    ).for_task(task_id)


@pytest.mark.parametrize(
    ("adapter_factory", "framework", "node"),
    [
        (PydanticAIAdapter, "pydantic-ai", "pydantic_ai_agent"),
        (LangGraphAdapter, "langgraph", "langgraph_model"),
    ],
    ids=["pydantic-ai", "langgraph"],
)
def test_adapter_mock_transport_success_observes_only_canonical_task_input(
    adapter_factory: AdapterFactory,
    framework: str,
    node: str,
) -> None:
    task = TaskInput(task_id="C-00001", prompt="Return a canonical answer.")
    response = TransportResponse(
        raw_text="canonical response",
        usage={"input_tokens": 5, "output_tokens": 2},
        status_code=200,
        observed_messages=tuple(canonical_messages(task.prompt)),
        framework_events=({"stream_mode": "messages"},),
    )
    transport = RecordingTransport(response)
    adapter = adapter_factory(transport)

    result = adapter.run(task, _context(framework, adapter.adapter_version, task.task_id))

    assert len(transport.calls) == 1
    received_task, received_messages, received_context = transport.calls[0]
    assert received_task is task
    assert received_messages == canonical_messages(task.prompt)
    assert received_context.task_id == task.task_id
    assert result.raw_text == "canonical response"
    assert result.model_call_count == 1
    assert result.tool_calls == ()
    assert [event.event_type for event in result.trace_events] == [
        "model_request",
        "framework_event",
        "model_response",
    ]
    assert result.trace_events[0].control_flow_node == node
    assert StrictPolicyGuard().check(task, result).comparable is True


@pytest.mark.parametrize(
    ("adapter_factory", "framework"),
    [(PydanticAIAdapter, "pydantic-ai"), (LangGraphAdapter, "langgraph")],
    ids=["pydantic-ai", "langgraph"],
)
def test_adapter_mock_transport_error_is_an_auditable_non_comparable_result(
    adapter_factory: AdapterFactory,
    framework: str,
) -> None:
    task = TaskInput(task_id="E-00001", prompt="Return an answer with a unit.")
    transport = RecordingTransport(
        TransportResponse(raw_text=None, status_code=503, error="provider unavailable")
    )
    adapter = adapter_factory(transport)

    result = adapter.run(task, _context(framework, adapter.adapter_version, task.task_id))
    decision = StrictPolicyGuard().check(task, result)

    assert result.error == "provider unavailable"
    assert result.raw_text is None
    assert result.network_events[0].status_code == 503
    assert result.network_events[0].is_model_endpoint is True
    assert decision.comparable is False
    assert "adapter_error" in decision.violations
    assert "empty_raw_response" in decision.violations
    assert result.trace_events[-1].event_type == "model_response"
    assert result.trace_events[-1].error == "provider unavailable"


@pytest.mark.parametrize(
    ("adapter_factory", "framework"),
    [(PydanticAIAdapter, "pydantic-ai"), (LangGraphAdapter, "langgraph")],
    ids=["pydantic-ai", "langgraph"],
)
def test_adapter_retries_timeout_then_uses_next_successful_response(
    adapter_factory: AdapterFactory,
    framework: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    task = TaskInput(task_id="A1-00001", prompt="Retry a timeout once.")
    transport = SequencedTransport(
        [
            TimeoutError("temporary provider timeout"),
            TransportResponse(raw_text="recovered response", status_code=200),
        ]
    )
    adapter = adapter_factory(transport)
    context = _context(framework, adapter.adapter_version, task.task_id)
    context.model = replace(context.model, retries=1)

    result = adapter.run(task, context)

    assert transport.calls == 2
    assert result.raw_text == "recovered response"
    assert result.error is None
    assert result.model_call_count == 2
    assert [event.event_type for event in result.trace_events].count("model_retry") == 1
    assert result.effective_request["framework_request_parameters"]["retry_attempt"] == 2
    assert "model retry:" in caplog.text
    # Strict preserves one-call comparability; a recovered retry is retained
    # as an auditable non-comparable task rather than silently scored.
    assert StrictPolicyGuard().check(task, result).comparable is False


@pytest.mark.parametrize(
    ("adapter_factory", "framework"),
    [(PydanticAIAdapter, "pydantic-ai"), (LangGraphAdapter, "langgraph")],
    ids=["pydantic-ai", "langgraph"],
)
def test_adapter_mock_transport_tool_attempt_is_recorded_and_rejected(
    adapter_factory: AdapterFactory,
    framework: str,
) -> None:
    task = TaskInput(task_id="A1-00001", prompt="Do not use tools.")
    transport = RecordingTransport(
        TransportResponse(
            raw_text="a tool-backed answer",
            status_code=200,
            tool_calls=(ToolCall(name="web_lookup", arguments_hash="args-hash"),),
            extra_network_events=(
                NetworkEvent(
                    host="tools.example.test",
                    method="POST",
                    path="/lookup",
                    body_hash="tool-body",
                    status_code=200,
                    latency_seconds=0.01,
                    is_model_endpoint=False,
                ),
            ),
        )
    )
    adapter = adapter_factory(transport)

    result = adapter.run(task, _context(framework, adapter.adapter_version, task.task_id))
    decision = StrictPolicyGuard().check(task, result)

    assert result.tool_calls == (ToolCall(name="web_lookup", arguments_hash="args-hash"),)
    assert any(event.event_type == "tool_call" for event in result.trace_events)
    assert decision.comparable is False
    assert "tool_calls=1" in decision.violations
    assert "non_model_network_calls=1" in decision.violations


def test_task_input_cannot_carry_dataset_records_or_evaluator_state() -> None:
    task = TaskInput(
        task_id="A1-00001",
        prompt="Only this prompt is visible to adapters.",
        cutoff_date="2024-06-01",
    )

    assert [field.name for field in fields(TaskInput)] == ["task_id", "prompt", "cutoff_date"]
    assert task.messages == [{"role": "user", "content": task.prompt}]
    assert not hasattr(task, "ground_truth")
    assert not hasattr(task, "record")
    assert not hasattr(task, "evaluator")
    with pytest.raises(TypeError):
        TaskInput(  # type: ignore[call-arg]
            task_id="A1-00001",
            prompt="Only this prompt is visible to adapters.",
            ground_truth={"answer": "secret"},
        )
    with pytest.raises((AttributeError, TypeError)):
        setattr(task, "ground_truth", {"answer": "secret"})
