"""Strict-policy and local-artifact security regression coverage."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmark_harness.adapters.base import effective_request
from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.contracts import (
    AdapterResult,
    NetworkEvent,
    RunContext,
    TaskInput,
    ToolCall,
    canonical_messages,
    sha256_text,
)
from benchmark_harness.trace.jsonl_sink import JsonlSink
from benchmark_harness.trace.policy_guard import StrictPolicyGuard


@pytest.fixture
def task() -> TaskInput:
    return TaskInput(task_id="C-00001", prompt="Return exactly one answer.")


def _profile() -> ResolvedModelProfile:
    return ResolvedModelProfile(
        mode="openai_compatible",
        endpoint="https://models.example.test/v1",
        model_name="test-model",
        api_key="never-persist-this",
        temperature=0.0,
        max_tokens=64,
        timeout_seconds=10.0,
        retries=0,
        seed=11,
    )


def _comparable_result(task: TaskInput) -> AdapterResult:
    context = RunContext(
        run_id="policy-test",
        framework="pydantic-ai",
        adapter_version="test-adapter",
        model=_profile(),
        trace_content_mode="full",
    ).for_task(task.task_id)
    context.emit("model_request", effective_input_hash=task.input_hash)
    context.emit("model_response", effective_input_hash=task.input_hash)
    messages = canonical_messages(task.prompt)
    return AdapterResult(
        raw_text="valid response",
        usage={"input_tokens": 3, "output_tokens": 2},
        latency_seconds=0.01,
        canonical_input_hash=task.input_hash,
        effective_input_hash=task.input_hash,
        effective_messages=tuple(messages),
        effective_request=effective_request(task, messages, context),
        model_call_count=1,
        tool_calls=(),
        network_events=(
            NetworkEvent(
                host="models.example.test",
                method="POST",
                path="/v1/chat/completions",
                body_hash="body-hash",
                status_code=200,
                latency_seconds=0.01,
                is_model_endpoint=True,
            ),
        ),
        trace_events=context.events,
    )


def test_policy_accepts_a_complete_canonical_one_turn_result(task: TaskInput) -> None:
    decision = StrictPolicyGuard().check(task, _comparable_result(task))

    assert decision.comparable is True
    assert decision.violations == ()


def _with_system_message(result: AdapterResult) -> AdapterResult:
    messages = [{"role": "system", "content": "injected"}, *result.effective_messages]
    return replace(
        result,
        effective_messages=tuple(messages),
        effective_request={**result.effective_request, "messages": messages},
    )


@pytest.mark.parametrize(
    ("mutate", "expected_violation"),
    [
        (
            lambda result: replace(result, effective_input_hash="not-the-canonical-hash"),
            "effective_input_hash_mismatch",
        ),
        (_with_system_message, "effective_messages_not_exactly_one_user_prompt"),
        (lambda result: replace(result, model_call_count=2), "model_call_count=2"),
        (
            lambda result: replace(result, tool_calls=(ToolCall(name="unexpected_lookup"),)),
            "tool_calls=1",
        ),
        (
            lambda result: replace(
                result,
                network_events=(
                    *result.network_events,
                    NetworkEvent(
                        host="search.example.test",
                        method="GET",
                        path="/lookup",
                        body_hash="external-body-hash",
                        status_code=200,
                        latency_seconds=0.02,
                        is_model_endpoint=False,
                    ),
                ),
            ),
            "non_model_network_calls=1",
        ),
    ],
    ids=["hash", "system-message", "model-count", "tool", "non-model-network"],
)
def test_policy_marks_fairness_gate_failures_non_comparable(
    task: TaskInput,
    mutate: object,
    expected_violation: str,
) -> None:
    result = _comparable_result(task)
    mutated = mutate(result)  # type: ignore[operator]

    decision = StrictPolicyGuard().check(task, mutated)

    assert decision.comparable is False
    assert expected_violation in decision.violations


def test_jsonl_artifacts_recursively_redact_credentials(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path, content_mode="full")
    authorization = "Bearer top-secret-value"
    cookie = "session=private-cookie"
    api_key = "api-key-that-must-not-appear"
    nested_token = "nested-token-that-must-not-appear"
    secret_key = "secret-key-that-must-not-appear"

    sink.write_effective_request(
        {
            "messages": [{"role": "user", "content": "safe prompt"}],
            "Authorization": authorization,
            "headers": {"Cookie": cookie, "X-Request-ID": "visible-request-id"},
            "credentials": {
                "api-key": api_key,
                "client_secret_key": secret_key,
                "nested": {"token": nested_token},
            },
        }
    )
    sink.write_network_events(
        run_id="redaction-test",
        task_id="A1-00001",
        framework="pydantic-ai",
        events=(
            NetworkEvent(
                host="models.example.test",
                method="POST",
                path="/v1/chat/completions",
                body_hash="safe-hash",
                status_code=200,
                latency_seconds=0.01,
                is_model_endpoint=True,
                headers={"authorization": authorization, "cookie": cookie},
            ),
        ),
    )

    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (sink.requests_path, sink.network_path, sink.trace_path)
    )
    assert authorization not in artifact_text
    assert cookie not in artifact_text
    assert api_key not in artifact_text
    assert secret_key not in artifact_text
    assert nested_token not in artifact_text
    assert artifact_text.count("[REDACTED]") >= 6

    request_row = json.loads(sink.requests_path.read_text(encoding="utf-8"))
    ledger_row = json.loads(sink.network_path.read_text(encoding="utf-8"))
    assert request_row["headers"]["X-Request-ID"] == "visible-request-id"
    assert request_row["credentials"]["nested"]["token"] == "[REDACTED]"
    assert ledger_row["headers"] == {"authorization": "[REDACTED]", "cookie": "[REDACTED]"}
    assert request_row["messages"] == [{"role": "user", "content": "safe prompt"}]
    assert request_row["messages"][0]["content"] != sha256_text("safe prompt")


def test_jsonl_redacts_inline_provider_errors_and_hashes_response_shaped_fields(
    tmp_path: Path,
) -> None:
    sink = JsonlSink(tmp_path, content_mode="hash_only")
    secret = "inline-provider-secret"
    prompt = "sensitive prompt text"
    response = "sensitive model response"

    sink.write_trace_events(
        (
            RunContext(
                run_id="redaction-test",
                framework="pydantic-ai",
                adapter_version="test-adapter",
                model=_profile(),
                trace_content_mode="hash_only",
            )
            .for_task("A1-00001")
            .emit(
                "model_response",
                error=f"Authorization: Bearer {secret}; response={response}",
                attributes={"prompt": prompt, "result": response},
            ),
        )
    )

    text = sink.trace_path.read_text(encoding="utf-8")
    row = json.loads(text)
    assert secret not in text
    assert prompt not in text
    assert response not in text
    assert row["attributes"]["prompt_sha256"] == sha256_text(prompt)
    assert row["attributes"]["result_sha256"] == sha256_text(response)
    assert "error_sha256" in row
