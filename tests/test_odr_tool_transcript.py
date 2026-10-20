"""Hermetic coverage for the ODR-to-host Chat Completions boundary."""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.contracts import RunContext, TaskInput
from benchmark_harness.experiment import (
    ExecutionDependencies,
    ExperimentExecutionRequest,
    build_plan,
    load_manifest,
    run_experiment,
)
from benchmark_harness.systems.adapters.odr import OdrAdapter, build_odr_descriptor
from benchmark_harness.systems.chat_transcript import (
    ChatTranscriptError,
    normalize_chat_transcript,
)
from benchmark_harness.systems.contracts import BudgetSpec, ExperimentBinding, TaskCapability
from benchmark_harness.systems.external_stdio import BudgetLedger
from benchmark_harness.systems.model_broker import HostModelBroker, HostModelBrokerError
from benchmark_harness.systems.regimes import build_retrieval_policy
from benchmark_harness.systems.runtime import AdapterBoundaryError
from benchmark_harness.systems.worker_diagnostics import worker_failure_subreason
from benchmark_harness.systems.workers.odr import _PinnedODRGraphModel

REPOSITORY_ROOT = Path(__file__).parents[1]
ODR_FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "pinned_odr"
MINI_MANIFEST = REPOSITORY_ROOT / "configs" / "exploratory-gpt41mini-20260817-v1.yaml"


def _paired_transcript() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"a":1}'},
                }
            ],
        },
        {"role": "tool", "content": "tool result", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "answer"},
    ]


def _profile(endpoint: str) -> ResolvedModelProfile:
    return ResolvedModelProfile(
        mode="openai_compatible",
        endpoint=endpoint,
        model_name="gpt-4.1-mini",
        api_key="unit-test-key",
        temperature=0.0,
        max_tokens=32,
        timeout_seconds=5.0,
        retries=0,
        seed=7,
    )


def _context(profile: ResolvedModelProfile) -> RunContext:
    return RunContext(
        run_id="odr-transcript-test",
        framework="open_deep_research",
        adapter_version="test",
        model=profile,
        trace_content_mode="hash_only",
    ).for_task("fixture-task")


def _ledger() -> BudgetLedger:
    from benchmark_harness.systems.contracts import BudgetSpec

    return BudgetLedger(BudgetSpec(8, 0, 0, 0, 1000, 1000, 30.0))


def test_odr_serializer_preserves_exact_paired_tool_history() -> None:
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="question"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call-1",
                    "name": "lookup",
                    "args": {"z": 2, "a": [1, 2]},
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="tool result", tool_call_id="call-1"),
        AIMessage(content="answer"),
    ]

    assert _PinnedODRGraphModel._message_payload(messages) == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"a":[1,2],"z":2}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "tool result", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "answer"},
    ]


@pytest.mark.parametrize(
    ("messages", "code"),
    [
        (_paired_transcript()[:-2], "tool_results_missing"),
        ([{"role": "tool", "content": "orphan", "tool_call_id": "missing"}], "tool_result_orphan"),
        (
            _paired_transcript()[:2]
            + [{"role": "tool", "content": "wrong", "tool_call_id": "call-2"}],
            "tool_result_orphan",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "same",
                            "type": "function",
                            "function": {"name": "a", "arguments": "{}"},
                        },
                        {
                            "id": "same",
                            "type": "function",
                            "function": {"name": "b", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "content": "result", "tool_call_id": "same"},
            ],
            "tool_call_id_duplicate",
        ),
        (
            _paired_transcript()[:2]
            + [{"role": "user", "content": "interposed"}]
            + _paired_transcript()[2:],
            "tool_result_not_immediate",
        ),
    ],
)
def test_tool_transcript_state_machine_fails_closed(
    messages: list[dict[str, Any]], code: str
) -> None:
    with pytest.raises(ChatTranscriptError) as caught:
        normalize_chat_transcript(messages)
    assert caught.value.code == code
    assert str(caught.value) == code


def test_fake_azure_rejects_old_orphan_shape_and_accepts_repaired_shape() -> None:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - HTTP handler API is fixed
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            response = {
                "model": payload["model"],
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }
            encoded = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        profile = _profile(f"http://127.0.0.1:{server.server_port}/v1")
        old_shape = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "model output"},
            {"role": "tool", "content": "tool result", "tool_call_id": "call-1"},
        ]
        with pytest.raises(HostModelBrokerError, match="model_request_tool_result_orphan"):
            HostModelBroker(profile, _context(profile), _ledger(), ("research",)).request(
                {"role": "research", "messages": old_shape}
            )
        assert requests == []

        broker = HostModelBroker(profile, _context(profile), _ledger(), ("research",))
        broker.request({"role": "research", "messages": _paired_transcript()})
        assert len(requests) == 1
        assert requests[0]["messages"] == _paired_transcript()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_provider_4xx_diagnostic_is_bounded_and_secret_free() -> None:
    request_body = json.dumps(
        {
            "error": {
                "code": "InvalidRequest",
                "type": "invalid_request_error",
                "param": "messages",
                "message": "prompt=private prompt api-key=never-persist-this",
                "innererror": {"code": "InvalidToolCall"},
            }
        }
    ).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - HTTP handler API is fixed
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("apim-request-id", "apim-unit-123")
            self.send_header("Content-Length", str(len(request_body)))
            self.end_headers()
            self.wfile.write(request_body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        profile = _profile(f"http://127.0.0.1:{server.server_port}/v1")
        broker = HostModelBroker(profile, _context(profile), _ledger(), ("research",))
        with pytest.raises(HostModelBrokerError, match="model_http_status=400") as caught:
            broker.request({"role": "research", "messages": _paired_transcript()})
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    diagnostic = caught.value.diagnostic
    assert diagnostic is not None
    assert diagnostic["kind"] == "provider_http_4xx"
    assert diagnostic["status"] == 400
    assert diagnostic["content_type"] == "application/json"
    assert diagnostic["body_bytes"] == len(request_body)
    assert diagnostic["body_sha256"] == hashlib.sha256(request_body).hexdigest()
    assert diagnostic["body_truncated"] is False
    assert diagnostic["apim_request_id"] == "apim-unit-123"
    assert diagnostic["provider_error"] == {
        "code": "InvalidRequest",
        "type": "invalid_request_error",
        "param": "messages",
        "innererror_code": "InvalidToolCall",
    }
    assert (
        diagnostic["message_sha256"]
        == hashlib.sha256(b"prompt=private prompt api-key=never-persist-this").hexdigest()
    )
    assert diagnostic["message_preview_omitted"] is True
    assert diagnostic["wire_model_name"] == "gpt-4.1-mini"
    assert diagnostic["logical_role"] == "research"
    assert diagnostic["model_call_ordinal"] == 1
    assert len(diagnostic["request_body_sha256"]) == 64
    assert diagnostic["message_shapes"][1]["keys"] == [
        "content",
        "role",
        "tool_calls",
    ]
    assert diagnostic["message_shapes"][1]["tool_call_count"] == 1
    serialized = json.dumps(diagnostic, ensure_ascii=False)
    assert "private prompt" not in serialized
    assert "never-persist-this" not in serialized
    assert "unit-test-key" not in serialized
    assert "Authorization" not in serialized


def test_odr_host_broker_forces_search_when_gateway_downgrades_first_search_turn() -> None:
    requests: list[dict[str, Any]] = []

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "gateway final answer",
                    }
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }

    profile = ResolvedModelProfile(
        mode="mock",
        endpoint="mock://local",
        model_name="fixture-model",
        api_key=None,
        temperature=0.0,
        max_tokens=32,
        timeout_seconds=5.0,
        retries=0,
        seed=None,
    )
    broker = HostModelBroker(
        profile,
        _context(profile),
        _ledger(),
        ("research",),
        responder=responder,
        force_first_search=True,
    )

    response = broker.request(
        {
            "role": "research",
            "messages": [{"role": "user", "content": "Find dated evidence for this task"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_filter_http",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )

    assert requests[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "search_filter_http"},
    }
    tool_call = response["choices"][0]["message"]["tool_calls"][0]
    assert response["choices"][0]["message"]["content"] == "gateway final answer"
    assert tool_call["function"]["name"] == "search_filter_http"
    assert json.loads(tool_call["function"]["arguments"]) == {
        "query": "Find dated evidence for this task",
        "max_results": 3,
    }
    assert broker.request_summary()["forced_search_tool_name"] == "search_filter_http"
    assert broker.request_summary()["forced_search_synthesized_count"] == 1
    assert broker.request_summary()["forced_search_query_source"] == "host_message"
    assert broker.request_summary()["forced_search_query_truncated_count"] == 0


def test_odr_forced_search_records_host_query_truncation() -> None:
    requests: list[dict[str, Any]] = []

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        return {
            "choices": [{"message": {"role": "assistant", "content": "downgraded content"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }

    profile = ResolvedModelProfile(
        mode="mock",
        endpoint="mock://local",
        model_name="fixture-model",
        api_key=None,
        temperature=0.0,
        max_tokens=32,
        timeout_seconds=5.0,
        retries=0,
        seed=None,
    )
    broker = HostModelBroker(
        profile,
        _context(profile),
        _ledger(),
        ("research",),
        responder=responder,
        force_first_search=True,
    )
    long_message = "x" * 513

    response = broker.request(
        {
            "role": "research",
            "messages": [{"role": "user", "content": long_message}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_filter_http",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )

    assert response["choices"][0]["message"]["content"] == "downgraded content"
    assert (
        json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])[
            "query"
        ]
        == long_message[:512]
    )
    assert broker.request_summary()["forced_search_query_source"] == "host_message"
    assert broker.request_summary()["forced_search_query_truncated_count"] == 1


@pytest.mark.parametrize(
    "tools",
    [
        [],
        [
            {
                "type": "function",
                "function": {"name": "search_filter_http"},
            },
            {"type": "function", "function": {"name": "other_tool"}},
        ],
        [{"type": "function", "function": {"name": "search"}}],
        [{"type": "function"}],
    ],
)
def test_odr_force_first_search_rejects_non_allowlisted_tool_sets(
    tools: list[dict[str, Any]],
) -> None:
    profile = _profile("mock://local")
    broker = HostModelBroker(
        profile,
        _context(profile),
        _ledger(),
        ("research",),
        force_first_search=True,
    )

    with pytest.raises(HostModelBrokerError, match="native_model_tools_forbidden"):
        broker.request(
            {
                "role": "research",
                "messages": [{"role": "user", "content": "question"}],
                "tools": tools,
            }
        )


def test_odr_child_tool_choice_is_rejected_at_the_host_seam() -> None:
    profile = _profile("mock://local")
    broker = HostModelBroker(
        profile,
        _context(profile),
        _ledger(),
        ("research",),
        force_first_search=True,
    )

    with pytest.raises(HostModelBrokerError, match="native_model_tools_forbidden"):
        broker.request(
            {
                "role": "research",
                "messages": [{"role": "user", "content": "question"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "search_filter_http"},
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "search_filter_http"},
                },
            }
        )


def test_odr_force_first_search_tool_choice_is_one_shot() -> None:
    requests: list[dict[str, Any]] = []

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": "provider selected search",
                "tool_calls": [
                    {
                        "id": "provider-search-1",
                        "type": "function",
                        "function": {
                            "name": "search_filter_http",
                            "arguments": '{"query":"provider query","max_results":3}',
                        },
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "second turn"}
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }

    profile = _profile("mock://local")
    broker = HostModelBroker(
        profile,
        _context(profile),
        _ledger(),
        ("research",),
        responder=responder,
        force_first_search=True,
    )
    request = {
        "role": "research",
        "messages": [{"role": "user", "content": "question"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "search_filter_http"},
            }
        ],
    }

    broker.request(request)
    broker.request(request)

    assert "tool_choice" in requests[0]
    assert "tool_choice" not in requests[1]
    assert broker.request_summary()["forced_search_synthesized_count"] == 0


def test_gpt_host_broker_keeps_native_search_tools_forbidden() -> None:
    profile = _profile("mock://local")
    broker = HostModelBroker(profile, _context(profile), _ledger(), ("research",))

    with pytest.raises(HostModelBrokerError, match="native_model_tools_forbidden"):
        broker.request(
            {
                "role": "research",
                "messages": [{"role": "user", "content": "question"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "search"},
                    }
                ],
            }
        )


def test_odr_tool_call_translation_does_not_ignore_a_later_choice() -> None:
    from benchmark_harness.systems.workers.odr import _response_tool_calls

    response = {
        "choices": [
            {"message": {"role": "assistant", "content": "first"}},
            {
                "message": {
                    "role": "assistant",
                    "content": "provider search",
                    "tool_calls": [
                        {
                            "id": "provider-search-2",
                            "type": "function",
                            "function": {
                                "name": "search_filter_http",
                                "arguments": '{"query":"q","max_results":3}',
                            },
                        }
                    ],
                }
            },
        ]
    }

    assert _response_tool_calls(response, expected_name="search_filter_http") == [
        {
            "name": "search_filter_http",
            "args": {"query": "q", "max_results": 3},
            "id": "provider-search-2",
            "type": "tool_call",
        }
    ]


@pytest.mark.parametrize("retrieval_regime", ["closed_book", "cutoff_controlled"])
def test_pinned_odr_runs_both_regimes_with_paired_tool_transcripts(
    retrieval_regime: str,
) -> None:
    descriptor = build_odr_descriptor(
        system_id="odr-transcript-fixture",
        backbone_id="fixture/model@sha",
        backbone_training_cutoff=date(2024, 1, 1),
        cutoff_source="fixture-only",
        retrieval_regime=retrieval_regime,
    )
    budget = BudgetSpec(8, 2, 0, 0, 1000, 1000, 30.0)
    binding = ExperimentBinding(
        experiment_id="odr-transcript-fixture-experiment",
        plan_id="odr-transcript-fixture-plan",
        plan_hash="odr-transcript-fixture-hash",
        cell_id="odr-transcript-fixture-cell",
        attempt_id="odr-transcript-fixture-cell:attempt-1",
        descriptor_hash=descriptor.descriptor_hash,
        task_id="fixture-task",
        retrieval_regime=retrieval_regime,
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
        seed=None,
    )
    requests: list[dict[str, Any]] = []

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        return {
            "choices": [{"message": {"role": "assistant", "content": "fixture answer"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }

    search_handler = (
        None
        if retrieval_regime == "closed_book"
        else lambda _query, _token, _index: {
            "results": [{"url": "https://fixture.example", "raw_content": "C" * 200}]
        }
    )
    adapter = OdrAdapter(
        descriptor=descriptor,
        retrieval_policy=build_retrieval_policy(retrieval_regime),
        source_dir=ODR_FIXTURE,
        search_handler=search_handler,
        search_token="fixture-search-token" if search_handler else None,
        model_responder=responder,
        repo_root=REPOSITORY_ROOT,
    )
    capability = TaskCapability(
        service_url=None,
        service_token="fixture-search-token" if search_handler else None,
        model_broker_capability="fixture-model-token",
        budget=budget,
        allow_calculation=True,
        allow_citations=search_handler is not None,
        require_structured_output=True,
        run_id="odr-transcript-fixture-run",
        trial_id="odr-transcript-fixture-trial",
        trace_mode="hash_only",
    )
    context = RunContext(
        run_id="odr-transcript-fixture-run",
        framework=descriptor.framework_id,
        adapter_version=descriptor.adapter_revision,
        model=profile,
        trace_content_mode="hash_only",
    )
    try:
        task = TaskInput("fixture-task", "fixture question", cutoff_date="2024-01-01")
        result = adapter.run_task(task, context, capability, binding)
    finally:
        adapter.close()

    assert result.raw_text == "fixture answer"
    assert requests
    assert any("tool_calls" in message for request in requests for message in request["messages"])
    assert any(
        message.get("tool_call_id") for request in requests for message in request["messages"]
    )
    if retrieval_regime == "cutoff_controlled":
        assert result.usage["search_calls"] == 1
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].result_hash
        assert result.effective_request["search_invocations"][0]["receipt_sha256"] == (
            result.tool_calls[0].result_hash
        )
        assert result.effective_request["model_broker"]["forced_search_synthesized_count"] == 1


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "q" * 2049, "max_results": 3},
        {"query": "", "max_results": 3},
        {"max_results": 3},
        {"query": "q", "max_results": 99999},
        {"query": 123, "max_results": 3},
    ],
)
def test_pinned_odr_provider_search_tool_call_invalid_fails_closed(
    arguments: dict[str, Any],
) -> None:
    descriptor = build_odr_descriptor(
        system_id="odr-invalid-provider-call",
        backbone_id="fixture/model@sha",
        backbone_training_cutoff=date(2024, 1, 1),
        cutoff_source="fixture-only",
        retrieval_regime="cutoff_controlled",
    )
    budget = BudgetSpec(8, 2, 0, 0, 1000, 1000, 30.0)
    binding = ExperimentBinding(
        experiment_id="odr-invalid-provider-call-experiment",
        plan_id="odr-invalid-provider-call-plan",
        plan_hash="odr-invalid-provider-call-hash",
        cell_id="odr-invalid-provider-call-cell",
        attempt_id="odr-invalid-provider-call-cell:attempt-1",
        descriptor_hash=descriptor.descriptor_hash,
        task_id="fixture-task",
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
        seed=None,
    )
    model_requests: list[dict[str, Any]] = []
    search_calls = 0

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        model_requests.append(request)
        search_bound = any(
            isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and tool["function"].get("name") == "search_filter_http"
            for tool in (request.get("tools") or [])
        )
        if search_bound:
            message = {
                "role": "assistant",
                "content": "provider response",
                "tool_calls": [
                    {
                        "id": "provider-invalid-search",
                        "type": "function",
                        "function": {
                            "name": "search_filter_http",
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "supervisor response"}
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }

    def search_handler(_query: str, _token: str, _index: int) -> dict[str, Any]:
        nonlocal search_calls
        search_calls += 1
        return {"results": [{"url": "https://fixture.example", "raw_content": "C" * 200}]}

    adapter = OdrAdapter(
        descriptor=descriptor,
        retrieval_policy=build_retrieval_policy("cutoff_controlled"),
        source_dir=ODR_FIXTURE,
        search_handler=search_handler,
        search_token="fixture-search-token",
        model_responder=responder,
        repo_root=REPOSITORY_ROOT,
    )
    context = RunContext(
        run_id="odr-invalid-provider-call-run",
        framework=descriptor.framework_id,
        adapter_version=descriptor.adapter_revision,
        model=profile,
        trace_content_mode="hash_only",
    )
    try:
        with pytest.raises(AdapterBoundaryError, match="odr_search_tool_call_invalid") as caught:
            adapter.run_task(
                TaskInput("fixture-task", "fixture question", cutoff_date="2024-01-01"),
                context,
                TaskCapability(
                    service_url=None,
                    service_token="fixture-search-token",
                    model_broker_capability="fixture-model-token",
                    budget=budget,
                    allow_calculation=True,
                    allow_citations=True,
                    require_structured_output=True,
                    run_id="odr-invalid-provider-call-run",
                    trial_id="odr-invalid-provider-call-trial",
                    trace_mode="hash_only",
                ),
                binding,
            )
    finally:
        adapter.close()

    assert any(
        isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and tool["function"].get("name") == "search_filter_http"
        for request in model_requests
        for tool in (request.get("tools") or [])
    )
    assert worker_failure_subreason(caught.value) == "odr_search_tool_call_invalid"
    assert search_calls == 0


def test_pinned_odr_provider_valid_search_tool_call_completes_without_synthesis() -> None:
    descriptor = build_odr_descriptor(
        system_id="odr-valid-provider-call",
        backbone_id="fixture/model@sha",
        backbone_training_cutoff=date(2024, 1, 1),
        cutoff_source="fixture-only",
        retrieval_regime="cutoff_controlled",
    )
    budget = BudgetSpec(8, 2, 0, 0, 1000, 1000, 30.0)
    binding = ExperimentBinding(
        experiment_id="odr-valid-provider-call-experiment",
        plan_id="odr-valid-provider-call-plan",
        plan_hash="odr-valid-provider-call-hash",
        cell_id="odr-valid-provider-call-cell",
        attempt_id="odr-valid-provider-call-cell:attempt-1",
        descriptor_hash=descriptor.descriptor_hash,
        task_id="fixture-task",
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
        seed=None,
    )
    model_requests: list[dict[str, Any]] = []
    search_calls = 0

    def responder(request: dict[str, Any]) -> dict[str, Any]:
        model_requests.append(request)
        search_bound = any(
            isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and tool["function"].get("name") == "search_filter_http"
            for tool in (request.get("tools") or [])
        )
        if search_bound:
            message = {
                "role": "assistant",
                "content": "provider response",
                "tool_calls": [
                    {
                        "id": "provider-valid-search",
                        "type": "function",
                        "function": {
                            "name": "search_filter_http",
                            "arguments": '{"query":"provider query","max_results":3}',
                        },
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "fixture answer"}
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }

    def search_handler(_query: str, _token: str, _index: int) -> dict[str, Any]:
        nonlocal search_calls
        search_calls += 1
        return {"results": [{"url": "https://fixture.example", "raw_content": "C" * 200}]}

    adapter = OdrAdapter(
        descriptor=descriptor,
        retrieval_policy=build_retrieval_policy("cutoff_controlled"),
        source_dir=ODR_FIXTURE,
        search_handler=search_handler,
        search_token="fixture-search-token",
        model_responder=responder,
        repo_root=REPOSITORY_ROOT,
    )
    context = RunContext(
        run_id="odr-valid-provider-call-run",
        framework=descriptor.framework_id,
        adapter_version=descriptor.adapter_revision,
        model=profile,
        trace_content_mode="hash_only",
    )
    try:
        result = adapter.run_task(
            TaskInput("fixture-task", "fixture question", cutoff_date="2024-01-01"),
            context,
            TaskCapability(
                service_url=None,
                service_token="fixture-search-token",
                model_broker_capability="fixture-model-token",
                budget=budget,
                allow_calculation=True,
                allow_citations=True,
                require_structured_output=True,
                run_id="odr-valid-provider-call-run",
                trial_id="odr-valid-provider-call-trial",
                trace_mode="hash_only",
            ),
            binding,
        )
    finally:
        adapter.close()

    assert result.raw_text == "fixture answer"
    assert search_calls == 1
    assert result.usage["search_calls"] == 1
    assert result.effective_request["model_broker"]["forced_search_synthesized_count"] == 0


def test_sanitized_diagnostic_survives_failed_attempt_artifact(tmp_path: Path) -> None:
    manifest = load_manifest(MINI_MANIFEST)
    plan = build_plan(manifest, selection="smoke")
    diagnostic = {
        "kind": "provider_http_4xx",
        "status": 400,
        "body_bytes": 7,
        "body_sha256": "b" * 64,
        "body_truncated": False,
        "wire_model_name": "gpt-4.1-mini",
        "logical_role": "research",
        "model_call_ordinal": 1,
        "request_body_sha256": "c" * 64,
        "message_shapes": [],
    }

    def factory(*_args: Any) -> Any:
        raise AdapterBoundaryError("model_http_status=400", diagnostic=diagnostic)

    task_ids = sorted({cell.task_id for cell in plan.cells})
    summary = run_experiment(
        ExperimentExecutionRequest(manifest, plan, tmp_path, selection="smoke", max_workers=3),
        dependencies=ExecutionDependencies(
            system_factory=factory,
            task_loader=lambda: [
                {"task_id": task_id, "prompt": "fixture prompt"} for task_id in task_ids
            ],
        ),
    )

    assert summary.failed_cells == len(plan.cells)
    artifact_path = (
        tmp_path
        / manifest.experiment_id
        / plan.plan_id
        / "cells"
        / plan.cells[0].cell_id
        / "attempt-1.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["error"] == "AdapterBoundaryError:model_http_status=400"
    assert artifact["error_diagnostic"] == diagnostic
    assert "fixture prompt" not in json.dumps(artifact)


def test_worker_subreason_is_written_to_attempt_ledger(tmp_path: Path) -> None:
    manifest = load_manifest(MINI_MANIFEST)
    plan = build_plan(manifest, selection="smoke")

    def factory(*_args: Any) -> Any:
        raise AdapterBoundaryError(
            "worker_error:RuntimeError",
            diagnostic={"sub_reason": "odr_search_tool_call_invalid"},
        )

    task_ids = sorted({cell.task_id for cell in plan.cells})
    run_experiment(
        ExperimentExecutionRequest(manifest, plan, tmp_path, selection="smoke", max_workers=3),
        dependencies=ExecutionDependencies(
            system_factory=factory,
            task_loader=lambda: [
                {"task_id": task_id, "prompt": "fixture prompt"} for task_id in task_ids
            ],
        ),
    )

    ledger_path = tmp_path / manifest.experiment_id / plan.plan_id / "attempts.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert rows[0]["sub_reason"] == "odr_search_tool_call_invalid"
