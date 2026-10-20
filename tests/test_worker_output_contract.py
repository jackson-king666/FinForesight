"""Hermetic tests for task-bound external-worker output paths."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from benchmark_harness.reference_evaluation.engine import parse_prediction
from benchmark_harness.systems.clock import DeclaredExperimentClock
from benchmark_harness.systems.worker_diagnostics import worker_failure_subreason
from benchmark_harness.systems.workers import gpt_researcher as gpt_worker
from benchmark_harness.systems.workers import odr as odr_worker
from benchmark_harness.systems.workers.common import (
    SEARCH_QUERY_MAX_CHARS,
    WorkerProtocolError,
    response_content,
    send_task_error,
    validate_search_query,
)
from benchmark_harness.systems.workers.gpt_researcher import (
    _load_pinned_runtime,
    _make_host_custom_retriever,
)
from benchmark_harness.systems.workers.odr import _PinnedODRGraphModel
from benchmark_harness.systems.workers.output_contract import (
    reject_fixture_sentinel,
    task_bound_answer_messages,
    task_category,
    task_output_contract,
)

REPOSITORY_ROOT = Path(__file__).parents[1]
GPT_FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "pinned_gpt"
ODR_WORKER = REPOSITORY_ROOT / "benchmark_harness/systems/workers/odr.py"
GPT_WORKER = REPOSITORY_ROOT / "benchmark_harness/systems/workers/gpt_researcher.py"


class ResearchQuestion(BaseModel):
    research_brief: str


class _OdrBroker:
    def __init__(self, task_prompt: str, *, search_query: str = "host-model search query") -> None:
        self.task_prompt = task_prompt
        self.search_query = search_query
        self.requests: list[dict[str, Any]] = []

    def request_model(
        self,
        *,
        role: str,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.requests.append(
            {
                "role": role,
                "messages": messages,
                "response_format": response_format,
                "tools": tools,
            }
        )
        request_text = json.dumps(messages, ensure_ascii=False)
        if "Generate one concise search query" in request_text:
            content: Any = {"query": self.search_query}
        else:
            content = {"research_brief": f"brief:{self.task_prompt}"}
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(content, ensure_ascii=False),
                    }
                }
            ]
        }


@pytest.mark.parametrize("task_prompt", ["first task", "second task"])
def test_odr_structured_and_tool_paths_keep_the_original_task(task_prompt: str) -> None:
    broker = _OdrBroker(task_prompt)
    model = _PinnedODRGraphModel(broker, task_prompt=task_prompt)

    structured = model.with_structured_output(ResearchQuestion)
    response = asyncio.run(structured.ainvoke([HumanMessage(content="upstream brief")]))
    assert response.research_brief == f"brief:{task_prompt}"
    assert broker.requests[0]["response_format"] == {"type": "json_object"}

    supervisor = model.bind_tools([{"name": "ConductResearch"}])
    tool_response = asyncio.run(supervisor.ainvoke([HumanMessage(content="supervisor")]))
    assert tool_response.tool_calls[0]["args"]["research_topic"] == task_prompt


class _GptBroker:
    def __init__(self, task_prompt: str) -> None:
        self.task_prompt = task_prompt
        self.model_requests: list[dict[str, Any]] = []
        self.search_requests: list[dict[str, Any]] = []

    def request_model(
        self,
        *,
        role: str,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.model_requests.append(
            {"role": role, "messages": messages, "response_format": response_format}
        )
        if role == "strategic":
            content: Any = {"questions": [self.task_prompt]}
        elif role == "fast":
            content = [{"query": self.task_prompt, "researchGoal": "task evidence"}]
        else:
            content = {
                "learnings": [
                    {
                        "insight": f"broker-result:{self.task_prompt}",
                        "sourceUrl": "https://evidence.example/report",
                    }
                ],
                "followUpQuestions": [],
            }
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(content, ensure_ascii=False),
                    }
                }
            ]
        }

    def request_search(self, *, query: str, max_results: int) -> dict[str, Any]:
        self.search_requests.append({"query": query, "max_results": max_results})
        return {
            "results": [
                {
                    "url": "https://evidence.example/report",
                    "title": "Evidence",
                    "raw_content": "receipt-backed content " * 30,
                }
            ]
        }


class _RetrieverChannel:
    def __init__(self) -> None:
        self.search_requests: list[dict[str, Any]] = []

    def request_search(self, *, query: str, max_results: int) -> dict[str, Any]:
        self.search_requests.append({"query": query, "max_results": max_results})
        return {
            "results": [
                {
                    "url": "https://evidence.example/retriever",
                    "title": "Evidence",
                    "raw_content": "content",
                }
            ]
        }


def test_search_query_validation_never_truncates_and_gpt_skips_over_limit() -> None:
    long_prompt = "A2H " + ("task-specific evidence " * 150)

    assert len(long_prompt) > 2048
    assert SEARCH_QUERY_MAX_CHARS == 2048
    assert validate_search_query("short query") == "short query"
    with pytest.raises(ValueError, match="search_query_invalid"):
        validate_search_query(long_prompt)
    with pytest.raises(ValueError, match="search_query_invalid"):
        validate_search_query("")

    channel = _RetrieverChannel()
    skip_reasons: list[str] = []
    retriever = _make_host_custom_retriever(
        object,
        channel,  # type: ignore[arg-type]
        fixture=False,
        search_skip_reasons=skip_reasons,
    )
    retriever(long_prompt).search(max_results=3)
    assert channel.search_requests == []
    assert skip_reasons == ["search_skipped_query_over_limit"]


def test_odr_search_tool_call_uses_host_query_but_research_topic_stays_task_bound() -> None:
    long_prompt = "A2H " + ("task-specific evidence " * 150)
    broker = _OdrBroker(long_prompt)
    model = _PinnedODRGraphModel(broker, task_prompt=long_prompt)

    supervisor = model.bind_tools([{"name": "search_filter_http"}])
    response = asyncio.run(supervisor.ainvoke([HumanMessage(content="supervisor")]))

    query = response.tool_calls[0]["args"]["query"]
    assert query == "host-model search query"
    assert query != long_prompt
    assert 1 <= len(query) <= SEARCH_QUERY_MAX_CHARS
    assert broker.requests[0]["tools"][0]["function"]["name"] == "search_filter_http"
    assert broker.requests[1]["tools"] is None
    assert any(
        long_prompt in message["content"]
        for message in broker.requests[1]["messages"]
        if isinstance(message.get("content"), str)
    )


def test_odr_provider_tool_call_does_not_generate_a_host_query() -> None:
    broker = _NullContentToolCallBroker("task-specific prompt")
    model = _PinnedODRGraphModel(broker, task_prompt="task-specific prompt")

    researcher = model.bind_tools([{"name": "search_filter_http"}])
    response = asyncio.run(researcher.ainvoke([HumanMessage(content="research")]))

    assert response.tool_calls[0]["args"] == {"query": "provider query", "max_results": 3}
    assert len(broker.requests) == 1
    assert "Generate one concise search query" not in json.dumps(
        broker.requests[0]["messages"], ensure_ascii=False
    )


def test_odr_search_query_generation_fails_closed_when_host_query_is_over_limit() -> None:
    long_query = "q" * (SEARCH_QUERY_MAX_CHARS + 1)
    broker = _OdrBroker("task-specific prompt", search_query=long_query)
    model = _PinnedODRGraphModel(broker, task_prompt="task-specific prompt")

    supervisor = model.bind_tools([{"name": "search_filter_http"}])
    with pytest.raises(RuntimeError, match="odr_search_query_invalid"):
        asyncio.run(supervisor.ainvoke([HumanMessage(content="supervisor")]))


class _NullContentToolCallBroker(_OdrBroker):
    def request_model(
        self,
        *,
        role: str,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if tools is None:
            return super().request_model(
                role=role,
                messages=messages,
                response_format=response_format,
                tools=tools,
            )
        self.requests.append(
            {
                "role": role,
                "messages": messages,
                "response_format": response_format,
                "tools": tools,
            }
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
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
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }


def test_odr_accepts_provider_tool_call_with_null_content() -> None:
    broker = _NullContentToolCallBroker("task-specific prompt")
    model = _PinnedODRGraphModel(broker, task_prompt="task-specific prompt")
    researcher = model.bind_tools([{"name": "search_filter_http"}])

    response = asyncio.run(researcher.ainvoke([HumanMessage(content="research")]))

    assert response.content == ""
    assert response.tool_calls == [
        {
            "name": "search_filter_http",
            "args": {"query": "provider query", "max_results": 3},
            "id": "provider-search-1",
            "type": "tool_call",
        }
    ]


@pytest.mark.parametrize("tool_calls", [None, []])
def test_response_content_rejects_null_content_without_tool_calls(
    tool_calls: list[dict[str, Any]] | None,
) -> None:
    message: dict[str, Any] = {"role": "assistant", "content": None}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls

    with pytest.raises(WorkerProtocolError, match="model_response_content_missing"):
        response_content({"choices": [{"message": message}]})


@pytest.mark.parametrize(
    "arguments",
    [
        {"query": "q" * (SEARCH_QUERY_MAX_CHARS + 1), "max_results": 3},
        {"query": "", "max_results": 3},
        {"max_results": 3},
        {"query": "q", "max_results": 99999},
        {"query": 123, "max_results": 3},
    ],
)
def test_odr_provider_search_tool_call_arguments_fail_closed(arguments: dict[str, Any]) -> None:
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "provider-search-invalid",
                            "type": "function",
                            "function": {
                                "name": "search_filter_http",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            }
        ]
    }

    with pytest.raises(RuntimeError, match="odr_search_tool_call_invalid"):
        odr_worker._response_tool_calls(response, expected_name="search_filter_http")


class _MalformedOdrResponseBroker(_OdrBroker):
    def request_model(self, **_kwargs: Any) -> dict[str, Any] | None:
        return None


def test_odr_non_mapping_model_response_fails_closed() -> None:
    broker = _MalformedOdrResponseBroker("task-specific prompt")
    model = _PinnedODRGraphModel(broker, task_prompt="task-specific prompt")

    with pytest.raises(RuntimeError, match="odr_model_response_invalid"):
        asyncio.run(model.ainvoke([HumanMessage(content="research")]))


class _CaptureErrorChannel:
    bound = True

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def send(self, _message_type: str, payload: dict[str, Any]) -> None:
        self.payloads.append(payload)


def test_send_task_error_emits_allowlisted_subreason() -> None:
    channel = _CaptureErrorChannel()

    send_task_error(channel, RuntimeError("odr_search_tool_call_invalid"))  # type: ignore[arg-type]

    assert channel.payloads == [
        {
            "error": "worker_error:RuntimeError",
            "sub_reason": "odr_search_tool_call_invalid",
        }
    ]


def test_failure_subreason_preserves_allowlisted_receipt_reason() -> None:
    assert worker_failure_subreason(RuntimeError("service_receipt_missing")) == (
        "service_receipt_missing"
    )
    assert worker_failure_subreason(RuntimeError("model_transport_error:ReadTimeout")) == (
        "model_transport_error"
    )
    assert worker_failure_subreason(RuntimeError("provider returned HTTP 503")) is None


def test_failure_subreason_reads_validated_worker_diagnostic_code() -> None:
    error = RuntimeError("worker_error:RuntimeError")
    error.diagnostic = {"code": "odr_structured_model_response_invalid"}  # type: ignore[attr-defined]

    assert worker_failure_subreason(error) == "odr_structured_model_response_invalid"


def _run_gpt_skill(
    *, closed_book: bool, task_prompt: str
) -> tuple[_GptBroker, str, list[str], list[str]]:
    broker = _GptBroker(task_prompt)
    (
        GPTResearcher,
        DeepResearchSkill,
        _log,
        _completion_outputs,
        _clock_paths,
        citations,
        search_skip_reasons,
    ) = _load_pinned_runtime(
        GPT_FIXTURE,
        broker,
        DeclaredExperimentClock(date(2025, 1, 1)),
        closed_book=closed_book,
        task_prompt=task_prompt,
        fixture=False,
    )
    config = GPTResearcher(
        query=task_prompt,
        report_type="deep",
        report_source="web",
        config_path=None,
        mcp_configs=None,
        mcp_strategy="disabled",
        verbose=False,
    )
    config.cfg.deep_research_breadth = 1
    config.cfg.deep_research_depth = 1
    config.cfg.deep_research_concurrency = 1
    result = asyncio.run(DeepResearchSkill(config).run())
    return broker, result, citations, search_skip_reasons


def test_gpt_live_seams_use_broker_content_and_receipt_urls() -> None:
    broker, result, citations, _search_skip_reasons = _run_gpt_skill(
        closed_book=False,
        task_prompt="task-specific research request",
    )

    assert "broker-result:task-specific research request" in result
    assert len(broker.search_requests) == 2
    assert all(
        request["query"] == "task-specific research request" for request in broker.search_requests
    )
    assert citations == ["https://evidence.example/report", "https://evidence.example/report"]


def test_gpt_closed_book_seam_never_requests_search() -> None:
    broker, result, citations, _search_skip_reasons = _run_gpt_skill(
        closed_book=True,
        task_prompt="closed-book task",
    )

    assert "broker-result:closed-book task" in result
    assert broker.search_requests == []
    assert citations == []


def test_gpt_over_limit_pre_search_is_skipped_and_audited() -> None:
    long_prompt = "A2H " + ("task-specific evidence " * 150)

    broker, _result, _citations, search_skip_reasons = _run_gpt_skill(
        closed_book=False,
        task_prompt=long_prompt,
    )

    assert broker.search_requests == []
    assert "pre_search_skipped_query_over_limit" in search_skip_reasons
    assert "generated_search_skipped_query_over_limit" in search_skip_reasons


class _FinalAnswerChannel:
    def __init__(self, content: str) -> None:
        self.content = content
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
        return {"choices": [{"message": {"role": "assistant", "content": self.content}}]}


@pytest.mark.parametrize(
    ("worker_name", "worker_final_stage"),
    [
        ("odr", odr_worker._request_task_bound_prediction),
        ("gpt-researcher", gpt_worker._request_task_bound_prediction),
    ],
)
@pytest.mark.parametrize(
    ("task_id", "response"),
    [
        (
            "A1-00001",
            '{"bull": 110, "base": 100, "bear": 90, "reversion_horizon": "3-6个月"}',
        ),
        ("A2F-00001", '["AAA", "BBB"]'),
        ("B-00001", '{"direction": "up", "probability_up": 0.5}'),
        ("C-00001", '{"predicted_value": 10, "metric": "revenue"}'),
        ("D-00001", '{"direction": "positive", "reasoning": "logic"}'),
        ("E-00001", '{"formula_used": "a+b", "answer": 3, "unit": "USD"}'),
    ],
)
def test_each_worker_final_stage_produces_parser_valid_a1_through_e(
    worker_name: str,
    worker_final_stage: Any,
    task_id: str,
    response: str,
) -> None:
    channel = _FinalAnswerChannel(response)
    prediction, messages = worker_final_stage(
        channel,
        task_id=task_id,
        task_prompt=f"{worker_name} task-specific prompt",
        research_context="broker-backed evidence",
        citations=[],
        fixture=False,
    )

    assert parse_prediction(task_category(task_id), prediction)["format_valid"] is True
    assert messages[1]["content"].startswith(f"Task id: {task_id}")
    assert channel.requests[0]["role"] == ("final_report" if worker_name == "odr" else "smart")
    assert channel.requests[0]["response_format"] == (
        None if task_id.startswith("A2") else {"type": "json_object"}
    )


@pytest.mark.parametrize(
    ("task_id", "response"),
    [
        (
            "A1-00001",
            '{"bull": 110, "base": 100, "bear": 90, "reversion_horizon": "3-6个月"}',
        ),
        ("A2F-00001", '["AAA", "BBB"]'),
        ("B-00001", '{"direction": "up", "probability_up": 0.5}'),
        ("C-00001", '{"predicted_value": 10, "metric": "revenue"}'),
        ("D-00001", '{"direction": "positive", "reasoning": "logic"}'),
        ("E-00001", '{"formula_used": "a+b", "answer": 3, "unit": "USD"}'),
    ],
)
def test_task_bound_contracts_remain_parser_compatible(
    task_id: str,
    response: str,
) -> None:
    category = task_category(task_id)
    messages = task_bound_answer_messages(task_id, "task-specific prompt", "evidence")
    assert "task-specific prompt" in messages[1]["content"]
    assert task_output_contract(task_id)
    assert parse_prediction(category, response)["format_valid"] is True


def test_live_worker_sources_have_no_fixture_output_sentinel() -> None:
    source = ODR_WORKER.read_text(encoding="utf-8") + GPT_WORKER.read_text(encoding="utf-8")
    assert "fixture.example" not in source
    assert "bounded benchmark question" not in source


def test_live_sentinel_guard_is_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="live_fixture_sentinel_detected"):
        reject_fixture_sentinel(
            {"query": "https://fixture.example/answer"},
            field="search_query",
        )

    with pytest.raises(RuntimeError, match="live_fixture_sentinel_detected"):
        reject_fixture_sentinel("bounded pinned finding", field="prediction")


class _SentinelOdrBroker(_OdrBroker):
    def request_model(
        self,
        *,
        role: str,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del role, messages, response_format
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "https://fixture.example/answer",
                    }
                }
            ]
        }


def test_odr_live_model_rejects_fixture_sentinel_response() -> None:
    model = _PinnedODRGraphModel(
        _SentinelOdrBroker("real task"),
        task_prompt="real task",
    )
    with pytest.raises(RuntimeError, match="live_fixture_sentinel_detected:odr_model_response"):
        asyncio.run(model.ainvoke([HumanMessage(content="real task evidence")]))
