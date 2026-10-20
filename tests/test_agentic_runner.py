"""Agentic profile integration tests with a shared fake Volc gateway."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from benchmark_harness.config import (
    DatasetConfig,
    ExecutionConfig,
    ExperimentConfig,
    ModelConfig,
    SearchConfig,
    TraceConfig,
)
from benchmark_harness.runner import BenchmarkRunner, _cutoff_from_record, _event_date_from_record
from benchmark_harness.search import CutoffSearchGateway, VolcHttpResponse


class _FakeSource:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def preflight(self) -> dict[str, Any]:
        return self.provenance()

    def provenance(self) -> dict[str, Any]:
        return {
            "source": "injected-test-source",
            "dataset_revision": "test-v0.3",
            "record_count": len(self.records),
        }

    def load_records(
        self, *, categories: list[str] | None = None, status: str | None = "ready"
    ) -> list[dict[str, Any]]:
        records = [
            record for record in self.records if status is None or record["status"] == status
        ]
        if categories:
            selected = {category.upper() for category in categories}
            records = [record for record in records if record["category"] in selected]
        return list(records)


class _FakeEvaluation:
    def parse_prediction(self, category: str, raw_text: str) -> dict[str, Any]:
        return {"format_valid": bool(raw_text), "category": category, "raw": raw_text}

    def evaluate_prediction(
        self, category: str, parsed: dict[str, Any], record: dict[str, Any]
    ) -> dict[str, Any]:
        return {"format_valid": parsed["format_valid"], "category": category}

    def summarize_results(
        self,
        predictions: list[dict[str, Any]],
        skipped: list[dict[str, Any]],
        *,
        agent: str,
        model: str,
    ) -> dict[str, Any]:
        return {
            "agent": agent,
            "model": model,
            "evaluated": len(predictions),
            "skipped": len(skipped),
        }

    def mock_response(self, record: dict[str, Any]) -> str:
        return json.dumps({"task_id": record["task_id"]})


class _FakeVolcTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> VolcHttpResponse:
        self.calls.append({"url": url, "body": json, "headers": headers})
        return VolcHttpResponse(
            status_code=200,
            request_id="fake-volc",
            payload={
                "Result": {
                    "WebResults": [
                        {
                            "Url": "https://source.example/pre-cutoff",
                            "Title": "Pre cutoff",
                            "Content": "safe source evidence",
                            "PublishTime": "2024-06-01T12:00:00Z",
                        }
                    ]
                }
            },
        )


class _ToolCallingModelHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible fake that makes the model, not the runner, search."""

    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - HTTP handler API is fixed
        raw_body = self.rfile.read(int(self.headers["Content-Length"]))
        payload = json.loads(raw_body)
        type(self).requests.append({"path": self.path, "payload": payload})
        messages = payload.get("messages", [])
        has_tool_return = any(
            isinstance(message, dict) and message.get("role") == "tool" for message in messages
        )
        if has_tool_return:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": '{"answer":"from native tool"}',
            }
            finish_reason = "stop"
        else:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-native-search",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"model selected historical evidence"}',
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        response = {
            "id": "native-tool-smoke",
            "object": "chat.completion",
            "created": 0,
            "model": payload["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class _FinalAnswerOnlyCompatibilityHandler(BaseHTTPRequestHandler):
    """Provider fixture that ignores tool_choice but answers when tools are omitted."""

    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - HTTP handler API is fixed
        raw_body = self.rfile.read(int(self.headers["Content-Length"]))
        payload = json.loads(raw_body)
        type(self).requests.append({"path": self.path, "payload": payload})
        tool_result_messages = [
            message
            for message in payload.get("messages", [])
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        tool_results = len(tool_result_messages)
        latest_tool_result_count = 0
        for message in reversed(payload.get("messages", [])):
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                break
            if message.get("role") == "tool":
                latest_tool_result_count += 1
        if latest_tool_result_count > 1:
            error_response = {"error": {"code": "parallel_tool_returns_not_supported"}}
            encoded = json.dumps(error_response).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        if tool_results and any(
            isinstance(message, dict) and "reasoning_content" in message
            for message in payload.get("messages", [])
        ):
            encoded = json.dumps({"error": {"code": "reasoning_content_not_supported"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        if tool_results >= 3 and not payload.get("tools"):
            message: dict[str, Any] = {"role": "assistant", "content": '{"answer":"final"}'}
            finish_reason = "stop"
        else:
            message = {
                "role": "assistant",
                "content": None,
                "reasoning_content": "provider-private reasoning",
                "tool_calls": [
                    {
                        "id": f"call-{tool_results + 1}",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"historical evidence"}',
                        },
                    }
                ],
            }
            if tool_results == 0:
                message["tool_calls"].append(
                    {
                        "id": "call-parallel-2",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"second historical evidence"}',
                        },
                    }
                )
            finish_reason = "tool_calls"
        response = {
            "id": "final-answer-only-smoke",
            "object": "chat.completion",
            "created": 0,
            "model": payload["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def _records() -> list[dict[str, Any]]:
    return [
        {
            "task_id": "A1-00001",
            "category": "A1",
            "variant": None,
            "time_band": "T1",
            "status": "ready",
            "prompt": "Answer A1 using cutoff evidence.",
            "seed": {"cutoff_date": "2024-06-01"},
            "ground_truth": {},
        },
        {
            "task_id": "D-00001",
            "category": "D",
            "variant": None,
            "time_band": "T1",
            "status": "ready",
            "prompt": "Counterfactual task without an allowed cutoff.",
            "seed": {},
            "ground_truth": {},
        },
        {
            "task_id": "E-00001",
            "category": "E",
            "variant": None,
            "time_band": "T1",
            "status": "ready",
            "prompt": "Formula task without an allowed cutoff.",
            "seed": {},
            "ground_truth": {},
        },
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_pydantic_ai_omits_tools_for_final_answer_after_budget_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = _FakeVolcTransport()
    gateway = CutoffSearchGateway(transport=transport, test_only=True)
    _FinalAnswerOnlyCompatibilityHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FinalAnswerOnlyCompatibilityHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("BENCHMARK_MODEL_ENDPOINT", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("BENCHMARK_MODEL_NAME", "native-tool-test-model")
    monkeypatch.setenv("BENCHMARK_MODEL_API_KEY", "local-test-only")
    config = ExperimentConfig(
        dataset=DatasetConfig(),
        model=ModelConfig(mode="openai_compatible"),
        search=SearchConfig(test_only=True),
        execution=ExecutionConfig(input_policy="agentic_web_search"),
        trace=TraceConfig(content_mode="hash_only", results_dir=tmp_path / "results-agentic"),
    )
    runner = BenchmarkRunner(
        config,
        dataset_source_factory=lambda _config: _FakeSource(_records()[:1]),
        evaluation_engine=_FakeEvaluation(),
        search_gateway_factory=lambda _config: gateway,
    )

    try:
        outcome = runner.run(baselines=["pydantic-ai"], run_id="final-answer-only", isolated=False)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    baseline = outcome["baselines"][0]
    prediction = _read_jsonl(
        tmp_path / "results-agentic" / "final-answer-only" / "pydantic-ai" / "predictions.jsonl"
    )[0]
    final_request = _FinalAnswerOnlyCompatibilityHandler.requests[-1]["payload"]
    assert baseline["comparable_task_count"] == 1
    assert prediction["tool_call_count"] == 3
    assert prediction["api_error"] is None
    assert "tools" not in final_request
    assert "tool_choice" not in final_request


def test_runner_normalizes_hub_decoded_seed_dates_before_cutoff_policy() -> None:
    record = {
        "seed": {
            "cutoff_date": datetime(2025, 1, 15),
            "event_date": datetime(2025, 1, 15),
        }
    }

    assert _cutoff_from_record(record) == "2025-01-15"
    assert _event_date_from_record(record) == "2025-01-15"


def test_agentic_frameworks_use_native_model_tools_and_skip_d_e_without_activity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = _FakeVolcTransport()
    gateway = CutoffSearchGateway(transport=transport, test_only=True)
    _ToolCallingModelHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ToolCallingModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("BENCHMARK_MODEL_ENDPOINT", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("BENCHMARK_MODEL_NAME", "native-tool-test-model")
    monkeypatch.setenv("BENCHMARK_MODEL_API_KEY", "local-test-only")
    config = ExperimentConfig(
        dataset=DatasetConfig(),
        model=ModelConfig(mode="openai_compatible"),
        search=SearchConfig(test_only=True),
        execution=ExecutionConfig(input_policy="agentic_web_search"),
        trace=TraceConfig(content_mode="hash_only", results_dir=tmp_path / "results-agentic"),
    )
    runner = BenchmarkRunner(
        config,
        dataset_source_factory=lambda _config: _FakeSource(_records()),
        evaluation_engine=_FakeEvaluation(),
        search_gateway_factory=lambda _config: gateway,
    )

    try:
        outcome = runner.run(
            baselines=["pydantic-ai", "langgraph"],
            run_id="agentic-fake",
            isolated=False,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert {row["baseline"] for row in outcome["baselines"]} == {"pydantic-ai", "langgraph"}
    assert len(transport.calls) == 2
    assert all(call["url"] == "https://mercury.volcengineapi.com/" for call in transport.calls)
    assert {call["body"]["TimeRange"] for call in transport.calls} == {"2019-06-01..2024-06-01"}
    assert {call["body"]["Query"] for call in transport.calls} == {
        "model selected historical evidence"
    }
    tool_requests = [
        request["payload"]
        for request in _ToolCallingModelHandler.requests
        if request["payload"].get("tools")
        and not any(
            message.get("role") == "tool"
            for message in request["payload"].get("messages", [])
            if isinstance(message, dict)
        )
    ]
    assert len(tool_requests) == 2
    assert len(_ToolCallingModelHandler.requests) == 4
    assert all(payload["tools"][0]["function"]["name"] == "search_web" for payload in tool_requests)
    assert all(payload.get("parallel_tool_calls") is False for payload in tool_requests)
    assert all(
        payload["tools"][0]["function"]["parameters"]["properties"]["query"]["type"] == "string"
        for payload in tool_requests
    )
    assert all(
        payload["messages"][0]
        == {"role": "system", "content": gateway.tool_contract.tool_policy_text}
        for payload in tool_requests
    )

    for framework in ("pydantic-ai", "langgraph"):
        directory = tmp_path / "results-agentic" / "agentic-fake" / framework
        predictions = {row["task_id"]: row for row in _read_jsonl(directory / "predictions.jsonl")}
        ledger = _read_jsonl(directory / "network_ledger.jsonl")
        audit = _read_jsonl(directory / "search_audit.jsonl")
        run_config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))

        assert predictions["A1-00001"]["tool_call_count"] == 1
        assert predictions["A1-00001"]["model_call_count"] == 2
        assert predictions["D-00001"]["skipped_no_cutoff"] is True
        assert predictions["E-00001"]["skipped_no_cutoff"] is True
        assert predictions["D-00001"]["model_call_count"] == 0
        assert predictions["E-00001"]["tool_call_count"] == 0
        assert {row["task_id"] for row in audit} == {"A1-00001"}
        assert sum(row.get("network_class") == "web_search" for row in ledger) == 1
        model_rows = [
            row for row in ledger if row.get("task_id") == "A1-00001" and "network_class" not in row
        ]
        assert len(model_rows) == 2
        assert all(row["is_model_endpoint"] is True for row in model_rows)
        trace = _read_jsonl(directory / "trace_events.jsonl")
        tool_events = [row for row in trace if row["event_type"] == "tool_call"]
        assert len(tool_events) == 1
        assert tool_events[0]["tool_name"] == "search_web"
        assert tool_events[0]["control_flow_node"] != "cutoff_search_gateway"
        assert run_config["search"]["tool_contract_hash"] == gateway.tool_contract.hash
        assert run_config["search"]["tool_contract"]["tool_policy_sha256"]
        assert run_config["search"]["test_only"] is True
