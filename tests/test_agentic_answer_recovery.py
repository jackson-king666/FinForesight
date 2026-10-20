"""Regression coverage for answer-only recovery after tool activity.

A transient transport failure on the final answer turn (after search has
already executed) must not void the task.  The recovery may only resume the
tool-result-backfilled final turn without tools; it must never replay the
search or exceed the shared model-call budget.
"""

from __future__ import annotations

import json
import threading
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
from benchmark_harness.runner import BenchmarkRunner
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
        del headers, timeout_seconds
        self.calls.append({"url": url, "body": json})
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


class _RecoveryModelHandler(BaseHTTPRequestHandler):
    """First turn tool call, transient 500 on the first final turn, then answer."""

    requests: list[dict[str, Any]] = []
    # Number of post-tool final answer turns to fail with 500 before answering.
    final_turn_failures: int = 0

    def do_POST(self) -> None:  # noqa: N802 - HTTP handler API is fixed
        raw_body = self.rfile.read(int(self.headers["Content-Length"]))
        payload = json.loads(raw_body)
        type(self).requests.append({"path": self.path, "payload": payload})
        has_tool_return = any(
            isinstance(message, dict) and message.get("role") == "tool"
            for message in payload.get("messages", [])
        )
        if has_tool_return and type(self).final_turn_failures > 0:
            type(self).final_turn_failures -= 1
            error = {"error": {"message": "provider unavailable", "type": "server_error"}}
            encoded = json.dumps(error).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        if has_tool_return:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": '{"answer":"recovered after transient failure"}',
            }
            finish_reason = "stop"
        else:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-recovery-search",
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
            "id": "answer-recovery-smoke",
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
        }
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run_with_handler(
    baseline: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    transport = _FakeVolcTransport()
    gateway = CutoffSearchGateway(transport=transport, test_only=True)
    _RecoveryModelHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RecoveryModelHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("BENCHMARK_MODEL_ENDPOINT", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("BENCHMARK_MODEL_NAME", "recovery-test-model")
    monkeypatch.setenv("BENCHMARK_MODEL_API_KEY", "local-test-only")
    monkeypatch.delenv("BENCHMARK_MODEL_RETRIES", raising=False)
    config = ExperimentConfig(
        dataset=DatasetConfig(),
        model=ModelConfig(mode="openai_compatible", retries=2),
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
        outcome = runner.run(baselines=[baseline], run_id="answer-recovery", isolated=False)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    return {
        "outcome": outcome,
        "transport": transport,
        "requests": list(_RecoveryModelHandler.requests),
        "output_dir": tmp_path / "results-agentic" / "answer-recovery" / baseline,
    }


@pytest.mark.parametrize("baseline", ["pydantic-ai", "langgraph"], ids=["pydantic-ai", "langgraph"])
def test_transient_final_turn_failure_recovers_answer_only_without_replaying_search(
    baseline: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _RecoveryModelHandler.final_turn_failures = 1
    result = _run_with_handler(baseline, tmp_path, monkeypatch)

    assert result["outcome"]["baselines"][0]["comparable_task_count"] == 1
    # The search executed exactly once: the recovery never replayed it.
    assert len(result["transport"].calls) == 1
    requests = result["requests"]
    # Three model requests: tool turn, failed final turn, recovered final turn.
    assert len(requests) == 3
    # The recovery request is answer-only: tool results backfilled, no tools.
    recovered_request = requests[-1]["payload"]
    assert any(
        isinstance(message, dict) and message.get("role") == "tool"
        for message in recovered_request.get("messages", [])
    )
    assert "tools" not in recovered_request
    assert "tool_choice" not in recovered_request

    prediction = _read_jsonl(result["output_dir"] / "predictions.jsonl")[0]
    assert prediction["api_error"] is None
    assert prediction["tool_call_count"] == 1
    assert prediction["model_call_count"] == 3
    assert prediction["response_present"] is True
    assert prediction["raw_response_sha256"]
    trace = _read_jsonl(result["output_dir"] / "trace_events.jsonl")
    recovery_events = [
        event
        for event in trace
        if event["event_type"] == "model_retry" and event["attributes"].get("answer_only")
    ]
    assert len(recovery_events) == 1


@pytest.mark.parametrize("baseline", ["pydantic-ai", "langgraph"], ids=["pydantic-ai", "langgraph"])
def test_recovery_retries_exhausted_stays_fail_closed_with_complete_evidence(
    baseline: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _RecoveryModelHandler.final_turn_failures = 10
    result = _run_with_handler(baseline, tmp_path, monkeypatch)

    assert result["outcome"]["baselines"][0]["comparable_task_count"] == 0
    # Search ran once and is never replayed by any recovery attempt.
    assert len(result["transport"].calls) == 1
    # Tool turn + failed final turn + bounded recovery attempts (retries=2,
    # capped by the shared 4-call model budget: 2 already used).
    assert len(result["requests"]) == 4
    # Every request after the tool turn carries the backfilled tool results,
    # and every recovery request is answer-only: no tools offered, so no
    # recovery request can repeat the search.
    for request in result["requests"][1:]:
        payload = request["payload"]
        assert any(
            isinstance(message, dict) and message.get("role") == "tool"
            for message in payload.get("messages", [])
        )
    for request in result["requests"][2:]:
        assert "tools" not in request["payload"]
        assert "tool_choice" not in request["payload"]
    prediction = _read_jsonl(result["output_dir"] / "predictions.jsonl")[0]
    assert prediction["api_error"] is not None
    assert prediction["tool_call_count"] == 1
    assert prediction["response_present"] is False
    trace = _read_jsonl(result["output_dir"] / "trace_events.jsonl")
    recovery_events = [
        event
        for event in trace
        if event["event_type"] == "model_retry" and event["attributes"].get("answer_only")
    ]
    assert len(recovery_events) == 2


def test_no_tool_activity_transient_failure_keeps_full_request_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When the model dies before any tool call, there is no search side effect
    # yet; the base retry loop may replay the whole request.  Answer-only
    # recovery must not engage, and the task still fails closed.

    class _AlwaysFailHandler(BaseHTTPRequestHandler):
        requests: list[dict[str, Any]] = []

        def do_POST(self) -> None:  # noqa: N802 - HTTP handler API is fixed
            raw_body = self.rfile.read(int(self.headers["Content-Length"]))
            type(self).requests.append(json.loads(raw_body))
            error = {"error": {"message": "provider unavailable", "type": "server_error"}}
            encoded = json.dumps(error).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    transport = _FakeVolcTransport()
    gateway = CutoffSearchGateway(transport=transport, test_only=True)
    _AlwaysFailHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AlwaysFailHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("BENCHMARK_MODEL_ENDPOINT", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("BENCHMARK_MODEL_NAME", "always-fail-model")
    monkeypatch.setenv("BENCHMARK_MODEL_API_KEY", "local-test-only")
    monkeypatch.delenv("BENCHMARK_MODEL_RETRIES", raising=False)
    config = ExperimentConfig(
        dataset=DatasetConfig(),
        model=ModelConfig(mode="openai_compatible", retries=1),
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
        outcome = runner.run(baselines=["pydantic-ai"], run_id="always-fail", isolated=False)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert outcome["baselines"][0]["comparable_task_count"] == 0
    # The model died before any tool call: the search was never executed.
    assert len(transport.calls) == 0
    prediction = _read_jsonl(
        tmp_path / "results-agentic" / "always-fail" / "pydantic-ai" / "predictions.jsonl"
    )[0]
    assert prediction["tool_call_count"] == 0
    assert prediction["api_error"] is not None
