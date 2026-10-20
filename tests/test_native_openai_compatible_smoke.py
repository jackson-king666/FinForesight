"""Local OpenAI-compatible integration smoke for both native SDK adapters."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from benchmark_harness.adapters.base import DirectTransportAdapter
from benchmark_harness.adapters.langgraph import LangGraphAdapter
from benchmark_harness.adapters.pydantic_ai import PydanticAIAdapter
from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.contracts import RunContext, TaskInput
from benchmark_harness.trace.policy_guard import StrictPolicyGuard


class _OpenAICompatibleHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - HTTP handler API is fixed
        raw_body = self.rfile.read(int(self.headers["Content-Length"]))
        payload = json.loads(raw_body)
        type(self).requests.append({"path": self.path, "payload": payload})
        response = {
            "id": "native-local-smoke",
            "object": "chat.completion",
            "created": 0,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"answer":"ok"}'},
                    "finish_reason": "stop",
                }
            ],
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


def test_native_adapters_use_one_non_streaming_chat_completions_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for environment_name in (
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING",
        "LANGCHAIN_TRACING_V2",
    ):
        monkeypatch.delenv(environment_name, raising=False)

    _OpenAICompatibleHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAICompatibleHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        profile = ResolvedModelProfile(
            mode="openai_compatible",
            endpoint=f"http://127.0.0.1:{server.server_port}/v1",
            model_name="shared-test-model",
            api_key="local-test-only",
            temperature=0.0,
            max_tokens=32,
            timeout_seconds=9.0,
            retries=0,
            seed=7,
        )
        task = TaskInput(task_id="C-00001", prompt="Return exactly one JSON object.")
        for adapter in (PydanticAIAdapter(), LangGraphAdapter(), DirectTransportAdapter()):
            context = RunContext(
                run_id="native-local-smoke",
                framework=adapter.name,
                adapter_version=adapter.adapter_version,
                model=profile,
                trace_content_mode="full",
            ).for_task(task.task_id)
            result = adapter.run(task, context)

            assert result.raw_text == '{"answer":"ok"}'
            assert StrictPolicyGuard().check(task, result).comparable is True
            assert len(result.network_events) == 1
            assert result.network_events[0].is_model_endpoint is True
            assert result.network_events[0].path == "/v1/chat/completions"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert len(_OpenAICompatibleHandler.requests) == 3
    assert all(
        request["path"] == "/v1/chat/completions" for request in _OpenAICompatibleHandler.requests
    )
    assert all(
        request["payload"]["messages"] == [{"role": "user", "content": task.prompt}]
        for request in _OpenAICompatibleHandler.requests
    )
    assert all(
        request["payload"].get("stream") in {None, False}
        for request in _OpenAICompatibleHandler.requests
    )
