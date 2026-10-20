"""HTTPX transports that capture actual provider requests without secrets."""

from __future__ import annotations

import json
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from benchmark_harness.contracts import NetworkEvent, RunContext, sha256_text


@dataclass(frozen=True, slots=True)
class CapturedHttpCall:
    """Internal request evidence. Payload is never written directly to the ledger."""

    event: NetworkEvent
    payload: dict[str, Any] | None


class _RecordingCore:
    def __init__(
        self,
        context: RunContext,
        *,
        on_model_request: Callable[[], None] | None = None,
    ) -> None:
        self.context = context
        self._on_model_request = on_model_request
        self.calls: list[CapturedHttpCall] = []

    def _before_request(self, request: httpx.Request) -> None:
        """Consume an agentic model budget before a model request leaves process."""

        if self._on_model_request is not None and self._is_model_endpoint(request.url):
            self._on_model_request()

    @property
    def network_events(self) -> tuple[NetworkEvent, ...]:
        return tuple(call.event for call in self.calls)

    def first_model_messages(self) -> tuple[dict[str, str], ...] | None:
        for call in self.calls:
            if not call.event.is_model_endpoint or not isinstance(call.payload, dict):
                continue
            messages = call.payload.get("messages")
            if not isinstance(messages, list):
                return ()
            normalized: list[dict[str, str]] = []
            for message in messages:
                if not isinstance(message, dict):
                    return ()
                role = message.get("role")
                content = message.get("content")
                if not isinstance(role, str) or not isinstance(content, str):
                    return ()
                normalized.append({"role": role, "content": content})
            return tuple(normalized)
        return None

    def first_model_request_parameters(self) -> dict[str, Any]:
        """Persist only scalar control fields from the actual JSON request body."""

        for call in self.calls:
            if not call.event.is_model_endpoint or not isinstance(call.payload, dict):
                continue
            allowed = {
                key: value
                for key, value in call.payload.items()
                if key
                in {
                    "model",
                    "temperature",
                    "max_tokens",
                    "max_completion_tokens",
                    "seed",
                    "stream",
                    "stream_options",
                }
            }
            return allowed
        return {}

    def _record(
        self,
        request: httpx.Request,
        *,
        status_code: int | None,
        latency_seconds: float,
        error: str | None = None,
    ) -> None:
        raw_body = request.content
        payload: dict[str, Any] | None = None
        try:
            parsed = json.loads(raw_body.decode("utf-8"))
            if isinstance(parsed, dict):
                payload = parsed
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None

        url = request.url
        host = url.host or "unknown"
        if url.port is not None:
            host = f"{host}:{url.port}"
        event = NetworkEvent(
            host=host,
            method=request.method,
            path=url.path,
            body_hash=sha256_text(raw_body.decode("utf-8", errors="replace")),
            status_code=status_code,
            latency_seconds=latency_seconds,
            is_model_endpoint=self._is_model_endpoint(url),
            error=error,
        )
        self.calls.append(CapturedHttpCall(event=event, payload=payload))

    def _is_model_endpoint(self, url: httpx.URL) -> bool:
        expected = urllib.parse.urlparse(self.context.model.endpoint)
        expected_path = expected.path.rstrip("/")
        if expected_path.endswith("/chat/completions"):
            path = expected_path
        elif expected_path.endswith("/v1"):
            path = expected_path + "/chat/completions"
        else:
            path = (
                expected_path + "/v1/chat/completions" if expected_path else "/v1/chat/completions"
            )
        expected_host = expected.netloc.rsplit("@", 1)[-1]
        actual_host = url.netloc.decode("ascii")
        return actual_host == expected_host and url.path == path


class RecordingSyncTransport(httpx.BaseTransport, _RecordingCore):
    """Wrap a synchronous HTTPX transport and record every outbound call."""

    def __init__(
        self,
        context: RunContext,
        *,
        on_model_request: Callable[[], None] | None = None,
    ) -> None:
        httpx.BaseTransport.__init__(self)
        _RecordingCore.__init__(self, context, on_model_request=on_model_request)
        self._delegate = httpx.HTTPTransport(retries=0)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self._before_request(request)
        started = time.perf_counter()
        try:
            response = self._delegate.handle_request(request)
        except BaseException as exc:
            self._record(
                request,
                status_code=None,
                latency_seconds=time.perf_counter() - started,
                error=type(exc).__name__,
            )
            raise
        self._record(
            request,
            status_code=response.status_code,
            latency_seconds=time.perf_counter() - started,
        )
        return response

    def close(self) -> None:
        self._delegate.close()


class RecordingAsyncTransport(httpx.AsyncBaseTransport, _RecordingCore):
    """Wrap an asynchronous HTTPX transport and record every outbound call."""

    def __init__(
        self,
        context: RunContext,
        *,
        on_model_request: Callable[[], None] | None = None,
    ) -> None:
        httpx.AsyncBaseTransport.__init__(self)
        _RecordingCore.__init__(self, context, on_model_request=on_model_request)
        self._delegate = httpx.AsyncHTTPTransport(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._before_request(request)
        started = time.perf_counter()
        try:
            response = await self._delegate.handle_async_request(request)
        except BaseException as exc:
            self._record(
                request,
                status_code=None,
                latency_seconds=time.perf_counter() - started,
                error=type(exc).__name__,
            )
            raise
        self._record(
            request,
            status_code=response.status_code,
            latency_seconds=time.perf_counter() - started,
        )
        return response

    async def aclose(self) -> None:
        await self._delegate.aclose()
