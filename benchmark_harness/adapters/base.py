"""Adapter helpers, a strict mock transport, and direct transport control."""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx

from benchmark_harness.contracts import (
    AdapterResult,
    NetworkEvent,
    RunContext,
    StrictTransport,
    TaskInput,
    ToolCall,
    TransportResponse,
    canonical_input_hash,
    sha256_json,
    sha256_text,
)

_LOGGER = logging.getLogger(__name__)
_RETRY_BACKOFF_SECONDS = 0.25


def package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def model_request_payload(
    messages: Sequence[dict[str, str]], context: RunContext
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": context.model.model_name,
        "messages": list(messages),
        "temperature": context.model.temperature,
        "max_tokens": context.model.max_tokens,
    }
    if context.model.seed is not None:
        payload["seed"] = context.model.seed
    return payload


def effective_request(
    task: TaskInput,
    messages: Sequence[dict[str, str]],
    context: RunContext,
    *,
    request_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the audited request shape without credentials."""

    message_list = list(messages)
    request = {
        "run_id": context.run_id,
        "task_id": task.task_id,
        "framework": context.framework,
        "adapter_version": context.adapter_version,
        "messages": message_list,
        "messages_sha256": sha256_json(message_list),
        "prompt_sha256": sha256_text(task.prompt),
        "model": context.model.public_dict(),
        "temperature": context.model.temperature,
        "max_tokens": context.model.max_tokens,
        "timeout_seconds": context.model.timeout_seconds,
        "retries": context.model.retries,
    }
    if request_parameters:
        request["framework_request_parameters"] = dict(request_parameters)
    return request


def _endpoint_parts(endpoint: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(endpoint)
    host = parsed.netloc or parsed.path or "unknown"
    base_path = parsed.path.rstrip("/")
    if base_path.endswith("/chat/completions"):
        return host, base_path
    if base_path.endswith("/v1"):
        return host, base_path + "/chat/completions"
    return host, (base_path + "/v1/chat/completions") if base_path else "/v1/chat/completions"


def model_network_event(
    context: RunContext,
    messages: Sequence[dict[str, str]],
    *,
    status_code: int | None,
    latency_seconds: float | None,
    error: str | None = None,
    request_parameters: Mapping[str, Any] | None = None,
) -> NetworkEvent:
    host, path = _endpoint_parts(context.model.endpoint)
    payload = model_request_payload(messages, context)
    if request_parameters:
        token_parameter = request_parameters.get("wire_max_tokens_key")
        if isinstance(token_parameter, str):
            payload[token_parameter] = payload.pop("max_tokens")
        if "stream" in request_parameters:
            payload["stream"] = request_parameters["stream"]
    return NetworkEvent(
        host=host,
        method="POST",
        path=path,
        body_hash=sha256_json(payload),
        status_code=status_code,
        latency_seconds=latency_seconds,
        is_model_endpoint=True,
        error=error,
    )


class MappingMockTransport:
    """One-call mock transport used only for non-performance pipeline smoke.

    The runner may build its response mapping from the upstream mock agent, but
    this object is passed only task ID and prompt. It never receives records or
    ground truth.
    """

    def __init__(self, responses: Mapping[str, str]) -> None:
        self._responses = dict(responses)

    def complete(
        self,
        task: TaskInput,
        messages: Sequence[dict[str, str]],
        context: RunContext,
    ) -> TransportResponse:
        del messages, context
        response = self._responses.get(task.task_id)
        if response is None:
            return TransportResponse(raw_text=None, status_code=404, error="missing mock response")
        return TransportResponse(
            raw_text=response,
            usage={"input_tokens": None, "output_tokens": None, "mock": True},
            status_code=200,
            request_parameters={"transport": "mock"},
        )


def result_from_transport(
    *,
    task: TaskInput,
    context: RunContext,
    transport: StrictTransport,
    control_flow_node: str,
) -> AdapterResult:
    """Run a transport with explicit, traceable retries for unusable results.

    Provider SDK retries remain disabled in framework adapters.  Retrying here
    makes every attempt visible in local traces and network ledgers.  A retry
    is deliberately not attempted after a native tool loop has emitted a tool
    call: rerunning that loop could duplicate a search or violate its budget.
    """

    messages = task.messages
    input_hash = canonical_input_hash(task.prompt)
    max_attempts = context.model.retries + 1
    started_total = time.perf_counter()
    all_network: list[NetworkEvent] = []
    total_model_calls = 0
    final_response: TransportResponse | None = None
    final_messages: list[dict[str, str]] = list(messages)
    final_request = effective_request(task, messages, context)
    final_hash = input_hash
    final_tool_calls: tuple[ToolCall, ...] = ()
    final_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        request = effective_request(task, messages, context)
        attempt_started = time.perf_counter()
        context.emit(
            "model_request",
            model_call_index=attempt,
            effective_input_hash=input_hash,
            control_flow_node=control_flow_node,
            attributes={"request": request, "attempt": attempt, "max_attempts": max_attempts},
        )
        try:
            response = transport.complete(task, messages, context)
        except Exception as exc:  # adapter errors become auditable records, not crashes
            elapsed = time.perf_counter() - attempt_started
            network_event = model_network_event(
                context,
                messages,
                status_code=None,
                latency_seconds=elapsed,
                error=type(exc).__name__,
            )
            attempt_network = (network_event,)
            observed_messages = list(messages)
            observed_hash = input_hash
            response = TransportResponse(
                raw_text=None,
                status_code=None,
                error=type(exc).__name__,
                network_events=attempt_network,
                observed_messages=tuple(observed_messages),
            )
        else:
            elapsed = time.perf_counter() - attempt_started
            observed_messages = (
                list(messages)
                if response.observed_messages is None
                else list(response.observed_messages)
            )
            observed_hash = sha256_json(observed_messages)
            if response.network_events is None:
                network_event = model_network_event(
                    context,
                    observed_messages,
                    status_code=response.status_code,
                    latency_seconds=elapsed,
                    error=response.error,
                    request_parameters=response.request_parameters,
                )
                attempt_network = (network_event, *response.extra_network_events)
            else:
                attempt_network = (*response.network_events, *response.extra_network_events)
                network_event = next(
                    (event for event in attempt_network if event.is_model_endpoint),
                    attempt_network[0] if attempt_network else None,
                )

        all_network.extend(attempt_network)
        observed_request_parameters = dict(response.request_parameters)
        observed_request_parameters["retry_attempt"] = attempt
        observed_request_parameters["retry_max_attempts"] = max_attempts
        final_request = effective_request(
            task,
            observed_messages,
            context,
            request_parameters=observed_request_parameters,
        )
        final_messages = observed_messages
        final_hash = observed_hash
        final_response = response
        final_tool_calls = response.tool_calls
        final_error = response.error
        total_model_calls += response.model_call_count or sum(
            1 for event in attempt_network if event.is_model_endpoint
        )

        for tool_call in response.tool_calls:
            context.emit(
                "tool_call",
                model_call_index=attempt,
                effective_input_hash=input_hash,
                control_flow_node=control_flow_node,
                tool_name=tool_call.name,
                tool_arguments_hash=tool_call.arguments_hash,
                tool_result_hash=tool_call.result_hash,
            )
        for framework_event in response.framework_events:
            context.emit(
                "framework_event",
                model_call_index=attempt,
                effective_input_hash=observed_hash,
                control_flow_node=control_flow_node,
                attributes=framework_event,
            )
        context.emit(
            "model_response",
            model_call_index=attempt,
            effective_input_hash=observed_hash,
            control_flow_node=control_flow_node,
            network=asdict(network_event) if network_event else None,
            usage=response.usage,
            latency_seconds=elapsed,
            error=response.error,
        )

        retry_reason = _retry_reason(response)
        if retry_reason is None or attempt >= max_attempts or response.tool_calls:
            if retry_reason is not None and response.tool_calls:
                _LOGGER.warning(
                    "model retry suppressed after native tool activity: "
                    "task=%s framework=%s reason=%s",
                    task.task_id,
                    context.framework,
                    retry_reason,
                )
            break
        delay = _RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
        _LOGGER.warning(
            "model retry: task=%s framework=%s attempt=%s/%s reason=%s backoff=%.2fs",
            task.task_id,
            context.framework,
            attempt + 1,
            max_attempts,
            retry_reason,
            delay,
        )
        context.emit(
            "model_retry",
            model_call_index=attempt,
            effective_input_hash=observed_hash,
            control_flow_node=control_flow_node,
            attributes={
                "reason": retry_reason,
                "next_attempt": attempt + 1,
                "backoff_seconds": delay,
            },
        )
        time.sleep(delay)

    assert final_response is not None  # The loop always runs at least once.
    if _retry_reason(final_response) is not None and not final_response.tool_calls:
        final_error = final_error or "empty_model_response"
    return AdapterResult(
        raw_text=final_response.raw_text,
        usage=final_response.usage,
        latency_seconds=time.perf_counter() - started_total,
        canonical_input_hash=input_hash,
        effective_input_hash=final_hash,
        effective_messages=tuple(final_messages),
        effective_request=final_request,
        model_call_count=total_model_calls,
        tool_calls=final_tool_calls,
        network_events=tuple(all_network),
        trace_events=context.events,
        error=final_error,
    )


def _retry_reason(response: TransportResponse) -> str | None:
    """Identify timeout/empty/error responses eligible for an outer retry."""

    if response.status_code is None:
        return "transport_unavailable"
    if response.status_code >= 500:
        return "provider_status_unusable"
    if response.error:
        lowered = response.error.lower()
        if any(marker in lowered for marker in ("timeout", "connection", "network", "transport")):
            return "transport_error"
        return None
    if not isinstance(response.raw_text, str) or not response.raw_text.strip():
        return "empty_or_unusable_response"
    return None


def transient_transport_failure(exc: BaseException) -> bool:
    """Classify a raised provider failure as transient and safe to retry.

    Answer-only recovery after tool activity is attempted only for failures
    that look like temporary transport problems — timeouts, connection
    resets, provider 5xx responses — never for contract or validation errors.
    """

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None:
        marker = id(current)
        if marker in seen:
            return False
        seen.add(marker)
        status_code = getattr(current, "status_code", None)
        if isinstance(status_code, int) and status_code >= 500:
            return True
        name = type(current).__name__.lower()
        if any(
            marker in name
            for marker in (
                "timeout",
                "connection",
                "connect",
                "network",
                "transport",
                "read",
                "eof",
                "remote",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


class OpenAICompatibleTransport:
    """Minimal direct control transport; it is not a framework ranking target."""

    def complete(
        self,
        task: TaskInput,
        messages: Sequence[dict[str, str]],
        context: RunContext,
    ) -> TransportResponse:
        del task
        _, path = _endpoint_parts(context.model.endpoint)
        parsed = urllib.parse.urlparse(context.model.endpoint)
        endpoint = f"{parsed.scheme}://{parsed.netloc}{path}"
        body = json.dumps(model_request_payload(messages, context), ensure_ascii=False).encode(
            "utf-8"
        )
        headers = {"Content-Type": "application/json"}
        if context.model.api_key:
            headers["Authorization"] = f"Bearer {context.model.api_key}"
        # The direct control is not rankable, but it still uses the same actual
        # request recorder and no-redirect policy as framework baselines. This
        # prevents a control run from masking an untracked fallback endpoint.
        from benchmark_harness.adapters.recording import RecordingSyncTransport

        recorder = RecordingSyncTransport(context)
        try:
            with httpx.Client(
                transport=recorder,
                timeout=httpx.Timeout(context.model.timeout_seconds),
                follow_redirects=False,
            ) as client:
                response = client.post(endpoint, content=body, headers=headers)
            if response.status_code >= 400:
                return TransportResponse(
                    raw_text=None,
                    status_code=response.status_code,
                    error=f"HTTP {response.status_code}",
                    network_events=recorder.network_events,
                    observed_messages=recorder.first_model_messages() or tuple(messages),
                    model_call_count=sum(
                        1 for event in recorder.network_events if event.is_model_endpoint
                    ),
                    request_parameters=recorder.first_model_request_parameters(),
                )
            payload = response.json()
            choice = payload.get("choices", [{}])[0]
            message = choice.get("message") or {}
            raw_text = message.get("content")
            usage = payload.get("usage") or {}
            tool_calls = tuple(
                ToolCall(
                    name=str(call.get("function", {}).get("name", "unknown")),
                    arguments_hash=sha256_text(str(call.get("function", {}).get("arguments", ""))),
                )
                for call in (message.get("tool_calls") or [])
            )
            return TransportResponse(
                raw_text=raw_text if isinstance(raw_text, str) else None,
                usage=usage if isinstance(usage, dict) else {},
                status_code=response.status_code,
                tool_calls=tool_calls,
                network_events=recorder.network_events,
                observed_messages=recorder.first_model_messages() or tuple(messages),
                model_call_count=sum(
                    1 for event in recorder.network_events if event.is_model_endpoint
                ),
                request_parameters=recorder.first_model_request_parameters(),
            )
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            return TransportResponse(
                raw_text=None,
                status_code=None,
                error=type(exc).__name__,
                network_events=recorder.network_events,
                observed_messages=recorder.first_model_messages() or tuple(messages),
                model_call_count=sum(
                    1 for event in recorder.network_events if event.is_model_endpoint
                ),
                request_parameters=recorder.first_model_request_parameters(),
            )


class DirectTransportAdapter:
    """OpenAI-compatible control baseline, excluded from framework ranking."""

    name = "direct_transport"
    adapter_version = "0.1.0"

    def __init__(self, transport: StrictTransport | None = None) -> None:
        self.transport = transport or OpenAICompatibleTransport()

    def run(self, task: TaskInput, context: RunContext) -> AdapterResult:
        return result_from_transport(
            task=task,
            context=context,
            transport=self.transport,
            control_flow_node="direct_transport",
        )
