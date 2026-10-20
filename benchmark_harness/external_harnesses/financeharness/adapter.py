"""Host-side FinanceHarness adapter and parent-owned capability brokers."""

from __future__ import annotations

import json
import select
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from benchmark_harness.adapters.recording import RecordingSyncTransport
from benchmark_harness.config import DeepResearchConfig, ExternalHarnessConfig, ResolvedModelProfile
from benchmark_harness.contracts import (
    TEMPORAL_FILTER_NETWORK_CLASS,
    AdapterResult,
    NetworkEvent,
    RunContext,
    TaskInput,
    ToolCall,
    sha256_json,
    sha256_text,
)
from benchmark_harness.external_harnesses.checkout import (
    ExternalHarnessError,
    ExternalHarnessManager,
)
from benchmark_harness.external_harnesses.financeharness.policy import (
    PURE_DEFERRED_TOOL_PREFIXES,
    SAFE_TOOL_NAMES,
)
from benchmark_harness.external_harnesses.ipc import (
    IpcChannel,
    IpcProtocolError,
    build_child_environment,
    terminate_process_group,
)
from benchmark_harness.model_identity import verify_provider_response_model
from benchmark_harness.search_service import (
    SearchReceiptRef,
    SearchServiceClient,
    SearchServiceSession,
)


class FinanceHarnessBoundaryError(RuntimeError):
    """Raised when the external worker violates the host IPC boundary."""


_MODEL_REQUEST_FIELDS = frozenset(
    {
        "messages",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "presence_penalty",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "response_format",
    }
)


def _completion_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    base = parsed.path.rstrip("/")
    if base.endswith("/chat/completions"):
        path = base
    elif base.endswith("/v1"):
        path = base + "/chat/completions"
    else:
        path = (base + "/v1/chat/completions") if base else "/v1/chat/completions"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


class DeepModelBroker:
    """Parent-only model broker used by the child's custom IPC client."""

    def __init__(
        self,
        profile: ResolvedModelProfile,
        context: RunContext,
        *,
        max_calls: int,
        responder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.profile = profile
        self.context = context
        self.max_calls = max_calls
        self.responder = responder
        self.call_count = 0
        self.network_events: list[NetworkEvent] = []
        self.requests: list[dict[str, Any]] = []
        self.provider_model_names: list[str] = []
        self.system_fingerprints: list[str] = []

    @property
    def provider_response_model_names(self) -> list[str]:
        """Observed provider model attestations under the explicit field name."""

        return self.provider_model_names

    def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        max_attempts = self.profile.retries + 1
        last_error: FinanceHarnessBoundaryError | None = None
        for attempt in range(max_attempts):
            if self.call_count >= self.max_calls:
                raise FinanceHarnessBoundaryError("deep_model_call_budget_exceeded")
            request = _normalize_model_request(payload, self.profile)
            self.requests.append(request)
            self.call_count += 1
            try:
                response = self._request_once(request)
                if not _usable_model_response(response):
                    raise FinanceHarnessBoundaryError("empty_model_response")
                provider_model_name = _verify_response_model(response, self.profile)
                if provider_model_name is not None:
                    self.provider_model_names.append(provider_model_name)
                fingerprint = response.get("system_fingerprint")
                if isinstance(fingerprint, str) and fingerprint.strip():
                    self.system_fingerprints.append(fingerprint.strip())
                return response
            except FinanceHarnessBoundaryError as exc:
                last_error = exc
                if attempt + 1 >= max_attempts:
                    raise
                time.sleep(0.25 * (2**attempt))
        raise last_error or FinanceHarnessBoundaryError("model_request_failed")

    def _request_once(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.responder is not None:
            try:
                response = self.responder(request)
            except Exception as exc:
                self.network_events.append(
                    _synthetic_model_event(request, status_code=None, error=type(exc).__name__)
                )
                raise FinanceHarnessBoundaryError(
                    f"mock_model_responder_error:{type(exc).__name__}"
                ) from exc
            if not isinstance(response, dict):
                self.network_events.append(
                    _synthetic_model_event(request, status_code=200, error="non_object_response")
                )
                raise FinanceHarnessBoundaryError("mock model responder returned a non-object")
            self.network_events.append(_synthetic_model_event(request, status_code=200))
            return response
        if self.profile.mode == "mock":
            self.network_events.append(_synthetic_model_event(request, status_code=200))
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "mock FinanceHarness response",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {},
            }
        if self.profile.mode != "openai_compatible":
            raise FinanceHarnessBoundaryError("unsupported_parent_model_mode")

        recorder = RecordingSyncTransport(self.context)
        headers = {"Content-Type": "application/json"}
        if self.profile.api_key:
            headers["Authorization"] = f"Bearer {self.profile.api_key}"
        try:
            with httpx.Client(
                transport=recorder,
                timeout=httpx.Timeout(self.profile.timeout_seconds),
                follow_redirects=False,
            ) as client:
                response = client.post(
                    _completion_endpoint(self.profile.endpoint),
                    json=request,
                    headers=headers,
                )
            value = response.json()
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            self.network_events.extend(recorder.network_events)
            raise FinanceHarnessBoundaryError(
                f"model_transport_error:{type(exc).__name__}"
            ) from exc
        self.network_events.extend(recorder.network_events)
        if response.status_code >= 400:
            raise FinanceHarnessBoundaryError(f"model_http_status={response.status_code}")
        if not isinstance(value, dict):
            raise FinanceHarnessBoundaryError("model_response_not_object")
        return value


def _synthetic_model_event(
    request: Mapping[str, Any], *, status_code: int | None, error: str | None = None
) -> NetworkEvent:
    return NetworkEvent(
        host="mock.parent.model",
        method="POST",
        path="/v1/chat/completions",
        body_hash=sha256_json(request),
        status_code=status_code,
        latency_seconds=0.0,
        is_model_endpoint=True,
        error=error,
    )


def _usable_model_response(response: Mapping[str, Any]) -> bool:
    choices = response.get("choices")
    return isinstance(choices, list) and bool(choices)


def _verify_response_model(
    response: Mapping[str, Any], profile: ResolvedModelProfile
) -> str | None:
    """Reject a provider response that attests to a different model."""

    try:
        return verify_provider_response_model(response, profile)
    except ValueError as exc:
        raise FinanceHarnessBoundaryError("model_identity_mismatch") from exc


def _normalize_model_request(
    payload: Mapping[str, Any], profile: ResolvedModelProfile
) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise FinanceHarnessBoundaryError("model_request_messages_missing")
    request = {key: value for key, value in payload.items() if key in _MODEL_REQUEST_FIELDS}
    request["model"] = profile.model_name
    request["temperature"] = profile.temperature
    request["max_tokens"] = profile.max_tokens
    if profile.seed is not None:
        request["seed"] = profile.seed
    request["stream"] = False
    tools = request.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise FinanceHarnessBoundaryError("model_request_tools_malformed")
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if not isinstance(name, str) or (
                name not in SAFE_TOOL_NAMES and not name.startswith(PURE_DEFERRED_TOOL_PREFIXES)
            ):
                raise FinanceHarnessBoundaryError("model_request_tool_not_allowlisted")
    return request


@dataclass(frozen=True, slots=True)
class DeepRunOutput:
    result: AdapterResult
    search_outcomes: tuple[Any, ...]
    search_ledger_rows: tuple[dict[str, Any], ...]
    research_audit_rows: tuple[dict[str, Any], ...]
    citations: tuple[str, ...]
    external_manifest: dict[str, Any]
    child_network_call_count: int = 0
    filter_model_call_count: int = 0
    child_search_service_call_count: int = 0
    child_forbidden_network_call_count: int = 0
    service_close: dict[str, Any] | None = None


class FinanceHarnessAdapter:
    """Typed host adapter for the restricted upstream ``Agent`` API."""

    name = "financeharness"
    adapter_version = "financeharness-external/v2"

    def __init__(
        self,
        manager: ExternalHarnessManager,
        *,
        deep_config: DeepResearchConfig | None = None,
        harness_config: ExternalHarnessConfig | None = None,
        search_service: SearchServiceClient | None = None,
        model_responder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        worker_command: list[str] | None = None,
        process_factory: Callable[..., Any] | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        self.manager = manager
        self.deep_config = deep_config or DeepResearchConfig()
        self.harness_config = harness_config or manager.config
        self.search_service = search_service
        self.model_responder = model_responder
        self.worker_command = worker_command
        self.process_factory = process_factory or subprocess.Popen
        self.repo_root = Path(repo_root or manager.repo_root).resolve()
        self.last_output: DeepRunOutput | None = None

    def run(self, task: TaskInput, context: RunContext) -> AdapterResult:
        """Adapter protocol entry point; details are available in ``last_output``."""

        return self.run_deep(task, context).result

    def run_deep(
        self,
        task: TaskInput,
        context: RunContext,
        *,
        retrieval_regime: str = "cutoff_controlled",
    ) -> DeepRunOutput:
        if retrieval_regime not in {
            "closed_book",
            "cutoff_controlled",
            "unrestricted_current_web",
        }:
            raise FinanceHarnessBoundaryError("unsupported_retrieval_regime")
        if retrieval_regime == "cutoff_controlled" and not task.cutoff_date:
            raise FinanceHarnessBoundaryError("deep_research_requires_cutoff")
        try:
            manifest = self.manager.verify()
        except ExternalHarnessError as exc:
            raise FinanceHarnessBoundaryError(str(exc)) from exc
        if retrieval_regime != "closed_book" and self.search_service is None:
            raise FinanceHarnessBoundaryError("deep_search_service_missing")
        service_session = None
        if retrieval_regime != "closed_book":
            assert self.search_service is not None
            assert task.cutoff_date is not None
            self.search_service.preflight()
            # SearchServiceClient enforces the configured external declaration;
            # injected test doubles may expose only the adapter protocol.
            expected_cohort = getattr(self.search_service, "expected_service_cohort_sha256", None)
            service_session = self.search_service.open_session(
                run_id=context.run_id,
                task_id=task.task_id,
                profile="deep_research",
                baseline=self.name,
                cutoff_date=task.cutoff_date,
                policy_version="deep_research/v2",
                expected_service_cohort_sha256=expected_cohort,
            )
        search: Any = service_session
        filter_network_events: list[NetworkEvent] = []
        broker = DeepModelBroker(
            context.model,
            context,
            max_calls=self.deep_config.max_model_calls,
            responder=self.model_responder,
        )
        process = None
        self._stderr_bytes = 0
        child_network_call_count = 0
        audit_rows: list[dict[str, Any]] = []
        service_receipts_for_close: tuple[SearchReceiptRef, ...] | None = None
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        try:
            self._active_task_id = task.task_id
            process = self._start_process(task)
            if process.stdout is None or process.stdin is None:
                raise FinanceHarnessBoundaryError("external_worker_stdio_unavailable")
            channel = IpcChannel(
                process.stdout,
                process.stdin,
                task_id=task.task_id,
                max_frame_bytes=self.deep_config.max_frame_bytes,
            )
            hello = self._read(channel, process, started)
            if hello.type != "hello":
                raise FinanceHarnessBoundaryError("external_worker_hello_required")
            if hello.payload.get("protocol") != self.harness_config.protocol:
                raise FinanceHarnessBoundaryError("external_worker_protocol_mismatch")
            if hello.payload.get("adapter") != self.name:
                raise FinanceHarnessBoundaryError("external_worker_adapter_mismatch")
            if hello.payload.get("adapter_version") != self.harness_config.version:
                raise FinanceHarnessBoundaryError("external_worker_adapter_version_mismatch")
            if hello.payload.get("child_network") != "forbidden":
                raise FinanceHarnessBoundaryError("external_worker_egress_policy_mismatch")
            task_start_payload: dict[str, Any] = {
                "task_id": task.task_id,
                "prompt": task.prompt,
                "cutoff_date": task.cutoff_date,
                "today": task.cutoff_date,
                "retrieval_regime": retrieval_regime,
                "grounding_review": self.deep_config.grounding_review,
            }
            if service_session is not None:
                task_start_payload["search_service"] = {
                    "base_url": self.search_service.base_url if self.search_service else None,
                    **service_session.descriptor.public_dict(),
                    "task_token": service_session.descriptor.task_token,
                }
            channel.send(
                "task.start",
                task_start_payload,
                request_id=request_id,
            )
            while True:
                envelope = self._read(channel, process, started)
                if envelope.type == "model.request":
                    response = broker.request(envelope.payload)
                    channel.send(
                        "model.response",
                        response,
                        request_id=envelope.request_id,
                    )
                    continue
                if envelope.type in {"search.request", "search.response"}:
                    raise FinanceHarnessBoundaryError("legacy_search_ipc_frame_rejected")
                if envelope.type == "audit.event":
                    row = dict(envelope.payload)
                    row.setdefault("task_id", task.task_id)
                    audit_rows.append(row)
                    network_count = row.get("network_call_count", 0)
                    if (
                        not isinstance(network_count, int)
                        or isinstance(network_count, bool)
                        or network_count < 0
                    ):
                        raise FinanceHarnessBoundaryError("external_worker_audit_count_invalid")
                    child_network_call_count += network_count
                    continue
                if envelope.type == "task.result":
                    service_receipts_for_close = _validated_service_receipts(
                        envelope.payload.get("service_receipts"),
                        service_session=service_session,
                    )
                    child_search_service_call_count = _validated_service_call_count(
                        envelope.payload.get("child_search_service_call_count", 0),
                        expected=len(service_receipts_for_close),
                        service_session=service_session,
                    )
                    service_close = (
                        service_session.close(
                            status="completed",
                            receipts=service_receipts_for_close,
                        )
                        if service_session is not None
                        else None
                    )
                    output = self._finish_result(
                        task=task,
                        context=context,
                        payload=envelope.payload,
                        broker=broker,
                        search=search,
                        retrieval_regime=retrieval_regime,
                        filter_network_events=filter_network_events,
                        audit_rows=audit_rows,
                        manifest=manifest,
                        child_network_call_count=child_network_call_count,
                        service_close=service_close,
                        child_search_service_call_count=child_search_service_call_count,
                        child_forbidden_network_call_count=_validated_nonnegative_count(
                            envelope.payload.get("child_forbidden_network_call_count", 0),
                            code="child_forbidden_network_call_count_invalid",
                        ),
                        service_session=service_session,
                        elapsed=time.perf_counter() - started,
                        service_receipts=service_receipts_for_close,
                    )
                    self.last_output = output
                    return output
                if envelope.type == "task.error":
                    service_receipts_for_close = _validated_service_receipts(
                        envelope.payload.get("service_receipts"),
                        service_session=service_session,
                    )
                    _validated_service_call_count(
                        envelope.payload.get("child_search_service_call_count", 0),
                        expected=len(service_receipts_for_close),
                        service_session=service_session,
                    )
                    raise FinanceHarnessBoundaryError(
                        str(envelope.payload.get("error", "external_worker_task_error"))
                    )
                raise FinanceHarnessBoundaryError(f"unexpected_external_message:{envelope.type}")
        except (IpcProtocolError, subprocess.SubprocessError) as exc:
            raise FinanceHarnessBoundaryError(str(exc)) from exc
        finally:
            if service_session is not None and service_session.close_result is None:
                try:
                    if service_receipts_for_close is None:
                        service_session.close(status="aborted")
                    else:
                        service_session.close(
                            status="aborted",
                            receipts=service_receipts_for_close,
                        )
                except Exception:
                    pass
            if process is not None:
                self._close_process(process)

    def _start_process(self, task: TaskInput) -> Any:
        command = self.worker_command or [
            str(self.manager.venv_python),
            "-m",
            "benchmark_harness.external_harnesses.financeharness.worker",
            "--source",
            str(self.manager.source_dir),
            "--task-id",
            task.task_id,
        ]
        env = build_child_environment(
            self.manager.source_dir,
            self.repo_root,
            python_executable=self.manager.venv_python,
        )
        try:
            process = self.process_factory(
                command,
                cwd=str(self.repo_root),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                process._benchmark_start_new_session = True
            except Exception:
                pass
            return process
        except (OSError, TypeError) as exc:
            raise FinanceHarnessBoundaryError("unable_to_start_external_worker") from exc

    def _read(self, channel: IpcChannel, process: Any, started: float):
        while True:
            remaining = self.deep_config.task_timeout_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                terminate_process_group(process)
                raise FinanceHarnessBoundaryError("external_worker_timeout")
            streams = [process.stdout]
            if process.stderr is not None:
                streams.append(process.stderr)
            try:
                ready, _, _ = select.select(streams, [], [], min(remaining, 1.0))
            except (OSError, TypeError):
                ready = [process.stdout]
            if process.stderr is not None and process.stderr in ready:
                reader = getattr(process.stderr, "read1", process.stderr.read)
                chunk = reader(64 * 1024)
                if chunk:
                    self._stderr_bytes += len(
                        chunk.encode("utf-8") if isinstance(chunk, str) else chunk
                    )
                    if self._stderr_bytes > self.deep_config.max_stderr_bytes:
                        terminate_process_group(process)
                        raise FinanceHarnessBoundaryError("external_worker_stderr_budget_exceeded")
            if process.stdout in ready:
                return channel.read()
            if process.poll() is not None:
                raise FinanceHarnessBoundaryError("external_worker_eof")

    def _finish_result(
        self,
        *,
        task: TaskInput,
        context: RunContext,
        payload: dict[str, Any],
        broker: DeepModelBroker,
        search: Any,
        retrieval_regime: str,
        audit_rows: list[dict[str, Any]],
        filter_network_events: list[NetworkEvent],
        manifest: dict[str, Any],
        child_network_call_count: int,
        elapsed: float,
        service_close: dict[str, Any] | None = None,
        child_search_service_call_count: int = 0,
        child_forbidden_network_call_count: int = 0,
        service_session: SearchServiceSession | None = None,
        service_receipts: tuple[SearchReceiptRef, ...] | None = None,
    ) -> DeepRunOutput:
        prediction = payload.get("prediction")
        raw_text = prediction if isinstance(prediction, str) else payload.get("raw_text")
        raw_text = raw_text if isinstance(raw_text, str) else None
        messages = payload.get("messages")
        if not isinstance(messages, list):
            messages = (
                broker.requests[0].get("messages", task.messages)
                if broker.requests
                else task.messages
            )
        if not isinstance(messages, list):
            messages = task.messages
        effective_messages = tuple(
            item
            for item in messages
            if isinstance(item, dict) and isinstance(item.get("role"), str)
        )
        effective_hash = sha256_json(list(effective_messages))
        tool_calls = _tool_calls_from_payload(payload)
        search_outcomes = tuple(getattr(search, "outcomes", ()))
        for outcome in search_outcomes:
            context.emit(
                "tool_call",
                effective_input_hash=effective_hash,
                control_flow_node="financeharness_external_worker",
                tool_name="search",
                tool_arguments_hash=outcome.query_sha256,
                tool_result_hash=outcome.response_sha256,
            )
        for index, request in enumerate(broker.requests, start=1):
            context.emit(
                "model_request",
                model_call_index=index,
                effective_input_hash=effective_hash,
                control_flow_node="financeharness_parent_model_broker",
                attributes={"request_sha256": sha256_json(request)},
            )
            context.emit(
                "model_response",
                model_call_index=index,
                effective_input_hash=effective_hash,
                control_flow_node="financeharness_parent_model_broker",
            )
        service_filter_events = _service_filter_events(service_close)
        filter_events = (*filter_network_events, *service_filter_events)
        handoff_receipts = (
            service_receipts
            if service_receipts is not None
            else (service_session.receipts if service_session else ())
        )
        for offset, event in enumerate(filter_events, start=broker.call_count + 1):
            context.emit(
                "model_request",
                model_call_index=offset,
                effective_input_hash=effective_hash,
                control_flow_node="financeharness_parent_temporal_filter",
                attributes={
                    "request_sha256": event.body_hash,
                    "filter": True,
                    "network_class": TEMPORAL_FILTER_NETWORK_CLASS,
                },
            )
            context.emit(
                "model_response",
                model_call_index=offset,
                effective_input_hash=effective_hash,
                control_flow_node="financeharness_parent_temporal_filter",
                network=asdict(event),
                error=event.error,
            )
        context.emit(
            "task_finished",
            effective_input_hash=effective_hash,
            control_flow_node="financeharness_external_worker",
            latency_seconds=elapsed,
            error=None if raw_text else "empty_model_response",
        )
        effective_request = {
            "run_id": context.run_id,
            "task_id": task.task_id,
            "framework": self.name,
            "adapter_version": self.adapter_version,
            "messages": list(effective_messages),
            "messages_sha256": effective_hash,
            "prompt_sha256": sha256_text(task.prompt),
            "model": context.model.public_dict(),
            "model_provenance": context.model.provenance_dict(
                provider_response_model_names=broker.provider_model_names,
                system_fingerprints=broker.system_fingerprints,
            ),
            "deep_research": {
                "protocol": self.harness_config.protocol,
                "retrieval_regime": retrieval_regime,
                "grounding_review": self.deep_config.grounding_review,
                "termination": payload.get("termination"),
                "rounds": payload.get("rounds"),
                "grounding_reviewed": payload.get("grounding_reviewed", False),
                "filter_model_call_count": len(filter_events),
                "service_receipt_count": child_search_service_call_count,
                "service_receipt_sha256s": [receipt.receipt_sha256 for receipt in handoff_receipts],
                "child_search_service_call_count": child_search_service_call_count,
                "child_forbidden_network_call_count": child_forbidden_network_call_count,
                "provider_response_model_names": sorted(set(broker.provider_model_names)),
                # Compatibility alias for existing policy/report consumers.
                "provider_model_names": sorted(set(broker.provider_model_names)),
                "system_fingerprints": sorted(set(broker.system_fingerprints)),
                "budget": {
                    "max_rounds": self.deep_config.max_rounds,
                    "max_model_calls": self.deep_config.max_model_calls,
                    "max_search_queries": self.deep_config.max_search_queries,
                    "max_visit_urls": self.deep_config.max_visit_urls,
                    "max_tool_calls": self.deep_config.max_tool_calls,
                },
            },
        }
        result = AdapterResult(
            raw_text=raw_text,
            usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else {},
            latency_seconds=elapsed,
            canonical_input_hash=task.input_hash,
            effective_input_hash=effective_hash,
            effective_messages=effective_messages,
            effective_request=effective_request,
            # Temporal-filter calls are parent-owned evidence review.  They
            # remain in ``network_events`` and trace events, but the locked
            # deep cognition budget counts only agent/reader/grounding broker
            # calls.
            model_call_count=broker.call_count,
            tool_calls=tool_calls,
            network_events=tuple(
                (
                    *broker.network_events,
                    *filter_events,
                )
            ),
            trace_events=context.events,
            trace_artifact_ref="trace_events.jsonl",
            error=payload.get("error") if isinstance(payload.get("error"), str) else None,
        )
        if isinstance(service_close, dict):
            ledgers = tuple(
                row for row in service_close.get("network_ledger", []) if isinstance(row, dict)
            )
            if service_session is not None:
                ledgers = ledgers + tuple(
                    {
                        "run_id": context.run_id,
                        "task_id": task.task_id,
                        "framework": self.name,
                        "network_class": "search_service_local",
                        "provider": "local",
                        "host": "127.0.0.1",
                        "method": "POST",
                        "path": f"/v1/sessions/{service_session.descriptor.session_id}/search",
                        "body_hash": receipt.receipt_sha256,
                        "status_code": 200,
                        "latency_seconds": None,
                        "is_model_endpoint": False,
                        "session_id": service_session.descriptor.session_id,
                        "service_request_id": receipt.service_request_id,
                        "call_index": receipt.call_index,
                        "receipt_sha256": receipt.receipt_sha256,
                    }
                    for receipt in handoff_receipts
                )
        else:
            ledgers = tuple(
                outcome.network_ledger(profile="deep_research") for outcome in search_outcomes
            )
        citations = tuple(value for value in payload.get("citations", []) if isinstance(value, str))
        return DeepRunOutput(
            result=result,
            search_outcomes=tuple(getattr(search, "outcomes", ())),
            search_ledger_rows=ledgers,
            research_audit_rows=tuple(audit_rows),
            citations=citations,
            external_manifest=manifest,
            child_network_call_count=child_network_call_count,
            filter_model_call_count=len(filter_events),
            child_search_service_call_count=child_search_service_call_count,
            child_forbidden_network_call_count=child_forbidden_network_call_count,
            service_close=service_close,
        )

    @staticmethod
    def _close_process(process: Any) -> None:
        try:
            if process.poll() is None:
                terminate_process_group(process)
            else:
                process.wait(timeout=2.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def _tool_calls_from_payload(payload: Mapping[str, Any]) -> tuple[ToolCall, ...]:
    raw_calls = payload.get("tool_log", payload.get("tool_calls", []))
    if not isinstance(raw_calls, list):
        return ()
    calls: list[ToolCall] = []
    for raw in raw_calls:
        if isinstance(raw, str):
            calls.append(ToolCall(name=raw))
            continue
        if not isinstance(raw, dict):
            continue
        name = raw.get("name") or raw.get("tool") or raw.get("tool_name")
        if not isinstance(name, str):
            continue
        arguments = raw.get("arguments", raw.get("input", ""))
        result = raw.get("result", raw.get("output", ""))
        calls.append(
            ToolCall(
                name=name,
                arguments_hash=sha256_text(str(arguments)) if arguments else None,
                result_hash=sha256_text(str(result)) if result else None,
            )
        )
    return tuple(calls)


def _service_filter_events(close: dict[str, Any] | None) -> tuple[NetworkEvent, ...]:
    if not isinstance(close, dict):
        return ()
    events: list[NetworkEvent] = []
    for row in close.get("network_ledger", []):
        if not isinstance(row, dict) or row.get("network_class") != "temporal_filter":
            continue
        events.append(
            NetworkEvent(
                host=str(row.get("host", "")),
                method=str(row.get("method", "POST")),
                path=str(row.get("path", "/v1/chat/completions")),
                body_hash=str(row.get("body_hash", "")),
                status_code=(
                    int(row["status_code"]) if isinstance(row.get("status_code"), int) else None
                ),
                latency_seconds=(
                    float(row["latency_seconds"])
                    if isinstance(row.get("latency_seconds"), int | float)
                    else None
                ),
                is_model_endpoint=bool(row.get("is_model_endpoint", True)),
                network_class="temporal_filter",
            )
        )
    return tuple(events)


def _validated_service_call_count(
    value: Any, *, expected: int, service_session: SearchServiceSession | None
) -> int:
    """Use validated child receipts as the service-call count authority."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FinanceHarnessBoundaryError("child_search_service_call_count_invalid")
    if service_session is not None and value != expected:
        raise FinanceHarnessBoundaryError("child_search_service_receipt_count_mismatch")
    return expected if service_session is not None else value


_SERVICE_RECEIPT_FIELDS = frozenset(
    {
        "receipt_index",
        "service_request_id",
        "receipt_sha256",
        "previous_receipt_sha256",
        "call_index",
        "replayed",
        "api_contract_sha256",
        "implementation_manifest_sha256",
        "capability_sha256",
        "service_cohort_sha256",
    }
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _validated_service_receipts(
    value: Any, *, service_session: SearchServiceSession | None
) -> tuple[SearchReceiptRef, ...]:
    """Validate receipt references emitted by the child before parent close.

    The parent opens the server session, but the child performs the searches.
    Consequently the child receipt list, not the parent's empty local list, is
    the close-session witness.  All fields are checked without coercion so a
    malformed child cannot silently manufacture a close payload.
    """

    if service_session is None:
        if value is None or value == []:
            return ()
        raise FinanceHarnessBoundaryError("child_search_service_receipts_unexpected")
    if not isinstance(value, list):
        raise FinanceHarnessBoundaryError("child_search_service_receipts_invalid")

    descriptor = service_session.descriptor
    identities = {
        "api_contract_sha256": descriptor.api_contract_sha256,
        "implementation_manifest_sha256": descriptor.implementation_manifest_sha256,
        "capability_sha256": descriptor.capability_sha256,
        "service_cohort_sha256": descriptor.service_cohort_sha256,
    }
    validated: list[SearchReceiptRef] = []
    previous_hash: str | None = None
    for expected_index, raw in enumerate(value, start=1):
        if not isinstance(raw, dict) or set(raw) != _SERVICE_RECEIPT_FIELDS:
            raise FinanceHarnessBoundaryError("child_search_service_receipts_invalid")
        receipt_index = raw["receipt_index"]
        service_request_id = raw["service_request_id"]
        receipt_hash = raw["receipt_sha256"]
        previous_receipt_hash = raw["previous_receipt_sha256"]
        call_index = raw["call_index"]
        replayed = raw["replayed"]
        if (
            not isinstance(receipt_index, int)
            or isinstance(receipt_index, bool)
            or receipt_index != expected_index
            or not isinstance(call_index, int)
            or isinstance(call_index, bool)
            or call_index != expected_index
            or not isinstance(service_request_id, str)
            or not service_request_id
            or len(service_request_id) > 128
            or any(character.isspace() for character in service_request_id)
            or not _is_sha256(receipt_hash)
            or not isinstance(replayed, bool)
        ):
            raise FinanceHarnessBoundaryError("child_search_service_receipts_invalid")
        if previous_receipt_hash != previous_hash:
            raise FinanceHarnessBoundaryError("child_search_service_receipt_chain_invalid")
        if previous_receipt_hash is not None and not _is_sha256(previous_receipt_hash):
            raise FinanceHarnessBoundaryError("child_search_service_receipt_chain_invalid")
        if any(raw[name] != expected for name, expected in identities.items()):
            raise FinanceHarnessBoundaryError("child_search_service_receipt_identity_invalid")
        validated.append(
            SearchReceiptRef(
                receipt_index=receipt_index,
                service_request_id=service_request_id,
                receipt_sha256=receipt_hash,
                previous_receipt_sha256=previous_receipt_hash,
                call_index=call_index,
                replayed=replayed,
                api_contract_sha256=raw["api_contract_sha256"],
                implementation_manifest_sha256=raw["implementation_manifest_sha256"],
                capability_sha256=raw["capability_sha256"],
                service_cohort_sha256=raw["service_cohort_sha256"],
            )
        )
        previous_hash = receipt_hash
    return tuple(validated)


def _validated_nonnegative_count(value: Any, *, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FinanceHarnessBoundaryError(code)
    return value


__all__ = [
    "DeepModelBroker",
    "DeepRunOutput",
    "FinanceHarnessAdapter",
    "FinanceHarnessBoundaryError",
]
