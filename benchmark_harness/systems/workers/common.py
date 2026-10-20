"""Small v3 child-channel helpers used by both disposable workers."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_harness.systems.clock import DeclaredExperimentClock
from benchmark_harness.systems.external_stdio import (
    V3Envelope,
    V3ProtocolError,
    decode_v3,
    encode_v3,
)
from benchmark_harness.systems.worker_diagnostics import worker_failure_subreason
from benchmark_harness.systems.workers.output_contract import (
    reject_fixture_sentinel,
    task_bound_answer_messages,
    task_category,
)
from benchmark_harness.temporal.policy import assert_no_temporal_leakage

SEARCH_QUERY_MAX_CHARS = 2048


class WorkerProtocolError(RuntimeError):
    """Raised when the parent sends an invalid v3 frame."""


class WorkerChannel:
    """Strict JSONL child side of the host-owned v3 protocol."""

    def __init__(self) -> None:
        self._task_id: str | None = None
        self._binding_hash: str | None = None
        self._send_seq = 0
        self._last_received_seq: int | None = None

    @property
    def bound(self) -> bool:
        return self._task_id is not None and self._binding_hash is not None

    @property
    def task_id(self) -> str:
        if self._task_id is None:
            raise WorkerProtocolError("worker_channel_not_bound")
        return self._task_id

    def read(self) -> V3Envelope:
        raw = sys.stdin.buffer.readline()
        if not raw:
            raise WorkerProtocolError("parent_closed_before_completion")
        try:
            envelope = decode_v3(raw)
        except V3ProtocolError as exc:
            raise WorkerProtocolError(str(exc)) from exc
        if self._task_id is None:
            self._task_id = envelope.task_id
            self._binding_hash = envelope.binding_hash
        elif envelope.task_id != self._task_id:
            raise WorkerProtocolError("parent_task_id_mismatch")
        if envelope.binding_hash != self._binding_hash:
            raise WorkerProtocolError("parent_binding_hash_mismatch")
        if self._last_received_seq is not None and envelope.seq <= self._last_received_seq:
            raise WorkerProtocolError("parent_sequence_not_monotonic")
        self._last_received_seq = envelope.seq
        return envelope

    def send(
        self,
        message_type: str,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
    ) -> V3Envelope:
        if not self.bound:
            raise WorkerProtocolError("worker_channel_not_bound")
        assert self._task_id is not None
        assert self._binding_hash is not None
        assert_no_temporal_leakage(payload)
        envelope = V3Envelope(
            type=message_type,
            request_id=request_id or f"worker-{self._send_seq}",
            task_id=self._task_id,
            seq=self._send_seq,
            binding_hash=self._binding_hash,
            payload=payload,
        )
        try:
            sys.stdout.buffer.write(encode_v3(envelope))
            sys.stdout.buffer.flush()
        except (OSError, TypeError, ValueError) as exc:
            raise WorkerProtocolError("worker_write_failed") from exc
        self._send_seq += 1
        return envelope

    def request_model(
        self,
        *,
        role: str,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": role, "messages": messages}
        if response_format is not None:
            payload["response_format"] = response_format
        if tools is not None:
            payload["tools"] = tools
        request = self.send(
            "model.request",
            payload,
        )
        response = self.read()
        if response.type != "model.response" or response.request_id != request.request_id:
            raise WorkerProtocolError("model_response_mismatch")
        return response.payload

    def request_search(self, *, query: str, max_results: int) -> dict[str, Any]:
        request = self.send(
            "search.request",
            {"query": query, "max_results": max_results},
        )
        response = self.read()
        if response.type != "search.response" or response.request_id != request.request_id:
            raise WorkerProtocolError("search_response_mismatch")
        return response.payload


def task_start(channel: WorkerChannel, *, adapter_id: str) -> tuple[V3Envelope, dict[str, Any]]:
    start = channel.read()
    if start.type != "task.start":
        raise WorkerProtocolError("task_start_required")
    if start.payload.get("protocol") != "external_research_stdio/v3":
        raise WorkerProtocolError("task_start_protocol_mismatch")
    if start.payload.get("adapter_id") != adapter_id:
        raise WorkerProtocolError("task_start_adapter_mismatch")
    experiment_as_of = start.payload.get("experiment_as_of")
    clock = start.payload.get("clock")
    if not isinstance(experiment_as_of, str) or not isinstance(clock, Mapping):
        raise WorkerProtocolError("declared_clock_missing")
    declared_clock = DeclaredExperimentClock(experiment_as_of)
    if clock.get("experiment_as_of") != declared_clock.isoformat():
        raise WorkerProtocolError("declared_clock_mismatch")
    return start, dict(start.payload)


def emit_trajectory(channel: WorkerChannel, *, node: str, detail: str) -> None:
    channel.send("trajectory.event", {"node": node, "detail": detail})


def response_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise WorkerProtocolError("model_response_choices_missing")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise WorkerProtocolError("model_response_choice_malformed")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise WorkerProtocolError("model_response_content_missing")
    content = message.get("content")
    if isinstance(content, str):
        return content
    # OpenAI-compatible tool-call responses may omit textual content (or set it
    # to null) while carrying the actual assistant turn in ``tool_calls``.
    # Normalize that valid intermediate turn to the empty string; callers that
    # require text still validate non-emptiness at their own boundary.
    if content is None and isinstance(message.get("tool_calls"), list) and message["tool_calls"]:
        return ""
    raise WorkerProtocolError("model_response_content_missing")


def citations_from_search(payload: Mapping[str, Any]) -> list[str]:
    results = payload.get("results")
    if not isinstance(results, list):
        raise WorkerProtocolError("search_response_results_missing")
    citations: list[str] = []
    for result in results:
        if isinstance(result, Mapping) and isinstance(result.get("url"), str):
            citations.append(result["url"])
    return list(dict.fromkeys(citations))


@dataclass(frozen=True, slots=True)
class DeclaredPromptBuilder:
    """Compatibility shim that makes the host clock the only prompt date."""

    clock: DeclaredExperimentClock

    def build(self, *, role: str, prompt: str, context: str = "") -> list[dict[str, str]]:
        date_text = self.clock.prompt_date()
        system = (
            f"You are the {role} role in a pinned integration worker. "
            f"The experiment date is {date_text}; do not use machine time."
        )
        user = prompt if not context else f"{prompt}\n\nEvidence:\n{context}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


def send_task_error(channel: WorkerChannel, error: BaseException) -> None:
    # Keep the historical class-only error field stable.  A classified
    # RuntimeError may additionally carry a host-validated diagnostic, but the
    # worker never sends exception text, tracebacks, or model content.
    error_name = "RuntimeError" if isinstance(error, RuntimeError) else type(error).__name__
    diagnostic = getattr(error, "diagnostic", None)
    payload: dict[str, Any] = {"error": f"worker_error:{error_name}"}
    sub_reason = worker_failure_subreason(error)
    if sub_reason is not None:
        payload["sub_reason"] = sub_reason
    if isinstance(diagnostic, Mapping):
        payload["diagnostic"] = dict(diagnostic)
    if channel.bound:
        try:
            channel.send("task.error", payload)
        except Exception:
            pass
    else:
        print(payload["error"], file=sys.stderr)


def source_import_path(source_dir: str | Path) -> Path:
    source = Path(source_dir).expanduser().resolve()
    src = source / "src"
    return src if src.is_dir() else source


def json_context(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)[:12000]


def validate_search_query(query: Any) -> str:
    """Validate, but never transform, a host Search+Filter query.

    Search+Filter owns the same ``1..2048`` contract at its service boundary.
    Worker code must not truncate a task or model output to make it fit.  The
    returned value is the original string so callers cannot accidentally hide
    a query transformation behind this helper.
    """

    if not isinstance(query, str):
        raise ValueError("search_query_invalid")
    stripped = query.strip()
    if not stripped or len(stripped) > SEARCH_QUERY_MAX_CHARS:
        raise ValueError("search_query_invalid")
    return query


def request_task_bound_prediction(
    channel: WorkerChannel,
    *,
    task_id: str,
    task_prompt: str,
    research_context: str,
    citations: list[str],
    role: str,
    fixture: bool,
    error_prefix: str,
) -> tuple[str, list[dict[str, str]]]:
    """Request and validate the answer for the original task only."""

    category = task_category(task_id) if not fixture else None
    messages = task_bound_answer_messages(
        task_id,
        task_prompt,
        research_context,
        citations,
    )
    if not fixture:
        reject_fixture_sentinel(messages, field=f"{error_prefix}_model_request")
    response = channel.request_model(
        role=role,
        messages=messages,
        response_format=None if category == "A2" else {"type": "json_object"},
    )
    prediction = response_content(response)
    if not prediction.strip():
        raise RuntimeError(f"{error_prefix}_task_bound_prediction_missing")
    if not fixture:
        reject_fixture_sentinel(prediction, field=f"{error_prefix}_prediction")
    return prediction, messages


__all__ = [
    "DeclaredPromptBuilder",
    "WorkerChannel",
    "WorkerProtocolError",
    "SEARCH_QUERY_MAX_CHARS",
    "citations_from_search",
    "emit_trajectory",
    "json_context",
    "request_task_bound_prediction",
    "response_content",
    "send_task_error",
    "source_import_path",
    "task_start",
    "validate_search_query",
]
