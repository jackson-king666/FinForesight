"""Strict adapter contracts and trace records.

The adapter boundary deliberately exposes only TaskInput(task_id, prompt,
cutoff_date).
Dataset records, ground truth, parsers, and evaluators remain in the runner
process on the other side of this boundary.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from benchmark_harness.config import ResolvedModelProfile

if TYPE_CHECKING:
    from benchmark_harness.search import AgenticSearchSession

TRACE_SCHEMA_VERSION = "1.0"
TEMPORAL_FILTER_NETWORK_CLASS = "temporal_filter"


def stable_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically for audit hashes."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_messages(prompt: str) -> list[dict[str, str]]:
    """The only legal model input for one_turn_strict."""

    return [{"role": "user", "content": prompt}]


def canonical_input_hash(prompt: str) -> str:
    return sha256_json(canonical_messages(prompt))


@dataclass(frozen=True, slots=True)
class TaskInput:
    """Adapter-safe task payload: no record, ground truth, or evaluator fields."""

    task_id: str
    prompt: str
    cutoff_date: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("TaskInput.task_id must be a non-empty string.")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("TaskInput.prompt must be a non-empty string.")
        if self.cutoff_date is not None and not isinstance(self.cutoff_date, str):
            raise ValueError("TaskInput.cutoff_date must be a YYYY-MM-DD string or None.")

    @property
    def messages(self) -> list[dict[str, str]]:
        return canonical_messages(self.prompt)

    @property
    def input_hash(self) -> str:
        return canonical_input_hash(self.prompt)


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments_hash: str | None = None
    result_hash: str | None = None


@dataclass(frozen=True, slots=True)
class NetworkEvent:
    """A redacted, endpoint-level ledger entry emitted by an adapter."""

    host: str
    method: str
    path: str
    body_hash: str
    status_code: int | None
    latency_seconds: float | None
    is_model_endpoint: bool
    error: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    # ``None`` identifies ordinary adapter/model traffic.  The deep profile
    # marks supervisor-owned temporal-filter calls explicitly so they remain
    # auditable without being counted as agent cognition.
    network_class: str | None = None


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One locally persisted event in the framework-neutral trace schema."""

    run_id: str
    task_id: str
    framework: str
    adapter_version: str
    event_seq: int
    event_type: str
    timestamp: str
    span_id: str | None = None
    parent_span_id: str | None = None
    model_call_index: int | None = None
    effective_input_hash: str | None = None
    control_flow_node: str | None = None
    tool_name: str | None = None
    tool_arguments_hash: str | None = None
    tool_result_hash: str | None = None
    network: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    latency_seconds: float | None = None
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunContext:
    """Per-task state. It is intentionally fresh for every task invocation."""

    run_id: str
    framework: str
    adapter_version: str
    model: ResolvedModelProfile
    trace_content_mode: str
    _event_seq: int = 0
    _events: list[TraceEvent] = field(default_factory=list)
    _task_id: str | None = field(default=None, init=False, repr=False)

    def emit(
        self,
        event_type: str,
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        model_call_index: int | None = None,
        effective_input_hash: str | None = None,
        control_flow_node: str | None = None,
        tool_name: str | None = None,
        tool_arguments_hash: str | None = None,
        tool_result_hash: str | None = None,
        network: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
        latency_seconds: float | None = None,
        error: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> TraceEvent:
        self._event_seq += 1
        event = TraceEvent(
            run_id=self.run_id,
            task_id=self.task_id,
            framework=self.framework,
            adapter_version=self.adapter_version,
            event_seq=self._event_seq,
            event_type=event_type,
            timestamp=datetime.now(UTC).isoformat(),
            span_id=span_id,
            parent_span_id=parent_span_id,
            model_call_index=model_call_index,
            effective_input_hash=effective_input_hash,
            control_flow_node=control_flow_node,
            tool_name=tool_name,
            tool_arguments_hash=tool_arguments_hash,
            tool_result_hash=tool_result_hash,
            network=network,
            usage=usage,
            latency_seconds=latency_seconds,
            error=error,
            attributes=attributes or {},
        )
        self._events.append(event)
        return event

    @property
    def task_id(self) -> str:
        """Set by Runner before passing this instance to the adapter."""

        task_id = getattr(self, "_task_id", None)
        if not isinstance(task_id, str):
            raise RuntimeError("RunContext task_id was not initialized.")
        return task_id

    def for_task(self, task_id: str) -> RunContext:
        """Make a fresh context; SDK/global state cannot leak across tasks."""

        cloned = RunContext(
            run_id=self.run_id,
            framework=self.framework,
            adapter_version=self.adapter_version,
            model=self.model,
            trace_content_mode=self.trace_content_mode,
        )
        cloned._task_id = task_id
        return cloned

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)


@dataclass(frozen=True, slots=True)
class AdapterResult:
    """Normalized result returned before upstream parsing/evaluation begins."""

    raw_text: str | None
    usage: dict[str, Any]
    latency_seconds: float
    canonical_input_hash: str
    effective_input_hash: str
    effective_messages: tuple[dict[str, str], ...]
    effective_request: dict[str, Any]
    model_call_count: int
    tool_calls: tuple[ToolCall, ...]
    network_events: tuple[NetworkEvent, ...]
    trace_events: tuple[TraceEvent, ...]
    trace_artifact_ref: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class TransportResponse:
    """Testable normalized provider response used by mock transport adapters."""

    raw_text: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    status_code: int | None = 200
    tool_calls: tuple[ToolCall, ...] = ()
    error: str | None = None
    extra_network_events: tuple[NetworkEvent, ...] = ()
    network_events: tuple[NetworkEvent, ...] | None = None
    observed_messages: tuple[dict[str, str], ...] | None = None
    framework_events: tuple[dict[str, Any], ...] = ()
    model_call_count: int = 1
    request_parameters: dict[str, Any] = field(default_factory=dict)


class StrictTransport(Protocol):
    """A test double can satisfy this without gaining access to dataset records."""

    def complete(
        self,
        task: TaskInput,
        messages: Sequence[dict[str, str]],
        context: RunContext,
    ) -> TransportResponse: ...


class Adapter(Protocol):
    """All real-framework adapters implement this narrow, record-free boundary."""

    name: str
    adapter_version: str

    def run(self, task: TaskInput, context: RunContext) -> AdapterResult: ...


@runtime_checkable
class AgenticAdapter(Adapter, Protocol):
    """Typed native tool-loop boundary for the agentic profile."""

    def run_agentic(
        self,
        task: TaskInput,
        context: RunContext,
        search_session: AgenticSearchSession,
    ) -> AdapterResult: ...
