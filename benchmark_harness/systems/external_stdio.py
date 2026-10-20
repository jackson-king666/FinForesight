"""Additive external_research_stdio/v3 transport and budget enforcement."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, BinaryIO, TextIO

from benchmark_harness.contracts import stable_json
from benchmark_harness.external_harnesses.ipc import (
    IPC_PROTOCOL as V2_PROTOCOL,
)
from benchmark_harness.external_harnesses.ipc import (
    decode_envelope as decode_v2,
)
from benchmark_harness.systems.contracts import BudgetSpec, ExperimentBinding, TaskCapability
from benchmark_harness.temporal.policy import assert_no_temporal_leakage

V3_PROTOCOL = "external_research_stdio/v3"
V3_ENVELOPE_FIELDS = frozenset(
    {"schema_version", "type", "request_id", "task_id", "seq", "binding_hash", "payload"}
)
V3_TYPES = frozenset(
    {
        "hello",
        "capability.manifest",
        "task.start",
        "model.request",
        "model.response",
        "search.request",
        "search.response",
        "trajectory.event",
        "task.result",
        "task.error",
    }
)
DEFAULT_MAX_FRAME_BYTES = 8 * 1024 * 1024


class V3ProtocolError(RuntimeError):
    """Raised for malformed, oversized, out-of-order, or leaking v3 frames."""


@dataclass(frozen=True, slots=True)
class FinanceHarnessV2CompatibilityBridge:
    """Explicit marker/decoder for the unchanged FinanceHarness v2 path."""

    legacy_protocol: str = V2_PROTOCOL
    protocol: str = V3_PROTOCOL

    def decode_legacy(self, raw: bytes | str) -> Any:
        return decode_v2(raw)


@dataclass(frozen=True, slots=True)
class V3Envelope:
    type: str
    request_id: str
    task_id: str
    seq: int
    binding_hash: str
    payload: dict[str, Any]
    schema_version: str = V3_PROTOCOL

    def __post_init__(self) -> None:
        if self.schema_version != V3_PROTOCOL:
            raise V3ProtocolError("unsupported v3 schema_version")
        if self.type not in V3_TYPES:
            raise V3ProtocolError(f"unsupported v3 message type: {self.type!r}")
        for name in ("request_id", "task_id", "binding_hash"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise V3ProtocolError(f"{name} must be a non-empty string")
        if not isinstance(self.seq, int) or isinstance(self.seq, bool) or self.seq < 0:
            raise V3ProtocolError("seq must be a non-negative integer")
        if not isinstance(self.payload, dict):
            raise V3ProtocolError("payload must be an object")
        try:
            assert_no_temporal_leakage(self.payload)
        except ValueError as exc:
            raise V3ProtocolError(str(exc)) from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "type": self.type,
            "request_id": self.request_id,
            "task_id": self.task_id,
            "seq": self.seq,
            "binding_hash": self.binding_hash,
            "payload": self.payload,
        }


def encode_v3(envelope: V3Envelope, *, max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> bytes:
    try:
        frame = stable_json(envelope.as_dict()).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise V3ProtocolError("v3 payload is not JSON serializable") from exc
    if len(frame) > max_frame_bytes:
        raise V3ProtocolError("v3 frame exceeds max_frame_bytes")
    return frame


def decode_v3(raw: bytes | str, *, max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> V3Envelope:
    raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else raw
    if not isinstance(raw_bytes, bytes) or len(raw_bytes) > max_frame_bytes:
        raise V3ProtocolError("v3 frame exceeds max_frame_bytes")
    if not raw_bytes.endswith(b"\n"):
        raise V3ProtocolError("v3 frame must end with a newline")
    try:
        value = json.loads(raw_bytes[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V3ProtocolError("v3 frame is not UTF-8 JSON") from exc
    if not isinstance(value, dict) or set(value) != V3_ENVELOPE_FIELDS:
        raise V3ProtocolError("v3 envelope fields are not exactly the protocol fields")
    return V3Envelope(
        schema_version=value["schema_version"],
        type=value["type"],
        request_id=value["request_id"],
        task_id=value["task_id"],
        seq=value["seq"],
        binding_hash=value["binding_hash"],
        payload=value["payload"],
    )


class V3Channel:
    """Sequence-checked JSONL channel bound to one immutable experiment cell."""

    def __init__(
        self,
        reader: BinaryIO | TextIO,
        writer: BinaryIO | TextIO,
        *,
        binding: ExperimentBinding,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.binding = binding
        self.max_frame_bytes = max_frame_bytes
        self._next_send_seq = 0
        self._last_received_seq: int | None = None

    def send(self, message_type: str, payload: dict[str, Any], *, request_id: str) -> V3Envelope:
        envelope = V3Envelope(
            type=message_type,
            request_id=request_id,
            task_id=self.binding.task_id,
            seq=self._next_send_seq,
            binding_hash=self.binding.plan_hash,
            payload=payload,
        )
        frame = encode_v3(envelope, max_frame_bytes=self.max_frame_bytes)
        try:
            if isinstance(
                self.writer, io.BufferedIOBase | io.RawIOBase | io.BytesIO
            ) or "b" in getattr(self.writer, "mode", ""):
                self.writer.write(frame)  # type: ignore[arg-type]
            else:
                self.writer.write(frame.decode("utf-8"))  # type: ignore[arg-type]
            self.writer.flush()
        except (OSError, TypeError, ValueError) as exc:
            raise V3ProtocolError("failed to write v3 frame") from exc
        self._next_send_seq += 1
        return envelope

    def read(self) -> V3Envelope:
        try:
            raw = self.reader.readline()
        except (OSError, ValueError) as exc:
            raise V3ProtocolError("failed to read v3 frame") from exc
        if not raw:
            raise V3ProtocolError("v3 peer closed stdout before completion")
        envelope = decode_v3(raw, max_frame_bytes=self.max_frame_bytes)
        if envelope.task_id != self.binding.task_id:
            raise V3ProtocolError("v3 task_id does not match binding")
        if envelope.binding_hash != self.binding.plan_hash:
            raise V3ProtocolError("v3 binding_hash does not match plan hash")
        if self._last_received_seq is not None and envelope.seq <= self._last_received_seq:
            raise V3ProtocolError("v3 receive sequence is not strictly monotonic")
        self._last_received_seq = envelope.seq
        return envelope


@dataclass(slots=True)
class BudgetLedger:
    """Host-side counters shared by all concurrent child units."""

    budget: BudgetSpec
    model_calls: int = 0
    search_calls: int = 0
    visit_calls: int = 0
    other_tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def consume(
        self,
        *,
        model_calls: int = 0,
        search_calls: int = 0,
        visit_calls: int = 0,
        other_tool_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        values = {
            "model_calls": model_calls,
            "search_calls": search_calls,
            "visit_calls": visit_calls,
            "other_tool_calls": other_tool_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values.values()
        ):
            raise V3ProtocolError("budget increments must be non-negative integers")
        projected = {name: getattr(self, name) + value for name, value in values.items()}
        limits = {
            "model_calls": self.budget.max_model_calls,
            "search_calls": self.budget.max_search_calls,
            "visit_calls": self.budget.max_visit_calls,
            "other_tool_calls": self.budget.max_other_tool_calls,
            "input_tokens": self.budget.max_input_tokens,
            "output_tokens": self.budget.max_output_tokens,
        }
        exceeded = [name for name, value in projected.items() if value > limits[name]]
        if exceeded:
            raise V3ProtocolError("budget exceeded: " + ", ".join(sorted(exceeded)))
        for name, value in projected.items():
            setattr(self, name, value)

    def remaining(self) -> dict[str, int]:
        return {
            "model_calls": self.budget.max_model_calls - self.model_calls,
            "search_calls": self.budget.max_search_calls - self.search_calls,
            "visit_calls": self.budget.max_visit_calls - self.visit_calls,
            "other_tool_calls": self.budget.max_other_tool_calls - self.other_tool_calls,
            "input_tokens": self.budget.max_input_tokens - self.input_tokens,
            "output_tokens": self.budget.max_output_tokens - self.output_tokens,
        }


@dataclass(frozen=True, slots=True)
class V3CapabilityManifest:
    """Serializable capability metadata with no capability token persistence."""

    capability: TaskCapability
    injected_tools: tuple[str, ...] = ("model_broker",)
    mcp_enabled: bool = False
    protocol: str = V3_PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "capability": self.capability.public_dict(),
            "budget_sha256": self.capability.budget.hash,
            "injected_tools": list(self.injected_tools),
            "mcp_enabled": self.mcp_enabled,
        }


@dataclass(frozen=True, slots=True)
class V3SearchRequest:
    """Typed host-owned search request; no URL or credential surface is exposed."""

    query: str
    max_results: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise V3ProtocolError("search query must be a non-empty string")
        if (
            not isinstance(self.max_results, int)
            or isinstance(self.max_results, bool)
            or not 1 <= self.max_results <= 100
        ):
            raise V3ProtocolError("search max_results must be an integer in [1, 100]")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> V3SearchRequest:
        allowed = {"query", "max_results"}
        if set(payload) - allowed:
            raise V3ProtocolError("search.request contains unsupported fields")
        return cls(query=payload.get("query", ""), max_results=payload.get("max_results", 5))

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "max_results": self.max_results}


@dataclass(frozen=True, slots=True)
class V3SearchResponse:
    """Typed search result with a host-issued receipt identity."""

    results: tuple[dict[str, Any], ...]
    receipt_sha256: str
    call_index: int
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.results, tuple) or any(
            not isinstance(result, dict) for result in self.results
        ):
            raise V3ProtocolError("search results must be a tuple of objects")
        if not isinstance(self.receipt_sha256, str) or not self.receipt_sha256:
            raise V3ProtocolError("search receipt_sha256 must be non-empty")
        if not isinstance(self.call_index, int) or isinstance(self.call_index, bool):
            raise V3ProtocolError("search call_index must be an integer")
        if self.error is not None and (not isinstance(self.error, str) or not self.error.strip()):
            raise V3ProtocolError("search error must be a non-empty string when present")

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": list(self.results),
            "receipt_sha256": self.receipt_sha256,
            "call_index": self.call_index,
            "error": self.error,
        }
