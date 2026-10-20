"""Strict stdio/JSONL transport for external research workers."""

from __future__ import annotations

import io
import json
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from benchmark_harness.contracts import stable_json

IPC_PROTOCOL = "external_research_stdio/v2"
DEFAULT_MAX_FRAME_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_STDERR_BYTES = 1024 * 1024
ENVELOPE_FIELDS = frozenset({"schema_version", "type", "request_id", "task_id", "seq", "payload"})
IPC_TYPES = frozenset(
    {
        "hello",
        "task.start",
        "model.request",
        "model.response",
        "audit.event",
        "task.result",
        "task.error",
    }
)


class IpcProtocolError(RuntimeError):
    """Raised for malformed, out-of-order, or oversized IPC frames."""


@dataclass(frozen=True, slots=True)
class Envelope:
    """One versioned JSONL message; no arbitrary top-level extensions."""

    type: str
    request_id: str
    task_id: str
    seq: int
    payload: dict[str, Any]
    schema_version: str = IPC_PROTOCOL

    def __post_init__(self) -> None:
        if self.schema_version != IPC_PROTOCOL:
            raise IpcProtocolError("unsupported IPC schema_version")
        if self.type not in IPC_TYPES:
            raise IpcProtocolError(f"unsupported IPC message type: {self.type!r}")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise IpcProtocolError("request_id must be a non-empty string")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise IpcProtocolError("task_id must be a non-empty string")
        if not isinstance(self.seq, int) or isinstance(self.seq, bool) or self.seq < 0:
            raise IpcProtocolError("seq must be a non-negative integer")
        if not isinstance(self.payload, dict):
            raise IpcProtocolError("payload must be a JSON object")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "type": self.type,
            "request_id": self.request_id,
            "task_id": self.task_id,
            "seq": self.seq,
            "payload": self.payload,
        }


def encode_envelope(envelope: Envelope, *, max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES) -> bytes:
    """Encode one frame and enforce the byte limit before writing it."""

    try:
        body = stable_json(envelope.as_dict()).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise IpcProtocolError("IPC payload is not JSON serializable") from exc
    if len(body) > max_frame_bytes:
        raise IpcProtocolError("IPC frame exceeds max_frame_bytes")
    return body


def decode_envelope(
    raw: bytes | str, *, max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES
) -> Envelope:
    """Parse exactly one newline-delimited envelope with closed fields."""

    if isinstance(raw, str):
        raw_bytes = raw.encode("utf-8")
    elif isinstance(raw, bytes):
        raw_bytes = raw
    else:
        raise IpcProtocolError("IPC frame must be bytes or text")
    if len(raw_bytes) > max_frame_bytes:
        raise IpcProtocolError("IPC frame exceeds max_frame_bytes")
    if not raw_bytes.endswith(b"\n"):
        raise IpcProtocolError("IPC frame must end with a newline")
    try:
        value = json.loads(raw_bytes[:-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IpcProtocolError("IPC frame is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or set(value) != ENVELOPE_FIELDS:
        raise IpcProtocolError("IPC envelope fields are not exactly the protocol fields")
    return Envelope(
        schema_version=value["schema_version"],
        type=value["type"],
        request_id=value["request_id"],
        task_id=value["task_id"],
        seq=value["seq"],
        payload=value["payload"],
    )


class IpcChannel:
    """Bidirectional sequence checker around binary or text stdio streams."""

    def __init__(
        self,
        reader: BinaryIO | TextIO,
        writer: BinaryIO | TextIO,
        *,
        task_id: str,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.task_id = task_id
        self.max_frame_bytes = max_frame_bytes
        self._next_send_seq = 0
        self._last_received_seq: int | None = None

    def send(
        self,
        message_type: str,
        payload: dict[str, Any],
        *,
        request_id: str,
        task_id: str | None = None,
    ) -> Envelope:
        envelope = Envelope(
            type=message_type,
            request_id=request_id,
            task_id=task_id or self.task_id,
            seq=self._next_send_seq,
            payload=payload,
        )
        frame = encode_envelope(envelope, max_frame_bytes=self.max_frame_bytes)
        try:
            if isinstance(
                self.writer, io.BufferedIOBase | io.RawIOBase | io.BytesIO
            ) or "b" in getattr(self.writer, "mode", ""):
                self.writer.write(frame)  # type: ignore[arg-type]
            else:
                self.writer.write(frame.decode("utf-8"))  # type: ignore[arg-type]
            self.writer.flush()
        except (OSError, TypeError, ValueError) as exc:
            raise IpcProtocolError("failed to write IPC frame") from exc
        self._next_send_seq += 1
        return envelope

    def read(self) -> Envelope:
        try:
            raw = self.reader.readline()
        except (OSError, ValueError) as exc:
            raise IpcProtocolError("failed to read IPC frame") from exc
        if not raw:
            raise IpcProtocolError("IPC peer closed stdout before task completion")
        if isinstance(raw, str):
            frame_size = len(raw.encode("utf-8"))
        else:
            frame_size = len(raw)
        if frame_size > self.max_frame_bytes:
            raise IpcProtocolError("IPC frame exceeds max_frame_bytes")
        envelope = decode_envelope(raw, max_frame_bytes=self.max_frame_bytes)
        if envelope.task_id != self.task_id:
            raise IpcProtocolError("IPC task_id does not match the active task")
        if self._last_received_seq is not None and envelope.seq <= self._last_received_seq:
            raise IpcProtocolError("IPC seq is not strictly monotonic")
        self._last_received_seq = envelope.seq
        return envelope


_SECRET_ENV_MARKERS = (
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "CREDENTIAL",
    "AUTHORIZATION",
)


def is_credential_environment_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_ENV_MARKERS) or upper.startswith(
        ("OPENAI_", "DASHSCOPE_", "ALIYUN_", "VOLC_", "BENCHMARK_MODEL_")
    )


def build_child_environment(
    source_dir: str | Path,
    repo_root: str | Path,
    *,
    python_executable: str | Path | None = None,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Construct a minimal child env with all model/search credentials removed."""

    source = Path(source_dir).resolve()
    root = Path(repo_root).resolve()
    base = dict(os.environ if base_environment is None else base_environment)
    safe = {
        name: value
        for name, value in base.items()
        if name in {"PATH", "PYTHONPATH", "PYTHONIOENCODING", "LANG", "LC_ALL"}
        and not is_credential_environment_name(name)
    }
    executable = Path(python_executable).resolve() if python_executable else None
    venv_bin = str(executable.parent) if executable else None
    safe["PATH"] = os.pathsep.join(
        part for part in (venv_bin, safe.get("PATH", os.defpath)) if part
    )
    safe["PYTHONPATH"] = os.pathsep.join((str(root), str(source)))
    safe["PYTHONUNBUFFERED"] = "1"
    safe["PYTHONNOUSERSITE"] = "1"
    return safe


def terminate_process_group(process: Any, *, grace_seconds: float = 2.0) -> None:
    """Terminate and reap a worker without leaving a child process behind."""

    if process.poll() is not None:
        return
    try:
        if getattr(process, "_benchmark_start_new_session", False):
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except (OSError, ProcessLookupError):
        pass
    try:
        process.wait(timeout=grace_seconds)
    except Exception:
        try:
            if getattr(process, "_benchmark_start_new_session", False):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=grace_seconds)
        except Exception:
            return


__all__ = [
    "DEFAULT_MAX_FRAME_BYTES",
    "DEFAULT_MAX_STDERR_BYTES",
    "ENVELOPE_FIELDS",
    "IPC_PROTOCOL",
    "IPC_TYPES",
    "Envelope",
    "IpcChannel",
    "IpcProtocolError",
    "build_child_environment",
    "decode_envelope",
    "encode_envelope",
    "is_credential_environment_name",
    "terminate_process_group",
]
