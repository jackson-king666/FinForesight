"""Local JSONL trace sink with recursive secret redaction."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from benchmark_harness.contracts import NetworkEvent, TraceEvent, sha256_text

_SENSITIVE_KEYS = {
    "authorization",
    "proxy_authorization",
    "cookie",
    "set_cookie",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "token",
    "secret",
    "password",
}
_REDACTED = "[REDACTED]"
_INLINE_SECRET_PATTERNS = (
    re.compile(
        r"(?i)(\b(?:authorization|proxy-authorization)\s*[:=]\s*(?:bearer\s+|basic\s+)?)"
        r"[^\s,;]+"
    ),
    re.compile(r"(?i)(\b(?:cookie|set-cookie)\s*[:=]\s*)[^\r\n]+"),
    re.compile(
        r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password)"
        r"\s*[:=]\s*)[^\s,;&]+"
    ),
    re.compile(
        r"(?i)([?&](?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password)=)"
        r"[^&#\s]+"
    ),
)
_HASH_ONLY_CONTENT_KEYS = {
    "content",
    "prompt",
    "response",
    "raw_text",
    "raw_response",
    "message",
    "output",
    "input",
    "result",
    "argument",
    "arguments",
    "reasoning",
    "thought",
    "error",
    "query",
    "url",
    "title",
    "excerpt",
    "candidate_native_attributes",
    "published_at",
}


def _redact_text(value: str) -> str:
    """Remove credential-shaped text from provider errors and arbitrary strings."""

    redacted = value
    for pattern in _INLINE_SECRET_PATTERNS:
        redacted = pattern.sub(r"\1" + _REDACTED, redacted)
    return redacted


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return (
        normalized in _SENSITIVE_KEYS
        or normalized.endswith("_token")
        or any(
            fragment in normalized
            for fragment in ("authorization", "cookie", "api_key", "apikey", "secret", "password")
        )
    )


def _is_hash_only_content_key(key: object) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return normalized in _HASH_ONLY_CONTENT_KEYS or normalized.endswith(
        (
            "_content",
            "_prompt",
            "_response",
            "_message",
            "_messages",
            "_output",
            "_input",
            "_result",
            "_arguments",
            "_reasoning",
            "_thought",
            "_error",
            "_excerpt",
            "_query",
            "_url",
            "_title",
        )
    )


def redact(value: Any) -> Any:
    """Redact sensitive mapping values before they reach an artifact."""

    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {
            str(key): _REDACTED if _is_sensitive_key(key) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _hash_only(value: Any) -> Any:
    if isinstance(value, dict):
        reduced: dict[str, Any] = {}
        for key, item in value.items():
            if _is_hash_only_content_key(key) and isinstance(item, str):
                reduced[f"{key}_sha256"] = sha256_text(item)
            else:
                reduced[key] = _hash_only(item)
        return reduced
    if isinstance(value, list | tuple):
        return [_hash_only(item) for item in value]
    return value


class JsonlSink:
    """Append-only, local source of truth for benchmark observability."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        content_mode: str,
        include_research_audit: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.content_mode = content_mode
        self.trace_path = self.output_dir / "trace_events.jsonl"
        self.network_path = self.output_dir / "network_ledger.jsonl"
        self.requests_path = self.output_dir / "effective_requests.jsonl"
        self.search_audit_path = self.output_dir / "search_audit.jsonl"
        self.research_audit_path = self.output_dir / "research_audit.jsonl"
        self.include_research_audit = include_research_audit
        for path in (
            self.trace_path,
            self.network_path,
            self.requests_path,
            self.search_audit_path,
            *((self.research_audit_path,) if include_research_audit else ()),
        ):
            path.touch(exist_ok=True)

    def write_trace_events(self, events: Iterable[TraceEvent]) -> None:
        self._append(self.trace_path, (event.to_dict() for event in events))

    def write_network_events(
        self,
        *,
        run_id: str,
        task_id: str,
        framework: str,
        events: Iterable[NetworkEvent],
    ) -> None:
        rows = (
            {
                "run_id": run_id,
                "task_id": task_id,
                "framework": framework,
                **_network_event_dict(event),
            }
            for event in events
        )
        self._append(self.network_path, rows)

    def write_network_ledger_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        """Persist policy-specific ledger extensions without losing redaction."""

        self._append(self.network_path, rows)

    def write_search_audit(self, rows: Iterable[dict[str, Any]]) -> None:
        """Persist accepted and rejected Volc candidates as local JSONL evidence."""

        self._append(self.search_audit_path, rows)

    def write_research_audit(self, rows: Iterable[dict[str, Any]]) -> None:
        """Persist child lifecycle evidence for the external research IPC."""

        if not self.include_research_audit:
            raise RuntimeError("research_audit.jsonl is disabled for this profile")
        self._append(self.research_audit_path, rows)

    def write_effective_request(self, request: dict[str, Any]) -> None:
        self._append(self.requests_path, [request])

    def _append(self, path: Path, rows: Iterable[dict[str, Any]]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                safe = redact(row)
                if self.content_mode == "hash_only":
                    safe = _hash_only(safe)
                handle.write(
                    json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str) + "\n"
                )


def _network_event_dict(event: NetworkEvent) -> dict[str, Any]:
    """Keep the legacy ledger shape for ordinary model/network events."""

    row = asdict(event)
    if row.get("network_class") is None:
        row.pop("network_class", None)
    return row
