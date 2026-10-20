"""Strict, local tool-transcript handling for Chat Completions requests.

The external workers may use framework-native message objects, but the host
broker accepts only this small allowlisted wire shape.  Keeping the parser in
one module lets the worker serializer and the host boundary enforce the same
correlation rules without sharing provider clients or credentials.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


class ChatTranscriptError(ValueError):
    """Raised with a stable code when a local tool transcript is malformed."""

    def __init__(self, code: str, *, index: int | None = None) -> None:
        self.code = code
        self.index = index
        super().__init__(code)


def _fail(code: str, *, index: int | None = None) -> None:
    raise ChatTranscriptError(code, index=index)


def _canonical_arguments(value: Any, *, index: int) -> str:
    if not isinstance(value, str):
        _fail("tool_arguments_malformed", index=index)

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON constant")

    try:
        parsed = json.loads(value, parse_constant=reject_constant)
        return json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        _fail("tool_arguments_malformed", index=index)
    raise AssertionError("unreachable")


def _message_content(message: Mapping[str, Any], *, index: int) -> str:
    content = message.get("content")
    if not isinstance(content, str):
        _fail("message_content_malformed", index=index)
    return content


def _tool_call(call: Any, *, index: int, call_index: int) -> dict[str, Any]:
    if not isinstance(call, Mapping):
        _fail("tool_call_malformed", index=index)
    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id.strip():
        _fail("tool_call_id_missing", index=index)
    if set(call) != {"id", "type", "function"}:
        _fail("tool_call_malformed", index=index)
    if call.get("type") != "function":
        _fail("tool_call_type_invalid", index=index)
    function = call.get("function")
    if not isinstance(function, Mapping) or set(function) != {"name", "arguments"}:
        _fail("tool_call_function_malformed", index=index)
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        _fail("tool_call_name_missing", index=index)
    arguments = _canonical_arguments(function.get("arguments"), index=call_index)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def normalize_chat_transcript(messages: Any) -> list[dict[str, Any]]:
    """Validate and canonicalize an allowlisted local Chat Completions transcript.

    Only ``system``, ``user``, ``assistant``, and ``tool`` messages are
    accepted.  Tool results must be the contiguous messages immediately after
    their assistant tool-call block, and every ID is single-use.
    """

    if not isinstance(messages, list) or not messages:
        _fail("messages_missing")

    normalized: list[dict[str, Any]] = []
    seen_tool_call_ids: set[str] = set()
    pending_tool_call_ids: list[str] = []

    for index, raw_message in enumerate(messages):
        if not isinstance(raw_message, Mapping):
            _fail("message_malformed", index=index)
        message = dict(raw_message)
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            _fail("message_role_invalid", index=index)
        if pending_tool_call_ids and role != "tool":
            _fail("tool_result_not_immediate", index=index)

        content = _message_content(message, index=index)
        if role in {"system", "user"}:
            if set(message) != {"role", "content"}:
                _fail("message_fields_malformed", index=index)
            normalized.append({"role": role, "content": content})
            continue

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                _fail("tool_call_id_missing", index=index)
            if set(message) != {"role", "content", "tool_call_id"}:
                _fail("message_fields_malformed", index=index)
            if tool_call_id not in pending_tool_call_ids:
                _fail("tool_result_orphan", index=index)
            pending_tool_call_ids.remove(tool_call_id)
            normalized.append({"role": "tool", "content": content, "tool_call_id": tool_call_id})
            continue

        # The only remaining role is assistant.  A tool-call block is either
        # absent or a non-empty list of exactly shaped function calls.
        if set(message) not in (
            {"role", "content"},
            {"role", "content", "tool_calls"},
        ):
            _fail("message_fields_malformed", index=index)
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls is None:
            normalized.append({"role": "assistant", "content": content})
            continue
        if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
            _fail("tool_calls_malformed", index=index)
        calls = [
            _tool_call(call, index=index, call_index=call_index)
            for call_index, call in enumerate(raw_tool_calls)
        ]
        call_ids = [call["id"] for call in calls]
        if len(set(call_ids)) != len(call_ids) or any(
            call_id in seen_tool_call_ids for call_id in call_ids
        ):
            _fail("tool_call_id_duplicate", index=index)
        seen_tool_call_ids.update(call_ids)
        pending_tool_call_ids.extend(call_ids)
        normalized.append({"role": "assistant", "content": content, "tool_calls": calls})

    if pending_tool_call_ids:
        _fail("tool_results_missing")
    return normalized


__all__ = ["ChatTranscriptError", "normalize_chat_transcript"]
