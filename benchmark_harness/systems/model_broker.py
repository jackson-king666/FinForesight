"""Parent-owned model IPC for non-ranking external research adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from benchmark_harness.adapters.recording import RecordingSyncTransport
from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.contracts import NetworkEvent, RunContext, sha256_json
from benchmark_harness.model_identity import verify_provider_response_model
from benchmark_harness.systems.chat_transcript import (
    ChatTranscriptError,
    normalize_chat_transcript,
)
from benchmark_harness.systems.external_stdio import BudgetLedger, V3ProtocolError


class HostModelBrokerError(RuntimeError):
    """Raised when a child model request violates the host-owned contract."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic = dict(diagnostic) if diagnostic is not None else None


_MODEL_REQUEST_FIELDS = frozenset(
    {
        "messages",
        "tools",
        "temperature",
        "top_p",
        "presence_penalty",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "response_format",
    }
)

_MAX_PROVIDER_ERROR_BODY_BYTES = 64 * 1024
_MAX_PROVIDER_FIELD_BYTES = 256
_ODR_SEARCH_TOOL_NAMES = frozenset({"search_filter_http"})
# Provider-selected ODR queries are validated at the service contract's 2048
# character boundary; this is independent from the host's synthesized-query
# fallback below.
_ODR_SEARCH_QUERY_MAX_CHARS = 2048
# A host-synthesized query is deliberately capped at 512 characters so a
# missing provider tool call cannot turn the full prompt into a search request.
_ODR_HOST_SYNTHESIZED_QUERY_MAX_CHARS = 512
_ODR_SEARCH_MAX_RESULTS_MIN = 1
_ODR_SEARCH_MAX_RESULTS_MAX = 100


def _bounded_provider_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if len(value.encode("utf-8")) > _MAX_PROVIDER_FIELD_BYTES:
        return None
    return value


def _message_shapes(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        content = message["content"]
        tool_calls = message.get("tool_calls")
        shapes.append(
            {
                "index": index,
                "role": message["role"],
                "keys": sorted(message),
                "content_bytes": len(content.encode("utf-8")),
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "tool_call_id_present": "tool_call_id" in message,
                "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
            }
        )
    return shapes


def _provider_http_4xx_diagnostic(
    response: httpx.Response,
    request: Mapping[str, Any],
    *,
    role: str,
    model_call_ordinal: int,
) -> dict[str, Any]:
    """Return bounded provider metadata without retaining request/response text."""

    raw_body = response.content
    diagnostic: dict[str, Any] = {
        "kind": "provider_http_4xx",
        "status": response.status_code,
        "content_type": _bounded_provider_string(response.headers.get("content-type")),
        "body_bytes": len(raw_body),
        "body_sha256": hashlib.sha256(raw_body).hexdigest(),
        "body_truncated": len(raw_body) > _MAX_PROVIDER_ERROR_BODY_BYTES,
        "wire_model_name": request.get("model"),
        "logical_role": role,
        "model_call_ordinal": model_call_ordinal,
        "request_body_sha256": sha256_json(request),
        "request_fields": {
            key: value
            for key, value in request.items()
            if key not in {"messages", "model"} and isinstance(value, str | int | float | bool)
        },
        "message_shapes": _message_shapes(list(request["messages"])),
    }
    apim_request_id = _bounded_provider_string(response.headers.get("apim-request-id"))
    if apim_request_id is not None:
        diagnostic["apim_request_id"] = apim_request_id

    # Only parse a bounded, JSON body.  The provider message is deliberately
    # represented by a hash; generic error text can echo prompts or secrets,
    # so no preview is emitted unless a future proven redactor is added.
    if len(raw_body) <= _MAX_PROVIDER_ERROR_BODY_BYTES:
        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        error = parsed.get("error") if isinstance(parsed, Mapping) else None
        if isinstance(error, Mapping):
            provider_error: dict[str, str] = {}
            for key in ("code", "type", "param"):
                value = _bounded_provider_string(error.get(key))
                if value is not None:
                    provider_error[key] = value
            innererror = error.get("innererror")
            if isinstance(innererror, Mapping):
                value = _bounded_provider_string(innererror.get("code"))
                if value is not None:
                    provider_error["innererror_code"] = value
            if provider_error:
                diagnostic["provider_error"] = provider_error
            message = error.get("message")
            if isinstance(message, str):
                diagnostic["message_sha256"] = hashlib.sha256(message.encode("utf-8")).hexdigest()
                diagnostic["message_preview_omitted"] = True
    return diagnostic


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


def _response_model_name(response: Mapping[str, Any], profile: ResolvedModelProfile) -> str | None:
    """Validate a provider model attestation without exposing secrets."""

    try:
        return verify_provider_response_model(response, profile)
    except ValueError as exc:
        raise HostModelBrokerError("model_identity_mismatch") from exc


def _integer_usage(usage: Mapping[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    raise HostModelBrokerError("model_usage_field_missing")


def _validated_usage(response: Mapping[str, Any]) -> tuple[int, int]:
    """Return provider usage or fail closed at the host accounting seam."""

    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise HostModelBrokerError("model_usage_missing")
    return (
        _integer_usage(usage, "prompt_tokens", "input_tokens"),
        _integer_usage(usage, "completion_tokens", "output_tokens"),
    )


def _conservative_mock_usage(request: Mapping[str, Any]) -> dict[str, int]:
    """Provide explicit accounting for the local mock provider.

    Real providers must return both usage fields.  The built-in mock has no
    tokenizer, so it uses a deterministic one-token-per-word lower bound and a
    one-token completion rather than silently charging zero.
    """

    input_tokens = 1
    for message in request.get("messages", []):
        if isinstance(message, Mapping) and isinstance(message.get("content"), str):
            input_tokens += len(message["content"].split())
    return {"prompt_tokens": input_tokens, "completion_tokens": 1}


def _normalize_model_request(
    payload: Mapping[str, Any],
    profile: ResolvedModelProfile,
    *,
    allow_search_tools: bool = False,
) -> tuple[str, dict[str, Any]]:
    role = payload.get("role")
    if not isinstance(role, str) or not role.strip():
        raise HostModelBrokerError("model_request_role_missing")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HostModelBrokerError("model_request_messages_missing")
    try:
        normalized_messages = normalize_chat_transcript(messages)
    except ChatTranscriptError as exc:
        raise HostModelBrokerError(f"model_request_{exc.code}") from exc
    # Native MCP/browser tools never cross this seam.  ODR may declare only its
    # parent-owned typed search function so the host can repair a gateway that
    # silently downgrades the first search-capable turn.
    tools = payload.get("tools")
    if tools is not None and (not allow_search_tools or _search_tool_name(tools) is None):
        raise HostModelBrokerError("native_model_tools_forbidden")
    if payload.get("tool_choice") is not None:
        # The host, rather than the child, owns the forced first-turn choice.
        raise HostModelBrokerError("native_model_tools_forbidden")
    request = {key: value for key, value in payload.items() if key in _MODEL_REQUEST_FIELDS}
    request["messages"] = normalized_messages
    request["model"] = profile.model_name
    request["temperature"] = profile.temperature
    request["max_tokens"] = profile.max_tokens
    if profile.top_p is not None:
        request["top_p"] = profile.top_p
    if profile.seed is not None:
        request["seed"] = profile.seed
    request["stream"] = False
    return role.strip(), request


def _search_tool_name(value: Any) -> str | None:
    """Return the single allow-listed ODR search function name, if present."""

    if not isinstance(value, list) or not value:
        return None
    names: list[str] = []
    for tool in value:
        if not isinstance(tool, Mapping):
            return None
        function = tool.get("function")
        if not isinstance(function, Mapping):
            return None
        name = function.get("name")
        if not isinstance(name, str) or name not in _ODR_SEARCH_TOOL_NAMES:
            return None
        names.append(name)
    if "search_filter_http" in names:
        return "search_filter_http"
    return names[0]


def _response_has_tool(response: Mapping[str, Any], name: str) -> bool:
    choices = response.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message")
        if not isinstance(message, Mapping):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if isinstance(function, Mapping) and function.get("name") == name:
                return True
    return False


def _validate_odr_search_tool_arguments(response: Mapping[str, Any]) -> None:
    """Validate provider-owned ODR search calls before they reach the worker.

    The pinned graph classifies malformed tool calls inside its own exception
    boundary.  Keep the fail-closed check at this parent seam so an invalid
    provider response becomes a task failure instead of a successful graph
    completion with no search receipt.
    """

    choices = response.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message")
        if not isinstance(message, Mapping):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if not isinstance(function, Mapping):
                continue
            if function.get("name") not in _ODR_SEARCH_TOOL_NAMES:
                continue
            raw_arguments = function.get("arguments")
            if not isinstance(raw_arguments, str):
                raise HostModelBrokerError("odr_search_tool_call_invalid")
            try:
                arguments = json.loads(raw_arguments)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise HostModelBrokerError("odr_search_tool_call_invalid") from exc
            if not isinstance(arguments, Mapping):
                raise HostModelBrokerError("odr_search_tool_call_invalid")
            query = arguments.get("query")
            if not isinstance(query, str) or not 1 <= len(query) <= _ODR_SEARCH_QUERY_MAX_CHARS:
                raise HostModelBrokerError("odr_search_tool_call_invalid")
            max_results = arguments.get("max_results", 5)
            if (
                not isinstance(max_results, int)
                or isinstance(max_results, bool)
                or not _ODR_SEARCH_MAX_RESULTS_MIN <= max_results <= _ODR_SEARCH_MAX_RESULTS_MAX
            ):
                raise HostModelBrokerError("odr_search_tool_call_invalid")


def _forced_search_response(
    response: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    tool_name: str,
) -> tuple[dict[str, Any], bool]:
    """Synthesize the typed search call a gateway omitted from its response."""

    messages = request.get("messages")
    query = "research task"
    query_truncated = False
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                normalized = " ".join(content.split())
                query_truncated = len(normalized) > _ODR_HOST_SYNTHESIZED_QUERY_MAX_CHARS
                query = normalized[:_ODR_HOST_SYNTHESIZED_QUERY_MAX_CHARS]
                break
    arguments: dict[str, Any] = {"query": query, "max_results": 3}
    forced = dict(response)
    choices = response.get("choices")
    first_choice = choices[0] if isinstance(choices, list) and choices else None
    choice = dict(first_choice) if isinstance(first_choice, Mapping) else {}
    first_message = choice.get("message")
    message = dict(first_message) if isinstance(first_message, Mapping) else {}
    message.setdefault("role", "assistant")
    message["tool_calls"] = [
        {
            "id": "hua174-forced-search",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }
    ]
    choice["index"] = choice.get("index", 0)
    choice["message"] = message
    choice["finish_reason"] = "tool_calls"
    forced["choices"] = [choice]
    return forced, query_truncated


@dataclass(slots=True)
class HostModelBroker:
    """One parent model identity shared by every child strategy role."""

    profile: ResolvedModelProfile
    context: RunContext
    ledger: BudgetLedger
    allowed_roles: tuple[str, ...]
    responder: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None
    force_first_search: bool = False
    call_count: int = field(default=0, init=False)
    network_events: list[NetworkEvent] = field(default_factory=list, init=False)
    requests: list[dict[str, Any]] = field(default_factory=list, init=False)
    role_model_ids: dict[str, str] = field(default_factory=dict, init=False)
    system_fingerprints: list[str] = field(default_factory=list, init=False)
    provider_model_names: list[str] = field(default_factory=list, init=False)
    forced_search_tool_name: str | None = field(default=None, init=False)
    forced_search_synthesized_count: int = field(default=0, init=False)
    forced_search_query_source: str | None = field(default=None, init=False)
    forced_search_query_truncated_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.allowed_roles = tuple(dict.fromkeys(self.allowed_roles))
        if not self.allowed_roles or any(not role.strip() for role in self.allowed_roles):
            raise HostModelBrokerError("model broker requires non-empty roles")

    @property
    def provider_response_model_names(self) -> list[str]:
        """Observed provider model attestations under the explicit field name."""

        return self.provider_model_names

    def request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise HostModelBrokerError("model_request_payload_malformed")
        role, request = _normalize_model_request(
            payload,
            self.profile,
            allow_search_tools=self.force_first_search,
        )
        if role not in self.allowed_roles:
            raise HostModelBrokerError("model_request_role_not_allowlisted")
        search_tool_name = (
            _search_tool_name(request.get("tools")) if self.force_first_search else None
        )
        force_search = search_tool_name is not None and self.forced_search_tool_name is None
        if force_search:
            request["tool_choice"] = {
                "type": "function",
                "function": {"name": search_tool_name},
            }
            self.forced_search_tool_name = search_tool_name
        try:
            self.ledger.consume(model_calls=1)
        except V3ProtocolError as exc:
            raise HostModelBrokerError(str(exc)) from exc
        self.call_count += 1
        self.role_model_ids[role] = self.profile.model_name
        audited_request = {"role": role, **request}
        self.requests.append(audited_request)
        try:
            response = self._request_once(
                request,
                role=role,
                model_call_ordinal=self.call_count,
            )
        except HostModelBrokerError:
            raise
        except Exception as exc:  # pragma: no cover - defensive transport boundary.
            self.network_events.append(
                _synthetic_model_event(request, status_code=None, error=type(exc).__name__)
            )
            raise HostModelBrokerError(f"model_transport_error:{type(exc).__name__}") from exc
        if not isinstance(response, Mapping):
            raise HostModelBrokerError("empty_model_response")
        if self.force_first_search and _response_has_tool(response, "search_filter_http"):
            _validate_odr_search_tool_arguments(response)
        if force_search and not _response_has_tool(response, search_tool_name):
            response, query_truncated = _forced_search_response(
                response,
                request,
                tool_name=search_tool_name,
            )
            self.forced_search_synthesized_count += 1
            self.forced_search_query_source = "host_message"
            if query_truncated:
                self.forced_search_query_truncated_count += 1
        if not _usable_model_response(response):
            raise HostModelBrokerError("empty_model_response")
        provider_model_name = _response_model_name(response, self.profile)
        if provider_model_name is not None:
            self.provider_model_names.append(provider_model_name)
        fingerprint = response.get("system_fingerprint")
        if isinstance(fingerprint, str) and fingerprint.strip():
            self.system_fingerprints.append(fingerprint.strip())
        input_tokens, output_tokens = _validated_usage(response)
        try:
            self.ledger.consume(input_tokens=input_tokens, output_tokens=output_tokens)
        except V3ProtocolError as exc:
            raise HostModelBrokerError(str(exc)) from exc
        return dict(response)

    def _request_once(
        self,
        request: dict[str, Any],
        *,
        role: str,
        model_call_ordinal: int,
    ) -> Mapping[str, Any]:
        if self.responder is not None:
            try:
                response = self.responder(dict(request))
            except Exception as exc:
                self.network_events.append(
                    _synthetic_model_event(request, status_code=None, error=type(exc).__name__)
                )
                raise HostModelBrokerError(
                    f"mock_model_responder_error:{type(exc).__name__}"
                ) from exc
            if not isinstance(response, Mapping):
                self.network_events.append(
                    _synthetic_model_event(request, status_code=200, error="non_object_response")
                )
                raise HostModelBrokerError("mock_model_responder_non_object")
            self.network_events.append(_synthetic_model_event(request, status_code=200))
            return response
        if self.profile.mode == "mock":
            self.network_events.append(_synthetic_model_event(request, status_code=200))
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "mock external research response",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": _conservative_mock_usage(request),
            }
        if self.profile.mode != "openai_compatible":
            raise HostModelBrokerError("unsupported_parent_model_mode")

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
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            self.network_events.extend(recorder.network_events)
            raise HostModelBrokerError(f"model_transport_error:{type(exc).__name__}") from exc
        self.network_events.extend(recorder.network_events)
        if response.status_code >= 400:
            diagnostic = (
                _provider_http_4xx_diagnostic(
                    response,
                    request,
                    role=role,
                    model_call_ordinal=model_call_ordinal,
                )
                if 400 <= response.status_code < 500
                else None
            )
            raise HostModelBrokerError(
                f"model_http_status={response.status_code}",
                diagnostic=diagnostic,
            )
        try:
            value = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise HostModelBrokerError(f"model_transport_error:{type(exc).__name__}") from exc
        if not isinstance(value, Mapping):
            raise HostModelBrokerError("model_response_not_object")
        return value

    def public_manifest(self) -> dict[str, Any]:
        return {
            "model_broker": "parent_owned",
            "model": self.profile.public_dict(),
            "roles": list(self.allowed_roles),
            "same_model_for_all_roles": True,
            "credential_in_child": False,
            "forced_first_search": self.force_first_search,
            "search_tool_policy": (
                "odr_typed_search_only" if self.force_first_search else "typed_v3_only"
            ),
        }

    def request_summary(self) -> dict[str, Any]:
        provider_response_model_names = sorted(set(self.provider_model_names))
        system_fingerprints = sorted(set(self.system_fingerprints))
        return {
            "call_count": self.call_count,
            "roles": dict(sorted(self.role_model_ids.items())),
            "provider_response_model_names": provider_response_model_names,
            # Retain the pre-correction key for existing consumers while the
            # explicit response-attestation key becomes canonical.
            "provider_model_names": provider_response_model_names,
            "forced_search_tool_name": self.forced_search_tool_name,
            "forced_search_synthesized_count": self.forced_search_synthesized_count,
            "forced_search_query_source": self.forced_search_query_source,
            "forced_search_query_truncated_count": self.forced_search_query_truncated_count,
            "system_fingerprints": system_fingerprints,
            "model_provenance": self.profile.provenance_dict(
                provider_response_model_names=provider_response_model_names,
                system_fingerprints=system_fingerprints,
            ),
            "requests_sha256": sha256_json(
                [
                    {
                        "role": request["role"],
                        "model": request["model"],
                        "messages_sha256": sha256_json(request["messages"]),
                    }
                    for request in self.requests
                ]
            ),
        }


__all__ = ["HostModelBroker", "HostModelBrokerError"]
