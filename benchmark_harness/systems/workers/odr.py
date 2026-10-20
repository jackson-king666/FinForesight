"""Disposable one-task worker for the pinned Open Deep Research graph.

The upstream checkout remains read-only.  Its ``get_all_tools`` factory and
compiled LangGraph are imported from the attested revision, while the host
owns the only search tool and every model call.  The small import shims below
are only for the benchmark's verify-only environment: ODR's optional MCP,
Tavily, and ``langchain`` packages are not part of the harness dependency
set.  They make importing the pinned modules deterministic without giving the
worker a network client.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from benchmark_harness.systems.adapters.odr import (
    ODR_LICENSE_SHA256,
    ODR_PROVENANCE,
)
from benchmark_harness.systems.chat_transcript import (
    ChatTranscriptError,
    normalize_chat_transcript,
)
from benchmark_harness.systems.clock import DeclaredExperimentClock
from benchmark_harness.systems.fixtures import verify_fixture_source
from benchmark_harness.systems.provenance import (
    verify_declared_file_hashes,
    verify_system_source,
)
from benchmark_harness.systems.worker_diagnostics import (
    build_odr_structured_output_diagnostic,
)
from benchmark_harness.systems.workers.common import (
    WorkerChannel,
    citations_from_search,
    emit_trajectory,
    json_context,
    request_task_bound_prediction,
    response_content,
    send_task_error,
    source_import_path,
    task_start,
    validate_search_query,
)
from benchmark_harness.systems.workers.guards import verify_odr_hooks
from benchmark_harness.systems.workers.output_contract import (
    reject_fixture_sentinel,
)


class SearchFilterInput(BaseModel):
    """Only the query and bounded result count cross the ODR tool seam."""

    query: str = Field(min_length=1, max_length=2048)
    max_results: int = Field(default=5, ge=1, le=100)


ODR_STRUCTURED_RETRY_CORRECTION = (
    "The previous output failed schema validation. Return only one JSON object that "
    "satisfies the same schema; do not answer the benchmark task directly and do not "
    "add other keys or text."
)


class ODRStructuredOutputError(RuntimeError):
    """A classified ODR schema failure with no model-content disclosure."""

    def __init__(self, diagnostic: Mapping[str, Any]) -> None:
        super().__init__("odr_structured_model_response_invalid")
        self.diagnostic = dict(diagnostic)


def _verify_source(source_dir: Path, *, fixture: bool = False) -> dict[str, Any]:
    if fixture:
        return verify_fixture_source(
            source_dir,
            ODR_PROVENANCE,
            system="open_deep_research",
            hook_verifier=verify_odr_hooks,
        )
    evidence = verify_system_source(source_dir, ODR_PROVENANCE)
    return {
        **evidence,
        "license_files": verify_declared_file_hashes(
            source_dir,
            {"LICENSE": ODR_LICENSE_SHA256},
        ),
        "worker_hooks": verify_odr_hooks(source_dir),
    }


def _install_optional_import_shims() -> None:
    """Make ODR's optional imports explicit and inert in the harness env."""

    if "langchain" not in sys.modules:
        langchain = types.ModuleType("langchain")
        langchain.__path__ = []  # type: ignore[attr-defined]
        chat_models = types.ModuleType("langchain.chat_models")
        chat_models.init_chat_model = lambda *_args, **_kwargs: None
        langchain.chat_models = chat_models  # type: ignore[attr-defined]
        sys.modules["langchain"] = langchain
        sys.modules["langchain.chat_models"] = chat_models

    if "langchain_mcp_adapters" not in sys.modules:
        adapters = types.ModuleType("langchain_mcp_adapters")
        client = types.ModuleType("langchain_mcp_adapters.client")

        class DisabledMCPClient:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("odr_mcp_disabled_by_host")

            async def get_tools(self) -> list[Any]:  # pragma: no cover - defensive.
                return []

        client.MultiServerMCPClient = DisabledMCPClient
        adapters.client = client  # type: ignore[attr-defined]
        sys.modules["langchain_mcp_adapters"] = adapters
        sys.modules["langchain_mcp_adapters.client"] = client

    if "mcp" not in sys.modules:
        mcp = types.ModuleType("mcp")
        mcp.McpError = type("McpError", (Exception,), {})
        sys.modules["mcp"] = mcp

    if "tavily" not in sys.modules:
        tavily = types.ModuleType("tavily")
        tavily.AsyncTavilyClient = type("AsyncTavilyClient", (), {})
        sys.modules["tavily"] = tavily


def _tool_name(value: Any) -> str:
    if isinstance(value, Mapping):
        name = value.get("name")
    else:
        name = getattr(value, "name", None)
        if name is None:
            name = getattr(value, "__name__", None)
    return name if isinstance(name, str) else ""


def _response_tool_calls(
    response: Mapping[str, Any], *, expected_name: str
) -> list[dict[str, Any]]:
    """Translate the host's bounded OpenAI tool calls to LangChain calls."""

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    for choice in choices:
        if not isinstance(choice, Mapping):
            raise RuntimeError("odr_search_tool_call_invalid")
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise RuntimeError("odr_search_tool_call_invalid")
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            continue
        if not isinstance(raw_calls, list):
            raise RuntimeError("odr_search_tool_call_invalid")
        if not raw_calls:
            continue
        calls: list[dict[str, Any]] = []
        for call in raw_calls:
            if not isinstance(call, Mapping):
                raise RuntimeError("odr_search_tool_call_invalid")
            function = call.get("function")
            call_id = call.get("id")
            name = function.get("name") if isinstance(function, Mapping) else None
            raw_arguments = function.get("arguments") if isinstance(function, Mapping) else None
            if (
                not isinstance(call_id, str)
                or not isinstance(name, str)
                or name != expected_name
                or not isinstance(raw_arguments, str)
            ):
                raise RuntimeError("odr_search_tool_call_invalid")
            try:
                arguments = json.loads(raw_arguments)
                validated_arguments = SearchFilterInput.model_validate(arguments)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError("odr_search_tool_call_invalid") from exc
            calls.append(
                {
                    "name": name,
                    "args": validated_arguments.model_dump(),
                    "id": call_id,
                    "type": "tool_call",
                }
            )
        return calls
    return []


@dataclass
class _PinnedODRGraphModel:
    """Minimal LangChain-compatible model that delegates to the host broker."""

    channel: WorkerChannel
    task_prompt: str = ""
    fixture: bool = False
    structured_schema: type[BaseModel] | None = None
    bound_tools: tuple[Any, ...] = ()
    structured_retry_attempts: int = 1
    state: dict[str, int] = field(default_factory=dict)
    request_log: list[list[dict[str, Any]]] = field(default_factory=list)

    def _copy(self, **changes: Any) -> _PinnedODRGraphModel:
        values = {
            "channel": self.channel,
            "task_prompt": self.task_prompt,
            "fixture": self.fixture,
            "structured_schema": self.structured_schema,
            "bound_tools": self.bound_tools,
            "structured_retry_attempts": self.structured_retry_attempts,
            "state": self.state,
            "request_log": self.request_log,
        }
        values.update(changes)
        return _PinnedODRGraphModel(**values)

    def with_structured_output(self, schema: type[BaseModel], **_kwargs: Any) -> Any:
        return self._copy(structured_schema=schema, structured_retry_attempts=1)

    def bind_tools(self, tools: Sequence[Any], **_kwargs: Any) -> Any:
        return self._copy(
            bound_tools=tuple(tools),
            structured_schema=None,
            structured_retry_attempts=1,
        )

    def with_retry(self, **_kwargs: Any) -> Any:
        """Retain the upstream finite retry bound at the host model seam."""

        if set(_kwargs) != {"stop_after_attempt"}:
            raise RuntimeError("odr_structured_retry_bound_invalid")
        attempts = _kwargs["stop_after_attempt"]
        if not isinstance(attempts, int) or isinstance(attempts, bool) or not 1 <= attempts <= 3:
            raise RuntimeError("odr_structured_retry_bound_invalid")
        return self._copy(structured_retry_attempts=attempts)

    def with_config(self, _config: Mapping[str, Any] | None = None, **_kwargs: Any) -> Any:
        return self

    def _bound_search_tools(self) -> list[dict[str, Any]] | None:
        if "search_filter_http" not in {_tool_name(tool) for tool in self.bound_tools}:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_filter_http",
                    "description": "Use the host-owned filtered search capability.",
                    "parameters": SearchFilterInput.model_json_schema(),
                },
            }
        ]

    @staticmethod
    def _message_payload(messages: Sequence[Any]) -> list[dict[str, Any]]:
        """Serialize LangChain messages into the strict local tool transcript."""

        def content_text(value: Any, *, index: int) -> str:
            if isinstance(value, str):
                return value
            try:
                return json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError):
                raise ChatTranscriptError("message_content_not_serializable", index=index)

        def arguments_text(value: Any, *, index: int) -> str:
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                except (TypeError, ValueError, json.JSONDecodeError):
                    raise ChatTranscriptError("tool_arguments_malformed", index=index)
            else:
                parsed = value
            try:
                return json.dumps(
                    parsed,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError):
                raise ChatTranscriptError("tool_arguments_malformed", index=index)

        def tool_call_payload(call: Any, *, index: int, call_index: int) -> dict[str, Any]:
            if isinstance(call, Mapping):
                call_id = call.get("id")
                name = call.get("name")
                arguments = call.get("args")
                function = call.get("function")
                if isinstance(function, Mapping):
                    name = function.get("name", name)
                    arguments = function.get("arguments", arguments)
            else:
                call_id = getattr(call, "id", None)
                name = getattr(call, "name", None)
                arguments = getattr(call, "args", None)
            if not isinstance(call_id, str) or not call_id.strip():
                raise ChatTranscriptError("tool_call_id_missing", index=index)
            if not isinstance(name, str) or not name.strip():
                raise ChatTranscriptError("tool_call_name_missing", index=index)
            return {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_text(arguments, index=call_index),
                },
            }

        result: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if isinstance(message, Mapping):
                payload = dict(message)
                if "content" in payload:
                    payload["content"] = content_text(payload["content"], index=index)
                if "tool_calls" in payload:
                    raw_tool_calls = payload["tool_calls"]
                    if not isinstance(raw_tool_calls, list):
                        raise ChatTranscriptError("tool_calls_malformed", index=index)
                    payload["tool_calls"] = [
                        tool_call_payload(call, index=index, call_index=call_index)
                        for call_index, call in enumerate(raw_tool_calls)
                    ]
            elif isinstance(message, BaseMessage):
                role = {
                    "human": "user",
                    "ai": "assistant",
                    "system": "system",
                    "tool": "tool",
                }.get(message.type)
                if role is None:
                    raise ChatTranscriptError("message_role_invalid", index=index)
                payload = {"role": role, "content": content_text(message.content, index=index)}
                if role == "assistant":
                    raw_tool_calls = getattr(message, "tool_calls", None)
                    if raw_tool_calls:
                        payload["tool_calls"] = [
                            tool_call_payload(call, index=index, call_index=call_index)
                            for call_index, call in enumerate(raw_tool_calls)
                        ]
                elif role == "tool":
                    payload["tool_call_id"] = getattr(message, "tool_call_id", None)
            else:
                raise ChatTranscriptError("message_type_unsupported", index=index)
            result.append(payload)
        return normalize_chat_transcript(result)

    def _host_completion(
        self,
        messages: Sequence[Any],
        role: str,
        *,
        response_format: dict[str, Any] | None = None,
        include_bound_tools: bool = True,
    ) -> Mapping[str, Any]:
        payload = self._message_payload(messages)
        if not self.fixture:
            reject_fixture_sentinel(payload, field="odr_model_request")
        self.request_log.append(payload)
        request_kwargs: dict[str, Any] = {
            "role": role,
            "messages": payload,
            "response_format": response_format,
        }
        tools = self._bound_search_tools() if include_bound_tools else None
        if tools is not None:
            request_kwargs["tools"] = tools
        response = self.channel.request_model(**request_kwargs)
        if not isinstance(response, Mapping):
            raise RuntimeError("odr_model_response_invalid")
        return response

    def _host_response(
        self,
        messages: Sequence[Any],
        role: str,
        *,
        response_format: dict[str, Any] | None = None,
        include_bound_tools: bool = True,
    ) -> str:
        response = self._host_completion(
            messages,
            role,
            response_format=response_format,
            include_bound_tools=include_bound_tools,
        )
        content = response_content(response)
        if not self.fixture:
            reject_fixture_sentinel(content, field="odr_model_response")
        return content

    def _structured_response(self, messages: Sequence[Any]) -> BaseModel:
        """Validate upstream structured output without weakening its contract."""

        assert self.structured_schema is not None
        schema_instruction = SystemMessage(
            content="Return only a JSON object matching this schema; do not include "
            "markdown or prose: "
            f"{json.dumps(self.structured_schema.model_json_schema(), sort_keys=True)}"
        )
        base_messages = [*messages, schema_instruction]
        for attempt in range(1, self.structured_retry_attempts + 1):
            request_messages = base_messages
            if attempt > 1:
                # Build each retry from the same original input so a failed
                # response is never echoed or accumulated in the transcript.
                request_messages = [
                    *base_messages,
                    SystemMessage(content=ODR_STRUCTURED_RETRY_CORRECTION),
                ]
            content = self._host_response(
                request_messages,
                "summarization",
                response_format={"type": "json_object"},
            )
            if self.fixture:
                if self.structured_schema.__name__ == "ResearchQuestion":
                    return self.structured_schema(research_brief=self.task_prompt)
                if self.structured_schema.__name__ == "ClarifyWithUser":
                    return self.structured_schema(
                        need_clarification=False,
                        question="",
                        verification=self.task_prompt,
                    )
            try:
                return self.structured_schema.model_validate_json(content)
            except (TypeError, ValueError) as exc:
                if attempt < self.structured_retry_attempts:
                    continue
                diagnostic = build_odr_structured_output_diagnostic(
                    schema_name=self.structured_schema.__name__,
                    structured_attempt=attempt,
                    broker_call_ordinal=len(self.request_log),
                    response=content,
                )
                raise ODRStructuredOutputError(diagnostic) from exc
        raise AssertionError("structured retry loop did not return or fail")

    def _host_search_query(self) -> str:
        """Ask the parent model for a task-bound query before using Search+Filter."""

        messages = [
            SystemMessage(
                content=(
                    "Generate one concise search query for the task below. Return only "
                    'a JSON object with exactly one string field named "query". Do not '
                    "include markdown, prose, URLs from memory, or any other fields."
                )
            ),
            HumanMessage(content=f"Task: {self.task_prompt}"),
        ]
        content = self._host_response(
            messages,
            "research",
            response_format={"type": "json_object"},
            include_bound_tools=False,
        )
        if self.fixture:
            return "fixture search query"
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("odr_search_query_invalid") from exc
        query = parsed.get("query") if isinstance(parsed, Mapping) else None
        try:
            return validate_search_query(query)
        except ValueError as exc:
            raise RuntimeError("odr_search_query_invalid") from exc

    async def ainvoke(self, messages: Sequence[Any], **_kwargs: Any) -> Any:
        if self.structured_schema is not None:
            return self._structured_response(messages)

        names = {_tool_name(tool) for tool in self.bound_tools}
        if "ConductResearch" in names:
            role = "research"
            self.state["supervisor_calls"] = self.state.get("supervisor_calls", 0) + 1
            content = self._host_response(messages, role)
            if self.state["supervisor_calls"] == 1:
                return AIMessage(
                    content=content,
                    tool_calls=[
                        {
                            "name": "ConductResearch",
                            "args": {"research_topic": self.task_prompt},
                            "id": "odr-conduct-1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(
                content=content,
                tool_calls=[
                    {
                        "name": "ResearchComplete",
                        "args": {},
                        "id": "odr-complete-1",
                        "type": "tool_call",
                    }
                ],
            )

        if "search_filter_http" in names:
            role = "research"
            self.state["researcher_calls"] = self.state.get("researcher_calls", 0) + 1
            first_researcher_call = self.state["researcher_calls"] == 1
            response = self._host_completion(
                messages,
                role,
                include_bound_tools=first_researcher_call,
            )
            content = response_content(response)
            if not self.fixture:
                reject_fixture_sentinel(content, field="odr_model_response")
            if self.state["researcher_calls"] == 1:
                response_tool_calls = _response_tool_calls(
                    response,
                    expected_name="search_filter_http",
                )
                if response_tool_calls:
                    return AIMessage(content=content, tool_calls=response_tool_calls)
                search_query = self._host_search_query()
                return AIMessage(
                    content=content,
                    tool_calls=[
                        {
                            "name": "search_filter_http",
                            "args": {
                                "query": search_query,
                                "max_results": 3,
                            },
                            "id": "odr-search-1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content=content)

        self.state["unbound_calls"] = self.state.get("unbound_calls", 0) + 1
        role = "compression" if self.state["unbound_calls"] == 1 else "final_report"
        return AIMessage(content=self._host_response(messages, role))


def _load_pinned_graph(
    source: Path,
    channel: WorkerChannel,
    search_tool: StructuredTool | None,
    *,
    closed_book: bool,
    task_prompt: str,
    fixture: bool,
) -> tuple[Any, _PinnedODRGraphModel]:
    """Import the pinned factory/graph and bind the host-owned tool into it."""

    _install_optional_import_shims()
    sys.path.insert(0, str(source_import_path(source)))
    try:
        from open_deep_research import deep_researcher as graph_module
        from open_deep_research import utils as upstream_utils
        from open_deep_research.configuration import Configuration, SearchAPI
    except (ImportError, OSError) as exc:
        raise RuntimeError("odr_pinned_runtime_import_failed") from exc

    if not callable(getattr(upstream_utils, "get_search_tool", None)) or not callable(
        getattr(upstream_utils, "get_all_tools", None)
    ):
        raise RuntimeError("odr_pinned_tool_factory_missing")
    if not hasattr(graph_module, "deep_researcher"):
        raise RuntimeError("odr_pinned_langgraph_missing")

    upstream_get_search_tool = upstream_utils.get_search_tool
    upstream_get_all_tools = upstream_utils.get_all_tools

    async def host_owned_tools(config: Mapping[str, Any]) -> list[Any]:
        configurable = Configuration.from_runnable_config(config)
        if configurable.search_api is not SearchAPI.NONE or configurable.mcp_config is not None:
            raise RuntimeError("odr_upstream_network_or_mcp_configuration_drift")
        # Exercise the pinned search factory itself.  NONE must produce no
        # native search tool before the typed host tool is appended.  In the
        # closed-book regime no replacement tool is appended at all; the
        # upstream graph still owns its normal tool loop, but it receives only
        # the non-network core tools.
        native_search_tools = await upstream_get_search_tool(SearchAPI.NONE)
        if native_search_tools:
            raise RuntimeError("odr_native_search_factory_not_empty")
        upstream_tools = await upstream_get_all_tools(config)
        names = {_tool_name(tool) for tool in upstream_tools}
        forbidden = names.intersection(
            {"tavily_search", "web_search", "search", "browser", "fetch"}
        )
        if forbidden:
            raise RuntimeError("odr_upstream_search_tool_leaked:" + ",".join(sorted(forbidden)))
        if closed_book:
            if search_tool is not None:
                raise RuntimeError("odr_closed_book_search_tool_present")
            return list(upstream_tools)
        if search_tool is None:
            raise RuntimeError("odr_cutoff_search_tool_missing")
        return [*upstream_tools, search_tool]

    # The graph's researcher and researcher_tools nodes resolve this global at
    # call time.  Patching the imported reference keeps the upstream graph and
    # its tool execution loop intact; it does not replace either node.
    graph_module.get_all_tools = host_owned_tools
    graph_model = _PinnedODRGraphModel(
        channel,
        task_prompt=task_prompt,
        fixture=fixture,
    )
    graph_module.configurable_model = graph_model
    return graph_module, graph_model


def _serialize_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return []
    return _PinnedODRGraphModel._message_payload(value)


def _request_task_bound_prediction(
    channel: WorkerChannel,
    *,
    task_id: str,
    task_prompt: str,
    research_context: str,
    citations: list[str],
    fixture: bool,
) -> tuple[str, list[dict[str, str]]]:
    return request_task_bound_prediction(
        channel,
        task_id=task_id,
        task_prompt=task_prompt,
        research_context=research_context,
        citations=citations,
        role="final_report",
        fixture=fixture,
        error_prefix="odr",
    )


def run_one_task(
    source_dir: str | Path,
    channel: WorkerChannel | None = None,
    *,
    fixture: bool = False,
) -> None:
    source = Path(source_dir).expanduser().resolve()
    _verify_source(source, fixture=fixture)
    sys.path.insert(0, str(source_import_path(source)))

    # Importing the pinned Configuration proves the task-scoped flags before
    # loading the graph.  The graph itself is loaded below and run once.
    _install_optional_import_shims()
    from open_deep_research.configuration import Configuration, SearchAPI

    upstream_config = Configuration(
        search_api=SearchAPI.NONE,
        mcp_config=None,
        allow_clarification=False,
    )
    if (
        upstream_config.search_api is not SearchAPI.NONE
        or upstream_config.mcp_config is not None
        or upstream_config.allow_clarification
    ):
        raise RuntimeError("odr_upstream_configuration_contract_failed")

    channel = channel or WorkerChannel()
    start, payload = task_start(channel, adapter_id="odr-typed-host-v3")
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise RuntimeError("odr_worker_configuration_missing")
    required = {
        "search_api": "none",
        "mcp_config": None,
        "allow_clarification": False,
        "native_search": False,
        "native_browser": False,
        "listener": False,
        "public_network": False,
        "telemetry": False,
    }
    if any(configuration.get(key) != value for key, value in required.items()):
        raise RuntimeError("odr_worker_configuration_drift")
    if payload.get("mcp_config") is not None or payload.get("allow_clarification") is not False:
        raise RuntimeError("odr_task_boundary_flags_invalid")
    policy = payload.get("policy")
    if not isinstance(policy, Mapping) or policy.get("regime") not in {
        "closed_book",
        "cutoff_controlled",
        "unrestricted_current_web",
    }:
        raise RuntimeError("odr_retrieval_policy_missing")
    closed_book = policy.get("regime") == "closed_book"
    search_capability = payload.get("search_capability")
    if closed_book and search_capability is not None:
        raise RuntimeError("odr_closed_book_search_capability_present")
    if not closed_book and not isinstance(search_capability, dict):
        raise RuntimeError("odr_search_capability_missing")

    channel.send(
        "hello",
        {
            "protocol": "external_research_stdio/v3",
            "adapter_id": "odr-typed-host-v3",
            "child_network": "forbidden",
            "mcp_enabled": False,
            "mcp_config": None,
            "search_api": "none",
            "allow_clarification": False,
            "native_search": False,
            "native_browser": False,
            "listener": False,
            "public_network": False,
            "telemetry": False,
            "ranking_eligible": False,
            "injected_search": None if closed_book else "search_filter_http/v1",
            "structured_tool": None if closed_book else "search_filter_http",
            "pinned_factory": "open_deep_research.utils.get_all_tools",
            "pinned_graph": "open_deep_research.deep_researcher.deep_researcher",
            "source_revision": ODR_PROVENANCE.revision,
        },
        request_id=start.request_id,
    )

    def search_filter_http(query: str, max_results: int = 5) -> str:
        if not fixture:
            reject_fixture_sentinel(query, field="odr_search_query")
        try:
            search_query = validate_search_query(query)
        except ValueError as exc:
            raise RuntimeError("odr_search_query_invalid") from exc
        response = channel.request_search(query=search_query, max_results=max_results)
        response_text = json_context(response)
        if not fixture:
            reject_fixture_sentinel(response_text, field="odr_search_response")
        search_filter_http.citations.extend(citations_from_search(response))
        return response_text

    search_filter_http.citations = []
    search_tool: StructuredTool | None = None
    if not closed_book:
        search_tool = StructuredTool.from_function(
            func=search_filter_http,
            name="search_filter_http",
            description="Use the host-owned filtered search capability.",
            args_schema=SearchFilterInput,
        )
        if search_tool.name != "search_filter_http":
            raise RuntimeError("odr_structured_tool_name_drift")

    task_prompt = payload.get("prompt")
    if not isinstance(task_prompt, str) or not task_prompt.strip():
        raise RuntimeError("odr_task_prompt_missing")
    if not fixture:
        reject_fixture_sentinel(task_prompt, field="odr_task_prompt")
    task_id = payload.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise RuntimeError("odr_task_id_missing")
    graph_module, graph_model = _load_pinned_graph(
        source,
        channel,
        search_tool,
        closed_book=closed_book,
        task_prompt=task_prompt,
        fixture=fixture,
    )
    clock = DeclaredExperimentClock(payload["experiment_as_of"])
    graph_config = {
        "configurable": {
            "search_api": "none",
            "mcp_config": None,
            "allow_clarification": False,
            "research_model": "host:parent",
            "compression_model": "host:parent",
            "final_report_model": "host:parent",
            "max_researcher_iterations": 1,
            "max_concurrent_research_units": 1,
            "max_react_tool_calls": 1,
        }
    }
    # ``get_today_str`` in the pinned graph is only used for prompt text.  It
    # is replaced with the declared clock so no machine date reaches a model.
    graph_module.get_today_str = lambda: clock.prompt_date()
    result = asyncio.run(
        graph_module.deep_researcher.ainvoke(
            {"messages": [HumanMessage(content=task_prompt)]},
            config=graph_config,
        )
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("odr_pinned_graph_result_malformed")
    citations = list(dict.fromkeys(search_filter_http.citations))
    graph_context = json_context(
        {
            "research_brief": result.get("research_brief"),
            "notes": result.get("notes"),
            "messages": _serialize_messages(result.get("messages")),
        }
    )
    prediction, final_messages = _request_task_bound_prediction(
        channel,
        task_id=task_id,
        task_prompt=task_prompt,
        research_context=graph_context,
        citations=citations,
        fixture=fixture,
    )
    if channel._last_received_seq is None:  # pragma: no cover - protocol guard.
        raise RuntimeError("odr_graph_did_not_use_parent_channel")

    emit_trajectory(channel, node="pinned_langgraph", detail="deep_researcher")
    channel.send(
        "task.result",
        {
            "prediction": prediction,
            "messages": final_messages,
            "citations": citations,
            "usage": {
                "worker_model_roles": 4,
                "worker_model_calls": len(graph_model.request_log) + 1,
                "worker_search_tools": 0 if closed_book else 1,
                "pinned_factory": True,
                "pinned_langgraph": True,
            },
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args(argv)
    channel = WorkerChannel()
    try:
        source = args.source.expanduser().resolve()
        _verify_source(source, fixture=args.fixture)
        run_one_task(source, channel, fixture=args.fixture)
    except BaseException as exc:
        send_task_error(channel, exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the stdio smoke.
    raise SystemExit(main())


__all__ = [
    "ODRStructuredOutputError",
    "ODR_STRUCTURED_RETRY_CORRECTION",
    "SearchFilterInput",
    "main",
    "run_one_task",
]
