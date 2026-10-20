"""LangGraph adapter for the strict one-turn protocol."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import httpx

from benchmark_harness.adapters.base import (
    _RETRY_BACKOFF_SECONDS,
    package_version,
    result_from_transport,
    transient_transport_failure,
)
from benchmark_harness.adapters.recording import RecordingSyncTransport
from benchmark_harness.contracts import (
    AdapterResult,
    RunContext,
    StrictTransport,
    TaskInput,
    ToolCall,
    TransportResponse,
    sha256_json,
)
from benchmark_harness.search import AgenticSearchSession


class AmbientLangSmithTracingError(RuntimeError):
    """Raised before graph construction when hosted tracing may be active."""


_TRACE_ENVIRONMENT_NAMES = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
)
_DISABLED_TRACE_VALUES = {"", "0", "false", "no", "off"}
_KNOWN_STREAM_MODES = {"updates", "messages", "tasks", "debug"}


def _configured_timeout(timeout_seconds: float) -> httpx.Timeout:
    """Return one explicit timeout budget for every HTTPX operation."""

    return httpx.Timeout(timeout_seconds)


def _ensure_no_ambient_langsmith_tracing() -> None:
    """Fail closed rather than risk an ambient hosted-tracing export.

    LangChain enables LangSmith from either legacy or current environment
    variables. A malformed non-empty value is rejected too: only explicit
    false-like values are safe for this strict, local-observability harness.
    """

    enabled_names = [
        name
        for name in _TRACE_ENVIRONMENT_NAMES
        if os.environ.get(name, "").strip().lower() not in _DISABLED_TRACE_VALUES
    ]
    if enabled_names:
        raise AmbientLangSmithTracingError("ambient_langsmith_tracing_enabled")

    # A caller can also enable tracing programmatically with
    # tracing_v2_enabled(), which bypasses the environment entirely. Avoid
    # importing LangChain until after the env gate, then reject that context.
    try:
        from langchain_core.tracers.context import tracing_v2_callback_var
    except ImportError:
        return
    if tracing_v2_callback_var.get() is not None:
        raise AmbientLangSmithTracingError("ambient_langsmith_tracing_context")


def _safe_hash(value: Any) -> str:
    """Hash arbitrary framework objects without persisting their representation."""

    try:
        return sha256_json(value)
    except (TypeError, ValueError):
        return sha256_json({"type": type(value).__name__, "repr": repr(value)})


def _safe_label(value: Any) -> str | None:
    """Keep framework control labels only when they cannot contain prose."""

    if not isinstance(value, str) or not value or len(value) > 128:
        return None
    if not all(character.isalnum() or character in "._:-/" for character in value):
        return None
    return value


def _stream_mode(value: Any) -> str:
    """Avoid recording arbitrary tuple content as a stream mode."""

    if isinstance(value, str) and value in _KNOWN_STREAM_MODES:
        return value
    return f"unknown:{_safe_hash(value)[:12]}"


def _record_control_evidence(summary: dict[str, Any], payload: dict[str, Any]) -> None:
    """Extract graph control flow only; never copy state or message content."""

    for source in (payload, payload.get("payload")):
        if not isinstance(source, dict):
            continue

        for key in ("node", "name", "langgraph_node"):
            if key not in source or "node" in summary:
                continue
            value = source[key]
            label = _safe_label(value)
            if label is not None:
                summary["node"] = label
            summary["node_hash"] = _safe_hash(value)

        for key in ("path", "langgraph_path"):
            if key not in source or "path_hash" in summary:
                continue
            value = source[key]
            summary["path_hash"] = _safe_hash(value)
            if isinstance(value, list | tuple):
                labels = [_safe_label(item) for item in value]
                if all(label is not None for label in labels):
                    summary["path"] = labels

        for key in ("task_id", "task", "id"):
            if key in source and "task_hash" not in summary:
                summary["task_hash"] = _safe_hash(source[key])

        for key in ("trigger", "triggers"):
            if key in source and "trigger_hash" not in summary:
                summary["trigger_hash"] = _safe_hash(source[key])

        if "result" in source and "result_hash" not in summary:
            summary["result_type"] = type(source["result"]).__name__
            summary["result_hash"] = _safe_hash(source["result"])

        if "error" in source and source["error"] is not None and "error_hash" not in summary:
            error = source["error"]
            summary["error_type"] = type(error).__name__
            summary["error_hash"] = _safe_hash(error)


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = [
            item if isinstance(item, str) else str(item.get("text", item)) for item in content
        ]
        return "".join(pieces)
    return str(content)


def _stream_event_summary(event: Any) -> dict[str, Any]:
    """Record redacted graph control flow without duplicating response text."""

    if isinstance(event, tuple) and event:
        summary: dict[str, Any] = {
            "stream_mode": _stream_mode(event[0]),
            "event_shape": "tuple",
            "tuple_size": len(event),
        }
        if len(event) > 1 and isinstance(event[1], dict):
            _record_control_evidence(summary, event[1])
            summary["event_type"] = _safe_label(event[1].get("type")) or "dict"
        return summary
    if isinstance(event, dict):
        summary = {
            "stream_mode": "untyped",
            "event_shape": "dict",
            "event_type": _safe_label(event.get("type")) or "dict",
        }
        _record_control_evidence(summary, event)
        return summary
    return {"stream_mode": "untyped", "event_shape": type(event).__name__}


def _chat_model_kwargs(context: RunContext, http_client: httpx.Client) -> dict[str, Any]:
    """Build the narrowly configured Chat Completions client arguments."""

    model_kwargs: dict[str, Any] = {
        "model": context.model.model_name,
        "base_url": context.model.endpoint,
        "api_key": context.model.api_key or "not-required",
        "temperature": context.model.temperature,
        "max_tokens": context.model.max_tokens,
        "timeout": context.model.timeout_seconds,
        "max_retries": 0,
        "http_client": http_client,
        # LangChain can otherwise auto-select the Responses API. The harness
        # accepts only a deterministic Chat Completions request shape.
        "use_responses_api": False,
        # Graph-level ``messages`` streaming must not silently turn the common
        # one-turn Chat Completions request into a provider SSE request. LangGraph
        # still emits a final local messages event from the non-stream response.
        "disable_streaming": True,
        # Keep callback construction local even after the ambient tracing gate.
        "callbacks": [],
    }
    if context.model.seed is not None:
        model_kwargs["seed"] = context.model.seed
    return model_kwargs


def _attach_search_result_hashes(
    tool_calls: Sequence[ToolCall], session: AgenticSearchSession | None
) -> tuple[ToolCall, ...]:
    """Pair actual model tool calls with the gateway outcomes they executed."""

    if session is None:
        return tuple(tool_calls)
    outcomes = iter(session.outcomes)
    paired: list[ToolCall] = []
    for call in tool_calls:
        if call.name != session.gateway.tool_contract.name:
            paired.append(call)
            continue
        outcome = next(outcomes, None)
        paired.append(
            replace(call, result_hash=outcome.response_sha256 if outcome is not None else None)
        )
    return tuple(paired)


def _answer_only_messages(messages: Sequence[Any]) -> list[Any] | None:
    """Message state ending in backfilled tool results, or ``None``.

    Drops trailing assistant messages with no text or tool calls (an empty
    final answer) and then requires the last message to be the tool result
    returned by the executed search.  If the state instead ends in an
    unanswered tool call, an answer-only continuation would re-execute the
    search, so no messages are returned.
    """

    trimmed = list(messages)
    while trimmed:
        last = trimmed[-1]
        last_type = getattr(last, "type", None)
        if (
            last_type == "ai"
            and not getattr(last, "content", "")
            and not getattr(last, "tool_calls", None)
        ):
            trimmed.pop()
            continue
        break
    if not trimmed or getattr(trimmed[-1], "type", None) != "tool":
        return None
    return trimmed


def _graph_state_messages(app: Any, run_config: dict[str, Any]) -> list[Any]:
    """Return the checkpointed message state of a run, or an empty list."""

    try:
        snapshot = app.get_state(run_config)
    except Exception:
        return []
    values = getattr(snapshot, "values", None) or {}
    messages = values.get("messages")
    return list(messages) if isinstance(messages, list | tuple) else []


class _LangGraphNativeTransport:
    """One StateGraph/model-node invocation with an in-memory checkpointer."""

    def complete(
        self,
        task: TaskInput,
        messages: Sequence[dict[str, str]],
        context: RunContext,
    ) -> TransportResponse:
        return self._complete(task, messages, context, search_session=None)

    def complete_agentic(
        self,
        task: TaskInput,
        messages: Sequence[dict[str, str]],
        context: RunContext,
        search_session: AgenticSearchSession,
    ) -> TransportResponse:
        """Run a native StateGraph tool loop bound to one gateway session."""

        return self._complete(task, messages, context, search_session=search_session)

    def _answer_only_recovery(
        self,
        *,
        task: TaskInput,
        context: RunContext,
        search_session: AgenticSearchSession,
        recorder: RecordingSyncTransport,
        model_kwargs: dict[str, Any],
        state_messages: Sequence[Any],
        emitted_tool_calls: Sequence[ToolCall],
        stream_summaries: Sequence[dict[str, Any]],
    ) -> TransportResponse | None:
        """Retry only the final answer turn after tool results are backfilled.

        A tool-free model-only graph is invoked over the checkpointed message
        state, so the already executed search can never be repeated.  Attempts
        are bounded by the configured model retries and the remaining shared
        model-call budget; every request still flows through the recording
        transport and the session budget counter.
        """

        messages = _answer_only_messages(state_messages)
        if messages is None:
            return None
        remaining = search_session.budget.max_model_calls - search_session.budget.model_calls
        attempts = min(context.model.retries, max(remaining, 0))
        if attempts <= 0:
            return None
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END, START, MessagesState, StateGraph

        final_llm = ChatOpenAI(**model_kwargs)

        for attempt in range(1, attempts + 1):
            if attempt > 1:
                time.sleep(_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
            context.emit(
                "model_retry",
                model_call_index=attempt,
                effective_input_hash=task.input_hash,
                control_flow_node="langgraph_agentic_tool_loop",
                attributes={
                    "reason": "transport_unavailable",
                    "answer_only": True,
                    "attempt": attempt,
                    "max_attempts": attempts,
                },
            )

            def final_model_node(state: MessagesState) -> dict[str, list[Any]]:
                return {"messages": [final_llm.invoke(state["messages"])]}

            workflow = StateGraph(MessagesState)
            workflow.add_node("final_model", final_model_node)
            workflow.add_edge(START, "final_model")
            workflow.add_edge("final_model", END)
            final_app = workflow.compile()
            try:
                final_state = final_app.invoke({"messages": messages})
            except Exception:
                continue
            response = final_state["messages"][-1]
            raw_text = _text_content(getattr(response, "content", ""))
            if not raw_text.strip():
                continue
            usage = getattr(response, "usage_metadata", None) or {}
            if not isinstance(usage, dict):
                usage = {"value": str(usage)}
            return TransportResponse(
                raw_text=raw_text,
                usage=usage,
                status_code=200,
                tool_calls=_attach_search_result_hashes(emitted_tool_calls, search_session),
                network_events=recorder.network_events,
                observed_messages=recorder.first_model_messages() or (),
                framework_events=tuple(stream_summaries),
                model_call_count=sum(
                    1 for event in recorder.network_events if event.is_model_endpoint
                ),
                request_parameters={
                    **recorder.first_model_request_parameters(),
                    "stream_modes": ["updates", "messages", "tasks", "debug"],
                    "provider_client_retries": 0,
                    "answer_only_recovery": True,
                },
            )
        return None

    def _complete(
        self,
        task: TaskInput,
        messages: Sequence[dict[str, str]],
        context: RunContext,
        *,
        search_session: AgenticSearchSession | None,
    ) -> TransportResponse:
        try:
            _ensure_no_ambient_langsmith_tracing()
        except AmbientLangSmithTracingError as exc:
            # Do not construct an HTTP client or graph after the egress gate
            # rejects the process environment. This is deliberately a zero-call
            # result, which the strict policy marks non-comparable.
            return TransportResponse(
                raw_text=None,
                status_code=None,
                error=type(exc).__name__,
                network_events=(),
                observed_messages=tuple(messages),
                framework_events=(
                    {
                        "stream_mode": "preflight",
                        "event_shape": "egress_guard",
                        "error_type": type(exc).__name__,
                    },
                ),
                model_call_count=0,
                request_parameters={"egress_guard": "ambient_langsmith_tracing_rejected"},
            )
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_core.tools import tool
            from langchain_openai import ChatOpenAI
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.graph import END, START, MessagesState, StateGraph
            from langgraph.prebuilt import ToolNode
        except ImportError as exc:
            raise RuntimeError(
                "LangGraph dependencies are unavailable; "
                "install the project with its runtime dependencies."
            ) from exc

        recorder = RecordingSyncTransport(
            context,
            on_model_request=(
                search_session.record_model_call if search_session is not None else None
            ),
        )
        http_client = httpx.Client(
            transport=recorder,
            timeout=_configured_timeout(context.model.timeout_seconds),
        )
        model_kwargs = _chat_model_kwargs(context, http_client)
        llm = ChatOpenAI(**model_kwargs)
        contract = search_session.gateway.tool_contract if search_session is not None else None
        if search_session is not None:

            @tool(
                contract.name,
                description=contract.tool_json_schema["function"]["description"],
            )
            def search_web(query: str) -> list[dict[str, str | None]]:
                """Search the cutoff-bounded Volc source set."""

                return search_session.search_web(query)

            # ``bind_tools`` is the LangGraph/LangChain native tool registry;
            # the model emits a tool call that ToolNode executes in-process.
            llm = llm.bind_tools([search_web], strict=True, parallel_tool_calls=False)
        response_holder: dict[str, Any] = {}
        emitted_tool_calls: list[ToolCall] = []

        def model_node(state: MessagesState) -> dict[str, list[Any]]:
            response = llm.invoke(state["messages"])
            response_holder["message"] = response
            for call in getattr(response, "tool_calls", None) or []:
                if not isinstance(call, dict):
                    continue
                emitted_tool_calls.append(
                    ToolCall(
                        name=str(call.get("name", "unknown")),
                        arguments_hash=sha256_json(call.get("args", {})),
                    )
                )
            return {"messages": [response]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("model", model_node)
        workflow.add_edge(START, "model")
        if search_session is None:
            workflow.add_edge("model", END)
        else:
            tool_node = ToolNode([search_web], handle_tool_errors=False)

            def route_after_model(state: dict[str, Any]) -> str:
                latest = state["messages"][-1]
                return "tools" if getattr(latest, "tool_calls", None) else END

            workflow.add_node("tools", tool_node)
            workflow.add_conditional_edges(
                "model",
                route_after_model,
                {"tools": "tools", END: END},
            )
            workflow.add_edge("tools", "model")
        # A new graph plus a new checkpointer/thread is built for every task.
        app = workflow.compile(checkpointer=InMemorySaver())
        run_config = {"configurable": {"thread_id": f"{context.run_id}:{task.task_id}"}}
        initial_messages: list[Any] = [HumanMessage(content=task.prompt)]
        if contract is not None:
            initial_messages.insert(0, SystemMessage(content=contract.tool_policy_text))
        initial_state = {"messages": initial_messages}

        stream_summaries: list[dict[str, Any]] = []
        recovered: TransportResponse | None = None
        try:
            try:
                stream = app.stream(
                    initial_state,
                    config=run_config,
                    stream_mode=["updates", "messages", "tasks", "debug"],
                    version="v2",
                )
            except TypeError:
                # The pinned version supports v2. This fallback keeps the adapter
                # diagnosable if a provider constrains the installed API surface.
                stream = app.stream(
                    initial_state,
                    config=run_config,
                    stream_mode=["updates", "messages", "tasks", "debug"],
                )
            for event in stream:
                stream_summaries.append(_stream_event_summary(event))
            if search_session is not None and emitted_tool_calls:
                raw_text = _text_content(getattr(response_holder.get("message"), "content", ""))
                if not raw_text.strip():
                    recovered = self._answer_only_recovery(
                        task=task,
                        context=context,
                        search_session=search_session,
                        recorder=recorder,
                        model_kwargs=model_kwargs,
                        state_messages=_graph_state_messages(app, run_config),
                        emitted_tool_calls=emitted_tool_calls,
                        stream_summaries=stream_summaries,
                    )
                    if recovered is not None:
                        return recovered
        except Exception as exc:
            if (
                search_session is not None
                and emitted_tool_calls
                and transient_transport_failure(exc)
            ):
                recovered = self._answer_only_recovery(
                    task=task,
                    context=context,
                    search_session=search_session,
                    recorder=recorder,
                    model_kwargs=model_kwargs,
                    state_messages=_graph_state_messages(app, run_config),
                    emitted_tool_calls=emitted_tool_calls,
                    stream_summaries=stream_summaries,
                )
                if recovered is not None:
                    return recovered
            wire_messages = recorder.first_model_messages()
            return TransportResponse(
                raw_text=None,
                status_code=None,
                error=type(exc).__name__,
                tool_calls=_attach_search_result_hashes(emitted_tool_calls, search_session),
                network_events=recorder.network_events,
                observed_messages=wire_messages if wire_messages is not None else (),
                framework_events=tuple(stream_summaries),
                model_call_count=sum(
                    1 for event in recorder.network_events if event.is_model_endpoint
                ),
                request_parameters={
                    **recorder.first_model_request_parameters(),
                    "stream_modes": ["updates", "messages", "tasks", "debug"],
                    "provider_client_retries": 0,
                },
            )
        finally:
            http_client.close()

        response = response_holder.get("message")
        if response is None:
            raise RuntimeError("LangGraph stream completed without a model response.")
        usage = getattr(response, "usage_metadata", None) or {}
        if not isinstance(usage, dict):
            usage = {"value": str(usage)}
        return TransportResponse(
            raw_text=_text_content(getattr(response, "content", "")),
            usage=usage,
            status_code=200,
            tool_calls=_attach_search_result_hashes(emitted_tool_calls, search_session),
            network_events=recorder.network_events,
            observed_messages=recorder.first_model_messages() or (),
            framework_events=tuple(stream_summaries),
            model_call_count=sum(1 for event in recorder.network_events if event.is_model_endpoint),
            request_parameters={
                **recorder.first_model_request_parameters(),
                "stream_modes": ["updates", "messages", "tasks", "debug"],
                "provider_client_retries": 0,
            },
        )


class LangGraphAdapter:
    """LangGraph baseline with local multi-mode stream evidence."""

    name = "langgraph"
    adapter_version = (
        f"langgraph=={package_version('langgraph')};"
        f"langchain-openai=={package_version('langchain-openai')}"
    )

    def __init__(self, transport: StrictTransport | None = None) -> None:
        self.transport = transport or _LangGraphNativeTransport()

    def run(self, task: TaskInput, context: RunContext) -> AdapterResult:
        return result_from_transport(
            task=task,
            context=context,
            transport=self.transport,
            control_flow_node="langgraph_model",
        )

    def run_agentic(
        self,
        task: TaskInput,
        context: RunContext,
        search_session: AgenticSearchSession,
    ) -> AdapterResult:
        """Run LangGraph's native model-driven ``search_web`` tool loop."""

        if not isinstance(self.transport, _LangGraphNativeTransport):
            raise RuntimeError("agentic_web_search requires the native LangGraph transport")
        transport = _LangGraphAgenticTransport(self.transport, search_session)
        return result_from_transport(
            task=task,
            context=context,
            transport=transport,
            control_flow_node="langgraph_agentic_tool_loop",
        )


class _LangGraphAgenticTransport:
    """Bind an otherwise record-free native graph call to one search session."""

    def __init__(
        self, transport: _LangGraphNativeTransport, search_session: AgenticSearchSession
    ) -> None:
        self._transport = transport
        self._search_session = search_session

    def complete(
        self,
        task: TaskInput,
        messages: Sequence[dict[str, str]],
        context: RunContext,
    ) -> TransportResponse:
        return self._transport.complete_agentic(task, messages, context, self._search_session)
