"""PydanticAI adapter for the strict one-turn protocol."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import httpx
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles.openai import OpenAIModelProfile

from benchmark_harness.adapters.base import (
    _RETRY_BACKOFF_SECONDS,
    package_version,
    result_from_transport,
    transient_transport_failure,
)
from benchmark_harness.adapters.recording import RecordingAsyncTransport
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


def _configured_timeout(timeout_seconds: float) -> httpx.Timeout:
    """Return one explicit timeout budget for every HTTPX operation."""

    return httpx.Timeout(timeout_seconds)


def _build_uninstrumented_agent(
    agent_type: type[Any],
    model: Any,
    *,
    instructions: str | Sequence[str] = (),
) -> Any:
    """Create an agent without PydanticAI OpenTelemetry instrumentation."""

    kwargs: dict[str, Any] = {
        "retries": 0,
        "system_prompt": (),
        "tools": (),
        "toolsets": (),
    }
    if instructions:
        kwargs["instructions"] = instructions
    agent = agent_type(model, **kwargs)
    # PydanticAI can be instrumented globally. Set the per-agent value so an
    # ambient Agent.instrument_all(True) cannot export run telemetry.
    agent.instrument = False
    return agent


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(item) for key, item in vars(value).items() if not key.startswith("_")
        }
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _extract_capture(
    run_messages: Sequence[Any],
) -> tuple[list[dict[str, str]], int, list[dict[str, Any]], list[ToolCall]]:
    """Recover the first actual PydanticAI model request for the fairness gate."""

    observed: list[dict[str, str]] = []
    requests = 0
    events: list[dict[str, Any]] = []
    tool_calls: list[ToolCall] = []
    for message in run_messages:
        parts = getattr(message, "parts", None)
        if not isinstance(parts, list | tuple):
            continue
        message_kind = getattr(message, "kind", type(message).__name__)
        message_type = type(message).__name__
        events.append(
            {
                "message_kind": str(message_kind),
                "message_type": message_type,
                "part_count": len(parts),
            }
        )
        is_model_request = message_type == "ModelRequest" or str(message_kind).lower() in {
            "request",
            "model_request",
        }
        for part in parts:
            part_kind = str(getattr(part, "part_kind", type(part).__name__))
            if part_kind == "tool-call":
                tool_calls.append(
                    ToolCall(
                        name=str(getattr(part, "tool_name", "unknown")),
                        arguments_hash=sha256_json(_jsonable(getattr(part, "args", {}))),
                    )
                )
        if not is_model_request:
            continue
        requests += 1
        if requests != 1:
            continue
        for part in parts:
            part_kind = str(getattr(part, "part_kind", type(part).__name__))
            if part_kind == "user-prompt":
                content = getattr(part, "content", "")
                observed.append(
                    {
                        "role": "user",
                        "content": content if isinstance(content, str) else str(content),
                    }
                )
            elif part_kind == "system-prompt":
                content = getattr(part, "content", "")
                observed.append(
                    {
                        "role": "system",
                        "content": content if isinstance(content, str) else str(content),
                    }
                )
            elif part_kind == "tool-return":
                content = getattr(part, "content", "")
                observed.append(
                    {
                        "role": "tool",
                        "content": content if isinstance(content, str) else str(content),
                    }
                )
            else:
                observed.append(
                    {"role": f"unexpected:{part_kind}", "content": str(_jsonable(part))}
                )
    return observed, requests, events, tool_calls


def _answer_only_history(captured: Sequence[Any]) -> list[Any] | None:
    """Resumable history ending in backfilled tool results, or ``None``.

    Drops trailing text-only model responses (an empty final answer) and then
    requires the last message to be the model request carrying the executed
    search results.  If the history instead ends in an unanswered tool call,
    resuming would re-execute the search, so no history is returned.
    """

    history = list(captured)
    while (
        history
        and isinstance(history[-1], ModelResponse)
        and not any(isinstance(part, ToolCallPart) for part in history[-1].parts)
    ):
        history.pop()
    if not history or not isinstance(history[-1], ModelRequest):
        return None
    if not any(isinstance(part, ToolReturnPart) for part in history[-1].parts):
        return None
    return history


def _attach_search_result_hashes(
    tool_calls: Sequence[ToolCall], session: AgenticSearchSession | None
) -> tuple[ToolCall, ...]:
    """Pair actual model tool-call parts with the gateway outcomes they caused."""

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


class _PydanticAIHardToolCapModel(WrapperModel):
    """Keep provider tool-call bursts out of the framework's executable loop."""

    def __init__(self, wrapped: Any, *, max_tool_calls: int) -> None:
        super().__init__(wrapped)
        self.max_tool_calls = max_tool_calls
        self.emitted_tool_calls = 0

    async def request(
        self, messages: Any, model_settings: Any, model_request_parameters: Any
    ) -> Any:
        settings = dict(model_settings or {})
        request_parameters = model_request_parameters
        if self.emitted_tool_calls >= self.max_tool_calls:
            # Some OpenAI-compatible endpoints ignore ``tool_choice: none``.
            # Removing only the native function definition leaves the same
            # PydanticAI message history (including tool returns) intact while
            # making the final framework model call answer-only.
            request_parameters = replace(request_parameters, function_tools=[])
        response = await self.wrapped.request(messages, settings, request_parameters)
        remaining = self.max_tool_calls - self.emitted_tool_calls
        allowed_in_response = min(remaining, 1)
        tool_parts = [part for part in response.parts if isinstance(part, ToolCallPart)]
        if len(tool_parts) > allowed_in_response:
            kept = 0
            parts = []
            for part in response.parts:
                if isinstance(part, ToolCallPart):
                    if kept >= allowed_in_response:
                        continue
                    kept += 1
                parts.append(part)
            response = replace(response, parts=parts)
            tool_parts = [part for part in response.parts if isinstance(part, ToolCallPart)]
        self.emitted_tool_calls += len(tool_parts)
        return response


class _PydanticAINativeTransport:
    """Native SDK path. It has no hosted tracing dependency."""

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
        """Run PydanticAI's native tool loop bound to one gateway session."""

        return self._complete(task, messages, context, search_session=search_session)

    def _answer_only_recovery(
        self,
        *,
        task: TaskInput,
        context: RunContext,
        search_session: AgenticSearchSession,
        captured: Sequence[Any],
        recorder: RecordingAsyncTransport,
        provider: Any,
        model_settings: dict[str, Any],
    ) -> TransportResponse | None:
        """Retry only the final answer turn after tool results are backfilled.

        The captured history already contains the executed search tool call and
        its return part, so resuming it with a tool-free agent can never repeat
        the search.  Attempts are bounded by the configured model retries and
        the remaining shared model-call budget; every request still flows
        through the recording transport and the session budget counter.
        """

        history = _answer_only_history(captured)
        if history is None:
            return None
        remaining = search_session.budget.max_model_calls - search_session.budget.model_calls
        attempts = min(context.model.retries, max(remaining, 0))
        if attempts <= 0:
            return None
        from pydantic_ai import Agent, UsageLimits
        from pydantic_ai.models.openai import OpenAIChatModel

        model = OpenAIChatModel(
            context.model.model_name,
            provider=provider,
            profile=OpenAIModelProfile(openai_chat_send_back_thinking_parts=False),
        )
        answer_agent = _build_uninstrumented_agent(Agent, model)
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                time.sleep(_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
            context.emit(
                "model_retry",
                model_call_index=attempt,
                effective_input_hash=task.input_hash,
                control_flow_node="pydantic_ai_agentic_tool_loop",
                attributes={
                    "reason": "transport_unavailable",
                    "answer_only": True,
                    "attempt": attempt,
                    "max_attempts": attempts,
                },
            )
            try:
                recovered = answer_agent.run_sync(
                    None,
                    message_history=list(history),
                    model_settings=model_settings,
                    usage_limits=UsageLimits(request_limit=1),
                )
            except Exception:
                continue
            raw_text = recovered.output
            if not isinstance(raw_text, str) or not raw_text.strip():
                continue
            usage_raw = recovered.usage
            usage = _jsonable(usage_raw)
            if not isinstance(usage, dict):
                usage = {"value": usage}
            observed, _, framework_events, tool_calls = _extract_capture(captured)
            wire_messages = recorder.first_model_messages()
            return TransportResponse(
                raw_text=raw_text,
                usage=usage,
                status_code=200,
                tool_calls=_attach_search_result_hashes(tool_calls, search_session),
                network_events=recorder.network_events,
                observed_messages=wire_messages if wire_messages is not None else tuple(observed),
                framework_events=tuple(framework_events),
                model_call_count=sum(
                    1 for event in recorder.network_events if event.is_model_endpoint
                ),
                request_parameters={
                    **recorder.first_model_request_parameters(),
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
        del messages
        try:
            from openai import AsyncOpenAI
            from pydantic_ai import Agent, UsageLimits, capture_run_messages
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
        except ImportError as exc:
            raise RuntimeError(
                "PydanticAI dependencies are unavailable; "
                "install the project with its runtime dependencies."
            ) from exc

        recorder = RecordingAsyncTransport(
            context,
            on_model_request=(
                search_session.record_model_call if search_session is not None else None
            ),
        )
        client = AsyncOpenAI(
            base_url=context.model.endpoint,
            api_key=context.model.api_key or "not-required",
            max_retries=0,
            timeout=context.model.timeout_seconds,
            http_client=httpx.AsyncClient(
                transport=recorder,
                timeout=_configured_timeout(context.model.timeout_seconds),
            ),
        )
        provider = OpenAIProvider(openai_client=client)
        # Explicitly use PydanticAI's Chat Completions model rather than its
        # deprecated auto-named alias, matching the LangGraph transport.
        model = OpenAIChatModel(
            context.model.model_name,
            provider=provider,
            profile=(
                OpenAIModelProfile(openai_chat_send_back_thinking_parts=False)
                if search_session is not None
                else None
            ),
        )
        # Strict uses no system prompt, tools, memory, or retries.  Agentic
        # uses PydanticAI's native tool registry, but still has a fresh agent
        # for every task and no hosted tracing.
        contract = search_session.gateway.tool_contract if search_session is not None else None
        if contract is not None:
            model = _PydanticAIHardToolCapModel(model, max_tool_calls=contract.max_tool_calls)
        agent = _build_uninstrumented_agent(
            Agent,
            model,
            instructions=contract.tool_policy_text if contract is not None else (),
        )
        if search_session is not None:

            @agent.tool_plain(
                name=contract.name,
                description=contract.tool_json_schema["function"]["description"],
                strict=True,
            )
            def search_web(query: str) -> list[dict[str, str | None]]:
                """Search the cutoff-bounded Volc source set."""

                return search_session.search_web(query)

        model_settings: dict[str, Any] = {
            "temperature": context.model.temperature,
            "max_tokens": context.model.max_tokens,
        }
        if contract is not None:
            # PydanticAI's usage limit protects execution, but a provider can
            # otherwise emit a four-call batch before that guard gets control.
            # Disable parallel native calls at the request boundary so the
            # configured shared budget is enforceable without an overflow
            # tool call appearing in benchmark evidence.
            model_settings["parallel_tool_calls"] = False
        if context.model.seed is not None:
            model_settings["seed"] = context.model.seed
        captured: Sequence[Any] = ()
        recovered: TransportResponse | None = None
        try:
            with capture_run_messages() as captured:
                result = agent.run_sync(
                    task.prompt,
                    model_settings=model_settings,
                    usage_limits=(
                        UsageLimits(
                            request_limit=contract.max_model_calls,
                            tool_calls_limit=contract.max_tool_calls,
                        )
                        if contract is not None
                        else None
                    ),
                )
            if search_session is not None and _answer_only_history(captured) is not None:
                raw_output = result.output
                if not isinstance(raw_output, str) or not raw_output.strip():
                    recovered = self._answer_only_recovery(
                        task=task,
                        context=context,
                        search_session=search_session,
                        captured=captured,
                        recorder=recorder,
                        provider=provider,
                        model_settings=model_settings,
                    )
                    if recovered is not None:
                        return recovered
        except Exception as exc:
            if search_session is not None and transient_transport_failure(exc):
                recovered = self._answer_only_recovery(
                    task=task,
                    context=context,
                    search_session=search_session,
                    captured=captured,
                    recorder=recorder,
                    provider=provider,
                    model_settings=model_settings,
                )
                if recovered is not None:
                    return recovered
            observed, _, framework_events, tool_calls = _extract_capture(captured)
            wire_messages = recorder.first_model_messages()
            return TransportResponse(
                raw_text=None,
                status_code=None,
                tool_calls=_attach_search_result_hashes(tool_calls, search_session),
                error=type(exc).__name__,
                network_events=recorder.network_events,
                observed_messages=wire_messages if wire_messages is not None else tuple(observed),
                framework_events=tuple(framework_events),
                model_call_count=sum(
                    1 for event in recorder.network_events if event.is_model_endpoint
                ),
                request_parameters={
                    **recorder.first_model_request_parameters(),
                    "provider_client_retries": 0,
                },
            )
        finally:
            try:
                asyncio.run(client.close())
            except RuntimeError:
                # The normal CLI path has no active event loop. If an embedding
                # application owns one, closing is deferred to its lifecycle.
                pass

        raw_text = result.output
        usage_raw = result.usage
        usage = _jsonable(usage_raw)
        if not isinstance(usage, dict):
            usage = {"value": usage}
        observed, _, framework_events, tool_calls = _extract_capture(captured)
        wire_messages = recorder.first_model_messages()
        return TransportResponse(
            raw_text=raw_text if isinstance(raw_text, str) else str(raw_text),
            usage=usage,
            status_code=200,
            tool_calls=_attach_search_result_hashes(tool_calls, search_session),
            network_events=recorder.network_events,
            observed_messages=wire_messages if wire_messages is not None else tuple(observed),
            framework_events=tuple(framework_events),
            model_call_count=sum(1 for event in recorder.network_events if event.is_model_endpoint),
            request_parameters={
                **recorder.first_model_request_parameters(),
                "provider_client_retries": 0,
            },
        )


class PydanticAIAdapter:
    """PydanticAI baseline with locally persisted capture-run message evidence."""

    name = "pydantic-ai"
    adapter_version = f"pydantic-ai-slim=={package_version('pydantic-ai-slim')}"

    def __init__(self, transport: StrictTransport | None = None) -> None:
        self.transport = transport or _PydanticAINativeTransport()

    def run(self, task: TaskInput, context: RunContext) -> AdapterResult:
        return result_from_transport(
            task=task,
            context=context,
            transport=self.transport,
            control_flow_node="pydantic_ai_agent",
        )

    def run_agentic(
        self,
        task: TaskInput,
        context: RunContext,
        search_session: AgenticSearchSession,
    ) -> AdapterResult:
        """Run the native model-driven ``search_web(query: str)`` tool loop."""

        if not isinstance(self.transport, _PydanticAINativeTransport):
            raise RuntimeError("agentic_web_search requires the native PydanticAI transport")
        transport = _PydanticAIAgenticTransport(self.transport, search_session)
        return result_from_transport(
            task=task,
            context=context,
            transport=transport,
            control_flow_node="pydantic_ai_agentic_tool_loop",
        )


class _PydanticAIAgenticTransport:
    """Bind an otherwise record-free native adapter call to one search session."""

    def __init__(
        self, transport: _PydanticAINativeTransport, search_session: AgenticSearchSession
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
