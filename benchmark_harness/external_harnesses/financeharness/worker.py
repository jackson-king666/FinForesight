"""FinanceHarness child shim.

Only this module imports the pinned upstream package.  It exposes a tiny
stdio-backed client and a registry containing the explicitly approved tools;
the upstream CLI, HTTP service, ``run_research`` helper, data tools, and skills
are deliberately unreachable from the benchmark path.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from types import SimpleNamespace
from typing import Any

from benchmark_harness.external_harnesses.ipc import (
    IPC_PROTOCOL,
    IpcChannel,
    IpcProtocolError,
)
from benchmark_harness.search_service import SearchServiceClient
from benchmark_harness.search_service.client import _descriptor_from_payload


class ChildBoundaryViolation(RuntimeError):
    """Raised when the child attempts a forbidden egress or process action."""


def _install_egress_guard(base_url: str) -> None:
    from urllib.parse import urlsplit

    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None:
        raise ChildBoundaryViolation("service_loopback_binding_invalid")
    allowed_port = parsed.port
    blocked = {
        "socket.accept",
        "socket.bind",
        "socket.listen",
        "socket.sendto",
        "socket.sendmsg",
        "subprocess.Popen",
        "os.system",
        "os.exec",
        "os.execve",
        "os.execv",
        "os.execvp",
        "os.fork",
        "os.posix_spawn",
    }

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event in blocked or event.startswith("os.exec"):
            raise ChildBoundaryViolation(f"child_egress_or_process_forbidden:{event}")
        if event == "socket.getaddrinfo":
            host = args[0] if args else None
            port = args[1] if len(args) > 1 else None
            if host != "127.0.0.1" or int(port or 0) != allowed_port:
                raise ChildBoundaryViolation("child_dns_or_non_loopback_forbidden")
        if event == "socket.connect":
            address = args[1] if len(args) > 1 else None
            if not isinstance(address, tuple) or len(address) < 2:
                raise ChildBoundaryViolation("child_socket_address_invalid")
            if address[0] != "127.0.0.1" or int(address[1]) != allowed_port:
                raise ChildBoundaryViolation("child_socket_target_forbidden")

    sys.addaudithook(audit)


class _IpcCompletionClient:
    """AsyncOpenAI-shaped object consumed by FinanceHarness providers."""

    def __init__(self, channel: IpcChannel) -> None:
        self.channel = channel
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self._counter = 0
        self.last_usage: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self._counter += 1
        request_id = f"model-{self._counter}"
        self.channel.send("model.request", _json_safe(kwargs), request_id=request_id)
        while True:
            envelope = self.channel.read()
            if envelope.type != "model.response" or envelope.request_id != request_id:
                raise IpcProtocolError("unexpected IPC frame while awaiting model response")
            response = envelope.payload
            self.last_usage = (
                response.get("usage") if isinstance(response.get("usage"), dict) else {}
            )
            return _completion_from_payload(response)


def _completion_from_payload(payload: dict[str, Any]) -> Any:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        choices = []
    normalized_choices: list[Any] = []
    for raw_choice in choices:
        if not isinstance(raw_choice, dict):
            continue
        raw_message = raw_choice.get("message")
        if not isinstance(raw_message, dict):
            raw_message = {}
        calls: list[Any] = []
        raw_calls = raw_message.get("tool_calls")
        if isinstance(raw_calls, list):
            for index, raw_call in enumerate(raw_calls):
                if not isinstance(raw_call, dict):
                    continue
                function = raw_call.get("function")
                if not isinstance(function, dict):
                    function = {}
                calls.append(
                    SimpleNamespace(
                        id=str(raw_call.get("id") or f"ipc-call-{index}"),
                        function=SimpleNamespace(
                            name=str(function.get("name") or ""),
                            arguments=str(function.get("arguments") or "{}"),
                        ),
                    )
                )
        normalized_choices.append(
            SimpleNamespace(
                message=SimpleNamespace(
                    content=raw_message.get("content") or "",
                    tool_calls=calls or None,
                ),
                finish_reason=str(raw_choice.get("finish_reason") or "stop"),
            )
        )
    return SimpleNamespace(choices=normalized_choices)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _make_registry(
    channel: IpcChannel,
    *,
    profile: Any,
    config: Any,
    service_session: Any,
    closed_book: bool = False,
) -> tuple[Any, Any]:
    """Build only search/visit/citations/plan/calc/pure-compute tools."""

    from financeharness.runtime.tool_registry import ToolError, ToolRegistry, ToolResponse, ToolSpec
    from financeharness.tools.compute import COMPUTE_SPECS
    from financeharness.tools.compute.arithmetic import SPEC as CALC_SPEC
    from financeharness.tools.core.plan import PLAN_SPEC
    from financeharness.tools.research.cache import FetchCache
    from financeharness.tools.research.citations import build_citations_spec
    from financeharness.tools.research.visit import build_visit_spec
    from pydantic import BaseModel, Field

    cache = FetchCache()
    client = _IpcCompletionClient(channel)

    class SearchRequest(BaseModel):
        queries: list[str] = Field(..., min_length=1)
        max_results: int | None = Field(None, ge=1, le=20)

    async def search_handler(request: SearchRequest) -> Any:
        max_results = request.max_results or 8
        all_results: list[dict[str, Any]] = []
        for query in request.queries:
            if not isinstance(query, str) or not query.strip():
                raise ToolError("search query must be non-empty")
            try:
                outcome = service_session.search(query)
            except Exception as exc:
                raise ToolError("search_service_request_failed") from exc
            for candidate in outcome.candidates:
                candidate = candidate.tool_payload()
                url = candidate.get("url")
                if not isinstance(url, str) or not url:
                    continue
                title = candidate.get("title")
                content = candidate.get("content")
                if isinstance(title, str):
                    cache.set_title(url, title)
                if isinstance(content, str) and content:
                    # Search content is already accepted and cutoff-filtered
                    # by the parent; this is the only cache warming path.
                    cache.set_content(url, content)
                all_results.append(
                    {
                        "url": url,
                        "title": title if isinstance(title, str) else None,
                        "snippet": content if isinstance(content, str) else None,
                        "query": query,
                    }
                )
        if not all_results:
            raise ToolError("parent Volc search returned no accepted candidates")
        rendered = "Found {} result(s):\n\n{}".format(
            len(all_results),
            "\n".join(
                f"{index}. **{row['title'] or row['url']}** — {row['url']}"
                for index, row in enumerate(
                    all_results[: max_results * len(request.queries)], start=1
                )
            ),
        )
        return ToolResponse(
            markdown=rendered,
            structured={"results": all_results, "count": len(all_results)},
            meta={"backend": "parent_volc"},
        )

    search_spec = ToolSpec(
        name="search",
        display_name="search",
        tier="core",
        description=(
            "Search parent-owned Volc sources under the task-bound retrieval "
            "policy; never access the network from the external worker."
        ),
        request_schema=SearchRequest,
        handler=search_handler,
        tags=("web", "parent-owned"),
    )

    async def forbidden_fetcher(_url: str) -> Any:
        raise ToolError("uncached_visit_forbidden")

    registry = ToolRegistry()
    if not closed_book:
        registry.register(search_spec)
        registry.register(
            build_visit_spec(
                cache,
                profile,
                fetcher=forbidden_fetcher,
                client=client,
                config=config,
            )
        )
        registry.register(build_citations_spec(cache))
    registry.register(PLAN_SPEC)
    registry.register(CALC_SPEC)
    for spec in COMPUTE_SPECS:
        # Pure compute tools are deferred, so loading one cannot expose data
        # tools or any upstream network helper.
        registry.register(spec)
    return registry, cache, client


async def _run(source: str, task_id: str) -> None:
    sys.path.insert(0, source)
    from financeharness.providers.profiles import Generation, ModelProfile
    from financeharness.runtime.agent import Agent
    from financeharness.runtime.config import load_runtime_config

    channel = IpcChannel(sys.stdin.buffer, sys.stdout.buffer, task_id=task_id)
    channel.send(
        "hello",
        {
            "protocol": IPC_PROTOCOL,
            "adapter": "financeharness",
            "adapter_version": "v2",
            "child_network": "forbidden",
        },
        request_id="hello",
    )
    start = channel.read()
    if start.type != "task.start":
        raise IpcProtocolError("task.start required after hello")
    payload = start.payload
    prompt = payload.get("prompt")
    cutoff_date = payload.get("cutoff_date")
    retrieval_regime = payload.get("retrieval_regime", "cutoff_controlled")
    if retrieval_regime not in {
        "closed_book",
        "cutoff_controlled",
        "unrestricted_current_web",
    }:
        raise IpcProtocolError("task.start retrieval_regime invalid")
    if not isinstance(prompt, str) or not prompt:
        raise IpcProtocolError("task.start prompt missing")
    if retrieval_regime == "cutoff_controlled" and (
        not isinstance(cutoff_date, str) or not cutoff_date
    ):
        raise IpcProtocolError("task.start cutoff_date missing")
    if payload.get("today") != cutoff_date:
        raise IpcProtocolError("FinanceHarness today must equal cutoff_date")
    service_payload = payload.get("search_service")
    if retrieval_regime == "closed_book" and service_payload is not None:
        raise IpcProtocolError("closed_book search capability must be absent")
    if retrieval_regime != "closed_book" and not isinstance(service_payload, dict):
        raise IpcProtocolError("task.start search_service capability missing")
    service_session = None
    if retrieval_regime != "closed_book":
        assert isinstance(service_payload, dict)
        base_url = service_payload.get("base_url")
        if not isinstance(base_url, str):
            raise IpcProtocolError("task.start search_service base_url missing")
        _install_egress_guard(base_url)
        service_descriptor = _descriptor_from_payload({"session": service_payload})
        service_client = SearchServiceClient(base_url=base_url)
        # The child receives an already-open, task-scoped bearer; no admin token or
        # session creation endpoint is reachable from this process.
        from benchmark_harness.search_service.client import SearchServiceSession

        service_session = SearchServiceSession(service_client, service_descriptor)

    runtime_config = load_runtime_config(
        overrides={
            "generation": {"max_tokens": 4096},
            "limits": {
                "max_rounds": 24,
                "max_transient_retries": 0,
                "max_regenerate_retries": 0,
                "max_truncation_escalations": 0,
                "per_call_timeout_s": 1800,
                "run_wall_clock_s": 1800,
            },
        }
    )
    profile = ModelProfile(
        name="parent-broker",
        model="parent-broker-model",
        base_url="http://ipc.invalid/v1",
        api_key_literal="EMPTY",
        provider="chat",
        generation=Generation(max_tokens=4096),
        timeout_s=1800,
    )
    registry, cache, client = _make_registry(
        channel,
        profile=profile,
        config=runtime_config,
        service_session=service_session,
        closed_book=retrieval_regime == "closed_book",
    )
    events: list[dict[str, Any]] = []
    emitted_tool_calls = 0
    emitted_rounds = 0

    def emit(kind: str, details: dict[str, Any]) -> None:
        nonlocal emitted_tool_calls, emitted_rounds
        if kind == "round_start":
            emitted_rounds += 1
            if emitted_rounds > 24:
                raise ChildBoundaryViolation("deep_round_budget_exceeded")
        if kind == "tool_call":
            emitted_tool_calls += 1
            if emitted_tool_calls > 40:
                raise ChildBoundaryViolation("deep_tool_call_budget_exceeded")
        row = {
            "event_type": kind,
            "task_id": task_id,
            "network_call_count": 0,
            "attributes": {
                key: value
                for key, value in _json_safe(details).items()
                if key not in {"content", "result", "text", "args"}
            },
        }
        events.append(row)
        channel.send("audit.event", row, request_id=f"audit-{len(events)}")

    channel.send(
        "audit.event",
        {"event_type": "task_started", "task_id": task_id, "network_call_count": 0},
        request_id="audit-task-started",
    )
    try:
        agent = Agent(
            profile=profile,
            registry=registry,
            config=runtime_config,
            client=client,
            system_prompt=(
                (
                    "This benchmark task is closed-book. Use only the registered "
                    "non-network tools; do not request search, credentials, direct "
                    "web access, skills, or clarification."
                )
                if retrieval_regime == "closed_book"
                else (
                    "This benchmark task is cutoff-bound. Today is "
                    f"{cutoff_date}. Use only the registered parent-backed tools; "
                    "do not infer or request credentials, direct web access, skills, "
                    "or clarification."
                )
                if retrieval_regime == "cutoff_controlled"
                else (
                    "This benchmark task is an unrestricted current-web leakage probe. "
                    f"The declared task date is {cutoff_date}. Use only the registered "
                    "parent-backed current-web tool; do not infer or request credentials, "
                    "direct web access, skills, or clarification."
                )
            ),
            skill_registry=None,
            grounding_review=bool(payload.get("grounding_review", True)),
            prompt_variant="research",
            finalize=lambda text: (
                text,
                {"citations": [citation.url for citation in cache.citations]},
            ),
        )
        trajectory = await agent.run(prompt, on_event=emit)
        citations = [citation.url for citation in cache.citations]
        channel.send(
            "audit.event",
            {"event_type": "task_finished", "task_id": task_id, "network_call_count": 0},
            request_id="audit-task-finished",
        )
        channel.send(
            "task.result",
            {
                "prediction": trajectory.get("prediction", ""),
                "termination": trajectory.get("termination"),
                "rounds": trajectory.get("rounds"),
                "messages": trajectory.get("messages", []),
                "tool_log": trajectory.get("tool_log", []),
                "citations": citations,
                "grounding_reviewed": trajectory.get("grounding_reviewed", False),
                "usage": getattr(client, "last_usage", {}),
                "service_receipts": [
                    receipt.as_dict()
                    for receipt in (service_session.receipts if service_session else ())
                ],
                "child_search_service_call_count": (
                    len(service_session.receipts) if service_session else 0
                ),
                "child_forbidden_network_call_count": 0,
            },
            request_id="task-result",
        )
    except BaseException as exc:
        channel.send(
            "task.error",
            {
                "error": f"{type(exc).__name__}: {exc}",
                "child_network_call_count": 0,
                "service_receipts": [
                    receipt.as_dict()
                    for receipt in (service_session.receipts if service_session else ())
                ],
                "child_search_service_call_count": (
                    len(service_session.receipts) if service_session else 0
                ),
            },
            request_id="task-error",
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Restricted FinanceHarness IPC worker")
    parser.add_argument("--source", required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args(argv)
    try:
        asyncio.run(_run(args.source, args.task_id))
    except BaseException as exc:
        print(f"financeharness-worker: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
