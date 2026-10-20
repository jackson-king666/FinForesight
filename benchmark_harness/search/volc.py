"""Volc Torchlight, cutoff-bound web search with framework-neutral audits."""

from __future__ import annotations

import json
import json as json_module
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.contracts import (
    TEMPORAL_FILTER_NETWORK_CLASS,
    NetworkEvent,
    sha256_json,
    sha256_text,
)
from benchmark_harness.search.cutoff import CutoffWindow, CutoffWindowError
from benchmark_harness.search.filter import (
    LlmFilterRequest,
    LlmTemporalFilter,
    LlmTemporalVerdict,
    VolcSearchCandidate,
    filter_volc_candidates,
)
from benchmark_harness.search.volc_core import (
    VOLC_SEARCH_ACTION,
    VOLC_SEARCH_HOST,
    VOLC_SEARCH_PATH,
    VOLC_SEARCH_REGION,
    VOLC_SEARCH_SERVICE,
    VOLC_SEARCH_URL,
    VOLC_SEARCH_VERSION,
    prepare_cutoff_volc_request,
)

_LOGGER = logging.getLogger(__name__)

# This is intentionally a small, framework-neutral contract.  Each framework
# registers a native function with this exact public name and signature; the
# model, rather than the runner, decides the query text.  Keep the text here so
# the policy and its hash cannot drift between the PydanticAI and LangGraph
# integrations.
SEARCH_WEB_TOOL_DESCRIPTION = (
    "Search cutoff-bounded Volc sources for information relevant to the user's task. "
    "Use a concise query; returned sources have already been cutoff-filtered."
)
SEARCH_WEB_TOOL_POLICY = (
    "For this cutoff-bounded research task, call search_web at least once before "
    "answering. Choose the query yourself. search_web is the only permitted web "
    "tool; use only its returned cutoff-filtered sources when citing web evidence."
)
SEARCH_WEB_TOOL_JSON_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": SEARCH_WEB_TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class SearchGatewayError(RuntimeError):
    """Fail-closed error from a search preflight or a tool invocation."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


@dataclass(frozen=True, slots=True)
class ToolContract:
    """The identical public tool budget and request contract for both frameworks."""

    name: str = "search_web"
    version: str = "agentic_web_search/v3"
    provider: str = "volc"
    endpoint: str = VOLC_SEARCH_URL
    max_tool_calls: int = 3
    max_model_calls: int = 4
    request_shape: Mapping[str, Any] = field(
        default_factory=lambda: {
            "action": VOLC_SEARCH_ACTION,
            "version": VOLC_SEARCH_VERSION,
            "search_type": "web",
            "count": 10,
            "filter": {"NeedContent": True, "NeedUrl": True},
            "cutoff_field": "TimeRange",
        }
    )
    tool_json_schema: Mapping[str, Any] = field(default_factory=lambda: SEARCH_WEB_TOOL_JSON_SCHEMA)
    tool_policy_text: str = SEARCH_WEB_TOOL_POLICY

    @property
    def hash(self) -> str:
        return sha256_json(
            {
                "name": self.name,
                "version": self.version,
                "provider": self.provider,
                "endpoint": self.endpoint,
                "max_tool_calls": self.max_tool_calls,
                "max_model_calls": self.max_model_calls,
                "request_shape": dict(self.request_shape),
                "tool_json_schema": dict(self.tool_json_schema),
                "tool_policy_text": self.tool_policy_text,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "max_tool_calls": self.max_tool_calls,
            "max_model_calls": self.max_model_calls,
            "request_shape": dict(self.request_shape),
            "tool_json_schema": dict(self.tool_json_schema),
            "tool_policy_sha256": sha256_text(self.tool_policy_text),
            "hash": self.hash,
        }


DEFAULT_TOOL_CONTRACT = ToolContract()


@dataclass(frozen=True, slots=True)
class SearchToolBudget:
    """Per-task budget that framework adapters must share with the gateway."""

    max_tool_calls: int = DEFAULT_TOOL_CONTRACT.max_tool_calls
    max_model_calls: int = DEFAULT_TOOL_CONTRACT.max_model_calls
    tool_calls: int = 0
    model_calls: int = 0

    def consume_tool_call(self) -> SearchToolBudget:
        if self.tool_calls >= self.max_tool_calls:
            raise SearchGatewayError("tool_call_budget_exceeded")
        return SearchToolBudget(
            max_tool_calls=self.max_tool_calls,
            max_model_calls=self.max_model_calls,
            tool_calls=self.tool_calls + 1,
            model_calls=self.model_calls,
        )

    def consume_model_call(self) -> SearchToolBudget:
        if self.model_calls >= self.max_model_calls:
            raise SearchGatewayError("model_call_budget_exceeded")
        return SearchToolBudget(
            max_tool_calls=self.max_tool_calls,
            max_model_calls=self.max_model_calls,
            tool_calls=self.tool_calls,
            model_calls=self.model_calls + 1,
        )


@dataclass(frozen=True, slots=True)
class VolcHttpResponse:
    """Small testable representation of a Volc WebSearch response."""

    status_code: int | None
    payload: Mapping[str, Any] | None
    error: str | None = None
    request_id: str | None = None


class VolcSearchTransport(Protocol):
    """Injectable Volc transport used by the production gateway and tests."""

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> VolcHttpResponse: ...


class VolcSdkTransport:
    """Signed Volc SDK transport; its only live endpoint is Torchlight WebSearch."""

    def __init__(self, *, access_key: str, secret_key: str) -> None:
        self.access_key = access_key
        self.secret_key = secret_key

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> VolcHttpResponse:
        if url != VOLC_SEARCH_URL:
            raise SearchGatewayError("search_endpoint_not_allowlisted")
        try:
            from volcengine.ApiInfo import ApiInfo
            from volcengine.base.Service import Service
            from volcengine.Credentials import Credentials
            from volcengine.ServiceInfo import ServiceInfo

            service_info = ServiceInfo(
                host=VOLC_SEARCH_HOST,
                header={},
                credentials=Credentials(
                    self.access_key,
                    self.secret_key,
                    VOLC_SEARCH_SERVICE,
                    VOLC_SEARCH_REGION,
                ),
                connection_timeout=int(timeout_seconds),
                socket_timeout=int(timeout_seconds),
                scheme="https",
            )
            service = Service(
                service_info,
                {
                    VOLC_SEARCH_ACTION: ApiInfo(
                        method="POST",
                        path=VOLC_SEARCH_PATH,
                        query={"Action": VOLC_SEARCH_ACTION, "Version": VOLC_SEARCH_VERSION},
                        header={"Content-Type": "application/json"},
                        form={},
                    )
                },
            )
            payload = service.json(VOLC_SEARCH_ACTION, {}, json_module.dumps(dict(json)))
            if isinstance(payload, str):
                payload = json_module.loads(payload)
            if not isinstance(payload, dict):
                payload = None
        except Exception as exc:
            return VolcHttpResponse(
                status_code=None,
                payload=None,
                error=f"volc_sdk_error:{type(exc).__name__}",
            )
        metadata = payload.get("ResponseMetadata") if isinstance(payload, Mapping) else None
        return VolcHttpResponse(
            status_code=200,
            payload=payload,
            error=None,
            request_id=(metadata.get("RequestId") if isinstance(metadata, Mapping) else None),
        )


class AliyunTemporalFilter:
    """Small supervisor-side LLM reviewer for otherwise ambiguous Volc rows."""

    def __init__(
        self,
        profile: ResolvedModelProfile,
        *,
        network_events: list[NetworkEvent] | None = None,
    ) -> None:
        self.profile = profile
        self.network_events = network_events

    def __call__(self, request: LlmFilterRequest) -> LlmTemporalVerdict:
        if self.profile.mode != "openai_compatible":
            raise RuntimeError("llm_filter_model_unavailable")
        from openai import OpenAI

        payload = request.as_dict()
        content = payload.get("content")
        if isinstance(content, str) and len(content) > 6_000:
            payload["content"] = content[:3_000] + "\n[...truncated...]\n" + content[-3_000:]
        client = OpenAI(
            base_url=self.profile.endpoint,
            api_key=self.profile.api_key or "not-required",
            timeout=self.profile.timeout_seconds,
            max_retries=0,
        )
        request_body = {
            "model": self.profile.model_name,
            "temperature": 0,
            "max_tokens": self.profile.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Review a search result for information after the supplied cutoff. "
                        "Reply with exactly KEEP, REJECT, or UNCERTAIN, followed by a "
                        "short reason."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        }
        max_attempts = self.profile.retries + 1
        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            status_code: int | None = None
            error: str | None = None
            try:
                completion = client.chat.completions.create(
                    **request_body,
                )
                status_code = 200
                message = completion.choices[0].message.content if completion.choices else None
                if not isinstance(message, str) or not message.strip():
                    raise ValueError("empty_llm_filter_response")
            except Exception as exc:
                error = type(exc).__name__
                self._record_network_event(
                    request_body,
                    status_code=status_code,
                    latency_seconds=time.perf_counter() - started,
                    error=error,
                )
                if attempt >= max_attempts:
                    raise RuntimeError("llm_filter_retry_exhausted") from exc
                delay = 0.25 * (2 ** (attempt - 1))
                _LOGGER.warning(
                    "temporal filter retry: attempt=%s/%s reason=%s backoff=%.2fs",
                    attempt + 1,
                    max_attempts,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
                continue
            self._record_network_event(
                request_body,
                status_code=status_code,
                latency_seconds=time.perf_counter() - started,
                error=error,
            )
            normalized = message.strip()
            upper = normalized.upper()
            if upper.startswith("KEEP"):
                return LlmTemporalVerdict(verdict="keep", reason=normalized[4:].strip(" :-"))
            if upper.startswith("REJECT"):
                return LlmTemporalVerdict(verdict="reject", reason=normalized[6:].strip(" :-"))
            return LlmTemporalVerdict(verdict="uncertain", reason=normalized[:500])
        raise AssertionError("temporal filter retry loop must return or raise")

    def _record_network_event(
        self,
        request_body: Mapping[str, Any],
        *,
        status_code: int | None,
        latency_seconds: float,
        error: str | None,
    ) -> None:
        if self.network_events is None:
            return
        parsed = urlparse(self.profile.endpoint)
        base = parsed.path.rstrip("/")
        if base.endswith("/chat/completions"):
            path = base
        elif base.endswith("/v1"):
            path = base + "/chat/completions"
        else:
            path = (base + "/v1/chat/completions") if base else "/v1/chat/completions"
        self.network_events.append(
            NetworkEvent(
                host=(
                    f"{parsed.hostname}:{parsed.port}"
                    if parsed.hostname and parsed.port
                    else (parsed.hostname or "unknown")
                ),
                method="POST",
                path=path,
                body_hash=sha256_json(request_body),
                status_code=status_code,
                latency_seconds=latency_seconds,
                is_model_endpoint=True,
                network_class=TEMPORAL_FILTER_NETWORK_CLASS,
                error=error,
            )
        )


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """A cutoff-compliant source returned to a tool consumer (content remains in memory)."""

    url: str
    title: str | None
    content: str | None
    published_at: str

    def tool_payload(self) -> dict[str, str | None]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "published_at": self.published_at,
        }


@dataclass(frozen=True, slots=True)
class SearchCandidateAudit:
    """One JSONL-safe candidate decision; never stores a query or page content."""

    task_id: str
    framework: str
    query_sha256: str
    url: str | None
    title: str | None
    content_sha256: str | None
    published_at: str | None
    accepted: bool
    reason: str
    response_sha256: str
    cutoff_date: str
    exclusive_cutoff_bound: str
    request_filter_sha256: str
    provider: str = "volc"
    test_only: bool = False
    time_range: str | None = None
    filter_stage: str | None = None
    filter_reason: str | None = None
    rule_signals: tuple[str, ...] = ()
    llm_verdict: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """One auditable search call, including rejected candidates and ledger fields."""

    task_id: str
    framework: str
    query_sha256: str
    cutoff_date: str
    exclusive_cutoff_bound: str
    request_filter_sha256: str
    response_sha256: str | None
    candidates: tuple[SearchCandidate, ...]
    audit_records: tuple[SearchCandidateAudit, ...]
    status_code: int | None
    request_id: str | None
    error_code: str | None
    tool_contract_hash: str
    tool_call_count: int
    max_tool_calls: int
    max_model_calls: int
    test_only: bool

    @property
    def comparable(self) -> bool:
        return (
            self.error_code is None
            and bool(self.audit_records)
            and self.tool_contract_hash == DEFAULT_TOOL_CONTRACT.hash
        )

    @property
    def ranking_eligible(self) -> bool:
        return self.comparable and not self.test_only

    def network_ledger(self, *, profile: str) -> dict[str, Any]:
        """Return the non-secret ledger extension required for web-search calls."""

        accepted = sum(1 for row in self.audit_records if row.accepted)
        rejected = len(self.audit_records) - accepted
        return {
            "profile": profile,
            "network_class": "web_search",
            "provider": "volc",
            "request_id": self.request_id,
            "query_hash": self.query_sha256,
            "cutoff_date": self.cutoff_date,
            "exclusive_cutoff_bound": self.exclusive_cutoff_bound,
            "request_filter_hash": self.request_filter_sha256,
            "accepted_count": accepted,
            "rejected_count": rejected,
            # Every returned candidate is audited and only candidates passing
            # the post-filter can be accepted.  Rejected/missing candidates do
            # not make an otherwise correctly bounded request non-compliant.
            "cutoff_compliant": self.error_code is None
            and all(not row.accepted or row.reason == "accepted" for row in self.audit_records),
            "error_code": self.error_code,
            "host": VOLC_SEARCH_HOST,
            "method": "POST",
            "path": VOLC_SEARCH_PATH,
            "action": VOLC_SEARCH_ACTION,
            "version": VOLC_SEARCH_VERSION,
            "body_hash": self.request_filter_sha256,
            "status_code": self.status_code,
            "is_model_endpoint": False,
        }


class AgenticSearchSession:
    """A per-task shared tool session with immutable, auditable call budgets."""

    def __init__(
        self,
        gateway: CutoffSearchGateway,
        *,
        task_id: str,
        framework: str,
        cutoff_date: str | None,
        category: str | None = None,
    ) -> None:
        self.gateway = gateway
        self.task_id = task_id
        self.framework = framework
        self.cutoff_date = cutoff_date
        self.category = category
        self._budget = SearchToolBudget(
            max_tool_calls=gateway.tool_contract.max_tool_calls,
            max_model_calls=gateway.tool_contract.max_model_calls,
        )
        self._outcomes: list[SearchOutcome] = []

    @property
    def budget(self) -> SearchToolBudget:
        return self._budget

    @property
    def outcomes(self) -> tuple[SearchOutcome, ...]:
        return tuple(self._outcomes)

    def record_model_call(self) -> None:
        self._budget = self._budget.consume_model_call()

    def search(self, query: str) -> SearchOutcome:
        self._budget = self._budget.consume_tool_call()
        outcome = self.gateway._search(  # noqa: SLF001 - session is the gateway's budget owner.
            task_id=self.task_id,
            framework=self.framework,
            query=query,
            cutoff_date=self.cutoff_date,
            category=self.category,
            tool_call_count=self._budget.tool_calls,
        )
        self._outcomes.append(outcome)
        return outcome

    def search_web(self, query: str) -> list[dict[str, str | None]]:
        """Execute the native ``search_web(query: str)`` tool contract.

        The framework integrations expose this method through their own native
        tool registries.  Keeping conversion here guarantees that every model
        issued query goes through the same budgeted, cutoff-filtered gateway.
        """

        return [candidate.tool_payload() for candidate in self.search(query).candidates]


class CutoffSearchGateway:
    """One production Volc WebSearch backend with no fallback or page fetch.

    Live traffic is signed with Volc credentials and sent only to
    ``mercury.volcengineapi.com``.
    """

    def __init__(
        self,
        *,
        access_key_env: str = "VOLC_ACCESS_KEY_ID",
        secret_key_env: str = "VOLC_SECRET_ACCESS_KEY",
        access_key: str | None = None,
        secret_key: str | None = None,
        # Compatibility for old test fixtures.  A single legacy api key can
        # only satisfy test-only transport injection; live Volc always needs
        # both access and secret keys.
        api_key_env: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        transport: VolcSearchTransport | None = None,
        test_only: bool = False,
        tool_contract: ToolContract = DEFAULT_TOOL_CONTRACT,
        lookback_years: int = 5,
        model_profile: ResolvedModelProfile | None = None,
        llm_filter: LlmTemporalFilter | None = None,
    ) -> None:
        if tool_contract.provider != "volc" or tool_contract.endpoint != VOLC_SEARCH_URL:
            raise ValueError("CutoffSearchGateway accepts only the Volc search contract.")
        if tool_contract.max_tool_calls != 3 or tool_contract.max_model_calls != 4:
            raise ValueError("The benchmark web-search budgets are fixed at 3 tools / 4 models.")
        if (
            tool_contract.name != "search_web"
            or dict(tool_contract.tool_json_schema) != SEARCH_WEB_TOOL_JSON_SCHEMA
            or tool_contract.tool_policy_text != SEARCH_WEB_TOOL_POLICY
        ):
            raise ValueError("The benchmark search_web tool schema and policy are fixed.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if not isinstance(lookback_years, int) or isinstance(lookback_years, bool):
            raise ValueError("lookback_years must be an integer.")
        self.access_key_env = api_key_env or access_key_env
        self.secret_key_env = secret_key_env
        self._access_key = access_key or api_key
        self._secret_key = secret_key
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.test_only = test_only
        self.tool_contract = tool_contract
        self.lookback_years = lookback_years
        self._model_profile = model_profile
        self._llm_filter = llm_filter

    def bind_model_profile(self, profile: ResolvedModelProfile) -> None:
        """Attach the supervisor's resolved model for ambiguous-row review."""

        self._model_profile = profile

    def preflight(self) -> dict[str, Any]:
        """Validate readiness without sending a search request.

        A fake injected transport in ``test_only`` mode is permitted without
        credentials. Every live run requires both Volc credentials; no
        generic web-search fallback is available.
        """

        if self.test_only:
            if self.transport is None:
                raise SearchGatewayError("test_only_transport_missing")
            return {
                "provider": "volc",
                "endpoint": VOLC_SEARCH_URL,
                "test_only": True,
                "ranking_eligible": False,
                "tool_contract_hash": self.tool_contract.hash,
            }
        if not self._resolved_access_key():
            raise SearchGatewayError("volc_access_key_missing")
        if not self._resolved_secret_key():
            raise SearchGatewayError("volc_secret_key_missing")
        return {
            "provider": "volc",
            "endpoint": VOLC_SEARCH_URL,
            "test_only": False,
            "ranking_eligible": True,
            "tool_contract_hash": self.tool_contract.hash,
        }

    def new_session(
        self,
        *,
        task_id: str,
        framework: str,
        cutoff_date: str | None,
        category: str | None = None,
    ) -> AgenticSearchSession:
        self._validate_session_inputs(
            task_id=task_id,
            framework=framework,
            cutoff_date=cutoff_date,
            category=category,
        )
        self.preflight()
        return AgenticSearchSession(
            self,
            task_id=task_id,
            framework=framework,
            cutoff_date=cutoff_date,
            category=category,
        )

    def search(
        self,
        *,
        task_id: str,
        framework: str,
        query: str,
        cutoff_date: str | None,
        category: str | None = None,
    ) -> SearchOutcome:
        """One-call convenience API; multi-call agents must retain a session."""

        return self.new_session(
            task_id=task_id,
            framework=framework,
            cutoff_date=cutoff_date,
            category=category,
        ).search(query)

    def _search(
        self,
        *,
        task_id: str,
        framework: str,
        query: str,
        cutoff_date: str | None,
        category: str | None,
        tool_call_count: int,
        network_events: list[NetworkEvent] | None = None,
    ) -> SearchOutcome:
        del category  # a valid seed cutoff, not a category inference, authorizes search.
        try:
            window, query_hash, body, request_filter_hash = prepare_cutoff_volc_request(
                query,
                cutoff_date,
                count=10,
                lookback_years=self.lookback_years,
            )
        except ValueError as exc:
            raise SearchGatewayError(str(exc)) from exc
        headers = {"Content-Type": "application/json"}
        access_key = self._resolved_access_key()
        secret_key = self._resolved_secret_key()
        if not self.test_only and not access_key:
            raise SearchGatewayError("volc_access_key_missing")
        if not self.test_only and not secret_key:
            raise SearchGatewayError("volc_secret_key_missing")

        transport = self.transport
        if transport is None:
            if self.test_only:
                raise SearchGatewayError("test_only_transport_missing")
            transport = VolcSdkTransport(
                access_key=access_key or "",
                secret_key=secret_key or "",
            )

        started = time.perf_counter()
        try:
            response = transport.post(
                VOLC_SEARCH_URL,
                json=body,
                headers=headers,
                timeout_seconds=self.timeout_seconds,
            )
        except SearchGatewayError:
            raise
        except Exception as exc:
            response = VolcHttpResponse(
                status_code=None,
                payload=None,
                error=f"transport_error:{exc}",
            )
        elapsed = time.perf_counter() - started
        del elapsed  # the root runner owns the canonical network ledger timing field.

        return self._outcome_from_response(
            task_id=task_id,
            framework=framework,
            query_hash=query_hash,
            window=window,
            request_filter_hash=request_filter_hash,
            time_range=str(body["TimeRange"]),
            response=response,
            tool_call_count=tool_call_count,
            network_events=network_events,
        )

    def _outcome_from_response(
        self,
        *,
        task_id: str,
        framework: str,
        query_hash: str,
        window: CutoffWindow,
        request_filter_hash: str,
        time_range: str,
        response: VolcHttpResponse,
        tool_call_count: int,
        network_events: list[NetworkEvent] | None,
    ) -> SearchOutcome:
        payload = response.payload
        response_hash = sha256_json(payload) if payload is not None else None
        if response.error or not 200 <= (response.status_code or 0) < 300:
            error = response.error or "volc_response_invalid"
            audit = _sentinel_audit(
                task_id=task_id,
                framework=framework,
                query_hash=query_hash,
                window=window,
                request_filter_hash=request_filter_hash,
                response_hash=response_hash or sha256_text(error),
                reason=error,
                test_only=self.test_only,
                time_range=time_range,
            )
            return SearchOutcome(
                task_id=task_id,
                framework=framework,
                query_sha256=query_hash,
                cutoff_date=window.cutoff_date,
                exclusive_cutoff_bound=window.exclusive_bound,
                request_filter_sha256=request_filter_hash,
                response_sha256=response_hash,
                candidates=(),
                audit_records=(audit,),
                status_code=response.status_code,
                request_id=response.request_id,
                error_code=_safe_error_code(error),
                tool_contract_hash=self.tool_contract.hash,
                tool_call_count=tool_call_count,
                max_tool_calls=self.tool_contract.max_tool_calls,
                max_model_calls=self.tool_contract.max_model_calls,
                test_only=self.test_only,
            )

        response_error = _volc_response_error(payload)
        if response_error:
            return self._malformed_outcome(
                task_id=task_id,
                framework=framework,
                query_hash=query_hash,
                window=window,
                request_filter_hash=request_filter_hash,
                time_range=time_range,
                response_hash=response_hash or sha256_text(response_error),
                response=response,
                tool_call_count=tool_call_count,
                reason=response_error,
            )

        raw_results = _volc_web_results(payload)
        if not isinstance(raw_results, list):
            return self._malformed_outcome(
                task_id=task_id,
                framework=framework,
                query_hash=query_hash,
                window=window,
                request_filter_hash=request_filter_hash,
                time_range=time_range,
                response_hash=response_hash or sha256_text("volc_payload_invalid"),
                response=response,
                tool_call_count=tool_call_count,
                reason="volc_payload_invalid",
            )

        audits: list[SearchCandidateAudit] = []
        candidates: list[SearchCandidate] = []
        normalized: list[VolcSearchCandidate] = []
        for raw in raw_results:
            if not isinstance(raw, Mapping):
                audits.append(
                    _sentinel_audit(
                        task_id=task_id,
                        framework=framework,
                        query_hash=query_hash,
                        window=window,
                        request_filter_hash=request_filter_hash,
                        response_hash=response_hash or sha256_text("invalid_candidate"),
                        reason="invalid_candidate",
                        test_only=self.test_only,
                        time_range=time_range,
                    )
                )
                continue
            normalized.append(VolcSearchCandidate.from_volc_result(raw))

        for decision in filter_volc_candidates(
            normalized,
            window.cutoff_date,
            llm_filter=self._temporal_filter(network_events=network_events),
        ).decisions:
            candidate = decision.candidate
            content = candidate.content or candidate.snippet
            published = _published_at_text(candidate.volc_publish_time)
            audit = SearchCandidateAudit(
                task_id=task_id,
                framework=framework,
                query_sha256=query_hash,
                url=candidate.url,
                title=candidate.title,
                content_sha256=sha256_text(content) if content is not None else None,
                published_at=published,
                accepted=decision.accepted,
                reason="accepted" if decision.accepted else decision.reason,
                response_sha256=response_hash or sha256_text("volc_response_missing"),
                cutoff_date=window.cutoff_date,
                exclusive_cutoff_bound=window.exclusive_bound,
                request_filter_sha256=request_filter_hash,
                test_only=self.test_only,
                time_range=time_range,
                filter_stage="llm" if decision.llm_verdict is not None else "rule",
                filter_reason=decision.reason,
                rule_signals=decision.rule_signals,
                llm_verdict=decision.llm_verdict,
            )
            audits.append(audit)
            if decision.accepted and candidate.url:
                candidates.append(
                    SearchCandidate(
                        url=candidate.url,
                        title=candidate.title,
                        content=content,
                        published_at=published or "unknown",
                    )
                )

        if not audits:
            audits.append(
                _sentinel_audit(
                    task_id=task_id,
                    framework=framework,
                    query_hash=query_hash,
                    window=window,
                    request_filter_hash=request_filter_hash,
                    response_hash=response_hash or sha256_text("empty_results"),
                    reason="no_candidates",
                    test_only=self.test_only,
                    time_range=time_range,
                )
            )
        return SearchOutcome(
            task_id=task_id,
            framework=framework,
            query_sha256=query_hash,
            cutoff_date=window.cutoff_date,
            exclusive_cutoff_bound=window.exclusive_bound,
            request_filter_sha256=request_filter_hash,
            response_sha256=response_hash,
            candidates=tuple(candidates),
            audit_records=tuple(audits),
            status_code=response.status_code,
            request_id=response.request_id,
            error_code=None,
            tool_contract_hash=self.tool_contract.hash,
            tool_call_count=tool_call_count,
            max_tool_calls=self.tool_contract.max_tool_calls,
            max_model_calls=self.tool_contract.max_model_calls,
            test_only=self.test_only,
        )

    def _malformed_outcome(
        self,
        *,
        task_id: str,
        framework: str,
        query_hash: str,
        window: CutoffWindow,
        request_filter_hash: str,
        time_range: str,
        response_hash: str,
        response: VolcHttpResponse,
        tool_call_count: int,
        reason: str,
    ) -> SearchOutcome:
        audit = _sentinel_audit(
            task_id=task_id,
            framework=framework,
            query_hash=query_hash,
            window=window,
            request_filter_hash=request_filter_hash,
            response_hash=response_hash,
            reason=reason,
            test_only=self.test_only,
            time_range=time_range,
        )
        return SearchOutcome(
            task_id=task_id,
            framework=framework,
            query_sha256=query_hash,
            cutoff_date=window.cutoff_date,
            exclusive_cutoff_bound=window.exclusive_bound,
            request_filter_sha256=request_filter_hash,
            response_sha256=response_hash,
            candidates=(),
            audit_records=(audit,),
            status_code=response.status_code,
            request_id=response.request_id,
            error_code=reason,
            tool_contract_hash=self.tool_contract.hash,
            tool_call_count=tool_call_count,
            max_tool_calls=self.tool_contract.max_tool_calls,
            max_model_calls=self.tool_contract.max_model_calls,
            test_only=self.test_only,
        )

    def _temporal_filter(
        self, *, network_events: list[NetworkEvent] | None = None
    ) -> LlmTemporalFilter | None:
        if self._llm_filter is not None:
            return self._llm_filter
        if self._model_profile is None or self._model_profile.mode != "openai_compatible":
            return None
        return AliyunTemporalFilter(self._model_profile, network_events=network_events)

    def _resolved_access_key(self) -> str | None:
        return _first_env_value(
            self._access_key,
            os.environ.get(self.access_key_env),
            os.environ.get("VOLC_ACCESS_KEY_ID"),
            os.environ.get("VOLC_ACCESSKEY"),
        )

    def _resolved_secret_key(self) -> str | None:
        return _first_env_value(
            self._secret_key,
            os.environ.get(self.secret_key_env),
            os.environ.get("VOLC_SECRET_ACCESS_KEY"),
            os.environ.get("VOLC_SECRETKEY"),
        )

    @staticmethod
    def _validate_session_inputs(
        *,
        task_id: str,
        framework: str,
        cutoff_date: str | None,
        category: str | None,
    ) -> None:
        del category
        if not isinstance(task_id, str) or not task_id:
            raise SearchGatewayError("task_id_missing")
        if not isinstance(framework, str) or not framework:
            raise SearchGatewayError("framework_missing")
        try:
            CutoffWindow.from_seed(cutoff_date)
        except CutoffWindowError as exc:
            raise SearchGatewayError(str(exc)) from exc


def write_search_audit_jsonl(
    path: str | Path,
    records: Sequence[SearchCandidateAudit],
    *,
    hash_only: bool = False,
) -> Path:
    """Append candidate decisions without retaining source content or queries.

    The record is already content-hashed.  In ``hash_only`` mode, the remaining
    source-identifying fields are represented by stable hashes too, matching
    the artifact sink's privacy contract.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for record in records:
            row = record.as_dict()
            if hash_only:
                for key in ("url", "title", "published_at"):
                    value = row.pop(key, None)
                    if isinstance(value, str):
                        row[f"{key}_sha256"] = sha256_text(value)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return output


def _sentinel_audit(
    *,
    task_id: str,
    framework: str,
    query_hash: str,
    window: CutoffWindow,
    request_filter_hash: str,
    response_hash: str,
    reason: str,
    test_only: bool,
    time_range: str | None = None,
) -> SearchCandidateAudit:
    return SearchCandidateAudit(
        task_id=task_id,
        framework=framework,
        query_sha256=query_hash,
        url=None,
        title=None,
        content_sha256=None,
        published_at=None,
        accepted=False,
        reason=reason,
        response_sha256=response_hash,
        cutoff_date=window.cutoff_date,
        exclusive_cutoff_bound=window.exclusive_bound,
        request_filter_sha256=request_filter_hash,
        test_only=test_only,
        time_range=time_range,
    )


def _safe_error_code(value: str) -> str:
    """Prevent provider response text from becoming a persisted error value."""

    if value.startswith("http_status_"):
        return value
    if value.startswith("network_error"):
        return "network_error"
    if value.startswith("transport_error"):
        return "transport_error"
    return "volc_request_failed"


def _first_env_value(*values: str | None) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _published_at_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return None


def _volc_web_results(payload: Mapping[str, Any] | None) -> list[Any] | None:
    if not isinstance(payload, Mapping):
        return None
    result = payload.get("Result")
    if isinstance(result, Mapping):
        web_results = result.get("WebResults")
        return web_results if isinstance(web_results, list) else None
    # Keep a narrow compatibility allowance for captured SDK fixtures while
    # retaining a Volc-specific result field and no generic provider fallback.
    web_results = payload.get("WebResults")
    return web_results if isinstance(web_results, list) else None


def _volc_response_error(payload: Mapping[str, Any] | None) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    metadata = payload.get("ResponseMetadata")
    if not isinstance(metadata, Mapping):
        return None
    error = metadata.get("Error")
    if not isinstance(error, Mapping):
        return None
    code = error.get("Code")
    return f"volc_error_{code}" if isinstance(code, str) and code else "volc_response_error"
