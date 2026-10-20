"""Live, plan-bound adapters for the sealed Path B v3 execution path.

The v3 coordinator deliberately does not know how a framework is started.  This
module is the one explicit production connector: it turns a sealed cell into
the adapter named by that cell, keeps model/provider credentials in the parent
process, and gives workers only the typed capability they need.  In particular,
an unrestricted cell never reuses the resident cutoff service or its request
builder.

The connector is intentionally separate from the legacy experiment executor.
It can be constructed for a dry dispatch without touching a source checkout or
starting a service; source verification and process startup happen only when a
cell is actually executed.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.contracts import NetworkEvent, TaskInput, sha256_json, sha256_text
from benchmark_harness.experiment.path_b_v3_backend import (
    PathBV3Cell,
    PathBV3CurrentWebBackend,
    PathBV3ExecutionRequest,
    PathBV3SystemCellExecutor,
    _materialize_cells,
)
from benchmark_harness.experiment.path_b_v3_runner import (
    MODEL_API_KEY_ENV,
    PATH_B_V3_GPT41_BACKBONE_ID,
    PATH_B_V3_GPT41_CUTOFF_SOURCE,
    PATH_B_V3_MINI_BACKBONE_ID,
    PATH_B_V3_MINI_CUTOFF_SOURCE,
    PathBV3ExecutionError,
    PathBV3Plans,
)
from benchmark_harness.search.filter import VolcSearchCandidate
from benchmark_harness.search.volc import (
    SearchCandidate,
    SearchCandidateAudit,
    SearchGatewayError,
    SearchOutcome,
    VolcHttpResponse,
    VolcSdkTransport,
    VolcSearchTransport,
    _volc_response_error,
    _volc_web_results,
)
from benchmark_harness.search.volc_core import (
    VOLC_SEARCH_HOST,
    VOLC_SEARCH_PATH,
    VOLC_SEARCH_URL,
)
from benchmark_harness.search_service import SearchServiceClient, SearchServiceSession
from benchmark_harness.search_service.api import (
    SEARCH_SERVICE_PROTOCOL,
    SearchServiceError,
    SearchServiceManifest,
)
from benchmark_harness.search_service.config import SearchServiceClientConfig, SearchServiceConfig
from benchmark_harness.search_service.engine import EngineSearchResult
from benchmark_harness.search_service.server import create_app
from benchmark_harness.systems.adapters.financeharness import FinanceHarnessSystem
from benchmark_harness.systems.adapters.gpt_researcher import GptResearcherAdapter
from benchmark_harness.systems.adapters.odr import OdrAdapter
from benchmark_harness.systems.contracts import ExperimentBinding, SystemDescriptor
from benchmark_harness.systems.regimes import RetrievalPolicy, build_retrieval_policy

DEFAULT_BATCH_SIZE = 8
MODEL_ENDPOINT_ENV = "BENCHMARK_MODEL_ENDPOINT"
MODEL_NAME_ENV = "BENCHMARK_MODEL_NAME"
SEARCH_SERVICE_URL_ENV = "BENCHMARK_SEARCH_SERVICE_URL"
SEARCH_SERVICE_TOKEN_FILE_ENV = "BENCHMARK_SEARCH_SERVICE_ADMIN_TOKEN_FILE"
SEARCH_SERVICE_COHORT_ENV = "BENCHMARK_EXPECTED_SERVICE_COHORT_SHA256"
VOLC_ACCESS_KEY_ENV = "VOLC_ACCESS_KEY_ID"
VOLC_SECRET_KEY_ENV = "VOLC_SECRET_ACCESS_KEY"

_SOURCE_ENV_BY_SYSTEM = {
    "financeharness": "BENCHMARK_FINANCE_SOURCE",
    "gpt-researcher": "BENCHMARK_GPT_SOURCE",
    "odr": "BENCHMARK_ODR_SOURCE",
}
_EXPECTED_ADAPTERS = {
    "financeharness": "financeharness-external-stdio-v2",
    "gpt-researcher": "gpt-researcher-typed-host-v3",
    "odr": "odr-typed-host-v3",
}
_PLACEHOLDERS = frozenset({"", "replace_me", "todo", "changeme", "none", "null"})
_AZURE_OPENAI_HOST_SUFFIX = ".services.ai.azure.com"
_AZURE_OPENAI_API_PATH = "/openai/v1"
_CHECKPOINT_LOGGER = logging.getLogger(__name__)


class PathBV3LiveError(PathBV3ExecutionError):
    """Raised when the explicit v3 live connector cannot be constructed."""


def _default_checkpoint_callback(payload: Mapping[str, Any]) -> None:
    """Log only the backend's bounded, score-free checkpoint projection."""

    safe = {
        key: payload[key]
        for key in (
            "schema_version",
            "event",
            "batch_index",
            "total_batches",
            "completed_cells",
            "pending_cells",
            "cell_count",
            "ranking_eligible",
            "namespaces",
        )
        if key in payload
    }
    _CHECKPOINT_LOGGER.info(
        "path-b-v3 checkpoint %s",
        json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PathBV3LiveError(f"sealed live descriptor field is missing: {field}")
    return value.strip()


def _usable(value: Any) -> bool:
    return isinstance(value, str) and value.strip().casefold() not in _PLACEHOLDERS


def _normalize_operator_azure_endpoint(value: str) -> str:
    """Validate and canonicalize the operator-supplied Azure endpoint."""

    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise PathBV3LiveError("operator model endpoint is malformed") from exc
    if (
        parsed.scheme.lower() != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port is not None
    ):
        raise PathBV3LiveError(
            "operator model endpoint must be an HTTPS URL without credentials or query"
        )
    normalized_host = host.lower()
    if not normalized_host.endswith(_AZURE_OPENAI_HOST_SUFFIX) or normalized_host == (
        _AZURE_OPENAI_HOST_SUFFIX.lstrip(".")
    ):
        raise PathBV3LiveError(
            "operator model endpoint must use an Azure services.ai.azure.com host"
        )
    normalized_path = parsed.path.rstrip("/") or "/"
    if normalized_path != _AZURE_OPENAI_API_PATH:
        raise PathBV3LiveError("operator model endpoint path must be /openai/v1")
    return f"https://{normalized_host}{_AZURE_OPENAI_API_PATH}"


def _resolved_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class PathBV3CurrentWebSearch:
    """Parent-owned unrestricted Volc transport with no cutoff filter.

    This is deliberately not a ``CutoffSearchGateway``.  The request body has
    no ``TimeRange`` field and the result is recorded as a leakage-probe
    observation.  Workers receive only the typed result payload; Volc keys are
    never put in a child environment or a capability manifest.
    """

    def __init__(
        self,
        *,
        access_key: str | None = None,
        secret_key: str | None = None,
        environ: Mapping[str, str] | None = None,
        transport: VolcSearchTransport | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._environ = dict(os.environ if environ is None else environ)
        self._access_key = access_key
        self._secret_key = secret_key
        self.transport = transport
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._network_ledger: list[dict[str, Any]] = []

    def preflight(self) -> dict[str, Any]:
        if self.transport is None:
            if not _usable(self._access_key or self._environ.get(VOLC_ACCESS_KEY_ENV)):
                raise PathBV3LiveError(f"{VOLC_ACCESS_KEY_ENV} is required for unrestricted probe")
            if not _usable(self._secret_key or self._environ.get(VOLC_SECRET_KEY_ENV)):
                raise PathBV3LiveError(f"{VOLC_SECRET_KEY_ENV} is required for unrestricted probe")
        return {
            "provider": "volc",
            "endpoint": VOLC_SEARCH_URL,
            "regime": "unrestricted_current_web",
            "time_range_filter": False,
            "ranking_eligible": False,
        }

    def handler(self, *, task_id: str, framework: str) -> _BoundSearchHandler:
        return _BoundSearchHandler(self, task_id=task_id, framework=framework)

    def search_outcome(
        self,
        *,
        task_id: str,
        framework: str,
        query: str,
        cutoff_date: str,
        call_index: int,
    ) -> SearchOutcome:
        self.preflight()
        if not isinstance(query, str) or not query.strip():
            raise SearchGatewayError("search_query_missing")
        if not isinstance(call_index, int) or isinstance(call_index, bool) or call_index < 1:
            raise SearchGatewayError("search_call_index_invalid")
        body: dict[str, Any] = {
            "Query": query.strip(),
            "SearchType": "web",
            "Count": 10,
            "Filter": {"NeedContent": True, "NeedUrl": True},
        }
        request_filter_hash = sha256_json(body)
        query_hash = sha256_text(query.strip())
        response = self._post(body)
        response_hash = sha256_json(response.payload) if response.payload is not None else None
        error = response.error or _volc_response_error(response.payload)
        if error is not None or not 200 <= (response.status_code or 0) < 300:
            reason = error or "volc_response_invalid"
            audit = _probe_audit(
                task_id=task_id,
                framework=framework,
                query_hash=query_hash,
                response_hash=response_hash or sha256_text(reason),
                request_filter_hash=request_filter_hash,
                cutoff_date=cutoff_date,
                reason=reason,
                accepted=False,
            )
            return SearchOutcome(
                task_id=task_id,
                framework=framework,
                query_sha256=query_hash,
                cutoff_date=cutoff_date,
                exclusive_cutoff_bound="",
                request_filter_sha256=request_filter_hash,
                response_sha256=response_hash,
                candidates=(),
                audit_records=(audit,),
                status_code=response.status_code,
                request_id=response.request_id,
                error_code=_safe_error(reason),
                tool_contract_hash=sha256_json(
                    {"protocol": "path-b-v3-unrestricted-current-web/v1"}
                ),
                tool_call_count=call_index,
                max_tool_calls=3,
                max_model_calls=4,
                test_only=False,
            )
        raw_results = _volc_web_results(response.payload)
        if not isinstance(raw_results, list):
            reason = "volc_payload_invalid"
            audit = _probe_audit(
                task_id=task_id,
                framework=framework,
                query_hash=query_hash,
                response_hash=response_hash or sha256_text(reason),
                request_filter_hash=request_filter_hash,
                cutoff_date=cutoff_date,
                reason=reason,
                accepted=False,
            )
            return _probe_outcome(
                task_id=task_id,
                framework=framework,
                cutoff_date=cutoff_date,
                query_hash=query_hash,
                request_filter_hash=request_filter_hash,
                response_hash=response_hash or sha256_text(reason),
                response=response,
                call_index=call_index,
                audits=(audit,),
                candidates=(),
                error_code=reason,
            )

        audits: list[SearchCandidateAudit] = []
        candidates: list[SearchCandidate] = []
        for raw in raw_results:
            if not isinstance(raw, Mapping):
                audits.append(
                    _probe_audit(
                        task_id=task_id,
                        framework=framework,
                        query_hash=query_hash,
                        response_hash=response_hash or sha256_text("invalid_candidate"),
                        request_filter_hash=request_filter_hash,
                        cutoff_date=cutoff_date,
                        reason="invalid_candidate",
                        accepted=False,
                    )
                )
                continue
            normalized = VolcSearchCandidate.from_volc_result(raw)
            model_payload = normalized.as_model_payload()
            published = model_payload.get("published_at")
            content = model_payload.get("content")
            accepted = bool(normalized.url and (content or model_payload.get("title")))
            reason = "unrestricted_current_web" if accepted else "missing_url_or_content"
            audits.append(
                _probe_audit(
                    task_id=task_id,
                    framework=framework,
                    query_hash=query_hash,
                    response_hash=response_hash or sha256_text("volc_response_missing"),
                    request_filter_hash=request_filter_hash,
                    cutoff_date=cutoff_date,
                    reason=reason,
                    accepted=accepted,
                    url=normalized.url,
                    title=normalized.title,
                    content=content,
                    published_at=published,
                )
            )
            if accepted and normalized.url:
                candidates.append(
                    SearchCandidate(
                        url=normalized.url,
                        title=normalized.title,
                        content=content,
                        published_at=published or "unknown",
                    )
                )
        if not audits:
            audits.append(
                _probe_audit(
                    task_id=task_id,
                    framework=framework,
                    query_hash=query_hash,
                    response_hash=response_hash or sha256_text("empty_results"),
                    request_filter_hash=request_filter_hash,
                    cutoff_date=cutoff_date,
                    reason="no_candidates",
                    accepted=False,
                )
            )
        outcome = _probe_outcome(
            task_id=task_id,
            framework=framework,
            cutoff_date=cutoff_date,
            query_hash=query_hash,
            request_filter_hash=request_filter_hash,
            response_hash=response_hash,
            response=response,
            call_index=call_index,
            audits=tuple(audits),
            candidates=tuple(candidates),
            error_code=None,
        )
        ledger = outcome.network_ledger(profile="unrestricted_current_web")
        ledger.update({"retrieval_regime": "unrestricted_current_web", "time_range_filter": False})
        with self._lock:
            self._network_ledger.append(ledger)
        return outcome

    def _post(self, body: Mapping[str, Any]) -> VolcHttpResponse:
        transport = self.transport
        if transport is None:
            transport = VolcSdkTransport(
                access_key=self._access_key or self._environ.get(VOLC_ACCESS_KEY_ENV, ""),
                secret_key=self._secret_key or self._environ.get(VOLC_SECRET_KEY_ENV, ""),
            )
        try:
            return transport.post(
                VOLC_SEARCH_URL,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout_seconds=self.timeout_seconds,
            )
        except Exception as exc:
            return VolcHttpResponse(
                status_code=None,
                payload=None,
                error=f"transport_error:{type(exc).__name__}",
            )

    @property
    def network_ledger(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(dict(row) for row in self._network_ledger)


class _BoundSearchHandler:
    """Task-bound callable used by ``TypedSearchCapability``."""

    def __init__(self, search: PathBV3CurrentWebSearch, *, task_id: str, framework: str) -> None:
        self.search = search
        self.task_id = task_id
        self.framework = framework
        self._task: TaskInput | None = None
        self._task_token = secrets.token_urlsafe(32)

    def bind_task(self, task: TaskInput, _binding: ExperimentBinding) -> None:
        if task.task_id != self.task_id:
            raise PathBV3LiveError("search handler task binding mismatch")
        self._task = task

    def __call__(self, query: str, token: str, call_index: int) -> Mapping[str, Any]:
        if token != self._task_token:
            raise SearchGatewayError("search_capability_token_invalid")
        if self._task is None:
            raise SearchGatewayError("search_task_binding_missing")
        outcome = self.search.search_outcome(
            task_id=self.task_id,
            framework=self.framework,
            query=query,
            cutoff_date=self._task.cutoff_date,
            call_index=call_index,
        )
        return {"results": [candidate.tool_payload() for candidate in outcome.candidates]}


def _safe_error(value: str) -> str:
    if value.startswith("http_status_"):
        return value
    if value.startswith("transport_error"):
        return "transport_error"
    return "volc_request_failed"


def _probe_audit(
    *,
    task_id: str,
    framework: str,
    query_hash: str,
    response_hash: str,
    request_filter_hash: str,
    cutoff_date: str,
    reason: str,
    accepted: bool,
    url: str | None = None,
    title: str | None = None,
    content: str | None = None,
    published_at: str | None = None,
) -> SearchCandidateAudit:
    return SearchCandidateAudit(
        task_id=task_id,
        framework=framework,
        query_sha256=query_hash,
        url=url,
        title=title,
        content_sha256=sha256_text(content) if content else None,
        published_at=published_at,
        accepted=accepted,
        reason=reason,
        response_sha256=response_hash,
        cutoff_date=cutoff_date,
        exclusive_cutoff_bound="",
        request_filter_sha256=request_filter_hash,
        time_range=None,
        filter_stage="unrestricted_probe",
        filter_reason="no_cutoff_filter_by_sealed_regime",
    )


def _probe_outcome(
    *,
    task_id: str,
    framework: str,
    cutoff_date: str,
    query_hash: str,
    request_filter_hash: str,
    response_hash: str,
    response: VolcHttpResponse,
    call_index: int,
    audits: tuple[SearchCandidateAudit, ...],
    candidates: tuple[SearchCandidate, ...],
    error_code: str | None,
) -> SearchOutcome:
    return SearchOutcome(
        task_id=task_id,
        framework=framework,
        query_sha256=query_hash,
        cutoff_date=cutoff_date,
        exclusive_cutoff_bound="",
        request_filter_sha256=request_filter_hash,
        response_sha256=response_hash,
        candidates=candidates,
        audit_records=audits,
        status_code=response.status_code,
        request_id=response.request_id,
        error_code=error_code,
        tool_contract_hash=sha256_json({"protocol": "path-b-v3-unrestricted-current-web/v1"}),
        tool_call_count=call_index,
        max_tool_calls=3,
        max_model_calls=4,
        test_only=False,
    )


class _CutoffSearchHandler:
    """Lazy task-scoped client for the resident cutoff service."""

    def __init__(
        self,
        client: SearchServiceClient,
        *,
        task_id: str,
        framework: str,
    ) -> None:
        self.client = client
        self.task_id = task_id
        self.framework = framework
        self._task: TaskInput | None = None
        self._session: SearchServiceSession | None = None
        self._task_token = secrets.token_urlsafe(32)

    def bind_task(self, task: TaskInput, _binding: ExperimentBinding) -> None:
        if task.task_id != self.task_id:
            raise PathBV3LiveError("cutoff search handler task binding mismatch")
        self._task = task

    def __call__(self, query: str, token: str, _call_index: int) -> Mapping[str, Any]:
        if token != self._task_token:
            raise SearchServiceError("search_capability_token_invalid", status_code=401)
        if self._task is None:
            raise SearchServiceError("search_task_binding_missing", status_code=409)
        if self._session is None:
            self.client.ready()
            self._session = self.client.open_session(
                run_id=f"path-b-v3:{self.task_id}",
                task_id=self.task_id,
                profile="deep_research",
                baseline=self.framework,
                cutoff_date=self._task.cutoff_date,
                policy_version="deep_research/v2",
                expected_service_cohort_sha256=self.client.expected_service_cohort_sha256,
            )
        outcome = self._session.search(query)
        return {"results": [candidate.tool_payload() for candidate in outcome.candidates]}

    def close(self, *, status: str = "completed") -> None:
        if self._session is not None and self._session.close_result is None:
            self._session.close(status=status)


class _ManagedSystem:
    """Expose the narrow system protocol and close task-scoped search state."""

    def __init__(
        self,
        adapter: Any,
        *,
        descriptor: SystemDescriptor,
        retrieval_policy: RetrievalPolicy,
        cleanup: Callable[[], None] | None = None,
    ) -> None:
        self._adapter = adapter
        self.descriptor = descriptor
        self.retrieval_policy = retrieval_policy
        self._cleanup = cleanup
        self._closed = False
        self._failed = False

    def configure_task(self, task: TaskInput, binding: ExperimentBinding) -> None:
        bind = getattr(self._cleanup, "bind_task", None)
        if callable(bind):
            bind(task, binding)

    def run_task(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._adapter.run_task(*args, **kwargs)
        except Exception:
            self._failed = True
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self._adapter, "close", None)
            if callable(close):
                close()
        finally:
            if self._cleanup is not None:
                cleanup = getattr(self._cleanup, "close", self._cleanup)
                if callable(cleanup):
                    try:
                        cleanup(status="aborted" if self._failed else "completed")
                    except TypeError:
                        cleanup()


class _CurrentWebServiceEngine:
    """Search-service engine facade backed by the parent current-web transport."""

    def __init__(self, search: PathBV3CurrentWebSearch) -> None:
        self._search = search

    def preflight(self) -> dict[str, Any]:
        return self._search.preflight()

    def search(
        self,
        *,
        task_id: str,
        framework: str,
        query: str,
        cutoff_date: str,
        call_index: int,
    ) -> EngineSearchResult:
        outcome = self._search.search_outcome(
            task_id=task_id,
            framework=framework,
            query=query,
            cutoff_date=cutoff_date,
            call_index=call_index,
        )
        event = NetworkEvent(
            host=VOLC_SEARCH_HOST,
            method="POST",
            path=VOLC_SEARCH_PATH,
            body_hash=outcome.request_filter_sha256,
            status_code=outcome.status_code,
            latency_seconds=None,
            is_model_endpoint=False,
            network_class="web_search",
        )
        return EngineSearchResult(outcome=outcome, network_events=(event,))


class PathBV3CurrentWebServiceClient(SearchServiceClient):
    """Ephemeral loopback service used only for FinanceHarness unrestricted cells."""

    def __init__(self, search: PathBV3CurrentWebSearch) -> None:
        self._temporary_root = Path(tempfile.mkdtemp(prefix="path-b-v3-current-web-"))
        port = _free_loopback_port()
        config = SearchServiceConfig(port=port, state_dir=self._temporary_root / "state")
        manifest = SearchServiceManifest(
            api_contract_sha256=sha256_json(
                {"protocol": SEARCH_SERVICE_PROTOCOL, "regime": "unrestricted_current_web"}
            ),
            implementation_manifest_sha256=sha256_json(
                {"implementation": "path-b-v3-current-web-parent-proxy/v1"}
            ),
            capability_sha256=sha256_json(
                {"provider": "volc", "time_range_filter": False, "ranking_eligible": False}
            ),
            service_cohort_sha256=sha256_json(
                {"protocol": SEARCH_SERVICE_PROTOCOL, "regime": "unrestricted_current_web"}
            ),
        )
        engine = _CurrentWebServiceEngine(search)
        # ``create_app`` accepts a state, not a manifest directly.  Constructing
        # it here avoids exposing a second route implementation while keeping
        # the current-web identity distinct from the resident cutoff daemon.
        from benchmark_harness.search_service.state import SearchServiceState

        state = SearchServiceState(config, engine=engine, manifest=manifest)
        app = create_app(config, state=state)
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise PathBV3LiveError(
                "uvicorn is required for FinanceHarness current-web proxy"
            ) from exc
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                log_config=None,
                access_log=False,
                server_header=False,
            )
        )
        self._server = server
        self._thread = threading.Thread(target=server.run, name="path-b-v3-web-proxy", daemon=True)
        self._thread.start()
        super().__init__(
            config=SearchServiceClientConfig(
                base_url=f"http://127.0.0.1:{port}",
                admin_token_file=config.admin_token_file,
                expected_service_cohort_sha256=manifest.service_cohort_sha256,
            )
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                self.health()
                return
            except SearchServiceError:
                time.sleep(0.05)
        self.close()
        raise PathBV3LiveError("current-web loopback proxy failed to become ready")

    def close(self) -> None:
        server = getattr(self, "_server", None)
        if server is not None:
            server.should_exit = True
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout=2.0)


class PathBV3LiveConnector:
    """Build the explicit v3 backend from the sealed cell descriptors."""

    def __init__(
        self,
        plans: PathBV3Plans,
        *,
        environ: Mapping[str, str] | None = None,
        repo_root: str | Path | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
        volc_transport: VolcSearchTransport | None = None,
    ) -> None:
        if not isinstance(plans, PathBV3Plans):
            raise TypeError("PathBV3LiveConnector requires validated PathBV3Plans")
        self.plans = plans
        self.environ = dict(os.environ if environ is None else environ)
        self.repo_root = Path(repo_root or plans.root.parent.parent).expanduser().resolve()
        self.batch_size = batch_size
        self._volc_transport = volc_transport
        self._current_search = PathBV3CurrentWebSearch(
            environ=self.environ,
            transport=volc_transport,
        )
        # Fail before task loading or adapter construction when the
        # unrestricted parent-side provider cannot be authorized.  An
        # injected transport is reserved for deterministic tests and carries
        # its own explicit capability, so it intentionally bypasses env-key
        # presence checks.
        self._current_search.preflight()
        self._cutoff_client: SearchServiceClient | None = None
        self._current_web_service: PathBV3CurrentWebServiceClient | None = None
        self._rows = self._load_system_rows()
        selected_checkpoint_callback = checkpoint_callback or _default_checkpoint_callback
        executor = PathBV3SystemCellExecutor(
            self.system_factory,
            self.model_profile_factory,
        )
        self.backend = PathBV3CurrentWebBackend(
            executor,
            batch_size=batch_size,
            checkpoint_callback=selected_checkpoint_callback,
            dry_dispatch_validator=self._validate_dry_dispatch,
            repo_root=self.repo_root,
        )

    def build_backend(self) -> PathBV3CurrentWebBackend:
        return self.backend

    def dry_dispatch(self, request: PathBV3ExecutionRequest) -> Mapping[str, Any]:
        if request.plans.index.get("index_sha256") != self.plans.index.get("index_sha256"):
            raise PathBV3LiveError("dry-dispatch plan does not match the connector binding")
        result = self.backend.dry_dispatch(request)
        # Exercise the sealed model identity and per-cell descriptor/policy
        # binding without starting an external checkout, worker, or service.
        # This makes dry dispatch catch an endpoint/model alias drift while
        # retaining its zero-result-artifact property.
        cells = _materialize_cells(request.plans)
        return {**dict(result), "validated_cell_bindings": len(cells)}

    def close(self) -> None:
        if self._current_web_service is not None:
            self._current_web_service.close()
            self._current_web_service = None

    def model_profile_factory(self, cell: PathBV3Cell) -> ResolvedModelProfile:
        backbone = self._backbone()
        settings = backbone.get("model_settings")
        if not isinstance(settings, Mapping):
            raise PathBV3LiveError("sealed model_settings are missing")
        model_name = _required_text(settings.get("api_model_name"), "model_settings.api_model_name")
        configured_name = self.environ.get(MODEL_NAME_ENV)
        endpoint_binding = settings.get("endpoint_binding", "sealed_artifact")
        if endpoint_binding == "operator_supplied_env":
            if not configured_name or configured_name.strip() != model_name:
                raise PathBV3LiveError(
                    f"{MODEL_NAME_ENV} must be exactly the sealed model wire name"
                )
        elif configured_name and configured_name.strip() != model_name:
            raise PathBV3LiveError(f"{MODEL_NAME_ENV} does not match the sealed model snapshot")
        if endpoint_binding == "operator_supplied_env":
            if "endpoint" in settings:
                raise PathBV3LiveError("operator-bound model settings must not seal an endpoint")
            endpoint_env = _required_text(
                settings.get("endpoint_env"), "model_settings.endpoint_env"
            )
            endpoint_value = self.environ.get(endpoint_env)
            if not _usable(endpoint_value):
                raise PathBV3LiveError(f"{endpoint_env} is required for the sealed mini model")
            endpoint = _normalize_operator_azure_endpoint(endpoint_value.strip())
        else:
            expected_endpoint = _required_text(settings.get("endpoint"), "model_settings.endpoint")
            endpoint = self.environ.get(MODEL_ENDPOINT_ENV, expected_endpoint).strip()
            if endpoint != expected_endpoint:
                raise PathBV3LiveError(
                    "BENCHMARK_MODEL_ENDPOINT does not match the sealed snapshot"
                )
        api_key_env = _required_text(settings.get("api_key_env"), "model_settings.api_key_env")
        if api_key_env != MODEL_API_KEY_ENV:
            raise PathBV3LiveError("sealed model API key environment is not the approved key")
        api_key = self.environ.get(api_key_env)
        if not _usable(api_key):
            raise PathBV3LiveError(f"{MODEL_API_KEY_ENV} is required; live execution never mocks")
        return ResolvedModelProfile(
            mode=_required_text(settings.get("mode"), "model_settings.mode"),
            endpoint=endpoint,
            model_name=model_name,
            api_key=api_key,
            temperature=float(settings.get("temperature", 0.2)),
            max_tokens=int(settings.get("max_output_tokens", 32768)),
            timeout_seconds=600.0,
            retries=int(settings.get("retries", 0)),
            seed=cell.seed,
            top_p=float(settings["top_p"]) if settings.get("top_p") is not None else None,
            snapshot_model_name=_required_text(
                settings.get("snapshot_model_name", model_name),
                "model_settings.snapshot_model_name",
            ),
            backbone_id=_required_text(backbone.get("backbone_id"), "backbone_id"),
            wire_name_binding=_required_text(
                settings.get("wire_name_binding", "sealed_path_b_v3_snapshot"),
                "model_settings.wire_name_binding",
            ),
        )

    def system_factory(self, cell: PathBV3Cell) -> Any:
        row = self._rows.get(cell.system_id)
        if row is None:
            raise PathBV3LiveError(f"sealed system descriptor is missing: {cell.system_id}")
        descriptor = self._descriptor(cell, row)
        policy = self._retrieval_policy(cell)
        source = self._source_for(cell.system_id)
        if cell.system_id == "financeharness":
            search_service: SearchServiceClient | None = None
            if cell.retrieval_regime == "cutoff_controlled":
                search_service = self._cutoff_service()
            else:
                search_service = self._current_service()
            adapter = FinanceHarnessSystem(
                descriptor=descriptor,
                source_dir=source,
                repo_root=self.repo_root,
                search_service=search_service,
            )
            return _ManagedSystem(adapter, descriptor=descriptor, retrieval_policy=policy)
        if cell.retrieval_regime == "cutoff_controlled":
            handler = _CutoffSearchHandler(
                self._cutoff_service(), task_id=cell.task_id, framework=cell.system_id
            )
        else:
            handler = self._current_search.handler(task_id=cell.task_id, framework=cell.system_id)
        if cell.system_id == "gpt-researcher":
            adapter = GptResearcherAdapter(
                descriptor=descriptor,
                retrieval_policy=policy,
                source_dir=source,
                repo_root=self.repo_root,
                search_handler=handler,
                search_token=handler._task_token,
            )
        elif cell.system_id == "odr":
            adapter = OdrAdapter(
                descriptor=descriptor,
                retrieval_policy=policy,
                source_dir=source,
                repo_root=self.repo_root,
                search_handler=handler,
                search_token=handler._task_token,
            )
        else:  # pragma: no cover - _load_system_rows closes the identity set.
            raise PathBV3LiveError(f"unsupported sealed system: {cell.system_id}")
        return _ManagedSystem(
            adapter,
            descriptor=descriptor,
            retrieval_policy=policy,
            cleanup=handler,
        )

    @staticmethod
    def _retrieval_policy(cell: PathBV3Cell) -> RetrievalPolicy:
        return build_retrieval_policy(
            cell.retrieval_regime,
            live_authorized=cell.retrieval_regime == "unrestricted_current_web",
            authorization_sha256=(
                cell.plan_sha256 if cell.retrieval_regime == "unrestricted_current_web" else None
            ),
        )

    def _validate_dry_dispatch(self, request: PathBV3ExecutionRequest) -> None:
        if request.plans.index.get("index_sha256") != self.plans.index.get("index_sha256"):
            raise PathBV3LiveError("dry-dispatch plan does not match the connector binding")
        cells = _materialize_cells(request.plans)
        for cell in cells:
            row = self._rows.get(cell.system_id)
            if row is None:  # pragma: no cover - materialization already checks the set.
                raise PathBV3LiveError(f"sealed system descriptor is missing: {cell.system_id}")
            self._descriptor(cell, row)
            self._retrieval_policy(cell)
            self.model_profile_factory(cell)

    def _backbone(self) -> Mapping[str, Any]:
        value = self.plans.manifests["t1_t2"].get("backbone")
        if not isinstance(value, Mapping):
            raise PathBV3LiveError("sealed backbone is missing")
        backbone_id = value.get("backbone_id")
        cutoff_sources = {
            PATH_B_V3_GPT41_BACKBONE_ID: PATH_B_V3_GPT41_CUTOFF_SOURCE,
            PATH_B_V3_MINI_BACKBONE_ID: PATH_B_V3_MINI_CUTOFF_SOURCE,
        }
        expected_cutoff_source = cutoff_sources.get(backbone_id)
        if expected_cutoff_source is None:
            raise PathBV3LiveError("sealed backbone is not an approved Path B v3 snapshot")
        if value.get("training_data_cutoff") != "2024-06-01":
            raise PathBV3LiveError("sealed backbone training cutoff drifted")
        if value.get("cutoff_source") != expected_cutoff_source:
            raise PathBV3LiveError("sealed backbone cutoff source drifted")
        return value

    def _load_system_rows(self) -> dict[str, Mapping[str, Any]]:
        raw = self.plans.manifests["t1_t2"].get("systems")
        if not isinstance(raw, list):
            raise PathBV3LiveError("sealed system descriptors are missing")
        rows: dict[str, Mapping[str, Any]] = {}
        for value in raw:
            if not isinstance(value, Mapping):
                raise PathBV3LiveError("sealed system descriptor is malformed")
            system_id = _required_text(value.get("system_id"), "system_id")
            if system_id in rows or system_id not in _EXPECTED_ADAPTERS:
                raise PathBV3LiveError("sealed system descriptor set is invalid")
            if value.get("adapter_id") != _EXPECTED_ADAPTERS[system_id]:
                raise PathBV3LiveError(f"sealed adapter identity drifted for {system_id}")
            if value.get("ranking_eligible") is not False:
                raise PathBV3LiveError("sealed system descriptor is ranking eligible")
            rows[system_id] = value
        if set(rows) != set(_EXPECTED_ADAPTERS):
            raise PathBV3LiveError("sealed system descriptor set is incomplete")
        return rows

    def _descriptor(self, cell: PathBV3Cell, row: Mapping[str, Any]) -> SystemDescriptor:
        backbone = self._backbone()
        training_cutoff = date.fromisoformat(
            _required_text(backbone.get("training_data_cutoff"), "training_data_cutoff")
        )
        if row.get("backbone_id") != backbone.get("backbone_id"):
            raise PathBV3LiveError("system descriptor backbone does not match the sealed plan")
        return SystemDescriptor(
            system_id=_required_text(row.get("system_id"), "system_id"),
            framework_id=_required_text(row.get("framework_id"), "framework_id"),
            framework_revision=_required_text(row.get("framework_revision"), "framework_revision"),
            strategy_id=_required_text(row.get("strategy_id"), "strategy_id"),
            strategy_revision=_required_text(row.get("strategy_revision"), "strategy_revision"),
            backbone_id=_required_text(row.get("backbone_id"), "backbone_id"),
            backbone_training_cutoff=training_cutoff,
            cutoff_source=_required_text(backbone.get("cutoff_source"), "cutoff_source"),
            adapter_id=_required_text(row.get("adapter_id"), "adapter_id"),
            adapter_revision=_required_text(row.get("adapter_revision"), "adapter_revision"),
            profile=_required_text(row.get("profile"), "profile"),
            retrieval_regime=cell.retrieval_regime,
            ranking_eligible=False,
            license_status=(
                row.get("license_status")
                if isinstance(row.get("license_status"), str)
                else "UNKNOWN"
            ),
        )

    def _source_for(self, system_id: str) -> Path:
        env_name = _SOURCE_ENV_BY_SYSTEM[system_id]
        raw = self.environ.get(env_name)
        if not _usable(raw):
            raise PathBV3LiveError(f"{env_name} is required for live {system_id} execution")
        path = _resolved_path(raw, self.repo_root)
        if not path.is_dir():
            raise PathBV3LiveError(f"{env_name} checkout is unavailable")
        return path

    def _cutoff_service(self) -> SearchServiceClient:
        if self._cutoff_client is None:
            url = self.environ.get(SEARCH_SERVICE_URL_ENV, "http://127.0.0.1:8765")
            raw_token = self.environ.get(
                SEARCH_SERVICE_TOKEN_FILE_ENV,
                str(self.repo_root / "data/search-service/admin.token"),
            )
            if not _usable(raw_token):
                raise PathBV3LiveError("Search+Filter admin token file is required")
            token_file = _resolved_path(raw_token, self.repo_root)
            if not token_file.is_file():
                raise PathBV3LiveError("Search+Filter admin token file is unavailable")
            self._cutoff_client = SearchServiceClient(
                config=SearchServiceClientConfig(
                    base_url=url,
                    admin_token_file=token_file,
                    expected_service_cohort_sha256=self.environ.get(SEARCH_SERVICE_COHORT_ENV),
                )
            )
        return self._cutoff_client

    def _current_service(self) -> PathBV3CurrentWebServiceClient:
        if self._current_web_service is None:
            self._current_web_service = PathBV3CurrentWebServiceClient(self._current_search)
        return self._current_web_service


def build_path_b_v3_live_backend(
    plans: PathBV3Plans,
    *,
    environ: Mapping[str, str] | None = None,
    repo_root: str | Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    checkpoint_callback: Callable[[Mapping[str, Any]], None] | None = None,
    volc_transport: VolcSearchTransport | None = None,
) -> tuple[PathBV3CurrentWebBackend, PathBV3LiveConnector]:
    """Return the explicit backend and its lifecycle owner."""

    connector = PathBV3LiveConnector(
        plans,
        environ=environ,
        repo_root=repo_root,
        batch_size=batch_size,
        checkpoint_callback=checkpoint_callback,
        volc_transport=volc_transport,
    )
    return connector.build_backend(), connector


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "PathBV3CurrentWebSearch",
    "PathBV3CurrentWebServiceClient",
    "PathBV3LiveConnector",
    "PathBV3LiveError",
    "build_path_b_v3_live_backend",
]
