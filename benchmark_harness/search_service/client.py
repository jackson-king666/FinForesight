"""Standard-library Search Service client used by adapters and the child worker."""

from __future__ import annotations

import http.client
import json
import secrets
import threading
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_harness.search.volc import (
    SearchCandidate,
    SearchCandidateAudit,
    SearchToolBudget,
    ToolContract,
)

from .api import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    SEARCH_SERVICE_PROTOCOL,
    SearchReceiptRef,
    SearchServiceError,
    SearchSessionDescriptor,
    ServiceLimits,
)
from .config import SearchServiceClientConfig


@dataclass(frozen=True, slots=True)
class ServiceSearchOutcome:
    """SearchOutcome-compatible result reconstructed from a service response."""

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
    receipt: SearchReceiptRef
    filter_review_count: int = 0
    filter_model_call_count: int = 0
    network_ledger_rows: tuple[dict[str, Any], ...] = ()

    def network_ledger(self, *, profile: str) -> dict[str, Any]:
        if self.network_ledger_rows:
            row = dict(self.network_ledger_rows[0])
            row.setdefault("profile", profile)
            return row
        accepted = sum(1 for row in self.audit_records if row.accepted)
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
            "rejected_count": len(self.audit_records) - accepted,
            "cutoff_compliant": self.error_code is None,
            "host": "mercury.volcengineapi.com",
            "method": "POST",
            "path": "/",
            "action": "WebSearch",
            "version": "2025-01-01",
            "body_hash": self.request_filter_sha256,
            "status_code": self.status_code,
            "is_model_endpoint": False,
        }


class _ServiceGatewayFacade:
    """Small facade retained only for the native adapter tool-contract seam."""

    def __init__(self, profile: str, *, test_only: bool) -> None:
        max_tools = 3 if profile == "agentic_web_search" else 8
        max_models = 4 if profile == "agentic_web_search" else 32
        self.tool_contract = ToolContract(
            version="agentic_web_search/v3"
            if profile == "agentic_web_search"
            else "deep_research/v2",
            max_tool_calls=max_tools,
            max_model_calls=max_models,
        )
        self.test_only = test_only


class SearchServiceClient:
    """Strict HTTP/JSON client; it never starts a service or follows redirects."""

    def __init__(
        self,
        config: SearchServiceClientConfig | None = None,
        *,
        base_url: str | None = None,
        admin_token_file: str | Path | None = None,
        transport: Any | None = None,
    ) -> None:
        selected = config or SearchServiceClientConfig()
        self.base_url = _validate_base_url(base_url or selected.base_url)
        self.admin_token_file = Path(admin_token_file or selected.admin_token_file)
        self.connect_timeout_seconds = selected.connect_timeout_seconds
        self.request_timeout_seconds = selected.request_timeout_seconds
        self.expected_service_cohort_sha256 = selected.expected_service_cohort_sha256
        self._transport = transport
        self._test_only = False

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/healthz", auth=None)

    def ready(self) -> dict[str, Any]:
        payload = self._request("GET", "/readyz", auth=self._admin_token())
        self._test_only = bool(payload.get("test_only", False))
        return payload

    preflight = ready

    def open_session(
        self,
        *,
        run_id: str,
        task_id: str,
        profile: str,
        baseline: str,
        cutoff_date: str,
        policy_version: str,
        expected_service_cohort_sha256: str | None = None,
    ) -> SearchServiceSession:
        expected_cohort = (
            expected_service_cohort_sha256
            if expected_service_cohort_sha256 is not None
            else self.expected_service_cohort_sha256
        )
        if not _is_sha256_hex(expected_cohort):
            raise SearchServiceError("service_cohort_required", status_code=412)
        request = {
            "schema_version": SEARCH_SERVICE_PROTOCOL,
            "request_id": secrets.token_hex(16),
            "run_id": run_id,
            "task_id": task_id,
            "profile": profile,
            "baseline": baseline,
            "cutoff_date": cutoff_date,
            "policy_version": policy_version,
            "expected_service_cohort_sha256": expected_cohort,
        }
        payload = self._request("POST", "/v1/sessions", body=request, auth=self._admin_token())
        descriptor = _descriptor_from_payload(payload)
        return SearchServiceSession(self, descriptor)

    def close_session(
        self,
        descriptor: SearchSessionDescriptor,
        receipts: tuple[SearchReceiptRef, ...],
        *,
        status: str = "completed",
    ) -> dict[str, Any]:
        body = {
            "schema_version": SEARCH_SERVICE_PROTOCOL,
            "request_id": secrets.token_hex(16),
            "session_id": descriptor.session_id,
            "run_id": descriptor.run_id,
            "task_id": descriptor.task_id,
            "profile": descriptor.profile,
            "baseline": descriptor.baseline,
            "cutoff_date": descriptor.cutoff_date,
            "status": status,
            "receipt_sha256s": [receipt.receipt_sha256 for receipt in receipts],
        }
        return self._request(
            "POST",
            f"/v1/sessions/{_path_token(descriptor.session_id)}/close",
            body=body,
            auth=self._admin_token(),
        )

    def drain(self) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/admin/drain",
            body={"schema_version": SEARCH_SERVICE_PROTOCOL, "request_id": secrets.token_hex(16)},
            auth=self._admin_token(),
        )

    def search(
        self,
        descriptor: SearchSessionDescriptor,
        *,
        request_id: str,
        call_index: int,
        query: str,
    ) -> ServiceSearchOutcome:
        body = {
            "schema_version": SEARCH_SERVICE_PROTOCOL,
            "request_id": request_id,
            "session_id": descriptor.session_id,
            "run_id": descriptor.run_id,
            "task_id": descriptor.task_id,
            "profile": descriptor.profile,
            "baseline": descriptor.baseline,
            "cutoff_date": descriptor.cutoff_date,
            "call_index": call_index,
            "query": query,
        }
        payload = self._request(
            "POST",
            f"/v1/sessions/{_path_token(descriptor.session_id)}/search",
            body=body,
            auth=descriptor.task_token,
        )
        return _outcome_from_payload(payload, descriptor)

    def _admin_token(self) -> str:
        try:
            token = self.admin_token_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SearchServiceError("service_admin_token_unavailable", status_code=503) from exc
        if not token or len(token) < 32:
            raise SearchServiceError("service_admin_token_invalid", status_code=503)
        return token

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        auth: str | None,
    ) -> dict[str, Any]:
        if not path.startswith("/") or "//" in path[1:]:
            raise SearchServiceError("service_path_invalid", status_code=400)
        encoded = b""
        headers = {"Accept": "application/json"}
        if auth is not None:
            headers["Authorization"] = f"Bearer {auth}"
        if body is not None:
            encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(encoded) > MAX_REQUEST_BYTES:
                raise SearchServiceError("request_too_large", status_code=413)
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(encoded))
        if self._transport is not None:
            return self._transport.request(method, path, body, auth)
        host = urllib.parse.urlsplit(self.base_url).hostname
        port = urllib.parse.urlsplit(self.base_url).port or 80
        connection = http.client.HTTPConnection(host, port, timeout=self.request_timeout_seconds)
        try:
            connection.request(method, path, body=encoded or None, headers=headers)
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, TimeoutError) as exc:
            raise SearchServiceError("service_unavailable", status_code=503) from exc
        finally:
            connection.close()
        if len(raw) > MAX_RESPONSE_BYTES:
            raise SearchServiceError("response_too_large", status_code=502)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SearchServiceError("service_response_invalid", status_code=502) from exc
        if not isinstance(value, dict):
            raise SearchServiceError("service_response_invalid", status_code=502)
        if response.status >= 400:
            error = value.get("error")
            code = error.get("code") if isinstance(error, dict) else None
            raise SearchServiceError(
                str(code or "service_request_failed"), status_code=response.status
            )
        return value


class SearchServiceSession:
    """Adapter-facing per-task session over the shared service client."""

    def __init__(self, client: SearchServiceClient, descriptor: SearchSessionDescriptor) -> None:
        self.client = client
        self.descriptor = descriptor
        self.gateway = _ServiceGatewayFacade(descriptor.profile, test_only=client._test_only)
        self.task_id = descriptor.task_id
        self.framework = descriptor.baseline
        self.cutoff_date = descriptor.cutoff_date
        self.category: str | None = None
        self._budget = SearchToolBudget(
            max_tool_calls=descriptor.limits.max_search_calls,
            max_model_calls=4 if descriptor.profile == "agentic_web_search" else 32,
        )
        self._outcomes: list[ServiceSearchOutcome] = []
        self._receipts: list[SearchReceiptRef] = []
        self._lock = threading.Lock()
        self._closed = False
        self.close_result: dict[str, Any] | None = None

    @property
    def budget(self) -> SearchToolBudget:
        return self._budget

    @property
    def outcomes(self) -> tuple[ServiceSearchOutcome, ...]:
        return tuple(self._outcomes)

    @property
    def receipts(self) -> tuple[SearchReceiptRef, ...]:
        return tuple(self._receipts)

    def record_model_call(self) -> None:
        with self._lock:
            self._budget = self._budget.consume_model_call()

    def search(self, query: str) -> ServiceSearchOutcome:
        with self._lock:
            if self._closed:
                raise SearchServiceError("session_closed", status_code=409)
            self._budget = self._budget.consume_tool_call()
            call_index = self._budget.tool_calls
        outcome = self.client.search(
            self.descriptor,
            request_id=secrets.token_hex(16),
            call_index=call_index,
            query=query,
        )
        with self._lock:
            self._outcomes.append(outcome)
            self._receipts.append(outcome.receipt)
        return outcome

    def search_web(self, query: str) -> list[dict[str, str | None]]:
        return [candidate.tool_payload() for candidate in self.search(query).candidates]

    def close(
        self,
        *,
        status: str = "completed",
        receipts: tuple[SearchReceiptRef, ...] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self.close_result is not None:
                return self.close_result
            self._closed = True
            result = self.client.close_session(
                self.descriptor,
                tuple(self._receipts) if receipts is None else receipts,
                status=status,
            )
            self.close_result = result
            return result


def _validate_base_url(value: str) -> str:
    if not isinstance(value, str):
        raise SearchServiceError("service_url_invalid", status_code=400)
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
        raise SearchServiceError("service_url_must_be_ipv4_loopback", status_code=400)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SearchServiceError("service_url_invalid", status_code=400)
    if parsed.path not in {"", "/"}:
        raise SearchServiceError("service_url_path_invalid", status_code=400)
    if parsed.port is None or not 1 <= parsed.port <= 65535:
        raise SearchServiceError("service_url_port_invalid", status_code=400)
    return f"http://127.0.0.1:{parsed.port}"


def _is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _path_token(value: str) -> str:
    if not value or any(ch not in "0123456789abcdef-" for ch in value.lower()):
        raise SearchServiceError("session_id_invalid", status_code=400)
    return urllib.parse.quote(value, safe="-")


def _descriptor_from_payload(payload: dict[str, Any]) -> SearchSessionDescriptor:
    value = payload.get("session")
    if not isinstance(value, dict):
        raise SearchServiceError("service_response_invalid", status_code=502)
    limits_value = value.get("limits")
    if not isinstance(limits_value, dict):
        raise SearchServiceError("service_response_invalid", status_code=502)
    try:
        limits = ServiceLimits(
            max_search_calls=int(limits_value["max_search_calls"]),
            filter_review_cap=int(limits_value["filter_review_cap"]),
            filter_attempt_cap=int(limits_value["filter_attempt_cap"]),
            ttl_seconds=int(limits_value["ttl_seconds"]),
            max_candidates=int(limits_value.get("max_candidates", 10)),
        )
        return SearchSessionDescriptor(
            session_id=str(value["session_id"]),
            task_token=str(value["task_token"]),
            run_id=str(value["run_id"]),
            task_id=str(value["task_id"]),
            profile=str(value["profile"]),
            baseline=str(value["baseline"]),
            cutoff_date=str(value["cutoff_date"]),
            policy_version=str(value["policy_version"]),
            expires_at=str(value["expires_at"]),
            instance_id=str(value["instance_id"]),
            limits=limits,
            api_contract_sha256=str(value["api_contract_sha256"]),
            implementation_manifest_sha256=str(value["implementation_manifest_sha256"]),
            capability_sha256=str(value["capability_sha256"]),
            service_cohort_sha256=str(value["service_cohort_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SearchServiceError("service_response_invalid", status_code=502) from exc


def _outcome_from_payload(
    payload: dict[str, Any], descriptor: SearchSessionDescriptor
) -> ServiceSearchOutcome:
    raw_receipt = payload.get("receipt")
    if not isinstance(raw_receipt, dict):
        raise SearchServiceError("service_receipt_missing", status_code=502)
    try:
        receipt = SearchReceiptRef(
            receipt_index=int(raw_receipt["receipt_index"]),
            service_request_id=str(raw_receipt["service_request_id"]),
            receipt_sha256=str(raw_receipt["receipt_sha256"]),
            previous_receipt_sha256=(
                str(raw_receipt["previous_receipt_sha256"])
                if raw_receipt.get("previous_receipt_sha256")
                else None
            ),
            call_index=int(raw_receipt["call_index"]),
            replayed=bool(raw_receipt.get("replayed", False)),
            api_contract_sha256=(
                str(raw_receipt["api_contract_sha256"])
                if raw_receipt.get("api_contract_sha256")
                else None
            ),
            implementation_manifest_sha256=(
                str(raw_receipt["implementation_manifest_sha256"])
                if raw_receipt.get("implementation_manifest_sha256")
                else None
            ),
            capability_sha256=(
                str(raw_receipt["capability_sha256"])
                if raw_receipt.get("capability_sha256")
                else None
            ),
            service_cohort_sha256=(
                str(raw_receipt["service_cohort_sha256"])
                if raw_receipt.get("service_cohort_sha256")
                else None
            ),
        )
        if any(
            value != expected
            for value, expected in (
                (receipt.api_contract_sha256, descriptor.api_contract_sha256),
                (
                    receipt.implementation_manifest_sha256,
                    descriptor.implementation_manifest_sha256,
                ),
                (receipt.capability_sha256, descriptor.capability_sha256),
                (receipt.service_cohort_sha256, descriptor.service_cohort_sha256),
            )
        ):
            raise SearchServiceError("service_receipt_identity_mismatch", status_code=502)
        candidates = tuple(
            SearchCandidate(
                url=str(item.get("url")) if item.get("url") else None,
                title=str(item.get("title")) if item.get("title") else None,
                content=str(item.get("content")) if item.get("content") else None,
                published_at=str(item.get("published_at") or "unknown"),
            )
            for item in payload.get("candidates", [])
            if isinstance(item, dict)
        )
        audits = tuple(
            _audit_from_payload(item, descriptor)
            for item in payload.get("audit_records", [])
            if isinstance(item, dict)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SearchServiceError("service_response_invalid", status_code=502) from exc
    return ServiceSearchOutcome(
        task_id=descriptor.task_id,
        framework=descriptor.baseline,
        query_sha256=str(payload.get("query_sha256", "")),
        cutoff_date=descriptor.cutoff_date,
        exclusive_cutoff_bound=str(payload.get("exclusive_cutoff_bound", "")),
        request_filter_sha256=str(payload.get("request_filter_sha256", "")),
        response_sha256=str(payload["response_sha256"]) if payload.get("response_sha256") else None,
        candidates=candidates,
        audit_records=audits,
        status_code=int(payload["status_code"]) if payload.get("status_code") is not None else None,
        request_id=str(payload["service_request_id"])
        if payload.get("service_request_id")
        else None,
        error_code=str(payload["error_code"]) if payload.get("error_code") else None,
        tool_contract_hash=str(payload.get("tool_contract_hash", descriptor.service_cohort_sha256)),
        tool_call_count=receipt.call_index,
        max_tool_calls=descriptor.limits.max_search_calls,
        max_model_calls=4 if descriptor.profile == "agentic_web_search" else 32,
        test_only=bool(payload.get("test_only", False)),
        receipt=receipt,
        filter_review_count=int(payload.get("filter_review_count", 0)),
        filter_model_call_count=int(payload.get("filter_model_call_count", 0)),
        network_ledger_rows=tuple(
            item for item in payload.get("network_ledger", []) if isinstance(item, dict)
        ),
    )


def _audit_from_payload(
    value: dict[str, Any], descriptor: SearchSessionDescriptor
) -> SearchCandidateAudit:
    return SearchCandidateAudit(
        task_id=descriptor.task_id,
        framework=descriptor.baseline,
        query_sha256=str(value.get("query_sha256", "")),
        url=str(value["url"]) if value.get("url") else None,
        title=str(value["title"]) if value.get("title") else None,
        content_sha256=str(value["content_sha256"]) if value.get("content_sha256") else None,
        published_at=str(value["published_at"]) if value.get("published_at") else None,
        accepted=bool(value.get("accepted", False)),
        reason=str(value.get("reason", "unknown")),
        response_sha256=str(value.get("response_sha256", "")),
        cutoff_date=str(value.get("cutoff_date", descriptor.cutoff_date)),
        exclusive_cutoff_bound=str(value.get("exclusive_cutoff_bound", "")),
        request_filter_sha256=str(value.get("request_filter_sha256", "")),
        provider="volc",
        test_only=bool(value.get("test_only", False)),
        time_range=str(value["time_range"]) if value.get("time_range") else None,
        filter_stage=str(value["filter_stage"]) if value.get("filter_stage") else None,
        filter_reason=str(value["filter_reason"]) if value.get("filter_reason") else None,
        rule_signals=tuple(
            str(item) for item in value.get("rule_signals", []) if isinstance(item, str)
        ),
        llm_verdict=str(value["llm_verdict"]) if value.get("llm_verdict") else None,
    )


__all__ = ["SearchServiceClient", "SearchServiceSession", "ServiceSearchOutcome"]
