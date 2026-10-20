"""In-memory session state and hash-only operational WAL for the service."""

from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from benchmark_harness.contracts import sha256_json, sha256_text
from benchmark_harness.profile_registry import get_profile
from benchmark_harness.search.cutoff import CutoffWindow, CutoffWindowError
from benchmark_harness.search.volc import SearchOutcome

from .api import (
    CLOSE_REQUEST_FIELDS,
    DRAIN_REQUEST_FIELDS,
    SEARCH_REQUEST_FIELDS,
    SEARCH_SERVICE_PROTOCOL,
    SESSION_REQUEST_FIELDS,
    SearchServiceError,
    SearchServiceManifest,
    SearchSessionDescriptor,
    ServiceLimits,
    canonical_json,
    hash_receipt,
    require_exact_fields,
    require_safe_request_id,
    require_string,
)
from .config import SearchServiceConfig
from .engine import EngineSearchResult, SearchFilterEngine
from .identities import service_manifest, validate_committed_identities


@dataclass(slots=True)
class ServiceSession:
    descriptor: SearchSessionDescriptor
    token_sha256: str
    created_monotonic: float
    # A re-entrant lock lets the idempotency gate cover the engine call while
    # `_record_search` appends the canonical receipt under the same lock.
    lock: threading.RLock = field(default_factory=threading.RLock)
    idempotency: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    receipts: list[dict[str, Any]] = field(default_factory=list)
    search_audits: list[dict[str, Any]] = field(default_factory=list)
    network_ledger: list[dict[str, Any]] = field(default_factory=list)
    filter_review_count: int = 0
    filter_model_call_count: int = 0
    closed: bool = False
    close_status: str | None = None
    contaminated: bool = False

    @property
    def call_count(self) -> int:
        return len(self.receipts)


class SearchServiceState:
    """Process-local state; restart invalidates all descriptors by instance id."""

    def __init__(
        self,
        config: SearchServiceConfig,
        *,
        engine: SearchFilterEngine | Any | None = None,
        manifest: SearchServiceManifest | None = None,
        instance_id: str | None = None,
        admin_token: str | None = None,
    ) -> None:
        try:
            validate_committed_identities()
        except ValueError as exc:
            raise SearchServiceError("service_identity_invalid", status_code=500) from exc
        self.config = config
        self.instance_id = instance_id or uuid.uuid4().hex
        self.admin_token = admin_token or secrets.token_urlsafe(32)
        self.admin_token_sha256 = sha256_text(self.admin_token)
        self.engine = engine or SearchFilterEngine(test_only=config.test_only)
        filter_profile = getattr(getattr(self.engine, "gateway", None), "_model_profile", None)
        self.manifest = manifest or service_manifest(
            test_only=config.test_only,
            filter_model=(filter_profile.public_dict() if filter_profile is not None else None),
        )
        self.sessions: dict[str, ServiceSession] = {}
        self.open_idempotency: dict[str, tuple[str, dict[str, Any]]] = {}
        self.lock = threading.RLock()
        self.inflight = threading.BoundedSemaphore(config.max_inflight_searches)
        self.pending = threading.BoundedSemaphore(config.max_inflight_searches + config.queue_limit)
        self.draining = False
        self.started_at = datetime.now(UTC).isoformat()
        self._write_secure_runtime_files()

    def _write_secure_runtime_files(self) -> None:
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.config.audit_dir.mkdir(parents=True, exist_ok=True)
        _chmod_directory(self.config.state_dir)
        _chmod_directory(self.config.audit_dir)
        _write_secure_text(self.config.admin_token_file, self.admin_token + "\n")
        state = {
            "protocol": SEARCH_SERVICE_PROTOCOL,
            "instance_id": self.instance_id,
            "pid": os.getpid(),
            "pid_start_marker": _pid_start_marker(os.getpid()),
            "started_at": self.started_at,
            "bind_host": self.config.bind_host,
            "port": self.config.port,
            "api_contract_sha256": self.manifest.api_contract_sha256,
            "implementation_manifest_sha256": self.manifest.implementation_manifest_sha256,
            "capability_sha256": self.manifest.capability_sha256,
            "service_cohort_sha256": self.manifest.service_cohort_sha256,
            "draining": False,
            "active_sessions": 0,
        }
        _write_secure_json(self.config.state_file, state)

    def ready_payload(self) -> dict[str, Any]:
        if self.draining:
            raise SearchServiceError("service_draining", status_code=503)
        try:
            provider = self.engine.preflight()
        except Exception as exc:
            raise SearchServiceError("service_provider_not_ready", status_code=503) from exc
        if not isinstance(provider, dict):
            provider = {}
        return {
            "status": "ready",
            "protocol": SEARCH_SERVICE_PROTOCOL,
            "instance_id": self.instance_id,
            "provider": "volc",
            "test_only": bool(self.config.test_only or provider.get("test_only", False)),
            **self.manifest.as_dict(),
        }

    def open_session(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.draining:
            raise SearchServiceError("service_draining", status_code=503)
        require_exact_fields(body, SESSION_REQUEST_FIELDS, code="session_fields_invalid")
        require_string(body.get("schema_version"), code="schema_version_invalid")
        if body["schema_version"] != SEARCH_SERVICE_PROTOCOL:
            raise SearchServiceError("schema_version_invalid", status_code=400)
        request_id = require_safe_request_id(body.get("request_id"))
        canonical = canonical_json(body)
        with self.lock:
            existing = self.open_idempotency.get(request_id)
            if existing is not None:
                previous_body, previous_response = existing
                if previous_body != canonical:
                    raise SearchServiceError("idempotency_conflict", status_code=409)
                return json.loads(json.dumps(previous_response))
        run_id = _safe_identifier(body.get("run_id"), "run_id_invalid")
        task_id = _safe_identifier(body.get("task_id"), "task_id_invalid")
        profile = require_string(body.get("profile"), code="profile_invalid")
        # The v1 wire key remains named ``baseline``, but session admission
        # treats its value as the canonical search principal / audit identity.
        # Public runner baselines are a separate contract and never widened
        # here; unknown, aliased, or profile-mismatched principals fail before
        # any session, token, WAL, engine, model, or search side effect.
        principal = require_string(body.get("baseline"), code="profile_search_principal_invalid")
        spec = get_profile(profile)
        if not spec.requires_search:
            raise SearchServiceError("profile_baseline_incompatible", status_code=400)
        if not spec.accepts_search_principal(principal):
            raise SearchServiceError("profile_search_principal_incompatible", status_code=400)
        cutoff_date = _valid_cutoff(body.get("cutoff_date"))
        expected_policy = (
            "agentic_web_search/v2" if profile == "agentic_web_search" else "deep_research/v2"
        )
        if body.get("policy_version") != expected_policy:
            raise SearchServiceError("policy_version_mismatch", status_code=412)
        expected_cohort = body.get("expected_service_cohort_sha256")
        if not _is_sha256_hex(expected_cohort):
            raise SearchServiceError("service_cohort_invalid", status_code=400)
        if expected_cohort != self.manifest.service_cohort_sha256:
            raise SearchServiceError("service_cohort_mismatch", status_code=412)
        limits = ServiceLimits.for_profile(profile)
        now = datetime.now(UTC)
        expires = now + timedelta(seconds=limits.ttl_seconds)
        session_id = uuid.uuid4().hex
        task_token = secrets.token_urlsafe(32)
        descriptor = SearchSessionDescriptor(
            session_id=session_id,
            task_token=task_token,
            run_id=run_id,
            task_id=task_id,
            profile=profile,
            baseline=principal,
            cutoff_date=cutoff_date,
            policy_version=body["policy_version"],
            expires_at=expires.isoformat(),
            instance_id=self.instance_id,
            limits=limits,
            api_contract_sha256=self.manifest.api_contract_sha256,
            implementation_manifest_sha256=self.manifest.implementation_manifest_sha256,
            capability_sha256=self.manifest.capability_sha256,
            service_cohort_sha256=self.manifest.service_cohort_sha256,
        )
        response = {
            "protocol": SEARCH_SERVICE_PROTOCOL,
            "session": {
                **descriptor.public_dict(),
                "task_token": task_token,
            },
        }
        with self.lock:
            self.sessions[session_id] = ServiceSession(
                descriptor=descriptor,
                token_sha256=sha256_text(task_token),
                created_monotonic=time.monotonic(),
            )
            self._update_state(active_sessions=self._active_sessions())
            self.open_idempotency[request_id] = (canonical, response)
        self._wal("session_open", descriptor, {"session_id_sha256": sha256_text(session_id)})
        return response

    def search(self, session_id: str, token: str, body: dict[str, Any]) -> dict[str, Any]:
        session = self._task_session(session_id, token)
        require_exact_fields(body, SEARCH_REQUEST_FIELDS, code="search_fields_invalid")
        if body.get("schema_version") != SEARCH_SERVICE_PROTOCOL:
            raise SearchServiceError("schema_version_invalid", status_code=400)
        request_id = require_safe_request_id(body.get("request_id"))
        self._check_binding(session, body)
        call_index = body.get("call_index")
        if not isinstance(call_index, int) or isinstance(call_index, bool) or call_index < 1:
            raise SearchServiceError("call_index_invalid", status_code=400)
        query = require_string(body.get("query"), code="query_invalid").strip()
        if not query or len(query) > 2048:
            raise SearchServiceError("query_invalid", status_code=400)
        canonical = canonical_json(body)
        # Keep the idempotency check and engine invocation in one per-session
        # critical section.  This deliberately serializes calls for a task so
        # two concurrent deliveries of the same request cannot consume two
        # search calls or fork the receipt chain.
        with session.lock:
            existing = session.idempotency.get(request_id)
            if existing is not None:
                previous_body, previous_response = existing
                if previous_body != canonical:
                    session.contaminated = True
                    raise SearchServiceError("idempotency_conflict", status_code=409)
                replay = json.loads(json.dumps(previous_response))
                if isinstance(replay.get("receipt"), dict):
                    replay["receipt"]["replayed"] = True
                return replay
            if session.closed or session.contaminated:
                raise SearchServiceError("session_not_usable", status_code=409)
            if call_index != session.call_count + 1:
                raise SearchServiceError("call_index_out_of_order", status_code=409)
            if session.call_count >= session.descriptor.limits.max_search_calls:
                raise SearchServiceError("search_budget_exceeded", status_code=429)
            if _expired(session.descriptor.expires_at):
                raise SearchServiceError("session_expired", status_code=410)
            if not self.pending.acquire(timeout=0):
                raise SearchServiceError("search_queue_full", status_code=429)
            if not self.inflight.acquire(timeout=self.config.drain_timeout_seconds):
                self.pending.release()
                raise SearchServiceError("search_queue_full", status_code=429)
            try:
                result = self.engine.search(
                    task_id=session.descriptor.task_id,
                    framework=session.descriptor.baseline,
                    query=query,
                    cutoff_date=session.descriptor.cutoff_date,
                    call_index=call_index,
                )
                normalized = _normalize_engine_result(result)
                if normalized.filter_review_count < 0 or normalized.filter_model_call_count < 0:
                    raise SearchServiceError("filter_budget_evidence_invalid", status_code=502)
                if len(normalized.outcome.candidates) > session.descriptor.limits.max_candidates:
                    raise SearchServiceError("candidate_budget_exceeded", status_code=502)
                if (
                    session.filter_review_count + normalized.filter_review_count
                    > session.descriptor.limits.filter_review_cap
                ):
                    raise SearchServiceError("filter_review_budget_exceeded", status_code=429)
                if (
                    session.filter_model_call_count + normalized.filter_model_call_count
                    > session.descriptor.limits.filter_attempt_cap
                ):
                    raise SearchServiceError("filter_attempt_budget_exceeded", status_code=429)
                response = self._record_search(
                    session, request_id, canonical, call_index, normalized
                )
                return response
            except SearchServiceError:
                raise
            except Exception as exc:
                raise SearchServiceError("search_failed", status_code=502) from exc
            finally:
                self.inflight.release()
                self.pending.release()

    def close(self, session_id: str, body: dict[str, Any]) -> dict[str, Any]:
        require_exact_fields(body, CLOSE_REQUEST_FIELDS, code="close_fields_invalid")
        if body.get("schema_version") != SEARCH_SERVICE_PROTOCOL:
            raise SearchServiceError("schema_version_invalid", status_code=400)
        require_safe_request_id(body.get("request_id"))
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise SearchServiceError("session_not_found", status_code=404)
        self._check_binding(session, body)
        status = body.get("status")
        if status not in {"completed", "aborted"}:
            raise SearchServiceError("close_status_invalid", status_code=400)
        receipt_hashes = body.get("receipt_sha256s")
        if not isinstance(receipt_hashes, list) or any(
            not isinstance(item, str) for item in receipt_hashes
        ):
            raise SearchServiceError("receipt_hashes_invalid", status_code=400)
        with session.lock:
            expected = [str(row["receipt_sha256"]) for row in session.receipts]
            if receipt_hashes != expected:
                session.contaminated = True
                raise SearchServiceError("receipt_chain_mismatch", status_code=409)
            session.closed = True
            session.close_status = status
            comparable = not session.contaminated and status == "completed"
            payload = {
                "protocol": SEARCH_SERVICE_PROTOCOL,
                "session": session.descriptor.public_dict(),
                "status": status,
                "comparable": comparable,
                "receipt_count": len(session.receipts),
                "receipts": list(session.receipts),
                "search_audit": list(session.search_audits),
                "network_ledger": list(session.network_ledger),
                "receipt_chain_head": expected[-1] if expected else None,
                "service_manifest": self.manifest.as_dict(),
            }
            self._update_state(active_sessions=self._active_sessions())
        self._wal(
            "session_close", session.descriptor, {"receipt_count": len(expected), "status": status}
        )
        return payload

    def drain(self, body: dict[str, Any]) -> dict[str, Any]:
        require_exact_fields(body, DRAIN_REQUEST_FIELDS, code="drain_fields_invalid")
        if body.get("schema_version") != SEARCH_SERVICE_PROTOCOL:
            raise SearchServiceError("schema_version_invalid", status_code=400)
        require_safe_request_id(body.get("request_id"))
        with self.lock:
            self.draining = True
            active = self._active_sessions()
            self._update_state(draining=True, active_sessions=active)
        return {
            "protocol": SEARCH_SERVICE_PROTOCOL,
            "status": "draining",
            "active_sessions": active,
        }

    def _task_session(self, session_id: str, token: str) -> ServiceSession:
        if not isinstance(session_id, str) or not session_id:
            raise SearchServiceError("session_id_invalid", status_code=400)
        with self.lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise SearchServiceError("session_not_found", status_code=404)
        if not hmac.compare_digest(session.token_sha256, sha256_text(token)):
            raise SearchServiceError("task_token_invalid", status_code=401)
        if session.descriptor.instance_id != self.instance_id:
            raise SearchServiceError("service_instance_mismatch", status_code=412)
        return session

    @staticmethod
    def _check_binding(session: ServiceSession, body: dict[str, Any]) -> None:
        for key in ("session_id", "run_id", "task_id", "profile", "baseline", "cutoff_date"):
            expected = (
                session.descriptor.session_id
                if key == "session_id"
                else getattr(session.descriptor, key)
            )
            if body.get(key) != expected:
                raise SearchServiceError("session_binding_mismatch", status_code=409)

    def _record_search(
        self,
        session: ServiceSession,
        request_id: str,
        canonical: str,
        call_index: int,
        result: EngineSearchResult,
    ) -> dict[str, Any]:
        outcome = result.outcome
        previous = session.receipts[-1]["receipt_sha256"] if session.receipts else None
        receipt_base = {
            "protocol": SEARCH_SERVICE_PROTOCOL,
            "session_id": session.descriptor.session_id,
            "run_id": session.descriptor.run_id,
            "task_id": session.descriptor.task_id,
            "profile": session.descriptor.profile,
            "baseline": session.descriptor.baseline,
            "cutoff_date": session.descriptor.cutoff_date,
            "request_id": request_id,
            "call_index": call_index,
            "query_sha256": outcome.query_sha256,
            "response_sha256": outcome.response_sha256,
            "previous_receipt_sha256": previous,
            "audit_sha256": sha256_json([audit.as_dict() for audit in outcome.audit_records]),
            "network_sha256": sha256_json(
                [
                    outcome.network_ledger(profile=session.descriptor.profile),
                    *[
                        {
                            "host": event.host,
                            "method": event.method,
                            "path": event.path,
                            "body_hash": event.body_hash,
                            "status_code": event.status_code,
                            "latency_seconds": event.latency_seconds,
                            "is_model_endpoint": event.is_model_endpoint,
                            "network_class": event.network_class,
                        }
                        for event in result.network_events
                    ],
                ]
            ),
            "service_cohort_sha256": session.descriptor.service_cohort_sha256,
        }
        receipt_hash = hash_receipt(receipt_base)
        receipt = {
            "receipt_index": len(session.receipts) + 1,
            "service_request_id": request_id,
            "receipt_sha256": receipt_hash,
            "previous_receipt_sha256": previous,
            "call_index": call_index,
            "replayed": False,
            "api_contract_sha256": session.descriptor.api_contract_sha256,
            "implementation_manifest_sha256": session.descriptor.implementation_manifest_sha256,
            "capability_sha256": session.descriptor.capability_sha256,
            "service_cohort_sha256": session.descriptor.service_cohort_sha256,
        }
        audits = [
            {
                **audit.as_dict(),
                "run_id": session.descriptor.run_id,
                "profile": session.descriptor.profile,
                "baseline": session.descriptor.baseline,
                "session_id": session.descriptor.session_id,
                "service_request_id": request_id,
                "call_index": call_index,
                "receipt_sha256": receipt_hash,
                "api_contract_sha256": session.descriptor.api_contract_sha256,
                "implementation_manifest_sha256": session.descriptor.implementation_manifest_sha256,
                "capability_sha256": session.descriptor.capability_sha256,
                "service_cohort_sha256": session.descriptor.service_cohort_sha256,
            }
            for audit in outcome.audit_records
        ]
        ledger = [
            {
                **outcome.network_ledger(profile=session.descriptor.profile),
                "run_id": session.descriptor.run_id,
                "task_id": session.descriptor.task_id,
                "session_id": session.descriptor.session_id,
                "network_class": "web_search",
                "service_request_id": request_id,
                "call_index": call_index,
                "receipt_sha256": receipt_hash,
            }
        ]
        for event in result.network_events:
            ledger.append(
                {
                    "run_id": session.descriptor.run_id,
                    "task_id": session.descriptor.task_id,
                    "session_id": session.descriptor.session_id,
                    "network_class": event.network_class or "temporal_filter",
                    "host": event.host,
                    "method": event.method,
                    "path": event.path,
                    "body_hash": event.body_hash,
                    "status_code": event.status_code,
                    "latency_seconds": event.latency_seconds,
                    "is_model_endpoint": event.is_model_endpoint,
                    "service_request_id": request_id,
                    "call_index": call_index,
                    "receipt_sha256": receipt_hash,
                }
            )
        response = {
            "protocol": SEARCH_SERVICE_PROTOCOL,
            "session_id": session.descriptor.session_id,
            "service_request_id": request_id,
            "call_index": call_index,
            "query_sha256": outcome.query_sha256,
            "response_sha256": outcome.response_sha256,
            "request_filter_sha256": outcome.request_filter_sha256,
            "exclusive_cutoff_bound": outcome.exclusive_cutoff_bound,
            "status_code": outcome.status_code,
            "error_code": outcome.error_code,
            "candidates": [candidate.tool_payload() for candidate in outcome.candidates],
            "audit_records": [audit.as_dict() for audit in outcome.audit_records],
            "network_ledger": ledger,
            "filter_review_count": result.filter_review_count,
            "filter_model_call_count": result.filter_model_call_count,
            "tool_contract_hash": outcome.tool_contract_hash,
            "test_only": outcome.test_only,
            "receipt": receipt,
            "budget": {
                "search_calls": call_index,
                "max_search_calls": session.descriptor.limits.max_search_calls,
                "filter_review_count": session.filter_review_count + result.filter_review_count,
                "filter_review_cap": session.descriptor.limits.filter_review_cap,
                "filter_model_call_count": (
                    session.filter_model_call_count + result.filter_model_call_count
                ),
                "filter_attempt_cap": session.descriptor.limits.filter_attempt_cap,
            },
        }
        with session.lock:
            session.receipts.append(receipt)
            session.search_audits.extend(audits)
            session.network_ledger.extend(ledger)
            session.filter_review_count += result.filter_review_count
            session.filter_model_call_count += result.filter_model_call_count
            session.idempotency[request_id] = (canonical, response)
        self._wal(
            "search",
            session.descriptor,
            {
                "request_id_sha256": sha256_text(request_id),
                "query_sha256": outcome.query_sha256,
                "receipt_sha256": receipt_hash,
                "call_index": call_index,
                "accepted_count": len(outcome.candidates),
                "filter_model_call_count": result.filter_model_call_count,
            },
        )
        return response

    def _wal(self, event: str, descriptor: SearchSessionDescriptor, values: dict[str, Any]) -> None:
        row = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "instance_id": self.instance_id,
            "run_id_sha256": sha256_text(descriptor.run_id),
            "task_id_sha256": sha256_text(descriptor.task_id),
            "session_id_sha256": sha256_text(descriptor.session_id),
            **values,
        }
        path = self.config.audit_dir / "service.wal.jsonl"
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            _chmod_file(path)
        except OSError as exc:
            raise SearchServiceError("service_audit_unavailable", status_code=503) from exc

    def _active_sessions(self) -> int:
        return sum(not session.closed for session in self.sessions.values())

    def _update_state(
        self, *, draining: bool | None = None, active_sessions: int | None = None
    ) -> None:
        try:
            state = json.loads(self.config.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state = {}
        if draining is not None:
            state["draining"] = draining
        if active_sessions is not None:
            state["active_sessions"] = active_sessions
        _write_secure_json(self.config.state_file, state)


def _normalize_engine_result(value: Any) -> EngineSearchResult:
    if isinstance(value, EngineSearchResult):
        return value
    if isinstance(value, SearchOutcome):
        return EngineSearchResult(outcome=value)
    outcome = getattr(value, "outcome", None)
    if isinstance(outcome, SearchOutcome):
        return EngineSearchResult(
            outcome=outcome,
            network_events=tuple(getattr(value, "network_events", ())),
            filter_review_count=int(getattr(value, "filter_review_count", 0)),
            filter_model_call_count=int(getattr(value, "filter_model_call_count", 0)),
        )
    raise SearchServiceError("engine_result_invalid", status_code=502)


def _is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_identifier(value: Any, code: str) -> str:
    text = require_string(value, code=code)
    if len(text) > 256 or any(ch in text for ch in "\r\n/\\"):
        raise SearchServiceError(code, status_code=400)
    return text


def _valid_cutoff(value: Any) -> str:
    cutoff = require_string(value, code="cutoff_date_invalid")
    try:
        window = CutoffWindow.from_seed(cutoff)
    except CutoffWindowError as exc:
        raise SearchServiceError("cutoff_date_invalid", status_code=400) from exc
    return window.cutoff_date


def _expired(value: str) -> bool:
    try:
        return datetime.fromisoformat(value).timestamp() <= time.time()
    except ValueError:
        return True


def _pid_start_marker(pid: int) -> str:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = stat.rsplit(")", 1)[-1].split()
        return fields[19] if len(fields) > 19 else str(pid)
    except OSError:
        return str(pid)


def _write_secure_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    _chmod_file(path)


def _write_secure_json(path: Path, value: dict[str, Any]) -> None:
    _write_secure_text(path, canonical_json(value) + "\n")


def _chmod_file(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _chmod_directory(path: Path) -> None:
    try:
        path.chmod(0o700)
    except OSError:
        pass


__all__ = ["SearchServiceState", "ServiceSession"]
