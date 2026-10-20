"""Closed, versioned JSON shapes for the resident Search+Filter service."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

SEARCH_SERVICE_PROTOCOL = "search_filter_http/v1"
SEARCH_SERVICE_BASE_URL = "http://127.0.0.1:8765"
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

SESSION_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "request_id",
        "run_id",
        "task_id",
        "profile",
        "baseline",
        "cutoff_date",
        "policy_version",
        "expected_service_cohort_sha256",
    }
)
SEARCH_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "request_id",
        "session_id",
        "run_id",
        "task_id",
        "profile",
        "baseline",
        "cutoff_date",
        "call_index",
        "query",
    }
)
CLOSE_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "request_id",
        "session_id",
        "run_id",
        "task_id",
        "profile",
        "baseline",
        "cutoff_date",
        "status",
        "receipt_sha256s",
    }
)
DRAIN_REQUEST_FIELDS = frozenset({"schema_version", "request_id"})


class SearchServiceError(RuntimeError):
    """A safe service/client error with no secret or content-bearing message."""

    def __init__(self, code: str, *, status_code: int = 503) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ServiceLimits:
    """Server-owned limits resolved from the profile, never client supplied."""

    max_search_calls: int
    filter_review_cap: int
    filter_attempt_cap: int
    ttl_seconds: int
    max_candidates: int = 10

    @classmethod
    def for_profile(cls, profile: str) -> ServiceLimits:
        if profile == "agentic_web_search":
            return cls(3, 30, 90, 3600)
        if profile == "deep_research":
            return cls(8, 80, 240, 1860)
        raise SearchServiceError("profile_not_supported", status_code=400)

    def as_dict(self) -> dict[str, int]:
        return {
            "max_search_calls": self.max_search_calls,
            "filter_review_cap": self.filter_review_cap,
            "filter_attempt_cap": self.filter_attempt_cap,
            "ttl_seconds": self.ttl_seconds,
            "max_candidates": self.max_candidates,
        }


@dataclass(frozen=True, slots=True)
class SearchSessionDescriptor:
    """Opaque task capability returned once when a session is opened."""

    session_id: str
    task_token: str
    run_id: str
    task_id: str
    profile: str
    baseline: str
    cutoff_date: str
    policy_version: str
    expires_at: str
    instance_id: str
    limits: ServiceLimits
    api_contract_sha256: str
    implementation_manifest_sha256: str
    capability_sha256: str
    service_cohort_sha256: str

    def public_dict(self) -> dict[str, Any]:
        """Return the descriptor without the bearer, which is never persisted."""

        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "profile": self.profile,
            "baseline": self.baseline,
            "cutoff_date": self.cutoff_date,
            "policy_version": self.policy_version,
            "expires_at": self.expires_at,
            "instance_id": self.instance_id,
            "limits": self.limits.as_dict(),
            "api_contract_sha256": self.api_contract_sha256,
            "implementation_manifest_sha256": self.implementation_manifest_sha256,
            "capability_sha256": self.capability_sha256,
            "service_cohort_sha256": self.service_cohort_sha256,
        }


@dataclass(frozen=True, slots=True)
class SearchReceiptRef:
    """A client-visible reference to one canonical service receipt."""

    receipt_index: int
    service_request_id: str
    receipt_sha256: str
    previous_receipt_sha256: str | None
    call_index: int
    replayed: bool = False
    api_contract_sha256: str | None = None
    implementation_manifest_sha256: str | None = None
    capability_sha256: str | None = None
    service_cohort_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "receipt_index": self.receipt_index,
            "service_request_id": self.service_request_id,
            "receipt_sha256": self.receipt_sha256,
            "previous_receipt_sha256": self.previous_receipt_sha256,
            "call_index": self.call_index,
            "replayed": self.replayed,
            "api_contract_sha256": self.api_contract_sha256,
            "implementation_manifest_sha256": self.implementation_manifest_sha256,
            "capability_sha256": self.capability_sha256,
            "service_cohort_sha256": self.service_cohort_sha256,
        }


@dataclass(frozen=True, slots=True)
class SearchServiceManifest:
    """Committed service identities used in cohort gates and artifacts."""

    api_contract_sha256: str
    implementation_manifest_sha256: str
    capability_sha256: str
    service_cohort_sha256: str
    protocol: str = SEARCH_SERVICE_PROTOCOL

    def as_dict(self) -> dict[str, str]:
        return {
            "protocol": self.protocol,
            "api_contract_sha256": self.api_contract_sha256,
            "implementation_manifest_sha256": self.implementation_manifest_sha256,
            "capability_sha256": self.capability_sha256,
            "service_cohort_sha256": self.service_cohort_sha256,
        }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_receipt(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def require_exact_fields(value: Any, fields: frozenset[str], *, code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise SearchServiceError(code, status_code=400)
    return value


def require_string(value: Any, *, code: str, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise SearchServiceError(code, status_code=400)
    return value


def require_safe_request_id(value: Any) -> str:
    request_id = require_string(value, code="request_id_invalid")
    if len(request_id) > 128 or any(ch.isspace() for ch in request_id):
        raise SearchServiceError("request_id_invalid", status_code=400)
    return request_id


def redact_error_body(code: str) -> dict[str, Any]:
    return {"error": {"code": code, "protocol": SEARCH_SERVICE_PROTOCOL}}


__all__ = [
    "CLOSE_REQUEST_FIELDS",
    "DRAIN_REQUEST_FIELDS",
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
    "SEARCH_REQUEST_FIELDS",
    "SEARCH_SERVICE_BASE_URL",
    "SEARCH_SERVICE_PROTOCOL",
    "SESSION_REQUEST_FIELDS",
    "SearchReceiptRef",
    "SearchServiceError",
    "SearchServiceManifest",
    "SearchSessionDescriptor",
    "ServiceLimits",
    "canonical_json",
    "hash_receipt",
    "redact_error_body",
    "require_exact_fields",
    "require_safe_request_id",
    "require_string",
]
