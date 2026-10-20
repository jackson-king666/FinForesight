"""Committed service capability and cohort identities."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmark_harness.contracts import sha256_json

from .api import SEARCH_SERVICE_PROTOCOL, SearchServiceManifest

PACKAGE_DIR = Path(__file__).resolve().parent
API_PATH = PACKAGE_DIR / "SEARCH_SERVICE_API.json"
MANIFEST_PATH = PACKAGE_DIR / "SEARCH_SERVICE_MANIFEST.json"

CAPABILITY_SPEC: dict[str, Any] = {
    "protocol": SEARCH_SERVICE_PROTOCOL,
    "provider": "volc",
    "host": "mercury.volcengineapi.com",
    "action": "WebSearch",
    "version": "2025-01-01",
    "service": "volc_torchlight_api",
    "region": "cn-beijing",
    "count": 10,
    "filter": {"NeedContent": True, "NeedUrl": True},
    "lookback_years": 5,
    "cutoff": {
        "time_range": "inclusive_cutoff_date",
        "audit_bound": "next_day_00:00:00Z_exclusive",
    },
    "filter_policy": "rule_then_llm_fail_closed",
    "network_policy": "no_fetch_no_fallback_volc_only",
}

FILTER_MODEL_IDENTITY_FIELDS = (
    "mode",
    "endpoint",
    "model_name",
    "temperature",
    "max_tokens",
    "timeout_seconds",
    "retries",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"service identity must be an object: {path}")
    return value


def filter_model_identity(filter_model: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the credential-free filter configuration bound to a cohort."""

    if not isinstance(filter_model, Mapping):
        return {}
    return {key: filter_model[key] for key in FILTER_MODEL_IDENTITY_FIELDS if key in filter_model}


def service_manifest(
    *, test_only: bool = False, filter_model: Mapping[str, Any] | None = None
) -> SearchServiceManifest:
    api = _read_json(API_PATH)
    implementation = _read_json(MANIFEST_PATH)
    api_hash = sha256_json(api)
    implementation_hash = sha256_json(implementation)
    capability_hash = sha256_json(CAPABILITY_SPEC)
    public_filter = filter_model_identity(filter_model)
    cohort_hash = sha256_json(
        {
            "protocol": SEARCH_SERVICE_PROTOCOL,
            "api_contract_sha256": api_hash,
            "implementation_manifest_sha256": implementation_hash,
            "capability_sha256": capability_hash,
            "filter_model": public_filter,
            "test_only": bool(test_only),
        }
    )
    return SearchServiceManifest(
        api_contract_sha256=api_hash,
        implementation_manifest_sha256=implementation_hash,
        capability_sha256=capability_hash,
        service_cohort_sha256=cohort_hash,
    )


def validate_committed_identities() -> None:
    """Fail closed if the packaged route/capability identities drift apart."""

    api = _read_json(API_PATH)
    implementation = _read_json(MANIFEST_PATH)
    if api.get("protocol") != SEARCH_SERVICE_PROTOCOL:
        raise ValueError("search service API identity protocol mismatch")
    if implementation.get("protocol") != SEARCH_SERVICE_PROTOCOL:
        raise ValueError("search service implementation identity protocol mismatch")
    routes = api.get("routes")
    required_routes = {
        "GET /healthz",
        "GET /readyz",
        "POST /v1/sessions",
        "POST /v1/sessions/{session_id}/search",
        "POST /v1/sessions/{session_id}/close",
        "POST /v1/admin/drain",
    }
    if not isinstance(routes, dict) or not required_routes <= set(routes):
        raise ValueError("search service API route identity is incomplete")
    forbidden = implementation.get("forbidden")
    if not isinstance(forbidden, list) or {
        "alternate_provider",
        "direct_fetch",
        "in_process_fallback",
        "cross_session_cache",
    } - set(forbidden):
        raise ValueError("search service forbidden capability identity is incomplete")


__all__ = [
    "API_PATH",
    "CAPABILITY_SPEC",
    "FILTER_MODEL_IDENTITY_FIELDS",
    "MANIFEST_PATH",
    "filter_model_identity",
    "service_manifest",
    "validate_committed_identities",
]
