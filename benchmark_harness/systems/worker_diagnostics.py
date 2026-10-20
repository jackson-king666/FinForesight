"""Typed, secret-free diagnostics for disposable external workers.

The worker side may report a small allow-listed diagnostic when it can classify
an expected failure.  The host validates the complete shape before persisting
it.  This module deliberately contains no model, prompt, transport, or
provider-specific response data.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

ODR_WORKER_ERROR_SCHEMA = "worker_error/v1"
ODR_COMPONENT = "odr"
ODR_STRUCTURED_OUTPUT_CODE = "odr_structured_model_response_invalid"
ODR_STRUCTURED_OUTPUT_SCHEMA = "ResearchQuestion"
ODR_STRUCTURED_OUTPUT_STAGE = "write_research_brief"

_DIAGNOSTIC_FIELDS = frozenset(
    {
        "schema_version",
        "component",
        "stage",
        "code",
        "schema_name",
        "structured_attempt",
        "broker_call_ordinal",
        "response_bytes",
        "response_sha256",
    }
)
_ALLOWED_CODES = frozenset({ODR_STRUCTURED_OUTPUT_CODE})
_ALLOWED_SCHEMAS = frozenset({ODR_STRUCTURED_OUTPUT_SCHEMA})
_ALLOWED_STAGES = frozenset({ODR_STRUCTURED_OUTPUT_STAGE})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ATTEMPTS = 3
_MAX_BROKER_CALL_ORDINAL = 256
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_FAILURE_SUBREASONS = frozenset(
    {
        "parent_closed_before_completion",
        "parent_task_id_mismatch",
        "parent_binding_hash_mismatch",
        "parent_sequence_not_monotonic",
        "worker_channel_not_bound",
        "worker_write_failed",
        "task_start_required",
        "task_start_protocol_mismatch",
        "task_start_adapter_mismatch",
        "declared_clock_missing",
        "declared_clock_mismatch",
        "service_receipt_missing",
        "model_response_content_missing",
        "model_response_choices_missing",
        "model_response_choice_malformed",
        "model_response_mismatch",
        "search_response_mismatch",
        "search_response_results_missing",
        "odr_search_query_invalid",
        "odr_search_tool_call_invalid",
        "odr_model_response_invalid",
        "odr_task_bound_prediction_missing",
        "gpt_task_bound_prediction_missing",
        "search_service_ready_timeout",
        "search_service_required",
        "service_identity_mismatch",
        "model_transport_error",
        "adapter_preflight_failed",
        "adapter_result_missing_raw_text",
        "adapter_result_empty_raw_text",
        "worker_eof",
        "worker_timeout",
        "worker_sequence_not_monotonic",
        "worker_binding_mismatch",
        ODR_STRUCTURED_OUTPUT_CODE,
    }
)


class WorkerDiagnosticError(ValueError):
    """Raised when a worker diagnostic is not an allow-listed object."""


def build_odr_structured_output_diagnostic(
    *,
    schema_name: str,
    structured_attempt: int,
    broker_call_ordinal: int,
    response: str,
    stage: str = ODR_STRUCTURED_OUTPUT_STAGE,
) -> dict[str, Any]:
    """Build the only ODR worker diagnostic currently allowed to cross IPC."""

    if schema_name not in _ALLOWED_SCHEMAS:
        raise WorkerDiagnosticError("odr diagnostic schema is not allow-listed")
    if stage not in _ALLOWED_STAGES:
        raise WorkerDiagnosticError("odr diagnostic stage is not allow-listed")
    if (
        not isinstance(structured_attempt, int)
        or isinstance(structured_attempt, bool)
        or not 1 <= structured_attempt <= _MAX_ATTEMPTS
    ):
        raise WorkerDiagnosticError("odr diagnostic attempt is invalid")
    if (
        not isinstance(broker_call_ordinal, int)
        or isinstance(broker_call_ordinal, bool)
        or not 1 <= broker_call_ordinal <= _MAX_BROKER_CALL_ORDINAL
    ):
        raise WorkerDiagnosticError("odr diagnostic broker ordinal is invalid")
    if not isinstance(response, str):
        raise WorkerDiagnosticError("odr diagnostic response is invalid")
    response_bytes = len(response.encode("utf-8"))
    if response_bytes > _MAX_RESPONSE_BYTES:
        raise WorkerDiagnosticError("odr diagnostic response is too large")
    return {
        "schema_version": ODR_WORKER_ERROR_SCHEMA,
        "component": ODR_COMPONENT,
        "stage": stage,
        "code": ODR_STRUCTURED_OUTPUT_CODE,
        "schema_name": schema_name,
        "structured_attempt": structured_attempt,
        "broker_call_ordinal": broker_call_ordinal,
        "response_bytes": response_bytes,
        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
    }


def validate_worker_diagnostic(value: Any) -> dict[str, Any]:
    """Validate and copy an allow-listed worker diagnostic at the host seam."""

    if not isinstance(value, Mapping) or set(value) != _DIAGNOSTIC_FIELDS:
        raise WorkerDiagnosticError("worker diagnostic fields are invalid")
    expected_strings = {
        "schema_version": ODR_WORKER_ERROR_SCHEMA,
        "component": ODR_COMPONENT,
        "stage": None,
        "code": None,
        "schema_name": None,
    }
    for key, expected in expected_strings.items():
        actual = value.get(key)
        if not isinstance(actual, str) or not actual.strip():
            raise WorkerDiagnosticError(f"worker diagnostic {key} is invalid")
        if expected is not None and actual != expected:
            raise WorkerDiagnosticError(f"worker diagnostic {key} is invalid")
    if value["stage"] not in _ALLOWED_STAGES:
        raise WorkerDiagnosticError("worker diagnostic stage is not allow-listed")
    if value["code"] not in _ALLOWED_CODES:
        raise WorkerDiagnosticError("worker diagnostic code is not allow-listed")
    if value["schema_name"] not in _ALLOWED_SCHEMAS:
        raise WorkerDiagnosticError("worker diagnostic schema is not allow-listed")

    integer_limits = {
        "structured_attempt": (1, _MAX_ATTEMPTS),
        "broker_call_ordinal": (1, _MAX_BROKER_CALL_ORDINAL),
        "response_bytes": (0, _MAX_RESPONSE_BYTES),
    }
    for key, (lower, upper) in integer_limits.items():
        actual = value[key]
        if not isinstance(actual, int) or isinstance(actual, bool) or not lower <= actual <= upper:
            raise WorkerDiagnosticError(f"worker diagnostic {key} is invalid")
    response_hash = value["response_sha256"]
    if not isinstance(response_hash, str) or not _SHA256.fullmatch(response_hash):
        raise WorkerDiagnosticError("worker diagnostic response hash is invalid")
    return dict(value)


def validate_worker_failure_subreason(value: Any) -> str:
    """Validate the bounded child failure token carried on ``task.error``."""

    if not isinstance(value, str) or value not in _FAILURE_SUBREASONS:
        raise WorkerDiagnosticError("worker failure subreason is not allow-listed")
    return value


def worker_failure_subreason(error: BaseException | str) -> str | None:
    """Return one bounded, secret-free reason from an external-worker failure.

    Boundary code may wrap the original error several times.  Inspect only
    allow-listed diagnostic codes and exact reason tokens; never persist an
    arbitrary provider message or exception text as a ledger field.
    """

    pending: list[BaseException | str] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if isinstance(current, BaseException):
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            diagnostic = getattr(current, "diagnostic", None)
            if isinstance(diagnostic, Mapping):
                for key in ("sub_reason", "code"):
                    code = diagnostic.get(key)
                    if isinstance(code, str) and code in _FAILURE_SUBREASONS:
                        return code
            text = str(current)
            cause = current.__cause__ or current.__context__
            if cause is not None:
                pending.append(cause)
        else:
            text = current
        for reason in _FAILURE_SUBREASONS:
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(reason)}(?![A-Za-z0-9_])", text):
                return reason
    return None


__all__ = [
    "ODR_COMPONENT",
    "ODR_STRUCTURED_OUTPUT_CODE",
    "ODR_STRUCTURED_OUTPUT_SCHEMA",
    "ODR_STRUCTURED_OUTPUT_STAGE",
    "ODR_WORKER_ERROR_SCHEMA",
    "WorkerDiagnosticError",
    "build_odr_structured_output_diagnostic",
    "validate_worker_failure_subreason",
    "validate_worker_diagnostic",
    "worker_failure_subreason",
]
