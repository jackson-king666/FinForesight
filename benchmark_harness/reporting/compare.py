"""Cross-baseline reporting from persisted, locally auditable artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from benchmark_harness.artifacts import ArtifactStore
from benchmark_harness.contracts import TEMPORAL_FILTER_NETWORK_CLASS
from benchmark_harness.profile_registry import ProfileRegistryError, get_profile

COMMON_CAVEATS = [
    "A2-H may use fundamentals_match_mode=prototype_fallback_nearest; inspect per-task metadata.",
    "Strict, agentic web-search, and FinanceHarness deep-research profiles are separate "
    "populations.",
]

INTEGRATION_COST = {
    "pydantic-ai": {"dependency_footprint": "pydantic-ai-slim[openai]==1.104.0", "cost": "low"},
    "langgraph": {
        "dependency_footprint": "langgraph==1.2.9 + langchain-openai",
        "cost": "medium",
    },
    "direct_transport": {"dependency_footprint": "stdlib urllib", "cost": "control-only"},
}

_REQUIRED_ARTIFACTS = (
    "predictions.jsonl",
    "metrics_summary.json",
    "run_config.json",
    "provenance.json",
    "trace_events.jsonl",
    "effective_requests.jsonl",
    "network_ledger.jsonl",
)
_TRACE_EVENTS = {"task_started", "model_request", "model_response", "task_finished"}
_SKIPPED_TRACE_EVENTS = {"task_started", "task_finished"}
_AGENTIC_PROFILE = "agentic_web_search"
_DEEP_PROFILE = "deep_research"
_VOLC_HOST = "mercury.volcengineapi.com"
_VOLC_PATH = "/"
_VOLC_ACTION = "WebSearch"
_VOLC_VERSION = "2025-01-01"
_MAX_AGENTIC_TOOL_CALLS = 3
_MAX_AGENTIC_MODEL_CALLS = 4
_MAX_DEEP_MODEL_CALLS = 32
_MAX_DEEP_SEARCH_CALLS = 8
_MAX_DEEP_VISIT_CALLS = 12
_MAX_DEEP_TOOL_CALLS = 40


class ReportMixError(ValueError):
    """Raised when a report would compare incompatible experiment populations."""


def _read_json_with_status(path: Path) -> tuple[dict[str, Any], bool]:
    """Read an object artifact without letting a broken report hide its absence."""

    if not path.is_file():
        return {}, False
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}, False
    return (value, True) if isinstance(value, dict) else ({}, False)


def _read_jsonl_with_status(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Return rows plus whether the complete JSONL artifact was readable."""

    if not path.is_file():
        return [], False
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    return rows, False
                rows.append(value)
    except (OSError, json.JSONDecodeError):
        return rows, False
    return rows, True


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Compatibility helper for consumers that only need readable rows."""

    return _read_jsonl_with_status(path)[0]


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _is_task_id(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _task_groups(rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_rows = 0
    for row in rows:
        task_id = row.get("task_id")
        if not _is_task_id(task_id):
            invalid_rows += 1
            continue
        groups[task_id].append(row)
    return groups, invalid_rows


def _trace_completeness(trace_rows: list[dict[str, Any]], task_ids: set[str]) -> float | None:
    if not task_ids:
        return None
    events: dict[str, set[str]] = defaultdict(set)
    for row in trace_rows:
        task_id = row.get("task_id")
        event_type = row.get("event_type")
        if _is_task_id(task_id) and isinstance(event_type, str):
            events[task_id].add(event_type)
    complete = sum(1 for task_id in task_ids if _TRACE_EVENTS <= events[task_id])
    return _rate(complete, len(task_ids))


def _request_has_policy_evidence(request: dict[str, Any]) -> bool:
    policy = request.get("policy")
    return (
        isinstance(policy, dict)
        and isinstance(policy.get("comparable"), bool)
        and isinstance(policy.get("violations"), list)
    )


def _request_is_complete(request: dict[str, Any], prediction: dict[str, Any]) -> bool:
    """Validate evidence persisted by the runner, including hash continuity."""

    if not _request_has_policy_evidence(request):
        return False
    if not isinstance(request.get("prompt_sha256"), str):
        return False
    if not isinstance(request.get("messages_sha256"), str):
        return False
    for key in ("canonical_input_hash", "effective_input_hash"):
        request_value = request.get(key)
        prediction_value = prediction.get(key)
        if (
            not isinstance(request_value, str)
            or not isinstance(prediction_value, str)
            or request_value != prediction_value
        ):
            return False
    if request.get("messages_sha256") != request.get("effective_input_hash"):
        return False
    if request.get("canonical_input_hash") != request.get("effective_input_hash"):
        return False
    policy = request["policy"]
    if policy["comparable"] is not prediction.get("comparable"):
        return False
    if policy["violations"] != prediction.get("policy_violations"):
        return False
    return True


def _valid_network_event(row: dict[str, Any]) -> bool:
    return (
        isinstance(row.get("is_model_endpoint"), bool)
        and all(
            isinstance(row.get(key), str) and row[key]
            for key in ("host", "method", "path", "body_hash")
        )
        and (row.get("status_code") is None or isinstance(row.get("status_code"), int))
        and (
            row.get("latency_seconds") is None
            or isinstance(row.get("latency_seconds"), int | float)
        )
    )


def _response_present(row: dict[str, Any]) -> bool:
    """Respect hash-only predictions, which intentionally omit raw response text."""

    if "response_present" in row:
        return row.get("response_present") is True
    return bool(row.get("raw_response"))


def _all_rate_one(*rates: float | None) -> bool:
    return all(rate == 1.0 for rate in rates)


def _input_policy(run_config: dict[str, Any], provenance: dict[str, Any]) -> str | None:
    execution = run_config.get("execution")
    policy = execution.get("input_policy") if isinstance(execution, dict) else None
    if isinstance(policy, str):
        return policy
    policy = run_config.get("input_policy")
    if isinstance(policy, str):
        return policy
    policy = provenance.get("input_policy")
    return policy if isinstance(policy, str) else None


def _policy_matches_prediction(request: dict[str, Any], prediction: dict[str, Any]) -> bool:
    if not _request_has_policy_evidence(request):
        return False
    policy = request["policy"]
    return policy["comparable"] is prediction.get("comparable") and policy[
        "violations"
    ] == prediction.get("policy_violations")


def _agentic_request_is_complete(request: dict[str, Any], prediction: dict[str, Any]) -> bool:
    """Validate request evidence without applying strict prompt-hash equality.

    Agentic requests contain native tool definitions and may contain follow-up
    messages, so their effective input intentionally differs from the strict
    one-turn canonical request.  D/E no-cutoff rows do not invoke an adapter
    at all and therefore persist policy-only request evidence.
    """

    if not _policy_matches_prediction(request, prediction):
        return False
    if prediction.get("skipped_no_cutoff") is True:
        return request.get("skipped_no_cutoff") is True
    for key in ("canonical_input_hash", "effective_input_hash"):
        request_value = request.get(key)
        prediction_value = prediction.get(key)
        if (
            not isinstance(request_value, str)
            or not isinstance(prediction_value, str)
            or request_value != prediction_value
        ):
            return False
    return isinstance(request.get("prompt_sha256"), str)


def _deep_request_is_complete(request: dict[str, Any], prediction: dict[str, Any]) -> bool:
    """Validate a deep request while allowing the external agent transcript."""

    if not _policy_matches_prediction(request, prediction):
        return False
    if prediction.get("skipped_no_cutoff") is True:
        return request.get("skipped_no_cutoff") is True
    if not isinstance(request.get("prompt_sha256"), str):
        return False
    for key in ("canonical_input_hash", "effective_input_hash", "messages_sha256"):
        if not isinstance(request.get(key), str) or request.get(key) != prediction.get(key):
            return False
    deep = request.get("deep_research")
    return isinstance(deep, dict) and deep.get("protocol") == "external_research_stdio/v2"


def _valid_research_audit(row: dict[str, Any]) -> bool:
    return (
        _is_task_id(row.get("task_id"))
        and isinstance(row.get("event_type"), str)
        and bool(row.get("event_type"))
        and isinstance(row.get("network_call_count", 0), int)
        and row.get("network_call_count", 0) >= 0
    )


def _is_volc_search_row(row: dict[str, Any]) -> bool:
    """Identify a complete, cutoff-compliant gateway ledger row."""

    return (
        _valid_network_event(row)
        and row.get("is_model_endpoint") is False
        and row.get("network_class") == "web_search"
        and row.get("provider") == "volc"
        and row.get("host") == _VOLC_HOST
        and row.get("method") == "POST"
        and row.get("path") == _VOLC_PATH
        and row.get("action") == _VOLC_ACTION
        and row.get("version") == _VOLC_VERSION
        and row.get("cutoff_compliant") is True
        and isinstance(row.get("cutoff_date"), str)
        and isinstance(row.get("exclusive_cutoff_bound"), str)
        and row["exclusive_cutoff_bound"].endswith("T00:00:00Z")
        and isinstance(row.get("query_hash"), str)
        and isinstance(row.get("request_filter_hash"), str)
    )


def _is_service_local_row(row: dict[str, Any]) -> bool:
    return (
        _valid_network_event(row)
        and row.get("network_class") == "search_service_local"
        and row.get("host") == "127.0.0.1"
        and row.get("is_model_endpoint") is False
        and isinstance(row.get("receipt_sha256"), str)
    )


def _is_temporal_filter_row(row: dict[str, Any]) -> bool:
    return (
        _valid_network_event(row)
        and row.get("network_class") == TEMPORAL_FILTER_NETWORK_CLASS
        and row.get("is_model_endpoint") is True
    )


def _valid_search_audit(row: dict[str, Any]) -> bool:
    """Validate the redacted, candidate-level evidence for a Volc invocation."""

    if not (
        _is_task_id(row.get("task_id"))
        and row.get("provider") == "volc"
        and isinstance(row.get("query_sha256"), str)
        and isinstance(row.get("request_filter_sha256"), str)
        and isinstance(row.get("response_sha256"), str)
        and isinstance(row.get("cutoff_date"), str)
        and isinstance(row.get("exclusive_cutoff_bound"), str)
        and row["exclusive_cutoff_bound"].endswith("T00:00:00Z")
        and isinstance(row.get("time_range"), str)
        and row["time_range"].endswith(f"..{row['cutoff_date']}")
        and isinstance(row.get("accepted"), bool)
        and isinstance(row.get("reason"), str)
        and isinstance(row.get("test_only"), bool)
    ):
        return False
    if not row["accepted"]:
        return True
    has_published_at = isinstance(row.get("published_at"), str) or isinstance(
        row.get("published_at_sha256"), str
    )
    return row.get("reason") == "accepted" and (
        has_published_at or row.get("llm_verdict") == "keep"
    )


def _agentic_trace_complete(
    trace_rows: list[dict[str, Any]],
    prediction_groups: dict[str, list[dict[str, Any]]],
) -> tuple[set[str], int]:
    """Return complete task IDs and malformed/extra native-tool trace count."""

    grouped, _ = _task_groups(trace_rows)
    complete: set[str] = set()
    invalid_tool_trace_rows = 0
    for task_id, prediction_rows in prediction_groups.items():
        if len(prediction_rows) != 1:
            continue
        prediction = prediction_rows[0]
        rows = grouped.get(task_id, [])
        events = {row.get("event_type") for row in rows if isinstance(row.get("event_type"), str)}
        skipped = prediction.get("skipped_no_cutoff") is True
        required = _SKIPPED_TRACE_EVENTS if skipped else _TRACE_EVENTS
        tool_rows = [row for row in rows if row.get("event_type") == "tool_call"]
        declared_tools = prediction.get("tool_call_count")
        if not isinstance(declared_tools, int) or declared_tools < 0:
            continue
        if len(tool_rows) != declared_tools:
            invalid_tool_trace_rows += abs(len(tool_rows) - declared_tools) or 1
            continue
        if any(row.get("tool_name") != "search_web" for row in tool_rows):
            invalid_tool_trace_rows += 1
            continue
        if any(row.get("control_flow_node") == "cutoff_search_gateway" for row in tool_rows):
            # A gateway-only trace is the old harness-injected path, not a
            # model-issued native tool call.
            invalid_tool_trace_rows += 1
            continue
        if skipped and ({"model_request", "model_response", "tool_call"} & events):
            invalid_tool_trace_rows += 1
            continue
        if required <= events:
            complete.add(task_id)
    return complete, invalid_tool_trace_rows


def _comparison_identity(run_config: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    """Extract the immutable population identity used to block mixed reports."""

    search = run_config.get("search")
    dataset = run_config.get("dataset")
    return {
        "input_policy": _input_policy(run_config, provenance),
        "policy_version": provenance.get("policy_version"),
        "dataset_revision": provenance.get("dataset_revision")
        or (dataset.get("revision") if isinstance(dataset, dict) else None),
        "evaluator_manifest_sha256": provenance.get("evaluator_manifest_sha256"),
        "tool_contract_hash": provenance.get("tool_contract_hash")
        or (search.get("tool_contract_hash") if isinstance(search, dict) else None),
        "provider_profile": provenance.get("provider_profile"),
        "external_harness_revision": provenance.get("external_harness_revision"),
        "external_harness_tree_sha": provenance.get("external_harness_tree_sha"),
        "external_harness_lock_sha256": provenance.get("external_harness_lock_sha256"),
        "ipc_protocol": provenance.get("ipc_protocol"),
        "service_cohort_sha256": provenance.get("service_cohort_sha256"),
        "service_api_contract_sha256": provenance.get("service_api_contract_sha256"),
        "service_implementation_manifest_sha256": provenance.get(
            "service_implementation_manifest_sha256"
        ),
        "service_capability_sha256": provenance.get("service_capability_sha256"),
        "deep_budget": provenance.get("deep_budget"),
        "bridge_sha256": provenance.get("bridge_sha256"),
        "capability_manifest_sha256": provenance.get("capability_manifest_sha256"),
        "budget_sha256": provenance.get("budget_sha256"),
    }


def _service_manifest_matches(manifest: dict[str, Any], provenance: dict[str, Any]) -> bool:
    if not isinstance(manifest, dict) or not provenance.get("service_cohort_sha256"):
        return False
    return (
        manifest.get("protocol") == "search_filter_http/v1"
        and manifest.get("service_cohort_sha256") == provenance.get("service_cohort_sha256")
        and manifest.get("api_contract_sha256") == provenance.get("service_api_contract_sha256")
        and manifest.get("implementation_manifest_sha256")
        == provenance.get("service_implementation_manifest_sha256")
        and manifest.get("capability_sha256") == provenance.get("service_capability_sha256")
    )


def _row_for_baseline(directory: Path) -> dict[str, Any]:
    summary, summary_valid = _read_json_with_status(directory / "metrics_summary.json")
    provenance, provenance_valid = _read_json_with_status(directory / "provenance.json")
    run_config, run_config_valid = _read_json_with_status(directory / "run_config.json")
    predictions, predictions_valid = _read_jsonl_with_status(directory / "predictions.jsonl")
    trace_rows, trace_valid = _read_jsonl_with_status(directory / "trace_events.jsonl")
    request_rows, requests_valid = _read_jsonl_with_status(directory / "effective_requests.jsonl")
    ledger_rows, ledger_valid = _read_jsonl_with_status(directory / "network_ledger.jsonl")
    audit_rows, audit_valid = _read_jsonl_with_status(directory / "search_audit.jsonl")
    research_audit_rows, research_audit_valid = _read_jsonl_with_status(
        directory / "research_audit.jsonl"
    )
    external_manifest, external_manifest_valid = _read_json_with_status(
        directory / "external_harness_manifest.json"
    )
    service_manifest, service_manifest_valid = _read_json_with_status(
        directory / "search_service_manifest.json"
    )
    service_identity_valid = service_manifest_valid and _service_manifest_matches(
        service_manifest, provenance
    )

    framework = str(provenance.get("framework", directory.name))
    input_policy = _input_policy(run_config, provenance)
    try:
        profile_spec = get_profile(input_policy or "")
    except ProfileRegistryError as exc:
        raise ReportMixError(str(exc)) from exc
    is_agentic = profile_spec.name == _AGENTIC_PROFILE
    is_deep = profile_spec.name == _DEEP_PROFILE
    task_count = len(predictions)
    prediction_groups, invalid_prediction_rows = _task_groups(predictions)
    task_ids = set(prediction_groups)
    duplicate_prediction_rows = sum(len(rows) - 1 for rows in prediction_groups.values())
    prediction_task_id_completeness_rate = _rate(len(task_ids), task_count)

    request_groups, invalid_request_rows = _task_groups(request_rows)
    request_orphan_rows = sum(
        len(rows) for task_id, rows in request_groups.items() if task_id not in task_ids
    )
    complete_requests: set[str] = set()
    policy_evidence_tasks: set[str] = set()
    policy_compliant_tasks: set[str] = set()
    for task_id, prediction_rows in prediction_groups.items():
        request_for_task = request_groups.get(task_id, [])
        if len(prediction_rows) != 1 or len(request_for_task) != 1:
            continue
        prediction = prediction_rows[0]
        request = request_for_task[0]
        if _request_has_policy_evidence(request):
            policy_evidence_tasks.add(task_id)
            policy = request["policy"]
            if policy["comparable"] is True and policy["violations"] == []:
                policy_compliant_tasks.add(task_id)
        request_complete = (
            _agentic_request_is_complete(request, prediction)
            if is_agentic
            else _deep_request_is_complete(request, prediction)
            if is_deep
            else _request_is_complete(request, prediction)
        )
        if request_complete:
            complete_requests.add(task_id)

    effective_request_completeness_rate = _rate(len(complete_requests), task_count)
    policy_evidence_completeness_rate = _rate(len(policy_evidence_tasks), task_count)
    policy_compliance_rate = _rate(len(policy_compliant_tasks), task_count)

    ledger_groups, invalid_ledger_rows = _task_groups(ledger_rows)
    ledger_orphan_rows = sum(
        len(rows) for task_id, rows in ledger_groups.items() if task_id not in task_ids
    )
    audit_groups, invalid_audit_rows = _task_groups(audit_rows)
    audit_orphan_rows = sum(
        len(rows) for task_id, rows in audit_groups.items() if task_id not in task_ids
    )
    complete_ledger_tasks: set[str] = set()
    non_model_compliant_tasks: set[str] = set()
    audit_complete_tasks: set[str] = set()
    budget_compliant_tasks: set[str] = set()
    research_complete_tasks: set[str] = set()
    child_network_compliant_tasks: set[str] = set()
    invalid_research_audit_rows = 0
    research_orphan_rows = 0
    non_model_network_event_count = 0

    if is_agentic:
        for task_id, prediction_rows in prediction_groups.items():
            if len(prediction_rows) != 1:
                continue
            prediction = prediction_rows[0]
            skipped = prediction.get("skipped_no_cutoff") is True
            model_calls = prediction.get("model_call_count")
            tool_calls = prediction.get("tool_call_count")
            all_events = ledger_groups.get(task_id, [])
            adapter_events = [
                event
                for event in all_events
                if event.get("is_model_endpoint") is True
                and event.get("network_class") != TEMPORAL_FILTER_NETWORK_CLASS
            ]
            filter_events = [event for event in all_events if _is_temporal_filter_row(event)]
            gateway_events = [event for event in all_events if _is_volc_search_row(event)]
            local_events = [event for event in all_events if _is_service_local_row(event)]
            task_audits = audit_groups.get(task_id, [])
            non_model_network_event_count += len(gateway_events)

            model_events_ok = (
                isinstance(model_calls, int)
                and 0 <= model_calls <= _MAX_AGENTIC_MODEL_CALLS
                and len(adapter_events) == model_calls
                and all(
                    _valid_network_event(event) and event.get("is_model_endpoint") is True
                    for event in adapter_events
                )
            )
            budget_ok = (
                isinstance(model_calls, int)
                and isinstance(tool_calls, int)
                and 0 <= model_calls <= _MAX_AGENTIC_MODEL_CALLS
                and 0 <= tool_calls <= _MAX_AGENTIC_TOOL_CALLS
            )
            if skipped:
                model_events_ok = model_events_ok and model_calls == 0
                budget_ok = budget_ok and model_calls == 0 and tool_calls == 0
            else:
                model_events_ok = model_events_ok and model_calls >= 1
                budget_ok = budget_ok and model_calls >= 1 and tool_calls >= 1
            if budget_ok:
                budget_compliant_tasks.add(task_id)

            volc_rows_ok = all(_is_volc_search_row(event) for event in gateway_events)
            expected_searches = tool_calls if isinstance(tool_calls, int) else -1
            volc_rows_ok = volc_rows_ok and len(gateway_events) == expected_searches
            local_rows_ok = len(local_events) == expected_searches and all(
                _is_service_local_row(event) for event in local_events
            )
            filter_calls = prediction.get("filter_model_call_count", 0)
            filter_rows_ok = (
                isinstance(filter_calls, int)
                and filter_calls == len(filter_events)
                and all(_is_temporal_filter_row(event) for event in filter_events)
            )
            service_required = bool(provenance.get("service_cohort_sha256"))
            if skipped:
                volc_rows_ok = volc_rows_ok and not task_audits
                local_rows_ok = local_rows_ok or not local_events or not service_required
                filter_rows_ok = filter_rows_ok or not filter_events or not service_required
            else:
                volc_rows_ok = volc_rows_ok and bool(gateway_events)
                local_rows_ok = (local_rows_ok and bool(local_events)) or not service_required
                filter_rows_ok = filter_rows_ok or not service_required
            if model_events_ok and volc_rows_ok and local_rows_ok and filter_rows_ok:
                complete_ledger_tasks.add(task_id)
            if volc_rows_ok and local_rows_ok and filter_rows_ok:
                non_model_compliant_tasks.add(task_id)

            audit_rows_ok = all(_valid_search_audit(audit) for audit in task_audits)
            search_keys = {
                (
                    event.get("query_hash"),
                    event.get("request_filter_hash"),
                    event.get("cutoff_date"),
                    event.get("exclusive_cutoff_bound"),
                )
                for event in gateway_events
            }
            audit_keys = {
                (
                    audit.get("query_sha256"),
                    audit.get("request_filter_sha256"),
                    audit.get("cutoff_date"),
                    audit.get("exclusive_cutoff_bound"),
                )
                for audit in task_audits
            }
            audit_rows_ok = audit_rows_ok and audit_keys <= search_keys
            if gateway_events:
                audit_rows_ok = audit_rows_ok and search_keys <= audit_keys
            else:
                audit_rows_ok = audit_rows_ok and not task_audits
            if audit_rows_ok:
                audit_complete_tasks.add(task_id)
    elif is_deep:
        research_groups, invalid_research_task_ids = _task_groups(research_audit_rows)
        invalid_research_audit_rows = invalid_research_task_ids + sum(
            1 for row in research_audit_rows if not _valid_research_audit(row)
        )
        research_orphan_rows = sum(
            len(rows) for task_id, rows in research_groups.items() if task_id not in task_ids
        )
        for task_id, prediction_rows in prediction_groups.items():
            if len(prediction_rows) != 1:
                continue
            prediction = prediction_rows[0]
            skipped = prediction.get("skipped_no_cutoff") is True
            model_calls = prediction.get("model_call_count")
            tool_calls = prediction.get("tool_call_count")
            search_calls = prediction.get("search_call_count")
            visit_calls = prediction.get("visit_count")
            child_calls = prediction.get("child_network_call_count")
            child_search_service_calls = prediction.get("child_search_service_call_count", 0)
            child_forbidden_calls = prediction.get("child_forbidden_network_call_count", 0)
            all_events = ledger_groups.get(task_id, [])
            filter_events = [
                event
                for event in all_events
                if event.get("network_class") == TEMPORAL_FILTER_NETWORK_CLASS
            ]
            adapter_events = [
                event
                for event in all_events
                if event.get("is_model_endpoint") is True
                and event.get("network_class") != TEMPORAL_FILTER_NETWORK_CLASS
            ]
            gateway_events = [event for event in all_events if _is_volc_search_row(event)]
            local_events = [event for event in all_events if _is_service_local_row(event)]
            task_audits = audit_groups.get(task_id, [])
            task_research_audits = research_groups.get(task_id, [])
            non_model_network_event_count += len(gateway_events)
            filter_calls = prediction.get("filter_model_call_count")
            filter_events_ok = all(
                _valid_network_event(event)
                and event.get("is_model_endpoint") is True
                and event.get("network_class") == TEMPORAL_FILTER_NETWORK_CLASS
                for event in filter_events
            )
            filter_events_ok = filter_events_ok and not any(
                event.get("is_model_endpoint") is True
                and event.get("network_class") not in (None, TEMPORAL_FILTER_NETWORK_CLASS)
                for event in all_events
            )
            if filter_calls is not None:
                filter_events_ok = (
                    filter_events_ok
                    and isinstance(filter_calls, int)
                    and not isinstance(filter_calls, bool)
                    and filter_calls == len(filter_events)
                )
            model_events_ok = (
                isinstance(model_calls, int)
                and 0 <= model_calls <= _MAX_DEEP_MODEL_CALLS
                and len(adapter_events) == model_calls
                and filter_events_ok
                and all(
                    _valid_network_event(event) and event.get("is_model_endpoint") is True
                    for event in adapter_events
                )
            )
            budget_ok = (
                isinstance(model_calls, int)
                and isinstance(tool_calls, int)
                and isinstance(search_calls, int)
                and isinstance(visit_calls, int)
                and 0 <= model_calls <= _MAX_DEEP_MODEL_CALLS
                and 0 <= tool_calls <= _MAX_DEEP_TOOL_CALLS
                and 0 <= search_calls <= _MAX_DEEP_SEARCH_CALLS
                and 0 <= visit_calls <= _MAX_DEEP_VISIT_CALLS
            )
            if skipped:
                model_events_ok = model_events_ok and model_calls == 0
                budget_ok = budget_ok and model_calls == 0 and tool_calls == 0
            else:
                model_events_ok = model_events_ok and model_calls >= 1
                budget_ok = budget_ok and model_calls >= 1 and search_calls >= 1
            if budget_ok:
                budget_compliant_tasks.add(task_id)

            volc_rows_ok = all(_is_volc_search_row(event) for event in gateway_events)
            expected_searches = search_calls if isinstance(search_calls, int) else -1
            volc_rows_ok = volc_rows_ok and len(gateway_events) == expected_searches
            local_rows_ok = len(local_events) == expected_searches and all(
                _is_service_local_row(event) for event in local_events
            )
            child_rows_ok = (
                isinstance(child_search_service_calls, int)
                and child_search_service_calls == expected_searches
                and child_forbidden_calls == 0
            )
            service_required = bool(provenance.get("service_cohort_sha256"))
            if skipped:
                volc_rows_ok = volc_rows_ok and not task_audits
                local_rows_ok = local_rows_ok or not local_events or not service_required
                child_rows_ok = (
                    child_rows_ok
                    or (not local_events and child_search_service_calls == 0)
                    or not service_required
                )
            else:
                volc_rows_ok = volc_rows_ok and bool(gateway_events)
                local_rows_ok = (local_rows_ok and bool(local_events)) or not service_required
                child_rows_ok = child_rows_ok or not service_required
            if model_events_ok and volc_rows_ok and local_rows_ok and child_rows_ok:
                complete_ledger_tasks.add(task_id)
            if volc_rows_ok and local_rows_ok and child_rows_ok:
                non_model_compliant_tasks.add(task_id)

            audit_rows_ok = all(_valid_search_audit(audit) for audit in task_audits)
            search_keys = {
                (
                    event.get("query_hash"),
                    event.get("request_filter_hash"),
                    event.get("cutoff_date"),
                    event.get("exclusive_cutoff_bound"),
                )
                for event in gateway_events
            }
            audit_keys = {
                (
                    audit.get("query_sha256"),
                    audit.get("request_filter_sha256"),
                    audit.get("cutoff_date"),
                    audit.get("exclusive_cutoff_bound"),
                )
                for audit in task_audits
            }
            audit_rows_ok = audit_rows_ok and audit_keys <= search_keys
            if gateway_events:
                audit_rows_ok = audit_rows_ok and search_keys <= audit_keys
            else:
                audit_rows_ok = audit_rows_ok and not task_audits
            if audit_rows_ok:
                audit_complete_tasks.add(task_id)

            research_types = {
                row.get("event_type")
                for row in task_research_audits
                if isinstance(row.get("event_type"), str)
            }
            research_ok = (
                all(_valid_research_audit(row) for row in task_research_audits)
                and {"task_started", "task_finished"} <= research_types
                and all(row.get("network_call_count", 0) == 0 for row in task_research_audits)
            )
            if skipped:
                research_ok = research_ok and not (
                    {"model_request", "model_response", "tool_call"} & research_types
                )
            if research_ok:
                research_complete_tasks.add(task_id)
            if child_calls == 0 and child_rows_ok:
                child_network_compliant_tasks.add(task_id)
    else:
        for task_id, prediction_rows in prediction_groups.items():
            all_events = ledger_groups.get(task_id, [])
            adapter_events = [
                event for event in all_events if event.get("is_model_endpoint") is True
            ]
            gateway_events = [
                event for event in all_events if event.get("is_model_endpoint") is not True
            ]
            model_events = [
                event for event in adapter_events if event.get("is_model_endpoint") is True
            ]
            non_model_events = gateway_events
            non_model_network_event_count += len(non_model_events)
            if (
                len(prediction_rows) == 1
                and len(model_events) == 1
                and _valid_network_event(model_events[0])
            ):
                complete_ledger_tasks.add(task_id)
            if not non_model_events:
                non_model_compliant_tasks.add(task_id)

    network_ledger_completeness_rate = _rate(len(complete_ledger_tasks), task_count)
    non_model_network_compliance_rate = _rate(len(non_model_compliant_tasks), task_count)
    search_audit_completeness_rate = (
        _rate(len(audit_complete_tasks), task_count) if (is_agentic or is_deep) else None
    )
    agentic_budget_compliance_rate = (
        _rate(len(budget_compliant_tasks), task_count) if is_agentic else None
    )
    deep_budget_compliance_rate = (
        _rate(len(budget_compliant_tasks), task_count) if is_deep else None
    )
    trace_completeness_rate = _trace_completeness(trace_rows, task_ids)
    trace_groups, invalid_trace_rows = _task_groups(trace_rows)
    invalid_trace_event_rows = sum(
        1
        for row in trace_rows
        if not isinstance(row.get("event_type"), str) or not row["event_type"]
    )
    trace_orphan_rows = sum(
        len(rows) for task_id, rows in trace_groups.items() if task_id not in task_ids
    )
    trace_complete_tasks = {
        task_id
        for task_id in task_ids
        if _TRACE_EVENTS
        <= {
            str(row.get("event_type"))
            for row in trace_groups.get(task_id, [])
            if isinstance(row.get("event_type"), str)
        }
    }
    invalid_tool_trace_rows = 0
    if is_agentic:
        trace_complete_tasks, invalid_tool_trace_rows = _agentic_trace_complete(
            trace_rows, prediction_groups
        )
        trace_completeness_rate = _rate(len(trace_complete_tasks), task_count)
    elif is_deep:
        trace_complete_tasks = {
            task_id
            for task_id, rows in prediction_groups.items()
            if len(rows) == 1
            and (
                (
                    _SKIPPED_TRACE_EVENTS
                    if rows[0].get("skipped_no_cutoff") is True
                    else _TRACE_EVENTS
                )
                <= {
                    str(row.get("event_type"))
                    for row in trace_groups.get(task_id, [])
                    if isinstance(row.get("event_type"), str)
                }
            )
        }
        trace_completeness_rate = _rate(len(trace_complete_tasks), task_count)
    per_task_complete_tasks = (
        trace_complete_tasks & complete_requests & complete_ledger_tasks & policy_evidence_tasks
    )
    if is_agentic:
        per_task_complete_tasks &= audit_complete_tasks & budget_compliant_tasks
    if is_deep:
        per_task_complete_tasks &= (
            audit_complete_tasks
            & budget_compliant_tasks
            & research_complete_tasks
            & child_network_compliant_tasks
        )
    per_task_artifact_completeness_rate = _rate(len(per_task_complete_tasks), task_count)

    comparable_count = sum(
        1
        for row in predictions
        if row.get("comparable") is True and not (is_deep and row.get("skipped_no_cutoff") is True)
    )
    api_error_count = sum(1 for row in predictions if row.get("api_error"))
    format_valid_count = sum(
        1
        for row in predictions
        if isinstance(row.get("parsed"), dict) and row["parsed"].get("format_valid") is True
    )
    model_one_count = sum(1 for row in predictions if row.get("model_call_count") == 1)
    no_tool_count = sum(1 for row in predictions if row.get("tool_call_count") == 0)
    one_model_call_compliance_rate = _rate(model_one_count, task_count)
    zero_tool_call_compliance_rate = _rate(no_tool_count, task_count)
    success_count = sum(
        1 for row in predictions if _response_present(row) and not row.get("api_error")
    )
    latencies = [
        float(row["latency_seconds"])
        for row in predictions
        if isinstance(row.get("latency_seconds"), int | float)
    ]
    input_tokens = sum(
        int((row.get("usage") or {}).get("input_tokens", 0) or 0) for row in predictions
    )
    output_tokens = sum(
        int((row.get("usage") or {}).get("output_tokens", 0) or 0) for row in predictions
    )
    violations = Counter(
        violation
        for row in predictions
        for violation in row.get("policy_violations", [])
        if isinstance(violation, str)
    )
    integration = INTEGRATION_COST.get(
        framework, {"dependency_footprint": "unknown", "cost": "unknown"}
    )
    non_performance = bool(
        provenance.get("mock_non_performance", False)
        or provenance.get("search_test_only", False)
        or any(audit.get("test_only") is True for audit in audit_rows)
    )
    artifact_files = {
        "predictions.jsonl": predictions_valid,
        "metrics_summary.json": summary_valid,
        "run_config.json": run_config_valid,
        "provenance.json": provenance_valid,
        "trace_events.jsonl": trace_valid,
        "effective_requests.jsonl": requests_valid,
        "network_ledger.jsonl": ledger_valid,
    }
    if is_agentic:
        artifact_files["search_audit.jsonl"] = audit_valid
        if provenance.get("service_cohort_sha256"):
            artifact_files["search_service_manifest.json"] = service_identity_valid
    if is_deep:
        artifact_files["search_audit.jsonl"] = audit_valid
        artifact_files["research_audit.jsonl"] = research_audit_valid
        artifact_files["external_harness_manifest.json"] = external_manifest_valid
        if provenance.get("service_cohort_sha256"):
            artifact_files["search_service_manifest.json"] = service_identity_valid
    artifact_file_completeness_rate = _rate(
        sum(1 for valid in artifact_files.values() if valid), len(artifact_files)
    )
    invalid_network_event_rows = sum(1 for row in ledger_rows if not _valid_network_event(row))
    invalid_search_audit_rows = sum(1 for row in audit_rows if not _valid_search_audit(row))
    artifact_complete = (
        task_count > 0
        and all(artifact_files.values())
        and invalid_prediction_rows == 0
        and duplicate_prediction_rows == 0
        and invalid_trace_rows == 0
        and invalid_trace_event_rows == 0
        and trace_orphan_rows == 0
        and invalid_request_rows == 0
        and request_orphan_rows == 0
        and invalid_ledger_rows == 0
        and invalid_network_event_rows == 0
        and ledger_orphan_rows == 0
        and (not is_agentic or invalid_audit_rows == 0)
        and (not is_agentic or invalid_search_audit_rows == 0)
        and (not is_agentic or audit_orphan_rows == 0)
        and (not is_agentic or invalid_tool_trace_rows == 0)
        and (not is_deep or invalid_research_audit_rows == 0)
        and (not is_deep or research_orphan_rows == 0)
        and (not is_deep or external_manifest.get("license_status") == "UNDECLARED")
        and _all_rate_one(
            prediction_task_id_completeness_rate,
            trace_completeness_rate,
            effective_request_completeness_rate,
            policy_evidence_completeness_rate,
            network_ledger_completeness_rate,
            per_task_artifact_completeness_rate,
        )
        and (
            not is_deep
            or _all_rate_one(
                _rate(len(research_complete_tasks), task_count),
                _rate(len(child_network_compliant_tasks), task_count),
            )
        )
    )
    rankable_framework = profile_spec.is_rankable_baseline(framework)
    strict_ranking_eligible = (
        rankable_framework
        and not non_performance
        and comparable_count == task_count
        and artifact_complete
        and _all_rate_one(
            policy_compliance_rate,
            one_model_call_compliance_rate,
            zero_tool_call_compliance_rate,
            non_model_network_compliance_rate,
        )
    )
    agentic_ranking_eligible = (
        rankable_framework
        and not non_performance
        and comparable_count == task_count
        and artifact_complete
        and _all_rate_one(
            policy_compliance_rate,
            agentic_budget_compliance_rate,
            non_model_network_compliance_rate,
            search_audit_completeness_rate,
        )
    )
    deep_ranking_eligible = (
        rankable_framework
        and not non_performance
        and comparable_count == task_count
        and not any(row.get("skipped_no_cutoff") is True for row in predictions)
        and artifact_complete
        and _all_rate_one(
            policy_compliance_rate,
            deep_budget_compliance_rate,
            non_model_network_compliance_rate,
            search_audit_completeness_rate,
            _rate(len(research_complete_tasks), task_count),
            _rate(len(child_network_compliant_tasks), task_count),
        )
    )
    ranking_eligible = (
        deep_ranking_eligible
        if is_deep
        else agentic_ranking_eligible
        if is_agentic
        else strict_ranking_eligible
    )
    return {
        "framework": framework,
        "framework_version": provenance.get("framework_version"),
        "adapter_version": provenance.get("adapter_version"),
        "adapter_lockfile_hash": provenance.get("lockfile_sha256"),
        "dataset_revision": provenance.get("dataset_revision"),
        "dataset_version": provenance.get("manifest_version", provenance.get("dataset_version")),
        "provider_profile": provenance.get("provider_profile"),
        "config_hash": provenance.get("config_hash"),
        "task_count": task_count,
        "comparable_task_count": comparable_count,
        "ranking_eligible": ranking_eligible,
        "mock_non_performance": non_performance,
        "success_rate": _rate(success_count, task_count),
        "api_error_rate": _rate(api_error_count, task_count),
        "one_model_call_compliance_rate": one_model_call_compliance_rate,
        "zero_tool_call_compliance_rate": zero_tool_call_compliance_rate,
        "zero_external_data_compliance_rate": (
            None if (is_agentic or is_deep) else non_model_network_compliance_rate
        ),
        "non_model_network_compliance_rate": non_model_network_compliance_rate,
        "volc_only_network_compliance_rate": (
            non_model_network_compliance_rate if (is_agentic or is_deep) else None
        ),
        "child_network_compliance_rate": (
            _rate(len(child_network_compliant_tasks), task_count) if is_deep else None
        ),
        "non_model_network_event_count": non_model_network_event_count,
        "trace_completeness_rate": trace_completeness_rate,
        "effective_request_completeness_rate": effective_request_completeness_rate,
        "call_log_completeness_rate": effective_request_completeness_rate,
        "policy_evidence_completeness_rate": policy_evidence_completeness_rate,
        "policy_compliance_rate": policy_compliance_rate,
        "network_ledger_completeness_rate": network_ledger_completeness_rate,
        "search_audit_completeness_rate": search_audit_completeness_rate,
        "agentic_budget_compliance_rate": agentic_budget_compliance_rate,
        "deep_budget_compliance_rate": deep_budget_compliance_rate,
        "research_audit_completeness_rate": (
            _rate(len(research_complete_tasks), task_count) if is_deep else None
        ),
        "artifact_completeness_rate": per_task_artifact_completeness_rate,
        "artifact_file_completeness_rate": artifact_file_completeness_rate,
        "artifact_complete": artifact_complete,
        "artifact_integrity": {
            "required_files": artifact_files,
            "invalid_prediction_rows": invalid_prediction_rows,
            "duplicate_prediction_rows": duplicate_prediction_rows,
            "invalid_trace_rows": invalid_trace_rows,
            "invalid_trace_event_rows": invalid_trace_event_rows,
            "trace_orphan_rows": trace_orphan_rows,
            "invalid_effective_request_rows": invalid_request_rows,
            "effective_request_orphan_rows": request_orphan_rows,
            "invalid_network_ledger_rows": invalid_ledger_rows,
            "invalid_network_event_rows": invalid_network_event_rows,
            "network_ledger_orphan_rows": ledger_orphan_rows,
            "invalid_search_audit_rows": invalid_audit_rows + invalid_search_audit_rows,
            "search_audit_orphan_rows": audit_orphan_rows,
            "invalid_tool_trace_rows": invalid_tool_trace_rows,
            "invalid_research_audit_rows": invalid_research_audit_rows,
            "research_audit_orphan_rows": research_orphan_rows,
            "external_manifest_valid": external_manifest_valid,
        },
        "format_valid_rate": _rate(format_valid_count, task_count),
        "p50_latency_seconds": median(latencies) if latencies else None,
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "input_tokens": input_tokens or None,
        "output_tokens": output_tokens or None,
        "cost": None,
        "policy_violations": dict(sorted(violations.items())),
        "dependency_footprint": integration["dependency_footprint"],
        "integration_cost": integration["cost"],
        "upstream_metrics": summary.get("upstream_summary", summary),
        "caveats": COMMON_CAVEATS,
        "_comparison_identity": _comparison_identity(run_config, provenance),
    }


def _looks_like_baseline_directory(directory: Path) -> bool:
    return any((directory / name).exists() for name in _REQUIRED_ARTIFACTS)


def build_comparison(run_dir: str | Path) -> dict[str, Any]:
    """Load artifacts and make an auditable comparison document."""

    root = Path(run_dir)
    rows = [
        _row_for_baseline(directory)
        for directory in sorted(root.iterdir())
        if directory.is_dir() and _looks_like_baseline_directory(directory)
    ]
    identities = {json.dumps(row["_comparison_identity"], sort_keys=True) for row in rows}
    if len(identities) > 1:
        raise ReportMixError(
            "Refusing to mix report populations with different input policy, dataset revision, "
            "evaluator manifest, tool contract, or policy version."
        )
    for row in rows:
        row.pop("_comparison_identity", None)
    return {
        "schema_version": "1.0",
        "run_id": root.name,
        "caveats": COMMON_CAVEATS,
        "baselines": rows,
    }


def write_comparison(run_dir: str | Path) -> dict[str, Path]:
    root = Path(run_dir)
    comparison = build_comparison(root)
    store = ArtifactStore(root.parent, root.name)
    json_path = store.write_json(root / "comparison.json", comparison)
    csv_path = root / "comparison.csv"
    rows: list[dict[str, Any]] = []
    for row in comparison["baselines"]:
        flat = {
            key: json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, dict | list)
            else value
            for key, value in row.items()
            if key != "upstream_metrics"
        }
        flat["upstream_metrics_json"] = json.dumps(
            row["upstream_metrics"], ensure_ascii=False, sort_keys=True
        )
        rows.append(flat)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["framework"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"json": json_path, "csv": csv_path}
