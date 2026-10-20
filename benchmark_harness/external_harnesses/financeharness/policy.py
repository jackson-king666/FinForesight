"""Fail-closed policy checks for the restricted FinanceHarness population."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from benchmark_harness.config import DeepResearchConfig
from benchmark_harness.contracts import (
    TEMPORAL_FILTER_NETWORK_CLASS,
    AdapterResult,
    TaskInput,
)
from benchmark_harness.search.cutoff import CutoffWindow, CutoffWindowError
from benchmark_harness.search.volc import SearchOutcome

SAFE_TOOL_NAMES = frozenset(
    {
        "search",
        "visit",
        "compose_citations",
        "calc",
        "update_plan",
        "load_tool",
    }
)
PURE_DEFERRED_TOOL_PREFIXES = ("compute_valuation_", "compute_risk_")


@dataclass(frozen=True, slots=True)
class DeepPolicyDecision:
    """Comparable only when the parent-owned boundary is fully evidenced."""

    comparable: bool
    violations: tuple[str, ...]
    skipped_no_cutoff: bool = False
    child_network_call_count: int = 0
    child_search_service_call_count: int = 0
    child_forbidden_network_call_count: int = 0
    model_call_count: int = 0
    filter_model_call_count: int = 0
    search_call_count: int = 0
    visit_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "comparable": self.comparable,
            "violations": list(self.violations),
            "skipped_no_cutoff": self.skipped_no_cutoff,
            "child_network_call_count": self.child_network_call_count,
            "child_search_service_call_count": self.child_search_service_call_count,
            "child_forbidden_network_call_count": self.child_forbidden_network_call_count,
            "model_call_count": self.model_call_count,
            "filter_model_call_count": self.filter_model_call_count,
            "search_call_count": self.search_call_count,
            "visit_count": self.visit_count,
            "allowed_tools": sorted(SAFE_TOOL_NAMES),
        }


class DeepResearchPolicyGuard:
    """Validate policy evidence after the external worker has exited."""

    def __init__(self, limits: DeepResearchConfig | None = None) -> None:
        self.limits = limits or DeepResearchConfig()

    def check(
        self,
        task: TaskInput,
        result: AdapterResult | None,
        *,
        search_outcomes: Sequence[SearchOutcome] = (),
        network_ledger_rows: Iterable[dict[str, Any]] = (),
        research_audit_rows: Iterable[dict[str, Any]] = (),
        child_network_call_count: int = 0,
        child_search_service_call_count: int = 0,
        child_forbidden_network_call_count: int = 0,
        citations: Iterable[str] = (),
        skipped_no_cutoff: bool = False,
    ) -> DeepPolicyDecision:
        violations: list[str] = []
        rows = tuple(network_ledger_rows)
        audits = tuple(research_audit_rows)
        if child_forbidden_network_call_count != 0:
            violations.append("child_forbidden_network_call_detected")
        if child_search_service_call_count < 0:
            violations.append("child_search_service_call_count_invalid")
        service_rows = any(
            isinstance(row, dict) and row.get("network_class") == "search_service_local"
            for row in rows
        )
        if service_rows and child_search_service_call_count != len(search_outcomes):
            violations.append("child_search_service_receipt_mismatch")
        if not isinstance(child_network_call_count, int) or child_network_call_count < 0:
            violations.append("child_network_call_count_invalid")
            child_network_call_count = 0
        if child_network_call_count:
            violations.append("child_network_egress_detected")

        try:
            CutoffWindow.from_seed(task.cutoff_date)
            valid_cutoff = True
        except CutoffWindowError:
            valid_cutoff = False
        if not valid_cutoff:
            if not skipped_no_cutoff:
                violations.append("bad_or_missing_cutoff_date")
            if result is not None:
                violations.append("model_called_without_valid_cutoff")
            if search_outcomes or rows or audits:
                violations.append("network_used_without_valid_cutoff")
            return DeepPolicyDecision(
                comparable=not violations and skipped_no_cutoff,
                violations=tuple(_deduplicate(violations)),
                skipped_no_cutoff=skipped_no_cutoff,
                child_network_call_count=child_network_call_count,
            )

        if skipped_no_cutoff:
            violations.append("valid_cutoff_task_was_skipped")
        if result is None:
            violations.append("missing_adapter_result")
        else:
            if result.error:
                violations.append("adapter_error")
            if not result.raw_text or not result.raw_text.strip():
                violations.append("empty_raw_response")
            deep_metadata = result.effective_request.get("deep_research")
            if not isinstance(deep_metadata, dict):
                violations.append("deep_result_metadata_missing")
            else:
                if deep_metadata.get("termination") != "answer":
                    violations.append("non_answer_termination")
                rounds = deep_metadata.get("rounds")
                if (
                    not isinstance(rounds, int)
                    or isinstance(rounds, bool)
                    or rounds < 1
                    or rounds > self.limits.max_rounds
                ):
                    violations.append("round_budget_evidence_invalid")

        model_calls = result.model_call_count if result is not None else 0
        filter_model_call_count = 0
        tool_calls = tuple(result.tool_calls) if result is not None else ()
        search_count = sum(1 for call in tool_calls if call.name in {"search", "search_web"})
        visit_count = sum(1 for call in tool_calls if call.name == "visit")
        if result is not None:
            for call in tool_calls:
                name = call.name
                if name not in SAFE_TOOL_NAMES and not name.startswith(PURE_DEFERRED_TOOL_PREFIXES):
                    violations.append(f"tool_not_allowlisted:{name}")
            if model_calls < 1:
                violations.append("missing_model_call")
            if model_calls > self.limits.max_model_calls:
                violations.append("model_call_budget_exceeded")
            if len(tool_calls) > self.limits.max_tool_calls:
                violations.append("tool_call_budget_exceeded")
            if search_count > self.limits.max_search_queries:
                violations.append("search_query_budget_exceeded")
            if visit_count > self.limits.max_visit_urls:
                violations.append("visit_url_budget_exceeded")
            filter_events = [
                event
                for event in result.network_events
                if event.network_class == TEMPORAL_FILTER_NETWORK_CLASS
            ]
            filter_model_call_count = len(filter_events)
            deep_metadata = result.effective_request.get("deep_research")
            reported_filter_calls = (
                deep_metadata.get("filter_model_call_count")
                if isinstance(deep_metadata, dict)
                else None
            )
            if reported_filter_calls is not None and (
                not isinstance(reported_filter_calls, int)
                or isinstance(reported_filter_calls, bool)
                or reported_filter_calls != filter_model_call_count
            ):
                violations.append("filter_model_ledger_mismatch")

            model_events = []
            for event in result.network_events:
                if event.network_class == TEMPORAL_FILTER_NETWORK_CLASS:
                    if not event.is_model_endpoint:
                        violations.append("temporal_filter_not_model_endpoint")
                    continue
                if not event.is_model_endpoint:
                    violations.append("adapter_network_not_model_allowlisted")
                    continue
                if event.network_class is not None:
                    violations.append("unknown_model_network_class")
                    continue
                model_events.append(event)
            if len(model_events) != model_calls:
                violations.append("model_network_ledger_mismatch")

        if not search_count:
            violations.append("missing_search_tool_call")
        if not search_outcomes:
            violations.append("missing_search_audit")
        for outcome in search_outcomes:
            if outcome.error_code:
                violations.append(f"search_error:{outcome.error_code}")
            if outcome.cutoff_date != task.cutoff_date:
                violations.append("audit_cutoff_date_mismatch")
            if not outcome.audit_records:
                violations.append("missing_search_audit")
            if outcome.tool_call_count > self.limits.max_search_queries:
                violations.append("search_query_budget_exceeded")
            for audit in outcome.audit_records:
                if audit.accepted and audit.reason != "accepted":
                    violations.append("accepted_search_reason_invalid")

        for row in rows:
            if row.get("network_class") == "search_service_local":
                if row.get("host") != "127.0.0.1" or row.get("is_model_endpoint") is not False:
                    violations.append("search_service_local_endpoint_invalid")
                continue
            if row.get("network_class") == "temporal_filter":
                if row.get("is_model_endpoint") is not True:
                    violations.append("temporal_filter_not_model_endpoint")
                continue
            if row.get("is_model_endpoint") is True:
                continue
            if row.get("network_class") != "web_search" or row.get("provider") != "volc":
                violations.append("non_volc_network_event")
            if row.get("cutoff_compliant") is not True:
                violations.append("search_cutoff_not_compliant")

        accepted_urls = {
            candidate.url
            for outcome in search_outcomes
            for candidate in outcome.candidates
            if isinstance(candidate.url, str) and candidate.url
        }
        for citation in citations:
            if isinstance(citation, str) and citation and citation not in accepted_urls:
                violations.append("citation_not_from_accepted_search")
        for audit in audits:
            if audit.get("task_id") not in {task.task_id, None}:
                violations.append("research_audit_task_mismatch")
            if audit.get("event_type") == "policy.violation":
                violations.append("child_policy_violation")
            if audit.get("network_call_count", 0):
                violations.append("child_audit_network_call")

        round_events = sum(1 for row in audits if row.get("event_type") == "round_start")
        if round_events > self.limits.max_rounds:
            violations.append("round_budget_exceeded")

        required_audit_types = {"task_started", "task_finished"}
        observed_audit_types = {
            row.get("event_type") for row in audits if isinstance(row.get("event_type"), str)
        }
        if not required_audit_types <= observed_audit_types:
            violations.append("research_audit_incomplete")

        return DeepPolicyDecision(
            comparable=not violations,
            violations=tuple(_deduplicate(violations)),
            child_network_call_count=child_network_call_count,
            child_search_service_call_count=child_search_service_call_count,
            child_forbidden_network_call_count=child_forbidden_network_call_count,
            model_call_count=model_calls,
            filter_model_call_count=filter_model_call_count,
            search_call_count=search_count,
            visit_count=visit_count,
        )


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


__all__ = [
    "PURE_DEFERRED_TOOL_PREFIXES",
    "SAFE_TOOL_NAMES",
    "DeepPolicyDecision",
    "DeepResearchPolicyGuard",
]
