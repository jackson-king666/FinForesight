"""Fairness gates for the separate ``agentic_web_search`` profile."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from benchmark_harness.contracts import TEMPORAL_FILTER_NETWORK_CLASS, AdapterResult, TaskInput
from benchmark_harness.search.cutoff import CutoffWindow, CutoffWindowError
from benchmark_harness.search.volc import DEFAULT_TOOL_CONTRACT, SearchOutcome
from benchmark_harness.search.volc_core import (
    VOLC_SEARCH_ACTION,
    VOLC_SEARCH_HOST,
    VOLC_SEARCH_PATH,
    VOLC_SEARCH_VERSION,
)


@dataclass(frozen=True, slots=True)
class AgenticPolicyDecision:
    """Comparable only when cutoff, budgets, gateway, and audit are all visible."""

    comparable: bool
    violations: tuple[str, ...]
    skipped_no_cutoff: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "comparable": self.comparable,
            "violations": list(self.violations),
            "skipped_no_cutoff": self.skipped_no_cutoff,
        }


class AgenticWebSearchPolicyGuard:
    """Fail closed without weakening the existing strict-profile policy guard."""

    def __init__(self, *, tool_contract_hash: str = DEFAULT_TOOL_CONTRACT.hash) -> None:
        self.tool_contract_hash = tool_contract_hash

    def check(
        self,
        task: TaskInput,
        result: AdapterResult | None,
        *,
        search_outcomes: Sequence[SearchOutcome] = (),
        skipped_no_cutoff: bool = False,
        network_ledger_rows: Iterable[dict[str, Any]] = (),
    ) -> AgenticPolicyDecision:
        """Check one agentic task without assuming a specific framework SDK.

        The runner calls this with ``result=None`` for the mandated D/E
        no-cutoff pre-skip, which verifies zero model/tool/network activity.
        """

        cutoff_date = getattr(task, "cutoff_date", None)
        try:
            CutoffWindow.from_seed(cutoff_date)
            valid_cutoff = True
        except CutoffWindowError:
            valid_cutoff = False
        violations: list[str] = []

        # ``is_model_endpoint`` is not merely descriptive: the recording
        # transport sets it only after matching a request against the resolved
        # model endpoint.  The adapter ledger must therefore contain *only*
        # those allowlisted model calls.  Volc is deliberately absent here: its
        # traffic is represented exclusively by the independently checked
        # gateway ledger below.
        if result is not None:
            adapter_events = tuple(result.network_events)
            non_model_events = [
                event
                for event in adapter_events
                if not event.is_model_endpoint
                and event.network_class != TEMPORAL_FILTER_NETWORK_CLASS
            ]
            if non_model_events:
                violations.append(f"adapter_network_not_model_allowlisted={len(non_model_events)}")

            model_events = [
                event
                for event in adapter_events
                if event.is_model_endpoint and event.network_class != TEMPORAL_FILTER_NETWORK_CLASS
            ]
            if not model_events:
                violations.append("missing_model_network_ledger")
            if len(model_events) > DEFAULT_TOOL_CONTRACT.max_model_calls:
                violations.append(f"model_network_call_budget_exceeded={len(model_events)}")
            if result.model_call_count != len(model_events):
                violations.append(
                    "model_network_ledger_mismatch="
                    f"reported:{result.model_call_count},observed:{len(model_events)}"
                )

        if not valid_cutoff:
            if not skipped_no_cutoff:
                violations.append("bad_or_missing_cutoff_date")
            if result is not None:
                violations.append("model_called_without_valid_cutoff")
            if search_outcomes:
                violations.append("tool_called_without_valid_cutoff")
            if tuple(network_ledger_rows):
                violations.append("network_used_without_valid_cutoff")
            return AgenticPolicyDecision(
                comparable=not violations and skipped_no_cutoff,
                violations=tuple(violations),
                skipped_no_cutoff=skipped_no_cutoff,
            )

        if skipped_no_cutoff:
            violations.append("valid_cutoff_task_was_skipped")
        if result is None:
            violations.append("missing_adapter_result")
        else:
            if result.model_call_count > DEFAULT_TOOL_CONTRACT.max_model_calls:
                violations.append(f"model_call_budget_exceeded={result.model_call_count}")
            if len(result.tool_calls) > DEFAULT_TOOL_CONTRACT.max_tool_calls:
                violations.append(f"tool_call_budget_exceeded={len(result.tool_calls)}")
            if result.error:
                violations.append("adapter_error")

        if not search_outcomes:
            violations.append("missing_search_audit")
        for outcome in search_outcomes:
            if outcome.tool_contract_hash != self.tool_contract_hash:
                violations.append("tool_contract_hash_mismatch")
            if not outcome.audit_records:
                violations.append("missing_search_audit")
            if outcome.error_code:
                violations.append(f"search_error:{outcome.error_code}")
            if outcome.tool_call_count > DEFAULT_TOOL_CONTRACT.max_tool_calls:
                violations.append("tool_call_budget_exceeded")
            if outcome.max_model_calls != DEFAULT_TOOL_CONTRACT.max_model_calls:
                violations.append("model_call_contract_mismatch")
            if outcome.max_tool_calls != DEFAULT_TOOL_CONTRACT.max_tool_calls:
                violations.append("tool_call_contract_mismatch")
            for audit in outcome.audit_records:
                if audit.cutoff_date != cutoff_date:
                    violations.append("audit_cutoff_date_mismatch")
                if not audit.exclusive_cutoff_bound.endswith("T00:00:00Z"):
                    violations.append("audit_cutoff_bound_invalid")
                if audit.accepted and audit.reason != "accepted":
                    violations.append("audit_accept_reason_invalid")
                if audit.accepted and not audit.published_at and audit.llm_verdict != "keep":
                    violations.append("accepted_candidate_missing_published_date")

        rows = tuple(network_ledger_rows)
        if rows:
            for row in rows:
                if row.get("network_class") == "search_service_local":
                    if row.get("host") != "127.0.0.1" or row.get("is_model_endpoint") is not False:
                        violations.append("search_service_local_endpoint_not_allowlisted")
                    continue
                if row.get("network_class") == "temporal_filter":
                    if row.get("is_model_endpoint") is not True:
                        violations.append("temporal_filter_not_model_endpoint")
                    continue
                if row.get("provider") != "volc" or row.get("host") != VOLC_SEARCH_HOST:
                    violations.append("search_endpoint_not_allowlisted")
                if row.get("method") != "POST" or row.get("path") != VOLC_SEARCH_PATH:
                    violations.append("search_method_or_path_not_allowlisted")
                if (
                    row.get("action") != VOLC_SEARCH_ACTION
                    or row.get("version") != VOLC_SEARCH_VERSION
                ):
                    violations.append("search_action_or_version_not_allowlisted")
                if not row.get("cutoff_compliant"):
                    violations.append("search_cutoff_not_compliant")
        else:
            violations.append("missing_search_network_ledger")

        return AgenticPolicyDecision(
            comparable=not violations,
            violations=tuple(_deduplicate(violations)),
            skipped_no_cutoff=False,
        )


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
