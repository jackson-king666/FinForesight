"""Fail-closed web-search primitives for the agentic benchmark profile.

The strict profile deliberately does not import or use this package.  The
objects here are shared by framework integrations so PydanticAI and LangGraph
receive the same Volc request shape, cutoff handling, budgets, and audit data.
"""

from benchmark_harness.search.cutoff import (
    BEventDateError,
    CutoffWindow,
    CutoffWindowError,
    validate_b_cutoff_not_after_event,
)
from benchmark_harness.search.filter import (
    LlmFilterRequest,
    LlmTemporalVerdict,
    VolcSearchCandidate,
    filter_volc_candidates,
    rule_filter_volc_candidate,
)
from benchmark_harness.search.policy import AgenticPolicyDecision, AgenticWebSearchPolicyGuard
from benchmark_harness.search.volc import (
    DEFAULT_TOOL_CONTRACT,
    AgenticSearchSession,
    CutoffSearchGateway,
    SearchCandidate,
    SearchCandidateAudit,
    SearchGatewayError,
    SearchOutcome,
    SearchToolBudget,
    ToolContract,
    VolcHttpResponse,
    VolcSdkTransport,
    VolcSearchTransport,
    write_search_audit_jsonl,
)
from benchmark_harness.search.volc_core import (
    VOLC_SEARCH_ACTION,
    VOLC_SEARCH_HOST,
    VOLC_SEARCH_PATH,
    VOLC_SEARCH_URL,
    VOLC_SEARCH_VERSION,
    build_cutoff_volc_request,
)

__all__ = [
    "DEFAULT_TOOL_CONTRACT",
    "AgenticPolicyDecision",
    "AgenticSearchSession",
    "AgenticWebSearchPolicyGuard",
    "BEventDateError",
    "CutoffSearchGateway",
    "CutoffWindow",
    "CutoffWindowError",
    "SearchCandidate",
    "SearchCandidateAudit",
    "SearchGatewayError",
    "SearchOutcome",
    "SearchToolBudget",
    "ToolContract",
    "LlmFilterRequest",
    "LlmTemporalVerdict",
    "VOLC_SEARCH_ACTION",
    "VOLC_SEARCH_HOST",
    "VOLC_SEARCH_PATH",
    "VOLC_SEARCH_URL",
    "VOLC_SEARCH_VERSION",
    "VolcHttpResponse",
    "VolcSdkTransport",
    "VolcSearchCandidate",
    "VolcSearchTransport",
    "build_cutoff_volc_request",
    "filter_volc_candidates",
    "rule_filter_volc_candidate",
    "write_search_audit_jsonl",
    "validate_b_cutoff_not_after_event",
]
