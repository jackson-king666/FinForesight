"""Mechanical fairness gates for the one_turn_strict profile."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from benchmark_harness.contracts import (
    AdapterResult,
    TaskInput,
    canonical_input_hash,
    canonical_messages,
)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    comparable: bool
    violations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"comparable": self.comparable, "violations": list(self.violations)}


class StrictPolicyGuard:
    """Reject input/control-flow drift instead of averaging it into a score."""

    def check(self, task: TaskInput, result: AdapterResult) -> PolicyDecision:
        expected_messages = canonical_messages(task.prompt)
        expected_hash = canonical_input_hash(task.prompt)
        violations: list[str] = []

        if result.canonical_input_hash != expected_hash:
            violations.append("canonical_input_hash_mismatch")
        if result.effective_input_hash != expected_hash:
            violations.append("effective_input_hash_mismatch")
        if list(result.effective_messages) != expected_messages:
            violations.append("effective_messages_not_exactly_one_user_prompt")

        request_messages = result.effective_request.get("messages")
        if request_messages != expected_messages:
            violations.append("effective_request_messages_mismatch")
        if result.effective_request.get("prompt_sha256") != _prompt_hash(task):
            violations.append("effective_request_prompt_hash_mismatch")
        # A configured retry policy is allowed for reliability.  If it is
        # actually used, the existing one-call/network gates below mark that
        # task non-comparable rather than silently relaxing strict fairness.
        if result.model_call_count != 1:
            violations.append(f"model_call_count={result.model_call_count}")
        if result.tool_calls:
            violations.append(f"tool_calls={len(result.tool_calls)}")

        non_model_events = [event for event in result.network_events if not event.is_model_endpoint]
        if non_model_events:
            violations.append(f"non_model_network_calls={len(non_model_events)}")

        if not result.network_events:
            violations.append("missing_model_network_ledger")
        model_events = [event for event in result.network_events if event.is_model_endpoint]
        if len(model_events) != 1:
            violations.append(f"model_network_calls={len(model_events)}")
        if result.error:
            violations.append("adapter_error")
        if not result.raw_text or not result.raw_text.strip():
            violations.append("empty_raw_response")

        event_types = {event.event_type for event in result.trace_events}
        required = {"model_request", "model_response"}
        missing = sorted(required - event_types)
        if missing:
            violations.append("missing_trace_events:" + ",".join(missing))

        return PolicyDecision(comparable=not violations, violations=tuple(violations))


def _prompt_hash(task: TaskInput) -> str:
    from benchmark_harness.contracts import sha256_text

    return sha256_text(task.prompt)


def violation_counts(decisions: Iterable[PolicyDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        for violation in decision.violations:
            counts[violation] = counts.get(violation, 0) + 1
    return counts
