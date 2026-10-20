"""Execution and ranking eligibility gates for registered systems."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from benchmark_harness.experiment.manifest import SystemBindingSpec

REQUIRED_RANKING_GATES = (
    "immutable_provenance",
    "security_review",
    "unified_backbone",
    "budget_audit",
    "deterministic_clock",
    "controlled_search_e2e",
    "license_resolved",
)


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    system_id: str
    execution_eligible: bool
    ranking_eligible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EligibilityReport:
    """Materialized v3 ranking request and the gates that actually decide it.

    ``ranking_requested`` is an explicit paper-pack intent.  It is deliberately
    separate from ``ranking_eligible`` so a new pack can remember to request a
    ranking without bypassing the existing comparability and provenance gates.
    """

    ranking_requested: bool
    ranking_eligible: bool
    retrieval_regime: str
    gates: Mapping[str, bool]
    comparable: bool
    artifact_complete: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranking_requested": self.ranking_requested,
            "ranking_eligible": self.ranking_eligible,
            "retrieval_regime": self.retrieval_regime,
            "gates": dict(sorted(self.gates.items())),
            "comparable": self.comparable,
            "artifact_complete": self.artifact_complete,
            "reasons": list(self.reasons),
        }


def evaluate_ranking_eligibility(
    *,
    ranking_requested: bool,
    retrieval_regime: str,
    gates: Mapping[str, bool],
    comparable: bool,
    artifact_complete: bool,
) -> EligibilityReport:
    """Compute effective v3 ranking eligibility without changing v2 defaults.

    The function is intentionally independent of ``SystemDescriptor`` and the
    legacy experiment manifest.  It is used only by the new paper-pack
    materializer, while the current v1/v2 execution path remains
    non-ranking-by-construction.
    """

    if not isinstance(ranking_requested, bool):
        raise ValueError("ranking_requested must be a boolean")
    if not isinstance(retrieval_regime, str) or not retrieval_regime.strip():
        raise ValueError("retrieval_regime must be a non-empty string")
    if not isinstance(gates, Mapping):
        raise ValueError("ranking gates must be an object")
    normalized_gates: dict[str, bool] = {}
    for gate, passed in gates.items():
        if not isinstance(gate, str) or not gate.strip():
            raise ValueError("ranking gate names must be non-empty strings")
        if not isinstance(passed, bool):
            raise ValueError(f"ranking gate {gate!r} must be a boolean")
        normalized_gates[gate] = passed
    if not isinstance(comparable, bool) or not isinstance(artifact_complete, bool):
        raise ValueError("comparable and artifact_complete must be booleans")

    reasons: list[str] = []
    if not ranking_requested:
        reasons.append("ranking was not requested")
    if retrieval_regime != "cutoff_controlled":
        reasons.append("ranking regime must be cutoff_controlled")
    if retrieval_regime == "unrestricted_current_web":
        reasons.append("unrestricted_current_web is isolated from the main leaderboard")
    for gate in REQUIRED_RANKING_GATES:
        if not normalized_gates.get(gate, False):
            reasons.append(f"missing ranking gate: {gate}")
    if not comparable:
        reasons.append("artifact population is not comparable")
    if not artifact_complete:
        reasons.append("ranking artifacts are incomplete")

    return EligibilityReport(
        ranking_requested=ranking_requested,
        ranking_eligible=not reasons,
        retrieval_regime=retrieval_regime,
        gates=normalized_gates,
        comparable=comparable,
        artifact_complete=artifact_complete,
        reasons=tuple(reasons),
    )


def evaluate_system_eligibility(
    system: SystemBindingSpec,
    *,
    passed_gates: Iterable[str] = (),
) -> EligibilityDecision:
    """Keep new systems runnable for fixtures but non-ranking by default."""

    passed = set(passed_gates)
    reasons: list[str] = []
    if system.license_status.upper() in {"CONFLICTING", "UNDECLARED", "UNKNOWN"}:
        reasons.append("license provenance is unresolved")
    missing = [gate for gate in REQUIRED_RANKING_GATES if gate not in passed]
    reasons.extend(f"missing ranking gate: {gate}" for gate in missing)
    return EligibilityDecision(
        system_id=system.system_id,
        execution_eligible=True,
        ranking_eligible=not reasons,
        reasons=tuple(reasons),
    )
