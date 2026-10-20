"""Typed, record-free contracts for external Deep Research systems."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from benchmark_harness.contracts import AdapterResult, RunContext, TaskInput


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _non_empty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _non_negative(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    """Comparable host-owned limits for model, retrieval, and tool work."""

    max_model_calls: int
    max_search_calls: int
    max_visit_calls: int
    max_other_tool_calls: int
    max_input_tokens: int
    max_output_tokens: int
    wall_clock_timeout_seconds: float

    def __post_init__(self) -> None:
        for name in (
            "max_model_calls",
            "max_search_calls",
            "max_visit_calls",
            "max_other_tool_calls",
            "max_input_tokens",
            "max_output_tokens",
        ):
            _non_negative(getattr(self, name), name)
        if (
            not isinstance(self.wall_clock_timeout_seconds, int | float)
            or isinstance(self.wall_clock_timeout_seconds, bool)
            or self.wall_clock_timeout_seconds <= 0
        ):
            raise ValueError("wall_clock_timeout_seconds must be a positive number.")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "max_model_calls": self.max_model_calls,
            "max_search_calls": self.max_search_calls,
            "max_visit_calls": self.max_visit_calls,
            "max_other_tool_calls": self.max_other_tool_calls,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "wall_clock_timeout_seconds": self.wall_clock_timeout_seconds,
        }

    @property
    def hash(self) -> str:
        return _hash_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class SystemDescriptor:
    """Complete identity of a benchmarked framework/strategy/backbone system."""

    system_id: str
    framework_id: str
    framework_revision: str
    strategy_id: str
    strategy_revision: str
    backbone_id: str
    backbone_training_cutoff: date
    cutoff_source: str
    adapter_id: str
    adapter_revision: str
    profile: str
    retrieval_regime: str
    ranking_eligible: bool = False
    license_status: str = "UNKNOWN"

    def __post_init__(self) -> None:
        for name in (
            "system_id",
            "framework_id",
            "framework_revision",
            "strategy_id",
            "strategy_revision",
            "backbone_id",
            "cutoff_source",
            "adapter_id",
            "adapter_revision",
            "profile",
            "retrieval_regime",
            "license_status",
        ):
            _non_empty(getattr(self, name), name)
        if not isinstance(self.backbone_training_cutoff, date):
            raise ValueError("backbone_training_cutoff must be a date.")
        if not isinstance(self.ranking_eligible, bool):
            raise ValueError("ranking_eligible must be a boolean.")

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "system_id": self.system_id,
            "framework_id": self.framework_id,
            "framework_revision": self.framework_revision,
            "strategy_id": self.strategy_id,
            "strategy_revision": self.strategy_revision,
            "backbone_id": self.backbone_id,
            "backbone_training_cutoff": self.backbone_training_cutoff.isoformat(),
            "cutoff_source": self.cutoff_source,
            "adapter_id": self.adapter_id,
            "adapter_revision": self.adapter_revision,
            "profile": self.profile,
            "retrieval_regime": self.retrieval_regime,
            "ranking_eligible": self.ranking_eligible,
            "license_status": self.license_status,
        }
        return value

    @property
    def descriptor_hash(self) -> str:
        return _hash_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class TaskCapability:
    """Task-scoped capabilities safe to hand to a child system.

    Capability tokens are intentionally write-only from the provenance view;
    callers must not serialize them in artifacts.  No field can carry a
    dataset record, evaluator state, provider credential, or general client.
    """

    service_url: str | None
    service_token: str | None
    model_broker_capability: str | None
    budget: BudgetSpec
    allow_calculation: bool
    allow_citations: bool
    require_structured_output: bool
    run_id: str
    trial_id: str
    trace_mode: str

    def __post_init__(self) -> None:
        for name in ("run_id", "trial_id", "trace_mode"):
            _non_empty(getattr(self, name), name)
        for name in ("service_url", "service_token", "model_broker_capability"):
            value = getattr(self, name)
            if value is not None:
                _non_empty(value, name)
        for name in ("allow_calculation", "allow_citations", "require_structured_output"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")

    def public_dict(self) -> dict[str, Any]:
        return {
            "service_url": self.service_url,
            "budget": self.budget.to_dict(),
            "budget_sha256": self.budget.hash,
            "allow_calculation": self.allow_calculation,
            "allow_citations": self.allow_citations,
            "require_structured_output": self.require_structured_output,
            "run_id": self.run_id,
            "trial_id": self.trial_id,
            "trace_mode": self.trace_mode,
        }


@dataclass(frozen=True, slots=True)
class ExperimentBinding:
    """Immutable identity binding for one planned system/task attempt."""

    experiment_id: str
    plan_id: str
    plan_hash: str
    cell_id: str
    attempt_id: str
    descriptor_hash: str
    task_id: str
    retrieval_regime: str
    trial: int
    budget: BudgetSpec
    # Added after the initial 0A contract so adapters can inject the declared
    # experiment clock without consulting host machine time.  The default is
    # retained for legacy v3 fixture bindings; clock-dependent adapters fail
    # closed when it is absent.
    experiment_as_of: date | None = None
    calc_policy: str = "calc_on"

    def __post_init__(self) -> None:
        for name in (
            "experiment_id",
            "plan_id",
            "plan_hash",
            "cell_id",
            "attempt_id",
            "descriptor_hash",
            "task_id",
            "retrieval_regime",
        ):
            _non_empty(getattr(self, name), name)
        if not isinstance(self.trial, int) or isinstance(self.trial, bool) or self.trial < 1:
            raise ValueError("trial must be a positive integer.")
        if self.experiment_as_of is not None and not isinstance(self.experiment_as_of, date):
            raise ValueError("experiment_as_of must be a date when provided.")
        if self.calc_policy not in {"calc_on", "calc_off"}:
            raise ValueError("calc_policy must be calc_on or calc_off.")

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "cell_id": self.cell_id,
            "attempt_id": self.attempt_id,
            "descriptor_hash": self.descriptor_hash,
            "task_id": self.task_id,
            "retrieval_regime": self.retrieval_regime,
            "trial": self.trial,
            "budget": self.budget.to_dict(),
            "budget_sha256": self.budget.hash,
            "experiment_as_of": (
                self.experiment_as_of.isoformat() if self.experiment_as_of is not None else None
            ),
        }
        if self.calc_policy != "calc_on":
            value["calc_policy"] = self.calc_policy
        return value


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Non-ranking readiness result returned before a task is executed."""

    ready: bool
    reasons: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ready, bool):
            raise ValueError("ready must be a boolean.")
        if any(not isinstance(reason, str) or not reason for reason in self.reasons):
            raise ValueError("preflight reasons must be non-empty strings.")


class DeepResearchSystem(Protocol):
    """Common host-owned lifecycle for external Deep Research adapters."""

    descriptor: SystemDescriptor

    def preflight(self, binding: ExperimentBinding) -> PreflightResult: ...

    def run_task(
        self,
        task: TaskInput,
        context: RunContext,
        capability: TaskCapability,
        binding: ExperimentBinding | None = None,
    ) -> AdapterResult: ...

    def close(self) -> None: ...
