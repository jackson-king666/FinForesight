"""Binding-aware FinanceHarness bridge for the exploratory execution plan.

FinanceHarness keeps its established v2 child IPC contract.  This module adds
the common v3-era lifecycle around that contract: source verification,
regime-owned capability selection, immutable cell binding, and an explicit
non-ranking descriptor.  It never creates a checkout or an environment during
execution; callers must provide the already verified manager/source.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmark_harness.config import (
    FINANCEHARNESS_LICENSE_STATUS,
    FINANCEHARNESS_PROTOCOL,
    FINANCEHARNESS_PYPROJECT_SHA256,
    FINANCEHARNESS_REPO_URL,
    FINANCEHARNESS_REVISION,
    FINANCEHARNESS_TREE_SHA,
    FINANCEHARNESS_UV_LOCK_SHA256,
    ExternalHarnessConfig,
    ResolvedModelProfile,
)
from benchmark_harness.contracts import AdapterResult, RunContext, TaskInput
from benchmark_harness.external_harnesses.checkout import (
    ExternalHarnessError,
    ExternalHarnessManager,
)
from benchmark_harness.external_harnesses.financeharness.adapter import (
    FinanceHarnessAdapter,
    FinanceHarnessBoundaryError,
)
from benchmark_harness.search_service import SearchServiceClient
from benchmark_harness.search_service.config import SearchServiceClientConfig
from benchmark_harness.systems.contracts import (
    DeepResearchSystem,
    ExperimentBinding,
    PreflightResult,
    SystemDescriptor,
    TaskCapability,
)

FINANCEHARNESS_ADAPTER_ID = "financeharness-external-stdio-v2"
FINANCEHARNESS_ADAPTER_REVISION = "2"
FINANCEHARNESS_MODEL_ROLES = ("research", "analysis", "final")


def _default_external_config(source: Path) -> ExternalHarnessConfig:
    if source.name != "financeharness":
        raise ExternalHarnessError(
            "FinanceHarness source must be the named financeharness checkout directory"
        )
    return ExternalHarnessConfig(
        checkout_root=source.parent,
        offline=True,
        license_status=FINANCEHARNESS_LICENSE_STATUS,
    )


class FinanceHarnessSystem(DeepResearchSystem):
    """Use the pinned FinanceHarness worker through its v2 compatibility bridge."""

    adapter_id = FINANCEHARNESS_ADAPTER_ID
    adapter_revision = FINANCEHARNESS_ADAPTER_REVISION

    def __init__(
        self,
        *,
        descriptor: SystemDescriptor,
        source_dir: str | Path,
        repo_root: str | Path | None = None,
        manager: ExternalHarnessManager | None = None,
        search_service: SearchServiceClient | None = None,
        model_responder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        worker_command: list[str] | None = None,
        process_factory: Callable[..., Any] | None = None,
    ) -> None:
        if descriptor.adapter_id != self.adapter_id:
            raise ValueError("descriptor adapter_id does not match FinanceHarness bridge")
        if descriptor.ranking_eligible:
            raise ValueError("FinanceHarness exploratory bridge is never ranking eligible")
        self.descriptor = descriptor
        self.source_dir = Path(source_dir).expanduser().resolve()
        self.repo_root = Path(repo_root or Path.cwd()).expanduser().resolve()
        self.manager = manager or ExternalHarnessManager(
            _default_external_config(self.source_dir),
            repo_root=self.repo_root,
        )
        self.search_service = search_service
        self._adapter = FinanceHarnessAdapter(
            self.manager,
            search_service=search_service,
            model_responder=model_responder,
            worker_command=worker_command,
            process_factory=process_factory,
            repo_root=self.repo_root,
        )
        self._closed = False

    def preflight(self, binding: ExperimentBinding) -> PreflightResult:
        reasons: list[str] = []
        checks: dict[str, bool] = {
            "descriptor_hash": binding.descriptor_hash == self.descriptor.descriptor_hash,
            "ranking_disabled": self.descriptor.ranking_eligible is False,
            "protocol_bridge": FINANCEHARNESS_PROTOCOL == "external_research_stdio/v2",
            "retrieval_regime": binding.retrieval_regime == self.descriptor.retrieval_regime,
            "closed_book_search_absent": (
                binding.retrieval_regime != "closed_book" or self.search_service is None
            ),
            "cutoff_search_present": (
                binding.retrieval_regime == "closed_book" or self.search_service is not None
            ),
        }
        for name, passed in checks.items():
            if not passed:
                reasons.append(name + "_failed")
        provenance: dict[str, Any] = {
            "repo_url": FINANCEHARNESS_REPO_URL,
            "revision": FINANCEHARNESS_REVISION,
            "tree_sha": FINANCEHARNESS_TREE_SHA,
            "dependency_lock_sha256": FINANCEHARNESS_UV_LOCK_SHA256,
            "project_sha256": FINANCEHARNESS_PYPROJECT_SHA256,
            "protocol": FINANCEHARNESS_PROTOCOL,
            "license_status": FINANCEHARNESS_LICENSE_STATUS,
            "ranking_eligible": False,
        }
        try:
            provenance.update(self.manager.verify())
            checks["source_environment"] = True
        except (ExternalHarnessError, OSError, ValueError) as exc:
            checks["source_environment"] = False
            reasons.append("source_environment:" + str(exc))
        return PreflightResult(
            ready=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
            provenance={"checks": checks, **provenance},
        )

    def run_task(
        self,
        task: TaskInput,
        context: RunContext,
        capability: TaskCapability,
        binding: ExperimentBinding | None = None,
    ) -> AdapterResult:
        if self._closed:
            raise FinanceHarnessBoundaryError("financeharness_adapter_is_closed")
        if binding is None:
            raise FinanceHarnessBoundaryError("experiment_binding_required")
        if task.task_id != binding.task_id:
            raise FinanceHarnessBoundaryError("task_binding_mismatch")
        if capability.budget != binding.budget:
            raise FinanceHarnessBoundaryError("capability_budget_mismatch")
        preflight = self.preflight(binding)
        if not preflight.ready:
            raise FinanceHarnessBoundaryError(
                "adapter_preflight_failed:" + ",".join(preflight.reasons)
            )
        return self._adapter.run_deep(
            task,
            context,
            retrieval_regime=binding.retrieval_regime,
        ).result

    def close(self) -> None:
        self._closed = True


def build_financeharness_descriptor(
    *,
    system_id: str,
    backbone_id: str,
    backbone_training_cutoff: Any,
    cutoff_source: str,
    retrieval_regime: str = "cutoff_controlled",
) -> SystemDescriptor:
    return SystemDescriptor(
        system_id=system_id,
        framework_id="financeharness",
        framework_revision=FINANCEHARNESS_REVISION,
        strategy_id="financeharness-deep",
        strategy_revision=FINANCEHARNESS_REVISION,
        backbone_id=backbone_id,
        backbone_training_cutoff=backbone_training_cutoff,
        cutoff_source=cutoff_source,
        adapter_id=FINANCEHARNESS_ADAPTER_ID,
        adapter_revision=FINANCEHARNESS_ADAPTER_REVISION,
        profile="deep_research",
        retrieval_regime=retrieval_regime,
        ranking_eligible=False,
        license_status=FINANCEHARNESS_LICENSE_STATUS,
    )


def real_financeharness_system(
    system: Any,
    retrieval_regime: str,
    *,
    repo_root: str | Path | None = None,
    search_service: SearchServiceClient | None = None,
    model_profile: ResolvedModelProfile | None = None,
    model_responder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> FinanceHarnessSystem:
    """Construct FinanceHarness only from an explicit pinned source path."""

    source = os.environ.get("BENCHMARK_FINANCE_SOURCE")
    if not source:
        raise ExternalHarnessError(
            "BENCHMARK_FINANCE_SOURCE is required for real FinanceHarness smoke"
        )
    if search_service is None and retrieval_regime == "cutoff_controlled":
        service_url = os.environ.get("BENCHMARK_SEARCH_SERVICE_URL", "http://127.0.0.1:8765")
        token_file = os.environ.get(
            "BENCHMARK_SEARCH_SERVICE_ADMIN_TOKEN_FILE",
            str(Path(repo_root or Path.cwd()) / "data/search-service/admin.token"),
        )
        search_service = SearchServiceClient(
            SearchServiceClientConfig(
                base_url=service_url,
                admin_token_file=token_file,
                expected_service_cohort_sha256=os.environ.get(
                    "BENCHMARK_EXPECTED_SERVICE_COHORT_SHA256"
                ),
            )
        )
    del model_profile  # The executor owns the per-task RunContext model profile.
    return FinanceHarnessSystem(
        descriptor=system.descriptor(retrieval_regime),
        source_dir=source,
        repo_root=repo_root,
        search_service=search_service,
        model_responder=model_responder,
    )


__all__ = [
    "FINANCEHARNESS_ADAPTER_ID",
    "FINANCEHARNESS_ADAPTER_REVISION",
    "FINANCEHARNESS_MODEL_ROLES",
    "FinanceHarnessSystem",
    "build_financeharness_descriptor",
    "real_financeharness_system",
]
