"""Binding-aware, append-only execution for experiment-plan/v2."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.contracts import RunContext, TaskInput
from benchmark_harness.dataset_sources.huggingface import HuggingFaceDatasetSource
from benchmark_harness.experiment.execution_contract import ensure_execution_contract
from benchmark_harness.experiment.live_model import declared_model_identity
from benchmark_harness.experiment.manifest import (
    ExperimentManifest,
    ManifestError,
    SystemBindingSpec,
)
from benchmark_harness.experiment.planner import ExperimentPlan, PlanCell, load_plan
from benchmark_harness.experiment.schedule import AttemptRecord, append_attempt, read_attempts
from benchmark_harness.systems.contracts import (
    BudgetSpec,
    DeepResearchSystem,
    ExperimentBinding,
    PreflightResult,
    TaskCapability,
)
from benchmark_harness.systems.regimes import RetrievalPolicy
from benchmark_harness.systems.worker_diagnostics import worker_failure_subreason

Selection = Literal["smoke", "dev", "test", "all"]


class ExecutionError(RuntimeError):
    """Raised when an execution request violates identity or policy."""


class ExecutionDependencyError(ExecutionError):
    """Raised when a real pinned-system dependency is not available."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _default_model_profile() -> ResolvedModelProfile:
    return ResolvedModelProfile(
        mode="mock",
        endpoint="mock://model",
        model_name="exploratory-smoke-model",
        api_key=None,
        temperature=1.0,
        max_tokens=65_536,
        timeout_seconds=60.0,
        retries=0,
        seed=20260817,
    )


def _budget_from_mapping(raw: Mapping[str, Any]) -> BudgetSpec:
    if not isinstance(raw, Mapping):
        raise ManifestError("manifest budget must be an object")

    def value(*names: str, default: int | float = 0) -> int | float:
        for name in names:
            if name in raw:
                return raw[name]
        return default

    return BudgetSpec(
        max_model_calls=int(value("model_calls", "max_model_calls")),
        max_search_calls=int(value("searches", "search_calls", "max_search_calls")),
        max_visit_calls=int(value("visits", "visit_calls", "max_visit_calls")),
        max_other_tool_calls=int(value("other_tools", "other_tool_calls", "max_other_tool_calls")),
        max_input_tokens=int(value("input_tokens", "max_input_tokens")),
        max_output_tokens=int(value("output_tokens", "max_output_tokens")),
        wall_clock_timeout_seconds=float(value("wall_clock_seconds", "wall_clock_timeout_seconds")),
    )


def _budget_map(manifest: ExperimentManifest) -> dict[str, BudgetSpec]:
    raw = manifest.v2_metadata.get("budgets")
    if isinstance(raw, Mapping):
        result: dict[str, BudgetSpec] = {}
        for system in manifest.systems:
            system_raw = raw.get(system.system_id, manifest.v2_metadata.get("budget", {}))
            if not isinstance(system_raw, Mapping):
                raise ManifestError(f"budget for {system.system_id!r} must be an object")
            result[system.system_id] = _budget_from_mapping(system_raw)
        return result
    raw_budget = manifest.v2_metadata.get("budget", {})
    if not isinstance(raw_budget, Mapping):
        raise ManifestError("manifest budget must be an object")
    budget = _budget_from_mapping(raw_budget)
    return {system.system_id: budget for system in manifest.systems}


TaskLoader = Callable[[], Iterable[Mapping[str, Any]]]
SystemFactory = Callable[[SystemBindingSpec, str], DeepResearchSystem]
CapabilityFactory = Callable[[ExperimentBinding, TaskInput], TaskCapability]
ModelProfileFactory = Callable[[str], ResolvedModelProfile]
SearchSessionFactory = Callable[[ExperimentBinding, TaskInput, str], Any]
LiveValidation = Callable[[tuple[PlanCell, ...]], dict[str, Any]]
ServiceIdentityFactory = Callable[[PlanCell], Mapping[str, Any] | None]


@dataclass(frozen=True, slots=True)
class ExecutionDependencies:
    """Explicit dependency seams; no default seam silently substitutes a fixture."""

    system_factory: SystemFactory
    task_loader: TaskLoader | None = None
    model_profile: ResolvedModelProfile = field(default_factory=_default_model_profile)
    capability_factory: CapabilityFactory | None = None
    model_profile_factory: ModelProfileFactory | None = None
    search_session_factory: SearchSessionFactory | None = None
    service_identity_factory: ServiceIdentityFactory | None = None
    live_validation: LiveValidation | None = None

    def load_tasks(self) -> dict[str, Mapping[str, Any]]:
        loader = self.task_loader or (lambda: HuggingFaceDatasetSource(offline=True).load_records())
        rows = tuple(loader())
        result = {
            str(row["task_id"]): row
            for row in rows
            if isinstance(row, Mapping) and row.get("task_id")
        }
        if len(result) != len(rows):
            raise ExecutionError("task loader returned duplicate or malformed task IDs")
        return result


@dataclass(frozen=True, slots=True)
class ExperimentExecutionRequest:
    manifest: ExperimentManifest | str | Path
    plan: ExperimentPlan | str | Path
    results_root: str | Path
    selection: Selection = "all"
    max_workers: int = 3
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.selection not in {"smoke", "dev", "test", "all"}:
            raise ValueError("selection must be smoke, dev, test, or all")
        if (
            not isinstance(self.max_workers, int)
            or isinstance(self.max_workers, bool)
            or self.max_workers < 1
        ):
            raise ValueError("max_workers must be positive")
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be positive")


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    experiment_id: str
    plan_id: str
    selection: str
    status: str
    total_cells: int
    completed_cells: int
    pending_cells: int
    failed_cells: int
    exhausted_cells: int
    attempts_written: int
    results_root: str
    reasons: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.status == "complete"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        value["complete"] = self.complete
        return value


def _resolve_manifest(value: ExperimentManifest | str | Path) -> ExperimentManifest:
    if isinstance(value, ExperimentManifest):
        return value
    from benchmark_harness.experiment.manifest import load_manifest

    return load_manifest(value)


def _resolve_plan(
    value: ExperimentPlan | str | Path, manifest: ExperimentManifest
) -> ExperimentPlan:
    if isinstance(value, ExperimentPlan):
        plan = value
    else:
        plan = load_plan(value, expected_manifest_hash=manifest.manifest_hash)
    if plan.manifest_hash != manifest.manifest_hash:
        raise ExecutionError("manifest hash does not match plan")
    return plan


def _selected_cells(plan: ExperimentPlan, selection: Selection) -> tuple[PlanCell, ...]:
    if selection == "all":
        return plan.cells
    if selection == "smoke":
        return tuple(cell for cell in plan.cells if cell.smoke)
    return tuple(cell for cell in plan.cells if cell.split == selection)


def _task_input(row: Mapping[str, Any]) -> TaskInput:
    seed = row.get("seed")
    cutoff = seed.get("cutoff_date") if isinstance(seed, Mapping) else None
    cutoff_date = str(cutoff)[:10] if cutoff is not None else None
    return TaskInput(
        task_id=str(row["task_id"]), prompt=str(row["prompt"]), cutoff_date=cutoff_date
    )


def _default_capability(binding: ExperimentBinding, task: TaskInput) -> TaskCapability:
    return TaskCapability(
        service_url=None,
        service_token=(
            "offline-smoke-token"
            if binding.retrieval_regime in {"cutoff_controlled", "unrestricted_current_web"}
            else None
        ),
        model_broker_capability="host-model-broker/v1",
        budget=binding.budget,
        allow_calculation=binding.calc_policy == "calc_on",
        allow_citations=binding.retrieval_regime != "closed_book",
        require_structured_output=True,
        run_id=binding.experiment_id,
        trial_id=f"{binding.cell_id}:trial-{binding.trial}",
        trace_mode="metadata_only",
    )


def _result_payload(
    result: Any,
    *,
    binding: ExperimentBinding,
    preflight: PreflightResult,
    model_provenance: Mapping[str, Any] | None = None,
    search_session: Mapping[str, Any] | None = None,
    service_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any]
    if hasattr(result, "__dataclass_fields__"):
        payload = asdict(result)
    elif isinstance(result, Mapping):
        payload = dict(result)
    else:
        payload = {"raw_text": str(result)}
    value = {
        "binding": binding.to_dict(),
        "preflight": {
            "ready": preflight.ready,
            "reasons": list(preflight.reasons),
            "provenance": preflight.provenance,
        },
        "result": payload,
        "ranking_eligible": False,
    }
    if model_provenance is not None:
        value["model_provenance"] = _merge_observed_model_provenance(model_provenance, payload)
    if search_session is not None:
        value["search_session"] = dict(search_session)
    if service_identity is not None:
        value["service_identity"] = dict(service_identity)
    return value


def _merge_observed_model_provenance(base: Mapping[str, Any], payload: Any) -> dict[str, Any]:
    """Carry provider attestations and system fingerprints into attempt lineage."""

    provider_names = set(_string_values(base.get("provider_response_model_names")))
    provider_names.update(_string_values(base.get("provider_model_names")))
    fingerprints = set(_string_values(base.get("system_fingerprints")))

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key in ("provider_response_model_names", "provider_model_names"):
                provider_names.update(_string_values(value.get(key)))
            fingerprints.update(_string_values(value.get("system_fingerprints")))
            for child in value.values():
                visit(child)
        elif isinstance(value, list | tuple):
            for child in value:
                visit(child)

    visit(payload)
    result = dict(base)
    result["provider_response_model_names"] = sorted(provider_names)
    # Keep the old key in the persisted provenance for readers of prior packs.
    result["provider_model_names"] = sorted(provider_names)
    result["system_fingerprints"] = sorted(fingerprints)
    return result


def _string_values(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple | set | frozenset):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _session_search_handler(session: Any) -> Callable[[str, str, int], Any]:
    """Wrap a task-scoped service session as a token-blind search handler."""

    def handler(query: str, _token: str, _call_index: int) -> list[Mapping[str, Any]]:
        search_web = getattr(session, "search_web", None)
        if not callable(search_web):
            raise ExecutionError("search session does not expose search_web")
        raw = search_web(query)
        if not isinstance(raw, list):
            raise ExecutionError("search session returned a non-list payload")
        return raw

    descriptor = getattr(session, "descriptor", None)
    token = getattr(descriptor, "task_token", None)
    if isinstance(token, str) and token.strip():
        handler._task_token = token  # type: ignore[attr-defined]
    return handler


def _session_provenance(session: Any, *, close_status: str | None) -> dict[str, Any]:
    descriptor = getattr(session, "descriptor", None)
    receipts = tuple(getattr(session, "receipts", ()) or ())
    return {
        "session_id": getattr(descriptor, "session_id", None),
        "task_id": getattr(descriptor, "task_id", None),
        "profile": getattr(descriptor, "profile", None),
        "search_principal": getattr(descriptor, "baseline", None),
        "cutoff_date": getattr(descriptor, "cutoff_date", None),
        "policy_version": getattr(descriptor, "policy_version", None),
        "api_contract_sha256": getattr(descriptor, "api_contract_sha256", None),
        "implementation_manifest_sha256": getattr(
            descriptor, "implementation_manifest_sha256", None
        ),
        "capability_sha256": getattr(descriptor, "capability_sha256", None),
        "service_cohort_sha256": getattr(descriptor, "service_cohort_sha256", None),
        "instance_id": getattr(descriptor, "instance_id", None),
        "receipt_sha256s": [receipt.receipt_sha256 for receipt in receipts],
        "receipt_count": len(receipts),
        "close_status": close_status,
    }


def _assert_session_principal(session: Any, *, cell: PlanCell) -> None:
    """Reject cross-cell or wrong-profile principal drift before execution.

    The session must carry the exact manifest ``system_id`` of the cell and
    the ``deep_research`` profile; anything else is a provenance breach and
    fails before any model or search side effect.
    """

    descriptor = getattr(session, "descriptor", None)
    principal = getattr(descriptor, "baseline", None) if descriptor is not None else None
    profile_name = getattr(descriptor, "profile", None) if descriptor is not None else None
    if principal != cell.system_id:
        raise ExecutionError(
            f"search principal mismatch: cell system {cell.system_id!r} "
            f"received session principal {principal!r}"
        )
    if profile_name != "deep_research":
        raise ExecutionError(
            f"session profile mismatch: cell system {cell.system_id!r} "
            f"received profile {profile_name!r}"
        )


def assert_calc_policy(cell: PlanCell, capability: TaskCapability) -> None:
    """Fail closed when a v4 calc-off cell receives a calculator capability."""

    if cell.calc_policy not in {"calc_on", "calc_off"}:
        raise ExecutionError(f"unknown calculation policy {cell.calc_policy!r}")
    expected = cell.calc_policy == "calc_on"
    if capability.allow_calculation is not expected:
        raise ExecutionError(
            f"calc capability leakage: cell={cell.cell_id} policy={cell.calc_policy} "
            f"capability.allow_calculation={capability.allow_calculation}"
        )


def _artifact_path(root: Path, cell: PlanCell, attempt: int) -> Path:
    return root / "cells" / cell.cell_id / f"attempt-{attempt}.json"


def _write_artifact(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (_stable_json(payload) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _run(
    request: ExperimentExecutionRequest,
    *,
    dependencies: ExecutionDependencies,
    resume: bool,
) -> ExecutionSummary:
    manifest = _resolve_manifest(request.manifest)
    plan = _resolve_plan(request.plan, manifest)
    if manifest.schema_version not in {"experiment/v2", "experiment/v4"}:
        raise ExecutionError("the executor requires an experiment/v2 or experiment/v4 manifest")
    cells = _selected_cells(plan, request.selection)
    root = Path(request.results_root).expanduser().resolve() / manifest.experiment_id / plan.plan_id
    contract_path = root / "execution-contract.json"
    existing_leaf = contract_path.exists() or (root.is_dir() and any(root.iterdir()))
    if dependencies.live_validation is not None and not existing_leaf:
        # Resolve the environment-only model gate before creating any result
        # file.  Existing leaves are contract-validated first so a reused or
        # tampered leaf cannot trigger model/search-service work first.
        from benchmark_harness.experiment.live_model import resolve_live_model_profile

        resolve_live_model_profile(manifest, cell_id="preflight")
    execution_contract = ensure_execution_contract(root, manifest, plan)
    if dependencies.live_validation is not None:
        live_runtime = dependencies.live_validation(cells)
    else:
        live_runtime = None
    ledger_path = root / "attempts.jsonl"
    previous = read_attempts(ledger_path)
    previous_by_cell: dict[str, list[AttemptRecord]] = {}
    for record in previous:
        previous_by_cell.setdefault(record.cell_id, []).append(record)
    task_rows = dependencies.load_tasks()
    missing_tasks = sorted({cell.task_id for cell in cells} - set(task_rows))
    if missing_tasks:
        raise ExecutionError("task loader is missing planned task(s): " + ",".join(missing_tasks))
    budgets = _budget_map(manifest)
    binding_by_cell = {
        binding.cell_id: binding
        for binding in plan.bindings(manifest, budgets)
        if binding.cell_id in {cell.cell_id for cell in cells}
    }
    reasons: list[str] = []
    identity = declared_model_identity(manifest)
    summary_model_provenance: dict[str, Any] = {
        "model_name": identity["wire_model_name"],
        "wire_model_name": identity["wire_model_name"],
        "snapshot_model_name": identity["snapshot_model_name"],
        "backbone_id": identity["backbone_id"],
        "wire_name_binding": identity["wire_name_binding"],
        "provider_response_model_names": [],
        "provider_model_names": [],
        "system_fingerprints": [],
    }
    completed = pending = failed = exhausted = attempts_written = 0
    terminal_states: dict[str, str] = {}
    work: list[tuple[PlanCell, ExperimentBinding, int]] = []
    for cell in cells:
        history = previous_by_cell.get(cell.cell_id, [])
        if any(record.status == "succeeded" for record in history):
            completed += 1
            terminal_states[cell.cell_id] = "succeeded"
            continue
        if history and not resume:
            pending += 1
            terminal_states[cell.cell_id] = "pending"
            continue
        attempt = max((record.attempt for record in history), default=0) + 1
        if attempt > request.max_attempts:
            exhausted += 1
            terminal_states[cell.cell_id] = "exhausted"
            continue
        planned_binding = binding_by_cell[cell.cell_id]
        # Plan cells describe the first attempt.  A resumed attempt is a new
        # immutable binding and must identify itself in its artifact; otherwise
        # retry evidence is indistinguishable from attempt one.
        binding = replace(
            planned_binding,
            attempt_id=f"{cell.cell_id}:attempt-{attempt}",
        )
        work.append((cell, binding, attempt))

    ledger_lock = threading.Lock()

    def execute(item: tuple[PlanCell, ExperimentBinding, int]) -> dict[str, Any]:
        cell, binding, attempt = item
        profile = (
            dependencies.model_profile_factory(cell.cell_id)
            if dependencies.model_profile_factory is not None
            else dependencies.model_profile
        )
        model_provenance = profile.provenance_dict()
        system: DeepResearchSystem | None = None
        session = None
        service_identity: dict[str, Any] | None = None
        succeeded = False
        reason: str | None = None
        try:
            if dependencies.service_identity_factory is not None:
                raw_service_identity = dependencies.service_identity_factory(cell)
                if raw_service_identity is not None:
                    if not isinstance(raw_service_identity, Mapping):
                        raise ExecutionError(
                            "service identity factory returned an invalid identity"
                        )
                    service_identity = dict(raw_service_identity)
            search_handler = None
            if (
                cell.retrieval_regime in {"cutoff_controlled", "unrestricted_current_web"}
                and dependencies.search_session_factory is not None
            ):
                task_for_session = _task_input(task_rows[cell.task_id])
                session = dependencies.search_session_factory(
                    binding, task_for_session, cell.system_id
                )
                if session is not None:
                    _assert_session_principal(session, cell=cell)
                    search_handler = _session_search_handler(session)
            # External workers are explicitly one-task processes.  A fresh
            # adapter instance per cell prevents a closed stdio child from
            # being reused across trials or regimes.
            system = (
                dependencies.system_factory(
                    manifest.system(cell.system_id), cell.retrieval_regime, search_handler
                )
                if search_handler is not None
                else dependencies.system_factory(
                    manifest.system(cell.system_id), cell.retrieval_regime
                )
            )
            preflight = system.preflight(binding)
            if not preflight.ready:
                reason = "preflight:" + ",".join(preflight.reasons)
                error_payload: dict[str, Any] = {
                    "binding": binding.to_dict(),
                    "preflight": preflight.provenance,
                    "error": reason,
                    "model_provenance": model_provenance,
                    "ranking_eligible": False,
                }
                if service_identity is not None:
                    error_payload["service_identity"] = service_identity
                artifact = _write_artifact(
                    _artifact_path(root, cell, attempt),
                    error_payload,
                )
                with ledger_lock:
                    append_attempt(
                        ledger_path,
                        AttemptRecord(
                            cell.cell_id,
                            attempt,
                            "failed",
                            artifact,
                            reason,
                            worker_failure_subreason(reason),
                        ),
                    )
                return {
                    "cell_id": cell.cell_id,
                    "attempt": attempt,
                    "status": "failed",
                    "reason": reason,
                    "model_provenance": model_provenance,
                }
            task = _task_input(task_rows[cell.task_id])
            capability = (
                dependencies.capability_factory(binding, task)
                if dependencies.capability_factory is not None
                else _default_capability(binding, task)
            )
            if manifest.schema_version == "experiment/v4":
                assert_calc_policy(cell, capability)
            context = RunContext(
                run_id=manifest.experiment_id,
                framework=manifest.system(cell.system_id).framework_id,
                adapter_version=manifest.system(cell.system_id).adapter_revision,
                model=profile,
                trace_content_mode="metadata_only",
            ).for_task(task.task_id)
            result = system.run_task(task, context, capability, binding)
            succeeded = True
            close_status = "completed"
            artifact = _write_artifact(
                _artifact_path(root, cell, attempt),
                _result_payload(
                    result,
                    binding=binding,
                    preflight=preflight,
                    model_provenance=model_provenance,
                    search_session=_session_provenance(session, close_status=close_status)
                    if session is not None
                    else None,
                    service_identity=service_identity,
                ),
            )
            with ledger_lock:
                append_attempt(
                    ledger_path,
                    AttemptRecord(cell.cell_id, attempt, "succeeded", artifact),
                )
            return {
                "cell_id": cell.cell_id,
                "attempt": attempt,
                "status": "succeeded",
                "model_provenance": _merge_observed_model_provenance(
                    model_provenance,
                    asdict(result) if hasattr(result, "__dataclass_fields__") else result,
                ),
            }
        except Exception as exc:  # keep the append-only ledger auditable per cell.
            succeeded = False
            reason = f"{type(exc).__name__}:{exc}"
            diagnostic = getattr(exc, "diagnostic", None)
            error_payload = {
                "binding": binding.to_dict(),
                "error": reason,
                "model_provenance": model_provenance,
                "ranking_eligible": False,
            }
            if service_identity is not None:
                error_payload["service_identity"] = service_identity
            if isinstance(diagnostic, Mapping):
                # Boundary exceptions expose already-sanitized metadata only;
                # never serialize an exception's provider response directly.
                error_payload["error_diagnostic"] = dict(diagnostic)
            artifact = _write_artifact(
                _artifact_path(root, cell, attempt),
                error_payload,
            )
            with ledger_lock:
                append_attempt(
                    ledger_path,
                    AttemptRecord(
                        cell.cell_id,
                        attempt,
                        "failed",
                        artifact,
                        reason,
                        worker_failure_subreason(exc),
                    ),
                )
            return {
                "cell_id": cell.cell_id,
                "attempt": attempt,
                "status": "failed",
                "reason": reason,
                "model_provenance": model_provenance,
            }
        finally:
            close_status = "completed" if succeeded else "aborted"
            if session is not None and getattr(session, "close_result", "unset") is None:
                try:
                    session.close(status=close_status)
                except Exception:
                    pass
            if system is not None:
                close = getattr(system, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

    if work:
        with ThreadPoolExecutor(max_workers=request.max_workers) as pool:
            for outcome in pool.map(execute, work):
                attempts_written += 1
                outcome_provenance = outcome.get("model_provenance")
                if isinstance(outcome_provenance, Mapping):
                    summary_model_provenance = _merge_observed_model_provenance(
                        summary_model_provenance, outcome_provenance
                    )
                if outcome["status"] == "succeeded":
                    completed += 1
                    terminal_states[outcome["cell_id"]] = "succeeded"
                else:
                    failed += 1
                    terminal_states[outcome["cell_id"]] = "failed"
                    reasons.append(f"{outcome['cell_id']}:{outcome.get('reason', '')}")

    total = len(cells)
    if completed == total:
        status = "complete"
    elif pending or any(state == "pending" for state in terminal_states.values()):
        status = "pending"
    else:
        status = "exhausted"
    summary = ExecutionSummary(
        experiment_id=manifest.experiment_id,
        plan_id=plan.plan_id,
        selection=request.selection,
        status=status,
        total_cells=total,
        completed_cells=completed,
        pending_cells=total - completed - failed - exhausted,
        failed_cells=failed,
        exhausted_cells=exhausted,
        attempts_written=attempts_written,
        results_root=str(root),
        reasons=tuple(dict.fromkeys(reasons)),
    )
    summary_dict = summary.as_dict()
    summary_dict["terminal_cells"] = dict(sorted(terminal_states.items()))
    summary_dict["execution_contract"] = execution_contract
    summary_dict["model_provenance"] = summary_model_provenance
    if live_runtime is not None:
        summary_dict["live_runtime"] = live_runtime
    (root / "execution-summary.json").write_text(
        json.dumps(summary_dict, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        from benchmark_harness.experiment.projector import project_execution_artifacts

        project_execution_artifacts(
            root, manifest, plan, selection=request.selection, records=task_rows
        )
    except Exception as exc:
        raise ExecutionError(f"analysis projection failed closed: {exc}") from exc
    return summary


def run_experiment(
    request: ExperimentExecutionRequest,
    *,
    dependencies: ExecutionDependencies,
) -> ExecutionSummary:
    return _run(request, dependencies=dependencies, resume=False)


def resume_experiment(
    request: ExperimentExecutionRequest,
    *,
    dependencies: ExecutionDependencies,
) -> ExecutionSummary:
    return _run(request, dependencies=dependencies, resume=True)


def real_system_factory(
    system: SystemBindingSpec,
    retrieval_regime: str,
    search_handler: Callable[[str, str, int], Any] | None = None,
    search_client: Any | None = None,
    *,
    retrieval_policy: RetrievalPolicy | None = None,
    repo_root: str | Path | None = None,
) -> DeepResearchSystem:
    """Build a pinned adapter from explicit source environment variables.

    Missing source/environment input is reported as a dependency error rather
    than silently falling back to a fixture or mutable local checkout.  For
    ``cutoff_controlled`` cells the caller must supply a task-scoped search
    handler (ODR/GPT) or an admin-authenticated service client
    (FinanceHarness).
    """

    descriptor = system.descriptor(retrieval_regime)
    if retrieval_regime in {
        "cutoff_controlled",
        "unrestricted_current_web",
    } and system.system_id not in {
        "financeharness",
        "finance-harness",
    }:
        if search_handler is None:
            raise ExecutionDependencyError(
                f"{retrieval_regime} requires a task-scoped search handler for {system.system_id!r}"
            )
    resolved_repo_root = Path(repo_root or Path.cwd()).expanduser().resolve()
    if system.system_id in {"odr", "open_deep_research"}:
        from benchmark_harness.systems.adapters.odr import OdrAdapter

        source = os.environ.get("BENCHMARK_ODR_SOURCE")
        if not source:
            raise ExecutionDependencyError("BENCHMARK_ODR_SOURCE is required for real ODR smoke")
        return OdrAdapter(
            descriptor=descriptor,
            source_dir=source,
            repo_root=resolved_repo_root,
            retrieval_policy=retrieval_policy,
            search_handler=search_handler,
            search_token=_search_handler_token(system, retrieval_regime, search_handler),
        )
    if system.system_id in {"gpt-researcher", "gpt_researcher"}:
        from benchmark_harness.systems.adapters.gpt_researcher import GptResearcherAdapter

        source = os.environ.get("BENCHMARK_GPT_SOURCE")
        if not source:
            raise ExecutionDependencyError("BENCHMARK_GPT_SOURCE is required for real GPT smoke")
        return GptResearcherAdapter(
            descriptor=descriptor,
            source_dir=source,
            repo_root=resolved_repo_root,
            retrieval_policy=retrieval_policy,
            search_handler=search_handler,
            search_token=_search_handler_token(system, retrieval_regime, search_handler),
        )
    if system.system_id in {"financeharness", "finance-harness"}:
        from benchmark_harness.systems.adapters.financeharness import real_financeharness_system

        try:
            return real_financeharness_system(
                system,
                retrieval_regime,
                repo_root=resolved_repo_root,
                search_service=search_client,
            )
        except Exception as exc:
            raise ExecutionDependencyError(str(exc)) from exc
    raise ExecutionDependencyError(
        f"no real adapter factory is configured for {system.system_id!r}; "
        "provide an explicit FinanceHarness adapter dependency"
    )


def _search_handler_token(
    system: SystemBindingSpec, retrieval_regime: str, search_handler: Any
) -> str | None:
    if (
        retrieval_regime not in {"cutoff_controlled", "unrestricted_current_web"}
        or search_handler is None
    ):
        return None
    token = getattr(search_handler, "_task_token", None)
    if not isinstance(token, str) or not token.strip():
        raise ExecutionDependencyError("task-scoped search handler is missing its capability token")
    return token


__all__ = [
    "assert_calc_policy",
    "ExecutionDependencies",
    "ExecutionDependencyError",
    "ExecutionError",
    "ExecutionSummary",
    "ExperimentExecutionRequest",
    "real_system_factory",
    "resume_experiment",
    "run_experiment",
]
