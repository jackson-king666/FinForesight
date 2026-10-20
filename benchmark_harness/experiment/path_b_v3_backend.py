"""Injected, resumable execution for the sealed Path B v3 plan set.

This module owns the execution-side boundary after ``path_b_v3_runner`` has
validated the immutable G3 bundle.  It deliberately knows nothing about the
dataset record, evaluator, or ranking table.  A caller supplies a cell
executor; this coordinator supplies only the score-blind ``TaskInput`` and a
cell-scoped output location, then persists an append-only attempt ledger.

The separation is important for the leakage probe: the cutoff and unrestricted
cells are dispatched to different physical namespaces before the injected
executor is called.  No executor may choose an output path or turn a cell into
a ranking input through this interface.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.contracts import RunContext, TaskInput
from benchmark_harness.experiment.path_b_v2 import sha256_json
from benchmark_harness.experiment.path_b_v3_runner import (
    PATH_B_V3_RESULTS_ROOTS,
    PathBV3ExecutionError,
    PathBV3ExecutionRequest,
    PathBV3Plans,
)
from benchmark_harness.systems.contracts import (
    BudgetSpec,
    ExperimentBinding,
    TaskCapability,
)

PATH_B_V3_BACKEND_SCHEMA_VERSION = "path-b-v3-execution-backend/v1"
PATH_B_V3_LEDGER_SCHEMA_VERSION = "path-b-v3-execution-ledger/v1"
PATH_B_V3_ARTIFACT_SCHEMA_VERSION = "path-b-v3-cell-artifact/v1"
PATH_B_V3_LEDGER_FILENAME = "execution-ledger.jsonl"
PATH_B_V3_SUMMARY_FILENAME = "execution-summary.json"
PATH_B_V3_CELL_DIRECTORY = "cells"

_FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "score",
        "scores",
        "gold",
        "ground_truth",
        "evaluator",
        "evaluator_output",
        "metric",
        "metrics",
        "api_key",
        "secret",
        "token",
        "credential",
        "credentials",
        "artifact_path",
        "output_path",
        "output_root",
        "results_root",
        "write_path",
    }
)
_REGIME_NAMESPACES = {
    "cutoff_controlled": "cutoff_controlled",
    "unrestricted_current_web": "unrestricted_current_web",
}


class PathBV3BackendError(PathBV3ExecutionError):
    """Raised when a cell executor crosses the v3 output or identity boundary."""


@dataclass(frozen=True, slots=True)
class PathBV3Cell:
    """One score-free execution identity materialized from a sealed plan cell."""

    experiment_kind: str
    experiment_id: str
    plan_sha256: str
    cell_id: str
    task_id: str
    system_id: str
    trial: int
    seed: int
    retrieval_regime: str
    ranking_eligible: bool
    budget: BudgetSpec
    budget_sha256: str
    execution_order: int
    output_namespace: str
    experiment_as_of: date | None = None

    def __post_init__(self) -> None:
        if self.ranking_eligible is not False:
            raise PathBV3BackendError("Path B v3 cells must remain non-ranking")
        if self.retrieval_regime not in {"cutoff_controlled", "unrestricted_current_web"}:
            raise PathBV3BackendError("unsupported Path B v3 retrieval regime")
        if self.output_namespace != _REGIME_NAMESPACES[self.retrieval_regime]:
            raise PathBV3BackendError("cell output namespace does not match retrieval regime")

    def public_dict(self) -> dict[str, Any]:
        return {
            "experiment_kind": self.experiment_kind,
            "experiment_id": self.experiment_id,
            "plan_sha256": self.plan_sha256,
            "cell_id": self.cell_id,
            "task_id": self.task_id,
            "system_id": self.system_id,
            "trial": self.trial,
            "seed": self.seed,
            "retrieval_regime": self.retrieval_regime,
            "ranking_eligible": False,
            "budget_sha256": self.budget_sha256,
            "execution_order": self.execution_order,
            "output_namespace": self.output_namespace,
            "experiment_as_of": (
                self.experiment_as_of.isoformat() if self.experiment_as_of is not None else None
            ),
        }


class PathBV3CellExecutor(Protocol):
    """Injected system runner for one planned cell.

    The callback returns an in-memory, JSON-compatible adapter result.  The
    backend, rather than the callback, writes the result artifact.  This keeps
    all persistent writes inside the exact cell namespace selected from the
    sealed plan.
    """

    def execute(
        self,
        cell: PathBV3Cell,
        task: TaskInput,
        *,
        output_root: Path,
        attempt: int,
    ) -> Mapping[str, Any] | None: ...


CheckpointCallback = Callable[[Mapping[str, Any]], None]
DryDispatchValidator = Callable[[PathBV3ExecutionRequest], None]
PathBV3SystemFactory = Callable[[PathBV3Cell], Any]
PathBV3ModelProfileFactory = Callable[[PathBV3Cell], ResolvedModelProfile]


class PathBV3SystemCellExecutor:
    """Adapt an injected system factory to the backend's cell protocol.

    The factory is intentionally caller-owned.  It must select the adapter and
    retrieval policy from the cell regime; this class never substitutes a
    cutoff system for an unrestricted cell.  Adapters return in-memory results,
    while the surrounding backend owns the only result write.
    """

    def __init__(
        self,
        system_factory: PathBV3SystemFactory,
        model_profile_factory: PathBV3ModelProfileFactory,
        *,
        capability_factory: (
            Callable[[PathBV3Cell, TaskInput, ExperimentBinding], TaskCapability] | None
        ) = None,
        trace_content_mode: str = "metadata_only",
    ) -> None:
        if not callable(system_factory) or not callable(model_profile_factory):
            raise TypeError("system and model profile factories must be callable")
        if not isinstance(trace_content_mode, str) or not trace_content_mode.strip():
            raise ValueError("trace_content_mode must be non-empty")
        self.system_factory = system_factory
        self.model_profile_factory = model_profile_factory
        self.capability_factory = capability_factory
        self.trace_content_mode = trace_content_mode

    def execute(
        self,
        cell: PathBV3Cell,
        task: TaskInput,
        *,
        output_root: Path,
        attempt: int,
    ) -> Mapping[str, Any] | None:
        del output_root
        if cell.experiment_as_of is None:
            raise PathBV3BackendError("sealed cell experiment clock is missing")
        if task.task_id != cell.task_id:
            raise PathBV3BackendError("system executor task identity mismatch")
        system = self.system_factory(cell)
        descriptor = getattr(system, "descriptor", None)
        run_task = getattr(system, "run_task", None)
        if descriptor is None or not callable(run_task):
            raise PathBV3BackendError("system factory returned an invalid adapter")
        if getattr(descriptor, "system_id", None) != cell.system_id:
            raise PathBV3BackendError("system descriptor identity does not match the sealed cell")
        if getattr(descriptor, "retrieval_regime", None) != cell.retrieval_regime:
            raise PathBV3BackendError("system retrieval regime does not match the sealed cell")
        if getattr(descriptor, "ranking_eligible", True) is not False:
            raise PathBV3BackendError("system descriptor is ranking eligible")
        policy = getattr(system, "retrieval_policy", None)
        if policy is not None:
            policy_regime = getattr(policy, "regime", None)
            policy_regime = getattr(policy_regime, "value", policy_regime)
            if policy_regime != cell.retrieval_regime:
                raise PathBV3BackendError("system policy does not match the sealed cell")
        profile = self.model_profile_factory(cell)
        if not isinstance(profile, ResolvedModelProfile):
            raise PathBV3BackendError("model profile factory returned an invalid profile")
        binding = ExperimentBinding(
            experiment_id=cell.experiment_id,
            plan_id=cell.experiment_id,
            plan_hash=cell.plan_sha256,
            cell_id=cell.cell_id,
            attempt_id=f"{cell.cell_id}:attempt-{attempt}",
            descriptor_hash=descriptor.descriptor_hash,
            task_id=cell.task_id,
            retrieval_regime=cell.retrieval_regime,
            trial=cell.trial,
            budget=cell.budget,
            experiment_as_of=cell.experiment_as_of,
        )
        capability = (
            self.capability_factory(cell, task, binding)
            if self.capability_factory is not None
            else _default_task_capability(cell)
        )
        if not isinstance(capability, TaskCapability):
            raise PathBV3BackendError("capability factory returned an invalid capability")
        context = RunContext(
            run_id=f"{cell.experiment_id}:{cell.cell_id}:attempt-{attempt}",
            framework=descriptor.framework_id,
            adapter_version=descriptor.adapter_revision,
            model=profile,
            trace_content_mode=self.trace_content_mode,
        )
        try:
            configure_task = getattr(system, "configure_task", None)
            if callable(configure_task):
                configure_task(task, binding)
            result = run_task(task, context, capability, binding)
        finally:
            close = getattr(system, "close", None)
            if callable(close):
                close()
        return _normalize_response(result)


@dataclass(frozen=True, slots=True)
class _LedgerState:
    successful_attempt: int | None
    next_attempt: int


class PathBV3CurrentWebBackend:
    """Coordinate injected current-web cells with strict resumability.

    ``batch_size`` bounds the amount of work between operator checkpoints.
    The callback receives only counts and public cell identities, never task
    prompts, model responses, credentials, or evaluator-owned fields.
    """

    def __init__(
        self,
        executor: PathBV3CellExecutor,
        *,
        batch_size: int = 8,
        checkpoint_callback: CheckpointCallback | None = None,
        dry_dispatch_validator: DryDispatchValidator | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not callable(getattr(executor, "execute", None)):
            raise TypeError("executor must expose execute(cell, task, output_root, attempt)")
        self.executor = executor
        self.batch_size = batch_size
        self.checkpoint_callback = checkpoint_callback
        self.dry_dispatch_validator = dry_dispatch_validator
        self.repo_root = Path(repo_root).expanduser().resolve() if repo_root else None

    def dry_dispatch(self, request: PathBV3ExecutionRequest) -> Mapping[str, Any]:
        """Validate a live dispatch without loading tasks or writing artifacts.

        This is intentionally a backend operation rather than a runner-only
        flag.  It materializes every sealed cell identity and checks the exact
        physical namespaces, but it does not touch the ledger, create result
        directories, invoke the executor, or persist a score-bearing value.
        """

        self._validate_request_roots(request)
        if self.dry_dispatch_validator is not None:
            self.dry_dispatch_validator(request)
        cells = _materialize_cells(request.plans)
        by_kind: dict[str, int] = {}
        for cell in cells:
            by_kind[cell.experiment_kind] = by_kind.get(cell.experiment_kind, 0) + 1
        return {
            "schema_version": PATH_B_V3_BACKEND_SCHEMA_VERSION,
            "status": "dry_dispatch",
            "execution_state": "dry_dispatch",
            "dataset_revision": request.plans.index.get("dataset_revision"),
            "dataset_tree_sha": request.plans.index.get("dataset_tree_sha"),
            "experiments": {
                kind: {"total_cells": count} for kind, count in sorted(by_kind.items())
            },
            "b_calibration": {
                "status": "no_available_case_sample",
                "available_case_count": 0,
            },
            "total_cells": len(cells),
            "ranking_eligible": False,
            "claim_tier": "exploratory",
            "results_roots": {
                kind: str(root) for kind, root in request.plans.results_roots.items()
            },
        }

    def __call__(self, request: PathBV3ExecutionRequest) -> Mapping[str, Any]:
        return self.execute(request)

    def execute(self, request: PathBV3ExecutionRequest) -> Mapping[str, Any]:
        self._validate_request(request)
        cells = _materialize_cells(request.plans)
        task_inputs = request.task_inputs
        states = self._load_states(request.plans, cells)
        pending = [cell for cell in cells if states[_state_key(cell)].successful_attempt is None]
        total_batches = (len(pending) + self.batch_size - 1) // self.batch_size
        attempts_written = 0
        completed_before = len(cells) - len(pending)

        for batch_index, start in enumerate(range(0, len(pending), self.batch_size), start=1):
            batch = pending[start : start + self.batch_size]
            self._checkpoint(
                {
                    "schema_version": PATH_B_V3_BACKEND_SCHEMA_VERSION,
                    "event": "before_batch",
                    "batch_index": batch_index,
                    "total_batches": total_batches,
                    "completed_cells": completed_before + attempts_written,
                    "pending_cells": len(pending) - start,
                    "cell_count": len(batch),
                    "ranking_eligible": False,
                    "namespaces": sorted({cell.output_namespace for cell in batch}),
                }
            )
            for cell in batch:
                task = task_inputs.get(cell.task_id)
                if not isinstance(task, TaskInput):
                    raise PathBV3BackendError(f"missing score-free TaskInput for {cell.task_id}")
                state = states[_state_key(cell)]
                attempt = state.next_attempt
                root = self._cell_root(request.plans, cell)
                try:
                    response = self.executor.execute(
                        cell,
                        task,
                        output_root=root,
                        attempt=attempt,
                    )
                    normalized = _normalize_response(response)
                    _assert_score_free_response(normalized)
                    artifact = {
                        "schema_version": PATH_B_V3_ARTIFACT_SCHEMA_VERSION,
                        "backend_schema_version": PATH_B_V3_BACKEND_SCHEMA_VERSION,
                        "cell": cell.public_dict(),
                        "attempt": attempt,
                        "ranking_eligible": False,
                        "response": normalized,
                    }
                    artifact_path = _artifact_path(root, cell, attempt)
                    artifact_sha = _write_json_new(artifact_path, artifact)
                    _append_ledger(
                        root / PATH_B_V3_LEDGER_FILENAME,
                        {
                            "schema_version": PATH_B_V3_LEDGER_SCHEMA_VERSION,
                            "cell_key": _state_key(cell),
                            "cell": cell.public_dict(),
                            "attempt": attempt,
                            "status": "succeeded",
                            "artifact": artifact_path.relative_to(root).as_posix(),
                            "artifact_sha256": artifact_sha,
                            "ranking_eligible": False,
                        },
                    )
                    attempts_written += 1
                    states[_state_key(cell)] = _LedgerState(
                        successful_attempt=attempt,
                        next_attempt=attempt + 1,
                    )
                except PathBV3BackendError:
                    raise
                except Exception as exc:
                    # Only the exception class is persisted.  Provider errors
                    # can contain credentials, prompts, or response content.
                    _append_ledger(
                        root / PATH_B_V3_LEDGER_FILENAME,
                        {
                            "schema_version": PATH_B_V3_LEDGER_SCHEMA_VERSION,
                            "cell_key": _state_key(cell),
                            "cell": cell.public_dict(),
                            "attempt": attempt,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "ranking_eligible": False,
                        },
                    )
                    raise PathBV3BackendError(
                        f"Path B v3 cell failed: {cell.cell_id} ({type(exc).__name__})"
                    ) from exc

        summary = self._summary(request.plans, cells, states, attempts_written)
        for root in self._roots_for_cells(request.plans, cells):
            selected = [cell for cell in cells if self._cell_root(request.plans, cell) == root]
            _write_json_new_or_same(
                root / PATH_B_V3_SUMMARY_FILENAME,
                {
                    "schema_version": PATH_B_V3_BACKEND_SCHEMA_VERSION,
                    "experiment_kind": selected[0].experiment_kind,
                    "plan_sha256": selected[0].plan_sha256,
                    "cell_count": len(selected),
                    "completed_cells": sum(
                        states[_state_key(cell)].successful_attempt is not None for cell in selected
                    ),
                    "ranking_eligible": False,
                    "claim_tier": "exploratory",
                },
            )
        for kind, root in self._summary_roots(request.plans).items():
            if kind == "b_calibration":
                _write_json_new_or_same(
                    root / PATH_B_V3_SUMMARY_FILENAME,
                    {
                        "schema_version": PATH_B_V3_BACKEND_SCHEMA_VERSION,
                        "experiment_kind": kind,
                        "status": "no_available_case_sample",
                        "available_case_count": 0,
                        "ranking_eligible": False,
                    },
                )
        return summary

    def _validate_request(self, request: PathBV3ExecutionRequest) -> None:
        self._validate_request_roots(request)
        if set(request.task_inputs) != set(request.plans.eligible_task_ids):
            raise PathBV3BackendError("backend task inputs do not exactly match the frozen cohort")
        for task_id, task in request.task_inputs.items():
            if task_id != task.task_id or not isinstance(task, TaskInput):
                raise PathBV3BackendError(f"task input identity mismatch for {task_id}")

    def _validate_request_roots(self, request: PathBV3ExecutionRequest) -> None:
        if not isinstance(request, PathBV3ExecutionRequest):
            raise PathBV3BackendError("backend request type is invalid")
        if not request.credentials.ready:
            raise PathBV3BackendError("credential gate must pass before backend dispatch")
        expected = request.plans.results_roots
        if set(request.results_roots) != set(expected):
            raise PathBV3BackendError("request results roots do not match the sealed plan")
        for kind, expected_root in expected.items():
            supplied = Path(request.results_roots[kind]).expanduser().resolve()
            if supplied != expected_root:
                raise PathBV3BackendError(f"{kind} results root does not match the sealed plan")
            _assert_safe_results_root(
                request.plans.root.parent.parent / PATH_B_V3_RESULTS_ROOTS[kind],
                self.repo_root or request.plans.root.parent.parent.resolve(),
            )

    def _cell_root(self, plans: PathBV3Plans, cell: PathBV3Cell) -> Path:
        root = plans.results_roots["cutoff_unrestricted"] / cell.output_namespace
        if cell.experiment_kind == "t1_t2":
            root = plans.results_roots["t1_t2"]
        elif cell.experiment_kind != "cutoff_unrestricted":
            raise PathBV3BackendError("unsupported cell experiment kind")
        _assert_safe_results_root(root, self.repo_root or plans.root.parent.parent.resolve())
        return root

    def _load_states(
        self,
        plans: PathBV3Plans,
        cells: tuple[PathBV3Cell, ...],
    ) -> dict[str, _LedgerState]:
        result = {
            _state_key(cell): _LedgerState(successful_attempt=None, next_attempt=1)
            for cell in cells
        }
        for root in self._roots_for_cells(plans, cells):
            ledger_path = root / PATH_B_V3_LEDGER_FILENAME
            if not ledger_path.exists():
                continue
            if ledger_path.is_symlink() or not ledger_path.is_file():
                raise PathBV3BackendError("execution ledger is not a regular file")
            try:
                lines = ledger_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError) as exc:
                raise PathBV3BackendError("execution ledger cannot be read") from exc
            seen_attempts: set[tuple[str, int]] = set()
            for line in lines:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PathBV3BackendError("execution ledger contains invalid JSON") from exc
                if not isinstance(row, Mapping):
                    raise PathBV3BackendError("execution ledger row is not an object")
                if row.get("schema_version") != PATH_B_V3_LEDGER_SCHEMA_VERSION:
                    raise PathBV3BackendError("execution ledger schema drifted")
                key = row.get("cell_key")
                attempt = row.get("attempt")
                if not isinstance(key, str) or key not in result:
                    raise PathBV3BackendError("execution ledger contains an unknown cell")
                if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
                    raise PathBV3BackendError("execution ledger attempt is invalid")
                identity = (key, attempt)
                if identity in seen_attempts:
                    raise PathBV3BackendError("execution ledger contains a duplicate attempt")
                seen_attempts.add(identity)
                cell = next(cell for cell in cells if _state_key(cell) == key)
                if (
                    row.get("cell") != cell.public_dict()
                    or row.get("ranking_eligible") is not False
                ):
                    raise PathBV3BackendError("execution ledger cell identity drifted")
                status = row.get("status")
                state = result[key]
                if status == "succeeded":
                    artifact_name = row.get("artifact")
                    artifact_sha = row.get("artifact_sha256")
                    if not isinstance(artifact_name, str) or not isinstance(artifact_sha, str):
                        raise PathBV3BackendError("successful ledger row is incomplete")
                    artifact_path = _resolve_child(root, artifact_name)
                    if not artifact_path.is_file() or _sha256_file(artifact_path) != artifact_sha:
                        raise PathBV3BackendError("successful cell artifact is missing or modified")
                    artifact = _read_json(artifact_path)
                    if (
                        artifact.get("cell") != cell.public_dict()
                        or artifact.get("attempt") != attempt
                    ):
                        raise PathBV3BackendError("cell artifact identity drifted")
                    if artifact.get("ranking_eligible") is not False:
                        raise PathBV3BackendError("cell artifact is ranking eligible")
                    if state.successful_attempt is not None:
                        raise PathBV3BackendError("cell has multiple successful attempts")
                    result[key] = _LedgerState(
                        successful_attempt=attempt,
                        next_attempt=max(state.next_attempt, attempt + 1),
                    )
                elif status == "failed":
                    if state.successful_attempt is not None:
                        raise PathBV3BackendError("failed attempt follows a successful attempt")
                    if not isinstance(row.get("error_type"), str) or not row["error_type"]:
                        raise PathBV3BackendError("failed ledger row is incomplete")
                    if state.successful_attempt is None:
                        result[key] = _LedgerState(
                            successful_attempt=None,
                            next_attempt=max(state.next_attempt, attempt + 1),
                        )
                else:
                    raise PathBV3BackendError("execution ledger status is invalid")
        return result

    def _roots_for_cells(
        self, plans: PathBV3Plans, cells: tuple[PathBV3Cell, ...]
    ) -> tuple[Path, ...]:
        roots = {self._cell_root(plans, cell) for cell in cells}
        return tuple(sorted(roots, key=str))

    def _summary_roots(self, plans: PathBV3Plans) -> dict[str, Path]:
        roots = plans.results_roots
        return {
            "t1_t2": roots["t1_t2"],
            "b_calibration": roots["b_calibration"],
            "cutoff_unrestricted": roots["cutoff_unrestricted"],
        }

    def _summary(
        self,
        plans: PathBV3Plans,
        cells: tuple[PathBV3Cell, ...],
        states: Mapping[str, _LedgerState],
        attempts_written: int,
    ) -> dict[str, Any]:
        by_kind: dict[str, dict[str, int]] = {}
        for kind in ("t1_t2", "cutoff_unrestricted"):
            selected = [cell for cell in cells if cell.experiment_kind == kind]
            by_kind[kind] = {
                "total_cells": len(selected),
                "completed_cells": sum(
                    states[_state_key(cell)].successful_attempt is not None for cell in selected
                ),
            }
        return {
            "schema_version": PATH_B_V3_BACKEND_SCHEMA_VERSION,
            "status": "complete",
            "execution_state": "complete",
            "dataset_revision": plans.index.get("dataset_revision"),
            "dataset_tree_sha": plans.index.get("dataset_tree_sha"),
            "experiments": by_kind,
            "b_calibration": {
                "status": "no_available_case_sample",
                "available_case_count": 0,
            },
            "total_cells": len(cells),
            "completed_cells": sum(
                states[_state_key(cell)].successful_attempt is not None for cell in cells
            ),
            "attempts_written": attempts_written,
            "ranking_eligible": False,
            "claim_tier": "exploratory",
            "results_roots": {kind: str(root) for kind, root in plans.results_roots.items()},
        }

    def _checkpoint(self, payload: Mapping[str, Any]) -> None:
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(dict(payload))


def _materialize_cells(plans: PathBV3Plans) -> tuple[PathBV3Cell, ...]:
    result: list[PathBV3Cell] = []
    for kind in ("t1_t2", "cutoff_unrestricted"):
        manifest = plans.manifests[kind]
        plan = plans.plans[kind]
        experiment_id = _required_text(manifest.get("experiment_id"), "experiment_id")
        plan_sha = _required_text(plan.get("plan_sha256"), "plan_sha256")
        budget = _budget_from_mapping(manifest.get("budget"))
        budget_sha256 = _required_text(
            manifest.get("budget_sha256"), f"{kind} manifest budget_sha256"
        )
        experiment_as_of = _parse_date(manifest.get("experiment_as_of"), f"{kind}.experiment_as_of")
        raw_cells = plan.get("cells")
        if not isinstance(raw_cells, list):
            raise PathBV3BackendError(f"{kind} plan cells are not a list")
        for index, raw in enumerate(raw_cells):
            if not isinstance(raw, Mapping):
                raise PathBV3BackendError(f"{kind} cell {index} is not an object")
            task_id = _required_text(raw.get("task_id"), f"{kind} cell task_id")
            if task_id not in plans.eligible_task_ids:
                raise PathBV3BackendError("cell task is outside the frozen cohort")
            system_id = _required_text(raw.get("system_id"), f"{kind} cell system_id")
            regime = _required_text(raw.get("retrieval_regime"), f"{kind} cell regime")
            if raw.get("budget_sha256") != budget_sha256:
                raise PathBV3BackendError("cell budget binding drifted")
            if kind == "t1_t2" and regime != "cutoff_controlled":
                raise PathBV3BackendError("T1/T2 cell is not cutoff controlled")
            if kind == "cutoff_unrestricted" and regime not in _REGIME_NAMESPACES:
                raise PathBV3BackendError("leakage cell regime is invalid")
            if kind == "cutoff_unrestricted":
                expected_filter = regime == "cutoff_controlled"
                if raw.get("cutoff_filter_enabled") is not expected_filter:
                    raise PathBV3BackendError("leakage cutoff filter binding drifted")
                expected_policy = "cutoff_controlled/v1" if expected_filter else "leakage_probe/v1"
                if raw.get("policy_identity") != expected_policy:
                    raise PathBV3BackendError("leakage policy identity drifted")
            if raw.get("ranking_eligible") is not False:
                raise PathBV3BackendError("sealed cell is ranking eligible")
            try:
                trial = _positive_int(raw.get("trial"), "trial")
                seed = _nonnegative_int(raw.get("seed"), "seed")
            except ValueError as exc:
                raise PathBV3BackendError(str(exc)) from exc
            if kind == "t1_t2":
                cell_id = _required_text(raw.get("cell_id"), "T1/T2 cell_id")
                execution_order = _nonnegative_int(raw.get("execution_order"), "execution_order")
            else:
                cell_id = _leakage_cell_id(experiment_id, raw)
                execution_order = index
            result.append(
                PathBV3Cell(
                    experiment_kind=kind,
                    experiment_id=experiment_id,
                    plan_sha256=plan_sha,
                    cell_id=cell_id,
                    task_id=task_id,
                    system_id=system_id,
                    trial=trial,
                    seed=seed,
                    retrieval_regime=regime,
                    ranking_eligible=False,
                    budget=budget,
                    budget_sha256=budget_sha256,
                    execution_order=execution_order,
                    output_namespace=_REGIME_NAMESPACES[regime],
                    experiment_as_of=experiment_as_of,
                )
            )
    kind_order = {"t1_t2": 0, "cutoff_unrestricted": 1}
    result.sort(
        key=lambda cell: (kind_order[cell.experiment_kind], cell.execution_order, cell.cell_id)
    )
    keys = [_state_key(cell) for cell in result]
    if len(keys) != len(set(keys)):
        raise PathBV3BackendError("sealed plan contains duplicate cell identities")
    return tuple(result)


def _budget_from_mapping(value: Any) -> BudgetSpec:
    if not isinstance(value, Mapping):
        raise PathBV3BackendError("sealed manifest budget is missing")
    try:
        return BudgetSpec(
            max_model_calls=_nonnegative_int(value.get("model_calls"), "model_calls"),
            max_search_calls=_nonnegative_int(value.get("searches"), "searches"),
            max_visit_calls=_nonnegative_int(value.get("visits"), "visits"),
            max_other_tool_calls=_nonnegative_int(value.get("other_tools"), "other_tools"),
            max_input_tokens=_nonnegative_int(value.get("input_tokens"), "input_tokens"),
            max_output_tokens=_nonnegative_int(value.get("output_tokens"), "output_tokens"),
            wall_clock_timeout_seconds=float(value["wall_clock_seconds"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PathBV3BackendError("sealed manifest budget is invalid") from exc


def _leakage_cell_id(experiment_id: str, raw: Mapping[str, Any]) -> str:
    identity = {
        "experiment_id": experiment_id,
        "task_id": raw.get("task_id"),
        "system_id": raw.get("system_id"),
        "trial": raw.get("trial"),
        "seed": raw.get("seed"),
        "retrieval_regime": raw.get("retrieval_regime"),
        "budget_sha256": raw.get("budget_sha256"),
    }
    return "cell-" + sha256_json(identity)[:24]


def _state_key(cell: PathBV3Cell) -> str:
    return sha256_json(
        {
            "experiment_kind": cell.experiment_kind,
            "experiment_id": cell.experiment_id,
            "plan_sha256": cell.plan_sha256,
            "cell_id": cell.cell_id,
        }
    )


def _artifact_path(root: Path, cell: PathBV3Cell, attempt: int) -> Path:
    return root / PATH_B_V3_CELL_DIRECTORY / _state_key(cell) / f"attempt-{attempt}.json"


def _normalize_response(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise PathBV3BackendError("cell executor must return an object or None")
    normalized = dict(value)
    try:
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)
    except (TypeError, ValueError) as exc:
        raise PathBV3BackendError("cell executor returned non-JSON data") from exc
    return normalized


def _default_task_capability(cell: PathBV3Cell) -> TaskCapability:
    return TaskCapability(
        service_url=None,
        service_token=None,
        model_broker_capability="host-model-broker/v1",
        budget=cell.budget,
        allow_calculation=True,
        allow_citations=True,
        require_structured_output=True,
        run_id=cell.experiment_id,
        trial_id=f"{cell.cell_id}:trial-{cell.trial}",
        trace_mode="metadata_only",
    )


def _assert_score_free_response(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() in _FORBIDDEN_OUTPUT_KEYS:
                raise PathBV3BackendError("cell response contains evaluator-owned data")
            if key == "ranking_eligible" and child is True:
                raise PathBV3BackendError("cell response attempted to become ranking eligible")
            _assert_score_free_response(child)
    elif isinstance(value, list | tuple):
        for child in value:
            _assert_score_free_response(child)


def _assert_safe_results_root(root: Path, repo_root: Path | None) -> None:
    lexical = Path(os.path.abspath(os.fspath(root.expanduser())))
    resolved = lexical.resolve()
    if repo_root is not None:
        expected_base = (repo_root / "results").resolve()
        try:
            resolved.relative_to(expected_base)
        except ValueError as exc:
            raise PathBV3BackendError(
                "results root is outside the repository results directory"
            ) from exc
    parts = lexical.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current /= part
        if current.is_symlink():
            raise PathBV3BackendError("results root contains a symlink")
    if resolved.exists() and not resolved.is_dir():
        raise PathBV3BackendError("results root is not a directory")


def _resolve_child(root: Path, name: str) -> Path:
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PathBV3BackendError("ledger artifact path escapes its results root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PathBV3BackendError("ledger artifact path escapes its results root") from exc
    return resolved


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PathBV3BackendError("cell artifact cannot be read") from exc
    if not isinstance(value, Mapping):
        raise PathBV3BackendError("cell artifact is not an object")
    return value


def _write_json_new(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PathBV3BackendError("refusing to overwrite an existing cell artifact")
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    try:
        fd, temporary = tempfile.mkstemp(prefix=".v3-artifact-", dir=str(path.parent))
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, ValueError) as exc:
        try:
            os.unlink(temporary)
        except (OSError, UnboundLocalError):
            pass
        raise PathBV3BackendError("cell artifact could not be written") from exc
    return hashlib.sha256(encoded).hexdigest()


def _write_json_new_or_same(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise PathBV3BackendError("execution summary is a symlink")
    if path.exists():
        existing = _read_json(path)
        if existing != value:
            raise PathBV3BackendError("sealed execution summary was modified")
        return
    _write_json_new(path, value)


def _append_ledger(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise PathBV3BackendError("execution ledger is not a regular file")
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise PathBV3BackendError("execution ledger could not be written") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PathBV3BackendError("cell artifact hash could not be read") from exc
    return digest.hexdigest()


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PathBV3BackendError(f"{name} must be a non-empty string")
    return value.strip()


def _parse_date(value: Any, name: str) -> date | None:
    if value is None:
        return None
    text = _required_text(value, name)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise PathBV3BackendError(f"{name} must be an ISO date") from exc
    if parsed.isoformat() != text:
        raise PathBV3BackendError(f"{name} must be an ISO date")
    return parsed


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    parsed = _nonnegative_int(value, name)
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


__all__ = [
    "PATH_B_V3_ARTIFACT_SCHEMA_VERSION",
    "PATH_B_V3_BACKEND_SCHEMA_VERSION",
    "PATH_B_V3_CELL_DIRECTORY",
    "PATH_B_V3_LEDGER_FILENAME",
    "PATH_B_V3_LEDGER_SCHEMA_VERSION",
    "PATH_B_V3_SUMMARY_FILENAME",
    "PathBV3BackendError",
    "PathBV3Cell",
    "PathBV3CellExecutor",
    "PathBV3CurrentWebBackend",
    "DryDispatchValidator",
    "PathBV3ModelProfileFactory",
    "PathBV3SystemFactory",
    "PathBV3SystemCellExecutor",
]
