"""Deterministic blocked-randomized experiment planning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_harness.experiment.manifest import ExperimentManifest, ManifestError
from benchmark_harness.systems.contracts import BudgetSpec, ExperimentBinding


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PlanCell:
    cell_id: str
    system_id: str
    task_id: str
    retrieval_regime: str
    trial: int
    execution_order: int
    attempt: int = 1
    split: str | None = None
    smoke: bool = False
    # The default keeps v1/v2 plan JSON and hashes byte-identical.  v4 E
    # controls explicitly materialize a second calc_off cell.
    calc_policy: str = "calc_on"

    def __post_init__(self) -> None:
        if self.calc_policy not in {"calc_on", "calc_off"}:
            raise ValueError("calc_policy must be calc_on or calc_off")

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "cell_id": self.cell_id,
            "system_id": self.system_id,
            "task_id": self.task_id,
            "retrieval_regime": self.retrieval_regime,
            "trial": self.trial,
            "execution_order": self.execution_order,
            "attempt": self.attempt,
        }
        if self.split is not None:
            value["split"] = self.split
            value["smoke"] = self.smoke
        if self.calc_policy != "calc_on":
            value["calc_policy"] = self.calc_policy
        return value


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    schema_version: str
    plan_id: str
    experiment_id: str
    manifest_hash: str
    seed: int
    cells: tuple[PlanCell, ...]
    matched_cell_ids: tuple[str, ...]
    plan_hash: str
    selection: str = "all"
    split_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {
            "experiment-plan/v1",
            "experiment-plan/v2",
            "experiment-plan/v4",
        }:
            raise ValueError("unsupported experiment plan schema version.")
        if not self.cells:
            raise ValueError("experiment plan must contain at least one cell.")
        if not self.plan_hash:
            raise ValueError("experiment plan must contain a plan hash.")
        if self.selection not in {"smoke", "dev", "test", "all"}:
            raise ValueError("experiment plan selection must be smoke, dev, test, or all.")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "experiment_id": self.experiment_id,
            "manifest_hash": self.manifest_hash,
            "seed": self.seed,
            "cells": [cell.to_dict() for cell in self.cells],
            "matched_cell_ids": list(self.matched_cell_ids),
        }
        if self.schema_version in {"experiment-plan/v2", "experiment-plan/v4"}:
            value["selection"] = self.selection
        if self.schema_version == "experiment-plan/v2":
            value["split_sha256"] = self.split_sha256
        if include_hash:
            value["plan_hash"] = self.plan_hash
        return value

    def bindings(
        self,
        manifest: ExperimentManifest,
        budgets: Mapping[str, BudgetSpec],
    ) -> tuple[ExperimentBinding, ...]:
        if manifest.manifest_hash != self.manifest_hash:
            raise ManifestError("manifest hash does not match experiment plan.")
        result: list[ExperimentBinding] = []
        for cell in self.cells:
            budget = budgets.get(cell.system_id)
            if budget is None:
                raise ManifestError(f"no budget declared for system {cell.system_id!r}.")
            descriptor = manifest.system(cell.system_id).descriptor(cell.retrieval_regime)
            result.append(
                ExperimentBinding(
                    experiment_id=self.experiment_id,
                    plan_id=self.plan_id,
                    plan_hash=self.plan_hash,
                    cell_id=cell.cell_id,
                    attempt_id=f"{cell.cell_id}:attempt-{cell.attempt}",
                    descriptor_hash=descriptor.descriptor_hash,
                    task_id=cell.task_id,
                    retrieval_regime=cell.retrieval_regime,
                    trial=cell.trial,
                    budget=budget,
                    experiment_as_of=manifest.experiment_as_of,
                    calc_policy=cell.calc_policy,
                )
            )
        return tuple(result)


def _split_rows(path: str | Path) -> tuple[dict[str, Any], ...]:
    split_path = Path(path).expanduser().resolve()
    try:
        raw = split_path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"split artifact is unavailable: {split_path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"invalid split JSON at {split_path}:{line_number}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("task_id"), str):
            raise ManifestError(f"split row {line_number} must contain a task_id.")
        if value.get("split") not in {"dev", "test"}:
            raise ManifestError(f"split row {line_number} has an invalid split.")
        if not isinstance(value.get("smoke", False), bool):
            raise ManifestError(f"split row {line_number}.smoke must be boolean.")
        rows.append(value)
    if not rows:
        raise ManifestError("split artifact must contain at least one row.")
    task_ids = [row["task_id"] for row in rows]
    if task_ids != sorted(task_ids) or len(task_ids) != len(set(task_ids)):
        raise ManifestError("split artifact must be unique and sorted by task_id.")
    return tuple(rows)


def _resolve_v4_section_path(manifest: ExperimentManifest, section: Mapping[str, Any]) -> Path:
    raw_path = section.get("path", section.get("ledger_path"))
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ManifestError("experiment/v4 cohort section requires a path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and manifest.source_path is not None:
        path = manifest.source_path.parent / path
    return path.resolve()


def _cohort_rows(manifest: ExperimentManifest) -> tuple[dict[str, Any], ...]:
    section = manifest.v4_metadata.get("cohort")
    if isinstance(section, Mapping):
        path = _resolve_v4_section_path(manifest, section)
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ManifestError(f"experiment/v4 cohort ledger is unavailable: {path}") from exc
        rows: list[dict[str, Any]] = []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            values = parsed
        else:
            values = [json.loads(line) for line in raw.splitlines() if line.strip()]
        for index, row in enumerate(values):
            if not isinstance(row, Mapping):
                raise ManifestError(f"experiment/v4 cohort row {index} must be an object")
            rows.append(dict(row))
    elif isinstance(section, list):
        rows = [dict(row) for row in section if isinstance(row, Mapping)]
    else:
        raise ManifestError("experiment/v4 requires a cohort ledger section")
    if not rows:
        raise ManifestError("experiment/v4 cohort ledger must not be empty")
    ids = [row.get("task_id") for row in rows]
    if any(not isinstance(task_id, str) or not task_id.strip() for task_id in ids):
        raise ManifestError("experiment/v4 cohort rows require non-empty task_id")
    if len(ids) != len(set(ids)):
        raise ManifestError("experiment/v4 cohort ledger contains duplicate task_id values")
    for index, row in enumerate(rows):
        category = row.get("category")
        if not isinstance(category, str) or not category.strip():
            raise ManifestError(f"experiment/v4 cohort row {index} requires category")
        if str(category).upper() not in {"A1", "A2", "B", "C", "D", "E"}:
            raise ManifestError(f"experiment/v4 cohort row {index} has unknown category")
    return tuple(sorted(rows, key=lambda row: str(row["task_id"])))


def build_plan(
    manifest: ExperimentManifest,
    task_ids: Iterable[str] | None = None,
    *,
    split_file: str | Path | None = None,
    selection: str = "all",
) -> ExperimentPlan:
    """Expand a manifest without consulting scores, timing, or failure rates."""

    if selection not in {"smoke", "dev", "test", "all"}:
        raise ManifestError("selection must be smoke, dev, test, or all.")
    v4_cohort: tuple[dict[str, Any], ...] = ()
    v4_by_task: dict[str, dict[str, Any]] = {}
    if manifest.schema_version == "experiment/v4":
        if split_file is not None:
            raise ManifestError("experiment/v4 uses the cohort ledger, not a split artifact")
        v4_cohort = _cohort_rows(manifest)
        v4_by_task = {str(row["task_id"]): row for row in v4_cohort}
    split_hash: str | None = None
    split_by_task: dict[str, dict[str, Any]] = {}
    if split_file is None and manifest.schema_version == "experiment/v2" and manifest.split_path:
        split_file = manifest.split_path
        if manifest.source_path is not None and not Path(split_file).is_absolute():
            split_file = manifest.source_path.parent / split_file
    if split_file is not None:
        split_path = Path(split_file).expanduser().resolve()
        try:
            split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ManifestError(f"split artifact is unavailable: {split_path}") from exc
        if manifest.split_sha256 and split_hash != manifest.split_sha256:
            raise ManifestError("split artifact hash does not match manifest.")
        split_by_task = {row["task_id"]: row for row in _split_rows(split_path)}

    raw_tasks = list(
        task_ids or (v4_by_task if manifest.schema_version == "experiment/v4" else split_by_task)
    )
    if any(not isinstance(task_id, str) or not task_id.strip() for task_id in raw_tasks):
        raise ManifestError("task_ids must be non-empty strings.")
    raw_tasks = [task_id.strip() for task_id in raw_tasks]
    if len(raw_tasks) != len(set(raw_tasks)):
        raise ManifestError("task_ids must be unique.")
    tasks = sorted(raw_tasks)
    if not tasks:
        raise ManifestError("at least one task_id is required to build a plan.")
    if selection != "all" and not split_by_task and manifest.schema_version != "experiment/v4":
        raise ManifestError("split_file is required for smoke/dev/test plan selection.")
    if split_by_task:
        missing = sorted(task_id for task_id in tasks if task_id not in split_by_task)
        if missing:
            raise ManifestError("split artifact has no row for task(s): " + ",".join(missing))
        if selection == "smoke":
            tasks = [task_id for task_id in tasks if split_by_task[task_id].get("smoke")]
        elif selection in {"dev", "test"}:
            tasks = [
                task_id for task_id in tasks if split_by_task[task_id].get("split") == selection
            ]
        if not tasks:
            raise ManifestError(f"split artifact contains no tasks for selection {selection!r}.")
    if manifest.schema_version == "experiment/v4":
        missing = sorted(task_id for task_id in tasks if task_id not in v4_by_task)
        if missing:
            raise ManifestError("cohort ledger has no row for task(s): " + ",".join(missing))
        if selection in {"dev", "test"}:
            tasks = [
                task_id
                for task_id in tasks
                if str(v4_by_task[task_id].get("split", "")) == selection
            ]
            if not tasks:
                raise ManifestError(f"cohort ledger contains no tasks for selection {selection!r}.")
        elif selection == "smoke":
            tasks = [task_id for task_id in tasks if bool(v4_by_task[task_id].get("smoke", False))]
            if not tasks:
                raise ManifestError("cohort ledger contains no smoke tasks for selection 'smoke'.")
    base_cells: list[PlanCell] = []
    population = manifest.v2_metadata.get("population") or {}
    applicability = population.get("applicability") if isinstance(population, Mapping) else None

    def regimes_for(task_id: str) -> tuple[str, ...]:
        if manifest.schema_version == "experiment/v4":
            category = str(v4_by_task[task_id].get("category", "")).upper()
            if category in {"A1", "A2", "B", "C"}:
                return tuple(manifest.retrieval_regimes)
            if category in {"D", "E"}:
                return ("closed_book",)
            raise ManifestError(f"unknown cohort category for task {task_id!r}")
        if not isinstance(applicability, Mapping):
            return manifest.retrieval_regimes
        category = str((split_by_task.get(task_id) or {}).get("category") or "")
        if not category:
            prefix = task_id.split("-", 1)[0].upper()
            category = (
                "D" if prefix.startswith("D") else "E" if prefix.startswith("E") else "FORECAST"
            )
        allowed = (
            applicability["controls"]
            if category.upper() in {"D", "E"}
            else applicability["forecast"]
        )
        selected = tuple(regime for regime in manifest.retrieval_regimes if regime in allowed)
        if not selected:
            raise ManifestError(f"no permitted retrieval regime for task {task_id!r}.")
        return selected

    def calc_policies_for(task_id: str) -> tuple[str, ...]:
        if (
            manifest.schema_version == "experiment/v4"
            and str(v4_by_task[task_id].get("category", "")).upper() == "E"
        ):
            return ("calc_on", "calc_off")
        return ("calc_on",)

    for trial in range(1, manifest.trial_count + 1):
        for task_id in tasks:
            for regime in regimes_for(task_id):
                for calc_policy in calc_policies_for(task_id):

                    def order_key(system: Any) -> str:
                        value = {
                            "seed": manifest.seed,
                            "task_id": task_id,
                            "trial": trial,
                            "retrieval_regime": regime,
                            "system_id": system.system_id,
                        }
                        if manifest.schema_version == "experiment/v4":
                            value["calc_policy"] = calc_policy
                        return _hash(value)

                    ordered_systems = sorted(manifest.systems, key=order_key)
                    for system in ordered_systems:
                        cell_key = {
                            "experiment_id": manifest.experiment_id,
                            "system_id": system.system_id,
                            "task_id": task_id,
                            "retrieval_regime": regime,
                            "trial": trial,
                        }
                        if manifest.schema_version == "experiment/v4":
                            cell_key["calc_policy"] = calc_policy
                        base_cells.append(
                            PlanCell(
                                cell_id="cell-" + _hash(cell_key)[:20],
                                system_id=system.system_id,
                                task_id=task_id,
                                retrieval_regime=regime,
                                trial=trial,
                                execution_order=0,
                                split=(
                                    split_by_task.get(task_id) or v4_by_task.get(task_id) or {}
                                ).get("split"),
                                smoke=bool(
                                    (
                                        split_by_task.get(task_id) or v4_by_task.get(task_id) or {}
                                    ).get("smoke", False)
                                ),
                                calc_policy=calc_policy,
                            )
                        )

    cells = tuple(
        PlanCell(
            cell_id=cell.cell_id,
            system_id=cell.system_id,
            task_id=cell.task_id,
            retrieval_regime=cell.retrieval_regime,
            trial=cell.trial,
            execution_order=index,
            attempt=cell.attempt,
            split=cell.split,
            smoke=cell.smoke,
            calc_policy=cell.calc_policy,
        )
        for index, cell in enumerate(base_cells)
    )
    expected_per_block = len(manifest.systems)
    blocks: dict[tuple[str, int, str, str], list[PlanCell]] = {}
    for cell in cells:
        blocks.setdefault(
            (cell.task_id, cell.trial, cell.retrieval_regime, cell.calc_policy), []
        ).append(cell)
    matched_cell_ids = tuple(
        cell.cell_id
        for cell in cells
        if len(blocks[(cell.task_id, cell.trial, cell.retrieval_regime, cell.calc_policy)])
        == expected_per_block
    )
    schema_version = (
        "experiment-plan/v4"
        if manifest.schema_version == "experiment/v4"
        else "experiment-plan/v2"
        if manifest.schema_version == "experiment/v2"
        else "experiment-plan/v1"
    )
    provisional = {
        "schema_version": schema_version,
        "experiment_id": manifest.experiment_id,
        "manifest_hash": manifest.manifest_hash,
        "seed": manifest.seed,
        "cells": [cell.to_dict() for cell in cells],
        "matched_cell_ids": list(matched_cell_ids),
    }
    if schema_version in {"experiment-plan/v2", "experiment-plan/v4"}:
        provisional["selection"] = selection
    if schema_version == "experiment-plan/v2":
        provisional["split_sha256"] = split_hash or manifest.split_sha256
    plan_hash = _hash(provisional)
    return ExperimentPlan(
        schema_version=schema_version,
        plan_id=f"{manifest.experiment_id}-{plan_hash[:12]}",
        experiment_id=manifest.experiment_id,
        manifest_hash=manifest.manifest_hash,
        seed=manifest.seed,
        cells=cells,
        matched_cell_ids=matched_cell_ids,
        plan_hash=plan_hash,
        selection=selection,
        split_sha256=split_hash or manifest.split_sha256,
    )


def load_plan(path: str | Path, *, expected_manifest_hash: str | None = None) -> ExperimentPlan:
    """Load and verify a plan file, including its self-declared hash."""

    plan_path = Path(path).expanduser().resolve()
    try:
        value = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"experiment plan is unavailable or invalid: {plan_path}") from exc
    if not isinstance(value, Mapping):
        raise ManifestError("experiment plan root must be an object.")
    schema_version = value.get("schema_version")
    if schema_version not in {"experiment-plan/v1", "experiment-plan/v2", "experiment-plan/v4"}:
        raise ManifestError("unsupported experiment plan schema version.")
    if expected_manifest_hash is not None and value.get("manifest_hash") != expected_manifest_hash:
        raise ManifestError("manifest hash does not match experiment plan.")
    try:
        cells = tuple(PlanCell(**dict(cell)) for cell in value["cells"])
        plan = ExperimentPlan(
            schema_version=schema_version,
            plan_id=str(value["plan_id"]),
            experiment_id=str(value["experiment_id"]),
            manifest_hash=str(value["manifest_hash"]),
            seed=int(value["seed"]),
            cells=cells,
            matched_cell_ids=tuple(str(item) for item in value.get("matched_cell_ids", [])),
            plan_hash=str(value["plan_hash"]),
            selection=str(value.get("selection", "all")),
            split_sha256=value.get("split_sha256"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError("experiment plan has an invalid shape.") from exc
    provisional = plan.to_dict(include_hash=False)
    provisional.pop("plan_id", None)
    if _hash(provisional) != plan.plan_hash:
        raise ManifestError("experiment plan hash does not match its contents.")
    return plan


def write_plan(path: str | Path, plan: ExperimentPlan) -> Path:
    """Write one stable, newline-terminated plan artifact."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
