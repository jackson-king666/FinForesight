"""Append-only attempt records and score-independent resume semantics."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_harness.experiment.planner import ExperimentPlan
from benchmark_harness.systems.contracts import BudgetSpec, ExperimentBinding


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    cell_id: str
    attempt: int
    status: str
    artifact_manifest_sha256: str | None = None
    reason: str | None = None
    sub_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.cell_id or not isinstance(self.cell_id, str):
            raise ValueError("cell_id must be a non-empty string.")
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt < 1:
            raise ValueError("attempt must be a positive integer.")
        if not isinstance(self.status, str) or not self.status.strip():
            raise ValueError("status must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "attempt": self.attempt,
            "status": self.status,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "reason": self.reason,
            "sub_reason": self.sub_reason,
        }


def read_attempts(path: str | Path) -> tuple[AttemptRecord, ...]:
    attempt_path = Path(path)
    if not attempt_path.exists():
        return ()
    rows: list[AttemptRecord] = []
    seen: set[tuple[str, int]] = set()
    try:
        lines = attempt_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read attempt ledger {attempt_path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict) and "failure_subreason" in value:
                # One pre-contract smoke writer used this spelling on otherwise
                # AttemptRecord-shaped rows. Accept it as a read-only alias so
                # those rows remain readable and new canonical rows append
                # without rewriting history. This does not make the external
                # hua174-attempt/v1 partial ledger resumable: its heterogeneous
                # schema is outside this reader and is intentionally excluded.
                if "sub_reason" in value:
                    raise ValueError("attempt record has duplicate subreason fields")
                value = {
                    **value,
                    "sub_reason": value["failure_subreason"],
                }
                del value["failure_subreason"]
            record = AttemptRecord(**value)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid attempt record at {attempt_path}:{line_number}") from exc
        identity = (record.cell_id, record.attempt)
        if identity in seen:
            raise ValueError(f"attempt ledger contains duplicate {identity!r}.")
        seen.add(identity)
        rows.append(record)
    return tuple(rows)


def append_attempt(path: str | Path, record: AttemptRecord) -> None:
    """Append one attempt only; an existing cell/attempt is never overwritten."""

    attempt_path = Path(path)
    existing = read_attempts(attempt_path)
    if any((row.cell_id, row.attempt) == (record.cell_id, record.attempt) for row in existing):
        raise ValueError(f"attempt {record.cell_id}:{record.attempt} already exists.")
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
    with attempt_path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()


def next_attempt_number(records: Iterable[AttemptRecord], cell_id: str) -> int:
    attempts = [record.attempt for record in records if record.cell_id == cell_id]
    return max(attempts, default=0) + 1


def resume_bindings(
    plan: ExperimentPlan,
    manifest,
    budgets: Mapping[str, BudgetSpec],
    records: Iterable[AttemptRecord],
) -> tuple[ExperimentBinding, ...]:
    """Return all cells with a new per-cell attempt number, independent of score."""

    known = tuple(records)
    bindings: list[ExperimentBinding] = []
    for cell in plan.cells:
        budget = budgets.get(cell.system_id)
        if budget is None:
            raise ValueError(f"no budget declared for system {cell.system_id!r}.")
        descriptor = manifest.system(cell.system_id).descriptor(cell.retrieval_regime)
        attempt = next_attempt_number(known, cell.cell_id)
        bindings.append(
            ExperimentBinding(
                experiment_id=plan.experiment_id,
                plan_id=plan.plan_id,
                plan_hash=plan.plan_hash,
                cell_id=cell.cell_id,
                attempt_id=f"{cell.cell_id}:attempt-{attempt}",
                descriptor_hash=descriptor.descriptor_hash,
                task_id=cell.task_id,
                retrieval_regime=cell.retrieval_regime,
                trial=cell.trial,
                budget=budget,
                experiment_as_of=manifest.experiment_as_of,
            )
        )
    return tuple(bindings)
