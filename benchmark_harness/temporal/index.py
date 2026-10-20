"""Fail-closed loading for the external task temporal index."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from benchmark_harness.temporal.contracts import (
    TemporalApplicability,
    TemporalIndexRow,
    parse_temporal_date,
)


class TemporalIndexError(ValueError):
    """Raised when a temporal index is missing, malformed, or incomplete."""


@dataclass(frozen=True, slots=True)
class TemporalIndex:
    rows: dict[str, TemporalIndexRow]
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if not self.rows:
            raise TemporalIndexError("temporal index must contain at least one row.")

    def row_for(self, task_id: str) -> TemporalIndexRow:
        try:
            return self.rows[task_id]
        except KeyError as exc:
            raise TemporalIndexError(f"temporal index has no row for task {task_id!r}.") from exc


def load_temporal_index(
    path: str | Path,
    *,
    expected_task_ids: Iterable[str] | None = None,
) -> TemporalIndex:
    """Load one reviewed JSONL row per task and return its immutable file hash."""

    index_path = Path(path).expanduser().resolve()
    try:
        raw = index_path.read_bytes()
    except OSError as exc:
        raise TemporalIndexError(f"temporal index is unavailable: {index_path}") from exc
    rows: dict[str, TemporalIndexRow] = {}
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TemporalIndexError(f"invalid JSON at {index_path}:{line_number}.") from exc
        if not isinstance(value, dict):
            raise TemporalIndexError(f"temporal index row {line_number} must be an object.")
        try:
            task_id = value["task_id"]
            if task_id in rows:
                raise TemporalIndexError(f"temporal index contains duplicate task_id {task_id!r}.")
            applicability = TemporalApplicability(str(value["temporal_applicability"]).lower())
            row = TemporalIndexRow(
                task_id=task_id,
                temporal_applicability=applicability,
                forecast_origin=parse_temporal_date(
                    value.get("forecast_origin"),
                    field_name=f"{index_path}:{line_number}.forecast_origin",
                ),
                outcome_available_at=parse_temporal_date(
                    value.get("outcome_available_at"),
                    field_name=f"{index_path}:{line_number}.outcome_available_at",
                ),
                source_time_band=value["source_time_band"],
                evidence=tuple(value.get("evidence", [])),
                review_status=value.get("review_status", ""),
                quality_audit=value.get("quality_audit", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, TemporalIndexError):
                raise
            raise TemporalIndexError(
                f"invalid temporal index row at {index_path}:{line_number}: {exc}"
            ) from exc
        rows[row.task_id] = row

    if not rows:
        raise TemporalIndexError("temporal index must contain at least one non-empty row.")
    if expected_task_ids is not None:
        expected = {str(task_id) for task_id in expected_task_ids}
        actual = set(rows)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if unexpected:
                details.append("unexpected=" + ",".join(unexpected))
            raise TemporalIndexError("temporal index task coverage mismatch: " + "; ".join(details))
    return TemporalIndex(rows=rows, path=index_path, sha256=hashlib.sha256(raw).hexdigest())
