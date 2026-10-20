"""Read-only ingestion of normalized raw result rows."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmark_harness.analysis.contracts import AnalysisError, AnalysisRow, coerce_analysis_rows
from benchmark_harness.analysis.lineage import ArtifactManifest, validate_artifact_manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AnalysisError(f"analysis input is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"invalid analysis JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise AnalysisError(f"analysis row at {path}:{line_number} must be an object.")
        rows.append(value)
    return rows


def ingest_rows(
    root: str | Path,
    *,
    expected_lineage: Mapping[str, Any] | None = None,
    filename: str = "analysis_rows.jsonl",
) -> tuple[tuple[AnalysisRow, ...], ArtifactManifest]:
    """Validate the artifact manifest before reading the analysis-row fixture."""

    root_path = Path(root).expanduser().resolve()
    manifest = validate_artifact_manifest(root_path, expected_lineage=expected_lineage)
    rows = coerce_analysis_rows(_read_jsonl(root_path / filename))
    if not rows:
        raise AnalysisError("analysis input contains no rows.")
    return rows, manifest
