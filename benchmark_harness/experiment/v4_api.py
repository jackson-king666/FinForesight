"""Public experiment/v4 orchestration and artifact API."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_harness.analysis.full_release import (
    SYSTEMS,
    control_experiments,
    e1_delta_web,
    e2_decision_tests,
    e3_leaderboard,
    e4_calibration,
    experiment_status,
)
from benchmark_harness.analysis.lineage import (
    ArtifactManifest,
    build_artifact_manifest,
    validate_artifact_manifest,
    write_artifact_manifest,
)
from benchmark_harness.experiment.manifest import ExperimentManifest, ManifestError, load_manifest
from benchmark_harness.experiment.planner import ExperimentPlan, load_plan
from benchmark_harness.experiment.projector import project_execution_artifacts


class V4ApiError(ValueError):
    """Raised when a full-release artifact cannot be assembled safely."""


@dataclass(frozen=True, slots=True)
class FullReleaseArtifacts:
    output_dir: str
    status: dict[str, Any]
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "status": self.status,
            "files": list(self.files),
        }


def _resolve_manifest(value: ExperimentManifest | str | Path) -> ExperimentManifest:
    return value if isinstance(value, ExperimentManifest) else load_manifest(value)


def _resolve_plan(
    value: ExperimentPlan | str | Path, manifest: ExperimentManifest
) -> ExperimentPlan:
    if isinstance(value, ExperimentPlan):
        plan = value
    else:
        plan = load_plan(value, expected_manifest_hash=manifest.manifest_hash)
    if plan.manifest_hash != manifest.manifest_hash:
        raise ManifestError("manifest hash does not match v4 plan")
    return plan


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise V4ApiError(f"analysis rows are unavailable: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise V4ApiError(f"invalid analysis row at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise V4ApiError(f"analysis row at {path}:{line_number} must be an object")
        rows.append(value)
    if not rows:
        raise V4ApiError("analysis rows are empty")
    return rows


def _t2_t3_scores(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        if row.get("retrieval_regime") != "cutoff_controlled":
            continue
        if row.get("analysis_eligible") is False:
            continue
        utility = row.get("formal_utility")
        if not isinstance(utility, int | float) or isinstance(utility, bool):
            metric_values = row.get("metric_values")
            utility = metric_values.get("utility") if isinstance(metric_values, dict) else None
        if not isinstance(utility, int | float) or isinstance(utility, bool):
            continue
        band = str(row.get("temporal_band", "")).upper()
        if band not in {"T2", "T3"}:
            continue
        key = (str(row.get("system_id")), str(row.get("task_id")), band)
        grouped.setdefault(key, []).append(float(utility))
    t2: dict[str, dict[str, float]] = {}
    t3: dict[str, dict[str, float]] = {}
    for (system, task, band), values in grouped.items():
        target = t2 if band == "T2" else t3
        target.setdefault(system, {})[task] = sum(values) / len(values)
    return t2, t3


def _artifact_is_complete(artifact: Any) -> bool:
    return isinstance(artifact, Mapping) and artifact.get("status") == "ok"


def _e1_is_complete(artifact: Mapping[str, Any]) -> bool:
    contrasts = artifact.get("contrasts")
    if not isinstance(contrasts, list):
        return False
    primary = {
        str(row.get("system_id")): row
        for row in contrasts
        if isinstance(row, Mapping) and row.get("contrast") == "unrestricted_minus_cutoff"
    }
    if set(primary) != set(SYSTEMS):
        return False
    return all(
        isinstance(row.get("matched_n"), int)
        and row["matched_n"] > 0
        and isinstance(row.get("mean_difference"), int | float)
        and not isinstance(row["mean_difference"], bool)
        and math.isfinite(float(row["mean_difference"]))
        for row in primary.values()
    )


def _e2_is_complete(artifact: Mapping[str, Any]) -> bool:
    if not _artifact_is_complete(artifact):
        return False
    t2_scores = artifact.get("t2_scores")
    t3_scores = artifact.get("t3_scores")
    return (
        isinstance(t2_scores, Mapping)
        and isinstance(t3_scores, Mapping)
        and set(SYSTEMS).issubset({str(system) for system in t2_scores})
        and set(SYSTEMS).issubset({str(system) for system in t3_scores})
    )


def _e3_is_complete(artifact: Mapping[str, Any]) -> bool:
    if not _artifact_is_complete(artifact):
        return False
    t2 = artifact.get("t2")
    return isinstance(t2, Mapping) and _artifact_is_complete(t2)


def _e4_is_complete(artifact: Mapping[str, Any]) -> bool:
    if not _artifact_is_complete(artifact):
        return False
    overall = artifact.get("overall")
    return (
        isinstance(overall, Mapping)
        and isinstance(overall.get("row_count"), int)
        and overall["row_count"] > 0
        and isinstance(overall.get("valid_output_count"), int)
        and overall["valid_output_count"] > 0
    )


def _derive_finding(e1: Mapping[str, Any]) -> str:
    """Derive the E1 finding from every declared system's primary contrast."""

    contrasts = e1.get("contrasts")
    if not isinstance(contrasts, list):
        return "inconclusive"
    primary = {
        str(row.get("system_id")): row
        for row in contrasts
        if isinstance(row, Mapping) and row.get("contrast") == "unrestricted_minus_cutoff"
    }
    differences = [
        float(row["mean_difference"])
        for system in SYSTEMS
        if isinstance((row := primary.get(system)), Mapping)
        and isinstance(row.get("mean_difference"), int | float)
        and not isinstance(row["mean_difference"], bool)
        and math.isfinite(float(row["mean_difference"]))
    ]
    if len(differences) != len(SYSTEMS):
        return "inconclusive"
    if all(difference > 0 for difference in differences):
        return "supported"
    if all(difference <= 0 for difference in differences):
        return "not_supported"
    return "inconclusive"


def _derive_analysis_state(
    e1: Mapping[str, Any],
    e2: Mapping[str, Any],
    e3: Mapping[str, Any],
    e4: Mapping[str, Any],
    controls: Mapping[str, Any],
    *,
    t3_pending: bool,
) -> tuple[str, str]:
    """Return independent analysis/finding states from the E1--E4 artifacts."""

    historical_ready = (
        _artifact_is_complete(e1)
        and _e1_is_complete(e1)
        and _e3_is_complete(e3)
        and _e4_is_complete(e4)
        and controls.get("status") == "ok"
    )
    if not historical_ready:
        return "data_incomplete", "inconclusive"
    if t3_pending or not _e2_is_complete(e2):
        return "ready", "inconclusive"
    return "complete", _derive_finding(e1)


def analyze_full_release(
    manifest: ExperimentManifest | str | Path,
    plan: ExperimentPlan | str | Path,
    root: str | Path,
    output: str | Path,
    *,
    records: Mapping[str, Mapping[str, Any]] | None = None,
) -> FullReleaseArtifacts:
    """Project v4 rows and write independent E1/E2/E3/E4 artifacts.

    When task records are supplied, projection is deliberately repeated from
    the append-only attempts ledger even if a previous ``analysis_rows.jsonl``
    exists. This makes post-run frozen-metric repairs deterministic without
    rerunning model or search work.
    """

    resolved_manifest = _resolve_manifest(manifest)
    if resolved_manifest.schema_version != "experiment/v4":
        raise V4ApiError("analyze_full_release requires an experiment/v4 manifest")
    resolved_plan = _resolve_plan(plan, resolved_manifest)
    requested_root = Path(root).expanduser().resolve()
    candidates = (
        requested_root,
        requested_root / resolved_manifest.experiment_id / resolved_plan.plan_id,
    )
    root_path = next(
        (candidate for candidate in candidates if (candidate / "analysis_rows.jsonl").is_file()),
        next(
            (candidate for candidate in candidates if (candidate / "attempts.jsonl").is_file()),
            requested_root,
        ),
    )
    rows_path = root_path / "analysis_rows.jsonl"
    if (records is not None or not rows_path.is_file()) and (
        root_path / "attempts.jsonl"
    ).is_file():
        project_execution_artifacts(
            root_path,
            resolved_manifest,
            resolved_plan,
            records=records,
        )
    rows = _read_rows(rows_path)
    observed_systems = {str(row.get("system_id")) for row in rows}
    unexpected = sorted(observed_systems - set(SYSTEMS))
    if unexpected:
        raise V4ApiError("analysis rows contain undeclared systems: " + ", ".join(unexpected))
    e1 = e1_delta_web(
        rows,
        seed=resolved_manifest.seed,
        resamples=resolved_manifest.bootstrap_resamples,
        system_ids=SYSTEMS,
    )
    t2, t3 = _t2_t3_scores(rows)
    e2 = e2_decision_tests(
        t2,
        t3,
        seed=resolved_manifest.seed,
        margin=float(resolved_manifest.v4_metadata.get("sample_plan", {}).get("e2_margin", 0.10))
        if isinstance(resolved_manifest.v4_metadata.get("sample_plan"), dict)
        else 0.10,
        alpha=resolved_manifest.alpha,
        resamples=resolved_manifest.bootstrap_resamples,
        system_ids=SYSTEMS,
    )
    e3 = e3_leaderboard(
        rows,
        seed=resolved_manifest.seed,
        resamples=resolved_manifest.bootstrap_resamples,
        alpha=resolved_manifest.alpha,
        system_ids=SYSTEMS,
    )
    e4 = e4_calibration(
        rows,
        seed=resolved_manifest.seed,
        resamples=resolved_manifest.bootstrap_resamples,
        system_ids=SYSTEMS,
    )
    controls = control_experiments(rows, system_ids=SYSTEMS)
    t3_pending = any(
        str(row.get("temporal_band", "")).upper() == "T3"
        and not row.get("analysis_eligible", False)
        for row in rows
    )
    analysis_state, finding = _derive_analysis_state(
        e1,
        e2,
        e3,
        e4,
        controls,
        t3_pending=t3_pending,
    )
    execution = "awaiting_settlement" if t3_pending else "terminal"
    ranking: dict[str, bool] = {}
    for system in SYSTEMS:
        cutoff_rows = [
            row
            for row in rows
            if row.get("system_id") == system and row.get("retrieval_regime") == "cutoff_controlled"
        ]
        ranking[system] = (
            e3.get("t2", {}).get("status") == "ok"
            and bool(cutoff_rows)
            and all(bool(row.get("ranking_eligible")) for row in cutoff_rows)
        )
    status = experiment_status(
        design_status="preregistered",
        execution_status=execution,
        analysis_status=analysis_state,
        finding=finding,
        ranking_eligibility=ranking,
    )
    exclusions = [
        {
            "system_id": row.get("system_id"),
            "task_id": row.get("task_id"),
            "trial": row.get("trial"),
            "regime": row.get("retrieval_regime"),
            "reason": row.get("failure_reason") or row.get("formal_utility_reason"),
        }
        for row in rows
        if row.get("analysis_eligible") is False or row.get("formal_utility") is None
    ]
    output_path = Path(output).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "e4": e4,
        "controls": controls,
        "status": status,
        "exclusions": exclusions,
    }
    for name, value in artifacts.items():
        (output_path / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    lineage = {
        "schema_version": "full-release-lineage/v1",
        "experiment_id": resolved_manifest.experiment_id,
        "manifest_sha256": resolved_manifest.manifest_hash,
        "plan_sha256": resolved_plan.plan_hash,
        "dataset_revision": resolved_manifest.dataset_revision,
        "formal_utility_version": "formal-utility/v1",
        "ranking_eligibility": ranking,
    }
    manifest_artifact = build_artifact_manifest(output_path, lineage=lineage)
    write_artifact_manifest(output_path, manifest_artifact)
    return FullReleaseArtifacts(
        output_dir=str(output_path),
        status=status,
        files=tuple(sorted(path.name for path in output_path.iterdir() if path.is_file())),
    )


def audit_full_release(root: str | Path) -> ArtifactManifest:
    """Validate a v4 analysis output without rerunning any analysis."""

    return validate_artifact_manifest(root)


__all__ = [
    "FullReleaseArtifacts",
    "V4ApiError",
    "analyze_full_release",
    "audit_full_release",
]
