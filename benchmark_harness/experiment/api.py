"""Public v2 experiment API used by the CLI and offline callers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from benchmark_harness.analysis.ingest import ingest_rows
from benchmark_harness.analysis.lineage import (
    ArtifactManifest,
    build_artifact_manifest,
    validate_artifact_manifest,
    write_artifact_manifest,
)
from benchmark_harness.analysis.metrics.catalog import load_metric_catalog
from benchmark_harness.analysis.metrics.exploratory import c2_status
from benchmark_harness.analysis.pipeline import analyze_rows
from benchmark_harness.experiment.execution_contract import (
    ExecutionContractError,
    validate_execution_contract,
)
from benchmark_harness.experiment.executor import (
    ExperimentExecutionRequest,
    resume_experiment,
    run_experiment,
)
from benchmark_harness.experiment.live_model import declared_model_identity
from benchmark_harness.experiment.manifest import (
    ExperimentManifest,
    ManifestError,
    load_manifest,
    validate_manifest_artifacts,
)
from benchmark_harness.experiment.planner import ExperimentPlan, load_plan
from benchmark_harness.experiment.projector import project_execution_artifacts


@dataclass(frozen=True, slots=True)
class AuditReport:
    experiment_id: str
    plan_id: str
    status: str
    checked_attempts: int
    valid_attempts: int
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "plan_id": self.plan_id,
            "status": self.status,
            "checked_attempts": self.checked_attempts,
            "valid_attempts": self.valid_attempts,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    manifest: ExperimentManifest | str | Path
    plan: ExperimentPlan | str | Path
    input_root: str | Path
    output_dir: str | Path
    selection: Literal["smoke", "dev", "test", "all"] = "all"
    analysis_as_of: date | str | None = None
    metric_name: str = "exploratory_task_utility_v1"

    def __post_init__(self) -> None:
        if self.selection not in {"smoke", "dev", "test", "all"}:
            raise ValueError("selection must be smoke, dev, test, or all")


@dataclass(frozen=True, slots=True)
class AnalysisArtifacts:
    output_dir: str
    identity: dict[str, Any]
    c2_status: str
    files: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "identity": self.identity,
            "c2_status": self.c2_status,
            "files": list(self.files),
        }


def _resolve_manifest(value: ExperimentManifest | str | Path) -> ExperimentManifest:
    return value if isinstance(value, ExperimentManifest) else load_manifest(value)


def _resolve_plan(
    value: ExperimentPlan | str | Path, manifest: ExperimentManifest
) -> ExperimentPlan:
    return (
        value
        if isinstance(value, ExperimentPlan)
        else load_plan(value, expected_manifest_hash=manifest.manifest_hash)
    )


def _resolve_analysis_root(
    input_root: str | Path, manifest: ExperimentManifest, plan: ExperimentPlan
) -> Path:
    root = Path(input_root).expanduser().resolve()
    nested = root / manifest.experiment_id / plan.plan_id
    if (nested / "attempts.jsonl").is_file():
        return nested
    if (root / "attempts.jsonl").is_file() or (root / "analysis_rows.jsonl").is_file():
        return root
    raise ManifestError(
        "analysis input must be an executor root or contain analysis_rows.jsonl: " + str(root)
    )


def audit_experiment(request: ExperimentExecutionRequest) -> AuditReport:
    """Audit append-only executor artifacts without reading model content."""

    manifest = _resolve_manifest(request.manifest)
    plan = _resolve_plan(request.plan, manifest)
    validate_manifest_artifacts(manifest)
    root = Path(request.results_root).expanduser().resolve() / manifest.experiment_id / plan.plan_id
    ledger = root / "attempts.jsonl"
    if not ledger.is_file():
        if root.is_dir() and any(root.iterdir()):
            try:
                validate_execution_contract(root, manifest, plan)
            except ExecutionContractError as exc:
                return AuditReport(
                    manifest.experiment_id,
                    plan.plan_id,
                    "invalid",
                    0,
                    0,
                    (f"execution_contract:{exc}",),
                )
        return AuditReport(
            manifest.experiment_id, plan.plan_id, "pending", 0, 0, ("attempt_ledger_missing",)
        )
    try:
        execution_contract = validate_execution_contract(root, manifest, plan)
    except ExecutionContractError as exc:
        return AuditReport(
            manifest.experiment_id,
            plan.plan_id,
            "invalid",
            0,
            0,
            (f"execution_contract:{exc}",),
        )
    try:
        rows = [
            json.loads(line)
            for line in ledger.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError("attempt ledger is invalid") from exc
    reasons: list[str] = []
    try:
        validate_artifact_manifest(
            root,
            expected_lineage={
                "manifest_sha256": manifest.manifest_hash,
                "plan_sha256": plan.plan_hash,
                "execution_contract": execution_contract,
                "ranking_eligible": False,
            },
        )
    except Exception as exc:
        reasons.append(f"artifact_manifest:{exc}")
    valid = 0
    cell_map = {cell.cell_id: cell for cell in plan.cells}
    for row in rows:
        cell_id = row.get("cell_id")
        attempt = row.get("attempt")
        artifact_hash = row.get("artifact_manifest_sha256")
        path = root / "cells" / str(cell_id) / f"attempt-{attempt}.json"
        if cell_id not in cell_map or not path.is_file():
            reasons.append(f"orphan_artifact:{cell_id}:{attempt}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != artifact_hash:
            reasons.append(f"artifact_hash_mismatch:{cell_id}:{attempt}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            reasons.append(f"artifact_invalid_json:{cell_id}:{attempt}")
            continue
        binding = payload.get("binding", {})
        if binding.get("plan_hash") != plan.plan_hash or binding.get("cell_id") != cell_id:
            reasons.append(f"binding_mismatch:{cell_id}:{attempt}")
            continue
        valid += 1
    status = "valid" if not reasons else "invalid"
    return AuditReport(
        manifest.experiment_id, plan.plan_id, status, len(rows), valid, tuple(reasons)
    )


def analyze_experiment(request: AnalysisRequest) -> AnalysisArtifacts:
    """Run the existing offline analysis pipeline with manifest-owned settings."""

    manifest = _resolve_manifest(request.manifest)
    plan = _resolve_plan(request.plan, manifest)
    validate_manifest_artifacts(manifest)
    analysis_root = _resolve_analysis_root(request.input_root, manifest, plan)
    execution_contract = validate_execution_contract(analysis_root, manifest, plan)
    model_identity = declared_model_identity(manifest)
    declared_metric = manifest.v2_metadata.get("analysis", {}).get("primary_metric")
    if manifest.schema_version == "experiment/v2" and request.metric_name != declared_metric:
        raise ManifestError(
            f"metric override is not permitted; manifest primary is {declared_metric!r}"
        )
    if request.analysis_as_of is not None:
        requested = date.fromisoformat(str(request.analysis_as_of)[:10])
        if manifest.analysis_as_of is not None and requested != manifest.analysis_as_of:
            raise ManifestError("analysis_as_of is manifest-owned and cannot be overridden")
    catalog_path = Path(manifest.metric_catalog_path)
    if not catalog_path.is_absolute() and manifest.source_path is not None:
        catalog_path = manifest.source_path.parent / catalog_path
    catalog = load_metric_catalog(catalog_path)
    if (analysis_root / "attempts.jsonl").is_file():
        records: dict[str, Any] | None = None
        try:
            from benchmark_harness.dataset_sources.huggingface import HuggingFaceDatasetSource

            records = {
                str(row["task_id"]): row
                for row in HuggingFaceDatasetSource(offline=True).load_records()
            }
        except Exception as exc:
            # Fail closed instead of silently degrading a scored projection:
            # without dataset records the frozen evaluator cannot materialize
            # metrics, and re-projecting would overwrite valid rows with
            # ``dataset_record_unavailable_for_frozen_scoring`` exclusions.
            if not (analysis_root / "analysis_rows.jsonl").is_file():
                raise ManifestError(
                    "dataset records are unavailable for frozen scoring and no "
                    "existing projection is present; analysis fails closed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        if records is not None:
            project_execution_artifacts(
                analysis_root,
                manifest,
                plan,
                selection=request.selection,
                records=records,
            )
    rows, artifact_manifest = ingest_rows(
        analysis_root,
        expected_lineage={
            "manifest_sha256": manifest.manifest_hash,
            "plan_sha256": plan.plan_hash,
            "execution_contract": execution_contract,
            "ranking_eligible": False,
        },
    )
    lineage = artifact_manifest.lineage
    result = analyze_rows(
        rows,
        catalog,
        metric_name=request.metric_name,
        seed=manifest.seed,
        bootstrap_resamples=manifest.bootstrap_resamples,
        identity={
            "experiment_id": manifest.experiment_id,
            "manifest_sha256": manifest.manifest_hash,
            "plan_sha256": plan.plan_hash,
            "dataset_revision": manifest.dataset_revision,
            "backbone_id": model_identity["backbone_id"],
            "snapshot_model_name": model_identity["snapshot_model_name"],
            "wire_model_name": model_identity["wire_model_name"],
            "wire_name_binding": model_identity["wire_name_binding"],
            "execution_contract": lineage.get("execution_contract", execution_contract),
            "evaluator_manifest_sha256": lineage.get("evaluator_manifest_sha256"),
            "temporal_index_sha256": lineage.get("temporal_index_sha256"),
            "cutoff_registry_sha256": lineage.get("cutoff_registry_sha256"),
            "metric_catalog_sha256": catalog.hash,
            "analysis_as_of": (manifest.analysis_as_of or manifest.experiment_as_of).isoformat(),
            "seed": manifest.seed,
            "raw_artifact_lineage": lineage,
        },
    )
    gate = c2_status(
        (system.system_id for system in manifest.systems),
        has_t2=any(row.temporal_band == "T2" and row.analysis_eligible for row in rows),
        resolved_t3=any(row.temporal_band == "T3" and row.analysis_eligible for row in rows),
    )
    c2 = dict(result.get("c2", {}))
    c2.update(
        {
            "status": gate["status"],
            "ranking_eligible": False,
            "reasons": list(gate["reasons"]),
            "correlations": None,
            "spearman": None,
            "pearson": None,
            "kendall": None,
        }
    )
    result["c2"] = c2
    output = Path(request.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("c1", "c2", "c3", "exclusions"):
        (output / f"{name}.json").write_text(
            json.dumps(result.get(name, []), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    identity = dict(result.get("identity", {}))
    identity["manifest_sha256"] = manifest.manifest_hash
    identity["plan_sha256"] = plan.plan_hash
    identity["c2_status"] = c2["status"]
    identity["c2_reasons"] = c2["reasons"]
    (output / "lineage.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_manifest = build_artifact_manifest(
        output,
        lineage={**identity, "ranking_eligible": False},
    )
    write_artifact_manifest(output, output_manifest)
    return AnalysisArtifacts(
        output_dir=str(output),
        identity=identity,
        c2_status=str(result.get("c2", {}).get("status", "exploratory_only")),
        files=tuple(sorted(path.name for path in output.glob("*.json"))),
    )


def report_analysis(analysis_dir: str | Path) -> ArtifactManifest:
    """Validate and return the immutable analysis artifact manifest."""

    root = Path(analysis_dir).expanduser().resolve()
    return validate_artifact_manifest(root, expected_lineage={"ranking_eligible": False})


__all__ = [
    "AnalysisArtifacts",
    "AnalysisRequest",
    "AuditReport",
    "analyze_experiment",
    "audit_experiment",
    "report_analysis",
    "resume_experiment",
    "run_experiment",
]
