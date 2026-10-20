"""Project append-only executor attempts into offline analysis rows."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmark_harness.analysis.contracts import AnalysisError
from benchmark_harness.analysis.lineage import (
    ArtifactManifest,
    build_artifact_manifest,
    write_artifact_manifest,
)
from benchmark_harness.analysis.metrics.formal import (
    a1_formal_utility,
    a2_formal_utility,
    b_formal_utility,
    b_probability,
    c_formal_utility,
    e_formal_utility,
    operational_utility,
)
from benchmark_harness.experiment.eligibility import evaluate_ranking_eligibility
from benchmark_harness.experiment.execution_contract import (
    build_execution_contract,
    validate_execution_contract,
)
from benchmark_harness.experiment.live_model import declared_model_identity
from benchmark_harness.experiment.manifest import (
    ExperimentManifest,
    ManifestError,
    validate_manifest_artifacts,
)
from benchmark_harness.experiment.planner import ExperimentPlan, PlanCell
from benchmark_harness.experiment.schedule import AttemptRecord, read_attempts
from benchmark_harness.experiment.t3_lifecycle import (
    T3RegistryError,
    load_t3_registry,
    prospective_gate,
)
from benchmark_harness.temporal import (
    assign_temporal_status,
    load_cutoff_registry,
    load_temporal_index,
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise AnalysisError(f"executor artifact is unavailable: {path}") from exc


def _resolve_manifest_path(manifest: ExperimentManifest, raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = manifest.source_path.parent if manifest.source_path is not None else Path.cwd()
    return (base / path).resolve()


def _category(task_id: str) -> str:
    prefix = task_id.split("-", 1)[0]
    if prefix.startswith("A1"):
        return "A1"
    if prefix.startswith("A2"):
        return "A2"
    if prefix in {"B", "C", "D", "E"}:
        return prefix
    raise AnalysisError(f"cannot derive dataset category from task_id {task_id!r}")


def _v4_cohort(manifest: ExperimentManifest) -> dict[str, dict[str, Any]]:
    section = manifest.v4_metadata.get("cohort")
    if isinstance(section, Mapping):
        raw_path = section.get("path", section.get("ledger_path"))
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise AnalysisError("experiment/v4 cohort section requires a path")
        path = Path(raw_path).expanduser()
        if not path.is_absolute() and manifest.source_path is not None:
            path = manifest.source_path.parent / path
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AnalysisError(f"experiment/v4 cohort ledger is unavailable: {path}") from exc
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        values = (
            parsed
            if isinstance(parsed, list)
            else [json.loads(line) for line in raw.splitlines() if line.strip()]
        )
    elif isinstance(section, list):
        values = section
    else:
        raise AnalysisError("experiment/v4 cohort metadata is missing")
    result: dict[str, dict[str, Any]] = {}
    for row in values:
        if not isinstance(row, Mapping) or not isinstance(row.get("task_id"), str):
            raise AnalysisError("experiment/v4 cohort rows require task_id")
        task_id = str(row["task_id"])
        if task_id in result:
            raise AnalysisError(f"duplicate experiment/v4 cohort task_id: {task_id}")
        result[task_id] = dict(row)
    return result


def _v4_t3_registry(manifest: ExperimentManifest) -> Any | None:
    section = manifest.v4_metadata.get("t3")
    if not isinstance(section, Mapping):
        return None
    raw_path = section.get("path", section.get("registry_path"))
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and manifest.source_path is not None:
        path = manifest.source_path.parent / path
    try:
        return load_t3_registry(path.resolve())
    except T3RegistryError as exc:
        raise AnalysisError(f"experiment/v4 T3 registry is invalid: {exc}") from exc


def _selected_cells(plan: ExperimentPlan, selection: str) -> tuple[PlanCell, ...]:
    if selection == "all":
        return plan.cells
    if selection == "smoke":
        return tuple(cell for cell in plan.cells if cell.smoke)
    if selection in {"dev", "test"}:
        return tuple(cell for cell in plan.cells if cell.split == selection)
    raise AnalysisError(f"unsupported projection selection {selection!r}")


def _read_artifact(
    root: Path,
    record: AttemptRecord,
    *,
    cell: PlanCell,
    plan: ExperimentPlan,
) -> dict[str, Any]:
    if not record.artifact_manifest_sha256:
        raise AnalysisError(f"attempt {cell.cell_id}:{record.attempt} has no artifact hash")
    path = root / "cells" / cell.cell_id / f"attempt-{record.attempt}.json"
    actual_hash = _sha256_file(path)
    if actual_hash != record.artifact_manifest_sha256:
        raise AnalysisError(f"attempt artifact hash drift: {cell.cell_id}:{record.attempt}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"attempt artifact is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"attempt artifact must be an object: {path}")
    binding = value.get("binding")
    if not isinstance(binding, Mapping):
        raise AnalysisError(f"attempt artifact has no binding: {path}")
    expected = {
        "cell_id": cell.cell_id,
        "task_id": cell.task_id,
        "plan_hash": plan.plan_hash,
        "retrieval_regime": cell.retrieval_regime,
        "trial": cell.trial,
    }
    if any(binding.get(key) != expected_value for key, expected_value in expected.items()):
        raise AnalysisError(f"attempt binding mismatch: {cell.cell_id}:{record.attempt}")
    return value


def _numeric_metrics(result: Mapping[str, Any]) -> dict[str, float | int]:
    values: Mapping[str, Any] | None = None
    for key in ("metric_values", "metrics"):
        candidate = result.get(key)
        if isinstance(candidate, Mapping):
            values = candidate
            break
    metrics: dict[str, float | int] = {}
    if values is not None:
        metrics.update(
            {
                str(key): value
                for key, value in values.items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
        )
    utility = result.get("utility")
    if isinstance(utility, int | float) and not isinstance(utility, bool):
        metrics.setdefault("exploratory_task_utility_v1", utility)
    return metrics


def _raw_text(result: Mapping[str, Any]) -> str | None:
    for key in ("raw_text", "prediction"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_projection_response(category: str, raw: str) -> dict[str, Any]:
    """Use the frozen parser while skipping unrelated JSON arrays in prose.

    Research responses often include citation markers such as ``[1]`` before
    their final structured answer.  The frozen parser validates a selected
    candidate, so this recovery only chooses a later category-shaped object;
    it does not add parser or evaluator rules of its own.
    """

    from benchmark_harness.reference_evaluation.engine import parse_prediction

    parsed = parse_prediction(category, raw)
    if parsed.get("format_valid"):
        return parsed
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(candidate, Mapping):
            continue
        # The PIT C overlay keeps its source task contract in
        # ``value_usd_millions`` while the frozen evaluator's historical
        # parser calls the same field ``predicted_value``.  Treat this as a
        # projection-only alias: the raw response and the sealed task prompt
        # remain unchanged, and no score is synthesized.
        if category == "C" and "predicted_value" not in candidate:
            value = candidate.get("value_usd_millions")
            if value is not None:
                candidate = {**candidate, "predicted_value": value}
        candidate_parsed = parse_prediction(
            category,
            json.dumps(candidate, ensure_ascii=False, sort_keys=True),
        )
        if candidate_parsed.get("format_valid"):
            return candidate_parsed
    return parsed


def _frozen_scoring(
    task_id: str,
    result: Mapping[str, Any],
    record: Mapping[str, Any] | None,
    *,
    category: str | None = None,
) -> tuple[dict[str, float | int], dict[str, Any], str | None]:
    """Materialize frozen parser/evaluator outputs from an immutable response.

    Never mutates the frozen engine.  A missing record, missing raw response,
    invalid parse, or invalid evaluation is an explicit exclusion reason, not
    a zero-filled metric.
    """

    from benchmark_harness.analysis.metrics.exploratory import exploratory_task_utility
    from benchmark_harness.reference_evaluation.engine import evaluate_prediction

    category = category or _category(task_id)
    if record is None:
        return {}, {}, "dataset_record_unavailable_for_frozen_scoring"
    raw = _raw_text(result)
    if raw is None:
        return {}, {}, "raw_prediction_missing"
    try:
        parsed = _parse_projection_response(category, raw)
    except (KeyError, TypeError, ValueError) as exc:
        return {}, {}, f"parse_invalid:{type(exc).__name__}"
    if not parsed.get("format_valid"):
        return {}, {}, "parse_invalid"
    evaluation_record: Mapping[str, Any] = record
    if category == "C":
        # ``versioned-task-manifest.jsonl`` stores the source-backed C target
        # in task units as ``value_usd_millions``.  The frozen evaluator uses
        # its legacy generic name ``future_value``.  Copy only that field for
        # the evaluator call; the original record and target remain intact.
        ground_truth = record.get("ground_truth")
        if isinstance(ground_truth, Mapping) and "future_value" not in ground_truth:
            target = ground_truth.get("value_usd_millions")
            if target is not None:
                evaluation_record = {
                    **record,
                    "ground_truth": {**ground_truth, "future_value": target},
                }
    try:
        frozen_metrics = dict(evaluate_prediction(category, parsed, evaluation_record))
    except (KeyError, TypeError, ValueError) as exc:
        return {}, {}, f"frozen_evaluator_error:{type(exc).__name__}"
    ground_truth = evaluation_record.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        return {}, frozen_metrics, "dataset_ground_truth_missing"
    try:
        utility, reason = exploratory_task_utility(category, parsed, ground_truth)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        utility, reason = None, f"frozen_evaluator_error:{type(exc).__name__}"
    if utility is None:
        return {}, frozen_metrics, f"utility_missing:{reason}"
    return {"exploratory_task_utility_v1": utility}, frozen_metrics, None


def _lineage(
    manifest: ExperimentManifest, plan: ExperimentPlan, *, selection: str
) -> dict[str, Any]:
    validated = validate_manifest_artifacts(manifest)
    try:
        identity = declared_model_identity(manifest)
    except ManifestError:
        model = manifest.v2_metadata.get("model", {})
        backbone = manifest.v2_metadata.get("backbone", {})
        identity = {
            "backbone_id": model.get("backbone_id", backbone.get("backbone_id"))
            if isinstance(model, Mapping) and isinstance(backbone, Mapping)
            else None,
            "snapshot_model_name": model.get("snapshot_model_name")
            if isinstance(model, Mapping)
            else None,
            "wire_model_name": model.get("api_model_name", model.get("model_name"))
            if isinstance(model, Mapping)
            else None,
            "wire_name_binding": model.get("wire_name_binding")
            if isinstance(model, Mapping)
            else None,
        }
    evaluator_path = (
        manifest.source_path.parent.parent if manifest.source_path is not None else Path.cwd()
    ) / "benchmark_harness/reference_evaluation/EVALUATOR_MANIFEST.json"
    if not evaluator_path.is_file():
        evaluator_path = (
            Path(__file__).resolve().parents[1] / "reference_evaluation/EVALUATOR_MANIFEST.json"
        )
    return {
        "experiment_id": manifest.experiment_id,
        "manifest_sha256": manifest.manifest_hash,
        "plan_sha256": plan.plan_hash,
        "dataset_revision": manifest.dataset_revision,
        "evaluator_manifest_sha256": _sha256_file(evaluator_path),
        "temporal_index_sha256": validated["temporal_index_sha256"],
        "cutoff_registry_sha256": validated.get("cutoff_registry_sha256"),
        "metric_catalog_sha256": validated.get("metric_catalog_sha256"),
        "backbone_id": identity["backbone_id"],
        "snapshot_model_name": identity["snapshot_model_name"],
        "wire_model_name": identity["wire_model_name"],
        "wire_name_binding": identity["wire_name_binding"],
        "execution_contract": build_execution_contract(manifest, plan),
        "analysis_as_of": (manifest.analysis_as_of or manifest.experiment_as_of).isoformat(),
        "selection": selection,
        "ranking_eligible": False,
    }


def project_execution_artifacts(
    root: str | Path,
    manifest: ExperimentManifest,
    plan: ExperimentPlan,
    *,
    selection: str = "all",
    records: Mapping[str, Mapping[str, Any]] | None = None,
) -> ArtifactManifest:
    """Create a deterministic, lineage-checked ``analysis_rows.jsonl``.

    The projector never invents predictions.  It materializes frozen
    parser/evaluator outputs and the catalog utility from immutable raw
    responses when dataset records are available; otherwise already-carried
    numeric metric values pass through.  Missing or invalid material is an
    explicit exclusion reason, never a zero-filled metric.
    """

    root_path = Path(root).expanduser().resolve()
    if plan.manifest_hash != manifest.manifest_hash:
        raise ManifestError("manifest hash does not match experiment plan")
    validate_execution_contract(root_path, manifest, plan)
    ledger_path = root_path / "attempts.jsonl"
    if not ledger_path.is_file():
        raise AnalysisError(f"executor attempt ledger is missing: {ledger_path}")
    task_records = records or {}
    records_by_cell = {
        str(task_id): record
        for task_id, record in task_records.items()
        if isinstance(record, Mapping)
    }
    attempt_records = read_attempts(ledger_path)
    by_cell: dict[str, list[AttemptRecord]] = defaultdict(list)
    for record in attempt_records:
        by_cell[record.cell_id].append(record)
    cells = _selected_cells(plan, selection)
    if not cells:
        raise AnalysisError(f"plan contains no cells for selection {selection!r}")

    temporal_path = _resolve_manifest_path(manifest, manifest.temporal_index_path)
    cutoff_path = _resolve_manifest_path(manifest, manifest.cutoff_registry_path)
    temporal = load_temporal_index(temporal_path)
    cutoff_registry = load_cutoff_registry(cutoff_path)
    is_v4 = manifest.schema_version == "experiment/v4"
    cohort = _v4_cohort(manifest) if is_v4 else {}
    t3_registry = _v4_t3_registry(manifest) if is_v4 else None
    gates_by_system = manifest.v4_metadata.get("ranking_gates", {}) if is_v4 else {}
    ranking_requested = manifest.v4_metadata.get("ranking_requested", True) if is_v4 else False
    rows: list[dict[str, Any]] = []
    for cell in cells:
        system = manifest.system(cell.system_id)
        cohort_row = cohort.get(cell.task_id, {}) if is_v4 else {}
        if is_v4 and cell.task_id not in cohort:
            raise AnalysisError(f"experiment/v4 cohort metadata missing for {cell.task_id!r}")
        # Historical cohort ledgers may predate metadata fields that are
        # already present in the immutable task supplement.  Carry only the
        # non-label control metadata into the derived row; never replace the
        # cohort category or any A/B/C label.  This keeps re-projection
        # compatible with an existing plan/cell binding while restoring D/E
        # grouping from the sealed source records.
        projection_cohort_row = dict(cohort_row)
        source_row = records_by_cell.get(cell.task_id)
        if isinstance(source_row, Mapping):
            for key in ("formula_family", "history_conflict", "stipulated_direction"):
                if projection_cohort_row.get(key) is None and key in source_row:
                    projection_cohort_row[key] = source_row[key]
        temporal_row = temporal.rows.get(cell.task_id)
        if temporal_row is not None:
            assignment = assign_temporal_status(
                temporal_row,
                cutoff_registry.for_backbone(system.backbone_id).training_data_cutoff,
                manifest.experiment_as_of,
                analysis_as_of=manifest.analysis_as_of or manifest.experiment_as_of,
                guard_days=manifest.guard_days,
            )
            paper_band = assignment.paper_band
            source_band = assignment.source_time_band
            temporal_status = assignment.status.value
            temporal_rule = assignment.applied_rule
            temporal_analysis_eligible = assignment.analysis_eligible
            temporal_band_eligible = assignment.band_eligible
            quality_audit_eligible = assignment.quality_audit_eligible
            quality_audit_reasons = list(assignment.quality_audit_reasons)
            temporal_input_hashes = assignment.input_hashes
        elif is_v4:
            paper_band = projection_cohort_row.get(
                "computed_band", projection_cohort_row.get("paper_band")
            )
            source_band = projection_cohort_row.get(
                "source_band", projection_cohort_row.get("source_time_band")
            )
            temporal_status = str(projection_cohort_row.get("temporal_status", "cohort_declared"))
            temporal_rule = "v4_cohort_ledger"
            temporal_analysis_eligible = bool(projection_cohort_row.get("analysis_eligible", True))
            temporal_band_eligible = bool(projection_cohort_row.get("band_eligible", True))
            quality_audit_eligible = bool(projection_cohort_row.get("quality_audit_eligible", True))
            quality_audit_reasons = list(projection_cohort_row.get("quality_audit_reasons", []))
            temporal_input_hashes = {}
        else:
            raise AnalysisError(f"temporal index has no row for task {cell.task_id!r}")
        history = sorted(by_cell.get(cell.cell_id, ()), key=lambda item: item.attempt)
        successful = [record for record in history if record.status == "succeeded"]
        record = successful[0] if successful else (history[-1] if history else None)
        payload: dict[str, Any] | None = None
        if record is not None:
            payload = _read_artifact(root_path, record, cell=cell, plan=plan)
        result = payload.get("result") if isinstance(payload, Mapping) else None
        result = result if isinstance(result, Mapping) else {}
        metric_values = _numeric_metrics(result)
        frozen_metrics = (
            dict(result["frozen_metrics"])
            if isinstance(result.get("frozen_metrics"), Mapping)
            else {}
        )
        scoring_reason: str | None = None
        if not frozen_metrics and record is not None and record.status == "succeeded":
            derived_metric_values, derived_frozen_metrics, scoring_reason = _frozen_scoring(
                cell.task_id,
                result,
                records_by_cell.get(cell.task_id),
                category=str(projection_cohort_row.get("category")) if is_v4 else None,
            )
            if not metric_values:
                metric_values = derived_metric_values
            frozen_metrics = derived_frozen_metrics
        category = str(cohort_row.get("category", _category(cell.task_id))).upper()
        formal_value: float | None = None
        formal_reason: str | None = None
        if is_v4:
            if category == "A1":
                formal_value, formal_reason = a1_formal_utility(frozen_metrics)
            elif category == "A2":
                formal_value, formal_reason = a2_formal_utility(frozen_metrics)
            elif category == "B":
                formal_value, formal_reason = b_formal_utility(frozen_metrics)
                # Supplemental artifacts intentionally contain only raw_text;
                # the frozen evaluator carries the parser's explicit
                # probability alongside Brier.  Prefer that immutable value
                # and retain the parser-shaped fallback for legacy artifacts.
                probability, probability_reason = b_probability(frozen_metrics)
                if probability is None and "probability_up" not in frozen_metrics:
                    parsed = result.get("parsed")
                    probability, probability_reason = (
                        b_probability(parsed)
                        if isinstance(parsed, Mapping)
                        else b_probability(result)
                    )
                if probability is not None:
                    metric_values["analysis_probability"] = probability
                elif probability_reason is not None:
                    # A frozen Brier value is not enough for the v4 B
                    # estimand: the analysis field must be copied from an
                    # explicit, in-range probability_up.  Never let a
                    # directional or legacy metric stand in for it.
                    formal_value = None
                    formal_reason = probability_reason
                    scoring_reason = scoring_reason or probability_reason
            elif category == "C":
                formal_value, formal_reason = c_formal_utility(frozen_metrics)
            elif category == "E":
                formal_value, formal_reason = e_formal_utility(frozen_metrics)
            if formal_value is not None:
                metric_values["utility"] = formal_value
                metric_values["formal_utility"] = formal_value
            else:
                scoring_reason = scoring_reason or formal_reason
        failure_reason = (
            record.reason if record is not None and record.status != "succeeded" else None
        )
        if is_v4 and formal_value is None and (scoring_reason or formal_reason):
            failure_reason = failure_reason or scoring_reason or formal_reason
        elif not metric_values:
            failure_reason = failure_reason or scoring_reason or "frozen_evaluator_metrics_missing"
        operational_value = operational_utility(formal_value, failure_reason) if is_v4 else None
        analysis_eligible = temporal_analysis_eligible
        t3_reason: str | None = None
        if (
            is_v4
            and t3_registry is not None
            and (
                str(projection_cohort_row.get("temporal_band", "")).upper() == "T3"
                or str(projection_cohort_row.get("phase", "")).upper() == "T3"
                or projection_cohort_row.get("prospective") is True
            )
        ):
            gate_status, t3_reason = prospective_gate(
                t3_registry,
                cell.task_id,
                system_id=cell.system_id,
                trial=cell.trial,
            )
            analysis_eligible = analysis_eligible and gate_status == "settled"
            if t3_reason is not None:
                failure_reason = failure_reason or t3_reason
        comparable = (
            (formal_value is not None if is_v4 else bool(metric_values))
            and record is not None
            and record.status == "succeeded"
        )
        if is_v4 and not analysis_eligible:
            comparable = False
        system_gates = (
            gates_by_system.get(cell.system_id, {}) if isinstance(gates_by_system, Mapping) else {}
        )
        if not isinstance(system_gates, Mapping):
            system_gates = {}
        ranking = (
            evaluate_ranking_eligibility(
                ranking_requested=bool(ranking_requested),
                retrieval_regime=cell.retrieval_regime,
                gates=system_gates,
                comparable=comparable,
                artifact_complete=record is not None and bool(metric_values),
            )
            if is_v4
            else None
        )
        rows.append(
            {
                "system_id": cell.system_id,
                "task_id": cell.task_id,
                "category": category,
                "variant": projection_cohort_row.get("variant") if is_v4 else None,
                "market": projection_cohort_row.get("market") if is_v4 else None,
                "matched_pair_id": (
                    projection_cohort_row.get("matched_pair_id") if is_v4 else None
                ),
                "event_cluster_id": (
                    projection_cohort_row.get("event_cluster_id") if is_v4 else None
                ),
                "calendar_block_id": (
                    projection_cohort_row.get("calendar_block_id") if is_v4 else None
                ),
                "horizon": projection_cohort_row.get("horizon") if is_v4 else None,
                "formula_family": (projection_cohort_row.get("formula_family") if is_v4 else None),
                "history_conflict": (
                    projection_cohort_row.get("history_conflict") if is_v4 else None
                ),
                "stipulated_direction": (
                    projection_cohort_row.get("stipulated_direction") if is_v4 else None
                ),
                "trial": cell.trial,
                "retrieval_regime": cell.retrieval_regime,
                "calc_policy": cell.calc_policy,
                "temporal_band": paper_band,
                "source_time_band": source_band,
                "temporal_status": temporal_status,
                "temporal_rule": temporal_rule,
                "metric_values": metric_values,
                "frozen_metrics": frozen_metrics,
                "formal_utility": formal_value,
                "formal_utility_reason": formal_reason,
                "operational_utility": operational_value,
                "scoring_reason": scoring_reason,
                "comparable": comparable,
                "analysis_eligible": analysis_eligible,
                "band_eligible": temporal_band_eligible,
                "quality_audit_eligible": quality_audit_eligible,
                "quality_audit_reasons": quality_audit_reasons,
                "failure_reason": failure_reason,
                "diagnostics": {
                    "attempt": record.attempt if record is not None else None,
                    "attempt_status": record.status if record is not None else "missing",
                    "temporal_input_hashes": temporal_input_hashes,
                    "t3_gate_reason": t3_reason,
                },
                "framework_id": system.framework_id,
                "strategy_id": system.strategy_id,
                "backbone_id": system.backbone_id,
                "model_provenance": (
                    payload.get("model_provenance")
                    if isinstance(payload, Mapping)
                    and isinstance(payload.get("model_provenance"), Mapping)
                    else {}
                ),
                "ranking_eligible": ranking.ranking_eligible if ranking is not None else False,
                "ranking_report": ranking.to_dict() if ranking is not None else None,
            }
        )

    rows.sort(
        key=lambda row: (
            row["task_id"],
            row["trial"],
            row["system_id"],
            row["retrieval_regime"],
            row.get("calc_policy", "calc_on"),
        )
    )
    raw = "".join(_stable_json(row) + "\n" for row in rows).encode("utf-8")
    try:
        (root_path / "analysis_rows.jsonl").write_bytes(raw)
    except OSError as exc:
        raise AnalysisError(f"cannot write projected analysis rows: {root_path}") from exc
    manifest_lineage = _lineage(manifest, plan, selection=selection)
    artifact_manifest = build_artifact_manifest(root_path, lineage=manifest_lineage)
    write_artifact_manifest(root_path, artifact_manifest)
    return artifact_manifest


__all__ = ["project_execution_artifacts"]
