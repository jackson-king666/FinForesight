"""Pure dynamic temporal-band assignment."""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta

from benchmark_harness.temporal.contracts import (
    TemporalApplicability,
    TemporalAssignment,
    TemporalIndexRow,
    TemporalStatus,
    parse_temporal_date,
)


def _hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assign_temporal_status(
    task_temporal_index: TemporalIndexRow,
    backbone_cutoff: date | str,
    experiment_as_of: date | str,
    *,
    analysis_as_of: date | str,
    guard_days: int = 30,
) -> TemporalAssignment:
    """Assign a paper status using only declared, immutable temporal inputs."""

    if not isinstance(task_temporal_index, TemporalIndexRow):
        raise TypeError("task_temporal_index must be a TemporalIndexRow.")
    cutoff = parse_temporal_date(backbone_cutoff, field_name="backbone_cutoff", allow_none=False)
    experiment = parse_temporal_date(
        experiment_as_of, field_name="experiment_as_of", allow_none=False
    )
    analysis = parse_temporal_date(
        analysis_as_of,
        field_name="analysis_as_of",
        allow_none=False,
    )
    assert cutoff is not None and experiment is not None and analysis is not None
    if not isinstance(guard_days, int) or isinstance(guard_days, bool) or guard_days < 0:
        raise ValueError("guard_days must be a non-negative integer.")
    if analysis < experiment:
        raise ValueError("analysis_as_of must be on or after experiment_as_of.")

    row = task_temporal_index
    status: TemporalStatus
    rule: str
    if row.temporal_applicability is TemporalApplicability.ATEMPORAL:
        status = TemporalStatus.ATEMPORAL_CONTROL
        rule = "atemporal_control"
    elif row.temporal_applicability is TemporalApplicability.COUNTERFACTUAL:
        status = TemporalStatus.COUNTERFACTUAL_CONTROL
        rule = "counterfactual_control"
    elif row.forecast_origin is not None and row.forecast_origin >= experiment:
        if row.outcome_available_at is not None and row.outcome_available_at <= analysis:
            status = TemporalStatus.T3_LIVE_RESOLVED
            rule = "forecast_origin_on_or_after_experiment_and_outcome_resolved"
        else:
            status = TemporalStatus.T3_LIVE_PENDING
            rule = "forecast_origin_on_or_after_experiment_and_outcome_pending"
    elif row.outcome_available_at is not None and row.outcome_available_at <= cutoff - timedelta(
        days=guard_days
    ):
        status = TemporalStatus.T1_EXPOSURE_PRONE
        rule = "outcome_available_before_backbone_cutoff_guard"
    elif (
        row.forecast_origin is not None
        and row.outcome_available_at is not None
        and row.forecast_origin >= cutoff + timedelta(days=guard_days)
        and row.outcome_available_at <= experiment - timedelta(days=guard_days)
    ):
        status = TemporalStatus.T2_PASTCAST
        rule = "forecast_origin_after_backbone_cutoff_and_outcome_before_experiment_guard"
    else:
        status = TemporalStatus.TEMPORAL_QUARANTINE
        rule = "incomplete_or_boundary_temporal_anchors"

    return TemporalAssignment(
        task_id=row.task_id,
        status=status,
        applied_rule=rule,
        source_time_band=row.source_time_band,
        input_hashes={
            "task_temporal_index_sha256": row.identity_hash,
            "backbone_cutoff_sha256": _hash(cutoff.isoformat()),
            "experiment_as_of_sha256": _hash(experiment.isoformat()),
            "analysis_as_of_sha256": _hash(analysis.isoformat()),
            "guard_days_sha256": _hash(guard_days),
        },
        quality_audit_eligible=row.quality_eligible,
        quality_audit_reasons=row.quality_reasons,
    )
