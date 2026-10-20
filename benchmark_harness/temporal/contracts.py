"""Immutable temporal-index and assignment contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any


class TemporalApplicability(StrEnum):
    FORECAST = "forecast"
    ATEMPORAL = "atemporal"
    COUNTERFACTUAL = "counterfactual"


class TemporalStatus(StrEnum):
    T1_EXPOSURE_PRONE = "T1_EXPOSURE_PRONE"
    T2_PASTCAST = "T2_PASTCAST"
    T3_LIVE_PENDING = "T3_LIVE_PENDING"
    T3_LIVE_RESOLVED = "T3_LIVE_RESOLVED"
    TEMPORAL_QUARANTINE = "TEMPORAL_QUARANTINE"
    ATEMPORAL_CONTROL = "ATEMPORAL_CONTROL"
    COUNTERFACTUAL_CONTROL = "COUNTERFACTUAL_CONTROL"


def parse_temporal_date(value: Any, *, field_name: str, allow_none: bool = True) -> date | None:
    """Parse day-precision ISO dates without using the machine's current date."""

    if value is None and allow_none:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            if text.endswith("Z"):
                return datetime.fromisoformat(text[:-1] + "+00:00").date()
            return date.fromisoformat(text[:10])
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO date or datetime.") from exc
    if value is None:
        raise ValueError(f"{field_name} is required.")
    raise ValueError(f"{field_name} must be an ISO date or datetime.")


def _hash_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TemporalIndexRow:
    """One reviewed, external temporal anchor for one task."""

    task_id: str
    temporal_applicability: TemporalApplicability
    forecast_origin: date | None
    outcome_available_at: date | None
    source_time_band: str
    evidence: tuple[Mapping[str, Any], ...] = ()
    review_status: str = "verified"
    quality_audit: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string.")
        applicability = self.temporal_applicability
        if not isinstance(applicability, TemporalApplicability):
            try:
                applicability = TemporalApplicability(str(applicability).strip().lower())
            except ValueError as exc:
                raise ValueError(
                    "temporal_applicability must be forecast, atemporal, or counterfactual."
                ) from exc
            object.__setattr__(self, "temporal_applicability", applicability)
        if not isinstance(self.source_time_band, str) or not self.source_time_band.strip():
            raise ValueError("source_time_band must be a non-empty string.")
        if self.review_status not in {"verified", "reviewed"}:
            raise ValueError("review_status must be 'verified' or 'reviewed'.")
        if applicability is TemporalApplicability.FORECAST and self.forecast_origin is None:
            raise ValueError("forecast rows require forecast_origin.")
        if not isinstance(self.evidence, tuple):
            object.__setattr__(self, "evidence", tuple(self.evidence))
        for item in self.evidence:
            if not isinstance(item, Mapping):
                raise ValueError("evidence entries must be JSON objects.")
        if not isinstance(self.quality_audit, Mapping):
            raise ValueError("quality_audit must be a JSON object.")
        eligible = self.quality_audit.get("analysis_eligible", True)
        reasons = self.quality_audit.get("reasons", [])
        if not isinstance(eligible, bool):
            raise ValueError("quality_audit.analysis_eligible must be a boolean.")
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) or not reason for reason in reasons
        ):
            raise ValueError("quality_audit.reasons must be a list of non-empty strings.")

    @property
    def quality_eligible(self) -> bool:
        return bool(self.quality_audit.get("analysis_eligible", True))

    @property
    def quality_reasons(self) -> tuple[str, ...]:
        return tuple(str(reason) for reason in self.quality_audit.get("reasons", []))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "temporal_applicability": self.temporal_applicability.value,
            "forecast_origin": self.forecast_origin.isoformat() if self.forecast_origin else None,
            "outcome_available_at": (
                self.outcome_available_at.isoformat() if self.outcome_available_at else None
            ),
            "source_time_band": self.source_time_band,
            "evidence": list(self.evidence),
            "review_status": self.review_status,
            "quality_audit": {
                "analysis_eligible": self.quality_eligible,
                "reasons": list(self.quality_reasons),
            },
        }

    @property
    def identity_hash(self) -> str:
        return _hash_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class TemporalAssignment:
    """Pure banding output retained on record/analysis side only."""

    task_id: str
    status: TemporalStatus
    applied_rule: str
    source_time_band: str
    input_hashes: dict[str, str] = field(default_factory=dict)
    quality_audit_eligible: bool = True
    quality_audit_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string.")
        if not isinstance(self.status, TemporalStatus):
            object.__setattr__(self, "status", TemporalStatus(str(self.status)))
        if not isinstance(self.applied_rule, str) or not self.applied_rule.strip():
            raise ValueError("applied_rule must be a non-empty string.")
        if not isinstance(self.source_time_band, str) or not self.source_time_band.strip():
            raise ValueError("source_time_band must be a non-empty string.")
        if not isinstance(self.input_hashes, dict):
            object.__setattr__(self, "input_hashes", dict(self.input_hashes))
        if any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in self.input_hashes.items()
        ):
            raise ValueError("input_hashes must map non-empty strings to hashes.")
        if not isinstance(self.quality_audit_eligible, bool):
            raise ValueError("quality_audit_eligible must be a boolean.")
        object.__setattr__(
            self,
            "quality_audit_reasons",
            tuple(str(reason) for reason in self.quality_audit_reasons if str(reason).strip()),
        )

    @property
    def band_eligible(self) -> bool:
        return self.status in {
            TemporalStatus.T1_EXPOSURE_PRONE,
            TemporalStatus.T2_PASTCAST,
            TemporalStatus.T3_LIVE_RESOLVED,
        }

    @property
    def analysis_eligible(self) -> bool:
        return self.band_eligible and self.quality_audit_eligible

    @property
    def paper_band(self) -> str | None:
        return {
            TemporalStatus.T1_EXPOSURE_PRONE: "T1",
            TemporalStatus.T2_PASTCAST: "T2",
            TemporalStatus.T3_LIVE_PENDING: "T3",
            TemporalStatus.T3_LIVE_RESOLVED: "T3",
        }.get(self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "applied_rule": self.applied_rule,
            "source_time_band": self.source_time_band,
            "input_hashes": dict(sorted(self.input_hashes.items())),
            "band_eligible": self.band_eligible,
            "quality_audit_eligible": self.quality_audit_eligible,
            "quality_audit_reasons": list(self.quality_audit_reasons),
            "analysis_eligible": self.analysis_eligible,
            "paper_band": self.paper_band,
        }
