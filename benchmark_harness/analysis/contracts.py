"""Data contracts shared by offline analysis stages."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


class AnalysisError(ValueError):
    """Raised when analysis input or lineage is incomplete."""


@dataclass(frozen=True, slots=True)
class AnalysisIdentity:
    """Identity of an analysis, independent of mutable output files."""

    experiment_id: str
    manifest_sha256: str
    plan_sha256: str
    dataset_revision: str
    evaluator_manifest_sha256: str
    temporal_index_sha256: str
    cutoff_registry_sha256: str
    metric_catalog_sha256: str
    analysis_as_of: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "manifest_sha256": self.manifest_sha256,
            "plan_sha256": self.plan_sha256,
            "dataset_revision": self.dataset_revision,
            "evaluator_manifest_sha256": self.evaluator_manifest_sha256,
            "temporal_index_sha256": self.temporal_index_sha256,
            "cutoff_registry_sha256": self.cutoff_registry_sha256,
            "metric_catalog_sha256": self.metric_catalog_sha256,
            "analysis_as_of": self.analysis_as_of,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class AnalysisRow:
    """Normalized row from an immutable raw artifact.

    ``metric_values`` contains frozen evaluator outputs copied by key; analysis
    never calls the parser/evaluator and never writes normalized values back to
    the raw prediction artifact.
    """

    system_id: str
    task_id: str
    category: str
    trial: int
    retrieval_regime: str
    temporal_band: str | None
    metric_values: dict[str, float | int | None]
    comparable: bool
    analysis_eligible: bool
    failure_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    framework_id: str | None = None
    strategy_id: str | None = None
    backbone_id: str | None = None
    ranking_eligible: bool = False

    def __post_init__(self) -> None:
        for name in ("system_id", "task_id", "category", "retrieval_regime"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise AnalysisError(f"{name} must be a non-empty string.")
        if not isinstance(self.trial, int) or isinstance(self.trial, bool) or self.trial < 1:
            raise AnalysisError("trial must be a positive integer.")
        if not isinstance(self.comparable, bool) or not isinstance(self.analysis_eligible, bool):
            raise AnalysisError("comparable and analysis_eligible must be booleans.")
        if self.temporal_band is not None and not isinstance(self.temporal_band, str):
            raise AnalysisError("temporal_band must be a string or None.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "task_id": self.task_id,
            "category": self.category,
            "trial": self.trial,
            "retrieval_regime": self.retrieval_regime,
            "temporal_band": self.temporal_band,
            "metric_values": dict(sorted(self.metric_values.items())),
            "comparable": self.comparable,
            "analysis_eligible": self.analysis_eligible,
            "failure_reason": self.failure_reason,
            "diagnostics": self.diagnostics,
            "framework_id": self.framework_id,
            "strategy_id": self.strategy_id,
            "backbone_id": self.backbone_id,
            "ranking_eligible": self.ranking_eligible,
        }


def coerce_analysis_rows(rows: Iterable[AnalysisRow | dict[str, Any]]) -> tuple[AnalysisRow, ...]:
    result: list[AnalysisRow] = []
    for index, row in enumerate(rows):
        if isinstance(row, AnalysisRow):
            result.append(row)
            continue
        if not isinstance(row, dict):
            raise AnalysisError(f"analysis row {index} must be an object.")
        metric_values = row.get("metric_values")
        if metric_values is None:
            metric_values = {
                key: value
                for key, value in row.items()
                if key.startswith("metric_") and isinstance(value, int | float | type(None))
            }
        if not isinstance(metric_values, dict):
            raise AnalysisError(f"analysis row {index}.metric_values must be an object.")
        result.append(
            AnalysisRow(
                system_id=row.get("system_id", ""),
                task_id=row.get("task_id", ""),
                category=row.get("category", ""),
                trial=row.get("trial", 0),
                retrieval_regime=row.get("retrieval_regime", ""),
                temporal_band=row.get("temporal_band", row.get("paper_band")),
                metric_values=metric_values,
                comparable=row.get("comparable", False),
                analysis_eligible=row.get("analysis_eligible", False),
                failure_reason=row.get("failure_reason"),
                diagnostics=row.get("diagnostics", {}),
                framework_id=row.get("framework_id"),
                strategy_id=row.get("strategy_id"),
                backbone_id=row.get("backbone_id"),
                ranking_eligible=row.get("ranking_eligible", False),
            )
        )
    return tuple(result)
