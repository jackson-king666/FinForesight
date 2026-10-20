"""Metric-catalog parsing with participant-independent normalizers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_harness.analysis.contracts import AnalysisError


@dataclass(frozen=True, slots=True)
class MetricSpec:
    name: str
    source: str
    direction: str
    primary: bool
    normalizer: Mapping[str, Any]
    weight: float = 1.0
    cap: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.source:
            raise AnalysisError("metric name and source must be non-empty.")
        if self.direction not in {"maximize", "minimize"}:
            raise AnalysisError(f"metric {self.name!r} has an invalid direction.")
        if not isinstance(self.primary, bool):
            raise AnalysisError("metric primary must be a boolean.")
        if not isinstance(self.normalizer, Mapping):
            raise AnalysisError(f"metric {self.name!r} normalizer must be an object.")
        kind = self.normalizer.get("kind")
        if kind not in {"fixed", "baseline"}:
            raise AnalysisError(
                f"metric {self.name!r} must use a fixed or declared-baseline normalizer."
            )
        if not isinstance(self.normalizer.get("value"), int | float) or isinstance(
            self.normalizer.get("value"), bool
        ):
            raise AnalysisError(f"metric {self.name!r} normalizer.value must be numeric.")
        if self.normalizer["value"] == 0:
            raise AnalysisError(f"metric {self.name!r} normalizer.value cannot be zero.")
        if (
            not isinstance(self.weight, int | float)
            or isinstance(self.weight, bool)
            or self.weight <= 0
        ):
            raise AnalysisError(f"metric {self.name!r} weight must be positive.")
        if self.cap is not None and (
            not isinstance(self.cap, int | float) or isinstance(self.cap, bool) or self.cap <= 0
        ):
            raise AnalysisError(f"metric {self.name!r} cap must be positive when present.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "direction": self.direction,
            "primary": self.primary,
            "normalizer": dict(self.normalizer),
            "weight": self.weight,
            "cap": self.cap,
        }

    def raw_value(self, row: Mapping[str, Any]) -> float | int | None:
        current: Any = row
        path = self.source.split(".")
        for key in path:
            if isinstance(current, Mapping) and key in current:
                current = current[key]
            else:
                current = None
                break
        if current is None and self.source.startswith("metrics."):
            values = row.get("metric_values")
            if isinstance(values, Mapping):
                current = values.get(self.name)
        if current is None:
            values = row.get("metric_values")
            if isinstance(values, Mapping):
                current = values.get(self.name)
        if current is None:
            return None
        if not isinstance(current, int | float) or isinstance(current, bool):
            raise AnalysisError(f"metric {self.name!r} source is not numeric.")
        return current


@dataclass(frozen=True, slots=True)
class MetricCatalog:
    schema_version: str
    metrics: tuple[MetricSpec, ...]
    source_path: Path | None = None
    utility_definition: Mapping[str, Any] | None = None
    categories: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {"metric-catalog/v1", "metric-catalog/v2"}:
            raise AnalysisError("unsupported metric catalog schema version.")
        if not self.metrics:
            raise AnalysisError("metric catalog must contain at least one metric.")
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise AnalysisError("metric catalog contains duplicate metric names.")

    def metric(self, name: str) -> MetricSpec:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise AnalysisError(f"metric catalog has no metric {name!r}.")

    @property
    def hash(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metrics": [metric.to_dict() for metric in self.metrics],
            **({"utility": dict(self.utility_definition)} if self.utility_definition else {}),
            **({"categories": dict(self.categories)} if self.categories else {}),
        }


def _metric_from_mapping(name: str, value: Any) -> MetricSpec:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"metric {name!r} must be an object.")
    return MetricSpec(
        name=name,
        source=value.get("source", f"metric_values.{name}"),
        direction=value.get("direction", "maximize"),
        primary=value.get("primary", False),
        normalizer=value.get("normalizer", {}),
        weight=value.get("weight", 1.0),
        cap=value.get("cap"),
    )


def metric_catalog_from_dict(
    value: Mapping[str, Any], *, source_path: Path | None = None
) -> MetricCatalog:
    schema_version = value.get("schema_version")
    if schema_version not in {"metric-catalog/v1", "metric-catalog/v2"}:
        raise AnalysisError("unsupported metric catalog schema version.")
    raw_metrics = value.get("metrics")
    if isinstance(raw_metrics, Mapping):
        metrics = tuple(_metric_from_mapping(str(name), item) for name, item in raw_metrics.items())
    elif isinstance(raw_metrics, list):
        metrics = tuple(
            _metric_from_mapping(str(item.get("name")), item)
            for item in raw_metrics
            if isinstance(item, Mapping) and item.get("name")
        )
    else:
        raise AnalysisError("metric catalog metrics must be an object or list.")
    return MetricCatalog(
        schema_version=str(schema_version),
        metrics=metrics,
        source_path=source_path,
        utility_definition=(
            dict(value.get("utility", {})) if isinstance(value.get("utility"), Mapping) else None
        ),
        categories=(
            dict(value.get("categories", {}))
            if isinstance(value.get("categories"), Mapping)
            else None
        ),
    )


def load_metric_catalog(path: str | Path) -> MetricCatalog:
    catalog_path = Path(path).expanduser().resolve()
    try:
        value = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"metric catalog is unavailable or invalid: {catalog_path}") from exc
    if not isinstance(value, Mapping):
        raise AnalysisError("metric catalog root must be an object.")
    return metric_catalog_from_dict(value, source_path=catalog_path)
