"""Versioned, exact-backbone training cutoff declarations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from benchmark_harness.temporal.contracts import parse_temporal_date


class CutoffRegistryError(ValueError):
    """Raised when a backbone cutoff registry is unavailable or invalid."""


@dataclass(frozen=True, slots=True)
class BackboneCutoff:
    backbone_id: str
    training_data_cutoff: date
    precision: str
    source: str
    declared_at: str
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("backbone_id", "precision", "source", "declared_at"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if self.precision not in {"day", "month", "year", "unknown"}:
            raise ValueError("precision must be day, month, year, or unknown.")
        if self.precision == "day" and not isinstance(self.training_data_cutoff, date):
            raise ValueError("day-precision cutoffs require a date.")

    @property
    def analysis_usable(self) -> bool:
        return self.precision == "day"

    def to_dict(self) -> dict[str, Any]:
        return {
            "backbone_id": self.backbone_id,
            "training_data_cutoff": self.training_data_cutoff.isoformat(),
            "precision": self.precision,
            "source": self.source,
            "declared_at": self.declared_at,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class CutoffRegistry:
    schema_version: str
    models: tuple[BackboneCutoff, ...]
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "backbone-cutoffs/v1":
            raise CutoffRegistryError("unsupported cutoff registry schema version.")
        if not self.models:
            raise CutoffRegistryError("cutoff registry must contain at least one model.")
        ids = [model.backbone_id for model in self.models]
        if len(ids) != len(set(ids)):
            raise CutoffRegistryError("cutoff registry contains duplicate backbone_id values.")

    def for_backbone(self, backbone_id: str) -> BackboneCutoff:
        for model in self.models:
            if model.backbone_id == backbone_id:
                return model
        raise CutoffRegistryError(f"no cutoff declaration for backbone {backbone_id!r}.")


def load_cutoff_registry(path: str | Path) -> CutoffRegistry:
    """Load YAML/JSON cutoff declarations without supplying defaults."""

    registry_path = Path(path).expanduser().resolve()
    try:
        raw = registry_path.read_bytes()
    except OSError as exc:
        raise CutoffRegistryError(f"cutoff registry is unavailable: {registry_path}") from exc
    try:
        import yaml

        value = yaml.safe_load(raw.decode("utf-8"))
    except ImportError as exc:
        raise CutoffRegistryError("PyYAML is required to load a cutoff registry.") from exc
    except Exception as exc:
        raise CutoffRegistryError(f"invalid cutoff registry YAML: {registry_path}") from exc
    if not isinstance(value, dict):
        raise CutoffRegistryError("cutoff registry must be a YAML object.")
    if value.get("schema_version") != "backbone-cutoffs/v1":
        raise CutoffRegistryError("cutoff registry schema_version must be backbone-cutoffs/v1.")
    raw_models = value.get("models")
    if not isinstance(raw_models, list):
        raise CutoffRegistryError("cutoff registry models must be a list.")
    models: list[BackboneCutoff] = []
    for index, item in enumerate(raw_models):
        if not isinstance(item, dict):
            raise CutoffRegistryError(f"cutoff registry model {index} must be an object.")
        try:
            cutoff = parse_temporal_date(
                item.get("training_data_cutoff"),
                field_name=f"models[{index}].training_data_cutoff",
                allow_none=False,
            )
            assert cutoff is not None
            declared_at = item["declared_at"]
            if isinstance(declared_at, datetime):
                declared_at = declared_at.isoformat()
            elif isinstance(declared_at, date):
                declared_at = declared_at.isoformat()
            model = BackboneCutoff(
                backbone_id=item["backbone_id"],
                training_data_cutoff=cutoff,
                precision=item["precision"],
                source=item["source"],
                declared_at=declared_at,
                notes=item.get("notes", ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CutoffRegistryError(f"invalid cutoff model {index}: {exc}") from exc
        models.append(model)
    try:
        return CutoffRegistry(
            schema_version=value["schema_version"],
            models=tuple(models),
            path=registry_path,
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    except (TypeError, ValueError, CutoffRegistryError) as exc:
        if isinstance(exc, CutoffRegistryError):
            raise
        raise CutoffRegistryError(str(exc)) from exc
