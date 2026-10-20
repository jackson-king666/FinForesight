"""Fail-closed checks preventing analysis-only temporal data from leaking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

FORBIDDEN_SYSTEM_FIELDS = frozenset(
    {
        "ground_truth",
        "evaluator_state",
        "source_time_band",
        "time_band",
        "dynamic_time_band",
        "temporal_assignment",
        "temporal_applicability",
        "temporal_index",
        "backbone_cutoff",
        "backbone_training_cutoff",
        "training_data_cutoff",
        "forecast_origin",
        "outcome_available_at",
        "provider_credentials",
        "admin_token",
        "full_dataset_record",
    }
)


class TemporalLeakageError(ValueError):
    """Raised when record/analysis-only fields reach a system-facing payload."""


def assert_no_temporal_leakage(value: Any, *, path: str = "payload") -> None:
    """Recursively reject forbidden field names in adapter/tool/IPC-shaped data."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in FORBIDDEN_SYSTEM_FIELDS:
                raise TemporalLeakageError(
                    f"{path}.{key} is analysis-only and cannot cross the boundary."
                )
            assert_no_temporal_leakage(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            assert_no_temporal_leakage(item, path=f"{path}[{index}]")
