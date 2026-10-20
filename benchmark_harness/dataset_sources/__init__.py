"""Pinned dataset sources for benchmark runs.

The Hugging Face Hub source is the default v0.6 data path.  The local checkout
adapter is deliberately exported separately because it is migration-only and
must never become an implicit fallback for a failed Hub load.
"""

from benchmark_harness.dataset_sources.base import DatasetSource
from benchmark_harness.dataset_sources.external_checkout import ExternalCheckoutDatasetSource
from benchmark_harness.dataset_sources.huggingface import (
    DATASET_CONFIGS,
    DEFAULT_HF_REPO_ID,
    DEFAULT_HF_REVISION,
    EXPECTED_RECORD_COUNTS,
    EXPECTED_TOTAL_RECORD_COUNT,
    DatasetConfigSpec,
    DatasetSourceError,
    DatasetValidationError,
    HuggingFaceDatasetSource,
    OfflineDatasetUnavailableError,
    config_specs_from_mapping,
)

_V3_EXPORTS = frozenset(
    {
        "DEFAULT_V3_CONFIG_NAMES",
        "V3DatasetSource",
        "V3DatasetSourceError",
        "V3OfflineDatasetUnavailableError",
    }
)


def __getattr__(name: str):
    """Load v3 lazily so calibration imports keep the dataset boundary acyclic."""

    if name in _V3_EXPORTS:
        from benchmark_harness.dataset_sources import v3

        value = getattr(v3, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "DATASET_CONFIGS",
    "DEFAULT_HF_REPO_ID",
    "DEFAULT_HF_REVISION",
    "EXPECTED_RECORD_COUNTS",
    "EXPECTED_TOTAL_RECORD_COUNT",
    "DatasetConfigSpec",
    "DatasetSource",
    "DatasetSourceError",
    "DatasetValidationError",
    "ExternalCheckoutDatasetSource",
    "HuggingFaceDatasetSource",
    "OfflineDatasetUnavailableError",
    "config_specs_from_mapping",
    "DEFAULT_V3_CONFIG_NAMES",
    "V3DatasetSource",
    "V3DatasetSourceError",
    "V3OfflineDatasetUnavailableError",
]
