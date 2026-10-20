"""Fail-closed, revision-pinned Hugging Face dataset loading.

This module intentionally imports :mod:`datasets` only after configuring the
repository-local cache environment.  It contains no local-checkout fallback:
an unavailable Hub revision or an offline cache miss makes the run
non-comparable instead of silently changing its input data.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_HF_REPO_ID = "sselaine27/benchmark-research"
DEFAULT_HF_REVISION = "e941b978af5f8a38fbeb4670ee2d77671a5b551e"
HF_RELEASE_LABEL = "v0.6.0"
DEFAULT_CACHE_DIR = Path("data/.huggingface")


class DatasetSourceError(RuntimeError):
    """Base error for a dataset source that cannot provide comparable input."""


class DatasetValidationError(DatasetSourceError):
    """Raised when a pinned dataset does not match the benchmark schema."""


class OfflineDatasetUnavailableError(DatasetSourceError):
    """Raised when required Hub data is absent from the local offline cache."""


@dataclass(frozen=True, slots=True)
class DatasetConfigSpec:
    """One Hub config and the records it is allowed to contain."""

    name: str
    category: str
    variant: str | None
    expected_count: int
    allow_any_variant: bool = False


# Ordering is intentional and matches the public Hub configuration matrix.
DATASET_CONFIGS: tuple[DatasetConfigSpec, ...] = (
    DatasetConfigSpec("a1", "A1", None, 72),
    DatasetConfigSpec("a2_f", "A2", "F", 10),
    DatasetConfigSpec("a2_t", "A2", "T", 10),
    DatasetConfigSpec("a2_h", "A2", "H", 10),
    DatasetConfigSpec("b", "B", None, 33, allow_any_variant=True),
    DatasetConfigSpec("c", "C", None, 180),
    DatasetConfigSpec("d", "D", None, 12),
    DatasetConfigSpec("e", "E", None, 12),
)
EXPECTED_RECORD_COUNTS: dict[str, int] = {
    spec.name: spec.expected_count for spec in DATASET_CONFIGS
}
EXPECTED_TOTAL_RECORD_COUNT = sum(EXPECTED_RECORD_COUNTS.values())

_REQUIRED_FIELDS = frozenset(
    {
        "task_id",
        "category",
        "variant",
        "time_band",
        "status",
        "seed",
        "prompt",
        "expected_output",
        "ground_truth",
        "metadata",
    }
)


def configure_huggingface_cache(cache_dir: str | Path, *, offline: bool) -> Path:
    """Point Hugging Face caches into the untracked ``data/.huggingface`` tree.

    The variables are set before importing ``datasets`` or calling its loader.
    ``HF_HOME`` is intentionally never set, so unrelated Hugging Face state is
    not redirected or unexpectedly captured by the benchmark cache.
    """

    root = Path(cache_dir).expanduser().resolve()
    hub_cache = root / "hub"
    datasets_cache = root / "datasets"
    hub_cache.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    if offline:
        # Both flags are required: Hub resolution and dataset-builder downloads
        # must fail locally rather than reaching the network.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    else:
        # A previous offline test/run may have set process-global flags. A live
        # source must not inherit that mode and falsely appear to be offline.
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE", None)
    return root


def _json_hash(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_specs_from_mapping(
    config_map: Mapping[str, str], expected_counts: Mapping[str, int]
) -> tuple[DatasetConfigSpec, ...]:
    """Convert the TOML category/config mapping into loader validation specs.

    The configuration schema stores selectors such as ``A2-F`` as keys and
    Hub config names such as ``a2_f`` as values.  Keeping this conversion here
    makes the source, rather than the runner, responsible for enforcing every
    configured count/category/variant combination.
    """

    if set(config_map.values()) != set(expected_counts):
        raise ValueError("config_map values must exactly match expected_counts keys.")

    specs: list[DatasetConfigSpec] = []
    for selector, config_name in config_map.items():
        if not isinstance(selector, str) or not isinstance(config_name, str):
            raise ValueError("Dataset config mappings must use non-empty strings.")
        category, separator, variant = selector.upper().partition("-")
        if category not in {"A1", "A2", "B", "C", "D", "E"}:
            raise ValueError(f"Unsupported dataset category selector {selector!r}.")
        if category == "A2" and variant not in {"F", "T", "H"}:
            raise ValueError("A2 dataset selectors must be A2-F, A2-T, or A2-H.")
        if category != "A2" and separator:
            raise ValueError(f"Only A2 has a fixed variant selector, got {selector!r}.")
        count = expected_counts[config_name]
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"Expected count for {config_name!r} must be a non-negative integer.")
        specs.append(
            DatasetConfigSpec(
                name=config_name,
                category=category,
                variant=variant if category == "A2" else None,
                expected_count=count,
                allow_any_variant=category == "B",
            )
        )
    return tuple(specs)


class HuggingFaceDatasetSource:
    """Load all v0.6 seed configs from the immutable Hugging Face revision.

    ``loader`` and ``download_config_factory`` are narrowly scoped test seams;
    production callers leave both unset and use ``datasets.load_dataset``.
    """

    source_name = "huggingface"

    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_HF_REPO_ID,
        revision: str = DEFAULT_HF_REVISION,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        offline: bool = False,
        config_specs: Iterable[DatasetConfigSpec] = DATASET_CONFIGS,
        config_map: Mapping[str, str] | None = None,
        expected_counts: Mapping[str, int] | None = None,
        loader: Callable[..., Any] | None = None,
        download_config_factory: Callable[..., Any] | None = None,
        loader_version: str | None = None,
    ) -> None:
        if not repo_id.strip():
            raise ValueError("repo_id must be a non-empty Hugging Face repository ID.")
        if not revision.strip():
            raise ValueError("revision must be a non-empty immutable revision.")

        if config_map is not None or expected_counts is not None:
            if config_map is None or expected_counts is None:
                raise ValueError("config_map and expected_counts must be supplied together.")
            if tuple(config_specs) != DATASET_CONFIGS:
                raise ValueError(
                    "Pass either config_specs or config_map/expected_counts, not both."
                )
            specs = config_specs_from_mapping(config_map, expected_counts)
        else:
            specs = tuple(config_specs)
        if not specs:
            raise ValueError("At least one dataset config is required.")
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError("Dataset config names must be unique.")

        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = Path(cache_dir)
        self.offline = offline
        self.config_specs = specs
        self._loader = loader
        self._download_config_factory = download_config_factory
        self._loader_version = loader_version
        self._records: tuple[dict[str, Any], ...] | None = None
        self._provenance: dict[str, Any] | None = None

    @property
    def is_migration_only(self) -> bool:
        """Keep the default source's status explicit for generic callers."""

        return False

    def preflight(self) -> dict[str, Any]:
        """Fetch/load every config, validate it, and return immutable provenance."""

        if self._provenance is not None:
            return json.loads(json.dumps(self._provenance, ensure_ascii=False))

        root = configure_huggingface_cache(self.cache_dir, offline=self.offline)
        loader, download_config_factory, loader_version = self._resolve_loader()
        records: list[dict[str, Any]] = []
        config_provenance: dict[str, dict[str, Any]] = {}
        seen_ids: set[str] = set()

        for spec in self.config_specs:
            dataset = self._load_config(loader, download_config_factory, spec)
            rows = self._materialize_rows(dataset, spec)
            self._validate_config_rows(rows, spec, seen_ids)
            records.extend(rows)
            config_provenance[spec.name] = {
                "split": "train",
                "record_count": len(rows),
                "dataset_fingerprint": self._dataset_fingerprint(dataset, rows),
            }

        if len(records) != sum(spec.expected_count for spec in self.config_specs):
            # This should be impossible after per-config validation, but it
            # protects customized config_specs from a future accidental merge.
            raise DatasetValidationError(
                "Dataset total record count does not match its configured count matrix."
            )

        records.sort(key=lambda record: record["task_id"])
        time_band_values = sorted({record["time_band"] for record in records})
        aggregate_fingerprint = _json_hash(
            {name: details["dataset_fingerprint"] for name, details in config_provenance.items()}
        )
        self._records = tuple(records)
        self._provenance = {
            "source": self.source_name,
            "repo_id": self.repo_id,
            "release_label": HF_RELEASE_LABEL,
            "dataset_revision": self.revision,
            "configs": config_provenance,
            "dataset_fingerprint": aggregate_fingerprint,
            "record_count": len(records),
            "time_band_values": time_band_values,
            "loader_version": loader_version,
            "cache_dir": str(root),
            "offline": self.offline,
        }
        return self.provenance()

    def provenance(self) -> dict[str, Any]:
        """Return a detached, JSON-safe input provenance record."""

        if self._provenance is None:
            self.preflight()
        assert self._provenance is not None
        return json.loads(json.dumps(self._provenance, ensure_ascii=False))

    def load_records(
        self,
        *,
        categories: Iterable[str] | None = None,
        status: str | None = "ready",
    ) -> list[dict[str, Any]]:
        """Return stable-ID-sorted records without changing their evaluation data."""

        if self._records is None:
            self.preflight()
        assert self._records is not None

        requested = {selector.upper() for selector in categories} if categories else None
        selected: list[dict[str, Any]] = []
        for record in self._records:
            if status is not None and record["status"] != status:
                continue
            if requested is not None and not self._matches_categories(record, requested):
                continue
            # Dataset features may decode date/timestamp fields to Python
            # ``datetime`` objects.  A JSON round trip would reject those and
            # turn a valid Hub load into a false failure; deepcopy still keeps
            # callers from mutating the cached record graph.
            selected.append(deepcopy(record))
        return selected

    def sync(self) -> dict[str, Any]:
        """Explicit data-sync entry point used by the CLI prefetch command."""

        return self.preflight()

    def _resolve_loader(self) -> tuple[Callable[..., Any], Callable[..., Any] | None, str]:
        if self._loader is not None:
            return (
                self._loader,
                self._download_config_factory,
                self._loader_version or "injected-test-loader",
            )
        try:
            # Import only after configure_huggingface_cache() has set both cache
            # variables and, for offline runs, both offline switches.
            import datasets
            from datasets import DownloadConfig, load_dataset
        except ImportError as exc:
            raise DatasetSourceError(
                "Hugging Face dataset support is unavailable; install the 'datasets' dependency."
            ) from exc
        return load_dataset, DownloadConfig, str(datasets.__version__)

    def _load_config(
        self,
        loader: Callable[..., Any],
        download_config_factory: Callable[..., Any] | None,
        spec: DatasetConfigSpec,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "name": spec.name,
            "split": "train",
            "revision": self.revision,
            "cache_dir": str(self.cache_dir.expanduser().resolve() / "datasets"),
        }
        if self.offline:
            if download_config_factory is None:
                raise DatasetSourceError(
                    "Offline loading requires datasets.DownloadConfig local_files_only support."
                )
            kwargs["download_config"] = download_config_factory(local_files_only=True)
        try:
            # ``streaming`` is deliberately omitted (the datasets default is
            # non-streaming); passing false explicitly keeps the contract clear.
            return loader(self.repo_id, streaming=False, **kwargs)
        except Exception as exc:  # datasets error types vary across releases.
            if self.offline:
                raise OfflineDatasetUnavailableError(
                    "Offline Hub dataset load failed for "
                    f"{self.repo_id}@{self.revision} config {spec.name!r}; "
                    "prefetch with data-sync before using offline mode."
                ) from exc
            raise DatasetSourceError(
                "Hugging Face dataset load failed for "
                f"{self.repo_id}@{self.revision} config {spec.name!r}."
            ) from exc

    @staticmethod
    def _materialize_rows(dataset: Any, spec: DatasetConfigSpec) -> list[dict[str, Any]]:
        try:
            raw_rows = list(dataset)
        except TypeError as exc:
            raise DatasetValidationError(
                f"Hub config {spec.name!r} did not return a non-streaming iterable dataset."
            ) from exc

        rows: list[dict[str, Any]] = []
        for index, raw_row in enumerate(raw_rows):
            if not isinstance(raw_row, Mapping):
                raise DatasetValidationError(
                    f"Hub config {spec.name!r} row {index} is not a JSON object."
                )
            rows.append(dict(raw_row))
        return rows

    @staticmethod
    def _dataset_fingerprint(dataset: Any, rows: list[dict[str, Any]]) -> str:
        fingerprint = getattr(dataset, "_fingerprint", None)
        if isinstance(fingerprint, str) and fingerprint:
            return fingerprint
        return _json_hash(rows)

    @staticmethod
    def _matches_categories(record: Mapping[str, Any], requested: set[str]) -> bool:
        category = str(record["category"]).upper()
        variant = record.get("variant")
        if category in requested:
            return True
        if isinstance(variant, str) and f"{category}-{variant.upper()}" in requested:
            return True
        return False

    @staticmethod
    def _validate_config_rows(
        rows: list[dict[str, Any]], spec: DatasetConfigSpec, seen_ids: set[str]
    ) -> None:
        if len(rows) != spec.expected_count:
            raise DatasetValidationError(
                f"Hub config {spec.name!r} expected {spec.expected_count} records, "
                f"found {len(rows)}."
            )

        for index, row in enumerate(rows):
            missing = sorted(_REQUIRED_FIELDS.difference(row))
            if missing:
                raise DatasetValidationError(
                    f"Hub config {spec.name!r} row {index} is missing required field(s): "
                    + ", ".join(missing)
                )

            task_id = row["task_id"]
            if not isinstance(task_id, str) or not task_id.strip():
                raise DatasetValidationError(
                    f"Hub config {spec.name!r} row {index} has an invalid task_id."
                )
            if task_id in seen_ids:
                raise DatasetValidationError(f"Hub dataset contains duplicate task_id {task_id!r}.")
            seen_ids.add(task_id)

            if row["category"] != spec.category:
                raise DatasetValidationError(
                    f"Hub config {spec.name!r} row {task_id!r} has category "
                    f"{row['category']!r}, expected {spec.category!r}."
                )
            if spec.allow_any_variant:
                if not isinstance(row["variant"], str) or not row["variant"].strip():
                    raise DatasetValidationError(
                        f"Hub config {spec.name!r} row {task_id!r} must declare "
                        "a non-empty variant."
                    )
            elif row["variant"] != spec.variant:
                raise DatasetValidationError(
                    f"Hub config {spec.name!r} row {task_id!r} has variant "
                    f"{row['variant']!r}, expected {spec.variant!r}."
                )
            if row["status"] != "ready":
                raise DatasetValidationError(
                    f"Hub config {spec.name!r} row {task_id!r} has status "
                    f"{row['status']!r}, expected 'ready'."
                )
            time_band = row["time_band"]
            if not isinstance(time_band, str) or not time_band.strip():
                raise DatasetValidationError(
                    f"Hub config {spec.name!r} row {task_id!r} has a missing or empty time_band."
                )
            if not isinstance(row["prompt"], str) or not row["prompt"].strip():
                raise DatasetValidationError(
                    f"Hub config {spec.name!r} row {task_id!r} has an invalid prompt."
                )
            for field in ("seed", "expected_output", "ground_truth", "metadata"):
                if not isinstance(row[field], Mapping):
                    raise DatasetValidationError(
                        f"Hub config {spec.name!r} row {task_id!r} field "
                        f"{field!r} must be an object."
                    )
