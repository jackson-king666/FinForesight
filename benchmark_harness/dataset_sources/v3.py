"""Variable-count, revision-pinned Hub source for the score-blind v3 audit."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_harness.dataset_sources.huggingface import configure_huggingface_cache
from benchmark_harness.experiment.v3 import (
    V3_DATASET_MANIFEST_SHA256,
    V3_DATASET_REPO_ID,
    V3_DATASET_REVISION,
    V3_DATASET_TREE_SHA,
    V3_EXPECTED_RECORD_COUNT,
    assert_score_blind,
)

DEFAULT_V3_CONFIG_NAMES = ("a1", "a2_f", "a2_t", "a2_h", "b", "c", "d", "e")


class V3DatasetSourceError(RuntimeError):
    """Raised when the variable-count v3 Hub source is unavailable or invalid."""


class V3OfflineDatasetUnavailableError(V3DatasetSourceError):
    """Raised when an offline v3 cache does not contain a requested config."""


@dataclass(frozen=True, slots=True)
class V3DatasetSource:
    """Load a variable-count v3 matrix without changing the v0.6 source."""

    cache_dir: Path = Path("data/.huggingface")
    repo_id: str = V3_DATASET_REPO_ID
    revision: str = V3_DATASET_REVISION
    config_names: tuple[str, ...] = DEFAULT_V3_CONFIG_NAMES
    expected_record_count: int = V3_EXPECTED_RECORD_COUNT
    offline: bool = True
    loader: Callable[..., Any] | None = None
    download_config_factory: Callable[..., Any] | None = None
    loader_version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.repo_id, str) or not self.repo_id.strip():
            raise V3DatasetSourceError("v3 Hub repo_id and revision must be non-empty")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise V3DatasetSourceError("v3 Hub repo_id and revision must be non-empty")
        if not self.config_names or any(
            not isinstance(name, str) or not name.strip() for name in self.config_names
        ):
            raise V3DatasetSourceError("v3 config_names must contain non-empty strings")
        if len(set(self.config_names)) != len(self.config_names):
            raise V3DatasetSourceError("v3 config_names must be unique")
        if not isinstance(self.expected_record_count, int) or self.expected_record_count < 1:
            raise V3DatasetSourceError("v3 expected_record_count must be positive")

    def _resolve_loader(self) -> tuple[Callable[..., Any], Callable[..., Any] | None, str]:
        if self.loader is not None:
            return (
                self.loader,
                self.download_config_factory,
                self.loader_version or "injected-test-loader",
            )
        try:
            import datasets
            from datasets import DownloadConfig, load_dataset
        except ImportError as exc:
            raise V3DatasetSourceError("Hugging Face datasets support is unavailable") from exc
        return load_dataset, DownloadConfig, str(datasets.__version__)

    def _load_config(
        self,
        loader: Callable[..., Any],
        download_config_factory: Callable[..., Any] | None,
        config_name: str,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "name": config_name,
            "split": "train",
            "revision": self.revision,
            "cache_dir": str(self.cache_dir.expanduser().resolve() / "datasets"),
        }
        if self.offline:
            if download_config_factory is None:
                raise V3DatasetSourceError(
                    "offline v3 loading requires DownloadConfig local_files_only support"
                )
            kwargs["download_config"] = download_config_factory(local_files_only=True)
        try:
            dataset = loader(self.repo_id, streaming=False, **kwargs)
            rows = list(dataset)
        except Exception as exc:  # dataset-builder exception classes vary by release.
            if self.offline:
                raise V3OfflineDatasetUnavailableError(
                    f"offline v3 Hub load failed for {self.repo_id}@{self.revision} "
                    f"config {config_name!r}"
                ) from exc
            raise V3DatasetSourceError(
                f"v3 Hub load failed for {self.repo_id}@{self.revision} config {config_name!r}"
            ) from exc
        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise V3DatasetSourceError(
                    f"v3 config {config_name!r} row {index} is not an object"
                )
            assert_score_blind(row, path=f"{config_name}[{index}]")
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise V3DatasetSourceError(
                    f"v3 config {config_name!r} row {index} has an invalid task_id"
                )
            normalized.append(dict(row))
        return normalized

    def preflight(self) -> dict[str, Any]:
        """Load all declared configs and return score-blind provenance."""

        root = configure_huggingface_cache(self.cache_dir, offline=self.offline)
        loader, download_config_factory, loader_version = self._resolve_loader()
        records: list[dict[str, Any]] = []
        config_counts: dict[str, int] = {}
        for config_name in self.config_names:
            rows = self._load_config(loader, download_config_factory, config_name)
            records.extend(rows)
            config_counts[config_name] = len(rows)
        task_ids = [str(row["task_id"]) for row in records]
        if len(records) != self.expected_record_count:
            raise V3DatasetSourceError(
                "v3 record count mismatch: "
                f"expected {self.expected_record_count}, found {len(records)}"
            )
        if len(task_ids) != len(set(task_ids)):
            raise V3DatasetSourceError("v3 dataset contains duplicate task_id values")
        return {
            "source": "huggingface-v3",
            "repo_id": self.repo_id,
            "dataset_revision": self.revision,
            "release_label": "v0.9",
            "tree_sha": V3_DATASET_TREE_SHA,
            "source_manifest_sha256": V3_DATASET_MANIFEST_SHA256,
            "config_names": list(self.config_names),
            "config_counts": dict(sorted(config_counts.items())),
            "record_count": len(records),
            "unique_task_count": len(set(task_ids)),
            "loader_version": loader_version,
            "cache_dir": str(root),
            "offline": self.offline,
        }

    def load_records(
        self,
        *,
        categories: Iterable[str] | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Load and return stable-ID-sorted metadata without evaluator fields."""

        loader, download_config_factory, _ = self._resolve_loader()
        records = [
            row
            for config_name in self.config_names
            for row in self._load_config(loader, download_config_factory, config_name)
        ]
        requested = {str(value).upper() for value in categories} if categories else None
        selected = []
        for row in sorted(records, key=lambda item: item["task_id"]):
            if status is not None and row.get("status") != status:
                continue
            if requested is not None:
                category = str(row.get("category", "")).upper()
                variant = row.get("variant")
                config_key = category
                if category == "A2" and isinstance(variant, str) and variant.strip():
                    config_key = f"A2-{variant.upper()}"
                elif category == "B":
                    subtype = row.get("event_subtype", row.get("subtype", variant))
                    if isinstance(subtype, str) and subtype.strip():
                        config_key = f"B-{subtype.upper()}"
                if category not in requested and config_key not in requested:
                    continue
            selected.append(deepcopy(row))
        return selected

    def provenance(self) -> dict[str, Any]:
        return self.preflight()

    def sync(self) -> dict[str, Any]:
        return self.preflight()


__all__ = [
    "DEFAULT_V3_CONFIG_NAMES",
    "V3DatasetSource",
    "V3DatasetSourceError",
    "V3OfflineDatasetUnavailableError",
]
