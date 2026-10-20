"""Unit coverage for the Hub-first, revision-pinned dataset source."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from benchmark_harness.dataset_sources.huggingface import (
    DATASET_CONFIGS,
    DEFAULT_HF_REPO_ID,
    DEFAULT_HF_REVISION,
    DatasetSourceError,
    DatasetValidationError,
    HuggingFaceDatasetSource,
    OfflineDatasetUnavailableError,
    config_specs_from_mapping,
)


class FakeDataset(list[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]], fingerprint: str) -> None:
        super().__init__(rows)
        self._fingerprint = fingerprint


@dataclass(frozen=True)
class FakeDownloadConfig:
    local_files_only: bool


def make_row(
    *,
    task_id: str,
    category: str,
    variant: str | None,
    status: str = "ready",
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "category": category,
        "variant": variant,
        "time_band": "T2",
        "status": status,
        "seed": {"cutoff_date": "2025-06-06"},
        "prompt": f"Prompt for {task_id}",
        "expected_output": {"answer": "str"},
        "ground_truth": {"answer": "ok"},
        "metadata": {"source": "fixture"},
    }


def make_hub_rows() -> dict[str, FakeDataset]:
    rows: dict[str, FakeDataset] = {}
    for spec in DATASET_CONFIGS:
        variant = "earnings" if spec.allow_any_variant else spec.variant
        config_rows = [
            make_row(
                task_id=f"{spec.name.upper()}-{index:05d}",
                category=spec.category,
                variant=variant,
            )
            for index in range(spec.expected_count, 0, -1)
        ]
        rows[spec.name] = FakeDataset(config_rows, fingerprint=f"fingerprint-{spec.name}")
    return rows


def test_hub_source_loads_the_pinned_full_matrix_and_sorts_task_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets = make_hub_rows()
    calls: list[tuple[str, dict[str, Any]]] = []

    def loader(repo_id: str, **kwargs: Any) -> FakeDataset:
        calls.append((repo_id, kwargs))
        return datasets[kwargs["name"]]

    monkeypatch.delenv("HF_HOME", raising=False)
    source = HuggingFaceDatasetSource(
        cache_dir=tmp_path / "data" / ".huggingface",
        loader=loader,
        loader_version="fake-datasets-1.0",
    )

    provenance = source.preflight()
    records = source.load_records()

    assert len(records) == 339
    assert [record["task_id"] for record in records] == sorted(
        record["task_id"] for record in records
    )
    assert [kwargs["name"] for _, kwargs in calls] == [spec.name for spec in DATASET_CONFIGS]
    assert all(repo_id == DEFAULT_HF_REPO_ID for repo_id, _ in calls)
    assert all(kwargs["split"] == "train" for _, kwargs in calls)
    assert all(kwargs["revision"] == DEFAULT_HF_REVISION for _, kwargs in calls)
    assert all(kwargs["streaming"] is False for _, kwargs in calls)
    assert all("download_config" not in kwargs for _, kwargs in calls)
    assert provenance["record_count"] == 339
    assert provenance["configs"]["b"]["record_count"] == 33
    assert provenance["configs"]["c"]["dataset_fingerprint"] == "fingerprint-c"
    assert provenance["time_band_values"] == ["T2"]
    assert provenance["loader_version"] == "fake-datasets-1.0"
    assert Path(os.environ["HF_HUB_CACHE"]) == tmp_path / "data" / ".huggingface" / "hub"
    assert Path(os.environ["HF_DATASETS_CACHE"]) == (
        tmp_path / "data" / ".huggingface" / "datasets"
    )
    assert "HF_HOME" not in os.environ


def test_hub_source_filters_a2_and_exact_variant_without_mutating_cache(tmp_path: Path) -> None:
    datasets = make_hub_rows()

    def loader(_repo_id: str, **kwargs: Any) -> FakeDataset:
        return datasets[kwargs["name"]]

    source = HuggingFaceDatasetSource(cache_dir=tmp_path / ".huggingface", loader=loader)
    all_a2 = source.load_records(categories=["A2"])
    a2_f = source.load_records(categories=["A2-F"])
    a2_f[0]["seed"]["changed"] = True

    assert len(all_a2) == 30
    assert {record["variant"] for record in all_a2} == {"F", "T", "H"}
    assert len(a2_f) == 10
    assert all(record["variant"] == "F" for record in a2_f)
    assert "changed" not in source.load_records(categories=["A2-F"])[0]["seed"]


def test_hub_source_returns_decoded_datetime_features_without_json_roundtrip_failure(
    tmp_path: Path,
) -> None:
    datasets = make_hub_rows()
    decoded_cutoff = datetime(2025, 6, 6)
    datasets["a1"][0]["seed"]["cutoff_date"] = decoded_cutoff

    def loader(_repo_id: str, **kwargs: Any) -> FakeDataset:
        return datasets[kwargs["name"]]

    records = HuggingFaceDatasetSource(
        cache_dir=tmp_path / ".huggingface", loader=loader
    ).load_records(categories=["A1"])

    assert any(record["seed"]["cutoff_date"] == decoded_cutoff for record in records)


def test_offline_load_sets_both_flags_uses_local_files_only_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def loader(_repo_id: str, **kwargs: Any) -> FakeDataset:
        calls.append(kwargs)
        raise FileNotFoundError("not in local cache")

    source = HuggingFaceDatasetSource(
        cache_dir=tmp_path / ".huggingface",
        offline=True,
        loader=loader,
        download_config_factory=FakeDownloadConfig,
    )

    with pytest.raises(OfflineDatasetUnavailableError, match="prefetch with data-sync"):
        source.preflight()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert calls[0]["download_config"] == FakeDownloadConfig(local_files_only=True)
    assert calls[0]["name"] == "a1"
    assert calls[0]["streaming"] is False


def test_hub_failure_never_falls_back_to_an_external_checkout(tmp_path: Path) -> None:
    def loader(_repo_id: str, **_kwargs: Any) -> FakeDataset:
        raise RuntimeError("Hub unavailable")

    source = HuggingFaceDatasetSource(cache_dir=tmp_path / ".huggingface", loader=loader)

    with pytest.raises(DatasetSourceError, match="Hugging Face dataset load failed"):
        source.preflight()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda rows: rows["a2_f"][0].__setitem__("variant", "T"), "variant"),
        (lambda rows: rows["d"][0].__setitem__("status", "template"), "status"),
        (lambda rows: rows["a1"][0].__setitem__("time_band", ""), "missing or empty time_band"),
        (lambda rows: rows["a1"][0].pop("time_band"), "time_band"),
        (lambda rows: rows["a1"].pop(), "expected 72 records"),
    ],
)
def test_hub_source_rejects_schema_variant_status_and_count_drift(
    tmp_path: Path, change: Any, message: str
) -> None:
    datasets = make_hub_rows()
    change(datasets)

    def loader(_repo_id: str, **kwargs: Any) -> FakeDataset:
        return datasets[kwargs["name"]]

    source = HuggingFaceDatasetSource(cache_dir=tmp_path / ".huggingface", loader=loader)

    with pytest.raises(DatasetValidationError, match=message):
        source.preflight()


def test_hub_source_rejects_duplicate_task_ids_across_configs(tmp_path: Path) -> None:
    datasets = make_hub_rows()
    datasets["b"][0]["task_id"] = datasets["a1"][0]["task_id"]

    def loader(_repo_id: str, **kwargs: Any) -> FakeDataset:
        return datasets[kwargs["name"]]

    with pytest.raises(DatasetValidationError, match="duplicate task_id"):
        HuggingFaceDatasetSource(cache_dir=tmp_path / ".huggingface", loader=loader).preflight()


def test_config_mapping_builds_variant_aware_loader_specs() -> None:
    specs = config_specs_from_mapping(
        {"A1": "one", "A2-F": "two", "B": "three"},
        {"one": 1, "two": 2, "three": 3},
    )

    assert [(spec.name, spec.category, spec.variant, spec.allow_any_variant) for spec in specs] == [
        ("one", "A1", None, False),
        ("two", "A2", "F", False),
        ("three", "B", None, True),
    ]
