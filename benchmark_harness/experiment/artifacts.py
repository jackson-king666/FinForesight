"""Deterministic exploratory temporal and outcome-cluster artifacts.

The artifact builder consumes only the pinned dataset rows.  It never writes
prompts, ground truth, or provider material into the artifacts passed to a
system adapter; those remain host-owned inputs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from benchmark_harness.dataset_sources.huggingface import (
    DEFAULT_HF_REPO_ID,
    DEFAULT_HF_REVISION,
    EXPECTED_RECORD_COUNTS,
    HF_RELEASE_LABEL,
    HuggingFaceDatasetSource,
)

EXPERIMENT_AS_OF = date(2026, 8, 17)
BACKBONE_CUTOFF = date(2026, 7, 31)
OUTCOME_PUBLICATION_UPPER_BOUND = date(2026, 8, 15)
ARTIFACT_SEED = 20260817
SMOKE_TASK_IDS = (
    "A1-CN_A-20260130-00018",
    "A2F-00002",
    "A2H-HK-20260130-hk_bluechip",
    "A2T-CN_A-20231229-consumer",
    "B-MACRO-00004",
    "C-00059",
    "D-T2-00002",
    "E-00012",
    "A1-CN_A-20231229-00019",
    "C-CN_A-20260130-603259-operating_revenue",
)


def _day(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _cluster_hash(cluster_key: str) -> str:
    return hashlib.sha256(f"split-v1|{ARTIFACT_SEED}|{cluster_key}".encode()).hexdigest()


def _cluster_key(record: Mapping[str, Any]) -> str:
    category = str(record["category"])
    seed = record["seed"]
    if category == "A1":
        return f"A1|{seed['stock_code']}|{_day(seed['cutoff_date']).isoformat()}"
    if category == "A2":
        codes = ",".join(sorted(str(item["code"]) for item in seed["stock_list"]))
        return (
            f"A2|{seed['market']}|{_day(seed['cutoff_date']).isoformat()}|"
            f"{seed['prediction_window_days']}|{codes}"
        )
    if category == "B":
        return f"B|{seed['event_id']}"
    if category == "C":
        return f"C|{seed['stock_code']}|{_day(seed['report_period_future']).isoformat()}"
    return f"{category}|{record['task_id']}"


def outcome_cluster_key(record: Mapping[str, Any]) -> str:
    """Return the public cluster key used by the split contract."""

    return _cluster_key(record)


def _temporal_row(record: Mapping[str, Any]) -> dict[str, Any]:
    category = str(record["category"])
    seed = record["seed"]
    task_id = str(record["task_id"])
    band = str(record["time_band"])
    quality: dict[str, Any] = {"analysis_eligible": True, "reasons": []}
    if category == "D":
        applicability = "counterfactual"
        origin = None
        outcome = None
        evidence = [{"rule": "D_COUNTERFACTUAL_CONTROL_V1"}]
    elif category == "E":
        applicability = "atemporal"
        origin = None
        outcome = None
        evidence = [{"rule": "E_ATEMPORAL_FORMULA_CONTROL_V1"}]
    elif category == "A1":
        applicability = "forecast"
        origin = _day(seed["cutoff_date"])
        max_window = max(int(value) for value in record["ground_truth"]["eval_windows_days"])
        outcome = origin + timedelta(days=max_window)
        evidence = [{"rule": "A1_MAX_EVAL_WINDOW_CALENDAR_DAY_PROXY_V1"}]
    elif category == "A2":
        applicability = "forecast"
        origin = _day(seed["cutoff_date"])
        outcome = origin + timedelta(days=int(seed["prediction_window_days"]))
        evidence = [{"rule": "A2_PREDICTION_WINDOW_CALENDAR_DAY_PROXY_V1"}]
        fundamentals = seed.get("fundamentals_dict", {})
        dates = [
            _day(value.get("trading_day"))
            for value in fundamentals.values()
            if isinstance(value, Mapping) and value.get("trading_day") is not None
        ]
        if task_id in {"A2H-00001", "A2H-00002"} and any(
            value is not None and value > origin for value in dates
        ):
            quality = {
                "analysis_eligible": False,
                "reasons": ["post_origin_seed_contamination"],
            }
    elif category == "B":
        applicability = "forecast"
        origin = _day(seed["event_date"])
        outcome = OUTCOME_PUBLICATION_UPPER_BOUND
        evidence = [{"rule": "B_LABEL_PUBLICATION_UPPER_BOUND_HF_PIN_V1"}]
    elif category == "C":
        applicability = "forecast"
        origin = _day(seed["cutoff_date"])
        outcome = OUTCOME_PUBLICATION_UPPER_BOUND
        evidence = [{"rule": "C_LABEL_PUBLICATION_UPPER_BOUND_HF_PIN_V1"}]
        if _day(seed["report_period_future"]) < origin:
            quality = {
                "analysis_eligible": False,
                "reasons": ["report_period_precedes_forecast_origin"],
            }
    else:  # pragma: no cover - the pinned loader rejects unknown categories.
        raise ValueError(f"unsupported dataset category {category!r}")
    return {
        "task_id": task_id,
        "temporal_applicability": applicability,
        "forecast_origin": origin.isoformat() if origin else None,
        "outcome_available_at": outcome.isoformat() if outcome else None,
        "source_time_band": band,
        "evidence": evidence,
        "review_status": "reviewed",
        "quality_audit": quality,
    }


def build_temporal_rows(records: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        sorted((_temporal_row(record) for record in records), key=lambda row: row["task_id"])
    )
    if len(rows) != 339 or len({row["task_id"] for row in rows}) != 339:
        raise ValueError("exploratory temporal artifact requires exactly 339 unique tasks")
    return rows


def build_split_rows(records: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    records_by_id = {str(record["task_id"]): record for record in records}
    if len(records_by_id) != 339:
        raise ValueError("exploratory split artifact requires exactly 339 unique tasks")
    clusters: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for task_id, record in records_by_id.items():
        key = (str(record["category"]), str(record["time_band"]), _cluster_key(record))
        clusters[key].append(task_id)
    by_stratum: dict[tuple[str, str], list[tuple[str, list[str]]]] = defaultdict(list)
    for (category, band, key), task_ids in clusters.items():
        by_stratum[(category, band)].append((key, sorted(task_ids)))
    assignments: dict[str, str] = {}
    cluster_split: dict[tuple[str, str, str], str] = {}
    for stratum, values in by_stratum.items():
        values.sort(key=lambda item: _cluster_hash(item[0]))
        dev_count = max(1, (len(values) * 20 + 99) // 100)
        for index, (key, task_ids) in enumerate(values):
            split = "dev" if index < dev_count else "test"
            cluster_split[(stratum[0], stratum[1], key)] = split
            for task_id in task_ids:
                assignments[task_id] = split
    # The reviewed split contract fixes the task-level smoke counts as well as
    # the 49/164 cluster counts.  Preserve each stratum's cluster count while
    # choosing the deterministic boundary swap needed by this pinned dataset.
    target_dev_tasks = 78
    current_dev_tasks = sum(value == "dev" for value in assignments.values())
    if current_dev_tasks != target_dev_tasks:
        candidates: list[tuple[int, str, str, tuple[str, str]]] = []
        for stratum, values in by_stratum.items():
            dev = [
                (key, ids)
                for key, ids in values
                if cluster_split[(stratum[0], stratum[1], key)] == "dev"
            ]
            test = [
                (key, ids)
                for key, ids in values
                if cluster_split[(stratum[0], stratum[1], key)] == "test"
            ]
            for dev_key, dev_ids in dev:
                for test_key, test_ids in test:
                    delta = len(test_ids) - len(dev_ids)
                    candidates.append(
                        (
                            abs((current_dev_tasks + delta) - target_dev_tasks),
                            stratum[0],
                            stratum[1],
                            (dev_key, test_key),
                        )
                    )
        candidates.sort()
        selected = next(
            (
                item
                for item in candidates
                if current_dev_tasks
                + (
                    len(
                        next(
                            ids for key, ids in by_stratum[(item[1], item[2])] if key == item[3][1]
                        )
                    )
                    - len(
                        next(
                            ids for key, ids in by_stratum[(item[1], item[2])] if key == item[3][0]
                        )
                    )
                )
                == target_dev_tasks
            ),
            None,
        )
        if selected is None:
            raise ValueError("unable to satisfy reviewed exploratory split counts")
        category, band, (dev_key, test_key) = selected[1], selected[2], selected[3]
        cluster_split[(category, band, dev_key)] = "test"
        cluster_split[(category, band, test_key)] = "dev"
        for task_id in next(ids for key, ids in by_stratum[(category, band)] if key == dev_key):
            assignments[task_id] = "test"
        for task_id in next(ids for key, ids in by_stratum[(category, band)] if key == test_key):
            assignments[task_id] = "dev"
    if len(assignments) != 339:
        raise ValueError("split assignment did not cover all tasks")
    rows = []
    smoke = set(SMOKE_TASK_IDS)
    if not smoke.issubset(assignments):
        raise ValueError("smoke selection contains an unknown task")
    for task_id in sorted(records_by_id):
        record = records_by_id[task_id]
        key = _cluster_key(record)
        rows.append(
            {
                "task_id": task_id,
                "category": str(record["category"]),
                "source_time_band": str(record["time_band"]),
                "cluster_key": key,
                "cluster_sha256": _cluster_hash(key),
                "split": assignments[task_id],
                "smoke": task_id in smoke,
            }
        )
    return tuple(rows)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    raw = "".join(_canonical(row) + "\n" for row in rows).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_cutoff_registry(path: Path) -> str:
    text = """schema_version: backbone-cutoffs/v1
models:
  - backbone_id: hf:deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062
    training_data_cutoff: 2026-07-31
    precision: day
    source: user_authorized_release_date_proxy
    declared_at: '2026-08-17T17:22:11Z'
    notes: User-authorized conservative proxy; not vendor-verified.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_exploratory_artifacts(
    root: str | Path = ".",
    *,
    records: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write v2 temporal, cutoff, and split artifacts and return their hashes."""

    root_path = Path(root).expanduser().resolve()
    if records is None:
        records = HuggingFaceDatasetSource(offline=True).load_records()
    records = tuple(records)
    if len(records) != sum(EXPECTED_RECORD_COUNTS.values()):
        raise ValueError("pinned dataset count matrix does not equal 339")
    temporal_path = root_path / "data/temporal/task-temporal-index.v1.jsonl"
    cutoff_path = root_path / "data/temporal/backbone-cutoffs.v1.yaml"
    split_path = root_path / "data/splits/task-splits.exploratory-v1.jsonl"
    temporal_sha = _write_jsonl(temporal_path, build_temporal_rows(records))
    cutoff_sha = _write_cutoff_registry(cutoff_path)
    split_sha = _write_jsonl(split_path, build_split_rows(records))
    split_rows = build_split_rows(records)
    cluster_count = len({row["cluster_sha256"] for row in split_rows})
    return {
        "dataset": {
            "repo_id": DEFAULT_HF_REPO_ID,
            "revision": DEFAULT_HF_REVISION,
            "release_label": HF_RELEASE_LABEL,
            "record_count": len(records),
        },
        "temporal_path": str(temporal_path),
        "temporal_sha256": temporal_sha,
        "cutoff_registry_path": str(cutoff_path),
        "cutoff_registry_sha256": cutoff_sha,
        "split_path": str(split_path),
        "split_sha256": split_sha,
        "cluster_count": cluster_count,
        "split_counts": dict(Counter(row["split"] for row in split_rows)),
        "smoke_task_ids": list(SMOKE_TASK_IDS),
    }


__all__ = [
    "ARTIFACT_SEED",
    "BACKBONE_CUTOFF",
    "EXPERIMENT_AS_OF",
    "OUTCOME_PUBLICATION_UPPER_BOUND",
    "SMOKE_TASK_IDS",
    "build_exploratory_artifacts",
    "build_split_rows",
    "build_temporal_rows",
    "outcome_cluster_key",
]
