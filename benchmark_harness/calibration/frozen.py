"""Authorized 400-item leakage-calibration collection and Labeler-A pass.

The frozen cohort is a new calibration version after the v1 pilot. It uses the
same provider response and deterministic cutoff filter as the pilot, but keeps
the calibration-only unrestricted collection in its own root and cache. No
paper experiment, score, answer, or ground truth is read here. The v2
annotation view may carry the bounded task prompt needed for its separate
relevance pass, but never the expected output or ground truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter
from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

from benchmark_harness.calibration.pilot import (
    PILOT_FILTER_POLICY_SHA256,
    PILOT_QUERY_CLASSES,
    PILOT_STRATA,
    PilotCollectionError,
    _annotation_views,
    _build_query,
    _cache_entry,
    _canonical_url,
    _date_text,
    _filter_state,
    _label_pilot_blind,
    _load_task_prompts,
    _load_tasks,
    _now_utc,
    _provisional_item_id,
    _raw_provider_time,
    _read_jsonl,
    _sampling_key,
    _sha256_canonical,
    _sha256_file,
    _unrestricted_time_range,
    _write_cache_json,
    _write_jsonl,
)
from benchmark_harness.calibration.validator import (
    CALIBRATION_SCHEMA_VERSION,
    EXPECTED_TRACKED_FILES,
    FROZEN_CALIBRATION_ID,
    HUB_REPO_ID,
    HUB_REVISION,
    JSON_FILES,
    JSONL_FILES,
    canonical_json_bytes,
    validate_calibration_root,
)
from benchmark_harness.contracts import sha256_json, sha256_text
from benchmark_harness.search import (
    DEFAULT_TOOL_CONTRACT,
    CutoffWindow,
    VolcHttpResponse,
    VolcSdkTransport,
    VolcSearchCandidate,
    filter_volc_candidates,
)
from benchmark_harness.search.volc import _volc_web_results

FROZEN_STRATA = PILOT_STRATA
FROZEN_ITEM_COUNT = 400
FROZEN_RAW_LABEL_COUNT = 800
FROZEN_QUERY_CLASSES = PILOT_QUERY_CLASSES
FROZEN_QUERY_TEMPLATE_VERSION = "frozen-query/v3"
FROZEN_COLLECTION_ID = "leakage-probe-v2-frozen400-20260831"
FROZEN_AUTHORIZATION_ID = "leakage-probe-v2-frozen400-20260831"
FROZEN_AUTHORIZATION_SCHEMA = "leakage-calibration-authorization/v1"
FROZEN_CACHE_SCHEMA = "leakage-calibration-cache/v1"
FROZEN_AUTHORIZATION_SCOPE = "calibration_frozen_400_collection_and_labeling"
FROZEN_REVIEWER_ID = "dev-labeler-a-frozen400"
FROZEN_TASK_LIMITS = {
    "A1-T1": 16,
    "B-T1": 17,
    "B-T2": 20,
    "C-T2": 20,
}
FROZEN_STATE_TARGETS = {"keep": 150, "reject": 150, "ambiguous": 100}
FROZEN_STRATUM_TARGETS = {
    "A1-T1": {"keep": 38, "reject": 38, "ambiguous": 25},
    "B-T1": {"keep": 38, "reject": 38, "ambiguous": 25},
    "B-T2": {"keep": 37, "reject": 37, "ambiguous": 25},
    "C-T2": {"keep": 37, "reject": 37, "ambiguous": 25},
}
FROZEN_POLICY = {
    "calibration_id": FROZEN_CALIBRATION_ID,
    "ranking_eligible": False,
    "regime": "unrestricted_current_web",
    "scope": "calibration_frozen_400",
    "schema_version": "leakage_probe/v2",
}
FROZEN_POLICY_SHA256 = sha256_json(FROZEN_POLICY)
FROZEN_SERVICE_COHORT_SHA256 = sha256_json(
    {
        "calibration_id": FROZEN_CALIBRATION_ID,
        "hub_revision": HUB_REVISION,
        "provider_id": "volc",
        "tool_contract_sha256": DEFAULT_TOOL_CONTRACT.hash,
    }
)
FROZEN_TEMPORAL_INDEX_SHA256 = "80d977e40e113172e778b6f78315e1fad132423c9fe9eca3ad6a5de34df4f21b"


class FrozenCalibrationError(PilotCollectionError):
    """Raised when the authorized frozen cohort cannot be completed safely."""


class FrozenCalibrationResult:
    """Safe summary returned after the 400-item collection and A pass."""

    def __init__(
        self,
        *,
        task_count: int,
        query_count: int,
        candidate_pool_count: int,
        item_count: int,
        label_a_count: int,
        calibration_sha256: str,
    ) -> None:
        self.collection_id = FROZEN_COLLECTION_ID
        self.task_count = task_count
        self.query_count = query_count
        self.candidate_pool_count = candidate_pool_count
        self.item_count = item_count
        self.label_a_count = label_a_count
        self.calibration_sha256 = calibration_sha256

    def as_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": FROZEN_CALIBRATION_ID,
            "calibration_sha256": self.calibration_sha256,
            "candidate_pool_count": self.candidate_pool_count,
            "collection_id": self.collection_id,
            "item_count": self.item_count,
            "label_a_count": self.label_a_count,
            "query_count": self.query_count,
            "task_count": self.task_count,
        }


def run_frozen(
    *,
    root: str | Path = "data/calibration/leakage-probe/v2",
    cache_root: str | Path = "data/calibration-cache/leakage-probe/v2",
    dataset_cache: str | Path = "data/.huggingface",
    timeout_seconds: float = 30.0,
) -> FrozenCalibrationResult:
    """Collect and label exactly the authorized 400-item frozen cohort."""

    root_path = Path(root).expanduser().resolve()
    cache_path = Path(cache_root).expanduser().resolve()
    _prepare_frozen_scaffold(root_path)
    try:
        state = validate_calibration_root(root_path)
    except ValueError as exc:
        raise FrozenCalibrationError("frozen calibration scaffold failed validation") from exc
    if state["calibration_id"] != FROZEN_CALIBRATION_ID:
        raise FrozenCalibrationError("unexpected calibration version")
    if state["calibration_status"] != "not_started" or state["counts"]["items"]:
        raise FrozenCalibrationError(
            "frozen calibration root is already populated; create a new version"
        )
    if timeout_seconds <= 0:
        raise FrozenCalibrationError("frozen calibration timeout must be positive")

    cache_path.mkdir(parents=True, exist_ok=True)
    _private_directory(cache_path)
    candidate_pool, query_rows, task_count, evidence_entries = _collect_frozen_candidates(
        cache_root=cache_path,
        dataset_cache=dataset_cache,
        timeout_seconds=timeout_seconds,
    )
    selected = _select_frozen_items(
        candidate_pool,
        excluded_item_ids=_load_pilot_item_ids(root_path),
    )
    item_rows, filter_rows = _materialize_frozen_items(
        selected, cache_root=cache_path, evidence_entries=evidence_entries
    )
    _write_jsonl(root_path / "query-corpus.jsonl", query_rows)
    _write_jsonl(root_path / "items.jsonl", item_rows)
    _write_jsonl(root_path / "filter-decisions.jsonl", filter_rows)
    task_prompts = _load_task_prompts(dataset_cache, item_rows)
    _write_jsonl(
        root_path / "annotation-view.jsonl",
        _annotation_views(item_rows, cache_path, task_prompts=task_prompts),
    )
    evidence_sha = _write_frozen_evidence_manifest(cache_path, evidence_entries)
    _refresh_frozen_state(
        root_path, status="incomplete", restricted_evidence_manifest_sha256=evidence_sha
    )
    _validate_frozen_state(root_path, expected_labels=0)

    label_count = _label_pilot_blind(root_path, reviewer_id=FROZEN_REVIEWER_ID)
    _refresh_frozen_state(
        root_path, status="incomplete", restricted_evidence_manifest_sha256=evidence_sha
    )
    validated = validate_calibration_root(root_path)
    if validated["counts"]["items"] != FROZEN_ITEM_COUNT or label_count != FROZEN_ITEM_COUNT:
        raise FrozenCalibrationError("frozen corpus or Labeler A output is incomplete")
    return FrozenCalibrationResult(
        task_count=task_count,
        query_count=len(query_rows),
        candidate_pool_count=len(candidate_pool),
        item_count=validated["counts"]["items"],
        label_a_count=validated["counts"]["raw_labels_a"],
        calibration_sha256=validated["calibration_sha256"],
    )


def _prepare_frozen_scaffold(root: Path) -> None:
    if root.exists():
        return
    source_root = root.parent / "v1"
    if not source_root.is_dir():
        raise FrozenCalibrationError("v1 calibration scaffold is unavailable")
    root.mkdir(parents=True, exist_ok=False)
    schema_paths = sorted(path for path in EXPECTED_TRACKED_FILES if path.startswith("schemas/"))
    for relative in ("rubric.md", *schema_paths):
        source = source_root / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    auth_schema_path = root / "schemas/authorization.schema.json"
    auth_schema = json.loads(auth_schema_path.read_text(encoding="utf-8"))
    auth_schema["title"] = "Explicit frozen 400-item calibration authorization"
    auth_schema["properties"]["calibration_id"] = {"const": FROZEN_CALIBRATION_ID}
    auth_schema["properties"]["scope"] = {"const": FROZEN_AUTHORIZATION_SCOPE}
    auth_schema["properties"]["frozen_item_count"] = {"const": FROZEN_ITEM_COUNT}
    auth_schema["properties"]["frozen_strata"] = {
        "additionalProperties": False,
        "properties": {
            stratum: {
                "additionalProperties": False,
                "properties": {
                    "ambiguous": {"const": counts["ambiguous"]},
                    "keep": {"const": counts["keep"]},
                    "reject": {"const": counts["reject"]},
                },
                "required": ["ambiguous", "keep", "reject"],
                "type": "object",
            }
            for stratum, counts in FROZEN_STRATUM_TARGETS.items()
        },
        "required": list(FROZEN_STRATA),
        "type": "object",
    }
    auth_schema["required"] = [
        *auth_schema["required"],
        "frozen_item_count",
        "frozen_strata",
    ]
    auth_schema_path.write_bytes(canonical_json_bytes(auth_schema))

    readme = """# Leakage calibration bundle v2

This directory contains the frozen 400-item calibration-only cohort produced
after the v1 pilot gate. It is separate from the v1 pilot and from every paper
experiment. The cohort has 150 keep, 150 reject, and 100 ambiguous/missing
sealed filter states, allocated by deterministic stratum round-robin. Labeler
A is present; independent Labeler B and adjudication are still outstanding.

Validate without network access:

    benchmark-harness leakage-calibration validate --root data/calibration/leakage-probe/v2

The blind view contains a bounded task prompt plus model-visible title,
snippet, content, and authoritative publication/update evidence. It contains
no expected output, ground truth, filter verdicts, scores, or opposite labels.
Full provider requests, responses, and unredacted candidate payloads remain in
the ignored restricted data/calibration-cache/leakage-probe/v2/ cache. This
enriched sample is for filter sensitivity/specificity and task-relevance
calibration; it cannot estimate natural leakage prevalence without inclusion
probabilities.
"""
    (root / "README.md").write_bytes(readme.encode("utf-8"))
    for relative in EXPECTED_TRACKED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative in JSONL_FILES:
            path.write_bytes(b"")
    _write_json(
        root / "authorization.json",
        {
            "authorization_id": FROZEN_AUTHORIZATION_ID,
            "calibration_id": FROZEN_CALIBRATION_ID,
            "calibration_unrestricted_collection_authorized": True,
            "frozen_item_count": FROZEN_ITEM_COUNT,
            "frozen_strata": FROZEN_STRATUM_TARGETS,
            "full_paper_unrestricted_authorized": False,
            "issue_id": "5e7ed4b6-3e65-4cfa-9da8-0ea82bc31fc4",
            "issued_at_utc": "2026-08-31T04:36:51Z",
            "pilot_item_count": 40,
            "pilot_strata": {stratum: 10 for stratum in FROZEN_STRATA},
            "same_candidates_cutoff_filter": True,
            "schema_version": FROZEN_AUTHORIZATION_SCHEMA,
            "scope": FROZEN_AUTHORIZATION_SCOPE,
            "source_comment_id": "01a0561b-18ca-7bc3-b610-0ff6475e3690",
            "status": "active",
        },
    )
    _write_json(
        root / "cohort.json",
        {
            "backbone_id": "openai:gpt-4.1",
            "backbone_training_cutoff": "2024-06-30",
            "calibration_status": "not_started",
            "cohort_id": FROZEN_CALIBRATION_ID,
            "experiment_as_of": "2026-08-17",
            "guard_days": 30,
            "hub_repo_id": HUB_REPO_ID,
            "hub_revision": HUB_REVISION,
            "live_unrestricted_authorized": False,
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "target_adjudicated_count": FROZEN_ITEM_COUNT,
            "target_item_count": FROZEN_ITEM_COUNT,
            "target_raw_label_count": FROZEN_RAW_LABEL_COUNT,
            "task_cutoff_semantics": "utc_calendar_day_inclusive_next_day_exclusive",
            "temporal_index_sha256": FROZEN_TEMPORAL_INDEX_SHA256,
        },
    )
    _write_json(
        root / "sampling-plan.json",
        {
            "domain_cap": 20,
            "frozen_set": {
                "ambiguous_or_missing": 100,
                "calibration_only": True,
                "keep": 150,
                "non_scored": True,
                "reject": 150,
                "reviewed_only": True,
                "size": FROZEN_ITEM_COUNT,
            },
            "pilot": {
                "double_blind": True,
                "enters_frozen_set": False,
                "size": 40,
                "strata": sorted(FROZEN_STRATA),
            },
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "sort_key": 'SHA256("leakage-calibration/v1|<hub_full_sha>|<item_id>")',
            "status": "frozen",
            "strata": [
                {
                    "ambiguous_or_missing": counts["ambiguous"],
                    "keep": counts["keep"],
                    "reject": counts["reject"],
                    "stratum": stratum,
                }
                for stratum, counts in FROZEN_STRATUM_TARGETS.items()
            ],
            "task_cap": 8,
        },
    )
    _write_json(
        root / "reports/calibration-report.json",
        {
            "calibration_status": "not_started",
            "claim_eligible": False,
            "cohen_kappa": None,
            "counts": {
                "adjudicated": 0,
                "determinate_binary_gold": 0,
                "items": 0,
                "leakage_negative": 0,
                "leakage_positive": 0,
                "raw_labels": 0,
                "raw_labels_a": 0,
                "raw_labels_b": 0,
                "unknown": 0,
                "unresolved_disputes": 0,
            },
            "exclusions_by_type": {},
            "filter_validated": False,
            "generated_at_utc": "2026-08-31T04:36:51Z",
            "hub_revision": HUB_REVISION,
            "inclusion_probability_note": (
                "This enriched sample cannot estimate natural prevalence; any prevalence "
                "estimate must use inclusion probabilities."
            ),
            "live_unrestricted_authorized": False,
            "quality_gates_passed": False,
            "raw_agreement": None,
            "schema_version": CALIBRATION_SCHEMA_VERSION,
        },
    )
    _write_json(
        root / "artifact_manifest.json",
        {
            "adjudication": {
                "adjudicated_observed": 0,
                "raw_labels_expected": FROZEN_RAW_LABEL_COUNT,
                "raw_labels_observed": 0,
                "status": "not_started",
                "unresolved_disputes": 0,
            },
            "calibration_id": FROZEN_CALIBRATION_ID,
            "calibration_status": "not_started",
            "clocks": {
                "backbone_id": "openai:gpt-4.1",
                "backbone_training_cutoff": "2024-06-30",
                "experiment_as_of": "2026-08-17",
                "guard_days": 30,
                "task_cutoff_semantics": "utc_calendar_day_inclusive_next_day_exclusive",
            },
            "files": [],
            "hub": {"repo_id": HUB_REPO_ID, "revision": HUB_REVISION},
            "live_unrestricted_authorized": False,
            "query_corpus": {"path": "query-corpus.jsonl", "sha256": sha256_text("")},
            "restricted_evidence_manifest_sha256": None,
            "rubric": {"path": "rubric.md", "sha256": _sha256_file(root / "rubric.md")},
            "sampling_plan": {
                "path": "sampling-plan.json",
                "sha256": _sha256_file(root / "sampling-plan.json"),
            },
            "schema_version": "leakage-calibration-manifest/v1",
        },
    )
    _refresh_frozen_manifest_files(root)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _collect_frozen_candidates(
    *,
    cache_root: Path,
    dataset_cache: str | Path,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[dict[str, Any]]]:
    tasks = _load_tasks(dataset_cache)
    access_key = os.environ.get("VOLC_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("VOLC_SECRET_ACCESS_KEY", "").strip()
    if not access_key or not secret_key:
        raise FrozenCalibrationError(
            "Volc credentials are unavailable for frozen calibration collection"
        )
    transport = VolcSdkTransport(access_key=access_key, secret_key=secret_key)
    executor = ThreadPoolExecutor(max_workers=len(FROZEN_QUERY_CLASSES))
    cached_response_by_call_id: dict[str, VolcHttpResponse] = {}
    future_by_call_id: dict[str, Any] = {}
    prefetch_task_count = 0
    for stratum in FROZEN_STRATA:
        task_limit = FROZEN_TASK_LIMITS[stratum]
        if len(tasks[stratum]) < task_limit:
            executor.shutdown(wait=False, cancel_futures=True)
            raise FrozenCalibrationError(
                f"Hub has fewer than {task_limit} reviewed tasks in {stratum}"
            )
        for row in _frozen_task_rows(tasks[stratum], task_limit):
            prefetch_task_count += 1
            task_id = str(row["task_id"])
            cutoff_date = _date_text(row.get("cutoff_date"), "cutoff_date", task_id)
            for query_class in FROZEN_QUERY_CLASSES:
                query = _frozen_query(row, query_class, cutoff_date)
                call_id = _frozen_call_id(stratum, prefetch_task_count, query_class)
                body = {
                    "Query": query,
                    "SearchType": "web",
                    "Count": 10,
                    "Filter": {"NeedContent": True, "NeedUrl": True},
                    "TimeRange": _unrestricted_time_range(cutoff_date),
                }
                request = {
                    "body": body,
                    "call_id": call_id,
                    "task_id": task_id,
                    "query": query,
                }
                cached_response = _cached_frozen_response(cache_root, request)
                if cached_response is not None:
                    cached_response_by_call_id[call_id] = cached_response
                    continue
                future_by_call_id[call_id] = executor.submit(
                    _frozen_post,
                    transport,
                    "https://mercury.volcengineapi.com/",
                    body,
                    timeout_seconds,
                )
    candidate_pool: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    evidence_entries: list[dict[str, Any]] = []
    previous_receipt_sha256: str | None = None
    task_count = 0
    for stratum in FROZEN_STRATA:
        task_limit = FROZEN_TASK_LIMITS[stratum]
        if len(tasks[stratum]) < task_limit:
            raise FrozenCalibrationError(
                f"Hub has fewer than {task_limit} reviewed tasks in {stratum}"
            )
        for row in _frozen_task_rows(tasks[stratum], task_limit):
            task_count += 1
            task_id = str(row["task_id"])
            cutoff_date = _date_text(row.get("cutoff_date"), "cutoff_date", task_id)
            forecast_origin = _date_text(row.get("forecast_origin"), "forecast_origin", task_id)
            for query_class in FROZEN_QUERY_CLASSES:
                query = _frozen_query(row, query_class, cutoff_date)
                query_rows.append(
                    {
                        "query_class": query_class,
                        "query_corpus_entry_id": sha256_text(
                            f"leakage-calibration-query/v2\0{task_id}\0{query_class}"
                        ),
                        "query_sha256": sha256_text(query),
                        "schema_version": CALIBRATION_SCHEMA_VERSION,
                        "task_id": task_id,
                        "template_version": FROZEN_QUERY_TEMPLATE_VERSION,
                    }
                )
                call_id = _frozen_call_id(stratum, task_count, query_class)
                body = {
                    "Query": query,
                    "SearchType": "web",
                    "Count": 10,
                    "Filter": {"NeedContent": True, "NeedUrl": True},
                    "TimeRange": _unrestricted_time_range(cutoff_date),
                }
                if call_id in cached_response_by_call_id:
                    response = cached_response_by_call_id[call_id]
                else:
                    try:
                        response = future_by_call_id[call_id].result()
                    except Exception as exc:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise FrozenCalibrationError(
                            f"Volc frozen calibration call failed for {call_id}"
                        ) from exc
                response_payload = response.payload
                response_hash = (
                    sha256_json(response_payload)
                    if response_payload is not None
                    else sha256_text(response.error or "volc_response_missing")
                )
                cache_call_path = cache_root / "calls" / f"{call_id}.json"
                _write_cache_json(
                    cache_call_path,
                    {
                        "authorization_id": FROZEN_AUTHORIZATION_ID,
                        "call_id": call_id,
                        "collection_id": FROZEN_COLLECTION_ID,
                        "query": query,
                        "request": body,
                        "response": response_payload,
                        "response_error": response.error,
                        "response_request_id": response.request_id,
                        "response_status_code": response.status_code,
                        "schema_version": FROZEN_CACHE_SCHEMA,
                        "task_id": task_id,
                    },
                )
                evidence_entries.append(_cache_entry(cache_call_path, cache_root))
                if response.error or not 200 <= (response.status_code or 0) < 300:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise FrozenCalibrationError(
                        f"Volc frozen calibration call failed for {call_id}"
                    )
                raw_results = _volc_web_results(response_payload)
                if raw_results is None:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise FrozenCalibrationError(f"Volc response shape invalid for {call_id}")
                receipt_payload = {
                    "call_id": call_id,
                    "candidate_count": len(raw_results),
                    "collection_id": FROZEN_COLLECTION_ID,
                    "previous_receipt_sha256": previous_receipt_sha256,
                    "provider_id": "volc",
                    "query_sha256": sha256_text(query),
                    "receipt_protocol": "search_receipt/v1",
                    "request_filter_sha256": sha256_json(body),
                    "response_sha256": response_hash,
                    "task_id": task_id,
                    "tool_contract_sha256": DEFAULT_TOOL_CONTRACT.hash,
                }
                receipt_sha256 = sha256_json(receipt_payload)
                previous_receipt_sha256 = receipt_sha256
                for candidate_index, raw_candidate in enumerate(raw_results):
                    if not isinstance(raw_candidate, Mapping):
                        continue
                    candidate = VolcSearchCandidate.from_volc_result(raw_candidate)
                    canonical_url = _canonical_url(candidate.url)
                    if canonical_url is None:
                        continue
                    filtered = filter_volc_candidates(
                        (candidate,), cutoff_date, llm_filter=None
                    ).decisions[0]
                    filter_state, filter_verdict = _filter_state(filtered)
                    candidate_pool.append(
                        {
                            "candidate": candidate,
                            "canonical_url": canonical_url,
                            "candidate_index": candidate_index,
                            "category": row["category"],
                            "call_id": call_id,
                            "cutoff_date": cutoff_date,
                            "filter_reason": filtered.reason,
                            "filter_state": filter_state,
                            "filter_verdict": filter_verdict,
                            "forecast_origin": forecast_origin,
                            "paper_band": row["paper_band"],
                            "previous_receipt_sha256": receipt_payload["previous_receipt_sha256"],
                            "query_sha256": sha256_text(query),
                            "raw_candidate": dict(raw_candidate),
                            "receipt_sha256": receipt_sha256,
                            "request_filter_sha256": receipt_payload["request_filter_sha256"],
                            "response_sha256": response_hash,
                            "stratum": stratum,
                            "task_id": task_id,
                            "updated_at_utc": _raw_provider_time(raw_candidate, "UpdateTime"),
                        }
                    )
    executor.shutdown(wait=True)
    query_rows.sort(key=lambda row: row["query_corpus_entry_id"])
    return candidate_pool, query_rows, task_count, evidence_entries


_FROZEN_REQUEST_LOCK = Lock()
_FROZEN_NEXT_REQUEST_AT = 0.0
_FROZEN_REQUEST_INTERVAL_SECONDS = 1.0


def _frozen_post(
    transport: VolcSdkTransport,
    endpoint: str,
    body: Mapping[str, Any],
    timeout_seconds: float,
) -> Any:
    """Throttle the bounded probe and retry provider rate-limit responses."""

    global _FROZEN_NEXT_REQUEST_AT
    for attempt in range(3):
        with _FROZEN_REQUEST_LOCK:
            delay = _FROZEN_NEXT_REQUEST_AT - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            _FROZEN_NEXT_REQUEST_AT = time.monotonic() + _FROZEN_REQUEST_INTERVAL_SECONDS
        response = transport.post(
            endpoint,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout_seconds=timeout_seconds,
        )
        payload = response.payload
        metadata = payload.get("ResponseMetadata") if isinstance(payload, Mapping) else None
        provider_error = metadata.get("Error") if isinstance(metadata, Mapping) else None
        error_code = provider_error.get("Code") if isinstance(provider_error, Mapping) else None
        if error_code != "rate_limit_exceeded" or attempt == 2:
            return response
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _cached_frozen_response(
    cache_root: Path, request: Mapping[str, Any]
) -> VolcHttpResponse | None:
    """Reuse only a complete response for the exact current request identity."""

    path = cache_root / "calls" / f"{request['call_id']}.json"
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cached, Mapping):
        return None
    if any(
        cached.get(name) != expected
        for name, expected in (
            ("authorization_id", FROZEN_AUTHORIZATION_ID),
            ("call_id", request["call_id"]),
            ("collection_id", FROZEN_COLLECTION_ID),
            ("query", request["query"]),
            ("request", request["body"]),
            ("schema_version", FROZEN_CACHE_SCHEMA),
            ("task_id", request["task_id"]),
        )
    ):
        return None
    response_payload = cached.get("response")
    response_error = cached.get("response_error")
    status_code = cached.get("response_status_code")
    if response_error is not None or not isinstance(status_code, int):
        return None
    if not 200 <= status_code < 300 or _volc_web_results(response_payload) is None:
        return None
    metadata = (
        response_payload.get("ResponseMetadata") if isinstance(response_payload, Mapping) else None
    )
    provider_error = metadata.get("Error") if isinstance(metadata, Mapping) else None
    if isinstance(provider_error, Mapping) and provider_error:
        return None
    request_id = cached.get("response_request_id")
    if request_id is not None and not isinstance(request_id, str):
        return None
    return VolcHttpResponse(
        status_code=status_code,
        payload=response_payload,
        error=None,
        request_id=request_id,
    )


def _frozen_task_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Choose a deterministic spread of reviewed tasks within one stratum."""

    if len(rows) <= limit:
        return rows[:limit]
    return [rows[(index * len(rows)) // limit] for index in range(limit)]


def _frozen_query(row: Mapping[str, Any], query_class: str, cutoff_date: str) -> str:
    """Generate a bounded, class-specific query without exposing task prompts."""

    base = _build_query(row, query_class, cutoff_date)
    year = cutoff_date[:4]
    historical_year = str(int(year) - 1)
    suffix = {
        "cutoff_anchored": (
            f"official annual report {historical_year} published before {cutoff_date}"
        ),
        "neutral_entity_topic": f"official historical annual report {historical_year}",
        "retrospective_outcome_challenge": (f"reported earnings annual report {historical_year}"),
    }[query_class]
    query = f"{base} {suffix}".strip()
    if len(query) > 512:
        raise FrozenCalibrationError("generated frozen query exceeded the bounded query contract")
    return query


def _load_pilot_item_ids(root: Path) -> set[str]:
    pilot_items_path = root.parent / "v1/items.jsonl"
    try:
        pilot_rows = _read_jsonl(pilot_items_path)
    except (OSError, ValueError) as exc:
        raise FrozenCalibrationError("v1 pilot item IDs are unavailable") from exc
    item_ids = {str(row.get("item_id", "")) for row in pilot_rows}
    if len(pilot_rows) != 40 or len(item_ids) != 40 or "" in item_ids:
        raise FrozenCalibrationError("v1 pilot must contain exactly 40 unique item IDs")
    return item_ids


def _select_frozen_items(
    candidate_pool: list[dict[str, Any]], *, excluded_item_ids: Collection[str] = ()
) -> list[dict[str, Any]]:
    excluded = set(excluded_item_ids)
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {
        stratum: {bucket: [] for bucket in FROZEN_STATE_TARGETS} for stratum in FROZEN_STRATA
    }
    seen: set[str] = set()
    for candidate in candidate_pool:
        item_id = _provisional_item_id(candidate)
        if item_id in seen or item_id in excluded:
            continue
        seen.add(item_id)
        bucket = candidate["filter_state"]
        if bucket == "missing":
            bucket = "ambiguous"
        if bucket not in FROZEN_STATE_TARGETS:
            continue
        enriched = dict(candidate)
        enriched["item_id"] = item_id
        pools[candidate["stratum"]][bucket].append(enriched)
    for stratum in FROZEN_STRATA:
        for bucket in FROZEN_STATE_TARGETS:
            pools[stratum][bucket].sort(key=lambda candidate: _sampling_key(candidate["item_id"]))
            for rank, candidate in enumerate(pools[stratum][bucket], start=1):
                candidate["deterministic_rank"] = rank
            if len(pools[stratum][bucket]) < FROZEN_STRATUM_TARGETS[stratum][bucket]:
                raise FrozenCalibrationError(
                    f"frozen stratum {stratum} has fewer than "
                    f"{FROZEN_STRATUM_TARGETS[stratum][bucket]} {bucket} candidates"
                )

    selected: list[dict[str, Any]] = []
    domain_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    for bucket in ("ambiguous", "keep", "reject"):
        remaining = {stratum: FROZEN_STRATUM_TARGETS[stratum][bucket] for stratum in FROZEN_STRATA}
        cursors = {stratum: 0 for stratum in FROZEN_STRATA}
        while any(remaining.values()):
            progressed = False
            for stratum in FROZEN_STRATA:
                if remaining[stratum] == 0:
                    continue
                pool = pools[stratum][bucket]
                while cursors[stratum] < len(pool):
                    candidate = pool[cursors[stratum]]
                    cursors[stratum] += 1
                    domain = _registrable_domain(candidate["canonical_url"])
                    task_id = candidate["task_id"]
                    if domain_counts[domain] >= 20 or task_counts[task_id] >= 8:
                        continue
                    chosen = dict(candidate)
                    chosen["inclusion_probability"] = min(
                        1.0,
                        FROZEN_STRATUM_TARGETS[stratum][bucket] / max(1, len(pool)),
                    )
                    selected.append(chosen)
                    domain_counts[domain] += 1
                    task_counts[task_id] += 1
                    remaining[stratum] -= 1
                    progressed = True
                    break
            if not progressed:
                raise FrozenCalibrationError(
                    f"frozen {bucket} selection exhausted under domain/task caps"
                )
    if len(selected) != FROZEN_ITEM_COUNT:
        raise FrozenCalibrationError("frozen selection did not produce exactly 400 items")
    return selected


def _registrable_domain(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    labels = host.split(".")
    if len(labels) < 2:
        return host
    two_label_suffixes = {
        "ac.cn",
        "co.au",
        "co.jp",
        "co.kr",
        "co.nz",
        "co.uk",
        "com.au",
        "com.cn",
        "com.hk",
        "com.tw",
        "gov.cn",
        "net.cn",
        "org.cn",
    }
    suffix = ".".join(labels[-2:])
    return ".".join(labels[-3:]) if suffix in two_label_suffixes and len(labels) >= 3 else suffix


def _materialize_frozen_items(
    selected: list[dict[str, Any]],
    *,
    cache_root: Path,
    evidence_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    captured_at = _now_utc()
    item_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    for candidate_record in selected:
        candidate: VolcSearchCandidate = candidate_record["candidate"]
        raw_url = candidate.url
        if raw_url is None:
            raise FrozenCalibrationError("selected candidate has no URL")
        canonical_url = candidate_record["canonical_url"]
        payload_sha256 = _sha256_canonical(candidate.as_model_payload())
        item_id = candidate_record["item_id"]
        candidate_cache_path = cache_root / "candidates" / f"{item_id}.json"
        _write_cache_json(
            candidate_cache_path,
            {
                "candidate": candidate_record["raw_candidate"],
                "call_id": candidate_record["call_id"],
                "candidate_index": candidate_record["candidate_index"],
                "query_sha256": candidate_record["query_sha256"],
                "schema_version": FROZEN_CACHE_SCHEMA,
                "task_id": candidate_record["task_id"],
            },
        )
        evidence_entries.append(_cache_entry(candidate_cache_path, cache_root))
        published_at = _raw_provider_time(candidate_record["raw_candidate"], "PublishTime")
        item_rows.append(
            {
                "backbone_id": "openai:gpt-4.1",
                "backbone_training_cutoff": "2024-06-30",
                "call_id": candidate_record["call_id"],
                "candidate_index": candidate_record["candidate_index"],
                "canonical_url": canonical_url,
                "canonical_url_sha256": sha256_text(canonical_url),
                "canonical_url_version": "url-canonicalizer/v1",
                "category": candidate_record["category"],
                "collection_id": FROZEN_COLLECTION_ID,
                "content_sha256": sha256_text(candidate.content or ""),
                "cutoff_declaration_sha256": sha256_json(
                    {
                        "backbone_training_cutoff": "2024-06-30",
                        "cutoff_date": candidate_record["cutoff_date"],
                        "experiment_as_of": "2026-08-17",
                        "guard_days": 30,
                        "task_id": candidate_record["task_id"],
                    }
                ),
                "deterministic_rank": candidate_record["deterministic_rank"],
                "domain": urlsplit(canonical_url).hostname,
                "exclusive_cutoff_bound": CutoffWindow.from_seed(
                    candidate_record["cutoff_date"]
                ).exclusive_bound,
                "experiment_as_of": "2026-08-17",
                "forecast_origin": candidate_record["forecast_origin"],
                "hub_repo_id": HUB_REPO_ID,
                "hub_revision": HUB_REVISION,
                "inclusion_probability": candidate_record["inclusion_probability"],
                "item_id": item_id,
                "model_visible_payload_sha256": payload_sha256,
                "original_url_sha256": sha256_text(raw_url),
                "paper_band": candidate_record["paper_band"],
                "policy_sha256": FROZEN_POLICY_SHA256,
                "previous_receipt_sha256": candidate_record["previous_receipt_sha256"],
                "provider_id": "volc",
                "provider_published_at_utc": published_at,
                "provider_updated_at_utc": candidate_record["updated_at_utc"],
                "query_sha256": candidate_record["query_sha256"],
                "receipt_protocol": "search_receipt/v1",
                "receipt_sha256": candidate_record["receipt_sha256"],
                "request_filter_sha256": candidate_record["request_filter_sha256"],
                "response_sha256": candidate_record["response_sha256"],
                "restricted_snapshot_bytes": candidate_cache_path.stat().st_size,
                "restricted_snapshot_locator": canonical_url,
                "restricted_snapshot_mime": "application/json",
                "restricted_snapshot_sha256": _sha256_file(candidate_cache_path),
                "restricted_snapshot_status": "available",
                "retrieved_at_utc": captured_at,
                "sampling_stratum": candidate_record["stratum"],
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "service_cohort_sha256": FROZEN_SERVICE_COHORT_SHA256,
                "snapshot_captured_at_utc": captured_at,
                "snippet_sha256": sha256_text(candidate.snippet or ""),
                "task_cutoff_date": candidate_record["cutoff_date"],
                "task_id": candidate_record["task_id"],
                "temporal_index_sha256": FROZEN_TEMPORAL_INDEX_SHA256,
                "title_sha256": sha256_text(candidate.title or ""),
                "tool_contract_sha256": DEFAULT_TOOL_CONTRACT.hash,
            }
        )
        filter_rows.append(
            {
                "decided_at_utc": captured_at,
                "filter_policy_sha256": PILOT_FILTER_POLICY_SHA256,
                "filter_state": candidate_record["filter_state"],
                "filter_verdict": candidate_record["filter_verdict"],
                "item_id": item_id,
                "schema_version": CALIBRATION_SCHEMA_VERSION,
            }
        )
    item_rows.sort(key=lambda row: row["item_id"])
    filter_rows.sort(key=lambda row: row["item_id"])
    return item_rows, filter_rows


def _frozen_call_id(stratum: str, task_number: int, query_class: str) -> str:
    value = f"frozen400-{stratum.lower().replace('-', '')}-{task_number:03d}-{query_class}"
    if not value.replace("-", "").replace("_", "").isalnum():
        raise FrozenCalibrationError("frozen call identity is unsafe")
    return value


def _write_frozen_evidence_manifest(cache_root: Path, entries: list[dict[str, Any]]) -> str:
    path = cache_root / "restricted-evidence-manifest.json"
    _write_cache_json(
        path,
        {
            "authorization_id": FROZEN_AUTHORIZATION_ID,
            "calibration_id": FROZEN_CALIBRATION_ID,
            "collection_id": FROZEN_COLLECTION_ID,
            "entries": sorted(entries, key=lambda entry: entry["path"]),
            "schema_version": "leakage-calibration-restricted-evidence/v1",
        },
    )
    return _sha256_file(path)


def _refresh_frozen_state(
    root: Path, *, status: str, restricted_evidence_manifest_sha256: str
) -> None:
    cohort = json.loads((root / "cohort.json").read_text(encoding="utf-8"))
    cohort["calibration_status"] = status
    (root / "cohort.json").write_bytes(canonical_json_bytes(cohort))
    items = _read_jsonl(root / "items.jsonl")
    labels_a = _read_jsonl(root / "labels/raw/annotator-a.jsonl")
    labels_b = _read_jsonl(root / "labels/raw/annotator-b.jsonl")
    adjudicated = _read_jsonl(root / "labels/adjudicated.jsonl")
    counts = {
        "adjudicated": len(adjudicated),
        "determinate_binary_gold": 0,
        "items": len(items),
        "leakage_negative": 0,
        "leakage_positive": 0,
        "raw_labels": len(labels_a) + len(labels_b),
        "raw_labels_a": len(labels_a),
        "raw_labels_b": len(labels_b),
        "unknown": len(items),
        "unresolved_disputes": 0,
    }
    report = json.loads((root / "reports/calibration-report.json").read_text(encoding="utf-8"))
    report.update(
        {
            "calibration_status": status,
            "claim_eligible": False,
            "cohen_kappa": None,
            "counts": counts,
            "exclusions_by_type": {
                "awaiting_labeler_b": max(0, len(items) - len(labels_b)),
                "awaiting_adjudication": max(0, len(items) - len(adjudicated)),
            },
            "filter_validated": False,
            "generated_at_utc": _now_utc(),
            "quality_gates_passed": False,
            "raw_agreement": None,
        }
    )
    (root / "reports/calibration-report.json").write_bytes(canonical_json_bytes(report))
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    manifest["calibration_status"] = status
    manifest["restricted_evidence_manifest_sha256"] = restricted_evidence_manifest_sha256
    manifest["adjudication"] = {
        "adjudicated_observed": len(adjudicated),
        "raw_labels_expected": FROZEN_RAW_LABEL_COUNT,
        "raw_labels_observed": len(labels_a) + len(labels_b),
        "status": "incomplete",
        "unresolved_disputes": 0,
    }
    (root / "artifact_manifest.json").write_bytes(canonical_json_bytes(manifest))
    _refresh_frozen_manifest_files(root)


def _refresh_frozen_manifest_files(root: Path) -> None:
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rubric"]["sha256"] = _sha256_file(root / "rubric.md")
    manifest["sampling_plan"]["sha256"] = _sha256_file(root / "sampling-plan.json")
    manifest["query_corpus"]["sha256"] = _sha256_file(root / "query-corpus.jsonl")
    entries: list[dict[str, Any]] = []
    for relative in EXPECTED_TRACKED_FILES:
        path = root / relative
        payload = path.read_bytes()
        row_count: int | None
        if relative in JSONL_FILES:
            row_count = len(payload.decode("utf-8").splitlines()) if payload else 0
        elif relative in JSON_FILES:
            row_count = 1
        else:
            row_count = None
        entries.append(
            {
                "bytes": len(payload),
                "path": relative,
                "row_count": row_count,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest["files"] = entries
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    (root / "CALIBRATION.sha256").write_text(
        hashlib.sha256(manifest_bytes).hexdigest() + "  artifact_manifest.json\n",
        encoding="utf-8",
    )


def _validate_frozen_state(root: Path, *, expected_labels: int) -> None:
    result = validate_calibration_root(root)
    if result["calibration_id"] != FROZEN_CALIBRATION_ID:
        raise FrozenCalibrationError("unexpected calibration version")
    if result["counts"]["items"] != FROZEN_ITEM_COUNT:
        raise FrozenCalibrationError("frozen item count failed validation")
    if result["counts"]["raw_labels_a"] != expected_labels:
        raise FrozenCalibrationError("frozen Labeler A count failed validation")


def _private_directory(path: Path) -> None:
    try:
        path.chmod(0o700)
    except OSError:
        pass


__all__ = ["FrozenCalibrationError", "FrozenCalibrationResult", "run_frozen"]
