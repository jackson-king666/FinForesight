"""Authorized fresh v4 relevance/outcome calibration collection.

The v4 run is deliberately narrower than the paper experiment.  It performs
three bounded Volc searches for each of ten approved tasks in each stratum,
applies the existing cutoff filter to those exact candidates, stores raw
provider payloads only in the ignored cache, and emits a new 40-item blind
Labeler-A pilot.  The v3 failed-gate audit and its 400-item cohort are copied
byte-for-byte at the top level and never relabeled.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

from benchmark_harness.calibration.pilot import (
    PILOT_FILTER_POLICY_SHA256,
    PILOT_QUERY_CLASSES,
    PILOT_STRATA,
    PilotCollectionError,
    _annotation_views,
    _blind_temporal_label,
    _build_query,
    _cache_entry,
    _canonical_url,
    _date_text,
    _filter_state,
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
from benchmark_harness.calibration.relevance_v4 import (
    RUBRIC_VERSION,
    label_relevance_outcome,
)
from benchmark_harness.calibration.validator import (
    CALIBRATION_SCHEMA_VERSION,
    EXPECTED_TRACKED_FILES,
    FRESH_RELEVANCE_AUTHORIZATION_ID,
    FRESH_RELEVANCE_CALIBRATION_ID,
    FRESH_RELEVANCE_COLLECTION_ID,
    FRESH_RELEVANCE_PILOT_JSONL_FILES,
    FRESH_RELEVANCE_RUBRIC_VERSION,
    FRESH_RELEVANCE_SCOPE,
    FRESH_RELEVANCE_SOURCE_COMMENT_ID,
    FRESH_RELEVANCE_TRACKED_FILES,
    HUB_REPO_ID,
    V4_JSONL_FILES,
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
from benchmark_harness.search.volc import _volc_response_error, _volc_web_results

V4_STRATA = PILOT_STRATA
V4_ITEM_COUNT = 40
V4_TARGET_PER_STRATUM = 10
V4_CACHE_SCHEMA = "leakage-calibration-cache/v1"
V4_AUTHORIZATION_SCHEMA = "leakage-calibration-authorization/v1"
V4_QUERY_TEMPLATE_VERSION = "pilot-query/v4"
V4_FILTER_SCHEMA = "cutoff-filter/v1"
V4_TEMPORAL_INDEX_SHA256 = "80d977e40e113172e778b6f78315e1fad132423c9fe9eca3ad6a5de34df4f21b"
V4_BACKBONE_ID = "openai:gpt-4.1"
V4_BACKBONE_TRAINING_CUTOFF = "2024-06-30"
V4_EXPERIMENT_AS_OF = "2026-08-17"
V4_GUARD_DAYS = 30
V4_POLICY = {
    "calibration_id": FRESH_RELEVANCE_CALIBRATION_ID,
    "ranking_eligible": False,
    "regime": "unrestricted_current_web",
    "scope": "calibration_relevance_rubric_v3_fresh_pilot",
    "schema_version": "leakage_probe/v4",
}
V4_POLICY_SHA256 = sha256_json(V4_POLICY)
V4_SERVICE_COHORT_SHA256 = sha256_json(
    {
        "calibration_id": FRESH_RELEVANCE_CALIBRATION_ID,
        "hub_revision": "941370f228b9bc03c1b8bd9547c098c42c06be23",
        "provider_id": "volc",
        "tool_contract_sha256": DEFAULT_TOOL_CONTRACT.hash,
    }
)
_CALL_PREFIX = "v4pilot40"


class FreshPilotCollectionError(PilotCollectionError):
    """Raised when the authorized v4 pilot cannot be collected safely."""


@dataclass(frozen=True, slots=True)
class FreshPilotResult:
    """Safe summary returned after the v4 collection and Labeler-A pass."""

    collection_id: str
    task_count: int
    query_count: int
    candidate_pool_count: int
    item_count: int
    label_a_count: int
    calibration_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "calibration_id": FRESH_RELEVANCE_CALIBRATION_ID,
            "calibration_sha256": self.calibration_sha256,
            "candidate_pool_count": self.candidate_pool_count,
            "collection_id": self.collection_id,
            "item_count": self.item_count,
            "label_a_count": self.label_a_count,
            "query_count": self.query_count,
            "task_count": self.task_count,
        }


def run_fresh_pilot(
    *,
    root: str | Path = "data/calibration/leakage-probe/v4",
    cache_root: str | Path = "data/calibration-cache/leakage-probe/v4",
    dataset_cache: str | Path = "data/.huggingface",
    timeout_seconds: float = 30.0,
) -> FreshPilotResult:
    """Collect and label the explicitly authorized fresh 40-item v4 pilot."""

    root_path = Path(root).expanduser().resolve()
    cache_path = Path(cache_root).expanduser().resolve()
    _prepare_v4_scaffold(root_path)
    if timeout_seconds <= 0:
        raise FreshPilotCollectionError("v4 pilot timeout must be positive")
    cache_path.mkdir(parents=True, exist_ok=True)
    _private_directory(cache_path)
    artifact_manifest_path = root_path / "artifact_manifest.json"
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    if artifact_manifest.get("restricted_evidence_manifest_sha256") is None:
        _write_v4_evidence_manifest(cache_path, [])
        _refresh_v4_state(
            root_path,
            _sha256_file(cache_path / "restricted-evidence-manifest.json"),
        )
    try:
        state = validate_calibration_root(root_path)
    except ValueError as exc:
        raise FreshPilotCollectionError("v4 scaffold failed validation before collection") from exc
    if state["calibration_id"] != FRESH_RELEVANCE_CALIBRATION_ID:
        raise FreshPilotCollectionError("unexpected calibration version for fresh pilot")
    if _pilot_has_rows(root_path):
        raise FreshPilotCollectionError("v4 pilot root is already populated; create a new version")

    candidate_pool, query_rows, task_count, evidence_entries = _collect_candidates(
        cache_root=cache_path,
        dataset_cache=dataset_cache,
        timeout_seconds=timeout_seconds,
    )
    selected = _select_items(candidate_pool, root_path)
    if len(selected) != V4_ITEM_COUNT:
        raise FreshPilotCollectionError("v4 selection did not produce exactly 40 items")
    item_rows, filter_rows = _materialize_items(
        selected,
        cache_root=cache_path,
        evidence_entries=evidence_entries,
    )
    task_prompts = _load_task_prompts(dataset_cache, item_rows)
    pilot_root = root_path / "pilot"
    _write_jsonl(pilot_root / "query-corpus.jsonl", query_rows)
    _write_jsonl(pilot_root / "items.jsonl", item_rows)
    _write_jsonl(pilot_root / "filter-decisions.jsonl", filter_rows)
    _write_jsonl(
        pilot_root / "annotation-view.jsonl",
        _annotation_views(item_rows, cache_path, task_prompts=task_prompts),
    )
    _write_v4_evidence_manifest(cache_path, evidence_entries)
    _refresh_v4_state(root_path, _sha256_file(cache_path / "restricted-evidence-manifest.json"))
    _validate_v4_pilot_state(root_path, expected_labels=0)
    label_count = _label_v4_pilot(root_path)
    _refresh_v4_state(root_path, _sha256_file(cache_path / "restricted-evidence-manifest.json"))
    validated = validate_calibration_root(root_path)
    if validated["counts"]["items"] != 400 or label_count != V4_ITEM_COUNT:
        raise FreshPilotCollectionError("v4 pilot corpus or Labeler A output is incomplete")
    return FreshPilotResult(
        collection_id=FRESH_RELEVANCE_COLLECTION_ID,
        task_count=task_count,
        query_count=len(query_rows),
        candidate_pool_count=len(candidate_pool),
        item_count=V4_ITEM_COUNT,
        label_a_count=label_count,
        calibration_sha256=validated["calibration_sha256"],
    )


def _prepare_v4_scaffold(root: Path) -> None:
    if root.exists():
        return
    source_root = root.parent / "v3"
    if not source_root.is_dir():
        raise FreshPilotCollectionError("v3 failed-gate audit is unavailable")
    root.mkdir(parents=True, exist_ok=False)
    for relative in EXPECTED_TRACKED_FILES:
        source = source_root / relative
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative in {
            "annotation-view.jsonl",
            "filter-decisions.jsonl",
            "items.jsonl",
            "query-corpus.jsonl",
        }:
            shutil.copyfile(source, target)
        elif relative.startswith("schemas/"):
            shutil.copyfile(source, target)
        elif relative == "rubric.md":
            target.write_text(_V4_RUBRIC, encoding="utf-8")
        elif relative in V4_JSONL_FILES:
            target.write_bytes(b"")

    for relative in FRESH_RELEVANCE_PILOT_JSONL_FILES:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(b"")
    auth_schema_path = root / "schemas/authorization.schema.json"
    auth_schema = json.loads(auth_schema_path.read_text(encoding="utf-8"))
    auth_schema["title"] = "Explicit fresh v4 relevance pilot authorization"
    auth_schema["properties"]["calibration_id"] = {"const": FRESH_RELEVANCE_CALIBRATION_ID}
    auth_schema["properties"]["scope"] = {"const": FRESH_RELEVANCE_SCOPE}
    auth_schema_path.write_bytes(canonical_json_bytes(auth_schema))
    raw_label_schema_path = root / "schemas/raw-label.schema.json"
    raw_label_schema = json.loads(raw_label_schema_path.read_text(encoding="utf-8"))
    raw_label_schema["title"] = "Fresh v4 blind raw leakage label"
    raw_label_schema["properties"]["rubric_version"] = {"const": FRESH_RELEVANCE_RUBRIC_VERSION}
    raw_label_schema_path.write_bytes(canonical_json_bytes(raw_label_schema))

    authorization = {
        "authorization_id": FRESH_RELEVANCE_AUTHORIZATION_ID,
        "calibration_id": FRESH_RELEVANCE_CALIBRATION_ID,
        "calibration_unrestricted_collection_authorized": True,
        "frozen_item_count": 400,
        "frozen_strata": {
            "A1-T1": {"ambiguous": 25, "keep": 38, "reject": 38},
            "B-T1": {"ambiguous": 25, "keep": 38, "reject": 38},
            "B-T2": {"ambiguous": 25, "keep": 37, "reject": 37},
            "C-T2": {"ambiguous": 25, "keep": 37, "reject": 37},
        },
        "full_paper_unrestricted_authorized": False,
        "issue_id": "5e7ed4b6-3e65-4cfa-9da8-0ea82bc31fc4",
        "issued_at_utc": "2026-08-31T08:53:46Z",
        "pilot_item_count": 40,
        "pilot_strata": {stratum: 10 for stratum in V4_STRATA},
        "same_candidates_cutoff_filter": True,
        "schema_version": V4_AUTHORIZATION_SCHEMA,
        "scope": FRESH_RELEVANCE_SCOPE,
        "source_comment_id": FRESH_RELEVANCE_SOURCE_COMMENT_ID,
        "status": "active",
    }
    _write_json(root / "authorization.json", authorization)
    cohort = json.loads((source_root / "cohort.json").read_text(encoding="utf-8"))
    cohort.update({"calibration_status": "incomplete", "cohort_id": FRESH_RELEVANCE_CALIBRATION_ID})
    _write_json(root / "cohort.json", cohort)
    report = json.loads(
        (source_root / "reports/calibration-report.json").read_text(encoding="utf-8")
    )
    report.update(
        {
            "calibration_status": "incomplete",
            "claim_eligible": False,
            "counts": {
                "adjudicated": 0,
                "determinate_binary_gold": 0,
                "items": 400,
                "leakage_negative": 0,
                "leakage_positive": 0,
                "raw_labels": 0,
                "raw_labels_a": 0,
                "raw_labels_b": 0,
                "unknown": 400,
                "unresolved_disputes": 0,
            },
            "exclusions_by_type": {"v4_fresh_pilot_not_in_400": 40},
            "filter_validated": False,
            "generated_at_utc": _now_utc(),
            "quality_gates_passed": False,
            "raw_agreement": None,
        }
    )
    _write_json(root / "reports/calibration-report.json", report)
    readme = """# Leakage calibration bundle v4

This directory is the fresh, explicitly authorized 40-item relevance/outcome
pilot. It preserves the v3 failed-gate audit and its 400-item cohort at the
top level; those top-level raw labels, adjudications, and binary leakage gold
remain empty and non-claiming.

The fresh pilot contains 10 items in each A1-T1, B-T1, B-T2, and C-T2 stratum.
It uses new bounded provider calls and applies the same-candidate cutoff filter
to each exact response. Raw requests, responses, and selected candidate
payloads are restricted to the ignored `data/calibration-cache/leakage-probe/v4/`
directory. The blind pilot view exposes only the task prompt, model-visible
title/snippet/content, and authoritative time evidence needed by the labeler.

The v4 rubric is `leakage-calibration-rubric/v3`; v3 remains an immutable
failed-gate audit. This pilot is exploratory calibration only, with
`live_unrestricted_authorized=false`, `claim_eligible=false`, and
`paper_probe_eligible=false`.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    sampling = json.loads((source_root / "sampling-plan.json").read_text(encoding="utf-8"))
    _write_json(root / "sampling-plan.json", sampling)
    _write_json(root / "artifact_manifest.json", _initial_manifest())
    (root / "CALIBRATION.sha256").write_text(
        "0" * 64 + "  artifact_manifest.json\n", encoding="utf-8"
    )


def _initial_manifest() -> dict[str, object]:
    return {
        "adjudication": {
            "adjudicated_observed": 0,
            "raw_labels_expected": 800,
            "raw_labels_observed": 0,
            "status": "incomplete",
            "unresolved_disputes": 0,
        },
        "calibration_id": FRESH_RELEVANCE_CALIBRATION_ID,
        "calibration_status": "incomplete",
        "clocks": {
            "backbone_id": V4_BACKBONE_ID,
            "backbone_training_cutoff": V4_BACKBONE_TRAINING_CUTOFF,
            "experiment_as_of": V4_EXPERIMENT_AS_OF,
            "guard_days": V4_GUARD_DAYS,
            "task_cutoff_semantics": "utc_calendar_day_inclusive_next_day_exclusive",
        },
        "files": [],
        "hub": {
            "repo_id": HUB_REPO_ID,
            "revision": "941370f228b9bc03c1b8bd9547c098c42c06be23",
        },
        "live_unrestricted_authorized": False,
        "query_corpus": {"path": "query-corpus.jsonl", "sha256": "0" * 64},
        "restricted_evidence_manifest_sha256": None,
        "rubric": {"path": "rubric.md", "sha256": "0" * 64},
        "sampling_plan": {"path": "sampling-plan.json", "sha256": "0" * 64},
        "schema_version": "leakage-calibration-manifest/v1",
    }


def _collect_candidates(
    *,
    cache_root: Path,
    dataset_cache: str | Path,
    timeout_seconds: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int, list[dict[str, object]]]:
    tasks = _load_tasks(dataset_cache)
    access_key = os.environ.get("VOLC_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("VOLC_SECRET_ACCESS_KEY", "").strip()
    if not access_key or not secret_key:
        raise FreshPilotCollectionError(
            "Volc credentials are unavailable for the authorized v4 pilot"
        )
    transport = VolcSdkTransport(access_key=access_key, secret_key=secret_key)
    candidate_pool: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    evidence_entries: list[dict[str, object]] = []
    previous_receipt_sha256: str | None = None
    task_count = 0
    for stratum in V4_STRATA:
        for row in tasks[stratum][:V4_TARGET_PER_STRATUM]:
            task_count += 1
            task_id = str(row["task_id"])
            cutoff_date = _date_text(row.get("cutoff_date"), "cutoff_date", task_id)
            forecast_origin = _date_text(row.get("forecast_origin"), "forecast_origin", task_id)
            for query_class in PILOT_QUERY_CLASSES:
                query = _build_v4_query(row, query_class, cutoff_date)
                query_rows.append(
                    {
                        "query_class": query_class,
                        "query_corpus_entry_id": sha256_text(
                            f"leakage-calibration-query/v4\0{task_id}\0{query_class}"
                        ),
                        "query_sha256": sha256_text(query),
                        "schema_version": CALIBRATION_SCHEMA_VERSION,
                        "task_id": task_id,
                        "template_version": V4_QUERY_TEMPLATE_VERSION,
                    }
                )
                call_id = _call_id(stratum, task_count, query_class)
                body = {
                    "Query": query,
                    "SearchType": "web",
                    "Count": 10,
                    "Filter": {"NeedContent": True, "NeedUrl": True},
                    "TimeRange": _unrestricted_time_range(cutoff_date),
                }
                response = _v4_post(
                    transport,
                    "https://mercury.volcengineapi.com/",
                    body,
                    timeout_seconds,
                )
                response_payload = response.payload
                provider_error = _volc_response_error(response_payload)
                response_hash = (
                    sha256_json(response_payload)
                    if response_payload is not None
                    else sha256_text(response.error or "volc_response_missing")
                )
                call_path = cache_root / "calls" / f"{call_id}.json"
                _write_cache_json(
                    call_path,
                    {
                        "authorization_id": FRESH_RELEVANCE_AUTHORIZATION_ID,
                        "call_id": call_id,
                        "collection_id": FRESH_RELEVANCE_COLLECTION_ID,
                        "query": query,
                        "request": body,
                        "response": response_payload,
                        "response_error": response.error or provider_error,
                        "response_request_id": response.request_id,
                        "response_status_code": response.status_code,
                        "schema_version": V4_CACHE_SCHEMA,
                        "task_id": task_id,
                    },
                )
                evidence_entries.append(_cache_entry(call_path, cache_root))
                if response.error or provider_error or not 200 <= (response.status_code or 0) < 300:
                    raise FreshPilotCollectionError(f"Volc v4 pilot call failed for {call_id}")
                raw_results = _volc_web_results(response_payload)
                if raw_results is None:
                    raise FreshPilotCollectionError(f"Volc v4 response shape invalid for {call_id}")
                receipt_payload = {
                    "call_id": call_id,
                    "candidate_count": len(raw_results),
                    "collection_id": FRESH_RELEVANCE_COLLECTION_ID,
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
                    filtered = filter_volc_candidates(
                        (candidate,), cutoff_date, llm_filter=None
                    ).decisions[0]
                    canonical_url = _canonical_url(candidate.url)
                    if canonical_url is None or not _candidate_text(candidate):
                        continue
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
    query_rows.sort(key=lambda row: row["query_corpus_entry_id"])
    return candidate_pool, query_rows, task_count, evidence_entries


_V4_REQUEST_LOCK = Lock()
_V4_NEXT_REQUEST_AT = 0.0
_V4_REQUEST_INTERVAL_SECONDS = 1.0


def _v4_post(
    transport: VolcSdkTransport,
    endpoint: str,
    body: Mapping[str, object],
    timeout_seconds: float,
) -> VolcHttpResponse:
    """Throttle the bounded probe and retry provider rate-limit responses."""

    global _V4_NEXT_REQUEST_AT
    for attempt in range(3):
        with _V4_REQUEST_LOCK:
            delay = _V4_NEXT_REQUEST_AT - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            _V4_NEXT_REQUEST_AT = time.monotonic() + _V4_REQUEST_INTERVAL_SECONDS
        response = transport.post(
            endpoint,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout_seconds=timeout_seconds,
        )
        error_code = _volc_response_error(response.payload)
        if error_code != "volc_error_rate_limit_exceeded" or attempt == 2:
            return response
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _build_v4_query(row: Mapping[str, object], query_class: str, cutoff_date: str) -> str:
    base = _build_query(row, query_class, cutoff_date)
    query = f"{base} fresh relevance calibration v4".strip()
    if len(query) > 512:
        raise FreshPilotCollectionError("generated v4 query exceeded the bounded query contract")
    return query


def _select_items(candidate_pool: list[dict[str, object]], root: Path) -> list[dict[str, object]]:
    prior_ids = _prior_item_ids(root)
    by_stratum: dict[str, list[dict[str, object]]] = {stratum: [] for stratum in V4_STRATA}
    for candidate in candidate_pool:
        item_id = _provisional_item_id(candidate)
        if item_id in prior_ids:
            continue
        candidate = dict(candidate)
        candidate["provisional_item_id"] = item_id
        by_stratum[str(candidate["stratum"])].append(candidate)

    selected: list[dict[str, object]] = []
    for stratum in V4_STRATA:
        pool = by_stratum[stratum]
        pool.sort(key=lambda candidate: _sampling_key(str(candidate["provisional_item_id"])))
        domains: Counter[str] = Counter()
        task_counts: Counter[str] = Counter()
        chosen = 0
        for rank, candidate in enumerate(pool, start=1):
            domain = urlsplit(str(candidate["canonical_url"])).hostname or ""
            task_id = str(candidate["task_id"])
            if domains[domain] >= 20 or task_counts[task_id] >= 8:
                continue
            picked = dict(candidate)
            picked["deterministic_rank"] = rank
            picked["inclusion_probability"] = min(1.0, V4_TARGET_PER_STRATUM / max(1, len(pool)))
            selected.append(picked)
            domains[domain] += 1
            task_counts[task_id] += 1
            chosen += 1
            if chosen == V4_TARGET_PER_STRATUM:
                break
        if chosen != V4_TARGET_PER_STRATUM:
            raise FreshPilotCollectionError(
                f"v4 stratum {stratum} has fewer than 10 cap-compliant fresh items"
            )
    return selected


def _prior_item_ids(root: Path) -> set[str]:
    prior: set[str] = set()
    for relative in ("items.jsonl", "pilot/items.jsonl"):
        path = root.parent / "v3" / relative
        if path.exists():
            prior.update(str(row["item_id"]) for row in _read_jsonl(path))
    source_v1 = root.parent / "v1/items.jsonl"
    if source_v1.exists():
        prior.update(str(row["item_id"]) for row in _read_jsonl(source_v1))
    source_v2 = root.parent / "v2/items.jsonl"
    if source_v2.exists():
        prior.update(str(row["item_id"]) for row in _read_jsonl(source_v2))
    return prior


def _materialize_items(
    selected: list[dict[str, object]],
    *,
    cache_root: Path,
    evidence_entries: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    captured_at = _now_utc()
    item_rows: list[dict[str, object]] = []
    filter_rows: list[dict[str, object]] = []
    for record in selected:
        candidate: VolcSearchCandidate = record["candidate"]  # type: ignore[assignment]
        raw_url = candidate.url
        if raw_url is None:
            raise FreshPilotCollectionError("selected v4 candidate has no URL")
        item_id = str(record["provisional_item_id"])
        candidate_path = cache_root / "candidates" / f"{item_id}.json"
        _write_cache_json(
            candidate_path,
            {
                "candidate": record["raw_candidate"],
                "call_id": record["call_id"],
                "candidate_index": record["candidate_index"],
                "collection_id": FRESH_RELEVANCE_COLLECTION_ID,
                "query_sha256": record["query_sha256"],
                "schema_version": V4_CACHE_SCHEMA,
                "task_id": record["task_id"],
            },
        )
        evidence_entries.append(_cache_entry(candidate_path, cache_root))
        canonical_url = str(record["canonical_url"])
        cutoff_date = str(record["cutoff_date"])
        task_id = str(record["task_id"])
        payload_sha256 = _sha256_canonical(candidate.as_model_payload())
        item_rows.append(
            {
                "backbone_id": V4_BACKBONE_ID,
                "backbone_training_cutoff": V4_BACKBONE_TRAINING_CUTOFF,
                "call_id": record["call_id"],
                "candidate_index": record["candidate_index"],
                "canonical_url": canonical_url,
                "canonical_url_sha256": sha256_text(canonical_url),
                "canonical_url_version": "url-canonicalizer/v1",
                "category": record["category"],
                "collection_id": FRESH_RELEVANCE_COLLECTION_ID,
                "content_sha256": sha256_text(candidate.content or ""),
                "cutoff_declaration_sha256": sha256_json(
                    {
                        "backbone_training_cutoff": V4_BACKBONE_TRAINING_CUTOFF,
                        "cutoff_date": cutoff_date,
                        "experiment_as_of": V4_EXPERIMENT_AS_OF,
                        "guard_days": V4_GUARD_DAYS,
                        "task_id": task_id,
                    }
                ),
                "deterministic_rank": record["deterministic_rank"],
                "domain": urlsplit(canonical_url).hostname,
                "exclusive_cutoff_bound": CutoffWindow.from_seed(cutoff_date).exclusive_bound,
                "experiment_as_of": V4_EXPERIMENT_AS_OF,
                "forecast_origin": record["forecast_origin"],
                "hub_repo_id": HUB_REPO_ID,
                "hub_revision": "941370f228b9bc03c1b8bd9547c098c42c06be23",
                "inclusion_probability": record["inclusion_probability"],
                "item_id": item_id,
                "model_visible_payload_sha256": payload_sha256,
                "original_url_sha256": sha256_text(raw_url),
                "paper_band": record["paper_band"],
                "policy_sha256": V4_POLICY_SHA256,
                "previous_receipt_sha256": record["previous_receipt_sha256"],
                "provider_id": "volc",
                "provider_published_at_utc": _raw_provider_time(
                    record["raw_candidate"], "PublishTime"
                ),
                "provider_updated_at_utc": record["updated_at_utc"],
                "query_sha256": record["query_sha256"],
                "receipt_protocol": "search_receipt/v1",
                "receipt_sha256": record["receipt_sha256"],
                "request_filter_sha256": record["request_filter_sha256"],
                "response_sha256": record["response_sha256"],
                "restricted_snapshot_bytes": candidate_path.stat().st_size,
                "restricted_snapshot_locator": canonical_url,
                "restricted_snapshot_mime": "application/json",
                "restricted_snapshot_sha256": _sha256_file(candidate_path),
                "restricted_snapshot_status": "available",
                "retrieved_at_utc": captured_at,
                "sampling_stratum": record["stratum"],
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "service_cohort_sha256": V4_SERVICE_COHORT_SHA256,
                "snapshot_captured_at_utc": captured_at,
                "snippet_sha256": sha256_text(candidate.snippet or ""),
                "task_cutoff_date": cutoff_date,
                "task_id": task_id,
                "temporal_index_sha256": V4_TEMPORAL_INDEX_SHA256,
                "title_sha256": sha256_text(candidate.title or ""),
                "tool_contract_sha256": DEFAULT_TOOL_CONTRACT.hash,
            }
        )
        filter_rows.append(
            {
                "decided_at_utc": captured_at,
                "filter_policy_sha256": PILOT_FILTER_POLICY_SHA256,
                "filter_state": record["filter_state"],
                "filter_verdict": record["filter_verdict"],
                "item_id": item_id,
                "schema_version": CALIBRATION_SCHEMA_VERSION,
            }
        )
    item_rows.sort(key=lambda row: row["item_id"])
    filter_rows.sort(key=lambda row: row["item_id"])
    return item_rows, filter_rows


def _label_v4_pilot(root: Path, *, reviewer_id: str = "dev-labeler-a-v4-fresh40") -> int:
    items = _read_jsonl(root / "pilot/items.jsonl")
    views = {row["item_id"]: row for row in _read_jsonl(root / "pilot/annotation-view.jsonl")}
    rubric_sha256 = _sha256_file(root / "rubric.md")
    labels: list[dict[str, object]] = []
    for item in items:
        item_id = str(item["item_id"])
        view = views.get(item_id)
        if view is None:
            raise FreshPilotCollectionError("v4 pilot annotation view is incomplete")
        temporal_label, evidence, _ = _blind_temporal_label(item, view)
        relevance, outcome, rationale = label_relevance_outcome(view)
        confidence = (
            0.25 if relevance == "indeterminate" else (0.7 if outcome == "indeterminate" else 0.85)
        )
        labels.append(
            {
                "annotation_view_sha256": view["annotation_view_sha256"],
                "confidence": confidence,
                "evidence": evidence,
                "item_id": item_id,
                "labeled_at_utc": _now_utc(),
                "outcome_revealing": outcome,
                "rationale": rationale[:500],
                "reviewer_id": reviewer_id,
                "reviewer_type": "labeler_a",
                "rubric_sha256": rubric_sha256,
                "rubric_version": FRESH_RELEVANCE_RUBRIC_VERSION,
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "task_relevance": relevance,
                "temporal_label": temporal_label,
            }
        )
    labels.sort(key=lambda row: row["item_id"])
    _write_jsonl(root / "pilot/labels/raw/annotator-a.jsonl", labels)
    return len(labels)


def _validate_v4_pilot_state(root: Path, *, expected_labels: int) -> None:
    result = validate_calibration_root(root)
    if result["calibration_id"] != FRESH_RELEVANCE_CALIBRATION_ID:
        raise FreshPilotCollectionError("v4 pilot validation selected the wrong calibration")
    if result["counts"]["items"] != 400:
        raise FreshPilotCollectionError("v4 top-level 400-item audit changed unexpectedly")
    labels = _read_jsonl(root / "pilot/labels/raw/annotator-a.jsonl")
    if len(labels) != expected_labels:
        raise FreshPilotCollectionError("v4 Labeler A output count is not at the requested stage")


def _write_v4_evidence_manifest(cache_root: Path, entries: list[dict[str, object]]) -> None:
    _write_cache_json(
        cache_root / "restricted-evidence-manifest.json",
        {
            "authorization_id": FRESH_RELEVANCE_AUTHORIZATION_ID,
            "calibration_id": FRESH_RELEVANCE_CALIBRATION_ID,
            "collection_id": FRESH_RELEVANCE_COLLECTION_ID,
            "entries": sorted(entries, key=lambda entry: str(entry["path"])),
            "schema_version": "leakage-calibration-restricted-evidence/v1",
        },
    )


def _refresh_v4_state(root: Path, evidence_sha256: str) -> None:
    cohort = json.loads((root / "cohort.json").read_text(encoding="utf-8"))
    cohort["calibration_status"] = "incomplete"
    _write_json(root / "cohort.json", cohort)
    labels_a = _read_jsonl(root / "labels/raw/annotator-a.jsonl")
    labels_b = _read_jsonl(root / "labels/raw/annotator-b.jsonl")
    adjudicated = _read_jsonl(root / "labels/adjudicated.jsonl")
    if labels_a or labels_b or adjudicated:
        raise FreshPilotCollectionError("v4 top-level audit labels must remain empty")
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "calibration_status": "incomplete",
            "restricted_evidence_manifest_sha256": evidence_sha256,
            "adjudication": {
                "adjudicated_observed": 0,
                "raw_labels_expected": 800,
                "raw_labels_observed": 0,
                "status": "incomplete",
                "unresolved_disputes": 0,
            },
        }
    )
    manifest["rubric"]["sha256"] = _sha256_file(root / "rubric.md")
    manifest["sampling_plan"]["sha256"] = _sha256_file(root / "sampling-plan.json")
    manifest["query_corpus"]["sha256"] = _sha256_file(root / "query-corpus.jsonl")
    files: list[dict[str, object]] = []
    for relative in FRESH_RELEVANCE_TRACKED_FILES:
        path = root / relative
        payload = path.read_bytes()
        row_count: int | None
        if relative in V4_JSONL_FILES:
            row_count = len([line for line in payload.decode("utf-8").splitlines() if line.strip()])
        elif relative.endswith(".json"):
            row_count = 1
        else:
            row_count = None
        files.append(
            {
                "bytes": len(payload),
                "path": relative,
                "row_count": row_count,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest["files"] = files
    manifest_bytes = canonical_json_bytes(manifest)
    (root / "artifact_manifest.json").write_bytes(manifest_bytes)
    (root / "CALIBRATION.sha256").write_text(
        hashlib.sha256(manifest_bytes).hexdigest() + "  artifact_manifest.json\n",
        encoding="utf-8",
    )


def _pilot_has_rows(root: Path) -> bool:
    return any(_read_jsonl(root / relative) for relative in FRESH_RELEVANCE_PILOT_JSONL_FILES)


def _call_id(stratum: str, task_number: int, query_class: str) -> str:
    value = f"{_CALL_PREFIX}-{stratum.lower().replace('-', '')}-{task_number:02d}-{query_class}"
    if any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in value
    ):
        raise FreshPilotCollectionError("v4 call identity is unsafe")
    return value


def _candidate_text(candidate: VolcSearchCandidate) -> str:
    return "\n".join(
        value.strip()
        for value in (candidate.title, candidate.snippet, candidate.content)
        if isinstance(value, str) and value.strip()
    )


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


_V4_RUBRIC = """# Retrieval leakage rubric v3

## Scope and blinding

This is the fresh v4 relevance/outcome pilot. A labeler may use only the
`task_prompt` and bounded `model_visible_title`, `model_visible_snippet`, and
`model_visible_content`. URL, publication/update time, temporal labels,
filter verdicts, scores, answers, ground truth, system identity, and the other
labeler's record are not evidence for relevance or outcome. Temporal labels are
recorded separately from authoritative evidence and do not change relevance.

## Mandatory decision tree

Apply these rules in order:

1. If the complete visible payload gives the family-specific requested
   outcome, label `relevant` and `direct`.
2. Otherwise apply the family relevance and outcome rules below.
3. If material truncation leaves the entity, event, period, or metric
   connection unresolved, label `indeterminate/indeterminate`.
4. Use `not_relevant/no` only when visible text proves a different subject,
   event, period, or metric and the missing tail could not change that proof.
5. A publication cutoff is temporal evidence only. It never changes relevance
   or outcome by itself.

Outcome is conditional: `not_relevant` always maps to `no`, and
`indeterminate` always maps to `indeterminate`.

## Family rules

### A1 valuation

Substantive same-security financial results, valuation multiples, capital
structure, dividend/capital action, trading volume, observed price, or an
explicit valuation driver are relevant. New inputs without a matching
target/fair-value price are `indirect`; a matching target/fair-value price or
range is `direct`. A profile, navigation page, fact-free report link, or only
the task's existing price is `not_relevant/no`.

### B direction

The same issuer and reporting episode/fiscal period are relevant even when the
page reports revenue, profit, EPS, book value, or yield instead of the task's
exact wording. Same-event financial facts, profit/revenue/EPS changes,
dividends, records, and ordinary guidance are `no` unless a security price or
return bridge is visible. `direct` requires the security's price/return move
to be explicitly tied to the named event/window. `indirect` requires an
explicit analyst, investor, rating, target, or market judgment tied to that
security; an annotator's general belief that good results may move price is
not enough.

### C metric forecast

The same company and target metric are relevant in any visible period. A
clearly named arithmetic component or financial driver is also relevant.
The exact target metric and target period fully present in the payload is
`direct`; another period, annual/TTM/cumulative value, guidance, consensus,
or explicit driver is `indirect`. Do not subtract a task-input number from a
page number across sources to manufacture `direct`. A different metric with
no explicit accounting/driver link is `not_relevant/no`.

## Truncation and rationale

Do not treat a missing tail as a mismatch. A complete decisive visible
connection takes precedence, but a material truncation that could change the
connection is `indeterminate/indeterminate`.

Every raw label uses:

`R=<entity_event_metric|explicit_driver|task_input_repeat|mismatch|unclear>;
O=<realized_exact|deterministic_equivalent|predictive_signal|no_new_signal|unclear>;
span=<short paraphrase>`

The 40-item pilot is exploratory calibration only. It is disjoint from the
v3 failed-gate pilot and the 400-item cohort. The top-level 400 raw labels,
adjudications, binary gold, and leakage positives/negatives remain empty.
`ranking_eligible=false`, `claim_eligible=false`, and
`live_unrestricted_authorized=false` remain in force.
"""


__all__ = [
    "FreshPilotCollectionError",
    "FreshPilotResult",
    "RUBRIC_VERSION",
    "run_fresh_pilot",
]
