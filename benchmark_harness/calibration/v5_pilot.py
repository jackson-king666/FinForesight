"""Authorized v5 rubric-v4 relevance/outcome pilot collection.

This module owns only the separately authorized 40-item calibration pilot. It
uses the existing provider transport and sealed same-candidate cutoff filter,
keeps raw requests/responses and selected candidates below the ignored cache,
and never writes the v5 top-level 400-item labels or adjudications.
"""

from __future__ import annotations

import hashlib
import json
import os
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
from benchmark_harness.calibration.relevance_v5 import label_relevance_outcome
from benchmark_harness.calibration.v5_scaffold import (
    DEFAULT_V5_ROOT,
    prepare_v5_scaffold,
)
from benchmark_harness.calibration.validator import (
    CALIBRATION_SCHEMA_VERSION,
    HUB_REPO_ID,
    HUB_REVISION,
    V5_JSON_FILES,
    V5_JSONL_FILES,
    V5_PILOT_JSONL_FILES,
    V5_RELEVANCE_AUTHORIZATION_ID,
    V5_RELEVANCE_CALIBRATION_ID,
    V5_RELEVANCE_COLLECTION_ID,
    V5_RELEVANCE_RUBRIC_VERSION,
    V5_TRACKED_FILES,
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

V5_STRATA = PILOT_STRATA
V5_ITEM_COUNT = 40
V5_TARGET_PER_STRATUM = 10
V5_CACHE_SCHEMA = "leakage-calibration-cache/v1"
V5_AUTHORIZATION_SCHEMA = "leakage-calibration-authorization/v1"
V5_QUERY_TEMPLATE_VERSION = "pilot-query/v5"
V5_FILTER_SCHEMA = "cutoff-filter/v1"
V5_TEMPORAL_INDEX_SHA256 = "80d977e40e113172e778b6f78315e1fad132423c9fe9eca3ad6a5de34df4f21b"
V5_BACKBONE_ID = "openai:gpt-4.1"
V5_BACKBONE_TRAINING_CUTOFF = "2024-06-30"
V5_EXPERIMENT_AS_OF = "2026-08-17"
V5_GUARD_DAYS = 30
V5_POLICY = {
    "calibration_id": V5_RELEVANCE_CALIBRATION_ID,
    "ranking_eligible": False,
    "regime": "unrestricted_current_web",
    "scope": "calibration_relevance_rubric_v4_fresh_pilot",
    "schema_version": "leakage_probe/v5",
}
V5_POLICY_SHA256 = sha256_json(V5_POLICY)
V5_SERVICE_COHORT_SHA256 = sha256_json(
    {
        "calibration_id": V5_RELEVANCE_CALIBRATION_ID,
        "hub_revision": HUB_REVISION,
        "provider_id": "volc",
        "tool_contract_sha256": DEFAULT_TOOL_CONTRACT.hash,
    }
)
_CALL_PREFIX = "v5pilot40"


class V5PilotCollectionError(PilotCollectionError):
    """Raised when the authorized v5 pilot cannot be collected safely."""


@dataclass(frozen=True, slots=True)
class V5PilotResult:
    """Safe summary returned after v5 collection and Labeler-A labeling."""

    collection_id: str
    task_count: int
    query_count: int
    candidate_pool_count: int
    item_count: int
    label_a_count: int
    calibration_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "calibration_id": V5_RELEVANCE_CALIBRATION_ID,
            "calibration_sha256": self.calibration_sha256,
            "candidate_pool_count": self.candidate_pool_count,
            "collection_id": self.collection_id,
            "item_count": self.item_count,
            "label_a_count": self.label_a_count,
            "query_count": self.query_count,
            "task_count": self.task_count,
        }


def run_v5_pilot(
    *,
    root: str | Path = DEFAULT_V5_ROOT,
    cache_root: str | Path = "data/calibration-cache/leakage-probe/v5",
    dataset_cache: str | Path = "data/.huggingface",
    timeout_seconds: float = 30.0,
) -> V5PilotResult:
    """Collect exactly the newly authorized v5 40-item pilot and label A."""

    root_path = Path(root).expanduser().resolve()
    cache_path = Path(cache_root).expanduser().resolve()
    try:
        prepare_v5_scaffold(root_path)
    except ValueError as exc:
        raise V5PilotCollectionError("v5 calibration scaffold is unavailable") from exc
    if timeout_seconds <= 0:
        raise V5PilotCollectionError("v5 pilot timeout must be positive")
    cache_path.mkdir(parents=True, exist_ok=True)
    _private_directory(cache_path)
    try:
        state = validate_calibration_root(root_path)
    except ValueError as exc:
        raise V5PilotCollectionError("v5 scaffold failed validation before collection") from exc
    if state.get("calibration_id") != V5_RELEVANCE_CALIBRATION_ID:
        raise V5PilotCollectionError("unexpected calibration version for v5 pilot")
    if _pilot_has_rows(root_path):
        try:
            validate_calibration_root(root_path)
        except ValueError as exc:
            raise V5PilotCollectionError("existing v5 pilot corpus failed validation") from exc
        labels = _read_jsonl(root_path / "pilot/labels/raw/annotator-a.jsonl")
        if labels:
            raise V5PilotCollectionError("v5 pilot Labeler A output already exists")
        evidence_sha = _manifest_evidence_sha(root_path)
        label_count = _label_v5_pilot(root_path)
        _refresh_v5_state(root_path, evidence_sha, pilot_n=label_count)
        validated = validate_calibration_root(root_path)
        if label_count != V5_ITEM_COUNT:
            raise V5PilotCollectionError("v5 pilot Labeler A output is incomplete")
        query_rows = _read_jsonl(root_path / "pilot/query-corpus.jsonl")
        item_rows = _read_jsonl(root_path / "pilot/items.jsonl")
        return V5PilotResult(
            collection_id=V5_RELEVANCE_COLLECTION_ID,
            task_count=len({str(row["task_id"]) for row in query_rows}),
            query_count=len(query_rows),
            candidate_pool_count=0,
            item_count=len(item_rows),
            label_a_count=label_count,
            calibration_sha256=validated["calibration_sha256"],
        )
    candidate_pool, query_rows, task_count, evidence_entries = _collect_candidates(
        cache_root=cache_path,
        dataset_cache=dataset_cache,
        timeout_seconds=timeout_seconds,
    )
    selected = _select_items(candidate_pool, root_path)
    if len(selected) != V5_ITEM_COUNT:
        raise V5PilotCollectionError("v5 selection did not produce exactly 40 items")
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
    evidence_sha = _write_evidence_manifest(cache_path, evidence_entries)
    _refresh_v5_state(root_path, evidence_sha, pilot_n=0)
    _validate_v5_pilot_state(root_path, expected_labels=0)
    label_count = _label_v5_pilot(root_path)
    _refresh_v5_state(root_path, evidence_sha, pilot_n=label_count)
    validated = validate_calibration_root(root_path)
    if validated["counts"]["items"] != 400 or label_count != V5_ITEM_COUNT:
        raise V5PilotCollectionError("v5 top-level audit or pilot Labeler A output is incomplete")
    return V5PilotResult(
        collection_id=V5_RELEVANCE_COLLECTION_ID,
        task_count=task_count,
        query_count=len(query_rows),
        candidate_pool_count=len(candidate_pool),
        item_count=V5_ITEM_COUNT,
        label_a_count=label_count,
        calibration_sha256=validated["calibration_sha256"],
    )


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
        raise V5PilotCollectionError("Volc credentials are unavailable for the authorized v5 pilot")
    transport = VolcSdkTransport(access_key=access_key, secret_key=secret_key)
    candidate_pool: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    evidence_entries: list[dict[str, object]] = []
    previous_receipt_sha256: str | None = None
    task_count = 0
    for stratum in V5_STRATA:
        for row in tasks[stratum][:V5_TARGET_PER_STRATUM]:
            task_count += 1
            task_id = str(row["task_id"])
            cutoff_date = _date_text(row.get("cutoff_date"), "cutoff_date", task_id)
            forecast_origin = _date_text(row.get("forecast_origin"), "forecast_origin", task_id)
            for query_class in PILOT_QUERY_CLASSES:
                query = _build_v5_query(row, query_class, cutoff_date)
                query_rows.append(
                    {
                        "query_class": query_class,
                        "query_corpus_entry_id": sha256_text(
                            f"leakage-calibration-query/v5\0{task_id}\0{query_class}"
                        ),
                        "query_sha256": sha256_text(query),
                        "schema_version": CALIBRATION_SCHEMA_VERSION,
                        "task_id": task_id,
                        "template_version": V5_QUERY_TEMPLATE_VERSION,
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
                response = _v5_post(
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
                        "authorization_id": V5_RELEVANCE_AUTHORIZATION_ID,
                        "call_id": call_id,
                        "collection_id": V5_RELEVANCE_COLLECTION_ID,
                        "query": query,
                        "request": body,
                        "response": response_payload,
                        "response_error": response.error or provider_error,
                        "response_request_id": response.request_id,
                        "response_status_code": response.status_code,
                        "schema_version": V5_CACHE_SCHEMA,
                        "task_id": task_id,
                    },
                )
                evidence_entries.append(_cache_entry(call_path, cache_root))
                if response.error or provider_error or not 200 <= (response.status_code or 0) < 300:
                    raise V5PilotCollectionError(f"Volc v5 pilot call failed for {call_id}")
                raw_results = _volc_web_results(response_payload)
                if raw_results is None:
                    raise V5PilotCollectionError(f"Volc v5 response shape invalid for {call_id}")
                receipt_payload = {
                    "call_id": call_id,
                    "candidate_count": len(raw_results),
                    "collection_id": V5_RELEVANCE_COLLECTION_ID,
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


_V5_REQUEST_LOCK = Lock()
_V5_NEXT_REQUEST_AT = 0.0
_V5_REQUEST_INTERVAL_SECONDS = 1.0


def _v5_post(
    transport: VolcSdkTransport,
    endpoint: str,
    body: Mapping[str, object],
    timeout_seconds: float,
) -> VolcHttpResponse:
    """Throttle the bounded probe and retry provider rate-limit responses."""

    global _V5_NEXT_REQUEST_AT
    for attempt in range(3):
        with _V5_REQUEST_LOCK:
            delay = _V5_NEXT_REQUEST_AT - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            _V5_NEXT_REQUEST_AT = time.monotonic() + _V5_REQUEST_INTERVAL_SECONDS
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


def _build_v5_query(row: Mapping[str, object], query_class: str, cutoff_date: str) -> str:
    base = _build_query(row, query_class, cutoff_date)
    query = f"{base} fresh relevance calibration v5".strip()
    if len(query) > 512:
        raise V5PilotCollectionError("generated v5 query exceeded the bounded query contract")
    return query


def _select_items(candidate_pool: list[dict[str, object]], root: Path) -> list[dict[str, object]]:
    prior_ids = _prior_item_ids(root)
    by_stratum: dict[str, list[dict[str, object]]] = {stratum: [] for stratum in V5_STRATA}
    seen_ids: set[str] = set()
    for candidate in candidate_pool:
        item_id = _provisional_item_id(candidate)
        if item_id in prior_ids or item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        record = dict(candidate)
        record["provisional_item_id"] = item_id
        by_stratum[str(record["stratum"])].append(record)

    selected: list[dict[str, object]] = []
    for stratum in V5_STRATA:
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
            picked["inclusion_probability"] = min(1.0, V5_TARGET_PER_STRATUM / max(1, len(pool)))
            selected.append(picked)
            domains[domain] += 1
            task_counts[task_id] += 1
            chosen += 1
            if chosen == V5_TARGET_PER_STRATUM:
                break
        if chosen != V5_TARGET_PER_STRATUM:
            raise V5PilotCollectionError(
                f"v5 stratum {stratum} has fewer than 10 cap-compliant fresh items"
            )
    return selected


def _prior_item_ids(root: Path) -> set[str]:
    prior: set[str] = set()
    for version in ("v1", "v2", "v3", "v4", "v5"):
        for relative in ("items.jsonl", "pilot/items.jsonl"):
            path = root.parent / version / relative
            if path.exists():
                prior.update(str(row["item_id"]) for row in _read_jsonl(path))
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
            raise V5PilotCollectionError("selected v5 candidate has no URL")
        item_id = str(record["provisional_item_id"])
        candidate_path = cache_root / "candidates" / f"{item_id}.json"
        _write_cache_json(
            candidate_path,
            {
                "candidate": record["raw_candidate"],
                "call_id": record["call_id"],
                "candidate_index": record["candidate_index"],
                "collection_id": V5_RELEVANCE_COLLECTION_ID,
                "query_sha256": record["query_sha256"],
                "schema_version": V5_CACHE_SCHEMA,
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
                "backbone_id": V5_BACKBONE_ID,
                "backbone_training_cutoff": V5_BACKBONE_TRAINING_CUTOFF,
                "call_id": record["call_id"],
                "candidate_index": record["candidate_index"],
                "canonical_url": canonical_url,
                "canonical_url_sha256": sha256_text(canonical_url),
                "canonical_url_version": "url-canonicalizer/v1",
                "category": record["category"],
                "collection_id": V5_RELEVANCE_COLLECTION_ID,
                "content_sha256": sha256_text(candidate.content or ""),
                "cutoff_declaration_sha256": sha256_json(
                    {
                        "backbone_training_cutoff": V5_BACKBONE_TRAINING_CUTOFF,
                        "cutoff_date": cutoff_date,
                        "experiment_as_of": V5_EXPERIMENT_AS_OF,
                        "guard_days": V5_GUARD_DAYS,
                        "task_id": task_id,
                    }
                ),
                "deterministic_rank": record["deterministic_rank"],
                "domain": urlsplit(canonical_url).hostname,
                "exclusive_cutoff_bound": CutoffWindow.from_seed(cutoff_date).exclusive_bound,
                "experiment_as_of": V5_EXPERIMENT_AS_OF,
                "forecast_origin": record["forecast_origin"],
                "hub_repo_id": HUB_REPO_ID,
                "hub_revision": HUB_REVISION,
                "inclusion_probability": record["inclusion_probability"],
                "item_id": item_id,
                "model_visible_payload_sha256": payload_sha256,
                "original_url_sha256": sha256_text(raw_url),
                "paper_band": record["paper_band"],
                "policy_sha256": V5_POLICY_SHA256,
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
                "service_cohort_sha256": V5_SERVICE_COHORT_SHA256,
                "snapshot_captured_at_utc": captured_at,
                "snippet_sha256": sha256_text(candidate.snippet or ""),
                "task_cutoff_date": cutoff_date,
                "task_id": task_id,
                "temporal_index_sha256": V5_TEMPORAL_INDEX_SHA256,
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


def _label_v5_pilot(root: Path, *, reviewer_id: str = "dev-labeler-a-v5-fresh40") -> int:
    items = _read_jsonl(root / "pilot/items.jsonl")
    views = {row["item_id"]: row for row in _read_jsonl(root / "pilot/annotation-view.jsonl")}
    rubric_sha256 = _sha256_file(root / "rubric.md")
    labels: list[dict[str, object]] = []
    for item in items:
        item_id = str(item["item_id"])
        view = views.get(item_id)
        if view is None:
            raise V5PilotCollectionError("v5 pilot annotation view is incomplete")
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
                "rubric_version": V5_RELEVANCE_RUBRIC_VERSION,
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "task_relevance": relevance,
                "temporal_label": temporal_label,
            }
        )
    labels.sort(key=lambda row: row["item_id"])
    _write_jsonl(root / "pilot/labels/raw/annotator-a.jsonl", labels)
    return len(labels)


def _validate_v5_pilot_state(root: Path, *, expected_labels: int) -> None:
    result = validate_calibration_root(root)
    if result.get("calibration_id") != V5_RELEVANCE_CALIBRATION_ID:
        raise V5PilotCollectionError("v5 pilot validation selected the wrong calibration")
    if result["counts"]["items"] != 400:
        raise V5PilotCollectionError("v5 top-level 400-item audit changed unexpectedly")
    labels = _read_jsonl(root / "pilot/labels/raw/annotator-a.jsonl")
    if len(labels) != expected_labels:
        raise V5PilotCollectionError("v5 Labeler A output count is not at the requested stage")


def _write_evidence_manifest(cache_root: Path, entries: list[dict[str, object]]) -> str:
    path = cache_root / "restricted-evidence-manifest.json"
    _write_cache_json(
        path,
        {
            "authorization_id": V5_RELEVANCE_AUTHORIZATION_ID,
            "calibration_id": V5_RELEVANCE_CALIBRATION_ID,
            "collection_id": V5_RELEVANCE_COLLECTION_ID,
            "entries": sorted(entries, key=lambda entry: str(entry["path"])),
            "schema_version": "leakage-calibration-restricted-evidence/v1",
        },
    )
    return _sha256_file(path)


def _manifest_evidence_sha(root: Path) -> str:
    try:
        manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V5PilotCollectionError("v5 artifact manifest is unavailable") from exc
    value = manifest.get("restricted_evidence_manifest_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise V5PilotCollectionError("v5 restricted evidence manifest hash is unavailable")
    return value


def _refresh_v5_state(root: Path, evidence_sha256: str, *, pilot_n: int) -> None:
    cohort = json.loads((root / "cohort.json").read_text(encoding="utf-8"))
    cohort["calibration_status"] = "incomplete"
    (root / "cohort.json").write_bytes(canonical_json_bytes(cohort))
    top_labels_a = _read_jsonl(root / "labels/raw/annotator-a.jsonl")
    top_labels_b = _read_jsonl(root / "labels/raw/annotator-b.jsonl")
    top_adjudicated = _read_jsonl(root / "labels/adjudicated.jsonl")
    if top_labels_a or top_labels_b or top_adjudicated:
        raise V5PilotCollectionError("v5 top-level audit labels must remain empty")
    report_path = root / "reports/calibration-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "calibration_status": "incomplete",
            "claim_eligible": False,
            "cohen_kappa": None,
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
            "exclusions_by_type": {"v5_active_pilot": V5_ITEM_COUNT},
            "filter_validated": False,
            "generated_at_utc": _now_utc(),
            "live_unrestricted_authorized": False,
            "pilot_n": pilot_n,
            "pilot_status": "pilot",
            "quality_gates_passed": False,
            "raw_agreement": None,
        }
    )
    report_path.write_bytes(canonical_json_bytes(report))
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "calibration_status": "incomplete",
            "restricted_evidence_manifest_sha256": evidence_sha256,
            "adjudication": {
                "adjudicated_observed": 0,
                "raw_labels_expected": 800,
                "raw_labels_observed": 0,
                "status": "pilot",
                "unresolved_disputes": 0,
            },
        }
    )
    manifest["rubric"]["sha256"] = _sha256_file(root / "rubric.md")
    manifest["sampling_plan"]["sha256"] = _sha256_file(root / "sampling-plan.json")
    manifest["query_corpus"]["sha256"] = _sha256_file(root / "query-corpus.jsonl")
    entries: list[dict[str, object]] = []
    for relative in V5_TRACKED_FILES:
        path = root / relative
        payload = path.read_bytes()
        if relative in V5_JSONL_FILES:
            row_count: int | None = len(payload.decode("utf-8").splitlines())
        elif relative in V5_JSON_FILES:
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


def _pilot_has_rows(root: Path) -> bool:
    return any(_read_jsonl(root / relative) for relative in V5_PILOT_JSONL_FILES)


def _call_id(stratum: str, task_number: int, query_class: str) -> str:
    value = f"{_CALL_PREFIX}-{stratum.lower().replace('-', '')}-{task_number:02d}-{query_class}"
    safe_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if any(character not in safe_characters for character in value):
        raise V5PilotCollectionError("v5 call identity is unsafe")
    return value


def _candidate_text(candidate: VolcSearchCandidate) -> str:
    return "\n".join(
        value.strip()
        for value in (candidate.title, candidate.snippet, candidate.content)
        if isinstance(value, str) and value.strip()
    )


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


__all__ = ["V5PilotCollectionError", "V5PilotResult", "run_v5_pilot"]
