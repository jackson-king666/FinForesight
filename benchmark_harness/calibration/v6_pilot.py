"""Authorized v6/rubric-v5 40-item pilot collection and Labeler-A pass."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

from benchmark_harness.calibration.pilot import (
    PILOT_FILTER_POLICY_SHA256,
    PILOT_QUERY_CLASSES,
    PILOT_STRATA,
    PilotCollectionError,
    _build_query,
    _cache_entry,
    _canonical_url,
    _date_text,
    _filter_state,
    _load_tasks,
    _now_utc,
    _read_jsonl,
    _sampling_key,
    _unrestricted_time_range,
    _write_cache_json,
    _write_jsonl,
)
from benchmark_harness.calibration.relevance_v6 import classify_v6
from benchmark_harness.calibration.v6_scaffold import DEFAULT_V6_ROOT, prepare_v6_scaffold
from benchmark_harness.calibration.v6_scope import TaskScopeError, build_task_scope
from benchmark_harness.calibration.validator import (
    CALIBRATION_SCHEMA_VERSION,
    HUB_REPO_ID,
    HUB_REVISION,
    V6_RELEVANCE_AUTHORIZATION_ID,
    V6_RELEVANCE_CALIBRATION_ID,
    V6_RELEVANCE_COLLECTION_ID,
    V6_RELEVANCE_RUBRIC_VERSION,
    V6_TRACKED_FILES,
    _v6_selected_digests,
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

V6_STRATA = PILOT_STRATA
V6_ITEM_COUNT = 40
V6_TARGET_PER_STRATUM = 10
V6_CACHE_SCHEMA = "leakage-calibration-cache/v1"
V6_QUERY_TEMPLATE_VERSION = "pilot-query/v6"
V6_FILTER_SCHEMA = "cutoff-filter/v1"
V6_TEMPORAL_INDEX_SHA256 = "80d977e40e113172e778b6f78315e1fad132423c9fe9eca3ad6a5de34df4f21b"
V6_BACKBONE_ID = "openai:gpt-4.1"
V6_BACKBONE_TRAINING_CUTOFF = "2024-06-30"
V6_EXPERIMENT_AS_OF = "2026-08-17"
V6_GUARD_DAYS = 30
V6_CONTENT_CHAR_LIMIT = 100_000
V6_POLICY = {
    "calibration_id": V6_RELEVANCE_CALIBRATION_ID,
    "ranking_eligible": False,
    "regime": "unrestricted_current_web",
    "scope": "calibration_relevance_rubric_v5_fresh_pilot",
    "schema_version": "leakage_probe/v6",
}
V6_POLICY_SHA256 = sha256_json(V6_POLICY)
V6_SERVICE_COHORT_SHA256 = sha256_json(
    {
        "calibration_id": V6_RELEVANCE_CALIBRATION_ID,
        "hub_revision": HUB_REVISION,
        "provider_id": "volc",
        "tool_contract_sha256": DEFAULT_TOOL_CONTRACT.hash,
    }
)
_CALL_PREFIX = "v6pilot40"
_REQUEST_LOCK = Lock()
_NEXT_REQUEST_AT = 0.0


class V6PilotCollectionError(PilotCollectionError):
    """Raised when the v6 authorization or pilot contract fails closed."""


class V6PilotResult:
    """Safe, non-ranking summary of the v6 pilot."""

    def __init__(
        self,
        *,
        task_count: int,
        query_count: int,
        candidate_pool_count: int,
        label_a_count: int,
        calibration_sha256: str,
    ) -> None:
        self.task_count = task_count
        self.query_count = query_count
        self.candidate_pool_count = candidate_pool_count
        self.label_a_count = label_a_count
        self.calibration_sha256 = calibration_sha256

    def as_dict(self) -> dict[str, object]:
        return {
            "calibration_id": V6_RELEVANCE_CALIBRATION_ID,
            "calibration_sha256": self.calibration_sha256,
            "candidate_pool_count": self.candidate_pool_count,
            "collection_id": V6_RELEVANCE_COLLECTION_ID,
            "item_count": V6_ITEM_COUNT,
            "label_a_count": self.label_a_count,
            "query_count": self.query_count,
            "task_count": self.task_count,
        }


def run_v6_pilot(
    *,
    root: str | Path = DEFAULT_V6_ROOT,
    cache_root: str | Path = "data/calibration-cache/leakage-probe/v6",
    dataset_cache: str | Path = "data/.huggingface",
    timeout_seconds: float = 30.0,
) -> V6PilotResult:
    """Collect the authorized disjoint pilot after scope and rubric preflight."""

    root_path = Path(root).expanduser().resolve()
    cache_path = Path(cache_root).expanduser().resolve()
    try:
        prepare_v6_scaffold(root_path)
    except ValueError as exc:
        raise V6PilotCollectionError("v6 calibration scaffold is unavailable") from exc
    if timeout_seconds <= 0:
        raise V6PilotCollectionError("v6 pilot timeout must be positive")
    _private_directory(cache_path)
    try:
        validate_calibration_root(root_path)
    except ValueError as exc:
        raise V6PilotCollectionError("v6 scaffold failed validation before collection") from exc
    rubric_sha256 = _sha256_file(root_path / "rubric.md")
    if _pilot_has_rows(root_path):
        labels = _read_jsonl(root_path / "pilot/labels/raw/annotator-a.jsonl")
        if labels:
            raise V6PilotCollectionError("v6 Labeler-A output already exists")
        label_count = _label_v6_pilot(root_path)
        _refresh_v6_state(root_path, _manifest_evidence_sha(root_path), pilot_n=label_count)
        result = validate_calibration_root(root_path)
        return V6PilotResult(
            task_count=len(
                {row["task_id"] for row in _read_jsonl(root_path / "pilot/query-corpus.jsonl")}
            ),
            query_count=len(_read_jsonl(root_path / "pilot/query-corpus.jsonl")),
            candidate_pool_count=0,
            label_a_count=label_count,
            calibration_sha256=result["calibration_sha256"],
        )

    tasks, scopes = _preflight_tasks(root_path, dataset_cache, rubric_sha256)
    # The proof is already materialized by scaffold_v6.  It is checked again
    # immediately before the first live call; this is the freeze boundary.
    _assert_rubric_frozen(root_path, rubric_sha256)
    candidate_pool, query_rows, task_count, evidence_entries = _collect_candidates(
        tasks=tasks,
        scopes=scopes,
        cache_root=cache_path,
        timeout_seconds=timeout_seconds,
        rubric_sha256=rubric_sha256,
    )
    selected = _select_items(candidate_pool, root_path)
    if len(selected) != V6_ITEM_COUNT:
        raise V6PilotCollectionError("v6 selection did not produce exactly 40 disjoint items")
    item_rows, filter_rows = _materialize_items(
        selected, cache_root=cache_path, evidence_entries=evidence_entries
    )
    views = _build_v6_views(item_rows, cache_path, scopes)
    pilot_root = root_path / "pilot"
    _write_jsonl(pilot_root / "query-corpus.jsonl", query_rows)
    _write_jsonl(pilot_root / "items.jsonl", item_rows)
    _write_jsonl(pilot_root / "filter-decisions.jsonl", filter_rows)
    _write_jsonl(pilot_root / "annotation-view.jsonl", views)
    _write_exclusion_proof(root_path, item_rows, views, query_rows)
    evidence_sha = _write_evidence_manifest(cache_path, evidence_entries)
    _refresh_v6_state(root_path, evidence_sha, pilot_n=0)
    validate_calibration_root(root_path)
    label_count = _label_v6_pilot(root_path)
    _refresh_v6_state(root_path, evidence_sha, pilot_n=label_count)
    validated = validate_calibration_root(root_path)
    if validated["counts"]["items"] != 400 or label_count != V6_ITEM_COUNT:
        raise V6PilotCollectionError("v6 top-level audit or Labeler-A pilot output is incomplete")
    return V6PilotResult(
        task_count=task_count,
        query_count=len(query_rows),
        candidate_pool_count=len(candidate_pool),
        label_a_count=label_count,
        calibration_sha256=validated["calibration_sha256"],
    )


def _preflight_tasks(
    root: Path, dataset_cache: str | Path, rubric_sha256: str
) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, object]]]:
    del rubric_sha256
    all_tasks = _load_tasks(dataset_cache)
    proof = json.loads((root / "pilot/exclusion-proof.json").read_text(encoding="utf-8"))
    excluded_semantic = set(proof["excluded_digests"]["semantic_key_sha256"])
    tasks: dict[str, list[dict[str, object]]] = {stratum: [] for stratum in V6_STRATA}
    scopes: dict[str, dict[str, object]] = {}
    for stratum in V6_STRATA:
        for row in all_tasks[stratum]:
            task_id = str(row["task_id"])
            if all(
                sha256_text(f"{task_id}\0{query_class}") in excluded_semantic
                for query_class in PILOT_QUERY_CLASSES
            ):
                continue
            try:
                scope = build_task_scope(row)
            except TaskScopeError as exc:
                raise V6PilotCollectionError(
                    f"task-scope preflight failed for {task_id}: {exc}"
                ) from exc
            tasks[stratum].append(row)
            scopes[task_id] = scope
            if len(tasks[stratum]) == V6_TARGET_PER_STRATUM:
                break
        if not tasks[stratum]:
            raise V6PilotCollectionError(
                f"task-scope preflight has no unused task/query scope in {stratum}"
            )
    if not scopes:
        raise V6PilotCollectionError("task-scope preflight produced no task scopes")
    return tasks, scopes


def _collect_candidates(
    *,
    tasks: dict[str, list[dict[str, object]]],
    scopes: Mapping[str, Mapping[str, object]],
    cache_root: Path,
    timeout_seconds: float,
    rubric_sha256: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int, list[dict[str, object]]]:
    access_key = os.environ.get("VOLC_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("VOLC_SECRET_ACCESS_KEY", "").strip()
    if not access_key or not secret_key:
        raise V6PilotCollectionError("Volc credentials are unavailable for the authorized v6 pilot")
    transport = VolcSdkTransport(access_key=access_key, secret_key=secret_key)
    candidate_pool: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    evidence_entries: list[dict[str, object]] = []
    previous_receipt_sha256: str | None = None
    task_count = 0
    for stratum in V6_STRATA:
        for row in tasks[stratum][:V6_TARGET_PER_STRATUM]:
            task_count += 1
            task_id = str(row["task_id"])
            cutoff_date = _date_text(row.get("cutoff_date"), "cutoff_date", task_id)
            forecast_origin = _date_text(row.get("forecast_origin"), "forecast_origin", task_id)
            for query_class in PILOT_QUERY_CLASSES:
                query = _build_v6_query(row, query_class, cutoff_date)
                query_rows.append(
                    {
                        "query_class": query_class,
                        "query_corpus_entry_id": sha256_text(
                            f"leakage-calibration-query/v6\0{task_id}\0{query_class}"
                        ),
                        "query_sha256": sha256_text(query),
                        "schema_version": CALIBRATION_SCHEMA_VERSION,
                        "task_id": task_id,
                        "template_version": V6_QUERY_TEMPLATE_VERSION,
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
                response = _v6_post(
                    transport, "https://mercury.volcengineapi.com/", body, timeout_seconds
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
                        "authorization_id": V6_RELEVANCE_AUTHORIZATION_ID,
                        "call_id": call_id,
                        "collection_id": V6_RELEVANCE_COLLECTION_ID,
                        "query": query,
                        "request": body,
                        "response": response_payload,
                        "response_error": response.error or provider_error,
                        "response_request_id": response.request_id,
                        "response_status_code": response.status_code,
                        "rubric_sha256": rubric_sha256,
                        "schema_version": V6_CACHE_SCHEMA,
                        "task_id": task_id,
                    },
                )
                evidence_entries.append(_cache_entry(call_path, cache_root))
                if response.error or provider_error or not 200 <= (response.status_code or 0) < 300:
                    raise V6PilotCollectionError(f"Volc v6 pilot call failed for {call_id}")
                raw_results = _volc_web_results(response_payload)
                if raw_results is None:
                    raise V6PilotCollectionError(f"Volc v6 response shape invalid for {call_id}")
                receipt = {
                    "call_id": call_id,
                    "candidate_count": len(raw_results),
                    "collection_id": V6_RELEVANCE_COLLECTION_ID,
                    "previous_receipt_sha256": previous_receipt_sha256,
                    "provider_id": "volc",
                    "query_sha256": sha256_text(query),
                    "receipt_protocol": "search_receipt/v1",
                    "request_filter_sha256": sha256_json(body),
                    "response_sha256": response_hash,
                    "task_id": task_id,
                    "tool_contract_sha256": DEFAULT_TOOL_CONTRACT.hash,
                }
                receipt_sha256 = sha256_json(receipt)
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
                            "previous_receipt_sha256": receipt["previous_receipt_sha256"],
                            "query_class": query_class,
                            "query_sha256": sha256_text(query),
                            "raw_candidate": dict(raw_candidate),
                            "receipt_sha256": receipt_sha256,
                            "request_filter_sha256": receipt["request_filter_sha256"],
                            "response_sha256": response_hash,
                            "scope": dict(scopes[task_id]),
                            "stratum": stratum,
                            "task_id": task_id,
                            "updated_at_utc": _raw_provider_time(raw_candidate, "UpdateTime"),
                        }
                    )
    query_rows.sort(key=lambda row: str(row["query_corpus_entry_id"]))
    return candidate_pool, query_rows, task_count, evidence_entries


def _build_v6_query(row: Mapping[str, object], query_class: str, cutoff_date: str) -> str:
    query = f"{_build_query(row, query_class, cutoff_date)} rubric v5 calibration".strip()
    if len(query) > 512:
        raise V6PilotCollectionError("generated v6 query exceeded the bounded contract")
    return query


def _v6_post(
    transport: VolcSdkTransport,
    endpoint: str,
    body: Mapping[str, object],
    timeout_seconds: float,
) -> VolcHttpResponse:
    global _NEXT_REQUEST_AT
    for attempt in range(3):
        with _REQUEST_LOCK:
            delay = _NEXT_REQUEST_AT - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            _NEXT_REQUEST_AT = time.monotonic() + 1.0
        response = transport.post(
            endpoint,
            json=body,
            headers={"Content-Type": "application/json"},
            timeout_seconds=timeout_seconds,
        )
        if (
            _volc_response_error(response.payload) != "volc_error_rate_limit_exceeded"
            or attempt == 2
        ):
            return response
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _select_items(candidate_pool: list[dict[str, object]], root: Path) -> list[dict[str, object]]:
    proof = json.loads((root / "pilot/exclusion-proof.json").read_text(encoding="utf-8"))
    excluded = proof["excluded_digests"]
    by_stratum: dict[str, list[dict[str, object]]] = {stratum: [] for stratum in V6_STRATA}
    seen: set[str] = set()
    seen_payloads: set[str] = set()
    for candidate in candidate_pool:
        item_id = _v6_item_id(candidate)
        visible_candidate = _visible_candidate(candidate)
        payload_sha = sha256_json(visible_candidate.as_model_payload())
        evidence = {
            "item_id": item_id,
            "receipt_sha256": str(candidate["receipt_sha256"]),
            "query_sha256": str(candidate["query_sha256"]),
            "canonical_url_sha256": sha256_text(str(candidate["canonical_url"])),
            "model_visible_payload_sha256": payload_sha,
            "url_payload_sha256": sha256_text(
                f"{sha256_text(str(candidate['canonical_url']))}|{payload_sha}"
            ),
            "semantic_key_sha256": sha256_text(
                f"{candidate['task_id']}\0{candidate['query_class']}"
            ),
        }
        if (
            item_id in seen
            or any(
                evidence[field] in set(excluded[field]) for field in evidence if field in excluded
            )
            or payload_sha in seen_payloads
        ):
            continue
        seen.add(item_id)
        seen_payloads.add(payload_sha)
        record = dict(candidate)
        record["provisional_item_id"] = item_id
        by_stratum[str(record["stratum"])].append(record)
    selected: list[dict[str, object]] = []
    for stratum in V6_STRATA:
        pool = by_stratum[stratum]
        pool.sort(key=lambda candidate: _sampling_key(str(candidate["provisional_item_id"])))
        domains: Counter[str] = Counter()
        tasks: Counter[str] = Counter()
        chosen = 0
        for rank, candidate in enumerate(pool, start=1):
            domain = urlsplit(str(candidate["canonical_url"])).hostname or ""
            task_id = str(candidate["task_id"])
            if domains[domain] >= 20 or tasks[task_id] >= 8:
                continue
            picked = dict(candidate)
            picked["deterministic_rank"] = rank
            picked["inclusion_probability"] = min(1.0, V6_TARGET_PER_STRATUM / max(1, len(pool)))
            selected.append(picked)
            domains[domain] += 1
            tasks[task_id] += 1
            chosen += 1
            if chosen == V6_TARGET_PER_STRATUM:
                break
        if chosen != V6_TARGET_PER_STRATUM:
            raise V6PilotCollectionError(f"v6 stratum {stratum} lacks 10 disjoint candidates")
    return selected


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
            raise V6PilotCollectionError("selected v6 candidate has no URL")
        item_id = str(record["provisional_item_id"])
        candidate_path = cache_root / "candidates" / f"{item_id}.json"
        _write_cache_json(
            candidate_path,
            {
                "candidate": record["raw_candidate"],
                "call_id": record["call_id"],
                "candidate_index": record["candidate_index"],
                "collection_id": V6_RELEVANCE_COLLECTION_ID,
                "query_sha256": record["query_sha256"],
                "scope": record["scope"],
                "schema_version": V6_CACHE_SCHEMA,
                "task_id": record["task_id"],
            },
        )
        evidence_entries.append(_cache_entry(candidate_path, cache_root))
        visible = _visible_candidate(record)
        canonical_url = str(record["canonical_url"])
        cutoff_date = str(record["cutoff_date"])
        task_id = str(record["task_id"])
        item_rows.append(
            {
                "backbone_id": V6_BACKBONE_ID,
                "backbone_training_cutoff": V6_BACKBONE_TRAINING_CUTOFF,
                "call_id": record["call_id"],
                "candidate_index": record["candidate_index"],
                "canonical_url": canonical_url,
                "canonical_url_sha256": sha256_text(canonical_url),
                "canonical_url_version": "url-canonicalizer/v1",
                "category": record["category"],
                "collection_id": V6_RELEVANCE_COLLECTION_ID,
                "content_sha256": sha256_text(visible.content or ""),
                "cutoff_declaration_sha256": sha256_json(
                    {
                        "backbone_training_cutoff": V6_BACKBONE_TRAINING_CUTOFF,
                        "cutoff_date": cutoff_date,
                        "experiment_as_of": V6_EXPERIMENT_AS_OF,
                        "guard_days": V6_GUARD_DAYS,
                        "task_id": task_id,
                    }
                ),
                "deterministic_rank": record["deterministic_rank"],
                "domain": urlsplit(canonical_url).hostname,
                "exclusive_cutoff_bound": CutoffWindow.from_seed(cutoff_date).exclusive_bound,
                "experiment_as_of": V6_EXPERIMENT_AS_OF,
                "forecast_origin": record["forecast_origin"],
                "hub_repo_id": HUB_REPO_ID,
                "hub_revision": HUB_REVISION,
                "inclusion_probability": record["inclusion_probability"],
                "item_id": item_id,
                "model_visible_payload_sha256": sha256_json(visible.as_model_payload()),
                "original_url_sha256": sha256_text(raw_url),
                "paper_band": record["paper_band"],
                "policy_sha256": V6_POLICY_SHA256,
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
                "service_cohort_sha256": V6_SERVICE_COHORT_SHA256,
                "snapshot_captured_at_utc": captured_at,
                "snippet_sha256": sha256_text(visible.snippet or ""),
                "task_cutoff_date": cutoff_date,
                "task_id": task_id,
                "task_scope_sha256": record["scope"]["task_scope_sha256"],
                "temporal_index_sha256": V6_TEMPORAL_INDEX_SHA256,
                "title_sha256": sha256_text(visible.title or ""),
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
    item_rows.sort(key=lambda row: str(row["item_id"]))
    filter_rows.sort(key=lambda row: str(row["item_id"]))
    return item_rows, filter_rows


def _build_v6_views(
    items: list[dict[str, object]], cache_root: Path, scopes: Mapping[str, Mapping[str, object]]
) -> list[dict[str, object]]:
    views: list[dict[str, object]] = []
    for item in items:
        path = cache_root / "candidates" / f"{item['item_id']}.json"
        cached = json.loads(path.read_text(encoding="utf-8"))
        raw_candidate = cached.get("candidate")
        if not isinstance(raw_candidate, Mapping):
            raise V6PilotCollectionError("v6 candidate cache is malformed")
        candidate = _visible_candidate(
            {
                "candidate": VolcSearchCandidate.from_volc_result(raw_candidate),
                "raw_candidate": raw_candidate,
            }
        )
        scope = scopes.get(str(item["task_id"]))
        if scope is None:
            raise V6PilotCollectionError("v6 task scope is missing while building blind view")
        view = {
            "category": item["category"],
            "item_id": item["item_id"],
            "model_visible_content": candidate.content,
            "model_visible_payload_sha256": item["model_visible_payload_sha256"],
            "model_visible_snippet": candidate.snippet,
            "model_visible_title": candidate.title,
            "paper_band": item["paper_band"],
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "task_id": item["task_id"],
            "task_scope": dict(scope),
            "truncation_metadata": _truncation_metadata(raw_candidate, candidate, scope),
        }
        view["annotation_view_sha256"] = sha256_json(view)
        views.append(view)
    views.sort(key=lambda row: str(row["item_id"]))
    return views


def _truncation_metadata(
    raw_candidate: Mapping[str, object], candidate: VolcSearchCandidate, scope: Mapping[str, object]
) -> dict[str, object]:
    content_bytes = len((candidate.content or "").encode("utf-8"))
    provider_truncated = any(
        raw_candidate.get(field) is True
        for field in ("IsTruncated", "ContentTruncated", "Truncated", "IsContentTruncated")
    )
    clipped = (
        isinstance(raw_candidate.get("Content"), str)
        and len(str(raw_candidate["Content"])) > V6_CONTENT_CHAR_LIMIT
    )
    proven = provider_truncated or clipped
    hidden_axis = "none"
    visible = _candidate_text(candidate).lower()
    if proven:
        entity_terms = (str(scope.get("entity_name", "")), str(scope.get("entity_ticker", "")))
        if not _contains_any(visible, entity_terms):
            hidden_axis = "entity"
        elif scope.get("family") == "B":
            hidden_axis = "event"
        elif scope.get("family") == "C":
            hidden_axis = "period"
        else:
            hidden_axis = "metric"
    return {
        "content_bytes": content_bytes,
        "content_ends_at_limit": clipped,
        "content_limit_bytes": content_bytes if clipped else V6_CONTENT_CHAR_LIMIT * 4,
        "hidden_axis": hidden_axis,
        "no_decisive_visible_span": bool(proven and hidden_axis != "none"),
        "proven_visible_truncation": proven,
        "provider_truncated": provider_truncated,
        "token_boundary": False,
    }


def _visible_candidate(record: Mapping[str, object]) -> VolcSearchCandidate:
    candidate = record["candidate"]
    if not isinstance(candidate, VolcSearchCandidate):
        raise V6PilotCollectionError("v6 candidate record is malformed")
    content = candidate.content or ""
    if len(content) > V6_CONTENT_CHAR_LIMIT:
        content = content[:V6_CONTENT_CHAR_LIMIT]
    return VolcSearchCandidate(
        url=candidate.url,
        title=candidate.title,
        snippet=candidate.snippet,
        content=content,
        volc_publish_time=candidate.volc_publish_time,
        site_name=candidate.site_name,
        auth_info_level=candidate.auth_info_level,
    )


def _v6_item_id(record: Mapping[str, object]) -> str:
    candidate = _visible_candidate(record)
    canonical = sha256_text(str(record["canonical_url"]))
    payload = sha256_json(candidate.as_model_payload())
    return sha256_text(
        "\0".join(
            (
                "retrieval_leakage_candidate/v1",
                str(record["receipt_sha256"]),
                str(record["candidate_index"]),
                canonical,
                payload,
                str(record["task_id"]),
                str(record["cutoff_date"]),
            )
        )
    )


def _label_v6_pilot(root: Path, *, reviewer_id: str = "dev-labeler-a-v6-rubric-v5") -> int:
    items = _read_jsonl(root / "pilot/items.jsonl")
    views = {row["item_id"]: row for row in _read_jsonl(root / "pilot/annotation-view.jsonl")}
    rubric_sha256 = _sha256_file(root / "rubric.md")
    labels: list[dict[str, object]] = []
    for item in items:
        item_id = str(item["item_id"])
        view = views.get(item_id)
        if view is None:
            raise V6PilotCollectionError("v6 pilot annotation view is incomplete")
        decision = classify_v6(view)
        labels.append(
            {
                "annotation_view_sha256": view["annotation_view_sha256"],
                "confidence": 0.25 if decision.task_relevance == "indeterminate" else 0.85,
                "entity_match": decision.entity_match,
                "event_match": decision.event_match,
                "evidence": [],
                "hidden_axis": decision.hidden_axis,
                "item_id": item_id,
                "labeled_at_utc": _now_utc(),
                "metric_match": decision.metric_match,
                "oclass": decision.oclass,
                "outcome_revealing": decision.outcome_revealing,
                "rationale": decision.rationale(str(view["task_scope"]["task_scope_sha256"])),
                "rclass": decision.rclass,
                "reviewer_id": reviewer_id,
                "reviewer_type": "labeler_a",
                "rubric_sha256": rubric_sha256,
                "rubric_version": V6_RELEVANCE_RUBRIC_VERSION,
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "task_relevance": decision.task_relevance,
                "task_scope_sha256": view["task_scope"]["task_scope_sha256"],
                "temporal_label": "indeterminate",
                "truncation": decision.truncation,
            }
        )
    labels.sort(key=lambda row: str(row["item_id"]))
    _write_jsonl(root / "pilot/labels/raw/annotator-a.jsonl", labels)
    return len(labels)


def _write_exclusion_proof(
    root: Path,
    items: list[dict[str, object]],
    views: list[dict[str, object]],
    queries: list[dict[str, object]],
) -> None:
    path = root / "pilot/exclusion-proof.json"
    proof = json.loads(path.read_text(encoding="utf-8"))
    proof["selected_digests"] = _v6_selected_digests(items, views, queries)
    proof["status"] = "complete"
    path.write_bytes(canonical_json_bytes(proof))


def _write_evidence_manifest(cache_root: Path, entries: list[dict[str, object]]) -> str:
    path = cache_root / "restricted-evidence-manifest.json"
    _write_cache_json(
        path,
        {
            "authorization_id": V6_RELEVANCE_AUTHORIZATION_ID,
            "calibration_id": V6_RELEVANCE_CALIBRATION_ID,
            "collection_id": V6_RELEVANCE_COLLECTION_ID,
            "entries": sorted(entries, key=lambda entry: str(entry["path"])),
            "schema_version": "leakage-calibration-restricted-evidence/v1",
        },
    )
    return _sha256_file(path)


def _refresh_v6_state(root: Path, evidence_sha256: str, *, pilot_n: int) -> None:
    top_a = _read_jsonl(root / "labels/raw/annotator-a.jsonl")
    top_b = _read_jsonl(root / "labels/raw/annotator-b.jsonl")
    top_adj = _read_jsonl(root / "labels/adjudicated.jsonl")
    if top_a or top_b or top_adj:
        raise V6PilotCollectionError("v6 top-level audit labels must remain empty")
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
            "exclusions_by_type": {"v6_active_pilot": V6_ITEM_COUNT},
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
    cohort_path = root / "cohort.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort["calibration_status"] = "incomplete"
    cohort_path.write_bytes(canonical_json_bytes(cohort))
    from benchmark_harness.calibration.v6_scaffold import _write_v6_manifest

    _write_v6_manifest(root, evidence_sha256=evidence_sha256)


def _assert_rubric_frozen(root: Path, rubric_sha256: str) -> None:
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("rubric", {}).get("sha256") != rubric_sha256:
        raise V6PilotCollectionError("v6 rubric SHA changed before live collection")


def _manifest_evidence_sha(root: Path) -> str:
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    value = manifest.get("restricted_evidence_manifest_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise V6PilotCollectionError("v6 evidence manifest SHA is unavailable")
    return value


def _pilot_has_rows(root: Path) -> bool:
    return any(
        _read_jsonl(root / relative)
        for relative in V6_TRACKED_FILES
        if relative.startswith("pilot/") and relative.endswith(".jsonl")
    )


def _call_id(stratum: str, task_number: int, query_class: str) -> str:
    value = f"{_CALL_PREFIX}-{stratum.lower().replace('-', '')}-{task_number:02d}-{query_class}"
    if any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in value
    ):
        raise V6PilotCollectionError("v6 call identity is unsafe")
    return value


def _candidate_text(candidate: VolcSearchCandidate) -> str:
    return "\n".join(
        value.strip()
        for value in (candidate.title, candidate.snippet, candidate.content)
        if isinstance(value, str) and value.strip()
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term and term.lower() in text for term in terms)


def _raw_provider_time(raw_candidate: Mapping[str, object], name: str) -> str | None:
    value = raw_candidate.get(name)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return None


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["V6PilotCollectionError", "V6PilotResult", "run_v6_pilot"]
