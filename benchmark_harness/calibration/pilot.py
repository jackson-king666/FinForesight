"""Bounded, explicitly authorized leakage-calibration pilot collection.

The pilot is deliberately separate from experiment execution.  It calls the
same Volc provider and model-visible payload shape used by Search+Filter with
a broad provider-required ``TimeRange``; the task cutoff is applied afterward
to that exact response.  The blind annotation view exposes only the bounded
model-visible candidate fields and authoritative time evidence; raw provider
requests/responses and unredacted candidate payloads are written only below
the ignored calibration cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from benchmark_harness.calibration.validator import (
    CALIBRATION_ID,
    CALIBRATION_SCHEMA_VERSION,
    EXPECTED_TRACKED_FILES,
    HUB_REPO_ID,
    HUB_REVISION,
    JSON_FILES,
    JSONL_FILES,
    canonical_json_bytes,
    validate_calibration_root,
)
from benchmark_harness.contracts import sha256_json, sha256_text
from benchmark_harness.dataset_sources.huggingface import configure_huggingface_cache
from benchmark_harness.search import (
    DEFAULT_TOOL_CONTRACT,
    CutoffWindow,
    VolcSdkTransport,
    VolcSearchCandidate,
    filter_volc_candidates,
)
from benchmark_harness.search.filter import parse_volc_publish_time
from benchmark_harness.search.volc import _volc_web_results

PILOT_STRATA = ("A1-T1", "B-T1", "B-T2", "C-T2")
PILOT_TARGET_PER_STRATUM = 10
PILOT_ITEM_COUNT = 40
PILOT_QUERY_CLASSES = (
    "cutoff_anchored",
    "neutral_entity_topic",
    "retrospective_outcome_challenge",
)
PILOT_COLLECTION_ID = "leakage-probe-v1-pilot40-20260831"
PILOT_AUTHORIZATION_ID = "leakage-probe-v1-pilot40-20260831"
PILOT_AUTHORIZATION_SCHEMA = "leakage-calibration-authorization/v1"
PILOT_CACHE_SCHEMA = "leakage-calibration-cache/v1"
PILOT_QUERY_TEMPLATE_VERSION = "pilot-query/v1"
PILOT_FILTER_SCHEMA = "cutoff-filter/v1"
PILOT_POLICY = {
    "calibration_id": CALIBRATION_ID,
    "ranking_eligible": False,
    "regime": "unrestricted_current_web",
    "scope": "calibration_pilot",
    "schema_version": "leakage_probe/v1",
}
PILOT_POLICY_SHA256 = sha256_json(PILOT_POLICY)
PILOT_SERVICE_COHORT_SHA256 = sha256_json(
    {
        "calibration_id": CALIBRATION_ID,
        "hub_revision": HUB_REVISION,
        "provider_id": "volc",
        "tool_contract_sha256": DEFAULT_TOOL_CONTRACT.hash,
    }
)
PILOT_FILTER_POLICY_SHA256 = sha256_json(
    {
        "fail_closed_ambiguous": True,
        "max_text_chars": 8_000,
        "schema_version": PILOT_FILTER_SCHEMA,
    }
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_HEX = frozenset("0123456789abcdefABCDEF")
_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_CREDENTIAL_QUERY_WORDS = ("token", "secret", "signature", "auth")


class PilotCollectionError(RuntimeError):
    """Raised when the explicitly authorized pilot cannot be collected safely."""


@dataclass(frozen=True, slots=True)
class PilotResult:
    """Safe summary returned after pilot collection and optional blind labeling."""

    collection_id: str
    task_count: int
    query_count: int
    candidate_pool_count: int
    item_count: int
    label_a_count: int
    calibration_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "calibration_sha256": self.calibration_sha256,
            "candidate_pool_count": self.candidate_pool_count,
            "collection_id": self.collection_id,
            "item_count": self.item_count,
            "label_a_count": self.label_a_count,
            "query_count": self.query_count,
            "task_count": self.task_count,
        }


def run_pilot(
    *,
    root: str | Path = "data/calibration/leakage-probe/v1",
    cache_root: str | Path = "data/calibration-cache/leakage-probe/v1",
    dataset_cache: str | Path = "data/.huggingface",
    timeout_seconds: float = 30.0,
) -> PilotResult:
    """Collect exactly the authorized 40-item pilot and run Labeler A.

    Authorization and the empty scaffold are validated before the first
    provider call.  The function has no option for a larger selection or a
    paper unrestricted run, which keeps the authorization boundary explicit.
    """

    root_path = Path(root).expanduser().resolve()
    cache_path = Path(cache_root).expanduser().resolve()
    _validate_authorized_empty_scaffold(root_path)
    if timeout_seconds <= 0:
        raise PilotCollectionError("pilot timeout must be positive")
    cache_path.mkdir(parents=True, exist_ok=True)
    _chmod_private(cache_path)
    candidate_pool, query_rows, task_count, evidence_entries = _collect_candidates(
        root_path=root_path,
        cache_root=cache_path,
        dataset_cache=dataset_cache,
        timeout_seconds=timeout_seconds,
    )
    selected = _select_pilot_items(candidate_pool)
    if len(selected) != PILOT_ITEM_COUNT:
        raise PilotCollectionError("pilot selection did not produce exactly 40 items")
    item_rows, filter_rows = _materialize_items(
        selected,
        cache_root=cache_path,
        query_rows=query_rows,
        evidence_entries=evidence_entries,
    )
    _write_jsonl(root_path / "query-corpus.jsonl", query_rows)
    _write_jsonl(root_path / "items.jsonl", item_rows)
    _write_jsonl(root_path / "filter-decisions.jsonl", filter_rows)
    _write_jsonl(root_path / "annotation-view.jsonl", _annotation_views(item_rows, cache_path))
    evidence_sha = _write_evidence_manifest(cache_path, evidence_entries)
    _refresh_tracked_state(
        root_path,
        status="pilot",
        restricted_evidence_manifest_sha256=evidence_sha,
    )
    # Validate the corpus before the blind label pass; Labeler A never reads
    # filter-decisions.jsonl, Hub prompts, expected outputs, or ground truth.
    _validate_pilot_state(root_path, expected_items=PILOT_ITEM_COUNT, expected_labels=0)
    label_result = label_pilot(root=root_path, cache_root=cache_path)
    label_count = label_result["label_a_count"]
    result = validate_calibration_root(root_path)
    if result["counts"]["items"] != PILOT_ITEM_COUNT or label_count != PILOT_ITEM_COUNT:
        raise PilotCollectionError("pilot corpus or Labeler A output is incomplete")
    return PilotResult(
        collection_id=PILOT_COLLECTION_ID,
        task_count=task_count,
        query_count=len(query_rows),
        candidate_pool_count=len(candidate_pool),
        item_count=result["counts"]["items"],
        label_a_count=result["counts"]["raw_labels_a"],
        calibration_sha256=result["calibration_sha256"],
    )


def label_pilot(
    *,
    root: str | Path = "data/calibration/leakage-probe/v1",
    cache_root: str | Path = "data/calibration-cache/leakage-probe/v1",
    reemit: bool = False,
) -> dict[str, Any]:
    """Run or explicitly re-emit the DEV/Labeler-A blind pass.

    ``reemit`` rebuilds the blind view from the private candidate cache before
    writing fresh A labels.  It is required when the view schema or evidence
    material changes, and is deliberately not the default so an accidental
    label overwrite cannot silently change a calibration history.
    """

    root_path = Path(root).expanduser().resolve()
    cache_path = Path(cache_root).expanduser().resolve()
    if reemit:
        items = _validate_reemit_source(root_path)
    else:
        result = validate_calibration_root(root_path)
        if result["calibration_status"] != "pilot" or result["counts"]["items"] != PILOT_ITEM_COUNT:
            raise PilotCollectionError("Labeler A requires a validated 40-item pilot corpus")
        if result["counts"]["raw_labels_a"]:
            raise PilotCollectionError("Labeler A output already exists; refusing to overwrite it")
        items = _read_jsonl(root_path / "items.jsonl")
    _validate_authorization_file(root_path)
    evidence_sha = _manifest_restricted_evidence_sha(root_path)
    if reemit:
        _write_jsonl(root_path / "annotation-view.jsonl", _annotation_views(items, cache_path))
    label_count = _label_pilot_blind(root_path)
    _refresh_tracked_state(
        root_path,
        status="pilot",
        restricted_evidence_manifest_sha256=evidence_sha,
    )
    validated = validate_calibration_root(root_path)
    if validated["counts"]["raw_labels_a"] != PILOT_ITEM_COUNT:
        raise PilotCollectionError("Labeler A did not produce exactly 40 labels")
    return {
        "calibration_sha256": validated["calibration_sha256"],
        "item_count": validated["counts"]["items"],
        "label_a_count": label_count,
        "status": "labeled_pilot",
    }


def _validate_reemit_source(root: Path) -> list[dict[str, Any]]:
    try:
        manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotCollectionError("pilot artifact manifest is unavailable") from exc
    if manifest.get("calibration_status") != "pilot":
        raise PilotCollectionError("Labeler A re-emission requires a pilot calibration root")
    items = _read_jsonl(root / "items.jsonl")
    if len(items) != PILOT_ITEM_COUNT:
        raise PilotCollectionError("Labeler A re-emission requires exactly 40 items")
    return items


def _validate_authorization_file(root: Path) -> dict[str, Any]:
    authorization_path = root / "authorization.json"
    try:
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotCollectionError("pilot authorization record is unavailable or invalid") from exc
    if authorization.get("schema_version") != PILOT_AUTHORIZATION_SCHEMA:
        raise PilotCollectionError("pilot authorization schema is not supported")
    if authorization.get("authorization_id") != PILOT_AUTHORIZATION_ID:
        raise PilotCollectionError("pilot authorization identity does not match this run")
    if authorization.get("calibration_unrestricted_collection_authorized") is not True:
        raise PilotCollectionError("pilot unrestricted collection is not explicitly authorized")
    if authorization.get("full_paper_unrestricted_authorized") is not False:
        raise PilotCollectionError("full paper unrestricted collection is not fail-closed")
    if authorization.get("same_candidates_cutoff_filter") is not True:
        raise PilotCollectionError("pilot must filter the exact unrestricted candidate response")
    return authorization


def _validate_authorized_empty_scaffold(root: Path) -> dict[str, Any]:
    try:
        result = validate_calibration_root(root)
    except ValueError as exc:
        raise PilotCollectionError(
            "calibration scaffold failed validation before collection"
        ) from exc
    _validate_authorization_file(root)
    if result["calibration_status"] not in {"not_started", "pilot"}:
        raise PilotCollectionError("calibration root is already frozen; create a new version")
    if result["counts"]["items"]:
        raise PilotCollectionError("pilot root is already populated; refusing to overwrite it")
    return result


def _load_tasks(dataset_cache: str | Path) -> dict[str, list[dict[str, Any]]]:
    configure_huggingface_cache(dataset_cache, offline=False)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise PilotCollectionError("Hugging Face dataset support is unavailable") from exc

    tasks: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in PILOT_STRATA}
    for config in ("a1", "b", "c"):
        try:
            dataset = load_dataset(
                HUB_REPO_ID,
                name=config,
                split="train",
                revision=HUB_REVISION,
                cache_dir=str(Path(dataset_cache).expanduser().resolve() / "datasets"),
                streaming=False,
            )
            rows = list(dataset)
        except Exception as exc:
            raise PilotCollectionError(f"Hub pilot task load failed for config {config!r}") from exc
        for raw_row in rows:
            if not isinstance(raw_row, Mapping):
                continue
            row = dict(raw_row)
            if (
                row.get("status") != "ready"
                or row.get("review_status") != "reviewed"
                or row.get("official_temporal_eligible") is not True
            ):
                continue
            category = row.get("category")
            paper_band = row.get("paper_band")
            stratum = f"{category}-{paper_band}"
            if stratum in tasks:
                tasks[stratum].append(row)
    for stratum, rows in tasks.items():
        rows.sort(key=lambda row: str(row.get("task_id", "")))
        if len(rows) < PILOT_TARGET_PER_STRATUM:
            raise PilotCollectionError(f"Hub has fewer than 10 reviewed tasks in {stratum}")
        task_ids = [row.get("task_id") for row in rows[:PILOT_TARGET_PER_STRATUM]]
        if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
            raise PilotCollectionError(f"Hub pilot task IDs are invalid in {stratum}")
    return tasks


def _load_task_prompts(
    dataset_cache: str | Path, item_rows: list[dict[str, Any]]
) -> dict[str, str]:
    """Load only task prompts needed to enrich a v2 blind annotation view."""

    tasks = _load_tasks(dataset_cache)
    prompts_by_task: dict[str, str] = {}
    for rows in tasks.values():
        for row in rows:
            task_id = row.get("task_id")
            prompt = row.get("prompt")
            if not isinstance(task_id, str) or not task_id:
                raise PilotCollectionError("Hub task IDs are invalid while loading prompts")
            if not isinstance(prompt, str) or not prompt.strip():
                raise PilotCollectionError(f"Hub task prompt is unavailable for {task_id}")
            previous = prompts_by_task.get(task_id)
            if previous is not None and previous != prompt:
                raise PilotCollectionError(f"Hub task prompt is inconsistent for {task_id}")
            prompts_by_task[task_id] = prompt
    selected: dict[str, str] = {}
    for item in item_rows:
        task_id = item.get("task_id")
        prompt = prompts_by_task.get(str(task_id))
        if prompt is None:
            raise PilotCollectionError(f"Hub task prompt is missing for item task {task_id}")
        selected[str(task_id)] = prompt
    return selected


def _collect_candidates(
    *,
    root_path: Path,
    cache_root: Path,
    dataset_cache: str | Path,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[dict[str, Any]]]:
    del root_path
    tasks = _load_tasks(dataset_cache)
    access_key = os.environ.get("VOLC_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("VOLC_SECRET_ACCESS_KEY", "").strip()
    if not access_key or not secret_key:
        raise PilotCollectionError(
            "Volc credentials are unavailable for authorized pilot collection"
        )
    transport = VolcSdkTransport(access_key=access_key, secret_key=secret_key)
    candidate_pool: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    evidence_entries: list[dict[str, Any]] = []
    previous_receipt_sha256: str | None = None
    task_count = 0
    for stratum in PILOT_STRATA:
        for row in tasks[stratum][:PILOT_TARGET_PER_STRATUM]:
            task_count += 1
            task_id = str(row["task_id"])
            cutoff_date = _date_text(row.get("cutoff_date"), "cutoff_date", task_id)
            forecast_origin = _date_text(row.get("forecast_origin"), "forecast_origin", task_id)
            for query_class in PILOT_QUERY_CLASSES:
                query = _build_query(row, query_class, cutoff_date)
                query_entry_id = sha256_text(
                    f"leakage-calibration-query/v1\0{task_id}\0{query_class}"
                )
                query_rows.append(
                    {
                        "query_class": query_class,
                        "query_corpus_entry_id": query_entry_id,
                        "query_sha256": sha256_text(query),
                        "schema_version": CALIBRATION_SCHEMA_VERSION,
                        "task_id": task_id,
                        "template_version": PILOT_QUERY_TEMPLATE_VERSION,
                    }
                )
                call_id = _call_id(stratum, task_count, query_class)
                body = {
                    "Query": query,
                    "SearchType": "web",
                    "Count": 10,
                    "Filter": {"NeedContent": True, "NeedUrl": True},
                    # The provider requires a calendar range.  This is a
                    # deliberately broad current-web range, never the task's
                    # cutoff; the sealed cutoff filter below is the only
                    # source of the controlled decision.
                    "TimeRange": _unrestricted_time_range(cutoff_date),
                }
                response = transport.post(
                    "https://mercury.volcengineapi.com/",
                    json=body,
                    headers={"Content-Type": "application/json"},
                    timeout_seconds=timeout_seconds,
                )
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
                        "authorization_id": PILOT_AUTHORIZATION_ID,
                        "call_id": call_id,
                        "collection_id": PILOT_COLLECTION_ID,
                        "query": query,
                        "request": body,
                        "response": response_payload,
                        "response_error": response.error,
                        "response_request_id": response.request_id,
                        "response_status_code": response.status_code,
                        "schema_version": PILOT_CACHE_SCHEMA,
                        "task_id": task_id,
                    },
                )
                evidence_entries.append(_cache_entry(cache_call_path, cache_root))
                if response.error or not 200 <= (response.status_code or 0) < 300:
                    raise PilotCollectionError(f"Volc pilot call failed for {call_id}")
                raw_results = _volc_web_results(response_payload)
                if raw_results is None:
                    raise PilotCollectionError(f"Volc pilot response shape invalid for {call_id}")
                receipt_payload = {
                    "call_id": call_id,
                    "candidate_count": len(raw_results),
                    "collection_id": PILOT_COLLECTION_ID,
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


def _select_pilot_items(candidate_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_stratum: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in PILOT_STRATA}
    for candidate in candidate_pool:
        by_stratum[candidate["stratum"]].append(candidate)
    for stratum in PILOT_STRATA:
        pool = by_stratum[stratum]
        pool.sort(key=lambda candidate: _sampling_key(_provisional_item_id(candidate)))
        domain_counts: Counter[str] = Counter()
        task_counts: Counter[str] = Counter()
        chosen = 0
        for rank, candidate in enumerate(pool, start=1):
            domain = urlsplit(candidate["canonical_url"]).hostname or ""
            task_id = candidate["task_id"]
            if domain_counts[domain] >= 20 or task_counts[task_id] >= 8:
                continue
            candidate = dict(candidate)
            candidate["deterministic_rank"] = rank
            candidate["inclusion_probability"] = min(
                1.0, PILOT_TARGET_PER_STRATUM / max(1, len(pool))
            )
            selected.append(candidate)
            domain_counts[domain] += 1
            task_counts[task_id] += 1
            chosen += 1
            if chosen == PILOT_TARGET_PER_STRATUM:
                break
        if chosen != PILOT_TARGET_PER_STRATUM:
            raise PilotCollectionError(
                f"pilot stratum {stratum} has fewer than 10 cap-compliant items"
            )
    return selected


def _materialize_items(
    selected: list[dict[str, Any]],
    *,
    cache_root: Path,
    query_rows: list[dict[str, Any]],
    evidence_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    del query_rows
    captured_at = _now_utc()
    item_rows: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    for candidate_record in selected:
        candidate: VolcSearchCandidate = candidate_record["candidate"]
        raw_url = candidate.url
        assert raw_url is not None
        canonical_url = candidate_record["canonical_url"]
        payload = candidate.as_model_payload()
        payload_sha256 = _sha256_canonical(payload)
        candidate_cache_path = (
            cache_root / "candidates" / f"{_provisional_item_id(candidate_record)}.json"
        )
        _write_cache_json(
            candidate_cache_path,
            {
                "candidate": candidate_record["raw_candidate"],
                "call_id": candidate_record["call_id"],
                "candidate_index": candidate_record["candidate_index"],
                "query_sha256": candidate_record["query_sha256"],
                "schema_version": PILOT_CACHE_SCHEMA,
                "task_id": candidate_record["task_id"],
            },
        )
        item_id = _provisional_item_id(candidate_record)
        evidence_entries.append(_cache_entry(candidate_cache_path, cache_root))
        published_at = _raw_provider_time(candidate_record["raw_candidate"], "PublishTime")
        snapshot_sha256 = _sha256_file(candidate_cache_path)
        task_id = candidate_record["task_id"]
        cutoff_date = candidate_record["cutoff_date"]
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
                "collection_id": PILOT_COLLECTION_ID,
                "content_sha256": sha256_text(candidate.content or ""),
                "cutoff_declaration_sha256": sha256_json(
                    {
                        "backbone_training_cutoff": "2024-06-30",
                        "cutoff_date": cutoff_date,
                        "experiment_as_of": "2026-08-17",
                        "guard_days": 30,
                        "task_id": task_id,
                    }
                ),
                "deterministic_rank": candidate_record["deterministic_rank"],
                "domain": urlsplit(canonical_url).hostname,
                "exclusive_cutoff_bound": CutoffWindow.from_seed(cutoff_date).exclusive_bound,
                "experiment_as_of": "2026-08-17",
                "forecast_origin": candidate_record["forecast_origin"],
                "hub_repo_id": HUB_REPO_ID,
                "hub_revision": HUB_REVISION,
                "inclusion_probability": candidate_record["inclusion_probability"],
                "item_id": item_id,
                "model_visible_payload_sha256": payload_sha256,
                "original_url_sha256": sha256_text(raw_url),
                "paper_band": candidate_record["paper_band"],
                "policy_sha256": PILOT_POLICY_SHA256,
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
                "restricted_snapshot_sha256": snapshot_sha256,
                "restricted_snapshot_status": "available",
                "retrieved_at_utc": captured_at,
                "sampling_stratum": candidate_record["stratum"],
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "service_cohort_sha256": PILOT_SERVICE_COHORT_SHA256,
                "snapshot_captured_at_utc": captured_at,
                "snippet_sha256": sha256_text(candidate.snippet or ""),
                "task_cutoff_date": cutoff_date,
                "task_id": task_id,
                "temporal_index_sha256": (
                    "80d977e40e113172e778b6f78315e1fad132423c9fe9eca3ad6a5de34df4f21b"
                ),
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


def _annotation_views(
    item_rows: list[dict[str, Any]],
    cache_root: Path,
    *,
    task_prompts: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build bounded annotation views, optionally including task prompts.

    The v1 pilot deliberately omits prompts so its temporal-only Labeler A
    pass remains unchanged.  The v2 relevance pass supplies a task-id keyed
    prompt mapping; only that prompt is added, never expected output or
    ground truth.
    """

    views: list[dict[str, Any]] = []
    for item in item_rows:
        cache_file = cache_root / "candidates" / f"{item['item_id']}.json"
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PilotCollectionError(
                "pilot candidate payload is unavailable while building the blind view"
            ) from exc
        raw_candidate = cached.get("candidate")
        if not isinstance(raw_candidate, Mapping):
            raise PilotCollectionError("pilot candidate payload is malformed")
        candidate = VolcSearchCandidate.from_volc_result(raw_candidate)
        if _sha256_canonical(candidate.as_model_payload()) != item["model_visible_payload_sha256"]:
            raise PilotCollectionError(
                "pilot candidate payload does not match the receipt-bound item"
            )
        if _canonical_url(candidate.url) != item["canonical_url"]:
            raise PilotCollectionError("pilot candidate URL does not match the receipt-bound item")
        published_at = _raw_provider_time(raw_candidate, "PublishTime")
        updated_at = _raw_provider_time(raw_candidate, "UpdateTime")
        if published_at != item["provider_published_at_utc"]:
            raise PilotCollectionError(
                "pilot publication evidence does not match the receipt-bound item"
            )
        if updated_at != item["provider_updated_at_utc"]:
            raise PilotCollectionError(
                "pilot update evidence does not match the receipt-bound item"
            )
        publish_evidence = (
            _authoritative_time_evidence(
                kind="publisher",
                source_field="volc.PublishTime",
                timestamp_utc=published_at,
                locator=item["canonical_url"],
                observed_at_utc=item["snapshot_captured_at_utc"],
            )
            if published_at is not None
            else None
        )
        view = {
            "authoritative_history_evidence": [],
            "authoritative_history_status": (
                "provider_update_time_only" if updated_at is not None else "not_available"
            ),
            "authoritative_publish_evidence": publish_evidence,
            "authoritative_publish_time_utc": published_at,
            "authoritative_update_time_utc": updated_at,
            "canonical_url": item["canonical_url"],
            "canonical_url_sha256": item["canonical_url_sha256"],
            "canonical_url_version": item["canonical_url_version"],
            "category": item["category"],
            "domain": item["domain"],
            "item_id": item["item_id"],
            "model_visible_content": candidate.content,
            "model_visible_payload_sha256": item["model_visible_payload_sha256"],
            "model_visible_snippet": candidate.snippet,
            "model_visible_title": candidate.title,
            "paper_band": item["paper_band"],
            "restricted_snapshot_locator": item["restricted_snapshot_locator"],
            "restricted_snapshot_sha256": item["restricted_snapshot_sha256"],
            "restricted_snapshot_status": item["restricted_snapshot_status"],
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "task_cutoff_date": item["task_cutoff_date"],
            "task_id": item["task_id"],
        }
        if task_prompts is not None:
            task_prompt = task_prompts.get(str(item["task_id"]))
            if not isinstance(task_prompt, str) or not task_prompt.strip():
                raise PilotCollectionError(
                    f"task prompt is unavailable for {item['task_id']} while building the view"
                )
            view["task_prompt"] = task_prompt
        view["annotation_view_sha256"] = _sha256_canonical(view)
        views.append(view)
    views.sort(key=lambda row: row["item_id"])
    return views


def _authoritative_time_evidence(
    *,
    kind: str,
    source_field: str,
    timestamp_utc: str,
    locator: str,
    observed_at_utc: str,
) -> dict[str, str]:
    return {
        "kind": kind,
        "locator": locator,
        "observed_at_utc": observed_at_utc,
        "sha256": sha256_text(f"{source_field}|{timestamp_utc}|{locator}"),
        "source_field": source_field,
        "timestamp_utc": timestamp_utc,
    }


def _label_evidence_view(value: Mapping[str, Any]) -> dict[str, str]:
    return {
        "kind": str(value["kind"]),
        "locator": str(value["locator"]),
        "observed_at_utc": str(value["observed_at_utc"]),
        "sha256": str(value["sha256"]),
    }


def _label_pilot_blind(root: Path, *, reviewer_id: str = "dev-labeler-a-pilot40") -> int:
    items = _read_jsonl(root / "items.jsonl")
    views = {row["item_id"]: row for row in _read_jsonl(root / "annotation-view.jsonl")}
    rubric_sha256 = _sha256_file(root / "rubric.md")
    labels: list[dict[str, Any]] = []
    for item in items:
        item_id = item["item_id"]
        view = views.get(item_id)
        if view is None:
            raise PilotCollectionError("pilot annotation view is incomplete")
        temporal_label, evidence, rationale = _blind_temporal_label(item, view)
        task_relevance = _blind_task_relevance(view)
        outcome_revealing = _blind_outcome_revealing(view)
        if task_relevance == "indeterminate":
            confidence = 0.5 if temporal_label != "indeterminate" else 0.25
        else:
            confidence = 0.85 if outcome_revealing != "indeterminate" else 0.7
            if temporal_label == "indeterminate":
                confidence = min(confidence, 0.7)
        labels.append(
            {
                "annotation_view_sha256": view["annotation_view_sha256"],
                "confidence": confidence,
                "evidence": evidence,
                "item_id": item_id,
                "labeled_at_utc": _now_utc(),
                "outcome_revealing": outcome_revealing,
                "rationale": (
                    f"{rationale} Task relevance is {task_relevance}; "
                    f"outcome exposure is {outcome_revealing}."
                )[:500],
                "reviewer_id": reviewer_id,
                "reviewer_type": "labeler_a",
                "rubric_sha256": rubric_sha256,
                "rubric_version": "leakage-calibration-rubric/v1",
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "task_relevance": task_relevance,
                "temporal_label": temporal_label,
            }
        )
    labels.sort(key=lambda row: row["item_id"])
    _write_jsonl(root / "labels/raw/annotator-a.jsonl", labels)
    return len(labels)


def _blind_temporal_label(
    item: Mapping[str, Any], view: Mapping[str, Any]
) -> tuple[str, list[dict[str, Any]], str]:
    cutoff = date.fromisoformat(str(item["task_cutoff_date"]))
    bound = datetime.fromisoformat(str(item["exclusive_cutoff_bound"][:-1]) + "+00:00")
    published_raw = view.get("authoritative_publish_time_utc")
    updated_raw = view.get("authoritative_update_time_utc")
    published = _label_timestamp(published_raw)
    updated = _label_timestamp(updated_raw)
    evidence: list[dict[str, Any]] = []
    publish_evidence = view.get("authoritative_publish_evidence")
    if isinstance(publish_evidence, Mapping):
        evidence.append(_label_evidence_view(publish_evidence))
    history_evidence = view.get("authoritative_history_evidence")
    if isinstance(history_evidence, list):
        evidence.extend(
            _label_evidence_view(value) for value in history_evidence if isinstance(value, Mapping)
        )
    if published is None:
        return (
            "indeterminate",
            evidence,
            "No timezone-qualified provider publication timestamp was available.",
        )
    if published >= bound:
        return (
            "post_cutoff",
            evidence,
            "Provider publication timestamp is after the task cutoff.",
        )
    if updated is not None and updated >= bound:
        return (
            "indeterminate",
            evidence,
            "The page has a post-cutoff update without version-history evidence.",
        )
    return (
        "pre_or_on_cutoff",
        evidence,
        f"Provider publication timestamp is on or before {cutoff.isoformat()}.",
    )


def _blind_task_relevance(view: Mapping[str, Any]) -> str:
    task_prompt = view.get("task_prompt")
    if not isinstance(task_prompt, str) or not task_prompt.strip():
        # v1's temporal-only view intentionally has no task prompt.
        return "indeterminate"
    terms = _task_entity_terms(task_prompt)
    candidate_text = _annotation_view_text(view)
    if not terms or not candidate_text:
        return "indeterminate"
    return "relevant" if any(term in candidate_text for term in terms) else "not_relevant"


def _blind_outcome_revealing(view: Mapping[str, Any]) -> str:
    """Label outcome exposure from the task prompt and candidate text only."""

    relevance = _blind_task_relevance(view)
    if relevance == "indeterminate":
        return "indeterminate"
    if relevance == "not_relevant":
        return "no"
    task_prompt = str(view.get("task_prompt", "")).casefold()
    candidate_text = _annotation_view_text(view)
    if any(token in task_prompt for token in ("事件", "surprise", "方向", "event")):
        direct_terms = (
            "eps",
            "surprise",
            "earnings",
            "quarter",
            "财报",
            "业绩",
            "净利润",
            "净利",
            "营收",
            "收入",
            "cpi",
            "pmi",
            "非农",
            "就业",
        )
    elif "近两个季度" in task_prompt or "past two quarters" in task_prompt:
        metric_terms = {
            "净利润": ("净利润", "净利", "net profit", "net income", "profit"),
            "营业收入": ("营业收入", "营收", "operating revenue", "revenue"),
            "毛利率": ("毛利率", "gross margin"),
            "净利率": ("净利率", "净利润率", "net margin"),
        }
        direct_terms = ()
        for metric_name, terms in metric_terms.items():
            if metric_name in task_prompt or any(
                term in task_prompt for term in terms if len(term) > 2
            ):
                direct_terms = terms
                break
    else:
        direct_terms = (
            "目标价",
            "股价",
            "收盘价",
            "市值",
            "估值",
            "target price",
            "share price",
            "stock price",
            "valuation",
        )
    return "direct" if any(term in candidate_text for term in direct_terms) else "indirect"


def _task_entity_terms(task_prompt: str) -> tuple[str, ...]:
    """Extract only task entity anchors needed for a blind relevance label."""

    compact = re.sub(r"\s+", " ", task_prompt).strip()
    terms: list[str] = []
    for pattern in (
        r"对\s*([^（(]+?)\s*[（(]",
        r"股票[:：]\s*([^（(]+?)\s*[（(]",
        r"以下是\s*(.+?)\s*近两个季度",
    ):
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            name = re.sub(r"\s+", "", match.group(1)).strip(" ：:,，")
            if name:
                terms.append(name.casefold())
    for code in re.findall(r"[（(]([A-Za-z]{1,8}|[0-9]{4,6})[）)]", compact):
        terms.append(code.casefold())
    return tuple(dict.fromkeys(term for term in terms if term))


def _annotation_view_text(view: Mapping[str, Any]) -> str:
    return " ".join(
        str(view.get(field) or "")
        for field in ("model_visible_title", "model_visible_snippet", "model_visible_content")
    ).casefold()


def _label_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        parsed = parse_volc_publish_time(raw)
    elif re.search(r"[+-][0-9]{2}:[0-9]{2}$", raw):
        parsed = parse_volc_publish_time(raw)
    else:
        return None
    if parsed is None or parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _filter_state(decision: Any) -> tuple[str, str]:
    if decision.reason in {"missing_url", "empty_candidate_text"}:
        return "missing", "missing"
    if decision.accepted:
        return "keep", "accept"
    if decision.rule_verdict == "ambiguous":
        return "ambiguous", "ambiguous"
    return "reject", "reject"


def _build_query(row: Mapping[str, Any], query_class: str, cutoff_date: str) -> str:
    category = str(row.get("category", ""))
    seed = row.get("seed")
    seed = seed if isinstance(seed, Mapping) else {}
    fields: list[Any]
    if category == "A1":
        fields = [seed.get("stock_name"), seed.get("market"), "financial news"]
    elif category == "B":
        fields = [
            seed.get("stock_name") or seed.get("macro_indicator"),
            seed.get("market"),
            seed.get("event_subtype"),
            "financial event",
        ]
    else:
        fields = [
            seed.get("stock_name"),
            seed.get("market"),
            seed.get("metric_name"),
            "financial information",
        ]
    topic = " ".join(_bounded_query_token(value) for value in fields if _bounded_query_token(value))
    if not topic:
        topic = f"{category} financial information"
    suffix = {
        "cutoff_anchored": f"before {cutoff_date}",
        "neutral_entity_topic": "historical financial information",
        "retrospective_outcome_challenge": "historical result context",
    }[query_class]
    query = f"{topic} {suffix}".strip()
    if len(query) > 512:
        raise PilotCollectionError("generated pilot query exceeded the bounded query contract")
    return query


def _bounded_query_token(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    normalized = " ".join(value.split())
    if not normalized or any(ord(char) < 32 for char in normalized):
        return ""
    return normalized[:80]


def _canonical_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    parsed = urlsplit(raw)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return None
    if parsed.username or parsed.password or parsed.fragment or any(char.isspace() for char in raw):
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port == 443:
        return None
    for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
        lower = key.lower()
        if any(word in lower for word in _CREDENTIAL_QUERY_WORDS) or lower.endswith("key"):
            return None
    try:
        hostname = parsed.hostname.encode("idna").decode("ascii").lower()
    except UnicodeError:
        return None
    if ":" in hostname:
        netloc = f"[{hostname}]"
    else:
        netloc = hostname
    if port is not None:
        netloc += f":{port}"
    path = _canonical_path(parsed.path)
    if path is None:
        return None
    return urlunsplit(("https", netloc, path, parsed.query, ""))


def _canonical_path(path: str) -> str | None:
    output: list[str] = []
    index = 0
    while index < len(path):
        char = path[index]
        if char == "%":
            if index + 2 >= len(path) or path[index + 1] not in _HEX or path[index + 2] not in _HEX:
                return None
            decoded = int(path[index + 1 : index + 3], 16)
            if chr(decoded) in _UNRESERVED:
                output.append(chr(decoded))
            else:
                output.append(f"%{path[index + 1 : index + 3].upper()}")
            index += 3
        else:
            if ord(char) > 127:
                encoded = char.encode("utf-8")
                output.extend(f"%{byte:02X}" for byte in encoded)
            else:
                output.append(char)
            index += 1
    return "".join(output)


def _candidate_text(candidate: VolcSearchCandidate) -> str:
    return "\n".join(
        value.strip()
        for value in (candidate.title, candidate.snippet, candidate.content)
        if isinstance(value, str) and value.strip()
    )


def _raw_provider_time(raw_candidate: Mapping[str, Any], name: str) -> str | None:
    parsed = _label_timestamp(raw_candidate.get(name))
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z") if parsed else None


def _provisional_item_id(candidate: Mapping[str, Any]) -> str:
    return sha256_text(
        "\0".join(
            (
                "retrieval_leakage_candidate/v1",
                str(candidate["receipt_sha256"]),
                str(candidate["candidate_index"]),
                sha256_text(str(candidate["canonical_url"])),
                _sha256_canonical(candidate["candidate"].as_model_payload()),
                str(candidate["task_id"]),
                str(candidate["cutoff_date"]),
            )
        )
    )


def _sampling_key(item_id: str) -> str:
    return sha256_text(f"leakage-calibration/v1|{HUB_REVISION}|{item_id}")


def _date_text(value: Any, name: str, task_id: str) -> str:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise PilotCollectionError(f"Hub {name} is invalid for pilot task {task_id}") from exc
        if parsed.isoformat() == value:
            return value
    raise PilotCollectionError(f"Hub {name} is unavailable for pilot task {task_id}")


def _unrestricted_time_range(cutoff_date: str) -> str:
    cutoff = date.fromisoformat(cutoff_date)
    try:
        start = cutoff.replace(year=cutoff.year - 5)
    except ValueError:
        start = cutoff.replace(year=cutoff.year - 5, day=28)
    end = max(cutoff, datetime.now(UTC).date())
    return f"{start.isoformat()}..{end.isoformat()}"


def _call_id(stratum: str, task_number: int, query_class: str) -> str:
    value = f"pilot40-{stratum.lower().replace('-', '')}-{task_number:02d}-{query_class}"
    if not _SAFE_ID.fullmatch(value):
        raise PilotCollectionError("pilot call identity is unsafe")
    return value


def _write_cache_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_private(path.parent)
    payload = canonical_json_bytes(value)
    path.write_bytes(payload)
    path.chmod(0o600)


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _cache_entry(path: Path, cache_root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "bytes": len(payload),
        "path": path.relative_to(cache_root).as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_evidence_manifest(cache_root: Path, entries: list[dict[str, Any]]) -> str:
    payload = {
        "authorization_id": PILOT_AUTHORIZATION_ID,
        "calibration_id": CALIBRATION_ID,
        "collection_id": PILOT_COLLECTION_ID,
        "entries": sorted(entries, key=lambda entry: entry["path"]),
        "schema_version": "leakage-calibration-restricted-evidence/v1",
    }
    path = cache_root / "restricted-evidence-manifest.json"
    _write_cache_json(path, payload)
    return _sha256_file(path)


def _refresh_tracked_state(
    root: Path, *, status: str, restricted_evidence_manifest_sha256: str
) -> None:
    cohort_path = root / "cohort.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cohort["calibration_status"] = status
    cohort_path.write_bytes(canonical_json_bytes(cohort))
    sampling_path = root / "sampling-plan.json"
    sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
    sampling["status"] = "pilot"
    sampling_path.write_bytes(canonical_json_bytes(sampling))
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
    report_path = root / "reports/calibration-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        {
            "calibration_status": status,
            "claim_eligible": False,
            "cohen_kappa": None,
            "counts": counts,
            "exclusions_by_type": {"pilot_not_adjudicated": len(items)},
            "filter_validated": bool(items and adjudicated),
            "generated_at_utc": _now_utc(),
            "quality_gates_passed": False,
            "raw_agreement": None,
        }
    )
    report_path.write_bytes(canonical_json_bytes(report))
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["calibration_status"] = status
    manifest["restricted_evidence_manifest_sha256"] = restricted_evidence_manifest_sha256
    manifest["adjudication"] = {
        "adjudicated_observed": len(adjudicated),
        "raw_labels_expected": 800,
        "raw_labels_observed": len(labels_a) + len(labels_b),
        "status": "pilot",
        "unresolved_disputes": 0,
    }
    for reference_name, relative in (
        ("rubric", "rubric.md"),
        ("sampling_plan", "sampling-plan.json"),
        ("query_corpus", "query-corpus.jsonl"),
    ):
        manifest[reference_name]["sha256"] = _sha256_file(root / relative)
    entries: list[dict[str, Any]] = []
    for relative in EXPECTED_TRACKED_FILES:
        path = root / relative
        payload = path.read_bytes()
        row_count: int | None
        if relative in JSONL_FILES:
            row_count = len([line for line in payload.decode("utf-8").splitlines() if line.strip()])
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


def _validate_pilot_state(root: Path, *, expected_items: int, expected_labels: int) -> None:
    result = validate_calibration_root(root)
    if result["counts"]["items"] != expected_items:
        raise PilotCollectionError("pilot item count failed validation")
    if result["counts"]["raw_labels_a"] != expected_labels:
        raise PilotCollectionError("pilot Labeler A count failed validation")


def _manifest_restricted_evidence_sha(root: Path) -> str:
    try:
        manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotCollectionError("pilot artifact manifest is unavailable") from exc
    value = manifest.get("restricted_evidence_manifest_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise PilotCollectionError("pilot restricted evidence manifest hash is unavailable")
    return value


def _sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _chmod_private(path: Path) -> None:
    try:
        path.chmod(0o700)
    except OSError:
        pass


__all__ = ["PilotCollectionError", "PilotResult", "label_pilot", "run_pilot"]
