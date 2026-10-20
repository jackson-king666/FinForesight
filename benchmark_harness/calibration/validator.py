"""Fail-closed validation for the versioned leakage-calibration bundle.

The validator is deliberately offline.  It verifies the tracked bundle and
never opens a network client, reads credentials, or materializes the restricted
calibration cache.  The blind annotation view may expose the bounded
model-visible candidate text needed by independent labelers; full provider
requests and responses remain cache-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from benchmark_harness.contracts import sha256_json, sha256_text

CALIBRATION_SCHEMA_VERSION = "leakage-calibration/v1"
CALIBRATION_MANIFEST_SCHEMA_VERSION = "leakage-calibration-manifest/v1"
CALIBRATION_ID = "leakage-probe/v1"
FROZEN_CALIBRATION_ID = "leakage-probe/v2"
RELEVANCE_CALIBRATION_ID = "leakage-probe/v3"
FRESH_RELEVANCE_CALIBRATION_ID = "leakage-probe/v4"
V5_RELEVANCE_CALIBRATION_ID = "leakage-probe/v5"
FRESH_RELEVANCE_COLLECTION_ID = "leakage-probe-v4-relevance-rubric-20260831"
FRESH_RELEVANCE_RUBRIC_VERSION = "leakage-calibration-rubric/v3"
FRESH_RELEVANCE_AUTHORIZATION_ID = "leakage-probe-v4-relevance-rubric-20260831"
FRESH_RELEVANCE_SCOPE = "calibration_relevance_rubric_v3_fresh_pilot_collection_and_labeling"
FRESH_RELEVANCE_SOURCE_COMMENT_ID = "01a05706-515a-733c-a653-1e5da6da369e"
V5_RELEVANCE_RUBRIC_VERSION = "leakage-calibration-rubric/v4"
V5_RELEVANCE_COLLECTION_ID = "leakage-probe-v5-relevance-rubric-v4-pilot40-20260831"
V5_RELEVANCE_AUTHORIZATION_ID = V5_RELEVANCE_COLLECTION_ID
V5_RELEVANCE_SCOPE = "calibration_relevance_rubric_v4_fresh_pilot_collection_and_labeling"
V5_RELEVANCE_SOURCE_COMMENT_ID = "01a0578c-9484-7941-9110-ffdac992e25c"
V6_RELEVANCE_CALIBRATION_ID = "leakage-probe/v6"
V6_RELEVANCE_RUBRIC_VERSION = "leakage-calibration-rubric/v5"
V6_RELEVANCE_COLLECTION_ID = "leakage-probe-v6-relevance-rubric-v5-pilot40-20260831"
V6_RELEVANCE_AUTHORIZATION_ID = V6_RELEVANCE_COLLECTION_ID
V6_RELEVANCE_SCOPE = "calibration_relevance_rubric_v5_fresh_pilot_collection_and_labeling"
V6_RELEVANCE_SOURCE_COMMENT_ID = "01a05851-7603-78b4-9486-a01f237f931d"
V7_RELEVANCE_CALIBRATION_ID = "leakage-probe/v7"
V7_RELEVANCE_RUBRIC_VERSION = "leakage-calibration-rubric/v6"
V7_RELEVANCE_COLLECTION_ID = "leakage-probe-v7-relevance-rubric-v6-pilot40-20260831"
V7_RELEVANCE_AUTHORIZATION_ID = V7_RELEVANCE_COLLECTION_ID
V7_RELEVANCE_SCOPE = (
    "calibration_relevance_rubric_v6_task_scope_v2_fresh_pilot_collection_and_labeling"
)
V7_RELEVANCE_SOURCE_COMMENT_ID = "01a05893-b0bd-7c6b-9f1f-10affe45d68d"
V7_PILOT_QUERY_CLASSES = frozenset(
    {
        "v7_neutral_entity_topic",
        "v7_cutoff_anchored",
        "v7_retrospective_outcome_challenge",
    }
)
CALIBRATION_IDS = frozenset(
    {
        CALIBRATION_ID,
        FROZEN_CALIBRATION_ID,
        RELEVANCE_CALIBRATION_ID,
        FRESH_RELEVANCE_CALIBRATION_ID,
        V5_RELEVANCE_CALIBRATION_ID,
        V6_RELEVANCE_CALIBRATION_ID,
        V7_RELEVANCE_CALIBRATION_ID,
    }
)
HUB_REPO_ID = "sselaine27/benchmark-research"
HUB_REVISION = "941370f228b9bc03c1b8bd9547c098c42c06be23"

EXPECTED_TRACKED_FILES = (
    "README.md",
    "annotation-view.jsonl",
    "authorization.json",
    "cohort.json",
    "filter-decisions.jsonl",
    "items.jsonl",
    "labels/adjudicated.jsonl",
    "labels/raw/annotator-a.jsonl",
    "labels/raw/annotator-b.jsonl",
    "query-corpus.jsonl",
    "reports/calibration-report.json",
    "rubric.md",
    "sampling-plan.json",
    "schemas/adjudication.schema.json",
    "schemas/authorization.schema.json",
    "schemas/cohort.schema.json",
    "schemas/item.schema.json",
    "schemas/raw-label.schema.json",
    "schemas/report.schema.json",
)
JSON_FILES = frozenset(
    {
        "cohort.json",
        "authorization.json",
        "sampling-plan.json",
        "reports/calibration-report.json",
        "schemas/adjudication.schema.json",
        "schemas/authorization.schema.json",
        "schemas/cohort.schema.json",
        "schemas/item.schema.json",
        "schemas/raw-label.schema.json",
        "schemas/report.schema.json",
    }
)
JSONL_FILES = frozenset(
    {
        "annotation-view.jsonl",
        "filter-decisions.jsonl",
        "items.jsonl",
        "labels/adjudicated.jsonl",
        "labels/raw/annotator-a.jsonl",
        "labels/raw/annotator-b.jsonl",
        "query-corpus.jsonl",
    }
)
V3_PILOT_JSONL_FILES = frozenset(
    {
        "pilot/annotation-view.jsonl",
        "pilot/filter-decisions.jsonl",
        "pilot/items.jsonl",
        "pilot/labels/raw/annotator-a.jsonl",
    }
)
V3_TRACKED_FILES = tuple(sorted((*EXPECTED_TRACKED_FILES, *V3_PILOT_JSONL_FILES)))
V3_JSON_FILES = JSON_FILES
V3_JSONL_FILES = JSONL_FILES | V3_PILOT_JSONL_FILES
V4_PILOT_JSONL_FILES = V3_PILOT_JSONL_FILES | {"pilot/query-corpus.jsonl"}
V4_TRACKED_FILES = tuple(sorted((*EXPECTED_TRACKED_FILES, *V4_PILOT_JSONL_FILES)))
V4_JSON_FILES = JSON_FILES
V4_JSONL_FILES = JSONL_FILES | V4_PILOT_JSONL_FILES
FRESH_RELEVANCE_PILOT_JSONL_FILES = V4_PILOT_JSONL_FILES
FRESH_RELEVANCE_TRACKED_FILES = V4_TRACKED_FILES
V5_PILOT_JSONL_FILES = V4_PILOT_JSONL_FILES
V5_TRACKED_FILES = V4_TRACKED_FILES
V5_JSON_FILES = JSON_FILES
V5_JSONL_FILES = V4_JSONL_FILES
V6_PILOT_JSONL_FILES = V4_PILOT_JSONL_FILES
V6_JSON_FILES = JSON_FILES | {"pilot/exclusion-proof.json"}
V6_JSONL_FILES = JSONL_FILES | V6_PILOT_JSONL_FILES
V6_TRACKED_FILES = tuple(
    sorted((*EXPECTED_TRACKED_FILES, *V6_PILOT_JSONL_FILES, "pilot/exclusion-proof.json"))
)
V7_PILOT_JSONL_FILES = V6_PILOT_JSONL_FILES
V7_JSON_FILES = V6_JSON_FILES | {
    "formula-table.json",
    "pilot/exclusion-proof.json",
    "schemas/conformance-fixtures.json",
}
V7_JSONL_FILES = JSONL_FILES | V7_PILOT_JSONL_FILES
V7_TRACKED_FILES = tuple(
    sorted(
        (
            *EXPECTED_TRACKED_FILES,
            *V7_PILOT_JSONL_FILES,
            "formula-table.json",
            "pilot/exclusion-proof.json",
            "schemas/conformance-fixtures.json",
        )
    )
)
SCHEMA_FILES = frozenset(path for path in EXPECTED_TRACKED_FILES if path.startswith("schemas/"))

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HUB_SHA = re.compile(r"^[0-9a-f]{40}$")
_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_UTC_DATETIME = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$")
_MIME = re.compile(r"^[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+$")

_CATEGORY_VALUES = frozenset({"A1", "A2-F", "A2-T", "A2-H", "B", "C", "D", "E"})
_PAPER_BANDS = frozenset({"T1", "T2", "T3", "D", "E"})
_STRATA = frozenset({"A1-T1", "B-T1", "B-T2", "C-T2"})
_TEMPORAL_LABELS = frozenset(
    {"pre_or_on_cutoff", "post_cutoff", "mixed", "indeterminate", "invalid"}
)
_TASK_RELEVANCE = frozenset({"relevant", "not_relevant", "indeterminate"})
_OUTCOME_REVEALING = frozenset({"direct", "indirect", "no", "indeterminate"})
_FILTER_STATES = frozenset({"keep", "reject", "ambiguous", "missing"})
_FILTER_VERDICTS = frozenset({"accept", "reject", "ambiguous", "missing"})
_SNAPSHOT_STATES = frozenset({"available", "missing", "indeterminate", "not_collected"})
_CALIBRATION_STATUSES = frozenset(
    {"not_started", "pilot", "incomplete", "filter_not_validated", "validated"}
)
_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "body",
        "cookie",
        "cookies",
        "content",
        "email",
        "full_snippet",
        "ground_truth",
        "header",
        "headers",
        "model_answer",
        "original_url",
        "page_body",
        "prompt",
        "provider_response",
        "raw_prompt",
        "raw_response",
        "response",
        "reviewer_email",
        "reviewer_name",
        "secret",
        "signed_url",
        "snippet",
        "token",
        "tokens",
    }
)

_MANIFEST_FIELDS = frozenset(
    {
        "adjudication",
        "calibration_id",
        "calibration_status",
        "clocks",
        "files",
        "hub",
        "live_unrestricted_authorized",
        "query_corpus",
        "restricted_evidence_manifest_sha256",
        "rubric",
        "sampling_plan",
        "schema_version",
    }
)
_COHORT_FIELDS = frozenset(
    {
        "backbone_id",
        "backbone_training_cutoff",
        "calibration_status",
        "cohort_id",
        "experiment_as_of",
        "guard_days",
        "hub_repo_id",
        "hub_revision",
        "live_unrestricted_authorized",
        "schema_version",
        "target_adjudicated_count",
        "target_item_count",
        "target_raw_label_count",
        "task_cutoff_semantics",
        "temporal_index_sha256",
    }
)
_AUTHORIZATION_FIELDS = frozenset(
    {
        "authorization_id",
        "calibration_id",
        "calibration_unrestricted_collection_authorized",
        "frozen_item_count",
        "frozen_strata",
        "full_paper_unrestricted_authorized",
        "issued_at_utc",
        "issue_id",
        "pilot_item_count",
        "pilot_strata",
        "same_candidates_cutoff_filter",
        "schema_version",
        "scope",
        "source_comment_id",
        "status",
    }
)
_SAMPLING_FIELDS = frozenset(
    {
        "domain_cap",
        "frozen_set",
        "pilot",
        "schema_version",
        "sort_key",
        "status",
        "strata",
        "task_cap",
    }
)
_ITEM_FIELDS = frozenset(
    {
        "backbone_id",
        "backbone_training_cutoff",
        "call_id",
        "candidate_index",
        "canonical_url",
        "canonical_url_sha256",
        "canonical_url_version",
        "category",
        "collection_id",
        "content_sha256",
        "cutoff_declaration_sha256",
        "deterministic_rank",
        "domain",
        "exclusive_cutoff_bound",
        "experiment_as_of",
        "forecast_origin",
        "hub_revision",
        "hub_repo_id",
        "inclusion_probability",
        "item_id",
        "model_visible_payload_sha256",
        "original_url_sha256",
        "paper_band",
        "policy_sha256",
        "previous_receipt_sha256",
        "provider_id",
        "provider_published_at_utc",
        "provider_updated_at_utc",
        "query_sha256",
        "receipt_protocol",
        "receipt_sha256",
        "request_filter_sha256",
        "response_sha256",
        "retrieved_at_utc",
        "restricted_snapshot_bytes",
        "restricted_snapshot_locator",
        "restricted_snapshot_mime",
        "restricted_snapshot_sha256",
        "restricted_snapshot_status",
        "sampling_stratum",
        "schema_version",
        "service_cohort_sha256",
        "snapshot_captured_at_utc",
        "snippet_sha256",
        "task_cutoff_date",
        "task_id",
        "task_scope_sha256",
        "title_sha256",
        "tool_contract_sha256",
        "temporal_index_sha256",
    }
)
_VIEW_FIELDS = frozenset(
    {
        "annotation_view_sha256",
        "authoritative_history_evidence",
        "authoritative_history_status",
        "authoritative_publish_evidence",
        "authoritative_publish_time_utc",
        "authoritative_update_time_utc",
        "canonical_url",
        "canonical_url_sha256",
        "canonical_url_version",
        "category",
        "domain",
        "item_id",
        "model_visible_content",
        "model_visible_payload_sha256",
        "model_visible_snippet",
        "model_visible_title",
        "paper_band",
        "restricted_snapshot_locator",
        "restricted_snapshot_sha256",
        "restricted_snapshot_status",
        "schema_version",
        "task_cutoff_date",
        "task_id",
        "task_prompt",
    }
)
_VIEW_EVIDENCE_FIELDS = frozenset(
    {"kind", "locator", "observed_at_utc", "sha256", "source_field", "timestamp_utc"}
)
_VIEW_HISTORY_STATUSES = frozenset({"not_available", "provider_update_time_only"})
_MAX_VIEW_TEXT_CHARS = 100_000
_V6_VIEW_FIELDS = frozenset(
    {
        "annotation_view_sha256",
        "category",
        "item_id",
        "model_visible_content",
        "model_visible_payload_sha256",
        "model_visible_snippet",
        "model_visible_title",
        "paper_band",
        "schema_version",
        "task_id",
        "task_scope",
        "truncation_metadata",
    }
)
_V6_SCOPE_FIELDS = frozenset(
    {
        "a1_target_type",
        "b_event_class",
        "b_event_date",
        "b_event_fingerprint",
        "b_reporting_package",
        "c_input_period",
        "c_target_metric",
        "c_target_period",
        "entity_name",
        "entity_ticker",
        "family",
        "normalizer_version",
        "requested_output",
        "source_spans",
        "task_scope_sha256",
    }
)
_V6_SPAN_FIELDS = frozenset({"field", "span"})
_V6_TRUNCATION_FIELDS = frozenset(
    {
        "content_bytes",
        "content_ends_at_limit",
        "content_limit_bytes",
        "hidden_axis",
        "no_decisive_visible_span",
        "proven_visible_truncation",
        "provider_truncated",
        "token_boundary",
    }
)
_V6_RCLASSES = frozenset(
    {
        "A1_DIRECT",
        "A1_INPUT",
        "A1_OTHER",
        "B_CROSS_PERIOD",
        "B_EXACT_DIRECT",
        "B_EXACT_JUDGMENT",
        "B_EXACT_NO_BRIDGE",
        "B_NOT_PROVEN",
        "C_COMPONENT_OR_DRIVER",
        "C_EXACT_PERIOD",
        "C_OTHER_PERIOD_OR_FORECAST",
        "C_TARGET_NO_VALUE",
        "C_UNRELATED",
        "ENTITY_MISMATCH",
        "TRUNCATED_SCOPE_ENTITY",
        "TRUNCATED_SCOPE_EVENT",
        "TRUNCATED_SCOPE_METRIC",
        "TRUNCATED_SCOPE_PERIOD",
    }
)
_V6_OCLASSES = frozenset(
    {"DIRECT_TARGET", "DIRECT_POST_EVENT_MOVE", "INDIRECT_INPUT", "NO_SIGNAL", "TRUNCATED_SCOPE"}
)
_V6_EVENT_MATCHES = frozenset({"exact", "contains_exact", "cross_period", "not_proven", "na"})
_V6_METRIC_MATCHES = frozenset({"exact", "component", "other", "none", "na"})
_V6_HIDDEN_AXES = frozenset({"none", "entity", "event", "period", "metric"})
_V6_EXCLUSION_FIELDS = frozenset(
    {
        "calibration_id",
        "excluded_digests",
        "schema_version",
        "selected_digests",
        "source_calibrations",
        "status",
    }
)
_V6_DIGEST_FIELDS = frozenset(
    {
        "annotation_view_sha256",
        "canonical_url_sha256",
        "item_id",
        "model_visible_payload_sha256",
        "query_sha256",
        "receipt_sha256",
        "semantic_key_sha256",
        "url_payload_sha256",
    }
)
_V7_SCOPE_FIELDS = frozenset(
    {
        "a1_target_type",
        "b_event_class",
        "b_match_anchors",
        "c_input_period",
        "c_target_metric",
        "c_target_period",
        "entity_name",
        "entity_ticker",
        "family",
        "normalizer_version",
        "requested_output",
        "source_spans",
        "task_scope_sha256",
    }
)
_V7_ANCHOR_FIELDS = frozenset(
    {"event_class", "release_date", "reporting_period", "fingerprint_facts"}
)
_V7_TRUNCATION_FIELDS = _V6_TRUNCATION_FIELDS
_V7_UNIT_FIELDS = frozenset({"field", "start_char", "end_char", "exact_text_sha256"})
_V7_VIEW_FIELDS = frozenset(
    {
        "annotation_view_sha256",
        "category",
        "evidence_units",
        "item_id",
        "model_visible_content",
        "model_visible_payload_sha256",
        "model_visible_snippet",
        "model_visible_title",
        "paper_band",
        "schema_version",
        "task_id",
        "task_scope",
        "truncation_metadata",
    }
)
_V7_RCLASSES = frozenset(
    {
        "A1_NO_SIGNAL",
        "A1_REQUESTED_ENDPOINT",
        "A1_VALUATION_INPUT",
        "B_EVENT_UNPROVEN",
        "B_EXACT_DIRECT",
        "B_EXACT_JUDGMENT",
        "B_EXACT_NO_BRIDGE",
        "C_DECLARED_COMPONENT",
        "C_EXACT_TARGET",
        "C_EXPLICIT_DRIVER",
        "C_FORECAST",
        "C_OTHER_PERIOD",
        "C_TARGET_NO_VALUE",
        "C_UNRELATED",
        "ENTITY_MISMATCH",
        "TRUNCATED_SCOPE_ENTITY",
        "TRUNCATED_SCOPE_EVENT",
        "TRUNCATED_SCOPE_METRIC",
        "TRUNCATED_SCOPE_PERIOD",
    }
)
_V7_OCLASSES = frozenset(
    {"DIRECT_TARGET", "DIRECT_POST_EVENT_MOVE", "INDIRECT_INPUT", "NO_SIGNAL", "TRUNCATED_SCOPE"}
)
_V7_EVENT_MATCHES = frozenset({"exact", "other_event", "event_unproven", "not_proven", "na"})
_V7_METRIC_MATCHES = frozenset({"exact", "component", "other", "none", "na"})
_V7_HIDDEN_AXES = frozenset({"none", "entity", "event", "period", "metric"})
_V7_EXCLUSION_FIELDS = _V6_EXCLUSION_FIELDS
_V7_DIGEST_FIELDS = _V6_DIGEST_FIELDS
_FILTER_FIELDS = frozenset(
    {
        "decided_at_utc",
        "filter_policy_sha256",
        "filter_state",
        "filter_verdict",
        "item_id",
        "schema_version",
    }
)
_RAW_LABEL_FIELDS = frozenset(
    {
        "annotation_view_sha256",
        "confidence",
        "evidence",
        "item_id",
        "labeled_at_utc",
        "outcome_revealing",
        "rationale",
        "reviewer_id",
        "reviewer_type",
        "rubric_sha256",
        "rubric_version",
        "schema_version",
        "task_relevance",
        "temporal_label",
    }
)
_V6_RAW_LABEL_FIELDS = _RAW_LABEL_FIELDS | {
    "entity_match",
    "event_match",
    "hidden_axis",
    "metric_match",
    "oclass",
    "rclass",
    "task_scope_sha256",
    "truncation",
}
_V7_RAW_LABEL_FIELDS = _RAW_LABEL_FIELDS | {
    "decisive_units",
    "entity_match",
    "event_anchor_matches",
    "event_match",
    "formula_id",
    "hidden_axis",
    "metric_match",
    "oclass",
    "period_relation",
    "rclass",
    "scope_relation",
    "signal_kind",
    "task_scope_sha256",
    "truncation",
}
_ADJUDICATION_FIELDS = frozenset(
    {
        "adjudicated_at_utc",
        "adjudicator_id",
        "annotation_view_sha256",
        "binary_gold",
        "item_id",
        "leakage_positive",
        "post_cutoff_present",
        "rationale",
        "raw_label_a_sha256",
        "raw_label_b_sha256",
        "resolution_status",
        "rubric_sha256",
        "rubric_version",
        "schema_version",
        "task_relevance",
        "temporal_label",
        "outcome_revealing",
    }
)
_QUERY_FIELDS = frozenset(
    {
        "query_class",
        "query_corpus_entry_id",
        "query_sha256",
        "schema_version",
        "task_id",
        "template_version",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "calibration_status",
        "claim_eligible",
        "cohen_kappa",
        "counts",
        "exclusions_by_type",
        "filter_validated",
        "generated_at_utc",
        "hub_revision",
        "inclusion_probability_note",
        "live_unrestricted_authorized",
        "pilot_n",
        "pilot_status",
        "quality_gates_passed",
        "raw_agreement",
        "schema_version",
    }
)
_REPORT_COUNT_FIELDS = frozenset(
    {
        "adjudicated",
        "determinate_binary_gold",
        "items",
        "leakage_negative",
        "leakage_positive",
        "raw_labels",
        "raw_labels_a",
        "raw_labels_b",
        "unknown",
        "unresolved_disputes",
    }
)


class CalibrationValidationError(ValueError):
    """Raised when a calibration bundle is missing or inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the bundle's canonical JSON representation (one LF terminator)."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CalibrationValidationError("value cannot be represented as canonical JSON") from exc
    return (text + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _error(path: str, message: str) -> CalibrationValidationError:
    return CalibrationValidationError(f"{path}: {message}")


def _read_utf8(path: Path, label: str) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise _error(label, "file is unavailable") from exc
    if raw.startswith(b"\xef\xbb\xbf"):
        raise _error(label, "UTF-8 BOM is forbidden")
    if b"\r" in raw:
        raise _error(label, "CRLF/CR bytes are forbidden; use LF")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _error(label, "file is not valid UTF-8") from exc


def _load_json(path: Path, relative: str) -> Any:
    text = _read_utf8(path, relative)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _error(relative, f"invalid JSON: {exc.msg}") from exc
    if text.encode("utf-8") != canonical_json_bytes(value):
        raise _error(relative, "JSON is not canonical sorted compact JSON")
    return value


def _load_jsonl(path: Path, relative: str) -> list[dict[str, Any]]:
    text = _read_utf8(path, relative)
    if text and not text.endswith("\n"):
        raise _error(relative, "non-empty JSONL files must end with LF")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise _error(relative, f"blank row at line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _error(relative, f"invalid JSON at line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise _error(relative, f"row {line_number} must be a JSON object")
        if line.encode("utf-8") + b"\n" != canonical_json_bytes(value):
            raise _error(relative, f"row {line_number} is not canonical sorted compact JSON")
        _reject_forbidden(value, f"{relative}:{line_number}")
        rows.append(value)
    return rows


def _reject_forbidden(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_KEYS:
                raise _error(label, f"forbidden raw-content/secret field {key!r}")
            _reject_forbidden(nested, label)
    elif isinstance(value, list):
        for nested in value:
            _reject_forbidden(nested, label)


def _strict_object(value: Any, fields: Iterable[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _error(label, "must be a JSON object")
    allowed = set(fields)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise _error(label, f"unknown field(s): {', '.join(unknown)}")
    return value


def _required(value: Mapping[str, Any], name: str, label: str) -> Any:
    if name not in value:
        raise _error(label, f"missing required field {name!r}")
    return value[name]


def _string(value: Any, name: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(label, f"{name} must be a non-empty string")
    return value


def _sha(value: Any, name: str, label: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise _error(label, f"{name} must be a lowercase SHA-256")
    return value


def _bool(value: Any, name: str, label: str) -> bool:
    if not isinstance(value, bool):
        raise _error(label, f"{name} must be boolean")
    return value


def _integer(value: Any, name: str, label: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _error(label, f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise _error(label, f"{name} must be >= {minimum}")
    return value


def _probability(value: Any, name: str, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise _error(label, f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise _error(label, f"{name} must be in (0, 1]")
    return parsed


def _unit_interval(value: Any, name: str, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise _error(label, f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise _error(label, f"{name} must be in [0, 1]")
    return parsed


def _nullable_unit_interval(value: Any, name: str, label: str) -> float | None:
    if value is None:
        return None
    return _unit_interval(value, name, label)


def _date(value: Any, name: str, label: str) -> date:
    if not isinstance(value, str) or not _DATE.fullmatch(value):
        raise _error(label, f"{name} must be a day-precision ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise _error(label, f"{name} is not a valid calendar date") from exc


def _utc_datetime(value: Any, name: str, label: str) -> datetime:
    if not isinstance(value, str) or not _UTC_DATETIME.fullmatch(value):
        raise _error(label, f"{name} must be RFC3339 UTC with a Z suffix")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise _error(label, f"{name} is not a valid timestamp") from exc
    if parsed.tzinfo != UTC:
        raise _error(label, f"{name} must use UTC")
    return parsed


def _safe_locator(value: Any, name: str, label: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    locator = _string(value, name, label)
    parsed = urlsplit(locator)
    if parsed.scheme not in {"https", "archive"}:
        raise _error(label, f"{name} must be an https or archive locator")
    if parsed.username or parsed.password or parsed.fragment:
        raise _error(label, f"{name} contains credentials or a fragment")
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise _error(label, f"{name} contains an invalid port") from exc
    if parsed_port == 443 and parsed.scheme == "https":
        raise _error(label, f"{name} must omit the default https port")
    for key, _ in parse_qsl(parsed.query, keep_blank_values=True):
        if any(word in key.lower() for word in ("token", "secret", "signature", "key", "auth")):
            raise _error(label, f"{name} contains a credential/signature query parameter")
    return locator


def _safe_canonical_url(value: Any, label: str) -> str:
    url = _string(value, "canonical_url", label)
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise _error(label, "canonical_url must be an https URL")
    if not url.startswith("https://"):
        raise _error(label, "canonical_url scheme must be lowercase")
    if parsed.username or parsed.password or parsed.fragment:
        raise _error(label, "canonical_url contains credentials or a fragment")
    if any(character.isspace() for character in url) or parsed.netloc != parsed.netloc.lower():
        raise _error(label, "canonical_url is not normalized")
    try:
        parsed_port = parsed.port
    except ValueError as exc:
        raise _error(label, "canonical_url contains an invalid port") from exc
    if parsed_port == 443:
        raise _error(label, "canonical_url must omit the default https port")
    try:
        expected_hostname = parsed.hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise _error(label, "canonical_url host is not valid IDNA") from exc
    if parsed.hostname != expected_hostname:
        raise _error(label, "canonical_url host must use lowercase IDNA")
    path = parsed.path
    normalized_path: list[str] = []
    index = 0
    while index < len(path):
        character = path[index]
        if character == "%":
            if index + 2 >= len(path) or not re.fullmatch(
                r"[0-9A-Fa-f]{2}", path[index + 1 : index + 3]
            ):
                raise _error(label, "canonical_url path contains an invalid percent escape")
            decoded = int(path[index + 1 : index + 3], 16)
            if decoded in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~":
                normalized_path.append(chr(decoded))
            else:
                normalized_path.append(f"%{path[index + 1 : index + 3].upper()}")
            index += 3
            continue
        if ord(character) > 127:
            raise _error(label, "canonical_url path must use percent-encoded UTF-8")
        normalized_path.append(character)
        index += 1
    if "".join(normalized_path) != path:
        raise _error(label, "canonical_url path is not canonically percent-encoded")
    _safe_locator(url, "canonical_url", label)
    return url


def _validate_schema_documents(root: Path) -> None:
    for relative in sorted(SCHEMA_FILES):
        schema = _load_json(root / relative, relative)
        if not isinstance(schema, dict):
            raise _error(relative, "schema must be an object")
        if schema.get("type") != "object":
            raise _error(relative, "schema root type must be object")
        if schema.get("additionalProperties") is not False:
            raise _error(relative, "schema root must set additionalProperties=false")
        if not isinstance(schema.get("properties"), dict):
            raise _error(relative, "schema must declare properties")
        required = schema.get("required")
        if not isinstance(required, list) or not required:
            raise _error(relative, "schema must declare required fields")


def _validate_manifest(root: Path) -> tuple[dict[str, Any], str]:
    relative = "artifact_manifest.json"
    path = root / relative
    try:
        manifest_bytes = path.read_bytes()
    except OSError as exc:
        raise _error(relative, "file is unavailable") from exc
    manifest = _load_json(path, relative)
    manifest = _strict_object(manifest, _MANIFEST_FIELDS, relative)
    if manifest.get("schema_version") != CALIBRATION_MANIFEST_SCHEMA_VERSION:
        raise _error(relative, "unsupported manifest schema_version")
    calibration_id = manifest.get("calibration_id")
    if calibration_id not in CALIBRATION_IDS:
        raise _error(relative, "calibration_id is not an approved calibration version")
    if calibration_id == RELEVANCE_CALIBRATION_ID:
        tracked_files, json_files, jsonl_files = (
            V3_TRACKED_FILES,
            V3_JSON_FILES,
            V3_JSONL_FILES,
        )
    elif calibration_id == FRESH_RELEVANCE_CALIBRATION_ID:
        tracked_files, json_files, jsonl_files = (
            V4_TRACKED_FILES,
            V4_JSON_FILES,
            V4_JSONL_FILES,
        )
    elif calibration_id == V5_RELEVANCE_CALIBRATION_ID:
        tracked_files, json_files, jsonl_files = (
            V5_TRACKED_FILES,
            V5_JSON_FILES,
            V5_JSONL_FILES,
        )
    elif calibration_id == V6_RELEVANCE_CALIBRATION_ID:
        tracked_files, json_files, jsonl_files = (
            V6_TRACKED_FILES,
            V6_JSON_FILES,
            V6_JSONL_FILES,
        )
    elif calibration_id == V7_RELEVANCE_CALIBRATION_ID:
        tracked_files, json_files, jsonl_files = (
            V7_TRACKED_FILES,
            V7_JSON_FILES,
            V7_JSONL_FILES,
        )
    else:
        tracked_files, json_files, jsonl_files = (
            EXPECTED_TRACKED_FILES,
            JSON_FILES,
            JSONL_FILES,
        )
    hub = _strict_object(
        _required(manifest, "hub", relative), {"repo_id", "revision"}, "artifact_manifest.hub"
    )
    if hub.get("repo_id") != HUB_REPO_ID or hub.get("revision") != HUB_REVISION:
        raise _error(relative, "manifest hub pin does not match the approved paper identity")
    clocks = _strict_object(
        _required(manifest, "clocks", relative),
        {
            "backbone_id",
            "backbone_training_cutoff",
            "experiment_as_of",
            "guard_days",
            "task_cutoff_semantics",
        },
        "artifact_manifest.clocks",
    )
    _string(clocks.get("backbone_id"), "backbone_id", "artifact_manifest.clocks")
    _date(
        clocks.get("backbone_training_cutoff"),
        "backbone_training_cutoff",
        "artifact_manifest.clocks",
    )
    _date(clocks.get("experiment_as_of"), "experiment_as_of", "artifact_manifest.clocks")
    _integer(clocks.get("guard_days"), "guard_days", "artifact_manifest.clocks", minimum=0)
    if clocks.get("task_cutoff_semantics") != "utc_calendar_day_inclusive_next_day_exclusive":
        raise _error(relative, "unsupported task cutoff semantics")
    for key, expected_path in (
        ("rubric", "rubric.md"),
        ("sampling_plan", "sampling-plan.json"),
        ("query_corpus", "query-corpus.jsonl"),
    ):
        reference = _strict_object(
            _required(manifest, key, relative), {"path", "sha256"}, f"artifact_manifest.{key}"
        )
        if reference.get("path") != expected_path:
            raise _error(relative, f"{key}.path must be {expected_path}")
        reference_sha256 = _sha(reference.get("sha256"), f"{key}.sha256", relative)
        if reference_sha256 != _sha256_file(root / expected_path):
            raise _error(relative, f"{key}.sha256 does not match {expected_path}")
    restricted_sha = manifest.get("restricted_evidence_manifest_sha256")
    if restricted_sha is not None:
        _sha(restricted_sha, "restricted_evidence_manifest_sha256", relative)
    adjudication = _strict_object(
        _required(manifest, "adjudication", relative),
        {
            "adjudicated_observed",
            "raw_labels_expected",
            "raw_labels_observed",
            "status",
            "unresolved_disputes",
        },
        "artifact_manifest.adjudication",
    )
    if adjudication.get("status") not in {"not_started", "pilot", "complete", "incomplete"}:
        raise _error(relative, "unsupported adjudication status")
    for name in (
        "adjudicated_observed",
        "raw_labels_expected",
        "raw_labels_observed",
        "unresolved_disputes",
    ):
        _integer(adjudication.get(name), name, "artifact_manifest.adjudication", minimum=0)
    status = manifest.get("calibration_status")
    if status not in _CALIBRATION_STATUSES:
        raise _error(relative, "unsupported calibration_status")
    if _bool(
        manifest.get("live_unrestricted_authorized"), "live_unrestricted_authorized", relative
    ):
        raise _error(relative, "live unrestricted authorization must remain false")
    files = _required(manifest, "files", relative)
    if not isinstance(files, list):
        raise _error(relative, "files must be a list")
    paths: list[str] = []
    for index, entry_value in enumerate(files):
        entry = _strict_object(
            entry_value, {"bytes", "path", "row_count", "sha256"}, f"{relative}.files[{index}]"
        )
        entry_label = f"{relative}.files[{index}]"
        entry_path = _string(_required(entry, "path", entry_label), "path", entry_label)
        if Path(entry_path).is_absolute() or ".." in Path(entry_path).parts:
            raise _error(relative, f"files[{index}].path must be relative and contained")
        paths.append(entry_path)
        _sha(_required(entry, "sha256", entry_label), "sha256", entry_label)
        _integer(_required(entry, "bytes", entry_label), "bytes", entry_label)
        row_count = _required(entry, "row_count", entry_label)
        if row_count is not None:
            _integer(row_count, "row_count", entry_label, minimum=0)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise _error(relative, "files must be unique and sorted by path")
    if tuple(paths) != tracked_files:
        raise _error(relative, "files must exactly allowlist the calibration bundle")
    for entry in files:
        entry_path = root / entry["path"]
        try:
            observed_bytes = entry_path.read_bytes()
        except OSError as exc:
            raise _error(relative, f"declared file is unavailable: {entry['path']}") from exc
        if not entry_path.is_file() or entry_path.is_symlink():
            raise _error(relative, f"declared path is not a regular file: {entry['path']}")
        if len(observed_bytes) != entry["bytes"]:
            raise _error(relative, f"byte count mismatch for {entry['path']}")
        if _sha256_bytes(observed_bytes) != entry["sha256"]:
            raise _error(relative, f"SHA-256 mismatch for {entry['path']}")
        expected_rows = None
        if entry["path"] in jsonl_files:
            expected_rows = len(_load_jsonl(entry_path, entry["path"]))
        elif entry["path"] in json_files:
            _load_json(entry_path, entry["path"])
            expected_rows = 1
        if entry["row_count"] != expected_rows:
            raise _error(relative, f"row_count mismatch for {entry['path']}")
    sidecar = root / "CALIBRATION.sha256"
    sidecar_text = _read_utf8(sidecar, "CALIBRATION.sha256")
    if not re.fullmatch(r"[0-9a-f]{64}  artifact_manifest\.json\n", sidecar_text):
        raise _error("CALIBRATION.sha256", "must contain '<sha256>  artifact_manifest.json' and LF")
    manifest_sha = _sha256_bytes(manifest_bytes)
    if not sidecar_text.startswith(manifest_sha):
        raise _error("CALIBRATION.sha256", "manifest hash does not match artifact_manifest.json")
    return manifest, manifest_sha


def _validate_cohort(cohort: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = _strict_object(cohort, _COHORT_FIELDS, "cohort.json")
    if value.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise _error("cohort.json", "unsupported schema_version")
    if value.get("cohort_id") != manifest.get("calibration_id"):
        raise _error("cohort.json", "cohort_id must match the artifact manifest")
    if value.get("hub_repo_id") != HUB_REPO_ID or value.get("hub_revision") != HUB_REVISION:
        raise _error("cohort.json", "Hub pin does not match the approved paper identity")
    _sha(value.get("temporal_index_sha256"), "temporal_index_sha256", "cohort.json")
    for name in ("experiment_as_of", "backbone_training_cutoff"):
        _date(value.get(name), name, "cohort.json")
    _string(value.get("backbone_id"), "backbone_id", "cohort.json")
    _integer(value.get("guard_days"), "guard_days", "cohort.json", minimum=0)
    if value.get("task_cutoff_semantics") != "utc_calendar_day_inclusive_next_day_exclusive":
        raise _error("cohort.json", "unsupported task cutoff semantics")
    if value.get("calibration_status") != manifest.get("calibration_status"):
        raise _error("cohort.json", "calibration_status disagrees with artifact manifest")
    if _bool(
        value.get("live_unrestricted_authorized"), "live_unrestricted_authorized", "cohort.json"
    ):
        raise _error("cohort.json", "live unrestricted authorization must remain false")
    for name, expected in (
        ("target_item_count", 400),
        ("target_raw_label_count", 800),
        ("target_adjudicated_count", 400),
    ):
        if _integer(value.get(name), name, "cohort.json", minimum=1) != expected:
            raise _error("cohort.json", f"{name} must be {expected}")
    return value


def _validate_authorization(value: Any, *, calibration_id: str | None = None) -> None:
    label = "authorization.json"
    authorization = _strict_object(value, _AUTHORIZATION_FIELDS, label)
    if authorization.get("schema_version") != "leakage-calibration-authorization/v1":
        raise _error(label, "unsupported schema_version")
    expected_calibration_id = calibration_id or authorization.get("calibration_id")
    if expected_calibration_id not in CALIBRATION_IDS:
        raise _error(label, "calibration_id is not an approved calibration version")
    if authorization.get("calibration_id") != expected_calibration_id:
        raise _error(label, "calibration_id does not match the artifact manifest")
    for name in ("authorization_id", "scope", "issue_id", "source_comment_id"):
        _string(authorization.get(name), name, label)
    expected_scope = {
        CALIBRATION_ID: "calibration_pilot_collection_and_labeling",
        FROZEN_CALIBRATION_ID: "calibration_frozen_400_collection_and_labeling",
        RELEVANCE_CALIBRATION_ID: "calibration_relevance_rubric_v2_no_collection",
        FRESH_RELEVANCE_CALIBRATION_ID: (
            "calibration_relevance_rubric_v3_fresh_pilot_collection_and_labeling"
        ),
        V5_RELEVANCE_CALIBRATION_ID: V5_RELEVANCE_SCOPE,
        V6_RELEVANCE_CALIBRATION_ID: V6_RELEVANCE_SCOPE,
        V7_RELEVANCE_CALIBRATION_ID: V7_RELEVANCE_SCOPE,
    }[expected_calibration_id]
    if authorization.get("scope") != expected_scope:
        raise _error(label, "authorization scope does not match the calibration version")
    if authorization.get("status") != "active":
        raise _error(label, "authorization status must be active")
    _utc_datetime(authorization.get("issued_at_utc"), "issued_at_utc", label)
    calibration_collection_authorized = _bool(
        authorization.get("calibration_unrestricted_collection_authorized"),
        "calibration_unrestricted_collection_authorized",
        label,
    )
    if (
        expected_calibration_id not in {RELEVANCE_CALIBRATION_ID}
        and not calibration_collection_authorized
    ):
        raise _error(label, "collection calibration requires explicit unrestricted authorization")
    if expected_calibration_id == RELEVANCE_CALIBRATION_ID:
        if calibration_collection_authorized:
            raise _error(label, "rubric calibration must not authorize a new collection")
    if _bool(
        authorization.get("full_paper_unrestricted_authorized"),
        "full_paper_unrestricted_authorized",
        label,
    ):
        raise _error(label, "full paper unrestricted authorization is forbidden in the pilot")
    if not _bool(
        authorization.get("same_candidates_cutoff_filter"),
        "same_candidates_cutoff_filter",
        label,
    ):
        raise _error(label, "pilot authorization must require filtering the same candidates")
    if _integer(authorization.get("pilot_item_count"), "pilot_item_count", label, minimum=1) != 40:
        raise _error(label, "pilot_item_count must be 40")
    strata = authorization.get("pilot_strata")
    if not isinstance(strata, dict) or set(strata) != _STRATA:
        raise _error(label, "pilot_strata must contain exactly the four approved strata")
    for stratum, count in strata.items():
        if _integer(count, stratum, label, minimum=1) != 10:
            raise _error(label, f"pilot_strata.{stratum} must be 10")
    if expected_calibration_id in {
        FROZEN_CALIBRATION_ID,
        RELEVANCE_CALIBRATION_ID,
        FRESH_RELEVANCE_CALIBRATION_ID,
        V5_RELEVANCE_CALIBRATION_ID,
        V6_RELEVANCE_CALIBRATION_ID,
        V7_RELEVANCE_CALIBRATION_ID,
    }:
        if (
            _integer(authorization.get("frozen_item_count"), "frozen_item_count", label, minimum=1)
            != 400
        ):
            raise _error(label, "frozen_item_count must be 400")
        frozen_strata = authorization.get("frozen_strata")
        expected_frozen_strata = {
            "A1-T1": {"keep": 38, "reject": 38, "ambiguous": 25},
            "B-T1": {"keep": 38, "reject": 38, "ambiguous": 25},
            "B-T2": {"keep": 37, "reject": 37, "ambiguous": 25},
            "C-T2": {"keep": 37, "reject": 37, "ambiguous": 25},
        }
        if frozen_strata != expected_frozen_strata:
            raise _error(label, "frozen_strata does not match the approved 400-item plan")


def _validate_sampling_plan(value: Any) -> None:
    value = _strict_object(value, _SAMPLING_FIELDS, "sampling-plan.json")
    if value.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise _error("sampling-plan.json", "unsupported schema_version")
    if value.get("status") not in {"scaffold", "pilot", "frozen"}:
        raise _error("sampling-plan.json", "unsupported sampling status")
    if value.get("sort_key") != 'SHA256("leakage-calibration/v1|<hub_full_sha>|<item_id>")':
        raise _error("sampling-plan.json", "sort_key does not match the frozen rule")
    _integer(value.get("domain_cap"), "domain_cap", "sampling-plan.json", minimum=1)
    _integer(value.get("task_cap"), "task_cap", "sampling-plan.json", minimum=1)
    pilot = _strict_object(
        value.get("pilot"),
        {"double_blind", "enters_frozen_set", "size", "strata"},
        "sampling-plan.pilot",
    )
    if pilot.get("double_blind") is not True or pilot.get("enters_frozen_set") is not False:
        raise _error(
            "sampling-plan.json", "pilot must be double blind and excluded from frozen set"
        )
    if _integer(pilot.get("size"), "size", "sampling-plan.pilot", minimum=1) != 40:
        raise _error("sampling-plan.json", "pilot size must be 40")
    if pilot.get("strata") != sorted(_STRATA):
        raise _error("sampling-plan.json", "pilot strata are not the four approved strata")
    frozen = _strict_object(
        value.get("frozen_set"),
        {
            "ambiguous_or_missing",
            "calibration_only",
            "keep",
            "non_scored",
            "reject",
            "reviewed_only",
            "size",
        },
        "sampling-plan.frozen_set",
    )
    if frozen != {
        "ambiguous_or_missing": 100,
        "calibration_only": True,
        "keep": 150,
        "non_scored": True,
        "reject": 150,
        "reviewed_only": True,
        "size": 400,
    }:
        raise _error(
            "sampling-plan.json", "frozen set counts/policy do not match the approved plan"
        )
    strata = value.get("strata")
    if not isinstance(strata, list):
        raise _error("sampling-plan.json", "strata must be a list")
    if [item.get("stratum") for item in strata if isinstance(item, dict)] != sorted(_STRATA):
        raise _error("sampling-plan.json", "strata must be sorted and complete")
    expected_counts = {
        "A1-T1": (38, 38, 25),
        "B-T1": (38, 38, 25),
        "B-T2": (37, 37, 25),
        "C-T2": (37, 37, 25),
    }
    for index, entry_value in enumerate(strata):
        entry = _strict_object(
            entry_value,
            {"ambiguous_or_missing", "keep", "reject", "stratum"},
            f"sampling-plan.strata[{index}]",
        )
        expected = expected_counts.get(entry.get("stratum"))
        if (
            expected is None
            or tuple(entry.get(name) for name in ("keep", "reject", "ambiguous_or_missing"))
            != expected
        ):
            raise _error(
                "sampling-plan.json", f"invalid count for stratum {entry.get('stratum')!r}"
            )


def _validate_query_rows(
    rows: list[dict[str, Any]], *, expected_classes: frozenset[str] | None = None
) -> None:
    ids: list[str] = []
    classes_by_task: dict[str, dict[str, int]] = {}
    approved_classes = expected_classes or frozenset(
        {
            "neutral_entity_topic",
            "cutoff_anchored",
            "retrospective_outcome_challenge",
        }
    )
    for index, row_value in enumerate(rows):
        label = f"query-corpus.jsonl:{index + 1}"
        row = _strict_object(row_value, _QUERY_FIELDS, label)
        if row.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise _error(label, "unsupported schema_version")
        ids.append(_string(row.get("query_corpus_entry_id"), "query_corpus_entry_id", label))
        task_id = _string(row.get("task_id"), "task_id", label)
        _string(row.get("template_version"), "template_version", label)
        _sha(row.get("query_sha256"), "query_sha256", label)
        query_class = row.get("query_class")
        if query_class not in approved_classes:
            raise _error(label, "unsupported query_class")
        task_classes = classes_by_task.setdefault(task_id, {})
        task_classes[query_class] = task_classes.get(query_class, 0) + 1
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise _error(
            "query-corpus.jsonl", "rows must be unique and sorted by query_corpus_entry_id"
        )
    if any(
        set(classes) != approved_classes or any(count != 1 for count in classes.values())
        for classes in classes_by_task.values()
    ):
        raise _error(
            "query-corpus.jsonl",
            "each task must have exactly one row for each approved query class",
        )


def _validate_item_rows(
    rows: list[dict[str, Any]], cohort: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    ids: list[str] = []
    items: dict[str, dict[str, Any]] = {}
    for index, row_value in enumerate(rows):
        label = f"items.jsonl:{index + 1}"
        row = _strict_object(row_value, _ITEM_FIELDS, label)
        if row.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise _error(label, "unsupported schema_version")
        item_id = _string(row.get("item_id"), "item_id", label)
        if not _SHA256.fullmatch(item_id):
            raise _error(label, "item_id must be a SHA-256")
        ids.append(item_id)
        if item_id in items:
            raise _error(label, "duplicate item_id")
        if row.get("hub_repo_id") != HUB_REPO_ID or row.get("hub_revision") != HUB_REVISION:
            raise _error(label, "Hub pin mismatch")
        if row.get("temporal_index_sha256") != cohort.get("temporal_index_sha256"):
            raise _error(label, "temporal_index_sha256 mismatch")
        for name in ("experiment_as_of", "backbone_id", "backbone_training_cutoff"):
            if row.get(name) != cohort.get(name):
                raise _error(label, f"{name} does not match cohort clock identity")
        _string(row.get("task_id"), "task_id", label)
        if row.get("category") not in _CATEGORY_VALUES or row.get("paper_band") not in _PAPER_BANDS:
            raise _error(label, "unsupported category or paper_band")
        origin = _date(row.get("forecast_origin"), "forecast_origin", label)
        cutoff = _date(row.get("task_cutoff_date"), "task_cutoff_date", label)
        bound = _utc_datetime(row.get("exclusive_cutoff_bound"), "exclusive_cutoff_bound", label)
        expected_bound = datetime.combine(
            cutoff + timedelta(days=1), datetime.min.time(), tzinfo=UTC
        )
        if bound != expected_bound:
            raise _error(label, "exclusive_cutoff_bound must be next-day 00:00:00Z")
        _date(row.get("backbone_training_cutoff"), "backbone_training_cutoff", label)
        _date(row.get("experiment_as_of"), "experiment_as_of", label)
        for name in (
            "cutoff_declaration_sha256",
            "content_sha256",
            "policy_sha256",
            "query_sha256",
            "receipt_sha256",
            "request_filter_sha256",
            "response_sha256",
            "service_cohort_sha256",
            "temporal_index_sha256",
            "tool_contract_sha256",
            "title_sha256",
            "snippet_sha256",
            "model_visible_payload_sha256",
            "original_url_sha256",
        ):
            _sha(row.get(name), name, label)
        _sha(row.get("previous_receipt_sha256"), "previous_receipt_sha256", label, nullable=True)
        _string(row.get("collection_id"), "collection_id", label)
        _string(row.get("call_id"), "call_id", label)
        _string(row.get("provider_id"), "provider_id", label)
        if row.get("receipt_protocol") != "search_receipt/v1":
            raise _error(label, "receipt_protocol must be search_receipt/v1")
        _integer(row.get("candidate_index"), "candidate_index", label, minimum=0)
        _integer(row.get("deterministic_rank"), "deterministic_rank", label, minimum=0)
        _probability(row.get("inclusion_probability"), "inclusion_probability", label)
        if row.get("sampling_stratum") not in _STRATA:
            raise _error(label, "unsupported sampling_stratum")
        canonical_url = _safe_canonical_url(row.get("canonical_url"), label)
        if row.get("canonical_url_version") != "url-canonicalizer/v1":
            raise _error(label, "unsupported canonical URL version")
        if row.get("canonical_url_sha256") != _sha256_bytes(canonical_url.encode("utf-8")):
            raise _error(label, "canonical_url_sha256 does not match canonical_url")
        parsed = urlsplit(canonical_url)
        domain = _string(row.get("domain"), "domain", label).lower()
        if domain != parsed.hostname.lower():
            raise _error(label, "domain does not match canonical_url host")
        for name in ("retrieved_at_utc", "snapshot_captured_at_utc"):
            _utc_datetime(row.get(name), name, label)
        for name in ("provider_published_at_utc", "provider_updated_at_utc"):
            if row.get(name) is not None:
                _utc_datetime(row.get(name), name, label)
        snapshot_status = row.get("restricted_snapshot_status")
        if snapshot_status not in _SNAPSHOT_STATES:
            raise _error(label, "unsupported restricted_snapshot_status")
        snapshot_sha = _sha(
            row.get("restricted_snapshot_sha256"),
            "restricted_snapshot_sha256",
            label,
            nullable=True,
        )
        snapshot_bytes = row.get("restricted_snapshot_bytes")
        snapshot_mime = row.get("restricted_snapshot_mime")
        snapshot_locator = _safe_locator(
            row.get("restricted_snapshot_locator"),
            "restricted_snapshot_locator",
            label,
            nullable=True,
        )
        if snapshot_status == "available":
            _sha(snapshot_sha, "restricted_snapshot_sha256", label)
            _integer(snapshot_bytes, "restricted_snapshot_bytes", label, minimum=0)
            if not isinstance(snapshot_mime, str) or not _MIME.fullmatch(snapshot_mime):
                raise _error(label, "available snapshot requires a valid MIME type")
            if snapshot_locator is None:
                raise _error(label, "available snapshot requires a safe locator")
        elif any(value is not None for value in (snapshot_sha, snapshot_bytes, snapshot_mime)):
            raise _error(label, "non-available snapshot cannot carry exact payload metadata")
        expected_id = _sha256_bytes(
            "\0".join(
                (
                    "retrieval_leakage_candidate/v1",
                    row["receipt_sha256"],
                    str(row["candidate_index"]),
                    row["canonical_url_sha256"],
                    row["model_visible_payload_sha256"],
                    row["task_id"],
                    row["task_cutoff_date"],
                )
            ).encode("utf-8")
        )
        if item_id != expected_id:
            raise _error(label, "item_id does not match the receipt-bound candidate formula")
        if not isinstance(origin, date):  # pragma: no cover - keeps type checkers honest.
            raise _error(label, "forecast_origin is invalid")
        items[item_id] = row
    if ids != sorted(ids):
        raise _error("items.jsonl", "rows must be sorted by item_id")
    return items


def _nullable_view_text(value: Any, name: str, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _error(label, f"{name} must be a string or null")
    if len(value) > _MAX_VIEW_TEXT_CHARS:
        raise _error(label, f"{name} exceeds the bounded annotation-view text limit")
    return value


def _validate_view_evidence(
    value: Any,
    name: str,
    label: str,
    *,
    expected_kind: str,
    expected_source_field: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    evidence = _strict_object(value, _VIEW_EVIDENCE_FIELDS, f"{label}.{name}")
    for field in _VIEW_EVIDENCE_FIELDS:
        _required(evidence, field, f"{label}.{name}")
    if evidence.get("kind") != expected_kind:
        raise _error(f"{label}.{name}", f"kind must be {expected_kind}")
    if evidence.get("source_field") != expected_source_field:
        raise _error(f"{label}.{name}", f"source_field must be {expected_source_field}")
    _safe_locator(evidence.get("locator"), "locator", f"{label}.{name}")
    _utc_datetime(evidence.get("observed_at_utc"), "observed_at_utc", f"{label}.{name}")
    _utc_datetime(evidence.get("timestamp_utc"), "timestamp_utc", f"{label}.{name}")
    _sha(evidence.get("sha256"), "sha256", f"{label}.{name}")
    return evidence


def _validate_view_history_evidence(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise _error(label, "authoritative_history_evidence must be a list")
    history: list[dict[str, Any]] = []
    for index, value in enumerate(value):
        evidence = _strict_object(
            value,
            _VIEW_EVIDENCE_FIELDS,
            f"{label}.authoritative_history_evidence[{index}]",
        )
        evidence_label = f"{label}.authoritative_history_evidence[{index}]"
        for field in _VIEW_EVIDENCE_FIELDS:
            _required(evidence, field, evidence_label)
        if evidence.get("kind") not in {"archive", "history", "versioned_archive"}:
            raise _error(evidence_label, "history evidence has an unsupported kind")
        _string(evidence.get("source_field"), "source_field", evidence_label)
        _safe_locator(evidence.get("locator"), "locator", evidence_label)
        _utc_datetime(evidence.get("observed_at_utc"), "observed_at_utc", evidence_label)
        _utc_datetime(evidence.get("timestamp_utc"), "timestamp_utc", evidence_label)
        _sha(evidence.get("sha256"), "sha256", evidence_label)
        history.append(evidence)
    return history


def _validate_view_rows(
    rows: list[dict[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
    *,
    require_task_prompt: bool = False,
) -> dict[str, dict[str, Any]]:
    ids: list[str] = []
    views: dict[str, dict[str, Any]] = {}
    for index, row_value in enumerate(rows):
        label = f"annotation-view.jsonl:{index + 1}"
        row = _strict_object(row_value, _VIEW_FIELDS, label)
        if row.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise _error(label, "unsupported schema_version")
        item_id = _string(row.get("item_id"), "item_id", label)
        item = items.get(item_id)
        if item is None:
            raise _error(label, "annotation view references an unknown item")
        if item_id in views:
            raise _error(label, "duplicate item_id")
        ids.append(item_id)
        model_visible_title = _nullable_view_text(
            row.get("model_visible_title"), "model_visible_title", label
        )
        model_visible_snippet = _nullable_view_text(
            row.get("model_visible_snippet"), "model_visible_snippet", label
        )
        model_visible_content = _nullable_view_text(
            row.get("model_visible_content"), "model_visible_content", label
        )
        task_prompt = row.get("task_prompt")
        if require_task_prompt:
            task_prompt = _string(task_prompt, "task_prompt", label)
        elif task_prompt is not None:
            task_prompt = _nullable_view_text(task_prompt, "task_prompt", label)
        if task_prompt is not None and len(task_prompt) > _MAX_VIEW_TEXT_CHARS:
            raise _error(label, "task_prompt exceeds the bounded annotation-view text limit")
        for value, item_field, name in (
            (model_visible_title, "title_sha256", "model_visible_title"),
            (model_visible_snippet, "snippet_sha256", "model_visible_snippet"),
            (model_visible_content, "content_sha256", "model_visible_content"),
        ):
            if _sha256_bytes((value or "").encode("utf-8")) != item.get(item_field):
                raise _error(label, f"{name} does not match item hash")
        for name in (
            "task_id",
            "category",
            "paper_band",
            "task_cutoff_date",
            "domain",
            "model_visible_payload_sha256",
            "canonical_url_sha256",
            "canonical_url_version",
        ):
            if row.get(name) != item.get(name):
                raise _error(label, f"{name} does not match item")
        canonical_url = _safe_canonical_url(row.get("canonical_url"), label)
        if canonical_url != item.get("canonical_url"):
            raise _error(label, "canonical_url does not match item")
        for name in (
            "restricted_snapshot_sha256",
            "restricted_snapshot_status",
            "restricted_snapshot_locator",
        ):
            if row.get(name) != item.get(name):
                raise _error(label, f"{name} does not match item")
        publish_time = row.get("authoritative_publish_time_utc")
        update_time = row.get("authoritative_update_time_utc")
        for name, value in (
            ("authoritative_publish_time_utc", publish_time),
            ("authoritative_update_time_utc", update_time),
        ):
            if value is not None:
                _utc_datetime(value, name, label)
        if publish_time != item.get("provider_published_at_utc"):
            raise _error(label, "authoritative_publish_time_utc does not match item")
        if update_time != item.get("provider_updated_at_utc"):
            raise _error(label, "authoritative_update_time_utc does not match item")
        publish_evidence = _validate_view_evidence(
            row.get("authoritative_publish_evidence"),
            "authoritative_publish_evidence",
            label,
            expected_kind="publisher",
            expected_source_field="volc.PublishTime",
        )
        if publish_time is None and publish_evidence is not None:
            raise _error(label, "publish evidence requires an authoritative publish time")
        if publish_time is not None:
            if publish_evidence is None or publish_evidence.get("timestamp_utc") != publish_time:
                raise _error(label, "publish evidence does not match authoritative publish time")
        history_evidence = _validate_view_history_evidence(
            row.get("authoritative_history_evidence"), label
        )
        history_status = row.get("authoritative_history_status")
        if history_status not in _VIEW_HISTORY_STATUSES:
            raise _error(label, "unsupported authoritative_history_status")
        if history_status == "not_available" and (history_evidence or update_time is not None):
            raise _error(label, "history status says unavailable but history data is present")
        if history_status == "provider_update_time_only" and update_time is None:
            raise _error(label, "provider_update_time_only requires an update time")
        view_payload = dict(row)
        declared_sha = view_payload.pop("annotation_view_sha256", None)
        _sha(declared_sha, "annotation_view_sha256", label)
        if declared_sha != _sha256_bytes(canonical_json_bytes(view_payload)):
            raise _error(label, "annotation_view_sha256 does not match the blind view")
        views[item_id] = row
    if ids != sorted(ids):
        raise _error("annotation-view.jsonl", "rows must be sorted by item_id")
    return views


def _validate_filter_rows(
    rows: list[dict[str, Any]], items: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    ids: list[str] = []
    decisions: dict[str, dict[str, Any]] = {}
    for index, row_value in enumerate(rows):
        label = f"filter-decisions.jsonl:{index + 1}"
        row = _strict_object(row_value, _FILTER_FIELDS, label)
        if row.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise _error(label, "unsupported schema_version")
        item_id = _string(row.get("item_id"), "item_id", label)
        if item_id not in items:
            raise _error(label, "filter decision references an unknown item")
        if item_id in decisions:
            raise _error(label, "duplicate item_id")
        ids.append(item_id)
        _sha(row.get("filter_policy_sha256"), "filter_policy_sha256", label)
        state = row.get("filter_state")
        verdict = row.get("filter_verdict")
        if state not in _FILTER_STATES or verdict not in _FILTER_VERDICTS:
            raise _error(label, "unsupported filter state/verdict")
        expected_verdict = {
            "keep": "accept",
            "reject": "reject",
            "ambiguous": "ambiguous",
            "missing": "missing",
        }[state]
        if verdict != expected_verdict:
            raise _error(label, "filter state and verdict disagree")
        _utc_datetime(row.get("decided_at_utc"), "decided_at_utc", label)
        decisions[item_id] = row
    if ids != sorted(ids):
        raise _error("filter-decisions.jsonl", "rows must be sorted by item_id")
    return decisions


def _validate_evidence(value: Any, label: str) -> None:
    if not isinstance(value, list):
        raise _error(label, "evidence must be a list")
    for index, evidence_value in enumerate(value):
        evidence = _strict_object(
            evidence_value,
            {"kind", "locator", "observed_at_utc", "sha256"},
            f"{label}.evidence[{index}]",
        )
        _string(evidence.get("kind"), "kind", f"{label}.evidence[{index}]")
        _safe_locator(evidence.get("locator"), "locator", f"{label}.evidence[{index}]")
        _utc_datetime(
            evidence.get("observed_at_utc"), "observed_at_utc", f"{label}.evidence[{index}]"
        )
        _sha(evidence.get("sha256"), "sha256", f"{label}.evidence[{index}]")


def _validate_raw_rows(
    rows: list[dict[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
    views: Mapping[str, Mapping[str, Any]],
    *,
    expected_reviewer_type: str,
    relative: str,
    expected_rubric_version: str = "leakage-calibration-rubric/v1",
    expected_rubric_sha256: str | None = None,
) -> tuple[dict[str, dict[str, Any]], int]:
    ids: list[str] = []
    labels: dict[str, dict[str, Any]] = {}
    for index, row_value in enumerate(rows):
        label = f"{relative}:{index + 1}"
        row = _strict_object(row_value, _RAW_LABEL_FIELDS, label)
        if row.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise _error(label, "unsupported schema_version")
        item_id = _string(row.get("item_id"), "item_id", label)
        if item_id not in items or item_id not in views:
            raise _error(label, "label references an unknown item/view")
        if item_id in labels:
            raise _error(label, "duplicate item_id")
        ids.append(item_id)
        if row.get("annotation_view_sha256") != views[item_id].get("annotation_view_sha256"):
            raise _error(label, "annotation_view_sha256 does not match blind view")
        reviewer_type = _string(row.get("reviewer_type"), "reviewer_type", label)
        if reviewer_type != expected_reviewer_type:
            raise _error(label, f"reviewer_type must be {expected_reviewer_type}")
        reviewer_id = _string(row.get("reviewer_id"), "reviewer_id", label)
        if "@" in reviewer_id or any(character.isspace() for character in reviewer_id):
            raise _error(label, "reviewer_id must be pseudonymous and contain no email/whitespace")
        if row.get("rubric_version") != expected_rubric_version:
            raise _error(label, "unsupported rubric_version")
        rubric_sha256 = _sha(row.get("rubric_sha256"), "rubric_sha256", label)
        if expected_rubric_sha256 is not None and rubric_sha256 != expected_rubric_sha256:
            raise _error(label, "rubric_sha256 does not match the versioned rubric")
        temporal = row.get("temporal_label")
        if temporal not in _TEMPORAL_LABELS:
            raise _error(label, "unsupported temporal_label")
        if (
            row.get("task_relevance") not in _TASK_RELEVANCE
            or row.get("outcome_revealing") not in _OUTCOME_REVEALING
        ):
            raise _error(label, "unsupported task_relevance or outcome_revealing")
        _validate_evidence(row.get("evidence"), label)
        if temporal in {"pre_or_on_cutoff", "post_cutoff", "mixed"} and not row["evidence"]:
            raise _error(label, "determinate labels require evidence")
        cutoff_bound = _utc_datetime(
            items[item_id]["exclusive_cutoff_bound"], "exclusive_cutoff_bound", label
        )
        published_at = items[item_id].get("provider_published_at_utc")
        if (
            temporal == "pre_or_on_cutoff"
            and published_at is not None
            and _utc_datetime(published_at, "provider_published_at_utc", label) >= cutoff_bound
        ):
            raise _error(label, "pre-cutoff label conflicts with a post-cutoff publication time")
        updated_at = items[item_id].get("provider_updated_at_utc")
        history_kinds = {"archive", "history", "versioned_archive"}
        if (
            temporal == "pre_or_on_cutoff"
            and updated_at is not None
            and _utc_datetime(updated_at, "provider_updated_at_utc", label) >= cutoff_bound
            and not any(evidence["kind"] in history_kinds for evidence in row["evidence"])
        ):
            raise _error(label, "post-cutoff update without version history is not determinate")
        _unit_interval(row.get("confidence"), "confidence", label)
        rationale = _string(row.get("rationale"), "rationale", label)
        if len(rationale) > 500:
            raise _error(label, "rationale exceeds the short-paraphrase limit")
        _utc_datetime(row.get("labeled_at_utc"), "labeled_at_utc", label)
        labels[item_id] = row
    if ids != sorted(ids):
        raise _error(relative, "rows must be sorted by item_id")
    return labels, len(rows)


def _row_hash(row: Mapping[str, Any]) -> str:
    return _sha256_bytes(canonical_json_bytes(dict(row)))


def _validate_adjudication_rows(
    rows: list[dict[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
    views: Mapping[str, Mapping[str, Any]],
    labels_a: Mapping[str, Mapping[str, Any]],
    labels_b: Mapping[str, Mapping[str, Any]],
    *,
    filter_decisions: Mapping[str, Mapping[str, Any]],
    expected_rubric_version: str = "leakage-calibration-rubric/v1",
    expected_rubric_sha256: str | None = None,
) -> tuple[dict[str, dict[str, Any]], bool, int, int, int, int]:
    ids: list[str] = []
    adjudicated: dict[str, dict[str, Any]] = {}
    filter_invalid = False
    determinate = positive = negative = unresolved = 0
    for index, row_value in enumerate(rows):
        label = f"labels/adjudicated.jsonl:{index + 1}"
        row = _strict_object(row_value, _ADJUDICATION_FIELDS, label)
        if row.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise _error(label, "unsupported schema_version")
        item_id = _string(row.get("item_id"), "item_id", label)
        if item_id not in items or item_id not in views:
            raise _error(label, "adjudication references an unknown item/view")
        if item_id in adjudicated:
            raise _error(label, "duplicate item_id")
        if item_id not in labels_a or item_id not in labels_b:
            raise _error(label, "adjudication requires both independent raw labels")
        ids.append(item_id)
        if row.get("annotation_view_sha256") != views[item_id].get("annotation_view_sha256"):
            raise _error(label, "annotation_view_sha256 does not match blind view")
        if row.get("raw_label_a_sha256") != _row_hash(labels_a[item_id]) or row.get(
            "raw_label_b_sha256"
        ) != _row_hash(labels_b[item_id]):
            raise _error(label, "raw label hash does not match the referenced label")
        if row.get("rubric_version") != expected_rubric_version:
            raise _error(label, "unsupported rubric_version")
        rubric_sha256 = _sha(row.get("rubric_sha256"), "rubric_sha256", label)
        if expected_rubric_sha256 is not None and rubric_sha256 != expected_rubric_sha256:
            raise _error(label, "rubric_sha256 does not match the versioned rubric")
        adjudicator = _string(row.get("adjudicator_id"), "adjudicator_id", label)
        if "@" in adjudicator or any(character.isspace() for character in adjudicator):
            raise _error(label, "adjudicator_id must be pseudonymous")
        temporal = row.get("temporal_label")
        relevance = row.get("task_relevance")
        outcome = row.get("outcome_revealing")
        if (
            temporal not in _TEMPORAL_LABELS
            or relevance not in _TASK_RELEVANCE
            or outcome not in _OUTCOME_REVEALING
        ):
            raise _error(label, "unsupported adjudicated label enum")
        resolution = row.get("resolution_status")
        if resolution not in {"resolved", "unresolved", "invalid"}:
            raise _error(label, "unsupported resolution_status")
        post = (
            temporal in {"post_cutoff", "mixed"}
            if temporal in _TEMPORAL_LABELS - {"indeterminate", "invalid"}
            else None
        )
        leakage = (
            (post is True and relevance == "relevant")
            if post is not None and relevance != "indeterminate"
            else None
        )
        if row.get("post_cutoff_present") != post or row.get("leakage_positive") != leakage:
            raise _error(label, "derived post/leakage fields do not match adjudicated labels")
        binary_gold = row.get("binary_gold")
        if binary_gold is not None and not isinstance(binary_gold, bool):
            raise _error(label, "binary_gold must be boolean or null")
        if resolution == "resolved" and leakage is not None:
            if binary_gold is not leakage:
                raise _error(label, "resolved binary_gold must equal leakage_positive")
            determinate += 1
            if leakage:
                positive += 1
            else:
                negative += 1
        elif binary_gold is not None:
            raise _error(label, "indeterminate/unresolved binary_gold must be null")
        if resolution != "resolved":
            unresolved += 1
        rationale = _string(row.get("rationale"), "rationale", label)
        if len(rationale) > 500:
            raise _error(label, "rationale exceeds the short-paraphrase limit")
        _utc_datetime(row.get("adjudicated_at_utc"), "adjudicated_at_utc", label)
        decision = filter_decisions.get(item_id)
        if (
            decision is not None
            and decision.get("filter_state") == "keep"
            and temporal in {"post_cutoff", "mixed"}
            and relevance == "relevant"
        ):
            filter_invalid = True
        adjudicated[item_id] = row
    if ids != sorted(ids):
        raise _error("labels/adjudicated.jsonl", "rows must be sorted by item_id")
    return adjudicated, filter_invalid, determinate, positive, negative, unresolved


def _validate_report(
    value: Any,
    *,
    manifest: Mapping[str, Any],
    counts: Mapping[str, int],
    filter_invalid: bool,
    manifest_sha: str,
) -> None:
    report = _strict_object(value, _REPORT_FIELDS, "reports/calibration-report.json")
    if report.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise _error("reports/calibration-report.json", "unsupported schema_version")
    if report.get("calibration_status") != manifest.get("calibration_status"):
        raise _error(
            "reports/calibration-report.json", "calibration_status disagrees with manifest"
        )
    if report.get("hub_revision") != HUB_REVISION:
        raise _error("reports/calibration-report.json", "hub_revision mismatch")
    if _bool(
        report.get("live_unrestricted_authorized"),
        "live_unrestricted_authorized",
        "reports/calibration-report.json",
    ):
        raise _error(
            "reports/calibration-report.json", "live unrestricted authorization must remain false"
        )
    report_counts = _strict_object(
        report.get("counts"), _REPORT_COUNT_FIELDS, "reports/calibration-report.json.counts"
    )
    normalized_report_counts: dict[str, int] = {}
    for name in _REPORT_COUNT_FIELDS:
        normalized_report_counts[name] = _integer(
            report_counts.get(name), name, "reports/calibration-report.json.counts", minimum=0
        )
    if normalized_report_counts != dict(counts):
        raise _error(
            "reports/calibration-report.json", "counts are not recomputable from the bundle"
        )
    filter_validated = _bool(
        report.get("filter_validated"), "filter_validated", "reports/calibration-report.json"
    )
    expected_filter_validated = (
        counts["items"] > 0 and counts["adjudicated"] > 0 and not filter_invalid
    )
    if filter_validated != expected_filter_validated:
        raise _error(
            "reports/calibration-report.json",
            "filter_validated is inconsistent with sealed adjudication",
        )
    if filter_invalid and report.get("calibration_status") != "filter_not_validated":
        raise _error(
            "reports/calibration-report.json",
            "post-cutoff accepted item requires filter_not_validated",
        )
    claim_eligible = _bool(
        report.get("claim_eligible"), "claim_eligible", "reports/calibration-report.json"
    )
    raw_agreement = _nullable_unit_interval(
        report.get("raw_agreement"), "raw_agreement", "reports/calibration-report.json"
    )
    cohen_kappa = report.get("cohen_kappa")
    if cohen_kappa is not None:
        if (
            not isinstance(cohen_kappa, int | float)
            or isinstance(cohen_kappa, bool)
            or not math.isfinite(float(cohen_kappa))
            or not -1 <= float(cohen_kappa) <= 1
        ):
            raise _error(
                "reports/calibration-report.json", "cohen_kappa must be null or in [-1, 1]"
            )
    quality_gates_passed = _bool(
        report.get("quality_gates_passed"),
        "quality_gates_passed",
        "reports/calibration-report.json",
    )
    complete = (
        counts["items"] == 400
        and counts["raw_labels_a"] == 400
        and counts["raw_labels_b"] == 400
        and counts["adjudicated"] == 400
        and counts["unresolved_disputes"] == 0
        and counts["determinate_binary_gold"] >= 380
        and counts["leakage_positive"] >= 100
        and counts["leakage_negative"] >= 100
    )
    if claim_eligible != (
        complete and not filter_invalid and report.get("calibration_status") == "validated"
    ):
        raise _error(
            "reports/calibration-report.json", "claim_eligible does not satisfy completion gates"
        )
    if report.get("calibration_status") == "validated":
        if (
            not quality_gates_passed
            or raw_agreement is None
            or raw_agreement < 0.9
            or cohen_kappa is None
            or float(cohen_kappa) < 0.8
        ):
            raise _error(
                "reports/calibration-report.json",
                "validated bundles require agreement >= 0.90 and Cohen kappa >= 0.80",
            )
    _utc_datetime(
        report.get("generated_at_utc"), "generated_at_utc", "reports/calibration-report.json"
    )
    note = _string(
        report.get("inclusion_probability_note"),
        "inclusion_probability_note",
        "reports/calibration-report.json",
    )
    if "prevalence" not in note.lower() or "inclusion" not in note.lower():
        raise _error(
            "reports/calibration-report.json",
            "report must state the enriched-sample prevalence limitation",
        )
    exclusions = report.get("exclusions_by_type")
    if not isinstance(exclusions, dict) or any(
        not isinstance(key, str)
        or not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        for key, value in exclusions.items()
    ):
        raise _error(
            "reports/calibration-report.json",
            "exclusions_by_type must map labels to non-negative integers",
        )
    if manifest_sha == "":  # pragma: no cover - keeps the argument explicit for callers.
        raise _error("reports/calibration-report.json", "manifest hash is unavailable")


def _validate_no_extra_files(root: Path, *, calibration_id: str) -> None:
    allowed = set(EXPECTED_TRACKED_FILES) | {"artifact_manifest.json", "CALIBRATION.sha256"}
    if calibration_id == RELEVANCE_CALIBRATION_ID:
        allowed |= set(V3_PILOT_JSONL_FILES)
    elif calibration_id == FRESH_RELEVANCE_CALIBRATION_ID:
        allowed |= set(V4_PILOT_JSONL_FILES)
    elif calibration_id == V5_RELEVANCE_CALIBRATION_ID:
        allowed |= set(V5_PILOT_JSONL_FILES)
    elif calibration_id == V6_RELEVANCE_CALIBRATION_ID:
        allowed |= set(V6_TRACKED_FILES)
    elif calibration_id == V7_RELEVANCE_CALIBRATION_ID:
        allowed |= set(V7_TRACKED_FILES)
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(root).as_posix() not in allowed:
            raise _error(
                "artifact_manifest.json", f"unallowlisted file present: {path.relative_to(root)}"
            )


def _validate_v3_pilot(
    root: Path,
    frozen_items: Mapping[str, Mapping[str, Any]],
    cohort: Mapping[str, Any],
    *,
    rubric_sha256: str,
) -> None:
    pilot_label = "pilot"
    pilot_items_rows = _load_jsonl(root / "pilot/items.jsonl", "pilot/items.jsonl")
    pilot_items = _validate_item_rows(pilot_items_rows, cohort)
    if len(pilot_items) != 40:
        raise _error(pilot_label, "v3 pilot requires exactly 40 items")
    if set(pilot_items).intersection(frozen_items):
        raise _error(pilot_label, "v3 pilot items must not be part of the 400-item cohort")
    stratum_counts = Counter(item.get("sampling_stratum") for item in pilot_items.values())
    if stratum_counts != Counter({stratum: 10 for stratum in _STRATA}):
        raise _error(pilot_label, "v3 pilot requires 10 items in each approved stratum")

    pilot_view_rows = _load_jsonl(
        root / "pilot/annotation-view.jsonl", "pilot/annotation-view.jsonl"
    )
    pilot_views = _validate_view_rows(pilot_view_rows, pilot_items, require_task_prompt=True)
    if set(pilot_views) != set(pilot_items):
        raise _error(pilot_label, "v3 pilot views must cover exactly the pilot items")
    pilot_filter_rows = _load_jsonl(
        root / "pilot/filter-decisions.jsonl", "pilot/filter-decisions.jsonl"
    )
    pilot_filters = _validate_filter_rows(pilot_filter_rows, pilot_items)
    if set(pilot_filters) != set(pilot_items):
        raise _error(pilot_label, "v3 pilot filters must cover exactly the pilot items")
    _validate_v3_pilot_source_identity(root, pilot_items_rows, pilot_filter_rows)

    pilot_label_rows = _load_jsonl(
        root / "pilot/labels/raw/annotator-a.jsonl",
        "pilot/labels/raw/annotator-a.jsonl",
    )
    pilot_labels, pilot_count = _validate_raw_rows(
        pilot_label_rows,
        pilot_items,
        pilot_views,
        expected_reviewer_type="labeler_a",
        relative="pilot/labels/raw/annotator-a.jsonl",
        expected_rubric_version="leakage-calibration-rubric/v2",
        expected_rubric_sha256=rubric_sha256,
    )
    if pilot_count != 40 or set(pilot_labels) != set(pilot_items):
        raise _error(pilot_label, "v3 pilot requires one Labeler A row per pilot item")
    if any(
        row.get("task_relevance") == "indeterminate"
        and row.get("outcome_revealing") != "indeterminate"
        for row in pilot_labels.values()
    ):
        raise _error(pilot_label, "indeterminate relevance must carry indeterminate outcome")


def _validate_v3_pilot_source_identity(
    root: Path,
    pilot_item_rows: list[dict[str, Any]],
    pilot_filter_rows: list[dict[str, Any]],
) -> None:
    source_root = root.parent / "v1"
    if not source_root.is_dir():
        raise _error("pilot", "v1 pilot root is required to validate the no-refetch lineage")
    source_items = _load_jsonl(source_root / "items.jsonl", "v1/items.jsonl")
    if [row.get("item_id") for row in pilot_item_rows] != [
        row.get("item_id") for row in source_items
    ]:
        raise _error("pilot", "v3 pilot item IDs do not match the existing v1 pilot")
    for current, source in zip(pilot_item_rows, source_items, strict=True):
        for field in ("receipt_sha256", "model_visible_payload_sha256"):
            if current.get(field) != source.get(field):
                raise _error("pilot", f"v3 pilot {field} does not match the v1 pilot")
    source_filters = _load_jsonl(
        source_root / "filter-decisions.jsonl", "v1/filter-decisions.jsonl"
    )
    for current, source in zip(pilot_filter_rows, source_filters, strict=True):
        for field in ("item_id", "filter_policy_sha256", "filter_state", "filter_verdict"):
            if current.get(field) != source.get(field):
                raise _error("pilot", f"v3 pilot filter {field} does not match the v1 pilot")


def _validate_v3_source_identity(
    root: Path,
    item_rows: list[dict[str, Any]],
    view_rows: list[dict[str, Any]],
) -> None:
    source_root = root.parent / "v2"
    if not source_root.is_dir():
        raise _error("v3", "v2 audit root is required to validate v3 lineage")
    source_items = _load_jsonl(source_root / "items.jsonl", "v2/items.jsonl")
    source_views = _load_jsonl(source_root / "annotation-view.jsonl", "v2/annotation-view.jsonl")
    if [row.get("item_id") for row in item_rows] != [row.get("item_id") for row in source_items]:
        raise _error("v3", "v3 item IDs do not match the v2 audit cohort")
    for current, source in zip(item_rows, source_items, strict=True):
        for field in ("receipt_sha256", "model_visible_payload_sha256"):
            if current.get(field) != source.get(field):
                raise _error("v3", f"v3 {field} does not match the v2 audit cohort")
    if [row.get("annotation_view_sha256") for row in view_rows] != [
        row.get("annotation_view_sha256") for row in source_views
    ]:
        raise _error("v3", "v3 annotation-view hashes do not match the v2 audit cohort")


def _validate_v4_source_identity(
    root: Path,
    item_rows: list[dict[str, Any]],
    view_rows: list[dict[str, Any]],
) -> None:
    source_root = root.parent / "v3"
    if not source_root.is_dir():
        raise _error("v4", "v3 failed-gate audit is required for v4 lineage")
    source_items = _load_jsonl(source_root / "items.jsonl", "v3/items.jsonl")
    source_views = _load_jsonl(source_root / "annotation-view.jsonl", "v3/annotation-view.jsonl")
    if [row.get("item_id") for row in item_rows] != [row.get("item_id") for row in source_items]:
        raise _error("v4", "v4 item IDs do not match the v3 audit cohort")
    for current, source in zip(item_rows, source_items, strict=True):
        for field in ("receipt_sha256", "model_visible_payload_sha256"):
            if current.get(field) != source.get(field):
                raise _error("v4", f"v4 {field} does not match the v3 audit cohort")
    if [row.get("annotation_view_sha256") for row in view_rows] != [
        row.get("annotation_view_sha256") for row in source_views
    ]:
        raise _error("v4", "v4 annotation-view hashes do not match the v3 audit cohort")


def _validate_v4_pilot(
    root: Path,
    frozen_items: Mapping[str, Mapping[str, Any]],
    cohort: Mapping[str, Any],
    *,
    rubric_sha256: str,
) -> None:
    """Validate the fresh, authorized v4 pilot without turning it into gold."""

    pilot_items_rows = _load_jsonl(root / "pilot/items.jsonl", "pilot/items.jsonl")
    pilot_filter_rows = _load_jsonl(
        root / "pilot/filter-decisions.jsonl", "pilot/filter-decisions.jsonl"
    )
    pilot_view_rows = _load_jsonl(
        root / "pilot/annotation-view.jsonl", "pilot/annotation-view.jsonl"
    )
    pilot_query_rows = _load_jsonl(root / "pilot/query-corpus.jsonl", "pilot/query-corpus.jsonl")
    pilot_label_rows = _load_jsonl(
        root / "pilot/labels/raw/annotator-a.jsonl",
        "pilot/labels/raw/annotator-a.jsonl",
    )
    required_pilot_files = (
        pilot_items_rows,
        pilot_filter_rows,
        pilot_view_rows,
        pilot_query_rows,
    )
    if not any(required_pilot_files) and not pilot_label_rows:
        return
    if any(not rows for rows in required_pilot_files):
        raise _error("pilot", "v4 pilot files must be populated together")
    pilot_items = _validate_item_rows(pilot_items_rows, cohort)
    if len(pilot_items) != 40:
        raise _error("pilot", "v4 pilot requires exactly 40 items")
    if set(pilot_items).intersection(frozen_items):
        raise _error("pilot", "v4 pilot items must not be part of the 400-item cohort")
    source_root = root.parent / "v3"
    if not source_root.is_dir():
        raise _error("pilot", "v3 failed-gate audit is required for v4 provenance")
    source_pilot_rows = _load_jsonl(source_root / "pilot/items.jsonl", "v3/pilot/items.jsonl")
    source_pilot_ids = {row.get("item_id") for row in source_pilot_rows}
    if len(source_pilot_ids) != len(source_pilot_rows):
        raise _error("pilot", "v3 audit pilot IDs are not unique")
    if set(pilot_items).intersection(source_pilot_ids):
        raise _error("pilot", "v4 pilot items must be disjoint from the v3 pilot")
    for item in pilot_items.values():
        if item.get("collection_id") != FRESH_RELEVANCE_COLLECTION_ID:
            raise _error("pilot", "v4 pilot item collection_id is not the fresh collection")
        if not str(item.get("call_id", "")).startswith("v4pilot40-"):
            raise _error("pilot", "v4 pilot call_id is not from the fresh collection")
    stratum_counts = Counter(item.get("sampling_stratum") for item in pilot_items.values())
    if stratum_counts != Counter({stratum: 10 for stratum in _STRATA}):
        raise _error("pilot", "v4 pilot requires 10 items in each approved stratum")

    pilot_views = _validate_view_rows(
        pilot_view_rows,
        pilot_items,
        require_task_prompt=True,
    )
    if set(pilot_views) != set(pilot_items):
        raise _error("pilot", "v4 pilot views must cover exactly the pilot items")
    pilot_filters = _validate_filter_rows(pilot_filter_rows, pilot_items)
    if set(pilot_filters) != set(pilot_items):
        raise _error("pilot", "v4 pilot filters must cover exactly the pilot items")
    _validate_query_rows(pilot_query_rows)
    if pilot_label_rows:
        pilot_labels, pilot_count = _validate_raw_rows(
            pilot_label_rows,
            pilot_items,
            pilot_views,
            expected_reviewer_type="labeler_a",
            relative="pilot/labels/raw/annotator-a.jsonl",
            expected_rubric_version=FRESH_RELEVANCE_RUBRIC_VERSION,
            expected_rubric_sha256=rubric_sha256,
        )
        if pilot_count != 40 or set(pilot_labels) != set(pilot_items):
            raise _error("pilot", "v4 pilot requires one Labeler A row per item")
        if any(
            row.get("task_relevance") == "indeterminate"
            and row.get("outcome_revealing") != "indeterminate"
            for row in pilot_labels.values()
        ):
            raise _error("pilot", "indeterminate relevance must carry indeterminate outcome")


def _validate_v5_source_identity(
    root: Path,
    item_rows: list[dict[str, Any]],
    view_rows: list[dict[str, Any]],
) -> None:
    """Require the rubric-only v5 bundle to retain the v4 audit inputs exactly."""

    source_root = root.parent / "v4"
    if not source_root.is_dir():
        raise _error("v5", "v4 audit root is required for v5 provenance")
    try:
        source_state = validate_calibration_root(source_root)
    except CalibrationValidationError as exc:
        raise _error("v5", "v4 audit root failed validation") from exc
    if source_state.get("calibration_id") != FRESH_RELEVANCE_CALIBRATION_ID:
        raise _error("v5", "v5 provenance source is not the v4 audit bundle")
    source_items = _load_jsonl(source_root / "items.jsonl", "v4/items.jsonl")
    source_views = _load_jsonl(source_root / "annotation-view.jsonl", "v4/annotation-view.jsonl")
    if [row.get("item_id") for row in item_rows] != [row.get("item_id") for row in source_items]:
        raise _error("v5", "v5 item IDs do not match the v4 audit cohort")
    for current, source in zip(item_rows, source_items, strict=True):
        for field in ("receipt_sha256", "model_visible_payload_sha256"):
            if current.get(field) != source.get(field):
                raise _error("v5", f"v5 {field} does not match the v4 audit cohort")
    if [row.get("annotation_view_sha256") for row in view_rows] != [
        row.get("annotation_view_sha256") for row in source_views
    ]:
        raise _error("v5", "v5 annotation-view hashes do not match the v4 audit cohort")
    for relative in (
        "annotation-view.jsonl",
        "filter-decisions.jsonl",
        "items.jsonl",
        "labels/adjudicated.jsonl",
        "labels/raw/annotator-a.jsonl",
        "labels/raw/annotator-b.jsonl",
        "query-corpus.jsonl",
        "sampling-plan.json",
        "schemas/adjudication.schema.json",
        "schemas/cohort.schema.json",
        "schemas/item.schema.json",
    ):
        if (root / relative).read_bytes() != (source_root / relative).read_bytes():
            raise _error("v5", f"v5 audit copy differs from v4 for {relative}")


def _validate_v5_pilot(root: Path) -> None:
    """Validate the separately authorized v5 40-item relevance pilot."""

    pilot_rows = {
        relative: _load_jsonl(root / relative, relative)
        for relative in sorted(V5_PILOT_JSONL_FILES)
    }
    if not any(pilot_rows.values()):
        report = _load_json(
            root / "reports/calibration-report.json", "reports/calibration-report.json"
        )
        if report.get("pilot_status") != "not_run" or report.get("pilot_n") != 0:
            raise _error("pilot", "empty v5 pilot must record pilot_status=not_run and pilot_n=0")
        return
    required = (
        "pilot/items.jsonl",
        "pilot/annotation-view.jsonl",
        "pilot/filter-decisions.jsonl",
        "pilot/query-corpus.jsonl",
    )
    if any(not pilot_rows[relative] for relative in required):
        raise _error("pilot", "v5 pilot files must be populated together")
    cohort = _load_json(root / "cohort.json", "cohort.json")
    frozen_items = _load_jsonl(root / "items.jsonl", "items.jsonl")
    pilot_items_rows = pilot_rows["pilot/items.jsonl"]
    pilot_items = _validate_item_rows(pilot_items_rows, cohort)
    if len(pilot_items) != 40:
        raise _error("pilot", "v5 pilot requires exactly 40 items")
    frozen_ids = {row["item_id"] for row in frozen_items}
    if set(pilot_items).intersection(frozen_ids):
        raise _error("pilot", "v5 pilot items must not be part of the 400-item cohort")
    prior_ids: set[str] = set()
    for version in ("v1", "v2", "v3", "v4"):
        for relative in ("items.jsonl", "pilot/items.jsonl"):
            path = root.parent / version / relative
            if path.exists():
                prior_ids.update(
                    row["item_id"] for row in _load_jsonl(path, f"{version}/{relative}")
                )
    if set(pilot_items).intersection(prior_ids):
        raise _error("pilot", "v5 pilot items must be disjoint from v1-v4 calibration items")
    for item in pilot_items.values():
        if item.get("collection_id") != V5_RELEVANCE_COLLECTION_ID:
            raise _error("pilot", "v5 pilot item collection_id is not the authorized collection")
        if not str(item.get("call_id", "")).startswith("v5pilot40-"):
            raise _error("pilot", "v5 pilot call_id is not from the authorized collection")
    if Counter(item.get("sampling_stratum") for item in pilot_items.values()) != Counter(
        {stratum: 10 for stratum in _STRATA}
    ):
        raise _error("pilot", "v5 pilot requires 10 items in each approved stratum")
    pilot_views = _validate_view_rows(
        pilot_rows["pilot/annotation-view.jsonl"], pilot_items, require_task_prompt=True
    )
    if set(pilot_views) != set(pilot_items):
        raise _error("pilot", "v5 pilot views must cover exactly the pilot items")
    pilot_filters = _validate_filter_rows(pilot_rows["pilot/filter-decisions.jsonl"], pilot_items)
    if set(pilot_filters) != set(pilot_items):
        raise _error("pilot", "v5 pilot filters must cover exactly the pilot items")
    _validate_query_rows(pilot_rows["pilot/query-corpus.jsonl"])
    pilot_label_rows = pilot_rows["pilot/labels/raw/annotator-a.jsonl"]
    if pilot_label_rows:
        rubric_sha256 = _sha256_file(root / "rubric.md")
        pilot_labels, pilot_count = _validate_raw_rows(
            pilot_label_rows,
            pilot_items,
            pilot_views,
            expected_reviewer_type="labeler_a",
            relative="pilot/labels/raw/annotator-a.jsonl",
            expected_rubric_version=V5_RELEVANCE_RUBRIC_VERSION,
            expected_rubric_sha256=rubric_sha256,
        )
        if pilot_count != 40 or set(pilot_labels) != set(pilot_items):
            raise _error("pilot", "v5 pilot requires one Labeler A row per item")
        if any(
            row.get("task_relevance") == "indeterminate"
            and row.get("outcome_revealing") != "indeterminate"
            for row in pilot_labels.values()
        ):
            raise _error("pilot", "indeterminate relevance must carry indeterminate outcome")
    report = _load_json(root / "reports/calibration-report.json", "reports/calibration-report.json")
    if report.get("pilot_status") != "pilot" or report.get("pilot_n") not in {0, 40}:
        raise _error(
            "pilot", "populated v5 pilot must record pilot_status=pilot and pilot_n 0 or 40"
        )


def _validate_v6_scope(value: Any, label: str) -> dict[str, Any]:
    scope = _strict_object(value, _V6_SCOPE_FIELDS, label)
    for field in (
        "entity_name",
        "entity_ticker",
        "family",
        "normalizer_version",
        "requested_output",
    ):
        _string(scope.get(field), field, label)
    if scope.get("family") not in {"A1", "B", "C"}:
        raise _error(label, "unsupported task scope family")
    if scope.get("normalizer_version") != "task-scope/v1":
        raise _error(label, "unsupported task scope normalizer")
    for field in (
        "a1_target_type",
        "b_event_class",
        "b_event_date",
        "b_event_fingerprint",
        "b_reporting_package",
        "c_input_period",
        "c_target_metric",
        "c_target_period",
    ):
        value_for_field = scope.get(field)
        if value_for_field is not None:
            _string(value_for_field, field, label)
    for field in ("b_event_date", "c_input_period", "c_target_period"):
        if scope.get(field) is not None:
            _date(scope[field], field, label)
    if scope.get("b_event_fingerprint") is not None:
        _sha(scope.get("b_event_fingerprint"), "b_event_fingerprint", label)
    if scope.get("family") == "A1" and scope.get("a1_target_type") is None:
        raise _error(label, "A1 scope requires a1_target_type")
    if scope.get("family") == "B" and not (
        scope.get("b_reporting_package") or scope.get("b_event_fingerprint")
    ):
        raise _error(label, "B scope requires a reporting package or event fingerprint")
    if scope.get("family") == "C" and not all(
        scope.get(field) for field in ("c_input_period", "c_target_metric", "c_target_period")
    ):
        raise _error(label, "C scope requires metric and both periods")
    spans = scope.get("source_spans")
    if not isinstance(spans, list) or not spans:
        raise _error(label, "source_spans must be a non-empty list")
    for index, span_value in enumerate(spans):
        span = _strict_object(span_value, _V6_SPAN_FIELDS, f"{label}.source_spans[{index}]")
        _string(span.get("field"), "field", f"{label}.source_spans[{index}]")
        text = _string(span.get("span"), "span", f"{label}.source_spans[{index}]")
        if len(text) > 160:
            raise _error(label, "source span exceeds the bounded length")
    scope_sha = _sha(scope.get("task_scope_sha256"), "task_scope_sha256", label)
    unsigned = dict(scope)
    unsigned.pop("task_scope_sha256", None)
    if scope_sha != sha256_json(unsigned):
        raise _error(label, "task_scope_sha256 does not match canonical scope fields")
    return scope


def _validate_v6_truncation(value: Any, label: str) -> dict[str, Any]:
    metadata = _strict_object(value, _V6_TRUNCATION_FIELDS, label)
    for field in (
        "content_ends_at_limit",
        "no_decisive_visible_span",
        "proven_visible_truncation",
        "provider_truncated",
        "token_boundary",
    ):
        _bool(metadata.get(field), field, label)
    content_bytes = _integer(metadata.get("content_bytes"), "content_bytes", label, minimum=0)
    content_limit = _integer(
        metadata.get("content_limit_bytes"), "content_limit_bytes", label, minimum=1
    )
    if content_bytes > 10_000_000 or content_limit > 10_000_000:
        raise _error(label, "truncation metadata exceeds the bounded range")
    hidden_axis = metadata.get("hidden_axis")
    if hidden_axis not in _V6_HIDDEN_AXES:
        raise _error(label, "unsupported truncation hidden_axis")
    proven = bool(metadata["proven_visible_truncation"])
    objective_proof = bool(metadata["provider_truncated"] or metadata["token_boundary"])
    boundary_proof = bool(metadata["content_ends_at_limit"] and content_bytes == content_limit)
    if proven != (objective_proof or boundary_proof):
        raise _error(label, "proven_visible_truncation lacks an objective truncation proof")
    if not proven and hidden_axis != "none":
        raise _error(label, "hidden_axis must be none when truncation is not proven")
    if proven and hidden_axis == "none":
        raise _error(label, "proven truncation requires a hidden axis")
    return metadata


def _validate_v6_view_rows(
    rows: list[dict[str, Any]], items: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    ids: list[str] = []
    views: dict[str, dict[str, Any]] = {}
    for index, row_value in enumerate(rows):
        label = f"pilot/annotation-view.jsonl:{index + 1}"
        row = _strict_object(row_value, _V6_VIEW_FIELDS, label)
        for field in _V6_VIEW_FIELDS:
            _required(row, field, label)
        if row.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise _error(label, "unsupported schema_version")
        item_id = _string(row.get("item_id"), "item_id", label)
        item = items.get(item_id)
        if item is None:
            raise _error(label, "annotation view references an unknown item")
        if item_id in views:
            raise _error(label, "duplicate item_id")
        ids.append(item_id)
        if row.get("task_id") != item.get("task_id"):
            raise _error(label, "task_id does not match item")
        if row.get("category") != item.get("category") or row.get("paper_band") != item.get(
            "paper_band"
        ):
            raise _error(label, "task identity does not match item")
        for value, item_field, field in (
            (row.get("model_visible_title"), "title_sha256", "model_visible_title"),
            (row.get("model_visible_snippet"), "snippet_sha256", "model_visible_snippet"),
            (row.get("model_visible_content"), "content_sha256", "model_visible_content"),
        ):
            text = _nullable_view_text(value, field, label)
            if _sha256_bytes((text or "").encode("utf-8")) != item.get(item_field):
                raise _error(label, f"{field} does not match item hash")
        scope = _validate_v6_scope(row.get("task_scope"), f"{label}.task_scope")
        if item.get("task_scope_sha256") != scope.get("task_scope_sha256"):
            raise _error(label, "task scope hash does not match item")
        _sha(row.get("model_visible_payload_sha256"), "model_visible_payload_sha256", label)
        if row.get("model_visible_payload_sha256") != item.get("model_visible_payload_sha256"):
            raise _error(label, "model_visible_payload_sha256 does not match item")
        _validate_v6_truncation(row.get("truncation_metadata"), f"{label}.truncation_metadata")
        declared = _sha(row.get("annotation_view_sha256"), "annotation_view_sha256", label)
        payload = dict(row)
        payload.pop("annotation_view_sha256", None)
        if declared != sha256_json(payload):
            raise _error(label, "annotation_view_sha256 does not match the blind view")
        views[item_id] = row
    if ids != sorted(ids):
        raise _error("pilot/annotation-view.jsonl", "rows must be sorted by item_id")
    return views


def _validate_v6_raw_rows(
    rows: list[dict[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
    views: Mapping[str, Mapping[str, Any]],
    *,
    rubric_sha256: str,
) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    ids: list[str] = []
    rationale_re = re.compile(
        r"^RCLASS=([^;]+); task_scope_sha256=([0-9a-f]{64}); "
        r"truncation=(none|proven); hidden_axis=(none|entity|event|period|metric); "
        r"entity_match=(match|mismatch); "
        r"event_match=(exact|contains_exact|cross_period|not_proven|na); "
        r"metric_match=(exact|component|other|none|na); "
        r"OCLASS=(DIRECT_TARGET|DIRECT_POST_EVENT_MOVE|INDIRECT_INPUT|NO_SIGNAL|TRUNCATED_SCOPE); "
        r"span=([^;\r\n]{1,120})$"
    )
    for index, row_value in enumerate(rows):
        label = f"pilot/labels/raw/annotator-a.jsonl:{index + 1}"
        row = _strict_object(row_value, _V6_RAW_LABEL_FIELDS, label)
        for field in _V6_RAW_LABEL_FIELDS:
            _required(row, field, label)
        if row.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise _error(label, "unsupported schema_version")
        item_id = _string(row.get("item_id"), "item_id", label)
        if item_id not in items or item_id not in views or item_id in labels:
            raise _error(label, "label references an unknown or duplicate item/view")
        ids.append(item_id)
        view = views[item_id]
        view_scope = _validate_v6_scope(view["task_scope"], f"{label}.task_scope")
        scope_sha = _sha(row.get("task_scope_sha256"), "task_scope_sha256", label)
        if scope_sha != view_scope["task_scope_sha256"]:
            raise _error(label, "label task_scope_sha256 does not match blind view")
        if row.get("annotation_view_sha256") != view.get("annotation_view_sha256"):
            raise _error(label, "annotation_view_sha256 does not match blind view")
        if row.get("reviewer_type") != "labeler_a":
            raise _error(label, "reviewer_type must be labeler_a")
        reviewer_id = _string(row.get("reviewer_id"), "reviewer_id", label)
        if "@" in reviewer_id or any(character.isspace() for character in reviewer_id):
            raise _error(label, "reviewer_id must be pseudonymous")
        if row.get("rubric_version") != V6_RELEVANCE_RUBRIC_VERSION:
            raise _error(label, "unsupported rubric_version")
        if row.get("rubric_sha256") != rubric_sha256:
            raise _error(label, "rubric_sha256 does not match rubric.md")
        if row.get("temporal_label") != "indeterminate":
            raise _error(label, "v6 blind pilot temporal_label must remain indeterminate")
        if row.get("evidence") != []:
            raise _error(label, "v6 blind pilot must not include hidden clock evidence")
        relevance = row.get("task_relevance")
        outcome = row.get("outcome_revealing")
        if relevance not in _TASK_RELEVANCE or outcome not in _OUTCOME_REVEALING:
            raise _error(label, "unsupported relevance/outcome label")
        if (relevance, outcome) not in {
            ("not_relevant", "no"),
            ("indeterminate", "indeterminate"),
            ("relevant", "direct"),
            ("relevant", "indirect"),
            ("relevant", "no"),
        }:
            raise _error(label, "invalid relevance/outcome label pair")
        rclass = row.get("rclass")
        truncation = row.get("truncation")
        hidden_axis = row.get("hidden_axis")
        if rclass not in _V6_RCLASSES or truncation not in {"none", "proven"}:
            raise _error(label, "unsupported structured rationale field")
        if hidden_axis not in _V6_HIDDEN_AXES:
            raise _error(label, "unsupported rationale hidden_axis")
        if row.get("entity_match") not in {"match", "mismatch"}:
            raise _error(label, "unsupported rationale entity_match")
        if row.get("event_match") not in _V6_EVENT_MATCHES:
            raise _error(label, "unsupported rationale event_match")
        if row.get("metric_match") not in _V6_METRIC_MATCHES:
            raise _error(label, "unsupported rationale metric_match")
        if row.get("oclass") not in _V6_OCLASSES:
            raise _error(label, "unsupported rationale OCLASS")
        if truncation == "none" and hidden_axis != "none":
            raise _error(label, "non-truncated rationale cannot hide an axis")
        if truncation == "proven" and hidden_axis == "none":
            raise _error(label, "truncated rationale requires a hidden axis")
        if rclass.startswith("TRUNCATED_SCOPE_"):
            if (relevance, outcome) != ("indeterminate", "indeterminate") or row.get(
                "oclass"
            ) != "TRUNCATED_SCOPE":
                raise _error(label, "truncated rationale must be indeterminate")
        rationale = _string(row.get("rationale"), "rationale", label)
        match = rationale_re.fullmatch(rationale)
        if match is None:
            raise _error(label, "rationale does not match the v6 grammar")
        fields = match.groups()
        if fields != (
            rclass,
            scope_sha,
            truncation,
            hidden_axis,
            row.get("entity_match"),
            row.get("event_match"),
            row.get("metric_match"),
            row.get("oclass"),
            fields[-1],
        ):
            if fields[:8] != (
                rclass,
                scope_sha,
                truncation,
                hidden_axis,
                row.get("entity_match"),
                row.get("event_match"),
                row.get("metric_match"),
                row.get("oclass"),
            ):
                raise _error(label, "structured rationale fields disagree with columns")
        if len(rationale) > 500:
            raise _error(label, "rationale exceeds the short-paraphrase limit")
        _unit_interval(row.get("confidence"), "confidence", label)
        _utc_datetime(row.get("labeled_at_utc"), "labeled_at_utc", label)
        labels[item_id] = row
    if ids != sorted(ids):
        raise _error("pilot/labels/raw/annotator-a.jsonl", "rows must be sorted by item_id")
    return labels


def _v6_source_exclusion_digests(root: Path) -> dict[str, list[str]]:
    values: dict[str, set[str]] = {field: set() for field in _V6_DIGEST_FIELDS}
    for version in ("v1", "v2", "v3", "v4", "v5"):
        source = root.parent / version
        if not source.is_dir():
            continue
        query_meta: dict[str, tuple[str, str]] = {}
        for relative in ("query-corpus.jsonl", "pilot/query-corpus.jsonl"):
            path = source / relative
            if not path.exists():
                continue
            for query in _load_jsonl(path, f"{version}/{relative}"):
                query_sha = query.get("query_sha256")
                task_id = query.get("task_id")
                query_class = query.get("query_class")
                if (
                    isinstance(query_sha, str)
                    and isinstance(task_id, str)
                    and isinstance(query_class, str)
                ):
                    query_meta[query_sha] = (task_id, query_class)
        for relative in ("items.jsonl", "pilot/items.jsonl"):
            path = source / relative
            if not path.exists():
                continue
            for item in _load_jsonl(path, f"{version}/{relative}"):
                for field in (
                    "item_id",
                    "receipt_sha256",
                    "query_sha256",
                    "canonical_url_sha256",
                    "model_visible_payload_sha256",
                ):
                    if isinstance(item.get(field), str):
                        values[field].add(item[field])
                canonical = item.get("canonical_url_sha256")
                payload = item.get("model_visible_payload_sha256")
                if isinstance(canonical, str) and isinstance(payload, str):
                    values["url_payload_sha256"].add(sha256_text(f"{canonical}|{payload}"))
                query_sha = item.get("query_sha256")
                if isinstance(query_sha, str) and query_sha in query_meta:
                    task_id, query_class = query_meta[query_sha]
                    values["semantic_key_sha256"].add(sha256_text(f"{task_id}\0{query_class}"))
        for relative in ("annotation-view.jsonl", "pilot/annotation-view.jsonl"):
            path = source / relative
            if path.exists():
                for view in _load_jsonl(path, f"{version}/{relative}"):
                    if isinstance(view.get("annotation_view_sha256"), str):
                        values["annotation_view_sha256"].add(view["annotation_view_sha256"])
    return {field: sorted(digests) for field, digests in values.items()}


def _v6_selected_digests(
    items: list[dict[str, Any]],
    views: list[dict[str, Any]],
    queries: list[dict[str, Any]],
) -> dict[str, list[str]]:
    values: dict[str, set[str]] = {field: set() for field in _V6_DIGEST_FIELDS}
    task_ids = {str(item["task_id"]) for item in items}
    query_meta = {
        str(query["query_sha256"]): (str(query["task_id"]), str(query["query_class"]))
        for query in queries
    }
    for item in items:
        for field in (
            "item_id",
            "receipt_sha256",
            "query_sha256",
            "canonical_url_sha256",
            "model_visible_payload_sha256",
        ):
            values[field].add(str(item[field]))
        values["url_payload_sha256"].add(
            sha256_text(
                str(item["canonical_url_sha256"]) + "|" + str(item["model_visible_payload_sha256"])
            )
        )
        query_sha = str(item["query_sha256"])
        if query_sha in query_meta:
            task_id, query_class = query_meta[query_sha]
            if task_id in task_ids:
                values["semantic_key_sha256"].add(sha256_text(f"{task_id}\0{query_class}"))
    for view in views:
        values["annotation_view_sha256"].add(str(view["annotation_view_sha256"]))
    return {field: sorted(digests) for field, digests in values.items()}


def _validate_v6_exclusion_proof(
    root: Path,
    pilot_items: list[dict[str, Any]],
    pilot_views: list[dict[str, Any]],
    pilot_queries: list[dict[str, Any]],
) -> None:
    proof = _load_json(root / "pilot/exclusion-proof.json", "pilot/exclusion-proof.json")
    proof = _strict_object(proof, _V6_EXCLUSION_FIELDS, "pilot/exclusion-proof.json")
    if proof.get("schema_version") != "leakage-calibration-exclusion-proof/v1":
        raise _error("pilot/exclusion-proof.json", "unsupported exclusion proof schema")
    if proof.get("calibration_id") != V6_RELEVANCE_CALIBRATION_ID:
        raise _error("pilot/exclusion-proof.json", "exclusion proof calibration mismatch")
    if proof.get("source_calibrations") != [
        "leakage-probe/v1",
        "leakage-probe/v2",
        "leakage-probe/v3",
        "leakage-probe/v4",
        "leakage-probe/v5",
    ]:
        raise _error("pilot/exclusion-proof.json", "exclusion proof source list is incomplete")
    status = proof.get("status")
    if status not in {"preflight", "complete"}:
        raise _error("pilot/exclusion-proof.json", "unsupported exclusion proof status")
    parsed: dict[str, dict[str, list[str]]] = {}
    for name in ("excluded_digests", "selected_digests"):
        value = _strict_object(proof.get(name), _V6_DIGEST_FIELDS, f"pilot/exclusion-proof.{name}")
        parsed[name] = {}
        for field in _V6_DIGEST_FIELDS:
            digests = value.get(field)
            if not isinstance(digests, list) or any(
                not isinstance(digest, str) or not _SHA256.fullmatch(digest) for digest in digests
            ):
                raise _error(
                    "pilot/exclusion-proof.json",
                    f"{name}.{field} must be sorted SHA-256 digests",
                )
            if digests != sorted(set(digests)):
                raise _error(
                    "pilot/exclusion-proof.json", f"{name}.{field} must be sorted and unique"
                )
            parsed[name][field] = digests
    expected_excluded = _v6_source_exclusion_digests(root)
    if parsed["excluded_digests"] != expected_excluded:
        raise _error("pilot/exclusion-proof.json", "excluded evidence union is not recomputable")
    expected_selected = _v6_selected_digests(pilot_items, pilot_views, pilot_queries)
    if parsed["selected_digests"] != expected_selected:
        raise _error("pilot/exclusion-proof.json", "selected evidence digests are not recomputable")
    if status == "preflight" and any(expected_selected.values()):
        raise _error("pilot/exclusion-proof.json", "preflight proof cannot claim selected evidence")
    if status == "complete" and len(pilot_items) != 40:
        raise _error("pilot/exclusion-proof.json", "complete proof requires the 40-item pilot")
    for field in _V6_DIGEST_FIELDS:
        if set(parsed["selected_digests"][field]).intersection(parsed["excluded_digests"][field]):
            raise _error("pilot/exclusion-proof.json", f"selected evidence overlaps {field}")


def _validate_v6_source_identity(
    root: Path,
    item_rows: list[dict[str, Any]],
    view_rows: list[dict[str, Any]],
) -> None:
    source_root = root.parent / "v5"
    if not source_root.is_dir():
        raise _error("v6", "v5 failed-gate audit is required for v6 provenance")
    try:
        source_state = validate_calibration_root(source_root)
    except CalibrationValidationError as exc:
        raise _error("v6", "v5 failed-gate audit failed validation") from exc
    if source_state.get("calibration_id") != V5_RELEVANCE_CALIBRATION_ID:
        raise _error("v6", "v6 provenance source is not v5")
    source_items = _load_jsonl(source_root / "items.jsonl", "v5/items.jsonl")
    source_views = _load_jsonl(source_root / "annotation-view.jsonl", "v5/annotation-view.jsonl")
    if [row.get("item_id") for row in item_rows] != [row.get("item_id") for row in source_items]:
        raise _error("v6", "v6 top-level item IDs do not match the v5 audit")
    if [row.get("annotation_view_sha256") for row in view_rows] != [
        row.get("annotation_view_sha256") for row in source_views
    ]:
        raise _error("v6", "v6 top-level annotation-view hashes do not match v5")
    for current, source in zip(item_rows, source_items, strict=True):
        for field in ("receipt_sha256", "model_visible_payload_sha256"):
            if current.get(field) != source.get(field):
                raise _error("v6", f"v6 top-level {field} differs from v5")
    for relative in (
        "annotation-view.jsonl",
        "filter-decisions.jsonl",
        "items.jsonl",
        "labels/adjudicated.jsonl",
        "labels/raw/annotator-a.jsonl",
        "labels/raw/annotator-b.jsonl",
        "query-corpus.jsonl",
        "sampling-plan.json",
        "schemas/adjudication.schema.json",
        "schemas/cohort.schema.json",
    ):
        if (root / relative).read_bytes() != (source_root / relative).read_bytes():
            raise _error("v6", f"v6 audit copy differs from v5 for {relative}")


def _validate_v6_pilot(root: Path) -> None:
    pilot_item_rows = _load_jsonl(root / "pilot/items.jsonl", "pilot/items.jsonl")
    pilot_view_rows = _load_jsonl(
        root / "pilot/annotation-view.jsonl", "pilot/annotation-view.jsonl"
    )
    pilot_filter_rows = _load_jsonl(
        root / "pilot/filter-decisions.jsonl", "pilot/filter-decisions.jsonl"
    )
    pilot_query_rows = _load_jsonl(root / "pilot/query-corpus.jsonl", "pilot/query-corpus.jsonl")
    pilot_label_rows = _load_jsonl(
        root / "pilot/labels/raw/annotator-a.jsonl",
        "pilot/labels/raw/annotator-a.jsonl",
    )
    _validate_v6_exclusion_proof(root, pilot_item_rows, pilot_view_rows, pilot_query_rows)
    if not any(
        (pilot_item_rows, pilot_view_rows, pilot_filter_rows, pilot_query_rows, pilot_label_rows)
    ):
        report = _load_json(
            root / "reports/calibration-report.json", "reports/calibration-report.json"
        )
        if report.get("pilot_status") != "not_run" or report.get("pilot_n") != 0:
            raise _error("pilot", "empty v6 pilot must record pilot_status=not_run and pilot_n=0")
        return
    if any(
        not rows for rows in (pilot_item_rows, pilot_view_rows, pilot_filter_rows, pilot_query_rows)
    ):
        raise _error("pilot", "v6 pilot collection files must be populated together")
    cohort = _load_json(root / "cohort.json", "cohort.json")
    frozen_items = _load_jsonl(root / "items.jsonl", "items.jsonl")
    pilot_items = _validate_item_rows(pilot_item_rows, cohort)
    if len(pilot_items) != 40:
        raise _error("pilot", "v6 pilot requires exactly 40 items")
    if set(pilot_items).intersection(row["item_id"] for row in frozen_items):
        raise _error("pilot", "v6 pilot overlaps the 400-item cohort")
    payload_digests = [str(item["model_visible_payload_sha256"]) for item in pilot_items.values()]
    if len(payload_digests) != len(set(payload_digests)):
        raise _error("pilot", "v6 pilot requires one unique model-visible payload per item")
    for item in pilot_items.values():
        _sha(item.get("task_scope_sha256"), "task_scope_sha256", "pilot/items.jsonl")
        if item.get("collection_id") != V6_RELEVANCE_COLLECTION_ID:
            raise _error("pilot", "v6 pilot item collection_id is not authorized")
        if not str(item.get("call_id", "")).startswith("v6pilot40-"):
            raise _error("pilot", "v6 pilot call_id is not from the authorized collection")
    if Counter(item.get("sampling_stratum") for item in pilot_items.values()) != Counter(
        {stratum: 10 for stratum in _STRATA}
    ):
        raise _error("pilot", "v6 pilot requires 10 items in each approved stratum")
    pilot_views = _validate_v6_view_rows(pilot_view_rows, pilot_items)
    if set(pilot_views) != set(pilot_items):
        raise _error("pilot", "v6 pilot views must cover exactly the pilot items")
    pilot_filters = _validate_filter_rows(pilot_filter_rows, pilot_items)
    if set(pilot_filters) != set(pilot_items):
        raise _error("pilot", "v6 pilot filters must cover exactly the pilot items")
    _validate_query_rows(pilot_query_rows)
    if pilot_label_rows:
        labels = _validate_v6_raw_rows(
            pilot_label_rows,
            pilot_items,
            pilot_views,
            rubric_sha256=_sha256_file(root / "rubric.md"),
        )
        if len(labels) != 40 or set(labels) != set(pilot_items):
            raise _error("pilot", "v6 pilot requires one Labeler A row per item")
    report = _load_json(root / "reports/calibration-report.json", "reports/calibration-report.json")
    if report.get("pilot_status") != "pilot" or report.get("pilot_n") not in {0, 40}:
        raise _error(
            "pilot", "populated v6 pilot must record pilot_status=pilot and pilot_n 0 or 40"
        )


def _validate_v7_scope(value: Any, label: str) -> dict[str, Any]:
    """Validate the readable task-scope/v2 contract used only by v7."""

    from benchmark_harness.calibration.v7_scope import TaskScopeError, validate_task_scope

    scope = _strict_object(value, _V7_SCOPE_FIELDS, label)
    try:
        validate_task_scope(scope)
    except TaskScopeError as exc:
        raise _error(label, str(exc)) from exc
    anchors = scope.get("b_match_anchors")
    if scope.get("family") == "B":
        if not isinstance(anchors, Mapping):
            raise _error(label, "B scope must carry readable match anchors")
        _strict_object(anchors, _V7_ANCHOR_FIELDS, f"{label}.b_match_anchors")
        _date(anchors.get("release_date"), "release_date", f"{label}.b_match_anchors")
        facts = anchors.get("fingerprint_facts")
        if not isinstance(facts, list) or not facts:
            raise _error(label, "B scope must carry at least one fingerprint fact")
        if not all(isinstance(fact, str) and fact.strip() for fact in facts):
            raise _error(label, "B fingerprint facts must be readable strings")
        if (
            sum(
                bool(anchors.get(name))
                for name in ("release_date", "reporting_period", "fingerprint_facts")
            )
            < 2
        ):
            raise _error(label, "B scope must expose at least two anchor types")
        if anchors.get("reporting_period") is not None:
            _string(anchors.get("reporting_period"), "reporting_period", f"{label}.b_match_anchors")
    return scope


def _validate_v7_truncation(value: Any, label: str) -> dict[str, Any]:
    return _validate_v6_truncation(value, label)


def _validate_v7_view_rows(
    rows: list[dict[str, Any]], items: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    from benchmark_harness.calibration.relevance_v7 import segment_visible_view

    ids: list[str] = []
    views: dict[str, dict[str, Any]] = {}
    for index, row_value in enumerate(rows):
        label = f"pilot/annotation-view.jsonl:{index + 1}"
        row = _strict_object(row_value, _V7_VIEW_FIELDS, label)
        for field in _V7_VIEW_FIELDS:
            _required(row, field, label)
        if row.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise _error(label, "unsupported schema_version")
        item_id = _string(row.get("item_id"), "item_id", label)
        item = items.get(item_id)
        if item is None or item_id in views:
            raise _error(label, "annotation view references an unknown or duplicate item")
        ids.append(item_id)
        for field in ("task_id", "category", "paper_band"):
            if row.get(field) != item.get(field):
                raise _error(label, f"{field} does not match item")
        for value, item_field, field in (
            (row.get("model_visible_title"), "title_sha256", "model_visible_title"),
            (row.get("model_visible_snippet"), "snippet_sha256", "model_visible_snippet"),
            (row.get("model_visible_content"), "content_sha256", "model_visible_content"),
        ):
            text_value = _nullable_view_text(value, field, label)
            if _sha256_bytes((text_value or "").encode("utf-8")) != item.get(item_field):
                raise _error(label, f"{field} does not match item hash")
        _sha(row.get("model_visible_payload_sha256"), "model_visible_payload_sha256", label)
        if row.get("model_visible_payload_sha256") != item.get("model_visible_payload_sha256"):
            raise _error(label, "model_visible_payload_sha256 does not match item")
        scope = _validate_v7_scope(row.get("task_scope"), f"{label}.task_scope")
        if item.get("task_scope_sha256") != scope.get("task_scope_sha256"):
            raise _error(label, "task scope hash does not match item")
        _validate_v7_truncation(row.get("truncation_metadata"), f"{label}.truncation_metadata")
        units = row.get("evidence_units")
        if not isinstance(units, list) or not units:
            raise _error(label, "evidence_units must be a non-empty list")
        expected_units = segment_visible_view(row)
        expected_public = [
            {
                "end_char": unit["end_char"],
                "exact_text_sha256": unit["exact_text_sha256"],
                "field": unit["field"],
                "start_char": unit["start_char"],
                "text": unit["text"],
            }
            for unit in expected_units
        ]
        if units != expected_public:
            raise _error(label, "evidence_units do not match deterministic view segmentation")
        for unit_index, unit_value in enumerate(units):
            unit_label = f"{label}.evidence_units[{unit_index}]"
            unit = _strict_object(unit_value, _V7_UNIT_FIELDS | {"text"}, unit_label)
            for field in _V7_UNIT_FIELDS | {"text"}:
                _required(unit, field, unit_label)
            field = unit.get("field")
            start = unit.get("start_char")
            end = unit.get("end_char")
            if field not in {
                "model_visible_title",
                "model_visible_snippet",
                "model_visible_content",
            }:
                raise _error(unit_label, "evidence unit field is not model-visible")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
            ):
                raise _error(unit_label, "evidence unit offsets must be integers")
            visible = row.get(field)
            if not isinstance(visible, str) or start < 0 or end <= start or end > len(visible):
                raise _error(unit_label, "evidence unit offsets are outside the view")
            if unit.get("text") != visible[start:end]:
                raise _error(unit_label, "evidence unit text does not match view slice")
            _sha(unit.get("exact_text_sha256"), "exact_text_sha256", unit_label)
            if unit.get("exact_text_sha256") != _sha256_bytes(unit["text"].encode("utf-8")):
                raise _error(unit_label, "evidence unit hash does not match view slice")
        declared = _sha(row.get("annotation_view_sha256"), "annotation_view_sha256", label)
        payload = dict(row)
        payload.pop("annotation_view_sha256", None)
        if declared != sha256_json(payload):
            raise _error(label, "annotation_view_sha256 does not match the blind view")
        views[item_id] = row
    if ids != sorted(ids):
        raise _error("pilot/annotation-view.jsonl", "rows must be sorted by item_id")
    return views


def _validate_v7_raw_rows(
    rows: list[dict[str, Any]],
    items: Mapping[str, Mapping[str, Any]],
    views: Mapping[str, Mapping[str, Any]],
    *,
    rubric_sha256: str,
) -> dict[str, dict[str, Any]]:
    from benchmark_harness.calibration.relevance_v7 import (
        PERIOD_RELATIONS,
        SCOPE_RELATIONS,
        SIGNAL_KINDS,
        derive_label_pair,
        validate_decisive_units,
    )

    labels: dict[str, dict[str, Any]] = {}
    ids: list[str] = []
    for index, row_value in enumerate(rows):
        label = f"pilot/labels/raw/annotator-a.jsonl:{index + 1}"
        row = _strict_object(row_value, _V7_RAW_LABEL_FIELDS, label)
        for field in _V7_RAW_LABEL_FIELDS:
            _required(row, field, label)
        if row.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise _error(label, "unsupported schema_version")
        item_id = _string(row.get("item_id"), "item_id", label)
        if item_id not in items or item_id not in views or item_id in labels:
            raise _error(label, "label references an unknown or duplicate item/view")
        ids.append(item_id)
        view = views[item_id]
        if row.get("annotation_view_sha256") != view.get("annotation_view_sha256"):
            raise _error(label, "annotation_view_sha256 does not match blind view")
        if row.get("reviewer_type") != "labeler_a":
            raise _error(label, "reviewer_type must be labeler_a")
        reviewer_id = _string(row.get("reviewer_id"), "reviewer_id", label)
        if "@" in reviewer_id or any(character.isspace() for character in reviewer_id):
            raise _error(label, "reviewer_id must be pseudonymous")
        if row.get("rubric_version") != V7_RELEVANCE_RUBRIC_VERSION:
            raise _error(label, "unsupported rubric_version")
        if row.get("rubric_sha256") != rubric_sha256:
            raise _error(label, "rubric_sha256 does not match rubric.md")
        if row.get("temporal_label") != "indeterminate" or row.get("evidence") != []:
            raise _error(label, "v7 blind pilot must not include temporal or clock evidence")
        relevance = row.get("task_relevance")
        outcome = row.get("outcome_revealing")
        if relevance not in _TASK_RELEVANCE or outcome not in _OUTCOME_REVEALING:
            raise _error(label, "unsupported relevance/outcome label")
        scope_relation = row.get("scope_relation")
        period_relation = row.get("period_relation")
        signal_kind = row.get("signal_kind")
        if (
            scope_relation not in SCOPE_RELATIONS
            or period_relation not in PERIOD_RELATIONS
            or signal_kind not in SIGNAL_KINDS
        ):
            raise _error(label, "unsupported v7 structured decision field")
        try:
            expected_pair = derive_label_pair(scope_relation, period_relation, signal_kind)
        except ValueError as exc:
            raise _error(label, str(exc)) from exc
        if (relevance, outcome) != expected_pair:
            raise _error(label, "stored label pair disagrees with derive_label_pair")
        anchors = row.get("event_anchor_matches")
        if not isinstance(anchors, list) or any(not isinstance(anchor, str) for anchor in anchors):
            raise _error(label, "event_anchor_matches must be a string list")
        if anchors != sorted(set(anchors)) or any(
            anchor not in {"release_date", "reporting_period", "fingerprint_fact"}
            for anchor in anchors
        ):
            raise _error(label, "event_anchor_matches are not canonical")
        formula_id = row.get("formula_id")
        if formula_id is not None and formula_id != "net_margin":
            raise _error(label, "unsupported formula_id")
        units = row.get("decisive_units")
        if (relevance, outcome) == ("indeterminate", "indeterminate"):
            if units != []:
                try:
                    validate_decisive_units(view, units)
                except ValueError as exc:
                    raise _error(label, str(exc)) from exc
        else:
            try:
                validate_decisive_units(view, units)
            except ValueError as exc:
                raise _error(label, str(exc)) from exc
        for field, allowed in (
            ("event_match", _V7_EVENT_MATCHES),
            ("metric_match", _V7_METRIC_MATCHES),
            ("hidden_axis", _V7_HIDDEN_AXES),
            ("oclass", _V7_OCLASSES),
        ):
            if row.get(field) not in allowed:
                raise _error(label, f"unsupported {field}")
        if row.get("truncation") not in {"none", "proven"}:
            raise _error(label, "unsupported truncation")
        if row.get("truncation") == "none" and row.get("hidden_axis") != "none":
            raise _error(label, "non-truncated label cannot hide an axis")
        if row.get("truncation") == "proven" and row.get("hidden_axis") == "none":
            raise _error(label, "truncated label requires a hidden axis")
        _string(row.get("rclass"), "rclass", label)
        if row.get("rclass") not in _V7_RCLASSES:
            raise _error(label, "unsupported rclass")
        scope = _validate_v7_scope(view.get("task_scope"), f"{label}.task_scope")
        scope_sha = _sha(row.get("task_scope_sha256"), "task_scope_sha256", label)
        if scope_sha != scope.get("task_scope_sha256"):
            raise _error(label, "task_scope_sha256 does not match blind view")
        rationale = _string(row.get("rationale"), "rationale", label)
        if len(rationale) > 500 or any(
            key in rationale.lower() for key in ("prompt=", "answer=", "url=")
        ):
            raise _error(label, "rationale is too long or contains hidden fields")
        required_fragments = (
            f"scope_relation={scope_relation}",
            f"period_relation={period_relation}",
            f"signal_kind={signal_kind}",
            f"formula_id={formula_id or 'none'}",
        )
        if any(fragment not in rationale for fragment in required_fragments):
            raise _error(label, "rationale does not mirror structured decision fields")
        _unit_interval(row.get("confidence"), "confidence", label)
        _utc_datetime(row.get("labeled_at_utc"), "labeled_at_utc", label)
        labels[item_id] = row
    if ids != sorted(ids):
        raise _error("pilot/labels/raw/annotator-a.jsonl", "rows must be sorted by item_id")
    return labels


def _v7_source_exclusion_digests(root: Path) -> dict[str, list[str]]:
    values: dict[str, set[str]] = {field: set() for field in _V7_DIGEST_FIELDS}
    for version in ("v1", "v2", "v3", "v4", "v5", "v6"):
        source = root.parent / version
        if not source.is_dir():
            continue
        query_meta: dict[str, tuple[str, str]] = {}
        for relative in ("query-corpus.jsonl", "pilot/query-corpus.jsonl"):
            path = source / relative
            if not path.exists():
                continue
            for query in _load_jsonl(path, f"{version}/{relative}"):
                query_sha = query.get("query_sha256")
                task_id = query.get("task_id")
                query_class = query.get("query_class")
                if all(isinstance(value, str) for value in (query_sha, task_id, query_class)):
                    query_meta[str(query_sha)] = (str(task_id), str(query_class))
        for relative in ("items.jsonl", "pilot/items.jsonl"):
            path = source / relative
            if not path.exists():
                continue
            for item in _load_jsonl(path, f"{version}/{relative}"):
                for field in (
                    "item_id",
                    "receipt_sha256",
                    "query_sha256",
                    "canonical_url_sha256",
                    "model_visible_payload_sha256",
                ):
                    if isinstance(item.get(field), str):
                        values[field].add(item[field])
                canonical = item.get("canonical_url_sha256")
                payload = item.get("model_visible_payload_sha256")
                if isinstance(canonical, str) and isinstance(payload, str):
                    values["url_payload_sha256"].add(sha256_text(f"{canonical}|{payload}"))
                query_sha = item.get("query_sha256")
                if isinstance(query_sha, str) and query_sha in query_meta:
                    task_id, query_class = query_meta[query_sha]
                    values["semantic_key_sha256"].add(sha256_text(f"{task_id}\0{query_class}"))
        for relative in ("annotation-view.jsonl", "pilot/annotation-view.jsonl"):
            path = source / relative
            if path.exists():
                for view in _load_jsonl(path, f"{version}/{relative}"):
                    if isinstance(view.get("annotation_view_sha256"), str):
                        values["annotation_view_sha256"].add(view["annotation_view_sha256"])
    return {field: sorted(digests) for field, digests in values.items()}


def _v7_selected_digests(
    items: list[dict[str, Any]], views: list[dict[str, Any]], queries: list[dict[str, Any]]
) -> dict[str, list[str]]:
    values: dict[str, set[str]] = {field: set() for field in _V7_DIGEST_FIELDS}
    task_ids = {str(item["task_id"]) for item in items}
    query_meta = {
        str(query["query_sha256"]): (str(query["task_id"]), str(query["query_class"]))
        for query in queries
    }
    for item in items:
        for field in (
            "item_id",
            "receipt_sha256",
            "query_sha256",
            "canonical_url_sha256",
            "model_visible_payload_sha256",
        ):
            values[field].add(str(item[field]))
        values["url_payload_sha256"].add(
            sha256_text(
                str(item["canonical_url_sha256"]) + "|" + str(item["model_visible_payload_sha256"])
            )
        )
        query_sha = str(item["query_sha256"])
        if query_sha in query_meta:
            task_id, query_class = query_meta[query_sha]
            if task_id in task_ids:
                values["semantic_key_sha256"].add(sha256_text(f"{task_id}\0{query_class}"))
    for view in views:
        values["annotation_view_sha256"].add(str(view["annotation_view_sha256"]))
    return {field: sorted(digests) for field, digests in values.items()}


def _validate_v7_exclusion_proof(
    root: Path,
    pilot_items: list[dict[str, Any]],
    pilot_views: list[dict[str, Any]],
    pilot_queries: list[dict[str, Any]],
) -> None:
    proof = _strict_object(
        _load_json(root / "pilot/exclusion-proof.json", "pilot/exclusion-proof.json"),
        _V7_EXCLUSION_FIELDS,
        "pilot/exclusion-proof.json",
    )
    if proof.get("schema_version") != "leakage-calibration-exclusion-proof/v1":
        raise _error("pilot/exclusion-proof.json", "unsupported exclusion proof schema")
    if proof.get("calibration_id") != V7_RELEVANCE_CALIBRATION_ID:
        raise _error("pilot/exclusion-proof.json", "exclusion proof calibration mismatch")
    if proof.get("source_calibrations") != [
        "leakage-probe/v1",
        "leakage-probe/v2",
        "leakage-probe/v3",
        "leakage-probe/v4",
        "leakage-probe/v5",
        "leakage-probe/v6",
    ]:
        raise _error("pilot/exclusion-proof.json", "exclusion proof source list is incomplete")
    status = proof.get("status")
    if status not in {"preflight", "complete"}:
        raise _error("pilot/exclusion-proof.json", "unsupported exclusion proof status")
    parsed: dict[str, dict[str, list[str]]] = {}
    for name in ("excluded_digests", "selected_digests"):
        value = _strict_object(proof.get(name), _V7_DIGEST_FIELDS, f"pilot/exclusion-proof.{name}")
        parsed[name] = {}
        for field in _V7_DIGEST_FIELDS:
            digests = value.get(field)
            if not isinstance(digests, list) or any(
                not isinstance(digest, str) or not _SHA256.fullmatch(digest) for digest in digests
            ):
                raise _error(
                    "pilot/exclusion-proof.json", f"{name}.{field} must be SHA-256 digests"
                )
            if digests != sorted(set(digests)):
                raise _error(
                    "pilot/exclusion-proof.json", f"{name}.{field} must be sorted and unique"
                )
            parsed[name][field] = digests
    if parsed["excluded_digests"] != _v7_source_exclusion_digests(root):
        raise _error("pilot/exclusion-proof.json", "excluded evidence union is not recomputable")
    if parsed["selected_digests"] != _v7_selected_digests(pilot_items, pilot_views, pilot_queries):
        raise _error("pilot/exclusion-proof.json", "selected evidence digests are not recomputable")
    if status == "preflight" and any(parsed["selected_digests"].values()):
        raise _error("pilot/exclusion-proof.json", "preflight proof cannot claim selected evidence")
    if status == "complete" and len(pilot_items) != 40:
        raise _error("pilot/exclusion-proof.json", "complete proof requires 40 items")
    for field in _V7_DIGEST_FIELDS:
        if set(parsed["selected_digests"][field]).intersection(parsed["excluded_digests"][field]):
            raise _error("pilot/exclusion-proof.json", f"selected evidence overlaps {field}")


def _validate_v7_source_identity(
    root: Path, item_rows: list[dict[str, Any]], view_rows: list[dict[str, Any]]
) -> None:
    source_root = root.parent / "v6"
    if not source_root.is_dir():
        raise _error("v7", "v6 failed-gate audit is required for v7 provenance")
    try:
        source_state = validate_calibration_root(source_root)
    except CalibrationValidationError as exc:
        raise _error("v7", "v6 audit root failed validation") from exc
    if source_state.get("calibration_id") != V6_RELEVANCE_CALIBRATION_ID:
        raise _error("v7", "v7 provenance source is not v6")
    source_items = _load_jsonl(source_root / "items.jsonl", "v6/items.jsonl")
    source_views = _load_jsonl(source_root / "annotation-view.jsonl", "v6/annotation-view.jsonl")
    if [row.get("item_id") for row in item_rows] != [row.get("item_id") for row in source_items]:
        raise _error("v7", "v7 top-level item IDs do not match v6")
    if [row.get("annotation_view_sha256") for row in view_rows] != [
        row.get("annotation_view_sha256") for row in source_views
    ]:
        raise _error("v7", "v7 top-level annotation-view hashes do not match v6")
    for current, source in zip(item_rows, source_items, strict=True):
        for field in ("receipt_sha256", "model_visible_payload_sha256"):
            if current.get(field) != source.get(field):
                raise _error("v7", f"v7 top-level {field} differs from v6")
    for relative in (
        "annotation-view.jsonl",
        "filter-decisions.jsonl",
        "items.jsonl",
        "labels/adjudicated.jsonl",
        "labels/raw/annotator-a.jsonl",
        "labels/raw/annotator-b.jsonl",
        "query-corpus.jsonl",
        "sampling-plan.json",
        "schemas/adjudication.schema.json",
        "schemas/cohort.schema.json",
    ):
        if (root / relative).read_bytes() != (source_root / relative).read_bytes():
            raise _error("v7", f"v7 audit copy differs from v6 for {relative}")


def _validate_v7_conformance(root: Path) -> None:
    from benchmark_harness.calibration.relevance_v7 import label_relevance_outcome

    value = _load_json(
        root / "schemas/conformance-fixtures.json", "schemas/conformance-fixtures.json"
    )
    if not isinstance(value, dict) or set(value) != {"fixtures", "schema_version"}:
        raise _error("schemas/conformance-fixtures.json", "fixture document fields are not exact")
    if value.get("schema_version") != "leakage-calibration-conformance/v1":
        raise _error("schemas/conformance-fixtures.json", "unsupported fixture schema")
    fixtures = value.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise _error("schemas/conformance-fixtures.json", "fixtures must be non-empty")
    ids: list[str] = []
    for index, fixture_value in enumerate(fixtures):
        label = f"schemas/conformance-fixtures.json.fixtures[{index}]"
        fixture = _strict_object(
            fixture_value,
            {"fixture_id", "view", "path_a", "path_b"},
            label,
        )
        fixture_id = _string(fixture.get("fixture_id"), "fixture_id", label)
        ids.append(fixture_id)
        path_a = _strict_object(
            fixture.get("path_a"), {"task_relevance", "outcome_revealing"}, f"{label}.path_a"
        )
        path_b = _strict_object(
            fixture.get("path_b"), {"task_relevance", "outcome_revealing"}, f"{label}.path_b"
        )
        if path_a != path_b:
            raise _error(label, "synthetic labeling paths disagree")
        view = fixture.get("view")
        if not isinstance(view, Mapping):
            raise _error(label, "fixture view must be an object")
        try:
            pair = label_relevance_outcome(view)[:2]
        except (TypeError, ValueError) as exc:
            raise _error(label, f"fixture classifier failed: {exc}") from exc
        if pair != (path_a.get("task_relevance"), path_a.get("outcome_revealing")):
            raise _error(label, "fixture expected pair disagrees with shared classifier")
    if ids != sorted(set(ids)):
        raise _error("schemas/conformance-fixtures.json", "fixture IDs must be sorted and unique")


def _validate_v7_pilot(root: Path) -> None:
    pilot_item_rows = _load_jsonl(root / "pilot/items.jsonl", "pilot/items.jsonl")
    pilot_view_rows = _load_jsonl(
        root / "pilot/annotation-view.jsonl", "pilot/annotation-view.jsonl"
    )
    pilot_filter_rows = _load_jsonl(
        root / "pilot/filter-decisions.jsonl", "pilot/filter-decisions.jsonl"
    )
    pilot_query_rows = _load_jsonl(root / "pilot/query-corpus.jsonl", "pilot/query-corpus.jsonl")
    pilot_label_rows = _load_jsonl(
        root / "pilot/labels/raw/annotator-a.jsonl", "pilot/labels/raw/annotator-a.jsonl"
    )
    _validate_v7_exclusion_proof(root, pilot_item_rows, pilot_view_rows, pilot_query_rows)
    if not any(
        (pilot_item_rows, pilot_view_rows, pilot_filter_rows, pilot_query_rows, pilot_label_rows)
    ):
        report = _load_json(
            root / "reports/calibration-report.json", "reports/calibration-report.json"
        )
        if report.get("pilot_status") != "not_run" or report.get("pilot_n") != 0:
            raise _error("pilot", "empty v7 pilot must record pilot_status=not_run and pilot_n=0")
        return
    if any(
        not rows for rows in (pilot_item_rows, pilot_view_rows, pilot_filter_rows, pilot_query_rows)
    ):
        raise _error("pilot", "v7 collection files must be populated together")
    cohort = _load_json(root / "cohort.json", "cohort.json")
    frozen_items = _load_jsonl(root / "items.jsonl", "items.jsonl")
    pilot_items = _validate_item_rows(pilot_item_rows, cohort)
    if len(pilot_items) != 40:
        raise _error("pilot", "v7 pilot requires exactly 40 items")
    if set(pilot_items).intersection(row["item_id"] for row in frozen_items):
        raise _error("pilot", "v7 pilot overlaps the 400-item cohort")
    prior_ids: set[str] = set()
    for version in ("v1", "v2", "v3", "v4", "v5", "v6"):
        for relative in ("items.jsonl", "pilot/items.jsonl"):
            path = root.parent / version / relative
            if path.exists():
                prior_ids.update(
                    row["item_id"] for row in _load_jsonl(path, f"{version}/{relative}")
                )
    if set(pilot_items).intersection(prior_ids):
        raise _error("pilot", "v7 pilot items must be disjoint from v1-v6 evidence")
    for item in pilot_items.values():
        if item.get("collection_id") != V7_RELEVANCE_COLLECTION_ID:
            raise _error("pilot", "v7 pilot item collection_id is not authorized")
        if not str(item.get("call_id", "")).startswith("v7pilot40-"):
            raise _error("pilot", "v7 pilot call_id is not from the authorized collection")
    if Counter(item.get("sampling_stratum") for item in pilot_items.values()) != Counter(
        {stratum: 10 for stratum in _STRATA}
    ):
        raise _error("pilot", "v7 pilot requires 10 items in each approved stratum")
    payloads = [str(item["model_visible_payload_sha256"]) for item in pilot_items.values()]
    urls = [str(item["canonical_url_sha256"]) for item in pilot_items.values()]
    url_payloads = [
        sha256_text(f"{item['canonical_url_sha256']}|{item['model_visible_payload_sha256']}")
        for item in pilot_items.values()
    ]
    if (
        len(payloads) != len(set(payloads))
        or len(urls) != len(set(urls))
        or len(url_payloads) != len(set(url_payloads))
    ):
        raise _error(
            "pilot", "v7 pilot requires unique payload, canonical URL, and URL+payload evidence"
        )
    for item in pilot_items.values():
        _sha(item.get("task_scope_sha256"), "task_scope_sha256", "pilot/items.jsonl")
    pilot_views = _validate_v7_view_rows(pilot_view_rows, pilot_items)
    if set(pilot_views) != set(pilot_items):
        raise _error("pilot", "v7 pilot views must cover exactly the pilot items")
    if len({str(view["annotation_view_sha256"]) for view in pilot_views.values()}) != 40:
        raise _error("pilot", "v7 pilot annotation views must be unique")
    for view in pilot_views.values():
        scope = view["task_scope"]
        if scope.get("family") == "B":
            anchors = scope["b_match_anchors"]
            if (
                sum(
                    bool(anchors.get(name))
                    for name in ("release_date", "reporting_period", "fingerprint_facts")
                )
                < 2
            ):
                raise _error(
                    "pilot", "every B pilot scope must have two readable event anchor types"
                )
    pilot_filters = _validate_filter_rows(pilot_filter_rows, pilot_items)
    if set(pilot_filters) != set(pilot_items):
        raise _error("pilot", "v7 pilot filters must cover exactly the pilot items")
    _validate_query_rows(pilot_query_rows, expected_classes=V7_PILOT_QUERY_CLASSES)
    if pilot_label_rows:
        labels = _validate_v7_raw_rows(
            pilot_label_rows,
            pilot_items,
            pilot_views,
            rubric_sha256=_sha256_file(root / "rubric.md"),
        )
        if len(labels) != 40 or set(labels) != set(pilot_items):
            raise _error("pilot", "v7 pilot requires one Labeler A row per item")
    report = _load_json(root / "reports/calibration-report.json", "reports/calibration-report.json")
    if report.get("pilot_status") != "pilot" or report.get("pilot_n") not in {0, 40}:
        raise _error(
            "pilot", "populated v7 pilot must record pilot_status=pilot and pilot_n 0 or 40"
        )


def validate_calibration_root(root: str | Path) -> dict[str, Any]:
    """Validate a calibration bundle and return recomputed, non-claiming counts."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise _error(str(root_path), "calibration root is unavailable")
    manifest, manifest_sha = _validate_manifest(root_path)
    _validate_no_extra_files(root_path, calibration_id=manifest["calibration_id"])
    _validate_schema_documents(root_path)
    cohort = _validate_cohort(_load_json(root_path / "cohort.json", "cohort.json"), manifest)
    calibration_id = manifest["calibration_id"]
    _validate_authorization(
        _load_json(root_path / "authorization.json", "authorization.json"),
        calibration_id=calibration_id,
    )
    rubric_version = {
        RELEVANCE_CALIBRATION_ID: "leakage-calibration-rubric/v2",
        FRESH_RELEVANCE_CALIBRATION_ID: "leakage-calibration-rubric/v3",
        V5_RELEVANCE_CALIBRATION_ID: V5_RELEVANCE_RUBRIC_VERSION,
        V6_RELEVANCE_CALIBRATION_ID: V6_RELEVANCE_RUBRIC_VERSION,
        V7_RELEVANCE_CALIBRATION_ID: V7_RELEVANCE_RUBRIC_VERSION,
    }.get(calibration_id, "leakage-calibration-rubric/v1")
    rubric_sha256 = (
        _sha256_file(root_path / "rubric.md")
        if calibration_id
        in {
            RELEVANCE_CALIBRATION_ID,
            FRESH_RELEVANCE_CALIBRATION_ID,
            V5_RELEVANCE_CALIBRATION_ID,
            V6_RELEVANCE_CALIBRATION_ID,
            V7_RELEVANCE_CALIBRATION_ID,
        }
        else None
    )
    _validate_sampling_plan(_load_json(root_path / "sampling-plan.json", "sampling-plan.json"))
    query_rows = _load_jsonl(root_path / "query-corpus.jsonl", "query-corpus.jsonl")
    _validate_query_rows(query_rows)
    item_rows = _load_jsonl(root_path / "items.jsonl", "items.jsonl")
    items = _validate_item_rows(item_rows, cohort)
    view_rows = _load_jsonl(root_path / "annotation-view.jsonl", "annotation-view.jsonl")
    views = _validate_view_rows(
        view_rows,
        items,
        require_task_prompt=calibration_id
        in {
            FROZEN_CALIBRATION_ID,
            RELEVANCE_CALIBRATION_ID,
            V5_RELEVANCE_CALIBRATION_ID,
            V6_RELEVANCE_CALIBRATION_ID,
            V7_RELEVANCE_CALIBRATION_ID,
        },
    )
    filter_rows = _load_jsonl(root_path / "filter-decisions.jsonl", "filter-decisions.jsonl")
    filter_decisions = _validate_filter_rows(filter_rows, items)
    if items and set(filter_decisions) != set(items):
        raise _error(
            "filter-decisions.jsonl",
            "every receipt-bound item must have exactly one sealed filter decision",
        )
    labels_a_rows = _load_jsonl(
        root_path / "labels/raw/annotator-a.jsonl", "labels/raw/annotator-a.jsonl"
    )
    labels_b_rows = _load_jsonl(
        root_path / "labels/raw/annotator-b.jsonl", "labels/raw/annotator-b.jsonl"
    )
    labels_a, raw_a = _validate_raw_rows(
        labels_a_rows,
        items,
        views,
        expected_reviewer_type="labeler_a",
        relative="labels/raw/annotator-a.jsonl",
        expected_rubric_version=rubric_version,
        expected_rubric_sha256=rubric_sha256,
    )
    labels_b, raw_b = _validate_raw_rows(
        labels_b_rows,
        items,
        views,
        expected_reviewer_type="labeler_b",
        relative="labels/raw/annotator-b.jsonl",
        expected_rubric_version=rubric_version,
        expected_rubric_sha256=rubric_sha256,
    )
    adjudication_rows = _load_jsonl(
        root_path / "labels/adjudicated.jsonl", "labels/adjudicated.jsonl"
    )
    adjudicated, filter_invalid, determinate, positive, negative, unresolved = (
        _validate_adjudication_rows(
            adjudication_rows,
            items,
            views,
            labels_a,
            labels_b,
            filter_decisions=filter_decisions,
            expected_rubric_version=rubric_version,
            expected_rubric_sha256=rubric_sha256,
        )
    )
    counts = {
        "adjudicated": len(adjudicated),
        "determinate_binary_gold": determinate,
        "items": len(items),
        "leakage_negative": negative,
        "leakage_positive": positive,
        "raw_labels": raw_a + raw_b,
        "raw_labels_a": raw_a,
        "raw_labels_b": raw_b,
        "unknown": len(items) - determinate,
        "unresolved_disputes": unresolved,
    }
    adjudication_metadata = manifest["adjudication"]
    if adjudication_metadata["raw_labels_expected"] != 800:
        raise _error("artifact_manifest.json", "raw_labels_expected must be 800")
    if (
        adjudication_metadata["raw_labels_observed"] != counts["raw_labels"]
        or adjudication_metadata["adjudicated_observed"] != counts["adjudicated"]
        or adjudication_metadata["unresolved_disputes"] != counts["unresolved_disputes"]
    ):
        raise _error("artifact_manifest.json", "adjudication counts are not recomputable")
    if (
        items
        and manifest["restricted_evidence_manifest_sha256"] is None
        and calibration_id != V5_RELEVANCE_CALIBRATION_ID
    ):
        raise _error(
            "artifact_manifest.json",
            "populated bundles must bind a restricted evidence manifest SHA-256",
        )
    _validate_report(
        _load_json(
            root_path / "reports/calibration-report.json", "reports/calibration-report.json"
        ),
        manifest=manifest,
        counts=counts,
        filter_invalid=filter_invalid,
        manifest_sha=manifest_sha,
    )
    if manifest.get("calibration_status") == "not_started" and any(counts.values()):
        raise _error(
            "artifact_manifest.json", "not_started bundles must not contain calibration rows"
        )
    if calibration_id == CALIBRATION_ID and manifest.get("calibration_status") == "pilot":
        if counts["items"] != 40:
            raise _error("artifact_manifest.json", "pilot bundles require exactly 40 items")
        stratum_counts: dict[str, int] = {stratum: 0 for stratum in _STRATA}
        for item in item_rows:
            stratum = item.get("sampling_stratum")
            if stratum in stratum_counts:
                stratum_counts[stratum] += 1
        if stratum_counts != {stratum: 10 for stratum in _STRATA}:
            raise _error(
                "artifact_manifest.json",
                "pilot bundles require 10 items in each approved stratum",
            )
        if counts["raw_labels_a"] not in {0, 40}:
            raise _error(
                "artifact_manifest.json",
                "pilot bundles must contain either no Labeler A rows or all 40",
            )
        if counts["raw_labels_b"] or counts["adjudicated"]:
            raise _error(
                "artifact_manifest.json",
                "pilot bundles must not contain CR/QA labels or adjudication",
            )
    if (
        calibration_id == FROZEN_CALIBRATION_ID
        and manifest.get("calibration_status") != "not_started"
    ):
        if counts["items"] != 400:
            raise _error("artifact_manifest.json", "frozen bundles require exactly 400 items")
        filter_counts = Counter(row.get("filter_state") for row in filter_decisions.values())
        selected_state_counts = Counter(
            {
                "keep": filter_counts.get("keep", 0),
                "reject": filter_counts.get("reject", 0),
                "ambiguous": filter_counts.get("ambiguous", 0) + filter_counts.get("missing", 0),
            }
        )
        if selected_state_counts != Counter({"keep": 150, "reject": 150, "ambiguous": 100}):
            raise _error(
                "artifact_manifest.json",
                "frozen bundles require 150 keep, 150 reject, and 100 ambiguous/missing "
                "filter states",
            )
        stratum_counts = Counter(item.get("sampling_stratum") for item in item_rows)
        expected_strata = {
            "A1-T1": 101,
            "B-T1": 101,
            "B-T2": 99,
            "C-T2": 99,
        }
        if stratum_counts != Counter(expected_strata):
            raise _error(
                "artifact_manifest.json",
                "frozen bundles require the approved per-stratum item counts",
            )
        if counts["raw_labels_a"] not in {0, 400}:
            raise _error(
                "artifact_manifest.json",
                "frozen bundles must contain either no Labeler A rows or all 400",
            )
        if counts["raw_labels_b"] not in {0, 400}:
            raise _error(
                "artifact_manifest.json",
                "frozen bundles must contain either no Labeler B rows or all 400",
            )
    if manifest.get("calibration_status") == "validated" and counts["raw_labels"] != 800:
        raise _error("artifact_manifest.json", "validated bundles require 800 raw labels")
    if calibration_id == RELEVANCE_CALIBRATION_ID:
        _validate_v3_source_identity(root_path, item_rows, view_rows)
        _validate_v3_pilot(
            root_path,
            items,
            cohort,
            rubric_sha256=_sha256_file(root_path / "rubric.md"),
        )
    if calibration_id == FRESH_RELEVANCE_CALIBRATION_ID:
        if labels_a or labels_b or adjudicated:
            raise _error("v4", "v4 must not relabel the 400-item audit cohort")
        _validate_v4_source_identity(root_path, item_rows, view_rows)
        _validate_v4_pilot(
            root_path,
            items,
            cohort,
            rubric_sha256=_sha256_file(root_path / "rubric.md"),
        )
    if calibration_id == V5_RELEVANCE_CALIBRATION_ID:
        if labels_a or labels_b or adjudicated:
            raise _error("v5", "v5 must not relabel the 400-item audit cohort")
        _validate_v5_source_identity(root_path, item_rows, view_rows)
        _validate_v5_pilot(root_path)
    if calibration_id == V6_RELEVANCE_CALIBRATION_ID:
        if labels_a or labels_b or adjudicated:
            raise _error("v6", "v6 must not relabel the 400-item audit cohort")
        _validate_v6_source_identity(root_path, item_rows, view_rows)
        _validate_v6_pilot(root_path)
    if calibration_id == V7_RELEVANCE_CALIBRATION_ID:
        if labels_a or labels_b or adjudicated:
            raise _error("v7", "v7 must not relabel the 400-item audit cohort")
        _validate_v7_source_identity(root_path, item_rows, view_rows)
        _validate_v7_conformance(root_path)
        _validate_v7_pilot(root_path)
    return {
        "status": "valid",
        "calibration_id": calibration_id,
        "calibration_status": manifest["calibration_status"],
        "calibration_sha256": manifest_sha,
        "hub_repo_id": HUB_REPO_ID,
        "hub_revision": HUB_REVISION,
        "counts": counts,
        "live_unrestricted_authorized": False,
        "paper_probe_eligible": manifest["calibration_status"] == "validated"
        and not filter_invalid,
    }


__all__ = [
    "CALIBRATION_ID",
    "CALIBRATION_IDS",
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationValidationError",
    "FROZEN_CALIBRATION_ID",
    "FRESH_RELEVANCE_CALIBRATION_ID",
    "FRESH_RELEVANCE_COLLECTION_ID",
    "FRESH_RELEVANCE_RUBRIC_VERSION",
    "RELEVANCE_CALIBRATION_ID",
    "V5_RELEVANCE_AUTHORIZATION_ID",
    "V5_RELEVANCE_CALIBRATION_ID",
    "V5_RELEVANCE_COLLECTION_ID",
    "V5_RELEVANCE_RUBRIC_VERSION",
    "V5_RELEVANCE_SCOPE",
    "V5_RELEVANCE_SOURCE_COMMENT_ID",
    "V5_PILOT_JSONL_FILES",
    "V5_TRACKED_FILES",
    "V6_RELEVANCE_AUTHORIZATION_ID",
    "V6_RELEVANCE_CALIBRATION_ID",
    "V6_RELEVANCE_COLLECTION_ID",
    "V6_RELEVANCE_RUBRIC_VERSION",
    "V6_RELEVANCE_SCOPE",
    "V6_RELEVANCE_SOURCE_COMMENT_ID",
    "V6_PILOT_JSONL_FILES",
    "V6_TRACKED_FILES",
    "canonical_json_bytes",
    "validate_calibration_root",
]
