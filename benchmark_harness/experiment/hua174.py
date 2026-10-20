"""Execution-side identity, plan, and artifact contracts for HUA-174.

The HUA-174 runner deliberately keeps dataset records and model responses on
the runner side of the adapter boundary.  The plan contains enough public
identity to reproduce a run, while its JSON artifacts contain only hashes,
counts, and bounded status metadata.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.reference_evaluation.engine import (
    evaluate_prediction,
    parse_prediction,
)
from benchmark_harness.search_service.identities import service_manifest

HUA174_EXPERIMENT_ID = "hua174-de-c254092-s2-v1"
HUA174_PLAN_SCHEMA = "hua174-plan/v1"
HUA174_CELL_SCHEMA = "hua174-cell/v1"
HUA174_ATTEMPT_SCHEMA = "hua174-attempt/v1"
HUA174_LEDGER_SCHEMA = "hua174-final-ledger-720/v1"
HUA174_DATA_REVISION = "c254092af254d1d0dcc1401e50d65788620ae4dd"
HUA174_DATA_DIRNAME = "hua174-de-c254092"
HUA174_FROZEN_EVALUATOR_SHA256 = "c911d8146a0ca02a31ec80595b13611c94f097af21d60525bc08f447c446d50b"
HUA174_D_FILE_SHA256 = "570b78840d677fdf9ad494953c90b8728751965142f2d7be76e42ac221c7a8c5"
HUA174_E_FILE_SHA256 = "37b1c6bba51115c7d579be06370563e6d9079a7065db95334e2f5ebb0239503a"
HUA174_D_TASK_ID_SET_SHA256_8 = "77e4880c"
HUA174_E_TASK_ID_SET_SHA256_8 = "0abf7529"
HUA174_EXPECTED_TASKS = 60
HUA174_EXPECTED_CELLS = 720
HUA174_BLOCK_SIZE = 24
HUA174_MAX_INFLIGHT = 32
HUA174_SERVICE_PORT = 8773
HUA174_TRIAL = 1
# The source rows declare this experiment clock; keep it immutable rather than
# silently replacing it with the machine's current date.
HUA174_EXPERIMENT_AS_OF = date(2026, 8, 17)
HUA174_CATEGORIES = ("D", "E")
HUA174_FRAMEWORKS = ("financeharness", "odr", "gpt-researcher")
HUA174_MODEL_KEYS = ("gpt41mini", "gemini31lite", "sonnet45", "deepseek-v4-flash")
HUA174_MODEL_KEY_ALIASES = {"deepseek_v4_flash": "deepseek-v4-flash"}
HUA174_RETRIEVAL_REGIME = "hua52-s2-cutoff-filter/v1"
HUA174_SEARCH_PROFILE = "deep_research"
HUA174_SEARCH_POLICY = "deep_research/v2"
HUA174_SEARCH_PROTOCOL = "search_filter_http/v1"
HUA174_BUDGET = {
    "max_model_calls": 32,
    "max_search_calls": 8,
    "max_visit_calls": 12,
    "max_other_tool_calls": 40,
    "max_input_tokens": 4_194_304,
    "max_output_tokens": 2_097_152,
    "wall_clock_timeout_seconds": 7200.0,
}
HUA174_ATTEMPT_POLICY = {
    "max_attempts": 3,
    "retryable": ["infra_service"],
    "non_retryable": ["infra_auth", "true_refusal", "model_error"],
}
HUA174_FILTER_PROFILE = {
    "mode": "openai_compatible",
    "temperature": 0.0,
    "max_tokens": 2048,
    "timeout_seconds": 60.0,
    "retries": 2,
    "policy": "rule_then_llm_fail_closed",
}

# This is identity metadata only.  Runtime secrets and provider endpoints are
# loaded from the selected model's own env file by operator_hua174.py.
HUA174_MODEL_BINDINGS: dict[str, dict[str, Any]] = {
    "gpt41mini": {
        "env_file": ".env",
        "backbone_id": "openai:gpt-4.1-mini-2025-04-14",
        "wire_model_name": "gpt-4.1-mini",
        "snapshot_model_name": "gpt-4.1-mini-2025-04-14",
    },
    "gemini31lite": {
        "env_file": ".env-gemini-3.1",
        "backbone_id": "gemini-3.1-flash-lite",
        "wire_model_name": "gemini-3.1-flash-lite",
        "snapshot_model_name": "gemini-3.1-flash-lite",
    },
    "sonnet45": {
        "env_file": ".env-cc-sonnet-4.5",
        "backbone_id": "claude-sonnet-4-5-20250929",
        "wire_model_name": "claude-sonnet-4-5-20250929",
        "snapshot_model_name": "claude-sonnet-4-5-20250929",
    },
    "deepseek-v4-flash": {
        "env_file": ".env-deepseek-v4-flash",
        "backbone_id": "deepseek-v4-flash",
        "wire_model_name": "deepseek-v4-flash",
        "snapshot_model_name": "deepseek-v4-flash",
    },
}

_SECRET_KEY_RE = re.compile(
    r"(?i)(?:api[_-]?key|secret|password|token|authorization|credential|private[_-]?key)"
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)(?:api[_-]?key|secret|password|authorization|bearer)\s*[:=]\s*[^\s,}]+"
)
_RAW_CONTENT_KEYS = frozenset(
    {
        "prompt",
        "raw_text",
        "output",
        "response",
        "ground_truth",
        "record",
        "candidates",
        "search_audit",
        "audit_records",
        "network_ledger",
    }
)
_SAFE_RECEIPT_KEYS = frozenset(
    {
        "protocol",
        "receipt_sha256",
        "previous_receipt_sha256",
        "receipt_chain_head",
        "receipt_index",
        "service_request_id",
        "call_index",
        "receipt_count",
        "comparable",
        "status",
        "filter_review_count",
        "filter_model_call_count",
        "search_call_count",
        "llm_verdicts",
        "api_contract_sha256",
        "implementation_manifest_sha256",
        "capability_sha256",
        "service_cohort_sha256",
        "network_sha256",
        "audit_sha256",
    }
)


class Hua174PlanError(ValueError):
    """Raised when the HUA-174 source or plan identity is unsafe."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_evaluator() -> None:
    evaluator_path = Path(__file__).resolve().parents[1] / "reference_evaluation" / "engine.py"
    if (
        not evaluator_path.is_file()
        or sha256_file(evaluator_path) != HUA174_FROZEN_EVALUATOR_SHA256
    ):
        raise Hua174PlanError("frozen evaluator hash is not the approved HUA-174 pin")


def _task_id_set_hash(task_ids: Sequence[str]) -> str:
    return sha256_text("\n".join(sorted(task_ids)))[:8]


def _safe_relative_source(category: str) -> str:
    return f"data/{HUA174_DATA_DIRNAME}/{category.lower()}_train.jsonl"


def _resolve_data_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser().resolve()
    if (root / "manifest.json").is_file():
        return root
    candidate = root / HUA174_DATA_DIRNAME
    if (candidate / "manifest.json").is_file():
        return candidate
    raise Hua174PlanError(f"HUA-174 data manifest does not exist: {root}")


@dataclass(frozen=True, slots=True)
class Hua174Task:
    task_id: str
    category: str
    cutoff_date: date
    prompt: str
    time_band: str
    source_pointer: str
    record_sha256: str
    record: Mapping[str, Any] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.category not in HUA174_CATEGORIES:
            raise Hua174PlanError(f"unsupported HUA-174 category: {self.category}")
        if not self.task_id or not self.prompt:
            raise Hua174PlanError("HUA-174 task identity and prompt are required")
        if not self.source_pointer.startswith("data/"):
            raise Hua174PlanError("HUA-174 source pointer must be repository-relative")

    def to_dict(self) -> dict[str, Any]:
        # Deliberately excludes prompt, ground truth, seed, and all other
        # record content.  The runner retains the record only in memory.
        return {
            "task_id": self.task_id,
            "category": self.category,
            "cutoff_date": self.cutoff_date.isoformat(),
            "time_band": self.time_band,
            "source_pointer": self.source_pointer,
            "record_sha256": self.record_sha256,
        }


@dataclass(frozen=True, slots=True)
class Hua174Cell:
    cell_id: str
    task_id: str
    category: str
    model_key: str
    framework_id: str
    retrieval_regime: str
    trial: int
    block_id: str
    block_position: int
    execution_order: int
    source_pointer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HUA174_CELL_SCHEMA,
            "cell_id": self.cell_id,
            "task_id": self.task_id,
            "category": self.category,
            "model_key": self.model_key,
            "framework_id": self.framework_id,
            "retrieval_regime": self.retrieval_regime,
            "trial": self.trial,
            "block_id": self.block_id,
            "block_position": self.block_position,
            "execution_order": self.execution_order,
            "source_pointer": self.source_pointer,
        }


@dataclass(frozen=True, slots=True)
class Hua174Plan:
    experiment_id: str
    plan_id: str
    plan_hash: str
    manifest: Mapping[str, Any]
    tasks: tuple[Hua174Task, ...]
    cells: tuple[Hua174Cell, ...]
    evaluator_sha256: str = HUA174_FROZEN_EVALUATOR_SHA256

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HUA174_PLAN_SCHEMA,
            "experiment_id": self.experiment_id,
            "evaluator_sha256": self.evaluator_sha256,
            "manifest": dict(self.manifest),
            "tasks": [task.to_dict() for task in self.tasks],
            "cells": [cell.to_dict() for cell in self.cells],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._unsigned_dict(),
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
        }


def _load_records(data_root: Path, category: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = data_root / "manifest.json"
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Hua174PlanError("HUA-174 manifest is unreadable") from exc
    if not isinstance(raw_manifest, dict):
        raise Hua174PlanError("HUA-174 manifest must be an object")
    if raw_manifest.get("revision") != HUA174_DATA_REVISION:
        raise Hua174PlanError("HUA-174 data revision does not match c254092")
    manifest_files = raw_manifest.get("files")
    if not isinstance(manifest_files, dict) or category.lower() not in manifest_files:
        raise Hua174PlanError(f"HUA-174 manifest lacks {category} file metadata")
    expected = manifest_files[category.lower()]
    if not isinstance(expected, dict):
        raise Hua174PlanError(f"HUA-174 {category} file metadata is invalid")
    expected_sha256 = {
        "D": HUA174_D_FILE_SHA256,
        "E": HUA174_E_FILE_SHA256,
    }[category]
    expected_task_id_hash = {
        "D": HUA174_D_TASK_ID_SET_SHA256_8,
        "E": HUA174_E_TASK_ID_SET_SHA256_8,
    }[category]
    if expected.get("sha256") != expected_sha256:
        raise Hua174PlanError(f"HUA-174 {category} manifest hash is not the approved pin")
    if expected.get("task_id_set_sha256_8") != expected_task_id_hash:
        raise Hua174PlanError(f"HUA-174 {category} manifest task-id hash is not the approved pin")
    if expected.get("n_records") != 30:
        raise Hua174PlanError(f"HUA-174 {category} manifest count is not 30")
    file_name = f"{category.lower()}_train.jsonl"
    path = data_root / file_name
    if not path.is_file():
        raise Hua174PlanError(f"HUA-174 source file is missing: {file_name}")
    actual_hash = sha256_file(path)
    if actual_hash != expected.get("sha256"):
        raise Hua174PlanError(f"HUA-174 {category} file hash mismatch")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise Hua174PlanError(f"HUA-174 {category} file is unreadable") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Hua174PlanError(f"invalid JSON in HUA-174 {category} line {line_number}") from exc
        if not isinstance(row, dict):
            raise Hua174PlanError(f"HUA-174 {category} line {line_number} is not an object")
        row["_source_line"] = line_number
        rows.append(row)
    expected_count = expected.get("n_records")
    if len(rows) != expected_count:
        raise Hua174PlanError(f"HUA-174 {category} record count mismatch")
    task_ids = [row.get("task_id") for row in rows]
    if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
        raise Hua174PlanError(f"HUA-174 {category} task id is missing")
    if len(set(task_ids)) != len(task_ids):
        raise Hua174PlanError(f"HUA-174 {category} task ids are not unique")
    if _task_id_set_hash(task_ids) != expected.get("task_id_set_sha256_8"):
        raise Hua174PlanError(f"HUA-174 {category} task id set hash mismatch")
    return rows, raw_manifest


def _task_from_row(row: Mapping[str, Any], category: str) -> Hua174Task:
    task_id = row.get("task_id")
    prompt = row.get("prompt")
    cutoff = row.get("cutoff_date")
    if not isinstance(task_id, str) or not isinstance(prompt, str) or not isinstance(cutoff, str):
        raise Hua174PlanError(f"HUA-174 {category} task has invalid required fields")
    try:
        cutoff_date = date.fromisoformat(cutoff)
    except ValueError as exc:
        raise Hua174PlanError(f"HUA-174 {category} task cutoff is invalid") from exc
    source_line = row.get("_source_line")
    if not isinstance(source_line, int):
        raise Hua174PlanError("HUA-174 source line is missing")
    record = {key: value for key, value in row.items() if key != "_source_line"}
    return Hua174Task(
        task_id=task_id,
        category=category,
        cutoff_date=cutoff_date,
        prompt=prompt,
        time_band=str(row.get("time_band") or "unknown"),
        source_pointer=f"{_safe_relative_source(category)}#line={source_line}",
        record_sha256=sha256_json(record),
        record=record,
    )


def _plan_manifest(raw_manifest: Mapping[str, Any]) -> dict[str, Any]:
    files = raw_manifest.get("files")
    if not isinstance(files, Mapping):
        raise Hua174PlanError("HUA-174 manifest files must be an object")
    manifest_files: dict[str, Any] = {}
    for category in HUA174_CATEGORIES:
        file_info = files.get(category.lower())
        if not isinstance(file_info, Mapping):
            raise Hua174PlanError(f"HUA-174 manifest file metadata missing for {category}")
        manifest_files[category.lower()] = {
            "path": _safe_relative_source(category),
            "sha256": file_info["sha256"],
            "n_records": file_info["n_records"],
            "task_id_set_sha256_8": file_info["task_id_set_sha256_8"],
        }
    base_service = service_manifest(test_only=False).as_dict()
    return {
        "revision": HUA174_DATA_REVISION,
        "source": raw_manifest.get("source"),
        "files": manifest_files,
        "models": [
            {
                "model_key": key,
                **{
                    name: value
                    for name, value in HUA174_MODEL_BINDINGS[key].items()
                    if name != "env_file"
                },
                "env_file": HUA174_MODEL_BINDINGS[key]["env_file"],
                "temperature": 0.2,
                "top_p": 1.0,
                "max_tokens": 32768,
                "timeout_seconds": 600.0,
                "retries": 2,
            }
            for key in HUA174_MODEL_KEYS
        ],
        "frameworks": list(HUA174_FRAMEWORKS),
        "categories": list(HUA174_CATEGORIES),
        "retrieval": {
            "regime": HUA174_RETRIEVAL_REGIME,
            "profile": HUA174_SEARCH_PROFILE,
            "policy_version": HUA174_SEARCH_POLICY,
            "protocol": HUA174_SEARCH_PROTOCOL,
            "provider": "volc",
            "pit_cutoff": "inclusive_cutoff_date_with_next_day_exclusive_bound",
            "filter": dict(HUA174_FILTER_PROFILE),
            "service_identity": base_service,
        },
        "budget": dict(HUA174_BUDGET),
        "attempt_policy": dict(HUA174_ATTEMPT_POLICY),
        "trial": HUA174_TRIAL,
        "block_size": HUA174_BLOCK_SIZE,
        "max_inflight": HUA174_MAX_INFLIGHT,
        "experiment_as_of": HUA174_EXPERIMENT_AS_OF.isoformat(),
    }


def build_hua174_plan(
    data_root: str | Path,
    *,
    model_keys: Sequence[str] = HUA174_MODEL_KEYS,
    frameworks: Sequence[str] = HUA174_FRAMEWORKS,
) -> Hua174Plan:
    """Load and validate the pinned source, then build the deterministic matrix."""

    normalized_models = tuple(HUA174_MODEL_KEY_ALIASES.get(key, key) for key in model_keys)
    normalized_frameworks = tuple(frameworks)
    if set(normalized_models) != set(HUA174_MODEL_KEYS):
        raise Hua174PlanError("HUA-174 plan must contain exactly the four approved models")
    if normalized_frameworks != HUA174_FRAMEWORKS:
        raise Hua174PlanError("HUA-174 plan must contain exactly the three approved frameworks")
    _verify_frozen_evaluator()
    root = _resolve_data_root(data_root)
    records_by_category: dict[str, list[dict[str, Any]]] = {}
    raw_manifest: Mapping[str, Any] | None = None
    for category in HUA174_CATEGORIES:
        records, manifest = _load_records(root, category)
        records_by_category[category] = records
        raw_manifest = manifest
    if raw_manifest is None:
        raise Hua174PlanError("HUA-174 manifest was not loaded")
    tasks = tuple(
        _task_from_row(row, category)
        for category in HUA174_CATEGORIES
        for row in records_by_category[category]
    )
    if len(tasks) != HUA174_EXPECTED_TASKS:
        raise Hua174PlanError("HUA-174 task count is not 60")
    manifest = _plan_manifest(raw_manifest)
    cells: list[Hua174Cell] = []
    execution_order = 0
    for task in tasks:
        for model_key in HUA174_MODEL_KEYS:
            for framework_id in HUA174_FRAMEWORKS:
                block_number = execution_order // HUA174_BLOCK_SIZE + 1
                block_position = execution_order % HUA174_BLOCK_SIZE + 1
                cells.append(
                    Hua174Cell(
                        cell_id=(
                            f"{task.category}-{task.task_id}-{model_key}-{framework_id}-t{HUA174_TRIAL}"
                        ),
                        task_id=task.task_id,
                        category=task.category,
                        model_key=model_key,
                        framework_id=framework_id,
                        retrieval_regime=HUA174_RETRIEVAL_REGIME,
                        trial=HUA174_TRIAL,
                        block_id=f"block-{block_number:03d}",
                        block_position=block_position,
                        execution_order=execution_order,
                        source_pointer=task.source_pointer,
                    )
                )
                execution_order += 1
    if len(cells) != HUA174_EXPECTED_CELLS:
        raise Hua174PlanError("HUA-174 cell count is not 720")
    unsigned = {
        "schema_version": HUA174_PLAN_SCHEMA,
        "experiment_id": HUA174_EXPERIMENT_ID,
        "evaluator_sha256": HUA174_FROZEN_EVALUATOR_SHA256,
        "manifest": manifest,
        "tasks": [task.to_dict() for task in tasks],
        "cells": [cell.to_dict() for cell in cells],
    }
    plan_hash = sha256_json(unsigned)
    plan_id = f"{HUA174_EXPERIMENT_ID}-{plan_hash[:16]}"
    return Hua174Plan(
        experiment_id=HUA174_EXPERIMENT_ID,
        plan_id=plan_id,
        plan_hash=plan_hash,
        manifest=manifest,
        tasks=tasks,
        cells=tuple(cells),
    )


def write_hua174_plan(plan: Hua174Plan, path: str | Path) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(plan.to_dict()) + "\n", encoding="utf-8")
    return target


def load_hua174_plan_json(path: str | Path, *, data_root: str | Path | None = None) -> Hua174Plan:
    """Validate a canonical plan file against the pinned source when supplied."""

    plan_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Hua174PlanError("HUA-174 plan file is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise Hua174PlanError("HUA-174 plan file must be an object")
    if data_root is None:
        raise Hua174PlanError("data_root is required to rebind HUA-174 task records")
    plan = build_hua174_plan(data_root)
    if payload.get("plan_hash") != plan.plan_hash or payload.get("plan_id") != plan.plan_id:
        raise Hua174PlanError("HUA-174 plan file identity does not match pinned data")
    return plan


def validate_hua174_plan(plan: Hua174Plan) -> None:
    if plan.experiment_id != HUA174_EXPERIMENT_ID:
        raise Hua174PlanError("unexpected HUA-174 experiment identity")
    if len(plan.tasks) != HUA174_EXPECTED_TASKS or len(plan.cells) != HUA174_EXPECTED_CELLS:
        raise Hua174PlanError("HUA-174 plan dimensions are invalid")
    if plan.evaluator_sha256 != HUA174_FROZEN_EVALUATOR_SHA256:
        raise Hua174PlanError("frozen evaluator hash is not bound")
    if sha256_json(plan._unsigned_dict()) != plan.plan_hash:
        raise Hua174PlanError("HUA-174 plan hash mismatch")
    expected_orders = list(range(HUA174_EXPECTED_CELLS))
    if [cell.execution_order for cell in plan.cells] != expected_orders:
        raise Hua174PlanError("HUA-174 cell execution order is not deterministic")


@dataclass(frozen=True, slots=True)
class FailureClassification:
    category: str
    reason: str
    persist_attempt: bool
    operator_event_only: bool
    score_as_model_failure: bool


def classify_hua174_failure(error: BaseException | str) -> FailureClassification:
    """Classify only by safe error class/text, never by model correctness."""

    text = str(error).lower().replace("-", "_")
    identifiers = set(re.findall(r"[a-z][a-z0-9_]*", text))
    status_codes = set(re.findall(r"\b(?:401|403|429|502|503|504)\b", text))
    if status_codes & {"401", "403"} or identifiers & {
        "unauthorized",
        "forbidden",
        "api_key",
        "credential",
    }:
        return FailureClassification(
            category="infra_auth",
            reason="provider_authentication_or_authorization",
            persist_attempt=False,
            operator_event_only=True,
            score_as_model_failure=False,
        )
    if "service_receipt_missing" in identifiers:
        return FailureClassification(
            category="model_error",
            reason="zero_task_retrieval",
            persist_attempt=True,
            operator_event_only=False,
            score_as_model_failure=True,
        )
    if status_codes & {"429", "502", "503", "504"} or identifiers & {
        "transport",
        "timeout",
        "connection",
        "worker_eof",
        "receipt_chain",
        "service_unavailable",
        "service_identity_mismatch",
        "search_service_required",
        "search_service_ready_timeout",
    }:
        return FailureClassification(
            category="infra_service",
            reason="service_or_transport_failure",
            persist_attempt=True,
            operator_event_only=False,
            score_as_model_failure=False,
        )
    if any(marker in text for marker in ("refusal", "safety refusal", "content_policy")):
        return FailureClassification(
            category="true_refusal",
            reason="model_refusal",
            persist_attempt=True,
            operator_event_only=False,
            score_as_model_failure=False,
        )
    return FailureClassification(
        category="model_error",
        reason="model_or_adapter_failure",
        persist_attempt=True,
        operator_event_only=False,
        score_as_model_failure=True,
    )


def _redacted_string(value: str) -> dict[str, Any]:
    return {"sha256": sha256_text(value), "length": len(value)}


def redact_hua174(value: Any, *, key: str | None = None) -> Any:
    """Return artifact-safe JSON with raw content and credentials removed."""

    key_lower = key.lower() if isinstance(key, str) else ""
    if isinstance(value, str):
        if key_lower in _RAW_CONTENT_KEYS or _SECRET_KEY_RE.search(key_lower):
            return _redacted_string(value)
        return _SECRET_VALUE_RE.sub("<redacted>", value)
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_hua174(item_value, key=str(item_key))
            for item_key, item_value in value.items()
            if not (isinstance(item_key, str) and _SECRET_KEY_RE.search(item_key.lower()))
        }
    if isinstance(value, list | tuple):
        return [redact_hua174(item, key=key) for item in value]
    if isinstance(value, int | float | bool) or value is None:
        return value
    return str(value)


def safe_hua174_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only hash/count fields from a service search or close receipt."""

    safe: dict[str, Any] = {}
    for key, item in value.items():
        if key in _SAFE_RECEIPT_KEYS:
            if key == "llm_verdicts" and isinstance(item, Sequence) and not isinstance(item, str):
                safe[key] = [str(verdict) for verdict in item if str(verdict) in {"KEEP", "DROP"}]
            elif isinstance(item, str | int | float | bool) or item is None:
                safe[key] = item
    return safe


class Hua174ArtifactWriter:
    """Thread-safe append-only writer for redacted HUA-174 evidence."""

    def __init__(self, root: str | Path, plan: Hua174Plan) -> None:
        validate_hua174_plan(plan)
        self.root = Path(root).expanduser().resolve()
        if self.root.name != plan.experiment_id:
            raise Hua174PlanError("artifact root must end with the HUA-174 experiment id")
        self.plan = plan
        self.root.mkdir(parents=True, exist_ok=True)
        self.plan_path = self.root / "plan.json"
        self.attempts_path = self.root / "attempts.jsonl"
        self.receipts_path = self.root / "receipts.jsonl"
        self.scores_path = self.root / "scores.jsonl"
        self.service_evidence_path = self.root / "service-evidence.jsonl"
        self.operator_events_path = self.root / "operator-events.jsonl"
        self.final_ledger_path = self.root / "final-ledger-720.jsonl"
        self._lock = threading.RLock()
        self.plan_path.write_text(canonical_json(plan.to_dict()) + "\n", encoding="utf-8")

    def _append(self, path: Path, value: Mapping[str, Any]) -> None:
        with self._lock, path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(redact_hua174(value)) + "\n")

    def append_attempt(self, attempt: Mapping[str, Any]) -> None:
        status = attempt.get("status")
        if status in {"success", "true_refusal"} and not isinstance(
            attempt.get("receipt"), Mapping
        ):
            raise Hua174PlanError("successful/refused HUA-174 attempts require a service receipt")
        payload = {
            "schema_version": HUA174_ATTEMPT_SCHEMA,
            "experiment_id": self.plan.experiment_id,
            "plan_id": self.plan.plan_id,
            "plan_hash": self.plan.plan_hash,
            **dict(attempt),
        }
        receipt = payload.pop("receipt", None)
        self._append(self.attempts_path, payload)
        if isinstance(receipt, Mapping):
            self.append_receipt(receipt)

    def append_receipt(self, receipt: Mapping[str, Any]) -> None:
        payload = {
            "experiment_id": self.plan.experiment_id,
            "plan_hash": self.plan.plan_hash,
            **safe_hua174_receipt(receipt),
        }
        self._append(self.receipts_path, payload)

    def append_score(self, score: Mapping[str, Any]) -> None:
        metrics = score.get("metrics")
        safe_metrics: dict[str, Any] = {}
        if isinstance(metrics, Mapping):
            for key in (
                "format_valid",
                "logic_adherence",
                "relative_error",
                "exact_match_0_2pct",
                "formula_text_match",
                "unit_match",
            ):
                if key in metrics:
                    safe_metrics[key] = metrics[key]
        payload = {
            "experiment_id": self.plan.experiment_id,
            "plan_hash": self.plan.plan_hash,
            "cell_id": score.get("cell_id"),
            "task_id": score.get("task_id"),
            "category": score.get("category"),
            "raw_text_sha256": score.get("raw_text_sha256"),
            "metrics": safe_metrics,
        }
        self._append(self.scores_path, payload)

    def append_service_evidence(self, evidence: Mapping[str, Any]) -> None:
        self._append(self.service_evidence_path, evidence)

    def append_operator_event(self, event: Mapping[str, Any]) -> None:
        self._append(self.operator_events_path, event)

    def write_final_ledger(self, results: Mapping[str, Mapping[str, Any]]) -> Path:
        rows: list[dict[str, Any]] = []
        for cell in self.plan.cells:
            result = dict(results.get(cell.cell_id, {}))
            row = {
                "schema_version": HUA174_LEDGER_SCHEMA,
                "experiment_id": self.plan.experiment_id,
                "plan_id": self.plan.plan_id,
                "plan_hash": self.plan.plan_hash,
                "cell_id": cell.cell_id,
                "task_id": cell.task_id,
                "category": cell.category,
                "model_key": cell.model_key,
                "framework_id": cell.framework_id,
                "block_id": cell.block_id,
                "status": result.get("status", "gateway_infra_unresolved"),
                "attempt_id": result.get("attempt_id"),
                "failure_category": result.get("failure_category"),
                "score": result.get("score"),
                "receipt_sha256": result.get("receipt_sha256"),
            }
            rows.append(redact_hua174(row))
        with self._lock, self.final_ledger_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
        return self.final_ledger_path


def score_hua174_prediction(task: Hua174Task, raw_text: str) -> dict[str, Any]:
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise Hua174PlanError("HUA-174 model output must be non-empty")
    parsed = parse_prediction(task.category, raw_text)
    metrics = evaluate_prediction(task.category, parsed, dict(task.record))
    return {
        "category": task.category,
        "task_id": task.task_id,
        "raw_text_sha256": sha256_text(raw_text),
        "parsed": parsed,
        "metrics": metrics,
    }


def hua174_service_identity() -> dict[str, Any]:
    return {
        **service_manifest(test_only=False).as_dict(),
        "protocol": HUA174_SEARCH_PROTOCOL,
        "filter_profile": dict(HUA174_FILTER_PROFILE),
    }


def build_hua174_profile_public(profile: ResolvedModelProfile) -> dict[str, Any]:
    """Expose a profile without allowing credentials into plan/artifact JSON."""

    return profile.public_dict()


__all__ = [
    "HUA174_ATTEMPT_SCHEMA",
    "HUA174_BLOCK_SIZE",
    "HUA174_BUDGET",
    "HUA174_ATTEMPT_POLICY",
    "HUA174_CATEGORIES",
    "HUA174_CELL_SCHEMA",
    "HUA174_DATA_REVISION",
    "HUA174_D_FILE_SHA256",
    "HUA174_D_TASK_ID_SET_SHA256_8",
    "HUA174_E_FILE_SHA256",
    "HUA174_E_TASK_ID_SET_SHA256_8",
    "HUA174_EXPECTED_CELLS",
    "HUA174_EXPECTED_TASKS",
    "HUA174_EXPERIMENT_ID",
    "HUA174_EXPERIMENT_AS_OF",
    "HUA174_FILTER_PROFILE",
    "HUA174_FRAMEWORKS",
    "HUA174_FROZEN_EVALUATOR_SHA256",
    "HUA174_LEDGER_SCHEMA",
    "HUA174_MAX_INFLIGHT",
    "HUA174_MODEL_BINDINGS",
    "HUA174_MODEL_KEYS",
    "HUA174_PLAN_SCHEMA",
    "HUA174_RETRIEVAL_REGIME",
    "HUA174_SEARCH_POLICY",
    "HUA174_SEARCH_PROFILE",
    "HUA174_SEARCH_PROTOCOL",
    "HUA174_SERVICE_PORT",
    "FailureClassification",
    "Hua174ArtifactWriter",
    "Hua174Cell",
    "Hua174Plan",
    "Hua174PlanError",
    "Hua174Task",
    "build_hua174_plan",
    "build_hua174_profile_public",
    "canonical_json",
    "classify_hua174_failure",
    "hua174_service_identity",
    "load_hua174_plan_json",
    "redact_hua174",
    "safe_hua174_receipt",
    "score_hua174_prediction",
    "sha256_file",
    "sha256_json",
    "sha256_text",
    "validate_hua174_plan",
    "write_hua174_plan",
]
