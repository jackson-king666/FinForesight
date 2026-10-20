"""Task-owned scope extraction for the rubric-v5 calibration pilot.

The scope is derived before retrieval from the immutable Hub task record.  It
is deliberately smaller than the prompt and never contains answers, gold, or
filter metadata.  The same canonical scope is placed in both blind views so
that labelers can apply the entity -> applicability -> outcome rubric without
seeing retrieval-time clocks.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date, datetime

from benchmark_harness.contracts import sha256_json

V6_SCOPE_SCHEMA = "leakage-calibration-task-scope/v1"
V6_SCOPE_NORMALIZER = "task-scope/v1"

_DATE_RE = re.compile(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_C_PROMPT_RE = re.compile(
    r"以下是\s*(?P<entity>[^\s，。；:：]+)\s*近两个季度.*?「(?P<metric>[^」]+)」",
    re.DOTALL,
)
_C_PROMPT_ASCII_RE = re.compile(
    r"(?P<entity>[A-Za-z][A-Za-z0-9 ._-]{1,80})\s+.*?\((?P<metric>[A-Za-z_][A-Za-z0-9_]*)\)",
    re.DOTALL,
)


class TaskScopeError(ValueError):
    """Raised when a task lacks enough source-owned scope for labeling."""


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskScopeError(f"{field} is missing")
    return " ".join(value.split())


def _date_text(value: object, field: str) -> str:
    if isinstance(value, datetime | date):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    if isinstance(value, str):
        match = _DATE_RE.search(value)
        if match:
            return "-".join(
                f"{int(part):02d}" if index else part for index, part in enumerate(match.groups())
            )
    raise TaskScopeError(f"{field} is missing or is not a calendar date")


def _seed(row: Mapping[str, object]) -> Mapping[str, object]:
    value = row.get("seed")
    if not isinstance(value, Mapping):
        raise TaskScopeError("seed is missing")
    return value


def _expected_output(row: Mapping[str, object]) -> str:
    value = row.get("expected_output")
    if not isinstance(value, Mapping) or not value:
        raise TaskScopeError("expected_output is missing")
    keys = [key for key in value if isinstance(key, str) and key.strip()]
    if len(keys) != len(value):
        raise TaskScopeError("expected_output contains a non-string field")
    return "|".join(sorted(keys))


def _source_span(field: str, value: object) -> dict[str, str]:
    text = _string(value, field)
    return {"field": field, "span": text[:160]}


def _metric_from_prompt(prompt: str, seed: Mapping[str, object]) -> str:
    match = _C_PROMPT_RE.search(prompt)
    if match:
        return _string(match.group("metric"), "C target metric")
    # The ASCII fallback is for synthetic offline fixtures only; live C rows
    # must use the explicit Chinese prompt form above.
    match = _C_PROMPT_ASCII_RE.search(prompt)
    if match and seed.get("metric_name") == match.group("metric"):
        return match.group("metric")
    raise TaskScopeError("C prompt must contain the required entity and 「metric」 scope")


def _reporting_package(seed: Mapping[str, object], prompt: str) -> str | None:
    event_description = seed.get("event_description")
    package = " ".join(
        value
        for value in (
            str(seed.get("event_subtype", "")).strip(),
            str(seed.get("event_id", "")).strip(),
            str(event_description or "").strip(),
        )
        if value
    )
    period_match = re.search(
        r"(?:20\d{2}\s*(?:年|Q|q|年\s*第)|FY\s*)[^\n，。；:：]{0,24}",
        f"{package} {prompt}",
    )
    if period_match:
        return " ".join(period_match.group(0).split())[:160]
    return None


def _event_fingerprint(seed: Mapping[str, object]) -> str | None:
    event_id = seed.get("event_id")
    event_date = seed.get("event_date") or seed.get("cutoff_date")
    description = seed.get("event_description")
    if not any(value is not None for value in (event_id, event_date, description)):
        return None
    values = {
        "event_date": _date_text(event_date, "event_date") if event_date is not None else None,
        "event_description": " ".join(str(description or "").split()),
        "event_id": str(event_id or "").strip(),
    }
    if not values["event_id"] and not values["event_description"]:
        return None
    return sha256_json(values)


def build_task_scope(row: Mapping[str, object]) -> dict[str, object]:
    """Build and hash the source-owned scope, failing closed when incomplete."""

    category = _string(row.get("category"), "category")
    if category not in {"A1", "B", "C"}:
        raise TaskScopeError(f"unsupported pilot category {category!r}")
    seed = _seed(row)
    entity_name = _string(seed.get("stock_name"), "entity_name")
    entity_ticker = _string(seed.get("stock_code"), "entity_ticker")
    prompt = _string(row.get("prompt"), "prompt")
    requested_output = _expected_output(row)
    cutoff = _date_text(seed.get("cutoff_date") or row.get("cutoff_date"), "cutoff_date")
    spans = [
        _source_span("entity", f"{entity_name} ({entity_ticker})"),
        _source_span("cutoff_date", cutoff),
    ]
    scope: dict[str, object] = {
        "a1_target_type": None,
        "b_event_class": None,
        "b_event_date": None,
        "b_event_fingerprint": None,
        "b_reporting_package": None,
        "c_input_period": None,
        "c_target_metric": None,
        "c_target_period": None,
        "entity_name": entity_name,
        "entity_ticker": entity_ticker,
        "family": category,
        "normalizer_version": V6_SCOPE_NORMALIZER,
        "requested_output": requested_output,
        "source_spans": spans,
    }
    if category == "A1":
        if "market_value_range" in prompt or "市场价值" in prompt:
            target_type = "market_value_range"
        elif "target price" in prompt.lower() or "目标价" in prompt:
            target_type = "target_price"
        elif "valuation" in prompt.lower() or "估值" in prompt:
            target_type = "valuation"
        else:
            raise TaskScopeError("A1 target type is not identifiable")
        scope["a1_target_type"] = target_type
        spans.append(_source_span("a1_target_type", target_type))
    elif category == "B":
        event_class = _string(seed.get("event_subtype"), "b_event_class")
        event_date = _date_text(seed.get("event_date") or seed.get("cutoff_date"), "event_date")
        package = _reporting_package(seed, prompt)
        fingerprint = _event_fingerprint(seed)
        if package is None and fingerprint is None:
            raise TaskScopeError("B task has neither a reporting package nor event fingerprint")
        scope.update(
            {
                "b_event_class": event_class,
                "b_event_date": event_date,
                "b_event_fingerprint": fingerprint,
                "b_reporting_package": package,
            }
        )
        spans.extend(
            [
                _source_span("b_event_class", event_class),
                _source_span("b_event_date", event_date),
            ]
        )
        if package is not None:
            spans.append(_source_span("b_reporting_package", package))
    else:
        metric = _metric_from_prompt(prompt, seed)
        seed_metric = _string(seed.get("metric_name"), "c_target_metric")
        if metric != seed_metric:
            raise TaskScopeError("C prompt metric disagrees with source-owned metric")
        input_period = _date_text(seed.get("report_period_historical"), "c_input_period")
        target_period = _date_text(seed.get("report_period_future"), "c_target_period")
        scope.update(
            {
                "c_input_period": input_period,
                "c_target_metric": metric,
                "c_target_period": target_period,
            }
        )
        spans.extend(
            [
                _source_span("c_target_metric", metric),
                _source_span("c_input_period", input_period),
                _source_span("c_target_period", target_period),
            ]
        )
    scope["source_spans"] = spans
    scope["task_scope_sha256"] = sha256_json(scope)
    return scope


def validate_task_scope(scope: Mapping[str, object]) -> str:
    """Validate the scope hash and the family-specific required fields."""

    required = {
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
    if set(scope) != required:
        raise TaskScopeError("task scope fields are not exact")
    if scope.get("normalizer_version") != V6_SCOPE_NORMALIZER:
        raise TaskScopeError("unsupported task scope normalizer")
    family = scope.get("family")
    if family not in {"A1", "B", "C"}:
        raise TaskScopeError("unsupported task scope family")
    if family == "A1" and not isinstance(scope.get("a1_target_type"), str):
        raise TaskScopeError("A1 task scope lacks target type")
    if family == "B" and not (
        isinstance(scope.get("b_reporting_package"), str)
        or isinstance(scope.get("b_event_fingerprint"), str)
    ):
        raise TaskScopeError("B task scope lacks event identity")
    if family == "C" and not all(
        isinstance(scope.get(field), str)
        for field in ("c_input_period", "c_target_metric", "c_target_period")
    ):
        raise TaskScopeError("C task scope lacks metric/period identity")
    declared = scope.get("task_scope_sha256")
    unsigned = dict(scope)
    unsigned.pop("task_scope_sha256", None)
    expected = sha256_json(unsigned)
    if declared != expected:
        raise TaskScopeError("task scope hash does not match canonical fields")
    return expected


__all__ = [
    "TaskScopeError",
    "V6_SCOPE_NORMALIZER",
    "V6_SCOPE_SCHEMA",
    "build_task_scope",
    "validate_task_scope",
]
