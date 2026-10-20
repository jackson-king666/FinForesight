# ruff: noqa: E501
"""Source-owned task scope for the rubric-v6 calibration pilot.

This module is intentionally independent of the v1--v6 scope normalizers.  A
scope is made from the immutable Hub task row before retrieval and contains
readable event anchors for B tasks; it never consumes a retrieved page, a
filter decision, a clock, an answer, or a label.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date, datetime

from benchmark_harness.contracts import sha256_json

V8_SCOPE_SCHEMA = "leakage-calibration-task-scope/v2"
V8_SCOPE_NORMALIZER = "task-scope/v2"

_DATE_RE = re.compile(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_CN_DATE_RE = re.compile(r"(20\d{2})年(\d{1,2})月(\d{1,2})日?")
_C_PROMPT_RE = re.compile(
    r"以下是\s*(?P<entity>[^\s，。；:：]+)\s*近两个季度.*?「(?P<metric>[^」]+)」",
    re.DOTALL,
)
_C_PROMPT_ASCII_RE = re.compile(
    r"(?P<entity>[A-Za-z][A-Za-z0-9 ._-]{1,80})\s+.*?\((?P<metric>[A-Za-z_][A-Za-z0-9_]*)\)",
    re.DOTALL,
)
_REPORTING_PERIOD_RE = re.compile(
    r"(?P<year>20\d{2})年?(?:第?(?P<quarter>[一二三四1-4])季度|(?P<label>一季报|二季报|中报|半年报|三季报|四季报|年报))",
    re.IGNORECASE,
)
_EPS_RE = re.compile(
    r"(?:EPS|每股收益|每股盈利)\s*(?:实际|为|是)?\s*[:：]?\s*(-?\d+(?:\.\d+)?)\s*[,，;；]?\s*"
    r"(?:一致预期|共识预期|consensus)\s*(?:为|是)?\s*[:：]?\s*(-?\d+(?:\.\d+)?)"
    r"(?:[,，;；]?\s*(?:surprise|超预期|低于预期)\s*[:：]?\s*(-?\d+(?:\.\d+)?%?))?",
    re.IGNORECASE,
)
_NUMBER_FACT_RE = re.compile(
    r"(?:同比增长|同比下降|增长|下降|净利润|营收|收入|销量|毛利率|净利率|EPS|每股收益)[^，。；\n]{0,30}"
)


class TaskScopeError(ValueError):
    """Raised when a task lacks source-owned scope needed by the rubric."""


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskScopeError(f"{field} is missing")
    return " ".join(value.split())


def _date_text(value: object, field: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        match = _DATE_RE.search(value) or _CN_DATE_RE.search(value)
        if match:
            year, month, day = match.groups()
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
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
    match = _C_PROMPT_ASCII_RE.search(prompt)
    if match and seed.get("metric_name") == match.group("metric"):
        return match.group("metric")
    raise TaskScopeError("C prompt must contain the required entity and metric scope")


def _reporting_period(description: str) -> str | None:
    match = _REPORTING_PERIOD_RE.search(description)
    if not match:
        return None
    year = int(match.group("year"))
    quarter = match.group("quarter")
    if quarter:
        quarter_map = {"一": 1, "二": 2, "三": 3, "四": 4}
        quarter_number = quarter_map[quarter] if quarter in quarter_map else int(quarter)
        return f"{year} Q{quarter_number}"
    label = match.group("label")
    label_map = {
        "一季报": "Q1",
        "二季报": "Q2",
        "中报": "Q2",
        "半年报": "Q2",
        "三季报": "Q3",
        "四季报": "Q4",
        "年报": "FY",
    }
    return f"{year} {label_map[label]}"


def _fingerprint_facts(seed: Mapping[str, object]) -> list[str]:
    description = _string(seed.get("event_description"), "event_description")
    facts: list[str] = []
    match = _EPS_RE.search(description)
    if match:
        facts.extend(
            [
                f"actual_eps={match.group(1)}",
                f"consensus_eps={match.group(2)}",
            ]
        )
        if match.group(3):
            facts.append(f"surprise={match.group(3)}")
    for fact_match in _NUMBER_FACT_RE.finditer(description):
        fact = " ".join(fact_match.group(0).split())
        if fact and fact not in facts:
            facts.append(f"fact={fact[:100]}")
    # Some reviewed T2 seeds express the fingerprint qualitatively.  Keep a
    # bounded, task-owned phrase rather than an opaque event hash.  This is
    # still only an input fact, never a page-derived verdict.
    if not facts:
        normalized = " ".join(description.split())
        if normalized:
            facts.append(f"description={normalized[:120]}")
    return sorted(set(facts))[:8]


def build_task_scope(row: Mapping[str, object]) -> dict[str, object]:
    """Build the canonical v2 scope and its hash from a Hub task row."""

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
        "b_match_anchors": None,
        "c_input_period": None,
        "c_target_metric": None,
        "c_target_period": None,
        "entity_name": entity_name,
        "entity_ticker": entity_ticker,
        "family": category,
        "normalizer_version": V8_SCOPE_NORMALIZER,
        "requested_output": requested_output,
        "source_spans": spans,
    }
    if category == "A1":
        prompt_lower = prompt.lower()
        if "market_value_range" in prompt or "市场价值" in prompt:
            target_type = "market_value_range"
        elif "target price" in prompt_lower or "目标价" in prompt:
            target_type = "target_price"
        elif "valuation" in prompt_lower or "估值" in prompt:
            target_type = "valuation"
        else:
            raise TaskScopeError("A1 target type is not identifiable")
        scope["a1_target_type"] = target_type
        spans.append(_source_span("a1_target_type", target_type))
    elif category == "B":
        event_class = _string(seed.get("event_subtype"), "b_event_class")
        release_date = _date_text(seed.get("event_date") or seed.get("cutoff_date"), "event_date")
        description = _string(seed.get("event_description"), "event_description")
        reporting_period = _reporting_period(description)
        fingerprint_facts = _fingerprint_facts(seed)
        if not (release_date and (reporting_period or fingerprint_facts)):
            raise TaskScopeError("B task needs at least two readable event anchor types")
        anchors = {
            "event_class": event_class,
            "release_date": release_date,
            "reporting_period": reporting_period,
            "fingerprint_facts": fingerprint_facts,
        }
        scope["b_event_class"] = event_class
        scope["b_match_anchors"] = anchors
        spans.extend(
            [
                _source_span("b_event_class", event_class),
                _source_span("b_release_date", release_date),
                _source_span("b_fingerprint_fact", fingerprint_facts[0]),
            ]
        )
        if reporting_period:
            spans.append(_source_span("b_reporting_period", reporting_period))
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
    """Validate exact v2 fields, readable B anchors, and the canonical hash."""

    required = {
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
    if set(scope) != required:
        raise TaskScopeError("task scope fields are not exact")
    if scope.get("normalizer_version") != V8_SCOPE_NORMALIZER:
        raise TaskScopeError("unsupported task scope normalizer")
    family = scope.get("family")
    if family not in {"A1", "B", "C"}:
        raise TaskScopeError("unsupported task scope family")
    for field in ("entity_name", "entity_ticker", "requested_output"):
        _string(scope.get(field), field)
    if family == "A1":
        _string(scope.get("a1_target_type"), "a1_target_type")
    elif family == "B":
        _string(scope.get("b_event_class"), "b_event_class")
        anchors = scope.get("b_match_anchors")
        if not isinstance(anchors, Mapping) or set(anchors) != {
            "event_class",
            "release_date",
            "reporting_period",
            "fingerprint_facts",
        }:
            raise TaskScopeError("B scope anchors are not exact")
        if anchors.get("event_class") != scope.get("b_event_class"):
            raise TaskScopeError("B event class is not consistent with anchors")
        _date_text(anchors.get("release_date"), "b_match_anchors.release_date")
        reporting_period = anchors.get("reporting_period")
        if reporting_period is not None:
            _string(reporting_period, "b_match_anchors.reporting_period")
        facts = anchors.get("fingerprint_facts")
        if (
            not isinstance(facts, list)
            or not facts
            or any(not isinstance(fact, str) or not fact.strip() for fact in facts)
        ):
            raise TaskScopeError("B scope requires readable fingerprint facts")
        if (
            sum(
                bool(anchors.get(name))
                for name in ("release_date", "reporting_period", "fingerprint_facts")
            )
            < 2
        ):
            raise TaskScopeError("B scope requires at least two anchor types")
    else:
        for field in ("c_input_period", "c_target_metric", "c_target_period"):
            _string(scope.get(field), field)
        for field in ("c_input_period", "c_target_period"):
            _date_text(scope.get(field), field)
    spans = scope.get("source_spans")
    if not isinstance(spans, list) or not spans:
        raise TaskScopeError("source_spans must be a non-empty list")
    for span in spans:
        if not isinstance(span, Mapping) or set(span) != {"field", "span"}:
            raise TaskScopeError("source span fields are not exact")
        _string(span.get("field"), "source span field")
        if len(_string(span.get("span"), "source span")) > 160:
            raise TaskScopeError("source span exceeds the bounded length")
    declared = scope.get("task_scope_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise TaskScopeError("task_scope_sha256 is invalid")
    unsigned = dict(scope)
    unsigned.pop("task_scope_sha256")
    expected = sha256_json(unsigned)
    if declared != expected:
        raise TaskScopeError("task_scope_sha256 does not match canonical scope fields")
    return expected


__all__ = [
    "TaskScopeError",
    "V8_SCOPE_NORMALIZER",
    "V8_SCOPE_SCHEMA",
    "build_task_scope",
    "validate_task_scope",
]
