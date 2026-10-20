"""Task-bound, model-visible relevance and outcome labeling for rubric v2.

This module deliberately consumes only the task prompt and the bounded
model-visible candidate fields.  It never reads item clocks, filter decisions,
scores, answers, or ground truth.  The returned rationale uses the controlled
reason codes recorded in ``leakage-calibration-rubric/v2``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

RUBRIC_VERSION = "leakage-calibration-rubric/v2"

_ENTITY_PATTERNS = (
    r"对\s*([^（(]+?)\s*[（(]",
    r"股票[:：]\s*([^（(]+?)\s*[（(]",
    r"以下是\s*(.+?)\s*近两个季度",
)
_METRIC_ALIASES = {
    "net_profit": ("净利润", "净利", "net profit", "net income"),
    "operating_revenue": (
        "营业总收入",
        "营业收入",
        "营收",
        "operating revenue",
        "revenue",
    ),
    "gross_margin": ("毛利率", "gross margin"),
    "net_margin": ("净利率", "净利润率", "net margin"),
}
_FINANCIAL_FACTS = (
    "财务",
    "财报",
    "业绩",
    "营收",
    "收入",
    "利润",
    "现金流",
    "负债",
    "估值",
    "市盈率",
    "市净率",
    "目标价",
    "公允价值",
    "股价",
    "revenue",
    "profit",
    "earnings",
    "cash flow",
    "valuation",
    "target price",
    "fair value",
)
_EVENT_FACTS = (
    "eps",
    "earnings",
    "surprise",
    "财报",
    "业绩",
    "公告",
    "发布",
    "事件",
    "cpi",
    "pmi",
    "非农",
    "利率",
    "guidance",
    "预期",
    "共识",
)
_REACTION_FACTS = (
    "上涨",
    "下跌",
    "涨幅",
    "跌幅",
    "反应",
    "盘后",
    "次日",
    "shares rose",
    "shares fell",
    "stock rose",
    "stock fell",
    "return",
    "reaction",
)
_PREDICTIVE_FACTS = (
    "guidance",
    "指引",
    "分析师",
    "评级",
    "upgrade",
    "downgrade",
    "consensus",
    "共识",
    "预期",
    "forecast",
    "预测",
    "driver",
    "驱动",
    "成本",
    "行业",
)
_REALIZED_FACTS = (
    "actual",
    "reported",
    "公布",
    "实际",
    "实现",
    "季度",
    "q1",
    "q2",
    "q3",
    "q4",
    "同比",
    "环比",
)
_ADMINISTRATIVE_FACTS = (
    "annual report",
    "年报",
    "公司简介",
    "投资者关系",
    "investor relations",
    "下载",
    "目录",
    "contact",
)


@dataclass(frozen=True, slots=True)
class _TaskContext:
    category: str
    entity_terms: tuple[str, ...]
    metric_terms: tuple[str, ...]
    event_terms: tuple[str, ...]
    event_date_terms: tuple[str, ...]
    event_period_terms: tuple[str, ...]


def label_relevance_outcome(view: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return ``task_relevance``, ``outcome_revealing``, and controlled rationale."""

    prompt = view.get("task_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _result("unclear", "unclear", "task prompt is unavailable")
    candidate_text = _model_visible_text(view)
    if not candidate_text:
        return _result("unclear", "unclear", "candidate text is unavailable")

    context = _task_context(str(view.get("category", "")), prompt)
    if not context.entity_terms:
        return _result("unclear", "unclear", "task entity is not identifiable")
    if not _contains_any(candidate_text, context.entity_terms):
        if len(candidate_text) < 40 or _looks_truncated(view):
            return _result("unclear", "unclear", "subject cannot be resolved from visible text")
        return _result("mismatch", "no_new_signal", "visible subject does not match the task")
    if _looks_truncated(view):
        return _result("unclear", "unclear", "visible payload is truncated")

    relevance_reason = _relevance_reason(context, candidate_text, view)
    if relevance_reason == "unclear":
        return _result(
            "unclear", "unclear", "entity is visible but the task connection is ambiguous"
        )
    if relevance_reason == "mismatch":
        return _result(
            "mismatch", "no_new_signal", "entity is visible without the requested task connection"
        )

    outcome_reason = _outcome_reason(context, candidate_text, relevance_reason)
    return _result(relevance_reason, outcome_reason, _span_text(relevance_reason, outcome_reason))


def _task_context(category: str, prompt: str) -> _TaskContext:
    compact_prompt = re.sub(r"\s+", " ", prompt).strip()
    entity_terms: list[str] = []
    for pattern in _ENTITY_PATTERNS:
        match = re.search(pattern, compact_prompt, flags=re.IGNORECASE)
        if match:
            name = re.sub(r"\s+", "", match.group(1)).strip(" ：:,，")
            if name:
                entity_terms.append(name.casefold())
    for code in re.findall(r"[（(]([A-Za-z]{1,8}|[0-9]{4,6})[）)]", compact_prompt):
        entity_terms.append(code.casefold())

    prompt_lower = compact_prompt.casefold()
    metric_terms = tuple(
        alias.casefold()
        for aliases in _METRIC_ALIASES.values()
        for alias in aliases
        if alias.casefold() in prompt_lower
    )
    event_terms = tuple(term for term in _EVENT_FACTS if term.casefold() in prompt_lower)
    return _TaskContext(
        category=category,
        entity_terms=tuple(dict.fromkeys(entity_terms)),
        metric_terms=tuple(dict.fromkeys(metric_terms)),
        event_terms=tuple(dict.fromkeys(event_terms)),
        event_date_terms=_task_event_dates(compact_prompt),
        event_period_terms=_task_event_periods(compact_prompt),
    )


def _relevance_reason(context: _TaskContext, candidate_text: str, view: Mapping[str, Any]) -> str:
    category = context.category
    if category == "A1":
        if _contains_any(candidate_text, _FINANCIAL_FACTS):
            if _contains_any(candidate_text, _ADMINISTRATIVE_FACTS) and not _contains_any(
                candidate_text, _FINANCIAL_FACTS[:18]
            ):
                return "mismatch"
            return "entity_event_metric"
        return "unclear" if _looks_truncated(view) else "mismatch"
    if category == "B":
        has_event_signal = _contains_any(
            candidate_text, _EVENT_FACTS + _REACTION_FACTS + context.metric_terms
        )
        connection = _event_connection(context, candidate_text)
        if has_event_signal and connection is True:
            if _contains_any(candidate_text, _ADMINISTRATIVE_FACTS) and not _contains_any(
                candidate_text, _EVENT_FACTS + _REACTION_FACTS + context.metric_terms
            ):
                return "mismatch"
            return "entity_event_metric"
        if has_event_signal and connection is None:
            return "unclear"
        return "unclear" if _looks_truncated(view) else "mismatch"
    if category == "C":
        if context.metric_terms and _contains_any(candidate_text, context.metric_terms):
            return "entity_event_metric"
        if _contains_any(candidate_text, _PREDICTIVE_FACTS):
            return "explicit_driver"
        return "unclear" if _looks_truncated(view) else "mismatch"
    return "explicit_driver" if _contains_any(candidate_text, _FINANCIAL_FACTS) else "mismatch"


def _outcome_reason(context: _TaskContext, candidate_text: str, relevance_reason: str) -> str:
    if relevance_reason == "explicit_driver":
        return "predictive_signal"
    if context.category == "A1":
        if _contains_any(candidate_text, ("target price", "fair value", "目标价", "公允价值")):
            return "realized_exact"
        if _contains_any(candidate_text, _PREDICTIVE_FACTS + ("估值", "市盈率", "市净率")):
            return "predictive_signal"
        return "no_new_signal"
    if context.category == "B":
        if (
            _contains_any(candidate_text, _REACTION_FACTS)
            and _event_connection(context, candidate_text) is True
        ):
            return "realized_exact"
        if _contains_any(candidate_text, ("actual", "实际", "consensus", "共识")):
            return "no_new_signal"
        if _contains_any(candidate_text, _PREDICTIVE_FACTS):
            return "predictive_signal"
        return "no_new_signal"
    if context.category == "C":
        if _contains_any(candidate_text, context.metric_terms) and _contains_any(
            candidate_text, _REALIZED_FACTS
        ):
            return "realized_exact"
        if _contains_any(candidate_text, _PREDICTIVE_FACTS):
            return "predictive_signal"
        return "no_new_signal"
    return (
        "predictive_signal" if _contains_any(candidate_text, _PREDICTIVE_FACTS) else "no_new_signal"
    )


def _result(relevance_reason: str, outcome_reason: str, span: str) -> tuple[str, str, str]:
    relevance = {
        "entity_event_metric": "relevant",
        "explicit_driver": "relevant",
        "task_input_repeat": "relevant",
        "mismatch": "not_relevant",
        "unclear": "indeterminate",
    }[relevance_reason]
    outcome = {
        "realized_exact": "direct",
        "deterministic_equivalent": "direct",
        "predictive_signal": "indirect",
        "no_new_signal": "no",
        "unclear": "indeterminate",
    }[outcome_reason]
    return relevance, outcome, f"R={relevance_reason}; O={outcome_reason}; span={span}"


def _span_text(relevance_reason: str, outcome_reason: str) -> str:
    if relevance_reason == "entity_event_metric" and outcome_reason == "realized_exact":
        return "matching entity and realized task outcome are visible"
    if relevance_reason == "explicit_driver":
        return "matching entity and an explicit outcome driver are visible"
    if outcome_reason == "no_new_signal":
        return "matching entity is visible without a new outcome signal"
    return "matching entity and task-connected financial evidence are visible"


def _model_visible_text(view: Mapping[str, Any]) -> str:
    return " ".join(
        str(view.get(field) or "")
        for field in ("model_visible_title", "model_visible_snippet", "model_visible_content")
    ).casefold()


def _task_event_dates(prompt: str) -> tuple[str, ...]:
    dates: list[str] = []
    for year, month, day in re.findall(r"\b(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?\b", prompt):
        dates.append(f"{year}{int(month):02d}{int(day):02d}")
    return tuple(dict.fromkeys(dates))


def _task_event_periods(prompt: str) -> tuple[str, ...]:
    periods: list[str] = []
    for quarter in re.findall(r"\bq([1-4])\b", prompt, flags=re.IGNORECASE):
        periods.append(f"q{quarter}")
    chinese_quarters = {"一": "q1", "二": "q2", "三": "q3", "四": "q4"}
    for chinese, normalized in chinese_quarters.items():
        if f"{chinese}季度" in prompt or f"{chinese}季报" in prompt:
            periods.append(normalized)
    if "上半年" in prompt or "半年报" in prompt:
        periods.append("h1")
    if "下半年" in prompt:
        periods.append("h2")
    return tuple(dict.fromkeys(periods))


def _event_connection(context: _TaskContext, candidate_text: str) -> bool | None:
    """Return whether a visible candidate is tied to the task event.

    An explicit task date wins over generic event words.  If only a period
    marker is visible, the connection is indeterminate because a same-quarter
    event can still be a different release.
    """

    normalized = re.sub(r"[^a-z0-9]", "", candidate_text.casefold())
    if context.event_date_terms:
        if any(date in normalized for date in context.event_date_terms):
            return True
        candidate_dates = _task_event_dates(candidate_text)
        if candidate_dates:
            return False
        if context.event_period_terms and _candidate_period_matches(
            context.event_period_terms, candidate_text
        ):
            return None
        return False
    if context.event_period_terms:
        if _candidate_period_matches(context.event_period_terms, candidate_text):
            return True
        if _task_event_dates(candidate_text):
            return False
        if _has_any_period_marker(candidate_text):
            return False
        return None
    return _contains_any(candidate_text, _EVENT_FACTS + _REACTION_FACTS)


def _candidate_period_matches(periods: tuple[str, ...], candidate_text: str) -> bool:
    normalized = re.sub(r"\s+", "", candidate_text.casefold())
    for period in periods:
        if period in normalized:
            return True
        if period == "h1" and any(term in normalized for term in ("上半年", "半年报")):
            return True
        if period == "h2" and "下半年" in normalized:
            return True
        quarter = period.removeprefix("q")
        if any(term in normalized for term in (f"第{quarter}季度", f"{quarter}季报")):
            return True
    return False


def _has_any_period_marker(candidate_text: str) -> bool:
    normalized = re.sub(r"\s+", "", candidate_text.casefold())
    return bool(
        re.search(r"\bq[1-4]\b", normalized)
        or re.search(r"第[一二三四1-4]季度|[一二三四1-4]季报", normalized)
        or any(term in normalized for term in ("上半年", "下半年", "半年报"))
    )


def _contains_any(text: str, terms: tuple[str, ...] | list[str]) -> bool:
    compact_text = re.sub(r"\s+", "", text)
    return any(re.sub(r"\s+", "", term.casefold()) in compact_text for term in terms)


def _looks_truncated(view: Mapping[str, Any]) -> bool:
    content = view.get("model_visible_content")
    return (
        not isinstance(content, str)
        or not content.strip()
        or content.rstrip().endswith(("...", "…"))
    )


__all__ = ["RUBRIC_VERSION", "label_relevance_outcome"]
