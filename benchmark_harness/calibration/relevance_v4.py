"""Task-bound relevance/outcome labeling for the fresh v4 pilot.

This module is separate from the immutable v3 failed-gate audit.  It consumes
only the task prompt and bounded model-visible candidate text; clocks, URLs,
filters, scores, and prior labels are not evidence for either field.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

RUBRIC_VERSION = "leakage-calibration-rubric/v3"

_METRICS = {
    "net_profit": ("净利润", "净利", "net profit", "net income", "profit"),
    "revenue": ("营业总收入", "营业收入", "营收", "revenue"),
    "gross_margin": ("毛利率", "gross margin"),
    "net_margin": ("净利率", "净利润率", "net margin"),
}
_A1_INPUTS = (
    "财务",
    "财报",
    "业绩",
    "营收",
    "收入",
    "利润",
    "现金流",
    "负债",
    "资产",
    "估值",
    "市盈率",
    "市净率",
    "资本结构",
    "股息",
    "分红",
    "成交量",
    "交易量",
    "股价",
    "收盘价",
    "目标价",
    "公允价值",
    "revenue",
    "profit",
    "earnings",
    "cash flow",
    "valuation",
    "capital structure",
    "dividend",
    "trading volume",
    "share price",
    "stock price",
    "fair value",
    "target price",
)
_ADMIN = (
    "annual report",
    "年报",
    "公司简介",
    "投资者关系",
    "investor relations",
    "下载",
    "目录",
    "contact",
)
_B_EVENT = (
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
    "指引",
    "预期",
    "共识",
    "revenue",
    "营收",
    "收入",
    "利润",
    "净利润",
    "net income",
    "net profit",
)
_PRICE_BRIDGE = (
    "股价",
    "股票价格",
    "该股",
    "证券",
    "share price",
    "stock price",
    "shares",
    "stock",
    "security",
)
_PRICE_MOVE = (
    "上涨",
    "下跌",
    "涨幅",
    "跌幅",
    "涨",
    "跌",
    "rose",
    "fell",
    "up",
    "down",
    "return",
    "回报",
    "reaction",
    "反应",
)
_PREDICTIVE = (
    "分析师",
    "评级",
    "目标价",
    "预期",
    "共识",
    "指引",
    "预测",
    "驱动",
    "driver",
    "guidance",
    "analyst",
    "rating",
    "target",
    "forecast",
    "consensus",
)
_REALIZED = (
    "actual",
    "reported",
    "公布",
    "实际",
    "实现",
    "季度",
    "季报",
    "同比",
    "环比",
    "q1",
    "q2",
    "q3",
    "q4",
)
_PERIODS = (
    "q1",
    "q2",
    "q3",
    "q4",
    "第一季度",
    "第二季度",
    "第三季度",
    "第四季度",
    "一季度",
    "二季度",
    "三季度",
    "四季度",
    "上半年",
    "下半年",
    "半年报",
    "quarter",
    "fiscal year",
)
_C_DRIVERS = (
    "成本",
    "费用",
    "毛利",
    "毛利率",
    "净利率",
    "价格",
    "销量",
    "产量",
    "margin",
    "cost",
    "expense",
    "volume",
    "pricing",
    "driver",
    "驱动",
    "guidance",
    "指引",
    "consensus",
    "共识",
)


@dataclass(frozen=True, slots=True)
class _Context:
    category: str
    entities: tuple[str, ...]
    metrics: tuple[str, ...]
    dates: tuple[str, ...]
    periods: tuple[str, ...]


def label_relevance_outcome(view: Mapping[str, object]) -> tuple[str, str, str]:
    """Return v4 ``task_relevance``, ``outcome_revealing``, and a short rationale."""

    prompt = view.get("task_prompt")
    text = _visible_text(view)
    if not isinstance(prompt, str) or not prompt.strip():
        return _result("unclear", "unclear", "task prompt is unavailable")
    if not text:
        return _result("unclear", "unclear", "candidate text is unavailable")
    context = _context(str(view.get("category", "")), prompt)
    if not context.entities:
        return _result("unclear", "unclear", "task entity is not identifiable")
    if not _has(text, context.entities):
        if _truncated(view) or len(text) < 40:
            return _result("unclear", "unclear", "subject cannot be resolved from visible text")
        return _result("mismatch", "no_new_signal", "visible subject does not match the task")

    if not _truncated(view):
        direct = _direct(context, text)
        if direct is not None:
            return _result("entity_event_metric", direct, _direct_span(context))
    reason = _relevance(context, text, view)
    if reason == "unclear":
        return _result("unclear", "unclear", "the visible task connection is ambiguous")
    if reason == "mismatch":
        return _result(
            "mismatch",
            "no_new_signal",
            "visible text proves a different subject, event, period, or metric",
        )
    outcome = _outcome(context, text)
    return _result(reason, outcome, _outcome_span(reason, outcome))


def _context(category: str, prompt: str) -> _Context:
    compact = re.sub(r"\s+", " ", prompt).strip()
    entities: list[str] = []
    for pattern in (
        r"对\s*([^（(]+?)\s*[（(]",
        r"股票[:：]\s*([^（(]+?)\s*[（(]",
        r"以下是\s*(.+?)\s*近两个季度",
    ):
        match = re.search(pattern, compact, flags=re.IGNORECASE)
        if match:
            value = re.sub(r"\s+", "", match.group(1)).strip(" ：:,，")
            if value:
                entities.append(value.casefold())
    entities.extend(
        value.casefold() for value in re.findall(r"[（(]([A-Za-z]{1,8}|[0-9]{4,6})[）)]", compact)
    )
    lower = compact.casefold()
    metrics = tuple(
        alias.casefold()
        for aliases in _METRICS.values()
        if any(alias.casefold() in lower for alias in aliases)
        for alias in aliases
    )
    dates = tuple(
        f"{year}{int(month):02d}{int(day):02d}"
        for year, month, day in re.findall(
            r"\b(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?\b", compact
        )
    )
    periods = list(
        f"q{value}" for value in re.findall(r"\bq([1-4])\b", compact, flags=re.IGNORECASE)
    )
    periods.extend(
        f"q{value}"
        for value, label in (("1", "一"), ("2", "二"), ("3", "三"), ("4", "四"))
        if f"{label}季度" in compact or f"{label}季报" in compact
    )
    if "上半年" in compact or "半年报" in compact:
        periods.append("h1")
    if "下半年" in compact:
        periods.append("h2")
    return _Context(
        category,
        tuple(dict.fromkeys(entities)),
        tuple(dict.fromkeys(metrics)),
        tuple(dict.fromkeys(dates)),
        tuple(dict.fromkeys(periods)),
    )


def _relevance(context: _Context, text: str, view: Mapping[str, object]) -> str:
    if context.category == "A1":
        if _has(text, _A1_INPUTS):
            return "entity_event_metric"
        return "unclear" if _truncated(view) else "mismatch"
    if context.category == "B":
        if not _has(text, _B_EVENT):
            return "unclear" if _truncated(view) else "mismatch"
        event = _same_event(context, text)
        return "unclear" if event is None else ("entity_event_metric" if event else "mismatch")
    if context.category == "C":
        if context.metrics and _has(text, context.metrics):
            return "entity_event_metric"
        if _has(text, _C_DRIVERS):
            return "explicit_driver"
        return "unclear" if _truncated(view) else "mismatch"
    return (
        "explicit_driver"
        if _has(text, _A1_INPUTS)
        else ("unclear" if _truncated(view) else "mismatch")
    )


def _outcome(context: _Context, text: str) -> str:
    if context.category == "A1":
        return "predictive_signal" if _has(text, _A1_INPUTS) else "no_new_signal"
    if context.category == "B":
        return (
            "predictive_signal"
            if _has(text, _PREDICTIVE) and _has_security_bridge(text)
            else "no_new_signal"
        )
    if context.category == "C":
        if (
            context.metrics
            and _has(text, context.metrics)
            and (_has(text, _PREDICTIVE) or _has(text, _REALIZED))
        ):
            return "predictive_signal"
        return "predictive_signal" if _has(text, _C_DRIVERS) else "no_new_signal"
    return "predictive_signal" if _has(text, _PREDICTIVE) else "no_new_signal"


def _direct(context: _Context, text: str) -> str | None:
    if context.category == "A1":
        return (
            "realized_exact"
            if _has(text, ("target price", "fair value", "目标价", "公允价值"))
            else None
        )
    if context.category == "B":
        return (
            "realized_exact"
            if _same_event(context, text) and _has_security_price_move(text)
            else None
        )
    if context.category == "C":
        if not context.metrics or not _has(text, context.metrics):
            return None
        if context.periods and not _has_period(context.periods, text):
            return None
        return "realized_exact" if _has(text, _REALIZED) and _has(text, _PERIODS) else None
    return None


def _same_event(context: _Context, text: str) -> bool | None:
    normalized = re.sub(r"[^a-z0-9]", "", text.casefold())
    if context.dates:
        if any(date in normalized for date in context.dates):
            return True
        candidate_dates = re.findall(r"\b20\d{6}\b", normalized)
        if candidate_dates:
            return False
    if context.periods:
        if _has_period(context.periods, text):
            return True
        if _has(text, _PERIODS):
            return False
    return True if _has(text, _B_EVENT) else None


def _has_period(periods: tuple[str, ...], text: str) -> bool:
    compact = re.sub(r"\s+", "", text.casefold())
    return any(
        period in compact
        or (period == "h1" and ("上半年" in compact or "半年报" in compact))
        or (period == "h2" and "下半年" in compact)
        or (
            period in {"q1", "q2", "q3", "q4"}
            and (f"第{period[1]}季度" in compact or f"{period[1]}季报" in compact)
        )
        for period in periods
    )


def _has_security_bridge(text: str) -> bool:
    return _has(text, _PRICE_BRIDGE) and _has(text, _PREDICTIVE)


def _has_security_price_move(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.casefold())
    bridge = "|".join(re.escape(value.casefold()) for value in _PRICE_BRIDGE)
    move = "|".join(re.escape(value.casefold()) for value in _PRICE_MOVE)
    return bool(
        re.search(rf"(?:{bridge}).{{0,60}}(?:{move})", normalized)
        or re.search(rf"(?:{move}).{{0,60}}(?:{bridge})", normalized)
    )


def _visible_text(view: Mapping[str, object]) -> str:
    return " ".join(
        str(view.get(field) or "")
        for field in ("model_visible_title", "model_visible_snippet", "model_visible_content")
    ).casefold()


def _truncated(view: Mapping[str, object]) -> bool:
    content = view.get("model_visible_content")
    return (
        not isinstance(content, str)
        or not content.strip()
        or content.rstrip().endswith(("...", "…"))
    )


def _has(text: str, terms: tuple[str, ...]) -> bool:
    compact = re.sub(r"\s+", "", text)
    return any(re.sub(r"\s+", "", term.casefold()) in compact for term in terms)


def _result(reason: str, outcome: str, span: str) -> tuple[str, str, str]:
    relevance = {
        "entity_event_metric": "relevant",
        "explicit_driver": "relevant",
        "task_input_repeat": "relevant",
        "mismatch": "not_relevant",
        "unclear": "indeterminate",
    }[reason]
    revealing = {
        "realized_exact": "direct",
        "deterministic_equivalent": "direct",
        "predictive_signal": "indirect",
        "no_new_signal": "no",
        "unclear": "indeterminate",
    }[outcome]
    if relevance == "not_relevant":
        revealing = "no"
    if relevance == "indeterminate":
        revealing = "indeterminate"
    return relevance, revealing, f"R={reason}; O={outcome}; span={span}"


def _direct_span(context: _Context) -> str:
    if context.category == "A1":
        return "matching security target or fair-value price is visible"
    if context.category == "B":
        return "same event and security price reaction are visible"
    return "matching metric and realized target-period value are visible"


def _outcome_span(reason: str, outcome: str) -> str:
    if reason == "explicit_driver":
        return "matching entity and an explicit financial driver are visible"
    if outcome == "predictive_signal":
        return "matching entity and a new guidance, driver, or period signal are visible"
    return "matching entity/event/metric is visible without a realized outcome"


__all__ = ["RUBRIC_VERSION", "label_relevance_outcome"]
