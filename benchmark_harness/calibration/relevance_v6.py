"""Offline Labeler-A classifier for rubric-v5 relevance calibration.

The classifier consumes only a v6 blind view.  In particular, URL, clocks,
sealed filter decisions, answers, and other labels are intentionally absent
from the input contract.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from benchmark_harness.calibration.v6_scope import validate_task_scope

RUBRIC_VERSION = "leakage-calibration-rubric/v5"

_A1_DIRECT = ("target price", "fair value", "目标价", "目标价格", "公允价值")
_A1_INPUT = (
    "revenue",
    "earnings",
    "profit",
    "cash flow",
    "balance sheet",
    "stock price",
    "share price",
    "trading volume",
    "return",
    "valuation",
    "market cap",
    "财报",
    "业绩",
    "营收",
    "收入",
    "利润",
    "现金流",
    "负债",
    "资产",
    "股价",
    "收盘价",
    "成交量",
    "回报",
    "估值",
    "市值",
    "分部估值",
    "股息",
    "分红",
    "回购",
    "拆股",
    "增发",
    "融资",
)
_B_EVENT = (
    "earnings",
    "financial report",
    "财报",
    "业绩",
    "公告",
    "eps",
    "每股收益",
    "revenue",
    "营收",
    "收入",
    "利润",
    "profit",
    "surprise",
    "超预期",
    "guidance",
    "指引",
    "cpi",
    "pmi",
    "非农",
    "利率",
)
_B_GENERIC = ("earnings", "revenue", "营收", "收入", "利润", "profit", "财报", "业绩")
_B_PRICE = ("stock", "shares", "share price", "stock price", "股价", "股票", "该股", "证券")
_B_MOVE = (
    "rose",
    "fell",
    "rallied",
    "dropped",
    "return",
    "reaction",
    "上涨",
    "下跌",
    "涨幅",
    "跌幅",
    "反应",
)
_B_JUDGMENT = (
    "analyst",
    "rating",
    "target",
    "recommendation",
    "investor",
    "分析师",
    "评级",
    "目标价",
    "建议",
    "投资者",
)
_C_FORECAST = ("forecast", "guidance", "consensus", "预期", "指引", "共识", "预测")
_C_DRIVER = (
    "component",
    "driver",
    "cost",
    "expense",
    "volume",
    "pricing",
    "margin",
    "成本",
    "费用",
    "销量",
    "价格",
    "毛利",
    "毛利率",
    "驱动",
    "原因",
    "贡献",
)
_C_RELATION = ("because", "due to", "driven by", "led by", "由于", "导致", "带动", "推动")
_METRIC_ALIASES = {
    "operating_revenue": ("operating_revenue", "revenue", "营业收入", "营业总收入", "营收"),
    "revenue": ("revenue", "营业收入", "营业总收入", "营收"),
    "net_profit": ("net profit", "net income", "净利润", "净利"),
    "net_margin": ("net margin", "净利润率", "净利率"),
    "gross_margin": ("gross margin", "毛利率"),
    "eps": ("eps", "每股收益", "每股盈利"),
}


@dataclass(frozen=True, slots=True)
class V6LabelDecision:
    task_relevance: str
    outcome_revealing: str
    rclass: str
    truncation: str
    hidden_axis: str
    entity_match: str
    event_match: str
    metric_match: str
    oclass: str
    span: str

    def rationale(self, task_scope_sha256: str) -> str:
        return (
            f"RCLASS={self.rclass}; task_scope_sha256={task_scope_sha256}; "
            f"truncation={self.truncation}; hidden_axis={self.hidden_axis}; "
            f"entity_match={self.entity_match}; event_match={self.event_match}; "
            f"metric_match={self.metric_match}; OCLASS={self.oclass}; span={self.span}"
        )


def classify_v6(view: Mapping[str, object]) -> V6LabelDecision:
    scope_value = view.get("task_scope")
    if not isinstance(scope_value, Mapping):
        raise ValueError("v6 view is missing task_scope")
    scope = dict(scope_value)
    validate_task_scope(scope)
    if view.get("category") != scope.get("family"):
        raise ValueError("v6 view category does not match task scope")
    visible = _visible_text(view)
    lower = visible.lower()
    truncation = view.get("truncation_metadata")
    if not isinstance(truncation, Mapping):
        raise ValueError("v6 view is missing truncation_metadata")
    proven = bool(truncation.get("proven_visible_truncation"))
    hidden_axis = str(truncation.get("hidden_axis", "none")) if proven else "none"
    if hidden_axis not in {"none", "entity", "event", "period", "metric"}:
        raise ValueError("v6 view has an unsupported hidden_axis")
    no_decisive = bool(truncation.get("no_decisive_visible_span"))
    entity = _first_text(scope, "entity_name", "entity_ticker")
    entity_present = _contains_any(lower, entity)
    if not entity_present and proven and hidden_axis == "entity" and no_decisive:
        return _decision(
            "indeterminate",
            "indeterminate",
            "TRUNCATED_SCOPE_ENTITY",
            "na",
            "na",
            "TRUNCATED_SCOPE",
            "proven",
            "entity",
            "match",
            visible,
        )
    if not entity_present:
        return _decision(
            "not_relevant",
            "no",
            "ENTITY_MISMATCH",
            "na",
            "na",
            "NO_SIGNAL",
            "proven" if proven else "none",
            hidden_axis if proven else "none",
            "mismatch",
            visible,
        )
    if proven and hidden_axis in {"event", "period", "metric"} and no_decisive:
        return _decision(
            "indeterminate",
            "indeterminate",
            f"TRUNCATED_SCOPE_{hidden_axis.upper()}",
            "not_proven" if hidden_axis == "event" else "na",
            "none" if hidden_axis == "metric" else "na",
            "TRUNCATED_SCOPE",
            "proven",
            hidden_axis,
            "match",
            visible,
        )
    family = str(scope["family"])
    if family == "A1":
        decision = _label_a1(lower)
    elif family == "B":
        decision = _label_b(lower, scope)
    else:
        decision = _label_c(lower, scope)
    return _decision(
        *decision,
        "proven" if proven else "none",
        hidden_axis if proven else "none",
        "match",
        visible,
    )


def label_relevance_outcome(view: Mapping[str, object]) -> tuple[str, str, str]:
    """Return the two labels and the fully structured v5 rationale."""

    decision = classify_v6(view)
    scope = view["task_scope"]
    assert isinstance(scope, Mapping)
    return (
        decision.task_relevance,
        decision.outcome_revealing,
        decision.rationale(str(scope["task_scope_sha256"])),
    )


def _label_a1(text: str) -> tuple[str, str, str, str, str, str]:
    if _contains_any(text, _A1_DIRECT) and _has_number(text):
        return "relevant", "direct", "A1_DIRECT", "na", "na", "DIRECT_TARGET"
    if _contains_any(text, _A1_INPUT):
        return "relevant", "indirect", "A1_INPUT", "na", "na", "INDIRECT_INPUT"
    return "not_relevant", "no", "A1_OTHER", "na", "na", "NO_SIGNAL"


def _label_b(text: str, scope: Mapping[str, object]) -> tuple[str, str, str, str, str, str]:
    event_match = _event_match(text, scope)
    if event_match == "cross_period":
        return "not_relevant", "no", "B_CROSS_PERIOD", event_match, "na", "NO_SIGNAL"
    if event_match == "not_proven":
        return "not_relevant", "no", "B_NOT_PROVEN", event_match, "na", "NO_SIGNAL"
    if _contains_any(text, _B_PRICE) and _contains_any(text, _B_MOVE):
        return "relevant", "direct", "B_EXACT_DIRECT", event_match, "na", "DIRECT_POST_EVENT_MOVE"
    if _contains_any(text, _B_JUDGMENT) and _contains_any(text, _B_PRICE):
        return "relevant", "indirect", "B_EXACT_JUDGMENT", event_match, "na", "INDIRECT_INPUT"
    return "relevant", "no", "B_EXACT_NO_BRIDGE", event_match, "na", "NO_SIGNAL"


def _label_c(text: str, scope: Mapping[str, object]) -> tuple[str, str, str, str, str, str]:
    metric = str(scope.get("c_target_metric", ""))
    metric_tokens = _METRIC_ALIASES.get(metric, (metric,))
    metric_match = "exact" if _contains_any(text, metric_tokens) else "none"
    if metric_match == "none":
        if _contains_any(text, _C_DRIVER) and _contains_any(text, _C_RELATION):
            return (
                "relevant",
                "indirect",
                "C_COMPONENT_OR_DRIVER",
                "na",
                "component",
                "INDIRECT_INPUT",
            )
        return "not_relevant", "no", "C_UNRELATED", "na", "none", "NO_SIGNAL"
    target_period = str(scope.get("c_target_period", ""))
    input_period = str(scope.get("c_input_period", ""))
    if _period_present(text, target_period) and _has_number(text):
        return "relevant", "direct", "C_EXACT_PERIOD", "na", "exact", "DIRECT_TARGET"
    if _period_present(text, input_period) or _contains_any(text, _C_FORECAST):
        return "relevant", "indirect", "C_OTHER_PERIOD_OR_FORECAST", "na", "exact", "INDIRECT_INPUT"
    if _contains_any(text, _C_DRIVER):
        return "relevant", "indirect", "C_COMPONENT_OR_DRIVER", "na", "component", "INDIRECT_INPUT"
    return "relevant", "no", "C_TARGET_NO_VALUE", "na", "exact", "NO_SIGNAL"


def _event_match(text: str, scope: Mapping[str, object]) -> str:
    target_date = str(scope.get("b_event_date", ""))
    if target_date and _period_present(text, target_date):
        return "exact"
    package = scope.get("b_reporting_package")
    if isinstance(package, str) and package:
        package_terms = [
            term for term in re.findall(r"20\d{2}|Q[1-4]|第[一二三四]季度", package, re.I)
        ]
        if package_terms and _contains_all(text, package_terms):
            return "contains_exact"
    years = set(re.findall(r"20\d{2}", text))
    target_year = target_date[:4] if len(target_date) >= 4 else ""
    if years and target_year and target_year not in years:
        return "cross_period"
    # A generic same-issuer earnings sentence is intentionally insufficient;
    # there is no same-event fallback without a task-owned period/fingerprint.
    return "not_proven"


def _period_present(text: str, period: str) -> bool:
    if not period:
        return False
    match = re.fullmatch(r"(20\d{2})-(\d{2})-(\d{2})", period)
    if not match:
        return period.lower() in text
    year, month, day = match.groups()
    if f"{year}-{month}-{day}" in text or f"{year}/{month}/{day}" in text:
        return True
    if f"{year}年{int(month)}月{int(day)}日" in text:
        return True
    quarter = (int(month) - 1) // 3 + 1
    return (
        f"{year} q{quarter}" in text
        or f"{year}q{quarter}" in text
        or f"{year}年{['一', '二', '三', '四'][quarter - 1]}季度" in text
    )


def _decision(
    task_relevance: str,
    outcome_revealing: str,
    rclass: str,
    event_match: str,
    metric_match: str,
    oclass: str,
    truncation: str,
    hidden_axis: str,
    entity_match: str,
    visible: str,
) -> V6LabelDecision:
    span = " ".join(visible.split()).replace(";", ",")[:120] or "no visible evidence"
    return V6LabelDecision(
        task_relevance=task_relevance,
        outcome_revealing=outcome_revealing,
        rclass=rclass,
        truncation=truncation,
        hidden_axis=hidden_axis,
        entity_match=entity_match,
        event_match=event_match,
        metric_match=metric_match,
        oclass=oclass,
        span=span,
    )


def _visible_text(view: Mapping[str, object]) -> str:
    return "\n".join(
        str(view.get(field) or "")
        for field in ("model_visible_title", "model_visible_snippet", "model_visible_content")
    ).strip()


def _first_text(scope: Mapping[str, object], *fields: str) -> tuple[str, ...]:
    return tuple(str(scope[field]).lower() for field in fields if scope.get(field))


def _contains_any(text: str, terms: tuple[str, ...] | list[str]) -> bool:
    return any(term.lower() in text for term in terms)


def _contains_all(text: str, terms: list[str]) -> bool:
    return all(term.lower() in text for term in terms)


def _has_number(text: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|亿|万)?", text))


__all__ = ["RUBRIC_VERSION", "V6LabelDecision", "classify_v6", "label_relevance_outcome"]
