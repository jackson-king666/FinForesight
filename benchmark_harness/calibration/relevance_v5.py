"""Task-bound relevance/outcome labeling for the v5 rubric audit.

The v5 rubric makes the decision order explicit: identify the entity, establish
family-specific applicability, and only then classify the outcome signal.  This
module is intentionally pure and offline.  It consumes the task prompt and
bounded model-visible text only; it does not inspect URLs, clocks, filters,
scores, or prior labels.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

RUBRIC_VERSION = "leakage-calibration-rubric/v4"

_METRIC_ALIASES = {
    "net_profit": ("净利润", "净利", "net profit", "net income"),
    "revenue": ("营业总收入", "营业收入", "营收", "revenue"),
    "gross_margin": ("毛利率", "gross margin"),
    "net_margin": ("净利率", "净利润率", "net margin"),
    "eps": ("每股收益", "每股盈利", "eps"),
}
_A1_FUNDAMENTAL = (
    "财务",
    "财报",
    "业绩",
    "营收",
    "收入",
    "利润",
    "现金流",
    "负债",
    "资产",
    "revenue",
    "profit",
    "earnings",
    "cash flow",
    "balance sheet",
)
_A1_MARKET = (
    "股价",
    "收盘价",
    "成交量",
    "交易量",
    "涨幅",
    "跌幅",
    "回报",
    "share price",
    "stock price",
    "trading volume",
    "return",
)
_A1_VALUATION = (
    "估值",
    "市盈率",
    "市净率",
    "市销率",
    "市值",
    "分部估值",
    "segment valuation",
    "market cap",
    "valuation",
    "multiple",
)
_A1_CAPITAL = (
    "股息",
    "分红",
    "回购",
    "拆股",
    "增发",
    "发行股票",
    "发行债券",
    "融资",
    "dividend",
    "buyback",
    "share repurchase",
    "stock split",
    "equity issuance",
    "debt issuance",
    "financing",
)
_A1_LINKED_FACTORS = (
    "风险",
    "资本成本",
    "恢复",
    "现金流",
    "earnings risk",
    "cost of capital",
    "recovery",
)
_A1_TARGET = ("目标价", "目标价格", "公允价值", "target price", "fair value")
_A1_PRICE = _A1_MARKET + ("price", "价格")
_B_EVENT = (
    "eps",
    "每股收益",
    "earnings",
    "财报",
    "业绩",
    "公告",
    "发布",
    "事件",
    "revenue",
    "营收",
    "收入",
    "利润",
    "净利润",
    "net income",
    "net profit",
    "surprise",
    "超预期",
    "cpi",
    "pmi",
    "非农",
    "利率",
    "guidance",
    "指引",
    "预期",
    "共识",
    "consensus",
)
_B_GENERIC_EVENT = frozenset(
    {"earnings", "revenue", "营收", "收入", "利润", "净利润", "net income", "net profit"}
)
_B_UNIQUE_EVENT = (
    "eps",
    "每股收益",
    "surprise",
    "超预期",
    "cpi",
    "pmi",
    "非农",
    "利率",
    "guidance",
    "指引",
    "consensus",
    "共识",
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
)
_PRICE_MOVE = (
    "上涨",
    "下跌",
    "涨幅",
    "跌幅",
    "上涨了",
    "下跌了",
    "rose",
    "fell",
    "rallied",
    "dropped",
    "return",
    "reaction",
    "反应",
)
_JUDGMENT = (
    "分析师",
    "评级",
    "目标价",
    "预期",
    "共识",
    "指引",
    "投资者",
    "analyst",
    "rating",
    "target",
    "forecast",
    "consensus",
    "investor",
    "expects",
    "expect",
)
_REALIZED = (
    "actual",
    "reported",
    "公布",
    "实际",
    "实现",
    "披露",
    "reported",
)
_C_RELATION = (
    "由于",
    "导致",
    "带动",
    "推动",
    "驱动",
    "原因",
    "贡献",
    "because",
    "due to",
    "driven by",
    "led by",
    "caused by",
)
_C_DRIVER = (
    "成本",
    "费用",
    "毛利",
    "毛利率",
    "价格",
    "销量",
    "产量",
    "margin",
    "cost",
    "expense",
    "volume",
    "pricing",
    "driver",
)
_C_COMPONENT = ("收入", "营收", "revenue", "成本", "费用", "cost", "expense", "margin")
_PERIOD_TERMS = ("q1", "q2", "q3", "q4", "h1", "h2", "fy", "第一季度", "第二季度")


@dataclass(frozen=True, slots=True)
class _Context:
    category: str
    entities: tuple[str, ...]
    metrics: tuple[str, ...]
    dates: tuple[str, ...]
    periods: tuple[str, ...]
    years: tuple[str, ...]
    numbers: tuple[str, ...]
    unique_events: tuple[str, ...]


def label_relevance_outcome(view: Mapping[str, object]) -> tuple[str, str, str]:
    """Return ``task_relevance``, ``outcome_revealing``, and rubric-v4 rationale."""

    prompt = view.get("task_prompt")
    visible = _visible_text(view)
    if not isinstance(prompt, str) or not prompt.strip() or not visible:
        return _result("UNCLEAR", "unclear", "na", "required task or visible text is unavailable")

    context = _context(str(view.get("category", "")), prompt)
    if not context.entities:
        return _result("UNCLEAR", "unclear", "na", "task entity is not identifiable")
    entity_match = _entity_match(context, visible, view)
    if entity_match == "mismatch":
        return _result("ENTITY_MISMATCH", "no", "none", "visible entity differs from the task")
    if entity_match == "unclear":
        return _result(
            "UNCLEAR", "unclear", "unclear", "entity is unresolved after visible truncation"
        )

    if context.category == "A1":
        rclass, oclass, event_match, span = _label_a1(context, visible, view)
    elif context.category == "B":
        rclass, event_match = _b_applicability(context, visible, view)
        if rclass == "B_SAME_EVENT":
            if _has_security_price_move(context, visible):
                oclass = "DIRECT_POST_EVENT_MOVE"
                span = "same event is explicitly tied to a security price move"
            elif _has_explicit_judgment(context, visible):
                oclass = "INDIRECT_INPUT"
                span = "same event has an explicit security-linked judgment"
            else:
                oclass = "SAME_EVENT_NO_PRICE_BRIDGE"
                span = "same-event facts are visible without a price bridge"
        else:
            oclass = "no"
            span = "the event or reporting period cannot establish applicability"
    elif context.category == "C":
        rclass, oclass, event_match, span = _label_c(context, visible, view)
    else:
        return _result("UNCLEAR", "unclear", "na", "unsupported task family")
    return _result(rclass, oclass, event_match, span)


def _context(category: str, prompt: str) -> _Context:
    compact = re.sub(r"\s+", " ", prompt).strip()
    entities: list[str] = []
    for match in re.finditer(
        r"(?P<name>[A-Za-z][A-Za-z0-9&.\-]{1,40}|[\u4e00-\u9fff]{2,20})\s*"
        r"[（(]\s*(?P<ticker>[A-Za-z0-9.\-]{1,12})\s*[）)]",
        compact,
    ):
        entities.extend((match.group("name"), match.group("ticker")))
    entities.extend(re.findall(r"[（(]([A-Za-z]{1,8}|[0-9]{4,6})[）)]", compact))
    if not entities:
        for pattern in (
            r"(?:股票|证券|公司|issuer|company)\s*[:：]?\s*([A-Za-z][A-Za-z0-9.\-]{1,30})",
            r"(?:分析|预测|研究)\s+([A-Za-z][A-Za-z0-9.\-]{1,30})",
        ):
            match = re.search(pattern, compact, flags=re.IGNORECASE)
            if match:
                entities.append(match.group(1))
                break
    lower = compact.casefold()
    metrics = tuple(
        metric
        for metric, aliases in _METRIC_ALIASES.items()
        if any(_has(lower, (alias,)) for alias in aliases)
    )
    dates = tuple(
        f"{year}{int(month):02d}{int(day):02d}"
        for year, month, day in re.findall(
            r"\b(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?\b", compact
        )
    )
    periods = list(f"q{value}" for value in re.findall(r"\bq([1-4])\b", lower))
    periods.extend(
        f"q{value}"
        for value, label in (("1", "一"), ("2", "二"), ("3", "三"), ("4", "四"))
        if f"{label}季度" in compact or f"{label}季报" in compact
    )
    if "上半年" in compact or "半年报" in compact or _has(lower, ("h1",)):
        periods.append("h1")
    if "下半年" in compact or _has(lower, ("h2",)):
        periods.append("h2")
    if _has(lower, ("fy", "fiscal year", "全年", "年度")):
        periods.append("fy")
    years = tuple(dict.fromkeys(re.findall(r"\b20\d{2}\b", compact)))
    numbers = tuple(
        value
        for value in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?%?", compact)
        if not (len(value.rstrip("%")) == 4 and value.rstrip("%").startswith("20"))
    )
    unique_events = tuple(term for term in _B_UNIQUE_EVENT if _has(lower, (term,)))
    return _Context(
        category=category.strip().upper(),
        entities=tuple(dict.fromkeys(value.casefold() for value in entities if value.strip())),
        metrics=metrics,
        dates=tuple(dict.fromkeys(dates)),
        periods=tuple(dict.fromkeys(periods)),
        years=years,
        numbers=tuple(dict.fromkeys(numbers)),
        unique_events=tuple(dict.fromkeys(unique_events)),
    )


def _label_a1(
    context: _Context, text: str, view: Mapping[str, object]
) -> tuple[str, str, str, str]:
    if _only_task_input_repeat(context, text):
        return "TASK_INPUT_REPEAT", "no", "na", "only the task-provided price is repeated"
    if _has(text, _A1_TARGET) and _has_number(text):
        return (
            "A1_TARGET",
            "DIRECT_TARGET",
            "na",
            "same-security target or fair-value price is visible",
        )
    if _has(text, _A1_FUNDAMENTAL):
        return "A1_FUNDAMENTAL", "INDIRECT_INPUT", "na", "same-security financial input is visible"
    if _has(text, _A1_MARKET):
        return "A1_MARKET", "INDIRECT_INPUT", "na", "same-security observed market input is visible"
    if _has(text, _A1_VALUATION):
        return "A1_VALUATION", "INDIRECT_INPUT", "na", "same-security valuation input is visible"
    if _has(text, _A1_CAPITAL):
        return "A1_CAPITAL", "INDIRECT_INPUT", "na", "same-security capital action is visible"
    if _has(text, _A1_LINKED_FACTORS):
        return (
            "A1_FUNDAMENTAL",
            "INDIRECT_INPUT",
            "na",
            "issuer-specific financial risk factor is visible",
        )
    if _truncated(view):
        return (
            "UNCLEAR",
            "unclear",
            "unclear",
            "visible truncation leaves A1 applicability unresolved",
        )
    return "NO_QUALIFYING_INPUT", "NO_SIGNAL", "na", "no qualifying A1 input is visible"


def _b_applicability(context: _Context, text: str, view: Mapping[str, object]) -> tuple[str, str]:
    if not _has(text, _B_EVENT):
        return ("B_EVENT_UNCLEAR", "unclear")
    candidate_dates, candidate_periods, candidate_years = _timeline(text)
    if context.dates:
        if set(context.dates).intersection(candidate_dates):
            return "B_SAME_EVENT", "exact"
        if candidate_dates:
            return "B_CROSS_PERIOD_ONLY", "cross_period_only"
        return "B_EVENT_UNCLEAR", "unclear"
    if context.periods:
        if (
            context.years
            and candidate_years
            and not set(context.years).intersection(candidate_years)
        ):
            return "B_CROSS_PERIOD_ONLY", "cross_period_only"
        if candidate_periods and not set(context.periods).intersection(candidate_periods):
            return "B_CROSS_PERIOD_ONLY", "cross_period_only"
        if candidate_periods.intersection(context.periods):
            if candidate_periods - set(context.periods):
                return "B_EVENT_UNCLEAR", "unclear"
            if context.years and not set(context.years).issubset(candidate_years):
                return "B_EVENT_UNCLEAR", "unclear"
            return "B_SAME_EVENT", "contains_exact"
        return "B_EVENT_UNCLEAR", "unclear"
    if context.unique_events:
        if _has(text, context.unique_events) and (
            not context.numbers or any(number in text for number in context.numbers)
        ):
            return "B_SAME_EVENT", "exact"
        return "B_EVENT_UNCLEAR", "unclear"
    # Generic earnings/revenue/profit is deliberately not a same-event fallback.
    return "B_EVENT_UNCLEAR", "unclear"


def _label_c(context: _Context, text: str, view: Mapping[str, object]) -> tuple[str, str, str, str]:
    candidate_metrics = _metric_families(text)
    if context.metrics and set(context.metrics).intersection(candidate_metrics):
        if _c_direct(context, text):
            return (
                "C_TARGET_METRIC",
                "DIRECT_TARGET",
                "exact",
                "target metric and target-period realized value are visible",
            )
        return (
            "C_TARGET_METRIC",
            "INDIRECT_INPUT",
            "contains_exact",
            "target metric is visible in a non-target or non-realized period",
        )
    if _has(text, _C_RELATION) and _has(text, _C_COMPONENT):
        return (
            "C_COMPONENT",
            "INDIRECT_INPUT",
            "none",
            "an explicit accounting component is linked to the target",
        )
    if _has(text, _C_RELATION) and _has(text, _C_DRIVER):
        return (
            "C_EXPLICIT_DRIVER",
            "INDIRECT_INPUT",
            "none",
            "an explicit financial driver is linked to the target",
        )
    if _truncated(view):
        return (
            "UNCLEAR",
            "unclear",
            "unclear",
            "visible truncation leaves C applicability unresolved",
        )
    return (
        "NO_QUALIFYING_INPUT",
        "NO_SIGNAL",
        "none",
        "a different metric has no explicit target connection",
    )


def _c_direct(context: _Context, text: str) -> bool:
    if not context.periods or not _has_period(context.periods, text):
        return False
    candidate_periods = _timeline(text)[1]
    if candidate_periods - set(context.periods):
        return False
    return _has(text, _REALIZED)


def _entity_match(context: _Context, text: str, view: Mapping[str, object]) -> str:
    if any(_contains_entity(text, entity) for entity in context.entities):
        return "match"
    candidate_entities = _candidate_entities(text)
    if candidate_entities and not set(candidate_entities).intersection(context.entities):
        return "mismatch"
    return "unclear" if _truncated(view) else "mismatch"


def _candidate_entities(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in re.finditer(
        r"(?P<name>[A-Za-z][A-Za-z0-9&.\-]{1,40}|[\u4e00-\u9fff]{2,20})\s*"
        r"[（(]\s*(?P<ticker>[A-Za-z0-9.\-]{1,12})\s*[）)]",
        text,
    ):
        values.extend((match.group("name").casefold(), match.group("ticker").casefold()))
    return tuple(dict.fromkeys(values))


def _contains_entity(text: str, entity: str) -> bool:
    if re.fullmatch(r"[a-z0-9.\-]+", entity):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(entity)}(?![a-z0-9])", text))
    return entity in text


def _has_security_price_move(context: _Context, text: str) -> bool:
    if not _has(text, _PRICE_MOVE):
        return False
    bridge_terms = _PRICE_BRIDGE + context.entities
    compact = re.sub(r"\s+", " ", text)
    bridge = "|".join(re.escape(term) for term in bridge_terms)
    move = "|".join(re.escape(term) for term in _PRICE_MOVE)
    return bool(
        re.search(rf"(?:{bridge}).{{0,120}}(?:{move})", compact)
        or re.search(rf"(?:{move}).{{0,120}}(?:{bridge})", compact)
    )


def _has_explicit_judgment(context: _Context, text: str) -> bool:
    return _has(text, _JUDGMENT) and any(
        _contains_entity(text, entity) for entity in context.entities
    )


def _only_task_input_repeat(context: _Context, text: str) -> bool:
    if not context.numbers or not _has(text, _A1_PRICE):
        return False
    visible_numbers = set(_numbers(text))
    if not visible_numbers or not visible_numbers.issubset(set(context.numbers)):
        return False
    independent = _A1_FUNDAMENTAL + _A1_VALUATION + _A1_CAPITAL + _A1_TARGET
    return not _has(text, independent)


def _timeline(text: str) -> tuple[tuple[str, ...], set[str], tuple[str, ...]]:
    compact = re.sub(r"\s+", "", text.casefold())
    dates = tuple(
        f"{year}{int(month):02d}{int(day):02d}"
        for year, month, day in re.findall(
            r"\b(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?\b", text
        )
    )
    periods = {f"q{value}" for value in re.findall(r"q([1-4])", compact)}
    periods.update(
        f"q{value}"
        for value, label in (("1", "一"), ("2", "二"), ("3", "三"), ("4", "四"))
        if f"{label}季度" in compact or f"{label}季报" in compact
    )
    if "上半年" in compact or "半年报" in compact or "h1" in compact:
        periods.add("h1")
    if "下半年" in compact or "h2" in compact:
        periods.add("h2")
    if "fy" in compact or "fiscalyear" in compact or "全年" in compact or "年度" in compact:
        periods.add("fy")
    years = tuple(dict.fromkeys(re.findall(r"\b20\d{2}\b", text)))
    return dates, periods, years


def _metric_families(text: str) -> set[str]:
    return {
        metric
        for metric, aliases in _METRIC_ALIASES.items()
        if any(_has(text, (alias,)) for alias in aliases)
    }


def _has_period(periods: tuple[str, ...], text: str) -> bool:
    return bool(set(periods).intersection(_timeline(text)[1]))


def _numbers(text: str) -> tuple[str, ...]:
    return tuple(
        value
        for value in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?%?", text)
        if not (len(value.rstrip("%")) == 4 and value.rstrip("%").startswith("20"))
    )


def _has_number(text: str) -> bool:
    return bool(_numbers(text))


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


def _has(text: str, terms: tuple[str, ...] | frozenset[str]) -> bool:
    compact = re.sub(r"\s+", "", text.casefold())
    return any(re.sub(r"\s+", "", term.casefold()) in compact for term in terms)


def _result(rclass: str, oclass: str, event_match: str, span: str) -> tuple[str, str, str]:
    relevance = {
        "A1_TARGET": "relevant",
        "A1_FUNDAMENTAL": "relevant",
        "A1_MARKET": "relevant",
        "A1_VALUATION": "relevant",
        "A1_CAPITAL": "relevant",
        "B_SAME_EVENT": "relevant",
        "C_TARGET_METRIC": "relevant",
        "C_COMPONENT": "relevant",
        "C_EXPLICIT_DRIVER": "relevant",
        "TASK_INPUT_REPEAT": "not_relevant",
        "NO_QUALIFYING_INPUT": "not_relevant",
        "B_CROSS_PERIOD_ONLY": "not_relevant",
        "ENTITY_MISMATCH": "not_relevant",
        "B_EVENT_UNCLEAR": "indeterminate",
        "UNCLEAR": "indeterminate",
    }[rclass]
    outcome = {
        "DIRECT_TARGET": "direct",
        "DIRECT_POST_EVENT_MOVE": "direct",
        "INDIRECT_INPUT": "indirect",
        "SAME_EVENT_NO_PRICE_BRIDGE": "no",
        "NO_SIGNAL": "no",
        "no": "no",
        "unclear": "indeterminate",
    }[oclass]
    if relevance == "not_relevant":
        outcome = "no"
    elif relevance == "indeterminate":
        outcome = "indeterminate"
    span = re.sub(r"[;\n]+", " ", span).strip()[:220]
    rationale = (
        f"RCLASS={rclass}; event_match={event_match}; "
        f"OCLASS={_oclass_for_rationale(oclass, relevance)}; span={span}"
    )
    return relevance, outcome, rationale


def _oclass_for_rationale(oclass: str, relevance: str) -> str:
    if relevance == "indeterminate":
        return "UNCLEAR"
    if relevance == "not_relevant":
        return "NO_SIGNAL"
    return oclass


__all__ = ["RUBRIC_VERSION", "label_relevance_outcome"]
