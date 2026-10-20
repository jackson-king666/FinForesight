# ruff: noqa: E501
"""Deterministic rubric-v6 classifier using local, auditable evidence units.

The classifier deliberately separates applicability from outcome.  It never
concatenates distant page fragments when deciding a label: every positive
decision cites a sentence, bullet, table row, or the single permitted adjacent
identity/fact pair from the frozen annotation view.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from benchmark_harness.calibration.v8_scope import TaskScopeError, validate_task_scope

RUBRIC_VERSION = "leakage-calibration-rubric/v6"

SCOPE_RELATIONS = frozenset(
    {
        "requested_endpoint",
        "valuation_input",
        "exact_event",
        "other_event",
        "event_unproven",
        "exact_metric",
        "declared_component",
        "explicit_driver",
        "issuer_only",
        "mismatch",
        "truncated",
    }
)
PERIOD_RELATIONS = frozenset(
    {"exact_target", "other_only", "mixed_includes_target", "absent", "na"}
)
SIGNAL_KINDS = frozenset(
    {
        "requested_endpoint_value",
        "realized_post_event_security_move",
        "realized_target_metric_value",
        "valuation_input",
        "event_security_judgment",
        "target_metric_other_period",
        "target_metric_forecast",
        "declared_metric_component",
        "explicit_metric_driver",
        "scope_mention_only",
        "none",
        "truncated_scope",
    }
)

_DIRECT_SIGNALS = frozenset(
    {
        "requested_endpoint_value",
        "realized_post_event_security_move",
        "realized_target_metric_value",
    }
)
_INDIRECT_SIGNALS = frozenset(
    {
        "valuation_input",
        "event_security_judgment",
        "target_metric_other_period",
        "target_metric_forecast",
        "declared_metric_component",
        "explicit_metric_driver",
    }
)
_ENTITY_STOPWORDS = ("the company", "公司", "该公司", "issuer", "发行人")
_A1_ENDPOINT = (
    "target price",
    "price target",
    "fair value",
    "market value",
    "目标价",
    "目标价格",
    "公允价值",
    "市场价值",
)
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
    "股息",
    "分红",
    "回购",
    "拆股",
    "增发",
    "融资",
)
_A1_DIRECTION = ("overvalued", "undervalued", "高估", "低估", "上涨", "下跌")
_A1_ACCOUNTING = (
    "asset fair value",
    "liability fair value",
    "bond fair value",
    "loan fair value",
    "derivative fair value",
    "资产公允价值",
    "负债公允价值",
    "债券公允价值",
    "贷款公允价值",
    "衍生品公允价值",
    "资产的公允价值",
    "负债的公允价值",
)
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
    "收盘",
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
    "买入",
    "卖出",
    "维持",
)
_B_EVENT_ALIASES = {
    "earnings": (
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
    ),
    "financial_report": (
        "earnings",
        "financial report",
        "财报",
        "业绩",
        "公告",
        "eps",
        "每股收益",
        "营收",
        "收入",
        "利润",
    ),
    "cpi": ("cpi", "消费者价格指数", "consumer price"),
    "pmi": ("pmi", "采购经理人指数"),
    "employment": ("employment", "payroll", "非农", "就业"),
}
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
    "由于",
    "导致",
    "带动",
    "推动",
)
_METRIC_ALIASES = {
    "operating_revenue": ("operating_revenue", "revenue", "营业收入", "营业总收入", "营收"),
    "revenue": ("revenue", "营业收入", "营业总收入", "营收"),
    "net_profit": ("net profit", "net income", "净利润", "净利"),
    "net_margin": ("net margin", "净利润率", "净利率"),
    "gross_margin": ("gross margin", "毛利率"),
    "eps": ("eps", "每股收益", "每股盈利"),
}
_DATE_RE = re.compile(r"(?<!\d)(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])[-+]?\d+(?:[,.]\d+)*(?:%|亿元|万元|元|百万|million|billion)?", re.I
)
_UNIT_RE = re.compile(r"(?:%|percent|亿元|万元|元|million|billion|per share|每股)", re.I)
_LINE_RE = re.compile(r"[^\r\n]+")
_SENTENCE_END_RE = re.compile(r"[。！？；;]|(?<!\d)[.!?](?!\d)|(?<=\d)\.(?=\s|$)")
_BULLET_RE = re.compile(r"\s*(?:[-*+•]|(?:\d+|[A-Za-z])[.)])\s+")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(term and term.lower() in lowered for term in terms)


def _has_number(text: str) -> bool:
    return _NUMBER_RE.search(text) is not None


def _normalized(value: str) -> str:
    return " ".join(value.lower().replace("，", ",").split())


def segment_visible_view(view: Mapping[str, object]) -> list[dict[str, object]]:
    """Split visible fields into stable sentence/bullet/table-row evidence units."""

    units: list[dict[str, object]] = []
    for field in ("model_visible_title", "model_visible_snippet", "model_visible_content"):
        value = view.get(field)
        if not isinstance(value, str) or not value:
            continue
        for line_match in _LINE_RE.finditer(value):
            line_start, line_end = line_match.span()
            line = value[line_start:line_end]
            if _is_atomic_visible_line(line):
                _append_unit(units, field, value, line_start, line_end)
                continue
            sentence_start = line_start
            for ending in _SENTENCE_END_RE.finditer(line):
                sentence_end = line_start + ending.end()
                _append_unit(units, field, value, sentence_start, sentence_end)
                sentence_start = sentence_end
            _append_unit(units, field, value, sentence_start, line_end)
    return units


def _is_atomic_visible_line(line: str) -> bool:
    stripped = line.strip()
    return bool(
        stripped
        and (
            _BULLET_RE.match(line) is not None
            or (stripped.startswith("|") and stripped.count("|") >= 2)
        )
    )


def _append_unit(
    units: list[dict[str, object]],
    field: str,
    value: str,
    raw_start: int,
    raw_end: int,
) -> None:
    start, end = raw_start, raw_end
    while start < end and value[start].isspace():
        start += 1
    while end > start and value[end - 1].isspace():
        end -= 1
    if start == end:
        return
    text = value[start:end]
    units.append(
        {
            "field": field,
            "start_char": start,
            "end_char": end,
            "exact_text_sha256": _sha256_text(text),
            "text": text,
        }
    )


def validate_decisive_units(
    view: Mapping[str, object], units: object, *, require_visible_segment: bool = True
) -> list[dict[str, object]]:
    """Re-slice each cited unit and verify its exact hash against the view."""

    if not isinstance(units, list) or not units:
        raise ValueError("decisive_units must be a non-empty list")
    segments = segment_visible_view(view)
    validated: list[dict[str, object]] = []
    for unit_value in units:
        if not isinstance(unit_value, Mapping):
            raise ValueError("decisive unit must be an object")
        if set(unit_value) - {"field", "start_char", "end_char", "exact_text_sha256"}:
            raise ValueError("decisive unit has unknown fields")
        field = unit_value.get("field")
        start = unit_value.get("start_char")
        end = unit_value.get("end_char")
        declared = unit_value.get("exact_text_sha256")
        if field not in {"model_visible_title", "model_visible_snippet", "model_visible_content"}:
            raise ValueError("decisive unit field is not model-visible")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
        ):
            raise ValueError("decisive unit offsets must be integers")
        value = view.get(field)
        if not isinstance(value, str) or start < 0 or end <= start or end > len(value):
            raise ValueError("decisive unit offsets are outside the visible field")
        text = value[start:end]
        if not isinstance(declared, str) or declared != _sha256_text(text):
            raise ValueError("decisive unit hash does not match the frozen view")
        if require_visible_segment and not any(
            segment["field"] == field
            and segment["start_char"] == start
            and segment["end_char"] == end
            and segment["exact_text_sha256"] == declared
            for segment in segments
        ):
            raise ValueError("decisive unit is not a frozen sentence/bullet/table-row segment")
        validated.append(
            {
                "field": field,
                "start_char": start,
                "end_char": end,
                "exact_text_sha256": declared,
            }
        )
    if len(validated) > 2:
        raise ValueError("at most two adjacent decisive units are permitted")
    if len(validated) == 2:
        first, second = validated
        if first["field"] != second["field"] or int(first["end_char"]) > int(second["start_char"]):
            raise ValueError("two decisive units must be adjacent within one visible field")
    return validated


def _unit_text(unit: Mapping[str, object]) -> str:
    value = unit.get("text")
    return value if isinstance(value, str) else ""


def _unit_group(units: list[dict[str, object]], index: int) -> list[dict[str, object]]:
    group = [units[index]]
    if index + 1 < len(units):
        current, following = units[index], units[index + 1]
        if (
            current["field"] == following["field"]
            and int(following["start_char"]) <= int(current["end_char"]) + 1
        ):
            group.append(following)
    return group


def _group_text(group: Sequence[Mapping[str, object]]) -> str:
    return " ".join(_unit_text(unit) for unit in group)


def _scope_entity(scope: Mapping[str, object]) -> tuple[str, ...]:
    values = tuple(
        value.lower()
        for value in (scope.get("entity_name"), scope.get("entity_ticker"))
        if isinstance(value, str) and value.strip()
    )
    return values or _ENTITY_STOPWORDS


def _entity_present(text: str, scope: Mapping[str, object]) -> bool:
    return _contains_any(text, _scope_entity(scope))


def _date_forms(iso_date: str) -> tuple[str, ...]:
    try:
        year, month, day = (int(part) for part in iso_date.split("-"))
    except ValueError:
        return (iso_date,)
    return (
        f"{year:04d}-{month:02d}-{day:02d}",
        f"{year:04d}.{month:02d}.{day:02d}",
        f"{year:04d}/{month:02d}/{day:02d}",
        f"{year}年{month}月{day}日",
        f"{year}年{month}月{day}号",
    )


def _anchor_matches(text: str, scope: Mapping[str, object]) -> list[str]:
    anchors = scope.get("b_match_anchors")
    if not isinstance(anchors, Mapping):
        return []
    matches: list[str] = []
    release_date = anchors.get("release_date")
    if isinstance(release_date, str) and _contains_any(text, _date_forms(release_date)):
        matches.append("release_date")
    reporting_period = anchors.get("reporting_period")
    if isinstance(reporting_period, str):
        period_tokens = [
            reporting_period,
            reporting_period.replace(" Q", "年"),
            reporting_period.replace(" Q", " Q"),
        ]
        if _contains_any(text, tuple(period_tokens)):
            matches.append("reporting_period")
    facts = anchors.get("fingerprint_facts")
    if isinstance(facts, list):
        for fact in facts:
            if not isinstance(fact, str):
                continue
            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?%?", fact)
            if numbers and any(number.lower() in text.lower() for number in numbers):
                matches.append("fingerprint_fact")
                break
            phrase = fact.split("=", 1)[-1].strip()
            phrase_tokens = [
                token for token in re.findall(r"[A-Za-z_]{3,}|[\u4e00-\u9fff]{2,}", phrase)
            ]
            if phrase_tokens and any(token.lower() in text.lower() for token in phrase_tokens[:4]):
                matches.append("fingerprint_fact")
                break
    return sorted(set(matches))


def _event_aliases(scope: Mapping[str, object]) -> tuple[str, ...]:
    event_class = str(scope.get("b_event_class", "")).lower()
    for key, aliases in _B_EVENT_ALIASES.items():
        if key in event_class or event_class in aliases:
            return aliases
    return tuple(term for term in (event_class, "earnings", "财报", "公告") if term)


def derive_label_pair(
    scope_relation: str, period_relation: str, signal_kind: str
) -> tuple[str, str]:
    """Map structured applicability/signal fields to the only allowed pair."""

    if scope_relation not in SCOPE_RELATIONS:
        raise ValueError("unsupported scope_relation")
    if period_relation not in PERIOD_RELATIONS:
        raise ValueError("unsupported period_relation")
    if signal_kind not in SIGNAL_KINDS:
        raise ValueError("unsupported signal_kind")
    if signal_kind == "truncated_scope" or scope_relation == "truncated":
        return "indeterminate", "indeterminate"
    if signal_kind in _DIRECT_SIGNALS:
        return "relevant", "direct"
    if signal_kind in _INDIRECT_SIGNALS:
        return "relevant", "indirect"
    if signal_kind == "scope_mention_only" and scope_relation in {
        "exact_event",
        "exact_metric",
    }:
        return "relevant", "no"
    if (
        scope_relation
        in {
            "other_event",
            "event_unproven",
            "issuer_only",
            "mismatch",
        }
        or signal_kind == "none"
    ):
        return "not_relevant", "no"
    raise ValueError("structured fields do not form an allowed label pair")


@dataclass(frozen=True, slots=True)
class V8LabelDecision:
    task_relevance: str
    outcome_revealing: str
    scope_relation: str
    period_relation: str
    signal_kind: str
    event_anchor_matches: tuple[str, ...]
    formula_id: str | None
    decisive_units: tuple[dict[str, object], ...]
    entity_match: str
    event_match: str
    metric_match: str
    truncation: str
    hidden_axis: str
    rclass: str
    oclass: str

    def rationale(self) -> str:
        evidence = ",".join(str(unit["exact_text_sha256"])[:12] for unit in self.decisive_units)
        return (
            f"scope_relation={self.scope_relation}; period_relation={self.period_relation}; "
            f"signal_kind={self.signal_kind}; event_anchor_matches={','.join(self.event_anchor_matches) or 'none'}; "
            f"formula_id={self.formula_id or 'none'}; decisive_unit_sha256={evidence or 'none'}"
        )


def classify_v8(view: Mapping[str, object]) -> V8LabelDecision:
    """Classify one blind v8 view in the frozen decision order."""

    scope_value = view.get("task_scope")
    if not isinstance(scope_value, Mapping):
        raise ValueError("v8 view is missing task_scope")
    scope = dict(scope_value)
    try:
        validate_task_scope(scope)
    except TaskScopeError as exc:
        raise ValueError(str(exc)) from exc
    if view.get("category") != scope.get("family"):
        raise ValueError("v8 view category does not match task scope")
    truncation = view.get("truncation_metadata")
    if not isinstance(truncation, Mapping):
        raise ValueError("v8 view is missing truncation_metadata")
    units = segment_visible_view(view)
    visible = " ".join(_unit_text(unit) for unit in units)
    proven = bool(truncation.get("proven_visible_truncation"))
    hidden_axis = str(truncation.get("hidden_axis", "none")) if proven else "none"
    no_decisive = bool(truncation.get("no_decisive_visible_span"))
    entity = _entity_present(visible, scope)
    if not entity and proven and hidden_axis == "entity" and no_decisive:
        return _decision(
            scope,
            "truncated",
            "na",
            "truncated_scope",
            (),
            None,
            (),
            "mismatch",
            "not_proven",
            "none",
            "proven",
            "entity",
            "TRUNCATED_SCOPE_ENTITY",
            "TRUNCATED_SCOPE",
        )
    if not entity:
        return _decision(
            scope,
            "mismatch",
            "na",
            "none",
            (),
            None,
            (),
            "mismatch",
            "na",
            "none",
            "proven" if proven else "none",
            hidden_axis if proven else "none",
            "ENTITY_MISMATCH",
            "NO_SIGNAL",
        )
    family = str(scope["family"])
    if family == "A1":
        decision = _classify_a1(units, scope)
    elif family == "B":
        decision = _classify_b(units, scope)
    else:
        decision = _classify_c(units, scope)
    if proven and no_decisive and not decision[5] and hidden_axis in {"event", "period", "metric"}:
        return _decision(
            scope,
            "truncated",
            "na",
            "truncated_scope",
            (),
            None,
            (),
            "match",
            "not_proven" if family == "B" else "na",
            "none",
            "proven",
            hidden_axis,
            f"TRUNCATED_SCOPE_{hidden_axis.upper()}",
            "TRUNCATED_SCOPE",
        )
    return _decision(
        scope,
        decision[0],
        decision[1],
        decision[2],
        decision[3],
        decision[4],
        decision[5],
        decision[6],
        decision[7],
        decision[8],
        "proven" if proven else "none",
        hidden_axis if proven else "none",
        decision[9],
        decision[10],
    )


def _decision(
    scope: Mapping[str, object],
    scope_relation: str,
    period_relation: str,
    signal_kind: str,
    anchors: Sequence[str],
    formula_id: str | None,
    units: Sequence[Mapping[str, object]],
    entity_match: str,
    event_match: str,
    metric_match: str,
    truncation: str,
    hidden_axis: str,
    rclass: str,
    oclass: str,
) -> V8LabelDecision:
    relevance, outcome = derive_label_pair(scope_relation, period_relation, signal_kind)
    normalized_units = tuple(
        {
            "field": unit["field"],
            "start_char": unit["start_char"],
            "end_char": unit["end_char"],
            "exact_text_sha256": unit["exact_text_sha256"],
        }
        for unit in units
    )
    return V8LabelDecision(
        task_relevance=relevance,
        outcome_revealing=outcome,
        scope_relation=scope_relation,
        period_relation=period_relation,
        signal_kind=signal_kind,
        event_anchor_matches=tuple(anchors),
        formula_id=formula_id,
        decisive_units=normalized_units,
        entity_match=entity_match,
        event_match=event_match,
        metric_match=metric_match,
        truncation=truncation,
        hidden_axis=hidden_axis,
        rclass=rclass,
        oclass=oclass,
    )


def _classify_a1(units: list[dict[str, object]], scope: Mapping[str, object]) -> tuple[object, ...]:
    for index, unit in enumerate(units):
        group = _unit_group(units, index)
        text = _group_text(group)
        if not _entity_present(text, scope):
            continue
        endpoint = _contains_any(text, _A1_ENDPOINT)
        accounting = _contains_any(text, _A1_ACCOUNTING)
        if endpoint and _has_number(text) and not accounting:
            return (
                "requested_endpoint",
                "na",
                "requested_endpoint_value",
                (),
                None,
                group,
                "match",
                "na",
                "na",
                "A1_REQUESTED_ENDPOINT",
                "DIRECT_TARGET",
            )
        if (
            _contains_any(text, _A1_INPUT)
            and (_has_number(text) or _contains_any(text, _A1_DIRECTION))
            and not accounting
        ):
            return (
                "valuation_input",
                "na",
                "valuation_input",
                (),
                None,
                group,
                "match",
                "na",
                "na",
                "A1_VALUATION_INPUT",
                "INDIRECT_INPUT",
            )
    return (
        "issuer_only",
        "na",
        "none",
        (),
        None,
        (),
        "match",
        "na",
        "none",
        "A1_NO_SIGNAL",
        "NO_SIGNAL",
    )


def _classify_b(units: list[dict[str, object]], scope: Mapping[str, object]) -> tuple[object, ...]:
    aliases = _event_aliases(scope)
    best_event: tuple[list[dict[str, object]], list[str]] | None = None
    for index, unit in enumerate(units):
        group = _unit_group(units, index)
        text = _group_text(group)
        if not _entity_present(text, scope) or not _contains_any(text, aliases):
            continue
        anchors = _anchor_matches(text, scope)
        if len(anchors) >= 2:
            best_event = (group, anchors)
            break
    if best_event is None:
        has_event_words = any(
            _entity_present(_unit_text(unit), scope) and _contains_any(_unit_text(unit), aliases)
            for unit in units
        )
        relation = "event_unproven" if has_event_words else "other_event"
        return (
            relation,
            "na",
            "none",
            (),
            None,
            (),
            "match",
            "not_proven",
            "none",
            "B_EVENT_UNPROVEN",
            "NO_SIGNAL",
        )
    group, anchors = best_event
    text = _group_text(group)
    if (
        _contains_any(text, _B_PRICE)
        and _contains_any(text, _B_MOVE)
        and _contains_any(text, ("after", "following", "reaction", "因", "后", "随后", "事件后"))
    ):
        return (
            "exact_event",
            "na",
            "realized_post_event_security_move",
            anchors,
            None,
            group,
            "match",
            "exact",
            "na",
            "B_EXACT_DIRECT",
            "DIRECT_POST_EVENT_MOVE",
        )
    if _contains_any(text, _B_JUDGMENT) and _contains_any(text, _B_PRICE):
        return (
            "exact_event",
            "na",
            "event_security_judgment",
            anchors,
            None,
            group,
            "match",
            "exact",
            "na",
            "B_EXACT_JUDGMENT",
            "INDIRECT_INPUT",
        )
    return (
        "exact_event",
        "na",
        "scope_mention_only",
        anchors,
        None,
        group,
        "match",
        "exact",
        "na",
        "B_EXACT_NO_BRIDGE",
        "NO_SIGNAL",
    )


def _period_relation(text: str, scope: Mapping[str, object]) -> str:
    target = str(scope.get("c_target_period", ""))
    source = str(scope.get("c_input_period", ""))
    target_match = target and _contains_any(text, _date_forms(target))
    source_match = source and _contains_any(text, _date_forms(source))
    if target_match and source_match:
        return "mixed_includes_target"
    if target_match:
        return "exact_target"
    if source_match or re.search(r"20\d{2}\s*(?:年|Q[1-4]|季度|年报|中报|季报)", text, re.I):
        return "other_only"
    return "absent"


def _classify_c(units: list[dict[str, object]], scope: Mapping[str, object]) -> tuple[object, ...]:
    metric = str(scope.get("c_target_metric", ""))
    aliases = _METRIC_ALIASES.get(metric, (metric,))
    for index, unit in enumerate(units):
        group = _unit_group(units, index)
        text = _group_text(group)
        if not _entity_present(text, scope) or not _contains_any(text, aliases):
            continue
        period = _period_relation(text, scope)
        if period in {"exact_target", "mixed_includes_target"} and _has_number(text):
            return (
                "exact_metric",
                period,
                "realized_target_metric_value",
                (),
                None,
                group,
                "match",
                "na",
                "exact",
                "C_EXACT_TARGET",
                "DIRECT_TARGET",
            )
        if _contains_any(text, _C_FORECAST):
            return (
                "exact_metric",
                period,
                "target_metric_forecast",
                (),
                None,
                group,
                "match",
                "na",
                "exact",
                "C_FORECAST",
                "INDIRECT_INPUT",
            )
        if period == "other_only" and (_has_number(text) or _contains_any(text, _A1_DIRECTION)):
            return (
                "exact_metric",
                period,
                "target_metric_other_period",
                (),
                None,
                group,
                "match",
                "na",
                "exact",
                "C_OTHER_PERIOD",
                "INDIRECT_INPUT",
            )
        return (
            "exact_metric",
            period,
            "scope_mention_only",
            (),
            None,
            group,
            "match",
            "na",
            "exact",
            "C_TARGET_NO_VALUE",
            "NO_SIGNAL",
        )
    # A declared net-margin formula is a local same-period table-block rule.
    if metric == "net_margin":
        for index, unit in enumerate(units):
            group = _unit_group(units, index)
            text = _group_text(group)
            if (
                _entity_present(text, scope)
                and _contains_any(text, ("attributable_net_profit", "归母净利润", "净利润"))
                and _contains_any(text, ("operating_revenue", "营业收入", "营业总收入"))
                and _has_number(text)
            ):
                return (
                    "declared_component",
                    _period_relation(text, scope),
                    "declared_metric_component",
                    (),
                    "net_margin",
                    group,
                    "match",
                    "na",
                    "component",
                    "C_DECLARED_COMPONENT",
                    "INDIRECT_INPUT",
                )
    for index, unit in enumerate(units):
        group = _unit_group(units, index)
        text = _group_text(group)
        if _entity_present(text, scope) and _contains_any(text, _C_DRIVER) and _has_number(text):
            return (
                "explicit_driver",
                _period_relation(text, scope),
                "explicit_metric_driver",
                (),
                None,
                group,
                "match",
                "na",
                "other",
                "C_EXPLICIT_DRIVER",
                "INDIRECT_INPUT",
            )
    return (
        "mismatch",
        "absent",
        "none",
        (),
        None,
        (),
        "match",
        "na",
        "none",
        "C_UNRELATED",
        "NO_SIGNAL",
    )


def label_relevance_outcome(view: Mapping[str, object]) -> tuple[str, str, str]:
    """Return the deterministic pair and auditable v8 rationale."""

    decision = classify_v8(view)
    return decision.task_relevance, decision.outcome_revealing, decision.rationale()


__all__ = [
    "PERIOD_RELATIONS",
    "RUBRIC_VERSION",
    "SIGNAL_KINDS",
    "SCOPE_RELATIONS",
    "V8LabelDecision",
    "classify_v8",
    "derive_label_pair",
    "label_relevance_outcome",
    "segment_visible_view",
    "validate_decisive_units",
]
