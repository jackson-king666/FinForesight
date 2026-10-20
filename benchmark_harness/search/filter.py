"""Fail-closed, lightweight temporal filtering for Volc search candidates.

This module deliberately operates only on Volc's returned title, snippet,
content, and publication value.  It does not fetch pages or expose browser
credentials.  Ambiguous documents can be passed to an injected supervisor-side
LLM reviewer; absent a keep verdict, they are rejected by default.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal, Protocol, TypeAlias

from benchmark_harness.contracts import sha256_text
from benchmark_harness.search.cutoff import CutoffWindow

RuleVerdict: TypeAlias = Literal["keep", "reject", "ambiguous"]
LlmVerdict: TypeAlias = Literal["keep", "reject", "uncertain"]

DEFAULT_MAX_FILTER_TEXT_CHARS = 8_000


@dataclass(frozen=True, slots=True)
class VolcSearchCandidate:
    """The model-visible subset of one Volc ``WebResults`` document."""

    url: str | None
    title: str | None = None
    snippet: str | None = None
    content: str | None = None
    volc_publish_time: str | datetime | int | float | None = None
    site_name: str | None = None
    auth_info_level: int | None = None

    @classmethod
    def from_volc_result(cls, value: Mapping[str, Any]) -> VolcSearchCandidate:
        """Normalize Volc's PascalCase result fields without retaining raw data."""

        return cls(
            url=_optional_text(value.get("Url")),
            title=_optional_text(value.get("Title")),
            snippet=_optional_text(value.get("Snippet")),
            content=_optional_text(value.get("Content")),
            volc_publish_time=_optional_publish_time(value.get("PublishTime")),
            site_name=_optional_text(value.get("SiteName")),
            auth_info_level=_optional_int(value.get("AuthInfoLevel")),
        )

    def as_model_payload(self) -> dict[str, str | None]:
        """Return only fields eligible to be shown to a framework-native tool."""

        return {
            "url": self.url,
            "title": self.title,
            "content": self.content or self.snippet,
            "snippet": self.snippet,
            "published_at": _publish_time_text(self.volc_publish_time),
        }


@dataclass(frozen=True, slots=True)
class RuleFilterDecision:
    """The deterministic temporal decision before any LLM call."""

    candidate: VolcSearchCandidate
    verdict: RuleVerdict
    reason: str
    signals: tuple[str, ...] = ()
    parsed_publish_time: str | None = None


@dataclass(frozen=True, slots=True)
class LlmFilterRequest:
    """Small, serializable input an injected temporal reviewer receives."""

    candidate: VolcSearchCandidate
    cutoff_date: str
    rule_signals: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return the content supplied to an LLM, excluding any transport secrets."""

        return {
            "cutoff_date": self.cutoff_date,
            "url": self.candidate.url,
            "title": self.candidate.title,
            "snippet": self.candidate.snippet,
            "content": self.candidate.content,
            "volc_publish_time": _publish_time_text(self.candidate.volc_publish_time),
            "rule_signals": list(self.rule_signals),
        }


@dataclass(frozen=True, slots=True)
class LlmTemporalVerdict:
    """Normalized LLM result; only ``keep`` permits ambiguous evidence."""

    verdict: LlmVerdict
    reason: str = ""


class LlmTemporalFilter(Protocol):
    """Supervisor-side temporal review boundary.

    Implementations may use the resolved Aliyun model client, while tests can
    provide a plain callable returning ``LlmTemporalVerdict`` or a small mapping.
    """

    def __call__(
        self, request: LlmFilterRequest
    ) -> LlmTemporalVerdict | Mapping[str, Any] | str: ...


@dataclass(frozen=True, slots=True)
class PostSearchFilterConfig:
    """Operational policy for ambiguous documents and bounded text inspection."""

    fail_closed_ambiguous: bool = True
    max_text_chars: int = DEFAULT_MAX_FILTER_TEXT_CHARS

    def __post_init__(self) -> None:
        if not isinstance(self.max_text_chars, int) or self.max_text_chars < 1:
            raise ValueError("max_text_chars must be a positive integer")


@dataclass(frozen=True, slots=True)
class CandidateFilterDecision:
    """The terminal keep/reject decision used for evidence and audit output."""

    candidate: VolcSearchCandidate
    accepted: bool
    reason: str
    rule_verdict: RuleVerdict
    rule_signals: tuple[str, ...] = ()
    llm_verdict: LlmVerdict | None = None
    llm_reason: str | None = None

    def as_audit_dict(self, *, hash_only: bool = False) -> dict[str, Any]:
        """Return content-safe audit data for a search or filter JSONL sink."""

        candidate = self.candidate
        row: dict[str, Any] = {
            "provider": "volc",
            "accepted": self.accepted,
            "reason": self.reason,
            "rule_verdict": self.rule_verdict,
            "rule_signals": list(self.rule_signals),
            "llm_verdict": self.llm_verdict,
            "llm_reason": self.llm_reason,
            "volc_publish_time": _publish_time_text(candidate.volc_publish_time),
            "content_sha256": sha256_text(candidate.content) if candidate.content else None,
            "snippet_sha256": sha256_text(candidate.snippet) if candidate.snippet else None,
        }
        if hash_only:
            row.update(
                {
                    "url_sha256": sha256_text(candidate.url) if candidate.url else None,
                    "title_sha256": sha256_text(candidate.title) if candidate.title else None,
                }
            )
        else:
            row.update({"url": candidate.url, "title": candidate.title})
        return row


@dataclass(frozen=True, slots=True)
class PostSearchFilterResult:
    """All candidate decisions plus the only documents allowed to reach an agent."""

    kept: tuple[VolcSearchCandidate, ...]
    decisions: tuple[CandidateFilterDecision, ...]

    @property
    def rejected(self) -> tuple[CandidateFilterDecision, ...]:
        return tuple(decision for decision in self.decisions if not decision.accepted)

    def model_payloads(self) -> list[dict[str, str | None]]:
        return [candidate.as_model_payload() for candidate in self.kept]


def rule_filter_volc_candidate(
    candidate: VolcSearchCandidate | Mapping[str, Any],
    cutoff_date: str | None,
    *,
    max_text_chars: int = DEFAULT_MAX_FILTER_TEXT_CHARS,
) -> RuleFilterDecision:
    """Classify a Volc candidate as deterministic keep, reject, or ambiguity.

    A late provider ``PublishTime`` is a hard reject.  Missing or malformed
    provider dates and future-looking title/snippet/content markers are
    ambiguous: a configured LLM reviewer can make a tightly scoped judgment,
    otherwise the terminal filter fails closed.
    """

    if not isinstance(max_text_chars, int) or max_text_chars < 1:
        raise ValueError("max_text_chars must be a positive integer")
    normalized = _coerce_candidate(candidate)
    window = CutoffWindow.from_seed(cutoff_date)
    cutoff = date.fromisoformat(window.cutoff_date)

    if not normalized.url:
        return RuleFilterDecision(
            candidate=normalized,
            verdict="reject",
            reason="missing_url",
        )
    if not _candidate_text(normalized):
        return RuleFilterDecision(
            candidate=normalized,
            verdict="reject",
            reason="empty_candidate_text",
        )

    signals: list[str] = []
    raw_publish_time = normalized.volc_publish_time
    parsed_publish_time = parse_volc_publish_time(raw_publish_time)
    if _is_missing_publish_time(raw_publish_time):
        signals.append("missing_volc_publish_time")
    elif parsed_publish_time is None:
        signals.append("invalid_volc_publish_time")
    elif parsed_publish_time.date() > cutoff:
        return RuleFilterDecision(
            candidate=normalized,
            verdict="reject",
            reason="volc_publish_time_after_cutoff",
            signals=("volc_publish_time_after_cutoff",),
            parsed_publish_time=_utc_isoformat(parsed_publish_time),
        )

    signals.extend(
        detect_future_temporal_markers(
            _candidate_text(normalized, max_text_chars=max_text_chars),
            cutoff,
        )
    )
    unique_signals = tuple(dict.fromkeys(signals))
    if unique_signals:
        return RuleFilterDecision(
            candidate=normalized,
            verdict="ambiguous",
            reason="temporal_judgment_required",
            signals=unique_signals,
            parsed_publish_time=(
                _utc_isoformat(parsed_publish_time) if parsed_publish_time is not None else None
            ),
        )
    return RuleFilterDecision(
        candidate=normalized,
        verdict="keep",
        reason="rule_keep",
        parsed_publish_time=_utc_isoformat(parsed_publish_time) if parsed_publish_time else None,
    )


def filter_volc_candidates(
    candidates: Iterable[VolcSearchCandidate | Mapping[str, Any]],
    cutoff_date: str | None,
    *,
    llm_filter: LlmTemporalFilter | None = None,
    config: PostSearchFilterConfig | None = None,
) -> PostSearchFilterResult:
    """Apply rule filtering, then optional LLM review, before returning evidence.

    The function is deliberately deterministic unless an injected reviewer is
    needed.  LLM errors, malformed answers, and uncertain verdicts reject an
    ambiguous document by default, so no uncertain candidate reaches a model.
    """

    policy = config or PostSearchFilterConfig()
    decisions = tuple(
        _terminal_decision(
            rule_filter_volc_candidate(
                candidate,
                cutoff_date,
                max_text_chars=policy.max_text_chars,
            ),
            cutoff_date=cutoff_date,
            llm_filter=llm_filter,
            fail_closed_ambiguous=policy.fail_closed_ambiguous,
        )
        for candidate in candidates
    )
    return PostSearchFilterResult(
        kept=tuple(decision.candidate for decision in decisions if decision.accepted),
        decisions=decisions,
    )


def detect_future_temporal_markers(text: str, cutoff_date: date | str) -> tuple[str, ...]:
    """Return safe audit signals for future date/year markers in visible text."""

    cutoff = _coerce_calendar_date(cutoff_date)
    markers: list[str] = []
    for found in _extract_explicit_dates(text):
        if found > cutoff:
            markers.append(f"future_date_marker:{found.isoformat()}")
    for match in re.finditer(r"(?<!\d)((?:19|20|21)\d{2})(?!\d)", text):
        year = int(match.group(1))
        if year > cutoff.year:
            markers.append(f"future_year_marker:{year}")
    return tuple(dict.fromkeys(markers))


def parse_volc_publish_time(value: object) -> datetime | None:
    """Parse common Volc publication values with no permissive third-party parser."""

    if value is None:
        return None
    if isinstance(value, datetime):
        return _as_utc(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return _from_epoch(value)
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.isdigit():
        return _from_epoch(int(raw))
    try:
        if len(raw) == 10:
            parsed_date = date.fromisoformat(raw)
            return datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=UTC)
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return _parse_non_iso_date(raw)
    return _as_utc(parsed)


def _terminal_decision(
    rule: RuleFilterDecision,
    *,
    cutoff_date: str | None,
    llm_filter: LlmTemporalFilter | None,
    fail_closed_ambiguous: bool,
) -> CandidateFilterDecision:
    if rule.verdict == "keep":
        return CandidateFilterDecision(
            candidate=rule.candidate,
            accepted=True,
            reason=rule.reason,
            rule_verdict=rule.verdict,
            rule_signals=rule.signals,
        )
    if rule.verdict == "reject":
        return CandidateFilterDecision(
            candidate=rule.candidate,
            accepted=False,
            reason=rule.reason,
            rule_verdict=rule.verdict,
            rule_signals=rule.signals,
        )

    if llm_filter is None:
        return _ambiguous_without_llm(rule, fail_closed=fail_closed_ambiguous)
    try:
        response = llm_filter(
            LlmFilterRequest(
                candidate=rule.candidate,
                cutoff_date=CutoffWindow.from_seed(cutoff_date).cutoff_date,
                rule_signals=rule.signals,
            )
        )
        verdict = _normalize_llm_verdict(response)
    except Exception as exc:  # The reviewer is an external service boundary.
        return _llm_failure_decision(
            rule,
            reason="llm_filter_error",
            detail=type(exc).__name__,
            fail_closed=fail_closed_ambiguous,
        )

    if verdict.verdict == "keep":
        return CandidateFilterDecision(
            candidate=rule.candidate,
            accepted=True,
            reason="llm_keep",
            rule_verdict=rule.verdict,
            rule_signals=rule.signals,
            llm_verdict=verdict.verdict,
            llm_reason=_bounded_reason(verdict.reason),
        )
    if verdict.verdict == "reject":
        return CandidateFilterDecision(
            candidate=rule.candidate,
            accepted=False,
            reason="llm_reject",
            rule_verdict=rule.verdict,
            rule_signals=rule.signals,
            llm_verdict=verdict.verdict,
            llm_reason=_bounded_reason(verdict.reason),
        )
    return _llm_failure_decision(
        rule,
        reason="llm_uncertain",
        detail=verdict.reason,
        fail_closed=fail_closed_ambiguous,
        llm_verdict=verdict.verdict,
    )


def _ambiguous_without_llm(
    rule: RuleFilterDecision,
    *,
    fail_closed: bool,
) -> CandidateFilterDecision:
    return CandidateFilterDecision(
        candidate=rule.candidate,
        accepted=not fail_closed,
        reason="ambiguous_without_llm" if fail_closed else "ambiguous_fail_open",
        rule_verdict=rule.verdict,
        rule_signals=rule.signals,
    )


def _llm_failure_decision(
    rule: RuleFilterDecision,
    *,
    reason: str,
    detail: str,
    fail_closed: bool,
    llm_verdict: LlmVerdict | None = None,
) -> CandidateFilterDecision:
    return CandidateFilterDecision(
        candidate=rule.candidate,
        accepted=not fail_closed,
        reason=reason if fail_closed else f"{reason}_fail_open",
        rule_verdict=rule.verdict,
        rule_signals=rule.signals,
        llm_verdict=llm_verdict,
        llm_reason=_bounded_reason(detail),
    )


def _normalize_llm_verdict(
    value: LlmTemporalVerdict | Mapping[str, Any] | str,
) -> LlmTemporalVerdict:
    if isinstance(value, LlmTemporalVerdict):
        return value
    if isinstance(value, str):
        raw_verdict = value
        reason = ""
    elif isinstance(value, Mapping):
        raw_verdict = value.get("verdict", value.get("decision", "uncertain"))
        reason = str(value.get("reason", ""))
    else:
        raise TypeError("llm_filter_return_invalid")
    normalized = str(raw_verdict).strip().lower()
    aliases: dict[str, LlmVerdict] = {
        "keep": "keep",
        "accept": "keep",
        "allowed": "keep",
        "no_leak": "keep",
        "reject": "reject",
        "drop": "reject",
        "leak": "reject",
        "uncertain": "uncertain",
        "unknown": "uncertain",
        "error": "uncertain",
    }
    return LlmTemporalVerdict(verdict=aliases.get(normalized, "uncertain"), reason=reason)


def _coerce_candidate(value: VolcSearchCandidate | Mapping[str, Any]) -> VolcSearchCandidate:
    if isinstance(value, VolcSearchCandidate):
        return value
    if isinstance(value, Mapping):
        if any(key in value for key in ("Url", "Title", "Snippet", "Content", "PublishTime")):
            return VolcSearchCandidate.from_volc_result(value)
        return VolcSearchCandidate(
            url=_optional_text(value.get("url")),
            title=_optional_text(value.get("title")),
            snippet=_optional_text(value.get("snippet")),
            content=_optional_text(value.get("content")),
            volc_publish_time=_optional_publish_time(
                value.get("volc_publish_time", value.get("published_at"))
            ),
            site_name=_optional_text(value.get("site_name")),
            auth_info_level=_optional_int(value.get("auth_info_level")),
        )
    raise TypeError("candidate must be VolcSearchCandidate or a mapping")


def _candidate_text(candidate: VolcSearchCandidate, *, max_text_chars: int | None = None) -> str:
    text = "\n".join(
        part.strip()
        for part in (candidate.title, candidate.snippet, candidate.content)
        if isinstance(part, str) and part.strip()
    )
    if max_text_chars is None or len(text) <= max_text_chars:
        return text
    # Scan both the beginning and end.  Search snippets are small, but Volc
    # content can be long and a footer-like leakage marker may be at the tail.
    head = max_text_chars // 2
    tail = max_text_chars - head
    return f"{text[:head]}\n[...truncated...]\n{text[-tail:]}"


def _extract_explicit_dates(text: str) -> tuple[date, ...]:
    dates: list[date] = []
    patterns = (
        r"(?<!\d)((?:19|20|21)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)",
        r"(?<!\d)((?:19|20|21)\d{2})年(\d{1,2})月(\d{1,2})日",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            found = _date_from_parts(*match.groups())
            if found is not None:
                dates.append(found)
    return tuple(dict.fromkeys(dates))


def _coerce_calendar_date(value: date | str) -> date:
    if isinstance(value, datetime):
        return _as_utc(value).date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("cutoff_date_invalid") from exc
        if parsed.isoformat() != value:
            raise ValueError("cutoff_date_invalid")
        return parsed
    raise ValueError("cutoff_date_invalid")


def _parse_non_iso_date(raw: str) -> datetime | None:
    match = re.fullmatch(r"((?:19|20|21)\d{2})/(\d{1,2})/(\d{1,2})", raw)
    if match is None:
        match = re.fullmatch(r"((?:19|20|21)\d{2})年(\d{1,2})月(\d{1,2})日", raw)
    if match is None:
        return None
    parsed = _date_from_parts(*match.groups())
    if parsed is None:
        return None
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)


def _date_from_parts(year: str, month: str, day: str) -> date | None:
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def _from_epoch(value: int | float) -> datetime | None:
    timestamp = float(value)
    if abs(timestamp) > 10_000_000_000:
        timestamp /= 1_000
    try:
        return datetime.fromtimestamp(timestamp, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _publish_time_text(value: str | datetime | int | float | None) -> str | None:
    if isinstance(value, datetime):
        return _utc_isoformat(value)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return str(value)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _is_missing_publish_time(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_publish_time(value: object) -> str | datetime | int | float | None:
    if isinstance(value, str | datetime):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value
    return None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _utc_isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _bounded_reason(value: str) -> str:
    return value.strip()[:500] if isinstance(value, str) and value.strip() else "no_reason_returned"
