"""Shared immutable Volc Torchlight request construction.

The agentic profile uses this provider request shape, and all search callers
keep their individual budgets and audit wrappers outside this module.  In
particular, a candidate can provide a query but never selects the time range.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from benchmark_harness.contracts import sha256_json, sha256_text
from benchmark_harness.search.cutoff import CutoffWindow

VOLC_SEARCH_HOST = "mercury.volcengineapi.com"
VOLC_SEARCH_URL = f"https://{VOLC_SEARCH_HOST}/"
VOLC_SEARCH_PATH = "/"
VOLC_SEARCH_ACTION = "WebSearch"
VOLC_SEARCH_VERSION = "2025-01-01"
VOLC_SEARCH_SERVICE = "volc_torchlight_api"
VOLC_SEARCH_REGION = "cn-beijing"

DEFAULT_VOLC_RESULT_COUNT = 10
DEFAULT_VOLC_LOOKBACK_YEARS = 5


@dataclass(frozen=True, slots=True)
class VolcSearchRequest:
    """One fully bounded Volc ``WebSearch`` request and its safe identifiers."""

    window: CutoffWindow
    start_date: str
    end_date: str
    query_sha256: str
    body: dict[str, Any]
    request_filter_sha256: str

    @property
    def time_range(self) -> str:
        """Return Volc's inclusive calendar-date interval."""

        return f"{self.start_date}..{self.end_date}"


def build_cutoff_volc_request(
    query: str,
    cutoff_date: str | None,
    *,
    count: int = DEFAULT_VOLC_RESULT_COUNT,
    lookback_years: int = DEFAULT_VOLC_LOOKBACK_YEARS,
    lookback_start: str | None = None,
) -> VolcSearchRequest:
    """Build the sole permitted Volc request payload for a seed cutoff.

    Volc's ``TimeRange`` is expressed as an inclusive pair of calendar days,
    unlike the exclusive timestamp passed to the previous provider.  The end
    day is therefore the exact seed ``cutoff_date``.  A caller may set an
    explicit ISO calendar-day lookback start, or configure a whole-year
    lookback; it may never select an end date after the seed cutoff.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("search_query_missing")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("search_result_limit_invalid")

    window = CutoffWindow.from_seed(cutoff_date)
    end = date.fromisoformat(window.cutoff_date)
    start = _resolve_lookback_start(
        end,
        lookback_years=lookback_years,
        lookback_start=lookback_start,
    )
    time_range = f"{start.isoformat()}..{end.isoformat()}"
    body: dict[str, Any] = {
        "Query": query,
        "SearchType": "web",
        "Count": count,
        "Filter": {
            "NeedContent": True,
            "NeedUrl": True,
        },
        "TimeRange": time_range,
    }
    return VolcSearchRequest(
        window=window,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        query_sha256=sha256_text(query),
        body=body,
        request_filter_sha256=sha256_json(body),
    )


def prepare_cutoff_volc_request(
    query: str,
    cutoff_date: str | None,
    *,
    count: int = DEFAULT_VOLC_RESULT_COUNT,
    lookback_years: int = DEFAULT_VOLC_LOOKBACK_YEARS,
    lookback_start: str | None = None,
) -> tuple[CutoffWindow, str, dict[str, Any], str]:
    """Return a compact tuple used by profile-specific Volc gateways."""

    request = build_cutoff_volc_request(
        query,
        cutoff_date,
        count=count,
        lookback_years=lookback_years,
        lookback_start=lookback_start,
    )
    return (
        request.window,
        request.query_sha256,
        request.body,
        request.request_filter_sha256,
    )


def _resolve_lookback_start(
    end: date,
    *,
    lookback_years: int,
    lookback_start: str | None,
) -> date:
    if lookback_start is not None:
        if not isinstance(lookback_start, str):
            raise ValueError("search_lookback_start_invalid")
        try:
            start = date.fromisoformat(lookback_start)
        except ValueError as exc:
            raise ValueError("search_lookback_start_invalid") from exc
        if start.isoformat() != lookback_start:
            raise ValueError("search_lookback_start_invalid")
        if start > end:
            raise ValueError("search_lookback_after_cutoff")
        return start

    if (
        not isinstance(lookback_years, int)
        or isinstance(lookback_years, bool)
        or lookback_years < 0
    ):
        raise ValueError("search_lookback_years_invalid")
    return _subtract_years(end, lookback_years)


def _subtract_years(value: date, years: int) -> date:
    """Subtract calendar years while making leap-day lookbacks deterministic."""

    target_year = value.year - years
    try:
        return value.replace(year=target_year)
    except ValueError:
        # The only date.replace year failure here is February 29 landing in a
        # non-leap year.  February 28 keeps the interval bounded and stable.
        return value.replace(year=target_year, day=28)
