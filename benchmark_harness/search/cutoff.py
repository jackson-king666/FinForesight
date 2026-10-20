"""Calendar-date cutoff parsing shared by search requests and policy checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta


class CutoffWindowError(ValueError):
    """Raised when an upstream seed cutoff is absent or not an ISO calendar date."""


class BEventDateError(ValueError):
    """Raised when a B seed cannot safely bound search by its event date."""


@dataclass(frozen=True, slots=True)
class CutoffWindow:
    """An inclusive calendar-day cutoff represented as an exclusive UTC bound."""

    cutoff_date: str
    exclusive_bound: str

    @classmethod
    def from_seed(cls, cutoff_date: str | None) -> CutoffWindow:
        """Accept only ``YYYY-MM-DD`` values from ``seed.cutoff_date``.

        The benchmark promises to include the entire stated calendar day, so
        requests use the next day at 00:00:00Z as their exclusive upper bound.
        """

        if not isinstance(cutoff_date, str):
            raise CutoffWindowError("cutoff_date_missing")
        try:
            parsed = date.fromisoformat(cutoff_date)
        except ValueError as exc:
            raise CutoffWindowError("cutoff_date_invalid") from exc
        if parsed.isoformat() != cutoff_date:
            raise CutoffWindowError("cutoff_date_invalid")

        exclusive = datetime.combine(parsed + timedelta(days=1), time.min, tzinfo=UTC)
        return cls(cutoff_date=cutoff_date, exclusive_bound=_utc_isoformat(exclusive))

    @property
    def exclusive_datetime(self) -> datetime:
        return _parse_utc_timestamp(self.exclusive_bound)

    def contains_published_at(self, value: str | None) -> tuple[bool, str | None]:
        """Return whether a legacy exclusive-bound timestamp is cutoff compliant.

        A date-only source is interpreted as the beginning of that UTC day;
        that is enough to determine whether it lies on/before the cutoff day.
        Unknown or malformed publication values are rejected rather than
        assumed compliant.
        """

        if not isinstance(value, str) or not value.strip():
            return False, "missing_published_date"
        try:
            published = _parse_published_timestamp(value)
        except ValueError:
            return False, "invalid_published_date"
        if published >= self.exclusive_datetime:
            return False, "published_after_cutoff"
        return True, None


def _parse_published_timestamp(value: str) -> datetime:
    raw = value.strip()
    if len(raw) == 10:
        parsed_date = date.fromisoformat(raw)
        return datetime.combine(parsed_date, time.min, tzinfo=UTC)
    return _parse_utc_timestamp(raw)


def _parse_utc_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        # An exclusive-bound publication timestamp must be unambiguous.
        raise ValueError("timestamp is missing a timezone")
    return parsed.astimezone(UTC)


def _utc_isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def validate_b_cutoff_not_after_event(
    cutoff_date: str | None,
    event_date: str | None,
) -> CutoffWindow:
    """Return B's cutoff window only when it does not postdate its event.

    B's task truth is event-bound, but the search request must always use the
    seed cutoff rather than silently substituting ``event_date``.  Distinct
    error codes let the dataset source/runner mark the task non-comparable
    before a model or network call occurs.
    """

    window = CutoffWindow.from_seed(cutoff_date)
    if not isinstance(event_date, str):
        raise BEventDateError("b_event_date_missing")
    try:
        parsed_event = date.fromisoformat(event_date)
    except ValueError as exc:
        raise BEventDateError("b_event_date_invalid") from exc
    if parsed_event.isoformat() != event_date:
        raise BEventDateError("b_event_date_invalid")
    if window.cutoff_date > event_date:
        raise BEventDateError("b_cutoff_after_event_date")
    return window
