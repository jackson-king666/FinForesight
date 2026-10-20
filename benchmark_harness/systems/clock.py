"""Declared experiment clocks for prompt-visible adapter time."""

from __future__ import annotations

from datetime import date
from typing import Any

from benchmark_harness.temporal.contracts import parse_temporal_date


class ClockPolicyError(ValueError):
    """Raised when an adapter tries to run without a declared experiment date."""


class DeclaredExperimentClock:
    """Immutable clock injected into prompt construction by the host."""

    def __init__(self, experiment_as_of: date | str | None) -> None:
        parsed = parse_temporal_date(
            experiment_as_of,
            field_name="experiment_as_of",
            allow_none=False,
        )
        if parsed is None:  # pragma: no cover - parse_temporal_date is fail-closed.
            raise ClockPolicyError("experiment_as_of is required")
        self._as_of = parsed

    @property
    def as_of(self) -> date:
        return self._as_of

    def today(self) -> date:
        """Return the declared date; never consult machine time."""

        return self._as_of

    def isoformat(self) -> str:
        return self._as_of.isoformat()

    def prompt_date(self) -> str:
        return self.isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "declared-experiment-clock/v1",
            "experiment_as_of": self.isoformat(),
            "machine_time_used_for_prompt": False,
        }


__all__ = ["ClockPolicyError", "DeclaredExperimentClock"]
