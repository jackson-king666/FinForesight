"""Narrow prompt-clock seam for disposable external workers.

The shim is intentionally data-only: a worker receives the declared date from
the host and calls ``prompt_date`` when constructing prompts.  It does not
patch process-global ``datetime`` and it has no network, credential, or
telemetry behavior.
"""

from __future__ import annotations

from benchmark_harness.systems.clock import DeclaredExperimentClock


def prompt_date(clock: DeclaredExperimentClock) -> str:
    """Return the only date value permitted in prompt-visible adapter text."""

    return clock.prompt_date()


__all__ = ["prompt_date"]
