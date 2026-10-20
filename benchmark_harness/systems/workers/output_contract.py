"""Task-bound output contracts for external research workers.

The contract is derived only from the public task identifier. It deliberately
contains no record fields, expected answers, scores, or evaluator state.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

# These strings were used by the earlier live-path fixture shims. Keep the
# guard narrow and explicit: ordinary test URLs and task text remain usable,
# but a known fixture answer cannot cross a live broker/search boundary or be
# emitted as a prediction.
FIXTURE_SENTINELS = (
    "fixture.example",
    "bounded benchmark question",
    "bounded pinned finding",
    "bounded evidence query",
    "bounded evidence",
    "bounded follow-up question",
    "bounded next question",
    "bounded question",
    "proceed with the bounded question.",
    "pinned GPT-Researcher nested research context",
)


def reject_fixture_sentinel(value: Any, *, field: str) -> None:
    """Fail closed if a known fixture sentinel reaches a live boundary."""

    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            text = str(value)
    lowered = text.casefold()
    if any(sentinel.casefold() in lowered for sentinel in FIXTURE_SENTINELS):
        raise RuntimeError(f"live_fixture_sentinel_detected:{field}")


def task_category(task_id: str) -> str:
    """Return the frozen evaluator category encoded in a task id."""

    normalized = task_id.strip().upper()
    if normalized.startswith("A1-"):
        return "A1"
    if normalized.startswith("A2"):
        return "A2"
    for category in ("B", "C", "D", "E"):
        if normalized.startswith(category + "-"):
            return category
    raise ValueError("task_category_missing")


def task_output_contract(task_id: str) -> str:
    """Describe the exact answer shape without supplying task truth."""

    contracts = {
        "A1": (
            'Return one JSON object with numeric "bull", "base", and "bear"; '
            '"reversion_horizon" must be exactly one of '
            '"1-4周", "1-3个月", "3-6个月", "6-12个月", "超过12个月".'
        ),
        "A2": (
            "Return one JSON array of strings containing the requested ranking, "
            "in rank order, with no wrapper object."
        ),
        "B": (
            'Return one JSON object with "direction" equal to "up" or "down" '
            'and numeric "probability_up" between 0 and 1.'
        ),
        "C": (
            'Return one JSON object with numeric "predicted_value"; include '
            'string "metric" only when the task asks for it.'
        ),
        "D": (
            'Return one JSON object with "direction" equal to "positive" or '
            '"negative" and, when useful, a string "reasoning".'
        ),
        "E": (
            'Return one JSON object with string "formula_used", numeric '
            '"answer", and string "unit".'
        ),
    }
    return contracts[task_category(task_id)]


def task_bound_answer_messages(
    task_id: str,
    task_prompt: str,
    research_context: str,
    citations: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Build the final model request bound to the original task and evidence."""

    citation_json = json.dumps(list(citations), ensure_ascii=False, sort_keys=True)
    context = research_context.strip() or "No research evidence was returned."
    try:
        contract = task_output_contract(task_id)
    except ValueError:
        # Legacy fixture bindings use a synthetic task id and carry their own
        # response contract in the task prompt.
        contract = "Follow the output format requested by the original task."
    return [
        {
            "role": "system",
            "content": (
                "You are the final answer stage for one benchmark task. Answer the "
                "original task exactly; do not answer a substitute research topic. "
                "Use only the original task and the supplied research evidence. "
                "Return only the requested JSON value, with no markdown or prose. "
                "Do not include scores, ground truth, or evaluator commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task id: {task_id}\n"
                f"Original task:\n{task_prompt}\n\n"
                f"Output contract:\n{contract}\n\n"
                f"Receipt-backed citation URLs (use only these when applicable): "
                f"{citation_json}\n\n"
                f"Research evidence:\n{context}"
            ),
        },
    ]


__all__ = [
    "FIXTURE_SENTINELS",
    "reject_fixture_sentinel",
    "task_bound_answer_messages",
    "task_category",
    "task_output_contract",
]
