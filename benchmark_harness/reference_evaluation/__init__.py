"""Frozen reference parser, evaluator, and summarizer implementation.

The public functions in this package are a behavioral snapshot of the
authoritative benchmark checkout named in :mod:`EVALUATOR_MANIFEST.json`.
They deliberately have no runtime dependency on that checkout.
"""

from .engine import (
    BundledEvaluationEngine,
    EvaluationEngine,
    evaluate_prediction,
    mock_response,
    parse_prediction,
    summarize_results,
)

__all__ = [
    "BundledEvaluationEngine",
    "EvaluationEngine",
    "evaluate_prediction",
    "mock_response",
    "parse_prediction",
    "summarize_results",
]
