"""Focused, synthetic guards for the immutable v8 classifier boundary."""

from __future__ import annotations

from benchmark_harness.calibration.relevance_v8 import derive_label_pair, segment_visible_view


def test_v8_segmenter_preserves_decimals_and_atomic_rows() -> None:
    text = (
        "2025-06-30\n"
        "Revenue was 208.04. EPS was 7.55%.\n"
        "- Net profit grew 8.2%; outlook remains positive\n"
        "| metric | value |\n"
    )
    units = segment_visible_view({"model_visible_content": text})
    assert [unit["text"] for unit in units] == [
        "2025-06-30",
        "Revenue was 208.04.",
        "EPS was 7.55%.",
        "- Net profit grew 8.2%; outlook remains positive",
        "| metric | value |",
    ]


def test_v8_pair_derivation_keeps_relevance_and_outcome_separate() -> None:
    assert derive_label_pair("exact_event", "exact_target", "realized_target_metric_value") == (
        "relevant",
        "direct",
    )
    assert derive_label_pair("valuation_input", "na", "valuation_input") == (
        "relevant",
        "indirect",
    )
    assert derive_label_pair("mismatch", "absent", "none") == ("not_relevant", "no")
