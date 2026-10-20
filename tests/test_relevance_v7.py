"""Focused guards for the rubric-v6/task-scope-v2 pilot."""

from __future__ import annotations

from benchmark_harness.calibration.pilot import PILOT_QUERY_CLASSES
from benchmark_harness.calibration.relevance_v7 import segment_visible_view
from benchmark_harness.calibration.v7_pilot import V7_QUERY_CLASSES, _label_unit
from benchmark_harness.calibration.v7_scope import build_task_scope


def test_v7_query_classes_are_fresh_semantic_keys() -> None:
    assert V7_QUERY_CLASSES == tuple(f"v7_{query_class}" for query_class in PILOT_QUERY_CLASSES)
    assert set(V7_QUERY_CLASSES).isdisjoint(PILOT_QUERY_CLASSES)


def test_v7_scope_accepts_chinese_third_quarter_reporting_period() -> None:
    row = {
        "category": "B",
        "expected_output": {"direction": "up|down"},
        "prompt": "Predict the stock direction.",
        "seed": {
            "cutoff_date": "2024-10-19",
            "event_date": "2024-10-19",
            "event_description": "2024年三季度业绩：EPS实际为2.10，一致预期为2.00。",
            "event_subtype": "earnings",
            "stock_code": "ACME",
            "stock_name": "Acme",
        },
    }

    scope = build_task_scope(row)

    anchors = scope["b_match_anchors"]
    assert isinstance(anchors, dict)
    assert anchors["reporting_period"] == "2024 Q3"


def test_v7_label_unit_excludes_internal_evidence_text() -> None:
    unit = _label_unit(
        {
            "end_char": 10,
            "exact_text_sha256": "a" * 64,
            "field": "model_visible_content",
            "start_char": 0,
            "text": "hidden from raw-label schema",
        }
    )

    assert set(unit) == {"field", "start_char", "end_char", "exact_text_sha256"}


def test_v7_segmenter_preserves_decimal_sentences_headers_bullets_and_table_rows() -> None:
    value = (
        "2025-06-30\n"
        "恒瑞医药（600276）\n"
        "营业收入为208.04亿元，同比增长7.55%。\n"
        "EPS was 7.55%. Revenue was 208.04.\n"
        "- 每股收益为1.23元；维持增长\n"
        "| 项目 | 2025-06-30 | 208.04 |\n"
    )
    units = segment_visible_view({"model_visible_content": value})
    texts = [unit["text"] for unit in units]

    assert texts == [
        "2025-06-30",
        "恒瑞医药（600276）",
        "营业收入为208.04亿元，同比增长7.55%。",
        "EPS was 7.55%.",
        "Revenue was 208.04.",
        "- 每股收益为1.23元；维持增长",
        "| 项目 | 2025-06-30 | 208.04 |",
    ]
