"""Focused tests for the additive full-run human leak-review contract."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from benchmark_harness.calibration.leak_review import (
    LEAK_REVIEW_LABEL_SCHEMA,
    LeakReviewError,
    ReviewerIdentity,
    ReviewFrame,
    agreement_gate,
    join_sealed_labels,
    validate_human_labels,
)


def _frame() -> ReviewFrame:
    return ReviewFrame(
        rubric_sha256="a" * 64,
        unit_of_analysis="candidate_receipt",
        dedup_rule="annotation_view_sha256",
        task_scope=("A1",),
        collector=ReviewerIdentity("collector"),
        reviewer_a=ReviewerIdentity("reviewer-a"),
        reviewer_b=ReviewerIdentity("reviewer-b"),
        adjudicator_c=ReviewerIdentity("adjudicator-c"),
        sampling_policy={"kind": "review_all"},
        frozen_at=datetime(2026, 9, 7, tzinfo=UTC),
    )


def _views(count: int = 20) -> list[dict[str, object]]:
    return [
        {
            "item_id": f"item-{index:03d}",
            "annotation_view_sha256": f"{index + 1:064x}",
            "category": "A1",
        }
        for index in range(count)
    ]


def _labels(
    frame: ReviewFrame, views: list[dict[str, object]], reviewer_id: str
) -> list[dict[str, object]]:
    return [
        {
            "schema_version": LEAK_REVIEW_LABEL_SCHEMA,
            "frame_sha256": frame.frame_sha256,
            "rubric_sha256": frame.rubric_sha256,
            "item_id": view["item_id"],
            "annotation_view_sha256": view["annotation_view_sha256"],
            "reviewer_id": reviewer_id,
            "reviewer_type": "human",
            "labeled_at": "2026-09-07T00:00:01Z",
            "task_relevance": "relevant" if index < 10 else "not_relevant",
            "outcome_revealing": "direct" if index < 10 else "no",
            "temporal_status": "post_cutoff" if index < 10 else "pre_cutoff",
            "supporting_span": "A bounded visible sentence.",
            "rationale": "The span supports the fixed label.",
        }
        for index, view in enumerate(views)
    ]


def test_human_labels_bind_frame_view_and_timestamp() -> None:
    frame = _frame()
    views = _views()
    labels = _labels(frame, views, "reviewer-a")
    validate_human_labels(labels, frame=frame, views=views)

    stale = [dict(labels[0], annotation_view_sha256="f" * 64)]
    with pytest.raises(LeakReviewError, match="stale annotation_view_sha256"):
        validate_human_labels(stale, frame=frame, views=views)

    unsigned = [dict(labels[0], frame_sha256="f" * 64)]
    with pytest.raises(LeakReviewError, match="frozen frame"):
        validate_human_labels(unsigned, frame=frame, views=views)


def test_raw_ab_gate_is_strict_and_adjudication_cannot_replace_it() -> None:
    frame = _frame()
    views = _views()
    labels_a = _labels(frame, views, "reviewer-a")
    labels_b = _labels(frame, views, "reviewer-b")
    gate = agreement_gate(labels_a, labels_b, frame=frame)
    assert gate["passed"] is True
    assert gate["fields"]["task_relevance"]["cohen_kappa"] == pytest.approx(1.0)

    degenerate_a = [dict(row, task_relevance="relevant") for row in labels_a]
    degenerate_b = [dict(row, task_relevance="relevant") for row in labels_b]
    failed = agreement_gate(degenerate_a, degenerate_b, frame=frame)
    assert failed["fields"]["task_relevance"]["passed"] is False
    assert "kappa_undefined" in failed["fields"]["task_relevance"]["failure_reasons"]


def test_join_requires_sealed_labels_and_preserves_unresolved_bounds() -> None:
    frame = _frame()
    receipts = [
        {"receipt_id": "receipt-1", "item_id": "item-1", "model_visible": True},
        {"receipt_id": "receipt-2", "item_id": "item-2", "model_visible": True},
    ]
    resolutions = [
        {
            "item_id": "item-1",
            "resolution_status": "resolved_initial",
            "resolved_labels": {
                "temporal_status": "post_cutoff",
                "task_relevance": "relevant",
                "outcome_revealing": "direct",
            },
        },
        {
            "item_id": "item-2",
            "resolution_status": "unresolved",
            "resolved_labels": {"temporal_status": "indeterminate"},
        },
    ]
    joined = join_sealed_labels(receipts, resolutions, frame=frame)
    assert joined["counts"]["leaked_direct"] == 1
    assert joined["sensitivity"]["direct_only"]["bounds"]["worst_case"] == 2
