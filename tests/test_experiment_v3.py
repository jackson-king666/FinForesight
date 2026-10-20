"""Score-blind experiment/v3 lock, gate, metric, and tamper contracts."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from benchmark_harness.cli import main
from benchmark_harness.dataset_sources.v3 import V3DatasetSource
from benchmark_harness.experiment.v3 import (
    CURRENT_DATA_INELIGIBLE,
    EXPLORATORY_ONLY,
    V3_ARTIFACT_FILENAMES,
    LockError,
    MatchingPolicy,
    MetricPolicy,
    ScoreBlindError,
    _eligibility_reasons,
    aggregate_trial_brier,
    bootstrap_seed,
    brier_score,
    brier_skill_score,
    build_cohort,
    build_dataset_lock,
    load_score_blind_records,
    operational_brier_score,
    paired_event_cluster_bootstrap,
    reliability_ece,
    reliability_report,
    validate_v3_artifacts,
    write_v3_artifacts,
)


def _row(task_id: str, band: str, *, score: object = None) -> dict[str, object]:
    row: dict[str, object] = {
        "task_id": task_id,
        "category": "B",
        "variant": "earnings",
        "time_band": band,
        "status": "ready",
        "stratum": "B|earnings|US|direction|template-1|issuer-1",
        "eligibility": {"eligible": True},
        "covariates": {"market_size": 1.0, "market": "US"},
        "seed": {"cutoff_date": "2024-01-01"},
        "official_temporal_eligible": True,
        "review_status": "reviewed",
        "outcome_available_at_source": "observed_current",
        "metadata": {
            "source_url": "https://example.test/source",
            "source_hash": "a" * 64,
        },
        "prompt": "score-blind fixture",
    }
    if score is not None:
        row["score"] = score
    return row


def test_dataset_lock_is_deterministic_and_detects_task_drift() -> None:
    rows = [_row("task-2", "T2"), _row("task-1", "T1")]
    first = build_dataset_lock(rows, expected_record_count=2)
    second = build_dataset_lock(list(reversed(rows)), expected_record_count=2)

    assert first.to_dict() == second.to_dict()
    assert first.population_manifest_sha256
    rows[0]["metadata"] = {"source_url": "https://example.test/changed"}
    with pytest.raises(LockError, match="drift"):
        first.verify(rows)


def test_dataset_lock_redacts_nested_evaluator_source_fields() -> None:
    first_row = _row("task-1", "T1")
    first_row["metadata"] = {
        "source_url": "https://example.test/source",
        "ground_truth": {"answer": "private-a", "nested": {"gold": 1}},
    }
    second_row = json.loads(json.dumps(first_row))
    second_row["metadata"]["ground_truth"] = {
        "answer": "private-b",
        "nested": {"gold": 999},
    }
    assert (
        build_dataset_lock([first_row], expected_record_count=1).to_dict()
        == build_dataset_lock([second_row], expected_record_count=1).to_dict()
    )


def test_matching_and_metric_policy_fields_are_immutable() -> None:
    with pytest.raises(LockError, match="matching strata"):
        MatchingPolicy(exact_strata=("category",))
    with pytest.raises(LockError, match="metric policy"):
        MetricPolicy(primary_baseline=0.4)


def test_score_fence_rejects_nested_outputs_and_result_paths(tmp_path: Path) -> None:
    with pytest.raises(ScoreBlindError, match="forbidden"):
        build_dataset_lock([_row("task-1", "T1", score=0.1)], expected_record_count=1)

    result_path = tmp_path / "results" / "attempts.jsonl"
    result_path.parent.mkdir()
    result_path.write_text(json.dumps(_row("task-1", "T1")) + "\n", encoding="utf-8")
    with pytest.raises(ScoreBlindError, match="path"):
        load_score_blind_records(result_path)


def test_current_gate_and_append_only_artifact_bundle(tmp_path: Path) -> None:
    audit = build_cohort(
        [_row("task-1", "T1"), _row("task-2", "T2")],
        expected_record_count=2,
    )
    assert audit.status == CURRENT_DATA_INELIGIBLE
    assert audit.b_status == EXPLORATORY_ONLY
    paths = write_v3_artifacts(audit, tmp_path)
    assert tuple(paths) == V3_ARTIFACT_FILENAMES
    assert validate_v3_artifacts(tmp_path)["status"] == "valid"
    write_v3_artifacts(audit, tmp_path)

    covariates = (tmp_path / "covariates.jsonl").read_text(encoding="utf-8")
    (tmp_path / "covariates.jsonl").write_text(
        covariates.replace('"market_size":1.0', '"market_size":2.0'), encoding="utf-8"
    )
    with pytest.raises(LockError, match="hash drift"):
        validate_v3_artifacts(tmp_path)


def test_common_support_excludes_out_of_bounds_propensity_rows() -> None:
    rows = [_row("task-1", "T1"), _row("task-2", "T2")]
    rows[0]["propensity"] = 0.01
    rows[1]["propensity"] = 0.5
    audit = build_cohort(rows, expected_record_count=2)
    assert audit.common_support["out_of_bounds_task_ids"] == ["task-1"]
    assert audit.common_support["common_support_task_ids"] == ["task-2"]


def test_eligibility_ledger_keeps_b_subtype_counts() -> None:
    audit = build_cohort(
        [_row("task-1", "T1"), _row("task-2", "T2")],
        expected_record_count=2,
        expected_eligible_counts={
            "T1": 1,
            "T2": 1,
            "B-EARNINGS-T1": 1,
            "B-EARNINGS-T2": 1,
        },
        expected_b_ledger={"eligible": 2, "excluded": 0, "total": 2},
    )
    assert audit.eligibility_counts["eligibility_counts_match_expected"] is True
    assert audit.eligibility_counts["eligible_by_band_and_category"]["B-EARNINGS-T1"] == 1


@pytest.mark.parametrize(
    ("field", "invalid_value", "reason"),
    [
        ("official_temporal_eligible", False, "official_temporal_eligible_required"),
        ("review_status", False, "review_status_not_reviewed"),
        (
            "outcome_available_at_source",
            False,
            "outcome_available_at_source_not_observed",
        ),
        ("status", False, "status_not_ready"),
    ],
)
def test_self_declared_eligibility_cannot_waive_required_row_predicates(
    field: str, invalid_value: object, reason: str
) -> None:
    row = _row("invalid-" + field, "T1")
    row[field] = invalid_value
    reasons = _eligibility_reasons(row)
    assert reason in reasons

    audit = build_cohort([row], expected_record_count=1)
    assert audit.eligibility_counts["eligible_by_band_and_category"] == {}
    assert audit.exclusions[0]["reasons"] == sorted(reasons)


@pytest.mark.parametrize(
    "field",
    [
        "official_temporal_eligible",
        "review_status",
        "outcome_available_at_source",
        "status",
    ],
)
def test_missing_required_row_predicate_cannot_be_self_declared_eligible(field: str) -> None:
    row = _row("missing-" + field, "T1")
    row.pop(field)
    reasons = _eligibility_reasons(row)
    assert any(field in reason or reason == "status_not_ready" for reason in reasons)
    audit = build_cohort([row], expected_record_count=1)
    assert audit.exclusions[0]["task_id"] == row["task_id"]


@pytest.mark.parametrize("value", [True, False])
def test_boolean_eligibility_shorthand_is_explicitly_rejected(value: bool) -> None:
    row = _row(f"boolean-{value}", "T1")
    row["eligibility"] = value
    reasons = _eligibility_reasons(row)
    assert "eligibility_boolean_shorthand_unsupported" in reasons
    if value is False:
        assert "explicitly_ineligible" in reasons
    audit = build_cohort([row], expected_record_count=1)
    assert audit.exclusions[0]["reasons"] == sorted(reasons)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda row: row["metadata"].pop("source_url"), "missing_source_url"),
        (lambda row: row["metadata"].pop("source_hash"), "missing_source_hash"),
        (lambda row: row.__setitem__("evidence_verified", False), "evidence_not_verified"),
        (lambda row: row.pop("stratum"), "missing_key_match_fields"),
    ],
)
def test_self_declared_eligibility_cannot_waive_existing_conditions(
    change: Callable[[dict[str, object]], object], reason: str
) -> None:
    row = _row("existing-" + reason, "T1")
    change(row)
    reasons = _eligibility_reasons(row)
    assert reason in reasons
    audit = build_cohort([row], expected_record_count=1)
    assert audit.exclusions[0]["reasons"] == sorted(reasons)


def test_future_revision_still_requires_predicates_before_eligibility() -> None:
    rows = [_row(f"future-{index}", "T1" if index < 20 else "T2") for index in range(40)]
    expected_counts = {
        "T1": 20,
        "T2": 20,
        "B-EARNINGS-T1": 20,
        "B-EARNINGS-T2": 20,
    }
    expected_b = {"eligible": 40, "excluded": 0, "total": 40}
    current_lock = build_dataset_lock(rows, revision="b" * 40, expected_record_count=40)
    eligible = build_cohort(
        rows,
        dataset_lock=current_lock,
        expected_record_count=40,
        expected_eligible_counts=expected_counts,
        expected_b_ledger=expected_b,
    )
    assert eligible.status == "ELIGIBLE"

    rows[0]["official_temporal_eligible"] = False
    changed_lock = build_dataset_lock(rows, revision="c" * 40, expected_record_count=40)
    ineligible = build_cohort(
        rows,
        dataset_lock=changed_lock,
        expected_record_count=40,
        expected_eligible_counts=expected_counts,
        expected_b_ledger={"eligible": 39, "excluded": 1, "total": 40},
    )
    assert ineligible.status == CURRENT_DATA_INELIGIBLE
    assert "official_temporal_eligible_required" in ineligible.exclusions[0]["reasons"]


def test_brier_bss_and_fixed_reliability_golden() -> None:
    probabilities = [0.1, 0.9]
    outcomes = [0, 1]
    assert brier_score(probabilities, outcomes) == pytest.approx(0.01)
    assert brier_skill_score(probabilities, outcomes) == pytest.approx(0.96)
    report = reliability_report(probabilities, outcomes)
    assert [entry["n"] for entry in report["bins"]] == [1, 0, 0, 0, 1]
    assert report["bins"][1]["mean_probability"] is None
    assert report["bins"][0]["wilson_lower"] == pytest.approx(0.0)
    assert reliability_ece(probabilities, outcomes) == pytest.approx(0.1)
    boundary_bins = reliability_report([0.2, 0.6, 1.0], [0, 1, 1])["bins"]
    assert [entry["n"] for entry in boundary_bins] == [0, 1, 0, 1, 1]


def test_invalid_and_multi_trial_brier_are_operationally_explicit() -> None:
    report = operational_brier_score([None, 0.5], [0, 1])
    assert report["brier_score"] == pytest.approx(0.625)
    assert report["valid_output_rate"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="valid probabilities"):
        brier_score([None, 0.5], [0, 1])

    aggregate = aggregate_trial_brier(
        [
            {"system_id": "s", "task_id": "t", "probability": 0.0, "outcome": 0},
            {"system_id": "s", "task_id": "t", "probability": None, "outcome": 1},
        ]
    )
    assert aggregate["rows"][0]["brier_score"] == pytest.approx(0.5)
    assert aggregate["rows"][0]["valid_output_rate"] == pytest.approx(0.5)


def test_paired_event_bootstrap_uses_dataset_pin_seed_and_common_draws() -> None:
    rows = [
        {"cluster_key": "event-a", "left": 1.0, "right": 0.0},
        {"cluster_key": "event-b", "left": 0.0, "right": 1.0},
    ]
    first = paired_event_cluster_bootstrap(
        rows,
        dataset_pin="941370f228b9bc03c1b8bd9547c098c42c06be23",
        analysis_version="experiment-v3-r1",
        resamples=100,
    )
    second = paired_event_cluster_bootstrap(
        rows,
        dataset_pin="941370f228b9bc03c1b8bd9547c098c42c06be23",
        analysis_version="experiment-v3-r1",
        resamples=100,
    )
    assert first == second
    assert first["estimate"] == pytest.approx(0.0)
    assert first["cluster_count"] == 2
    assert first["seed_sha256"] == bootstrap_seed(
        "941370f228b9bc03c1b8bd9547c098c42c06be23", "experiment-v3-r1"
    )


def test_v3_cli_is_offline_and_has_a_clean_runtime_validate_command(tmp_path: Path) -> None:
    records = tmp_path / "task-metadata.jsonl"
    records.write_text(
        "".join(json.dumps(_row(task, band)) + "\n" for task, band in (("a", "T1"), ("b", "T2"))),
        encoding="utf-8",
    )
    output = tmp_path / "v3-lock"
    assert (
        main(
            [
                "experiment",
                "v3-lock",
                "--records",
                str(records),
                "--output",
                str(output),
                "--expected-record-count",
                "2",
            ]
        )
        == 0
    )
    assert main(["experiment", "v3-validate", "--root", str(output)]) == 0


def test_variable_count_hub_source_keeps_v06_source_contract_untouched(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def loader(_repo_id: str, **kwargs: object) -> list[dict[str, object]]:
        calls.append(kwargs)
        row = _row(kwargs["name"], "T1" if kwargs["name"] == "a1" else "T2")
        if kwargs["name"] == "a1":
            row["category"] = "A1"
        return [row]

    source = V3DatasetSource(
        cache_dir=tmp_path / "huggingface",
        config_names=("a1", "b"),
        expected_record_count=2,
        loader=loader,
        download_config_factory=lambda **kwargs: kwargs,
    )
    provenance = source.preflight()
    assert provenance["record_count"] == 2
    assert provenance["unique_task_count"] == 2
    assert [call["revision"] for call in calls] == ["941370f228b9bc03c1b8bd9547c098c42c06be23"] * 2
    assert len(source.load_records(categories=["B"])) == 1
