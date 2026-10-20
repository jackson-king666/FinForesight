"""Score-blind Path B v3 review, cohort, probe, and calibration contracts."""

from __future__ import annotations

import json

import pytest

from benchmark_harness.experiment.path_b_v2 import (
    PathBV2ScoreBlindError,
    PathBV2ValidationError,
    build_default_review_contract,
    compute_attestation_sha256,
    compute_evidence_package_sha256,
)
from benchmark_harness.experiment.path_b_v3 import (
    build_b_calibration_spec,
    build_canonical_roster,
    build_leakage_probe_plan,
    build_t1_t2_analysis_spec,
    import_blinded_attestations,
    materialize_b_calibration_report,
    materialize_t1_t2_cohort,
    render_adjudication_report,
    write_sealed_review_pack,
)


def _package(task_id: str, band: str) -> dict[str, object]:
    package: dict[str, object] = {
        "task_identity": {"task_id": task_id, "category": "A1", "time_band": band},
        "temporal_provenance": {"task_id": task_id, "official_temporal_eligible": True},
        "event_evidence": {"task_id": task_id, "published_at": "2024-01-02T00:00:00Z"},
        "outcome_derivation": {"task_id": task_id, "status": "reviewable"},
    }
    package["evidence_package_sha256"] = compute_evidence_package_sha256(package)
    return package


def _row(task_id: str, band: str, package: dict[str, object]) -> dict[str, object]:
    return {
        "task_id": task_id,
        "category": "A1",
        "time_band": band,
        "task_family": "forecast",
        "paper_band": "P1",
        "outcome_type": "direction",
        "collector_id": "collector-1",
        "evidence_package_sha256": package["evidence_package_sha256"],
        "eligibility": {"eligible": True, "reasons": []},
    }


def _attestation(task_id: str, package: dict[str, object], reviewer: str) -> dict[str, object]:
    value: dict[str, object] = {
        "task_id": task_id,
        "decision": "eligible",
        "reviewer_id": reviewer,
        "review_method": "human_official_api_snapshot",
        "reviewed_at": "2026-09-02T00:00:00Z",
        "evidence_package_sha256": package["evidence_package_sha256"],
        "notes": "review complete",
    }
    value["attestation_sha256"] = compute_attestation_sha256(value)
    return value


def test_roster_rejects_score_keys_and_stale_attestations() -> None:
    package = _package("t1", "T1")
    row = _row("t1", "T1", package)
    roster = build_canonical_roster(
        [row], [package], review_contract=build_default_review_contract()
    )
    assert roster["roster_count"] == 1
    assert roster["rows"][0]["collector_id"] == "collector-1"
    with pytest.raises(PathBV2ScoreBlindError):
        build_canonical_roster(
            [{**row, "score": 0.2}], [package], review_contract=build_default_review_contract()
        )
    stale = _attestation("t1", package, "reviewer-1")
    stale["evidence_package_sha256"] = "0" * 64
    stale["attestation_sha256"] = compute_attestation_sha256(stale)
    with pytest.raises(PathBV2ValidationError, match="stale"):
        import_blinded_attestations(
            [stale], roster, review_contract=build_default_review_contract()
        )


def test_sealed_pack_is_append_only_and_omits_collector_identity_from_blinded_queue(
    tmp_path,
) -> None:
    package = _package("t1", "T1")
    roster = build_canonical_roster(
        [_row("t1", "T1", package)], [package], review_contract=build_default_review_contract()
    )
    output = tmp_path / "sealed"
    result = write_sealed_review_pack(str(output), roster)
    assert result["roster_sha256"] == roster["roster_sha256"]
    queue = json.loads((output / "blinded-review-queue.jsonl").read_text())
    assert "collector_id" not in queue
    with pytest.raises(PathBV2ValidationError, match="must not already exist"):
        write_sealed_review_pack(str(output), roster)


def test_adjudication_requires_independent_reviewers_and_is_deterministic() -> None:
    package = _package("t1", "T1")
    roster = build_canonical_roster(
        [_row("t1", "T1", package)], [package], review_contract=build_default_review_contract()
    )
    first = _attestation("t1", package, "reviewer-a")
    second = _attestation("t1", package, "reviewer-b")
    report = render_adjudication_report(
        [first], [second], roster, review_contract=build_default_review_contract()
    )
    assert report["adjudicated"][0]["decision"] == "eligible"
    with pytest.raises(PathBV2ValidationError, match="independent"):
        render_adjudication_report(
            [first], [first], roster, review_contract=build_default_review_contract()
        )


def test_probe_pairs_cutoff_and_unrestricted_without_ranking_or_root_overlap() -> None:
    plan = build_leakage_probe_plan(
        [
            {"task_id": "t1", "system_id": "odr", "trial": 1, "budget_sha256": "a" * 64, "seed": 3},
        ],
        results_root="results/leakage-probe-v3",
    )
    assert plan["ranking_eligible"] is False
    assert len(plan["cells"]) == 2
    assert {cell["retrieval_regime"] for cell in plan["cells"]} == {
        "cutoff_controlled",
        "unrestricted_current_web",
    }
    with pytest.raises(PathBV2ValidationError, match="isolated"):
        build_leakage_probe_plan([], results_root="results/v2")


def test_t1_t2_cohort_is_deterministic_and_detects_spec_tampering() -> None:
    packages = [_package("t1", "T1"), _package("t2", "T2")]
    rows = [_row("t1", "T1", packages[0]), _row("t2", "T2", packages[1])]
    spec = build_t1_t2_analysis_spec()
    first = materialize_t1_t2_cohort(rows, spec)
    second = materialize_t1_t2_cohort(list(reversed(rows)), spec)
    assert first["cohort_sha256"] == second["cohort_sha256"]
    bad_spec = json.loads(json.dumps(spec))
    bad_spec["weighting"]["method"] = "post_hoc"
    with pytest.raises(PathBV2ValidationError, match="spec hash"):
        materialize_t1_t2_cohort(rows, bad_spec)


def test_b_calibration_is_available_case_only_and_excludes_bls_macro_and_inverted() -> None:
    spec = build_b_calibration_spec()
    rows = [
        {
            "task_id": "earn-ok",
            "category": "B-EARNINGS",
            "event_chain": "complete",
            "price_chain": "complete",
            "temporal_order": "valid",
            "rationale": "consistent",
        },
        {
            "task_id": "bls",
            "category": "B-EARNINGS",
            "event_chain": "missing_bls",
            "price_chain": "complete",
            "temporal_order": "valid",
            "rationale": "consistent",
        },
        {
            "task_id": "macro",
            "category": "B-MACRO",
            "event_chain": "complete",
            "price_chain": "complete",
            "temporal_order": "valid",
            "rationale": "consistent",
        },
        {
            "task_id": "inverted",
            "category": "B-EARNINGS",
            "event_chain": "complete",
            "price_chain": "complete",
            "temporal_order": "inverted",
            "rationale": "consistent",
        },
    ]
    report = materialize_b_calibration_report(rows, spec)
    assert report["available_case_task_ids"] == ["earn-ok"]
    assert report["complete_calibration"] is False
    assert report["exclusions"]["bls"] == "missing_bls_first_party_evidence"
    assert report["exclusions"]["macro"] == "b_macro_out_of_scope"
