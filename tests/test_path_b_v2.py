"""Offline contract, evidence, and tamper tests for Path B v2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_harness.cli import main
from benchmark_harness.experiment.path_b_v2 import (
    PATH_B_V2_RETURN_FORMULA,
    PATH_B_V2_REVIEW_METHODS,
    PathBV2ScoreBlindError,
    PathBV2ValidationError,
    assert_path_b_v2_score_blind,
    build_default_review_contract,
    build_evidence_package_manifest,
    build_task_identity_lock,
    compute_attestation_sha256,
    compute_evidence_package_sha256,
    validate_attestation,
    validate_eligibility_rows,
    validate_evidence_package,
    validate_review_contract,
    validate_task_identity_unchanged,
)


def _package(*, category: str = "B", variant: str = "earnings") -> dict[str, object]:
    package: dict[str, object] = {
        "task_identity": {
            "task_id": "b-1",
            "category": category,
            "variant": variant,
            "time_band": "T1",
        },
        "temporal_provenance": {
            "task_id": "b-1",
            "official_temporal_eligible": True,
            "outcome_available_at_source": "observed_current",
            "cutoff_date": "2024-01-01",
        },
        "event_evidence": {
            "task_id": "b-1",
            "event_id": "earnings-1",
            "source_url": "https://events.example.test/earnings-1",
            "source_sha256": "a" * 64,
            "published_at": "2024-01-10T15:00:00Z",
        },
        "outcome_derivation": {
            "task_id": "b-1",
            "price_evidence_status": "available",
            "price_chain": {
                "pre": {
                    "source_url": "https://prices.example.test/instrument-pre",
                    "source_sha256": "b" * 64,
                    "observed_at": "2024-01-09T21:00:00Z",
                    "price": 100.0,
                },
                "post": {
                    "source_url": "https://prices.example.test/instrument-post",
                    "source_sha256": "c" * 64,
                    "observed_at": "2024-01-11T21:00:00Z",
                    "price": 110.0,
                },
                "benchmark_pre": {
                    "source_url": "https://prices.example.test/benchmark-pre",
                    "source_sha256": "d" * 64,
                    "observed_at": "2024-01-09T21:00:00Z",
                    "price": 200.0,
                },
                "benchmark_post": {
                    "source_url": "https://prices.example.test/benchmark-post",
                    "source_sha256": "e" * 64,
                    "observed_at": "2024-01-11T21:00:00Z",
                    "price": 204.0,
                },
            },
            "return_formula": {
                "id": "excess_return_v1",
                "expression": PATH_B_V2_RETURN_FORMULA,
            },
            "instrument": "instrument-1",
            "benchmark": "benchmark-1",
            "pre_event_session": "2024-01-09|close",
            "post_event_session": "2024-01-11|close",
            "market_calendar": "XNYS",
            "timezone": "America/New_York",
            "excess_return": 0.08,
            "recomputed_excess_return": 0.08,
        },
    }
    package["evidence_package_sha256"] = compute_evidence_package_sha256(package)
    return package


def _attestation(
    package: dict[str, object], *, reviewer_id: str = "reviewer-1"
) -> dict[str, object]:
    attestation: dict[str, object] = {
        "task_id": "b-1",
        "decision": "eligible",
        "reviewer_id": reviewer_id,
        "review_method": "human_first_party_archive_snapshot",
        "reviewed_at": "2024-01-12T12:00:00Z",
        "evidence_package_sha256": package["evidence_package_sha256"],
        "notes": "independent review completed",
    }
    attestation["attestation_sha256"] = compute_attestation_sha256(attestation)
    return attestation


def _row(
    package: dict[str, object], *, eligible: bool = True, category: str = "B"
) -> dict[str, object]:
    return {
        "task_id": "b-1",
        "category": category,
        "variant": "earnings",
        "status": "ready",
        "official_temporal_eligible": True,
        "review_status": "reviewed",
        "outcome_available_at_source": "observed_current",
        "collector_id": "collector-1",
        "stratum": "B|earnings|US|direction|template-1",
        "evidence_verified": True,
        "metadata": {
            "source_url": "https://source.example.test/b-1",
            "source_sha256": "f" * 64,
        },
        "eligibility": {"eligible": eligible, "reasons": [] if eligible else ["manual_exclusion"]},
        "evidence_package_sha256": package["evidence_package_sha256"],
    }


def test_contract_freezes_pin_scope_and_four_review_methods() -> None:
    contract = build_default_review_contract()
    assert validate_review_contract(contract) == contract
    assert contract["review_methods"] == [
        "human_first_party_live_and_snapshot",
        "human_first_party_archive_snapshot",
        "human_official_api_snapshot",
        "automated_precheck_plus_human_review",
    ]
    for method in ("automated_only", "url_present_only", "modeled_date"):
        invalid = json.loads(json.dumps(contract))
        invalid["review_methods"][0] = method
        with pytest.raises(PathBV2ValidationError, match="four human methods"):
            validate_review_contract(invalid)


def test_attestation_is_typed_hashed_and_reviewer_must_differ() -> None:
    package = _package()
    attestation = _attestation(package)
    summary = validate_attestation(
        attestation,
        expected_task_id="b-1",
        expected_evidence_package_sha256=package["evidence_package_sha256"],
        collector_id="collector-1",
    )
    assert summary["collector_identity_checked"] is True

    invalid = dict(attestation)
    invalid["reviewer_id"] = "collector-1"
    invalid["attestation_sha256"] = compute_attestation_sha256(invalid)
    with pytest.raises(PathBV2ValidationError, match="differ"):
        validate_attestation(invalid, collector_id="collector-1")

    invalid = dict(attestation)
    invalid["decision"] = True
    with pytest.raises(PathBV2ValidationError, match="decision"):
        validate_attestation(invalid)

    for method in PATH_B_V2_REVIEW_METHODS:
        valid_method = dict(attestation)
        valid_method["review_method"] = method
        valid_method["attestation_sha256"] = compute_attestation_sha256(valid_method)
        assert validate_attestation(valid_method)["review_method"] == method

    invalid = dict(attestation)
    invalid["review_method"] = "automated_only"
    invalid["attestation_sha256"] = compute_attestation_sha256(invalid)
    with pytest.raises(PathBV2ValidationError, match="allowed human"):
        validate_attestation(invalid)


def test_b_evidence_requires_separate_event_price_chain_and_fixed_return() -> None:
    package = _package()
    assert validate_evidence_package(package)["category"] == "B-EARNINGS"

    tampered = json.loads(json.dumps(package))
    tampered["outcome_derivation"]["return_formula"]["expression"] = "post - pre"
    tampered["evidence_package_sha256"] = compute_evidence_package_sha256(tampered)
    with pytest.raises(PathBV2ValidationError, match="formula"):
        validate_evidence_package(tampered)

    tampered = json.loads(json.dumps(package))
    tampered["outcome_derivation"]["price_evidence_status"] = "not_applicable"
    tampered["evidence_package_sha256"] = compute_evidence_package_sha256(tampered)
    with pytest.raises(PathBV2ValidationError, match="not_applicable"):
        validate_evidence_package(tampered)

    tampered = json.loads(json.dumps(package))
    tampered["outcome_derivation"]["price_chain"]["post"]["observed_at"] = "2024-01-09T21:00:00Z"
    tampered["evidence_package_sha256"] = compute_evidence_package_sha256(tampered)
    with pytest.raises(PathBV2ValidationError, match="post-event"):
        validate_evidence_package(tampered)


def test_eligibility_is_explicit_and_counts_are_conserved_with_c_excluded() -> None:
    package = _package()
    attestation = _attestation(package)
    c_package = _package(category="C", variant="t2")
    rows = [_row(package), {**_row(c_package, eligible=False, category="C"), "task_id": "c-1"}]
    result = validate_eligibility_rows(
        rows,
        [attestation],
        [package],
        review_contract=build_default_review_contract(),
    )
    assert result["total_count"] == result["eligible_count"] + result["excluded_count"] == 2
    assert result["eligible_task_ids"] == ["b-1"]
    assert result["excluded_task_ids"] == ["c-1"]
    assert result["c_excluded"] is True

    c_with_package = _row(c_package, eligible=False, category="C")
    c_with_package["task_id"] = "c-1"
    with pytest.raises(PathBV2ValidationError, match="outside Path B scope"):
        validate_eligibility_rows([c_with_package], [], [c_package])

    for shorthand in (True, False, "false", 0, 1):
        invalid = _row(package)
        invalid["eligibility"] = shorthand
        with pytest.raises(PathBV2ValidationError, match="boolean shorthand|JSON boolean"):
            validate_eligibility_rows([invalid], [attestation], [package])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("official_temporal_eligible", False, "temporal"),
        ("review_status", "pending", "reviewed"),
        ("outcome_available_at_source", "modeled_2024", "observed"),
        ("status", "draft", "ready"),
    ],
)
def test_self_declared_eligibility_cannot_waive_row_predicates(
    field: str, value: object, message: str
) -> None:
    package = _package()
    row = _row(package)
    row[field] = value
    with pytest.raises(PathBV2ValidationError, match=message):
        validate_eligibility_rows([row], [_attestation(package)], [package])


def test_package_manifest_binds_contract_and_detects_hash_drift() -> None:
    package = _package()
    contract = build_default_review_contract()
    manifest = build_evidence_package_manifest([package], review_contract=contract)
    assert manifest["package_count"] == 1

    validate_eligibility_rows(
        [_row(package)],
        [_attestation(package)],
        [package],
        manifest=manifest,
        review_contract=contract,
    )
    invalid = dict(manifest)
    invalid["packages"] = [{"task_id": "b-1", "evidence_package_sha256": "0" * 64}]
    with pytest.raises(PathBV2ValidationError, match="manifest hash mismatch"):
        validate_eligibility_rows(
            [_row(package)],
            [_attestation(package)],
            [package],
            manifest=invalid,
            review_contract=contract,
        )


def test_task_identity_lock_is_score_blind_and_detects_drift() -> None:
    baseline = [{"task_id": "a", "category": "A1", "variant": "t1"}]
    current = [{"task_id": "a", "category": "A1", "variant": "t1"}]
    assert build_task_identity_lock(baseline) == build_task_identity_lock(current)
    assert validate_task_identity_unchanged(current, baseline)["status"] == "unchanged"
    with pytest.raises(PathBV2ScoreBlindError, match="forbidden"):
        build_task_identity_lock([{**baseline[0], "score": 0.1}])
    changed = [{"task_id": "a", "category": "A2-T", "variant": "t1"}]
    with pytest.raises(PathBV2ValidationError, match="identity drift"):
        validate_task_identity_unchanged(changed, baseline)


def test_clean_runtime_cli_validates_only_path_b_pack(tmp_path: Path) -> None:
    package = _package()
    contract = build_default_review_contract()
    root = tmp_path / "path_b_v2"
    root.mkdir()
    (root / "path_b_v2_review_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "eligibility.jsonl").write_text(
        json.dumps(_row(package), sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "attestations.jsonl").write_text(
        json.dumps(_attestation(package), sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "evidence-packages.jsonl").write_text(
        json.dumps(package, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "evidence-package-manifest.json").write_text(
        json.dumps(
            build_evidence_package_manifest([package], review_contract=contract), sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    assert (
        main(
            [
                "path-b-v2-validate",
                "--root",
                str(root),
                "--contract",
                str(root / "path_b_v2_review_contract.json"),
            ]
        )
        == 0
    )

    (root / "raw-A.jsonl").write_text("must not be read\n", encoding="utf-8")
    assert (
        main(
            [
                "path-b-v2-validate",
                "--root",
                str(root),
                "--contract",
                str(root / "path_b_v2_review_contract.json"),
            ]
        )
        == 2
    )


def test_score_fence_rejects_forbidden_keys_without_reading_values() -> None:
    with pytest.raises(PathBV2ScoreBlindError, match="forbidden"):
        assert_path_b_v2_score_blind({"result": object()})
