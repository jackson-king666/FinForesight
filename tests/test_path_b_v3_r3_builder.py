import csv
import json
from collections.abc import Mapping

import pytest

from benchmark_harness.experiment.path_b_v2 import (
    PathBV2ValidationError,
    assert_path_b_v2_score_blind,
    build_default_review_contract,
    canonical_json,
    sha256_json,
    validate_evidence_package,
)
from benchmark_harness.experiment.path_b_v3_r3_builder import (
    PATH_B_V3_R3_PACKAGE_FORMAT,
    prepare_path_b_v3_r3_review_pack,
)


def _source_item(kind: str, *, source_url: str, source_hash: str, published_at: str | None) -> dict:
    return {
        "cache_key": f"private-source-key-{kind}",
        "fetched_at": "2026-09-02T00:00:00Z",
        "kind": kind,
        "parser_version": "test-only",
        "published_at": published_at,
        "snapshot_path": f"evidence/{kind}.snapshot",
        "snapshot_sha256": source_hash,
        "source_url": source_url,
    }


def _source_package(task_id: str, *, include_event: bool = False) -> dict[str, object]:
    items = [
        _source_item(
            "baseline_task_identity",
            source_url="https://example.test/hub/task",
            source_hash="a" * 64,
            published_at=None,
        ),
    ]
    if include_event:
        items.append(
            _source_item(
                "event_snapshot",
                source_url="https://example.test/event",
                source_hash="b" * 64,
                published_at=None,
            )
        )
    items.append(
        _source_item(
            "price_snapshot",
            source_url="https://example.test/price",
            source_hash="c" * 64,
            published_at="2025-01-02T00:00:00Z",
        )
    )
    return {
        "task_id": task_id,
        "items": items,
        "evidence_package_sha256": sha256_json(items),
    }


def _ledger_row(
    task_id: str, category: str, variant: str, package: Mapping[str, object]
) -> dict[str, str]:
    return {
        "task_id": task_id,
        "scope_role": "both" if category == "B" else "temporal_t1_t2",
        "category": category,
        "variant": variant,
        "paper_band": "T1",
        "review_status": "draft",
        "official_temporal_eligible": "false",
        "collector_id": "collector-v2",
        "reviewed_at": "",
        "event_evidence_status": "draft",
        "price_evidence_status": "draft",
        "exclusion_reason_code": "human_attestation_missing",
        "evidence_package_sha256": str(package["evidence_package_sha256"]),
    }


def _temporal_row(task_id: str, *, complete: bool = True) -> dict[str, str]:
    return {
        "task_id": task_id,
        "category": "A1",
        "evidence_url": "https://example.test/official-temporal" if complete else "",
        "evidence_published_at": "2025-01-03T00:00:00Z" if complete else "",
        "content_sha256": "d" * 64 if complete else "",
        "outcome_available_at_source": ("observed_official_snapshot" if complete else ""),
        "manual_review_eligible": "是" if complete else "否",
        # These source-only columns must not cross the builder boundary.
        "predicted_metric": "sensitive-source-payload",
        "unit": "sensitive-source-payload",
        "final_value": "sensitive-source-payload",
        "evidence_rationale": "sensitive-source-payload",
    }


def _event_row(task_id: str) -> dict[str, str]:
    return {
        "task_id": task_id,
        "first_public_at": "2025-01-04T00:00:00Z",
        "evidence_url": "https://example.test/official-event",
        "content_sha256": "e" * 64,
        "price_evidence_url": "https://example.test/reaction-price",
        "price_evidence_published_at": "2025-01-02T00:00:00Z",
        "price_content_sha256": "f" * 64,
        "manual_review_eligible": "是",
        "event_summary": "sensitive-source-payload",
        "direction_link": "sensitive-source-payload",
        "evidence_rationale": "sensitive-source-payload",
        "reviewer": "sensitive-source-payload",
    }


def _write_csv(path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_source(tmp_path, *, include_temporal: bool = True):
    source = tmp_path / "path_b_v2"
    calibration = source / "calibration"
    calibration.mkdir(parents=True)
    a_package = _source_package("A1-00001")
    b_package = _source_package("B-EARN-00001", include_event=True)
    packages = [a_package, b_package]
    ledger = [
        _ledger_row("A1-00001", "A1", "", a_package),
        _ledger_row("B-EARN-00001", "B", "earnings", b_package),
    ]
    _write_csv(calibration / "review_ledger_v2.csv", ledger)
    (calibration / "evidence_packages_v2.jsonl").write_text(
        "".join(canonical_json(package) + "\n" for package in packages),
        encoding="utf-8",
    )
    temporal_rows = [_temporal_row("A1-00001", complete=include_temporal)]
    _write_csv(calibration / "temporal_provenance_table.csv", temporal_rows)
    _write_csv(calibration / "b_event_evidence_table.csv", [_event_row("B-EARN-00001")])
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        canonical_json(build_default_review_contract()) + "\n", encoding="utf-8"
    )
    return source, contract_path, packages


def test_r3_builder_rebinds_all_rows_and_is_deterministic(tmp_path) -> None:
    source, contract_path, source_packages = _write_source(tmp_path)
    first = tmp_path / "r3-first"
    second = tmp_path / "r3-second"
    first_result = prepare_path_b_v3_r3_review_pack(
        source, first, contract_path=contract_path, expected_task_count=None
    )
    second_result = prepare_path_b_v3_r3_review_pack(
        source, second, contract_path=contract_path, expected_task_count=None
    )

    assert first_result["derived_package_count"] == 2
    assert first_result["reviewer_export_count"] == 2
    assert first_result["validator_ready_package_count"] == 1
    assert first_result["manifest_sha256"] == second_result["manifest_sha256"]
    assert first_result["roster_sha256"] == second_result["roster_sha256"]
    assert (
        first_result["reviewer_export_file_sha256"] == second_result["reviewer_export_file_sha256"]
    )
    for path in first.iterdir():
        assert path.read_bytes() == (second / path.name).read_bytes()

    raw_hashes = {str(package["evidence_package_sha256"]) for package in source_packages}
    roster = json.loads((first / "canonical-roster.json").read_text(encoding="utf-8"))
    assert {row["evidence_package_sha256"] for row in roster["rows"]}.isdisjoint(raw_hashes)
    assert roster["schema_version"] == "path-b-v3-r3-review-pack/v1"


def test_r3_export_has_review_fields_and_remains_score_blind(tmp_path) -> None:
    source, contract_path, _ = _write_source(tmp_path)
    output = tmp_path / "r3"
    prepare_path_b_v3_r3_review_pack(
        source, output, contract_path=contract_path, expected_task_count=None
    )
    export_text = (output / "blinded-review-export.jsonl").read_text(encoding="utf-8")
    export = [json.loads(line) for line in export_text.splitlines()]
    assert len(export) == 2
    assert_path_b_v2_score_blind(export)
    assert "collector_id" not in export_text
    assert "sensitive-source-payload" not in export_text
    for row in export:
        assert {
            "official_time_evidence",
            "official_source_urls",
            "official_snapshot_refs",
            "published_at",
            "stratum",
            "evidence_package_content_summary",
            "evidence_package_sha256",
        } <= row.keys()
        assert row["review_status"] == "draft"
        assert row["official_temporal_eligible"] is False
        assert row["evidence_package_content_summary"]["format"] == PATH_B_V3_R3_PACKAGE_FORMAT
        assert row["evidence_package_content_summary"]["component_keys"] == [
            "task_identity",
            "temporal_provenance",
            "event_evidence",
            "outcome_derivation",
        ]

    instructions = (output / "review-instructions.md").read_text(encoding="utf-8")
    assert "pre-attestation source projections" in instructions
    assert "They are not a veto" in instructions
    assert "Five frozen decision criteria" in instructions
    for criterion in range(1, 6):
        assert f"{criterion}." in instructions
    manifest = json.loads((output / "sealed-review-pack-manifest.json").read_text())
    assert manifest["review_state"] == "pending_independent_attestation"
    assert manifest["supersedes"]["r2_cr_import_sha256"].startswith("f53b007b")
    assert manifest["supersedes"]["r2_qa_import_sha256"].startswith("e60911ea")


def test_r3_incomplete_b_package_fails_closed_but_complete_a_package_validates(tmp_path) -> None:
    source, contract_path, _ = _write_source(tmp_path)
    output = tmp_path / "r3"
    prepare_path_b_v3_r3_review_pack(
        source, output, contract_path=contract_path, expected_task_count=None
    )
    packages = [
        json.loads(line)
        for line in (output / "derived-evidence-packages.jsonl").read_text().splitlines()
    ]
    by_task = {package["task_id"]: package for package in packages}
    assert validate_evidence_package(by_task["A1-00001"])["category"] == "A1"
    with pytest.raises(PathBV2ValidationError, match="price_evidence_status"):
        validate_evidence_package(by_task["B-EARN-00001"])
    package_manifest = json.loads(
        (output / "derived-evidence-package-manifest.json").read_text(encoding="utf-8")
    )
    b_status = next(
        entry for entry in package_manifest["packages"] if entry["task_id"] == "B-EARN-00001"
    )
    assert b_status["validation_status"] == "not_validator_ready"
    assert "missing_price_chain_pre" in b_status["exclusion_reasons"]
    assert b_status["source_package_sha256"] != b_status["evidence_package_sha256"]


def test_r3_preserves_missing_temporal_material_without_fabricating_date(tmp_path) -> None:
    source, contract_path, _ = _write_source(tmp_path, include_temporal=False)
    output = tmp_path / "r3"
    prepare_path_b_v3_r3_review_pack(
        source, output, contract_path=contract_path, expected_task_count=None
    )
    export = [
        json.loads(line)
        for line in (output / "blinded-review-export.jsonl").read_text().splitlines()
    ]
    a_row = next(row for row in export if row["task_id"] == "A1-00001")
    assert a_row["published_at"] is None
    assert a_row["evidence_package_content_summary"]["validation_status"] == ("not_validator_ready")
    assert (
        "missing_official_temporal_evidence"
        in a_row["evidence_package_content_summary"]["exclusion_reasons"]
    )


def test_r3_output_is_append_only(tmp_path) -> None:
    source, contract_path, _ = _write_source(tmp_path)
    output = tmp_path / "r3"
    prepare_path_b_v3_r3_review_pack(
        source, output, contract_path=contract_path, expected_task_count=None
    )
    with pytest.raises(PathBV2ValidationError, match="must not already exist"):
        prepare_path_b_v3_r3_review_pack(
            source, output, contract_path=contract_path, expected_task_count=None
        )
