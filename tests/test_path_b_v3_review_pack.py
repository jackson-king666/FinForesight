import csv
import json

import pytest

from benchmark_harness.experiment.path_b_v2 import (
    PathBV2ValidationError,
    assert_path_b_v2_score_blind,
    build_default_review_contract,
    canonical_json,
    sha256_json,
)
from benchmark_harness.experiment.path_b_v3 import (
    PATH_B_V3_REVIEWER_EXPORT_REQUIRED_FIELDS,
    PATH_B_V3_REVIEWER_EXPORT_SCHEMA,
    build_path_b_v2_canonical_roster,
)
from benchmark_harness.experiment.path_b_v3_review_pack import (
    prepare_path_b_v3_review_pack,
)


def _source_package(task_id: str) -> dict[str, object]:
    items = [
        {
            "kind": "baseline_task_identity",
            "fetched_at": "2026-09-02T00:00:00Z",
            "published_at": "2024-01-01T00:00:00Z",
            "snapshot_path": "snapshots/identity.json",
            "snapshot_sha256": "a" * 64,
            "source_url": "https://example.test/identity",
        }
    ]
    return {
        "task_id": task_id,
        "items": items,
        "evidence_package_sha256": sha256_json(items),
    }


def _ledger_row(
    task_id: str, category: str, variant: str, paper_band: str, package_hash: str
) -> dict[str, str]:
    return {
        "task_id": task_id,
        "scope_role": "temporal_t1_t2" if category != "B" else "both",
        "category": category,
        "variant": variant,
        "paper_band": paper_band,
        "collector_id": "collector-v2",
        "evidence_package_sha256": package_hash,
        "official_temporal_eligible": "false",
        "review_status": "draft",
        "reviewed_at": "",
        "event_evidence_status": "draft",
        "price_evidence_status": "draft",
        "exclusion_reason_code": "",
    }


def test_current_source_package_form_is_bound_without_rewriting_hashes() -> None:
    packages = [_source_package(task_id) for task_id in ("a1", "a2", "earn", "macro")]
    rows = [
        _ledger_row("a1", "A1", "", "T1", packages[0]["evidence_package_sha256"]),
        _ledger_row("a2", "A2", "T", "T2", packages[1]["evidence_package_sha256"]),
        _ledger_row("earn", "B", "earnings", "T2", packages[2]["evidence_package_sha256"]),
        _ledger_row("macro", "B", "macro", "T2", packages[3]["evidence_package_sha256"]),
    ]
    roster = build_path_b_v2_canonical_roster(
        rows,
        packages,
        review_contract=build_default_review_contract(),
    )
    assert roster["roster_count"] == 4
    assert [row["category"] for row in roster["rows"]] == [
        "A1",
        "A2-T",
        "B-EARNINGS",
        "B-MACRO",
    ]
    assert {row["evidence_package_sha256"] for row in roster["rows"]} == {
        package["evidence_package_sha256"] for package in packages
    }


def test_prepare_current_pin_pack_is_pending_and_collector_free_in_export(tmp_path) -> None:
    source = tmp_path / "path_b_v2"
    calibration = source / "calibration"
    calibration.mkdir(parents=True)
    packages = [_source_package(task_id) for task_id in ("a1", "a2")]
    rows = [
        _ledger_row("a1", "A1", "", "T1", packages[0]["evidence_package_sha256"]),
        _ledger_row("a2", "A2", "T", "T2", packages[1]["evidence_package_sha256"]),
    ]
    with (calibration / "review_ledger_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (calibration / "evidence_packages_v2.jsonl").write_text(
        "".join(canonical_json(package) + "\n" for package in packages),
        encoding="utf-8",
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        canonical_json(build_default_review_contract()) + "\n", encoding="utf-8"
    )

    output = tmp_path / "review-pack"
    result = prepare_path_b_v3_review_pack(
        source,
        output,
        contract_path=contract_path,
    )
    assert result["ledger_count"] == 2
    assert result["package_count"] == 2
    assert result["review_state"] == "pending_independent_attestation"
    assert (output / "attestations.jsonl").read_text() == ""
    export = [
        json.loads(line)
        for line in (output / "blinded-review-export.jsonl").read_text().splitlines()
    ]
    assert len(export) == 2
    assert all(PATH_B_V3_REVIEWER_EXPORT_REQUIRED_FIELDS <= row.keys() for row in export)
    assert all("collector_id" not in row for row in export)
    assert all(row["review_status"] == "draft" for row in export)
    assert all(row["official_temporal_eligible"] is False for row in export)
    assert all(row["published_at"] == "2024-01-01T00:00:00Z" for row in export)
    assert all(row["official_source_urls"] == ["https://example.test/identity"] for row in export)
    assert all(
        row["official_snapshot_refs"][0]["snapshot_path"] == "snapshots/identity.json"
        for row in export
    )
    assert all(
        set(row["stratum"])
        == {
            "category",
            "time_band",
            "task_family",
            "paper_band",
            "outcome_type",
        }
        for row in export
    )
    assert all(row["evidence_package_content_summary"]["item_count"] == 1 for row in export)
    assert_path_b_v2_score_blind(export)
    manifest = json.loads((output / "sealed-review-pack-manifest.json").read_text())
    assert manifest["review_state"] == "pending_independent_attestation"
    assert manifest["attestation_count"] == 0
    assert manifest["reviewer_export_schema"] == PATH_B_V3_REVIEWER_EXPORT_SCHEMA
    assert manifest["reviewer_export_count"] == 2
    assert manifest["supersedes"]["disposition"] == "invalidated_not_for_adjudication"
    assert manifest["supersedes"]["qa_import_sha256"].startswith("a2485bc4")
    assert manifest["supersedes"]["reviewer_export_sha256"].startswith("ea4615c7")
    with pytest.raises(PathBV2ValidationError, match="must not already exist"):
        prepare_path_b_v3_review_pack(source, output, contract_path=contract_path)


def test_prepare_preserves_missing_publication_evidence_as_missing(tmp_path) -> None:
    source = tmp_path / "path_b_v2"
    calibration = source / "calibration"
    calibration.mkdir(parents=True)
    package = _source_package("a1")
    package["items"][0]["published_at"] = None
    package["evidence_package_sha256"] = sha256_json(package["items"])
    row = _ledger_row("a1", "A1", "", "T1", package["evidence_package_sha256"])
    with (calibration / "review_ledger_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    (calibration / "evidence_packages_v2.jsonl").write_text(
        canonical_json(package) + "\n", encoding="utf-8"
    )
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        canonical_json(build_default_review_contract()) + "\n", encoding="utf-8"
    )

    output = tmp_path / "review-pack"
    prepare_path_b_v3_review_pack(source, output, contract_path=contract_path)
    row_export = json.loads((output / "blinded-review-export.jsonl").read_text(encoding="utf-8"))
    assert row_export["published_at"] is None
    assert row_export["published_at_candidates"] == []
    assert row_export["evidence_package_content_summary"]["item_refs"][0]["published_at"] is None
