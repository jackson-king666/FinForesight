"""Prepare a score-blind Path B v3 review pack from the pinned Hub source."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmark_harness.experiment.path_b_v2 import (
    PathBV2ValidationError,
    assert_path_b_v2_score_blind,
    sha256_json,
    validate_review_contract,
)
from benchmark_harness.experiment.path_b_v3 import (
    PATH_B_V3_SOURCE_LOCK_SCHEMA,
    build_path_b_v2_canonical_roster,
    write_sealed_review_pack,
)

_LEDGER_RELATIVE_PATH = Path("calibration/review_ledger_v2.csv")
_PACKAGES_RELATIVE_PATH = Path("calibration/evidence_packages_v2.jsonl")
PATH_B_V3_SUPERSEDED_REVIEWER_EXPORT_SHA256 = (
    "ea4615c7317360a67704f51f5eadd836bdf84584fcbb8628015bb9eda76c141c"
)
PATH_B_V3_INVALIDATED_QA_IMPORT_SHA256 = (
    "a2485bc4548e91bb53798d63c2f0f331822cd49b671a712b871d695a9466c8b0"
)
_REQUIRED_LEDGER_COLUMNS = frozenset(
    {
        "task_id",
        "scope_role",
        "category",
        "variant",
        "paper_band",
        "collector_id",
        "evidence_package_sha256",
        "official_temporal_eligible",
        "review_status",
        "reviewed_at",
        "event_evidence_status",
        "price_evidence_status",
        "exclusion_reason_code",
    }
)

_PACKAGE_REVIEW_FIELDS = (
    "kind",
    "source_url",
    "snapshot_path",
    "snapshot_sha256",
    "published_at",
    "fetched_at",
    "parser_version",
)


def _read_ledger(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = frozenset(reader.fieldnames or ())
            missing = sorted(_REQUIRED_LEDGER_COLUMNS - fieldnames)
            if missing:
                raise PathBV2ValidationError(
                    f"Path B ledger is missing required score-blind columns: {missing}"
                )
            return [
                {field: str(row.get(field) or "") for field in _REQUIRED_LEDGER_COLUMNS}
                for row in reader
            ]
    except OSError as exc:
        raise PathBV2ValidationError(f"unable to read Path B ledger: {path}") from exc


def _read_packages(path: Path) -> list[Mapping[str, Any]]:
    packages: list[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PathBV2ValidationError(f"unable to read Path B evidence packages: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PathBV2ValidationError(
                f"invalid Path B evidence package JSON at line {line_number}"
            ) from exc
        if not isinstance(value, Mapping):
            raise PathBV2ValidationError(
                f"Path B evidence package at line {line_number} must be an object"
            )
        packages.append(value)
    return packages


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _explicit_boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise PathBV2ValidationError(f"{name} must be an explicit true/false value")


def _package_review_refs(package: Mapping[str, Any], task_id: str) -> list[dict[str, Any]]:
    items = package.get("items")
    if not isinstance(items, list):
        raise PathBV2ValidationError(f"evidence package items are unavailable for {task_id}")
    refs: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise PathBV2ValidationError(
                f"evidence package item {index} is not reviewable for {task_id}"
            )
        refs.append({field: item.get(field) for field in _PACKAGE_REVIEW_FIELDS})
    return refs


def _build_reviewer_export_rows(
    ledger_rows: list[dict[str, str]],
    packages: list[Mapping[str, Any]],
    roster: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project only human-verifiable pre-score evidence into the export."""

    ledger_by_task = {row["task_id"]: row for row in ledger_rows}
    package_by_task: dict[str, Mapping[str, Any]] = {}
    for index, package in enumerate(packages):
        task_id = _optional_text(package.get("task_id"))
        if task_id is None:
            raise PathBV2ValidationError(f"evidence package {index} has no task_id")
        if task_id in package_by_task:
            raise PathBV2ValidationError(f"duplicate evidence package {task_id}")
        package_by_task[task_id] = package

    exported: list[dict[str, Any]] = []
    for index, roster_row in enumerate(roster["rows"]):
        task_id = roster_row["task_id"]
        ledger_row = ledger_by_task.get(task_id)
        package = package_by_task.get(task_id)
        if ledger_row is None or package is None:
            raise PathBV2ValidationError(
                f"reviewer export source is incomplete for roster task {task_id}"
            )
        refs = _package_review_refs(package, task_id)
        published_values = sorted(
            {
                value
                for value in (_optional_text(ref.get("published_at")) for ref in refs)
                if value is not None
            }
        )
        source_urls = sorted(
            {
                value
                for value in (_optional_text(ref.get("source_url")) for ref in refs)
                if value is not None
            }
        )
        snapshot_refs = [
            {
                "snapshot_path": _optional_text(ref.get("snapshot_path")),
                "snapshot_sha256": _optional_text(ref.get("snapshot_sha256")),
            }
            for ref in refs
            if ref.get("snapshot_path") is not None or ref.get("snapshot_sha256") is not None
        ]
        official_time_evidence = {
            "candidate_source_urls": source_urls,
            "candidate_snapshot_refs": snapshot_refs,
            "published_at_values": published_values,
            "status": "requires_human_officiality_confirmation",
        }
        stratum = {
            field: roster_row[field]
            for field in (
                "category",
                "time_band",
                "task_family",
                "paper_band",
                "outcome_type",
            )
        }
        content_summary = {
            "format": "path_b_v2_task_id_items_v1",
            "hash_algorithm": "sha256(canonical_json(items))",
            "item_count": len(refs),
            "item_kinds": [ref.get("kind") for ref in refs],
            "item_refs": refs,
            "source_url_count": len(source_urls),
            "snapshot_ref_count": len(snapshot_refs),
            "published_at_count": len(published_values),
        }
        row = {
            "task_id": task_id,
            "category": roster_row["category"],
            "time_band": roster_row["time_band"],
            "task_family": roster_row["task_family"],
            "paper_band": roster_row["paper_band"],
            "outcome_type": roster_row["outcome_type"],
            "stratum": stratum,
            "review_status": _optional_text(ledger_row.get("review_status")),
            "reviewed_at": _optional_text(ledger_row.get("reviewed_at")),
            "official_temporal_eligible": _explicit_boolean(
                ledger_row.get("official_temporal_eligible"),
                f"ledger row {index}.official_temporal_eligible",
            ),
            "official_time_evidence": official_time_evidence,
            "official_source_urls": source_urls,
            "official_snapshot_refs": snapshot_refs,
            "published_at": published_values[0] if len(published_values) == 1 else None,
            "published_at_candidates": published_values,
            "evidence_status": {
                "event": _optional_text(ledger_row.get("event_evidence_status")),
                "price": _optional_text(ledger_row.get("price_evidence_status")),
            },
            "evidence_package_content_summary": content_summary,
            "evidence_package_sha256": roster_row["evidence_package_sha256"],
        }
        assert_path_b_v2_score_blind(row, path=f"$.reviewer_export[{index}]")
        exported.append(row)
    return exported


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PathBV2ValidationError(f"unable to hash source file: {path}") from exc


def _source_lock(
    *,
    contract: Mapping[str, Any],
    ledger_path: Path,
    packages_path: Path,
    ledger_count: int,
    package_count: int,
    roster_count: int,
) -> dict[str, Any]:
    body = {
        "schema_version": PATH_B_V3_SOURCE_LOCK_SCHEMA,
        "dataset_lock": dict(contract["dataset_lock"]),
        "review_contract_sha256": sha256_json(contract),
        "source_files": {
            str(_LEDGER_RELATIVE_PATH): _sha256_file(ledger_path),
            str(_PACKAGES_RELATIVE_PATH): _sha256_file(packages_path),
        },
        "ledger_row_count": ledger_count,
        "package_row_count": package_count,
        "roster_row_count": roster_count,
        "package_format": "path_b_v2_task_id_items_v1",
        "package_hash": "sha256(canonical_json(items))",
        "time_band_source": "paper_band",
        "review_boundary": "score_blind",
    }
    return {**body, "source_lock_sha256": sha256_json(body)}


def _review_instructions() -> str:
    return """# Path B v3 blinded review handoff

State: `pending_independent_attestation`.

This pack is bound to the current Path B v2 dataset pin. The sealed roster
contains collector identity for the later self-sign check, while the blinded
review export contains only task identity, frozen strata, review status,
official-time-evidence URL/snapshot references, explicit publication dates,
and an evidence-package content summary. It contains no predictions, scores,
results, or evaluator output. Do not add any of those fields to a review
artifact.
The export is a v2 reviewer schema; a missing URL, snapshot, or published date
is preserved as missing and must be treated as unverifiable, not repaired by
invention. The export references are candidates for officiality and require
the reviewer's source check before an eligible decision.
The 13 `B-MACRO` rows are retained for audit completeness only and are
excluded from Path B eligibility, T1/T2, and calibration; no row is presumed
eligible merely because it appears in this roster.

The prior reviewer export with SHA-256
`ea4615c7317360a67704f51f5eadd836bdf84584fcbb8628015bb9eda76c141c` is
superseded and invalidated. The QA import with SHA-256
`a2485bc4548e91bb53798d63c2f0f331822cd49b671a712b871d695a9466c8b0` and any
CR output based on that export are invalidated as well; neither may enter
deterministic adjudication.

CR and QA must review independently. The collector must not attest their own
row. Each reviewer must write one JSON object per reviewed task using exactly
these eight fields: `task_id`, `decision`, `reviewer_id`, `review_method`,
`reviewed_at`, `evidence_package_sha256`, `notes`, and `attestation_sha256`.
Use one of the four methods frozen in the review contract. A row that cannot
be verified is `ineligible`; do not invent a publication date or replace an
observed date with a modeled date.

Keep CR and QA imports separate until deterministic adjudication. After both
imports cover the same roster, run the v3 adjudicator and then construct the
v2 eligibility/package bundle for `path-b-v2-validate`. No cohort, scored plan,
or experiment may be generated from this pending pack.
"""


def prepare_path_b_v3_review_pack(
    source_root: str | Path,
    output: str | Path,
    *,
    contract_path: str | Path = "configs/path-b-v2-review-contract.json",
) -> dict[str, Any]:
    """Generate the current-pin roster and collector-free reviewer export.

    The source root is expected to be an offline copy of the ``path_b_v2``
    directory at the locked Hub revision. Only the review ledger and the
    score-blind evidence-package index are read; task JSONL files and all
    evaluator-owned fields are outside this preparation boundary.
    """

    source = Path(source_root).expanduser().resolve()
    if not source.is_dir():
        raise PathBV2ValidationError(f"Path B source root is not a directory: {source}")
    output_path = Path(output).expanduser().resolve()
    try:
        output_path.relative_to(source)
    except ValueError:
        pass
    else:
        raise PathBV2ValidationError("review-pack output must be outside the protected source root")

    contract_file = Path(contract_path).expanduser().resolve()
    try:
        contract = json.loads(contract_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PathBV2ValidationError(f"unable to read review contract: {contract_file}") from exc
    contract = validate_review_contract(contract)

    ledger_path = source / _LEDGER_RELATIVE_PATH
    packages_path = source / _PACKAGES_RELATIVE_PATH
    ledger_rows = _read_ledger(ledger_path)
    packages = _read_packages(packages_path)
    roster = build_path_b_v2_canonical_roster(
        ledger_rows,
        packages,
        review_contract=contract,
    )
    reviewer_rows = _build_reviewer_export_rows(ledger_rows, packages, roster)
    source_lock = _source_lock(
        contract=contract,
        ledger_path=ledger_path,
        packages_path=packages_path,
        ledger_count=len(ledger_rows),
        package_count=len(packages),
        roster_count=roster["roster_count"],
    )
    result = write_sealed_review_pack(
        str(output_path),
        roster,
        source_lock=source_lock,
        review_instructions=_review_instructions(),
        reviewer_rows=reviewer_rows,
        supersedes={
            "reviewer_export_sha256": PATH_B_V3_SUPERSEDED_REVIEWER_EXPORT_SHA256,
            "reviewer_export_file_sha256": PATH_B_V3_SUPERSEDED_REVIEWER_EXPORT_SHA256,
            "qa_import_sha256": PATH_B_V3_INVALIDATED_QA_IMPORT_SHA256,
            "disposition": "invalidated_not_for_adjudication",
        },
    )
    return {
        **result,
        "ledger_count": len(ledger_rows),
        "package_count": len(packages),
        "contract_sha256": sha256_json(contract),
        "dataset_revision": contract["dataset_lock"]["revision"],
        "dataset_tree_sha": contract["dataset_lock"]["tree_sha"],
        "superseded_reviewer_export_sha256": PATH_B_V3_SUPERSEDED_REVIEWER_EXPORT_SHA256,
        "invalidated_qa_import_sha256": PATH_B_V3_INVALIDATED_QA_IMPORT_SHA256,
    }


__all__ = [
    "PATH_B_V3_INVALIDATED_QA_IMPORT_SHA256",
    "PATH_B_V3_SUPERSEDED_REVIEWER_EXPORT_SHA256",
    "prepare_path_b_v3_review_pack",
]
