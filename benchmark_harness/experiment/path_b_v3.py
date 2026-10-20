"""Score-blind Path B v3 sealing and analysis-preparation contracts.

This additive module prepares review, leakage-probe, T1/T2, and B-calibration
artifacts.  It never reads outcomes, predictions, score artifacts, Hub caches,
or the frozen evaluator.  Scored execution is deliberately outside this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from benchmark_harness.experiment.path_b_v2 import (
    PATH_B_V2_REVIEW_METHOD_SET,
    PathBV2ValidationError,
    assert_path_b_v2_score_blind,
    canonical_json,
    compute_attestation_sha256,
    sha256_json,
    validate_review_contract,
)

PATH_B_V3_REVIEW_SCHEMA = "path-b-v3-review-pack/v1"
PATH_B_V3_SOURCE_LOCK_SCHEMA = "path-b-v3-source-lock/v1"
PATH_B_V3_REVIEWER_EXPORT_SCHEMA = "path-b-v3-reviewer-export/v2"
PATH_B_V3_REVIEWER_EXPORT_REQUIRED_FIELDS = frozenset(
    {
        "task_id",
        "category",
        "time_band",
        "task_family",
        "paper_band",
        "outcome_type",
        "stratum",
        "review_status",
        "official_temporal_eligible",
        "official_time_evidence",
        "official_source_urls",
        "official_snapshot_refs",
        "published_at",
        "published_at_candidates",
        "evidence_package_content_summary",
        "evidence_package_sha256",
    }
)
T1_T2_ANALYSIS_SCHEMA = "t1-t2-analysis-spec/v1"
B_CALIBRATION_SCHEMA = "b-calibration-spec/v1"
LEAKAGE_PROBE_SCHEMA = "leakage-probe-plan/v3"

# The current Path B ledger does not carry explicit task_family/outcome_type
# columns.  These are the frozen, score-blind strata derived from its
# category/variant identity fields.  In particular, this mapping never looks
# at prompt, expected-output, ground-truth, or evaluator-owned fields.
_PATH_B_V2_STRATA = {
    "A1": ("temporal_price_forecast", "multi_horizon_price_band"),
    "A2-T": ("temporal_ranking_forecast", "cross_sectional_ranking"),
    "B-EARNINGS": ("earnings_event_reaction", "binary_direction_probability"),
    "B-MACRO": ("macro_event_reaction", "binary_direction_probability"),
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PathBV2ValidationError(f"{name} must be an object")
    return value


def _hash_without(value: Mapping[str, Any], field: str) -> str:
    copied = dict(value)
    copied.pop(field, None)
    return sha256_json(copied)


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PathBV2ValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _sha(value: Any, name: str) -> str:
    parsed = _nonempty(value, name)
    if len(parsed) != 64 or any(char not in "0123456789abcdef" for char in parsed):
        raise PathBV2ValidationError(f"{name} must be a lowercase SHA-256")
    return parsed


def _assert_no_collector_identity(value: Any, *, path: str = "$") -> None:
    """Reject collector identity from every reviewer-facing record."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = "".join(char if char.isalnum() else "_" for char in str(key).casefold())
            if normalized.startswith("collector"):
                raise PathBV2ValidationError(f"collector identity is forbidden at {path}.{key}")
            _assert_no_collector_identity(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            _assert_no_collector_identity(child, path=f"{path}[{index}]")


def _reviewer_export_rows(
    reviewer_rows: Sequence[Mapping[str, Any]],
    queue_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the richer, score-blind export used by CR and QA."""

    if isinstance(reviewer_rows, str | bytes | bytearray):
        raise PathBV2ValidationError("reviewer_rows must be an array")
    exported: list[dict[str, Any]] = []
    for index, row in enumerate(reviewer_rows):
        if not isinstance(row, Mapping):
            raise PathBV2ValidationError(f"reviewer export row {index} must be an object")
        exported.append(dict(row))
    if len(exported) != len(queue_rows):
        raise PathBV2ValidationError("reviewer export must cover the complete roster")
    expected_ids = [row["task_id"] for row in queue_rows]
    actual_ids = [row.get("task_id") for row in exported]
    if actual_ids != expected_ids:
        raise PathBV2ValidationError("reviewer export task order must match the sealed queue")
    for index, (row, queue_row) in enumerate(zip(exported, queue_rows, strict=True)):
        missing = sorted(PATH_B_V3_REVIEWER_EXPORT_REQUIRED_FIELDS - row.keys())
        if missing:
            raise PathBV2ValidationError(
                f"reviewer export row {index} is missing required fields: {missing}"
            )
        if row["evidence_package_sha256"] != queue_row["evidence_package_sha256"]:
            raise PathBV2ValidationError(
                f"reviewer export has a stale evidence package hash for {queue_row['task_id']}"
            )
        if row["task_id"] != queue_row["task_id"]:
            raise PathBV2ValidationError("reviewer export task identity does not match the queue")
    assert_path_b_v2_score_blind(exported, path="$.reviewer_export")
    _assert_no_collector_identity(exported, path="$.reviewer_export")
    return exported


def _package_task_id_and_hash(package: Mapping[str, Any], index: int) -> tuple[str, str]:
    """Read either the v2 component package or the pinned source package form.

    The current Hub ``evidence_packages_v2.jsonl`` stores a task id, an
    ordered ``items`` list, and its hash over that list.  The local v2
    validator consumes the later four-component package contract.  The
    score-blind roster needs only bind the source package identity, so it
    accepts both forms without rewriting or weakening either contract.
    """

    package = _mapping(package, f"evidence package {index}")
    assert_path_b_v2_score_blind(package, path=f"$.packages[{index}]")
    claimed_hash = _sha(
        package.get("evidence_package_sha256"),
        f"evidence package {index}.evidence_package_sha256",
    )
    if "task_identity" in package:
        identity = _mapping(package.get("task_identity"), f"evidence package {index}.task_identity")
        task_id = _nonempty(
            identity.get("task_id"), f"evidence package {index}.task_identity.task_id"
        )
        computed_hash = _hash_without(package, "evidence_package_sha256")
    else:
        if set(package) != {"task_id", "items", "evidence_package_sha256"}:
            raise PathBV2ValidationError(
                f"evidence package {index} is neither a component package nor "
                "the pinned source form"
            )
        task_id = _nonempty(package.get("task_id"), f"evidence package {index}.task_id")
        items = package.get("items")
        if not isinstance(items, list):
            raise PathBV2ValidationError(f"evidence package {index}.items must be an array")
        computed_hash = sha256_json(items)
    if claimed_hash != computed_hash:
        raise PathBV2ValidationError(f"evidence package hash mismatch for {task_id}")
    return task_id, claimed_hash


def _path_b_v2_category(category: Any, variant: Any, name: str) -> str:
    raw_category = _nonempty(category, f"{name}.category").upper().replace("_", "-")
    raw_variant = str(variant or "").strip().upper().replace("_", "-")
    if raw_category == "A1" and not raw_variant:
        return "A1"
    if raw_category == "A2" and raw_variant == "T":
        return "A2-T"
    if raw_category == "B" and raw_variant in {"EARNINGS", "B-EARNINGS"}:
        return "B-EARNINGS"
    if raw_category == "B" and raw_variant in {"MACRO", "B-MACRO"}:
        return "B-MACRO"
    raise PathBV2ValidationError(
        f"{name} has an unsupported Path B category/variant: {category!r}/{variant!r}"
    )


def build_path_b_v2_roster_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Project the pinned 543-row ledger into score-blind roster rows.

    Only identity and pre-score strata cross this boundary.  The source
    ledger's review/eligibility fields are intentionally not copied; those
    decisions must be made independently from the blinded export.
    """

    roster_rows: list[dict[str, str]] = []
    for index, source in enumerate(rows):
        source = _mapping(source, f"Path B ledger row {index}")
        task_id = _nonempty(source.get("task_id"), f"Path B ledger row {index}.task_id")
        category = _path_b_v2_category(
            source.get("category"), source.get("variant"), f"Path B ledger row {index}"
        )
        # The pinned ledger exposes paper_band but not a second time_band
        # column; for this source contract they are the same frozen temporal
        # partition.  Callers may still provide time_band explicitly.
        time_band = _nonempty(
            source.get("time_band") or source.get("paper_band"),
            f"Path B ledger row {index}.time_band",
        )
        paper_band = _nonempty(source.get("paper_band"), f"Path B ledger row {index}.paper_band")
        if time_band not in {"T1", "T2"} or paper_band not in {"T1", "T2"}:
            raise PathBV2ValidationError(
                f"Path B ledger row {index} has an unsupported time/paper band"
            )
        _nonempty(source.get("scope_role"), f"Path B ledger row {index}.scope_role")
        collector = _nonempty(source.get("collector_id"), f"Path B ledger row {index}.collector_id")
        package_hash = _sha(
            source.get("evidence_package_sha256"),
            f"Path B ledger row {index}.evidence_package_sha256",
        )
        task_family, outcome_type = _PATH_B_V2_STRATA[category]
        projected = {
            "task_id": task_id,
            "category": category,
            "time_band": time_band,
            "task_family": task_family,
            "paper_band": paper_band,
            "outcome_type": outcome_type,
            "collector_id": collector,
            "evidence_package_sha256": package_hash,
        }
        assert_path_b_v2_score_blind(projected, path=f"$.roster_rows[{index}]")
        roster_rows.append(projected)
    if len({row["task_id"] for row in roster_rows}) != len(roster_rows):
        raise PathBV2ValidationError("Path B ledger contains duplicate task IDs")
    return roster_rows


def build_path_b_v2_canonical_roster(
    ledger_rows: Sequence[Mapping[str, Any]],
    packages: Sequence[Mapping[str, Any]],
    *,
    review_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a roster directly from the current Hub v2 ledger/package files."""

    projected_rows = build_path_b_v2_roster_rows(ledger_rows)
    package_ids: set[str] = set()
    for index, package in enumerate(packages):
        task_id, _package_hash = _package_task_id_and_hash(package, index)
        package_ids.add(task_id)
    row_ids = {row["task_id"] for row in projected_rows}
    if package_ids != row_ids:
        missing = sorted(row_ids - package_ids)
        extra = sorted(package_ids - row_ids)
        raise PathBV2ValidationError(
            f"ledger/package task identities differ (missing={missing[:3]}, extra={extra[:3]})"
        )
    return build_canonical_roster(projected_rows, packages, review_contract=review_contract)


def build_canonical_roster(
    rows: Sequence[Mapping[str, Any]],
    packages: Sequence[Mapping[str, Any]],
    *,
    review_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a score-blind roster bound to current evidence-package identities."""

    contract = validate_review_contract(review_contract)
    package_hashes: dict[str, str] = {}
    for index, package in enumerate(packages):
        task_id, claimed_hash = _package_task_id_and_hash(package, index)
        if task_id in package_hashes:
            raise PathBV2ValidationError(f"duplicate evidence package {task_id}")
        package_hashes[task_id] = claimed_hash
    roster_rows: list[dict[str, str]] = []
    for row in rows:
        assert_path_b_v2_score_blind(row)
        task_id = _nonempty(row.get("task_id"), "roster task_id")
        collector = _nonempty(row.get("collector_id"), f"roster collector_id for {task_id}")
        package_hash = _sha(
            row.get("evidence_package_sha256"), f"roster package hash for {task_id}"
        )
        if package_hashes.get(task_id) != package_hash:
            raise PathBV2ValidationError(f"roster package is stale or absent for {task_id}")
        roster_rows.append(
            {
                "task_id": task_id,
                "category": _nonempty(row.get("category"), f"roster category for {task_id}"),
                "time_band": _nonempty(row.get("time_band"), f"roster time_band for {task_id}"),
                "task_family": _nonempty(
                    row.get("task_family"), f"roster task_family for {task_id}"
                ),
                "paper_band": _nonempty(row.get("paper_band"), f"roster paper_band for {task_id}"),
                "outcome_type": _nonempty(
                    row.get("outcome_type"), f"roster outcome_type for {task_id}"
                ),
                "collector_id": collector,
                "evidence_package_sha256": package_hash,
            }
        )
    roster_rows.sort(key=lambda row: row["task_id"])
    if len({row["task_id"] for row in roster_rows}) != len(roster_rows):
        raise PathBV2ValidationError("canonical roster has duplicate task IDs")
    body = {
        "schema_version": PATH_B_V3_REVIEW_SCHEMA,
        "review_contract_sha256": sha256_json(contract),
        "roster_count": len(roster_rows),
        "rows": roster_rows,
    }
    return {**body, "roster_sha256": sha256_json(body)}


def import_blinded_attestations(
    attestations: Sequence[Mapping[str, Any]],
    roster: Mapping[str, Any],
    *,
    review_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate independent, current-hash reviewer decisions without scores."""

    contract = validate_review_contract(review_contract)
    assert_path_b_v2_score_blind(roster)
    if roster.get("review_contract_sha256") != sha256_json(contract):
        raise PathBV2ValidationError("roster is bound to a different review contract")
    if roster.get("roster_sha256") != _hash_without(roster, "roster_sha256"):
        raise PathBV2ValidationError("roster hash mismatch")
    roster_rows = _mapping({row["task_id"]: row for row in roster.get("rows", [])}, "roster rows")
    imported: list[dict[str, Any]] = []
    for value in attestations:
        assert_path_b_v2_score_blind(value)
        task_id = _nonempty(value.get("task_id"), "attestation.task_id")
        row = roster_rows.get(task_id)
        if row is None:
            raise PathBV2ValidationError(f"attestation task is not in roster: {task_id}")
        if value.get("evidence_package_sha256") != row["evidence_package_sha256"]:
            raise PathBV2ValidationError(f"stale evidence package hash for {task_id}")
        reviewer = _nonempty(value.get("reviewer_id"), "attestation.reviewer_id")
        if reviewer == row["collector_id"]:
            raise PathBV2ValidationError(f"reviewer must be independent of collector for {task_id}")
        if value.get("review_method") not in PATH_B_V2_REVIEW_METHOD_SET:
            raise PathBV2ValidationError(f"attestation uses non-human review method for {task_id}")
        if value.get("decision") not in {"eligible", "ineligible"}:
            raise PathBV2ValidationError(f"attestation decision invalid for {task_id}")
        if value.get("attestation_sha256") != compute_attestation_sha256(value):
            raise PathBV2ValidationError(f"attestation hash mismatch for {task_id}")
        imported.append(dict(value))
    if len({item["task_id"] for item in imported}) != len(imported):
        raise PathBV2ValidationError("one blinded import must contain one decision per task")
    return sorted(imported, key=lambda item: item["task_id"])


def render_adjudication_report(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    roster: Mapping[str, Any],
    *,
    review_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce deterministic dual-review adjudication; disagreements stay sealed."""

    first_rows = import_blinded_attestations(first, roster, review_contract=review_contract)
    second_rows = import_blinded_attestations(second, roster, review_contract=review_contract)
    left = {row["task_id"]: row for row in first_rows}
    right = {row["task_id"]: row for row in second_rows}
    if set(left) != set(right):
        raise PathBV2ValidationError("reviewer imports must cover the same roster tasks")
    adjudicated: list[dict[str, Any]] = []
    disagreements: list[str] = []
    for task_id in sorted(left):
        if left[task_id]["reviewer_id"] == right[task_id]["reviewer_id"]:
            raise PathBV2ValidationError(f"reviewers must be independent for {task_id}")
        if left[task_id]["decision"] != right[task_id]["decision"]:
            disagreements.append(task_id)
            continue
        adjudicated.append(
            {
                "task_id": task_id,
                "decision": left[task_id]["decision"],
                "reviewer_attestation_sha256s": sorted(
                    [left[task_id]["attestation_sha256"], right[task_id]["attestation_sha256"]]
                ),
            }
        )
    body = {
        "schema_version": "path-b-v3-adjudication/v1",
        "roster_sha256": roster["roster_sha256"],
        "adjudicated": adjudicated,
        "unresolved_disagreement_task_ids": disagreements,
        "sealed": not disagreements,
    }
    return {**body, "adjudication_sha256": sha256_json(body)}


def write_sealed_review_pack(
    output: str,
    roster: Mapping[str, Any],
    *,
    source_lock: Mapping[str, Any] | None = None,
    review_instructions: str | None = None,
    reviewer_rows: Sequence[Mapping[str, Any]] | None = None,
    supersedes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an immutable score-blind roster and reviewer queue exactly once.

    ``source_lock`` and ``review_instructions`` are used by the current-pin
    preparation command.  ``reviewer_rows`` is the richer v2 reviewer export:
    it exposes only pre-score evidence needed for a human eligibility review,
    while the stable queue remains identity-only.  The resulting pack is
    immutable but remains explicitly ``pending_independent_attestation`` until
    CR and QA provide their separate imports.
    """

    from pathlib import Path

    assert_path_b_v2_score_blind(roster)
    if roster.get("roster_sha256") != _hash_without(roster, "roster_sha256"):
        raise PathBV2ValidationError("roster hash mismatch")
    source_lock_hash: str | None = None
    if source_lock is not None:
        assert_path_b_v2_score_blind(source_lock, path="$.source_lock")
        source_lock_hash = _sha(
            source_lock.get("source_lock_sha256"), "source_lock.source_lock_sha256"
        )
        if source_lock_hash != _hash_without(source_lock, "source_lock_sha256"):
            raise PathBV2ValidationError("source lock hash mismatch")
    if review_instructions is not None:
        if not isinstance(review_instructions, str) or not review_instructions.strip():
            raise PathBV2ValidationError("review_instructions must be a non-empty string")
    if supersedes is not None:
        supersedes = dict(supersedes)
        assert_path_b_v2_score_blind(supersedes, path="$.supersedes")
        _assert_no_collector_identity(supersedes, path="$.supersedes")
    queue_rows = [
        {
            "task_id": row["task_id"],
            "category": row["category"],
            "time_band": row["time_band"],
            "task_family": row["task_family"],
            "paper_band": row["paper_band"],
            "outcome_type": row["outcome_type"],
            "evidence_package_sha256": row["evidence_package_sha256"],
        }
        for row in roster["rows"]
    ]
    reviewer_export_rows = (
        _reviewer_export_rows(reviewer_rows, queue_rows)
        if reviewer_rows is not None
        else [dict(row) for row in queue_rows]
    )
    root = Path(output)
    if root.exists():
        raise PathBV2ValidationError("sealed review-pack output must not already exist")
    root.mkdir(parents=True)
    roster_path = root / "canonical-roster.json"
    queue_path = root / "blinded-review-queue.jsonl"
    export_path = root / "blinded-review-export.jsonl"
    attestations_path = root / "attestations.jsonl"
    roster_path.write_text(canonical_json(roster) + "\n", encoding="utf-8")
    queue_text = "".join(canonical_json(row) + "\n" for row in queue_rows)
    queue_path.write_text(queue_text, encoding="utf-8")
    # Keep the stable queue identity-only while the descriptive export carries
    # the evidence fields required for a blinded human decision.
    export_text = "".join(canonical_json(row) + "\n" for row in reviewer_export_rows)
    export_path.write_text(export_text, encoding="utf-8")
    attestations_path.write_text("", encoding="utf-8")

    if source_lock is not None:
        (root / "source-lock.json").write_text(canonical_json(source_lock) + "\n", encoding="utf-8")
    if review_instructions is not None:
        (root / "review-instructions.md").write_text(review_instructions, encoding="utf-8")

    review_state = "pending_independent_attestation" if source_lock is not None else "sealed"
    manifest_body = {
        "schema_version": "path-b-v3-sealed-review-pack/v1",
        "roster_sha256": roster["roster_sha256"],
        "review_queue_count": len(queue_rows),
        "review_queue_sha256": sha256_json(queue_rows),
        "reviewer_export_schema": (
            PATH_B_V3_REVIEWER_EXPORT_SCHEMA
            if reviewer_rows is not None
            else "path-b-v3-reviewer-export/v1"
        ),
        "reviewer_export_count": len(reviewer_export_rows),
        "reviewer_export_sha256": sha256_json(reviewer_export_rows),
        "reviewer_export_file_sha256": hashlib.sha256(export_text.encode("utf-8")).hexdigest(),
        "attestations_filename": attestations_path.name,
        "attestation_count": 0,
        "review_state": review_state,
    }
    if source_lock is not None:
        manifest_body["source_lock_sha256"] = source_lock_hash
    if review_instructions is not None:
        manifest_body["review_instructions_sha256"] = sha256_json(review_instructions)
    if supersedes is not None:
        manifest_body["supersedes"] = supersedes
    manifest = {**manifest_body, "manifest_sha256": sha256_json(manifest_body)}
    manifest_path = root / "sealed-review-pack-manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "root": str(root),
        "roster_sha256": roster["roster_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "review_state": review_state,
        "review_queue_count": len(queue_rows),
        "reviewer_export_count": len(reviewer_export_rows),
        "reviewer_export_sha256": manifest["reviewer_export_sha256"],
        "reviewer_export_file_sha256": manifest["reviewer_export_file_sha256"],
    }
    if source_lock is not None:
        result["source_lock_sha256"] = source_lock_hash
    return result


def build_leakage_probe_plan(
    pair_rows: Sequence[Mapping[str, Any]], *, results_root: str
) -> dict[str, Any]:
    """Build a paired cutoff/unrestricted plan with isolated non-ranking output."""

    assert_path_b_v2_score_blind(pair_rows)
    root = _nonempty(results_root, "results_root")
    if "leakage-probe" not in root or root.rstrip("/").endswith("/v2"):
        raise PathBV2ValidationError(
            "unrestricted leakage probe results root must be physically isolated"
        )
    cells: list[dict[str, Any]] = []
    for row in pair_rows:
        task_id = _nonempty(row.get("task_id"), "probe task_id")
        system_id = _nonempty(row.get("system_id"), "probe system_id")
        trial = row.get("trial")
        if not isinstance(trial, int) or isinstance(trial, bool) or trial < 1:
            raise PathBV2ValidationError("probe trial must be a positive integer")
        budget_hash = _sha(row.get("budget_sha256"), "probe budget_sha256")
        seed = row.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise PathBV2ValidationError("probe seed must be an integer")
        for regime in ("cutoff_controlled", "unrestricted_current_web"):
            cells.append(
                {
                    "task_id": task_id,
                    "system_id": system_id,
                    "trial": trial,
                    "budget_sha256": budget_hash,
                    "seed": seed,
                    "retrieval_regime": regime,
                    "ranking_eligible": False,
                    "cutoff_filter_enabled": regime == "cutoff_controlled",
                    "policy_identity": "cutoff_controlled/v1"
                    if regime == "cutoff_controlled"
                    else "leakage_probe/v1",
                }
            )
    body = {
        "schema_version": LEAKAGE_PROBE_SCHEMA,
        "results_root": root,
        "ranking_eligible": False,
        "cells": sorted(
            cells, key=lambda x: (x["task_id"], x["system_id"], x["trial"], x["retrieval_regime"])
        ),
    }
    return {**body, "plan_sha256": sha256_json(body)}


def build_t1_t2_analysis_spec() -> dict[str, Any]:
    body = {
        "schema_version": T1_T2_ANALYSIS_SCHEMA,
        "stratum_keys": ["category", "task_family", "paper_band", "outcome_type"],
        "matching": {"method": "deterministic_1_to_1", "sort_key": "sha256(task_id)"},
        "weighting": {"method": "inverse_stratum_frequency"},
        "excluded_categories": ["C", "B-MACRO"],
    }
    return {**body, "spec_sha256": sha256_json(body)}


def materialize_t1_t2_cohort(
    rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]
) -> dict[str, Any]:
    assert_path_b_v2_score_blind(rows)
    if spec.get("spec_sha256") != _hash_without(spec, "spec_sha256"):
        raise PathBV2ValidationError("t1-t2 analysis spec hash mismatch")
    if (
        spec.get("schema_version") != T1_T2_ANALYSIS_SCHEMA
        or spec.get("weighting", {}).get("method") != "inverse_stratum_frequency"
    ):
        raise PathBV2ValidationError("t1-t2 analysis spec is unsupported")
    keys = spec["stratum_keys"]
    groups: dict[tuple[str, ...], dict[str, list[Mapping[str, Any]]]] = {}
    for row in rows:
        if row.get("category") in set(spec["excluded_categories"]):
            continue
        if row.get("eligibility", {}).get("eligible") is not True:
            continue
        band = row.get("time_band")
        if band not in {"T1", "T2"}:
            continue
        stratum = tuple(_nonempty(row.get(key), key) for key in keys)
        groups.setdefault(stratum, {"T1": [], "T2": []})[band].append(row)
    pairs: list[dict[str, Any]] = []
    for stratum, bands in sorted(groups.items()):
        left = sorted(bands["T1"], key=lambda row: sha256_json(row["task_id"]))
        right = sorted(bands["T2"], key=lambda row: sha256_json(row["task_id"]))
        for t1, t2 in zip(left, right, strict=False):
            pairs.append(
                {
                    "stratum": list(stratum),
                    "t1_task_id": t1["task_id"],
                    "t2_task_id": t2["task_id"],
                    "weight": 1.0 / len(left),
                }
            )
    body = {"schema_version": "t1-t2-cohort/v1", "spec_sha256": spec["spec_sha256"], "pairs": pairs}
    return {**body, "cohort_sha256": sha256_json(body)}


def build_b_calibration_spec() -> dict[str, Any]:
    body = {
        "schema_version": B_CALIBRATION_SCHEMA,
        "bins": [round(index / 10, 1) for index in range(11)],
        "baselines": ["constant_0.5", "frozen_historical_rate"],
        "population": "available_case_b_earnings",
        "excluded_categories": ["B-MACRO"],
    }
    return {**body, "spec_sha256": sha256_json(body)}


def materialize_b_calibration_report(
    rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]
) -> dict[str, Any]:
    assert_path_b_v2_score_blind(rows)
    if (
        spec.get("spec_sha256") != _hash_without(spec, "spec_sha256")
        or spec.get("schema_version") != B_CALIBRATION_SCHEMA
    ):
        raise PathBV2ValidationError("b calibration spec hash mismatch")
    exclusions: dict[str, str] = {}
    included: list[str] = []
    for row in rows:
        task_id = _nonempty(row.get("task_id"), "calibration task_id")
        if row.get("category") == "B-MACRO":
            exclusions[task_id] = "b_macro_out_of_scope"
        elif row.get("event_chain") == "missing_bls":
            exclusions[task_id] = "missing_bls_first_party_evidence"
        elif row.get("temporal_order") == "inverted":
            exclusions[task_id] = "event_outcome_time_inversion"
        elif row.get("rationale") != "consistent":
            exclusions[task_id] = "contradictory_rationale_requires_overlay_resign"
        elif (
            row.get("category") != "B-EARNINGS"
            or row.get("event_chain") != "complete"
            or row.get("price_chain") != "complete"
        ):
            exclusions[task_id] = "incomplete_available_case_evidence"
        else:
            included.append(task_id)
    body = {
        "schema_version": "b-calibration-report/v1",
        "spec_sha256": spec["spec_sha256"],
        "available_case_task_ids": sorted(included),
        "exclusions": dict(sorted(exclusions.items())),
        "complete_calibration": False,
        "score_blind": True,
    }
    return {**body, "report_sha256": sha256_json(body)}


__all__ = [
    "B_CALIBRATION_SCHEMA",
    "LEAKAGE_PROBE_SCHEMA",
    "PATH_B_V3_REVIEW_SCHEMA",
    "PATH_B_V3_REVIEWER_EXPORT_REQUIRED_FIELDS",
    "PATH_B_V3_REVIEWER_EXPORT_SCHEMA",
    "PATH_B_V3_SOURCE_LOCK_SCHEMA",
    "T1_T2_ANALYSIS_SCHEMA",
    "build_b_calibration_spec",
    "build_canonical_roster",
    "build_path_b_v2_canonical_roster",
    "build_path_b_v2_roster_rows",
    "build_leakage_probe_plan",
    "build_t1_t2_analysis_spec",
    "import_blinded_attestations",
    "materialize_b_calibration_report",
    "materialize_t1_t2_cohort",
    "render_adjudication_report",
    "write_sealed_review_pack",
]
