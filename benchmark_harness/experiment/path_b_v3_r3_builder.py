"""Build the append-only Path B v3 r3 component review pack.

The pinned Hub export predates the four-component Path B v2 package contract:
it contains an ``items`` index and a hash over that index.  This module makes
that boundary explicit.  It reads only the score-blind calibration tables,
derives a new four-component package for every ledger row, and records which
derived packages can already satisfy the frozen validator.  Missing evidence is
preserved and fails closed; no task content, forecast, outcome value, or
evaluator-owned field is copied into the pack.

The resulting r3 pack is a pending review artifact, not a scored bundle.  The
roster is deliberately rebound to the new component-package hashes so a later
attestation cannot accidentally bind to the superseded source-index hashes.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark_harness.experiment.path_b_v2 import (
    PATH_B_V2_EVIDENCE_COMPONENTS,
    PathBV2ValidationError,
    assert_path_b_v2_score_blind,
    canonical_json,
    compute_evidence_package_sha256,
    sha256_json,
    validate_evidence_package,
    validate_review_contract,
)
from benchmark_harness.experiment.path_b_v3 import (
    PATH_B_V3_REVIEWER_EXPORT_REQUIRED_FIELDS,
    _reviewer_export_rows,
    build_path_b_v2_canonical_roster,
)

_LEDGER_RELATIVE_PATH = Path("calibration/review_ledger_v2.csv")
_PACKAGES_RELATIVE_PATH = Path("calibration/evidence_packages_v2.jsonl")
_TEMPORAL_RELATIVE_PATH = Path("calibration/temporal_provenance_table.csv")
_EVENT_RELATIVE_PATH = Path("calibration/b_event_evidence_table.csv")

PATH_B_V3_R3_REVIEW_SCHEMA = "path-b-v3-r3-review-pack/v1"
PATH_B_V3_R3_SOURCE_LOCK_SCHEMA = "path-b-v3-r3-source-lock/v1"
PATH_B_V3_R3_REVIEWER_EXPORT_SCHEMA = "path-b-v3-r3-reviewer-export/v1"
PATH_B_V3_R3_PACKAGE_MANIFEST_SCHEMA = "path-b-v3-r3-derived-package-manifest/v1"
PATH_B_V3_R3_PACKAGE_FORMAT = "path-b-v2-component-package/v1"
PATH_B_V3_R3_EXPECTED_TASK_COUNT = 543

# These are immutable disclosure anchors.  The r2 artifacts remain available
# in the repository, but cannot be used by the new adjudication.
PATH_B_V3_R2_PACK_MANIFEST_SHA256 = (
    "ab4953154904d56586feb1895963cba28cd81161d44560986d083160a1f4e77b"
)
PATH_B_V3_R2_REVIEWER_EXPORT_FILE_SHA256 = (
    "040b6790eeda5bc50f7a764bfdfa3863deb4abce4c5ae95f8f0bcf75ab338ae5"
)
PATH_B_V3_R2_CR_IMPORT_SHA256 = "f53b007b36173c9bc928df22e252692299d400c13ac18cea34dbac9f78caf957"
PATH_B_V3_R2_QA_IMPORT_SHA256 = "e60911ea6aa4fe5f2a5f3050ad1a7d71c4116568217006669d02457d1cc69c50"
PATH_B_V3_PRIOR_REVIEWER_EXPORT_FILE_SHA256 = (
    "ea4615c7317360a67704f51f5eadd836bdf84584fcbb8628015bb9eda76c141c"
)
PATH_B_V3_PRIOR_QA_IMPORT_SHA256 = (
    "a2485bc4548e91bb53798d63c2f0f331822cd49b671a712b871d695a9466c8b0"
)

_LEDGER_COLUMNS = frozenset(
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
_TEMPORAL_COLUMNS = frozenset(
    {
        "task_id",
        "category",
        "evidence_url",
        "evidence_published_at",
        "content_sha256",
        "outcome_available_at_source",
        "manual_review_eligible",
    }
)
_EVENT_COLUMNS = frozenset(
    {
        "task_id",
        "first_public_at",
        "evidence_url",
        "content_sha256",
        "price_evidence_url",
        "price_evidence_published_at",
        "price_content_sha256",
        "manual_review_eligible",
    }
)
_RAW_ITEM_FIELDS = (
    "kind",
    "source_url",
    "snapshot_path",
    "snapshot_sha256",
    "published_at",
)
_REQUIRED_B_PRICE_ROLES = ("pre", "post", "benchmark_pre", "benchmark_post")
_STRATA = {
    "A1": ("temporal_price_forecast", "multi_horizon_price_band"),
    "A2-T": ("temporal_ranking_forecast", "cross_sectional_ranking"),
    "B-EARNINGS": ("earnings_event_reaction", "binary_direction_probability"),
    "B-MACRO": ("macro_event_reaction", "binary_direction_probability"),
}


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _required_text(value: Any, name: str) -> str:
    parsed = _optional_text(value)
    if parsed is None:
        raise PathBV2ValidationError(f"{name} must be a non-empty string")
    return parsed


def _explicit_boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise PathBV2ValidationError(f"{name} must be an explicit true/false value")


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PathBV2ValidationError(f"unable to read source file: {path}") from exc


def _hash_without(value: Mapping[str, Any], field: str) -> str:
    copied = dict(value)
    copied.pop(field, None)
    return sha256_json(copied)


def _read_csv_rows(
    path: Path, *, required_columns: frozenset[str], output_columns: frozenset[str]
) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = frozenset(reader.fieldnames or ())
            missing = sorted(required_columns - fieldnames)
            if missing:
                raise PathBV2ValidationError(
                    f"source table {path.name} is missing required columns: {missing}"
                )
            # Project immediately to an allowlist.  In particular, the source
            # temporal table also contains evaluator-owned value columns; they
            # never cross this boundary.
            return [
                {field: str(row.get(field) or "").strip() for field in output_columns}
                for row in reader
            ]
    except OSError as exc:
        raise PathBV2ValidationError(f"unable to read source table: {path}") from exc


def _read_ledger(path: Path) -> list[dict[str, str]]:
    return _read_csv_rows(
        path,
        required_columns=_LEDGER_COLUMNS,
        output_columns=_LEDGER_COLUMNS,
    )


def _read_evidence_packages(path: Path) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PathBV2ValidationError(f"unable to read source evidence packages: {path}") from exc
    packages: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PathBV2ValidationError(
                f"invalid source evidence package JSON at line {line_number}"
            ) from exc
        if not isinstance(value, Mapping):
            raise PathBV2ValidationError(
                f"source evidence package at line {line_number} must be an object"
            )
        packages.append(value)
    return packages


def _read_table_map(
    path: Path, *, required_columns: frozenset[str], output_columns: frozenset[str]
) -> dict[str, Mapping[str, str]]:
    rows = _read_csv_rows(
        path,
        required_columns=required_columns,
        output_columns=output_columns,
    )
    mapped: dict[str, Mapping[str, str]] = {}
    for index, row in enumerate(rows):
        task_id = _required_text(row.get("task_id"), f"{path.name} row {index}.task_id")
        if task_id in mapped:
            raise PathBV2ValidationError(f"duplicate task_id in {path.name}: {task_id}")
        mapped[task_id] = row
    return mapped


def _canonical_category(row: Mapping[str, Any]) -> str:
    category = _required_text(row.get("category"), "ledger.category").upper().replace("_", "-")
    variant = str(row.get("variant") or "").strip().upper().replace("_", "-")
    if category == "A1":
        return "A1"
    if category in {"A2", "A2-T"} and variant in {"T", "TEMPORAL", ""}:
        return "A2-T"
    if category == "B" and variant in {"EARNINGS", "B-EARNINGS"}:
        return "B-EARNINGS"
    if category == "B" and variant in {"MACRO", "B-MACRO"}:
        return "B-MACRO"
    raise PathBV2ValidationError(
        f"unsupported Path B category/variant: {row.get('category')!r}/{row.get('variant')!r}"
    )


def _package_items(package: Mapping[str, Any], task_id: str) -> list[Mapping[str, Any]]:
    items = package.get("items")
    if not isinstance(items, list):
        raise PathBV2ValidationError(f"source package items are unavailable for {task_id}")
    refs: list[Mapping[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise PathBV2ValidationError(
                f"source package item {index} is not an object for {task_id}"
            )
        refs.append(item)
    return refs


def _safe_item_ref(item: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only reviewable source references, never cache or payload fields."""

    return {field: _optional_text(item.get(field)) for field in _RAW_ITEM_FIELDS}


def _source_refs(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_safe_item_ref(item) for item in items]


def _first_item(items: Sequence[Mapping[str, Any]], kind: str) -> Mapping[str, Any] | None:
    return next((item for item in items if item.get("kind") == kind), None)


def _source_projection(ledger_row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "review_status": _optional_text(ledger_row.get("review_status")),
        "official_temporal_eligible": _explicit_boolean(
            ledger_row.get("official_temporal_eligible"),
            "ledger.official_temporal_eligible",
        ),
        "state": "pre_attestation_projection",
        "is_decision": False,
    }


def _component_source_fields(
    component: dict[str, Any], *, source_url: Any, source_sha256: Any, published_at: Any
) -> None:
    url = _optional_text(source_url)
    source_hash = _optional_text(source_sha256)
    published = _optional_text(published_at)
    if url is not None:
        component["source_url"] = url
    if source_hash is not None:
        component["source_sha256"] = source_hash
    if published is not None:
        component["published_at"] = published


def _derive_temporal_component(
    *,
    task_id: str,
    category: str,
    temporal_row: Mapping[str, str] | None,
    event_row: Mapping[str, str] | None,
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    source = temporal_row if temporal_row is not None else event_row
    if source is None:
        available = "unobserved_missing_source_row"
        source_url = source_hash = published_at = None
    elif temporal_row is not None:
        available = _optional_text(source.get("outcome_available_at_source"))
        available = available or "unobserved_missing_availability"
        source_url = source.get("evidence_url")
        source_hash = source.get("content_sha256")
        published_at = source.get("evidence_published_at")
    else:
        # B has no temporal table row.  An event publication is an observed
        # time anchor, but it is not a completed price/outcome chain.
        event_url = _optional_text(source.get("evidence_url"))
        event_hash = _optional_text(source.get("content_sha256"))
        event_time = _optional_text(source.get("first_public_at"))
        available = (
            "observed_event_publication"
            if event_url and event_hash and event_time
            else "unobserved_missing_event_evidence"
        )
        source_url, source_hash, published_at = event_url, event_hash, event_time

    candidate_complete = bool(
        _optional_text(source_url)
        and _optional_text(source_hash)
        and _optional_text(published_at)
        and available.startswith("observed_")
    )
    component: dict[str, Any] = {
        "task_id": task_id,
        "official_temporal_eligible": candidate_complete,
        "outcome_available_at_source": available,
        "source_projection": dict(projection),
        "derivation_status": "candidate" if candidate_complete else "incomplete",
    }
    _component_source_fields(
        component,
        source_url=source_url,
        source_sha256=source_hash,
        published_at=published_at,
    )
    if category == "B-MACRO":
        component["derivation_status"] = "audit_only"
    return component


def _derive_event_component(
    *,
    task_id: str,
    category: str,
    temporal_row: Mapping[str, str] | None,
    event_row: Mapping[str, str] | None,
) -> dict[str, Any]:
    component: dict[str, Any] = {"task_id": task_id}
    if category == "B-EARNINGS" or category == "B-MACRO":
        component["event_id"] = task_id
        if event_row is not None:
            _component_source_fields(
                component,
                source_url=event_row.get("evidence_url"),
                source_sha256=event_row.get("content_sha256"),
                published_at=event_row.get("first_public_at"),
            )
        component["evidence_type"] = "first_party_event_snapshot"
    elif temporal_row is not None:
        _component_source_fields(
            component,
            source_url=temporal_row.get("evidence_url"),
            source_sha256=temporal_row.get("content_sha256"),
            published_at=temporal_row.get("evidence_published_at"),
        )
        component["evidence_type"] = "temporal_source_snapshot"
    else:
        component["evidence_type"] = "missing_source_snapshot"
    component["verification_status"] = (
        "candidate"
        if component.get("source_url")
        and component.get("source_sha256")
        and component.get("published_at")
        else "incomplete"
    )
    return component


def _price_refs(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_safe_item_ref(item) for item in items if item.get("kind") == "price_snapshot"]


def _derive_outcome_component(
    *,
    task_id: str,
    category: str,
    temporal_row: Mapping[str, str] | None,
    event_row: Mapping[str, str] | None,
    price_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if category in {"B-EARNINGS", "B-MACRO"}:
        component: dict[str, Any] = {
            "task_id": task_id,
            "derivation_status": "incomplete",
            "price_evidence_status": "incomplete",
            "available_price_refs": [dict(ref) for ref in price_refs],
            "missing_components": list(_REQUIRED_B_PRICE_ROLES),
        }
        if event_row is not None:
            _component_source_fields(
                component,
                source_url=event_row.get("price_evidence_url"),
                source_sha256=event_row.get("price_content_sha256"),
                published_at=event_row.get("price_evidence_published_at"),
            )
        if category == "B-MACRO":
            component["derivation_status"] = "audit_only"
        return component

    component = {
        "task_id": task_id,
        "derivation_status": "candidate" if temporal_row is not None else "incomplete",
        "outcome_available_at_source": (
            _optional_text(temporal_row.get("outcome_available_at_source"))
            if temporal_row is not None
            else "unobserved_missing_source_row"
        ),
    }
    if temporal_row is not None:
        _component_source_fields(
            component,
            source_url=temporal_row.get("evidence_url"),
            source_sha256=temporal_row.get("content_sha256"),
            published_at=temporal_row.get("evidence_published_at"),
        )
    return component


def _derive_package(
    *,
    ledger_row: Mapping[str, str],
    source_package: Mapping[str, Any],
    temporal_row: Mapping[str, str] | None,
    event_row: Mapping[str, str] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = _required_text(ledger_row.get("task_id"), "ledger.task_id")
    category = _canonical_category(ledger_row)
    task_family, outcome_type = _STRATA[category]
    paper_band = _required_text(ledger_row.get("paper_band"), f"{task_id}.paper_band")
    items = _package_items(source_package, task_id)
    baseline = _first_item(items, "baseline_task_identity")
    identity: dict[str, Any] = {
        "task_id": task_id,
        "category": category,
        "variant": (
            "temporal"
            if category in {"A1", "A2-T"}
            else "earnings"
            if category == "B-EARNINGS"
            else "macro"
        ),
        "task_family": task_family,
        "paper_band": paper_band,
        "time_band": paper_band,
        "outcome_type": outcome_type,
        "stratum": {
            "category": category,
            "time_band": paper_band,
            "task_family": task_family,
            "paper_band": paper_band,
            "outcome_type": outcome_type,
        },
    }
    if baseline is not None:
        _component_source_fields(
            identity,
            source_url=baseline.get("source_url"),
            source_sha256=baseline.get("snapshot_sha256"),
            published_at=None,
        )

    projection = _source_projection(ledger_row)
    package: dict[str, Any] = {
        "task_id": task_id,
        "task_identity": identity,
        "temporal_provenance": _derive_temporal_component(
            task_id=task_id,
            category=category,
            temporal_row=temporal_row,
            event_row=event_row,
            projection=projection,
        ),
        "event_evidence": _derive_event_component(
            task_id=task_id,
            category=category,
            temporal_row=temporal_row,
            event_row=event_row,
        ),
        "outcome_derivation": _derive_outcome_component(
            task_id=task_id,
            category=category,
            temporal_row=temporal_row,
            event_row=event_row,
            price_refs=_price_refs(items),
        ),
    }
    package["evidence_package_sha256"] = compute_evidence_package_sha256(package)
    assert_path_b_v2_score_blind(package, path=f"$.derived_packages[{task_id}]")

    missing: list[str] = []
    temporal = package["temporal_provenance"]
    if temporal.get("official_temporal_eligible") is not True:
        missing.append("missing_official_temporal_evidence")
    event = package["event_evidence"]
    if not all(event.get(field) for field in ("source_url", "source_sha256", "published_at")):
        missing.append("missing_event_evidence")
    if category in {"B-EARNINGS", "B-MACRO"}:
        missing.extend(f"missing_price_chain_{role}" for role in _REQUIRED_B_PRICE_ROLES)
    if category == "B-MACRO":
        missing.append("b_macro_audit_only")

    try:
        validate_evidence_package(package, expected_task_id=task_id)
    except PathBV2ValidationError as exc:
        validation_status = "not_validator_ready"
        validation_error = str(exc)
        if not missing:
            missing.append("validator_rejected_incomplete_source")
    else:
        validation_status = "validator_ready"
        validation_error = None
    status = {
        "task_id": task_id,
        "category": category,
        "source_package_sha256": _required_text(
            ledger_row.get("evidence_package_sha256"),
            f"{task_id}.source_package_sha256",
        ),
        "evidence_package_sha256": package["evidence_package_sha256"],
        "package_format": PATH_B_V3_R3_PACKAGE_FORMAT,
        "validation_status": validation_status,
        "validation_error": validation_error,
        "exclusion_reasons": sorted(set(missing)),
    }
    assert_path_b_v2_score_blind(status, path=f"$.derived_package_status[{task_id}]")
    return package, status


def _published_candidates(
    *,
    items: Sequence[Mapping[str, Any]],
    temporal_row: Mapping[str, str] | None,
    event_row: Mapping[str, str] | None,
) -> list[str]:
    values: set[str] = set()
    for item in items:
        value = _optional_text(item.get("published_at"))
        if value is not None:
            values.add(value)
    if temporal_row is not None:
        value = _optional_text(temporal_row.get("evidence_published_at"))
        if value is not None:
            values.add(value)
    if event_row is not None:
        for field in ("first_public_at", "price_evidence_published_at"):
            value = _optional_text(event_row.get(field))
            if value is not None:
                values.add(value)
    return sorted(values)


def _snapshot_refs(
    *,
    items: Sequence[Mapping[str, Any]],
    temporal_row: Mapping[str, str] | None,
    event_row: Mapping[str, str] | None,
) -> list[dict[str, Any]]:
    refs: set[tuple[str | None, str | None]] = set()
    for item in items:
        path = _optional_text(item.get("snapshot_path"))
        source_hash = _optional_text(item.get("snapshot_sha256"))
        if path is not None or source_hash is not None:
            refs.add((path, source_hash))
    for row in (temporal_row, event_row):
        if row is None:
            continue
        source_hash = _optional_text(row.get("content_sha256"))
        if source_hash is not None:
            refs.add((None, source_hash))
    return [
        {"snapshot_path": path, "snapshot_sha256": source_hash}
        for path, source_hash in sorted(refs, key=lambda ref: (ref[0] or "", ref[1] or ""))
    ]


def _official_urls(
    *,
    items: Sequence[Mapping[str, Any]],
    temporal_row: Mapping[str, str] | None,
    event_row: Mapping[str, str] | None,
) -> list[str]:
    urls: set[str] = set()
    for item in items:
        value = _optional_text(item.get("source_url"))
        if value is not None and value.startswith(("http://", "https://")):
            urls.add(value)
    for row in (temporal_row, event_row):
        if row is None:
            continue
        for field in ("evidence_url", "price_evidence_url"):
            value = _optional_text(row.get(field))
            if value is not None and value.startswith(("http://", "https://")):
                urls.add(value)
    return sorted(urls)


def _component_summary(package: Mapping[str, Any], status: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": PATH_B_V3_R3_PACKAGE_FORMAT,
        "hash_algorithm": "sha256(canonical_json(package_without_evidence_package_sha256))",
        "component_keys": list(PATH_B_V2_EVIDENCE_COMPONENTS),
        "validation_status": status["validation_status"],
        "exclusion_reasons": list(status["exclusion_reasons"]),
        "component_summaries": {
            name: {
                "task_id": package[name].get("task_id"),
                "source_url": package[name].get("source_url"),
                "source_sha256": package[name].get("source_sha256"),
                "published_at": package[name].get("published_at"),
                "derivation_status": package[name].get("derivation_status"),
            }
            for name in PATH_B_V2_EVIDENCE_COMPONENTS
        },
    }


def _reviewer_rows(
    *,
    ledger_rows: Sequence[Mapping[str, str]],
    source_packages: Mapping[str, Mapping[str, Any]],
    derived_packages: Mapping[str, Mapping[str, Any]],
    statuses: Mapping[str, Mapping[str, Any]],
    roster: Mapping[str, Any],
    temporal_rows: Mapping[str, Mapping[str, str]],
    event_rows: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    ledger_by_task = {row["task_id"]: row for row in ledger_rows}
    output: list[dict[str, Any]] = []
    for index, roster_row in enumerate(roster["rows"]):
        task_id = roster_row["task_id"]
        ledger = ledger_by_task[task_id]
        source = source_packages[task_id]
        items = _package_items(source, task_id)
        temporal = temporal_rows.get(task_id)
        event = event_rows.get(task_id)
        package = derived_packages[task_id]
        status = statuses[task_id]
        published = (
            _optional_text(temporal.get("evidence_published_at"))
            if temporal is not None
            else _optional_text(event.get("first_public_at"))
            if event is not None
            else None
        )
        published_candidates = _published_candidates(
            items=items,
            temporal_row=temporal,
            event_row=event,
        )
        source_urls = _official_urls(
            items=items,
            temporal_row=temporal,
            event_row=event,
        )
        snapshot_refs = _snapshot_refs(
            items=items,
            temporal_row=temporal,
            event_row=event,
        )
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
        row: dict[str, Any] = {
            "task_id": task_id,
            "category": roster_row["category"],
            "time_band": roster_row["time_band"],
            "task_family": roster_row["task_family"],
            "paper_band": roster_row["paper_band"],
            "outcome_type": roster_row["outcome_type"],
            "stratum": stratum,
            # These two values are deliberately source-side projections.  The
            # instructions below say explicitly that neither is a veto.
            "review_status": _optional_text(ledger.get("review_status")),
            "official_temporal_eligible": _explicit_boolean(
                ledger.get("official_temporal_eligible"),
                f"ledger row {index}.official_temporal_eligible",
            ),
            "official_time_evidence": {
                "candidate_source_urls": source_urls,
                "candidate_snapshot_refs": snapshot_refs,
                "published_at": published,
                "published_at_candidates": published_candidates,
                "source_projection": _source_projection(ledger),
                "status": "requires_independent_human_verification",
            },
            "official_source_urls": source_urls,
            "official_snapshot_refs": snapshot_refs,
            "published_at": published,
            "published_at_candidates": published_candidates,
            "source_evidence_status": {
                "event": _optional_text(ledger.get("event_evidence_status")),
                "price": _optional_text(ledger.get("price_evidence_status")),
            },
            "evidence_package_content_summary": _component_summary(package, status),
            "evidence_package_sha256": package["evidence_package_sha256"],
        }
        if not PATH_B_V3_REVIEWER_EXPORT_REQUIRED_FIELDS <= row.keys():
            raise PathBV2ValidationError(
                f"r3 reviewer export row {index} is missing required fields"
            )
        assert_path_b_v2_score_blind(row, path=f"$.reviewer_export[{index}]")
        output.append(row)
    queue = [
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
    _reviewer_export_rows(output, queue)
    return output


def _package_manifest(
    statuses: Sequence[Mapping[str, Any]], *, review_contract: Mapping[str, Any]
) -> dict[str, Any]:
    entries = [dict(status) for status in statuses]
    entries.sort(key=lambda entry: entry["task_id"])
    body = {
        "schema_version": PATH_B_V3_R3_PACKAGE_MANIFEST_SCHEMA,
        "review_contract_sha256": sha256_json(review_contract),
        "package_format": PATH_B_V3_R3_PACKAGE_FORMAT,
        "package_count": len(entries),
        "validator_ready_package_count": sum(
            entry["validation_status"] == "validator_ready" for entry in entries
        ),
        "packages": entries,
    }
    manifest = {**body, "manifest_sha256": sha256_json(body)}
    assert_path_b_v2_score_blind(manifest, path="$.derived_package_manifest")
    return manifest


def _source_lock(
    *,
    contract: Mapping[str, Any],
    source_files: Mapping[str, str],
    ledger_count: int,
    raw_package_count: int,
    derived_package_manifest: Mapping[str, Any],
    derived_package_file_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema_version": PATH_B_V3_R3_SOURCE_LOCK_SCHEMA,
        "dataset_lock": dict(contract["dataset_lock"]),
        "review_contract_sha256": sha256_json(contract),
        "source_files": dict(sorted(source_files.items())),
        "ledger_row_count": ledger_count,
        "raw_package_row_count": raw_package_count,
        "derived_package_row_count": derived_package_manifest["package_count"],
        "validator_ready_package_count": derived_package_manifest["validator_ready_package_count"],
        "raw_package_format": "path_b_v2_task_id_items_v1",
        "raw_package_hash": "sha256(canonical_json(items))",
        "derived_package_format": PATH_B_V3_R3_PACKAGE_FORMAT,
        "derived_package_hash": ("sha256(canonical_json(package_without_evidence_package_sha256))"),
        "derived_package_file_sha256": derived_package_file_sha256,
        "rebind": "roster_and_attestations_bind_new_component_package_hashes",
        "review_boundary": "score_blind",
    }
    lock = {**body, "source_lock_sha256": sha256_json(body)}
    assert_path_b_v2_score_blind(lock, path="$.source_lock")
    return lock


def _review_instructions() -> str:
    return """# Path B v3 r3 blinded review handoff

State: `pending_independent_attestation`.

This append-only r3 pack is bound to the frozen Hub revision and tree in the
review contract.  The source Hub package index used an old item-list hash; r3
derives and binds a new four-component package hash for every ledger row.  The
derived package manifest records incomplete source material explicitly.  A
package that is not validator-ready is not repaired, silently promoted, or
used as evidence for an eligible decision.

The blinded export contains task identity, frozen stratum, official URL and
snapshot candidates, explicit `published_at` (or `null` when absent), source
evidence status, the new package hash, and a content summary of the four
components.  It contains no collector identity, prediction, score, result,
or evaluator output.  The sealed canonical roster retains collector identity
only for the later self-sign check and is not the reviewer export.

## Source-side projection rule

`review_status` and `official_temporal_eligible` in the export are copied as
pre-attestation source projections.  For this pending source they are
constructively `draft` and JSON `false` for all rows.  They are not a veto and
must not be used as a reviewer decision input.  The reviewer's independent
attestation supplies the reviewed and confirmed state.  Missing URLs, hashes,
or dates remain missing; do not invent dates, replace observed dates with
modeled dates, or infer a complete evidence chain from a candidate reference.

## Five frozen decision criteria

1. Verify an official first-party temporal/event URL and the cited snapshot or
   content hash; a locator without verifiable source content is insufficient.
2. Verify an explicit timezone-bearing `published_at`; a missing, ambiguous,
   or modeled date is not eligible.
3. Verify that the new package SHA binds the current four components
   (`task_identity`, `temporal_provenance`, `event_evidence`, and
   `outcome_derivation`) and that the component content is complete for the
   claimed decision.
4. For B-EARNINGS, verify the event plus pre/post instrument and benchmark
   price chain, ordering, sessions, calendar, timezone, and frozen formula;
   incomplete chains fail closed.  B-MACRO and C remain audit/excluded rows.
5. Apply independent reviewer checks and the frozen stratum; the collector
   must not attest their own row.  Only the attestation decision changes the
   reviewed state, never a source-side draft/false projection.

The following r2 artifacts are retained for disclosure but explicitly
invalidated and must not enter r3 adjudication: pack manifest
`ab4953154904d56586feb1895963cba28cd81161d44560986d083160a1f4e77b`, r2
reviewer export file `040b6790eeda5bc50f7a764bfdfa3863deb4abce4c5ae95f8f0bcf75ab338ae5`,
CR import
`f53b007b36173c9bc928df22e252692299d400c13ac18cea34dbac9f78caf957`, and QA
import
`e60911ea6aa4fe5f2a5f3050ad1a7d71c4116568217006669d02457d1cc69c50`.  The
earlier invalidated export `ea4615c7317360a67704f51f5eadd836bdf84584fcbb8628015bb9eda76c141c`
and QA import `a2485bc4548e91bb53798d63c2f0f331822cd49b671a712b871d695a9466c8b0`
are also disclosure-only.

CR and QA must review this export independently.  Each import must contain
exactly the eight frozen attestation fields and must bind the r3 package hash.
Keep imports private until both are received; then run deterministic
adjudication, apply the sealed adjudication rule for any repeated projection
disagreement, and build a separate validator-ready v2 bundle.  Do not freeze
a cohort, generate a scored plan, or run an experiment from this pending pack.
"""


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    text = "".join(canonical_json(row) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_r3_pack(
    output: Path,
    *,
    roster: Mapping[str, Any],
    reviewer_rows: Sequence[Mapping[str, Any]],
    derived_packages: Sequence[Mapping[str, Any]],
    validator_ready_packages: Sequence[Mapping[str, Any]],
    derived_manifest: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    review_instructions: str,
) -> dict[str, Any]:
    if output.exists():
        raise PathBV2ValidationError("r3 review-pack output must not already exist")
    assert_path_b_v2_score_blind(roster, path="$.roster")
    if roster.get("roster_sha256") != _hash_without(roster, "roster_sha256"):
        raise PathBV2ValidationError("r3 roster hash mismatch")
    assert_path_b_v2_score_blind(source_lock, path="$.source_lock")
    if source_lock.get("source_lock_sha256") != _hash_without(source_lock, "source_lock_sha256"):
        raise PathBV2ValidationError("r3 source lock hash mismatch")
    assert_path_b_v2_score_blind(derived_manifest, path="$.derived_package_manifest")
    if derived_manifest.get("manifest_sha256") != _hash_without(
        derived_manifest, "manifest_sha256"
    ):
        raise PathBV2ValidationError("r3 derived package manifest hash mismatch")
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
    checked_export = _reviewer_export_rows(reviewer_rows, queue_rows)
    assert_path_b_v2_score_blind(derived_packages, path="$.derived_packages")
    assert_path_b_v2_score_blind(validator_ready_packages, path="$.validator_ready_packages")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True)

    roster_path = output / "canonical-roster.json"
    queue_path = output / "blinded-review-queue.jsonl"
    export_path = output / "blinded-review-export.jsonl"
    attestations_path = output / "attestations.jsonl"
    source_lock_path = output / "source-lock.json"
    instructions_path = output / "review-instructions.md"
    packages_path = output / "derived-evidence-packages.jsonl"
    validator_packages_path = output / "validator-ready-evidence-packages.jsonl"
    package_manifest_path = output / "derived-evidence-package-manifest.json"

    roster_path.write_text(canonical_json(roster) + "\n", encoding="utf-8")
    queue_file_sha = _write_jsonl(queue_path, queue_rows)
    export_file_sha = _write_jsonl(export_path, checked_export)
    attestations_path.write_text("", encoding="utf-8")
    source_lock_path.write_text(canonical_json(source_lock) + "\n", encoding="utf-8")
    instructions_path.write_text(review_instructions, encoding="utf-8")
    package_file_sha = _write_jsonl(packages_path, derived_packages)
    validator_package_file_sha = _write_jsonl(validator_packages_path, validator_ready_packages)
    package_manifest_path.write_text(canonical_json(derived_manifest) + "\n", encoding="utf-8")

    files = {
        path.name: _sha256_file(path)
        for path in (
            roster_path,
            queue_path,
            export_path,
            attestations_path,
            source_lock_path,
            instructions_path,
            packages_path,
            validator_packages_path,
            package_manifest_path,
        )
    }
    supersedes = {
        "disposition": "invalidated_not_for_adjudication",
        "r2_pack_manifest_sha256": PATH_B_V3_R2_PACK_MANIFEST_SHA256,
        "r2_reviewer_export_file_sha256": PATH_B_V3_R2_REVIEWER_EXPORT_FILE_SHA256,
        "r2_cr_import_sha256": PATH_B_V3_R2_CR_IMPORT_SHA256,
        "r2_qa_import_sha256": PATH_B_V3_R2_QA_IMPORT_SHA256,
        "prior_reviewer_export_file_sha256": PATH_B_V3_PRIOR_REVIEWER_EXPORT_FILE_SHA256,
        "prior_qa_import_sha256": PATH_B_V3_PRIOR_QA_IMPORT_SHA256,
    }
    assert_path_b_v2_score_blind(supersedes, path="$.supersedes")
    manifest_body = {
        "schema_version": PATH_B_V3_R3_REVIEW_SCHEMA,
        "roster_sha256": roster["roster_sha256"],
        "review_queue_count": len(queue_rows),
        "review_queue_file_sha256": queue_file_sha,
        "reviewer_export_schema": PATH_B_V3_R3_REVIEWER_EXPORT_SCHEMA,
        "reviewer_export_count": len(checked_export),
        "reviewer_export_sha256": sha256_json(checked_export),
        "reviewer_export_file_sha256": export_file_sha,
        "derived_package_count": len(derived_packages),
        "derived_package_file_sha256": package_file_sha,
        "validator_ready_package_count": len(validator_ready_packages),
        "validator_ready_package_file_sha256": validator_package_file_sha,
        "derived_package_manifest_sha256": derived_manifest["manifest_sha256"],
        "source_lock_sha256": source_lock["source_lock_sha256"],
        "review_instructions_sha256": sha256_json(review_instructions),
        "attestations_filename": attestations_path.name,
        "attestation_count": 0,
        "review_state": "pending_independent_attestation",
        "supersedes": supersedes,
        "files": files,
    }
    manifest = {**manifest_body, "manifest_sha256": sha256_json(manifest_body)}
    manifest_path = output / "sealed-review-pack-manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return {
        "root": str(output),
        "roster_sha256": roster["roster_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "review_state": manifest["review_state"],
        "reviewer_export_count": len(checked_export),
        "reviewer_export_sha256": manifest["reviewer_export_sha256"],
        "reviewer_export_file_sha256": export_file_sha,
        "derived_package_count": len(derived_packages),
        "derived_package_manifest_sha256": derived_manifest["manifest_sha256"],
        "validator_ready_package_count": len(validator_ready_packages),
        "source_lock_sha256": source_lock["source_lock_sha256"],
        "supersedes": supersedes,
    }


def prepare_path_b_v3_r3_review_pack(
    source_root: str | Path,
    output: str | Path,
    *,
    contract_path: str | Path = "configs/path-b-v2-review-contract.json",
    expected_task_count: int | None = PATH_B_V3_R3_EXPECTED_TASK_COUNT,
) -> dict[str, Any]:
    """Derive and seal a deterministic r3 review pack from the pinned source."""

    source = Path(source_root).expanduser().resolve()
    if not source.is_dir():
        raise PathBV2ValidationError(f"Path B source root is not a directory: {source}")
    output_path = Path(output).expanduser().resolve()
    try:
        output_path.relative_to(source)
    except ValueError:
        pass
    else:
        raise PathBV2ValidationError("r3 review-pack output must be outside source root")

    contract_file = Path(contract_path).expanduser().resolve()
    try:
        contract = validate_review_contract(json.loads(contract_file.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PathBV2ValidationError(f"unable to read review contract: {contract_file}") from exc

    ledger_path = source / _LEDGER_RELATIVE_PATH
    packages_path = source / _PACKAGES_RELATIVE_PATH
    temporal_path = source / _TEMPORAL_RELATIVE_PATH
    event_path = source / _EVENT_RELATIVE_PATH
    ledger_rows = _read_ledger(ledger_path)
    source_package_rows = _read_evidence_packages(packages_path)
    temporal_rows = _read_table_map(
        temporal_path,
        required_columns=_TEMPORAL_COLUMNS,
        output_columns=_TEMPORAL_COLUMNS,
    )
    event_rows = _read_table_map(
        event_path,
        required_columns=_EVENT_COLUMNS,
        output_columns=_EVENT_COLUMNS,
    )
    if expected_task_count is not None and len(ledger_rows) != expected_task_count:
        raise PathBV2ValidationError(
            f"pinned Path B ledger count {len(ledger_rows)} does not match expected "
            f"{expected_task_count}"
        )
    if len(source_package_rows) != len(ledger_rows):
        raise PathBV2ValidationError("source ledger/package row counts differ")

    ledger_by_task = {row["task_id"]: row for row in ledger_rows}
    if len(ledger_by_task) != len(ledger_rows):
        raise PathBV2ValidationError("source ledger contains duplicate task IDs")
    source_packages: dict[str, Mapping[str, Any]] = {}
    for index, package in enumerate(source_package_rows):
        task_id = _required_text(package.get("task_id"), f"source package {index}.task_id")
        if task_id in source_packages:
            raise PathBV2ValidationError(f"duplicate source package task_id: {task_id}")
        source_packages[task_id] = package
    if set(source_packages) != set(ledger_by_task):
        raise PathBV2ValidationError("source ledger/package task identities differ")

    # Verify the raw source binding before deriving anything.  This checks the
    # pinned item-list hashes, while the new roster below binds only new hashes.
    build_path_b_v2_canonical_roster(
        ledger_rows,
        source_package_rows,
        review_contract=contract,
    )

    derived_packages: dict[str, Mapping[str, Any]] = {}
    statuses: dict[str, Mapping[str, Any]] = {}
    rebound_ledger_rows: list[dict[str, str]] = []
    for task_id in sorted(ledger_by_task):
        package, status = _derive_package(
            ledger_row=ledger_by_task[task_id],
            source_package=source_packages[task_id],
            temporal_row=temporal_rows.get(task_id),
            event_row=event_rows.get(task_id),
        )
        derived_packages[task_id] = package
        statuses[task_id] = status
        rebound_ledger_rows.append(
            {
                **ledger_by_task[task_id],
                "evidence_package_sha256": package["evidence_package_sha256"],
            }
        )
    roster = build_path_b_v2_canonical_roster(
        rebound_ledger_rows,
        list(derived_packages.values()),
        review_contract=contract,
    )
    roster_body = {**roster, "schema_version": PATH_B_V3_R3_REVIEW_SCHEMA}
    roster_body.pop("roster_sha256", None)
    roster = {**roster_body, "roster_sha256": sha256_json(roster_body)}

    status_entries = [statuses[task_id] for task_id in sorted(statuses)]
    derived_manifest = _package_manifest(status_entries, review_contract=contract)
    derived_package_list = [derived_packages[task_id] for task_id in sorted(derived_packages)]
    validator_ready_packages = [
        package
        for package in derived_package_list
        if statuses[package["task_id"]]["validation_status"] == "validator_ready"
    ]
    # The package file hash is deterministic and is also included in the source
    # lock.  Use the exact bytes written by _write_jsonl.
    derived_package_text = "".join(
        canonical_json(package) + "\n" for package in derived_package_list
    )
    source_files = {
        str(_LEDGER_RELATIVE_PATH): _sha256_file(ledger_path),
        str(_PACKAGES_RELATIVE_PATH): _sha256_file(packages_path),
        str(_TEMPORAL_RELATIVE_PATH): _sha256_file(temporal_path),
        str(_EVENT_RELATIVE_PATH): _sha256_file(event_path),
    }
    source_lock = _source_lock(
        contract=contract,
        source_files=source_files,
        ledger_count=len(ledger_rows),
        raw_package_count=len(source_package_rows),
        derived_package_manifest=derived_manifest,
        derived_package_file_sha256=hashlib.sha256(
            derived_package_text.encode("utf-8")
        ).hexdigest(),
    )
    reviewer_rows = _reviewer_rows(
        ledger_rows=ledger_rows,
        source_packages=source_packages,
        derived_packages=derived_packages,
        statuses=statuses,
        roster=roster,
        temporal_rows=temporal_rows,
        event_rows=event_rows,
    )
    result = _write_r3_pack(
        output_path,
        roster=roster,
        reviewer_rows=reviewer_rows,
        derived_packages=derived_package_list,
        validator_ready_packages=validator_ready_packages,
        derived_manifest=derived_manifest,
        source_lock=source_lock,
        review_instructions=_review_instructions(),
    )
    return {
        **result,
        "ledger_count": len(ledger_rows),
        "raw_package_count": len(source_package_rows),
        "temporal_table_row_count": len(temporal_rows),
        "event_table_row_count": len(event_rows),
        "contract_sha256": sha256_json(contract),
        "dataset_revision": contract["dataset_lock"]["revision"],
        "dataset_tree_sha": contract["dataset_lock"]["tree_sha"],
        "source_files": source_files,
    }


__all__ = [
    "PATH_B_V3_R2_CR_IMPORT_SHA256",
    "PATH_B_V3_R2_PACK_MANIFEST_SHA256",
    "PATH_B_V3_R2_QA_IMPORT_SHA256",
    "PATH_B_V3_R2_REVIEWER_EXPORT_FILE_SHA256",
    "PATH_B_V3_R3_EXPECTED_TASK_COUNT",
    "PATH_B_V3_R3_PACKAGE_FORMAT",
    "PATH_B_V3_R3_PACKAGE_MANIFEST_SCHEMA",
    "PATH_B_V3_R3_REVIEWER_EXPORT_SCHEMA",
    "PATH_B_V3_R3_REVIEW_SCHEMA",
    "PATH_B_V3_R3_SOURCE_LOCK_SCHEMA",
    "prepare_path_b_v3_r3_review_pack",
]
