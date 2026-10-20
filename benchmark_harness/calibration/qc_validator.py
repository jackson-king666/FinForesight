"""Fail-closed, offline validation and artifact I/O for Evidence-QC.

The v8 calibration root is a protected source bundle.  This module reads only
the v8 seal and explicitly allow-listed pilot files; it never calls the v8
bundle validator and never opens the top-level 400-item files.  All derived
outputs are written to a separate, append-only review root.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from benchmark_harness.calibration.evidence_qc import (
    EXPECTED_ITEM_COUNT,
    FIELDS,
    POLICY_SCHEMA_VERSION,
    RESOLUTION_STATUSES,
    EvidenceQCError,
    assess_evidence_qc,
)

V7_CALIBRATION_SHA256 = "c744640c5866bcfe459f61b9f140b17319bf6bdbe45efd488492f4a3dabc9592"
V8_CALIBRATION_SHA256 = "8992dab93a315ffd0f8f1b3737808c6edcf3e27e18f3df3ac51e494a9e849293"
POLICY_BINDING_COMMENT_ID = "01a05af3-1221-7935-a337-d7262bd96bff"
POLICY_HANDOFF_COMMENT_ID = "01a05af3-f97e-75fa-973d-d42f3c6fc274"
FROZEN_POLICY_TIMESTAMP = "2026-09-01T03:16:18Z"
# The two pins are canonicalized to zeroes while hashing qc_validator.py so
# the engine digest is stable rather than self-referential.
FROZEN_ENGINE_SHA256 = "9f4dc12d7266b62746f940516da73a0f79f920d60d684fd04324ec1d4d902888"
FROZEN_POLICY_SHA256 = "958cb4015cf5c9ca95509273ea5720adf3817de5ed6e665905daa2d3a0f0e3bb"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")

DEFAULT_ENGINE_FILES = (
    "benchmark_harness/calibration/evidence_qc.py",
    "benchmark_harness/calibration/qc_validator.py",
)
ENGINE_HASH_ALGORITHM = "sha256(path\\0canonical-engine-bytes\\n)"
B_FACING_PACK_SCHEMA = "evidence-qc-b-facing-manifest/v1"
B_FACING_BINDING_SCHEMA = "evidence-qc-b-facing-binding/v1"
PILOT_SOURCE_PATHS = frozenset(
    {
        "pilot/annotation-view.jsonl",
        "pilot/exclusion-proof.json",
        "pilot/filter-decisions.jsonl",
        "pilot/items.jsonl",
        "pilot/labels/raw/annotator-a.jsonl",
        "pilot/query-corpus.jsonl",
    }
)
B_FACING_SOURCE_PATHS = frozenset(
    {
        "CALIBRATION.sha256",
        "pilot/annotation-view.jsonl",
        "pilot/exclusion-proof.json",
        "rubric.md",
    }
)
B_FACING_PACK_FILES = frozenset(
    {
        "POLICY.sha256",
        "README.md",
        "engine-manifest.json",
        "inputs/annotation-view.jsonl",
        "policy-binding.json",
        "policy.json",
        "rubric.md",
        "source-manifest.json",
        "source-seal/CALIBRATION.sha256",
        "source-seal/exclusion-proof.json",
    }
)

# These names are deliberately checked recursively in generated QC mappings.
# A reviewer artifact must never smuggle the old gold/probe/claim state into a
# sidecar under a different filename.
FORBIDDEN_FIELDS = frozenset(
    {
        "400",
        "answer",
        "binary_gold",
        "claim_eligible",
        "gold",
        "ground_truth",
        "leakage_positive",
        "model_score",
        "paper_probe_eligible",
        "probe",
        "quality_gates_passed",
        "ranking_score",
        "unrestricted_probe",
        "validated",
    }
)

_POLICY_FIELDS = frozenset(
    {
        "adjudication",
        "amendment_id",
        "binding_comment_id",
        "bootstrap",
        "calibration_id",
        "engine",
        "expected_item_count",
        "fields",
        "frozen_at_utc",
        "frozen_before_labeler_b",
        "handoff_comment_id",
        "integrity",
        "routing",
        "schema_version",
        "source",
        "tiers",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "v7_calibration_sha256",
        "v8_calibration_sha256",
        "v8_pilot_paths",
        "v8_source_seal_path",
        "v8_rubric_sha256",
    }
)
_ENGINE_FIELDS = frozenset({"files", "hash_algorithm", "sha256"})
_FIELD_POLICY_FIELDS = frozenset({"labels", "purpose"})
_BINDING_FIELDS = frozenset(
    {
        "a_label_artifact_sha256",
        "binding_comment_id",
        "b_label_artifact_sha256",
        "engine_sha256",
        "frozen_at_utc",
        "handoff_comment_id",
        "policy_sha256",
        "review_status",
        "schema_version",
        "source_v7_calibration_sha256",
        "source_v8_calibration_sha256",
    }
)
_SOURCE_MANIFEST_FIELDS = frozenset(
    {
        "disjointness_proof",
        "files",
        "pilot_item_count",
        "pilot_strata",
        "schema_version",
        "source_calibration_id",
        "source_v7_calibration_sha256",
        "source_v8_calibration_sha256",
        "source_v8_rubric_sha256",
    }
)
_SOURCE_FILE_FIELDS = frozenset({"bytes", "path", "rows", "sha256"})
_MANIFEST_FILE_FIELDS = frozenset({"bytes", "path", "sha256"})
_SOURCE_FILE_BYTES = {
    "pilot/annotation-view.jsonl": 1_000_832,
    "pilot/filter-decisions.jsonl": 12_036,
    "pilot/items.jsonl": 118_228,
    "pilot/labels/raw/annotator-a.jsonl": 52_061,
    "pilot/query-corpus.jsonl": 39_153,
}
_SOURCE_FILE_ROWS = {
    "pilot/annotation-view.jsonl": 40,
    "pilot/filter-decisions.jsonl": 40,
    "pilot/items.jsonl": 40,
    "pilot/labels/raw/annotator-a.jsonl": 40,
    "pilot/query-corpus.jsonl": 120,
}
_B_FACING_BINDING_FIELDS = frozenset(
    {
        "binding_comment_id",
        "engine_sha256",
        "frozen_at_utc",
        "handoff_comment_id",
        "policy_sha256",
        "review_scope",
        "schema_version",
        "source_v7_calibration_sha256",
        "source_v8_calibration_sha256",
        "status",
    }
)
_B_FACING_ENGINE_MANIFEST_FIELDS = frozenset({"engine_sha256", "files", "schema_version"})
_B_FACING_ENGINE_FILE_FIELDS = frozenset({"path", "sha256"})
_B_FACING_MANIFEST_FIELDS = frozenset(
    {
        "engine_sha256",
        "files",
        "policy_sha256",
        "schema_version",
        "source_v7_calibration_sha256",
        "source_v8_calibration_sha256",
        "status",
    }
)


class QCValidationError(EvidenceQCError):
    """Raised when a policy, source, label, or review artifact fails closed."""


def canonical_json_bytes(value: object) -> bytes:
    """Return the shared cross-runtime JSON representation used by hashes."""

    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _error(message: str) -> QCValidationError:
    return QCValidationError(message)


def _reject_forbidden(value: object, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in FORBIDDEN_FIELDS:
                raise _error(f"forbidden Evidence-QC field at {path}: {key}")
            _reject_forbidden(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden(child, path=f"{path}[{index}]")


def _reject_named_fields(value: object, names: set[str], *, path: str = "root") -> None:
    """Reject recursively named fields that would disclose another vote."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in names:
                raise _error(f"forbidden field at {path}: {key}")
            _reject_named_fields(child, names, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_named_fields(child, names, path=f"{path}[{index}]")


def _strict_fields(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise _error(f"{label} fields are not exact (missing={missing}, extra={extra})")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(f"{label} must be a non-empty string")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise _error(f"{label} must be a lowercase SHA-256 digest")
    return value


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not UTC_RE.fullmatch(value):
        raise _error(f"{label} must be an RFC3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise _error(f"{label} is not a valid UTC timestamp") from exc
    return parsed.astimezone(UTC)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise _error(f"missing JSON artifact: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _error(f"unable to read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise _error(f"JSON artifact must contain an object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Read JSONL without network, cache, or source-root discovery side effects."""

    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise _error(f"missing JSONL artifact: {path}") from exc
    except (OSError, UnicodeError) as exc:
        raise _error(f"unable to read JSONL artifact: {path}") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _error(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise _error(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(dict(row)) for row in rows))


def _repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parents[2]


def _canonical_engine_bytes(relative: str, path: Path) -> bytes:
    """Remove only the two self-binding pins before hashing the QC engine."""

    data = path.read_bytes()
    if relative != "benchmark_harness/calibration/qc_validator.py":
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _error(f"QC engine source is not UTF-8: {relative}") from exc
    for name in ("FROZEN_ENGINE_SHA256", "FROZEN_POLICY_SHA256"):
        pattern = rf'(?m)^({name}\s*=\s*")[0-9a-f]{{64}}(")$'
        text, replacements = re.subn(
            pattern,
            lambda match: f"{match.group(1)}{'0' * 64}{match.group(2)}",
            text,
        )
        if replacements != 1:
            raise _error(f"QC engine source pin is unavailable: {name}")
    return text.encode("utf-8")


def compute_engine_sha256(
    repository_root: str | Path | None = None,
    *,
    files: Sequence[str] = DEFAULT_ENGINE_FILES,
) -> str:
    """Hash canonical QC engine sources with path framing."""

    root = Path(repository_root).expanduser().resolve() if repository_root else _repository_root()
    hasher = hashlib.sha256()
    for relative in sorted(files):
        path = root / relative
        if not path.is_file():
            raise _error(f"QC engine source is unavailable: {relative}")
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_canonical_engine_bytes(relative, path))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _parse_sha_sidecar(path: Path, expected_name: str) -> str:
    try:
        line = next(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except (FileNotFoundError, OSError, UnicodeError, StopIteration) as exc:
        raise _error(f"missing or empty SHA-256 sidecar: {path}") from exc
    parts = line.split()
    if len(parts) != 2 or parts[1] != expected_name:
        raise _error(f"invalid SHA-256 sidecar entry: {path}")
    return _digest(parts[0], str(path))


def load_policy(
    policy_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> tuple[dict[str, object], str]:
    """Load and verify the frozen policy plus its engine binding."""

    path = Path(policy_path).expanduser().resolve()
    value = _read_json(path)
    _reject_forbidden(value)
    _strict_fields(value, _POLICY_FIELDS, "policy")
    if value.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise _error("unsupported Evidence-QC policy schema")
    if value.get("amendment_id") != "Evidence-QC Amendment v8-R1":
        raise _error("unexpected Evidence-QC amendment")
    if value.get("calibration_id") != "leakage-probe/v8":
        raise _error("policy must target the v8 calibration")
    if value.get("binding_comment_id") != POLICY_BINDING_COMMENT_ID:
        raise _error("policy is not bound to the PM implementation-ready spec")
    if value.get("handoff_comment_id") != POLICY_HANDOFF_COMMENT_ID:
        raise _error("policy is not bound to the current DEV handoff")
    if value.get("frozen_at_utc") != FROZEN_POLICY_TIMESTAMP:
        raise _error("policy freeze timestamp is not the frozen v8-R1 timestamp")
    frozen_at = _utc(value.get("frozen_at_utc"), "policy.frozen_at_utc")
    if value.get("frozen_before_labeler_b") is not True:
        raise _error("policy must be frozen before Labeler B")
    if value.get("expected_item_count") != EXPECTED_ITEM_COUNT:
        raise _error("Evidence-QC policy must use exactly 40 pilot items")

    source = value.get("source")
    if not isinstance(source, dict):
        raise _error("policy.source must be an object")
    _strict_fields(source, _SOURCE_FIELDS, "policy.source")
    if source.get("v7_calibration_sha256") != V7_CALIBRATION_SHA256:
        raise _error("policy v7 source seal mismatch")
    if source.get("v8_calibration_sha256") != V8_CALIBRATION_SHA256:
        raise _error("policy v8 source seal mismatch")
    _digest(source.get("v8_rubric_sha256"), "policy.source.v8_rubric_sha256")
    if source.get("v8_source_seal_path") != "data/calibration/leakage-probe/v8/CALIBRATION.sha256":
        raise _error("policy source seal path is not the v8 calibration seal")
    pilot_paths = source.get("v8_pilot_paths")
    if not isinstance(pilot_paths, dict) or set(pilot_paths) != PILOT_SOURCE_PATHS:
        raise _error("policy.source.v8_pilot_paths must bind exactly the six pilot files")
    for name, digest in pilot_paths.items():
        if not isinstance(name, str) or name not in PILOT_SOURCE_PATHS:
            raise _error("policy pilot paths must be restricted to v8/pilot")
        _digest(digest, f"policy.source.v8_pilot_paths.{name}")

    raw_fields = value.get("fields")
    if not isinstance(raw_fields, dict) or set(raw_fields) != set(FIELDS):
        raise _error("policy.fields must define task_relevance and outcome_revealing")
    for field in FIELDS:
        field_policy = raw_fields[field]
        if not isinstance(field_policy, dict):
            raise _error(f"policy.fields.{field} must be an object")
        _strict_fields(field_policy, _FIELD_POLICY_FIELDS, f"policy.fields.{field}")
        labels = field_policy.get("labels")
        if not isinstance(labels, list) or set(labels) != set(
            {"relevant", "not_relevant", "indeterminate"}
            if field == "task_relevance"
            else {"direct", "indirect", "no", "indeterminate"}
        ):
            raise _error(f"policy.fields.{field}.labels are not exact")
        _string(field_policy.get("purpose"), f"policy.fields.{field}.purpose")

    tiers = value.get("tiers")
    if not isinstance(tiers, list):
        raise _error("policy.tiers must be a list")
    expected_tiers = [
        {"name": "strong", "min_agreement_count": 36, "min_kappa": 0.8},
        {"name": "qualified", "min_agreement_count": 34, "min_kappa": 0.6},
        {"name": "limited", "min_agreement_count": 32, "min_kappa": 0.5},
    ]
    if tiers != expected_tiers:
        raise _error("policy tier thresholds are not the frozen v8-R1 values")

    bootstrap = value.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise _error("policy.bootstrap must be an object")
    if bootstrap != {
        "cluster_key": "item_id",
        "confidence": 0.95,
        "resamples": 1000,
        "seed": 20260831,
    }:
        raise _error("policy bootstrap settings are not frozen")
    for name in ("adjudication", "integrity", "routing"):
        if not isinstance(value.get(name), dict):
            raise _error(f"policy.{name} must be an object")
    if value["adjudication"] != {
        "c_must_be_third_identity": True,
        "c_must_not_see_raw_votes": True,
        "c_min_resolved_for_qualified": 36,
        "min_resolved_per_stratum_for_qualified": 8,
        "no_additional_reviewers_until_favorable": True,
        "majority": "two_of_three",
        "same_initial_labels_are_not_replaced": True,
        "three_distinct_or_abstain": "unresolved",
    }:
        raise _error("policy adjudication rules are not frozen")
    if value["integrity"] != {
        "allow_400_input": False,
        "allow_gold_or_ground_truth": False,
        "allow_provider_or_network": False,
        "allow_source_root_writes": False,
        "allow_unrestricted_probe": False,
        "source_seal_must_match": True,
    }:
        raise _error("policy integrity rules are not frozen")
    if value["routing"] != {
        "evidence_dependent_claim_tier": "minimum_of_fields",
        "initial_and_adjudicated_sets_are_disjoint": True,
        "main_outcome_categories": ["no", "indirect", "direct"],
        "point_estimate_excludes": ["unresolved", "protocol_invalid"],
        "sensitivity_sets_are_fixed": [
            "direct_only",
            "revealing_direct_or_indirect",
        ],
        "task_mask_is_shared_across_systems": True,
    }:
        raise _error("policy routing rules are not frozen")

    engine = value.get("engine")
    if not isinstance(engine, dict):
        raise _error("policy.engine must be an object")
    _strict_fields(engine, _ENGINE_FIELDS, "policy.engine")
    if engine.get("hash_algorithm") != ENGINE_HASH_ALGORITHM:
        raise _error("unsupported QC engine hash algorithm")
    engine_files = engine.get("files")
    if not isinstance(engine_files, list) or tuple(engine_files) != DEFAULT_ENGINE_FILES:
        raise _error("policy engine files are not the repository QC engines")
    expected_engine = _digest(engine.get("sha256"), "policy.engine.sha256")
    if expected_engine != FROZEN_ENGINE_SHA256:
        raise _error("policy engine hash is not the frozen QC engine hash")
    actual_engine = compute_engine_sha256(repository_root, files=tuple(engine_files))
    if expected_engine != actual_engine:
        raise _error("QC engine source hash does not match frozen policy")

    policy_digest = sha256_file(path)
    if policy_digest != FROZEN_POLICY_SHA256:
        raise _error("policy.json is not the frozen v8-R1 policy")
    sidecar_digest = _parse_sha_sidecar(path.with_name("POLICY.sha256"), path.name)
    if sidecar_digest != policy_digest:
        raise _error("POLICY.sha256 does not match policy.json")
    # Keep the parsed timestamp live so malformed timezone data cannot be
    # optimized away by future refactors.
    if frozen_at.tzinfo is None:
        raise _error("policy freeze timestamp has no timezone")
    return value, policy_digest


def validate_policy(
    policy_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Return a compact policy validation result."""

    policy, policy_sha256 = load_policy(policy_path, repository_root=repository_root)
    return {
        "engine_sha256": policy["engine"]["sha256"],
        "frozen_at_utc": policy["frozen_at_utc"],
        "policy_sha256": policy_sha256,
        "schema_version": policy["schema_version"],
        "source_v7_calibration_sha256": V7_CALIBRATION_SHA256,
        "source_v8_calibration_sha256": V8_CALIBRATION_SHA256,
    }


def assert_overlay_outside_source(
    overlay_root: str | Path,
    source_root: str | Path,
) -> None:
    """Reject an overlay placed at or below the protected v8 source root."""

    overlay = Path(overlay_root).expanduser().resolve()
    source = Path(source_root).expanduser().resolve()
    try:
        overlay.relative_to(source)
    except ValueError:
        return
    raise _error("Evidence-QC overlay must be outside the v8 source root")


def assert_allowed_source_path(path: str | Path, source_root: str | Path) -> None:
    """Allow only explicit v8/pilot reads and reject every other path."""

    candidate = Path(path).expanduser().resolve()
    source = Path(source_root).expanduser().resolve()
    try:
        relative = candidate.relative_to(source)
    except ValueError as exc:
        raise _error(f"source path is outside the v8 root: {candidate}") from exc
    relative_text = relative.as_posix()
    if relative_text not in PILOT_SOURCE_PATHS:
        raise _error(f"protected v8 source path is not readable: {relative_text}")


def validate_source_pilot(
    source_root: str | Path,
    *,
    policy: Mapping[str, object],
) -> dict[str, object]:
    """Validate source seal and pilot-only exact joins without opening 400 rows."""

    root = Path(source_root).expanduser().resolve()
    source = policy.get("source")
    if not isinstance(source, Mapping):
        raise _error("policy source is unavailable")
    seal_path = root / "CALIBRATION.sha256"
    try:
        first_line = next(
            line for line in seal_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    except (FileNotFoundError, OSError, UnicodeError, StopIteration) as exc:
        raise _error("v8 source calibration seal is unavailable") from exc
    source_digest = first_line.split()[0] if first_line.split() else ""
    if source_digest != source.get("v8_calibration_sha256"):
        raise _error("v8 source calibration seal does not match policy")
    seal_parts = first_line.split()
    if len(seal_parts) != 2 or seal_parts[1] != "artifact_manifest.json":
        raise _error("v8 source calibration seal has an unexpected target")

    pilot_items_path = root / "pilot/items.jsonl"
    pilot_views_path = root / "pilot/annotation-view.jsonl"
    pilot_filter_path = root / "pilot/filter-decisions.jsonl"
    pilot_a_path = root / "pilot/labels/raw/annotator-a.jsonl"
    pilot_query_path = root / "pilot/query-corpus.jsonl"
    for path in (
        pilot_items_path,
        pilot_views_path,
        pilot_filter_path,
        pilot_a_path,
        pilot_query_path,
    ):
        assert_allowed_source_path(path, root)
    items = read_jsonl(pilot_items_path)
    views = read_jsonl(pilot_views_path)
    labels_a = read_jsonl(pilot_a_path)
    _reject_forbidden(items)
    _reject_forbidden(views)
    _reject_forbidden(labels_a)
    if len(items) != EXPECTED_ITEM_COUNT or len(views) != EXPECTED_ITEM_COUNT:
        raise _error("v8 pilot source must contain exactly 40 items and views")
    item_ids = _unique_ids(items, "source pilot items")
    view_ids = _unique_ids(views, "source pilot views")
    label_ids = _unique_ids(labels_a, "source pilot Labeler A")
    if item_ids != view_ids or item_ids != label_ids:
        raise _error("v8 pilot source items, views, and A labels do not join exactly")
    _validate_reviewer_rows(labels_a, "labeler_a", policy)
    proof_path = root / "pilot/exclusion-proof.json"
    assert_allowed_source_path(proof_path, root)
    proof = _read_json(proof_path)
    _reject_forbidden(proof)
    if proof.get("calibration_id") != "leakage-probe/v8" or proof.get("status") != "complete":
        raise _error("v8 pilot exclusion proof is not complete")
    selected = proof.get("selected_digests")
    source_calibrations = proof.get("source_calibrations")
    if (
        not isinstance(selected, Mapping)
        or not isinstance(selected.get("item_id"), list)
        or len(selected["item_id"]) != EXPECTED_ITEM_COUNT
        or not isinstance(source_calibrations, list)
        or "leakage-probe/v7" not in source_calibrations
    ):
        raise _error("v8 pilot exclusion proof does not bind the disjoint 40")
    for view in views:
        item_id = str(view["item_id"])
        view_hash = view.get("annotation_view_sha256")
        if not isinstance(view_hash, str) or not SHA256_RE.fullmatch(view_hash):
            raise _error(f"v8 pilot view hash is invalid for {item_id}")
    view_hashes = {str(row["item_id"]): row.get("annotation_view_sha256") for row in views}
    for row in labels_a:
        if row.get("annotation_view_sha256") != view_hashes[str(row["item_id"])]:
            raise _error("v8 pilot A label/view hash mismatch")
    expected_paths = source.get("v8_pilot_paths")
    if not isinstance(expected_paths, Mapping):
        raise _error("policy pilot path binding is unavailable")
    observed_paths = {
        "pilot/items.jsonl": pilot_items_path,
        "pilot/annotation-view.jsonl": pilot_views_path,
        "pilot/exclusion-proof.json": proof_path,
        "pilot/filter-decisions.jsonl": pilot_filter_path,
        "pilot/labels/raw/annotator-a.jsonl": pilot_a_path,
        "pilot/query-corpus.jsonl": pilot_query_path,
    }
    for relative, path in observed_paths.items():
        expected = expected_paths.get(relative)
        if expected is not None and sha256_file(path) != expected:
            raise _error(f"v8 pilot source hash mismatch: {relative}")
    return {
        "item_count": len(items),
        "label_a_count": len(labels_a),
        "pilot_view_count": len(views),
        "source_v8_calibration_sha256": source_digest,
    }


def validate_source_manifest(
    manifest: Mapping[str, object],
    *,
    policy: Mapping[str, object],
) -> dict[str, object]:
    """Validate a pilot-only source manifest without opening protected rows."""

    _reject_forbidden(manifest)
    _strict_fields(manifest, _SOURCE_MANIFEST_FIELDS, "source manifest")
    if manifest.get("schema_version") != "evidence-qc-source-manifest/v1":
        raise _error("unsupported Evidence-QC source manifest schema")
    if manifest.get("source_calibration_id") != "leakage-probe/v8":
        raise _error("source manifest must target the v8 pilot")
    if manifest.get("source_v7_calibration_sha256") != V7_CALIBRATION_SHA256:
        raise _error("source manifest v7 seal mismatch")
    if manifest.get("source_v8_calibration_sha256") != V8_CALIBRATION_SHA256:
        raise _error("source manifest v8 seal mismatch")
    expected_rubric = policy["source"]["v8_rubric_sha256"]
    if manifest.get("source_v8_rubric_sha256") != expected_rubric:
        raise _error("source manifest rubric mismatch")
    if manifest.get("pilot_item_count") != EXPECTED_ITEM_COUNT:
        raise _error("source manifest pilot count is not 40")
    if manifest.get("pilot_strata") != {
        "A1-T1": 10,
        "B-T1": 10,
        "B-T2": 10,
        "C-T2": 10,
    }:
        raise _error("source manifest strata are not the frozen v8 strata")

    pilot_paths = policy["source"]["v8_pilot_paths"]
    if not isinstance(pilot_paths, Mapping):
        raise _error("policy pilot path binding is unavailable")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != len(_SOURCE_FILE_BYTES):
        raise _error("source manifest file list is incomplete")
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, Mapping):
            raise _error("source manifest file entry is not an object")
        _strict_fields(entry, _SOURCE_FILE_FIELDS, "source manifest file entry")
        path = entry.get("path")
        prefix = "data/calibration/leakage-probe/v8/"
        if not isinstance(path, str) or not path.startswith(prefix):
            raise _error("source manifest contains an unexpected pilot path")
        relative = path.removeprefix(prefix)
        if relative in seen or relative not in _SOURCE_FILE_BYTES:
            raise _error("source manifest contains an unexpected pilot path")
        seen.add(relative)
        if relative not in pilot_paths:
            raise _error("source manifest path is not bound by policy")
        if entry.get("bytes") != _SOURCE_FILE_BYTES[relative]:
            raise _error(f"source manifest byte count mismatch: {relative}")
        if entry.get("rows") != _SOURCE_FILE_ROWS[relative]:
            raise _error(f"source manifest row count mismatch: {relative}")
        if entry.get("sha256") != pilot_paths[relative]:
            raise _error(f"source manifest hash mismatch: {relative}")
    if seen != set(_SOURCE_FILE_BYTES):
        raise _error("source manifest does not cover all pilot files")

    proof = manifest.get("disjointness_proof")
    if not isinstance(proof, Mapping):
        raise _error("source manifest disjointness proof is unavailable")
    _strict_fields(proof, frozenset({"path", "sha256"}), "source manifest disjointness proof")
    if proof != {
        "path": "data/calibration/leakage-probe/v8/pilot/exclusion-proof.json",
        "sha256": pilot_paths["pilot/exclusion-proof.json"],
    }:
        raise _error("source manifest disjointness proof mismatch")
    return {
        "pilot_item_count": EXPECTED_ITEM_COUNT,
        "source_v8_calibration_sha256": V8_CALIBRATION_SHA256,
        "status": "valid",
    }


def _unique_ids(rows: Sequence[Mapping[str, object]], label: str) -> set[str]:
    ids: list[str] = []
    for row in rows:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise _error(f"{label} contains a missing item_id")
        ids.append(item_id)
    if len(ids) != len(set(ids)):
        raise _error(f"{label} contains duplicate item_id values")
    return set(ids)


def _label_times(rows: Sequence[Mapping[str, object]], label: str) -> list[datetime]:
    times: list[datetime] = []
    for row in rows:
        value = row.get("labeled_at_utc")
        if value is not None:
            times.append(_utc(value, f"{label}.labeled_at_utc"))
    return times


def validate_exact_coverage(
    views: Sequence[Mapping[str, object]],
    labels_a: Sequence[Mapping[str, object]],
    labels_b: Sequence[Mapping[str, object]],
    *,
    policy: Mapping[str, object],
) -> dict[str, object]:
    """Validate 40 unique views and exact A/B joins before statistics."""

    expected_count = policy.get("expected_item_count")
    if expected_count != EXPECTED_ITEM_COUNT:
        raise _error("policy item count is not 40")
    if (
        len(views) != expected_count
        or len(labels_a) != expected_count
        or len(labels_b) != expected_count
    ):
        raise _error("A/B/views must each contain exactly 40 rows")
    view_ids = _unique_ids(views, "views")
    ids_a = _unique_ids(labels_a, "Labeler A")
    ids_b = _unique_ids(labels_b, "Labeler B")
    if view_ids != ids_a or view_ids != ids_b:
        raise _error("A/B/views do not have exact 40-item coverage")
    view_hashes: dict[str, str] = {}
    for view in views:
        _reject_forbidden(view)
        item_id = str(view["item_id"])
        view_hash = view.get("annotation_view_sha256")
        if not isinstance(view_hash, str) or not SHA256_RE.fullmatch(view_hash):
            raise _error(f"view hash is invalid for {item_id}")
        view_hashes[item_id] = view_hash
    for reviewer, rows in (("A", labels_a), ("B", labels_b)):
        for row in rows:
            item_id = str(row["item_id"])
            if row.get("annotation_view_sha256") != view_hashes[item_id]:
                raise _error(f"Labeler {reviewer} view hash mismatch for {item_id}")
            _digest(row.get("annotation_view_sha256"), f"Labeler {reviewer} view hash")
    _validate_reviewer_rows(labels_a, "labeler_a", policy)
    _validate_reviewer_rows(labels_b, "labeler_b", policy)
    a_times = _label_times(labels_a, "Labeler A")
    b_times = _label_times(labels_b, "Labeler B")
    if len(a_times) != len(labels_a) or len(b_times) != len(labels_b):
        raise _error("all A/B rows must carry immutable labeling timestamps")
    frozen_at = _utc(policy.get("frozen_at_utc"), "policy.frozen_at_utc")
    if b_times and any(value <= frozen_at for value in b_times):
        raise _error("Labeler B timestamps must be after the frozen policy")
    if a_times and b_times and max(a_times) >= min(b_times):
        raise _error("Labeler A must be completed before Labeler B")
    return {"item_count": expected_count, "view_ids": sorted(view_ids)}


def _validate_reviewer_rows(
    rows: Sequence[Mapping[str, object]],
    expected_type: str,
    policy: Mapping[str, object],
) -> None:
    reviewer_ids: set[str] = set()
    source = policy.get("source")
    expected_rubric = source.get("v8_rubric_sha256") if isinstance(source, Mapping) else None
    for row in rows:
        _reject_forbidden(row)
        reviewer_id = row.get("reviewer_id")
        if not isinstance(reviewer_id, str) or not reviewer_id.strip():
            raise _error(f"{expected_type} row is missing reviewer_id")
        reviewer_ids.add(reviewer_id)
        reviewer_type = row.get("reviewer_type")
        if reviewer_type is not None and reviewer_type != expected_type:
            raise _error(f"{expected_type} row has an unexpected reviewer_type")
        rubric_sha = row.get("rubric_sha256")
        if expected_rubric is not None and rubric_sha != expected_rubric:
            raise _error(f"{expected_type} row is bound to a different rubric")
        for field in FIELDS:
            labels = row.get(field)
            allowed = (
                {"relevant", "not_relevant", "indeterminate"}
                if field == "task_relevance"
                else {"direct", "indirect", "no", "indeterminate"}
            )
            if labels not in allowed:
                raise _error(f"{expected_type} has an invalid {field} value")
    if len(reviewer_ids) != 1:
        raise _error(f"{expected_type} must have one stable reviewer identity")


def validate_reviewer_independence(
    labels_a: Sequence[Mapping[str, object]],
    labels_b: Sequence[Mapping[str, object]],
    *,
    policy: Mapping[str, object],
) -> dict[str, object]:
    """Verify A/B are distinct blind identities and B follows the freeze."""

    _validate_reviewer_rows(labels_a, "labeler_a", policy)
    _validate_reviewer_rows(labels_b, "labeler_b", policy)
    reviewer_a = str(labels_a[0]["reviewer_id"])
    reviewer_b = str(labels_b[0]["reviewer_id"])
    if reviewer_a == reviewer_b:
        raise _error("Labeler A and B reviewer identities must be distinct")
    return {"reviewer_a": reviewer_a, "reviewer_b": reviewer_b, "independent": True}


def validate_blind_c(
    labels_c: Sequence[Mapping[str, object]],
    queue: Sequence[Mapping[str, object]],
    views: Sequence[Mapping[str, object]],
    *,
    policy: Mapping[str, object],
    labels_a: Sequence[Mapping[str, object]] | None = None,
    labels_b: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Validate C's limited queue join without exposing A/B values to C."""

    queue_ids = [row.get("item_id") for row in queue]
    if any(not isinstance(item_id, str) for item_id in queue_ids) or len(queue_ids) != len(
        set(queue_ids)
    ):
        raise _error("C queue IDs must be unique strings")
    view_hashes = {str(view["item_id"]): view.get("annotation_view_sha256") for view in views}
    expected_ids = set(queue_ids)
    seen: set[str] = set()
    reviewer_ids: set[str] = set()
    frozen_at = _utc(policy.get("frozen_at_utc"), "policy.frozen_at_utc")
    source = policy.get("source")
    expected_rubric = source.get("v8_rubric_sha256") if isinstance(source, Mapping) else None
    hidden_vote_fields = {
        "a_label",
        "annotator_a",
        "annotator_b",
        "b_label",
        "label_a",
        "label_b",
        "labeler_a",
        "labeler_b",
        "raw_a",
        "raw_b",
        "vote_a",
        "vote_b",
    }
    for row in labels_c:
        _reject_forbidden(row)
        _reject_named_fields(row, hidden_vote_fields)
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or item_id in seen:
            raise _error("C rows must be unique queued item IDs")
        if item_id not in expected_ids:
            raise _error("C may label only the blind disagreement queue")
        seen.add(item_id)
        if row.get("annotation_view_sha256") != view_hashes.get(item_id):
            raise _error("C view hash does not match the frozen view")
        reviewer_id = row.get("reviewer_id")
        if not isinstance(reviewer_id, str) or not reviewer_id:
            raise _error("C row is missing reviewer_id")
        reviewer_ids.add(reviewer_id)
        reviewer_type = row.get("reviewer_type")
        if reviewer_type is not None and reviewer_type != "reviewer_c":
            raise _error("C row has an unexpected reviewer_type")
        rubric_sha = row.get("rubric_sha256")
        if expected_rubric is not None and rubric_sha != expected_rubric:
            raise _error("C row is bound to a different rubric")
        label_time = row.get("labeled_at_utc")
        if label_time is None:
            raise _error("all C rows must carry immutable labeling timestamps")
        if _utc(label_time, "C.labeled_at_utc") <= frozen_at:
            raise _error("C timestamp must follow the frozen policy")
        for field in FIELDS:
            value = row.get(field)
            if value is None or (isinstance(value, str) and value in {"abstain", "indeterminate"}):
                continue
            allowed = (
                {"relevant", "not_relevant"}
                if field == "task_relevance"
                else {"direct", "indirect", "no"}
            )
            if value not in allowed:
                raise _error(f"C has an invalid {field} value")
    if labels_c and len(reviewer_ids) != 1:
        raise _error("C must have one stable reviewer identity")
    if reviewer_ids and labels_a is not None and labels_b is not None:
        prior_ids = {
            row.get("reviewer_id") for row in (*labels_a, *labels_b) if row.get("reviewer_id")
        }
        if reviewer_ids.intersection(prior_ids):
            raise _error("C must be a third reviewer identity")
    return {
        "queued_count": len(queue),
        "c_count": len(labels_c),
        "reviewer_count": len(reviewer_ids),
    }


def validate_blind_queue(queue: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Ensure the C queue contains no label values or hidden reviewer context."""

    seen: set[str] = set()
    previous = ""
    for row in queue:
        _strict_fields(
            row,
            frozenset({"annotation_view_sha256", "disagreement_fields", "item_id", "queue_status"}),
            "disagreement queue row",
        )
        item_id = _string(row.get("item_id"), "queue.item_id")
        if item_id in seen or item_id < previous:
            raise _error("disagreement queue must be sorted and unique")
        seen.add(item_id)
        previous = item_id
        _digest(row.get("annotation_view_sha256"), "queue.annotation_view_sha256")
        fields = row.get("disagreement_fields")
        if (
            not isinstance(fields, list)
            or not fields
            or any(field not in FIELDS for field in fields)
        ):
            raise _error("queue disagreement_fields are invalid")
        if row.get("queue_status") != "pending_c":
            raise _error("queue rows must be pending_c")
    _reject_forbidden(list(queue))
    return {"queued_count": len(queue), "queue_item_ids": sorted(seen)}


def _safe_relative(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise _error(f"artifact path escapes review root: {path}") from exc
    if not relative.parts or ".." in relative.parts:
        raise _error(f"invalid artifact relative path: {relative}")
    return relative.as_posix()


def _manifest_files(root: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "QC.sha256":
            continue
        relative = _safe_relative(path, root)
        files.append(
            {
                "bytes": path.stat().st_size,
                "path": relative,
                "sha256": sha256_file(path),
            }
        )
    return files


def _source_evidence_file(source_root: str | Path, relative: str) -> Path:
    """Resolve one source file that is safe to expose to independent B."""

    root = Path(source_root).expanduser().resolve()
    candidate = (root / relative).resolve()
    try:
        observed = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise _error(f"source evidence path escapes the v8 root: {relative}") from exc
    if observed not in B_FACING_SOURCE_PATHS:
        raise _error(f"source evidence path is not B-facing safe: {relative}")
    if not candidate.is_file():
        raise _error(f"source evidence file is unavailable: {relative}")
    return candidate


def _read_v8_source_seal(source_root: str | Path) -> str:
    """Read only the v8 seal line; never discover or open protected rows."""

    digest = _parse_sha_sidecar(
        _source_evidence_file(source_root, "CALIBRATION.sha256"),
        "artifact_manifest.json",
    )
    if digest != V8_CALIBRATION_SHA256:
        raise _error("v8 source calibration seal does not match policy")
    return digest


def _validate_b_facing_views(
    path: Path,
    *,
    policy: Mapping[str, object],
) -> dict[str, object]:
    """Validate the 40 model-visible rows without loading any reviewer vote."""

    source = policy.get("source")
    if not isinstance(source, Mapping) or not isinstance(source.get("v8_pilot_paths"), Mapping):
        raise _error("policy pilot path binding is unavailable")
    expected = source["v8_pilot_paths"].get("pilot/annotation-view.jsonl")
    _digest(expected, "policy pilot annotation-view hash")
    if sha256_file(path) != expected:
        raise _error("B-facing annotation view hash does not match policy")
    views = read_jsonl(path)
    if len(views) != EXPECTED_ITEM_COUNT:
        raise _error("B-facing annotation view must contain exactly 40 rows")
    _unique_ids(views, "B-facing annotation views")
    hidden_fields = set(FIELDS) | {
        "label",
        "labeled_at_utc",
        "reviewer_id",
        "reviewer_type",
        "rubric_sha256",
        "vote",
    }
    for row in views:
        _reject_forbidden(row)
        _reject_named_fields(row, hidden_fields)
        _digest(row.get("annotation_view_sha256"), "B-facing annotation view hash")
    return {"item_count": EXPECTED_ITEM_COUNT, "view_count": len(views)}


def _validate_exclusion_proof(path: Path) -> dict[str, object]:
    """Validate hash-only disjointness evidence copied into the B-facing pack."""

    proof = _read_json(path)
    _reject_forbidden(proof)
    if proof.get("calibration_id") != "leakage-probe/v8" or proof.get("status") != "complete":
        raise _error("B-facing exclusion proof is not complete")
    source_calibrations = proof.get("source_calibrations")
    if not isinstance(source_calibrations, list) or "leakage-probe/v7" not in source_calibrations:
        raise _error("B-facing exclusion proof does not bind the v7 disjoint source")
    excluded = proof.get("excluded_digests")
    if not isinstance(excluded, Mapping) or not excluded:
        raise _error("B-facing exclusion proof has no digest sets")
    for name, values in excluded.items():
        if not isinstance(name, str) or not isinstance(values, list):
            raise _error("B-facing exclusion proof digest sets are malformed")
        for value in values:
            _digest(value, f"B-facing exclusion proof.{name}")
    return {"status": "valid"}


def _validate_engine_manifest(
    path: Path,
    *,
    policy: Mapping[str, object],
    repository_root: str | Path | None,
) -> dict[str, object]:
    """Bind the clean-runtime source/test files to the policy engine digest."""

    manifest = _read_json(path)
    _reject_forbidden(manifest)
    _strict_fields(manifest, _B_FACING_ENGINE_MANIFEST_FIELDS, "B-facing engine manifest")
    if manifest.get("schema_version") != "evidence-qc-engine-manifest/v1":
        raise _error("unsupported B-facing engine manifest schema")
    engine = policy.get("engine")
    if not isinstance(engine, Mapping):
        raise _error("policy engine binding is unavailable")
    expected_engine = engine.get("sha256")
    if manifest.get("engine_sha256") != expected_engine:
        raise _error("B-facing engine manifest hash mismatch")
    expected_paths = set(DEFAULT_ENGINE_FILES) | {
        "tests/test_evidence_qc.py",
        "tests/test_relevance_v8.py",
    }
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != len(expected_paths):
        raise _error("B-facing engine manifest file list is incomplete")
    root = Path(repository_root).expanduser().resolve() if repository_root else _repository_root()
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise _error("B-facing engine manifest entry is not an object")
        _strict_fields(entry, _B_FACING_ENGINE_FILE_FIELDS, "B-facing engine manifest entry")
        relative = entry.get("path")
        if not isinstance(relative, str) or relative in seen or relative not in expected_paths:
            raise _error("B-facing engine manifest contains an unexpected path")
        seen.add(relative)
        _digest(entry.get("sha256"), f"B-facing engine manifest.{relative}")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise _error("B-facing engine manifest path escapes the repository") from exc
        if not candidate.is_file() or sha256_file(candidate) != entry.get("sha256"):
            raise _error(f"B-facing engine source/test hash mismatch: {relative}")
    if seen != expected_paths:
        raise _error("B-facing engine manifest does not cover source and tests")
    return {"engine_sha256": expected_engine, "file_count": len(seen)}


def _validate_pack_manifest_files(
    root: Path,
    manifest: Mapping[str, object],
    *,
    allowed_paths: frozenset[str],
    label: str,
) -> str:
    """Check a manifest and its append-only file set before reading contents."""

    qc_path = root / "QC.sha256"
    manifest_path = root / "manifest.json"
    expected_manifest_sha = _parse_sha_sidecar(qc_path, manifest_path.name)
    if expected_manifest_sha != sha256_file(manifest_path):
        raise _error("QC.sha256 does not match manifest.json")
    listed = manifest.get("files")
    if not isinstance(listed, list):
        raise _error(f"{label} files are unavailable")
    expected_by_path: dict[str, Mapping[str, object]] = {}
    for entry in listed:
        if not isinstance(entry, Mapping):
            raise _error(f"{label} file entry is not an object")
        _strict_fields(entry, _MANIFEST_FILE_FIELDS, f"{label} file entry")
        path_text = entry.get("path")
        if not isinstance(path_text, str) or path_text in expected_by_path:
            raise _error(f"{label} has duplicate or invalid paths")
        if path_text not in allowed_paths:
            raise _error(f"{label} contains an unexpected path: {path_text}")
        lower_path = path_text.lower()
        if any(
            fragment in lower_path
            for fragment in ("400", "adjudicated", "annotator-a", "annotator-b", "gold", "probe")
        ):
            raise _error(f"{label} exposes a disallowed artifact path: {path_text}")
        candidate = (root / path_text).resolve()
        if _safe_relative(candidate, root) != path_text or not candidate.is_file():
            raise _error(f"{label} path is unavailable: {path_text}")
        if entry.get("bytes") != candidate.stat().st_size or entry.get("sha256") != sha256_file(
            candidate
        ):
            raise _error(f"{label} hash mismatch: {path_text}")
        expected_by_path[path_text] = entry
    actual_paths = {
        relative
        for relative in (_safe_relative(path, root) for path in root.rglob("*"))
        if (root / relative).is_file() and relative not in {"QC.sha256", "manifest.json"}
    }
    if actual_paths != set(expected_by_path) or actual_paths != set(allowed_paths):
        raise _error(f"{label} does not cover exactly the allowed review files")
    return expected_manifest_sha


def _validate_b_facing_pack_contents(
    root: Path,
    *,
    manifest: Mapping[str, object],
    policy: Mapping[str, object],
    policy_sha256: str,
    policy_path: str | Path,
    source_root: str | Path | None,
    repository_root: str | Path | None,
) -> dict[str, object]:
    """Validate a B-facing pack without reading raw A, 400, gold, or probe data."""

    if Path(policy_path).expanduser().resolve() != (root / "policy.json").resolve():
        raise _error("B-facing validation requires the pack's frozen policy.json")
    if source_root is not None:
        assert_overlay_outside_source(root, source_root)
        _read_v8_source_seal(source_root)
    _reject_forbidden(manifest)
    _strict_fields(manifest, _B_FACING_MANIFEST_FIELDS, "B-facing manifest")
    if manifest.get("schema_version") != B_FACING_PACK_SCHEMA:
        raise _error("unsupported B-facing review manifest schema")
    if manifest.get("policy_sha256") != policy_sha256:
        raise _error("B-facing manifest is bound to a different policy")
    if manifest.get("engine_sha256") != policy["engine"]["sha256"]:
        raise _error("B-facing manifest engine hash mismatch")
    if manifest.get("source_v7_calibration_sha256") != V7_CALIBRATION_SHA256:
        raise _error("B-facing manifest v7 seal mismatch")
    if manifest.get("source_v8_calibration_sha256") != V8_CALIBRATION_SHA256:
        raise _error("B-facing manifest v8 seal mismatch")
    if manifest.get("status") != "ready_for_independent_b":
        raise _error("B-facing manifest has an unexpected status")
    manifest_sha256 = _validate_pack_manifest_files(
        root,
        manifest,
        allowed_paths=B_FACING_PACK_FILES,
        label="B-facing manifest",
    )

    source_manifest = _read_json(root / "source-manifest.json")
    validate_source_manifest(source_manifest, policy=policy)
    if sha256_file(root / "rubric.md") != policy["source"]["v8_rubric_sha256"]:
        raise _error("B-facing rubric hash mismatch")
    _validate_b_facing_views(root / "inputs/annotation-view.jsonl", policy=policy)
    seal_digest = _parse_sha_sidecar(
        root / "source-seal/CALIBRATION.sha256", "artifact_manifest.json"
    )
    if seal_digest != V8_CALIBRATION_SHA256:
        raise _error("B-facing copied source seal mismatch")
    _validate_exclusion_proof(root / "source-seal/exclusion-proof.json")
    source_paths = policy["source"]["v8_pilot_paths"]
    if (
        sha256_file(root / "source-seal/exclusion-proof.json")
        != source_paths["pilot/exclusion-proof.json"]
    ):
        raise _error("B-facing exclusion proof hash mismatch")

    binding = _read_json(root / "policy-binding.json")
    _reject_forbidden(binding)
    _strict_fields(binding, _B_FACING_BINDING_FIELDS, "B-facing policy binding")
    if binding.get("schema_version") != B_FACING_BINDING_SCHEMA:
        raise _error("unsupported B-facing policy-binding schema")
    if binding.get("policy_sha256") != policy_sha256:
        raise _error("B-facing policy binding policy hash mismatch")
    if binding.get("engine_sha256") != policy["engine"]["sha256"]:
        raise _error("B-facing policy binding engine hash mismatch")
    if binding.get("source_v7_calibration_sha256") != V7_CALIBRATION_SHA256:
        raise _error("B-facing policy binding v7 seal mismatch")
    if binding.get("source_v8_calibration_sha256") != V8_CALIBRATION_SHA256:
        raise _error("B-facing policy binding v8 seal mismatch")
    if binding.get("frozen_at_utc") != policy["frozen_at_utc"]:
        raise _error("B-facing policy binding freeze timestamp mismatch")
    if binding.get("review_scope") != "pilot-only-independent-review":
        raise _error("B-facing policy binding scope mismatch")
    if binding.get("status") != "awaiting_independent_labeler_b":
        raise _error("B-facing policy binding has an unexpected status")
    _validate_engine_manifest(
        root / "engine-manifest.json",
        policy=policy,
        repository_root=repository_root,
    )
    return {
        "engine_sha256": policy["engine"]["sha256"],
        "manifest_sha256": manifest_sha256,
        "policy_sha256": policy_sha256,
        "schema_version": B_FACING_PACK_SCHEMA,
        "status": manifest["status"],
        "view_count": EXPECTED_ITEM_COUNT,
    }


def write_b_facing_pack(
    output_root: str | Path,
    *,
    policy_path: str | Path,
    source_root: str | Path,
    source_manifest_path: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Create the immutable pilot-only pack presented to independent B."""

    root = Path(output_root).expanduser().resolve()
    if root.exists():
        raise _error(f"append-only B-facing root already exists: {root}")
    repository = (
        Path(repository_root).expanduser().resolve() if repository_root else _repository_root()
    )
    policy, policy_sha256 = load_policy(policy_path, repository_root=repository)
    source_root_path = Path(source_root).expanduser().resolve()
    assert_overlay_outside_source(root, source_root_path)
    source_manifest_source = (
        Path(source_manifest_path).expanduser().resolve()
        if source_manifest_path
        else Path(policy_path).expanduser().resolve().with_name("source-manifest.json")
    )
    source_manifest = _read_json(source_manifest_source)
    validate_source_manifest(source_manifest, policy=policy)
    _read_v8_source_seal(source_root_path)
    root.mkdir(parents=True, exist_ok=False)
    try:
        view_source = _source_evidence_file(source_root_path, "pilot/annotation-view.jsonl")
        proof_source = _source_evidence_file(source_root_path, "pilot/exclusion-proof.json")
        rubric_source = _source_evidence_file(source_root_path, "rubric.md")
        source_paths = policy["source"]["v8_pilot_paths"]
        if sha256_file(view_source) != source_paths["pilot/annotation-view.jsonl"]:
            raise _error("v8 pilot annotation view does not match policy")
        if sha256_file(proof_source) != source_paths["pilot/exclusion-proof.json"]:
            raise _error("v8 pilot exclusion proof does not match policy")
        if sha256_file(rubric_source) != policy["source"]["v8_rubric_sha256"]:
            raise _error("v8 rubric does not match policy")

        (root / "inputs").mkdir()
        (root / "source-seal").mkdir()
        shutil.copyfile(view_source, root / "inputs/annotation-view.jsonl")
        shutil.copyfile(rubric_source, root / "rubric.md")
        shutil.copyfile(source_manifest_source, root / "source-manifest.json")
        shutil.copyfile(policy_path, root / "policy.json")
        shutil.copyfile(
            Path(policy_path).expanduser().resolve().with_name("POLICY.sha256"),
            root / "POLICY.sha256",
        )
        shutil.copyfile(
            _source_evidence_file(source_root_path, "CALIBRATION.sha256"),
            root / "source-seal/CALIBRATION.sha256",
        )
        shutil.copyfile(proof_source, root / "source-seal/exclusion-proof.json")
        (root / "README.md").write_text(
            f"""# v8-R1 B-facing review pack

This immutable pack is the only input surface supplied to independent Labeler B.
It contains the 40 frozen model-visible views, the frozen rubric, policy and
source-binding evidence, and no reviewer labels or derived agreement outputs.
The v7 seal is `{V7_CALIBRATION_SHA256}` and the v8 seal is
`{V8_CALIBRATION_SHA256}`. Policy freeze is `{policy["frozen_at_utc"]}` and
`frozen_before_labeler_b` is true.

The pack is append-only: do not edit it after B starts. B's labels belong in a
new external output directory and are checked by `qc-assess` only after this
pack and the independent B artifact are sealed.

From a clean checkout, verify the complete manifest and hashes with:

```text
uv run benchmark-harness qc-validate \\
  --policy data/calibration/evidence-qc/v8-r1/b-facing-review-pack/policy.json \\
  --root data/calibration/evidence-qc/v8-r1/b-facing-review-pack \\
  --source-root data/calibration/leakage-probe/v8
```

The validator reads only the source seal when `--source-root` is supplied; it
does not open raw A, the 400-item source, gold, or probe material.
""",
            encoding="utf-8",
        )
        engine_paths = sorted(
            set(DEFAULT_ENGINE_FILES) | {"tests/test_evidence_qc.py", "tests/test_relevance_v8.py"}
        )
        engine_manifest = {
            "engine_sha256": policy["engine"]["sha256"],
            "files": [
                {"path": relative, "sha256": sha256_file(repository / relative)}
                for relative in engine_paths
            ],
            "schema_version": "evidence-qc-engine-manifest/v1",
        }
        _reject_forbidden(engine_manifest)
        _write_json(root / "engine-manifest.json", engine_manifest)
        binding = {
            "binding_comment_id": policy["binding_comment_id"],
            "engine_sha256": policy["engine"]["sha256"],
            "frozen_at_utc": policy["frozen_at_utc"],
            "handoff_comment_id": policy["handoff_comment_id"],
            "policy_sha256": policy_sha256,
            "review_scope": "pilot-only-independent-review",
            "schema_version": B_FACING_BINDING_SCHEMA,
            "source_v7_calibration_sha256": policy["source"]["v7_calibration_sha256"],
            "source_v8_calibration_sha256": policy["source"]["v8_calibration_sha256"],
            "status": "awaiting_independent_labeler_b",
        }
        _reject_forbidden(binding)
        _write_json(root / "policy-binding.json", binding)
        manifest = {
            "engine_sha256": policy["engine"]["sha256"],
            "files": _manifest_files(root),
            "policy_sha256": policy_sha256,
            "schema_version": B_FACING_PACK_SCHEMA,
            "source_v7_calibration_sha256": policy["source"]["v7_calibration_sha256"],
            "source_v8_calibration_sha256": policy["source"]["v8_calibration_sha256"],
            "status": "ready_for_independent_b",
        }
        _reject_forbidden(manifest)
        _write_json(root / "manifest.json", manifest)
        manifest_sha256 = sha256_file(root / "manifest.json")
        (root / "QC.sha256").write_text(f"{manifest_sha256}  manifest.json\n", encoding="utf-8")
        checked = _validate_b_facing_pack_contents(
            root,
            manifest=manifest,
            policy=policy,
            policy_sha256=policy_sha256,
            policy_path=root / "policy.json",
            source_root=source_root_path,
            repository_root=repository,
        )
        return {"output_root": str(root), **checked}
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def validate_b_facing_pack(
    review_root: str | Path,
    *,
    policy_path: str | Path,
    source_root: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate a B-facing pack through its frozen policy and source seal."""

    root = Path(review_root).expanduser().resolve()
    policy, policy_sha256 = load_policy(policy_path, repository_root=repository_root)
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema_version") != B_FACING_PACK_SCHEMA:
        raise _error("review root is not a B-facing Evidence-QC pack")
    return _validate_b_facing_pack_contents(
        root,
        manifest=manifest,
        policy=policy,
        policy_sha256=policy_sha256,
        policy_path=policy_path,
        source_root=source_root,
        repository_root=repository_root,
    )


def write_review_pack(
    output_root: str | Path,
    *,
    policy: Mapping[str, object],
    policy_sha256: str,
    views: Sequence[Mapping[str, object]],
    labels_a: Sequence[Mapping[str, object]],
    labels_b: Sequence[Mapping[str, object]],
    assessment: Mapping[str, object],
    labels_c: Sequence[Mapping[str, object]] | None = None,
    source_manifest: Mapping[str, object] | None = None,
    source_root: str | Path | None = None,
) -> dict[str, object]:
    """Write one new review pack; an existing output directory is never changed."""

    root = Path(output_root).expanduser().resolve()
    if source_root is not None:
        assert_overlay_outside_source(root, source_root)
    if root.exists():
        raise _error(f"append-only review root already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    labels_c = list(labels_c or [])
    try:
        _reject_forbidden(assessment)
        validate_exact_coverage(views, labels_a, labels_b, policy=policy)
        validate_reviewer_independence(labels_a, labels_b, policy=policy)
        queue = assessment.get("disagreement_queue")
        if not isinstance(queue, list):
            raise _error("assessment disagreement_queue is unavailable")
        validate_blind_queue(queue)
        validate_blind_c(
            labels_c,
            queue,
            views,
            policy=policy,
            labels_a=labels_a,
            labels_b=labels_b,
        )
        if labels_c:
            resolution = assessment.get("resolution")
            if not isinstance(resolution, list):
                raise _error("assessment resolution is unavailable")
        else:
            resolution = assessment.get("resolution", [])

        inputs = root / "inputs"
        labels_root = root / "labels/raw"
        inputs.mkdir()
        labels_root.mkdir(parents=True)
        sorted_views = sorted(views, key=lambda row: str(row["item_id"]))
        sorted_a = sorted(labels_a, key=lambda row: str(row["item_id"]))
        sorted_b = sorted(labels_b, key=lambda row: str(row["item_id"]))
        _write_jsonl(inputs / "annotation-view.jsonl", sorted_views)
        _write_jsonl(labels_root / "annotator-a.jsonl", sorted_a)
        _write_jsonl(labels_root / "annotator-b.jsonl", sorted_b)
        if labels_c:
            _write_jsonl(
                labels_root / "reviewer-c.jsonl",
                sorted(labels_c, key=lambda row: str(row["item_id"])),
            )

        metrics = assessment.get("metrics")
        routing = assessment.get("routing")
        sensitivity_plan = assessment.get("sensitivity_plan")
        if (
            not isinstance(metrics, Mapping)
            or not isinstance(routing, Mapping)
            or not isinstance(sensitivity_plan, Mapping)
        ):
            raise _error("assessment is missing metrics, routing, or sensitivity plan")
        _write_json(root / "pre-adjudication-metrics.json", metrics)
        _write_jsonl(root / "disagreement-queue.jsonl", queue)
        _write_jsonl(root / "resolution.jsonl", resolution)
        _write_json(root / "routing.json", routing)
        _write_json(root / "sensitivity-plan.json", sensitivity_plan)
        if source_manifest is not None:
            validate_source_manifest(source_manifest, policy=policy)
            _write_json(root / "source-manifest.json", source_manifest)

        binding = {
            "a_label_artifact_sha256": sha256_file(labels_root / "annotator-a.jsonl"),
            "binding_comment_id": policy["binding_comment_id"],
            "b_label_artifact_sha256": sha256_file(labels_root / "annotator-b.jsonl"),
            "engine_sha256": policy["engine"]["sha256"],
            "frozen_at_utc": policy["frozen_at_utc"],
            "handoff_comment_id": policy["handoff_comment_id"],
            "policy_sha256": policy_sha256,
            "review_status": assessment.get("qc_disposition"),
            "schema_version": "evidence-qc-policy-binding/v1",
            "source_v7_calibration_sha256": policy["source"]["v7_calibration_sha256"],
            "source_v8_calibration_sha256": policy["source"]["v8_calibration_sha256"],
        }
        _reject_forbidden(binding)
        _write_json(root / "policy-binding.json", binding)
        manifest = {
            "files": _manifest_files(root),
            "policy_sha256": policy_sha256,
            "qc_disposition": assessment.get("qc_disposition"),
            "schema_version": "evidence-qc-manifest/v1",
            "source_v8_calibration_sha256": policy["source"]["v8_calibration_sha256"],
        }
        _reject_forbidden(manifest)
        _write_json(root / "manifest.json", manifest)
        manifest_hash = sha256_file(root / "manifest.json")
        (root / "QC.sha256").write_text(f"{manifest_hash}  manifest.json\n", encoding="utf-8")
        return {
            "manifest_sha256": manifest_hash,
            "output_root": str(root),
            "qc_disposition": assessment.get("qc_disposition"),
            "schema_version": "evidence-qc-manifest/v1",
        }
    except Exception:
        # A failed construction must not leave a misleading partial pack that
        # a later run could accidentally treat as append-only evidence.
        shutil.rmtree(root, ignore_errors=True)
        raise


def validate_review_pack(
    review_root: str | Path,
    *,
    policy_path: str | Path,
    source_root: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Verify a review pack's hashes, joins, blind queue, and no-gold boundary."""

    root = Path(review_root).expanduser().resolve()
    policy, policy_sha256 = load_policy(policy_path, repository_root=repository_root)
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") == B_FACING_PACK_SCHEMA:
        return _validate_b_facing_pack_contents(
            root,
            manifest=manifest,
            policy=policy,
            policy_sha256=policy_sha256,
            policy_path=policy_path,
            source_root=source_root,
            repository_root=repository_root,
        )
    if source_root is not None:
        assert_overlay_outside_source(root, source_root)
        validate_source_pilot(source_root, policy=policy)
    qc_path = root / "QC.sha256"
    _reject_forbidden(manifest)
    if manifest.get("schema_version") != "evidence-qc-manifest/v1":
        raise _error("unsupported review manifest schema")
    if manifest.get("policy_sha256") != policy_sha256:
        raise _error("review manifest is bound to a different policy")
    if manifest.get("source_v8_calibration_sha256") != V8_CALIBRATION_SHA256:
        raise _error("review manifest v8 seal mismatch")
    expected_manifest_sha = _parse_sha_sidecar(qc_path, manifest_path.name)
    if expected_manifest_sha != sha256_file(manifest_path):
        raise _error("QC.sha256 does not match manifest.json")
    listed = manifest.get("files")
    if not isinstance(listed, list):
        raise _error("review manifest files are unavailable")
    expected_by_path: dict[str, Mapping[str, object]] = {}
    for entry in listed:
        if not isinstance(entry, Mapping):
            raise _error("review manifest file entry is not an object")
        path_text = entry.get("path")
        if not isinstance(path_text, str) or path_text in expected_by_path:
            raise _error("review manifest has duplicate or invalid paths")
        candidate = (root / path_text).resolve()
        if _safe_relative(candidate, root) != path_text or not candidate.is_file():
            raise _error(f"review manifest path is unavailable: {path_text}")
        if entry.get("bytes") != candidate.stat().st_size or entry.get("sha256") != sha256_file(
            candidate
        ):
            raise _error(f"review manifest hash mismatch: {path_text}")
        expected_by_path[path_text] = entry
    actual_paths = {
        relative
        for relative in (_safe_relative(path, root) for path in root.rglob("*"))
        if (root / relative).is_file() and relative not in {"QC.sha256", "manifest.json"}
    }
    if actual_paths != set(expected_by_path):
        raise _error("review manifest does not cover exactly the review files")
    source_manifest_path = root / "source-manifest.json"
    if source_manifest_path.exists():
        validate_source_manifest(_read_json(source_manifest_path), policy=policy)

    binding = _read_json(root / "policy-binding.json")
    _reject_forbidden(binding)
    _strict_fields(binding, _BINDING_FIELDS, "policy-binding")
    if binding.get("schema_version") != "evidence-qc-policy-binding/v1":
        raise _error("unsupported policy-binding schema")
    if binding.get("policy_sha256") != policy_sha256:
        raise _error("policy-binding policy hash mismatch")
    if binding.get("engine_sha256") != policy["engine"]["sha256"]:
        raise _error("policy-binding engine hash mismatch")
    if binding.get("source_v8_calibration_sha256") != V8_CALIBRATION_SHA256:
        raise _error("policy-binding v8 seal mismatch")
    if binding.get("source_v7_calibration_sha256") != V7_CALIBRATION_SHA256:
        raise _error("policy-binding v7 seal mismatch")
    if binding.get("frozen_at_utc") != policy["frozen_at_utc"]:
        raise _error("policy-binding freeze timestamp mismatch")

    views = read_jsonl(root / "inputs/annotation-view.jsonl")
    labels_a = read_jsonl(root / "labels/raw/annotator-a.jsonl")
    labels_b = read_jsonl(root / "labels/raw/annotator-b.jsonl")
    validate_exact_coverage(views, labels_a, labels_b, policy=policy)
    validate_reviewer_independence(labels_a, labels_b, policy=policy)
    queue = read_jsonl(root / "disagreement-queue.jsonl")
    validate_blind_queue(queue)
    c_path = root / "labels/raw/reviewer-c.jsonl"
    labels_c = read_jsonl(c_path) if c_path.exists() else []
    validate_blind_c(
        labels_c,
        queue,
        views,
        policy=policy,
        labels_a=labels_a,
        labels_b=labels_b,
    )
    resolution = read_jsonl(root / "resolution.jsonl")
    _reject_forbidden(resolution)
    if len(resolution) != EXPECTED_ITEM_COUNT:
        raise _error("review resolution must cover exactly 40 items")
    routing = _read_json(root / "routing.json")
    _reject_forbidden(routing)
    if routing.get("N_input") != EXPECTED_ITEM_COUNT:
        raise _error("routing N_input must be 40")
    status_counts = routing.get("status_counts")
    if (
        not isinstance(status_counts, Mapping)
        or sum(int(status_counts.get(status, 0)) for status in RESOLUTION_STATUSES)
        != EXPECTED_ITEM_COUNT
    ):
        raise _error("routing status counts do not conserve N_input")
    metrics = _read_json(root / "pre-adjudication-metrics.json")
    _reject_forbidden(metrics)
    if metrics.get("item_count") != EXPECTED_ITEM_COUNT:
        raise _error("pre-adjudication metrics must cover exactly 40 items")
    computed = assess_evidence_qc(
        views,
        labels_a,
        labels_b,
        policy=policy,
        labels_c=labels_c if c_path.exists() else None,
    )
    if (
        metrics != computed["metrics"]
        or queue != computed["disagreement_queue"]
        or resolution != computed["resolution"]
        or routing != computed["routing"]
    ):
        raise _error("review outputs are not reproducible from the sealed raw inputs")
    sensitivity_plan = _read_json(root / "sensitivity-plan.json")
    _reject_forbidden(sensitivity_plan)
    if sensitivity_plan != computed["sensitivity_plan"]:
        raise _error("sensitivity plan is not the frozen plan")
    if manifest.get("qc_disposition") != computed["qc_disposition"]:
        raise _error("manifest disposition is not reproducible from raw inputs")
    return {
        "manifest_sha256": expected_manifest_sha,
        "policy_sha256": policy_sha256,
        "qc_disposition": manifest.get("qc_disposition"),
        "resolution_count": len(resolution),
        "schema_version": "evidence-qc-manifest/v1",
    }


def assess_and_write_review_pack(
    output_root: str | Path,
    *,
    policy: Mapping[str, object],
    policy_sha256: str,
    views: Sequence[Mapping[str, object]],
    labels_a: Sequence[Mapping[str, object]],
    labels_b: Sequence[Mapping[str, object]],
    labels_c: Sequence[Mapping[str, object]] | None = None,
    source_manifest: Mapping[str, object] | None = None,
    source_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate inputs, compute offline results, and create one new pack."""

    if source_root is not None:
        validate_source_pilot(source_root, policy=policy)
    _reject_forbidden(list(views))
    validate_exact_coverage(views, labels_a, labels_b, policy=policy)
    validate_reviewer_independence(labels_a, labels_b, policy=policy)
    if labels_c is not None:
        # The queue is generated only after A/B metrics are recomputed, so C
        # can never influence the frozen tier policy.
        preliminary = assess_evidence_qc(views, labels_a, labels_b, policy=policy)
        validate_blind_queue(preliminary["disagreement_queue"])
        validate_blind_c(
            labels_c,
            preliminary["disagreement_queue"],
            views,
            policy=policy,
            labels_a=labels_a,
            labels_b=labels_b,
        )
    assessment = assess_evidence_qc(
        views,
        labels_a,
        labels_b,
        policy=policy,
        labels_c=labels_c,
    )
    return write_review_pack(
        output_root,
        policy=policy,
        policy_sha256=policy_sha256,
        views=views,
        labels_a=labels_a,
        labels_b=labels_b,
        assessment=assessment,
        labels_c=labels_c,
        source_manifest=source_manifest,
        source_root=source_root,
    )


__all__ = [
    "B_FACING_BINDING_SCHEMA",
    "B_FACING_PACK_SCHEMA",
    "DEFAULT_ENGINE_FILES",
    "FROZEN_ENGINE_SHA256",
    "FROZEN_POLICY_SHA256",
    "FROZEN_POLICY_TIMESTAMP",
    "FORBIDDEN_FIELDS",
    "POLICY_BINDING_COMMENT_ID",
    "POLICY_HANDOFF_COMMENT_ID",
    "QCValidationError",
    "V7_CALIBRATION_SHA256",
    "V8_CALIBRATION_SHA256",
    "assert_allowed_source_path",
    "assert_overlay_outside_source",
    "assess_and_write_review_pack",
    "canonical_json_bytes",
    "compute_engine_sha256",
    "load_policy",
    "read_jsonl",
    "sha256_file",
    "validate_blind_c",
    "validate_blind_queue",
    "validate_b_facing_pack",
    "validate_exact_coverage",
    "validate_policy",
    "validate_reviewer_independence",
    "validate_review_pack",
    "validate_source_manifest",
    "validate_source_pilot",
    "write_b_facing_pack",
    "write_review_pack",
]
