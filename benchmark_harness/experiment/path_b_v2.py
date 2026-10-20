"""Offline Path B v2 review-contract and evidence-package validation.

This module is an additive boundary for the Path B v2 review pack.  It only
consumes small, JSON/JSONL review artifacts and never opens a dataset cache,
raw A material, a score/result ledger, or an evaluator output.  The contract
is intentionally strict: an ambiguous boolean, an unverifiable attestation,
or an incomplete price chain fails closed.

The validator is kept separate from :mod:`benchmark_harness.experiment.v3` so
the legacy and frozen experiment contracts remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

PATH_B_V2_SCHEMA_VERSION = "path-b-v2/v1"
PATH_B_V2_REVIEW_CONTRACT_SCHEMA_VERSION = "path-b-v2-review-contract/v1"
PATH_B_V2_MANIFEST_SCHEMA_VERSION = "path-b-v2-evidence-package-manifest/v1"

PATH_B_V2_REVIEW_CONTRACT_FILENAME = "path_b_v2_review_contract.json"
PATH_B_V2_ELIGIBILITY_FILENAME = "eligibility.jsonl"
PATH_B_V2_ATTESTATIONS_FILENAME = "attestations.jsonl"
PATH_B_V2_PACKAGES_FILENAME = "evidence-packages.jsonl"
PATH_B_V2_MANIFEST_FILENAME = "evidence-package-manifest.json"

# The four human-review methods are a closed enum.  In particular, a URL, a
# modeled date, or an automated-only decision is not a human attestation.
PATH_B_V2_REVIEW_METHODS = (
    "human_first_party_live_and_snapshot",
    "human_first_party_archive_snapshot",
    "human_official_api_snapshot",
    "automated_precheck_plus_human_review",
)
PATH_B_V2_REVIEW_METHOD_SET = frozenset(PATH_B_V2_REVIEW_METHODS)
PATH_B_V2_DECISIONS = frozenset({"eligible", "ineligible"})
PATH_B_V2_SCOPE_CATEGORIES = ("A1", "A2-T", "B-EARNINGS")
PATH_B_V2_SCOPE_CATEGORY_SET = frozenset(PATH_B_V2_SCOPE_CATEGORIES)
PATH_B_V2_EXCLUDED_CATEGORIES = ("A2-F", "A2-H", "B-MACRO", "C", "D", "E")
PATH_B_V2_EXCLUDED_CATEGORY_SET = frozenset(PATH_B_V2_EXCLUDED_CATEGORIES)

PATH_B_V2_RETURN_FORMULA_ID = "excess_return_v1"
PATH_B_V2_RETURN_FORMULA = (
    "(post_price / pre_price - 1) - (post_benchmark_price / pre_benchmark_price - 1)"
)

PATH_B_V2_DATASET_REPO_ID = "sselaine27/benchmark-research"
PATH_B_V2_DATASET_REVISION = "951edec3021353bf1f0db23a49cf8596a7148490"
PATH_B_V2_DATASET_TREE_SHA = "410c6c450abd5df4d73d6827004dd12a1cbf0c1b"

PATH_B_V2_REQUIRED_ATTESTATION_FIELDS = (
    "task_id",
    "decision",
    "reviewer_id",
    "review_method",
    "reviewed_at",
    "evidence_package_sha256",
    "notes",
    "attestation_sha256",
)
PATH_B_V2_EVIDENCE_COMPONENTS = (
    "task_identity",
    "temporal_provenance",
    "event_evidence",
    "outcome_derivation",
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMMUTABLE_SHA = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$|^[0-9a-f]{64}$")
_FORBIDDEN_KEY_TOKENS = frozenset(
    {
        "score",
        "scores",
        "score_value",
        "result",
        "results",
        "result_value",
        "prediction",
        "predictions",
        "prediction_value",
        "model_output",
        "system_output",
        "utility",
        "utilities",
        "metric_values",
        "parsed_output",
    }
)
_IDENTITY_FIELDS = (
    "task_id",
    "category",
    "variant",
    "task_family",
    "paper_band",
    "time_band",
    "stratum",
    "prompt_hash",
)


class PathBV2ValidationError(ValueError):
    """Raised when a Path B v2 contract or review artifact is invalid."""


class PathBV2ScoreBlindError(PathBV2ValidationError):
    """Raised when a score/result/prediction field crosses the review boundary."""


def canonical_json(value: Any) -> str:
    """Return the one canonical JSON representation used by all v2 hashes."""

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PathBV2ValidationError("value is not canonically JSON serializable") from exc


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value using the v2 canonical representation."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _normal_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _is_forbidden_key(value: Any) -> bool:
    normalized = _normal_key(value)
    return normalized in _FORBIDDEN_KEY_TOKENS or any(
        token in normalized for token in ("score", "result", "prediction")
    )


def assert_path_b_v2_score_blind(value: Any, *, path: str = "$") -> None:
    """Reject score/result/prediction keys without inspecting their values."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_forbidden_key(key):
                raise PathBV2ScoreBlindError(
                    f"score/result/prediction field is forbidden at {path}.{key}"
                )
            assert_path_b_v2_score_blind(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            assert_path_b_v2_score_blind(child, path=f"{path}[{index}]")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PathBV2ValidationError(f"{name} must be an object")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PathBV2ValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, name: str) -> str:
    parsed = _require_nonempty_string(value, name)
    if not _SHA256.fullmatch(parsed):
        raise PathBV2ValidationError(f"{name} must be a lowercase SHA-256")
    return parsed


def _require_immutable_sha(value: Any, name: str) -> str:
    parsed = _require_nonempty_string(value, name)
    if not _IMMUTABLE_SHA.fullmatch(parsed):
        raise PathBV2ValidationError(f"{name} must be a lowercase immutable SHA")
    return parsed


def _require_exact_keys(
    value: Mapping[str, Any],
    required: Iterable[str],
    name: str,
    *,
    optional: Iterable[str] = (),
) -> None:
    required_set = set(required)
    optional_set = set(optional)
    observed = set(value)
    missing = sorted(required_set - observed)
    extra = sorted(observed - required_set - optional_set)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extra:
            details.append(f"unknown {', '.join(extra)}")
        raise PathBV2ValidationError(f"{name} has invalid fields ({'; '.join(details)})")


def _parse_timestamp(value: Any, name: str) -> datetime:
    parsed = _require_nonempty_string(value, name)
    try:
        timestamp = datetime.fromisoformat(parsed.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PathBV2ValidationError(f"{name} must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise PathBV2ValidationError(f"{name} must include an explicit timezone")
    return timestamp


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PathBV2ValidationError(f"{name} must be a finite positive number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise PathBV2ValidationError(f"{name} must be a finite positive number")
    return parsed


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PathBV2ValidationError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PathBV2ValidationError(f"{name} must be a finite number")
    return parsed


def _copy_without(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    copied = dict(value)
    copied.pop(field, None)
    return copied


def _canonical_category(value: Any, variant: Any = None) -> str:
    raw = _normal_key(value).replace("_", "-")
    variant_key = _normal_key(variant).replace("_", "-") if variant is not None else ""
    if raw in {"a1", "a1-t1", "a1-t"}:
        return "A1"
    if raw in {"a2", "a2-t", "a2-t1", "a2t"}:
        return "A2-T" if not variant_key or variant_key in {"t", "t1", "temporal"} else raw.upper()
    if raw in {"b", "b-earnings", "b-earnings-t1", "b-earnings-t2", "earnings"}:
        return (
            "B-EARNINGS"
            if raw == "earnings"
            or not variant_key
            or variant_key
            in {
                "earnings",
                "b-earnings",
                "t1",
                "t2",
            }
            else raw.upper()
        )
    if raw in {"b-macro", "macro"}:
        return "B-MACRO"
    if raw in {"c", "c-t1", "c-t2"}:
        return "C"
    return raw.upper()


def _category_from_identity(identity: Mapping[str, Any], name: str) -> str:
    category = identity.get("category")
    if category is None:
        category = identity.get("task_category")
    return _canonical_category(category, identity.get("variant"))


def _task_id(value: Any, name: str = "task_id") -> str:
    return _require_nonempty_string(value, name)


def _source_fields(row: Mapping[str, Any]) -> tuple[Any, Any]:
    containers: list[Mapping[str, Any]] = [row]
    for key in ("source", "metadata", "evidence"):
        nested = row.get(key)
        if isinstance(nested, Mapping):
            containers.append(nested)
    url: Any = None
    source_hash: Any = None
    for container in containers:
        if url is None:
            url = container.get("source_url", container.get("url"))
        if source_hash is None:
            source_hash = container.get(
                "source_sha256",
                container.get("source_hash", container.get("sha256", container.get("hash"))),
            )
    return url, source_hash


def _matching_present(row: Mapping[str, Any]) -> bool:
    if isinstance(row.get("stratum"), str) and row["stratum"].strip():
        return True
    matching = row.get("matching", row.get("matching_fields"))
    if isinstance(matching, Mapping):
        return any(
            isinstance(matching.get(key), str) and matching[key].strip()
            for key in ("stratum", "matching_key", "key")
        )
    return isinstance(row.get("matching_key"), str) and bool(row["matching_key"].strip())


def _eligibility_object(row: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    if "eligibility" not in row:
        raise PathBV2ValidationError(f"eligibility row {index} is missing eligibility object")
    value = row["eligibility"]
    if isinstance(value, bool):
        raise PathBV2ValidationError(
            f"eligibility row {index} uses unsupported boolean shorthand; use eligibility.eligible"
        )
    if not isinstance(value, Mapping):
        raise PathBV2ValidationError(
            f"eligibility row {index}.eligibility must be an object with a JSON boolean eligible"
        )
    eligibility = value
    eligible = eligibility.get("eligible")
    if not isinstance(eligible, bool):
        raise PathBV2ValidationError(
            f"eligibility row {index}.eligibility.eligible must be a JSON boolean"
        )
    reasons = eligibility.get("reasons")
    if reasons is not None:
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) or not reason.strip() for reason in reasons
        ):
            raise PathBV2ValidationError(
                f"eligibility row {index}.eligibility.reasons must be a list of strings"
            )
    return eligibility


def _contract_hash_without_self(contract: Mapping[str, Any]) -> str:
    return sha256_json(contract)


def _default_design_lock() -> dict[str, Any]:
    return {
        "scope_categories": list(PATH_B_V2_SCOPE_CATEGORIES),
        "excluded_categories": list(PATH_B_V2_EXCLUDED_CATEGORIES),
        "c_excluded": True,
        "current_data_status": "CURRENT_DATA_INELIGIBLE",
        "b_status": "EXPLORATORY_ONLY",
        "task_identity_policy": "existing_task_ids_unchanged",
    }


def build_default_review_contract() -> dict[str, Any]:
    """Build the frozen, score-blind default contract used by the config."""

    dataset_lock = {
        "repo_id": PATH_B_V2_DATASET_REPO_ID,
        "revision": PATH_B_V2_DATASET_REVISION,
        "tree_sha": PATH_B_V2_DATASET_TREE_SHA,
    }
    design_lock = _default_design_lock()
    return {
        "schema_version": PATH_B_V2_REVIEW_CONTRACT_SCHEMA_VERSION,
        "dataset_lock": dataset_lock,
        "design_lock": design_lock,
        "dataset_lock_sha256": sha256_json(dataset_lock),
        "design_lock_sha256": sha256_json(design_lock),
        "review_methods": list(PATH_B_V2_REVIEW_METHODS),
        "required_attestation_fields": list(PATH_B_V2_REQUIRED_ATTESTATION_FIELDS),
        "collector_reviewer_must_differ": True,
        "boolean_eligibility_contract": "json_boolean_only_nested_eligible",
        "evidence_package_components": list(PATH_B_V2_EVIDENCE_COMPONENTS),
        "return_formula": {
            "id": PATH_B_V2_RETURN_FORMULA_ID,
            "expression": PATH_B_V2_RETURN_FORMULA,
        },
    }


def validate_review_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the immutable Path B v2 review contract."""

    contract = _require_mapping(contract, "review contract")
    _require_exact_keys(
        contract,
        (
            "schema_version",
            "dataset_lock",
            "design_lock",
            "dataset_lock_sha256",
            "design_lock_sha256",
            "review_methods",
            "required_attestation_fields",
            "collector_reviewer_must_differ",
            "boolean_eligibility_contract",
            "evidence_package_components",
            "return_formula",
        ),
        "review contract",
    )
    if contract["schema_version"] != PATH_B_V2_REVIEW_CONTRACT_SCHEMA_VERSION:
        raise PathBV2ValidationError("review contract schema_version is unsupported")

    dataset_lock = _require_mapping(contract["dataset_lock"], "review contract.dataset_lock")
    _require_exact_keys(dataset_lock, ("repo_id", "revision", "tree_sha"), "dataset lock")
    _require_nonempty_string(dataset_lock["repo_id"], "dataset_lock.repo_id")
    _require_immutable_sha(dataset_lock["revision"], "dataset_lock.revision")
    _require_immutable_sha(dataset_lock["tree_sha"], "dataset_lock.tree_sha")
    if contract["dataset_lock_sha256"] != sha256_json(dict(dataset_lock)):
        raise PathBV2ValidationError("dataset lock hash mismatch")
    _require_sha256(contract["dataset_lock_sha256"], "dataset_lock_sha256")

    design_lock = _require_mapping(contract["design_lock"], "review contract.design_lock")
    _require_exact_keys(
        design_lock,
        (
            "scope_categories",
            "excluded_categories",
            "c_excluded",
            "current_data_status",
            "b_status",
            "task_identity_policy",
        ),
        "design lock",
    )
    if design_lock["scope_categories"] != list(PATH_B_V2_SCOPE_CATEGORIES):
        raise PathBV2ValidationError("design lock Path B scope is not A1/A2-T/B-EARNINGS")
    if design_lock["excluded_categories"] != list(PATH_B_V2_EXCLUDED_CATEGORIES):
        raise PathBV2ValidationError("design lock excluded category set is not frozen")
    if design_lock["c_excluded"] is not True:
        raise PathBV2ValidationError("design lock must exclude C")
    if design_lock["current_data_status"] != "CURRENT_DATA_INELIGIBLE":
        raise PathBV2ValidationError("current data status must remain CURRENT_DATA_INELIGIBLE")
    if design_lock["b_status"] != "EXPLORATORY_ONLY":
        raise PathBV2ValidationError("B status must remain EXPLORATORY_ONLY")
    if design_lock["task_identity_policy"] != "existing_task_ids_unchanged":
        raise PathBV2ValidationError("task identity policy is not immutable")
    if contract["design_lock_sha256"] != sha256_json(dict(design_lock)):
        raise PathBV2ValidationError("design lock hash mismatch")
    _require_sha256(contract["design_lock_sha256"], "design_lock_sha256")

    if contract["review_methods"] != list(PATH_B_V2_REVIEW_METHODS):
        raise PathBV2ValidationError("review_methods must contain exactly the four human methods")
    if contract["required_attestation_fields"] != list(PATH_B_V2_REQUIRED_ATTESTATION_FIELDS):
        raise PathBV2ValidationError("required attestation field order is not frozen")
    if contract["collector_reviewer_must_differ"] is not True:
        raise PathBV2ValidationError("collector_reviewer_must_differ must be true")
    if contract["boolean_eligibility_contract"] != "json_boolean_only_nested_eligible":
        raise PathBV2ValidationError("eligibility boolean contract is unsupported")
    if contract["evidence_package_components"] != list(PATH_B_V2_EVIDENCE_COMPONENTS):
        raise PathBV2ValidationError("evidence package component binding is incomplete")
    formula = _require_mapping(contract["return_formula"], "review contract.return_formula")
    _require_exact_keys(formula, ("id", "expression"), "return formula")
    if (
        formula["id"] != PATH_B_V2_RETURN_FORMULA_ID
        or formula["expression"] != PATH_B_V2_RETURN_FORMULA
    ):
        raise PathBV2ValidationError("return formula is not the frozen excess-return formula")
    return deepcopy(dict(contract))


def compute_attestation_sha256(attestation: Mapping[str, Any]) -> str:
    """Compute the self-hash of an attestation without reading its claimed hash."""

    return sha256_json(_copy_without(attestation, "attestation_sha256"))


def validate_attestation(
    attestation: Mapping[str, Any],
    *,
    expected_task_id: str | None = None,
    expected_evidence_package_sha256: str | None = None,
    collector_id: str | None = None,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one fail-closed, human attestation bound to one package."""

    attestation = _require_mapping(attestation, "attestation")
    assert_path_b_v2_score_blind(attestation)
    _require_exact_keys(attestation, PATH_B_V2_REQUIRED_ATTESTATION_FIELDS, "attestation")
    task_id = _task_id(attestation["task_id"], "attestation.task_id")
    decision = _require_nonempty_string(attestation["decision"], "attestation.decision")
    if decision not in PATH_B_V2_DECISIONS:
        raise PathBV2ValidationError("attestation.decision must be eligible or ineligible")
    reviewer_id = _require_nonempty_string(attestation["reviewer_id"], "attestation.reviewer_id")
    method = _require_nonempty_string(attestation["review_method"], "attestation.review_method")
    methods = PATH_B_V2_REVIEW_METHOD_SET
    if contract is not None:
        validated_contract = validate_review_contract(contract)
        methods = frozenset(validated_contract["review_methods"])
    if method not in methods:
        raise PathBV2ValidationError("attestation.review_method is not an allowed human method")
    _parse_timestamp(attestation["reviewed_at"], "attestation.reviewed_at")
    package_sha = _require_sha256(
        attestation["evidence_package_sha256"], "attestation.evidence_package_sha256"
    )
    _require_nonempty_string(attestation["notes"], "attestation.notes")
    claimed_hash = _require_sha256(
        attestation["attestation_sha256"], "attestation.attestation_sha256"
    )
    if claimed_hash != compute_attestation_sha256(attestation):
        raise PathBV2ValidationError("attestation hash mismatch")
    if expected_task_id is not None and task_id != _task_id(expected_task_id, "expected_task_id"):
        raise PathBV2ValidationError("attestation task_id does not match eligibility row")
    if expected_evidence_package_sha256 is not None:
        expected_hash = _require_sha256(
            expected_evidence_package_sha256, "expected_evidence_package_sha256"
        )
        if package_sha != expected_hash:
            raise PathBV2ValidationError("attestation is bound to a different evidence package")
    collector_checked = collector_id is not None
    if collector_id is not None:
        parsed_collector = _require_nonempty_string(collector_id, "collector_id")
        if reviewer_id == parsed_collector:
            raise PathBV2ValidationError("reviewer_id must differ from collector_id")
    return {
        "task_id": task_id,
        "decision": decision,
        "reviewer_id": reviewer_id,
        "review_method": method,
        "evidence_package_sha256": package_sha,
        "collector_identity_checked": collector_checked,
        "attestation_sha256": claimed_hash,
    }


def _component_task_id(component: Mapping[str, Any], name: str) -> str | None:
    value = component.get("task_id")
    if value is None:
        return None
    return _task_id(value, f"{name}.task_id")


def _component_source_hash(component: Mapping[str, Any], name: str) -> str:
    value = component.get(
        "source_sha256",
        component.get("source_hash", component.get("sha256", component.get("hash"))),
    )
    return _require_sha256(value, f"{name}.source_sha256")


def _component_source_url(component: Mapping[str, Any], name: str) -> str:
    url = _require_nonempty_string(
        component.get("source_url", component.get("url")), f"{name}.source_url"
    )
    if not url.startswith(("http://", "https://")):
        raise PathBV2ValidationError(f"{name}.source_url must be an http(s) URL")
    return url


def _price_chain(outcome: Mapping[str, Any]) -> Mapping[str, Any]:
    chain = outcome.get("price_chain")
    alias = outcome.get("price_evidence")
    if chain is not None and alias is not None and chain != alias:
        raise PathBV2ValidationError("outcome_derivation price_chain/price_evidence disagree")
    if chain is None:
        chain = alias
    return _require_mapping(chain, "outcome_derivation.price_chain")


def _formula(outcome: Mapping[str, Any]) -> Mapping[str, Any]:
    formula = outcome.get("return_formula")
    flat_id = outcome.get("formula_id")
    flat_expression = outcome.get("formula")
    if formula is not None and (flat_id is not None or flat_expression is not None):
        if (
            not isinstance(formula, Mapping)
            or formula.get("id") != flat_id
            or formula.get("expression") != flat_expression
        ):
            raise PathBV2ValidationError("outcome return formula aliases disagree")
    if formula is None and (flat_id is not None or flat_expression is not None):
        formula = {"id": flat_id, "expression": flat_expression}
    if formula is None:
        raise PathBV2ValidationError("outcome_derivation.return_formula is missing")
    formula = _require_mapping(formula, "outcome_derivation.return_formula")
    _require_exact_keys(formula, ("id", "expression"), "outcome return formula")
    if formula["id"] != PATH_B_V2_RETURN_FORMULA_ID:
        raise PathBV2ValidationError("outcome return formula id is unsupported")
    if formula["expression"] != PATH_B_V2_RETURN_FORMULA:
        raise PathBV2ValidationError("outcome return formula expression is not frozen")
    return formula


def _price_snapshot(snapshot: Any, role: str) -> tuple[float, datetime, str, str]:
    item = _require_mapping(snapshot, f"price_chain.{role}")
    url = _component_source_url(item, f"price_chain.{role}")
    source_hash = _component_source_hash(item, f"price_chain.{role}")
    observed_at = _parse_timestamp(item.get("observed_at"), f"price_chain.{role}.observed_at")
    price = _finite_positive(item.get("price"), f"price_chain.{role}.price")
    return price, observed_at, url, source_hash


def compute_return_derivation_sha256(outcome_derivation: Mapping[str, Any]) -> str:
    """Hash the formula, price chain, and reported return when supplied."""

    outcome_derivation = _require_mapping(outcome_derivation, "outcome_derivation")
    payload: dict[str, Any] = {
        "formula": dict(_formula(outcome_derivation)),
        "price_chain": dict(_price_chain(outcome_derivation)),
    }
    if "excess_return" in outcome_derivation:
        payload["excess_return"] = outcome_derivation["excess_return"]
    return sha256_json(payload)


def validate_b_evidence(
    event_evidence: Mapping[str, Any], outcome_derivation: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the separate event and price evidence chain for B-earnings."""

    event = _require_mapping(event_evidence, "event_evidence")
    outcome = _require_mapping(outcome_derivation, "outcome_derivation")
    event_id = _require_nonempty_string(event.get("event_id"), "event_evidence.event_id")
    event_url = _component_source_url(event, "event_evidence")
    event_hash = _component_source_hash(event, "event_evidence")
    published_at = _parse_timestamp(event.get("published_at"), "event_evidence.published_at")
    status = _require_nonempty_string(
        outcome.get("price_evidence_status"), "outcome_derivation.price_evidence_status"
    )
    if status != "available":
        raise PathBV2ValidationError(
            "B-earnings requires price_evidence_status=available; not_applicable is invalid"
        )

    chain = _price_chain(outcome)
    _require_exact_keys(chain, ("pre", "post", "benchmark_pre", "benchmark_post"), "price chain")
    snapshots = {role: _price_snapshot(chain[role], role) for role in chain}
    pre_price, pre_at, _pre_url, _pre_hash = snapshots["pre"]
    post_price, post_at, _post_url, _post_hash = snapshots["post"]
    benchmark_pre, benchmark_pre_at, _benchmark_pre_url, _benchmark_pre_hash = snapshots[
        "benchmark_pre"
    ]
    benchmark_post, benchmark_post_at, _benchmark_post_url, _benchmark_post_hash = snapshots[
        "benchmark_post"
    ]
    if pre_at > published_at or benchmark_pre_at > published_at:
        raise PathBV2ValidationError("pre-event price snapshot occurs after event publication")
    if post_at < published_at or benchmark_post_at < published_at:
        raise PathBV2ValidationError("post-event price snapshot occurs before event publication")

    _formula(outcome)
    instrument = _require_nonempty_string(
        outcome.get("instrument"), "outcome_derivation.instrument"
    )
    benchmark = _require_nonempty_string(outcome.get("benchmark"), "outcome_derivation.benchmark")
    pre_session = _require_nonempty_string(
        outcome.get("pre_event_session"), "outcome_derivation.pre_event_session"
    )
    post_session = _require_nonempty_string(
        outcome.get("post_event_session"), "outcome_derivation.post_event_session"
    )
    if pre_session == post_session:
        raise PathBV2ValidationError("pre_event_session and post_event_session must differ")
    calendar = outcome.get("market_calendar")
    calendar_alias = outcome.get("calendar")
    if calendar is not None and calendar_alias is not None and calendar != calendar_alias:
        raise PathBV2ValidationError("market_calendar/calendar aliases disagree")
    if calendar is None:
        calendar = calendar_alias
    timezone = _require_nonempty_string(outcome.get("timezone"), "outcome_derivation.timezone")
    _require_nonempty_string(calendar, "outcome_derivation.market_calendar")
    reported = _finite_number(outcome.get("excess_return"), "outcome_derivation.excess_return")
    recomputed = _finite_number(
        outcome.get("recomputed_excess_return"),
        "outcome_derivation.recomputed_excess_return",
    )
    expected = (post_price / pre_price - 1.0) - (benchmark_post / benchmark_pre - 1.0)
    if not math.isclose(reported, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise PathBV2ValidationError("reported excess return does not match the frozen formula")
    if not math.isclose(recomputed, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise PathBV2ValidationError("recomputed excess return does not match the frozen formula")
    optional_hash = outcome.get("return_derivation_sha256")
    optional_hash_alias = outcome.get("return_attestation_sha256")
    if (
        optional_hash is not None
        and optional_hash_alias is not None
        and optional_hash != optional_hash_alias
    ):
        raise PathBV2ValidationError("return derivation hash aliases disagree")
    if optional_hash is None:
        optional_hash = optional_hash_alias
    if optional_hash is not None:
        claimed_hash = _require_sha256(optional_hash, "return_derivation_sha256")
        if claimed_hash != compute_return_derivation_sha256(outcome):
            raise PathBV2ValidationError("return derivation hash mismatch")

    return {
        "event_id": event_id,
        "event_source_url": event_url,
        "event_source_sha256": event_hash,
        "published_at": published_at.isoformat(),
        "instrument": instrument,
        "benchmark": benchmark,
        "pre_event_session": pre_session,
        "post_event_session": post_session,
        "market_calendar": str(calendar),
        "timezone": timezone,
        "formula_id": PATH_B_V2_RETURN_FORMULA_ID,
        "excess_return_recomputed": expected,
    }


def validate_evidence_package(
    package: Mapping[str, Any], *, expected_task_id: str | None = None
) -> dict[str, Any]:
    """Validate package hash binding, temporal provenance, and B semantics."""

    package = _require_mapping(package, "evidence package")
    assert_path_b_v2_score_blind(package)
    _require_exact_keys(
        package,
        (*PATH_B_V2_EVIDENCE_COMPONENTS, "evidence_package_sha256"),
        "evidence package",
        optional=("task_id",),
    )
    package_task_id = (
        _task_id(package["task_identity"].get("task_id"), "task_identity.task_id")
        if isinstance(package["task_identity"], Mapping)
        else None
    )
    if package_task_id is None:
        raise PathBV2ValidationError("evidence package task_identity.task_id is missing")
    if (
        "task_id" in package
        and _task_id(package["task_id"], "evidence package.task_id") != package_task_id
    ):
        raise PathBV2ValidationError("evidence package.task_id does not match task identity")
    if expected_task_id is not None and package_task_id != _task_id(
        expected_task_id, "expected_task_id"
    ):
        raise PathBV2ValidationError("evidence package task_id does not match eligibility row")
    identity = _require_mapping(package["task_identity"], "task_identity")
    _require_nonempty_string(identity.get("category"), "task_identity.category")
    category = _category_from_identity(identity, "task_identity")
    if category not in PATH_B_V2_SCOPE_CATEGORY_SET:
        raise PathBV2ValidationError(
            f"evidence package category {category!r} is outside Path B scope"
        )
    _require_nonempty_string(identity.get("variant"), "task_identity.variant")

    temporal = _require_mapping(package["temporal_provenance"], "temporal_provenance")
    if temporal.get("official_temporal_eligible") is not True:
        raise PathBV2ValidationError("temporal_provenance.official_temporal_eligible must be true")
    available = temporal.get("outcome_available_at_source")
    if not isinstance(available, str) or not available.startswith("observed_"):
        raise PathBV2ValidationError(
            "temporal_provenance.outcome_available_at_source must start with observed_"
        )
    _require_nonempty_string(available, "temporal_provenance.outcome_available_at_source")
    _component_task_id(temporal, "temporal_provenance")
    event = _require_mapping(package["event_evidence"], "event_evidence")
    outcome = _require_mapping(package["outcome_derivation"], "outcome_derivation")
    for component_name, component in (
        ("event_evidence", event),
        ("outcome_derivation", outcome),
    ):
        component_task_id = _component_task_id(component, component_name)
        if component_task_id is not None and component_task_id != package_task_id:
            raise PathBV2ValidationError(f"{component_name}.task_id does not match task identity")
    if category == "B-EARNINGS":
        b_summary = validate_b_evidence(event, outcome)
    else:
        # A1/A2-T packages still bind event/outcome rows, but only B has a
        # return-specific price-chain contract.
        if not event or not outcome:
            raise PathBV2ValidationError("event and outcome evidence components cannot be empty")
        b_summary = None

    claimed_hash = _require_sha256(package["evidence_package_sha256"], "evidence_package_sha256")
    if claimed_hash != compute_evidence_package_sha256(package):
        raise PathBV2ValidationError("evidence package hash mismatch")
    return {
        "task_id": package_task_id,
        "category": category,
        "evidence_package_sha256": claimed_hash,
        "b_evidence": b_summary,
    }


def compute_evidence_package_sha256(package: Mapping[str, Any]) -> str:
    """Compute the package hash over all four bound evidence components."""

    return sha256_json(_copy_without(package, "evidence_package_sha256"))


def build_evidence_package_manifest(
    packages: Sequence[Mapping[str, Any]], *, review_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a deterministic manifest binding every package to the contract."""

    contract = validate_review_contract(review_contract)
    entries: list[dict[str, str]] = []
    for index, package in enumerate(packages):
        summary = validate_evidence_package(package)
        entries.append(
            {
                "task_id": summary["task_id"],
                "evidence_package_sha256": summary["evidence_package_sha256"],
            }
        )
        if not isinstance(package, Mapping):
            raise PathBV2ValidationError(f"package {index} must be an object")
    entries.sort(key=lambda entry: entry["task_id"])
    if len({entry["task_id"] for entry in entries}) != len(entries):
        raise PathBV2ValidationError("evidence package manifest contains duplicate task_id values")
    manifest = {
        "schema_version": PATH_B_V2_MANIFEST_SCHEMA_VERSION,
        "review_contract_sha256": sha256_json(contract),
        "package_count": len(entries),
        "packages": entries,
    }
    return {**manifest, "manifest_sha256": sha256_json(manifest)}


def validate_evidence_package_manifest(
    manifest: Mapping[str, Any],
    packages: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    expected_review_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate manifest self-hash and every package hash/identity binding."""

    manifest = _require_mapping(manifest, "evidence package manifest")
    assert_path_b_v2_score_blind(manifest)
    _require_exact_keys(
        manifest,
        (
            "schema_version",
            "review_contract_sha256",
            "package_count",
            "packages",
            "manifest_sha256",
        ),
        "evidence package manifest",
    )
    if manifest["schema_version"] != PATH_B_V2_MANIFEST_SCHEMA_VERSION:
        raise PathBV2ValidationError("evidence package manifest schema is unsupported")
    contract_hash = _require_sha256(
        manifest["review_contract_sha256"], "manifest.review_contract_sha256"
    )
    if expected_review_contract_sha256 is not None:
        expected_hash = _require_sha256(
            expected_review_contract_sha256, "expected_review_contract_sha256"
        )
        if contract_hash != expected_hash:
            raise PathBV2ValidationError("manifest is bound to a different review contract")
    package_entries = manifest["packages"]
    if not isinstance(package_entries, list):
        raise PathBV2ValidationError("manifest.packages must be a list")
    if not isinstance(manifest["package_count"], int) or isinstance(
        manifest["package_count"], bool
    ):
        raise PathBV2ValidationError("manifest.package_count must be an integer")
    if manifest["package_count"] != len(package_entries):
        raise PathBV2ValidationError("manifest package_count does not match package list")
    observed_entries: dict[str, str] = {}
    for index, entry in enumerate(package_entries):
        entry = _require_mapping(entry, f"manifest.packages[{index}]")
        _require_exact_keys(
            entry,
            ("task_id", "evidence_package_sha256"),
            f"manifest.packages[{index}]",
        )
        task_id = _task_id(entry["task_id"], f"manifest.packages[{index}].task_id")
        package_hash = _require_sha256(
            entry["evidence_package_sha256"],
            f"manifest.packages[{index}].evidence_package_sha256",
        )
        if task_id in observed_entries:
            raise PathBV2ValidationError("manifest contains duplicate task_id values")
        observed_entries[task_id] = package_hash
    if list(observed_entries) != sorted(observed_entries):
        raise PathBV2ValidationError("manifest packages must be sorted by task_id")
    if manifest["manifest_sha256"] != sha256_json(_copy_without(manifest, "manifest_sha256")):
        raise PathBV2ValidationError("evidence package manifest hash mismatch")

    if isinstance(packages, Mapping):
        package_map = dict(packages)
    else:
        package_map = {}
        for package in packages:
            package_summary = validate_evidence_package(package)
            task_id = package_summary["task_id"]
            if task_id in package_map:
                raise PathBV2ValidationError("duplicate evidence package task_id")
            package_map[task_id] = package
    if set(package_map) != set(observed_entries):
        raise PathBV2ValidationError("manifest task identities do not match evidence packages")
    for task_id, package in package_map.items():
        summary = validate_evidence_package(package, expected_task_id=task_id)
        if summary["evidence_package_sha256"] != observed_entries[task_id]:
            raise PathBV2ValidationError(f"manifest hash mismatch for package {task_id}")
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "review_contract_sha256": contract_hash,
        "package_count": len(observed_entries),
        "task_ids": list(observed_entries),
    }


def _records_by_task(
    values: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]], name: str
) -> dict[str, Mapping[str, Any]]:
    if isinstance(values, Mapping):
        result = {}
        for task_id, value in values.items():
            parsed_task_id = _task_id(task_id, f"{name} key")
            if parsed_task_id in result:
                raise PathBV2ValidationError(f"duplicate {name} task_id {parsed_task_id}")
            result[parsed_task_id] = _require_mapping(value, f"{name}[{parsed_task_id}]")
        return result
    result = {}
    for index, value in enumerate(values):
        item = _require_mapping(value, f"{name}[{index}]")
        raw_task_id = item.get("task_id")
        if raw_task_id is None and name == "evidence packages":
            identity = item.get("task_identity")
            if isinstance(identity, Mapping):
                raw_task_id = identity.get("task_id")
        task_id = _task_id(raw_task_id, f"{name}[{index}].task_id")
        if task_id in result:
            raise PathBV2ValidationError(f"duplicate {name} task_id {task_id}")
        result[task_id] = item
    return result


def validate_eligibility_rows(
    rows: Sequence[Mapping[str, Any]],
    attestations: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    packages: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any] | None = None,
    review_contract: Mapping[str, Any] | None = None,
    expected_task_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate row qualification and conserve eligible/excluded queue counts."""

    if not isinstance(rows, Sequence) or isinstance(rows, str | bytes | bytearray):
        raise PathBV2ValidationError("eligibility rows must be a JSON array")
    contract = validate_review_contract(review_contract) if review_contract is not None else None
    package_map = _records_by_task(packages, "evidence packages")
    package_summaries = {
        task_id: validate_evidence_package(package, expected_task_id=task_id)
        for task_id, package in package_map.items()
    }
    attestation_map = _records_by_task(attestations, "attestations")
    row_map: dict[str, Mapping[str, Any]] = {}
    row_indexes: dict[str, int] = {}
    eligible_ids: list[str] = []
    excluded_ids: list[str] = []
    exclusions: dict[str, list[str]] = {}
    for index, row_value in enumerate(rows):
        row = _require_mapping(row_value, f"eligibility row {index}")
        assert_path_b_v2_score_blind(row, path=f"$[{index}]")
        task_id = _task_id(row.get("task_id"), f"eligibility row {index}.task_id")
        if task_id in row_map:
            raise PathBV2ValidationError(f"duplicate eligibility task_id {task_id}")
        row_map[task_id] = row
        row_indexes[task_id] = index
        eligibility = _eligibility_object(row, index)
        eligible = eligibility["eligible"]
        category = _canonical_category(row.get("category"), row.get("variant"))
        if eligible and category not in PATH_B_V2_SCOPE_CATEGORY_SET:
            raise PathBV2ValidationError(
                f"eligible task {task_id} category {category!r} is outside Path B scope"
            )
        if category == "C" and eligible:
            raise PathBV2ValidationError(f"C task {task_id} cannot enter Path B")
        if not eligible:
            excluded_ids.append(task_id)
            reasons = eligibility.get("reasons")
            exclusions[task_id] = list(reasons) if isinstance(reasons, list) else ["ineligible"]
            continue

        if row.get("official_temporal_eligible") is not True:
            raise PathBV2ValidationError(
                f"eligible task {task_id} lacks official temporal eligibility"
            )
        if row.get("review_status") != "reviewed":
            raise PathBV2ValidationError(f"eligible task {task_id} is not reviewed")
        available = row.get("outcome_available_at_source")
        if not isinstance(available, str) or not available.startswith("observed_"):
            raise PathBV2ValidationError(f"eligible task {task_id} lacks observed source outcome")
        if row.get("status") != "ready":
            raise PathBV2ValidationError(f"eligible task {task_id} is not ready")
        source_url, source_hash = _source_fields(row)
        _component_source_url({"source_url": source_url}, f"eligibility row {index}")
        _require_sha256(source_hash, f"eligibility row {index}.source_sha256")
        evidence_verified = row.get("evidence_verified")
        if evidence_verified is None and isinstance(row.get("evidence"), Mapping):
            evidence_verified = row["evidence"].get("verified")
        if evidence_verified is not True:
            raise PathBV2ValidationError(f"eligible task {task_id} evidence is not verified")
        if not _matching_present(row):
            raise PathBV2ValidationError(f"eligible task {task_id} has no matching stratum")
        collector_id = _require_nonempty_string(
            row.get("collector_id"), f"eligibility row {index}.collector_id"
        )
        package_hash = row.get("evidence_package_sha256", row.get("evidence_package_hash"))
        package_hash = _require_sha256(
            package_hash, f"eligibility row {index}.evidence_package_sha256"
        )
        if task_id not in package_map:
            raise PathBV2ValidationError(f"eligible task {task_id} has no evidence package")
        package_summary = package_summaries[task_id]
        if package_summary["evidence_package_sha256"] != package_hash:
            raise PathBV2ValidationError(f"eligible task {task_id} package hash is not bound")
        if task_id not in attestation_map:
            raise PathBV2ValidationError(f"eligible task {task_id} has no human attestation")
        attestation_summary = validate_attestation(
            attestation_map[task_id],
            expected_task_id=task_id,
            expected_evidence_package_sha256=package_hash,
            collector_id=collector_id,
            contract=contract,
        )
        if attestation_summary["decision"] != "eligible":
            raise PathBV2ValidationError(f"eligible task {task_id} has a non-eligible attestation")
        eligible_ids.append(task_id)

    if set(attestation_map) - set(row_map):
        raise PathBV2ValidationError("attestation contains a task absent from eligibility rows")
    if set(package_map) - set(row_map):
        raise PathBV2ValidationError(
            "evidence package contains a task absent from eligibility rows"
        )
    for task_id, attestation in attestation_map.items():
        row = row_map[task_id]
        eligibility = _eligibility_object(row, row_indexes[task_id])
        expected_decision = "eligible" if eligibility["eligible"] else "ineligible"
        expected_package_hash = package_summaries.get(task_id, {}).get("evidence_package_sha256")
        summary = validate_attestation(
            attestation,
            expected_task_id=task_id,
            expected_evidence_package_sha256=expected_package_hash,
            collector_id=row.get("collector_id"),
            contract=contract,
        )
        if summary["decision"] != expected_decision:
            raise PathBV2ValidationError(
                f"attestation decision does not match eligibility for task {task_id}"
            )
    if expected_task_ids is not None:
        expected = {_task_id(task_id, "expected_task_ids item") for task_id in expected_task_ids}
        if expected != set(row_map):
            raise PathBV2ValidationError(
                "eligibility task identities differ from expected task lock"
            )
    if manifest is not None:
        if contract is None:
            raise PathBV2ValidationError("manifest validation requires the review contract")
        validate_evidence_package_manifest(
            manifest,
            package_map,
            expected_review_contract_sha256=sha256_json(contract),
        )
    if len(eligible_ids) + len(excluded_ids) != len(row_map):
        raise PathBV2ValidationError("eligibility queue/count conservation failed")
    return {
        "status": "valid",
        "total_count": len(row_map),
        "eligible_count": len(eligible_ids),
        "excluded_count": len(excluded_ids),
        "eligible_task_ids": sorted(eligible_ids),
        "excluded_task_ids": sorted(excluded_ids),
        "exclusions": {task_id: exclusions[task_id] for task_id in sorted(exclusions)},
        "scope_categories": list(PATH_B_V2_SCOPE_CATEGORIES),
        "c_excluded": True,
    }


def _identity_view(row: Mapping[str, Any]) -> dict[str, Any]:
    identity = row.get("task_identity")
    if isinstance(identity, Mapping):
        candidate = identity
    else:
        candidate = row
    task_id = _task_id(candidate.get("task_id"), "task identity.task_id")
    return {
        field: candidate[field]
        for field in _IDENTITY_FIELDS
        if field in candidate and field != "task_id"
    } | {"task_id": task_id}


def build_task_identity_lock(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a score-blind identity lock for an old/new task set comparison."""

    identities = []
    for row in rows:
        row = _require_mapping(row, "task identity row")
        assert_path_b_v2_score_blind(row)
        identities.append(_identity_view(row))
    identities.sort(key=lambda item: item["task_id"])
    task_ids = [item["task_id"] for item in identities]
    if len(task_ids) != len(set(task_ids)):
        raise PathBV2ValidationError("task identity lock contains duplicate task_id values")
    payload = {"schema_version": PATH_B_V2_SCHEMA_VERSION, "identities": identities}
    return {**payload, "identity_sha256": sha256_json(payload)}


def validate_task_identity_unchanged(
    current_rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Fail closed if the current pin changes task IDs or immutable identity fields."""

    current = build_task_identity_lock(current_rows)
    baseline = build_task_identity_lock(baseline_rows)
    if current["identity_sha256"] != baseline["identity_sha256"]:
        raise PathBV2ValidationError(
            "task identity drift detected between baseline and current pin"
        )
    return {
        "status": "unchanged",
        "task_count": len(current["identities"]),
        "identity_sha256": current["identity_sha256"],
    }


def _read_json(path: Path, name: str) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PathBV2ValidationError(f"unable to read {name}: {path}") from exc
    return value


def _read_jsonl(path: Path, name: str) -> list[Mapping[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PathBV2ValidationError(f"unable to read {name}: {path}") from exc
    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PathBV2ValidationError(f"invalid {name} JSON at line {index}") from exc
        rows.append(_require_mapping(value, f"{name} line {index}"))
    return rows


def _resolve_one(root: Path, names: Sequence[str], label: str) -> Path:
    candidates = [root / name for name in names if (root / name).is_file()]
    if len(candidates) != 1:
        if not candidates:
            raise PathBV2ValidationError(f"{label} is missing from {root}")
        raise PathBV2ValidationError(f"{label} has multiple aliases in {root}")
    return candidates[0]


def _assert_pack_boundary(root: Path) -> None:
    forbidden_tokens = (
        "raw_a",
        "annotator_a",
        "labeler_a",
        "labels_raw",
        "400",
        "gold",
        "probe",
    )
    try:
        paths = tuple(path for path in root.rglob("*") if path.is_file())
    except OSError as exc:
        raise PathBV2ValidationError(f"unable to inspect review-pack boundary: {root}") from exc
    for path in paths:
        normalized = _normal_key(str(path.relative_to(root)))
        if any(token in normalized for token in forbidden_tokens):
            raise PathBV2ValidationError(
                f"B-facing pack contains forbidden protected artifact name: {path.name}"
            )


def validate_path_b_v2_bundle(
    root: str | Path, *, contract_path: str | Path | None = None
) -> dict[str, Any]:
    """Validate a clean-runtime Path B v2 review bundle without network access."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise PathBV2ValidationError(f"Path B v2 review-pack root is not a directory: {root_path}")
    _assert_pack_boundary(root_path)
    if contract_path is None:
        contract_candidate = _resolve_one(
            root_path,
            (PATH_B_V2_REVIEW_CONTRACT_FILENAME, "path-b-v2-review-contract.json"),
            "review contract",
        )
    else:
        contract_candidate = Path(contract_path).expanduser().resolve()
    contract = validate_review_contract(_read_json(contract_candidate, "review contract"))
    eligibility_path = _resolve_one(
        root_path,
        (PATH_B_V2_ELIGIBILITY_FILENAME, "eligibility-ledger.jsonl"),
        "eligibility ledger",
    )
    attestations_path = _resolve_one(
        root_path,
        (PATH_B_V2_ATTESTATIONS_FILENAME, "human-attestations.jsonl"),
        "attestations",
    )
    packages_path = _resolve_one(
        root_path,
        (PATH_B_V2_PACKAGES_FILENAME, "evidence_packages.jsonl"),
        "evidence packages",
    )
    manifest_path = _resolve_one(
        root_path,
        (PATH_B_V2_MANIFEST_FILENAME, "evidence_package_manifest.json"),
        "evidence manifest",
    )
    eligibility = _read_jsonl(eligibility_path, "eligibility ledger")
    attestations = _read_jsonl(attestations_path, "attestations")
    packages = _read_jsonl(packages_path, "evidence packages")
    manifest = _read_json(manifest_path, "evidence package manifest")
    result = validate_eligibility_rows(
        eligibility,
        attestations,
        packages,
        manifest=_require_mapping(manifest, "evidence package manifest"),
        review_contract=contract,
    )
    manifest_summary = validate_evidence_package_manifest(
        _require_mapping(manifest, "evidence package manifest"),
        _records_by_task(packages, "evidence packages"),
        expected_review_contract_sha256=sha256_json(contract),
    )
    return {
        **result,
        "schema_version": PATH_B_V2_SCHEMA_VERSION,
        "review_contract_sha256": sha256_json(contract),
        "manifest_sha256": manifest_summary["manifest_sha256"],
        "package_count": manifest_summary["package_count"],
        "status": "valid",
    }


# Descriptive aliases make the clean-runtime command and library boundary
# discoverable without duplicating validation logic.
validate_path_b_v2 = validate_path_b_v2_bundle
validate_path_b_v2_review_pack = validate_path_b_v2_bundle
validate_path_b_v2_records = validate_eligibility_rows


__all__ = [
    "PATH_B_V2_ATTESTATIONS_FILENAME",
    "PATH_B_V2_DATASET_REPO_ID",
    "PATH_B_V2_DATASET_REVISION",
    "PATH_B_V2_DATASET_TREE_SHA",
    "PATH_B_V2_DECISIONS",
    "PATH_B_V2_EVIDENCE_COMPONENTS",
    "PATH_B_V2_ELIGIBILITY_FILENAME",
    "PATH_B_V2_EXCLUDED_CATEGORIES",
    "PATH_B_V2_MANIFEST_FILENAME",
    "PATH_B_V2_MANIFEST_SCHEMA_VERSION",
    "PATH_B_V2_PACKAGES_FILENAME",
    "PATH_B_V2_REQUIRED_ATTESTATION_FIELDS",
    "PATH_B_V2_REVIEW_CONTRACT_FILENAME",
    "PATH_B_V2_REVIEW_CONTRACT_SCHEMA_VERSION",
    "PATH_B_V2_REVIEW_METHODS",
    "PATH_B_V2_REVIEW_METHOD_SET",
    "PATH_B_V2_RETURN_FORMULA",
    "PATH_B_V2_RETURN_FORMULA_ID",
    "PATH_B_V2_SCHEMA_VERSION",
    "PATH_B_V2_SCOPE_CATEGORIES",
    "PathBV2ScoreBlindError",
    "PathBV2ValidationError",
    "assert_path_b_v2_score_blind",
    "build_default_review_contract",
    "build_evidence_package_manifest",
    "build_task_identity_lock",
    "canonical_json",
    "compute_attestation_sha256",
    "compute_evidence_package_sha256",
    "compute_return_derivation_sha256",
    "sha256_json",
    "validate_attestation",
    "validate_b_evidence",
    "validate_evidence_package",
    "validate_evidence_package_manifest",
    "validate_eligibility_rows",
    "validate_path_b_v2",
    "validate_path_b_v2_bundle",
    "validate_path_b_v2_records",
    "validate_path_b_v2_review_pack",
    "validate_review_contract",
    "validate_task_identity_unchanged",
]
