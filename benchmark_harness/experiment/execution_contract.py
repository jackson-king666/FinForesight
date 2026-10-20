"""Atomic run-leaf contracts that prevent cross-lineage ledger reuse."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmark_harness.experiment.live_model import declared_model_identity
from benchmark_harness.experiment.manifest import ExperimentManifest, ManifestError
from benchmark_harness.experiment.planner import ExperimentPlan

EXECUTION_CONTRACT_FILENAME = "execution-contract.json"
EXECUTION_CONTRACT_SCHEMA_VERSION = "execution-contract/v1"
EXECUTION_CONTRACT_V2_SCHEMA_VERSION = "execution-contract/v2"


class ExecutionContractError(ManifestError):
    """Raised when a run leaf is missing or belongs to another lineage."""


def build_execution_contract(manifest: ExperimentManifest, plan: ExperimentPlan) -> dict[str, Any]:
    """Build the identity sentinel for one manifest/plan result leaf."""

    if manifest.manifest_hash != plan.manifest_hash:
        raise ExecutionContractError("manifest hash does not match execution plan")
    if manifest.schema_version == "experiment/v4":
        metadata = manifest.v4_metadata
        dataset = metadata.get("dataset", {})
        if not isinstance(dataset, Mapping):
            dataset = {}
        supplements = metadata.get("supplements", dataset.get("supplements", []))
        if not isinstance(supplements, list):
            raise ExecutionContractError("experiment/v4 supplements must be a list")
        supplement_hashes: list[str] = []
        for supplement in supplements:
            if isinstance(supplement, Mapping):
                digest = supplement.get("sha256", supplement.get("hash"))
            else:
                digest = supplement
            if not isinstance(digest, str) or len(digest) != 64:
                raise ExecutionContractError("experiment/v4 supplement hashes must be SHA-256")
            supplement_hashes.append(digest)
        cohort = metadata.get("cohort", {})
        cohort_sha256 = cohort.get("sha256") if isinstance(cohort, Mapping) else None
        t3 = metadata.get("t3", {})
        t3_sha256 = t3.get("sha256", t3.get("registry_sha256")) if isinstance(t3, Mapping) else None
        calculation_policy = metadata.get("calculation_policy", {})
        formal_version = (
            metadata.get("analysis", {}).get("formal_utility_version")
            if isinstance(metadata.get("analysis"), Mapping)
            else None
        )
        canonical = json.dumps(
            calculation_policy,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        result = {
            "schema_version": EXECUTION_CONTRACT_V2_SCHEMA_VERSION,
            "experiment_id": manifest.experiment_id,
            "manifest_hash": manifest.manifest_hash,
            "plan_id": plan.plan_id,
            "plan_hash": plan.plan_hash,
            "dataset_revision": manifest.dataset_revision,
            "dataset_base_pin": {
                "repo_id": manifest.dataset_repo_id,
                "revision": manifest.dataset_revision,
                "tree_sha": dataset.get("tree_sha", dataset.get("tree_sha256")),
            },
            "supplement_hashes": sorted(supplement_hashes),
            "cohort_sha256": cohort_sha256,
            "t3_registry_sha256": t3_sha256,
            "calculation_policy_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "formal_utility_version": formal_version or "formal-utility/v1",
            "retrieval_regimes": list(manifest.retrieval_regimes),
            "systems": [system.system_id for system in manifest.systems],
            "trials": manifest.trial_count,
        }
        public_metadata = metadata.get("metadata", {})
        historical_execution = (
            public_metadata.get("historical_execution", {})
            if isinstance(public_metadata, Mapping)
            else {}
        )
        service_identity = (
            historical_execution.get("service_identity")
            if isinstance(historical_execution, Mapping)
            else None
        )
        if service_identity is not None:
            if not isinstance(service_identity, Mapping):
                raise ExecutionContractError("historical service identity must be an object")
            result["search_service_identity"] = dict(service_identity)
        return result
    identity = declared_model_identity(manifest)
    return {
        "schema_version": EXECUTION_CONTRACT_SCHEMA_VERSION,
        "experiment_id": manifest.experiment_id,
        "manifest_hash": manifest.manifest_hash,
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash,
        "dataset_revision": manifest.dataset_revision,
        "backbone_id": identity["backbone_id"],
        "snapshot_model_name": identity["snapshot_model_name"],
        "wire_model_name": identity["wire_model_name"],
        "wire_name_binding": identity["wire_name_binding"],
    }


def _read_contract(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExecutionContractError("execution contract is invalid") from exc
    if not isinstance(value, Mapping):
        raise ExecutionContractError("execution contract is invalid")
    return value


def validate_execution_contract(
    root: str | Path, manifest: ExperimentManifest, plan: ExperimentPlan
) -> dict[str, Any]:
    """Validate the sentinel before any attempt ledger is read."""

    root_path = Path(root).expanduser().resolve()
    path = root_path / EXECUTION_CONTRACT_FILENAME
    if not path.is_file():
        raise ExecutionContractError("execution contract is missing")
    expected = build_execution_contract(manifest, plan)
    observed = _read_contract(path)
    if dict(observed) != expected:
        raise ExecutionContractError("execution contract mismatch")
    return expected


def _write_contract_atomically(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=".execution-contract-", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def ensure_execution_contract(
    root: str | Path, manifest: ExperimentManifest, plan: ExperimentPlan
) -> dict[str, Any]:
    """Create an empty run-leaf sentinel atomically or validate an existing one."""

    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    path = root_path / EXECUTION_CONTRACT_FILENAME
    if path.exists():
        return validate_execution_contract(root_path, manifest, plan)
    try:
        has_other_files = any(root_path.iterdir())
    except OSError as exc:
        raise ExecutionContractError("execution contract root is unavailable") from exc
    if has_other_files:
        raise ExecutionContractError("execution contract is missing for non-empty run leaf")
    expected = build_execution_contract(manifest, plan)
    _write_contract_atomically(path, expected)
    return expected


__all__ = [
    "EXECUTION_CONTRACT_FILENAME",
    "EXECUTION_CONTRACT_SCHEMA_VERSION",
    "EXECUTION_CONTRACT_V2_SCHEMA_VERSION",
    "ExecutionContractError",
    "build_execution_contract",
    "ensure_execution_contract",
    "validate_execution_contract",
]
