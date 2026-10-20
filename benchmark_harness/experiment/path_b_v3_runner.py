"""Fail-closed execution boundary for the current-pin Path B v3 plans.

The repository's legacy ``experiment`` parser is intentionally not used here:
the sealed G3 plans have their own schema and are bound to the 951 revision.
This module validates that complete immutable bundle, exposes a record-free
task loader, and performs the credential gate before an execution backend can
be called.  The backend is injected by the operator because the unrestricted
current-web transport is a separate, explicitly isolated capability.

No function in this module writes a result, reads the frozen evaluator, or
falls back to the legacy 339-row/e941 data path.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from benchmark_harness.contracts import TaskInput
from benchmark_harness.dataset_sources.v3 import V3DatasetSource, V3DatasetSourceError
from benchmark_harness.experiment.manifest import ManifestError
from benchmark_harness.experiment.path_b_v2 import (
    PathBV2ValidationError,
    sha256_json,
    validate_path_b_v2_bundle,
    validate_review_contract,
)

PATH_B_V3_DATASET_REPO_ID = "sselaine27/benchmark-research"
PATH_B_V3_DATASET_REVISION = "951edec3021353bf1f0db23a49cf8596a7148490"
PATH_B_V3_DATASET_TREE_SHA = "410c6c450abd5df4d73d6827004dd12a1cbf0c1b"

PATH_B_V3_G2_SCHEMA_VERSION = "path-b-v3-g2-freeze/v1"
PATH_B_V3_G3_INDEX_SCHEMA_VERSION = "path-b-v3-g3-plan-index/v1"
PATH_B_V3_MANIFEST_SCHEMA_VERSION = "path-b-v3-experiment-manifest/v1"
PATH_B_V3_T1_T2_PLAN_SCHEMA = "path-b-v3-t1-t2-plan/v1"
PATH_B_V3_B_CALIBRATION_PLAN_SCHEMA = "path-b-v3-b-calibration-plan/v1"
PATH_B_V3_LEAKAGE_PLAN_SCHEMA = "leakage-probe-plan/v3"

PATH_B_V3_ROSTER_COUNT = 543
PATH_B_V3_ELIGIBLE_COUNT = 405
PATH_B_V3_INELIGIBLE_COUNT = 138
PATH_B_V3_T1_T2_PAIR_COUNT = 112
PATH_B_V3_T1_T2_CELL_COUNT = 2016
PATH_B_V3_LEAKAGE_PAIR_INPUT_COUNT = 3645
PATH_B_V3_LEAKAGE_CELL_COUNT = 7290

PATH_B_V3_G2_MANIFEST_FILENAME = "g2-freeze-manifest.json"
PATH_B_V3_G3_INDEX_FILENAME = "g3-plan-index.json"
PATH_B_V3_G2_MANIFEST_SHA256 = "09a62ea803a0326a2ea3ca20b4c322b72851d9866a6e4914a03eb748d253be08"
PATH_B_V3_G2_ROSTER_SHA256 = "83a728adcfec48cf612325aa3d59715c9fc5007d5a8feb337c8e3095599504f0"
PATH_B_V3_G2_ATTESTATION_MANIFEST_SHA256 = (
    "263773c390289133d7017ed40d06eeff79ce8ea4e066e56cfe21516d89fe72df"
)
PATH_B_V3_G2_PACKAGE_MANIFEST_SHA256 = (
    "72da9c6c3ea11c56771b614f204d9cf124d1f707775ffc546d921c0911508365"
)
PATH_B_V3_G2_ADJUDICATION_SHA256 = (
    "863b34023e05edf43b3fa49337e6a01b65591f4bc4cb8f19297a318da4639b23"
)
PATH_B_V3_T1_T2_SPEC_SHA256 = "5b837fdfec2215ed78619a774668cedaaf5573368648e1c01d05a968d8c7d4c0"
PATH_B_V3_T1_T2_COHORT_SHA256 = "9730bf6e03ae4ffe55426870a0e53e8879122e2d05fbbdb8ec28fdd9e0569447"
PATH_B_V3_B_SPEC_SHA256 = "cb8336e5b146b4c903b34278cfa3d4069d3ea563cda451b9cd2556335dd598b2"
PATH_B_V3_B_REPORT_SHA256 = "7cca6b64c7c994d23eafe884681443744adf1911bf238c307c9755ce14703c3e"
PATH_B_V3_G3_INDEX_SHA256 = "fd2bd81efc77329910e1249dc088be214d6cf88ddb8a66a38a1d51fe24f5f79c"

PATH_B_V3_RESULTS_ROOTS = {
    "t1_t2": "results/path-b-v3-t1-t2-r3",
    "b_calibration": "results/path-b-v3-b-calibration-r3",
    "cutoff_unrestricted": "results/path-b-v3-leakage-probe-r3",
}
PATH_B_V3_MANIFEST_FILES = {
    "t1_t2": ("t1-t2-manifest.json", "t1-t2-plan.json"),
    "b_calibration": ("b-calibration-manifest.json", "b-calibration-plan.json"),
    "cutoff_unrestricted": (
        "cutoff-unrestricted-manifest.json",
        "cutoff-unrestricted-plan.json",
    ),
}
PATH_B_V3_MANIFEST_SHA256S = {
    "t1_t2": "875026ca01f917d82b7e32ddabd709d3fa07a5d76600a2c0405e97b61df44bd6",
    "b_calibration": "30c77587ae0f5ff412863121b3e571698ef62b857e0a24f5747395980040289d",
    "cutoff_unrestricted": ("9291723b9f7fda341d183a716c95c3ed002404049793505c1c7b119d806c684e"),
}
PATH_B_V3_PLAN_SHA256S = {
    "t1_t2": "5c67197466444b59a6eee3150536736bf73b728d73aa642b2f6656510772e46b",
    "b_calibration": "55ffb6223d6973318f09fe12f0cc3e19f5751fd1236c3ab4f029b127e9624adf",
    "cutoff_unrestricted": ("e6acee7e87029fdb5cb89b36206ec11153ed43c57fa25b5d77e34e19aae81b71"),
}
PATH_B_V3_GPT41_BACKBONE_ID = "openai:gpt-4.1-2025-04-14"
PATH_B_V3_GPT41_API_MODEL_NAME = "gpt-4.1-2025-04-14"
PATH_B_V3_GPT41_CUTOFF_SOURCE = "https://developers.openai.com/api/docs/models/gpt-4.1"
PATH_B_V3_MINI_SEAL_ID = "path-b-v3-g3-r4-mini"
PATH_B_V3_MINI_BACKBONE_ID = "openai:gpt-4.1-mini-2025-04-14"
PATH_B_V3_MINI_SNAPSHOT_MODEL_NAME = "gpt-4.1-mini-2025-04-14"
PATH_B_V3_MINI_API_MODEL_NAME = "gpt-4.1-mini"
PATH_B_V3_MINI_CUTOFF_SOURCE = "https://developers.openai.com/api/docs/models/gpt-4.1-mini"
PATH_B_V3_MINI_ENDPOINT_BINDING = "operator_supplied_env"
PATH_B_V3_MINI_ENDPOINT_ENV = "BENCHMARK_MODEL_ENDPOINT"
PATH_B_V3_MINI_WIRE_NAME_BINDING = "user_authorized_azure_wire_mapping/v1"
PATH_B_V3_MINI_G3_INDEX_SHA256 = "d21a69ba331113fe94c067a6dc1bbb5922a08115e86def6c5db56eefcf34bd2e"
PATH_B_V3_MINI_MANIFEST_SHA256S = {
    "t1_t2": "6980810ad3e75b657cb7d831a0cbc80231ac4dd58b76679d43d36f83e748a836",
    "b_calibration": "e0d6bf10e6a2de9a564930d83c9d6622e28872069ef51fc05d550186a10131e4",
    "cutoff_unrestricted": "f05c1d0e575b7cba6f5d420636fff34257e96a820b77a0a5d607a9481b0e763b",
}
PATH_B_V3_MINI_PLAN_SHA256S = {
    "t1_t2": "4c133158a3941789a2624862e87aa091f2d38c78c8fae78b7b3bcbfac1131ec8",
    "b_calibration": "97b9e1663f08da544cb432c981e1130385ba75eca3861a57a36acd2f965f5220",
    "cutoff_unrestricted": "3008561f15859dd49a5623c6a623ecffdcf1c8b79c45002a312e510e8e7aa432",
}
PATH_B_V3_TASK_IDS_SHA256 = "d3954538b3360b662b70e3c0c3695d8a41ec541d663b6a28024ed4c5ce454ad5"
PATH_B_V3_LEAKAGE_TASK_IDS_SHA256 = (
    "c7c9fbbaea1ff803ce3092f402069de21c55ab9e2a41ad81bce57be0d891223f"
)
PATH_B_V3_REVIEW_CONTRACT_SHA256 = (
    "95cde35269bffc8ef498e557ae9c9594b5707a4cdbb460d674e47cd678cc6258"
)
PATH_B_V3_BUDGET_SHA256 = "c14e598f7ccaaa6fd7234c9c2c1aa3d42dca964ffeacfb0731418bb4ca37d1f1"

MODEL_API_KEY_ENV = "BENCHMARK_MODEL_API_KEY"
SEARCH_SERVICE_TOKEN_FILE_ENV = "BENCHMARK_SEARCH_SERVICE_ADMIN_TOKEN_FILE"
SEARCH_SERVICE_TOKEN_ENV = "BENCHMARK_SEARCH_SERVICE_ADMIN_TOKEN"
UNRESTRICTED_CREDENTIALS_FILE_ENV = "BENCHMARK_V3_UNRESTRICTED_CREDENTIALS_FILE"
UNRESTRICTED_CREDENTIALS_ENV = "BENCHMARK_V3_UNRESTRICTED_CREDENTIALS"


@dataclass(frozen=True, slots=True)
class PathBV3G3Seal:
    """Expected immutable identity for one G3 model seal."""

    seal_id: str
    backbone_id: str
    snapshot_model_name: str
    api_model_name: str
    cutoff_source: str
    endpoint_binding: str
    endpoint_env: str
    wire_name_binding: str
    seed_namespace: str
    index_sha256: str
    manifest_sha256s: Mapping[str, str]
    plan_sha256s: Mapping[str, str]


_PATH_B_V3_GPT41_SEAL = PathBV3G3Seal(
    seal_id="path-b-v3-g3-r3-gpt41",
    backbone_id=PATH_B_V3_GPT41_BACKBONE_ID,
    snapshot_model_name=PATH_B_V3_GPT41_API_MODEL_NAME,
    api_model_name=PATH_B_V3_GPT41_API_MODEL_NAME,
    cutoff_source=PATH_B_V3_GPT41_CUTOFF_SOURCE,
    endpoint_binding="sealed_artifact",
    endpoint_env="BENCHMARK_MODEL_ENDPOINT",
    wire_name_binding="sealed_path_b_v3_snapshot",
    seed_namespace="path-b-v3-r3|openai:gpt-4.1-2025-04-14",
    index_sha256=PATH_B_V3_G3_INDEX_SHA256,
    manifest_sha256s=PATH_B_V3_MANIFEST_SHA256S,
    plan_sha256s=PATH_B_V3_PLAN_SHA256S,
)
PATH_B_V3_MINI_SEAL = PathBV3G3Seal(
    seal_id=PATH_B_V3_MINI_SEAL_ID,
    backbone_id=PATH_B_V3_MINI_BACKBONE_ID,
    snapshot_model_name=PATH_B_V3_MINI_SNAPSHOT_MODEL_NAME,
    api_model_name=PATH_B_V3_MINI_API_MODEL_NAME,
    cutoff_source=PATH_B_V3_MINI_CUTOFF_SOURCE,
    endpoint_binding=PATH_B_V3_MINI_ENDPOINT_BINDING,
    endpoint_env=PATH_B_V3_MINI_ENDPOINT_ENV,
    wire_name_binding=PATH_B_V3_MINI_WIRE_NAME_BINDING,
    seed_namespace="path-b-v3-r4-mini|openai:gpt-4.1-mini-2025-04-14",
    index_sha256=PATH_B_V3_MINI_G3_INDEX_SHA256,
    manifest_sha256s=PATH_B_V3_MINI_MANIFEST_SHA256S,
    plan_sha256s=PATH_B_V3_MINI_PLAN_SHA256S,
)
_PATH_B_V3_G3_SEALS = {
    _PATH_B_V3_GPT41_SEAL.seal_id: _PATH_B_V3_GPT41_SEAL,
    PATH_B_V3_MINI_SEAL.seal_id: PATH_B_V3_MINI_SEAL,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_SYSTEM_IDS = frozenset({"financeharness", "gpt-researcher", "odr"})
_ALLOWED_PLAN_KEY_NAMES = frozenset(
    {
        "physical_results_roots",
        "results_root",
        "cutoff_results_root",
        "unrestricted_results_root",
        "physical_results_root",
        "score_selection",
    }
)
_FORBIDDEN_PLAN_KEY_PARTS = (
    "score",
    "prediction",
    "result",
    "evaluator",
    "metric",
    "gold",
    "ground_truth",
    "actual",
    "target",
    "label",
)
_PLACEHOLDER_CREDENTIAL_VALUES = frozenset(
    {
        "",
        "replace_me",
        "your_value_here",
        "todo",
        "changeme",
        "none",
        "null",
    }
)


class PathBV3RunnerError(ManifestError):
    """Raised when a sealed Path B v3 input or execution boundary is invalid."""


class PathBV3ValidationError(PathBV3RunnerError):
    """Raised when a current-pin manifest, plan, or G2 binding is invalid."""


class PathBV3CredentialError(PathBV3RunnerError):
    """Raised before execution when a required live credential is unavailable."""


class PathBV3ExecutionError(PathBV3RunnerError):
    """Raised when no explicitly supplied live execution backend is available."""


@dataclass(frozen=True, slots=True)
class PathBV3CredentialStatus:
    """Presence-only credential status; secret values never cross this object."""

    model_api_key_present: bool
    search_admin_token_present: bool
    unrestricted_probe_present: bool
    missing: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.missing

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_api_key_present": self.model_api_key_present,
            "search_admin_token_present": self.search_admin_token_present,
            "unrestricted_probe_present": self.unrestricted_probe_present,
            "missing": list(self.missing),
            "ready": self.ready,
        }


@dataclass(frozen=True, slots=True)
class PathBV3Plans:
    """Validated, score-free references to one immutable G3 plan bundle."""

    root: Path
    g2_root: Path
    index: Mapping[str, Any]
    manifests: Mapping[str, Mapping[str, Any]]
    plans: Mapping[str, Mapping[str, Any]]
    eligible_task_ids: tuple[str, ...]

    @property
    def results_roots(self) -> dict[str, Path]:
        return {
            kind: (self.root.parent.parent / root).resolve()
            for kind, root in PATH_B_V3_RESULTS_ROOTS.items()
        }

    def summary(self) -> dict[str, Any]:
        return _validation_summary(self)


class PathBV3ExecutionBackend(Protocol):
    """Operator-owned backend for the already validated plan set.

    The backend must write only to ``request.results_roots`` and preserve the
    plan's non-ranking flag.  Keeping it injected prevents this coordinator
    from accidentally routing unrestricted cells through the cutoff service.
    """

    def __call__(self, request: PathBV3ExecutionRequest) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class PathBV3ExecutionRequest:
    """Safe backend input containing tasks but no evaluator or gold fields."""

    plans: PathBV3Plans
    task_inputs: Mapping[str, TaskInput]
    credentials: PathBV3CredentialStatus
    results_roots: Mapping[str, Path]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PathBV3ValidationError(f"{name} must be an object")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PathBV3ValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _sha(value: Any, name: str) -> str:
    parsed = _nonempty(value, name)
    if _SHA256.fullmatch(parsed) is None:
        raise PathBV3ValidationError(f"{name} must be a lowercase SHA-256")
    return parsed


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PathBV3ValidationError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise PathBV3ValidationError(f"{name} must be at least {minimum}")
    return value


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PathBV3ValidationError(f"unable to read {name}: {path}") from exc
    return _mapping(value, name)


def _hash_without(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return sha256_json(body)


def _verify_self_hash(value: Mapping[str, Any], field: str, name: str) -> str:
    claimed = _sha(value.get(field), f"{name}.{field}")
    if claimed != _hash_without(value, field):
        raise PathBV3ValidationError(f"{name} hash mismatch")
    return claimed


def _seal_for_index(index: Mapping[str, Any]) -> PathBV3G3Seal:
    """Resolve a known G3 seal without accepting an arbitrary self-hashed bundle."""

    seal_id = index.get("seal_id")
    if seal_id is None:
        if index.get("index_sha256") == PATH_B_V3_G3_INDEX_SHA256:
            return _PATH_B_V3_GPT41_SEAL
        raise PathBV3ValidationError("G3 plan index has no recognized model seal")
    if not isinstance(seal_id, str) or seal_id not in _PATH_B_V3_G3_SEALS:
        raise PathBV3ValidationError("G3 plan index has an unsupported model seal")
    seal = _PATH_B_V3_G3_SEALS[seal_id]
    expected_binding = {
        "api_model_name": seal.api_model_name,
        "backbone_id": seal.backbone_id,
        "endpoint_binding": seal.endpoint_binding,
        "endpoint_env": seal.endpoint_env,
        "seal_id": seal.seal_id,
        "snapshot_model_name": seal.snapshot_model_name,
        "wire_name_binding": seal.wire_name_binding,
    }
    if dict(_mapping(index.get("model_binding"), "G3 model binding")) != expected_binding:
        raise PathBV3ValidationError("G3 model binding does not match its recognized seal")
    return seal


def _resolve_inside(root: Path, relative_name: Any, name: str) -> Path:
    relative = _nonempty(relative_name, name)
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PathBV3ValidationError(f"{name} must be a safe relative path")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise PathBV3ValidationError(f"{name} escapes the sealed plan root") from exc
    if not resolved.is_file():
        raise PathBV3ValidationError(f"declared {name} is unavailable: {resolved}")
    return resolved


def _assert_score_blind_plan(value: Any, *, path: str = "$") -> None:
    """Reject score-bearing plan keys while allowing declared result roots."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            normalized = re.sub(r"[^a-z0-9]+", "_", key_text.casefold()).strip("_")
            if key_text not in _ALLOWED_PLAN_KEY_NAMES and any(
                part in normalized for part in _FORBIDDEN_PLAN_KEY_PARTS
            ):
                raise PathBV3ValidationError(
                    f"score/result/prediction/evaluator field is forbidden at {path}.{key_text}"
                )
            _assert_score_blind_plan(child, path=f"{path}.{key_text}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            _assert_score_blind_plan(child, path=f"{path}[{index}]")


def _validate_g2(g2_root: Path, contract_path: Path) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    if not g2_root.is_dir():
        raise PathBV3ValidationError(f"G2 freeze root is not a directory: {g2_root}")
    contract = validate_review_contract(_read_json(contract_path, "review contract"))
    contract_sha = sha256_json(contract)
    if contract_sha != PATH_B_V3_REVIEW_CONTRACT_SHA256:
        raise PathBV3ValidationError("review contract is not the frozen Path B v2 contract")
    try:
        v2_result = validate_path_b_v2_bundle(g2_root, contract_path=contract_path)
    except (PathBV2ValidationError, OSError, ValueError) as exc:
        raise PathBV3ValidationError(f"G2 Path B v2 validation failed: {exc}") from exc
    if v2_result.get("status") != "valid":
        raise PathBV3ValidationError("G2 Path B v2 validator did not return valid")
    if v2_result.get("total_count") != PATH_B_V3_ROSTER_COUNT:
        raise PathBV3ValidationError("G2 validator did not cover the 543-row roster")
    if v2_result.get("eligible_count") != PATH_B_V3_ELIGIBLE_COUNT:
        raise PathBV3ValidationError("G2 validator eligible count differs from the sealed cohort")
    manifest_path = g2_root / PATH_B_V3_G2_MANIFEST_FILENAME
    manifest = _read_json(manifest_path, "G2 freeze manifest")
    _assert_score_blind_plan(manifest, path="$.g2_manifest")
    _verify_self_hash(manifest, "manifest_sha256", "G2 freeze manifest")
    if manifest.get("manifest_sha256") != PATH_B_V3_G2_MANIFEST_SHA256:
        raise PathBV3ValidationError("G2 freeze manifest is not the sealed current-pin manifest")
    if manifest.get("schema_version") != PATH_B_V3_G2_SCHEMA_VERSION:
        raise PathBV3ValidationError("unsupported G2 freeze schema")
    if manifest.get("status") != "g2_frozen":
        raise PathBV3ValidationError("G2 freeze is not marked g2_frozen")
    if manifest.get("dataset_revision") != PATH_B_V3_DATASET_REVISION:
        raise PathBV3ValidationError("G2 dataset revision is not the current 951 pin")
    if manifest.get("dataset_tree_sha") != PATH_B_V3_DATASET_TREE_SHA:
        raise PathBV3ValidationError("G2 dataset tree does not match the current 951 pin")
    counts = _mapping(manifest.get("counts"), "G2 counts")
    expected_counts = {
        "roster": PATH_B_V3_ROSTER_COUNT,
        "eligible": PATH_B_V3_ELIGIBLE_COUNT,
        "ineligible": PATH_B_V3_INELIGIBLE_COUNT,
        "validator_packages": PATH_B_V3_ELIGIBLE_COUNT,
        "t1_t2_pairs": PATH_B_V3_T1_T2_PAIR_COUNT,
        "b_available_case": 0,
        "b_earnings_excluded": 125,
        "b_macro_audit_only": 13,
    }
    if dict(counts) != expected_counts:
        raise PathBV3ValidationError(f"G2 counts do not match the sealed cohort: {counts}")
    expected_manifest_fields = {
        "review_contract_sha256": contract_sha,
        "roster_sha256": PATH_B_V3_G2_ROSTER_SHA256,
        "attestation_manifest_sha256": PATH_B_V3_G2_ATTESTATION_MANIFEST_SHA256,
        "evidence_package_manifest_sha256": PATH_B_V3_G2_PACKAGE_MANIFEST_SHA256,
        "adjudication_sha256": PATH_B_V3_G2_ADJUDICATION_SHA256,
        "t1_t2_analysis_spec_sha256": PATH_B_V3_T1_T2_SPEC_SHA256,
        "t1_t2_cohort_sha256": PATH_B_V3_T1_T2_COHORT_SHA256,
        "b_calibration_spec_sha256": PATH_B_V3_B_SPEC_SHA256,
        "b_calibration_report_sha256": PATH_B_V3_B_REPORT_SHA256,
    }
    if any(manifest.get(key) != value for key, value in expected_manifest_fields.items()):
        raise PathBV3ValidationError("G2 freeze references do not match the sealed artifacts")
    expected_anchors = {
        "cr_import_file_sha256": (
            "029cbe7c530356bc8834a82f891d5a54dcc21b1201ce7d9da7e64d0a7923d4dc"
        ),
        "qa_import_file_sha256": (
            "b8db9000e32ba5465d5092c4a47f74b915447a962f497b8040cb873e41ce451b"
        ),
        "r3_derived_manifest_file_sha256": (
            "7f67e96e63b1beddf214b715aeb0b3302103d86ade6361c3229a5cdaf4807311"
        ),
        "r3_export_file_sha256": (
            "1125d026f708fe3ca7520d1f9807d09ce2fdcd98a2efd5c7e0aea96f410485c2"
        ),
        "r3_pack_manifest_sha256": (
            "73b784f6c938510ef289e33ae407982c5298b69566a31d82f13718d89bba3d02"
        ),
        "r3_roster_file_sha256": (
            "349dda37a0749563e01da939e47c8be526b36821a34bf657c8e598d99bbdacdb"
        ),
        "r3_source_lock_file_sha256": (
            "9204da7e0d5b1e3159decc431e369eb10cc30041f8da7f8a2baa53f6da9ae7da"
        ),
    }
    if dict(_mapping(manifest.get("source_anchors"), "G2 source anchors")) != expected_anchors:
        raise PathBV3ValidationError("G2 source anchors do not match the sealed r3 pack")
    invalidated = _mapping(manifest.get("invalidated_prior_rounds"), "G2 invalidated rounds")
    if invalidated.get("disposition") != "retained_for_disclosure_not_for_adjudication":
        raise PathBV3ValidationError("G2 does not preserve the required r2 invalidation record")
    old_imports = sorted(invalidated.get("r2_adjudication_imports", []))
    if old_imports != sorted(
        [
            "e60911ea6aa4fe5f2a5f3050ad1a7d71c4116568217006669d02457d1cc69c50",
            "f53b007b36173c9bc928df22e252692299d400c13ac18cea34dbac9f78caf957",
        ]
    ):
        raise PathBV3ValidationError("G2 r2 invalidation anchors are incomplete")
    eligible = tuple(sorted(str(value) for value in v2_result.get("eligible_task_ids", [])))
    if len(eligible) != PATH_B_V3_ELIGIBLE_COUNT or len(set(eligible)) != len(eligible):
        raise PathBV3ValidationError("G2 eligible task identity set is malformed")
    if manifest.get("roster_sha256") != PATH_B_V3_G2_ROSTER_SHA256:
        raise PathBV3ValidationError("G2 roster binding mismatch")
    return manifest, eligible


def _validate_common_manifest(
    manifest: Mapping[str, Any],
    *,
    kind: str,
    expected_plan_filename: str,
    seal: PathBV3G3Seal,
) -> None:
    if manifest.get("schema_version") != PATH_B_V3_MANIFEST_SCHEMA_VERSION:
        raise PathBV3ValidationError(f"{kind} uses an unsupported manifest schema")
    _verify_self_hash(manifest, "manifest_sha256", f"{kind} manifest")
    if manifest.get("manifest_sha256") != seal.manifest_sha256s[kind]:
        raise PathBV3ValidationError(f"{kind} manifest is not the sealed current-pin artifact")
    dataset = _mapping(manifest.get("dataset"), f"{kind} manifest.dataset")
    if dict(dataset) != {
        "repo_id": PATH_B_V3_DATASET_REPO_ID,
        "revision": PATH_B_V3_DATASET_REVISION,
        "tree_sha": PATH_B_V3_DATASET_TREE_SHA,
    }:
        raise PathBV3ValidationError(f"{kind} dataset binding is not the 951 pin")
    contract = _mapping(manifest.get("source_contract"), f"{kind} manifest.source_contract")
    expected_contract = {
        "evidence_package_manifest_sha256": PATH_B_V3_G2_PACKAGE_MANIFEST_SHA256,
        "g2_freeze_manifest_sha256": PATH_B_V3_G2_MANIFEST_SHA256,
        "review_contract_sha256": PATH_B_V3_REVIEW_CONTRACT_SHA256,
        "roster_sha256": PATH_B_V3_G2_ROSTER_SHA256,
    }
    if dict(contract) != expected_contract:
        raise PathBV3ValidationError(f"{kind} manifest source contract drifted")
    if manifest.get("ranking_eligible") is not False:
        raise PathBV3ValidationError(f"{kind} is not explicitly non-ranking")
    if manifest.get("score_selection") != "forbidden":
        raise PathBV3ValidationError(f"{kind} manifest permits score selection")
    if manifest.get("claim_tier") != "exploratory":
        raise PathBV3ValidationError(f"{kind} must remain exploratory")
    if manifest.get("plan_filename") != expected_plan_filename:
        raise PathBV3ValidationError(f"{kind} plan filename is not the sealed plan")
    if seal.endpoint_binding == "operator_supplied_env":
        expected_binding = {
            "api_model_name": seal.api_model_name,
            "backbone_id": seal.backbone_id,
            "endpoint_binding": seal.endpoint_binding,
            "endpoint_env": seal.endpoint_env,
            "seal_id": seal.seal_id,
            "snapshot_model_name": seal.snapshot_model_name,
            "wire_name_binding": seal.wire_name_binding,
        }
        if dict(_mapping(manifest.get("model_binding"), f"{kind} model binding")) != (
            expected_binding
        ):
            raise PathBV3ValidationError(f"{kind} model binding drifted")
    _sha(manifest.get("plan_sha256"), f"{kind} manifest.plan_sha256")
    if manifest.get("plan_sha256") != seal.plan_sha256s[kind]:
        raise PathBV3ValidationError(f"{kind} plan binding is not the sealed plan")
    trials = _mapping(manifest.get("trials"), f"{kind} manifest.trials")
    if trials.get("count") != 3 or trials.get("order") != "blocked_randomized":
        raise PathBV3ValidationError(f"{kind} trial schedule drifted")
    systems = manifest.get("systems")
    if not isinstance(systems, list) or len(systems) != len(_EXPECTED_SYSTEM_IDS):
        raise PathBV3ValidationError(f"{kind} manifest must declare exactly three systems")
    observed_systems: set[str] = set()
    for index, raw_system in enumerate(systems):
        system = _mapping(raw_system, f"{kind} manifest.systems[{index}]")
        system_id = _nonempty(system.get("system_id"), f"{kind} system_id")
        observed_systems.add(system_id)
        if system.get("ranking_eligible") is not False:
            raise PathBV3ValidationError(f"{kind} system {system_id} is ranking eligible")
        if system.get("profile") != "deep_research":
            raise PathBV3ValidationError(f"{kind} system {system_id} has the wrong profile")
        if system.get("backbone_id") != seal.backbone_id:
            raise PathBV3ValidationError(f"{kind} system {system_id} has the wrong backbone")
        for field in ("adapter_id", "adapter_revision", "framework_revision", "strategy_revision"):
            _nonempty(system.get(field), f"{kind} system {system_id}.{field}")
    if observed_systems != _EXPECTED_SYSTEM_IDS:
        raise PathBV3ValidationError(f"{kind} systems do not match the sealed system set")
    backbone = _mapping(manifest.get("backbone"), f"{kind} manifest.backbone")
    if (
        backbone.get("backbone_id") != seal.backbone_id
        or backbone.get("training_data_cutoff") != "2024-06-01"
        or backbone.get("cutoff_source") != seal.cutoff_source
    ):
        raise PathBV3ValidationError(f"{kind} backbone identity drifted")
    settings = _mapping(backbone.get("model_settings"), f"{kind} backbone.model_settings")
    identity_matches = (
        settings.get("api_key_env") == MODEL_API_KEY_ENV
        and settings.get("api_model_name") == seal.api_model_name
        and settings.get("backbone_id") == seal.backbone_id
    )
    if seal.endpoint_binding == "operator_supplied_env":
        identity_matches = identity_matches and (
            settings.get("snapshot_model_name") == seal.snapshot_model_name
            and settings.get("wire_name_binding") == seal.wire_name_binding
        )
    if not identity_matches:
        raise PathBV3ValidationError(f"{kind} model identity drifted")
    if settings.get("endpoint_env") != seal.endpoint_env:
        raise PathBV3ValidationError(f"{kind} model endpoint binding drifted")
    if settings.get("name_env") != "BENCHMARK_MODEL_NAME":
        raise PathBV3ValidationError(f"{kind} model name binding drifted")
    if seal.endpoint_binding == "operator_supplied_env":
        if settings.get("endpoint_binding") != seal.endpoint_binding or "endpoint" in settings:
            raise PathBV3ValidationError(
                f"{kind} must bind the operator endpoint without sealing its value"
            )
    else:
        if settings.get("endpoint_binding", "sealed_artifact") != seal.endpoint_binding:
            raise PathBV3ValidationError(f"{kind} sealed endpoint binding drifted")
        _nonempty(settings.get("endpoint"), f"{kind} model_settings.endpoint")
    budget = _mapping(manifest.get("budget"), f"{kind} manifest.budget")
    if manifest.get("budget_sha256") != PATH_B_V3_BUDGET_SHA256:
        raise PathBV3ValidationError(f"{kind} budget binding is not the sealed budget")
    if sha256_json(budget) != manifest.get("budget_sha256"):
        raise PathBV3ValidationError(f"{kind} budget hash mismatch")
    _sha(manifest.get("config_source_sha256"), f"{kind} manifest.config_source_sha256")
    result_root = _nonempty(manifest.get("results_root"), f"{kind} manifest.results_root")
    if result_root != PATH_B_V3_RESULTS_ROOTS[kind]:
        raise PathBV3ValidationError(f"{kind} results root is not physically isolated")
    _resolve_safe_root(result_root, f"{kind} results root")


def _resolve_safe_root(value: str, name: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or not value.startswith("results/"):
        raise PathBV3ValidationError(f"{name} must be a relative results/ path")
    return candidate


def _validate_t1_t2_plan(plan: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != PATH_B_V3_T1_T2_PLAN_SCHEMA:
        raise PathBV3ValidationError("T1/T2 plan schema is unsupported")
    if plan.get("experiment_id") != manifest.get("experiment_id"):
        raise PathBV3ValidationError("T1/T2 plan experiment identity mismatch")
    if plan.get("analysis_spec_sha256") != PATH_B_V3_T1_T2_SPEC_SHA256:
        raise PathBV3ValidationError("T1/T2 analysis spec binding mismatch")
    if plan.get("cohort_sha256") != PATH_B_V3_T1_T2_COHORT_SHA256:
        raise PathBV3ValidationError("T1/T2 cohort binding mismatch")
    if plan.get("claim_tier") != "exploratory" or plan.get("selection") != "all":
        raise PathBV3ValidationError("T1/T2 plan selection or claim tier drifted")
    if plan.get("ranking_eligible") is not False:
        raise PathBV3ValidationError("T1/T2 plan is ranking eligible")
    if manifest.get("retrieval_regimes") != ["cutoff_controlled"]:
        raise PathBV3ValidationError("T1/T2 manifest retrieval regime drifted")
    cells = plan.get("cells")
    if not isinstance(cells, list) or len(cells) != PATH_B_V3_T1_T2_CELL_COUNT:
        raise PathBV3ValidationError("T1/T2 plan cell count does not match the sealed plan")
    if plan.get("cell_count") != len(cells):
        raise PathBV3ValidationError("T1/T2 declared cell_count is wrong")
    orders: list[int] = []
    cell_ids: set[str] = set()
    groups: defaultdict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for index, raw_cell in enumerate(cells):
        cell = _mapping(raw_cell, f"T1/T2 cell {index}")
        cell_id = _nonempty(cell.get("cell_id"), f"T1/T2 cell {index}.cell_id")
        if cell_id in cell_ids:
            raise PathBV3ValidationError("T1/T2 plan contains duplicate cell IDs")
        cell_ids.add(cell_id)
        orders.append(_integer(cell.get("execution_order"), f"T1/T2 cell {index}.execution_order"))
        if cell.get("ranking_eligible") is not False:
            raise PathBV3ValidationError("T1/T2 cell is ranking eligible")
        if cell.get("retrieval_regime") != "cutoff_controlled":
            raise PathBV3ValidationError("T1/T2 cell does not use cutoff_controlled retrieval")
        band = cell.get("band")
        if band not in {"T1", "T2"}:
            raise PathBV3ValidationError("T1/T2 cell band is invalid")
        system_id = _nonempty(cell.get("system_id"), f"T1/T2 cell {index}.system_id")
        if system_id not in _EXPECTED_SYSTEM_IDS:
            raise PathBV3ValidationError("T1/T2 cell system is not sealed")
        trial = _integer(cell.get("trial"), f"T1/T2 cell {index}.trial", minimum=1)
        if trial > 3:
            raise PathBV3ValidationError("T1/T2 cell trial is outside the sealed schedule")
        _nonempty(cell.get("task_id"), f"T1/T2 cell {index}.task_id")
        pair_id = _nonempty(cell.get("pair_id"), f"T1/T2 cell {index}.pair_id")
        if cell.get("budget_sha256") != manifest.get("budget_sha256"):
            raise PathBV3ValidationError("T1/T2 cell budget differs from its manifest")
        _integer(cell.get("seed"), f"T1/T2 cell {index}.seed", minimum=0)
        groups[(pair_id, system_id, trial)].append(cell)
    if sorted(orders) != list(range(PATH_B_V3_T1_T2_CELL_COUNT)):
        raise PathBV3ValidationError("T1/T2 execution order is not a complete deterministic order")
    if len(groups) != PATH_B_V3_T1_T2_PAIR_COUNT * 3 * 3:
        raise PathBV3ValidationError("T1/T2 pair/system/trial groups are incomplete")
    pair_band_tasks: defaultdict[str, dict[str, str]] = defaultdict(dict)
    for (pair_id, _system_id, _trial), group in groups.items():
        if len(group) != 2 or {cell.get("band") for cell in group} != {"T1", "T2"}:
            raise PathBV3ValidationError(
                f"T1/T2 pair group is not a paired two-band group: {pair_id}"
            )
        seeds = {cell.get("seed") for cell in group}
        budgets = {cell.get("budget_sha256") for cell in group}
        if len(seeds) != 1 or len(budgets) != 1:
            raise PathBV3ValidationError(f"T1/T2 pair group has unpaired seed/budget: {pair_id}")
        for cell in group:
            band = str(cell["band"])
            task_id = str(cell["task_id"])
            prior = pair_band_tasks[pair_id].get(band)
            if prior is not None and prior != task_id:
                raise PathBV3ValidationError(f"T1/T2 pair has unstable task identity: {pair_id}")
            pair_band_tasks[pair_id][band] = task_id
    if len(pair_band_tasks) != PATH_B_V3_T1_T2_PAIR_COUNT or any(
        set(bands) != {"T1", "T2"} for bands in pair_band_tasks.values()
    ):
        raise PathBV3ValidationError("T1/T2 plan does not contain 112 complete pairs")
    task_ids = sorted({str(cell["task_id"]) for cell in cells})
    population = _mapping(manifest.get("task_population"), "T1/T2 task population")
    if len(task_ids) != population.get("task_count") or sha256_json(task_ids) != population.get(
        "task_ids_sha256"
    ):
        raise PathBV3ValidationError("T1/T2 task population hash mismatch")
    if population.get("pair_count") != PATH_B_V3_T1_T2_PAIR_COUNT:
        raise PathBV3ValidationError("T1/T2 task population pair count mismatch")
    if sha256_json(task_ids) != PATH_B_V3_TASK_IDS_SHA256:
        raise PathBV3ValidationError("T1/T2 task identity set is not the sealed 224-task set")


def _validate_b_calibration_plan(plan: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != PATH_B_V3_B_CALIBRATION_PLAN_SCHEMA:
        raise PathBV3ValidationError("B calibration plan schema is unsupported")
    if plan.get("experiment_id") != manifest.get("experiment_id"):
        raise PathBV3ValidationError("B calibration plan experiment identity mismatch")
    if plan.get("analysis_spec_sha256") != PATH_B_V3_B_SPEC_SHA256:
        raise PathBV3ValidationError("B calibration spec binding mismatch")
    if plan.get("cohort_sha256") != PATH_B_V3_G2_ROSTER_SHA256:
        raise PathBV3ValidationError("B calibration cohort binding mismatch")
    if plan.get("claim_tier") != "exploratory" or plan.get("selection") != "all":
        raise PathBV3ValidationError("B calibration selection or claim tier drifted")
    if plan.get("ranking_eligible") is not False:
        raise PathBV3ValidationError("B calibration plan is ranking eligible")
    if manifest.get("retrieval_regimes") != []:
        raise PathBV3ValidationError(
            "B calibration must not declare an executable retrieval regime"
        )
    if plan.get("available_case_count") != 0 or plan.get("available_case_task_ids") != []:
        raise PathBV3ValidationError("B calibration is not the sealed zero-available-case plan")
    if plan.get("excluded_count") != PATH_B_V3_ROSTER_COUNT:
        raise PathBV3ValidationError("B calibration exclusion count differs from G2")
    if plan.get("status") != "no_available_case_sample":
        raise PathBV3ValidationError("B calibration must explicitly record no available sample")
    if not isinstance(plan.get("disclosure"), str) or not plan["disclosure"].strip():
        raise PathBV3ValidationError("B calibration disclosure is missing")
    if plan.get("cells") != []:
        raise PathBV3ValidationError("B calibration zero-sample plan must have no cells")


def _expected_leakage_isolation() -> dict[str, str]:
    return {
        "cache_namespace": "path-b-v3-r3-unrestricted-isolated",
        "credential_namespace": "path-b-v3-r3-unrestricted-isolated",
        "cutoff_profile": "cutoff_controlled/v1",
        "cutoff_results_root": ("results/path-b-v3-leakage-probe-r3/cutoff_controlled"),
        "unrestricted_profile": "leakage_probe/v1",
        "unrestricted_results_root": (
            "results/path-b-v3-leakage-probe-r3/unrestricted_current_web"
        ),
    }


def _validate_leakage_plan(plan: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != PATH_B_V3_LEAKAGE_PLAN_SCHEMA:
        raise PathBV3ValidationError("leakage plan schema is unsupported")
    if plan.get("experiment_id") != manifest.get("experiment_id"):
        raise PathBV3ValidationError("leakage plan experiment identity mismatch")
    if plan.get("claim_tier") != "exploratory" or plan.get("selection") != "all":
        raise PathBV3ValidationError("leakage plan selection or claim tier drifted")
    if plan.get("ranking_eligible") is not False:
        raise PathBV3ValidationError("leakage plan is ranking eligible")
    if manifest.get("retrieval_regimes") != [
        "cutoff_controlled",
        "unrestricted_current_web",
    ]:
        raise PathBV3ValidationError("leakage manifest retrieval regimes drifted")
    if plan.get("results_root") != PATH_B_V3_RESULTS_ROOTS["cutoff_unrestricted"]:
        raise PathBV3ValidationError("leakage plan results root is not isolated")
    if (
        dict(_mapping(plan.get("physical_isolation"), "leakage physical isolation"))
        != _expected_leakage_isolation()
    ):
        raise PathBV3ValidationError("leakage physical isolation binding drifted")
    authorization = _mapping(plan.get("authorization"), "leakage authorization")
    if (
        authorization.get("authorizing_party") != "JY"
        or authorization.get("unrestricted_current_web") is not True
    ):
        raise PathBV3ValidationError("leakage unrestricted authorization is missing")
    audit = _mapping(plan.get("audit_requirements"), "leakage audit requirements")
    expected_audit = {
        "page_capture_timestamp_and_hash",
        "policy_identity",
        "query_receipts",
        "ranking_eligible",
        "visit_receipts",
    }
    if (
        set(audit) != expected_audit
        or any(audit.get(key) is not True for key in expected_audit - {"ranking_eligible"})
        or audit.get("ranking_eligible") is not False
    ):
        raise PathBV3ValidationError("leakage audit requirements are incomplete")
    if plan.get("task_count") != PATH_B_V3_ELIGIBLE_COUNT:
        raise PathBV3ValidationError("leakage task count differs from G2")
    if plan.get("pair_input_count") != PATH_B_V3_LEAKAGE_PAIR_INPUT_COUNT:
        raise PathBV3ValidationError("leakage pair input count differs from the sealed plan")
    if plan.get("cohort_roster_sha256") != PATH_B_V3_G2_ROSTER_SHA256:
        raise PathBV3ValidationError("leakage cohort binding mismatch")
    cells = plan.get("cells")
    if not isinstance(cells, list) or len(cells) != PATH_B_V3_LEAKAGE_CELL_COUNT:
        raise PathBV3ValidationError("leakage plan cell count does not match the sealed plan")
    groups: defaultdict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, str, int, str]] = set()
    for index, raw_cell in enumerate(cells):
        cell = _mapping(raw_cell, f"leakage cell {index}")
        task_id = _nonempty(cell.get("task_id"), f"leakage cell {index}.task_id")
        system_id = _nonempty(cell.get("system_id"), f"leakage cell {index}.system_id")
        if system_id not in _EXPECTED_SYSTEM_IDS:
            raise PathBV3ValidationError("leakage cell system is not sealed")
        trial = _integer(cell.get("trial"), f"leakage cell {index}.trial", minimum=1)
        if trial > 3:
            raise PathBV3ValidationError("leakage cell trial is outside the sealed schedule")
        regime = cell.get("retrieval_regime")
        if regime not in {"cutoff_controlled", "unrestricted_current_web"}:
            raise PathBV3ValidationError("leakage cell retrieval regime is invalid")
        if cell.get("ranking_eligible") is not False:
            raise PathBV3ValidationError("leakage cell is ranking eligible")
        expected_filter = regime == "cutoff_controlled"
        if cell.get("cutoff_filter_enabled") is not expected_filter:
            raise PathBV3ValidationError("leakage cutoff filter flag does not match regime")
        expected_policy = "cutoff_controlled/v1" if expected_filter else "leakage_probe/v1"
        if cell.get("policy_identity") != expected_policy:
            raise PathBV3ValidationError("leakage policy identity does not match regime")
        if cell.get("budget_sha256") != manifest.get("budget_sha256"):
            raise PathBV3ValidationError("leakage cell budget differs from its manifest")
        _integer(cell.get("seed"), f"leakage cell {index}.seed", minimum=0)
        identity = (task_id, system_id, trial, str(regime))
        if identity in identities:
            raise PathBV3ValidationError("leakage plan contains duplicate cell identity")
        identities.add(identity)
        groups[(task_id, system_id, trial)].append(cell)
    if len(groups) != PATH_B_V3_LEAKAGE_PAIR_INPUT_COUNT:
        raise PathBV3ValidationError("leakage plan does not contain 3645 paired inputs")
    for group in groups.values():
        if len(group) != 2 or {cell.get("retrieval_regime") for cell in group} != {
            "cutoff_controlled",
            "unrestricted_current_web",
        }:
            raise PathBV3ValidationError("leakage pair is missing one retrieval regime")
        if (
            len({cell.get("seed") for cell in group}) != 1
            or len({cell.get("budget_sha256") for cell in group}) != 1
        ):
            raise PathBV3ValidationError("leakage pair does not share seed and budget")
    task_ids = sorted({str(cell["task_id"]) for cell in cells})
    population = _mapping(manifest.get("task_population"), "leakage task population")
    if len(task_ids) != population.get("task_count") or sha256_json(task_ids) != population.get(
        "task_ids_sha256"
    ):
        raise PathBV3ValidationError("leakage task population hash mismatch")
    if sha256_json(task_ids) != PATH_B_V3_LEAKAGE_TASK_IDS_SHA256:
        raise PathBV3ValidationError("leakage task identity set is not the sealed 405-task set")


def _validate_index(
    root: Path,
    index: Mapping[str, Any],
    g2_manifest: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
    *,
    seal: PathBV3G3Seal,
) -> None:
    _verify_self_hash(index, "index_sha256", "G3 plan index")
    if index.get("index_sha256") != seal.index_sha256:
        raise PathBV3ValidationError("G3 plan index is not the sealed current-pin index")
    if index.get("schema_version") != PATH_B_V3_G3_INDEX_SCHEMA_VERSION:
        raise PathBV3ValidationError("unsupported G3 plan-index schema")
    if index.get("status") != "sealed_plans_ready":
        raise PathBV3ValidationError("G3 plan index is not sealed_plans_ready")
    if index.get("ranking_eligible") is not False:
        raise PathBV3ValidationError("G3 plan index is ranking eligible")
    if (
        index.get("dataset_revision") != PATH_B_V3_DATASET_REVISION
        or index.get("dataset_tree_sha") != PATH_B_V3_DATASET_TREE_SHA
    ):
        raise PathBV3ValidationError("G3 plan index is not bound to the 951 pin")
    if index.get("g2_freeze_manifest_sha256") != g2_manifest.get("manifest_sha256"):
        raise PathBV3ValidationError("G3 plan index is not bound to the validated G2 freeze")
    if index.get("physical_results_roots") != list(PATH_B_V3_RESULTS_ROOTS.values()):
        raise PathBV3ValidationError("G3 physical results roots drifted")
    expected_experiments = {
        "t1_t2": {
            "cell_count": PATH_B_V3_T1_T2_CELL_COUNT,
            "pair_count": PATH_B_V3_T1_T2_PAIR_COUNT,
            "execution_state": "not_started",
            "manifest_sha256": seal.manifest_sha256s["t1_t2"],
            "plan_sha256": seal.plan_sha256s["t1_t2"],
        },
        "b_calibration": {
            "available_case_count": 0,
            "execution_state": "no_available_case_sample",
            "manifest_sha256": seal.manifest_sha256s["b_calibration"],
            "plan_sha256": seal.plan_sha256s["b_calibration"],
        },
        "cutoff_unrestricted": {
            "cell_count": PATH_B_V3_LEAKAGE_CELL_COUNT,
            "execution_state": "not_started",
            "manifest_sha256": seal.manifest_sha256s["cutoff_unrestricted"],
            "plan_sha256": seal.plan_sha256s["cutoff_unrestricted"],
            "ranking_eligible": False,
        },
    }
    if dict(_mapping(index.get("experiments"), "G3 experiments")) != expected_experiments:
        raise PathBV3ValidationError("G3 experiment index entries drifted")
    for kind, (manifest_filename, plan_filename) in PATH_B_V3_MANIFEST_FILES.items():
        _resolve_inside(root, manifest_filename, f"{kind} manifest")
        _resolve_inside(root, plan_filename, f"{kind} plan")
        if manifests[kind].get("manifest_sha256") != index["experiments"][kind]["manifest_sha256"]:
            raise PathBV3ValidationError(f"G3 index manifest reference mismatch for {kind}")
        if plans[kind].get("plan_sha256") != index["experiments"][kind]["plan_sha256"]:
            raise PathBV3ValidationError(f"G3 index plan reference mismatch for {kind}")


def _validation_summary(plans: PathBV3Plans) -> dict[str, Any]:
    experiments: dict[str, dict[str, Any]] = {}
    for kind in PATH_B_V3_MANIFEST_FILES:
        manifest = plans.manifests[kind]
        plan = plans.plans[kind]
        entry: dict[str, Any] = {
            "experiment_id": manifest["experiment_id"],
            "manifest_sha256": manifest["manifest_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "results_root": manifest["results_root"],
            "ranking_eligible": False,
            "claim_tier": "exploratory",
        }
        if kind == "t1_t2":
            entry.update({"cell_count": len(plan["cells"]), "pair_count": plan["pair_count"]})
        elif kind == "b_calibration":
            entry.update(
                {
                    "available_case_count": plan["available_case_count"],
                    "execution_state": plan["status"],
                }
            )
        else:
            entry.update(
                {
                    "cell_count": len(plan["cells"]),
                    "pair_input_count": plan["pair_input_count"],
                    "execution_state": "not_started",
                }
            )
        experiments[kind] = entry
    return {
        "status": "valid",
        "schema_version": PATH_B_V3_G3_INDEX_SCHEMA_VERSION,
        "dataset_repo_id": PATH_B_V3_DATASET_REPO_ID,
        "dataset_revision": PATH_B_V3_DATASET_REVISION,
        "dataset_tree_sha": PATH_B_V3_DATASET_TREE_SHA,
        "g2_freeze_manifest_sha256": PATH_B_V3_G2_MANIFEST_SHA256,
        "index_sha256": plans.index["index_sha256"],
        "path_b_roster_count": PATH_B_V3_ROSTER_COUNT,
        "cohort_task_count": PATH_B_V3_ELIGIBLE_COUNT,
        "ranking_eligible": False,
        "physical_results_roots": list(PATH_B_V3_RESULTS_ROOTS.values()),
        "experiments": experiments,
    }


def _load_plan_set(
    root: str | Path,
    *,
    g2_root: str | Path | None = None,
    contract_path: str | Path = "configs/path-b-v2-review-contract.json",
) -> PathBV3Plans:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise PathBV3ValidationError(f"G3 plan root is not a directory: {root_path}")
    g2_path = (
        Path(g2_root).expanduser().resolve()
        if g2_root is not None
        else root_path.parent / "path-b-v3-g2-freeze-951edec-r3"
    )
    contract = Path(contract_path).expanduser().resolve()
    g2_manifest, eligible = _validate_g2(g2_path, contract)
    index = _read_json(root_path / PATH_B_V3_G3_INDEX_FILENAME, "G3 plan index")
    _assert_score_blind_plan(index, path="$.g3_index")
    _verify_self_hash(index, "index_sha256", "G3 plan index")
    seal = _seal_for_index(index)
    manifests: dict[str, Mapping[str, Any]] = {}
    plans: dict[str, Mapping[str, Any]] = {}
    for kind, (manifest_filename, plan_filename) in PATH_B_V3_MANIFEST_FILES.items():
        manifest = _read_json(root_path / manifest_filename, f"{kind} manifest")
        _assert_score_blind_plan(manifest, path=f"$.{kind}.manifest")
        _validate_common_manifest(
            manifest,
            kind=kind,
            expected_plan_filename=plan_filename,
            seal=seal,
        )
        plan_path = _resolve_inside(root_path, manifest.get("plan_filename"), f"{kind} plan")
        if plan_path.name != plan_filename:
            raise PathBV3ValidationError(f"{kind} manifest resolves to the wrong plan file")
        plan = _read_json(plan_path, f"{kind} plan")
        _verify_self_hash(plan, "plan_sha256", f"{kind} plan")
        if plan.get("plan_sha256") != seal.plan_sha256s[kind]:
            raise PathBV3ValidationError(f"{kind} plan is not the sealed current-pin plan")
        _assert_score_blind_plan(plan, path=f"$.{kind}.plan")
        if kind == "t1_t2":
            _validate_t1_t2_plan(plan, manifest)
        elif kind == "b_calibration":
            _validate_b_calibration_plan(plan, manifest)
        else:
            _validate_leakage_plan(plan, manifest)
        manifests[kind] = manifest
        plans[kind] = plan
    _validate_index(root_path, index, g2_manifest, manifests, plans, seal=seal)
    all_t1_t2_ids = {str(cell["task_id"]) for cell in plans["t1_t2"]["cells"]}
    leakage_ids = {str(cell["task_id"]) for cell in plans["cutoff_unrestricted"]["cells"]}
    if all_t1_t2_ids | leakage_ids != set(eligible) or all_t1_t2_ids - set(eligible):
        raise PathBV3ValidationError(
            "G3 task populations are not bound to the frozen eligible cohort"
        )
    plan_set = PathBV3Plans(
        root=root_path,
        g2_root=g2_path,
        index=index,
        manifests=manifests,
        plans=plans,
        eligible_task_ids=eligible,
    )
    return plan_set


def load_path_b_v3_plans(
    root: str | Path,
    *,
    g2_root: str | Path | None = None,
    contract_path: str | Path = "configs/path-b-v2-review-contract.json",
) -> PathBV3Plans:
    """Load a validated object for the sealed current-pin G3 plan set."""

    return _load_plan_set(root, g2_root=g2_root, contract_path=contract_path)


def validate_path_b_v3_plans(
    root: str | Path,
    *,
    g2_root: str | Path | None = None,
    contract_path: str | Path = "configs/path-b-v2-review-contract.json",
) -> dict[str, Any]:
    """Validate all three sealed current-pin G3 manifest/plan pairs.

    This is the replacement for ``experiment validate`` for the ``path-b-v3``
    artifacts.  It is offline and score-blind; it never loads the Hub source.
    """

    return _load_plan_set(root, g2_root=g2_root, contract_path=contract_path).summary()


def _load_validated_plans(
    value: PathBV3Plans | str | Path,
    *,
    g2_root: str | Path | None,
    contract_path: str | Path,
) -> PathBV3Plans:
    if isinstance(value, PathBV3Plans):
        return value
    # Validate on every path-based dispatch so a summary generated earlier
    # cannot authorize files changed in the meantime.
    return _load_plan_set(value, g2_root=g2_root, contract_path=contract_path)


def _task_ids_for_plans(plans: PathBV3Plans) -> tuple[str, ...]:
    t1_t2_ids = {str(cell["task_id"]) for cell in plans.plans["t1_t2"]["cells"]}
    leakage_ids = {str(cell["task_id"]) for cell in plans.plans["cutoff_unrestricted"]["cells"]}
    selected = t1_t2_ids | leakage_ids
    if selected != set(plans.eligible_task_ids):
        raise PathBV3ValidationError(
            "task loader selection is not exactly the frozen 405-task cohort"
        )
    return tuple(sorted(selected))


def _cutoff_date(row: Mapping[str, Any], task_id: str) -> str:
    seed = row.get("seed")
    candidate = seed.get("cutoff_date") if isinstance(seed, Mapping) else None
    if candidate is None:
        candidate = row.get("cutoff_date")
    value = _nonempty(candidate, f"source row {task_id}.cutoff_date")
    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError as exc:
        raise PathBV3ValidationError(f"source row {task_id} has an invalid cutoff_date") from exc


def load_path_b_v3_task_inputs(
    plans: PathBV3Plans | str | Path,
    *,
    g2_root: str | Path | None = None,
    contract_path: str | Path = "configs/path-b-v2-review-contract.json",
    cache_dir: str | Path = "data/.huggingface",
    loader: Callable[..., Any] | None = None,
) -> dict[str, TaskInput]:
    """Load only ``TaskInput`` values from the exact 951-pin 543-row source.

    The source loader may see its own score-blind rows, but this function
    immediately projects each selected row to the adapter-safe three-field
    ``TaskInput``.  No source row, outcome, or evaluator-owned field is
    returned to a backend.
    """

    validated = _load_validated_plans(plans, g2_root=g2_root, contract_path=contract_path)
    selected_ids = set(_task_ids_for_plans(validated))
    source = V3DatasetSource(
        cache_dir=Path(cache_dir),
        repo_id=PATH_B_V3_DATASET_REPO_ID,
        revision=PATH_B_V3_DATASET_REVISION,
        config_names=("a1", "a2_t", "b"),
        expected_record_count=PATH_B_V3_ROSTER_COUNT,
        offline=False,
        loader=loader,
    )
    try:
        rows = source.load_records()
    except (V3DatasetSourceError, OSError, ValueError) as exc:
        raise PathBV3ValidationError(f"951-pin Path B task load failed: {exc}") from exc
    if len(rows) != PATH_B_V3_ROSTER_COUNT:
        raise PathBV3ValidationError(
            f"951-pin Path B source returned {len(rows)} rows, expected {PATH_B_V3_ROSTER_COUNT}"
        )
    by_task: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PathBV3ValidationError(f"951-pin source row {index} is not an object")
        task_id = _nonempty(row.get("task_id"), f"951-pin source row {index}.task_id")
        if task_id in by_task:
            raise PathBV3ValidationError(f"951-pin source contains duplicate task_id {task_id}")
        by_task[task_id] = row
    missing = sorted(selected_ids - set(by_task))
    if missing:
        raise PathBV3ValidationError(
            "951-pin source is missing planned task(s): " + ",".join(missing)
        )
    result: dict[str, TaskInput] = {}
    for task_id in sorted(selected_ids):
        row = by_task[task_id]
        prompt = _nonempty(row.get("prompt"), f"951-pin source row {task_id}.prompt")
        result[task_id] = TaskInput(
            task_id=task_id,
            prompt=prompt,
            cutoff_date=_cutoff_date(row, task_id),
        )
    return result


def check_path_b_v3_credentials(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> PathBV3CredentialStatus:
    """Check required live credentials without returning or logging secrets."""

    env = os.environ if environ is None else environ
    base = Path(cwd or Path.cwd()).expanduser().resolve()

    model_present = _usable_credential_value(env.get(MODEL_API_KEY_ENV))
    direct_search = str(env.get(SEARCH_SERVICE_TOKEN_ENV, "")).strip()
    search_file_text = str(env.get(SEARCH_SERVICE_TOKEN_FILE_ENV, "")).strip()
    search_file = (
        _credential_path(search_file_text, base)
        if _usable_credential_value(search_file_text)
        else base / "data/search-service/admin.token"
    )
    search_present = _usable_credential_value(direct_search) or _nonempty_file(search_file)

    direct_unrestricted = str(env.get(UNRESTRICTED_CREDENTIALS_ENV, "")).strip()
    unrestricted_file_text = str(env.get(UNRESTRICTED_CREDENTIALS_FILE_ENV, "")).strip()
    if not unrestricted_file_text:
        unrestricted_file_text = str(env.get("BENCHMARK_UNRESTRICTED_CREDENTIALS_FILE", "")).strip()
    unrestricted_file = (
        _credential_path(unrestricted_file_text, base)
        if _usable_credential_value(unrestricted_file_text)
        else None
    )
    unrestricted_present = _usable_credential_value(direct_unrestricted) or (
        unrestricted_file is not None and _nonempty_file(unrestricted_file)
    )
    missing = tuple(
        name
        for name, present in (
            (MODEL_API_KEY_ENV, model_present),
            (SEARCH_SERVICE_TOKEN_FILE_ENV, search_present),
            (UNRESTRICTED_CREDENTIALS_FILE_ENV, unrestricted_present),
        )
        if not present
    )
    return PathBV3CredentialStatus(
        model_api_key_present=model_present,
        search_admin_token_present=search_present,
        unrestricted_probe_present=unrestricted_present,
        missing=missing,
    )


def _nonempty_file(path: Path) -> bool:
    try:
        value = path.read_bytes().strip()
        if not path.is_file() or not value:
            return False
        try:
            return _usable_credential_value(value.decode("utf-8"))
        except UnicodeDecodeError:
            return True
    except OSError:
        return False


def _usable_credential_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().casefold()
    return bool(normalized) and normalized not in _PLACEHOLDER_CREDENTIAL_VALUES


def _credential_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def require_path_b_v3_credentials(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> PathBV3CredentialStatus:
    status = check_path_b_v3_credentials(environ=environ, cwd=cwd)
    if not status.ready:
        missing = ", ".join(status.missing)
        raise PathBV3CredentialError(
            "path-b-v3 live execution requires credentials; missing "
            f"{missing}. No mock execution or score artifact will be created."
        )
    return status


def run_path_b_v3(
    plans: PathBV3Plans | str | Path,
    *,
    g2_root: str | Path | None = None,
    contract_path: str | Path = "configs/path-b-v2-review-contract.json",
    cache_dir: str | Path = "data/.huggingface",
    loader: Callable[..., Any] | None = None,
    backend: PathBV3ExecutionBackend | None = None,
    validate_only: bool = False,
    dry_dispatch: bool = False,
    environ: Mapping[str, str] | None = None,
    credential_cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Validate, credential-gate, and dispatch the sealed three-experiment set.

    With ``validate_only=True`` this is entirely offline.  A live dispatch
    requires an injected backend; refusing a missing backend is deliberate so
    unrestricted cells cannot be silently sent through the cutoff adapter.
    """

    validated = _load_validated_plans(plans, g2_root=g2_root, contract_path=contract_path)
    if validate_only and dry_dispatch:
        raise PathBV3ExecutionError("--validate-only and --dry-dispatch are mutually exclusive")
    if validate_only:
        return {**validated.summary(), "execution_state": "validated_only"}
    credentials = require_path_b_v3_credentials(environ=environ, cwd=credential_cwd)
    if backend is None:
        raise PathBV3ExecutionError(
            "path-b-v3 credentials are present, but no explicit current-web execution "
            "backend was supplied; refusing to route unrestricted cells through the legacy runner"
        )
    if dry_dispatch:
        dry_request = PathBV3ExecutionRequest(
            plans=validated,
            task_inputs={},
            credentials=credentials,
            results_roots=validated.results_roots,
        )
        dry_method = getattr(backend, "dry_dispatch", None)
        if not callable(dry_method):
            raise PathBV3ExecutionError(
                "--dry-dispatch requires an explicit backend dry_dispatch method"
            )
        output = dry_method(dry_request)
        if output is None:
            output = {}
        if not isinstance(output, Mapping):
            raise PathBV3ExecutionError("path-b-v3 dry-dispatch backend must return an object")
        if output.get("ranking_eligible") is True:
            raise PathBV3ExecutionError(
                "path-b-v3 dry-dispatch attempted to mark results ranking eligible"
            )
        return {
            "status": "dry_dispatch",
            **dict(output),
            "dataset_revision": PATH_B_V3_DATASET_REVISION,
            "dataset_tree_sha": PATH_B_V3_DATASET_TREE_SHA,
            "ranking_eligible": False,
            "results_roots": {kind: str(path) for kind, path in validated.results_roots.items()},
        }
    task_inputs = load_path_b_v3_task_inputs(
        validated,
        cache_dir=cache_dir,
        loader=loader,
        contract_path=contract_path,
    )
    request = PathBV3ExecutionRequest(
        plans=validated,
        task_inputs=task_inputs,
        credentials=credentials,
        results_roots=validated.results_roots,
    )
    output = backend(request)
    if output is None:
        output = {}
    if not isinstance(output, Mapping):
        raise PathBV3ExecutionError("path-b-v3 execution backend must return an object")
    if output.get("ranking_eligible") is True:
        raise PathBV3ExecutionError("path-b-v3 backend attempted to mark results ranking eligible")
    return {
        "status": "executed",
        **dict(output),
        "dataset_revision": PATH_B_V3_DATASET_REVISION,
        "dataset_tree_sha": PATH_B_V3_DATASET_TREE_SHA,
        "ranking_eligible": False,
        "results_roots": {kind: str(path) for kind, path in validated.results_roots.items()},
    }


__all__ = [
    "MODEL_API_KEY_ENV",
    "SEARCH_SERVICE_TOKEN_ENV",
    "SEARCH_SERVICE_TOKEN_FILE_ENV",
    "UNRESTRICTED_CREDENTIALS_ENV",
    "UNRESTRICTED_CREDENTIALS_FILE_ENV",
    "PATH_B_V3_BUDGET_SHA256",
    "PATH_B_V3_DATASET_REPO_ID",
    "PATH_B_V3_DATASET_REVISION",
    "PATH_B_V3_DATASET_TREE_SHA",
    "PATH_B_V3_ELIGIBLE_COUNT",
    "PATH_B_V3_G2_MANIFEST_SHA256",
    "PATH_B_V3_G2_ROSTER_SHA256",
    "PATH_B_V3_LEAKAGE_CELL_COUNT",
    "PATH_B_V3_RESULTS_ROOTS",
    "PATH_B_V3_ROSTER_COUNT",
    "PathBV3CredentialError",
    "PathBV3CredentialStatus",
    "PathBV3ExecutionBackend",
    "PathBV3ExecutionError",
    "PathBV3ExecutionRequest",
    "PathBV3Plans",
    "PathBV3RunnerError",
    "PathBV3ValidationError",
    "check_path_b_v3_credentials",
    "load_path_b_v3_plans",
    "load_path_b_v3_task_inputs",
    "require_path_b_v3_credentials",
    "run_path_b_v3",
    "validate_path_b_v3_plans",
]
