"""Deterministic model-only re-sealing for the current-pin Path B v3 plans.

The G2 evidence and cohort freeze are independent of the model deployment.  A
model re-seal therefore copies the already validated G3 geometry, changes only
the model binding and experiment identity, and writes a new append-only plan
root.  The operator endpoint is deliberately represented by its environment
binding rather than persisted in the sealed artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark_harness.experiment.path_b_v2 import sha256_json
from benchmark_harness.experiment.path_b_v3_runner import (
    PATH_B_V3_G3_INDEX_FILENAME,
    PATH_B_V3_G3_INDEX_SHA256,
    PATH_B_V3_MANIFEST_FILES,
    PathBV3ValidationError,
    validate_path_b_v3_plans,
)

PATH_B_V3_MINI_SEAL_ID = "path-b-v3-g3-r4-mini"
PATH_B_V3_MINI_BACKBONE_ID = "openai:gpt-4.1-mini-2025-04-14"
PATH_B_V3_MINI_SNAPSHOT_MODEL_NAME = "gpt-4.1-mini-2025-04-14"
PATH_B_V3_MINI_API_MODEL_NAME = "gpt-4.1-mini"
PATH_B_V3_MINI_CUTOFF_SOURCE = "https://developers.openai.com/api/docs/models/gpt-4.1-mini"
PATH_B_V3_MINI_CUTOFF = "2024-06-01"
PATH_B_V3_MINI_ENDPOINT_ENV = "BENCHMARK_MODEL_ENDPOINT"
PATH_B_V3_MINI_ENDPOINT_BINDING = "operator_supplied_env"
PATH_B_V3_MINI_WIRE_NAME_BINDING = "user_authorized_azure_wire_mapping/v1"
PATH_B_V3_MINI_PLAN_ROOT = "path-b-v3-g3-plans-951edec-r4-mini"

_EXPERIMENT_IDS = {
    "t1_t2": "path-b-v3-r4-mini-t1-t2",
    "b_calibration": "path-b-v3-r4-mini-b-calibration",
    "cutoff_unrestricted": "path-b-v3-r4-mini-cutoff-unrestricted",
}
_EXPECTED_MANIFEST_FILES = {
    "t1_t2": "t1-t2-manifest.json",
    "b_calibration": "b-calibration-manifest.json",
    "cutoff_unrestricted": "cutoff-unrestricted-manifest.json",
}
_EXPECTED_PLAN_FILES = {
    "t1_t2": "t1-t2-plan.json",
    "b_calibration": "b-calibration-plan.json",
    "cutoff_unrestricted": "cutoff-unrestricted-plan.json",
}


class PathBV3ModelResealError(PathBV3ValidationError):
    """Raised when a model-only re-seal cannot be produced safely."""


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PathBV3ModelResealError(f"unable to read {name}: {path}") from exc
    if not isinstance(value, dict):
        raise PathBV3ModelResealError(f"{name} must be a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _with_hash(body: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = dict(body)
    value.pop(field, None)
    value[field] = sha256_json(value)
    return value


def _model_binding() -> dict[str, str]:
    return {
        "api_model_name": PATH_B_V3_MINI_API_MODEL_NAME,
        "backbone_id": PATH_B_V3_MINI_BACKBONE_ID,
        "endpoint_binding": PATH_B_V3_MINI_ENDPOINT_BINDING,
        "endpoint_env": PATH_B_V3_MINI_ENDPOINT_ENV,
        "seal_id": PATH_B_V3_MINI_SEAL_ID,
        "snapshot_model_name": PATH_B_V3_MINI_SNAPSHOT_MODEL_NAME,
        "wire_name_binding": PATH_B_V3_MINI_WIRE_NAME_BINDING,
    }


def _assert_no_endpoint(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "endpoint":
                raise PathBV3ModelResealError(
                    "operator-supplied model endpoint leaked into sealed artifact at "
                    f"{path}.endpoint"
                )
            _assert_no_endpoint(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_endpoint(child, path=f"{path}[{index}]")


def _reseal_manifest(manifest: Mapping[str, Any], kind: str) -> dict[str, Any]:
    result = deepcopy(dict(manifest))
    result["experiment_id"] = _EXPERIMENT_IDS[kind]
    result["model_binding"] = _model_binding()
    backbone = result.get("backbone")
    if not isinstance(backbone, dict):
        raise PathBV3ModelResealError(f"{kind} manifest backbone is missing")
    backbone["backbone_id"] = PATH_B_V3_MINI_BACKBONE_ID
    backbone["cutoff_source"] = PATH_B_V3_MINI_CUTOFF_SOURCE
    backbone["training_data_cutoff"] = PATH_B_V3_MINI_CUTOFF
    settings = backbone.get("model_settings")
    if not isinstance(settings, dict):
        raise PathBV3ModelResealError(f"{kind} manifest model_settings are missing")
    settings.pop("endpoint", None)
    settings.update(
        {
            "api_model_name": PATH_B_V3_MINI_API_MODEL_NAME,
            "backbone_id": PATH_B_V3_MINI_BACKBONE_ID,
            "endpoint_binding": PATH_B_V3_MINI_ENDPOINT_BINDING,
            "endpoint_env": PATH_B_V3_MINI_ENDPOINT_ENV,
            "snapshot_model_name": PATH_B_V3_MINI_SNAPSHOT_MODEL_NAME,
            "wire_name_binding": PATH_B_V3_MINI_WIRE_NAME_BINDING,
        }
    )
    systems = result.get("systems")
    if not isinstance(systems, list):
        raise PathBV3ModelResealError(f"{kind} manifest systems are missing")
    for system in systems:
        if not isinstance(system, dict):
            raise PathBV3ModelResealError(f"{kind} manifest has a malformed system")
        system["backbone_id"] = PATH_B_V3_MINI_BACKBONE_ID
    trials = result.get("trials")
    if not isinstance(trials, dict):
        raise PathBV3ModelResealError(f"{kind} manifest trials are missing")
    trials["seed_namespace"] = f"path-b-v3-r4-mini|{PATH_B_V3_MINI_BACKBONE_ID}"
    result.pop("manifest_sha256", None)
    return result


def _reseal_plan(plan: Mapping[str, Any], kind: str) -> dict[str, Any]:
    result = deepcopy(dict(plan))
    result["experiment_id"] = _EXPERIMENT_IDS[kind]
    result.pop("plan_sha256", None)
    return result


def build_path_b_v3_mini_reseal(
    source_root: str | Path,
    output_root: str | Path,
    *,
    g2_root: str | Path | None = None,
    contract_path: str | Path = "configs/path-b-v2-review-contract.json",
) -> dict[str, Any]:
    """Build a new mini-model G3 root from an already sealed G3 root.

    The source root is validated before it is read.  The destination must not
    exist, making the operation append-only and preventing an accidental
    replacement of either the prior G3 artifacts or a previous mini re-seal.
    """

    source = Path(source_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise PathBV3ModelResealError(f"re-seal output already exists: {destination}")
    if not source.is_dir():
        raise PathBV3ModelResealError(f"re-seal source is not a directory: {source}")
    source_index = _read_json(source / PATH_B_V3_G3_INDEX_FILENAME, "source G3 index")
    if source_index.get("index_sha256") != PATH_B_V3_G3_INDEX_SHA256:
        raise PathBV3ModelResealError("re-seal source is not the sealed current G3 root")
    source_summary = validate_path_b_v3_plans(
        source,
        g2_root=g2_root,
        contract_path=contract_path,
    )

    manifests: dict[str, dict[str, Any]] = {}
    plans: dict[str, dict[str, Any]] = {}
    for kind, (manifest_filename, plan_filename) in PATH_B_V3_MANIFEST_FILES.items():
        if (
            manifest_filename != _EXPECTED_MANIFEST_FILES[kind]
            or plan_filename != _EXPECTED_PLAN_FILES[kind]
        ):
            raise PathBV3ModelResealError(f"unexpected sealed file mapping for {kind}")
        manifests[kind] = _reseal_manifest(
            _read_json(source / manifest_filename, f"{kind} source manifest"), kind
        )
        plans[kind] = _reseal_plan(_read_json(source / plan_filename, f"{kind} source plan"), kind)
        plans[kind] = _with_hash(plans[kind], "plan_sha256")
        manifests[kind]["plan_sha256"] = plans[kind]["plan_sha256"]
        manifests[kind] = _with_hash(manifests[kind], "manifest_sha256")

    index = deepcopy(source_index)
    index["seal_id"] = PATH_B_V3_MINI_SEAL_ID
    index["model_binding"] = _model_binding()
    index["experiments"] = {
        kind: {
            key: value
            for key, value in source_index["experiments"][kind].items()
            if key not in {"manifest_sha256", "plan_sha256"}
        }
        for kind in PATH_B_V3_MANIFEST_FILES
    }
    for kind in PATH_B_V3_MANIFEST_FILES:
        index["experiments"][kind]["manifest_sha256"] = manifests[kind]["manifest_sha256"]
        index["experiments"][kind]["plan_sha256"] = plans[kind]["plan_sha256"]
    index.pop("index_sha256", None)
    index = _with_hash(index, "index_sha256")

    generated = [*manifests.values(), *plans.values(), index]
    for value in generated:
        _assert_no_endpoint(value)

    destination.mkdir(parents=True)
    for kind, (manifest_filename, plan_filename) in PATH_B_V3_MANIFEST_FILES.items():
        _write_json(destination / manifest_filename, manifests[kind])
        _write_json(destination / plan_filename, plans[kind])
    _write_json(destination / PATH_B_V3_G3_INDEX_FILENAME, index)
    return {
        "status": "sealed",
        "source_index_sha256": source_summary["index_sha256"],
        "index_sha256": index["index_sha256"],
        "root": str(destination),
        "seal_id": PATH_B_V3_MINI_SEAL_ID,
        "model_binding": _model_binding(),
        "manifests": {
            kind: {
                "manifest_sha256": manifests[kind]["manifest_sha256"],
                "plan_sha256": plans[kind]["plan_sha256"],
            }
            for kind in PATH_B_V3_MANIFEST_FILES
        },
    }


__all__ = [
    "PATH_B_V3_MINI_API_MODEL_NAME",
    "PATH_B_V3_MINI_BACKBONE_ID",
    "PATH_B_V3_MINI_CUTOFF",
    "PATH_B_V3_MINI_CUTOFF_SOURCE",
    "PATH_B_V3_MINI_ENDPOINT_BINDING",
    "PATH_B_V3_MINI_ENDPOINT_ENV",
    "PATH_B_V3_MINI_PLAN_ROOT",
    "PATH_B_V3_MINI_SEAL_ID",
    "PATH_B_V3_MINI_SNAPSHOT_MODEL_NAME",
    "PATH_B_V3_MINI_WIRE_NAME_BINDING",
    "PathBV3ModelResealError",
    "build_path_b_v3_mini_reseal",
]
