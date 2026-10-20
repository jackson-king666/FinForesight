"""Strict experiment/v1 manifest parsing and identity validation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from benchmark_harness.calibration.leak_review import (
    STRONG_AGREEMENT_FLOOR,
    STRONG_KAPPA_FLOOR,
)
from benchmark_harness.experiment.data_inventory import (
    V4_DATASET_REPO_ID,
    V4_DATASET_REVISION,
    V4_DATASET_TREE_SHA,
    V4_EXPECTED_CONFIG_COUNTS,
    V4_EXPECTED_RECORD_COUNT,
)
from benchmark_harness.systems.contracts import SystemDescriptor
from benchmark_harness.temporal.contracts import parse_temporal_date


class ManifestError(ValueError):
    """Raised when a manifest is incomplete, ambiguous, or not immutable."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ManifestError(f"declared artifact is unavailable: {path}") from exc
    return digest.hexdigest()


_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REGIMES = frozenset({"closed_book", "cutoff_controlled", "unrestricted_current_web"})
_PROFILES = frozenset({"one_turn_strict", "agentic_web_search", "deep_research"})
_MANIFEST_FIELDS_V1 = frozenset(
    {
        "schema_version",
        "experiment_id",
        "experiment_as_of",
        "claim_scope",
        "dataset",
        "temporal",
        "systems",
        "retrieval_regimes",
        "trials",
        "analysis",
    }
)
_MANIFEST_FIELDS_V2 = _MANIFEST_FIELDS_V1 | frozenset(
    {
        "analysis_as_of",
        "backbone",
        "budget",
        "model",
        "metadata",
        "policy",
        "population",
        "provenance",
        "scheduler",
        "split",
    }
)
_MANIFEST_FIELDS_V4 = _MANIFEST_FIELDS_V2 | frozenset(
    {
        "cohort",
        "t3",
        "calculation_policy",
        "sample_plan",
        "hypothesis_families",
        "completion_rules",
        "source_review",
        "generator",
        "supplements",
        "ranking_requested",
        "ranking_gates",
        "license_disposition",
    }
)
_POPULATION_FIELDS = frozenset({"applicability"})
_APPLICABILITY_FIELDS = frozenset({"forecast", "controls"})
_DATASET_FIELDS = frozenset(
    {
        "repo_id",
        "revision",
        "release_label",
        "configs",
        "record_count",
        "dataset_fingerprint",
        "tree_sha",
        "tree_sha256",
        "source_manifest_sha256",
        "supplements",
    }
)
_TEMPORAL_FIELDS = frozenset(
    {
        "index_path",
        "index_sha256",
        "cutoff_registry",
        "cutoff_registry_sha256",
        "guard_days",
    }
)
_TRIAL_FIELDS = frozenset({"count", "seed", "order", "model_seed_derivation"})
_ANALYSIS_FIELDS = frozenset(
    {
        "metric_catalog",
        "metric_catalog_sha256",
        "bootstrap_resamples",
        "alpha",
        "category_weights",
        "primary_metric",
        "stats",
        "formal_utility_version",
        "reliability_bins",
    }
)
_SYSTEM_FIELDS = frozenset(
    {
        "system_id",
        "adapter",
        "config",
        "framework_id",
        "framework_revision",
        "strategy_id",
        "strategy_revision",
        "backbone_id",
        "backbone_training_cutoff",
        "cutoff_source",
        "adapter_id",
        "adapter_revision",
        "profile",
        "license_status",
        "license_files",
        "source_provenance",
        "execution_eligible",
        "ranking_eligible",
    }
)
_SYSTEM_FIELDS_V4 = _SYSTEM_FIELDS | frozenset({"gates", "license_disposition"})


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _required_sha(value: Any, field_name: str, pattern: re.Pattern[str]) -> str:
    parsed = _required_string(value, field_name)
    if not pattern.fullmatch(parsed):
        raise ManifestError(f"{field_name} must be an immutable lowercase SHA.")
    return parsed


def _required_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ManifestError(f"{field_name} must be a positive integer.")
    return value


def _strict_mapping(value: Any, field_name: str, allowed: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{field_name} must be an object.")
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise ManifestError(f"{field_name} contains unknown field(s): {', '.join(unknown)}.")
    return value


@dataclass(frozen=True, slots=True)
class SystemBindingSpec:
    """A manifest entry that can be expanded into one descriptor per regime."""

    system_id: str
    adapter: str
    config: str
    framework_id: str
    framework_revision: str
    strategy_id: str
    strategy_revision: str
    backbone_id: str
    backbone_training_cutoff: date
    cutoff_source: str
    adapter_id: str
    adapter_revision: str
    profile: str
    license_status: str = "UNKNOWN"
    license_files: tuple[str, ...] = ()
    source_provenance: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    execution_eligible: bool | None = None
    ranking_eligible: bool | None = None
    gates: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    license_disposition: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "system_id",
            "adapter",
            "config",
            "framework_id",
            "framework_revision",
            "strategy_id",
            "strategy_revision",
            "backbone_id",
            "cutoff_source",
            "adapter_id",
            "adapter_revision",
            "profile",
            "license_status",
        ):
            _required_string(getattr(self, name), name)
        if self.profile not in _PROFILES:
            raise ManifestError(f"unsupported system profile {self.profile!r}.")
        if not isinstance(self.backbone_training_cutoff, date):
            raise ManifestError("system backbone_training_cutoff must be a date.")
        if any(not isinstance(path, str) or not path.strip() for path in self.license_files):
            raise ManifestError("system license_files must contain non-empty strings.")
        if not isinstance(self.source_provenance, Mapping):
            raise ManifestError("system source_provenance must be an object.")
        if self.execution_eligible is not None and not isinstance(self.execution_eligible, bool):
            raise ManifestError("system execution_eligible must be boolean.")
        if self.ranking_eligible is not None and not isinstance(self.ranking_eligible, bool):
            raise ManifestError("system ranking_eligible must be boolean.")
        if not isinstance(self.gates, Mapping):
            raise ManifestError("system gates must be an object.")
        if self.license_disposition is not None:
            _required_string(self.license_disposition, "license_disposition")

    def descriptor(self, retrieval_regime: str) -> SystemDescriptor:
        if retrieval_regime not in _REGIMES:
            raise ManifestError(f"unsupported retrieval regime {retrieval_regime!r}.")
        return SystemDescriptor(
            system_id=self.system_id,
            framework_id=self.framework_id,
            framework_revision=self.framework_revision,
            strategy_id=self.strategy_id,
            strategy_revision=self.strategy_revision,
            backbone_id=self.backbone_id,
            backbone_training_cutoff=self.backbone_training_cutoff,
            cutoff_source=self.cutoff_source,
            adapter_id=self.adapter_id,
            adapter_revision=self.adapter_revision,
            profile=self.profile,
            retrieval_regime=retrieval_regime,
            ranking_eligible=False,
            license_status=self.license_status,
        )

    def to_dict(self, *, include_extended: bool = True) -> dict[str, Any]:
        value = {
            "system_id": self.system_id,
            "adapter": self.adapter,
            "config": self.config,
            "framework_id": self.framework_id,
            "framework_revision": self.framework_revision,
            "strategy_id": self.strategy_id,
            "strategy_revision": self.strategy_revision,
            "backbone_id": self.backbone_id,
            "backbone_training_cutoff": self.backbone_training_cutoff.isoformat(),
            "cutoff_source": self.cutoff_source,
            "adapter_id": self.adapter_id,
            "adapter_revision": self.adapter_revision,
            "profile": self.profile,
            "license_status": self.license_status,
        }
        if not include_extended:
            return value
        if self.license_files:
            value["license_files"] = list(self.license_files)
        if self.source_provenance:
            value["source_provenance"] = dict(self.source_provenance)
        if self.execution_eligible is not None:
            value["execution_eligible"] = self.execution_eligible
        if self.ranking_eligible is not None:
            value["ranking_eligible"] = self.ranking_eligible
        if self.gates:
            value["gates"] = dict(self.gates)
        if self.license_disposition is not None:
            value["license_disposition"] = self.license_disposition
        return value


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    schema_version: str
    experiment_id: str
    experiment_as_of: date
    claim_scope: str
    dataset_repo_id: str
    dataset_revision: str
    temporal_index_path: str
    temporal_index_sha256: str
    cutoff_registry_path: str
    guard_days: int
    systems: tuple[SystemBindingSpec, ...]
    retrieval_regimes: tuple[str, ...]
    trial_count: int
    seed: int
    order: str
    metric_catalog_path: str
    bootstrap_resamples: int
    alpha: float
    source_path: Path | None = None
    # v2 fields are optional so the v1 public contract remains source and
    # constructor compatible.  They are populated only by a v2 manifest.
    analysis_as_of: date | None = None
    split_path: str | None = None
    split_sha256: str | None = None
    v2_metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    # v4 keeps its additive sections in a separate mapping so legacy v1/v2
    # constructors and manifest hashes remain unchanged.
    v4_metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version not in {"experiment/v1", "experiment/v2", "experiment/v4"}:
            raise ManifestError(
                "schema_version must be experiment/v1, experiment/v2, or experiment/v4."
            )
        for name in (
            "experiment_id",
            "claim_scope",
            "dataset_repo_id",
            "dataset_revision",
            "temporal_index_path",
            "temporal_index_sha256",
            "cutoff_registry_path",
            "metric_catalog_path",
        ):
            _required_string(getattr(self, name), name)
        if not _SHA1.fullmatch(self.dataset_revision):
            raise ManifestError("dataset_revision must be an immutable 40-character SHA.")
        if not _SHA256.fullmatch(self.temporal_index_sha256):
            raise ManifestError("temporal_index_sha256 must be a 64-character SHA-256.")
        if (
            not isinstance(self.guard_days, int)
            or isinstance(self.guard_days, bool)
            or self.guard_days < 0
        ):
            raise ManifestError("guard_days must be a non-negative integer.")
        if not self.systems:
            raise ManifestError("manifest must declare at least one system.")
        ids = [system.system_id for system in self.systems]
        if len(ids) != len(set(ids)):
            raise ManifestError("manifest contains duplicate system_id values.")
        if not self.retrieval_regimes:
            raise ManifestError("manifest must declare at least one retrieval regime.")
        if len(set(self.retrieval_regimes)) != len(self.retrieval_regimes):
            raise ManifestError("manifest contains duplicate retrieval regimes.")
        if any(regime not in _REGIMES for regime in self.retrieval_regimes):
            raise ManifestError("manifest contains an unknown retrieval regime.")
        if self.order != "blocked_randomized":
            raise ManifestError("trials.order must be blocked_randomized.")
        _required_positive_int(self.trial_count, "trials.count")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ManifestError("trials.seed must be an integer.")
        _required_positive_int(self.bootstrap_resamples, "analysis.bootstrap_resamples")
        if (
            not isinstance(self.alpha, int | float)
            or isinstance(self.alpha, bool)
            or not 0 < self.alpha < 1
        ):
            raise ManifestError("analysis.alpha must be between 0 and 1.")
        if self.schema_version in {"experiment/v2", "experiment/v4"}:
            if self.analysis_as_of is None:
                raise ManifestError(f"{self.schema_version} requires analysis_as_of.")
            if not isinstance(self.analysis_as_of, date):
                raise ManifestError("analysis_as_of must be a day-precision date.")
        if self.schema_version == "experiment/v2":
            if not self.split_path or not _SHA256.fullmatch(self.split_sha256 or ""):
                raise ManifestError("experiment/v2 requires split.path and split.sha256.")
            if tuple(self.retrieval_regimes) != ("closed_book", "cutoff_controlled"):
                raise ManifestError(
                    "experiment/v2 exploratory tranche permits only closed_book and "
                    "cutoff_controlled regimes."
                )
            if len(self.systems) != 3:
                raise ManifestError("experiment/v2 exploratory tranche requires exactly 3 systems.")
            model = self.v2_metadata.get("model", self.v2_metadata.get("backbone", {}))
            if not isinstance(model, Mapping) or not str(model.get("backbone_id", "")).strip():
                raise ManifestError("experiment/v2 requires an immutable backbone_id.")
            budget = self.v2_metadata.get("budget")
            if not isinstance(budget, Mapping):
                raise ManifestError("experiment/v2 requires a finite per-cell budget.")
            if self.v2_metadata.get("ranking_eligible", False) is not False:
                raise ManifestError("experiment/v2 must remain non-ranking.")
        if self.schema_version == "experiment/v4":
            expected_systems = {"financeharness", "odr", "gpt-researcher"}
            if self.experiment_id != "hua40-full-v1":
                raise ManifestError("experiment/v4 requires experiment_id hua40-full-v1.")
            if {system.system_id for system in self.systems} != expected_systems:
                raise ManifestError(
                    "experiment/v4 requires exactly financeharness, odr, and gpt-researcher."
                )
            if tuple(self.retrieval_regimes) != (
                "closed_book",
                "cutoff_controlled",
                "unrestricted_current_web",
            ):
                raise ManifestError(
                    "experiment/v4 requires closed_book, cutoff_controlled, and "
                    "unrestricted_current_web regimes."
                )
            if self.trial_count != 3:
                raise ManifestError("experiment/v4 requires exactly three trials.")
            if not isinstance(self.v4_metadata, Mapping):
                raise ManifestError("experiment/v4 metadata must be an object.")
            dataset = self.v4_metadata.get("dataset", {})
            if not isinstance(dataset, Mapping):
                raise ManifestError("experiment/v4 dataset metadata must be an object.")
            if self.dataset_repo_id != V4_DATASET_REPO_ID:
                raise ManifestError(
                    "experiment/v4 dataset must use the sselaine27/benchmark-research repo."
                )
            if self.dataset_revision != V4_DATASET_REVISION:
                raise ManifestError(
                    "experiment/v4 dataset revision must use the pinned current revision."
                )
            if dataset.get("repo_id") != V4_DATASET_REPO_ID:
                raise ManifestError("experiment/v4 dataset.repo_id does not match the current pin.")
            if dataset.get("revision") != V4_DATASET_REVISION:
                raise ManifestError(
                    "experiment/v4 dataset.revision does not match the current pin."
                )
            if (
                dataset.get("tree_sha") != V4_DATASET_TREE_SHA
                or dataset.get("tree_sha256") != V4_DATASET_TREE_SHA
            ):
                raise ManifestError(
                    "experiment/v4 dataset tree SHA does not match the current pin."
                )
            if dataset.get("record_count") != V4_EXPECTED_RECORD_COUNT:
                raise ManifestError(
                    f"experiment/v4 dataset must declare {V4_EXPECTED_RECORD_COUNT} records."
                )
            configs = dataset.get("configs")
            if not isinstance(configs, Mapping) or set(configs) != set(V4_EXPECTED_CONFIG_COUNTS):
                raise ManifestError(
                    "experiment/v4 dataset configs must declare exactly the eight pinned configs."
                )
            for config_name, expected_count in V4_EXPECTED_CONFIG_COUNTS.items():
                config = configs.get(config_name)
                if not isinstance(config, Mapping) or config.get("count") != expected_count:
                    raise ManifestError(
                        f"experiment/v4 dataset config {config_name!r} must declare "
                        f"count {expected_count}."
                    )
            source_manifest_sha256 = dataset.get("source_manifest_sha256")
            if not isinstance(source_manifest_sha256, str) or not _SHA256.fullmatch(
                source_manifest_sha256
            ):
                raise ManifestError(
                    "experiment/v4 dataset.source_manifest_sha256 must be a SHA-256."
                )
            supplements = self.v4_metadata.get("supplements", dataset.get("supplements"))
            if supplements is None:
                supplements = []
            if not isinstance(supplements, list):
                raise ManifestError("experiment/v4 supplements must be a list.")
            for index, supplement in enumerate(supplements):
                digest = (
                    supplement.get("sha256", supplement.get("hash"))
                    if isinstance(supplement, Mapping)
                    else supplement
                )
                if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                    raise ManifestError(
                        f"experiment/v4 supplements[{index}] must carry a SHA-256 hash."
                    )
            for system in self.systems:
                if system.execution_eligible is None or system.ranking_eligible is None:
                    raise ManifestError(
                        f"experiment/v4 system {system.system_id!r} must declare "
                        "execution_eligible "
                        "and ranking_eligible."
                    )
                if not system.source_provenance:
                    raise ManifestError(
                        f"experiment/v4 system {system.system_id!r} must declare source provenance."
                    )
                if not system.license_disposition:
                    raise ManifestError(
                        f"experiment/v4 system {system.system_id!r} must declare "
                        "license disposition."
                    )
            calculation_policy = self.v4_metadata.get("calculation_policy")
            if not isinstance(calculation_policy, Mapping):
                raise ManifestError("experiment/v4 calculation_policy is required.")
            if calculation_policy.get("default") != "calc_on":
                raise ManifestError("experiment/v4 calculation_policy.default must be calc_on.")
            if set(calculation_policy.get("e_controls", ())) != {"calc_on", "calc_off"}:
                raise ManifestError(
                    "experiment/v4 calculation_policy.e_controls must contain calc_on and calc_off."
                )
            source_review = self.v4_metadata.get("source_review")
            if not isinstance(source_review, Mapping):
                raise ManifestError("experiment/v4 source_review is required.")
            if source_review.get("reviewers_required") != ["A", "B", "C"]:
                raise ManifestError(
                    "experiment/v4 source_review must require reviewers A, B, and C."
                )
            if source_review.get("reviewer_type") != "human":
                raise ManifestError("experiment/v4 source_review reviewer_type must be human.")
            if source_review.get("raw_ab_agreement_minimum") != STRONG_AGREEMENT_FLOOR:
                raise ManifestError(
                    "experiment/v4 source_review agreement floor cannot be lowered."
                )
            if source_review.get("cohen_kappa_minimum") != STRONG_KAPPA_FLOOR:
                raise ManifestError("experiment/v4 source_review kappa floor cannot be lowered.")
            if source_review.get("adjudication_cannot_replace_raw_agreement") is not True:
                raise ManifestError(
                    "experiment/v4 source_review must preserve the raw agreement gate."
                )
            for section_name, expected_schema in (
                ("cohort", "cohort-ledger/v1"),
                ("t3", "t3-registry/v1"),
            ):
                section = self.v4_metadata.get(section_name)
                if section_name == "cohort" and isinstance(section, list) and section:
                    continue
                if not isinstance(section, Mapping):
                    raise ManifestError(f"experiment/v4 {section_name} section is required.")
                raw_path = section.get("path", section.get("ledger_path"))
                if not isinstance(raw_path, str) or not raw_path.strip():
                    raise ManifestError(f"experiment/v4 {section_name} path is required.")
                declared_hash = section.get("sha256", section.get("registry_sha256"))
                if not isinstance(declared_hash, str) or not _SHA256.fullmatch(declared_hash):
                    raise ManifestError(f"experiment/v4 {section_name} hash must be SHA-256.")
                if section.get("schema") != expected_schema:
                    raise ManifestError(
                        f"experiment/v4 {section_name} schema must be {expected_schema}."
                    )
            analysis_metadata = self.v4_metadata.get("analysis", {})
            if (
                not isinstance(analysis_metadata, Mapping)
                or analysis_metadata.get("formal_utility_version") != "formal-utility/v1"
            ):
                raise ManifestError("experiment/v4 must declare formal-utility/v1.")
            weights = (
                analysis_metadata.get("category_weights")
                if isinstance(analysis_metadata, Mapping)
                else None
            )
            if weights != {"A1": 0.25, "A2": 0.25, "B": 0.25, "C": 0.25}:
                raise ManifestError(
                    "experiment/v4 category_weights must be the frozen equal A1/A2/B/C weights."
                )
            if (
                self.v4_metadata.get("ranking_requested", True) is not True
                and self.v4_metadata.get("ranking_requested", True) is not False
            ):
                raise ManifestError("experiment/v4 ranking_requested must be boolean.")
            gates = self.v4_metadata.get("ranking_gates", {})
            if not isinstance(gates, Mapping):
                raise ManifestError("experiment/v4 ranking_gates must be an object.")
            for system_id in expected_systems:
                system_gates = gates.get(system_id, {})
                if not isinstance(system_gates, Mapping):
                    raise ManifestError(f"ranking_gates.{system_id} must be an object.")
                for gate, value in system_gates.items():
                    if not isinstance(gate, str) or not isinstance(value, bool):
                        raise ManifestError("ranking gate values must be boolean.")

    @property
    def manifest_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(include_source=False),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    def system(self, system_id: str) -> SystemBindingSpec:
        for system in self.systems:
            if system.system_id == system_id:
                return system
        raise ManifestError(f"manifest has no system {system_id!r}.")

    def to_dict(self, *, include_source: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "experiment_as_of": self.experiment_as_of.isoformat(),
            "claim_scope": self.claim_scope,
            "dataset": {
                "repo_id": self.dataset_repo_id,
                "revision": self.dataset_revision,
            },
            "temporal": {
                "index_path": self.temporal_index_path,
                "index_sha256": self.temporal_index_sha256,
                "cutoff_registry": self.cutoff_registry_path,
                "guard_days": self.guard_days,
            },
            "systems": [
                system.to_dict(include_extended=self.schema_version == "experiment/v4")
                for system in self.systems
            ],
            "retrieval_regimes": list(self.retrieval_regimes),
            "trials": {
                "count": self.trial_count,
                "seed": self.seed,
                "order": self.order,
            },
            "analysis": {
                "metric_catalog": self.metric_catalog_path,
                "bootstrap_resamples": self.bootstrap_resamples,
                "alpha": self.alpha,
            },
        }
        if self.schema_version in {"experiment/v2", "experiment/v4"}:
            value["analysis_as_of"] = (
                self.analysis_as_of.isoformat() if self.analysis_as_of else None
            )
            if self.schema_version == "experiment/v2":
                value["split"] = {
                    "path": self.split_path,
                    "sha256": self.split_sha256,
                }
                if "split" in self.v2_metadata:
                    value["split"].update(
                        {
                            key: self.v2_metadata["split"][key]
                            for key in ("algorithm", "train_count", "dev_count", "test_count")
                            if key in self.v2_metadata["split"]
                        }
                    )
            for key in (
                "backbone",
                "model",
                "budget",
                "policy",
                "population",
                "provenance",
                "scheduler",
            ):
                if key in self.v2_metadata:
                    value[key] = self.v2_metadata[key]
            dataset = value["dataset"]
            for key in (
                "release_label",
                "configs",
                "record_count",
                "dataset_fingerprint",
                "tree_sha",
                "tree_sha256",
                "source_manifest_sha256",
                "supplements",
            ):
                if key in self.v2_metadata.get("dataset", {}):
                    dataset[key] = self.v2_metadata["dataset"][key]
            temporal = value["temporal"]
            if "cutoff_registry_sha256" in self.v2_metadata:
                temporal["cutoff_registry_sha256"] = self.v2_metadata["cutoff_registry_sha256"]
            trials = value["trials"]
            if "model_seed_derivation" in self.v2_metadata:
                trials["model_seed_derivation"] = self.v2_metadata["model_seed_derivation"]
            analysis = value["analysis"]
            for key in (
                "metric_catalog_sha256",
                "category_weights",
                "primary_metric",
                "stats",
                "formal_utility_version",
                "reliability_bins",
            ):
                if key in self.v2_metadata.get("analysis", {}):
                    analysis[key] = self.v2_metadata["analysis"][key]
            if "metadata" in self.v2_metadata:
                value["metadata"] = self.v2_metadata["metadata"]
            if self.schema_version == "experiment/v4":
                for key, section in self.v4_metadata.items():
                    if key in {
                        "dataset",
                        "backbone",
                        "model",
                        "budget",
                        "policy",
                        "population",
                        "provenance",
                        "scheduler",
                        "metadata",
                        "analysis",
                    }:
                        continue
                    value[key] = section
                if "ranking_requested" in self.v4_metadata:
                    value["ranking_requested"] = self.v4_metadata["ranking_requested"]
                if "ranking_gates" in self.v4_metadata:
                    value["ranking_gates"] = self.v4_metadata["ranking_gates"]
                if "license_disposition" in self.v4_metadata:
                    value["license_disposition"] = self.v4_metadata["license_disposition"]
        if include_source and self.source_path is not None:
            value["source_path"] = str(self.source_path)
        return value


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ImportError as exc:
        raise ManifestError("PyYAML is required to load experiment manifests.") from exc
    except OSError as exc:
        raise ManifestError(f"manifest is unavailable: {path}") from exc
    except Exception as exc:
        raise ManifestError(f"manifest is not valid YAML: {path}") from exc
    if not isinstance(value, Mapping):
        raise ManifestError("manifest root must be a YAML object.")
    return value


def _system_from_mapping(
    value: Any,
    index: int,
    *,
    schema_version: str = "experiment/v1",
) -> SystemBindingSpec:
    allowed = _SYSTEM_FIELDS_V4 if schema_version == "experiment/v4" else _SYSTEM_FIELDS
    value = _strict_mapping(value, f"systems[{index}]", allowed)
    required = (
        "system_id",
        "adapter",
        "config",
        "framework_id",
        "framework_revision",
        "strategy_id",
        "strategy_revision",
        "backbone_id",
        "backbone_training_cutoff",
        "cutoff_source",
        "adapter_id",
        "adapter_revision",
        "profile",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise ManifestError(
            f"systems[{index}] is missing required identity field(s): {', '.join(missing)}."
        )
    try:
        cutoff = parse_temporal_date(
            value["backbone_training_cutoff"],
            field_name=f"systems[{index}].backbone_training_cutoff",
            allow_none=False,
        )
        assert cutoff is not None
        return SystemBindingSpec(
            system_id=value["system_id"],
            adapter=value["adapter"],
            config=value["config"],
            framework_id=value["framework_id"],
            framework_revision=value["framework_revision"],
            strategy_id=value["strategy_id"],
            strategy_revision=value["strategy_revision"],
            backbone_id=value["backbone_id"],
            backbone_training_cutoff=cutoff,
            cutoff_source=value["cutoff_source"],
            adapter_id=value["adapter_id"],
            adapter_revision=value["adapter_revision"],
            profile=value["profile"],
            license_status=value.get("license_status", "UNKNOWN"),
            license_files=tuple(value.get("license_files", ())),
            source_provenance=value.get("source_provenance", {}),
            execution_eligible=value.get("execution_eligible"),
            ranking_eligible=value.get("ranking_eligible"),
            gates=value.get("gates", {}),
            license_disposition=value.get("license_disposition"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ManifestError):
            raise
        raise ManifestError(f"invalid systems[{index}] identity: {exc}") from exc


def _population_from_mapping(value: Any) -> dict[str, Any]:
    """Validate the optional regime-applicability matrix.

    ``population.applicability`` maps task populations to permitted retrieval
    regimes: ``forecast`` covers A1/A2/B/C tasks and ``controls`` covers D/E.
    Manifest-declared lists must be non-empty, regime-valid, and disjoint.
    """

    value = _strict_mapping(value, "population", _POPULATION_FIELDS)
    applicability = value.get("applicability")
    if applicability is None:
        return {}
    applicability = _strict_mapping(
        applicability, "population.applicability", _APPLICABILITY_FIELDS
    )
    for name in ("forecast", "controls"):
        items = applicability[name] if name in applicability else None
        if items is None:
            raise ManifestError(f"population.applicability must declare {name} regimes.")
        if (
            not isinstance(items, list)
            or not items
            or any(not isinstance(item, str) or item not in _REGIMES for item in items)
        ):
            raise ManifestError(f"population.applicability.{name} must be a non-empty regime list.")
        applicability[name] = list(items)
    if not set(applicability["controls"]).issubset({"closed_book"}):
        raise ManifestError("controls may only receive closed_book retrieval.")
    return {"applicability": applicability}


def load_manifest(path: str | Path) -> ExperimentManifest:
    """Parse a YAML manifest and reject absent identity/preregistration inputs."""

    manifest_path = Path(path).expanduser().resolve()
    value = _load_yaml(manifest_path)
    schema_version = value.get("schema_version")
    allowed_manifest_fields = (
        _MANIFEST_FIELDS_V4
        if schema_version == "experiment/v4"
        else _MANIFEST_FIELDS_V2
        if schema_version == "experiment/v2"
        else _MANIFEST_FIELDS_V1
    )
    value = _strict_mapping(value, "manifest", allowed_manifest_fields)
    try:
        dataset = _strict_mapping(value["dataset"], "dataset", _DATASET_FIELDS)
        temporal = _strict_mapping(value["temporal"], "temporal", _TEMPORAL_FIELDS)
        trials = _strict_mapping(value["trials"], "trials", _TRIAL_FIELDS)
        analysis = _strict_mapping(value["analysis"], "analysis", _ANALYSIS_FIELDS)
        systems = tuple(
            _system_from_mapping(item, index, schema_version=schema_version)
            for index, item in enumerate(value["systems"])
        )
        experiment_as_of = parse_temporal_date(
            value["experiment_as_of"], field_name="experiment_as_of", allow_none=False
        )
        assert experiment_as_of is not None
        v4_metadata: dict[str, Any] = {}
        if schema_version in {"experiment/v2", "experiment/v4"}:
            analysis_as_of = parse_temporal_date(
                value.get("analysis_as_of", value["experiment_as_of"]),
                field_name="analysis_as_of",
                allow_none=False,
            )
            assert analysis_as_of is not None
        else:
            analysis_as_of = None
        if schema_version == "experiment/v2":
            split = _strict_mapping(
                value.get("split"),
                "split",
                frozenset(
                    {"path", "sha256", "algorithm", "train_count", "dev_count", "test_count"}
                ),
            )
            metadata: dict[str, Any] = {
                key: value[key]
                for key in (
                    "backbone",
                    "model",
                    "budget",
                    "policy",
                    "provenance",
                    "scheduler",
                    "metadata",
                )
                if key in value
            }
            if "population" in value:
                metadata["population"] = _population_from_mapping(value["population"])
            metadata["dataset"] = {
                key: dataset[key]
                for key in ("release_label", "configs", "record_count", "dataset_fingerprint")
                if key in dataset
            }
            metadata["split"] = dict(split)
            if "cutoff_registry_sha256" in temporal:
                metadata["cutoff_registry_sha256"] = temporal["cutoff_registry_sha256"]
            if "model_seed_derivation" in trials:
                metadata["model_seed_derivation"] = trials["model_seed_derivation"]
            metadata["analysis"] = {
                key: analysis[key]
                for key in (
                    "metric_catalog_sha256",
                    "category_weights",
                    "primary_metric",
                    "stats",
                    "formal_utility_version",
                    "reliability_bins",
                )
                if key in analysis
            }
            v4_metadata = {}
        elif schema_version == "experiment/v4":
            split = None
            metadata = {
                key: value[key]
                for key in (
                    "backbone",
                    "model",
                    "budget",
                    "policy",
                    "provenance",
                    "scheduler",
                    "metadata",
                )
                if key in value
            }
            if "population" in value:
                metadata["population"] = _population_from_mapping(value["population"])
            metadata["dataset"] = dict(dataset)
            metadata["analysis"] = {
                key: analysis[key]
                for key in (
                    "metric_catalog_sha256",
                    "category_weights",
                    "primary_metric",
                    "stats",
                    "formal_utility_version",
                    "reliability_bins",
                )
                if key in analysis
            }
            if "cutoff_registry_sha256" in temporal:
                metadata["cutoff_registry_sha256"] = temporal["cutoff_registry_sha256"]
            if "model_seed_derivation" in trials:
                metadata["model_seed_derivation"] = trials["model_seed_derivation"]
            # Preserve the raw v4 sections for identity hashing and for the
            # planner/executor.  Strict top-level validation happened above;
            # section-specific validators below reject malformed sections.
            for key in (
                "cohort",
                "t3",
                "calculation_policy",
                "sample_plan",
                "hypothesis_families",
                "completion_rules",
                "source_review",
                "generator",
                "supplements",
                "ranking_requested",
                "ranking_gates",
                "license_disposition",
            ):
                if key in value:
                    v4_metadata[key] = value[key]
            v4_metadata.update({key: metadata[key] for key in metadata})
            # Keep the legacy metadata view populated as well: executor and
            # model helpers can therefore share the v2 access path without
            # weakening v4's separate schema identity.
            metadata = dict(metadata)
        else:
            analysis_as_of = None
            split = None
            metadata = {}
        manifest = ExperimentManifest(
            schema_version=value["schema_version"],
            experiment_id=value["experiment_id"],
            experiment_as_of=experiment_as_of,
            claim_scope=value["claim_scope"],
            dataset_repo_id=dataset["repo_id"],
            dataset_revision=dataset["revision"],
            temporal_index_path=temporal["index_path"],
            temporal_index_sha256=temporal["index_sha256"],
            cutoff_registry_path=temporal["cutoff_registry"],
            guard_days=temporal["guard_days"],
            systems=systems,
            retrieval_regimes=tuple(value["retrieval_regimes"]),
            trial_count=trials["count"],
            seed=trials["seed"],
            order=trials["order"],
            metric_catalog_path=analysis["metric_catalog"],
            bootstrap_resamples=analysis["bootstrap_resamples"],
            alpha=analysis["alpha"],
            source_path=manifest_path,
            analysis_as_of=analysis_as_of,
            split_path=split["path"] if split is not None else None,
            split_sha256=split["sha256"] if split is not None else None,
            v2_metadata=metadata,
            v4_metadata=v4_metadata,
        )
    except KeyError as exc:
        raise ManifestError(f"manifest is missing required field {exc.args[0]!r}.") from exc
    except TypeError as exc:
        raise ManifestError(
            "manifest sections must be objects/lists with the declared schema."
        ) from exc
    return manifest


def validate_manifest_artifacts(manifest: ExperimentManifest) -> dict[str, Any]:
    """Verify every immutable v2 input named by a manifest.

    Manifest parsing intentionally remains usable for inspection and v1
    compatibility.  The explicit ``experiment validate`` gate calls this
    function before planning or execution so a changed temporal, split,
    cutoff, or metric-catalog input cannot be mistaken for the declared
    experiment.  The catalog hash is the parsed-object identity from
    ``MetricCatalog.hash``; the other hashes attest canonical file bytes.
    """

    if not isinstance(manifest, ExperimentManifest):
        raise ManifestError("validate_manifest_artifacts requires an ExperimentManifest")
    base = manifest.source_path.parent if manifest.source_path is not None else Path.cwd()

    model_identity: dict[str, str | None] | None = None
    if manifest.schema_version in {"experiment/v2", "experiment/v4"}:
        # Keep the strict mini transport/scientific identity check in the live
        # module while invoking it lazily here to avoid an import cycle during
        # manifest construction.  Generic v2 manifests remain compatible.
        from benchmark_harness.experiment.live_model import declared_model_identity

        model_identity = declared_model_identity(manifest)

    def resolve(raw: str) -> Path:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else (base / path).resolve()

    temporal_path = resolve(manifest.temporal_index_path)
    temporal_sha = _sha256_file(temporal_path)
    if temporal_sha != manifest.temporal_index_sha256:
        raise ManifestError(
            f"temporal index hash mismatch: expected {manifest.temporal_index_sha256}, "
            f"observed {temporal_sha}"
        )

    from benchmark_harness.temporal import load_temporal_index

    temporal_index = load_temporal_index(temporal_path)
    if manifest.schema_version == "experiment/v2" and len(temporal_index.rows) != 339:
        raise ManifestError(
            f"experiment/v2 temporal index must contain 339 rows, got {len(temporal_index.rows)}"
        )

    result: dict[str, Any] = {
        "temporal_index_path": str(temporal_path),
        "temporal_index_sha256": temporal_sha,
        "temporal_index_rows": len(temporal_index.rows),
    }
    if manifest.schema_version == "experiment/v4":
        metadata = manifest.v4_metadata

        def declared_path(section_name: str) -> tuple[Path | None, str | None]:
            section = metadata.get(section_name)
            if not isinstance(section, Mapping):
                return None, None
            raw_path = section.get("path", section.get("ledger_path"))
            if not isinstance(raw_path, str) or not raw_path.strip():
                return None, None
            declared_hash = section.get("sha256", section.get("registry_sha256"))
            return resolve(raw_path), declared_hash if isinstance(declared_hash, str) else None

        cohort_path, cohort_hash = declared_path("cohort")
        if cohort_path is not None:
            observed = _sha256_file(cohort_path)
            if cohort_hash is not None and observed != cohort_hash:
                raise ManifestError("experiment/v4 cohort ledger hash mismatch")
            result["cohort_path"] = str(cohort_path)
            result["cohort_sha256"] = observed
        t3_path, t3_hash = declared_path("t3")
        if t3_path is not None:
            observed = _sha256_file(t3_path)
            if t3_hash is not None and observed != t3_hash:
                raise ManifestError("experiment/v4 T3 registry hash mismatch")
            result["t3_registry_path"] = str(t3_path)
            result["t3_registry_sha256"] = observed
        declared_cutoff_sha = metadata.get("cutoff_registry_sha256")
        if not isinstance(declared_cutoff_sha, str) or not _SHA256.fullmatch(declared_cutoff_sha):
            raise ManifestError("experiment/v4 cutoff registry hash is missing")
        cutoff_path = resolve(manifest.cutoff_registry_path)
        cutoff_sha = _sha256_file(cutoff_path)
        if cutoff_sha != declared_cutoff_sha:
            raise ManifestError("experiment/v4 cutoff registry hash mismatch")
        from benchmark_harness.temporal import load_cutoff_registry

        cutoff_registry = load_cutoff_registry(cutoff_path)
        for system in manifest.systems:
            cutoff = cutoff_registry.for_backbone(system.backbone_id)
            if not cutoff.analysis_usable:
                raise ManifestError(f"backbone cutoff for {system.system_id} is not day-precision")
            if system.backbone_training_cutoff != cutoff.training_data_cutoff:
                raise ManifestError(
                    f"system {system.system_id!r} training cutoff does not match cutoff registry"
                )
            if model_identity is not None and system.backbone_id != model_identity["backbone_id"]:
                raise ManifestError(
                    f"system {system.system_id!r} backbone does not match model identity"
                )
        analysis_metadata = metadata.get("analysis", {})
        declared_catalog_sha = (
            analysis_metadata.get("metric_catalog_sha256")
            if isinstance(analysis_metadata, Mapping)
            else None
        )
        if isinstance(declared_catalog_sha, str):
            catalog_path = resolve(manifest.metric_catalog_path)
            from benchmark_harness.analysis.metrics.catalog import load_metric_catalog

            catalog = load_metric_catalog(catalog_path)
            if catalog.hash != declared_catalog_sha:
                raise ManifestError("experiment/v4 metric catalog hash mismatch")
            result["metric_catalog_path"] = str(catalog_path)
            result["metric_catalog_sha256"] = catalog.hash
        result["cutoff_registry_path"] = str(cutoff_path)
        result["cutoff_registry_sha256"] = cutoff_sha
        return result
    if manifest.schema_version != "experiment/v2":
        return result

    if manifest.split_path is None or manifest.split_sha256 is None:
        raise ManifestError("experiment/v2 split identity is missing")
    split_path = resolve(manifest.split_path)
    split_sha = _sha256_file(split_path)
    if split_sha != manifest.split_sha256:
        raise ManifestError(
            f"split artifact hash mismatch: expected {manifest.split_sha256}, observed {split_sha}"
        )
    split_rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            split_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("task_id"), str):
                raise ValueError(f"row {line_number} must contain task_id")
            split_rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ManifestError(f"split artifact is invalid: {split_path}") from exc
    task_ids = [row["task_id"] for row in split_rows]
    if task_ids != sorted(task_ids) or len(task_ids) != len(set(task_ids)):
        raise ManifestError("split artifact must be unique and sorted by task_id")
    if len(task_ids) != 339 or set(task_ids) != set(temporal_index.rows):
        raise ManifestError("split and temporal artifacts do not have identical 339-task coverage")

    cutoff_path = resolve(manifest.cutoff_registry_path)
    declared_cutoff_sha = manifest.v2_metadata.get("cutoff_registry_sha256")
    if not isinstance(declared_cutoff_sha, str) or len(declared_cutoff_sha) != 64:
        raise ManifestError("experiment/v2 cutoff registry hash is missing")
    cutoff_sha = _sha256_file(cutoff_path)
    if cutoff_sha != declared_cutoff_sha:
        raise ManifestError(
            f"cutoff registry hash mismatch: expected {declared_cutoff_sha}, observed {cutoff_sha}"
        )
    from benchmark_harness.temporal import load_cutoff_registry

    registry = load_cutoff_registry(cutoff_path)
    for system in manifest.systems:
        cutoff = registry.for_backbone(system.backbone_id)
        if not cutoff.analysis_usable:
            raise ManifestError(f"backbone cutoff for {system.system_id} is not day-precision")
        if model_identity is not None:
            if system.backbone_id != model_identity["backbone_id"]:
                raise ManifestError(
                    f"system {system.system_id!r} backbone does not match model identity"
                )
            if system.backbone_training_cutoff != cutoff.training_data_cutoff:
                raise ManifestError(
                    f"system {system.system_id!r} training cutoff does not match cutoff registry"
                )

    catalog_path = resolve(manifest.metric_catalog_path)
    from benchmark_harness.analysis.metrics.catalog import load_metric_catalog

    catalog = load_metric_catalog(catalog_path)
    declared_catalog_sha = manifest.v2_metadata.get("analysis", {}).get("metric_catalog_sha256")
    if not isinstance(declared_catalog_sha, str) or len(declared_catalog_sha) != 64:
        raise ManifestError("experiment/v2 metric catalog hash is missing")
    if catalog.hash != declared_catalog_sha:
        raise ManifestError(
            "metric catalog hash mismatch: "
            f"expected {declared_catalog_sha}, observed {catalog.hash}"
        )

    result.update(
        {
            "split_path": str(split_path),
            "split_sha256": split_sha,
            "split_rows": len(split_rows),
            "cutoff_registry_path": str(cutoff_path),
            "cutoff_registry_sha256": cutoff_sha,
            "metric_catalog_path": str(catalog_path),
            "metric_catalog_sha256": catalog.hash,
            "metric_catalog_schema_version": catalog.schema_version,
        }
    )
    return result
