"""Score-blind, immutable experiment/v3 design and analysis contracts.

The v3 path is deliberately additive.  It materializes only task metadata and
pre-registered design decisions; it never loads a system score, result, or
prediction.  Legacy experiment/v1 and experiment/v2 manifests continue to use
their existing 339-row contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from types import MappingProxyType
from typing import Any

V3_SCHEMA_VERSION = "experiment/v3"
V3_DATASET_REPO_ID = "sselaine27/benchmark-research"
V3_DATASET_REVISION = "941370f228b9bc03c1b8bd9547c098c42c06be23"
# The Hub tree identity is a Git tree SHA-1; the source manifest is SHA-256.
V3_DATASET_TREE_SHA = "f020732fa58fd8c44e8c2a2afbb31d7866e4bf64"
V3_DATASET_TREE_SHA256 = V3_DATASET_TREE_SHA  # import-compatible descriptive alias
V3_DATASET_MANIFEST_SHA256 = "b2a14c444dfc700b2415981f08aa2ddd1240d2fcb59c780499e2f312458381ba"
V3_RELEASE_LABEL = "v0.9"
V3_EXPECTED_RECORD_COUNT = 1_119
V3_EXPECTED_UNIQUE_TASK_COUNT = 1_119
CURRENT_DATA_INELIGIBLE = "CURRENT_DATA_INELIGIBLE"
EXPLORATORY_ONLY = "EXPLORATORY_ONLY"

V3_EXPECTED_ELIGIBLE_COUNTS = {
    "T1": 33,
    "T2": 136,
    "A1-T1": 16,
    "B-EARNINGS-T1": 17,
    "B-EARNINGS-T2": 46,
    "B-MACRO-T2": 13,
    "C-T2": 77,
}
V3_EXPECTED_B_LEDGER = {"eligible": 76, "excluded": 62, "total": 138}

RELIABILITY_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
RELIABILITY_BIN_COUNT = 5
WILSON_Z_95 = 1.959963984540054

SCORE_BLIND_FIELDS = frozenset(
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
        "metrics",
        "evaluation",
        "evaluated",
        "parsed_output",
    }
)
_REDACTED_SOURCE_FIELDS = frozenset(
    {"ground_truth", "expected_output", "gold", "actual", "outcome", "label", "target"}
)
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ExperimentV3Error(ValueError):
    """Raised when a v3 lock, cohort, or statistic is invalid."""


class ScoreBlindError(ExperimentV3Error):
    """Raised when score/result/prediction material crosses the v3 boundary."""


class LockError(ExperimentV3Error):
    """Raised when an immutable lock or its inputs drift."""


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible values deterministically for identity hashes."""

    def encode(value: Any) -> Any:
        if isinstance(value, date | datetime):
            return value.isoformat()
        raise TypeError(f"unsupported value for canonical JSON: {type(value).__name__}")

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=encode,
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentV3Error("value is not canonically JSON serializable") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _normal_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _is_score_blind_field(value: Any) -> bool:
    normalized = _normal_key(value)
    if normalized in SCORE_BLIND_FIELDS:
        return True
    return any(
        token in normalized
        for token in ("score", "scores", "result", "results", "prediction", "predictions")
    )


def assert_score_blind(value: Any, *, path: str = "$") -> None:
    """Reject score/result/prediction keys without inspecting their values.

    Source rows may contain ground-truth/expected-output fields because those
    belong to the frozen evaluator boundary.  The v3 cohort builder redacts
    those fields and never traverses them.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = _normal_key(key)
            if normalized in _REDACTED_SOURCE_FIELDS:
                continue
            if _is_score_blind_field(key):
                raise ScoreBlindError(f"score/result/prediction field is forbidden at {path}.{key}")
            assert_score_blind(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, child in enumerate(value):
            assert_score_blind(child, path=f"{path}[{index}]")


def assert_score_blind_path(path: str | Path) -> Path:
    """Reject paths whose name advertises a score/result/prediction artifact."""

    candidate = Path(path).expanduser().resolve()
    if any(_is_score_blind_field(part) for part in candidate.parts):
        raise ScoreBlindError(
            f"score/result/prediction path is outside the v3 input boundary: {path}"
        )
    return candidate


@dataclass
class ScoreBlindGate:
    """One-way gate used by callers to make score access fail closed."""

    locked: bool = False

    def freeze(self) -> None:
        self.locked = True

    def read_score(self, path: str | Path) -> None:
        if not self.locked:
            raise ScoreBlindError("score access requires a frozen design lock")
        raise ScoreBlindError(f"score access is forbidden after the design lock: {path}")

    read_result = read_score
    read_prediction = read_score


def _require_nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentV3Error(f"{name} must be a non-empty string")
    return value.strip()


def _require_sha(value: Any, name: str, pattern: re.Pattern[str] = _SHA256) -> str:
    parsed = _require_nonempty(value, name)
    if not pattern.fullmatch(parsed):
        raise ExperimentV3Error(f"{name} must be a lowercase immutable SHA")
    return parsed


def _safe_row(row: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ExperimentV3Error(f"dataset row {index} must be an object")
    assert_score_blind(row, path=f"$[{index}]")
    _require_nonempty(row.get("task_id"), f"dataset row {index}.task_id")
    return dict(row)


def _redacted_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the only row view allowed to enter v3 hashes or artifacts."""

    return _redact_source_value(row)


def _redact_source_value(value: Any) -> Any:
    """Recursively remove evaluator-owned source fields without traversing them."""

    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key in value:
            if _normal_key(key) in _REDACTED_SOURCE_FIELDS:
                continue
            redacted[str(key)] = _redact_source_value(value[key])
        return redacted
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_redact_source_value(child) for child in value]
    return value


def _sorted_rows(records: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    rows = tuple(_safe_row(row, index=index) for index, row in enumerate(records))
    task_ids = [row["task_id"] for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ExperimentV3Error("dataset contains duplicate task_id values")
    return tuple(sorted(rows, key=lambda row: row["task_id"]))


def _mapping(row: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = row.get(key)
        if isinstance(value, Mapping):
            return _redact_source_value(value)
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, Mapping):
                return _redact_source_value(value)
    return {}


def _field(row: Mapping[str, Any], *keys: str) -> Any:
    metadata = row.get("metadata")
    evidence = row.get("evidence")
    covariates = row.get("covariates")
    matching = row.get("matching")
    containers = [row]
    for candidate in (metadata, evidence, covariates, matching):
        if isinstance(candidate, Mapping):
            containers.append(candidate)
    for key in keys:
        for container in containers:
            if key in container:
                return container[key]
    return None


def _field_view(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    return _redact_source_value(
        {key: _field(row, key) for key in keys if _field(row, key) is not None}
    )


def _config_key(row: Mapping[str, Any]) -> str:
    category = str(row.get("category", "")).upper()
    variant = row.get("variant")
    if category == "A2" and variant:
        return f"A2-{str(variant).upper()}"
    subtype = _field(row, "event_subtype", "subtype", "b_subtype") or variant
    if category == "B" and subtype:
        return f"B-{str(subtype).upper()}"
    return category


@dataclass(frozen=True, slots=True)
class DatasetLock:
    """Hash-only identity of a score-blind dataset population."""

    schema_version: str
    repo_id: str
    revision: str
    tree_sha256: str
    source_manifest_sha256: str
    release_label: str
    record_count: int
    unique_task_count: int
    config_counts: Mapping[str, int]
    task_ids_sha256: str
    dataset_sha256: str
    evidence_sha256: str
    covariates_sha256: str
    matching_fields_sha256: str
    seed_sha256: str
    population_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "experiment-v3-dataset-lock/v1":
            raise LockError("unsupported dataset lock schema")
        _require_nonempty(self.repo_id, "dataset lock.repo_id")
        _require_sha(self.revision, "dataset lock.revision", _SHA1)
        _require_sha(self.tree_sha256, "dataset lock.tree_sha", _SHA1)
        for name in (
            "source_manifest_sha256",
            "task_ids_sha256",
            "dataset_sha256",
            "evidence_sha256",
            "covariates_sha256",
            "matching_fields_sha256",
            "seed_sha256",
            "population_manifest_sha256",
        ):
            _require_sha(getattr(self, name), f"dataset lock.{name}")
        if (
            not isinstance(self.record_count, int)
            or isinstance(self.record_count, bool)
            or self.record_count < 1
        ):
            raise LockError("dataset lock.record_count must be positive")
        if (
            not isinstance(self.unique_task_count, int)
            or isinstance(self.unique_task_count, bool)
            or self.unique_task_count != self.record_count
        ):
            raise LockError("dataset lock requires one unique task per record")
        if not isinstance(self.config_counts, Mapping):
            raise LockError("dataset lock.config_counts must be an object")
        if any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for key, value in self.config_counts.items()
        ):
            raise LockError("dataset lock.config_counts contains an invalid count")
        if sum(self.config_counts.values()) != self.record_count:
            raise LockError("dataset lock.config_counts do not sum to record_count")
        object.__setattr__(
            self,
            "config_counts",
            MappingProxyType(dict(sorted(self.config_counts.items()))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "tree_sha256": self.tree_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "release_label": self.release_label,
            "record_count": self.record_count,
            "unique_task_count": self.unique_task_count,
            "config_counts": dict(sorted(self.config_counts.items())),
            "task_ids_sha256": self.task_ids_sha256,
            "dataset_sha256": self.dataset_sha256,
            "evidence_sha256": self.evidence_sha256,
            "covariates_sha256": self.covariates_sha256,
            "matching_fields_sha256": self.matching_fields_sha256,
            "seed_sha256": self.seed_sha256,
            "population_manifest_sha256": self.population_manifest_sha256,
        }

    @property
    def lock_sha256(self) -> str:
        return sha256_json(self.to_dict())

    @property
    def dataset_lock_sha256(self) -> str:
        return self.lock_sha256

    def verify(self, records: Iterable[Mapping[str, Any]]) -> None:
        observed = build_dataset_lock(
            records,
            repo_id=self.repo_id,
            revision=self.revision,
            tree_sha256=self.tree_sha256,
            source_manifest_sha256=self.source_manifest_sha256,
            release_label=self.release_label,
            expected_record_count=self.record_count,
            expected_unique_task_count=self.unique_task_count,
        )
        if observed.to_dict() != self.to_dict():
            raise LockError("dataset lock drift detected")


def build_dataset_lock(
    records: Iterable[Mapping[str, Any]],
    *,
    repo_id: str = V3_DATASET_REPO_ID,
    revision: str = V3_DATASET_REVISION,
    tree_sha256: str = V3_DATASET_TREE_SHA256,
    source_manifest_sha256: str = V3_DATASET_MANIFEST_SHA256,
    release_label: str = V3_RELEASE_LABEL,
    expected_record_count: int | None = V3_EXPECTED_RECORD_COUNT,
    expected_unique_task_count: int | None = V3_EXPECTED_UNIQUE_TASK_COUNT,
    expected_count: int | None = None,
) -> DatasetLock:
    """Build a deterministic lock from non-score dataset fields only."""

    if expected_count is not None:
        if expected_record_count not in {None, V3_EXPECTED_RECORD_COUNT}:
            raise LockError("pass only one of expected_count and expected_record_count")
        expected_record_count = expected_count
        if expected_unique_task_count == V3_EXPECTED_UNIQUE_TASK_COUNT:
            expected_unique_task_count = expected_count
    elif (
        expected_record_count is not None
        and expected_record_count != V3_EXPECTED_RECORD_COUNT
        and expected_unique_task_count == V3_EXPECTED_UNIQUE_TASK_COUNT
    ):
        expected_unique_task_count = expected_record_count
    elif (
        expected_record_count is None
        and expected_unique_task_count == V3_EXPECTED_UNIQUE_TASK_COUNT
    ):
        expected_unique_task_count = None
    rows = _sorted_rows(records)
    if not rows:
        raise LockError("cannot lock an empty dataset")
    if expected_record_count is not None and len(rows) != expected_record_count:
        raise LockError(
            f"dataset record count mismatch: expected {expected_record_count}, found {len(rows)}"
        )
    if expected_unique_task_count is not None and len(rows) != expected_unique_task_count:
        raise LockError(
            "dataset unique task count mismatch: "
            f"expected {expected_unique_task_count}, found {len(rows)}"
        )
    redacted = [_redacted_row(row) for row in rows]
    task_ids = [row["task_id"] for row in rows]
    evidence = [
        {
            "task_id": row["task_id"],
            "evidence": _field_view(
                row,
                (
                    "evidence",
                    "source_url",
                    "url",
                    "source_hash",
                    "hash",
                    "source_tier",
                    "verified",
                ),
            ),
        }
        for row in rows
    ]
    covariates = [
        {"task_id": row["task_id"], "covariates": _mapping(row, "covariates")} for row in rows
    ]
    matching = [
        {
            "task_id": row["task_id"],
            "matching": _field_view(
                row,
                (
                    "category",
                    "variant",
                    "market",
                    "metric",
                    "metric_output_family",
                    "template_identity",
                    "entity",
                    "horizon",
                    "issuer",
                    "event_subtype",
                    "industry",
                    "stock_universe_hash",
                    "signal",
                    "point_in_time",
                    "track",
                    "frequency",
                    "source_tier",
                ),
            ),
        }
        for row in rows
    ]
    seeds = [
        {"task_id": row["task_id"], "seed": _redact_source_value(row.get("seed"))} for row in rows
    ]
    config_counts = Counter(_config_key(row) for row in rows)
    dataset_payload = {
        "repo_id": repo_id,
        "revision": revision,
        "release_label": release_label,
        "rows": redacted,
    }
    population_payload = {
        "record_count": len(rows),
        "task_ids": task_ids,
        "dataset_sha256": sha256_json(redacted),
    }
    return DatasetLock(
        schema_version="experiment-v3-dataset-lock/v1",
        repo_id=_require_nonempty(repo_id, "repo_id"),
        revision=_require_sha(revision, "revision", _SHA1),
        tree_sha256=_require_sha(tree_sha256, "tree_sha256", _SHA1),
        source_manifest_sha256=_require_sha(source_manifest_sha256, "source_manifest_sha256"),
        release_label=_require_nonempty(release_label, "release_label"),
        record_count=len(rows),
        unique_task_count=len(task_ids),
        config_counts=dict(sorted(config_counts.items())),
        task_ids_sha256=sha256_json(task_ids),
        dataset_sha256=sha256_json(dataset_payload),
        evidence_sha256=sha256_json(evidence),
        covariates_sha256=sha256_json(covariates),
        matching_fields_sha256=sha256_json(matching),
        seed_sha256=sha256_json(seeds),
        population_manifest_sha256=sha256_json(population_payload),
    )


@dataclass(frozen=True, slots=True)
class MatchingPolicy:
    """Pre-score matching/weighting rules frozen before any score path."""

    exact_strata: tuple[str, ...] = (
        "category",
        "variant",
        "market",
        "metric_output_family",
        "template_identity",
    )
    category_specific_fields: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "A1": ("entity", "horizon"),
            "A2": ("industry", "stock_universe_hash", "signal", "point_in_time", "source_tier"),
            "B": ("event_subtype", "issuer"),
            "C": ("entity", "metric", "track", "frequency", "source_tier"),
        }
    )
    propensity_model: str = "l2_logistic"
    standardized_numeric: bool = True
    one_hot_categorical: bool = True
    main_effects_only: bool = True
    regularization_c: float = 1.0
    imputation: str = "none"
    overlap_weights: Mapping[str, str] = field(
        default_factory=lambda: {"T1": "1-e(X)", "T2": "e(X)"}
    )
    propensity_bounds: tuple[float, float] = (0.05, 0.95)
    max_abs_smd: float = 0.10
    min_ess: float = 20.0

    def __post_init__(self) -> None:
        expected_category_specific_fields = {
            "A1": ("entity", "horizon"),
            "A2": ("industry", "stock_universe_hash", "signal", "point_in_time", "source_tier"),
            "B": ("event_subtype", "issuer"),
            "C": ("entity", "metric", "track", "frequency", "source_tier"),
        }
        if self.exact_strata != (
            "category",
            "variant",
            "market",
            "metric_output_family",
            "template_identity",
        ):
            raise LockError("v3 exact matching strata are not mutable")
        if {
            key: tuple(value) for key, value in self.category_specific_fields.items()
        } != expected_category_specific_fields:
            raise LockError("v3 category-specific matching fields are not mutable")
        if self.propensity_model != "l2_logistic":
            raise LockError("v3 requires l2 logistic propensity")
        if (
            not self.standardized_numeric
            or not self.one_hot_categorical
            or not self.main_effects_only
        ):
            raise LockError("v3 propensity feature construction is not mutable")
        if self.regularization_c != 1.0 or self.imputation != "none":
            raise LockError("v3 propensity regularization/imputation is not mutable")
        if dict(self.overlap_weights) != {"T1": "1-e(X)", "T2": "e(X)"}:
            raise LockError("v3 overlap weights are not mutable")
        if self.propensity_bounds != (0.05, 0.95) or self.max_abs_smd != 0.10:
            raise LockError("v3 overlap/balance bounds are not mutable")
        if self.min_ess != 20.0:
            raise LockError("v3 ESS threshold is not mutable")

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_strata": list(self.exact_strata),
            "category_specific_fields": {
                key: list(value) for key, value in sorted(self.category_specific_fields.items())
            },
            "propensity_model": self.propensity_model,
            "standardized_numeric": self.standardized_numeric,
            "one_hot_categorical": self.one_hot_categorical,
            "main_effects_only": self.main_effects_only,
            "regularization_c": self.regularization_c,
            "imputation": self.imputation,
            "overlap_weights": dict(sorted(self.overlap_weights.items())),
            "propensity_bounds": list(self.propensity_bounds),
            "max_abs_smd": self.max_abs_smd,
            "min_ess": self.min_ess,
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class EstimandPolicy:
    name: str = "t1_minus_t2_utility"
    formula: str = "E_w[U|T1] - E_w[U|T2]"
    positive_direction: str = "T1 higher"
    unit: str = "paper-level"

    def __post_init__(self) -> None:
        if (self.name, self.formula, self.positive_direction, self.unit) != (
            "t1_minus_t2_utility",
            "E_w[U|T1] - E_w[U|T2]",
            "T1 higher",
            "paper-level",
        ):
            raise LockError("v3 estimand is not mutable")

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "formula": self.formula,
            "positive_direction": self.positive_direction,
            "unit": self.unit,
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class MetricPolicy:
    brier_formula: str = "mean((p-y)^2)"
    bss_formula: str = "1-BS/0.25"
    primary_baseline: float = 0.5
    secondary_baseline: str = "leave_one_out_band_prevalence"
    reliability_bins: tuple[tuple[float, float, bool], ...] = (
        (0.0, 0.2, False),
        (0.2, 0.4, False),
        (0.4, 0.6, False),
        (0.6, 0.8, False),
        (0.8, 1.0, True),
    )
    invalid_prediction_policy: str = "operational_brier_1_and_valid_output_rate"
    valid_only_secondary: bool = True
    multi_trial_policy: str = "per_attempt_then_arithmetic_mean_within_system_task"
    reliability_task_policy: str = "arithmetic_mean_valid_p_per_task_zero_valid_reported"
    sparse_bin_threshold: int = 10
    secondary_binning_sensitivity: str = "equal_frequency_tie_break_only"
    bootstrap_resamples: int = 10_000
    bootstrap_confidence: float = 0.95
    bootstrap_unit: str = "paired_event_cluster"
    bootstrap_seed: str = "SHA256(dataset_pin | analysis_version)"
    multiple_testing: str = "holm_family"

    def __post_init__(self) -> None:
        if (
            self.brier_formula != "mean((p-y)^2)"
            or self.bss_formula != "1-BS/0.25"
            or self.primary_baseline != 0.5
            or self.secondary_baseline != "leave_one_out_band_prevalence"
            or self.invalid_prediction_policy != "operational_brier_1_and_valid_output_rate"
            or self.valid_only_secondary is not True
            or self.multi_trial_policy != "per_attempt_then_arithmetic_mean_within_system_task"
            or self.reliability_task_policy
            != "arithmetic_mean_valid_p_per_task_zero_valid_reported"
            or self.bootstrap_resamples != 10_000
            or self.bootstrap_confidence != 0.95
            or self.bootstrap_unit != "paired_event_cluster"
            or self.bootstrap_seed != "SHA256(dataset_pin | analysis_version)"
            or self.multiple_testing != "holm_family"
        ):
            raise LockError("v3 metric policy is not mutable")
        if self.sparse_bin_threshold != 10:
            raise LockError("v3 sparse reliability threshold is not mutable")
        if self.secondary_binning_sensitivity != "equal_frequency_tie_break_only":
            raise LockError("v3 reliability sensitivity is not mutable")
        expected_bins = tuple(
            (lower, upper, index == RELIABILITY_BIN_COUNT - 1)
            for index, (lower, upper) in enumerate(
                zip(RELIABILITY_BIN_EDGES[:-1], RELIABILITY_BIN_EDGES[1:], strict=True)
            )
        )
        if self.reliability_bins != expected_bins:
            raise LockError("v3 reliability bins are not mutable")

    def to_dict(self) -> dict[str, Any]:
        return {
            "brier_formula": self.brier_formula,
            "bss_formula": self.bss_formula,
            "primary_baseline": self.primary_baseline,
            "secondary_baseline": self.secondary_baseline,
            "reliability_bins": [
                {"lower": lower, "upper": upper, "upper_inclusive": inclusive}
                for lower, upper, inclusive in self.reliability_bins
            ],
            "invalid_prediction_policy": self.invalid_prediction_policy,
            "valid_only_secondary": self.valid_only_secondary,
            "multi_trial_policy": self.multi_trial_policy,
            "reliability_task_policy": self.reliability_task_policy,
            "sparse_bin_threshold": self.sparse_bin_threshold,
            "secondary_binning_sensitivity": self.secondary_binning_sensitivity,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_confidence": self.bootstrap_confidence,
            "bootstrap_unit": self.bootstrap_unit,
            "bootstrap_seed": self.bootstrap_seed,
            "multiple_testing": self.multiple_testing,
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class DesignLock:
    """One-way design identity binding all score-blind analysis decisions."""

    schema_version: str
    analysis_version: str
    dataset_lock_sha256: str
    population_manifest_sha256: str
    matching: Mapping[str, Any]
    estimand: Mapping[str, Any]
    metrics: Mapping[str, Any]
    states: Mapping[str, str]
    seed_namespace: str
    component_hashes: Mapping[str, str]
    design_lock_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "experiment-v3-design-lock/v1":
            raise LockError("unsupported design lock schema")
        _require_nonempty(self.analysis_version, "design lock.analysis_version")
        _require_sha(self.dataset_lock_sha256, "design lock.dataset_lock_sha256")
        _require_sha(self.population_manifest_sha256, "design lock.population_manifest_sha256")
        _require_nonempty(self.seed_namespace, "design lock.seed_namespace")
        if not isinstance(self.matching, Mapping):
            raise LockError("design lock.matching must be an object")
        if not isinstance(self.estimand, Mapping):
            raise LockError("design lock.estimand must be an object")
        if not isinstance(self.metrics, Mapping):
            raise LockError("design lock.metrics must be an object")
        if not isinstance(self.states, Mapping) or not isinstance(self.component_hashes, Mapping):
            raise LockError("design lock states/component_hashes must be objects")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.states.items()
        ):
            raise LockError("design lock states must map strings to strings")
        for key, value in self.component_hashes.items():
            if not isinstance(key, str) or not _SHA256.fullmatch(str(value)):
                raise LockError(f"design lock component hash {key!r} is invalid")
        if self.design_lock_sha256 != sha256_json(self.to_dict(include_hash=False)):
            raise LockError("design lock hash does not match its contents")
        object.__setattr__(self, "matching", MappingProxyType(dict(self.matching)))
        object.__setattr__(self, "estimand", MappingProxyType(dict(self.estimand)))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "states", MappingProxyType(dict(sorted(self.states.items()))))
        object.__setattr__(
            self,
            "component_hashes",
            MappingProxyType(dict(sorted(self.component_hashes.items()))),
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "analysis_version": self.analysis_version,
            "dataset_lock_sha256": self.dataset_lock_sha256,
            "population_manifest_sha256": self.population_manifest_sha256,
            "matching": dict(self.matching),
            "estimand": dict(self.estimand),
            "metrics": dict(self.metrics),
            "states": dict(sorted(self.states.items())),
            "seed_namespace": self.seed_namespace,
            "component_hashes": dict(sorted(self.component_hashes.items())),
        }
        if include_hash:
            value["design_lock_sha256"] = self.design_lock_sha256
        return value

    def verify(self, dataset_lock: DatasetLock, *, policy: V3Policy | None = None) -> None:
        if self.design_lock_sha256 != sha256_json(self.to_dict(include_hash=False)):
            raise LockError("design lock hash drift detected")
        if self.dataset_lock_sha256 != dataset_lock.lock_sha256:
            raise LockError("design lock dataset hash mismatch")
        if self.population_manifest_sha256 != dataset_lock.population_manifest_sha256:
            raise LockError("design lock population manifest hash mismatch")
        core_components = {
            "dataset",
            "population_manifest",
            "matching",
            "estimand",
            "metrics",
            "bins",
            "baseline",
            "seed",
        }
        observed = freeze_design_lock(
            dataset_lock,
            analysis_version=self.analysis_version,
            seed_namespace=self.seed_namespace,
            policy=policy or V3Policy(),
            states=self.states,
            artifact_hashes={
                key: value
                for key, value in self.component_hashes.items()
                if key not in core_components
            },
        )
        if observed.to_dict() != self.to_dict():
            raise LockError("design policy/hash drift detected")


@dataclass(frozen=True, slots=True)
class V3Policy:
    matching: MatchingPolicy = field(default_factory=MatchingPolicy)
    estimand: EstimandPolicy = field(default_factory=EstimandPolicy)
    metrics: MetricPolicy = field(default_factory=MetricPolicy)
    score_selection: str = "forbidden"
    live_unrestricted_authorized: bool = False
    main_ranking_regime: str = "cutoff_controlled"

    def __post_init__(self) -> None:
        if self.score_selection != "forbidden":
            raise LockError("score-based selection is forbidden")
        if self.live_unrestricted_authorized is not False:
            raise LockError("unrestricted current web must remain unauthorized")
        if self.main_ranking_regime != "cutoff_controlled":
            raise LockError("main ranking regime must be cutoff_controlled")

    def to_dict(self) -> dict[str, Any]:
        return {
            "matching": self.matching.to_dict(),
            "estimand": self.estimand.to_dict(),
            "metrics": self.metrics.to_dict(),
            "score_selection": self.score_selection,
            "live_unrestricted_authorized": self.live_unrestricted_authorized,
            "main_ranking_regime": self.main_ranking_regime,
        }


def freeze_design_lock(
    dataset_lock: DatasetLock,
    *,
    analysis_version: str = "experiment-v3-r1",
    seed_namespace: str = "v3",
    policy: V3Policy | None = None,
    states: Mapping[str, str] | None = None,
    artifact_hashes: Mapping[str, str] | None = None,
) -> DesignLock:
    """Freeze policy/hash identities before any score/result/prediction path."""

    selected_policy = policy or V3Policy()
    selected_states = {
        "t1_t2_effect": CURRENT_DATA_INELIGIBLE,
        "b_calibration": EXPLORATORY_ONLY,
        "unrestricted_current_web": EXPLORATORY_ONLY,
    }
    if states is not None:
        selected_states.update(states)
    expected_state_keys = {
        "t1_t2_effect",
        "b_calibration",
        "unrestricted_current_web",
    }
    if set(selected_states) != expected_state_keys:
        raise LockError("v3 design lock states are incomplete or unknown")
    if selected_states["t1_t2_effect"] not in {CURRENT_DATA_INELIGIBLE, "ELIGIBLE"}:
        raise LockError("v3 T1/T2 state is invalid")
    if selected_states["b_calibration"] != EXPLORATORY_ONLY:
        raise LockError("v3 B calibration must remain exploratory-only")
    if selected_states["unrestricted_current_web"] != EXPLORATORY_ONLY:
        raise LockError("unrestricted current web must remain exploratory-only")
    component_hashes = {
        "dataset": dataset_lock.lock_sha256,
        "population_manifest": dataset_lock.population_manifest_sha256,
        "matching": selected_policy.matching.sha256,
        "estimand": selected_policy.estimand.sha256,
        "metrics": selected_policy.metrics.sha256,
        "bins": sha256_json(selected_policy.metrics.to_dict()["reliability_bins"]),
        "baseline": sha256_json(
            {
                "primary": selected_policy.metrics.primary_baseline,
                "secondary": selected_policy.metrics.secondary_baseline,
            }
        ),
        "seed": sha256_text(seed_namespace),
    }
    if artifact_hashes:
        for key, value in artifact_hashes.items():
            component_hashes[str(key)] = _require_sha(value, f"artifact hash {key}")
    provisional = {
        "schema_version": "experiment-v3-design-lock/v1",
        "analysis_version": _require_nonempty(analysis_version, "analysis_version"),
        "dataset_lock_sha256": dataset_lock.lock_sha256,
        "population_manifest_sha256": dataset_lock.population_manifest_sha256,
        "matching": selected_policy.matching.to_dict(),
        "estimand": selected_policy.estimand.to_dict(),
        "metrics": selected_policy.metrics.to_dict(),
        "states": dict(sorted(selected_states.items())),
        "seed_namespace": _require_nonempty(seed_namespace, "seed_namespace"),
        "component_hashes": dict(sorted(component_hashes.items())),
    }
    return DesignLock(**provisional, design_lock_sha256=sha256_json(provisional))


def build_preregistration(
    dataset_lock: DatasetLock,
    design_lock: DesignLock,
    *,
    policy: V3Policy | None = None,
) -> dict[str, Any]:
    selected_policy = policy or V3Policy()
    return {
        "schema_version": "experiment-v3-preregistration/v1",
        "analysis_version": design_lock.analysis_version,
        "dataset_lock_sha256": dataset_lock.lock_sha256,
        "population_manifest_sha256": dataset_lock.population_manifest_sha256,
        "design_lock_sha256": design_lock.design_lock_sha256,
        "dataset": {
            "repo_id": dataset_lock.repo_id,
            "revision": dataset_lock.revision,
            "tree_sha256": dataset_lock.tree_sha256,
            "source_manifest_sha256": dataset_lock.source_manifest_sha256,
            "release_label": dataset_lock.release_label,
        },
        "estimand": selected_policy.estimand.to_dict(),
        "matching": selected_policy.matching.to_dict(),
        "metrics": selected_policy.metrics.to_dict(),
        "states": dict(sorted(design_lock.states.items())),
        "score_selection": selected_policy.score_selection,
        "live_unrestricted_authorized": selected_policy.live_unrestricted_authorized,
    }


def _explicit_eligibility(row: Mapping[str, Any]) -> tuple[bool | None, tuple[str, ...]]:
    candidate = row.get("eligibility")
    # Mapping form is the only supported explicit contract.  Keep a boolean
    # visible when it is supplied at row-level instead of accidentally
    # replacing it with a metadata value; the shorthand is then rejected
    # explicitly by the eligibility ledger below.
    if not isinstance(candidate, Mapping | bool):
        metadata = row.get("metadata")
        candidate = metadata.get("eligibility") if isinstance(metadata, Mapping) else None
    if isinstance(candidate, bool):
        reasons = ["eligibility_boolean_shorthand_unsupported"]
        if not candidate:
            reasons.append("explicitly_ineligible")
        return candidate, tuple(reasons)
    if isinstance(candidate, Mapping):
        eligible = candidate.get("eligible")
        reasons = candidate.get("reasons", candidate.get("exclusions", ()))
        if isinstance(eligible, bool):
            if isinstance(reasons, str):
                reasons = (reasons,)
            if not isinstance(reasons, Iterable):
                reasons = ()
            return eligible, tuple(str(reason) for reason in reasons if str(reason).strip())
    return None, ()


def _eligibility_reasons(row: Mapping[str, Any]) -> tuple[str, ...]:
    explicit, explicit_reasons = _explicit_eligibility(row)
    reasons: list[str] = []
    if _field(row, "official_temporal_eligible") is not True:
        reasons.append("official_temporal_eligible_required")
    if _field(row, "review_status") != "reviewed":
        reasons.append("review_status_not_reviewed")
    outcome_source = _field(row, "outcome_available_at_source")
    if not isinstance(outcome_source, str) or not outcome_source.startswith("observed_"):
        reasons.append("outcome_available_at_source_not_observed")
    if row.get("status") != "ready":
        reasons.append("status_not_ready")
    tier = _field(row, "source_tier", "evidence_tier")
    if tier is not None and str(tier).casefold() not in {"official", "reviewed", "observed"}:
        reasons.append("source_not_official_reviewed_or_observed")
    verified = _field(row, "evidence_verified", "verified")
    if verified is False:
        reasons.append("evidence_not_verified")
    modeled = _field(row, "modeled", "is_modeled")
    if modeled is True:
        reasons.append("modeled_source")
    pit = _field(row, "point_in_time", "is_pit", "pit")
    if pit is False:
        reasons.append("not_point_in_time")
    lookup = _field(row, "lookup_status", "lookup")
    if isinstance(lookup, str) and lookup.casefold() in {"failed", "error", "missing"}:
        reasons.append("lookup_failed")
    url = _field(row, "source_url", "evidence_url", "url", "canonical_url")
    source_hash = _field(row, "source_hash", "evidence_hash", "hash")
    if not isinstance(url, str) or not url.strip():
        reasons.append("missing_source_url")
    if not isinstance(source_hash, str) or not source_hash.strip():
        reasons.append("missing_source_hash")
    category = str(row.get("category", "")).upper()
    matching = _field(row, "stratum", "matching_key", "template_identity")
    if not matching:
        required = ["market", "metric_output_family", "template_identity"]
        if category in {"A1", "A2", "B", "C"}:
            required.extend(V3Policy().matching.category_specific_fields.get(category, ()))
        if any(_field(row, name) is None for name in required):
            reasons.append("missing_key_match_fields")
    if explicit is False and not explicit_reasons:
        reasons.append("explicitly_ineligible")
    reasons.extend(explicit_reasons)
    return tuple(dict.fromkeys(reasons))


def _stratum(row: Mapping[str, Any], policy: MatchingPolicy) -> str | None:
    direct = _field(row, "stratum", "matching_key", "exact_stratum")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    category = str(row.get("category", "")).upper()
    values: list[str] = []
    fields = list(policy.exact_strata) + list(policy.category_specific_fields.get(category, ()))
    for name in fields:
        value = _field(row, name)
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        values.append(f"{name}={canonical_json(value)}")
    return "|".join(values)


def _covariate_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate = row.get("covariates")
    if not isinstance(candidate, Mapping):
        metadata = row.get("metadata")
        candidate = metadata.get("covariates") if isinstance(metadata, Mapping) else None
    if isinstance(candidate, Mapping):
        return _redact_source_value({str(key): value for key, value in sorted(candidate.items())})
    return {}


def _valid_covariates(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, tuple[str, ...]]:
    if not rows:
        return False, ()
    keys = sorted({key for row in rows for key in _covariate_dict(row)})
    if not keys:
        return False, ()
    missing = tuple(key for key in keys if any(key not in _covariate_dict(row) for row in rows))
    return not missing, tuple(keys)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + inverse)
    inverse = math.exp(max(value, -700.0))
    return inverse / (1.0 + inverse)


def _fit_l2_propensity(group: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Fit the declared score-blind L2 main-effect propensity model."""

    if not group:
        return {}
    covariate_maps = [_covariate_dict(row) for row in group]
    keys = sorted({key for values in covariate_maps for key in values})
    if not keys or any(key not in values for values in covariate_maps for key in keys):
        return {}
    feature_values: list[list[float]] = []
    for key in keys:
        values = [values[key] for values in covariate_maps]
        numeric_values = [_numeric(value) for value in values]
        if all(value is not None for value in numeric_values):
            numeric = [float(value) for value in numeric_values if value is not None]
            center = mean(numeric)
            deviation = math.sqrt(mean((value - center) ** 2 for value in numeric))
            feature_values.append(
                [(value - center) / deviation if deviation else 0.0 for value in numeric]
            )
        else:
            categories = sorted({str(value) for value in values})
            feature_values.extend(
                [[float(str(value) == category) for value in values] for category in categories]
            )
    design = [
        [1.0] + [features[index] for features in feature_values] for index in range(len(group))
    ]
    targets = [int(str(row.get("time_band", "")).upper() == "T2") for row in group]
    parameters = [0.0] * len(design[0])
    learning_rate = 0.25 / max(1.0, len(feature_values))
    for _ in range(400):
        gradient = [0.0] * len(parameters)
        for vector, target in zip(design, targets, strict=True):
            linear = sum(weight * value for weight, value in zip(parameters, vector, strict=True))
            error = _sigmoid(linear) - target
            for index, value in enumerate(vector):
                gradient[index] += error * value
        scale = 1.0 / len(group)
        for index in range(1, len(parameters)):
            gradient[index] += parameters[index]
        updated = [
            parameter - learning_rate * scale * value
            for parameter, value in zip(parameters, gradient, strict=True)
        ]
        if max(abs(left - right) for left, right in zip(parameters, updated, strict=True)) < 1e-10:
            parameters = updated
            break
        parameters = updated
    return {
        str(row["task_id"]): _sigmoid(
            sum(weight * value for weight, value in zip(parameters, vector, strict=True))
        )
        for row, vector in zip(group, design, strict=True)
    }


def _propensity(group: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    explicit: dict[str, float] = {}
    for row in group:
        value = _field(row, "propensity", "propensity_score")
        numeric = _numeric(value)
        if numeric is not None:
            explicit[str(row["task_id"])] = numeric
    if len(explicit) == len(group):
        return explicit
    fitted = _fit_l2_propensity(group)
    if len(fitted) == len(group):
        return fitted
    n_t1 = sum(str(row.get("time_band", "")).upper() == "T1" for row in group)
    n_t2 = sum(str(row.get("time_band", "")).upper() == "T2" for row in group)
    if not n_t1 or not n_t2:
        return {}
    # A saturated exact-stratum estimate is the deterministic fallback for a
    # score-blind audit.  The production fit is still declared as L2 logistic
    # in the lock; this fallback never consumes outcomes.
    value = n_t2 / (n_t1 + n_t2)
    return {str(row["task_id"]): value for row in group}


def _weighted_ess(weights: Sequence[float]) -> float:
    total = sum(weights)
    denominator = sum(weight * weight for weight in weights)
    return total * total / denominator if denominator else 0.0


def _standardized_mean_difference(
    rows: Sequence[Mapping[str, Any]], propensities: Mapping[str, float], key: str
) -> float | None:
    raw_values = [_covariate_dict(row).get(key) for row in rows]
    numeric = all(_numeric(value) is not None for value in raw_values)
    categories = [None] if numeric else sorted({str(value) for value in raw_values})
    smds: list[float] = []
    for category in categories:
        values: list[tuple[str, float, float]] = []
        for row in rows:
            raw = _covariate_dict(row).get(key)
            value = _numeric(raw) if numeric else float(str(raw) == category)
            if value is None:
                return None
            propensity = propensities.get(str(row["task_id"]))
            if propensity is None:
                return None
            band = str(row.get("time_band", "")).upper()
            weight = (1.0 - propensity) if band == "T1" else propensity
            values.append((band, value, weight))
        t1 = [(value, weight) for band, value, weight in values if band == "T1"]
        t2 = [(value, weight) for band, value, weight in values if band == "T2"]
        if not t1 or not t2:
            return None
        mean_t1 = sum(value * weight for value, weight in t1) / sum(weight for _, weight in t1)
        mean_t2 = sum(value * weight for value, weight in t2) / sum(weight for _, weight in t2)
        var_t1 = sum(weight * (value - mean_t1) ** 2 for value, weight in t1) / sum(
            weight for _, weight in t1
        )
        var_t2 = sum(weight * (value - mean_t2) ** 2 for value, weight in t2) / sum(
            weight for _, weight in t2
        )
        pooled = math.sqrt((var_t1 + var_t2) / 2.0)
        difference = abs(mean_t1 - mean_t2)
        if pooled == 0:
            if difference == 0:
                smds.append(0.0)
            else:
                smds.append(math.inf)
        else:
            smds.append(difference / pooled)
    return max(smds, default=0.0)


def build_common_support(
    rows: Sequence[Mapping[str, Any]], *, policy: MatchingPolicy | None = None
) -> dict[str, Any]:
    """Compute score-blind exact-stratum support, overlap weights, and balance gates."""

    selected_policy = policy or MatchingPolicy()
    eligible = [row for row in rows if str(row.get("time_band", "")).upper() in {"T1", "T2"}]
    by_stratum: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    missing_stratum: list[str] = []
    for row in eligible:
        stratum = _stratum(row, selected_policy)
        if stratum is None:
            missing_stratum.append(str(row["task_id"]))
        else:
            by_stratum[stratum].append(row)
    shared = {
        key: group
        for key, group in sorted(by_stratum.items())
        if {str(row.get("time_band", "")).upper() for row in group} == {"T1", "T2"}
    }
    common_rows = [row for group in shared.values() for row in group]
    propensities: dict[str, float] = {}
    for group in shared.values():
        propensities.update(_propensity(group))
    lower, upper = selected_policy.propensity_bounds
    out_of_bounds = sorted(
        task_id for task_id, propensity in propensities.items() if not lower <= propensity <= upper
    )
    supported_rows = [
        row
        for row in common_rows
        if str(row["task_id"]) in propensities
        and lower <= propensities[str(row["task_id"])] <= upper
    ]
    candidate_covariate_keys = sorted(
        {key for row in supported_rows for key in _covariate_dict(row)}
    )
    missing_covariate_task_ids = sorted(
        str(row["task_id"])
        for row in supported_rows
        if not candidate_covariate_keys
        or any(key not in _covariate_dict(row) for key in candidate_covariate_keys)
    )
    usable_rows = [
        row for row in supported_rows if str(row["task_id"]) not in missing_covariate_task_ids
    ]
    covariates_ok, covariate_keys = _valid_covariates(supported_rows)
    smd_values = {
        key: _standardized_mean_difference(usable_rows, propensities, key) for key in covariate_keys
    }
    smd_values = {key: value for key, value in smd_values.items() if value is not None}
    weights_t1 = [
        1.0 - propensities[str(row["task_id"])]
        for row in usable_rows
        if str(row.get("time_band", "")).upper() == "T1" and str(row["task_id"]) in propensities
    ]
    weights_t2 = [
        propensities[str(row["task_id"])]
        for row in usable_rows
        if str(row.get("time_band", "")).upper() == "T2" and str(row["task_id"]) in propensities
    ]
    ess = {"T1": _weighted_ess(weights_t1), "T2": _weighted_ess(weights_t2)}
    reasons: list[str] = []
    if not shared:
        reasons.append("no_common_exact_strata")
    if missing_stratum:
        reasons.append("missing_key_match_fields")
    if out_of_bounds:
        reasons.append("propensity_outside_[0.05,0.95]")
    if not covariates_ok:
        reasons.append("missing_covariates_no_imputation")
    if smd_values and max(smd_values.values()) > selected_policy.max_abs_smd:
        reasons.append("absolute_smd_gt_0.10")
    if any(value < selected_policy.min_ess for value in ess.values()):
        reasons.append("ess_lt_20")
    ranking_eligible = bool(shared) and not reasons
    return {
        "schema_version": "experiment-v3-common-support/v1",
        "estimand_population": "T1_vs_T2",
        "exact_strata": sorted(shared),
        "candidate_task_ids": sorted(str(row["task_id"]) for row in common_rows),
        "common_support_task_ids": sorted(str(row["task_id"]) for row in usable_rows),
        "missing_stratum_task_ids": sorted(missing_stratum),
        "missing_covariate_task_ids": missing_covariate_task_ids,
        "excluded_task_ids": sorted(set(out_of_bounds).union(missing_covariate_task_ids)),
        "propensity_scores": dict(sorted(propensities.items())),
        "overlap_weight_formula": {"T1": "1-e(X)", "T2": "e(X)"},
        "propensity_bounds": {"lower": lower, "upper": upper},
        "out_of_bounds_task_ids": out_of_bounds,
        "covariate_keys": covariate_keys,
        "standardized_mean_differences": dict(sorted(smd_values.items())),
        "ess": ess,
        "balance_gate": {
            "max_abs_smd": selected_policy.max_abs_smd,
            "min_ess": selected_policy.min_ess,
            "passed": covariates_ok
            and not any(reason in {"absolute_smd_gt_0.10", "ess_lt_20"} for reason in reasons),
        },
        "ranking_eligible": ranking_eligible,
        "reasons": reasons,
    }


@dataclass(frozen=True, slots=True)
class CohortAudit:
    """Score-blind v3 materialization and its two population gates."""

    dataset_lock: DatasetLock
    design_lock: DesignLock
    eligibility_counts: Mapping[str, Any]
    exclusions: tuple[Mapping[str, Any], ...]
    covariates: tuple[Mapping[str, Any], ...]
    common_support: Mapping[str, Any]
    preregistration: Mapping[str, Any]
    status: str
    b_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "b_status": self.b_status,
            "dataset_lock_sha256": self.dataset_lock.lock_sha256,
            "design_lock_sha256": self.design_lock.design_lock_sha256,
            "eligibility_counts": dict(self.eligibility_counts),
            "common_support": dict(self.common_support),
        }


def build_cohort(
    records: Iterable[Mapping[str, Any]],
    *,
    dataset_lock: DatasetLock | None = None,
    policy: V3Policy | None = None,
    expected_record_count: int | None = V3_EXPECTED_RECORD_COUNT,
    expected_eligible_counts: Mapping[str, int] | None = None,
    expected_b_ledger: Mapping[str, int] | None = None,
    analysis_version: str = "experiment-v3-r1",
    seed_namespace: str = "v3",
) -> CohortAudit:
    """Build eligibility/support artifacts without consuming outcomes or scores."""

    rows = _sorted_rows(records)
    if dataset_lock is None:
        dataset_lock = build_dataset_lock(rows, expected_record_count=expected_record_count)
    else:
        dataset_lock.verify(rows)
    selected_policy = policy or V3Policy()
    # Freeze the policy identity before touching any cohort/B-population
    # material.  The final lock below adds the hashes of the append-only audit
    # artifacts to this same policy identity.
    freeze_design_lock(
        dataset_lock,
        analysis_version=analysis_version,
        seed_namespace=seed_namespace,
        policy=selected_policy,
    )
    gate = ScoreBlindGate()
    gate.freeze()
    exclusions: list[Mapping[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    raw_counts: Counter[str] = Counter()
    eligibility_counts: Counter[str] = Counter()
    for row in rows:
        task_id = str(row["task_id"])
        category = str(row.get("category", "")).upper()
        band = str(row.get("time_band", "")).upper()
        if band in {"T1", "T2"}:
            raw_counts[band] += 1
            raw_counts[f"{category}-{band}"] += 1
            config_key = _config_key(row)
            if config_key != category:
                raw_counts[f"{config_key}-{band}"] += 1
        reasons: list[str] = []
        if row.get("status") != "ready":
            reasons.append("status_not_ready")
        if band not in {"T1", "T2"}:
            reasons.append("not_t1_t2")
        if category in {"D", "E"}:
            reasons.append("control_category")
        reasons.extend(_eligibility_reasons(row))
        reasons = list(dict.fromkeys(reasons))
        stratum = _stratum(row, selected_policy.matching)
        if not reasons and stratum is None:
            reasons.append("missing_key_match_fields")
        if reasons:
            exclusions.append(
                {
                    "task_id": task_id,
                    "category": category,
                    "time_band": band,
                    "reasons": sorted(reasons),
                }
            )
            continue
        eligibility_counts[band] += 1
        eligibility_counts[f"{category}-{band}"] += 1
        config_key = _config_key(row)
        if config_key != category:
            eligibility_counts[f"{config_key}-{band}"] += 1
        cohort_rows.append(
            {
                "task_id": task_id,
                "category": category,
                "variant": row.get("variant"),
                "time_band": band,
                "stratum": stratum,
                "covariates": _covariate_dict(row),
                "propensity": _field(row, "propensity", "propensity_score"),
                "issuer": _field(row, "issuer"),
                "entity": _field(row, "entity"),
                "event_subtype": _field(row, "event_subtype", "subtype", "b_subtype"),
            }
        )
    common_support = build_common_support(cohort_rows, policy=selected_policy.matching)
    support_passed = bool(common_support["ranking_eligible"])
    expected_counts = dict(expected_eligible_counts or V3_EXPECTED_ELIGIBLE_COUNTS)
    expected_counts_match = all(
        eligibility_counts.get(key, 0) == expected for key, expected in expected_counts.items()
    )
    expected_b = dict(expected_b_ledger or V3_EXPECTED_B_LEDGER)
    b_eligible = eligibility_counts.get("B-T1", 0) + eligibility_counts.get("B-T2", 0)
    b_excluded = sum(
        1
        for row in exclusions
        if row.get("category") == "B" and row.get("time_band") in {"T1", "T2"}
    )
    b_ledger_matches = {
        "eligible": b_eligible,
        "excluded": b_excluded,
        "total": b_eligible + b_excluded,
    } == expected_b
    b_earnings_rows = [row for row in cohort_rows if _config_key(row) == "B-EARNINGS"]
    issuers_by_band: dict[str, set[str]] = defaultdict(set)
    for row in b_earnings_rows:
        issuer = row.get("issuer")
        if issuer is not None and str(issuer).strip():
            issuers_by_band[str(row["time_band"])].add(str(issuer))
    common_issuer_ids = sorted(issuers_by_band.get("T1", set()) & issuers_by_band.get("T2", set()))
    current_snapshot = (
        dataset_lock.revision == V3_DATASET_REVISION
        and dataset_lock.record_count == V3_EXPECTED_RECORD_COUNT
    )
    status = (
        CURRENT_DATA_INELIGIBLE
        if current_snapshot or not support_passed or not expected_counts_match
        else "ELIGIBLE"
    )
    b_status = EXPLORATORY_ONLY
    counts: dict[str, Any] = {
        "schema_version": "experiment-v3-eligibility-counts/v1",
        "record_count": len(rows),
        "raw_by_band_and_category": dict(sorted(raw_counts.items())),
        "eligible_by_band_and_category": dict(sorted(eligibility_counts.items())),
        "excluded_by_reason": dict(
            sorted(Counter(reason for row in exclusions for reason in row["reasons"]).items())
        ),
        "b_ledger": {
            "eligible": b_eligible,
            "excluded": b_excluded,
            "total": b_eligible + b_excluded,
        },
        "eligibility_counts_match_expected": expected_counts_match,
        "b_ledger_matches_expected": b_ledger_matches,
        "b_exploratory": {
            "status": EXPLORATORY_ONLY,
            "ranking_eligible": False,
            "estimand": "equal_issuer_weight_after_issuer_means",
            "common_issuer_count": len(common_issuer_ids),
            "common_issuer_ids": common_issuer_ids,
            "expected_common_issuer_count": 15,
        },
        "expected_current_pin": {
            "eligible_counts": expected_counts,
            "b_ledger": expected_b,
        },
        "t1_t2_status": status,
        "b_status": b_status,
    }
    covariates = tuple(
        {
            "task_id": row["task_id"],
            "category": row["category"],
            "time_band": row["time_band"],
            "stratum": row["stratum"],
            "covariates": row["covariates"],
        }
        for row in sorted(cohort_rows, key=lambda item: item["task_id"])
    )
    artifact_hashes = {
        "eligibility_counts": sha256_json(counts),
        "exclusions": sha256_json(list(exclusions)),
        "covariates": sha256_json(list(covariates)),
        "common_support": sha256_json(common_support),
    }
    design_lock = freeze_design_lock(
        dataset_lock,
        analysis_version=analysis_version,
        seed_namespace=seed_namespace,
        policy=selected_policy,
        states={"t1_t2_effect": status, "b_calibration": b_status},
        artifact_hashes=artifact_hashes,
    )
    preregistration = build_preregistration(dataset_lock, design_lock, policy=selected_policy)
    return CohortAudit(
        dataset_lock=dataset_lock,
        design_lock=design_lock,
        eligibility_counts=counts,
        exclusions=tuple(exclusions),
        covariates=covariates,
        common_support=common_support,
        preregistration=preregistration,
        status=status,
        b_status=b_status,
    )


build_v3_cohort = build_cohort
build_cohort_audit = build_cohort


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_once(path: Path, content: str) -> None:
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise LockError(f"unable to read existing immutable artifact: {path}") from exc
        if existing != content:
            raise LockError(f"append-only artifact drift/overwrite refused: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError:
        _write_once(path, content)
    except OSError as exc:
        raise LockError(f"unable to write immutable artifact: {path}") from exc


V3_ARTIFACT_FILENAMES = (
    "dataset-lock.json",
    "eligibility-counts.json",
    "exclusions.jsonl",
    "covariates.jsonl",
    "common-support.json",
    "preregistration.json",
    "design-lock.json",
    "design-lock.sha256",
)


def write_v3_artifacts(audit: CohortAudit, output_dir: str | Path) -> dict[str, Path]:
    """Write the v3 lock/audit bundle append-only and idempotently."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    values = {
        "dataset-lock.json": _json_text(audit.dataset_lock.to_dict()),
        "eligibility-counts.json": _json_text(audit.eligibility_counts),
        "exclusions.jsonl": _jsonl_text(audit.exclusions),
        "covariates.jsonl": _jsonl_text(audit.covariates),
        "common-support.json": _json_text(audit.common_support),
        "preregistration.json": _json_text(audit.preregistration),
        "design-lock.json": _json_text(audit.design_lock.to_dict()),
        "design-lock.sha256": audit.design_lock.design_lock_sha256 + "\n",
    }
    for name, content in values.items():
        _write_once(root / name, content)
    return {name: root / name for name in V3_ARTIFACT_FILENAMES}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LockError(f"invalid v3 JSON artifact: {path}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise LockError(f"invalid v3 JSONL artifact: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LockError(f"invalid v3 JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise LockError(f"v3 JSONL row must be an object: {path}:{line_number}")
        assert_score_blind(value, path=f"{path}:{line_number}")
        rows.append(value)
    return rows


def _read_design_lock(value: Any) -> DesignLock:
    if not isinstance(value, Mapping):
        raise LockError("design-lock.json must contain an object")
    required = {
        "schema_version",
        "analysis_version",
        "dataset_lock_sha256",
        "population_manifest_sha256",
        "matching",
        "estimand",
        "metrics",
        "states",
        "seed_namespace",
        "component_hashes",
        "design_lock_sha256",
    }
    if set(value) != required:
        raise LockError("design-lock.json has an unexpected field set")
    return DesignLock(
        schema_version=str(value["schema_version"]),
        analysis_version=str(value["analysis_version"]),
        dataset_lock_sha256=str(value["dataset_lock_sha256"]),
        population_manifest_sha256=str(value["population_manifest_sha256"]),
        matching=dict(value["matching"]),
        estimand=dict(value["estimand"]),
        metrics=dict(value["metrics"]),
        states=dict(value["states"]),
        seed_namespace=str(value["seed_namespace"]),
        component_hashes=dict(value["component_hashes"]),
        design_lock_sha256=str(value["design_lock_sha256"]),
    )


def load_dataset_lock(path: str | Path) -> DatasetLock:
    """Load a dataset lock without consulting a Hub or any score artifact."""

    candidate = Path(path).expanduser().resolve()
    value = _load_json(candidate)
    if not isinstance(value, Mapping):
        raise LockError("dataset-lock.json must contain an object")
    required = {
        "schema_version",
        "repo_id",
        "revision",
        "tree_sha256",
        "source_manifest_sha256",
        "release_label",
        "record_count",
        "unique_task_count",
        "config_counts",
        "task_ids_sha256",
        "dataset_sha256",
        "evidence_sha256",
        "covariates_sha256",
        "matching_fields_sha256",
        "seed_sha256",
        "population_manifest_sha256",
    }
    if set(value) != required:
        raise LockError("dataset-lock.json has an unexpected field set")
    try:
        return DatasetLock(
            schema_version=str(value["schema_version"]),
            repo_id=str(value["repo_id"]),
            revision=str(value["revision"]),
            tree_sha256=str(value["tree_sha256"]),
            source_manifest_sha256=str(value["source_manifest_sha256"]),
            release_label=str(value["release_label"]),
            record_count=value["record_count"],
            unique_task_count=value["unique_task_count"],
            config_counts=value["config_counts"],
            task_ids_sha256=str(value["task_ids_sha256"]),
            dataset_sha256=str(value["dataset_sha256"]),
            evidence_sha256=str(value["evidence_sha256"]),
            covariates_sha256=str(value["covariates_sha256"]),
            matching_fields_sha256=str(value["matching_fields_sha256"]),
            seed_sha256=str(value["seed_sha256"]),
            population_manifest_sha256=str(value["population_manifest_sha256"]),
        )
    except LockError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise LockError("dataset-lock.json has an invalid shape") from exc


def load_design_lock(path: str | Path) -> DesignLock:
    """Load and self-verify a design lock."""

    candidate = Path(path).expanduser().resolve()
    try:
        return _read_design_lock(_load_json(candidate))
    except LockError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise LockError("design-lock.json has an invalid shape") from exc


def validate_v3_artifacts(root: str | Path) -> dict[str, Any]:
    """Validate every lock/audit artifact and all component hashes fail-closed."""

    directory = Path(root).expanduser().resolve()
    paths = {name: directory / name for name in V3_ARTIFACT_FILENAMES}
    if not directory.is_dir():
        raise LockError(f"v3 artifact directory is unavailable: {directory}")
    unexpected = sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.relative_to(directory).as_posix() not in paths
    )
    if unexpected:
        raise LockError("unexpected v3 artifact(s): " + ", ".join(unexpected))
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise LockError("v3 artifact(s) missing: " + ", ".join(missing))
    dataset_lock = load_dataset_lock(paths["dataset-lock.json"])
    design_lock = load_design_lock(paths["design-lock.json"])
    design_lock.verify(dataset_lock)
    try:
        declared_design_hash = paths["design-lock.sha256"].read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise LockError("design-lock.sha256 is unavailable") from exc
    if declared_design_hash != design_lock.design_lock_sha256:
        raise LockError("design-lock.sha256 does not match design-lock.json")
    counts = _load_json(paths["eligibility-counts.json"])
    common_support = _load_json(paths["common-support.json"])
    preregistration = _load_json(paths["preregistration.json"])
    if not isinstance(counts, Mapping) or not isinstance(common_support, Mapping):
        raise LockError("v3 counts/common-support artifacts must contain objects")
    exclusions = _load_jsonl(paths["exclusions.jsonl"])
    covariates = _load_jsonl(paths["covariates.jsonl"])
    component_values = {
        "eligibility_counts": sha256_json(counts),
        "exclusions": sha256_json(exclusions),
        "covariates": sha256_json(covariates),
        "common_support": sha256_json(common_support),
    }
    for key, observed in component_values.items():
        if design_lock.component_hashes.get(key) != observed:
            raise LockError(f"v3 artifact hash drift detected: {key}")
    if not isinstance(preregistration, Mapping):
        raise LockError("preregistration.json must contain an object")
    if preregistration.get("design_lock_sha256") != design_lock.design_lock_sha256:
        raise LockError("preregistration does not bind design lock")
    if preregistration.get("dataset_lock_sha256") != dataset_lock.lock_sha256:
        raise LockError("preregistration does not bind dataset lock")
    for field_name, component_name in (
        ("matching", "matching"),
        ("estimand", "estimand"),
        ("metrics", "metrics"),
    ):
        if design_lock.component_hashes.get(component_name) != sha256_json(
            preregistration.get(field_name)
        ):
            raise LockError(f"preregistration {field_name} hash drift detected")
    if preregistration.get("states") != dict(sorted(design_lock.states.items())):
        raise LockError("preregistration states drift detected")
    if counts.get("t1_t2_status") != design_lock.states.get("t1_t2_effect"):
        raise LockError("eligibility counts do not bind T1/T2 state")
    if counts.get("b_status") != design_lock.states.get("b_calibration"):
        raise LockError("eligibility counts do not bind B state")
    return {
        "status": "valid",
        "schema_version": "experiment/v3",
        "dataset_lock_sha256": dataset_lock.lock_sha256,
        "population_manifest_sha256": dataset_lock.population_manifest_sha256,
        "design_lock_sha256": design_lock.design_lock_sha256,
        "t1_t2_status": counts.get("t1_t2_status"),
        "b_status": counts.get("b_status"),
        "record_count": dataset_lock.record_count,
        "artifact_files": list(V3_ARTIFACT_FILENAMES),
    }


def load_score_blind_records(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load only a JSONL task source after checking its path and row keys."""

    candidate = assert_score_blind_path(path)
    rows = _load_jsonl(candidate)
    return _sorted_rows(rows)


def _validate_binary_pairs(
    probabilities: Sequence[Any], outcomes: Sequence[Any]
) -> tuple[list[float | None], list[int]]:
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ExperimentV3Error("probabilities and outcomes must be equally sized and non-empty")
    ys: list[int] = []
    for value in outcomes:
        if isinstance(value, bool):
            ys.append(int(value))
        elif isinstance(value, int | float) and not isinstance(value, bool) and value in {0, 1}:
            ys.append(int(value))
        else:
            raise ExperimentV3Error("outcomes must be binary")
    ps: list[float | None] = []
    for value in probabilities:
        if value is None or isinstance(value, bool) or not isinstance(value, int | float):
            ps.append(None)
            continue
        numeric = float(value)
        ps.append(numeric if math.isfinite(numeric) and 0.0 <= numeric <= 1.0 else None)
    return ps, ys


def operational_brier_score(
    probabilities: Sequence[Any], outcomes: Sequence[Any]
) -> dict[str, float | int]:
    ps, ys = _validate_binary_pairs(probabilities, outcomes)
    losses = [
        1.0 if probability is None else (probability - outcome) ** 2
        for probability, outcome in zip(ps, ys, strict=True)
    ]
    valid = sum(probability is not None for probability in ps)
    return {
        "brier_score": mean(losses),
        "valid_output_rate": valid / len(ps),
        "valid_count": valid,
        "invalid_count": len(ps) - valid,
        "n": len(ps),
    }


def brier_score(
    probabilities: Sequence[Any],
    outcomes: Sequence[Any],
    *,
    operational: bool = False,
) -> float:
    """Return raw evaluator BS, optionally assigning invalid predictions loss 1."""

    report = operational_brier_score(probabilities, outcomes)
    if not operational and report["invalid_count"]:
        raise ExperimentV3Error("raw Brier score requires valid probabilities")
    return float(report["brier_score"])


def brier_skill_score(
    probabilities_or_brier: Sequence[Any] | float,
    outcomes: Sequence[Any] | None = None,
    *,
    baseline: float = 0.5,
    operational: bool = False,
) -> float:
    """Compute unclipped BSS; the primary fixed baseline is p=0.5."""

    if baseline != 0.5 and not 0.0 <= baseline <= 1.0:
        raise ExperimentV3Error("baseline must be between 0 and 1")
    if outcomes is None:
        if isinstance(probabilities_or_brier, bool) or not isinstance(
            probabilities_or_brier, int | float
        ):
            raise ExperimentV3Error("a scalar Brier score is required without outcomes")
        score = float(probabilities_or_brier)
        denominator = 0.25 if baseline == 0.5 else baseline * (1.0 - baseline)
    else:
        if isinstance(probabilities_or_brier, str | bytes):
            raise ExperimentV3Error("probabilities must be a sequence")
        score = brier_score(probabilities_or_brier, outcomes, operational=operational)
        denominator = mean((baseline - outcome) ** 2 for outcome in outcomes)
    if denominator == 0:
        raise ExperimentV3Error("Brier baseline has zero variance")
    return 1.0 - score / denominator


def _wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if not total:
        return None, None
    proportion = successes / total
    z2 = WILSON_Z_95**2
    center = (proportion + z2 / (2 * total)) / (1 + z2 / total)
    half = (
        WILSON_Z_95
        * math.sqrt(proportion * (1 - proportion) / total + z2 / (4 * total * total))
        / (1 + z2 / total)
    )
    return max(0.0, center - half), min(1.0, center + half)


def reliability_report(probabilities: Sequence[Any], outcomes: Sequence[Any]) -> dict[str, Any]:
    """Return fixed five-bin reliability rows, retaining empty bins."""

    ps, ys = _validate_binary_pairs(probabilities, outcomes)
    bins: list[list[tuple[float, int]]] = [[] for _ in range(RELIABILITY_BIN_COUNT)]
    invalid_count = 0
    for probability, outcome in zip(ps, ys, strict=True):
        if probability is None:
            invalid_count += 1
            continue
        index = next(
            index
            for index, upper in enumerate(RELIABILITY_BIN_EDGES[1:])
            if probability < upper or (index == RELIABILITY_BIN_COUNT - 1 and probability <= upper)
        )
        bins[index].append((probability, outcome))
    report_bins: list[dict[str, Any]] = []
    for index, values in enumerate(bins):
        lower, upper = RELIABILITY_BIN_EDGES[index], RELIABILITY_BIN_EDGES[index + 1]
        n = len(values)
        mean_probability = mean(value for value, _ in values) if values else None
        event_rate = mean(outcome for _, outcome in values) if values else None
        ci_lower, ci_upper = _wilson_interval(sum(outcome for _, outcome in values), n)
        report_bins.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "upper_inclusive": index == RELIABILITY_BIN_COUNT - 1,
                "n": n,
                "mean_probability": mean_probability,
                "event_rate": event_rate,
                "wilson_lower": ci_lower,
                "wilson_upper": ci_upper,
                "sparse": n < 10,
            }
        )
    valid_count = len(ps) - invalid_count
    ece = (
        sum(
            entry["n"] / valid_count * abs(entry["mean_probability"] - entry["event_rate"])
            for entry in report_bins
            if entry["n"]
        )
        if valid_count
        else None
    )
    return {
        "bins": report_bins,
        "ece": ece,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "n": len(ps),
        "binning": "five_equal_width_[0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1]",
    }


def reliability_ece(probabilities: Sequence[Any], outcomes: Sequence[Any]) -> float | None:
    return reliability_report(probabilities, outcomes)["ece"]


def reliability_bins(probabilities: Sequence[Any], outcomes: Sequence[Any]) -> list[dict[str, Any]]:
    return reliability_report(probabilities, outcomes)["bins"]


expected_calibration_error = reliability_ece
compute_brier_score = brier_score
compute_brier_skill_score = brier_skill_score


def aggregate_trial_brier(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Score-blind-to-design grouping for operational per-attempt trial BS."""

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    valid_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ExperimentV3Error(f"trial row {index} must be an object")
        system = _require_nonempty(row.get("system_id"), f"trial row {index}.system_id")
        task = _require_nonempty(row.get("task_id"), f"trial row {index}.task_id")
        probability = row.get("probability", row.get("p"))
        outcome = row.get("outcome", row.get("y"))
        _, ys = _validate_binary_pairs([probability], [outcome])
        ps, _ = _validate_binary_pairs([probability], ys)
        loss = 1.0 if ps[0] is None else (ps[0] - ys[0]) ** 2
        grouped[(system, task)].append(loss)
        if ps[0] is not None:
            valid_grouped[(system, task)].append(ps[0])
    aggregate = [
        {
            "system_id": system,
            "task_id": task,
            "attempt_count": len(losses),
            "brier_score": mean(losses),
            "valid_output_rate": len(valid_grouped.get((system, task), ())) / len(losses),
        }
        for (system, task), losses in sorted(grouped.items())
    ]
    return {
        "policy": "per_attempt_invalid_1_then_arithmetic_mean_within_system_task",
        "rows": aggregate,
        "system_task_count": len(aggregate),
    }


multi_trial_operational_brier = aggregate_trial_brier


def task_mean_probabilities(
    rows: Iterable[Mapping[str, Any]], *, task_key: str = "task_id"
) -> dict[str, Any]:
    """Average valid probabilities per task and explicitly report zero-valid tasks."""

    values: dict[str, list[float]] = defaultdict(list)
    totals: Counter[str] = Counter()
    for row in rows:
        task_id = _require_nonempty(row.get(task_key), task_key)
        totals[task_id] += 1
        probability = row.get("probability", row.get("p"))
        if (
            not isinstance(probability, bool)
            and isinstance(probability, int | float)
            and math.isfinite(probability)
            and 0.0 <= probability <= 1.0
        ):
            values[task_id].append(float(probability))
    return {
        "task_means": {
            task_id: (mean(values[task_id]) if values.get(task_id) else None)
            for task_id in sorted(totals)
        },
        "zero_valid_tasks": sorted(task_id for task_id in totals if not values.get(task_id)),
        "task_count": len(totals),
    }


def bootstrap_seed(dataset_pin: str, analysis_version: str) -> str:
    dataset = _require_nonempty(dataset_pin, "dataset_pin")
    version = _require_nonempty(analysis_version, "analysis_version")
    return sha256_text(f"{dataset}|{version}")


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ExperimentV3Error("bootstrap distribution is empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def paired_event_cluster_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_pin: str,
    analysis_version: str,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap paired event clusters with common draws and a frozen seed."""

    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise ExperimentV3Error("bootstrap resamples must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ExperimentV3Error("bootstrap confidence must be between 0 and 1")
    grouped: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ExperimentV3Error(f"bootstrap row {index} must be an object")
        cluster = row.get("cluster_key", row.get("event_cluster", row.get("issuer")))
        cluster_id = _require_nonempty(cluster, f"bootstrap row {index}.cluster_key")
        if "difference" in row:
            difference = row["difference"]
        else:
            left = row.get("left", row.get("t1"))
            right = row.get("right", row.get("t2"))
            if _numeric(left) is None or _numeric(right) is None:
                raise ExperimentV3Error("bootstrap rows require difference or paired left/right")
            difference = float(left) - float(right)
        if _numeric(difference) is None:
            raise ExperimentV3Error("bootstrap differences must be finite numbers")
        grouped[cluster_id].append(float(difference))
    clusters = sorted(grouped)
    if not clusters:
        raise ExperimentV3Error("paired event bootstrap requires at least one cluster")
    cluster_means = {cluster: mean(grouped[cluster]) for cluster in clusters}
    observed = mean(cluster_means.values())
    seed_hash = bootstrap_seed(dataset_pin, analysis_version)
    rng = random.Random(int(seed_hash[:16], 16))
    draws = [
        mean(cluster_means[clusters[rng.randrange(len(clusters))]] for _ in clusters)
        for _ in range(resamples)
    ]
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": observed,
        "ci_lower": _quantile(draws, tail),
        "ci_upper": _quantile(draws, 1.0 - tail),
        "cluster_count": len(clusters),
        "resamples": resamples,
        "confidence": confidence,
        "seed_sha256": seed_hash,
        "seed": int(seed_hash[:16], 16),
        "unit": "paired_event_cluster_common_draws",
    }


outcome_cluster_bootstrap = paired_event_cluster_bootstrap


def load_v3_artifacts(root: str | Path) -> dict[str, Any]:
    """Alias emphasizing that this operation is validation-only."""

    return validate_v3_artifacts(root)


__all__ = [
    "CURRENT_DATA_INELIGIBLE",
    "EXPLORATORY_ONLY",
    "ExperimentV3Error",
    "ScoreBlindError",
    "LockError",
    "ScoreBlindGate",
    "DatasetLock",
    "MatchingPolicy",
    "EstimandPolicy",
    "MetricPolicy",
    "V3Policy",
    "DesignLock",
    "CohortAudit",
    "V3_ARTIFACT_FILENAMES",
    "V3_DATASET_MANIFEST_SHA256",
    "V3_DATASET_REPO_ID",
    "V3_DATASET_REVISION",
    "V3_DATASET_TREE_SHA256",
    "V3_DATASET_TREE_SHA",
    "V3_EXPECTED_B_LEDGER",
    "V3_EXPECTED_ELIGIBLE_COUNTS",
    "V3_EXPECTED_RECORD_COUNT",
    "V3_EXPECTED_UNIQUE_TASK_COUNT",
    "RELIABILITY_BIN_EDGES",
    "assert_score_blind",
    "assert_score_blind_path",
    "aggregate_trial_brier",
    "brier_score",
    "brier_skill_score",
    "bootstrap_seed",
    "build_cohort",
    "build_cohort_audit",
    "build_common_support",
    "build_dataset_lock",
    "build_preregistration",
    "build_v3_cohort",
    "canonical_json",
    "expected_calibration_error",
    "compute_brier_score",
    "compute_brier_skill_score",
    "freeze_design_lock",
    "load_score_blind_records",
    "load_dataset_lock",
    "load_design_lock",
    "load_v3_artifacts",
    "multi_trial_operational_brier",
    "operational_brier_score",
    "paired_event_cluster_bootstrap",
    "reliability_ece",
    "reliability_bins",
    "reliability_report",
    "sha256_json",
    "task_mean_probabilities",
    "validate_v3_artifacts",
    "write_v3_artifacts",
]
