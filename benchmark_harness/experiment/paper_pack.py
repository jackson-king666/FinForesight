"""The opt-in experiment/v3 paper-pack contract.

This module is intentionally separate from :mod:`manifest`.  Historical
experiment/v1 and experiment/v2 manifests retain their non-ranking behavior;
only a v3 paper pack can request a ranking, and the request is still subject to
all effective eligibility gates.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from benchmark_harness.calibration import validate_calibration_root
from benchmark_harness.experiment.eligibility import EligibilityReport, evaluate_ranking_eligibility

PAPER_PACK_V3 = "experiment/v3"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_REGIMES = frozenset({"closed_book", "cutoff_controlled", "unrestricted_current_web"})
_ROOT_FIELDS = frozenset(
    {
        "calibration",
        "claim_scope",
        "dataset",
        "experiment_as_of",
        "experiment_id",
        "metadata",
        "policy",
        "retrieval_regimes",
        "schema_version",
        "systems",
    }
)
_DATASET_FIELDS = frozenset(
    {"dataset_fingerprint", "record_count", "release_label", "repo_id", "revision"}
)
_CALIBRATION_FIELDS = frozenset({"path", "sha256"})
_POLICY_FIELDS = frozenset(
    {
        "live_unrestricted_authorized",
        "main_leaderboard_regime",
        "ranking_eligible_default",
        "score_selection",
    }
)
_SYSTEM_FIELDS = frozenset(
    {
        "artifact_complete",
        "comparable",
        "metadata",
        "ranking_gates",
        "ranking_requested",
        "system_id",
    }
)


class PaperPackV3Error(ValueError):
    """Raised when a v3 paper pack is ambiguous or violates its policy."""


def _strict_mapping(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PaperPackV3Error(f"{label} must be an object")
    unknown = sorted(set(value) - fields)
    if unknown:
        raise PaperPackV3Error(f"{label} contains unknown field(s): {', '.join(unknown)}")
    return value


def _required_string(value: Any, name: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PaperPackV3Error(f"{label}.{name} must be a non-empty string")
    return value.strip()


def _required_bool(value: Any, name: str, label: str) -> bool:
    if not isinstance(value, bool):
        raise PaperPackV3Error(f"{label}.{name} must be boolean")
    return value


@dataclass(frozen=True, slots=True)
class PaperPackV3Policy:
    """Policy defaults for a new paper pack."""

    ranking_eligible_default: bool = True
    main_leaderboard_regime: str = "cutoff_controlled"
    score_selection: str = "forbidden"
    live_unrestricted_authorized: bool = False

    def __post_init__(self) -> None:
        if self.ranking_eligible_default is not True:
            raise PaperPackV3Error("experiment/v3 ranking_eligible_default must be true")
        if self.main_leaderboard_regime != "cutoff_controlled":
            raise PaperPackV3Error("experiment/v3 main leaderboard must use cutoff_controlled")
        if self.score_selection != "forbidden":
            raise PaperPackV3Error("score-based selection is forbidden")
        if self.live_unrestricted_authorized is not False:
            raise PaperPackV3Error("live_unrestricted_authorized must remain false")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranking_eligible_default": self.ranking_eligible_default,
            "main_leaderboard_regime": self.main_leaderboard_regime,
            "score_selection": self.score_selection,
            "live_unrestricted_authorized": self.live_unrestricted_authorized,
        }


@dataclass(frozen=True, slots=True)
class PaperPackV3System:
    """A system's explicit ranking request and immutable gate inputs."""

    system_id: str
    ranking_requested: bool
    ranking_gates: Mapping[str, bool]
    comparable: bool
    artifact_complete: bool
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        _required_string(self.system_id, "system_id", "system")
        if not isinstance(self.ranking_requested, bool):
            raise PaperPackV3Error("system.ranking_requested must be boolean")
        if not isinstance(self.ranking_gates, Mapping):
            raise PaperPackV3Error("system.ranking_gates must be an object")
        normalized: dict[str, bool] = {}
        for gate, passed in self.ranking_gates.items():
            if not isinstance(gate, str) or not gate.strip():
                raise PaperPackV3Error("ranking gate names must be non-empty strings")
            if not isinstance(passed, bool):
                raise PaperPackV3Error(f"ranking gate {gate!r} must be boolean")
            normalized[gate] = passed
        object.__setattr__(self, "ranking_gates", normalized)
        if not isinstance(self.comparable, bool) or not isinstance(self.artifact_complete, bool):
            raise PaperPackV3Error("system comparable/artifact_complete must be boolean")
        if not isinstance(self.metadata, Mapping):
            raise PaperPackV3Error("system.metadata must be an object")

    def eligibility(
        self,
        retrieval_regime: str,
        *,
        comparable: bool | None = None,
        artifact_complete: bool | None = None,
        gates: Mapping[str, bool] | None = None,
    ) -> EligibilityReport:
        return evaluate_ranking_eligibility(
            ranking_requested=self.ranking_requested,
            retrieval_regime=retrieval_regime,
            gates=self.ranking_gates if gates is None else gates,
            comparable=self.comparable if comparable is None else comparable,
            artifact_complete=self.artifact_complete
            if artifact_complete is None
            else artifact_complete,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "ranking_requested": self.ranking_requested,
            "ranking_gates": dict(sorted(self.ranking_gates.items())),
            "comparable": self.comparable,
            "artifact_complete": self.artifact_complete,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PaperPackV3:
    """Parsed v3 pack with explicit materialization helpers."""

    schema_version: str
    experiment_id: str
    experiment_as_of: date
    claim_scope: str
    dataset: Mapping[str, Any]
    calibration_path: str
    calibration_sha256: str
    policy: PaperPackV3Policy
    systems: tuple[PaperPackV3System, ...]
    retrieval_regimes: tuple[str, ...]
    source_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != PAPER_PACK_V3:
            raise PaperPackV3Error("schema_version must be experiment/v3")
        if not self.systems:
            raise PaperPackV3Error("experiment/v3 must declare at least one system")
        ids = [system.system_id for system in self.systems]
        if len(ids) != len(set(ids)):
            raise PaperPackV3Error("experiment/v3 contains duplicate system_id values")
        if not self.retrieval_regimes or len(set(self.retrieval_regimes)) != len(
            self.retrieval_regimes
        ):
            raise PaperPackV3Error("experiment/v3 retrieval_regimes must be unique and non-empty")
        if any(regime not in _REGIMES for regime in self.retrieval_regimes):
            raise PaperPackV3Error("experiment/v3 contains an unknown retrieval regime")
        if "cutoff_controlled" not in self.retrieval_regimes:
            raise PaperPackV3Error("experiment/v3 must declare cutoff_controlled")
        if not _SHA256.fullmatch(self.calibration_sha256):
            raise PaperPackV3Error("calibration.sha256 must be a lowercase SHA-256")
        if not isinstance(self.dataset, Mapping) or not isinstance(self.metadata, Mapping):
            raise PaperPackV3Error("dataset and metadata must be objects")

    def system(self, system_id: str) -> PaperPackV3System:
        for system in self.systems:
            if system.system_id == system_id:
                return system
        raise PaperPackV3Error(f"experiment/v3 has no system {system_id!r}")

    def effective_eligibility(
        self,
        system_id: str,
        retrieval_regime: str,
        *,
        comparable: bool | None = None,
        artifact_complete: bool | None = None,
        gates: Mapping[str, bool] | None = None,
    ) -> EligibilityReport:
        if retrieval_regime not in self.retrieval_regimes:
            raise PaperPackV3Error(f"retrieval regime {retrieval_regime!r} is not in the pack")
        return self.system(system_id).eligibility(
            retrieval_regime,
            comparable=comparable,
            artifact_complete=artifact_complete,
            gates=gates,
        )

    def materialize_rows(self) -> list[dict[str, Any]]:
        """Emit one row per system/regime with no rank or score selection."""

        rows: list[dict[str, Any]] = []
        for system in self.systems:
            for regime in self.retrieval_regimes:
                decision = system.eligibility(regime)
                rows.append(
                    {
                        "system_id": system.system_id,
                        "retrieval_regime": regime,
                        "main_leaderboard_regime": regime == self.policy.main_leaderboard_regime,
                        "ranking_requested": decision.ranking_requested,
                        "ranking_eligible": decision.ranking_eligible,
                        "eligibility_gates": dict(sorted(decision.gates.items())),
                        "comparable": decision.comparable,
                        "artifact_complete": decision.artifact_complete,
                        "eligibility_reasons": list(decision.reasons),
                    }
                )
        return rows

    def validate_calibration(self) -> dict[str, Any]:
        """Verify the declared calibration path and exact manifest hash offline."""

        path = _resolve_declared_path(self.source_path, self.calibration_path)
        result = validate_calibration_root(path)
        if result["calibration_sha256"] != self.calibration_sha256:
            raise PaperPackV3Error(
                "declared calibration SHA does not match artifact_manifest.json: "
                f"expected {self.calibration_sha256}, observed {result['calibration_sha256']}"
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "experiment_as_of": self.experiment_as_of.isoformat(),
            "claim_scope": self.claim_scope,
            "dataset": dict(self.dataset),
            "calibration": {"path": self.calibration_path, "sha256": self.calibration_sha256},
            "policy": self.policy.to_dict(),
            "systems": [system.to_dict() for system in self.systems],
            "retrieval_regimes": list(self.retrieval_regimes),
            "metadata": dict(self.metadata),
        }


def _resolve_declared_path(source_path: Path, declared: str) -> Path:
    direct = (source_path.parent / declared).resolve()
    if direct.exists():
        return direct
    repository_relative = (source_path.parent.parent / declared).resolve()
    return repository_relative


def load_paper_pack_v3(path: str | Path) -> PaperPackV3:
    """Load an explicit v3 template/manifest without executing an experiment."""

    source_path = Path(path).expanduser().resolve()
    try:
        with source_path.open(encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except OSError as exc:
        raise PaperPackV3Error(f"paper pack is unavailable: {source_path}") from exc
    except yaml.YAMLError as exc:
        raise PaperPackV3Error(f"paper pack is invalid YAML: {source_path}") from exc
    root = _strict_mapping(value, _ROOT_FIELDS, "paper pack")
    if root.get("schema_version") != PAPER_PACK_V3:
        raise PaperPackV3Error("schema_version must be experiment/v3")
    experiment_id = _required_string(root.get("experiment_id"), "experiment_id", "paper pack")
    as_of_text = _required_string(root.get("experiment_as_of"), "experiment_as_of", "paper pack")
    try:
        experiment_as_of = date.fromisoformat(as_of_text)
    except ValueError as exc:
        raise PaperPackV3Error("experiment_as_of must be a day-precision ISO date") from exc
    claim_scope = _required_string(root.get("claim_scope"), "claim_scope", "paper pack")
    dataset = _strict_mapping(root.get("dataset"), _DATASET_FIELDS, "paper pack.dataset")
    if "revision" in dataset and not _SHA1.fullmatch(str(dataset["revision"])):
        raise PaperPackV3Error("paper pack.dataset.revision must be a 40-character SHA")
    calibration = _strict_mapping(
        root.get("calibration"), _CALIBRATION_FIELDS, "paper pack.calibration"
    )
    calibration_path = _required_string(calibration.get("path"), "path", "paper pack.calibration")
    calibration_sha256 = _required_string(
        calibration.get("sha256"), "sha256", "paper pack.calibration"
    )
    if not _SHA256.fullmatch(calibration_sha256):
        raise PaperPackV3Error("paper pack.calibration.sha256 must be a lowercase SHA-256")
    policy_value = _strict_mapping(root.get("policy"), _POLICY_FIELDS, "paper pack.policy")
    policy = PaperPackV3Policy(
        ranking_eligible_default=_required_bool(
            policy_value.get("ranking_eligible_default"),
            "ranking_eligible_default",
            "paper pack.policy",
        ),
        main_leaderboard_regime=_required_string(
            policy_value.get("main_leaderboard_regime"),
            "main_leaderboard_regime",
            "paper pack.policy",
        ),
        score_selection=_required_string(
            policy_value.get("score_selection"), "score_selection", "paper pack.policy"
        ),
        live_unrestricted_authorized=_required_bool(
            policy_value.get("live_unrestricted_authorized", False),
            "live_unrestricted_authorized",
            "paper pack.policy",
        ),
    )
    systems_value = root.get("systems")
    if not isinstance(systems_value, list):
        raise PaperPackV3Error("paper pack.systems must be a list")
    systems: list[PaperPackV3System] = []
    for index, system_value in enumerate(systems_value):
        label = f"paper pack.systems[{index}]"
        system = _strict_mapping(system_value, _SYSTEM_FIELDS, label)
        if "ranking_eligible" in system:
            raise PaperPackV3Error(f"{label} may not author ranking_eligible")
        gates = system.get("ranking_gates", {})
        if not isinstance(gates, Mapping):
            raise PaperPackV3Error(f"{label}.ranking_gates must be an object")
        systems.append(
            PaperPackV3System(
                system_id=_required_string(system.get("system_id"), "system_id", label),
                ranking_requested=system.get("ranking_requested", policy.ranking_eligible_default),
                ranking_gates=dict(gates),
                comparable=system.get("comparable", False),
                artifact_complete=system.get("artifact_complete", False),
                metadata=system.get("metadata", {}),
            )
        )
    regimes_value = root.get("retrieval_regimes")
    if not isinstance(regimes_value, list) or any(
        not isinstance(regime, str) for regime in regimes_value
    ):
        raise PaperPackV3Error("paper pack.retrieval_regimes must be a list of strings")
    metadata = root.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise PaperPackV3Error("paper pack.metadata must be an object")
    return PaperPackV3(
        schema_version=PAPER_PACK_V3,
        experiment_id=experiment_id,
        experiment_as_of=experiment_as_of,
        claim_scope=claim_scope,
        dataset=dict(dataset),
        calibration_path=calibration_path,
        calibration_sha256=calibration_sha256,
        policy=policy,
        systems=tuple(systems),
        retrieval_regimes=tuple(regimes_value),
        source_path=source_path,
        metadata=dict(metadata),
    )


def validate_paper_pack_v3(
    path: str | Path, *, validate_calibration: bool = True
) -> dict[str, Any]:
    """Validate a v3 pack and, by default, its declared offline calibration."""

    pack = load_paper_pack_v3(path)
    calibration = pack.validate_calibration() if validate_calibration else None
    return {
        "status": "valid",
        "schema_version": PAPER_PACK_V3,
        "experiment_id": pack.experiment_id,
        "ranking_requested_default": pack.policy.ranking_eligible_default,
        "live_unrestricted_authorized": pack.policy.live_unrestricted_authorized,
        "calibration": calibration,
        "rows": pack.materialize_rows(),
    }


__all__ = [
    "PAPER_PACK_V3",
    "PaperPackV3",
    "PaperPackV3Error",
    "PaperPackV3Policy",
    "PaperPackV3System",
    "load_paper_pack_v3",
    "validate_paper_pack_v3",
]
