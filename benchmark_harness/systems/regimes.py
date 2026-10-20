"""Explicit retrieval-regime identities and non-ranking leakage policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RetrievalRegime(StrEnum):
    CLOSED_BOOK = "closed_book"
    CUTOFF_CONTROLLED = "cutoff_controlled"
    UNRESTRICTED_CURRENT_WEB = "unrestricted_current_web"


class RetrievalPolicyError(ValueError):
    """Raised when a retrieval regime is missing its explicit policy inputs."""


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    """Immutable policy identity kept separate from system ranking metadata."""

    regime: RetrievalRegime
    policy_version: str
    namespace: str
    ranking_eligible: bool
    live_authorized: bool
    calibration_sha256: str | None = None
    authorization_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.policy_version not in {
            "closed_book/v1",
            "cutoff_controlled/v1",
            "leakage_probe/v1",
        }:
            raise RetrievalPolicyError("unsupported retrieval policy version")
        if not self.namespace.strip():
            raise RetrievalPolicyError("retrieval namespace must be non-empty")
        if not isinstance(self.ranking_eligible, bool) or not isinstance(
            self.live_authorized, bool
        ):
            raise RetrievalPolicyError("retrieval policy flags must be boolean")
        if self.regime is RetrievalRegime.UNRESTRICTED_CURRENT_WEB:
            if self.policy_version != "leakage_probe/v1" or self.namespace != "leakage_probe/v1":
                raise RetrievalPolicyError(
                    "unrestricted web must use the leakage_probe/v1 identity"
                )
            if self.ranking_eligible:
                raise RetrievalPolicyError("unrestricted web is never ranking eligible")
            if self.calibration_sha256 is not None and not re.fullmatch(
                r"[0-9a-f]{64}", self.calibration_sha256
            ):
                raise RetrievalPolicyError("calibration_sha256 must be a lowercase SHA-256")
            if self.authorization_sha256 is not None and not re.fullmatch(
                r"[0-9a-f]{64}", self.authorization_sha256
            ):
                raise RetrievalPolicyError("authorization_sha256 must be a lowercase SHA-256")
        elif self.ranking_eligible:
            raise RetrievalPolicyError("integration retrieval policies remain non-ranking")

    @property
    def is_leakage_probe(self) -> bool:
        return self.regime is RetrievalRegime.UNRESTRICTED_CURRENT_WEB

    def validate_capability(self, *, search_handler_present: bool) -> None:
        if self.regime is RetrievalRegime.CLOSED_BOOK and search_handler_present:
            raise RetrievalPolicyError("closed_book cannot receive a search capability")
        if self.regime is not RetrievalRegime.CLOSED_BOOK and not search_handler_present:
            raise RetrievalPolicyError("retrieval regime requires a host-owned search capability")

    def assert_live_execution_allowed(self) -> None:
        if self.is_leakage_probe and not self.live_authorized:
            raise RetrievalPolicyError(
                "live unrestricted-current-web execution requires explicit authorization"
            )
        if (
            self.is_leakage_probe
            and self.calibration_sha256 is None
            and self.authorization_sha256 is None
        ):
            raise RetrievalPolicyError(
                "live unrestricted-current-web requires calibration or authorization evidence"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.policy_version,
            "regime": self.regime.value,
            "namespace": self.namespace,
            "ranking_eligible": self.ranking_eligible,
            "live_authorized": self.live_authorized,
            "calibration_sha256": self.calibration_sha256,
            "authorization_sha256": self.authorization_sha256,
        }


def build_retrieval_policy(
    regime: RetrievalRegime | str,
    *,
    calibration_sha256: str | None = None,
    live_authorized: bool = False,
    authorization_sha256: str | None = None,
) -> RetrievalPolicy:
    selected = RetrievalRegime(regime)
    if selected is RetrievalRegime.CLOSED_BOOK:
        return RetrievalPolicy(
            regime=selected,
            policy_version="closed_book/v1",
            namespace="closed_book/v1",
            ranking_eligible=False,
            live_authorized=False,
        )
    if selected is RetrievalRegime.CUTOFF_CONTROLLED:
        return RetrievalPolicy(
            regime=selected,
            policy_version="cutoff_controlled/v1",
            namespace="cutoff_controlled/v1",
            ranking_eligible=False,
            live_authorized=False,
        )
    return RetrievalPolicy(
        regime=selected,
        policy_version="leakage_probe/v1",
        namespace="leakage_probe/v1",
        ranking_eligible=False,
        live_authorized=live_authorized,
        calibration_sha256=calibration_sha256,
        authorization_sha256=authorization_sha256,
    )


__all__ = [
    "RetrievalPolicy",
    "RetrievalPolicyError",
    "RetrievalRegime",
    "build_retrieval_policy",
]
