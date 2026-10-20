"""Shared snapshot, transport, and provider-response identity rules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmark_harness.config import ResolvedModelProfile

MINI_SNAPSHOT_MODEL_NAME = "gpt-4.1-mini-2025-04-14"
MINI_SNAPSHOT_BACKBONE_ID = "openai:gpt-4.1-mini-2025-04-14"
MINI_WIRE_MODEL_NAME = "gpt-4.1-mini"
MINI_WIRE_NAME_BINDING = "user_authorized_azure_wire_mapping/v1"
MINI_PROVIDER_RESPONSE_MODEL_NAMES = frozenset({MINI_WIRE_MODEL_NAME, MINI_SNAPSHOT_MODEL_NAME})


def is_mini_snapshot_profile(profile: ResolvedModelProfile) -> bool:
    """Return whether a runtime profile carries the isolated mini identity."""

    return (
        profile.backbone_id == MINI_SNAPSHOT_BACKBONE_ID
        and profile.snapshot_model_name == MINI_SNAPSHOT_MODEL_NAME
        and profile.model_name == MINI_WIRE_MODEL_NAME
    )


def verify_provider_response_model(
    response: Mapping[str, Any], profile: ResolvedModelProfile
) -> str | None:
    """Validate and return a provider model attestation.

    The mini pack requires a string-valued response attestation and accepts only
    the exact Azure wire echo or the exact dated snapshot name.  Legacy and
    generic profiles retain their prior optional-attestation behavior.
    """

    observed = response.get("model")
    if is_mini_snapshot_profile(profile):
        if not isinstance(observed, str):
            raise ValueError("model_identity_mismatch")
        if observed not in MINI_PROVIDER_RESPONSE_MODEL_NAMES:
            raise ValueError("model_identity_mismatch")
        return observed
    if observed is None:
        return None
    if not isinstance(observed, str) or observed.strip() != profile.model_name:
        raise ValueError("model_identity_mismatch")
    return observed.strip()
