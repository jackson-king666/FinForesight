"""The single registry for runnable benchmark profiles and baselines.

Configuration, execution, and reporting all consume this module so an agent
cannot accidentally add a profile to only one of those paths.  A baseline may
be accepted for control purposes without being eligible for ranking.  Search
principals are a separate admission set: an experiment system may hold a
task-scoped Search+Filter session under its canonical principal without ever
becoming a public runner baseline or ranking candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


class ProfileRegistryError(ValueError):
    """Raised when a profile or baseline is outside the public contract."""


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    """Immutable support and comparability rules for one profile."""

    name: str
    policy_version: str
    allowed_baselines: frozenset[str]
    ranking_baselines: frozenset[str]
    requires_search: bool
    requires_separate_results: bool
    allowed_search_principals: frozenset[str] = frozenset()
    execution_kind: str = "strict"
    audit_kind: str = "strict"
    results_namespace: str = "results"
    requires_cutoff: bool = False
    requires_external_harness: bool = False

    def accepts_baseline(self, baseline: str) -> bool:
        return baseline in self.allowed_baselines

    def is_rankable_baseline(self, baseline: str) -> bool:
        return baseline in self.ranking_baselines

    def accepts_search_principal(self, principal: str) -> bool:
        return principal in self.allowed_search_principals


STRICT_PROFILE: Final = ProfileSpec(
    name="one_turn_strict",
    policy_version="one_turn_strict/v1",
    allowed_baselines=frozenset({"pydantic-ai", "langgraph", "direct_transport"}),
    ranking_baselines=frozenset({"pydantic-ai", "langgraph"}),
    requires_search=False,
    requires_separate_results=False,
)
AGENTIC_PROFILE: Final = ProfileSpec(
    name="agentic_web_search",
    policy_version="agentic_web_search/v3",
    allowed_baselines=frozenset({"pydantic-ai", "langgraph"}),
    ranking_baselines=frozenset({"pydantic-ai", "langgraph"}),
    requires_search=True,
    requires_separate_results=True,
    allowed_search_principals=frozenset({"pydantic-ai", "langgraph"}),
    execution_kind="agentic_web_search",
    audit_kind="search",
    results_namespace="results-agentic-web-search",
)
DEEP_RESEARCH_PROFILE: Final = ProfileSpec(
    name="deep_research",
    policy_version="deep_research/v2",
    allowed_baselines=frozenset({"financeharness"}),
    ranking_baselines=frozenset({"financeharness"}),
    requires_search=True,
    requires_separate_results=True,
    allowed_search_principals=frozenset({"financeharness", "odr", "gpt-researcher"}),
    execution_kind="deep_research",
    audit_kind="research",
    results_namespace="results-deep-research",
    requires_cutoff=True,
    requires_external_harness=True,
)

PROFILES: Final[dict[str, ProfileSpec]] = {
    STRICT_PROFILE.name: STRICT_PROFILE,
    AGENTIC_PROFILE.name: AGENTIC_PROFILE,
    DEEP_RESEARCH_PROFILE.name: DEEP_RESEARCH_PROFILE,
}

BASELINE_ALIASES: Final[dict[str, str]] = {
    "pydantic-ai": "pydantic-ai",
    "pydantic_ai": "pydantic-ai",
    "langgraph": "langgraph",
    "direct_transport": "direct_transport",
    "direct-transport": "direct_transport",
    "financeharness": "financeharness",
    "finance-harness": "financeharness",
}


def get_profile(name: str) -> ProfileSpec:
    """Return a registered profile or fail before any external side effect."""

    try:
        return PROFILES[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(PROFILES))
        raise ProfileRegistryError(
            f"Unsupported profile {name!r}; choose one of: {allowed}."
        ) from exc


def normalize_baseline(name: str) -> str:
    """Normalize only the retained public baseline spellings."""

    try:
        return BASELINE_ALIASES[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(BASELINE_ALIASES.values())))
        raise ProfileRegistryError(
            f"Unsupported baseline {name!r}; choose one of: {allowed}."
        ) from exc


def supported_profiles() -> tuple[str, ...]:
    return tuple(PROFILES)


def supported_baselines(profile: str) -> tuple[str, ...]:
    return tuple(sorted(get_profile(profile).allowed_baselines))


def ranking_baselines(profile: str) -> tuple[str, ...]:
    return tuple(sorted(get_profile(profile).ranking_baselines))


def execution_kinds() -> tuple[str, ...]:
    """Return the exhaustive execution routes known to the harness."""

    return tuple(spec.execution_kind for spec in PROFILES.values())
