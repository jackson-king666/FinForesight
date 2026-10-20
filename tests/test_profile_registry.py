"""Profile and baseline registry contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark_harness.config import HF_DATASET_REPO_ID, HF_DATASET_REVISION, load_experiment_config
from benchmark_harness.profile_registry import (
    ProfileRegistryError,
    get_profile,
    normalize_baseline,
    ranking_baselines,
    supported_baselines,
    supported_profiles,
)
from benchmark_harness.runner import BenchmarkRunner, RunnerError, _normalize_baseline


def _mock_config(tmp_path: Path):
    path = tmp_path / "strict.toml"
    path.write_text(
        f'''[dataset]
source = "huggingface"
repo_id = "{HF_DATASET_REPO_ID}"
revision = "{HF_DATASET_REVISION}"

[model]
mode = "mock"

[execution]
input_policy = "one_turn_strict"

[trace]
results_dir = "results"
''',
        encoding="utf-8",
    )
    return load_experiment_config(path)


def test_registry_exposes_exactly_the_three_public_profiles() -> None:
    assert set(supported_profiles()) == {
        "one_turn_strict",
        "agentic_web_search",
        "deep_research",
    }
    assert len(supported_profiles()) == 3
    assert get_profile("one_turn_strict").name == "one_turn_strict"
    assert get_profile("agentic_web_search").name == "agentic_web_search"
    assert get_profile("deep_research").name == "deep_research"

    with pytest.raises(ProfileRegistryError, match="Unsupported profile"):
        get_profile("removed_profile")


def test_profile_baseline_compatibility_keeps_control_out_of_ranking() -> None:
    strict = get_profile("one_turn_strict")
    agentic = get_profile("agentic_web_search")

    assert strict.accepts_baseline("direct_transport") is True
    assert strict.is_rankable_baseline("direct_transport") is False
    assert ranking_baselines("one_turn_strict") == ("langgraph", "pydantic-ai")
    assert agentic.accepts_baseline("direct_transport") is False


def test_removed_baseline_rejects_before_any_external_side_effect() -> None:
    with pytest.raises(RunnerError, match="Unsupported baseline"):
        _normalize_baseline("open_deep_research")
    with pytest.raises(ProfileRegistryError, match="Unsupported baseline"):
        normalize_baseline("gpt_researcher")


def test_deep_search_principals_do_not_widen_public_baselines() -> None:
    deep = get_profile("deep_research")
    assert supported_baselines("deep_research") == ("financeharness",)
    assert ranking_baselines("deep_research") == ("financeharness",)
    assert deep.allowed_search_principals == frozenset({"financeharness", "odr", "gpt-researcher"})
    for principal in ("financeharness", "odr", "gpt-researcher"):
        assert deep.accepts_search_principal(principal) is True
        assert deep.accepts_baseline(principal) is False or principal == "financeharness"
    for alias in ("open_deep_research", "gpt_researcher", "finance-harness"):
        assert deep.accepts_search_principal(alias) is False
    with pytest.raises(ProfileRegistryError, match="Unsupported baseline"):
        normalize_baseline("odr")
    with pytest.raises(ProfileRegistryError, match="Unsupported baseline"):
        normalize_baseline("gpt-researcher")
    assert get_profile("one_turn_strict").allowed_search_principals == frozenset()
    assert get_profile("agentic_web_search").allowed_search_principals == frozenset(
        {"pydantic-ai", "langgraph"}
    )


def test_runner_rejects_removed_baseline_before_dataset_or_model_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _mock_config(tmp_path)
    monkeypatch.setattr(
        "benchmark_harness.runner._make_dataset_source",
        lambda _config: pytest.fail("dataset source must not be created"),
    )
    monkeypatch.setattr(
        "benchmark_harness.runner._make_search_gateway",
        lambda _config: pytest.fail("search gateway must not be created"),
    )

    with pytest.raises(RunnerError, match="Unsupported baseline"):
        BenchmarkRunner(config).run(
            baselines=["open_deep_research"], isolated=False, run_id="removed-profile"
        )
