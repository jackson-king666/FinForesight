"""Pinned v0.6 configuration-schema regression coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark_harness.config import (
    DATASET_EXPECTED_COUNTS,
    HF_DATASET_REPO_ID,
    HF_DATASET_REVISION,
    ConfigError,
    load_experiment_config,
)
from benchmark_harness.dataset_sources.huggingface import (
    DEFAULT_HF_REPO_ID,
    DEFAULT_HF_REVISION,
    EXPECTED_RECORD_COUNTS,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _config(
    *,
    policy: str = "one_turn_strict",
    results_dir: str = "results",
    content_mode: str | None = None,
) -> str:
    trace_content_mode = f'content_mode = "{content_mode}"\n' if content_mode else ""
    return f'''[dataset]
source = "huggingface"
repo_id = "{HF_DATASET_REPO_ID}"
revision = "{HF_DATASET_REVISION}"
data_dir = "data"

[model]
mode = "mock"

[execution]
input_policy = "{policy}"

[trace]
{trace_content_mode}results_dir = "{results_dir}"
'''


def test_hub_defaults_lock_v03_source_count_matrix_and_cache_layout(tmp_path: Path) -> None:
    config = load_experiment_config(_write(tmp_path / "strict.toml", _config()))

    assert config.dataset.source == "huggingface"
    assert config.dataset.repo_id == HF_DATASET_REPO_ID
    assert config.dataset.revision == HF_DATASET_REVISION
    assert config.dataset.expected_counts == DATASET_EXPECTED_COUNTS
    assert config.dataset.cache_dir == tmp_path / "data" / ".huggingface"
    assert config.evaluation.require_golden_parity is True
    assert config.model.retries == 2


def test_config_and_loader_declarations_share_the_v06_pin_and_count_matrix() -> None:
    assert HF_DATASET_REPO_ID == DEFAULT_HF_REPO_ID
    assert HF_DATASET_REVISION == DEFAULT_HF_REVISION
    assert DATASET_EXPECTED_COUNTS == EXPECTED_RECORD_COUNTS
    assert sum(DATASET_EXPECTED_COUNTS.values()) == 339


def test_agentic_profile_requires_a_separate_results_directory(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="separate trace.results_dir"):
        load_experiment_config(
            _write(
                tmp_path / "agentic.toml",
                _config(policy="agentic_web_search", results_dir="results"),
            )
        )


def test_removed_profile_is_rejected_before_execution(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Unsupported profile"):
        load_experiment_config(
            _write(
                tmp_path / "removed-profile.toml",
                _config(
                    policy="deep_research_cutoff",
                    results_dir="results-legacy-research",
                ),
            )
        )


def test_removed_nested_research_configuration_is_rejected(tmp_path: Path) -> None:
    text = _config() + '\n[deep_research]\ncontract_version = "legacy"\n'
    with pytest.raises(ConfigError, match="Unsupported top-level configuration"):
        load_experiment_config(_write(tmp_path / "removed-config.toml", text))


def test_golden_parity_gate_cannot_be_disabled(tmp_path: Path) -> None:
    text = _config() + "\n[evaluation]\nrequire_golden_parity = false\n"

    with pytest.raises(ConfigError, match="cannot be disabled"):
        load_experiment_config(_write(tmp_path / "strict.toml", text))


def test_agentic_search_uses_service_config_and_rejects_legacy_search(tmp_path: Path) -> None:
    config = load_experiment_config(
        _write(
            tmp_path / "agentic.toml",
            _config(policy="agentic_web_search", results_dir="results-agentic"),
        )
    )

    assert config.search_service is not None
    assert config.search_service.base_url == "http://127.0.0.1:8765"

    exa_config = _config(policy="agentic_web_search", results_dir="results-agentic") + (
        '\n[search]\nprovider = "exa"\n'
    )
    with pytest.raises(ConfigError, match="legacy non-strict"):
        load_experiment_config(_write(tmp_path / "exa.toml", exa_config))


def test_aliyun_model_profile_defaults_and_explicit_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_text = _config().replace('mode = "mock"', 'mode = "openai_compatible"')
    config = load_experiment_config(_write(tmp_path / "openai.toml", config_text))

    for name in (
        "BENCHMARK_MODEL_ENDPOINT",
        "BENCHMARK_MODEL_NAME",
        "BENCHMARK_MODEL_API_KEY",
        "BENCHMARK_MODEL_TIMEOUT",
        "ALIYUN_BASE_URL",
        "ALIYUN_MODEL",
        "ALIYUN_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_MODEL",
        "BAILIAN_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ALIYUN_BASE_URL", "https://dashscope.example.test/compatible-mode/v1")
    monkeypatch.setenv("ALIYUN_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-secret")
    monkeypatch.setenv("ALIYUN_TIMEOUT", "600")
    monkeypatch.setenv("BENCHMARK_MODEL_RETRIES", "3")

    profile = config.resolve_model(require_ready=True)
    assert profile is not None
    assert profile.endpoint == "https://dashscope.example.test/compatible-mode/v1"
    assert profile.model_name == "deepseek-v4-flash"
    assert profile.api_key == "dashscope-secret"
    assert profile.timeout_seconds == 600.0
    assert profile.retries == 3

    monkeypatch.setenv("BENCHMARK_MODEL_ENDPOINT", "https://explicit.example.test/v1")
    monkeypatch.setenv("BENCHMARK_MODEL_NAME", "explicit-model")
    monkeypatch.setenv("BENCHMARK_MODEL_API_KEY", "explicit-secret")
    monkeypatch.setenv("BENCHMARK_MODEL_TIMEOUT", "30")

    overridden = config.resolve_model(require_ready=True)
    assert overridden is not None
    assert overridden.endpoint == "https://explicit.example.test/v1"
    assert overridden.model_name == "explicit-model"
    assert overridden.api_key == "explicit-secret"
    assert overridden.timeout_seconds == 30.0
