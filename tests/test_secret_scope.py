"""Regression tests for experiment dotenv and baseline-worker secret scope."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
from typing import Any

import benchmark_harness.cli as cli
from benchmark_harness.runner import _start_scrubbed_process


def _capture_environment(queue: Any) -> None:
    queue.put(dict(os.environ))


def test_experiment_dotenv_imports_only_agent_model_settings(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "\n".join(
            (
                "BENCHMARK_MODEL_ENDPOINT=https://model.example.test/v1",
                "ALIYUN_MODEL=deepseek-v4-flash",
                "DASHSCOPE_API_KEY=agent-secret",
                "BAILIAN_TIMEOUT=600",
                "VOLC_ACCESS_KEY_ID=volc-secret",
                "SEARCH_FILTER_MODEL_API_KEY=filter-secret",
                "HTTP_PROXY=http://proxy.example.test:8080",
                "UNRELATED_SETTING=ignored",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    for name in (
        "BENCHMARK_MODEL_ENDPOINT",
        "ALIYUN_MODEL",
        "DASHSCOPE_API_KEY",
        "BAILIAN_TIMEOUT",
        "VOLC_ACCESS_KEY_ID",
        "SEARCH_FILTER_MODEL_API_KEY",
        "HTTP_PROXY",
        "UNRELATED_SETTING",
    ):
        monkeypatch.delenv(name, raising=False)

    cli._load_experiment_dotenv()

    assert os.environ["BENCHMARK_MODEL_ENDPOINT"] == "https://model.example.test/v1"
    assert os.environ["ALIYUN_MODEL"] == "deepseek-v4-flash"
    assert os.environ["DASHSCOPE_API_KEY"] == "agent-secret"
    assert os.environ["BAILIAN_TIMEOUT"] == "600"
    assert "VOLC_ACCESS_KEY_ID" not in os.environ
    assert "SEARCH_FILTER_MODEL_API_KEY" not in os.environ
    assert "HTTP_PROXY" not in os.environ
    assert "UNRELATED_SETTING" not in os.environ


def test_scrubbed_baseline_spawn_drops_service_proxy_and_ambient_credentials(
    monkeypatch: Any,
) -> None:
    forbidden = {
        "VOLC_ACCESS_KEY_ID": "volc-secret",
        "SEARCH_FILTER_MODEL_API_KEY": "filter-secret",
        "HTTP_PROXY": "http://proxy.example.test:8080",
        "HTTPS_PROXY": "https://proxy.example.test:8443",
        "ALL_PROXY": "http://proxy.example.test:1080",
        "NO_PROXY": "127.0.0.1",
        "http_proxy": "http://proxy.example.test:8080",
        "https_proxy": "https://proxy.example.test:8443",
        "all_proxy": "http://proxy.example.test:1080",
        "no_proxy": "127.0.0.1",
        "SEARCH_SERVICE_ADMIN_TOKEN": "admin-secret",
        "ADMIN_TOKEN": "admin-secret",
        "OPENAI_API_KEY": "ambient-provider-secret",
    }
    for name, value in forbidden.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("BENCHMARK_MODEL_API_KEY", "agent-secret")
    monkeypatch.setenv("ALIYUN_API_KEY", "agent-secret")

    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_capture_environment, args=(queue,))
    _start_scrubbed_process(process)
    process.join(timeout=10)
    try:
        assert process.exitcode == 0
        try:
            child_environment = queue.get(timeout=2)
        except Empty as exc:  # pragma: no cover - diagnostic guard
            raise AssertionError("scrubbed worker did not return its environment") from exc
    finally:
        queue.close()
        queue.join_thread()

    for name in forbidden:
        assert name not in child_environment
    assert child_environment["BENCHMARK_MODEL_API_KEY"] == "agent-secret"
    assert child_environment["ALIYUN_API_KEY"] == "agent-secret"
