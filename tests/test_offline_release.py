"""Release-specific offline and installation contracts."""

from __future__ import annotations

import importlib.util
import json
import socket
from importlib.metadata import distribution
from importlib.resources import files
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("offline_demo", ROOT / "scripts/offline_demo.py")
assert SPEC and SPEC.loader
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


def test_offline_demo_preserves_artifacts_without_network(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access forbidden")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    for key in list(__import__("os").environ):
        if any(part in key for part in ("API_KEY", "VOLC", "SEARCH_FILTER", "BENCHMARK_MODEL")):
            monkeypatch.delenv(key)
    output = tmp_path / "demo"
    manifest = DEMO.run(output)
    predictions = [
        json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()
    ]
    assert {row["category"] for row in predictions} == {"A1", "A2", "B", "C", "D", "E"}
    assert {row["variant"] for row in predictions if row["category"] == "A2"} >= {"F", "T", "H"}
    assert manifest["synthetic"] is True
    for name, digest in manifest["files"].items():
        assert DEMO.file_sha256(output / name) == digest
    summary = json.loads((output / "metrics_summary.json").read_text())
    assert set(summary["forecasting"]["by_category"]) == {"A1", "A2", "B", "C"}
    assert set(summary["diagnostic"]["by_category"]) == {"D", "E"}
    before = (output / "predictions.jsonl").read_bytes()
    with pytest.raises(ValueError, match="empty directory"):
        DEMO.run(output)
    assert (output / "predictions.jsonl").read_bytes() == before


def test_malformed_and_missing_fixtures_do_not_create_output(tmp_path):
    fixture = tmp_path / "bad.json"
    fixture.write_text('{"synthetic":true,"records":[{}]}')
    with pytest.raises(ValueError, match="missing fixture field"):
        DEMO.run(tmp_path / "output", fixture)
    assert not (tmp_path / "output").exists()
    with pytest.raises(FileNotFoundError):
        DEMO.run(tmp_path / "output", tmp_path / "missing.json")


def test_installed_resources_and_entrypoint_aliases():
    evaluator = files("benchmark_harness.reference_evaluation")
    for name in ("EVALUATOR_MANIFEST.json", "NOTICE.md", "engine.py"):
        assert evaluator.joinpath(name).is_file()
    resources = files("benchmark_harness.search_service")
    assert any(item.name.endswith(".json") for item in resources.iterdir())
    entries = {entry.name: entry.value for entry in distribution("finforesight").entry_points}
    assert entries["finforesight"] == entries["benchmark-harness"] == "benchmark_harness.cli:main"
