"""Verify resources and console aliases from an installed distribution."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import distribution
from importlib.resources import files


def main() -> None:
    dist = distribution("finforesight")
    entries = {entry.name: entry.value for entry in dist.entry_points}
    assert entries["finforesight"] == entries["benchmark-harness"] == "benchmark_harness.cli:main"
    root = files("benchmark_harness.reference_evaluation")
    manifest = json.loads(root.joinpath("EVALUATOR_MANIFEST.json").read_text())
    assert root.joinpath("NOTICE.md").is_file()
    assert (
        hashlib.sha256(root.joinpath("engine.py").read_bytes()).hexdigest()
        == manifest["frozen_file"]["sha256"]
    )
    service = files("benchmark_harness.search_service")
    assert any(path.name.endswith(".json") for path in service.iterdir())
    print("Installed evaluator, notices, service resources and CLI aliases verified")


if __name__ == "__main__":
    main()
