"""Phase 0 characterization locks for the legacy benchmark surface."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

from benchmark_harness.cli import _experiment_command, _parser
from benchmark_harness.config import ExperimentConfig
from benchmark_harness.reporting.compare import _REQUIRED_ARTIFACTS

REPOSITORY_ROOT = Path(__file__).parents[1]
EVALUATOR_ROOT = REPOSITORY_ROOT / "benchmark_harness" / "reference_evaluation"
FROZEN_EVALUATOR_SHA256 = "c911d8146a0ca02a31ec80595b13611c94f097af21d60525bc08f447c446d50b"
CONFIG_HASH = "2ff60922848b59bd0b2c19a1b4ef6acf76fce6d9da59c6e261653d0684c58d9c"


def test_cli_commands_and_default_config_hash_remain_characterized() -> None:
    parser = _parser()
    command_action = next(action for action in parser._actions if action.dest == "command")

    assert set(command_action.choices) == {
        "preflight",
        "data-sync",
        "external-sync",
        "run",
        "report",
        "search-service",
        "experiment",
        "leakage-calibration",
        "qc-assess",
        "qc-pack",
        "qc-validate",
        "path-b-v2-validate",
        "path-b-v3-review-pack",
        "path-b-v3-r3-review-pack",
        "path-b-v3-mini-reseal",
        "path-b-v3-validate",
        "path-b-v3-run",
    }
    leakage_action = command_action.choices["leakage-calibration"]
    leakage_subcommands = next(
        action for action in leakage_action._actions if action.dest == "leakage_calibration_action"
    )
    assert set(leakage_subcommands.choices) == {
        "validate",
        "pilot",
        "pilot-v4",
        "pilot-v5",
        "scaffold-v5",
        "pilot-v6",
        "scaffold-v6",
        "freeze-400",
        "label-pilot",
    }
    config = ExperimentConfig()
    assert config.config_hash == CONFIG_HASH
    public_dict = config.public_dict()
    assert "api_key" not in public_dict["model"]
    assert "secret_key" not in public_dict["search"]


def test_runner_and_compare_share_the_legacy_required_artifact_contract() -> None:
    assert _REQUIRED_ARTIFACTS == (
        "predictions.jsonl",
        "metrics_summary.json",
        "run_config.json",
        "provenance.json",
        "trace_events.jsonl",
        "effective_requests.jsonl",
        "network_ledger.jsonl",
    )


def test_frozen_evaluator_hash_and_manifest_lineage_remain_unchanged() -> None:
    engine_path = EVALUATOR_ROOT / "engine.py"
    manifest = json.loads((EVALUATOR_ROOT / "EVALUATOR_MANIFEST.json").read_text(encoding="utf-8"))

    assert hashlib.sha256(engine_path.read_bytes()).hexdigest() == FROZEN_EVALUATOR_SHA256
    assert manifest["source"]["commit"] == "6c3ba942a33bb1ce941a88e4b5a78a11c3153f32"
    assert manifest["frozen_file"] == {
        "path": "engine.py",
        "sha256": FROZEN_EVALUATOR_SHA256,
    }


def test_offline_experiment_report_does_not_require_a_manifest(tmp_path: Path) -> None:
    (tmp_path / "c1.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("fixture\n", encoding="utf-8")

    assert _experiment_command(
        Namespace(experiment_action="report", analysis_dir=str(tmp_path))
    ) == {"status": "ready", "files": ["c1.json"]}
