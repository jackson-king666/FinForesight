"""Golden parity for the bundled reference evaluator freeze."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmark_harness.reference_evaluation import BundledEvaluationEngine

SOURCE_COMMIT = "6c3ba942a33bb1ce941a88e4b5a78a11c3153f32"

PACKAGE_ROOT = Path(__file__).parents[1] / "benchmark_harness" / "reference_evaluation"


def _source_root() -> Path:
    configured = os.environ.get("BENCHMARK_REFERENCE_EVALUATION_SOURCE_ROOT")
    if not configured:
        pytest.skip("set BENCHMARK_REFERENCE_EVALUATION_SOURCE_ROOT for optional parity")
    root = Path(configured).expanduser()
    if not root.is_dir():
        pytest.skip("authoritative reference-evaluation checkout is unavailable")
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == SOURCE_COMMIT
    return root


def _source_operation(root: Path, payload: dict[str, Any]) -> Any:
    script = """
import json
import sys

from src.agents.mock_agent import mock_agent
from src.run_benchmark import evaluate_prediction, parse_prediction, summarize_results

payload = json.loads(sys.stdin.read())
operation = payload["operation"]
if operation == "vectors":
    output = []
    for vector in payload["vectors"]:
        parsed = parse_prediction(vector["category"], vector["response"])
        output.append({
            "parsed": parsed,
            "metrics": evaluate_prediction(vector["category"], parsed, vector["record"]),
        })
elif operation == "summary":
    output = summarize_results(
        payload["predictions"],
        payload["skipped"],
        agent=payload["agent"],
        model=payload["model"],
    )
elif operation == "mock":
    output = [mock_agent(record["prompt"], record) for record in payload["records"]]
else:
    raise ValueError(operation)
print(json.dumps(output, ensure_ascii=False, allow_nan=False, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        input=json.dumps(payload, ensure_ascii=False),
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _vectors() -> list[dict[str, Any]]:
    return [
        {
            "category": "A1",
            "response": '{"bull": 120, "base": 100, "bear": 90, "reversion_horizon": "3-6个月"}',
            "record": {
                "category": "A1",
                "ground_truth": {"actual_prices": {"30": 95, "90": 105, "180": 110, "365": 130}},
            },
        },
        {
            "category": "A2",
            "response": '["A", "B", "C", "D", "E", "F"]',
            "record": {
                "category": "A2",
                "variant": "F",
                "seed": {"stock_list": [{"code": code} for code in "ABCDEF"]},
                "ground_truth": {
                    "actual_returns": {
                        "A": 0.3,
                        "B": 0.2,
                        "C": 0.1,
                        "D": -0.1,
                        "E": -0.2,
                        "F": -0.3,
                    },
                    "actual_ranking": ["A", "B", "C", "D", "E", "F"],
                },
            },
        },
        # The evaluator is shared across the F/T/H A2 task variants, but retain
        # representative T and H records in the parity corpus so a future
        # freeze cannot accidentally narrow the supported dataset surface.
        {
            "category": "A2",
            "response": '{"ranking": ["T3", "T2", "T1"]}',
            "record": {
                "category": "A2",
                "variant": "T",
                "seed": {
                    "signal_variant": "T",
                    "stock_list": [{"code": code} for code in ("T1", "T2", "T3")],
                },
                "ground_truth": {
                    "actual_returns": {"T1": -0.1, "T2": 0.1, "T3": 0.2},
                    "actual_ranking": ["T3", "T2", "T1"],
                },
            },
        },
        {
            "category": "A2",
            "response": '["H2", "H1", "H3"]',
            "record": {
                "category": "A2",
                "variant": "H",
                "seed": {
                    "signal_variant": "H",
                    "stock_list": [{"code": code} for code in ("H1", "H2", "H3")],
                },
                "ground_truth": {
                    "actual_returns": {"H1": 0.0, "H2": 0.2, "H3": -0.2},
                    "actual_ranking": ["H2", "H1", "H3"],
                },
            },
        },
        # Keep parser failure behavior in the golden corpus too: one raw
        # garbage response and one syntactically valid but malformed ranking.
        {
            "category": "A2",
            "response": "not JSON at all",
            "record": {
                "category": "A2",
                "variant": "T",
                "seed": {"stock_list": [{"code": code} for code in ("T1", "T2", "T3")]},
                "ground_truth": {
                    "actual_returns": {"T1": 0.1, "T2": 0.0, "T3": -0.1},
                    "actual_ranking": ["T1", "T2", "T3"],
                },
            },
        },
        {
            "category": "A2",
            "response": '{"ranking": ["H1", 7, "H3"]}',
            "record": {
                "category": "A2",
                "variant": "H",
                "seed": {"stock_list": [{"code": code} for code in ("H1", "H2", "H3")]},
                "ground_truth": {
                    "actual_returns": {"H1": 0.1, "H2": 0.0, "H3": -0.1},
                    "actual_ranking": ["H1", "H2", "H3"],
                },
            },
        },
        {
            "category": "B",
            "response": '{"direction": "up", "probability_up": 0.8}',
            "record": {
                "category": "B",
                "variant": "earnings",
                "ground_truth": {"actual_direction": "up"},
            },
        },
        {
            "category": "C",
            "response": '{"predicted_value": 90, "metric": "margin"}',
            "record": {
                "category": "C",
                "seed": {"metric_name": "margin"},
                "ground_truth": {"future_value": 100},
            },
        },
        {
            "category": "D",
            "response": '{"direction": "positive", "reasoning": "test"}',
            "record": {
                "category": "D",
                "time_band": "T1",
                "ground_truth": {"logic_direction": "positive"},
            },
        },
        {
            "category": "E",
            "response": '{"formula_used": "x / y", "answer": 10, "unit": "%"}',
            "record": {
                "category": "E",
                "ground_truth": {
                    "correct_answer": 10,
                    "correct_formula": "x / y",
                    "answer_unit": "%",
                },
            },
        },
        {
            "category": "B",
            "response": '{"direction": "sideways", "probability_up": 0.5}',
            "record": {
                "category": "B",
                "variant": "earnings",
                "ground_truth": {"actual_direction": "up"},
            },
        },
    ]


def test_frozen_parser_and_evaluator_match_authoritative_checkout() -> None:
    root = _source_root()
    vectors = _vectors()
    source_results = _source_operation(root, {"operation": "vectors", "vectors": vectors})
    bundled = BundledEvaluationEngine()

    frozen_results = []
    for vector in vectors:
        parsed = bundled.parse_prediction(vector["category"], vector["response"])
        frozen_results.append(
            {
                "parsed": parsed,
                "metrics": bundled.evaluate_prediction(
                    vector["category"], parsed, vector["record"]
                ),
            }
        )

    assert frozen_results == source_results


def test_frozen_summarizer_matches_authoritative_checkout() -> None:
    root = _source_root()
    bundled = BundledEvaluationEngine()
    predictions = []
    for index, vector in enumerate(_vectors()[:6]):
        parsed = bundled.parse_prediction(vector["category"], vector["response"])
        predictions.append(
            {
                "task_id": f"golden-{index}",
                "category": vector["category"],
                "variant": vector["record"].get("variant"),
                "parsed": parsed,
                "metrics": bundled.evaluate_prediction(
                    vector["category"], parsed, vector["record"]
                ),
                "latency_seconds": 0.01 + index,
            }
        )
    skipped = [{"task_id": "golden-skipped"}]
    payload = {
        "operation": "summary",
        "predictions": predictions,
        "skipped": skipped,
        "agent": "golden",
        "model": "golden-model",
    }
    assert bundled.summarize_results(
        predictions, skipped, agent="golden", model="golden-model"
    ) == _source_operation(root, payload)


def test_frozen_mock_response_matches_authoritative_smoke_helper() -> None:
    root = _source_root()
    records = []
    for index, vector in enumerate(_vectors()[:6]):
        record = {"task_id": f"mock-{index}", "prompt": "unused", **vector["record"]}
        records.append(record)
    bundled = BundledEvaluationEngine()
    assert [bundled.mock_response(record) for record in records] == _source_operation(
        root, {"operation": "mock", "records": records}
    )


def test_summary_preserves_time_band_grouping() -> None:
    bundled = BundledEvaluationEngine()
    record = {
        "task_id": "D-time-band",
        "category": "D",
        "time_band": "T2",
        "ground_truth": {"logic_direction": "positive"},
    }
    parsed = bundled.parse_prediction("D", '{"direction": "positive", "reasoning": "test"}')
    metrics = bundled.evaluate_prediction("D", parsed, record)
    summary = bundled.summarize_results(
        [
            {
                "task_id": record["task_id"],
                "category": "D",
                "parsed": parsed,
                "metrics": metrics,
                "latency_seconds": 0.01,
            }
        ],
        [],
        agent="test",
        model="test",
    )

    assert summary["by_category"]["D"]["by_time_band"]["T2"]["count"] == 1


def test_evaluator_manifest_pins_freeze_and_source_hashes() -> None:
    manifest = json.loads((PACKAGE_ROOT / "EVALUATOR_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["source"]["commit"] == SOURCE_COMMIT
    assert manifest["source"]["license"] == "CC-BY-4.0"
    assert manifest["notice_file"] == "NOTICE.md"
    assert manifest["frozen_file"] == {
        "path": "engine.py",
        "sha256": hashlib.sha256((PACKAGE_ROOT / "engine.py").read_bytes()).hexdigest(),
    }
    assert {entry["path"] for entry in manifest["source_files"]} >= {
        "src/run_benchmark.py",
        "src/parsers/common.py",
        "src/evaluators/a1_evaluator.py",
        "src/evaluators/e_evaluator.py",
    }
