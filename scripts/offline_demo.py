"""Evaluate fixed synthetic responses without model, search, or dataset access."""

from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path

from benchmark_harness.artifacts import ArtifactStore, file_sha256
from benchmark_harness.reference_evaluation import BundledEvaluationEngine

DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "examples/offline/fixtures.json"


def run(output: Path, fixture: Path = DEFAULT_FIXTURE) -> dict:
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    if payload.get("synthetic") is not True or not payload.get("records"):
        raise ValueError("expected nonempty synthetic fixture")
    engine = BundledEvaluationEngine()
    predictions = []
    for record in payload["records"]:
        for field in ("task_id", "category", "prompt", "cutoff_date", "ground_truth", "response"):
            if field not in record:
                raise ValueError(f"missing fixture field: {field}")
        if record["category"] not in {"A1", "A2", "B", "C", "D", "E"}:
            raise ValueError("unknown fixture category")
        parsed = engine.parse_prediction(record["category"], record["response"])
        predictions.append(
            {
                "task_id": record["task_id"],
                "category": record["category"],
                "variant": record.get("variant"),
                "raw_response": record["response"],
                "parsed": parsed,
                "metrics": engine.evaluate_prediction(record["category"], parsed, record),
                "latency_seconds": 0.0,
                "synthetic": True,
            }
        )
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be absent or an empty directory")
    store = ArtifactStore(output.parent, output.name)
    store.write_jsonl(output / "predictions.jsonl", predictions)
    groups = {}
    for name, categories in (("forecasting", {"A1", "A2", "B", "C"}), ("diagnostic", {"D", "E"})):
        groups[name] = engine.summarize_results(
            [p for p in predictions if p["category"] in categories],
            [],
            agent="synthetic-fixture",
            model="fixed-responses",
        )
    summary = {"synthetic": True, "metric_version": "frozen-reference/v1", **groups}
    store.write_json(output / "metrics_summary.json", summary)
    evaluator = files("benchmark_harness.reference_evaluation")
    identity = json.loads(evaluator.joinpath("EVALUATOR_MANIFEST.json").read_text())
    manifest = {
        "synthetic": True,
        "fixture_sha256": file_sha256(fixture),
        "evaluator": identity["frozen_file"],
        "metric_version": "frozen-reference/v1",
        "files": {
            name: file_sha256(output / name)
            for name in ("predictions.jsonl", "metrics_summary.json")
        },
    }
    store.write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args()
    try:
        run(args.output, args.fixture)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"offline demo: {exc}\n")


if __name__ == "__main__":
    main()
