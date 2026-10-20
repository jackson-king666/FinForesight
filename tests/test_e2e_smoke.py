"""Non-performance mock smoke coverage for runner artifacts and comparison output."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from benchmark_harness.config import (
    DatasetConfig,
    ExecutionConfig,
    ExperimentConfig,
    ModelConfig,
    TraceConfig,
)
from benchmark_harness.runner import BenchmarkRunner


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _make_external_mock_dataset(tmp_path: Path) -> tuple[Path, str]:
    """Make a tiny immutable-looking upstream checkout with A1, C, and E seeds."""

    root = tmp_path / "external-dataset"
    (root / "scripts").mkdir(parents=True)
    (root / "src" / "agents").mkdir(parents=True)
    (root / "seeds").mkdir()
    records = {
        "A1": {
            "task_id": "A1-00001",
            "category": "A1",
            "variant": None,
            "time_band": "T1",
            "status": "ready",
            "prompt": "Return the A1 answer.",
            "ground_truth": {"mock_response": "A1 mock response"},
        },
        "C": {
            "task_id": "C-00001",
            "category": "C",
            "variant": "C1",
            "time_band": "T2",
            "status": "ready",
            "prompt": "Return the C answer.",
            "ground_truth": {"mock_response": "C mock response"},
        },
        "E": {
            "task_id": "E-00001",
            "category": "E",
            "variant": None,
            "time_band": "T3",
            "status": "ready",
            "prompt": "Return formula_used, answer, and unit.",
            "ground_truth": {"mock_response": "E mock response"},
        },
    }
    manifest = {
        "dataset_name": "mock-external-benchmark",
        "version": "test-v1",
        "task_files": {
            category: {"file": f"seeds/{category.lower()}.jsonl", "status": "ready"}
            for category in records
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "scripts" / "validate.py").write_text(
        "print('mock external dataset is valid')\n", encoding="utf-8"
    )
    (root / "src" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "agents" / "mock_agent.py").write_text(
        "def mock_agent(prompt, record):\n"
        "    assert prompt == record['prompt']\n"
        "    return record['ground_truth']['mock_response']\n",
        encoding="utf-8",
    )
    (root / "src" / "run_benchmark.py").write_text(
        "import json\n\n"
        "def load_jsonl(path):\n"
        "    with open(path, encoding='utf-8') as handle:\n"
        "        return [json.loads(line) for line in handle if line.strip()]\n\n"
        "def parse_prediction(category, response):\n"
        "    return {'format_valid': bool(response), 'category': category, 'raw': response}\n\n"
        "def evaluate_prediction(category, parsed, record):\n"
        "    return {\n"
        "        'category': category,\n"
        "        'format_valid': parsed['format_valid'],\n"
        "        'correct': parsed['raw'] == record['ground_truth']['mock_response'],\n"
        "    }\n\n"
        "def summarize_results(predictions, skipped, *, agent, model):\n"
        "    return {\n"
        "        'upstream_marker': 'fake-upstream-summarizer',\n"
        "        'agent': agent,\n"
        "        'model': model,\n"
        "        'prediction_count': len(predictions),\n"
        "        'skipped_count': len(skipped),\n"
        "    }\n",
        encoding="utf-8",
    )
    for category, record in records.items():
        (root / "seeds" / f"{category.lower()}.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )

    _git(["init", "--quiet"], root)
    _git(["add", "."], root)
    _git(
        [
            "-c",
            "user.name=Smoke Test",
            "-c",
            "user.email=smoke@example.test",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        root,
    )
    revision = _git(["rev-parse", "HEAD"], root).stdout.strip()
    return root, revision


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class _FixtureEvaluationEngine:
    """Test-only evaluator for a deliberately non-v0.6 migration fixture."""

    def parse_prediction(self, category: str, raw_text: str) -> dict[str, object]:
        return {"format_valid": bool(raw_text), "category": category, "raw": raw_text}

    def evaluate_prediction(
        self, category: str, parsed: dict[str, object], record: dict[str, object]
    ) -> dict[str, object]:
        ground_truth = record["ground_truth"]
        assert isinstance(ground_truth, dict)
        return {
            "category": category,
            "format_valid": parsed["format_valid"],
            "correct": parsed["raw"] == ground_truth["mock_response"],
        }

    def summarize_results(
        self,
        predictions: list[dict[str, object]],
        skipped: list[dict[str, object]],
        *,
        agent: str,
        model: str,
    ) -> dict[str, object]:
        return {
            "upstream_marker": "fake-upstream-summarizer",
            "agent": agent,
            "model": model,
            "prediction_count": len(predictions),
            "skipped_count": len(skipped),
        }

    def mock_response(self, record: dict[str, object]) -> str:
        ground_truth = record["ground_truth"]
        assert isinstance(ground_truth, dict)
        value = ground_truth["mock_response"]
        assert isinstance(value, str)
        return value


def test_mock_smoke_generates_non_performance_artifacts_and_comparison(tmp_path: Path) -> None:
    dataset_root, revision = _make_external_mock_dataset(tmp_path)
    results_dir = tmp_path / "results"
    config = ExperimentConfig(
        dataset=DatasetConfig(
            source="external_checkout",
            root=dataset_root,
            revision=revision,
        ),
        model=ModelConfig(mode="mock"),
        execution=ExecutionConfig(input_policy="one_turn_strict", concurrency=1),
        trace=TraceConfig(content_mode="full", results_dir=results_dir),
    )
    runner = BenchmarkRunner(config, evaluation_engine=_FixtureEvaluationEngine())

    preflight = runner.preflight()
    outcome = runner.run(
        baselines=["pydantic-ai", "langgraph"],
        categories=["A1", "C", "E"],
        run_id="mock-e2e",
        isolated=False,
    )

    assert preflight["dataset"]["dataset_revision"] == revision
    assert preflight["model_endpoint_ready"] is False
    assert outcome["run_id"] == "mock-e2e"
    assert {row["baseline"] for row in outcome["baselines"]} == {"pydantic-ai", "langgraph"}

    run_dir = results_dir / "mock-e2e"
    for baseline in ("pydantic-ai", "langgraph"):
        baseline_dir = run_dir / baseline
        expected_paths = {
            "predictions": baseline_dir / "predictions.jsonl",
            "summary": baseline_dir / "metrics_summary.json",
            "config": baseline_dir / "run_config.json",
            "traces": baseline_dir / "trace_events.jsonl",
            "ledger": baseline_dir / "network_ledger.jsonl",
            "requests": baseline_dir / "effective_requests.jsonl",
            "provenance": baseline_dir / "provenance.json",
        }
        assert all(path.is_file() for path in expected_paths.values())

        predictions = _read_jsonl(expected_paths["predictions"])
        traces = _read_jsonl(expected_paths["traces"])
        ledger = _read_jsonl(expected_paths["ledger"])
        requests = _read_jsonl(expected_paths["requests"])
        summary = json.loads(expected_paths["summary"].read_text(encoding="utf-8"))
        provenance = json.loads(expected_paths["provenance"].read_text(encoding="utf-8"))

        assert len(predictions) == 3
        assert all(row["mock_non_performance"] is True for row in predictions)
        assert all(row["comparable"] is True for row in predictions)
        assert all(row["model_call_count"] == 1 for row in predictions)
        assert all(row["tool_call_count"] == 0 for row in predictions)
        assert all(row["non_model_network_call_count"] == 0 for row in predictions)
        assert summary["harness"]["mock_non_performance"] is True
        assert summary["upstream_summary"]["upstream_marker"] == "fake-upstream-summarizer"
        assert summary["upstream_summary"]["prediction_count"] == 3
        assert provenance["mock_non_performance"] is True
        assert provenance["dataset_revision"] == revision
        assert len(ledger) == 3
        assert all(row["is_model_endpoint"] is True for row in ledger)
        assert all(row["policy"]["comparable"] is True for row in requests)
        assert all(
            row["messages"] == [{"role": "user", "content": row["messages"][0]["content"]}]
            for row in requests
        )
        event_types_by_task: dict[str, set[str]] = {}
        for event in traces:
            task_events = event_types_by_task.setdefault(str(event["task_id"]), set())
            task_events.add(str(event["event_type"]))
        assert all(
            {"task_started", "model_request", "model_response", "task_finished"} <= event_types
            for event_types in event_types_by_task.values()
        )

    comparison = json.loads((run_dir / "comparison.json").read_text(encoding="utf-8"))
    comparison_csv = (run_dir / "comparison.csv").read_text(encoding="utf-8")
    assert len(comparison["baselines"]) == 2
    assert all(row["mock_non_performance"] is True for row in comparison["baselines"])
    assert all(row["ranking_eligible"] is False for row in comparison["baselines"])
    assert "mock_non_performance" in comparison_csv
    legacy_b_caveat = "B has " + "0 ready seeds"
    assert all(legacy_b_caveat not in caveat for caveat in comparison["caveats"])


def test_progress_callback_reports_completed_tasks_without_breaking_a_run(tmp_path: Path) -> None:
    dataset_root, revision = _make_external_mock_dataset(tmp_path)
    updates = []
    config = ExperimentConfig(
        dataset=DatasetConfig(
            source="external_checkout",
            root=dataset_root,
            revision=revision,
        ),
        model=ModelConfig(mode="mock"),
        execution=ExecutionConfig(input_policy="one_turn_strict", concurrency=1),
        trace=TraceConfig(content_mode="full", results_dir=tmp_path / "progress-results"),
    )

    def progress(update) -> None:
        updates.append(update)
        # A display callback is intentionally non-critical and cannot fail a run.
        raise RuntimeError("display unavailable")

    outcome = BenchmarkRunner(config, evaluation_engine=_FixtureEvaluationEngine()).run(
        baselines=["pydantic-ai"],
        categories=["A1", "C"],
        run_id="progress-smoke",
        isolated=False,
        progress_callback=progress,
    )

    assert outcome["run_id"] == "progress-smoke"
    assert [(update.completed, update.total) for update in updates] == [(1, 2), (2, 2)]
    assert [update.task_id for update in updates] == ["A1-00001", "C-00001"]
    assert all(update.baseline == "pydantic-ai" for update in updates)
    assert all(update.elapsed_seconds >= 0 and update.eta_seconds >= 0 for update in updates)


def test_hash_only_smoke_keeps_upstream_scoring_in_memory_but_not_prompt_or_response_text(
    tmp_path: Path,
) -> None:
    dataset_root, revision = _make_external_mock_dataset(tmp_path)
    results_dir = tmp_path / "results"
    config = ExperimentConfig(
        dataset=DatasetConfig(
            source="external_checkout",
            root=dataset_root,
            revision=revision,
        ),
        model=ModelConfig(mode="mock"),
        execution=ExecutionConfig(input_policy="one_turn_strict", concurrency=1),
        trace=TraceConfig(content_mode="hash_only", results_dir=results_dir),
    )

    BenchmarkRunner(config, evaluation_engine=_FixtureEvaluationEngine()).run(
        baselines=["pydantic-ai"],
        categories=["A1"],
        run_id="hash-only-e2e",
        isolated=False,
    )

    baseline_dir = results_dir / "hash-only-e2e" / "pydantic-ai"
    prediction = _read_jsonl(baseline_dir / "predictions.jsonl")[0]
    summary = json.loads((baseline_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            baseline_dir / "predictions.jsonl",
            baseline_dir / "trace_events.jsonl",
            baseline_dir / "network_ledger.jsonl",
            baseline_dir / "effective_requests.jsonl",
        )
    )

    assert prediction["prompt"] is None
    assert prediction["raw_response"] is None
    assert isinstance(prediction["prompt_sha256"], str)
    assert isinstance(prediction["raw_response_sha256"], str)
    assert prediction["parsed"]["raw"] == {"sha256": prediction["raw_response_sha256"]}
    assert summary["upstream_summary"]["prediction_count"] == 1
    assert "Return the A1 answer." not in artifact_text
    assert "A1 mock response" not in artifact_text
