"""DatasetBridge contract tests using a disposable external dataset checkout."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from benchmark_harness.dataset_bridge import (
    DATASET_ROOT_ENV,
    DatasetBridge,
    DatasetImportError,
    DatasetRevisionError,
    DatasetRootError,
    DatasetValidationError,
)


@pytest.fixture(autouse=True)
def isolate_upstream_src_imports(monkeypatch: pytest.MonkeyPatch):
    """Keep fake ``src`` packages from leaking between bridge tests."""

    original_path = list(sys.path)
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "src" or name.startswith("src.")
    }
    for name in list(original_modules):
        monkeypatch.delitem(sys.modules, name, raising=False)

    yield

    for name in [name for name in sys.modules if name == "src" or name.startswith("src.")]:
        sys.modules.pop(name, None)
    sys.modules.update(original_modules)
    sys.path[:] = original_path


def make_fake_dataset(
    tmp_path: Path,
    *,
    validation_exit_code: int = 0,
    inject_outside_src_module: bool = False,
) -> tuple[Path, str]:
    """Build a tiny git-backed external dataset that exposes upstream APIs."""

    root = tmp_path / "dataset"
    (root / "scripts").mkdir(parents=True)
    (root / "src" / "agents").mkdir(parents=True)
    (root / "seeds").mkdir()

    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_name": "fake-benchmark-research",
                "version": "test-v1",
                "task_files": {"A1": {"file": "seeds/a1.jsonl", "status": "ready"}},
            }
        ),
        encoding="utf-8",
    )
    (root / "scripts" / "validate.py").write_text(
        "import sys\nprint('fake validation')\nraise SystemExit("
        + str(validation_exit_code)
        + ")\n",
        encoding="utf-8",
    )
    (root / "src" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "agents" / "mock_agent.py").write_text(
        "def mock_agent(prompt, record):\n"
        "    return '{\\\"mock_task_id\\\": \\\"' + record['task_id'] + '\\\"}'\n",
        encoding="utf-8",
    )
    outside_import = ""
    if inject_outside_src_module:
        outside_import = (
            "import sys\nimport types\n"
            "outside = types.ModuleType('src.outside')\n"
            "outside.__file__ = '/outside-root/src/outside.py'\n"
            "sys.modules['src.outside'] = outside\n\n"
        )
    run_benchmark_source = (
        "import json\n\n"
        + outside_import
        + (
            "def load_jsonl(path):\n"
            "    with open(path, encoding='utf-8') as file:\n"
            "        return [json.loads(line) for line in file if line.strip()]\n\n"
            "def parse_prediction(category, response):\n"
            "    return {'upstream_parser': category, 'raw': response, 'format_valid': True}\n\n"
            "def evaluate_prediction(category, parsed, record):\n"
            "    return {'upstream_evaluator': category, 'task_id': record['task_id'], "
            "'format_valid': parsed['format_valid']}\n\n"
            "def summarize_results(predictions, skipped, agent, model):\n"
            "    return {'upstream_summary': True, 'agent': agent, 'model': model, "
            "'count': len(predictions), 'skipped': len(skipped)}\n"
        )
    )
    (root / "src" / "run_benchmark.py").write_text(run_benchmark_source, encoding="utf-8")
    record = {
        "task_id": "A1-00001",
        "category": "A1",
        "variant": None,
        "time_band": "T1",
        "status": "ready",
        "prompt": "Return JSON.",
        "ground_truth": {"future": 1},
    }
    (root / "seeds" / "a1.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    run_git(["init", "--quiet"], root)
    run_git(["add", "."], root)
    run_git(
        [
            "-c",
            "user.name=Bridge Test",
            "-c",
            "user.email=bridge@example.test",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        root,
    )
    revision = run_git(["rev-parse", "HEAD"], root).stdout.strip()
    return root, revision


def run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_bridge_reuses_upstream_apis_and_preserves_records(tmp_path: Path) -> None:
    root, revision = make_fake_dataset(tmp_path)
    bridge = DatasetBridge(root, revision=revision)

    provenance = bridge.preflight()
    records = bridge.load_records(categories=["A1"])
    assert provenance["dataset_revision"] == revision
    assert provenance["manifest_version"] == "test-v1"
    assert provenance["validation"]["status"] == "passed"
    assert records == [
        {
            "task_id": "A1-00001",
            "category": "A1",
            "variant": None,
            "time_band": "T1",
            "status": "ready",
            "prompt": "Return JSON.",
            "ground_truth": {"future": 1},
        }
    ]

    parsed = bridge.parse_prediction("A1", '{"answer": 1}')
    metrics = bridge.evaluate_prediction("A1", parsed, records[0])
    summary = bridge.summarize_results(
        [
            {
                "task_id": records[0]["task_id"],
                "category": records[0]["category"],
                "parsed": parsed,
                "metrics": metrics,
            }
        ],
        [],
        agent="mock-smoke",
        model="not-a-performance-baseline",
    )

    assert parsed["upstream_parser"] == "A1"
    assert metrics["upstream_evaluator"] == "A1"
    assert summary["upstream_summary"] is True
    assert json.loads(bridge.mock_response(records[0])) == {"mock_task_id": "A1-00001"}
    # Parsed output, metrics, summary, and a canonical upstream-compatible row
    # all serialize without bridge-specific conversion.
    json.dumps({"parsed": parsed, "metrics": metrics, "summary": summary}, allow_nan=False)


def test_bridge_uses_environment_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, revision = make_fake_dataset(tmp_path)
    monkeypatch.setenv(DATASET_ROOT_ENV, str(root))

    bridge = DatasetBridge.from_environment(revision=revision)
    assert bridge.provenance()["dataset_root"] == str(root.resolve())


def test_preflight_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(DatasetRootError, match="does not exist"):
        DatasetBridge(tmp_path / "missing").preflight()


def test_preflight_rejects_wrong_revision(tmp_path: Path) -> None:
    root, _ = make_fake_dataset(tmp_path)

    with pytest.raises(DatasetRevisionError, match="revision mismatch"):
        DatasetBridge(root, revision="0" * 40).preflight()


def test_preflight_rejects_dirty_tracked_checkout(tmp_path: Path) -> None:
    root, revision = make_fake_dataset(tmp_path)
    (root / "src" / "run_benchmark.py").write_text(
        "# altered after pinned commit\n", encoding="utf-8"
    )

    with pytest.raises(DatasetRevisionError, match="tracked modifications"):
        DatasetBridge(root, revision=revision).preflight()


def test_preflight_rejects_failed_upstream_validation(tmp_path: Path) -> None:
    root, revision = make_fake_dataset(tmp_path, validation_exit_code=1)

    with pytest.raises(DatasetValidationError, match="validation failed"):
        DatasetBridge(root, revision=revision).preflight()


def test_preflight_rejects_preloaded_src_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, revision = make_fake_dataset(tmp_path)
    shadowed_src = types.ModuleType("src")
    shadowed_src.__file__ = str(tmp_path / "unrelated" / "src" / "__init__.py")
    monkeypatch.setitem(sys.modules, "src", shadowed_src)

    with pytest.raises(DatasetImportError, match="shadowing"):
        DatasetBridge(root, revision=revision).preflight()


def test_preflight_rejects_outside_src_module_loaded_by_upstream(tmp_path: Path) -> None:
    root, revision = make_fake_dataset(tmp_path, inject_outside_src_module=True)

    with pytest.raises(DatasetImportError, match="shadowing"):
        DatasetBridge(root, revision=revision).preflight()


def test_preflight_restores_import_globals_and_allows_isolated_reimports(tmp_path: Path) -> None:
    root, revision = make_fake_dataset(tmp_path)
    original_path = list(sys.path)
    first = DatasetBridge(root, revision=revision)
    first.preflight()

    assert sys.path == original_path
    assert not any(name == "src" or name.startswith("src.") for name in sys.modules)
    assert not list((root / "src").rglob("__pycache__"))
    # A second bridge cannot silently reuse the first bridge's parser/evaluator
    # module objects, while the first bridge's captured upstream functions work.
    second = DatasetBridge(root, revision=revision)
    second.preflight()
    assert first.parse_prediction("A1", "{}") == second.parse_prediction("A1", "{}")


@pytest.mark.skipif(
    not os.environ.get(DATASET_ROOT_ENV),
    reason="set BENCHMARK_DATASET_ROOT to run against a real external checkout",
)
def test_real_dataset_preflight_when_configured() -> None:
    """Exercise the pinned real dataset only when the caller explicitly opts in."""

    provenance = DatasetBridge.from_environment().preflight()
    assert provenance["dataset_revision"] == provenance["expected_dataset_revision"]
    assert provenance["manifest_version"] == "0.2.0"
