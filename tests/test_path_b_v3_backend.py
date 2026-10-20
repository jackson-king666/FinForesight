"""Boundary and resume tests for the injected Path B v3 backend."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from benchmark_harness.config import ResolvedModelProfile
from benchmark_harness.contracts import TaskInput
from benchmark_harness.experiment.path_b_v3_backend import (
    PATH_B_V3_LEDGER_FILENAME,
    PathBV3BackendError,
    PathBV3CurrentWebBackend,
    PathBV3SystemCellExecutor,
    _materialize_cells,
)
from benchmark_harness.experiment.path_b_v3_runner import (
    PathBV3CredentialStatus,
    PathBV3ExecutionRequest,
    PathBV3Plans,
)
from benchmark_harness.systems.contracts import SystemDescriptor


def _plans(tmp_path: Path) -> PathBV3Plans:
    budget = {
        "input_tokens": 100,
        "model_calls": 2,
        "other_tools": 2,
        "output_tokens": 100,
        "searches": 2,
        "visits": 2,
        "wall_clock_seconds": 30,
    }
    t1 = {
        "band": "T1",
        "budget_sha256": "b" * 64,
        "cell_id": "cell-t1",
        "execution_order": 0,
        "ranking_eligible": False,
        "retrieval_regime": "cutoff_controlled",
        "seed": 1,
        "system_id": "odr",
        "task_id": "task-1",
        "trial": 1,
    }
    cutoff = {
        "budget_sha256": "b" * 64,
        "cutoff_filter_enabled": True,
        "policy_identity": "cutoff_controlled/v1",
        "ranking_eligible": False,
        "retrieval_regime": "cutoff_controlled",
        "seed": 2,
        "system_id": "odr",
        "task_id": "task-1",
        "trial": 1,
    }
    unrestricted = {
        "budget_sha256": "b" * 64,
        "cutoff_filter_enabled": False,
        "policy_identity": "leakage_probe/v1",
        "ranking_eligible": False,
        "retrieval_regime": "unrestricted_current_web",
        "seed": 2,
        "system_id": "odr",
        "task_id": "task-1",
        "trial": 1,
    }
    manifest = {
        "experiment_id": "path-b-v3-test",
        "budget": budget,
        "budget_sha256": "b" * 64,
        "experiment_as_of": "2026-09-03",
    }
    root = tmp_path / "repo" / "artifacts" / "g3"
    return PathBV3Plans(
        root=root,
        g2_root=root.parent / "g2",
        index={"dataset_revision": "951", "dataset_tree_sha": "tree"},
        manifests={
            "t1_t2": manifest,
            "b_calibration": {
                **manifest,
                "experiment_id": "path-b-v3-test-b",
            },
            "cutoff_unrestricted": manifest,
        },
        plans={
            "t1_t2": {
                "plan_sha256": "1" * 64,
                "cells": [t1],
            },
            "b_calibration": {"cells": []},
            "cutoff_unrestricted": {
                "plan_sha256": "2" * 64,
                "cells": [cutoff, unrestricted],
            },
        },
        eligible_task_ids=("task-1",),
    )


def _request(plans: PathBV3Plans) -> PathBV3ExecutionRequest:
    return PathBV3ExecutionRequest(
        plans=plans,
        task_inputs={
            "task-1": TaskInput(
                task_id="task-1",
                prompt="score-free prompt",
                cutoff_date="2025-01-01",
            )
        },
        credentials=PathBV3CredentialStatus(
            model_api_key_present=True,
            search_admin_token_present=True,
            unrestricted_probe_present=True,
        ),
        results_roots=plans.results_roots,
    )


@dataclass
class _Executor:
    calls: list[tuple[str, str, int, Path]]
    fail_once: bool = False

    def execute(
        self,
        cell: Any,
        task: TaskInput,
        *,
        output_root: Path,
        attempt: int,
    ) -> dict[str, Any]:
        self.calls.append((cell.retrieval_regime, task.task_id, attempt, output_root))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("provider detail is never persisted")
        return {"raw_text": f"answer-{cell.cell_id}"}


def test_backend_uses_separate_namespaces_and_is_idempotent(tmp_path: Path) -> None:
    plans = _plans(tmp_path)
    executor = _Executor(calls=[])
    checkpoints: list[dict[str, Any]] = []
    backend = PathBV3CurrentWebBackend(
        executor,
        batch_size=1,
        checkpoint_callback=checkpoints.append,
    )

    first = backend(_request(plans))
    assert first["completed_cells"] == 3
    assert first["total_cells"] == 3
    assert len(executor.calls) == 3
    assert {path.name for _, _, _, path in executor.calls} == {
        "path-b-v3-t1-t2-r3",
        "cutoff_controlled",
        "unrestricted_current_web",
    }
    assert len(checkpoints) == 3
    assert all(item["ranking_eligible"] is False for item in checkpoints)
    assert all(item["namespaces"] for item in checkpoints)

    second = backend(_request(plans))
    assert second["completed_cells"] == 3
    assert second["attempts_written"] == 0
    assert len(executor.calls) == 3

    t1_ledger = plans.results_roots["t1_t2"] / PATH_B_V3_LEDGER_FILENAME
    leakage_ledger = (
        plans.results_roots["cutoff_unrestricted"] / "cutoff_controlled" / PATH_B_V3_LEDGER_FILENAME
    )
    assert t1_ledger.is_file()
    assert leakage_ledger.is_file()


def test_backend_resumes_failed_cell_with_a_new_attempt(tmp_path: Path) -> None:
    plans = _plans(tmp_path)
    executor = _Executor(calls=[], fail_once=True)
    backend = PathBV3CurrentWebBackend(executor, batch_size=3)

    with pytest.raises(PathBV3BackendError, match="cell failed"):
        backend(_request(plans))
    assert len(executor.calls) == 1

    result = backend(_request(plans))
    assert result["completed_cells"] == 3
    assert [attempt for _, _, attempt, _ in executor.calls] == [1, 2, 1, 1]


@pytest.mark.parametrize(
    "response",
    [
        {"gold": "forbidden"},
        {"output_path": "../escape"},
        {"ranking_eligible": True},
    ],
)
def test_backend_rejects_score_or_path_bearing_response(
    tmp_path: Path, response: dict[str, Any]
) -> None:
    plans = _plans(tmp_path)

    class BadExecutor:
        def execute(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
            return response

    with pytest.raises(PathBV3BackendError):
        PathBV3CurrentWebBackend(BadExecutor())(_request(plans))

    assert not list((plans.results_roots["t1_t2"]).glob("**/attempt-*.json"))


def test_system_executor_binds_task_regime_clock_and_attempt() -> None:
    plans = _plans(Path("/tmp"))
    cell = _materialize_cells(plans)[0]
    seen: list[tuple[str, str, str, int, date | None]] = []

    descriptor = SystemDescriptor(
        system_id="odr",
        framework_id="open_deep_research",
        framework_revision="framework",
        strategy_id="odr-deep-research",
        strategy_revision="strategy",
        backbone_id="openai:gpt-4.1-2025-04-14",
        backbone_training_cutoff=date(2024, 6, 1),
        cutoff_source="https://example.test/model",
        adapter_id="odr-typed-host-v3",
        adapter_revision="adapter",
        profile="deep_research",
        retrieval_regime="cutoff_controlled",
        ranking_eligible=False,
    )

    class FakeSystem:
        def __init__(self) -> None:
            self.descriptor = descriptor

        def run_task(self, task, context, capability, binding):
            seen.append(
                (
                    task.task_id,
                    context.run_id,
                    binding.attempt_id,
                    binding.trial,
                    binding.experiment_as_of,
                )
            )
            assert capability.budget == cell.budget
            return {"raw_text": "adapter result"}

        def close(self) -> None:
            seen.append(("closed", "", "", 0, None))

    profile = ResolvedModelProfile(
        mode="openai_compatible",
        endpoint="https://example.test/v1",
        model_name="gpt-4.1-2025-04-14",
        api_key="not-persisted",
        temperature=0.2,
        max_tokens=128,
        timeout_seconds=30,
        retries=0,
        seed=cell.seed,
    )
    executor = PathBV3SystemCellExecutor(lambda _cell: FakeSystem(), lambda _cell: profile)

    result = executor.execute(
        cell,
        TaskInput(task_id="task-1", prompt="prompt", cutoff_date="2025-01-01"),
        output_root=Path("/tmp"),
        attempt=4,
    )

    assert result == {"raw_text": "adapter result"}
    assert seen[0] == (
        "task-1",
        "path-b-v3-test:cell-t1:attempt-4",
        "cell-t1:attempt-4",
        1,
        date(2026, 9, 3),
    )
    assert seen[-1][0] == "closed"
