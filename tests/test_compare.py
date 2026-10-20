"""Regression coverage for comparison artifact-audit gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark_harness.reporting.compare import ReportMixError, build_comparison


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prediction(task_id: str, *, response_present: bool = True) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "raw_response": None,
        "response_present": response_present,
        "api_error": None,
        "parsed": {"format_valid": True},
        "comparable": True,
        "policy_violations": [],
        "canonical_input_hash": f"input-{task_id}",
        "effective_input_hash": f"input-{task_id}",
        "model_call_count": 1,
        "tool_call_count": 0,
        "latency_seconds": 0.1,
        "usage": {"input_tokens": 2, "output_tokens": 1},
    }


def _request(task_id: str, *, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "prompt_sha256": f"prompt-{task_id}",
        "messages_sha256": f"input-{task_id}",
        "canonical_input_hash": f"input-{task_id}",
        "effective_input_hash": f"input-{task_id}",
        "policy": policy if policy is not None else {"comparable": True, "violations": []},
    }


def _ledger(task_id: str, *, is_model_endpoint: bool = True) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "host": "models.example.test",
        "method": "POST",
        "path": "/v1/chat/completions",
        "body_hash": f"body-{task_id}",
        "status_code": 200,
        "latency_seconds": 0.1,
        "is_model_endpoint": is_model_endpoint,
    }


def _trace(task_id: str) -> list[dict[str, Any]]:
    return [
        {"task_id": task_id, "event_type": event}
        for event in (
            "task_started",
            "model_request",
            "model_response",
            "task_finished",
        )
    ]


def _write_complete_baseline(run_dir: Path, name: str = "pydantic-ai") -> Path:
    baseline = run_dir / name
    baseline.mkdir(parents=True)
    task_ids = ["A1-00001", "C-00001"]
    _write_json(baseline / "metrics_summary.json", {"upstream_summary": {"score": 1.0}})
    _write_json(baseline / "run_config.json", {"input_policy": "one_turn_strict"})
    _write_json(
        baseline / "provenance.json",
        {
            "framework": name,
            "framework_version": "test-framework",
            "manifest_version": "test-v1",
            "mock_non_performance": False,
        },
    )
    _write_jsonl(baseline / "predictions.jsonl", [_prediction(task_id) for task_id in task_ids])
    _write_jsonl(
        baseline / "trace_events.jsonl",
        [row for task_id in task_ids for row in _trace(task_id)],
    )
    _write_jsonl(baseline / "effective_requests.jsonl", [_request(task_id) for task_id in task_ids])
    _write_jsonl(baseline / "network_ledger.jsonl", [_ledger(task_id) for task_id in task_ids])
    return baseline


def _write_complete_agentic_baseline(
    run_dir: Path,
    *,
    search_test_only: bool = False,
) -> Path:
    """Persist a live-shaped native-tool run without contacting a service."""

    baseline = run_dir / "pydantic-ai"
    baseline.mkdir(parents=True)
    task_id = "A1-00001"
    prediction = _prediction(task_id)
    prediction.update({"model_call_count": 2, "tool_call_count": 1})
    request = _request(task_id)
    request.update(
        {
            "input_policy": "agentic_web_search",
            "agentic_web_search": {"tool_contract_hash": "agentic-contract"},
        }
    )
    volc_ledger = {
        "task_id": task_id,
        "host": "mercury.volcengineapi.com",
        "method": "POST",
        "path": "/",
        "action": "WebSearch",
        "version": "2025-01-01",
        "body_hash": "request-filter",
        "status_code": 200,
        "latency_seconds": 0.1,
        "is_model_endpoint": False,
        "network_class": "web_search",
        "provider": "volc",
        "query_hash": "query-hash",
        "request_filter_hash": "request-filter",
        "cutoff_date": "2024-01-01",
        "exclusive_cutoff_bound": "2024-01-02T00:00:00Z",
        "cutoff_compliant": True,
    }
    audit = {
        "task_id": task_id,
        "framework": "pydantic-ai",
        "provider": "volc",
        "query_sha256": "query-hash",
        "request_filter_sha256": "request-filter",
        "response_sha256": "response-hash",
        "cutoff_date": "2024-01-01",
        "exclusive_cutoff_bound": "2024-01-02T00:00:00Z",
        "time_range": "2019-01-01..2024-01-01",
        "accepted": True,
        "reason": "accepted",
        "published_at": "2024-01-01T08:00:00Z",
        "test_only": search_test_only,
    }
    _write_json(baseline / "metrics_summary.json", {"upstream_summary": {"score": 1.0}})
    _write_json(
        baseline / "run_config.json",
        {
            "execution": {"input_policy": "agentic_web_search"},
            "search": {"tool_contract_hash": "agentic-contract"},
        },
    )
    _write_json(
        baseline / "provenance.json",
        {
            "framework": "pydantic-ai",
            "framework_version": "test-framework",
            "manifest_version": "test-v1",
            "mock_non_performance": False,
            "search_test_only": search_test_only,
            "tool_contract_hash": "agentic-contract",
        },
    )
    _write_jsonl(baseline / "predictions.jsonl", [prediction])
    _write_jsonl(
        baseline / "trace_events.jsonl",
        [
            *_trace(task_id),
            {"task_id": task_id, "event_type": "tool_call", "tool_name": "search_web"},
        ],
    )
    _write_jsonl(baseline / "effective_requests.jsonl", [request])
    _write_jsonl(
        baseline / "network_ledger.jsonl",
        [
            _ledger(task_id),
            {
                **_ledger(task_id),
                "body_hash": "follow-up-body",
            },
            volc_ledger,
        ],
    )
    _write_jsonl(baseline / "search_audit.jsonl", [audit])
    return baseline


def _only_row(run_dir: Path) -> dict[str, Any]:
    comparison = build_comparison(run_dir)
    assert len(comparison["baselines"]) == 1
    return comparison["baselines"][0]


def test_comparison_uses_hash_only_response_presence_and_manifest_version(tmp_path: Path) -> None:
    _write_complete_baseline(tmp_path)

    row = _only_row(tmp_path)

    assert row["dataset_version"] == "test-v1"
    assert row["success_rate"] == 1.0
    assert row["effective_request_completeness_rate"] == 1.0
    assert row["call_log_completeness_rate"] == 1.0
    assert row["policy_evidence_completeness_rate"] == 1.0
    assert row["policy_compliance_rate"] == 1.0
    assert row["network_ledger_completeness_rate"] == 1.0
    assert row["non_model_network_compliance_rate"] == 1.0
    assert row["artifact_file_completeness_rate"] == 1.0
    assert row["artifact_completeness_rate"] == 1.0
    assert row["artifact_complete"] is True
    assert row["ranking_eligible"] is True


def test_comparison_disqualifies_missing_required_request_artifact(tmp_path: Path) -> None:
    baseline = _write_complete_baseline(tmp_path)
    (baseline / "effective_requests.jsonl").unlink()

    row = _only_row(tmp_path)

    assert row["effective_request_completeness_rate"] == 0.0
    assert row["call_log_completeness_rate"] == 0.0
    assert row["policy_evidence_completeness_rate"] == 0.0
    assert row["artifact_file_completeness_rate"] == 6 / 7
    assert row["artifact_complete"] is False
    assert row["ranking_eligible"] is False
    assert row["artifact_integrity"]["required_files"]["effective_requests.jsonl"] is False


def test_comparison_disqualifies_malformed_truncated_request_log(tmp_path: Path) -> None:
    baseline = _write_complete_baseline(tmp_path)
    (baseline / "effective_requests.jsonl").write_text(
        json.dumps(_request("A1-00001")) + '\n{"task_id":',
        encoding="utf-8",
    )

    row = _only_row(tmp_path)

    assert row["effective_request_completeness_rate"] == 0.5
    assert row["artifact_file_completeness_rate"] == 6 / 7
    assert row["artifact_complete"] is False
    assert row["ranking_eligible"] is False
    assert row["artifact_integrity"]["required_files"]["effective_requests.jsonl"] is False


def test_comparison_disqualifies_truncated_ledger_and_non_model_network(tmp_path: Path) -> None:
    baseline = _write_complete_baseline(tmp_path)
    _write_jsonl(
        baseline / "network_ledger.jsonl",
        [
            _ledger("A1-00001"),
            _ledger("A1-00001", is_model_endpoint=False),
        ],
    )

    row = _only_row(tmp_path)

    assert row["network_ledger_completeness_rate"] == 0.5
    assert row["non_model_network_compliance_rate"] == 0.5
    assert row["zero_external_data_compliance_rate"] == 0.5
    assert row["non_model_network_event_count"] == 1
    assert row["artifact_completeness_rate"] == 0.5
    assert row["artifact_complete"] is False
    assert row["ranking_eligible"] is False


def test_comparison_requires_persisted_policy_evidence_for_ranking(tmp_path: Path) -> None:
    baseline = _write_complete_baseline(tmp_path)
    _write_jsonl(
        baseline / "effective_requests.jsonl",
        [
            _request("A1-00001"),
            _request("C-00001", policy={"comparable": True}),
        ],
    )

    row = _only_row(tmp_path)

    assert row["policy_evidence_completeness_rate"] == 0.5
    assert row["policy_compliance_rate"] == 0.5
    assert row["effective_request_completeness_rate"] == 0.5
    assert row["artifact_complete"] is False
    assert row["ranking_eligible"] is False


def test_comparison_disqualifies_policy_or_request_hash_mismatch(tmp_path: Path) -> None:
    baseline = _write_complete_baseline(tmp_path)
    corrupted = _request("A1-00001")
    corrupted["messages_sha256"] = "different-input"
    _write_jsonl(
        baseline / "effective_requests.jsonl",
        [corrupted, _request("C-00001", policy={"comparable": False, "violations": []})],
    )

    row = _only_row(tmp_path)

    assert row["effective_request_completeness_rate"] == 0.0
    assert row["artifact_complete"] is False
    assert row["ranking_eligible"] is False


def test_comparison_rejects_mixed_profile_populations(tmp_path: Path) -> None:
    strict = _write_complete_baseline(tmp_path, "pydantic-ai")
    agentic = _write_complete_baseline(tmp_path, "langgraph")
    _write_json(
        strict / "run_config.json",
        {"execution": {"input_policy": "one_turn_strict"}},
    )
    _write_json(
        agentic / "run_config.json",
        {
            "execution": {"input_policy": "agentic_web_search"},
            "search": {"tool_contract_hash": "agentic-contract"},
        },
    )

    with pytest.raises(ReportMixError, match="Refusing to mix report populations"):
        build_comparison(tmp_path)


def test_agentic_comparison_accepts_live_shaped_cutoff_compliant_run(tmp_path: Path) -> None:
    _write_complete_agentic_baseline(tmp_path)

    row = _only_row(tmp_path)

    assert row["artifact_complete"] is True
    assert row["ranking_eligible"] is True
    assert row["agentic_budget_compliance_rate"] == 1.0
    assert row["volc_only_network_compliance_rate"] == 1.0
    assert row["search_audit_completeness_rate"] == 1.0
    assert row["zero_external_data_compliance_rate"] is None


def test_agentic_comparison_marks_test_only_run_non_ranking(tmp_path: Path) -> None:
    _write_complete_agentic_baseline(tmp_path, search_test_only=True)

    row = _only_row(tmp_path)

    assert row["artifact_complete"] is True
    assert row["mock_non_performance"] is True
    assert row["ranking_eligible"] is False


def test_agentic_comparison_rejects_non_volc_gateway_evidence(tmp_path: Path) -> None:
    baseline = _write_complete_agentic_baseline(tmp_path)
    ledger_rows = [
        json.loads(line)
        for line in (baseline / "network_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    ledger_rows[-1]["host"] = "unapproved.example.test"
    _write_jsonl(baseline / "network_ledger.jsonl", ledger_rows)

    row = _only_row(tmp_path)

    assert row["volc_only_network_compliance_rate"] == 0.0
    assert row["ranking_eligible"] is False


def test_agentic_comparison_rejects_over_budget_model_trace(tmp_path: Path) -> None:
    baseline = _write_complete_agentic_baseline(tmp_path)
    prediction = json.loads((baseline / "predictions.jsonl").read_text(encoding="utf-8"))
    prediction["model_call_count"] = 5
    _write_jsonl(baseline / "predictions.jsonl", [prediction])

    row = _only_row(tmp_path)

    assert row["agentic_budget_compliance_rate"] == 0.0
    assert row["ranking_eligible"] is False
