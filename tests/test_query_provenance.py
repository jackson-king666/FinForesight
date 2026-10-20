from __future__ import annotations

import json
from pathlib import Path

from benchmark_harness.reporting.query_provenance import (
    DEFAULT_FILTER_PROBE_QUERY,
    KNOWN_FILTER_PROBE_QUERY_SHA256S,
    LEGACY_FILTER_PROBE_QUERY_SHA256,
    audit_query_provenance,
    classify_query_provenance,
    normalized_prompt,
    sha256_text,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def test_query_provenance_classifies_the_three_allowed_sources() -> None:
    prompt = "A long task prompt that the host may synthesize."
    assert (
        classify_query_provenance(
            {"query_sha256": sha256_text(normalized_prompt(prompt)[:512])},
            prompt=prompt,
        )
        == "host_synthesized"
    )
    assert (
        classify_query_provenance(
            {"query_sha256": sha256_text(DEFAULT_FILTER_PROBE_QUERY)}, prompt=prompt
        )
        == "host_filter_probe"
    )
    assert LEGACY_FILTER_PROBE_QUERY_SHA256 in KNOWN_FILTER_PROBE_QUERY_SHA256S
    assert (
        classify_query_provenance({"query_sha256": LEGACY_FILTER_PROBE_QUERY_SHA256}, prompt=prompt)
        == "host_filter_probe"
    )
    assert classify_query_provenance({"query_sha256": "unknown"}, prompt=prompt) == "model_selected"


def test_offline_audit_joins_ledger_wal_and_prompt_hashes(tmp_path: Path) -> None:
    task_id = "D-CN_A-P01a"
    prompt = "Historical evidence for this task."
    ledger = tmp_path / "ledger.jsonl"
    wal = tmp_path / "service.wal.jsonl"
    prompts = tmp_path / "d_train.jsonl"
    _write_jsonl(
        ledger,
        [
            {
                "cell_id": "cell-1",
                "task_id": task_id,
                "framework_id": "odr",
                "model_key": "model",
                "status": "success",
                "receipt_sha256": "receipt-1",
            }
        ],
    )
    _write_jsonl(
        prompts,
        [{"task_id": task_id, "prompt": prompt}],
    )
    _write_jsonl(
        wal,
        [
            {
                "event": "search",
                "task_id_sha256": sha256_text(task_id),
                "receipt_sha256": "receipt-1",
                "query_sha256": sha256_text(normalized_prompt(prompt)[:512]),
            }
        ],
    )

    rows = audit_query_provenance(ledger, [wal], [prompts])

    assert rows == [
        {
            "cell_id": "cell-1",
            "framework_id": "odr",
            "model_key": "model",
            "prompt_source": str(prompts),
            "query_provenance": "host_synthesized",
            "query_sha256s": [sha256_text(normalized_prompt(prompt)[:512])],
            "receipt_sha256": "receipt-1",
            "receipt_correction": "none",
            "classification_method": "offline WAL hash equality",
            "evidence_level": "hash_only",
            "status": "success",
            "task_id": task_id,
            "wal_event_count": 1,
        }
    ]


def test_service_evidence_classifies_probe_without_wal_filter_flag(tmp_path: Path) -> None:
    task_id = "D-CN_A-P01a"
    cell_id = "D-D-CN_A-P01a-gpt41mini-odr-t1"
    ledger = tmp_path / "ledger.jsonl"
    wal = tmp_path / "service.wal.jsonl"
    service_evidence = tmp_path / "service-evidence.jsonl"
    _write_jsonl(
        ledger,
        [
            {
                "cell_id": cell_id,
                "task_id": task_id,
                "status": "success",
                "receipt_sha256": "probe-receipt",
            }
        ],
    )
    _write_jsonl(
        wal,
        [
            {
                "event": "search",
                "task_id_sha256": sha256_text(task_id),
                "receipt_sha256": "probe-receipt",
                "query_sha256": "an-unseen-probe-hash",
            }
        ],
    )
    _write_jsonl(
        service_evidence,
        [
            {
                "cell_id": cell_id,
                "agent_search_call_count": 0,
                "probe_search_call_count": 1,
                "filter_probe": None,
            }
        ],
    )

    rows = audit_query_provenance(ledger, [wal], [], service_evidence_paths=[service_evidence])

    assert rows[0]["query_provenance"] == "host_filter_probe"
    assert rows[0]["classification_method"] == "primary artifact: agent=0, probe=1; excluded"
    assert rows[0]["evidence_level"] == "service_evidence"
    assert rows[0]["receipt_correction"] == "excluded_probe_only"


def test_service_evidence_corrects_probe_receipt_pointer_before_joining_wal(
    tmp_path: Path,
) -> None:
    task_id = "D-CN_A-P01a"
    cell_id = "D-D-CN_A-P01a-gpt41mini-odr-t1"
    ledger = tmp_path / "ledger.jsonl"
    wal = tmp_path / "service.wal.jsonl"
    service_evidence = tmp_path / "service-evidence.jsonl"
    agent_query_hash = "agent-query-hash"
    _write_jsonl(
        ledger,
        [
            {
                "cell_id": cell_id,
                "task_id": task_id,
                "status": "success",
                "receipt_sha256": "probe-receipt",
            }
        ],
    )
    _write_jsonl(
        wal,
        [
            {
                "event": "search",
                "task_id_sha256": sha256_text(task_id),
                "receipt_sha256": "agent-receipt",
                "query_sha256": agent_query_hash,
            },
            {
                "event": "search",
                "task_id_sha256": sha256_text(task_id),
                "receipt_sha256": "probe-receipt",
                "query_sha256": sha256_text(DEFAULT_FILTER_PROBE_QUERY),
            },
        ],
    )
    _write_jsonl(
        service_evidence,
        [
            {
                "cell_id": cell_id,
                "agent_receipt_sha256s": ["agent-receipt"],
                "agent_search_call_count": 1,
                "filter_probe": True,
                "probe_receipt_sha256s": ["probe-receipt"],
                "probe_search_call_count": 1,
            }
        ],
    )

    rows = audit_query_provenance(ledger, [wal], [], service_evidence_paths=[service_evidence])

    assert rows[0]["query_provenance"] == "model_selected"
    assert rows[0]["query_sha256s"] == [agent_query_hash]
    assert rows[0]["agent_receipt_sha256"] == "agent-receipt"
    assert rows[0]["receipt_sha256"] == "probe-receipt"
    assert rows[0]["receipt_correction"] == "probe_pointer_to_agent_receipt"
    assert rows[0]["evidence_level"] == "service_evidence"
