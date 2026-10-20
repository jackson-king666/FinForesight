"""Offline classification of HUA-174 service queries by their provenance.

The audit consumes only hash-bearing ledgers, service WALs, local prompt
files, and optional service-evidence records.  It never calls a provider and
never rewrites an authoritative ledger.  When service evidence is absent, the
output explicitly records the degraded hash-only evidence level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

PROVENANCE_VALUES = frozenset({"model_selected", "host_synthesized", "host_filter_probe"})
DEFAULT_FILTER_PROBE_QUERY = "historical stock market news March 2024"
DEFAULT_FILTER_PROBE_QUERY_SHA256 = (
    "896597f3418ed28c013c154797787495f45ed00c4e44d1ac82e12330472af03b"
)
LEGACY_FILTER_PROBE_QUERY_SHA256 = (
    "3d91a17d0ef7cee88582f8956a51ba6fb0a6038c399b80f5b6d073efa58b036d"
)
KNOWN_FILTER_PROBE_QUERY_SHA256S = frozenset(
    {DEFAULT_FILTER_PROBE_QUERY_SHA256, LEGACY_FILTER_PROBE_QUERY_SHA256}
)
HOST_SYNTHESIZED_QUERY_MAX_CHARS = 512


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalized_prompt(value: str) -> str:
    return " ".join(value.split())


def _prompt_text(value: Mapping[str, Any]) -> str | None:
    prompt = value.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    messages = value.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, str):
        parts = [
            item.get("content")
            for item in messages
            if isinstance(item, Mapping) and isinstance(item.get("content"), str)
        ]
        text = "\n".join(part for part in parts if part)
        return text or None
    return None


def _json_records(path: Path) -> Iterable[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, Mapping):
                yield value
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, Mapping):
        yield value
    elif isinstance(value, list):
        yield from (item for item in value if isinstance(item, Mapping))


def _prompt_records(paths: Iterable[Path]) -> dict[str, tuple[str | None, str, str | None]]:
    prompts: dict[str, tuple[str | None, str, str | None]] = {}
    for source in paths:
        candidates = sorted(source.rglob("*") if source.is_dir() else (source,))
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            try:
                if path.suffix.lower() == ".json":
                    value = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(value, Mapping) and "task_id" not in value:
                        prefix_records = {
                            task_id: details.get("prefix512_sha256")
                            for task_id, details in value.items()
                            if isinstance(task_id, str)
                            and isinstance(details, Mapping)
                            and isinstance(details.get("prefix512_sha256"), str)
                        }
                        if prefix_records:
                            for task_id, prefix_sha256 in prefix_records.items():
                                prompts[sha256_text(task_id)] = (
                                    None,
                                    str(path),
                                    prefix_sha256,
                                )
                            continue
                records = _json_records(path)
                for record in records:
                    task_id = record.get("task_id")
                    prompt = _prompt_text(record)
                    if isinstance(task_id, str) and prompt:
                        prefix_sha256 = record.get("normalized_prompt_prefix512_sha256")
                        if not isinstance(prefix_sha256, str):
                            prefix_sha256 = None
                        prompts[sha256_text(task_id)] = (prompt, str(path), prefix_sha256)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
    return prompts


def _wal_events(paths: Iterable[Path]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in paths:
        for record in _json_records(path):
            nested = record.get("event")
            if isinstance(nested, Mapping):
                event = dict(nested)
                for key in ("cell_id", "line", "source"):
                    if key in record and key not in event:
                        event[key] = record[key]
            else:
                event = dict(record)
            if event.get("event") != "search":
                continue
            if not isinstance(event.get("query_sha256"), str):
                continue
            events.append(event)
    return events


def _service_evidence_records(paths: Iterable[Path]) -> dict[str, Mapping[str, Any]]:
    evidence: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        for record in _json_records(path):
            nested = record.get("record")
            candidate = nested if isinstance(nested, Mapping) else record
            cell_id = candidate.get("cell_id")
            if isinstance(cell_id, str):
                evidence[cell_id] = candidate
    return evidence


def _probe_hashes(
    *,
    filter_probe_query: str,
    filter_probe_queries: Iterable[str] = (),
    filter_probe_query_hashes: Iterable[str] = (),
) -> frozenset[str]:
    hashes = set(KNOWN_FILTER_PROBE_QUERY_SHA256S)
    for query in (filter_probe_query, *filter_probe_queries):
        if isinstance(query, str) and query.strip():
            hashes.add(sha256_text(normalized_prompt(query)))
    hashes.update(value for value in filter_probe_query_hashes if isinstance(value, str) and value)
    return frozenset(hashes)


def _string_values(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [item for item in value if isinstance(item, str)]
    return []


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _receipt_groups(evidence: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    all_receipts = _string_values(evidence.get("receipt_sha256s"))
    explicit_agent = _string_values(evidence.get("agent_receipt_sha256s"))
    explicit_probe = _string_values(evidence.get("probe_receipt_sha256s"))
    agent_count = _int_value(evidence.get("agent_search_call_count"))
    probe_count = _int_value(evidence.get("probe_search_call_count"))

    if explicit_agent:
        agent_receipts = explicit_agent
    elif agent_count > 0:
        # Older service-evidence rows only recorded the combined receipt list.
        agent_receipts = all_receipts[:agent_count]
    elif agent_count == 0 and probe_count >= 1:
        agent_receipts = []
    else:
        agent_receipts = all_receipts

    if explicit_probe:
        probe_receipts = explicit_probe
    elif probe_count >= 1:
        probe_receipts = all_receipts[len(agent_receipts) :]
    else:
        probe_receipts = []
    return agent_receipts, probe_receipts


def _service_probe_hashes(evidence: Mapping[str, Any]) -> frozenset[str]:
    hashes: set[str] = set()
    for key in ("filter_probe_query_sha256", "probe_query_sha256"):
        value = evidence.get(key)
        if isinstance(value, str) and value:
            hashes.add(value)
    for key in ("filter_probe_query_sha256s", "probe_query_sha256s"):
        hashes.update(_string_values(evidence.get(key)))
    for key in ("filter_probe_query", "probe_query"):
        value = evidence.get(key)
        if isinstance(value, str) and value.strip():
            hashes.add(sha256_text(normalized_prompt(value)))
    return frozenset(hashes)


def _probe_only(evidence: Mapping[str, Any]) -> bool:
    return (
        _int_value(evidence.get("agent_search_call_count")) == 0
        and _int_value(evidence.get("probe_search_call_count")) >= 1
    )


def _event_is_probe(
    event: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    agent_receipts: Sequence[str],
    probe_receipts: Sequence[str],
    probe_hashes: frozenset[str],
) -> bool:
    receipt = event.get("receipt_sha256")
    if isinstance(receipt, str):
        if receipt in agent_receipts:
            return False
        if receipt in probe_receipts:
            return True
    if _probe_only(evidence):
        return True
    if event.get("filter_probe") is True:
        return True
    return event.get("query_sha256") in probe_hashes


def classify_query_provenance(
    event: Mapping[str, Any],
    *,
    prompt: str | None,
    filter_probe_query: str = DEFAULT_FILTER_PROBE_QUERY,
    filter_probe_queries: Iterable[str] = (),
    filter_probe_query_hashes: Iterable[str] = (),
    prompt_prefix_sha256: str | None = None,
) -> str:
    query_sha256 = event.get("query_sha256")
    probe_hashes = _probe_hashes(
        filter_probe_query=filter_probe_query,
        filter_probe_queries=filter_probe_queries,
        filter_probe_query_hashes=filter_probe_query_hashes,
    )
    if query_sha256 in probe_hashes:
        return "host_filter_probe"
    if event.get("filter_probe") is True:
        return "host_filter_probe"
    if query_sha256 == prompt_prefix_sha256:
        return "host_synthesized"
    if prompt:
        synthesized = normalized_prompt(prompt)[:HOST_SYNTHESIZED_QUERY_MAX_CHARS]
        if query_sha256 == sha256_text(synthesized):
            return "host_synthesized"
    return "model_selected"


def audit_query_provenance(
    ledger_path: Path,
    wal_paths: Iterable[Path],
    prompt_paths: Iterable[Path],
    *,
    filter_probe_query: str = DEFAULT_FILTER_PROBE_QUERY,
    filter_probe_queries: Iterable[str] = (),
    filter_probe_query_hashes: Iterable[str] = (),
    service_evidence_paths: Iterable[Path] = (),
) -> list[dict[str, Any]]:
    filter_probe_queries = tuple(filter_probe_queries)
    filter_probe_query_hashes = tuple(filter_probe_query_hashes)
    prompts = _prompt_records(prompt_paths)
    events = _wal_events(wal_paths)
    service_evidence = _service_evidence_records(service_evidence_paths)
    probe_hashes = _probe_hashes(
        filter_probe_query=filter_probe_query,
        filter_probe_queries=filter_probe_queries,
        filter_probe_query_hashes=filter_probe_query_hashes,
    )
    by_receipt: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        receipt = event.get("receipt_sha256")
        if isinstance(receipt, str):
            by_receipt.setdefault(receipt, []).append(event)

    rows: list[dict[str, Any]] = []
    for ledger in _json_records(ledger_path):
        task_id = ledger.get("task_id")
        task_hash = sha256_text(task_id) if isinstance(task_id, str) else None
        receipt = ledger.get("receipt_sha256")
        cell_id = ledger.get("cell_id")
        service = service_evidence.get(cell_id) if isinstance(cell_id, str) else None
        agent_receipts: list[str] = []
        probe_receipts: list[str] = []
        corrected_receipt: str | None = None
        if service is not None:
            agent_receipts, probe_receipts = _receipt_groups(service)
            corrected_receipt = agent_receipts[-1] if agent_receipts else None

        join_receipt = corrected_receipt if corrected_receipt is not None else receipt
        matches = by_receipt.get(join_receipt, []) if isinstance(join_receipt, str) else []
        if service is not None and agent_receipts and not _probe_only(service):
            matches = [
                event
                for event in matches
                if not _event_is_probe(
                    event,
                    evidence=service,
                    agent_receipts=agent_receipts,
                    probe_receipts=probe_receipts,
                    probe_hashes=probe_hashes | _service_probe_hashes(service),
                )
            ]
        if not matches and not (service is not None and (_probe_only(service) or receipt is None)):
            continue
        prompt_info = prompts.get(task_hash) if task_hash else None
        prompt = prompt_info[0] if prompt_info else None
        prompt_prefix_sha256 = prompt_info[2] if prompt_info else None
        provenance = {
            classify_query_provenance(
                event,
                prompt=prompt,
                filter_probe_query=filter_probe_query,
                filter_probe_queries=filter_probe_queries,
                filter_probe_query_hashes=probe_hashes,
                prompt_prefix_sha256=prompt_prefix_sha256,
            )
            for event in matches
        }
        if service is not None and _probe_only(service):
            selected = "host_filter_probe"
        elif "host_filter_probe" in provenance:
            selected = "host_filter_probe"
        elif "host_synthesized" in provenance:
            selected = "host_synthesized"
        else:
            selected = "model_selected"
        assert selected in PROVENANCE_VALUES
        if service is None:
            receipt_correction = "none"
            classification_method = "offline WAL hash equality"
            evidence_level = "hash_only"
        elif _probe_only(service):
            receipt_correction = "excluded_probe_only"
            classification_method = (
                "primary artifact: "
                f"agent={_int_value(service.get('agent_search_call_count'))}, "
                f"probe={_int_value(service.get('probe_search_call_count'))}; excluded"
            )
            evidence_level = "service_evidence"
        elif not matches and receipt is None:
            receipt_correction = "null_receipt_artifact_annex"
            classification_method = (
                "HUA-213 §3.3–3.4 classification; packaged primary artifact annex; "
                "no matching WAL for annex receipt in PM inputs"
            )
            evidence_level = "artifact_annex"
        elif receipt is None and corrected_receipt is not None:
            receipt_correction = "null_receipt_artifact_annex"
            classification_method = (
                "PM offline prompt512 hash equality with probe override; CR HUA-213 corroborated"
            )
            evidence_level = "service_evidence"
        elif corrected_receipt is not None and receipt != corrected_receipt:
            receipt_correction = "probe_pointer_to_agent_receipt"
            classification_method = (
                "PM offline prompt512 hash equality with probe override; CR HUA-213 corroborated"
            )
            evidence_level = "service_evidence"
        else:
            receipt_correction = "none"
            classification_method = (
                "PM offline prompt512 hash equality with probe override; CR HUA-213 corroborated"
            )
            evidence_level = "service_evidence"
        row = {
            "cell_id": cell_id,
            "task_id": task_id,
            "framework_id": ledger.get("framework_id"),
            "model_key": ledger.get("model_key"),
            "status": ledger.get("status"),
            "query_provenance": selected,
            "query_sha256s": sorted(
                {event["query_sha256"] for event in matches if event.get("query_sha256")}
            ),
            "receipt_sha256": receipt,
            "receipt_correction": receipt_correction,
            "classification_method": classification_method,
            "evidence_level": evidence_level,
            "wal_event_count": len(matches),
            "prompt_source": prompt_info[1] if prompt_info else None,
        }
        if service is not None:
            row["agent_receipt_sha256"] = corrected_receipt
        rows.append(row)
    return rows


def _write_jsonl(rows: Iterable[Mapping[str, Any]], output: TextIO) -> None:
    for row in rows:
        output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="query-provenance-audit")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--wal", type=Path, action="append", required=True)
    parser.add_argument("--prompts", type=Path, action="append", default=[])
    parser.add_argument("--service-evidence", type=Path, action="append", default=[])
    parser.add_argument("--filter-probe-query", action="append", default=[])
    parser.add_argument("--filter-probe-query-sha256", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    probe_queries = args.filter_probe_query or [DEFAULT_FILTER_PROBE_QUERY]
    rows = audit_query_provenance(
        args.ledger,
        args.wal,
        args.prompts,
        filter_probe_query=probe_queries[0],
        filter_probe_queries=probe_queries[1:],
        filter_probe_query_hashes=args.filter_probe_query_sha256,
        service_evidence_paths=args.service_evidence,
    )
    if args.output is None:
        _write_jsonl(rows, sys.stdout)
    else:
        with args.output.open("w", encoding="utf-8") as handle:
            _write_jsonl(rows, handle)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FILTER_PROBE_QUERY",
    "DEFAULT_FILTER_PROBE_QUERY_SHA256",
    "HOST_SYNTHESIZED_QUERY_MAX_CHARS",
    "KNOWN_FILTER_PROBE_QUERY_SHA256S",
    "LEGACY_FILTER_PROBE_QUERY_SHA256",
    "PROVENANCE_VALUES",
    "audit_query_provenance",
    "classify_query_provenance",
    "main",
    "normalized_prompt",
    "sha256_text",
]
