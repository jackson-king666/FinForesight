"""Prospective T3 lifecycle registry coverage (WP-B)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmark_harness.experiment.t3_lifecycle import (
    T3_GENESIS_PREV_SHA256,
    T3_REGISTRY_SCHEMA,
    T3_WAVE_SET_SCHEMA,
    T3LockEntry,
    T3RegistryError,
    T3ScheduleEntry,
    T3SettlementEntry,
    append_lock,
    append_schedule,
    append_settlement,
    build_t3_wave_set,
    load_t3_registry,
    prospective_gate,
    validate_t3_registry,
)
from benchmark_harness.experiment.v3 import sha256_json

ORIGIN = "2026-03-10T00:00:00Z"
CUTOFF = "2026-03-10T00:00:00Z"
DEADLINE = "2026-03-10T23:30:00Z"
EXPECTED_SETTLEMENT = "2026-03-12T00:00:00Z"
GENERATED = "2026-03-10T12:00:00Z"
LOCKED = "2026-03-10T23:00:00Z"
# 09:00 at +08:00 is 01:00 UTC the same day: after the 23:30 UTC deadline.
PUBLISHED = "2026-03-11T09:00:00+08:00"
SETTLED = "2026-03-11T02:00:00Z"

PROMPT_SHA = "a" * 64
COHORT_SHA = "b" * 64
POLICY_SHA = "c" * 64
RESPONSE_SHA = "d" * 64
OUTCOME_SHA = "e" * 64

TASK = "B-T3-00001"
TASK_OTHER_WAVE = "B-T3-00002"
WAVE_A = "wave-2026-03-a"
WAVE_B = "wave-2026-03-b"


def _schedule(task_id: str = TASK, wave_id: str = WAVE_A, **overrides: object) -> T3ScheduleEntry:
    fields: dict[str, object] = {
        "task_id": task_id,
        "category": "B",
        "wave_id": wave_id,
        "scheduled_origin": ORIGIN,
        "information_cutoff": CUTOFF,
        "prediction_deadline": DEADLINE,
        "expected_settlement": EXPECTED_SETTLEMENT,
    }
    fields.update(overrides)
    return T3ScheduleEntry(**fields)  # type: ignore[arg-type]


def _lock(
    task_id: str = TASK,
    system_id: str = "financeharness",
    trial: int = 1,
    **overrides: object,
) -> T3LockEntry:
    fields: dict[str, object] = {
        "task_id": task_id,
        "system_id": system_id,
        "trial": trial,
        "prompt_sha256": PROMPT_SHA,
        "cohort_sha256": COHORT_SHA,
        "policy_sha256": POLICY_SHA,
        "response_sha256": RESPONSE_SHA,
        "generated_at": GENERATED,
        "locked_at": LOCKED,
    }
    fields.update(overrides)
    return T3LockEntry(**fields)  # type: ignore[arg-type]


def _settlement(task_id: str = TASK, **overrides: object) -> T3SettlementEntry:
    fields: dict[str, object] = {
        "task_id": task_id,
        "outcome": {"direction": "up", "realized_return": 0.0125},
        "outcome_source_url": "https://example.org/events/b-t3-00001",
        "outcome_published_at": PUBLISHED,
        "outcome_sha256": OUTCOME_SHA,
        "settled_at": SETTLED,
        "settler_identity": "settler-a",
        "review_status": "reviewed",
    }
    fields.update(overrides)
    return T3SettlementEntry(**fields)  # type: ignore[arg-type]


def _seed_registry(root: Path, *, with_lock: bool = True, with_settlement: bool = False) -> Path:
    path = root / "t3-registry.jsonl"
    append_schedule(path, _schedule())
    append_schedule(path, _schedule(task_id=TASK_OTHER_WAVE, wave_id=WAVE_B))
    if with_lock:
        append_lock(path, _lock())
    if with_settlement:
        append_settlement(path, _settlement())
    return path


def _rewrite_line(path: Path, index: int, **changes: object) -> None:
    """Patch one registry line on disk and re-seal its own entry hash."""

    lines = path.read_text(encoding="utf-8").splitlines()
    line = json.loads(lines[index])
    line.update(changes)
    line["entry_sha256"] = sha256_json(
        {key: value for key, value in line.items() if key != "entry_sha256"}
    )
    lines[index] = json.dumps(line, ensure_ascii=False, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_happy_path_schedule_lock_settle_gate_settled(tmp_path: Path) -> None:
    path = _seed_registry(tmp_path, with_settlement=True)
    assert prospective_gate(path, TASK, system_id="financeharness", trial=1) == ("settled", None)
    registry = load_t3_registry(path)
    assert prospective_gate(registry, TASK, system_id="financeharness", trial=1) == (
        "settled",
        None,
    )
    assert [line["entry_kind"] for line in registry.entries] == [
        "schedule",
        "schedule",
        "lock",
        "settlement",
    ]
    assert registry.entries[0]["prev_entry_sha256"] == T3_GENESIS_PREV_SHA256
    for previous, current in zip(registry.entries, registry.entries[1:]):
        assert current["prev_entry_sha256"] == previous["entry_sha256"]
    assert registry.registry_sha256 == registry.entries[-1]["entry_sha256"]
    assert registry.schedule_for(TASK) == _schedule()
    assert registry.lock_for(TASK, "financeharness", 1) == _lock()
    assert registry.settlement_for(TASK) == _settlement()


def test_gate_pending_when_lock_only(tmp_path: Path) -> None:
    path = _seed_registry(tmp_path)
    assert prospective_gate(path, TASK, system_id="financeharness", trial=1) == (
        "pending",
        "settlement_pending",
    )


def test_gate_prediction_lock_missing(tmp_path: Path) -> None:
    path = _seed_registry(tmp_path, with_lock=False)
    assert prospective_gate(path, TASK, system_id="financeharness", trial=1) == (
        "pending",
        "prediction_lock_missing",
    )
    settled_elsewhere = _seed_registry(tmp_path / "other", with_settlement=True)
    # A settlement for the task does not help a different system/trial identity.
    assert prospective_gate(settled_elsewhere, TASK, system_id="odr", trial=1) == (
        "pending",
        "prediction_lock_missing",
    )
    assert prospective_gate(settled_elsewhere, TASK, system_id="financeharness", trial=2) == (
        "pending",
        "prediction_lock_missing",
    )
    absent = prospective_gate(tmp_path / "absent.jsonl", TASK, system_id="financeharness", trial=1)
    assert absent == (
        "pending",
        "prediction_lock_missing",
    )


def test_late_lock_rejected_at_append(tmp_path: Path) -> None:
    path = tmp_path / "t3-registry.jsonl"
    append_schedule(path, _schedule())
    with pytest.raises(T3RegistryError, match="lock_late"):
        append_lock(path, _lock(locked_at="2026-03-10T23:30:01Z"))
    # The deadline itself is the latest admissible lock instant.
    append_lock(path, _lock(locked_at=DEADLINE))


def test_gate_reports_lock_late_when_crafted_on_disk(tmp_path: Path) -> None:
    path = _seed_registry(tmp_path)
    _rewrite_line(path, 2, locked_at="2026-03-10T23:45:00Z")
    registry = load_t3_registry(path)  # hashes and chain are intact; lateness is semantic
    assert prospective_gate(registry, TASK, system_id="financeharness", trial=1) == (
        "pending",
        "lock_late",
    )
    assert prospective_gate(path, TASK, system_id="financeharness", trial=1) == (
        "pending",
        "lock_late",
    )


def test_tampered_registry_fails_load_and_gate(tmp_path: Path) -> None:
    path = _seed_registry(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "financeharness" in text
    path.write_text(text.replace("financeharness", "ginanceharness", 1), encoding="utf-8")
    with pytest.raises(T3RegistryError, match="lock_tampered"):
        load_t3_registry(path)
    assert prospective_gate(path, TASK, system_id="financeharness", trial=1) == (
        "pending",
        "lock_tampered",
    )


def test_broken_chain_fails_load(tmp_path: Path) -> None:
    path = _seed_registry(tmp_path)
    _rewrite_line(path, 1, prev_entry_sha256="1" * 64)
    with pytest.raises(T3RegistryError, match="registry_chain_broken"):
        load_t3_registry(path)


def test_missing_registry_fails_load(tmp_path: Path) -> None:
    with pytest.raises(T3RegistryError, match="registry_missing"):
        load_t3_registry(tmp_path / "nope.jsonl")


def test_timezone_boundary_ordering_across_offsets(tmp_path: Path) -> None:
    path = tmp_path / "t3-registry.jsonl"
    append_schedule(path, _schedule())
    # 07:15 at +08:00 on 2026-03-11 is 23:15 UTC on 2026-03-10: before the deadline.
    append_lock(path, _lock(locked_at="2026-03-11T07:15:00+08:00"))
    # 07:45 at +08:00 is 23:45 UTC: after the 23:30 UTC deadline even though the
    # local date is already the next day.
    with pytest.raises(T3RegistryError, match="lock_late"):
        append_lock(path, _lock(trial=2, locked_at="2026-03-11T07:45:00+08:00"))
    registry = load_t3_registry(path)
    assert registry.locks[0].locked_at == datetime(2026, 3, 10, 23, 15, tzinfo=UTC)
    stored = json.loads(path.read_text(encoding="utf-8").splitlines()[1])
    assert stored["locked_at"] == "2026-03-10T23:15:00Z"
    assert _lock(locked_at="2026-03-11T07:15:00+08:00").to_dict()["locked_at"] == (
        "2026-03-10T23:15:00Z"
    )


def test_settlement_timezone_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "t3-registry.jsonl"
    append_schedule(path, _schedule())
    append_lock(path, _lock())
    # 08:00 at +08:00 is exactly the 00:00 UTC information cutoff: independent.
    append_settlement(path, _settlement(outcome_published_at="2026-03-10T08:00:00+08:00"))
    assert prospective_gate(path, TASK, system_id="financeharness", trial=1) == ("settled", None)


def test_pending_settlement_review_does_not_admit_scoring(tmp_path: Path) -> None:
    path = _seed_registry(tmp_path)
    append_settlement(path, _settlement(review_status="pending"))
    assert prospective_gate(path, TASK, system_id="financeharness", trial=1) == (
        "pending",
        "settlement_pending",
    )


def test_settlement_published_before_cutoff_rejected(tmp_path: Path) -> None:
    path = _seed_registry(tmp_path)
    with pytest.raises(T3RegistryError, match="settlement_not_independent"):
        append_settlement(path, _settlement(outcome_published_at="2026-03-09T23:59:59Z"))


def test_gate_reports_settlement_not_independent_when_crafted(tmp_path: Path) -> None:
    path = _seed_registry(tmp_path, with_settlement=True)
    _rewrite_line(path, 3, outcome_published_at="2026-03-09T23:59:59Z")
    assert prospective_gate(path, TASK, system_id="financeharness", trial=1) == (
        "pending",
        "settlement_not_independent",
    )


def test_outcome_content_guard_rejects_locked_text(tmp_path: Path) -> None:
    path = _seed_registry(tmp_path)
    with pytest.raises(T3RegistryError, match="settlement_not_independent"):
        append_settlement(path, _settlement(), outcome_text_sha256=RESPONSE_SHA)
    with pytest.raises(T3RegistryError, match="settlement_not_independent"):
        append_settlement(path, _settlement(), outcome_text_sha256=PROMPT_SHA)
    with pytest.raises(T3RegistryError, match="settlement_not_independent"):
        append_settlement(path, _settlement(outcome_sha256=RESPONSE_SHA))
    # Distinct outcome content settles cleanly.
    append_settlement(path, _settlement(), outcome_text_sha256="f" * 64)
    assert prospective_gate(path, TASK, system_id="financeharness", trial=1) == ("settled", None)
    other = tmp_path / "second.jsonl"
    append_schedule(other, _schedule())
    append_lock(other, _lock())
    with pytest.raises(T3RegistryError, match="64-hex"):
        append_settlement(other, _settlement(), outcome_text_sha256="not-a-sha")


def test_gate_reports_crafted_outcome_equal_to_locked_response(tmp_path: Path) -> None:
    path = _seed_registry(tmp_path, with_settlement=True)
    _rewrite_line(path, 3, outcome_sha256=RESPONSE_SHA)
    assert prospective_gate(path, TASK, system_id="financeharness", trial=1) == (
        "pending",
        "settlement_not_independent",
    )


def test_settlement_without_lock_rejected(tmp_path: Path) -> None:
    path = tmp_path / "t3-registry.jsonl"
    append_schedule(path, _schedule())
    with pytest.raises(T3RegistryError, match="prediction_lock_missing"):
        append_settlement(path, _settlement())


def test_settlement_without_schedule_rejected(tmp_path: Path) -> None:
    path = tmp_path / "t3-registry.jsonl"
    append_schedule(path, _schedule())
    append_lock(path, _lock())
    with pytest.raises(T3RegistryError, match="schedule_missing"):
        append_settlement(path, _settlement(task_id="B-T3-99999"))
    with pytest.raises(T3RegistryError, match="schedule_missing"):
        append_lock(path, _lock(task_id="B-T3-99999"))


def test_naive_and_day_only_timestamps_rejected() -> None:
    with pytest.raises(T3RegistryError, match="RFC3339"):
        _schedule(prediction_deadline="2026-03-10T23:30:00")
    with pytest.raises(T3RegistryError, match="RFC3339"):
        _schedule(prediction_deadline="2026-03-10")
    with pytest.raises(T3RegistryError, match="RFC3339"):
        _lock(locked_at=datetime(2026, 3, 10, 23, 0))
    with pytest.raises(T3RegistryError, match="RFC3339"):
        _settlement(settled_at="2026-03-11T02:00:00+0800")
    aware = _lock(locked_at=datetime(2026, 3, 10, 23, 0, tzinfo=UTC))
    assert aware.locked_at == datetime(2026, 3, 10, 23, 0, tzinfo=UTC)


def test_schedule_ordering_invariants() -> None:
    with pytest.raises(T3RegistryError, match="schedule_order_invalid"):
        _schedule(prediction_deadline=EXPECTED_SETTLEMENT)
    with pytest.raises(T3RegistryError, match="schedule_order_invalid"):
        _schedule(information_cutoff="2026-03-10T23:30:01Z")
    with pytest.raises(T3RegistryError, match="schedule_order_invalid"):
        _schedule(scheduled_origin="2026-03-10T00:00:01Z")


def test_settlement_ordering_invariants() -> None:
    with pytest.raises(T3RegistryError, match="settlement_order_invalid"):
        _settlement(settled_at="2026-03-11T00:59:59Z")


def test_lock_field_validation() -> None:
    with pytest.raises(T3RegistryError, match="trial"):
        _lock(trial=0)
    with pytest.raises(T3RegistryError, match="trial"):
        _lock(trial=True)
    with pytest.raises(T3RegistryError, match="64-hex"):
        _lock(prompt_sha256="abc123")
    with pytest.raises(T3RegistryError, match="lock_order_invalid"):
        _lock(generated_at="2026-03-10T23:00:00Z", locked_at=GENERATED)


def test_settlement_field_validation() -> None:
    with pytest.raises(T3RegistryError, match="review_status"):
        _settlement(review_status="approved")
    with pytest.raises(T3RegistryError, match="outcome_source_url"):
        _settlement(outcome_source_url="  ")
    with pytest.raises(T3RegistryError, match="settler_identity"):
        _settlement(settler_identity="")
    with pytest.raises(T3RegistryError, match="outcome must be an object"):
        _settlement(outcome="up")


def test_wave_set_requires_two_distinct_waves() -> None:
    with pytest.raises(T3RegistryError, match="insufficient_waves"):
        build_t3_wave_set([_schedule(), _schedule(task_id=TASK_OTHER_WAVE)])
    with pytest.raises(T3RegistryError, match="wave_set_empty"):
        build_t3_wave_set([])
    sealed = build_t3_wave_set([_schedule(), _schedule(task_id=TASK_OTHER_WAVE, wave_id=WAVE_B)])
    assert sealed["schema_version"] == T3_WAVE_SET_SCHEMA
    assert sealed["wave_ids"] == [WAVE_A, WAVE_B]
    assert sealed["schedule_count"] == 2
    assert len(sealed["wave_set_sha256"]) == 64
    again = build_t3_wave_set([_schedule(task_id=TASK_OTHER_WAVE, wave_id=WAVE_B), _schedule()])
    assert again == sealed
    with pytest.raises(T3RegistryError, match="duplicate_schedule"):
        build_t3_wave_set([_schedule(), _schedule(wave_id=WAVE_B)])


def test_duplicate_entries_rejected(tmp_path: Path) -> None:
    path = _seed_registry(tmp_path, with_settlement=True)
    with pytest.raises(T3RegistryError, match="duplicate_schedule"):
        append_schedule(path, _schedule(wave_id=WAVE_B))
    with pytest.raises(T3RegistryError, match="duplicate_lock"):
        append_lock(path, _lock())
    with pytest.raises(T3RegistryError, match="duplicate_settlement"):
        append_settlement(path, _settlement())
    # A different trial is a different lock identity.
    append_lock(path, _lock(trial=2))
    assert prospective_gate(path, TASK, system_id="financeharness", trial=2) == ("settled", None)


def test_hash_chain_determinism(tmp_path: Path) -> None:
    first = _seed_registry(tmp_path / "a", with_settlement=True)
    second = _seed_registry(tmp_path / "b", with_settlement=True)
    assert first.read_bytes() == second.read_bytes()
    loaded_first = validate_t3_registry(first)
    loaded_second = load_t3_registry(second)
    assert loaded_first.registry_sha256 == loaded_second.registry_sha256
    assert [line["entry_sha256"] for line in loaded_first.entries] == [
        line["entry_sha256"] for line in loaded_second.entries
    ]


def test_entry_roundtrip_and_strict_fields() -> None:
    schedule = _schedule()
    assert T3ScheduleEntry.from_dict(schedule.to_dict()) == schedule
    lock = _lock()
    assert T3LockEntry.from_dict(lock.to_dict()) == lock
    settlement = _settlement()
    assert T3SettlementEntry.from_dict(settlement.to_dict()) == settlement
    assert settlement.entry_sha256 == sha256_json(settlement.to_dict())
    with pytest.raises(T3RegistryError, match="unexpected field set"):
        T3ScheduleEntry.from_dict({**schedule.to_dict(), "score": 1})
    with pytest.raises(T3RegistryError, match="unexpected field set"):
        T3LockEntry.from_dict({"task_id": TASK})


def test_registry_line_schema_and_hashes(tmp_path: Path) -> None:
    path = _seed_registry(tmp_path)
    line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert line["schema_version"] == T3_REGISTRY_SCHEMA
    assert line["entry_kind"] == "schedule"
    assert line["prev_entry_sha256"] == T3_GENESIS_PREV_SHA256
    assert line["entry_sha256"] == sha256_json(
        {key: value for key, value in line.items() if key != "entry_sha256"}
    )
