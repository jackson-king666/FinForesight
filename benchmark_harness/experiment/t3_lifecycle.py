"""Prospective T3 lifecycle registry: schedule, prediction lock, settlement.

The registry is an append-only JSONL file of ``t3-registry/v1`` entries.  Every
line embeds the previous line's hash (genesis = 64 zero hex digits) and its own
hash over the canonical content including that previous hash, so any rewrite,
reorder, or insertion is detected on load.  All timestamps are timezone-bearing
RFC3339 instants declared by the caller; the machine clock is never read.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from benchmark_harness.experiment.v3 import canonical_json, sha256_json

T3_REGISTRY_SCHEMA = "t3-registry/v1"
T3_WAVE_SET_SCHEMA = "t3-wave-set/v1"
T3_GENESIS_PREV_SHA256 = "0" * 64

T3_GATE_STATUSES = ("settled", "pending")
T3_GATE_REASONS = (
    "prediction_lock_missing",
    "lock_late",
    "lock_tampered",
    "settlement_pending",
    "settlement_not_independent",
)

_REVIEW_STATUSES = ("reviewed", "pending")
_CHAIN_FIELDS = ("schema_version", "entry_kind", "prev_entry_sha256", "entry_sha256")
_PAYLOAD_FIELDS: dict[str, tuple[str, ...]] = {
    "schedule": (
        "task_id",
        "category",
        "wave_id",
        "scheduled_origin",
        "information_cutoff",
        "prediction_deadline",
        "expected_settlement",
    ),
    "lock": (
        "task_id",
        "system_id",
        "trial",
        "prompt_sha256",
        "cohort_sha256",
        "policy_sha256",
        "response_sha256",
        "generated_at",
        "locked_at",
    ),
    "settlement": (
        "task_id",
        "outcome",
        "outcome_source_url",
        "outcome_published_at",
        "outcome_sha256",
        "settled_at",
        "settler_identity",
        "review_status",
    ),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$"
)

__all__ = [
    "T3_GATE_REASONS",
    "T3_GATE_STATUSES",
    "T3_GENESIS_PREV_SHA256",
    "T3_REGISTRY_SCHEMA",
    "T3_WAVE_SET_SCHEMA",
    "T3LockEntry",
    "T3Registry",
    "T3RegistryError",
    "T3ScheduleEntry",
    "T3SettlementEntry",
    "append_lock",
    "append_schedule",
    "append_settlement",
    "build_t3_wave_set",
    "load_t3_registry",
    "prospective_gate",
    "validate_t3_registry",
    "write_t3_entries",
]


class T3RegistryError(ValueError):
    """Raised when a T3 registry entry, chain, or wave set is invalid."""


def _require_nonempty(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise T3RegistryError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, field_name: str) -> str:
    parsed = _require_nonempty(value, field_name)
    if _SHA256_RE.fullmatch(parsed) is None:
        raise T3RegistryError(f"{field_name} must be a lowercase 64-hex SHA-256")
    return parsed


def _parse_instant(value: Any, field_name: str) -> datetime:
    """Parse a timezone-bearing RFC3339 instant; naive and day-only values fail."""

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise T3RegistryError(f"{field_name} must be a timezone-bearing RFC3339 instant")
        return value.astimezone(UTC)
    if isinstance(value, str):
        text = value.strip()
        if _RFC3339_RE.fullmatch(text) is None:
            raise T3RegistryError(f"{field_name} must be a timezone-bearing RFC3339 instant")
        candidate = text[:-1] + "+00:00" if text[-1] in "Zz" else text
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise T3RegistryError(
                f"{field_name} must be a timezone-bearing RFC3339 instant"
            ) from exc
        return parsed.astimezone(UTC)
    raise T3RegistryError(f"{field_name} must be a timezone-bearing RFC3339 instant")


def _format_instant(value: datetime) -> str:
    """Canonical UTC serialization used inside every hashed payload."""

    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _payload(kind: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise T3RegistryError(f"{kind} entry must be an object")
    expected = set(_PAYLOAD_FIELDS[kind])
    if set(value) != expected:
        raise T3RegistryError(f"{kind} entry has an unexpected field set")
    return dict(value)


@dataclass(frozen=True, slots=True)
class T3ScheduleEntry:
    """Predeclared prospective window for one T3 task in one origin wave.

    Timestamp fields accept timezone-bearing RFC3339 strings or aware
    datetimes and are normalized to UTC.
    """

    task_id: str
    category: str
    wave_id: str
    scheduled_origin: datetime
    information_cutoff: datetime
    prediction_deadline: datetime
    expected_settlement: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _require_nonempty(self.task_id, "schedule.task_id"))
        object.__setattr__(self, "category", _require_nonempty(self.category, "schedule.category"))
        object.__setattr__(self, "wave_id", _require_nonempty(self.wave_id, "schedule.wave_id"))
        for name in (
            "scheduled_origin",
            "information_cutoff",
            "prediction_deadline",
            "expected_settlement",
        ):
            object.__setattr__(self, name, _parse_instant(getattr(self, name), f"schedule.{name}"))
        if not self.prediction_deadline < self.expected_settlement:
            raise T3RegistryError(
                "schedule_order_invalid: prediction_deadline must be before expected_settlement"
            )
        if self.scheduled_origin > self.information_cutoff:
            raise T3RegistryError(
                "schedule_order_invalid: scheduled_origin must be on or before information_cutoff"
            )
        if self.information_cutoff > self.prediction_deadline:
            raise T3RegistryError(
                "schedule_order_invalid: information_cutoff must be on or before "
                "prediction_deadline"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "wave_id": self.wave_id,
            "scheduled_origin": _format_instant(self.scheduled_origin),
            "information_cutoff": _format_instant(self.information_cutoff),
            "prediction_deadline": _format_instant(self.prediction_deadline),
            "expected_settlement": _format_instant(self.expected_settlement),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> T3ScheduleEntry:
        payload = _payload("schedule", value)
        return cls(
            task_id=payload["task_id"],
            category=payload["category"],
            wave_id=payload["wave_id"],
            scheduled_origin=payload["scheduled_origin"],
            information_cutoff=payload["information_cutoff"],
            prediction_deadline=payload["prediction_deadline"],
            expected_settlement=payload["expected_settlement"],
        )

    @property
    def entry_sha256(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class T3LockEntry:
    """One system/trial prediction lock bound to a scheduled T3 task.

    Lateness against the schedule prediction deadline is checked by
    ``append_lock`` and ``prospective_gate``, which can see the schedule.
    """

    task_id: str
    system_id: str
    trial: int
    prompt_sha256: str
    cohort_sha256: str
    policy_sha256: str
    response_sha256: str
    generated_at: datetime
    locked_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _require_nonempty(self.task_id, "lock.task_id"))
        object.__setattr__(self, "system_id", _require_nonempty(self.system_id, "lock.system_id"))
        if not isinstance(self.trial, int) or isinstance(self.trial, bool) or self.trial < 1:
            raise T3RegistryError("lock.trial must be an integer >= 1")
        for name in ("prompt_sha256", "cohort_sha256", "policy_sha256", "response_sha256"):
            object.__setattr__(self, name, _require_sha256(getattr(self, name), f"lock.{name}"))
        object.__setattr__(
            self, "generated_at", _parse_instant(self.generated_at, "lock.generated_at")
        )
        object.__setattr__(self, "locked_at", _parse_instant(self.locked_at, "lock.locked_at"))
        if self.generated_at > self.locked_at:
            raise T3RegistryError("lock_order_invalid: generated_at must be on or before locked_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "system_id": self.system_id,
            "trial": self.trial,
            "prompt_sha256": self.prompt_sha256,
            "cohort_sha256": self.cohort_sha256,
            "policy_sha256": self.policy_sha256,
            "response_sha256": self.response_sha256,
            "generated_at": _format_instant(self.generated_at),
            "locked_at": _format_instant(self.locked_at),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> T3LockEntry:
        payload = _payload("lock", value)
        return cls(
            task_id=payload["task_id"],
            system_id=payload["system_id"],
            trial=payload["trial"],
            prompt_sha256=payload["prompt_sha256"],
            cohort_sha256=payload["cohort_sha256"],
            policy_sha256=payload["policy_sha256"],
            response_sha256=payload["response_sha256"],
            generated_at=payload["generated_at"],
            locked_at=payload["locked_at"],
        )

    @property
    def entry_sha256(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class T3SettlementEntry:
    """Independently evidenced outcome appended after the prediction deadline."""

    task_id: str
    outcome: Mapping[str, Any]
    outcome_source_url: str
    outcome_published_at: datetime
    outcome_sha256: str
    settled_at: datetime
    settler_identity: str
    review_status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _require_nonempty(self.task_id, "settlement.task_id"))
        if not isinstance(self.outcome, Mapping):
            raise T3RegistryError("settlement.outcome must be an object")
        outcome = MappingProxyType(dict(self.outcome))
        try:
            canonical_json(dict(outcome))
        except ValueError as exc:
            raise T3RegistryError(
                "settlement.outcome must be canonically JSON serializable"
            ) from exc
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(
            self,
            "outcome_source_url",
            _require_nonempty(self.outcome_source_url, "settlement.outcome_source_url"),
        )
        object.__setattr__(
            self,
            "outcome_published_at",
            _parse_instant(self.outcome_published_at, "settlement.outcome_published_at"),
        )
        object.__setattr__(
            self,
            "outcome_sha256",
            _require_sha256(self.outcome_sha256, "settlement.outcome_sha256"),
        )
        object.__setattr__(
            self, "settled_at", _parse_instant(self.settled_at, "settlement.settled_at")
        )
        if self.settled_at < self.outcome_published_at:
            raise T3RegistryError(
                "settlement_order_invalid: settled_at must be on or after outcome_published_at"
            )
        object.__setattr__(
            self,
            "settler_identity",
            _require_nonempty(self.settler_identity, "settlement.settler_identity"),
        )
        if self.review_status not in _REVIEW_STATUSES:
            raise T3RegistryError("settlement.review_status must be 'reviewed' or 'pending'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "outcome": dict(self.outcome),
            "outcome_source_url": self.outcome_source_url,
            "outcome_published_at": _format_instant(self.outcome_published_at),
            "outcome_sha256": self.outcome_sha256,
            "settled_at": _format_instant(self.settled_at),
            "settler_identity": self.settler_identity,
            "review_status": self.review_status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> T3SettlementEntry:
        payload = _payload("settlement", value)
        return cls(
            task_id=payload["task_id"],
            outcome=payload["outcome"],
            outcome_source_url=payload["outcome_source_url"],
            outcome_published_at=payload["outcome_published_at"],
            outcome_sha256=payload["outcome_sha256"],
            settled_at=payload["settled_at"],
            settler_identity=payload["settler_identity"],
            review_status=payload["review_status"],
        )

    @property
    def entry_sha256(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class T3Registry:
    """A hash-verified T3 registry chain in append order."""

    entries: tuple[Mapping[str, Any], ...]
    schedules: tuple[T3ScheduleEntry, ...]
    locks: tuple[T3LockEntry, ...]
    settlements: tuple[T3SettlementEntry, ...]

    @property
    def registry_sha256(self) -> str:
        if not self.entries:
            return T3_GENESIS_PREV_SHA256
        return str(self.entries[-1]["entry_sha256"])

    def schedule_for(self, task_id: str) -> T3ScheduleEntry | None:
        for entry in self.schedules:
            if entry.task_id == task_id:
                return entry
        return None

    def lock_for(self, task_id: str, system_id: str, trial: int) -> T3LockEntry | None:
        for entry in self.locks:
            if entry.task_id == task_id and entry.system_id == system_id and entry.trial == trial:
                return entry
        return None

    def locks_for(self, task_id: str) -> tuple[T3LockEntry, ...]:
        return tuple(entry for entry in self.locks if entry.task_id == task_id)

    def settlement_for(self, task_id: str) -> T3SettlementEntry | None:
        for entry in self.settlements:
            if entry.task_id == task_id:
                return entry
        return None


def _registry_line(kind: str, entry: Any, prev_entry_sha256: str) -> dict[str, Any]:
    line: dict[str, Any] = {
        "schema_version": T3_REGISTRY_SCHEMA,
        "entry_kind": kind,
        **entry.to_dict(),
        "prev_entry_sha256": prev_entry_sha256,
    }
    line["entry_sha256"] = sha256_json(line)
    return line


def _line_payload_hash(line: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in line.items() if key != "entry_sha256"})


def _line_intact(line: Mapping[str, Any]) -> bool:
    digest = line.get("entry_sha256")
    return (
        isinstance(digest, str)
        and _SHA256_RE.fullmatch(digest) is not None
        and digest == _line_payload_hash(line)
    )


def _kind_lines(registry: T3Registry, kind: str) -> Iterable[Mapping[str, Any]]:
    return (line for line in registry.entries if line.get("entry_kind") == kind)


def _load_for_append(path: Path) -> T3Registry:
    if not path.exists():
        return T3Registry(entries=(), schedules=(), locks=(), settlements=())
    return load_t3_registry(path)


def _append_line(path: Path, line: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(line) + "\n")
    except OSError as exc:
        raise T3RegistryError(f"unable to append registry entry: {path}") from exc
    return str(line["entry_sha256"])


def append_schedule(registry_path: str | Path, entry: T3ScheduleEntry) -> str:
    """Append a schedule entry; one schedule per task_id, never rewritten."""

    if not isinstance(entry, T3ScheduleEntry):
        raise T3RegistryError("append_schedule requires a T3ScheduleEntry")
    path = Path(registry_path).expanduser()
    registry = _load_for_append(path)
    if registry.schedule_for(entry.task_id) is not None:
        raise T3RegistryError(f"duplicate_schedule: {entry.task_id}")
    return _append_line(path, _registry_line("schedule", entry, registry.registry_sha256))


def append_lock(registry_path: str | Path, entry: T3LockEntry) -> str:
    """Append a prediction lock; locked_at must be on or before the deadline."""

    if not isinstance(entry, T3LockEntry):
        raise T3RegistryError("append_lock requires a T3LockEntry")
    path = Path(registry_path).expanduser()
    registry = _load_for_append(path)
    schedule = registry.schedule_for(entry.task_id)
    if schedule is None:
        raise T3RegistryError(f"schedule_missing: no schedule for {entry.task_id}")
    if registry.lock_for(entry.task_id, entry.system_id, entry.trial) is not None:
        raise T3RegistryError(
            f"duplicate_lock: {entry.task_id}/{entry.system_id}/trial-{entry.trial}"
        )
    if entry.locked_at > schedule.prediction_deadline:
        raise T3RegistryError(
            f"lock_late: locked_at {entry.locked_at.isoformat()} is after prediction_deadline "
            f"{schedule.prediction_deadline.isoformat()}"
        )
    return _append_line(path, _registry_line("lock", entry, registry.registry_sha256))


def append_settlement(
    registry_path: str | Path,
    entry: T3SettlementEntry,
    *,
    outcome_text_sha256: str | None = None,
) -> str:
    """Append an independent settlement; requires a prior valid lock.

    ``outcome_text_sha256`` is the caller-supplied hash of the raw outcome
    text.  A settlement whose outcome content equals the locked response or
    prompt text is post-cutoff forecast text mislabeled as an outcome and is
    rejected as ``settlement_not_independent``.
    """

    if not isinstance(entry, T3SettlementEntry):
        raise T3RegistryError("append_settlement requires a T3SettlementEntry")
    path = Path(registry_path).expanduser()
    registry = _load_for_append(path)
    schedule = registry.schedule_for(entry.task_id)
    if schedule is None:
        raise T3RegistryError(f"schedule_missing: no schedule for {entry.task_id}")
    if registry.settlement_for(entry.task_id) is not None:
        raise T3RegistryError(f"duplicate_settlement: {entry.task_id}")
    locks = registry.locks_for(entry.task_id)
    if not any(lock.locked_at <= schedule.prediction_deadline for lock in locks):
        raise T3RegistryError(f"prediction_lock_missing: no valid lock for {entry.task_id}")
    if entry.outcome_published_at < schedule.information_cutoff:
        raise T3RegistryError(
            "settlement_not_independent: outcome_published_at "
            f"{entry.outcome_published_at.isoformat()} precedes information_cutoff "
            f"{schedule.information_cutoff.isoformat()}"
        )
    locked_text_hashes = {
        digest for lock in locks for digest in (lock.response_sha256, lock.prompt_sha256)
    }
    if entry.outcome_sha256 in locked_text_hashes:
        raise T3RegistryError(
            "settlement_not_independent: outcome content equals locked response/prompt text"
        )
    if outcome_text_sha256 is not None:
        text_sha256 = _require_sha256(outcome_text_sha256, "outcome_text_sha256")
        if text_sha256 in locked_text_hashes:
            raise T3RegistryError(
                "settlement_not_independent: outcome text equals locked response/prompt text"
            )
    return _append_line(path, _registry_line("settlement", entry, registry.registry_sha256))


def write_t3_entries(
    registry_path: str | Path,
    entries: Iterable[T3ScheduleEntry | T3LockEntry | T3SettlementEntry],
) -> tuple[str, ...]:
    """Append an ordered batch of entries through the per-kind writers."""

    digests: list[str] = []
    for entry in entries:
        if isinstance(entry, T3ScheduleEntry):
            digests.append(append_schedule(registry_path, entry))
        elif isinstance(entry, T3LockEntry):
            digests.append(append_lock(registry_path, entry))
        elif isinstance(entry, T3SettlementEntry):
            digests.append(append_settlement(registry_path, entry))
        else:
            raise T3RegistryError("registry entry kind is unknown")
    return tuple(digests)


def load_t3_registry(path: str | Path) -> T3Registry:
    """Load a registry, re-verifying every entry hash and chain continuity."""

    registry_path = Path(path).expanduser()
    if not registry_path.exists():
        raise T3RegistryError(f"registry_missing: {registry_path}")
    try:
        raw_lines = registry_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise T3RegistryError(f"unable to read registry: {registry_path}") from exc
    entries: list[Mapping[str, Any]] = []
    schedules: list[T3ScheduleEntry] = []
    locks: list[T3LockEntry] = []
    settlements: list[T3SettlementEntry] = []
    seen: set[tuple[str, ...]] = set()
    prev_entry_sha256 = T3_GENESIS_PREV_SHA256
    for line_number, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            continue
        try:
            line = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise T3RegistryError(f"registry_invalid: {registry_path}:{line_number}") from exc
        if not isinstance(line, dict):
            raise T3RegistryError(f"registry_invalid: entry must be an object:{line_number}")
        if line.get("schema_version") != T3_REGISTRY_SCHEMA:
            raise T3RegistryError(f"registry_schema_unknown at line {line_number}")
        kind = line.get("entry_kind")
        if kind not in _PAYLOAD_FIELDS:
            raise T3RegistryError(f"registry entry kind is unknown at line {line_number}")
        if set(line) != set(_PAYLOAD_FIELDS[kind]) | set(_CHAIN_FIELDS):
            raise T3RegistryError(f"registry_invalid: unexpected field set at line {line_number}")
        entry_sha256 = line["entry_sha256"]
        if not isinstance(entry_sha256, str) or _SHA256_RE.fullmatch(entry_sha256) is None:
            raise T3RegistryError(f"registry_invalid: entry hash malformed at line {line_number}")
        if _line_payload_hash(line) != entry_sha256:
            raise T3RegistryError(f"lock_tampered: entry hash mismatch at line {line_number}")
        if line["prev_entry_sha256"] != prev_entry_sha256:
            raise T3RegistryError(f"registry_chain_broken at line {line_number}")
        payload = {key: line[key] for key in _PAYLOAD_FIELDS[kind]}
        if kind == "schedule":
            entry: T3ScheduleEntry | T3LockEntry | T3SettlementEntry = T3ScheduleEntry.from_dict(
                payload
            )
            dedupe_key = ("schedule", entry.task_id)
        elif kind == "lock":
            entry = T3LockEntry.from_dict(payload)
            dedupe_key = ("lock", entry.task_id, entry.system_id, str(entry.trial))
        else:
            entry = T3SettlementEntry.from_dict(payload)
            dedupe_key = ("settlement", entry.task_id)
        if dedupe_key in seen:
            raise T3RegistryError(f"duplicate_entry at line {line_number}")
        seen.add(dedupe_key)
        if isinstance(entry, T3ScheduleEntry):
            schedules.append(entry)
        elif isinstance(entry, T3LockEntry):
            locks.append(entry)
        else:
            settlements.append(entry)
        entries.append(line)
        prev_entry_sha256 = entry_sha256
    return T3Registry(
        entries=tuple(entries),
        schedules=tuple(schedules),
        locks=tuple(locks),
        settlements=tuple(settlements),
    )


def validate_t3_registry(path: str | Path) -> T3Registry:
    """Load and fully re-verify a registry; any drift raises T3RegistryError."""

    return load_t3_registry(path)


def build_t3_wave_set(schedules: Iterable[T3ScheduleEntry]) -> dict[str, Any]:
    """Seal the predeclared origin waves of a schedule set (>= 2 distinct waves)."""

    entries = tuple(schedules)
    if not entries:
        raise T3RegistryError("wave_set_empty: at least one schedule is required")
    task_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, T3ScheduleEntry):
            raise T3RegistryError("wave sets seal T3ScheduleEntry values only")
        task_ids.append(entry.task_id)
    if len(task_ids) != len(set(task_ids)):
        raise T3RegistryError("duplicate_schedule: wave set task_id values must be unique")
    wave_ids = sorted({entry.wave_id for entry in entries})
    if len(wave_ids) < 2:
        raise T3RegistryError(
            "insufficient_waves: a sealed wave set requires at least two distinct wave_ids"
        )
    ordered = sorted(entries, key=lambda entry: entry.task_id)
    payload: dict[str, Any] = {
        "schema_version": T3_WAVE_SET_SCHEMA,
        "wave_ids": wave_ids,
        "schedule_count": len(ordered),
        "schedule_entry_sha256": [entry.entry_sha256 for entry in ordered],
    }
    payload["wave_set_sha256"] = sha256_json(payload)
    return payload


def _coerce_registry(registry: T3Registry | str | Path) -> tuple[T3Registry | None, str | None]:
    if isinstance(registry, T3Registry):
        return registry, None
    path = Path(registry).expanduser()
    if not path.exists():
        return None, "prediction_lock_missing"
    try:
        return load_t3_registry(path), None
    except T3RegistryError:
        return None, "lock_tampered"


def prospective_gate(
    registry: T3Registry | str | Path,
    task_id: str,
    *,
    system_id: str,
    trial: int,
) -> tuple[str, str | None]:
    """Return the scorer-admission state for one system/trial on a T3 task.

    ``("settled", None)`` is required for scorer admission.  Every other
    outcome is ``("pending", reason)`` with one of ``T3_GATE_REASONS``.
    """

    resolved, load_reason = _coerce_registry(registry)
    if resolved is None:
        return ("pending", load_reason)
    schedule = resolved.schedule_for(task_id)
    lock: T3LockEntry | None = None
    lock_line: Mapping[str, Any] | None = None
    for candidate, line in zip(resolved.locks, _kind_lines(resolved, "lock"), strict=True):
        if (
            candidate.task_id == task_id
            and candidate.system_id == system_id
            and candidate.trial == trial
        ):
            lock, lock_line = candidate, line
            break
    if lock is None:
        return ("pending", "prediction_lock_missing")
    if lock_line is None or not _line_intact(lock_line):
        return ("pending", "lock_tampered")
    if schedule is None:
        return ("pending", "lock_tampered")
    if lock.locked_at > schedule.prediction_deadline:
        return ("pending", "lock_late")
    settlement: T3SettlementEntry | None = None
    settlement_line: Mapping[str, Any] | None = None
    settlement_lines = _kind_lines(resolved, "settlement")
    for candidate, line in zip(resolved.settlements, settlement_lines, strict=True):
        if candidate.task_id == task_id:
            settlement, settlement_line = candidate, line
            break
    if settlement is None:
        return ("pending", "settlement_pending")
    if settlement.review_status != "reviewed":
        return ("pending", "settlement_pending")
    if settlement_line is None or not _line_intact(settlement_line):
        return ("pending", "lock_tampered")
    if settlement.outcome_published_at < schedule.information_cutoff:
        return ("pending", "settlement_not_independent")
    if settlement.outcome_sha256 in {lock.response_sha256, lock.prompt_sha256}:
        return ("pending", "settlement_not_independent")
    return ("settled", None)
