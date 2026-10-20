"""Score-independent inventory and disposition contracts for experiment/v4.

The functions in this module deliberately operate on metadata rows only.  They
do not load a Hub dataset, inspect predictions, or make a readiness decision
from scores.  This keeps the inventory useful before execution and makes the
result safe to seal as part of a v4 manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

V4_DATASET_REPO_ID = "sselaine27/benchmark-research"
V4_DATASET_REVISION = "951edec3021353bf1f0db23a49cf8596a7148490"
V4_DATASET_TREE_SHA = "410c6c450abd5df4d73d6827004dd12a1cbf0c1b"
V4_DATASET_TREE_SHA256 = V4_DATASET_TREE_SHA
V4_EXPECTED_RECORD_COUNT = 1_119
V4_EXPECTED_CONFIG_COUNTS = {
    "a1": 244,
    "a2_f": 112,
    "a2_t": 161,
    "a2_h": 110,
    "b": 138,
    "c": 294,
    "d": 30,
    "e": 30,
}
V4_REQUIRED_ROW_PREDICATES = (
    "official_temporal_eligible",
    "review_status",
    "outcome_available_at_source",
    "status",
)
EXPECTED_E_FORMULA_FAMILIES = (
    "arithmetic",
    "percentage_change",
    "weighted_average",
    "compound_growth",
    "duration_approx",
    "black_scholes_call",
    "present_value",
    "ratio",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class DataInventoryError(ValueError):
    """Raised when inventory or collection-ledger input is unsafe."""


def _records(value: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    if isinstance(value, Mapping) or isinstance(value, str | bytes):
        raise DataInventoryError("records must be an iterable of objects")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise DataInventoryError(f"record {index} must be an object")
        result.append(dict(row))
    return tuple(result)


def _first(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _nested(row: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = row
        for part in path:
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(part)
        if value is not None:
            return value
    return None


def _task_id(row: Mapping[str, Any], index: int) -> str:
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise DataInventoryError(f"record {index} has no non-empty task_id")
    return task_id.strip()


def _category(row: Mapping[str, Any]) -> str:
    value = _first(row, "category", "task_category")
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    task_id = row.get("task_id")
    if isinstance(task_id, str):
        prefix = task_id.split("-", 1)[0].upper()
        if prefix.startswith("A1"):
            return "A1"
        if prefix.startswith("A2"):
            return "A2"
        if prefix in {"B", "C", "D", "E"}:
            return prefix
    return "UNKNOWN"


def _variant(row: Mapping[str, Any]) -> str | None:
    value = _first(row, "variant", "signal_variant", "task_variant")
    return value.strip().upper() if isinstance(value, str) and value.strip() else None


def _config(row: Mapping[str, Any]) -> str:
    configured = _first(row, "config", "config_name", "dataset_config")
    if isinstance(configured, str) and configured.strip():
        return configured.strip().lower().replace("-", "_")
    category = _category(row).lower()
    variant = _variant(row)
    if category == "a2" and variant:
        return f"a2_{variant.lower()}"
    return category


def _source_band(row: Mapping[str, Any]) -> str | None:
    value = _first(row, "paper_band", "source_time_band", "time_band", "temporal_band")
    return value.strip().upper() if isinstance(value, str) and value.strip() else None


def _declared_as_of(value: date | datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise DataInventoryError("as_of datetime must carry a timezone")
        return value.astimezone(UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise DataInventoryError(
                "as_of must be an ISO date or timezone-aware timestamp"
            ) from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            try:
                parsed = datetime.fromisoformat(text + "T00:00:00")
            except ValueError as exc:
                raise DataInventoryError(
                    "as_of must be an ISO date or timezone-aware timestamp"
                ) from exc
        return parsed.astimezone(UTC)
    raise DataInventoryError("as_of must be an ISO date or timezone-aware timestamp")


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return None
        return value.astimezone(UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _is_observed(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("observed_")


def compute_inventory(
    records: Iterable[Mapping[str, Any]],
    *,
    as_of: date | datetime | str | None = None,
) -> dict[str, Any]:
    """Return a reproducible, score-blind inventory and gap mask.

    ``as_of`` is explicit on purpose.  When supplied, a future
    ``forecast_origin`` is counted as a prospective/T3 row; the function never
    consults the machine clock.  The returned object contains task IDs only in
    gap masks and no prediction or score fields.
    """

    rows = _records(records)
    task_ids = [_task_id(row, index) for index, row in enumerate(rows)]
    if len(task_ids) != len(set(task_ids)):
        raise DataInventoryError("records contain duplicate task_id values")
    declared = _declared_as_of(as_of)
    config_rows: dict[str, list[str]] = defaultdict(list)
    predicate_missing: dict[str, list[str]] = {name: [] for name in V4_REQUIRED_ROW_PREDICATES}
    source_bands: dict[str, Counter[str]] = defaultdict(Counter)
    b_counts: Counter[str] = Counter()
    t3_rows: list[str] = []
    for row, task_id in zip(rows, task_ids, strict=True):
        config_rows[_config(row)].append(task_id)
        category = _category(row)
        band = _source_band(row)
        if category in {"A1", "A2", "B", "C"} and band in {"T1", "T2"}:
            source_bands[category][band] += 1
        if category == "B":
            subtype = _first(row, "event_subtype", "subtype", "b_subtype", "variant")
            label = str(subtype).strip().lower() if subtype is not None else "unknown"
            b_counts[label] += 1
        origin = _first(row, "forecast_origin", "prediction_origin", "origin_timestamp")
        origin_time = _timestamp(origin)
        explicit_t3 = str(_first(row, "temporal_band", "paper_band", "phase") or "").upper() == "T3"
        if explicit_t3 or (
            declared is not None and origin_time is not None and origin_time > declared
        ):
            t3_rows.append(task_id)
        if row.get("official_temporal_eligible") is not True:
            predicate_missing["official_temporal_eligible"].append(task_id)
        if row.get("review_status") != "reviewed":
            predicate_missing["review_status"].append(task_id)
        if not _is_observed(row.get("outcome_available_at_source")):
            predicate_missing["outcome_available_at_source"].append(task_id)
        if row.get("status") != "ready":
            predicate_missing["status"].append(task_id)
    config_counts = {name: len(values) for name, values in sorted(config_rows.items())}
    deltas = {
        name: config_counts.get(name, 0) - expected
        for name, expected in sorted(V4_EXPECTED_CONFIG_COUNTS.items())
    }
    for name, count in config_counts.items():
        if name not in deltas:
            deltas[name] = count
    gap_masks = {
        predicate: {
            "missing_count": len(task_list),
            "task_ids": sorted(task_list),
            "by_config": {
                config: sorted(
                    task_id for task_id in task_list if task_id in set(config_rows[config])
                )
                for config in sorted(config_rows)
            },
        }
        for predicate, task_list in predicate_missing.items()
    }
    return {
        "schema_version": "data-inventory/v1",
        "dataset_repo_id": V4_DATASET_REPO_ID,
        "dataset_revision": V4_DATASET_REVISION,
        "tree_sha": V4_DATASET_TREE_SHA,
        "record_count": len(rows),
        "unique_task_count": len(set(task_ids)),
        "expected_record_count": V4_EXPECTED_RECORD_COUNT,
        "expected_config_counts": dict(V4_EXPECTED_CONFIG_COUNTS),
        "config_counts": config_counts,
        "config_count_deltas": dict(sorted(deltas.items())),
        "gap_masks": gap_masks,
        "source_band_splits": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(source_bands.items())
        },
        "b_subtype_counts": dict(sorted(b_counts.items())),
        "zero_t3_confirmation": {
            "as_of": as_of.isoformat() if isinstance(as_of, date | datetime) else as_of,
            "t3_count": len(t3_rows),
            "task_ids": sorted(t3_rows),
            "confirmed_zero": not t3_rows,
        },
    }


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value.strip()) is not None


def _url(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith(("https://", "http://"))


def validate_b_event_chain(record: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """Validate the complete official event and four-price B evidence chain."""

    if not isinstance(record, Mapping):
        raise DataInventoryError("B record must be an object")
    reasons: list[str] = []
    event_id = _first(record, "event_id", "event_identity", "official_event_id", "release_id")
    if not isinstance(event_id, str) or not event_id.strip():
        reasons.append("missing_event_identity")
    release = _first(record, "release_timestamp", "release_time", "event_release_timestamp")
    if release is None:
        reasons.append("missing_release_timestamp")
    elif _timestamp(release) is None:
        reasons.append("release_timestamp_naive_or_invalid")
    source_url = _first(
        record,
        "first_party_url",
        "event_source_url",
        "official_source_url",
        "source_url",
    )
    if not _url(source_url):
        reasons.append("missing_first_party_url")
    snapshot_hash = _first(
        record,
        "event_snapshot_sha256",
        "snapshot_sha256",
        "source_snapshot_sha256",
        "event_source_sha256",
    )
    if not _sha(snapshot_hash):
        reasons.append("missing_event_snapshot_hash")
    evidence = record.get("price_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else record
    names = {
        "pre": ("pre_price", "price_pre", "baseline_price"),
        "post": ("post_price", "price_post", "reaction_price"),
        "benchmark_pre": ("benchmark_pre_price", "pre_benchmark_price"),
        "benchmark_post": ("benchmark_post_price", "post_benchmark_price"),
    }
    prices: dict[str, Any] = {}
    for label, aliases in names.items():
        value = _first(evidence, *aliases)
        prices[label] = value
        if not _positive_finite(value):
            reasons.append(f"missing_{label}_price" if value is None else f"invalid_{label}_price")
    for label in names:
        units = _first(
            evidence,
            f"{label}_unit",
            f"{label}_units",
            "price_unit" if label in {"pre", "post"} else "benchmark_unit",
        )
        if not isinstance(units, str) or not units.strip():
            reasons.append(f"missing_{label}_units")
        timestamp = _first(
            evidence,
            f"{label}_timestamp",
            f"{label}_trading_timestamp",
            f"{label}_observed_at",
        )
        if timestamp is None:
            reasons.append(f"missing_{label}_timestamp")
        elif _timestamp(timestamp) is None:
            reasons.append(f"{label}_timestamp_naive_or_invalid")
        url = _first(evidence, f"{label}_url", f"{label}_source_url", "source_url")
        if not _url(url):
            reasons.append(f"missing_{label}_url")
        digest = _first(evidence, f"{label}_sha256", f"{label}_snapshot_sha256")
        if not _sha(digest):
            reasons.append(f"missing_{label}_hash")
    if not reasons:
        excess = (float(prices["post"]) / float(prices["pre"]) - 1.0) - (
            float(prices["benchmark_post"]) / float(prices["benchmark_pre"]) - 1.0
        )
        supplied_return = _first(record, "excess_return", "ground_truth_return", "actual_return")
        if supplied_return is not None and (
            not isinstance(supplied_return, int | float)
            or not math.isfinite(float(supplied_return))
            or not math.isclose(float(supplied_return), excess, rel_tol=1e-9, abs_tol=1e-12)
        ):
            reasons.append("ground_truth_conflict")
        direction = "up" if excess > 0 else "down"
        supplied_direction = _first(
            record, "direction", "actual_direction", "ground_truth_direction"
        )
        if supplied_direction is not None and str(supplied_direction).lower() != direction:
            reasons.append("ground_truth_conflict")
        if excess == 0:
            zero_direction = _first(
                record, "direction", "actual_direction", "ground_truth_direction"
            )
            if zero_direction is not None and str(zero_direction).lower() != "down":
                reasons.append("zero_return_must_map_down")
    return not reasons, tuple(dict.fromkeys(reasons))


def validate_e_family_allocation(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_families: Sequence[str] = EXPECTED_E_FORMULA_FAMILIES,
    min_cases: int = 10,
) -> dict[str, Any]:
    """Check all eight E families, parameter diversity, and hard controls."""

    if min_cases < 1:
        raise DataInventoryError("min_cases must be positive")
    rows = [row for row in _records(records) if _category(row) == "E"]
    counts: Counter[str] = Counter()
    cases: dict[str, set[str]] = defaultdict(set)
    hard_rows: list[str] = []
    for row in rows:
        family = _first(row, "formula_family", "family", "formula", "formula_type")
        if not isinstance(family, str) or not family.strip():
            family = "unknown"
        family = family.strip().lower()
        counts[family] += 1
        case = _first(row, "parameter_case_id", "case_id", "parameters_hash", "task_id")
        if case is not None:
            cases[family].add(str(case))
        difficulty = str(_first(row, "difficulty", "difficulty_level") or "").lower()
        if difficulty == "hard" or family in {"duration_approx", "black_scholes_call"}:
            hard_rows.append(str(row.get("task_id", "")))
    deficits: dict[str, int] = {}
    for family in expected_families:
        deficits[family] = max(0, min_cases - len(cases.get(family.lower(), set())))
    missing_families = [family for family in expected_families if not cases.get(family.lower())]
    hard_families = {
        family: len(cases.get(family.lower(), set()))
        for family in ("duration_approx", "black_scholes_call")
    }
    if any(value == 0 for value in hard_families.values()) and not hard_rows:
        hard_rows = []
    return {
        "schema_version": "e-family-allocation/v1",
        "record_count": len(rows),
        "expected_family_count": len(expected_families),
        "family_counts": dict(sorted(counts.items())),
        "case_counts": {
            family: len(cases.get(family.lower(), set())) for family in expected_families
        },
        "family_deficits": dict(sorted(deficits.items())),
        "missing_families": missing_families,
        "hard_case_task_ids": sorted(set(hard_rows)),
        "hard_family_counts": hard_families,
        "valid": not missing_families
        and not any(deficits.values())
        and all(value > 0 for value in hard_families.values()),
        "deficits": {
            "families": missing_families,
            "parameter_cases": {key: value for key, value in deficits.items() if value},
            "hard_families": [key for key, value in hard_families.items() if value == 0],
        },
    }


def validate_d_pairs(
    records: Iterable[Mapping[str, Any]], *, min_pairs: int = 30
) -> dict[str, Any]:
    """Validate matched positive/negative D controls without treating them as T3."""

    rows = [row for row in _records(records) if _category(row) == "D"]
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _first(row, "matched_pair_id", "pair_id", "base_scenario_id", "scenario_id")
        if not isinstance(key, str) or not key.strip():
            key = str(row.get("task_id", ""))
        groups[key].append(row)
    valid_pairs: list[str] = []
    mismatches: dict[str, list[str]] = {}
    for key, pair in sorted(groups.items()):
        if len(pair) != 2:
            mismatches[key] = ["pair_size"]
            continue
        reasons: list[str] = []
        directions = {
            str(_first(row, "stipulated_direction", "direction", "label") or "") for row in pair
        }
        if len(directions) != 2:
            reasons.append("opposite_direction_missing")
        identities = {str(_first(row, "market", "entity", "security_id") or "") for row in pair}
        if len(identities) != 1:
            reasons.append("entity_or_market_mismatch")
        context = {
            str(_first(row, "context_hash", "scenario_hash", "prices_hash") or "") for row in pair
        }
        if len(context) != 1:
            reasons.append("context_mismatch")
        if not all("history_conflict" in row for row in pair):
            reasons.append("history_conflict_annotation_missing")
        if reasons:
            mismatches[key] = reasons
        else:
            valid_pairs.append(key)
    return {
        "schema_version": "d-pairs/v1",
        "record_count": len(rows),
        "pair_count": len(groups),
        "valid_pair_count": len(valid_pairs),
        "minimum_pair_count": min_pairs,
        "valid_pair_ids": valid_pairs,
        "mismatches": mismatches,
        "valid": len(valid_pairs) >= min_pairs,
        "deficit": max(0, min_pairs - len(valid_pairs)),
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


_LEDGER_ACTIONS = frozenset({"repair", "collect", "exclude", "regenerate"})
_LEDGER_STATUSES = frozenset({"planned", "in_progress", "delivered", "excluded"})


def _ledger_entry(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "ledger_id",
        "task_slice",
        "action",
        "reason_codes",
        "source_evidence",
        "owner",
        "status",
    }
    if set(value) != expected:
        raise DataInventoryError("collection ledger entry fields are not exact")
    if not isinstance(value["ledger_id"], str) or not value["ledger_id"].strip():
        raise DataInventoryError("ledger_id must be non-empty")
    if value["action"] not in _LEDGER_ACTIONS:
        raise DataInventoryError("collection ledger action is invalid")
    if value["status"] not in _LEDGER_STATUSES:
        raise DataInventoryError("collection ledger status is invalid")
    if not isinstance(value["reason_codes"], list) or any(
        not isinstance(item, str) or not item.strip() for item in value["reason_codes"]
    ):
        raise DataInventoryError("reason_codes must be a list of non-empty strings")
    if not isinstance(value["source_evidence"], list):
        raise DataInventoryError("source_evidence must be a list")
    if not isinstance(value["owner"], str) or not value["owner"].strip():
        raise DataInventoryError("owner must be non-empty")
    for evidence in value["source_evidence"]:
        if not isinstance(evidence, Mapping):
            raise DataInventoryError("source evidence must be objects")
        if not _url(evidence.get("url")) or not _sha(evidence.get("sha256")):
            raise DataInventoryError("source evidence needs a URL and SHA-256")
        if _timestamp(evidence.get("collected_at")) is None:
            raise DataInventoryError("source evidence collected_at must be timezone-aware")
    return {
        "ledger_id": str(value["ledger_id"]).strip(),
        "task_slice": str(value["task_slice"]).strip(),
        "action": value["action"],
        "reason_codes": list(value["reason_codes"]),
        "source_evidence": [dict(item) for item in value["source_evidence"]],
        "owner": str(value["owner"]).strip(),
        "status": value["status"],
    }


@dataclass(frozen=True, slots=True)
class CollectionLedger:
    """Append-only metadata disposition ledger with a deterministic seal."""

    entries: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        normalized = tuple(_ledger_entry(entry) for entry in self.entries)
        ids = [entry["ledger_id"] for entry in normalized]
        if len(ids) != len(set(ids)):
            raise DataInventoryError("collection ledger contains duplicate ledger_id values")
        object.__setattr__(self, "entries", normalized)

    @property
    def ledger_sha256(self) -> str:
        return _sha256_json(list(self.entries))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "collection-ledger/v1",
            "entries": [dict(entry) for entry in self.entries],
            "ledger_sha256": self.ledger_sha256,
        }

    def append(self, entry: Mapping[str, Any]) -> CollectionLedger:
        candidate = _ledger_entry(entry)
        if any(row["ledger_id"] == candidate["ledger_id"] for row in self.entries):
            raise DataInventoryError(f"duplicate ledger_id: {candidate['ledger_id']}")
        return CollectionLedger((*self.entries, candidate))


def build_disposition_ledger(records: Iterable[Mapping[str, Any]]) -> CollectionLedger:
    """Create one disposition row per B task and each missing predicate."""

    rows = _records(records)
    entries: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        task_id = _task_id(row, index)
        category = _category(row)
        reasons: list[str] = []
        if category == "B":
            ok, b_reasons = validate_b_event_chain(row)
            if not ok:
                reasons.extend(b_reasons)
        for predicate in V4_REQUIRED_ROW_PREDICATES:
            passed = (
                row.get(predicate) is True
                if predicate == "official_temporal_eligible"
                else row.get(predicate) == "reviewed"
                if predicate == "review_status"
                else _is_observed(row.get(predicate))
                if predicate == "outcome_available_at_source"
                else row.get(predicate) == "ready"
            )
            if not passed:
                reasons.append(f"missing_{predicate}")
        reasons = list(dict.fromkeys(reasons or ["source_row_ready"]))
        action = (
            "exclude"
            if category == "B" and any(reason.startswith("missing_") for reason in reasons)
            else "repair"
        )
        status = "excluded" if action == "exclude" else "planned"
        entries.append(
            {
                "ledger_id": f"{category.lower()}:{task_id}",
                "task_slice": f"{category}:{task_id}",
                "action": action,
                "reason_codes": reasons,
                "source_evidence": [],
                "owner": "unassigned",
                "status": status,
            }
        )
    return CollectionLedger(tuple(entries))


def write_collection_ledger(path: str | Path, ledger: CollectionLedger) -> Path:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise DataInventoryError(f"collection ledger is append-only and already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(_canonical(dict(entry)) + "\n" for entry in ledger.entries)
    target.write_text(lines, encoding="utf-8")
    manifest = target.with_name(target.name + ".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "collection-ledger-manifest/v1",
                "ledger_path": target.name,
                "entry_count": len(ledger.entries),
                "ledger_sha256": ledger.ledger_sha256,
                "sha256": hashlib.sha256(lines.encode("utf-8")).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def load_collection_ledger(path: str | Path) -> CollectionLedger:
    target = Path(path).expanduser().resolve()
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise DataInventoryError(f"collection ledger is unavailable: {target}") from exc
    entries = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError
            entries.append(_ledger_entry(value))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DataInventoryError(f"invalid collection ledger row {line_number}") from exc
    ledger = CollectionLedger(tuple(entries))
    manifest = target.with_name(target.name + ".manifest.json")
    if manifest.is_file():
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DataInventoryError("collection ledger manifest is invalid") from exc
        if value.get("ledger_sha256") != ledger.ledger_sha256:
            raise DataInventoryError("collection ledger manifest hash mismatch")
    return ledger


__all__ = [
    "CollectionLedger",
    "DataInventoryError",
    "EXPECTED_E_FORMULA_FAMILIES",
    "V4_DATASET_REPO_ID",
    "V4_DATASET_REVISION",
    "V4_DATASET_TREE_SHA",
    "V4_DATASET_TREE_SHA256",
    "V4_EXPECTED_CONFIG_COUNTS",
    "V4_EXPECTED_RECORD_COUNT",
    "build_disposition_ledger",
    "compute_inventory",
    "load_collection_ledger",
    "validate_b_event_chain",
    "validate_d_pairs",
    "validate_e_family_allocation",
    "write_collection_ledger",
]
