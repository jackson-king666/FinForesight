"""Pure, offline Evidence-QC calculations for the v8-R1 amendment.

This module intentionally does not know how a calibration bundle is collected
and never opens a file, starts a provider, or calls a probe.  The validator
module owns the I/O and protocol checks; this module owns the deterministic
statistics and sample routing that operate on already supplied mappings.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

QC_SCHEMA_VERSION = "evidence-qc/v1"
POLICY_SCHEMA_VERSION = "evidence-qc-policy/v1"
EXPECTED_ITEM_COUNT = 40
BOOTSTRAP_SEED = 20260831
BOOTSTRAP_RESAMPLES = 1_000
BOOTSTRAP_CONFIDENCE = 0.95

FIELDS = ("task_relevance", "outcome_revealing")
FIELD_LABELS = {
    "task_relevance": frozenset({"relevant", "not_relevant", "indeterminate"}),
    "outcome_revealing": frozenset({"direct", "indirect", "no", "indeterminate"}),
}
RESOLUTION_STATUSES = frozenset(
    {
        "pending_independent_c",
        "protocol_invalid",
        "resolved_adjudicated",
        "resolved_initial",
        "unresolved",
    }
)
TIER_ORDER = {"not_qualified": 0, "limited": 1, "qualified": 2, "strong": 3}


class EvidenceQCError(ValueError):
    """Raised when Evidence-QC input cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class TierRule:
    """Integer-count and kappa gate for one claim-strength tier."""

    name: str
    min_agreement_count: int
    min_kappa: float

    def as_dict(self) -> dict[str, object]:
        return {
            "min_agreement_count": self.min_agreement_count,
            "min_kappa": self.min_kappa,
            "name": self.name,
        }


DEFAULT_TIER_RULES = (
    TierRule("strong", 36, 0.80),
    TierRule("qualified", 34, 0.60),
    TierRule("limited", 32, 0.50),
)


def _json_hash(value: object) -> str:
    import json

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _as_sequence(
    rows: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    if isinstance(rows, Mapping) or isinstance(rows, str | bytes):
        raise EvidenceQCError("label rows must be a sequence of objects")
    result = list(rows)
    if any(not isinstance(row, Mapping) for row in result):
        raise EvidenceQCError("every label row must be an object")
    return result


def _index_rows(
    rows: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    for row in _as_sequence(rows):
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise EvidenceQCError("every label row needs a non-empty item_id")
        if item_id in indexed:
            raise EvidenceQCError(f"duplicate item_id: {item_id}")
        indexed[item_id] = row
    return indexed


def _index_views(
    views: Sequence[Mapping[str, object]]
    | Iterable[Mapping[str, object]]
    | Mapping[str, Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    if isinstance(views, Mapping):
        indexed = dict(views)
        if any(
            not isinstance(key, str) or not isinstance(value, Mapping)
            for key, value in indexed.items()
        ):
            raise EvidenceQCError("views must map string item IDs to objects")
        for item_id, view in indexed.items():
            if view.get("item_id", item_id) != item_id:
                raise EvidenceQCError(f"view item_id mismatch: {item_id}")
        return indexed
    indexed: dict[str, Mapping[str, object]] = {}
    for view in _as_sequence(views):
        item_id = view.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise EvidenceQCError("every view needs a non-empty item_id")
        if item_id in indexed:
            raise EvidenceQCError(f"duplicate view item_id: {item_id}")
        indexed[item_id] = view
    return indexed


def _field_pairs(
    labels_a: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    labels_b: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    field: str,
) -> tuple[list[str], list[object], list[object]]:
    if field not in FIELD_LABELS:
        raise EvidenceQCError(f"unsupported Evidence-QC field: {field}")
    indexed_a = _index_rows(labels_a)
    indexed_b = _index_rows(labels_b)
    if set(indexed_a) != set(indexed_b):
        missing_a = sorted(set(indexed_b) - set(indexed_a))
        missing_b = sorted(set(indexed_a) - set(indexed_b))
        raise EvidenceQCError(
            f"raw label coverage differs (missing_a={missing_a}, missing_b={missing_b})"
        )
    item_ids = sorted(indexed_a)
    values_a = [indexed_a[item_id].get(field) for item_id in item_ids]
    values_b = [indexed_b[item_id].get(field) for item_id in item_ids]
    allowed = FIELD_LABELS[field]
    for reviewer, values in (("A", values_a), ("B", values_b)):
        if any(value not in allowed for value in values):
            raise EvidenceQCError(f"{reviewer} contains an unsupported {field} label")
    return item_ids, values_a, values_b


def agreement_count(values_a: Sequence[object], values_b: Sequence[object]) -> int:
    """Return the exact number of equal paired labels."""

    if len(values_a) != len(values_b):
        raise EvidenceQCError("paired label vectors must have equal length")
    if not values_a:
        raise EvidenceQCError("paired label vectors must not be empty")
    return sum(left == right for left, right in zip(values_a, values_b, strict=True))


def agreement_rate(values_a: Sequence[object], values_b: Sequence[object]) -> float:
    """Return raw agreement without rounding."""

    return agreement_count(values_a, values_b) / len(values_a)


def cohen_kappa(values_a: Sequence[object], values_b: Sequence[object]) -> float | None:
    """Compute Cohen's kappa from paired categorical labels.

    A zero denominator is returned as ``None``.  Treating a degenerate
    marginal distribution as a perfect reliability score would make the
    policy's ``kappa undefined`` integrity/quality rule impossible to audit.
    """

    if len(values_a) != len(values_b):
        raise EvidenceQCError("paired label vectors must have equal length")
    if not values_a:
        raise EvidenceQCError("paired label vectors must not be empty")
    n = len(values_a)
    observed = agreement_count(values_a, values_b) / n
    counts_a = Counter(values_a)
    counts_b = Counter(values_b)
    categories = set(counts_a) | set(counts_b)
    expected = sum(counts_a[value] * counts_b[value] for value in categories) / (n * n)
    if expected == 1.0:
        return None
    result = (observed - expected) / (1.0 - expected)
    if not math.isfinite(result):
        return None
    return result


def wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> tuple[float, float]:
    """Return a deterministic Wilson interval for a binomial proportion."""

    if trials <= 0 or successes < 0 or successes > trials:
        raise EvidenceQCError("Wilson interval counts are invalid")
    if not 0.0 < confidence < 1.0:
        raise EvidenceQCError("confidence must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    n = float(trials)
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise EvidenceQCError("cannot take a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise EvidenceQCError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_bootstrap_kappa(
    values_a: Sequence[object],
    values_b: Sequence[object],
    *,
    cluster_keys: Sequence[object] | None = None,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, object]:
    """Compute a seeded cluster-bootstrap confidence interval for kappa."""

    if len(values_a) != len(values_b) or not values_a:
        raise EvidenceQCError("bootstrap vectors must be non-empty and parallel")
    if cluster_keys is None:
        cluster_keys = list(range(len(values_a)))
    if len(cluster_keys) != len(values_a):
        raise EvidenceQCError("cluster keys must align with labels")
    if resamples <= 0:
        raise EvidenceQCError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise EvidenceQCError("confidence must be in (0, 1)")

    grouped: dict[object, list[tuple[object, object]]] = defaultdict(list)
    for key, value_a, value_b in zip(cluster_keys, values_a, values_b, strict=True):
        try:
            hash(key)
        except TypeError as exc:
            raise EvidenceQCError("cluster keys must be hashable") from exc
        grouped[key].append((value_a, value_b))
    ordered_keys = sorted(grouped, key=lambda value: (type(value).__name__, repr(value)))
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        sampled_pairs = [
            pair
            for key in (rng.choice(ordered_keys) for _ in ordered_keys)
            for pair in grouped[key]
        ]
        sample_a = [pair[0] for pair in sampled_pairs]
        sample_b = [pair[1] for pair in sampled_pairs]
        kappa = cohen_kappa(sample_a, sample_b)
        if kappa is not None:
            samples.append(kappa)
    alpha = (1.0 - confidence) / 2.0
    interval = None
    if samples:
        interval = {
            "lower": _percentile(samples, alpha),
            "upper": _percentile(samples, 1.0 - alpha),
        }
    return {
        "cluster_count": len(ordered_keys),
        "confidence": confidence,
        "interval": interval,
        "resamples": resamples,
        "resamples_used": len(samples),
        "seed": seed,
    }


def _rules_from_policy(policy: Mapping[str, object] | None) -> tuple[TierRule, ...]:
    if policy is None:
        return DEFAULT_TIER_RULES
    raw_rules = policy.get("tiers")
    if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, str | bytes):
        raise EvidenceQCError("policy tiers must be a sequence")
    rules: list[TierRule] = []
    for raw in raw_rules:
        if not isinstance(raw, Mapping):
            raise EvidenceQCError("each policy tier must be an object")
        name = raw.get("name")
        count = raw.get("min_agreement_count")
        kappa = raw.get("min_kappa")
        if (
            not isinstance(name, str)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or not isinstance(kappa, int | float)
            or isinstance(kappa, bool)
        ):
            raise EvidenceQCError("policy tier has invalid threshold types")
        rules.append(TierRule(name, count, float(kappa)))
    if not rules:
        raise EvidenceQCError("policy must define at least one tier")
    return tuple(rules)


def classify_tier(
    exact_agreement_count: int,
    kappa: float | None,
    *,
    rules: Sequence[TierRule] = DEFAULT_TIER_RULES,
) -> str:
    """Assign a tier using integer agreement and an unrounded kappa."""

    if exact_agreement_count < 0:
        raise EvidenceQCError("agreement count must be non-negative")
    for rule in rules:
        if rule.name not in TIER_ORDER:
            raise EvidenceQCError(f"unsupported tier name: {rule.name}")
        if (
            exact_agreement_count >= rule.min_agreement_count
            and kappa is not None
            and math.isfinite(kappa)
            and kappa >= rule.min_kappa
        ):
            return rule.name
    return "not_qualified"


def compute_field_metrics(
    field: str,
    values_a: Sequence[object],
    values_b: Sequence[object],
    *,
    cluster_keys: Sequence[object] | None = None,
    tier_rules: Sequence[TierRule] = DEFAULT_TIER_RULES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    bootstrap_confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, object]:
    """Recompute one field's agreement, marginals, confusion, and intervals."""

    if field not in FIELD_LABELS:
        raise EvidenceQCError(f"unsupported Evidence-QC field: {field}")
    allowed = FIELD_LABELS[field]
    if any(value not in allowed for value in (*values_a, *values_b)):
        raise EvidenceQCError(f"unsupported value in {field}")
    n = len(values_a)
    exact = agreement_count(values_a, values_b)
    kappa = cohen_kappa(values_a, values_b)
    categories = sorted(set(values_a) | set(values_b), key=str)
    confusion = {str(left): {str(right): 0 for right in categories} for left in categories}
    for left, right in zip(values_a, values_b, strict=True):
        confusion[str(left)][str(right)] += 1
    counts_a = Counter(str(value) for value in values_a)
    counts_b = Counter(str(value) for value in values_b)
    class_specific = {
        category: {
            "a_count": counts_a.get(category, 0),
            "agreement_count": confusion[category].get(category, 0),
            "agreement_rate_of_input": confusion[category].get(category, 0) / n,
            "b_count": counts_b.get(category, 0),
        }
        for category in categories
    }
    kappa_ci = cluster_bootstrap_kappa(
        values_a,
        values_b,
        cluster_keys=cluster_keys,
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
        confidence=bootstrap_confidence,
    )
    return {
        "agreement_count": exact,
        "agreement_rate": exact / n,
        "agreement_ci": {
            "lower": wilson_interval(exact, n, confidence=bootstrap_confidence)[0],
            "upper": wilson_interval(exact, n, confidence=bootstrap_confidence)[1],
        },
        "class_specific_agreement": class_specific,
        "cohen_kappa": kappa,
        "cohen_kappa_ci": kappa_ci,
        "confusion_matrix": confusion,
        "field": field,
        "marginals": {"a": dict(counts_a), "b": dict(counts_b)},
        "n": n,
        "tier": classify_tier(exact, kappa, rules=tier_rules),
    }


def compute_pre_adjudication_metrics(
    labels_a: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    labels_b: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    *,
    policy: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Recompute both fields from raw A/B rows before any C resolution."""

    indexed_a = _index_rows(labels_a)
    indexed_b = _index_rows(labels_b)
    if set(indexed_a) != set(indexed_b):
        raise EvidenceQCError("raw A/B rows do not have exact item coverage")
    item_ids = sorted(indexed_a)
    policy_bootstrap = policy.get("bootstrap", {}) if policy else {}
    if not isinstance(policy_bootstrap, Mapping):
        raise EvidenceQCError("policy bootstrap settings must be an object")
    seed = policy_bootstrap.get("seed", BOOTSTRAP_SEED)
    resamples = policy_bootstrap.get("resamples", BOOTSTRAP_RESAMPLES)
    confidence = policy_bootstrap.get("confidence", BOOTSTRAP_CONFIDENCE)
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or not isinstance(resamples, int)
        or isinstance(resamples, bool)
        or not isinstance(confidence, int | float)
        or isinstance(confidence, bool)
    ):
        raise EvidenceQCError("policy bootstrap settings have invalid types")
    rules = _rules_from_policy(policy)
    fields: dict[str, object] = {}
    for field in FIELDS:
        values_a = [indexed_a[item_id].get(field) for item_id in item_ids]
        values_b = [indexed_b[item_id].get(field) for item_id in item_ids]
        fields[field] = compute_field_metrics(
            field,
            values_a,
            values_b,
            cluster_keys=item_ids,
            tier_rules=rules,
            bootstrap_seed=seed,
            bootstrap_resamples=resamples,
            bootstrap_confidence=float(confidence),
        )
    return {
        "bootstrap": {
            "confidence": float(confidence),
            "cluster_key": policy_bootstrap.get("cluster_key", "item_id"),
            "resamples": resamples,
            "seed": seed,
        },
        "fields": fields,
        "item_count": len(item_ids),
        "item_ids_sha256": _json_hash(item_ids),
        "schema_version": QC_SCHEMA_VERSION,
    }


def build_disagreement_queue(
    views: Sequence[Mapping[str, object]]
    | Iterable[Mapping[str, object]]
    | Mapping[str, Mapping[str, object]],
    labels_a: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    labels_b: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build a C queue that contains no A/B label values or rationale."""

    view_index = _index_views(views)
    index_a = _index_rows(labels_a)
    index_b = _index_rows(labels_b)
    if set(view_index) != set(index_a) or set(index_a) != set(index_b):
        raise EvidenceQCError("disagreement queue requires exact view/A/B coverage")
    queue: list[dict[str, object]] = []
    for item_id in sorted(view_index):
        fields = [
            field for field in FIELDS if index_a[item_id].get(field) != index_b[item_id].get(field)
        ]
        if not fields:
            continue
        view_hash = view_index[item_id].get("annotation_view_sha256")
        if not isinstance(view_hash, str) or len(view_hash) != 64:
            raise EvidenceQCError(f"view hash is missing for {item_id}")
        queue.append(
            {
                "annotation_view_sha256": view_hash,
                "disagreement_fields": fields,
                "item_id": item_id,
                "queue_status": "pending_c",
            }
        )
    return queue


def build_sensitivity_plan() -> dict[str, object]:
    """Return the pre-locked three-class and binary sensitivity definitions."""

    return {
        "binary_sensitivity": [
            {"id": "direct_only", "include": ["direct"]},
            {
                "id": "revealing_direct_or_indirect",
                "include": ["direct", "indirect"],
            },
        ],
        "main_outcome_categories": ["no", "indirect", "direct"],
        "schema_version": QC_SCHEMA_VERSION,
        "unresolved_policy": {
            "bounds": "fixed_input_best_worst",
            "point_estimate": "exclude_unresolved_and_protocol_invalid",
        },
    }


def _c_value(row: Mapping[str, object], field: str) -> object | None:
    value = row.get(field)
    if value is None or (isinstance(value, str) and value in {"abstain", "indeterminate"}):
        return None
    return value


def resolve_three_way(
    views: Sequence[Mapping[str, object]]
    | Iterable[Mapping[str, object]]
    | Mapping[str, Mapping[str, object]],
    labels_a: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    labels_b: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    labels_c: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    disagreement_queue: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Resolve only queued disagreements using a blind C row and 2/3 majority."""

    view_index = _index_views(views)
    index_a = _index_rows(labels_a)
    index_b = _index_rows(labels_b)
    index_c = _index_rows(labels_c)
    if set(view_index) != set(index_a) or set(index_a) != set(index_b):
        raise EvidenceQCError("three-way resolution requires exact view/A/B coverage")
    queue_index: dict[str, Mapping[str, object]] = {}
    for row in disagreement_queue:
        if not isinstance(row, Mapping):
            raise EvidenceQCError("disagreement queue rows must be objects")
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or item_id in queue_index:
            raise EvidenceQCError("disagreement queue IDs must be non-empty and unique")
        queue_index[item_id] = row
    expected_queue = {
        item_id
        for item_id in index_a
        if any(index_a[item_id].get(field) != index_b[item_id].get(field) for field in FIELDS)
    }
    if set(queue_index) != expected_queue:
        raise EvidenceQCError("disagreement queue does not match raw A/B differences")
    if set(index_c) - expected_queue:
        raise EvidenceQCError("C may label only queued disagreements")

    results: list[dict[str, object]] = []
    for item_id in sorted(index_a):
        view_hash = view_index[item_id].get("annotation_view_sha256")
        if not isinstance(view_hash, str) or len(view_hash) != 64:
            raise EvidenceQCError(f"view hash is missing for {item_id}")
        fields: dict[str, object] = {}
        resolved_labels: dict[str, object] = {}
        protocol_invalid = False
        pending_independent_c = False
        unresolved_reasons: set[str] = set()
        for field in FIELDS:
            value_a = index_a[item_id].get(field)
            value_b = index_b[item_id].get(field)
            if value_a not in FIELD_LABELS[field] or value_b not in FIELD_LABELS[field]:
                protocol_invalid = True
                fields[field] = {"status": "protocol_invalid"}
                continue
            if value_a == value_b:
                if item_id in index_c and index_c[item_id].get(field) not in {None, value_a}:
                    protocol_invalid = True
                    fields[field] = {"status": "protocol_invalid"}
                else:
                    resolved_labels[field] = value_a
                    fields[field] = {
                        "label": value_a,
                        "status": "resolved_initial",
                    }
                continue
            if item_id not in queue_index:
                protocol_invalid = True
                fields[field] = {"status": "protocol_invalid"}
                continue
            c_row = index_c.get(item_id)
            if c_row is None:
                pending_independent_c = True
                fields[field] = {
                    "label_a": value_a,
                    "label_b": value_b,
                    "status": "pending_independent_c",
                }
                continue
            value_c = _c_value(c_row, field)
            if value_c not in FIELD_LABELS[field]:
                unresolved_reasons.add("c_abstain")
                fields[field] = {
                    "label_a": value_a,
                    "label_b": value_b,
                    "reason_code": "c_abstain",
                    "status": "unresolved",
                }
                continue
            if value_c == value_a or value_c == value_b:
                resolved_labels[field] = value_c
                fields[field] = {
                    "label": value_c,
                    "label_a": value_a,
                    "label_b": value_b,
                    "label_c": value_c,
                    "status": "resolved_adjudicated",
                }
            else:
                unresolved_reasons.add("c_three_distinct")
                fields[field] = {
                    "label_a": value_a,
                    "label_b": value_b,
                    "label_c": value_c,
                    "reason_code": "c_three_distinct",
                    "status": "unresolved",
                }
        if protocol_invalid:
            status = "protocol_invalid"
            reason_code = "protocol_invalid_label_or_override"
        elif pending_independent_c:
            status = "pending_independent_c"
            reason_code = "pending_independent_c"
        elif all(value.get("status") == "resolved_initial" for value in fields.values()):
            status = "resolved_initial"
            reason_code = "initial_consensus"
        elif all(
            value.get("status") in {"resolved_initial", "resolved_adjudicated"}
            for value in fields.values()
        ):
            status = "resolved_adjudicated"
            reason_code = "c_two_of_three"
        else:
            status = "unresolved"
            reason_code = (
                "c_abstain"
                if unresolved_reasons == {"c_abstain"}
                else "c_three_distinct"
                if unresolved_reasons == {"c_three_distinct"}
                else "c_abstain_or_three_distinct"
            )
        results.append(
            {
                "annotation_view_sha256": view_hash,
                "field_resolutions": fields,
                "item_id": item_id,
                "reason_code": reason_code,
                "resolution_status": status,
                "resolved_labels": resolved_labels,
            }
        )
    return results


def _provenance_qualified(view: Mapping[str, object]) -> bool:
    explicit = view.get("provenance_qualified")
    if isinstance(explicit, bool):
        return explicit
    snapshot_status = view.get("restricted_snapshot_status")
    if snapshot_status is None:
        return True
    return snapshot_status == "available"


def route_samples(
    views: Sequence[Mapping[str, object]]
    | Iterable[Mapping[str, object]]
    | Mapping[str, Mapping[str, object]],
    resolutions: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Route every input item and prove count conservation for the overlay."""

    view_index = _index_views(views)
    resolution_index = _index_rows(resolutions)
    if set(view_index) != set(resolution_index):
        raise EvidenceQCError("routing requires one resolution row per input view")
    status_counts = Counter(
        str(resolution_index[item_id].get("resolution_status")) for item_id in view_index
    )
    if any(status not in RESOLUTION_STATUSES for status in status_counts):
        raise EvidenceQCError("routing received an unsupported resolution status")

    item_routes: list[dict[str, object]] = []
    relevance_ids: list[str] = []
    outcome_ids: list[str] = []
    outcome_counts: Counter[str] = Counter()
    sensitivity_ids = {"direct_only": [], "revealing_direct_or_indirect": []}
    stratum_totals: Counter[str] = Counter()
    stratum_resolved: Counter[str] = Counter()
    unresolved_or_invalid = sum(
        status_counts.get(status, 0)
        for status in {"pending_independent_c", "protocol_invalid", "unresolved"}
    )
    for item_id in sorted(view_index):
        view = view_index[item_id]
        resolution = resolution_index[item_id]
        status = str(resolution["resolution_status"])
        stratum = view.get("sampling_stratum")
        if not isinstance(stratum, str) or not stratum:
            category = view.get("category", "unknown")
            band = view.get("paper_band", "unknown")
            stratum = f"{category}-{band}"
        stratum_totals[stratum] += 1
        if status in {"resolved_initial", "resolved_adjudicated"}:
            stratum_resolved[stratum] += 1
        labels = resolution.get("resolved_labels")
        labels = labels if isinstance(labels, Mapping) else {}
        relevance = labels.get("task_relevance")
        outcome = labels.get("outcome_revealing")
        provenance = _provenance_qualified(view)
        relevance_eligible = (
            status in {"resolved_initial", "resolved_adjudicated"}
            and provenance
            and relevance in {"relevant", "not_relevant"}
        )
        outcome_eligible = relevance == "relevant" and outcome in {"direct", "indirect", "no"}
        if relevance_eligible:
            relevance_ids.append(item_id)
        if (
            outcome_eligible
            and status in {"resolved_initial", "resolved_adjudicated"}
            and provenance
        ):
            outcome_ids.append(item_id)
            outcome_counts[str(outcome)] += 1
            if outcome == "direct":
                sensitivity_ids["direct_only"].append(item_id)
            if outcome in {"direct", "indirect"}:
                sensitivity_ids["revealing_direct_or_indirect"].append(item_id)
        if status == "protocol_invalid":
            reason = "protocol_invalid"
        elif status == "pending_independent_c":
            reason = "pending_independent_c"
        elif status == "unresolved":
            reason = str(resolution.get("reason_code", "unresolved"))
        elif not provenance:
            reason = "provenance_not_qualified"
        elif relevance == "indeterminate":
            reason = "relevance_indeterminate"
        elif relevance != "relevant":
            reason = "relevance_not_relevant"
        elif outcome not in {"direct", "indirect", "no"}:
            reason = "outcome_indeterminate"
        else:
            reason = "eligible"
        item_routes.append(
            {
                "item_id": item_id,
                "outcome_revealing": outcome,
                "provenance_qualified": provenance,
                "reason_code": reason,
                "resolution_status": status,
                "task_relevance": relevance,
            }
        )

    input_count = len(view_index)
    conserved = sum(status_counts.get(status, 0) for status in RESOLUTION_STATUSES)
    if conserved != input_count:
        raise EvidenceQCError("routing status counts do not conserve N_input")
    task_mask = sorted(view_index)
    bounds = {
        "direct": {
            "lower": len(sensitivity_ids["direct_only"]),
            "upper": len(sensitivity_ids["direct_only"]) + unresolved_or_invalid,
        },
        "revealing_direct_or_indirect": {
            "lower": len(sensitivity_ids["revealing_direct_or_indirect"]),
            "upper": len(sensitivity_ids["revealing_direct_or_indirect"]) + unresolved_or_invalid,
        },
    }
    return {
        "N_input": input_count,
        "N_pending_independent_c": status_counts.get("pending_independent_c", 0),
        "N_protocol_invalid": status_counts.get("protocol_invalid", 0),
        "N_resolved_adjudicated": status_counts.get("resolved_adjudicated", 0),
        "N_resolved_initial": status_counts.get("resolved_initial", 0),
        "N_unresolved": status_counts.get("unresolved", 0),
        "input_denominator_bounds": bounds,
        "item_routes": item_routes,
        "mask_sha256": _json_hash(task_mask),
        "outcome_three_class": {
            "counts": dict(outcome_counts),
            "item_ids": outcome_ids,
        },
        "relevance_resolved_item_ids": relevance_ids,
        "schema_version": QC_SCHEMA_VERSION,
        "sensitivity": {
            key: {"count": len(value), "item_ids": value} for key, value in sensitivity_ids.items()
        },
        "status_counts": {
            status: status_counts.get(status, 0) for status in sorted(RESOLUTION_STATUSES)
        },
        "strata": {
            stratum: {
                "input": stratum_totals[stratum],
                "resolved": stratum_resolved[stratum],
            }
            for stratum in sorted(stratum_totals)
        },
        "task_mask": task_mask,
    }


def assess_evidence_qc(
    views: Sequence[Mapping[str, object]]
    | Iterable[Mapping[str, object]]
    | Mapping[str, Mapping[str, object]],
    labels_a: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    labels_b: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    *,
    policy: Mapping[str, object] | None = None,
    labels_c: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build an offline assessment, optionally after a supplied C pass."""

    view_index = _index_views(views)
    metrics = compute_pre_adjudication_metrics(labels_a, labels_b, policy=policy)
    queue = build_disagreement_queue(view_index, labels_a, labels_b)
    if labels_c is None:
        c_rows: list[Mapping[str, object]] = []
    else:
        c_rows = _as_sequence(labels_c)
    resolutions = resolve_three_way(view_index, labels_a, labels_b, c_rows, queue)
    routing = route_samples(view_index, resolutions)
    fields = metrics["fields"]
    assert isinstance(fields, Mapping)
    tier_values = [str(fields[field]["tier"]) for field in FIELDS]
    lowest_tier = min(tier_values, key=lambda value: TIER_ORDER[value])
    qualified_coverage = routing["N_resolved_initial"] + routing[
        "N_resolved_adjudicated"
    ] >= 36 and all(stratum_data["resolved"] >= 8 for stratum_data in routing["strata"].values())
    pending_c = bool(queue) and not c_rows
    if pending_c:
        disposition = "PENDING_INDEPENDENT_C"
    elif lowest_tier in {"strong", "qualified"} and qualified_coverage:
        disposition = "CONTINUE_EVIDENCE_QUALIFIED"
    else:
        disposition = "CONTINUE_EVIDENCE_LIMITED/NOT_QUALIFIED"
    return {
        "disagreement_count": len(queue),
        "disagreement_queue": queue,
        "field_tiers": {field: fields[field]["tier"] for field in FIELDS},
        "lowest_field_tier": lowest_tier,
        "metrics": metrics,
        "qc_disposition": disposition,
        "resolution": resolutions,
        "routing": routing,
        "schema_version": QC_SCHEMA_VERSION,
        "sensitivity_plan": build_sensitivity_plan(),
    }


__all__ = [
    "BOOTSTRAP_CONFIDENCE",
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "DEFAULT_TIER_RULES",
    "EXPECTED_ITEM_COUNT",
    "EvidenceQCError",
    "FIELDS",
    "FIELD_LABELS",
    "POLICY_SCHEMA_VERSION",
    "QC_SCHEMA_VERSION",
    "RESOLUTION_STATUSES",
    "TierRule",
    "agreement_count",
    "agreement_rate",
    "assess_evidence_qc",
    "build_disagreement_queue",
    "build_sensitivity_plan",
    "classify_tier",
    "cluster_bootstrap_kappa",
    "cohen_kappa",
    "compute_field_metrics",
    "compute_pre_adjudication_metrics",
    "resolve_three_way",
    "route_samples",
    "wilson_interval",
]
