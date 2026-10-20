"""Full-run human leak-review contract (``leak-review-*/v1``).

This module is the v4 full-release human leak-review gate from the HUA-40
spec (section 5.8).  It has a new identity and never touches the frozen v8-R1
Evidence-QC policy or the protected historical 400 items; it reuses the pure
agreement mechanics of ``benchmark_harness.calibration.evidence_qc`` and the
canonical hashing helpers of ``benchmark_harness.calibration.qc_validator``.

The strong gate thresholds are module constants with no function parameter
that could lower them: raw A/B exact agreement >= 0.90 AND unrounded Cohen
kappa >= 0.80, separately for ``task_relevance`` and ``outcome_revealing``,
with the agreement-count requirement ``ceil(0.90 * N)`` for the actual
denominator N.  Undefined kappa does not pass, and an empty denominator
fails closed.

Adjudication resolves disagreements for downstream routing only.  It cannot
raise the pre-adjudication kappa: ``agreement_gate`` always recomputes from
the raw A/B rows, so a report produced after ``adjudicate`` is byte-identical
to the one produced before it (asserted in ``tests/test_leak_review.py``).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

from benchmark_harness.calibration.evidence_qc import (
    BOOTSTRAP_CONFIDENCE,
    FIELD_LABELS,
    RESOLUTION_STATUSES,
    agreement_count,
    build_disagreement_queue,
    cluster_bootstrap_kappa,
    cohen_kappa,
    resolve_three_way,
    wilson_interval,
)
from benchmark_harness.calibration.qc_validator import (
    SHA256_RE,
    canonical_json_bytes,
    sha256_bytes,
)

LEAK_REVIEW_FRAME_SCHEMA = "leak-review-frame/v1"
LEAK_REVIEW_LABEL_SCHEMA = "leak-review-label/v1"
LEAK_REVIEW_GATE_SCHEMA = "leak-review-gate/v1"
LEAK_REVIEW_JOIN_SCHEMA = "leak-review-join/v1"
STRONG_AGREEMENT_FLOOR = 0.90
STRONG_KAPPA_FLOOR = 0.80
FIELDS = ("task_relevance", "outcome_revealing")
TEMPORAL_LABELS = ("pre_cutoff", "post_cutoff", "indeterminate")

_RESOLVED_STATUSES = frozenset({"resolved_adjudicated", "resolved_initial"})
_ROSTER_ROLES = ("collector", "reviewer_a", "reviewer_b", "adjudicator_c")
_RESOLVED_LABEL_KEYS = frozenset({"outcome_revealing", "task_relevance", "temporal_status"})


class LeakReviewError(ValueError):
    """Raised when the full-run leak-review contract cannot be applied safely."""


def _non_empty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _parse_rfc3339(value: object, label: str) -> datetime:
    if not _non_empty_str(value):
        raise LeakReviewError(f"{label} must be an RFC3339 timestamp string")
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise LeakReviewError(f"{label} is not a valid RFC3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        raise LeakReviewError(f"{label} must carry a timezone offset")
    return parsed


def _as_rows(
    rows: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    label: str,
) -> list[Mapping[str, object]]:
    if isinstance(rows, Mapping) or isinstance(rows, str | bytes):
        raise LeakReviewError(f"{label} must be a sequence of objects")
    result = list(rows)
    if any(not isinstance(row, Mapping) for row in result):
        raise LeakReviewError(f"{label} rows must be objects")
    return result


@dataclass(frozen=True, slots=True)
class ReviewerIdentity:
    """One roster identity; the full-run leak review admits humans only."""

    identity: str
    reviewer_type: str = "human"

    def __post_init__(self) -> None:
        if not _non_empty_str(self.identity):
            raise LeakReviewError("reviewer identity must be a non-empty string")
        if self.reviewer_type != "human":
            raise LeakReviewError(
                "reviewer_type must be 'human': agent/tool/bot identities may not sign labels"
            )

    def to_dict(self) -> dict[str, str]:
        return {"identity": self.identity, "reviewer_type": self.reviewer_type}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ReviewerIdentity:
        if not isinstance(value, Mapping) or set(value) != {"identity", "reviewer_type"}:
            raise LeakReviewError("reviewer entries must carry exactly identity/reviewer_type")
        return cls(identity=value["identity"], reviewer_type=value["reviewer_type"])


def _canonical_sampling_policy(policy: object) -> Mapping[str, object]:
    if not isinstance(policy, Mapping):
        raise LeakReviewError("sampling_policy must be an object")
    kind = policy.get("kind")
    if kind == "review_all":
        if set(policy) != {"kind"}:
            raise LeakReviewError("review_all sampling policy carries no extra keys")
        return MappingProxyType({"kind": "review_all"})
    if kind != "probability":
        raise LeakReviewError("sampling_policy kind must be review_all or probability")
    if set(policy) != {"kind", "inclusion_probabilities"}:
        raise LeakReviewError("probability sampling policy needs exactly inclusion_probabilities")
    raw = policy.get("inclusion_probabilities")
    if not isinstance(raw, Mapping) or not raw:
        raise LeakReviewError("inclusion_probabilities must be a non-empty object")
    probabilities: dict[str, float] = {}
    for stratum, probability in raw.items():
        if not _non_empty_str(stratum):
            raise LeakReviewError("inclusion probability strata must be non-empty strings")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, int | float)
            or not math.isfinite(probability)
            or not 0.0 < float(probability) <= 1.0
        ):
            raise LeakReviewError("inclusion probabilities must lie in (0, 1]")
        probabilities[stratum] = float(probability)
    return MappingProxyType(
        {"inclusion_probabilities": MappingProxyType(probabilities), "kind": "probability"}
    )


def _sampling_policy_dict(policy: Mapping[str, object]) -> dict[str, object]:
    if policy["kind"] == "review_all":
        return {"kind": "review_all"}
    return {
        "inclusion_probabilities": dict(policy["inclusion_probabilities"]),
        "kind": "probability",
    }


@dataclass(frozen=True, slots=True)
class ReviewFrame:
    """Frozen review frame (schema ``leak-review-frame/v1``).

    The frame seals the rubric hash, unit of analysis, dedup rule, task/horizon
    scope, the four distinct human roster identities, and the sampling policy
    before any labeling.  ``frozen_at`` is a declared timezone-aware datetime;
    this module never reads a machine clock.
    """

    rubric_sha256: str
    unit_of_analysis: str
    dedup_rule: str
    task_scope: tuple[str, ...]
    collector: ReviewerIdentity
    reviewer_a: ReviewerIdentity
    reviewer_b: ReviewerIdentity
    adjudicator_c: ReviewerIdentity
    sampling_policy: Mapping[str, object]
    frozen_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.rubric_sha256, str) or not SHA256_RE.fullmatch(self.rubric_sha256):
            raise LeakReviewError("rubric_sha256 must be a lowercase SHA-256 digest")
        for name in ("unit_of_analysis", "dedup_rule"):
            if not _non_empty_str(getattr(self, name)):
                raise LeakReviewError(f"{name} must be a non-empty string")
        scope = tuple(self.task_scope)
        if not scope or any(not _non_empty_str(entry) for entry in scope):
            raise LeakReviewError("task_scope must be a non-empty tuple of scope strings")
        object.__setattr__(self, "task_scope", scope)
        roster = (self.collector, self.reviewer_a, self.reviewer_b, self.adjudicator_c)
        if any(not isinstance(reviewer, ReviewerIdentity) for reviewer in roster):
            raise LeakReviewError("roster entries must be ReviewerIdentity values")
        identities = [reviewer.identity for reviewer in roster]
        if len(set(identities)) != len(identities):
            raise LeakReviewError(
                "collector, reviewer_a, reviewer_b, adjudicator_c must be four distinct identities"
            )
        object.__setattr__(
            self, "sampling_policy", _canonical_sampling_policy(self.sampling_policy)
        )
        if (
            not isinstance(self.frozen_at, datetime)
            or self.frozen_at.tzinfo is None
            or self.frozen_at.tzinfo.utcoffset(self.frozen_at) is None
        ):
            raise LeakReviewError("frozen_at must be a timezone-aware RFC3339 datetime")

    def roster_identities(self) -> dict[str, str]:
        return {role: getattr(self, role).identity for role in _ROSTER_ROLES}

    @property
    def frame_sha256(self) -> str:
        """Self-hash over the canonical payload (excludes the hash itself)."""

        return sha256_bytes(canonical_json_bytes(self._payload()))

    def _payload(self) -> dict[str, object]:
        return {
            "dedup_rule": self.dedup_rule,
            "frozen_at": self.frozen_at.isoformat(),
            "roster": {role: getattr(self, role).to_dict() for role in _ROSTER_ROLES},
            "rubric_sha256": self.rubric_sha256,
            "sampling_policy": _sampling_policy_dict(self.sampling_policy),
            "schema_version": LEAK_REVIEW_FRAME_SCHEMA,
            "task_scope": list(self.task_scope),
            "unit_of_analysis": self.unit_of_analysis,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "frame_sha256": self.frame_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ReviewFrame:
        if not isinstance(value, Mapping):
            raise LeakReviewError("frame must be an object")
        expected = {
            "dedup_rule",
            "frame_sha256",
            "frozen_at",
            "roster",
            "rubric_sha256",
            "sampling_policy",
            "schema_version",
            "task_scope",
            "unit_of_analysis",
        }
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise LeakReviewError(f"frame fields are not exact (missing={missing}, extra={extra})")
        if value["schema_version"] != LEAK_REVIEW_FRAME_SCHEMA:
            raise LeakReviewError("unsupported leak-review frame schema")
        embedded = value["frame_sha256"]
        if not isinstance(embedded, str) or not SHA256_RE.fullmatch(embedded):
            raise LeakReviewError("frame_sha256 must be a lowercase SHA-256 digest")
        payload = {key: value[key] for key in expected - {"frame_sha256"}}
        if sha256_bytes(canonical_json_bytes(payload)) != embedded:
            raise LeakReviewError("frame payload does not match frame_sha256")
        roster = value["roster"]
        if not isinstance(roster, Mapping) or set(roster) != set(_ROSTER_ROLES):
            raise LeakReviewError(
                "frame roster must define collector/reviewer_a/reviewer_b/adjudicator_c"
            )
        task_scope = value["task_scope"]
        if not isinstance(task_scope, Sequence) or isinstance(task_scope, str | bytes):
            raise LeakReviewError("task_scope must be a list of strings")
        return cls(
            rubric_sha256=value["rubric_sha256"],
            unit_of_analysis=value["unit_of_analysis"],
            dedup_rule=value["dedup_rule"],
            task_scope=tuple(task_scope),
            collector=ReviewerIdentity.from_dict(roster["collector"]),
            reviewer_a=ReviewerIdentity.from_dict(roster["reviewer_a"]),
            reviewer_b=ReviewerIdentity.from_dict(roster["reviewer_b"]),
            adjudicator_c=ReviewerIdentity.from_dict(roster["adjudicator_c"]),
            sampling_policy=value["sampling_policy"],
            frozen_at=_parse_rfc3339(value["frozen_at"], "frame.frozen_at"),
        )


def validate_frame(frame: ReviewFrame | Mapping[str, object]) -> dict[str, object]:
    """Re-validate a frame (or its serialized mapping) and summarize it."""

    if isinstance(frame, Mapping):
        frame = ReviewFrame.from_dict(frame)
    if not isinstance(frame, ReviewFrame):
        raise LeakReviewError("validate_frame expects a ReviewFrame or serialized frame")
    return {
        "frame_sha256": frame.frame_sha256,
        "roster_identities": frame.roster_identities(),
        "sampling_kind": frame.sampling_policy["kind"],
        "schema_version": LEAK_REVIEW_FRAME_SCHEMA,
        "task_scope": list(frame.task_scope),
    }


def freeze_frame(path: str | Path, frame: ReviewFrame) -> str:
    """Write the frozen frame exactly once and return its self-hash."""

    if not isinstance(frame, ReviewFrame):
        raise LeakReviewError("freeze_frame expects a ReviewFrame")
    target = Path(path)
    if target.exists():
        raise LeakReviewError(f"frame artifact is write-once and already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_dict()
    try:
        with target.open("xb") as handle:
            handle.write(canonical_json_bytes(payload))
    except FileExistsError as exc:
        raise LeakReviewError(f"frame artifact is write-once and already exists: {target}") from exc
    except OSError as exc:
        raise LeakReviewError(f"unable to write frame artifact: {target}") from exc
    return str(payload["frame_sha256"])


def load_frame(path: str | Path) -> ReviewFrame:
    """Load a frozen frame, re-hashing the payload before trusting it."""

    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LeakReviewError(f"missing frame artifact: {target}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LeakReviewError(f"unable to read frame artifact: {target}") from exc
    if not isinstance(value, dict):
        raise LeakReviewError("frame artifact must contain an object")
    return ReviewFrame.from_dict(value)


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
            raise LeakReviewError("views must map string item IDs to objects")
        return indexed
    indexed: dict[str, Mapping[str, object]] = {}
    for view in _as_rows(views, "views"):
        item_id = view.get("item_id")
        if not _non_empty_str(item_id):
            raise LeakReviewError("every view needs a non-empty item_id")
        if item_id in indexed:
            raise LeakReviewError(f"duplicate view item_id: {item_id}")
        indexed[str(item_id)] = view
    return indexed


def validate_human_labels(
    labels: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    *,
    frame: ReviewFrame,
    views: Sequence[Mapping[str, object]]
    | Iterable[Mapping[str, object]]
    | Mapping[str, Mapping[str, object]],
) -> None:
    """Fail-closed validation of one reviewer's raw label rows.

    Every row must bind the frozen frame: label schema version, an item_id
    with a matching annotation view hash (stale view hash rejected), a roster
    reviewer identity, ``reviewer_type == "human"`` (fabricated agent/tool
    signatures rejected), frame/rubric hashes, a non-empty supporting span and
    rationale, and a timezone-aware ``labeled_at`` at or after the frame freeze
    (stale signatures rejected).  Label vocabularies come from
    ``evidence_qc.FIELD_LABELS`` plus ``TEMPORAL_LABELS``; B-category items
    require entity/event-class plus at least two readable event anchors, and
    C-category items require the target metric/period or a complete
    derivation.  Duplicate ``(item_id, reviewer_id)`` rows are rejected.
    """

    if not isinstance(frame, ReviewFrame):
        raise LeakReviewError("validate_human_labels expects a ReviewFrame")
    view_index = _index_views(views)
    view_hashes: dict[str, str] = {}
    for item_id, view in view_index.items():
        digest = view.get("annotation_view_sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise LeakReviewError(f"view hash is invalid for {item_id}")
        view_hashes[item_id] = digest
    reviewer_identities = {frame.reviewer_a.identity, frame.reviewer_b.identity}
    probability_sampling = frame.sampling_policy["kind"] == "probability"
    seen: set[tuple[str, str]] = set()
    for row in _as_rows(labels, "labels"):
        if row.get("schema_version") != LEAK_REVIEW_LABEL_SCHEMA:
            raise LeakReviewError(f"label schema must be {LEAK_REVIEW_LABEL_SCHEMA}")
        if row.get("frame_sha256") != frame.frame_sha256:
            raise LeakReviewError(f"label {row.get('item_id')} does not bind the frozen frame")
        if row.get("rubric_sha256") != frame.rubric_sha256:
            raise LeakReviewError(f"label {row.get('item_id')} does not bind the frozen rubric")
        item_id = row.get("item_id")
        if not _non_empty_str(item_id):
            raise LeakReviewError("every label row needs a non-empty item_id")
        item_id = str(item_id)
        if item_id not in view_index:
            raise LeakReviewError(f"label references an unknown item_id: {item_id}")
        if row.get("annotation_view_sha256") != view_hashes[item_id]:
            raise LeakReviewError(f"stale annotation_view_sha256 for {item_id}")
        reviewer_id = row.get("reviewer_id")
        if not _non_empty_str(reviewer_id) or reviewer_id not in reviewer_identities:
            raise LeakReviewError("label reviewer_id must be reviewer_a or reviewer_b")
        if row.get("reviewer_type") != "human":
            raise LeakReviewError(
                "label reviewer_type must be 'human' (agent/tool signatures are rejected)"
            )
        key = (item_id, str(reviewer_id))
        if key in seen:
            raise LeakReviewError(f"duplicate (item_id, reviewer_id) label: {key}")
        seen.add(key)
        labeled_at = _parse_rfc3339(row.get("labeled_at"), f"label {item_id} labeled_at")
        if labeled_at < frame.frozen_at:
            raise LeakReviewError(f"stale labeled_at for {item_id}: predates the frozen frame")
        for field in FIELDS:
            allowed = FIELD_LABELS.get(field)
            if allowed is None:
                raise LeakReviewError(f"unsupported leak-review field: {field}")
            if row.get(field) not in allowed:
                raise LeakReviewError(f"label {item_id} has an unsupported {field} value")
        if row.get("temporal_status") not in TEMPORAL_LABELS:
            raise LeakReviewError(f"label {item_id} has an unsupported temporal_status value")
        span = row.get("supporting_span", row.get("evidence_span", row.get("span")))
        if not _non_empty_str(span):
            raise LeakReviewError(f"label {item_id} requires a supporting evidence span")
        if not _non_empty_str(row.get("rationale")):
            raise LeakReviewError(f"label {item_id} requires a rationale")
        if probability_sampling:
            stratum = view_index[item_id].get("sampling_stratum")
            if not _non_empty_str(stratum):
                raise LeakReviewError(
                    f"view {item_id} must declare a sampling_stratum under probability sampling"
                )
            probabilities = frame.sampling_policy["inclusion_probabilities"]
            if str(stratum) not in probabilities:
                raise LeakReviewError(f"no frozen inclusion probability for stratum '{stratum}'")
        category = str(view_index[item_id].get("category", "")).strip().upper()
        if category == "B":
            if not _non_empty_str(row.get("event_entity")):
                raise LeakReviewError(f"label {item_id} (B category) requires event_entity")
            if not _non_empty_str(row.get("event_class")):
                raise LeakReviewError(f"label {item_id} (B category) requires event_class")
            anchors = row.get("event_anchors")
            if (
                not isinstance(anchors, Sequence)
                or isinstance(anchors, str | bytes)
                or len(anchors) < 2
                or any(not _non_empty_str(anchor) for anchor in anchors)
            ):
                raise LeakReviewError(
                    f"label {item_id} (B category) requires at least two readable event_anchors"
                )
        if category == "C" and not (
            _non_empty_str(row.get("target_metric_period"))
            or _non_empty_str(row.get("complete_derivation"))
        ):
            raise LeakReviewError(
                f"label {item_id} (C category) requires target_metric_period or complete_derivation"
            )


def _index_label_rows(
    rows: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    label: str,
) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    for row in _as_rows(rows, label):
        item_id = row.get("item_id")
        if not _non_empty_str(item_id):
            raise LeakReviewError(f"{label} rows need a non-empty item_id")
        if item_id in indexed:
            raise LeakReviewError(f"{label} contains duplicate item_id: {item_id}")
        indexed[str(item_id)] = row
    return indexed


def _cluster_keys(
    cluster_keys: Mapping[str, object] | Sequence[object] | None,
    item_ids: Sequence[str],
) -> list[object] | None:
    if cluster_keys is None:
        return None
    if isinstance(cluster_keys, Mapping):
        missing = [item_id for item_id in item_ids if item_id not in cluster_keys]
        if missing:
            raise LeakReviewError(f"cluster_keys are missing items: {missing}")
        return [cluster_keys[item_id] for item_id in item_ids]
    if isinstance(cluster_keys, str | bytes):
        raise LeakReviewError("cluster_keys must be a mapping or a sequence")
    keys = list(cluster_keys)
    if len(keys) != len(item_ids):
        raise LeakReviewError("cluster_keys must align with the paired items")
    return keys


def _confusion_marginals(
    values_a: Sequence[object],
    values_b: Sequence[object],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    categories = sorted({str(value) for value in (*values_a, *values_b)})
    confusion = {left: {right: 0 for right in categories} for left in categories}
    for left, right in zip(values_a, values_b, strict=True):
        confusion[str(left)][str(right)] += 1
    counts_a = Counter(str(value) for value in values_a)
    counts_b = Counter(str(value) for value in values_b)
    return confusion, {"a": dict(counts_a), "b": dict(counts_b)}


def _field_gate(
    values_a: Sequence[object],
    values_b: Sequence[object],
    *,
    item_ids: Sequence[str],
    strata_of: Mapping[str, str],
    cluster_keys: Sequence[object] | None,
) -> dict[str, object]:
    n = len(values_a)
    if n == 0:
        return {
            "agreement_count": 0,
            "agreement_rate": None,
            "agreement_wilson": None,
            "cohen_kappa": None,
            "cohen_kappa_ci": None,
            "confusion_matrix": {},
            "failure_reasons": ["empty_denominator"],
            "marginals": {"a": {}, "b": {}},
            "n": 0,
            "passed": False,
            "required_count": 0,
            "strata": {},
        }
    exact = agreement_count(values_a, values_b)
    required = math.ceil(STRONG_AGREEMENT_FLOOR * n)
    kappa = cohen_kappa(values_a, values_b)
    lower, upper = wilson_interval(count=exact, trials=n) if False else wilson_interval(exact, n)
    kappa_ci = cluster_bootstrap_kappa(values_a, values_b, cluster_keys=cluster_keys)
    confusion, marginals = _confusion_marginals(values_a, values_b)
    strata: dict[str, object] = {}
    for stratum in sorted(set(strata_of.values())):
        positions = [
            index for index, item_id in enumerate(item_ids) if strata_of[item_id] == stratum
        ]
        stratum_a = [values_a[index] for index in positions]
        stratum_b = [values_b[index] for index in positions]
        stratum_confusion, stratum_marginals = _confusion_marginals(stratum_a, stratum_b)
        strata[stratum] = {
            "agreement_count": agreement_count(stratum_a, stratum_b),
            "confusion_matrix": stratum_confusion,
            "marginals": stratum_marginals,
            "n": len(positions),
        }
    failure_reasons: list[str] = []
    if exact < required:
        failure_reasons.append("agreement_below_floor")
    if kappa is None:
        failure_reasons.append("kappa_undefined")
    elif kappa < STRONG_KAPPA_FLOOR:
        failure_reasons.append("kappa_below_floor")
    return {
        "agreement_count": exact,
        "agreement_rate": exact / n,
        "agreement_wilson": {
            "confidence": BOOTSTRAP_CONFIDENCE,
            "lower": lower,
            "upper": upper,
        },
        "cohen_kappa": kappa,
        "cohen_kappa_ci": kappa_ci,
        "confusion_matrix": confusion,
        "failure_reasons": failure_reasons,
        "marginals": marginals,
        "n": n,
        "passed": not failure_reasons,
        "required_count": required,
        "strata": strata,
    }


def agreement_gate(
    labels_a: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    labels_b: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    *,
    frame: ReviewFrame,
    cluster_keys: Mapping[str, object] | Sequence[object] | None = None,
) -> dict[str, object]:
    """Compute the strong A/B agreement gate (schema ``leak-review-gate/v1``).

    Pure function over the raw A/B rows.  Per field it reports N, the exact
    agreement count/rate, the required count ``ceil(0.90 * N)``, the unrounded
    Cohen kappa (``None`` fails with ``kappa_undefined``), a Wilson interval
    on the agreement rate, a seeded cluster-bootstrap kappa interval, and
    per-stratum confusion matrices with prevalence/marginals.  A field passes
    only when the count meets the floor AND kappa is defined and >= 0.80; the
    gate passes only when both fields pass.  An empty denominator fails with
    ``empty_denominator``.  The floors are module constants and cannot be
    lowered through any parameter.
    """

    if not isinstance(frame, ReviewFrame):
        raise LeakReviewError("agreement_gate expects a ReviewFrame")
    index_a = _index_label_rows(labels_a, "labels_a")
    index_b = _index_label_rows(labels_b, "labels_b")
    if set(index_a) != set(index_b):
        missing_a = sorted(set(index_b) - set(index_a))
        missing_b = sorted(set(index_a) - set(index_b))
        raise LeakReviewError(
            f"A/B label coverage differs (missing_a={missing_a}, missing_b={missing_b})"
        )
    item_ids = sorted(index_a)
    keys = _cluster_keys(cluster_keys, item_ids)
    strata_of: dict[str, str] = {}
    for item_id in item_ids:
        stratum_a = index_a[item_id].get("sampling_stratum")
        stratum_b = index_b[item_id].get("sampling_stratum")
        if stratum_a is not None and stratum_b is not None and stratum_a != stratum_b:
            raise LeakReviewError(f"A/B sampling_stratum mismatch for {item_id}")
        stratum = stratum_a if stratum_a is not None else stratum_b
        strata_of[item_id] = str(stratum) if stratum is not None else "unspecified"
    fields: dict[str, object] = {}
    for field in FIELDS:
        allowed = FIELD_LABELS.get(field)
        if allowed is None:
            raise LeakReviewError(f"unsupported leak-review field: {field}")
        values_a = [index_a[item_id].get(field) for item_id in item_ids]
        values_b = [index_b[item_id].get(field) for item_id in item_ids]
        if any(value not in allowed for value in (*values_a, *values_b)):
            raise LeakReviewError(f"labels contain an unsupported {field} value")
        fields[field] = _field_gate(
            values_a,
            values_b,
            item_ids=item_ids,
            strata_of=strata_of,
            cluster_keys=keys,
        )
    return {
        "fields": fields,
        "floors": {"agreement": STRONG_AGREEMENT_FLOOR, "kappa": STRONG_KAPPA_FLOOR},
        "frame_sha256": frame.frame_sha256,
        "item_count": len(item_ids),
        "passed": all(report["passed"] for report in fields.values()),
        "schema_version": LEAK_REVIEW_GATE_SCHEMA,
    }


def blind_c_queue(
    views: Sequence[Mapping[str, object]]
    | Iterable[Mapping[str, object]]
    | Mapping[str, Mapping[str, object]],
    labels_a: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    labels_b: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build the blind adjudicator queue; delegates to evidence_qc."""

    return build_disagreement_queue(views, labels_a, labels_b)


def adjudicate(
    views: Sequence[Mapping[str, object]]
    | Iterable[Mapping[str, object]]
    | Mapping[str, Mapping[str, object]],
    labels_a: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    labels_b: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    labels_c: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    queue: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Resolve queued A/B disagreements with the blind adjudicator C.

    Delegates to ``evidence_qc.resolve_three_way``.  Adjudication resolves
    labels for downstream routing only; it cannot raise the pre-adjudication
    kappa, because ``agreement_gate`` always recomputes from the raw A/B rows
    (asserted in ``tests/test_leak_review.py``).
    """

    return resolve_three_way(views, labels_a, labels_b, labels_c, queue)


def _validate_resolved_labels(item_id: str, labels: Mapping[str, object]) -> None:
    unknown = sorted(set(labels) - _RESOLVED_LABEL_KEYS)
    if unknown:
        raise LeakReviewError(f"resolution {item_id} carries unknown resolved labels: {unknown}")
    for field in FIELDS:
        value = labels.get(field)
        if value is not None and value not in FIELD_LABELS[field]:
            raise LeakReviewError(f"resolution {item_id} has an unsupported {field} value")
    temporal = labels.get("temporal_status")
    if temporal is not None and temporal not in TEMPORAL_LABELS:
        raise LeakReviewError(f"resolution {item_id} has an unsupported temporal_status value")


def _construct_state(labels: Mapping[str, object], target_revealing: str) -> str:
    conjuncts = (
        (labels.get("temporal_status"), "post_cutoff"),
        (labels.get("task_relevance"), "relevant"),
        (labels.get("outcome_revealing"), target_revealing),
    )
    if all(value == target for value, target in conjuncts):
        return "true"
    for value, target in conjuncts:
        if value is not None and value != "indeterminate" and value != target:
            return "false"
    return "unknown"


def _classify_receipt(
    model_visible: bool,
    resolution: Mapping[str, object] | None,
) -> tuple[str, str]:
    if not model_visible:
        return "not_leaked", "not_model_visible"
    assert resolution is not None  # guaranteed by the caller for visible receipts
    status = str(resolution["resolution_status"])
    if status not in _RESOLVED_STATUSES:
        return "unresolved", f"resolution_{status}"
    labels = resolution["resolved_labels"]
    assert isinstance(labels, Mapping)
    direct = _construct_state(labels, "direct")
    indirect = _construct_state(labels, "indirect")
    if direct == "true":
        return "leaked_direct", "direct_answer_leak"
    if indirect == "true":
        return "leaked_indirect", "indirect_exposure"
    if direct == "false" and indirect == "false":
        if labels.get("task_relevance") == "not_relevant":
            return "not_leaked", "task_relevance_not_relevant"
        if labels.get("temporal_status") == "pre_cutoff":
            return "not_leaked", "temporal_pre_cutoff"
        return "not_leaked", "outcome_not_revealing"
    for construct in ("temporal_status", "task_relevance", "outcome_revealing"):
        value = labels.get(construct)
        if value is None or value == "indeterminate":
            return "unresolved", f"{construct}_unresolved"
    return "unresolved", "unresolved"


def join_sealed_labels(
    receipts: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    resolutions: Sequence[Mapping[str, object]] | Iterable[Mapping[str, object]],
    *,
    frame: ReviewFrame,
) -> dict[str, object]:
    """Join model-visible receipts to sealed labels (schema ``leak-review-join/v1``).

    The join runs only over sealed labels: every resolution row must carry a
    ``resolution_status`` produced by ``adjudicate`` (or ``resolved_initial``
    for unanimous A/B agreement).  A direct answer-leak event requires
    ``temporal_status == "post_cutoff"`` AND ``task_relevance == "relevant"``
    AND ``outcome_revealing == "direct"`` on a receipt with
    ``model_visible is True``; ``indirect`` revealing is counted separately.
    Unresolved or indeterminate items stay unresolved and enter only the
    worst-case bound.  The sensitivity sets are fixed: ``direct_only`` and
    ``direct_or_indirect``.  Every receipt is accounted for exactly once
    across leaked_direct/leaked_indirect/not_leaked/unresolved; any
    conservation violation raises ``LeakReviewError``.
    """

    if not isinstance(frame, ReviewFrame):
        raise LeakReviewError("join_sealed_labels expects a ReviewFrame")
    resolution_index: dict[str, Mapping[str, object]] = {}
    for row in _as_rows(resolutions, "resolutions"):
        item_id = row.get("item_id")
        if not _non_empty_str(item_id):
            raise LeakReviewError("every resolution row needs a non-empty item_id")
        item_id = str(item_id)
        if item_id in resolution_index:
            raise LeakReviewError(f"duplicate resolution item_id: {item_id}")
        status = row.get("resolution_status")
        if status not in RESOLUTION_STATUSES:
            raise LeakReviewError(
                f"resolution {item_id} is not sealed (unknown resolution_status: {status})"
            )
        labels = row.get("resolved_labels", {})
        if not isinstance(labels, Mapping):
            raise LeakReviewError(f"resolution {item_id} resolved_labels must be an object")
        if status in _RESOLVED_STATUSES and not labels:
            raise LeakReviewError(f"resolution {item_id} is sealed but carries no labels")
        _validate_resolved_labels(item_id, labels)
        resolution_index[item_id] = row

    buckets: dict[str, list[str]] = {
        "leaked_direct": [],
        "leaked_indirect": [],
        "not_leaked": [],
        "unresolved": [],
    }
    items: list[dict[str, object]] = []
    seen_receipts: set[str] = set()
    for receipt in _as_rows(receipts, "receipts"):
        receipt_id = receipt.get("receipt_id", receipt.get("item_id"))
        if not _non_empty_str(receipt_id):
            raise LeakReviewError("every receipt needs a non-empty receipt_id or item_id")
        receipt_id = str(receipt_id)
        if receipt_id in seen_receipts:
            raise LeakReviewError(f"duplicate receipt: {receipt_id}")
        seen_receipts.add(receipt_id)
        item_id = receipt.get("item_id")
        if not _non_empty_str(item_id):
            raise LeakReviewError(f"receipt {receipt_id} needs a non-empty item_id")
        item_id = str(item_id)
        model_visible = receipt.get("model_visible")
        if not isinstance(model_visible, bool):
            raise LeakReviewError(f"receipt {receipt_id} model_visible must be a bool")
        resolution = resolution_index.get(item_id)
        if model_visible and resolution is None:
            raise LeakReviewError(
                f"no sealed resolution for model-visible receipt {receipt_id} (item {item_id})"
            )
        bucket, reason = _classify_receipt(model_visible, resolution)
        buckets[bucket].append(receipt_id)
        items.append(
            {
                "bucket": bucket,
                "item_id": item_id,
                "model_visible": model_visible,
                "reason": reason,
                "receipt_id": receipt_id,
            }
        )
    accounted = sum(len(ids) for ids in buckets.values())
    if accounted != len(seen_receipts) or any(
        set(buckets[left]) & set(buckets[right])
        for left in buckets
        for right in buckets
        if left != right
    ):
        raise LeakReviewError("denominator conservation violated: receipts are not partitioned")
    leaked_direct = sorted(buckets["leaked_direct"])
    leaked_indirect = sorted(buckets["leaked_indirect"])
    unresolved_count = len(buckets["unresolved"])
    sensitivity = {
        "direct_only": {
            "bounds": {
                "best_case": len(leaked_direct),
                "worst_case": len(leaked_direct) + unresolved_count,
            },
            "count": len(leaked_direct),
            "receipt_ids": leaked_direct,
        },
        "direct_or_indirect": {
            "bounds": {
                "best_case": len(leaked_direct) + len(leaked_indirect),
                "worst_case": len(leaked_direct) + len(leaked_indirect) + unresolved_count,
            },
            "count": len(leaked_direct) + len(leaked_indirect),
            "receipt_ids": sorted(leaked_direct + leaked_indirect),
        },
    }
    return {
        "buckets": {name: sorted(ids) for name, ids in buckets.items()},
        "conserved": True,
        "counts": {
            "leaked_direct": len(leaked_direct),
            "leaked_indirect": len(leaked_indirect),
            "not_leaked": len(buckets["not_leaked"]),
            "receipts": len(seen_receipts),
            "unresolved": unresolved_count,
        },
        "denominator": len(seen_receipts),
        "frame_sha256": frame.frame_sha256,
        "items": items,
        "schema_version": LEAK_REVIEW_JOIN_SCHEMA,
        "sensitivity": sensitivity,
    }


__all__ = [
    "FIELDS",
    "LEAK_REVIEW_FRAME_SCHEMA",
    "LEAK_REVIEW_GATE_SCHEMA",
    "LEAK_REVIEW_JOIN_SCHEMA",
    "LEAK_REVIEW_LABEL_SCHEMA",
    "LeakReviewError",
    "ReviewFrame",
    "ReviewerIdentity",
    "STRONG_AGREEMENT_FLOOR",
    "STRONG_KAPPA_FLOOR",
    "TEMPORAL_LABELS",
    "adjudicate",
    "agreement_gate",
    "blind_c_queue",
    "freeze_frame",
    "join_sealed_labels",
    "load_frame",
    "validate_frame",
    "validate_human_labels",
]
