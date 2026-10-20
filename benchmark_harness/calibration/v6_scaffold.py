"""Offline v6/rubric-v5 calibration scaffold."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from benchmark_harness.calibration.validator import (
    EXPECTED_TRACKED_FILES,
    HUB_REPO_ID,
    HUB_REVISION,
    V5_RELEVANCE_CALIBRATION_ID,
    V6_JSON_FILES,
    V6_JSONL_FILES,
    V6_PILOT_JSONL_FILES,
    V6_RELEVANCE_AUTHORIZATION_ID,
    V6_RELEVANCE_CALIBRATION_ID,
    V6_RELEVANCE_RUBRIC_VERSION,
    V6_RELEVANCE_SCOPE,
    V6_RELEVANCE_SOURCE_COMMENT_ID,
    V6_TRACKED_FILES,
    _v6_source_exclusion_digests,
    canonical_json_bytes,
    validate_calibration_root,
)

DEFAULT_V6_ROOT = Path("data/calibration/leakage-probe/v6")
V6_ISSUED_AT_UTC = "2026-08-31T14:55:28Z"
V6_AUTHORIZATION_SCHEMA = "leakage-calibration-authorization/v1"
_V6_RATIONALE_GRAMMAR = (
    "RCLASS=<enum>; task_scope_sha256=<64hex>; truncation=<none|proven>; "
    "hidden_axis=<none|entity|event|period|metric>; "
    "entity_match=<match|mismatch>; "
    "event_match=<exact|contains_exact|cross_period|not_proven|na>; "
    "metric_match=<exact|component|other|none|na>; "
    "OCLASS=<DIRECT_TARGET|DIRECT_POST_EVENT_MOVE|INDIRECT_INPUT|NO_SIGNAL|TRUNCATED_SCOPE>; "
    "span=<short visible evidence>"
)

_V6_RUBRIC = f"""# Retrieval leakage rubric v5

## Status and scope

This bundle is a smoke/exploratory calibration pilot. It is not a paper
experiment, has no binary gold labels, and is never ranking evidence.
`ranking_eligible=false`, `claim_eligible=false`, and
`live_unrestricted_authorized=false` remain mandatory. Only the bounded pilot
collection is authorized; an unrestricted paper probe is not authorized.

## Preflight and blinding

Before any retrieval call, build and freeze the SHA-256 of the source-owned
task scope. The scope contains the family, entity name/ticker, requested
output shape, A1 target type, B event class/date/reporting package or unique
event fingerprint, C target metric and input/target periods, source spans, and
the normalizer version. Missing B period and missing unique event identity is a
preflight failure and must be resampled, never labeled indeterminate. C scope
must parse `以下是 {{company}} 近两个季度` and the target metric inside `「...」`.

Labelers A and B see the same task scope and the model-visible title, snippet,
and content. URL, publication/update clocks, temporal and cutoff-filter
verdicts, system identity, answers, scores, ground truth, and all other
labeler records are hidden. Truncation metadata is objective metadata only.

## Decision order

1. Identify the entity. An explicit different entity is `not_relevant/no`.
2. Establish family-specific applicability. A weak fact, link, missing field,
   contradiction, fetch failure, or annotator uncertainty is not
   `indeterminate`.
3. Classify outcome only after applicability. Use `indeterminate/indeterminate`
   only when visible truncation is proven, its hidden axis is entity, event,
   period, or metric, and no decisive visible span remains. If a decisive span
   is visible, make a determinate decision.

## Family rules

### A1 valuation

An explicit target price/fair value/range is `relevant/direct`. Same-security
financial, market, valuation, or issuer-specific capital inputs are
`relevant/indirect`. Other pages are `not_relevant/no`.

### B direction

Same issuer is not the same event. Use the task-owned event class and reporting
package/date or unique fingerprint. Generic earnings/revenue/profit is never a
same-event fallback. Cross-period or unproven event material is
`not_relevant/no`. Exact target-security post-event price/return movement is
`relevant/direct`. Exact-event analyst/rating/target/recommendation material
with a security bridge is `relevant/indirect`. Exact event facts without a
price bridge are `relevant/no`. Generic guidance, holdings, sentiment,
investor expectations, or operating forecasts without that bridge are not a
qualifying outcome signal.

### C metric forecast

The exact company, target metric, and target-period realized value are
`relevant/direct`. Another period, guidance, consensus, component, or driver is
`relevant/indirect`. A target metric named without a value is `relevant/no`.
An unrelated metric is `not_relevant/no`. Do not turn a task-input value into a
direct result by subtraction.

## Rationale grammar

Every raw label uses exactly:

`{_V6_RATIONALE_GRAMMAR}`

The pilot samples exactly 10 items in each approved stratum, disjoint from the
v3/v4/v5 pilots and the frozen 400 at item, receipt, query, URL/payload,
annotation-view, and semantic task/query-class evidence levels. Search uses the
existing sealed same-candidate cutoff filter. Raw payloads remain in the
ignored cache. No labels or gold are copied from prior calibration leaves.
"""

_V6_README = """# Leakage calibration bundle v6

This is the rubric-v5 exploratory calibration pilot. Its top-level 400-item
inputs are an immutable copy of the v5 failed-gate audit. The active pilot is a
new, disjoint 40-item collection with ten items in each approved stratum.

The pilot has Labeler-A relevance/outcome labels only; independent B labels and
the agreement gate are intentionally a later review step. It creates no binary
gold and cannot enter a ranking table or paper claim. `live_unrestricted_authorized=false`
and `paper_probe_authorized=false` remain in force.
"""


class V6ScaffoldError(ValueError):
    """Raised when the immutable v5 lineage cannot seed the v6 scaffold."""


def prepare_v6_scaffold(
    root: str | Path = DEFAULT_V6_ROOT,
    *,
    source_root: str | Path | None = None,
) -> Path:
    """Create the v6 scaffold without touching any prior calibration leaf."""

    root_path = Path(root).expanduser().resolve()
    if root_path.exists():
        manifest_path = root_path / "artifact_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise V6ScaffoldError("existing v6 root has no readable manifest") from exc
        if manifest.get("calibration_id") != V6_RELEVANCE_CALIBRATION_ID:
            raise V6ScaffoldError("existing root is not the v6 calibration bundle")
        # A preflight-only scaffold may have been created before the evidence
        # union was corrected to operate at the item/query-class level.  It is
        # safe to refresh that proof while the active pilot is still empty;
        # once collection starts, the root becomes an append-only lineage.
        if not any(
            (root_path / relative).read_bytes().strip()
            for relative in V6_PILOT_JSONL_FILES
            if (root_path / relative).is_file()
        ):
            excluded = _v6_source_exclusion_digests(root_path)
            _write_json(
                root_path / "pilot/exclusion-proof.json",
                {
                    "calibration_id": V6_RELEVANCE_CALIBRATION_ID,
                    "excluded_digests": excluded,
                    "schema_version": "leakage-calibration-exclusion-proof/v1",
                    "selected_digests": {field: [] for field in excluded},
                    "source_calibrations": [
                        "leakage-probe/v1",
                        "leakage-probe/v2",
                        "leakage-probe/v3",
                        "leakage-probe/v4",
                        "leakage-probe/v5",
                    ],
                    "status": "preflight",
                },
            )
            _write_v6_manifest(
                root_path,
                evidence_sha256=str(manifest.get("restricted_evidence_manifest_sha256", ""))
                or None,
            )
        return root_path

    source = (
        Path(source_root).expanduser().resolve()
        if source_root is not None
        else root_path.parent / "v5"
    )
    if not source.is_dir():
        raise V6ScaffoldError("immutable v5 audit root is unavailable")
    try:
        source_state = validate_calibration_root(source)
    except ValueError as exc:
        raise V6ScaffoldError("v5 source failed validation") from exc
    if source_state.get("calibration_id") != V5_RELEVANCE_CALIBRATION_ID:
        raise V6ScaffoldError("v6 scaffold source must be the v5 audit root")

    root_path.mkdir(parents=True, exist_ok=False)
    for relative in EXPECTED_TRACKED_FILES:
        target = root_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative in {"README.md", "rubric.md", "authorization.json", "cohort.json"}:
            continue
        shutil.copyfile(source / relative, target)
    for relative in V6_PILOT_JSONL_FILES:
        target = root_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    (root_path / "pilot/exclusion-proof.json").parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        root_path / "pilot/exclusion-proof.json",
        {
            "calibration_id": V6_RELEVANCE_CALIBRATION_ID,
            "excluded_digests": _v6_source_exclusion_digests(root_path),
            "schema_version": "leakage-calibration-exclusion-proof/v1",
            "selected_digests": {field: [] for field in _v6_source_exclusion_digests(root_path)},
            "source_calibrations": [
                "leakage-probe/v1",
                "leakage-probe/v2",
                "leakage-probe/v3",
                "leakage-probe/v4",
                "leakage-probe/v5",
            ],
            "status": "preflight",
        },
    )
    _write_v6_schemas(root_path, source)
    _write_v6_authorization(root_path)
    _write_v6_cohort(root_path, source)
    _write_v6_report(root_path, source)
    (root_path / "README.md").write_text(_V6_README, encoding="utf-8")
    (root_path / "rubric.md").write_text(_V6_RUBRIC, encoding="utf-8")
    _write_v6_manifest(root_path)
    return root_path


def scaffold_v6(root: str | Path = DEFAULT_V6_ROOT) -> dict[str, object]:
    root_path = prepare_v6_scaffold(root)
    result = validate_calibration_root(root_path)
    report = _load_json(root_path / "reports/calibration-report.json")
    result["pilot_status"] = report.get("pilot_status")
    result["pilot_n"] = report.get("pilot_n")
    return result


def _write_v6_schemas(root: Path, source: Path) -> None:
    auth = _load_json(source / "schemas/authorization.schema.json")
    auth["title"] = "Explicit v6 rubric-v5 pilot authorization"
    auth["properties"]["calibration_id"] = {"const": V6_RELEVANCE_CALIBRATION_ID}
    auth["properties"]["scope"] = {"const": V6_RELEVANCE_SCOPE}
    auth["properties"]["calibration_unrestricted_collection_authorized"] = {"const": True}
    _write_json(root / "schemas/authorization.schema.json", auth)

    raw = _load_json(source / "schemas/raw-label.schema.json")
    raw["title"] = "V6 rubric-v5 blind raw relevance/outcome label"
    raw["properties"]["rubric_version"] = {"const": V6_RELEVANCE_RUBRIC_VERSION}
    for field in (
        "entity_match",
        "event_match",
        "hidden_axis",
        "metric_match",
        "oclass",
        "rclass",
        "task_scope_sha256",
        "truncation",
    ):
        raw["properties"][field] = {"type": "string"}
    _write_json(root / "schemas/raw-label.schema.json", raw)

    item = _load_json(source / "schemas/item.schema.json")
    item["properties"]["task_scope_sha256"] = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    _write_json(root / "schemas/item.schema.json", item)

    report = _load_json(source / "schemas/report.schema.json")
    report["title"] = "V6 rubric-v5 exploratory calibration report"
    report["properties"]["pilot_n"] = {"type": "integer", "minimum": 0, "maximum": 40}
    report["properties"]["pilot_status"] = {"enum": ["not_run", "pilot"]}
    _write_json(root / "schemas/report.schema.json", report)


def _write_v6_authorization(root: Path) -> None:
    _write_json(
        root / "authorization.json",
        {
            "authorization_id": V6_RELEVANCE_AUTHORIZATION_ID,
            "calibration_id": V6_RELEVANCE_CALIBRATION_ID,
            "calibration_unrestricted_collection_authorized": True,
            "frozen_item_count": 400,
            "frozen_strata": {
                "A1-T1": {"ambiguous": 25, "keep": 38, "reject": 38},
                "B-T1": {"ambiguous": 25, "keep": 38, "reject": 38},
                "B-T2": {"ambiguous": 25, "keep": 37, "reject": 37},
                "C-T2": {"ambiguous": 25, "keep": 37, "reject": 37},
            },
            "full_paper_unrestricted_authorized": False,
            "issue_id": "5e7ed4b6-3e65-4cfa-9da8-0ea82bc31fc4",
            "issued_at_utc": V6_ISSUED_AT_UTC,
            "pilot_item_count": 40,
            "pilot_strata": {stratum: 10 for stratum in ("A1-T1", "B-T1", "B-T2", "C-T2")},
            "same_candidates_cutoff_filter": True,
            "schema_version": V6_AUTHORIZATION_SCHEMA,
            "scope": V6_RELEVANCE_SCOPE,
            "source_comment_id": V6_RELEVANCE_SOURCE_COMMENT_ID,
            "status": "active",
        },
    )


def _write_v6_cohort(root: Path, source: Path) -> None:
    cohort = _load_json(source / "cohort.json")
    cohort.update(
        {
            "calibration_status": "incomplete",
            "cohort_id": V6_RELEVANCE_CALIBRATION_ID,
            "live_unrestricted_authorized": False,
        }
    )
    _write_json(root / "cohort.json", cohort)


def _write_v6_report(root: Path, source: Path) -> None:
    report = _load_json(source / "reports/calibration-report.json")
    report.update(
        {
            "calibration_status": "incomplete",
            "claim_eligible": False,
            "cohen_kappa": None,
            "counts": {
                "adjudicated": 0,
                "determinate_binary_gold": 0,
                "items": 400,
                "leakage_negative": 0,
                "leakage_positive": 0,
                "raw_labels": 0,
                "raw_labels_a": 0,
                "raw_labels_b": 0,
                "unknown": 400,
                "unresolved_disputes": 0,
            },
            "exclusions_by_type": {"v6_active_pilot_not_run": 40},
            "filter_validated": False,
            "generated_at_utc": V6_ISSUED_AT_UTC,
            "live_unrestricted_authorized": False,
            "pilot_n": 0,
            "pilot_status": "not_run",
            "quality_gates_passed": False,
            "raw_agreement": None,
        }
    )
    _write_json(root / "reports/calibration-report.json", report)


def _write_v6_manifest(root: Path, *, evidence_sha256: str | None = None) -> None:
    manifest: dict[str, object] = {
        "adjudication": {
            "adjudicated_observed": 0,
            "raw_labels_expected": 800,
            "raw_labels_observed": 0,
            "status": "incomplete",
            "unresolved_disputes": 0,
        },
        "calibration_id": V6_RELEVANCE_CALIBRATION_ID,
        "calibration_status": "incomplete",
        "clocks": {
            "backbone_id": "openai:gpt-4.1",
            "backbone_training_cutoff": "2024-06-30",
            "experiment_as_of": "2026-08-17",
            "guard_days": 30,
            "task_cutoff_semantics": "utc_calendar_day_inclusive_next_day_exclusive",
        },
        "files": [],
        "hub": {"repo_id": HUB_REPO_ID, "revision": HUB_REVISION},
        "live_unrestricted_authorized": False,
        "query_corpus": {"path": "query-corpus.jsonl", "sha256": ""},
        "restricted_evidence_manifest_sha256": evidence_sha256
        or hashlib.sha256(b"leakage-calibration/v6/not-collected").hexdigest(),
        "rubric": {"path": "rubric.md", "sha256": _sha256_file(root / "rubric.md")},
        "sampling_plan": {
            "path": "sampling-plan.json",
            "sha256": _sha256_file(root / "sampling-plan.json"),
        },
        "schema_version": "leakage-calibration-manifest/v1",
    }
    manifest["query_corpus"]["sha256"] = _sha256_file(root / "query-corpus.jsonl")  # type: ignore[index]
    entries: list[dict[str, object]] = []
    for relative in V6_TRACKED_FILES:
        payload = (root / relative).read_bytes()
        entries.append(
            {
                "bytes": len(payload),
                "path": relative,
                "row_count": (
                    len(payload.decode("utf-8").splitlines())
                    if relative in V6_JSONL_FILES
                    else 1
                    if relative in V6_JSON_FILES
                    else None
                ),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest["files"] = entries
    manifest_bytes = canonical_json_bytes(manifest)
    (root / "artifact_manifest.json").write_bytes(manifest_bytes)
    (root / "CALIBRATION.sha256").write_text(
        hashlib.sha256(manifest_bytes).hexdigest() + "  artifact_manifest.json\n", encoding="utf-8"
    )


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V6ScaffoldError(f"JSON source {path} is not an object")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["DEFAULT_V6_ROOT", "V6ScaffoldError", "prepare_v6_scaffold", "scaffold_v6"]
