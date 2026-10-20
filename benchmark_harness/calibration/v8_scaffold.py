# ruff: noqa: E501
"""Offline scaffold for the independent rubric-v6/task-scope-v2 pilot."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

from benchmark_harness.calibration.relevance_v8 import classify_v8
from benchmark_harness.calibration.v8_scope import V8_SCOPE_NORMALIZER
from benchmark_harness.calibration.v8_validator import (
    EXPECTED_TRACKED_FILES,
    HUB_REPO_ID,
    HUB_REVISION,
    V6_RELEVANCE_CALIBRATION_ID,
    V8_JSON_FILES,
    V8_PILOT_JSONL_FILES,
    V8_RELEVANCE_AUTHORIZATION_ID,
    V8_RELEVANCE_CALIBRATION_ID,
    V8_RELEVANCE_RUBRIC_VERSION,
    V8_RELEVANCE_SCOPE,
    V8_RELEVANCE_SOURCE_COMMENT_ID,
    V8_TRACKED_FILES,
    _v8_source_exclusion_digests,
    canonical_json_bytes,
    validate_calibration_root,
)
from benchmark_harness.contracts import sha256_json

DEFAULT_V8_ROOT = Path("data/calibration/leakage-probe/v8")
V8_ISSUED_AT_UTC = "2026-08-31T16:25:09Z"
V8_AUTHORIZATION_SCHEMA = "leakage-calibration-authorization/v1"


class V8ScaffoldError(ValueError):
    """Raised when the immutable v6 lineage cannot seed v8."""


def prepare_v8_scaffold(
    root: str | Path = DEFAULT_V8_ROOT, *, source_root: str | Path | None = None
) -> Path:
    """Create v8 from v6 without changing any previous calibration leaf."""

    root_path = Path(root).expanduser().resolve()
    source = (
        Path(source_root).expanduser().resolve()
        if source_root is not None
        else root_path.parent / "v6"
    )
    if root_path.exists():
        _assert_identity(root_path)
        if not _pilot_has_rows(root_path):
            _write_exclusion_proof(root_path, status="preflight")
            _write_manifest(root_path)
        return root_path
    if not source.is_dir():
        raise V8ScaffoldError("immutable v6 audit root is unavailable")
    try:
        source_state = validate_calibration_root(source)
    except ValueError as exc:
        raise V8ScaffoldError("v6 source failed validation") from exc
    if source_state.get("calibration_id") != V6_RELEVANCE_CALIBRATION_ID:
        raise V8ScaffoldError("v8 source must be the v6 failed-gate audit")
    root_path.mkdir(parents=True, exist_ok=False)
    for relative in EXPECTED_TRACKED_FILES:
        target = root_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative in {"README.md", "rubric.md", "authorization.json", "cohort.json"}:
            continue
        shutil.copyfile(source / relative, target)
    for relative in V8_PILOT_JSONL_FILES:
        target = root_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    _write_exclusion_proof(root_path, status="preflight")
    _write_schemas(root_path, source)
    _write_authorization(root_path)
    _write_cohort(root_path, source)
    _write_report(root_path, source)
    (root_path / "README.md").write_text(_README, encoding="utf-8")
    (root_path / "rubric.md").write_text(_rubric_text(), encoding="utf-8")
    _write_json(root_path / "formula-table.json", _formula_table())
    _write_json(root_path / "schemas/conformance-fixtures.json", _conformance_fixtures())
    _write_manifest(root_path)
    return root_path


def scaffold_v8(root: str | Path = DEFAULT_V8_ROOT) -> dict[str, object]:
    root_path = prepare_v8_scaffold(root)
    result = validate_calibration_root(root_path)
    report = _load_json(root_path / "reports/calibration-report.json")
    result["pilot_status"] = report.get("pilot_status")
    result["pilot_n"] = report.get("pilot_n")
    return result


def _assert_identity(root: Path) -> None:
    try:
        manifest = _load_json(root / "artifact_manifest.json")
    except (OSError, json.JSONDecodeError) as exc:
        raise V8ScaffoldError("existing v8 root has no readable manifest") from exc
    if manifest.get("calibration_id") != V8_RELEVANCE_CALIBRATION_ID:
        raise V8ScaffoldError("existing root is not the v8 calibration bundle")


def _pilot_has_rows(root: Path) -> bool:
    return any(
        (root / relative).read_bytes().strip()
        for relative in V8_PILOT_JSONL_FILES
        if (root / relative).is_file()
    )


def _write_exclusion_proof(root: Path, *, status: str) -> None:
    excluded = _v8_source_exclusion_digests(root)
    _write_json(
        root / "pilot/exclusion-proof.json",
        {
            "calibration_id": V8_RELEVANCE_CALIBRATION_ID,
            "excluded_digests": excluded,
            "schema_version": "leakage-calibration-exclusion-proof/v1",
            "selected_digests": {field: [] for field in excluded},
            "source_calibrations": [
                "leakage-probe/v1",
                "leakage-probe/v2",
                "leakage-probe/v3",
                "leakage-probe/v4",
                "leakage-probe/v5",
                "leakage-probe/v6",
                "leakage-probe/v7",
            ],
            "status": status,
        },
    )


def _write_schemas(root: Path, source: Path) -> None:
    auth = _load_json(source / "schemas/authorization.schema.json")
    auth["title"] = "Explicit v8 rubric-v6 pilot authorization"
    auth["properties"]["calibration_id"] = {"const": V8_RELEVANCE_CALIBRATION_ID}
    auth["properties"]["scope"] = {"const": V8_RELEVANCE_SCOPE}
    auth["properties"]["calibration_unrestricted_collection_authorized"] = {"const": True}
    _write_json(root / "schemas/authorization.schema.json", auth)

    raw = _load_json(source / "schemas/raw-label.schema.json")
    raw["title"] = "V8 rubric-v6 structured blind raw label"
    raw["properties"]["rubric_version"] = {"const": V8_RELEVANCE_RUBRIC_VERSION}
    for field in (
        "decisive_units",
        "event_anchor_matches",
        "formula_id",
        "period_relation",
        "scope_relation",
        "signal_kind",
    ):
        raw["properties"][field] = {"type": ["array", "string", "null"]}
    _write_json(root / "schemas/raw-label.schema.json", raw)

    item = _load_json(source / "schemas/item.schema.json")
    item["properties"]["task_scope_sha256"] = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    _write_json(root / "schemas/item.schema.json", item)

    report = _load_json(source / "schemas/report.schema.json")
    report["title"] = "V8 rubric-v6 exploratory calibration report"
    report["properties"]["pilot_n"] = {"type": "integer", "minimum": 0, "maximum": 40}
    report["properties"]["pilot_status"] = {"enum": ["not_run", "pilot"]}
    _write_json(root / "schemas/report.schema.json", report)


def _write_authorization(root: Path) -> None:
    _write_json(
        root / "authorization.json",
        {
            "authorization_id": V8_RELEVANCE_AUTHORIZATION_ID,
            "calibration_id": V8_RELEVANCE_CALIBRATION_ID,
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
            "issued_at_utc": V8_ISSUED_AT_UTC,
            "pilot_item_count": 40,
            "pilot_strata": {stratum: 10 for stratum in ("A1-T1", "B-T1", "B-T2", "C-T2")},
            "same_candidates_cutoff_filter": True,
            "schema_version": V8_AUTHORIZATION_SCHEMA,
            "scope": V8_RELEVANCE_SCOPE,
            "source_comment_id": V8_RELEVANCE_SOURCE_COMMENT_ID,
            "status": "active",
        },
    )


def _write_cohort(root: Path, source: Path) -> None:
    cohort = _load_json(source / "cohort.json")
    cohort.update(
        {
            "calibration_status": "incomplete",
            "cohort_id": V8_RELEVANCE_CALIBRATION_ID,
            "live_unrestricted_authorized": False,
        }
    )
    _write_json(root / "cohort.json", cohort)


def _write_report(root: Path, source: Path) -> None:
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
            "exclusions_by_type": {"v8_active_pilot_not_run": 40},
            "filter_validated": False,
            "generated_at_utc": V8_ISSUED_AT_UTC,
            "live_unrestricted_authorized": False,
            "pilot_n": 0,
            "pilot_status": "not_run",
            "quality_gates_passed": False,
            "raw_agreement": None,
        }
    )
    _write_json(root / "reports/calibration-report.json", report)


def _rubric_text() -> str:
    scope_path = Path(__file__).with_name("v8_scope.py")
    classifier_path = Path(__file__).with_name("relevance_v8.py")
    return f"""# Retrieval leakage rubric v6

status=smoke; exploratory=true; ranking_eligible=false; claim_eligible=false
calibration_id={V8_RELEVANCE_CALIBRATION_ID}
scope_normalizer={V8_SCOPE_NORMALIZER}
scope_module_sha256={_sha256_file(scope_path)}
classifier_module_sha256={_sha256_file(classifier_path)}
formula_table=net_margin:attributable_net_profit/operating_revenue

This bundle is a bounded, exploratory 40-item calibration pilot. It is not a
paper experiment, ranking table, unrestricted probe, binary-gold corpus, or
claim. v3 through v6 and the frozen 400-item audit are immutable inputs.

Before any provider call, freeze this rubric, the schemas, task-scope/v2,
formula-table.json, and the synthetic conformance fixtures. A changed rule
byte fails closed. The only authorized collection is 10 items each from
A1-T1, B-T1, B-T2, and C-T2, selected by evidence-level exclusion from v1-v6
and the frozen 400. The same-candidate cutoff filter remains mandatory.

Decision order is preflight, strict truncation, entity, family applicability,
direct, indirect, relevant/no, not_relevant/no. Local sentence, bullet, and
table-row units are the only evidence; each label cites one unit, with only
an adjacent identity/fact pair (or one same-block formula pair) permitted.
The shared derive_label_pair function is authoritative for the stored pair.

B requires the same entity, event class, and at least two readable anchors from
release date, reporting period, and task-owned fingerprint facts. A date or an
opaque hash alone never proves an event. C other-period target metrics are
relevant/indirect; unrelated metrics require the frozen complete formula or an
explicit driver. Accounting fair value is not an equity target endpoint.

Labeler A is the only pilot labeler in this leaf. Temporal labels, filter
verdicts, answers, gold, scores, and ranking eligibility remain unavailable
or false. The quality gates remain agreement >= 36/40 and Cohen kappa >= 0.80
for each field; this pilot does not claim that gate has passed.
"""


_README = """# Leakage calibration bundle v8

This is the rubric-v6/task-scope-v2 exploratory calibration pilot. The
top-level 400-item inputs are an immutable audit copy of v6. The active leaf
contains one fresh, evidence-disjoint 40-item pilot with ten items in each
approved stratum.

Only Labeler A is present in this leaf. No binary gold, ranking evidence,
unrestricted paper probe, or paper claim is produced.
"""


def _formula_table() -> dict[str, object]:
    return {
        "formulas": [
            {
                "components": ["attributable_net_profit", "operating_revenue"],
                "expression": "attributable_net_profit / operating_revenue",
                "formula_id": "net_margin",
                "target_metric": "net_margin",
            }
        ],
        "schema_version": "leakage-calibration-formula-table/v1",
    }


def _base_scope(family: str) -> dict[str, object]:
    scope: dict[str, object] = {
        "a1_target_type": "target_price" if family == "A1" else None,
        "b_event_class": "earnings" if family == "B" else None,
        "b_match_anchors": (
            {
                "event_class": "earnings",
                "fingerprint_facts": ["actual_eps=2.10", "consensus_eps=2.00"],
                "release_date": "2024-10-19",
                "reporting_period": "2024 Q3",
            }
            if family == "B"
            else None
        ),
        "c_input_period": "2025-03-31" if family == "C" else None,
        "c_target_metric": "operating_revenue" if family == "C" else None,
        "c_target_period": "2025-06-30" if family == "C" else None,
        "entity_name": "Acme",
        "entity_ticker": "ACME",
        "family": family,
        "normalizer_version": V8_SCOPE_NORMALIZER,
        "requested_output": "direction" if family == "B" else "predicted_value",
        "source_spans": [{"field": "entity", "span": "Acme (ACME)"}],
    }
    scope["task_scope_sha256"] = sha256_json(scope)
    return scope


def _fixture_view(
    fixture_id: str, family: str, text: str, *, truncated: bool = False
) -> dict[str, object]:
    scope = _base_scope(family)
    view: dict[str, object] = {
        "category": family,
        "model_visible_content": text,
        "model_visible_snippet": text,
        "model_visible_title": f"Acme {fixture_id}",
        "task_scope": scope,
        "truncation_metadata": {
            "content_bytes": len(text.encode("utf-8")),
            "content_ends_at_limit": truncated,
            "content_limit_bytes": len(text.encode("utf-8")) if truncated else 400000,
            "hidden_axis": "period" if truncated else "none",
            "no_decisive_visible_span": truncated,
            "proven_visible_truncation": truncated,
            "provider_truncated": False,
            "token_boundary": False,
        },
    }
    return view


def _conformance_fixtures() -> dict[str, object]:
    specs = [
        ("a1-accounting-fair-value", "A1", "Acme asset fair value was 120 million."),
        ("a1-link-shell", "A1", "Acme investor relations archive and downloads."),
        ("a1-direct-endpoint", "A1", "Acme target price is 120 USD per share."),
        ("b-date-only", "B", "Acme earnings were released on 2024-10-19."),
        ("b-exact-facts", "B", "Acme earnings 2024-10-19 EPS actual 2.10, consensus 2.00."),
        (
            "b-direct-move",
            "B",
            "Acme earnings 2024-10-19 EPS actual 2.10, consensus 2.00; the stock rose 5% after the event.",
        ),
        ("c-exact-target", "C", "Acme operating_revenue on 2025-06-30 was 108 million."),
        ("c-other-period", "C", "Acme operating_revenue on 2025-03-31 was 100 million."),
        ("c-incomplete-formula", "C", "Acme attributable_net_profit was 10 million."),
        ("c-truncation", "C", "Acme operating_revenue", True),
    ]
    fixtures: list[dict[str, object]] = []
    for spec in specs:
        fixture_id, family, text, *flags = spec
        view = _fixture_view(fixture_id, family, text, truncated=bool(flags and flags[0]))
        decision = classify_v8(view)
        pair = {
            "outcome_revealing": decision.outcome_revealing,
            "task_relevance": decision.task_relevance,
        }
        fixtures.append(
            {
                "fixture_id": fixture_id,
                "path_a": pair,
                "path_b": dict(pair),
                "view": view,
            }
        )
    fixtures.sort(key=lambda row: str(row["fixture_id"]))
    return {"fixtures": fixtures, "schema_version": "leakage-calibration-conformance/v1"}


def _write_manifest(root: Path) -> None:
    manifest: dict[str, object] = {
        "adjudication": {
            "adjudicated_observed": 0,
            "raw_labels_expected": 800,
            "raw_labels_observed": 0,
            "status": "incomplete",
            "unresolved_disputes": 0,
        },
        "calibration_id": V8_RELEVANCE_CALIBRATION_ID,
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
        "query_corpus": {
            "path": "query-corpus.jsonl",
            "sha256": _sha256_file(root / "query-corpus.jsonl"),
        },
        "restricted_evidence_manifest_sha256": hashlib.sha256(
            b"leakage-calibration/v8/not-collected"
        ).hexdigest(),
        "rubric": {"path": "rubric.md", "sha256": _sha256_file(root / "rubric.md")},
        "sampling_plan": {
            "path": "sampling-plan.json",
            "sha256": _sha256_file(root / "sampling-plan.json"),
        },
        "schema_version": "leakage-calibration-manifest/v1",
    }
    entries: list[dict[str, object]] = []
    for relative in V8_TRACKED_FILES:
        payload = (root / relative).read_bytes()
        entries.append(
            {
                "bytes": len(payload),
                "path": relative,
                "row_count": (
                    len(payload.decode("utf-8").splitlines())
                    if relative
                    in (
                        set(V8_PILOT_JSONL_FILES)
                        | {path for path in EXPECTED_TRACKED_FILES if path.endswith(".jsonl")}
                    )
                    else 1
                    if relative in V8_JSON_FILES
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
        raise V8ScaffoldError(f"{path} is not an object")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["DEFAULT_V8_ROOT", "V8ScaffoldError", "prepare_v8_scaffold", "scaffold_v8"]
