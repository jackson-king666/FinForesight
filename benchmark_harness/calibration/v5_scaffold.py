"""Offline creation of the rubric-v4, zero-label v5 calibration scaffold."""

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
    V5_JSONL_FILES,
    V5_PILOT_JSONL_FILES,
    V5_RELEVANCE_AUTHORIZATION_ID,
    V5_RELEVANCE_CALIBRATION_ID,
    V5_RELEVANCE_RUBRIC_VERSION,
    V5_RELEVANCE_SCOPE,
    V5_RELEVANCE_SOURCE_COMMENT_ID,
    V5_TRACKED_FILES,
    canonical_json_bytes,
    validate_calibration_root,
)

DEFAULT_V5_ROOT = Path("data/calibration/leakage-probe/v5")
V5_AUTHORIZATION_SCHEMA = "leakage-calibration-authorization/v1"
V5_ISSUED_AT_UTC = "2026-08-31T10:08:35Z"

_RATIONALE_GRAMMAR = (
    "RCLASS=<B_SAME_EVENT|B_CROSS_PERIOD_ONLY|B_EVENT_UNCLEAR|A1_TARGET|"
    "A1_FUNDAMENTAL|A1_MARKET|A1_VALUATION|A1_CAPITAL|C_TARGET_METRIC|"
    "C_COMPONENT|C_EXPLICIT_DRIVER|TASK_INPUT_REPEAT|NO_QUALIFYING_INPUT|"
    "ENTITY_MISMATCH|UNCLEAR>; event_match=<exact|contains_exact|"
    "cross_period_only|none|unclear|na>; OCLASS=<DIRECT_TARGET|"
    "DIRECT_POST_EVENT_MOVE|INDIRECT_INPUT|SAME_EVENT_NO_PRICE_BRIDGE|"
    "NO_SIGNAL|UNCLEAR>; span=<short visible paraphrase>"
)

_V5_RUBRIC = f"""# Retrieval leakage rubric v4

## Scope and blinding

This v5 artifact is a rubric-only audit scaffold. A labeler may use the task
prompt and bounded model-visible title, snippet, and content. URL, publication
or update time, temporal labels, filter verdicts, scores, answers, ground
truth, system identity, and another labeler's record are not evidence for
relevance or outcome. Temporal labels remain a separate field.

## Mandatory decision order

Apply these checks in order for every item:

1. Identify the entity or security. If visible truncation prevents that, use
   `indeterminate/indeterminate`. If the visible payload explicitly names a
   different entity, use `not_relevant/no`.
2. Establish family-specific applicability. Direct keywords cannot bypass
   this check. If applicability fails, stop with `not_relevant/no`.
3. Only after applicability is established, classify the outcome as direct,
   indirect, or no signal. Use indeterminate only when a key discriminating
   fact is truncated or contradictory.

## Family rules

### A1 valuation

Same-security financial or operating inputs, observed price/return/volume,
valuation or segment valuation, and issuer-specific capital actions are
relevant indirect inputs. An explicit per-share target or fair-value price or
range is direct. A profile, navigation page, fact-free link, award/personnel
page, or only a task-provided price is `not_relevant/no`.

### B direction

The same issuer is not the same event. Use
`B_event_key=(security,event_class,reporting_package)`, where the reporting
package includes fiscal year, quarter/half/year package, and report type.
If the prompt has no period, require a unique visible event fingerprint such
as an exact date, actual EPS, consensus, surprise, or task-specific fact;
generic earnings/revenue/profit is never a same-event fallback. A visible
other period is immediately `not_relevant/no`. Same-event financial facts
without a security price bridge are `relevant/no`; a security-linked analyst,
investor, rating, target, or market judgment is indirect; an explicit
post-event security price or return move is direct.

### C metric forecast

The same company and target metric are relevant in any period. The exact
target-period realized metric is direct. Another period, annual/TTM/cumulative
value, guidance, consensus, or an explicit accounting component or driver is
indirect. Do not subtract a task-input value from a webpage value to create a
direct result. An unrelated metric without an explicit connection is
`not_relevant/no`.

## Rationale grammar

Every raw label uses this grammar:

`{_RATIONALE_GRAMMAR}`

This v5 bundle is exploratory rubric calibration only. The active pilot is a
separately authorized 40-item collection; no labels or binary gold are copied
from v3 or v4. `ranking_eligible=false`, `claim_eligible=false`, and
`live_unrestricted_authorized=false` remain in force. No unrestricted paper
probe is authorized by this bundle.
"""

_V5_README = """# Leakage calibration bundle v5

This is a rubric-v4 calibration bundle. Its top-level 400-item inputs are
copied from the immutable v4 audit and are not relabeled. The separately
authorized active v5 pilot contains 40 new items; it is not a paper experiment,
does not produce gold labels, and does not enter a ranking table.

The v5 rubric is `leakage-calibration-rubric/v4` and applies the decision
order entity -> applicability -> outcome. v3 and v4 remain immutable audit
leaves. `live_unrestricted_authorized=false`, `claim_eligible=false`, and
`ranking_eligible=false` remain in force. Only the bounded calibration
collection is authorized.
"""


class V5ScaffoldError(ValueError):
    """Raised when the v5 scaffold cannot be created safely."""


def prepare_v5_scaffold(
    root: str | Path = DEFAULT_V5_ROOT,
    *,
    source_root: str | Path | None = None,
) -> Path:
    """Create or verify the v5 bundle from the immutable v4 audit."""

    root_path = Path(root).expanduser().resolve()
    if root_path.exists():
        try:
            state = validate_calibration_root(root_path)
        except ValueError as exc:
            try:
                existing_manifest = _load_json(root_path / "artifact_manifest.json")
                existing_pilot = any(
                    (root_path / relative).read_bytes() for relative in V5_PILOT_JSONL_FILES
                )
            except (OSError, V5ScaffoldError) as read_exc:
                raise V5ScaffoldError("existing v5 root failed validation") from read_exc
            if (
                existing_manifest.get("calibration_id") != V5_RELEVANCE_CALIBRATION_ID
                or existing_pilot
            ):
                raise V5ScaffoldError("existing v5 root failed validation") from exc
            source = root_path.parent / "v4"
            if not source.is_dir():
                raise V5ScaffoldError("immutable v4 audit root is unavailable") from exc
            try:
                source_state = validate_calibration_root(source)
            except ValueError as source_exc:
                raise V5ScaffoldError("v4 audit root failed validation") from source_exc
            if source_state.get("calibration_id") != "leakage-probe/v4":
                raise V5ScaffoldError("scaffold source must be the v4 audit root") from exc
            _write_v5_schemas(root_path, source)
            _write_v5_authorization(root_path)
            _write_v5_cohort(root_path, source)
            _write_v5_report(root_path, source)
            (root_path / "README.md").write_text(_V5_README, encoding="utf-8")
            (root_path / "rubric.md").write_text(_V5_RUBRIC, encoding="utf-8")
            _write_v5_manifest(root_path)
            state = validate_calibration_root(root_path)
        if state.get("calibration_id") != V5_RELEVANCE_CALIBRATION_ID:
            raise V5ScaffoldError("existing root is not the v5 calibration bundle")
        return root_path

    source = Path(source_root).expanduser().resolve() if source_root else root_path.parent / "v4"
    if not source.is_dir():
        raise V5ScaffoldError("immutable v4 audit root is unavailable")
    try:
        source_state = validate_calibration_root(source)
    except ValueError as exc:
        raise V5ScaffoldError("v4 audit root failed validation") from exc
    if source_state.get("calibration_id") != "leakage-probe/v4":
        raise V5ScaffoldError("scaffold source must be the v4 audit root")

    root_path.mkdir(parents=True, exist_ok=False)
    for relative in EXPECTED_TRACKED_FILES:
        target = root_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative in {"README.md", "rubric.md", "authorization.json", "cohort.json"}:
            continue
        shutil.copyfile(source / relative, target)
    for relative in V5_PILOT_JSONL_FILES:
        target = root_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")

    _write_v5_schemas(root_path, source)
    _write_v5_authorization(root_path)
    _write_v5_cohort(root_path, source)
    _write_v5_report(root_path, source)
    (root_path / "README.md").write_text(_V5_README, encoding="utf-8")
    (root_path / "rubric.md").write_text(_V5_RUBRIC, encoding="utf-8")
    _write_v5_manifest(root_path)
    return root_path


def scaffold_v5(root: str | Path = DEFAULT_V5_ROOT) -> dict[str, object]:
    """Create or verify the v5 bundle and return its validation state."""

    root_path = prepare_v5_scaffold(root)
    result = validate_calibration_root(root_path)
    report = _load_json(root_path / "reports/calibration-report.json")
    result["pilot_status"] = report.get("pilot_status")
    result["pilot_n"] = report.get("pilot_n")
    return result


def _write_v5_schemas(root: Path, source: Path) -> None:
    auth_path = root / "schemas/authorization.schema.json"
    auth_schema = _load_json(source / "schemas/authorization.schema.json")
    auth_schema["title"] = "Explicit v5 rubric-v4 pilot authorization"
    auth_schema["properties"]["calibration_id"] = {"const": V5_RELEVANCE_CALIBRATION_ID}
    auth_schema["properties"]["calibration_unrestricted_collection_authorized"] = {"const": True}
    auth_schema["properties"]["scope"] = {"const": V5_RELEVANCE_SCOPE}
    _write_json(auth_path, auth_schema)

    raw_path = root / "schemas/raw-label.schema.json"
    raw_schema = _load_json(source / "schemas/raw-label.schema.json")
    raw_schema["title"] = "V5 rubric-v4 blind raw leakage label"
    raw_schema["properties"]["rubric_version"] = {"const": V5_RELEVANCE_RUBRIC_VERSION}
    _write_json(raw_path, raw_schema)

    report_path = root / "schemas/report.schema.json"
    report_schema = _load_json(source / "schemas/report.schema.json")
    report_schema["title"] = "V5 rubric-only leakage calibration report"
    report_schema["properties"]["pilot_n"] = {"minimum": 0, "maximum": 40, "type": "integer"}
    report_schema["properties"]["pilot_status"] = {"enum": ["not_run", "pilot"]}
    _write_json(report_path, report_schema)


def _write_v5_authorization(root: Path) -> None:
    _write_json(
        root / "authorization.json",
        {
            "authorization_id": V5_RELEVANCE_AUTHORIZATION_ID,
            "calibration_id": V5_RELEVANCE_CALIBRATION_ID,
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
            "issued_at_utc": V5_ISSUED_AT_UTC,
            "pilot_item_count": 40,
            "pilot_strata": {stratum: 10 for stratum in ("A1-T1", "B-T1", "B-T2", "C-T2")},
            "same_candidates_cutoff_filter": True,
            "schema_version": V5_AUTHORIZATION_SCHEMA,
            "scope": V5_RELEVANCE_SCOPE,
            "source_comment_id": V5_RELEVANCE_SOURCE_COMMENT_ID,
            "status": "active",
        },
    )


def _write_v5_cohort(root: Path, source: Path) -> None:
    cohort = _load_json(source / "cohort.json")
    cohort.update(
        {
            "calibration_status": "incomplete",
            "cohort_id": V5_RELEVANCE_CALIBRATION_ID,
            "live_unrestricted_authorized": False,
        }
    )
    _write_json(root / "cohort.json", cohort)


def _write_v5_report(root: Path, source: Path) -> None:
    report = _load_json(source / "reports/calibration-report.json")
    report.update(
        {
            "calibration_status": "incomplete",
            "claim_eligible": False,
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
            "exclusions_by_type": {"v5_active_pilot_not_run": 40},
            "filter_validated": False,
            "generated_at_utc": V5_ISSUED_AT_UTC,
            "live_unrestricted_authorized": False,
            "pilot_n": 0,
            "pilot_status": "not_run",
            "quality_gates_passed": False,
            "raw_agreement": None,
        }
    )
    _write_json(root / "reports/calibration-report.json", report)


def _write_v5_manifest(root: Path) -> None:
    manifest: dict[str, object] = {
        "adjudication": {
            "adjudicated_observed": 0,
            "raw_labels_expected": 800,
            "raw_labels_observed": 0,
            "status": "incomplete",
            "unresolved_disputes": 0,
        },
        "calibration_id": V5_RELEVANCE_CALIBRATION_ID,
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
        "restricted_evidence_manifest_sha256": None,
        "rubric": {"path": "rubric.md", "sha256": ""},
        "sampling_plan": {"path": "sampling-plan.json", "sha256": ""},
        "schema_version": "leakage-calibration-manifest/v1",
    }
    manifest["rubric"]["sha256"] = _sha256_file(root / "rubric.md")  # type: ignore[index]
    manifest["sampling_plan"]["sha256"] = _sha256_file(root / "sampling-plan.json")  # type: ignore[index]
    manifest["query_corpus"]["sha256"] = _sha256_file(root / "query-corpus.jsonl")  # type: ignore[index]
    files: list[dict[str, object]] = []
    for relative in V5_TRACKED_FILES:
        payload = (root / relative).read_bytes()
        row_count: int | None
        if relative in V5_JSONL_FILES:
            row_count = len(payload.decode("utf-8").splitlines())
        elif relative.endswith(".json"):
            row_count = 1
        else:
            row_count = None
        files.append(
            {
                "bytes": len(payload),
                "path": relative,
                "row_count": row_count,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest["files"] = files
    manifest_bytes = canonical_json_bytes(manifest)
    (root / "artifact_manifest.json").write_bytes(manifest_bytes)
    (root / "CALIBRATION.sha256").write_text(
        hashlib.sha256(manifest_bytes).hexdigest() + "  artifact_manifest.json\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V5ScaffoldError(f"unable to read JSON source {path}") from exc
    if not isinstance(value, dict):
        raise V5ScaffoldError(f"JSON source {path} is not an object")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["DEFAULT_V5_ROOT", "V5ScaffoldError", "prepare_v5_scaffold", "scaffold_v5"]
