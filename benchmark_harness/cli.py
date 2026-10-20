"""Command-line entry points for preflight, isolated run, and report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from benchmark_harness.calibration import (
    PilotCollectionError,
    QCValidationError,
    assess_and_write_review_pack,
    label_pilot,
    load_policy,
    run_fresh_pilot,
    run_frozen,
    run_pilot,
    run_v5_pilot,
    run_v6_pilot,
    scaffold_v5,
    scaffold_v6,
    validate_calibration_root,
    validate_policy,
    write_b_facing_pack,
)
from benchmark_harness.calibration.qc_validator import (
    assert_allowed_source_path,
    assert_overlay_outside_source,
    read_jsonl,
    validate_review_pack,
    validate_source_pilot,
)
from benchmark_harness.config import ConfigError, load_experiment_config
from benchmark_harness.dataset_bridge import DatasetBridgeError
from benchmark_harness.dataset_sources import DatasetSourceError
from benchmark_harness.experiment import (
    AnalysisRequest,
    DataInventoryError,
    ExecutionDependencies,
    ExecutionError,
    ExperimentExecutionRequest,
    ManifestError,
    T3LockEntry,
    T3RegistryError,
    T3ScheduleEntry,
    T3SettlementEntry,
    V4ApiError,
    analyze_experiment,
    analyze_full_release,
    append_lock,
    append_schedule,
    append_settlement,
    audit_experiment,
    audit_full_release,
    build_cohort,
    build_disposition_ledger,
    build_path_b_v3_mini_reseal,
    build_plan,
    compute_inventory,
    load_manifest,
    load_path_b_v3_plans,
    load_plan,
    load_score_blind_records,
    load_t3_registry,
    prepare_path_b_v3_r3_review_pack,
    prepare_path_b_v3_review_pack,
    real_system_factory,
    report_analysis,
    resume_experiment,
    run_experiment,
    run_path_b_v3,
    validate_manifest_artifacts,
    validate_path_b_v2_bundle,
    validate_path_b_v3_plans,
    validate_t3_registry,
    validate_v3_artifacts,
    write_collection_ledger,
    write_plan,
    write_v3_artifacts,
)
from benchmark_harness.experiment.live_model import LiveRuntimeError
from benchmark_harness.experiment.v3 import V3_EXPECTED_RECORD_COUNT
from benchmark_harness.external_harnesses.checkout import ExternalHarnessError
from benchmark_harness.profile_registry import supported_baselines, supported_profiles
from benchmark_harness.reporting.compare import write_comparison
from benchmark_harness.runner import BenchmarkRunner, RunnerError
from benchmark_harness.search import SearchGatewayError
from benchmark_harness.search_service.api import SearchServiceError
from benchmark_harness.search_service.config import (
    SearchServiceConfigError,
    load_search_service_config,
)
from benchmark_harness.search_service.lifecycle import SearchServiceLifecycle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finforesight",
        description="FinForesight financial forecasting research framework.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    preflight = subcommands.add_parser("preflight", help="Validate dataset and endpoint readiness.")
    preflight.add_argument("--config", required=True, help="Path to experiment TOML.")

    data_sync = subcommands.add_parser(
        "data-sync", help="Prefetch all pinned Hugging Face configs into data/.huggingface."
    )
    data_sync.add_argument("--config", required=True, help="Path to experiment TOML.")

    external_sync = subcommands.add_parser(
        "external-sync",
        help="Explicitly clone/install a pinned external harness (never implicit).",
    )
    external_sync.add_argument("--config", required=True, help="Path to experiment TOML.")
    external_sync.add_argument(
        "--harness",
        default="financeharness",
        choices=["financeharness", "finance-harness"],
        help="External harness to synchronize (only FinanceHarness is supported).",
    )
    external_sync.add_argument(
        "--acknowledge-undeclared-license",
        action="store_true",
        help="Acknowledge that the external source declares no license.",
    )

    run = subcommands.add_parser("run", help="Run a configured profile in child processes.")
    run.add_argument("--config", required=True, help="Path to experiment TOML.")
    run.add_argument(
        "--baseline",
        action="append",
        required=True,
        help=(
            "Baseline (profile-dependent; supported values: "
            + ", ".join(
                sorted(
                    {
                        baseline
                        for profile in supported_profiles()
                        for baseline in supported_baselines(profile)
                    }
                )
            )
            + "). Repeat for comparisons."
        ),
    )
    run.add_argument(
        "--categories",
        action="append",
        default=[],
        help="Comma-separated category selectors (for example A1,C,E). Repeat if needed.",
    )
    run.add_argument("--run-id", default=None, help="Optional safe output directory name.")

    report = subcommands.add_parser("report", help="Rebuild comparison JSON/CSV from artifacts.")
    report.add_argument("--run-id", required=True)
    report.add_argument(
        "--results-dir",
        default="results",
        help="Results root (default: ./results; use the config's trace.results_dir if changed).",
    )

    experiment = subcommands.add_parser(
        "experiment", help="Validate and plan offline paper experiments and analysis."
    )
    experiment_subcommands = experiment.add_subparsers(dest="experiment_action", required=True)
    validate = experiment_subcommands.add_parser(
        "validate", help="Validate an experiment manifest."
    )
    validate.add_argument("--manifest", required=True)
    plan = experiment_subcommands.add_parser("plan", help="Build a deterministic experiment plan.")
    plan.add_argument("--manifest", required=True)
    plan.add_argument("--task-id", action="append", default=[])
    plan.add_argument("--task-file", default=None, help="JSON array or JSONL task-id file.")
    plan.add_argument("--split-file", default=None, help="Outcome-cluster split JSONL artifact.")
    plan.add_argument("--selection", choices=("smoke", "dev", "test", "all"), default="all")
    plan.add_argument("--output", required=True)
    for action, help_text in (
        ("run", "Run a binding-aware experiment selection."),
        ("resume", "Resume append-only experiment attempts."),
    ):
        execution = experiment_subcommands.add_parser(action, help=help_text)
        execution.add_argument("--manifest", required=True)
        execution.add_argument("--plan", required=True)
        execution.add_argument("--results-root", required=True)
        execution.add_argument(
            "--selection", choices=("smoke", "dev", "test", "all"), default="all"
        )
        execution.add_argument("--max-workers", type=int, default=3)
        execution.add_argument("--max-attempts", type=int, default=3)
    audit = experiment_subcommands.add_parser(
        "audit", help="Validate raw artifact lineage offline."
    )
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--plan", required=True)
    audit.add_argument("--results-root", required=True)
    analyze = experiment_subcommands.add_parser("analyze", help="Build offline C1/C2/C3 outputs.")
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--plan", required=True)
    analyze.add_argument("--input-root", required=True)
    analyze.add_argument("--output", dest="output", required=True)
    analyze.add_argument("--selection", choices=("smoke", "dev", "test", "all"), default="all")
    analyze.add_argument("--analysis-as-of", default=None)
    analyze.add_argument("--metric", default="exploratory_task_utility_v1")
    paper_report = experiment_subcommands.add_parser(
        "report", help="List generated analysis artifacts."
    )
    paper_report.add_argument("--analysis-dir", required=True)

    v3_lock = experiment_subcommands.add_parser(
        "v3-lock",
        help="Build an offline score-blind experiment/v3 lock and cohort bundle.",
    )
    v3_lock.add_argument(
        "--records",
        required=True,
        help="JSONL task metadata source; score/result/prediction paths are rejected.",
    )
    v3_lock.add_argument("--output", required=True, help="Append-only v3 artifact directory.")
    v3_lock.add_argument(
        "--expected-record-count",
        type=int,
        default=V3_EXPECTED_RECORD_COUNT,
        help=f"Expected variable record count (default: {V3_EXPECTED_RECORD_COUNT}).",
    )
    v3_lock.add_argument("--analysis-version", default="experiment-v3-r1")
    v3_lock.add_argument("--seed-namespace", default="v3")
    v3_validate = experiment_subcommands.add_parser(
        "v3-validate", help="Validate an existing offline experiment/v3 lock bundle."
    )
    v3_validate.add_argument("--root", required=True, help="v3 artifact directory.")

    v4_inventory = experiment_subcommands.add_parser(
        "v4-inventory", help="Build the score-blind current-pin inventory and gap masks."
    )
    v4_inventory.add_argument("--records", required=True, help="JSON array or JSONL metadata rows.")
    v4_inventory.add_argument("--as-of", default=None, help="Declared UTC date or timestamp.")
    v4_inventory.add_argument("--output", default=None, help="Optional JSON output path.")

    v4_ledger = experiment_subcommands.add_parser(
        "v4-ledger", help="Create the append-only source disposition ledger."
    )
    v4_ledger.add_argument("--records", required=True, help="JSON array or JSONL metadata rows.")
    v4_ledger.add_argument("--output", required=True, help="New collection ledger JSONL path.")

    for action, help_text in (
        ("t3-schedule", "Append one prospective T3 schedule entry."),
        ("t3-lock", "Append one pre-outcome T3 prediction lock."),
        ("t3-settle", "Append one independently sourced T3 settlement."),
    ):
        t3_entry = experiment_subcommands.add_parser(action, help=help_text)
        t3_entry.add_argument(
            "--registry", required=True, help="Append-only T3 registry JSONL path."
        )
        t3_entry.add_argument(
            "--entry", required=True, help="JSON object containing the entry payload."
        )
        if action == "t3-settle":
            t3_entry.add_argument(
                "--outcome-text-sha256",
                default=None,
                help="Optional SHA-256 of the raw outcome text for independence checking.",
            )
    t3_validate = experiment_subcommands.add_parser(
        "t3-validate", help="Verify a chained prospective T3 registry."
    )
    t3_validate.add_argument("--registry", required=True, help="T3 registry JSONL path.")

    v4_audit = experiment_subcommands.add_parser(
        "v4-audit", help="Verify a generated experiment/v4 analysis artifact directory."
    )
    v4_audit.add_argument("--root", required=True, help="Full-release analysis artifact directory.")

    leakage_calibration = subcommands.add_parser(
        "leakage-calibration", help="Validate an offline leakage-calibration bundle."
    )
    leakage_calibration_subcommands = leakage_calibration.add_subparsers(
        dest="leakage_calibration_action", required=True
    )
    leakage_validate = leakage_calibration_subcommands.add_parser(
        "validate", help="Validate calibration hashes, schemas, receipts, and labels."
    )
    leakage_validate.add_argument(
        "--root",
        default="data/calibration/leakage-probe/v1",
        help="Calibration bundle root (default: data/calibration/leakage-probe/v1).",
    )
    leakage_pilot = leakage_calibration_subcommands.add_parser(
        "pilot", help="Collect and blindly label the explicitly authorized 40-item pilot."
    )
    leakage_pilot.add_argument(
        "--root",
        default="data/calibration/leakage-probe/v1",
        help="Calibration bundle root (default: data/calibration/leakage-probe/v1).",
    )
    leakage_pilot.add_argument(
        "--cache-root",
        default="data/calibration-cache/leakage-probe/v1",
        help="Ignored restricted payload cache root.",
    )
    leakage_pilot.add_argument(
        "--dataset-cache",
        default="data/.huggingface",
        help="Ignored Hugging Face cache root.",
    )
    leakage_pilot.add_argument("--timeout", type=float, default=30.0)
    leakage_fresh_pilot = leakage_calibration_subcommands.add_parser(
        "pilot-v4",
        help="Collect and blindly label the authorized fresh v4 relevance pilot.",
    )
    leakage_fresh_pilot.add_argument(
        "--root",
        default="data/calibration/leakage-probe/v4",
        help="Calibration bundle root (default: data/calibration/leakage-probe/v4).",
    )
    leakage_fresh_pilot.add_argument(
        "--cache-root",
        default="data/calibration-cache/leakage-probe/v4",
        help="Ignored restricted payload cache root.",
    )
    leakage_fresh_pilot.add_argument(
        "--dataset-cache",
        default="data/.huggingface",
        help="Ignored Hugging Face cache root.",
    )
    leakage_fresh_pilot.add_argument("--timeout", type=float, default=30.0)
    leakage_v5_pilot = leakage_calibration_subcommands.add_parser(
        "pilot-v5",
        help="Collect and blindly label the authorized fresh v5 rubric-v4 pilot.",
    )
    leakage_v5_pilot.add_argument(
        "--root",
        default="data/calibration/leakage-probe/v5",
        help="Calibration bundle root (default: data/calibration/leakage-probe/v5).",
    )
    leakage_v5_pilot.add_argument(
        "--cache-root",
        default="data/calibration-cache/leakage-probe/v5",
        help="Ignored restricted payload cache root.",
    )
    leakage_v5_pilot.add_argument(
        "--dataset-cache",
        default="data/.huggingface",
        help="Ignored Hugging Face cache root.",
    )
    leakage_v5_pilot.add_argument("--timeout", type=float, default=30.0)
    leakage_v5_scaffold = leakage_calibration_subcommands.add_parser(
        "scaffold-v5",
        help="Create or validate the offline rubric-v4 zero-label v5 scaffold.",
    )
    leakage_v5_scaffold.add_argument(
        "--root",
        default="data/calibration/leakage-probe/v5",
        help="Calibration bundle root (default: data/calibration/leakage-probe/v5).",
    )
    leakage_v6_pilot = leakage_calibration_subcommands.add_parser(
        "pilot-v6",
        help="Collect and blindly label the authorized fresh v6 rubric-v5 pilot.",
    )
    leakage_v6_pilot.add_argument(
        "--root",
        default="data/calibration/leakage-probe/v6",
        help="Calibration bundle root (default: data/calibration/leakage-probe/v6).",
    )
    leakage_v6_pilot.add_argument(
        "--cache-root",
        default="data/calibration-cache/leakage-probe/v6",
        help="Ignored restricted payload cache root.",
    )
    leakage_v6_pilot.add_argument(
        "--dataset-cache",
        default="data/.huggingface",
        help="Ignored Hugging Face cache root.",
    )
    leakage_v6_pilot.add_argument("--timeout", type=float, default=30.0)
    leakage_v6_scaffold = leakage_calibration_subcommands.add_parser(
        "scaffold-v6",
        help="Create or validate the offline rubric-v5 zero-label v6 scaffold.",
    )
    leakage_v6_scaffold.add_argument(
        "--root",
        default="data/calibration/leakage-probe/v6",
        help="Calibration bundle root (default: data/calibration/leakage-probe/v6).",
    )
    leakage_frozen = leakage_calibration_subcommands.add_parser(
        "freeze-400", help="Collect and blindly label the authorized frozen 400-item cohort."
    )
    leakage_frozen.add_argument(
        "--root",
        default="data/calibration/leakage-probe/v2",
        help="Calibration bundle root (default: data/calibration/leakage-probe/v2).",
    )
    leakage_frozen.add_argument(
        "--cache-root",
        default="data/calibration-cache/leakage-probe/v2",
        help="Ignored restricted payload cache root.",
    )
    leakage_frozen.add_argument(
        "--dataset-cache",
        default="data/.huggingface",
        help="Ignored Hugging Face cache root.",
    )
    leakage_frozen.add_argument("--timeout", type=float, default=30.0)
    leakage_label = leakage_calibration_subcommands.add_parser(
        "label-pilot", help="Run the DEV/Labeler-A blind pass over an existing pilot corpus."
    )
    leakage_label.add_argument(
        "--root",
        default="data/calibration/leakage-probe/v1",
        help="Calibration bundle root (default: data/calibration/leakage-probe/v1).",
    )
    leakage_label.add_argument(
        "--cache-root",
        default="data/calibration-cache/leakage-probe/v1",
        help="Ignored restricted payload cache root.",
    )
    leakage_label.add_argument(
        "--reemit",
        action="store_true",
        help="Rebuild the blind view from the ignored cache before re-emitting Labeler A.",
    )

    qc_assess = subcommands.add_parser(
        "qc-assess",
        help="Assess an Evidence-QC A/B pair offline and write one append-only review pack.",
    )
    qc_assess.add_argument(
        "--policy",
        default="data/calibration/evidence-qc/v8-r1/policy.json",
        help="Frozen Evidence-QC policy JSON.",
    )
    qc_assess.add_argument("--views", required=True, help="Frozen pilot annotation-view JSONL.")
    qc_assess.add_argument("--label-a", required=True, help="Raw Labeler-A JSONL.")
    qc_assess.add_argument("--label-b", required=True, help="Raw Labeler-B JSONL.")
    qc_assess.add_argument(
        "--label-c", default=None, help="Optional blind C JSONL after B is sealed."
    )
    qc_assess.add_argument("--output", required=True, help="New review-pack directory.")
    qc_assess.add_argument(
        "--source-root",
        default="data/calibration/leakage-probe/v8",
        help="v8 root; only CALIBRATION.sha256 and v8/pilot files are read.",
    )
    qc_assess.add_argument(
        "--source-manifest",
        default=None,
        help="Optional pilot-only source manifest to include in the review pack.",
    )

    qc_pack = subcommands.add_parser(
        "qc-pack",
        help="Create an immutable pilot-only review pack for independent B.",
    )
    qc_pack.add_argument(
        "--policy",
        default="data/calibration/evidence-qc/v8-r1/policy.json",
        help="Frozen Evidence-QC policy JSON.",
    )
    qc_pack.add_argument(
        "--source-root",
        default="data/calibration/leakage-probe/v8",
        help="v8 root; only B-facing pilot evidence and the source seal are read.",
    )
    qc_pack.add_argument(
        "--source-manifest",
        default=None,
        help="Pilot-only source manifest (default: next to --policy).",
    )
    qc_pack.add_argument("--output", required=True, help="New B-facing pack directory.")

    qc_validate = subcommands.add_parser(
        "qc-validate",
        help="Validate an Evidence-QC review pack offline without reading the 400-item source.",
    )
    qc_validate.add_argument("--root", required=True, help="Review-pack directory.")
    qc_validate.add_argument(
        "--policy",
        default="data/calibration/evidence-qc/v8-r1/policy.json",
        help="Frozen Evidence-QC policy JSON.",
    )
    qc_validate.add_argument(
        "--source-root",
        default="data/calibration/leakage-probe/v8",
        help="v8 root; only CALIBRATION.sha256 and v8/pilot files are read.",
    )

    path_b_v2_validate = subcommands.add_parser(
        "path-b-v2-validate",
        help="Validate a Path B v2 review pack offline in a clean runtime.",
    )
    path_b_v2_validate.add_argument(
        "--root", required=True, help="Path B v2 review-pack directory."
    )
    path_b_v2_validate.add_argument(
        "--contract",
        default="configs/path-b-v2-review-contract.json",
        help="Frozen Path B v2 review contract JSON.",
    )

    path_b_v3_review_pack = subcommands.add_parser(
        "path-b-v3-review-pack",
        help="Prepare a score-blind current-pin Path B v3 reviewer pack.",
    )
    path_b_v3_review_pack.add_argument(
        "--source-root",
        required=True,
        help="Offline path_b_v2 source directory at the locked Hub revision.",
    )
    path_b_v3_review_pack.add_argument(
        "--output",
        required=True,
        help="New immutable reviewer-pack directory.",
    )
    path_b_v3_review_pack.add_argument(
        "--contract",
        default="configs/path-b-v2-review-contract.json",
        help="Frozen Path B v2 review contract JSON.",
    )

    path_b_v3_r3_review_pack = subcommands.add_parser(
        "path-b-v3-r3-review-pack",
        help="Derive and seal the append-only Path B v3 r3 component review pack.",
    )
    path_b_v3_r3_review_pack.add_argument(
        "--source-root",
        required=True,
        help="Offline path_b_v2 source directory at the locked Hub revision.",
    )
    path_b_v3_r3_review_pack.add_argument(
        "--output",
        required=True,
        help="New immutable r3 reviewer-pack directory.",
    )
    path_b_v3_r3_review_pack.add_argument(
        "--contract",
        default="configs/path-b-v2-review-contract.json",
        help="Frozen Path B v2 review contract JSON.",
    )

    path_b_v3_mini_reseal = subcommands.add_parser(
        "path-b-v3-mini-reseal",
        help="Create an append-only Path B v3 G3 plan root sealed for GPT-4.1 mini.",
    )
    path_b_v3_mini_reseal.add_argument(
        "--source-root", required=True, help="Existing sealed GPT-4.1 Path B v3 G3 root."
    )
    path_b_v3_mini_reseal.add_argument(
        "--output", required=True, help="New append-only GPT-4.1 mini G3 plan root."
    )
    path_b_v3_mini_reseal.add_argument(
        "--g2-root", default=None, help="Sealed G2 freeze directory (defaults to its sibling)."
    )
    path_b_v3_mini_reseal.add_argument(
        "--contract",
        default="configs/path-b-v2-review-contract.json",
        help="Frozen Path B v2 review contract JSON.",
    )

    path_b_v3_validate = subcommands.add_parser(
        "path-b-v3-validate",
        help="Validate all three sealed current-pin Path B v3 manifest/plan pairs.",
    )
    path_b_v3_validate.add_argument(
        "--root", required=True, help="Sealed Path B v3 G3 plan directory."
    )
    path_b_v3_validate.add_argument(
        "--g2-root", default=None, help="Sealed G2 freeze directory (defaults to its sibling)."
    )
    path_b_v3_validate.add_argument(
        "--contract",
        default="configs/path-b-v2-review-contract.json",
        help="Frozen Path B v2 review contract JSON.",
    )

    path_b_v3_run = subcommands.add_parser(
        "path-b-v3-run",
        help="Credential-gate and dispatch the sealed current-pin Path B v3 plan set.",
    )
    path_b_v3_run.add_argument("--root", required=True, help="Sealed Path B v3 G3 plan directory.")
    path_b_v3_run.add_argument(
        "--g2-root", default=None, help="Sealed G2 freeze directory (defaults to its sibling)."
    )
    path_b_v3_run.add_argument(
        "--contract",
        default="configs/path-b-v2-review-contract.json",
        help="Frozen Path B v2 review contract JSON.",
    )
    path_b_v3_run.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the sealed plans without loading data, credentials, or results.",
    )
    path_b_v3_run.add_argument(
        "--dry-dispatch",
        action="store_true",
        help=(
            "Validate credentials and every sealed cell without invoking a system "
            "or writing results."
        ),
    )
    path_b_v3_run.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Maximum live cells between sanitized checkpoints (default: 8).",
    )
    path_b_v3_run.add_argument(
        "--start-search-service",
        action="store_true",
        help="Explicitly start the resident loopback Search+Filter service before live dispatch.",
    )
    path_b_v3_run.add_argument(
        "--search-service-config",
        default=None,
        help="Search+Filter daemon TOML used with --start-search-service.",
    )

    service = subcommands.add_parser(
        "search-service", help="Explicitly serve and manage the resident Search+Filter service."
    )
    service_subcommands = service.add_subparsers(dest="service_action", required=True)
    for action in ("serve", "start", "status", "stop"):
        action_parser = service_subcommands.add_parser(action)
        action_parser.add_argument("--config", required=True, help="Path to daemon TOML.")
        if action == "stop":
            action_parser.add_argument("--force", action="store_true")
    return parser


def _categories(values: Sequence[str]) -> list[str] | None:
    categories = [
        category.strip() for value in values for category in value.split(",") if category.strip()
    ]
    return categories or None


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "search-service" or (
        args.command == "leakage-calibration"
        and args.leakage_calibration_action
        in {"pilot", "pilot-v4", "pilot-v5", "pilot-v6", "freeze-400"}
    ):
        _load_service_dotenv()
    else:
        # Never import service-owned credentials into an experiment process.
        # The resident service is the only owner of Volc/filter settings.
        _load_experiment_dotenv()
        if args.command == "path-b-v3-run" and not args.validate_only:
            # The v3 live connector owns unrestricted parent-side Volc access;
            # child environments remain stripped by the adapter isolation seam.
            _load_service_dotenv()
    try:
        if args.command == "search-service":
            lifecycle = SearchServiceLifecycle(load_search_service_config(args.config))
            if args.service_action == "serve":
                lifecycle.serve()
                result = {"status": "stopped"}
            elif args.service_action == "start":
                result = lifecycle.start().as_dict()
            elif args.service_action == "status":
                status = lifecycle.status()
                result = status.as_dict()
                if status.state == "stopped":
                    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
                    return 1
                if not status.ready:
                    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
                    return 2
            elif args.service_action == "stop":
                status = lifecycle.stop(force=args.force)
                result = status.as_dict()
                if status.state != "stopped":
                    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
                    return 2
            else:  # pragma: no cover - argparse guarantees this branch.
                raise AssertionError(args.service_action)
        elif args.command == "preflight":
            config = load_experiment_config(args.config)
            result = BenchmarkRunner(config).preflight()
        elif args.command == "data-sync":
            config = load_experiment_config(args.config)
            result = BenchmarkRunner(config).data_sync()
        elif args.command == "external-sync":
            config = load_experiment_config(args.config)
            if config.execution.input_policy != "deep_research":
                raise ConfigError("external-sync requires a deep_research configuration.")
            result = BenchmarkRunner(config).external_sync(
                acknowledge_undeclared_license=args.acknowledge_undeclared_license
            )
        elif args.command == "run":
            config = load_experiment_config(args.config)
            result = BenchmarkRunner(config).run(
                baselines=args.baseline,
                categories=_categories(args.categories),
                run_id=args.run_id,
            )
        elif args.command == "report":
            run_dir = Path(args.results_dir).expanduser().resolve() / args.run_id
            paths = write_comparison(run_dir)
            result = {
                "run_id": args.run_id,
                "comparison": {key: str(value) for key, value in paths.items()},
            }
        elif args.command == "qc-assess":
            policy, policy_sha256 = load_policy(args.policy)
            views_path = Path(args.views).expanduser().resolve()
            label_a_path = Path(args.label_a).expanduser().resolve()
            label_b_path = Path(args.label_b).expanduser().resolve()
            output_path = Path(args.output).expanduser().resolve()
            label_c_path = Path(args.label_c).expanduser().resolve() if args.label_c else None
            source_root = (
                Path(args.source_root).expanduser().resolve() if args.source_root else None
            )
            source_manifest = None
            if source_root is not None:
                validate_source_pilot(source_root, policy=policy)
                assert_allowed_source_path(views_path, source_root)
                assert_allowed_source_path(label_a_path, source_root)
                assert_overlay_outside_source(output_path, source_root)
                try:
                    label_b_path.relative_to(source_root)
                except ValueError:
                    pass
                else:
                    raise QCValidationError(
                        "Labeler B must be supplied outside the protected v8 root"
                    )
                if label_c_path is not None:
                    try:
                        label_c_path.relative_to(source_root)
                    except ValueError:
                        pass
                    else:
                        raise QCValidationError(
                            "Reviewer C must be supplied outside the protected v8 root"
                        )
            if args.source_manifest:
                source_manifest = _read_qc_json(args.source_manifest)
            views = read_jsonl(views_path)
            labels_a = read_jsonl(label_a_path)
            labels_b = read_jsonl(label_b_path)
            labels_c = read_jsonl(label_c_path) if label_c_path else None
            result = assess_and_write_review_pack(
                output_path,
                policy=policy,
                policy_sha256=policy_sha256,
                views=views,
                labels_a=labels_a,
                labels_b=labels_b,
                labels_c=labels_c,
                source_manifest=source_manifest,
                source_root=source_root,
            )
        elif args.command == "qc-pack":
            result = write_b_facing_pack(
                args.output,
                policy_path=args.policy,
                source_root=args.source_root,
                source_manifest_path=args.source_manifest,
            )
        elif args.command == "qc-validate":
            validate_policy(args.policy)
            result = validate_review_pack(
                args.root,
                policy_path=args.policy,
                source_root=args.source_root,
            )
        elif args.command == "path-b-v2-validate":
            result = validate_path_b_v2_bundle(args.root, contract_path=args.contract)
        elif args.command == "path-b-v3-review-pack":
            result = prepare_path_b_v3_review_pack(
                args.source_root,
                args.output,
                contract_path=args.contract,
            )
        elif args.command == "path-b-v3-r3-review-pack":
            result = prepare_path_b_v3_r3_review_pack(
                args.source_root,
                args.output,
                contract_path=args.contract,
            )
        elif args.command == "path-b-v3-mini-reseal":
            result = build_path_b_v3_mini_reseal(
                args.source_root,
                args.output,
                g2_root=args.g2_root,
                contract_path=args.contract,
            )
        elif args.command == "path-b-v3-validate":
            result = validate_path_b_v3_plans(
                args.root,
                g2_root=args.g2_root,
                contract_path=args.contract,
            )
        elif args.command == "path-b-v3-run":
            if args.start_search_service and (args.validate_only or args.dry_dispatch):
                raise ConfigError("--start-search-service is only available for a live dispatch")
            if args.batch_size < 1:
                raise ConfigError("--batch-size must be a positive integer")
            if args.validate_only:
                result = run_path_b_v3(
                    args.root,
                    g2_root=args.g2_root,
                    contract_path=args.contract,
                    validate_only=True,
                )
            else:
                from benchmark_harness.experiment.path_b_v3_live import (
                    build_path_b_v3_live_backend,
                )

                plans = load_path_b_v3_plans(
                    args.root,
                    g2_root=args.g2_root,
                    contract_path=args.contract,
                )
                backend, connector = build_path_b_v3_live_backend(
                    plans,
                    repo_root=Path.cwd(),
                    batch_size=args.batch_size,
                )
                if args.start_search_service:
                    if not args.search_service_config:
                        connector.close()
                        raise ConfigError(
                            "--search-service-config is required with --start-search-service"
                        )
                    lifecycle = SearchServiceLifecycle(
                        load_search_service_config(args.search_service_config)
                    )
                    lifecycle.start()
                try:
                    result = run_path_b_v3(
                        plans,
                        contract_path=args.contract,
                        backend=backend,
                        dry_dispatch=args.dry_dispatch,
                        credential_cwd=Path.cwd(),
                    )
                finally:
                    connector.close()
        elif args.command == "experiment":
            result = _experiment_command(args)
        elif args.command == "leakage-calibration":
            if args.leakage_calibration_action == "validate":
                result = validate_calibration_root(args.root)
            elif args.leakage_calibration_action == "pilot":
                result = run_pilot(
                    root=args.root,
                    cache_root=args.cache_root,
                    dataset_cache=args.dataset_cache,
                    timeout_seconds=args.timeout,
                ).as_dict()
            elif args.leakage_calibration_action == "pilot-v4":
                result = run_fresh_pilot(
                    root=args.root,
                    cache_root=args.cache_root,
                    dataset_cache=args.dataset_cache,
                    timeout_seconds=args.timeout,
                ).as_dict()
            elif args.leakage_calibration_action == "pilot-v5":
                result = run_v5_pilot(
                    root=args.root,
                    cache_root=args.cache_root,
                    dataset_cache=args.dataset_cache,
                    timeout_seconds=args.timeout,
                ).as_dict()
            elif args.leakage_calibration_action == "pilot-v6":
                result = run_v6_pilot(
                    root=args.root,
                    cache_root=args.cache_root,
                    dataset_cache=args.dataset_cache,
                    timeout_seconds=args.timeout,
                ).as_dict()
            elif args.leakage_calibration_action == "scaffold-v5":
                result = scaffold_v5(root=args.root)
            elif args.leakage_calibration_action == "scaffold-v6":
                result = scaffold_v6(root=args.root)
            elif args.leakage_calibration_action == "freeze-400":
                result = run_frozen(
                    root=args.root,
                    cache_root=args.cache_root,
                    dataset_cache=args.dataset_cache,
                    timeout_seconds=args.timeout,
                ).as_dict()
            elif args.leakage_calibration_action == "label-pilot":
                result = label_pilot(
                    root=args.root,
                    cache_root=args.cache_root,
                    reemit=args.reemit,
                )
            else:
                raise ValueError(
                    f"unsupported leakage-calibration action {args.leakage_calibration_action!r}"
                )
        else:  # argparse guarantees this branch is unreachable.
            raise AssertionError(f"Unsupported command: {args.command}")
    except (
        ConfigError,
        DataInventoryError,
        DatasetBridgeError,
        DatasetSourceError,
        ExternalHarnessError,
        RunnerError,
        SearchGatewayError,
        SearchServiceError,
        SearchServiceConfigError,
        ManifestError,
        ExecutionError,
        LiveRuntimeError,
        T3RegistryError,
        V4ApiError,
        PilotCollectionError,
        QCValidationError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print(f"benchmark-harness: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "experiment" and isinstance(result, dict):
        if result.get("status") in {"pending", "exhausted"}:
            return 1
    return 0


def _read_qc_json(path: str | Path) -> dict[str, object]:
    candidate = Path(path).expanduser().resolve()
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QCValidationError(f"unable to read QC JSON: {candidate}") from exc
    if not isinstance(value, dict):
        raise QCValidationError(f"QC JSON must contain an object: {candidate}")
    return value


def _read_json_records(path: str | Path) -> list[dict[str, Any]]:
    candidate = Path(path).expanduser().resolve()
    try:
        text = candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DataInventoryError(f"unable to read records: {candidate}") from exc
    try:
        value: Any = json.loads(text)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataInventoryError(f"invalid records JSON at line {line_number}") from exc
            if not isinstance(item, dict):
                raise DataInventoryError(f"record at line {line_number} must be an object")
            rows.append(item)
        return rows
    if isinstance(value, dict):
        value = value.get("records")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise DataInventoryError("records must be a JSON array, JSONL objects, or {records: [...]}")
    return list(value)


def _read_json_object(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise T3RegistryError(f"unable to read entry JSON: {candidate}") from exc
    if not isinstance(value, dict):
        raise T3RegistryError("entry JSON must contain an object")
    return value


def _write_json(path: str | Path, value: object) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def _experiment_command(args: argparse.Namespace) -> dict[str, object]:
    if args.experiment_action == "report":
        analysis_dir = Path(args.analysis_dir).expanduser().resolve()
        v4_markers = {
            "e1.json",
            "e2.json",
            "e3.json",
            "e4.json",
            "controls.json",
            "status.json",
        }
        if any((analysis_dir / marker).exists() for marker in v4_markers):
            return {
                "status": "ready",
                "files": sorted(path.name for path in analysis_dir.iterdir() if path.is_file()),
            }
        v2_markers = {
            "artifact_manifest.json",
            "analysis_rows.jsonl",
            "lineage.json",
            "c2.json",
            "c3.json",
            "exclusions.json",
        }
        if any((analysis_dir / marker).exists() for marker in v2_markers):
            return {"status": "ready", **report_analysis(analysis_dir).to_dict()}
        return {
            "status": "ready",
            "files": sorted(path.name for path in analysis_dir.glob("c*.json") if path.is_file()),
        }

    if args.experiment_action == "v3-lock":
        records = load_score_blind_records(args.records)
        audit = build_cohort(
            records,
            expected_record_count=args.expected_record_count,
            analysis_version=args.analysis_version,
            seed_namespace=args.seed_namespace,
        )
        paths = write_v3_artifacts(audit, args.output)
        return {
            "status": "locked",
            "t1_t2_status": audit.status,
            "b_status": audit.b_status,
            "record_count": audit.dataset_lock.record_count,
            "population_manifest_sha256": audit.dataset_lock.population_manifest_sha256,
            "design_lock_sha256": audit.design_lock.design_lock_sha256,
            "artifacts": {name: str(path) for name, path in sorted(paths.items())},
        }

    if args.experiment_action == "v3-validate":
        return validate_v3_artifacts(args.root)

    if args.experiment_action == "v4-inventory":
        inventory = compute_inventory(_read_json_records(args.records), as_of=args.as_of)
        output = _write_json(args.output, inventory) if args.output else None
        result: dict[str, object] = {"status": "inventoried", **inventory}
        if output is not None:
            result["output"] = str(output)
        return result

    if args.experiment_action == "v4-ledger":
        ledger = build_disposition_ledger(_read_json_records(args.records))
        output = write_collection_ledger(args.output, ledger)
        return {
            "status": "ledger_written",
            "path": str(output),
            "manifest": str(output.with_name(output.name + ".manifest.json")),
            **ledger.to_dict(),
        }

    if args.experiment_action in {"t3-schedule", "t3-lock", "t3-settle"}:
        payload = _read_json_object(args.entry)
        if args.experiment_action == "t3-schedule":
            digest = append_schedule(args.registry, T3ScheduleEntry.from_dict(payload))
        elif args.experiment_action == "t3-lock":
            digest = append_lock(args.registry, T3LockEntry.from_dict(payload))
        else:
            digest = append_settlement(
                args.registry,
                T3SettlementEntry.from_dict(payload),
                outcome_text_sha256=args.outcome_text_sha256,
            )
        registry = load_t3_registry(args.registry)
        return {
            "status": "appended",
            "entry_sha256": digest,
            "registry_sha256": registry.registry_sha256,
            "schedule_count": len(registry.schedules),
            "lock_count": len(registry.locks),
            "settlement_count": len(registry.settlements),
        }

    if args.experiment_action == "t3-validate":
        registry = validate_t3_registry(args.registry)
        return {
            "status": "valid",
            "registry_sha256": registry.registry_sha256,
            "schedule_count": len(registry.schedules),
            "lock_count": len(registry.locks),
            "settlement_count": len(registry.settlements),
        }

    if args.experiment_action == "v4-audit":
        return {"status": "valid", **audit_full_release(args.root).to_dict()}

    manifest = load_manifest(args.manifest)
    if args.experiment_action == "validate":
        artifacts = validate_manifest_artifacts(manifest)
        return {
            "status": "valid",
            "manifest_hash": manifest.manifest_hash,
            "experiment_id": manifest.experiment_id,
            "systems": [system.system_id for system in manifest.systems],
            "retrieval_regimes": list(manifest.retrieval_regimes),
            "trial_count": manifest.trial_count,
            "artifacts": artifacts,
        }
    if args.experiment_action == "plan":
        task_ids = list(args.task_id)
        if args.task_file:
            task_path = Path(args.task_file).expanduser().resolve()
            text = task_path.read_text(encoding="utf-8")
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                value = [json.loads(line)["task_id"] for line in text.splitlines() if line.strip()]
            if isinstance(value, list):
                task_ids.extend(
                    item.get("task_id") if isinstance(item, dict) else item for item in value
                )
            else:
                raise ManifestError("--task-file must contain a JSON array or JSONL objects.")
        plan = build_plan(
            manifest,
            task_ids or None,
            split_file=args.split_file,
            selection=args.selection,
        )
        write_plan(args.output, plan)
        return {"status": "planned", "plan_id": plan.plan_id, "plan_hash": plan.plan_hash}
    if args.experiment_action in {"run", "resume"}:
        request = ExperimentExecutionRequest(
            manifest=manifest,
            plan=load_plan(args.plan, expected_manifest_hash=manifest.manifest_hash),
            results_root=args.results_root,
            selection=args.selection,
            max_workers=args.max_workers,
            max_attempts=args.max_attempts,
        )
        dependencies = _live_execution_dependencies(manifest)
        summary = (
            run_experiment(request, dependencies=dependencies)
            if args.experiment_action == "run"
            else resume_experiment(request, dependencies=dependencies)
        )
        return summary.as_dict()
    if args.experiment_action == "audit":
        if not hasattr(args, "plan"):
            raise ManifestError("audit requires --plan")
        if manifest.schema_version == "experiment/v4":
            return {"status": "valid", **audit_full_release(args.results_root).to_dict()}
        plan = load_plan(args.plan, expected_manifest_hash=manifest.manifest_hash)
        report = audit_experiment(
            ExperimentExecutionRequest(
                manifest=manifest,
                plan=plan,
                results_root=args.results_root,
                selection="all",
            )
        )
        return report.to_dict()
    if args.experiment_action == "analyze":
        if manifest.schema_version == "experiment/v4":
            artifacts = analyze_full_release(
                manifest,
                load_plan(args.plan, expected_manifest_hash=manifest.manifest_hash),
                args.input_root,
                args.output,
            )
            return {"status": "analyzed", **artifacts.to_dict()}
        artifacts = analyze_experiment(
            AnalysisRequest(
                manifest=manifest,
                plan=load_plan(args.plan, expected_manifest_hash=manifest.manifest_hash),
                input_root=args.input_root,
                output_dir=args.output,
                selection=args.selection,
                analysis_as_of=args.analysis_as_of,
                metric_name=args.metric,
            )
        )
        return {"status": "analyzed", **artifacts.to_dict()}
    raise ManifestError(f"unsupported experiment action {args.experiment_action!r}")


def _live_execution_dependencies(manifest: Any) -> Any:
    """Build the identity-checked live dependency set for experiment run/resume.

    Any missing OpenAI/search/source input raises ``LiveRuntimeError`` before
    cell 1; the CLI reports it and exits non-zero instead of falling back to
    a mock or an alias.
    """

    from benchmark_harness.experiment.live_model import (
        build_live_search_session_factory,
        resolve_live_model_profile,
        validate_live_runtime,
    )

    # Enforce the snapshot/wire identity, API key, and endpoint before any
    # other runtime check so operators see the model shortfall first.
    resolve_live_model_profile(manifest, cell_id="preflight")

    def factory(system: Any, regime: str, search_handler: Any = None) -> Any:
        return real_system_factory(system, regime, search_handler=search_handler)

    return ExecutionDependencies(
        system_factory=factory,
        model_profile_factory=lambda cell_id: resolve_live_model_profile(manifest, cell_id=cell_id),
        search_session_factory=build_live_search_session_factory(manifest),
        live_validation=lambda cells: validate_live_runtime(manifest, tuple(cells)),
    )


def _load_service_dotenv() -> None:
    """Load only service-owned provider settings from an optional local .env."""

    values = dotenv_values(Path(".env"))
    allowed_prefixes = ("VOLC_", "SEARCH_FILTER_MODEL_")
    for name, value in values.items():
        if (
            isinstance(name, str)
            and isinstance(value, str)
            and name.startswith(allowed_prefixes)
            and name not in os.environ
        ):
            os.environ[name] = value


def _load_experiment_dotenv() -> None:
    """Load only non-secret experiment settings from the local operator env."""

    values = dotenv_values(Path(".env"))
    allowed_prefixes = (
        "BENCHMARK_MODEL_",
        "ALIYUN_",
        "DASHSCOPE_",
        "BAILIAN_",
    )
    allowed_names = {
        "BENCHMARK_FINANCE_SOURCE",
        "BENCHMARK_ODR_SOURCE",
        "BENCHMARK_GPT_SOURCE",
        "BENCHMARK_SEARCH_SERVICE_URL",
        "BENCHMARK_SEARCH_SERVICE_ADMIN_TOKEN_FILE",
        "BENCHMARK_V3_UNRESTRICTED_CREDENTIALS_FILE",
    }
    for name, value in values.items():
        if (
            isinstance(name, str)
            and isinstance(value, str)
            and (name.startswith(allowed_prefixes) or name in allowed_names)
            and name not in os.environ
        ):
            os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
