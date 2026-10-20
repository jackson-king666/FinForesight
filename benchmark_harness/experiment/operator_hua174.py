"""HUA-174 model/profile and adapter binding helpers.

This module is intentionally execution-side: it reads one selected model env
file, constructs the existing typed adapter, and returns safe evidence.  It
never serializes credentials or passes the dataset record across an adapter
boundary.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from benchmark_harness.config import (
    FINANCEHARNESS_LICENSE_STATUS,
    ExternalHarnessConfig,
    ResolvedModelProfile,
    sanitized_endpoint,
)
from benchmark_harness.contracts import AdapterResult, RunContext, TaskInput
from benchmark_harness.experiment.hua174 import (
    HUA174_BUDGET,
    HUA174_EXPERIMENT_AS_OF,
    HUA174_MODEL_BINDINGS,
    HUA174_MODEL_KEY_ALIASES,
    HUA174_SEARCH_POLICY,
    HUA174_SEARCH_PROFILE,
    HUA174_SEARCH_PROTOCOL,
    Hua174Cell,
    Hua174Plan,
    Hua174Task,
    safe_hua174_receipt,
    sha256_text,
)
from benchmark_harness.external_harnesses.checkout import ExternalHarnessManager
from benchmark_harness.search_service import SearchServiceClient
from benchmark_harness.search_service.api import SearchReceiptRef, SearchServiceError
from benchmark_harness.search_service.client import SearchServiceSession
from benchmark_harness.systems.adapters.financeharness import (
    FinanceHarnessSystem,
    build_financeharness_descriptor,
)
from benchmark_harness.systems.adapters.gpt_researcher import (
    GptResearcherAdapter,
    build_gpt_researcher_descriptor,
)
from benchmark_harness.systems.adapters.odr import OdrAdapter, build_odr_descriptor
from benchmark_harness.systems.contracts import (
    BudgetSpec,
    ExperimentBinding,
    SystemDescriptor,
    TaskCapability,
)
from benchmark_harness.systems.regimes import build_retrieval_policy

HUA174_MODEL_ENV_FILES = {key: value["env_file"] for key, value in HUA174_MODEL_BINDINGS.items()}
HUA174_FRAMEWORK_ENV_KEYS = {
    "financeharness": "BENCHMARK_FINANCE_SOURCE",
    "odr": "BENCHMARK_ODR_SOURCE",
    "gpt-researcher": "BENCHMARK_GPT_SOURCE",
}
HUA174_ADAPTER_RETRIEVAL_REGIME = "cutoff_controlled"
# Search within the D/E cutoff era so Volc returns dated evidence; the future
# marker in historical result content then exercises the real LLM filter.
HUA174_FILTER_PROBE_QUERY = "historical stock market news March 2024"
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Hua174OperatorError(RuntimeError):
    """Raised when a HUA-174 runtime binding is incomplete or unsafe."""


def _declared_service_cohort(search_client: SearchServiceClient) -> str:
    value = getattr(search_client, "expected_service_cohort_sha256", None)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Hua174OperatorError("expected_service_cohort_required")
    return value


def _normalize_model_key(model_key: str) -> str:
    normalized = HUA174_MODEL_KEY_ALIASES.get(model_key, model_key)
    if normalized not in HUA174_MODEL_BINDINGS:
        raise Hua174OperatorError(f"unsupported HUA-174 model key: {model_key}")
    return normalized


def load_hua174_env(path: str | Path) -> dict[str, str]:
    """Parse a model env file without mutating process environment."""

    source = Path(path).expanduser().resolve()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise Hua174OperatorError("model_env_file_unavailable") from exc
    values: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            continue
        name, raw_value = stripped.split("=", 1)
        name = name.strip()
        if not _ENV_NAME_RE.fullmatch(name):
            raise Hua174OperatorError("model_env_name_invalid")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[name] = value
    return values


@dataclass(frozen=True, slots=True)
class Hua174ModelConfig:
    model_key: str
    endpoint: str
    model_name: str
    api_key: str = field(repr=False)
    env_file: str
    backbone_id: str
    wire_model_name: str
    snapshot_model_name: str
    timeout_seconds: float = 600.0
    retries: int = 2

    def public_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "endpoint": sanitized_endpoint(self.endpoint),
            "model_name": self.model_name,
            "wire_model_name": self.wire_model_name,
            "snapshot_model_name": self.snapshot_model_name,
            "backbone_id": self.backbone_id,
            "env_file": self.env_file,
            "temperature": 0.2,
            "top_p": 1.0,
            "max_tokens": 32768,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
        }


def resolve_hua174_model_config(
    model_key: str,
    values: Mapping[str, str],
    *,
    env_file: str | Path | None = None,
) -> Hua174ModelConfig:
    """Resolve exactly one model; there is no cross-model credential fallback."""

    normalized = _normalize_model_key(model_key)
    binding = HUA174_MODEL_BINDINGS[normalized]
    required = (
        "BENCHMARK_MODEL_ENDPOINT",
        "BENCHMARK_MODEL_NAME",
        "BENCHMARK_MODEL_API_KEY",
    )
    missing = [name for name in required if not str(values.get(name, "")).strip()]
    if missing:
        raise Hua174OperatorError("missing " + ", ".join(missing))
    endpoint = str(values["BENCHMARK_MODEL_ENDPOINT"]).strip()
    model_name = str(values["BENCHMARK_MODEL_NAME"]).strip()
    api_key = str(values["BENCHMARK_MODEL_API_KEY"]).strip()
    if not endpoint.startswith(("http://", "https://")):
        raise Hua174OperatorError("BENCHMARK_MODEL_ENDPOINT_invalid")
    configured_retries = str(values.get("BENCHMARK_MODEL_RETRIES", "2")).strip()
    if configured_retries != "2":
        raise Hua174OperatorError("BENCHMARK_MODEL_RETRIES_must_be_2")
    if model_name != binding["wire_model_name"]:
        raise Hua174OperatorError("model_identity_mismatch")
    return Hua174ModelConfig(
        model_key=normalized,
        endpoint=endpoint,
        model_name=model_name,
        api_key=api_key,
        env_file=str(env_file) if env_file is not None else binding["env_file"],
        backbone_id=binding["backbone_id"],
        wire_model_name=binding["wire_model_name"],
        snapshot_model_name=binding["snapshot_model_name"],
    )


def build_hua174_model_profile(config: Hua174ModelConfig) -> ResolvedModelProfile:
    return ResolvedModelProfile(
        mode="openai_compatible",
        endpoint=config.endpoint,
        model_name=config.model_name,
        api_key=config.api_key,
        temperature=0.2,
        max_tokens=32768,
        timeout_seconds=600.0,
        retries=2,
        seed=None,
        top_p=1.0,
        snapshot_model_name=config.snapshot_model_name,
        backbone_id=config.backbone_id,
        wire_name_binding=config.wire_model_name,
    )


def model_env_path(model_key: str, env_root: str | Path) -> Path:
    normalized = _normalize_model_key(model_key)
    return Path(env_root).expanduser().resolve() / HUA174_MODEL_ENV_FILES[normalized]


def load_hua174_model_config(model_key: str, env_root: str | Path) -> Hua174ModelConfig:
    path = model_env_path(model_key, env_root)
    return resolve_hua174_model_config(model_key, load_hua174_env(path), env_file=path)


def load_hua174_model_configs(env_root: str | Path) -> dict[str, Hua174ModelConfig]:
    return {key: load_hua174_model_config(key, env_root) for key in HUA174_MODEL_BINDINGS}


def source_paths_from_values(values: Mapping[str, str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for framework_id, env_name in HUA174_FRAMEWORK_ENV_KEYS.items():
        value = str(values.get(env_name, "")).strip()
        if not value:
            raise Hua174OperatorError(f"missing {env_name}")
        paths[framework_id] = Path(value).expanduser().resolve()
    return paths


def _training_cutoff(task: Hua174Task) -> date:
    value = task.record.get("backbone_training_cutoff")
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    # The pinned rows declare 2024-06-30.  A fallback here is a validation
    # default only; the source loader retains the row hash and record.
    return date(2024, 6, 30)


def build_hua174_descriptor(
    cell: Hua174Cell, config: Hua174ModelConfig, task: Hua174Task
) -> SystemDescriptor:
    common = {
        "system_id": f"{cell.model_key}-{cell.framework_id}-s2",
        "backbone_id": config.backbone_id,
        "backbone_training_cutoff": _training_cutoff(task),
        "cutoff_source": task.source_pointer,
        "retrieval_regime": HUA174_ADAPTER_RETRIEVAL_REGIME,
    }
    if cell.framework_id == "financeharness":
        return build_financeharness_descriptor(**common)
    if cell.framework_id == "odr":
        return build_odr_descriptor(
            **common,
            framework_id="odr",
            strategy_id="odr-deep-research",
        )
    if cell.framework_id == "gpt-researcher":
        return build_gpt_researcher_descriptor(
            **common,
            framework_id="gpt-researcher",
            strategy_id="gpt-researcher-deep-research",
        )
    raise Hua174OperatorError(f"unsupported HUA-174 framework: {cell.framework_id}")


def build_hua174_budget() -> BudgetSpec:
    return BudgetSpec(**HUA174_BUDGET)


def build_hua174_binding(
    plan: Hua174Plan,
    cell: Hua174Cell,
    descriptor: SystemDescriptor,
    *,
    attempt_id: str,
) -> ExperimentBinding:
    return ExperimentBinding(
        experiment_id=plan.experiment_id,
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        cell_id=cell.cell_id,
        attempt_id=attempt_id,
        descriptor_hash=descriptor.descriptor_hash,
        task_id=cell.task_id,
        retrieval_regime=HUA174_ADAPTER_RETRIEVAL_REGIME,
        trial=cell.trial,
        budget=build_hua174_budget(),
        experiment_as_of=HUA174_EXPERIMENT_AS_OF,
        calc_policy="calc_on",
    )


def _source_for_framework(source_paths: Mapping[str, str | Path], framework_id: str) -> Path:
    try:
        source = Path(source_paths[framework_id]).expanduser().resolve()
    except KeyError as exc:
        raise Hua174OperatorError(f"source path missing for {framework_id}") from exc
    if not source.exists():
        raise Hua174OperatorError(f"source checkout unavailable for {framework_id}")
    return source


def _finance_provenance_root(source: Path, repo_root: str | Path | None) -> Path:
    """Use the root recorded by the external checkout manifest for verification."""

    candidate = source.parents[2] if len(source.parents) > 2 else Path(repo_root or Path.cwd())
    if (candidate / "data" / "external-harnesses").is_dir():
        return candidate
    return Path(repo_root or Path.cwd()).expanduser().resolve()


def build_hua174_adapter(
    cell: Hua174Cell,
    config: Hua174ModelConfig,
    task: Hua174Task,
    *,
    descriptor: SystemDescriptor,
    source_paths: Mapping[str, str | Path],
    search_service: SearchServiceClient | None = None,
    search_handler: Callable[[str, str, int], Mapping[str, Any] | list[Mapping[str, Any]]]
    | None = None,
    search_token: str | None = None,
    model_responder: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    worker_exchange: Callable[..., Any] | None = None,
    repo_root: str | Path | None = None,
) -> Any:
    del task
    source = _source_for_framework(source_paths, cell.framework_id)
    policy = build_retrieval_policy(HUA174_ADAPTER_RETRIEVAL_REGIME)
    if cell.framework_id == "financeharness":
        manager = ExternalHarnessManager(
            ExternalHarnessConfig(
                checkout_root=source.parent,
                offline=True,
                license_status=FINANCEHARNESS_LICENSE_STATUS,
            ),
            repo_root=_finance_provenance_root(source, repo_root),
        )
        return FinanceHarnessSystem(
            descriptor=descriptor,
            source_dir=source,
            repo_root=repo_root,
            manager=manager,
            search_service=search_service,
            model_responder=model_responder,
        )
    if cell.framework_id == "odr":
        return OdrAdapter(
            descriptor=descriptor,
            retrieval_policy=policy,
            source_dir=source,
            search_handler=search_handler,
            search_token=search_token,
            model_responder=model_responder,
            worker_exchange=worker_exchange,
            repo_root=repo_root,
        )
    if cell.framework_id == "gpt-researcher":
        return GptResearcherAdapter(
            descriptor=descriptor,
            retrieval_policy=policy,
            source_dir=source,
            search_handler=search_handler,
            search_token=search_token,
            model_responder=model_responder,
            worker_exchange=worker_exchange,
            repo_root=repo_root,
        )
    raise Hua174OperatorError(f"unsupported HUA-174 framework: {cell.framework_id}")


def _principal(cell: Hua174Cell) -> str:
    return cell.framework_id


def _public_filter_verdict(value: Any) -> str | None:
    if value in {"KEEP", "keep", "accept"}:
        return "KEEP"
    if value in {"DROP", "drop", "reject", "leak"}:
        return "DROP"
    return None


def _outcome_evidence(session: SearchServiceSession | None) -> dict[str, Any]:
    if session is None:
        return {}
    verdicts: list[str] = []
    receipt_hashes: list[str] = []
    filter_model_calls = 0
    for outcome in session.outcomes:
        filter_model_calls += outcome.filter_model_call_count
        receipt_hashes.append(outcome.receipt.receipt_sha256)
        for audit in outcome.audit_records:
            verdict = _public_filter_verdict(getattr(audit, "llm_verdict", None))
            if verdict is not None:
                verdicts.append(verdict)
    evidence: dict[str, Any] = {
        "protocol": HUA174_SEARCH_PROTOCOL,
        "framework_id": session.framework,
        "search_call_count": len(session.outcomes),
        "filter_review_count": sum(outcome.filter_review_count for outcome in session.outcomes),
        "filter_model_call_count": filter_model_calls,
        "llm_verdicts": verdicts,
        "receipt_sha256s": receipt_hashes,
    }
    if session.close_result is not None:
        close = session.close_result
        manifest = close.get("service_manifest", {})
        if isinstance(manifest, Mapping):
            for key in (
                "api_contract_sha256",
                "implementation_manifest_sha256",
                "capability_sha256",
                "service_cohort_sha256",
            ):
                if isinstance(manifest.get(key), str):
                    evidence[key] = manifest[key]
        evidence.update(
            safe_hua174_receipt(
                {
                    "status": close.get("status"),
                    "comparable": close.get("comparable"),
                    "receipt_count": close.get("receipt_count"),
                    "receipt_chain_head": close.get("receipt_chain_head"),
                    "service_manifest": close.get("service_manifest", {}),
                }
            )
        )
    return evidence


def run_hua174_filter_probe(
    search_client: SearchServiceClient,
    cell: Hua174Cell,
    task: Hua174Task,
    *,
    attempt_id: str,
    expected_service_cohort_sha256: str | None = None,
) -> dict[str, Any]:
    """Exercise the real filter path with a bounded, hash-only operator probe.

    A model trajectory is allowed to answer without invoking its search tool.
    That is useful model behavior to record, but it cannot prove that the
    resident service executed its filter-model branch.  The smoke gate uses
    this explicit probe to verify the service contract without relabeling the
    model's own search count; the returned evidence keeps the two counts
    separate.
    """

    search_client.ready()
    session = search_client.open_session(
        run_id=f"{attempt_id}-filter-probe",
        task_id=task.task_id,
        profile=HUA174_SEARCH_PROFILE,
        baseline=_principal(cell),
        cutoff_date=task.cutoff_date.isoformat(),
        policy_version=HUA174_SEARCH_POLICY,
        expected_service_cohort_sha256=(
            expected_service_cohort_sha256
            if expected_service_cohort_sha256 is not None
            else _declared_service_cohort(search_client)
        ),
    )
    try:
        session.search(HUA174_FILTER_PROBE_QUERY)
        session.close(status="completed")
        evidence = _outcome_evidence(session)
        evidence.update(
            {
                "filter_probe": True,
                "filter_probe_query_sha256": sha256_text(HUA174_FILTER_PROBE_QUERY),
                "agent_search_call_count": 0,
                "probe_search_call_count": len(session.outcomes),
            }
        )
        return evidence
    finally:
        if session.close_result is None:
            session.close(status="aborted")


def _finance_evidence(adapter: Any) -> dict[str, Any]:
    output = getattr(getattr(adapter, "_adapter", None), "last_output", None)
    if output is None:
        return {}
    outcomes = tuple(getattr(output, "search_outcomes", ()))
    close = getattr(output, "service_close", None) or {}
    verdicts: list[str] = []
    for outcome in outcomes:
        for audit in getattr(outcome, "audit_records", ()):
            verdict = _public_filter_verdict(getattr(audit, "llm_verdict", None))
            if verdict is not None:
                verdicts.append(verdict)
    close_audits = close.get("search_audit", ())
    if not verdicts and isinstance(close_audits, list):
        for audit in close_audits:
            if isinstance(audit, Mapping):
                verdict = _public_filter_verdict(audit.get("llm_verdict"))
                if verdict is not None:
                    verdicts.append(verdict)
    receipt_hashes = [
        receipt.receipt_sha256
        for receipt in getattr(output, "service_receipts", ())
        if isinstance(receipt, SearchReceiptRef)
    ]
    if not receipt_hashes:
        close_receipts = close.get("receipts", ())
        if isinstance(close_receipts, list):
            receipt_hashes = [
                receipt["receipt_sha256"]
                for receipt in close_receipts
                if isinstance(receipt, Mapping) and isinstance(receipt.get("receipt_sha256"), str)
            ]
    close_receipt_count = close.get("receipt_count")
    search_call_count = (
        close_receipt_count
        if isinstance(close_receipt_count, int) and not isinstance(close_receipt_count, bool)
        else len(outcomes)
    )
    evidence: dict[str, Any] = {
        "protocol": HUA174_SEARCH_PROTOCOL,
        "framework_id": "financeharness",
        "search_call_count": search_call_count,
        "filter_review_count": sum(
            getattr(outcome, "filter_review_count", 0) for outcome in outcomes
        ),
        "filter_model_call_count": int(getattr(output, "filter_model_call_count", 0)),
        "llm_verdicts": verdicts,
        "receipt_sha256s": receipt_hashes,
        "receipt_count": close.get("receipt_count"),
        "comparable": close.get("comparable"),
    }
    manifest = close.get("service_manifest", {})
    if isinstance(manifest, Mapping):
        for key in (
            "api_contract_sha256",
            "implementation_manifest_sha256",
            "capability_sha256",
            "service_cohort_sha256",
        ):
            if isinstance(manifest.get(key), str):
                evidence[key] = manifest[key]
    return evidence


@dataclass(frozen=True, slots=True)
class Hua174CellRun:
    cell: Hua174Cell
    task: Hua174Task
    descriptor: SystemDescriptor
    binding: ExperimentBinding
    result: AdapterResult
    service_evidence: Mapping[str, Any]


def run_hua174_cell(
    plan: Hua174Plan,
    cell: Hua174Cell,
    config: Hua174ModelConfig,
    *,
    source_paths: Mapping[str, str | Path],
    search_client: SearchServiceClient | None,
    attempt_id: str | None = None,
    model_responder: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    worker_exchange: Callable[..., Any] | None = None,
    repo_root: str | Path | None = None,
    adapter_factory: Callable[..., Any] | None = None,
    expected_service_cohort_sha256: str | None = None,
) -> Hua174CellRun:
    task_by_id = {task.task_id: task for task in plan.tasks}
    try:
        task = task_by_id[cell.task_id]
    except KeyError as exc:
        raise Hua174OperatorError("cell_task_not_in_plan") from exc
    profile = build_hua174_model_profile(config)
    descriptor = build_hua174_descriptor(cell, config, task)
    binding = build_hua174_binding(
        plan,
        cell,
        descriptor,
        attempt_id=attempt_id or secrets.token_hex(12),
    )
    session: SearchServiceSession | None = None
    adapter: Any | None = None
    try:
        search_handler = None
        search_token = None
        if cell.framework_id != "financeharness":
            if search_client is None:
                raise Hua174OperatorError("search_service_required")
            search_client.ready()
            session = search_client.open_session(
                run_id=binding.attempt_id,
                task_id=task.task_id,
                profile=HUA174_SEARCH_PROFILE,
                baseline=_principal(cell),
                cutoff_date=task.cutoff_date.isoformat(),
                policy_version=HUA174_SEARCH_POLICY,
                expected_service_cohort_sha256=(
                    expected_service_cohort_sha256
                    if expected_service_cohort_sha256 is not None
                    else _declared_service_cohort(search_client)
                ),
            )
            search_token = session.descriptor.task_token

            def search_handler(query: str, _token: str, _call_index: int):
                if session is None:
                    raise SearchServiceError("service_session_unavailable", status_code=503)
                if _token != session.descriptor.task_token:
                    raise SearchServiceError("task_token_invalid", status_code=401)
                if _call_index != len(session.outcomes) + 1:
                    raise SearchServiceError("call_index_out_of_order", status_code=409)
                outcome = session.search(query)
                return [candidate.tool_payload() for candidate in outcome.candidates]

        if adapter_factory is not None:
            adapter = adapter_factory(
                cell=cell,
                config=config,
                task=task,
                descriptor=descriptor,
                search_service=search_client,
                search_handler=search_handler,
                search_token=search_token,
            )
        else:
            adapter = build_hua174_adapter(
                cell,
                config,
                task,
                descriptor=descriptor,
                source_paths=source_paths,
                search_service=search_client,
                search_handler=search_handler,
                search_token=search_token,
                model_responder=model_responder,
                worker_exchange=worker_exchange,
                repo_root=repo_root,
            )
        preflight = adapter.preflight(binding)
        if not preflight.ready:
            raise Hua174OperatorError("adapter_preflight_failed")
        context = RunContext(
            run_id=binding.attempt_id,
            framework=cell.framework_id,
            adapter_version=descriptor.adapter_revision,
            model=profile,
            trace_content_mode="hash_only",
        ).for_task(task.task_id)
        capability = TaskCapability(
            service_url=search_client.base_url if search_client is not None else None,
            service_token=search_token,
            model_broker_capability="host-model-broker/v1",
            budget=binding.budget,
            allow_calculation=True,
            allow_citations=True,
            require_structured_output=True,
            run_id=binding.attempt_id,
            trial_id=cell.cell_id,
            trace_mode="hash_only",
        )
        result = adapter.run_task(
            TaskInput(
                task_id=task.task_id,
                prompt=task.prompt,
                cutoff_date=task.cutoff_date.isoformat(),
            ),
            context,
            capability,
            binding,
        )
        if not isinstance(result, AdapterResult) or not isinstance(result.raw_text, str):
            raise Hua174OperatorError("adapter_result_missing_raw_text")
        if not result.raw_text.strip():
            raise Hua174OperatorError("adapter_result_empty_raw_text")
        if session is not None and session.close_result is None:
            session.close(status="completed")
        if cell.framework_id == "financeharness":
            evidence = _finance_evidence(adapter)
        else:
            evidence = _outcome_evidence(session)
        return Hua174CellRun(
            cell=cell,
            task=task,
            descriptor=descriptor,
            binding=binding,
            result=result,
            service_evidence=evidence,
        )
    finally:
        if session is not None and session.close_result is None:
            try:
                session.close(status="completed" if adapter is not None else "aborted")
            except Exception:
                pass
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                pass


__all__ = [
    "HUA174_FRAMEWORK_ENV_KEYS",
    "HUA174_ADAPTER_RETRIEVAL_REGIME",
    "HUA174_FILTER_PROBE_QUERY",
    "HUA174_MODEL_ENV_FILES",
    "Hua174CellRun",
    "Hua174ModelConfig",
    "Hua174OperatorError",
    "build_hua174_adapter",
    "build_hua174_binding",
    "build_hua174_budget",
    "build_hua174_descriptor",
    "build_hua174_model_profile",
    "load_hua174_env",
    "load_hua174_model_config",
    "load_hua174_model_configs",
    "model_env_path",
    "resolve_hua174_model_config",
    "run_hua174_filter_probe",
    "run_hua174_cell",
    "source_paths_from_values",
]
