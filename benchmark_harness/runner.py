"""Isolated benchmark execution for strict and cutoff-bound web-search profiles."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

from benchmark_harness.adapters import (
    DirectTransportAdapter,
    LangGraphAdapter,
    MappingMockTransport,
    PydanticAIAdapter,
)
from benchmark_harness.artifacts import ArtifactStore, file_sha256, make_run_id
from benchmark_harness.config import ConfigError, ExperimentConfig, ResolvedModelProfile
from benchmark_harness.contracts import (
    Adapter,
    AdapterResult,
    AgenticAdapter,
    NetworkEvent,
    RunContext,
    TaskInput,
    sha256_json,
    sha256_text,
)
from benchmark_harness.dataset_sources import (
    DatasetSource,
    ExternalCheckoutDatasetSource,
    HuggingFaceDatasetSource,
)
from benchmark_harness.external_harnesses.checkout import (
    ExternalHarnessError,
    ExternalHarnessManager,
    manager_from_config,
)
from benchmark_harness.external_harnesses.financeharness.adapter import (
    FinanceHarnessAdapter,
    FinanceHarnessBoundaryError,
)
from benchmark_harness.external_harnesses.financeharness.policy import DeepResearchPolicyGuard
from benchmark_harness.profile_registry import (
    ProfileRegistryError,
    get_profile,
    normalize_baseline,
)
from benchmark_harness.reference_evaluation import BundledEvaluationEngine, EvaluationEngine
from benchmark_harness.reporting.compare import write_comparison
from benchmark_harness.search import (
    DEFAULT_TOOL_CONTRACT,
    AgenticWebSearchPolicyGuard,
    BEventDateError,
    CutoffSearchGateway,
    SearchGatewayError,
    validate_b_cutoff_not_after_event,
)
from benchmark_harness.search_service import (
    SearchServiceClient,
    SearchServiceClientConfig,
    SearchServiceError,
)
from benchmark_harness.trace.jsonl_sink import JsonlSink, redact
from benchmark_harness.trace.policy_guard import PolicyDecision, StrictPolicyGuard, violation_counts


class RunnerError(RuntimeError):
    """Raised when a run cannot safely produce a comparable artifact set."""


@dataclass(frozen=True, slots=True)
class BaselineOutcome:
    baseline: str
    output_dir: Path
    task_count: int
    comparable_task_count: int
    mock_non_performance: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "output_dir": str(self.output_dir),
            "task_count": self.task_count,
            "comparable_task_count": self.comparable_task_count,
            "mock_non_performance": self.mock_non_performance,
        }


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    """One completed-task progress snapshot emitted without task content."""

    baseline: str
    completed: int
    total: int
    task_id: str
    category: str | None
    elapsed_seconds: float
    eta_seconds: float


def _stderr_progress(update: ProgressUpdate) -> None:
    """Keep CLI JSON on stdout while providing live human-readable status."""

    category = update.category or "unknown"
    print(
        "[progress] "
        f"baseline={update.baseline} task={update.completed}/{update.total} "
        f"task_id={update.task_id} category={category} "
        f"elapsed={update.elapsed_seconds:.1f}s eta={update.eta_seconds:.1f}s",
        file=sys.stderr,
        flush=True,
    )


class _ProgressReporter:
    """Derive ETA from a small rolling mean of actual completed task durations."""

    def __init__(
        self,
        *,
        baseline: str,
        total: int,
        callback: Callable[[ProgressUpdate], None] | None,
    ) -> None:
        self._baseline = baseline
        self._total = total
        self._callback = callback or _stderr_progress
        self._started = time.perf_counter()
        self._latencies: deque[float] = deque(maxlen=10)
        self._completed = 0

    def complete(self, *, task_id: str, category: str | None, started: float) -> None:
        self._completed += 1
        self._latencies.append(max(0.0, time.perf_counter() - started))
        rolling_mean = sum(self._latencies) / len(self._latencies)
        update = ProgressUpdate(
            baseline=self._baseline,
            completed=self._completed,
            total=self._total,
            task_id=task_id,
            category=category,
            elapsed_seconds=time.perf_counter() - self._started,
            eta_seconds=rolling_mean * max(0, self._total - self._completed),
        )
        try:
            self._callback(update)
        except Exception:
            # A display hook must never make a benchmark task fail.
            return


def _hash_text_values(value: Any) -> Any:
    """Preserve result shape while removing arbitrary text from hash-only artifacts."""

    if isinstance(value, str):
        return {"sha256": sha256_text(value)}
    if isinstance(value, dict):
        return {str(key): _hash_text_values(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_hash_text_values(item) for item in value]
    return value


def _prediction_artifact(evaluation_row: dict[str, Any], *, content_mode: str) -> dict[str, Any]:
    """Make a persisted row without weakening in-memory frozen evaluation."""

    artifact = dict(redact(evaluation_row))
    if content_mode != "hash_only":
        return artifact

    for key in ("prompt", "raw_response"):
        value = evaluation_row.get(key)
        artifact[key] = None
        artifact[f"{key}_sha256"] = sha256_text(value) if isinstance(value, str) else None
    artifact["parsed"] = _hash_text_values(evaluation_row.get("parsed"))
    artifact["metrics"] = _hash_text_values(evaluation_row.get("metrics"))
    if evaluation_row.get("api_error"):
        artifact["api_error"] = "error_redacted"
    return artifact


def _make_dataset_source(config: ExperimentConfig) -> DatasetSource:
    if config.dataset.source == "huggingface":
        return HuggingFaceDatasetSource(
            repo_id=config.dataset.repo_id,
            revision=config.dataset.revision,
            cache_dir=config.dataset.cache_dir,
            offline=config.dataset.offline,
            config_map=config.dataset.config_map,
            expected_counts=config.dataset.expected_counts,
        )
    if config.dataset.source == "external_checkout":
        if config.dataset.root is None:
            raise RunnerError("external_checkout requires an explicit dataset.root.")
        return ExternalCheckoutDatasetSource(config.dataset.root, revision=config.dataset.revision)
    raise RunnerError(f"Unsupported dataset source {config.dataset.source!r}.")


def _evaluator_manifest(config: ExperimentConfig) -> tuple[EvaluationEngine, str]:
    """Verify the bundled evaluator identity before it can produce a score."""

    if not config.evaluation.require_golden_parity:
        raise RunnerError("Golden evaluator parity cannot be disabled for a v0.3 score.")
    path = config.evaluation.manifest_path
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"Frozen evaluator manifest is unavailable: {path}") from exc
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("commit") != config.evaluation.source_commit:
        raise RunnerError(
            "Frozen evaluator manifest does not match the configured authoritative source commit."
        )
    frozen = manifest.get("frozen_file")
    if not isinstance(frozen, dict) or frozen.get("path") != "engine.py":
        raise RunnerError("Frozen evaluator manifest does not describe engine.py.")
    engine_path = path.parent / "engine.py"
    if not engine_path.is_file() or file_sha256(engine_path) != frozen.get("sha256"):
        raise RunnerError("Frozen evaluator engine hash does not match EVALUATOR_MANIFEST.json.")
    return BundledEvaluationEngine(), file_sha256(path)


def _make_search_gateway(config: ExperimentConfig) -> CutoffSearchGateway:
    return CutoffSearchGateway(
        access_key_env=config.search.access_key_env,
        secret_key_env=config.search.secret_key_env,
        lookback_years=config.search.lookback_years,
        test_only=config.search.test_only,
    )


def _make_search_service_client(config: ExperimentConfig) -> SearchServiceClient:
    binding = config.search_service or SearchServiceClientConfig()
    return SearchServiceClient(
        SearchServiceClientConfig(
            base_url=binding.base_url,
            admin_token_file=binding.admin_token_file,
            connect_timeout_seconds=binding.connect_timeout_seconds,
            request_timeout_seconds=binding.request_timeout_seconds,
            expected_service_cohort_sha256=binding.expected_service_cohort_sha256,
        )
    )


def _make_external_harness_manager(config: ExperimentConfig) -> ExternalHarnessManager:
    try:
        return manager_from_config(config, repo_root=Path(__file__).resolve().parents[1])
    except ExternalHarnessError as exc:
        raise RunnerError(str(exc)) from exc


def _bind_gateway_model_profile(gateway: object, profile: ResolvedModelProfile) -> None:
    """Keep ambiguous-row LLM review in the supervisor that owns credentials.

    Test gateways are intentionally duck-typed and need not implement this
    optional hook, while the production Volc gateways receive the resolved
    Aliyun/OpenAI-compatible profile before any session is opened.
    """

    bind = getattr(gateway, "bind_model_profile", None)
    if callable(bind):
        bind(profile)


class BenchmarkRunner:
    """Owns data/evaluator access; adapters receive no records or ground truth."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        dataset_source_factory: Callable[[ExperimentConfig], DatasetSource] | None = None,
        evaluation_engine: EvaluationEngine | None = None,
        search_gateway_factory: Callable[[ExperimentConfig], CutoffSearchGateway] | None = None,
        search_service_client_factory: Callable[[ExperimentConfig], SearchServiceClient]
        | None = None,
        external_harness_adapter_factory: Callable[..., FinanceHarnessAdapter] | None = None,
    ) -> None:
        self.config = config
        self._dataset_source_factory = dataset_source_factory
        self._evaluation_engine = evaluation_engine
        self._search_gateway_factory = search_gateway_factory
        self._search_service_client_factory = search_service_client_factory
        self._external_harness_adapter_factory = external_harness_adapter_factory

    def _dataset_source(self) -> DatasetSource:
        return (
            self._dataset_source_factory(self.config)
            if self._dataset_source_factory is not None
            else _make_dataset_source(self.config)
        )

    def _evaluation(self) -> tuple[EvaluationEngine, str]:
        if self._evaluation_engine is not None:
            # Test injection cannot make a performance claim and is marked by
            # its stable synthetic identity in provenance.
            return self._evaluation_engine, "injected-test-evaluator"
        return _evaluator_manifest(self.config)

    def _search_gateway(self) -> CutoffSearchGateway:
        return (
            self._search_gateway_factory(self.config)
            if self._search_gateway_factory is not None
            else _make_search_gateway(self.config)
        )

    def _search_service_client(self) -> SearchServiceClient:
        return (
            self._search_service_client_factory(self.config)
            if self._search_service_client_factory is not None
            else _make_search_service_client(self.config)
        )

    def preflight(self) -> dict[str, Any]:
        dataset = self._dataset_source().preflight()
        _, evaluator_manifest_sha = self._evaluation()
        profile = self.config.resolve_model(require_ready=False)
        real_endpoint_ready = profile is not None and profile.mode == "openai_compatible"
        search: dict[str, Any] | None = None
        external: dict[str, Any] | None = None
        if self.config.is_agentic_web_search or self.config.is_deep_research:
            if self._search_gateway_factory is not None:
                search = self._search_gateway().preflight()
            else:
                search = self._search_service_client().preflight()
        if self.config.is_deep_research:
            try:
                external = _make_external_harness_manager(self.config).verify()
            except ExternalHarnessError as exc:
                raise RunnerError(str(exc)) from exc
        return {
            "config_hash": self.config.config_hash,
            "input_policy": self.config.execution.input_policy,
            "dataset": dataset,
            "evaluator_manifest_sha256": evaluator_manifest_sha,
            "model_endpoint_ready": real_endpoint_ready,
            "model": profile.public_dict() if profile else None,
            "model_env_requirements": {
                "endpoint_env": self.config.model.endpoint_env,
                "name_env": self.config.model.name_env,
                "api_key_env": self.config.model.api_key_env,
            },
            "search": search,
            "external_harness": external,
            "mock_smoke_available": True,
            "note": (
                "No real endpoint configured; use scripts/offline_demo.py for a synthetic "
                "pipeline smoke."
                if not real_endpoint_ready and self.config.model.mode != "mock"
                else None
            ),
        }

    def data_sync(self) -> dict[str, Any]:
        """Prefetch all Hub configs into the local untracked cache."""

        if self.config.dataset.source != "huggingface":
            raise RunnerError("data-sync is available only for dataset.source = 'huggingface'.")
        source = self._dataset_source()
        return source.sync()

    def external_sync(self, *, acknowledge_undeclared_license: bool = False) -> dict[str, Any]:
        """Explicitly synchronize the configured external harness."""

        if not self.config.is_deep_research:
            raise RunnerError("external-sync requires a deep_research configuration.")
        try:
            return _make_external_harness_manager(self.config).sync(
                acknowledge_undeclared_license=acknowledge_undeclared_license
            )
        except ExternalHarnessError as exc:
            raise RunnerError(str(exc)) from exc

    def run(
        self,
        *,
        baselines: Sequence[str],
        categories: Iterable[str] | None = None,
        run_id: str | None = None,
        isolated: bool = True,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
    ) -> dict[str, Any]:
        if self.config.is_agentic_web_search and self.config.trace.results_dir.name == "results":
            raise RunnerError(
                "agentic_web_search requires a separate results directory from one_turn_strict."
            )
        if not baselines:
            raise RunnerError("At least one --baseline is required.")
        if isolated and any(
            value is not None
            for value in (
                self._dataset_source_factory,
                self._evaluation_engine,
                self._search_gateway_factory,
                self._search_service_client_factory,
                self._external_harness_adapter_factory,
            )
        ):
            raise RunnerError("Injected test dependencies require isolated=False.")

        normalized = [_normalize_baseline(item) for item in baselines]
        if len(set(normalized)) != len(normalized):
            raise RunnerError("Each baseline may be selected only once per run.")
        profile_spec = get_profile(self.config.execution.input_policy)
        incompatible = [
            baseline for baseline in normalized if not profile_spec.accepts_baseline(baseline)
        ]
        if incompatible:
            raise RunnerError(
                f"Profile {profile_spec.name!r} does not support baseline(s): "
                + ", ".join(incompatible)
                + "."
            )
        profile = self.config.resolve_model(require_ready=True)
        if profile is None:
            raise ConfigError("Unable to resolve model profile for execution.")
        service_client = None
        if self.config.is_agentic_web_search or self.config.is_deep_research:
            if self._search_gateway_factory is not None:
                self._search_gateway().preflight()
            else:
                service_client = self._search_service_client()
                service_client.preflight()
        if self.config.is_deep_research:
            _make_external_harness_manager(self.config).verify()

        resolved_run_id = _validate_run_id(run_id or make_run_id(self.config.config_hash))
        category_list = [str(category) for category in categories] if categories else None
        target_run_dir = self.config.trace.results_dir / resolved_run_id
        if target_run_dir.exists():
            raise RunnerError(
                f"Run directory already exists: {target_run_dir}. Choose a new --run-id to avoid "
                "mixing append-only trace artifacts."
            )
        store = ArtifactStore(self.config.trace.results_dir, resolved_run_id)
        outcomes: list[BaselineOutcome] = []

        for baseline in normalized:
            if isolated:
                outcome = _run_isolated_baseline(
                    self.config, profile, baseline, category_list, resolved_run_id, store.run_dir
                )
            else:
                outcome = _run_baseline(
                    self.config,
                    profile,
                    baseline,
                    category_list,
                    resolved_run_id,
                    store.run_dir,
                    dataset_source=self._dataset_source(),
                    evaluation=self._evaluation(),
                    search_gateway=(
                        self._search_gateway()
                        if (
                            (self.config.is_agentic_web_search or self.config.is_deep_research)
                            and self._search_gateway_factory is not None
                        )
                        else None
                    ),
                    search_service=service_client,
                    external_harness_adapter_factory=self._external_harness_adapter_factory,
                    progress_callback=progress_callback,
                )
            outcomes.append(outcome)

        comparison_paths = write_comparison(store.run_dir)
        return {
            "run_id": resolved_run_id,
            "run_dir": str(store.run_dir),
            "baselines": [outcome.as_dict() for outcome in outcomes],
            "comparison": {name: str(path) for name, path in comparison_paths.items()},
        }


def _normalize_baseline(name: str) -> str:
    try:
        return normalize_baseline(name)
    except ProfileRegistryError as exc:
        raise RunnerError(str(exc)) from exc


def _validate_run_id(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise RunnerError("run_id must be a non-empty single path component.")
    return value


def _run_isolated_baseline(
    config: ExperimentConfig,
    profile: ResolvedModelProfile,
    baseline: str,
    categories: list[str] | None,
    run_id: str,
    run_dir: Path,
) -> BaselineOutcome:
    """Run one baseline in a fresh spawned interpreter and synchronously join it."""

    context = mp.get_context("spawn")
    queue: mp.Queue[dict[str, Any]] = context.Queue()
    process = context.Process(
        target=_baseline_worker,
        args=(config, profile, baseline, categories, run_id, str(run_dir), queue),
        daemon=False,
    )
    _start_scrubbed_process(process)
    process.join()
    try:
        message = queue.get(timeout=2)
    except Exception as exc:
        raise RunnerError(
            f"Isolated {baseline} worker exited with code {process.exitcode} without a result."
        ) from exc
    finally:
        queue.close()
        queue.join_thread()

    if process.exitcode != 0 or not message.get("ok"):
        raise RunnerError(
            f"Isolated {baseline} worker failed: {message.get('error', 'unknown worker failure')}"
        )
    return BaselineOutcome(
        baseline=baseline,
        output_dir=Path(message["output_dir"]),
        task_count=int(message["task_count"]),
        comparable_task_count=int(message["comparable_task_count"]),
        mock_non_performance=bool(message["mock_non_performance"]),
    )


def _baseline_worker(
    config: ExperimentConfig,
    profile: ResolvedModelProfile,
    baseline: str,
    categories: list[str] | None,
    run_id: str,
    run_dir: str,
    queue: mp.Queue[dict[str, Any]],
) -> None:
    _scrub_baseline_environment()
    try:
        outcome = _run_baseline(
            config,
            profile,
            baseline,
            categories,
            run_id,
            Path(run_dir),
            progress_callback=_stderr_progress,
        )
    except BaseException as exc:
        queue.put({"ok": False, "error": f"{exc}\n{traceback.format_exc()}"})
        return
    queue.put({"ok": True, **outcome.as_dict()})


def _make_adapter(baseline: str, *, mock_responses: dict[str, str] | None) -> Adapter:
    transport = MappingMockTransport(mock_responses) if mock_responses is not None else None
    if baseline == "pydantic-ai":
        return PydanticAIAdapter(transport=transport)
    if baseline == "langgraph":
        return LangGraphAdapter(transport=transport)
    if baseline == "direct_transport":
        return DirectTransportAdapter(transport=transport)
    raise AssertionError(f"Unexpected normalized baseline: {baseline}")


class _IsolatedAdapterWorker:
    """A baseline-local worker that receives no dataset records or evaluator."""

    def __init__(self, baseline: str) -> None:
        self._context = mp.get_context("spawn")
        self._parent_connection, child_connection = self._context.Pipe()
        self._process = self._context.Process(
            target=_adapter_worker_loop,
            args=(baseline, child_connection),
            daemon=False,
        )
        _start_scrubbed_process(self._process)
        child_connection.close()

    def run(
        self,
        *,
        task: TaskInput,
        context: RunContext,
        mock_response: str | None,
    ) -> tuple[AdapterResult, RunContext]:
        self._parent_connection.send(
            {"command": "run", "task": task, "context": context, "mock_response": mock_response}
        )
        timeout = context.model.timeout_seconds + 30.0
        if not self._parent_connection.poll(timeout):
            self._process.terminate()
            self._process.join()
            raise RunnerError(f"Adapter worker exceeded {timeout:.1f}s for task {task.task_id}.")
        try:
            reply = self._parent_connection.recv()
        except (EOFError, OSError) as exc:
            raise RunnerError(
                f"Adapter worker exited unexpectedly for task {task.task_id} "
                f"(exit code {self._process.exitcode})."
            ) from exc
        if not reply.get("ok"):
            raise RunnerError(
                f"Adapter worker failed for {task.task_id}: {reply.get('error', 'unknown error')}"
            )
        return reply["result"], reply["context"]

    def close(self) -> None:
        try:
            if self._process.is_alive():
                try:
                    self._parent_connection.send({"command": "stop"})
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self._process.join(timeout=5)
        finally:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join()
            self._parent_connection.close()


def _adapter_worker_loop(baseline: str, connection: Any) -> None:
    """Run framework SDK code without data source, record, or evaluator objects."""

    _scrub_baseline_environment()
    os.environ.pop("BENCHMARK_DATASET_ROOT", None)
    try:
        while True:
            try:
                message = connection.recv()
            except EOFError:
                return
            if message.get("command") == "stop":
                return
            task = message["task"]
            context = message["context"]
            mock_response = message.get("mock_response")
            try:
                responses = (
                    {task.task_id: mock_response} if isinstance(mock_response, str) else None
                )
                adapter = _make_adapter(baseline, mock_responses=responses)
                result = adapter.run(task, context)
                connection.send({"ok": True, "result": result, "context": context})
            except BaseException as exc:
                connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


_AGENT_MODEL_ENV_PREFIXES = (
    "BENCHMARK_MODEL_",
    "ALIYUN_",
    "DASHSCOPE_",
    "BAILIAN_",
)
_BASELINE_PROXY_ENV_NAMES = frozenset(
    name
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    )
)
_CREDENTIAL_ENV_SEGMENTS = frozenset(
    {"ACCESS", "KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTHORIZATION"}
)


def _baseline_env_name_forbidden(name: str) -> bool:
    """Identify service/proxy credentials that never belong in a baseline worker."""

    upper = name.upper()
    if upper.startswith(("VOLC_", "SEARCH_FILTER_MODEL_")):
        return True
    if name in _BASELINE_PROXY_ENV_NAMES:
        return True
    # Agent model settings are resolved into the profile before the worker is
    # spawned. Preserve only the documented model aliases; generic provider,
    # admin, and token variables are not part of the baseline contract.
    if upper.startswith(_AGENT_MODEL_ENV_PREFIXES):
        return "ADMIN" in upper.split("_") or "TOKEN" in upper.split("_")
    segments = set(upper.split("_"))
    return bool(segments & _CREDENTIAL_ENV_SEGMENTS)


def _scrub_baseline_environment() -> None:
    """Remove service, proxy, and ambient credential variables in-place."""

    for name in tuple(os.environ):
        if _baseline_env_name_forbidden(name):
            os.environ.pop(name, None)


@contextmanager
def _scrubbed_parent_environment():
    """Give a spawned baseline only its allowed environment at exec time."""

    original = dict(os.environ)
    _scrub_baseline_environment()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def _start_scrubbed_process(process: Any) -> None:
    """Start a multiprocessing worker while its inherited env is scrubbed."""

    with _scrubbed_parent_environment():
        process.start()


def _seed_calendar_date(record: dict[str, Any], field: str) -> str | None:
    """Read only a seed date, normalizing HF's decoded datetime feature."""

    seed = record.get("seed")
    value = seed.get(field) if isinstance(seed, dict) else None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value if isinstance(value, str) else None


def _cutoff_from_record(record: dict[str, Any]) -> str | None:
    return _seed_calendar_date(record, "cutoff_date")


def _event_date_from_record(record: dict[str, Any]) -> str | None:
    return _seed_calendar_date(record, "event_date")


def _base_evaluation_row(
    *,
    task_id: str,
    category: str,
    record: dict[str, Any],
    baseline: str,
    profile: ResolvedModelProfile,
    prompt: str,
    result: AdapterResult | None,
    decision: PolicyDecision | Any,
    mock_non_performance: bool,
    policy: str,
    skipped_no_cutoff: bool = False,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "category": category,
        "variant": record.get("variant"),
        "time_band": record.get("time_band"),
        "agent": baseline,
        "model": profile.model_name,
        "prompt": prompt,
        "raw_response": result.raw_text if result else None,
        "api_error": result.error if result else None,
        "response_present": bool(result and result.raw_text and result.raw_text.strip()),
        "latency_seconds": result.latency_seconds if result else 0.0,
        "parsed": {},
        "metrics": {},
        "comparable": decision.comparable,
        "policy_violations": list(decision.violations),
        "canonical_input_hash": result.canonical_input_hash if result else None,
        "effective_input_hash": result.effective_input_hash if result else None,
        "model_call_count": result.model_call_count if result else 0,
        "filter_model_call_count": 0,
        "filter_review_count": 0,
        "service_receipt_count": 0,
        "child_search_service_call_count": 0,
        "child_forbidden_network_call_count": 0,
        "tool_call_count": len(result.tool_calls) if result else 0,
        "non_model_network_call_count": (
            sum(1 for event in result.network_events if not event.is_model_endpoint)
            if result
            else 0
        ),
        "usage": result.usage if result else {},
        "trace_artifact_ref": "trace_events.jsonl",
        "mock_non_performance": mock_non_performance,
        "input_policy": policy,
        "skipped_no_cutoff": skipped_no_cutoff,
    }


def _write_adapter_artifacts(
    sink: JsonlSink,
    *,
    run_id: str,
    task_id: str,
    baseline: str,
    result: AdapterResult,
    decision: PolicyDecision | Any,
    extra_request: dict[str, Any] | None = None,
    persisted_network_events: Iterable[NetworkEvent] | None = None,
) -> None:
    sink.write_trace_events(result.trace_events)
    sink.write_network_events(
        run_id=run_id,
        task_id=task_id,
        framework=baseline,
        events=(
            result.network_events if persisted_network_events is None else persisted_network_events
        ),
    )
    request_artifact = {
        **result.effective_request,
        "canonical_input_hash": result.canonical_input_hash,
        "effective_input_hash": result.effective_input_hash,
        "policy": decision.as_dict(),
    }
    if extra_request:
        request_artifact.update(extra_request)
    sink.write_effective_request(request_artifact)


def _run_baseline(
    config: ExperimentConfig,
    profile: ResolvedModelProfile,
    baseline: str,
    categories: list[str] | None,
    run_id: str,
    run_dir: Path,
    *,
    dataset_source: DatasetSource | None = None,
    evaluation: tuple[EvaluationEngine, str] | None = None,
    search_gateway: CutoffSearchGateway | None = None,
    search_service: SearchServiceClient | None = None,
    progress_callback: Callable[[ProgressUpdate], None] | None = None,
    external_harness_adapter_factory: Callable[..., FinanceHarnessAdapter] | None = None,
) -> BaselineOutcome:
    """The child-process implementation, with data/evaluation outside adapters."""

    if config.is_deep_research:
        return _run_deep_research_baseline(
            config,
            profile,
            baseline,
            categories,
            run_id,
            run_dir,
            dataset_source=dataset_source,
            evaluation=evaluation,
            search_gateway=search_gateway,
            search_service=search_service,
            progress_callback=progress_callback,
            external_harness_adapter_factory=external_harness_adapter_factory,
        )

    source = dataset_source or _make_dataset_source(config)
    source.preflight()
    engine, evaluator_manifest_sha = evaluation or _evaluator_manifest(config)
    records = source.load_records(categories=categories, status="ready")
    # An injected evaluator is a test seam only: frozen-manifest evaluation is
    # the sole path that may support a performance claim.
    mock_non_performance = (
        profile.mode == "mock" or evaluator_manifest_sha == "injected-test-evaluator"
    )
    mock_responses = (
        {str(record["task_id"]): engine.mock_response(record) for record in records}
        if mock_non_performance
        else None
    )
    adapter = _make_adapter(baseline, mock_responses=None)
    output_dir = run_dir / baseline
    sink = JsonlSink(output_dir, content_mode=config.trace.content_mode)
    store = ArtifactStore(run_dir.parent, run_id)
    context_template = RunContext(
        run_id=run_id,
        framework=baseline,
        adapter_version=adapter.adapter_version,
        model=profile,
        trace_content_mode=config.trace.content_mode,
    )
    strict_guard = StrictPolicyGuard()
    agentic_guard = AgenticWebSearchPolicyGuard()
    gateway = search_gateway
    service_client = search_service
    service_manifest_payload: dict[str, Any] | None = None
    if config.is_agentic_web_search and gateway is None and service_client is None:
        service_client = _make_search_service_client(config)
    if gateway is not None:
        _bind_gateway_model_profile(gateway, profile)
        gateway.preflight()
    elif service_client is not None:
        ready = service_client.preflight()
        service_manifest_payload = {
            key: ready.get(key)
            for key in (
                "protocol",
                "api_contract_sha256",
                "implementation_manifest_sha256",
                "capability_sha256",
                "service_cohort_sha256",
                "test_only",
            )
        }
        store.write_json(output_dir / "search_service_manifest.json", service_manifest_payload)

    predictions: list[dict[str, Any]] = []
    scorable_predictions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    strict_decisions: list[PolicyDecision] = []
    agentic_decisions: list[Any] = []
    # The strict adapter worker remains record-free and process-isolated.  The
    # agentic profile instead binds the native framework tool function to this
    # process's one cutoff gateway session; moving that session into the strict
    # worker would either serialize it or reintroduce runner-injected search.
    adapter_worker = _IsolatedAdapterWorker(baseline) if config.is_strict else None
    progress = _ProgressReporter(
        baseline=baseline,
        total=len(records),
        callback=progress_callback,
    )
    try:
        for record in records:
            task_started = time.perf_counter()
            task_id = str(record.get("task_id"))
            prompt = record.get("prompt")
            category = record.get("category")
            if not isinstance(prompt, str) or not isinstance(category, str):
                skipped.append(
                    {
                        "task_id": task_id,
                        "category": category,
                        "reason": "malformed upstream record",
                    }
                )
                progress.complete(
                    task_id=task_id,
                    category=category if isinstance(category, str) else None,
                    started=task_started,
                )
                continue
            cutoff_date = _cutoff_from_record(record)
            task = TaskInput(task_id=task_id, prompt=prompt, cutoff_date=cutoff_date)
            task_context = context_template.for_task(task_id)
            task_context.emit(
                "task_started", effective_input_hash=task.input_hash, control_flow_node="runner"
            )

            if config.is_strict:
                assert adapter_worker is not None
                result, task_context = adapter_worker.run(
                    task=task,
                    context=task_context,
                    mock_response=(
                        mock_responses.get(task_id) if mock_responses is not None else None
                    ),
                )
                decision = strict_guard.check(task, result)
                strict_decisions.append(decision)
                task_context.emit(
                    "task_finished",
                    effective_input_hash=result.effective_input_hash,
                    control_flow_node="runner",
                    latency_seconds=result.latency_seconds,
                    error=result.error,
                    attributes=decision.as_dict(),
                )
                _write_adapter_artifacts(
                    sink,
                    run_id=run_id,
                    task_id=task_id,
                    baseline=baseline,
                    result=replace(result, trace_events=task_context.events),
                    decision=decision,
                )
                row = _base_evaluation_row(
                    task_id=task_id,
                    category=category,
                    record=record,
                    baseline=baseline,
                    profile=profile,
                    prompt=prompt,
                    result=result,
                    decision=decision,
                    mock_non_performance=mock_non_performance,
                    policy=config.execution.input_policy,
                )
                row["parsed"] = engine.parse_prediction(category, result.raw_text or "")
                row["metrics"] = engine.evaluate_prediction(category, row["parsed"], record)
                predictions.append(
                    _prediction_artifact(row, content_mode=config.trace.content_mode)
                )
                if decision.comparable:
                    scorable_predictions.append(_upstream_row(row))
                else:
                    skipped.append(
                        {
                            "task_id": task_id,
                            "category": category,
                            "reason": "non-comparable strict policy: "
                            + ", ".join(decision.violations),
                        }
                    )
                progress.complete(task_id=task_id, category=category, started=task_started)
                continue

            if gateway is None and service_client is None:
                raise RunnerError("agentic search service is not configured")
            expected_no_cutoff = category in {"D", "E"} and cutoff_date is None
            if cutoff_date is None:
                decision = agentic_guard.check(task, None, skipped_no_cutoff=expected_no_cutoff)
                agentic_decisions.append(decision)
                task_context.emit(
                    "task_finished",
                    effective_input_hash=task.input_hash,
                    control_flow_node="agentic_web_search_policy",
                    attributes=decision.as_dict(),
                )
                sink.write_trace_events(task_context.events)
                sink.write_effective_request(
                    {
                        "run_id": run_id,
                        "task_id": task_id,
                        "framework": baseline,
                        "policy": decision.as_dict(),
                        "input_policy": config.execution.input_policy,
                        "skipped_no_cutoff": expected_no_cutoff,
                    }
                )
                row = _base_evaluation_row(
                    task_id=task_id,
                    category=category,
                    record=record,
                    baseline=baseline,
                    profile=profile,
                    prompt=prompt,
                    result=None,
                    decision=decision,
                    mock_non_performance=mock_non_performance,
                    policy=config.execution.input_policy,
                    skipped_no_cutoff=expected_no_cutoff,
                )
                predictions.append(
                    _prediction_artifact(row, content_mode=config.trace.content_mode)
                )
                skipped.append(
                    {
                        "task_id": task_id,
                        "category": category,
                        "reason": (
                            "skipped_no_cutoff"
                            if expected_no_cutoff
                            else "invalid_or_missing_cutoff"
                        ),
                    }
                )
                progress.complete(task_id=task_id, category=category, started=task_started)
                continue

            if category == "B":
                try:
                    validate_b_cutoff_not_after_event(cutoff_date, _event_date_from_record(record))
                except BEventDateError as exc:
                    decision = _noncomparable_agentic_decision(str(exc))
                    agentic_decisions.append(decision)
                    task_context.emit(
                        "task_finished",
                        effective_input_hash=task.input_hash,
                        control_flow_node="agentic_web_search_policy",
                        attributes=decision.as_dict(),
                    )
                    sink.write_trace_events(task_context.events)
                    row = _base_evaluation_row(
                        task_id=task_id,
                        category=category,
                        record=record,
                        baseline=baseline,
                        profile=profile,
                        prompt=prompt,
                        result=None,
                        decision=decision,
                        mock_non_performance=mock_non_performance,
                        policy=config.execution.input_policy,
                    )
                    predictions.append(
                        _prediction_artifact(row, content_mode=config.trace.content_mode)
                    )
                    skipped.append({"task_id": task_id, "category": category, "reason": str(exc)})
                    progress.complete(task_id=task_id, category=category, started=task_started)
                    continue

            try:
                if gateway is not None:
                    search_session = gateway.new_session(
                        task_id=task_id,
                        framework=baseline,
                        cutoff_date=cutoff_date,
                        category=category,
                    )
                else:
                    assert service_client is not None
                    service_client.preflight()
                    search_session = service_client.open_session(
                        run_id=run_id,
                        task_id=task_id,
                        profile="agentic_web_search",
                        baseline=baseline,
                        cutoff_date=cutoff_date,
                        policy_version="agentic_web_search/v2",
                    )
            except (SearchGatewayError, SearchServiceError) as exc:
                decision = _noncomparable_agentic_decision(exc.code)
                agentic_decisions.append(decision)
                task_context.emit(
                    "task_finished",
                    effective_input_hash=task.input_hash,
                    control_flow_node="agentic_web_search_policy",
                    attributes=decision.as_dict(),
                )
                sink.write_trace_events(task_context.events)
                row = _base_evaluation_row(
                    task_id=task_id,
                    category=category,
                    record=record,
                    baseline=baseline,
                    profile=profile,
                    prompt=prompt,
                    result=None,
                    decision=decision,
                    mock_non_performance=mock_non_performance,
                    policy=config.execution.input_policy,
                )
                predictions.append(
                    _prediction_artifact(row, content_mode=config.trace.content_mode)
                )
                skipped.append({"task_id": task_id, "category": category, "reason": exc.code})
                progress.complete(task_id=task_id, category=category, started=task_started)
                continue

            # The runner neither writes a query nor provides evidence to the
            # model.  Each native framework registers ``search_web(query: str)``
            # and its actual model-issued calls enter this shared session.
            if not isinstance(adapter, AgenticAdapter):
                raise RunnerError(
                    f"{baseline} does not implement the native agentic search tool contract."
                )
            try:
                result = adapter.run_agentic(task, task_context, search_session)
            except Exception:
                if service_client is not None and gateway is None:
                    try:
                        search_session.close(status="aborted")
                    except Exception:
                        pass
                raise
            search_outcomes = search_session.outcomes
            service_close = None
            if service_client is not None and gateway is None:
                service_close = search_session.close(status="completed")
                _emit_service_filter_trace(task_context, service_close)
            search_ledgers = tuple(
                ledger
                for outcome in search_outcomes
                for ledger in (
                    getattr(outcome, "network_ledger_rows", ())
                    or (outcome.network_ledger(profile=config.execution.input_policy),)
                )
            )
            if service_client is not None and gateway is None:
                search_ledgers = search_ledgers + tuple(
                    _service_local_ledger(
                        run_id=run_id,
                        task_id=task_id,
                        baseline=baseline,
                        session_id=search_session.descriptor.session_id,
                        service_request_id=receipt.service_request_id,
                        receipt_sha256=receipt.receipt_sha256,
                        call_index=receipt.call_index,
                    )
                    for receipt in search_session.receipts
                )
            sink.write_search_audit(
                service_close.get("search_audit", [])
                if isinstance(service_close, dict)
                else [
                    audit.as_dict()
                    for outcome in search_outcomes
                    for audit in outcome.audit_records
                ]
            )
            sink.write_network_ledger_rows(
                {
                    "run_id": run_id,
                    "task_id": task_id,
                    "framework": baseline,
                    **ledger,
                }
                for ledger in search_ledgers
            )
            decision = agentic_guard.check(
                task,
                result,
                search_outcomes=search_outcomes,
                network_ledger_rows=search_ledgers,
            )
            agentic_decisions.append(decision)
            task_context.emit(
                "task_finished",
                effective_input_hash=result.effective_input_hash,
                control_flow_node="runner",
                latency_seconds=result.latency_seconds,
                error=result.error,
                attributes=decision.as_dict(),
            )
            _write_adapter_artifacts(
                sink,
                run_id=run_id,
                task_id=task_id,
                baseline=baseline,
                result=replace(result, trace_events=task_context.events),
                decision=decision,
                extra_request={
                    "agentic_web_search": {
                        "tool_contract_hash": search_session.gateway.tool_contract.hash,
                        "tool_policy_sha256": sha256_text(
                            search_session.gateway.tool_contract.tool_policy_text
                        ),
                        "model_call_budget": search_session.budget.model_calls,
                        "tool_call_budget": search_session.budget.tool_calls,
                        "search_call_count": len(search_outcomes),
                        "service_receipt_count": len(getattr(search_session, "receipts", ())),
                        "service_receipt_sha256s": [
                            receipt.receipt_sha256
                            for receipt in getattr(search_session, "receipts", ())
                        ],
                        "service_manifest": (
                            service_close.get("service_manifest")
                            if isinstance(service_close, dict)
                            else None
                        ),
                        **_service_identity_fields(service_close),
                    }
                },
            )
            row = _base_evaluation_row(
                task_id=task_id,
                category=category,
                record=record,
                baseline=baseline,
                profile=profile,
                prompt=prompt,
                result=result,
                decision=decision,
                mock_non_performance=mock_non_performance,
                policy=config.execution.input_policy,
            )
            row.update(
                {
                    "filter_review_count": sum(
                        int(getattr(outcome, "filter_review_count", 0))
                        for outcome in search_outcomes
                    ),
                    "filter_model_call_count": sum(
                        int(getattr(outcome, "filter_model_call_count", 0))
                        for outcome in search_outcomes
                    ),
                    "service_receipt_count": len(getattr(search_session, "receipts", ())),
                }
            )
            row["parsed"] = engine.parse_prediction(category, result.raw_text or "")
            row["metrics"] = engine.evaluate_prediction(category, row["parsed"], record)
            predictions.append(_prediction_artifact(row, content_mode=config.trace.content_mode))
            if decision.comparable:
                scorable_predictions.append(_upstream_row(row))
            else:
                skipped.append(
                    {
                        "task_id": task_id,
                        "category": category,
                        "reason": "non-comparable agentic policy: "
                        + ", ".join(decision.violations),
                    }
                )
            progress.complete(task_id=task_id, category=category, started=task_started)
    finally:
        if adapter_worker is not None:
            adapter_worker.close()

    upstream_summary = engine.summarize_results(
        scorable_predictions, skipped, agent=baseline, model=profile.model_name
    )
    decisions = strict_decisions if config.is_strict else agentic_decisions
    summary = dict(upstream_summary)
    summary["upstream_summary"] = upstream_summary
    summary["harness"] = {
        "input_policy": config.execution.input_policy,
        "mock_non_performance": mock_non_performance,
        "total_adapter_tasks": len(predictions),
        "comparable_task_count": len(scorable_predictions),
        "non_comparable_task_count": len(predictions) - len(scorable_predictions),
        "policy_violation_counts": violation_counts(decisions),
    }
    source_provenance = source.provenance()
    tool_contract_hash = (
        gateway.tool_contract.hash if gateway is not None else DEFAULT_TOOL_CONTRACT.hash
    )
    provenance = {
        **source_provenance,
        "run_id": run_id,
        "framework": baseline,
        "framework_version": adapter.adapter_version,
        "adapter_version": adapter.adapter_version,
        "lockfile_sha256": file_sha256(Path(__file__).resolve().parents[1] / "uv.lock"),
        "config_hash": config.config_hash,
        "input_policy": config.execution.input_policy,
        "policy_version": get_profile(config.execution.input_policy).policy_version,
        "evaluator_manifest_sha256": evaluator_manifest_sha,
        "tool_contract_hash": tool_contract_hash,
        "tool_policy_sha256": sha256_text(
            gateway.tool_contract.tool_policy_text
            if gateway is not None
            else DEFAULT_TOOL_CONTRACT.tool_policy_text
        ),
        "service_cohort_sha256": (
            service_manifest_payload.get("service_cohort_sha256")
            if isinstance(service_manifest_payload, dict)
            else None
        ),
        "service_api_contract_sha256": (
            service_manifest_payload.get("api_contract_sha256")
            if isinstance(service_manifest_payload, dict)
            else None
        ),
        "service_implementation_manifest_sha256": (
            service_manifest_payload.get("implementation_manifest_sha256")
            if isinstance(service_manifest_payload, dict)
            else None
        ),
        "service_capability_sha256": (
            service_manifest_payload.get("capability_sha256")
            if isinstance(service_manifest_payload, dict)
            else None
        ),
        "search_test_only": (
            gateway.test_only
            if gateway is not None
            else bool(service_manifest_payload and service_manifest_payload.get("test_only"))
        ),
        "provider_profile": profile.public_dict(),
        "mock_non_performance": mock_non_performance,
        "baseline_isolation": (
            "fresh spawned baseline subprocess plus record-free adapter worker"
            if config.is_strict
            else "fresh spawned baseline subprocess plus native framework tool session"
        ),
        "categories": categories,
    }
    run_config = {
        **config.public_dict(),
        "run_id": run_id,
        "baseline": baseline,
        "categories": categories,
        "resolved_model": profile.public_dict(),
        "mock_non_performance": mock_non_performance,
    }
    if service_manifest_payload is not None:
        run_config["search_service"] = {
            "protocol": "search_filter_http/v1",
            "service_client_only": True,
            **_service_identity_fields({"service_manifest": service_manifest_payload}),
        }
    if gateway is not None:
        run_config["search"]["tool_contract_hash"] = gateway.tool_contract.hash
        run_config["search"]["tool_contract"] = gateway.tool_contract.as_dict()
        run_config["search"]["test_only"] = gateway.test_only
    elif service_client is not None:
        run_config["search_service"] = {
            "protocol": "search_filter_http/v1",
            "tool_contract_hash": DEFAULT_TOOL_CONTRACT.hash,
            "service_client_only": True,
            **_service_identity_fields({"service_manifest": service_manifest_payload}),
        }
    store.write_jsonl(output_dir / "predictions.jsonl", predictions)
    store.write_json(output_dir / "metrics_summary.json", summary)
    store.write_json(output_dir / "run_config.json", run_config)
    store.write_json(output_dir / "provenance.json", provenance)
    return BaselineOutcome(
        baseline=baseline,
        output_dir=output_dir,
        task_count=len(predictions),
        comparable_task_count=len(scorable_predictions),
        mock_non_performance=mock_non_performance,
    )


def _run_deep_research_baseline(
    config: ExperimentConfig,
    profile: ResolvedModelProfile,
    baseline: str,
    categories: list[str] | None,
    run_id: str,
    run_dir: Path,
    *,
    dataset_source: DatasetSource | None = None,
    evaluation: tuple[EvaluationEngine, str] | None = None,
    search_gateway: CutoffSearchGateway | None = None,
    search_service: SearchServiceClient | None = None,
    progress_callback: Callable[[ProgressUpdate], None] | None = None,
    external_harness_adapter_factory: Callable[..., FinanceHarnessAdapter] | None = None,
) -> BaselineOutcome:
    """Run the external FinanceHarness population behind the host IPC boundary."""

    source = dataset_source or _make_dataset_source(config)
    source.preflight()
    engine, evaluator_manifest_sha = evaluation or _evaluator_manifest(config)
    records = source.load_records(categories=categories, status="ready")
    mock_non_performance = (
        profile.mode == "mock" or evaluator_manifest_sha == "injected-test-evaluator"
    )
    manager = _make_external_harness_manager(config)
    external_manifest = manager.verify()
    gateway = search_gateway
    service_client = search_service
    if gateway is None and service_client is None:
        service_client = _make_search_service_client(config)
    if gateway is not None:
        _bind_gateway_model_profile(gateway, profile)
        gateway.preflight()
    elif service_client is not None:
        service_client.preflight()

    if external_harness_adapter_factory is not None:
        try:
            adapter = external_harness_adapter_factory(
                config=config,
                profile=profile,
                baseline=baseline,
                gateway=gateway,
                manager=manager,
                search_service=service_client,
            )
        except TypeError:
            adapter = external_harness_adapter_factory(config, profile, gateway)
    else:
        adapter = FinanceHarnessAdapter(
            manager,
            deep_config=config.deep_research,
            harness_config=config.external_harnesses.get("financeharness"),
            search_service=service_client,
            repo_root=Path(__file__).resolve().parents[1],
        )
    if not isinstance(adapter, FinanceHarnessAdapter) and not callable(
        getattr(adapter, "run_deep", None)
    ):
        raise RunnerError("deep_research adapter does not implement run_deep")

    output_dir = run_dir / baseline
    sink = JsonlSink(
        output_dir, content_mode=config.trace.content_mode, include_research_audit=True
    )
    store = ArtifactStore(run_dir.parent, run_id)
    service_manifest_payload: dict[str, Any] | None = None
    if service_client is not None and gateway is None:
        ready = service_client.preflight()
        service_manifest_payload = {
            key: ready.get(key)
            for key in (
                "protocol",
                "api_contract_sha256",
                "implementation_manifest_sha256",
                "capability_sha256",
                "service_cohort_sha256",
                "test_only",
            )
        }
        store.write_json(output_dir / "search_service_manifest.json", service_manifest_payload)
    store.write_json(output_dir / "external_harness_manifest.json", external_manifest)
    context_template = RunContext(
        run_id=run_id,
        framework=baseline,
        adapter_version=adapter.adapter_version,
        model=profile,
        trace_content_mode=config.trace.content_mode,
    )
    guard = DeepResearchPolicyGuard(config.deep_research)
    predictions: list[dict[str, Any]] = []
    scorable_predictions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    decisions: list[Any] = []
    progress = _ProgressReporter(
        baseline=baseline,
        total=len(records),
        callback=progress_callback,
    )

    for record in records:
        task_started = time.perf_counter()
        task_id = str(record.get("task_id"))
        prompt = record.get("prompt")
        category = record.get("category")
        if not isinstance(prompt, str) or not isinstance(category, str):
            skipped.append(
                {"task_id": task_id, "category": category, "reason": "malformed upstream record"}
            )
            progress.complete(
                task_id=task_id,
                category=category if isinstance(category, str) else None,
                started=task_started,
            )
            continue
        cutoff_date = _cutoff_from_record(record)
        task = TaskInput(task_id=task_id, prompt=prompt, cutoff_date=cutoff_date)
        task_context = context_template.for_task(task_id)
        task_context.emit(
            "task_started", effective_input_hash=task.input_hash, control_flow_node="runner"
        )
        expected_no_cutoff = category in {"D", "E"} and cutoff_date is None

        if cutoff_date is None or (category == "B" and _invalid_b_cutoff(cutoff_date, record)):
            reason = "skipped_no_cutoff" if expected_no_cutoff else "invalid_or_missing_cutoff"
            if category == "B" and cutoff_date is not None:
                reason = "cutoff_after_event_date"
            decision = guard.check(task, None, skipped_no_cutoff=expected_no_cutoff)
            if reason != "skipped_no_cutoff":
                decision = _deep_decision_with_violation(decision, reason)
            decisions.append(decision)
            task_context.emit(
                "task_finished",
                effective_input_hash=task.input_hash,
                control_flow_node="deep_research_policy",
                attributes=decision.as_dict(),
            )
            audit_rows = (
                {"task_id": task_id, "event_type": "task_started", "network_call_count": 0},
                {"task_id": task_id, "event_type": "task_finished", "network_call_count": 0},
            )
            sink.write_research_audit(audit_rows)
            sink.write_trace_events(task_context.events)
            sink.write_effective_request(
                {
                    "run_id": run_id,
                    "task_id": task_id,
                    "framework": baseline,
                    "input_policy": config.execution.input_policy,
                    "canonical_input_hash": task.input_hash,
                    "effective_input_hash": task.input_hash,
                    "prompt_sha256": sha256_text(prompt),
                    "messages_sha256": task.input_hash,
                    "policy": decision.as_dict(),
                    "skipped_no_cutoff": expected_no_cutoff,
                    "deep_research": {
                        "protocol": config.external_harnesses["financeharness"].protocol,
                        "skipped_no_cutoff": expected_no_cutoff,
                    },
                }
            )
            row = _base_evaluation_row(
                task_id=task_id,
                category=category,
                record=record,
                baseline=baseline,
                profile=profile,
                prompt=prompt,
                result=None,
                decision=decision,
                mock_non_performance=mock_non_performance,
                policy=config.execution.input_policy,
                skipped_no_cutoff=expected_no_cutoff,
            )
            row.update(
                {
                    "search_call_count": 0,
                    "visit_count": 0,
                    "child_network_call_count": 0,
                    "research_audit_complete": True,
                }
            )
            # A policy-only skip is not a frozen-evaluator input.  In
            # particular, D/E's legal no-cutoff decision is comparable at the
            # policy layer but is not scoreable without a cutoff.
            row["parsed"] = {}
            row["metrics"] = {}
            predictions.append(_prediction_artifact(row, content_mode=config.trace.content_mode))
            skipped.append({"task_id": task_id, "category": category, "reason": reason})
            progress.complete(task_id=task_id, category=category, started=task_started)
            continue

        try:
            deep_output = adapter.run_deep(task, task_context)
        except FinanceHarnessBoundaryError as exc:
            decision = _deep_decision_with_violation(
                guard.check(task, None),
                f"external_boundary_error:{type(exc).__name__}",
            )
            decisions.append(decision)
            task_context.emit(
                "task_finished",
                effective_input_hash=task.input_hash,
                control_flow_node="deep_research_policy",
                error=str(exc),
                attributes=decision.as_dict(),
            )
            sink.write_trace_events(task_context.events)
            sink.write_research_audit(
                (
                    {"task_id": task_id, "event_type": "task_started", "network_call_count": 0},
                    {
                        "task_id": task_id,
                        "event_type": "task_finished",
                        "network_call_count": 0,
                        "error": type(exc).__name__,
                    },
                )
            )
            sink.write_effective_request(
                {
                    "run_id": run_id,
                    "task_id": task_id,
                    "framework": baseline,
                    "input_policy": config.execution.input_policy,
                    "canonical_input_hash": task.input_hash,
                    "effective_input_hash": task.input_hash,
                    "prompt_sha256": sha256_text(prompt),
                    "messages_sha256": task.input_hash,
                    "policy": decision.as_dict(),
                    "deep_research": {
                        "protocol": config.external_harnesses["financeharness"].protocol,
                    },
                }
            )
            row = _base_evaluation_row(
                task_id=task_id,
                category=category,
                record=record,
                baseline=baseline,
                prompt=prompt,
                profile=profile,
                result=None,
                decision=decision,
                mock_non_performance=mock_non_performance,
                policy=config.execution.input_policy,
            )
            row.update({"search_call_count": 0, "visit_count": 0, "child_network_call_count": 0})
            # Boundary failures are non-comparable and must not be converted
            # into blank frozen-evaluator scores.
            row["parsed"] = {}
            row["metrics"] = {}
            predictions.append(_prediction_artifact(row, content_mode=config.trace.content_mode))
            skipped.append({"task_id": task_id, "category": category, "reason": str(exc)})
            progress.complete(task_id=task_id, category=category, started=task_started)
            continue

        result = deep_output.result
        if isinstance(deep_output.service_close, dict):
            close_manifest = deep_output.service_close.get("service_manifest")
            if isinstance(close_manifest, dict):
                service_manifest_payload = {
                    **(service_manifest_payload or {}),
                    **close_manifest,
                }
        sink.write_search_audit(
            deep_output.service_close.get("search_audit", [])
            if isinstance(deep_output.service_close, dict)
            else [
                audit.as_dict()
                for outcome in deep_output.search_outcomes
                for audit in outcome.audit_records
            ]
        )
        sink.write_network_ledger_rows(
            {
                "run_id": run_id,
                "task_id": task_id,
                "framework": baseline,
                **ledger,
            }
            for ledger in deep_output.search_ledger_rows
        )
        sink.write_research_audit(deep_output.research_audit_rows)
        decision = guard.check(
            task,
            result,
            search_outcomes=deep_output.search_outcomes,
            network_ledger_rows=deep_output.search_ledger_rows,
            research_audit_rows=deep_output.research_audit_rows,
            child_network_call_count=deep_output.child_network_call_count,
            child_search_service_call_count=deep_output.child_search_service_call_count,
            child_forbidden_network_call_count=deep_output.child_forbidden_network_call_count,
            citations=deep_output.citations,
        )
        decisions.append(decision)
        _write_adapter_artifacts(
            sink,
            run_id=run_id,
            task_id=task_id,
            baseline=baseline,
            result=result,
            decision=decision,
            extra_request={
                "deep_research": {
                    "protocol": config.external_harnesses["financeharness"].protocol,
                    "external_revision": external_manifest.get("revision"),
                    "external_tree_sha": external_manifest.get("tree_sha"),
                    "external_manifest_sha256": sha256_json(external_manifest),
                    "search_call_count": len(deep_output.search_outcomes),
                    "visit_count": decision.visit_count,
                    "child_network_call_count": deep_output.child_network_call_count,
                    "child_search_service_call_count": deep_output.child_search_service_call_count,
                    "child_forbidden_network_call_count": (
                        deep_output.child_forbidden_network_call_count
                    ),
                    "service_receipt_count": len(
                        deep_output.service_close.get("receipts", [])
                        if isinstance(deep_output.service_close, dict)
                        else []
                    ),
                    "service_receipt_sha256s": [
                        receipt.get("receipt_sha256")
                        for receipt in (
                            deep_output.service_close.get("receipts", [])
                            if isinstance(deep_output.service_close, dict)
                            else []
                        )
                        if isinstance(receipt, dict)
                        and isinstance(receipt.get("receipt_sha256"), str)
                    ],
                    "service_manifest": (
                        deep_output.service_close.get("service_manifest")
                        if isinstance(deep_output.service_close, dict)
                        else None
                    ),
                    **_service_identity_fields(deep_output.service_close),
                }
            },
        )
        row = _base_evaluation_row(
            task_id=task_id,
            category=category,
            record=record,
            baseline=baseline,
            profile=profile,
            prompt=prompt,
            result=result,
            decision=decision,
            mock_non_performance=mock_non_performance,
            policy=config.execution.input_policy,
        )
        row.update(
            {
                "search_call_count": len(deep_output.search_outcomes),
                "visit_count": decision.visit_count,
                "filter_model_call_count": decision.filter_model_call_count,
                "child_network_call_count": deep_output.child_network_call_count,
                "child_search_service_call_count": deep_output.child_search_service_call_count,
                "child_forbidden_network_call_count": (
                    deep_output.child_forbidden_network_call_count
                ),
                "service_receipt_count": len(
                    deep_output.service_close.get("receipts", [])
                    if isinstance(deep_output.service_close, dict)
                    else []
                ),
                "research_audit_complete": not any(
                    row.get("event_type") == "policy.violation"
                    for row in deep_output.research_audit_rows
                ),
            }
        )
        if decision.comparable:
            row["parsed"] = engine.parse_prediction(category, result.raw_text or "")
            row["metrics"] = engine.evaluate_prediction(category, row["parsed"], record)
        else:
            # Keep non-comparable adapter output in the artifact, but never
            # submit it to the frozen evaluator or scoring aggregate.
            row["parsed"] = {}
            row["metrics"] = {}
        predictions.append(_prediction_artifact(row, content_mode=config.trace.content_mode))
        if decision.comparable:
            scorable_predictions.append(_upstream_row(row))
        else:
            skipped.append(
                {
                    "task_id": task_id,
                    "category": category,
                    "reason": "non-comparable deep_research policy: "
                    + ", ".join(decision.violations),
                }
            )
        progress.complete(task_id=task_id, category=category, started=task_started)

    upstream_summary = engine.summarize_results(
        scorable_predictions, skipped, agent=baseline, model=profile.model_name
    )
    summary = dict(upstream_summary)
    summary["by_time_band"] = _summary_by_time_band(predictions)
    summary["upstream_summary"] = upstream_summary
    summary["harness"] = {
        "input_policy": config.execution.input_policy,
        "mock_non_performance": mock_non_performance,
        "total_adapter_tasks": len(predictions),
        "comparable_task_count": len(scorable_predictions),
        "non_comparable_task_count": len(predictions) - len(scorable_predictions),
        "policy_violation_counts": violation_counts(decisions),
        "by_time_band": _summary_by_time_band(predictions),
    }
    source_provenance = source.provenance()
    provenance = {
        **source_provenance,
        "run_id": run_id,
        "framework": baseline,
        "framework_version": adapter.adapter_version,
        "adapter_version": adapter.adapter_version,
        "lockfile_sha256": file_sha256(Path(__file__).resolve().parents[1] / "uv.lock"),
        "config_hash": config.config_hash,
        "input_policy": config.execution.input_policy,
        "policy_version": get_profile(config.execution.input_policy).policy_version,
        "evaluator_manifest_sha256": evaluator_manifest_sha,
        "tool_contract_hash": DEFAULT_TOOL_CONTRACT.hash,
        "tool_policy_sha256": sha256_text(DEFAULT_TOOL_CONTRACT.tool_policy_text),
        "search_test_only": (
            gateway.test_only
            if gateway is not None
            else bool(service_manifest_payload and service_manifest_payload.get("test_only"))
        ),
        "provider_profile": profile.public_dict(),
        "mock_non_performance": mock_non_performance,
        "baseline_isolation": (
            "external FinanceHarness Python 3.12 worker via external_research_stdio/v2"
        ),
        "external_harness_manifest": external_manifest,
        "external_harness_manifest_sha256": sha256_json(external_manifest),
        "external_harness_revision": external_manifest.get("revision"),
        "external_harness_tree_sha": external_manifest.get("tree_sha"),
        "external_harness_lock_sha256": external_manifest.get("uv_lock_sha256"),
        "external_harness_project_sha256": external_manifest.get("pyproject_sha256"),
        "ipc_protocol": config.external_harnesses["financeharness"].protocol,
        "service_manifest": (service_manifest_payload),
        **_service_identity_fields({"service_manifest": service_manifest_payload}),
        "bridge_sha256": external_manifest.get("bridge_sha256"),
        "capability_manifest_sha256": external_manifest.get("capability_manifest_sha256"),
        "budget_sha256": external_manifest.get("budget_sha256"),
        "deep_budget": asdict(config.deep_research) if config.deep_research else None,
        "child_network_call_count": sum(
            row.get("child_network_call_count", 0) for row in predictions
        ),
        "child_search_service_call_count": sum(
            row.get("child_search_service_call_count", 0) for row in predictions
        ),
        "child_forbidden_network_call_count": sum(
            row.get("child_forbidden_network_call_count", 0) for row in predictions
        ),
        "categories": categories,
    }
    run_config = {
        **config.public_dict(),
        "run_id": run_id,
        "baseline": baseline,
        "categories": categories,
        "resolved_model": profile.public_dict(),
        "mock_non_performance": mock_non_performance,
    }
    if service_manifest_payload is not None:
        run_config["search_service"] = {
            "protocol": "search_filter_http/v1",
            "service_client_only": True,
            **_service_identity_fields({"service_manifest": service_manifest_payload}),
        }
    store.write_jsonl(output_dir / "predictions.jsonl", predictions)
    store.write_json(output_dir / "metrics_summary.json", summary)
    store.write_json(output_dir / "run_config.json", run_config)
    store.write_json(output_dir / "provenance.json", provenance)
    return BaselineOutcome(
        baseline=baseline,
        output_dir=output_dir,
        task_count=len(predictions),
        comparable_task_count=len(scorable_predictions),
        mock_non_performance=mock_non_performance,
    )


def _invalid_b_cutoff(cutoff_date: str, record: dict[str, Any]) -> bool:
    try:
        validate_b_cutoff_not_after_event(cutoff_date, _event_date_from_record(record))
    except BEventDateError:
        return True
    return False


def _deep_decision_with_violation(decision: Any, violation: str) -> Any:
    from benchmark_harness.external_harnesses.financeharness.policy import DeepPolicyDecision

    return DeepPolicyDecision(
        comparable=False,
        violations=tuple(dict.fromkeys([*decision.violations, violation])),
        skipped_no_cutoff=decision.skipped_no_cutoff,
        child_network_call_count=decision.child_network_call_count,
        model_call_count=decision.model_call_count,
        filter_model_call_count=decision.filter_model_call_count,
        search_call_count=decision.search_call_count,
        visit_count=decision.visit_count,
    )


def _summary_by_time_band(predictions: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in predictions:
        band = row.get("time_band")
        if isinstance(band, str) and band:
            counts[band] = counts.get(band, 0) + 1
    return dict(sorted(counts.items()))


def _service_local_ledger(
    *,
    run_id: str,
    task_id: str,
    baseline: str,
    session_id: str,
    service_request_id: str,
    receipt_sha256: str,
    call_index: int,
) -> dict[str, Any]:
    """Record the authorized adapter-to-service hop separately from Volc."""

    return {
        "run_id": run_id,
        "task_id": task_id,
        "framework": baseline,
        "network_class": "search_service_local",
        "provider": "local",
        "host": "127.0.0.1",
        "method": "POST",
        "path": f"/v1/sessions/{session_id}/search",
        "body_hash": receipt_sha256,
        "status_code": 200,
        "latency_seconds": None,
        "is_model_endpoint": False,
        "session_id": session_id,
        "service_request_id": service_request_id,
        "call_index": call_index,
        "receipt_sha256": receipt_sha256,
    }


def _service_identity_fields(close: dict[str, Any] | None) -> dict[str, Any]:
    manifest = close.get("service_manifest") if isinstance(close, dict) else None
    if not isinstance(manifest, dict):
        return {
            "service_cohort_sha256": None,
            "service_api_contract_sha256": None,
            "service_implementation_manifest_sha256": None,
            "service_capability_sha256": None,
        }
    return {
        "service_cohort_sha256": manifest.get("service_cohort_sha256"),
        "service_api_contract_sha256": manifest.get("api_contract_sha256"),
        "service_implementation_manifest_sha256": manifest.get("implementation_manifest_sha256"),
        "service_capability_sha256": manifest.get("capability_sha256"),
    }


def _emit_service_filter_trace(context: RunContext, close: dict[str, Any] | None) -> None:
    """Persist service-owned filter calls without charging agent cognition."""

    if not isinstance(close, dict):
        return
    for row in close.get("network_ledger", []):
        if not isinstance(row, dict) or row.get("network_class") != "temporal_filter":
            continue
        attributes = {
            "filter": True,
            "network_class": "temporal_filter",
            "service_request_id": row.get("service_request_id"),
            "receipt_sha256": row.get("receipt_sha256"),
        }
        context.emit(
            "model_request",
            control_flow_node="parent_temporal_filter",
            attributes=attributes,
        )
        context.emit(
            "model_response",
            control_flow_node="parent_temporal_filter",
            network=row,
            error=row.get("error") if isinstance(row.get("error"), str) else None,
            attributes=attributes,
        )


def _noncomparable_agentic_decision(reason: str) -> Any:
    """Avoid a hidden relaxed policy path for pre-adapter failures."""

    from benchmark_harness.search import AgenticPolicyDecision

    return AgenticPolicyDecision(comparable=False, violations=(reason,))


def _upstream_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep only the exact frozen-upstream prediction-row fields for scoring."""

    return {
        key: row[key]
        for key in (
            "task_id",
            "category",
            "variant",
            "time_band",
            "agent",
            "model",
            "prompt",
            "raw_response",
            "api_error",
            "latency_seconds",
            "parsed",
            "metrics",
        )
    }
