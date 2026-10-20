"""Shared fail-closed host runtime for the ODR and GPT adapter gates."""

from __future__ import annotations

import re
import select
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark_harness.contracts import (
    AdapterResult,
    NetworkEvent,
    RunContext,
    TaskInput,
    ToolCall,
    sha256_json,
)
from benchmark_harness.external_harnesses.ipc import terminate_process_group
from benchmark_harness.systems.capabilities import TypedSearchCapability
from benchmark_harness.systems.clock import ClockPolicyError, DeclaredExperimentClock
from benchmark_harness.systems.contracts import (
    BudgetSpec,
    ExperimentBinding,
    PreflightResult,
    SystemDescriptor,
    TaskCapability,
)
from benchmark_harness.systems.external_stdio import (
    BudgetLedger,
    V3Envelope,
    V3ProtocolError,
    V3SearchRequest,
    V3SearchResponse,
    decode_v3,
    encode_v3,
)
from benchmark_harness.systems.isolation import IsolationSpec, build_isolated_environment
from benchmark_harness.systems.model_broker import HostModelBroker, HostModelBrokerError
from benchmark_harness.systems.provenance import (
    SystemProvenanceError,
    SystemSourceProvenance,
    verify_declared_file_hashes,
    verify_system_source,
)
from benchmark_harness.systems.regimes import RetrievalPolicy, RetrievalPolicyError
from benchmark_harness.systems.worker_diagnostics import (
    WorkerDiagnosticError,
    validate_worker_diagnostic,
    validate_worker_failure_subreason,
)
from benchmark_harness.temporal.policy import assert_no_temporal_leakage


class AdapterBoundaryError(RuntimeError):
    """Raised when a worker or host configuration crosses the adapter boundary."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic = dict(diagnostic) if diagnostic is not None else None


WorkerExchange = Callable[[V3Envelope | None], V3Envelope | None]
SourceVerifier = Callable[[str | Path, SystemSourceProvenance], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class AdapterGateReport:
    """Machine-readable gate evidence for one integration-only adapter."""

    adapter_id: str
    adapter_revision: str
    protocol: str
    status: str
    ready: bool
    ranking_eligible: bool
    phase: str = "integration"
    checks: dict[str, bool] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status != "integration_only":
            raise ValueError("adapter gate status must remain integration_only")
        if self.ranking_eligible:
            raise ValueError("adapter gates are never ranking eligible")
        if any(
            not isinstance(key, str) or not isinstance(value, bool)
            for key, value in self.checks.items()
        ):
            raise ValueError("adapter gate checks must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "adapter_revision": self.adapter_revision,
            "protocol": self.protocol,
            "status": self.status,
            "ready": self.ready,
            "ranking_eligible": False,
            "phase": self.phase,
            "checks": dict(sorted(self.checks.items())),
            "reasons": list(self.reasons),
            "provenance": dict(self.provenance),
            "policy": dict(self.policy),
        }


@dataclass(frozen=True, slots=True)
class AdapterRun:
    """Result plus the gate report used to authorize that result."""

    result: AdapterResult
    gate_report: AdapterGateReport


_FORBIDDEN_FLAG_KEYS = frozenset(
    {
        "native_search",
        "native_browser",
        "browser",
        "fetch",
        "listener",
        "public_network",
        "telemetry",
        "credentials",
        "provider_credentials",
        "mcp",
        "mcp_config",
        "clarification",
        "allow_clarification",
        "fallback_retriever",
    }
)
_WORKER_ERROR_NAME = re.compile(r"^worker_error:[A-Za-z_][A-Za-z0-9_]*$")


def _truthy_forbidden_flag(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if key == "mcp_config":
        return value is not None
    if key == "credentials" or key == "provider_credentials":
        return value is not None
    return bool(value)


def _safe_worker_error(value: Any) -> str:
    """Retain only the historical class-only worker error shape."""

    if isinstance(value, str) and _WORKER_ERROR_NAME.fullmatch(value):
        return value
    return "worker_task_error"


def _safe_worker_subreason(value: Any) -> str | None:
    """Validate the optional, allow-listed child failure token."""

    if value is None:
        return None
    try:
        return validate_worker_failure_subreason(value)
    except WorkerDiagnosticError as exc:
        raise AdapterBoundaryError("worker_error_subreason_invalid") from exc


def _safe_broker_failure_summary(broker: HostModelBroker) -> dict[str, Any]:
    """Expose only bounded, already-sanitized broker accounting on failure."""

    summary = broker.request_summary()
    return {
        key: summary[key]
        for key in (
            "call_count",
            "roles",
            "provider_response_model_names",
            "provider_model_names",
            "system_fingerprints",
            "requests_sha256",
            "forced_search_tool_name",
            "forced_search_synthesized_count",
            "forced_search_query_source",
            "forced_search_query_truncated_count",
        )
    }


class V3HostAdapter:
    """Typed host-owned execution seam shared by both researched frameworks.

    The runtime deliberately supports an in-memory ``worker_exchange`` seam
    for deterministic gate tests.  A production worker process can be wired
    through the same v3 envelopes, but neither native framework networking nor
    MCP is ever made available by this class.
    """

    adapter_id = "external-research-v3"
    adapter_revision = "v3"
    gate_phase = "integration"
    protocol = "external_research_stdio/v3"
    force_first_search = False

    def __init__(
        self,
        *,
        descriptor: SystemDescriptor,
        provenance: SystemSourceProvenance,
        retrieval_policy: RetrievalPolicy,
        model_roles: tuple[str, ...],
        configuration: Mapping[str, Any],
        source_dir: str | Path | None = None,
        source_verifier: SourceVerifier | None = None,
        worker_exchange: WorkerExchange | None = None,
        worker_command: list[str] | None = None,
        isolation_spec: IsolationSpec | None = None,
        process_factory: Callable[..., Any] | None = None,
        search_handler: (
            Callable[[str, str, int], Mapping[str, Any] | list[Mapping[str, Any]]] | None
        ) = None,
        search_token: str | None = None,
        model_responder: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
        license_files: Mapping[str, str] | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        self.descriptor = descriptor
        self.provenance = provenance
        self.retrieval_policy = retrieval_policy
        self.model_roles = tuple(dict.fromkeys(model_roles))
        self.configuration = dict(configuration)
        self.source_dir = Path(source_dir).resolve() if source_dir is not None else None
        self.source_verifier = source_verifier
        self.worker_exchange = worker_exchange
        self.worker_command = list(worker_command) if worker_command is not None else None
        self.isolation_spec = isolation_spec
        self.process_factory = process_factory or subprocess.Popen
        self._process: Any | None = None
        self._stderr_bytes = 0
        self.search_handler = search_handler
        self.search_token = search_token
        self.model_responder = model_responder
        self.license_files = dict(license_files or {})
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.last_gate_report: AdapterGateReport | None = None
        self._closed = False
        if descriptor.ranking_eligible:
            raise AdapterBoundaryError("external adapter descriptor cannot be ranking eligible")
        if descriptor.adapter_id != self.adapter_id:
            raise AdapterBoundaryError(
                "descriptor adapter_id does not match adapter implementation"
            )

    def _verify_source(self) -> dict[str, Any]:
        if self.source_verifier is not None:
            evidence = self.source_verifier(self.source_dir or self.repo_root, self.provenance)
        elif self.source_dir is not None:
            evidence = verify_system_source(self.source_dir, self.provenance)
        else:
            raise SystemProvenanceError("exact external source checkout is required")
        if not isinstance(evidence, dict):
            raise SystemProvenanceError("source verifier must return an object")
        if self.license_files:
            if self.source_dir is None:
                declared = evidence.get("license_files")
                if declared != self.license_files:
                    raise SystemProvenanceError(
                        "license-file evidence is missing from source verifier"
                    )
            else:
                evidence = {
                    **evidence,
                    "license_files": verify_declared_file_hashes(
                        self.source_dir, self.license_files
                    ),
                }
        return evidence

    def _validate_configuration(self) -> dict[str, bool]:
        return {
            "worker_configuration_declared": bool(self.configuration),
            "ranking_disabled": self.descriptor.ranking_eligible is False,
        }

    def _validate_hello(self, payload: Mapping[str, Any]) -> dict[str, bool]:
        if payload.get("protocol") != self.protocol:
            raise AdapterBoundaryError("worker_protocol_mismatch")
        if payload.get("adapter_id") != self.adapter_id:
            raise AdapterBoundaryError("worker_adapter_id_mismatch")
        if payload.get("child_network") != "forbidden":
            raise AdapterBoundaryError("worker_child_network_must_be_forbidden")
        for key in _FORBIDDEN_FLAG_KEYS:
            if key in payload and _truthy_forbidden_flag(payload, key):
                raise AdapterBoundaryError(f"worker_{key}_forbidden")
        if payload.get("mcp_enabled") is not False:
            raise AdapterBoundaryError("worker_mcp_must_be_disabled")
        return {
            "protocol_exact": True,
            "adapter_id_exact": True,
            "child_network_forbidden": True,
            "mcp_disabled": True,
        }

    def _configuration_payload(self, clock: DeclaredExperimentClock) -> dict[str, Any]:
        return {
            **self.configuration,
            "declared_experiment_clock": clock.to_dict(),
            "ranking_eligible": False,
        }

    def _postprocess_search_response(self, response: V3SearchResponse) -> V3SearchResponse:
        """Apply adapter-specific result policy before the worker sees results."""

        return response

    def _verify_worker_hooks(self) -> dict[str, Any]:
        """Verify adapter-specific worker hooks after source provenance.

        The generic runtime cannot know where an upstream framework exposes its
        hooks.  Concrete adapters override this for their pinned source trees;
        an attested callback remains supported for deterministic protocol tests.
        """

        return {"mode": "attested" if self.source_verifier is not None else "generic"}

    def preflight(self, binding: ExperimentBinding) -> PreflightResult:
        checks: dict[str, bool] = {}
        reasons: list[str] = []
        provenance: dict[str, Any] = {}
        try:
            checks["descriptor_hash"] = binding.descriptor_hash == self.descriptor.descriptor_hash
            if not checks["descriptor_hash"]:
                reasons.append("descriptor_hash_mismatch")
            checks["budget_binding"] = isinstance(binding.budget, BudgetSpec)
            if not checks["budget_binding"]:
                reasons.append("budget_binding_missing")
            checks["retrieval_regime"] = (
                binding.retrieval_regime == self.retrieval_policy.regime.value
            )
            if not checks["retrieval_regime"]:
                reasons.append("retrieval_regime_mismatch")
            checks["ranking_disabled"] = (
                binding.retrieval_regime == self.retrieval_policy.regime.value
                and self.descriptor.ranking_eligible is False
                and self.retrieval_policy.ranking_eligible is False
            )
            if not checks["ranking_disabled"]:
                reasons.append("ranking_must_remain_disabled")
            checks.update(self._validate_configuration())
            if not checks["worker_configuration_declared"]:
                reasons.append("worker_configuration_missing")
            for check_name, passed in checks.items():
                if not passed and check_name != "source_provenance":
                    reason = f"{check_name}_failed"
                    if reason not in reasons:
                        reasons.append(reason)
            if self.worker_exchange is None and self.worker_command is None:
                reasons.append("worker_exchange_missing")
                checks["worker_exchange"] = False
            else:
                checks["worker_exchange"] = True
            if self.worker_command is not None:
                checks["worker_isolation"] = self.isolation_spec is not None
                if not checks["worker_isolation"]:
                    reasons.append("worker_isolation_spec_missing")
            if binding.experiment_as_of is None:
                checks["declared_experiment_clock"] = False
                reasons.append("declared_experiment_clock_missing")
            else:
                DeclaredExperimentClock(binding.experiment_as_of)
                checks["declared_experiment_clock"] = True
            try:
                self.retrieval_policy.validate_capability(
                    search_handler_present=self.search_handler is not None
                )
                checks["retrieval_capability"] = True
            except RetrievalPolicyError as exc:
                checks["retrieval_capability"] = False
                reasons.append(str(exc))
            if self.retrieval_policy.is_leakage_probe:
                self.retrieval_policy.assert_live_execution_allowed()
            checks["offline_or_authorized_regime"] = True
            try:
                provenance = self._verify_source()
                provenance = {
                    **provenance,
                    "worker_hooks": self._verify_worker_hooks(),
                }
            except (SystemProvenanceError, ValueError) as exc:
                checks["source_provenance"] = False
                raise exc
            checks["source_provenance"] = True
            checks["worker_source_hooks"] = True
        except (
            AdapterBoundaryError,
            ClockPolicyError,
            RetrievalPolicyError,
            SystemProvenanceError,
            ValueError,
        ) as exc:
            if "worker_hook" in str(exc):
                checks["worker_source_hooks"] = False
            checks.setdefault("source_provenance", False if "source" in str(exc) else True)
            if str(exc) not in reasons:
                reasons.append(str(exc))
        report = AdapterGateReport(
            adapter_id=self.adapter_id,
            adapter_revision=self.adapter_revision,
            protocol=self.protocol,
            status="integration_only",
            ready=not reasons,
            ranking_eligible=False,
            phase=self.gate_phase,
            checks=checks,
            reasons=tuple(reasons),
            provenance=provenance,
            policy=self.retrieval_policy.to_dict(),
        )
        self.last_gate_report = report
        return PreflightResult(
            ready=report.ready,
            reasons=report.reasons,
            provenance=report.to_dict(),
        )

    def run_task(
        self,
        task: TaskInput,
        context: RunContext,
        capability: TaskCapability,
        binding: ExperimentBinding | None = None,
    ) -> AdapterResult:
        """Run one task through the typed v3 seam after verify-only preflight."""

        if self._closed:
            raise AdapterBoundaryError("adapter_is_closed")
        if binding is None:
            raise AdapterBoundaryError("experiment_binding_required")
        preflight = self.preflight(binding)
        if not preflight.ready:
            raise AdapterBoundaryError("adapter_preflight_failed:" + ",".join(preflight.reasons))
        if capability.budget != binding.budget:
            raise AdapterBoundaryError("capability_budget_mismatch")
        if task.task_id != binding.task_id:
            raise AdapterBoundaryError("task_binding_mismatch")
        if self.retrieval_policy.regime.value == "cutoff_controlled" and not task.cutoff_date:
            # A cutoff-controlled child cannot be given a synthetic date for
            # atemporal/counterfactual rows.  Keep the policy fail-closed and
            # let the executor record the non-comparable cell explicitly.
            raise AdapterBoundaryError("cutoff_date_required_for_cutoff_controlled")
        try:
            clock = DeclaredExperimentClock(binding.experiment_as_of)
        except ClockPolicyError as exc:  # pragma: no cover - preflight already checks this.
            raise AdapterBoundaryError(str(exc)) from exc
        task_context = context.for_task(task.task_id)
        ledger = BudgetLedger(binding.budget)
        broker = HostModelBroker(
            task_context.model,
            task_context,
            ledger,
            self.model_roles,
            responder=self.model_responder,
            force_first_search=self.force_first_search,
        )
        search = None
        if self.search_handler is not None:
            token = self.search_token or capability.service_token
            if not token:
                raise AdapterBoundaryError("search_capability_token_missing")
            search = TypedSearchCapability(
                self.search_handler,
                capability_token=token,
                ledger=ledger,
            )
        request_id = uuid.uuid4().hex
        host_send_seq = 0
        worker_last_seq: int | None = None
        trajectory_hashes: list[str] = []
        started = time.monotonic()
        configuration = self._configuration_payload(clock)
        start_payload = {
            "protocol": self.protocol,
            "adapter_id": self.adapter_id,
            "adapter_revision": self.adapter_revision,
            "task_id": task.task_id,
            "prompt": task.prompt,
            "cutoff_date": task.cutoff_date,
            "experiment_as_of": clock.isoformat(),
            "clock": clock.to_dict(),
            "configuration": configuration,
            "policy": self.retrieval_policy.to_dict(),
            "model_broker": broker.public_manifest(),
            "search_capability": search.public_manifest() if search else None,
            "mcp_config": None,
            "allow_clarification": False,
        }
        assert_no_temporal_leakage(start_payload)
        outgoing = V3Envelope(
            type="task.start",
            request_id=request_id,
            task_id=task.task_id,
            seq=host_send_seq,
            binding_hash=binding.plan_hash,
            payload=start_payload,
        )
        child = self._exchange(outgoing, started, binding)
        worker_last_seq = self._validate_worker_envelope(child, binding, worker_last_seq)
        if child.type != "hello":
            raise AdapterBoundaryError("worker_hello_required")
        try:
            hello_checks = self._validate_hello(child.payload)
        except Exception as exc:
            raise AdapterBoundaryError(str(exc)) from exc
        checks = dict(self.last_gate_report.checks if self.last_gate_report else {})
        checks.update({f"hello_{key}": value for key, value in hello_checks.items()})
        child = self._exchange(None, started, binding)
        worker_last_seq = self._validate_worker_envelope(child, binding, worker_last_seq)
        while True:
            if time.monotonic() - started > binding.budget.wall_clock_timeout_seconds:
                raise AdapterBoundaryError("worker_timeout")
            if child.type == "model.request":
                self._reject_forbidden_child_flags(child.payload)
                try:
                    response = broker.request(child.payload)
                except HostModelBrokerError as exc:
                    raise AdapterBoundaryError(str(exc), diagnostic=exc.diagnostic) from exc
                host_send_seq += 1
                outgoing = self._host_envelope(
                    "model.response", response, child.request_id, task, binding, host_send_seq
                )
                child = self._exchange(outgoing, started, binding)
            elif child.type == "search.request":
                if search is None:
                    raise AdapterBoundaryError("search_capability_not_granted")
                try:
                    request = V3SearchRequest.from_payload(child.payload)
                    response = search.invoke(request.query, max_results=request.max_results)
                    response = self._postprocess_search_response(response)
                    if response.error is not None:
                        raise AdapterBoundaryError("search_capability_error:" + response.error)
                except (V3ProtocolError, RuntimeError) as exc:
                    raise AdapterBoundaryError(str(exc)) from exc
                host_send_seq += 1
                outgoing = self._host_envelope(
                    "search.response",
                    response.to_dict(),
                    child.request_id,
                    task,
                    binding,
                    host_send_seq,
                )
                child = self._exchange(outgoing, started, binding)
            elif child.type == "trajectory.event":
                self._reject_forbidden_child_flags(child.payload)
                trajectory_hashes.append(sha256_json(child.payload))
                task_context.emit(
                    "trajectory_event",
                    control_flow_node=self.adapter_id,
                    attributes={"payload_sha256": trajectory_hashes[-1]},
                )
                child = self._exchange(None, started, binding)
            elif child.type == "task.result":
                # Validate the terminal frame before accepting it.  A result
                # is still worker-controlled data and must carry the same
                # task/binding identity and monotonic sequence as every other
                # frame on the channel.
                worker_last_seq = self._validate_worker_envelope(child, binding, worker_last_seq)
                break
            elif child.type == "task.error":
                error = _safe_worker_error(child.payload.get("error"))
                sub_reason = _safe_worker_subreason(child.payload.get("sub_reason"))
                if sub_reason is not None and error != "worker_error:RuntimeError":
                    raise AdapterBoundaryError("worker_error_subreason_mismatch")
                raw_diagnostic = child.payload.get("diagnostic")
                if raw_diagnostic is not None:
                    if error != "worker_error:RuntimeError":
                        raise AdapterBoundaryError("worker_error_diagnostic_mismatch")
                    try:
                        diagnostic = validate_worker_diagnostic(raw_diagnostic)
                    except WorkerDiagnosticError as exc:
                        raise AdapterBoundaryError("worker_error_diagnostic_invalid") from exc
                    if diagnostic["broker_call_ordinal"] > broker.call_count:
                        raise AdapterBoundaryError("worker_error_diagnostic_ordinal_invalid")
                    if sub_reason is not None:
                        diagnostic["sub_reason"] = sub_reason
                    diagnostic["broker_summary"] = _safe_broker_failure_summary(broker)
                    raise AdapterBoundaryError(error, diagnostic=diagnostic)
                if sub_reason is not None:
                    raise AdapterBoundaryError(
                        error,
                        diagnostic={
                            "sub_reason": sub_reason,
                            "broker_summary": _safe_broker_failure_summary(broker),
                        },
                    )
                raise AdapterBoundaryError(error)
            else:
                raise AdapterBoundaryError(f"unexpected_worker_message:{child.type}")
            if child.type != "task.result":
                worker_last_seq = self._validate_worker_envelope(child, binding, worker_last_seq)

        return self._finish_result(
            task=task,
            context=task_context,
            payload=child.payload,
            broker=broker,
            search=search,
            trajectory_hashes=trajectory_hashes,
            checks=checks,
            started=started,
            clock=clock,
        )

    def _host_envelope(
        self,
        message_type: str,
        payload: Mapping[str, Any],
        request_id: str,
        task: TaskInput,
        binding: ExperimentBinding,
        seq: int,
    ) -> V3Envelope:
        value = dict(payload)
        assert_no_temporal_leakage(value)
        return V3Envelope(
            type=message_type,
            request_id=request_id,
            task_id=task.task_id,
            seq=seq,
            binding_hash=binding.plan_hash,
            payload=value,
        )

    def _exchange(
        self,
        outgoing: V3Envelope | None,
        started: float,
        binding: ExperimentBinding,
    ) -> V3Envelope:
        if self.worker_exchange is None and self.worker_command is None:
            raise AdapterBoundaryError("worker_exchange_missing")
        if time.monotonic() - started > binding.budget.wall_clock_timeout_seconds:
            raise AdapterBoundaryError("worker_timeout")
        if self.worker_exchange is not None:
            try:
                incoming = self.worker_exchange(outgoing)
            except Exception as exc:
                raise AdapterBoundaryError(f"worker_exchange_error:{type(exc).__name__}") from exc
        else:
            incoming = self._process_exchange(outgoing, started, binding)
        if not isinstance(incoming, V3Envelope):
            raise AdapterBoundaryError("worker_exchange_must_return_v3_envelope")
        return incoming

    def _start_process(self) -> None:
        if self.worker_command is None:
            raise AdapterBoundaryError("worker_command_missing")
        if self.isolation_spec is None:
            raise AdapterBoundaryError("worker_isolation_spec_missing")
        environment = build_isolated_environment(self.isolation_spec)
        try:
            self._process = self.process_factory(
                self.worker_command,
                cwd=str(self.repo_root),
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except (OSError, TypeError) as exc:
            raise AdapterBoundaryError("worker_start_failed") from exc

    def _process_exchange(
        self,
        outgoing: V3Envelope | None,
        started: float,
        binding: ExperimentBinding,
    ) -> V3Envelope:
        if self._process is None:
            self._start_process()
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise AdapterBoundaryError("worker_stdio_unavailable")
        if outgoing is not None:
            try:
                process.stdin.write(encode_v3(outgoing))
                process.stdin.flush()
            except (OSError, TypeError, ValueError) as exc:
                raise AdapterBoundaryError("worker_write_failed") from exc
        remaining = binding.budget.wall_clock_timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            self._terminate_process()
            raise AdapterBoundaryError("worker_timeout")
        streams = [process.stdout]
        if process.stderr is not None:
            streams.append(process.stderr)
        try:
            ready, _, _ = select.select(streams, [], [], remaining)
        except (OSError, TypeError) as exc:
            self._terminate_process()
            raise AdapterBoundaryError("worker_select_failed") from exc
        if process.stderr is not None and process.stderr in ready:
            reader = getattr(process.stderr, "read1", process.stderr.read)
            chunk = reader(64 * 1024)
            self._stderr_bytes += len(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)
            if self._stderr_bytes > 1024 * 1024:
                self._terminate_process()
                raise AdapterBoundaryError("worker_stderr_budget_exceeded")
        if process.stdout not in ready:
            self._terminate_process()
            raise AdapterBoundaryError("worker_timeout")
        raw = process.stdout.readline()
        if not raw:
            self._terminate_process()
            raise AdapterBoundaryError("worker_eof")
        try:
            return decode_v3(raw)
        except V3ProtocolError as exc:
            self._terminate_process()
            raise AdapterBoundaryError(str(exc)) from exc

    def _terminate_process(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                terminate_process_group(process)
            else:
                process.wait(timeout=2.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        finally:
            self._process = None

    @staticmethod
    def _validate_worker_envelope(
        envelope: V3Envelope,
        binding: ExperimentBinding,
        last_seq: int | None,
    ) -> int:
        if envelope.task_id != binding.task_id or envelope.binding_hash != binding.plan_hash:
            raise AdapterBoundaryError("worker_binding_mismatch")
        if last_seq is not None and envelope.seq <= last_seq:
            raise AdapterBoundaryError("worker_sequence_not_monotonic")
        return envelope.seq

    @staticmethod
    def _reject_forbidden_child_flags(payload: Mapping[str, Any]) -> None:
        for key in _FORBIDDEN_FLAG_KEYS:
            if key in payload and _truthy_forbidden_flag(payload, key):
                raise AdapterBoundaryError(f"worker_{key}_forbidden")

    def _finish_result(
        self,
        *,
        task: TaskInput,
        context: RunContext,
        payload: Mapping[str, Any],
        broker: HostModelBroker,
        search: TypedSearchCapability | None,
        trajectory_hashes: list[str],
        checks: Mapping[str, bool],
        started: float,
        clock: DeclaredExperimentClock,
    ) -> AdapterResult:
        prediction = payload.get("prediction", payload.get("raw_text"))
        if prediction is not None and not isinstance(prediction, str):
            raise AdapterBoundaryError("worker_prediction_must_be_string")
        raw_text = prediction if isinstance(prediction, str) else None
        messages = payload.get("messages")
        if not isinstance(messages, list):
            messages = (
                broker.requests[0].get("messages", task.messages)
                if broker.requests
                else task.messages
            )
        effective_messages = tuple(
            value
            for value in messages
            if isinstance(value, dict)
            and isinstance(value.get("role"), str)
            and isinstance(value.get("content"), str)
        )
        if not effective_messages:
            effective_messages = tuple(task.messages)
        effective_hash = sha256_json(list(effective_messages))
        raw_citations = payload.get("citations", [])
        if not isinstance(raw_citations, list | tuple) or any(
            not isinstance(citation, str) or not citation.strip() for citation in raw_citations
        ):
            raise AdapterBoundaryError("worker_citations_malformed")
        citations = tuple(citation.strip() for citation in raw_citations)
        for index, request in enumerate(broker.requests, start=1):
            context.emit(
                "model_request",
                model_call_index=index,
                effective_input_hash=effective_hash,
                control_flow_node="parent_model_broker",
                attributes={"request_sha256": sha256_json(request)},
            )
            context.emit(
                "model_response",
                model_call_index=index,
                effective_input_hash=effective_hash,
                control_flow_node="parent_model_broker",
            )
        tool_calls = tuple(
            ToolCall(
                name="search",
                arguments_hash=call.query_sha256,
                result_hash=call.receipt_sha256,
            )
            for call in (search.calls if search else ())
        )
        for call in tool_calls:
            context.emit(
                "tool_call",
                effective_input_hash=effective_hash,
                control_flow_node=self.adapter_id,
                tool_name=call.name,
                tool_arguments_hash=call.arguments_hash,
                tool_result_hash=call.result_hash,
            )
        context.emit(
            "task_finished",
            effective_input_hash=effective_hash,
            control_flow_node=self.adapter_id,
            latency_seconds=time.monotonic() - started,
            error=None if raw_text is not None else "empty_worker_result",
        )
        ledger = broker.ledger
        usage = payload.get("usage") if isinstance(payload.get("usage"), Mapping) else {}
        usage = {
            **dict(usage),
            "model_calls": ledger.model_calls,
            "search_calls": ledger.search_calls,
            "input_tokens": ledger.input_tokens,
            "output_tokens": ledger.output_tokens,
        }
        provenance = self.last_gate_report.provenance if self.last_gate_report else {}
        model_provenance = context.model.provenance_dict(
            provider_response_model_names=broker.provider_model_names,
            system_fingerprints=broker.system_fingerprints,
        )
        effective_request = {
            "adapter_id": self.adapter_id,
            "adapter_revision": self.adapter_revision,
            "protocol": self.protocol,
            "framework": self.descriptor.framework_id,
            "policy": self.retrieval_policy.to_dict(),
            "clock": clock.to_dict(),
            "configuration": dict(self.configuration),
            "model_broker": broker.request_summary(),
            "search_capability": search.public_manifest() if search else None,
            "search_invocations": [call.to_dict() for call in (search.calls if search else ())],
            "citations": list(citations),
            "citation_continuity": {
                "search_receipts": [
                    call.receipt_sha256 for call in (search.calls if search else ())
                ],
                "trajectory_event_sha256s": list(trajectory_hashes),
            },
            "trajectory_event_sha256s": list(trajectory_hashes),
            "model_provenance": model_provenance,
            "budget": {
                "limits": ledger.budget.to_dict(),
                "used": {
                    "model_calls": ledger.model_calls,
                    "search_calls": ledger.search_calls,
                    "visit_calls": ledger.visit_calls,
                    "other_tool_calls": ledger.other_tool_calls,
                    "input_tokens": ledger.input_tokens,
                    "output_tokens": ledger.output_tokens,
                },
            },
            "gate_checks": dict(checks),
            "provenance": provenance,
            "ranking_eligible": False,
        }
        network_events: tuple[NetworkEvent, ...] = tuple(broker.network_events)
        return AdapterResult(
            raw_text=raw_text,
            usage=usage,
            latency_seconds=time.monotonic() - started,
            canonical_input_hash=task.input_hash,
            effective_input_hash=effective_hash,
            effective_messages=effective_messages,
            effective_request=effective_request,
            model_call_count=broker.call_count,
            tool_calls=tool_calls,
            network_events=network_events,
            trace_events=context.events,
            trace_artifact_ref="trajectory_events.jsonl",
            error=payload.get("error") if isinstance(payload.get("error"), str) else None,
        )

    def close(self) -> None:
        self._closed = True
        self._terminate_process()


__all__ = [
    "AdapterBoundaryError",
    "AdapterGateReport",
    "AdapterRun",
    "V3HostAdapter",
    "WorkerExchange",
]
