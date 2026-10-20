"""Open Deep Research adapter gate with typed host-owned search injection."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from benchmark_harness.systems.contracts import SystemDescriptor
from benchmark_harness.systems.fixtures import is_fixture_source, verify_fixture_source
from benchmark_harness.systems.isolation import IsolationSpec
from benchmark_harness.systems.provenance import SystemSourceProvenance
from benchmark_harness.systems.regimes import RetrievalPolicy, build_retrieval_policy
from benchmark_harness.systems.runtime import V3HostAdapter
from benchmark_harness.systems.workers.guards import verify_odr_hooks

ODR_REPO_URL = "https://github.com/langchain-ai/open_deep_research"
ODR_REVISION = "1b7d2e80db9faa586165c60e09096dbbfd483a64"
ODR_TREE_SHA = "3bf2795b06fbc76447511365df3b2c02743019b7"
ODR_UV_LOCK_SHA256 = "ca1d96c3e7e9ed95e231773361a94a400a1942b725f1ab875c7c354795800791"
ODR_PYPROJECT_SHA256 = "57454bd01a58641b931f6879aa13030e798cfcdffe06c46ea22d606d4b667f6a"
ODR_LICENSE_SHA256 = "3404cb66935968edefe1149c9bfcb44bd8532c437c01ac095ec53b6e195ce110"
ODR_ADAPTER_ID = "odr-typed-host-v3"
ODR_ADAPTER_REVISION = "3b-typed-injection-v2"
# The pinned upstream pyproject and LICENSE both declare MIT.  Keep this
# separate from the fixture's byte hash: the fixture hash attests the exact
# bytes while this field records the license identity used by the manifest.
ODR_LICENSE_STATUS = "MIT"
ODR_MODEL_ROLES = ("summarization", "research", "compression", "final_report")

ODR_PROVENANCE = SystemSourceProvenance(
    repo_url=ODR_REPO_URL,
    revision=ODR_REVISION,
    tree_sha=ODR_TREE_SHA,
    dependency_lock_sha256=ODR_UV_LOCK_SHA256,
    project_sha256=ODR_PYPROJECT_SHA256,
    python_version="3.12",
    license_status=ODR_LICENSE_STATUS,
)

ODR_CONFIGURATION: dict[str, Any] = {
    "search_api": "none",
    "mcp_config": None,
    "allow_clarification": False,
    "native_search": False,
    "native_browser": False,
    "listener": False,
    "public_network": False,
    "telemetry": False,
    "compatibility_shim": "task_scoped_disposable_worker",
    "model_roles": list(ODR_MODEL_ROLES),
}


class OdrAdapter(V3HostAdapter):
    """Integration-only ODR bridge; it never contributes a ranking row."""

    adapter_id = ODR_ADAPTER_ID
    adapter_revision = ODR_ADAPTER_REVISION
    gate_phase = "3B"
    force_first_search = True

    def __init__(
        self,
        *,
        descriptor: SystemDescriptor,
        retrieval_policy: RetrievalPolicy | None = None,
        source_dir: str | Path | None = None,
        source_verifier: (
            Callable[[str | Path, SystemSourceProvenance], dict[str, Any]] | None
        ) = None,
        worker_exchange=None,
        worker_command: list[str] | None = None,
        isolation_spec: IsolationSpec | None = None,
        search_handler=None,
        search_token: str | None = None,
        model_responder=None,
        configuration: Mapping[str, Any] | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        policy = retrieval_policy or build_retrieval_policy(descriptor.retrieval_regime)
        root = Path(repo_root or Path.cwd()).resolve()
        resolved_source = Path(source_dir).resolve() if source_dir is not None else None
        resolved_command = list(worker_command) if worker_command is not None else None
        resolved_isolation = isolation_spec
        if worker_exchange is None and resolved_command is None and resolved_source is not None:
            resolved_command = [
                sys.executable,
                "-m",
                "benchmark_harness.systems.workers.odr",
                "--source",
                str(resolved_source),
            ]
            if is_fixture_source(resolved_source):
                resolved_command.append("--fixture")
        if (
            resolved_command is not None
            and resolved_isolation is None
            and resolved_source is not None
        ):
            resolved_isolation = IsolationSpec(
                source_dir=resolved_source,
                repo_root=root,
                loopback_url="http://127.0.0.1:1",
                python_executable=Path(sys.executable),
            )
        super().__init__(
            descriptor=descriptor,
            provenance=ODR_PROVENANCE,
            retrieval_policy=policy,
            model_roles=ODR_MODEL_ROLES,
            configuration={**ODR_CONFIGURATION, **dict(configuration or {})},
            source_dir=resolved_source,
            source_verifier=source_verifier,
            worker_exchange=worker_exchange,
            worker_command=resolved_command,
            isolation_spec=resolved_isolation,
            search_handler=search_handler,
            search_token=search_token,
            model_responder=model_responder,
            license_files={"LICENSE": ODR_LICENSE_SHA256},
            repo_root=root,
        )

    def _verify_worker_hooks(self) -> dict[str, Any]:
        if self.source_dir is None:
            return super()._verify_worker_hooks()
        return verify_odr_hooks(self.source_dir)

    def _verify_source(self) -> dict[str, Any]:
        if (
            self.source_verifier is None
            and self.source_dir is not None
            and is_fixture_source(self.source_dir)
        ):
            return verify_fixture_source(
                self.source_dir,
                self.provenance,
                system="open_deep_research",
                hook_verifier=verify_odr_hooks,
            )
        return super()._verify_source()

    def _validate_configuration(self) -> dict[str, bool]:
        checks = super()._validate_configuration()
        expected = {
            "search_api": "none",
            "mcp_config": None,
            "allow_clarification": False,
            "native_search": False,
            "native_browser": False,
            "listener": False,
            "public_network": False,
            "telemetry": False,
            "compatibility_shim": "task_scoped_disposable_worker",
        }
        for key, value in expected.items():
            checks[f"odr_{key}"] = self.configuration.get(key) == value
        if not all(checks[f"odr_{key}"] for key in expected):
            checks["odr_typed_injection"] = False
        else:
            checks["odr_typed_injection"] = True
        checks["odr_model_roles"] = (
            tuple(self.configuration.get("model_roles", ())) == ODR_MODEL_ROLES
        )
        return checks

    def _validate_hello(self, payload: Mapping[str, Any]) -> dict[str, bool]:
        checks = super()._validate_hello(payload)
        expected = {
            "search_api": "none",
            "mcp_config": None,
            "allow_clarification": False,
            "native_search": False,
            "native_browser": False,
            "listener": False,
            "public_network": False,
            "ranking_eligible": False,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"odr_worker_{key}_mismatch")
            checks[f"{key}_exact"] = True
        expected_search = "search_filter_http/v1" if self.search_handler is not None else None
        checks["typed_search_only"] = payload.get("injected_search") == expected_search
        if not checks["typed_search_only"]:
            raise RuntimeError("odr_worker_typed_search_missing")
        checks["pinned_tool_factory"] = (
            payload.get("pinned_factory") == "open_deep_research.utils.get_all_tools"
        )
        checks["pinned_langgraph"] = (
            payload.get("pinned_graph") == "open_deep_research.deep_researcher.deep_researcher"
        )
        if not checks["pinned_tool_factory"] or not checks["pinned_langgraph"]:
            raise RuntimeError("odr_pinned_runtime_missing")
        return checks


def build_odr_descriptor(
    *,
    system_id: str,
    framework_id: str = "open_deep_research",
    strategy_id: str = "odr-deep-research",
    backbone_id: str,
    backbone_training_cutoff: date,
    cutoff_source: str,
    profile: str = "deep_research",
    retrieval_regime: str = "cutoff_controlled",
    strategy_revision: str = ODR_REVISION,
) -> SystemDescriptor:
    """Build a descriptor while requiring caller-owned backbone/cutoff inputs."""

    return SystemDescriptor(
        system_id=system_id,
        framework_id=framework_id,
        framework_revision=ODR_REVISION,
        strategy_id=strategy_id,
        strategy_revision=strategy_revision,
        backbone_id=backbone_id,
        backbone_training_cutoff=backbone_training_cutoff,
        cutoff_source=cutoff_source,
        adapter_id=ODR_ADAPTER_ID,
        adapter_revision=ODR_ADAPTER_REVISION,
        profile=profile,
        retrieval_regime=retrieval_regime,
        ranking_eligible=False,
        license_status=ODR_LICENSE_STATUS,
    )


__all__ = [
    "ODR_ADAPTER_ID",
    "ODR_ADAPTER_REVISION",
    "ODR_CONFIGURATION",
    "ODR_LICENSE_SHA256",
    "ODR_LICENSE_STATUS",
    "ODR_MODEL_ROLES",
    "ODR_PROVENANCE",
    "ODR_REPO_URL",
    "ODR_REVISION",
    "ODR_TREE_SHA",
    "ODR_UV_LOCK_SHA256",
    "OdrAdapter",
    "build_odr_descriptor",
]
