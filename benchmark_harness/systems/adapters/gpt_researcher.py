"""GPT-Researcher adapter gate with custom-retriever, non-ranking policy."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

from benchmark_harness.systems.contracts import SystemDescriptor
from benchmark_harness.systems.external_stdio import V3SearchResponse
from benchmark_harness.systems.fixtures import is_fixture_source, verify_fixture_source
from benchmark_harness.systems.isolation import IsolationSpec
from benchmark_harness.systems.provenance import (
    SystemSourceProvenance,
    verify_declared_file_hashes,
    verify_system_source,
)
from benchmark_harness.systems.regimes import RetrievalPolicy, build_retrieval_policy
from benchmark_harness.systems.runtime import V3HostAdapter
from benchmark_harness.systems.workers.guards import verify_gpt_researcher_hooks

GPT_RESEARCHER_REPO_URL = "https://github.com/assafelovic/gpt-researcher"
GPT_RESEARCHER_REVISION = "5d84d2f5553e70a2765a8ff3a0d2672d60437ce8"
GPT_RESEARCHER_TREE_SHA = "ae751c3f519cf69a72e758756d9f845960c701e9"
GPT_RESEARCHER_UV_LOCK_SHA256 = "bad03a459c52521301ad9445d27ed6b83c146c0d481eee57a0a974e2ccea6e3a"
GPT_RESEARCHER_PYPROJECT_SHA256 = "f6b7735d8d386fe6ba72ae687319e3bdc3f600aa9da55d82af954990cfeb4883"
GPT_RESEARCHER_PACKAGE_METADATA_SHA256 = GPT_RESEARCHER_PYPROJECT_SHA256
GPT_RESEARCHER_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
GPT_RESEARCHER_ADAPTER_ID = "gpt-researcher-typed-host-v3"
GPT_RESEARCHER_ADAPTER_REVISION = "3c-custom-retriever-v1"
GPT_RESEARCHER_LICENSE_STATUS = "CONFLICTING"
GPT_RESEARCHER_MODEL_ROLES = ("fast", "smart", "strategic")
GPT_RESEARCHER_BENCHMARK_LOCK = "configs/external/gpt-researcher/uv.lock"
GPT_RESEARCHER_BENCHMARK_PROJECT = "configs/external/gpt-researcher/pyproject.toml"
GPT_RESEARCHER_BENCHMARK_PROJECT_SHA256 = (
    "fe6cd2d15a89dfa74efc3ef3bf94c5ce0a5c7ad110d5385e6cbc0d3a8249ce04"
)
GPT_RESEARCHER_LOCK_METADATA = "configs/external/gpt-researcher/LOCK_METADATA.json"
GPT_RESEARCHER_LOCK_METADATA_SHA256 = (
    "5cc9c46a9462a58c978de2899a8c4938bf591a43936b3997090677c12960826c"
)

GPT_RESEARCHER_PROVENANCE = SystemSourceProvenance(
    repo_url=GPT_RESEARCHER_REPO_URL,
    revision=GPT_RESEARCHER_REVISION,
    tree_sha=GPT_RESEARCHER_TREE_SHA,
    dependency_lock_sha256=GPT_RESEARCHER_UV_LOCK_SHA256,
    project_sha256=GPT_RESEARCHER_PYPROJECT_SHA256,
    python_version="3.12",
    license_status=GPT_RESEARCHER_LICENSE_STATUS,
)

GPT_RESEARCHER_CONFIGURATION: dict[str, Any] = {
    "retriever": "custom",
    "fallback_retriever": False,
    "scraper": "disabled",
    "embeddings": "disabled",
    "source_curation": "disabled",
    "telemetry": False,
    "mcp_config": None,
    "native_browser": False,
    "public_network": False,
    "listener": False,
    "clock_injection": "declared_experiment_clock/v1",
    "clock_compatibility_shim": "task_scoped_disposable_worker",
    "benchmark_lock_path": GPT_RESEARCHER_BENCHMARK_LOCK,
    "benchmark_lock_metadata_path": GPT_RESEARCHER_LOCK_METADATA,
    "benchmark_lock_project_path": GPT_RESEARCHER_BENCHMARK_PROJECT,
    "benchmark_lock_python": "3.12",
    "benchmark_lock_runtime": "3.12.13",
    "benchmark_lock_resolver": "uv",
    "benchmark_lock_sync": "verify_only",
    "model_roles": list(GPT_RESEARCHER_MODEL_ROLES),
}


class GptResearcherAdapter(V3HostAdapter):
    """Integration-only GPT-Researcher bridge; ranking is explicitly disabled."""

    adapter_id = GPT_RESEARCHER_ADAPTER_ID
    adapter_revision = GPT_RESEARCHER_ADAPTER_REVISION
    gate_phase = "3C"

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
        benchmark_lock_path: str | Path | None = None,
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
                "benchmark_harness.systems.workers.gpt_researcher",
                "--source",
                str(resolved_source),
                "--repo-root",
                str(root),
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
        self.benchmark_lock_path = (
            Path(benchmark_lock_path).resolve()
            if benchmark_lock_path is not None
            else root / GPT_RESEARCHER_BENCHMARK_LOCK
        )
        super().__init__(
            descriptor=descriptor,
            provenance=GPT_RESEARCHER_PROVENANCE,
            retrieval_policy=policy,
            model_roles=GPT_RESEARCHER_MODEL_ROLES,
            configuration={**GPT_RESEARCHER_CONFIGURATION, **dict(configuration or {})},
            source_dir=resolved_source,
            source_verifier=source_verifier,
            worker_exchange=worker_exchange,
            worker_command=resolved_command,
            isolation_spec=resolved_isolation,
            search_handler=search_handler,
            search_token=search_token,
            model_responder=model_responder,
            license_files={"LICENSE": GPT_RESEARCHER_LICENSE_SHA256},
            repo_root=root,
        )

    def _verify_worker_hooks(self) -> dict[str, Any]:
        if self.source_dir is None:
            return super()._verify_worker_hooks()
        return verify_gpt_researcher_hooks(self.source_dir)

    def _verify_source(self) -> dict[str, Any]:
        if self.source_verifier is not None:
            evidence = self.source_verifier(self.source_dir or self.repo_root, self.provenance)
        elif self.source_dir is not None and is_fixture_source(self.source_dir):
            evidence = verify_fixture_source(
                self.source_dir,
                self.provenance,
                system="gpt_researcher",
                hook_verifier=verify_gpt_researcher_hooks,
            )
        elif self.source_dir is not None:
            # GPT-Researcher did not ship the benchmark-owned Python 3.12 lock
            # at this source revision.  Verify the immutable source tree and
            # its project metadata, then verify our generated lock separately.
            evidence = verify_system_source(
                self.source_dir,
                self.provenance,
                lock_filename=None,
                project_filename="pyproject.toml",
            )
        else:
            raise ValueError("exact GPT-Researcher source checkout is required")
        if not isinstance(evidence, dict):
            raise ValueError("source verifier must return an object")
        if self.source_dir is not None:
            evidence = {
                **evidence,
                "license_files": verify_declared_file_hashes(
                    self.source_dir, {"LICENSE": GPT_RESEARCHER_LICENSE_SHA256}
                ),
            }
        elif evidence.get("license_files") != {"LICENSE": GPT_RESEARCHER_LICENSE_SHA256}:
            raise ValueError("GPT-Researcher license evidence is missing")
        relative_lock = self.benchmark_lock_path.relative_to(self.repo_root)
        relative_project = (
            (self.repo_root / GPT_RESEARCHER_BENCHMARK_PROJECT)
            .resolve()
            .relative_to(self.repo_root)
        )
        relative_metadata = (
            (self.repo_root / GPT_RESEARCHER_LOCK_METADATA).resolve().relative_to(self.repo_root)
        )
        benchmark_files = verify_declared_file_hashes(
            self.repo_root,
            {
                relative_lock.as_posix(): GPT_RESEARCHER_UV_LOCK_SHA256,
                relative_project.as_posix(): GPT_RESEARCHER_BENCHMARK_PROJECT_SHA256,
                relative_metadata.as_posix(): GPT_RESEARCHER_LOCK_METADATA_SHA256,
            },
        )
        evidence = {
            **evidence,
            "benchmark_lock": {
                "path": relative_lock.as_posix(),
                "python_version": "3.12",
                "runtime": "3.12.13",
                "resolver": "uv",
                "project_input": relative_project.as_posix(),
                "metadata_path": relative_metadata.as_posix(),
                "sync": "verify_only",
                "sha256": benchmark_files[relative_lock.as_posix()],
                "project_sha256": benchmark_files[relative_project.as_posix()],
                "metadata_sha256": benchmark_files[relative_metadata.as_posix()],
            },
            "package_metadata": {
                "path": "pyproject.toml",
                "sha256": GPT_RESEARCHER_PACKAGE_METADATA_SHA256,
                "license": "MIT",
                "license_sha256": GPT_RESEARCHER_LICENSE_SHA256,
            },
            "license_status": GPT_RESEARCHER_LICENSE_STATUS,
            "license_disposition": "adapter_allowed_ranking_blocked",
        }
        return evidence

    def _validate_configuration(self) -> dict[str, bool]:
        checks = super()._validate_configuration()
        expected = {
            "retriever": "custom",
            "fallback_retriever": False,
            "scraper": "disabled",
            "embeddings": "disabled",
            "source_curation": "disabled",
            "telemetry": False,
            "mcp_config": None,
            "native_browser": False,
            "public_network": False,
            "listener": False,
            "clock_injection": "declared_experiment_clock/v1",
            "clock_compatibility_shim": "task_scoped_disposable_worker",
            "benchmark_lock_path": GPT_RESEARCHER_BENCHMARK_LOCK,
            "benchmark_lock_metadata_path": GPT_RESEARCHER_LOCK_METADATA,
            "benchmark_lock_project_path": GPT_RESEARCHER_BENCHMARK_PROJECT,
            "benchmark_lock_python": "3.12",
            "benchmark_lock_runtime": "3.12.13",
            "benchmark_lock_resolver": "uv",
            "benchmark_lock_sync": "verify_only",
        }
        for key, value in expected.items():
            checks[f"gpt_{key}"] = self.configuration.get(key) == value
        checks["gpt_roles"] = (
            tuple(self.configuration.get("model_roles", ())) == GPT_RESEARCHER_MODEL_ROLES
        )
        checks["gpt_conflicting_license_nonranking"] = (
            self.descriptor.license_status == GPT_RESEARCHER_LICENSE_STATUS
            and self.descriptor.ranking_eligible is False
        )
        return checks

    def _validate_hello(self, payload: Mapping[str, Any]) -> dict[str, bool]:
        checks = super()._validate_hello(payload)
        expected = {
            "retriever": "custom",
            "retriever_env": "custom",
            "fallback_retriever": False,
            "scraper": "disabled",
            "embeddings": "disabled",
            "source_curation": "disabled",
            "telemetry": False,
            "mcp_config": None,
            "native_browser": False,
            "public_network": False,
            "listener": False,
            "clock_injection": "declared_experiment_clock/v1",
            "license_status": GPT_RESEARCHER_LICENSE_STATUS,
            "ranking_eligible": False,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"gpt_worker_{key}_mismatch")
            checks[f"{key}_exact"] = True
        expected_search = "search_filter_http/v1" if self.search_handler is not None else None
        checks["typed_search_only"] = payload.get("injected_search") == expected_search
        if not checks["typed_search_only"]:
            raise RuntimeError("gpt_worker_typed_search_missing")
        checks["pinned_deep_path"] = (
            payload.get("pinned_deep_path")
            == "gpt_researcher.skills.deep_research.DeepResearchSkill"
        )
        checks["pinned_agent"] = payload.get("pinned_agent") == "gpt_researcher.agent.GPTResearcher"
        if not checks["pinned_deep_path"] or not checks["pinned_agent"]:
            raise RuntimeError("gpt_pinned_runtime_missing")
        return checks

    def _postprocess_search_response(self, response: V3SearchResponse) -> V3SearchResponse:
        """Accept only retriever-provided content; never turn URLs into fetches."""

        if response.error is not None:
            return response
        normalized: list[dict[str, Any]] = []
        for result in response.results:
            item = dict(result)
            raw_content = item.get("raw_content")
            if not isinstance(raw_content, str) or len(raw_content.strip()) < 100:
                # Preserve the result identity but make the missing/short
                # content explicit.  The disabled scraper cannot fill it in.
                item["raw_content"] = ""
                item["content_status"] = "empty_or_short"
            else:
                item["content_status"] = "accepted_direct"
            normalized.append(item)
        return V3SearchResponse(
            results=tuple(normalized),
            receipt_sha256=response.receipt_sha256,
            call_index=response.call_index,
            error=response.error,
        )


def build_gpt_researcher_descriptor(
    *,
    system_id: str,
    framework_id: str = "gpt_researcher",
    strategy_id: str = "gpt-researcher",
    backbone_id: str,
    backbone_training_cutoff: date,
    cutoff_source: str,
    profile: str = "deep_research",
    retrieval_regime: str = "cutoff_controlled",
    strategy_revision: str = GPT_RESEARCHER_REVISION,
) -> SystemDescriptor:
    """Build a descriptor while requiring caller-owned backbone/cutoff inputs."""

    return SystemDescriptor(
        system_id=system_id,
        framework_id=framework_id,
        framework_revision=GPT_RESEARCHER_REVISION,
        strategy_id=strategy_id,
        strategy_revision=strategy_revision,
        backbone_id=backbone_id,
        backbone_training_cutoff=backbone_training_cutoff,
        cutoff_source=cutoff_source,
        adapter_id=GPT_RESEARCHER_ADAPTER_ID,
        adapter_revision=GPT_RESEARCHER_ADAPTER_REVISION,
        profile=profile,
        retrieval_regime=retrieval_regime,
        ranking_eligible=False,
        license_status=GPT_RESEARCHER_LICENSE_STATUS,
    )


__all__ = [
    "GPT_RESEARCHER_ADAPTER_ID",
    "GPT_RESEARCHER_ADAPTER_REVISION",
    "GPT_RESEARCHER_BENCHMARK_LOCK",
    "GPT_RESEARCHER_BENCHMARK_PROJECT",
    "GPT_RESEARCHER_BENCHMARK_PROJECT_SHA256",
    "GPT_RESEARCHER_CONFIGURATION",
    "GPT_RESEARCHER_LICENSE_SHA256",
    "GPT_RESEARCHER_LICENSE_STATUS",
    "GPT_RESEARCHER_MODEL_ROLES",
    "GPT_RESEARCHER_LOCK_METADATA",
    "GPT_RESEARCHER_LOCK_METADATA_SHA256",
    "GPT_RESEARCHER_PACKAGE_METADATA_SHA256",
    "GPT_RESEARCHER_PROVENANCE",
    "GPT_RESEARCHER_REPO_URL",
    "GPT_RESEARCHER_REVISION",
    "GPT_RESEARCHER_TREE_SHA",
    "GPT_RESEARCHER_UV_LOCK_SHA256",
    "GptResearcherAdapter",
    "build_gpt_researcher_descriptor",
]
