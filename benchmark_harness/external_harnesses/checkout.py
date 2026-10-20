"""Explicit FinanceHarness checkout and isolated Python environment manager."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmark_harness.config import DeepResearchConfig, ExternalHarnessConfig
from benchmark_harness.contracts import sha256_json, sha256_text
from benchmark_harness.external_harnesses.financeharness.policy import (
    PURE_DEFERRED_TOOL_PREFIXES,
    SAFE_TOOL_NAMES,
)
from benchmark_harness.external_harnesses.ipc import ENVELOPE_FIELDS, IPC_PROTOCOL
from benchmark_harness.external_harnesses.provenance import (
    ExternalHarnessIntegrityError,
    read_manifest,
    run_git,
    verify_git_checkout,
    write_manifest,
)


class ExternalHarnessError(RuntimeError):
    """Raised when a synchronized external harness cannot be used safely."""


class ExternalHarnessManager:
    """Manage a pinned checkout without doing hidden work during benchmark runs."""

    MANIFEST_NAME = ".benchmark-harness-manifest.json"

    def __init__(
        self,
        config: ExternalHarnessConfig,
        *,
        deep_config: DeepResearchConfig | None = None,
        repo_root: str | Path | None = None,
        command_runner: Any | None = None,
    ) -> None:
        self.config = config
        self.deep_config = deep_config or DeepResearchConfig()
        self.repo_root = Path(repo_root or Path.cwd()).resolve()
        self.checkout_root = Path(config.checkout_root).resolve()
        self.source_path = self.checkout_root / config.adapter
        self.source_dir = self.source_path.resolve()
        self.venv_dir = self.source_dir / ".venv"
        self.venv_python = self.venv_dir / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        # Keep host provenance beside (not inside) the clean upstream worktree;
        # an untracked manifest would make ``git status`` invalidate the pin.
        self.manifest_path = self.checkout_root / self.MANIFEST_NAME
        self._command_runner = command_runner or subprocess.run

    @property
    def source_relative(self) -> str:
        try:
            return str(self.source_dir.relative_to(self.repo_root))
        except ValueError:
            return str(self.source_dir)

    def _verify_source(self, *, require_environment: bool = True) -> dict[str, Any]:
        if self.source_path.is_symlink() and not self.source_dir.is_relative_to(self.checkout_root):
            raise ExternalHarnessError("external checkout source symlink escapes checkout_root")
        try:
            identity = verify_git_checkout(
                self.source_dir,
                repo_url=self.config.repo_url,
                revision=self.config.revision,
                tree_sha=self.config.tree_sha,
                uv_lock_sha256=self.config.uv_lock_sha256,
                pyproject_sha256=self.config.pyproject_sha256,
            )
        except ExternalHarnessIntegrityError as exc:
            raise ExternalHarnessError(str(exc)) from exc
        if not require_environment:
            return identity
        if not self.venv_python.is_file():
            raise ExternalHarnessError(
                f"FinanceHarness isolated Python is missing: {self.venv_python}"
            )
        try:
            version = subprocess.run(
                [
                    str(self.venv_python),
                    "-c",
                    "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10.0,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise ExternalHarnessError("FinanceHarness isolated Python cannot be executed") from exc
        if version != self.config.python_version:
            raise ExternalHarnessError(
                f"FinanceHarness Python {version!r} != required {self.config.python_version!r}"
            )
        return identity

    def verify(self) -> dict[str, Any]:
        """Verify an existing source and environment; never clone/install/network."""

        identity = self._verify_source()
        manifest = read_manifest(self.manifest_path)
        expected = self._manifest_identity(identity)
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ExternalHarnessError(f"external manifest mismatch for {key}")
        return {
            **manifest,
            "source_dir": str(self.source_dir),
            "venv_python": str(self.venv_python),
        }

    def _manifest_identity(self, identity: dict[str, Any]) -> dict[str, Any]:
        safe_tools = {
            "names": sorted(SAFE_TOOL_NAMES),
            "pure_deferred_prefixes": list(PURE_DEFERRED_TOOL_PREFIXES),
        }
        bridge = {
            "protocol": self.config.protocol,
            "envelope_fields": sorted(ENVELOPE_FIELDS),
        }
        return {
            "schema_version": "1",
            "adapter": self.config.adapter,
            "adapter_version": self.config.version,
            "repo_url": self.config.repo_url,
            "revision": self.config.revision,
            "tree_sha": identity["tree_sha"],
            "uv_lock_sha256": identity["uv_lock_sha256"],
            "pyproject_sha256": identity["pyproject_sha256"],
            "python_version": self.config.python_version,
            "protocol": self.config.protocol,
            "license_status": self.config.license_status,
            "license_acknowledged": True,
            "source_relative": self.source_relative,
            "ipc_protocol_sha256": sha256_text(IPC_PROTOCOL),
            "ipc_schema_sha256": sha256_json(sorted(ENVELOPE_FIELDS)),
            "bridge_sha256": sha256_json(bridge),
            "safe_tools_sha256": sha256_json(safe_tools),
            "capability_manifest_sha256": sha256_json({"safe_tools": safe_tools}),
            "budget_sha256": sha256_json(asdict(self.deep_config)),
        }

    def _run(self, args: list[str], *, cwd: Path, timeout: float) -> None:
        try:
            self._command_runner(
                args,
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ExternalHarnessError(f"external sync command failed: {' '.join(args)}") from exc

    def sync(self, *, acknowledge_undeclared_license: bool = False) -> dict[str, Any]:
        """Clone/check out/install only after an explicit user acknowledgement."""

        if self.config.license_status == "UNDECLARED" and not acknowledge_undeclared_license:
            raise ExternalHarnessError(
                "FinanceHarness license is UNDECLARED; rerun external-sync with "
                "--acknowledge-undeclared-license."
            )
        self.checkout_root.mkdir(parents=True, exist_ok=True)
        if self.config.offline:
            if not self.source_dir.is_dir():
                raise ExternalHarnessError("offline external-sync requires an existing checkout")
        elif not self.source_dir.is_dir():
            staging = self.checkout_root / f".{self.config.adapter}.sync-{os.getpid()}"
            if staging.exists():
                shutil.rmtree(staging)
            try:
                self._run(
                    ["git", "clone", "--no-checkout", self.config.repo_url, str(staging)],
                    cwd=self.checkout_root,
                    timeout=600.0,
                )
                run_git(["checkout", "--detach", self.config.revision], cwd=staging)
                staging.replace(self.source_dir)
            except Exception:
                if staging.exists():
                    shutil.rmtree(staging)
                raise
        else:
            # A pre-existing checkout is never silently advanced.  It must
            # already be the exact requested revision.
            pass

        identity = self._verify_source(require_environment=False)
        sync_args = ["uv", "sync", "--locked", "--python", self.config.python_version]
        if self.config.offline:
            sync_args.append("--offline")
        self._run(sync_args, cwd=self.source_dir, timeout=1800.0)
        if not self.venv_python.is_file():
            raise ExternalHarnessError("uv sync completed without creating the isolated Python")
        self._exclude_environment_from_git_status()
        identity = self._verify_source()
        manifest = self._manifest_identity(identity)
        write_manifest(self.manifest_path, manifest)
        return {
            **manifest,
            "source_dir": str(self.source_dir),
            "venv_python": str(self.venv_python),
        }

    def _exclude_environment_from_git_status(self) -> None:
        """Keep the runtime-only venv out of clean-worktree provenance checks."""

        exclude = self.source_dir / ".git" / "info" / "exclude"
        if not exclude.is_file():
            return
        current = exclude.read_text(encoding="utf-8")
        if ".venv/" not in current.splitlines():
            with exclude.open("a", encoding="utf-8") as handle:
                handle.write("\n# benchmark-harness isolated runtime\n.venv/\n")

    def manifest(self) -> dict[str, Any]:
        """Read the persisted provenance without performing verification."""

        return read_manifest(self.manifest_path)


def manager_from_config(
    config: Any, *, repo_root: str | Path | None = None
) -> ExternalHarnessManager:
    harnesses = getattr(config, "external_harnesses", {})
    harness = harnesses.get("financeharness") if isinstance(harnesses, dict) else None
    if not isinstance(harness, ExternalHarnessConfig):
        raise ExternalHarnessError("deep_research has no financeharness external configuration")
    return ExternalHarnessManager(
        harness,
        deep_config=getattr(config, "deep_research", None),
        repo_root=repo_root,
    )


__all__ = ["ExternalHarnessError", "ExternalHarnessManager", "manager_from_config"]
