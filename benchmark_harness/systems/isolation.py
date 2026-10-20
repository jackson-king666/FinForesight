"""Reusable credential-free child isolation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from benchmark_harness.external_harnesses.ipc import build_child_environment


class IsolationError(ValueError):
    """Raised when a child capability would permit unsafe network/process access."""


@dataclass(frozen=True, slots=True)
class IsolationSpec:
    source_dir: Path
    repo_root: Path
    loopback_url: str
    python_executable: Path | None = None

    def __post_init__(self) -> None:
        parsed = urlsplit(self.loopback_url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None:
            raise IsolationError("child loopback_url must be an explicit 127.0.0.1 HTTP endpoint.")
        if not self.source_dir.is_absolute() or not self.repo_root.is_absolute():
            raise IsolationError("isolation paths must be absolute.")

    @property
    def allowed_host_port(self) -> tuple[str, int]:
        parsed = urlsplit(self.loopback_url)
        assert parsed.port is not None
        return "127.0.0.1", parsed.port


def build_isolated_environment(
    spec: IsolationSpec,
    *,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the existing minimal child environment for any registered system."""

    return build_child_environment(
        spec.source_dir,
        spec.repo_root,
        python_executable=spec.python_executable,
        base_environment=base_environment,
    )
