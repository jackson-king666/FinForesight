"""Generic verify-only checkout facade for registered external systems."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark_harness.systems.provenance import SystemSourceProvenance, verify_system_source


def verify_existing_checkout(
    source_dir: str | Path,
    provenance: SystemSourceProvenance,
    *,
    lock_filename: str = "uv.lock",
    project_filename: str = "pyproject.toml",
) -> dict[str, Any]:
    """Verify a caller-prepared checkout; clone/install remains explicit elsewhere."""

    return verify_system_source(
        source_dir,
        provenance,
        lock_filename=lock_filename,
        project_filename=project_filename,
    )
