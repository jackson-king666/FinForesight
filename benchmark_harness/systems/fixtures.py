"""Immutable source-fixture attestations used by offline adapter tests."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from benchmark_harness.external_harnesses.provenance import sha256_file
from benchmark_harness.systems.provenance import (
    SystemProvenanceError,
    SystemSourceProvenance,
)

FIXTURE_MANIFEST = ".benchmark-fixture.json"


def is_fixture_source(source_dir: str | Path) -> bool:
    return (Path(source_dir).expanduser().resolve() / FIXTURE_MANIFEST).is_file()


def verify_fixture_source(
    source_dir: str | Path,
    provenance: SystemSourceProvenance,
    *,
    system: str,
    hook_verifier: Callable[[str | Path], dict[str, object]],
) -> dict[str, Any]:
    """Verify a checked-in, read-only source fixture against its attestation."""

    source = Path(source_dir).expanduser().resolve()
    manifest_path = source / FIXTURE_MANIFEST
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise SystemProvenanceError("source fixture manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemProvenanceError("source fixture manifest is malformed") from exc
    if not isinstance(manifest, Mapping) or manifest.get("system") != system:
        raise SystemProvenanceError("source fixture system identity mismatch")
    if manifest.get("revision") != provenance.revision:
        raise SystemProvenanceError("source fixture revision mismatch")
    if manifest.get("tree_sha") != provenance.tree_sha:
        raise SystemProvenanceError("source fixture upstream tree mismatch")
    declared_files = manifest.get("files")
    if not isinstance(declared_files, Mapping) or not declared_files:
        raise SystemProvenanceError("source fixture file attestation is missing")
    observed: dict[str, str] = {}
    for relative, expected in sorted(declared_files.items()):
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise SystemProvenanceError("source fixture file attestation is malformed")
        target = (source / relative).resolve()
        if not target.is_file() or target.is_symlink() or not target.is_relative_to(source):
            raise SystemProvenanceError(f"source fixture file is unavailable: {relative}")
        actual = sha256_file(target)
        if actual != expected:
            raise SystemProvenanceError(f"source fixture file drifted: {relative}")
        observed[relative] = actual
    upstream_files = manifest.get("upstream_file_sha256")
    if not isinstance(upstream_files, Mapping) or set(upstream_files) != set(declared_files):
        raise SystemProvenanceError("source fixture upstream file attestation is missing")
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in upstream_files.values()
    ):
        raise SystemProvenanceError("source fixture upstream file attestation is malformed")
    license_relative = manifest.get("license_file", "LICENSE")
    license_sha = manifest.get("license_sha256")
    if not isinstance(license_relative, str) or not isinstance(license_sha, str):
        raise SystemProvenanceError("source fixture license attestation is missing")
    license_path = (source / license_relative).resolve()
    if not license_path.is_file() or sha256_file(license_path) != license_sha:
        raise SystemProvenanceError("source fixture license drifted")
    return {
        **provenance.to_dict(),
        "source_mode": "offline_fixture",
        "origin": "repository_fixture",
        "head": provenance.revision,
        "tree_sha": provenance.tree_sha,
        "clean": True,
        "lock_verified": False,
        "project_verified": False,
        "fixture_manifest": FIXTURE_MANIFEST,
        "fixture_files": observed,
        "license_files": {license_relative: license_sha},
        "worker_hooks": hook_verifier(source),
    }


__all__ = ["FIXTURE_MANIFEST", "is_fixture_source", "verify_fixture_source"]
