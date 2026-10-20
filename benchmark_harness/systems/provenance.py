"""Generic external-source provenance without hidden checkout/install work."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_harness.external_harnesses.provenance import (
    ExternalHarnessIntegrityError,
    ensure_contained_tree,
    run_git,
    sha256_file,
)


class SystemProvenanceError(ValueError):
    """Raised when a system source cannot be verified against its declaration."""


@dataclass(frozen=True, slots=True)
class SystemSourceProvenance:
    repo_url: str
    revision: str
    tree_sha: str
    dependency_lock_sha256: str
    project_sha256: str
    python_version: str
    license_status: str

    def __post_init__(self) -> None:
        if not self.repo_url or not self.revision or not self.tree_sha:
            raise SystemProvenanceError("repository provenance fields must be non-empty.")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise SystemProvenanceError("revision must be a lowercase Git SHA.")
        if not re.fullmatch(r"[0-9a-f]{40}", self.tree_sha):
            raise SystemProvenanceError("tree_sha must be a lowercase Git SHA.")
        for name in ("dependency_lock_sha256", "project_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)):
                raise SystemProvenanceError(f"{name} must be a lowercase SHA-256.")
        if not self.python_version or not self.license_status:
            raise SystemProvenanceError("python_version and license_status are required.")

    def to_dict(self) -> dict[str, str]:
        return {
            "repo_url": self.repo_url,
            "revision": self.revision,
            "tree_sha": self.tree_sha,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "project_sha256": self.project_sha256,
            "python_version": self.python_version,
            "license_status": self.license_status,
        }


def verify_system_source(
    source_dir: str | Path,
    provenance: SystemSourceProvenance,
    *,
    lock_filename: str | None = "uv.lock",
    project_filename: str | None = "pyproject.toml",
) -> dict[str, Any]:
    """Verify a clean, contained checkout; never clone, install, or mutate it."""

    source = Path(source_dir).expanduser().resolve()
    try:
        ensure_contained_tree(source)
        origin = run_git(["remote", "get-url", "origin"], cwd=source)
        head = run_git(["rev-parse", "HEAD"], cwd=source)
        tree_sha = run_git(["rev-parse", "HEAD^{tree}"], cwd=source)
        status = run_git(["status", "--porcelain"], cwd=source)
    except ExternalHarnessIntegrityError as exc:
        raise SystemProvenanceError(str(exc)) from exc

    def normalize_origin(value: str) -> str:
        return value.rstrip("/").removesuffix(".git")

    if normalize_origin(origin) != normalize_origin(provenance.repo_url):
        raise SystemProvenanceError("external origin does not match provenance.")
    if head != provenance.revision or tree_sha != provenance.tree_sha:
        raise SystemProvenanceError("external source revision/tree does not match provenance.")
    if status:
        raise SystemProvenanceError("external source has uncommitted changes.")
    lock_hash = None
    project_hash = None
    if lock_filename is not None:
        lock_hash = sha256_file(source / lock_filename)
        if lock_hash != provenance.dependency_lock_sha256:
            raise SystemProvenanceError("dependency lock hash does not match provenance.")
    if project_filename is not None:
        project_hash = sha256_file(source / project_filename)
        if project_hash != provenance.project_sha256:
            raise SystemProvenanceError("project hash does not match provenance.")
    return {
        **provenance.to_dict(),
        "origin": origin,
        "head": head,
        "tree_sha": tree_sha,
        "lock_verified": lock_filename is not None,
        "project_verified": project_filename is not None,
        "clean": True,
    }


def verify_declared_file_hashes(
    source_dir: str | Path,
    expected: Mapping[str, str],
) -> dict[str, str]:
    """Verify immutable source files without cloning, installing, or rewriting.

    Adapter gates use this for license notices and benchmark-owned lockfiles
    whose provenance must remain visible independently of the upstream tree
    check.  Paths are contained under ``source_dir`` before they are read.
    """

    source = Path(source_dir).expanduser().resolve()
    try:
        ensure_contained_tree(source)
    except ExternalHarnessIntegrityError as exc:
        raise SystemProvenanceError(str(exc)) from exc
    observed: dict[str, str] = {}
    for relative, expected_hash in sorted(expected.items()):
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise SystemProvenanceError("declared file paths must be relative")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise SystemProvenanceError(f"invalid expected SHA-256 for {relative!r}")
        target = (source / relative).resolve()
        try:
            if not target.is_file() or not target.is_relative_to(source):
                raise SystemProvenanceError(f"declared file is unavailable: {relative!r}")
            if target.is_symlink() and not target.resolve().is_relative_to(source):
                raise SystemProvenanceError(f"declared file escapes source: {relative!r}")
            observed[relative] = sha256_file(target)
        except ExternalHarnessIntegrityError as exc:
            raise SystemProvenanceError(str(exc)) from exc
        if observed[relative] != expected_hash:
            raise SystemProvenanceError(f"declared file hash does not match for {relative!r}")
    return observed
