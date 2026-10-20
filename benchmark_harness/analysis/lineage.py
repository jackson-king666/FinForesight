"""Artifact hashing and fail-closed lineage validation for offline analysis."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_harness.analysis.contracts import AnalysisError


class LineageError(AnalysisError):
    """Raised when raw artifacts are missing or their hashes drift."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_count(path: Path) -> int | None:
    if path.suffix != ".jsonl":
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    schema_version: str
    files: tuple[dict[str, Any], ...]
    lineage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "files": list(self.files),
            "lineage": self.lineage,
        }


def build_artifact_manifest(
    root: str | Path,
    *,
    lineage: Mapping[str, Any],
    schema_version: str = "artifact-manifest/v1",
) -> ArtifactManifest:
    """Hash every file except artifact_manifest.json itself."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise LineageError(f"artifact root is unavailable: {root_path}")
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root_path.rglob("*") if item.is_file()):
        relative = path.relative_to(root_path).as_posix()
        if relative == "artifact_manifest.json":
            continue
        entries.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "row_count": _row_count(path),
            }
        )
    return ArtifactManifest(
        schema_version=schema_version,
        files=tuple(entries),
        lineage=dict(lineage),
    )


def write_artifact_manifest(root: str | Path, manifest: ArtifactManifest) -> Path:
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    path = root_path / "artifact_manifest.json"
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def validate_artifact_manifest(
    root: str | Path,
    *,
    expected_lineage: Mapping[str, Any] | None = None,
) -> ArtifactManifest:
    """Verify file set, hashes, row counts, and required lineage before ingest."""

    root_path = Path(root).expanduser().resolve()
    manifest_path = root_path / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise LineageError(f"artifact manifest is missing: {manifest_path}")
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("artifact manifest root must be an object")
        raw_files = value["files"]
        if not isinstance(raw_files, list) or any(
            not isinstance(item, Mapping) for item in raw_files
        ):
            raise TypeError("artifact manifest files must be a list of objects")
        raw_lineage = value["lineage"]
        if not isinstance(raw_lineage, Mapping):
            raise TypeError("artifact manifest lineage must be an object")
        manifest = ArtifactManifest(
            schema_version=value["schema_version"],
            files=tuple(dict(item) for item in raw_files),
            lineage=dict(raw_lineage),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LineageError(f"artifact manifest is invalid: {manifest_path}") from exc
    if manifest.schema_version != "artifact-manifest/v1":
        raise LineageError("unsupported artifact manifest schema version.")
    if expected_lineage is not None:
        mismatches = [
            key
            for key, expected in expected_lineage.items()
            if manifest.lineage.get(key) != expected
        ]
        if mismatches:
            raise LineageError("artifact lineage mismatch: " + ", ".join(sorted(mismatches)))
    actual = build_artifact_manifest(root_path, lineage=manifest.lineage)
    if any(
        not isinstance(entry.get("path"), str)
        or not entry.get("path")
        or not isinstance(entry.get("sha256"), str)
        or not isinstance(entry.get("row_count"), int | type(None))
        for entry in manifest.files
    ):
        raise LineageError("artifact manifest contains an invalid file entry.")
    expected_files = {entry["path"]: entry for entry in manifest.files}
    if len(expected_files) != len(manifest.files):
        raise LineageError("artifact manifest contains duplicate file paths.")
    actual_files = {entry.get("path"): entry for entry in actual.files}
    if set(expected_files) != set(actual_files):
        raise LineageError("artifact file set differs from artifact_manifest.json.")
    for path, expected in expected_files.items():
        observed = actual_files[path]
        if expected.get("sha256") != observed.get("sha256"):
            raise LineageError(f"artifact hash drift detected for {path}.")
        if expected.get("bytes") != observed.get("bytes"):
            raise LineageError(f"artifact byte-count drift detected for {path}.")
        if expected.get("row_count") != observed.get("row_count"):
            raise LineageError(f"artifact row-count drift detected for {path}.")
    return manifest
