"""Atomic result artifact persistence."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def make_run_id(config_hash: str) -> str:
    """Create a human-sortable ID while retaining a configuration fingerprint."""

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + config_hash[:10]


def file_sha256(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactStore:
    """Result directory owner. JSON artifacts are replace-atomically written."""

    def __init__(self, results_root: str | Path, run_id: str) -> None:
        self.results_root = Path(results_root)
        self.run_id = run_id
        self.run_dir = self.results_root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def baseline_dir(self, baseline: str) -> Path:
        directory = self.run_dir / baseline
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def write_json(self, path: str | Path, value: Any) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
        )
        self._atomic_write(target, payload)
        return target

    def write_jsonl(self, path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n" for row in rows
        )
        self._atomic_write(target, payload)
        return target

    @staticmethod
    def _atomic_write(target: Path, content: str) -> None:
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent), text=True
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
