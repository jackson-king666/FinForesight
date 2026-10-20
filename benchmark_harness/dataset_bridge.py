"""Safe, read-only access to the external benchmark dataset.

The dataset intentionally remains outside this repository.  Its package is named
``src``, which is common enough to be vulnerable to accidental import shadowing;
this module validates both the checkout revision and every imported ``src`` module
before exposing the upstream loader, parser, evaluator, and summarizer.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from threading import RLock
from types import ModuleType
from typing import Any

DATASET_ROOT_ENV = "BENCHMARK_DATASET_ROOT"
DEFAULT_DATASET_REVISION = "6c3ba942a33bb1ce941a88e4b5a78a11c3153f32"
_UPSTREAM_IMPORT_LOCK = RLock()


class DatasetBridgeError(RuntimeError):
    """Base exception for an unsafe or unusable external dataset checkout."""


class DatasetRootError(DatasetBridgeError):
    """Raised when the configured dataset root is absent or malformed."""


class DatasetRevisionError(DatasetBridgeError):
    """Raised when the dataset checkout does not match the configured revision."""


class DatasetValidationError(DatasetBridgeError):
    """Raised when the upstream validation script reports a failure."""


class DatasetImportError(DatasetBridgeError):
    """Raised when the dataset's ``src`` package cannot be imported safely."""


class DatasetBridge:
    """Bridge the harness to the pinned external dataset without copying its code.

    ``preflight()`` is intentionally the only path that imports the upstream
    package.  It must succeed before any loader/parser/evaluator method can be
    used.  The bridge never modifies the external checkout.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        revision: str = DEFAULT_DATASET_REVISION,
    ) -> None:
        configured_root = root if root is not None else os.environ.get(DATASET_ROOT_ENV)
        if not configured_root:
            raise DatasetRootError(
                f"Dataset root is required; set {DATASET_ROOT_ENV} or pass root explicitly."
            )

        self.root = Path(configured_root).expanduser().resolve()
        self.revision = revision
        self._ready = False
        self._manifest: dict[str, Any] | None = None
        self._upstream: ModuleType | None = None
        self._mock_agent: Any = None
        self._provenance: dict[str, Any] | None = None

    @classmethod
    def from_environment(cls, *, revision: str = DEFAULT_DATASET_REVISION) -> DatasetBridge:
        """Create a bridge from ``BENCHMARK_DATASET_ROOT``."""

        return cls(revision=revision)

    def preflight(self) -> dict[str, Any]:
        """Validate the checkout and import the upstream APIs exactly once.

        The upstream validator is run in its own checkout before importing any
        dataset code, so a malformed or partial mount fails before a benchmark can
        produce artifacts that look comparable.
        """

        if self._ready:
            return self.provenance()

        self._validate_layout()
        manifest = self._read_manifest()
        actual_revision = self._read_git_revision()
        if actual_revision != self.revision:
            raise DatasetRevisionError(
                "Dataset revision mismatch: "
                f"expected {self.revision}, found {actual_revision} at {self.root}."
            )
        self._assert_clean_tracked_worktree()

        validation_output = self._run_upstream_validation()
        upstream, mock_agent = self._import_upstream()

        self._manifest = manifest
        self._upstream = upstream
        self._mock_agent = mock_agent
        self._provenance = {
            "dataset_root": str(self.root),
            "dataset_revision": actual_revision,
            "expected_dataset_revision": self.revision,
            "manifest_dataset_name": manifest.get("dataset_name"),
            "manifest_version": manifest.get("version"),
            "validation": {
                "command": [sys.executable, "scripts/validate.py"],
                "status": "passed",
                "output": validation_output,
            },
        }
        self._ready = True
        return self.provenance()

    def provenance(self) -> dict[str, Any]:
        """Return JSON-serializable provenance for ``provenance.json`` artifacts."""

        self._ensure_ready()
        assert self._provenance is not None
        # JSON round-tripping returns a detached value and verifies the artifact
        # payload is serializable without exposing mutable bridge state.
        return json.loads(json.dumps(self._provenance, ensure_ascii=False, allow_nan=False))

    def load_records(
        self,
        *,
        categories: Iterable[str] | None = None,
        status: str | None = "ready",
    ) -> list[dict[str, Any]]:
        """Load unmodified records from manifest-selected upstream seed files.

        ``A2`` selects all A2 variants; an exact selector such as ``A2-F`` selects
        only that manifest entry.  Filtering occurs after calling upstream
        ``load_jsonl`` and does not mutate records or remove evaluation fields.
        """

        self._ensure_ready()
        assert self._manifest is not None
        upstream = self._require_upstream()

        task_files = self._manifest.get("task_files")
        if not isinstance(task_files, Mapping):
            raise DatasetRootError("manifest.json must contain a task_files object.")

        requested = self._normalize_categories(categories, task_files)
        records: list[dict[str, Any]] = []
        for task_key, task_info in task_files.items():
            if not self._matches_category(str(task_key), requested):
                continue
            if not isinstance(task_info, Mapping) or not isinstance(task_info.get("file"), str):
                raise DatasetRootError(f"manifest task_files.{task_key} must contain a seed file.")

            seed_path = self._resolve_dataset_path(task_info["file"])
            loaded = upstream.load_jsonl(seed_path)
            if not isinstance(loaded, list):
                raise DatasetBridgeError(
                    f"Upstream load_jsonl returned {type(loaded).__name__}, expected a list."
                )

            for record in loaded:
                if not isinstance(record, dict):
                    raise DatasetBridgeError(
                        "Upstream load_jsonl returned a non-object record; refusing to alter it."
                    )
                if status is None or record.get("status") == status:
                    records.append(record)
        return records

    def parse_prediction(self, category: str, raw_text: str) -> dict[str, Any]:
        """Parse a raw model response with the upstream category parser."""

        self._ensure_ready()
        if not isinstance(raw_text, str):
            raise TypeError("raw_text must be a string returned by an adapter.")
        parsed = self._require_upstream().parse_prediction(category, raw_text)
        return self._json_safe("upstream parse_prediction", parsed)

    def evaluate_prediction(
        self,
        category: str,
        parsed: Mapping[str, Any],
        record: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a parsed response with the upstream evaluator.

        This method is deliberately separate from adapter execution: only the
        runner/bridge sees the complete upstream record and its ground truth.
        """

        self._ensure_ready()
        metrics = self._require_upstream().evaluate_prediction(category, dict(parsed), dict(record))
        return self._json_safe("upstream evaluate_prediction", metrics)

    def summarize_results(
        self,
        predictions: Iterable[Mapping[str, Any]],
        skipped: Iterable[Mapping[str, Any]],
        *,
        agent: str,
        model: str,
    ) -> dict[str, Any]:
        """Delegate aggregate metrics to upstream ``summarize_results`` unchanged."""

        self._ensure_ready()
        # The upstream function expects lists; the dictionaries retain every
        # upstream-compatible prediction-row field supplied by the runner.
        summary = self._require_upstream().summarize_results(
            list(predictions), list(skipped), agent=agent, model=model
        )
        return self._json_safe("upstream summarize_results", summary)

    def mock_response(self, record: dict[str, Any]) -> str:
        """Return an upstream mock response for non-performance pipeline smoke only.

        The upstream mock consumes ground truth by design.  It is intentionally
        confined to this bridge helper and must never be passed to a real adapter
        or represented as a framework performance score.
        """

        self._ensure_ready()
        if not isinstance(record.get("prompt"), str):
            raise ValueError("Mock smoke record must contain a string prompt.")
        response = self._mock_agent(record["prompt"], record)
        if not isinstance(response, str):
            raise DatasetBridgeError(
                f"Upstream mock_agent returned {type(response).__name__}, expected a string."
            )
        return response

    def _ensure_ready(self) -> None:
        if not self._ready:
            self.preflight()

    def _validate_layout(self) -> None:
        if not self.root.is_dir():
            raise DatasetRootError(
                f"Dataset root does not exist or is not a directory: {self.root}"
            )

        for relative_path in ("manifest.json", "src/run_benchmark.py", "scripts/validate.py"):
            path = self.root / relative_path
            if not path.is_file():
                raise DatasetRootError(f"Dataset root is missing required file: {relative_path}")

    def _read_manifest(self) -> dict[str, Any]:
        manifest_path = self.root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetRootError(f"Unable to read manifest.json at {self.root}: {exc}") from exc
        if not isinstance(manifest, dict):
            raise DatasetRootError("manifest.json must contain a JSON object.")
        return manifest

    def _read_git_revision(self) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.root), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise DatasetRevisionError(f"Unable to invoke git for {self.root}: {exc}") from exc
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            raise DatasetRevisionError(
                f"Unable to resolve the dataset Git revision at {self.root}: {details}"
            )
        return completed.stdout.strip()

    def _run_upstream_validation(self) -> str:
        command = [sys.executable, "scripts/validate.py"]
        try:
            completed = subprocess.run(
                command,
                cwd=self.root,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise DatasetValidationError(
                f"Unable to run upstream validation in {self.root}: {exc}"
            ) from exc
        output_parts = (completed.stdout.strip(), completed.stderr.strip())
        output = "\n".join(part for part in output_parts if part)
        if completed.returncode != 0:
            raise DatasetValidationError(
                "Upstream dataset validation failed "
                f"with exit code {completed.returncode}: {output or 'no output'}"
            )
        return output

    def _import_upstream(self) -> tuple[ModuleType, Any]:
        """Load upstream APIs without leaving a globally importable ``src`` package."""

        with _UPSTREAM_IMPORT_LOCK:
            self._assert_no_preloaded_src_modules()
            original_path = list(sys.path)
            original_dont_write_bytecode = sys.dont_write_bytecode
            try:
                # The dataset is an external read-only input, not a place for
                # harness-generated __pycache__ files.
                sys.dont_write_bytecode = True
                sys.path.insert(0, str(self.root))
                upstream = importlib.import_module("src.run_benchmark")
                mock_module = importlib.import_module("src.agents.mock_agent")
                self._assert_loaded_src_modules_belong_to_root()
                for name in (
                    "load_jsonl",
                    "parse_prediction",
                    "evaluate_prediction",
                    "summarize_results",
                ):
                    if not callable(getattr(upstream, name, None)):
                        raise DatasetImportError(
                            f"Upstream src.run_benchmark is missing required callable: {name}"
                        )
                mock_agent = getattr(mock_module, "mock_agent", None)
                if not callable(mock_agent):
                    raise DatasetImportError(
                        "Upstream src.agents.mock_agent is missing callable mock_agent."
                    )
                return upstream, mock_agent
            except DatasetImportError:
                raise
            except Exception as exc:
                raise DatasetImportError(
                    f"Unable to import upstream dataset modules from {self.root}: {exc}"
                ) from exc
            finally:
                sys.path[:] = original_path
                sys.dont_write_bytecode = original_dont_write_bytecode
                self._remove_loaded_src_modules()

    @staticmethod
    def _remove_loaded_src_modules() -> None:
        for module_name in [
            name for name in sys.modules if name == "src" or name.startswith("src.")
        ]:
            sys.modules.pop(module_name, None)

    @staticmethod
    def _assert_no_preloaded_src_modules() -> None:
        loaded = sorted(name for name in sys.modules if name == "src" or name.startswith("src."))
        if loaded:
            raise DatasetImportError(
                "Detected preloaded src modules; refusing cache-based shadowing: "
                + ", ".join(loaded)
            )

    def _assert_loaded_src_modules_belong_to_root(self) -> None:
        for module_name, module in tuple(sys.modules.items()):
            if module_name != "src" and not module_name.startswith("src."):
                continue
            if not isinstance(module, ModuleType):
                raise DatasetImportError(
                    f"Loaded {module_name!r} is not a module; refusing shadowed import."
                )
            module_file = getattr(module, "__file__", None)
            if not module_file or not self._is_inside_root(Path(module_file)):
                raise DatasetImportError(
                    "Detected src import shadowing before benchmark evaluation: "
                    f"{module_name} resolves to {module_file!r}, not under {self.root}."
                )

    def _assert_clean_tracked_worktree(self) -> None:
        """Reject tracked content drift even when ``HEAD`` still has the pinned SHA.

        Untracked files are intentionally allowed: Python cache directories and a
        caller's mount-local notes do not alter a tracked parser, evaluator, seed,
        manifest, or validator.  Any staged or unstaged tracked change does.
        """

        try:
            completed = subprocess.run(
                ["git", "-C", str(self.root), "diff", "--quiet", "HEAD", "--"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise DatasetRevisionError(
                f"Unable to verify tracked dataset content at {self.root}: {exc}"
            ) from exc
        if completed.returncode == 0:
            return
        if completed.returncode != 1:
            details = (completed.stderr or completed.stdout).strip()
            raise DatasetRevisionError(
                f"Unable to verify tracked dataset content at {self.root}: {details}"
            )

        status = subprocess.run(
            ["git", "-C", str(self.root), "status", "--short", "--untracked-files=no"],
            check=False,
            capture_output=True,
            text=True,
        )
        changed_paths = status.stdout.strip() or "tracked files changed"
        raise DatasetRevisionError(
            "Dataset checkout contains tracked modifications and is not the pinned release: "
            f"{changed_paths}"
        )

    def _resolve_dataset_path(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        if not self._is_inside_root(candidate):
            raise DatasetRootError(f"Manifest path escapes dataset root: {relative_path}")
        if not candidate.is_file():
            raise DatasetRootError(f"Manifest references missing seed file: {relative_path}")
        return candidate

    def _is_inside_root(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.root)
        except ValueError:
            return False
        return True

    @staticmethod
    def _matches_category(task_key: str, requested: set[str] | None) -> bool:
        if requested is None:
            return True
        base_category = task_key.split("-", maxsplit=1)[0]
        return task_key in requested or base_category in requested

    @staticmethod
    def _normalize_categories(
        categories: Iterable[str] | None, task_files: Mapping[str, Any]
    ) -> set[str] | None:
        if categories is None:
            return None
        requested = {category for category in categories}
        if not requested or not all(
            isinstance(category, str) and category for category in requested
        ):
            raise ValueError("categories must contain one or more non-empty strings.")
        available = {str(task_key) for task_key in task_files}
        base_categories = {task_key.split("-", maxsplit=1)[0] for task_key in available}
        available.update(base_categories)
        unknown = requested - available
        if unknown:
            raise ValueError(f"Unknown dataset category selector(s): {sorted(unknown)}")
        return requested

    def _require_upstream(self) -> ModuleType:
        assert self._upstream is not None
        return self._upstream

    @staticmethod
    def _json_safe(operation: str, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise DatasetBridgeError(
                f"{operation} returned {type(value).__name__}, expected a JSON object."
            )
        try:
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DatasetBridgeError(
                f"{operation} returned a non-JSON-serializable value: {exc}"
            ) from exc
        return value


__all__ = [
    "DATASET_ROOT_ENV",
    "DEFAULT_DATASET_REVISION",
    "DatasetBridge",
    "DatasetBridgeError",
    "DatasetImportError",
    "DatasetRevisionError",
    "DatasetRootError",
    "DatasetValidationError",
]
