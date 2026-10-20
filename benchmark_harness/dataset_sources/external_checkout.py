"""Explicit migration-only access to a legacy local dataset checkout."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class ExternalCheckoutDatasetSource:
    """Wrap the legacy bridge without letting it become a Hub-load fallback.

    This adapter exists only for controlled migration and golden-parity work. It
    is never constructed by :class:`HuggingFaceDatasetSource`, so a Hub error
    cannot silently change an experiment's data origin.
    """

    source_name = "external_checkout"
    is_migration_only = True

    def __init__(self, root: str | Path, *, revision: str) -> None:
        self.root = Path(root)
        self.revision = revision
        self._bridge: Any | None = None
        self._provenance: dict[str, Any] | None = None

    def preflight(self) -> dict[str, Any]:
        if self._provenance is None:
            # The dynamic ``src`` import lives solely inside the existing legacy
            # bridge and is reached only when this migration-only source is
            # explicitly requested.
            from benchmark_harness.dataset_bridge import DatasetBridge

            self._bridge = DatasetBridge(self.root, revision=self.revision)
            provenance = self._bridge.preflight()
            self._provenance = {
                **provenance,
                "source": self.source_name,
                "migration_only": True,
            }
        return self.provenance()

    def provenance(self) -> dict[str, Any]:
        if self._provenance is None:
            self.preflight()
        assert self._provenance is not None
        return json.loads(json.dumps(self._provenance, ensure_ascii=False))

    def load_records(
        self,
        *,
        categories: Iterable[str] | None = None,
        status: str | None = "ready",
    ) -> list[dict[str, Any]]:
        if self._bridge is None:
            self.preflight()
        assert self._bridge is not None
        return self._bridge.load_records(categories=categories, status=status)
