"""Dataset-loading boundary kept separate from frozen evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol


class DatasetSource(Protocol):
    """Runner-facing source contract; it never owns parsing or scoring."""

    source_name: str

    def preflight(self) -> dict[str, Any]: ...

    def provenance(self) -> dict[str, Any]: ...

    def load_records(
        self,
        *,
        categories: Iterable[str] | None = None,
        status: str | None = "ready",
    ) -> list[dict[str, Any]]: ...
