"""Stable JSON figure data without rendering or external calls."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def write_figure_data(output_dir: str | Path, result: Mapping[str, Any]) -> Path:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "figures-data.json"
    figure_data = {
        "schema_version": "figure-data/v1",
        "c1": result.get("c1", {}),
        "c2": result.get("c2", {}),
        "c3": result.get("c3", {}),
    }
    path.write_text(
        json.dumps(figure_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
