"""Stable CSV exports for C1/C2/C3 report consumers."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> Path:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})
    return path


def write_tables(output_dir: str | Path, result: Mapping[str, Any]) -> tuple[Path, ...]:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    c1 = result.get("c1", {})
    c1_path = root / "c1_contrasts.csv"
    paths.append(_write_csv(c1_path, list(c1.get("contrasts", []))))
    c2 = result.get("c2", {})
    c2_path = root / "c2_system_scores.csv"
    c2_rows = [
        {"system_id": system, "band": "T2", "score": score}
        for system, score in sorted(c2.get("t2_scores", {}).items())
    ] + [
        {"system_id": system, "band": "T3", "score": score}
        for system, score in sorted(c2.get("t3_scores", {}).items())
    ]
    paths.append(_write_csv(c2_path, c2_rows))
    c3_path = root / "c3_system_scores.csv"
    paths.append(
        _write_csv(
            c3_path,
            [
                {"system_id": system, "score": score}
                for system, score in sorted(
                    c2.get("system_scores", result.get("c3", {}).get("system_scores", {})).items()
                )
            ],
        )
    )
    return tuple(paths)
