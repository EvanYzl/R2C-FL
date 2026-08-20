from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import atomic_json, atomic_parquet, sha256_file


class ChunkedTableWriter:
    """Append-only Parquet chunks with a deterministic, restart-auditable index."""

    def __init__(self, run_dir: Path, table_name: str, flush_rows: int) -> None:
        self.table_name = table_name
        self.root = run_dir / "tables" / table_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.flush_rows = int(flush_rows)
        self.buffer: list[dict[str, Any]] = []
        existing = sorted(self.root.glob("part-*.parquet"))
        self.part_number = len(existing)
        self.row_count = 0
        for path in existing:
            try:
                self.row_count += len(pd.read_parquet(path, columns=[]))
            except Exception:
                self.row_count += len(pd.read_parquet(path))

    def append(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.flush_rows:
            self.flush()

    def extend(self, rows: list[dict[str, Any]]) -> None:
        self.buffer.extend(rows)
        if len(self.buffer) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        output = self.root / f"part-{self.part_number:06d}.parquet"
        atomic_parquet(output, pd.DataFrame(self.buffer))
        self.row_count += len(self.buffer)
        self.part_number += 1
        self.buffer.clear()

    def finalize(self) -> dict[str, Any]:
        self.flush()
        parts = sorted(self.root.glob("part-*.parquet"))
        index = {
            "table": self.table_name,
            "rows": self.row_count,
            "parts": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in parts
            ],
        }
        atomic_json(self.root / "_index.json", index)
        return index


def read_chunked_table(run_dir: Path, table_name: str) -> pd.DataFrame:
    root = run_dir / "tables" / table_name
    parts = sorted(root.glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)

