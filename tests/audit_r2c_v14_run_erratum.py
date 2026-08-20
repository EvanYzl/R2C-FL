from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from r2c_baselines.logging_io import read_chunked_table


ERRATUM_ID = "R2C_V14_AUDIT_ROLE_ALIAS_ERRATUM_20260819"
FROZEN_AUDITOR_PATH = Path(__file__).with_name("audit_r2c_v14_run.py")
FROZEN_AUDITOR_SHA256 = (
    "4e677fc85e00ad805a06dcd87d0c2133df3235f6e12ba085a86bcf0235c6f525"
)
CANONICAL_BEFORE_ROLE = "pre20"
CANONICAL_AFTER_ROLE = "post20"
AUDITOR_BEFORE_ALIAS = "before"
AUDITOR_AFTER_ALIAS = "after"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_auditor() -> ModuleType:
    actual = _sha256_file(FROZEN_AUDITOR_PATH)
    if actual != FROZEN_AUDITOR_SHA256:
        raise RuntimeError(
            "Frozen v14 auditor hash drift: "
            f"{actual} != {FROZEN_AUDITOR_SHA256}"
        )
    spec = importlib.util.spec_from_file_location(
        "r2c_v14_frozen_auditor_for_role_alias_erratum",
        FROZEN_AUDITOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load frozen auditor: {FROZEN_AUDITOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_window_contract(round_frame: pd.DataFrame) -> dict[str, Any]:
    required = {"round", "event_offset_round", "auc20_window_role"}
    missing = sorted(required - set(round_frame.columns))
    if missing:
        raise AssertionError(f"Missing exact-window columns: {missing}")

    roles = round_frame["auc20_window_role"].astype(str)
    if roles.isin([AUDITOR_BEFORE_ALIAS, AUDITOR_AFTER_ALIAS]).any():
        raise AssertionError(
            "Run output already contains the auditor-only before/after aliases"
        )

    before = round_frame.loc[
        roles.eq(CANONICAL_BEFORE_ROLE), "event_offset_round"
    ].astype(int)
    after = round_frame.loc[
        roles.eq(CANONICAL_AFTER_ROLE), "event_offset_round"
    ].astype(int)
    event = round_frame.loc[roles.eq("event"), "event_offset_round"].astype(int)

    expected_before = list(range(-20, 0))
    expected_after = list(range(1, 21))
    if sorted(before.tolist()) != expected_before:
        raise AssertionError("Canonical pre20 window is not exactly -20..-1")
    if sorted(after.tolist()) != expected_after:
        raise AssertionError("Canonical post20 window is not exactly +1..+20")
    if event.tolist() != [0]:
        raise AssertionError("Canonical event row is not exactly offset 0")

    return {
        "canonical_before_role": CANONICAL_BEFORE_ROLE,
        "canonical_after_role": CANONICAL_AFTER_ROLE,
        "before_count": int(len(before)),
        "after_count": int(len(after)),
        "event_count": int(len(event)),
        "before_min": int(before.min()),
        "before_max": int(before.max()),
        "after_min": int(after.min()),
        "after_max": int(after.max()),
    }


def compatibility_round_frame(round_frame: pd.DataFrame) -> pd.DataFrame:
    """Return an in-memory audit view; never mutate the recorded run table."""

    canonical_window_contract(round_frame)
    compatible = round_frame.copy(deep=True)
    compatible["auc20_window_role"] = compatible["auc20_window_role"].replace(
        {
            CANONICAL_BEFORE_ROLE: AUDITOR_BEFORE_ALIAS,
            CANONICAL_AFTER_ROLE: AUDITOR_AFTER_ALIAS,
        }
    )
    return compatible


def audit_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    job_path = run_dir / "job.json"
    if not job_path.is_file():
        raise AssertionError(f"Missing v14 job file: {job_path}")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if job.get("scenario_id") != "S4" or int(job.get("rounds", -1)) < 520:
        raise AssertionError(
            "This erratum is restricted to full-window v14 S4 runs"
        )

    recorded_rounds = read_chunked_table(run_dir, "round_metrics")
    structure = canonical_window_contract(recorded_rounds)
    frozen_auditor = _load_frozen_auditor()
    frozen_reader: Callable[[Path, str], pd.DataFrame] = (
        frozen_auditor.read_chunked_table
    )

    def compatibility_reader(path: Path, table_name: str) -> pd.DataFrame:
        frame = frozen_reader(path, table_name)
        if table_name == "round_metrics":
            return compatibility_round_frame(frame)
        return frame

    frozen_auditor.read_chunked_table = compatibility_reader
    try:
        result = dict(frozen_auditor.audit_run(run_dir))
    finally:
        frozen_auditor.read_chunked_table = frozen_reader

    result.update(
        {
            "audit_erratum_id": ERRATUM_ID,
            "frozen_auditor_sha256": FROZEN_AUDITOR_SHA256,
            "recorded_tables_mutated": False,
            "compatibility_mapping": {
                CANONICAL_BEFORE_ROLE: AUDITOR_BEFORE_ALIAS,
                CANONICAL_AFTER_ROLE: AUDITOR_AFTER_ALIAS,
            },
            "canonical_window_contract": structure,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_run(args.run_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
