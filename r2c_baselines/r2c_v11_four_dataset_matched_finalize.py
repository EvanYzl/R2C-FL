from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import SCHEMA_VERSION
from .config import DATASETS, PLOT_ROOT, QUEUE_ROOT, RUN_ROOT
from .r2c_table1_finalize import (
    METHODS,
    R2C_METHOD,
    SCENARIOS,
    _baseline_table1_summaries,
    _json_value,
    _r2c_summaries,
    _run_audits,
    _table_values,
)
from .r2c_v11_four_dataset_matched_queue import FORMAL_INTERPRETATION
from .r2c_v7 import PROTOCOL_VERSION
from .utils import atomic_csv, atomic_json, atomic_parquet, sha256_file, sha256_text, utc_now


MANIFEST_PATH = QUEUE_ROOT / "r2c_v11_four_dataset_matched_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_v11_four_dataset_matched_queue_state.json"


def _stamp(value: str) -> str:
    return value[:19].replace("-", "").replace(":", "").replace("T", "_")


def _paths(stem: str, extension: str, stamp: str) -> tuple[Path, Path]:
    return PLOT_ROOT / f"{stem}_{stamp}.{extension}", PLOT_ROOT / f"{stem}.{extension}"


def _write_parquet(stem: str, frame: pd.DataFrame, stamp: str) -> list[str]:
    versioned, active = _paths(stem, "parquet", stamp)
    atomic_parquet(versioned, frame)
    atomic_parquet(active, frame)
    return [str(versioned), str(active)]


def _write_csv(stem: str, frame: pd.DataFrame, stamp: str) -> list[str]:
    versioned, active = _paths(stem, "csv", stamp)
    atomic_csv(versioned, frame)
    atomic_csv(active, frame)
    return [str(versioned), str(active)]


def _write_json(stem: str, payload: dict[str, Any], stamp: str) -> list[str]:
    versioned, active = _paths(stem, "json", stamp)
    atomic_json(versioned, payload)
    atomic_json(active, payload)
    return [str(versioned), str(active)]


def _load_jobs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("v11 matched Table 1 manifest protocol drift")
    if int(payload.get("formal_seed", -1)) != 20260811:
        raise RuntimeError("v11 matched Table 1 manifest seed drift")
    jobs = list(payload.get("jobs", []))
    if len(jobs) != 8:
        raise RuntimeError(f"Expected 8 matched v11 formal jobs, found {len(jobs)}")
    if any(job.get("status") != "completed" or not job.get("actual_run_id") for job in jobs):
        raise RuntimeError("All 8 matched v11 formal jobs must complete before finalization")
    for job in jobs:
        if any(int(job[key]) != 20260811 for key in ("seed", "partition_seed", "trace_seed")):
            raise RuntimeError(f"Seed mismatch in {job['job_id']}")
    return payload, jobs


def finalize() -> dict[str, Any]:
    manifest, jobs = _load_jobs()
    audit_reports = _run_audits(jobs)
    r2c_summary, window_reports = _r2c_summaries(jobs)
    baseline_summary = _baseline_table1_summaries()
    combined = pd.concat([baseline_summary, r2c_summary], ignore_index=True, sort=False)
    expected = set(itertools.product(METHODS, DATASETS.keys(), SCENARIOS))
    observed = set(combined[["method_id", "dataset_id", "scenario_id"]].itertuples(index=False, name=None))
    if expected != observed or len(combined) != 64 or combined["run_id"].duplicated().any():
        raise RuntimeError("Matched v11 Table 1 run matrix is incomplete or duplicated")
    if not (combined["seed"].astype(int) == 20260811).all():
        raise RuntimeError("Combined Table 1 is not uniformly seed 20260811")
    if not (combined["source_kind"].astype(str) == "REPRODUCED").all():
        raise RuntimeError("Combined Table 1 contains non-REPRODUCED provenance")
    if not (combined["status"].astype(str) == "completed").all():
        raise RuntimeError("Combined Table 1 contains incomplete runs")

    values = _table_values(combined)
    r2c_values = values[values["method_id"] == R2C_METHOD].copy()
    generated = utc_now()
    stamp = _stamp(generated)
    run_ids = combined["run_id"].astype(str).sort_values().tolist()
    input_paths = [
        PLOT_ROOT / "run_summary.parquet",
        PLOT_ROOT / "run_manifest.parquet",
        PLOT_ROOT / "round_metrics.parquet",
        PLOT_ROOT / "baseline_qc_report.json",
        MANIFEST_PATH,
    ] + [RUN_ROOT / str(job["actual_run_id"]) / "result.json" for job in jobs]
    input_hash = sha256_text("\n".join(f"{path}:{sha256_file(path)}" for path in input_paths))

    outputs: list[str] = []
    outputs += _write_parquet("r2c_v11_table1_run_summary", r2c_summary, stamp)
    outputs += _write_parquet("table1_v11_matched_combined_run_summary", combined, stamp)
    outputs += _write_parquet("table1_v11_matched_combined_values", values, stamp)
    outputs += _write_csv("table1_v11_matched_combined_values", values, stamp)
    outputs += _write_json("r2c_v11_table1_audit_report", {"generated_utc": generated, "runs": audit_reports}, stamp)
    outputs += _write_json(
        "r2c_v11_table1_auc20_windows",
        {"generated_utc": generated, "runs": window_reports},
        stamp,
    )
    outputs += _write_json(
        "r2c_v11_table1_values",
        {
            "generated_utc": generated,
            "method_id": R2C_METHOD,
            "formal_seed": 20260811,
            "protocol_version": PROTOCOL_VERSION,
            "formal_interpretation": FORMAL_INTERPRETATION,
            "cells": [
                {key: _json_value(value) for key, value in row.items()}
                for row in r2c_values.to_dict("records")
            ],
        },
        stamp,
    )
    qc = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated,
        "complete": True,
        "formal_seed": 20260811,
        "protocol_version": PROTOCOL_VERSION,
        "formal_interpretation": FORMAL_INTERPRETATION,
        "baseline_qc_complete": True,
        "baseline_table1_runs": int(len(baseline_summary)),
        "r2c_table1_runs": int(len(r2c_summary)),
        "combined_table1_runs": int(len(combined)),
        "unique_run_ids": not combined["run_id"].duplicated().any(),
        "matrix_complete": expected == observed,
        "matched_seed_ok": bool((combined["seed"].astype(int) == 20260811).all()),
        "source_kind_ok": bool((combined["source_kind"].astype(str) == "REPRODUCED").all()),
        "status_ok": bool((combined["status"].astype(str) == "completed").all()),
        "r2c_audits_passed": int(sum(report["status"] == "passed" for report in audit_reports)),
        "r2c_strict_auc20_windows": int(sum(report["complete"] for report in window_reports)),
        "r2c_table1_cells": int(len(r2c_values)),
        "combined_table1_cells": int(len(values)),
        "input_files_hash": input_hash,
        "input_run_ids_hash": sha256_text("\n".join(run_ids)),
        "matched_manifest_hash": sha256_file(MANIFEST_PATH),
    }
    outputs += _write_json("r2c_v11_table1_qc_report", qc, stamp)
    result = {
        "status": "completed",
        "qc_complete": True,
        "formal_seed": 20260811,
        "formal_r2c_runs": 8,
        "combined_table1_runs": 64,
        "r2c_table1_cells": int(len(r2c_values)),
        "version_stamp": stamp,
        "outputs": outputs,
    }
    outputs += _write_json("r2c_v11_table1_aggregation_result", result, stamp)
    result["outputs"] = outputs
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    state.update(
        {
            "status": "formal_completed_audited",
            "audit_complete": True,
            "aggregation_complete": True,
            "updated_utc": utc_now(),
        }
    )
    atomic_json(STATE_PATH, state)
    return result


def main() -> None:
    print(json.dumps(finalize(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
