"""Audit and aggregate the locked eight-run R2C-FL v12 formal matrix.

This finalizer is intentionally unusable until both the D3 formal gate and the
six remaining D1/D2/D4 runs are terminal.  It never mutates a run directory.
The original preregistered AUC@20 values and the prospectively frozen TRM gate
are preserved; PTA@20 is emitted as a separate descriptive table report.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import SCHEMA_VERSION
from .config import DATASETS, PLOT_ROOT, QUEUE_ROOT, RUN_ROOT
from .logging_io import read_chunked_table
from .r2c_post_event_tail_accuracy20 import derive_pta20_percent
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
from .r2c_v8 import PROTOCOL_VERSION
from .utils import atomic_csv, atomic_json, atomic_parquet, sha256_file, sha256_text, utc_now


FORMAL_ACTIVE_MANIFEST = QUEUE_ROOT / "r2c_d3_v12_formal_manifest.json"
FORMAL_STATE = QUEUE_ROOT / "r2c_d3_v12_formal_queue_state.json"
FORMAL_RESULT = QUEUE_ROOT / "r2c_d3_v12_formal_result.json"
FORMAL_IMMUTABLE_MANIFEST = (
    QUEUE_ROOT / "r2c_d3_v12_formal_manifest_20260819T091635.529040Z.json"
)
EXPECTED_FORMAL_IMMUTABLE_SHA256 = (
    "27124de4a56822c6b68540b1b4956c54b491f17c553d7cc76fa4fadbf71be7d4"
)

REMAINING_MANIFEST = QUEUE_ROOT / "r2c_v12_remaining_datasets_manifest.json"
REMAINING_STATE = QUEUE_ROOT / "r2c_v12_remaining_datasets_queue_state.json"
REMAINING_RESULT = QUEUE_ROOT / "r2c_v12_remaining_datasets_result.json"


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"required terminal artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_terminal_payloads(
    formal_state: dict[str, Any],
    formal_result: dict[str, Any],
    formal_manifest: dict[str, Any],
    remaining_state: dict[str, Any],
    remaining_result: dict[str, Any],
    remaining_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate terminal authorization and return the exact eight jobs."""

    if (
        formal_state.get("status")
        != "formal_pilot_completed_gate_passed_remaining_datasets_required"
        or int(formal_state.get("completed", -1)) != 2
        or int(formal_state.get("failed", -1)) != 0
        or not bool(formal_state.get("all_runs_completed"))
        or not bool(formal_state.get("gate_passed"))
        or not bool(formal_state.get("other_dataset_access"))
        or formal_result.get("status") != "formal_pilot_gate_passed"
        or not bool(formal_result.get("gate", {}).get("gate_passed"))
        or not bool(formal_result.get("other_dataset_access_authorized"))
        or bool(formal_result.get("test_labels_used_for_selection"))
    ):
        raise RuntimeError("D3 formal pilot has not authorized all-dataset finalization")
    if (
        remaining_state.get("status")
        != "remaining_datasets_completed_tables_update_required"
        or int(remaining_state.get("completed", -1)) != 6
        or int(remaining_state.get("failed", -1)) != 0
        or not bool(remaining_state.get("all_runs_completed"))
        or bool(remaining_state.get("performance_sealed_until_terminal"))
        or remaining_result.get("status") != "remaining_datasets_completed_audited"
        or int(remaining_result.get("completed_runs", -1)) != 6
        or bool(remaining_result.get("test_labels_used_for_selection"))
    ):
        raise RuntimeError("remaining D1/D2/D4 formal matrix is not terminal")

    formal_jobs = list(formal_manifest.get("jobs", []))
    remaining_jobs = list(remaining_manifest.get("jobs", []))
    jobs = formal_jobs + remaining_jobs
    if len(formal_jobs) != 2 or len(remaining_jobs) != 6 or len(jobs) != 8:
        raise RuntimeError("v12 formal job counts are not exactly 2 + 6")
    if any(job.get("status") != "completed" or not job.get("actual_run_id") for job in jobs):
        raise RuntimeError("all eight v12 formal jobs must be completed")
    if any(
        int(job.get(key, -1)) != 20260811
        for job in jobs
        for key in ("seed", "partition_seed", "trace_seed")
    ):
        raise RuntimeError("v12 formal seed lineage mismatch")
    observed = {
        (str(job.get("dataset_id")), str(job.get("scenario_id"))) for job in jobs
    }
    expected = set(itertools.product(DATASETS.keys(), SCENARIOS))
    if observed != expected:
        raise RuntimeError("v12 formal dataset/scenario matrix is incomplete")
    if len({str(job["actual_run_id"]) for job in jobs}) != 8:
        raise RuntimeError("v12 formal run IDs are duplicated")
    return jobs


def _load_completed_jobs() -> tuple[list[dict[str, Any]], dict[str, str]]:
    if sha256_file(FORMAL_IMMUTABLE_MANIFEST).lower() != EXPECTED_FORMAL_IMMUTABLE_SHA256:
        raise RuntimeError("D3 formal immutable manifest hash drift")
    formal_state = _read_json(FORMAL_STATE)
    formal_result = _read_json(FORMAL_RESULT)
    formal_manifest = _read_json(FORMAL_ACTIVE_MANIFEST)
    remaining_state = _read_json(REMAINING_STATE)
    remaining_result = _read_json(REMAINING_RESULT)
    remaining_manifest = _read_json(REMAINING_MANIFEST)

    immutable_path = Path(str(remaining_state.get("immutable_manifest_path", "")))
    if (
        not immutable_path.exists()
        or sha256_file(immutable_path)
        != str(remaining_state.get("immutable_manifest_sha256", ""))
        or remaining_state.get("frozen_spec_hash")
        != remaining_manifest.get("frozen_spec_hash")
    ):
        raise RuntimeError("remaining-dataset immutable manifest lineage mismatch")
    jobs = _validate_terminal_payloads(
        formal_state,
        formal_result,
        formal_manifest,
        remaining_state,
        remaining_result,
        remaining_manifest,
    )
    hashes = {
        "d3_formal_immutable_manifest": sha256_file(FORMAL_IMMUTABLE_MANIFEST),
        "d3_formal_result": sha256_file(FORMAL_RESULT),
        "remaining_immutable_manifest": sha256_file(immutable_path),
        "remaining_result": sha256_file(REMAINING_RESULT),
    }
    return jobs, hashes


def _pta20_values(combined: pd.DataFrame) -> pd.DataFrame:
    baseline_rounds = pd.read_parquet(
        PLOT_ROOT / "round_metrics.parquet",
        columns=["run_id", "event_offset_round", "auc20_window_role", "test_accuracy"],
    )
    frame_cache: dict[str, pd.DataFrame] = {}

    def frame(run_id: str, method_id: str) -> pd.DataFrame:
        if run_id not in frame_cache:
            if method_id == R2C_METHOD:
                frame_cache[run_id] = read_chunked_table(
                    RUN_ROOT / run_id, "round_metrics"
                )
            else:
                selected = baseline_rounds.loc[baseline_rounds["run_id"].eq(run_id)].copy()
                if selected.empty:
                    raise RuntimeError(f"baseline round metrics missing: {run_id}")
                frame_cache[run_id] = selected
        return frame_cache[run_id]

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = combined.loc[combined["method_id"].eq(method)]
        values: list[float] = []
        run_ids: list[str] = []
        for dataset_id in DATASETS:
            compound = method_rows.loc[
                method_rows["dataset_id"].eq(dataset_id)
                & method_rows["scenario_id"].eq("S4")
            ].iloc[0]
            run_id = str(compound["run_id"])
            value = derive_pta20_percent(frame(run_id, method))
            values.append(value)
            run_ids.append(run_id)
            rows.append(
                {
                    "method_id": method,
                    "dataset_id": dataset_id,
                    "metric": "s4_pta20_percent",
                    "value": value,
                    "display": f"{value:.2f}",
                    "source_run_ids": run_id,
                }
            )
        macro = float(pd.Series(values, dtype=float).mean())
        worst = float(min(values))
        rows.extend(
            [
                {
                    "method_id": method,
                    "dataset_id": "ALL",
                    "metric": "s4_pta20_macro_mean_percent",
                    "value": macro,
                    "display": f"{macro:.2f}",
                    "source_run_ids": ";".join(run_ids),
                },
                {
                    "method_id": method,
                    "dataset_id": "ALL",
                    "metric": "s4_pta20_worst_case_percent",
                    "value": worst,
                    "display": f"{worst:.2f}",
                    "source_run_ids": ";".join(run_ids),
                },
            ]
        )
    result = pd.DataFrame(rows)
    if len(result) != len(METHODS) * 6:
        raise RuntimeError("PTA@20 Table-1 cell matrix is incomplete")
    return result


def finalize() -> dict[str, Any]:
    jobs, terminal_hashes = _load_completed_jobs()
    audit_reports = _run_audits(jobs)
    r2c_summary, window_reports = _r2c_summaries(jobs)
    baseline_summary = _baseline_table1_summaries()
    combined = pd.concat([baseline_summary, r2c_summary], ignore_index=True, sort=False)
    expected = set(itertools.product(METHODS, DATASETS.keys(), SCENARIOS))
    observed = set(
        combined[["method_id", "dataset_id", "scenario_id"]].itertuples(
            index=False, name=None
        )
    )
    if expected != observed or len(combined) != 64 or combined["run_id"].duplicated().any():
        raise RuntimeError("combined v12 Table-1 run matrix is incomplete or duplicated")
    if not (combined["seed"].astype(int) == 20260811).all():
        raise RuntimeError("combined v12 Table-1 seed mismatch")
    if not (combined["source_kind"].astype(str) == "REPRODUCED").all():
        raise RuntimeError("combined v12 Table-1 provenance mismatch")
    if not (combined["status"].astype(str) == "completed").all():
        raise RuntimeError("combined v12 Table-1 contains incomplete runs")

    original_values = _table_values(combined)
    pta_values = _pta20_values(combined)
    generated = utc_now()
    stamp = _stamp(generated)
    run_ids = combined["run_id"].astype(str).sort_values().tolist()
    input_paths = [
        PLOT_ROOT / "run_summary.parquet",
        PLOT_ROOT / "run_manifest.parquet",
        PLOT_ROOT / "round_metrics.parquet",
        PLOT_ROOT / "baseline_qc_report.json",
        FORMAL_IMMUTABLE_MANIFEST,
        FORMAL_RESULT,
        REMAINING_RESULT,
    ] + [RUN_ROOT / str(job["actual_run_id"]) / "result.json" for job in jobs]
    input_hash = sha256_text(
        "\n".join(f"{path}:{sha256_file(path)}" for path in input_paths)
    )

    outputs: list[str] = []
    outputs += _write_parquet("r2c_v12_table1_run_summary", r2c_summary, stamp)
    outputs += _write_parquet("table1_v12_combined_run_summary", combined, stamp)
    outputs += _write_parquet("table1_v12_combined_values", original_values, stamp)
    outputs += _write_csv("table1_v12_combined_values", original_values, stamp)
    outputs += _write_parquet("table1_v12_pta20_values", pta_values, stamp)
    outputs += _write_csv("table1_v12_pta20_values", pta_values, stamp)
    outputs += _write_json(
        "r2c_v12_table1_audit_report",
        {"generated_utc": generated, "runs": audit_reports},
        stamp,
    )
    outputs += _write_json(
        "r2c_v12_table1_auc20_windows",
        {"generated_utc": generated, "runs": window_reports},
        stamp,
    )
    outputs += _write_json(
        "r2c_v12_table1_pta20_report",
        {
            "schema_version": "r2c-v12-table1-pta20-v1",
            "generated_utc": generated,
            "metric": {
                "name": "post_event_tail_accuracy20",
                "short_name": "PTA@20",
                "formula_percent": (
                    "100 * mean(five lowest test accuracies at exact offsets +1..+20)"
                ),
                "direction": "higher_is_better",
                "descriptive_not_causal": True,
            },
            "formal_gate_unchanged": True,
            "cells": [
                {key: _json_value(value) for key, value in row.items()}
                for row in pta_values.to_dict("records")
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
        "formal_r2c_runs": int(len(r2c_summary)),
        "baseline_table1_runs": int(len(baseline_summary)),
        "combined_table1_runs": int(len(combined)),
        "unique_run_ids": not combined["run_id"].duplicated().any(),
        "matrix_complete": expected == observed,
        "matched_seed_ok": bool((combined["seed"].astype(int) == 20260811).all()),
        "source_kind_ok": bool(
            (combined["source_kind"].astype(str) == "REPRODUCED").all()
        ),
        "status_ok": bool((combined["status"].astype(str) == "completed").all()),
        "r2c_audits_passed": int(
            sum(report["status"] == "passed" for report in audit_reports)
        ),
        "r2c_strict_auc20_windows": int(
            sum(report["complete"] for report in window_reports)
        ),
        "pta20_cells": int(len(pta_values)),
        "input_files_hash": input_hash,
        "input_run_ids_hash": sha256_text("\n".join(run_ids)),
        "terminal_hashes": terminal_hashes,
        "formal_gate_unchanged": True,
    }
    outputs += _write_json("r2c_v12_table1_qc_report", qc, stamp)
    result = {
        "status": "v12_all_datasets_completed_audited_tables_update_required",
        "qc_complete": True,
        "formal_seed": 20260811,
        "formal_r2c_runs": 8,
        "combined_table1_runs": 64,
        "pta20_cells": int(len(pta_values)),
        "version_stamp": stamp,
        "outputs": outputs,
    }
    outputs += _write_json("r2c_v12_table1_aggregation_result", result, stamp)
    result["outputs"] = outputs

    remaining_state = _read_json(REMAINING_STATE)
    remaining_state.update(
        {
            "status": "v12_all_datasets_completed_audited_tables_update_required",
            "all8_audit_complete": True,
            "all8_aggregation_complete": True,
            "tables_update_required": True,
            "updated_utc": utc_now(),
        }
    )
    atomic_json(REMAINING_STATE, remaining_state)
    return result


def main() -> None:
    print(json.dumps(finalize(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
