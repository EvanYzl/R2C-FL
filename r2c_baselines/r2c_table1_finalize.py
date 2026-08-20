from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import SCHEMA_VERSION
from .aggregate import _run_summary
from .config import BASELINES, DATASETS, PLOT_ROOT, QUEUE_ROOT, RUN_ROOT
from .logging_io import read_chunked_table
from .metrics import recovery_auc20
from .utils import atomic_csv, atomic_json, atomic_parquet, sha256_file, sha256_text, utc_now


FORMAL_SEED = 20260811
R2C_METHOD = "R2C-FL"
SCENARIOS = ("S0", "S4")
METHODS = tuple(BASELINES) + (R2C_METHOD,)
AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / "tests" / "audit_r2c_run.py"


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _load_r2c_formal_jobs() -> list[dict[str, Any]]:
    path = QUEUE_ROOT / "r2c_table1_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = [job for job in payload["jobs"] if job["stage"] == "formal"]
    if len(jobs) != 8:
        raise RuntimeError(f"Expected 8 formal R2C-FL jobs, found {len(jobs)}")
    if any(job["status"] != "completed" or not job["actual_run_id"] for job in jobs):
        raise RuntimeError("All 8 formal R2C-FL jobs must be completed before finalization")
    return jobs


def _run_audits(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for job in jobs:
        run_dir = RUN_ROOT / str(job["actual_run_id"])
        process = subprocess.run(
            [sys.executable, str(AUDIT_SCRIPT), str(run_dir)],
            cwd=str(Path(__file__).resolve().parents[1]),
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(process.stdout.strip().splitlines()[-1])
        if report.get("status") != "passed":
            raise RuntimeError(f"R2C audit did not pass for {run_dir.name}: {report}")
        reports.append(report)
    return reports


def _r2c_summaries(jobs: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for job in jobs:
        run_dir = RUN_ROOT / str(job["actual_run_id"])
        manifest = pd.read_parquet(run_dir / "run_manifest.parquet")
        if len(manifest) != 1:
            raise RuntimeError(f"Expected one run manifest row for {run_dir.name}")
        manifest_row = manifest.iloc[0]
        summary, rounds, _ = _run_summary(run_dir)
        if len(rounds) != int(manifest_row["round_budget"]):
            raise RuntimeError(f"Incomplete round budget for {run_dir.name}")
        if rounds[["run_id", "round"]].duplicated().any():
            raise RuntimeError(f"Duplicate round key for {run_dir.name}")

        final_elapsed = float(rounds.sort_values("round").iloc[-1]["algorithm_elapsed_s"])
        rows.append(
            {
                "run_id": str(manifest_row["run_id"]),
                "dataset_id": str(manifest_row["dataset_id"]),
                "scenario_id": str(manifest_row["scenario_id"]),
                "method_id": str(manifest_row["method_id"]),
                "seed": int(manifest_row["seed"]),
                "source_kind": str(manifest_row["source_kind"]),
                "status": str(manifest_row["status"]),
                "round_budget": int(manifest_row["round_budget"]),
                "last50_accuracy": float(summary["last50_accuracy"]),
                "recovery_deficit_auc20": summary["recovery_deficit_auc20"],
                "recovery_auc20_complete": bool(summary["recovery_auc20_complete"]),
                "recovery_missing_reason": summary["recovery_missing_reason"],
                "target_accuracy": summary["target_accuracy"],
                "target_reached": bool(summary["target_reached"]),
                "rounds_to_target": summary["rounds_to_target"],
                "algorithm_wall_tta_s": summary["algorithm_wall_tta_s"],
                "tta_censored": bool(summary["tta_censored"]),
                "final_algorithm_elapsed_s": final_elapsed,
            }
        )

        scenario = str(manifest_row["scenario_id"])
        if scenario == "S4":
            offsets = rounds["event_offset_round"].astype(int)
            event_rows = rounds[offsets == 0]
            if len(event_rows) != 1:
                raise RuntimeError(f"Expected one event round for {run_dir.name}")
            event_round = int(event_rows.iloc[0]["round"])
            recomputed = recovery_auc20(rounds["round"], rounds["test_accuracy"], event_round)
            stored = float(summary["recovery_deficit_auc20"])
            pre_count = int(offsets.between(-20, -1).sum())
            post_count = int(offsets.between(1, 20).sum())
            matched = abs(float(recomputed["recovery_deficit_auc20"]) - stored) <= 1e-12
            if pre_count != 20 or post_count != 20 or not recomputed["recovery_auc20_complete"] or not matched:
                raise RuntimeError(f"Invalid strict AUC@20 window for {run_dir.name}")
            windows.append(
                {
                    "run_id": run_dir.name,
                    "event_round": event_round,
                    "pre_window_rounds": pre_count,
                    "post_window_rounds": post_count,
                    "stored_auc20": stored,
                    "recomputed_auc20": float(recomputed["recovery_deficit_auc20"]),
                    "absolute_error": abs(float(recomputed["recovery_deficit_auc20"]) - stored),
                    "complete": True,
                }
            )
        elif summary["recovery_deficit_auc20"] is not None:
            raise RuntimeError(f"Stationary run unexpectedly has AUC@20 for {run_dir.name}")
    return pd.DataFrame(rows), windows


def _baseline_table1_summaries() -> pd.DataFrame:
    qc = json.loads((PLOT_ROOT / "baseline_qc_report.json").read_text(encoding="utf-8"))
    if not qc.get("complete"):
        raise RuntimeError("Existing baseline aggregation QC is not complete")
    summary = pd.read_parquet(PLOT_ROOT / "run_summary.parquet")
    manifests = pd.read_parquet(PLOT_ROOT / "run_manifest.parquet")
    selected = summary[
        summary["method_id"].isin(BASELINES)
        & summary["dataset_id"].isin(DATASETS)
        & summary["scenario_id"].isin(SCENARIOS)
        & (summary["seed"] == FORMAL_SEED)
    ].copy()
    if len(selected) != 56 or selected["run_id"].duplicated().any():
        raise RuntimeError(f"Expected 56 unique Table 1 baseline runs, found {len(selected)}")
    manifest_columns = manifests[
        ["run_id", "source_kind", "status", "round_budget"]
    ].copy()
    selected = selected.merge(manifest_columns, on="run_id", how="left", validate="one_to_one")
    rounds = pd.read_parquet(
        PLOT_ROOT / "round_metrics.parquet",
        columns=["run_id", "round", "algorithm_elapsed_s"],
    )
    finals = (
        rounds[rounds["run_id"].isin(selected["run_id"])]
        .sort_values(["run_id", "round"])
        .groupby("run_id", as_index=False)
        .tail(1)[["run_id", "algorithm_elapsed_s"]]
        .rename(columns={"algorithm_elapsed_s": "final_algorithm_elapsed_s"})
    )
    selected = selected.merge(finals, on="run_id", how="left", validate="one_to_one")
    return selected[
        [
            "run_id",
            "dataset_id",
            "scenario_id",
            "method_id",
            "seed",
            "source_kind",
            "status",
            "round_budget",
            "last50_accuracy",
            "recovery_deficit_auc20",
            "recovery_auc20_complete",
            "recovery_missing_reason",
            "target_accuracy",
            "target_reached",
            "rounds_to_target",
            "algorithm_wall_tta_s",
            "tta_censored",
            "final_algorithm_elapsed_s",
        ]
    ].copy()


def _display(value: float, digits: int, exact_zero_dagger: bool = False) -> str:
    if exact_zero_dagger and value == 0.0:
        return "$0^{\\dagger}$"
    return f"{value:.{digits}f}"


def _table_values(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        method: str,
        dataset_id: str,
        metric: str,
        value: float,
        display: str,
        run_ids: list[str],
        censored: bool = False,
    ) -> None:
        rows.append(
            {
                "method_id": method,
                "dataset_id": dataset_id,
                "dataset_name": DATASETS[dataset_id].name if dataset_id in DATASETS else "Across datasets",
                "metric": metric,
                "value": float(value),
                "display": display,
                "censored": bool(censored),
                "source_run_ids": ";".join(run_ids),
            }
        )

    for method in METHODS:
        method_rows = combined[combined["method_id"] == method]
        for dataset_id in DATASETS:
            stationary = method_rows[
                (method_rows["dataset_id"] == dataset_id) & (method_rows["scenario_id"] == "S0")
            ].iloc[0]
            compound = method_rows[
                (method_rows["dataset_id"] == dataset_id) & (method_rows["scenario_id"] == "S4")
            ].iloc[0]
            s0_accuracy = float(stationary["last50_accuracy"]) * 100.0
            s4_accuracy = float(compound["last50_accuracy"]) * 100.0
            auc_pp = float(compound["recovery_deficit_auc20"]) * 100.0
            censored = bool(compound["tta_censored"])
            tta_seconds = (
                float(compound["final_algorithm_elapsed_s"])
                if censored
                else float(compound["algorithm_wall_tta_s"])
            )
            tta_hours = tta_seconds / 3600.0
            add(method, dataset_id, "s0_last50_accuracy_pct", s0_accuracy, _display(s0_accuracy, 2), [str(stationary["run_id"])])
            add(method, dataset_id, "s4_last50_accuracy_pct", s4_accuracy, _display(s4_accuracy, 2), [str(compound["run_id"])])
            add(method, dataset_id, "s4_recovery_deficit_auc20_pp", auc_pp, _display(auc_pp, 3, True), [str(compound["run_id"])])
            tta_display = _display(tta_hours, 3)
            if censored:
                tta_display = f"$>{tta_display}$"
            add(method, dataset_id, "s4_tta_h", tta_hours, tta_display, [str(compound["run_id"])], censored)

        compound_rows = method_rows[method_rows["scenario_id"] == "S4"].sort_values("dataset_id")
        values = compound_rows["recovery_deficit_auc20"].astype(float).to_numpy() * 100.0
        run_ids = compound_rows["run_id"].astype(str).tolist()
        macro = float(values.mean())
        worst = float(values.max())
        add(method, "ALL", "s4_auc20_macro_mean_pp", macro, _display(macro, 3, True), run_ids)
        add(method, "ALL", "s4_auc20_worst_pp", worst, _display(worst, 3, True), run_ids)
    result = pd.DataFrame(rows)
    if len(result) != len(METHODS) * 18:
        raise RuntimeError(f"Expected {len(METHODS) * 18} Table 1 cells, found {len(result)}")
    return result


def finalize() -> dict[str, Any]:
    jobs = _load_r2c_formal_jobs()
    audit_reports = _run_audits(jobs)
    r2c_summary, window_reports = _r2c_summaries(jobs)
    baseline_summary = _baseline_table1_summaries()
    combined = pd.concat([baseline_summary, r2c_summary], ignore_index=True, sort=False)

    expected = set(itertools.product(METHODS, DATASETS.keys(), SCENARIOS))
    observed = set(combined[["method_id", "dataset_id", "scenario_id"]].itertuples(index=False, name=None))
    if expected != observed or len(combined) != 64 or combined["run_id"].duplicated().any():
        raise RuntimeError("Combined Table 1 run matrix is incomplete or duplicated")
    if not (combined["source_kind"] == "REPRODUCED").all():
        raise RuntimeError("Combined Table 1 contains a non-reproduced run")
    if not (combined["status"] == "completed").all():
        raise RuntimeError("Combined Table 1 contains a non-completed run")

    values = _table_values(combined)
    r2c_values = values[values["method_id"] == R2C_METHOD].copy()
    generated = utc_now()
    run_ids = combined["run_id"].astype(str).sort_values().tolist()
    input_paths = [
        PLOT_ROOT / "run_summary.parquet",
        PLOT_ROOT / "run_manifest.parquet",
        PLOT_ROOT / "round_metrics.parquet",
        PLOT_ROOT / "baseline_qc_report.json",
        QUEUE_ROOT / "r2c_table1_manifest.json",
    ] + [RUN_ROOT / str(job["actual_run_id"]) / "result.json" for job in jobs]
    input_hash = sha256_text("\n".join(f"{path}:{sha256_file(path)}" for path in input_paths))

    atomic_parquet(PLOT_ROOT / "r2c_table1_run_summary.parquet", r2c_summary)
    atomic_parquet(PLOT_ROOT / "table1_combined_run_summary.parquet", combined)
    atomic_parquet(PLOT_ROOT / "table1_combined_values.parquet", values)
    atomic_csv(PLOT_ROOT / "table1_combined_values.csv", values)
    atomic_json(PLOT_ROOT / "r2c_table1_audit_report.json", {"generated_utc": generated, "runs": audit_reports})
    atomic_json(PLOT_ROOT / "r2c_table1_auc20_windows.json", {"generated_utc": generated, "runs": window_reports})
    atomic_json(
        PLOT_ROOT / "r2c_table1_values.json",
        {
            "generated_utc": generated,
            "method_id": R2C_METHOD,
            "cells": [{key: _json_value(value) for key, value in row.items()} for row in r2c_values.to_dict("records")],
        },
    )

    qc = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated,
        "complete": True,
        "baseline_qc_complete": True,
        "baseline_table1_runs": int(len(baseline_summary)),
        "r2c_table1_runs": int(len(r2c_summary)),
        "combined_table1_runs": int(len(combined)),
        "unique_run_ids": not combined["run_id"].duplicated().any(),
        "matrix_complete": expected == observed,
        "source_kind_ok": bool((combined["source_kind"] == "REPRODUCED").all()),
        "status_ok": bool((combined["status"] == "completed").all()),
        "r2c_audits_passed": int(sum(report["status"] == "passed" for report in audit_reports)),
        "r2c_strict_auc20_windows": int(sum(report["complete"] for report in window_reports)),
        "r2c_table1_cells": int(len(r2c_values)),
        "combined_table1_cells": int(len(values)),
        "input_files_hash": input_hash,
        "input_run_ids_hash": sha256_text("\n".join(run_ids)),
    }
    atomic_json(PLOT_ROOT / "r2c_table1_qc_report.json", qc)
    result = {
        "status": "completed",
        "qc_complete": True,
        "formal_r2c_runs": 8,
        "combined_table1_runs": 64,
        "r2c_table1_cells": 18,
        "outputs": str(PLOT_ROOT),
    }
    atomic_json(PLOT_ROOT / "r2c_table1_aggregation_result.json", result)
    return result


def main() -> None:
    print(json.dumps(finalize(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
