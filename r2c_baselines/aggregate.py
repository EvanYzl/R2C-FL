from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import SCHEMA_VERSION
from .config import BASELINES, PLOT_ROOT, QUEUE_ROOT, RUN_ROOT
from .logging_io import read_chunked_table
from .metrics import recovery_auc20
from .utils import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    git_commit,
    hash_files,
    sha256_file,
    sha256_text,
    utc_now,
)


def _completed_formal_runs() -> list[Path]:
    values: list[Path] = []
    for success in RUN_ROOT.glob("*/_SUCCESS.json"):
        run_dir = success.parent
        manifest_path = run_dir / "run_manifest.parquet"
        if not manifest_path.exists():
            continue
        manifest = pd.read_parquet(manifest_path)
        if (
            len(manifest) == 1
            and manifest.iloc[0]["source_kind"] == "REPRODUCED"
            and manifest.iloc[0]["status"] == "completed"
            and manifest.iloc[0]["method_id"] in BASELINES
        ):
            values.append(run_dir)
    return sorted(values, key=lambda path: path.name)


def _all_parts(run_dirs: Iterable[Path], table_name: str) -> list[Path]:
    parts: list[Path] = []
    for run_dir in run_dirs:
        parts.extend(sorted((run_dir / "tables" / table_name).glob("part-*.parquet")))
    return parts


def _stream_union(parts: list[Path], output: Path) -> int:
    if not parts:
        return 0
    schemas = [pq.read_schema(path) for path in parts]
    schema = pa.unify_schemas(schemas)
    temp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    writer = pq.ParquetWriter(temp, schema, compression="snappy")
    rows = 0
    try:
        for path in parts:
            table = pq.read_table(path)
            if table.schema != schema:
                table = table.cast(schema)
            writer.write_table(table)
            rows += table.num_rows
    finally:
        writer.close()
    os.replace(temp, output)
    return rows


def _energy_wh(samples: pd.DataFrame) -> float | None:
    if len(samples) < 2 or samples["gpu_power_w"].notna().sum() < 2:
        return None
    value = samples.dropna(subset=["gpu_power_w", "elapsed_s"]).sort_values("elapsed_s")
    if len(value) < 2:
        return None
    return float(np.trapezoid(value["gpu_power_w"], value["elapsed_s"]) / 3600.0)


def _directory_mib(path: Path) -> float:
    return float(sum(file.stat().st_size for file in path.rglob("*") if file.is_file()) / 2**20)


def _run_summary(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(run_dir / "run_manifest.parquet").iloc[0]
    rounds = read_chunked_table(run_dir, "round_metrics").sort_values("round")
    clients = read_chunked_table(run_dir, "client_round_metrics")
    systems = read_chunked_table(run_dir, "system_samples")
    event_rows = rounds[rounds["event_id"].notna()]
    event_round = None
    event_id = None
    if len(event_rows):
        event_round = int(event_rows.iloc[0]["round"] - event_rows.iloc[0]["event_offset_round"])
        event_id = str(event_rows.iloc[0]["event_id"])
    recovery = recovery_auc20(rounds["round"], rounds["test_accuracy"], event_round)
    target = manifest.get("target_accuracy")
    target = None if pd.isna(target) else float(target)
    reached = rounds[rounds["test_accuracy"] >= target] if target is not None else pd.DataFrame()
    target_reached = bool(len(reached))
    rounds_to_target = int(reached.iloc[0]["round"]) if target_reached else None
    wall_tta = float(reached.iloc[0]["algorithm_elapsed_s"]) if target_reached else None
    bytes_upload = int(rounds["bytes_upload"].sum())
    bytes_download = int(rounds["bytes_download"].sum())
    gpu_algorithm_seconds = float(rounds["gpu_time_s"].sum())
    cpu_seconds = float(rounds["cpu_time_s"].sum())
    summary = {
        "run_id": manifest["run_id"],
        "final_accuracy": float(rounds.iloc[-1]["test_accuracy"]),
        "last50_accuracy": float(rounds.tail(50)["test_accuracy"].mean()),
        "final_loss": float(rounds.iloc[-1]["test_loss"]),
        "pre_event_accuracy": recovery["pre_event_accuracy"],
        "max_accuracy_drop": recovery["max_drop"],
        "recovery_deficit_auc20": recovery["recovery_deficit_auc20"],
        "recovery_auc20_complete": recovery["recovery_auc20_complete"],
        "recovery_missing_reason": recovery["recovery_missing_reason"],
        "recovery_half_life_rounds": recovery["recovery_half_life_rounds"],
        "target_accuracy": target,
        "target_reached": target_reached,
        "rounds_to_target": rounds_to_target,
        "algorithm_wall_tta_s": wall_tta,
        "tta_censored": not target_reached,
        "audit_wall_total_s": float(rounds["audit_wall_s"].sum()),
        "end_to_end_wall_total_s": float(rounds.iloc[-1]["elapsed_wall_s"]),
        "bytes_upload_total": bytes_upload,
        "bytes_download_total": bytes_download,
        "payload_equiv_total": int(rounds["payload_equiv_upload"].sum()),
        "gpu_hours_algorithm": gpu_algorithm_seconds / 3600.0,
        "gpu_hours_audit": 0.0,
        "cpu_hours": cpu_seconds / 3600.0,
        "gpu_energy_est_wh": _energy_wh(systems),
        "peak_allocated_mib": float(systems["memory_allocated_mib"].max()) if len(systems) else None,
        "peak_reserved_mib": float(systems["memory_reserved_mib"].max()) if len(systems) else None,
        "peak_cpu_rss_mib": float(systems["cpu_rss_mib"].max()) if len(systems) else None,
        "peak_disk_mib": _directory_mib(run_dir),
        "final_participation_jfi": float(rounds.iloc[-1]["participation_jfi"]),
        "worst10_participation": float(rounds.iloc[-1]["worst10_participation"]),
        "deadline_completion_rate": float(rounds.iloc[-1]["deadline_completion_rate"]),
        "certified_rate": None,
        "fallback_rate": None,
        "rank_error_rate": None,
        "mean_commit_fraction": None,
        "candidate_compute_saved_fraction": None,
    }
    recovery_row = pd.DataFrame()
    if event_round is not None:
        recovery_row = pd.DataFrame(
            [
                {
                    "dataset_id": manifest["dataset_id"],
                    "scenario_id": manifest["scenario_id"],
                    "severity": manifest["severity"],
                    "method_id": manifest["method_id"],
                    "seed": int(manifest["seed"]),
                    "event_id": event_id,
                    "horizon_rounds": 20,
                    "pre_event_accuracy": recovery["pre_event_accuracy"],
                    "max_drop": recovery["max_drop"],
                    "recovery_deficit_auc20": recovery["recovery_deficit_auc20"],
                    "recovery_auc20_complete": recovery["recovery_auc20_complete"],
                    "recovery_missing_reason": recovery["recovery_missing_reason"],
                    "recovery_half_life_rounds": recovery["recovery_half_life_rounds"],
                    "post_event_round20_accuracy": recovery["post_event_round20_accuracy"],
                    "dynamic_accuracy_deficit": None,
                    "run_id": manifest["run_id"],
                }
            ]
        )
    return summary, rounds, recovery_row


def _provenance_columns(run_ids: list[str], raw_paths: list[Path]) -> dict[str, str]:
    return {
        "input_files_hash": hash_files(raw_paths, PLOT_ROOT),
        "input_run_ids_hash": sha256_text("\n".join(sorted(run_ids))),
        "aggregation_commit": git_commit(Path(__file__).resolve().parent),
        "generated_utc": utc_now(),
    }


def _add_provenance(frame: pd.DataFrame, provenance: dict[str, str]) -> pd.DataFrame:
    value = frame.copy()
    for key, item in provenance.items():
        value[key] = item
    return value


def aggregate() -> dict[str, Any]:
    run_dirs = _completed_formal_runs()
    if not run_dirs:
        raise RuntimeError("No completed formal baseline runs found")
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    manifests = pd.concat(
        [pd.read_parquet(run_dir / "run_manifest.parquet") for run_dir in run_dirs],
        ignore_index=True,
    )
    if manifests["run_id"].duplicated().any():
        raise RuntimeError("Duplicate run_id in completed formal manifests")
    atomic_parquet(PLOT_ROOT / "run_manifest.parquet", manifests)
    raw_counts = {"run_manifest": len(manifests)}
    for table_name in (
        "round_metrics",
        "client_round_metrics",
        "evaluation_by_class",
        "system_samples",
        "stage_timings",
        "failure_events",
    ):
        parts = _all_parts(run_dirs, table_name)
        if parts:
            raw_counts[table_name] = _stream_union(parts, PLOT_ROOT / f"{table_name}.parquet")
        else:
            raw_counts[table_name] = 0

    summary_rows: list[dict[str, Any]] = []
    round_by_run: dict[str, pd.DataFrame] = {}
    recovery_frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        summary, rounds, recovery = _run_summary(run_dir)
        summary_rows.append(summary)
        round_by_run[str(summary["run_id"])] = rounds
        if len(recovery):
            recovery_frames.append(recovery)
    run_summary = pd.DataFrame(summary_rows).merge(
        manifests[["run_id", "dataset_id", "scenario_id", "severity", "method_id", "seed"]],
        on="run_id",
        how="left",
    )
    recovery_summary = pd.concat(recovery_frames, ignore_index=True) if recovery_frames else pd.DataFrame()

    # Paired S0 trajectories are the frozen control for dynamic deficits.
    control = {
        (row.dataset_id, row.method_id, int(row.seed)): row.run_id
        for row in manifests.itertuples()
        if row.scenario_id == "S0"
    }
    if len(recovery_summary):
        for index, row in recovery_summary.iterrows():
            key = (row["dataset_id"], row["method_id"], int(row["seed"]))
            control_id = control.get(key)
            if control_id is None:
                continue
            dynamic = round_by_run[row["run_id"]][["round", "test_accuracy"]]
            stationary = round_by_run[control_id][["round", "test_accuracy"]]
            paired = stationary.merge(dynamic, on="round", suffixes=("_s0", "_dynamic"))
            recovery_summary.loc[index, "dynamic_accuracy_deficit"] = float(
                (paired["test_accuracy_s0"] - paired["test_accuracy_dynamic"]).mean()
            )

    raw_paths = [
        PLOT_ROOT / "run_manifest.parquet",
        PLOT_ROOT / "round_metrics.parquet",
        PLOT_ROOT / "client_round_metrics.parquet",
    ]
    run_ids = manifests["run_id"].astype(str).tolist()
    provenance = _provenance_columns(run_ids, [path for path in raw_paths if path.exists()])
    run_summary = _add_provenance(run_summary, provenance)
    recovery_summary = _add_provenance(recovery_summary, provenance)
    atomic_parquet(PLOT_ROOT / "run_summary.parquet", run_summary)
    atomic_parquet(PLOT_ROOT / "recovery_summary.parquet", recovery_summary)

    efficiency = run_summary[
        [
            "dataset_id",
            "scenario_id",
            "severity",
            "method_id",
            "seed",
            "algorithm_wall_tta_s",
            "tta_censored",
            "recovery_deficit_auc20",
            "last50_accuracy",
            "gpu_hours_algorithm",
            "gpu_hours_audit",
            "gpu_energy_est_wh",
            "worst10_participation",
            "deadline_completion_rate",
            "bytes_upload_total",
            "bytes_download_total",
            "final_participation_jfi",
        ]
    ].copy()
    efficiency["communication_gib"] = (
        efficiency["bytes_upload_total"] + efficiency["bytes_download_total"]
    ) / 2**30
    efficiency["participation_jfi"] = efficiency.pop("final_participation_jfi")
    efficiency["pareto_time_descriptive"] = False
    efficiency["pareto_comm_descriptive"] = False
    for _, group in efficiency.groupby(["dataset_id", "scenario_id", "severity"]):
        indices = group.index
        for index in indices:
            row = efficiency.loc[index]
            if not row["tta_censored"] and pd.notna(row["algorithm_wall_tta_s"]):
                dominated = group[
                    (group["algorithm_wall_tta_s"] <= row["algorithm_wall_tta_s"])
                    & (group["recovery_deficit_auc20"].fillna(np.inf) <= (row["recovery_deficit_auc20"] if pd.notna(row["recovery_deficit_auc20"]) else np.inf))
                    & (
                        (group["algorithm_wall_tta_s"] < row["algorithm_wall_tta_s"])
                        | (group["recovery_deficit_auc20"].fillna(np.inf) < (row["recovery_deficit_auc20"] if pd.notna(row["recovery_deficit_auc20"]) else np.inf))
                    )
                ]
                efficiency.loc[index, "pareto_time_descriptive"] = len(dominated) == 0
            dominated_comm = group[
                (group["communication_gib"] <= row["communication_gib"])
                & (group["last50_accuracy"] >= row["last50_accuracy"])
                & (
                    (group["communication_gib"] < row["communication_gib"])
                    | (group["last50_accuracy"] > row["last50_accuracy"])
                )
            ]
            efficiency.loc[index, "pareto_comm_descriptive"] = len(dominated_comm) == 0
    efficiency = efficiency.drop(columns=["bytes_upload_total", "bytes_download_total"])
    efficiency = _add_provenance(efficiency, provenance)
    atomic_parquet(PLOT_ROOT / "efficiency_fairness_summary.parquet", efficiency)

    paired_rows: list[dict[str, Any]] = []
    # Baseline-versus-FedAvg descriptive effects within each completed cell.
    for (dataset_id, scenario_id, seed), group in run_summary.groupby(["dataset_id", "scenario_id", "seed"]):
        reference = group[group["method_id"] == "FedAvg"]
        if len(reference) != 1:
            continue
        ref = reference.iloc[0]
        for row in group.itertuples():
            if row.method_id == "FedAvg":
                continue
            for metric, direction in (
                ("last50_accuracy", "higher_is_better"),
                ("recovery_deficit_auc20", "lower_is_better"),
            ):
                a = getattr(row, metric)
                b = ref[metric]
                if pd.isna(a) or pd.isna(b):
                    continue
                paired_rows.append(
                    {
                        "figure_id": "F3" if metric == "last50_accuracy" else "F4",
                        "metric": metric,
                        "dataset_id": dataset_id,
                        "scenario_id": scenario_id,
                        "severity": "base",
                        "method_a": row.method_id,
                        "method_b": "FedAvg",
                        "seed": int(seed),
                        "value_a": float(a),
                        "value_b": float(b),
                        "paired_difference": float(a - b),
                        "relative_difference": float((a - b) / abs(b)) if b != 0 else None,
                        "uncertainty_basis": "single_seed_paired_descriptive",
                        "effect_direction": direction,
                        "input_run_ids_hash": sha256_text("\n".join(sorted([row.run_id, ref["run_id"]]))),
                    }
                )
    paired = pd.DataFrame(paired_rows)
    paired = _add_provenance(paired, provenance)
    atomic_parquet(PLOT_ROOT / "paired_effects.parquet", paired)

    if (QUEUE_ROOT / "scheduler_events.parquet").exists():
        scheduler = pd.read_parquet(QUEUE_ROOT / "scheduler_events.parquet")
        atomic_parquet(PLOT_ROOT / "scheduler_events.parquet", scheduler)

    formal_expected = 77
    qc = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "baseline_completed_runs": len(manifests),
        "baseline_expected_runs": formal_expected,
        "complete": len(manifests) == formal_expected,
        "unique_run_ids": not manifests["run_id"].duplicated().any(),
        "source_kind_ok": bool((manifests["source_kind"] == "REPRODUCED").all()),
        "status_ok": bool((manifests["status"] == "completed").all()),
        "auc20_horizon_ok": bool(len(recovery_summary) == 0 or (recovery_summary["horizon_rounds"] == 20).all()),
        "legacy_auc_field_absent": "recovery_deficit_auc" not in recovery_summary.columns,
        "raw_row_counts": raw_counts,
        "run_ids_hash": provenance["input_run_ids_hash"],
    }
    atomic_json(PLOT_ROOT / "baseline_qc_report.json", qc)
    atomic_json(
        PLOT_ROOT / "baseline_data_provenance.json",
        {
            **provenance,
            "run_ids": sorted(run_ids),
            "outputs": {
                path.name: sha256_file(path)
                for path in PLOT_ROOT.glob("*.parquet")
                if path.name in {
                    "run_manifest.parquet",
                    "round_metrics.parquet",
                    "client_round_metrics.parquet",
                    "run_summary.parquet",
                    "recovery_summary.parquet",
                    "efficiency_fairness_summary.parquet",
                    "paired_effects.parquet",
                }
            },
        },
    )
    inventory_path = PLOT_ROOT / "plot_data_inventory.csv"
    if inventory_path.exists():
        inventory = pd.read_csv(inventory_path)
        created = {
            "run_manifest.parquet",
            "round_metrics.parquet",
            "client_round_metrics.parquet",
            "system_samples.parquet",
            "stage_timings.parquet",
            "run_summary.parquet",
            "recovery_summary.parquet",
            "efficiency_fairness_summary.parquet",
            "paired_effects.parquet",
        }
        status = "READY_BASELINE" if qc["complete"] else "READY_BASELINE_PARTIAL"
        inventory.loc[inventory["file_name"].isin(created), "status"] = status
        for trace_name in ("scenario_trace.parquet", "dynamic_events.csv", "data_partition_stats.parquet"):
            if (PLOT_ROOT / trace_name).exists():
                inventory.loc[inventory["file_name"] == trace_name, "status"] = "READY"
        atomic_csv(inventory_path, inventory)
    result = {
        "status": "completed",
        "completed_formal_baselines": len(manifests),
        "expected_formal_baselines": formal_expected,
        "qc_complete": qc["complete"],
        "outputs": str(PLOT_ROOT),
    }
    atomic_json(PLOT_ROOT / "baseline_aggregation_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(aggregate(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

