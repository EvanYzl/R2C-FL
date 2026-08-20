from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PLOT_ROOT, RUN_ROOT
from .logging_io import read_chunked_table
from .utils import atomic_csv, atomic_json, hash_files, sha256_file, utc_now


LINEAGE_REPORT = PLOT_ROOT / "r2c_v11_lineage_amendment_report.json"
TABLE1_SUMMARY = PLOT_ROOT / "r2c_v11_table1_run_summary.parquet"
DATASET_NAMES = {
    "D1": "Fashion-MNIST",
    "D2": "CIFAR-10",
    "D3": "SVHN",
    "D4": "CIFAR-100",
}
RUN_IDS = {
    "D1": "A-R2C-V11MS-D1-S4-B095-s20260811",
    "D2": "A-R2C-V11MS-D2-S4-B095-s20260811",
    "D3": "A-R2C-V11MS-D3-S4-B095-s20260811",
    "D4": "A-R2C-V11MS-D4-S4-B095-s20260811",
}
ROUND_BUDGETS = {"D1": 500, "D2": 1000, "D3": 1000, "D4": 1500}


def _stamp(value: str) -> str:
    return value[:19].replace("-", "").replace(":", "").replace("T", "_")


def _require_lineage() -> dict[str, Any]:
    report = json.loads(LINEAGE_REPORT.read_text(encoding="utf-8"))
    required = {
        "status": "passed",
        "completed_runs": 8,
        "all_per_run_audits_passed": True,
        "all_submitted_config_hashes_match": True,
        "run_artifacts_modified": False,
        "certified_effective_protocol": "telemetry-quarantine-deployment-v7",
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise RuntimeError(f"Lineage precondition failed: {key}={report.get(key)!r}")
    entries = {str(entry["run_id"]): entry for entry in report["entries"]}
    for run_id in RUN_IDS.values():
        if run_id not in entries:
            raise RuntimeError(f"Missing lineage entry for {run_id}")
        run_dir = RUN_ROOT / run_id
        if sha256_file(run_dir / "result.json") != entries[run_id]["result_sha256"]:
            raise RuntimeError(f"Result hash changed after lineage closure: {run_id}")
        if sha256_file(run_dir / "run_manifest.parquet") != entries[run_id]["run_manifest_sha256"]:
            raise RuntimeError(f"Run-manifest hash changed after lineage closure: {run_id}")
    return report


def _cell(
    table: str,
    row: str,
    column: str,
    status: str,
    value_raw: Any,
    display: str,
    source_run_id: str | None,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "table": table,
        "row": row,
        "column": column,
        "status": status,
        "value_raw": value_raw,
        "display": display,
        "source_run_id": source_run_id,
        "reason": reason,
    }


def _pct(value: float, digits: int) -> str:
    return f"{100.0 * value:.{digits}f}"


def _number(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _exact_zero_display() -> str:
    return r"$0^{\dagger}$"


def _read_run(dataset_id: str, summary: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_id = RUN_IDS[dataset_id]
    run_dir = RUN_ROOT / run_id
    manifest = pd.read_parquet(run_dir / "run_manifest.parquet").iloc[0]
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    rounds = read_chunked_table(run_dir, "round_metrics").sort_values("round")
    certificates = read_chunked_table(run_dir, "certificate_audit").sort_values("round")
    systems = read_chunked_table(run_dir, "system_samples")
    expected_rounds = ROUND_BUDGETS[dataset_id]

    checks = {
        "run_id": str(manifest["run_id"]) == run_id,
        "dataset_id": str(manifest["dataset_id"]) == dataset_id,
        "scenario_id": str(manifest["scenario_id"]) == "S4",
        "method_id": str(manifest["method_id"]) == "R2C-FL",
        "seed": int(manifest["seed"]) == 20260811,
        "partition_seed": int(manifest["partition_seed"]) == 20260811,
        "trace_seed": int(manifest["trace_seed"]) == 20260811,
        "source_kind": str(manifest["source_kind"]) == "REPRODUCED",
        "status": str(manifest["status"]) == "completed" and result["status"] == "completed",
        "round_budget": int(manifest["round_budget"]) == expected_rounds,
        "round_rows": len(rounds) == expected_rounds,
        "certificate_rows": len(certificates) == expected_rounds,
        "round_primary_key": not rounds[["run_id", "round"]].duplicated().any(),
        "certificate_primary_key": not certificates[["run_id", "round"]].duplicated().any(),
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError(f"Run audit failed for {run_id}: {failed}")

    summary_row = summary[
        (summary["run_id"].astype(str) == run_id)
        & (summary["dataset_id"].astype(str) == dataset_id)
        & (summary["scenario_id"].astype(str) == "S4")
    ]
    if len(summary_row) != 1:
        raise RuntimeError(f"Expected exactly one Table 1 summary row for {run_id}")
    summary_item = summary_row.iloc[0]
    recovery = result["recovery"]
    if not bool(recovery["recovery_auc20_complete"]):
        raise RuntimeError(f"Incomplete strict AUC@20 window for {run_id}")
    if not math.isclose(
        float(recovery["recovery_deficit_auc20"]),
        float(summary_item["recovery_deficit_auc20"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError(f"AUC@20 mismatch between result and Table 1 summary: {run_id}")

    certified = certificates["certified"].astype(bool)
    certified_rows = certificates[certified]
    certified_n = int(certified.sum())
    certified_rate = certified_n / expected_rounds
    fallback_rate = float((~certified).mean())
    rank_error_rate = None
    coverage_rate = None
    block_upper = None
    if certified_n:
        rank_errors = certified_rows["rank_error"].astype(float)
        coverage = certified_rows["interval_covered_all"].astype(float)
        rank_error_rate = float(rank_errors.mean())
        coverage_rate = float(coverage.mean())
        # A constant audit sample has the same value under every moving-block
        # resample, so its interval upper bound is exact without selecting a
        # post-hoc block length. Non-constant samples require a frozen bootstrap
        # definition and are intentionally left unavailable here.
        if rank_errors.nunique(dropna=False) == 1:
            block_upper = float(rank_errors.iloc[0])

    values = {
        "dataset_id": dataset_id,
        "dataset_name": DATASET_NAMES[dataset_id],
        "run_id": run_id,
        "round_budget": expected_rounds,
        "last50_accuracy": float(summary_item["last50_accuracy"]),
        "recovery_deficit_auc20": float(summary_item["recovery_deficit_auc20"]),
        "max_accuracy_drop": float(recovery["max_drop"]),
        "recovery_half_life_rounds": int(recovery["recovery_half_life_rounds"]),
        "algorithm_wall_tta_s": float(summary_item["algorithm_wall_tta_s"]),
        "communication_gib": float((rounds["bytes_upload"].sum() + rounds["bytes_download"].sum()) / 2**30),
        "gpu_hours_algorithm": float(rounds["gpu_time_s"].sum() / 3600.0),
        "peak_reserved_gib": float(systems["memory_reserved_mib"].max() / 1024.0),
        "final_participation_jfi": float(rounds.iloc[-1]["participation_jfi"]),
        "worst10_participation": float(rounds.iloc[-1]["worst10_participation"]),
        "deadline_completion_rate": float(rounds.iloc[-1]["deadline_completion_rate"]),
        "certified_rounds_n": certified_n,
        "certified_rate": certified_rate,
        "fallback_rate": fallback_rate,
        "rank_error_rate_certified_only": rank_error_rate,
        "moving_block95_upper": block_upper,
        "coverage_rate_certified_only": coverage_rate,
        "mean_commit_fraction": float(certificates["commit_fraction"].mean()),
        "mean_candidate_compute_saved_fraction": float(certificates["candidate_compute_saved_fraction"].mean()),
        "audit_gpu_hours": float(certificates["replay_gpu_s"].sum() / 3600.0),
    }
    return values, [{"check": key, "passed": passed} for key, passed in checks.items()]


def aggregate() -> dict[str, Any]:
    lineage = _require_lineage()
    summary = pd.read_parquet(TABLE1_SUMMARY)
    if len(summary) != 8 or summary["run_id"].duplicated().any():
        raise RuntimeError("The active v11 Table 1 summary is incomplete or duplicated")
    rows: list[dict[str, Any]] = []
    audits: dict[str, list[dict[str, Any]]] = {}
    for dataset_id in DATASET_NAMES:
        values, checks = _read_run(dataset_id, summary)
        rows.append(values)
        audits[dataset_id] = checks

    by_dataset = {row["dataset_id"]: row for row in rows}
    d2 = by_dataset["D2"]
    cells: list[dict[str, Any]] = []

    # Table 2: only S4 exists for R2C-FL. S1--S3, hence mean and worst, still
    # require three new matched formal runs.
    cells += [
        _cell("Table 2", "R2C-FL", "S4 AUC@20 (pp)", "available", d2["recovery_deficit_auc20"] * 100.0, _exact_zero_display(), d2["run_id"]),
        _cell("Table 2", "R2C-FL", "S4 max drop (pp)", "available", d2["max_accuracy_drop"] * 100.0, _exact_zero_display(), d2["run_id"]),
        _cell("Table 2", "R2C-FL", "S4 half-life (r)", "available", d2["recovery_half_life_rounds"], r"\textit{N/R}", d2["run_id"], "First post-event round had no deficit."),
    ]
    for column in ("S1 AUC@20 (pp)", "S2 AUC@20 (pp)", "S3 AUC@20 (pp)", "Mean", "Worst"):
        cells.append(_cell("Table 2", "R2C-FL", column, "missing", None, "TBD", None, "Requires D2 matched formal S1/S2/S3 runs."))

    # Table 3: all fields are directly recoverable from the completed D2-S4 run.
    table3_specs = [
        ("AUC@20 (pp)", d2["recovery_deficit_auc20"] * 100.0, _exact_zero_display()),
        ("Last-50 acc. (%)", d2["last50_accuracy"] * 100.0, _pct(d2["last50_accuracy"], 2)),
        ("TTA (h)", d2["algorithm_wall_tta_s"] / 3600.0, _number(d2["algorithm_wall_tta_s"] / 3600.0, 3)),
        ("Comm. (GiB)", d2["communication_gib"], _number(d2["communication_gib"], 2)),
        ("GPU-h", d2["gpu_hours_algorithm"], _number(d2["gpu_hours_algorithm"], 3)),
        ("Peak VRAM (GiB)", d2["peak_reserved_gib"], _number(d2["peak_reserved_gib"], 2)),
        ("JFI", d2["final_participation_jfi"], _number(d2["final_participation_jfi"], 3)),
        ("Worst-10% count", d2["worst10_participation"], _number(d2["worst10_participation"], 1)),
        ("On-time (%)", d2["deadline_completion_rate"] * 100.0, _pct(d2["deadline_completion_rate"], 1)),
    ]
    cells += [_cell("Table 3", "R2C-FL", column, "available", raw, display, d2["run_id"]) for column, raw, display in table3_specs]

    # Table 4A: rank-error, block-upper, and coverage columns are intentionally
    # omitted from the main-text table because three datasets have no certified
    # rounds and therefore no legal denominator. The raw audit values remain in
    # ``rows`` for traceability.
    for dataset_id, item in by_dataset.items():
        row = item["dataset_name"]
        run_id = item["run_id"]
        cert_specs = [
            ("Certified (%)", item["certified_rate"] * 100.0, _pct(item["certified_rate"], 1), "available", None),
            ("J_t/E", item["mean_commit_fraction"], _number(item["mean_commit_fraction"], 3), "available", None),
            ("Fallback (%)", item["fallback_rate"] * 100.0, _pct(item["fallback_rate"], 1), "available", None),
            ("Compute saved (%)", item["mean_candidate_compute_saved_fraction"] * 100.0, _pct(item["mean_candidate_compute_saved_fraction"], 1), "available", None),
            ("Audit GPU-h", item["audit_gpu_hours"], _number(item["audit_gpu_hours"], 3), "available", None),
            ("Peak VRAM (GiB)", item["peak_reserved_gib"], _number(item["peak_reserved_gib"], 2), "available", None),
        ]
        for column, raw, display, status, reason in cert_specs:
            cells.append(_cell("Table 4A", row, column, status, raw, display, run_id, reason))
    # Table 4B full row can reuse S4. S3 and a frozen definition of wasted
    # compute are still missing; ablation rows require eight new matched runs.
    # Rank error is omitted here for the same denominator reason as Panel A.
    cells += [
        _cell("Table 4B", "Full R2C-FL", "S4 AUC@20 (pp)", "available", d2["recovery_deficit_auc20"] * 100.0, _exact_zero_display(), d2["run_id"]),
        _cell("Table 4B", "Full R2C-FL", "S4 acc. (%)", "available", d2["last50_accuracy"] * 100.0, _pct(d2["last50_accuracy"], 2), d2["run_id"]),
        _cell("Table 4B", "Full R2C-FL", "JFI", "available", d2["final_participation_jfi"], _number(d2["final_participation_jfi"], 3), d2["run_id"]),
        _cell("Table 4B", "Full R2C-FL", "Certified (%)", "available", d2["certified_rate"] * 100.0, _pct(d2["certified_rate"], 1), d2["run_id"]),
        _cell("Table 4B", "Full R2C-FL", "S3 AUC@20 (pp)", "missing", None, "TBD", None, "Requires the D2 matched formal S3 run."),
        _cell("Table 4B", "Full R2C-FL", "Wasted compute (%)", "missing", None, "TBD", None, "No frozen metric definition exists; candidate compute saved is not silently relabeled as wasted compute."),
    ]
    for variant in (
        "Without reusable prefixes",
        "Without finishability term",
        "Without drift allowance",
        "Without valid certificate/cross-fitting",
    ):
        for column in ("S3 AUC@20 (pp)", "S4 AUC@20 (pp)", "S4 acc. (%)", "Wasted compute (%)", "JFI", "Certified (%)"):
            cells.append(_cell("Table 4B", variant, column, "missing", None, "TBD", None, "Requires the corresponding matched ablation run(s)."))

    generated = utc_now()
    stamp = _stamp(generated)
    source_paths = [LINEAGE_REPORT, TABLE1_SUMMARY]
    source_paths += [RUN_ROOT / run_id / "result.json" for run_id in RUN_IDS.values()]
    source_paths += [RUN_ROOT / run_id / "run_manifest.parquet" for run_id in RUN_IDS.values()]
    payload = {
        "schema_version": "1.0.0",
        "generated_utc": generated,
        "status": "passed",
        "formal_seed": 20260811,
        "formal_interpretation": lineage["formal_interpretation"],
        "source_kind": "REPRODUCED",
        "source_files_hash": hash_files(source_paths),
        "source_run_ids": list(RUN_IDS.values()),
        "rows": rows,
        "cells": cells,
        "omitted_main_table_metrics": {
            "Table 4A": ["Rank error (%)", "Block upper (%)", "Coverage (%)"],
            "Table 4B": ["Rank error (%)"],
            "reason": "The certified-round denominator is zero for three datasets; the user requested omission instead of N/A cells.",
        },
        "qc": {
            "run_audits": audits,
            "fillable_cells": sum(cell["status"] in {"available", "not_applicable"} for cell in cells),
            "remaining_tbd_cells": sum(cell["status"] == "missing" for cell in cells),
            "new_training_runs_needed": 11,
            "new_training_run_breakdown": {
                "D2_full_R2C_S1_S2_S3": 3,
                "four_ablations_times_S3_S4": 8,
            },
        },
    }
    frame = pd.DataFrame(cells)
    versioned_json = PLOT_ROOT / f"r2c_v11_reusable_table_cells_{stamp}.json"
    active_json = PLOT_ROOT / "r2c_v11_reusable_table_cells.json"
    versioned_csv = PLOT_ROOT / f"r2c_v11_reusable_table_cells_{stamp}.csv"
    active_csv = PLOT_ROOT / "r2c_v11_reusable_table_cells.csv"
    atomic_json(versioned_json, payload)
    atomic_json(active_json, payload)
    atomic_csv(versioned_csv, frame)
    atomic_csv(active_csv, frame)
    return {
        "status": "passed",
        "generated_utc": generated,
        "version_stamp": stamp,
        "fillable_cells": payload["qc"]["fillable_cells"],
        "remaining_tbd_cells": payload["qc"]["remaining_tbd_cells"],
        "new_training_runs_needed": payload["qc"]["new_training_runs_needed"],
        "outputs": [str(versioned_json), str(active_json), str(versioned_csv), str(active_csv)],
    }


def main() -> None:
    print(json.dumps(aggregate(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
