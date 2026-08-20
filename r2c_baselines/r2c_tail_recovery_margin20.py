"""Derive the post-hoc Tail-Recovery Margin@20 from immutable run artifacts.

This module never writes into a run directory.  It reads the exact pre20 and
post20 windows already recorded by formal runs, validates their offsets, and
emits a lineage-bound descriptive report for Tables 1--4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASELINE_METHODS = (
    "FedAvg",
    "FedAU",
    "F3AST",
    "FedAWE",
    "PowerOfChoice",
    "Oort",
    "TiFL",
)
ALL_METHODS = BASELINE_METHODS + ("R2C-FL",)
DATASETS = ("D1", "D2", "D3", "D4")
SCENARIOS = ("S1", "S2", "S3", "S4")
SCENARIO_BLOCK = {"S1": "B2a", "S2": "B2b", "S3": "B2c", "S4": "B1b"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_parquet_parts(directory: Path) -> pd.DataFrame:
    parts = sorted(directory.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no Parquet parts under {directory}")
    return pd.concat((pd.read_parquet(part) for part in parts), ignore_index=True)


def exact_event_windows(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = {"test_accuracy", "event_offset_round", "auc20_window_role"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing event-window columns: {sorted(missing)}")

    pre = frame.loc[
        frame["auc20_window_role"].eq("pre20"),
        ["event_offset_round", "test_accuracy"],
    ].sort_values("event_offset_round")
    post = frame.loc[
        frame["auc20_window_role"].eq("post20"),
        ["event_offset_round", "test_accuracy"],
    ].sort_values("event_offset_round")

    expected_pre = np.arange(-20, 0, dtype=float)
    expected_post = np.arange(1, 21, dtype=float)
    if len(pre) != 20 or not np.array_equal(pre["event_offset_round"].to_numpy(), expected_pre):
        raise ValueError("pre20 must contain exact offsets -20..-1")
    if len(post) != 20 or not np.array_equal(post["event_offset_round"].to_numpy(), expected_post):
        raise ValueError("post20 must contain exact offsets +1..+20")

    pre_accuracy = pre["test_accuracy"].to_numpy(dtype=float)
    post_accuracy = post["test_accuracy"].to_numpy(dtype=float)
    if not np.isfinite(pre_accuracy).all() or not np.isfinite(post_accuracy).all():
        raise ValueError("event windows contain non-finite accuracy")
    return pre_accuracy, post_accuracy


def derive_window_metrics(frame: pd.DataFrame, *, tail_k: int = 5) -> dict[str, float]:
    if tail_k <= 0 or tail_k > 20:
        raise ValueError("tail_k must be in 1..20")
    pre, post = exact_event_windows(frame)
    post_sorted = np.sort(post)
    pre_sorted = np.sort(pre)
    return {
        "trm20_pp": 100.0 * (float(post_sorted[:tail_k].mean()) - float(pre.mean())),
        "signed_delta20_pp": 100.0 * (float(post.mean()) - float(pre.mean())),
        "median_shift20_pp": 100.0 * (float(np.median(post)) - float(np.median(pre))),
        "tail_shift20_pp": 100.0
        * (float(post_sorted[:tail_k].mean()) - float(pre_sorted[:tail_k].mean())),
        "q25_shift20_pp": 100.0
        * (float(np.quantile(post, 0.25)) - float(np.quantile(pre, 0.25))),
    }


def composite_parts_hash(
    repo_root: Path,
    run_ids: Iterable[str],
    *,
    table_name: str,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for run_id in sorted(set(run_ids)):
        directory = repo_root / "experiments" / "runs" / run_id / "tables" / table_name
        for part in sorted(directory.glob("part-*.parquet")):
            relative = part.relative_to(repo_root).as_posix().encode("utf-8")
            digest.update(relative)
            digest.update(b"\0")
            digest.update(bytes.fromhex(sha256_file(part)))
            digest.update(b"\n")
            count += 1
    return count, digest.hexdigest().upper()


def build_report(repo_root: Path) -> dict[str, object]:
    baseline_path = repo_root / "figures" / "main_text_plot_data" / "round_metrics.parquet"
    manifest_path = repo_root / "figures" / "main_text_plot_data" / "run_manifest.parquet"
    baseline = pd.read_parquet(baseline_path)
    table_cache: dict[tuple[str, str], pd.DataFrame] = {}
    audited_records: list[dict[str, object]] = []

    def baseline_frame(run_id: str) -> pd.DataFrame:
        frame = baseline.loc[baseline["run_id"].eq(run_id)].copy()
        if frame.empty:
            raise KeyError(f"baseline run not found: {run_id}")
        return frame

    def r2c_table(run_id: str, table_name: str) -> pd.DataFrame:
        key = (run_id, table_name)
        if key not in table_cache:
            table_cache[key] = read_parquet_parts(
                repo_root / "experiments" / "runs" / run_id / "tables" / table_name
            )
        return table_cache[key]

    def r2c_frame(run_id: str) -> pd.DataFrame:
        return r2c_table(run_id, "round_metrics")

    def record(source: str, run_id: str, frame: pd.DataFrame) -> dict[str, float]:
        values = derive_window_metrics(frame)
        audited_records.append({"source": source, "run_id": run_id, **values})
        return values

    table1: dict[str, dict[str, float]] = {}
    for dataset in DATASETS:
        dataset_values: dict[str, float] = {}
        for method in BASELINE_METHODS:
            run_id = f"B1b-{dataset}-S4-{method}-s20260811"
            dataset_values[method] = record("baseline", run_id, baseline_frame(run_id))["trm20_pp"]
        run_id = f"A-R2C-V11MS-{dataset}-S4-B095-s20260811"
        dataset_values["R2C-FL"] = record("r2c", run_id, r2c_frame(run_id))["trm20_pp"]
        table1[dataset] = dataset_values
    table1["macro_mean"] = {
        method: float(np.mean([table1[dataset][method] for dataset in DATASETS]))
        for method in ALL_METHODS
    }
    table1["worst_case"] = {
        method: float(np.min([table1[dataset][method] for dataset in DATASETS]))
        for method in ALL_METHODS
    }

    table2: dict[str, dict[str, float]] = {method: {} for method in ALL_METHODS}
    comparison_rows: list[dict[str, float | str]] = []
    for scenario in SCENARIOS:
        for method in BASELINE_METHODS:
            run_id = f"{SCENARIO_BLOCK[scenario]}-D2-{scenario}-{method}-s20260811"
            values = record("baseline", run_id, baseline_frame(run_id))
            table2[method][scenario] = values["trm20_pp"]
            comparison_rows.append({"method": method, "scenario": scenario, **values})
        run_id = (
            f"A-R2C-T234-D2-{scenario}-FULL-s20260811"
            if scenario != "S4"
            else "A-R2C-V11MS-D2-S4-B095-s20260811"
        )
        values = record("r2c", run_id, r2c_frame(run_id))
        table2["R2C-FL"][scenario] = values["trm20_pp"]
        comparison_rows.append({"method": "R2C-FL", "scenario": scenario, **values})
    for method in ALL_METHODS:
        scenario_values = [table2[method][scenario] for scenario in SCENARIOS]
        table2[method]["mean"] = float(np.mean(scenario_values))
        table2[method]["worst"] = float(np.min(scenario_values))

    ablation_runs = {
        "full": {
            "S3": "A-R2C-T234-D2-S3-FULL-s20260811",
            "S4": "A-R2C-V11MS-D2-S4-B095-s20260811",
        },
        "no_reusable_prefix": {
            "S3": "A-R2C-T234-D2-S3-A1-NOPREFIX-s20260811",
            "S4": "A-R2C-T234-D2-S4-A1-NOPREFIX-s20260811",
        },
        "no_finishability": {
            "S3": "A-R2C-T234-D2-S3-A2-NOFINISH-s20260811",
            "S4": "A-R2C-T234-D2-S4-A2-NOFINISH-s20260811",
        },
        "no_drift_quarantine": {
            "S3": "A-R2C-T234-D2-S3-A3-NOQUAR-s20260811",
            "S4": "A-R2C-T234-D2-S4-A3-NOQUAR-s20260811",
        },
        "no_valid_crossfit": {
            "S3": "A-R2C-T234-D2-S3-A4-NOCROSSFIT-s20260811",
            "S4": "A-R2C-T234-D2-S4-A4-NOCROSSFIT-s20260811",
        },
    }
    table4b: dict[str, dict[str, float]] = {}
    for variant, scenarios in ablation_runs.items():
        table4b[variant] = {}
        for scenario, run_id in scenarios.items():
            table4b[variant][scenario] = record("r2c", run_id, r2c_frame(run_id))["trm20_pp"]

    table4a_operational: dict[str, dict[str, float]] = {}
    for dataset in DATASETS:
        run_id = f"A-R2C-V11MS-{dataset}-S4-B095-s20260811"
        certificates = r2c_table(run_id, "certificate_audit")
        systems = r2c_table(run_id, "system_samples")
        table4a_operational[dataset] = {
            "topk_overlap_percent": 100.0
            * float(certificates["topk_intersection_size"].mean())
            / 10.0,
            "anchor_rank_agreement_percent": 100.0
            * float(certificates["anchor_rank_agreement"].mean()),
            "hajek_effective_sample_size": float(
                certificates["hajek_effective_sample_size"].mean()
            ),
            "candidate_compute_saved_percent": 100.0
            * float(certificates["candidate_compute_saved_fraction"].mean()),
            "audit_gpu_hours": float(certificates["replay_gpu_s"].sum()) / 3600.0,
            "peak_vram_gib": float(systems["memory_reserved_mib"].max()) / 1024.0,
            "omitted_certified_percent": 100.0 * float(certificates["certified"].mean()),
        }

    table4b_operational: dict[str, dict[str, float]] = {}
    for variant, scenarios in ablation_runs.items():
        run_id = scenarios["S4"]
        certificates = r2c_table(run_id, "certificate_audit")
        table4b_operational[variant] = {
            "mean_certificate_margin_gamma": float(certificates["gamma_at_commit"].mean()),
            "omitted_certified_percent": 100.0 * float(certificates["certified"].mean()),
        }

    comparison = pd.DataFrame(comparison_rows)
    comparison_summary = {}
    for key in (
        "signed_delta20_pp",
        "median_shift20_pp",
        "tail_shift20_pp",
        "q25_shift20_pp",
        "trm20_pp",
    ):
        values = comparison[key]
        comparison_summary[key] = {
            "unique_at_0.001_pp": int(values.round(3).nunique()),
            "cell_count": int(len(values)),
            "min_pp": float(values.min()),
            "max_pp": float(values.max()),
            "iqr_pp": float(values.quantile(0.75) - values.quantile(0.25)),
        }
    comparison_summary["trm20_pp"]["decision"] = (
        "selected: non-clipped, complete, five-round lower-tail robust, and directly interpretable"
    )

    all_r2c_run_ids = {run_id for run_id, _ in table_cache}
    round_part_count, round_part_hash = composite_parts_hash(
        repo_root,
        all_r2c_run_ids,
        table_name="round_metrics",
    )
    certificate_part_count, certificate_part_hash = composite_parts_hash(
        repo_root,
        all_r2c_run_ids,
        table_name="certificate_audit",
    )
    system_part_count, system_part_hash = composite_parts_hash(
        repo_root,
        all_r2c_run_ids,
        table_name="system_samples",
    )
    return {
        "schema_version": "r2c-posthoc-tail-recovery-margin20-v1",
        "generated_local": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "formal_run_mutations": False,
        "metric": {
            "name": "tail_recovery_margin20",
            "display_name": "Tail-Recovery Margin@20",
            "short_name": "TRM@20",
            "formula_pp": (
                "100 * (mean(five lowest test accuracies in exact post20) "
                "- mean(test accuracy in exact pre20))"
            ),
            "direction": "higher_is_better",
            "window": "exactly 20 completed rounds before and after the frozen event",
            "tail_size": 5,
            "interpretation": "descriptive lower-tail recovery stability; not causal or counterfactual",
            "status_completed_tables": "post_hoc_selected_after_observing_auc_floor_saturation",
            "status_v12_and_later": "prospectively_frozen_before_candidate_runs",
        },
        "selection_comparison_on_table2_32_cells": comparison_summary,
        "table1_trm20_pp": table1,
        "table2_trm20_pp": table2,
        "table3_s4_trm20_pp": {method: table1["D2"][method] for method in ALL_METHODS},
        "table4b_trm20_pp": table4b,
        "table4a_operational_diagnostics": table4a_operational,
        "table4b_operational_diagnostics": table4b_operational,
        "source_lineage": {
            "baseline_round_metrics": {
                "path": str(baseline_path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": sha256_file(baseline_path),
            },
            "baseline_run_manifest": {
                "path": str(manifest_path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": sha256_file(manifest_path),
            },
            "r2c_round_metric_parts_count": round_part_count,
            "r2c_round_metric_parts_composite_sha256": round_part_hash,
            "r2c_certificate_audit_parts_count": certificate_part_count,
            "r2c_certificate_audit_parts_composite_sha256": certificate_part_hash,
            "r2c_system_sample_parts_count": system_part_count,
            "r2c_system_sample_parts_composite_sha256": system_part_hash,
            "r2c_run_ids": sorted(all_r2c_run_ids),
        },
        "audit": {
            "all_windows_complete": True,
            "derived_records": len(audited_records),
            "table2_cells": len(comparison_rows),
            "formal_training_rerun": False,
            "formal_results_overwritten": False,
            "independent_cross_model_review": "waived_by_user",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamped-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.repo_root.resolve())
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    if args.timestamped_output:
        args.timestamped_output.parent.mkdir(parents=True, exist_ok=True)
        args.timestamped_output.write_text(payload, encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "timestamped_output": str(args.timestamped_output.resolve()) if args.timestamped_output else None,
        "derived_records": report["audit"]["derived_records"],
        "all_windows_complete": report["audit"]["all_windows_complete"],
    }, indent=2))


if __name__ == "__main__":
    main()
