"""Derive Post-event Lower-Quartile Accuracy@20 from immutable artifacts.

LQA@20 is the linearly interpolated 25th percentile of test accuracy at exact
event offsets +1..+20, reported as a percentage.  It is a bounded, continuous
lower-tail diagnostic that is less sensitive to one anomalous round than the
post-event nadir.  This module is read-only with respect to formal run
directories and never changes the preregistered Recovery-deficit AUC@20 or any
frozen optimization gate.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .r2c_tail_recovery_margin20 import (
    ALL_METHODS,
    BASELINE_METHODS,
    DATASETS,
    SCENARIOS,
    SCENARIO_BLOCK,
    build_report as build_trm_report,
    derive_window_metrics,
    exact_event_windows,
    read_parquet_parts,
)


def derive_lqa20_percent(frame: pd.DataFrame) -> float:
    """Return the type-7 25th percentile of the exact post20 accuracies in %."""

    _, post = exact_event_windows(frame)
    return 100.0 * float(np.quantile(post, 0.25, method="linear"))


def _candidate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    pre, post = exact_event_windows(frame)
    window = derive_window_metrics(frame)
    return {
        "lqa20_percent": derive_lqa20_percent(frame),
        "pma20_percent": 100.0 * float(post.mean()),
        "post_median20_percent": 100.0 * float(np.median(post)),
        "post_nadir20_percent": 100.0 * float(post.min()),
        "pta20_percent": 100.0 * float(np.sort(post)[:5].mean()),
        "mean_retention20_percent": 100.0 * float(post.mean() / pre.mean()),
        "retained_rounds20_percent": 100.0 * float(np.mean(post >= pre.mean())),
        "mean_minus_sd20_percent": 100.0 * float(post.mean() - post.std(ddof=0)),
        "trm20_pp": float(window["trm20_pp"]),
        "signed_delta20_pp": float(window["signed_delta20_pp"]),
        "tail_shift20_pp": float(window["tail_shift20_pp"]),
        "q25_shift20_pp": float(window["q25_shift20_pp"]),
    }


def build_report(repo_root: Path) -> dict[str, object]:
    """Build the lineage-bound LQA@20 report for the four main-text tables."""

    baseline_path = repo_root / "figures" / "main_text_plot_data" / "round_metrics.parquet"
    baseline = pd.read_parquet(baseline_path)
    table_cache: dict[str, pd.DataFrame] = {}
    derived_records: list[dict[str, object]] = []

    def baseline_frame(run_id: str) -> pd.DataFrame:
        frame = baseline.loc[baseline["run_id"].eq(run_id)].copy()
        if frame.empty:
            raise KeyError(f"baseline run not found: {run_id}")
        return frame

    def r2c_frame(run_id: str) -> pd.DataFrame:
        if run_id not in table_cache:
            table_cache[run_id] = read_parquet_parts(
                repo_root / "experiments" / "runs" / run_id / "tables" / "round_metrics"
            )
        return table_cache[run_id]

    def record(source: str, run_id: str, frame: pd.DataFrame) -> dict[str, float]:
        values = _candidate_metrics(frame)
        derived_records.append({"source": source, "run_id": run_id, **values})
        return values

    table1: dict[str, dict[str, float]] = {}
    for dataset in DATASETS:
        dataset_values: dict[str, float] = {}
        for method in BASELINE_METHODS:
            run_id = f"B1b-{dataset}-S4-{method}-s20260811"
            dataset_values[method] = record(
                "baseline", run_id, baseline_frame(run_id)
            )["lqa20_percent"]
        run_id = f"A-R2C-V11MS-{dataset}-S4-B095-s20260811"
        dataset_values["R2C-FL"] = record("r2c", run_id, r2c_frame(run_id))[
            "lqa20_percent"
        ]
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
    comparison_rows: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        for method in BASELINE_METHODS:
            run_id = f"{SCENARIO_BLOCK[scenario]}-D2-{scenario}-{method}-s20260811"
            values = record("baseline", run_id, baseline_frame(run_id))
            table2[method][scenario] = values["lqa20_percent"]
            comparison_rows.append({"method": method, "scenario": scenario, **values})
        run_id = (
            f"A-R2C-T234-D2-{scenario}-FULL-s20260811"
            if scenario != "S4"
            else "A-R2C-V11MS-D2-S4-B095-s20260811"
        )
        values = record("r2c", run_id, r2c_frame(run_id))
        table2["R2C-FL"][scenario] = values["lqa20_percent"]
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
            table4b[variant][scenario] = record("r2c", run_id, r2c_frame(run_id))[
                "lqa20_percent"
            ]

    comparison = pd.DataFrame(comparison_rows)
    candidate_keys = (
        "lqa20_percent",
        "pma20_percent",
        "post_median20_percent",
        "post_nadir20_percent",
        "pta20_percent",
        "mean_retention20_percent",
        "retained_rounds20_percent",
        "mean_minus_sd20_percent",
        "trm20_pp",
        "signed_delta20_pp",
        "tail_shift20_pp",
        "q25_shift20_pp",
    )
    comparison_summary: dict[str, dict[str, float | int | str]] = {}
    for key in candidate_keys:
        values = comparison[key]
        exact_unique = np.sort(values.unique())
        gaps = np.diff(exact_unique)
        comparison_summary[key] = {
            "unique_at_0.001": int(values.round(3).nunique()),
            "unique_at_0.01": int(values.round(2).nunique()),
            "cell_count": int(len(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "range": float(values.max() - values.min()),
            "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
            "population_sd": float(values.std(ddof=0)),
            "minimum_exact_pair_gap": float(gaps.min()) if len(gaps) else 0.0,
        }
    comparison_summary["lqa20_percent"]["decision"] = (
        "selected: 32/32 Table-2 cells distinct at two decimals, minimum exact "
        "pair gap 0.03 percentage points, bounded and non-clipped, and more "
        "outlier-robust than the single-round nadir"
    )
    comparison_summary["post_nadir20_percent"]["decision"] = (
        "rejected despite high spread because a single anomalous round determines the value"
    )
    comparison_summary["pma20_percent"]["decision"] = (
        "rejected for display because only 31/32 cells remain unique at two decimals"
    )
    comparison_summary["mean_retention20_percent"]["decision"] = (
        "rejected because the pre-window denominator can reward a weak pre-event level"
    )

    trm_report = build_trm_report(repo_root)
    return {
        "schema_version": "r2c-posthoc-post-event-lower-quartile-accuracy20-v1",
        "generated_local": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "formal_run_mutations": False,
        "formal_gate_mutations": False,
        "metric": {
            "name": "post_event_lower_quartile_accuracy20",
            "display_name": "Post-event Lower-Quartile Accuracy@20",
            "short_name": "LQA@20",
            "formula_percent": (
                "100 * type-7 linear Q0.25(test accuracy at exact event offsets +1..+20)"
            ),
            "direction": "higher_is_better",
            "range_percent": [0.0, 100.0],
            "window": "exactly 20 completed rounds after the frozen event",
            "quantile": 0.25,
            "quantile_method": "linear (Hyndman-Fan type 7; NumPy method=linear)",
            "interpretation": (
                "descriptive lower-quartile post-event predictive performance; "
                "not causal or counterfactual"
            ),
            "status_completed_tables": (
                "post_hoc_selected_after_user_requested_a_different_high_discrimination_metric"
            ),
            "status_frozen_gates": "unchanged",
        },
        "selection_comparison_on_table2_32_cells": comparison_summary,
        "raw_table2_candidate_metrics": comparison_rows,
        "raw_derived_metrics": derived_records,
        "table1_lqa20_percent": table1,
        "table2_lqa20_percent": table2,
        "table3_s4_lqa20_percent": {
            method: table1["D2"][method] for method in ALL_METHODS
        },
        "table4b_lqa20_percent": table4b,
        "table4a_operational_diagnostics": trm_report["table4a_operational_diagnostics"],
        "table4b_operational_diagnostics": trm_report["table4b_operational_diagnostics"],
        "source_lineage": trm_report["source_lineage"],
        "audit": {
            "all_windows_complete": True,
            "derived_records": len(derived_records),
            "table2_cells": len(comparison_rows),
            "table2_lqa_unique_at_0.001": int(
                comparison["lqa20_percent"].round(3).nunique()
            ),
            "table2_lqa_unique_at_0.01": int(
                comparison["lqa20_percent"].round(2).nunique()
            ),
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
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "timestamped_output": (
                    str(args.timestamped_output.resolve())
                    if args.timestamped_output
                    else None
                ),
                "derived_records": report["audit"]["derived_records"],
                "all_windows_complete": report["audit"]["all_windows_complete"],
                "table2_lqa_unique_at_0.01": report["audit"][
                    "table2_lqa_unique_at_0.01"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
