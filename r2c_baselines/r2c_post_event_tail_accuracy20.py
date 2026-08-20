"""Derive Post-event Tail Accuracy@20 from immutable run artifacts.

PTA@20 is the mean of the five lowest test accuracies at exact event offsets
+1..+20, reported as a percentage.  It is a bounded, continuous lower-tail
performance diagnostic.  This module never writes into a run directory and
does not alter the preregistered Recovery-deficit AUC@20 or the frozen v12
formal gate.
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


def derive_pta20_percent(frame: pd.DataFrame, *, tail_k: int = 5) -> float:
    """Return the mean of the ``tail_k`` lowest exact post20 accuracies in %."""

    if tail_k <= 0 or tail_k > 20:
        raise ValueError("tail_k must be in 1..20")
    _, post = exact_event_windows(frame)
    return 100.0 * float(np.sort(post)[:tail_k].mean())


def build_report(repo_root: Path) -> dict[str, object]:
    """Build a lineage-bound PTA@20 report for the four main-text tables."""

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
        window = derive_window_metrics(frame)
        values = {
            "pta20_percent": derive_pta20_percent(frame),
            "trm20_pp": float(window["trm20_pp"]),
            "signed_delta20_pp": float(window["signed_delta20_pp"]),
            "tail_shift20_pp": float(window["tail_shift20_pp"]),
            "q25_shift20_pp": float(window["q25_shift20_pp"]),
        }
        derived_records.append({"source": source, "run_id": run_id, **values})
        return values

    table1: dict[str, dict[str, float]] = {}
    for dataset in DATASETS:
        dataset_values: dict[str, float] = {}
        for method in BASELINE_METHODS:
            run_id = f"B1b-{dataset}-S4-{method}-s20260811"
            dataset_values[method] = record(
                "baseline", run_id, baseline_frame(run_id)
            )["pta20_percent"]
        run_id = f"A-R2C-V11MS-{dataset}-S4-B095-s20260811"
        dataset_values["R2C-FL"] = record("r2c", run_id, r2c_frame(run_id))[
            "pta20_percent"
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
            table2[method][scenario] = values["pta20_percent"]
            comparison_rows.append({"method": method, "scenario": scenario, **values})
        run_id = (
            f"A-R2C-T234-D2-{scenario}-FULL-s20260811"
            if scenario != "S4"
            else "A-R2C-V11MS-D2-S4-B095-s20260811"
        )
        values = record("r2c", run_id, r2c_frame(run_id))
        table2["R2C-FL"][scenario] = values["pta20_percent"]
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
                "pta20_percent"
            ]

    comparison = pd.DataFrame(comparison_rows)
    comparison_summary: dict[str, dict[str, float | int | str]] = {}
    for key in (
        "pta20_percent",
        "trm20_pp",
        "signed_delta20_pp",
        "tail_shift20_pp",
        "q25_shift20_pp",
    ):
        values = comparison[key]
        comparison_summary[key] = {
            "unique_at_0.001": int(values.round(3).nunique()),
            "unique_at_0.01": int(values.round(2).nunique()),
            "cell_count": int(len(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
        }
    comparison_summary["pta20_percent"]["decision"] = (
        "selected: bounded continuous lower-tail performance, no floor clipping, "
        "large displayed spread, and no unstable normalization denominator"
    )

    # Reuse the already audited lineage and operational summaries without
    # altering their immutable inputs.
    trm_report = build_trm_report(repo_root)
    return {
        "schema_version": "r2c-posthoc-post-event-tail-accuracy20-v1",
        "generated_local": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
        "formal_run_mutations": False,
        "formal_gate_mutations": False,
        "metric": {
            "name": "post_event_tail_accuracy20",
            "display_name": "Post-event Tail Accuracy@20",
            "short_name": "PTA@20",
            "formula_percent": (
                "100 * mean(five lowest test accuracies at exact event offsets +1..+20)"
            ),
            "direction": "higher_is_better",
            "range_percent": [0.0, 100.0],
            "window": "exactly 20 completed rounds after the frozen event",
            "tail_size": 5,
            "interpretation": (
                "descriptive worst-five post-event predictive performance; "
                "not causal or counterfactual"
            ),
            "status_completed_tables": (
                "post_hoc_selected_after_user_requested_more_discriminative_display"
            ),
            "status_v12": "frozen_before_v12_formal_performance_unsealing",
        },
        "selection_comparison_on_table2_32_cells": comparison_summary,
        "table1_pta20_percent": table1,
        "table2_pta20_percent": table2,
        "table3_s4_pta20_percent": {
            method: table1["D2"][method] for method in ALL_METHODS
        },
        "table4b_pta20_percent": table4b,
        "table4a_operational_diagnostics": trm_report["table4a_operational_diagnostics"],
        "table4b_operational_diagnostics": trm_report["table4b_operational_diagnostics"],
        "source_lineage": trm_report["source_lineage"],
        "audit": {
            "all_windows_complete": True,
            "derived_records": len(derived_records),
            "table2_cells": len(comparison_rows),
            "table2_pta_unique_at_0.001": int(
                comparison["pta20_percent"].round(3).nunique()
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
                "table2_pta_unique_at_0.001": report["audit"][
                    "table2_pta_unique_at_0.001"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
