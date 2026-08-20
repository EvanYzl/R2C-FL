from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import (
    BASELINES,
    DATASETS,
    DEV_SEED,
    EXPERIMENT_ROOT,
    FORMAL_SEED,
    PLOT_ROOT,
    SCENARIOS,
)
from .data import prepare_partition, raw_data_checksum, write_partition_stats
from .run import UPSTREAM_COMMITS
from .traces import TRACE_GENERATOR_VERSION, prepare_trace, write_formal_trace_tables
from .utils import atomic_csv, atomic_json, environment_snapshot, git_commit, utc_now


def _update_registries(dataset_checksums: dict[str, str]) -> None:
    dataset_path = PLOT_ROOT / "dataset_registry.csv"
    datasets = pd.read_csv(dataset_path)
    for dataset_id, checksum in dataset_checksums.items():
        mask = datasets["dataset_id"] == dataset_id
        datasets.loc[mask, "data_checksum"] = checksum
    atomic_csv(dataset_path, datasets)

    method_path = PLOT_ROOT / "method_registry.csv"
    methods = pd.read_csv(method_path)
    adapter_commit = git_commit(EXPERIMENT_ROOT / "r2c_baselines")
    for method_id, upstream in UPSTREAM_COMMITS.items():
        mask = methods["method_id"] == method_id
        methods.loc[mask, "upstream_commit"] = upstream
        methods.loc[mask, "local_adapter_commit"] = adapter_commit
    atomic_csv(method_path, methods)

    scenario_path = PLOT_ROOT / "scenario_registry.csv"
    scenarios = pd.read_csv(scenario_path)
    scenarios.loc[:, "generator_commit"] = TRACE_GENERATOR_VERSION
    atomic_csv(scenario_path, scenarios)


def prepare_all(skip_plot_traces: bool = False, force: bool = False) -> dict[str, object]:
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    formal_partitions: dict[str, dict[str, object]] = {}
    dev_partitions: dict[str, dict[str, object]] = {}
    checksums: dict[str, str] = {}
    traces: list[dict[str, object]] = []
    for dataset_id, spec in DATASETS.items():
        checksums[dataset_id] = raw_data_checksum(dataset_id)
        formal_partitions[dataset_id] = prepare_partition(dataset_id, FORMAL_SEED, force=force)
        dev_partitions[dataset_id] = prepare_partition(dataset_id, DEV_SEED, force=force)
        for scenario_id in SCENARIOS:
            traces.append(prepare_trace(dataset_id, scenario_id, FORMAL_SEED, force=force))
            calibration_rounds = max(100, int(round(0.20 * spec.round_budget)))
            traces.append(
                prepare_trace(
                    dataset_id,
                    scenario_id,
                    DEV_SEED,
                    rounds=calibration_rounds,
                    force=force,
                )
            )
        # Target pilots use the full development S0 horizon.
        traces.append(prepare_trace(dataset_id, "S0", DEV_SEED, rounds=spec.round_budget, force=force))
    _update_registries(checksums)
    write_partition_stats(DATASETS.keys(), FORMAL_SEED, PLOT_ROOT / "data_partition_stats.parquet")
    if not skip_plot_traces:
        write_formal_trace_tables(DATASETS.keys(), SCENARIOS, FORMAL_SEED, PLOT_ROOT)
    report = {
        "status": "completed",
        "generated_utc": utc_now(),
        "formal_seed": FORMAL_SEED,
        "dev_seed": DEV_SEED,
        "dataset_checksums": checksums,
        "formal_partitions": formal_partitions,
        "dev_partitions": dev_partitions,
        "trace_count": len(traces),
        "plot_trace_materialized": not skip_plot_traces,
        "environment": environment_snapshot(),
    }
    atomic_json(EXPERIMENT_ROOT / "frozen_assets" / "prepare_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-plot-traces", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_all(args.skip_plot_traces, args.force), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

