"""Audit and aggregate the locked eight-run R2C-FL v13 M4 matrix.

This finalizer is deliberately unusable until the D3 M3 confirmation gate has
passed and the exact D1--D4 x S0/S4 M4 matrix is terminal.  It replays the
independent run auditor, recomputes the v13 DARE contract and terminal metrics,
and verifies those values against the sealed M4 report before writing anything.
Run directories and formal metrics are never mutated.  PTA@20 remains a
post-hoc descriptive display alongside the original AUC@20 and TRM evidence.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import SCHEMA_VERSION
from . import r2c_v13_all_datasets_queue as m4
from .config import DATASETS, PLOT_ROOT, RUN_ROOT
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
from .r2c_v13 import PROTOCOL_VERSION
from .utils import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    sha256_file,
    sha256_text,
    utc_now,
)


FORMAL_SEED = 20260811
FORMAL_INTERPRETATION = (
    "outcome_informed_engineering_reevaluation_not_untouched_confirmation"
)
TERMINAL_METRIC_COLUMNS = (
    "dataset_id",
    "scenario_id",
    "run_id",
    "round_budget",
    "last50_test_accuracy",
    "pta20_percent",
    "algorithm_tta_round",
    "algorithm_tta_s",
    "recovery_deficit_auc20",
    "trm20_pp",
    "event_round",
    "source_kind",
    "formal_interpretation",
    "test_labels_used_for_selection",
)


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


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    tests = package.parent / "tests"
    paths = {
        "r2c_v13_all_datasets_finalize.py": Path(__file__).resolve(),
        "r2c_v13_all_datasets_queue.py": package / "r2c_v13_all_datasets_queue.py",
        "r2c_v13.py": package / "r2c_v13.py",
        "r2c_post_event_tail_accuracy20.py": package
        / "r2c_post_event_tail_accuracy20.py",
        "tests/audit_r2c_run.py": tests / "audit_r2c_run.py",
        "tests/test_r2c_v13_all_datasets_finalize.py": tests
        / "test_r2c_v13_all_datasets_finalize.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _validate_terminal_payloads(
    m3_context: dict[str, Any],
    state: dict[str, Any],
    result: dict[str, Any],
    manifest: dict[str, Any],
    stored_runs: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Validate terminal M3/M4 authorization and return exact M4 jobs."""

    if (
        state.get("status") != "m4_completed_audited_tables_update_required"
        or int(state.get("completed", -1)) != 8
        or int(state.get("failed", -1)) != 0
        or int(state.get("total", -1)) != 8
        or not bool(state.get("all_runs_completed"))
        or bool(state.get("performance_sealed_until_terminal"))
        or not bool(state.get("formal_test_access"))
        or not bool(state.get("other_dataset_access"))
        or result.get("status") != "m4_all_datasets_completed_audited"
        or result.get("evaluation_split") != "test"
        or result.get("source_kind") != "REPRODUCED"
        or not bool(result.get("formal_test_access"))
        or not bool(result.get("other_dataset_access"))
        or bool(result.get("test_labels_used_for_selection"))
        or int(result.get("completed_runs", -1)) != 8
        or result.get("formal_interpretation") != FORMAL_INTERPRETATION
    ):
        raise RuntimeError("v13 M4 all-dataset matrix is not terminal and authorized")

    schedule_id = str(m3_context.get("schedule_id", ""))
    m3_manifest = dict(m3_context.get("manifest", {}))
    expected_selected = {
        "schedule_id": schedule_id,
        "source_m3_frozen_spec_hash": str(m3_manifest.get("frozen_spec_hash", "")),
    }
    if (
        not schedule_id
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("evaluation_split") != "test"
        or int(manifest.get("seed", -1)) != FORMAL_SEED
        or not bool(manifest.get("formal_test_access"))
        or not bool(manifest.get("other_dataset_access"))
        or bool(manifest.get("test_labels_used_for_selection"))
        or not bool(manifest.get("candidate_locked_before_all_datasets"))
        or manifest.get("formal_interpretation") != FORMAL_INTERPRETATION
        or manifest.get("selected_candidate") != expected_selected
        or result.get("selected_candidate") != expected_selected
        or result.get("frozen_spec_hash") != manifest.get("frozen_spec_hash")
        or state.get("frozen_spec_hash") != manifest.get("frozen_spec_hash")
    ):
        raise RuntimeError("v13 M4 candidate, protocol, or immutable lineage mismatch")

    jobs = list(manifest.get("jobs", []))
    expected_order = [
        (dataset_id, scenario)
        for dataset_id in m4.DATASET_ORDER
        for scenario in m4.SCENARIO_ORDER
    ]
    observed_order = [
        (str(job.get("dataset_id")), str(job.get("scenario_id"))) for job in jobs
    ]
    if len(jobs) != 8 or observed_order != expected_order:
        raise RuntimeError("v13 M4 dataset/scenario matrix or order is incomplete")

    for job in jobs:
        dataset_id = str(job.get("dataset_id"))
        if (
            job.get("status") != "completed"
            or not job.get("actual_run_id")
            or job.get("mode") != "formal"
            or job.get("method_id") != R2C_METHOD
            or job.get("method_version") != "v13"
            or job.get("evaluation_split") != "test"
            or job.get("selected_schedule_id") != schedule_id
            or job.get("source_m3_frozen_spec_hash")
            != expected_selected["source_m3_frozen_spec_hash"]
            or not bool(job.get("formal_test_access"))
            or not bool(job.get("other_dataset_access"))
            or bool(job.get("test_labels_used_for_selection"))
            or job.get("formal_interpretation") != FORMAL_INTERPRETATION
            or int(job.get("rounds", -1)) != int(DATASETS[dataset_id].round_budget)
            or any(
                int(job.get(key, -1)) != FORMAL_SEED
                for key in ("seed", "partition_seed", "trace_seed")
            )
        ):
            raise RuntimeError(f"v13 M4 job contract mismatch: {job.get('job_id')}")

    run_ids = [str(job["actual_run_id"]) for job in jobs]
    if len(set(run_ids)) != 8 or list(result.get("run_ids", [])) != run_ids:
        raise RuntimeError("v13 M4 terminal run IDs are duplicated or out of order")

    if len(stored_runs) != 8 or any(
        column not in stored_runs.columns for column in TERMINAL_METRIC_COLUMNS
    ):
        raise RuntimeError("v13 M4 stored terminal metric frame is incomplete")
    stored_order = list(
        stored_runs[["dataset_id", "scenario_id"]].astype(str).itertuples(
            index=False, name=None
        )
    )
    if (
        stored_order != expected_order
        or stored_runs["run_id"].astype(str).tolist() != run_ids
        or stored_runs["run_id"].astype(str).duplicated().any()
        or not stored_runs["source_kind"].astype(str).eq("REPRODUCED").all()
        or stored_runs["test_labels_used_for_selection"].astype(bool).any()
        or not stored_runs["formal_interpretation"]
        .astype(str)
        .eq(FORMAL_INTERPRETATION)
        .all()
    ):
        raise RuntimeError("v13 M4 stored terminal metrics violate lineage")
    return jobs


def _load_completed_jobs() -> tuple[list[dict[str, Any]], dict[str, str], pd.DataFrame]:
    required = (
        m4.MANIFEST_PATH,
        m4.STATE_PATH,
        m4.RESULT_PATH,
        m4.RUNS_PATH,
        m4.RUNS_CSV_PATH,
    )
    if any(not path.exists() for path in required):
        raise RuntimeError("v13 M4 terminal artifacts do not exist")

    m3_context = m4._m3_context()
    manifest = _read_json(m4.MANIFEST_PATH)
    state = _read_json(m4.STATE_PATH)
    result = _read_json(m4.RESULT_PATH)
    m4._assert_frozen_manifest(manifest)
    immutable_path = Path(str(state.get("immutable_manifest_path", ""))).resolve()
    if (
        not immutable_path.exists()
        or sha256_file(immutable_path).lower()
        != str(state.get("immutable_manifest_sha256", "")).lower()
        or state.get("frozen_spec_hash") != manifest.get("frozen_spec_hash")
    ):
        raise RuntimeError("v13 M4 immutable manifest hash drift")

    stored_runs = pd.read_parquet(m4.RUNS_PATH)
    jobs = _validate_terminal_payloads(
        m3_context, state, result, manifest, stored_runs
    )
    recomputed = pd.DataFrame([m4._run_metrics(job) for job in jobs])
    try:
        pd.testing.assert_frame_equal(
            stored_runs.loc[:, TERMINAL_METRIC_COLUMNS].reset_index(drop=True),
            recomputed.loc[:, TERMINAL_METRIC_COLUMNS].reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as exc:
        raise RuntimeError(
            "v13 M4 recomputed terminal metrics differ from the sealed report"
        ) from exc

    terminal_hashes = {
        "m3_immutable_manifest": sha256_file(m3_context["immutable_path"]),
        "m3_result": sha256_file(m4.M3_RESULT_PATH),
        "m4_immutable_manifest": sha256_file(immutable_path),
        "m4_active_manifest": sha256_file(m4.MANIFEST_PATH),
        "m4_result": sha256_file(m4.RESULT_PATH),
        "m4_runs": sha256_file(m4.RUNS_PATH),
    }
    return jobs, terminal_hashes, recomputed


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
                selected = baseline_rounds.loc[
                    baseline_rounds["run_id"].eq(run_id)
                ].copy()
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
        raise RuntimeError("v13 PTA@20 Table-1 cell matrix is incomplete")
    return result


def finalize() -> dict[str, Any]:
    jobs, terminal_hashes, terminal_metrics = _load_completed_jobs()
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
    expected_budgets = {
        dataset_id: int(DATASETS[dataset_id].round_budget) for dataset_id in DATASETS
    }
    if expected != observed or len(combined) != 64 or combined["run_id"].duplicated().any():
        raise RuntimeError("combined v13 Table-1 run matrix is incomplete or duplicated")
    if not (combined["seed"].astype(int) == FORMAL_SEED).all():
        raise RuntimeError("combined v13 Table-1 seed mismatch")
    if not (combined["source_kind"].astype(str) == "REPRODUCED").all():
        raise RuntimeError("combined v13 Table-1 provenance mismatch")
    if not (combined["status"].astype(str) == "completed").all():
        raise RuntimeError("combined v13 Table-1 contains incomplete runs")
    if any(
        int(row.round_budget) != expected_budgets[str(row.dataset_id)]
        for row in combined[["dataset_id", "round_budget"]].itertuples(index=False)
    ):
        raise RuntimeError("combined v13 Table-1 budget mismatch")

    original_values = _table_values(combined)
    pta_values = _pta20_values(combined)
    generated = utc_now()
    stamp = _stamp(generated)
    run_ids = combined["run_id"].astype(str).sort_values().tolist()
    m4_state = _read_json(m4.STATE_PATH)
    m4_immutable = Path(str(m4_state["immutable_manifest_path"])).resolve()
    m3_state = _read_json(m4.M3_STATE_PATH)
    m3_immutable = Path(str(m3_state["immutable_manifest_path"])).resolve()
    input_paths = [
        PLOT_ROOT / "run_summary.parquet",
        PLOT_ROOT / "run_manifest.parquet",
        PLOT_ROOT / "round_metrics.parquet",
        PLOT_ROOT / "baseline_qc_report.json",
        m3_immutable,
        m4.M3_RESULT_PATH,
        m4_immutable,
        m4.MANIFEST_PATH,
        m4.RESULT_PATH,
        m4.RUNS_PATH,
    ]
    for job in jobs:
        run_dir = RUN_ROOT / str(job["actual_run_id"])
        input_paths.extend(
            [
                run_dir / "job.json",
                run_dir / "run_manifest.parquet",
                run_dir / "result.json",
                run_dir / "_SUCCESS.json",
            ]
        )
    input_hash = sha256_text(
        "\n".join(f"{path}:{sha256_file(path)}" for path in input_paths)
    )

    outputs: list[str] = []
    outputs += _write_parquet("r2c_v13_table1_run_summary", r2c_summary, stamp)
    outputs += _write_parquet("table1_v13_combined_run_summary", combined, stamp)
    outputs += _write_parquet("table1_v13_combined_values", original_values, stamp)
    outputs += _write_csv("table1_v13_combined_values", original_values, stamp)
    outputs += _write_parquet("table1_v13_pta20_values", pta_values, stamp)
    outputs += _write_csv("table1_v13_pta20_values", pta_values, stamp)
    outputs += _write_parquet("r2c_v13_m4_terminal_metrics", terminal_metrics, stamp)
    outputs += _write_csv("r2c_v13_m4_terminal_metrics", terminal_metrics, stamp)
    outputs += _write_json(
        "r2c_v13_table1_audit_report",
        {"generated_utc": generated, "runs": audit_reports},
        stamp,
    )
    outputs += _write_json(
        "r2c_v13_table1_auc20_windows",
        {"generated_utc": generated, "runs": window_reports},
        stamp,
    )
    outputs += _write_json(
        "r2c_v13_table1_pta20_report",
        {
            "schema_version": "r2c-v13-table1-pta20-v1",
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

    implementation_hashes = _implementation_hashes()
    qc = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated,
        "complete": True,
        "formal_seed": FORMAL_SEED,
        "protocol_version": PROTOCOL_VERSION,
        "formal_interpretation": FORMAL_INTERPRETATION,
        "formal_r2c_runs": int(len(r2c_summary)),
        "baseline_table1_runs": int(len(baseline_summary)),
        "combined_table1_runs": int(len(combined)),
        "unique_run_ids": not combined["run_id"].duplicated().any(),
        "matrix_complete": expected == observed,
        "matched_seed_ok": bool((combined["seed"].astype(int) == FORMAL_SEED).all()),
        "source_kind_ok": bool(
            (combined["source_kind"].astype(str) == "REPRODUCED").all()
        ),
        "status_ok": bool((combined["status"].astype(str) == "completed").all()),
        "budget_ok": True,
        "r2c_audits_passed": int(
            sum(report["status"] == "passed" for report in audit_reports)
        ),
        "r2c_strict_auc20_windows": int(
            sum(report["complete"] for report in window_reports)
        ),
        "v13_dare_contracts_recomputed": int(len(terminal_metrics)),
        "m4_terminal_metrics_exactly_reproduced": True,
        "pta20_cells": int(len(pta_values)),
        "input_files_hash": input_hash,
        "input_run_ids_hash": sha256_text("\n".join(run_ids)),
        "terminal_hashes": terminal_hashes,
        "implementation_hashes": implementation_hashes,
        "formal_gate_unchanged": True,
        "formal_run_mutations": False,
        "cross_model_review": "waived_by_user",
    }
    outputs += _write_json("r2c_v13_table1_qc_report", qc, stamp)

    aggregation_paths = [str(path) for path in _paths(
        "r2c_v13_table1_aggregation_result", "json", stamp
    )]
    result = {
        "status": "v13_all_datasets_completed_audited_tables_update_required",
        "qc_complete": True,
        "formal_seed": FORMAL_SEED,
        "formal_r2c_runs": 8,
        "combined_table1_runs": 64,
        "pta20_cells": int(len(pta_values)),
        "version_stamp": stamp,
        "outputs": outputs + aggregation_paths,
    }
    _write_json("r2c_v13_table1_aggregation_result", result, stamp)

    m4_state.update(
        {
            "status": "v13_all_datasets_completed_audited_tables_update_required",
            "all8_audit_complete": True,
            "all8_aggregation_complete": True,
            "tables_update_required": True,
            "updated_utc": utc_now(),
        }
    )
    atomic_json(m4.STATE_PATH, m4_state)
    return result


def main() -> None:
    print(json.dumps(finalize(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
