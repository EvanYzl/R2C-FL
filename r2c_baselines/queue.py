from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    BASELINES,
    DATASETS,
    DEFAULT_METHOD_CONFIG,
    DEV_SEED,
    FORMAL_SEED,
    PLOT_ROOT,
    QUEUE_ROOT,
    RUN_ROOT,
    SEARCH_VARIANTS,
)
from .logging_io import read_chunked_table
from .utils import atomic_csv, atomic_json, atomic_parquet, canonical_json, config_hash, utc_now


MANIFEST_PATH = QUEUE_ROOT / "baseline_manifest.json"
STATE_PATH = QUEUE_ROOT / "queue_state.json"
SELECTED_CONFIG_PATH = QUEUE_ROOT / "selected_hyperparameters.json"
TARGET_PATH = QUEUE_ROOT / "frozen_targets.json"
SCHEDULER_EVENTS_PATH = QUEUE_ROOT / "scheduler_events.parquet"


def _job(
    job_id: str,
    stage: str,
    mode: str,
    method_id: str,
    dataset_id: str,
    scenario_id: str,
    rounds: int,
    method_config: dict[str, Any],
    block_id: str,
    seed: int,
    full_logging: bool,
    variant_index: int | None = None,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "base_run_id": job_id,
        "stage": stage,
        "mode": mode,
        "method_id": method_id,
        "dataset_id": dataset_id,
        "scenario_id": scenario_id,
        "rounds": int(rounds),
        "method_config": method_config,
        "block_id": block_id,
        "seed": int(seed),
        "partition_seed": int(seed),
        "trace_seed": int(seed),
        "evaluation_split": "test" if mode == "formal" else "validation",
        "full_logging": bool(full_logging),
        "client_microbatch": 1,
        "variant_index": variant_index,
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def build_manifest(persist: bool = True) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    # One two-round adapter smoke per baseline.
    for method_id in BASELINES:
        jobs.append(
            _job(
                f"SMOKE2-D1-S4-{method_id}-serial1-s{DEV_SEED}",
                "smoke",
                "sanity",
                method_id,
                "D1",
                "S4",
                2,
                dict(DEFAULT_METHOD_CONFIG[method_id]),
                "G3",
                DEV_SEED,
                True,
            )
        )
    # Twenty-round data/model gates.
    for dataset_id in DATASETS:
        jobs.append(
            _job(
                f"GATE2-{dataset_id}-S0-FedAvg-serial1-s{DEV_SEED}",
                "gate",
                "sanity",
                "FedAvg",
                dataset_id,
                "S0",
                20,
                dict(DEFAULT_METHOD_CONFIG["FedAvg"]),
                "G4",
                DEV_SEED,
                True,
            )
        )
    # Full stationary validation pilots used once to freeze target accuracy.
    for dataset_id, spec in DATASETS.items():
        target_prefix = "TARGET2" if dataset_id == "D1" else "TARGET"
        jobs.append(
            _job(
                f"{target_prefix}-{dataset_id}-S0-FedAvg-s{DEV_SEED}",
                "target",
                "target",
                "FedAvg",
                dataset_id,
                "S0",
                spec.round_budget,
                dict(DEFAULT_METHOD_CONFIG["FedAvg"]),
                "G9",
                DEV_SEED,
                False,
            )
        )
    # 7 baselines x 4 datasets x 3 preregistered configurations = 84 jobs.
    for method_id in BASELINES:
        for dataset_id, spec in DATASETS.items():
            calibration_rounds = max(100, int(round(0.20 * spec.round_budget)))
            for variant_index, variant in enumerate(SEARCH_VARIANTS[method_id]):
                jobs.append(
                    _job(
                        f"CAL-{dataset_id}-S4-{method_id}-v{variant_index}-s{DEV_SEED}",
                        "calibration",
                        "calibration",
                        method_id,
                        dataset_id,
                        "S4",
                        calibration_rounds,
                        dict(variant),
                        "G10",
                        DEV_SEED,
                        False,
                        variant_index,
                    )
                )
    # B1: four datasets, paired S0/S4, seven baselines = 56 jobs.
    for scenario_id, block_id in (("S0", "B1a"), ("S4", "B1b")):
        for dataset_id, spec in DATASETS.items():
            for method_id in BASELINES:
                jobs.append(
                    _job(
                        f"{block_id}-{dataset_id}-{scenario_id}-{method_id}-s{FORMAL_SEED}",
                        "formal",
                        "formal",
                        method_id,
                        dataset_id,
                        scenario_id,
                        spec.round_budget,
                        {"resolve_from_calibration": True},
                        block_id,
                        FORMAL_SEED,
                        True,
                    )
                )
    # B2: D2 factor isolation S1/S2/S3, seven baselines = 21 jobs.
    for scenario_id, block_id in (("S1", "B2a"), ("S2", "B2b"), ("S3", "B2c")):
        for method_id in BASELINES:
            jobs.append(
                _job(
                    f"{block_id}-D2-{scenario_id}-{method_id}-s{FORMAL_SEED}",
                    "formal",
                    "formal",
                    method_id,
                    "D2",
                    scenario_id,
                    DATASETS["D2"].round_budget,
                    {"resolve_from_calibration": True},
                    block_id,
                    FORMAL_SEED,
                    True,
                )
            )
    counts = pd.Series([job["stage"] for job in jobs]).value_counts().to_dict()
    if counts.get("formal") != 77 or counts.get("calibration") != 84:
        raise AssertionError(counts)
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "formal_seed": FORMAL_SEED,
        "dev_seed": DEV_SEED,
        "formal_baseline_jobs": 77,
        "calibration_jobs": 84,
        "jobs": jobs,
    }
    if persist:
        QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
        timestamped = QUEUE_ROOT / f"baseline_manifest_{utc_now().replace(':', '').replace('-', '')}.json"
        atomic_json(timestamped, manifest)
        atomic_json(MANIFEST_PATH, manifest)
        atomic_json(
            STATE_PATH,
            {
                "status": "ready",
                "created_utc": utc_now(),
                "updated_utc": utc_now(),
                "current_job_id": None,
                "completed": 0,
                "failed": 0,
                "total": len(jobs),
            },
        )
    return manifest


def _scheduler_event(events: list[dict[str, Any]], job: dict[str, Any], event_type: str, **extra: Any) -> None:
    value = {
        "schema_version": "2.1.0",
        "job_id": job["job_id"],
        "run_id": job.get("actual_run_id"),
        "host_id": None,
        "gpu_index": 0,
        "gpu_uuid": None,
        "event_utc": utc_now(),
        "event_type": event_type,
        "queue_position": extra.pop("queue_position", None),
        "attempt": job.get("attempts", 0),
        "exit_code": extra.pop("exit_code", None),
        "reason": extra.pop("reason", None),
    }
    value.update(extra)
    events.append(value)
    atomic_parquet(SCHEDULER_EVENTS_PATH, pd.DataFrame(events))


def _load_scheduler_events() -> list[dict[str, Any]]:
    if SCHEDULER_EVENTS_PATH.exists():
        return pd.read_parquet(SCHEDULER_EVENTS_PATH).to_dict("records")
    return []


def _actual_run_id(base: str) -> tuple[str, str | None]:
    base_dir = RUN_ROOT / base
    if not base_dir.exists():
        return base, None
    if (base_dir / "_SUCCESS.json").exists():
        return base, None
    attempt = 2
    while (RUN_ROOT / f"{base}-a{attempt}").exists():
        attempt += 1
    return f"{base}-a{attempt}", base


def _freeze_targets(manifest: dict[str, Any]) -> None:
    rows = []
    targets: dict[str, float] = {}
    for job in manifest["jobs"]:
        if job["stage"] != "target" or job["status"] != "completed":
            continue
        result = json.loads((RUN_ROOT / job["actual_run_id"] / "result.json").read_text(encoding="utf-8"))
        target = 0.90 * float(result["last50_accuracy"])
        targets[job["dataset_id"]] = target
        rows.append(
            {
                "dataset_id": job["dataset_id"],
                "dev_seed": DEV_SEED,
                "run_id": job["actual_run_id"],
                "last50_validation_accuracy": result["last50_accuracy"],
                "derived_target_accuracy": target,
                "frozen_utc": utc_now(),
                "config_hash": config_hash(job["method_config"]),
            }
        )
    if len(targets) != len(DATASETS):
        return
    atomic_json(TARGET_PATH, targets)
    atomic_parquet(PLOT_ROOT / "target_pilots.parquet", pd.DataFrame(rows))
    registry_path = PLOT_ROOT / "dataset_registry.csv"
    registry = pd.read_csv(registry_path)
    for dataset_id, target in targets.items():
        mask = registry["dataset_id"] == dataset_id
        registry.loc[mask, "target_accuracy"] = target
        registry.loc[mask, "target_frozen_at"] = utc_now()
    atomic_csv(registry_path, registry)


def _freeze_calibration(manifest: dict[str, Any]) -> None:
    rows = []
    selected: dict[str, dict[str, Any]] = {}
    calibration_jobs = [job for job in manifest["jobs"] if job["stage"] == "calibration"]
    if not calibration_jobs or any(job["status"] != "completed" for job in calibration_jobs):
        return
    for job in calibration_jobs:
        result = json.loads((RUN_ROOT / job["actual_run_id"] / "result.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "trial_id": job["job_id"],
                "method_id": job["method_id"],
                "dataset_id": job["dataset_id"],
                "dev_seed": DEV_SEED,
                "config_hash": config_hash(job["method_config"]),
                "parameters_json": canonical_json(job["method_config"]),
                "budget_rounds": job["rounds"],
                "validation_objective": "last_window_validation_accuracy_minus_recovery_deficit_auc20",
                "validation_value": result["validation_objective"],
                "selected": False,
                "start_utc": None,
                "end_utc": result["completed_utc"],
                "status": "completed",
                "failure_reason": None,
            }
        )
    frame = pd.DataFrame(rows)
    for (method_id, dataset_id), group in frame.groupby(["method_id", "dataset_id"]):
        winner_index = group.sort_values(["validation_value", "trial_id"], ascending=[False, True]).index[0]
        frame.loc[winner_index, "selected"] = True
        parameters = json.loads(frame.loc[winner_index, "parameters_json"])
        selected.setdefault(method_id, {})[dataset_id] = parameters
    atomic_parquet(PLOT_ROOT / "hyperparameter_trials.parquet", frame)
    atomic_json(SELECTED_CONFIG_PATH, selected)
    matrix_rows = []
    for dataset_id in DATASETS:
        matrix_rows.append(
            {
                "dataset_id": dataset_id,
                "method_set": "BASELINE7",
                "config_count": 3,
                "budget_fraction": 0.2,
                "dev_seed": DEV_SEED,
                "jobs": len(BASELINES) * 3,
                "status": "DONE",
            }
        )
    atomic_csv(QUEUE_ROOT / "calibration_matrix.csv", pd.DataFrame(matrix_rows))


def _resolve_job(job: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(job)
    if job["stage"] == "formal":
        if not SELECTED_CONFIG_PATH.exists() or not TARGET_PATH.exists():
            raise RuntimeError("Formal jobs require frozen target pilots and calibration winners")
        selected = json.loads(SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
        targets = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
        resolved["method_config"] = selected[job["method_id"]][job["dataset_id"]]
        resolved["target_accuracy"] = float(targets[job["dataset_id"]])
    else:
        resolved["target_accuracy"] = None
    actual, retry_of = _actual_run_id(job["base_run_id"])
    resolved["run_id"] = actual
    resolved["retry_of_run_id"] = retry_of
    resolved["queue_utc"] = utc_now()
    return resolved


def worker(stop_after_stage: str | None = None) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        build_manifest()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    events = _load_scheduler_events()
    stage_order = ["smoke", "gate", "target", "calibration", "formal"]
    state["status"] = "running"
    atomic_json(STATE_PATH, state)
    for stage in stage_order:
        stage_jobs = [job for job in manifest["jobs"] if job["stage"] == stage]
        for queue_position, job in enumerate(stage_jobs):
            if job["status"] == "completed":
                continue
            resolved = _resolve_job(job)
            job["attempts"] = int(job.get("attempts", 0)) + 1
            job["actual_run_id"] = resolved["run_id"]
            job["status"] = "running"
            state.update({"current_job_id": job["job_id"], "updated_utc": utc_now()})
            atomic_json(MANIFEST_PATH, manifest)
            atomic_json(STATE_PATH, state)
            _scheduler_event(events, job, "submitted", queue_position=queue_position)
            _scheduler_event(events, job, "assigned", queue_position=queue_position)
            job_file = QUEUE_ROOT / "active_jobs" / f"{resolved['run_id']}.json"
            atomic_json(job_file, resolved)
            _scheduler_event(events, job, "started", queue_position=queue_position)
            log_path = QUEUE_ROOT / "worker_logs" / f"{resolved['run_id']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
                process = subprocess.run(
                    [sys.executable, "-m", "r2c_baselines.run", "--job", str(job_file)],
                    cwd=str(Path(__file__).resolve().parents[1]),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if process.returncode == 0 and (RUN_ROOT / resolved["run_id"] / "_SUCCESS.json").exists():
                job["status"] = "completed"
                state["completed"] = int(state.get("completed", 0)) + 1
                _scheduler_event(events, job, "completed", exit_code=0)
            else:
                job["status"] = "failed"
                job["failure_reason"] = f"exit_code={process.returncode};log={log_path}"
                state["failed"] = int(state.get("failed", 0)) + 1
                state["status"] = "failed"
                _scheduler_event(events, job, "failed", exit_code=process.returncode, reason=job["failure_reason"])
                atomic_json(MANIFEST_PATH, manifest)
                atomic_json(STATE_PATH, state)
                return state
            atomic_json(MANIFEST_PATH, manifest)
            state["updated_utc"] = utc_now()
            atomic_json(STATE_PATH, state)
        if stage == "target":
            _freeze_targets(manifest)
        if stage == "calibration":
            _freeze_calibration(manifest)
        if stop_after_stage == stage:
            state.update({"status": "stopped_at_stage", "current_job_id": None, "updated_utc": utc_now()})
            atomic_json(STATE_PATH, state)
            return state
    state.update({"status": "completed", "current_job_id": None, "updated_utc": utc_now()})
    atomic_json(STATE_PATH, state)
    return state


def status() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"status": "not_built"}
    value = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        counts = pd.Series([job["status"] for job in manifest["jobs"]]).value_counts().to_dict()
        stage_counts = {}
        for stage in {job["stage"] for job in manifest["jobs"]}:
            stage_counts[stage] = pd.Series(
                [job["status"] for job in manifest["jobs"] if job["stage"] == stage]
            ).value_counts().to_dict()
        value["counts"] = counts
        value["stage_counts"] = stage_counts
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--stop-after-stage", choices=["smoke", "gate", "target", "calibration", "formal"])
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "build":
        result = build_manifest()
    elif args.command == "worker":
        result = worker(args.stop_after_stage)
    else:
        result = status()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
