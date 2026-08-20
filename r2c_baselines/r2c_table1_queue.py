from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DATASETS,
    DEV_SEED,
    FORMAL_SEED,
    PLOT_ROOT,
    QUEUE_ROOT,
    R2C_COMMON_CONFIG,
    R2C_SEARCH_VARIANTS,
    RUN_ROOT,
)
from .logging_io import read_chunked_table
from .utils import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    canonical_json,
    config_hash,
    sha256_file,
    utc_now,
)


MANIFEST_PATH = QUEUE_ROOT / "r2c_table1_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_table1_queue_state.json"
CLIP_PATH = QUEUE_ROOT / "r2c_table1_delta_clips.json"
SELECTED_CONFIG_PATH = QUEUE_ROOT / "r2c_table1_selected_hyperparameters.json"
TARGET_PATH = QUEUE_ROOT / "frozen_targets.json"
SCHEDULER_EVENTS_PATH = QUEUE_ROOT / "r2c_table1_scheduler_events.parquet"

EVAL_MICROBATCH = {"D1": 8, "D2": 4, "D3": 4, "D4": 2}
STAGE_ORDER = ("norm_pilot", "calibration", "formal")


def _base_method_config(dataset_id: str) -> dict[str, Any]:
    value = dict(R2C_COMMON_CONFIG)
    value["r2c_eval_microbatch"] = EVAL_MICROBATCH[dataset_id]
    return value


def _job(
    job_id: str,
    stage: str,
    mode: str,
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
        "method_id": "R2C-FL",
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


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = ("config.py", "data.py", "training.py", "r2c.py", "run.py")
    return {name: sha256_file(package / name) for name in names}


def build_manifest(persist: bool = True, force: bool = False) -> dict[str, Any]:
    if MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    jobs: list[dict[str, Any]] = []

    # Disjoint development-seed pilots freeze clipping before any calibration
    # or formal test-label access.  The deliberately large clip records raw
    # complete-update norms without binding subsequent choices to test data.
    for dataset_id in DATASETS:
        config = _base_method_config(dataset_id)
        config.update({"lr_mult": 1.0, "r2c_delta_clip": 1.0e9})
        jobs.append(
            _job(
                f"R2C-NORM-{dataset_id}-S4-s{DEV_SEED}",
                "norm_pilot",
                "norm_pilot",
                dataset_id,
                "S4",
                20,
                config,
                "R2C-NORM",
                DEV_SEED,
                True,
            )
        )

    # Three preregistered learning-rate multipliers per dataset.  All other
    # mechanism constants, including the independently frozen norm clip, are
    # shared across variants.
    for dataset_id, spec in DATASETS.items():
        rounds = max(100, int(round(0.20 * spec.round_budget)))
        for variant_index, variant in enumerate(R2C_SEARCH_VARIANTS):
            jobs.append(
                _job(
                    f"R2C-CAL-{dataset_id}-S4-v{variant_index}-s{DEV_SEED}",
                    "calibration",
                    "calibration",
                    dataset_id,
                    "S4",
                    rounds,
                    {
                        "resolve_from_norm_clip": True,
                        "variant": {"lr_mult": float(variant["lr_mult"])},
                    },
                    "R2C-CAL",
                    DEV_SEED,
                    False,
                    variant_index,
                )
            )

    # Table 1: R2C-FL x four datasets x paired stationary/compound traces.
    for scenario_id, block_id in (("S0", "B1a"), ("S4", "B1b")):
        for dataset_id, spec in DATASETS.items():
            jobs.append(
                _job(
                    f"R2C-{block_id}-{dataset_id}-{scenario_id}-s{FORMAL_SEED}",
                    "formal",
                    "formal",
                    dataset_id,
                    scenario_id,
                    spec.round_budget,
                    {"resolve_from_calibration": True},
                    block_id,
                    FORMAL_SEED,
                    True,
                )
            )

    counts = pd.Series([job["stage"] for job in jobs]).value_counts().to_dict()
    expected = {"norm_pilot": 4, "calibration": 12, "formal": 8}
    if counts != expected:
        raise AssertionError({"actual": counts, "expected": expected})
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "formal_seed": FORMAL_SEED,
        "dev_seed": DEV_SEED,
        "method_id": "R2C-FL",
        "table1_formal_jobs": 8,
        "norm_pilot_jobs": 4,
        "calibration_jobs": 12,
        "test_labels_for_selection": False,
        "formal_evaluation_split": "test",
        "development_evaluation_split": "validation",
        "implementation_hashes": _implementation_hashes(),
        "jobs": jobs,
    }
    if persist:
        QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_table1_manifest_{stamp}.json", manifest)
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


def _load_scheduler_events() -> list[dict[str, Any]]:
    if SCHEDULER_EVENTS_PATH.exists():
        return pd.read_parquet(SCHEDULER_EVENTS_PATH).to_dict("records")
    return []


def _scheduler_event(
    events: list[dict[str, Any]], job: dict[str, Any], event_type: str, **extra: Any
) -> None:
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


def _actual_run_id(base: str) -> tuple[str, str | None]:
    base_dir = RUN_ROOT / base
    if not base_dir.exists() or (base_dir / "_SUCCESS.json").exists():
        return base, None
    attempt = 2
    while (RUN_ROOT / f"{base}-a{attempt}").exists():
        attempt += 1
    return f"{base}-a{attempt}", base


def _freeze_norm_clips(manifest: dict[str, Any]) -> None:
    jobs = [job for job in manifest["jobs"] if job["stage"] == "norm_pilot"]
    if not jobs or any(job["status"] != "completed" for job in jobs):
        return
    clips: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for job in jobs:
        run_id = str(job["actual_run_id"])
        frame = read_chunked_table(RUN_ROOT / run_id, "client_round_metrics")
        mask = frame["admitted"].astype(bool) & np.isfinite(frame["delta_norm_raw"])
        values = frame.loc[mask, "delta_norm_raw"].to_numpy(dtype=np.float64)
        if len(values) < 100:
            raise RuntimeError(f"Insufficient norm observations for {job['dataset_id']}: {len(values)}")
        clip = max(float(np.quantile(values, 0.95, method="linear")), 1.0e-12)
        clips[job["dataset_id"]] = clip
        rows.append(
            {
                "dataset_id": job["dataset_id"],
                "dev_seed": DEV_SEED,
                "run_id": run_id,
                "rounds": job["rounds"],
                "observations": len(values),
                "quantile": 0.95,
                "delta_clip": clip,
                "minimum": float(values.min()),
                "median": float(np.median(values)),
                "maximum": float(values.max()),
                "frozen_utc": utc_now(),
                "test_labels_used": False,
            }
        )
    if len(clips) != len(DATASETS):
        raise AssertionError(clips)
    payload = {
        "schema_version": "1.0.0",
        "frozen_utc": utc_now(),
        "derivation": "development-seed 95th percentile of complete admitted-client update norms",
        "test_labels_used": False,
        "clips": clips,
    }
    atomic_json(CLIP_PATH, payload)
    atomic_parquet(PLOT_ROOT / "r2c_norm_pilots.parquet", pd.DataFrame(rows))


def _freeze_calibration(manifest: dict[str, Any]) -> None:
    jobs = [job for job in manifest["jobs"] if job["stage"] == "calibration"]
    if not jobs or any(job["status"] != "completed" for job in jobs):
        return
    rows: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {"R2C-FL": {}}
    for job in jobs:
        run_id = str(job["actual_run_id"])
        result = json.loads((RUN_ROOT / run_id / "result.json").read_text(encoding="utf-8"))
        resolved_job = json.loads(
            (QUEUE_ROOT / "active_jobs" / f"{run_id}.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "trial_id": job["job_id"],
                "method_id": "R2C-FL",
                "dataset_id": job["dataset_id"],
                "dev_seed": DEV_SEED,
                "variant_index": job["variant_index"],
                "config_hash": config_hash(resolved_job["method_config"]),
                "parameters_json": canonical_json(resolved_job["method_config"]),
                "budget_rounds": job["rounds"],
                "validation_objective": "last_window_validation_accuracy_minus_recovery_deficit_auc20",
                "validation_value": result["validation_objective"],
                "selected": False,
                "end_utc": result["completed_utc"],
                "status": "completed",
                "test_labels_used": False,
            }
        )
    frame = pd.DataFrame(rows)
    if frame["validation_value"].isna().any():
        bad = frame.loc[frame["validation_value"].isna(), "trial_id"].tolist()
        raise RuntimeError(f"Null calibration objectives: {bad}")
    for dataset_id, group in frame.groupby("dataset_id"):
        winner = group.sort_values(
            ["validation_value", "trial_id"], ascending=[False, True]
        ).index[0]
        frame.loc[winner, "selected"] = True
        selected["R2C-FL"][dataset_id] = json.loads(frame.loc[winner, "parameters_json"])
    if len(selected["R2C-FL"]) != len(DATASETS):
        raise AssertionError(selected)
    atomic_parquet(PLOT_ROOT / "r2c_hyperparameter_trials.parquet", frame)
    atomic_json(
        SELECTED_CONFIG_PATH,
        {
            "schema_version": "1.0.0",
            "frozen_utc": utc_now(),
            "selection_split": "validation",
            "test_labels_used": False,
            "selected": selected,
        },
    )
    matrix = pd.DataFrame(
        [
            {
                "dataset_id": dataset_id,
                "method_id": "R2C-FL",
                "config_count": 3,
                "budget_fraction": 0.2,
                "dev_seed": DEV_SEED,
                "jobs": 3,
                "status": "DONE",
            }
            for dataset_id in DATASETS
        ]
    )
    atomic_csv(QUEUE_ROOT / "r2c_table1_calibration_matrix.csv", matrix)


def _resolve_job(job: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(job)
    if job["stage"] == "norm_pilot":
        resolved["target_accuracy"] = None
    elif job["stage"] == "calibration":
        if not CLIP_PATH.exists():
            raise RuntimeError("Calibration requires frozen development norm clips")
        clip_payload = json.loads(CLIP_PATH.read_text(encoding="utf-8"))
        config = _base_method_config(job["dataset_id"])
        config.update(dict(job["method_config"]["variant"]))
        config["r2c_eval_microbatch"] = EVAL_MICROBATCH[job["dataset_id"]]
        config["r2c_delta_clip"] = float(clip_payload["clips"][job["dataset_id"]])
        resolved["method_config"] = config
        resolved["target_accuracy"] = None
    else:
        if not SELECTED_CONFIG_PATH.exists() or not TARGET_PATH.exists():
            raise RuntimeError("Formal jobs require frozen validation winners and targets")
        selected_payload = json.loads(SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
        targets = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
        resolved["method_config"] = selected_payload["selected"]["R2C-FL"][job["dataset_id"]]
        resolved["target_accuracy"] = float(targets[job["dataset_id"]])
    actual, retry_of = _actual_run_id(job["base_run_id"])
    resolved["run_id"] = actual
    resolved["retry_of_run_id"] = retry_of
    resolved["queue_utc"] = utc_now()
    return resolved


def _sync_state(state: dict[str, Any], manifest: dict[str, Any]) -> None:
    statuses = [job["status"] for job in manifest["jobs"]]
    state["completed"] = statuses.count("completed")
    state["failed"] = statuses.count("failed")
    state["total"] = len(statuses)
    state["updated_utc"] = utc_now()


def worker(stop_after_stage: str | None = None) -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        build_manifest()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    events = _load_scheduler_events()
    state["status"] = "running"
    _sync_state(state, manifest)
    atomic_json(STATE_PATH, state)
    for stage in STAGE_ORDER:
        stage_jobs = [job for job in manifest["jobs"] if job["stage"] == stage]
        for queue_position, job in enumerate(stage_jobs):
            if job["status"] == "completed":
                continue
            resolved = _resolve_job(job)
            job["attempts"] = int(job.get("attempts", 0)) + 1
            job["actual_run_id"] = resolved["run_id"]
            job["status"] = "running"
            job["failure_reason"] = None
            state.update({"current_job_id": job["job_id"], "status": "running"})
            _sync_state(state, manifest)
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
            success = (RUN_ROOT / resolved["run_id"] / "_SUCCESS.json").exists()
            if process.returncode == 0 and success:
                job["status"] = "completed"
                _scheduler_event(events, job, "completed", exit_code=0)
            else:
                job["status"] = "failed"
                job["failure_reason"] = f"exit_code={process.returncode};log={log_path}"
                state["status"] = "failed"
                _scheduler_event(
                    events,
                    job,
                    "failed",
                    exit_code=process.returncode,
                    reason=job["failure_reason"],
                )
                _sync_state(state, manifest)
                atomic_json(MANIFEST_PATH, manifest)
                atomic_json(STATE_PATH, state)
                return state
            _sync_state(state, manifest)
            atomic_json(MANIFEST_PATH, manifest)
            atomic_json(STATE_PATH, state)
        if stage == "norm_pilot":
            _freeze_norm_clips(manifest)
        elif stage == "calibration":
            _freeze_calibration(manifest)
        if stop_after_stage == stage:
            state.update({"status": "stopped_at_stage", "current_job_id": None})
            _sync_state(state, manifest)
            atomic_json(STATE_PATH, state)
            return state
    state.update({"status": "completed", "current_job_id": None})
    _sync_state(state, manifest)
    atomic_json(STATE_PATH, state)
    return state


def status() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"status": "not_built"}
    value = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        value["counts"] = pd.Series(
            [job["status"] for job in manifest["jobs"]]
        ).value_counts().to_dict()
        value["stage_counts"] = {
            stage: pd.Series(
                [job["status"] for job in manifest["jobs"] if job["stage"] == stage]
            ).value_counts().to_dict()
            for stage in STAGE_ORDER
        }
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--force", action="store_true")
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--stop-after-stage", choices=STAGE_ORDER)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "build":
        result = build_manifest(force=args.force)
    elif args.command == "worker":
        result = worker(args.stop_after_stage)
    else:
        result = status()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
