from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEV_SEED, PLOT_ROOT, QUEUE_ROOT, RUN_ROOT
from .r2c_v2 import PROTOCOL_VERSION
from .utils import atomic_csv, atomic_json, atomic_parquet, canonical_json, config_hash, sha256_file, utc_now


MANIFEST_PATH = QUEUE_ROOT / "r2c_d2_opt_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d2_opt_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d2_opt_scheduler_events.parquet"
SCREEN_TABLE_PATH = PLOT_ROOT / "r2c_d2_opt_screen_trials.parquet"
SCREEN_CSV_PATH = PLOT_ROOT / "r2c_d2_opt_screen_trials.csv"
SCREEN_SELECTION_PATH = QUEUE_ROOT / "r2c_d2_opt_screen_selection.json"


VARIANTS: tuple[dict[str, Any], ...] = (
    {"lr_mult": 1.5, "r2c_v2_temperature": 0.20, "r2c_v2_floor_fraction": 0.10, "r2c_v2_finish_weight": 0.75, "r2c_v2_anchor_per_fold": 16},
    {"lr_mult": 1.5, "r2c_v2_temperature": 0.10, "r2c_v2_floor_fraction": 0.05, "r2c_v2_finish_weight": 0.75, "r2c_v2_anchor_per_fold": 16},
    {"lr_mult": 1.5, "r2c_v2_temperature": 0.35, "r2c_v2_floor_fraction": 0.20, "r2c_v2_finish_weight": 0.75, "r2c_v2_anchor_per_fold": 16},
    {"lr_mult": 1.0, "r2c_v2_temperature": 0.20, "r2c_v2_floor_fraction": 0.10, "r2c_v2_finish_weight": 0.75, "r2c_v2_anchor_per_fold": 16},
    {"lr_mult": 2.0, "r2c_v2_temperature": 0.20, "r2c_v2_floor_fraction": 0.10, "r2c_v2_finish_weight": 0.75, "r2c_v2_anchor_per_fold": 16},
    {"lr_mult": 1.5, "r2c_v2_temperature": 0.20, "r2c_v2_floor_fraction": 0.10, "r2c_v2_finish_weight": 1.25, "r2c_v2_anchor_per_fold": 32},
)


def _common_config() -> dict[str, Any]:
    return {
        "r2c_protocol_version": PROTOCOL_VERSION,
        "r2c_v2_scout_steps": 1,
        "r2c_eval_microbatch": 4,
        "r2c_v2_value_clip": 4.0,
        "r2c_v2_scale_floor": 1.0e-4,
        "r2c_v2_uncertainty_floor_fraction": 0.35,
        "r2c_v2_finish_scale": 0.08,
        "r2c_v2_radius_multiplier": 1.0,
        "r2c_v2_agreement_threshold": 0.80,
        "r2c_v2_propensity_power": 0.0,
        "r2c_v2_completion_power": 0.0,
        "r2c_v2_weight_cap": 2.0,
        "r2c_delta_clip": 0.6107748032302804,
        "r2c_v2_audit_replay": False,
    }


def _job(index: int, variant: dict[str, Any]) -> dict[str, Any]:
    config = _common_config()
    config.update(variant)
    job_id = f"A-R2C-D2-S4-SCREEN-V{index}-s{DEV_SEED}"
    return {
        "job_id": job_id,
        "base_run_id": job_id,
        "stage": "screen",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": "D2",
        "scenario_id": "S4",
        "rounds": 200,
        "method_config": config,
        "block_id": "A-R2C-D2-SCREEN",
        "seed": DEV_SEED,
        "partition_seed": DEV_SEED,
        "trace_seed": DEV_SEED,
        "evaluation_split": "validation",
        "full_logging": False,
        "client_microbatch": 1,
        "target_accuracy": None,
        "variant_index": index,
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = ("data.py", "training.py", "r2c.py", "r2c_v2.py", "run.py", Path(__file__).name)
    return {name: sha256_file(package / name) for name in names}


def build_manifest(persist: bool = True, force: bool = False) -> dict[str, Any]:
    if persist and MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    jobs = [_job(index, variant) for index, variant in enumerate(VARIANTS)]
    if len(jobs) != 6 or len({job["job_id"] for job in jobs}) != 6:
        raise AssertionError("D2 screen must contain six unique jobs")
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D2_only_validation_screen",
        "protocol_version": PROTOCOL_VERSION,
        "dev_seed": DEV_SEED,
        "formal_test_access": False,
        "selection_objective": "last50_validation_accuracy - recovery_deficit_auc20; algorithm_elapsed_s tie-break",
        "implementation_hashes": _implementation_hashes(),
        "jobs": jobs,
    }
    if persist:
        QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d2_opt_manifest_{stamp}.json", manifest)
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


def _load_events() -> list[dict[str, Any]]:
    if EVENTS_PATH.exists():
        return pd.read_parquet(EVENTS_PATH).to_dict("records")
    return []


def _event(events: list[dict[str, Any]], job: dict[str, Any], event_type: str, **extra: Any) -> None:
    row = {
        "schema_version": "2.1.0",
        "job_id": job["job_id"],
        "run_id": job.get("actual_run_id"),
        "event_utc": utc_now(),
        "event_type": event_type,
        "attempt": int(job.get("attempts", 0)),
        "exit_code": extra.pop("exit_code", None),
        "reason": extra.pop("reason", None),
    }
    row.update(extra)
    events.append(row)
    atomic_parquet(EVENTS_PATH, pd.DataFrame(events))


def _actual_run_id(base: str) -> tuple[str, str | None]:
    base_dir = RUN_ROOT / base
    if not base_dir.exists() or (base_dir / "_SUCCESS.json").exists():
        return base, None
    attempt = 2
    while (RUN_ROOT / f"{base}-a{attempt}").exists():
        attempt += 1
    return f"{base}-a{attempt}", base


def _sync(state: dict[str, Any], manifest: dict[str, Any]) -> None:
    statuses = [job["status"] for job in manifest["jobs"]]
    state["completed"] = statuses.count("completed")
    state["failed"] = statuses.count("failed")
    state["total"] = len(statuses)
    state["updated_utc"] = utc_now()


def _resolved_job(job: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(job)
    actual, retry_of = _actual_run_id(str(job["base_run_id"]))
    resolved["run_id"] = actual
    resolved["retry_of_run_id"] = retry_of
    resolved["queue_utc"] = utc_now()
    return resolved


def freeze_screen(manifest: dict[str, Any]) -> None:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        return
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run_id = str(job["actual_run_id"])
        result = json.loads((RUN_ROOT / run_id / "result.json").read_text(encoding="utf-8"))
        auc = result["recovery"]["recovery_deficit_auc20"]
        if auc is None:
            raise RuntimeError(f"Incomplete AUC@20 in {run_id}")
        rows.append(
            {
                "run_id": run_id,
                "variant_index": int(job["variant_index"]),
                "config_hash": config_hash(job["method_config"]),
                "parameters_json": canonical_json(job["method_config"]),
                "rounds": int(job["rounds"]),
                "last50_validation_accuracy": float(result["last50_accuracy"]),
                "recovery_deficit_auc20": float(auc),
                "algorithm_elapsed_s": float(result["algorithm_elapsed_s"]),
                "validation_objective": float(result["last50_accuracy"]) - float(auc),
                "test_labels_used": False,
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["validation_objective", "algorithm_elapsed_s", "variant_index"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    frame["screen_rank"] = range(1, len(frame) + 1)
    frame["selected_for_gate"] = frame["screen_rank"] <= 2
    atomic_parquet(SCREEN_TABLE_PATH, frame)
    atomic_csv(SCREEN_CSV_PATH, frame)
    selected = []
    for row in frame.loc[frame["selected_for_gate"]].to_dict("records"):
        selected.append(
            {
                "screen_rank": int(row["screen_rank"]),
                "variant_index": int(row["variant_index"]),
                "run_id": row["run_id"],
                "validation_objective": float(row["validation_objective"]),
                "method_config": json.loads(row["parameters_json"]),
            }
        )
    atomic_json(
        SCREEN_SELECTION_PATH,
        {
            "schema_version": "1.0.0",
            "frozen_utc": utc_now(),
            "selection_split": "validation",
            "test_labels_used": False,
            "objective": manifest["selection_objective"],
            "selected": selected,
        },
    )


def worker() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        build_manifest()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    events = _load_events()
    state["status"] = "running"
    _sync(state, manifest)
    atomic_json(STATE_PATH, state)
    for position, job in enumerate(manifest["jobs"]):
        if job["status"] == "completed":
            continue
        resolved = _resolved_job(job)
        job["attempts"] = int(job.get("attempts", 0)) + 1
        job["actual_run_id"] = resolved["run_id"]
        job["status"] = "running"
        job["failure_reason"] = None
        state.update({"status": "running", "current_job_id": job["job_id"]})
        _sync(state, manifest)
        atomic_json(MANIFEST_PATH, manifest)
        atomic_json(STATE_PATH, state)
        _event(events, job, "submitted", queue_position=position)
        job_file = QUEUE_ROOT / "active_jobs" / f"{resolved['run_id']}.json"
        atomic_json(job_file, resolved)
        _event(events, job, "started", queue_position=position)
        log_path = QUEUE_ROOT / "worker_logs" / f"{resolved['run_id']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", newline="\n") as handle:
            process = subprocess.run(
                [sys.executable, "-m", "r2c_baselines.run", "--job", str(job_file)],
                cwd=str(Path(__file__).resolve().parents[1]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        success = (RUN_ROOT / str(resolved["run_id"]) / "_SUCCESS.json").exists()
        if process.returncode == 0 and success:
            job["status"] = "completed"
            _event(events, job, "completed", exit_code=0)
        else:
            job["status"] = "failed"
            job["failure_reason"] = f"exit_code={process.returncode};log={log_path}"
            state["status"] = "failed"
            _event(events, job, "failed", exit_code=process.returncode, reason=job["failure_reason"])
            _sync(state, manifest)
            atomic_json(MANIFEST_PATH, manifest)
            atomic_json(STATE_PATH, state)
            return state
        _sync(state, manifest)
        atomic_json(MANIFEST_PATH, manifest)
        atomic_json(STATE_PATH, state)
    freeze_screen(manifest)
    state.update({"status": "screen_completed", "current_job_id": None})
    _sync(state, manifest)
    atomic_json(STATE_PATH, state)
    return state


def status() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"status": "not_built"}
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        state["jobs"] = [
            {
                "job_id": job["job_id"],
                "status": job["status"],
                "attempts": job["attempts"],
                "actual_run_id": job["actual_run_id"],
            }
            for job in manifest["jobs"]
        ]
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--force", action="store_true")
    sub.add_parser("worker")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "build":
        value = build_manifest(force=args.force)
    elif args.command == "worker":
        value = worker()
    else:
        value = status()
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
