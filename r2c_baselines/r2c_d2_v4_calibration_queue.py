from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEV_SEED, PLOT_ROOT, QUEUE_ROOT, RUN_ROOT
from .logging_io import read_chunked_table
from .metrics import recovery_auc20
from .r2c_v4 import PROTOCOL_VERSION
from .utils import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    canonical_json,
    config_hash,
    sha256_file,
    utc_now,
)


BETAS = (0.80, 0.90, 0.95)
TARGET_ACCURACY = 0.5948964
MIN_LAST50 = 0.660996
MAX_ADVANCE_AUC = 0.002562
MANIFEST_PATH = QUEUE_ROOT / "r2c_d2_v4_calibration_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d2_v4_calibration_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d2_v4_calibration_scheduler_events.parquet"
SELECTION_PATH = QUEUE_ROOT / "r2c_d2_v4_calibration_selection.json"
TABLE_PATH = PLOT_ROOT / "r2c_d2_v4_calibration_candidates.parquet"
CSV_PATH = PLOT_ROOT / "r2c_d2_v4_calibration_candidates.csv"


def _common_config() -> dict[str, Any]:
    return {
        "lr_mult": 1.5,
        "r2c_delta_clip": 0.6107748032302804,
        "r2c_eval_microbatch": 4,
        "r2c_protocol_version": PROTOCOL_VERSION,
        "r2c_v2_agreement_threshold": 0.8,
        "r2c_v2_anchor_per_fold": 16,
        "r2c_v2_audit_replay": False,
        "r2c_v2_completion_power": 0.0,
        "r2c_v2_finish_scale": 0.08,
        "r2c_v2_finish_weight": 0.75,
        "r2c_v2_floor_fraction": 0.20,
        "r2c_v2_propensity_power": 0.0,
        "r2c_v2_radius_multiplier": 1.0,
        "r2c_v2_scale_floor": 1.0e-4,
        "r2c_v2_scout_steps": 1,
        "r2c_v2_temperature": 0.35,
        "r2c_v2_uncertainty_floor_fraction": 0.35,
        "r2c_v2_value_clip": 4.0,
        "r2c_v2_weight_cap": 2.0,
        "r2c_v3_fixed_server_alpha": 0.75,
        "r2c_v3_guard_per_fold": 32,
        "r2c_v3_guard_relative_tolerance": 0.0,
        "r2c_v3_server_alphas": [1.0, 0.75, 0.5, 0.25, 0.0],
        "r2c_v4_deployment_ema_betas": list(BETAS),
        "r2c_v4_primary_deployment_beta": BETAS[0],
    }


def _job() -> dict[str, Any]:
    job_id = f"A-R2C-D2-S4-V4-CAL-EMA3-s{DEV_SEED}"
    return {
        "job_id": job_id,
        "base_run_id": job_id,
        "stage": "v4_shared_ema_calibration",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": "D2",
        "scenario_id": "S4",
        "rounds": 1000,
        "method_config": _common_config(),
        "block_id": "A-R2C-D2-V4-CAL",
        "seed": DEV_SEED,
        "partition_seed": DEV_SEED,
        "trace_seed": DEV_SEED,
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "variant_label": "EMA3",
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        "data.py",
        "training.py",
        "r2c.py",
        "r2c_v2.py",
        "r2c_v3.py",
        "r2c_v4.py",
        "run.py",
        Path(__file__).name,
    )
    return {name: sha256_file(package / name) for name in names}


def build_manifest(persist: bool = True, force: bool = False) -> dict[str, Any]:
    if persist and MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D2_only_v4_shared_ema_validation_calibration",
        "protocol_version": PROTOCOL_VERSION,
        "dev_seed": DEV_SEED,
        "formal_test_access": False,
        "rounds_per_job": 1000,
        "candidate_betas": list(BETAS),
        "selection_rule": "eligible Last50 > 0.660996; then minimum AUC@20, higher Last50, smaller beta",
        "advance_rule": {"last50_gt": MIN_LAST50, "auc20_lte": MAX_ADVANCE_AUC},
        "implementation_hashes": _implementation_hashes(),
        "jobs": [_job()],
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d2_v4_calibration_manifest_{stamp}.json", manifest)
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
                "total": 1,
            },
        )
    return manifest


def _events() -> list[dict[str, Any]]:
    return pd.read_parquet(EVENTS_PATH).to_dict("records") if EVENTS_PATH.exists() else []


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
    path = RUN_ROOT / base
    if not path.exists() or (path / "_SUCCESS.json").exists():
        return base, None
    attempt = 2
    while (RUN_ROOT / f"{base}-a{attempt}").exists():
        attempt += 1
    return f"{base}-a{attempt}", base


def _sync(state: dict[str, Any], manifest: dict[str, Any]) -> None:
    statuses = [job["status"] for job in manifest["jobs"]]
    state.update(
        {
            "completed": statuses.count("completed"),
            "failed": statuses.count("failed"),
            "total": len(statuses),
            "updated_utc": utc_now(),
        }
    )


def _resolved(job: dict[str, Any]) -> dict[str, Any]:
    value = dict(job)
    run_id, retry_of = _actual_run_id(str(job["base_run_id"]))
    value.update({"run_id": run_id, "retry_of_run_id": retry_of, "queue_utc": utc_now()})
    return value


def _select_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.loc[frame["last50_validation_accuracy"] > MIN_LAST50].copy()
    return eligible.sort_values(
        ["recovery_deficit_auc20", "last50_validation_accuracy", "deployment_beta"],
        ascending=[True, False, True],
        kind="mergesort",
    )


def freeze_selection(manifest: dict[str, Any]) -> dict[str, Any]:
    job = manifest["jobs"][0]
    if job["status"] != "completed":
        raise RuntimeError("Cannot freeze incomplete v4 calibration")
    run_id = str(job["actual_run_id"])
    run_dir = RUN_ROOT / run_id
    rounds = read_chunked_table(run_dir, "round_metrics").sort_values("round")
    candidates = read_chunked_table(run_dir, "deployment_candidate_metrics")
    event_rows = rounds.loc[rounds["event_offset_round"].astype(int) == 0]
    if len(event_rows) != 1:
        raise RuntimeError("V4 calibration must have exactly one event boundary")
    event_round = int(event_rows.iloc[0]["round"])
    rows: list[dict[str, Any]] = []
    for beta, group in candidates.groupby("deployment_beta", sort=True):
        group = group.sort_values("round")
        if group["round"].astype(int).tolist() != list(range(1, int(job["rounds"]) + 1)):
            raise RuntimeError(f"Incomplete deployment trajectory for beta={beta}")
        recovery = recovery_auc20(
            group["round"].astype(int).tolist(),
            group["test_accuracy"].astype(float).tolist(),
            event_round,
        )
        if not recovery["recovery_auc20_complete"]:
            raise RuntimeError(f"Incomplete AUC@20 for beta={beta}")
        merged = group.merge(
            rounds[["round", "algorithm_elapsed_s"]], on="round", validate="one_to_one"
        )
        reached = merged.loc[merged["test_accuracy"].astype(float) >= TARGET_ACCURACY]
        rows.append(
            {
                "run_id": run_id,
                "deployment_beta": float(beta),
                "last50_validation_accuracy": float(group.tail(50)["test_accuracy"].mean()),
                "recovery_deficit_auc20": float(recovery["recovery_deficit_auc20"]),
                "max_drop": float(recovery["max_drop"]),
                "tta_round_shared_calibration": (
                    None if reached.empty else int(reached.iloc[0]["round"])
                ),
                "algorithm_tta_s_shared_calibration": (
                    None if reached.empty else float(reached.iloc[0]["algorithm_elapsed_s"])
                ),
                "accuracy_eligible": bool(float(group.tail(50)["test_accuracy"].mean()) > MIN_LAST50),
                "advance_eligible": bool(
                    float(group.tail(50)["test_accuracy"].mean()) > MIN_LAST50
                    and float(recovery["recovery_deficit_auc20"]) <= MAX_ADVANCE_AUC
                ),
                "test_labels_used": False,
            }
        )
    frame = pd.DataFrame(rows)
    ranked = _select_candidate_frame(frame)
    if ranked.empty:
        winner = None
        advance = False
    else:
        winner = ranked.iloc[0]
        advance = bool(winner["advance_eligible"])
    frame["rank"] = None
    for rank, index in enumerate(ranked.index, start=1):
        frame.loc[index, "rank"] = rank
    atomic_parquet(TABLE_PATH, frame.sort_values("deployment_beta"))
    atomic_csv(CSV_PATH, frame.sort_values("deployment_beta"))
    winner_config = None
    if winner is not None:
        winner_config = dict(job["method_config"])
        winner_config["r2c_v4_deployment_ema_betas"] = [float(winner["deployment_beta"])]
        winner_config["r2c_v4_primary_deployment_beta"] = float(winner["deployment_beta"])
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "dataset_id": "D2",
        "protocol_version": PROTOCOL_VERSION,
        "selection_split": "validation",
        "test_labels_used": False,
        "run_id": run_id,
        "candidate_betas": list(BETAS),
        "winner_beta": None if winner is None else float(winner["deployment_beta"]),
        "winner_last50_validation_accuracy": (
            None if winner is None else float(winner["last50_validation_accuracy"])
        ),
        "winner_recovery_deficit_auc20": (
            None if winner is None else float(winner["recovery_deficit_auc20"])
        ),
        "advance_to_full_validation": advance,
        "winner_method_config": winner_config,
        "source_config_hash": config_hash(job["method_config"]),
        "winner_config_hash": None if winner_config is None else config_hash(winner_config),
        "selection_rule": manifest["selection_rule"],
        "advance_rule": manifest["advance_rule"],
    }
    atomic_json(SELECTION_PATH, payload)
    return payload


def worker() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        build_manifest()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    events = _events()
    job = manifest["jobs"][0]
    if job["status"] != "completed":
        resolved = _resolved(job)
        job.update(
            {
                "attempts": int(job.get("attempts", 0)) + 1,
                "actual_run_id": resolved["run_id"],
                "status": "running",
                "failure_reason": None,
            }
        )
        state.update({"status": "running", "current_job_id": job["job_id"]})
        _sync(state, manifest)
        atomic_json(MANIFEST_PATH, manifest)
        atomic_json(STATE_PATH, state)
        _event(events, job, "submitted")
        job_file = QUEUE_ROOT / "active_jobs" / f"{resolved['run_id']}.json"
        atomic_json(job_file, resolved)
        log_path = QUEUE_ROOT / "worker_logs" / f"{resolved['run_id']}.log"
        _event(events, job, "started", log_path=str(log_path))
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
    selection = freeze_selection(manifest)
    state.update(
        {
            "status": (
                "calibration_completed_advance"
                if selection["advance_to_full_validation"]
                else "calibration_completed_no_advance"
            ),
            "current_job_id": None,
            "advance_to_full_validation": bool(selection["advance_to_full_validation"]),
            "winner_beta": selection["winner_beta"],
        }
    )
    _sync(state, manifest)
    atomic_json(STATE_PATH, state)
    return state


def status() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"status": "not_built"}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


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
