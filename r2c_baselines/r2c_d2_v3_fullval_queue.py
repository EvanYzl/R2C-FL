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
from .r2c_v3 import PROTOCOL_VERSION
from .utils import atomic_csv, atomic_json, atomic_parquet, config_hash, sha256_file, utc_now


WINNER_PATH = QUEUE_ROOT / "r2c_d2_v3_gate_winner.json"
MANIFEST_PATH = QUEUE_ROOT / "r2c_d2_v3_fullval_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d2_v3_fullval_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d2_v3_fullval_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d2_v3_full_validation_result.json"
RUNS_PATH = PLOT_ROOT / "r2c_d2_v3_full_validation_runs.parquet"
RUNS_CSV_PATH = PLOT_ROOT / "r2c_d2_v3_full_validation_runs.csv"
TARGET_ACCURACY = 0.5948964
THRESHOLDS = {
    "s0_last50_accuracy": 0.660996,
    "s4_last50_accuracy": 0.660996,
    "s4_recovery_deficit_auc20": 0.002062,
    "s4_algorithm_tta_s": 483.753226799716,
}
CLOSE_LIMITS = {
    "accuracy_fraction": 0.002,
    "auc_fraction": 0.0005,
    "tta_multiplier": 1.05,
}


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        "data.py",
        "training.py",
        "r2c.py",
        "r2c_v2.py",
        "r2c_v3.py",
        "run.py",
        Path(__file__).name,
    )
    return {name: sha256_file(package / name) for name in names}


def _job(scenario_id: str, config: dict[str, Any]) -> dict[str, Any]:
    job_id = f"A-R2C-D2-{scenario_id}-V3-FULLVAL-F075-s{DEV_SEED}"
    return {
        "job_id": job_id,
        "base_run_id": job_id,
        "stage": "v3_full_validation",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": "D2",
        "scenario_id": scenario_id,
        "rounds": 1000,
        "method_config": config,
        "block_id": "A-R2C-D2-V3-FULLVAL",
        "seed": DEV_SEED,
        "partition_seed": DEV_SEED,
        "trace_seed": DEV_SEED,
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "variant_label": "F075",
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def build_manifest(persist: bool = True, force: bool = False) -> dict[str, Any]:
    if persist and MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not WINNER_PATH.exists():
        raise RuntimeError("V3 full validation requires frozen gate winner")
    winner = json.loads(WINNER_PATH.read_text(encoding="utf-8"))
    if winner.get("test_labels_used") or winner.get("selection_split") != "validation":
        raise RuntimeError("V3 gate winner is not validation-only")
    if winner.get("variant_label") != "F075":
        raise RuntimeError("Unexpected V3 gate winner")
    config = dict(winner["method_config"])
    if config.get("r2c_protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("V3 protocol drift in gate winner")
    jobs = [_job(scenario_id, config) for scenario_id in ("S0", "S4")]
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D2_only_v3_full_validation",
        "protocol_version": PROTOCOL_VERSION,
        "dev_seed": DEV_SEED,
        "formal_test_access": False,
        "rounds_per_job": 1000,
        "target_accuracy": TARGET_ACCURACY,
        "strict_thresholds": THRESHOLDS,
        "close_limits": CLOSE_LIMITS,
        "formal_authorization_rule": "all four strict, or exactly three strict and the sole miss is within its frozen close limit",
        "winner_hash": config_hash(winner),
        "implementation_hashes": _implementation_hashes(),
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d2_v3_fullval_manifest_{stamp}.json", manifest)
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
                "total": 2,
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


def _time_to_accuracy(run_dir: Path, target: float) -> tuple[int | None, float | None]:
    rounds = read_chunked_table(run_dir, "round_metrics")
    reached = rounds.loc[rounds["test_accuracy"].astype(float) >= float(target)]
    if reached.empty:
        return None, None
    first = reached.sort_values("round").iloc[0]
    return int(first["round"]), float(first["algorithm_elapsed_s"])


def _evaluate_authorization(observed: dict[str, float | None]) -> dict[str, Any]:
    checks = {
        "s0_last50_accuracy": float(observed["s0_last50_accuracy"]) > THRESHOLDS["s0_last50_accuracy"],
        "s4_last50_accuracy": float(observed["s4_last50_accuracy"]) > THRESHOLDS["s4_last50_accuracy"],
        "s4_recovery_deficit_auc20": float(observed["s4_recovery_deficit_auc20"])
        < THRESHOLDS["s4_recovery_deficit_auc20"],
        "s4_algorithm_tta_s": observed["s4_algorithm_tta_s"] is not None
        and float(observed["s4_algorithm_tta_s"]) < THRESHOLDS["s4_algorithm_tta_s"],
    }
    close = {
        "s0_last50_accuracy": float(observed["s0_last50_accuracy"])
        >= THRESHOLDS["s0_last50_accuracy"] - CLOSE_LIMITS["accuracy_fraction"],
        "s4_last50_accuracy": float(observed["s4_last50_accuracy"])
        >= THRESHOLDS["s4_last50_accuracy"] - CLOSE_LIMITS["accuracy_fraction"],
        "s4_recovery_deficit_auc20": float(observed["s4_recovery_deficit_auc20"])
        <= THRESHOLDS["s4_recovery_deficit_auc20"] + CLOSE_LIMITS["auc_fraction"],
        "s4_algorithm_tta_s": observed["s4_algorithm_tta_s"] is not None
        and float(observed["s4_algorithm_tta_s"])
        <= THRESHOLDS["s4_algorithm_tta_s"] * CLOSE_LIMITS["tta_multiplier"],
    }
    misses = [key for key, value in checks.items() if not value]
    authorized = len(misses) == 0 or (len(misses) == 1 and close[misses[0]])
    return {
        "strict_checks": checks,
        "close_checks": close,
        "strict_passes": sum(checks.values()),
        "sole_miss": misses[0] if len(misses) == 1 else None,
        "formal_authorized": bool(authorized),
    }


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot freeze incomplete full validation")
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run_id = str(job["actual_run_id"])
        run_dir = RUN_ROOT / run_id
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        auc = result["recovery"]["recovery_deficit_auc20"]
        if job["scenario_id"] == "S4" and auc is None:
            raise RuntimeError("Full-validation S4 lacks complete AUC@20")
        tta_round, tta_s = _time_to_accuracy(run_dir, TARGET_ACCURACY)
        rows.append(
            {
                "run_id": run_id,
                "scenario_id": job["scenario_id"],
                "last50_validation_accuracy": float(result["last50_accuracy"]),
                "recovery_deficit_auc20": None if auc is None else float(auc),
                "algorithm_elapsed_s": float(result["algorithm_elapsed_s"]),
                "tta_round": tta_round,
                "algorithm_tta_s": tta_s,
                "source_kind": "CALIBRATION",
                "test_labels_used": False,
            }
        )
    frame = pd.DataFrame(rows)
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    s0 = frame.loc[frame["scenario_id"] == "S0"].iloc[0]
    s4 = frame.loc[frame["scenario_id"] == "S4"].iloc[0]
    observed: dict[str, float | None] = {
        "s0_last50_accuracy": float(s0["last50_validation_accuracy"]),
        "s4_last50_accuracy": float(s4["last50_validation_accuracy"]),
        "s4_recovery_deficit_auc20": float(s4["recovery_deficit_auc20"]),
        "s4_algorithm_tta_s": (
            None if pd.isna(s4["algorithm_tta_s"]) else float(s4["algorithm_tta_s"])
        ),
    }
    authorization = _evaluate_authorization(observed)
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "dataset_id": "D2",
        "protocol_version": PROTOCOL_VERSION,
        "variant_label": "F075",
        "selection_split": "validation",
        "test_labels_used": False,
        "thresholds": THRESHOLDS,
        "close_limits": CLOSE_LIMITS,
        "observed": observed,
        **authorization,
        "run_ids": frame["run_id"].tolist(),
    }
    atomic_json(RESULT_PATH, payload)
    return payload


def worker() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        build_manifest()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    events = _events()
    state["status"] = "running"
    _sync(state, manifest)
    atomic_json(STATE_PATH, state)
    for position, job in enumerate(manifest["jobs"]):
        if job["status"] == "completed":
            continue
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
        _event(events, job, "submitted", queue_position=position)
        job_file = QUEUE_ROOT / "active_jobs" / f"{resolved['run_id']}.json"
        atomic_json(job_file, resolved)
        log_path = QUEUE_ROOT / "worker_logs" / f"{resolved['run_id']}.log"
        _event(events, job, "started", queue_position=position, log_path=str(log_path))
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
    result = freeze_result(manifest)
    state.update(
        {
            "status": (
                "full_validation_completed_authorized"
                if result["formal_authorized"]
                else "full_validation_completed_not_authorized"
            ),
            "current_job_id": None,
            "formal_authorized": bool(result["formal_authorized"]),
        }
    )
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
                key: job.get(key)
                for key in ("job_id", "scenario_id", "status", "attempts", "actual_run_id")
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
