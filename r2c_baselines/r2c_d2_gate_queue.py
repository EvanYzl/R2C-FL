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


SCREEN_SELECTION_PATH = QUEUE_ROOT / "r2c_d2_opt_screen_selection.json"
MANIFEST_PATH = QUEUE_ROOT / "r2c_d2_gate_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d2_gate_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d2_gate_scheduler_events.parquet"
GATE_TABLE_PATH = PLOT_ROOT / "r2c_d2_gate_trials.parquet"
GATE_CSV_PATH = PLOT_ROOT / "r2c_d2_gate_trials.csv"
WINNER_PATH = QUEUE_ROOT / "r2c_d2_gate_winner.json"
TARGET_ACCURACY = 0.5948964


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = ("data.py", "training.py", "r2c.py", "r2c_v2.py", "run.py", Path(__file__).name)
    return {name: sha256_file(package / name) for name in names}


def _job(variant_index: int, scenario_id: str, config: dict[str, Any]) -> dict[str, Any]:
    job_id = f"A-R2C-D2-{scenario_id}-GATE-V{variant_index}-s{DEV_SEED}"
    return {
        "job_id": job_id,
        "base_run_id": job_id,
        "stage": "paired_gate",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": "D2",
        "scenario_id": scenario_id,
        "rounds": 400,
        "method_config": config,
        "block_id": "A-R2C-D2-GATE",
        "seed": DEV_SEED,
        "partition_seed": DEV_SEED,
        "trace_seed": DEV_SEED,
        "evaluation_split": "validation",
        "full_logging": False,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "variant_index": int(variant_index),
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def build_manifest(persist: bool = True, force: bool = False) -> dict[str, Any]:
    if persist and MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not SCREEN_SELECTION_PATH.exists():
        raise RuntimeError("Paired gate requires frozen validation screen selection")
    selection = json.loads(SCREEN_SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("test_labels_used") or selection.get("selection_split") != "validation":
        raise RuntimeError("Screen selection violates validation-only gate")
    selected = selection.get("selected", [])
    if len(selected) != 2:
        raise RuntimeError("Paired gate requires exactly two screen winners")
    jobs: list[dict[str, Any]] = []
    for entry in selected:
        config = dict(entry["method_config"])
        if config.get("r2c_protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError("Protocol version drift in screen selection")
        for scenario_id in ("S0", "S4"):
            jobs.append(_job(int(entry["variant_index"]), scenario_id, config))
    if len(jobs) != 4 or len({job["job_id"] for job in jobs}) != 4:
        raise AssertionError("Paired gate must contain four unique jobs")
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D2_only_paired_validation_gate",
        "protocol_version": PROTOCOL_VERSION,
        "dev_seed": DEV_SEED,
        "formal_test_access": False,
        "rounds_per_job": 400,
        "selection_objective": "0.5*(S0_last50 + S4_last50) - S4_recovery_deficit_auc20; paired_algorithm_elapsed_s tie-break",
        "screen_selection_hash": config_hash(selection),
        "implementation_hashes": _implementation_hashes(),
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d2_gate_manifest_{stamp}.json", manifest)
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
                "total": 4,
            },
        )
    return manifest


def _events() -> list[dict[str, Any]]:
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


def freeze_gate(manifest: dict[str, Any]) -> None:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        return
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run_id = str(job["actual_run_id"])
        result = json.loads((RUN_ROOT / run_id / "result.json").read_text(encoding="utf-8"))
        auc = result["recovery"]["recovery_deficit_auc20"]
        if job["scenario_id"] == "S4" and auc is None:
            raise RuntimeError(f"Incomplete S4 AUC@20 in {run_id}")
        rows.append(
            {
                "run_id": run_id,
                "variant_index": int(job["variant_index"]),
                "scenario_id": job["scenario_id"],
                "config_hash": config_hash(job["method_config"]),
                "parameters_json": canonical_json(job["method_config"]),
                "rounds": int(job["rounds"]),
                "last50_validation_accuracy": float(result["last50_accuracy"]),
                "recovery_deficit_auc20": None if auc is None else float(auc),
                "algorithm_elapsed_s": float(result["algorithm_elapsed_s"]),
                "test_labels_used": False,
            }
        )
    frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for variant_index, group in frame.groupby("variant_index"):
        if set(group["scenario_id"]) != {"S0", "S4"}:
            raise RuntimeError(f"Incomplete paired gate for V{variant_index}")
        s0 = group.loc[group["scenario_id"] == "S0"].iloc[0]
        s4 = group.loc[group["scenario_id"] == "S4"].iloc[0]
        objective = 0.5 * (
            float(s0["last50_validation_accuracy"]) + float(s4["last50_validation_accuracy"])
        ) - float(s4["recovery_deficit_auc20"])
        summary_rows.append(
            {
                "variant_index": int(variant_index),
                "s0_run_id": s0["run_id"],
                "s4_run_id": s4["run_id"],
                "s0_last50_validation_accuracy": float(s0["last50_validation_accuracy"]),
                "s4_last50_validation_accuracy": float(s4["last50_validation_accuracy"]),
                "s4_recovery_deficit_auc20": float(s4["recovery_deficit_auc20"]),
                "paired_algorithm_elapsed_s": float(s0["algorithm_elapsed_s"]) + float(s4["algorithm_elapsed_s"]),
                "gate_objective": objective,
                "parameters_json": s0["parameters_json"],
                "test_labels_used": False,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["gate_objective", "paired_algorithm_elapsed_s", "variant_index"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    summary["gate_rank"] = range(1, len(summary) + 1)
    summary["selected_for_full_validation"] = summary["gate_rank"] == 1
    atomic_parquet(GATE_TABLE_PATH, summary)
    atomic_csv(GATE_CSV_PATH, summary)
    winner = summary.iloc[0]
    atomic_json(
        WINNER_PATH,
        {
            "schema_version": "1.0.0",
            "frozen_utc": utc_now(),
            "selection_split": "validation",
            "test_labels_used": False,
            "objective": manifest["selection_objective"],
            "variant_index": int(winner["variant_index"]),
            "gate_objective": float(winner["gate_objective"]),
            "source_run_ids": [winner["s0_run_id"], winner["s4_run_id"]],
            "method_config": json.loads(winner["parameters_json"]),
        },
    )


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
        _event(events, job, "started", queue_position=position)
        log_path = QUEUE_ROOT / "worker_logs" / f"{resolved['run_id']}.log"
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
    freeze_gate(manifest)
    state.update({"status": "gate_completed", "current_job_id": None})
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
            {key: job.get(key) for key in ("job_id", "status", "attempts", "actual_run_id")}
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
