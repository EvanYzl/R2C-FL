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
from .utils import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    canonical_json,
    config_hash,
    sha256_file,
    utc_now,
)


SCREEN_SELECTION_PATH = QUEUE_ROOT / "r2c_d2_v3_screen_selection.json"
MANIFEST_PATH = QUEUE_ROOT / "r2c_d2_v3_gate_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d2_v3_gate_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d2_v3_gate_scheduler_events.parquet"
GATE_RUNS_PATH = PLOT_ROOT / "r2c_d2_v3_gate_runs.parquet"
GATE_RUNS_CSV_PATH = PLOT_ROOT / "r2c_d2_v3_gate_runs.csv"
GATE_TABLE_PATH = PLOT_ROOT / "r2c_d2_v3_gate_trials.parquet"
GATE_CSV_PATH = PLOT_ROOT / "r2c_d2_v3_gate_trials.csv"
WINNER_PATH = QUEUE_ROOT / "r2c_d2_v3_gate_winner.json"
TARGET_ACCURACY = 0.5948964


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


def _job(
    screen_rank: int,
    label: str,
    scenario_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    job_id = f"A-R2C-D2-{scenario_id}-V3-GATE-{label}-s{DEV_SEED}"
    return {
        "job_id": job_id,
        "base_run_id": job_id,
        "stage": "v3_paired_gate",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": "D2",
        "scenario_id": scenario_id,
        "rounds": 400,
        "method_config": config,
        "block_id": "A-R2C-D2-V3-GATE",
        "seed": DEV_SEED,
        "partition_seed": DEV_SEED,
        "trace_seed": DEV_SEED,
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "screen_rank": int(screen_rank),
        "variant_label": label,
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def build_manifest(persist: bool = True, force: bool = False) -> dict[str, Any]:
    if persist and MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not SCREEN_SELECTION_PATH.exists():
        raise RuntimeError("V3 gate requires frozen screen selection")
    selection = json.loads(SCREEN_SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("test_labels_used") or selection.get("selection_split") != "validation":
        raise RuntimeError("V3 screen selection is not validation-only")
    selected = selection.get("selected", [])
    if len(selected) != 2:
        raise RuntimeError("V3 paired gate requires exactly two screen selections")
    jobs: list[dict[str, Any]] = []
    for rank, entry in enumerate(selected, start=1):
        config = dict(entry["method_config"])
        if config.get("r2c_protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError("V3 protocol drift in screen selection")
        label = str(entry["variant_label"])
        for scenario_id in ("S0", "S4"):
            jobs.append(_job(rank, label, scenario_id, config))
    if len(jobs) != 4 or len({job["job_id"] for job in jobs}) != 4:
        raise AssertionError("V3 paired gate must contain four unique jobs")
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D2_only_v3_paired_validation_gate",
        "protocol_version": PROTOCOL_VERSION,
        "dev_seed": DEV_SEED,
        "formal_test_access": False,
        "rounds_per_job": 400,
        "target_accuracy": TARGET_ACCURACY,
        "selection_objective": "0.5*(S0_last50 + S4_last50) - 2*S4_recovery_deficit_auc20; paired_algorithm_elapsed_s; screen_rank",
        "screen_selection_hash": config_hash(selection),
        "implementation_hashes": _implementation_hashes(),
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d2_v3_gate_manifest_{stamp}.json", manifest)
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


def freeze_gate(manifest: dict[str, Any]) -> None:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        return
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run_id = str(job["actual_run_id"])
        run_dir = RUN_ROOT / run_id
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        auc = result["recovery"]["recovery_deficit_auc20"]
        if job["scenario_id"] == "S4" and auc is None:
            raise RuntimeError(f"Incomplete S4 AUC@20 in {run_id}")
        tta_round, tta_s = _time_to_accuracy(run_dir, TARGET_ACCURACY)
        rows.append(
            {
                "run_id": run_id,
                "variant_label": job["variant_label"],
                "screen_rank": int(job["screen_rank"]),
                "scenario_id": job["scenario_id"],
                "config_hash": config_hash(job["method_config"]),
                "parameters_json": canonical_json(job["method_config"]),
                "last50_validation_accuracy": float(result["last50_accuracy"]),
                "recovery_deficit_auc20": None if auc is None else float(auc),
                "algorithm_elapsed_s": float(result["algorithm_elapsed_s"]),
                "target_accuracy": TARGET_ACCURACY,
                "tta_round": tta_round,
                "algorithm_tta_s": tta_s,
                "test_labels_used": False,
            }
        )
    run_frame = pd.DataFrame(rows)
    atomic_parquet(GATE_RUNS_PATH, run_frame)
    atomic_csv(GATE_RUNS_CSV_PATH, run_frame)
    summaries: list[dict[str, Any]] = []
    for label, group in run_frame.groupby("variant_label", sort=False):
        if set(group["scenario_id"]) != {"S0", "S4"}:
            raise RuntimeError(f"Incomplete S0/S4 gate pair for {label}")
        s0 = group.loc[group["scenario_id"] == "S0"].iloc[0]
        s4 = group.loc[group["scenario_id"] == "S4"].iloc[0]
        score = 0.5 * (
            float(s0["last50_validation_accuracy"])
            + float(s4["last50_validation_accuracy"])
        ) - 2.0 * float(s4["recovery_deficit_auc20"])
        summaries.append(
            {
                "variant_label": label,
                "screen_rank": int(s0["screen_rank"]),
                "s0_run_id": s0["run_id"],
                "s4_run_id": s4["run_id"],
                "s0_last50_validation_accuracy": float(s0["last50_validation_accuracy"]),
                "s4_last50_validation_accuracy": float(s4["last50_validation_accuracy"]),
                "s4_recovery_deficit_auc20": float(s4["recovery_deficit_auc20"]),
                "s4_algorithm_tta_s": (
                    None if pd.isna(s4["algorithm_tta_s"]) else float(s4["algorithm_tta_s"])
                ),
                "paired_algorithm_elapsed_s": float(s0["algorithm_elapsed_s"])
                + float(s4["algorithm_elapsed_s"]),
                "gate_score": score,
                "parameters_json": s0["parameters_json"],
                "test_labels_used": False,
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["gate_score", "paired_algorithm_elapsed_s", "screen_rank"],
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
            "variant_label": str(winner["variant_label"]),
            "gate_score": float(winner["gate_score"]),
            "source_run_ids": [str(winner["s0_run_id"]), str(winner["s4_run_id"])],
            "method_config": json.loads(str(winner["parameters_json"])),
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
            {
                key: job.get(key)
                for key in (
                    "job_id",
                    "variant_label",
                    "scenario_id",
                    "status",
                    "attempts",
                    "actual_run_id",
                )
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
