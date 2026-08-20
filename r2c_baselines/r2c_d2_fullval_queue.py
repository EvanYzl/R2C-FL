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
from .r2c_v2 import PROTOCOL_VERSION
from .utils import atomic_csv, atomic_json, atomic_parquet, config_hash, sha256_file, utc_now


WINNER_PATH = QUEUE_ROOT / "r2c_d2_gate_winner.json"
MANIFEST_PATH = QUEUE_ROOT / "r2c_d2_fullval_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d2_fullval_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d2_fullval_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d2_full_validation_result.json"
TABLE_PATH = PLOT_ROOT / "r2c_d2_full_validation.parquet"
CSV_PATH = PLOT_ROOT / "r2c_d2_full_validation.csv"

TARGET_ACCURACY = 0.5948964
S0_ACCURACY_GATE = 0.660996
S4_ACCURACY_GATE = 0.660996
S4_AUC_GATE = 0.002062
S4_TTA_GATE_S = 483.753226799716


def _hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = ("data.py", "training.py", "r2c.py", "r2c_v2.py", "run.py", Path(__file__).name)
    return {name: sha256_file(package / name) for name in names}


def _job(scenario_id: str, config: dict[str, Any], variant_index: int) -> dict[str, Any]:
    job_id = f"A-R2C-D2-{scenario_id}-FULLVAL-V{variant_index}-s{DEV_SEED}"
    return {
        "job_id": job_id,
        "base_run_id": job_id,
        "stage": "full_validation",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": "D2",
        "scenario_id": scenario_id,
        "rounds": 1000,
        "method_config": config,
        "block_id": "A-R2C-D2-FULLVAL",
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
    if not WINNER_PATH.exists():
        raise RuntimeError("Full validation requires frozen paired-gate winner")
    winner = json.loads(WINNER_PATH.read_text(encoding="utf-8"))
    if winner.get("test_labels_used") or winner.get("selection_split") != "validation":
        raise RuntimeError("Winner violates validation-only selection")
    config = dict(winner["method_config"])
    if config.get("r2c_protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Protocol version drift")
    variant_index = int(winner["variant_index"])
    jobs = [_job(scenario_id, config, variant_index) for scenario_id in ("S0", "S4")]
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D2_only_full_validation",
        "protocol_version": PROTOCOL_VERSION,
        "dev_seed": DEV_SEED,
        "formal_test_access": False,
        "winner_hash": config_hash(winner),
        "formal_launch_rule": "at least 3 of 4 frozen validation gates",
        "frozen_gates": {
            "s0_last50_accuracy_gt": S0_ACCURACY_GATE,
            "s4_last50_accuracy_gt": S4_ACCURACY_GATE,
            "s4_recovery_deficit_auc20_lt": S4_AUC_GATE,
            "s4_algorithm_tta_s_lt": S4_TTA_GATE_S,
            "target_accuracy": TARGET_ACCURACY,
        },
        "implementation_hashes": _hashes(),
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d2_fullval_manifest_{stamp}.json", manifest)
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


def _load_events() -> list[dict[str, Any]]:
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


def _actual(base: str) -> tuple[str, str | None]:
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
        completed=statuses.count("completed"),
        failed=statuses.count("failed"),
        total=len(statuses),
        updated_utc=utc_now(),
    )


def freeze_result(manifest: dict[str, Any]) -> None:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        return
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run_id = str(job["actual_run_id"])
        run_dir = RUN_ROOT / run_id
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        rounds = read_chunked_table(run_dir, "round_metrics")
        hits = rounds.loc[rounds["test_accuracy"] >= TARGET_ACCURACY]
        first_hit_round = None if hits.empty else int(hits.iloc[0]["round"])
        tta_s = None if hits.empty else float(hits.iloc[0]["algorithm_elapsed_s"])
        auc = result["recovery"]["recovery_deficit_auc20"]
        rows.append(
            {
                "run_id": run_id,
                "scenario_id": job["scenario_id"],
                "rounds": int(job["rounds"]),
                "last50_validation_accuracy": float(result["last50_accuracy"]),
                "recovery_deficit_auc20": None if auc is None else float(auc),
                "first_target_round": first_hit_round,
                "algorithm_tta_s": tta_s,
                "algorithm_elapsed_s": float(result["algorithm_elapsed_s"]),
                "test_labels_used": False,
            }
        )
    frame = pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)
    s0 = frame.loc[frame["scenario_id"] == "S0"].iloc[0]
    s4 = frame.loc[frame["scenario_id"] == "S4"].iloc[0]
    checks = {
        "s0_accuracy": float(s0["last50_validation_accuracy"]) > S0_ACCURACY_GATE,
        "s4_accuracy": float(s4["last50_validation_accuracy"]) > S4_ACCURACY_GATE,
        "s4_auc20": float(s4["recovery_deficit_auc20"]) < S4_AUC_GATE,
        "s4_tta": pd.notna(s4["algorithm_tta_s"]) and float(s4["algorithm_tta_s"]) < S4_TTA_GATE_S,
    }
    passed = sum(bool(value) for value in checks.values())
    frame["formal_launch_gate_passed"] = passed >= 3
    atomic_parquet(TABLE_PATH, frame)
    atomic_csv(CSV_PATH, frame)
    atomic_json(
        RESULT_PATH,
        {
            "schema_version": "1.0.0",
            "completed_utc": utc_now(),
            "selection_split": "validation",
            "test_labels_used": False,
            "checks": checks,
            "passed_count": passed,
            "required_count": 3,
            "formal_launch_gate_passed": passed >= 3,
            "frozen_gates": manifest["frozen_gates"],
            "rows": frame.to_dict("records"),
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
        run_id, retry_of = _actual(str(job["base_run_id"]))
        resolved = dict(job)
        resolved.update(run_id=run_id, retry_of_run_id=retry_of, queue_utc=utc_now())
        job.update(attempts=int(job.get("attempts", 0)) + 1, actual_run_id=run_id, status="running", failure_reason=None)
        state.update(status="running", current_job_id=job["job_id"])
        _sync(state, manifest)
        atomic_json(MANIFEST_PATH, manifest)
        atomic_json(STATE_PATH, state)
        _event(events, job, "submitted", queue_position=position)
        job_file = QUEUE_ROOT / "active_jobs" / f"{run_id}.json"
        atomic_json(job_file, resolved)
        _event(events, job, "started", queue_position=position)
        log_path = QUEUE_ROOT / "worker_logs" / f"{run_id}.log"
        with log_path.open("w", encoding="utf-8", newline="\n") as handle:
            process = subprocess.run(
                [sys.executable, "-m", "r2c_baselines.run", "--job", str(job_file)],
                cwd=str(Path(__file__).resolve().parents[1]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        success = (RUN_ROOT / run_id / "_SUCCESS.json").exists()
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
    freeze_result(manifest)
    gate = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    state.update(
        status="full_validation_completed",
        current_job_id=None,
        formal_launch_gate_passed=bool(gate["formal_launch_gate_passed"]),
        passed_count=int(gate["passed_count"]),
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
