from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import FORMAL_SEED, PLOT_ROOT, QUEUE_ROOT, RUN_ROOT
from .logging_io import read_chunked_table
from .metrics import recovery_auc20
from .r2c_v2 import PROTOCOL_VERSION
from .utils import atomic_csv, atomic_json, atomic_parquet, config_hash, sha256_file, utc_now


WINNER_PATH = QUEUE_ROOT / "r2c_d2_gate_winner.json"
VALIDATION_PATH = QUEUE_ROOT / "r2c_d2_full_validation_result.json"
MANIFEST_PATH = QUEUE_ROOT / "r2c_d2_formal_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d2_formal_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d2_formal_scheduler_events.parquet"
AUDIT_PATH = QUEUE_ROOT / "r2c_d2_formal_audit_report.json"
RESULT_PATH = QUEUE_ROOT / "r2c_d2_formal_result.json"
RUN_TABLE_PATH = PLOT_ROOT / "r2c_d2_optimized_formal_runs.parquet"
RUN_CSV_PATH = PLOT_ROOT / "r2c_d2_optimized_formal_runs.csv"
VALUE_TABLE_PATH = PLOT_ROOT / "r2c_d2_optimized_table1_values.parquet"
VALUE_CSV_PATH = PLOT_ROOT / "r2c_d2_optimized_table1_values.csv"
VALUE_JSON_PATH = PLOT_ROOT / "r2c_d2_optimized_table1_values.json"
AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / "tests" / "audit_r2c_run.py"

TARGET_ACCURACY = 0.5948964
BASELINE_THRESHOLDS = {
    "s0_last50_accuracy": {
        "value": 0.687200,
        "direction": "greater",
        "method": "FedAvg",
        "unit": "fraction",
    },
    "s4_last50_accuracy": {
        "value": 0.688744,
        "direction": "greater",
        "method": "FedAU",
        "unit": "fraction",
    },
    "s4_recovery_deficit_auc20": {
        "value": 0.002062,
        "direction": "less",
        "method": "F3AST",
        "unit": "fraction",
    },
    "s4_algorithm_tta_s": {
        "value": 483.753226799716,
        "direction": "less",
        "method": "PowerOfChoice",
        "unit": "seconds",
    },
}


def _hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = ("config.py", "data.py", "training.py", "r2c.py", "r2c_v2.py", "run.py", Path(__file__).name)
    return {name: sha256_file(package / name) for name in names}


def _training_config(config: dict[str, Any]) -> dict[str, Any]:
    value = dict(config)
    value.pop("r2c_v2_audit_replay", None)
    return value


def _job(scenario_id: str, config: dict[str, Any], variant_index: int) -> dict[str, Any]:
    job_id = f"A-R2C-D2-{scenario_id}-FORMAL-V{variant_index}-s{FORMAL_SEED}"
    return {
        "job_id": job_id,
        "base_run_id": job_id,
        "stage": "formal",
        "mode": "formal",
        "method_id": "R2C-FL",
        "display_method_id": "A-R2C",
        "dataset_id": "D2",
        "scenario_id": scenario_id,
        "rounds": 1000,
        "method_config": config,
        "block_id": "A-R2C-D2-FORMAL",
        "seed": FORMAL_SEED,
        "partition_seed": FORMAL_SEED,
        "trace_seed": FORMAL_SEED,
        "evaluation_split": "test",
        "full_logging": True,
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
    if not WINNER_PATH.exists() or not VALIDATION_PATH.exists():
        raise RuntimeError("Formal D2 pair requires frozen winner and completed full validation")
    winner = json.loads(WINNER_PATH.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    if winner.get("selection_split") != "validation" or winner.get("test_labels_used"):
        raise RuntimeError("Winner violates validation-only selection")
    if not validation.get("formal_launch_gate_passed") or validation.get("test_labels_used"):
        raise RuntimeError("Frozen full-validation gate does not authorize formal test access")
    if int(validation.get("passed_count", 0)) < int(validation.get("required_count", 3)):
        raise RuntimeError("Insufficient frozen validation gates")

    winner_config = dict(winner["method_config"])
    if winner_config.get("r2c_protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Protocol version drift")
    formal_config = dict(winner_config)
    formal_config["r2c_v2_audit_replay"] = True
    if _training_config(formal_config) != _training_config(winner_config):
        raise RuntimeError("Formal training configuration differs from the frozen winner")
    variant_index = int(winner["variant_index"])
    jobs = [_job(scenario_id, formal_config, variant_index) for scenario_id in ("S0", "S4")]
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D2_only_single_formal_pair",
        "protocol_version": PROTOCOL_VERSION,
        "formal_seed": FORMAL_SEED,
        "formal_test_access": "one_frozen_S0_S4_pair",
        "formal_jobs": 2,
        "test_labels_for_selection": False,
        "selection_split": "validation",
        "formal_evaluation_split": "test",
        "winner_hash": config_hash(winner),
        "validation_gate_hash": config_hash(validation),
        "training_config_hash": config_hash(_training_config(formal_config)),
        "audit_only_config_change": {"r2c_v2_audit_replay": [False, True]},
        "target_accuracy": TARGET_ACCURACY,
        "success_rule": "strictly beat at least 3 of 4 prelocked D2 Table 1 baselines",
        "required_wins": 3,
        "baseline_thresholds": BASELINE_THRESHOLDS,
        "implementation_hashes": _hashes(),
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d2_formal_manifest_{stamp}.json", manifest)
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
                "test_pairs_authorized": 1,
                "test_pairs_started": 0,
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


def _run_audits(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run_dir = RUN_ROOT / str(job["actual_run_id"])
        process = subprocess.run(
            [sys.executable, str(AUDIT_SCRIPT), str(run_dir)],
            cwd=str(Path(__file__).resolve().parents[1]),
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(process.stdout.strip().splitlines()[-1])
        if report.get("status") != "passed":
            raise RuntimeError(f"Formal audit failed for {run_dir.name}: {report}")
        reports.append(report)
    atomic_json(
        AUDIT_PATH,
        {
            "schema_version": "1.0.0",
            "completed_utc": utc_now(),
            "complete": len(reports) == 2,
            "reports": reports,
        },
    )
    return reports


def _strict_better(value: float | None, threshold: dict[str, Any]) -> bool:
    if value is None:
        return False
    if threshold["direction"] == "greater":
        return float(value) > float(threshold["value"])
    return float(value) < float(threshold["value"])


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Both formal D2 jobs must complete before finalization")
    audit_reports = _run_audits(manifest)
    run_rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run_id = str(job["actual_run_id"])
        run_dir = RUN_ROOT / run_id
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        rounds = read_chunked_table(run_dir, "round_metrics").sort_values("round")
        run_manifest = pd.read_parquet(run_dir / "run_manifest.parquet")
        if len(rounds) != 1000 or rounds["round"].nunique() != 1000:
            raise RuntimeError(f"Incomplete formal round budget: {run_id}")
        if len(run_manifest) != 1 or str(run_manifest.iloc[0]["source_kind"]) != "REPRODUCED":
            raise RuntimeError(f"Formal provenance contract failed: {run_id}")
        hits = rounds.loc[rounds["test_accuracy"] >= TARGET_ACCURACY]
        first_hit_round = None if hits.empty else int(hits.iloc[0]["round"])
        tta_s = None if hits.empty else float(hits.iloc[0]["algorithm_elapsed_s"])
        auc = result["recovery"]["recovery_deficit_auc20"]
        if job["scenario_id"] == "S4":
            event_rows = rounds.loc[rounds["event_offset_round"] == 0]
            if len(event_rows) != 1:
                raise RuntimeError(f"S4 event cardinality failed: {run_id}")
            recomputed = recovery_auc20(
                rounds["round"], rounds["test_accuracy"], int(event_rows.iloc[0]["round"])
            )
            if not recomputed["recovery_auc20_complete"] or auc is None:
                raise RuntimeError(f"S4 strict AUC@20 is incomplete: {run_id}")
            if abs(float(recomputed["recovery_deficit_auc20"]) - float(auc)) > 1.0e-12:
                raise RuntimeError(f"S4 AUC@20 mismatch: {run_id}")
        run_rows.append(
            {
                "run_id": run_id,
                "dataset_id": "D2",
                "scenario_id": job["scenario_id"],
                "seed": FORMAL_SEED,
                "source_kind": "REPRODUCED",
                "rounds": 1000,
                "last50_accuracy": float(result["last50_accuracy"]),
                "recovery_deficit_auc20": None if auc is None else float(auc),
                "target_accuracy": TARGET_ACCURACY,
                "target_reached": tta_s is not None,
                "first_target_round": first_hit_round,
                "algorithm_tta_s": tta_s,
                "algorithm_elapsed_s": float(result["algorithm_elapsed_s"]),
            }
        )
    run_frame = pd.DataFrame(run_rows).sort_values("scenario_id").reset_index(drop=True)
    atomic_parquet(RUN_TABLE_PATH, run_frame)
    atomic_csv(RUN_CSV_PATH, run_frame)
    s0 = run_frame.loc[run_frame["scenario_id"] == "S0"].iloc[0]
    s4 = run_frame.loc[run_frame["scenario_id"] == "S4"].iloc[0]
    observed = {
        "s0_last50_accuracy": float(s0["last50_accuracy"]),
        "s4_last50_accuracy": float(s4["last50_accuracy"]),
        "s4_recovery_deficit_auc20": float(s4["recovery_deficit_auc20"]),
        "s4_algorithm_tta_s": None if pd.isna(s4["algorithm_tta_s"]) else float(s4["algorithm_tta_s"]),
    }
    metric_to_run = {
        "s0_last50_accuracy": str(s0["run_id"]),
        "s4_last50_accuracy": str(s4["run_id"]),
        "s4_recovery_deficit_auc20": str(s4["run_id"]),
        "s4_algorithm_tta_s": str(s4["run_id"]),
    }
    value_rows: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    for metric, threshold in BASELINE_THRESHOLDS.items():
        value = observed[metric]
        passed = _strict_better(value, threshold)
        checks[metric] = passed
        value_rows.append(
            {
                "dataset_id": "D2",
                "method_id": "A-R2C",
                "metric": metric,
                "value": value,
                "unit": threshold["unit"],
                "direction": threshold["direction"],
                "baseline_method": threshold["method"],
                "baseline_threshold": threshold["value"],
                "strictly_better": passed,
                "source_run_id": metric_to_run[metric],
            }
        )
    value_frame = pd.DataFrame(value_rows)
    atomic_parquet(VALUE_TABLE_PATH, value_frame)
    atomic_csv(VALUE_CSV_PATH, value_frame)
    wins = sum(bool(value) for value in checks.values())
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "dataset_id": "D2",
        "method_id": "A-R2C",
        "protocol_version": PROTOCOL_VERSION,
        "formal_seed": FORMAL_SEED,
        "source_kind": "REPRODUCED",
        "formal_test_pairs_used": 1,
        "test_labels_used_for_selection": False,
        "audit_complete": len(audit_reports) == 2,
        "checks": checks,
        "wins": wins,
        "required_wins": int(manifest["required_wins"]),
        "goal_met": wins >= int(manifest["required_wins"]),
        "baseline_thresholds": BASELINE_THRESHOLDS,
        "observed": observed,
        "run_ids": run_frame["run_id"].tolist(),
    }
    atomic_json(VALUE_JSON_PATH, payload)
    atomic_json(RESULT_PATH, payload)
    return payload


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
        job.update(
            attempts=int(job.get("attempts", 0)) + 1,
            actual_run_id=run_id,
            status="running",
            failure_reason=None,
        )
        state.update(
            status="running",
            current_job_id=job["job_id"],
            test_pairs_started=1,
        )
        _sync(state, manifest)
        atomic_json(MANIFEST_PATH, manifest)
        atomic_json(STATE_PATH, state)
        _event(events, job, "submitted", queue_position=position)
        job_file = QUEUE_ROOT / "active_jobs" / f"{run_id}.json"
        atomic_json(job_file, resolved)
        _event(events, job, "started", queue_position=position)
        log_path = QUEUE_ROOT / "worker_logs" / f"{run_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
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
    try:
        result = freeze_result(manifest)
    except Exception as error:
        state.update(status="finalization_failed", current_job_id=None, failure_reason=repr(error))
        _sync(state, manifest)
        atomic_json(STATE_PATH, state)
        raise
    state.update(
        status="completed_goal_met" if result["goal_met"] else "completed_goal_not_met",
        current_job_id=None,
        goal_met=bool(result["goal_met"]),
        wins=int(result["wins"]),
        required_wins=int(result["required_wins"]),
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
