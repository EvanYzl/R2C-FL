from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import FORMAL_SEED, PLOT_ROOT, PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .logging_io import read_chunked_table
from .r2c_v4 import PROTOCOL_VERSION
from .utils import atomic_csv, atomic_json, atomic_parquet, config_hash, sha256_file, utc_now


ROUNDS = 1000
TARGET_ACCURACY = 0.7986707616707616
FORMAL_INTERPRETATION = "matched_seed_engineering_reevaluation_not_untouched_confirmation"
SCENARIOS = ("S0", "S4")
THRESHOLDS = {
    "s0_last50_accuracy": 0.8902665949600491,
    "s4_last50_accuracy": 0.8913398893669331,
    "s4_recovery_deficit_auc20": 0.0000877765826674815,
    "s4_algorithm_tta_s": 202.493191700109,
}
CLOSE_LIMITS = {
    "accuracy_fraction": 0.0015,
    "auc_fraction": 0.0001,
    "tta_multiplier": 1.05,
}

FULLVAL_RESULT_PATH = QUEUE_ROOT / "r2c_d3_v5_fullval_result.json"
FULLVAL_MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v5_fullval_manifest.json"
BASELINE_TABLE_PATH = PLOT_ROOT / "table1_v4_matched_combined_values_20260816_140305.csv"
PLAN_PATH = PROJECT_ROOT / "refine-logs" / "D3_V5_OPTIMIZATION_PLAN_20260816_221904.md"
MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v5_formal_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v5_formal_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v5_formal_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v5_formal_result.json"
RUNS_PATH = PLOT_ROOT / "r2c_d3_v5_formal_runs.parquet"
RUNS_CSV_PATH = PLOT_ROOT / "r2c_d3_v5_formal_runs.csv"


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        "config.py",
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


def _source_hashes() -> dict[str, str]:
    paths = (FULLVAL_RESULT_PATH, FULLVAL_MANIFEST_PATH, BASELINE_TABLE_PATH, PLAN_PATH)
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _verify_frozen_thresholds() -> dict[str, Any]:
    table = pd.read_csv(BASELINE_TABLE_PATH)
    d3 = table.loc[(table["dataset_id"] == "D3") & (table["method_id"] != "R2C-FL")].copy()
    mapping = {
        "s0_last50_accuracy": ("s0_last50_accuracy_pct", "max", 0.01),
        "s4_last50_accuracy": ("s4_last50_accuracy_pct", "max", 0.01),
        "s4_recovery_deficit_auc20": ("s4_recovery_deficit_auc20_pp", "min", 0.01),
        "s4_algorithm_tta_s": ("s4_tta_h", "min", 3600.0),
    }
    lineage: dict[str, Any] = {}
    for key, (metric, direction, multiplier) in mapping.items():
        rows = d3.loc[d3["metric"] == metric].copy()
        rows["value"] = rows["value"].astype(float)
        row = rows.loc[rows["value"].idxmax() if direction == "max" else rows["value"].idxmin()]
        value = float(row["value"]) * multiplier
        if abs(value - THRESHOLDS[key]) > 1.0e-12:
            raise RuntimeError(f"Frozen external threshold drift for {key}: {value}")
        lineage[key] = {
            "method_id": str(row["method_id"]),
            "source_metric": metric,
            "threshold": value,
            "source_run_ids": str(row["source_run_ids"]),
        }
    return lineage


def _load_winner() -> dict[str, Any]:
    if not FULLVAL_RESULT_PATH.exists():
        raise RuntimeError("Formal D3 queue requires the frozen Phase B result")
    result = json.loads(FULLVAL_RESULT_PATH.read_text(encoding="utf-8"))
    if result.get("selection_split") != "validation" or result.get("test_labels_used"):
        raise RuntimeError("Phase B winner was not selected on validation only")
    if not result.get("formal_authorized") or not result.get("winner"):
        raise RuntimeError("Phase B did not authorize formal D3 execution")
    winner = dict(result["winner"])
    config = dict(winner["method_config"])
    if config.get("r2c_protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Phase B winner protocol drift")
    if config_hash(config) != winner.get("config_hash"):
        raise RuntimeError("Phase B winner config hash drift")
    return winner


def _formal_config(winner: dict[str, Any]) -> dict[str, Any]:
    config = dict(winner["method_config"])
    # Audit replay is instrumentation-only and excluded from algorithm elapsed time.
    config["r2c_v2_audit_replay"] = True
    return config


def _job(winner: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    alpha = float(winner["alpha"])
    beta = float(winner["beta"])
    label = f"A{int(round(alpha * 1000)):04d}-B{int(round(beta * 1000)):04d}"
    run_id = f"A-R2C-D3-{scenario_id}-V5FORMAL-{label}-s{FORMAL_SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v5_matched_seed_formal",
        "mode": "formal",
        "method_id": "R2C-FL",
        "dataset_id": "D3",
        "scenario_id": scenario_id,
        "rounds": ROUNDS,
        "method_config": _formal_config(winner),
        "block_id": "A-R2C-D3-V5-FORMAL",
        "seed": FORMAL_SEED,
        "partition_seed": FORMAL_SEED,
        "trace_seed": FORMAL_SEED,
        "evaluation_split": "test",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "variant_label": label,
        "formal_interpretation": FORMAL_INTERPRETATION,
        "test_labels_used_for_selection": False,
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def _build_jobs(winner: dict[str, Any]) -> list[dict[str, Any]]:
    return [_job(winner, scenario) for scenario in SCENARIOS]


def build_manifest(persist: bool = True, force: bool = False) -> dict[str, Any]:
    if persist and MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if force and MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if any(job.get("status") != "pending" or int(job.get("attempts", 0)) for job in existing["jobs"]):
            raise RuntimeError("Refusing to rebuild a started D3 v5 formal manifest")
    if int(FORMAL_SEED) != 20260811:
        raise RuntimeError(f"Expected formal seed 20260811, found {FORMAL_SEED}")
    winner = _load_winner()
    threshold_lineage = _verify_frozen_thresholds()
    jobs = _build_jobs(winner)
    if len(jobs) != 2 or len({job["job_id"] for job in jobs}) != 2:
        raise AssertionError("D3 v5 formal queue must contain exactly one S0/S4 pair")
    formal_config = _formal_config(winner)
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_only_v5_single_matched_seed_formal_pair",
        "protocol_version": PROTOCOL_VERSION,
        "formal_test_access": True,
        "formal_interpretation": FORMAL_INTERPRETATION,
        "test_labels_used_for_selection": False,
        "formal_seed": FORMAL_SEED,
        "rounds_per_job": ROUNDS,
        "scenario_order": list(SCENARIOS),
        "job_order": [job["job_id"] for job in jobs],
        "validation_winner": winner,
        "validation_winner_config_hash": winner["config_hash"],
        "formal_config_hash": config_hash(formal_config),
        "audit_only_config_change": {"r2c_v2_audit_replay": [False, True]},
        "thresholds": THRESHOLDS,
        "close_limits": CLOSE_LIMITS,
        "threshold_lineage": threshold_lineage,
        "stopping_rule": (
            "four strict wins, or exactly three strict wins and the sole miss is within its frozen close limit"
        ),
        "implementation_hashes": _implementation_hashes(),
        "source_hashes": _source_hashes(),
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d3_v5_formal_manifest_{stamp}.json", manifest)
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


def _successful_run_id(job: dict[str, Any]) -> str | None:
    candidates: list[Path] = []
    if job.get("actual_run_id"):
        candidates.append(RUN_ROOT / str(job["actual_run_id"]))
    candidates.append(RUN_ROOT / str(job["base_run_id"]))
    candidates.extend(sorted(RUN_ROOT.glob(f"{job['base_run_id']}-a*")))
    seen: set[str] = set()
    for path in candidates:
        if path.name in seen:
            continue
        seen.add(path.name)
        if (path / "_SUCCESS.json").exists() and (path / "result.json").exists():
            return path.name
    return None


def _reconcile_successes(manifest: dict[str, Any], events: list[dict[str, Any]]) -> int:
    changed = 0
    for job in manifest["jobs"]:
        run_id = _successful_run_id(job)
        if run_id is not None and (job.get("status") != "completed" or job.get("actual_run_id") != run_id):
            job.update({"status": "completed", "actual_run_id": run_id, "failure_reason": None})
            _event(events, job, "reconciled_completed", reason="existing_success_output")
            changed += 1
    return changed


def _actual_run_id(base: str) -> tuple[str, str | None]:
    if not (RUN_ROOT / base).exists():
        return base, None
    attempt = 2
    while (RUN_ROOT / f"{base}-a{attempt}").exists():
        attempt += 1
    return f"{base}-a{attempt}", base


def _resolved(job: dict[str, Any]) -> dict[str, Any]:
    value = dict(job)
    run_id, retry_of = _actual_run_id(str(job["base_run_id"]))
    value.update({"run_id": run_id, "retry_of_run_id": retry_of, "queue_utc": utc_now()})
    return value


def _assert_frozen_manifest(manifest: dict[str, Any]) -> None:
    if not manifest.get("formal_test_access") or manifest.get("test_labels_used_for_selection"):
        raise RuntimeError("D3 formal access contract drift")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("Implementation hashes changed after D3 v5 formal freeze")
    if manifest.get("source_hashes") != _source_hashes():
        raise RuntimeError("Source hashes changed after D3 v5 formal freeze")
    jobs = manifest.get("jobs", [])
    if len(jobs) != 2 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("D3 v5 formal matrix drift")
    for job, scenario in zip(jobs, SCENARIOS):
        if job.get("scenario_id") != scenario or job.get("evaluation_split") != "test":
            raise RuntimeError(f"Scenario/test split drift in {job['job_id']}")
        if any(int(job[key]) != FORMAL_SEED for key in ("seed", "partition_seed", "trace_seed")):
            raise RuntimeError(f"Formal seed drift in {job['job_id']}")
        if not job.get("full_logging") or not job["method_config"].get("r2c_v2_audit_replay"):
            raise RuntimeError(f"Formal audit logging disabled in {job['job_id']}")


def _time_to_accuracy(run_dir: Path, target: float) -> tuple[int | None, float | None]:
    rounds = read_chunked_table(run_dir, "round_metrics")
    reached = rounds.loc[rounds["test_accuracy"].astype(float) >= float(target)]
    if reached.empty:
        return None, None
    first = reached.sort_values("round").iloc[0]
    return int(first["round"]), float(first["algorithm_elapsed_s"])


def _evaluate_termination(observed: dict[str, float | None]) -> dict[str, Any]:
    tta = observed["s4_algorithm_tta_s"]
    checks = {
        "s0_last50_accuracy": float(observed["s0_last50_accuracy"]) > THRESHOLDS["s0_last50_accuracy"],
        "s4_last50_accuracy": float(observed["s4_last50_accuracy"]) > THRESHOLDS["s4_last50_accuracy"],
        "s4_recovery_deficit_auc20": float(observed["s4_recovery_deficit_auc20"])
        < THRESHOLDS["s4_recovery_deficit_auc20"],
        "s4_algorithm_tta_s": tta is not None and float(tta) < THRESHOLDS["s4_algorithm_tta_s"],
    }
    close = {
        "s0_last50_accuracy": float(observed["s0_last50_accuracy"])
        >= THRESHOLDS["s0_last50_accuracy"] - CLOSE_LIMITS["accuracy_fraction"],
        "s4_last50_accuracy": float(observed["s4_last50_accuracy"])
        >= THRESHOLDS["s4_last50_accuracy"] - CLOSE_LIMITS["accuracy_fraction"],
        "s4_recovery_deficit_auc20": float(observed["s4_recovery_deficit_auc20"])
        <= THRESHOLDS["s4_recovery_deficit_auc20"] + CLOSE_LIMITS["auc_fraction"],
        "s4_algorithm_tta_s": tta is not None
        and float(tta) <= THRESHOLDS["s4_algorithm_tta_s"] * CLOSE_LIMITS["tta_multiplier"],
    }
    misses = [name for name, passed in checks.items() if not passed]
    goal_met = not misses or (len(misses) == 1 and close[misses[0]])
    return {
        "strict_checks": checks,
        "close_checks": close,
        "strict_passes": sum(checks.values()),
        "sole_miss": misses[0] if len(misses) == 1 else None,
        "goal_met": bool(goal_met),
    }


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot freeze incomplete D3 v5 formal pair")
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run_id = str(job["actual_run_id"])
        run_dir = RUN_ROOT / run_id
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        run_manifest = pd.read_parquet(run_dir / "run_manifest.parquet").iloc[0]
        if str(run_manifest["source_kind"]) != "REPRODUCED" or str(run_manifest["status"]) != "completed":
            raise RuntimeError(f"Formal provenance/status failure for {run_id}")
        for key in ("seed", "partition_seed", "trace_seed"):
            if int(run_manifest[key]) != FORMAL_SEED:
                raise RuntimeError(f"{key} mismatch for {run_id}")
        if str(run_manifest["upstream_commit"]) != PROTOCOL_VERSION:
            raise RuntimeError(f"Protocol lineage mismatch for {run_id}")
        expected_rows = {
            "round_metrics": ROUNDS,
            "client_round_metrics": ROUNDS * 100,
            "checkpoint_metrics": ROUNDS * 20,
            "certificate_audit": ROUNDS,
            "deployment_candidate_metrics": ROUNDS,
        }
        for table, expected in expected_rows.items():
            if int(result["table_indices"][table]["rows"]) != expected:
                raise RuntimeError(f"{run_id} {table} row-count failure")
        recovery = result["recovery"]
        if job["scenario_id"] == "S4" and (
            not recovery["recovery_auc20_complete"] or recovery["recovery_deficit_auc20"] is None
        ):
            raise RuntimeError(f"Strict AUC@20 window incomplete for {run_id}")
        tta_round, tta_s = _time_to_accuracy(run_dir, TARGET_ACCURACY)
        rows.append(
            {
                "run_id": run_id,
                "dataset_id": "D3",
                "scenario_id": job["scenario_id"],
                "seed": FORMAL_SEED,
                "last50_test_accuracy": float(result["last50_accuracy"]),
                "recovery_deficit_auc20": recovery["recovery_deficit_auc20"],
                "recovery_auc20_complete": bool(recovery["recovery_auc20_complete"]),
                "algorithm_elapsed_s": float(result["algorithm_elapsed_s"]),
                "tta_round": tta_round,
                "algorithm_tta_s": tta_s,
                "source_kind": "REPRODUCED",
                "protocol_version": PROTOCOL_VERSION,
                "formal_interpretation": FORMAL_INTERPRETATION,
                "test_labels_used_for_selection": False,
            }
        )
    frame = pd.DataFrame(rows)
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    s0 = frame.loc[frame["scenario_id"] == "S0"].iloc[0]
    s4 = frame.loc[frame["scenario_id"] == "S4"].iloc[0]
    observed: dict[str, float | None] = {
        "s0_last50_accuracy": float(s0["last50_test_accuracy"]),
        "s4_last50_accuracy": float(s4["last50_test_accuracy"]),
        "s4_recovery_deficit_auc20": float(s4["recovery_deficit_auc20"]),
        "s4_algorithm_tta_s": None if pd.isna(s4["algorithm_tta_s"]) else float(s4["algorithm_tta_s"]),
    }
    termination = _evaluate_termination(observed)
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "dataset_id": "D3",
        "formal_seed": FORMAL_SEED,
        "formal_interpretation": FORMAL_INTERPRETATION,
        "source_kind": "REPRODUCED",
        "selection_split": "validation",
        "test_labels_used_for_selection": False,
        "thresholds": THRESHOLDS,
        "close_limits": CLOSE_LIMITS,
        "observed": observed,
        **termination,
        "run_ids": frame["run_id"].tolist(),
        "manifest_hash": config_hash(manifest),
    }
    atomic_json(RESULT_PATH, payload)
    return payload


def worker() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        build_manifest()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    events = _events()
    _assert_frozen_manifest(manifest)
    if _reconcile_successes(manifest, events):
        _sync(state, manifest)
        atomic_json(MANIFEST_PATH, manifest)
        atomic_json(STATE_PATH, state)
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
            "status": "formal_completed_goal_met" if result["goal_met"] else "formal_completed_goal_not_met",
            "current_job_id": None,
            "goal_met": bool(result["goal_met"]),
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

