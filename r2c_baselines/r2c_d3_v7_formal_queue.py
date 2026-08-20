from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import FORMAL_SEED, PLOT_ROOT, PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .data import partition_asset_path, partition_meta_path, prepare_partition
from .metrics import recovery_auc20
from .r2c_d3_v7_phase_e_queue import _audit_run, _verified_chunked_table
from .r2c_v6 import PROTOCOL_VERSION
from .traces import prepare_trace, trace_asset_path, trace_meta_path
from .utils import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    config_hash,
    sha256_file,
    utc_now,
)


DATASET_ID = "D3"
SCENARIOS = ("S0", "S4")
ROUNDS = 1000
EVENT_ROUND = 500
TARGET_ACCURACY = 0.7986707616707616
MAX_ATTEMPTS = 3
ACCURACY_CLOSE = 0.0015
AUC_CLOSE = 0.0001
TTA_CLOSE_MULTIPLIER = 1.05

FORMAL_THRESHOLDS = {
    "s0_last50_accuracy": 0.8902665949600491,
    "s4_last50_accuracy": 0.8913398893669331,
    "s4_recovery_deficit_auc20": 0.0000877765826674815,
    "s4_algorithm_tta_s": 202.493191700109,
}

PLAN_PATH = (
    PROJECT_ROOT
    / "refine-logs"
    / "D3_V7_TELEMETRY_SYNC_PLAN_AMENDMENT_20260817_164238.md"
)
EXPECTED_PLAN_SHA256 = (
    "7BFE71BFD29FE38EFD72DF25E3DA90BC8FD2FCD18EE8ECC5131BDBCD314F07EC"
)
PHASE_F_RESULT_PATH = QUEUE_ROOT / "r2c_d3_v7_phase_f_result.json"
PHASE_F_MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v7_phase_f_manifest.json"
PHASE_F_RUNS_PATH = PLOT_ROOT / "r2c_d3_v7_phase_f_runs.parquet"

MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v7_formal_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v7_formal_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v7_formal_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v7_formal_result.json"
RUNS_PATH = PLOT_ROOT / "r2c_d3_v7_formal_runs.parquet"
RUNS_CSV_PATH = PLOT_ROOT / "r2c_d3_v7_formal_runs.csv"
AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / "tests" / "audit_r2c_run.py"


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
        "r2c_v5.py",
        "r2c_v6.py",
        "r2c_d3_v7_phase_e_queue.py",
        "r2c_d3_v7_phase_f_queue.py",
        "run.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    values["tests/audit_r2c_run.py"] = sha256_file(AUDIT_SCRIPT)
    return values


def _verify_phase_f_result() -> dict[str, Any]:
    if not PHASE_F_RESULT_PATH.exists():
        raise RuntimeError("D3 v7 Phase F result does not exist")
    result = json.loads(PHASE_F_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        not bool(result.get("formal_authorized"))
        or result.get("selection_split") != "validation"
        or bool(result.get("formal_test_access"))
        or bool(result.get("test_labels_used"))
        or not bool(result.get("metric_authorized"))
        or not bool(result.get("diversity_preserved"))
        or not all(bool(value) for value in result.get("detector_checks", {}).values())
    ):
        raise RuntimeError("D3 v7 Phase F did not authorize formal reevaluation")
    state = json.loads(
        (QUEUE_ROOT / "r2c_d3_v7_phase_f_queue_state.json").read_text(encoding="utf-8")
    )
    if state.get("status") != "phase_f_completed_formal_authorized":
        raise RuntimeError("D3 v7 Phase F terminal state does not authorize formal reevaluation")
    return result


def _source_hashes() -> dict[str, str]:
    paths = (PLAN_PATH, PHASE_F_RESULT_PATH, PHASE_F_MANIFEST_PATH, PHASE_F_RUNS_PATH)
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _ensure_assets() -> dict[str, str]:
    prepare_partition(DATASET_ID, FORMAL_SEED)
    paths: list[Path] = [
        partition_asset_path(DATASET_ID, FORMAL_SEED),
        partition_meta_path(DATASET_ID, FORMAL_SEED),
    ]
    for scenario in SCENARIOS:
        prepare_trace(DATASET_ID, scenario, FORMAL_SEED, rounds=ROUNDS)
        paths.extend(
            [
                trace_asset_path(DATASET_ID, scenario, FORMAL_SEED, ROUNDS),
                trace_meta_path(DATASET_ID, scenario, FORMAL_SEED, ROUNDS),
            ]
        )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _formal_config(phase_f_result: dict[str, Any]) -> dict[str, Any]:
    config = dict(phase_f_result["method_config"])
    if (
        config.get("r2c_protocol_version") != PROTOCOL_VERSION
        or bool(config.get("r2c_v2_audit_replay"))
    ):
        raise RuntimeError("Frozen v7 Phase F winner config drift")
    config["r2c_v2_audit_replay"] = True
    return config


def _job(scenario: str, config: dict[str, Any]) -> dict[str, Any]:
    run_id = f"A-R2C-D3-{scenario}-V7FORMAL-SYNC-s{FORMAL_SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v7_matched_seed_formal_engineering_reevaluation",
        "mode": "formal",
        "method_id": "R2C-FL",
        "dataset_id": DATASET_ID,
        "scenario_id": scenario,
        "rounds": ROUNDS,
        "method_config": dict(config),
        "block_id": "A-R2C-D3-V7-FORMAL",
        "seed": FORMAL_SEED,
        "partition_seed": FORMAL_SEED,
        "trace_seed": FORMAL_SEED,
        "evaluation_split": "test",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "formal_interpretation": "matched_seed_engineering_reevaluation_not_untouched_confirmation",
        "test_labels_used_for_selection": False,
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def build_manifest(persist: bool = True, force: bool = False) -> dict[str, Any]:
    if persist and MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if force and MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if any(job.get("status") != "pending" or int(job.get("attempts", 0)) for job in existing["jobs"]):
            raise RuntimeError("Refusing to rebuild a started D3 v7 formal manifest")
    if int(FORMAL_SEED) != 20260811:
        raise RuntimeError(f"Expected matched formal seed 20260811, found {FORMAL_SEED}")
    if sha256_file(PLAN_PATH) != EXPECTED_PLAN_SHA256.lower():
        raise RuntimeError("Frozen v7 plan hash drift")
    phase_f_result = _verify_phase_f_result()
    config = _formal_config(phase_f_result)
    jobs = [_job(scenario, config) for scenario in SCENARIOS]
    validation_config = dict(phase_f_result["method_config"])
    changed_keys = sorted(key for key in set(config) | set(validation_config) if config.get(key) != validation_config.get(key))
    if changed_keys != ["r2c_v2_audit_replay"]:
        raise RuntimeError(f"Unexpected validation-to-formal config changes: {changed_keys}")
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_only_v7_matched_seed_formal_engineering_reevaluation",
        "formal_test_access": True,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "engineering_reevaluation": True,
        "seed": FORMAL_SEED,
        "rounds_per_job": ROUNDS,
        "scenario_order": list(SCENARIOS),
        "job_order": [job["job_id"] for job in jobs],
        "validation_to_formal_changed_keys": changed_keys,
        "formal_thresholds": FORMAL_THRESHOLDS,
        "close_limits": {
            "accuracy_fraction": ACCURACY_CLOSE,
            "auc_fraction": AUC_CLOSE,
            "tta_multiplier": TTA_CLOSE_MULTIPLIER,
        },
        "clean_gpu_launch_gate": {
            "samples": 2,
            "utilization_strictly_below_percent": 10,
            "minimum_separation_seconds": 30,
        },
        "implementation_hashes": _implementation_hashes(),
        "source_hashes": _source_hashes(),
        "asset_hashes": _ensure_assets(),
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d3_v7_formal_manifest_{stamp}.json", manifest)
        atomic_json(MANIFEST_PATH, manifest)
        atomic_json(
            STATE_PATH,
            {
                "status": "ready_waiting_clean_gpu_gate",
                "created_utc": utc_now(),
                "updated_utc": utc_now(),
                "current_job_id": None,
                "completed": 0,
                "failed": 0,
                "total": 2,
                "goal_met": False,
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
    if not manifest.get("formal_test_access") or manifest.get("other_dataset_access"):
        raise RuntimeError("D3 v7 formal access contract drift")
    if manifest.get("test_labels_used_for_selection") or not manifest.get("engineering_reevaluation"):
        raise RuntimeError("D3 v7 formal interpretation drift")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("Implementation hashes changed after D3 v7 formal freeze")
    if manifest.get("source_hashes") != _source_hashes():
        raise RuntimeError("Source hashes changed after D3 v7 formal freeze")
    if manifest.get("asset_hashes") != _ensure_assets():
        raise RuntimeError("Asset hashes changed after D3 v7 formal freeze")
    phase_f_result = _verify_phase_f_result()
    config = _formal_config(phase_f_result)
    jobs = manifest.get("jobs", [])
    if len(jobs) != 2 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("D3 v7 formal matrix drift")
    for job, scenario in zip(jobs, SCENARIOS):
        if (
            job.get("scenario_id") != scenario
            or job.get("evaluation_split") != "test"
            or job.get("mode") != "formal"
            or int(job.get("rounds", -1)) != ROUNDS
            or any(int(job[key]) != FORMAL_SEED for key in ("seed", "partition_seed", "trace_seed"))
            or job.get("method_config") != config
            or not bool(job.get("full_logging"))
        ):
            raise RuntimeError(f"D3 v7 formal protocol drift in {job['job_id']}")


def _time_to_accuracy(rounds: pd.DataFrame) -> tuple[int | None, float | None]:
    reached = rounds.loc[rounds["test_accuracy"].astype(float) >= TARGET_ACCURACY]
    if reached.empty:
        return None, None
    first = reached.sort_values("round").iloc[0]
    return int(first["round"]), float(first["algorithm_elapsed_s"])


def _run_metrics(run_id: str, scenario: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_id
    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    run_manifest = pd.read_parquet(run_dir / "run_manifest.parquet").iloc[0]
    rounds = _verified_chunked_table(run_dir, "round_metrics").sort_values("round")
    if rounds["round"].astype(int).tolist() != list(range(1, ROUNDS + 1)):
        raise RuntimeError(f"Incomplete v7 formal trajectory: {run_id}")
    if (
        str(run_manifest["source_kind"]) != "REPRODUCED"
        or job.get("evaluation_split") != "test"
        or job.get("mode") != "formal"
        or any(int(job[key]) != FORMAL_SEED for key in ("seed", "partition_seed", "trace_seed"))
    ):
        raise RuntimeError(f"V7 formal source/seed/split mismatch: {run_id}")
    triggered = rounds.loc[rounds["telemetry_shift_trigger"].astype(bool)]
    forbidden_input_clean = bool(
        not rounds["telemetry_shift_labels_used"].astype(bool).any()
        and not rounds["telemetry_shift_scenario_metadata_used"].astype(bool).any()
    )
    auc: float | None = None
    if scenario == "S4":
        events = rounds.loc[rounds["event_offset_round"].astype(int) == 0]
        if len(events) != 1 or int(events.iloc[0]["round"]) != EVENT_ROUND:
            raise RuntimeError(f"V7 formal event drift: {run_id}")
        direct = recovery_auc20(
            rounds["round"].astype(int).tolist(),
            rounds["test_accuracy"].astype(float).tolist(),
            EVENT_ROUND,
        )
        if not direct["recovery_auc20_complete"]:
            raise RuntimeError(f"V7 formal lacks strict AUC@20: {run_id}")
        auc = float(direct["recovery_deficit_auc20"])
        stored = result["recovery"]["recovery_deficit_auc20"]
        if stored is None or abs(float(stored) - auc) > 1.0e-15:
            raise RuntimeError(f"V7 formal stored/direct AUC mismatch: {run_id}")
    direct_last50 = float(rounds.tail(50)["test_accuracy"].astype(float).mean())
    if abs(float(result["last50_accuracy"]) - direct_last50) > 1.0e-15:
        raise RuntimeError(f"V7 formal Last50 mismatch: {run_id}")
    tta_round, tta_s = _time_to_accuracy(rounds)
    return {
        "run_id": run_id,
        "scenario_id": scenario,
        "last50_accuracy": direct_last50,
        "recovery_deficit_auc20": auc,
        "tta_round": tta_round,
        "algorithm_tta_s": tta_s,
        "algorithm_elapsed_s": float(result["algorithm_elapsed_s"]),
        "trigger_count": len(triggered),
        "trigger_rounds_json": json.dumps(triggered["round"].astype(int).tolist(), separators=(",", ":")),
        "forbidden_input_clean": forbidden_input_clean,
        "source_kind": str(run_manifest["source_kind"]),
        "formal_interpretation": job["formal_interpretation"],
        "test_labels_used_for_selection": False,
    }


def _metric_rule(observed: dict[str, float | None]) -> dict[str, Any]:
    tta = observed["s4_algorithm_tta_s"]
    strict = {
        "s0_last50_accuracy": float(observed["s0_last50_accuracy"])
        > FORMAL_THRESHOLDS["s0_last50_accuracy"],
        "s4_last50_accuracy": float(observed["s4_last50_accuracy"])
        > FORMAL_THRESHOLDS["s4_last50_accuracy"],
        "s4_recovery_deficit_auc20": float(observed["s4_recovery_deficit_auc20"])
        < FORMAL_THRESHOLDS["s4_recovery_deficit_auc20"],
        "s4_algorithm_tta_s": tta is not None
        and float(tta) < FORMAL_THRESHOLDS["s4_algorithm_tta_s"],
    }
    close = {
        "s0_last50_accuracy": float(observed["s0_last50_accuracy"])
        >= FORMAL_THRESHOLDS["s0_last50_accuracy"] - ACCURACY_CLOSE,
        "s4_last50_accuracy": float(observed["s4_last50_accuracy"])
        >= FORMAL_THRESHOLDS["s4_last50_accuracy"] - ACCURACY_CLOSE,
        "s4_recovery_deficit_auc20": float(observed["s4_recovery_deficit_auc20"])
        <= FORMAL_THRESHOLDS["s4_recovery_deficit_auc20"] + AUC_CLOSE,
        "s4_algorithm_tta_s": tta is not None
        and float(tta) <= FORMAL_THRESHOLDS["s4_algorithm_tta_s"] * TTA_CLOSE_MULTIPLIER,
    }
    passes = sum(strict.values())
    misses = [key for key, value in strict.items() if not value]
    sole_miss = misses[0] if len(misses) == 1 else None
    goal = bool(passes == 4 or (passes == 3 and sole_miss is not None and close[sole_miss]))
    return {
        "strict_checks": strict,
        "close_checks": close,
        "strict_passes": passes,
        "sole_miss": sole_miss,
        "metric_goal_met": goal,
    }


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot freeze incomplete D3 v7 formal reevaluation")
    rows = [
        _run_metrics(str(job["actual_run_id"]), str(job["scenario_id"]))
        for job in manifest["jobs"]
    ]
    frame = pd.DataFrame(rows)
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    s0 = frame.loc[frame["scenario_id"] == "S0"].iloc[0]
    s4 = frame.loc[frame["scenario_id"] == "S4"].iloc[0]
    observed: dict[str, float | None] = {
        "s0_last50_accuracy": float(s0["last50_accuracy"]),
        "s4_last50_accuracy": float(s4["last50_accuracy"]),
        "s4_recovery_deficit_auc20": float(s4["recovery_deficit_auc20"]),
        "s4_algorithm_tta_s": None if pd.isna(s4["algorithm_tta_s"]) else float(s4["algorithm_tta_s"]),
    }
    metric = _metric_rule(observed)
    detector_checks = {
        "s0_zero_triggers": int(s0["trigger_count"]) == 0,
        "s4_exactly_one_trigger": int(s4["trigger_count"]) == 1,
        "s4_trigger_at_event": str(s4["trigger_rounds_json"]) == "[500]",
        "forbidden_input_clean": bool(s0["forbidden_input_clean"] and s4["forbidden_input_clean"]),
    }
    goal_met = bool(metric["metric_goal_met"] and all(detector_checks.values()))
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "dataset_id": DATASET_ID,
        "source_kind": "REPRODUCED",
        "evaluation_split": "test",
        "engineering_reevaluation": True,
        "test_labels_used_for_selection": False,
        "observed": observed,
        **metric,
        "detector_checks": detector_checks,
        "goal_met": goal_met,
        "run_ids": frame["run_id"].astype(str).tolist(),
        "runs_path": str(RUNS_PATH),
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
        if int(job.get("attempts", 0)) >= MAX_ATTEMPTS:
            state.update({"status": "failed_max_attempts", "current_job_id": job["job_id"]})
            _sync(state, manifest)
            atomic_json(STATE_PATH, state)
            return state
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
        audit_log = QUEUE_ROOT / "worker_logs" / f"{resolved['run_id']}.audit.log"
        audit = _audit_run(str(resolved["run_id"]), audit_log) if success else None
        if process.returncode == 0 and success and audit is not None and audit.returncode == 0:
            job["status"] = "completed"
            _event(events, job, "completed", exit_code=0, audit_log_path=str(audit_log))
        else:
            audit_code = None if audit is None else audit.returncode
            job["status"] = "failed"
            job["failure_reason"] = (
                f"train_exit={process.returncode};audit_exit={audit_code};"
                f"log={log_path};audit_log={audit_log}"
            )
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
            "result_path": str(RESULT_PATH),
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
