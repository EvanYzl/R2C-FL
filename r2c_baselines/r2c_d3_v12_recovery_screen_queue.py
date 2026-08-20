from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .data import partition_asset_path, partition_meta_path, prepare_partition
from .logging_io import read_chunked_table
from .r2c_d3_v7_phase_e_queue import _audit_run
from .r2c_tail_recovery_margin20 import derive_window_metrics
from .r2c_v7 import PROTOCOL_VERSION as V7_PROTOCOL_VERSION
from .r2c_v8 import PROTOCOL_VERSION
from .traces import prepare_trace, trace_asset_path, trace_meta_path
from .utils import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    config_hash,
    sha256_file,
    utc_now,
)


PLAN_ID = "R2C_V12_TAIL_RECOVERY_20260819_131758"
PLAN_PATH = (
    PROJECT_ROOT
    / "refine-logs"
    / "R2C_V12_TAIL_RECOVERY_OPTIMIZATION_PLAN_20260819_131758.md"
)
SOURCE_RUN_ID = "A-R2C-D3-S4-V9J-HOLD-s20260810"
SOURCE_RUN_DIR = RUN_ROOT / SOURCE_RUN_ID

MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v12_recovery_screen_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v12_recovery_screen_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v12_recovery_screen_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v12_recovery_screen_result.json"
RUNS_PATH = QUEUE_ROOT / "r2c_d3_v12_recovery_screen_runs.parquet"
RUNS_CSV_PATH = QUEUE_ROOT / "r2c_d3_v12_recovery_screen_runs.csv"
SMOKE_RESULT_PATH = QUEUE_ROOT / "r2c_d3_v12_recovery_smoke_result.json"

DATASET_ID = "D3"
SCENARIO_ID = "S4"
SEED = 20260810
ROUNDS = 1000
EVENT_ROUND = 500
TARGET_ACCURACY = 0.7986707616707616
ORDINARY_BETA = 0.9
TRIGGER_BETA = 1.0
MAX_ATTEMPTS = 3

# Frozen before the first v12 candidate run.  This is a compact 2x2 screen of
# strength and duration, not an adaptive or open-ended sweep.
CANDIDATES: tuple[tuple[str, float, int], ...] = (
    ("P1-B000-D05", 0.0, 5),
    ("P2-B000-D20", 0.0, 20),
    ("P3-B050-D05", 0.5, 5),
    ("P4-B050-D20", 0.5, 20),
)

BASELINE_ENVELOPE = {
    "s0_last50_accuracy": 0.8947065247065247,
    "s4_last50_accuracy": 0.8933715533715534,
    "s4_trm20_pp": -0.092137592137609,
    "s4_algorithm_tta_s": 251.75663530116435,
}


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        "config.py",
        "data.py",
        "training.py",
        "traces.py",
        "r2c.py",
        "r2c_v2.py",
        "r2c_v3.py",
        "r2c_v4.py",
        "r2c_v5.py",
        "r2c_v6.py",
        "r2c_v7.py",
        "r2c_v8.py",
        "run.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    values["tests/audit_r2c_run.py"] = sha256_file(
        Path(__file__).resolve().parents[1] / "tests" / "audit_r2c_run.py"
    )
    values["tests/test_r2c_v8.py"] = sha256_file(
        Path(__file__).resolve().parents[1] / "tests" / "test_r2c_v8.py"
    )
    return values


def _source_lineage() -> dict[str, str]:
    paths = (
        PLAN_PATH,
        SOURCE_RUN_DIR / "job.json",
        SOURCE_RUN_DIR / "result.json",
        SOURCE_RUN_DIR / "_SUCCESS.json",
        SOURCE_RUN_DIR / "run_manifest.parquet",
        SOURCE_RUN_DIR / "tables" / "round_metrics" / "_index.json",
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in paths
    }


def _source_config() -> dict[str, Any]:
    job = json.loads((SOURCE_RUN_DIR / "job.json").read_text(encoding="utf-8"))
    if (
        job.get("dataset_id") != DATASET_ID
        or job.get("scenario_id") != SCENARIO_ID
        or job.get("evaluation_split") != "validation"
        or int(job.get("rounds", -1)) != ROUNDS
        or any(int(job.get(key, -1)) != SEED for key in ("seed", "partition_seed", "trace_seed"))
    ):
        raise RuntimeError("v9 validation source scope drift")
    config = dict(job["method_config"])
    if (
        config.get("r2c_protocol_version") != V7_PROTOCOL_VERSION
        or config.get("r2c_v4_deployment_ema_betas") != [ORDINARY_BETA]
        or float(config.get("r2c_v4_primary_deployment_beta", -1.0)) != ORDINARY_BETA
        or float(config.get("r2c_v7_trigger_deployment_beta", -1.0)) != TRIGGER_BETA
    ):
        raise RuntimeError("v9 validation source configuration drift")
    return config


def _candidate_config(pulse_beta: float, pulse_rounds: int) -> dict[str, Any]:
    config = _source_config()
    config.pop("r2c_v7_trigger_deployment_beta", None)
    config.update(
        {
            "r2c_protocol_version": PROTOCOL_VERSION,
            "r2c_v4_deployment_ema_betas": [ORDINARY_BETA],
            "r2c_v4_primary_deployment_beta": ORDINARY_BETA,
            "r2c_v8_trigger_deployment_beta": TRIGGER_BETA,
            "r2c_v8_recovery_pulse_beta": float(pulse_beta),
            "r2c_v8_recovery_pulse_rounds": int(pulse_rounds),
            "r2c_v12_plan_id": PLAN_ID,
        }
    )
    return config


def _ensure_assets(seed: int, rounds: int, scenario: str) -> dict[str, str]:
    prepare_partition(DATASET_ID, seed)
    prepare_trace(DATASET_ID, scenario, seed, rounds=rounds)
    paths = (
        partition_asset_path(DATASET_ID, seed),
        partition_meta_path(DATASET_ID, seed),
        trace_asset_path(DATASET_ID, scenario, seed, rounds),
        trace_meta_path(DATASET_ID, scenario, seed, rounds),
    )
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in paths
    }


def _job(candidate_id: str, pulse_beta: float, pulse_rounds: int) -> dict[str, Any]:
    run_id = f"A-R2C-D3-S4-V12SCREEN-{candidate_id}-s{SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v12_validation_recovery_pulse_screen",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": DATASET_ID,
        "scenario_id": SCENARIO_ID,
        "rounds": ROUNDS,
        "method_config": _candidate_config(pulse_beta, pulse_rounds),
        "block_id": "A-R2C-D3-V12-RECOVERY-SCREEN",
        "seed": SEED,
        "partition_seed": SEED,
        "trace_seed": SEED,
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "candidate_id": candidate_id,
        "pulse_beta": float(pulse_beta),
        "pulse_rounds": int(pulse_rounds),
        "formal_test_access": False,
        "test_labels_used_for_selection": False,
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def _job_spec(job: dict[str, Any]) -> dict[str, Any]:
    excluded = {"status", "attempts", "actual_run_id", "failure_reason"}
    return {key: value for key, value in job.items() if key not in excluded}


def _frozen_spec(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "scope",
        "plan_id",
        "protocol_version",
        "selection_split",
        "seed",
        "rounds_per_job",
        "event_round",
        "formal_test_access",
        "other_dataset_access",
        "test_labels_used_for_selection",
        "performance_sealed_until_terminal",
        "ordinary_beta",
        "trigger_beta",
        "candidate_grid",
        "baseline_envelope",
        "selection_rule",
        "completion_rule",
        "source_lineage",
        "asset_hashes",
        "implementation_hashes",
        "job_order",
        "max_attempts",
    )
    value = {key: manifest[key] for key in keys}
    value["jobs"] = [_job_spec(job) for job in manifest["jobs"]]
    return value


def build_manifest(persist: bool = True, force: bool = False) -> dict[str, Any]:
    if persist and MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if force and MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if any(job.get("status") != "pending" or int(job.get("attempts", 0)) for job in existing["jobs"]):
            raise RuntimeError("Refusing to rebuild a started v12 screen manifest")
    jobs = [_job(candidate_id, beta, duration) for candidate_id, beta, duration in CANDIDATES]
    if len(jobs) != 4 or len({job["job_id"] for job in jobs}) != 4:
        raise AssertionError("v12 screen must contain four unique candidates")
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_validation_only_v12_recovery_pulse_screen",
        "plan_id": PLAN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "selection_split": "validation",
        "seed": SEED,
        "rounds_per_job": ROUNDS,
        "event_round": EVENT_ROUND,
        "formal_test_access": False,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "performance_sealed_until_terminal": True,
        "ordinary_beta": ORDINARY_BETA,
        "trigger_beta": TRIGGER_BETA,
        "candidate_grid": [
            {"candidate_id": candidate_id, "pulse_beta": beta, "pulse_rounds": duration}
            for candidate_id, beta, duration in CANDIDATES
        ],
        "baseline_envelope": BASELINE_ENVELOPE,
        "selection_rule": (
            "eligible iff S4 last50 and TRM20 strictly exceed their matched baseline maxima "
            "and algorithm TTA is strictly below its matched baseline minimum; select by "
            "TRM20 desc, last50 desc, TTA asc, pulse burden asc, immutable candidate order"
        ),
        "completion_rule": "all four full-budget validation candidates complete and audit before selection",
        "source_lineage": _source_lineage(),
        "asset_hashes": _ensure_assets(SEED, ROUNDS, SCENARIO_ID),
        "implementation_hashes": _implementation_hashes(),
        "job_order": [job["job_id"] for job in jobs],
        "max_attempts": MAX_ATTEMPTS,
        "jobs": jobs,
    }
    manifest["frozen_spec_hash"] = config_hash(_frozen_spec(manifest))
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        immutable_path = QUEUE_ROOT / f"r2c_d3_v12_recovery_screen_manifest_{stamp}.json"
        atomic_json(immutable_path, manifest)
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
                "all_runs_completed": False,
                "performance_sealed_until_terminal": True,
                "frozen_spec_hash": manifest["frozen_spec_hash"],
                "immutable_manifest_path": str(immutable_path),
                "immutable_manifest_sha256": sha256_file(immutable_path),
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
    if (
        manifest.get("scope") != "D3_validation_only_v12_recovery_pulse_screen"
        or manifest.get("plan_id") != PLAN_ID
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("selection_split") != "validation"
        or int(manifest.get("seed", -1)) != SEED
        or int(manifest.get("rounds_per_job", -1)) != ROUNDS
        or bool(manifest.get("formal_test_access"))
        or bool(manifest.get("other_dataset_access"))
        or bool(manifest.get("test_labels_used_for_selection"))
        or not bool(manifest.get("performance_sealed_until_terminal"))
    ):
        raise RuntimeError("v12 screen manifest scope drift")
    if manifest.get("source_lineage") != _source_lineage():
        raise RuntimeError("v12 screen source lineage drift")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("v12 screen implementation drift after freeze")
    if manifest.get("frozen_spec_hash") != config_hash(_frozen_spec(manifest)):
        raise RuntimeError("v12 screen frozen specification drift")
    jobs = list(manifest.get("jobs", []))
    if len(jobs) != 4 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("v12 screen candidate order drift")
    for job, (candidate_id, beta, duration) in zip(jobs, CANDIDATES):
        config = dict(job["method_config"])
        if (
            job.get("candidate_id") != candidate_id
            or float(job.get("pulse_beta", -1.0)) != beta
            or int(job.get("pulse_rounds", -1)) != duration
            or job.get("dataset_id") != DATASET_ID
            or job.get("scenario_id") != SCENARIO_ID
            or job.get("evaluation_split") != "validation"
            or int(job.get("rounds", -1)) != ROUNDS
            or any(int(job.get(key, -1)) != SEED for key in ("seed", "partition_seed", "trace_seed"))
            or config.get("r2c_protocol_version") != PROTOCOL_VERSION
            or config.get("r2c_v4_deployment_ema_betas") != [ORDINARY_BETA]
            or float(config.get("r2c_v8_recovery_pulse_beta", -1.0)) != beta
            or int(config.get("r2c_v8_recovery_pulse_rounds", -1)) != duration
        ):
            raise RuntimeError(f"v12 candidate drift: {job['job_id']}")


def _successful_run_id(job: dict[str, Any]) -> str | None:
    paths: list[Path] = []
    if job.get("actual_run_id"):
        paths.append(RUN_ROOT / str(job["actual_run_id"]))
    paths.append(RUN_ROOT / str(job["base_run_id"]))
    paths.extend(sorted(RUN_ROOT.glob(f"{job['base_run_id']}-a*")))
    seen: set[str] = set()
    for path in paths:
        if path.name in seen:
            continue
        seen.add(path.name)
        if (path / "_SUCCESS.json").exists() and (path / "result.json").exists():
            audit_log = QUEUE_ROOT / "worker_logs" / f"{path.name}.reconcile.audit.log"
            if _audit_run(path.name, audit_log).returncode == 0:
                return path.name
    return None


def _reconcile_successes(manifest: dict[str, Any], events: list[dict[str, Any]]) -> int:
    changed = 0
    for job in manifest["jobs"]:
        run_id = _successful_run_id(job)
        if run_id is not None and (
            job.get("status") != "completed" or job.get("actual_run_id") != run_id
        ):
            job.update({"status": "completed", "actual_run_id": run_id, "failure_reason": None})
            _event(events, job, "reconciled_completed", reason="existing_success_output_and_audit")
            changed += 1
    return changed


def _time_to_accuracy(rounds: pd.DataFrame) -> tuple[int | None, float | None]:
    reached = rounds.loc[rounds["test_accuracy"].astype(float) >= TARGET_ACCURACY]
    if reached.empty:
        return None, None
    first = reached.sort_values("round").iloc[0]
    return int(first["round"]), float(first["algorithm_elapsed_s"])


def _run_metrics(job: dict[str, Any]) -> dict[str, Any]:
    run_id = str(job["actual_run_id"])
    run_dir = RUN_ROOT / run_id
    actual_job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    manifest = pd.read_parquet(run_dir / "run_manifest.parquet").iloc[0]
    rounds = read_chunked_table(run_dir, "round_metrics").sort_values("round")
    if (
        len(rounds) != ROUNDS
        or rounds["round"].astype(int).tolist() != list(range(1, ROUNDS + 1))
        or str(manifest["source_kind"]) != "CALIBRATION"
        or actual_job.get("evaluation_split") != "validation"
        or actual_job.get("method_config") != job.get("method_config")
        or actual_job.get("test_labels_used_for_selection")
    ):
        raise RuntimeError(f"v12 screen run contract mismatch: {run_id}")
    event = rounds.loc[rounds["event_offset_round"].astype(int) == 0]
    if len(event) != 1 or int(event.iloc[0]["round"]) != EVENT_ROUND:
        raise RuntimeError(f"v12 screen event mismatch: {run_id}")
    pulse_beta = float(job["pulse_beta"])
    pulse_rounds = int(job["pulse_rounds"])
    trigger = rounds["telemetry_shift_trigger"].astype(bool)
    hold = rounds["deployment_quarantine_applied"].astype(bool)
    pulse = rounds["deployment_recovery_pulse_applied"].astype(bool)
    expected_pulse_rounds = list(range(EVENT_ROUND + 1, EVENT_ROUND + pulse_rounds + 1))
    if (
        rounds.loc[trigger, "round"].astype(int).tolist() != [EVENT_ROUND]
        or rounds.loc[hold, "round"].astype(int).tolist() != [EVENT_ROUND]
        or rounds.loc[pulse, "round"].astype(int).tolist() != expected_pulse_rounds
        or rounds["deployment_pulse_labels_used"].astype(bool).any()
        or rounds["deployment_pulse_scenario_metadata_used"].astype(bool).any()
        or not rounds["deployment_pulse_state_server_only"].astype(bool).all()
    ):
        raise RuntimeError(f"v12 screen pulse lineage mismatch: {run_id}")
    metrics = derive_window_metrics(rounds)
    tta_round, tta_s = _time_to_accuracy(rounds)
    last50 = float(rounds.tail(50)["test_accuracy"].astype(float).mean())
    strict = {
        "s4_last50_accuracy": last50 > BASELINE_ENVELOPE["s4_last50_accuracy"],
        "s4_trm20_pp": float(metrics["trm20_pp"]) > BASELINE_ENVELOPE["s4_trm20_pp"],
        "s4_algorithm_tta_s": tta_s is not None
        and float(tta_s) < BASELINE_ENVELOPE["s4_algorithm_tta_s"],
    }
    return {
        "candidate_id": job["candidate_id"],
        "run_id": run_id,
        "pulse_beta": pulse_beta,
        "pulse_rounds": pulse_rounds,
        "pulse_burden": (1.0 - pulse_beta) * pulse_rounds,
        "last50_validation_accuracy": last50,
        "trm20_pp": float(metrics["trm20_pp"]),
        "signed_delta20_pp": float(metrics["signed_delta20_pp"]),
        "tta_round": tta_round,
        "algorithm_tta_s": tta_s,
        "strict_last50_win": bool(strict["s4_last50_accuracy"]),
        "strict_trm20_win": bool(strict["s4_trm20_pp"]),
        "strict_tta_win": bool(strict["s4_algorithm_tta_s"]),
        "screen_gate_met": bool(all(strict.values())),
        "trigger_count": int(trigger.sum()),
        "hold_count": int(hold.sum()),
        "pulse_count": int(pulse.sum()),
        "test_labels_used": False,
    }


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot select from an incomplete v12 screen")
    rows = [_run_metrics(job) for job in manifest["jobs"]]
    frame = pd.DataFrame(rows)
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    eligible = frame.loc[frame["screen_gate_met"].astype(bool)].copy()
    selected: dict[str, Any] | None = None
    if not eligible.empty:
        eligible["candidate_order"] = eligible["candidate_id"].map(
            {candidate_id: index for index, (candidate_id, _, _) in enumerate(CANDIDATES)}
        )
        eligible = eligible.sort_values(
            [
                "trm20_pp",
                "last50_validation_accuracy",
                "algorithm_tta_s",
                "pulse_burden",
                "candidate_order",
            ],
            ascending=[False, False, True, True, True],
            kind="mergesort",
        )
        selected = eligible.iloc[0].drop(labels=["candidate_order"]).to_dict()
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "status": "selected" if selected is not None else "no_candidate_passed",
        "selection_split": "validation",
        "seed": SEED,
        "formal_test_access": False,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "baseline_envelope": BASELINE_ENVELOPE,
        "selection_rule": manifest["selection_rule"],
        "selected": selected,
        "eligible_candidate_ids": eligible["candidate_id"].astype(str).tolist(),
        "run_ids": frame["run_id"].astype(str).tolist(),
        "runs_path": str(RUNS_PATH),
        "frozen_spec_hash": manifest["frozen_spec_hash"],
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
            "status": (
                "screen_completed_selected_validation_pair_required"
                if result["status"] == "selected"
                else "screen_completed_no_candidate_passed"
            ),
            "current_job_id": None,
            "all_runs_completed": True,
            "performance_sealed_until_terminal": False,
            "result_path": str(RESULT_PATH),
            "selected_candidate_id": (
                None if result["selected"] is None else result["selected"]["candidate_id"]
            ),
        }
    )
    _sync(state, manifest)
    atomic_json(STATE_PATH, state)
    return state


def _smoke_job(
    run_id: str,
    scenario: str,
    rounds: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stage": "d3_v12_recovery_pulse_smoke",
        "mode": "sanity",
        "method_id": "R2C-FL",
        "dataset_id": DATASET_ID,
        "scenario_id": scenario,
        "rounds": rounds,
        "method_config": config,
        "block_id": "R2C-D3-V12-RECOVERY-SMOKE",
        "seed": SEED,
        "partition_seed": SEED,
        "trace_seed": SEED,
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": 0.0,
        "formal_test_access": False,
        "test_labels_used_for_selection": False,
    }


def _run_smoke_job(job: dict[str, Any]) -> dict[str, Any]:
    run_id, retry_of = _actual_run_id(str(job["run_id"]))
    value = dict(job)
    value.update({"run_id": run_id, "retry_of_run_id": retry_of})
    job_path = QUEUE_ROOT / "active_jobs" / f"{run_id}.json"
    atomic_json(job_path, value)
    log_path = QUEUE_ROOT / "worker_logs" / f"{run_id}.log"
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        process = subprocess.run(
            [sys.executable, "-m", "r2c_baselines.run", "--job", str(job_path)],
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    success = (RUN_ROOT / run_id / "_SUCCESS.json").exists()
    audit_log = QUEUE_ROOT / "worker_logs" / f"{run_id}.audit.log"
    audit = _audit_run(run_id, audit_log) if success else None
    return {
        "run_id": run_id,
        "train_exit": process.returncode,
        "audit_exit": None if audit is None else audit.returncode,
        "passed": bool(process.returncode == 0 and success and audit is not None and audit.returncode == 0),
        "log_path": str(log_path),
        "audit_log_path": str(audit_log),
    }


def smoke() -> dict[str, Any]:
    prepare_partition(DATASET_ID, SEED)
    prepare_trace(DATASET_ID, "S0", SEED, rounds=10)
    prepare_trace(DATASET_ID, "S4", SEED, rounds=40)
    source_config = _source_config()
    pulse_config = _candidate_config(0.5, 5)
    jobs = (
        _smoke_job(f"R2C-D3-S0-V12-SMOKE-V7REF-s{SEED}", "S0", 10, source_config),
        _smoke_job(f"R2C-D3-S0-V12-SMOKE-V8-s{SEED}", "S0", 10, pulse_config),
        _smoke_job(f"R2C-D3-S4-V12-SMOKE-V8-s{SEED}", "S4", 40, pulse_config),
    )
    rows = [_run_smoke_job(job) for job in jobs]
    if not all(row["passed"] for row in rows):
        payload = {"status": "failed", "completed_utc": utc_now(), "runs": rows}
        atomic_json(SMOKE_RESULT_PATH, payload)
        return payload
    v7 = read_chunked_table(RUN_ROOT / rows[0]["run_id"], "round_metrics").sort_values("round")
    v8 = read_chunked_table(RUN_ROOT / rows[1]["run_id"], "round_metrics").sort_values("round")
    s4 = read_chunked_table(RUN_ROOT / rows[2]["run_id"], "round_metrics").sort_values("round")
    identity_columns = (
        "global_model_hash",
        "evaluation_model_hash",
        "test_accuracy",
        "test_loss",
        "selected_clients",
        "completed_clients",
    )
    no_trigger_identity = bool(
        not v8["telemetry_shift_trigger"].astype(bool).any()
        and not v8["deployment_recovery_pulse_applied"].astype(bool).any()
        and all(
            np.array_equal(v7[column].to_numpy(), v8[column].to_numpy())
            for column in identity_columns
        )
    )
    trigger_rounds = s4.loc[s4["telemetry_shift_trigger"].astype(bool), "round"].astype(int).tolist()
    hold_rounds = s4.loc[s4["deployment_quarantine_applied"].astype(bool), "round"].astype(int).tolist()
    pulse_rounds = s4.loc[s4["deployment_recovery_pulse_applied"].astype(bool), "round"].astype(int).tolist()
    event_rounds = s4.loc[s4["event_offset_round"].astype(int) == 0, "round"].astype(int).tolist()
    pulse_contract = bool(
        len(event_rounds) == 1
        and trigger_rounds == event_rounds
        and hold_rounds == event_rounds
        and pulse_rounds == list(range(event_rounds[0] + 1, event_rounds[0] + 6))
        and not s4["deployment_pulse_labels_used"].astype(bool).any()
        and not s4["deployment_pulse_scenario_metadata_used"].astype(bool).any()
    )
    payload = {
        "status": "passed" if no_trigger_identity and pulse_contract else "failed",
        "completed_utc": utc_now(),
        "runs": rows,
        "no_trigger_exact_identity": no_trigger_identity,
        "pulse_contract": pulse_contract,
        "event_rounds": event_rounds,
        "trigger_rounds": trigger_rounds,
        "hold_rounds": hold_rounds,
        "pulse_rounds": pulse_rounds,
    }
    atomic_json(SMOKE_RESULT_PATH, payload)
    return payload


def status() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"status": "not_built"}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--force", action="store_true")
    sub.add_parser("smoke")
    sub.add_parser("worker")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "build":
        value = build_manifest(force=args.force)
    elif args.command == "smoke":
        value = smoke()
    elif args.command == "worker":
        value = worker()
    else:
        value = status()
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
