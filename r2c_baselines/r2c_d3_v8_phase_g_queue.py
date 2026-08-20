from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import r2c_d3_v7_phase_e_queue as v7
from .config import PLOT_ROOT, PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .r2c_v6 import (
    DEFAULT_COOLDOWN_ROUNDS,
    DEFAULT_FRACTION_THRESHOLD,
    DEFAULT_MIN_COMPARABLE_CLIENTS,
    PROTOCOL_VERSION as TREATMENT_PROTOCOL_VERSION,
)
from .utils import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    canonical_json,
    config_hash,
    sha256_file,
    utc_now,
)


DATASET_ID = "D3"
SCENARIO_ID = "S4"
SEEDS = (20260808, 20260809, 20260810)
ROUNDS = 600
EVENT_ROUND = 300
TARGET_ACCURACY = v7.TARGET_ACCURACY
MAX_ATTEMPTS = 3
V8_DURATION_RATIO = 1.25
V8_LOG_RATIO_THRESHOLD = math.log(V8_DURATION_RATIO)

PLAN_PATH = (
    PROJECT_ROOT
    / "refine-logs"
    / "D3_V8_TELEMETRY_THRESHOLD_PLAN_AMENDMENT_20260817_184400.md"
)
EXPECTED_PLAN_SHA256 = "3943A2F2148285C01A22C3754CF21EDB086CDB692272148A391A245282FE3485"
V7_RESULT_PATH = QUEUE_ROOT / "r2c_d3_v7_phase_e_result.json"
EXPECTED_V7_RESULT_SHA256 = "8A928F0C17E482A528CC3610B80AF52EB2279F20D447AF04FBC3AD096973EA72"

CONTROL_RUNS: dict[int, str] = {
    20260808: "A-R2C-D3-S4-V7E-CONTROL-s20260808",
    20260809: "A-R2C-D3-S4-V7E-CONTROL-s20260809",
    20260810: "A-R2C-D3-S4-V6DIV-L0450-s20260810",
}
EXPECTED_CONTROL_HASHES: dict[int, dict[str, str]] = {
    20260808: {
        "job.json": "E241BAA01D081B47BB7998ECD72D33D38104F7246E864E307F5C9600B62ED749",
        "result.json": "2AAA34EA1ACC8D50C9E2FF39BA7884ED73E12F1AAAD336808567BF5418EC9BDF",
        "_SUCCESS.json": "2AAA34EA1ACC8D50C9E2FF39BA7884ED73E12F1AAAD336808567BF5418EC9BDF",
        "run_manifest.parquet": "590306753F1C21DFBB4C389B2E11BAC05FD239D6123A63303305CBC529CAACD4",
    },
    20260809: {
        "job.json": "0A199BD8512052F719EA8FC66F40283439BA9A94AD61B61050D4C134CAD73280",
        "result.json": "84759136F1C3A36B27C126C6BED3FFADE34973EF9DC0054830335E3920D501A1",
        "_SUCCESS.json": "84759136F1C3A36B27C126C6BED3FFADE34973EF9DC0054830335E3920D501A1",
        "run_manifest.parquet": "B39D53FF64E67548FA430DCBA05B4C8A8DDE356876EEC0069B1AC89F08498B1C",
    },
    20260810: {
        "job.json": "23D64116AC80829339081901CA005FC580E83C8883A2D89136B0038997EE6E22",
        "result.json": "615C9333FEE8F8977FD59A9279C38F76B805407A6C4C47387B4ADF83AB678EFD",
        "_SUCCESS.json": "615C9333FEE8F8977FD59A9279C38F76B805407A6C4C47387B4ADF83AB678EFD",
        "run_manifest.parquet": "BCF42CFC872D295CEFC3074803EDDB0C8F50714C20FE79E8E0BF6238F643D00F",
    },
}

MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v8_phase_g_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v8_phase_g_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v8_phase_g_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v8_phase_g_result.json"
RUNS_PATH = PLOT_ROOT / "r2c_d3_v8_phase_g_runs.parquet"
RUNS_CSV_PATH = PLOT_ROOT / "r2c_d3_v8_phase_g_runs.csv"
PAIRS_PATH = PLOT_ROOT / "r2c_d3_v8_phase_g_pairs.parquet"
PAIRS_CSV_PATH = PLOT_ROOT / "r2c_d3_v8_phase_g_pairs.csv"
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
        "run.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    values["tests/audit_r2c_run.py"] = sha256_file(AUDIT_SCRIPT)
    return values


def _source_hashes() -> dict[str, str]:
    values = {
        str(PLAN_PATH.relative_to(PROJECT_ROOT)): sha256_file(PLAN_PATH),
        str(V7_RESULT_PATH.relative_to(PROJECT_ROOT)): sha256_file(V7_RESULT_PATH),
    }
    for seed, run_id in CONTROL_RUNS.items():
        run_dir = RUN_ROOT / run_id
        for name in EXPECTED_CONTROL_HASHES[seed]:
            path = run_dir / name
            values[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)
    return values


def _audit_run(run_id: str, suffix: str = "audit") -> subprocess.CompletedProcess[str]:
    log_path = QUEUE_ROOT / "worker_logs" / f"{run_id}.{suffix}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        return subprocess.run(
            [sys.executable, str(AUDIT_SCRIPT), str(RUN_ROOT / run_id)],
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


def _verify_controls(run_audit: bool) -> dict[str, Any]:
    if sha256_file(PLAN_PATH).upper() != EXPECTED_PLAN_SHA256:
        raise RuntimeError("Frozen v8 plan hash drift")
    if sha256_file(V7_RESULT_PATH).upper() != EXPECTED_V7_RESULT_SHA256:
        raise RuntimeError("Frozen v7 Phase E result hash drift")
    v7_result = json.loads(V7_RESULT_PATH.read_text(encoding="utf-8"))
    if v7_result.get("phase_f_authorized") or v7_result.get("phase_e_passed"):
        raise RuntimeError("v8 requires the recorded v7 Phase E gate failure")

    configs: list[dict[str, Any]] = []
    for seed, run_id in CONTROL_RUNS.items():
        run_dir = RUN_ROOT / run_id
        for name, expected in EXPECTED_CONTROL_HASHES[seed].items():
            if sha256_file(run_dir / name).upper() != expected:
                raise RuntimeError(f"Frozen control hash drift: {run_id}/{name}")
        job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
        if (
            job.get("dataset_id") != DATASET_ID
            or job.get("scenario_id") != SCENARIO_ID
            or job.get("evaluation_split") != "validation"
            or int(job.get("rounds", -1)) != ROUNDS
            or any(int(job[key]) != seed for key in ("seed", "partition_seed", "trace_seed"))
            or not bool(job.get("full_logging"))
            or bool(job.get("method_config", {}).get("r2c_v2_audit_replay"))
        ):
            raise RuntimeError(f"Frozen control contract drift: {run_id}")
        rounds = v7._verified_chunked_table(run_dir, "round_metrics")
        if rounds["round"].astype(int).tolist() != list(range(1, ROUNDS + 1)):
            raise RuntimeError(f"Frozen control round budget incomplete: {run_id}")
        if run_audit and _audit_run(run_id, "phase_g_reuse.audit").returncode != 0:
            raise RuntimeError(f"Frozen control audit failed: {run_id}")
        configs.append(dict(job["method_config"]))

    if any(config != configs[0] for config in configs[1:]):
        raise RuntimeError("Frozen controls do not share one method configuration")
    return configs[0]


def _treatment_config(control_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(control_config)
    config.update(
        {
            "r2c_protocol_version": TREATMENT_PROTOCOL_VERSION,
            "r2c_v6_duration_log_ratio_threshold": V8_LOG_RATIO_THRESHOLD,
            "r2c_v6_changed_fraction_threshold": DEFAULT_FRACTION_THRESHOLD,
            "r2c_v6_min_comparable_clients": DEFAULT_MIN_COMPARABLE_CLIENTS,
            "r2c_v6_cooldown_rounds": DEFAULT_COOLDOWN_ROUNDS,
        }
    )
    return config


def _job(seed: int, control_config: dict[str, Any]) -> dict[str, Any]:
    run_id = f"A-R2C-D3-S4-V8G-SYNC-s{seed}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v8_phase_g_multi_seed_validation_repair",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": DATASET_ID,
        "scenario_id": SCENARIO_ID,
        "rounds": ROUNDS,
        "method_config": _treatment_config(control_config),
        "block_id": "A-R2C-D3-V8-PHASE-G",
        "seed": int(seed),
        "partition_seed": int(seed),
        "trace_seed": int(seed),
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "variant": "telemetry_sync_v8",
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
            raise RuntimeError("Refusing to rebuild a started D3 v8 Phase G manifest")

    control_config = _verify_controls(run_audit=persist)
    asset_hashes = v7._ensure_assets()
    jobs = [_job(seed, control_config) for seed in SEEDS]
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_only_v8_phase_g_multi_seed_validation_repair",
        "formal_test_access": False,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "rounds_per_job": ROUNDS,
        "event_round": EVENT_ROUND,
        "development_seeds": list(SEEDS),
        "new_job_count": len(jobs),
        "reused_control_run_ids": {str(seed): CONTROL_RUNS[seed] for seed in SEEDS},
        "job_order": [job["job_id"] for job in jobs],
        "detector_config": {
            "duration_ratio": V8_DURATION_RATIO,
            "log_ratio_threshold": V8_LOG_RATIO_THRESHOLD,
            "changed_fraction_threshold": DEFAULT_FRACTION_THRESHOLD,
            "min_comparable_clients": DEFAULT_MIN_COMPARABLE_CLIENTS,
            "cooldown_rounds": DEFAULT_COOLDOWN_ROUNDS,
        },
        "hard_gates": {
            "treatment_trigger_count_per_seed": 1,
            "treatment_trigger_round": EVENT_ROUND,
            "mean_auc_strictly_lower": True,
            "worst_auc_no_higher": True,
            "auc_no_worse_seed_count_min": 2,
            "max_per_seed_last50_loss_pp": 0.25,
            "max_mean_last50_loss_pp": 0.15,
            "max_mean_tta_overhead_fraction": 0.02,
            "max_per_seed_tta_overhead_fraction": 0.05,
            "max_per_seed_final_jfi_loss": 0.01,
            "max_per_seed_final_worst10_loss": 2.0,
        },
        "implementation_hashes": _implementation_hashes(),
        "source_hashes": _source_hashes(),
        "asset_hashes": asset_hashes,
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d3_v8_phase_g_manifest_{stamp}.json", manifest)
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
                "phase_h_authorized": False,
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
        if run_id is None or (job.get("status") == "completed" and job.get("actual_run_id") == run_id):
            continue
        if _audit_run(run_id, "phase_g_reconcile.audit").returncode != 0:
            raise RuntimeError(f"Existing success output failed audit: {run_id}")
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
    if manifest.get("formal_test_access") or manifest.get("other_dataset_access"):
        raise RuntimeError("D3 v8 Phase G attempted prohibited access")
    if manifest.get("test_labels_used_for_selection"):
        raise RuntimeError("D3 v8 Phase G attempted test-label selection")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("Implementation hashes changed after D3 v8 Phase G freeze")
    if manifest.get("source_hashes") != _source_hashes():
        raise RuntimeError("Source hashes changed after D3 v8 Phase G freeze")
    if manifest.get("asset_hashes") != v7._ensure_assets():
        raise RuntimeError("Asset hashes changed after D3 v8 Phase G freeze")
    jobs = manifest.get("jobs", [])
    if len(jobs) != len(SEEDS) or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("D3 v8 Phase G job matrix drift")
    control_config = _verify_controls(run_audit=False)
    treatment_config = _treatment_config(control_config)
    for job, seed in zip(jobs, SEEDS):
        if (
            job.get("dataset_id") != DATASET_ID
            or job.get("scenario_id") != SCENARIO_ID
            or job.get("evaluation_split") != "validation"
            or int(job.get("rounds", -1)) != ROUNDS
            or not bool(job.get("full_logging"))
            or any(int(job[key]) != seed for key in ("seed", "partition_seed", "trace_seed"))
            or job.get("variant") != "telemetry_sync_v8"
            or job.get("method_config") != treatment_config
        ):
            raise RuntimeError(f"D3 v8 Phase G protocol drift in {job['job_id']}")


def _evaluate_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        subset = frame.loc[frame["seed"].astype(int) == seed]
        if set(subset["variant"].astype(str)) != {"control", "telemetry_sync_v8"}:
            raise RuntimeError(f"Incomplete D3 v8 Phase G pair for seed {seed}")
        control = subset.loc[subset["variant"] == "control"].iloc[0]
        treatment = subset.loc[subset["variant"] == "telemetry_sync_v8"].iloc[0]
        control_tta = control["algorithm_tta_s"]
        treatment_tta = treatment["algorithm_tta_s"]
        overhead = (
            math.inf
            if pd.isna(control_tta) or pd.isna(treatment_tta) or float(control_tta) <= 0.0
            else float(treatment_tta) / float(control_tta) - 1.0
        )
        rows.append(
            {
                "seed": seed,
                "control_run_id": control["run_id"],
                "treatment_run_id": treatment["run_id"],
                "control_auc20": float(control["recovery_deficit_auc20"]),
                "treatment_auc20": float(treatment["recovery_deficit_auc20"]),
                "auc_delta": float(treatment["recovery_deficit_auc20"])
                - float(control["recovery_deficit_auc20"]),
                "last50_loss_pp": 100.0
                * (
                    float(control["last50_validation_accuracy"])
                    - float(treatment["last50_validation_accuracy"])
                ),
                "tta_overhead_fraction": overhead,
                "final_jfi_loss": float(control["final_participation_jfi"])
                - float(treatment["final_participation_jfi"]),
                "final_worst10_loss": float(control["final_worst10_participation"])
                - float(treatment["final_worst10_participation"]),
                "treatment_trigger_count": int(treatment["trigger_count"]),
                "treatment_trigger_rounds_json": treatment["trigger_rounds_json"],
                "forbidden_input_clean": bool(treatment["forbidden_input_clean"]),
            }
        )
    pairs = pd.DataFrame(rows)
    overhead = pairs["tta_overhead_fraction"].astype(float)
    gates = {
        "trigger_exactly_once_each_seed": bool((pairs["treatment_trigger_count"] == 1).all()),
        "trigger_at_event_each_seed": bool(
            (pairs["treatment_trigger_rounds_json"] == canonical_json([EVENT_ROUND])).all()
        ),
        "forbidden_input_clean": bool(pairs["forbidden_input_clean"].all()),
        "mean_auc_strictly_lower": bool(
            pairs["treatment_auc20"].astype(float).mean()
            < pairs["control_auc20"].astype(float).mean()
        ),
        "worst_auc_no_higher": bool(
            pairs["treatment_auc20"].astype(float).max()
            <= pairs["control_auc20"].astype(float).max()
        ),
        "auc_no_worse_at_least_two_seeds": bool((pairs["auc_delta"] <= 0.0).sum() >= 2),
        "per_seed_last50_loss_within_0p25pp": bool((pairs["last50_loss_pp"] <= 0.25).all()),
        "mean_last50_loss_within_0p15pp": bool(pairs["last50_loss_pp"].mean() <= 0.15),
        "mean_tta_overhead_within_2pct": bool(
            np.isfinite(overhead).all() and overhead.mean() <= 0.02
        ),
        "per_seed_tta_overhead_within_5pct": bool(
            np.isfinite(overhead).all() and (overhead <= 0.05).all()
        ),
        "per_seed_final_jfi_loss_within_0p01": bool((pairs["final_jfi_loss"] <= 0.01).all()),
        "per_seed_final_worst10_loss_within_2": bool(
            (pairs["final_worst10_loss"] <= 2.0).all()
        ),
    }
    return pairs, gates


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot freeze incomplete D3 v8 Phase G")
    control_config = _verify_controls(run_audit=True)
    treatment_config = _treatment_config(control_config)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows.append(v7._run_metrics(CONTROL_RUNS[seed], seed, "control"))
    for job in manifest["jobs"]:
        metrics = v7._run_metrics(str(job["actual_run_id"]), int(job["seed"]), "telemetry_sync")
        metrics["variant"] = "telemetry_sync_v8"
        if metrics["config_hash"] != config_hash(treatment_config):
            raise RuntimeError(f"D3 v8 config hash mismatch: {job['job_id']}")
        rows.append(metrics)
    frame = pd.DataFrame(rows).sort_values(["seed", "variant"], kind="mergesort")
    pairs, gates = _evaluate_pairs(frame)
    phase_h_authorized = bool(all(gates.values()))
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    atomic_parquet(PAIRS_PATH, pairs)
    atomic_csv(PAIRS_CSV_PATH, pairs)
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "dataset_id": DATASET_ID,
        "scenario_id": SCENARIO_ID,
        "selection_split": "validation",
        "formal_test_access": False,
        "test_labels_used": False,
        "development_seeds": list(SEEDS),
        "new_job_count": len(manifest["jobs"]),
        "reused_control_run_ids": {str(seed): CONTROL_RUNS[seed] for seed in SEEDS},
        "run_count": len(frame),
        "pair_count": len(pairs),
        "detector_duration_ratio": V8_DURATION_RATIO,
        "gates": gates,
        "phase_g_passed": phase_h_authorized,
        "phase_h_authorized": phase_h_authorized,
        "runs_path": str(RUNS_PATH),
        "pairs_path": str(PAIRS_PATH),
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
        job["attempts"] = int(job.get("attempts", 0)) + 1
        resolved = _resolved(job)
        job.update(
            {
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
        audit = _audit_run(str(resolved["run_id"])) if success else None
        if process.returncode == 0 and success and audit is not None and audit.returncode == 0:
            job["status"] = "completed"
            _event(events, job, "completed", exit_code=0)
        else:
            audit_code = None if audit is None else audit.returncode
            job["status"] = "failed"
            job["failure_reason"] = (
                f"train_exit={process.returncode};audit_exit={audit_code};log={log_path}"
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
                "phase_g_completed_phase_h_authorized"
                if result["phase_h_authorized"]
                else "phase_g_completed_gate_failed"
            ),
            "current_job_id": None,
            "phase_h_authorized": bool(result["phase_h_authorized"]),
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
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

