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

from .config import PLOT_ROOT, PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .data import partition_asset_path, partition_meta_path, prepare_partition
from .logging_io import read_chunked_table
from .metrics import recovery_auc20
from .r2c_v5 import PROTOCOL_VERSION as CONTROL_PROTOCOL_VERSION
from .r2c_v6 import (
    DEFAULT_COOLDOWN_ROUNDS,
    DEFAULT_FRACTION_THRESHOLD,
    DEFAULT_LOG_RATIO_THRESHOLD,
    DEFAULT_MIN_COMPARABLE_CLIENTS,
    PROTOCOL_VERSION as TREATMENT_PROTOCOL_VERSION,
)
from .traces import prepare_trace, trace_asset_path, trace_meta_path
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
NEW_JOB_MATRIX = (
    (20260808, "control"),
    (20260808, "telemetry_sync"),
    (20260809, "control"),
    (20260809, "telemetry_sync"),
    (20260810, "telemetry_sync"),
)
ROUNDS = 600
EVENT_ROUND = 300
TARGET_ACCURACY = 0.7986707616707616
MAX_ATTEMPTS = 3

PLAN_PATH = (
    PROJECT_ROOT
    / "refine-logs"
    / "D3_V7_TELEMETRY_SYNC_PLAN_AMENDMENT_20260817_164238.md"
)
EXPECTED_PLAN_SHA256 = (
    "7BFE71BFD29FE38EFD72DF25E3DA90BC8FD2FCD18EE8ECC5131BDBCD314F07EC"
)
SOURCE_CONTROL_RUN_ID = "A-R2C-D3-S4-V6DIV-L0450-s20260810"
SOURCE_CONTROL_DIR = RUN_ROOT / SOURCE_CONTROL_RUN_ID
EXPECTED_SOURCE_HASHES = {
    "job.json": "23D64116AC80829339081901CA005FC580E83C8883A2D89136B0038997EE6E22",
    "result.json": "615C9333FEE8F8977FD59A9279C38F76B805407A6C4C47387B4ADF83AB678EFD",
    "_SUCCESS.json": "615C9333FEE8F8977FD59A9279C38F76B805407A6C4C47387B4ADF83AB678EFD",
    "run_manifest.parquet": "BCF42CFC872D295CEFC3074803EDDB0C8F50714C20FE79E8E0BF6238F643D00F",
}

MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v7_phase_e_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v7_phase_e_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v7_phase_e_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v7_phase_e_result.json"
RUNS_PATH = PLOT_ROOT / "r2c_d3_v7_phase_e_runs.parquet"
RUNS_CSV_PATH = PLOT_ROOT / "r2c_d3_v7_phase_e_runs.csv"
PAIRS_PATH = PLOT_ROOT / "r2c_d3_v7_phase_e_pairs.parquet"
PAIRS_CSV_PATH = PLOT_ROOT / "r2c_d3_v7_phase_e_pairs.csv"
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
    values = {str(PLAN_PATH.relative_to(PROJECT_ROOT)): sha256_file(PLAN_PATH)}
    for name in EXPECTED_SOURCE_HASHES:
        path = SOURCE_CONTROL_DIR / name
        values[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)
    return values


def _ensure_assets() -> dict[str, str]:
    for seed in SEEDS:
        prepare_partition(DATASET_ID, seed)
        prepare_trace(DATASET_ID, SCENARIO_ID, seed, rounds=ROUNDS)
    paths: list[Path] = []
    for seed in SEEDS:
        paths.extend(
            [
                partition_asset_path(DATASET_ID, seed),
                partition_meta_path(DATASET_ID, seed),
                trace_asset_path(DATASET_ID, SCENARIO_ID, seed, ROUNDS),
                trace_meta_path(DATASET_ID, SCENARIO_ID, seed, ROUNDS),
            ]
        )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _verified_chunked_table(run_dir: Path, table_name: str) -> pd.DataFrame:
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    index = result["table_indices"][table_name]
    table_dir = run_dir / "tables" / table_name
    expected_parts = index["parts"]
    actual_names = sorted(path.name for path in table_dir.glob("part-*.parquet"))
    expected_names = [str(part["path"]) for part in expected_parts]
    if actual_names != expected_names:
        raise RuntimeError(f"{run_dir.name}/{table_name} part-set mismatch")
    for part in expected_parts:
        path = table_dir / str(part["path"])
        if sha256_file(path) != str(part["sha256"]):
            raise RuntimeError(f"{run_dir.name}/{table_name}/{path.name} hash mismatch")
    frame = read_chunked_table(run_dir, table_name)
    if len(frame) != int(index["rows"]):
        raise RuntimeError(f"{run_dir.name}/{table_name} row-count mismatch")
    return frame


def _audit_run(run_id: str, log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        return subprocess.run(
            [sys.executable, str(AUDIT_SCRIPT), str(RUN_ROOT / run_id)],
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


def _verify_source_control(run_audit: bool) -> dict[str, Any]:
    if sha256_file(PLAN_PATH) != EXPECTED_PLAN_SHA256.lower():
        raise RuntimeError("Frozen v7 plan hash drift")
    for name, expected in EXPECTED_SOURCE_HASHES.items():
        if sha256_file(SOURCE_CONTROL_DIR / name) != expected.lower():
            raise RuntimeError(f"Frozen seed-20260810 control hash drift: {name}")
    job = json.loads((SOURCE_CONTROL_DIR / "job.json").read_text(encoding="utf-8"))
    if (
        job.get("dataset_id") != DATASET_ID
        or job.get("scenario_id") != SCENARIO_ID
        or job.get("evaluation_split") != "validation"
        or int(job.get("rounds", -1)) != ROUNDS
        or any(int(job[key]) != 20260810 for key in ("seed", "partition_seed", "trace_seed"))
        or not bool(job.get("full_logging"))
    ):
        raise RuntimeError("Frozen seed-20260810 control protocol drift")
    config = dict(job["method_config"])
    if (
        config.get("r2c_protocol_version") != CONTROL_PROTOCOL_VERSION
        or float(config.get("r2c_v5_history_mix")) != 0.45
        or float(config.get("r2c_v5_history_temperature")) != 1.0
        or config.get("r2c_v4_deployment_ema_betas") != [0.9]
        or float(config.get("r2c_v4_primary_deployment_beta")) != 0.9
        or bool(config.get("r2c_v2_audit_replay"))
    ):
        raise RuntimeError("Frozen seed-20260810 control method configuration drift")
    rounds = _verified_chunked_table(SOURCE_CONTROL_DIR, "round_metrics")
    if rounds["round"].astype(int).tolist() != list(range(1, ROUNDS + 1)):
        raise RuntimeError("Frozen seed-20260810 control round budget is incomplete")
    if run_audit:
        audit_log = QUEUE_ROOT / "worker_logs" / f"{SOURCE_CONTROL_RUN_ID}.phase_e_reuse.audit.log"
        audit = _audit_run(SOURCE_CONTROL_RUN_ID, audit_log)
        if audit.returncode != 0:
            raise RuntimeError(f"Frozen seed-20260810 control audit failed: {audit_log}")
    return config


def _treatment_config(control_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(control_config)
    config.update(
        {
            "r2c_protocol_version": TREATMENT_PROTOCOL_VERSION,
            "r2c_v6_duration_log_ratio_threshold": DEFAULT_LOG_RATIO_THRESHOLD,
            "r2c_v6_changed_fraction_threshold": DEFAULT_FRACTION_THRESHOLD,
            "r2c_v6_min_comparable_clients": DEFAULT_MIN_COMPARABLE_CLIENTS,
            "r2c_v6_cooldown_rounds": DEFAULT_COOLDOWN_ROUNDS,
        }
    )
    return config


def _job(seed: int, variant: str, control_config: dict[str, Any]) -> dict[str, Any]:
    label = "CONTROL" if variant == "control" else "SYNC"
    run_id = f"A-R2C-D3-S4-V7E-{label}-s{seed}"
    method_config = (
        dict(control_config)
        if variant == "control"
        else _treatment_config(control_config)
    )
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v7_phase_e_multi_seed_validation",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": DATASET_ID,
        "scenario_id": SCENARIO_ID,
        "rounds": ROUNDS,
        "method_config": method_config,
        "block_id": "A-R2C-D3-V7-PHASE-E",
        "seed": int(seed),
        "partition_seed": int(seed),
        "trace_seed": int(seed),
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "variant": variant,
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
            raise RuntimeError("Refusing to rebuild a started D3 v7 Phase E manifest")
    control_config = _verify_source_control(run_audit=persist)
    asset_hashes = _ensure_assets()
    jobs = [_job(seed, variant, control_config) for seed, variant in NEW_JOB_MATRIX]
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_only_v7_phase_e_multi_development_seed_validation",
        "formal_test_access": False,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "rounds_per_job": ROUNDS,
        "event_round": EVENT_ROUND,
        "development_seeds": list(SEEDS),
        "new_job_count": len(jobs),
        "reused_control_run_id": SOURCE_CONTROL_RUN_ID,
        "job_order": [job["job_id"] for job in jobs],
        "detector_config": {
            "log_ratio_threshold": DEFAULT_LOG_RATIO_THRESHOLD,
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
            "max_per_seed_tta_overhead_fraction": 0.02,
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
        atomic_json(QUEUE_ROOT / f"r2c_d3_v7_phase_e_manifest_{stamp}.json", manifest)
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
                "phase_f_authorized": False,
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
    if manifest.get("formal_test_access") or manifest.get("other_dataset_access"):
        raise RuntimeError("D3 v7 Phase E attempted prohibited access")
    if manifest.get("test_labels_used_for_selection"):
        raise RuntimeError("D3 v7 Phase E attempted test-label selection")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("Implementation hashes changed after D3 v7 Phase E freeze")
    if manifest.get("source_hashes") != _source_hashes():
        raise RuntimeError("Source hashes changed after D3 v7 Phase E freeze")
    if manifest.get("asset_hashes") != _ensure_assets():
        raise RuntimeError("Asset hashes changed after D3 v7 Phase E freeze")
    jobs = manifest.get("jobs", [])
    if len(jobs) != len(NEW_JOB_MATRIX) or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("D3 v7 Phase E job matrix drift")
    control_config = _verify_source_control(run_audit=False)
    treatment_config = _treatment_config(control_config)
    for job, (seed, variant) in zip(jobs, NEW_JOB_MATRIX):
        if (
            job.get("dataset_id") != DATASET_ID
            or job.get("scenario_id") != SCENARIO_ID
            or job.get("evaluation_split") != "validation"
            or int(job.get("rounds", -1)) != ROUNDS
            or not bool(job.get("full_logging"))
            or any(int(job[key]) != seed for key in ("seed", "partition_seed", "trace_seed"))
            or job.get("variant") != variant
        ):
            raise RuntimeError(f"D3 v7 Phase E protocol drift in {job['job_id']}")
        expected_config = control_config if variant == "control" else treatment_config
        if job.get("method_config") != expected_config:
            raise RuntimeError(f"D3 v7 Phase E method-config drift in {job['job_id']}")


def _time_to_accuracy(rounds: pd.DataFrame) -> tuple[int | None, float | None]:
    reached = rounds.loc[rounds["test_accuracy"].astype(float) >= TARGET_ACCURACY]
    if reached.empty:
        return None, None
    first = reached.sort_values("round").iloc[0]
    return int(first["round"]), float(first["algorithm_elapsed_s"])


def _run_metrics(run_id: str, seed: int, variant: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / run_id
    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    manifest = pd.read_parquet(run_dir / "run_manifest.parquet").iloc[0]
    rounds = _verified_chunked_table(run_dir, "round_metrics").sort_values("round")
    if rounds["round"].astype(int).tolist() != list(range(1, ROUNDS + 1)):
        raise RuntimeError(f"Incomplete Phase E trajectory: {run_id}")
    if (
        str(manifest["source_kind"]) != "CALIBRATION"
        or job.get("evaluation_split") != "validation"
        or any(int(job[key]) != seed for key in ("seed", "partition_seed", "trace_seed"))
    ):
        raise RuntimeError(f"Phase E source/seed/split mismatch: {run_id}")
    events = rounds.loc[rounds["event_offset_round"].astype(int) == 0]
    if len(events) != 1 or int(events.iloc[0]["round"]) != EVENT_ROUND:
        raise RuntimeError(f"Phase E event-round drift: {run_id}")
    direct = recovery_auc20(
        rounds["round"].astype(int).tolist(),
        rounds["test_accuracy"].astype(float).tolist(),
        EVENT_ROUND,
    )
    if not direct["recovery_auc20_complete"]:
        raise RuntimeError(f"Phase E lacks strict AUC@20: {run_id}")
    auc = float(direct["recovery_deficit_auc20"])
    stored_auc = result["recovery"]["recovery_deficit_auc20"]
    if stored_auc is None or abs(float(stored_auc) - auc) > 1.0e-15:
        raise RuntimeError(f"Phase E stored/direct AUC mismatch: {run_id}")
    direct_last50 = float(rounds.tail(50)["test_accuracy"].astype(float).mean())
    if abs(float(result["last50_accuracy"]) - direct_last50) > 1.0e-15:
        raise RuntimeError(f"Phase E Last50 mismatch: {run_id}")
    tta_round, tta_s = _time_to_accuracy(rounds)
    last = rounds.iloc[-1]
    trigger_count: int | None = None
    trigger_rounds: list[int] | None = None
    forbidden_input_clean: bool | None = None
    if variant == "telemetry_sync":
        required = {
            "telemetry_shift_trigger",
            "telemetry_shift_labels_used",
            "telemetry_shift_scenario_metadata_used",
            "deployment_synchronization_applied",
        }
        if not required.issubset(rounds.columns):
            raise RuntimeError(f"Phase E telemetry audit fields missing: {run_id}")
        triggered = rounds.loc[rounds["telemetry_shift_trigger"].astype(bool)]
        trigger_count = len(triggered)
        trigger_rounds = triggered["round"].astype(int).tolist()
        forbidden_input_clean = bool(
            not rounds["telemetry_shift_labels_used"].astype(bool).any()
            and not rounds["telemetry_shift_scenario_metadata_used"].astype(bool).any()
        )
        if not np.array_equal(
            rounds["telemetry_shift_trigger"].astype(bool).to_numpy(),
            rounds["deployment_synchronization_applied"].astype(bool).to_numpy(),
        ):
            raise RuntimeError(f"Phase E trigger/synchronization mismatch: {run_id}")
    return {
        "run_id": run_id,
        "seed": int(seed),
        "variant": variant,
        "protocol_version": job["method_config"]["r2c_protocol_version"],
        "config_hash": config_hash(job["method_config"]),
        "last50_validation_accuracy": direct_last50,
        "recovery_deficit_auc20": auc,
        "tta_round": tta_round,
        "algorithm_tta_s": tta_s,
        "algorithm_elapsed_s": float(result["algorithm_elapsed_s"]),
        "final_participation_jfi": float(last["participation_jfi"]),
        "final_worst10_participation": float(last["worst10_participation"]),
        "trigger_count": trigger_count,
        "trigger_rounds_json": None if trigger_rounds is None else canonical_json(trigger_rounds),
        "forbidden_input_clean": forbidden_input_clean,
        "source_kind": str(manifest["source_kind"]),
        "test_labels_used": False,
    }


def _evaluate_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        subset = frame.loc[frame["seed"].astype(int) == seed]
        if set(subset["variant"].astype(str)) != {"control", "telemetry_sync"}:
            raise RuntimeError(f"Incomplete Phase E pair for seed {seed}")
        control = subset.loc[subset["variant"] == "control"].iloc[0]
        treatment = subset.loc[subset["variant"] == "telemetry_sync"].iloc[0]
        control_tta = control["algorithm_tta_s"]
        treatment_tta = treatment["algorithm_tta_s"]
        tta_overhead = (
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
                "tta_overhead_fraction": tta_overhead,
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
        "per_seed_tta_overhead_within_2pct": bool(
            np.isfinite(pairs["tta_overhead_fraction"].astype(float)).all()
            and (pairs["tta_overhead_fraction"].astype(float) <= 0.02).all()
        ),
        "per_seed_final_jfi_loss_within_0p01": bool((pairs["final_jfi_loss"] <= 0.01).all()),
        "per_seed_final_worst10_loss_within_2": bool((pairs["final_worst10_loss"] <= 2.0).all()),
    }
    return pairs, gates


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot freeze incomplete D3 v7 Phase E")
    _verify_source_control(run_audit=True)
    rows = [
        _run_metrics(SOURCE_CONTROL_RUN_ID, 20260810, "control")
    ]
    for job in manifest["jobs"]:
        rows.append(
            _run_metrics(str(job["actual_run_id"]), int(job["seed"]), str(job["variant"]))
        )
    frame = pd.DataFrame(rows).sort_values(["seed", "variant"], kind="mergesort")
    pairs, gates = _evaluate_pairs(frame)
    phase_f_authorized = bool(all(gates.values()))
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
        "reused_control_run_id": SOURCE_CONTROL_RUN_ID,
        "run_count": len(frame),
        "pair_count": len(pairs),
        "gates": gates,
        "phase_e_passed": phase_f_authorized,
        "phase_f_authorized": phase_f_authorized,
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
                "phase_e_completed_phase_f_authorized"
                if result["phase_f_authorized"]
                else "phase_e_completed_gate_failed"
            ),
            "current_job_id": None,
            "phase_f_authorized": bool(result["phase_f_authorized"]),
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
