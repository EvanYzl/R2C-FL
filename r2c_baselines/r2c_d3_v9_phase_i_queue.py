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
)
from .r2c_v7 import (
    DEFAULT_TRIGGER_DEPLOYMENT_BETA,
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
V9_DURATION_RATIO = 1.25
V9_LOG_RATIO_THRESHOLD = math.log(V9_DURATION_RATIO)

PLAN_PATH = (
    PROJECT_ROOT
    / "refine-logs"
    / "D3_V9_TELEMETRY_QUARANTINE_PLAN_AMENDMENT_20260817_200035.md"
)
EXPECTED_PLAN_SHA256 = "6529BD7D35827884117A250ECE2A0C07A84E21DA702C314F21FDC831F1AEB264"
V8_RESULT_PATH = QUEUE_ROOT / "r2c_d3_v8_phase_g_result.json"
EXPECTED_V8_RESULT_SHA256 = "635926BC7043C1787B0CB46A1D68A2617FA2916531CCDAB38F8BD53739057DF9"

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

MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v9_phase_i_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v9_phase_i_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v9_phase_i_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v9_phase_i_result.json"
RUNS_PATH = PLOT_ROOT / "r2c_d3_v9_phase_i_runs.parquet"
RUNS_CSV_PATH = PLOT_ROOT / "r2c_d3_v9_phase_i_runs.csv"
PAIRS_PATH = PLOT_ROOT / "r2c_d3_v9_phase_i_pairs.parquet"
PAIRS_CSV_PATH = PLOT_ROOT / "r2c_d3_v9_phase_i_pairs.csv"
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
        "r2c_v7.py",
        "run.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    values["tests/audit_r2c_run.py"] = sha256_file(AUDIT_SCRIPT)
    return values


def _source_hashes() -> dict[str, str]:
    values = {
        str(PLAN_PATH.relative_to(PROJECT_ROOT)): sha256_file(PLAN_PATH),
        str(V8_RESULT_PATH.relative_to(PROJECT_ROOT)): sha256_file(V8_RESULT_PATH),
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
        raise RuntimeError("Frozen v9 plan hash drift")
    if sha256_file(V8_RESULT_PATH).upper() != EXPECTED_V8_RESULT_SHA256:
        raise RuntimeError("Frozen v8 Phase G result hash drift")
    v8_result = json.loads(V8_RESULT_PATH.read_text(encoding="utf-8"))
    if v8_result.get("phase_h_authorized") or v8_result.get("phase_g_passed"):
        raise RuntimeError("v9 requires the recorded v8 Phase G gate failure")

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
        if run_audit and _audit_run(run_id, "phase_i_reuse.audit").returncode != 0:
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
            "r2c_v6_duration_log_ratio_threshold": V9_LOG_RATIO_THRESHOLD,
            "r2c_v6_changed_fraction_threshold": DEFAULT_FRACTION_THRESHOLD,
            "r2c_v6_min_comparable_clients": DEFAULT_MIN_COMPARABLE_CLIENTS,
            "r2c_v6_cooldown_rounds": DEFAULT_COOLDOWN_ROUNDS,
            "r2c_v7_trigger_deployment_beta": DEFAULT_TRIGGER_DEPLOYMENT_BETA,
        }
    )
    return config


def _job(seed: int, control_config: dict[str, Any]) -> dict[str, Any]:
    run_id = f"A-R2C-D3-S4-V9I-HOLD-s{seed}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v9_phase_i_multi_seed_validation_quarantine",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": DATASET_ID,
        "scenario_id": SCENARIO_ID,
        "rounds": ROUNDS,
        "method_config": _treatment_config(control_config),
        "block_id": "A-R2C-D3-V9-PHASE-I",
        "seed": int(seed),
        "partition_seed": int(seed),
        "trace_seed": int(seed),
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "variant": "telemetry_quarantine_v9",
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
            raise RuntimeError("Refusing to rebuild a started D3 v9 Phase I manifest")

    control_config = _verify_controls(run_audit=persist)
    asset_hashes = v7._ensure_assets()
    jobs = [_job(seed, control_config) for seed in SEEDS]
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_only_v9_phase_i_multi_seed_validation_quarantine",
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
            "duration_ratio": V9_DURATION_RATIO,
            "log_ratio_threshold": V9_LOG_RATIO_THRESHOLD,
            "changed_fraction_threshold": DEFAULT_FRACTION_THRESHOLD,
            "min_comparable_clients": DEFAULT_MIN_COMPARABLE_CLIENTS,
            "cooldown_rounds": DEFAULT_COOLDOWN_ROUNDS,
            "trigger_deployment_beta": DEFAULT_TRIGGER_DEPLOYMENT_BETA,
        },
        "hard_gates": {
            "treatment_trigger_count_per_seed": 1,
            "treatment_trigger_round": EVENT_ROUND,
            "treatment_quarantine_count_per_seed": 1,
            "treatment_quarantine_round": EVENT_ROUND,
            "treatment_synchronization_count_per_seed": 0,
            "trigger_hash_hold_required": True,
            "trigger_global_training_advance_required": True,
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
        atomic_json(QUEUE_ROOT / f"r2c_d3_v9_phase_i_manifest_{stamp}.json", manifest)
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
                "phase_j_authorized": False,
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
        if _audit_run(run_id, "phase_i_reconcile.audit").returncode != 0:
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
        raise RuntimeError("D3 v9 Phase I attempted prohibited access")
    if manifest.get("test_labels_used_for_selection"):
        raise RuntimeError("D3 v9 Phase I attempted test-label selection")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("Implementation hashes changed after D3 v9 Phase I freeze")
    if manifest.get("source_hashes") != _source_hashes():
        raise RuntimeError("Source hashes changed after D3 v9 Phase I freeze")
    if manifest.get("asset_hashes") != v7._ensure_assets():
        raise RuntimeError("Asset hashes changed after D3 v9 Phase I freeze")
    jobs = manifest.get("jobs", [])
    if len(jobs) != len(SEEDS) or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("D3 v9 Phase I job matrix drift")
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
            or job.get("variant") != "telemetry_quarantine_v9"
            or job.get("method_config") != treatment_config
        ):
            raise RuntimeError(f"D3 v9 Phase I protocol drift in {job['job_id']}")


def _quarantine_run_metrics(run_id: str, seed: int) -> dict[str, Any]:
    metrics = v7._run_metrics(run_id, seed, "control")
    run_dir = RUN_ROOT / run_id
    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    rounds = v7._verified_chunked_table(run_dir, "round_metrics").sort_values("round")
    required = {
        "telemetry_shift_trigger",
        "telemetry_shift_labels_used",
        "telemetry_shift_scenario_metadata_used",
        "deployment_synchronization_applied",
        "deployment_quarantine_applied",
        "deployment_shift_response_applied",
        "deployment_trigger_action",
        "configured_trigger_deployment_beta",
        "effective_primary_deployment_beta",
        "primary_deployment_model_hash_before",
        "primary_deployment_model_hash_after",
        "global_model_hash",
    }
    if not required.issubset(rounds.columns):
        missing = sorted(required - set(rounds.columns))
        raise RuntimeError(f"Phase I quarantine audit fields missing in {run_id}: {missing}")

    trigger = rounds["telemetry_shift_trigger"].astype(bool)
    quarantine = rounds["deployment_quarantine_applied"].astype(bool)
    response = rounds["deployment_shift_response_applied"].astype(bool)
    synchronization = rounds["deployment_synchronization_applied"].astype(bool)
    triggered = rounds.loc[trigger]
    trigger_rounds = triggered["round"].astype(int).tolist()
    quarantine_rounds = rounds.loc[quarantine, "round"].astype(int).tolist()
    configured_beta = float(job["method_config"]["r2c_v7_trigger_deployment_beta"])
    primary_beta = float(job["method_config"]["r2c_v4_primary_deployment_beta"])
    effective = rounds["effective_primary_deployment_beta"].astype(float).to_numpy()
    expected_effective = np.where(trigger.to_numpy(), configured_beta, primary_beta)
    expected_action = np.where(trigger.to_numpy(), "hold", "none")
    hash_hold = bool(
        len(triggered) > 0
        and (
            triggered["primary_deployment_model_hash_before"].astype(str).to_numpy()
            == triggered["primary_deployment_model_hash_after"].astype(str).to_numpy()
        ).all()
    )
    global_advanced = True
    for round_number in trigger_rounds:
        if round_number <= 1:
            global_advanced = False
            break
        current = str(
            rounds.loc[rounds["round"].astype(int) == round_number, "global_model_hash"].iloc[0]
        )
        previous = str(
            rounds.loc[
                rounds["round"].astype(int) == round_number - 1, "global_model_hash"
            ].iloc[0]
        )
        if current == previous:
            global_advanced = False
            break

    metrics.update(
        {
            "variant": "telemetry_quarantine_v9",
            "trigger_count": int(trigger.sum()),
            "trigger_rounds_json": canonical_json(trigger_rounds),
            "quarantine_count": int(quarantine.sum()),
            "quarantine_rounds_json": canonical_json(quarantine_rounds),
            "synchronization_count": int(synchronization.sum()),
            "response_count": int(response.sum()),
            "forbidden_input_clean": bool(
                not rounds["telemetry_shift_labels_used"].astype(bool).any()
                and not rounds["telemetry_shift_scenario_metadata_used"].astype(bool).any()
            ),
            "quarantine_matches_trigger": bool(np.array_equal(quarantine, trigger)),
            "response_matches_trigger": bool(np.array_equal(response, trigger)),
            "action_matches_trigger": bool(
                np.array_equal(
                    rounds["deployment_trigger_action"].astype(str).to_numpy(),
                    expected_action,
                )
            ),
            "configured_beta_is_one": bool(
                configured_beta == DEFAULT_TRIGGER_DEPLOYMENT_BETA == 1.0
                and np.allclose(
                    rounds["configured_trigger_deployment_beta"].astype(float).to_numpy(),
                    1.0,
                    atol=0.0,
                    rtol=0.0,
                )
            ),
            "effective_beta_matches_contract": bool(
                np.allclose(effective, expected_effective, atol=0.0, rtol=0.0)
            ),
            "trigger_deployment_hash_held": hash_hold,
            "trigger_global_training_advanced": bool(global_advanced),
        }
    )
    return metrics


def _evaluate_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        subset = frame.loc[frame["seed"].astype(int) == seed]
        if set(subset["variant"].astype(str)) != {"control", "telemetry_quarantine_v9"}:
            raise RuntimeError(f"Incomplete D3 v9 Phase I pair for seed {seed}")
        control = subset.loc[subset["variant"] == "control"].iloc[0]
        treatment = subset.loc[subset["variant"] == "telemetry_quarantine_v9"].iloc[0]
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
                "treatment_quarantine_count": int(treatment["quarantine_count"]),
                "treatment_quarantine_rounds_json": treatment["quarantine_rounds_json"],
                "treatment_synchronization_count": int(treatment["synchronization_count"]),
                "treatment_response_count": int(treatment["response_count"]),
                "quarantine_matches_trigger": bool(treatment["quarantine_matches_trigger"]),
                "response_matches_trigger": bool(treatment["response_matches_trigger"]),
                "action_matches_trigger": bool(treatment["action_matches_trigger"]),
                "configured_beta_is_one": bool(treatment["configured_beta_is_one"]),
                "effective_beta_matches_contract": bool(
                    treatment["effective_beta_matches_contract"]
                ),
                "trigger_deployment_hash_held": bool(
                    treatment["trigger_deployment_hash_held"]
                ),
                "trigger_global_training_advanced": bool(
                    treatment["trigger_global_training_advanced"]
                ),
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
        "quarantine_exactly_once_each_seed": bool(
            (pairs["treatment_quarantine_count"] == 1).all()
        ),
        "quarantine_at_event_each_seed": bool(
            (pairs["treatment_quarantine_rounds_json"] == canonical_json([EVENT_ROUND])).all()
        ),
        "zero_hard_synchronizations_each_seed": bool(
            (pairs["treatment_synchronization_count"] == 0).all()
        ),
        "one_response_each_seed": bool((pairs["treatment_response_count"] == 1).all()),
        "quarantine_matches_trigger": bool(pairs["quarantine_matches_trigger"].all()),
        "response_matches_trigger": bool(pairs["response_matches_trigger"].all()),
        "action_matches_trigger": bool(pairs["action_matches_trigger"].all()),
        "configured_beta_is_one": bool(pairs["configured_beta_is_one"].all()),
        "effective_beta_matches_contract": bool(
            pairs["effective_beta_matches_contract"].all()
        ),
        "trigger_deployment_hash_held": bool(
            pairs["trigger_deployment_hash_held"].all()
        ),
        "trigger_global_training_advanced": bool(
            pairs["trigger_global_training_advanced"].all()
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
        raise RuntimeError("Cannot freeze incomplete D3 v9 Phase I")
    control_config = _verify_controls(run_audit=True)
    treatment_config = _treatment_config(control_config)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        rows.append(v7._run_metrics(CONTROL_RUNS[seed], seed, "control"))
    for job in manifest["jobs"]:
        metrics = _quarantine_run_metrics(str(job["actual_run_id"]), int(job["seed"]))
        if metrics["config_hash"] != config_hash(treatment_config):
            raise RuntimeError(f"D3 v9 config hash mismatch: {job['job_id']}")
        rows.append(metrics)
    frame = pd.DataFrame(rows).sort_values(["seed", "variant"], kind="mergesort")
    pairs, gates = _evaluate_pairs(frame)
    phase_j_authorized = bool(all(gates.values()))
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
        "detector_duration_ratio": V9_DURATION_RATIO,
        "trigger_deployment_beta": DEFAULT_TRIGGER_DEPLOYMENT_BETA,
        "gates": gates,
        "phase_i_passed": phase_j_authorized,
        "phase_j_authorized": phase_j_authorized,
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
                "phase_i_completed_phase_j_authorized"
                if result["phase_j_authorized"]
                else "phase_i_completed_gate_failed"
            ),
            "current_job_id": None,
            "phase_j_authorized": bool(result["phase_j_authorized"]),
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
