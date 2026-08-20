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
from .metrics import recovery_auc20
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
BETAS = (0.90, 0.925, 0.95, 0.975, 0.99)
CONTROL_BETA = 0.90
ROUNDS = 600
EVENT_ROUND = 300
TARGET_ACCURACY = v7.TARGET_ACCURACY
MAX_ATTEMPTS = 3
V9_DURATION_RATIO = 1.25
V9_LOG_RATIO_THRESHOLD = math.log(V9_DURATION_RATIO)

PLAN_PATH = PROJECT_ROOT / "refine-logs" / "D3_V10_ROBUST_DEPLOYMENT_SMOOTHING_PLAN_AMENDMENT_20260818_005854.md"
EXPECTED_PLAN_SHA256 = "78E30CA1F2D80F897E8BE34948D760DFDF869D4135BDB4B51FD082DF8CDB4501"
SOURCE_MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v9_phase_i_manifest.json"
EXPECTED_SOURCE_MANIFEST_SHA256 = "72DC58DF1D370EF2E9F092C34341BF23D70DD456CCC60B0163FFF7EE124980C8"
SOURCE_RESULT_PATH = QUEUE_ROOT / "r2c_d3_v9_phase_i_result.json"
EXPECTED_SOURCE_RESULT_SHA256 = "D661AB833127A18766A1C39D0332A05E2F0A3C6E14C5DCF8BA78D8390CB6E6AE"
SOURCE_PAIRS_PATH = PLOT_ROOT / "r2c_d3_v9_phase_i_pairs.parquet"
EXPECTED_SOURCE_PAIRS_SHA256 = "AA724D70CDDAA9CEDC0BA26833D5793735B2576C22C09023D6C3516098F3C2CE"
PHASE_A_CANDIDATES_PATH = PLOT_ROOT / "r2c_d3_v5_screen_candidates.parquet"
EXPECTED_PHASE_A_CANDIDATES_SHA256 = "0CD51B5B903142650EDDB3261B381258C4A549F2069EC14E32A27A44332A6377"

SOURCE_RUNS: dict[int, str] = {
    20260808: "A-R2C-D3-S4-V9I-HOLD-s20260808",
    20260809: "A-R2C-D3-S4-V9I-HOLD-s20260809",
    20260810: "A-R2C-D3-S4-V9I-HOLD-s20260810",
}
EXPECTED_SOURCE_JOB_HASHES: dict[int, str] = {
    20260808: "68595763F3F850DFEE2EAF2E917DF2195DDD301E13139A1201211802ED1A214B",
    20260809: "8E0CF521101DE3A25B00F3DB769B1ABF530CDF067B52141C4B44710092AF8321",
    20260810: "E1B817F57C857C3FC4FCDE7B4DC64A6EE4DAC7E20A8BD7ADD28DE79DBC7125CE",
}

MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v10_phase_k_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v10_phase_k_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v10_phase_k_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v10_phase_k_result.json"
CANDIDATES_PATH = PLOT_ROOT / "r2c_d3_v10_phase_k_candidates.parquet"
CANDIDATES_CSV_PATH = PLOT_ROOT / "r2c_d3_v10_phase_k_candidates.csv"
AGGREGATES_PATH = PLOT_ROOT / "r2c_d3_v10_phase_k_aggregates.parquet"
AGGREGATES_CSV_PATH = PLOT_ROOT / "r2c_d3_v10_phase_k_aggregates.csv"
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
        str(SOURCE_MANIFEST_PATH.relative_to(PROJECT_ROOT)): sha256_file(SOURCE_MANIFEST_PATH),
        str(SOURCE_RESULT_PATH.relative_to(PROJECT_ROOT)): sha256_file(SOURCE_RESULT_PATH),
        str(SOURCE_PAIRS_PATH.relative_to(PROJECT_ROOT)): sha256_file(SOURCE_PAIRS_PATH),
        str(PHASE_A_CANDIDATES_PATH.relative_to(PROJECT_ROOT)): sha256_file(PHASE_A_CANDIDATES_PATH),
    }
    for run_id in SOURCE_RUNS.values():
        path = RUN_ROOT / run_id / "job.json"
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


def _verify_validation_source(run_audit: bool) -> dict[str, Any]:
    if sha256_file(PLAN_PATH).upper() != EXPECTED_PLAN_SHA256:
        raise RuntimeError("Frozen v10 plan hash drift")
    frozen = (
        (SOURCE_MANIFEST_PATH, EXPECTED_SOURCE_MANIFEST_SHA256),
        (SOURCE_RESULT_PATH, EXPECTED_SOURCE_RESULT_SHA256),
        (SOURCE_PAIRS_PATH, EXPECTED_SOURCE_PAIRS_SHA256),
        (PHASE_A_CANDIDATES_PATH, EXPECTED_PHASE_A_CANDIDATES_SHA256),
    )
    for path, expected in frozen:
        if sha256_file(path).upper() != expected:
            raise RuntimeError(f"Frozen v10 validation source hash drift: {path.name}")

    source_result = json.loads(SOURCE_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        not bool(source_result.get("phase_j_authorized"))
        or source_result.get("selection_split") != "validation"
        or bool(source_result.get("formal_test_access"))
        or bool(source_result.get("test_labels_used"))
    ):
        raise RuntimeError("Frozen v9 Phase-I validation authorization drift")

    phase_a = pd.read_parquet(PHASE_A_CANDIDATES_PATH)
    evidence = phase_a.loc[
        phase_a["alpha"].astype(float).eq(1.0)
        & phase_a["beta"].astype(float).isin([0.90, 0.95])
    ].sort_values("beta")
    if (
        len(evidence) != 2
        or not evidence["complete"].astype(bool).all()
        or not np.allclose(
            evidence["recovery_deficit_auc20"].astype(float).to_numpy(),
            0.0,
            atol=0.0,
            rtol=0.0,
        )
        or float(evidence.iloc[1]["last50_validation_accuracy"])
        <= float(evidence.iloc[0]["last50_validation_accuracy"])
        or int(evidence.iloc[1]["target_hit_round"])
        - int(evidence.iloc[0]["target_hit_round"])
        != 5
    ):
        raise RuntimeError("Frozen pre-formal beta-0.95 validation evidence drift")

    configs: list[dict[str, Any]] = []
    for seed, run_id in SOURCE_RUNS.items():
        run_dir = RUN_ROOT / run_id
        if sha256_file(run_dir / "job.json").upper() != EXPECTED_SOURCE_JOB_HASHES[seed]:
            raise RuntimeError(f"Frozen v9 source-job hash drift: {run_id}")
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
            raise RuntimeError(f"Frozen v9 validation-source contract drift: {run_id}")
        config = dict(job["method_config"])
        if (
            config.get("r2c_protocol_version") != TREATMENT_PROTOCOL_VERSION
            or config.get("r2c_v4_deployment_ema_betas") != [CONTROL_BETA]
            or float(config.get("r2c_v4_primary_deployment_beta")) != CONTROL_BETA
            or float(config.get("r2c_v7_trigger_deployment_beta")) != 1.0
        ):
            raise RuntimeError(f"Frozen v9 deployment configuration drift: {run_id}")
        rounds = v7._verified_chunked_table(run_dir, "round_metrics")
        if rounds["round"].astype(int).tolist() != list(range(1, ROUNDS + 1)):
            raise RuntimeError(f"Frozen v9 source round budget incomplete: {run_id}")
        if run_audit and _audit_run(run_id, "phase_k_source.audit").returncode != 0:
            raise RuntimeError(f"Frozen v9 source audit failed: {run_id}")
        configs.append(config)

    if any(config != configs[0] for config in configs[1:]):
        raise RuntimeError("Frozen v9 sources do not share one method configuration")
    return configs[0]


def _treatment_config(source_config: dict[str, Any]) -> dict[str, Any]:
    config = dict(source_config)
    config["r2c_v4_deployment_ema_betas"] = list(BETAS)
    config["r2c_v4_primary_deployment_beta"] = CONTROL_BETA
    return config


def _job(seed: int, source_config: dict[str, Any]) -> dict[str, Any]:
    run_id = f"A-R2C-D3-S4-V10K-BETAS-s{seed}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v10_phase_k_multi_seed_validation_smoothing_screen",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": DATASET_ID,
        "scenario_id": SCENARIO_ID,
        "rounds": ROUNDS,
        "method_config": _treatment_config(source_config),
        "block_id": "A-R2C-D3-V10-PHASE-K",
        "seed": int(seed),
        "partition_seed": int(seed),
        "trace_seed": int(seed),
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "variant": "robust_deployment_smoothing_v10",
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
            raise RuntimeError("Refusing to rebuild a started D3 v10 Phase K manifest")

    source_config = _verify_validation_source(run_audit=persist)
    asset_hashes = v7._ensure_assets()
    jobs = [_job(seed, source_config) for seed in SEEDS]
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_only_v10_phase_k_multi_seed_validation_smoothing_screen",
        "formal_test_access": False,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "rounds_per_job": ROUNDS,
        "event_round": EVENT_ROUND,
        "development_seeds": list(SEEDS),
        "new_job_count": len(jobs),
        "candidate_betas": list(BETAS),
        "within_trajectory_control_beta": CONTROL_BETA,
        "source_v9_run_ids": {str(seed): SOURCE_RUNS[seed] for seed in SEEDS},
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
            "trigger_count_per_seed": 1,
            "trigger_round": EVENT_ROUND,
            "quarantine_count_per_seed": 1,
            "quarantine_round": EVENT_ROUND,
            "synchronization_count_per_seed": 0,
            "trigger_hash_hold_required": True,
            "trigger_global_training_advance_required": True,
            "mean_auc_no_higher": True,
            "worst_auc_no_higher": True,
            "mean_or_worst_auc_strictly_lower": True,
            "auc_no_worse_seed_count_min": 2,
            "max_per_seed_last50_loss_pp": 0.25,
            "max_mean_last50_loss_pp": 0.15,
            "max_mean_target_hit_delay_rounds": 10,
            "max_per_seed_target_hit_delay_rounds": 20,
        },
        "selection_rule": (
            "eligible candidates ordered by worst AUC asc, mean AUC asc, mean Last50 desc, "
            "mean target-hit round asc, beta asc; exactly one winner"
        ),
        "implementation_hashes": _implementation_hashes(),
        "source_hashes": _source_hashes(),
        "asset_hashes": asset_hashes,
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d3_v10_phase_k_manifest_{stamp}.json", manifest)
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
                "phase_l_authorized": False,
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
        if _audit_run(run_id, "phase_k_reconcile.audit").returncode != 0:
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
        raise RuntimeError("D3 v10 Phase K attempted prohibited access")
    if manifest.get("test_labels_used_for_selection"):
        raise RuntimeError("D3 v10 Phase K attempted test-label selection")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("Implementation hashes changed after D3 v10 Phase K freeze")
    if manifest.get("source_hashes") != _source_hashes():
        raise RuntimeError("Source hashes changed after D3 v10 Phase K freeze")
    if manifest.get("asset_hashes") != v7._ensure_assets():
        raise RuntimeError("Asset hashes changed after D3 v10 Phase K freeze")
    jobs = manifest.get("jobs", [])
    if len(jobs) != len(SEEDS) or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("D3 v10 Phase K job matrix drift")
    source_config = _verify_validation_source(run_audit=False)
    treatment_config = _treatment_config(source_config)
    for job, seed in zip(jobs, SEEDS):
        if (
            job.get("dataset_id") != DATASET_ID
            or job.get("scenario_id") != SCENARIO_ID
            or job.get("evaluation_split") != "validation"
            or int(job.get("rounds", -1)) != ROUNDS
            or not bool(job.get("full_logging"))
            or any(int(job[key]) != seed for key in ("seed", "partition_seed", "trace_seed"))
            or job.get("variant") != "robust_deployment_smoothing_v10"
            or job.get("method_config") != treatment_config
        ):
            raise RuntimeError(f"D3 v10 Phase K protocol drift in {job['job_id']}")


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
        raise RuntimeError(f"Phase K quarantine audit fields missing in {run_id}: {missing}")

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
            "variant": "robust_deployment_smoothing_v10",
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


def _candidate_metrics(run_id: str, seed: int) -> list[dict[str, Any]]:
    run_dir = RUN_ROOT / run_id
    rounds = v7._verified_chunked_table(run_dir, "round_metrics").sort_values("round")
    candidates = v7._verified_chunked_table(run_dir, "deployment_candidate_metrics")
    required = {
        "round",
        "deployment_beta",
        "effective_deployment_beta",
        "deployment_synchronization_applied",
        "deployment_quarantine_applied",
        "deployment_shift_response_applied",
        "deployment_trigger_action",
        "configured_trigger_deployment_beta",
        "deployment_model_hash_before",
        "deployment_model_hash_after",
        "is_primary",
        "test_accuracy",
    }
    if not required.issubset(candidates.columns):
        raise RuntimeError(f"Phase K deployment fields missing in {run_id}")
    if len(candidates) != ROUNDS * len(BETAS):
        raise RuntimeError(f"Phase K deployment-row budget drift in {run_id}")
    if candidates.duplicated(["round", "deployment_beta"]).any():
        raise RuntimeError(f"Phase K duplicate round/beta key in {run_id}")
    observed_betas = tuple(sorted(candidates["deployment_beta"].astype(float).unique()))
    if observed_betas != tuple(sorted(BETAS)):
        raise RuntimeError(f"Phase K candidate-beta drift in {run_id}: {observed_betas}")

    rows: list[dict[str, Any]] = []
    for beta, group in candidates.groupby("deployment_beta", sort=True):
        beta = float(beta)
        group = group.sort_values("round")
        if group["round"].astype(int).tolist() != list(range(1, ROUNDS + 1)):
            raise RuntimeError(f"Phase K incomplete beta trajectory: {run_id}/{beta}")
        event = group["round"].astype(int).eq(EVENT_ROUND)
        expected_effective = np.where(event.to_numpy(), 1.0, beta)
        expected_action = np.where(event.to_numpy(), "hold", "none")
        if (
            not np.allclose(
                group["effective_deployment_beta"].astype(float).to_numpy(),
                expected_effective,
                atol=0.0,
                rtol=0.0,
            )
            or not group["deployment_quarantine_applied"].astype(bool).equals(event)
            or not group["deployment_shift_response_applied"].astype(bool).equals(event)
            or group["deployment_synchronization_applied"].astype(bool).any()
            or not np.array_equal(
                group["deployment_trigger_action"].astype(str).to_numpy(), expected_action
            )
            or not np.allclose(
                group["configured_trigger_deployment_beta"].astype(float).to_numpy(),
                1.0,
                atol=0.0,
                rtol=0.0,
            )
        ):
            raise RuntimeError(f"Phase K candidate action/beta contract drift: {run_id}/{beta}")
        event_row = group.loc[event].iloc[0]
        if str(event_row["deployment_model_hash_before"]) != str(
            event_row["deployment_model_hash_after"]
        ):
            raise RuntimeError(f"Phase K trigger hash hold failed: {run_id}/{beta}")
        expected_primary = beta == CONTROL_BETA
        if not bool((group["is_primary"].astype(bool) == expected_primary).all()):
            raise RuntimeError(f"Phase K primary-candidate lineage drift: {run_id}/{beta}")
        if expected_primary and not np.allclose(
            group["test_accuracy"].astype(float).to_numpy(),
            rounds["test_accuracy"].astype(float).to_numpy(),
            atol=0.0,
            rtol=0.0,
        ):
            raise RuntimeError(f"Phase K primary candidate/round mismatch: {run_id}")

        recovery = recovery_auc20(
            group["round"].astype(int).tolist(),
            group["test_accuracy"].astype(float).tolist(),
            EVENT_ROUND,
        )
        if not bool(recovery["recovery_auc20_complete"]):
            raise RuntimeError(f"Phase K strict AUC incomplete: {run_id}/{beta}")
        merged = group[["round", "test_accuracy"]].merge(
            rounds[["round", "algorithm_elapsed_s"]], on="round", validate="one_to_one"
        )
        reached = merged.loc[merged["test_accuracy"].astype(float) >= TARGET_ACCURACY]
        rows.append(
            {
                "schema_version": "1.0.0",
                "run_id": run_id,
                "seed": int(seed),
                "beta": beta,
                "last50_validation_accuracy": float(
                    group.tail(50)["test_accuracy"].astype(float).mean()
                ),
                "recovery_deficit_auc20": float(recovery["recovery_deficit_auc20"]),
                "recovery_auc20_complete": True,
                "target_hit_round": None if reached.empty else int(reached.iloc[0]["round"]),
                "shared_algorithm_tta_s": (
                    None if reached.empty else float(reached.iloc[0]["algorithm_elapsed_s"])
                ),
                "complete": True,
                "test_labels_used": False,
            }
        )
    return rows


def _structure_gates(structure: pd.DataFrame) -> dict[str, bool]:
    return {
        "trigger_exactly_once_each_seed": bool((structure["trigger_count"] == 1).all()),
        "trigger_at_event_each_seed": bool(
            (structure["trigger_rounds_json"] == canonical_json([EVENT_ROUND])).all()
        ),
        "quarantine_exactly_once_each_seed": bool((structure["quarantine_count"] == 1).all()),
        "quarantine_at_event_each_seed": bool(
            (structure["quarantine_rounds_json"] == canonical_json([EVENT_ROUND])).all()
        ),
        "zero_hard_synchronizations_each_seed": bool(
            (structure["synchronization_count"] == 0).all()
        ),
        "one_response_each_seed": bool((structure["response_count"] == 1).all()),
        "quarantine_matches_trigger": bool(structure["quarantine_matches_trigger"].all()),
        "response_matches_trigger": bool(structure["response_matches_trigger"].all()),
        "action_matches_trigger": bool(structure["action_matches_trigger"].all()),
        "configured_beta_is_one": bool(structure["configured_beta_is_one"].all()),
        "effective_beta_matches_contract": bool(
            structure["effective_beta_matches_contract"].all()
        ),
        "trigger_deployment_hash_held": bool(
            structure["trigger_deployment_hash_held"].all()
        ),
        "trigger_global_training_advanced": bool(
            structure["trigger_global_training_advanced"].all()
        ),
        "forbidden_input_clean": bool(structure["forbidden_input_clean"].all()),
    }


def _aggregate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    control = (
        frame.loc[frame["beta"].astype(float).eq(CONTROL_BETA)]
        .sort_values("seed")
        .set_index("seed")
    )
    if control.index.astype(int).tolist() != list(SEEDS):
        raise RuntimeError("Phase K within-trajectory control matrix incomplete")
    rows: list[dict[str, Any]] = []
    control_mean_auc = float(control["recovery_deficit_auc20"].astype(float).mean())
    control_worst_auc = float(control["recovery_deficit_auc20"].astype(float).max())
    for beta in BETAS:
        subset = frame.loc[frame["beta"].astype(float).eq(beta)].sort_values("seed").set_index("seed")
        if subset.index.astype(int).tolist() != list(SEEDS):
            raise RuntimeError(f"Phase K beta matrix incomplete: {beta}")
        auc_delta = subset["recovery_deficit_auc20"].astype(float) - control[
            "recovery_deficit_auc20"
        ].astype(float)
        last50_loss_pp = 100.0 * (
            control["last50_validation_accuracy"].astype(float)
            - subset["last50_validation_accuracy"].astype(float)
        )
        control_hit = control["target_hit_round"]
        candidate_hit = subset["target_hit_round"]
        target_all = bool(control_hit.notna().all() and candidate_hit.notna().all())
        delays = (
            candidate_hit.astype(float) - control_hit.astype(float)
            if target_all
            else pd.Series([math.inf] * len(SEEDS), index=subset.index)
        )
        mean_auc = float(subset["recovery_deficit_auc20"].astype(float).mean())
        worst_auc = float(subset["recovery_deficit_auc20"].astype(float).max())
        mean_or_worst_strict = bool(
            mean_auc < control_mean_auc - 1.0e-15
            or worst_auc < control_worst_auc - 1.0e-15
            or (control_mean_auc == 0.0 and control_worst_auc == 0.0)
        )
        gates = {
            "auc_no_worse_at_least_two_seeds": bool((auc_delta <= 1.0e-15).sum() >= 2),
            "mean_auc_no_higher": bool(mean_auc <= control_mean_auc + 1.0e-15),
            "worst_auc_no_higher": bool(worst_auc <= control_worst_auc + 1.0e-15),
            "mean_or_worst_auc_strictly_lower": mean_or_worst_strict,
            "per_seed_last50_loss_within_0p25pp": bool((last50_loss_pp <= 0.25).all()),
            "mean_last50_loss_within_0p15pp": bool(last50_loss_pp.mean() <= 0.15),
            "target_reached_all_seeds": target_all,
            "mean_target_hit_delay_within_10_rounds": bool(
                target_all and float(delays.mean()) <= 10.0
            ),
            "per_seed_target_hit_delay_within_20_rounds": bool(
                target_all and (delays <= 20.0).all()
            ),
        }
        eligible = bool(beta != CONTROL_BETA and all(gates.values()))
        row: dict[str, Any] = {
            "schema_version": "1.0.0",
            "beta": float(beta),
            "mean_auc20": mean_auc,
            "worst_auc20": worst_auc,
            "mean_last50_validation_accuracy": float(
                subset["last50_validation_accuracy"].astype(float).mean()
            ),
            "mean_target_hit_round": (
                None if not target_all else float(candidate_hit.astype(float).mean())
            ),
            "auc_no_worse_seed_count": int((auc_delta <= 1.0e-15).sum()),
            "mean_last50_loss_pp": float(last50_loss_pp.mean()),
            "max_last50_loss_pp": float(last50_loss_pp.max()),
            "mean_target_hit_delay_rounds": None if not target_all else float(delays.mean()),
            "max_target_hit_delay_rounds": None if not target_all else float(delays.max()),
            "eligible": eligible,
        }
        row.update(gates)
        rows.append(row)
    return pd.DataFrame(rows)


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot freeze incomplete D3 v10 Phase K")
    source_config = _verify_validation_source(run_audit=False)
    treatment_config = _treatment_config(source_config)
    structures: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run_id = str(job["actual_run_id"])
        seed = int(job["seed"])
        metrics = _quarantine_run_metrics(run_id, seed)
        if metrics["config_hash"] != config_hash(treatment_config):
            raise RuntimeError(f"D3 v10 config hash mismatch: {job['job_id']}")
        structures.append(metrics)
        candidates.extend(_candidate_metrics(run_id, seed))
    structure = pd.DataFrame(structures).sort_values("seed")
    structure_checks = _structure_gates(structure)
    frame = pd.DataFrame(candidates).sort_values(["beta", "seed"], kind="mergesort")
    if len(frame) != len(SEEDS) * len(BETAS):
        raise RuntimeError("D3 v10 candidate matrix size drift")
    aggregates = _aggregate_candidates(frame)
    eligible = aggregates.loc[aggregates["eligible"].astype(bool)].sort_values(
        [
            "worst_auc20",
            "mean_auc20",
            "mean_last50_validation_accuracy",
            "mean_target_hit_round",
            "beta",
        ],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    )
    phase_l_authorized = bool(all(structure_checks.values()) and not eligible.empty)
    winner: dict[str, Any] | None = None
    if phase_l_authorized:
        selected = eligible.iloc[0]
        winner_config = dict(source_config)
        winner_config["r2c_v4_deployment_ema_betas"] = [float(selected["beta"])]
        winner_config["r2c_v4_primary_deployment_beta"] = float(selected["beta"])
        winner = {
            "beta": float(selected["beta"]),
            "mean_auc20": float(selected["mean_auc20"]),
            "worst_auc20": float(selected["worst_auc20"]),
            "mean_last50_validation_accuracy": float(
                selected["mean_last50_validation_accuracy"]
            ),
            "mean_target_hit_round": float(selected["mean_target_hit_round"]),
            "method_config": winner_config,
            "config_hash": config_hash(winner_config),
        }
    atomic_parquet(CANDIDATES_PATH, frame)
    atomic_csv(CANDIDATES_CSV_PATH, frame)
    atomic_parquet(AGGREGATES_PATH, aggregates)
    atomic_csv(AGGREGATES_CSV_PATH, aggregates)
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "dataset_id": DATASET_ID,
        "scenario_id": SCENARIO_ID,
        "selection_split": "validation",
        "formal_test_access": False,
        "test_labels_used": False,
        "development_seeds": list(SEEDS),
        "candidate_betas": list(BETAS),
        "new_job_count": len(manifest["jobs"]),
        "candidate_count": len(frame),
        "aggregate_count": len(aggregates),
        "detector_duration_ratio": V9_DURATION_RATIO,
        "trigger_deployment_beta": DEFAULT_TRIGGER_DEPLOYMENT_BETA,
        "structure_gates": structure_checks,
        "phase_k_passed": phase_l_authorized,
        "phase_l_authorized": phase_l_authorized,
        "winner": winner,
        "candidates_path": str(CANDIDATES_PATH),
        "aggregates_path": str(AGGREGATES_PATH),
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
                "phase_k_completed_phase_l_authorized"
                if result["phase_l_authorized"]
                else "phase_k_completed_gate_failed"
            ),
            "current_job_id": None,
            "phase_l_authorized": bool(result["phase_l_authorized"]),
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
