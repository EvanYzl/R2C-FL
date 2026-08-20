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
from .r2c_post_event_tail_accuracy20 import derive_pta20_percent
from .r2c_v7 import PROTOCOL_VERSION as V7_PROTOCOL_VERSION
from .r2c_v13 import PROTOCOL_VERSION, SCHEDULES, schedule_lambda, schedule_rounds
from .traces import prepare_trace, trace_asset_path, trace_meta_path
from .utils import atomic_csv, atomic_json, atomic_parquet, config_hash, sha256_file, utc_now


PLAN_ID = "R2C_V13_DARE_20260819_191647"
PLAN_PATH = PROJECT_ROOT / "refine-logs" / "EXPERIMENT_PLAN_20260819_191647.md"
SOURCE_RUN_ID = "A-R2C-D3-S4-V9J-HOLD-s20260810"
SOURCE_RUN_DIR = RUN_ROOT / SOURCE_RUN_ID
CONTROL_RUN_ID = "A-R2C-D3-S4-V12SCREEN-P3-B050-D05-s20260810"
CONTROL_RUN_DIR = RUN_ROOT / CONTROL_RUN_ID

MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v13_validation_screen_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v13_validation_screen_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v13_validation_screen_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v13_validation_screen_result.json"
RUNS_PATH = QUEUE_ROOT / "r2c_d3_v13_validation_screen_runs.parquet"
RUNS_CSV_PATH = QUEUE_ROOT / "r2c_d3_v13_validation_screen_runs.csv"

DATASET_ID = "D3"
SCENARIO_ID = "S4"
SEED = 20260810
ROUNDS = 1000
EVENT_ROUND = 500
TARGET_ACCURACY = 0.7986707616707616
ORDINARY_BETA = 0.9
MAX_ATTEMPTS = 3
DECISION_ATOL = 1e-12

# Immutable candidate order.  No candidate may be added after the screen starts.
CANDIDATES: tuple[str, ...] = ("DARE-L5", "DARE-C5", "DARE-L8")

# Derived once from the already-audited, completed validation control above and
# frozen before any v13 full-budget screen output exists.
CONTROL_METRICS = {
    "s4_last50_accuracy": 0.9045154245154243,
    "s4_pta20_percent": 89.27108927108927,
    "s4_algorithm_tta_s": 211.69667249906342,
}

# Prospectively fixed validation eligibility margins.  They are also the
# denominators used to put all three gains on a common decision scale.
CLOSE_MARGINS = {
    "s4_last50_pp": 0.15,
    "s4_pta20_pp": 0.50,
    "s4_algorithm_tta_percent": 5.0,
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
        "r2c_v13.py",
        "r2c_post_event_tail_accuracy20.py",
        "run.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    tests = Path(__file__).resolve().parents[1] / "tests"
    for name in (
        "audit_r2c_run.py",
        "test_r2c_v13.py",
        "test_r2c_d3_v13_validation_screen_queue.py",
    ):
        values[f"tests/{name}"] = sha256_file(tests / name)
    return values


def _lineage(paths: tuple[Path, ...]) -> dict[str, str]:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _source_lineage() -> dict[str, str]:
    return _lineage(
        (
            PLAN_PATH,
            SOURCE_RUN_DIR / "job.json",
            SOURCE_RUN_DIR / "result.json",
            SOURCE_RUN_DIR / "_SUCCESS.json",
            SOURCE_RUN_DIR / "run_manifest.parquet",
            SOURCE_RUN_DIR / "tables" / "round_metrics" / "_index.json",
        )
    )


def _control_lineage() -> dict[str, str]:
    return _lineage(
        (
            CONTROL_RUN_DIR / "job.json",
            CONTROL_RUN_DIR / "result.json",
            CONTROL_RUN_DIR / "_SUCCESS.json",
            CONTROL_RUN_DIR / "run_manifest.parquet",
            CONTROL_RUN_DIR / "tables" / "round_metrics" / "_index.json",
        )
    )


def _source_config() -> dict[str, Any]:
    job = json.loads((SOURCE_RUN_DIR / "job.json").read_text(encoding="utf-8"))
    if (
        job.get("dataset_id") != DATASET_ID
        or job.get("scenario_id") != SCENARIO_ID
        or job.get("evaluation_split") != "validation"
        or int(job.get("rounds", -1)) != ROUNDS
        or any(int(job.get(key, -1)) != SEED for key in ("seed", "partition_seed", "trace_seed"))
    ):
        raise RuntimeError("v13 validation source scope drift")
    config = dict(job["method_config"])
    if (
        config.get("r2c_protocol_version") != V7_PROTOCOL_VERSION
        or config.get("r2c_v4_deployment_ema_betas") != [ORDINARY_BETA]
        or float(config.get("r2c_v4_primary_deployment_beta", -1.0)) != ORDINARY_BETA
    ):
        raise RuntimeError("v13 validation source configuration drift")
    return config


def _candidate_config(schedule_id: str) -> dict[str, Any]:
    if schedule_id not in CANDIDATES or schedule_id not in SCHEDULES:
        raise ValueError(f"Unregistered DARE schedule: {schedule_id}")
    config = _source_config()
    config.pop("r2c_v7_trigger_deployment_beta", None)
    config.update(
        {
            "r2c_protocol_version": PROTOCOL_VERSION,
            "r2c_v4_deployment_ema_betas": [ORDINARY_BETA],
            "r2c_v4_primary_deployment_beta": ORDINARY_BETA,
            "r2c_v13_schedule_id": schedule_id,
            "r2c_v13_plan_id": PLAN_ID,
        }
    )
    return config


def _ensure_assets() -> dict[str, str]:
    prepare_partition(DATASET_ID, SEED)
    prepare_trace(DATASET_ID, SCENARIO_ID, SEED, rounds=ROUNDS)
    paths = (
        partition_asset_path(DATASET_ID, SEED),
        partition_meta_path(DATASET_ID, SEED),
        trace_asset_path(DATASET_ID, SCENARIO_ID, SEED, ROUNDS),
        trace_meta_path(DATASET_ID, SCENARIO_ID, SEED, ROUNDS),
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _time_to_accuracy(rounds: pd.DataFrame) -> tuple[int | None, float | None]:
    reached = rounds.loc[rounds["test_accuracy"].astype(float) >= TARGET_ACCURACY]
    if reached.empty:
        return None, None
    first = reached.sort_values("round").iloc[0]
    return int(first["round"]), float(first["algorithm_elapsed_s"])


def _derive_control_metrics() -> dict[str, float]:
    rounds = read_chunked_table(CONTROL_RUN_DIR, "round_metrics").sort_values("round")
    if (
        len(rounds) != ROUNDS
        or rounds["round"].astype(int).tolist() != list(range(1, ROUNDS + 1))
        or rounds.loc[rounds["event_offset_round"].astype(int) == 0, "round"].astype(int).tolist()
        != [EVENT_ROUND]
    ):
        raise RuntimeError("validation control round/event contract drift")
    _, tta_s = _time_to_accuracy(rounds)
    if tta_s is None:
        raise RuntimeError("validation control no longer reaches frozen TTA target")
    return {
        "s4_last50_accuracy": float(rounds.tail(50)["test_accuracy"].astype(float).mean()),
        "s4_pta20_percent": float(derive_pta20_percent(rounds)),
        "s4_algorithm_tta_s": float(tta_s),
    }


def _assert_control_metrics() -> None:
    derived = _derive_control_metrics()
    for key, expected in CONTROL_METRICS.items():
        if not np.isclose(float(derived[key]), float(expected), rtol=0.0, atol=1e-12):
            raise RuntimeError(f"validation control metric drift: {key}")


def _job(schedule_id: str) -> dict[str, Any]:
    run_id = f"A-R2C-D3-S4-V13SCREEN-{schedule_id}-s{SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v13_validation_dare_schedule_screen",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": DATASET_ID,
        "scenario_id": SCENARIO_ID,
        "rounds": ROUNDS,
        "method_config": _candidate_config(schedule_id),
        "block_id": "A-R2C-D3-V13-DARE-VALIDATION-SCREEN",
        "seed": SEED,
        "partition_seed": SEED,
        "trace_seed": SEED,
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "candidate_id": schedule_id,
        "schedule_id": schedule_id,
        "recovery_rounds": schedule_rounds(schedule_id),
        "formal_test_access": False,
        "other_dataset_access": False,
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
        "target_accuracy",
        "formal_test_access",
        "other_dataset_access",
        "test_labels_used_for_selection",
        "performance_sealed_until_terminal",
        "ordinary_beta",
        "candidate_order",
        "control_run_id",
        "control_metrics",
        "close_margins",
        "decision_numerical_atol",
        "selection_rule",
        "normalization_rule",
        "completion_rule",
        "source_lineage",
        "control_lineage",
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
            raise RuntimeError("Refusing to rebuild a started v13 validation manifest")
    _assert_control_metrics()
    jobs = [_job(candidate_id) for candidate_id in CANDIDATES]
    if len(jobs) != 3 or len({job["job_id"] for job in jobs}) != 3:
        raise AssertionError("v13 validation screen must contain exactly three unique candidates")
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_validation_only_v13_DARE_schedule_screen",
        "plan_id": PLAN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "selection_split": "validation",
        "seed": SEED,
        "rounds_per_job": ROUNDS,
        "event_round": EVENT_ROUND,
        "target_accuracy": TARGET_ACCURACY,
        "formal_test_access": False,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "performance_sealed_until_terminal": True,
        "ordinary_beta": ORDINARY_BETA,
        "candidate_order": list(CANDIDATES),
        "control_run_id": CONTROL_RUN_ID,
        "control_metrics": CONTROL_METRICS,
        "close_margins": CLOSE_MARGINS,
        "decision_numerical_atol": DECISION_ATOL,
        "selection_rule": (
            "eligible iff normalized gains for S4 last50, PTA20, and algorithm TTA are all >= -1 "
            "within an explicit 1e-12 floating-point comparison tolerance; "
            "select maximum minimum normalized gain, then shorter recovery window, then immutable "
            "candidate order DARE-L5 < DARE-C5 < DARE-L8"
        ),
        "normalization_rule": (
            "last50_gain_pp/0.15; PTA20_gain_pp/0.50; "
            "TTA_relative_improvement_percent/5.0"
        ),
        "completion_rule": "all three full-budget validation candidates complete and audit before any performance read",
        "source_lineage": _source_lineage(),
        "control_lineage": _control_lineage(),
        "asset_hashes": _ensure_assets(),
        "implementation_hashes": _implementation_hashes(),
        "job_order": [job["job_id"] for job in jobs],
        "max_attempts": MAX_ATTEMPTS,
        "jobs": jobs,
    }
    manifest["frozen_spec_hash"] = config_hash(_frozen_spec(manifest))
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        immutable_path = QUEUE_ROOT / f"r2c_d3_v13_validation_screen_manifest_{stamp}.json"
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
                "formal_test_access": False,
                "other_dataset_access": False,
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
        manifest.get("scope") != "D3_validation_only_v13_DARE_schedule_screen"
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
        raise RuntimeError("v13 validation manifest scope drift")
    if manifest.get("control_metrics") != CONTROL_METRICS or manifest.get("close_margins") != CLOSE_MARGINS:
        raise RuntimeError("v13 validation control or margin drift")
    if manifest.get("source_lineage") != _source_lineage():
        raise RuntimeError("v13 validation source lineage drift")
    if manifest.get("control_lineage") != _control_lineage():
        raise RuntimeError("v13 validation control lineage drift")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("v13 validation implementation drift after freeze")
    if manifest.get("frozen_spec_hash") != config_hash(_frozen_spec(manifest)):
        raise RuntimeError("v13 validation frozen specification drift")
    jobs = list(manifest.get("jobs", []))
    if len(jobs) != 3 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("v13 validation candidate order drift")
    for job, schedule_id in zip(jobs, CANDIDATES):
        config = dict(job["method_config"])
        if (
            job.get("candidate_id") != schedule_id
            or job.get("schedule_id") != schedule_id
            or int(job.get("recovery_rounds", -1)) != schedule_rounds(schedule_id)
            or job.get("dataset_id") != DATASET_ID
            or job.get("scenario_id") != SCENARIO_ID
            or job.get("evaluation_split") != "validation"
            or int(job.get("rounds", -1)) != ROUNDS
            or any(int(job.get(key, -1)) != SEED for key in ("seed", "partition_seed", "trace_seed"))
            or config.get("r2c_protocol_version") != PROTOCOL_VERSION
            or config.get("r2c_v4_deployment_ema_betas") != [ORDINARY_BETA]
            or float(config.get("r2c_v4_primary_deployment_beta", -1.0)) != ORDINARY_BETA
            or config.get("r2c_v13_schedule_id") != schedule_id
            or "r2c_v7_trigger_deployment_beta" in config
        ):
            raise RuntimeError(f"v13 validation candidate drift: {job['job_id']}")


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
        if run_id is not None and (job.get("status") != "completed" or job.get("actual_run_id") != run_id):
            job.update({"status": "completed", "actual_run_id": run_id, "failure_reason": None})
            _event(events, job, "reconciled_completed", reason="existing_success_output_and_audit")
            changed += 1
    return changed


def _selection_fields(last50: float, pta20: float, tta_s: float | None) -> dict[str, Any]:
    last50_gain_pp = 100.0 * (float(last50) - CONTROL_METRICS["s4_last50_accuracy"])
    pta_gain_pp = float(pta20) - CONTROL_METRICS["s4_pta20_percent"]
    tta_gain_percent = (
        float("-inf")
        if tta_s is None
        else 100.0 * (CONTROL_METRICS["s4_algorithm_tta_s"] - float(tta_s))
        / CONTROL_METRICS["s4_algorithm_tta_s"]
    )
    normalized = {
        "last50": last50_gain_pp / CLOSE_MARGINS["s4_last50_pp"],
        "pta20": pta_gain_pp / CLOSE_MARGINS["s4_pta20_pp"],
        "tta": tta_gain_percent / CLOSE_MARGINS["s4_algorithm_tta_percent"],
    }
    return {
        "last50_gain_pp": float(last50_gain_pp),
        "pta20_gain_pp": float(pta_gain_pp),
        "tta_gain_percent": float(tta_gain_percent),
        "normalized_last50_gain": float(normalized["last50"]),
        "normalized_pta20_gain": float(normalized["pta20"]),
        "normalized_tta_gain": float(normalized["tta"]),
        "maximin_normalized_gain": float(min(normalized.values())),
        "strict_last50_win": bool(last50_gain_pp > 0.0),
        "strict_pta20_win": bool(pta_gain_pp > 0.0),
        "strict_tta_win": bool(tta_gain_percent > 0.0),
        "validation_eligible": bool(
            all(value >= -1.0 - DECISION_ATOL for value in normalized.values())
        ),
    }


def _run_metrics(job: dict[str, Any]) -> dict[str, Any]:
    run_id = str(job["actual_run_id"])
    run_dir = RUN_ROOT / run_id
    actual_job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    run_manifest = pd.read_parquet(run_dir / "run_manifest.parquet").iloc[0]
    rounds = read_chunked_table(run_dir, "round_metrics").sort_values("round")
    if (
        len(rounds) != ROUNDS
        or rounds["round"].astype(int).tolist() != list(range(1, ROUNDS + 1))
        or str(run_manifest["source_kind"]) != "CALIBRATION"
        or actual_job.get("evaluation_split") != "validation"
        or actual_job.get("method_config") != job.get("method_config")
        or actual_job.get("formal_test_access")
        or actual_job.get("test_labels_used_for_selection")
    ):
        raise RuntimeError(f"v13 validation run contract mismatch: {run_id}")
    event = rounds.loc[rounds["event_offset_round"].astype(int) == 0, "round"].astype(int).tolist()
    if event != [EVENT_ROUND]:
        raise RuntimeError(f"v13 validation event mismatch: {run_id}")
    schedule_id = str(job["schedule_id"])
    duration = schedule_rounds(schedule_id)
    trigger = rounds["telemetry_shift_trigger"].astype(bool)
    hold = rounds["deployment_dare_hold_applied"].astype(bool)
    envelope = rounds["deployment_dare_envelope_applied"].astype(bool)
    tracking = rounds["deployment_dare_tracking_applied"].astype(bool)
    captured = rounds["deployment_dare_pre_anchor_captured"].astype(bool)
    released = rounds["deployment_dare_pre_anchor_released"].astype(bool)
    expected_envelope = list(range(EVENT_ROUND + 1, EVENT_ROUND + duration + 1))
    expected_tracking = list(range(EVENT_ROUND + duration + 1, ROUNDS + 1))
    actual_lambdas = rounds.loc[envelope, "deployment_dare_lambda_value"].astype(float).tolist()
    expected_lambdas = [schedule_lambda(schedule_id, index) for index in range(1, duration + 1)]
    forbidden = (
        rounds["deployment_dare_labels_used"].astype(bool)
        | rounds["deployment_dare_scenario_metadata_used"].astype(bool)
        | rounds["deployment_dare_future_trace_used"].astype(bool)
    )
    tracking_rows = rounds.loc[tracking]
    if (
        rounds.loc[trigger, "round"].astype(int).tolist() != [EVENT_ROUND]
        or rounds.loc[hold, "round"].astype(int).tolist() != [EVENT_ROUND]
        or rounds.loc[envelope, "round"].astype(int).tolist() != expected_envelope
        or rounds.loc[tracking, "round"].astype(int).tolist() != expected_tracking
        or rounds.loc[captured, "round"].astype(int).tolist() != [EVENT_ROUND]
        or rounds.loc[released, "round"].astype(int).tolist() != [EVENT_ROUND + duration]
        or not np.allclose(actual_lambdas, expected_lambdas, rtol=0.0, atol=1e-12)
        or not rounds["deployment_dare_configured_schedule_id"].astype(str).eq(schedule_id).all()
        or forbidden.any()
        or not rounds["deployment_dare_state_server_only"].astype(bool).all()
        or not tracking_rows["evaluation_model_hash"].astype(str).eq(
            tracking_rows["global_model_hash"].astype(str)
        ).all()
    ):
        raise RuntimeError(f"v13 validation DARE lineage mismatch: {run_id}")
    tta_round, tta_s = _time_to_accuracy(rounds)
    last50 = float(rounds.tail(50)["test_accuracy"].astype(float).mean())
    pta20 = float(derive_pta20_percent(rounds))
    decision = _selection_fields(last50, pta20, tta_s)
    return {
        "candidate_id": job["candidate_id"],
        "run_id": run_id,
        "schedule_id": schedule_id,
        "recovery_rounds": duration,
        "last50_validation_accuracy": last50,
        "pta20_percent": pta20,
        "tta_round": tta_round,
        "algorithm_tta_s": tta_s,
        **decision,
        "trigger_count": int(trigger.sum()),
        "hold_count": int(hold.sum()),
        "envelope_count": int(envelope.sum()),
        "tracking_count": int(tracking.sum()),
        "anchor_capture_count": int(captured.sum()),
        "anchor_release_count": int(released.sum()),
        "forbidden_inputs_any": bool(forbidden.any()),
        "test_labels_used": False,
    }


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot select from an incomplete v13 validation screen")
    rows = [_run_metrics(job) for job in manifest["jobs"]]
    frame = pd.DataFrame(rows)
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    eligible = frame.loc[frame["validation_eligible"].astype(bool)].copy()
    selected: dict[str, Any] | None = None
    eligible_ids: list[str] = []
    if not eligible.empty:
        eligible["candidate_order"] = eligible["candidate_id"].map(
            {candidate_id: index for index, candidate_id in enumerate(CANDIDATES)}
        )
        eligible = eligible.sort_values(
            ["maximin_normalized_gain", "recovery_rounds", "candidate_order"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        eligible_ids = eligible["candidate_id"].astype(str).tolist()
        selected = eligible.iloc[0].drop(labels=["candidate_order"]).to_dict()
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "status": "selected" if selected is not None else "no_candidate_eligible",
        "selection_split": "validation",
        "seed": SEED,
        "formal_test_access": False,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "control_run_id": CONTROL_RUN_ID,
        "control_metrics": CONTROL_METRICS,
        "close_margins": CLOSE_MARGINS,
        "selection_rule": manifest["selection_rule"],
        "normalization_rule": manifest["normalization_rule"],
        "selected": selected,
        "eligible_candidate_ids_ranked": eligible_ids,
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
                "m1_completed_selected_m2_required"
                if result["status"] == "selected"
                else "m1_completed_no_candidate_eligible_stop"
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
