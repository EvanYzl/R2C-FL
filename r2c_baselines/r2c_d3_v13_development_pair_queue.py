from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import r2c_d3_v13_validation_screen_queue as screen
from .config import PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .data import partition_asset_path, partition_meta_path, prepare_partition
from .logging_io import read_chunked_table
from .r2c_d3_v7_phase_e_queue import _audit_run
from .r2c_post_event_tail_accuracy20 import derive_pta20_percent
from .r2c_v7 import PROTOCOL_VERSION as V11_PROTOCOL_VERSION
from .r2c_v13 import PROTOCOL_VERSION, schedule_lambda, schedule_rounds
from .traces import prepare_trace, trace_asset_path, trace_meta_path
from .utils import atomic_csv, atomic_json, atomic_parquet, config_hash, sha256_file, utc_now


PLAN_ID = screen.PLAN_ID
PLAN_PATH = screen.PLAN_PATH
SCREEN_MANIFEST_PATH = screen.MANIFEST_PATH
SCREEN_STATE_PATH = screen.STATE_PATH
SCREEN_RESULT_PATH = screen.RESULT_PATH
SCREEN_RUNS_PATH = screen.RUNS_PATH
SCREEN_RUNS_CSV_PATH = screen.RUNS_CSV_PATH

V11_SOURCE_RUN_ID = "A-R2C-V11MS-D3-S4-B095-s20260811"
V11_SOURCE_RUN_DIR = RUN_ROOT / V11_SOURCE_RUN_ID

MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v13_development_pair_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v13_development_pair_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v13_development_pair_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v13_development_pair_result.json"
RUNS_PATH = QUEUE_ROOT / "r2c_d3_v13_development_pair_runs.parquet"
RUNS_CSV_PATH = QUEUE_ROOT / "r2c_d3_v13_development_pair_runs.csv"

DATASET_ID = "D3"
SCENARIOS = ("S0", "S4")
METHOD_ORDER = (("v11", "S0"), ("v11", "S4"), ("v13", "S0"), ("v13", "S4"))
SEED = 20260809
ROUNDS = 1000
EVENT_ROUND = 500
TARGET_ACCURACY = screen.TARGET_ACCURACY
ORDINARY_BETA = 0.95
TRIGGER_BETA = 1.0
MAX_ATTEMPTS = 3
DECISION_ATOL = 1e-12

GATE_MARGINS = {
    "s0_last50_pp": 0.15,
    "s4_last50_pp": 0.15,
    "s4_pta20_pp": 0.50,
    "s4_algorithm_tta_relative": 0.05,
}

IDENTITY_COLUMNS = (
    "round",
    "global_model_hash",
    "evaluation_model_hash",
    "primary_deployment_model_hash_before",
    "primary_deployment_model_hash_after",
)


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
        "r2c_d3_v13_validation_screen_queue.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    tests = Path(__file__).resolve().parents[1] / "tests"
    for name in (
        "audit_r2c_run.py",
        "test_r2c_v13.py",
        "test_r2c_d3_v13_validation_screen_queue.py",
        "test_r2c_d3_v13_development_pair_queue.py",
    ):
        values[f"tests/{name}"] = sha256_file(tests / name)
    return values


def _screen_context() -> dict[str, Any]:
    for path in (SCREEN_MANIFEST_PATH, SCREEN_STATE_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(SCREEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(SCREEN_STATE_PATH.read_text(encoding="utf-8"))
    if not SCREEN_RESULT_PATH.exists():
        raise RuntimeError("v13 M1 screen is not terminal with a selected candidate")
    result = json.loads(SCREEN_RESULT_PATH.read_text(encoding="utf-8"))
    screen._assert_frozen_manifest(manifest)
    if (
        state.get("status") != "m1_completed_selected_m2_required"
        or int(state.get("completed", -1)) != 3
        or int(state.get("failed", -1)) != 0
        or not bool(state.get("all_runs_completed"))
        or bool(state.get("performance_sealed_until_terminal"))
        or result.get("status") != "selected"
        or result.get("selected") is None
        or result.get("selection_split") != "validation"
        or bool(result.get("formal_test_access"))
        or bool(result.get("other_dataset_access"))
        or bool(result.get("test_labels_used_for_selection"))
    ):
        raise RuntimeError("v13 M1 screen is not terminal with a selected candidate")
    immutable_path = Path(str(state["immutable_manifest_path"])).resolve()
    if (
        not immutable_path.exists()
        or sha256_file(immutable_path).lower() != str(state["immutable_manifest_sha256"]).lower()
        or str(state.get("frozen_spec_hash")) != str(manifest.get("frozen_spec_hash"))
        or str(result.get("frozen_spec_hash")) != str(manifest.get("frozen_spec_hash"))
    ):
        raise RuntimeError("v13 M1 immutable lineage mismatch")
    selected = dict(result["selected"])
    schedule_id = str(selected.get("candidate_id"))
    if schedule_id not in screen.CANDIDATES or not bool(selected.get("validation_eligible")):
        raise RuntimeError("v13 M1 selected candidate is not eligible")
    jobs = [job for job in manifest["jobs"] if job.get("candidate_id") == schedule_id]
    if len(jobs) != 1 or jobs[0].get("status") != "completed" or not jobs[0].get("actual_run_id"):
        raise RuntimeError("v13 M1 selected job is not uniquely completed")
    selected_job = dict(jobs[0])
    run_id = str(selected_job["actual_run_id"])
    audit_log = QUEUE_ROOT / "worker_logs" / f"{run_id}.audit.log"
    if not audit_log.exists() or not (RUN_ROOT / run_id / "_SUCCESS.json").exists():
        raise RuntimeError("v13 M1 selected audit/success evidence is missing")
    return {
        "manifest": manifest,
        "state": state,
        "result": result,
        "immutable_path": immutable_path,
        "selected": selected,
        "selected_job": selected_job,
        "selected_run_id": run_id,
        "schedule_id": schedule_id,
        "audit_log": audit_log,
    }


def _source_lineage(context: dict[str, Any]) -> dict[str, str]:
    selected_run_dir = RUN_ROOT / str(context["selected_run_id"])
    paths = (
        PLAN_PATH,
        Path(context["immutable_path"]),
        SCREEN_MANIFEST_PATH,
        SCREEN_STATE_PATH,
        SCREEN_RESULT_PATH,
        SCREEN_RUNS_PATH,
        SCREEN_RUNS_CSV_PATH,
        Path(context["audit_log"]),
        selected_run_dir / "job.json",
        selected_run_dir / "result.json",
        selected_run_dir / "_SUCCESS.json",
        selected_run_dir / "run_manifest.parquet",
        selected_run_dir / "tables" / "round_metrics" / "_index.json",
        V11_SOURCE_RUN_DIR / "job.json",
        V11_SOURCE_RUN_DIR / "_SUCCESS.json",
        V11_SOURCE_RUN_DIR / "run_manifest.parquet",
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _v11_config() -> dict[str, Any]:
    job = json.loads((V11_SOURCE_RUN_DIR / "job.json").read_text(encoding="utf-8"))
    if (
        job.get("dataset_id") != DATASET_ID
        or job.get("scenario_id") != "S4"
        or int(job.get("rounds", -1)) != ROUNDS
        or job.get("evaluation_split") != "test"
    ):
        raise RuntimeError("v11 matched source scope drift")
    config = dict(job["method_config"])
    if (
        config.get("r2c_protocol_version") != V11_PROTOCOL_VERSION
        or config.get("r2c_v4_deployment_ema_betas") != [ORDINARY_BETA]
        or float(config.get("r2c_v4_primary_deployment_beta", -1.0)) != ORDINARY_BETA
        or float(config.get("r2c_v7_trigger_deployment_beta", -1.0)) != TRIGGER_BETA
    ):
        raise RuntimeError("v11 matched source configuration drift")
    config["r2c_v2_audit_replay"] = False
    return config


def _v13_config(schedule_id: str) -> dict[str, Any]:
    if schedule_id not in screen.CANDIDATES:
        raise ValueError(f"Unregistered selected schedule: {schedule_id}")
    config = _v11_config()
    config.pop("r2c_v7_trigger_deployment_beta", None)
    config.update(
        {
            "r2c_protocol_version": PROTOCOL_VERSION,
            "r2c_v13_schedule_id": schedule_id,
            "r2c_v13_plan_id": PLAN_ID,
        }
    )
    return config


def _ensure_assets() -> dict[str, str]:
    prepare_partition(DATASET_ID, SEED)
    paths = [partition_asset_path(DATASET_ID, SEED), partition_meta_path(DATASET_ID, SEED)]
    for scenario in SCENARIOS:
        prepare_trace(DATASET_ID, scenario, SEED, rounds=ROUNDS)
        paths.extend(
            (
                trace_asset_path(DATASET_ID, scenario, SEED, ROUNDS),
                trace_meta_path(DATASET_ID, scenario, SEED, ROUNDS),
            )
        )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _job(method_version: str, scenario: str, context: dict[str, Any]) -> dict[str, Any]:
    if (method_version, scenario) not in METHOD_ORDER:
        raise ValueError("job is outside immutable M2 method/scenario order")
    schedule_id = str(context["schedule_id"])
    method_label = "V11-B095" if method_version == "v11" else f"V13-{schedule_id}-B095"
    run_id = f"A-R2C-D3-{scenario}-V13M2-{method_label}-s{SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v13_independent_seed_matched_development_pair",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "method_version": method_version,
        "dataset_id": DATASET_ID,
        "scenario_id": scenario,
        "rounds": ROUNDS,
        "method_config": _v11_config() if method_version == "v11" else _v13_config(schedule_id),
        "block_id": "A-R2C-D3-V13-M2-MATCHED-DEVELOPMENT",
        "seed": SEED,
        "partition_seed": SEED,
        "trace_seed": SEED,
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "selected_schedule_id": schedule_id,
        "source_screen_run_id": str(context["selected_run_id"]),
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
        "comparator_protocol_version",
        "selection_split",
        "seed",
        "rounds_per_job",
        "event_round",
        "target_accuracy",
        "ordinary_beta",
        "trigger_beta",
        "formal_test_access",
        "other_dataset_access",
        "test_labels_used_for_selection",
        "candidate_locked_before_pair",
        "performance_sealed_until_terminal",
        "selected_candidate",
        "gate_metrics",
        "gate_margins",
        "gate_rule",
        "identity_columns",
        "decision_numerical_atol",
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
            raise RuntimeError("Refusing to rebuild a started v13 M2 manifest")
    context = _screen_context()
    jobs = [_job(method, scenario, context) for method, scenario in METHOD_ORDER]
    if [(job["method_version"], job["scenario_id"]) for job in jobs] != list(METHOD_ORDER):
        raise AssertionError("v13 M2 order must be v11 S0/S4 then v13 S0/S4")
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_validation_seed20260809_v11_v13_matched_pair",
        "plan_id": PLAN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "comparator_protocol_version": V11_PROTOCOL_VERSION,
        "selection_split": "validation",
        "seed": SEED,
        "rounds_per_job": ROUNDS,
        "event_round": EVENT_ROUND,
        "target_accuracy": TARGET_ACCURACY,
        "ordinary_beta": ORDINARY_BETA,
        "trigger_beta": TRIGGER_BETA,
        "formal_test_access": False,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "candidate_locked_before_pair": True,
        "performance_sealed_until_terminal": True,
        "selected_candidate": {
            "schedule_id": str(context["schedule_id"]),
            "source_screen_run_id": str(context["selected_run_id"]),
            "source_frozen_spec_hash": str(context["manifest"]["frozen_spec_hash"]),
        },
        "gate_metrics": (
            "S0 last50 accuracy; S4 last50 accuracy; S4 PTA20; S4 algorithm TTA"
        ),
        "gate_margins": GATE_MARGINS,
        "gate_rule": (
            "pass iff four strict wins, or exactly three strict wins and the sole miss is close: "
            "accuracy no worse than 0.15 pp, PTA20 no worse than 0.50 pp, or TTA no worse "
            "than 5 percent relative; S0 hash identity is an additional mandatory structural gate"
        ),
        "identity_columns": list(IDENTITY_COLUMNS),
        "decision_numerical_atol": DECISION_ATOL,
        "source_lineage": _source_lineage(context),
        "asset_hashes": _ensure_assets(),
        "implementation_hashes": _implementation_hashes(),
        "job_order": [job["job_id"] for job in jobs],
        "max_attempts": MAX_ATTEMPTS,
        "jobs": jobs,
    }
    manifest["frozen_spec_hash"] = config_hash(_frozen_spec(manifest))
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        immutable_path = QUEUE_ROOT / f"r2c_d3_v13_development_pair_manifest_{stamp}.json"
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
    context = _screen_context()
    expected_selected = {
        "schedule_id": str(context["schedule_id"]),
        "source_screen_run_id": str(context["selected_run_id"]),
        "source_frozen_spec_hash": str(context["manifest"]["frozen_spec_hash"]),
    }
    if (
        manifest.get("scope") != "D3_validation_seed20260809_v11_v13_matched_pair"
        or manifest.get("plan_id") != PLAN_ID
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("comparator_protocol_version") != V11_PROTOCOL_VERSION
        or manifest.get("selection_split") != "validation"
        or int(manifest.get("seed", -1)) != SEED
        or int(manifest.get("rounds_per_job", -1)) != ROUNDS
        or float(manifest.get("ordinary_beta", -1.0)) != ORDINARY_BETA
        or bool(manifest.get("formal_test_access"))
        or bool(manifest.get("other_dataset_access"))
        or bool(manifest.get("test_labels_used_for_selection"))
        or not bool(manifest.get("candidate_locked_before_pair"))
        or not bool(manifest.get("performance_sealed_until_terminal"))
        or manifest.get("selected_candidate") != expected_selected
        or manifest.get("gate_margins") != GATE_MARGINS
        or manifest.get("identity_columns") != list(IDENTITY_COLUMNS)
    ):
        raise RuntimeError("v13 M2 manifest scope drift")
    if manifest.get("source_lineage") != _source_lineage(context):
        raise RuntimeError("v13 M2 source lineage drift")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("v13 M2 implementation drift after freeze")
    if manifest.get("frozen_spec_hash") != config_hash(_frozen_spec(manifest)):
        raise RuntimeError("v13 M2 frozen specification drift")
    jobs = list(manifest.get("jobs", []))
    if len(jobs) != 4 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("v13 M2 job order drift")
    for job, (method_version, scenario) in zip(jobs, METHOD_ORDER):
        expected_config = _v11_config() if method_version == "v11" else _v13_config(context["schedule_id"])
        if (
            job.get("method_version") != method_version
            or job.get("scenario_id") != scenario
            or job.get("dataset_id") != DATASET_ID
            or job.get("evaluation_split") != "validation"
            or int(job.get("rounds", -1)) != ROUNDS
            or any(int(job.get(key, -1)) != SEED for key in ("seed", "partition_seed", "trace_seed"))
            or job.get("method_config") != expected_config
            or job.get("selected_schedule_id") != context["schedule_id"]
            or bool(job.get("formal_test_access"))
            or bool(job.get("other_dataset_access"))
        ):
            raise RuntimeError(f"v13 M2 job drift: {job.get('job_id')}")


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
            audit_log = QUEUE_ROOT / "worker_logs" / f"{path.name}.m2.reconcile.audit.log"
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


def _time_to_accuracy(rounds: pd.DataFrame) -> tuple[int | None, float | None]:
    reached = rounds.loc[rounds["test_accuracy"].astype(float) >= TARGET_ACCURACY]
    if reached.empty:
        return None, None
    first = reached.sort_values("round").iloc[0]
    return int(first["round"]), float(first["algorithm_elapsed_s"])


def _audit_v13_contract(rounds: pd.DataFrame, scenario: str, schedule_id: str, run_id: str) -> None:
    trigger = rounds["telemetry_shift_trigger"].astype(bool)
    hold = rounds["deployment_dare_hold_applied"].astype(bool)
    envelope = rounds["deployment_dare_envelope_applied"].astype(bool)
    tracking = rounds["deployment_dare_tracking_applied"].astype(bool)
    captured = rounds["deployment_dare_pre_anchor_captured"].astype(bool)
    released = rounds["deployment_dare_pre_anchor_released"].astype(bool)
    forbidden = (
        rounds["deployment_dare_labels_used"].astype(bool)
        | rounds["deployment_dare_scenario_metadata_used"].astype(bool)
        | rounds["deployment_dare_future_trace_used"].astype(bool)
    )
    if forbidden.any() or not rounds["deployment_dare_state_server_only"].astype(bool).all():
        raise RuntimeError(f"v13 M2 forbidden DARE input: {run_id}")
    if scenario == "S0":
        if (
            trigger.any()
            or hold.any()
            or envelope.any()
            or tracking.any()
            or captured.any()
            or released.any()
            or not rounds["deployment_dare_phase"].astype(str).eq("ordinary").all()
        ):
            raise RuntimeError(f"v13 M2 S0 no-trigger mismatch: {run_id}")
        return
    duration = schedule_rounds(schedule_id)
    expected_envelope = list(range(EVENT_ROUND + 1, EVENT_ROUND + duration + 1))
    expected_tracking = list(range(EVENT_ROUND + duration + 1, ROUNDS + 1))
    actual_lambdas = rounds.loc[envelope, "deployment_dare_lambda_value"].astype(float).tolist()
    expected_lambdas = [schedule_lambda(schedule_id, index) for index in range(1, duration + 1)]
    tracking_rows = rounds.loc[tracking]
    if (
        rounds.loc[trigger, "round"].astype(int).tolist() != [EVENT_ROUND]
        or rounds.loc[hold, "round"].astype(int).tolist() != [EVENT_ROUND]
        or rounds.loc[envelope, "round"].astype(int).tolist() != expected_envelope
        or rounds.loc[tracking, "round"].astype(int).tolist() != expected_tracking
        or rounds.loc[captured, "round"].astype(int).tolist() != [EVENT_ROUND]
        or rounds.loc[released, "round"].astype(int).tolist() != [EVENT_ROUND + duration]
        or not np.allclose(actual_lambdas, expected_lambdas, rtol=0.0, atol=1e-12)
        or not tracking_rows["evaluation_model_hash"].astype(str).eq(
            tracking_rows["global_model_hash"].astype(str)
        ).all()
    ):
        raise RuntimeError(f"v13 M2 S4 DARE contract mismatch: {run_id}")


def _run_metrics(job: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
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
        or bool(actual_job.get("formal_test_access"))
        or bool(actual_job.get("test_labels_used_for_selection"))
    ):
        raise RuntimeError(f"v13 M2 run contract mismatch: {run_id}")
    method_version = str(job["method_version"])
    scenario = str(job["scenario_id"])
    trigger = rounds["telemetry_shift_trigger"].astype(bool)
    hold = rounds["deployment_quarantine_applied"].astype(bool)
    if scenario == "S0":
        if trigger.any() or hold.any():
            raise RuntimeError(f"v13 M2 S0 telemetry mismatch: {run_id}")
    elif scenario == "S4":
        event_rounds = rounds.loc[
            rounds["event_offset_round"].astype(int) == 0, "round"
        ].astype(int).tolist()
        if event_rounds != [EVENT_ROUND] or rounds.loc[trigger, "round"].astype(int).tolist() != [EVENT_ROUND]:
            raise RuntimeError(f"v13 M2 S4 event mismatch: {run_id}")
    else:
        raise RuntimeError(f"unexpected M2 scenario: {scenario}")
    if method_version == "v13":
        _audit_v13_contract(rounds, scenario, str(job["selected_schedule_id"]), run_id)
    elif method_version == "v11":
        if scenario == "S4" and rounds.loc[hold, "round"].astype(int).tolist() != [EVENT_ROUND]:
            raise RuntimeError(f"v13 M2 v11 S4 quarantine mismatch: {run_id}")
    else:
        raise RuntimeError(f"unexpected M2 method version: {method_version}")
    tta_round, tta_s = _time_to_accuracy(rounds)
    pta20 = float(derive_pta20_percent(rounds)) if scenario == "S4" else None
    metrics = {
        "method_version": method_version,
        "scenario_id": scenario,
        "run_id": run_id,
        "last50_validation_accuracy": float(rounds.tail(50)["test_accuracy"].astype(float).mean()),
        "pta20_percent": pta20,
        "tta_round": tta_round,
        "algorithm_tta_s": tta_s,
        "trigger_count": int(trigger.sum()),
        "hold_count": int(hold.sum()),
        "test_labels_used": False,
    }
    return metrics, rounds


def evaluate_gate(
    *,
    v11_s0_last50: float,
    v13_s0_last50: float,
    v11_s4_last50: float,
    v13_s4_last50: float,
    v11_s4_pta20: float,
    v13_s4_pta20: float,
    v11_s4_tta_s: float | None,
    v13_s4_tta_s: float | None,
) -> dict[str, Any]:
    deltas = {
        "s0_last50_accuracy": 100.0 * (float(v13_s0_last50) - float(v11_s0_last50)),
        "s4_last50_accuracy": 100.0 * (float(v13_s4_last50) - float(v11_s4_last50)),
        "s4_pta20": float(v13_s4_pta20) - float(v11_s4_pta20),
        "s4_algorithm_tta": (
            None
            if v11_s4_tta_s is None or v13_s4_tta_s is None
            else (float(v11_s4_tta_s) - float(v13_s4_tta_s)) / float(v11_s4_tta_s)
        ),
    }
    strict = {
        "s0_last50_accuracy": deltas["s0_last50_accuracy"] > 0.0,
        "s4_last50_accuracy": deltas["s4_last50_accuracy"] > 0.0,
        "s4_pta20": deltas["s4_pta20"] > 0.0,
        "s4_algorithm_tta": deltas["s4_algorithm_tta"] is not None
        and deltas["s4_algorithm_tta"] > 0.0,
    }
    close = {
        "s0_last50_accuracy": deltas["s0_last50_accuracy"]
        >= -GATE_MARGINS["s0_last50_pp"] - DECISION_ATOL,
        "s4_last50_accuracy": deltas["s4_last50_accuracy"]
        >= -GATE_MARGINS["s4_last50_pp"] - DECISION_ATOL,
        "s4_pta20": deltas["s4_pta20"] >= -GATE_MARGINS["s4_pta20_pp"] - DECISION_ATOL,
        "s4_algorithm_tta": deltas["s4_algorithm_tta"] is not None
        and deltas["s4_algorithm_tta"]
        >= -GATE_MARGINS["s4_algorithm_tta_relative"] - DECISION_ATOL,
    }
    strict_names = [name for name, passed in strict.items() if bool(passed)]
    misses = [name for name, passed in strict.items() if not bool(passed)]
    sole_miss_close = len(misses) == 1 and bool(close[misses[0]])
    passed = len(strict_names) == 4 or (len(strict_names) == 3 and sole_miss_close)
    return {
        "deltas": deltas,
        "strict_wins": {name: bool(value) for name, value in strict.items()},
        "within_close_margin": {name: bool(value) for name, value in close.items()},
        "strict_win_count": len(strict_names),
        "misses": misses,
        "sole_miss_close": bool(sole_miss_close),
        "performance_gate_passed": bool(passed),
    }


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot evaluate an incomplete v13 M2 pair")
    rows: list[dict[str, Any]] = []
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for job in manifest["jobs"]:
        metrics, frame = _run_metrics(job)
        rows.append(metrics)
        frames[(str(job["method_version"]), str(job["scenario_id"]))] = frame
    if set(frames) != set(METHOD_ORDER):
        raise RuntimeError("v13 M2 result matrix is incomplete")
    v11_s0 = frames[("v11", "S0")]
    v13_s0 = frames[("v13", "S0")]
    identity = {
        column: bool(np.array_equal(v11_s0[column].to_numpy(), v13_s0[column].to_numpy()))
        for column in IDENTITY_COLUMNS
    }
    s0_identity_passed = bool(all(identity.values()))
    by_key = {(row["method_version"], row["scenario_id"]): row for row in rows}
    gate = evaluate_gate(
        v11_s0_last50=by_key[("v11", "S0")]["last50_validation_accuracy"],
        v13_s0_last50=by_key[("v13", "S0")]["last50_validation_accuracy"],
        v11_s4_last50=by_key[("v11", "S4")]["last50_validation_accuracy"],
        v13_s4_last50=by_key[("v13", "S4")]["last50_validation_accuracy"],
        v11_s4_pta20=by_key[("v11", "S4")]["pta20_percent"],
        v13_s4_pta20=by_key[("v13", "S4")]["pta20_percent"],
        v11_s4_tta_s=by_key[("v11", "S4")]["algorithm_tta_s"],
        v13_s4_tta_s=by_key[("v13", "S4")]["algorithm_tta_s"],
    )
    overall_passed = bool(gate["performance_gate_passed"] and s0_identity_passed)
    frame = pd.DataFrame(rows)
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "status": "m2_gate_passed" if overall_passed else "m2_gate_failed",
        "selection_split": "validation",
        "seed": SEED,
        "formal_test_access": False,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "selected_candidate": manifest["selected_candidate"],
        "gate_rule": manifest["gate_rule"],
        "gate": gate,
        "s0_identity_columns": identity,
        "s0_identity_passed": s0_identity_passed,
        "overall_gate_passed": overall_passed,
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
    passed = bool(result["overall_gate_passed"])
    state.update(
        {
            "status": "m2_completed_gate_passed_m3_required" if passed else "m2_completed_gate_failed_stop",
            "current_job_id": None,
            "all_runs_completed": True,
            "performance_sealed_until_terminal": False,
            "result_path": str(RESULT_PATH),
            "gate_passed": passed,
            "strict_win_count": int(result["gate"]["strict_win_count"]),
            "s0_identity_passed": bool(result["s0_identity_passed"]),
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
