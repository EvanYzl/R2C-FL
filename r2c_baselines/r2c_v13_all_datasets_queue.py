from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import r2c_d3_v13_formal_confirmation_queue as formal
from .config import DATASETS, FORMAL_SEED, PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .data import partition_asset_path, partition_meta_path, prepare_partition
from .logging_io import read_chunked_table
from .metrics import recovery_auc20
from .r2c_d3_v7_phase_e_queue import _audit_run
from .r2c_post_event_tail_accuracy20 import derive_pta20_percent
from .r2c_tail_recovery_margin20 import derive_window_metrics
from .r2c_v13 import PROTOCOL_VERSION, schedule_lambda, schedule_rounds
from .traces import prepare_trace, trace_asset_path, trace_meta_path
from .utils import atomic_csv, atomic_json, atomic_parquet, config_hash, sha256_file, utc_now


PLAN_ID = formal.PLAN_ID
PLAN_PATH = formal.PLAN_PATH
DATASET_ORDER = ("D1", "D2", "D3", "D4")
SCENARIO_ORDER = ("S0", "S4")
SEED = FORMAL_SEED
MAX_ATTEMPTS = 3
DATASET_CONFIG_KEYS = ("lr_mult", "r2c_delta_clip", "r2c_eval_microbatch")

M3_MANIFEST_PATH = formal.MANIFEST_PATH
M3_STATE_PATH = formal.STATE_PATH
M3_RESULT_PATH = formal.RESULT_PATH
M3_RUNS_PATH = formal.RUNS_PATH
M3_RUNS_CSV_PATH = formal.RUNS_CSV_PATH

V11_MANIFEST_PATH = (
    QUEUE_ROOT / "r2c_v11_four_dataset_matched_manifest_20260817T192402.166886Z.json"
)
EXPECTED_V11_MANIFEST_SHA256 = (
    "7dabda949d5a96bca55dd342e791a816832fe4db30f1f4f9f19d3c857d7ea159"
)

MANIFEST_PATH = QUEUE_ROOT / "r2c_v13_all_datasets_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_v13_all_datasets_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_v13_all_datasets_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_v13_all_datasets_result.json"
RUNS_PATH = QUEUE_ROOT / "r2c_v13_all_datasets_runs.parquet"
RUNS_CSV_PATH = QUEUE_ROOT / "r2c_v13_all_datasets_runs.csv"


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        "config.py",
        "data.py",
        "training.py",
        "traces.py",
        "metrics.py",
        "r2c.py",
        "r2c_v2.py",
        "r2c_v3.py",
        "r2c_v4.py",
        "r2c_v5.py",
        "r2c_v6.py",
        "r2c_v7.py",
        "r2c_v13.py",
        "r2c_post_event_tail_accuracy20.py",
        "r2c_tail_recovery_margin20.py",
        "r2c_d3_v13_formal_confirmation_queue.py",
        "run.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    tests = Path(__file__).resolve().parents[1] / "tests"
    for name in (
        "audit_r2c_run.py",
        "test_r2c_v13.py",
        "test_r2c_d3_v13_formal_confirmation_queue.py",
        "test_r2c_v13_all_datasets_queue.py",
    ):
        values[f"tests/{name}"] = sha256_file(tests / name)
    return values


def _m3_context() -> dict[str, Any]:
    required = (
        M3_MANIFEST_PATH,
        M3_STATE_PATH,
        M3_RESULT_PATH,
        M3_RUNS_PATH,
        M3_RUNS_CSV_PATH,
    )
    if any(not path.exists() for path in required):
        raise RuntimeError("v13 M3 did not authorize the all-dataset matrix")
    manifest = json.loads(M3_MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(M3_STATE_PATH.read_text(encoding="utf-8"))
    result = json.loads(M3_RESULT_PATH.read_text(encoding="utf-8"))
    formal._assert_frozen_manifest(manifest)
    if (
        state.get("status") != "m3_completed_gate_passed_m4_unlocked"
        or int(state.get("completed", -1)) != 4
        or int(state.get("failed", -1)) != 0
        or not bool(state.get("all_runs_completed"))
        or bool(state.get("performance_sealed_until_terminal"))
        or not bool(state.get("gate_passed"))
        or not bool(state.get("s0_identity_passed"))
        or not bool(state.get("other_dataset_access"))
        or result.get("status") != "m3_gate_passed"
        or result.get("evaluation_split") != "test"
        or not bool(result.get("formal_test_access"))
        or bool(result.get("other_dataset_access"))
        or bool(result.get("test_labels_used_for_selection"))
        or not bool(result.get("overall_gate_passed"))
        or not bool(result.get("s0_identity_passed"))
        or int(result.get("gate", {}).get("strict_win_count", -1)) < 3
    ):
        raise RuntimeError("v13 M3 did not authorize the all-dataset matrix")
    immutable_path = Path(str(state["immutable_manifest_path"])).resolve()
    if (
        not immutable_path.exists()
        or sha256_file(immutable_path).lower()
        != str(state.get("immutable_manifest_sha256", "")).lower()
        or str(state.get("frozen_spec_hash")) != str(manifest.get("frozen_spec_hash"))
        or str(result.get("frozen_spec_hash")) != str(manifest.get("frozen_spec_hash"))
    ):
        raise RuntimeError("v13 M3 immutable lineage mismatch")
    schedule_id = str(manifest["selected_candidate"]["schedule_id"])
    audit_logs: list[Path] = []
    run_dirs: list[Path] = []
    for job in manifest["jobs"]:
        if job.get("status") != "completed" or not job.get("actual_run_id"):
            raise RuntimeError("v13 M3 contains an incomplete formal source job")
        run_id = str(job["actual_run_id"])
        run_dir = RUN_ROOT / run_id
        audit_log = QUEUE_ROOT / "worker_logs" / f"{run_id}.audit.log"
        if not audit_log.exists() or not (run_dir / "_SUCCESS.json").exists():
            raise RuntimeError("v13 M3 formal audit/success evidence is missing")
        audit_logs.append(audit_log)
        run_dirs.append(run_dir)
    return {
        "manifest": manifest,
        "state": state,
        "result": result,
        "immutable_path": immutable_path,
        "schedule_id": schedule_id,
        "audit_logs": audit_logs,
        "run_dirs": run_dirs,
    }


def _v11_dataset_context() -> dict[str, dict[str, Any]]:
    if (
        not V11_MANIFEST_PATH.exists()
        or sha256_file(V11_MANIFEST_PATH).lower() != EXPECTED_V11_MANIFEST_SHA256
    ):
        raise RuntimeError("v11 immutable four-dataset manifest hash drift")
    manifest = json.loads(V11_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = [
        (dataset_id, scenario)
        for dataset_id in DATASET_ORDER
        for scenario in SCENARIO_ORDER
    ]
    jobs = list(manifest.get("jobs", []))
    if [(job.get("dataset_id"), job.get("scenario_id")) for job in jobs] != expected:
        raise RuntimeError("v11 immutable dataset/scenario order drift")
    values: dict[str, dict[str, Any]] = {}
    for dataset_id in DATASET_ORDER:
        pair = [job for job in jobs if job.get("dataset_id") == dataset_id]
        if (
            len(pair) != 2
            or pair[0].get("method_config") != pair[1].get("method_config")
            or pair[0].get("target_accuracy") != pair[1].get("target_accuracy")
            or int(pair[0].get("rounds", -1)) != int(DATASETS[dataset_id].round_budget)
            or int(pair[1].get("rounds", -1)) != int(DATASETS[dataset_id].round_budget)
        ):
            raise RuntimeError(f"v11 matched source drift for {dataset_id}")
        config = dict(pair[0]["method_config"])
        if (
            config.get("r2c_v4_deployment_ema_betas") != [formal.ORDINARY_BETA]
            or float(config.get("r2c_v4_primary_deployment_beta", -1.0))
            != formal.ORDINARY_BETA
            or float(config.get("r2c_v7_trigger_deployment_beta", -1.0))
            != formal.TRIGGER_BETA
            or not bool(config.get("r2c_v2_audit_replay"))
        ):
            raise RuntimeError(f"v11 matched configuration drift for {dataset_id}")
        values[dataset_id] = {
            "config_overrides": {key: config[key] for key in DATASET_CONFIG_KEYS},
            "target_accuracy": float(pair[0]["target_accuracy"]),
            "client_microbatch": int(pair[0].get("client_microbatch", 1)),
        }
    return values


def _dataset_method_config(
    dataset_id: str,
    context: dict[str, Any],
    dataset_context: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    schedule_id = str(context["schedule_id"])
    config = formal._v13_config(schedule_id)
    config.update(dataset_context[dataset_id]["config_overrides"])
    if (
        config.get("r2c_protocol_version") != PROTOCOL_VERSION
        or config.get("r2c_v13_plan_id") != PLAN_ID
        or config.get("r2c_v13_schedule_id") != schedule_id
        or config.get("r2c_v4_deployment_ema_betas") != [formal.ORDINARY_BETA]
        or float(config.get("r2c_v4_primary_deployment_beta", -1.0))
        != formal.ORDINARY_BETA
        or not bool(config.get("r2c_v2_audit_replay"))
        or "r2c_v7_trigger_deployment_beta" in config
    ):
        raise RuntimeError(f"v13 all-dataset configuration drift for {dataset_id}")
    return config


def _asset_paths() -> list[Path]:
    paths: list[Path] = []
    for dataset_id in DATASET_ORDER:
        rounds = int(DATASETS[dataset_id].round_budget)
        paths.extend(
            (
                partition_asset_path(dataset_id, SEED),
                partition_meta_path(dataset_id, SEED),
            )
        )
        for scenario in SCENARIO_ORDER:
            paths.extend(
                (
                    trace_asset_path(dataset_id, scenario, SEED, rounds),
                    trace_meta_path(dataset_id, scenario, SEED, rounds),
                )
            )
    return paths


def _ensure_assets() -> dict[str, str]:
    for dataset_id in DATASET_ORDER:
        rounds = int(DATASETS[dataset_id].round_budget)
        prepare_partition(dataset_id, SEED)
        for scenario in SCENARIO_ORDER:
            prepare_trace(dataset_id, scenario, SEED, rounds=rounds)
    paths = _asset_paths()
    if any(not path.exists() for path in paths):
        raise RuntimeError("v13 all-dataset asset preparation is incomplete")
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _existing_asset_hashes() -> dict[str, str]:
    paths = _asset_paths()
    if any(not path.exists() for path in paths):
        raise RuntimeError("v13 all-dataset frozen asset is missing")
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _source_lineage(context: dict[str, Any]) -> dict[str, str]:
    paths: list[Path] = [
        PLAN_PATH,
        Path(context["immutable_path"]),
        M3_MANIFEST_PATH,
        M3_STATE_PATH,
        M3_RESULT_PATH,
        M3_RUNS_PATH,
        M3_RUNS_CSV_PATH,
        V11_MANIFEST_PATH,
    ]
    paths.extend(context["audit_logs"])
    for run_dir in context["run_dirs"]:
        paths.extend(
            (
                run_dir / "job.json",
                run_dir / "result.json",
                run_dir / "_SUCCESS.json",
                run_dir / "run_manifest.parquet",
                run_dir / "tables" / "round_metrics" / "_index.json",
            )
        )
    if any(not path.exists() for path in paths):
        raise RuntimeError("v13 all-dataset source lineage is incomplete")
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _job(
    dataset_id: str,
    scenario: str,
    context: dict[str, Any],
    dataset_context: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if dataset_id not in DATASET_ORDER or scenario not in SCENARIO_ORDER:
        raise ValueError("job is outside immutable M4 dataset/scenario order")
    schedule_id = str(context["schedule_id"])
    rounds = int(DATASETS[dataset_id].round_budget)
    run_id = f"A-R2C-{dataset_id}-{scenario}-V13M4-{schedule_id}-B095-s{SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "v13_all_datasets_locked_formal_engineering_reevaluation",
        "mode": "formal",
        "method_id": "R2C-FL",
        "method_version": "v13",
        "dataset_id": dataset_id,
        "scenario_id": scenario,
        "rounds": rounds,
        "method_config": _dataset_method_config(dataset_id, context, dataset_context),
        "block_id": "A-R2C-V13-M4-ALL-DATASETS",
        "seed": SEED,
        "partition_seed": SEED,
        "trace_seed": SEED,
        "evaluation_split": "test",
        "full_logging": True,
        "client_microbatch": int(dataset_context[dataset_id]["client_microbatch"]),
        "target_accuracy": float(dataset_context[dataset_id]["target_accuracy"]),
        "selected_schedule_id": schedule_id,
        "source_m3_frozen_spec_hash": str(context["manifest"]["frozen_spec_hash"]),
        "formal_test_access": True,
        "other_dataset_access": True,
        "test_labels_used_for_selection": False,
        "formal_interpretation": (
            "outcome_informed_engineering_reevaluation_not_untouched_confirmation"
        ),
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
        "evaluation_split",
        "seed",
        "dataset_order",
        "scenario_order",
        "round_budgets",
        "formal_test_access",
        "other_dataset_access",
        "test_labels_used_for_selection",
        "candidate_locked_before_all_datasets",
        "performance_sealed_until_terminal",
        "selected_candidate",
        "formal_interpretation",
        "diagnostic_metrics",
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
        if any(
            job.get("status") != "pending" or int(job.get("attempts", 0))
            for job in existing["jobs"]
        ):
            raise RuntimeError("Refusing to rebuild a started v13 all-dataset manifest")
    context = _m3_context()
    dataset_context = _v11_dataset_context()
    jobs = [
        _job(dataset_id, scenario, context, dataset_context)
        for dataset_id in DATASET_ORDER
        for scenario in SCENARIO_ORDER
    ]
    expected_order = [
        (dataset_id, scenario)
        for dataset_id in DATASET_ORDER
        for scenario in SCENARIO_ORDER
    ]
    if [(job["dataset_id"], job["scenario_id"]) for job in jobs] != expected_order:
        raise AssertionError("v13 M4 order must be D1--D4 x S0/S4")
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "v13_D1_D4_S0_S4_seed20260811_locked_engineering_reevaluation",
        "plan_id": PLAN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_split": "test",
        "seed": SEED,
        "dataset_order": list(DATASET_ORDER),
        "scenario_order": list(SCENARIO_ORDER),
        "round_budgets": {
            dataset_id: int(DATASETS[dataset_id].round_budget)
            for dataset_id in DATASET_ORDER
        },
        "formal_test_access": True,
        "other_dataset_access": True,
        "test_labels_used_for_selection": False,
        "candidate_locked_before_all_datasets": True,
        "performance_sealed_until_terminal": True,
        "selected_candidate": {
            "schedule_id": str(context["schedule_id"]),
            "source_m3_frozen_spec_hash": str(context["manifest"]["frozen_spec_hash"]),
        },
        "formal_interpretation": (
            "outcome_informed_engineering_reevaluation_not_untouched_confirmation"
        ),
        "diagnostic_metrics": (
            "last50 accuracy; PTA20; algorithm TTA; Recovery-deficit AUC20; TRM20"
        ),
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
        immutable_path = QUEUE_ROOT / f"r2c_v13_all_datasets_manifest_{stamp}.json"
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
                "formal_test_access": True,
                "other_dataset_access": True,
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
    context = _m3_context()
    dataset_context = _v11_dataset_context()
    expected_selected = {
        "schedule_id": str(context["schedule_id"]),
        "source_m3_frozen_spec_hash": str(context["manifest"]["frozen_spec_hash"]),
    }
    if (
        manifest.get("scope")
        != "v13_D1_D4_S0_S4_seed20260811_locked_engineering_reevaluation"
        or manifest.get("plan_id") != PLAN_ID
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("evaluation_split") != "test"
        or int(manifest.get("seed", -1)) != SEED
        or manifest.get("dataset_order") != list(DATASET_ORDER)
        or manifest.get("scenario_order") != list(SCENARIO_ORDER)
        or not bool(manifest.get("formal_test_access"))
        or not bool(manifest.get("other_dataset_access"))
        or bool(manifest.get("test_labels_used_for_selection"))
        or not bool(manifest.get("candidate_locked_before_all_datasets"))
        or not bool(manifest.get("performance_sealed_until_terminal"))
        or manifest.get("selected_candidate") != expected_selected
    ):
        raise RuntimeError("v13 all-dataset manifest scope drift")
    if manifest.get("source_lineage") != _source_lineage(context):
        raise RuntimeError("v13 all-dataset source lineage drift")
    if manifest.get("asset_hashes") != _existing_asset_hashes():
        raise RuntimeError("v13 all-dataset asset drift")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("v13 all-dataset implementation drift")
    if manifest.get("frozen_spec_hash") != config_hash(_frozen_spec(manifest)):
        raise RuntimeError("v13 all-dataset frozen specification drift")
    jobs = list(manifest.get("jobs", []))
    expected_order = [
        (dataset_id, scenario)
        for dataset_id in DATASET_ORDER
        for scenario in SCENARIO_ORDER
    ]
    if len(jobs) != 8 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("v13 all-dataset matrix/order drift")
    for job, (dataset_id, scenario) in zip(jobs, expected_order):
        if (
            job.get("dataset_id") != dataset_id
            or job.get("scenario_id") != scenario
            or job.get("method_version") != "v13"
            or job.get("mode") != "formal"
            or job.get("evaluation_split") != "test"
            or int(job.get("rounds", -1)) != int(DATASETS[dataset_id].round_budget)
            or any(
                int(job.get(key, -1)) != SEED
                for key in ("seed", "partition_seed", "trace_seed")
            )
            or job.get("method_config")
            != _dataset_method_config(dataset_id, context, dataset_context)
            or job.get("selected_schedule_id") != context["schedule_id"]
            or not bool(job.get("formal_test_access"))
            or not bool(job.get("other_dataset_access"))
            or bool(job.get("test_labels_used_for_selection"))
        ):
            raise RuntimeError(f"v13 all-dataset job drift: {job.get('job_id')}")


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
            audit_log = QUEUE_ROOT / "worker_logs" / f"{path.name}.m4.reconcile.audit.log"
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


def _audit_v13_contract(
    rounds: pd.DataFrame,
    scenario: str,
    schedule_id: str,
    event_round: int,
    budget: int,
    run_id: str,
) -> None:
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
        raise RuntimeError(f"v13 all-dataset forbidden DARE input: {run_id}")
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
            raise RuntimeError(f"v13 all-dataset S0 no-trigger mismatch: {run_id}")
        return
    duration = schedule_rounds(schedule_id)
    expected_envelope = list(range(event_round + 1, event_round + duration + 1))
    expected_tracking = list(range(event_round + duration + 1, budget + 1))
    actual_lambdas = rounds.loc[envelope, "deployment_dare_lambda_value"].astype(float).tolist()
    expected_lambdas = [schedule_lambda(schedule_id, index) for index in range(1, duration + 1)]
    tracking_rows = rounds.loc[tracking]
    if (
        rounds.loc[trigger, "round"].astype(int).tolist() != [event_round]
        or rounds.loc[hold, "round"].astype(int).tolist() != [event_round]
        or rounds.loc[envelope, "round"].astype(int).tolist() != expected_envelope
        or rounds.loc[tracking, "round"].astype(int).tolist() != expected_tracking
        or rounds.loc[captured, "round"].astype(int).tolist() != [event_round]
        or rounds.loc[released, "round"].astype(int).tolist()
        != [event_round + duration]
        or not np.allclose(actual_lambdas, expected_lambdas, rtol=0.0, atol=1e-12)
        or not tracking_rows["evaluation_model_hash"].astype(str).eq(
            tracking_rows["global_model_hash"].astype(str)
        ).all()
    ):
        raise RuntimeError(f"v13 all-dataset S4 DARE contract mismatch: {run_id}")


def _time_to_accuracy(
    rounds: pd.DataFrame, target: float
) -> tuple[int | None, float | None]:
    reached = rounds.loc[rounds["test_accuracy"].astype(float) >= target]
    if reached.empty:
        return None, None
    first = reached.sort_values("round").iloc[0]
    return int(first["round"]), float(first["algorithm_elapsed_s"])


def _run_metrics(job: dict[str, Any]) -> dict[str, Any]:
    run_id = str(job["actual_run_id"])
    run_dir = RUN_ROOT / run_id
    actual_job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    run_manifest = pd.read_parquet(run_dir / "run_manifest.parquet").iloc[0]
    rounds = read_chunked_table(run_dir, "round_metrics").sort_values("round")
    budget = int(job["rounds"])
    if (
        len(rounds) != budget
        or rounds["round"].astype(int).tolist() != list(range(1, budget + 1))
        or str(run_manifest["source_kind"]) != "REPRODUCED"
        or actual_job.get("mode") != "formal"
        or actual_job.get("evaluation_split") != "test"
        or actual_job.get("method_config") != job.get("method_config")
        or bool(actual_job.get("test_labels_used_for_selection"))
    ):
        raise RuntimeError(f"v13 all-dataset run contract mismatch: {run_id}")
    scenario = str(job["scenario_id"])
    event_rows = rounds.loc[
        pd.to_numeric(rounds["event_offset_round"], errors="coerce") == 0
    ]
    event_round: int | None = None
    if scenario == "S4":
        if len(event_rows) != 1:
            raise RuntimeError(f"v13 all-dataset S4 event mismatch: {run_id}")
        event_round = int(event_rows.iloc[0]["round"])
        if event_round != budget // 2:
            raise RuntimeError(f"v13 all-dataset S4 event boundary drift: {run_id}")
    elif scenario != "S0":
        raise RuntimeError(f"unexpected v13 all-dataset scenario: {scenario}")
    _audit_v13_contract(
        rounds,
        scenario,
        str(job["selected_schedule_id"]),
        0 if event_round is None else event_round,
        budget,
        run_id,
    )
    tta_round, tta_s = _time_to_accuracy(rounds, float(job["target_accuracy"]))
    pta20: float | None = None
    auc20: float | None = None
    trm20: float | None = None
    if event_round is not None:
        pta20 = float(derive_pta20_percent(rounds))
        recovery = recovery_auc20(rounds["round"], rounds["test_accuracy"], event_round)
        if not bool(recovery["recovery_auc20_complete"]):
            raise RuntimeError(f"v13 all-dataset AUC@20 window incomplete: {run_id}")
        auc20 = float(recovery["recovery_deficit_auc20"])
        trm20 = float(derive_window_metrics(rounds)["trm20_pp"])
    return {
        "dataset_id": str(job["dataset_id"]),
        "scenario_id": scenario,
        "run_id": run_id,
        "round_budget": budget,
        "last50_test_accuracy": float(rounds.tail(50)["test_accuracy"].astype(float).mean()),
        "pta20_percent": pta20,
        "algorithm_tta_round": tta_round,
        "algorithm_tta_s": tta_s,
        "recovery_deficit_auc20": auc20,
        "trm20_pp": trm20,
        "event_round": event_round,
        "source_kind": str(run_manifest["source_kind"]),
        "formal_interpretation": str(job["formal_interpretation"]),
        "test_labels_used_for_selection": False,
    }


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot freeze an incomplete v13 all-dataset matrix")
    rows = [_run_metrics(job) for job in manifest["jobs"]]
    frame = pd.DataFrame(rows)
    expected = [
        (dataset_id, scenario)
        for dataset_id in DATASET_ORDER
        for scenario in SCENARIO_ORDER
    ]
    if list(zip(frame["dataset_id"], frame["scenario_id"])) != expected:
        raise RuntimeError("v13 all-dataset terminal matrix/order mismatch")
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "status": "m4_all_datasets_completed_audited",
        "evaluation_split": "test",
        "source_kind": "REPRODUCED",
        "formal_test_access": True,
        "other_dataset_access": True,
        "test_labels_used_for_selection": False,
        "formal_interpretation": manifest["formal_interpretation"],
        "selected_candidate": manifest["selected_candidate"],
        "completed_runs": len(rows),
        "run_ids": frame["run_id"].astype(str).tolist(),
        "runs_path": str(RUNS_PATH),
        "frozen_spec_hash": manifest["frozen_spec_hash"],
        "tables_update_required": True,
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
    state.update({"status": "running", "performance_sealed_until_terminal": True})
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
            "status": "m4_completed_audited_tables_update_required",
            "current_job_id": None,
            "all_runs_completed": True,
            "performance_sealed_until_terminal": False,
            "result_path": str(RESULT_PATH),
            "tables_update_required": bool(result["tables_update_required"]),
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
