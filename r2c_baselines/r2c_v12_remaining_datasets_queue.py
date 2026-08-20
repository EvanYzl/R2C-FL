from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATASETS, FORMAL_SEED, PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .data import partition_asset_path, partition_meta_path, prepare_partition
from .logging_io import read_chunked_table
from .r2c_d3_v7_phase_e_queue import _audit_run
from .r2c_tail_recovery_margin20 import derive_window_metrics
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
DATASET_ORDER = ("D1", "D2", "D4")
SCENARIO_ORDER = ("S0", "S4")
MAX_ATTEMPTS = 3
DATASET_CONFIG_KEYS = ("lr_mult", "r2c_delta_clip", "r2c_eval_microbatch")
TARGET_ACCURACY = {
    "D1": 0.7903500000000001,
    "D2": 0.5948964000000001,
    "D4": 0.2658744,
}

FORMAL_MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v12_formal_manifest_20260819T091635.529040Z.json"
FORMAL_STATE_PATH = QUEUE_ROOT / "r2c_d3_v12_formal_queue_state.json"
FORMAL_RESULT_PATH = QUEUE_ROOT / "r2c_d3_v12_formal_result.json"
EXPECTED_FORMAL_MANIFEST_SHA256 = (
    "27124de4a56822c6b68540b1b4956c54b491f17c553d7cc76fa4fadbf71be7d4"
)
EXPECTED_FORMAL_FROZEN_SPEC_SHA256 = (
    "e5baf48aa9864c5d9c8461624c5e558c0b6aad17c2b4d44ddc18c925e48adbb4"
)
EXPECTED_FORMAL_QUEUE_SOURCE_SHA256 = (
    "47e0abb73de7fed9bee43b78224f3fc7f95f82cc0360b94c638070ea0a7f572c"
)

V11_MANIFEST_PATH = (
    QUEUE_ROOT / "r2c_v11_four_dataset_matched_manifest_20260817T192402.166886Z.json"
)
EXPECTED_V11_MANIFEST_SHA256 = (
    "7dabda949d5a96bca55dd342e791a816832fe4db30f1f4f9f19d3c857d7ea159"
)

MANIFEST_PATH = QUEUE_ROOT / "r2c_v12_remaining_datasets_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_v12_remaining_datasets_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_v12_remaining_datasets_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_v12_remaining_datasets_result.json"
RUNS_PATH = QUEUE_ROOT / "r2c_v12_remaining_datasets_runs.parquet"
RUNS_CSV_PATH = QUEUE_ROOT / "r2c_v12_remaining_datasets_runs.csv"


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
        "r2c_v8.py",
        "r2c_tail_recovery_margin20.py",
        "r2c_d3_v12_formal_queue.py",
        "run.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    values["tests/audit_r2c_run.py"] = sha256_file(
        Path(__file__).resolve().parents[1] / "tests" / "audit_r2c_run.py"
    )
    return values


def _formal_context() -> dict[str, Any]:
    formal_source = Path(__file__).resolve().parent / "r2c_d3_v12_formal_queue.py"
    if sha256_file(FORMAL_MANIFEST_PATH) != EXPECTED_FORMAL_MANIFEST_SHA256:
        raise RuntimeError("v12 formal immutable manifest hash drift")
    if sha256_file(formal_source) != EXPECTED_FORMAL_QUEUE_SOURCE_SHA256:
        raise RuntimeError("v12 formal queue source hash drift")
    if not FORMAL_RESULT_PATH.exists():
        raise RuntimeError("v12 D3 formal pilot result is not terminal")
    state = json.loads(FORMAL_STATE_PATH.read_text(encoding="utf-8"))
    result = json.loads(FORMAL_RESULT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(FORMAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("frozen_spec_hash") != EXPECTED_FORMAL_FROZEN_SPEC_SHA256
        or state.get("status")
        != "formal_pilot_completed_gate_passed_remaining_datasets_required"
        or int(state.get("completed", -1)) != 2
        or int(state.get("failed", -1)) != 0
        or not bool(state.get("all_runs_completed"))
        or not bool(state.get("gate_passed"))
        or not bool(state.get("other_dataset_access"))
        or result.get("status") != "formal_pilot_gate_passed"
        or not bool(result.get("other_dataset_access_authorized"))
        or not bool(result.get("gate", {}).get("gate_passed"))
        or int(result.get("gate", {}).get("strict_win_count", -1)) < 3
        or not all(bool(v) for v in result.get("structural_checks", {}).values())
        or bool(result.get("test_labels_used_for_selection"))
    ):
        raise RuntimeError("v12 D3 formal pilot did not authorize remaining datasets")
    jobs = list(manifest.get("jobs", []))
    if len(jobs) != 2 or jobs[0].get("method_config") != jobs[1].get("method_config"):
        raise RuntimeError("v12 D3 formal pilot configuration mismatch")
    return {
        "state": state,
        "result": result,
        "manifest": manifest,
        "method_config": dict(jobs[0]["method_config"]),
    }


def _v11_dataset_configs() -> dict[str, dict[str, Any]]:
    if sha256_file(V11_MANIFEST_PATH) != EXPECTED_V11_MANIFEST_SHA256:
        raise RuntimeError("v11 immutable four-dataset manifest hash drift")
    manifest = json.loads(V11_MANIFEST_PATH.read_text(encoding="utf-8"))
    configs: dict[str, dict[str, Any]] = {}
    for dataset_id in DATASET_ORDER:
        jobs = [job for job in manifest.get("jobs", []) if job.get("dataset_id") == dataset_id]
        if len(jobs) != 2 or jobs[0].get("method_config") != jobs[1].get("method_config"):
            raise RuntimeError(f"v11 dataset config mismatch for {dataset_id}")
        configs[dataset_id] = {
            key: jobs[0]["method_config"][key] for key in DATASET_CONFIG_KEYS
        }
    return configs


def _dataset_method_config(
    dataset_id: str, context: dict[str, Any], dataset_configs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    config = dict(context["method_config"])
    for key, value in dataset_configs[dataset_id].items():
        config[key] = value
    if (
        config.get("r2c_protocol_version") != PROTOCOL_VERSION
        or config.get("r2c_v12_plan_id") != PLAN_ID
        or not bool(config.get("r2c_v2_audit_replay"))
        or float(config.get("r2c_v4_primary_deployment_beta", -1.0)) != 0.9
        or float(config.get("r2c_v8_trigger_deployment_beta", -1.0)) != 1.0
        or float(config.get("r2c_v8_recovery_pulse_beta", -1.0)) != 0.5
        or int(config.get("r2c_v8_recovery_pulse_rounds", -1)) != 5
    ):
        raise RuntimeError(f"v12 remaining-dataset config drift for {dataset_id}")
    return config


def _ensure_assets() -> dict[str, str]:
    paths: list[Path] = []
    for dataset_id in DATASET_ORDER:
        rounds = int(DATASETS[dataset_id].round_budget)
        prepare_partition(dataset_id, FORMAL_SEED)
        paths.extend(
            [
                partition_asset_path(dataset_id, FORMAL_SEED),
                partition_meta_path(dataset_id, FORMAL_SEED),
            ]
        )
        for scenario in SCENARIO_ORDER:
            prepare_trace(dataset_id, scenario, FORMAL_SEED, rounds=rounds)
            paths.extend(
                [
                    trace_asset_path(dataset_id, scenario, FORMAL_SEED, rounds),
                    trace_meta_path(dataset_id, scenario, FORMAL_SEED, rounds),
                ]
            )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _source_lineage(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "d3_formal_manifest_sha256": sha256_file(FORMAL_MANIFEST_PATH),
        "d3_formal_result_sha256": sha256_file(FORMAL_RESULT_PATH),
        "d3_formal_frozen_spec_hash": EXPECTED_FORMAL_FROZEN_SPEC_SHA256,
        "d3_formal_strict_win_count": int(context["result"]["gate"]["strict_win_count"]),
        "d3_formal_gate_values": dict(context["result"]["gate"]["values"]),
        "v11_dataset_config_manifest_sha256": sha256_file(V11_MANIFEST_PATH),
    }


def _job(
    dataset_id: str,
    scenario: str,
    context: dict[str, Any],
    dataset_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rounds = int(DATASETS[dataset_id].round_budget)
    run_id = f"A-R2C-V12-{dataset_id}-{scenario}-P3-B050-D05-s{FORMAL_SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "v12_remaining_datasets_locked_formal",
        "mode": "formal",
        "method_id": "R2C-FL",
        "dataset_id": dataset_id,
        "scenario_id": scenario,
        "rounds": rounds,
        "method_config": _dataset_method_config(dataset_id, context, dataset_configs),
        "block_id": "A-R2C-V12-REMAINING-DATASETS",
        "seed": FORMAL_SEED,
        "partition_seed": FORMAL_SEED,
        "trace_seed": FORMAL_SEED,
        "evaluation_split": "test",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY[dataset_id],
        "candidate_id": "P3-B050-D05",
        "pulse_beta": 0.5,
        "pulse_rounds": 5,
        "formal_interpretation": (
            "outcome_informed_engineering_reevaluation_not_untouched_confirmation"
        ),
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
        "evaluation_split",
        "seed",
        "dataset_order",
        "scenario_order",
        "round_budgets",
        "candidate_locked_before_other_datasets",
        "performance_sealed_until_terminal",
        "test_labels_used_for_selection",
        "selected_candidate",
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
            raise RuntimeError("refusing to rebuild a started v12 remaining-dataset manifest")
    context = _formal_context()
    dataset_configs = _v11_dataset_configs()
    jobs = [
        _job(dataset_id, scenario, context, dataset_configs)
        for dataset_id in DATASET_ORDER
        for scenario in SCENARIO_ORDER
    ]
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "v12_remaining_D1_D2_D4_locked_formal_matrix",
        "plan_id": PLAN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_split": "test",
        "seed": FORMAL_SEED,
        "dataset_order": list(DATASET_ORDER),
        "scenario_order": list(SCENARIO_ORDER),
        "round_budgets": {
            dataset_id: int(DATASETS[dataset_id].round_budget)
            for dataset_id in DATASET_ORDER
        },
        "candidate_locked_before_other_datasets": True,
        "performance_sealed_until_terminal": True,
        "test_labels_used_for_selection": False,
        "selected_candidate": {
            "candidate_id": "P3-B050-D05",
            "ordinary_beta": 0.9,
            "trigger_beta": 1.0,
            "pulse_beta": 0.5,
            "pulse_rounds": 5,
        },
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
        immutable_path = QUEUE_ROOT / f"r2c_v12_remaining_datasets_manifest_{stamp}.json"
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


def _event(
    events: list[dict[str, Any]], job: dict[str, Any], event_type: str, **extra: Any
) -> None:
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
            audit_log = QUEUE_ROOT / "worker_logs" / f"{path.name}.v12remaining.reconcile.audit.log"
            if _audit_run(path.name, audit_log).returncode == 0:
                return path.name
    return None


def _reconcile_successes(
    manifest: dict[str, Any], events: list[dict[str, Any]]
) -> int:
    changed = 0
    for job in manifest["jobs"]:
        run_id = _successful_run_id(job)
        if run_id is not None and (
            job.get("status") != "completed" or job.get("actual_run_id") != run_id
        ):
            job.update(
                {"status": "completed", "actual_run_id": run_id, "failure_reason": None}
            )
            _event(events, job, "reconciled_completed", reason="success_output_and_audit")
            changed += 1
    return changed


def _assert_frozen_manifest(manifest: dict[str, Any]) -> None:
    context = _formal_context()
    dataset_configs = _v11_dataset_configs()
    if (
        manifest.get("scope") != "v12_remaining_D1_D2_D4_locked_formal_matrix"
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("evaluation_split") != "test"
        or int(manifest.get("seed", -1)) != FORMAL_SEED
        or manifest.get("dataset_order") != list(DATASET_ORDER)
        or manifest.get("scenario_order") != list(SCENARIO_ORDER)
        or not bool(manifest.get("candidate_locked_before_other_datasets"))
        or not bool(manifest.get("performance_sealed_until_terminal"))
        or bool(manifest.get("test_labels_used_for_selection"))
    ):
        raise RuntimeError("v12 remaining-dataset scope drift")
    if manifest.get("source_lineage") != _source_lineage(context):
        raise RuntimeError("v12 remaining-dataset source lineage drift")
    if manifest.get("asset_hashes") != _ensure_assets():
        raise RuntimeError("v12 remaining-dataset asset drift")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("v12 remaining-dataset implementation drift")
    if manifest.get("frozen_spec_hash") != config_hash(_frozen_spec(manifest)):
        raise RuntimeError("v12 remaining-dataset frozen specification drift")
    jobs = list(manifest.get("jobs", []))
    expected_order = [
        (dataset_id, scenario)
        for dataset_id in DATASET_ORDER
        for scenario in SCENARIO_ORDER
    ]
    if len(jobs) != 6 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("v12 remaining-dataset matrix/order drift")
    for job, (dataset_id, scenario) in zip(jobs, expected_order):
        if (
            job.get("dataset_id") != dataset_id
            or job.get("scenario_id") != scenario
            or job.get("mode") != "formal"
            or job.get("evaluation_split") != "test"
            or int(job.get("rounds", -1)) != int(DATASETS[dataset_id].round_budget)
            or any(
                int(job.get(key, -1)) != FORMAL_SEED
                for key in ("seed", "partition_seed", "trace_seed")
            )
            or job.get("method_config")
            != _dataset_method_config(dataset_id, context, dataset_configs)
            or bool(job.get("test_labels_used_for_selection"))
        ):
            raise RuntimeError(f"v12 remaining-dataset protocol drift in {job.get('job_id')}")


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
    ):
        raise RuntimeError(f"v12 remaining-dataset run contract mismatch: {run_id}")
    scenario = str(job["scenario_id"])
    trigger = rounds["deployment_pulse_telemetry_trigger"].astype(bool)
    hold = rounds["deployment_pulse_hold_applied"].astype(bool)
    pulse = rounds["deployment_recovery_pulse_applied"].astype(bool)
    response = rounds["deployment_pulse_response_applied"].astype(bool)
    if (
        rounds["deployment_pulse_labels_used"].astype(bool).any()
        or rounds["deployment_pulse_scenario_metadata_used"].astype(bool).any()
        or not rounds["deployment_pulse_state_server_only"].astype(bool).all()
    ):
        raise RuntimeError(f"v12 remaining-dataset forbidden pulse input: {run_id}")
    event_round: int | None = None
    trm20_pp: float | None = None
    if scenario == "S0":
        if trigger.any() or hold.any() or pulse.any() or response.any():
            raise RuntimeError(f"v12 remaining-dataset S0 no-trigger mismatch: {run_id}")
    elif scenario == "S4":
        event = rounds.loc[
            pd.to_numeric(rounds["event_offset_round"], errors="coerce") == 0
        ]
        if len(event) != 1:
            raise RuntimeError(f"v12 remaining-dataset S4 event mismatch: {run_id}")
        event_round = int(event.iloc[0]["round"])
        if (
            rounds.loc[trigger, "round"].astype(int).tolist() != [event_round]
            or rounds.loc[hold, "round"].astype(int).tolist() != [event_round]
            or rounds.loc[pulse, "round"].astype(int).tolist()
            != list(range(event_round + 1, event_round + 6))
            or rounds.loc[response, "round"].astype(int).tolist()
            != list(range(event_round, event_round + 6))
        ):
            raise RuntimeError(f"v12 remaining-dataset S4 pulse mismatch: {run_id}")
        trm20_pp = float(derive_window_metrics(rounds)["trm20_pp"])
    else:
        raise RuntimeError(f"unexpected scenario: {scenario}")
    tta_round, tta_s = _time_to_accuracy(rounds, float(job["target_accuracy"]))
    return {
        "dataset_id": str(job["dataset_id"]),
        "scenario_id": scenario,
        "run_id": run_id,
        "round_budget": budget,
        "last50_accuracy": float(rounds.tail(50)["test_accuracy"].astype(float).mean()),
        "trm20_pp": trm20_pp,
        "tta_round": tta_round,
        "algorithm_tta_s": tta_s,
        "event_round": event_round,
        "trigger_count": int(trigger.sum()),
        "hold_count": int(hold.sum()),
        "pulse_count": int(pulse.sum()),
        "response_count": int(response.sum()),
        "source_kind": str(run_manifest["source_kind"]),
    }


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("cannot freeze incomplete v12 remaining-dataset matrix")
    rows = [_run_metrics(job) for job in manifest["jobs"]]
    frame = pd.DataFrame(rows)
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "status": "remaining_datasets_completed_audited",
        "source_kind": "REPRODUCED",
        "evaluation_split": "test",
        "test_labels_used_for_selection": False,
        "selected_candidate": dict(manifest["selected_candidate"]),
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
            state.update(
                {"status": "failed_max_attempts", "current_job_id": job["job_id"]}
            )
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
            _event(
                events,
                job,
                "failed",
                exit_code=process.returncode,
                reason=job["failure_reason"],
            )
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
            "status": "remaining_datasets_completed_tables_update_required",
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
