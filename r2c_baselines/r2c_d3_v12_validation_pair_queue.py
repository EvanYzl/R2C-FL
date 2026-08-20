from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from . import r2c_d3_v12_recovery_screen_queue as screen
from .config import PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .data import partition_asset_path, partition_meta_path, prepare_partition
from .logging_io import read_chunked_table
from .r2c_d3_v7_phase_e_queue import _audit_run
from .r2c_tail_recovery_margin20 import derive_window_metrics
from .r2c_v8 import PROTOCOL_VERSION
from .traces import prepare_trace, trace_asset_path, trace_meta_path
from .utils import atomic_csv, atomic_json, atomic_parquet, config_hash, sha256_file, utc_now


PLAN_ID = screen.PLAN_ID
PLAN_PATH = screen.PLAN_PATH

SCREEN_MANIFEST_PATH = screen.MANIFEST_PATH
SCREEN_STATE_PATH = screen.STATE_PATH
SCREEN_RESULT_PATH = screen.RESULT_PATH
SCREEN_RUNS_PATH = screen.RUNS_PATH
SCREEN_RUNS_CSV_PATH = screen.RUNS_CSV_PATH

MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v12_validation_pair_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v12_validation_pair_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v12_validation_pair_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v12_validation_pair_result.json"
RUNS_PATH = QUEUE_ROOT / "r2c_d3_v12_validation_pair_runs.parquet"
RUNS_CSV_PATH = QUEUE_ROOT / "r2c_d3_v12_validation_pair_runs.csv"

DATASET_ID = "D3"
SCENARIOS = ("S0", "S4")
SEED = 20260810
ROUNDS = 1000
EVENT_ROUND = 500
TARGET_ACCURACY = screen.TARGET_ACCURACY
ORDINARY_BETA = screen.ORDINARY_BETA
TRIGGER_BETA = screen.TRIGGER_BETA
MAX_ATTEMPTS = 3

BASELINE_ENVELOPE = dict(screen.BASELINE_ENVELOPE)
GATE_MARGINS = {
    "accuracy_fraction": 0.0015,  # 0.15 percentage points
    "trm20_pp": 0.10,
    "tta_relative": 0.05,
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
        "r2c_d3_v12_recovery_screen_queue.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    values["tests/audit_r2c_run.py"] = sha256_file(
        Path(__file__).resolve().parents[1] / "tests" / "audit_r2c_run.py"
    )
    values["tests/test_r2c_v12_validation_pair_queue.py"] = sha256_file(
        Path(__file__).resolve().parents[1]
        / "tests"
        / "test_r2c_v12_validation_pair_queue.py"
    )
    return values


def _screen_context() -> dict[str, Any]:
    for path in (SCREEN_MANIFEST_PATH, SCREEN_STATE_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(SCREEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(SCREEN_STATE_PATH.read_text(encoding="utf-8"))
    if not SCREEN_RESULT_PATH.exists():
        raise RuntimeError("v12 screen is not terminal with a selected validation candidate")
    result = json.loads(SCREEN_RESULT_PATH.read_text(encoding="utf-8"))
    screen._assert_frozen_manifest(manifest)
    if (
        state.get("status") != "screen_completed_selected_validation_pair_required"
        or int(state.get("completed", -1)) != 4
        or int(state.get("failed", -1)) != 0
        or not bool(state.get("all_runs_completed"))
        or result.get("status") != "selected"
        or result.get("selected") is None
        or result.get("selection_split") != "validation"
        or bool(result.get("formal_test_access"))
        or bool(result.get("test_labels_used_for_selection"))
    ):
        raise RuntimeError("v12 screen is not terminal with a selected validation candidate")
    immutable_path = Path(str(state["immutable_manifest_path"])).resolve()
    if (
        not immutable_path.exists()
        or sha256_file(immutable_path) != str(state["immutable_manifest_sha256"])
        or str(state.get("frozen_spec_hash")) != str(manifest.get("frozen_spec_hash"))
        or str(result.get("frozen_spec_hash")) != str(manifest.get("frozen_spec_hash"))
    ):
        raise RuntimeError("v12 screen immutable lineage mismatch")
    selected = dict(result["selected"])
    candidate_id = str(selected["candidate_id"])
    jobs = [job for job in manifest["jobs"] if job.get("candidate_id") == candidate_id]
    if len(jobs) != 1 or jobs[0].get("status") != "completed" or not jobs[0].get("actual_run_id"):
        raise RuntimeError("selected v12 candidate is not a unique audited completed job")
    selected_job = dict(jobs[0])
    run_id = str(selected_job["actual_run_id"])
    audit_log = QUEUE_ROOT / "worker_logs" / f"{run_id}.audit.log"
    if not audit_log.exists() or not (RUN_ROOT / run_id / "_SUCCESS.json").exists():
        raise RuntimeError("selected v12 candidate audit/success evidence is missing")
    return {
        "manifest": manifest,
        "state": state,
        "result": result,
        "immutable_path": immutable_path,
        "selected": selected,
        "selected_job": selected_job,
        "selected_run_id": run_id,
        "audit_log": audit_log,
    }


def _source_lineage(context: dict[str, Any]) -> dict[str, str]:
    run_dir = RUN_ROOT / str(context["selected_run_id"])
    paths = (
        PLAN_PATH,
        Path(context["immutable_path"]),
        SCREEN_MANIFEST_PATH,
        SCREEN_STATE_PATH,
        SCREEN_RESULT_PATH,
        SCREEN_RUNS_PATH,
        SCREEN_RUNS_CSV_PATH,
        Path(context["audit_log"]),
        run_dir / "job.json",
        run_dir / "result.json",
        run_dir / "_SUCCESS.json",
        run_dir / "run_manifest.parquet",
        run_dir / "tables" / "round_metrics" / "_index.json",
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _selected_config(context: dict[str, Any]) -> dict[str, Any]:
    job = dict(context["selected_job"])
    config = dict(job["method_config"])
    selected = dict(context["selected"])
    if (
        config.get("r2c_protocol_version") != PROTOCOL_VERSION
        or config.get("r2c_v4_deployment_ema_betas") != [ORDINARY_BETA]
        or float(config.get("r2c_v4_primary_deployment_beta", -1.0)) != ORDINARY_BETA
        or float(config.get("r2c_v8_trigger_deployment_beta", -1.0)) != TRIGGER_BETA
        or float(config.get("r2c_v8_recovery_pulse_beta", -1.0))
        != float(selected["pulse_beta"])
        or int(config.get("r2c_v8_recovery_pulse_rounds", -1))
        != int(selected["pulse_rounds"])
        or config.get("r2c_v12_plan_id") != PLAN_ID
    ):
        raise RuntimeError("selected v12 candidate configuration drift")
    return config


def _ensure_assets(seed: int, rounds: int) -> dict[str, str]:
    prepare_partition(DATASET_ID, seed)
    paths = [partition_asset_path(DATASET_ID, seed), partition_meta_path(DATASET_ID, seed)]
    for scenario in SCENARIOS:
        prepare_trace(DATASET_ID, scenario, seed, rounds=rounds)
        paths.extend(
            (
                trace_asset_path(DATASET_ID, scenario, seed, rounds),
                trace_meta_path(DATASET_ID, scenario, seed, rounds),
            )
        )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _job(scenario: str, context: dict[str, Any]) -> dict[str, Any]:
    selected = dict(context["selected"])
    candidate_id = str(selected["candidate_id"])
    run_id = f"A-R2C-D3-{scenario}-V12PAIR-{candidate_id}-s{SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v12_validation_locked_pair",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": DATASET_ID,
        "scenario_id": scenario,
        "rounds": ROUNDS,
        "method_config": _selected_config(context),
        "block_id": "A-R2C-D3-V12-VALIDATION-LOCKED-PAIR",
        "seed": SEED,
        "partition_seed": SEED,
        "trace_seed": SEED,
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "candidate_id": candidate_id,
        "pulse_beta": float(selected["pulse_beta"]),
        "pulse_rounds": int(selected["pulse_rounds"]),
        "source_screen_run_id": str(context["selected_run_id"]),
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
        "candidate_locked_before_pair",
        "performance_sealed_until_terminal",
        "selected_candidate",
        "baseline_envelope",
        "gate_margins",
        "gate_rule",
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
            raise RuntimeError("Refusing to rebuild a started v12 validation-pair manifest")
    context = _screen_context()
    selected = dict(context["selected"])
    jobs = [_job(scenario, context) for scenario in SCENARIOS]
    if len(jobs) != 2 or [job["scenario_id"] for job in jobs] != list(SCENARIOS):
        raise AssertionError("v12 validation pair must be exactly S0 then S4")
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_validation_only_v12_selected_candidate_pair",
        "plan_id": PLAN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "selection_split": "validation",
        "seed": SEED,
        "rounds_per_job": ROUNDS,
        "event_round": EVENT_ROUND,
        "formal_test_access": False,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "candidate_locked_before_pair": True,
        "performance_sealed_until_terminal": True,
        "selected_candidate": {
            "candidate_id": str(selected["candidate_id"]),
            "pulse_beta": float(selected["pulse_beta"]),
            "pulse_rounds": int(selected["pulse_rounds"]),
            "source_screen_run_id": str(context["selected_run_id"]),
        },
        "baseline_envelope": BASELINE_ENVELOPE,
        "gate_margins": GATE_MARGINS,
        "gate_rule": (
            "pass iff four strict wins, or exactly three strict wins and the sole miss is "
            "within 0.15 pp accuracy, 0.10 pp TRM20, or 5 percent relative TTA"
        ),
        "source_lineage": _source_lineage(context),
        "asset_hashes": _ensure_assets(SEED, ROUNDS),
        "implementation_hashes": _implementation_hashes(),
        "job_order": [job["job_id"] for job in jobs],
        "max_attempts": MAX_ATTEMPTS,
        "jobs": jobs,
    }
    manifest["frozen_spec_hash"] = config_hash(_frozen_spec(manifest))
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        immutable_path = QUEUE_ROOT / f"r2c_d3_v12_validation_pair_manifest_{stamp}.json"
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
    context = _screen_context()
    selected = dict(context["selected"])
    expected_selected = {
        "candidate_id": str(selected["candidate_id"]),
        "pulse_beta": float(selected["pulse_beta"]),
        "pulse_rounds": int(selected["pulse_rounds"]),
        "source_screen_run_id": str(context["selected_run_id"]),
    }
    if (
        manifest.get("scope") != "D3_validation_only_v12_selected_candidate_pair"
        or manifest.get("plan_id") != PLAN_ID
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("selection_split") != "validation"
        or int(manifest.get("seed", -1)) != SEED
        or int(manifest.get("rounds_per_job", -1)) != ROUNDS
        or bool(manifest.get("formal_test_access"))
        or bool(manifest.get("other_dataset_access"))
        or bool(manifest.get("test_labels_used_for_selection"))
        or not bool(manifest.get("candidate_locked_before_pair"))
        or not bool(manifest.get("performance_sealed_until_terminal"))
        or manifest.get("selected_candidate") != expected_selected
        or manifest.get("baseline_envelope") != BASELINE_ENVELOPE
        or manifest.get("gate_margins") != GATE_MARGINS
    ):
        raise RuntimeError("v12 validation-pair scope drift")
    if manifest.get("source_lineage") != _source_lineage(context):
        raise RuntimeError("v12 validation-pair source lineage drift")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("v12 validation-pair implementation drift after freeze")
    if manifest.get("frozen_spec_hash") != config_hash(_frozen_spec(manifest)):
        raise RuntimeError("v12 validation-pair frozen specification drift")
    jobs = list(manifest.get("jobs", []))
    if len(jobs) != 2 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("v12 validation-pair order drift")
    expected_config = _selected_config(context)
    for job, scenario in zip(jobs, SCENARIOS):
        if (
            job.get("dataset_id") != DATASET_ID
            or job.get("scenario_id") != scenario
            or job.get("evaluation_split") != "validation"
            or int(job.get("rounds", -1)) != ROUNDS
            or any(int(job.get(key, -1)) != SEED for key in ("seed", "partition_seed", "trace_seed"))
            or job.get("method_config") != expected_config
            or job.get("candidate_id") != expected_selected["candidate_id"]
            or float(job.get("pulse_beta", -1.0)) != expected_selected["pulse_beta"]
            or int(job.get("pulse_rounds", -1)) != expected_selected["pulse_rounds"]
            or job.get("source_screen_run_id") != expected_selected["source_screen_run_id"]
        ):
            raise RuntimeError(f"v12 validation-pair job drift: {job.get('job_id')}")


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
            audit_log = QUEUE_ROOT / "worker_logs" / f"{path.name}.pair.reconcile.audit.log"
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
    run_manifest = pd.read_parquet(run_dir / "run_manifest.parquet").iloc[0]
    rounds = read_chunked_table(run_dir, "round_metrics").sort_values("round")
    if (
        len(rounds) != ROUNDS
        or rounds["round"].astype(int).tolist() != list(range(1, ROUNDS + 1))
        or str(run_manifest["source_kind"]) != "CALIBRATION"
        or actual_job.get("evaluation_split") != "validation"
        or actual_job.get("method_config") != job.get("method_config")
        or bool(actual_job.get("test_labels_used_for_selection"))
    ):
        raise RuntimeError(f"v12 validation-pair run contract mismatch: {run_id}")
    scenario = str(job["scenario_id"])
    trigger = rounds["telemetry_shift_trigger"].astype(bool)
    hold = rounds["deployment_quarantine_applied"].astype(bool)
    pulse = rounds["deployment_recovery_pulse_applied"].astype(bool)
    if (
        rounds["deployment_pulse_labels_used"].astype(bool).any()
        or rounds["deployment_pulse_scenario_metadata_used"].astype(bool).any()
        or not rounds["deployment_pulse_state_server_only"].astype(bool).all()
    ):
        raise RuntimeError(f"v12 validation-pair forbidden pulse input: {run_id}")
    trm20_pp: float | None = None
    signed_delta20_pp: float | None = None
    if scenario == "S0":
        if trigger.any() or hold.any() or pulse.any():
            raise RuntimeError(f"v12 S0 no-trigger identity mismatch: {run_id}")
    elif scenario == "S4":
        offsets = pd.to_numeric(rounds["event_offset_round"], errors="coerce")
        event = rounds.loc[offsets == 0]
        pulse_rounds = int(job["pulse_rounds"])
        expected_pulse_rounds = list(range(EVENT_ROUND + 1, EVENT_ROUND + pulse_rounds + 1))
        if (
            len(event) != 1
            or int(event.iloc[0]["round"]) != EVENT_ROUND
            or rounds.loc[trigger, "round"].astype(int).tolist() != [EVENT_ROUND]
            or rounds.loc[hold, "round"].astype(int).tolist() != [EVENT_ROUND]
            or rounds.loc[pulse, "round"].astype(int).tolist() != expected_pulse_rounds
        ):
            raise RuntimeError(f"v12 S4 pulse contract mismatch: {run_id}")
        window = derive_window_metrics(rounds)
        trm20_pp = float(window["trm20_pp"])
        signed_delta20_pp = float(window["signed_delta20_pp"])
    else:
        raise RuntimeError(f"unexpected validation-pair scenario: {scenario}")
    tta_round, tta_s = _time_to_accuracy(rounds)
    return {
        "scenario_id": scenario,
        "candidate_id": job["candidate_id"],
        "run_id": run_id,
        "last50_validation_accuracy": float(rounds.tail(50)["test_accuracy"].astype(float).mean()),
        "trm20_pp": trm20_pp,
        "signed_delta20_pp": signed_delta20_pp,
        "tta_round": tta_round,
        "algorithm_tta_s": tta_s,
        "trigger_count": int(trigger.sum()),
        "hold_count": int(hold.sum()),
        "pulse_count": int(pulse.sum()),
        "test_labels_used": False,
    }


def evaluate_gate(
    *,
    s0_last50_accuracy: float,
    s4_last50_accuracy: float,
    s4_trm20_pp: float,
    s4_algorithm_tta_s: float | None,
) -> dict[str, Any]:
    values: dict[str, float | None] = {
        "s0_last50_accuracy": float(s0_last50_accuracy),
        "s4_last50_accuracy": float(s4_last50_accuracy),
        "s4_trm20_pp": float(s4_trm20_pp),
        "s4_algorithm_tta_s": (
            None if s4_algorithm_tta_s is None else float(s4_algorithm_tta_s)
        ),
    }
    strict = {
        "s0_last50_accuracy": values["s0_last50_accuracy"]
        > BASELINE_ENVELOPE["s0_last50_accuracy"],
        "s4_last50_accuracy": values["s4_last50_accuracy"]
        > BASELINE_ENVELOPE["s4_last50_accuracy"],
        "s4_trm20_pp": values["s4_trm20_pp"] > BASELINE_ENVELOPE["s4_trm20_pp"],
        "s4_algorithm_tta_s": values["s4_algorithm_tta_s"] is not None
        and values["s4_algorithm_tta_s"] < BASELINE_ENVELOPE["s4_algorithm_tta_s"],
    }
    close = {
        "s0_last50_accuracy": values["s0_last50_accuracy"]
        >= BASELINE_ENVELOPE["s0_last50_accuracy"] - GATE_MARGINS["accuracy_fraction"],
        "s4_last50_accuracy": values["s4_last50_accuracy"]
        >= BASELINE_ENVELOPE["s4_last50_accuracy"] - GATE_MARGINS["accuracy_fraction"],
        "s4_trm20_pp": values["s4_trm20_pp"]
        >= BASELINE_ENVELOPE["s4_trm20_pp"] - GATE_MARGINS["trm20_pp"],
        "s4_algorithm_tta_s": values["s4_algorithm_tta_s"] is not None
        and values["s4_algorithm_tta_s"]
        <= BASELINE_ENVELOPE["s4_algorithm_tta_s"] * (1.0 + GATE_MARGINS["tta_relative"]),
    }
    strict_wins = [name for name, passed in strict.items() if bool(passed)]
    misses = [name for name, passed in strict.items() if not bool(passed)]
    sole_miss_close = len(misses) == 1 and bool(close[misses[0]])
    gate_passed = len(strict_wins) == 4 or (len(strict_wins) == 3 and sole_miss_close)
    return {
        "values": values,
        "baseline_envelope": BASELINE_ENVELOPE,
        "close_margins": GATE_MARGINS,
        "strict_wins": {name: bool(value) for name, value in strict.items()},
        "within_close_margin": {name: bool(value) for name, value in close.items()},
        "strict_win_count": len(strict_wins),
        "misses": misses,
        "sole_miss_close": bool(sole_miss_close),
        "gate_passed": bool(gate_passed),
    }


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot evaluate an incomplete v12 validation pair")
    rows = [_run_metrics(job) for job in manifest["jobs"]]
    frame = pd.DataFrame(rows)
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    by_scenario = {str(row["scenario_id"]): row for row in rows}
    if set(by_scenario) != set(SCENARIOS):
        raise RuntimeError("v12 validation pair is missing S0 or S4")
    gate = evaluate_gate(
        s0_last50_accuracy=float(by_scenario["S0"]["last50_validation_accuracy"]),
        s4_last50_accuracy=float(by_scenario["S4"]["last50_validation_accuracy"]),
        s4_trm20_pp=float(by_scenario["S4"]["trm20_pp"]),
        s4_algorithm_tta_s=by_scenario["S4"]["algorithm_tta_s"],
    )
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "status": "validation_gate_passed" if gate["gate_passed"] else "validation_gate_failed",
        "selection_split": "validation",
        "seed": SEED,
        "formal_test_access": False,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "selected_candidate": manifest["selected_candidate"],
        "gate_rule": manifest["gate_rule"],
        "gate": gate,
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
                "validation_pair_completed_gate_passed_formal_required"
                if result["status"] == "validation_gate_passed"
                else "validation_pair_completed_gate_failed_stop_before_formal"
            ),
            "current_job_id": None,
            "all_runs_completed": True,
            "performance_sealed_until_terminal": False,
            "result_path": str(RESULT_PATH),
            "gate_passed": bool(result["gate"]["gate_passed"]),
            "strict_win_count": int(result["gate"]["strict_win_count"]),
        }
    )
    _sync(state, manifest)
    atomic_json(STATE_PATH, state)
    return state


def status() -> dict[str, Any]:
    manifest = build_manifest(persist=False)
    return {
        "manifest": manifest,
        "state": json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "worker", "status"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        payload = build_manifest(force=args.force)
    elif args.command == "worker":
        payload = worker()
    else:
        payload = status()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
