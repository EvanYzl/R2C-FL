from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import FORMAL_SEED, PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
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
PLAN_PATH = (
    PROJECT_ROOT
    / "refine-logs"
    / "R2C_V12_TAIL_RECOVERY_OPTIMIZATION_PLAN_20260819_131758.md"
)

VALIDATION_RESULT_PATH = QUEUE_ROOT / "r2c_d3_v12_validation_pair_result.json"
VALIDATION_STATE_PATH = QUEUE_ROOT / "r2c_d3_v12_validation_pair_queue_state.json"
VALIDATION_IMMUTABLE_MANIFEST_PATH = (
    QUEUE_ROOT / "r2c_d3_v12_validation_pair_manifest_20260819T080446.356698Z.json"
)
EXPECTED_VALIDATION_RESULT_SHA256 = (
    "6fefd602a47389a2820c5a6bce9b399c153c9d15b47fdc2103e8cbef27e31af2"
)
EXPECTED_VALIDATION_MANIFEST_SHA256 = (
    "2d1f6ca32503c49157b744195fcc60abb9fac26d2700e220145152fd870f4fb1"
)

V11_RESULT_PATH = QUEUE_ROOT / "r2c_v11_four_dataset_matched_result.json"
V11_BASELINE_RUN_IDS = {
    "S0": "A-R2C-V11MS-D3-S0-B095-s20260811",
    "S4": "A-R2C-V11MS-D3-S4-B095-s20260811",
}

MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v12_formal_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v12_formal_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v12_formal_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v12_formal_result.json"
RUNS_PATH = QUEUE_ROOT / "r2c_d3_v12_formal_runs.parquet"
RUNS_CSV_PATH = QUEUE_ROOT / "r2c_d3_v12_formal_runs.csv"

DATASET_ID = "D3"
SCENARIOS = ("S0", "S4")
ROUNDS = 1000
EVENT_ROUND = 500
TARGET_ACCURACY = 0.7986707616707616
MAX_ATTEMPTS = 3

# Frozen before any v12 formal-test access, from the already audited v11 matched
# D3 test runs. Higher is better except for algorithm TTA.
FORMAL_BASELINE_ENVELOPE = {
    "s0_last50_accuracy": 0.8995851259987707,
    "s4_last50_accuracy": 0.8989850952673631,
    "s4_trm20_pp": 0.03207590657652304,
    "s4_algorithm_tta_s": 216.6765661003301,
}
GATE_MARGINS = {
    "accuracy_fraction": 0.0015,
    "trm20_pp": 0.10,
    "tta_relative": 0.05,
}


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
        "r2c_d3_v12_recovery_screen_queue.py",
        "r2c_d3_v12_validation_pair_queue.py",
        "run.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    values["tests/audit_r2c_run.py"] = sha256_file(
        Path(__file__).resolve().parents[1] / "tests" / "audit_r2c_run.py"
    )
    return values


def _source_hashes() -> dict[str, str]:
    paths = [
        PLAN_PATH,
        VALIDATION_RESULT_PATH,
        VALIDATION_STATE_PATH,
        VALIDATION_IMMUTABLE_MANIFEST_PATH,
        V11_RESULT_PATH,
    ]
    for run_id in V11_BASELINE_RUN_IDS.values():
        run_dir = RUN_ROOT / run_id
        paths.extend(
            [
                run_dir / "_SUCCESS.json",
                run_dir / "result.json",
                run_dir / "tables" / "round_metrics" / "_index.json",
            ]
        )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _ensure_assets() -> dict[str, str]:
    prepare_partition(DATASET_ID, FORMAL_SEED)
    paths = [
        partition_asset_path(DATASET_ID, FORMAL_SEED),
        partition_meta_path(DATASET_ID, FORMAL_SEED),
    ]
    for scenario in SCENARIOS:
        prepare_trace(DATASET_ID, scenario, FORMAL_SEED, rounds=ROUNDS)
        paths.extend(
            [
                trace_asset_path(DATASET_ID, scenario, FORMAL_SEED, ROUNDS),
                trace_meta_path(DATASET_ID, scenario, FORMAL_SEED, ROUNDS),
            ]
        )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _validation_context() -> dict[str, Any]:
    if sha256_file(VALIDATION_RESULT_PATH) != EXPECTED_VALIDATION_RESULT_SHA256:
        raise RuntimeError("v12 validation-pair result hash drift")
    if (
        sha256_file(VALIDATION_IMMUTABLE_MANIFEST_PATH)
        != EXPECTED_VALIDATION_MANIFEST_SHA256
    ):
        raise RuntimeError("v12 validation-pair immutable manifest hash drift")
    result = json.loads(VALIDATION_RESULT_PATH.read_text(encoding="utf-8"))
    state = json.loads(VALIDATION_STATE_PATH.read_text(encoding="utf-8"))
    source_manifest = json.loads(
        VALIDATION_IMMUTABLE_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    selected = dict(result.get("selected_candidate", {}))
    if (
        result.get("status") != "validation_gate_passed"
        or not bool(result.get("gate", {}).get("gate_passed"))
        or int(result.get("gate", {}).get("strict_win_count", -1)) != 4
        or result.get("selection_split") != "validation"
        or bool(result.get("formal_test_access"))
        or bool(result.get("other_dataset_access"))
        or bool(result.get("test_labels_used_for_selection"))
        or state.get("status")
        != "validation_pair_completed_gate_passed_formal_required"
        or int(state.get("completed", -1)) != 2
        or int(state.get("failed", -1)) != 0
        or selected.get("candidate_id") != "P3-B050-D05"
        or float(selected.get("pulse_beta", -1.0)) != 0.5
        or int(selected.get("pulse_rounds", -1)) != 5
    ):
        raise RuntimeError("v12 validation pair did not authorize formal test")
    jobs = list(source_manifest.get("jobs", []))
    if len(jobs) != 2 or jobs[0].get("method_config") != jobs[1].get("method_config"):
        raise RuntimeError("v12 validation-pair configuration mismatch")
    return {
        "result": result,
        "state": state,
        "source_manifest": source_manifest,
        "selected": selected,
        "validation_config": dict(jobs[0]["method_config"]),
    }


def _formal_config(context: dict[str, Any]) -> dict[str, Any]:
    validation_config = dict(context["validation_config"])
    config = dict(validation_config)
    if (
        config.get("r2c_protocol_version") != PROTOCOL_VERSION
        or config.get("r2c_v12_plan_id") != PLAN_ID
        or bool(config.get("r2c_v2_audit_replay"))
        or float(config.get("r2c_v4_primary_deployment_beta", -1.0)) != 0.9
        or float(config.get("r2c_v8_trigger_deployment_beta", -1.0)) != 1.0
        or float(config.get("r2c_v8_recovery_pulse_beta", -1.0)) != 0.5
        or int(config.get("r2c_v8_recovery_pulse_rounds", -1)) != 5
    ):
        raise RuntimeError("v12 locked candidate configuration drift")
    config["r2c_v2_audit_replay"] = True
    changed = sorted(
        key
        for key in set(config) | set(validation_config)
        if config.get(key) != validation_config.get(key)
    )
    if changed != ["r2c_v2_audit_replay"]:
        raise RuntimeError(f"unexpected validation-to-formal config changes: {changed}")
    return config


def _baseline_metrics() -> dict[str, float]:
    s0 = read_chunked_table(RUN_ROOT / V11_BASELINE_RUN_IDS["S0"], "round_metrics")
    s4 = read_chunked_table(RUN_ROOT / V11_BASELINE_RUN_IDS["S4"], "round_metrics")
    s0 = s0.sort_values("round")
    s4 = s4.sort_values("round")
    reached = s4.loc[s4["test_accuracy"].astype(float) >= TARGET_ACCURACY]
    if reached.empty:
        raise RuntimeError("v11 D3 formal baseline never reached TTA target")
    values = {
        "s0_last50_accuracy": float(s0.tail(50)["test_accuracy"].astype(float).mean()),
        "s4_last50_accuracy": float(s4.tail(50)["test_accuracy"].astype(float).mean()),
        "s4_trm20_pp": float(derive_window_metrics(s4)["trm20_pp"]),
        "s4_algorithm_tta_s": float(
            reached.sort_values("round").iloc[0]["algorithm_elapsed_s"]
        ),
    }
    for key, expected in FORMAL_BASELINE_ENVELOPE.items():
        if abs(values[key] - expected) > 1.0e-12:
            raise RuntimeError(f"frozen v11 formal baseline drift for {key}")
    return values


def _source_lineage(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "validation_result_sha256": sha256_file(VALIDATION_RESULT_PATH),
        "validation_immutable_manifest_sha256": sha256_file(
            VALIDATION_IMMUTABLE_MANIFEST_PATH
        ),
        "validation_strict_win_count": int(
            context["result"]["gate"]["strict_win_count"]
        ),
        "validation_gate_values": dict(context["result"]["gate"]["values"]),
        "formal_baseline_run_ids": dict(V11_BASELINE_RUN_IDS),
        "formal_baseline_envelope": _baseline_metrics(),
    }


def _job(scenario: str, context: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(context["selected"]["candidate_id"])
    run_id = f"A-R2C-D3-{scenario}-V12FORMAL-{candidate_id}-s{FORMAL_SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v12_locked_formal_pilot",
        "mode": "formal",
        "method_id": "R2C-FL",
        "dataset_id": DATASET_ID,
        "scenario_id": scenario,
        "rounds": ROUNDS,
        "method_config": _formal_config(context),
        "block_id": "A-R2C-D3-V12-FORMAL-PILOT",
        "seed": FORMAL_SEED,
        "partition_seed": FORMAL_SEED,
        "trace_seed": FORMAL_SEED,
        "evaluation_split": "test",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "candidate_id": candidate_id,
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
        "rounds_per_job",
        "event_round",
        "formal_test_access",
        "other_dataset_access_before_pilot_gate",
        "test_labels_used_for_selection",
        "candidate_locked_before_formal",
        "performance_sealed_until_terminal",
        "selected_candidate",
        "baseline_envelope",
        "gate_margins",
        "gate_rule",
        "source_lineage",
        "source_hashes",
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
            raise RuntimeError("refusing to rebuild a started v12 formal manifest")
    if int(FORMAL_SEED) != 20260811:
        raise RuntimeError(f"expected formal seed 20260811, found {FORMAL_SEED}")
    context = _validation_context()
    jobs = [_job(scenario, context) for scenario in SCENARIOS]
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_only_v12_locked_formal_pilot",
        "plan_id": PLAN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_split": "test",
        "seed": FORMAL_SEED,
        "rounds_per_job": ROUNDS,
        "event_round": EVENT_ROUND,
        "formal_test_access": True,
        "other_dataset_access_before_pilot_gate": False,
        "test_labels_used_for_selection": False,
        "candidate_locked_before_formal": True,
        "performance_sealed_until_terminal": True,
        "selected_candidate": {
            "candidate_id": "P3-B050-D05",
            "pulse_beta": 0.5,
            "pulse_rounds": 5,
        },
        "baseline_envelope": dict(FORMAL_BASELINE_ENVELOPE),
        "gate_margins": dict(GATE_MARGINS),
        "gate_rule": (
            "pass iff four strict wins, or exactly three strict wins and the sole miss "
            "is within 0.15 pp accuracy, 0.10 pp TRM20, or 5 percent relative TTA"
        ),
        "source_lineage": _source_lineage(context),
        "source_hashes": _source_hashes(),
        "asset_hashes": _ensure_assets(),
        "implementation_hashes": _implementation_hashes(),
        "job_order": [job["job_id"] for job in jobs],
        "max_attempts": MAX_ATTEMPTS,
        "jobs": jobs,
    }
    manifest["frozen_spec_hash"] = config_hash(_frozen_spec(manifest))
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        immutable_path = QUEUE_ROOT / f"r2c_d3_v12_formal_manifest_{stamp}.json"
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
                "other_dataset_access": False,
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
            audit_log = QUEUE_ROOT / "worker_logs" / f"{path.name}.v12formal.reconcile.audit.log"
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
    context = _validation_context()
    if (
        manifest.get("scope") != "D3_only_v12_locked_formal_pilot"
        or manifest.get("plan_id") != PLAN_ID
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("evaluation_split") != "test"
        or int(manifest.get("seed", -1)) != FORMAL_SEED
        or int(manifest.get("rounds_per_job", -1)) != ROUNDS
        or not bool(manifest.get("formal_test_access"))
        or bool(manifest.get("other_dataset_access_before_pilot_gate"))
        or bool(manifest.get("test_labels_used_for_selection"))
        or not bool(manifest.get("candidate_locked_before_formal"))
        or not bool(manifest.get("performance_sealed_until_terminal"))
        or manifest.get("baseline_envelope") != FORMAL_BASELINE_ENVELOPE
        or manifest.get("gate_margins") != GATE_MARGINS
    ):
        raise RuntimeError("v12 formal scope drift")
    if manifest.get("source_lineage") != _source_lineage(context):
        raise RuntimeError("v12 formal source lineage drift")
    if manifest.get("source_hashes") != _source_hashes():
        raise RuntimeError("v12 formal source hash drift")
    if manifest.get("asset_hashes") != _ensure_assets():
        raise RuntimeError("v12 formal asset drift")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("v12 formal implementation drift after freeze")
    if manifest.get("frozen_spec_hash") != config_hash(_frozen_spec(manifest)):
        raise RuntimeError("v12 formal frozen specification drift")
    jobs = list(manifest.get("jobs", []))
    if len(jobs) != 2 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("v12 formal order drift")
    expected_config = _formal_config(context)
    for job, scenario in zip(jobs, SCENARIOS):
        if (
            job.get("dataset_id") != DATASET_ID
            or job.get("scenario_id") != scenario
            or job.get("mode") != "formal"
            or job.get("evaluation_split") != "test"
            or int(job.get("rounds", -1)) != ROUNDS
            or any(
                int(job.get(key, -1)) != FORMAL_SEED
                for key in ("seed", "partition_seed", "trace_seed")
            )
            or job.get("method_config") != expected_config
            or job.get("candidate_id") != "P3-B050-D05"
            or not bool(job.get("full_logging"))
            or bool(job.get("test_labels_used_for_selection"))
        ):
            raise RuntimeError(f"v12 formal protocol drift in {job.get('job_id')}")


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
        or str(run_manifest["source_kind"]) != "REPRODUCED"
        or actual_job.get("mode") != "formal"
        or actual_job.get("evaluation_split") != "test"
        or actual_job.get("method_config") != job.get("method_config")
        or bool(actual_job.get("test_labels_used_for_selection"))
    ):
        raise RuntimeError(f"v12 formal run contract mismatch: {run_id}")
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
        raise RuntimeError(f"v12 formal forbidden pulse input: {run_id}")
    trm20_pp: float | None = None
    if scenario == "S0":
        if trigger.any() or hold.any() or pulse.any() or response.any():
            raise RuntimeError(f"v12 formal S0 no-trigger mismatch: {run_id}")
    elif scenario == "S4":
        event = rounds.loc[
            pd.to_numeric(rounds["event_offset_round"], errors="coerce") == 0
        ]
        if (
            len(event) != 1
            or int(event.iloc[0]["round"]) != EVENT_ROUND
            or rounds.loc[trigger, "round"].astype(int).tolist() != [EVENT_ROUND]
            or rounds.loc[hold, "round"].astype(int).tolist() != [EVENT_ROUND]
            or rounds.loc[pulse, "round"].astype(int).tolist()
            != list(range(EVENT_ROUND + 1, EVENT_ROUND + 6))
            or rounds.loc[response, "round"].astype(int).tolist()
            != list(range(EVENT_ROUND, EVENT_ROUND + 6))
        ):
            raise RuntimeError(f"v12 formal S4 pulse contract mismatch: {run_id}")
        trm20_pp = float(derive_window_metrics(rounds)["trm20_pp"])
    else:
        raise RuntimeError(f"unexpected v12 formal scenario: {scenario}")
    tta_round, tta_s = _time_to_accuracy(rounds)
    return {
        "scenario_id": scenario,
        "run_id": run_id,
        "last50_accuracy": float(rounds.tail(50)["test_accuracy"].astype(float).mean()),
        "trm20_pp": trm20_pp,
        "tta_round": tta_round,
        "algorithm_tta_s": tta_s,
        "trigger_count": int(trigger.sum()),
        "hold_count": int(hold.sum()),
        "pulse_count": int(pulse.sum()),
        "response_count": int(response.sum()),
        "source_kind": str(run_manifest["source_kind"]),
        "test_labels_used_for_selection": False,
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
    base = FORMAL_BASELINE_ENVELOPE
    strict = {
        "s0_last50_accuracy": values["s0_last50_accuracy"] > base["s0_last50_accuracy"],
        "s4_last50_accuracy": values["s4_last50_accuracy"] > base["s4_last50_accuracy"],
        "s4_trm20_pp": values["s4_trm20_pp"] > base["s4_trm20_pp"],
        "s4_algorithm_tta_s": values["s4_algorithm_tta_s"] is not None
        and values["s4_algorithm_tta_s"] < base["s4_algorithm_tta_s"],
    }
    close = {
        "s0_last50_accuracy": values["s0_last50_accuracy"]
        >= base["s0_last50_accuracy"] - GATE_MARGINS["accuracy_fraction"],
        "s4_last50_accuracy": values["s4_last50_accuracy"]
        >= base["s4_last50_accuracy"] - GATE_MARGINS["accuracy_fraction"],
        "s4_trm20_pp": values["s4_trm20_pp"]
        >= base["s4_trm20_pp"] - GATE_MARGINS["trm20_pp"],
        "s4_algorithm_tta_s": values["s4_algorithm_tta_s"] is not None
        and values["s4_algorithm_tta_s"]
        <= base["s4_algorithm_tta_s"] * (1.0 + GATE_MARGINS["tta_relative"]),
    }
    wins = [name for name, passed in strict.items() if bool(passed)]
    misses = [name for name, passed in strict.items() if not bool(passed)]
    sole_miss_close = len(misses) == 1 and bool(close[misses[0]])
    gate_passed = len(wins) == 4 or (len(wins) == 3 and sole_miss_close)
    return {
        "values": values,
        "baseline_envelope": dict(base),
        "close_margins": dict(GATE_MARGINS),
        "strict_wins": {name: bool(value) for name, value in strict.items()},
        "within_close_margin": {name: bool(value) for name, value in close.items()},
        "strict_win_count": len(wins),
        "misses": misses,
        "sole_miss_close": bool(sole_miss_close),
        "gate_passed": bool(gate_passed),
    }


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("cannot freeze incomplete v12 formal pilot")
    rows = [_run_metrics(job) for job in manifest["jobs"]]
    frame = pd.DataFrame(rows)
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    by_scenario = {str(row["scenario_id"]): row for row in rows}
    gate = evaluate_gate(
        s0_last50_accuracy=float(by_scenario["S0"]["last50_accuracy"]),
        s4_last50_accuracy=float(by_scenario["S4"]["last50_accuracy"]),
        s4_trm20_pp=float(by_scenario["S4"]["trm20_pp"]),
        s4_algorithm_tta_s=by_scenario["S4"]["algorithm_tta_s"],
    )
    structural_checks = {
        "s0_no_trigger": int(by_scenario["S0"]["trigger_count"]) == 0,
        "s0_no_hold": int(by_scenario["S0"]["hold_count"]) == 0,
        "s0_no_pulse": int(by_scenario["S0"]["pulse_count"]) == 0,
        "s4_one_trigger": int(by_scenario["S4"]["trigger_count"]) == 1,
        "s4_one_hold": int(by_scenario["S4"]["hold_count"]) == 1,
        "s4_five_pulse_rounds": int(by_scenario["S4"]["pulse_count"]) == 5,
        "s4_six_response_rounds": int(by_scenario["S4"]["response_count"]) == 6,
    }
    gate_passed = bool(gate["gate_passed"] and all(structural_checks.values()))
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "status": "formal_pilot_gate_passed" if gate_passed else "formal_pilot_gate_failed",
        "dataset_id": DATASET_ID,
        "source_kind": "REPRODUCED",
        "evaluation_split": "test",
        "formal_interpretation": (
            "outcome_informed_engineering_reevaluation_not_untouched_confirmation"
        ),
        "test_labels_used_for_selection": False,
        "other_dataset_access_authorized": gate_passed,
        "selected_candidate": dict(manifest["selected_candidate"]),
        "gate": gate,
        "structural_checks": structural_checks,
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
            "status": (
                "formal_pilot_completed_gate_passed_remaining_datasets_required"
                if result["status"] == "formal_pilot_gate_passed"
                else "formal_pilot_completed_gate_failed_stop"
            ),
            "current_job_id": None,
            "all_runs_completed": True,
            "performance_sealed_until_terminal": False,
            "other_dataset_access": bool(result["other_dataset_access_authorized"]),
            "gate_passed": bool(result["other_dataset_access_authorized"]),
            "strict_win_count": int(result["gate"]["strict_win_count"]),
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
