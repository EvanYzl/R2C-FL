from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import r2c_d3_v14_validation_screen_erratum_queue as m1_authority
from . import r2c_d3_v14_validation_screen_queue as screen
from .config import PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .data import partition_asset_path, partition_meta_path, prepare_partition
from .logging_io import read_chunked_table
from .r2c_post_event_lower_quartile_accuracy20 import derive_lqa20_percent
from .r2c_v7 import PROTOCOL_VERSION as V11_PROTOCOL_VERSION
from .r2c_v14 import (
    FAST_BETA,
    PROTOCOL_VERSION,
    WARMUP_ROUNDS,
    candidate_recovery_rounds,
    candidate_stable_beta,
)
from .traces import prepare_trace, trace_asset_path, trace_meta_path
from .utils import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    config_hash,
    sha256_file,
    utc_now,
)


PLAN_ID = screen.PLAN_ID
PLAN_PATH = screen.PLAN_PATH
M1_MANIFEST_PATH = m1_authority.MANIFEST_PATH
M1_STATE_PATH = m1_authority.STATE_PATH
M1_RESULT_PATH = m1_authority.RESULT_PATH
M1_RUNS_PATH = m1_authority.RUNS_PATH
M1_RUNS_CSV_PATH = m1_authority.RUNS_CSV_PATH

MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v14_development_pair_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v14_development_pair_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v14_development_pair_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v14_development_pair_result.json"
RUNS_PATH = QUEUE_ROOT / "r2c_d3_v14_development_pair_runs.parquet"
RUNS_CSV_PATH = QUEUE_ROOT / "r2c_d3_v14_development_pair_runs.csv"

DATASET_ID = "D3"
SCENARIOS = ("S0", "S4")
METHOD_ORDER = (("v11", "S0"), ("v11", "S4"), ("v14", "S0"), ("v14", "S4"))
SEED = 20260809
ROUNDS = 1000
EVENT_ROUND = 500
TARGET_ACCURACY = screen.TARGET_ACCURACY
MAX_ATTEMPTS = 3
DECISION_ATOL = 1.0e-12

GATE_MARGINS = {
    "s0_last50_pp": 0.15,
    "s4_last50_pp": 0.15,
    "s4_lqa20_pp": 0.50,
    "s4_algorithm_tta_relative": 0.05,
}

# Learning must be bit-identical. Deployment hashes are intentionally excluded:
# CMTR changes only the deployment router while preserving the global learner.
LEARNING_IDENTITY_COLUMNS = ("round", "global_model_hash")


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
        "r2c_v14.py",
        "r2c_post_event_lower_quartile_accuracy20.py",
        "run.py",
        "run_v14.py",
        "r2c_d3_v14_validation_screen_queue.py",
        "r2c_d3_v14_validation_screen_erratum_queue.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    tests = Path(__file__).resolve().parents[1] / "tests"
    for name in (
        "audit_r2c_run.py",
        "audit_r2c_v14_run.py",
        "audit_r2c_v14_run_erratum.py",
        "test_r2c_v14.py",
        "test_r2c_d3_v14_validation_screen_queue.py",
        "test_r2c_d3_v14_validation_screen_erratum_queue.py",
        "test_r2c_d3_v14_development_pair_queue.py",
    ):
        values[f"tests/{name}"] = sha256_file(tests / name)
    return values


def _assert_m1_immutable_authority(
    state: dict[str, Any], manifest: dict[str, Any]
) -> Path:
    immutable_path = Path(str(state.get("immutable_manifest_path", ""))).resolve()
    immutable_sha256 = str(state.get("immutable_manifest_sha256", ""))
    if (
        not immutable_path.is_file()
        or sha256_file(immutable_path).lower() != immutable_sha256.lower()
    ):
        raise RuntimeError("v14 M1 immutable authority mismatch")
    immutable = json.loads(immutable_path.read_text(encoding="utf-8"))
    if m1_authority._frozen_spec(immutable) != m1_authority._frozen_spec(manifest):
        raise RuntimeError("v14 M1 active/immutable frozen specifications diverged")
    return immutable_path


def _read_audit_payload(path: Path) -> dict[str, Any]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"empty audit log: {path}")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid audit log JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid audit log payload: {path}")
    return payload


def _is_valid_erratum_audit_payload(
    payload: dict[str, Any], run_id: str, expected_rounds: int = ROUNDS
) -> bool:
    window = payload.get("canonical_window_contract")
    return bool(
        payload.get("status") == "passed"
        and payload.get("run_id") == run_id
        and payload.get("audit_erratum_id") == m1_authority.ERRATUM_ID
        and not bool(payload.get("recorded_tables_mutated"))
        and int(payload.get("round_rows", -1)) == expected_rounds
        and isinstance(window, dict)
        and int(window.get("before_count", -1)) == 20
        and int(window.get("after_count", -1)) == 20
        and int(window.get("event_count", -1)) == 1
    )


def _validated_m1_erratum_audit_log(run_id: str) -> Path:
    # The preserved first candidate has both the superseded failed audit and the
    # continuation reconciliation audit. Prefer reconciliation and never accept
    # existence alone as evidence of a passing audit.
    candidates = (
        QUEUE_ROOT / "worker_logs" / f"{run_id}.reconcile.v14.audit.log",
        QUEUE_ROOT / "worker_logs" / f"{run_id}.v14.audit.log",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = _read_audit_payload(path)
        except RuntimeError:
            continue
        if _is_valid_erratum_audit_payload(payload, run_id):
            return path
    raise RuntimeError("v14 M1 selected run has no passing erratum audit evidence")


def _m1_context() -> dict[str, Any]:
    for path in (M1_MANIFEST_PATH, M1_STATE_PATH):
        if not path.exists():
            raise RuntimeError("v14 M1 is not terminal with one eligible selected candidate")
    if not M1_RESULT_PATH.exists():
        raise RuntimeError("v14 M1 is not terminal with one eligible selected candidate")
    manifest = json.loads(M1_MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(M1_STATE_PATH.read_text(encoding="utf-8"))
    result = json.loads(M1_RESULT_PATH.read_text(encoding="utf-8"))
    m1_authority._assert_frozen_manifest(manifest)
    immutable_path = _assert_m1_immutable_authority(state, manifest)
    selected = result.get("selected")
    if (
        state.get("status") != "m1_completed_selected_m2_required"
        or int(state.get("completed", -1)) != len(screen.CANDIDATES)
        or int(state.get("failed", -1)) != 0
        or not bool(state.get("all_runs_completed"))
        or bool(state.get("performance_sealed_until_terminal"))
        or bool(state.get("formal_test_access"))
        or bool(state.get("other_dataset_access"))
        or result.get("status") != "selected"
        or not isinstance(selected, dict)
        or result.get("selection_split") != "validation"
        or bool(result.get("formal_test_access"))
        or bool(result.get("other_dataset_access"))
        or bool(result.get("test_labels_used_for_selection"))
        or result.get("frozen_spec_hash") != manifest.get("frozen_spec_hash")
    ):
        raise RuntimeError("v14 M1 is not terminal with one eligible selected candidate")
    candidate_id = str(selected.get("candidate_id"))
    if (
        candidate_id not in screen.CANDIDATES
        or not bool(selected.get("validation_eligible"))
        or not bool(selected.get("global_learning_hash_identity"))
    ):
        raise RuntimeError("v14 M1 selected candidate is not eligible")
    jobs = [job for job in manifest["jobs"] if job.get("candidate_id") == candidate_id]
    if len(jobs) != 1:
        raise RuntimeError("v14 M1 selected candidate is not unique")
    selected_job = dict(jobs[0])
    run_id = str(selected_job.get("actual_run_id") or "")
    if (
        selected_job.get("status") != "completed"
        or not run_id
        or str(selected.get("run_id")) != run_id
    ):
        raise RuntimeError("v14 M1 selected run lineage is incomplete")
    run_dir = RUN_ROOT / run_id
    audit_log = _validated_m1_erratum_audit_log(run_id)
    if not (run_dir / "_SUCCESS.json").exists():
        raise RuntimeError("v14 M1 selected run audit/success evidence is missing")
    return {
        "manifest": manifest,
        "state": state,
        "result": result,
        "selected": dict(selected),
        "selected_job": selected_job,
        "selected_run_id": run_id,
        "candidate_id": candidate_id,
        "immutable_path": immutable_path,
        "audit_log": audit_log,
    }


def _source_lineage(context: dict[str, Any]) -> dict[str, str]:
    selected_dir = RUN_ROOT / str(context["selected_run_id"])
    control_dir = screen.CONTROL_RUN_DIR
    paths = (
        PLAN_PATH,
        Path(context["immutable_path"]),
        M1_MANIFEST_PATH,
        M1_STATE_PATH,
        M1_RESULT_PATH,
        M1_RUNS_PATH,
        M1_RUNS_CSV_PATH,
        Path(context["audit_log"]),
        selected_dir / "job.json",
        selected_dir / "result.json",
        selected_dir / "_SUCCESS.json",
        selected_dir / "run_manifest.parquet",
        selected_dir / "tables" / "round_metrics" / "_index.json",
        control_dir / "job.json",
        control_dir / "result.json",
        control_dir / "_SUCCESS.json",
        control_dir / "run_manifest.parquet",
        control_dir / "tables" / "round_metrics" / "_index.json",
    )
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _v11_config() -> dict[str, Any]:
    config = screen._source_config()
    config["r2c_v2_audit_replay"] = False
    if (
        config.get("r2c_protocol_version") != V11_PROTOCOL_VERSION
        or config.get("r2c_v4_deployment_ema_betas") != [0.95]
        or float(config.get("r2c_v4_primary_deployment_beta", -1.0)) != 0.95
        or float(config.get("r2c_v7_trigger_deployment_beta", -1.0)) != 1.0
    ):
        raise RuntimeError("v14 M2 v11 comparator configuration drift")
    return config


def _v14_config(candidate_id: str) -> dict[str, Any]:
    config = screen._candidate_config(candidate_id)
    config["r2c_v2_audit_replay"] = False
    stable_beta = candidate_stable_beta(candidate_id)
    if (
        config.get("r2c_protocol_version") != PROTOCOL_VERSION
        or config.get("r2c_v4_deployment_ema_betas") != [FAST_BETA, stable_beta]
        or float(config.get("r2c_v4_primary_deployment_beta", -1.0)) != stable_beta
        or config.get("r2c_v14_candidate_id") != candidate_id
        or "r2c_v7_trigger_deployment_beta" in config
    ):
        raise RuntimeError("v14 M2 selected CMTR configuration drift")
    return config


def _learning_config(config: dict[str, Any]) -> dict[str, Any]:
    value = dict(config)
    for key in (
        "r2c_protocol_version",
        "r2c_v4_deployment_ema_betas",
        "r2c_v4_primary_deployment_beta",
        "r2c_v7_trigger_deployment_beta",
        "r2c_v14_candidate_id",
        "r2c_v14_fast_beta",
        "r2c_v14_warmup_rounds",
        "r2c_v14_plan_id",
    ):
        value.pop(key, None)
    return value


def _asset_paths() -> tuple[Path, ...]:
    return (
        partition_asset_path(DATASET_ID, SEED),
        partition_meta_path(DATASET_ID, SEED),
        trace_asset_path(DATASET_ID, "S0", SEED, ROUNDS),
        trace_meta_path(DATASET_ID, "S0", SEED, ROUNDS),
        trace_asset_path(DATASET_ID, "S4", SEED, ROUNDS),
        trace_meta_path(DATASET_ID, "S4", SEED, ROUNDS),
    )


def _ensure_assets() -> dict[str, str]:
    prepare_partition(DATASET_ID, SEED)
    for scenario in SCENARIOS:
        prepare_trace(DATASET_ID, scenario, SEED, rounds=ROUNDS)
    paths = _asset_paths()
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _current_asset_hashes() -> dict[str, str]:
    paths = _asset_paths()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _job(method_version: str, scenario: str, context: dict[str, Any]) -> dict[str, Any]:
    if (method_version, scenario) not in METHOD_ORDER:
        raise ValueError("job is outside immutable v14 M2 method/scenario order")
    candidate_id = str(context["candidate_id"])
    method_label = "V11-B095" if method_version == "v11" else f"V14-{candidate_id}"
    run_id = f"A-R2C-D3-{scenario}-V14M2-{method_label}-s{SEED}"
    job: dict[str, Any] = {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v14_independent_seed_matched_development_pair",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "method_version": method_version,
        "dataset_id": DATASET_ID,
        "scenario_id": scenario,
        "rounds": ROUNDS,
        "method_config": _v11_config() if method_version == "v11" else _v14_config(candidate_id),
        "block_id": "A-R2C-D3-V14-M2-MATCHED-DEVELOPMENT",
        "seed": SEED,
        "partition_seed": SEED,
        "trace_seed": SEED,
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "selected_candidate_id": candidate_id,
        "source_screen_run_id": str(context["selected_run_id"]),
        "formal_test_access": False,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }
    if method_version == "v14":
        job.update(
            {
                "fast_beta": FAST_BETA,
                "stable_beta": candidate_stable_beta(candidate_id),
                "warmup_rounds": WARMUP_ROUNDS,
                "recovery_rounds": candidate_recovery_rounds(candidate_id),
                "deployment_state_count": 2,
            }
        )
    return job


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
        "formal_test_access",
        "other_dataset_access",
        "test_labels_used_for_selection",
        "candidate_locked_before_pair",
        "performance_sealed_until_terminal",
        "selected_candidate",
        "gate_metrics",
        "gate_margins",
        "gate_rule",
        "learning_identity_columns",
        "learning_identity_rule",
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
    # The terminal M1 gate is deliberately first: no M2 file or seed-09 asset can
    # be created merely by invoking build while M1 is incomplete/ineligible.
    context = _m1_context()
    if persist and MANIFEST_PATH.exists() and not force:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        _assert_frozen_manifest(manifest)
        return manifest
    if force and MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if any(
            job.get("status") != "pending" or int(job.get("attempts", 0))
            for job in existing["jobs"]
        ):
            raise RuntimeError("Refusing to rebuild a started v14 M2 manifest")
    jobs = [_job(method, scenario, context) for method, scenario in METHOD_ORDER]
    if [(job["method_version"], job["scenario_id"]) for job in jobs] != list(METHOD_ORDER):
        raise AssertionError("v14 M2 order must be v11 S0/S4 then v14 S0/S4")
    if _learning_config(_v11_config()) != _learning_config(_v14_config(context["candidate_id"])):
        raise RuntimeError("v14 M2 learning configurations are not exactly matched")
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_validation_seed20260809_v11_v14_matched_pair",
        "plan_id": PLAN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "comparator_protocol_version": V11_PROTOCOL_VERSION,
        "selection_split": "validation",
        "seed": SEED,
        "rounds_per_job": ROUNDS,
        "event_round": EVENT_ROUND,
        "target_accuracy": TARGET_ACCURACY,
        "formal_test_access": False,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "candidate_locked_before_pair": True,
        "performance_sealed_until_terminal": True,
        "selected_candidate": {
            "candidate_id": str(context["candidate_id"]),
            "source_screen_run_id": str(context["selected_run_id"]),
            "source_frozen_spec_hash": str(context["manifest"]["frozen_spec_hash"]),
        },
        "gate_metrics": (
            "S0 last50 accuracy; S4 last50 accuracy; S4 post-event LQA20; "
            "S4 algorithm TTA"
        ),
        "gate_margins": GATE_MARGINS,
        "gate_rule": (
            "pass iff four strict wins, or exactly three strict wins and the sole miss "
            "is close: either last50 no worse than 0.15 pp, LQA20 no worse than "
            "0.50 pp, or TTA no worse than 5 percent relative; equality is not a "
            "strict win; exact global learning-model hash identity in both S0 and S4 "
            "is an additional mandatory structural gate"
        ),
        "learning_identity_columns": list(LEARNING_IDENTITY_COLUMNS),
        "learning_identity_rule": "exact rowwise identity in both S0 and S4",
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
        immutable_path = QUEUE_ROOT / f"r2c_d3_v14_development_pair_manifest_{stamp}.json"
        atomic_json(immutable_path, manifest)
        atomic_json(MANIFEST_PATH, manifest)
        now = utc_now()
        atomic_json(
            STATE_PATH,
            {
                "status": "ready",
                "created_utc": now,
                "updated_utc": now,
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


def _assert_immutable_authority(state: dict[str, Any], manifest: dict[str, Any]) -> None:
    immutable_path = Path(str(state.get("immutable_manifest_path", "")))
    if (
        not immutable_path.is_file()
        or sha256_file(immutable_path) != state.get("immutable_manifest_sha256")
    ):
        raise RuntimeError("v14 M2 immutable manifest hash mismatch")
    immutable = json.loads(immutable_path.read_text(encoding="utf-8"))
    if _frozen_spec(immutable) != _frozen_spec(manifest):
        raise RuntimeError("v14 M2 active/immutable frozen specifications diverged")


def _assert_frozen_manifest(manifest: dict[str, Any]) -> None:
    context = _m1_context()
    expected_selected = {
        "candidate_id": str(context["candidate_id"]),
        "source_screen_run_id": str(context["selected_run_id"]),
        "source_frozen_spec_hash": str(context["manifest"]["frozen_spec_hash"]),
    }
    if (
        manifest.get("scope") != "D3_validation_seed20260809_v11_v14_matched_pair"
        or manifest.get("plan_id") != PLAN_ID
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("comparator_protocol_version") != V11_PROTOCOL_VERSION
        or manifest.get("selection_split") != "validation"
        or int(manifest.get("seed", -1)) != SEED
        or int(manifest.get("rounds_per_job", -1)) != ROUNDS
        or int(manifest.get("event_round", -1)) != EVENT_ROUND
        or float(manifest.get("target_accuracy", -1.0)) != TARGET_ACCURACY
        or bool(manifest.get("formal_test_access"))
        or bool(manifest.get("other_dataset_access"))
        or bool(manifest.get("test_labels_used_for_selection"))
        or not bool(manifest.get("candidate_locked_before_pair"))
        or not bool(manifest.get("performance_sealed_until_terminal"))
        or manifest.get("selected_candidate") != expected_selected
        or manifest.get("gate_margins") != GATE_MARGINS
        or manifest.get("learning_identity_columns") != list(LEARNING_IDENTITY_COLUMNS)
    ):
        raise RuntimeError("v14 M2 manifest scope drift")
    if manifest.get("source_lineage") != _source_lineage(context):
        raise RuntimeError("v14 M2 source lineage drift")
    if manifest.get("asset_hashes") != _current_asset_hashes():
        raise RuntimeError("v14 M2 seed-matched asset drift")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("v14 M2 implementation drift after freeze")
    if manifest.get("frozen_spec_hash") != config_hash(_frozen_spec(manifest)):
        raise RuntimeError("v14 M2 frozen specification drift")
    jobs = list(manifest.get("jobs", []))
    if len(jobs) != 4 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("v14 M2 job order drift")
    if _learning_config(_v11_config()) != _learning_config(_v14_config(context["candidate_id"])):
        raise RuntimeError("v14 M2 learning configuration drift")
    for job, (method_version, scenario) in zip(jobs, METHOD_ORDER):
        expected_config = (
            _v11_config()
            if method_version == "v11"
            else _v14_config(str(context["candidate_id"]))
        )
        if (
            job.get("method_version") != method_version
            or job.get("scenario_id") != scenario
            or job.get("dataset_id") != DATASET_ID
            or job.get("evaluation_split") != "validation"
            or int(job.get("rounds", -1)) != ROUNDS
            or any(int(job.get(key, -1)) != SEED for key in ("seed", "partition_seed", "trace_seed"))
            or job.get("method_config") != expected_config
            or job.get("selected_candidate_id") != context["candidate_id"]
            or job.get("source_screen_run_id") != context["selected_run_id"]
            or bool(job.get("formal_test_access"))
            or bool(job.get("other_dataset_access"))
            or bool(job.get("test_labels_used_for_selection"))
        ):
            raise RuntimeError(f"v14 M2 job drift: {job.get('job_id')}")


def _audit_script_name(job: dict[str, Any]) -> str:
    method_version = str(job.get("method_version"))
    scenario = str(job.get("scenario_id"))
    rounds = int(job.get("rounds", ROUNDS))
    if method_version == "v11":
        return "audit_r2c_run.py"
    if method_version == "v14" and scenario == "S0":
        return "audit_r2c_v14_run.py"
    if method_version == "v14" and scenario == "S4" and rounds >= 520:
        return "audit_r2c_v14_run_erratum.py"
    if method_version == "v14" and scenario == "S4":
        return "audit_r2c_v14_run.py"
    raise RuntimeError(f"unsupported v14 M2 audit target: {method_version}/{scenario}")


def _audit_log_passed(job: dict[str, Any], run_id: str, log_path: Path) -> bool:
    try:
        payload = _read_audit_payload(log_path)
    except RuntimeError:
        return False
    if payload.get("status") != "passed" or payload.get("run_id") != run_id:
        return False
    if _audit_script_name(job) == "audit_r2c_v14_run_erratum.py":
        return _is_valid_erratum_audit_payload(
            payload, run_id, int(job.get("rounds", ROUNDS))
        )
    return True


def _audit_run(job: dict[str, Any], run_id: str, log_path: Path) -> subprocess.CompletedProcess[str]:
    script_name = _audit_script_name(job)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        return subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "tests" / script_name),
                str(RUN_ROOT / run_id),
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


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
            audit_log = (
                QUEUE_ROOT
                / "worker_logs"
                / f"{path.name}.m2.reconcile.{job['method_version']}.audit.log"
            )
            audit = _audit_run(job, path.name, audit_log)
            if audit.returncode == 0 and _audit_log_passed(job, path.name, audit_log):
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
    reached = rounds.loc[rounds["test_accuracy"].astype(float) >= TARGET_ACCURACY].sort_values("round")
    if reached.empty:
        return None, None
    first = reached.iloc[0]
    return int(first["round"]), float(first["algorithm_elapsed_s"])


def _audit_v14_contract(rounds: pd.DataFrame, scenario: str, candidate_id: str, run_id: str) -> None:
    stable_beta = candidate_stable_beta(candidate_id)
    recovery_rounds = candidate_recovery_rounds(candidate_id)
    trigger_actual = rounds.loc[rounds["telemetry_shift_trigger"].astype(bool), "round"].astype(int).tolist()
    hold_actual = rounds.loc[rounds["deployment_quarantine_applied"].astype(bool), "round"].astype(int).tolist()
    recovery_actual = rounds.loc[rounds["deployment_cmtr_recovery_applied"].astype(bool), "round"].astype(int).tolist()
    warmup_actual = rounds.loc[rounds["deployment_cmtr_warmup_applied"].astype(bool), "round"].astype(int).tolist()
    if scenario == "S0":
        expected_trigger: list[int] = []
        expected_recovery: list[int] = []
    elif scenario == "S4":
        expected_trigger = [EVENT_ROUND]
        expected_recovery = list(range(EVENT_ROUND + 1, EVENT_ROUND + recovery_rounds + 1))
    else:
        raise RuntimeError(f"unexpected v14 M2 scenario: {scenario}")
    selected_beta = rounds["selected_deployment_beta"].astype(float)
    expected_fast_rounds = set(range(1, WARMUP_ROUNDS + 1)) | set(expected_recovery)
    actual_fast_rounds = set(
        rounds.loc[np.isclose(selected_beta, FAST_BETA, rtol=0.0, atol=0.0), "round"].astype(int)
    )
    forbidden_columns = (
        "deployment_cmtr_labels_used",
        "deployment_cmtr_validation_predictions_used",
        "deployment_cmtr_test_predictions_used",
        "deployment_cmtr_scenario_metadata_used",
        "deployment_cmtr_event_round_used",
        "deployment_cmtr_future_trace_used",
        "deployment_cmtr_raw_global_deployment_used",
    )
    forbidden_any = bool(
        np.column_stack([rounds[column].astype(bool).to_numpy() for column in forbidden_columns]).any()
    )
    if (
        trigger_actual != expected_trigger
        or hold_actual != expected_trigger
        or recovery_actual != expected_recovery
        or warmup_actual != list(range(1, WARMUP_ROUNDS + 1))
        or actual_fast_rounds != expected_fast_rounds
        or not rounds["deployment_cmtr_state_server_only"].astype(bool).all()
        or forbidden_any
    ):
        raise RuntimeError(f"v14 M2 CMTR causal contract mismatch: {run_id}")
    if scenario == "S4" and not np.isclose(
        rounds.loc[rounds["round"] == EVENT_ROUND, "selected_deployment_beta"].iloc[0],
        stable_beta,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError(f"v14 M2 trigger hold deployment mismatch: {run_id}")


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
        or bool(actual_job.get("other_dataset_access"))
        or bool(actual_job.get("test_labels_used_for_selection"))
    ):
        raise RuntimeError(f"v14 M2 run contract mismatch: {run_id}")
    method_version = str(job["method_version"])
    scenario = str(job["scenario_id"])
    trigger = rounds["telemetry_shift_trigger"].astype(bool)
    hold = rounds["deployment_quarantine_applied"].astype(bool)
    if scenario == "S0":
        if trigger.any() or hold.any():
            raise RuntimeError(f"v14 M2 S0 telemetry mismatch: {run_id}")
    elif scenario == "S4":
        event_rounds = rounds.loc[rounds["event_offset_round"].astype(int) == 0, "round"].astype(int).tolist()
        if event_rounds != [EVENT_ROUND] or rounds.loc[trigger, "round"].astype(int).tolist() != [EVENT_ROUND]:
            raise RuntimeError(f"v14 M2 S4 event mismatch: {run_id}")
    else:
        raise RuntimeError(f"unexpected v14 M2 scenario: {scenario}")
    if method_version == "v14":
        _audit_v14_contract(rounds, scenario, str(job["selected_candidate_id"]), run_id)
    elif method_version == "v11":
        if scenario == "S4" and rounds.loc[hold, "round"].astype(int).tolist() != [EVENT_ROUND]:
            raise RuntimeError(f"v14 M2 v11 S4 quarantine mismatch: {run_id}")
    else:
        raise RuntimeError(f"unexpected v14 M2 method version: {method_version}")
    tta_round, tta_s = _time_to_accuracy(rounds)
    lqa20 = float(derive_lqa20_percent(rounds)) if scenario == "S4" else None
    metrics = {
        "method_version": method_version,
        "scenario_id": scenario,
        "run_id": run_id,
        "last50_validation_accuracy": float(rounds.tail(50)["test_accuracy"].astype(float).mean()),
        "lqa20_percent": lqa20,
        "tta_round": tta_round,
        "algorithm_tta_s": tta_s,
        "trigger_count": int(trigger.sum()),
        "hold_count": int(hold.sum()),
        "formal_test_access": False,
        "test_labels_used": False,
    }
    return metrics, rounds


def evaluate_gate(
    *,
    v11_s0_last50: float,
    v14_s0_last50: float,
    v11_s4_last50: float,
    v14_s4_last50: float,
    v11_s4_lqa20: float,
    v14_s4_lqa20: float,
    v11_s4_tta_s: float | None,
    v14_s4_tta_s: float | None,
) -> dict[str, Any]:
    tta_delta = (
        None
        if v11_s4_tta_s is None
        or v14_s4_tta_s is None
        or float(v11_s4_tta_s) <= 0.0
        else (float(v11_s4_tta_s) - float(v14_s4_tta_s)) / float(v11_s4_tta_s)
    )
    deltas: dict[str, float | None] = {
        "s0_last50_accuracy": 100.0 * (float(v14_s0_last50) - float(v11_s0_last50)),
        "s4_last50_accuracy": 100.0 * (float(v14_s4_last50) - float(v11_s4_last50)),
        "s4_lqa20": float(v14_s4_lqa20) - float(v11_s4_lqa20),
        "s4_algorithm_tta": tta_delta,
    }
    strict = {
        "s0_last50_accuracy": bool(deltas["s0_last50_accuracy"] > DECISION_ATOL),
        "s4_last50_accuracy": bool(deltas["s4_last50_accuracy"] > DECISION_ATOL),
        "s4_lqa20": bool(deltas["s4_lqa20"] > DECISION_ATOL),
        "s4_algorithm_tta": bool(tta_delta is not None and tta_delta > DECISION_ATOL),
    }
    close = {
        "s0_last50_accuracy": bool(
            deltas["s0_last50_accuracy"] >= -GATE_MARGINS["s0_last50_pp"] - DECISION_ATOL
        ),
        "s4_last50_accuracy": bool(
            deltas["s4_last50_accuracy"] >= -GATE_MARGINS["s4_last50_pp"] - DECISION_ATOL
        ),
        "s4_lqa20": bool(deltas["s4_lqa20"] >= -GATE_MARGINS["s4_lqa20_pp"] - DECISION_ATOL),
        "s4_algorithm_tta": bool(
            tta_delta is not None
            and tta_delta >= -GATE_MARGINS["s4_algorithm_tta_relative"] - DECISION_ATOL
        ),
    }
    strict_names = [name for name, passed in strict.items() if passed]
    misses = [name for name, passed in strict.items() if not passed]
    sole_miss_close = len(misses) == 1 and close[misses[0]]
    passed = len(strict_names) == 4 or (len(strict_names) == 3 and sole_miss_close)
    return {
        "deltas": deltas,
        "strict_wins": strict,
        "within_close_margin": close,
        "strict_win_count": len(strict_names),
        "misses": misses,
        "sole_miss_close": bool(sole_miss_close),
        "performance_gate_passed": bool(passed),
    }


def _learning_identity(
    v11_rounds: pd.DataFrame, v14_rounds: pd.DataFrame
) -> dict[str, bool]:
    return {
        column: bool(
            len(v11_rounds) == len(v14_rounds)
            and np.array_equal(
                v11_rounds[column].astype(str).to_numpy(),
                v14_rounds[column].astype(str).to_numpy(),
            )
        )
        for column in LEARNING_IDENTITY_COLUMNS
    }


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot evaluate an incomplete v14 M2 pair")
    rows: list[dict[str, Any]] = []
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for job in manifest["jobs"]:
        metrics, frame = _run_metrics(job)
        rows.append(metrics)
        frames[(str(job["method_version"]), str(job["scenario_id"]))] = frame
    if set(frames) != set(METHOD_ORDER):
        raise RuntimeError("v14 M2 result matrix is incomplete")
    identity_by_scenario = {
        scenario: _learning_identity(frames[("v11", scenario)], frames[("v14", scenario)])
        for scenario in SCENARIOS
    }
    learning_identity_passed = bool(
        all(all(values.values()) for values in identity_by_scenario.values())
    )
    by_key = {(row["method_version"], row["scenario_id"]): row for row in rows}
    gate = evaluate_gate(
        v11_s0_last50=by_key[("v11", "S0")]["last50_validation_accuracy"],
        v14_s0_last50=by_key[("v14", "S0")]["last50_validation_accuracy"],
        v11_s4_last50=by_key[("v11", "S4")]["last50_validation_accuracy"],
        v14_s4_last50=by_key[("v14", "S4")]["last50_validation_accuracy"],
        v11_s4_lqa20=by_key[("v11", "S4")]["lqa20_percent"],
        v14_s4_lqa20=by_key[("v14", "S4")]["lqa20_percent"],
        v11_s4_tta_s=by_key[("v11", "S4")]["algorithm_tta_s"],
        v14_s4_tta_s=by_key[("v14", "S4")]["algorithm_tta_s"],
    )
    overall_passed = bool(gate["performance_gate_passed"] and learning_identity_passed)
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
        "learning_identity_by_scenario": identity_by_scenario,
        "learning_identity_passed": learning_identity_passed,
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
    _assert_immutable_authority(state, manifest)
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
        module = "r2c_baselines.run_v14" if job["method_version"] == "v14" else "r2c_baselines.run"
        with log_path.open("w", encoding="utf-8", newline="\n") as handle:
            process = subprocess.run(
                [sys.executable, "-m", module, "--job", str(job_file)],
                cwd=str(Path(__file__).resolve().parents[1]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        success = (RUN_ROOT / str(resolved["run_id"]) / "_SUCCESS.json").exists()
        audit_log = (
            QUEUE_ROOT
            / "worker_logs"
            / f"{resolved['run_id']}.m2.{job['method_version']}.audit.log"
        )
        audit = _audit_run(job, str(resolved["run_id"]), audit_log) if success else None
        if (
            process.returncode == 0
            and success
            and audit is not None
            and audit.returncode == 0
            and _audit_log_passed(job, str(resolved["run_id"]), audit_log)
        ):
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
            "status": (
                "m2_completed_gate_passed_m3_required"
                if passed
                else "m2_completed_gate_failed_stop"
            ),
            "current_job_id": None,
            "all_runs_completed": True,
            "performance_sealed_until_terminal": False,
            "result_path": str(RESULT_PATH),
            "gate_passed": passed,
            "strict_win_count": int(result["gate"]["strict_win_count"]),
            "learning_identity_passed": bool(result["learning_identity_passed"]),
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
