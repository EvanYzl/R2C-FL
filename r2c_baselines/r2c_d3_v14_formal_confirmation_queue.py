from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import r2c_d3_v14_development_pair_queue as development
from .config import PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .data import partition_asset_path, partition_meta_path, prepare_partition
from .logging_io import read_chunked_table
from .r2c_post_event_lower_quartile_accuracy20 import derive_lqa20_percent
from .r2c_v7 import PROTOCOL_VERSION as V11_PROTOCOL_VERSION
from .r2c_v14 import PROTOCOL_VERSION
from .traces import prepare_trace, trace_asset_path, trace_meta_path
from .utils import atomic_csv, atomic_json, atomic_parquet, config_hash, sha256_file, utc_now


PLAN_ID = development.PLAN_ID
PLAN_PATH = development.PLAN_PATH
DEVELOPMENT_MANIFEST_PATH = development.MANIFEST_PATH
DEVELOPMENT_STATE_PATH = development.STATE_PATH
DEVELOPMENT_RESULT_PATH = development.RESULT_PATH
DEVELOPMENT_RUNS_PATH = development.RUNS_PATH
DEVELOPMENT_RUNS_CSV_PATH = development.RUNS_CSV_PATH

MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v14_formal_confirmation_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v14_formal_confirmation_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v14_formal_confirmation_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v14_formal_confirmation_result.json"
RUNS_PATH = QUEUE_ROOT / "r2c_d3_v14_formal_confirmation_runs.parquet"
RUNS_CSV_PATH = QUEUE_ROOT / "r2c_d3_v14_formal_confirmation_runs.csv"

DATASET_ID = "D3"
METHOD_ORDER = development.METHOD_ORDER
SCENARIOS = development.SCENARIOS
SEED = 20260812
ROUNDS = 1000
EVENT_ROUND = 500
TARGET_ACCURACY = development.TARGET_ACCURACY
MAX_ATTEMPTS = 3
GATE_MARGINS = dict(development.GATE_MARGINS)
LEARNING_IDENTITY_COLUMNS = development.LEARNING_IDENTITY_COLUMNS


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
        "r2c_d3_v14_development_pair_queue.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    tests = Path(__file__).resolve().parents[1] / "tests"
    for name in (
        "audit_r2c_run.py",
        "audit_r2c_v14_run.py",
        "audit_r2c_v14_run_erratum.py",
        "test_r2c_v14.py",
        "test_r2c_d3_v14_validation_screen_erratum_queue.py",
        "test_r2c_d3_v14_development_pair_queue.py",
        "test_r2c_d3_v14_formal_confirmation_queue.py",
    ):
        values[f"tests/{name}"] = sha256_file(tests / name)
    return values


def _development_context() -> dict[str, Any]:
    for path in (DEVELOPMENT_MANIFEST_PATH, DEVELOPMENT_STATE_PATH):
        if not path.exists():
            raise RuntimeError("v14 M2 is not terminal with a passed gate")
    if not DEVELOPMENT_RESULT_PATH.exists():
        raise RuntimeError("v14 M2 is not terminal with a passed gate")
    manifest = json.loads(DEVELOPMENT_MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(DEVELOPMENT_STATE_PATH.read_text(encoding="utf-8"))
    result = json.loads(DEVELOPMENT_RESULT_PATH.read_text(encoding="utf-8"))
    development._assert_frozen_manifest(manifest)
    development._assert_immutable_authority(state, manifest)
    if (
        state.get("status") != "m2_completed_gate_passed_m3_required"
        or int(state.get("completed", -1)) != 4
        or int(state.get("failed", -1)) != 0
        or not bool(state.get("all_runs_completed"))
        or bool(state.get("performance_sealed_until_terminal"))
        or not bool(state.get("gate_passed"))
        or not bool(state.get("learning_identity_passed"))
        or result.get("status") != "m2_gate_passed"
        or not bool(result.get("overall_gate_passed"))
        or not bool(result.get("learning_identity_passed"))
        or int(result.get("gate", {}).get("strict_win_count", -1)) < 3
        or result.get("selection_split") != "validation"
        or bool(result.get("formal_test_access"))
        or bool(result.get("other_dataset_access"))
        or result.get("frozen_spec_hash") != manifest.get("frozen_spec_hash")
    ):
        raise RuntimeError("v14 M2 is not terminal with a passed gate")
    candidate_id = str(manifest.get("selected_candidate", {}).get("candidate_id", ""))
    if candidate_id not in development.screen.CANDIDATES:
        raise RuntimeError("v14 M2 selected candidate drift")
    immutable_path = Path(str(state.get("immutable_manifest_path", ""))).resolve()
    if (
        not immutable_path.is_file()
        or sha256_file(immutable_path).lower()
        != str(state.get("immutable_manifest_sha256", "")).lower()
    ):
        raise RuntimeError("v14 M2 immutable lineage mismatch")
    audit_logs: list[Path] = []
    run_dirs: list[Path] = []
    for job in manifest["jobs"]:
        if job.get("status") != "completed" or not job.get("actual_run_id"):
            raise RuntimeError("v14 M2 contains an incomplete source job")
        run_id = str(job["actual_run_id"])
        run_dir = RUN_ROOT / run_id
        candidates = (
            QUEUE_ROOT / "worker_logs" / f"{run_id}.m2.reconcile.{job['method_version']}.audit.log",
            QUEUE_ROOT / "worker_logs" / f"{run_id}.m2.{job['method_version']}.audit.log",
        )
        audit_log = next(
            (
                path
                for path in candidates
                if path.is_file() and development._audit_log_passed(job, run_id, path)
            ),
            None,
        )
        if audit_log is None or not (run_dir / "_SUCCESS.json").exists():
            raise RuntimeError("v14 M2 source audit/success evidence is missing")
        audit_logs.append(audit_log)
        run_dirs.append(run_dir)
    return {
        "manifest": manifest,
        "state": state,
        "result": result,
        "immutable_path": immutable_path,
        "candidate_id": candidate_id,
        "audit_logs": audit_logs,
        "run_dirs": run_dirs,
    }


def _source_lineage(context: dict[str, Any]) -> dict[str, str]:
    paths: list[Path] = [
        PLAN_PATH,
        Path(context["immutable_path"]),
        DEVELOPMENT_MANIFEST_PATH,
        DEVELOPMENT_STATE_PATH,
        DEVELOPMENT_RESULT_PATH,
        DEVELOPMENT_RUNS_PATH,
        DEVELOPMENT_RUNS_CSV_PATH,
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
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _v11_config() -> dict[str, Any]:
    config = development._v11_config()
    config["r2c_v2_audit_replay"] = True
    return config


def _v14_config(candidate_id: str) -> dict[str, Any]:
    config = development._v14_config(candidate_id)
    config["r2c_v2_audit_replay"] = True
    return config


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
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in _asset_paths()}


def _current_asset_hashes() -> dict[str, str]:
    for path in _asset_paths():
        if not path.exists():
            raise FileNotFoundError(path)
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in _asset_paths()}


def _job(method_version: str, scenario: str, context: dict[str, Any]) -> dict[str, Any]:
    if (method_version, scenario) not in METHOD_ORDER:
        raise ValueError("job is outside immutable v14 M3 method/scenario order")
    candidate_id = str(context["candidate_id"])
    method_label = "V11-B095" if method_version == "v11" else f"V14-{candidate_id}"
    run_id = f"A-R2C-D3-{scenario}-V14M3-{method_label}-s{SEED}"
    job: dict[str, Any] = {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v14_untouched_formal_confirmation",
        "mode": "formal",
        "method_id": "R2C-FL",
        "method_version": method_version,
        "dataset_id": DATASET_ID,
        "scenario_id": scenario,
        "rounds": ROUNDS,
        "method_config": _v11_config() if method_version == "v11" else _v14_config(candidate_id),
        "block_id": "A-R2C-D3-V14-M3-UNTOUCHED-FORMAL",
        "seed": SEED,
        "partition_seed": SEED,
        "trace_seed": SEED,
        "evaluation_split": "test",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "selected_candidate_id": candidate_id,
        "source_development_frozen_spec_hash": str(context["manifest"]["frozen_spec_hash"]),
        "formal_test_access": True,
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
                "fast_beta": development.FAST_BETA,
                "stable_beta": development.candidate_stable_beta(candidate_id),
                "warmup_rounds": development.WARMUP_ROUNDS,
                "recovery_rounds": development.candidate_recovery_rounds(candidate_id),
                "deployment_state_count": 2,
            }
        )
    return job


def _job_spec(job: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in job.items()
        if key not in {"status", "attempts", "actual_run_id", "failure_reason"}
    }


def _frozen_spec(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "scope",
        "plan_id",
        "protocol_version",
        "comparator_protocol_version",
        "evaluation_split",
        "seed",
        "rounds_per_job",
        "event_round",
        "target_accuracy",
        "formal_test_access",
        "other_dataset_access",
        "test_labels_used_for_selection",
        "candidate_locked_before_formal",
        "performance_sealed_until_terminal",
        "selected_candidate",
        "gate_metrics",
        "gate_margins",
        "gate_rule",
        "learning_identity_columns",
        "learning_identity_rule",
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
    # Formal assets and files are unreachable until the independent validation
    # pair has passed its prospectively frozen performance and identity gates.
    context = _development_context()
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
            raise RuntimeError("Refusing to rebuild a started v14 M3 manifest")
    jobs = [_job(method, scenario, context) for method, scenario in METHOD_ORDER]
    if [(job["method_version"], job["scenario_id"]) for job in jobs] != list(METHOD_ORDER):
        raise AssertionError("v14 M3 order must be v11 S0/S4 then v14 S0/S4")
    if development._learning_config(_v11_config()) != development._learning_config(
        _v14_config(context["candidate_id"])
    ):
        raise RuntimeError("v14 M3 learning configurations are not exactly matched")
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_test_seed20260812_v11_v14_untouched_confirmation",
        "plan_id": PLAN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "comparator_protocol_version": V11_PROTOCOL_VERSION,
        "evaluation_split": "test",
        "seed": SEED,
        "rounds_per_job": ROUNDS,
        "event_round": EVENT_ROUND,
        "target_accuracy": TARGET_ACCURACY,
        "formal_test_access": True,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "candidate_locked_before_formal": True,
        "performance_sealed_until_terminal": True,
        "selected_candidate": {
            "candidate_id": str(context["candidate_id"]),
            "source_development_frozen_spec_hash": str(context["manifest"]["frozen_spec_hash"]),
        },
        "gate_metrics": (
            "S0 last50 accuracy; S4 last50 accuracy; S4 post-event LQA20; "
            "S4 algorithm TTA"
        ),
        "gate_margins": GATE_MARGINS,
        "gate_rule": (
            "pass iff four strict wins, or exactly three strict wins and the sole miss "
            "is close: last50 no worse than 0.15 pp, LQA20 no worse than 0.50 pp, "
            "or TTA no worse than 5 percent relative; equality is not a strict win; "
            "exact global learning-model hash identity in both S0 and S4 is mandatory"
        ),
        "learning_identity_columns": list(LEARNING_IDENTITY_COLUMNS),
        "learning_identity_rule": "exact rowwise identity in both S0 and S4",
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
        immutable_path = QUEUE_ROOT / f"r2c_d3_v14_formal_confirmation_manifest_{stamp}.json"
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
                "formal_test_access": True,
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


def _assert_immutable_authority(state: dict[str, Any], manifest: dict[str, Any]) -> None:
    immutable_path = Path(str(state.get("immutable_manifest_path", "")))
    if (
        not immutable_path.is_file()
        or sha256_file(immutable_path) != state.get("immutable_manifest_sha256")
    ):
        raise RuntimeError("v14 M3 immutable manifest hash mismatch")
    immutable = json.loads(immutable_path.read_text(encoding="utf-8"))
    if _frozen_spec(immutable) != _frozen_spec(manifest):
        raise RuntimeError("v14 M3 active/immutable frozen specifications diverged")


def _assert_frozen_manifest(manifest: dict[str, Any]) -> None:
    context = _development_context()
    expected_selected = {
        "candidate_id": str(context["candidate_id"]),
        "source_development_frozen_spec_hash": str(context["manifest"]["frozen_spec_hash"]),
    }
    if (
        manifest.get("scope") != "D3_test_seed20260812_v11_v14_untouched_confirmation"
        or manifest.get("plan_id") != PLAN_ID
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("comparator_protocol_version") != V11_PROTOCOL_VERSION
        or manifest.get("evaluation_split") != "test"
        or int(manifest.get("seed", -1)) != SEED
        or int(manifest.get("rounds_per_job", -1)) != ROUNDS
        or int(manifest.get("event_round", -1)) != EVENT_ROUND
        or not bool(manifest.get("formal_test_access"))
        or bool(manifest.get("other_dataset_access"))
        or bool(manifest.get("test_labels_used_for_selection"))
        or not bool(manifest.get("candidate_locked_before_formal"))
        or not bool(manifest.get("performance_sealed_until_terminal"))
        or manifest.get("selected_candidate") != expected_selected
        or manifest.get("gate_margins") != GATE_MARGINS
        or manifest.get("learning_identity_columns") != list(LEARNING_IDENTITY_COLUMNS)
    ):
        raise RuntimeError("v14 M3 manifest scope drift")
    if manifest.get("source_lineage") != _source_lineage(context):
        raise RuntimeError("v14 M3 source lineage drift")
    if manifest.get("asset_hashes") != _current_asset_hashes():
        raise RuntimeError("v14 M3 untouched formal asset drift")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("v14 M3 implementation drift after freeze")
    if manifest.get("frozen_spec_hash") != config_hash(_frozen_spec(manifest)):
        raise RuntimeError("v14 M3 frozen specification drift")
    jobs = list(manifest.get("jobs", []))
    if len(jobs) != 4 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("v14 M3 job order drift")
    if development._learning_config(_v11_config()) != development._learning_config(
        _v14_config(context["candidate_id"])
    ):
        raise RuntimeError("v14 M3 learning configuration drift")
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
            or job.get("evaluation_split") != "test"
            or job.get("mode") != "formal"
            or int(job.get("rounds", -1)) != ROUNDS
            or any(int(job.get(key, -1)) != SEED for key in ("seed", "partition_seed", "trace_seed"))
            or job.get("method_config") != expected_config
            or job.get("selected_candidate_id") != context["candidate_id"]
            or not bool(job.get("formal_test_access"))
            or bool(job.get("other_dataset_access"))
            or bool(job.get("test_labels_used_for_selection"))
        ):
            raise RuntimeError(f"v14 M3 job drift: {job.get('job_id')}")
        if method_version == "v14" and (
            float(job.get("fast_beta", -1.0)) != development.FAST_BETA
            or float(job.get("stable_beta", -1.0))
            != development.candidate_stable_beta(str(context["candidate_id"]))
            or int(job.get("warmup_rounds", -1)) != development.WARMUP_ROUNDS
            or int(job.get("recovery_rounds", -1))
            != development.candidate_recovery_rounds(str(context["candidate_id"]))
            or int(job.get("deployment_state_count", -1)) != 2
        ):
            raise RuntimeError(f"v14 M3 CMTR job metadata drift: {job.get('job_id')}")


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
                / f"{path.name}.m3.reconcile.{job['method_version']}.audit.log"
            )
            audit = development._audit_run(job, path.name, audit_log)
            if (
                audit.returncode == 0
                and development._audit_log_passed(job, path.name, audit_log)
            ):
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


def _run_metrics(job: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    run_id = str(job["actual_run_id"])
    run_dir = RUN_ROOT / run_id
    actual_job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    run_manifest = pd.read_parquet(run_dir / "run_manifest.parquet").iloc[0]
    rounds = read_chunked_table(run_dir, "round_metrics").sort_values("round")
    if (
        len(rounds) != ROUNDS
        or rounds["round"].astype(int).tolist() != list(range(1, ROUNDS + 1))
        or str(run_manifest["source_kind"]) != "REPRODUCED"
        or actual_job.get("evaluation_split") != "test"
        or actual_job.get("mode") != "formal"
        or actual_job.get("method_config") != job.get("method_config")
        or bool(actual_job.get("test_labels_used_for_selection"))
    ):
        raise RuntimeError(f"v14 M3 run contract mismatch: {run_id}")
    method_version = str(job["method_version"])
    scenario = str(job["scenario_id"])
    trigger = rounds["telemetry_shift_trigger"].astype(bool)
    hold = rounds["deployment_quarantine_applied"].astype(bool)
    if scenario == "S0":
        if trigger.any() or hold.any():
            raise RuntimeError(f"v14 M3 S0 telemetry mismatch: {run_id}")
    elif scenario == "S4":
        event_rounds = rounds.loc[rounds["event_offset_round"].astype(int) == 0, "round"].astype(int).tolist()
        if event_rounds != [EVENT_ROUND] or rounds.loc[trigger, "round"].astype(int).tolist() != [EVENT_ROUND]:
            raise RuntimeError(f"v14 M3 S4 event mismatch: {run_id}")
    else:
        raise RuntimeError(f"unexpected v14 M3 scenario: {scenario}")
    if method_version == "v14":
        development._audit_v14_contract(
            rounds, scenario, str(job["selected_candidate_id"]), run_id
        )
    elif method_version == "v11":
        if scenario == "S4" and rounds.loc[hold, "round"].astype(int).tolist() != [EVENT_ROUND]:
            raise RuntimeError(f"v14 M3 v11 S4 quarantine mismatch: {run_id}")
    else:
        raise RuntimeError(f"unexpected v14 M3 method version: {method_version}")
    tta_round, tta_s = development._time_to_accuracy(rounds)
    lqa20 = float(derive_lqa20_percent(rounds)) if scenario == "S4" else None
    metrics = {
        "method_version": method_version,
        "scenario_id": scenario,
        "run_id": run_id,
        "last50_test_accuracy": float(rounds.tail(50)["test_accuracy"].astype(float).mean()),
        "lqa20_percent": lqa20,
        "tta_round": tta_round,
        "algorithm_tta_s": tta_s,
        "trigger_count": int(trigger.sum()),
        "hold_count": int(hold.sum()),
        "test_labels_used_for_selection": False,
    }
    return metrics, rounds


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot evaluate an incomplete v14 M3 confirmation")
    rows: list[dict[str, Any]] = []
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for job in manifest["jobs"]:
        metrics, frame = _run_metrics(job)
        rows.append(metrics)
        frames[(str(job["method_version"]), str(job["scenario_id"]))] = frame
    if set(frames) != set(METHOD_ORDER):
        raise RuntimeError("v14 M3 result matrix is incomplete")
    identity_by_scenario = {
        scenario: development._learning_identity(
            frames[("v11", scenario)], frames[("v14", scenario)]
        )
        for scenario in SCENARIOS
    }
    learning_identity_passed = bool(
        all(all(values.values()) for values in identity_by_scenario.values())
    )
    by_key = {(row["method_version"], row["scenario_id"]): row for row in rows}
    gate = development.evaluate_gate(
        v11_s0_last50=by_key[("v11", "S0")]["last50_test_accuracy"],
        v14_s0_last50=by_key[("v14", "S0")]["last50_test_accuracy"],
        v11_s4_last50=by_key[("v11", "S4")]["last50_test_accuracy"],
        v14_s4_last50=by_key[("v14", "S4")]["last50_test_accuracy"],
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
        "status": "m3_gate_passed" if overall_passed else "m3_gate_failed",
        "evaluation_split": "test",
        "seed": SEED,
        "formal_test_access": True,
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
        resolved = development._resolved(job)
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
            / f"{resolved['run_id']}.m3.{job['method_version']}.audit.log"
        )
        audit = development._audit_run(job, str(resolved["run_id"]), audit_log) if success else None
        if (
            process.returncode == 0
            and success
            and audit is not None
            and audit.returncode == 0
            and development._audit_log_passed(job, str(resolved["run_id"]), audit_log)
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
                "m3_completed_gate_passed_m4_unlocked"
                if passed
                else "m3_completed_gate_failed_stop"
            ),
            "current_job_id": None,
            "all_runs_completed": True,
            "performance_sealed_until_terminal": False,
            "result_path": str(RESULT_PATH),
            "gate_passed": passed,
            "strict_win_count": int(result["gate"]["strict_win_count"]),
            "learning_identity_passed": bool(result["learning_identity_passed"]),
            "other_dataset_access": passed,
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
