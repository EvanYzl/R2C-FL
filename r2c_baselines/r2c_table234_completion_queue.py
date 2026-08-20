from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATASETS, DEV_SEED, FORMAL_SEED, PLOT_ROOT, PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .data import partition_asset_path, partition_meta_path, prepare_partition
from .logging_io import read_chunked_table
from .metrics import recovery_auc20
from .r2c_d3_v7_phase_e_queue import _audit_run
from .r2c_v7 import PROTOCOL_VERSION
from .traces import prepare_trace, trace_asset_path, trace_meta_path
from .utils import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    config_hash,
    sha256_file,
    sha256_text,
    utc_now,
)


PLAN_ID = "R2C_TABLE234_COMPLETION_20260818_202701"
PLAN_PATH = PROJECT_ROOT / "refine-logs" / "R2C_TABLE234_COMPLETION_PLAN_20260818_202701.md"
SOURCE_V11_MANIFEST_PATH = QUEUE_ROOT / "r2c_v11_four_dataset_matched_manifest.json"
SOURCE_V11_STATE_PATH = QUEUE_ROOT / "r2c_v11_four_dataset_matched_queue_state.json"
TARGET_PATH = QUEUE_ROOT / "frozen_targets.json"
AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / "tests" / "audit_r2c_run.py"

MANIFEST_PATH = QUEUE_ROOT / "r2c_table234_completion_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_table234_completion_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_table234_completion_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_table234_completion_result.json"
RUNS_PATH = PLOT_ROOT / "r2c_table234_completion_runs.parquet"
RUNS_CSV_PATH = PLOT_ROOT / "r2c_table234_completion_runs.csv"
SMOKE_RESULT_PATH = QUEUE_ROOT / "r2c_table234_completion_smoke_result.json"

DATASET_ID = "D2"
ROUNDS = int(DATASETS[DATASET_ID].round_budget)
FULL_SCENARIOS = ("S1", "S2", "S3")
ABLATION_SCENARIOS = ("S3", "S4")
ABLATIONS: tuple[tuple[str, str, str], ...] = (
    ("no_reusable_prefix", "A1-NOPREFIX", "r2c_ablation_no_reusable_prefix"),
    ("no_finishability", "A2-NOFINISH", "r2c_ablation_no_finishability"),
    ("no_drift_quarantine", "A3-NOQUAR", "r2c_ablation_no_drift_quarantine"),
    ("no_valid_crossfit", "A4-NOCROSSFIT", "r2c_ablation_no_valid_crossfit"),
)
MAX_ATTEMPTS = 3
FORMAL_INTERPRETATION = "matched_seed_engineering_reevaluation_not_untouched_confirmation"


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
        "run.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    values["tests/audit_r2c_run.py"] = sha256_file(AUDIT_SCRIPT)
    return values


def _source_lineage() -> dict[str, str]:
    paths = (PLAN_PATH, SOURCE_V11_MANIFEST_PATH, SOURCE_V11_STATE_PATH, TARGET_PATH)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _source_config() -> tuple[dict[str, Any], dict[str, Any]]:
    state = json.loads(SOURCE_V11_STATE_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(SOURCE_V11_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        state.get("status") != "formal_completed_audited"
        or not bool(state.get("all_runs_completed"))
        or int(state.get("completed", -1)) != 8
        or int(state.get("failed", -1)) != 0
    ):
        raise RuntimeError("Completed v11 matched-seed source is not terminal and audited")
    jobs = [job for job in manifest.get("jobs", []) if job.get("dataset_id") == DATASET_ID]
    if len(jobs) != 2 or {job.get("scenario_id") for job in jobs} != {"S0", "S4"}:
        raise RuntimeError("D2 v11 source pair is incomplete")
    if any(job.get("status") != "completed" for job in jobs):
        raise RuntimeError("D2 v11 source pair is not completed")
    configs = [dict(job["method_config"]) for job in jobs]
    if config_hash(configs[0]) != config_hash(configs[1]):
        raise RuntimeError("D2 v11 S0/S4 configuration mismatch")
    config = configs[0]
    if (
        config.get("r2c_protocol_version") != PROTOCOL_VERSION
        or not bool(config.get("r2c_v2_audit_replay"))
        or config.get("r2c_v4_deployment_ema_betas") != [0.95]
        or float(config.get("r2c_v4_primary_deployment_beta", -1.0)) != 0.95
        or float(config.get("r2c_v7_trigger_deployment_beta", -1.0)) != 1.0
    ):
        raise RuntimeError("D2 v11 frozen learning configuration drift")
    lineage = {
        "source_manifest_sha256": sha256_file(SOURCE_V11_MANIFEST_PATH),
        "source_config_hash": config_hash(config),
        "source_run_ids": sorted(str(job["actual_run_id"]) for job in jobs),
        "source_protocol_version": PROTOCOL_VERSION,
    }
    return config, lineage


def _variant_config(base: dict[str, Any], variant: str) -> dict[str, Any]:
    value = dict(base)
    value["r2c_ablation_variant"] = variant
    value["r2c_completion_plan_id"] = PLAN_ID
    if variant != "full":
        matches = [flag for name, _, flag in ABLATIONS if name == variant]
        if len(matches) != 1:
            raise ValueError(f"Unknown completion variant: {variant}")
        value[matches[0]] = True
    enabled = [flag for _, _, flag in ABLATIONS if bool(value.get(flag, False))]
    if len(enabled) > 1:
        raise RuntimeError(f"More than one ablation enabled for {variant}: {enabled}")
    expected = 0 if variant == "full" else 1
    if len(enabled) != expected:
        raise RuntimeError(f"Ablation cardinality mismatch for {variant}")
    return value


def _ensure_assets(seed: int, rounds: int, scenarios: tuple[str, ...]) -> dict[str, str]:
    prepare_partition(DATASET_ID, seed)
    paths = [partition_asset_path(DATASET_ID, seed), partition_meta_path(DATASET_ID, seed)]
    for scenario in scenarios:
        prepare_trace(DATASET_ID, scenario, seed, rounds=rounds)
        paths.extend(
            [
                trace_asset_path(DATASET_ID, scenario, seed, rounds),
                trace_meta_path(DATASET_ID, scenario, seed, rounds),
            ]
        )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _job(
    scenario: str,
    variant: str,
    label: str,
    config: dict[str, Any],
    target_accuracy: float,
) -> dict[str, Any]:
    run_id = f"A-R2C-T234-D2-{scenario}-{label}-s{FORMAL_SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "table234_completion_matched_seed_engineering_reevaluation",
        "mode": "formal",
        "method_id": "R2C-FL",
        "dataset_id": DATASET_ID,
        "scenario_id": scenario,
        "rounds": ROUNDS,
        "method_config": dict(config),
        "block_id": "A-R2C-TABLE234-COMPLETION",
        "seed": int(FORMAL_SEED),
        "partition_seed": int(FORMAL_SEED),
        "trace_seed": int(FORMAL_SEED),
        "evaluation_split": "test",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": float(target_accuracy),
        "variant": variant,
        "variant_label": label,
        "formal_interpretation": FORMAL_INTERPRETATION,
        "test_labels_used_for_selection": False,
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def _job_spec(job: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "job_id",
        "base_run_id",
        "stage",
        "mode",
        "method_id",
        "dataset_id",
        "scenario_id",
        "rounds",
        "method_config",
        "block_id",
        "seed",
        "partition_seed",
        "trace_seed",
        "evaluation_split",
        "full_logging",
        "client_microbatch",
        "target_accuracy",
        "variant",
        "variant_label",
        "formal_interpretation",
        "test_labels_used_for_selection",
    )
    return {key: job[key] for key in keys}


def _frozen_spec(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "scope",
        "plan_id",
        "protocol_version",
        "formal_seed",
        "formal_test_access",
        "engineering_reevaluation",
        "test_labels_used_for_selection",
        "performance_sealed_until_terminal",
        "completion_rule",
        "source_lineage",
        "source_config_lineage",
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
            raise RuntimeError("Refusing to rebuild a started Tables 2--4 completion manifest")
    if int(FORMAL_SEED) != 20260811:
        raise RuntimeError("The matched formal seed must remain 20260811")
    base, source_config_lineage = _source_config()
    targets = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    target = float(targets[DATASET_ID])
    jobs: list[dict[str, Any]] = []
    full = _variant_config(base, "full")
    for scenario in FULL_SCENARIOS:
        jobs.append(_job(scenario, "full", "FULL", full, target))
    for variant, label, _ in ABLATIONS:
        config = _variant_config(base, variant)
        for scenario in ABLATION_SCENARIOS:
            jobs.append(_job(scenario, variant, label, config, target))
    if len(jobs) != 11 or len({job["job_id"] for job in jobs}) != 11:
        raise AssertionError("Tables 2--4 completion manifest must contain exactly 11 unique jobs")
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D2_tables_2_3_4_completion",
        "plan_id": PLAN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "formal_seed": int(FORMAL_SEED),
        "formal_test_access": True,
        "engineering_reevaluation": True,
        "test_labels_used_for_selection": False,
        "performance_sealed_until_terminal": True,
        "completion_rule": "all 11 jobs completed and audited; no metric-dependent stopping",
        "source_lineage": _source_lineage(),
        "source_config_lineage": source_config_lineage,
        "asset_hashes": _ensure_assets(
            FORMAL_SEED,
            ROUNDS,
            tuple(sorted(set(FULL_SCENARIOS + ABLATION_SCENARIOS))),
        ),
        "implementation_hashes": _implementation_hashes(),
        "job_order": [job["job_id"] for job in jobs],
        "max_attempts": MAX_ATTEMPTS,
        "jobs": jobs,
    }
    manifest["frozen_spec_hash"] = config_hash(_frozen_spec(manifest))
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        immutable_path = QUEUE_ROOT / f"r2c_table234_completion_manifest_{stamp}.json"
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
                "total": 11,
                "all_runs_completed": False,
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
        manifest.get("scope") != "D2_tables_2_3_4_completion"
        or manifest.get("plan_id") != PLAN_ID
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or int(manifest.get("formal_seed", -1)) != 20260811
        or not bool(manifest.get("formal_test_access"))
        or bool(manifest.get("test_labels_used_for_selection"))
        or not bool(manifest.get("performance_sealed_until_terminal"))
    ):
        raise RuntimeError("Tables 2--4 completion manifest scope drift")
    if manifest.get("source_lineage") != _source_lineage():
        raise RuntimeError("Tables 2--4 source lineage changed after freeze")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("Tables 2--4 implementation changed after manifest freeze")
    if manifest.get("frozen_spec_hash") != config_hash(_frozen_spec(manifest)):
        raise RuntimeError("Tables 2--4 frozen job specification drift")
    jobs = list(manifest.get("jobs", []))
    if len(jobs) != 11 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("Tables 2--4 job matrix/order drift")
    for job in jobs:
        if (
            job.get("dataset_id") != DATASET_ID
            or job.get("scenario_id") not in {"S1", "S2", "S3", "S4"}
            or int(job.get("rounds", -1)) != ROUNDS
            or job.get("evaluation_split") != "test"
            or any(int(job[key]) != 20260811 for key in ("seed", "partition_seed", "trace_seed"))
            or not bool(job.get("full_logging"))
        ):
            raise RuntimeError(f"Protocol drift in {job['job_id']}")


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
            audit_log = QUEUE_ROOT / "worker_logs" / f"{path.name}.reconcile.audit.log"
            audit = _audit_run(path.name, audit_log)
            if audit.returncode == 0:
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


def _run_metrics(job: dict[str, Any]) -> dict[str, Any]:
    run_id = str(job["actual_run_id"])
    run_dir = RUN_ROOT / run_id
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    run_manifest = pd.read_parquet(run_dir / "run_manifest.parquet").iloc[0]
    rounds = read_chunked_table(run_dir, "round_metrics").sort_values("round")
    certificates = read_chunked_table(run_dir, "certificate_audit").sort_values("round")
    if (
        len(rounds) != ROUNDS
        or len(certificates) != ROUNDS
        or str(run_manifest["source_kind"]) != "REPRODUCED"
        or str(run_manifest["status"]) != "completed"
    ):
        raise RuntimeError(f"Incomplete formal output for {run_id}")
    recovery = result["recovery"]
    event = rounds.loc[rounds["event_offset_round"].astype(int) == 0]
    if len(event) != 1:
        raise RuntimeError(f"Missing unique event round for {run_id}")
    event_round = int(event.iloc[0]["round"])
    direct = recovery_auc20(rounds["round"], rounds["test_accuracy"], event_round)
    if (
        not bool(direct["recovery_auc20_complete"])
        or not bool(recovery["recovery_auc20_complete"])
        or recovery["recovery_deficit_auc20"] is None
        or abs(float(direct["recovery_deficit_auc20"]) - float(recovery["recovery_deficit_auc20"])) > 1.0e-12
    ):
        raise RuntimeError(f"Strict AUC@20 mismatch for {run_id}")
    used = float(certificates["candidate_compute_used_s"].astype(float).sum())
    wasted = float(certificates["candidate_compute_wasted_s"].astype(float).sum())
    if used <= 0.0 or not 0.0 <= wasted <= used + 1.0e-9:
        raise RuntimeError(f"Invalid aggregate wasted compute for {run_id}")
    return {
        "run_id": run_id,
        "dataset_id": DATASET_ID,
        "scenario_id": job["scenario_id"],
        "variant": job["variant"],
        "variant_label": job["variant_label"],
        "round_budget": ROUNDS,
        "event_round": event_round,
        "seed": int(FORMAL_SEED),
        "last50_test_accuracy": float(result["last50_accuracy"]),
        "recovery_deficit_auc20": float(recovery["recovery_deficit_auc20"]),
        "recovery_auc20_complete": True,
        "final_participation_jfi": float(rounds.iloc[-1]["participation_jfi"]),
        "certified_fraction": float(certificates["certified"].astype(bool).mean()),
        "candidate_compute_used_s": used,
        "candidate_compute_wasted_s": wasted,
        "candidate_compute_wasted_fraction": wasted / used,
        "algorithm_elapsed_s": float(result["algorithm_elapsed_s"]),
        "source_kind": "REPRODUCED",
        "protocol_version": PROTOCOL_VERSION,
        "formal_interpretation": FORMAL_INTERPRETATION,
        "test_labels_used_for_selection": False,
    }


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot freeze an incomplete Tables 2--4 matrix")
    rows = [_run_metrics(job) for job in manifest["jobs"]]
    frame = pd.DataFrame(rows)
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "status": "completed_audited_pending_latex_finalize",
        "completed_runs": len(frame),
        "formal_seed": int(FORMAL_SEED),
        "source_kind": "REPRODUCED",
        "performance_unsealed_after_terminal": True,
        "run_ids": frame["run_id"].tolist(),
        "frozen_spec_hash": manifest["frozen_spec_hash"],
        "runs_sha256": sha256_file(RUNS_PATH),
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
    freeze_result(manifest)
    state.update(
        {
            "status": "completed_audited_pending_latex_finalize",
            "current_job_id": None,
            "all_runs_completed": True,
            "performance_sealed_until_terminal": False,
            "result_path": str(RESULT_PATH),
        }
    )
    _sync(state, manifest)
    atomic_json(STATE_PATH, state)
    return state


def smoke() -> dict[str, Any]:
    base, _ = _source_config()
    _ensure_assets(DEV_SEED, 2, ("S3",))
    rows: list[dict[str, Any]] = []
    for variant, label, _ in ABLATIONS:
        config = _variant_config(base, variant)
        base_id = f"R2C-T234-SMOKE-D2-S3-{label}-s{DEV_SEED}"
        run_id, retry_of = _actual_run_id(base_id)
        job = {
            "run_id": run_id,
            "retry_of_run_id": retry_of,
            "stage": "table234_completion_ablation_smoke",
            "mode": "sanity",
            "method_id": "R2C-FL",
            "dataset_id": DATASET_ID,
            "scenario_id": "S3",
            "rounds": 2,
            "method_config": config,
            "block_id": "R2C-T234-SMOKE",
            "seed": int(DEV_SEED),
            "partition_seed": int(DEV_SEED),
            "trace_seed": int(DEV_SEED),
            "evaluation_split": "validation",
            "full_logging": True,
            "client_microbatch": 1,
            "target_accuracy": 0.0,
            "variant": variant,
            "variant_label": label,
            "test_labels_used_for_selection": False,
        }
        job_path = QUEUE_ROOT / "active_jobs" / f"{run_id}.json"
        atomic_json(job_path, job)
        log_path = QUEUE_ROOT / "worker_logs" / f"{run_id}.log"
        with log_path.open("w", encoding="utf-8", newline="\n") as handle:
            process = subprocess.run(
                [sys.executable, "-m", "r2c_baselines.run", "--job", str(job_path)],
                cwd=str(Path(__file__).resolve().parents[1]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        success = (RUN_ROOT / run_id / "_SUCCESS.json").exists()
        audit_log = QUEUE_ROOT / "worker_logs" / f"{run_id}.audit.log"
        audit = _audit_run(run_id, audit_log) if success else None
        audit_code = None if audit is None else audit.returncode
        passed = bool(process.returncode == 0 and success and audit_code == 0)
        rows.append(
            {
                "variant": variant,
                "run_id": run_id,
                "train_exit": process.returncode,
                "audit_exit": audit_code,
                "passed": passed,
                "log_path": str(log_path),
                "audit_log_path": str(audit_log),
            }
        )
        if not passed:
            payload = {"status": "failed", "completed_utc": utc_now(), "rows": rows}
            atomic_json(SMOKE_RESULT_PATH, payload)
            return payload
    payload = {"status": "passed", "completed_utc": utc_now(), "rows": rows}
    atomic_json(SMOKE_RESULT_PATH, payload)
    return payload


def status() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"status": "not_built"}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--force", action="store_true")
    sub.add_parser("smoke")
    sub.add_parser("worker")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "build":
        value = build_manifest(force=args.force)
    elif args.command == "smoke":
        value = smoke()
    elif args.command == "worker":
        value = worker()
    else:
        value = status()
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

