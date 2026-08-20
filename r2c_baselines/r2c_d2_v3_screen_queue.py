from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEV_SEED, PLOT_ROOT, QUEUE_ROOT, RUN_ROOT
from .logging_io import read_chunked_table
from .r2c_v3 import PROTOCOL_VERSION
from .utils import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    canonical_json,
    config_hash,
    sha256_file,
    utc_now,
)


MANIFEST_PATH = QUEUE_ROOT / "r2c_d2_v3_screen_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d2_v3_screen_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d2_v3_screen_scheduler_events.parquet"
SCREEN_TABLE_PATH = PLOT_ROOT / "r2c_d2_v3_screen_trials.parquet"
SCREEN_CSV_PATH = PLOT_ROOT / "r2c_d2_v3_screen_trials.csv"
SCREEN_SELECTION_PATH = QUEUE_ROOT / "r2c_d2_v3_screen_selection.json"


VARIANTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("F085", {"r2c_v3_fixed_server_alpha": 0.85}),
    ("F075", {"r2c_v3_fixed_server_alpha": 0.75}),
    ("F0625", {"r2c_v3_fixed_server_alpha": 0.625}),
    ("F050", {"r2c_v3_fixed_server_alpha": 0.50}),
    ("GMIN50", {"r2c_v3_guard_rule": "minimax", "r2c_v3_min_server_alpha": 0.50}),
    ("GMIN25", {"r2c_v3_guard_rule": "minimax", "r2c_v3_min_server_alpha": 0.25}),
    ("GMEAN25", {"r2c_v3_guard_rule": "mean_loss", "r2c_v3_min_server_alpha": 0.25}),
    (
        "V5F085",
        {
            "r2c_v3_fixed_server_alpha": 0.85,
            "r2c_v2_anchor_per_fold": 32,
            "r2c_v2_finish_weight": 1.25,
            "r2c_v2_floor_fraction": 0.10,
            "r2c_v2_temperature": 0.20,
        },
    ),
)


def _common_config() -> dict[str, Any]:
    return {
        "lr_mult": 1.5,
        "r2c_delta_clip": 0.6107748032302804,
        "r2c_eval_microbatch": 4,
        "r2c_protocol_version": PROTOCOL_VERSION,
        "r2c_v2_agreement_threshold": 0.8,
        "r2c_v2_anchor_per_fold": 16,
        "r2c_v2_audit_replay": False,
        "r2c_v2_completion_power": 0.0,
        "r2c_v2_finish_scale": 0.08,
        "r2c_v2_finish_weight": 0.75,
        "r2c_v2_floor_fraction": 0.20,
        "r2c_v2_propensity_power": 0.0,
        "r2c_v2_radius_multiplier": 1.0,
        "r2c_v2_scale_floor": 1.0e-4,
        "r2c_v2_scout_steps": 1,
        "r2c_v2_temperature": 0.35,
        "r2c_v2_uncertainty_floor_fraction": 0.35,
        "r2c_v2_value_clip": 4.0,
        "r2c_v2_weight_cap": 2.0,
        "r2c_v3_guard_per_fold": 32,
        "r2c_v3_guard_relative_tolerance": 0.0,
        "r2c_v3_server_alphas": [1.0, 0.75, 0.5, 0.25, 0.0],
    }


def _job(index: int, label: str, overrides: dict[str, Any]) -> dict[str, Any]:
    config = _common_config()
    config.update(overrides)
    job_id = f"A-R2C-D2-S4-V3-SCREEN-{label}-s{DEV_SEED}"
    return {
        "job_id": job_id,
        "base_run_id": job_id,
        "stage": "v3_screen",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": "D2",
        "scenario_id": "S4",
        "rounds": 200,
        "method_config": config,
        "block_id": "A-R2C-D2-V3-SCREEN",
        "seed": DEV_SEED,
        "partition_seed": DEV_SEED,
        "trace_seed": DEV_SEED,
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": None,
        "variant_index": int(index),
        "variant_label": label,
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        "data.py",
        "training.py",
        "r2c.py",
        "r2c_v2.py",
        "r2c_v3.py",
        "run.py",
        Path(__file__).name,
    )
    return {name: sha256_file(package / name) for name in names}


def build_manifest(persist: bool = True, force: bool = False) -> dict[str, Any]:
    if persist and MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    jobs = [_job(index, label, overrides) for index, (label, overrides) in enumerate(VARIANTS)]
    if len(jobs) != 8 or len({job["job_id"] for job in jobs}) != 8:
        raise AssertionError("D2 v3 screen must contain eight unique jobs")
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D2_only_v3_validation_screen",
        "protocol_version": PROTOCOL_VERSION,
        "dev_seed": DEV_SEED,
        "formal_test_access": False,
        "rounds_per_job": 200,
        "selection_objective": "last50_validation_accuracy - 2*recovery_deficit_auc20",
        "eligibility": {
            "recovery_deficit_auc20_lte": 0.002562,
            "last50_validation_accuracy_gte": 0.485,
        },
        "selection_count": 2,
        "implementation_hashes": _implementation_hashes(),
        "jobs": jobs,
    }
    if persist:
        QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d2_v3_screen_manifest_{stamp}.json", manifest)
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


def _actual_run_id(base: str) -> tuple[str, str | None]:
    path = RUN_ROOT / base
    if not path.exists() or (path / "_SUCCESS.json").exists():
        return base, None
    attempt = 2
    while (RUN_ROOT / f"{base}-a{attempt}").exists():
        attempt += 1
    return f"{base}-a{attempt}", base


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


def _resolved(job: dict[str, Any]) -> dict[str, Any]:
    value = dict(job)
    run_id, retry_of = _actual_run_id(str(job["base_run_id"]))
    value.update({"run_id": run_id, "retry_of_run_id": retry_of, "queue_utc": utc_now()})
    return value


def freeze_screen(manifest: dict[str, Any]) -> None:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        return
    rows: list[dict[str, Any]] = []
    thresholds = manifest["eligibility"]
    for job in manifest["jobs"]:
        run_id = str(job["actual_run_id"])
        run_dir = RUN_ROOT / run_id
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        auc = result["recovery"]["recovery_deficit_auc20"]
        if auc is None or not result["recovery"]["recovery_auc20_complete"]:
            raise RuntimeError(f"Incomplete AUC@20 in screen run {run_id}")
        certificates = read_chunked_table(run_dir, "certificate_audit")
        alphas = certificates["server_step_alpha"].astype(float).to_numpy()
        last50 = float(result["last50_accuracy"])
        auc = float(auc)
        rows.append(
            {
                "run_id": run_id,
                "variant_index": int(job["variant_index"]),
                "variant_label": job["variant_label"],
                "config_hash": config_hash(job["method_config"]),
                "parameters_json": canonical_json(job["method_config"]),
                "rounds": int(job["rounds"]),
                "last50_validation_accuracy": last50,
                "recovery_deficit_auc20": auc,
                "algorithm_elapsed_s": float(result["algorithm_elapsed_s"]),
                "screen_score": last50 - 2.0 * auc,
                "eligible": bool(
                    auc <= float(thresholds["recovery_deficit_auc20_lte"])
                    and last50 >= float(thresholds["last50_validation_accuracy_gte"])
                ),
                "mean_server_alpha": float(np.mean(alphas)),
                "min_server_alpha": float(np.min(alphas)),
                "max_server_alpha": float(np.max(alphas)),
                "alpha_p10": float(np.quantile(alphas, 0.10)),
                "alpha_p90": float(np.quantile(alphas, 0.90)),
                "test_labels_used": False,
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["eligible", "screen_score", "algorithm_elapsed_s", "variant_index"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    eligible = frame.loc[frame["eligible"]]
    if len(eligible) >= int(manifest["selection_count"]):
        chosen_indices = eligible.index[: int(manifest["selection_count"])].tolist()
    else:
        chosen_indices = eligible.index.tolist()
        for index in frame.index:
            if index not in chosen_indices:
                chosen_indices.append(int(index))
            if len(chosen_indices) == int(manifest["selection_count"]):
                break
    frame["screen_rank"] = range(1, len(frame) + 1)
    frame["selected_for_gate"] = frame.index.isin(chosen_indices)
    atomic_parquet(SCREEN_TABLE_PATH, frame)
    atomic_csv(SCREEN_CSV_PATH, frame)
    selected = []
    for _, row in frame.loc[frame["selected_for_gate"]].sort_values("screen_rank").iterrows():
        selected.append(
            {
                "variant_index": int(row["variant_index"]),
                "variant_label": str(row["variant_label"]),
                "run_id": str(row["run_id"]),
                "screen_score": float(row["screen_score"]),
                "eligible": bool(row["eligible"]),
                "method_config": json.loads(str(row["parameters_json"])),
            }
        )
    atomic_json(
        SCREEN_SELECTION_PATH,
        {
            "schema_version": "1.0.0",
            "frozen_utc": utc_now(),
            "selection_split": "validation",
            "test_labels_used": False,
            "objective": manifest["selection_objective"],
            "eligibility": manifest["eligibility"],
            "selected": selected,
        },
    )


def worker() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        build_manifest()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    events = _events()
    state["status"] = "running"
    _sync(state, manifest)
    atomic_json(STATE_PATH, state)
    for position, job in enumerate(manifest["jobs"]):
        if job["status"] == "completed":
            continue
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
        if process.returncode == 0 and success:
            job["status"] = "completed"
            _event(events, job, "completed", exit_code=0)
        else:
            job["status"] = "failed"
            job["failure_reason"] = f"exit_code={process.returncode};log={log_path}"
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
    freeze_screen(manifest)
    state.update({"status": "screen_completed", "current_job_id": None})
    _sync(state, manifest)
    atomic_json(STATE_PATH, state)
    return state


def status() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"status": "not_built"}
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        state["jobs"] = [
            {
                key: job.get(key)
                for key in (
                    "job_id",
                    "variant_label",
                    "status",
                    "attempts",
                    "actual_run_id",
                )
            }
            for job in manifest["jobs"]
        ]
    return state


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
