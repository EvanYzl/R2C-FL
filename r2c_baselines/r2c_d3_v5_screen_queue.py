from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DEV_SEED, PLOT_ROOT, PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .logging_io import read_chunked_table
from .metrics import recovery_auc20
from .r2c_v4 import PROTOCOL_VERSION
from .utils import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    config_hash,
    sha256_file,
    utc_now,
)


ALPHAS = (0.75, 0.875, 1.0)
BETAS = (0.0, 0.80, 0.90, 0.95)
ROUNDS = 600
TARGET_ACCURACY = 0.7986707616707616
EXPECTED_SOURCE_SHA256 = "E7839A280825190C5B59ACF3A174C30B2901F661207B912DCC48473F131E18A6"

SOURCE_MANIFEST_PATH = (
    QUEUE_ROOT / "r2c_v4_table1_matched_manifest_20260816T035638.500920Z.json"
)
PLAN_PATH = PROJECT_ROOT / "refine-logs" / "D3_V5_OPTIMIZATION_PLAN_20260816_221904.md"
MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v5_screen_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v5_screen_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v5_screen_scheduler_events.parquet"
SELECTION_PATH = QUEUE_ROOT / "r2c_d3_v5_screen_selection.json"
TABLE_PATH = PLOT_ROOT / "r2c_d3_v5_screen_candidates.parquet"
CSV_PATH = PLOT_ROOT / "r2c_d3_v5_screen_candidates.csv"


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
        "run.py",
        Path(__file__).name,
    )
    return {name: sha256_file(package / name) for name in names}


def _source_hashes() -> dict[str, str]:
    return {
        str(SOURCE_MANIFEST_PATH.relative_to(PROJECT_ROOT)): sha256_file(SOURCE_MANIFEST_PATH),
        str(PLAN_PATH.relative_to(PROJECT_ROOT)): sha256_file(PLAN_PATH),
    }


def _base_config() -> dict[str, Any]:
    if sha256_file(SOURCE_MANIFEST_PATH) != EXPECTED_SOURCE_SHA256.lower():
        raise RuntimeError("Frozen matched-seed v4 source manifest hash drift")
    source = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    matches = [
        job
        for job in source.get("jobs", [])
        if job.get("dataset_id") == "D3" and job.get("scenario_id") == "S4"
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one frozen D3/S4 v4 source job")
    config = dict(matches[0]["method_config"])
    if config.get("r2c_protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Frozen D3 source protocol drift")
    if float(config.get("r2c_v3_fixed_server_alpha")) != 0.75:
        raise RuntimeError("Frozen D3 source alpha drift")
    if config.get("r2c_v4_deployment_ema_betas") != [0.95]:
        raise RuntimeError("Frozen D3 source beta drift")
    config.update(
        {
            "r2c_v2_audit_replay": False,
            "r2c_v4_deployment_ema_betas": list(BETAS),
            "r2c_v4_primary_deployment_beta": BETAS[0],
        }
    )
    return config


def _job(alpha: float, base_config: dict[str, Any]) -> dict[str, Any]:
    label = f"A{int(round(alpha * 1000)):04d}"
    run_id = f"A-R2C-D3-S4-V5SCREEN-{label}-s{DEV_SEED}"
    config = dict(base_config)
    config["r2c_v3_fixed_server_alpha"] = float(alpha)
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v5_validation_attenuation_screen",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": "D3",
        "scenario_id": "S4",
        "rounds": ROUNDS,
        "method_config": config,
        "block_id": "A-R2C-D3-V5-SCREEN",
        "seed": DEV_SEED,
        "partition_seed": DEV_SEED,
        "trace_seed": DEV_SEED,
        "evaluation_split": "validation",
        "full_logging": False,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "variant_label": label,
        "test_labels_used_for_selection": False,
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def build_manifest(persist: bool = True, force: bool = False) -> dict[str, Any]:
    if persist and MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if force and MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if any(job.get("status") != "pending" or int(job.get("attempts", 0)) for job in existing["jobs"]):
            raise RuntimeError("Refusing to rebuild a started D3 v5 screen manifest")
    if int(DEV_SEED) != 20260810:
        raise RuntimeError(f"Expected development seed 20260810, found {DEV_SEED}")
    base_config = _base_config()
    jobs = [_job(alpha, base_config) for alpha in ALPHAS]
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_only_v5_phase_a_validation_attenuation_screen",
        "protocol_version": PROTOCOL_VERSION,
        "formal_test_access": False,
        "test_labels_used_for_selection": False,
        "dev_seed": DEV_SEED,
        "rounds_per_job": ROUNDS,
        "candidate_alphas": list(ALPHAS),
        "candidate_betas": list(BETAS),
        "target_accuracy": TARGET_ACCURACY,
        "job_order": [job["job_id"] for job in jobs],
        "selection_rule": (
            "among complete target-reaching candidates, retain two with minimum ordinal "
            "rank sum over Last50(desc), AUC20(asc), target-hit-round(asc); frozen tie breakers"
        ),
        "implementation_hashes": _implementation_hashes(),
        "source_hashes": _source_hashes(),
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d3_v5_screen_manifest_{stamp}.json", manifest)
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
            return path.name
    return None


def _reconcile_successes(manifest: dict[str, Any], events: list[dict[str, Any]]) -> int:
    changed = 0
    for job in manifest["jobs"]:
        run_id = _successful_run_id(job)
        if run_id is not None and (job.get("status") != "completed" or job.get("actual_run_id") != run_id):
            job.update({"status": "completed", "actual_run_id": run_id, "failure_reason": None})
            _event(events, job, "reconciled_completed", reason="existing_success_output")
            changed += 1
    return changed


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
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("D3 v5 screen protocol drift")
    if manifest.get("formal_test_access") or manifest.get("test_labels_used_for_selection"):
        raise RuntimeError("D3 v5 screen attempted formal test access")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("Implementation hashes changed after D3 v5 screen freeze")
    if manifest.get("source_hashes") != _source_hashes():
        raise RuntimeError("Source hashes changed after D3 v5 screen freeze")
    jobs = manifest.get("jobs", [])
    if len(jobs) != len(ALPHAS):
        raise RuntimeError("D3 v5 screen job count drift")
    for job, alpha in zip(jobs, ALPHAS):
        if any(int(job[key]) != DEV_SEED for key in ("seed", "partition_seed", "trace_seed")):
            raise RuntimeError(f"Development seed drift in {job['job_id']}")
        if job.get("evaluation_split") != "validation" or int(job.get("rounds", -1)) != ROUNDS:
            raise RuntimeError(f"Validation protocol drift in {job['job_id']}")
        config = job["method_config"]
        if float(config.get("r2c_v3_fixed_server_alpha")) != alpha:
            raise RuntimeError(f"Alpha drift in {job['job_id']}")
        if tuple(config.get("r2c_v4_deployment_ema_betas", ())) != BETAS:
            raise RuntimeError(f"Beta grid drift in {job['job_id']}")


def _assign_ordinal_rank(
    frame: pd.DataFrame, name: str, columns: list[str], ascending: list[bool]
) -> None:
    order = frame.sort_values(columns, ascending=ascending, kind="mergesort").index.tolist()
    frame[name] = pd.NA
    for rank, index in enumerate(order, start=1):
        frame.loc[index, name] = rank


def _rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.loc[
        frame["complete"].astype(bool) & frame["target_hit_round"].notna()
    ].copy()
    if len(eligible) < 2:
        raise RuntimeError("Fewer than two complete target-reaching D3 v5 screen candidates")
    _assign_ordinal_rank(
        eligible,
        "accuracy_rank",
        ["last50_validation_accuracy", "recovery_deficit_auc20", "target_hit_round", "alpha", "beta"],
        [False, True, True, False, True],
    )
    _assign_ordinal_rank(
        eligible,
        "auc_rank",
        ["recovery_deficit_auc20", "last50_validation_accuracy", "target_hit_round", "alpha", "beta"],
        [True, False, True, False, True],
    )
    _assign_ordinal_rank(
        eligible,
        "tta_round_rank",
        ["target_hit_round", "last50_validation_accuracy", "recovery_deficit_auc20", "alpha", "beta"],
        [True, False, True, False, True],
    )
    eligible["rank_sum"] = (
        eligible["accuracy_rank"].astype(int)
        + eligible["auc_rank"].astype(int)
        + eligible["tta_round_rank"].astype(int)
    )
    return eligible.sort_values(
        ["rank_sum", "last50_validation_accuracy", "recovery_deficit_auc20", "target_hit_round", "alpha", "beta"],
        ascending=[True, False, True, True, False, True],
        kind="mergesort",
    )


def freeze_selection(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot freeze an incomplete D3 v5 screen")
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run_id = str(job["actual_run_id"])
        run_dir = RUN_ROOT / run_id
        rounds = read_chunked_table(run_dir, "round_metrics").sort_values("round")
        candidates = read_chunked_table(run_dir, "deployment_candidate_metrics")
        if rounds["round"].astype(int).tolist() != list(range(1, ROUNDS + 1)):
            raise RuntimeError(f"Incomplete round trajectory for {run_id}")
        event_rows = rounds.loc[rounds["event_offset_round"].astype(int) == 0]
        if len(event_rows) != 1:
            raise RuntimeError(f"Expected one registered event in {run_id}")
        event_round = int(event_rows.iloc[0]["round"])
        alpha = float(job["method_config"]["r2c_v3_fixed_server_alpha"])
        for beta, group in candidates.groupby("deployment_beta", sort=True):
            group = group.sort_values("round")
            complete = group["round"].astype(int).tolist() == list(range(1, ROUNDS + 1))
            recovery = recovery_auc20(
                group["round"].astype(int).tolist(),
                group["test_accuracy"].astype(float).tolist(),
                event_round,
            )
            merged = group.merge(
                rounds[["round", "algorithm_elapsed_s"]], on="round", validate="one_to_one"
            )
            reached = merged.loc[merged["test_accuracy"].astype(float) >= TARGET_ACCURACY]
            rows.append(
                {
                    "run_id": run_id,
                    "alpha": alpha,
                    "beta": float(beta),
                    "last50_validation_accuracy": float(group.tail(50)["test_accuracy"].mean()),
                    "recovery_deficit_auc20": recovery["recovery_deficit_auc20"],
                    "recovery_auc20_complete": bool(recovery["recovery_auc20_complete"]),
                    "target_hit_round": None if reached.empty else int(reached.iloc[0]["round"]),
                    "shared_screen_tta_s": (
                        None if reached.empty else float(reached.iloc[0]["algorithm_elapsed_s"])
                    ),
                    "complete": bool(complete and recovery["recovery_auc20_complete"]),
                    "test_labels_used": False,
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != len(ALPHAS) * len(BETAS):
        raise RuntimeError(f"Expected 12 D3 v5 screen candidates, found {len(frame)}")
    ranked = _rank_candidates(frame)
    rank_columns = ["accuracy_rank", "auc_rank", "tta_round_rank", "rank_sum"]
    for column in rank_columns:
        frame[column] = pd.NA
        frame.loc[ranked.index, column] = ranked[column]
    frame["overall_rank"] = pd.NA
    for rank, index in enumerate(ranked.index, start=1):
        frame.loc[index, "overall_rank"] = rank
    selected = ranked.head(2)
    frame["selected_for_full_validation"] = frame.index.isin(selected.index)
    atomic_parquet(TABLE_PATH, frame.sort_values(["alpha", "beta"]))
    atomic_csv(CSV_PATH, frame.sort_values(["alpha", "beta"]))

    base = _base_config()
    frozen: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        config = dict(base)
        config.update(
            {
                "r2c_v3_fixed_server_alpha": float(row["alpha"]),
                "r2c_v4_deployment_ema_betas": [float(row["beta"])],
                "r2c_v4_primary_deployment_beta": float(row["beta"]),
            }
        )
        frozen.append(
            {
                "alpha": float(row["alpha"]),
                "beta": float(row["beta"]),
                "last50_validation_accuracy": float(row["last50_validation_accuracy"]),
                "recovery_deficit_auc20": float(row["recovery_deficit_auc20"]),
                "target_hit_round": int(row["target_hit_round"]),
                "rank_sum": int(row["rank_sum"]),
                "overall_rank": int(row["overall_rank"]),
                "method_config": config,
                "config_hash": config_hash(config),
            }
        )
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "dataset_id": "D3",
        "scenario_id": "S4",
        "protocol_version": PROTOCOL_VERSION,
        "selection_split": "validation",
        "test_labels_used": False,
        "formal_test_authorized": False,
        "screen_manifest_hash": config_hash(manifest),
        "candidate_count": len(frame),
        "selection_rule": manifest["selection_rule"],
        "selected_candidates": frozen,
    }
    atomic_json(SELECTION_PATH, payload)
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
            _event(events, job, "failed", exit_code=process.returncode, reason=job["failure_reason"])
            _sync(state, manifest)
            atomic_json(MANIFEST_PATH, manifest)
            atomic_json(STATE_PATH, state)
            return state
        _sync(state, manifest)
        atomic_json(MANIFEST_PATH, manifest)
        atomic_json(STATE_PATH, state)
    selection = freeze_selection(manifest)
    state.update(
        {
            "status": "screen_completed_selection_frozen",
            "current_job_id": None,
            "selected_candidates": [
                {"alpha": value["alpha"], "beta": value["beta"]}
                for value in selection["selected_candidates"]
            ],
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

