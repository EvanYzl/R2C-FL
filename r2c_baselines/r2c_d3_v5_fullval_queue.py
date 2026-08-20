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
from .r2c_v4 import PROTOCOL_VERSION
from .utils import atomic_csv, atomic_json, atomic_parquet, config_hash, sha256_file, utc_now


ROUNDS = 1000
TARGET_ACCURACY = 0.7986707616707616
BASELINE_METHODS = ("PowerOfChoice", "F3AST")
SCENARIOS = ("S0", "S4")
ACCURACY_CLOSE = 0.0015
AUC_CLOSE = 0.0001
TTA_CLOSE_MULTIPLIER = 1.05

SCREEN_SELECTION_PATH = QUEUE_ROOT / "r2c_d3_v5_screen_selection.json"
SCREEN_MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v5_screen_manifest_20260816T143505.552300Z.json"
BASELINE_CONFIG_PATH = QUEUE_ROOT / "selected_hyperparameters.json"
PLAN_PATH = PROJECT_ROOT / "refine-logs" / "D3_V5_OPTIMIZATION_PLAN_20260816_221904.md"
MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v5_fullval_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v5_fullval_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v5_fullval_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v5_fullval_result.json"
RUNS_PATH = PLOT_ROOT / "r2c_d3_v5_fullval_runs.parquet"
RUNS_CSV_PATH = PLOT_ROOT / "r2c_d3_v5_fullval_runs.csv"
COMPARISON_PATH = PLOT_ROOT / "r2c_d3_v5_fullval_comparison.json"


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    names = (
        "config.py",
        "data.py",
        "training.py",
        "methods.py",
        "r2c.py",
        "r2c_v2.py",
        "r2c_v3.py",
        "r2c_v4.py",
        "run.py",
        Path(__file__).name,
    )
    return {name: sha256_file(package / name) for name in names}


def _source_hashes() -> dict[str, str]:
    paths = (SCREEN_SELECTION_PATH, SCREEN_MANIFEST_PATH, BASELINE_CONFIG_PATH, PLAN_PATH)
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _load_screen_selection() -> dict[str, Any]:
    if not SCREEN_SELECTION_PATH.exists():
        raise RuntimeError("Phase B requires the frozen Phase A selection")
    selection = json.loads(SCREEN_SELECTION_PATH.read_text(encoding="utf-8"))
    if selection.get("selection_split") != "validation" or selection.get("test_labels_used"):
        raise RuntimeError("Phase A selection was not validation-only")
    if selection.get("formal_test_authorized"):
        raise RuntimeError("Phase A must not authorize formal test access")
    candidates = selection.get("selected_candidates", [])
    if len(candidates) != 2:
        raise RuntimeError(f"Phase A must freeze exactly two candidates, found {len(candidates)}")
    for candidate in candidates:
        config = candidate["method_config"]
        if config.get("r2c_protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError("Selected candidate protocol drift")
        beta = float(candidate["beta"])
        if config.get("r2c_v4_deployment_ema_betas") != [beta]:
            raise RuntimeError("Selected candidate must contain one frozen deployment beta")
        if float(config.get("r2c_v4_primary_deployment_beta")) != beta:
            raise RuntimeError("Selected candidate primary beta drift")
        if config_hash(config) != candidate.get("config_hash"):
            raise RuntimeError("Selected candidate config hash drift")
    return selection


def _baseline_configs() -> dict[str, dict[str, Any]]:
    selected = json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    configs = {method: dict(selected[method]["D3"]) for method in BASELINE_METHODS}
    if configs["PowerOfChoice"] != {"lr_mult": 1.0, "pow_d": 16}:
        raise RuntimeError("Frozen D3 PowerOfChoice config drift")
    if configs["F3AST"] != {"f3ast_beta": 0.0005, "lr_mult": 1.0}:
        raise RuntimeError("Frozen D3 F3AST config drift")
    return configs


def _r2c_job(position: int, candidate: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    label = f"C{position}-A{int(round(float(candidate['alpha']) * 1000)):04d}-B{int(round(float(candidate['beta']) * 1000)):04d}"
    run_id = f"A-R2C-D3-{scenario_id}-V5FULLVAL-{label}-s{DEV_SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v5_matched_full_validation",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": "D3",
        "scenario_id": scenario_id,
        "rounds": ROUNDS,
        "method_config": dict(candidate["method_config"]),
        "block_id": "A-R2C-D3-V5-FULLVAL",
        "seed": DEV_SEED,
        "partition_seed": DEV_SEED,
        "trace_seed": DEV_SEED,
        "evaluation_split": "validation",
        "full_logging": False,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "variant_label": label,
        "candidate_position": position,
        "test_labels_used_for_selection": False,
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def _baseline_job(method_id: str, config: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    run_id = f"BVAL-D3-{scenario_id}-{method_id}-s{DEV_SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v5_matched_full_validation_baseline",
        "mode": "calibration",
        "method_id": method_id,
        "dataset_id": "D3",
        "scenario_id": scenario_id,
        "rounds": ROUNDS,
        "method_config": dict(config),
        "block_id": "D3-V5-FULLVAL-BASELINE",
        "seed": DEV_SEED,
        "partition_seed": DEV_SEED,
        "trace_seed": DEV_SEED,
        "evaluation_split": "validation",
        "full_logging": False,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "variant_label": method_id,
        "candidate_position": None,
        "test_labels_used_for_selection": False,
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def _build_jobs(
    selection: dict[str, Any], baseline_configs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for method_id in BASELINE_METHODS:
        jobs.extend(_baseline_job(method_id, baseline_configs[method_id], scenario) for scenario in SCENARIOS)
    for position, candidate in enumerate(selection["selected_candidates"], start=1):
        jobs.extend(_r2c_job(position, candidate, scenario) for scenario in SCENARIOS)
    return jobs


def build_manifest(persist: bool = True, force: bool = False) -> dict[str, Any]:
    if persist and MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if force and MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if any(job.get("status") != "pending" or int(job.get("attempts", 0)) for job in existing["jobs"]):
            raise RuntimeError("Refusing to rebuild a started D3 v5 full-validation manifest")
    selection = _load_screen_selection()
    jobs = _build_jobs(selection, _baseline_configs())
    if len(jobs) != 8 or len({job["job_id"] for job in jobs}) != 8:
        raise AssertionError("D3 v5 full validation must contain exactly 8 unique jobs")
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_only_v5_phase_b_matched_full_validation",
        "protocol_version": PROTOCOL_VERSION,
        "formal_test_access": False,
        "test_labels_used_for_selection": False,
        "dev_seed": DEV_SEED,
        "rounds_per_job": ROUNDS,
        "baseline_methods": list(BASELINE_METHODS),
        "scenario_order": list(SCENARIOS),
        "job_order": [job["job_id"] for job in jobs],
        "screen_selection_hash": config_hash(selection),
        "close_limits": {
            "accuracy_fraction": ACCURACY_CLOSE,
            "auc_fraction": AUC_CLOSE,
            "tta_multiplier": TTA_CLOSE_MULTIPLIER,
        },
        "formal_authorization_rule": (
            "four strict wins, or exactly three strict wins and the sole miss is within its frozen close limit"
        ),
        "winner_rule": (
            "strict_passes(desc), normalized_margin_score(desc), screen_overall_rank(asc)"
        ),
        "implementation_hashes": _implementation_hashes(),
        "source_hashes": _source_hashes(),
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d3_v5_fullval_manifest_{stamp}.json", manifest)
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
    if manifest.get("formal_test_access") or manifest.get("test_labels_used_for_selection"):
        raise RuntimeError("D3 v5 full validation attempted formal test access")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("Implementation hashes changed after D3 v5 full-validation freeze")
    if manifest.get("source_hashes") != _source_hashes():
        raise RuntimeError("Source hashes changed after D3 v5 full-validation freeze")
    jobs = manifest.get("jobs", [])
    if len(jobs) != 8 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("D3 v5 full-validation job matrix drift")
    for job in jobs:
        if any(int(job[key]) != DEV_SEED for key in ("seed", "partition_seed", "trace_seed")):
            raise RuntimeError(f"Development seed drift in {job['job_id']}")
        if job.get("evaluation_split") != "validation" or int(job.get("rounds", -1)) != ROUNDS:
            raise RuntimeError(f"Validation protocol drift in {job['job_id']}")


def _time_to_accuracy(run_dir: Path, target: float) -> tuple[int | None, float | None]:
    rounds = read_chunked_table(run_dir, "round_metrics")
    reached = rounds.loc[rounds["test_accuracy"].astype(float) >= float(target)]
    if reached.empty:
        return None, None
    first = reached.sort_values("round").iloc[0]
    return int(first["round"]), float(first["algorithm_elapsed_s"])


def _evaluate_candidate(
    observed: dict[str, float | None], envelope: dict[str, float]
) -> dict[str, Any]:
    tta = observed["s4_algorithm_tta_s"]
    checks = {
        "s0_last50_accuracy": float(observed["s0_last50_accuracy"]) > envelope["s0_last50_accuracy"],
        "s4_last50_accuracy": float(observed["s4_last50_accuracy"]) > envelope["s4_last50_accuracy"],
        "s4_recovery_deficit_auc20": float(observed["s4_recovery_deficit_auc20"])
        < envelope["s4_recovery_deficit_auc20"],
        "s4_algorithm_tta_s": tta is not None and float(tta) < envelope["s4_algorithm_tta_s"],
    }
    close = {
        "s0_last50_accuracy": float(observed["s0_last50_accuracy"])
        >= envelope["s0_last50_accuracy"] - ACCURACY_CLOSE,
        "s4_last50_accuracy": float(observed["s4_last50_accuracy"])
        >= envelope["s4_last50_accuracy"] - ACCURACY_CLOSE,
        "s4_recovery_deficit_auc20": float(observed["s4_recovery_deficit_auc20"])
        <= envelope["s4_recovery_deficit_auc20"] + AUC_CLOSE,
        "s4_algorithm_tta_s": tta is not None
        and float(tta) <= envelope["s4_algorithm_tta_s"] * TTA_CLOSE_MULTIPLIER,
    }
    misses = [name for name, passed in checks.items() if not passed]
    authorized = not misses or (len(misses) == 1 and close[misses[0]])
    tta_margin = (
        -1.0e9
        if tta is None
        else (envelope["s4_algorithm_tta_s"] - float(tta))
        / (0.05 * envelope["s4_algorithm_tta_s"])
    )
    margin_score = (
        (float(observed["s0_last50_accuracy"]) - envelope["s0_last50_accuracy"])
        / ACCURACY_CLOSE
        + (float(observed["s4_last50_accuracy"]) - envelope["s4_last50_accuracy"])
        / ACCURACY_CLOSE
        + (envelope["s4_recovery_deficit_auc20"] - float(observed["s4_recovery_deficit_auc20"]))
        / AUC_CLOSE
        + tta_margin
    )
    return {
        "strict_checks": checks,
        "close_checks": close,
        "strict_passes": sum(checks.values()),
        "sole_miss": misses[0] if len(misses) == 1 else None,
        "formal_authorized": bool(authorized),
        "normalized_margin_score": float(margin_score),
    }


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot freeze incomplete D3 v5 full validation")
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run_id = str(job["actual_run_id"])
        run_dir = RUN_ROOT / run_id
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        auc = result["recovery"]["recovery_deficit_auc20"]
        if job["scenario_id"] == "S4" and auc is None:
            raise RuntimeError(f"D3 full-validation S4 lacks complete AUC@20: {run_id}")
        tta_round, tta_s = _time_to_accuracy(run_dir, TARGET_ACCURACY)
        rows.append(
            {
                "run_id": run_id,
                "method_id": job["method_id"],
                "candidate_position": job.get("candidate_position"),
                "variant_label": job["variant_label"],
                "scenario_id": job["scenario_id"],
                "last50_validation_accuracy": float(result["last50_accuracy"]),
                "recovery_deficit_auc20": None if auc is None else float(auc),
                "algorithm_elapsed_s": float(result["algorithm_elapsed_s"]),
                "tta_round": tta_round,
                "algorithm_tta_s": tta_s,
                "source_kind": "CALIBRATION",
                "test_labels_used": False,
            }
        )
    frame = pd.DataFrame(rows)
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    baseline = frame.loc[frame["method_id"].isin(BASELINE_METHODS)]
    envelope = {
        "s0_last50_accuracy": float(
            baseline.loc[baseline["scenario_id"] == "S0", "last50_validation_accuracy"].max()
        ),
        "s4_last50_accuracy": float(
            baseline.loc[baseline["scenario_id"] == "S4", "last50_validation_accuracy"].max()
        ),
        "s4_recovery_deficit_auc20": float(
            baseline.loc[baseline["scenario_id"] == "S4", "recovery_deficit_auc20"].min()
        ),
        "s4_algorithm_tta_s": float(
            baseline.loc[baseline["scenario_id"] == "S4", "algorithm_tta_s"].min()
        ),
    }
    selection = _load_screen_selection()
    evaluated: list[dict[str, Any]] = []
    for candidate in selection["selected_candidates"]:
        position = int(candidate["overall_rank"])
        subset = frame.loc[
            (frame["method_id"] == "R2C-FL") & (frame["candidate_position"].astype("Int64") == position)
        ]
        if set(subset["scenario_id"]) != set(SCENARIOS):
            raise RuntimeError(f"Incomplete full-validation pair for screen rank {position}")
        s0 = subset.loc[subset["scenario_id"] == "S0"].iloc[0]
        s4 = subset.loc[subset["scenario_id"] == "S4"].iloc[0]
        observed: dict[str, float | None] = {
            "s0_last50_accuracy": float(s0["last50_validation_accuracy"]),
            "s4_last50_accuracy": float(s4["last50_validation_accuracy"]),
            "s4_recovery_deficit_auc20": float(s4["recovery_deficit_auc20"]),
            "s4_algorithm_tta_s": None if pd.isna(s4["algorithm_tta_s"]) else float(s4["algorithm_tta_s"]),
        }
        evaluated.append(
            {
                "screen_overall_rank": position,
                "alpha": float(candidate["alpha"]),
                "beta": float(candidate["beta"]),
                "config_hash": candidate["config_hash"],
                "method_config": candidate["method_config"],
                "observed": observed,
                **_evaluate_candidate(observed, envelope),
                "run_ids": subset.sort_values("scenario_id")["run_id"].tolist(),
            }
        )
    authorized = [candidate for candidate in evaluated if candidate["formal_authorized"]]
    authorized.sort(
        key=lambda value: (
            -int(value["strict_passes"]),
            -float(value["normalized_margin_score"]),
            int(value["screen_overall_rank"]),
        )
    )
    winner = authorized[0] if authorized else None
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "dataset_id": "D3",
        "selection_split": "validation",
        "test_labels_used": False,
        "baseline_envelope": envelope,
        "candidate_evaluations": evaluated,
        "formal_authorized": winner is not None,
        "winner": winner,
        "fullval_manifest_hash": config_hash(manifest),
    }
    atomic_json(RESULT_PATH, payload)
    atomic_json(COMPARISON_PATH, payload)
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
    result = freeze_result(manifest)
    state.update(
        {
            "status": (
                "full_validation_completed_authorized"
                if result["formal_authorized"]
                else "full_validation_completed_not_authorized"
            ),
            "current_job_id": None,
            "formal_authorized": bool(result["formal_authorized"]),
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

