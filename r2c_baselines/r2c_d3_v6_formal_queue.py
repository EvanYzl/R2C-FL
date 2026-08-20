from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import FORMAL_SEED, PLOT_ROOT, PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .logging_io import read_chunked_table
from .metrics import recovery_auc20
from .r2c_v5 import PROTOCOL_VERSION
from .utils import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    config_hash,
    sha256_file,
    utc_now,
)


ROUNDS = 1000
TARGET_ACCURACY = 0.7986707616707616
FORMAL_INTERPRETATION = "matched_seed_engineering_reevaluation_not_untouched_confirmation"
SCENARIOS = ("S0", "S4")
THRESHOLDS = {
    "s0_last50_accuracy": 0.8902665949600491,
    "s4_last50_accuracy": 0.8913398893669331,
    "s4_recovery_deficit_auc20": 0.0000877765826674815,
    "s4_algorithm_tta_s": 202.493191700109,
}
CLOSE_LIMITS = {
    "accuracy_fraction": 0.0015,
    "auc_fraction": 0.0001,
    "tta_multiplier": 1.05,
}
EXPECTED_BASELINE_TABLE_SHA256 = (
    "9792BC96F628C5EE3E2F517FEBC17EDCC3A7349715974D829424662069DD41E6"
)
EXPECTED_PLAN_SHA256 = (
    "560E0BC89F9A5FB21D2F54D894CCCE14A83FF3662735F9724200A3CF7CE6709D"
)

FULLVAL_RESULT_PATH = QUEUE_ROOT / "r2c_d3_v6_fullval_result.json"
FULLVAL_MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v6_fullval_manifest.json"
BASELINE_TABLE_PATH = (
    PLOT_ROOT / "table1_v4_matched_combined_values_20260816_140305.csv"
)
PLAN_PATH = (
    PROJECT_ROOT
    / "refine-logs"
    / "D3_V6_DIVERSITY_PLAN_AMENDMENT_20260817_103843.md"
)
MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v6_formal_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v6_formal_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v6_formal_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v6_formal_result.json"
RUNS_PATH = PLOT_ROOT / "r2c_d3_v6_formal_runs.parquet"
RUNS_CSV_PATH = PLOT_ROOT / "r2c_d3_v6_formal_runs.csv"
AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / "tests" / "audit_r2c_run.py"


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
        "run.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    values["tests/audit_r2c_run.py"] = sha256_file(AUDIT_SCRIPT)
    return values


def _source_paths() -> tuple[Path, ...]:
    return (FULLVAL_RESULT_PATH, FULLVAL_MANIFEST_PATH, BASELINE_TABLE_PATH, PLAN_PATH)


def _source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in _source_paths()
    }


def _verify_static_sources() -> None:
    if sha256_file(BASELINE_TABLE_PATH) != EXPECTED_BASELINE_TABLE_SHA256.lower():
        raise RuntimeError("Frozen Table 1 baseline source hash drift")
    if sha256_file(PLAN_PATH) != EXPECTED_PLAN_SHA256.lower():
        raise RuntimeError("Frozen D3 v6 plan hash drift")


def _verify_frozen_thresholds() -> dict[str, Any]:
    _verify_static_sources()
    table = pd.read_csv(BASELINE_TABLE_PATH)
    d3 = table.loc[
        (table["dataset_id"] == "D3") & (table["method_id"] != "R2C-FL")
    ].copy()
    mapping = {
        "s0_last50_accuracy": ("s0_last50_accuracy_pct", "max", 0.01),
        "s4_last50_accuracy": ("s4_last50_accuracy_pct", "max", 0.01),
        "s4_recovery_deficit_auc20": (
            "s4_recovery_deficit_auc20_pp",
            "min",
            0.01,
        ),
        "s4_algorithm_tta_s": ("s4_tta_h", "min", 3600.0),
    }
    lineage: dict[str, Any] = {}
    for key, (metric, direction, multiplier) in mapping.items():
        rows = d3.loc[d3["metric"] == metric].copy()
        rows["value"] = rows["value"].astype(float)
        index = rows["value"].idxmax() if direction == "max" else rows["value"].idxmin()
        row = rows.loc[index]
        value = float(row["value"]) * multiplier
        if abs(value - THRESHOLDS[key]) > 1.0e-12:
            raise RuntimeError(f"Frozen external threshold drift for {key}: {value}")
        lineage[key] = {
            "method_id": str(row["method_id"]),
            "source_metric": metric,
            "threshold": value,
            "source_run_ids": str(row["source_run_ids"]),
        }
    return lineage


def _load_winner() -> dict[str, Any]:
    _verify_static_sources()
    if not FULLVAL_RESULT_PATH.exists() or not FULLVAL_MANIFEST_PATH.exists():
        raise RuntimeError("Formal D3 v6 queue requires a frozen Phase D result and manifest")
    result = json.loads(FULLVAL_RESULT_PATH.read_text(encoding="utf-8"))
    if result.get("selection_split") != "validation" or result.get("test_labels_used"):
        raise RuntimeError("Phase D winner was not selected on validation only")
    if not result.get("formal_authorized") or not result.get("winner"):
        raise RuntimeError("Phase D did not authorize formal D3 execution")
    winner = dict(result["winner"])
    if not winner.get("formal_authorized") or not winner.get("diversity_preserved"):
        raise RuntimeError("Phase D winner did not pass metric and diversity gates")
    config = dict(winner["method_config"])
    if config.get("r2c_protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Phase D winner protocol drift")
    if config_hash(config) != winner.get("config_hash"):
        raise RuntimeError("Phase D winner config hash drift")
    if float(config.get("r2c_v3_fixed_server_alpha")) != 1.0:
        raise RuntimeError("Phase D winner alpha drift")
    if config.get("r2c_v4_deployment_ema_betas") != [0.9]:
        raise RuntimeError("Phase D winner beta drift")
    if float(config.get("r2c_v5_history_temperature")) != 1.0:
        raise RuntimeError("Phase D winner history temperature drift")
    if float(config.get("r2c_v5_history_mix")) != float(winner["history_mix"]):
        raise RuntimeError("Phase D winner history-mix drift")
    return winner


def _formal_config(winner: dict[str, Any]) -> dict[str, Any]:
    config = dict(winner["method_config"])
    if bool(config.get("r2c_v2_audit_replay")):
        raise RuntimeError("Phase D winner unexpectedly enabled audit replay")
    # Audit replay is instrumentation-only and excluded from algorithm elapsed time.
    config["r2c_v2_audit_replay"] = True
    return config


def _job(winner: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    history_mix = float(winner["history_mix"])
    rank = int(winner["phase_c_overall_rank"])
    label = f"R{rank}-L{int(round(history_mix * 1000)):04d}"
    run_id = f"A-R2C-D3-{scenario_id}-V6DIV-FORMAL-{label}-s{FORMAL_SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v6_diversity_matched_seed_formal",
        "mode": "formal",
        "method_id": "R2C-FL",
        "dataset_id": "D3",
        "scenario_id": scenario_id,
        "rounds": ROUNDS,
        "method_config": _formal_config(winner),
        "block_id": "A-R2C-D3-V6-DIVERSITY-FORMAL",
        "seed": FORMAL_SEED,
        "partition_seed": FORMAL_SEED,
        "trace_seed": FORMAL_SEED,
        "evaluation_split": "test",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "variant_label": label,
        "history_mix": history_mix,
        "phase_c_overall_rank": rank,
        "formal_interpretation": FORMAL_INTERPRETATION,
        "test_labels_used_for_selection": False,
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def _build_jobs(winner: dict[str, Any]) -> list[dict[str, Any]]:
    return [_job(winner, scenario) for scenario in SCENARIOS]


def build_manifest(persist: bool = True, force: bool = False) -> dict[str, Any]:
    if persist and MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if force and MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if any(
            job.get("status") != "pending" or int(job.get("attempts", 0))
            for job in existing["jobs"]
        ):
            raise RuntimeError("Refusing to rebuild a started D3 v6 formal manifest")
    if int(FORMAL_SEED) != 20260811:
        raise RuntimeError(f"Expected formal seed 20260811, found {FORMAL_SEED}")
    winner = _load_winner()
    threshold_lineage = _verify_frozen_thresholds()
    jobs = _build_jobs(winner)
    if len(jobs) != 2 or len({job["job_id"] for job in jobs}) != 2:
        raise RuntimeError("D3 v6 formal queue must contain exactly one S0/S4 pair")
    formal_config = _formal_config(winner)
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_only_v6_single_matched_seed_formal_pair",
        "protocol_version": PROTOCOL_VERSION,
        "formal_test_access": True,
        "formal_interpretation": FORMAL_INTERPRETATION,
        "test_labels_used_for_selection": False,
        "formal_seed": FORMAL_SEED,
        "rounds_per_job": ROUNDS,
        "scenario_order": list(SCENARIOS),
        "job_order": [job["job_id"] for job in jobs],
        "validation_winner": winner,
        "validation_winner_config_hash": winner["config_hash"],
        "formal_config_hash": config_hash(formal_config),
        "audit_only_config_change": {"r2c_v2_audit_replay": [False, True]},
        "thresholds": THRESHOLDS,
        "close_limits": CLOSE_LIMITS,
        "threshold_lineage": threshold_lineage,
        "stopping_rule": (
            "four strict wins, or exactly three strict wins and the sole miss is within its frozen close limit"
        ),
        "resource_gate": "two GPU utilization samples below 10 percent at least 30 seconds apart",
        "implementation_hashes": _implementation_hashes(),
        "source_hashes": _source_hashes(),
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d3_v6_formal_manifest_{stamp}.json", manifest)
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
                "total": 2,
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
        if run_id is not None and (
            job.get("status") != "completed" or job.get("actual_run_id") != run_id
        ):
            job.update(
                {"status": "completed", "actual_run_id": run_id, "failure_reason": None}
            )
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
        raise RuntimeError("D3 v6 formal protocol drift")
    if not manifest.get("formal_test_access") or manifest.get("test_labels_used_for_selection"):
        raise RuntimeError("D3 v6 formal access contract drift")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("Implementation hashes changed after D3 v6 formal freeze")
    if manifest.get("source_hashes") != _source_hashes():
        raise RuntimeError("Source hashes changed after D3 v6 formal freeze")
    jobs = manifest.get("jobs", [])
    if len(jobs) != 2 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("D3 v6 formal matrix drift")
    for job, scenario in zip(jobs, SCENARIOS):
        if (
            job.get("dataset_id") != "D3"
            or job.get("scenario_id") != scenario
            or job.get("evaluation_split") != "test"
        ):
            raise RuntimeError(f"Scenario/test split drift in {job['job_id']}")
        if any(
            int(job[key]) != FORMAL_SEED
            for key in ("seed", "partition_seed", "trace_seed")
        ):
            raise RuntimeError(f"Formal seed drift in {job['job_id']}")
        if (
            int(job.get("rounds", -1)) != ROUNDS
            or not job.get("full_logging")
            or not job["method_config"].get("r2c_v2_audit_replay")
        ):
            raise RuntimeError(f"Formal logging/budget drift in {job['job_id']}")


def _time_to_accuracy(rounds: pd.DataFrame, target: float) -> tuple[int | None, float | None]:
    reached = rounds.loc[rounds["test_accuracy"].astype(float) >= float(target)]
    if reached.empty:
        return None, None
    first = reached.sort_values("round").iloc[0]
    return int(first["round"]), float(first["algorithm_elapsed_s"])


def _evaluate_termination(observed: dict[str, float | None]) -> dict[str, Any]:
    tta = observed["s4_algorithm_tta_s"]
    checks = {
        "s0_last50_accuracy": float(observed["s0_last50_accuracy"])
        > THRESHOLDS["s0_last50_accuracy"],
        "s4_last50_accuracy": float(observed["s4_last50_accuracy"])
        > THRESHOLDS["s4_last50_accuracy"],
        "s4_recovery_deficit_auc20": float(observed["s4_recovery_deficit_auc20"])
        < THRESHOLDS["s4_recovery_deficit_auc20"],
        "s4_algorithm_tta_s": tta is not None
        and float(tta) < THRESHOLDS["s4_algorithm_tta_s"],
    }
    close = {
        "s0_last50_accuracy": float(observed["s0_last50_accuracy"])
        >= THRESHOLDS["s0_last50_accuracy"] - CLOSE_LIMITS["accuracy_fraction"],
        "s4_last50_accuracy": float(observed["s4_last50_accuracy"])
        >= THRESHOLDS["s4_last50_accuracy"] - CLOSE_LIMITS["accuracy_fraction"],
        "s4_recovery_deficit_auc20": float(observed["s4_recovery_deficit_auc20"])
        <= THRESHOLDS["s4_recovery_deficit_auc20"] + CLOSE_LIMITS["auc_fraction"],
        "s4_algorithm_tta_s": tta is not None
        and float(tta) <= THRESHOLDS["s4_algorithm_tta_s"] * CLOSE_LIMITS["tta_multiplier"],
    }
    misses = [name for name, passed in checks.items() if not passed]
    goal_met = not misses or (len(misses) == 1 and close[misses[0]])
    return {
        "strict_checks": checks,
        "close_checks": close,
        "strict_passes": sum(checks.values()),
        "sole_miss": misses[0] if len(misses) == 1 else None,
        "goal_met": bool(goal_met),
    }


def _verified_rounds(run_dir: Path) -> pd.DataFrame:
    root = run_dir / "tables" / "round_metrics"
    index = json.loads((root / "_index.json").read_text(encoding="utf-8"))
    for part in index.get("parts", []):
        path = root / str(part["path"])
        if path.stat().st_size != int(part["bytes"]) or sha256_file(path) != str(
            part["sha256"]
        ):
            raise RuntimeError(f"Formal round chunk drift: {path}")
    rounds = read_chunked_table(run_dir, "round_metrics").sort_values("round")
    if (
        len(rounds) != int(index["rows"])
        or rounds["round"].astype(int).tolist() != list(range(1, ROUNDS + 1))
    ):
        raise RuntimeError(f"Formal round trajectory incomplete: {run_dir.name}")
    return rounds


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot freeze incomplete D3 v6 formal pair")
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run_id = str(job["actual_run_id"])
        run_dir = RUN_ROOT / run_id
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        run_manifest = pd.read_parquet(run_dir / "run_manifest.parquet").iloc[0]
        if (
            str(run_manifest["source_kind"]) != "REPRODUCED"
            or str(run_manifest["status"]) != "completed"
        ):
            raise RuntimeError(f"Formal provenance/status failure for {run_id}")
        for key in ("seed", "partition_seed", "trace_seed"):
            if int(run_manifest[key]) != FORMAL_SEED:
                raise RuntimeError(f"{key} mismatch for {run_id}")
        if str(run_manifest["upstream_commit"]) != PROTOCOL_VERSION:
            raise RuntimeError(f"Protocol lineage mismatch for {run_id}")
        expected_rows = {
            "round_metrics": ROUNDS,
            "client_round_metrics": ROUNDS * 100,
            "checkpoint_metrics": ROUNDS * 20,
            "certificate_audit": ROUNDS,
            "deployment_candidate_metrics": ROUNDS,
        }
        for table, expected in expected_rows.items():
            if int(result["table_indices"][table]["rows"]) != expected:
                raise RuntimeError(f"{run_id} {table} row-count failure")
        rounds = _verified_rounds(run_dir)
        auc: float | None = None
        if job["scenario_id"] == "S4":
            events = rounds.loc[rounds["event_offset_round"].astype(int) == 0]
            if len(events) != 1 or int(events.iloc[0]["round"]) != 500:
                raise RuntimeError(f"Formal S4 event drift for {run_id}")
            direct = recovery_auc20(
                rounds["round"].astype(int).tolist(),
                rounds["test_accuracy"].astype(float).tolist(),
                500,
            )
            if not direct["recovery_auc20_complete"]:
                raise RuntimeError(f"Strict AUC@20 window incomplete for {run_id}")
            auc = float(direct["recovery_deficit_auc20"])
            stored = result["recovery"]["recovery_deficit_auc20"]
            if stored is None or abs(float(stored) - auc) > 1.0e-15:
                raise RuntimeError(f"Stored/direct formal AUC mismatch for {run_id}")
        tta_round, tta_s = _time_to_accuracy(rounds, TARGET_ACCURACY)
        rows.append(
            {
                "run_id": run_id,
                "dataset_id": "D3",
                "scenario_id": job["scenario_id"],
                "seed": FORMAL_SEED,
                "last50_test_accuracy": float(result["last50_accuracy"]),
                "recovery_deficit_auc20": auc,
                "recovery_auc20_complete": bool(
                    result["recovery"]["recovery_auc20_complete"]
                ),
                "algorithm_elapsed_s": float(result["algorithm_elapsed_s"]),
                "tta_round": tta_round,
                "algorithm_tta_s": tta_s,
                "source_kind": "REPRODUCED",
                "protocol_version": PROTOCOL_VERSION,
                "formal_interpretation": FORMAL_INTERPRETATION,
                "test_labels_used_for_selection": False,
            }
        )
    frame = pd.DataFrame(rows)
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    s0 = frame.loc[frame["scenario_id"] == "S0"].iloc[0]
    s4 = frame.loc[frame["scenario_id"] == "S4"].iloc[0]
    observed: dict[str, float | None] = {
        "s0_last50_accuracy": float(s0["last50_test_accuracy"]),
        "s4_last50_accuracy": float(s4["last50_test_accuracy"]),
        "s4_recovery_deficit_auc20": float(s4["recovery_deficit_auc20"]),
        "s4_algorithm_tta_s": (
            None if pd.isna(s4["algorithm_tta_s"]) else float(s4["algorithm_tta_s"])
        ),
    }
    termination = _evaluate_termination(observed)
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "dataset_id": "D3",
        "formal_seed": FORMAL_SEED,
        "formal_interpretation": FORMAL_INTERPRETATION,
        "source_kind": "REPRODUCED",
        "selection_split": "validation",
        "test_labels_used_for_selection": False,
        "thresholds": THRESHOLDS,
        "close_limits": CLOSE_LIMITS,
        "observed": observed,
        **termination,
        "run_ids": frame["run_id"].tolist(),
        "manifest_hash": config_hash(manifest),
    }
    atomic_json(RESULT_PATH, payload)
    return payload


def _audit_run(run_id: str, log_path: Path) -> subprocess.CompletedProcess[str]:
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        return subprocess.run(
            [sys.executable, str(AUDIT_SCRIPT), str(RUN_ROOT / run_id)],
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )


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
                "formal_completed_goal_met"
                if result["goal_met"]
                else "formal_completed_goal_not_met"
            ),
            "current_job_id": None,
            "goal_met": bool(result["goal_met"]),
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
