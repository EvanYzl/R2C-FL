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
from .r2c_v5 import PROTOCOL_VERSION
from .utils import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    config_hash,
    sha256_file,
    utc_now,
)


HISTORY_MIXES = (0.15, 0.30, 0.45, 0.60)
HISTORY_TEMPERATURE = 1.0
ROUNDS = 600
TARGET_ACCURACY = 0.7986707616707616
CONTROL_FINAL_JFI = 0.8643830945875212
CONTROL_FINAL_WORST10 = 27.8
EXPECTED_SOURCE_RESULT_SHA256 = (
    "9EAC04D44D3BE00F0EFA81484EABA3F7485CC2551B59384E1E39CE9E8F39FC59"
)
EXPECTED_PLAN_SHA256 = (
    "560E0BC89F9A5FB21D2F54D894CCCE14A83FF3662735F9724200A3CF7CE6709D"
)

SOURCE_RESULT_PATH = QUEUE_ROOT / "r2c_d3_v5_fullval_result.json"
PLAN_PATH = (
    PROJECT_ROOT
    / "refine-logs"
    / "D3_V6_DIVERSITY_PLAN_AMENDMENT_20260817_103843.md"
)
MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v6_diversity_screen_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v6_diversity_screen_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v6_diversity_screen_scheduler_events.parquet"
SELECTION_PATH = QUEUE_ROOT / "r2c_d3_v6_diversity_screen_selection.json"
TABLE_PATH = PLOT_ROOT / "r2c_d3_v6_diversity_screen_candidates.parquet"
CSV_PATH = PLOT_ROOT / "r2c_d3_v6_diversity_screen_candidates.csv"
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


def _source_hashes() -> dict[str, str]:
    return {
        str(SOURCE_RESULT_PATH.relative_to(PROJECT_ROOT)): sha256_file(SOURCE_RESULT_PATH),
        str(PLAN_PATH.relative_to(PROJECT_ROOT)): sha256_file(PLAN_PATH),
    }


def _base_config() -> dict[str, Any]:
    if sha256_file(SOURCE_RESULT_PATH) != EXPECTED_SOURCE_RESULT_SHA256.lower():
        raise RuntimeError("Frozen D3 v5 full-validation result hash drift")
    if sha256_file(PLAN_PATH) != EXPECTED_PLAN_SHA256.lower():
        raise RuntimeError("Frozen D3 v6 diversity plan hash drift")
    source = json.loads(SOURCE_RESULT_PATH.read_text(encoding="utf-8"))
    winner = source.get("winner")
    if not isinstance(winner, dict) or not winner.get("formal_authorized"):
        raise RuntimeError("Expected the frozen D3 v5 validation winner")
    config = dict(winner["method_config"])
    if float(config.get("r2c_v3_fixed_server_alpha")) != 1.0:
        raise RuntimeError("Frozen D3 v5 winner alpha drift")
    if config.get("r2c_v4_deployment_ema_betas") != [0.9]:
        raise RuntimeError("Frozen D3 v5 winner beta drift")
    config.update(
        {
            "r2c_protocol_version": PROTOCOL_VERSION,
            "r2c_v2_audit_replay": False,
            "r2c_v4_deployment_ema_betas": [0.9],
            "r2c_v4_primary_deployment_beta": 0.9,
            "r2c_v5_history_temperature": HISTORY_TEMPERATURE,
        }
    )
    return config


def _job(history_mix: float, base_config: dict[str, Any]) -> dict[str, Any]:
    label = f"L{int(round(history_mix * 1000)):04d}"
    run_id = f"A-R2C-D3-S4-V6DIV-{label}-s{DEV_SEED}"
    config = dict(base_config)
    config["r2c_v5_history_mix"] = float(history_mix)
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v6_validation_history_diversity_screen",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": "D3",
        "scenario_id": "S4",
        "rounds": ROUNDS,
        "method_config": config,
        "block_id": "A-R2C-D3-V6-DIVERSITY-SCREEN",
        "seed": DEV_SEED,
        "partition_seed": DEV_SEED,
        "trace_seed": DEV_SEED,
        "evaluation_split": "validation",
        "full_logging": True,
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
        if any(
            job.get("status") != "pending" or int(job.get("attempts", 0))
            for job in existing["jobs"]
        ):
            raise RuntimeError("Refusing to rebuild a started D3 v6 diversity manifest")
    if int(DEV_SEED) != 20260810:
        raise RuntimeError(f"Expected development seed 20260810, found {DEV_SEED}")
    base_config = _base_config()
    jobs = [_job(value, base_config) for value in HISTORY_MIXES]
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_only_v6_phase_c_validation_history_diversity_screen",
        "protocol_version": PROTOCOL_VERSION,
        "formal_test_access": False,
        "test_labels_used_for_selection": False,
        "dev_seed": DEV_SEED,
        "rounds_per_job": ROUNDS,
        "candidate_history_mixes": list(HISTORY_MIXES),
        "history_temperature": HISTORY_TEMPERATURE,
        "control_final_jfi": CONTROL_FINAL_JFI,
        "control_final_worst10": CONTROL_FINAL_WORST10,
        "target_accuracy": TARGET_ACCURACY,
        "job_order": [job["job_id"] for job in jobs],
        "selection_rule": (
            "strict JFI/worst10 improvement over frozen 600-round control; then minimum "
            "ordinal rank sum over Last50(desc), AUC20(asc), target-hit-round(asc), "
            "JFI(desc), worst10(desc), with frozen tie breakers"
        ),
        "implementation_hashes": _implementation_hashes(),
        "source_hashes": _source_hashes(),
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(
            QUEUE_ROOT / f"r2c_d3_v6_diversity_screen_manifest_{stamp}.json",
            manifest,
        )
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
        raise RuntimeError("D3 v6 diversity protocol drift")
    if manifest.get("formal_test_access") or manifest.get("test_labels_used_for_selection"):
        raise RuntimeError("D3 v6 diversity screen attempted formal test access")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("Implementation hashes changed after D3 v6 diversity freeze")
    if manifest.get("source_hashes") != _source_hashes():
        raise RuntimeError("Source hashes changed after D3 v6 diversity freeze")
    jobs = manifest.get("jobs", [])
    if len(jobs) != len(HISTORY_MIXES):
        raise RuntimeError("D3 v6 diversity job count drift")
    for job, history_mix in zip(jobs, HISTORY_MIXES):
        if any(
            int(job[key]) != DEV_SEED for key in ("seed", "partition_seed", "trace_seed")
        ):
            raise RuntimeError(f"Development seed drift in {job['job_id']}")
        if (
            job.get("evaluation_split") != "validation"
            or int(job.get("rounds", -1)) != ROUNDS
            or not bool(job.get("full_logging"))
        ):
            raise RuntimeError(f"Validation protocol drift in {job['job_id']}")
        config = job["method_config"]
        if config.get("r2c_protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"Protocol drift in {job['job_id']}")
        if float(config.get("r2c_v5_history_mix")) != history_mix:
            raise RuntimeError(f"History-mix drift in {job['job_id']}")
        if float(config.get("r2c_v5_history_temperature")) != HISTORY_TEMPERATURE:
            raise RuntimeError(f"History-temperature drift in {job['job_id']}")
        if float(config.get("r2c_v3_fixed_server_alpha")) != 1.0:
            raise RuntimeError(f"Alpha drift in {job['job_id']}")
        if config.get("r2c_v4_deployment_ema_betas") != [0.9]:
            raise RuntimeError(f"Deployment beta drift in {job['job_id']}")


def _assign_ordinal_rank(
    frame: pd.DataFrame, name: str, columns: list[str], ascending: list[bool]
) -> None:
    order = frame.sort_values(columns, ascending=ascending, kind="mergesort").index.tolist()
    frame[name] = pd.NA
    for rank, index in enumerate(order, start=1):
        frame.loc[index, name] = rank


def _rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.loc[
        frame["complete"].astype(bool)
        & frame["target_hit_round"].notna()
        & (frame["final_participation_jfi"].astype(float) > CONTROL_FINAL_JFI)
        & (frame["final_worst10_participation"].astype(float) > CONTROL_FINAL_WORST10)
    ].copy()
    if eligible.empty:
        return eligible
    tie = [
        "last50_validation_accuracy",
        "recovery_deficit_auc20",
        "final_participation_jfi",
        "final_worst10_participation",
        "target_hit_round",
        "history_mix",
    ]
    directions = [False, True, False, False, True, True]
    rank_specs = (
        ("accuracy_rank", ["last50_validation_accuracy"] + tie[1:], directions),
        (
            "auc_rank",
            ["recovery_deficit_auc20", "last50_validation_accuracy"] + tie[2:],
            [True, False, False, False, True, True],
        ),
        (
            "tta_round_rank",
            ["target_hit_round", "last50_validation_accuracy", "recovery_deficit_auc20"]
            + tie[2:4]
            + ["history_mix"],
            [True, False, True, False, False, True],
        ),
        (
            "jfi_rank",
            ["final_participation_jfi", "last50_validation_accuracy", "recovery_deficit_auc20"]
            + ["final_worst10_participation", "target_hit_round", "history_mix"],
            [False, False, True, False, True, True],
        ),
        (
            "worst10_rank",
            ["final_worst10_participation", "last50_validation_accuracy", "recovery_deficit_auc20"]
            + ["final_participation_jfi", "target_hit_round", "history_mix"],
            [False, False, True, False, True, True],
        ),
    )
    for name, columns, ascending in rank_specs:
        _assign_ordinal_rank(eligible, name, columns, ascending)
    rank_columns = [name for name, _, _ in rank_specs]
    eligible["rank_sum"] = sum(eligible[name].astype(int) for name in rank_columns)
    return eligible.sort_values(
        ["rank_sum"] + tie,
        ascending=[True] + directions,
        kind="mergesort",
    )


def freeze_selection(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot freeze an incomplete D3 v6 diversity screen")
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run_id = str(job["actual_run_id"])
        run_dir = RUN_ROOT / run_id
        rounds = read_chunked_table(run_dir, "round_metrics").sort_values("round")
        if rounds["round"].astype(int).tolist() != list(range(1, ROUNDS + 1)):
            raise RuntimeError(f"Incomplete round trajectory for {run_id}")
        event_rows = rounds.loc[rounds["event_offset_round"].astype(int) == 0]
        if len(event_rows) != 1:
            raise RuntimeError(f"Expected one registered event in {run_id}")
        event_round = int(event_rows.iloc[0]["round"])
        recovery = recovery_auc20(
            rounds["round"].astype(int).tolist(),
            rounds["test_accuracy"].astype(float).tolist(),
            event_round,
        )
        reached = rounds.loc[rounds["test_accuracy"].astype(float) >= TARGET_ACCURACY]
        rows.append(
            {
                "run_id": run_id,
                "history_mix": float(job["method_config"]["r2c_v5_history_mix"]),
                "last50_validation_accuracy": float(rounds.tail(50)["test_accuracy"].mean()),
                "recovery_deficit_auc20": recovery["recovery_deficit_auc20"],
                "recovery_auc20_complete": bool(recovery["recovery_auc20_complete"]),
                "target_hit_round": None if reached.empty else int(reached.iloc[0]["round"]),
                "target_hit_algorithm_s": (
                    None if reached.empty else float(reached.iloc[0]["algorithm_elapsed_s"])
                ),
                "final_participation_jfi": float(rounds.iloc[-1]["participation_jfi"]),
                "final_worst10_participation": float(
                    rounds.iloc[-1]["worst10_participation"]
                ),
                "complete": bool(recovery["recovery_auc20_complete"]),
                "test_labels_used": False,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != len(HISTORY_MIXES):
        raise RuntimeError(f"Expected four D3 v6 diversity candidates, found {len(frame)}")
    ranked = _rank_candidates(frame)
    rank_columns = [
        "accuracy_rank",
        "auc_rank",
        "tta_round_rank",
        "jfi_rank",
        "worst10_rank",
        "rank_sum",
    ]
    for column in rank_columns:
        frame[column] = pd.NA
        if column in ranked:
            frame.loc[ranked.index, column] = ranked[column]
    frame["diversity_gate_passed"] = (
        frame["complete"].astype(bool)
        & (frame["final_participation_jfi"] > CONTROL_FINAL_JFI)
        & (frame["final_worst10_participation"] > CONTROL_FINAL_WORST10)
    )
    frame["overall_rank"] = pd.NA
    for rank, index in enumerate(ranked.index, start=1):
        frame.loc[index, "overall_rank"] = rank
    selected = ranked.head(2) if len(ranked) >= 2 else ranked.iloc[0:0]
    frame["selected_for_full_validation"] = frame.index.isin(selected.index)
    atomic_parquet(TABLE_PATH, frame.sort_values("history_mix"))
    atomic_csv(CSV_PATH, frame.sort_values("history_mix"))

    base = _base_config()
    frozen: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        config = dict(base)
        config["r2c_v5_history_mix"] = float(row["history_mix"])
        frozen.append(
            {
                "history_mix": float(row["history_mix"]),
                "history_temperature": HISTORY_TEMPERATURE,
                "last50_validation_accuracy": float(row["last50_validation_accuracy"]),
                "recovery_deficit_auc20": float(row["recovery_deficit_auc20"]),
                "target_hit_round": int(row["target_hit_round"]),
                "final_participation_jfi": float(row["final_participation_jfi"]),
                "final_worst10_participation": float(
                    row["final_worst10_participation"]
                ),
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
        "h2_full_validation_authorized": len(frozen) == 2,
        "screen_manifest_hash": config_hash(manifest),
        "candidate_count": len(frame),
        "eligible_candidate_count": len(ranked),
        "selection_rule": manifest["selection_rule"],
        "selected_candidates": frozen,
    }
    atomic_json(SELECTION_PATH, payload)
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
    selection = freeze_selection(manifest)
    authorized = bool(selection["h2_full_validation_authorized"])
    state.update(
        {
            "status": (
                "screen_completed_selection_frozen"
                if authorized
                else "screen_completed_h2_closed"
            ),
            "current_job_id": None,
            "h2_full_validation_authorized": authorized,
            "selected_candidates": [
                {"history_mix": value["history_mix"]}
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
