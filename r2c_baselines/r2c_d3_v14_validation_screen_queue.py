from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
from .data import partition_asset_path, partition_meta_path, prepare_partition
from .logging_io import read_chunked_table
from .r2c_post_event_lower_quartile_accuracy20 import derive_lqa20_percent
from .r2c_v7 import PROTOCOL_VERSION as V7_PROTOCOL_VERSION
from .r2c_v14 import (
    CANDIDATES as CMTR_CANDIDATES,
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


PLAN_ID = "R2C_V14_CMTR_20260819_215527"
PLAN_PATH = PROJECT_ROOT / "refine-logs" / "EXPERIMENT_PLAN_20260819_215527.md"
CONTROL_RUN_ID = "A-R2C-D3-S4-V11FULLVAL-B095-s20260810"
CONTROL_RUN_DIR = RUN_ROOT / CONTROL_RUN_ID

MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v14_validation_screen_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v14_validation_screen_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v14_validation_screen_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v14_validation_screen_result.json"
RUNS_PATH = QUEUE_ROOT / "r2c_d3_v14_validation_screen_runs.parquet"
RUNS_CSV_PATH = QUEUE_ROOT / "r2c_d3_v14_validation_screen_runs.csv"

DATASET_ID = "D3"
SCENARIO_ID = "S4"
SEED = 20260810
ROUNDS = 1000
EVENT_ROUND = 500
TARGET_ACCURACY = 0.7986707616707616
MAX_ATTEMPTS = 3
DECISION_ATOL = 1.0e-12

# Closed before any full-budget v14 output.
CANDIDATES: tuple[str, ...] = (
    "CMTR-B950-R20",
    "CMTR-B975-R10",
    "CMTR-B975-R20",
)
if tuple(CMTR_CANDIDATES) != CANDIDATES:
    raise AssertionError("The registered v14 candidate order drifted")

# Derived from the already completed and independently audited validation
# comparator before any M1 candidate is launched.
CONTROL_METRICS = {
    "s4_last50_accuracy": 0.9047338247338247,
    "s4_lqa20_percent": 89.62940212940212,
    "s4_algorithm_tta_s": 260.9428432000568,
}
CLOSE_MARGINS = {
    "s4_last50_pp": 0.15,
    "s4_lqa20_pp": 0.50,
    "s4_algorithm_tta_percent": 5.0,
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
        "r2c_v14.py",
        "r2c_post_event_lower_quartile_accuracy20.py",
        "run.py",
        "run_v14.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    tests = Path(__file__).resolve().parents[1] / "tests"
    for name in (
        "audit_r2c_run.py",
        "audit_r2c_v14_run.py",
        "test_r2c_v14.py",
        "test_r2c_d3_v14_validation_screen_queue.py",
    ):
        values[f"tests/{name}"] = sha256_file(tests / name)
    return values


def _lineage(paths: tuple[Path, ...]) -> dict[str, str]:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _control_lineage() -> dict[str, str]:
    return _lineage(
        (
            PLAN_PATH,
            CONTROL_RUN_DIR / "job.json",
            CONTROL_RUN_DIR / "result.json",
            CONTROL_RUN_DIR / "_SUCCESS.json",
            CONTROL_RUN_DIR / "run_manifest.parquet",
            CONTROL_RUN_DIR / "tables" / "round_metrics" / "_index.json",
        )
    )


def _source_config() -> dict[str, Any]:
    job = json.loads((CONTROL_RUN_DIR / "job.json").read_text(encoding="utf-8"))
    if (
        job.get("dataset_id") != DATASET_ID
        or job.get("scenario_id") != SCENARIO_ID
        or job.get("evaluation_split") != "validation"
        or int(job.get("rounds", -1)) != ROUNDS
        or any(
            int(job.get(key, -1)) != SEED
            for key in ("seed", "partition_seed", "trace_seed")
        )
        or bool(job.get("formal_test_access", False))
    ):
        raise RuntimeError("v14 validation comparator scope drift")
    config = dict(job["method_config"])
    if (
        config.get("r2c_protocol_version") != V7_PROTOCOL_VERSION
        or config.get("r2c_v4_deployment_ema_betas") != [0.95]
        or float(config.get("r2c_v4_primary_deployment_beta", -1.0)) != 0.95
    ):
        raise RuntimeError("v14 validation comparator configuration drift")
    return config


def _candidate_config(candidate_id: str) -> dict[str, Any]:
    if candidate_id not in CANDIDATES:
        raise ValueError(f"Unregistered CMTR candidate: {candidate_id}")
    stable_beta = candidate_stable_beta(candidate_id)
    config = _source_config()
    config.pop("r2c_v7_trigger_deployment_beta", None)
    config.update(
        {
            "r2c_protocol_version": PROTOCOL_VERSION,
            "r2c_v4_deployment_ema_betas": [FAST_BETA, stable_beta],
            "r2c_v4_primary_deployment_beta": stable_beta,
            "r2c_v14_candidate_id": candidate_id,
            "r2c_v14_fast_beta": FAST_BETA,
            "r2c_v14_warmup_rounds": WARMUP_ROUNDS,
            "r2c_v14_plan_id": PLAN_ID,
        }
    )
    return config


def _ensure_assets() -> dict[str, str]:
    prepare_partition(DATASET_ID, SEED)
    prepare_trace(DATASET_ID, SCENARIO_ID, SEED, rounds=ROUNDS)
    paths = (
        partition_asset_path(DATASET_ID, SEED),
        partition_meta_path(DATASET_ID, SEED),
        trace_asset_path(DATASET_ID, SCENARIO_ID, SEED, ROUNDS),
        trace_meta_path(DATASET_ID, SCENARIO_ID, SEED, ROUNDS),
    )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _time_to_accuracy(rounds: pd.DataFrame) -> tuple[int | None, float | None]:
    reached = rounds.loc[
        rounds["test_accuracy"].astype(float) >= TARGET_ACCURACY
    ].sort_values("round")
    if reached.empty:
        return None, None
    first = reached.iloc[0]
    return int(first["round"]), float(first["algorithm_elapsed_s"])


def _derive_control_metrics() -> dict[str, float]:
    rounds = read_chunked_table(CONTROL_RUN_DIR, "round_metrics").sort_values("round")
    if (
        len(rounds) != ROUNDS
        or rounds["round"].astype(int).tolist() != list(range(1, ROUNDS + 1))
        or rounds.loc[
            rounds["event_offset_round"].astype(int) == 0, "round"
        ].astype(int).tolist()
        != [EVENT_ROUND]
    ):
        raise RuntimeError("validation comparator round/event contract drift")
    _, tta_s = _time_to_accuracy(rounds)
    if tta_s is None:
        raise RuntimeError("validation comparator no longer reaches the frozen target")
    return {
        "s4_last50_accuracy": float(
            rounds.tail(50)["test_accuracy"].astype(float).mean()
        ),
        "s4_lqa20_percent": float(derive_lqa20_percent(rounds)),
        "s4_algorithm_tta_s": float(tta_s),
    }


def _assert_control_metrics() -> None:
    derived = _derive_control_metrics()
    for key, expected in CONTROL_METRICS.items():
        if not np.isclose(float(derived[key]), float(expected), rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"validation comparator metric drift: {key}")


def _job(candidate_id: str) -> dict[str, Any]:
    stable_beta = candidate_stable_beta(candidate_id)
    recovery_rounds = candidate_recovery_rounds(candidate_id)
    run_id = f"A-R2C-D3-S4-V14SCREEN-{candidate_id}-s{SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v14_validation_cmtr_screen",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": DATASET_ID,
        "scenario_id": SCENARIO_ID,
        "rounds": ROUNDS,
        "method_config": _candidate_config(candidate_id),
        "block_id": "A-R2C-D3-V14-CMTR-VALIDATION-SCREEN",
        "seed": SEED,
        "partition_seed": SEED,
        "trace_seed": SEED,
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "candidate_id": candidate_id,
        "fast_beta": FAST_BETA,
        "stable_beta": stable_beta,
        "warmup_rounds": WARMUP_ROUNDS,
        "recovery_rounds": recovery_rounds,
        "deployment_state_count": 2,
        "formal_test_access": False,
        "other_dataset_access": False,
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
        "target_accuracy",
        "formal_test_access",
        "other_dataset_access",
        "test_labels_used_for_selection",
        "performance_sealed_until_terminal",
        "fast_beta",
        "warmup_rounds",
        "candidate_order",
        "control_run_id",
        "control_metrics",
        "close_margins",
        "decision_numerical_atol",
        "selection_rule",
        "normalization_rule",
        "completion_rule",
        "control_lineage",
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
            raise RuntimeError("Refusing to rebuild a started v14 validation manifest")
    _assert_control_metrics()
    jobs = [_job(candidate_id) for candidate_id in CANDIDATES]
    if len(jobs) != 3 or len({job["job_id"] for job in jobs}) != 3:
        raise AssertionError("v14 M1 must contain exactly three unique candidates")
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_validation_only_v14_CMTR_screen",
        "plan_id": PLAN_ID,
        "protocol_version": PROTOCOL_VERSION,
        "selection_split": "validation",
        "seed": SEED,
        "rounds_per_job": ROUNDS,
        "event_round": EVENT_ROUND,
        "target_accuracy": TARGET_ACCURACY,
        "formal_test_access": False,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "performance_sealed_until_terminal": True,
        "fast_beta": FAST_BETA,
        "warmup_rounds": WARMUP_ROUNDS,
        "candidate_order": list(CANDIDATES),
        "control_run_id": CONTROL_RUN_ID,
        "control_metrics": CONTROL_METRICS,
        "close_margins": CLOSE_MARGINS,
        "decision_numerical_atol": DECISION_ATOL,
        "selection_rule": (
            "eligible iff all three S4 metrics are strict wins, or exactly two are "
            "strict wins and the sole miss is within its registered close margin; "
            "equality is not a strict win; exact 1000/1000 global learning-model "
            "hash identity to the comparator is mandatory"
        ),
        "normalization_rule": (
            "last50_gain_pp/0.15; LQA20_gain_pp/0.50; "
            "TTA_relative_improvement_percent/5.0"
        ),
        "completion_rule": (
            "all three full-budget validation candidates complete and independently "
            "audit before any candidate performance read"
        ),
        "control_lineage": _control_lineage(),
        "asset_hashes": _ensure_assets(),
        "implementation_hashes": _implementation_hashes(),
        "job_order": [job["job_id"] for job in jobs],
        "max_attempts": MAX_ATTEMPTS,
        "jobs": jobs,
    }
    manifest["frozen_spec_hash"] = config_hash(_frozen_spec(manifest))
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        immutable_path = (
            QUEUE_ROOT / f"r2c_d3_v14_validation_screen_manifest_{stamp}.json"
        )
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
    if not EVENTS_PATH.exists():
        return []
    return pd.read_parquet(EVENTS_PATH).to_dict("records")


def _event(
    events: list[dict[str, Any]],
    job: dict[str, Any],
    event_type: str,
    **extra: Any,
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
    value.update(
        {"run_id": run_id, "retry_of_run_id": retry_of, "queue_utc": utc_now()}
    )
    return value


def _assert_frozen_manifest(manifest: dict[str, Any]) -> None:
    if (
        manifest.get("scope") != "D3_validation_only_v14_CMTR_screen"
        or manifest.get("plan_id") != PLAN_ID
        or manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("selection_split") != "validation"
        or int(manifest.get("seed", -1)) != SEED
        or int(manifest.get("rounds_per_job", -1)) != ROUNDS
        or bool(manifest.get("formal_test_access"))
        or bool(manifest.get("other_dataset_access"))
        or bool(manifest.get("test_labels_used_for_selection"))
        or not bool(manifest.get("performance_sealed_until_terminal"))
        or float(manifest.get("fast_beta", -1.0)) != FAST_BETA
        or int(manifest.get("warmup_rounds", -1)) != WARMUP_ROUNDS
    ):
        raise RuntimeError("v14 validation manifest scope drift")
    if (
        manifest.get("candidate_order") != list(CANDIDATES)
        or manifest.get("control_metrics") != CONTROL_METRICS
        or manifest.get("close_margins") != CLOSE_MARGINS
        or manifest.get("control_lineage") != _control_lineage()
        or manifest.get("implementation_hashes") != _implementation_hashes()
    ):
        raise RuntimeError("v14 validation manifest evidence/configuration drift")
    if manifest.get("frozen_spec_hash") != config_hash(_frozen_spec(manifest)):
        raise RuntimeError("v14 validation frozen specification drift")
    jobs = list(manifest.get("jobs", []))
    if len(jobs) != 3 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("v14 validation candidate order drift")
    for job, candidate_id in zip(jobs, CANDIDATES):
        stable_beta = candidate_stable_beta(candidate_id)
        config = dict(job["method_config"])
        if (
            job.get("candidate_id") != candidate_id
            or float(job.get("fast_beta", -1.0)) != FAST_BETA
            or float(job.get("stable_beta", -1.0)) != stable_beta
            or int(job.get("warmup_rounds", -1)) != WARMUP_ROUNDS
            or int(job.get("recovery_rounds", -1))
            != candidate_recovery_rounds(candidate_id)
            or int(job.get("deployment_state_count", -1)) != 2
            or job.get("dataset_id") != DATASET_ID
            or job.get("scenario_id") != SCENARIO_ID
            or job.get("evaluation_split") != "validation"
            or int(job.get("rounds", -1)) != ROUNDS
            or any(
                int(job.get(key, -1)) != SEED
                for key in ("seed", "partition_seed", "trace_seed")
            )
            or config.get("r2c_protocol_version") != PROTOCOL_VERSION
            or config.get("r2c_v4_deployment_ema_betas") != [FAST_BETA, stable_beta]
            or float(config.get("r2c_v4_primary_deployment_beta", -1.0))
            != stable_beta
            or config.get("r2c_v14_candidate_id") != candidate_id
            or float(config.get("r2c_v14_fast_beta", -1.0)) != FAST_BETA
            or int(config.get("r2c_v14_warmup_rounds", -1)) != WARMUP_ROUNDS
            or "r2c_v7_trigger_deployment_beta" in config
        ):
            raise RuntimeError(f"v14 validation candidate drift: {job['job_id']}")


def _assert_immutable_authority(state: dict[str, Any], manifest: dict[str, Any]) -> None:
    immutable_path = Path(str(state["immutable_manifest_path"]))
    if (
        not immutable_path.is_file()
        or sha256_file(immutable_path) != state.get("immutable_manifest_sha256")
    ):
        raise RuntimeError("v14 immutable manifest hash mismatch")
    immutable = json.loads(immutable_path.read_text(encoding="utf-8"))
    if _frozen_spec(immutable) != _frozen_spec(manifest):
        raise RuntimeError("v14 active/immutable frozen specifications diverged")


def _audit_run(run_id: str, log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        return subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parents[1]
                    / "tests"
                    / "audit_r2c_v14_run.py"
                ),
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
                QUEUE_ROOT / "worker_logs" / f"{path.name}.reconcile.v14.audit.log"
            )
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
            _event(
                events,
                job,
                "reconciled_completed",
                reason="existing_success_output_and_independent_v14_audit",
            )
            changed += 1
    return changed


def _selection_fields(
    last50: float, lqa20: float, tta_s: float | None, *, identity: bool = True
) -> dict[str, Any]:
    last50_gain_pp = 100.0 * (
        float(last50) - CONTROL_METRICS["s4_last50_accuracy"]
    )
    lqa_gain_pp = float(lqa20) - CONTROL_METRICS["s4_lqa20_percent"]
    tta_gain_percent = (
        float("-inf")
        if tta_s is None
        else 100.0
        * (CONTROL_METRICS["s4_algorithm_tta_s"] - float(tta_s))
        / CONTROL_METRICS["s4_algorithm_tta_s"]
    )
    gains = {
        "last50": last50_gain_pp,
        "lqa20": lqa_gain_pp,
        "tta": tta_gain_percent,
    }
    margins = {
        "last50": CLOSE_MARGINS["s4_last50_pp"],
        "lqa20": CLOSE_MARGINS["s4_lqa20_pp"],
        "tta": CLOSE_MARGINS["s4_algorithm_tta_percent"],
    }
    strict = {key: value > DECISION_ATOL for key, value in gains.items()}
    strict_wins = sum(strict.values())
    misses = [key for key, won in strict.items() if not won]
    sole_close_miss = bool(
        strict_wins == 2
        and len(misses) == 1
        and gains[misses[0]] >= -margins[misses[0]] - DECISION_ATOL
    )
    eligible = bool(identity and (strict_wins == 3 or sole_close_miss))
    normalized = {key: gains[key] / margins[key] for key in gains}
    return {
        "last50_gain_pp": float(last50_gain_pp),
        "lqa20_gain_pp": float(lqa_gain_pp),
        "tta_gain_percent": float(tta_gain_percent),
        "normalized_last50_gain": float(normalized["last50"]),
        "normalized_lqa20_gain": float(normalized["lqa20"]),
        "normalized_tta_gain": float(normalized["tta"]),
        "maximin_normalized_gain": float(min(normalized.values())),
        "strict_last50_win": bool(strict["last50"]),
        "strict_lqa20_win": bool(strict["lqa20"]),
        "strict_tta_win": bool(strict["tta"]),
        "strict_win_count": int(strict_wins),
        "sole_close_miss": sole_close_miss,
        "global_learning_hash_identity": bool(identity),
        "validation_eligible": eligible,
    }


def _run_metrics(job: dict[str, Any]) -> dict[str, Any]:
    run_id = str(job["actual_run_id"])
    run_dir = RUN_ROOT / run_id
    actual_job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    run_manifest = pd.read_parquet(run_dir / "run_manifest.parquet").iloc[0]
    rounds = read_chunked_table(run_dir, "round_metrics").sort_values("round")
    control_rounds = read_chunked_table(CONTROL_RUN_DIR, "round_metrics").sort_values(
        "round"
    )
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
        raise RuntimeError(f"v14 validation run contract mismatch: {run_id}")
    event = rounds.loc[
        rounds["event_offset_round"].astype(int) == 0, "round"
    ].astype(int).tolist()
    if event != [EVENT_ROUND]:
        raise RuntimeError(f"v14 validation event mismatch: {run_id}")
    candidate_id = str(job["candidate_id"])
    stable_beta = candidate_stable_beta(candidate_id)
    recovery_rounds = candidate_recovery_rounds(candidate_id)
    trigger_rounds = rounds.loc[
        rounds["telemetry_shift_trigger"].astype(bool), "round"
    ].astype(int).tolist()
    hold_rounds = rounds.loc[
        rounds["deployment_quarantine_applied"].astype(bool), "round"
    ].astype(int).tolist()
    recovery_actual = rounds.loc[
        rounds["deployment_cmtr_recovery_applied"].astype(bool), "round"
    ].astype(int).tolist()
    warmup_actual = rounds.loc[
        rounds["deployment_cmtr_warmup_applied"].astype(bool), "round"
    ].astype(int).tolist()
    expected_recovery = list(
        range(EVENT_ROUND + 1, EVENT_ROUND + recovery_rounds + 1)
    )
    selected_beta = rounds["selected_deployment_beta"].astype(float)
    expected_fast_rounds = set(range(1, WARMUP_ROUNDS + 1)) | set(expected_recovery)
    actual_fast_rounds = set(
        rounds.loc[
            np.isclose(selected_beta, FAST_BETA, rtol=0.0, atol=0.0), "round"
        ].astype(int)
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
        np.column_stack(
            [rounds[column].astype(bool).to_numpy() for column in forbidden_columns]
        ).any()
    )
    identity = bool(
        len(control_rounds) == len(rounds)
        and rounds["global_model_hash"].astype(str).tolist()
        == control_rounds["global_model_hash"].astype(str).tolist()
    )
    if (
        trigger_rounds != [EVENT_ROUND]
        or hold_rounds != [EVENT_ROUND]
        or recovery_actual != expected_recovery
        or warmup_actual != list(range(1, WARMUP_ROUNDS + 1))
        or actual_fast_rounds != expected_fast_rounds
        or not rounds["deployment_cmtr_state_server_only"].astype(bool).all()
        or forbidden_any
        or not identity
        or not np.isclose(
            rounds.loc[rounds["round"] == EVENT_ROUND, "selected_deployment_beta"].iloc[0],
            stable_beta,
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise RuntimeError(f"v14 validation CMTR/global lineage mismatch: {run_id}")
    tta_round, tta_s = _time_to_accuracy(rounds)
    last50 = float(rounds.tail(50)["test_accuracy"].astype(float).mean())
    lqa20 = float(derive_lqa20_percent(rounds))
    decision = _selection_fields(last50, lqa20, tta_s, identity=identity)
    return {
        "candidate_id": candidate_id,
        "run_id": run_id,
        "fast_beta": FAST_BETA,
        "stable_beta": stable_beta,
        "warmup_rounds": WARMUP_ROUNDS,
        "recovery_rounds": recovery_rounds,
        "deployment_state_count": 2,
        "last50_validation_accuracy": last50,
        "lqa20_percent": lqa20,
        "tta_round": tta_round,
        "algorithm_tta_s": tta_s,
        **decision,
        "trigger_count": len(trigger_rounds),
        "hold_count": len(hold_rounds),
        "recovery_count": len(recovery_actual),
        "warmup_count": len(warmup_actual),
        "forbidden_inputs_any": forbidden_any,
        "test_labels_used": False,
    }


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot select from an incomplete v14 validation screen")
    rows = [_run_metrics(job) for job in manifest["jobs"]]
    frame = pd.DataFrame(rows)
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    eligible = frame.loc[frame["validation_eligible"].astype(bool)].copy()
    selected: dict[str, Any] | None = None
    eligible_ids: list[str] = []
    if not eligible.empty:
        eligible["candidate_order"] = eligible["candidate_id"].map(
            {candidate_id: index for index, candidate_id in enumerate(CANDIDATES)}
        )
        eligible = eligible.sort_values(
            ["maximin_normalized_gain", "recovery_rounds", "candidate_order"],
            ascending=[False, True, True],
            kind="mergesort",
        )
        eligible_ids = eligible["candidate_id"].astype(str).tolist()
        selected = eligible.iloc[0].drop(labels=["candidate_order"]).to_dict()
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "status": "selected" if selected is not None else "no_candidate_eligible",
        "selection_split": "validation",
        "seed": SEED,
        "formal_test_access": False,
        "other_dataset_access": False,
        "test_labels_used_for_selection": False,
        "control_run_id": CONTROL_RUN_ID,
        "control_metrics": CONTROL_METRICS,
        "close_margins": CLOSE_MARGINS,
        "selection_rule": manifest["selection_rule"],
        "normalization_rule": manifest["normalization_rule"],
        "selected": selected,
        "eligible_candidate_ids_ranked": eligible_ids,
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
                [
                    sys.executable,
                    "-m",
                    "r2c_baselines.run_v14",
                    "--job",
                    str(job_file),
                ],
                cwd=str(Path(__file__).resolve().parents[1]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        success = (RUN_ROOT / str(resolved["run_id"]) / "_SUCCESS.json").exists()
        audit_log = (
            QUEUE_ROOT / "worker_logs" / f"{resolved['run_id']}.v14.audit.log"
        )
        audit = _audit_run(str(resolved["run_id"]), audit_log) if success else None
        if (
            process.returncode == 0
            and success
            and audit is not None
            and audit.returncode == 0
        ):
            job["status"] = "completed"
            _event(
                events,
                job,
                "completed",
                exit_code=0,
                audit_log_path=str(audit_log),
            )
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
                "m1_completed_selected_m2_required"
                if result["status"] == "selected"
                else "m1_completed_no_candidate_eligible_stop"
            ),
            "current_job_id": None,
            "all_runs_completed": True,
            "performance_sealed_until_terminal": False,
            "result_path": str(RESULT_PATH),
            "selected_candidate_id": (
                None
                if result["selected"] is None
                else result["selected"]["candidate_id"]
            ),
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
