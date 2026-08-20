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


ROUNDS = 1000
TARGET_ACCURACY = 0.7986707616707616
SCENARIOS = ("S0", "S4")
ACCURACY_CLOSE = 0.0015
AUC_CLOSE = 0.0001
TTA_CLOSE_MULTIPLIER = 1.05

BASELINE_ENVELOPE = {
    "s0_last50_accuracy": 0.8947065247065247,
    "s4_last50_accuracy": 0.8933715533715534,
    "s4_recovery_deficit_auc20": 0.00025832650832653957,
    "s4_algorithm_tta_s": 251.75663530116435,
}
H1_DIVERSITY_THRESHOLDS = {
    "S0": {
        "final_participation_jfi": 0.831826341278384,
        "final_worst10_participation": 45.0,
    },
    "S4": {
        "final_participation_jfi": 0.871397425194888,
        "final_worst10_participation": 47.9,
    },
}

EXPECTED_SCREEN_MANIFEST_SHA256 = (
    "2CB5822DC19A22691BAA23BAFBA1EAE6D9F5F04CD1DE456F42A8BD626DB459F4"
)
EXPECTED_BASELINE_RESULT_SHA256 = (
    "9EAC04D44D3BE00F0EFA81484EABA3F7485CC2551B59384E1E39CE9E8F39FC59"
)
EXPECTED_PLAN_SHA256 = (
    "560E0BC89F9A5FB21D2F54D894CCCE14A83FF3662735F9724200A3CF7CE6709D"
)
EXPECTED_H1_INDEX_SHA256 = {
    "S0": "5A6002C40F70467EA2889AFB739BBCC05F3ADF11C04E29A52681884D7C4D1D0A",
    "S4": "734032FC375655D8C1DE5C582FA8A56AA09EE9CCE2ED0B0CA9E70779882C65E3",
}

SCREEN_SELECTION_PATH = QUEUE_ROOT / "r2c_d3_v6_diversity_screen_selection.json"
SCREEN_MANIFEST_PATH = (
    QUEUE_ROOT
    / "r2c_d3_v6_diversity_screen_manifest_20260817T025535.341110Z.json"
)
BASELINE_RESULT_PATH = QUEUE_ROOT / "r2c_d3_v5_fullval_result.json"
PLAN_PATH = (
    PROJECT_ROOT
    / "refine-logs"
    / "D3_V6_DIVERSITY_PLAN_AMENDMENT_20260817_103843.md"
)
H1_RUN_IDS = {
    "S0": "A-R2C-D3-S0-V5FULLVAL-C1-A1000-B0900-s20260810",
    "S4": "A-R2C-D3-S4-V5FULLVAL-C1-A1000-B0900-s20260810",
}
H1_INDEX_PATHS = {
    scenario: RUN_ROOT / run_id / "tables" / "round_metrics" / "_index.json"
    for scenario, run_id in H1_RUN_IDS.items()
}

MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v6_fullval_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_d3_v6_fullval_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_d3_v6_fullval_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_d3_v6_fullval_result.json"
RUNS_PATH = PLOT_ROOT / "r2c_d3_v6_fullval_runs.parquet"
RUNS_CSV_PATH = PLOT_ROOT / "r2c_d3_v6_fullval_runs.csv"
COMPARISON_PATH = PLOT_ROOT / "r2c_d3_v6_fullval_comparison.json"
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
    return (
        SCREEN_SELECTION_PATH,
        SCREEN_MANIFEST_PATH,
        BASELINE_RESULT_PATH,
        PLAN_PATH,
        H1_INDEX_PATHS["S0"],
        H1_INDEX_PATHS["S4"],
    )


def _source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in _source_paths()
    }


def _verify_static_sources() -> None:
    expected = {
        SCREEN_MANIFEST_PATH: EXPECTED_SCREEN_MANIFEST_SHA256,
        BASELINE_RESULT_PATH: EXPECTED_BASELINE_RESULT_SHA256,
        PLAN_PATH: EXPECTED_PLAN_SHA256,
        H1_INDEX_PATHS["S0"]: EXPECTED_H1_INDEX_SHA256["S0"],
        H1_INDEX_PATHS["S4"]: EXPECTED_H1_INDEX_SHA256["S4"],
    }
    for path, value in expected.items():
        if not path.exists() or sha256_file(path) != value.lower():
            raise RuntimeError(f"Frozen Phase D source hash drift: {path}")


def _verified_chunked_table(run_dir: Path, table_name: str) -> pd.DataFrame:
    root = run_dir / "tables" / table_name
    index_path = root / "_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for part in index.get("parts", []):
        path = root / str(part["path"])
        if path.stat().st_size != int(part["bytes"]):
            raise RuntimeError(f"Chunk byte-size drift: {path}")
        if sha256_file(path) != str(part["sha256"]):
            raise RuntimeError(f"Chunk SHA-256 drift: {path}")
    frame = read_chunked_table(run_dir, table_name)
    if len(frame) != int(index["rows"]):
        raise RuntimeError(f"Chunk row-count drift: {root}")
    return frame


def _load_h1_diversity_thresholds() -> dict[str, dict[str, float]]:
    _verify_static_sources()
    observed: dict[str, dict[str, float]] = {}
    for scenario, run_id in H1_RUN_IDS.items():
        frame = _verified_chunked_table(RUN_ROOT / run_id, "round_metrics").sort_values(
            "round"
        )
        if frame["round"].astype(int).tolist() != list(range(1, ROUNDS + 1)):
            raise RuntimeError(f"Frozen H1 diversity source is incomplete: {run_id}")
        last = frame.iloc[-1]
        observed[scenario] = {
            "final_participation_jfi": float(last["participation_jfi"]),
            "final_worst10_participation": float(last["worst10_participation"]),
        }
        for name, expected in H1_DIVERSITY_THRESHOLDS[scenario].items():
            if abs(observed[scenario][name] - expected) > 1.0e-15:
                raise RuntimeError(f"Frozen H1 {scenario} {name} drift")
    return observed


def _load_baseline_envelope() -> dict[str, float]:
    _verify_static_sources()
    payload = json.loads(BASELINE_RESULT_PATH.read_text(encoding="utf-8"))
    if payload.get("selection_split") != "validation" or payload.get(
        "test_labels_used"
    ):
        raise RuntimeError("Frozen baseline envelope was not validation-only")
    observed = {name: float(value) for name, value in payload["baseline_envelope"].items()}
    if observed != BASELINE_ENVELOPE:
        raise RuntimeError("Frozen D3 validation baseline envelope drift")
    return observed


def _load_screen_selection() -> dict[str, Any]:
    _verify_static_sources()
    if not SCREEN_SELECTION_PATH.exists():
        raise RuntimeError("Phase D requires a frozen Phase C selection")
    selection = json.loads(SCREEN_SELECTION_PATH.read_text(encoding="utf-8"))
    if (
        selection.get("selection_split") != "validation"
        or selection.get("test_labels_used")
        or selection.get("formal_test_authorized")
        or not selection.get("h2_full_validation_authorized")
    ):
        raise RuntimeError("Phase C did not authorize validation-only Phase D")
    if selection.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Phase C protocol drift")
    candidates = selection.get("selected_candidates", [])
    if len(candidates) != 2:
        raise RuntimeError(f"Phase C must freeze exactly two candidates, found {len(candidates)}")
    for candidate in candidates:
        config = candidate["method_config"]
        if config.get("r2c_protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError("Phase C selected-candidate protocol drift")
        if config_hash(config) != candidate.get("config_hash"):
            raise RuntimeError("Phase C selected-candidate config hash drift")
        if float(config.get("r2c_v3_fixed_server_alpha")) != 1.0:
            raise RuntimeError("Phase C selected-candidate alpha drift")
        if config.get("r2c_v4_deployment_ema_betas") != [0.9]:
            raise RuntimeError("Phase C selected-candidate beta drift")
        if float(config.get("r2c_v4_primary_deployment_beta")) != 0.9:
            raise RuntimeError("Phase C selected-candidate primary beta drift")
        if float(config.get("r2c_v5_history_temperature")) != 1.0:
            raise RuntimeError("Phase C selected-candidate history temperature drift")
        if float(config.get("r2c_v5_history_mix")) not in (0.15, 0.30, 0.45, 0.60):
            raise RuntimeError("Phase C selected-candidate history mix drift")
    return selection


def _r2c_job(position: int, candidate: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    history_mix = float(candidate["history_mix"])
    label = f"C{position}-R{int(candidate['overall_rank'])}-L{int(round(history_mix * 1000)):04d}"
    run_id = f"A-R2C-D3-{scenario_id}-V6DIV-FULLVAL-{label}-s{DEV_SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "d3_v6_diversity_matched_full_validation",
        "mode": "calibration",
        "method_id": "R2C-FL",
        "dataset_id": "D3",
        "scenario_id": scenario_id,
        "rounds": ROUNDS,
        "method_config": dict(candidate["method_config"]),
        "block_id": "A-R2C-D3-V6-DIVERSITY-FULLVAL",
        "seed": DEV_SEED,
        "partition_seed": DEV_SEED,
        "trace_seed": DEV_SEED,
        "evaluation_split": "validation",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": TARGET_ACCURACY,
        "variant_label": label,
        "candidate_position": position,
        "screen_overall_rank": int(candidate["overall_rank"]),
        "history_mix": history_mix,
        "test_labels_used_for_selection": False,
        "status": "pending",
        "attempts": 0,
        "actual_run_id": None,
        "failure_reason": None,
    }


def _build_jobs(selection: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for position, candidate in enumerate(selection["selected_candidates"], start=1):
        jobs.extend(_r2c_job(position, candidate, scenario) for scenario in SCENARIOS)
    return jobs


def build_manifest(persist: bool = True, force: bool = False) -> dict[str, Any]:
    if persist and MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if force and MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if any(
            job.get("status") != "pending" or int(job.get("attempts", 0))
            for job in existing["jobs"]
        ):
            raise RuntimeError("Refusing to rebuild a started D3 v6 full-validation manifest")
    if int(DEV_SEED) != 20260810:
        raise RuntimeError(f"Expected development seed 20260810, found {DEV_SEED}")
    selection = _load_screen_selection()
    envelope = _load_baseline_envelope()
    diversity = _load_h1_diversity_thresholds()
    jobs = _build_jobs(selection)
    if len(jobs) != 4 or len({job["job_id"] for job in jobs}) != 4:
        raise RuntimeError("D3 v6 full validation must contain exactly four unique jobs")
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D3_only_v6_phase_d_matched_full_validation",
        "protocol_version": PROTOCOL_VERSION,
        "formal_test_access": False,
        "test_labels_used_for_selection": False,
        "dev_seed": DEV_SEED,
        "rounds_per_job": ROUNDS,
        "scenario_order": list(SCENARIOS),
        "job_order": [job["job_id"] for job in jobs],
        "screen_selection_hash": config_hash(selection),
        "reused_baseline_envelope": envelope,
        "h1_diversity_thresholds": diversity,
        "close_limits": {
            "accuracy_fraction": ACCURACY_CLOSE,
            "auc_fraction": AUC_CLOSE,
            "tta_multiplier": TTA_CLOSE_MULTIPLIER,
        },
        "formal_authorization_rule": (
            "four strict metric wins, or exactly three strict wins and one close miss; "
            "and strict S0/S4 JFI plus worst10 improvements over frozen H1 controls"
        ),
        "winner_rule": (
            "strict_passes(desc), normalized_margin_score(desc), phase_c_overall_rank(asc)"
        ),
        "resource_gate": "two GPU utilization samples below 10 percent at least 30 seconds apart",
        "implementation_hashes": _implementation_hashes(),
        "source_hashes": _source_hashes(),
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_d3_v6_fullval_manifest_{stamp}.json", manifest)
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
        raise RuntimeError("D3 v6 full-validation protocol drift")
    if manifest.get("formal_test_access") or manifest.get("test_labels_used_for_selection"):
        raise RuntimeError("D3 v6 full validation attempted formal test access")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("Implementation hashes changed after D3 v6 full-validation freeze")
    if manifest.get("source_hashes") != _source_hashes():
        raise RuntimeError("Source hashes changed after D3 v6 full-validation freeze")
    jobs = manifest.get("jobs", [])
    if len(jobs) != 4 or [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("D3 v6 full-validation job matrix drift")
    for job in jobs:
        if any(int(job[key]) != DEV_SEED for key in ("seed", "partition_seed", "trace_seed")):
            raise RuntimeError(f"Development seed drift in {job['job_id']}")
        if (
            job.get("dataset_id") != "D3"
            or job.get("scenario_id") not in SCENARIOS
            or job.get("evaluation_split") != "validation"
            or int(job.get("rounds", -1)) != ROUNDS
            or not bool(job.get("full_logging"))
        ):
            raise RuntimeError(f"Validation protocol drift in {job['job_id']}")
        config = job["method_config"]
        if config.get("r2c_protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"Candidate protocol drift in {job['job_id']}")


def _time_to_accuracy(rounds: pd.DataFrame, target: float) -> tuple[int | None, float | None]:
    reached = rounds.loc[rounds["test_accuracy"].astype(float) >= float(target)]
    if reached.empty:
        return None, None
    first = reached.sort_values("round").iloc[0]
    return int(first["round"]), float(first["algorithm_elapsed_s"])


def _evaluate_metric_rule(
    observed: dict[str, float | None], envelope: dict[str, float]
) -> dict[str, Any]:
    tta = observed["s4_algorithm_tta_s"]
    checks = {
        "s0_last50_accuracy": float(observed["s0_last50_accuracy"])
        > envelope["s0_last50_accuracy"],
        "s4_last50_accuracy": float(observed["s4_last50_accuracy"])
        > envelope["s4_last50_accuracy"],
        "s4_recovery_deficit_auc20": float(observed["s4_recovery_deficit_auc20"])
        < envelope["s4_recovery_deficit_auc20"],
        "s4_algorithm_tta_s": tta is not None
        and float(tta) < envelope["s4_algorithm_tta_s"],
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
    metric_authorized = not misses or (len(misses) == 1 and close[misses[0]])
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
        + (
            envelope["s4_recovery_deficit_auc20"]
            - float(observed["s4_recovery_deficit_auc20"])
        )
        / AUC_CLOSE
        + tta_margin
    )
    return {
        "strict_checks": checks,
        "close_checks": close,
        "strict_passes": sum(checks.values()),
        "sole_miss": misses[0] if len(misses) == 1 else None,
        "metric_authorized": bool(metric_authorized),
        "normalized_margin_score": float(margin_score),
    }


def _diversity_checks(observed: dict[str, dict[str, float]]) -> dict[str, bool]:
    return {
        f"{scenario.lower()}_{metric}": float(observed[scenario][metric])
        > H1_DIVERSITY_THRESHOLDS[scenario][metric]
        for scenario in SCENARIOS
        for metric in ("final_participation_jfi", "final_worst10_participation")
    }


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot freeze incomplete D3 v6 full validation")
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run_id = str(job["actual_run_id"])
        run_dir = RUN_ROOT / run_id
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        rounds = _verified_chunked_table(run_dir, "round_metrics").sort_values("round")
        if rounds["round"].astype(int).tolist() != list(range(1, ROUNDS + 1)):
            raise RuntimeError(f"Incomplete D3 v6 full-validation trajectory: {run_id}")
        auc: float | None = None
        if job["scenario_id"] == "S4":
            events = rounds.loc[rounds["event_offset_round"].astype(int) == 0]
            if len(events) != 1 or int(events.iloc[0]["round"]) != 500:
                raise RuntimeError(f"D3 v6 S4 event drift: {run_id}")
            direct = recovery_auc20(
                rounds["round"].astype(int).tolist(),
                rounds["test_accuracy"].astype(float).tolist(),
                500,
            )
            if not direct["recovery_auc20_complete"]:
                raise RuntimeError(f"D3 v6 S4 lacks strict AUC@20: {run_id}")
            auc = float(direct["recovery_deficit_auc20"])
            stored = result["recovery"]["recovery_deficit_auc20"]
            if stored is None or abs(float(stored) - auc) > 1.0e-15:
                raise RuntimeError(f"D3 v6 S4 stored/direct AUC mismatch: {run_id}")
        tta_round, tta_s = _time_to_accuracy(rounds, TARGET_ACCURACY)
        last = rounds.iloc[-1]
        rows.append(
            {
                "run_id": run_id,
                "candidate_position": int(job["candidate_position"]),
                "screen_overall_rank": int(job["screen_overall_rank"]),
                "history_mix": float(job["history_mix"]),
                "scenario_id": job["scenario_id"],
                "last50_validation_accuracy": float(result["last50_accuracy"]),
                "recovery_deficit_auc20": auc,
                "algorithm_elapsed_s": float(result["algorithm_elapsed_s"]),
                "tta_round": tta_round,
                "algorithm_tta_s": tta_s,
                "final_participation_jfi": float(last["participation_jfi"]),
                "final_worst10_participation": float(last["worst10_participation"]),
                "source_kind": "CALIBRATION",
                "test_labels_used": False,
            }
        )
    frame = pd.DataFrame(rows)
    atomic_parquet(RUNS_PATH, frame)
    atomic_csv(RUNS_CSV_PATH, frame)
    envelope = _load_baseline_envelope()
    selection = _load_screen_selection()
    evaluated: list[dict[str, Any]] = []
    for position, candidate in enumerate(selection["selected_candidates"], start=1):
        subset = frame.loc[frame["candidate_position"].astype(int) == position]
        if set(subset["scenario_id"]) != set(SCENARIOS):
            raise RuntimeError(f"Incomplete Phase D pair for candidate {position}")
        s0 = subset.loc[subset["scenario_id"] == "S0"].iloc[0]
        s4 = subset.loc[subset["scenario_id"] == "S4"].iloc[0]
        observed: dict[str, float | None] = {
            "s0_last50_accuracy": float(s0["last50_validation_accuracy"]),
            "s4_last50_accuracy": float(s4["last50_validation_accuracy"]),
            "s4_recovery_deficit_auc20": float(s4["recovery_deficit_auc20"]),
            "s4_algorithm_tta_s": (
                None if pd.isna(s4["algorithm_tta_s"]) else float(s4["algorithm_tta_s"])
            ),
        }
        diversity_observed = {
            "S0": {
                "final_participation_jfi": float(s0["final_participation_jfi"]),
                "final_worst10_participation": float(
                    s0["final_worst10_participation"]
                ),
            },
            "S4": {
                "final_participation_jfi": float(s4["final_participation_jfi"]),
                "final_worst10_participation": float(
                    s4["final_worst10_participation"]
                ),
            },
        }
        metric = _evaluate_metric_rule(observed, envelope)
        diversity = _diversity_checks(diversity_observed)
        diversity_preserved = all(diversity.values())
        evaluated.append(
            {
                "candidate_position": position,
                "phase_c_overall_rank": int(candidate["overall_rank"]),
                "history_mix": float(candidate["history_mix"]),
                "config_hash": candidate["config_hash"],
                "method_config": candidate["method_config"],
                "observed": observed,
                "diversity_observed": diversity_observed,
                "diversity_checks": diversity,
                "diversity_preserved": diversity_preserved,
                **metric,
                "formal_authorized": bool(metric["metric_authorized"] and diversity_preserved),
                "run_ids": subset.sort_values("scenario_id")["run_id"].tolist(),
            }
        )
    authorized = [candidate for candidate in evaluated if candidate["formal_authorized"]]
    authorized.sort(
        key=lambda value: (
            -int(value["strict_passes"]),
            -float(value["normalized_margin_score"]),
            int(value["phase_c_overall_rank"]),
        )
    )
    winner = authorized[0] if authorized else None
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "dataset_id": "D3",
        "selection_split": "validation",
        "test_labels_used": False,
        "reused_baseline_envelope": envelope,
        "h1_diversity_thresholds": _load_h1_diversity_thresholds(),
        "candidate_evaluations": evaluated,
        "formal_authorized": winner is not None,
        "winner": winner,
        "fullval_manifest_hash": config_hash(manifest),
    }
    atomic_json(RESULT_PATH, payload)
    atomic_json(COMPARISON_PATH, payload)
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
                "full_validation_completed_authorized"
                if result["formal_authorized"]
                else "full_validation_completed_not_authorized"
            ),
            "current_job_id": None,
            "formal_authorized": bool(result["formal_authorized"]),
            "winner": (
                None
                if result["winner"] is None
                else {
                    "history_mix": result["winner"]["history_mix"],
                    "phase_c_overall_rank": result["winner"]["phase_c_overall_rank"],
                }
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
