from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BASELINES, DATASETS, FORMAL_SEED, PLOT_ROOT, QUEUE_ROOT, RUN_ROOT
from .logging_io import read_chunked_table
from .r2c_v4 import PROTOCOL_VERSION
from .utils import (
    atomic_csv,
    atomic_json,
    atomic_parquet,
    config_hash,
    sha256_file,
    sha256_text,
    utc_now,
)


SOURCE_V4_MANIFEST_PATH = QUEUE_ROOT / "r2c_d2_v4_formal_manifest.json"
SOURCE_SELECTED_CONFIG_PATH = QUEUE_ROOT / "r2c_table1_selected_hyperparameters.json"
TARGET_PATH = QUEUE_ROOT / "frozen_targets.json"
BASELINE_QC_PATH = PLOT_ROOT / "baseline_qc_report.json"
BASELINE_SUMMARY_PATH = PLOT_ROOT / "run_summary.parquet"
BASELINE_MANIFEST_PATH = PLOT_ROOT / "run_manifest.parquet"

MANIFEST_PATH = QUEUE_ROOT / "r2c_v4_table1_matched_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_v4_table1_matched_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_v4_table1_matched_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_v4_table1_matched_result.json"
RUNS_PATH = PLOT_ROOT / "r2c_v4_table1_matched_runs.parquet"
RUNS_CSV_PATH = PLOT_ROOT / "r2c_v4_table1_matched_runs.csv"

FORMAL_INTERPRETATION = "matched_seed_engineering_reevaluation_not_untouched_confirmation"
PRIMARY_BETA = 0.95
FAST_SERVER_ALPHA = 0.75
DATASET_ORDER = ("D1", "D2", "D3", "D4")
SCENARIO_ORDER = ("S0", "S4")


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
    paths = (
        SOURCE_V4_MANIFEST_PATH,
        SOURCE_SELECTED_CONFIG_PATH,
        TARGET_PATH,
        BASELINE_QC_PATH,
        BASELINE_SUMMARY_PATH,
        BASELINE_MANIFEST_PATH,
    )
    return {str(path.relative_to(QUEUE_ROOT.parent.parent)): sha256_file(path) for path in paths}


def _verify_matched_baselines() -> dict[str, Any]:
    qc = json.loads(BASELINE_QC_PATH.read_text(encoding="utf-8"))
    if not qc.get("complete"):
        raise RuntimeError("Baseline aggregation QC must be complete before matched-seed execution")
    summary = pd.read_parquet(BASELINE_SUMMARY_PATH)
    selected = summary[
        summary["method_id"].isin(BASELINES)
        & summary["dataset_id"].isin(DATASET_ORDER)
        & summary["scenario_id"].isin(SCENARIO_ORDER)
        & (summary["seed"].astype(int) == int(FORMAL_SEED))
    ].copy()
    if len(selected) != 56 or selected["run_id"].duplicated().any():
        raise RuntimeError(f"Expected 56 unique matched Table 1 baseline runs, found {len(selected)}")
    manifests = pd.read_parquet(BASELINE_MANIFEST_PATH)
    lineage = manifests[manifests["run_id"].isin(selected["run_id"])].copy()
    if len(lineage) != 56 or lineage["run_id"].duplicated().any():
        raise RuntimeError("Matched baseline run manifests are incomplete or duplicated")
    for key in ("seed", "partition_seed", "trace_seed"):
        if not (lineage[key].astype(int) == int(FORMAL_SEED)).all():
            raise RuntimeError(f"Baseline {key} is not uniformly {FORMAL_SEED}")
    if not (lineage["source_kind"].astype(str) == "REPRODUCED").all():
        raise RuntimeError("Matched baseline matrix contains non-REPRODUCED provenance")
    if not (lineage["status"].astype(str) == "completed").all():
        raise RuntimeError("Matched baseline matrix contains incomplete runs")
    run_ids = sorted(selected["run_id"].astype(str).tolist())
    return {
        "count": len(run_ids),
        "seed": int(FORMAL_SEED),
        "run_ids_hash": sha256_text("\n".join(run_ids)),
    }


def _frozen_dataset_configs() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    source_manifest = json.loads(SOURCE_V4_MANIFEST_PATH.read_text(encoding="utf-8"))
    if source_manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("D2 v4 source manifest protocol drift")
    source_jobs = source_manifest.get("jobs", [])
    source_configs = [dict(job["method_config"]) for job in source_jobs]
    if len(source_configs) != 2 or config_hash(source_configs[0]) != config_hash(source_configs[1]):
        raise RuntimeError("D2 v4 formal pair does not share one source config")
    base = source_configs[0]
    if base.get("r2c_v4_deployment_ema_betas") != [PRIMARY_BETA]:
        raise RuntimeError("D2 v4 source beta is not the frozen primary beta")
    if float(base.get("r2c_v3_fixed_server_alpha")) != FAST_SERVER_ALPHA:
        raise RuntimeError("D2 v4 source fast server alpha drift")

    selected_payload = json.loads(SOURCE_SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    if selected_payload.get("test_labels_used"):
        raise RuntimeError("Legacy dataset-specific config selection used test labels")
    selected = selected_payload["selected"]["R2C-FL"]
    configs: dict[str, dict[str, Any]] = {}
    lineage: dict[str, Any] = {}
    for dataset_id in DATASET_ORDER:
        legacy = dict(selected[dataset_id])
        value = dict(base)
        if dataset_id != "D2":
            value.update(
                {
                    "lr_mult": float(legacy["lr_mult"]),
                    "r2c_delta_clip": float(legacy["r2c_delta_clip"]),
                    "r2c_eval_microbatch": int(legacy["r2c_eval_microbatch"]),
                }
            )
        value.update(
            {
                "r2c_protocol_version": PROTOCOL_VERSION,
                "r2c_v2_audit_replay": True,
                "r2c_v3_fixed_server_alpha": FAST_SERVER_ALPHA,
                "r2c_v4_deployment_ema_betas": [PRIMARY_BETA],
                "r2c_v4_primary_deployment_beta": PRIMARY_BETA,
            }
        )
        configs[dataset_id] = value
        lineage[dataset_id] = {
            "config_hash": config_hash(value),
            "lr_mult": float(value["lr_mult"]),
            "r2c_delta_clip": float(value["r2c_delta_clip"]),
            "r2c_eval_microbatch": int(value["r2c_eval_microbatch"]),
            "dataset_specific_source": (
                "D2_v4_validation_winner" if dataset_id == "D2" else "legacy_table1_validation_freeze"
            ),
        }
    return configs, lineage


def _job(
    dataset_id: str,
    scenario_id: str,
    config: dict[str, Any],
    target_accuracy: float,
) -> dict[str, Any]:
    run_id = f"A-R2C-V4MS-{dataset_id}-{scenario_id}-s{FORMAL_SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "v4_table1_matched_formal",
        "mode": "formal",
        "method_id": "R2C-FL",
        "dataset_id": dataset_id,
        "scenario_id": scenario_id,
        "rounds": int(DATASETS[dataset_id].round_budget),
        "method_config": dict(config),
        "block_id": "A-R2C-V4-TABLE1-MATCHED",
        "seed": int(FORMAL_SEED),
        "partition_seed": int(FORMAL_SEED),
        "trace_seed": int(FORMAL_SEED),
        "evaluation_split": "test",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": float(target_accuracy),
        "variant_label": "V4MS-B950-A750",
        "formal_interpretation": FORMAL_INTERPRETATION,
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
            raise RuntimeError("Refusing to force-rebuild a started matched-seed formal manifest")
    if int(FORMAL_SEED) != 20260811:
        raise RuntimeError(f"Expected matched formal seed 20260811, found {FORMAL_SEED}")
    targets = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    configs, config_lineage = _frozen_dataset_configs()
    baseline_lineage = _verify_matched_baselines()
    jobs = [
        _job(dataset_id, scenario_id, configs[dataset_id], float(targets[dataset_id]))
        for dataset_id in DATASET_ORDER
        for scenario_id in SCENARIO_ORDER
    ]
    if len(jobs) != 8 or len({job["job_id"] for job in jobs}) != 8:
        raise AssertionError("Matched-seed v4 queue must contain exactly 8 unique jobs")
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D1_D4_v4_table1_matched_seed_formal",
        "protocol_version": PROTOCOL_VERSION,
        "formal_seed": int(FORMAL_SEED),
        "formal_test_access": True,
        "formal_interpretation": FORMAL_INTERPRETATION,
        "test_labels_used_for_selection": False,
        "dataset_order": list(DATASET_ORDER),
        "scenario_order": list(SCENARIO_ORDER),
        "job_order": [job["job_id"] for job in jobs],
        "fixed_fast_server_alpha": FAST_SERVER_ALPHA,
        "fixed_primary_deployment_beta": PRIMARY_BETA,
        "dataset_config_lineage": config_lineage,
        "matched_baseline_lineage": baseline_lineage,
        "source_hashes": _source_hashes(),
        "implementation_hashes": _implementation_hashes(),
        "completion_rule": "all 8 jobs completed; no metric-dependent early stopping",
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_v4_table1_matched_manifest_{stamp}.json", manifest)
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
                "total": 8,
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
            job.update(
                {
                    "status": "completed",
                    "actual_run_id": run_id,
                    "failure_reason": None,
                }
            )
            _event(events, job, "reconciled_completed", reason="existing_success_output")
            changed += 1
    return changed


def _actual_run_id(base: str) -> tuple[str, str | None]:
    path = RUN_ROOT / base
    if not path.exists():
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
        raise RuntimeError("Matched-seed manifest protocol drift")
    if int(manifest.get("formal_seed", -1)) != 20260811:
        raise RuntimeError("Matched-seed manifest seed drift")
    current = _implementation_hashes()
    if current != manifest.get("implementation_hashes"):
        raise RuntimeError("Implementation hashes changed after formal manifest freeze")
    if len(manifest.get("jobs", [])) != 8:
        raise RuntimeError("Matched-seed manifest no longer contains exactly 8 jobs")
    for job in manifest["jobs"]:
        if any(int(job[key]) != 20260811 for key in ("seed", "partition_seed", "trace_seed")):
            raise RuntimeError(f"Seed drift in {job['job_id']}")
        config = job["method_config"]
        if config.get("r2c_protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"Protocol drift in {job['job_id']}")
        if config.get("r2c_v4_deployment_ema_betas") != [PRIMARY_BETA]:
            raise RuntimeError(f"Deployment beta drift in {job['job_id']}")


def _time_to_accuracy(run_dir: Path, target: float) -> tuple[int | None, float | None]:
    rounds = read_chunked_table(run_dir, "round_metrics")
    reached = rounds.loc[rounds["test_accuracy"].astype(float) >= float(target)]
    if reached.empty:
        return None, None
    first = reached.sort_values("round").iloc[0]
    return int(first["round"]), float(first["algorithm_elapsed_s"])


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot freeze incomplete matched-seed v4 Table 1 matrix")
    rows: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        run_id = str(job["actual_run_id"])
        run_dir = RUN_ROOT / run_id
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        run_manifest = pd.read_parquet(run_dir / "run_manifest.parquet").iloc[0]
        if str(run_manifest["source_kind"]) != "REPRODUCED" or str(run_manifest["status"]) != "completed":
            raise RuntimeError(f"Formal provenance/status failure for {run_id}")
        for key in ("seed", "partition_seed", "trace_seed"):
            if int(run_manifest[key]) != 20260811:
                raise RuntimeError(f"{key} mismatch for {run_id}")
        if str(run_manifest["upstream_commit"]) != PROTOCOL_VERSION:
            raise RuntimeError(f"Protocol lineage mismatch for {run_id}")
        expected_rounds = int(job["rounds"])
        indices = result["table_indices"]
        expected_rows = {
            "round_metrics": expected_rounds,
            "client_round_metrics": expected_rounds * 100,
            "checkpoint_metrics": expected_rounds * 20,
            "certificate_audit": expected_rounds,
            "deployment_candidate_metrics": expected_rounds,
        }
        for table, expected in expected_rows.items():
            if int(indices[table]["rows"]) != expected:
                raise RuntimeError(f"{run_id} {table} rows {indices[table]['rows']} != {expected}")
        recovery = result["recovery"]
        if job["scenario_id"] == "S4" and (
            not recovery["recovery_auc20_complete"]
            or recovery["recovery_deficit_auc20"] is None
        ):
            raise RuntimeError(f"Strict AUC@20 window incomplete for {run_id}")
        tta_round, tta_s = _time_to_accuracy(run_dir, float(job["target_accuracy"]))
        rows.append(
            {
                "run_id": run_id,
                "dataset_id": job["dataset_id"],
                "scenario_id": job["scenario_id"],
                "round_budget": expected_rounds,
                "seed": 20260811,
                "last50_test_accuracy": float(result["last50_accuracy"]),
                "recovery_deficit_auc20": recovery["recovery_deficit_auc20"],
                "recovery_auc20_complete": bool(recovery["recovery_auc20_complete"]),
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
    payload = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "status": "formal_completed_pending_audit",
        "datasets": list(DATASET_ORDER),
        "scenarios": list(SCENARIO_ORDER),
        "formal_seed": 20260811,
        "protocol_version": PROTOCOL_VERSION,
        "source_kind": "REPRODUCED",
        "formal_interpretation": FORMAL_INTERPRETATION,
        "test_labels_used_for_selection": False,
        "completed_runs": len(frame),
        "run_ids": frame["run_id"].tolist(),
        "manifest_hash": config_hash(manifest),
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
    freeze_result(manifest)
    state.update(
        {
            "status": "formal_completed_pending_audit",
            "current_job_id": None,
            "all_runs_completed": True,
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

