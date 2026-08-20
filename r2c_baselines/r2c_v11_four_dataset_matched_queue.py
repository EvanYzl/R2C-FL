from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .config import BASELINES, DATASETS, FORMAL_SEED, PLOT_ROOT, PROJECT_ROOT, QUEUE_ROOT, RUN_ROOT
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


DATASET_ORDER = ("D1", "D2", "D3", "D4")
SCENARIO_ORDER = ("S0", "S4")
PRIMARY_BETA = 0.95
TRIGGER_BETA = 1.0
MAX_ATTEMPTS = 3
FORMAL_INTERPRETATION = "matched_seed_engineering_reevaluation_not_untouched_confirmation"
DATASET_CONFIG_KEYS = ("lr_mult", "r2c_delta_clip", "r2c_eval_microbatch")

PLAN_PATH = (
    PROJECT_ROOT
    / "refine-logs"
    / "R2C_V11_MATCHED_SEED_FOUR_DATASET_FORMAL_PLAN_20260818_031428.md"
)
EXPECTED_PLAN_SHA256 = "3297AC0DE0B0B3A2B4715CDD5C4E1F52F16457AA4088B75B70BC375A06FF5162"
V11_INITIAL_MANIFEST_PATH = (
    QUEUE_ROOT / "r2c_d3_v11_single_seed_fullval_manifest_20260817T175427.960389Z.json"
)
V11_TERMINAL_MANIFEST_PATH = QUEUE_ROOT / "r2c_d3_v11_single_seed_fullval_manifest.json"
V11_STATE_PATH = QUEUE_ROOT / "r2c_d3_v11_single_seed_fullval_queue_state.json"
V11_RESULT_PATH = QUEUE_ROOT / "r2c_d3_v11_single_seed_fullval_result.json"
V4_INITIAL_MANIFEST_PATH = (
    QUEUE_ROOT / "r2c_v4_table1_matched_manifest_20260816T035638.500920Z.json"
)
SELECTED_CONFIG_PATH = QUEUE_ROOT / "r2c_table1_selected_hyperparameters.json"
TARGET_PATH = QUEUE_ROOT / "frozen_targets.json"
BASELINE_QC_PATH = PLOT_ROOT / "baseline_qc_report.json"
BASELINE_SUMMARY_PATH = PLOT_ROOT / "run_summary.parquet"
BASELINE_MANIFEST_PATH = PLOT_ROOT / "run_manifest.parquet"

EXPECTED_SOURCE_SHA256 = {
    PLAN_PATH: EXPECTED_PLAN_SHA256,
    V11_INITIAL_MANIFEST_PATH: "DF5D35695B118B806F97967B0F6B5CAE9D256F96C9949B4A43471DDE4302084B",
    V11_TERMINAL_MANIFEST_PATH: "52D70012B56C69DEDC7495294E0D3D3B41D1EE559E1878F1308D57AB32773AB8",
    V11_STATE_PATH: "7241401B32E7A27ED33CC53575D2C1B071ED849C78015FB7DE1C8D526B0359E7",
    V11_RESULT_PATH: "AD9CDF6C0BB1998D44B264B1BD7D52CA04E01617461886F9D48D28E084EDEDC1",
    V4_INITIAL_MANIFEST_PATH: "E7839A280825190C5B59ACF3A174C30B2901F661207B912DCC48473F131E18A6",
    SELECTED_CONFIG_PATH: "FB24CFA3767F6516BDF3B8BC547232A281ADD8024E6480D8C534E5E0ADE611E5",
    TARGET_PATH: "9FC5EF7690430369BF595498D7B205AF58B6FB3D59624B1F01C26651234EAAB6",
    BASELINE_QC_PATH: "3668A84F331C4F1987D03E4E67E6C1CAFA31C76F4A6516E83639070249862DA0",
    BASELINE_SUMMARY_PATH: "279B837F656733D0750E640894556879BD9960C57B8E1B1DFBDD2C7E2D6B9CA3",
    BASELINE_MANIFEST_PATH: "347558F83248E4C149CE73692DBCFF87C52F2760EF20AB6340A40D8D9DF799AC",
}

MANIFEST_PATH = QUEUE_ROOT / "r2c_v11_four_dataset_matched_manifest.json"
STATE_PATH = QUEUE_ROOT / "r2c_v11_four_dataset_matched_queue_state.json"
EVENTS_PATH = QUEUE_ROOT / "r2c_v11_four_dataset_matched_scheduler_events.parquet"
RESULT_PATH = QUEUE_ROOT / "r2c_v11_four_dataset_matched_result.json"
RUNS_PATH = PLOT_ROOT / "r2c_v11_four_dataset_matched_runs.parquet"
RUNS_CSV_PATH = PLOT_ROOT / "r2c_v11_four_dataset_matched_runs.csv"
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
        "r2c_v6.py",
        "r2c_v7.py",
        "run.py",
        "r2c_table1_finalize.py",
        "r2c_v11_four_dataset_matched_finalize.py",
        Path(__file__).name,
    )
    values = {name: sha256_file(package / name) for name in names}
    values["tests/audit_r2c_run.py"] = sha256_file(AUDIT_SCRIPT)
    return values


def _verify_source_hashes() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in EXPECTED_SOURCE_SHA256.items():
        actual = sha256_file(path).upper()
        if actual != expected:
            raise RuntimeError(f"Frozen source hash drift for {path}: {actual} != {expected}")
        observed[str(path.relative_to(PROJECT_ROOT))] = actual.lower()
    return observed


def _verify_v11_authorization() -> tuple[dict[str, Any], dict[str, Any]]:
    _verify_source_hashes()
    initial = json.loads(V11_INITIAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    terminal = json.loads(V11_TERMINAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(V11_STATE_PATH.read_text(encoding="utf-8"))
    result = json.loads(V11_RESULT_PATH.read_text(encoding="utf-8"))
    initial_jobs = list(initial.get("jobs", []))
    terminal_jobs = list(terminal.get("jobs", []))
    if (
        len(initial_jobs) != 2
        or len(terminal_jobs) != 2
        or any(job.get("dataset_id") != "D3" for job in initial_jobs)
        or any(job.get("evaluation_split") != "validation" for job in initial_jobs)
        or any(job.get("status") != "pending" for job in initial_jobs)
        or any(job.get("status") != "completed" for job in terminal_jobs)
    ):
        raise RuntimeError("Frozen D3 v11 validation manifest boundary is invalid")
    configs = [dict(job["method_config"]) for job in initial_jobs]
    if config_hash(configs[0]) != config_hash(configs[1]):
        raise RuntimeError("D3 v11 validation pair does not share one configuration")
    config = configs[0]
    if (
        config.get("r2c_protocol_version") != PROTOCOL_VERSION
        or bool(config.get("r2c_v2_audit_replay"))
        or config.get("r2c_v4_deployment_ema_betas") != [PRIMARY_BETA]
        or float(config.get("r2c_v4_primary_deployment_beta", -1.0)) != PRIMARY_BETA
        or float(config.get("r2c_v7_trigger_deployment_beta", -1.0)) != TRIGGER_BETA
    ):
        raise RuntimeError("Accepted D3 v11 configuration drift")
    if config_hash(config) != config_hash(dict(result.get("method_config", {}))):
        raise RuntimeError("D3 v11 validation result configuration mismatch")
    if (
        state.get("status") != "v11_validation_completed_review_required"
        or not bool(state.get("validation_goal_met"))
        or bool(state.get("formal_authorized"))
        or not bool(result.get("validation_goal_met"))
        or bool(result.get("formal_authorized"))
        or not bool(result.get("review_required"))
        or result.get("selection_split") != "validation"
        or bool(result.get("formal_test_access"))
        or bool(result.get("test_labels_used"))
    ):
        raise RuntimeError("D3 v11 did not reach the recorded user-review authorization boundary")
    evidence = {
        "validation_result_sha256": EXPECTED_SOURCE_SHA256[V11_RESULT_PATH].lower(),
        "validation_goal_met": True,
        "strict_wins": int(result["strict_passes"]),
        "sole_close_miss": result.get("sole_miss"),
        "user_acceptance_recorded_in_plan": True,
        "test_labels_used": False,
    }
    return config, evidence


def _verify_matched_baselines() -> dict[str, Any]:
    qc = json.loads(BASELINE_QC_PATH.read_text(encoding="utf-8"))
    if not qc.get("complete"):
        raise RuntimeError("Baseline aggregation QC must be complete")
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
    base, authorization = _verify_v11_authorization()
    v4 = json.loads(V4_INITIAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        len(v4.get("jobs", [])) != 8
        or bool(v4.get("test_labels_used_for_selection"))
        or any(job.get("status") != "pending" for job in v4["jobs"])
    ):
        raise RuntimeError("Immutable v4 pre-formal configuration source is invalid")
    configs: dict[str, dict[str, Any]] = {}
    lineage: dict[str, Any] = {}
    for dataset_id in DATASET_ORDER:
        source_jobs = [job for job in v4["jobs"] if job["dataset_id"] == dataset_id]
        source_configs = [dict(job["method_config"]) for job in source_jobs]
        if len(source_configs) != 2 or config_hash(source_configs[0]) != config_hash(source_configs[1]):
            raise RuntimeError(f"v4 dataset configuration source mismatch for {dataset_id}")
        value = dict(base)
        for key in DATASET_CONFIG_KEYS:
            value[key] = source_configs[0][key]
        value["r2c_v2_audit_replay"] = True
        if (
            value.get("r2c_protocol_version") != PROTOCOL_VERSION
            or value.get("r2c_v4_deployment_ema_betas") != [PRIMARY_BETA]
            or float(value.get("r2c_v4_primary_deployment_beta", -1.0)) != PRIMARY_BETA
            or float(value.get("r2c_v7_trigger_deployment_beta", -1.0)) != TRIGGER_BETA
            or not bool(value.get("r2c_v2_audit_replay"))
        ):
            raise RuntimeError(f"Frozen v11 formal configuration drift for {dataset_id}")
        changed = sorted(key for key in set(base) | set(value) if base.get(key) != value.get(key))
        expected = sorted(
            ["r2c_v2_audit_replay"]
            + [key for key in DATASET_CONFIG_KEYS if base.get(key) != source_configs[0].get(key)]
        )
        if changed != expected:
            raise RuntimeError(f"Unexpected v11-to-formal changes for {dataset_id}: {changed}")
        configs[dataset_id] = value
        lineage[dataset_id] = {
            "config_hash": config_hash(value),
            "validation_to_formal_changed_keys": changed,
            "lr_mult": float(value["lr_mult"]),
            "r2c_delta_clip": float(value["r2c_delta_clip"]),
            "r2c_eval_microbatch": int(value["r2c_eval_microbatch"]),
            "dataset_specific_source": "immutable_v4_preformal_validation_freeze",
        }
    return configs, {"authorization": authorization, "datasets": lineage}


def _ensure_assets() -> dict[str, str]:
    paths: list[Path] = []
    for dataset_id in DATASET_ORDER:
        rounds = int(DATASETS[dataset_id].round_budget)
        prepare_partition(dataset_id, FORMAL_SEED)
        paths.extend(
            [partition_asset_path(dataset_id, FORMAL_SEED), partition_meta_path(dataset_id, FORMAL_SEED)]
        )
        for scenario_id in SCENARIO_ORDER:
            prepare_trace(dataset_id, scenario_id, FORMAL_SEED, rounds=rounds)
            paths.extend(
                [
                    trace_asset_path(dataset_id, scenario_id, FORMAL_SEED, rounds),
                    trace_meta_path(dataset_id, scenario_id, FORMAL_SEED, rounds),
                ]
            )
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def _job(
    dataset_id: str,
    scenario_id: str,
    config: dict[str, Any],
    target_accuracy: float,
) -> dict[str, Any]:
    run_id = f"A-R2C-V11MS-{dataset_id}-{scenario_id}-B095-s{FORMAL_SEED}"
    return {
        "job_id": run_id,
        "base_run_id": run_id,
        "stage": "v11_four_dataset_matched_seed_formal_engineering_reevaluation",
        "mode": "formal",
        "method_id": "R2C-FL",
        "dataset_id": dataset_id,
        "scenario_id": scenario_id,
        "rounds": int(DATASETS[dataset_id].round_budget),
        "method_config": dict(config),
        "block_id": "A-R2C-V11-FOUR-DATASET-MATCHED",
        "seed": int(FORMAL_SEED),
        "partition_seed": int(FORMAL_SEED),
        "trace_seed": int(FORMAL_SEED),
        "evaluation_split": "test",
        "full_logging": True,
        "client_microbatch": 1,
        "target_accuracy": float(target_accuracy),
        "variant_label": "V11MS-B095-A100-H045-QV7",
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
            raise RuntimeError("Refusing to force-rebuild a started v11 matched formal manifest")
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
        raise AssertionError("v11 matched queue must contain exactly 8 unique jobs")
    manifest = {
        "schema_version": "1.0.0",
        "created_utc": utc_now(),
        "scope": "D1_D4_v11_table1_matched_seed_formal",
        "protocol_version": PROTOCOL_VERSION,
        "formal_seed": int(FORMAL_SEED),
        "formal_test_access": True,
        "engineering_reevaluation": True,
        "user_authorized_after_validation_review": True,
        "formal_interpretation": FORMAL_INTERPRETATION,
        "test_labels_used_for_selection": False,
        "dataset_order": list(DATASET_ORDER),
        "scenario_order": list(SCENARIO_ORDER),
        "job_order": [job["job_id"] for job in jobs],
        "primary_deployment_beta": PRIMARY_BETA,
        "trigger_deployment_beta": TRIGGER_BETA,
        "dataset_config_lineage": config_lineage,
        "matched_baseline_lineage": baseline_lineage,
        "clean_gpu_launch_gate": {
            "samples": 2,
            "utilization_strictly_below_percent": 10,
            "minimum_separation_seconds": 30,
            "no_other_training_worker": True,
        },
        "source_hashes": _verify_source_hashes(),
        "asset_hashes": _ensure_assets(),
        "implementation_hashes": _implementation_hashes(),
        "max_attempts": MAX_ATTEMPTS,
        "completion_rule": "all 8 jobs completed; no metric-dependent early stopping",
        "performance_sealed_until_terminal": True,
        "jobs": jobs,
    }
    if persist:
        stamp = utc_now().replace(":", "").replace("-", "")
        atomic_json(QUEUE_ROOT / f"r2c_v11_four_dataset_matched_manifest_{stamp}.json", manifest)
        atomic_json(MANIFEST_PATH, manifest)
        atomic_json(
            STATE_PATH,
            {
                "status": "ready_waiting_clean_gpu_gate",
                "created_utc": utc_now(),
                "updated_utc": utc_now(),
                "current_job_id": None,
                "completed": 0,
                "failed": 0,
                "total": 8,
                "all_runs_completed": False,
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
    if (
        manifest.get("protocol_version") != PROTOCOL_VERSION
        or int(manifest.get("formal_seed", -1)) != 20260811
        or not bool(manifest.get("formal_test_access"))
        or not bool(manifest.get("user_authorized_after_validation_review"))
        or bool(manifest.get("test_labels_used_for_selection"))
        or not bool(manifest.get("performance_sealed_until_terminal"))
    ):
        raise RuntimeError("v11 four-dataset matched manifest scope drift")
    if manifest.get("implementation_hashes") != _implementation_hashes():
        raise RuntimeError("Implementation hashes changed after formal manifest freeze")
    if manifest.get("source_hashes") != _verify_source_hashes():
        raise RuntimeError("Source hashes changed after formal manifest freeze")
    jobs = list(manifest.get("jobs", []))
    expected_pairs = [(d, s) for d in DATASET_ORDER for s in SCENARIO_ORDER]
    if len(jobs) != 8 or [(j["dataset_id"], j["scenario_id"]) for j in jobs] != expected_pairs:
        raise RuntimeError("v11 matched manifest job matrix/order drift")
    if [job["job_id"] for job in jobs] != manifest.get("job_order"):
        raise RuntimeError("v11 matched manifest job-id order drift")
    expected_configs, _ = _frozen_dataset_configs()
    for job in jobs:
        if any(int(job[key]) != 20260811 for key in ("seed", "partition_seed", "trace_seed")):
            raise RuntimeError(f"Seed drift in {job['job_id']}")
        if job.get("evaluation_split") != "test" or int(job["rounds"]) != int(
            DATASETS[job["dataset_id"]].round_budget
        ):
            raise RuntimeError(f"Split/budget drift in {job['job_id']}")
        if config_hash(job["method_config"]) != config_hash(expected_configs[job["dataset_id"]]):
            raise RuntimeError(f"Configuration drift in {job['job_id']}")


def _time_to_accuracy(rounds: pd.DataFrame, target: float) -> tuple[int | None, float | None]:
    reached = rounds.loc[rounds["test_accuracy"].astype(float) >= float(target)]
    if reached.empty:
        return None, None
    first = reached.sort_values("round").iloc[0]
    return int(first["round"]), float(first["algorithm_elapsed_s"])


def freeze_result(manifest: dict[str, Any]) -> dict[str, Any]:
    if any(job["status"] != "completed" for job in manifest["jobs"]):
        raise RuntimeError("Cannot freeze incomplete v11 four-dataset matched matrix")
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
        expected_rows = {
            "round_metrics": expected_rounds,
            "client_round_metrics": expected_rounds * 100,
            "checkpoint_metrics": expected_rounds * 20,
            "certificate_audit": expected_rounds,
            "deployment_candidate_metrics": expected_rounds,
        }
        indices = result["table_indices"]
        for table, expected in expected_rows.items():
            if int(indices[table]["rows"]) != expected:
                raise RuntimeError(f"{run_id} {table} rows {indices[table]['rows']} != {expected}")
        rounds = read_chunked_table(run_dir, "round_metrics").sort_values("round")
        if len(rounds) != expected_rounds or rounds[["run_id", "round"]].duplicated().any():
            raise RuntimeError(f"Round key/budget failure for {run_id}")
        recovery = result["recovery"]
        event_round: int | None = None
        if job["scenario_id"] == "S4":
            offsets = rounds["event_offset_round"].astype(int)
            event = rounds.loc[offsets == 0]
            if len(event) != 1 or int(offsets.between(-20, -1).sum()) != 20 or int(
                offsets.between(1, 20).sum()
            ) != 20:
                raise RuntimeError(f"Strict AUC@20 role window failure for {run_id}")
            event_round = int(event.iloc[0]["round"])
            recomputed = recovery_auc20(rounds["round"], rounds["test_accuracy"], event_round)
            if (
                not bool(recovery["recovery_auc20_complete"])
                or recovery["recovery_deficit_auc20"] is None
                or not bool(recomputed["recovery_auc20_complete"])
                or abs(
                    float(recomputed["recovery_deficit_auc20"])
                    - float(recovery["recovery_deficit_auc20"])
                )
                > 1.0e-12
            ):
                raise RuntimeError(f"Strict AUC@20 value mismatch for {run_id}")
        elif recovery["recovery_deficit_auc20"] is not None:
            raise RuntimeError(f"Stationary run unexpectedly has AUC@20 for {run_id}")
        tta_round, tta_s = _time_to_accuracy(rounds, float(job["target_accuracy"]))
        rows.append(
            {
                "run_id": run_id,
                "dataset_id": job["dataset_id"],
                "scenario_id": job["scenario_id"],
                "round_budget": expected_rounds,
                "event_round": event_round,
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
        "status": "formal_completed_pending_finalize",
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
            "status": "formal_completed_pending_finalize",
            "current_job_id": None,
            "all_runs_completed": True,
            "result_path": str(RESULT_PATH),
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
