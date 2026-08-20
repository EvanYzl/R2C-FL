from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATASETS, PLOT_ROOT, QUEUE_ROOT, RUN_ROOT
from .r2c_v11_four_dataset_matched_finalize import finalize as finalize_table1
from .r2c_v11_four_dataset_matched_queue import (
    FORMAL_INTERPRETATION,
    MANIFEST_PATH,
    STATE_PATH,
)
from .r2c_v7 import PROTOCOL_VERSION
from .utils import atomic_json, config_hash, sha256_file, utc_now


IMMUTABLE_MANIFEST_PATH = (
    QUEUE_ROOT / "r2c_v11_four_dataset_matched_manifest_20260817T192402.166886Z.json"
)
EXPECTED_IMMUTABLE_MANIFEST_SHA256 = (
    "7DABDA949D5A96BCA55DD342E791A816832FE4DB30F1F4F9F19D3C857D7EA159"
)
KNOWN_FALLBACK_LINEAGE = "local-race-to-commit-v1"
QUEUE_RESULT_PATH = QUEUE_ROOT / "r2c_v11_four_dataset_matched_result.json"
ACTIVE_REPORT_PATH = PLOT_ROOT / "r2c_v11_lineage_amendment_report.json"
ACTIVE_CLOSURE_PATH = PLOT_ROOT / "r2c_v11_table1_closure_report.json"
ACTIVE_QC_PATH = PLOT_ROOT / "r2c_v11_table1_qc_report.json"


def _stamp(value: str) -> str:
    return value[:19].replace("-", "").replace(":", "").replace("T", "_")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_lineage() -> dict[str, Any]:
    immutable_hash = sha256_file(IMMUTABLE_MANIFEST_PATH).upper()
    _require(
        immutable_hash == EXPECTED_IMMUTABLE_MANIFEST_SHA256,
        f"Immutable manifest hash drift: {immutable_hash}",
    )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    jobs = list(manifest.get("jobs", []))
    _require(len(jobs) == 8, f"Expected 8 formal jobs, found {len(jobs)}")
    _require(
        manifest.get("protocol_version") == PROTOCOL_VERSION,
        "Active manifest protocol drift",
    )
    _require(
        all(job.get("status") == "completed" and job.get("actual_run_id") for job in jobs),
        "Lineage closure requires eight completed jobs",
    )

    run_py_path = Path(__file__).resolve().parent / "run.py"
    frozen_run_py_hash = str(manifest["implementation_hashes"]["run.py"])
    observed_run_py_hash = sha256_file(run_py_path)
    _require(
        observed_run_py_hash == frozen_run_py_hash,
        "run.py changed after the formal manifest freeze",
    )

    entries: list[dict[str, Any]] = []
    compared_fields = (
        "dataset_id",
        "scenario_id",
        "rounds",
        "seed",
        "partition_seed",
        "trace_seed",
        "method_id",
        "evaluation_split",
        "method_config",
    )
    for job in jobs:
        run_id = str(job["actual_run_id"])
        active_job_path = QUEUE_ROOT / "active_jobs" / f"{run_id}.json"
        run_dir = RUN_ROOT / run_id
        run_manifest_path = run_dir / "run_manifest.parquet"
        result_path = run_dir / "result.json"
        success_path = run_dir / "_SUCCESS.json"
        audit_path = QUEUE_ROOT / "worker_logs" / f"{run_id}.audit.log"
        for path in (active_job_path, run_manifest_path, result_path, success_path, audit_path):
            _require(path.exists(), f"Missing lineage evidence: {path}")

        submitted = json.loads(active_job_path.read_text(encoding="utf-8"))
        _require(submitted.get("run_id") == run_id, f"Submitted run ID mismatch for {run_id}")
        for field in compared_fields:
            _require(
                submitted.get(field) == job.get(field),
                f"Submitted job field {field} differs from frozen manifest for {run_id}",
            )
        _require(
            submitted["method_config"].get("r2c_protocol_version") == PROTOCOL_VERSION,
            f"Submitted job lacks the frozen v7 protocol for {run_id}",
        )
        _require(
            all(int(submitted[key]) == 20260811 for key in ("seed", "partition_seed", "trace_seed")),
            f"Submitted seed drift for {run_id}",
        )
        _require(submitted.get("evaluation_split") == "test", f"Split drift for {run_id}")

        full_config = dict(submitted)
        full_config["dataset_spec"] = DATASETS[str(submitted["dataset_id"])].__dict__
        submitted_config_hash = config_hash(full_config)
        run_manifest = pd.read_parquet(run_manifest_path)
        _require(len(run_manifest) == 1, f"Run manifest cardinality failure for {run_id}")
        row = run_manifest.iloc[0]
        _require(str(row["run_id"]) == run_id, f"Run manifest ID mismatch for {run_id}")
        _require(str(row["status"]) == "completed", f"Run status failure for {run_id}")
        _require(str(row["source_kind"]) == "REPRODUCED", f"Source-kind failure for {run_id}")
        _require(
            all(int(row[key]) == 20260811 for key in ("seed", "partition_seed", "trace_seed")),
            f"Run-manifest seed drift for {run_id}",
        )
        _require(
            str(row["config_hash"]) == submitted_config_hash,
            f"Submitted/run-manifest config hash mismatch for {run_id}",
        )
        _require(
            str(row["upstream_commit"]) == KNOWN_FALLBACK_LINEAGE,
            f"Unexpected original lineage value for {run_id}",
        )

        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        _require(audit.get("status") == "passed", f"Per-run audit failure for {run_id}")
        _require(
            audit.get("protocol_version") == PROTOCOL_VERSION,
            f"Per-run audit protocol mismatch for {run_id}",
        )
        _require(int(audit.get("rounds", -1)) == int(job["rounds"]), f"Budget mismatch for {run_id}")
        _require(
            bool(audit.get("recovery_auc20_complete")) == (job["scenario_id"] == "S4"),
            f"Strict AUC@20 audit-window mismatch for {run_id}",
        )

        entries.append(
            {
                "run_id": run_id,
                "dataset_id": job["dataset_id"],
                "scenario_id": job["scenario_id"],
                "rounds": int(job["rounds"]),
                "seed": 20260811,
                "recorded_upstream_commit": KNOWN_FALLBACK_LINEAGE,
                "effective_protocol_version": PROTOCOL_VERSION,
                "lineage_evidence": "submitted_job_protocol_plus_exact_config_hash_plus_passing_audit",
                "submitted_config_hash": submitted_config_hash,
                "run_manifest_config_hash": str(row["config_hash"]),
                "submitted_job_sha256": sha256_file(active_job_path),
                "run_manifest_sha256": sha256_file(run_manifest_path),
                "result_sha256": sha256_file(result_path),
                "success_sha256": sha256_file(success_path),
                "audit_log_sha256": sha256_file(audit_path),
                "strict_auc20_window_complete": bool(audit.get("recovery_auc20_complete")),
            }
        )

    return {
        "schema_version": "1.0.0",
        "generated_utc": utc_now(),
        "status": "passed",
        "amendment_type": "non_mutating_postrun_metadata_lineage_certificate",
        "run_artifacts_modified": False,
        "known_writer_bug": "run.py protocol-lineage allowlist omitted v7 and used the generic R2C fallback",
        "known_recorded_lineage": KNOWN_FALLBACK_LINEAGE,
        "certified_effective_protocol": PROTOCOL_VERSION,
        "formal_interpretation": FORMAL_INTERPRETATION,
        "immutable_manifest_path": str(IMMUTABLE_MANIFEST_PATH),
        "immutable_manifest_sha256": immutable_hash.lower(),
        "active_manifest_path": str(MANIFEST_PATH),
        "active_manifest_sha256": sha256_file(MANIFEST_PATH),
        "frozen_run_py_sha256": frozen_run_py_hash,
        "observed_run_py_sha256": observed_run_py_hash,
        "completed_runs": len(entries),
        "all_submitted_config_hashes_match": True,
        "all_per_run_audits_passed": True,
        "entries": entries,
    }


def close() -> dict[str, Any]:
    report = verify_lineage()
    stamp = _stamp(str(report["generated_utc"]))
    versioned_report_path = PLOT_ROOT / f"r2c_v11_lineage_amendment_report_{stamp}.json"
    atomic_json(versioned_report_path, report)
    atomic_json(ACTIVE_REPORT_PATH, report)

    aggregation = finalize_table1()
    qc = json.loads(ACTIVE_QC_PATH.read_text(encoding="utf-8"))
    _require(bool(qc.get("complete")), "v11 Table 1 QC is not complete")
    _require(int(qc.get("r2c_audits_passed", 0)) == 8, "Not all R2C audits passed")
    _require(int(qc.get("combined_table1_runs", 0)) == 64, "Combined Table 1 run count mismatch")

    queue_result = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "status": "formal_completed_audited",
        "completed_runs": 8,
        "formal_seed": 20260811,
        "protocol_version": PROTOCOL_VERSION,
        "source_kind": "REPRODUCED",
        "formal_interpretation": FORMAL_INTERPRETATION,
        "run_artifacts_modified": False,
        "lineage_amendment_report": str(ACTIVE_REPORT_PATH),
        "lineage_amendment_report_sha256": sha256_file(ACTIVE_REPORT_PATH),
        "aggregation_result": str(PLOT_ROOT / "r2c_v11_table1_aggregation_result.json"),
        "qc_report": str(ACTIVE_QC_PATH),
        "run_ids": [entry["run_id"] for entry in report["entries"]],
    }
    atomic_json(QUEUE_RESULT_PATH, queue_result)

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    state.update(
        {
            "status": "formal_completed_audited",
            "current_job_id": None,
            "all_runs_completed": True,
            "audit_complete": True,
            "aggregation_complete": True,
            "lineage_amendment_complete": True,
            "lineage_amendment_report": str(ACTIVE_REPORT_PATH),
            "result_path": str(QUEUE_RESULT_PATH),
            "updated_utc": utc_now(),
        }
    )
    atomic_json(STATE_PATH, state)

    closure = {
        "schema_version": "1.0.0",
        "completed_utc": utc_now(),
        "status": "completed",
        "qc_complete": True,
        "formal_runs": 8,
        "combined_runs": 64,
        "run_artifacts_modified": False,
        "lineage_amendment_report": str(ACTIVE_REPORT_PATH),
        "lineage_amendment_report_sha256": sha256_file(ACTIVE_REPORT_PATH),
        "table1_qc_report": str(ACTIVE_QC_PATH),
        "table1_qc_report_sha256": sha256_file(ACTIVE_QC_PATH),
        "aggregation": aggregation,
    }
    versioned_closure_path = PLOT_ROOT / f"r2c_v11_table1_closure_report_{stamp}.json"
    atomic_json(versioned_closure_path, closure)
    atomic_json(ACTIVE_CLOSURE_PATH, closure)
    closure["versioned_closure_report"] = str(versioned_closure_path)
    closure["active_closure_report"] = str(ACTIVE_CLOSURE_PATH)
    return closure


def main() -> None:
    print(json.dumps(close(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
