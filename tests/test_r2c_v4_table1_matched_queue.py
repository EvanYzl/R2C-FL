from __future__ import annotations

import json
from pathlib import Path

from r2c_baselines import r2c_v4_table1_matched_queue as queue


def test_matched_manifest_has_eight_frozen_seed_jobs() -> None:
    manifest = queue.build_manifest(persist=False)
    jobs = manifest["jobs"]
    assert manifest["formal_seed"] == 20260811
    assert manifest["protocol_version"] == "dual-timescale-deployment-v4"
    assert manifest["matched_baseline_lineage"]["count"] == 56
    assert len(jobs) == 8
    assert [job["job_id"] for job in jobs] == manifest["job_order"]
    assert [(job["dataset_id"], job["scenario_id"]) for job in jobs] == [
        (dataset_id, scenario_id)
        for dataset_id in queue.DATASET_ORDER
        for scenario_id in queue.SCENARIO_ORDER
    ]
    assert all(
        job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260811
        for job in jobs
    )
    assert all(job["evaluation_split"] == "test" and job["full_logging"] for job in jobs)
    assert all(
        job["method_config"]["r2c_v4_deployment_ema_betas"] == [0.95]
        and job["method_config"]["r2c_v4_primary_deployment_beta"] == 0.95
        and job["method_config"]["r2c_v3_fixed_server_alpha"] == 0.75
        and job["method_config"]["r2c_v2_audit_replay"] is True
        for job in jobs
    )


def test_dataset_specific_config_sources_are_frozen() -> None:
    manifest = queue.build_manifest(persist=False)
    lineage = manifest["dataset_config_lineage"]
    assert lineage["D1"]["lr_mult"] == 1.5
    assert lineage["D2"]["lr_mult"] == 1.5
    assert lineage["D3"]["lr_mult"] == 1.0
    assert lineage["D4"]["lr_mult"] == 1.0
    assert [lineage[key]["r2c_eval_microbatch"] for key in queue.DATASET_ORDER] == [8, 4, 4, 2]
    assert lineage["D2"]["dataset_specific_source"] == "D2_v4_validation_winner"


def test_success_reconciliation_prevents_duplicate_rerun(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(queue, "RUN_ROOT", tmp_path)
    run_id = "A-R2C-V4MS-D1-S0-s20260811"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "result.json").write_text("{}", encoding="utf-8")
    (run_dir / "_SUCCESS.json").write_text("{}", encoding="utf-8")
    events_path = tmp_path / "events.parquet"
    monkeypatch.setattr(queue, "EVENTS_PATH", events_path)
    manifest = {
        "jobs": [
            {
                "job_id": run_id,
                "base_run_id": run_id,
                "actual_run_id": None,
                "status": "running",
                "attempts": 1,
                "failure_reason": None,
            }
        ]
    }
    events: list[dict[str, object]] = []
    assert queue._reconcile_successes(manifest, events) == 1
    assert manifest["jobs"][0]["status"] == "completed"
    assert manifest["jobs"][0]["actual_run_id"] == run_id
    assert events[-1]["event_type"] == "reconciled_completed"


def test_partial_run_uses_new_attempt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(queue, "RUN_ROOT", tmp_path)
    base = "A-R2C-V4MS-D4-S4-s20260811"
    (tmp_path / base).mkdir()
    (tmp_path / f"{base}-a2").mkdir()
    actual, retry_of = queue._actual_run_id(base)
    assert actual == f"{base}-a3"
    assert retry_of == base

