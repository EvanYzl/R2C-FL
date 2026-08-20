from __future__ import annotations

import json
from pathlib import Path

from r2c_baselines import r2c_v11_four_dataset_matched_queue as queue


def test_manifest_has_exactly_eight_matched_seed_formal_jobs() -> None:
    manifest = queue.build_manifest(persist=False)
    jobs = manifest["jobs"]
    assert manifest["formal_seed"] == 20260811
    assert manifest["protocol_version"] == "telemetry-quarantine-deployment-v7"
    assert manifest["matched_baseline_lineage"]["count"] == 56
    assert manifest["user_authorized_after_validation_review"] is True
    assert manifest["performance_sealed_until_terminal"] is True
    assert len(jobs) == 8
    assert [job["job_id"] for job in jobs] == manifest["job_order"]
    assert [(job["dataset_id"], job["scenario_id"]) for job in jobs] == [
        (dataset_id, scenario_id)
        for dataset_id in queue.DATASET_ORDER
        for scenario_id in queue.SCENARIO_ORDER
    ]
    assert [job["rounds"] for job in jobs] == [500, 500, 1000, 1000, 1000, 1000, 1500, 1500]
    assert all(
        job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260811
        for job in jobs
    )
    assert all(job["evaluation_split"] == "test" and job["full_logging"] for job in jobs)
    assert all(
        job["method_config"]["r2c_protocol_version"] == "telemetry-quarantine-deployment-v7"
        and job["method_config"]["r2c_v4_deployment_ema_betas"] == [0.95]
        and job["method_config"]["r2c_v4_primary_deployment_beta"] == 0.95
        and job["method_config"]["r2c_v7_trigger_deployment_beta"] == 1.0
        and job["method_config"]["r2c_v2_audit_replay"] is True
        for job in jobs
    )


def test_config_composition_uses_only_preformal_dataset_fields() -> None:
    manifest = queue.build_manifest(persist=False)
    initial_v11 = json.loads(queue.V11_INITIAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    initial_v4 = json.loads(queue.V4_INITIAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    base = dict(initial_v11["jobs"][0]["method_config"])
    for dataset_id in queue.DATASET_ORDER:
        job = next(job for job in manifest["jobs"] if job["dataset_id"] == dataset_id)
        config = dict(job["method_config"])
        source = next(job for job in initial_v4["jobs"] if job["dataset_id"] == dataset_id)[
            "method_config"
        ]
        for key in queue.DATASET_CONFIG_KEYS:
            assert config[key] == source[key]
        non_dataset_keys = set(base) - set(queue.DATASET_CONFIG_KEYS) - {"r2c_v2_audit_replay"}
        assert all(config[key] == base[key] for key in non_dataset_keys)
        assert config["r2c_v2_audit_replay"] is True
    d3 = next(job for job in manifest["jobs"] if job["dataset_id"] == "D3")["method_config"]
    assert sorted(key for key in set(base) | set(d3) if base.get(key) != d3.get(key)) == [
        "r2c_v2_audit_replay"
    ]


def test_source_and_implementation_hashes_are_bound() -> None:
    manifest = queue.build_manifest(persist=False)
    assert manifest["source_hashes"] == queue._verify_source_hashes()
    assert manifest["implementation_hashes"] == queue._implementation_hashes()
    assert manifest["dataset_config_lineage"]["authorization"]["validation_goal_met"] is True
    assert manifest["dataset_config_lineage"]["authorization"]["test_labels_used"] is False


def test_success_reconciliation_prevents_duplicate_rerun(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(queue, "RUN_ROOT", tmp_path)
    run_id = "A-R2C-V11MS-D1-S0-B095-s20260811"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "result.json").write_text("{}", encoding="utf-8")
    (run_dir / "_SUCCESS.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(queue, "EVENTS_PATH", tmp_path / "events.parquet")
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
    base = "A-R2C-V11MS-D4-S4-B095-s20260811"
    (tmp_path / base).mkdir()
    (tmp_path / f"{base}-a2").mkdir()
    actual, retry_of = queue._actual_run_id(base)
    assert actual == f"{base}-a3"
    assert retry_of == base
