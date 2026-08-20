from __future__ import annotations

from pathlib import Path

import pandas as pd

from r2c_baselines import r2c_d3_v6_diversity_screen_queue as queue
from r2c_baselines.r2c_v5 import PROTOCOL_VERSION


def test_manifest_is_four_job_validation_only_diversity_screen() -> None:
    manifest = queue.build_manifest(persist=False)
    jobs = manifest["jobs"]
    assert manifest["formal_test_access"] is False
    assert manifest["test_labels_used_for_selection"] is False
    assert manifest["dev_seed"] == 20260810
    assert manifest["rounds_per_job"] == 600
    assert manifest["candidate_history_mixes"] == list(queue.HISTORY_MIXES)
    assert len(jobs) == 4
    assert [job["job_id"] for job in jobs] == manifest["job_order"]
    assert all(job["dataset_id"] == "D3" and job["scenario_id"] == "S4" for job in jobs)
    assert all(job["evaluation_split"] == "validation" for job in jobs)
    assert all(job["full_logging"] is True for job in jobs)
    assert all(job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260810 for job in jobs)
    assert [job["method_config"]["r2c_v5_history_mix"] for job in jobs] == list(
        queue.HISTORY_MIXES
    )
    assert all(job["method_config"]["r2c_protocol_version"] == PROTOCOL_VERSION for job in jobs)
    assert all(job["method_config"]["r2c_v5_history_temperature"] == 1.0 for job in jobs)
    assert all(job["method_config"]["r2c_v3_fixed_server_alpha"] == 1.0 for job in jobs)
    assert all(job["method_config"]["r2c_v4_deployment_ema_betas"] == [0.9] for job in jobs)
    assert all(job["method_config"]["r2c_v2_audit_replay"] is False for job in jobs)


def test_rank_candidates_uses_five_metrics_and_hard_diversity_gate() -> None:
    frame = pd.DataFrame(
        [
            {"history_mix": 0.15, "last50_validation_accuracy": 0.90, "recovery_deficit_auc20": 0.002, "target_hit_round": 80, "final_participation_jfi": 0.90, "final_worst10_participation": 31.0, "complete": True},
            {"history_mix": 0.30, "last50_validation_accuracy": 0.89, "recovery_deficit_auc20": 0.000, "target_hit_round": 70, "final_participation_jfi": 0.91, "final_worst10_participation": 32.0, "complete": True},
            {"history_mix": 0.45, "last50_validation_accuracy": 0.88, "recovery_deficit_auc20": 0.001, "target_hit_round": 90, "final_participation_jfi": 0.92, "final_worst10_participation": 33.0, "complete": True},
            {"history_mix": 0.60, "last50_validation_accuracy": 0.95, "recovery_deficit_auc20": 0.000, "target_hit_round": 50, "final_participation_jfi": 0.80, "final_worst10_participation": 10.0, "complete": True},
        ]
    )
    ranked = queue._rank_candidates(frame)
    assert len(ranked) == 3
    assert 0.60 not in set(ranked["history_mix"])
    assert set(ranked.columns) >= {
        "accuracy_rank",
        "auc_rank",
        "tta_round_rank",
        "jfi_rank",
        "worst10_rank",
        "rank_sum",
    }
    assert (ranked["final_participation_jfi"] > queue.CONTROL_FINAL_JFI).all()
    assert (ranked["final_worst10_participation"] > queue.CONTROL_FINAL_WORST10).all()


def test_success_reconciliation_prevents_duplicate_run(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(queue, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(queue, "EVENTS_PATH", tmp_path / "events.parquet")
    run_id = "A-R2C-D3-S4-V6DIV-L0150-s20260810"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "result.json").write_text("{}", encoding="utf-8")
    (run_dir / "_SUCCESS.json").write_text("{}", encoding="utf-8")
    manifest = {
        "jobs": [
            {
                "job_id": run_id,
                "base_run_id": run_id,
                "actual_run_id": None,
                "status": "running",
                "attempts": 1,
            }
        ]
    }
    events: list[dict[str, object]] = []
    assert queue._reconcile_successes(manifest, events) == 1
    assert manifest["jobs"][0]["status"] == "completed"
    assert manifest["jobs"][0]["actual_run_id"] == run_id


def test_partial_run_uses_new_attempt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(queue, "RUN_ROOT", tmp_path)
    base = "A-R2C-D3-S4-V6DIV-L0600-s20260810"
    (tmp_path / base).mkdir()
    (tmp_path / f"{base}-a2").mkdir()
    actual, retry_of = queue._actual_run_id(base)
    assert actual == f"{base}-a3"
    assert retry_of == base
