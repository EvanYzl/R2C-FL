from __future__ import annotations

from pathlib import Path

import pandas as pd

from r2c_baselines import r2c_d3_v5_screen_queue as queue


def test_screen_manifest_is_validation_only_and_frozen() -> None:
    manifest = queue.build_manifest(persist=False)
    jobs = manifest["jobs"]
    assert manifest["formal_test_access"] is False
    assert manifest["test_labels_used_for_selection"] is False
    assert manifest["dev_seed"] == 20260810
    assert manifest["rounds_per_job"] == 600
    assert len(jobs) == 3
    assert [job["job_id"] for job in jobs] == manifest["job_order"]
    assert all(job["dataset_id"] == "D3" and job["scenario_id"] == "S4" for job in jobs)
    assert all(job["evaluation_split"] == "validation" for job in jobs)
    assert all(job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260810 for job in jobs)
    assert [job["method_config"]["r2c_v3_fixed_server_alpha"] for job in jobs] == list(queue.ALPHAS)
    assert all(tuple(job["method_config"]["r2c_v4_deployment_ema_betas"]) == queue.BETAS for job in jobs)
    assert all(job["method_config"]["r2c_v2_audit_replay"] is False for job in jobs)


def test_rank_candidates_uses_all_three_metrics_and_frozen_tiebreakers() -> None:
    frame = pd.DataFrame(
        [
            {"alpha": 1.0, "beta": 0.0, "last50_validation_accuracy": 0.90, "recovery_deficit_auc20": 0.002, "target_hit_round": 80, "complete": True},
            {"alpha": 0.875, "beta": 0.8, "last50_validation_accuracy": 0.89, "recovery_deficit_auc20": 0.000, "target_hit_round": 70, "complete": True},
            {"alpha": 0.75, "beta": 0.95, "last50_validation_accuracy": 0.88, "recovery_deficit_auc20": 0.001, "target_hit_round": 90, "complete": True},
        ]
    )
    ranked = queue._rank_candidates(frame)
    assert set(ranked.columns) >= {"accuracy_rank", "auc_rank", "tta_round_rank", "rank_sum"}
    assert ranked.iloc[0]["alpha"] == 0.875
    assert int(ranked.iloc[0]["rank_sum"]) == 4


def test_success_reconciliation_prevents_duplicate_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(queue, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(queue, "EVENTS_PATH", tmp_path / "events.parquet")
    run_id = "A-R2C-D3-S4-V5SCREEN-A0750-s20260810"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    (run_dir / "result.json").write_text("{}", encoding="utf-8")
    (run_dir / "_SUCCESS.json").write_text("{}", encoding="utf-8")
    manifest = {"jobs": [{"job_id": run_id, "base_run_id": run_id, "actual_run_id": None, "status": "running", "attempts": 1}]}
    events: list[dict[str, object]] = []
    assert queue._reconcile_successes(manifest, events) == 1
    assert manifest["jobs"][0]["status"] == "completed"
    assert manifest["jobs"][0]["actual_run_id"] == run_id


def test_partial_screen_run_uses_new_attempt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(queue, "RUN_ROOT", tmp_path)
    base = "A-R2C-D3-S4-V5SCREEN-A1000-s20260810"
    (tmp_path / base).mkdir()
    (tmp_path / f"{base}-a2").mkdir()
    actual, retry_of = queue._actual_run_id(base)
    assert actual == f"{base}-a3"
    assert retry_of == base

