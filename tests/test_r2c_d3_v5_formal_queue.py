from __future__ import annotations

from pathlib import Path

from r2c_baselines import r2c_d3_v5_formal_queue as queue


def _winner() -> dict[str, object]:
    config = {
        "r2c_protocol_version": queue.PROTOCOL_VERSION,
        "r2c_v2_audit_replay": False,
        "r2c_v3_fixed_server_alpha": 1.0,
        "r2c_v4_deployment_ema_betas": [0.8],
        "r2c_v4_primary_deployment_beta": 0.8,
    }
    return {"alpha": 1.0, "beta": 0.8, "method_config": config, "config_hash": "validation-hash"}


def test_formal_jobs_are_exactly_one_matched_seed_pair() -> None:
    jobs = queue._build_jobs(_winner())
    assert len(jobs) == 2
    assert [job["scenario_id"] for job in jobs] == ["S0", "S4"]
    assert all(job["dataset_id"] == "D3" and job["evaluation_split"] == "test" for job in jobs)
    assert all(job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260811 for job in jobs)
    assert all(job["full_logging"] and job["method_config"]["r2c_v2_audit_replay"] for job in jobs)


def test_termination_accepts_four_strict_wins() -> None:
    observed = {
        "s0_last50_accuracy": queue.THRESHOLDS["s0_last50_accuracy"] + 0.001,
        "s4_last50_accuracy": queue.THRESHOLDS["s4_last50_accuracy"] + 0.001,
        "s4_recovery_deficit_auc20": 0.0,
        "s4_algorithm_tta_s": queue.THRESHOLDS["s4_algorithm_tta_s"] - 1.0,
    }
    result = queue._evaluate_termination(observed)
    assert result["strict_passes"] == 4
    assert result["goal_met"] is True


def test_termination_accepts_three_wins_and_close_accuracy() -> None:
    observed = {
        "s0_last50_accuracy": queue.THRESHOLDS["s0_last50_accuracy"] + 0.001,
        "s4_last50_accuracy": queue.THRESHOLDS["s4_last50_accuracy"] - 0.001,
        "s4_recovery_deficit_auc20": 0.0,
        "s4_algorithm_tta_s": queue.THRESHOLDS["s4_algorithm_tta_s"] - 1.0,
    }
    result = queue._evaluate_termination(observed)
    assert result["strict_passes"] == 3
    assert result["sole_miss"] == "s4_last50_accuracy"
    assert result["goal_met"] is True


def test_termination_rejects_two_close_misses() -> None:
    observed = {
        "s0_last50_accuracy": queue.THRESHOLDS["s0_last50_accuracy"] - 0.001,
        "s4_last50_accuracy": queue.THRESHOLDS["s4_last50_accuracy"] - 0.001,
        "s4_recovery_deficit_auc20": 0.0,
        "s4_algorithm_tta_s": queue.THRESHOLDS["s4_algorithm_tta_s"] - 1.0,
    }
    result = queue._evaluate_termination(observed)
    assert result["strict_passes"] == 2
    assert result["goal_met"] is False


def test_frozen_thresholds_match_current_table1_bundle() -> None:
    lineage = queue._verify_frozen_thresholds()
    assert lineage["s0_last50_accuracy"]["method_id"] == "PowerOfChoice"
    assert lineage["s4_last50_accuracy"]["method_id"] == "PowerOfChoice"
    assert lineage["s4_recovery_deficit_auc20"]["method_id"] == "F3AST"
    assert lineage["s4_algorithm_tta_s"]["method_id"] == "F3AST"


def test_partial_formal_run_uses_new_attempt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(queue, "RUN_ROOT", tmp_path)
    base = "A-R2C-D3-S4-V5FORMAL-A1000-B0800-s20260811"
    (tmp_path / base).mkdir()
    actual, retry_of = queue._actual_run_id(base)
    assert actual == f"{base}-a2"
    assert retry_of == base

