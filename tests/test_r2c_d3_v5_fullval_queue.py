from __future__ import annotations

from pathlib import Path

from r2c_baselines import r2c_d3_v5_fullval_queue as queue


def _candidate(position: int, alpha: float, beta: float) -> dict[str, object]:
    config = {
        "r2c_protocol_version": queue.PROTOCOL_VERSION,
        "r2c_v3_fixed_server_alpha": alpha,
        "r2c_v4_deployment_ema_betas": [beta],
        "r2c_v4_primary_deployment_beta": beta,
    }
    return {
        "overall_rank": position,
        "alpha": alpha,
        "beta": beta,
        "method_config": config,
        "config_hash": "test",
    }


def test_build_jobs_contains_two_candidates_and_two_frontier_baselines() -> None:
    selection = {"selected_candidates": [_candidate(1, 1.0, 0.8), _candidate(2, 0.875, 0.9)]}
    baselines = {
        "PowerOfChoice": {"lr_mult": 1.0, "pow_d": 16},
        "F3AST": {"lr_mult": 1.0, "f3ast_beta": 0.0005},
    }
    jobs = queue._build_jobs(selection, baselines)
    assert len(jobs) == 8
    assert len({job["job_id"] for job in jobs}) == 8
    assert all(job["dataset_id"] == "D3" and job["evaluation_split"] == "validation" for job in jobs)
    assert all(job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260810 for job in jobs)
    assert [(job["method_id"], job["scenario_id"]) for job in jobs[:4]] == [
        (method, scenario) for method in queue.BASELINE_METHODS for scenario in queue.SCENARIOS
    ]


def test_authorization_accepts_four_strict_wins() -> None:
    envelope = {
        "s0_last50_accuracy": 0.89,
        "s4_last50_accuracy": 0.89,
        "s4_recovery_deficit_auc20": 0.001,
        "s4_algorithm_tta_s": 200.0,
    }
    observed = {
        "s0_last50_accuracy": 0.90,
        "s4_last50_accuracy": 0.90,
        "s4_recovery_deficit_auc20": 0.0,
        "s4_algorithm_tta_s": 190.0,
    }
    result = queue._evaluate_candidate(observed, envelope)
    assert result["strict_passes"] == 4
    assert result["formal_authorized"] is True


def test_authorization_accepts_three_wins_and_one_close_miss() -> None:
    envelope = {
        "s0_last50_accuracy": 0.89,
        "s4_last50_accuracy": 0.89,
        "s4_recovery_deficit_auc20": 0.001,
        "s4_algorithm_tta_s": 200.0,
    }
    observed = {
        "s0_last50_accuracy": 0.90,
        "s4_last50_accuracy": 0.889,
        "s4_recovery_deficit_auc20": 0.0,
        "s4_algorithm_tta_s": 190.0,
    }
    result = queue._evaluate_candidate(observed, envelope)
    assert result["strict_passes"] == 3
    assert result["sole_miss"] == "s4_last50_accuracy"
    assert result["formal_authorized"] is True


def test_authorization_rejects_two_misses_even_when_close() -> None:
    envelope = {
        "s0_last50_accuracy": 0.89,
        "s4_last50_accuracy": 0.89,
        "s4_recovery_deficit_auc20": 0.001,
        "s4_algorithm_tta_s": 200.0,
    }
    observed = {
        "s0_last50_accuracy": 0.889,
        "s4_last50_accuracy": 0.889,
        "s4_recovery_deficit_auc20": 0.0,
        "s4_algorithm_tta_s": 190.0,
    }
    result = queue._evaluate_candidate(observed, envelope)
    assert result["strict_passes"] == 2
    assert result["formal_authorized"] is False


def test_partial_fullval_run_uses_new_attempt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(queue, "RUN_ROOT", tmp_path)
    base = "BVAL-D3-S4-PowerOfChoice-s20260810"
    (tmp_path / base).mkdir()
    actual, retry_of = queue._actual_run_id(base)
    assert actual == f"{base}-a2"
    assert retry_of == base

