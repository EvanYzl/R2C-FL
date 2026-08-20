from __future__ import annotations

from pathlib import Path

import pytest

from r2c_baselines import r2c_d3_v6_fullval_queue as queue


def _candidate(rank: int, history_mix: float) -> dict[str, object]:
    config = {
        "r2c_protocol_version": queue.PROTOCOL_VERSION,
        "r2c_v3_fixed_server_alpha": 1.0,
        "r2c_v4_deployment_ema_betas": [0.9],
        "r2c_v4_primary_deployment_beta": 0.9,
        "r2c_v5_history_mix": history_mix,
        "r2c_v5_history_temperature": 1.0,
    }
    return {
        "overall_rank": rank,
        "history_mix": history_mix,
        "method_config": config,
        "config_hash": queue.config_hash(config),
    }


def test_build_jobs_is_exactly_two_candidate_d3_validation_pairs() -> None:
    selection = {
        "selected_candidates": [_candidate(1, 0.30), _candidate(2, 0.45)]
    }
    jobs = queue._build_jobs(selection)
    assert len(jobs) == 4
    assert len({job["job_id"] for job in jobs}) == 4
    assert [(job["candidate_position"], job["scenario_id"]) for job in jobs] == [
        (1, "S0"),
        (1, "S4"),
        (2, "S0"),
        (2, "S4"),
    ]
    assert all(job["dataset_id"] == "D3" for job in jobs)
    assert all(job["evaluation_split"] == "validation" for job in jobs)
    assert all(job["full_logging"] is True for job in jobs)
    assert all(
        job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260810
        for job in jobs
    )


def test_phase_d_cannot_load_without_frozen_phase_c_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue, "SCREEN_SELECTION_PATH", tmp_path / "missing.json")
    with pytest.raises(RuntimeError, match="requires a frozen Phase C selection"):
        queue._load_screen_selection()


def test_frozen_baseline_envelope_matches_validation_source() -> None:
    assert queue._load_baseline_envelope() == queue.BASELINE_ENVELOPE


def test_frozen_h1_diversity_thresholds_match_verified_runs() -> None:
    assert queue._load_h1_diversity_thresholds() == queue.H1_DIVERSITY_THRESHOLDS


def test_metric_rule_accepts_four_wins_and_three_plus_close() -> None:
    envelope = {
        "s0_last50_accuracy": 0.89,
        "s4_last50_accuracy": 0.89,
        "s4_recovery_deficit_auc20": 0.001,
        "s4_algorithm_tta_s": 200.0,
    }
    four = queue._evaluate_metric_rule(
        {
            "s0_last50_accuracy": 0.90,
            "s4_last50_accuracy": 0.90,
            "s4_recovery_deficit_auc20": 0.0,
            "s4_algorithm_tta_s": 190.0,
        },
        envelope,
    )
    assert four["strict_passes"] == 4
    assert four["metric_authorized"] is True
    close = queue._evaluate_metric_rule(
        {
            "s0_last50_accuracy": 0.90,
            "s4_last50_accuracy": 0.889,
            "s4_recovery_deficit_auc20": 0.0,
            "s4_algorithm_tta_s": 190.0,
        },
        envelope,
    )
    assert close["strict_passes"] == 3
    assert close["sole_miss"] == "s4_last50_accuracy"
    assert close["metric_authorized"] is True


def test_metric_rule_rejects_two_close_misses() -> None:
    result = queue._evaluate_metric_rule(
        {
            "s0_last50_accuracy": 0.889,
            "s4_last50_accuracy": 0.889,
            "s4_recovery_deficit_auc20": 0.0,
            "s4_algorithm_tta_s": 190.0,
        },
        {
            "s0_last50_accuracy": 0.89,
            "s4_last50_accuracy": 0.89,
            "s4_recovery_deficit_auc20": 0.001,
            "s4_algorithm_tta_s": 200.0,
        },
    )
    assert result["strict_passes"] == 2
    assert result["metric_authorized"] is False


def test_diversity_gate_requires_strict_improvement_in_both_scenarios() -> None:
    observed = {
        scenario: {
            "final_participation_jfi": values["final_participation_jfi"] + 0.001,
            "final_worst10_participation": values[
                "final_worst10_participation"
            ]
            + 0.1,
        }
        for scenario, values in queue.H1_DIVERSITY_THRESHOLDS.items()
    }
    assert all(queue._diversity_checks(observed).values())
    observed["S4"]["final_worst10_participation"] = queue.H1_DIVERSITY_THRESHOLDS[
        "S4"
    ]["final_worst10_participation"]
    assert queue._diversity_checks(observed)["s4_final_worst10_participation"] is False


def test_partial_phase_d_run_uses_new_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue, "RUN_ROOT", tmp_path)
    base = "A-R2C-D3-S4-V6DIV-FULLVAL-C1-R1-L0300-s20260810"
    (tmp_path / base).mkdir()
    (tmp_path / f"{base}-a2").mkdir()
    actual, retry_of = queue._actual_run_id(base)
    assert actual == f"{base}-a3"
    assert retry_of == base
