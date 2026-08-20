from __future__ import annotations

from pathlib import Path

import pytest

from r2c_baselines import r2c_d3_v6_formal_queue as queue


def _winner() -> dict[str, object]:
    config = {
        "r2c_protocol_version": queue.PROTOCOL_VERSION,
        "r2c_v2_audit_replay": False,
        "r2c_v3_fixed_server_alpha": 1.0,
        "r2c_v4_deployment_ema_betas": [0.9],
        "r2c_v4_primary_deployment_beta": 0.9,
        "r2c_v5_history_mix": 0.30,
        "r2c_v5_history_temperature": 1.0,
    }
    return {
        "phase_c_overall_rank": 1,
        "history_mix": 0.30,
        "formal_authorized": True,
        "diversity_preserved": True,
        "method_config": config,
        "config_hash": queue.config_hash(config),
    }


def test_build_jobs_is_exactly_one_d3_test_pair_with_matched_seed() -> None:
    jobs = queue._build_jobs(_winner())
    assert len(jobs) == 2
    assert [job["scenario_id"] for job in jobs] == ["S0", "S4"]
    assert len({job["job_id"] for job in jobs}) == 2
    assert all(job["dataset_id"] == "D3" and job["evaluation_split"] == "test" for job in jobs)
    assert all(job["full_logging"] is True for job in jobs)
    assert all(
        job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260811
        for job in jobs
    )
    assert all(job["method_config"]["r2c_v2_audit_replay"] is True for job in jobs)
    assert all(job["method_config"]["r2c_protocol_version"] == queue.PROTOCOL_VERSION for job in jobs)


def test_formal_config_changes_only_audit_replay() -> None:
    winner = _winner()
    original = dict(winner["method_config"])
    formal = queue._formal_config(winner)
    assert original["r2c_v2_audit_replay"] is False
    assert formal["r2c_v2_audit_replay"] is True
    formal_without = dict(formal)
    formal_without["r2c_v2_audit_replay"] = False
    assert formal_without == original


def test_formal_cannot_load_without_phase_d_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue, "FULLVAL_RESULT_PATH", tmp_path / "missing_result.json")
    monkeypatch.setattr(queue, "FULLVAL_MANIFEST_PATH", tmp_path / "missing_manifest.json")
    with pytest.raises(RuntimeError, match="requires a frozen Phase D result and manifest"):
        queue._load_winner()


def test_external_thresholds_match_frozen_table1_source() -> None:
    lineage = queue._verify_frozen_thresholds()
    assert set(lineage) == set(queue.THRESHOLDS)
    for name, value in queue.THRESHOLDS.items():
        assert lineage[name]["threshold"] == pytest.approx(value, abs=1.0e-12)


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


def test_termination_accepts_three_wins_and_one_close_miss() -> None:
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


def test_termination_rejects_two_misses_even_when_close() -> None:
    observed = {
        "s0_last50_accuracy": queue.THRESHOLDS["s0_last50_accuracy"] - 0.001,
        "s4_last50_accuracy": queue.THRESHOLDS["s4_last50_accuracy"] - 0.001,
        "s4_recovery_deficit_auc20": 0.0,
        "s4_algorithm_tta_s": queue.THRESHOLDS["s4_algorithm_tta_s"] - 1.0,
    }
    result = queue._evaluate_termination(observed)
    assert result["strict_passes"] == 2
    assert result["goal_met"] is False


def test_partial_formal_run_uses_new_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(queue, "RUN_ROOT", tmp_path)
    base = "A-R2C-D3-S4-V6DIV-FORMAL-R1-L0300-s20260811"
    (tmp_path / base).mkdir()
    actual, retry_of = queue._actual_run_id(base)
    assert actual == f"{base}-a2"
    assert retry_of == base
