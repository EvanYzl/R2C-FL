from __future__ import annotations

from r2c_baselines.r2c_d2_v4_formal_queue import _evaluate_termination


def test_v4_formal_termination_rule() -> None:
    strict = _evaluate_termination(
        {
            "s0_last50_accuracy": 0.70,
            "s4_last50_accuracy": 0.70,
            "s4_recovery_deficit_auc20": 0.001,
            "s4_algorithm_tta_s": 450.0,
        }
    )
    assert strict["strict_passes"] == 4 and strict["termination_condition_met"]
    close = _evaluate_termination(
        {
            "s0_last50_accuracy": 0.70,
            "s4_last50_accuracy": 0.70,
            "s4_recovery_deficit_auc20": 0.0024,
            "s4_algorithm_tta_s": 450.0,
        }
    )
    assert close["strict_passes"] == 3 and close["termination_condition_met"]
    far = _evaluate_termination(
        {
            "s0_last50_accuracy": 0.70,
            "s4_last50_accuracy": 0.70,
            "s4_recovery_deficit_auc20": 0.008,
            "s4_algorithm_tta_s": 450.0,
        }
    )
    assert not far["termination_condition_met"]
