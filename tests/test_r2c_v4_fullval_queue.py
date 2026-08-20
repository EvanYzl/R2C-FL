from __future__ import annotations

from r2c_baselines.r2c_d2_v4_fullval_queue import _evaluate_authorization


def test_v4_fullval_requires_strict_or_one_close_miss() -> None:
    strict = _evaluate_authorization(
        {
            "s0_last50_accuracy": 0.67,
            "s4_last50_accuracy": 0.67,
            "s4_recovery_deficit_auc20": 0.001,
            "s4_algorithm_tta_s": 450.0,
        }
    )
    assert strict["strict_passes"] == 4 and strict["formal_authorized"]
    close = _evaluate_authorization(
        {
            "s0_last50_accuracy": 0.67,
            "s4_last50_accuracy": 0.67,
            "s4_recovery_deficit_auc20": 0.0024,
            "s4_algorithm_tta_s": 450.0,
        }
    )
    assert close["strict_passes"] == 3 and close["formal_authorized"]
    far = _evaluate_authorization(
        {
            "s0_last50_accuracy": 0.67,
            "s4_last50_accuracy": 0.67,
            "s4_recovery_deficit_auc20": 0.008,
            "s4_algorithm_tta_s": 450.0,
        }
    )
    assert not far["formal_authorized"]
