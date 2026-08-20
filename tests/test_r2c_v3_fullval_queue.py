from __future__ import annotations

from r2c_baselines.r2c_d2_v3_fullval_queue import (
    TARGET_ACCURACY,
    _evaluate_authorization,
    build_manifest,
)


def test_v3_full_validation_is_one_frozen_pair_without_test_access() -> None:
    manifest = build_manifest(persist=False)
    jobs = manifest["jobs"]
    assert len(jobs) == 2
    assert {job["scenario_id"] for job in jobs} == {"S0", "S4"}
    assert manifest["formal_test_access"] is False
    assert manifest["target_accuracy"] == TARGET_ACCURACY
    for job in jobs:
        assert job["rounds"] == 1000
        assert job["evaluation_split"] == "validation"
        assert job["full_logging"] is True
        assert job["method_config"]["r2c_v3_fixed_server_alpha"] == 0.75


def test_authorization_requires_four_or_three_plus_close() -> None:
    four = _evaluate_authorization(
        {
            "s0_last50_accuracy": 0.67,
            "s4_last50_accuracy": 0.67,
            "s4_recovery_deficit_auc20": 0.002,
            "s4_algorithm_tta_s": 470.0,
        }
    )
    assert four["formal_authorized"] is True and four["strict_passes"] == 4
    close = _evaluate_authorization(
        {
            "s0_last50_accuracy": 0.67,
            "s4_last50_accuracy": 0.67,
            "s4_recovery_deficit_auc20": 0.0025,
            "s4_algorithm_tta_s": 470.0,
        }
    )
    assert close["formal_authorized"] is True and close["strict_passes"] == 3
    far = _evaluate_authorization(
        {
            "s0_last50_accuracy": 0.67,
            "s4_last50_accuracy": 0.67,
            "s4_recovery_deficit_auc20": 0.003,
            "s4_algorithm_tta_s": 470.0,
        }
    )
    assert far["formal_authorized"] is False
