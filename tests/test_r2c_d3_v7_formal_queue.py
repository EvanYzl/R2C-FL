from __future__ import annotations

import pytest

from r2c_baselines import r2c_d3_v7_formal_queue as queue


def _validation_result() -> dict[str, object]:
    return {
        "formal_authorized": True,
        "method_config": {
            "r2c_protocol_version": queue.PROTOCOL_VERSION,
            "r2c_v2_audit_replay": False,
            "r2c_v5_history_mix": 0.45,
            "r2c_v4_deployment_ema_betas": [0.9],
        },
    }


def test_formal_build_is_closed_before_phase_f_authorization() -> None:
    if queue.PHASE_F_RESULT_PATH.exists():
        pytest.skip("Phase F has already reached a terminal result")
    with pytest.raises(RuntimeError, match="Phase F result does not exist"):
        queue.build_manifest(persist=False)


def test_formal_manifest_is_exact_matched_seed_pair_when_mock_authorized(monkeypatch) -> None:
    phase_f = _validation_result()
    monkeypatch.setattr(queue, "_verify_phase_f_result", lambda: phase_f)
    monkeypatch.setattr(queue, "_source_hashes", lambda: {"phase_f": "frozen"})
    monkeypatch.setattr(queue, "_ensure_assets", lambda: {"assets": "frozen"})
    monkeypatch.setattr(queue, "_implementation_hashes", lambda: {"implementation": "frozen"})
    manifest = queue.build_manifest(persist=False)
    jobs = manifest["jobs"]
    assert len(jobs) == 2
    assert [job["scenario_id"] for job in jobs] == ["S0", "S4"]
    assert [job["job_id"] for job in jobs] == manifest["job_order"]
    assert manifest["formal_test_access"] is True
    assert manifest["other_dataset_access"] is False
    assert manifest["engineering_reevaluation"] is True
    assert manifest["validation_to_formal_changed_keys"] == ["r2c_v2_audit_replay"]
    for job in jobs:
        assert job["dataset_id"] == "D3"
        assert job["mode"] == "formal"
        assert job["evaluation_split"] == "test"
        assert job["rounds"] == 1000
        assert job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260811
        assert job["method_config"]["r2c_v2_audit_replay"] is True
        assert job["formal_interpretation"].startswith("matched_seed_engineering_reevaluation")


def test_formal_config_changes_only_audit_replay() -> None:
    validation = _validation_result()
    formal = queue._formal_config(validation)
    assert formal["r2c_v2_audit_replay"] is True
    for key, value in validation["method_config"].items():
        if key != "r2c_v2_audit_replay":
            assert formal[key] == value


def test_formal_metric_rule_accepts_four_wins_or_one_close_miss() -> None:
    thresholds = queue.FORMAL_THRESHOLDS
    wins = {
        "s0_last50_accuracy": thresholds["s0_last50_accuracy"] + 0.001,
        "s4_last50_accuracy": thresholds["s4_last50_accuracy"] + 0.001,
        "s4_recovery_deficit_auc20": thresholds["s4_recovery_deficit_auc20"] - 0.00001,
        "s4_algorithm_tta_s": thresholds["s4_algorithm_tta_s"] - 1.0,
    }
    strict = queue._metric_rule(wins)
    assert strict["strict_passes"] == 4
    assert strict["metric_goal_met"] is True

    close = dict(wins)
    close["s4_recovery_deficit_auc20"] = (
        thresholds["s4_recovery_deficit_auc20"] + 0.5 * queue.AUC_CLOSE
    )
    near = queue._metric_rule(close)
    assert near["strict_passes"] == 3
    assert near["sole_miss"] == "s4_recovery_deficit_auc20"
    assert near["metric_goal_met"] is True

    close["s4_recovery_deficit_auc20"] = (
        thresholds["s4_recovery_deficit_auc20"] + 1.1 * queue.AUC_CLOSE
    )
    far = queue._metric_rule(close)
    assert far["strict_passes"] == 3
    assert far["metric_goal_met"] is False
