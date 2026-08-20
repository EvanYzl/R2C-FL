from __future__ import annotations

import json

import pytest

from r2c_baselines import r2c_d3_v11_single_seed_fullval_queue as queue


def test_single_seed_selection_freezes_beta_095_from_validation_only_evidence():
    evidence = queue._selection_evidence()
    assert evidence["selection_seed"] == 20260808
    assert evidence["source_run_id"] == queue.V10_RUN_ID
    assert evidence["selected_beta"] == pytest.approx(0.95)
    assert set(evidence["eligible_betas"]) == {0.9, 0.925, 0.95}
    assert evidence["recovery_deficit_auc20"] == pytest.approx(0.0)
    assert evidence["target_hit_round"] - evidence["control_target_hit_round"] == 7
    assert evidence["test_labels_used"] is False


def test_frozen_config_changes_only_deployment_beta_fields():
    config = queue._frozen_config()
    source = json.loads(queue.V9_PHASE_J_INITIAL_MANIFEST_PATH.read_text(encoding="utf-8"))[
        "jobs"
    ][0]["method_config"]
    beta_keys = {"r2c_v4_deployment_ema_betas", "r2c_v4_primary_deployment_beta"}
    assert {key: value for key, value in config.items() if key not in beta_keys} == {
        key: value for key, value in source.items() if key not in beta_keys
    }
    assert config["r2c_v4_deployment_ema_betas"] == [0.95]
    assert config["r2c_v4_primary_deployment_beta"] == pytest.approx(0.95)
    assert config["r2c_v7_trigger_deployment_beta"] == pytest.approx(1.0)
    assert config["r2c_v2_audit_replay"] is False


def test_manifest_is_exactly_two_validation_jobs_on_one_seed():
    manifest = queue.build_manifest(persist=False)
    assert manifest["single_seed_only"] is True
    assert manifest["formal_test_access"] is False
    assert manifest["other_dataset_access"] is False
    assert manifest["review_required_after_completion"] is True
    assert manifest["scenario_order"] == ["S0", "S4"]
    assert len(manifest["jobs"]) == 2
    assert {job["scenario_id"] for job in manifest["jobs"]} == {"S0", "S4"}
    for job in manifest["jobs"]:
        assert job["dataset_id"] == "D3"
        assert job["evaluation_split"] == "validation"
        assert job["rounds"] == 1000
        assert {job["seed"], job["partition_seed"], job["trace_seed"]} == {20260810}
        assert job["test_labels_used_for_selection"] is False
        assert job["method_config"]["r2c_v4_deployment_ema_betas"] == [0.95]


def _winning_observed() -> dict[str, float]:
    envelope = queue.BASELINE_ENVELOPE
    return {
        "s0_last50_accuracy": envelope["s0_last50_accuracy"] + 0.001,
        "s4_last50_accuracy": envelope["s4_last50_accuracy"] + 0.001,
        "s4_recovery_deficit_auc20": envelope["s4_recovery_deficit_auc20"] - 0.00001,
        "s4_algorithm_tta_s": envelope["s4_algorithm_tta_s"] - 1.0,
    }


def test_metric_rule_accepts_four_wins_and_three_plus_close():
    four = queue._metric_rule(_winning_observed())
    assert four["strict_passes"] == 4
    assert four["metric_goal_met"] is True

    close = _winning_observed()
    close["s4_recovery_deficit_auc20"] = (
        queue.BASELINE_ENVELOPE["s4_recovery_deficit_auc20"] + 0.5 * queue.AUC_CLOSE
    )
    near = queue._metric_rule(close)
    assert near["strict_passes"] == 3
    assert near["sole_miss"] == "s4_recovery_deficit_auc20"
    assert near["close_checks"]["s4_recovery_deficit_auc20"] is True
    assert near["metric_goal_met"] is True


def test_metric_rule_rejects_non_close_miss_and_never_authorizes_formal():
    far = _winning_observed()
    far["s4_recovery_deficit_auc20"] = (
        queue.BASELINE_ENVELOPE["s4_recovery_deficit_auc20"] + 1.1 * queue.AUC_CLOSE
    )
    decision = queue._metric_rule(far)
    assert decision["strict_passes"] == 3
    assert decision["close_checks"]["s4_recovery_deficit_auc20"] is False
    assert decision["metric_goal_met"] is False

    manifest = queue.build_manifest(persist=False)
    assert manifest["formal_test_access"] is False
    assert all(job["mode"] == "calibration" for job in manifest["jobs"])
