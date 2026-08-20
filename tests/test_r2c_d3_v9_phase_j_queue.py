from __future__ import annotations

import pandas as pd

from r2c_baselines import r2c_d3_v9_phase_j_queue as queue


def _frozen_config() -> dict[str, object]:
    return {
        "r2c_protocol_version": queue.PROTOCOL_VERSION,
        "r2c_v2_audit_replay": False,
        "r2c_v7_trigger_deployment_beta": 1.0,
    }


def test_phase_j_manifest_is_exact_validation_pair_when_phase_i_authorized(
    monkeypatch,
) -> None:
    monkeypatch.setattr(queue, "_verify_phase_i_result", lambda: {"phase_j_authorized": True})
    monkeypatch.setattr(queue, "_source_hashes", lambda: {"phase_i": "frozen"})
    monkeypatch.setattr(queue, "_ensure_assets", lambda: {"assets": "frozen"})
    monkeypatch.setattr(queue, "_implementation_hashes", lambda: {"implementation": "frozen"})
    monkeypatch.setattr(queue, "_frozen_config", _frozen_config)
    manifest = queue.build_manifest(persist=False)
    jobs = manifest["jobs"]
    assert len(jobs) == 2
    assert [job["scenario_id"] for job in jobs] == ["S0", "S4"]
    assert [job["job_id"] for job in jobs] == manifest["job_order"]
    assert manifest["formal_test_access"] is False
    assert manifest["other_dataset_access"] is False
    assert manifest["quarantine_contract"]["s4_trigger_round"] == 500
    assert manifest["quarantine_contract"]["trigger_deployment_beta"] == 1.0
    for job in jobs:
        assert job["dataset_id"] == "D3"
        assert job["evaluation_split"] == "validation"
        assert job["rounds"] == 1000
        assert job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260810
        assert job["method_config"]["r2c_protocol_version"] == queue.PROTOCOL_VERSION
        assert job["method_config"]["r2c_v2_audit_replay"] is False
        assert job["method_config"]["r2c_v7_trigger_deployment_beta"] == 1.0


def test_phase_j_metric_rule_accepts_four_strict_wins() -> None:
    observed = {
        "s0_last50_accuracy": queue.BASELINE_ENVELOPE["s0_last50_accuracy"] + 0.001,
        "s4_last50_accuracy": queue.BASELINE_ENVELOPE["s4_last50_accuracy"] + 0.001,
        "s4_recovery_deficit_auc20": queue.BASELINE_ENVELOPE[
            "s4_recovery_deficit_auc20"
        ]
        - 0.00001,
        "s4_algorithm_tta_s": queue.BASELINE_ENVELOPE["s4_algorithm_tta_s"] - 1.0,
    }
    result = queue._metric_rule(observed)
    assert result["strict_passes"] == 4
    assert result["sole_miss"] is None
    assert result["metric_authorized"] is True


def test_phase_j_metric_rule_requires_sole_miss_to_be_close() -> None:
    envelope = queue.BASELINE_ENVELOPE
    close = {
        "s0_last50_accuracy": envelope["s0_last50_accuracy"] + 0.001,
        "s4_last50_accuracy": envelope["s4_last50_accuracy"] + 0.001,
        "s4_recovery_deficit_auc20": envelope["s4_recovery_deficit_auc20"] + 0.00005,
        "s4_algorithm_tta_s": envelope["s4_algorithm_tta_s"] - 1.0,
    }
    result = queue._metric_rule(close)
    assert result["strict_passes"] == 3
    assert result["sole_miss"] == "s4_recovery_deficit_auc20"
    assert result["metric_authorized"] is True

    far = dict(close)
    far["s4_recovery_deficit_auc20"] = (
        envelope["s4_recovery_deficit_auc20"] + queue.AUC_CLOSE + 0.00001
    )
    failed = queue._metric_rule(far)
    assert failed["strict_passes"] == 3
    assert failed["metric_authorized"] is False


def _contract_row(scenario: str) -> pd.Series:
    triggered = scenario == "S4"
    return pd.Series(
        {
            "trigger_count": int(triggered),
            "trigger_rounds_json": "[500]" if triggered else "[]",
            "quarantine_count": int(triggered),
            "quarantine_rounds_json": "[500]" if triggered else "[]",
            "response_count": int(triggered),
            "synchronization_count": 0,
            "quarantine_matches_trigger": True,
            "response_matches_trigger": True,
            "action_matches_trigger": True,
            "configured_beta_is_one": True,
            "effective_beta_matches_contract": True,
            "trigger_deployment_hash_held": True,
            "trigger_global_training_advanced": True,
            "forbidden_input_clean": True,
        }
    )


def test_phase_j_quarantine_rule_requires_the_complete_structural_contract() -> None:
    checks = queue._quarantine_rule(_contract_row("S0"), _contract_row("S4"))
    assert all(checks.values())

    broken = _contract_row("S4")
    broken["trigger_deployment_hash_held"] = False
    failed = queue._quarantine_rule(_contract_row("S0"), broken)
    assert failed["s4_trigger_deployment_hash_held"] is False
    assert all(value for key, value in failed.items() if key != "s4_trigger_deployment_hash_held")
