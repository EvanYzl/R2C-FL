from __future__ import annotations

import pytest

from r2c_baselines import r2c_d3_v7_phase_f_queue as queue


def test_phase_f_build_is_closed_before_phase_e_authorization() -> None:
    if queue.PHASE_E_RESULT_PATH.exists():
        pytest.skip("Phase E has already reached a terminal result")
    with pytest.raises(RuntimeError, match="Phase E result does not exist"):
        queue.build_manifest(persist=False)


def test_phase_f_manifest_is_exact_validation_pair_when_gate_is_mock_authorized(
    monkeypatch,
) -> None:
    monkeypatch.setattr(queue, "_verify_phase_e_result", lambda: {"phase_f_authorized": True})
    monkeypatch.setattr(queue, "_source_hashes", lambda: {"phase_e": "frozen"})
    monkeypatch.setattr(queue, "_ensure_assets", lambda: {"assets": "frozen"})
    monkeypatch.setattr(queue, "_implementation_hashes", lambda: {"implementation": "frozen"})
    manifest = queue.build_manifest(persist=False)
    jobs = manifest["jobs"]
    assert len(jobs) == 2
    assert [job["scenario_id"] for job in jobs] == ["S0", "S4"]
    assert [job["job_id"] for job in jobs] == manifest["job_order"]
    assert manifest["formal_test_access"] is False
    assert manifest["other_dataset_access"] is False
    for job in jobs:
        assert job["dataset_id"] == "D3"
        assert job["evaluation_split"] == "validation"
        assert job["rounds"] == 1000
        assert job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260810
        assert job["method_config"]["r2c_protocol_version"] == queue.PROTOCOL_VERSION
        assert job["method_config"]["r2c_v2_audit_replay"] is False


def test_phase_f_metric_rule_accepts_four_strict_wins() -> None:
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


def test_phase_f_metric_rule_requires_sole_miss_to_be_close() -> None:
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
