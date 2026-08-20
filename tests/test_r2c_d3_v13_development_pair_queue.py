from __future__ import annotations

import json

from r2c_baselines import r2c_d3_v13_development_pair_queue as queue
from r2c_baselines.r2c_v7 import PROTOCOL_VERSION as V11_PROTOCOL_VERSION
from r2c_baselines.r2c_v13 import PROTOCOL_VERSION


def _context(schedule_id: str = "DARE-L5") -> dict[str, object]:
    return {
        "schedule_id": schedule_id,
        "selected_run_id": f"selected-{schedule_id}",
    }


def test_v11_and_v13_configs_are_learning_matched() -> None:
    v11 = queue._v11_config()
    v13 = queue._v13_config("DARE-C5")
    assert v11["r2c_protocol_version"] == V11_PROTOCOL_VERSION
    assert v13["r2c_protocol_version"] == PROTOCOL_VERSION
    assert v11["r2c_v4_deployment_ema_betas"] == [0.95]
    assert v13["r2c_v4_deployment_ema_betas"] == [0.95]
    assert v11["r2c_v4_primary_deployment_beta"] == 0.95
    assert v13["r2c_v4_primary_deployment_beta"] == 0.95
    assert v11["r2c_v2_audit_replay"] is False
    assert v13["r2c_v2_audit_replay"] is False
    assert v11["r2c_v7_trigger_deployment_beta"] == 1.0
    assert "r2c_v7_trigger_deployment_beta" not in v13
    assert v13["r2c_v13_schedule_id"] == "DARE-C5"

    v11_learning = dict(v11)
    v13_learning = dict(v13)
    for config in (v11_learning, v13_learning):
        config.pop("r2c_protocol_version", None)
    v11_learning.pop("r2c_v7_trigger_deployment_beta", None)
    v13_learning.pop("r2c_v13_schedule_id", None)
    v13_learning.pop("r2c_v13_plan_id", None)
    assert v11_learning == v13_learning


def test_job_matrix_is_exactly_matched_v11_then_v13() -> None:
    context = _context("DARE-L8")
    jobs = [queue._job(method, scenario, context) for method, scenario in queue.METHOD_ORDER]
    assert [(job["method_version"], job["scenario_id"]) for job in jobs] == [
        ("v11", "S0"),
        ("v11", "S4"),
        ("v13", "S0"),
        ("v13", "S4"),
    ]
    assert len({job["job_id"] for job in jobs}) == 4
    for job in jobs:
        assert job["dataset_id"] == "D3"
        assert job["rounds"] == 1000
        assert job["evaluation_split"] == "validation"
        assert job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260809
        assert job["selected_schedule_id"] == "DARE-L8"
        assert job["formal_test_access"] is False
        assert job["other_dataset_access"] is False
        assert job["test_labels_used_for_selection"] is False


def test_gate_accepts_four_wins() -> None:
    result = queue.evaluate_gate(
        v11_s0_last50=0.90,
        v13_s0_last50=0.901,
        v11_s4_last50=0.89,
        v13_s4_last50=0.90,
        v11_s4_pta20=88.0,
        v13_s4_pta20=89.0,
        v11_s4_tta_s=100.0,
        v13_s4_tta_s=90.0,
    )
    assert result["strict_win_count"] == 4
    assert result["performance_gate_passed"] is True


def test_gate_accepts_three_wins_and_sole_close_identity() -> None:
    result = queue.evaluate_gate(
        v11_s0_last50=0.90,
        v13_s0_last50=0.90,
        v11_s4_last50=0.89,
        v13_s4_last50=0.90,
        v11_s4_pta20=88.0,
        v13_s4_pta20=89.0,
        v11_s4_tta_s=100.0,
        v13_s4_tta_s=90.0,
    )
    assert result["strict_win_count"] == 3
    assert result["misses"] == ["s0_last50_accuracy"]
    assert result["sole_miss_close"] is True
    assert result["performance_gate_passed"] is True


def test_gate_rejects_far_sole_miss_or_only_two_wins() -> None:
    far = queue.evaluate_gate(
        v11_s0_last50=0.90,
        v13_s0_last50=0.89849,
        v11_s4_last50=0.89,
        v13_s4_last50=0.90,
        v11_s4_pta20=88.0,
        v13_s4_pta20=89.0,
        v11_s4_tta_s=100.0,
        v13_s4_tta_s=90.0,
    )
    assert far["strict_win_count"] == 3
    assert far["sole_miss_close"] is False
    assert far["performance_gate_passed"] is False

    two = queue.evaluate_gate(
        v11_s0_last50=0.90,
        v13_s0_last50=0.90,
        v11_s4_last50=0.89,
        v13_s4_last50=0.89,
        v11_s4_pta20=88.0,
        v13_s4_pta20=89.0,
        v11_s4_tta_s=100.0,
        v13_s4_tta_s=90.0,
    )
    assert two["strict_win_count"] == 2
    assert two["performance_gate_passed"] is False


def test_missing_tta_is_never_a_win_or_close() -> None:
    result = queue.evaluate_gate(
        v11_s0_last50=0.90,
        v13_s0_last50=0.901,
        v11_s4_last50=0.89,
        v13_s4_last50=0.90,
        v11_s4_pta20=88.0,
        v13_s4_pta20=89.0,
        v11_s4_tta_s=100.0,
        v13_s4_tta_s=None,
    )
    assert result["strict_wins"]["s4_algorithm_tta"] is False
    assert result["within_close_margin"]["s4_algorithm_tta"] is False
    assert result["performance_gate_passed"] is False


def test_screen_context_hard_blocks_before_terminal_result(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    state = tmp_path / "state.json"
    result = tmp_path / "missing-result.json"
    manifest.write_text(json.dumps({}), encoding="utf-8")
    state.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(queue, "SCREEN_MANIFEST_PATH", manifest)
    monkeypatch.setattr(queue, "SCREEN_STATE_PATH", state)
    monkeypatch.setattr(queue, "SCREEN_RESULT_PATH", result)
    try:
        queue._screen_context()
    except RuntimeError as exc:
        assert "not terminal" in str(exc)
    else:
        raise AssertionError("M2 context must hard-block before terminal M1 selection")


def test_identity_columns_cover_global_evaluation_and_deployment_hashes() -> None:
    assert queue.IDENTITY_COLUMNS == (
        "round",
        "global_model_hash",
        "evaluation_model_hash",
        "primary_deployment_model_hash_before",
        "primary_deployment_model_hash_after",
    )
