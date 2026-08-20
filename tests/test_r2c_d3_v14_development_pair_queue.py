from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from r2c_baselines import r2c_d3_v14_development_pair_queue as queue
from r2c_baselines import r2c_d3_v14_validation_screen_erratum_queue as m1_authority
from r2c_baselines.r2c_v7 import PROTOCOL_VERSION as V11_PROTOCOL_VERSION
from r2c_baselines.r2c_v14 import FAST_BETA, PROTOCOL_VERSION, WARMUP_ROUNDS


def _context(candidate_id: str = "CMTR-B975-R20") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "selected_run_id": f"selected-{candidate_id}",
    }


def _gate(**overrides: float | None) -> dict[str, object]:
    values: dict[str, float | None] = {
        "v11_s0_last50": 0.900,
        "v14_s0_last50": 0.901,
        "v11_s4_last50": 0.890,
        "v14_s4_last50": 0.900,
        "v11_s4_lqa20": 88.0,
        "v14_s4_lqa20": 89.0,
        "v11_s4_tta_s": 100.0,
        "v14_s4_tta_s": 90.0,
    }
    values.update(overrides)
    return queue.evaluate_gate(**values)  # type: ignore[arg-type]


def test_v11_and_v14_configs_preserve_exact_learning_configuration() -> None:
    v11 = queue._v11_config()
    v14 = queue._v14_config("CMTR-B975-R20")
    assert v11["r2c_protocol_version"] == V11_PROTOCOL_VERSION
    assert v14["r2c_protocol_version"] == PROTOCOL_VERSION
    assert v11["r2c_v4_deployment_ema_betas"] == [0.95]
    assert v14["r2c_v4_deployment_ema_betas"] == [FAST_BETA, 0.975]
    assert v14["r2c_v4_primary_deployment_beta"] == 0.975
    assert v14["r2c_v14_warmup_rounds"] == WARMUP_ROUNDS
    assert v11["r2c_v7_trigger_deployment_beta"] == 1.0
    assert "r2c_v7_trigger_deployment_beta" not in v14
    assert queue._learning_config(v11) == queue._learning_config(v14)


def test_job_matrix_is_exactly_v11_s0_s4_then_locked_v14_s0_s4() -> None:
    context = _context("CMTR-B950-R20")
    jobs = [queue._job(method, scenario, context) for method, scenario in queue.METHOD_ORDER]
    assert [(job["method_version"], job["scenario_id"]) for job in jobs] == [
        ("v11", "S0"),
        ("v11", "S4"),
        ("v14", "S0"),
        ("v14", "S4"),
    ]
    assert len({job["job_id"] for job in jobs}) == 4
    for job in jobs:
        assert job["dataset_id"] == "D3"
        assert job["rounds"] == 1000
        assert job["evaluation_split"] == "validation"
        assert job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260809
        assert job["selected_candidate_id"] == "CMTR-B950-R20"
        assert job["formal_test_access"] is False
        assert job["other_dataset_access"] is False
        assert job["test_labels_used_for_selection"] is False


def test_gate_accepts_four_strict_wins() -> None:
    result = _gate()
    assert result["strict_win_count"] == 4
    assert result["performance_gate_passed"] is True


def test_gate_accepts_exactly_three_wins_and_sole_close_equality() -> None:
    result = _gate(v14_s0_last50=0.900)
    assert result["strict_win_count"] == 3
    assert result["misses"] == ["s0_last50_accuracy"]
    assert result["sole_miss_close"] is True
    assert result["strict_wins"]["s0_last50_accuracy"] is False
    assert result["performance_gate_passed"] is True


def test_gate_rejects_far_sole_miss_and_only_two_wins() -> None:
    far = _gate(v14_s0_last50=0.89849)
    assert far["strict_win_count"] == 3
    assert far["sole_miss_close"] is False
    assert far["performance_gate_passed"] is False
    two = _gate(v14_s0_last50=0.900, v14_s4_last50=0.890)
    assert two["strict_win_count"] == 2
    assert two["performance_gate_passed"] is False


def test_missing_tta_is_never_a_win_or_close() -> None:
    result = _gate(v14_s4_tta_s=None)
    assert result["strict_wins"]["s4_algorithm_tta"] is False
    assert result["within_close_margin"]["s4_algorithm_tta"] is False
    assert result["performance_gate_passed"] is False


def test_learning_identity_is_exact_and_checked_per_scenario() -> None:
    control = pd.DataFrame({"round": [1, 2], "global_model_hash": ["a", "b"]})
    identical = control.copy()
    drifted = pd.DataFrame({"round": [1, 2], "global_model_hash": ["a", "c"]})
    assert all(queue._learning_identity(control, identical).values())
    assert queue._learning_identity(control, drifted)["global_model_hash"] is False
    assert queue.LEARNING_IDENTITY_COLUMNS == ("round", "global_model_hash")


def test_m1_context_hard_blocks_before_terminal_result(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "m1-manifest.json"
    state = tmp_path / "m1-state.json"
    result = tmp_path / "missing-result.json"
    manifest.write_text(json.dumps({}), encoding="utf-8")
    state.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(queue, "M1_MANIFEST_PATH", manifest)
    monkeypatch.setattr(queue, "M1_STATE_PATH", state)
    monkeypatch.setattr(queue, "M1_RESULT_PATH", result)
    with pytest.raises(RuntimeError, match="not terminal"):
        queue._m1_context()


def test_m1_paths_bind_only_to_erratum_continuation_authority() -> None:
    assert queue.M1_MANIFEST_PATH == m1_authority.MANIFEST_PATH
    assert queue.M1_STATE_PATH == m1_authority.STATE_PATH
    assert queue.M1_RESULT_PATH == m1_authority.RESULT_PATH
    assert queue.M1_RUNS_PATH == m1_authority.RUNS_PATH
    assert queue.M1_RUNS_CSV_PATH == m1_authority.RUNS_CSV_PATH


def _erratum_payload(run_id: str, *, status: str = "passed") -> dict[str, object]:
    return {
        "status": status,
        "run_id": run_id,
        "audit_erratum_id": m1_authority.ERRATUM_ID,
        "recorded_tables_mutated": False,
        "round_rows": 1000,
        "canonical_window_contract": {
            "before_count": 20,
            "after_count": 20,
            "event_count": 1,
        },
    }


def test_m1_audit_evidence_prefers_passing_erratum_reconcile_log(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "selected-run"
    log_root = tmp_path / "worker_logs"
    log_root.mkdir()
    failed = log_root / f"{run_id}.v14.audit.log"
    passed = log_root / f"{run_id}.reconcile.v14.audit.log"
    failed.write_text(json.dumps({"status": "failed", "run_id": run_id}), encoding="utf-8")
    passed.write_text(json.dumps(_erratum_payload(run_id)), encoding="utf-8")
    monkeypatch.setattr(queue, "QUEUE_ROOT", tmp_path)
    assert queue._validated_m1_erratum_audit_log(run_id) == passed


def test_m1_audit_evidence_rejects_superseded_or_non_erratum_log(
    tmp_path: Path, monkeypatch
) -> None:
    run_id = "selected-run"
    log_root = tmp_path / "worker_logs"
    log_root.mkdir()
    old_log = log_root / f"{run_id}.v14.audit.log"
    old_log.write_text(json.dumps({"status": "passed", "run_id": run_id}), encoding="utf-8")
    monkeypatch.setattr(queue, "QUEUE_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="no passing erratum audit evidence"):
        queue._validated_m1_erratum_audit_log(run_id)


def test_v14_s4_uses_erratum_auditor_but_s0_remains_on_frozen_auditor() -> None:
    assert queue._audit_script_name({"method_version": "v11", "scenario_id": "S4"}) == (
        "audit_r2c_run.py"
    )
    assert queue._audit_script_name({"method_version": "v14", "scenario_id": "S0"}) == (
        "audit_r2c_v14_run.py"
    )
    assert queue._audit_script_name({"method_version": "v14", "scenario_id": "S4"}) == (
        "audit_r2c_v14_run_erratum.py"
    )
    assert queue._audit_script_name(
        {"method_version": "v14", "scenario_id": "S4", "rounds": 500}
    ) == "audit_r2c_v14_run.py"


def test_build_hard_gate_precedes_assets_and_all_m2_writes(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "m2-manifest.json"
    state = tmp_path / "m2-state.json"
    assets_called = False

    def blocked() -> dict[str, object]:
        raise RuntimeError("M1 blocked")

    def assets() -> dict[str, str]:
        nonlocal assets_called
        assets_called = True
        return {}

    monkeypatch.setattr(queue, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(queue, "STATE_PATH", state)
    monkeypatch.setattr(queue, "_m1_context", blocked)
    monkeypatch.setattr(queue, "_ensure_assets", assets)
    with pytest.raises(RuntimeError, match="M1 blocked"):
        queue.build_manifest(persist=True)
    assert assets_called is False
    assert manifest.exists() is False
    assert state.exists() is False


def test_registered_metric_is_lqa20_not_zero_saturated_tail_deficit() -> None:
    context = _context()
    jobs = [queue._job(method, scenario, context) for method, scenario in queue.METHOD_ORDER]
    manifest = {
        "gate_metrics": "S0 last50; S4 last50; S4 post-event LQA20; S4 algorithm TTA",
        "jobs": jobs,
    }
    assert "LQA20" in manifest["gate_metrics"]
    assert "PTA20" not in manifest["gate_metrics"]
    assert queue.GATE_MARGINS["s4_lqa20_pp"] == 0.50
