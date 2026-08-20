from __future__ import annotations

import json

import pandas as pd
import pytest

from r2c_baselines import r2c_d3_v14_formal_confirmation_queue as queue
from r2c_baselines.r2c_v7 import PROTOCOL_VERSION as V11_PROTOCOL_VERSION
from r2c_baselines.r2c_v14 import PROTOCOL_VERSION


def _context(candidate_id: str = "CMTR-B975-R20") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "manifest": {"frozen_spec_hash": "m2-frozen"},
    }


def _identity_frame(drift: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "round": [1, 2],
            "global_model_hash": ["g1", "drift" if drift else "g2"],
        }
    )


def test_formal_configs_are_learning_matched_and_enable_audit_replay() -> None:
    v11 = queue._v11_config()
    v14 = queue._v14_config("CMTR-B975-R10")
    assert v11["r2c_protocol_version"] == V11_PROTOCOL_VERSION
    assert v14["r2c_protocol_version"] == PROTOCOL_VERSION
    assert v11["r2c_v2_audit_replay"] is True
    assert v14["r2c_v2_audit_replay"] is True
    assert queue.development._learning_config(v11) == queue.development._learning_config(v14)


def test_formal_matrix_is_exactly_v11_then_locked_v14() -> None:
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
        assert job["mode"] == "formal"
        assert job["evaluation_split"] == "test"
        assert job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260812
        assert job["formal_test_access"] is True
        assert job["other_dataset_access"] is False
        assert job["test_labels_used_for_selection"] is False


def test_development_context_blocks_when_m2_authority_is_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(queue, "DEVELOPMENT_MANIFEST_PATH", tmp_path / "missing-manifest.json")
    monkeypatch.setattr(queue, "DEVELOPMENT_STATE_PATH", tmp_path / "missing-state.json")
    monkeypatch.setattr(queue, "DEVELOPMENT_RESULT_PATH", tmp_path / "missing-result.json")
    with pytest.raises(RuntimeError, match="M2 is not terminal"):
        queue._development_context()


def test_development_context_blocks_before_passed_terminal_result(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "m2-manifest.json"
    state = tmp_path / "m2-state.json"
    result = tmp_path / "missing-result.json"
    manifest.write_text(json.dumps({}), encoding="utf-8")
    state.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(queue, "DEVELOPMENT_MANIFEST_PATH", manifest)
    monkeypatch.setattr(queue, "DEVELOPMENT_STATE_PATH", state)
    monkeypatch.setattr(queue, "DEVELOPMENT_RESULT_PATH", result)
    with pytest.raises(RuntimeError, match="M2 is not terminal"):
        queue._development_context()


def test_build_gate_precedes_formal_assets_and_files(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "m3-manifest.json"
    state = tmp_path / "m3-state.json"
    assets_called = False

    def blocked() -> dict[str, object]:
        raise RuntimeError("M2 blocked")

    def assets() -> dict[str, str]:
        nonlocal assets_called
        assets_called = True
        return {}

    monkeypatch.setattr(queue, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(queue, "STATE_PATH", state)
    monkeypatch.setattr(queue, "_development_context", blocked)
    monkeypatch.setattr(queue, "_ensure_assets", assets)
    with pytest.raises(RuntimeError, match="M2 blocked"):
        queue.build_manifest(persist=True)
    assert assets_called is False
    assert manifest.exists() is False
    assert state.exists() is False


def test_formal_result_requires_performance_gate_and_both_scenario_identity(
    tmp_path, monkeypatch
) -> None:
    jobs = []
    for method, scenario in queue.METHOD_ORDER:
        jobs.append(
            {
                "job_id": f"{method}-{scenario}",
                "method_version": method,
                "scenario_id": scenario,
                "status": "completed",
                "actual_run_id": f"{method}-{scenario}",
            }
        )
    manifest = {
        "jobs": jobs,
        "selected_candidate": {"candidate_id": "CMTR-B975-R20"},
        "gate_rule": "registered",
        "frozen_spec_hash": "frozen",
    }
    frames = {
        ("v11", "S0"): _identity_frame(),
        ("v14", "S0"): _identity_frame(),
        ("v11", "S4"): _identity_frame(),
        ("v14", "S4"): _identity_frame(),
    }
    metrics = {
        ("v11", "S0"): (0.900, None, 100.0),
        ("v14", "S0"): (0.900, None, 100.0),
        ("v11", "S4"): (0.890, 88.0, 100.0),
        ("v14", "S4"): (0.900, 89.0, 90.0),
    }

    def run_metrics(job):
        key = (job["method_version"], job["scenario_id"])
        last50, lqa20, tta = metrics[key]
        return (
            {
                "method_version": key[0],
                "scenario_id": key[1],
                "run_id": job["actual_run_id"],
                "last50_test_accuracy": last50,
                "lqa20_percent": lqa20,
                "algorithm_tta_s": tta,
            },
            frames[key],
        )

    monkeypatch.setattr(queue, "_run_metrics", run_metrics)
    monkeypatch.setattr(queue, "RUNS_PATH", tmp_path / "runs.parquet")
    monkeypatch.setattr(queue, "RUNS_CSV_PATH", tmp_path / "runs.csv")
    monkeypatch.setattr(queue, "RESULT_PATH", tmp_path / "result.json")
    passed = queue.freeze_result(manifest)
    assert passed["gate"]["strict_win_count"] == 3
    assert passed["gate"]["sole_miss_close"] is True
    assert passed["learning_identity_passed"] is True
    assert passed["overall_gate_passed"] is True

    frames[("v14", "S4")] = _identity_frame(drift=True)
    failed = queue.freeze_result(manifest)
    assert failed["gate"]["performance_gate_passed"] is True
    assert failed["learning_identity_passed"] is False
    assert failed["overall_gate_passed"] is False


def test_formal_gate_uses_lqa20_and_not_old_zero_saturated_metric() -> None:
    assert queue.GATE_MARGINS["s4_lqa20_pp"] == 0.50
    names = queue.development.evaluate_gate.__code__.co_varnames
    assert any("lqa20" in name for name in names)
    assert not any("pta20" in name for name in names)


def test_formal_v14_s4_audit_is_bound_to_erratum_contract() -> None:
    job = {"method_version": "v14", "scenario_id": "S4", "rounds": 1000}
    assert queue.development._audit_script_name(job) == "audit_r2c_v14_run_erratum.py"
