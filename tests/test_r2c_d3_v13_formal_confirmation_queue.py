from __future__ import annotations

import json

import pandas as pd

from r2c_baselines import r2c_d3_v13_formal_confirmation_queue as queue
from r2c_baselines.r2c_v7 import PROTOCOL_VERSION as V11_PROTOCOL_VERSION
from r2c_baselines.r2c_v13 import PROTOCOL_VERSION


def _context(schedule_id: str = "DARE-L5") -> dict[str, object]:
    return {
        "schedule_id": schedule_id,
        "manifest": {"frozen_spec_hash": "m2-frozen-spec"},
    }


def _identity_frame(tag: str = "same") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "round": [1, 2],
            "global_model_hash": [f"g1-{tag}", f"g2-{tag}"],
            "evaluation_model_hash": [f"g1-{tag}", f"g2-{tag}"],
            "primary_deployment_model_hash_before": [f"d0-{tag}", f"d1-{tag}"],
            "primary_deployment_model_hash_after": [f"d1-{tag}", f"d2-{tag}"],
        }
    )


def test_v11_and_v13_formal_configs_are_learning_matched() -> None:
    v11 = queue._v11_config()
    v13 = queue._v13_config("DARE-C5")
    assert v11["r2c_protocol_version"] == V11_PROTOCOL_VERSION
    assert v13["r2c_protocol_version"] == PROTOCOL_VERSION
    assert v11["r2c_v4_deployment_ema_betas"] == [0.95]
    assert v13["r2c_v4_deployment_ema_betas"] == [0.95]
    assert v11["r2c_v4_primary_deployment_beta"] == 0.95
    assert v13["r2c_v4_primary_deployment_beta"] == 0.95
    assert v11["r2c_v2_audit_replay"] is True
    assert v13["r2c_v2_audit_replay"] is True
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


def test_formal_job_matrix_is_exactly_v11_then_v13() -> None:
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
        assert job["stage"] == "d3_v13_untouched_formal_confirmation"
        assert job["mode"] == "formal"
        assert job["dataset_id"] == "D3"
        assert job["rounds"] == 1000
        assert job["evaluation_split"] == "test"
        assert job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260812
        assert job["selected_schedule_id"] == "DARE-L8"
        assert job["formal_test_access"] is True
        assert job["other_dataset_access"] is False
        assert job["test_labels_used_for_selection"] is False
        assert job["method_config"]["r2c_v2_audit_replay"] is True


def test_development_context_hard_blocks_before_terminal_result(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    state = tmp_path / "state.json"
    result = tmp_path / "missing-result.json"
    manifest.write_text(json.dumps({}), encoding="utf-8")
    state.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(queue, "DEVELOPMENT_MANIFEST_PATH", manifest)
    monkeypatch.setattr(queue, "DEVELOPMENT_STATE_PATH", state)
    monkeypatch.setattr(queue, "DEVELOPMENT_RESULT_PATH", result)
    try:
        queue._development_context()
    except RuntimeError as exc:
        assert "not terminal" in str(exc)
    else:
        raise AssertionError("M3 must hard-block before terminal passed M2 evidence")


def test_development_context_hard_blocks_when_m2_authority_is_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(queue, "DEVELOPMENT_MANIFEST_PATH", tmp_path / "missing-manifest.json")
    monkeypatch.setattr(queue, "DEVELOPMENT_STATE_PATH", tmp_path / "missing-state.json")
    try:
        queue._development_context()
    except RuntimeError as exc:
        assert "not terminal" in str(exc)
    else:
        raise AssertionError("M3 must hard-block before the M2 authority exists")


def test_nonpersistent_manifest_is_formal_sealed_and_hash_stable(monkeypatch) -> None:
    context = _context("DARE-L5")
    monkeypatch.setattr(queue, "_development_context", lambda: context)
    monkeypatch.setattr(queue, "_source_lineage", lambda _: {"source": "hash"})
    monkeypatch.setattr(queue, "_ensure_assets", lambda: {"asset": "hash"})
    monkeypatch.setattr(queue, "_implementation_hashes", lambda: {"implementation": "hash"})
    manifest = queue.build_manifest(persist=False)
    queue._assert_frozen_manifest(manifest)
    assert manifest["evaluation_split"] == "test"
    assert manifest["formal_test_access"] is True
    assert manifest["other_dataset_access"] is False
    assert manifest["candidate_locked_before_formal"] is True
    assert manifest["performance_sealed_until_terminal"] is True
    assert manifest["job_order"] == [job["job_id"] for job in manifest["jobs"]]
    before = queue.config_hash(queue._frozen_spec(manifest))
    manifest["jobs"][0].update(
        {"status": "running", "attempts": 1, "actual_run_id": "attempt", "failure_reason": None}
    )
    after = queue.config_hash(queue._frozen_spec(manifest))
    assert before == after == manifest["frozen_spec_hash"]


def test_identity_columns_cover_global_evaluation_and_deployment_hashes() -> None:
    assert queue.IDENTITY_COLUMNS == (
        "round",
        "global_model_hash",
        "evaluation_model_hash",
        "primary_deployment_model_hash_before",
        "primary_deployment_model_hash_after",
    )


def test_formal_result_requires_performance_and_s0_identity(tmp_path, monkeypatch) -> None:
    jobs = [queue._job(method, scenario, _context()) for method, scenario in queue.METHOD_ORDER]
    for job in jobs:
        job.update({"status": "completed", "actual_run_id": job["base_run_id"]})
    manifest = {
        "jobs": jobs,
        "selected_candidate": {
            "schedule_id": "DARE-L5",
            "source_development_frozen_spec_hash": "m2-frozen-spec",
        },
        "gate_rule": "frozen-test-rule",
        "frozen_spec_hash": "formal-frozen-spec",
    }

    equal = _identity_frame()
    frames = {
        ("v11", "S0"): equal.copy(),
        ("v11", "S4"): pd.DataFrame(),
        ("v13", "S0"): equal.copy(),
        ("v13", "S4"): pd.DataFrame(),
    }
    metrics = {
        ("v11", "S0"): {"last50_test_accuracy": 0.900, "pta20_percent": None, "algorithm_tta_s": 100.0},
        ("v11", "S4"): {"last50_test_accuracy": 0.890, "pta20_percent": 88.0, "algorithm_tta_s": 100.0},
        ("v13", "S0"): {"last50_test_accuracy": 0.901, "pta20_percent": None, "algorithm_tta_s": 90.0},
        ("v13", "S4"): {"last50_test_accuracy": 0.900, "pta20_percent": 89.0, "algorithm_tta_s": 90.0},
    }

    def fake_run_metrics(job):
        key = (job["method_version"], job["scenario_id"])
        row = {"method_version": key[0], "scenario_id": key[1], "run_id": job["actual_run_id"]}
        row.update(metrics[key])
        return row, frames[key]

    monkeypatch.setattr(queue, "_run_metrics", fake_run_metrics)
    monkeypatch.setattr(queue, "RUNS_PATH", tmp_path / "runs.parquet")
    monkeypatch.setattr(queue, "RUNS_CSV_PATH", tmp_path / "runs.csv")
    monkeypatch.setattr(queue, "RESULT_PATH", tmp_path / "result.json")
    passed = queue.freeze_result(manifest)
    assert passed["gate"]["strict_win_count"] == 4
    assert passed["s0_identity_passed"] is True
    assert passed["overall_gate_passed"] is True
    assert passed["status"] == "m3_gate_passed"

    frames[("v13", "S0")] = _identity_frame("different")
    rejected = queue.freeze_result(manifest)
    assert rejected["gate"]["strict_win_count"] == 4
    assert rejected["s0_identity_passed"] is False
    assert rejected["overall_gate_passed"] is False
    assert rejected["status"] == "m3_gate_failed"
