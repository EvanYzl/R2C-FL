from __future__ import annotations

import json

import pandas as pd
import pytest

from r2c_baselines import r2c_v14_all_datasets_queue as queue
from r2c_baselines.r2c_v14 import FAST_BETA, PROTOCOL_VERSION, WARMUP_ROUNDS


def _m3_context(candidate_id: str = "CMTR-B975-R20") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "manifest": {"frozen_spec_hash": "m3-frozen-spec"},
    }


def _dataset_context() -> dict[str, dict[str, object]]:
    return {
        "D1": {
            "config_overrides": {"lr_mult": 1.5, "r2c_delta_clip": 0.61, "r2c_eval_microbatch": 8},
            "target_accuracy": 0.79,
            "client_microbatch": 1,
        },
        "D2": {
            "config_overrides": {"lr_mult": 1.5, "r2c_delta_clip": 0.62, "r2c_eval_microbatch": 4},
            "target_accuracy": 0.59,
            "client_microbatch": 1,
        },
        "D3": {
            "config_overrides": {"lr_mult": 1.0, "r2c_delta_clip": 0.53, "r2c_eval_microbatch": 4},
            "target_accuracy": 0.80,
            "client_microbatch": 1,
        },
        "D4": {
            "config_overrides": {"lr_mult": 1.0, "r2c_delta_clip": 0.68, "r2c_eval_microbatch": 2},
            "target_accuracy": 0.27,
            "client_microbatch": 1,
        },
    }


def _rounds(
    budget: int,
    event_round: int | None,
    candidate_id: str = "CMTR-B975-R20",
) -> pd.DataFrame:
    values = pd.DataFrame({"round": list(range(1, budget + 1))})
    for column in (
        "telemetry_shift_trigger",
        "deployment_quarantine_applied",
        "deployment_cmtr_recovery_applied",
        "deployment_cmtr_warmup_applied",
        "deployment_cmtr_labels_used",
        "deployment_cmtr_validation_predictions_used",
        "deployment_cmtr_test_predictions_used",
        "deployment_cmtr_scenario_metadata_used",
        "deployment_cmtr_event_round_used",
        "deployment_cmtr_future_trace_used",
        "deployment_cmtr_raw_global_deployment_used",
    ):
        values[column] = False
    values["deployment_cmtr_state_server_only"] = True
    values["selected_deployment_beta"] = queue.candidate_stable_beta(candidate_id)
    warmup = values["round"] <= min(WARMUP_ROUNDS, budget)
    values.loc[warmup, "deployment_cmtr_warmup_applied"] = True
    values.loc[warmup, "selected_deployment_beta"] = FAST_BETA
    if event_round is None:
        return values
    event = values["round"] == event_round
    values.loc[event, "telemetry_shift_trigger"] = True
    values.loc[event, "deployment_quarantine_applied"] = True
    recovery = values["round"].between(
        event_round + 1,
        event_round + queue.candidate_recovery_rounds(candidate_id),
    )
    values.loc[recovery, "deployment_cmtr_recovery_applied"] = True
    values.loc[recovery, "selected_deployment_beta"] = FAST_BETA
    return values


def test_job_matrix_is_exactly_d1_through_d4_s0_s4() -> None:
    context = _m3_context("CMTR-B975-R10")
    dataset_context = _dataset_context()
    jobs = [
        queue._job(dataset_id, scenario, context, dataset_context)
        for dataset_id in queue.DATASET_ORDER
        for scenario in queue.SCENARIO_ORDER
    ]
    assert [(job["dataset_id"], job["scenario_id"]) for job in jobs] == [
        ("D1", "S0"),
        ("D1", "S4"),
        ("D2", "S0"),
        ("D2", "S4"),
        ("D3", "S0"),
        ("D3", "S4"),
        ("D4", "S0"),
        ("D4", "S4"),
    ]
    assert [job["rounds"] for job in jobs] == [500, 500, 1000, 1000, 1000, 1000, 1500, 1500]
    assert len({job["job_id"] for job in jobs}) == 8
    for job in jobs:
        assert job["mode"] == "formal"
        assert job["method_version"] == "v14"
        assert job["evaluation_split"] == "test"
        assert job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260811
        assert job["selected_candidate_id"] == "CMTR-B975-R10"
        assert job["formal_test_access"] is True
        assert job["other_dataset_access"] is True
        assert job["test_labels_used_for_selection"] is False


def test_dataset_config_keeps_locked_v14_and_only_dataset_overrides() -> None:
    context = _m3_context("CMTR-B975-R20")
    dataset_context = _dataset_context()
    for dataset_id in queue.DATASET_ORDER:
        config = queue._dataset_method_config(dataset_id, context, dataset_context)
        assert config["r2c_protocol_version"] == PROTOCOL_VERSION
        assert config["r2c_v14_candidate_id"] == "CMTR-B975-R20"
        assert config["r2c_v14_plan_id"] == queue.PLAN_ID
        assert config["r2c_v2_audit_replay"] is True
        assert config["r2c_v4_deployment_ema_betas"] == [FAST_BETA, 0.975]
        assert config["r2c_v4_primary_deployment_beta"] == 0.975
        assert "r2c_v7_trigger_deployment_beta" not in config
        for key, expected in dataset_context[dataset_id]["config_overrides"].items():
            assert config[key] == expected


def test_m3_context_hard_blocks_before_authority_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(queue, "M3_MANIFEST_PATH", tmp_path / "missing-manifest.json")
    monkeypatch.setattr(queue, "M3_STATE_PATH", tmp_path / "missing-state.json")
    monkeypatch.setattr(queue, "M3_RESULT_PATH", tmp_path / "missing-result.json")
    monkeypatch.setattr(queue, "M3_RUNS_PATH", tmp_path / "missing-runs.parquet")
    monkeypatch.setattr(queue, "M3_RUNS_CSV_PATH", tmp_path / "missing-runs.csv")
    with pytest.raises(RuntimeError, match="did not authorize"):
        queue._m3_context()


def test_build_gate_precedes_other_dataset_assets_and_files(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "m4-manifest.json"
    state = tmp_path / "m4-state.json"
    assets_called = False

    def blocked() -> dict[str, object]:
        raise RuntimeError("M3 blocked")

    def assets() -> dict[str, str]:
        nonlocal assets_called
        assets_called = True
        return {}

    monkeypatch.setattr(queue, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(queue, "STATE_PATH", state)
    monkeypatch.setattr(queue, "_m3_context", blocked)
    monkeypatch.setattr(queue, "_ensure_assets", assets)
    with pytest.raises(RuntimeError, match="M3 blocked"):
        queue.build_manifest(persist=True)
    assert assets_called is False
    assert manifest.exists() is False
    assert state.exists() is False


def test_nonpersistent_manifest_is_sealed_and_runtime_hash_stable(monkeypatch) -> None:
    context = _m3_context("CMTR-B950-R20")
    dataset_context = _dataset_context()
    monkeypatch.setattr(queue, "_m3_context", lambda: context)
    monkeypatch.setattr(queue, "_v11_dataset_context", lambda: dataset_context)
    monkeypatch.setattr(queue, "_source_lineage", lambda _: {"source": "hash"})
    monkeypatch.setattr(queue, "_ensure_assets", lambda: {"asset": "hash"})
    monkeypatch.setattr(queue, "_existing_asset_hashes", lambda: {"asset": "hash"})
    monkeypatch.setattr(queue, "_implementation_hashes", lambda: {"implementation": "hash"})
    manifest = queue.build_manifest(persist=False)
    queue._assert_frozen_manifest(manifest)
    assert manifest["formal_test_access"] is True
    assert manifest["other_dataset_access"] is True
    assert manifest["candidate_locked_before_all_datasets"] is True
    assert manifest["performance_sealed_until_terminal"] is True
    assert len(manifest["jobs"]) == 8
    assert "LQA20" in manifest["diagnostic_metrics"]
    before = queue.config_hash(queue._frozen_spec(manifest))
    manifest["jobs"][0].update(
        {"status": "running", "attempts": 1, "actual_run_id": "attempt", "failure_reason": None}
    )
    after = queue.config_hash(queue._frozen_spec(manifest))
    assert before == after == manifest["frozen_spec_hash"]


def test_parameterized_cmtr_contract_supports_all_round_budgets() -> None:
    for budget in (500, 1000, 1500):
        queue._audit_v14_contract(
            _rounds(budget, None),
            "S0",
            "CMTR-B975-R20",
            None,
            budget,
            f"s0-{budget}",
        )
        event_round = budget // 2
        queue._audit_v14_contract(
            _rounds(budget, event_round),
            "S4",
            "CMTR-B975-R20",
            event_round,
            budget,
            f"s4-{budget}",
        )


def test_terminal_result_requires_exact_eight_run_order(tmp_path, monkeypatch) -> None:
    context = _m3_context("CMTR-B950-R20")
    dataset_context = _dataset_context()
    jobs = [
        queue._job(dataset_id, scenario, context, dataset_context)
        for dataset_id in queue.DATASET_ORDER
        for scenario in queue.SCENARIO_ORDER
    ]
    for job in jobs:
        job.update({"status": "completed", "actual_run_id": job["base_run_id"]})
    manifest = {
        "jobs": jobs,
        "formal_interpretation": "gated_engineering_reevaluation_after_untouched_D3_confirmation",
        "selected_candidate": {
            "candidate_id": "CMTR-B950-R20",
            "source_m3_frozen_spec_hash": "m3-frozen-spec",
        },
        "frozen_spec_hash": "m4-frozen-spec",
    }

    def fake_metrics(job):
        return {
            "dataset_id": job["dataset_id"],
            "scenario_id": job["scenario_id"],
            "run_id": job["actual_run_id"],
            "source_kind": "REPRODUCED",
        }

    monkeypatch.setattr(queue, "_run_metrics", fake_metrics)
    monkeypatch.setattr(queue, "RUNS_PATH", tmp_path / "runs.parquet")
    monkeypatch.setattr(queue, "RUNS_CSV_PATH", tmp_path / "runs.csv")
    monkeypatch.setattr(queue, "RESULT_PATH", tmp_path / "result.json")
    result = queue.freeze_result(manifest)
    assert result["status"] == "m4_all_datasets_completed_audited"
    assert result["completed_runs"] == 8
    assert result["tables_update_required"] is True
    assert result["test_labels_used_for_selection"] is False


def test_all_dataset_audit_routing_preserves_500_round_exception() -> None:
    audit_name = queue.formal.development._audit_script_name
    assert audit_name({"method_version": "v14", "scenario_id": "S4", "rounds": 500}) == (
        "audit_r2c_v14_run.py"
    )
    assert audit_name({"method_version": "v14", "scenario_id": "S4", "rounds": 1000}) == (
        "audit_r2c_v14_run_erratum.py"
    )
    assert audit_name({"method_version": "v14", "scenario_id": "S4", "rounds": 1500}) == (
        "audit_r2c_v14_run_erratum.py"
    )


def test_all_dataset_erratum_payload_accepts_registered_1500_round_budget(tmp_path) -> None:
    run_id = "d4-s4-v14"
    log_path = tmp_path / "audit.log"
    log_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "run_id": run_id,
                "audit_erratum_id": queue.formal.development.m1_authority.ERRATUM_ID,
                "recorded_tables_mutated": False,
                "round_rows": 1500,
                "canonical_window_contract": {
                    "before_count": 20,
                    "after_count": 20,
                    "event_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    job = {"method_version": "v14", "scenario_id": "S4", "rounds": 1500}
    assert queue.formal.development._audit_log_passed(job, run_id, log_path)
