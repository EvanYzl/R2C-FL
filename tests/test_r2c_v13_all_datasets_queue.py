from __future__ import annotations

import pandas as pd

from r2c_baselines import r2c_v13_all_datasets_queue as queue
from r2c_baselines.r2c_v13 import PROTOCOL_VERSION, schedule_lambda, schedule_rounds


def _m3_context(schedule_id: str = "DARE-L5") -> dict[str, object]:
    return {
        "schedule_id": schedule_id,
        "manifest": {"frozen_spec_hash": "m3-frozen-spec"},
    }


def _dataset_context() -> dict[str, dict[str, object]]:
    return {
        "D1": {
            "config_overrides": {
                "lr_mult": 1.5,
                "r2c_delta_clip": 0.61,
                "r2c_eval_microbatch": 8,
            },
            "target_accuracy": 0.79,
            "client_microbatch": 1,
        },
        "D2": {
            "config_overrides": {
                "lr_mult": 1.5,
                "r2c_delta_clip": 0.62,
                "r2c_eval_microbatch": 4,
            },
            "target_accuracy": 0.59,
            "client_microbatch": 1,
        },
        "D3": {
            "config_overrides": {
                "lr_mult": 1.0,
                "r2c_delta_clip": 0.53,
                "r2c_eval_microbatch": 4,
            },
            "target_accuracy": 0.80,
            "client_microbatch": 1,
        },
        "D4": {
            "config_overrides": {
                "lr_mult": 1.0,
                "r2c_delta_clip": 0.68,
                "r2c_eval_microbatch": 2,
            },
            "target_accuracy": 0.27,
            "client_microbatch": 1,
        },
    }


def _rounds(budget: int, event_round: int | None, schedule_id: str) -> pd.DataFrame:
    values = pd.DataFrame({"round": list(range(1, budget + 1))})
    for column in (
        "telemetry_shift_trigger",
        "deployment_dare_hold_applied",
        "deployment_dare_envelope_applied",
        "deployment_dare_tracking_applied",
        "deployment_dare_pre_anchor_captured",
        "deployment_dare_pre_anchor_released",
        "deployment_dare_labels_used",
        "deployment_dare_scenario_metadata_used",
        "deployment_dare_future_trace_used",
    ):
        values[column] = False
    values["deployment_dare_state_server_only"] = True
    values["deployment_dare_phase"] = "ordinary"
    values["deployment_dare_lambda_value"] = float("nan")
    values["global_model_hash"] = [f"g-{value}" for value in values["round"]]
    values["evaluation_model_hash"] = [f"e-{value}" for value in values["round"]]
    if event_round is None:
        return values
    duration = schedule_rounds(schedule_id)
    event_index = values["round"] == event_round
    values.loc[event_index, "telemetry_shift_trigger"] = True
    values.loc[event_index, "deployment_dare_hold_applied"] = True
    values.loc[event_index, "deployment_dare_pre_anchor_captured"] = True
    values.loc[event_index, "deployment_dare_phase"] = "trigger_hold"
    for index in range(1, duration + 1):
        round_id = event_round + index
        row = values["round"] == round_id
        values.loc[row, "deployment_dare_envelope_applied"] = True
        values.loc[row, "deployment_dare_phase"] = "recovery_envelope"
        values.loc[row, "deployment_dare_lambda_value"] = schedule_lambda(schedule_id, index)
    release = values["round"] == event_round + duration
    values.loc[release, "deployment_dare_pre_anchor_released"] = True
    tracking = values["round"] > event_round + duration
    values.loc[tracking, "deployment_dare_tracking_applied"] = True
    values.loc[tracking, "deployment_dare_phase"] = "persistent_tracking"
    values.loc[tracking, "evaluation_model_hash"] = values.loc[tracking, "global_model_hash"]
    return values


def test_job_matrix_is_exactly_d1_through_d4_s0_s4() -> None:
    context = _m3_context("DARE-L8")
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
        assert job["method_version"] == "v13"
        assert job["evaluation_split"] == "test"
        assert job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260811
        assert job["selected_schedule_id"] == "DARE-L8"
        assert job["formal_test_access"] is True
        assert job["other_dataset_access"] is True
        assert job["test_labels_used_for_selection"] is False
        assert job["formal_interpretation"].startswith("outcome_informed")


def test_dataset_config_keeps_locked_v13_and_only_dataset_overrides() -> None:
    context = _m3_context("DARE-C5")
    dataset_context = _dataset_context()
    for dataset_id in queue.DATASET_ORDER:
        config = queue._dataset_method_config(dataset_id, context, dataset_context)
        assert config["r2c_protocol_version"] == PROTOCOL_VERSION
        assert config["r2c_v13_schedule_id"] == "DARE-C5"
        assert config["r2c_v13_plan_id"] == queue.PLAN_ID
        assert config["r2c_v2_audit_replay"] is True
        assert config["r2c_v4_deployment_ema_betas"] == [0.95]
        assert config["r2c_v4_primary_deployment_beta"] == 0.95
        assert "r2c_v7_trigger_deployment_beta" not in config
        for key, expected in dataset_context[dataset_id]["config_overrides"].items():
            assert config[key] == expected


def test_m3_context_hard_blocks_before_authority_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(queue, "M3_MANIFEST_PATH", tmp_path / "missing-manifest.json")
    monkeypatch.setattr(queue, "M3_STATE_PATH", tmp_path / "missing-state.json")
    monkeypatch.setattr(queue, "M3_RESULT_PATH", tmp_path / "missing-result.json")
    monkeypatch.setattr(queue, "M3_RUNS_PATH", tmp_path / "missing-runs.parquet")
    monkeypatch.setattr(queue, "M3_RUNS_CSV_PATH", tmp_path / "missing-runs.csv")
    try:
        queue._m3_context()
    except RuntimeError as exc:
        assert "did not authorize" in str(exc)
    else:
        raise AssertionError("M4 must hard-block until M3 passes")


def test_nonpersistent_manifest_is_sealed_and_runtime_hash_stable(monkeypatch) -> None:
    context = _m3_context("DARE-L5")
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
    before = queue.config_hash(queue._frozen_spec(manifest))
    manifest["jobs"][0].update(
        {"status": "running", "attempts": 1, "actual_run_id": "attempt", "failure_reason": None}
    )
    after = queue.config_hash(queue._frozen_spec(manifest))
    assert before == after == manifest["frozen_spec_hash"]


def test_parameterized_dare_contract_supports_all_round_budgets() -> None:
    for budget in (500, 1000, 1500):
        queue._audit_v13_contract(
            _rounds(budget, None, "DARE-L5"),
            "S0",
            "DARE-L5",
            0,
            budget,
            f"s0-{budget}",
        )
        event_round = budget // 2
        queue._audit_v13_contract(
            _rounds(budget, event_round, "DARE-C5"),
            "S4",
            "DARE-C5",
            event_round,
            budget,
            f"s4-{budget}",
        )


def test_terminal_result_requires_exact_eight_run_order(tmp_path, monkeypatch) -> None:
    context = _m3_context("DARE-L5")
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
        "formal_interpretation": "outcome_informed_engineering_reevaluation_not_untouched_confirmation",
        "selected_candidate": {
            "schedule_id": "DARE-L5",
            "source_m3_frozen_spec_hash": "m3-frozen-spec",
        },
        "frozen_spec_hash": "m4-frozen-spec",
    }

    def fake_metrics(job):
        return {
            "dataset_id": job["dataset_id"],
            "scenario_id": job["scenario_id"],
            "run_id": job["actual_run_id"],
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
