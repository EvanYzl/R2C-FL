from __future__ import annotations

import copy

import pytest

from r2c_baselines import r2c_v12_remaining_datasets_queue as queue


def context() -> dict:
    return {
        "method_config": {
            "lr_mult": 1.0,
            "r2c_delta_clip": 0.53,
            "r2c_eval_microbatch": 4,
            "r2c_protocol_version": queue.PROTOCOL_VERSION,
            "r2c_v12_plan_id": queue.PLAN_ID,
            "r2c_v2_audit_replay": True,
            "r2c_v4_primary_deployment_beta": 0.9,
            "r2c_v8_trigger_deployment_beta": 1.0,
            "r2c_v8_recovery_pulse_beta": 0.5,
            "r2c_v8_recovery_pulse_rounds": 5,
        }
    }


def dataset_configs() -> dict[str, dict]:
    return {
        "D1": {"lr_mult": 1.5, "r2c_delta_clip": 0.60, "r2c_eval_microbatch": 8},
        "D2": {"lr_mult": 1.5, "r2c_delta_clip": 0.61, "r2c_eval_microbatch": 4},
        "D4": {"lr_mult": 1.0, "r2c_delta_clip": 0.68, "r2c_eval_microbatch": 2},
    }


@pytest.mark.parametrize(
    ("dataset_id", "expected_rounds"), (("D1", 500), ("D2", 1000), ("D4", 1500))
)
def test_dataset_config_and_budget(dataset_id: str, expected_rounds: int) -> None:
    job = queue._job(dataset_id, "S0", context(), dataset_configs())
    assert job["rounds"] == expected_rounds
    assert job["mode"] == "formal"
    assert job["evaluation_split"] == "test"
    assert job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260811
    assert job["method_config"]["r2c_v4_primary_deployment_beta"] == 0.9
    assert job["method_config"]["r2c_v8_recovery_pulse_beta"] == 0.5
    assert job["method_config"]["r2c_v8_recovery_pulse_rounds"] == 5
    for key, value in dataset_configs()[dataset_id].items():
        assert job["method_config"][key] == value


def test_matrix_order_is_d1_d2_d4_then_s0_s4() -> None:
    observed = [
        (dataset_id, scenario)
        for dataset_id in queue.DATASET_ORDER
        for scenario in queue.SCENARIO_ORDER
    ]
    assert observed == [
        ("D1", "S0"),
        ("D1", "S4"),
        ("D2", "S0"),
        ("D2", "S4"),
        ("D4", "S0"),
        ("D4", "S4"),
    ]


def test_dataset_config_changes_only_three_keys() -> None:
    base = context()["method_config"]
    value = queue._dataset_method_config("D1", context(), dataset_configs())
    changed = sorted(key for key in set(base) | set(value) if base.get(key) != value.get(key))
    assert changed == sorted(queue.DATASET_CONFIG_KEYS)


def test_job_spec_excludes_runtime_fields() -> None:
    job = queue._job("D1", "S0", context(), dataset_configs())
    spec = queue._job_spec(job)
    for key in ("status", "attempts", "actual_run_id", "failure_reason"):
        assert key not in spec


def test_frozen_spec_ignores_runtime_status() -> None:
    jobs = [
        queue._job(dataset_id, scenario, context(), dataset_configs())
        for dataset_id in queue.DATASET_ORDER
        for scenario in queue.SCENARIO_ORDER
    ]
    manifest = {
        "schema_version": "1.0.0",
        "scope": "v12_remaining_D1_D2_D4_locked_formal_matrix",
        "plan_id": queue.PLAN_ID,
        "protocol_version": queue.PROTOCOL_VERSION,
        "evaluation_split": "test",
        "seed": 20260811,
        "dataset_order": list(queue.DATASET_ORDER),
        "scenario_order": list(queue.SCENARIO_ORDER),
        "round_budgets": {"D1": 500, "D2": 1000, "D4": 1500},
        "candidate_locked_before_other_datasets": True,
        "performance_sealed_until_terminal": True,
        "test_labels_used_for_selection": False,
        "selected_candidate": {
            "candidate_id": "P3-B050-D05",
            "ordinary_beta": 0.9,
            "trigger_beta": 1.0,
            "pulse_beta": 0.5,
            "pulse_rounds": 5,
        },
        "source_lineage": {},
        "asset_hashes": {},
        "implementation_hashes": {},
        "job_order": [job["job_id"] for job in jobs],
        "max_attempts": 3,
        "jobs": jobs,
    }
    before = queue._frozen_spec(manifest)
    changed = copy.deepcopy(manifest)
    changed["jobs"][0].update({"status": "running", "attempts": 1, "actual_run_id": "x"})
    assert queue._frozen_spec(changed) == before


def test_formal_gate_is_hard_block_before_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(queue, "FORMAL_RESULT_PATH", queue.QUEUE_ROOT / "definitely_missing.json")
    with pytest.raises(RuntimeError, match="not terminal"):
        queue._formal_context()
