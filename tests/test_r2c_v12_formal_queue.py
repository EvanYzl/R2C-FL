from __future__ import annotations

import copy

import pytest

from r2c_baselines import r2c_d3_v12_formal_queue as formal


def validation_context() -> dict:
    config = {
        "r2c_protocol_version": formal.PROTOCOL_VERSION,
        "r2c_v12_plan_id": formal.PLAN_ID,
        "r2c_v2_audit_replay": False,
        "r2c_v4_primary_deployment_beta": 0.9,
        "r2c_v8_trigger_deployment_beta": 1.0,
        "r2c_v8_recovery_pulse_beta": 0.5,
        "r2c_v8_recovery_pulse_rounds": 5,
    }
    return {
        "selected": {"candidate_id": "P3-B050-D05"},
        "validation_config": config,
    }


def test_formal_config_changes_only_audit_replay() -> None:
    context = validation_context()
    observed = formal._formal_config(context)
    assert observed["r2c_v2_audit_replay"] is True
    expected = dict(context["validation_config"])
    expected["r2c_v2_audit_replay"] = True
    assert observed == expected


@pytest.mark.parametrize("scenario", formal.SCENARIOS)
def test_formal_job_contract(scenario: str) -> None:
    job = formal._job(scenario, validation_context())
    assert job["scenario_id"] == scenario
    assert job["mode"] == "formal"
    assert job["evaluation_split"] == "test"
    assert job["rounds"] == 1000
    assert job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260811
    assert job["candidate_id"] == "P3-B050-D05"
    assert job["pulse_beta"] == 0.5
    assert job["pulse_rounds"] == 5
    assert job["test_labels_used_for_selection"] is False


def test_job_spec_excludes_mutable_fields() -> None:
    job = formal._job("S0", validation_context())
    spec = formal._job_spec(job)
    for key in ("status", "attempts", "actual_run_id", "failure_reason"):
        assert key not in spec


def test_gate_accepts_four_strict_wins() -> None:
    base = formal.FORMAL_BASELINE_ENVELOPE
    result = formal.evaluate_gate(
        s0_last50_accuracy=base["s0_last50_accuracy"] + 1.0e-6,
        s4_last50_accuracy=base["s4_last50_accuracy"] + 1.0e-6,
        s4_trm20_pp=base["s4_trm20_pp"] + 1.0e-6,
        s4_algorithm_tta_s=base["s4_algorithm_tta_s"] - 1.0e-6,
    )
    assert result["strict_win_count"] == 4
    assert result["gate_passed"] is True


def test_gate_accepts_exactly_three_plus_close() -> None:
    base = formal.FORMAL_BASELINE_ENVELOPE
    result = formal.evaluate_gate(
        s0_last50_accuracy=base["s0_last50_accuracy"] - 0.001,
        s4_last50_accuracy=base["s4_last50_accuracy"] + 1.0e-6,
        s4_trm20_pp=base["s4_trm20_pp"] + 1.0e-6,
        s4_algorithm_tta_s=base["s4_algorithm_tta_s"] - 1.0e-6,
    )
    assert result["strict_win_count"] == 3
    assert result["misses"] == ["s0_last50_accuracy"]
    assert result["sole_miss_close"] is True
    assert result["gate_passed"] is True


def test_gate_rejects_two_strict_wins() -> None:
    base = formal.FORMAL_BASELINE_ENVELOPE
    result = formal.evaluate_gate(
        s0_last50_accuracy=base["s0_last50_accuracy"],
        s4_last50_accuracy=base["s4_last50_accuracy"],
        s4_trm20_pp=base["s4_trm20_pp"] + 1.0e-6,
        s4_algorithm_tta_s=base["s4_algorithm_tta_s"] - 1.0e-6,
    )
    assert result["strict_win_count"] == 2
    assert result["gate_passed"] is False


def test_gate_rejects_three_with_distant_miss() -> None:
    base = formal.FORMAL_BASELINE_ENVELOPE
    result = formal.evaluate_gate(
        s0_last50_accuracy=base["s0_last50_accuracy"] - 0.002,
        s4_last50_accuracy=base["s4_last50_accuracy"] + 1.0e-6,
        s4_trm20_pp=base["s4_trm20_pp"] + 1.0e-6,
        s4_algorithm_tta_s=base["s4_algorithm_tta_s"] - 1.0e-6,
    )
    assert result["strict_win_count"] == 3
    assert result["sole_miss_close"] is False
    assert result["gate_passed"] is False


def test_frozen_spec_ignores_runtime_status() -> None:
    jobs = [formal._job(s, validation_context()) for s in formal.SCENARIOS]
    manifest = {
        "schema_version": "1.0.0",
        "scope": "D3_only_v12_locked_formal_pilot",
        "plan_id": formal.PLAN_ID,
        "protocol_version": formal.PROTOCOL_VERSION,
        "evaluation_split": "test",
        "seed": 20260811,
        "rounds_per_job": 1000,
        "event_round": 500,
        "formal_test_access": True,
        "other_dataset_access_before_pilot_gate": False,
        "test_labels_used_for_selection": False,
        "candidate_locked_before_formal": True,
        "performance_sealed_until_terminal": True,
        "selected_candidate": {
            "candidate_id": "P3-B050-D05",
            "pulse_beta": 0.5,
            "pulse_rounds": 5,
        },
        "baseline_envelope": formal.FORMAL_BASELINE_ENVELOPE,
        "gate_margins": formal.GATE_MARGINS,
        "gate_rule": "frozen",
        "source_lineage": {},
        "source_hashes": {},
        "asset_hashes": {},
        "implementation_hashes": {},
        "job_order": [job["job_id"] for job in jobs],
        "max_attempts": 3,
        "jobs": jobs,
    }
    before = formal._frozen_spec(manifest)
    changed = copy.deepcopy(manifest)
    changed["jobs"][0].update(
        {"status": "running", "attempts": 1, "actual_run_id": "x"}
    )
    assert formal._frozen_spec(changed) == before
