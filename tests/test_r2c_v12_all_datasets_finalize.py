from __future__ import annotations

import copy

import pytest

from r2c_baselines.r2c_v12_all_datasets_finalize import _validate_terminal_payloads


def _payloads():
    jobs = [
        {
            "dataset_id": dataset,
            "scenario_id": scenario,
            "status": "completed",
            "actual_run_id": f"v12-{dataset}-{scenario}",
            "seed": 20260811,
            "partition_seed": 20260811,
            "trace_seed": 20260811,
        }
        for dataset in ("D1", "D2", "D3", "D4")
        for scenario in ("S0", "S4")
    ]
    formal_state = {
        "status": "formal_pilot_completed_gate_passed_remaining_datasets_required",
        "completed": 2,
        "failed": 0,
        "all_runs_completed": True,
        "gate_passed": True,
        "other_dataset_access": True,
    }
    formal_result = {
        "status": "formal_pilot_gate_passed",
        "gate": {"gate_passed": True},
        "other_dataset_access_authorized": True,
        "test_labels_used_for_selection": False,
    }
    remaining_state = {
        "status": "remaining_datasets_completed_tables_update_required",
        "completed": 6,
        "failed": 0,
        "all_runs_completed": True,
        "performance_sealed_until_terminal": False,
    }
    remaining_result = {
        "status": "remaining_datasets_completed_audited",
        "completed_runs": 6,
        "test_labels_used_for_selection": False,
    }
    formal_manifest = {"jobs": [jobs[4], jobs[5]]}
    remaining_manifest = {"jobs": jobs[:4] + jobs[6:]}
    return (
        formal_state,
        formal_result,
        formal_manifest,
        remaining_state,
        remaining_result,
        remaining_manifest,
    )


def test_terminal_payloads_accept_exact_eight_run_matrix() -> None:
    jobs = _validate_terminal_payloads(*_payloads())
    assert len(jobs) == 8
    assert {(job["dataset_id"], job["scenario_id"]) for job in jobs} == {
        (dataset, scenario)
        for dataset in ("D1", "D2", "D3", "D4")
        for scenario in ("S0", "S4")
    }


def test_terminal_payloads_reject_failed_d3_gate() -> None:
    payloads = list(copy.deepcopy(_payloads()))
    payloads[1]["gate"]["gate_passed"] = False
    with pytest.raises(RuntimeError, match="has not authorized"):
        _validate_terminal_payloads(*payloads)


def test_terminal_payloads_reject_missing_dataset_scenario_cell() -> None:
    payloads = list(copy.deepcopy(_payloads()))
    payloads[5]["jobs"][0]["dataset_id"] = "D3"
    with pytest.raises(RuntimeError, match="matrix is incomplete"):
        _validate_terminal_payloads(*payloads)
