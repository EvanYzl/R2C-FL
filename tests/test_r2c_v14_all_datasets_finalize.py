from __future__ import annotations

import copy

import pandas as pd
import pytest

from r2c_baselines.r2c_v14 import PROTOCOL_VERSION
from r2c_baselines.r2c_v14_all_datasets_finalize import (
    FORMAL_INTERPRETATION,
    _validate_terminal_payloads,
)


def _payloads():
    candidate_id = "CMTR-B975-R20"
    m3_hash = "m3-frozen-spec"
    jobs = [
        {
            "job_id": f"v14-{dataset}-{scenario}",
            "dataset_id": dataset,
            "scenario_id": scenario,
            "status": "completed",
            "actual_run_id": f"v14-{dataset}-{scenario}",
            "mode": "formal",
            "method_id": "R2C-FL",
            "method_version": "v14",
            "rounds": {"D1": 500, "D2": 1000, "D3": 1000, "D4": 1500}[dataset],
            "seed": 20260811,
            "partition_seed": 20260811,
            "trace_seed": 20260811,
            "evaluation_split": "test",
            "selected_candidate_id": candidate_id,
            "source_m3_frozen_spec_hash": m3_hash,
            "formal_test_access": True,
            "other_dataset_access": True,
            "test_labels_used_for_selection": False,
            "formal_interpretation": FORMAL_INTERPRETATION,
        }
        for dataset in ("D1", "D2", "D3", "D4")
        for scenario in ("S0", "S4")
    ]
    selected = {
        "candidate_id": candidate_id,
        "source_m3_frozen_spec_hash": m3_hash,
    }
    m3_context = {
        "candidate_id": candidate_id,
        "manifest": {"frozen_spec_hash": m3_hash},
    }
    state = {
        "status": "m4_completed_audited_tables_update_required",
        "completed": 8,
        "failed": 0,
        "total": 8,
        "all_runs_completed": True,
        "performance_sealed_until_terminal": False,
        "formal_test_access": True,
        "other_dataset_access": True,
        "frozen_spec_hash": "m4-frozen-spec",
    }
    result = {
        "status": "m4_all_datasets_completed_audited",
        "evaluation_split": "test",
        "source_kind": "REPRODUCED",
        "formal_test_access": True,
        "other_dataset_access": True,
        "test_labels_used_for_selection": False,
        "formal_interpretation": FORMAL_INTERPRETATION,
        "selected_candidate": selected,
        "completed_runs": 8,
        "run_ids": [job["actual_run_id"] for job in jobs],
        "frozen_spec_hash": "m4-frozen-spec",
    }
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "evaluation_split": "test",
        "seed": 20260811,
        "formal_test_access": True,
        "other_dataset_access": True,
        "test_labels_used_for_selection": False,
        "candidate_locked_before_all_datasets": True,
        "formal_interpretation": FORMAL_INTERPRETATION,
        "selected_candidate": selected,
        "frozen_spec_hash": "m4-frozen-spec",
        "jobs": jobs,
    }
    stored = pd.DataFrame(
        [
            {
                "dataset_id": job["dataset_id"],
                "scenario_id": job["scenario_id"],
                "run_id": job["actual_run_id"],
                "round_budget": job["rounds"],
                "last50_test_accuracy": 0.5,
                "lqa20_percent": None if job["scenario_id"] == "S0" else 50.0,
                "algorithm_tta_round": None,
                "algorithm_tta_s": None,
                "recovery_deficit_auc20": (
                    None if job["scenario_id"] == "S0" else 0.01
                ),
                "trm20_pp": None if job["scenario_id"] == "S0" else -1.0,
                "event_round": (
                    None if job["scenario_id"] == "S0" else job["rounds"] // 2
                ),
                "source_kind": "REPRODUCED",
                "formal_interpretation": FORMAL_INTERPRETATION,
                "test_labels_used_for_selection": False,
            }
            for job in jobs
        ]
    )
    return m3_context, state, result, manifest, stored


def test_terminal_payloads_accept_exact_v14_m4_matrix() -> None:
    jobs = _validate_terminal_payloads(*_payloads())
    assert len(jobs) == 8
    assert [(job["dataset_id"], job["scenario_id"]) for job in jobs] == [
        (dataset, scenario)
        for dataset in ("D1", "D2", "D3", "D4")
        for scenario in ("S0", "S4")
    ]


def test_terminal_payloads_reject_nonterminal_state() -> None:
    payloads = list(copy.deepcopy(_payloads()))
    payloads[1]["completed"] = 7
    with pytest.raises(RuntimeError, match="not terminal"):
        _validate_terminal_payloads(*payloads)


def test_terminal_payloads_reject_missing_matrix_cell() -> None:
    payloads = list(copy.deepcopy(_payloads()))
    payloads[3]["jobs"][0]["dataset_id"] = "D2"
    with pytest.raises(RuntimeError, match="matrix or order"):
        _validate_terminal_payloads(*payloads)


def test_terminal_payloads_reject_m3_candidate_lineage_drift() -> None:
    payloads = list(copy.deepcopy(_payloads()))
    payloads[3]["selected_candidate"]["candidate_id"] = "CMTR-B950-R20"
    with pytest.raises(RuntimeError, match="candidate, protocol"):
        _validate_terminal_payloads(*payloads)


def test_terminal_payloads_reject_nonreproduced_stored_run() -> None:
    payloads = list(copy.deepcopy(_payloads()))
    payloads[4].loc[0, "source_kind"] = "SYNTHETIC"
    with pytest.raises(RuntimeError, match="violate lineage"):
        _validate_terminal_payloads(*payloads)


def test_terminal_payloads_reject_missing_lqa_event_window() -> None:
    payloads = list(copy.deepcopy(_payloads()))
    payloads[4].loc[
        payloads[4]["scenario_id"].eq("S4"), "lqa20_percent"
    ] = None
    with pytest.raises(RuntimeError, match="event-window metrics"):
        _validate_terminal_payloads(*payloads)
