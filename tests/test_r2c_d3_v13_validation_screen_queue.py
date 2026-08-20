from __future__ import annotations

import math

import numpy as np

from r2c_baselines import r2c_d3_v13_validation_screen_queue as queue
from r2c_baselines.r2c_v13 import PROTOCOL_VERSION, schedule_lambda, schedule_rounds


def test_candidate_order_scope_and_config_are_closed() -> None:
    assert queue.CANDIDATES == ("DARE-L5", "DARE-C5", "DARE-L8")
    jobs = [queue._job(candidate_id) for candidate_id in queue.CANDIDATES]
    assert len(jobs) == 3
    assert len({job["job_id"] for job in jobs}) == 3
    for job, candidate_id in zip(jobs, queue.CANDIDATES):
        assert job["dataset_id"] == "D3"
        assert job["scenario_id"] == "S4"
        assert job["rounds"] == 1000
        assert job["evaluation_split"] == "validation"
        assert job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260810
        assert job["formal_test_access"] is False
        assert job["other_dataset_access"] is False
        assert job["test_labels_used_for_selection"] is False
        assert job["schedule_id"] == candidate_id
        assert job["recovery_rounds"] == schedule_rounds(candidate_id)
        config = job["method_config"]
        assert config["r2c_protocol_version"] == PROTOCOL_VERSION
        assert config["r2c_v13_schedule_id"] == candidate_id
        assert config["r2c_v4_deployment_ema_betas"] == [0.9]
        assert config["r2c_v4_primary_deployment_beta"] == 0.9
        assert "r2c_v7_trigger_deployment_beta" not in config


def test_control_metrics_rederive_exactly() -> None:
    derived = queue._derive_control_metrics()
    assert derived.keys() == queue.CONTROL_METRICS.keys()
    for key, expected in queue.CONTROL_METRICS.items():
        assert np.isclose(derived[key], expected, rtol=0.0, atol=1e-12)


def test_selection_normalization_and_close_boundary() -> None:
    control = queue.CONTROL_METRICS
    same = queue._selection_fields(
        control["s4_last50_accuracy"],
        control["s4_pta20_percent"],
        control["s4_algorithm_tta_s"],
    )
    assert same["validation_eligible"] is True
    assert same["maximin_normalized_gain"] == 0.0
    assert same["strict_last50_win"] is False
    assert same["strict_pta20_win"] is False
    assert same["strict_tta_win"] is False

    boundary = queue._selection_fields(
        control["s4_last50_accuracy"] - 0.0015,
        control["s4_pta20_percent"] - 0.50,
        control["s4_algorithm_tta_s"] * 1.05,
    )
    assert boundary["validation_eligible"] is True
    assert np.allclose(
        [
            boundary["normalized_last50_gain"],
            boundary["normalized_pta20_gain"],
            boundary["normalized_tta_gain"],
        ],
        [-1.0, -1.0, -1.0],
        rtol=0.0,
        atol=1e-10,
    )

    outside = queue._selection_fields(
        control["s4_last50_accuracy"] - 0.0015001,
        control["s4_pta20_percent"],
        control["s4_algorithm_tta_s"],
    )
    assert outside["validation_eligible"] is False

    missing_tta = queue._selection_fields(
        control["s4_last50_accuracy"], control["s4_pta20_percent"], None
    )
    assert missing_tta["validation_eligible"] is False
    assert math.isinf(missing_tta["normalized_tta_gain"])
    assert missing_tta["normalized_tta_gain"] < 0.0


def test_schedule_contracts_are_exact_and_end_at_one() -> None:
    expected = {
        "DARE-L5": [0.2, 0.4, 0.6, 0.8, 1.0],
        "DARE-C5": [math.sqrt(value / 5.0) for value in range(1, 6)],
        "DARE-L8": [value / 8.0 for value in range(1, 9)],
    }
    for schedule_id, values in expected.items():
        actual = [
            schedule_lambda(schedule_id, index)
            for index in range(1, schedule_rounds(schedule_id) + 1)
        ]
        assert np.allclose(actual, values, rtol=0.0, atol=1e-15)
        assert actual[-1] == 1.0
        assert all(left < right for left, right in zip(actual, actual[1:]))


def test_nonpersistent_manifest_is_sealed_and_hash_stable_under_status_only() -> None:
    manifest = queue.build_manifest(persist=False)
    queue._assert_frozen_manifest(manifest)
    assert manifest["candidate_order"] == list(queue.CANDIDATES)
    assert manifest["performance_sealed_until_terminal"] is True
    assert manifest["completion_rule"].startswith("all three full-budget")
    before = queue.config_hash(queue._frozen_spec(manifest))
    manifest["jobs"][0].update(
        {"status": "running", "attempts": 1, "actual_run_id": "attempt", "failure_reason": None}
    )
    after = queue.config_hash(queue._frozen_spec(manifest))
    assert before == after == manifest["frozen_spec_hash"]
