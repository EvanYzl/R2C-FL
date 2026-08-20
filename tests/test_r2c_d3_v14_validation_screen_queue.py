from __future__ import annotations

import math

import numpy as np

from r2c_baselines import r2c_d3_v14_validation_screen_queue as queue
from r2c_baselines.r2c_v14 import (
    FAST_BETA,
    PROTOCOL_VERSION,
    WARMUP_ROUNDS,
    candidate_recovery_rounds,
    candidate_stable_beta,
)


def test_candidate_order_scope_and_config_are_closed() -> None:
    assert queue.CANDIDATES == (
        "CMTR-B950-R20",
        "CMTR-B975-R10",
        "CMTR-B975-R20",
    )
    jobs = [queue._job(candidate_id) for candidate_id in queue.CANDIDATES]
    assert len(jobs) == 3
    assert len({job["job_id"] for job in jobs}) == 3
    for job, candidate_id in zip(jobs, queue.CANDIDATES):
        stable_beta = candidate_stable_beta(candidate_id)
        assert job["dataset_id"] == "D3"
        assert job["scenario_id"] == "S4"
        assert job["rounds"] == 1000
        assert job["evaluation_split"] == "validation"
        assert job["seed"] == job["partition_seed"] == job["trace_seed"] == 20260810
        assert job["formal_test_access"] is False
        assert job["other_dataset_access"] is False
        assert job["test_labels_used_for_selection"] is False
        assert job["fast_beta"] == FAST_BETA
        assert job["stable_beta"] == stable_beta
        assert job["warmup_rounds"] == WARMUP_ROUNDS
        assert job["recovery_rounds"] == candidate_recovery_rounds(candidate_id)
        assert job["deployment_state_count"] == 2
        config = job["method_config"]
        assert config["r2c_protocol_version"] == PROTOCOL_VERSION
        assert config["r2c_v14_candidate_id"] == candidate_id
        assert config["r2c_v14_fast_beta"] == FAST_BETA
        assert config["r2c_v14_warmup_rounds"] == WARMUP_ROUNDS
        assert config["r2c_v4_deployment_ema_betas"] == [FAST_BETA, stable_beta]
        assert config["r2c_v4_primary_deployment_beta"] == stable_beta
        assert "r2c_v7_trigger_deployment_beta" not in config


def test_control_metrics_rederive_exactly() -> None:
    derived = queue._derive_control_metrics()
    assert derived.keys() == queue.CONTROL_METRICS.keys()
    for key, expected in queue.CONTROL_METRICS.items():
        assert np.isclose(derived[key], expected, rtol=0.0, atol=1.0e-12)


def test_selection_requires_three_wins_or_two_plus_sole_close() -> None:
    control = queue.CONTROL_METRICS
    same = queue._selection_fields(
        control["s4_last50_accuracy"],
        control["s4_lqa20_percent"],
        control["s4_algorithm_tta_s"],
    )
    assert same["strict_win_count"] == 0
    assert same["validation_eligible"] is False

    three = queue._selection_fields(
        control["s4_last50_accuracy"] + 0.0001,
        control["s4_lqa20_percent"] + 0.01,
        control["s4_algorithm_tta_s"] * 0.99,
    )
    assert three["strict_win_count"] == 3
    assert three["validation_eligible"] is True

    two_close = queue._selection_fields(
        control["s4_last50_accuracy"] + 0.0001,
        control["s4_lqa20_percent"] + 0.01,
        control["s4_algorithm_tta_s"] * 1.05,
    )
    assert two_close["strict_win_count"] == 2
    assert two_close["sole_close_miss"] is True
    assert two_close["validation_eligible"] is True

    two_outside = queue._selection_fields(
        control["s4_last50_accuracy"] + 0.0001,
        control["s4_lqa20_percent"] + 0.01,
        control["s4_algorithm_tta_s"] * 1.050001,
    )
    assert two_outside["strict_win_count"] == 2
    assert two_outside["validation_eligible"] is False

    one_win = queue._selection_fields(
        control["s4_last50_accuracy"] + 0.0001,
        control["s4_lqa20_percent"] - 0.01,
        control["s4_algorithm_tta_s"] * 1.01,
    )
    assert one_win["strict_win_count"] == 1
    assert one_win["validation_eligible"] is False


def test_equality_missing_tta_and_learning_identity_do_not_pass() -> None:
    control = queue.CONTROL_METRICS
    equality = queue._selection_fields(
        control["s4_last50_accuracy"],
        control["s4_lqa20_percent"] + 0.01,
        control["s4_algorithm_tta_s"] * 0.99,
    )
    assert equality["strict_last50_win"] is False
    assert equality["strict_win_count"] == 2
    assert equality["sole_close_miss"] is True
    assert equality["validation_eligible"] is True

    missing_tta = queue._selection_fields(
        control["s4_last50_accuracy"] + 0.0001,
        control["s4_lqa20_percent"] + 0.01,
        None,
    )
    assert missing_tta["validation_eligible"] is False
    assert math.isinf(missing_tta["normalized_tta_gain"])
    assert missing_tta["normalized_tta_gain"] < 0.0

    identity_failure = queue._selection_fields(
        control["s4_last50_accuracy"] + 0.0001,
        control["s4_lqa20_percent"] + 0.01,
        control["s4_algorithm_tta_s"] * 0.99,
        identity=False,
    )
    assert identity_failure["strict_win_count"] == 3
    assert identity_failure["validation_eligible"] is False


def test_nonpersistent_manifest_is_sealed_and_status_hash_stable() -> None:
    manifest = queue.build_manifest(persist=False)
    queue._assert_frozen_manifest(manifest)
    assert manifest["candidate_order"] == list(queue.CANDIDATES)
    assert manifest["performance_sealed_until_terminal"] is True
    assert manifest["formal_test_access"] is False
    assert manifest["other_dataset_access"] is False
    assert manifest["completion_rule"].startswith("all three full-budget")
    before = queue.config_hash(queue._frozen_spec(manifest))
    manifest["jobs"][0].update(
        {
            "status": "running",
            "attempts": 1,
            "actual_run_id": "attempt",
            "failure_reason": None,
        }
    )
    after = queue.config_hash(queue._frozen_spec(manifest))
    assert before == after == manifest["frozen_spec_hash"]
