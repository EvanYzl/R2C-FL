from __future__ import annotations

import pandas as pd

from r2c_baselines import r2c_d3_v7_phase_e_queue as queue


def test_phase_e_manifest_is_exact_five_job_validation_matrix() -> None:
    manifest = queue.build_manifest(persist=False)
    jobs = manifest["jobs"]
    assert len(jobs) == 5
    assert [(job["seed"], job["variant"]) for job in jobs] == list(queue.NEW_JOB_MATRIX)
    assert [job["job_id"] for job in jobs] == manifest["job_order"]
    assert manifest["formal_test_access"] is False
    assert manifest["other_dataset_access"] is False
    assert manifest["test_labels_used_for_selection"] is False
    for job in jobs:
        assert job["dataset_id"] == "D3"
        assert job["scenario_id"] == "S4"
        assert job["evaluation_split"] == "validation"
        assert job["rounds"] == 600
        assert job["full_logging"] is True
        assert job["seed"] == job["partition_seed"] == job["trace_seed"]
        assert job["method_config"]["r2c_v2_audit_replay"] is False


def test_treatment_changes_only_protocol_and_frozen_detector_fields() -> None:
    control = queue._verify_source_control(run_audit=False)
    treatment = queue._treatment_config(control)
    detector_keys = {
        "r2c_v6_duration_log_ratio_threshold",
        "r2c_v6_changed_fraction_threshold",
        "r2c_v6_min_comparable_clients",
        "r2c_v6_cooldown_rounds",
    }
    assert set(treatment) - set(control) == detector_keys
    for key, value in control.items():
        if key == "r2c_protocol_version":
            assert value == queue.CONTROL_PROTOCOL_VERSION
            assert treatment[key] == queue.TREATMENT_PROTOCOL_VERSION
        else:
            assert treatment[key] == value
    assert treatment["r2c_v6_duration_log_ratio_threshold"] == queue.DEFAULT_LOG_RATIO_THRESHOLD
    assert treatment["r2c_v6_changed_fraction_threshold"] == queue.DEFAULT_FRACTION_THRESHOLD
    assert treatment["r2c_v6_min_comparable_clients"] == queue.DEFAULT_MIN_COMPARABLE_CLIENTS
    assert treatment["r2c_v6_cooldown_rounds"] == queue.DEFAULT_COOLDOWN_ROUNDS


def _passing_frame() -> pd.DataFrame:
    rows = []
    for position, seed in enumerate(queue.SEEDS):
        control_auc = 0.003 + 0.001 * position
        control_accuracy = 0.90 + 0.001 * position
        control_tta = 100.0 + position
        control_jfi = 0.95
        control_worst10 = 45.0
        rows.extend(
            [
                {
                    "run_id": f"control-{seed}",
                    "seed": seed,
                    "variant": "control",
                    "recovery_deficit_auc20": control_auc,
                    "last50_validation_accuracy": control_accuracy,
                    "algorithm_tta_s": control_tta,
                    "final_participation_jfi": control_jfi,
                    "final_worst10_participation": control_worst10,
                    "trigger_count": None,
                    "trigger_rounds_json": None,
                    "forbidden_input_clean": None,
                },
                {
                    "run_id": f"sync-{seed}",
                    "seed": seed,
                    "variant": "telemetry_sync",
                    "recovery_deficit_auc20": control_auc - 0.0005,
                    "last50_validation_accuracy": control_accuracy - 0.001,
                    "algorithm_tta_s": control_tta * 1.01,
                    "final_participation_jfi": control_jfi - 0.005,
                    "final_worst10_participation": control_worst10 - 1.0,
                    "trigger_count": 1,
                    "trigger_rounds_json": "[300]",
                    "forbidden_input_clean": True,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_phase_e_gate_requires_all_preregistered_conditions() -> None:
    pairs, gates = queue._evaluate_pairs(_passing_frame())
    assert len(pairs) == 3
    assert all(gates.values())

    failed = _passing_frame()
    failed.loc[
        (failed["seed"] == queue.SEEDS[0]) & (failed["variant"] == "telemetry_sync"),
        "trigger_rounds_json",
    ] = "[301]"
    _, failed_gates = queue._evaluate_pairs(failed)
    assert failed_gates["trigger_at_event_each_seed"] is False
    assert all(value for key, value in failed_gates.items() if key != "trigger_at_event_each_seed")
