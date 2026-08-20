from __future__ import annotations

import math

import pandas as pd

from r2c_baselines import r2c_d3_v8_phase_g_queue as queue


def test_phase_g_manifest_is_exact_three_treatment_validation_matrix() -> None:
    manifest = queue.build_manifest(persist=False)
    jobs = manifest["jobs"]
    assert len(jobs) == 3
    assert [job["seed"] for job in jobs] == list(queue.SEEDS)
    assert [job["job_id"] for job in jobs] == manifest["job_order"]
    assert manifest["formal_test_access"] is False
    assert manifest["other_dataset_access"] is False
    assert manifest["test_labels_used_for_selection"] is False
    assert manifest["detector_config"]["duration_ratio"] == 1.25
    for job in jobs:
        assert job["dataset_id"] == "D3"
        assert job["scenario_id"] == "S4"
        assert job["evaluation_split"] == "validation"
        assert job["rounds"] == 600
        assert job["full_logging"] is True
        assert job["seed"] == job["partition_seed"] == job["trace_seed"]
        assert job["variant"] == "telemetry_sync_v8"
        assert job["method_config"]["r2c_v2_audit_replay"] is False


def test_v8_changes_only_frozen_detector_fields_from_control() -> None:
    control = queue._verify_controls(run_audit=False)
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
            assert treatment[key] == queue.TREATMENT_PROTOCOL_VERSION
        else:
            assert treatment[key] == value
    assert treatment["r2c_v6_duration_log_ratio_threshold"] == math.log(1.25)
    assert treatment["r2c_v6_changed_fraction_threshold"] == 0.25
    assert treatment["r2c_v6_min_comparable_clients"] == 10
    assert treatment["r2c_v6_cooldown_rounds"] == 50


def _passing_frame() -> pd.DataFrame:
    rows = []
    for position, seed in enumerate(queue.SEEDS):
        auc = 0.003 + 0.001 * position
        accuracy = 0.90 + 0.001 * position
        tta = 100.0 + position
        rows.extend(
            [
                {
                    "run_id": f"control-{seed}",
                    "seed": seed,
                    "variant": "control",
                    "recovery_deficit_auc20": auc,
                    "last50_validation_accuracy": accuracy,
                    "algorithm_tta_s": tta,
                    "final_participation_jfi": 0.95,
                    "final_worst10_participation": 45.0,
                    "trigger_count": None,
                    "trigger_rounds_json": None,
                    "forbidden_input_clean": None,
                },
                {
                    "run_id": f"sync-{seed}",
                    "seed": seed,
                    "variant": "telemetry_sync_v8",
                    "recovery_deficit_auc20": auc - 0.0005,
                    "last50_validation_accuracy": accuracy - 0.001,
                    "algorithm_tta_s": tta * (1.01 + 0.005 * position),
                    "final_participation_jfi": 0.945,
                    "final_worst10_participation": 44.0,
                    "trigger_count": 1,
                    "trigger_rounds_json": "[300]",
                    "forbidden_input_clean": True,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_phase_g_gate_requires_all_preregistered_conditions() -> None:
    pairs, gates = queue._evaluate_pairs(_passing_frame())
    assert len(pairs) == 3
    assert all(gates.values())

    failed = _passing_frame()
    failed.loc[
        (failed["seed"] == queue.SEEDS[0])
        & (failed["variant"] == "telemetry_sync_v8"),
        "trigger_rounds_json",
    ] = "[301]"
    _, failed_gates = queue._evaluate_pairs(failed)
    assert failed_gates["trigger_at_event_each_seed"] is False
    assert all(value for key, value in failed_gates.items() if key != "trigger_at_event_each_seed")


def test_phase_g_tta_requires_mean_two_percent_and_each_five_percent() -> None:
    failed = _passing_frame()
    failed.loc[
        (failed["seed"] == queue.SEEDS[0])
        & (failed["variant"] == "telemetry_sync_v8"),
        "algorithm_tta_s",
    ] = 106.0
    _, gates = queue._evaluate_pairs(failed)
    assert gates["per_seed_tta_overhead_within_5pct"] is False

    failed = _passing_frame()
    for seed in queue.SEEDS:
        mask = (failed["seed"] == seed) & (failed["variant"] == "telemetry_sync_v8")
        control_tta = float(
            failed.loc[(failed["seed"] == seed) & (failed["variant"] == "control"), "algorithm_tta_s"].iloc[0]
        )
        failed.loc[mask, "algorithm_tta_s"] = control_tta * 1.03
    _, gates = queue._evaluate_pairs(failed)
    assert gates["mean_tta_overhead_within_2pct"] is False
    assert gates["per_seed_tta_overhead_within_5pct"] is True

