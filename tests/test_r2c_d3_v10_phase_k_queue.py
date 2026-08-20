from __future__ import annotations

import pandas as pd

from r2c_baselines import r2c_d3_v10_phase_k_queue as queue


def test_phase_k_manifest_is_exact_three_seed_multi_beta_validation_matrix() -> None:
    manifest = queue.build_manifest(persist=False)
    jobs = manifest["jobs"]
    assert len(jobs) == 3
    assert [job["seed"] for job in jobs] == list(queue.SEEDS)
    assert [job["job_id"] for job in jobs] == manifest["job_order"]
    assert manifest["candidate_betas"] == list(queue.BETAS)
    assert manifest["within_trajectory_control_beta"] == queue.CONTROL_BETA
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
        assert job["variant"] == "robust_deployment_smoothing_v10"
        config = job["method_config"]
        assert config["r2c_v2_audit_replay"] is False
        assert config["r2c_v4_deployment_ema_betas"] == list(queue.BETAS)
        assert config["r2c_v4_primary_deployment_beta"] == queue.CONTROL_BETA
        assert config["r2c_v7_trigger_deployment_beta"] == 1.0


def test_phase_k_changes_only_parallel_deployment_beta_grid() -> None:
    source = queue._verify_validation_source(run_audit=False)
    treatment = queue._treatment_config(source)
    assert set(treatment) == set(source)
    changed = {key for key in source if treatment[key] != source[key]}
    assert changed == {"r2c_v4_deployment_ema_betas"}
    assert source["r2c_v4_deployment_ema_betas"] == [queue.CONTROL_BETA]
    assert treatment["r2c_v4_deployment_ema_betas"] == list(queue.BETAS)
    assert treatment["r2c_v4_primary_deployment_beta"] == queue.CONTROL_BETA


def _candidate_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    control_auc = {20260808: 0.0004, 20260809: 0.0003, 20260810: 0.0005}
    for seed in queue.SEEDS:
        for beta in queue.BETAS:
            auc = control_auc[seed]
            accuracy = 0.90
            hit: int | None = 100
            if beta == 0.925:
                auc -= 0.0001
                accuracy -= 0.0005
                hit += 5
            elif beta == 0.95:
                auc -= 0.0002
                accuracy -= 0.0010
                hit += 7
            elif beta == 0.975:
                auc += 0.0001
                hit += 4
            elif beta == 0.99:
                auc -= 0.00025
                hit = None
            rows.append(
                {
                    "run_id": f"seed-{seed}",
                    "seed": seed,
                    "beta": beta,
                    "last50_validation_accuracy": accuracy,
                    "recovery_deficit_auc20": auc,
                    "target_hit_round": hit,
                    "complete": True,
                }
            )
    return pd.DataFrame(rows)


def test_phase_k_aggregate_applies_robust_eligibility_and_order() -> None:
    aggregates = queue._aggregate_candidates(_candidate_frame())
    control = aggregates.loc[aggregates["beta"].eq(queue.CONTROL_BETA)].iloc[0]
    beta_925 = aggregates.loc[aggregates["beta"].eq(0.925)].iloc[0]
    beta_950 = aggregates.loc[aggregates["beta"].eq(0.95)].iloc[0]
    beta_975 = aggregates.loc[aggregates["beta"].eq(0.975)].iloc[0]
    beta_990 = aggregates.loc[aggregates["beta"].eq(0.99)].iloc[0]
    assert bool(control["eligible"]) is False
    assert bool(beta_925["eligible"]) is True
    assert bool(beta_950["eligible"]) is True
    assert bool(beta_975["eligible"]) is False
    assert bool(beta_990["eligible"]) is False
    eligible = aggregates.loc[aggregates["eligible"].astype(bool)].sort_values(
        ["worst_auc20", "mean_auc20", "mean_last50_validation_accuracy", "mean_target_hit_round", "beta"],
        ascending=[True, True, False, True, True],
        kind="mergesort",
    )
    assert float(eligible.iloc[0]["beta"]) == 0.95


def test_phase_k_requires_strict_auc_improvement() -> None:
    frame = _candidate_frame()
    control = frame.loc[frame["beta"].eq(queue.CONTROL_BETA)].set_index("seed")
    mask = frame["beta"].eq(0.925)
    frame.loc[mask, "recovery_deficit_auc20"] = frame.loc[mask, "seed"].map(
        control["recovery_deficit_auc20"]
    )
    aggregates = queue._aggregate_candidates(frame)
    candidate = aggregates.loc[aggregates["beta"].eq(0.925)].iloc[0]
    assert bool(candidate["mean_auc_no_higher"]) is True
    assert bool(candidate["worst_auc_no_higher"]) is True
    assert bool(candidate["mean_or_worst_auc_strictly_lower"]) is False
    assert bool(candidate["eligible"]) is False


def test_phase_k_structure_gate_rejects_one_broken_hash_hold() -> None:
    rows = []
    for seed in queue.SEEDS:
        rows.append(
            {
                "seed": seed,
                "trigger_count": 1,
                "trigger_rounds_json": "[300]",
                "quarantine_count": 1,
                "quarantine_rounds_json": "[300]",
                "synchronization_count": 0,
                "response_count": 1,
                "quarantine_matches_trigger": True,
                "response_matches_trigger": True,
                "action_matches_trigger": True,
                "configured_beta_is_one": True,
                "effective_beta_matches_contract": True,
                "trigger_deployment_hash_held": True,
                "trigger_global_training_advanced": True,
                "forbidden_input_clean": True,
            }
        )
    frame = pd.DataFrame(rows)
    assert all(queue._structure_gates(frame).values())
    frame.loc[0, "trigger_deployment_hash_held"] = False
    gates = queue._structure_gates(frame)
    assert gates["trigger_deployment_hash_held"] is False
    assert all(value for key, value in gates.items() if key != "trigger_deployment_hash_held")
