from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from r2c_baselines.logging_io import read_chunked_table
from r2c_baselines.r2c import history_balanced_conditional_targets
from r2c_baselines.r2c_v2 import PROTOCOL_VERSION as R2C_V2_PROTOCOL_VERSION
from r2c_baselines.r2c_v3 import PROTOCOL_VERSION as R2C_V3_PROTOCOL_VERSION
from r2c_baselines.r2c_v4 import PROTOCOL_VERSION as R2C_V4_PROTOCOL_VERSION
from r2c_baselines.r2c_v5 import PROTOCOL_VERSION as R2C_V5_PROTOCOL_VERSION
from r2c_baselines.r2c_v6 import (
    DEFAULT_COOLDOWN_ROUNDS,
    DEFAULT_FRACTION_THRESHOLD,
    DEFAULT_LOG_RATIO_THRESHOLD,
    DEFAULT_MIN_COMPARABLE_CLIENTS,
    DEPLOYMENT_RULE as R2C_V6_DEPLOYMENT_RULE,
    PROTOCOL_VERSION as R2C_V6_PROTOCOL_VERSION,
)
from r2c_baselines.r2c_v7 import (
    DEFAULT_TRIGGER_DEPLOYMENT_BETA,
    DEPLOYMENT_RULE as R2C_V7_DEPLOYMENT_RULE,
    PROTOCOL_VERSION as R2C_V7_PROTOCOL_VERSION,
    validated_trigger_deployment_beta,
)
from r2c_baselines.r2c_v8 import (
    DEPLOYMENT_RULE as R2C_V8_DEPLOYMENT_RULE,
    PROTOCOL_VERSION as R2C_V8_PROTOCOL_VERSION,
    DeploymentRecoveryPulse,
    validated_trigger_deployment_beta as validated_v8_trigger_deployment_beta,
)
from r2c_baselines.r2c_v13 import (
    DEPLOYMENT_RULE as R2C_V13_DEPLOYMENT_RULE,
    PROTOCOL_VERSION as R2C_V13_PROTOCOL_VERSION,
    DualAnchorRecoveryEnvelope,
)
from r2c_baselines.utils import canonical_json, sha256_text


def _unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if frame.duplicated(columns).any():
        raise AssertionError(f"Duplicate {label} primary key: {columns}")


def audit_run(run_dir: Path) -> dict[str, object]:
    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    manifest = pd.read_parquet(run_dir / "run_manifest.parquet").iloc[0]
    rounds = int(job["rounds"])
    round_frame = read_chunked_table(run_dir, "round_metrics")
    client = read_chunked_table(run_dir, "client_round_metrics")
    checkpoints = read_chunked_table(run_dir, "checkpoint_metrics")
    certificates = read_chunked_table(run_dir, "certificate_audit")
    deployment = read_chunked_table(run_dir, "deployment_candidate_metrics")
    protocol_version = job.get("method_config", {}).get("r2c_protocol_version")
    method_config = dict(job.get("method_config", {}))
    completion_variant = method_config.get("r2c_ablation_variant")
    no_reusable_prefix = bool(
        method_config.get("r2c_ablation_no_reusable_prefix", False)
    )
    no_finishability = bool(
        method_config.get("r2c_ablation_no_finishability", False)
    )
    no_drift_quarantine = bool(
        method_config.get("r2c_ablation_no_drift_quarantine", False)
    )
    no_valid_crossfit = bool(
        method_config.get("r2c_ablation_no_valid_crossfit", False)
    )
    is_anchor_protocol = protocol_version in {
        R2C_V2_PROTOCOL_VERSION,
        R2C_V3_PROTOCOL_VERSION,
        R2C_V4_PROTOCOL_VERSION,
        R2C_V5_PROTOCOL_VERSION,
        R2C_V6_PROTOCOL_VERSION,
        R2C_V7_PROTOCOL_VERSION,
        R2C_V8_PROTOCOL_VERSION,
        R2C_V13_PROTOCOL_VERSION,
    }

    if len(round_frame) != rounds or len(client) != 100 * rounds or len(certificates) != rounds:
        raise AssertionError(
            {
                "round_rows": len(round_frame),
                "client_rows": len(client),
                "certificate_rows": len(certificates),
                "expected_rounds": rounds,
            }
        )
    _unique(round_frame, ["run_id", "round"], "round")
    _unique(client, ["run_id", "round", "client_id"], "client-round")
    _unique(certificates, ["run_id", "round"], "certificate")
    _unique(checkpoints, ["run_id", "round", "client_id", "checkpoint_j"], "checkpoint")

    expected_checkpoint_rows = 0
    previous_model_hash = str(manifest["initial_model_hash"])
    selection_history_counts = np.zeros(100, dtype=np.int64)
    for round_number in range(1, rounds + 1):
        clients = client.loc[client["round"] == round_number]
        certificate = certificates.loc[certificates["round"] == round_number].iloc[0]
        checkpoint = checkpoints.loc[checkpoints["round"] == round_number]
        admitted = clients.loc[clients["admitted"].astype(bool)]
        selected = clients.loc[clients["selected"].astype(bool)]
        eligible = clients.loc[clients["aggregation_eligible"].astype(bool)]
        m = len(admitted)
        if m != min(20, int(round_frame.loc[round_frame["round"] == round_number, "available_clients"].iloc[0])):
            raise AssertionError(f"Bad admission cardinality in round {round_number}")
        if len(selected) != 10 or not set(selected["client_id"]).issubset(set(admitted["client_id"])):
            raise AssertionError(f"Bad exact-K selection in round {round_number}")
        if not set(eligible["client_id"]).issubset(set(selected["client_id"])):
            raise AssertionError(f"Ineligible aggregation subset in round {round_number}")
        positions = sorted(admitted["admission_draw_position"].astype(int).tolist())
        if positions != list(range(m)):
            raise AssertionError(f"Admission order is not replayable in round {round_number}")
        if abs(float(admitted["conditional_target_s"].sum()) - 10.0) > 1e-10:
            raise AssertionError(f"Conditional targets do not sum to K in round {round_number}")
        products = admitted["admission_prob"].to_numpy() * admitted["conditional_target_s"].to_numpy()
        if not np.allclose(products, admitted["inclusion_prob_pi"].to_numpy(), atol=1e-12, rtol=0):
            raise AssertionError(f"Overall inclusion probabilities mismatch in round {round_number}")
        normalized_sum = float(eligible["hajek_weight_normalized"].sum())
        expected_sum = 0.0 if bool(certificate["no_op_round"]) else 1.0
        if abs(normalized_sum - expected_sum) > 1e-8:
            raise AssertionError(f"Hajek normalization mismatch in round {round_number}")
        if int(certificate["aggregation_subset_n"]) != len(eligible):
            raise AssertionError(f"Certificate aggregation count mismatch in round {round_number}")

        commit_j = int(certificate["commit_j"])
        expected_checkpoint_rows += m if is_anchor_protocol else m * commit_j
        if checkpoint["checkpoint_j"].max() != commit_j:
            raise AssertionError(f"Checkpoint ledger does not end at commit in round {round_number}")
        if is_anchor_protocol:
            if commit_j != 1 or len(checkpoint) != m:
                raise AssertionError(f"Anchor protocol must have one checkpoint per admitted client in round {round_number}")
            if not (checkpoint["protocol_version"] == protocol_version).all():
                raise AssertionError(f"Anchor checkpoint protocol mismatch in round {round_number}")
            if str(certificate["protocol_version"]) != protocol_version:
                raise AssertionError(f"Anchor certificate protocol mismatch in round {round_number}")
            fired = bool(checkpoint["certificate_fired"].astype(bool).all())
            if bool(certificate["certified"]) != fired:
                raise AssertionError(f"V2 certificate flag mismatch in round {round_number}")
            expected_fallback = (
                "invalid_single_fold_ablation"
                if no_valid_crossfit
                else "budget_limited_anchor_commit"
            )
            if not bool(certificate["certified"]) and str(certificate["fallback_reason"]) != expected_fallback:
                raise AssertionError(f"V2 lacks explicit budget-limited commit in round {round_number}")
            if str(certificate["aggregation_weight_rule"]) != "capped_power_stabilized":
                raise AssertionError(f"V2 aggregation rule mismatch in round {round_number}")
            if not (0.0 <= float(certificate["anchor_rank_agreement"]) <= 1.0):
                raise AssertionError(f"V2 anchor agreement out of range in round {round_number}")
            if protocol_version in {
                R2C_V3_PROTOCOL_VERSION,
                R2C_V4_PROTOCOL_VERSION,
                R2C_V5_PROTOCOL_VERSION,
                R2C_V6_PROTOCOL_VERSION,
            }:
                alpha = float(certificate["server_step_alpha"])
                if not 0.0 <= alpha <= 1.0:
                    raise AssertionError(f"Guarded server step out of range in round {round_number}")
                if not bool(certificate["guard_disjoint_from_selection_anchor"]):
                    raise AssertionError(f"Guard isolation missing in round {round_number}")
            if protocol_version in {
                R2C_V4_PROTOCOL_VERSION,
                R2C_V5_PROTOCOL_VERSION,
                R2C_V6_PROTOCOL_VERSION,
            }:
                expected_deployment_rule = (
                    R2C_V6_DEPLOYMENT_RULE
                    if protocol_version == R2C_V6_PROTOCOL_VERSION
                    else "server_only_parameter_ema"
                )
                if str(certificate["deployment_rule"]) != expected_deployment_rule:
                    raise AssertionError(f"V4 deployment rule mismatch in round {round_number}")
                if not bool(certificate["deployment_state_server_only"]):
                    raise AssertionError(f"V4 deployment-state isolation missing in round {round_number}")
            if protocol_version in {R2C_V5_PROTOCOL_VERSION, R2C_V6_PROTOCOL_VERSION}:
                if bool(certificate["selection_history_labels_used"]):
                    raise AssertionError(f"V5 history target used labels in round {round_number}")
                if not bool(certificate["selection_history_state_server_only"]):
                    raise AssertionError(f"V5 history state is not server-only in round {round_number}")
                if str(certificate["history_target_rule"]) != "negative_log1p_cumulative_selection_count":
                    raise AssertionError(f"V5 history-target rule mismatch in round {round_number}")
                expected_counts_hash = sha256_text(
                    canonical_json(selection_history_counts.astype(int).tolist())
                )
                if str(certificate["history_counts_hash"]) != expected_counts_hash:
                    raise AssertionError(f"V5 pre-round history lineage mismatch in round {round_number}")

                checkpoint_ordered = checkpoint.reset_index(drop=True)
                checkpoint_clients = checkpoint_ordered["client_id"].astype(int).to_numpy()
                expected_counts = selection_history_counts[checkpoint_clients]
                logged_counts = checkpoint_ordered["selection_history_count_before"].astype(int).to_numpy()
                if not np.array_equal(logged_counts, expected_counts):
                    raise AssertionError(f"V5 admitted history counts mismatch in round {round_number}")
                history_mix = float(certificate["history_target_mix"])
                history_temperature = float(certificate["history_target_temperature"])
                adaptive_floor = float(certificate["adaptive_floor_fraction"])
                floor = adaptive_floor * 10.0 / float(len(checkpoint_ordered))
                anchor, history, final, _ = history_balanced_conditional_targets(
                    checkpoint_ordered["score_q_hat"].astype(float).to_numpy(),
                    expected_counts,
                    10,
                    float(job["method_config"]["r2c_v2_temperature"]),
                    history_temperature,
                    floor,
                    history_mix,
                )
                for values, column in (
                    (anchor, "anchor_conditional_target_s"),
                    (history, "history_conditional_target_s"),
                ):
                    if not np.allclose(
                        values,
                        checkpoint_ordered[column].astype(float).to_numpy(),
                        atol=1.0e-12,
                        rtol=0.0,
                    ):
                        raise AssertionError(f"V5 {column} mismatch in round {round_number}")
                admitted_by_checkpoint = (
                    clients.set_index("client_id").loc[checkpoint_clients].reset_index()
                )
                if not np.allclose(
                    final,
                    admitted_by_checkpoint["conditional_target_s"].astype(float).to_numpy(),
                    atol=1.0e-12,
                    rtol=0.0,
                ):
                    raise AssertionError(f"V5 final target mismatch in round {round_number}")
                expected_target_hash = sha256_text(
                    canonical_json(
                        {
                            "admitted_client_ids": checkpoint_clients.tolist(),
                            "selection_history_counts": expected_counts.astype(int).tolist(),
                            "anchor_targets": anchor.tolist(),
                            "history_targets": history.tolist(),
                            "final_targets": final.tolist(),
                        }
                    )
                )
                if str(certificate["history_target_hash"]) != expected_target_hash:
                    raise AssertionError(f"V5 history-target hash mismatch in round {round_number}")
            if completion_variant is not None:
                if str(certificate["ablation_variant"]) != str(completion_variant):
                    raise AssertionError(f"Ablation lineage mismatch in round {round_number}")
                if bool(certificate["reusable_prefix_enabled"]) == no_reusable_prefix:
                    raise AssertionError(f"Reusable-prefix flag mismatch in round {round_number}")
                if bool(certificate["finishability_score_enabled"]) == no_finishability:
                    raise AssertionError(f"Finishability flag mismatch in round {round_number}")
                if int(certificate["crossfit_fold_count"]) != (1 if no_valid_crossfit else 2):
                    raise AssertionError(f"Cross-fit fold count mismatch in round {round_number}")
                if bool(certificate["selection_certificate_valid"]) == no_valid_crossfit:
                    raise AssertionError(f"Selection-certificate validity mismatch in round {round_number}")
                expected_finish_weight = (
                    0.0
                    if no_finishability
                    else float(method_config["r2c_v2_finish_weight"])
                )
                if not np.isclose(
                    float(certificate["effective_finishability_weight"]),
                    expected_finish_weight,
                    atol=0.0,
                    rtol=0.0,
                ):
                    raise AssertionError(f"Effective finishability weight mismatch in round {round_number}")
                if no_valid_crossfit and bool(certificate["certified"]):
                    raise AssertionError(f"Invalid single-fold ablation certified round {round_number}")
                if no_valid_crossfit and bool(certificate["guard_disjoint_from_selection_anchor"]):
                    raise AssertionError(f"Single-fold ablation claims guard isolation in round {round_number}")
                if no_reusable_prefix and selected["resumed"].astype(bool).any():
                    raise AssertionError(f"No-prefix ablation resumed a checkpoint in round {round_number}")
                if not no_reusable_prefix and not selected["resumed"].astype(bool).all():
                    raise AssertionError(f"Reusable-prefix run failed to resume in round {round_number}")
                used_compute = float(certificate["candidate_compute_used_s"])
                wasted_compute = float(certificate["candidate_compute_wasted_s"])
                wasted_fraction = float(certificate["candidate_compute_wasted_fraction"])
                if used_compute <= 0.0 or not 0.0 <= wasted_compute <= used_compute + 1.0e-12:
                    raise AssertionError(f"Invalid wasted-compute accounting in round {round_number}")
                if not np.isclose(
                    wasted_fraction,
                    wasted_compute / used_compute,
                    atol=1.0e-12,
                    rtol=0.0,
                ):
                    raise AssertionError(f"Wasted-compute fraction mismatch in round {round_number}")
        else:
            if bool(certificate["certified"]):
                if not bool(checkpoint.loc[checkpoint["checkpoint_j"] == commit_j, "certificate_fired"].all()):
                    raise AssertionError(f"Certified commit not fired in round {round_number}")
            else:
                if commit_j != 10 or checkpoint["certificate_fired"].astype(bool).any():
                    raise AssertionError(f"Fallback is not an explicit final checkpoint in round {round_number}")
        if checkpoint["round_start_model_hash"].nunique() != 1:
            raise AssertionError(f"Multiple round-start model hashes in round {round_number}")
        if str(checkpoint["round_start_model_hash"].iloc[0]) != previous_model_hash:
            raise AssertionError(f"Round-start model lineage mismatch in round {round_number}")
        if int(checkpoint["admission_seed"].iloc[0]) != int(certificate["admission_seed"]):
            raise AssertionError(f"Admission seed mismatch in round {round_number}")
        previous_model_hash = str(
            round_frame.loc[round_frame["round"] == round_number, "global_model_hash"].iloc[0]
        )
        if protocol_version in {
            R2C_V5_PROTOCOL_VERSION,
            R2C_V6_PROTOCOL_VERSION,
            R2C_V7_PROTOCOL_VERSION,
            R2C_V8_PROTOCOL_VERSION,
            R2C_V13_PROTOCOL_VERSION,
        }:
            selection_history_counts[selected["client_id"].astype(int).to_numpy()] += 1

    if protocol_version in {
        R2C_V4_PROTOCOL_VERSION,
        R2C_V5_PROTOCOL_VERSION,
        R2C_V6_PROTOCOL_VERSION,
        R2C_V7_PROTOCOL_VERSION,
        R2C_V8_PROTOCOL_VERSION,
        R2C_V13_PROTOCOL_VERSION,
    }:
        configured_betas = sorted(
            float(value)
            for value in job["method_config"].get("r2c_v4_deployment_ema_betas", [])
        )
        primary_beta = float(job["method_config"]["r2c_v4_primary_deployment_beta"])
        if len(deployment) != rounds * len(configured_betas):
            raise AssertionError("V4 deployment-candidate budget mismatch")
        _unique(deployment, ["run_id", "round", "deployment_beta"], "deployment candidate")
        if sorted(deployment["deployment_beta"].astype(float).unique().tolist()) != configured_betas:
            raise AssertionError("V4 deployment beta set mismatch")
        if not (
            deployment.groupby("round")["is_primary"].apply(lambda values: int(values.astype(bool).sum()))
            == 1
        ).all():
            raise AssertionError("V4 must have exactly one primary deployment state per round")
        primary = deployment.loc[deployment["is_primary"].astype(bool)].sort_values("round")
        if not np.allclose(primary["deployment_beta"].astype(float), primary_beta, atol=0, rtol=0):
            raise AssertionError("V4 primary deployment beta mismatch")
        ordered_round = round_frame.sort_values("round")
        if not np.allclose(
            primary["test_accuracy"].to_numpy(dtype=np.float64),
            ordered_round["test_accuracy"].to_numpy(dtype=np.float64),
            atol=0,
            rtol=0,
        ):
            raise AssertionError("V4 primary deployment accuracy does not match round metrics")
        if not (
            primary["evaluation_model_hash"].astype(str).to_numpy()
            == ordered_round["evaluation_model_hash"].astype(str).to_numpy()
        ).all():
            raise AssertionError("V4 primary deployment hash does not match round metrics")

    if protocol_version in {
        R2C_V6_PROTOCOL_VERSION,
        R2C_V7_PROTOCOL_VERSION,
        R2C_V8_PROTOCOL_VERSION,
        R2C_V13_PROTOCOL_VERSION,
    }:
        config = job["method_config"]
        is_quarantine_protocol = protocol_version == R2C_V7_PROTOCOL_VERSION
        is_pulse_protocol = protocol_version == R2C_V8_PROTOCOL_VERSION
        is_dare_protocol = protocol_version == R2C_V13_PROTOCOL_VERSION
        if is_dare_protocol:
            expected_deployment_protocol = R2C_V13_PROTOCOL_VERSION
            expected_deployment_rule = R2C_V13_DEPLOYMENT_RULE
            expected_trigger_beta = 1.0
            expected_pulse = None
            expected_dare = DualAnchorRecoveryEnvelope.from_config(config)
        elif is_pulse_protocol:
            expected_deployment_protocol = R2C_V8_PROTOCOL_VERSION
            expected_deployment_rule = R2C_V8_DEPLOYMENT_RULE
            expected_trigger_beta = validated_v8_trigger_deployment_beta(config)
            expected_pulse = DeploymentRecoveryPulse.from_config(config)
            expected_dare = None
        elif is_quarantine_protocol:
            expected_deployment_protocol = R2C_V7_PROTOCOL_VERSION
            expected_deployment_rule = R2C_V7_DEPLOYMENT_RULE
            expected_trigger_beta = validated_trigger_deployment_beta(config)
            expected_pulse = None
            expected_dare = None
        else:
            expected_deployment_protocol = R2C_V6_PROTOCOL_VERSION
            expected_deployment_rule = R2C_V6_DEPLOYMENT_RULE
            expected_trigger_beta = 0.0
            expected_pulse = None
            expected_dare = None
        expected_log_threshold = float(
            config.get("r2c_v6_duration_log_ratio_threshold", DEFAULT_LOG_RATIO_THRESHOLD)
        )
        expected_fraction_threshold = float(
            config.get("r2c_v6_changed_fraction_threshold", DEFAULT_FRACTION_THRESHOLD)
        )
        expected_min_comparable = int(
            config.get("r2c_v6_min_comparable_clients", DEFAULT_MIN_COMPARABLE_CLIENTS)
        )
        cooldown_rounds = int(
            config.get("r2c_v6_cooldown_rounds", DEFAULT_COOLDOWN_ROUNDS)
        )
        primary_beta = float(config["r2c_v4_primary_deployment_beta"])
        expected_cooldown_before = 0
        expected_sync_count = 0
        telemetry_columns = [
            "telemetry_shift_round",
            "telemetry_shift_comparable_clients",
            "telemetry_shift_changed_clients",
            "telemetry_shift_changed_fraction",
            "telemetry_shift_log_ratio_threshold",
            "telemetry_shift_fraction_threshold",
            "telemetry_shift_min_comparable_clients",
            "telemetry_shift_trigger",
            "telemetry_shift_cooldown_before",
            "telemetry_shift_cooldown_after",
            "telemetry_shift_synchronization_count",
            "telemetry_shift_state_server_only",
            "telemetry_shift_labels_used",
            "telemetry_shift_scenario_metadata_used",
        ]
        required_round_columns = telemetry_columns + [
            "deployment_synchronization_applied",
            "effective_primary_deployment_beta",
            "primary_deployment_model_hash_before",
            "primary_deployment_model_hash_after",
        ]
        quarantine_columns = [
            "deployment_quarantine_applied",
            "deployment_shift_response_applied",
            "deployment_trigger_action",
            "configured_trigger_deployment_beta",
        ]
        pulse_columns = [
            "deployment_recovery_pulse_applied",
            "deployment_pulse_round",
            "deployment_pulse_telemetry_trigger",
            "deployment_pulse_response_applied",
            "deployment_pulse_hold_applied",
            "deployment_pulse_recovery_applied",
            "deployment_pulse_phase",
            "deployment_pulse_override_beta",
            "deployment_pulse_configured_trigger_beta",
            "deployment_pulse_configured_recovery_beta",
            "deployment_pulse_configured_recovery_rounds",
            "deployment_pulse_remaining_before",
            "deployment_pulse_remaining_after",
            "deployment_pulse_activation_count",
            "deployment_pulse_state_server_only",
            "deployment_pulse_labels_used",
            "deployment_pulse_scenario_metadata_used",
        ]
        dare_columns = [
            "deployment_dare_round",
            "deployment_dare_telemetry_trigger",
            "deployment_dare_response_applied",
            "deployment_dare_hold_applied",
            "deployment_dare_envelope_applied",
            "deployment_dare_tracking_applied",
            "deployment_dare_phase",
            "deployment_dare_lambda_value",
            "deployment_dare_equivalent_beta",
            "deployment_dare_configured_schedule_id",
            "deployment_dare_configured_recovery_rounds",
            "deployment_dare_recovery_index",
            "deployment_dare_remaining_before",
            "deployment_dare_remaining_after",
            "deployment_dare_activation_count",
            "deployment_dare_pre_anchor_capture_requested",
            "deployment_dare_post_anchor_capture_requested",
            "deployment_dare_pre_anchor_captured",
            "deployment_dare_pre_anchor_released",
            "deployment_dare_pre_anchor_hash",
            "deployment_dare_post_anchor_hash",
            "deployment_dare_anchor_tensor_count_before",
            "deployment_dare_anchor_tensor_count_after",
            "deployment_dare_anchor_bytes_before",
            "deployment_dare_anchor_bytes_after",
            "deployment_dare_state_server_only",
            "deployment_dare_labels_used",
            "deployment_dare_scenario_metadata_used",
            "deployment_dare_future_trace_used",
        ]
        if is_quarantine_protocol or is_pulse_protocol or is_dare_protocol:
            required_round_columns += quarantine_columns
        if is_pulse_protocol:
            required_round_columns += pulse_columns
        if is_dare_protocol:
            required_round_columns += [
                "deployment_recovery_envelope_applied",
                "deployment_post_shift_tracking_applied",
            ] + dare_columns
        for required_column in required_round_columns:
            if required_column not in round_frame.columns:
                raise AssertionError(
                    f"Telemetry round column missing for {protocol_version}: {required_column}"
                )

        dare_pre_anchor_hash: str | None = None
        dare_post_anchor_hash: str | None = None
        dare_anchor_tensor_count = 0
        dare_anchor_bytes = 0
        for round_number in range(1, rounds + 1):
            round_row = round_frame.loc[round_frame["round"] == round_number].iloc[0]
            certificate = certificates.loc[certificates["round"] == round_number].iloc[0]
            checkpoint = checkpoints.loc[checkpoints["round"] == round_number]
            candidates = deployment.loc[deployment["round"] == round_number]

            for column in telemetry_columns:
                cert_value = certificate[column]
                round_value = round_row[column]
                if isinstance(cert_value, (float, np.floating)):
                    if not np.isclose(float(cert_value), float(round_value), atol=0.0, rtol=0.0):
                        raise AssertionError(
                            f"V6 certificate/round {column} mismatch in round {round_number}"
                        )
                elif cert_value != round_value:
                    raise AssertionError(
                        f"V6 certificate/round {column} mismatch in round {round_number}"
                    )
                if not (checkpoint[column] == cert_value).all():
                    raise AssertionError(
                        f"V6 checkpoint/certificate {column} mismatch in round {round_number}"
                    )

            if int(certificate["telemetry_shift_round"]) != round_number:
                raise AssertionError(f"V6 telemetry round mismatch in round {round_number}")
            if not np.isclose(
                float(certificate["telemetry_shift_log_ratio_threshold"]),
                expected_log_threshold,
                atol=0.0,
                rtol=0.0,
            ):
                raise AssertionError("V6 telemetry log-ratio threshold differs from frozen config")
            if not np.isclose(
                float(certificate["telemetry_shift_fraction_threshold"]),
                expected_fraction_threshold,
                atol=0.0,
                rtol=0.0,
            ):
                raise AssertionError("V6 telemetry fraction threshold differs from frozen config")
            if int(certificate["telemetry_shift_min_comparable_clients"]) != expected_min_comparable:
                raise AssertionError("V6 minimum comparable-client count differs from frozen config")
            if not bool(certificate["telemetry_shift_state_server_only"]):
                raise AssertionError("V6 detector state is not server-only")
            if bool(certificate["telemetry_shift_labels_used"]):
                raise AssertionError("V6 detector used labels")
            if bool(certificate["telemetry_shift_scenario_metadata_used"]):
                raise AssertionError("V6 detector used scenario metadata")
            if str(certificate["selection_protocol_version"]) != R2C_V5_PROTOCOL_VERSION:
                raise AssertionError("V6 did not preserve the frozen v5 selection protocol")
            if str(certificate["deployment_protocol_version"]) != expected_deployment_protocol:
                raise AssertionError("Telemetry deployment protocol lineage mismatch")
            if str(certificate["deployment_rule"]) != expected_deployment_rule:
                raise AssertionError("Telemetry deployment rule mismatch")
            if is_quarantine_protocol or is_pulse_protocol:
                for column in quarantine_columns:
                    cert_value = certificate[column]
                    round_value = round_row[column]
                    if isinstance(cert_value, (float, np.floating)):
                        if not np.isclose(
                            float(cert_value), float(round_value), atol=0.0, rtol=0.0
                        ):
                            raise AssertionError(
                                f"Quarantine certificate/round {column} mismatch in round {round_number}"
                            )
                    elif cert_value != round_value:
                        raise AssertionError(
                            f"Quarantine certificate/round {column} mismatch in round {round_number}"
                        )
                    if not (checkpoint[column] == cert_value).all():
                        raise AssertionError(
                            f"Quarantine checkpoint/certificate {column} mismatch in round {round_number}"
                        )
                if not np.isclose(
                    float(certificate["configured_trigger_deployment_beta"]),
                    expected_trigger_beta,
                    atol=0.0,
                    rtol=0.0,
                ):
                    raise AssertionError("Configured trigger beta mismatch")

            comparable = int(certificate["telemetry_shift_comparable_clients"])
            changed = int(certificate["telemetry_shift_changed_clients"])
            fraction = float(certificate["telemetry_shift_changed_fraction"])
            expected_fraction = float(changed / comparable) if comparable else 0.0
            if changed < 0 or changed > comparable:
                raise AssertionError(f"V6 invalid changed-client count in round {round_number}")
            if not np.isclose(fraction, expected_fraction, atol=1.0e-15, rtol=0.0):
                raise AssertionError(f"V6 changed fraction mismatch in round {round_number}")
            cooldown_before = int(certificate["telemetry_shift_cooldown_before"])
            if cooldown_before != expected_cooldown_before:
                raise AssertionError(f"V6 cooldown lineage mismatch in round {round_number}")
            expected_trigger = bool(
                cooldown_before == 0
                and comparable >= expected_min_comparable
                and fraction >= expected_fraction_threshold
            )
            trigger = bool(certificate["telemetry_shift_trigger"])
            if trigger != expected_trigger:
                raise AssertionError(f"V6 telemetry trigger predicate mismatch in round {round_number}")
            if trigger:
                expected_sync_count += 1
                expected_cooldown_after = cooldown_rounds
            else:
                expected_cooldown_after = max(0, cooldown_before - 1)
            if int(certificate["telemetry_shift_cooldown_after"]) != expected_cooldown_after:
                raise AssertionError(f"V6 cooldown update mismatch in round {round_number}")
            if int(certificate["telemetry_shift_synchronization_count"]) != expected_sync_count:
                raise AssertionError(f"V6 synchronization count mismatch in round {round_number}")
            expected_cooldown_before = expected_cooldown_after

            pulse_observation = (
                expected_pulse.step(round_number, trigger)
                if expected_pulse is not None
                else None
            )
            dare_observation = (
                expected_dare.step(round_number, trigger)
                if expected_dare is not None
                else None
            )
            expected_sync = bool(
                trigger
                and protocol_version == R2C_V6_PROTOCOL_VERSION
                and not no_drift_quarantine
            )
            expected_quarantine = bool(
                not no_drift_quarantine
                and (
                    (trigger and is_quarantine_protocol)
                    or (
                        pulse_observation is not None
                        and pulse_observation.hold_applied
                    )
                    or (
                        dare_observation is not None
                        and dare_observation.hold_applied
                    )
                )
            )
            expected_recovery_pulse = bool(
                not no_drift_quarantine
                and pulse_observation is not None
                and pulse_observation.recovery_applied
            )
            expected_recovery_envelope = bool(
                not no_drift_quarantine
                and dare_observation is not None
                and dare_observation.envelope_applied
            )
            expected_post_shift_tracking = bool(
                not no_drift_quarantine
                and dare_observation is not None
                and dare_observation.tracking_applied
            )
            expected_response = bool(
                not no_drift_quarantine
                and (
                    dare_observation.response_applied
                    if dare_observation is not None
                    else (
                        pulse_observation.response_applied
                        if pulse_observation is not None
                        else trigger
                    )
                )
            )
            expected_action = (
                "hold"
                if expected_quarantine
                else (
                    "recovery_pulse"
                    if expected_recovery_pulse
                    else (
                        "recovery_envelope"
                        if expected_recovery_envelope
                        else (
                            "post_shift_tracking"
                            if expected_post_shift_tracking
                            else ("hard_sync" if expected_sync else "none")
                        )
                    )
                )
            )
            if bool(certificate["deployment_synchronization_applied"]) != expected_sync:
                raise AssertionError(f"Telemetry synchronization mismatch in round {round_number}")
            if is_quarantine_protocol or is_pulse_protocol or is_dare_protocol:
                if bool(certificate["deployment_quarantine_applied"]) != expected_quarantine:
                    raise AssertionError(f"Quarantine mismatch in round {round_number}")
                if bool(certificate["deployment_shift_response_applied"]) != expected_response:
                    raise AssertionError(f"Shift response mismatch in round {round_number}")
                if str(certificate["deployment_trigger_action"]) != expected_action:
                    raise AssertionError(f"Shift action mismatch in round {round_number}")
            if is_pulse_protocol:
                if bool(certificate["deployment_recovery_pulse_applied"]) != expected_recovery_pulse:
                    raise AssertionError(f"V8 recovery-pulse mismatch in round {round_number}")
                assert pulse_observation is not None
                expected_pulse_fields = pulse_observation.audit_fields()
                for column, expected_value in expected_pulse_fields.items():
                    cert_value = certificate[column]
                    round_value = round_row[column]
                    if expected_value is None:
                        if not (pd.isna(cert_value) and pd.isna(round_value)):
                            raise AssertionError(
                                f"V8 null pulse field mismatch for {column} in round {round_number}"
                            )
                    elif isinstance(expected_value, (float, np.floating)):
                        if not (
                            np.isclose(float(cert_value), float(expected_value), atol=0.0, rtol=0.0)
                            and np.isclose(float(round_value), float(expected_value), atol=0.0, rtol=0.0)
                        ):
                            raise AssertionError(
                                f"V8 pulse field mismatch for {column} in round {round_number}"
                            )
                    elif cert_value != expected_value or round_value != expected_value:
                        raise AssertionError(
                            f"V8 pulse field mismatch for {column} in round {round_number}"
                        )
                    checkpoint_values = checkpoint[column]
                    if expected_value is None:
                        if not checkpoint_values.isna().all():
                            raise AssertionError(
                                f"V8 checkpoint null mismatch for {column} in round {round_number}"
                            )
                    elif not (checkpoint_values == expected_value).all():
                        raise AssertionError(
                            f"V8 checkpoint mismatch for {column} in round {round_number}"
                        )
            if is_dare_protocol:
                if bool(certificate["deployment_recovery_envelope_applied"]) != expected_recovery_envelope:
                    raise AssertionError(
                        f"V13 recovery-envelope mismatch in round {round_number}"
                    )
                if bool(certificate["deployment_post_shift_tracking_applied"]) != expected_post_shift_tracking:
                    raise AssertionError(
                        f"V13 post-shift tracking mismatch in round {round_number}"
                    )
                assert dare_observation is not None
                state_only_fields = dare_observation.audit_fields()
                for suffix in (
                    "pre_anchor_captured",
                    "pre_anchor_released",
                    "pre_anchor_hash",
                    "post_anchor_hash",
                    "anchor_tensor_count_before",
                    "anchor_tensor_count_after",
                    "anchor_bytes_before",
                    "anchor_bytes_after",
                ):
                    state_only_fields.pop(f"deployment_dare_{suffix}")
                for column, expected_value in state_only_fields.items():
                    cert_value = certificate[column]
                    round_value = round_row[column]
                    if expected_value is None:
                        if not (pd.isna(cert_value) and pd.isna(round_value)):
                            raise AssertionError(
                                f"V13 null DARE field mismatch for {column} in round {round_number}"
                            )
                    elif isinstance(expected_value, (float, np.floating)):
                        if not (
                            np.isclose(float(cert_value), float(expected_value), atol=0.0, rtol=0.0)
                            and np.isclose(float(round_value), float(expected_value), atol=0.0, rtol=0.0)
                        ):
                            raise AssertionError(
                                f"V13 DARE field mismatch for {column} in round {round_number}"
                            )
                    elif cert_value != expected_value or round_value != expected_value:
                        raise AssertionError(
                            f"V13 DARE field mismatch for {column} in round {round_number}"
                        )
                    checkpoint_values = checkpoint[column]
                    if expected_value is None:
                        if not checkpoint_values.isna().all():
                            raise AssertionError(
                                f"V13 checkpoint null mismatch for {column} in round {round_number}"
                            )
                    elif not (checkpoint_values == expected_value).all():
                        raise AssertionError(
                            f"V13 checkpoint mismatch for {column} in round {round_number}"
                        )
                for column in dare_columns:
                    cert_value = certificate[column]
                    round_value = round_row[column]
                    if pd.isna(cert_value) and pd.isna(round_value):
                        pass
                    elif cert_value != round_value:
                        raise AssertionError(
                            f"V13 certificate/round mismatch for {column} in round {round_number}"
                        )
                    checkpoint_values = checkpoint[column]
                    if pd.isna(cert_value):
                        if not checkpoint_values.isna().all():
                            raise AssertionError(
                                f"V13 checkpoint null mismatch for {column} in round {round_number}"
                            )
                    elif not (checkpoint_values == cert_value).all():
                        raise AssertionError(
                            f"V13 checkpoint/certificate mismatch for {column} in round {round_number}"
                        )
                if not bool(certificate["deployment_dare_state_server_only"]):
                    raise AssertionError("V13 DARE state is not server-only")
                if any(
                    bool(certificate[column])
                    for column in (
                        "deployment_dare_labels_used",
                        "deployment_dare_scenario_metadata_used",
                        "deployment_dare_future_trace_used",
                    )
                ):
                    raise AssertionError("V13 DARE used a forbidden input")
            effective_primary_beta = (
                float(dare_observation.equivalent_beta)
                if expected_response
                and dare_observation is not None
                and dare_observation.equivalent_beta is not None
                else float(pulse_observation.override_beta)
                if expected_response
                and pulse_observation is not None
                and pulse_observation.override_beta is not None
                else (expected_trigger_beta if expected_response else primary_beta)
            )
            if not np.isclose(
                float(certificate["effective_primary_deployment_beta"]),
                effective_primary_beta,
                atol=0.0,
                rtol=0.0,
            ):
                raise AssertionError(f"V6 effective primary beta mismatch in round {round_number}")
            if not bool(certificate["deployment_hash_lineage_recorded"]):
                raise AssertionError(f"V6 deployment hash lineage missing in round {round_number}")
            before_hashes = json.loads(str(certificate["deployment_model_hashes_before_json"]))
            after_hashes = json.loads(str(certificate["deployment_model_hashes_after_json"]))
            for candidate in candidates.itertuples(index=False):
                beta = float(candidate.deployment_beta)
                beta_key = format(beta, ".17g")
                expected_effective_beta = (
                    float(dare_observation.equivalent_beta)
                    if expected_response
                    and dare_observation is not None
                    and dare_observation.equivalent_beta is not None
                    else float(pulse_observation.override_beta)
                    if expected_response
                    and pulse_observation is not None
                    and pulse_observation.override_beta is not None
                    else (expected_trigger_beta if expected_response else beta)
                )
                if not np.isclose(
                    float(candidate.effective_deployment_beta),
                    expected_effective_beta,
                    atol=0.0,
                    rtol=0.0,
                ):
                    raise AssertionError(f"V6 candidate beta mismatch in round {round_number}")
                if bool(candidate.deployment_synchronization_applied) != expected_sync:
                    raise AssertionError(f"Telemetry candidate synchronization mismatch in round {round_number}")
                if is_quarantine_protocol or is_pulse_protocol or is_dare_protocol:
                    if bool(candidate.deployment_quarantine_applied) != expected_quarantine:
                        raise AssertionError(f"Candidate quarantine mismatch in round {round_number}")
                    if bool(candidate.deployment_shift_response_applied) != expected_response:
                        raise AssertionError(f"Candidate response mismatch in round {round_number}")
                    if str(candidate.deployment_trigger_action) != expected_action:
                        raise AssertionError(f"Candidate action mismatch in round {round_number}")
                    if not np.isclose(
                        float(candidate.configured_trigger_deployment_beta),
                        expected_trigger_beta,
                        atol=0.0,
                        rtol=0.0,
                    ):
                        raise AssertionError(
                            f"Candidate trigger beta mismatch in round {round_number}"
                        )
                if is_pulse_protocol:
                    if bool(candidate.deployment_recovery_pulse_applied) != expected_recovery_pulse:
                        raise AssertionError(
                            f"V8 candidate recovery-pulse mismatch in round {round_number}"
                        )
                    if str(candidate.deployment_pulse_phase) != str(pulse_observation.phase):
                        raise AssertionError(
                            f"V8 candidate pulse phase mismatch in round {round_number}"
                        )
                if is_dare_protocol:
                    assert dare_observation is not None
                    if bool(candidate.deployment_recovery_envelope_applied) != expected_recovery_envelope:
                        raise AssertionError(
                            f"V13 candidate envelope mismatch in round {round_number}"
                        )
                    if bool(candidate.deployment_post_shift_tracking_applied) != expected_post_shift_tracking:
                        raise AssertionError(
                            f"V13 candidate tracking mismatch in round {round_number}"
                        )
                    if str(candidate.deployment_dare_phase) != str(dare_observation.phase):
                        raise AssertionError(
                            f"V13 candidate phase mismatch in round {round_number}"
                        )
                    expected_lambda = dare_observation.lambda_value
                    if expected_lambda is None:
                        if not pd.isna(candidate.deployment_dare_lambda):
                            raise AssertionError(
                                f"V13 candidate lambda mismatch in round {round_number}"
                            )
                    elif not np.isclose(
                        float(candidate.deployment_dare_lambda),
                        float(expected_lambda),
                        atol=0.0,
                        rtol=0.0,
                    ):
                        raise AssertionError(
                            f"V13 candidate lambda mismatch in round {round_number}"
                        )
                if str(candidate.deployment_model_hash_before) != str(before_hashes[beta_key]):
                    raise AssertionError(f"V6 pre-update hash mismatch in round {round_number}")
                if str(candidate.deployment_model_hash_after) != str(after_hashes[beta_key]):
                    raise AssertionError(f"V6 post-update hash mismatch in round {round_number}")
                if str(candidate.evaluation_model_hash) != str(after_hashes[beta_key]):
                    raise AssertionError(f"V6 evaluation hash mismatch in round {round_number}")
            primary_key = format(primary_beta, ".17g")
            if is_dare_protocol:
                assert dare_observation is not None
                actual_pre_hash = certificate["deployment_dare_pre_anchor_hash"]
                actual_post_hash = certificate["deployment_dare_post_anchor_hash"]
                actual_captured = bool(certificate["deployment_dare_pre_anchor_captured"])
                actual_released = bool(certificate["deployment_dare_pre_anchor_released"])
                count_before = int(certificate["deployment_dare_anchor_tensor_count_before"])
                count_after = int(certificate["deployment_dare_anchor_tensor_count_after"])
                bytes_before = int(certificate["deployment_dare_anchor_bytes_before"])
                bytes_after = int(certificate["deployment_dare_anchor_bytes_after"])
                if trigger and expected_response:
                    dare_pre_anchor_hash = str(before_hashes[primary_key])
                    dare_post_anchor_hash = str(round_row["global_model_hash"])
                    dare_anchor_tensor_count = count_after
                    dare_anchor_bytes = bytes_after
                    if not actual_captured or actual_released:
                        raise AssertionError(
                            f"V13 trigger anchor lifecycle mismatch in round {round_number}"
                        )
                    if count_after <= 0 or bytes_after <= 0:
                        raise AssertionError(
                            f"V13 trigger did not materialize exactly one model anchor in round {round_number}"
                        )
                    if before_hashes != after_hashes:
                        raise AssertionError(
                            f"V13 trigger hold changed deployment parameters in round {round_number}"
                        )
                elif expected_recovery_envelope:
                    if actual_captured:
                        raise AssertionError(
                            f"V13 recaptured an anchor inside the envelope in round {round_number}"
                        )
                    if count_before != dare_anchor_tensor_count or bytes_before != dare_anchor_bytes:
                        raise AssertionError(
                            f"V13 anchor cost changed inside the envelope in round {round_number}"
                        )
                    if dare_observation.remaining_after == 0:
                        if not actual_released or count_after != 0 or bytes_after != 0:
                            raise AssertionError(
                                f"V13 final envelope step did not release the anchor in round {round_number}"
                            )
                    elif actual_released or count_after != dare_anchor_tensor_count or bytes_after != dare_anchor_bytes:
                        raise AssertionError(
                            f"V13 anchor lifecycle changed early in round {round_number}"
                        )
                elif expected_post_shift_tracking:
                    if actual_captured or actual_released or any(
                        value != 0
                        for value in (count_before, count_after, bytes_before, bytes_after)
                    ):
                        raise AssertionError(
                            f"V13 persistent tracking retained an extra anchor in round {round_number}"
                        )
                elif not no_drift_quarantine:
                    if actual_captured or actual_released or any(
                        value != 0
                        for value in (count_before, count_after, bytes_before, bytes_after)
                    ):
                        raise AssertionError(
                            f"V13 S0 ordinary path allocated anchor state in round {round_number}"
                        )

                if dare_pre_anchor_hash is None:
                    if not (pd.isna(actual_pre_hash) and pd.isna(actual_post_hash)):
                        raise AssertionError(
                            f"V13 anchor hashes appeared before a trigger in round {round_number}"
                        )
                elif (
                    str(actual_pre_hash) != dare_pre_anchor_hash
                    or str(actual_post_hash) != dare_post_anchor_hash
                ):
                    raise AssertionError(
                        f"V13 anchor hash lineage mismatch in round {round_number}"
                    )
                if expected_post_shift_tracking or (
                    expected_recovery_envelope
                    and dare_observation.lambda_value == 1.0
                ):
                    if str(after_hashes[primary_key]) != str(round_row["global_model_hash"]):
                        raise AssertionError(
                            f"V13 lambda=1 did not track the global model in round {round_number}"
                        )
            if str(round_row["primary_deployment_model_hash_before"]) != str(
                before_hashes[primary_key]
            ):
                raise AssertionError(f"V6 primary pre-update hash mismatch in round {round_number}")
            if str(round_row["primary_deployment_model_hash_after"]) != str(
                after_hashes[primary_key]
            ):
                raise AssertionError(f"V6 primary post-update hash mismatch in round {round_number}")
            if trigger and protocol_version == R2C_V6_PROTOCOL_VERSION and any(
                str(value) != str(round_row["global_model_hash"])
                for value in after_hashes.values()
            ):
                raise AssertionError(f"V6 hard sync did not reach fast model in round {round_number}")
            if expected_quarantine:
                if before_hashes != after_hashes:
                    raise AssertionError(
                        f"V7 quarantine changed deployment parameters in round {round_number}"
                    )
                if round_number <= 1:
                    raise AssertionError("V7 quarantine cannot audit global advance in round 1")
                previous_global_hash = str(
                    round_frame.loc[
                        round_frame["round"] == round_number - 1, "global_model_hash"
                    ].iloc[0]
                )
                if str(round_row["global_model_hash"]) == previous_global_hash:
                    raise AssertionError(
                        f"V7 global training model did not advance in trigger round {round_number}"
                    )
            if completion_variant is not None:
                if bool(certificate["drift_quarantine_enabled"]) == no_drift_quarantine:
                    raise AssertionError(
                        f"Drift-quarantine ablation flag mismatch in round {round_number}"
                    )

    if len(checkpoints) != expected_checkpoint_rows:
        raise AssertionError(
            f"Checkpoint budget mismatch: {len(checkpoints)} != {expected_checkpoint_rows}"
        )
    numeric = checkpoints[
        [
            "value_hat",
            "finish_prob_hat",
            "score_q_hat",
            "score_variance",
            "radius_b",
            "lower_bound",
            "upper_bound",
        ]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise AssertionError("Non-finite score/certificate inputs")

    if job["mode"] == "formal":
        if str(manifest["source_kind"]) != "REPRODUCED" or job["evaluation_split"] != "test":
            raise AssertionError("Formal source/split contract violated")
        if job["scenario_id"] == "S4":
            recovery = result["recovery"]
            if not recovery["recovery_auc20_complete"] or recovery["recovery_deficit_auc20"] is None:
                raise AssertionError("Formal S4 lacks exact Recovery-deficit AUC@20")

    return {
        "status": "passed",
        "run_id": job["run_id"],
        "rounds": rounds,
        "round_rows": len(round_frame),
        "client_rows": len(client),
        "checkpoint_rows": len(checkpoints),
        "certificate_rows": len(certificates),
        "source_kind": str(manifest["source_kind"]),
        "protocol_version": protocol_version,
        "recovery_auc20_complete": bool(result["recovery"]["recovery_auc20_complete"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_run(args.run_dir), sort_keys=True))


if __name__ == "__main__":
    main()
