from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from r2c_baselines.config import NUM_CLIENTS, SELECTED_K
from r2c_baselines.logging_io import read_chunked_table
from r2c_baselines.r2c import history_balanced_conditional_targets
from r2c_baselines.r2c_v3 import PROTOCOL_VERSION as V3_PROTOCOL_VERSION
from r2c_baselines.r2c_v4 import validated_deployment_betas
from r2c_baselines.r2c_v5 import PROTOCOL_VERSION as V5_PROTOCOL_VERSION
from r2c_baselines.r2c_v6 import (
    DEFAULT_COOLDOWN_ROUNDS,
    DEFAULT_FRACTION_THRESHOLD,
    DEFAULT_LOG_RATIO_THRESHOLD,
    DEFAULT_MIN_COMPARABLE_CLIENTS,
)
from r2c_baselines.r2c_v14 import (
    DEPLOYMENT_RULE,
    FAST_BETA,
    PROTOCOL_VERSION,
    CausalMultiTimescaleRouter,
    candidate_recovery_rounds,
    candidate_stable_beta,
    validated_candidate_id,
    validated_fast_beta,
    validated_warmup_rounds,
)
from r2c_baselines.utils import canonical_json, sha256_text


def _unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if frame.duplicated(columns).any():
        raise AssertionError(f"Duplicate {label} primary key: {columns}")


def _equal(actual: Any, expected: Any) -> bool:
    if expected is None:
        return bool(pd.isna(actual))
    if isinstance(expected, (bool, np.bool_)):
        return bool(actual) is bool(expected)
    if isinstance(expected, (float, np.floating)):
        return bool(
            np.isclose(float(actual), float(expected), atol=0.0, rtol=0.0)
        )
    if isinstance(expected, (int, np.integer)):
        return int(actual) == int(expected)
    return str(actual) == str(expected)


def _assert_replicated(
    *,
    round_number: int,
    field: str,
    expected: Any,
    round_row: pd.Series,
    certificate: pd.Series,
    checkpoints: pd.DataFrame,
) -> None:
    for label, actual in (
        ("round", round_row[field]),
        ("certificate", certificate[field]),
    ):
        if not _equal(actual, expected):
            raise AssertionError(
                f"CMTR {label} {field} mismatch in round {round_number}: "
                f"{actual!r} != {expected!r}"
            )
    if not all(_equal(actual, expected) for actual in checkpoints[field]):
        raise AssertionError(
            f"CMTR checkpoint {field} mismatch in round {round_number}"
        )


def _normalized_record(row: pd.Series) -> dict[str, Any]:
    record = row.to_dict()
    record.pop("schema_version", None)
    record.pop("certificate_record_hash", None)
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        if pd.isna(value):
            normalized[key] = None
        elif isinstance(value, np.generic):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


def audit_run(run_dir: Path) -> dict[str, object]:
    required_files = ["job.json", "result.json", "run_manifest.parquet"]
    missing = [name for name in required_files if not (run_dir / name).is_file()]
    if missing:
        raise AssertionError(f"Missing v14 run files: {missing}")

    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    manifest_frame = pd.read_parquet(run_dir / "run_manifest.parquet")
    if len(manifest_frame) != 1:
        raise AssertionError("Run manifest must contain exactly one row")
    manifest = manifest_frame.iloc[0]
    config = dict(job.get("method_config", {}))
    if config.get("r2c_protocol_version") != PROTOCOL_VERSION:
        raise AssertionError("This auditor accepts only the frozen R2C-v14 protocol")
    if job.get("method_id") != "R2C-FL":
        raise AssertionError("R2C-v14 auditor received a non-R2C method")

    candidate_id = validated_candidate_id(config)
    fast_beta = validated_fast_beta(config)
    stable_beta = candidate_stable_beta(candidate_id)
    warmup_rounds = validated_warmup_rounds(config)
    recovery_rounds = candidate_recovery_rounds(candidate_id)
    configured_betas = validated_deployment_betas(
        config.get("r2c_v4_deployment_ema_betas", [])
    )
    if configured_betas != (FAST_BETA, stable_beta):
        raise AssertionError("CMTR deployment-state set differs from the frozen grid")
    primary_beta = float(config.get("r2c_v4_primary_deployment_beta"))
    if primary_beta != stable_beta:
        raise AssertionError("CMTR configured primary state is not the stable EMA")

    rounds = int(job["rounds"])
    round_frame = read_chunked_table(run_dir, "round_metrics")
    client = read_chunked_table(run_dir, "client_round_metrics")
    checkpoints = read_chunked_table(run_dir, "checkpoint_metrics")
    certificates = read_chunked_table(run_dir, "certificate_audit")
    deployment = read_chunked_table(run_dir, "deployment_candidate_metrics")
    expected_rows = {
        "round_metrics": rounds,
        "client_round_metrics": NUM_CLIENTS * rounds,
        "certificate_audit": rounds,
        "deployment_candidate_metrics": len(configured_betas) * rounds,
    }
    observed_rows = {
        "round_metrics": len(round_frame),
        "client_round_metrics": len(client),
        "certificate_audit": len(certificates),
        "deployment_candidate_metrics": len(deployment),
    }
    for table_name, expected in expected_rows.items():
        if observed_rows[table_name] != expected:
            raise AssertionError(
                f"{table_name} budget mismatch: "
                f"{observed_rows[table_name]} != {expected}"
            )
        index = result.get("table_indices", {}).get(table_name, {})
        if int(index.get("rows", -1)) != expected:
            raise AssertionError(f"Result index mismatch for {table_name}")

    _unique(round_frame, ["run_id", "round"], "round")
    _unique(client, ["run_id", "round", "client_id"], "client-round")
    _unique(certificates, ["run_id", "round"], "certificate")
    _unique(
        checkpoints,
        ["run_id", "round", "client_id", "checkpoint_j"],
        "checkpoint",
    )
    _unique(
        deployment,
        ["run_id", "round", "deployment_beta"],
        "deployment candidate",
    )
    expected_round_sequence = np.arange(1, rounds + 1, dtype=np.int64)
    if not np.array_equal(
        round_frame.sort_values("round")["round"].to_numpy(dtype=np.int64),
        expected_round_sequence,
    ):
        raise AssertionError("Round sequence is incomplete or reordered")
    if set(round_frame["run_id"].astype(str)) != {str(job["run_id"])}:
        raise AssertionError("Round run_id lineage mismatch")

    for seed_field in ("seed", "partition_seed", "trace_seed"):
        if int(manifest[seed_field]) != int(job[seed_field]):
            raise AssertionError(f"Manifest {seed_field} lineage mismatch")
    if str(manifest["upstream_commit"]) != PROTOCOL_VERSION:
        raise AssertionError("Manifest upstream protocol lineage mismatch")
    if str(manifest["checkpoint_serialization_version"]) != (
        "r2c-telemetry-multitimescale-router-single-checkpoint-v14"
    ):
        raise AssertionError("Manifest checkpoint schema does not identify v14")
    if int(manifest["round_budget"]) != rounds:
        raise AssertionError("Manifest round budget mismatch")
    if job["mode"] == "formal":
        if (
            str(manifest["source_kind"]) != "REPRODUCED"
            or job["evaluation_split"] != "test"
            or not bool(job.get("formal_test_access", False))
        ):
            raise AssertionError("Formal source/split/access contract violated")
    else:
        if (
            str(manifest["source_kind"]) != str(job["mode"]).upper()
            or job["evaluation_split"] != "validation"
            or bool(job.get("formal_test_access", False))
        ):
            raise AssertionError("Calibration source/split/access contract violated")

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
    cmtr_probe = CausalMultiTimescaleRouter.from_config(config)
    cmtr_columns = list(cmtr_probe.step(1, False).audit_fields())
    required_round_columns = telemetry_columns + cmtr_columns + [
        "deployment_cmtr_recovery_applied",
        "deployment_cmtr_warmup_applied",
        "deployment_cmtr_stable_applied",
        "deployment_synchronization_applied",
        "deployment_quarantine_applied",
        "deployment_shift_response_applied",
        "deployment_trigger_action",
        "configured_trigger_deployment_beta",
        "effective_primary_deployment_beta",
        "primary_deployment_model_hash_before",
        "primary_deployment_model_hash_after",
        "selected_deployment_beta",
        "selected_deployment_model_hash_after",
    ]
    for table_name, frame in (
        ("round", round_frame),
        ("certificate", certificates),
        ("checkpoint", checkpoints),
    ):
        missing_columns = [
            column
            for column in (
                telemetry_columns
                + cmtr_columns
                + [
                    "deployment_cmtr_recovery_applied",
                    "deployment_cmtr_warmup_applied",
                    "deployment_cmtr_stable_applied",
                    "deployment_synchronization_applied",
                    "deployment_quarantine_applied",
                    "deployment_shift_response_applied",
                    "deployment_trigger_action",
                    "configured_trigger_deployment_beta",
                    "effective_primary_deployment_beta",
                ]
            )
            if column not in frame.columns
        ]
        if missing_columns:
            raise AssertionError(f"Missing {table_name} audit columns: {missing_columns}")
    missing_round = [c for c in required_round_columns if c not in round_frame.columns]
    if missing_round:
        raise AssertionError(f"Missing v14 round columns: {missing_round}")

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
    no_drift_quarantine = bool(
        config.get("r2c_ablation_no_drift_quarantine", False)
    )
    expected_cooldown_before = 0
    expected_trigger_count = 0
    expected_checkpoint_rows = 0
    previous_global_hash = str(manifest["initial_model_hash"])
    previous_deployment_hashes = {
        format(float(beta), ".17g"): str(manifest["initial_model_hash"])
        for beta in configured_betas
    }
    selection_history_counts = np.zeros(NUM_CLIENTS, dtype=np.int64)
    router = CausalMultiTimescaleRouter.from_config(config)

    for round_number in range(1, rounds + 1):
        round_row = round_frame.loc[round_frame["round"] == round_number].iloc[0]
        round_clients = client.loc[client["round"] == round_number]
        certificate = certificates.loc[certificates["round"] == round_number].iloc[0]
        round_checkpoints = checkpoints.loc[
            checkpoints["round"] == round_number
        ].reset_index(drop=True)
        round_candidates = deployment.loc[
            deployment["round"] == round_number
        ].sort_values("deployment_beta")

        if len(round_clients) != NUM_CLIENTS:
            raise AssertionError(f"Client budget mismatch in round {round_number}")
        admitted = round_clients.loc[round_clients["admitted"].astype(bool)]
        selected = round_clients.loc[round_clients["selected"].astype(bool)]
        eligible = round_clients.loc[
            round_clients["aggregation_eligible"].astype(bool)
        ]
        admitted_n = min(20, int(round_row["available_clients"]))
        if len(admitted) != admitted_n:
            raise AssertionError(f"Admission cardinality mismatch in round {round_number}")
        if len(selected) != SELECTED_K or not set(selected["client_id"]).issubset(
            set(admitted["client_id"])
        ):
            raise AssertionError(f"Exact-K selection mismatch in round {round_number}")
        if not set(eligible["client_id"]).issubset(set(selected["client_id"])):
            raise AssertionError(f"Aggregation subset mismatch in round {round_number}")
        if sorted(admitted["admission_draw_position"].astype(int)) != list(
            range(admitted_n)
        ):
            raise AssertionError(f"Admission order is not replayable in round {round_number}")
        if not np.isclose(
            admitted["conditional_target_s"].astype(float).sum(),
            SELECTED_K,
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise AssertionError(f"Conditional targets do not sum to K in round {round_number}")
        inclusion = (
            admitted["admission_prob"].astype(float).to_numpy()
            * admitted["conditional_target_s"].astype(float).to_numpy()
        )
        if not np.allclose(
            inclusion,
            admitted["inclusion_prob_pi"].astype(float).to_numpy(),
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise AssertionError(f"Inclusion probabilities mismatch in round {round_number}")
        expected_hajek_sum = 0.0 if bool(certificate["no_op_round"]) else 1.0
        if not np.isclose(
            eligible["hajek_weight_normalized"].astype(float).sum(),
            expected_hajek_sum,
            atol=1.0e-8,
            rtol=0.0,
        ):
            raise AssertionError(f"Hajek normalization mismatch in round {round_number}")
        if int(certificate["aggregation_subset_n"]) != len(eligible):
            raise AssertionError(f"Aggregation count mismatch in round {round_number}")

        expected_checkpoint_rows += admitted_n
        if (
            len(round_checkpoints) != admitted_n
            or int(certificate["commit_j"]) != 1
            or set(round_checkpoints["checkpoint_j"].astype(int)) != {1}
        ):
            raise AssertionError(f"Single-checkpoint budget mismatch in round {round_number}")
        if not (round_checkpoints["protocol_version"] == PROTOCOL_VERSION).all():
            raise AssertionError(f"Checkpoint protocol mismatch in round {round_number}")
        if str(certificate["protocol_version"]) != PROTOCOL_VERSION:
            raise AssertionError(f"Certificate protocol mismatch in round {round_number}")
        if not (
            (round_checkpoints["selection_protocol_version"] == V5_PROTOCOL_VERSION).all()
            and str(certificate["selection_protocol_version"]) == V5_PROTOCOL_VERSION
            and (round_checkpoints["fast_update_protocol_version"] == V3_PROTOCOL_VERSION).all()
            and str(certificate["fast_update_protocol_version"]) == V3_PROTOCOL_VERSION
            and (round_checkpoints["deployment_protocol_version"] == PROTOCOL_VERSION).all()
            and str(certificate["deployment_protocol_version"]) == PROTOCOL_VERSION
        ):
            raise AssertionError(f"Protocol lineage mismatch in round {round_number}")
        if (
            str(certificate["deployment_rule"]) != DEPLOYMENT_RULE
            or not bool(certificate["deployment_state_server_only"])
            or bool(certificate["selection_history_labels_used"])
            or not bool(certificate["selection_history_state_server_only"])
        ):
            raise AssertionError(f"Server-only deployment/selection contract failed in round {round_number}")

        checkpoint_clients = round_checkpoints["client_id"].astype(int).to_numpy()
        expected_counts = selection_history_counts[checkpoint_clients]
        if not np.array_equal(
            round_checkpoints["selection_history_count_before"].astype(int).to_numpy(),
            expected_counts,
        ):
            raise AssertionError(f"Selection-history counts mismatch in round {round_number}")
        expected_counts_hash = sha256_text(
            canonical_json(selection_history_counts.astype(int).tolist())
        )
        if str(certificate["history_counts_hash"]) != expected_counts_hash:
            raise AssertionError(f"Selection-history hash mismatch in round {round_number}")
        history_mix = float(certificate["history_target_mix"])
        history_temperature = float(certificate["history_target_temperature"])
        adaptive_floor = float(certificate["adaptive_floor_fraction"])
        anchor, history, final, _ = history_balanced_conditional_targets(
            round_checkpoints["score_q_hat"].astype(float).to_numpy(),
            expected_counts,
            SELECTED_K,
            float(config["r2c_v2_temperature"]),
            history_temperature,
            adaptive_floor * SELECTED_K / len(round_checkpoints),
            history_mix,
        )
        for expected_values, column in (
            (anchor, "anchor_conditional_target_s"),
            (history, "history_conditional_target_s"),
        ):
            if not np.allclose(
                expected_values,
                round_checkpoints[column].astype(float).to_numpy(),
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise AssertionError(f"{column} mismatch in round {round_number}")
        admitted_by_checkpoint = (
            round_clients.set_index("client_id").loc[checkpoint_clients].reset_index()
        )
        if not np.allclose(
            final,
            admitted_by_checkpoint["conditional_target_s"].astype(float).to_numpy(),
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise AssertionError(f"Final history target mismatch in round {round_number}")
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
            raise AssertionError(f"History-target hash mismatch in round {round_number}")

        if round_checkpoints["round_start_model_hash"].nunique() != 1:
            raise AssertionError(f"Multiple round-start hashes in round {round_number}")
        if str(round_checkpoints["round_start_model_hash"].iloc[0]) != previous_global_hash:
            raise AssertionError(f"Global model lineage mismatch in round {round_number}")
        if str(certificate["round_start_model_hash"]) != previous_global_hash:
            raise AssertionError(f"Certificate model lineage mismatch in round {round_number}")
        if int(round_checkpoints["admission_seed"].iloc[0]) != int(
            certificate["admission_seed"]
        ):
            raise AssertionError(f"Admission-seed lineage mismatch in round {round_number}")
        previous_global_hash = str(round_row["global_model_hash"])

        record_hash = str(certificate["certificate_record_hash"])
        if record_hash != sha256_text(canonical_json(_normalized_record(certificate))):
            raise AssertionError(f"Certificate record hash mismatch in round {round_number}")

        for column in telemetry_columns:
            expected_value = certificate[column]
            _assert_replicated(
                round_number=round_number,
                field=column,
                expected=expected_value,
                round_row=round_row,
                certificate=certificate,
                checkpoints=round_checkpoints,
            )
        if int(certificate["telemetry_shift_round"]) != round_number:
            raise AssertionError(f"Telemetry round mismatch in round {round_number}")
        if not (
            np.isclose(
                float(certificate["telemetry_shift_log_ratio_threshold"]),
                expected_log_threshold,
                atol=0.0,
                rtol=0.0,
            )
            and np.isclose(
                float(certificate["telemetry_shift_fraction_threshold"]),
                expected_fraction_threshold,
                atol=0.0,
                rtol=0.0,
            )
            and int(certificate["telemetry_shift_min_comparable_clients"])
            == expected_min_comparable
        ):
            raise AssertionError("Telemetry thresholds differ from frozen config")
        if (
            not bool(certificate["telemetry_shift_state_server_only"])
            or bool(certificate["telemetry_shift_labels_used"])
            or bool(certificate["telemetry_shift_scenario_metadata_used"])
        ):
            raise AssertionError("Telemetry detector used a forbidden input")
        comparable = int(certificate["telemetry_shift_comparable_clients"])
        changed = int(certificate["telemetry_shift_changed_clients"])
        fraction = float(certificate["telemetry_shift_changed_fraction"])
        if changed < 0 or changed > comparable:
            raise AssertionError(f"Invalid telemetry count in round {round_number}")
        if not np.isclose(
            fraction,
            changed / comparable if comparable else 0.0,
            atol=1.0e-15,
            rtol=0.0,
        ):
            raise AssertionError(f"Telemetry fraction mismatch in round {round_number}")
        if int(certificate["telemetry_shift_cooldown_before"]) != expected_cooldown_before:
            raise AssertionError(f"Telemetry cooldown lineage mismatch in round {round_number}")
        detector_trigger = bool(
            expected_cooldown_before == 0
            and comparable >= expected_min_comparable
            and fraction >= expected_fraction_threshold
        )
        if bool(certificate["telemetry_shift_trigger"]) != detector_trigger:
            raise AssertionError(f"Telemetry trigger predicate mismatch in round {round_number}")
        if detector_trigger:
            expected_trigger_count += 1
            expected_cooldown_after = cooldown_rounds
        else:
            expected_cooldown_after = max(0, expected_cooldown_before - 1)
        if (
            int(certificate["telemetry_shift_cooldown_after"])
            != expected_cooldown_after
            or int(certificate["telemetry_shift_synchronization_count"])
            != expected_trigger_count
        ):
            raise AssertionError(f"Telemetry state update mismatch in round {round_number}")
        expected_cooldown_before = expected_cooldown_after

        observation = router.step(
            round_number, detector_trigger and not no_drift_quarantine
        )
        for field, expected_value in observation.audit_fields().items():
            _assert_replicated(
                round_number=round_number,
                field=field,
                expected=expected_value,
                round_row=round_row,
                certificate=certificate,
                checkpoints=round_checkpoints,
            )
        if (
            observation.configured_candidate_id != candidate_id
            or observation.configured_fast_beta != fast_beta
            or observation.configured_stable_beta != stable_beta
            or observation.configured_warmup_rounds != warmup_rounds
            or observation.configured_recovery_rounds != recovery_rounds
        ):
            raise AssertionError("CMTR reconstructed configuration mismatch")
        if (
            not observation.state_server_only
            or observation.labels_used
            or observation.validation_predictions_used
            or observation.test_predictions_used
            or observation.scenario_metadata_used
            or observation.event_round_used
            or observation.future_trace_used
            or observation.raw_global_deployment_used
        ):
            raise AssertionError("CMTR used a forbidden input or raw global deployment")

        expected_action = (
            "hold"
            if observation.hold_applied
            else "cmtr_recovery_fast"
            if observation.recovery_fast_applied
            else "cmtr_warmup_fast"
            if observation.warmup_fast_applied
            else "cmtr_stable"
        )
        convenience = {
            "deployment_cmtr_recovery_applied": observation.recovery_fast_applied,
            "deployment_cmtr_warmup_applied": observation.warmup_fast_applied,
            "deployment_cmtr_stable_applied": observation.stable_route_applied,
            "deployment_synchronization_applied": False,
            "deployment_quarantine_applied": observation.hold_applied,
            "deployment_shift_response_applied": observation.response_applied,
            "deployment_trigger_action": expected_action,
            "configured_trigger_deployment_beta": 1.0,
            "effective_primary_deployment_beta": observation.update_beta_for(
                observation.selected_beta
            ),
        }
        for field, expected_value in convenience.items():
            _assert_replicated(
                round_number=round_number,
                field=field,
                expected=expected_value,
                round_row=round_row,
                certificate=certificate,
                checkpoints=round_checkpoints,
            )

        if len(round_candidates) != 2:
            raise AssertionError(f"CMTR state count mismatch in round {round_number}")
        if sorted(round_candidates["deployment_beta"].astype(float)) != list(
            configured_betas
        ):
            raise AssertionError(f"CMTR beta set mismatch in round {round_number}")
        if int(round_candidates["is_primary"].astype(bool).sum()) != 1:
            raise AssertionError(f"CMTR fixed-primary count mismatch in round {round_number}")
        fixed_primary = round_candidates.loc[
            round_candidates["is_primary"].astype(bool)
        ].iloc[0]
        if float(fixed_primary["deployment_beta"]) != stable_beta:
            raise AssertionError(f"CMTR fixed primary mismatch in round {round_number}")
        if int(round_candidates["is_selected_for_deployment"].astype(bool).sum()) != 1:
            raise AssertionError(f"CMTR selected-state count mismatch in round {round_number}")
        selected_candidate = round_candidates.loc[
            round_candidates["is_selected_for_deployment"].astype(bool)
        ].iloc[0]
        if float(selected_candidate["deployment_beta"]) != observation.selected_beta:
            raise AssertionError(f"CMTR selected beta mismatch in round {round_number}")
        before_hashes = json.loads(str(certificate["deployment_model_hashes_before_json"]))
        after_hashes = json.loads(str(certificate["deployment_model_hashes_after_json"]))
        if not bool(certificate["deployment_hash_lineage_recorded"]):
            raise AssertionError(f"Deployment hash lineage absent in round {round_number}")
        for candidate in round_candidates.itertuples(index=False):
            beta = float(candidate.deployment_beta)
            beta_key = format(beta, ".17g")
            if not np.isclose(
                float(candidate.effective_deployment_beta),
                observation.update_beta_for(beta),
                atol=0.0,
                rtol=0.0,
            ):
                raise AssertionError(f"CMTR update beta mismatch in round {round_number}")
            if (
                str(candidate.deployment_cmtr_phase) != observation.phase
                or str(candidate.deployment_cmtr_candidate_id) != candidate_id
                or str(candidate.deployment_cmtr_selected_role)
                != observation.selected_role
                or float(candidate.deployment_cmtr_selected_beta)
                != observation.selected_beta
            ):
                raise AssertionError(f"CMTR candidate-route audit mismatch in round {round_number}")
            if str(candidate.deployment_model_hash_before) != str(
                before_hashes[beta_key]
            ) or str(candidate.deployment_model_hash_after) != str(after_hashes[beta_key]):
                raise AssertionError(f"CMTR candidate hash ledger mismatch in round {round_number}")
            if str(candidate.evaluation_model_hash) != str(after_hashes[beta_key]):
                raise AssertionError(f"CMTR candidate evaluation hash mismatch in round {round_number}")
            if str(before_hashes[beta_key]) != previous_deployment_hashes[beta_key]:
                raise AssertionError(f"CMTR inter-round hash lineage mismatch in round {round_number}")
            previous_deployment_hashes[beta_key] = str(after_hashes[beta_key])
        if observation.hold_applied and before_hashes != after_hashes:
            raise AssertionError(f"CMTR trigger hold mutated an EMA in round {round_number}")

        selected_key = format(observation.selected_beta, ".17g")
        primary_key = format(stable_beta, ".17g")
        if (
            str(round_row["primary_deployment_model_hash_before"])
            != str(before_hashes[primary_key])
            or str(round_row["primary_deployment_model_hash_after"])
            != str(after_hashes[primary_key])
            or str(round_row["selected_deployment_model_hash_after"])
            != str(after_hashes[selected_key])
            or str(round_row["evaluation_model_hash"])
            != str(after_hashes[selected_key])
            or str(round_row["evaluation_model_role"])
            != "server_only_multitimescale_ema_router"
        ):
            raise AssertionError(f"CMTR selected/fixed hash lineage mismatch in round {round_number}")
        if not (
            np.isclose(
                float(selected_candidate["test_accuracy"]),
                float(round_row["test_accuracy"]),
                atol=0.0,
                rtol=0.0,
            )
            and np.isclose(
                float(selected_candidate["test_loss"]),
                float(round_row["test_loss"]),
                atol=0.0,
                rtol=0.0,
            )
            and str(selected_candidate["evaluation_model_hash"])
            == str(round_row["evaluation_model_hash"])
        ):
            raise AssertionError(f"Selected deployment metrics mismatch in round {round_number}")

        selection_history_counts[
            selected["client_id"].astype(int).to_numpy()
        ] += 1

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

    if job["scenario_id"] == "S4" and rounds >= 520:
        before_offsets = set(
            round_frame.loc[
                round_frame["auc20_window_role"] == "before", "event_offset_round"
            ].astype(int)
        )
        after_offsets = set(
            round_frame.loc[
                round_frame["auc20_window_role"] == "after", "event_offset_round"
            ].astype(int)
        )
        if before_offsets != set(range(-20, 0)) or after_offsets != set(range(1, 21)):
            raise AssertionError("S4 event window is not exact -20..-1/+1..+20")
        recovery = result["recovery"]
        if (
            not bool(recovery["recovery_auc20_complete"])
            or recovery["recovery_deficit_auc20"] is None
        ):
            raise AssertionError("S4 lacks exact Recovery-deficit AUC@20")

    return {
        "status": "passed",
        "run_id": str(job["run_id"]),
        "protocol_version": PROTOCOL_VERSION,
        "candidate_id": candidate_id,
        "rounds": rounds,
        "round_rows": len(round_frame),
        "client_rows": len(client),
        "checkpoint_rows": len(checkpoints),
        "certificate_rows": len(certificates),
        "deployment_candidate_rows": len(deployment),
        "source_kind": str(manifest["source_kind"]),
        "recovery_auc20_complete": bool(
            result.get("recovery", {}).get("recovery_auc20_complete", False)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_run(args.run_dir), sort_keys=True))


if __name__ == "__main__":
    main()
