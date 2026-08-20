from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from .config import CANDIDATE_M, NUM_CLIENTS, SELECTED_K
from .data import FederatedData
from .r2c import (
    CHECKPOINT_SERIALIZATION_VERSION,
    TELEMETRY_BYTES_PER_CHECKPOINT,
    R2CRoundResult,
    _parameter_norms,
    _sampled_state_checksums,
    conditional_inclusion_targets,
    history_balanced_conditional_targets,
    pivotal_sample,
)
from .traces import Trace
from .training import LocalTrainer
from .utils import canonical_json, sha256_text


PROTOCOL_VERSION = "anchor-bounded-overhead-v2"


def _robust_location_scale(values: np.ndarray, floor: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    location = float(np.median(values))
    mad = float(np.median(np.abs(values - location)))
    return location, max(float(floor), 1.4826 * mad)


def _score_from_anchor(
    base0: np.ndarray,
    base1: np.ndarray,
    current0: np.ndarray,
    current1: np.ndarray,
    finish_prob: np.ndarray,
    scale_floor: float,
    value_clip: float,
    finish_weight: float,
    radius_multiplier: float,
) -> dict[str, np.ndarray | float]:
    gain0_vectors = base0[None, :] - current0
    gain1_vectors = base1[None, :] - current1
    gain0 = gain0_vectors.mean(axis=1)
    gain1 = gain1_vectors.mean(axis=1)
    raw_gain = 0.5 * (gain0 + gain1)
    location, scale = _robust_location_scale(raw_gain, scale_floor)
    z0 = np.clip((gain0 - location) / scale, -value_clip, value_clip)
    z1 = np.clip((gain1 - location) / scale, -value_clip, value_clip)
    value_hat = np.clip(0.5 * (z0 + z1), -value_clip, value_clip)
    finish_term = float(finish_weight) * np.log(np.maximum(finish_prob, 1.0e-4))
    score = value_hat + finish_term

    combined = np.concatenate([gain0_vectors, gain1_vectors], axis=1) / scale
    score_variance = np.var(combined, axis=1, ddof=1)
    standard_error = np.sqrt(score_variance / combined.shape[1])
    fold_disagreement = 0.5 * np.abs(z0 - z1)
    radius = float(radius_multiplier) * standard_error + fold_disagreement

    fold0_score = z0 + finish_term
    fold1_score = z1 + finish_term
    order0 = np.lexsort((np.arange(len(score)), -fold0_score))
    order1 = np.lexsort((np.arange(len(score)), -fold1_score))
    agreement = len(set(order0[:SELECTED_K].tolist()) & set(order1[:SELECTED_K].tolist())) / float(
        SELECTED_K
    )
    return {
        "gain0_vectors": gain0_vectors,
        "gain1_vectors": gain1_vectors,
        "raw_gain": raw_gain,
        "scale": float(scale),
        "value_hat": value_hat,
        "score": score,
        "score_variance": score_variance,
        "radius": radius,
        "agreement": float(agreement),
    }


def run_r2c_v2_round(
    model: torch.nn.Module,
    trainer: LocalTrainer,
    data: FederatedData,
    trace: Trace,
    round_number: int,
    available: np.ndarray,
    sample_counts: np.ndarray,
    learning_rate: float,
    payload_bytes: int,
    seed: int,
    config: dict[str, Any],
    run_id: str,
    round_start_model_hash: str,
    full_logging: bool,
    selection_history_counts: np.ndarray | None = None,
) -> R2CRoundResult:
    """Bounded-overhead R2C with a shared validation anchor and stable weights.

    All M candidates perform a short useful prefix once.  A common, disjoint
    validation anchor ranks the resulting states, exact-K pivotal sampling
    commits K candidates, and only those candidates resume.  Aggregation uses
    capped, low-power propensity correction to avoid the effective-sample-size
    collapse observed in the original completion-corrected Hájek rule.
    """

    available = np.asarray(available, dtype=np.int64)
    if len(available) < SELECTED_K:
        raise RuntimeError(
            f"insufficient_candidates: round={round_number}, available={len(available)}, k={SELECTED_K}"
        )
    m = min(CANDIDATE_M, len(available))
    admission_seed = int(
        np.random.SeedSequence(
            [seed, round_number, int(round_start_model_hash[:8], 16), 0xAD115510]
        ).generate_state(1, dtype=np.uint32)[0]
    )
    admission_rng = np.random.default_rng(admission_seed)
    admitted = np.asarray(admission_rng.choice(available, size=m, replace=False), dtype=np.int64)
    admission_probability = float(m / len(available))
    data_states = [trace.data_state_id(round_number, int(client_id)) for client_id in admitted]

    local_steps = int(trainer.local_steps)
    scout_steps = int(config.get("r2c_v2_scout_steps", 1))
    if not 1 <= scout_steps < local_steps:
        raise ValueError("r2c_v2_scout_steps must be in [1, local_steps)")
    anchor_per_fold = int(config.get("r2c_v2_anchor_per_fold", 16))
    eval_microbatch = int(config.get("r2c_eval_microbatch", trainer.max_parallel_clients))
    value_clip = float(config.get("r2c_v2_value_clip", 4.0))
    scale_floor = float(config.get("r2c_v2_scale_floor", 1.0e-4))
    temperature = float(config.get("r2c_v2_temperature", 0.20))
    base_floor_fraction = float(config.get("r2c_v2_floor_fraction", 0.10))
    uncertainty_floor_fraction = float(config.get("r2c_v2_uncertainty_floor_fraction", 0.35))
    finish_weight = float(config.get("r2c_v2_finish_weight", 0.75))
    finish_scale = float(config.get("r2c_v2_finish_scale", 0.08))
    radius_multiplier = float(config.get("r2c_v2_radius_multiplier", 1.0))
    agreement_threshold = float(config.get("r2c_v2_agreement_threshold", 0.80))
    propensity_power = float(config.get("r2c_v2_propensity_power", 0.0))
    completion_power = float(config.get("r2c_v2_completion_power", 0.0))
    weight_cap = float(config.get("r2c_v2_weight_cap", 2.0))
    delta_clip = float(config.get("r2c_delta_clip", math.inf))
    audit_replay = bool(config.get("r2c_v2_audit_replay", full_logging))
    history_enabled = selection_history_counts is not None
    history_mix = float(config.get("r2c_v5_history_mix", 0.0))
    history_temperature = float(config.get("r2c_v5_history_temperature", 1.0))
    no_reusable_prefix = bool(config.get("r2c_ablation_no_reusable_prefix", False))
    no_finishability = bool(config.get("r2c_ablation_no_finishability", False))
    no_valid_crossfit = bool(config.get("r2c_ablation_no_valid_crossfit", False))
    effective_finish_weight = 0.0 if no_finishability else finish_weight
    if temperature <= 0 or finish_scale <= 0 or delta_clip <= 0 or weight_cap < 1:
        raise ValueError("Invalid R2C-v2 numerical configuration")
    if not 0 <= propensity_power <= 1 or not 0 <= completion_power <= 1:
        raise ValueError("Correction powers must lie in [0, 1]")
    if not 0.0 <= history_mix <= 1.0 or history_temperature <= 0.0:
        raise ValueError("Invalid R2C-v5 history-target configuration")
    if history_mix > 0.0 and not history_enabled:
        raise ValueError("Positive history-target mixing requires pre-round selection counts")
    if history_enabled:
        selection_history_counts = np.asarray(selection_history_counts, dtype=np.int64)
        if selection_history_counts.shape != (NUM_CLIENTS,) or np.any(selection_history_counts < 0):
            raise ValueError("Selection-history state must be a nonnegative NUM_CLIENTS vector")

    idx = round_number - 1
    deadline_s = float(trace.deadline[idx])
    _, _, predicted_total = trace.simulated_times(
        round_number, admitted, local_steps, payload_bytes
    )
    slack_scale = max(1.0e-9, finish_scale * deadline_s)
    finish_prob = 1.0 / (
        1.0 + np.exp(-np.clip((deadline_s - predicted_total) / slack_scale, -20.0, 20.0))
    )

    algorithm_wall_s = 0.0
    algorithm_gpu_s = 0.0
    checkpoint_write_wall_s = 0.0
    checkpoint_read_wall_s = 0.0
    telemetry_upload_bytes = m * TELEMETRY_BYTES_PER_CHECKPOINT

    base_params = trainer.stacked_global(1)
    base0_values, wall0, gpu0 = trainer.r2c_anchor_losses(
        base_params, 0, anchor_per_fold, eval_microbatch
    )
    if no_valid_crossfit:
        base1_values = base0_values.copy()
        wall1 = 0.0
        gpu1 = 0.0
    else:
        base1_values, wall1, gpu1 = trainer.r2c_anchor_losses(
            base_params, 1, anchor_per_fold, eval_microbatch
        )
    algorithm_wall_s += wall0 + wall1
    algorithm_gpu_s += gpu0 + gpu1
    base0 = base0_values[0]
    base1 = base1_values[0]
    del base_params

    params = trainer.stacked_global(m)
    global_start = trainer.stacked_global(m)
    first_local_loss: np.ndarray | None = None
    latest_local_loss = np.full(m, np.nan, dtype=np.float64)
    for local_step in range(scout_steps):
        step_result = trainer.r2c_train_step(
            params,
            admitted,
            round_number,
            data_states,
            local_step,
            learning_rate,
        )
        if first_local_loss is None:
            first_local_loss = step_result.loss.copy()
        latest_local_loss = step_result.loss.copy()
        algorithm_wall_s += step_result.wall_s
        algorithm_gpu_s += step_result.gpu_s
    assert first_local_loss is not None

    write_start = time.perf_counter()
    checkpoint_cpu = {name: value.detach().cpu().clone() for name, value in params.items()}
    checkpoint_write_wall_s = time.perf_counter() - write_start
    algorithm_wall_s += checkpoint_write_wall_s
    commit_checksums = _sampled_state_checksums(checkpoint_cpu, admitted)

    current0, wall0, gpu0 = trainer.r2c_anchor_losses(
        params, 0, anchor_per_fold, eval_microbatch
    )
    if no_valid_crossfit:
        current1 = current0.copy()
        wall1 = 0.0
        gpu1 = 0.0
    else:
        current1, wall1, gpu1 = trainer.r2c_anchor_losses(
            params, 1, anchor_per_fold, eval_microbatch
        )
    algorithm_wall_s += wall0 + wall1
    algorithm_gpu_s += gpu0 + gpu1
    score_data = _score_from_anchor(
        base0,
        base1,
        current0,
        current1,
        finish_prob,
        scale_floor,
        value_clip,
        effective_finish_weight,
        radius_multiplier,
    )
    value_hat = np.asarray(score_data["value_hat"], dtype=np.float64)
    score = np.asarray(score_data["score"], dtype=np.float64)
    score_variance = np.asarray(score_data["score_variance"], dtype=np.float64)
    radius = np.asarray(score_data["radius"], dtype=np.float64)
    lower = score - radius
    upper = score + radius
    agreement = float(score_data["agreement"])
    order = np.lexsort((admitted, -score))
    top_positions = order[:SELECTED_K]
    outsider_positions = order[SELECTED_K:]
    topk = admitted[top_positions]
    gamma = (
        None
        if len(outsider_positions) == 0
        else float(np.min(lower[top_positions]) - np.max(upper[outsider_positions]))
    )
    raw_certified = bool(
        len(outsider_positions) == 0
        or (gamma is not None and gamma > 0.0 and agreement >= agreement_threshold)
    )
    certificate_valid = not no_valid_crossfit
    certified = bool(raw_certified and certificate_valid)

    adaptive_floor_fraction = float(
        np.clip(
            base_floor_fraction + uncertainty_floor_fraction * (1.0 - agreement),
            0.0,
            0.95,
        )
    )
    floor = adaptive_floor_fraction * SELECTED_K / m
    if history_enabled:
        assert selection_history_counts is not None
        admitted_history_counts = selection_history_counts[admitted].astype(np.int64)
        anchor_targets, history_targets, targets, history_scores = (
            history_balanced_conditional_targets(
                score,
                admitted_history_counts,
                SELECTED_K,
                temperature,
                history_temperature,
                floor,
                history_mix,
            )
        )
    else:
        admitted_history_counts = np.zeros(m, dtype=np.int64)
        anchor_targets = conditional_inclusion_targets(score, SELECTED_K, temperature, floor)
        history_targets = anchor_targets.copy()
        history_scores = np.zeros(m, dtype=np.float64)
        targets = anchor_targets
    score_record = {
        "protocol_version": PROTOCOL_VERSION,
        "round": round_number,
        "round_start_model_hash": round_start_model_hash,
        "admission_seed": admission_seed,
        "client_ids": admitted.tolist(),
        "scores": score.tolist(),
        "radii": radius.tolist(),
        "targets": targets.tolist(),
        "anchor_agreement": agreement,
    }
    if history_mix > 0.0:
        score_record.update(
            {
                "history_target_rule": "negative_log1p_cumulative_selection_count",
                "history_target_mix": history_mix,
                "history_target_temperature": history_temperature,
                "selection_history_counts": admitted_history_counts.tolist(),
                "anchor_targets": anchor_targets.tolist(),
                "history_targets": history_targets.tolist(),
            }
        )
    score_record_hash = sha256_text(canonical_json(score_record))
    pivotal_seed = int(
        np.random.SeedSequence(
            [seed, round_number, scout_steps, int(score_record_hash[:8], 16), 0xD37111E]
        ).generate_state(1, dtype=np.uint32)[0]
    )
    selected_positions = pivotal_sample(
        targets, SELECTED_K, np.random.default_rng(pivotal_seed)
    )
    selected = np.sort(admitted[selected_positions])
    position_map = {int(client_id): position for position, client_id in enumerate(admitted)}
    selected_positions = np.asarray(
        [position_map[int(client_id)] for client_id in selected], dtype=np.int64
    )

    selected_index_cpu = torch.from_numpy(selected_positions)
    if no_reusable_prefix:
        selected_index_gpu = torch.as_tensor(selected_positions, device=trainer.device)
        selected_params = {
            name: value.index_select(0, selected_index_gpu).clone()
            for name, value in global_start.items()
        }
        resume_start_step = 0
    else:
        read_start = time.perf_counter()
        selected_params = {
            name: value.index_select(0, selected_index_cpu).to(trainer.device)
            for name, value in checkpoint_cpu.items()
        }
        checkpoint_read_wall_s = time.perf_counter() - read_start
        algorithm_wall_s += checkpoint_read_wall_s
        resume_start_step = scout_steps
    selected_states = [trace.data_state_id(round_number, int(client_id)) for client_id in selected]
    selected_latest_loss = latest_local_loss[selected_positions].copy()
    for local_step in range(resume_start_step, local_steps):
        step_result = trainer.r2c_train_step(
            selected_params,
            selected,
            round_number,
            selected_states,
            local_step,
            learning_rate,
        )
        selected_latest_loss = step_result.loss.copy()
        algorithm_wall_s += step_result.wall_s
        algorithm_gpu_s += step_result.gpu_s

    selected_start = {
        name: value.index_select(
            0, torch.as_tensor(selected_positions, device=value.device)
        )
        for name, value in global_start.items()
    }
    raw_norms = _parameter_norms(selected_params, selected_start)
    clipped_norms = np.minimum(raw_norms, delta_clip)
    clip_scales = np.minimum(1.0, delta_clip / np.maximum(raw_norms, 1.0e-12))

    selected_total = predicted_total[selected_positions]
    eligible = selected[selected_total <= deadline_s]
    eligible_set = set(int(value) for value in eligible)
    target_selected = targets[selected_positions]
    finish_selected = finish_prob[selected_positions]
    inclusion_selected = admission_probability * target_selected
    effective_selected = inclusion_selected * np.maximum(finish_selected, 1.0e-4)
    raw_weights = np.zeros(len(selected), dtype=np.float64)
    for position, client_id in enumerate(selected):
        if int(client_id) not in eligible_set:
            continue
        denominator = (
            max(float(target_selected[position]), 1.0e-4) ** propensity_power
            * max(float(finish_selected[position]), 1.0e-4) ** completion_power
        )
        raw_weights[position] = float(sample_counts[int(client_id)]) / denominator
    positive = raw_weights > 0
    if np.any(positive):
        cap_value = weight_cap * float(np.median(raw_weights[positive]))
        raw_weights[positive] = np.minimum(raw_weights[positive], cap_value)
    normalized_weights = (
        raw_weights / raw_weights.sum() if raw_weights.sum() > 0 else raw_weights.copy()
    )

    aggregate_start = time.perf_counter()
    with torch.no_grad():
        for name, global_parameter in model.named_parameters():
            delta_tensor = selected_params[name] - selected_start[name]
            scale_tensor = torch.as_tensor(
                clip_scales, device=delta_tensor.device, dtype=delta_tensor.dtype
            ).view(len(selected), *([1] * global_parameter.ndim))
            weight_tensor = torch.as_tensor(
                normalized_weights, device=delta_tensor.device, dtype=delta_tensor.dtype
            ).view(len(selected), *([1] * global_parameter.ndim))
            global_parameter.add_((delta_tensor * scale_tensor * weight_tensor).sum(dim=0))
    aggregate_wall_s = time.perf_counter() - aggregate_start

    checkpoint_rows: list[dict[str, Any]] = []
    if full_logging:
        ranks = np.empty(m, dtype=np.int64)
        ranks[order] = np.arange(1, m + 1, dtype=np.int64)
        gain0_vectors = np.asarray(score_data["gain0_vectors"], dtype=np.float64)
        gain1_vectors = np.asarray(score_data["gain1_vectors"], dtype=np.float64)
        state_bytes = int(
            sum(value[0].numel() * value[0].element_size() for value in checkpoint_cpu.values())
        )
        for position, client_id in enumerate(admitted):
            values = np.concatenate([gain0_vectors[position], gain1_vectors[position]])
            checkpoint_rows.append(
                {
                    "run_id": run_id,
                    "round": round_number,
                    "client_id": int(client_id),
                    "round_start_model_hash": round_start_model_hash,
                    "admission_seed": admission_seed,
                    "checkpoint_j": 1,
                    "tau_steps": scout_steps,
                    "tau_fraction": scout_steps / local_steps,
                    "checkpoint_elapsed_s": float(
                        trace.step_before[idx, int(client_id)] * scout_steps
                    ),
                    "checkpoint_batch_elapsed_s": float(
                        np.max(trace.step_before[idx, admitted]) * scout_steps
                    ),
                    "report_fold": -1,
                    "scale_fold": -1,
                    "value_sample_n": int(len(values)),
                    "value_sample_sum": float(values.sum()),
                    "value_sample_sq_sum": float(np.square(values).sum()),
                    "robust_scale": float(score_data["scale"]),
                    "value_clip_c": value_clip,
                    "value_hat": float(value_hat[position]),
                    "finish_prob_hat": float(finish_prob[position]),
                    "finish_bootstrap_replicates": 0,
                    "finish_bootstrap_seed": None,
                    "lambda": effective_finish_weight,
                    "score_q_hat": float(score[position]),
                    "score_sample_n": int(len(values)),
                    "score_variance": float(score_variance[position]),
                    "support_width": 2.0 * value_clip,
                    "radius_b": float(radius[position]),
                    "lower_bound": float(lower[position]),
                    "upper_bound": float(upper[position]),
                    "drift_coordinate": "shared_anchor_prefix",
                    "remaining_time_s": max(
                        0.0,
                        deadline_s
                        - float(trace.step_before[idx, int(client_id)] * scout_steps),
                    ),
                    "drift_rho_per_s": 0.0,
                    "rank_at_checkpoint": int(ranks[position]),
                    "in_current_topk": bool(position in set(top_positions.tolist())),
                    "alive_candidate": True,
                    "gamma_margin": gamma,
                    "certificate_fired": certified,
                    "state_bytes": state_bytes,
                    "state_write_s": checkpoint_write_wall_s / m,
                    "state_read_s": 0.0,
                    "state_checksum": commit_checksums[position],
                    "score_record_hash": score_record_hash,
                    "protocol_version": PROTOCOL_VERSION,
                    "anchor_rank_agreement": agreement,
                    "adaptive_floor_fraction": adaptive_floor_fraction,
                    "selection_history_count_before": (
                        int(admitted_history_counts[position]) if history_enabled else None
                    ),
                    "anchor_conditional_target_s": (
                        float(anchor_targets[position]) if history_enabled else None
                    ),
                    "history_conditional_target_s": (
                        float(history_targets[position]) if history_enabled else None
                    ),
                    "history_score": (
                        float(history_scores[position]) if history_enabled else None
                    ),
                    "history_target_mix": history_mix if history_enabled else None,
                    "history_target_temperature": (
                        history_temperature if history_enabled else None
                    ),
                    "reusable_prefix_enabled": not no_reusable_prefix,
                    "finishability_score_enabled": not no_finishability,
                    "crossfit_fold_count": 1 if no_valid_crossfit else 2,
                    "selection_certificate_valid": certificate_valid,
                }
            )

    audit_start = time.perf_counter()
    audit_gpu_s = 0.0
    hidden_topk = topk.copy()
    interval_covered = True
    rank_error_bool: bool | None = None
    replay_io_bytes = 0
    full_final = trainer.stacked_global(m)
    with torch.no_grad():
        selected_index_gpu = torch.as_tensor(selected_positions, device=trainer.device)
        for name in full_final:
            full_final[name].index_copy_(0, selected_index_gpu, selected_params[name])
    if audit_replay:
        stopped = np.asarray(
            [value for value in admitted if int(value) not in set(selected.tolist())],
            dtype=np.int64,
        )
        stopped_positions = np.asarray(
            [position_map[int(value)] for value in stopped], dtype=np.int64
        )
        if len(stopped):
            if no_reusable_prefix:
                stopped_index_gpu = torch.as_tensor(
                    stopped_positions, device=trainer.device
                )
                stopped_params = {
                    name: value.index_select(0, stopped_index_gpu).clone()
                    for name, value in global_start.items()
                }
                stopped_start_step = 0
            else:
                stopped_index_cpu = torch.from_numpy(stopped_positions)
                stopped_params = {
                    name: value.index_select(0, stopped_index_cpu).to(trainer.device)
                    for name, value in checkpoint_cpu.items()
                }
                stopped_start_step = scout_steps
            stopped_states = [trace.data_state_id(round_number, int(value)) for value in stopped]
            for local_step in range(stopped_start_step, local_steps):
                step_result = trainer.r2c_train_step(
                    stopped_params,
                    stopped,
                    round_number,
                    stopped_states,
                    local_step,
                    learning_rate,
                )
                audit_gpu_s += step_result.gpu_s
            stopped_index_gpu = torch.as_tensor(stopped_positions, device=trainer.device)
            with torch.no_grad():
                for name in full_final:
                    full_final[name].index_copy_(0, stopped_index_gpu, stopped_params[name])
        hidden0, _, hidden_gpu0 = trainer.r2c_anchor_losses(
            full_final, 0, anchor_per_fold, eval_microbatch
        )
        if no_valid_crossfit:
            hidden1 = hidden0.copy()
            hidden_gpu1 = 0.0
        else:
            hidden1, _, hidden_gpu1 = trainer.r2c_anchor_losses(
                full_final, 1, anchor_per_fold, eval_microbatch
            )
        audit_gpu_s += hidden_gpu0 + hidden_gpu1
        hidden_data = _score_from_anchor(
            base0,
            base1,
            hidden0,
            hidden1,
            finish_prob,
            scale_floor,
            value_clip,
            effective_finish_weight,
            radius_multiplier,
        )
        hidden_scores = np.asarray(hidden_data["score"], dtype=np.float64)
        hidden_order = np.lexsort((admitted, -hidden_scores))
        hidden_topk = admitted[hidden_order[:SELECTED_K]]
        rank_error_bool = set(topk.tolist()) != set(hidden_topk.tolist())
        interval_covered = bool(
            np.all(hidden_scores >= lower) and np.all(hidden_scores <= upper)
        )
        replay_io_bytes = (
            0
            if no_reusable_prefix
            else int(
                sum(value.numel() * value.element_size() for value in checkpoint_cpu.values())
            )
        )
    audit_wall_s = time.perf_counter() - audit_start if audit_replay else 0.0

    full_norms = _parameter_norms(full_final, global_start)
    full_clipped_norms = np.minimum(full_norms, delta_clip)
    utility_full = np.full(NUM_CLIENTS, np.nan, dtype=np.float64)
    admission_full = np.zeros(NUM_CLIENTS, dtype=np.float64)
    target_full = np.full(NUM_CLIENTS, np.nan, dtype=np.float64)
    inclusion_full = np.full(NUM_CLIENTS, np.nan, dtype=np.float64)
    finish_full = np.full(NUM_CLIENTS, np.nan, dtype=np.float64)
    effective_full = np.full(NUM_CLIENTS, np.nan, dtype=np.float64)
    hajek_raw_full = np.full(NUM_CLIENTS, np.nan, dtype=np.float64)
    hajek_norm_full = np.full(NUM_CLIENTS, np.nan, dtype=np.float64)
    delta_raw_full = np.full(NUM_CLIENTS, np.nan, dtype=np.float64)
    delta_clipped_full = np.full(NUM_CLIENTS, np.nan, dtype=np.float64)
    loss_before_full = np.full(NUM_CLIENTS, np.nan, dtype=np.float64)
    loss_after_full = np.full(NUM_CLIENTS, np.nan, dtype=np.float64)
    checkpoint_checksum: list[str | None] = [None] * NUM_CLIENTS
    upload_checksum: list[str | None] = [None] * NUM_CLIENTS
    for position, client_id in enumerate(admitted):
        client_id = int(client_id)
        utility_full[client_id] = float(score[position])
        admission_full[client_id] = admission_probability
        target_full[client_id] = float(targets[position])
        inclusion_full[client_id] = admission_probability * targets[position]
        finish_full[client_id] = float(finish_prob[position])
        effective_full[client_id] = inclusion_full[client_id] * finish_full[client_id]
        delta_raw_full[client_id] = float(full_norms[position])
        delta_clipped_full[client_id] = float(full_clipped_norms[position])
        checkpoint_checksum[client_id] = commit_checksums[position]
    selected_checksums = _sampled_state_checksums(
        {name: value.detach().cpu() for name, value in selected_params.items()}, selected
    )
    for position, client_id in enumerate(selected):
        client_id = int(client_id)
        hajek_raw_full[client_id] = float(raw_weights[position])
        hajek_norm_full[client_id] = float(normalized_weights[position])
        loss_before_full[client_id] = float(first_local_loss[selected_positions[position]])
        loss_after_full[client_id] = float(selected_latest_loss[position])
        upload_checksum[client_id] = selected_checksums[position]

    step_before = trace.step_before[idx, admitted]
    step_after = trace.step_after[idx, admitted]
    switch_step = max(1, int(math.ceil(0.4 * local_steps)))
    full_step_s = switch_step * step_before + (local_steps - switch_step) * step_after
    prefix_step_s = np.minimum(scout_steps, switch_step) * step_before + max(
        0, scout_steps - switch_step
    ) * step_after
    candidate_full_s = float(full_step_s.sum())
    used_s = float(prefix_step_s.sum())
    for client_id in selected:
        position = position_map[int(client_id)]
        used_s += float(
            full_step_s[position]
            if no_reusable_prefix
            else full_step_s[position] - prefix_step_s[position]
        )
    saved_s = max(0.0, candidate_full_s - used_s)
    discarded_prefix_s = (
        float(prefix_step_s[selected_positions].sum()) if no_reusable_prefix else 0.0
    )
    eligible_set_for_compute = set(int(value) for value in eligible)
    late_positions = np.asarray(
        [position_map[int(value)] for value in selected if int(value) not in eligible_set_for_compute],
        dtype=np.int64,
    )
    late_noncontributing_s = (
        0.0
        if len(late_positions) == 0
        else float(
            (
                full_step_s[late_positions]
                if no_reusable_prefix
                else full_step_s[late_positions] - prefix_step_s[late_positions]
            ).sum()
        )
    )
    wasted_s = discarded_prefix_s + late_noncontributing_s
    effective_sample_size = (
        float(1.0 / np.square(normalized_weights).sum())
        if np.square(normalized_weights).sum() > 0
        else 0.0
    )
    max_inverse = (
        float(np.max(1.0 / np.maximum(effective_selected[raw_weights > 0], 1.0e-12)))
        if np.any(raw_weights > 0)
        else None
    )
    certificate_payload = {
        "run_id": run_id,
        "round": round_number,
        "dataset_id": trace.dataset_id,
        "scenario_id": trace.scenario_id,
        "severity": "base",
        "protocol_version": PROTOCOL_VERSION,
        "round_start_model_hash": round_start_model_hash,
        "admission_seed": admission_seed,
        "admitted_order_hash": sha256_text(",".join(map(str, admitted.tolist()))),
        "pivotal_seed": pivotal_seed,
        "delta": None,
        "certified": certified,
        "fallback_reason": (
            None
            if certified
            else (
                "invalid_single_fold_ablation"
                if no_valid_crossfit
                else "budget_limited_anchor_commit"
            )
        ),
        "commit_j": 1,
        "commit_steps": scout_steps,
        "commit_fraction": scout_steps / local_steps,
        "gamma_at_commit": gamma,
        "vacuous_margin": len(outsider_positions) == 0,
        "committed_topk_hash": sha256_text(",".join(map(str, sorted(topk.tolist())))),
        "selected_set_hash": sha256_text(",".join(map(str, selected.tolist()))),
        "on_time_subset_hash": sha256_text(",".join(map(str, eligible.tolist()))),
        "aggregation_subset_n": len(eligible),
        "no_op_round": len(eligible) == 0,
        "hidden_final_topk_hash": sha256_text(",".join(map(str, sorted(hidden_topk.tolist())))),
        "topk_intersection_size": len(set(topk.tolist()) & set(hidden_topk.tolist())),
        "rank_error": rank_error_bool if certified and audit_replay else None,
        "interval_covered_all": interval_covered if audit_replay else None,
        "drift_envelope_held": interval_covered if audit_replay else None,
        "candidate_compute_full_s": candidate_full_s,
        "candidate_compute_used_s": used_s,
        "candidate_compute_saved_s": saved_s,
        "candidate_compute_saved_fraction": saved_s / candidate_full_s if candidate_full_s else 0.0,
        "candidate_compute_wasted_s": wasted_s,
        "candidate_compute_wasted_fraction": wasted_s / used_s if used_s else 0.0,
        "discarded_restarted_prefix_compute_s": discarded_prefix_s,
        "late_noncontributing_compute_s": late_noncontributing_s,
        "wasted_compute_definition": "discarded_restarted_prefix_plus_late_noncontributing_local_work_over_used_candidate_compute",
        "replay_seed": int(
            np.random.SeedSequence([seed, round_number, 0xA0D17]).generate_state(
                1, dtype=np.uint32
            )[0]
        ),
        "replay_gpu_s": audit_gpu_s,
        "replay_cpu_s": max(0.0, audit_wall_s - audit_gpu_s),
        "replay_io_bytes": replay_io_bytes,
        "max_effective_inverse_weight": max_inverse,
        "hajek_effective_sample_size": effective_sample_size,
        "anchor_rank_agreement": agreement,
        "adaptive_floor_fraction": adaptive_floor_fraction,
        "aggregation_weight_rule": "capped_power_stabilized",
        "propensity_power": propensity_power,
        "completion_power": completion_power,
        "weight_cap": weight_cap,
        "audit_replay_enabled": audit_replay,
        "checkpoint_serialization_version": CHECKPOINT_SERIALIZATION_VERSION,
        "ablation_variant": str(config.get("r2c_ablation_variant", "full")),
        "reusable_prefix_enabled": not no_reusable_prefix,
        "finishability_score_enabled": not no_finishability,
        "configured_finishability_weight": finish_weight,
        "effective_finishability_weight": effective_finish_weight,
        "crossfit_fold_count": 1 if no_valid_crossfit else 2,
        "selection_certificate_valid": certificate_valid,
        "raw_certificate_predicate": raw_certified,
    }
    if history_enabled:
        assert selection_history_counts is not None
        certificate_payload.update(
            {
                "history_target_rule": "negative_log1p_cumulative_selection_count",
                "history_target_mix": history_mix,
                "history_target_temperature": history_temperature,
                "history_state_before_round": True,
                "history_counts_hash": sha256_text(
                    canonical_json(selection_history_counts.astype(int).tolist())
                ),
                "history_target_hash": sha256_text(
                    canonical_json(
                        {
                            "admitted_client_ids": admitted.tolist(),
                            "selection_history_counts": admitted_history_counts.tolist(),
                            "anchor_targets": anchor_targets.tolist(),
                            "history_targets": history_targets.tolist(),
                            "final_targets": targets.tolist(),
                        }
                    )
                ),
                "history_target_l1_from_anchor": float(
                    np.abs(targets - anchor_targets).sum()
                ),
                "history_count_min_admitted": int(admitted_history_counts.min()),
                "history_count_max_admitted": int(admitted_history_counts.max()),
            }
        )
    certificate_payload["certificate_record_hash"] = sha256_text(
        canonical_json(certificate_payload)
    )

    return R2CRoundResult(
        admitted=admitted,
        selected=selected,
        eligible=eligible,
        topk=topk,
        prefix_steps=scout_steps,
        certified=certified,
        fallback_reason=certificate_payload["fallback_reason"],
        gamma=gamma,
        utility_scores=utility_full,
        admission_prob=admission_full,
        conditional_targets=target_full,
        inclusion_prob=inclusion_full,
        finish_prob=finish_full,
        effective_probability=effective_full,
        hajek_weight_raw=hajek_raw_full,
        hajek_weight_normalized=hajek_norm_full,
        delta_norm_raw=delta_raw_full,
        delta_norm_clipped=delta_clipped_full,
        local_loss_before=loss_before_full,
        local_loss_after=loss_after_full,
        checkpoint_checksum=checkpoint_checksum,
        upload_checksum=upload_checksum,
        algorithm_train_wall_s=algorithm_wall_s,
        algorithm_gpu_s=algorithm_gpu_s,
        checkpoint_write_wall_s=checkpoint_write_wall_s,
        checkpoint_read_wall_s=checkpoint_read_wall_s,
        aggregate_wall_s=aggregate_wall_s,
        audit_wall_s=audit_wall_s,
        audit_gpu_s=audit_gpu_s,
        telemetry_upload_bytes=telemetry_upload_bytes,
        checkpoint_rows=checkpoint_rows,
        certificate_row=certificate_payload,
        no_op=len(eligible) == 0,
    )
