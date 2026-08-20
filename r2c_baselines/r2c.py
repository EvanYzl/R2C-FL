from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .config import CANDIDATE_M, NUM_CLIENTS, SELECTED_K
from .data import FederatedData
from .traces import Trace
from .training import LocalTrainer
from .utils import canonical_json, sha256_text


TELEMETRY_BYTES_PER_CHECKPOINT = 512
CHECKPOINT_SERIALIZATION_VERSION = "r2c-cpu-ring-v1"


def _rng(seed: int, round_number: int, stream: int, *parts: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), int(round_number), int(stream), *[int(value) for value in parts], 0x523243]
        )
    )


def project_capped_simplex(
    values: np.ndarray,
    total: float,
    lower: float | np.ndarray,
    upper: float = 1.0,
    tolerance: float = 1e-12,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lower_values = np.broadcast_to(np.asarray(lower, dtype=np.float64), values.shape)
    if not np.isfinite(values).all() or not np.isfinite(lower_values).all():
        raise ValueError("Capped-simplex inputs must be finite")
    if np.any(lower_values < 0) or np.any(lower_values > upper):
        raise ValueError("Invalid capped-simplex lower bound")
    if lower_values.sum() - tolerance > total or upper * len(values) + tolerance < total:
        raise ValueError("Infeasible capped-simplex total")
    lo = float(np.min(values - upper)) - 1.0
    hi = float(np.max(values - lower_values)) + 1.0
    for _ in range(200):
        midpoint = 0.5 * (lo + hi)
        projected = np.clip(values - midpoint, lower_values, upper)
        if projected.sum() > total:
            lo = midpoint
        else:
            hi = midpoint
    projected = np.clip(values - 0.5 * (lo + hi), lower_values, upper)
    residual = float(total - projected.sum())
    if abs(residual) > tolerance:
        if residual > 0:
            room = upper - projected
        else:
            room = projected - lower_values
        order = np.argsort(-room, kind="stable")
        for index in order:
            amount = min(abs(residual), float(room[index]))
            projected[index] += math.copysign(amount, residual)
            residual = float(total - projected.sum())
            if abs(residual) <= tolerance:
                break
    if abs(float(projected.sum()) - total) > tolerance:
        raise RuntimeError("Capped-simplex projection failed to reach requested total")
    return projected


def conditional_inclusion_targets(
    scores: np.ndarray, k: int, temperature: float, floor: float
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = (scores - float(np.max(scores))) / float(temperature)
    weights = np.exp(np.clip(logits, -50.0, 50.0))
    weights /= weights.sum()
    initial = np.full(len(scores), floor, dtype=np.float64)
    initial += (float(k) - floor * len(scores)) * weights
    return project_capped_simplex(initial, float(k), floor, 1.0)


def history_balanced_conditional_targets(
    anchor_scores: np.ndarray,
    selection_history_counts: np.ndarray,
    k: int,
    anchor_temperature: float,
    history_temperature: float,
    floor: float,
    history_mix: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mix utility and label-free exposure targets without changing exact-K mass."""

    anchor_scores = np.asarray(anchor_scores, dtype=np.float64)
    history_counts = np.asarray(selection_history_counts, dtype=np.float64)
    if anchor_scores.ndim != 1 or history_counts.shape != anchor_scores.shape:
        raise ValueError("Anchor scores and selection-history counts must be matching vectors")
    if not np.isfinite(anchor_scores).all() or not np.isfinite(history_counts).all():
        raise ValueError("Anchor scores and selection-history counts must be finite")
    if np.any(history_counts < 0.0):
        raise ValueError("Selection-history counts must be nonnegative")
    if not 0.0 <= float(history_mix) <= 1.0:
        raise ValueError("History-target mixture must lie in [0, 1]")
    if float(history_temperature) <= 0.0:
        raise ValueError("History-target temperature must be positive")

    anchor_targets = conditional_inclusion_targets(
        anchor_scores, int(k), float(anchor_temperature), float(floor)
    )
    history_scores = -np.log1p(history_counts)
    history_targets = conditional_inclusion_targets(
        history_scores, int(k), float(history_temperature), float(floor)
    )
    final_targets = (
        (1.0 - float(history_mix)) * anchor_targets
        + float(history_mix) * history_targets
    )
    if abs(float(final_targets.sum()) - float(k)) > 1.0e-10:
        raise RuntimeError("History-balanced conditional targets do not sum to K")
    if np.any(final_targets < float(floor) - 1.0e-12) or np.any(final_targets > 1.0 + 1.0e-12):
        raise RuntimeError("History-balanced conditional targets violate pivotal bounds")
    return anchor_targets, history_targets, final_targets, history_scores


def pivotal_sample(targets: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Deville--Tille two-coordinate pivotal sampling."""

    probabilities = np.asarray(targets, dtype=np.float64).copy()
    if abs(float(probabilities.sum()) - float(k)) > 1e-10:
        raise ValueError("Pivotal targets must sum to k")
    tolerance = 1e-12
    while True:
        fractional = np.flatnonzero((probabilities > tolerance) & (probabilities < 1.0 - tolerance))
        if len(fractional) < 2:
            break
        first, second = int(fractional[0]), int(fractional[1])
        a, b = float(probabilities[first]), float(probabilities[second])
        total = a + b
        draw = float(rng.random())
        if total < 1.0:
            if draw < a / max(total, tolerance):
                probabilities[first], probabilities[second] = total, 0.0
            else:
                probabilities[first], probabilities[second] = 0.0, total
        else:
            denominator = max(tolerance, 2.0 - total)
            probability_first_one = (1.0 - b) / denominator
            if draw < probability_first_one:
                probabilities[first], probabilities[second] = 1.0, total - 1.0
            else:
                probabilities[first], probabilities[second] = total - 1.0, 1.0
        probabilities = np.clip(probabilities, 0.0, 1.0)
    selected = np.flatnonzero(probabilities >= 1.0 - 1e-10)
    if len(selected) != int(k):
        order = np.lexsort((np.arange(len(probabilities)), -probabilities))
        selected = np.sort(order[: int(k)])
    if len(np.unique(selected)) != int(k):
        raise RuntimeError("Pivotal sampling did not return exactly k unique entries")
    return np.asarray(selected, dtype=np.int64)


def _robust_scale(values: np.ndarray, floor: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(float(floor), 1.4826 * mad)


def _finish_bootstrap(
    observed_steps: np.ndarray,
    observed_bandwidth: np.ndarray,
    remaining_steps: int,
    prefix_elapsed_s: float,
    payload_bytes: int,
    deadline_s: float,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    observed_steps = np.asarray(observed_steps, dtype=np.float64)
    observed_bandwidth = np.asarray(observed_bandwidth, dtype=np.float64)
    if remaining_steps > 0:
        step_draws = rng.choice(observed_steps, size=(replicates, int(remaining_steps)), replace=True)
        future_step_s = step_draws.sum(axis=1)
    else:
        future_step_s = np.zeros(replicates, dtype=np.float64)
    bandwidth_draws = rng.choice(observed_bandwidth, size=replicates, replace=True)
    upload_s = float(payload_bytes) * 8.0 / np.maximum(1.0, bandwidth_draws)
    total_s = float(prefix_elapsed_s) + future_step_s + upload_s
    return total_s <= float(deadline_s)


def _sampled_state_checksums(
    state: dict[str, torch.Tensor], client_ids: np.ndarray, samples_per_tensor: int = 16
) -> list[str]:
    """Fast deterministic content checksums for a CPU checkpoint ring.

    Exact bitwise restore is separately enforced by the checkpoint gate.  The
    per-checkpoint ledger uses evenly spaced tensor samples to keep the formal
    D4 audit tractable while still detecting state swaps or corruption.
    """

    digests = [hashlib.sha256() for _ in client_ids]
    for name in sorted(state):
        array = state[name].detach().cpu().contiguous().numpy()
        for position, digest in enumerate(digests):
            flat = array[position].reshape(-1)
            count = min(int(samples_per_tensor), len(flat))
            indices = np.linspace(0, len(flat) - 1, num=count, dtype=np.int64)
            digest.update(name.encode("utf-8"))
            digest.update(np.asarray(flat[indices]).tobytes())
    return [digest.hexdigest() for digest in digests]


def _parameter_norms(
    final_params: dict[str, torch.Tensor], start_params: dict[str, torch.Tensor]
) -> np.ndarray:
    count = next(iter(final_params.values())).shape[0]
    norms_sq = torch.zeros(count, device=next(iter(final_params.values())).device, dtype=torch.float64)
    for name in final_params:
        delta = (final_params[name] - start_params[name]).reshape(count, -1).double()
        norms_sq += torch.square(delta).sum(dim=1)
    return torch.sqrt(norms_sq).detach().cpu().numpy()


@dataclass
class R2CRoundResult:
    admitted: np.ndarray
    selected: np.ndarray
    eligible: np.ndarray
    topk: np.ndarray
    prefix_steps: int
    certified: bool
    fallback_reason: str | None
    gamma: float | None
    utility_scores: np.ndarray
    admission_prob: np.ndarray
    conditional_targets: np.ndarray
    inclusion_prob: np.ndarray
    finish_prob: np.ndarray
    effective_probability: np.ndarray
    hajek_weight_raw: np.ndarray
    hajek_weight_normalized: np.ndarray
    delta_norm_raw: np.ndarray
    delta_norm_clipped: np.ndarray
    local_loss_before: np.ndarray
    local_loss_after: np.ndarray
    checkpoint_checksum: list[str | None]
    upload_checksum: list[str | None]
    algorithm_train_wall_s: float
    algorithm_gpu_s: float
    checkpoint_write_wall_s: float
    checkpoint_read_wall_s: float
    aggregate_wall_s: float
    audit_wall_s: float
    audit_gpu_s: float
    telemetry_upload_bytes: int
    checkpoint_rows: list[dict[str, Any]]
    certificate_row: dict[str, Any]
    no_op: bool


def run_r2c_round(
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
) -> R2CRoundResult:
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
    checkpoint_steps = list(range(1, local_steps + 1))
    checkpoint_count = len(checkpoint_steps)
    bootstrap_replicates = int(config["r2c_bootstrap_replicates"])
    heldout_limit = int(config["r2c_heldout_per_fold"])
    eval_microbatch = int(config.get("r2c_eval_microbatch", trainer.max_parallel_clients))
    value_clip = float(config["r2c_value_clip"])
    scale_floor = float(config["r2c_scale_floor"])
    penalty_lambda = float(config["r2c_lambda"])
    delta = float(config["r2c_delta"])
    temperature = float(config["r2c_temperature"])
    completion_floor = float(config["r2c_completion_floor"])
    delta_clip = float(config.get("r2c_delta_clip", math.inf))
    if delta_clip <= 0:
        raise ValueError("r2c_delta_clip must be positive")
    idx = round_number - 1
    deadline_s = float(trace.deadline[idx])
    switch_step = max(1, int(math.ceil(0.4 * local_steps)))
    step_matrix = np.empty((m, local_steps), dtype=np.float64)
    bandwidth_matrix = np.empty((m, local_steps), dtype=np.float64)
    for local_step in range(local_steps):
        after = local_step >= switch_step
        step_matrix[:, local_step] = (
            trace.step_after[idx, admitted] if after else trace.step_before[idx, admitted]
        )
        bandwidth_matrix[:, local_step] = (
            trace.bandwidth_after[idx, admitted] if after else trace.bandwidth_before[idx, admitted]
        )

    params = trainer.stacked_global(m)
    global_start = trainer.stacked_global(m)
    algorithm_wall_s = 0.0
    algorithm_gpu_s = 0.0
    checkpoint_write_wall_s = 0.0
    checkpoint_read_wall_s = 0.0
    checkpoint_rows: list[dict[str, Any]] = []

    base_fold0, wall, gpu = trainer.r2c_heldout_losses(
        params, admitted, data_states, 0, heldout_limit, eval_microbatch
    )
    algorithm_wall_s += wall
    algorithm_gpu_s += gpu
    base_fold1, wall, gpu = trainer.r2c_heldout_losses(
        params, admitted, data_states, 1, heldout_limit, eval_microbatch
    )
    algorithm_wall_s += wall
    algorithm_gpu_s += gpu

    previous_scores: np.ndarray | None = None
    previous_batch_elapsed = 0.0
    drift_rho = float(config["r2c_initial_drift_per_s"])
    commit_payload: dict[str, Any] | None = None
    checkpoint_cpu: dict[str, torch.Tensor] | None = None
    commit_checksums: list[str] = []
    telemetry_upload_bytes = 0
    latest_fold_losses = (base_fold0, base_fold1)

    for checkpoint_j, checkpoint_step in enumerate(checkpoint_steps, start=1):
        step_result = trainer.r2c_train_step(
            params,
            admitted,
            round_number,
            data_states,
            checkpoint_step - 1,
            learning_rate,
        )
        algorithm_wall_s += step_result.wall_s
        algorithm_gpu_s += step_result.gpu_s

        write_start = time.perf_counter()
        checkpoint_cpu = {
            name: value.detach().cpu().clone() for name, value in params.items()
        }
        state_write_s = time.perf_counter() - write_start
        checkpoint_write_wall_s += state_write_s
        algorithm_wall_s += state_write_s
        commit_checksums = _sampled_state_checksums(checkpoint_cpu, admitted)

        current_fold0, wall0, gpu0 = trainer.r2c_heldout_losses(
            params, admitted, data_states, 0, heldout_limit, eval_microbatch
        )
        current_fold1, wall1, gpu1 = trainer.r2c_heldout_losses(
            params, admitted, data_states, 1, heldout_limit, eval_microbatch
        )
        algorithm_wall_s += wall0 + wall1
        algorithm_gpu_s += gpu0 + gpu1
        latest_fold_losses = (current_fold0, current_fold1)

        improvements = (base_fold0 - current_fold0, base_fold1 - current_fold1)
        report_fold = (checkpoint_j - 1) % 2
        scale_fold = 1 - report_fold
        prefix_elapsed = step_matrix[:, :checkpoint_step].sum(axis=1)
        telemetry_s = (
            TELEMETRY_BYTES_PER_CHECKPOINT
            * 8.0
            / np.maximum(1.0, bandwidth_matrix[:, checkpoint_step - 1])
        )
        receive_elapsed = prefix_elapsed + telemetry_s
        checkpoint_batch_elapsed = float(np.max(receive_elapsed))
        remaining_time = max(0.0, deadline_s - checkpoint_batch_elapsed)
        telemetry_upload_bytes += m * TELEMETRY_BYTES_PER_CHECKPOINT

        value_hat = np.empty(m, dtype=np.float64)
        finish_prob = np.empty(m, dtype=np.float64)
        score_hat = np.empty(m, dtype=np.float64)
        score_variance = np.empty(m, dtype=np.float64)
        score_n = np.full(m, bootstrap_replicates, dtype=np.int64)
        radius = np.empty(m, dtype=np.float64)
        robust_scales = np.empty(m, dtype=np.float64)
        value_vectors: list[np.ndarray] = []
        finish_seeds: list[int] = []
        log_factor = math.log(4.0 * m * checkpoint_count / delta)
        support_width = 2.0
        for position, client_id in enumerate(admitted):
            report_raw = np.clip(improvements[report_fold][position], -value_clip, value_clip)
            scale_raw = np.clip(improvements[scale_fold][position], -value_clip, value_clip)
            robust_scale = _robust_scale(scale_raw, scale_floor)
            normalized_values = np.clip(report_raw / robust_scale, -1.0, 1.0)
            robust_scales[position] = robust_scale
            value_vectors.append(normalized_values)
            value_hat[position] = float(np.mean(normalized_values))
            bootstrap_seed = int(
                np.random.SeedSequence(
                    [seed, round_number, checkpoint_j, int(client_id), 0xF1A15]
                ).generate_state(1, dtype=np.uint32)[0]
            )
            finish_seeds.append(bootstrap_seed)
            finish_rng = np.random.default_rng(bootstrap_seed)
            completion = _finish_bootstrap(
                step_matrix[position, :checkpoint_step],
                bandwidth_matrix[position, :checkpoint_step],
                local_steps - checkpoint_step,
                float(receive_elapsed[position]),
                payload_bytes,
                deadline_s,
                bootstrap_replicates,
                finish_rng,
            )
            finish_prob[position] = float(np.mean(completion))
            score_hat[position] = (
                value_hat[position] * finish_prob[position]
                - penalty_lambda * (1.0 - finish_prob[position])
            )
            pair_rng = _rng(seed, round_number, 151, checkpoint_j, int(client_id))
            sampled_values = normalized_values[
                pair_rng.integers(0, len(normalized_values), size=bootstrap_replicates)
            ]
            paired = sampled_values * completion.astype(np.float64) - penalty_lambda * (
                1.0 - completion.astype(np.float64)
            )
            score_variance[position] = float(np.var(paired, ddof=1))

        if previous_scores is not None:
            elapsed_delta = max(1e-12, checkpoint_batch_elapsed - previous_batch_elapsed)
            observed_slope = float(np.max(np.abs(score_hat - previous_scores)) / elapsed_delta)
            drift_rho = max(drift_rho, observed_slope)
        for position in range(m):
            radius[position] = (
                math.sqrt(
                    2.0 * score_variance[position] * log_factor / score_n[position]
                )
                + 3.0 * support_width * log_factor / score_n[position]
                + drift_rho * remaining_time
            )
        lower = score_hat - radius
        upper = score_hat + radius
        order = np.lexsort((admitted, -score_hat))
        top_positions = order[:SELECTED_K]
        topk = admitted[top_positions]
        outsider_positions = order[SELECTED_K:]
        vacuous = len(outsider_positions) == 0
        gamma = None if vacuous else float(np.min(lower[top_positions]) - np.max(upper[outsider_positions]))
        certified = bool(vacuous or (gamma is not None and gamma > 0.0))
        floor = 0.05 * SELECTED_K / m
        targets = conditional_inclusion_targets(score_hat, SELECTED_K, temperature, floor)
        ranks = np.empty(m, dtype=np.int64)
        ranks[order] = np.arange(1, m + 1, dtype=np.int64)
        score_record_hash = sha256_text(
            canonical_json(
                {
                    "round": round_number,
                    "checkpoint_j": checkpoint_j,
                    "round_start_model_hash": round_start_model_hash,
                    "admission_seed": admission_seed,
                    "client_ids": admitted.tolist(),
                    "scores": score_hat.tolist(),
                    "radii": radius.tolist(),
                    "targets": targets.tolist(),
                }
            )
        )
        if full_logging:
            state_bytes = int(
                sum(value[0].numel() * value[0].element_size() for value in checkpoint_cpu.values())
            )
            for position, client_id in enumerate(admitted):
                report_values = value_vectors[position]
                checkpoint_rows.append(
                    {
                        "run_id": run_id,
                        "round": round_number,
                        "client_id": int(client_id),
                        "round_start_model_hash": round_start_model_hash,
                        "admission_seed": admission_seed,
                        "checkpoint_j": checkpoint_j,
                        "tau_steps": checkpoint_step,
                        "tau_fraction": checkpoint_step / local_steps,
                        "checkpoint_elapsed_s": float(receive_elapsed[position]),
                        "checkpoint_batch_elapsed_s": checkpoint_batch_elapsed,
                        "report_fold": report_fold,
                        "scale_fold": scale_fold,
                        "value_sample_n": int(len(report_values)),
                        "value_sample_sum": float(report_values.sum()),
                        "value_sample_sq_sum": float(np.square(report_values).sum()),
                        "robust_scale": float(robust_scales[position]),
                        "value_clip_c": value_clip,
                        "value_hat": float(value_hat[position]),
                        "finish_prob_hat": float(finish_prob[position]),
                        "finish_bootstrap_replicates": bootstrap_replicates,
                        "finish_bootstrap_seed": finish_seeds[position],
                        "lambda": penalty_lambda,
                        "score_q_hat": float(score_hat[position]),
                        "score_sample_n": int(score_n[position]),
                        "score_variance": float(score_variance[position]),
                        "support_width": support_width,
                        "radius_b": float(radius[position]),
                        "lower_bound": float(lower[position]),
                        "upper_bound": float(upper[position]),
                        "drift_coordinate": "server_elapsed_seconds",
                        "remaining_time_s": remaining_time,
                        "drift_rho_per_s": drift_rho,
                        "rank_at_checkpoint": int(ranks[position]),
                        "in_current_topk": bool(position in set(top_positions.tolist())),
                        "alive_candidate": True,
                        "gamma_margin": gamma,
                        "certificate_fired": certified,
                        "state_bytes": state_bytes,
                        "state_write_s": state_write_s / m,
                        "state_read_s": 0.0,
                        "state_checksum": commit_checksums[position],
                        "score_record_hash": score_record_hash,
                    }
                )
        commit_payload = {
            "checkpoint_j": checkpoint_j,
            "checkpoint_step": checkpoint_step,
            "scores": score_hat.copy(),
            "finish_prob": finish_prob.copy(),
            "radius": radius.copy(),
            "lower": lower.copy(),
            "upper": upper.copy(),
            "targets": targets.copy(),
            "topk": topk.copy(),
            "gamma": gamma,
            "vacuous": vacuous,
            "certified": certified,
            "score_record_hash": score_record_hash,
            "drift_rho": drift_rho,
            "checkpoint_batch_elapsed": checkpoint_batch_elapsed,
        }
        previous_scores = score_hat.copy()
        previous_batch_elapsed = checkpoint_batch_elapsed
        if certified:
            break

    assert commit_payload is not None and checkpoint_cpu is not None
    if not commit_payload["certified"]:
        commit_payload["fallback_reason"] = "no_positive_margin"
    else:
        commit_payload["fallback_reason"] = None

    pivotal_seed = int(
        np.random.SeedSequence(
            [
                seed,
                round_number,
                int(commit_payload["checkpoint_j"]),
                int(commit_payload["score_record_hash"][:8], 16),
                0xD37111E,
            ]
        ).generate_state(1, dtype=np.uint32)[0]
    )
    selected_positions = pivotal_sample(
        commit_payload["targets"], SELECTED_K, np.random.default_rng(pivotal_seed)
    )
    selected = np.sort(admitted[selected_positions])
    position_map = {int(client_id): position for position, client_id in enumerate(admitted)}
    selected_positions = np.asarray([position_map[int(client_id)] for client_id in selected], dtype=np.int64)

    read_start = time.perf_counter()
    selected_index_cpu = torch.from_numpy(selected_positions)
    selected_params = {
        name: value.index_select(0, selected_index_cpu).to(trainer.device)
        for name, value in checkpoint_cpu.items()
    }
    checkpoint_read_wall_s = time.perf_counter() - read_start
    algorithm_wall_s += checkpoint_read_wall_s
    selected_states = [trace.data_state_id(round_number, int(client_id)) for client_id in selected]
    for local_step in range(int(commit_payload["checkpoint_step"]), local_steps):
        step_result = trainer.r2c_train_step(
            selected_params,
            selected,
            round_number,
            selected_states,
            local_step,
            learning_rate,
        )
        algorithm_wall_s += step_result.wall_s
        algorithm_gpu_s += step_result.gpu_s

    selected_start = {
        name: value.index_select(0, torch.as_tensor(selected_positions, device=value.device))
        for name, value in global_start.items()
    }
    raw_norms = _parameter_norms(selected_params, selected_start)
    clipped_norms = np.minimum(raw_norms, delta_clip)
    clip_scales = np.minimum(1.0, delta_clip / np.maximum(raw_norms, 1e-12))

    _, _, selected_total = trace.simulated_times(
        round_number, selected, local_steps, payload_bytes
    )
    eligible = selected[selected_total <= deadline_s]
    eligible_set = set(int(value) for value in eligible)
    admitted_probability = admission_probability
    finish_at_commit = np.asarray(
        [commit_payload["finish_prob"][position_map[int(client_id)]] for client_id in selected],
        dtype=np.float64,
    )
    targets_selected = np.asarray(
        [commit_payload["targets"][position_map[int(client_id)]] for client_id in selected],
        dtype=np.float64,
    )
    inclusion_selected = admitted_probability * targets_selected
    effective_selected = inclusion_selected * np.maximum(finish_at_commit, completion_floor)
    floor = 0.05 * SELECTED_K / m
    pi_min = admitted_probability * floor
    raw_weights = np.zeros(len(selected), dtype=np.float64)
    for position, client_id in enumerate(selected):
        if int(client_id) in eligible_set:
            denominator = max(
                float(effective_selected[position]), pi_min * completion_floor
            )
            raw_weights[position] = float(sample_counts[int(client_id)]) / denominator
    normalized_weights = (
        raw_weights / raw_weights.sum() if raw_weights.sum() > 0 else raw_weights.copy()
    )

    aggregate_start = time.perf_counter()
    with torch.no_grad():
        named = dict(model.named_parameters())
        for name, global_parameter in named.items():
            delta_tensor = selected_params[name] - selected_start[name]
            scale = torch.as_tensor(
                clip_scales, device=delta_tensor.device, dtype=delta_tensor.dtype
            ).view(len(selected), *([1] * global_parameter.ndim))
            weight = torch.as_tensor(
                normalized_weights, device=delta_tensor.device, dtype=delta_tensor.dtype
            ).view(len(selected), *([1] * global_parameter.ndim))
            global_parameter.add_((delta_tensor * scale * weight).sum(dim=0))
    aggregate_wall_s = time.perf_counter() - aggregate_start

    selected_fold0, wall0, gpu0 = trainer.r2c_heldout_losses(
        selected_params, selected, selected_states, 0, heldout_limit, eval_microbatch
    )
    selected_fold1, wall1, gpu1 = trainer.r2c_heldout_losses(
        selected_params, selected, selected_states, 1, heldout_limit, eval_microbatch
    )
    algorithm_wall_s += wall0 + wall1
    algorithm_gpu_s += gpu0 + gpu1
    start_loss_map = {
        int(client_id): float(
            0.5
            * (
                base_fold0[position_map[int(client_id)]].mean()
                + base_fold1[position_map[int(client_id)]].mean()
            )
        )
        for client_id in selected
    }
    end_loss_map = {
        int(client_id): float(0.5 * (selected_fold0[position].mean() + selected_fold1[position].mean()))
        for position, client_id in enumerate(selected)
    }

    # Analysis-only hidden final-order replay.  It never changes selection or
    # the model update and its time is excluded from algorithm_elapsed_s.
    audit_start = time.perf_counter()
    audit_gpu_s = 0.0
    stopped = np.asarray([value for value in admitted if int(value) not in set(selected.tolist())], dtype=np.int64)
    stopped_positions = np.asarray([position_map[int(value)] for value in stopped], dtype=np.int64)
    if len(stopped):
        stopped_index_cpu = torch.from_numpy(stopped_positions)
        stopped_params = {
            name: value.index_select(0, stopped_index_cpu).to(trainer.device)
            for name, value in checkpoint_cpu.items()
        }
        stopped_states = [trace.data_state_id(round_number, int(value)) for value in stopped]
        for local_step in range(int(commit_payload["checkpoint_step"]), local_steps):
            step_result = trainer.r2c_train_step(
                stopped_params,
                stopped,
                round_number,
                stopped_states,
                local_step,
                learning_rate,
            )
            audit_gpu_s += step_result.gpu_s
    else:
        stopped_params = {}

    full_final = trainer.stacked_global(m)
    with torch.no_grad():
        for name in full_final:
            selected_index_gpu = torch.as_tensor(selected_positions, device=trainer.device)
            full_final[name].index_copy_(0, selected_index_gpu, selected_params[name])
            if len(stopped):
                stopped_index_gpu = torch.as_tensor(stopped_positions, device=trainer.device)
                full_final[name].index_copy_(0, stopped_index_gpu, stopped_params[name])
    hidden_fold0, _, hidden_gpu0 = trainer.r2c_heldout_losses(
        full_final, admitted, data_states, 0, heldout_limit, eval_microbatch
    )
    hidden_fold1, _, hidden_gpu1 = trainer.r2c_heldout_losses(
        full_final, admitted, data_states, 1, heldout_limit, eval_microbatch
    )
    audit_gpu_s += hidden_gpu0 + hidden_gpu1
    final_improvements = (base_fold0 - hidden_fold0, base_fold1 - hidden_fold1)
    hidden_scores = np.empty(m, dtype=np.float64)
    for position, client_id in enumerate(admitted):
        report_fold = (checkpoint_count - 1) % 2
        scale_fold = 1 - report_fold
        scale = _robust_scale(
            np.clip(final_improvements[scale_fold][position], -value_clip, value_clip),
            scale_floor,
        )
        values = np.clip(
            np.clip(final_improvements[report_fold][position], -value_clip, value_clip) / scale,
            -1.0,
            1.0,
        )
        # Replay the identical seeded finishability estimator used by the
        # operational final checkpoint.  Keeping the audit seed identical is
        # part of the hidden-final-order isolation contract.
        final_finish_seed = int(
            np.random.SeedSequence(
                [seed, round_number, checkpoint_count, int(client_id), 0xF1A15]
            ).generate_state(1, dtype=np.uint32)[0]
        )
        finish_rng = np.random.default_rng(final_finish_seed)
        completion = _finish_bootstrap(
            step_matrix[position],
            bandwidth_matrix[position],
            0,
            float(
                step_matrix[position].sum()
                + TELEMETRY_BYTES_PER_CHECKPOINT
                * 8.0
                / max(1.0, bandwidth_matrix[position, -1])
            ),
            payload_bytes,
            deadline_s,
            bootstrap_replicates,
            finish_rng,
        )
        p_hat = float(np.mean(completion))
        hidden_scores[position] = float(values.mean()) * p_hat - penalty_lambda * (1.0 - p_hat)
    hidden_order = np.lexsort((admitted, -hidden_scores))
    hidden_topk = admitted[hidden_order[:SELECTED_K]]
    topk_at_commit = np.asarray(commit_payload["topk"], dtype=np.int64)
    rank_error_bool = set(topk_at_commit.tolist()) != set(hidden_topk.tolist())
    interval_covered = bool(
        np.all(hidden_scores >= commit_payload["lower"])
        and np.all(hidden_scores <= commit_payload["upper"])
    )
    audit_wall_s = time.perf_counter() - audit_start

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
        utility_full[client_id] = float(commit_payload["scores"][position])
        admission_full[client_id] = admission_probability
        target_full[client_id] = float(commit_payload["targets"][position])
        inclusion_full[client_id] = admission_probability * target_full[client_id]
        finish_full[client_id] = float(commit_payload["finish_prob"][position])
        effective_full[client_id] = inclusion_full[client_id] * max(
            finish_full[client_id], completion_floor
        )
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
        loss_before_full[client_id] = start_loss_map[client_id]
        loss_after_full[client_id] = end_loss_map[client_id]
        upload_checksum[client_id] = selected_checksums[position]

    candidate_full_s = float(step_matrix.sum())
    commit_step = int(commit_payload["checkpoint_step"])
    used_s = float(step_matrix[:, :commit_step].sum())
    for client_id in selected:
        used_s += float(step_matrix[position_map[int(client_id)], commit_step:].sum())
    saved_s = max(0.0, candidate_full_s - used_s)
    max_inverse = (
        float(np.max(1.0 / np.maximum(effective_selected[raw_weights > 0], 1e-12)))
        if np.any(raw_weights > 0)
        else None
    )
    effective_sample_size = (
        float(1.0 / np.square(normalized_weights).sum())
        if np.square(normalized_weights).sum() > 0
        else 0.0
    )
    committed_topk_hash = sha256_text(",".join(map(str, sorted(topk_at_commit.tolist()))))
    selected_set_hash = sha256_text(",".join(map(str, selected.tolist())))
    on_time_hash = sha256_text(",".join(map(str, eligible.tolist())))
    hidden_hash = sha256_text(",".join(map(str, sorted(hidden_topk.tolist()))))
    certificate_payload = {
        "run_id": run_id,
        "round": round_number,
        "dataset_id": trace.dataset_id,
        "scenario_id": trace.scenario_id,
        "severity": "base",
        "round_start_model_hash": round_start_model_hash,
        "admission_seed": admission_seed,
        "admitted_order_hash": sha256_text(",".join(map(str, admitted.tolist()))),
        "pivotal_seed": pivotal_seed,
        "delta": delta,
        "certified": bool(commit_payload["certified"]),
        "fallback_reason": commit_payload["fallback_reason"],
        "commit_j": int(commit_payload["checkpoint_j"]),
        "commit_steps": commit_step,
        "commit_fraction": commit_step / local_steps,
        "gamma_at_commit": commit_payload["gamma"],
        "vacuous_margin": bool(commit_payload["vacuous"]),
        "committed_topk_hash": committed_topk_hash,
        "selected_set_hash": selected_set_hash,
        "on_time_subset_hash": on_time_hash,
        "aggregation_subset_n": len(eligible),
        "no_op_round": len(eligible) == 0,
        "hidden_final_topk_hash": hidden_hash,
        "topk_intersection_size": len(set(topk_at_commit.tolist()) & set(hidden_topk.tolist())),
        "rank_error": bool(rank_error_bool) if commit_payload["certified"] else None,
        "interval_covered_all": interval_covered,
        "drift_envelope_held": interval_covered,
        "candidate_compute_full_s": candidate_full_s,
        "candidate_compute_used_s": used_s,
        "candidate_compute_saved_s": saved_s,
        "candidate_compute_saved_fraction": saved_s / candidate_full_s if candidate_full_s else 0.0,
        "replay_seed": int(
            np.random.SeedSequence([seed, round_number, 0xA0D17]).generate_state(1, dtype=np.uint32)[0]
        ),
        "replay_gpu_s": audit_gpu_s,
        "replay_cpu_s": max(0.0, audit_wall_s - audit_gpu_s),
        "replay_io_bytes": int(
            sum(value.numel() * value.element_size() for value in checkpoint_cpu.values())
        ),
        "max_effective_inverse_weight": max_inverse,
        "hajek_effective_sample_size": effective_sample_size,
    }
    certificate_payload["certificate_record_hash"] = sha256_text(
        canonical_json(certificate_payload)
    )

    return R2CRoundResult(
        admitted=admitted,
        selected=selected,
        eligible=eligible,
        topk=topk_at_commit,
        prefix_steps=commit_step,
        certified=bool(commit_payload["certified"]),
        fallback_reason=commit_payload["fallback_reason"],
        gamma=commit_payload["gamma"],
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
