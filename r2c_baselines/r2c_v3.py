from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .data import FederatedData
from .r2c import R2CRoundResult
from .r2c_v2 import PROTOCOL_VERSION as V2_PROTOCOL_VERSION, run_r2c_v2_round
from .traces import Trace
from .training import LocalTrainer
from .utils import canonical_json, sha256_text


PROTOCOL_VERSION = "anchor-guarded-trust-v3"


def _validated_alphas(values: Any) -> np.ndarray:
    alphas = np.asarray(values, dtype=np.float64)
    if alphas.ndim != 1 or len(alphas) == 0 or not np.isfinite(alphas).all():
        raise ValueError("Server alphas must be a finite nonempty vector")
    if np.any(alphas < 0.0) or np.any(alphas > 1.0):
        raise ValueError("Server alphas must lie in [0, 1]")
    if len(np.unique(alphas)) != len(alphas):
        raise ValueError("Server alphas must be unique")
    return np.sort(alphas)[::-1]


def _choose_alpha(
    alphas: np.ndarray,
    fold0_loss: np.ndarray,
    fold1_loss: np.ndarray,
    rule: str,
    relative_tolerance: float,
    minimum_alpha: float = 0.0,
) -> tuple[int, np.ndarray]:
    """Choose a server step using only two validation guard folds."""

    alphas = _validated_alphas(alphas)
    fold0_loss = np.asarray(fold0_loss, dtype=np.float64)
    fold1_loss = np.asarray(fold1_loss, dtype=np.float64)
    if fold0_loss.shape != alphas.shape or fold1_loss.shape != alphas.shape:
        raise ValueError("Guard loss arrays must match server alphas")
    if not np.isfinite(fold0_loss).all() or not np.isfinite(fold1_loss).all():
        raise ValueError("Guard losses must be finite")
    zero_positions = np.flatnonzero(np.isclose(alphas, 0.0, atol=0.0, rtol=0.0))
    if len(zero_positions) != 1:
        raise ValueError("Adaptive server-alpha rules require exactly one alpha=0")
    if not 0.0 <= float(minimum_alpha) <= 1.0:
        raise ValueError("Minimum server alpha must lie in [0, 1]")
    zero = int(zero_positions[0])
    base0 = max(float(fold0_loss[zero]), 1.0e-12)
    base1 = max(float(fold1_loss[zero]), 1.0e-12)
    relative = np.maximum(fold0_loss / base0 - 1.0, fold1_loss / base1 - 1.0)
    allowed = np.flatnonzero(alphas >= float(minimum_alpha) - 1.0e-12)
    if len(allowed) == 0:
        raise ValueError("No candidate server alpha satisfies the configured minimum")
    if rule == "largest_noninferior":
        accepted = allowed[relative[allowed] <= float(relative_tolerance) + 1.0e-12]
        chosen = int(accepted[0]) if len(accepted) else int(allowed[-1])
    elif rule == "minimax":
        best = float(np.min(relative[allowed]))
        chosen = int(allowed[np.flatnonzero(relative[allowed] <= best + 1.0e-12)[0]])
    elif rule == "mean_loss":
        mean_relative = 0.5 * (fold0_loss / base0 + fold1_loss / base1) - 1.0
        best = float(np.min(mean_relative[allowed]))
        chosen = int(allowed[np.flatnonzero(mean_relative[allowed] <= best + 1.0e-12)[0]])
    else:
        raise ValueError(f"Unknown server guard rule: {rule}")
    return chosen, relative


def run_r2c_v3_round(
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
    """A-R2C-v2 selection with a validation-only guarded server trust step."""

    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    result = run_r2c_v2_round(
        model=model,
        trainer=trainer,
        data=data,
        trace=trace,
        round_number=round_number,
        available=available,
        sample_counts=sample_counts,
        learning_rate=learning_rate,
        payload_bytes=payload_bytes,
        seed=seed,
        config=config,
        run_id=run_id,
        round_start_model_hash=round_start_model_hash,
        full_logging=full_logging,
        selection_history_counts=selection_history_counts,
    )
    after = {name: value.detach().clone() for name, value in model.named_parameters()}

    fixed_alpha = config.get("r2c_v3_fixed_server_alpha")
    rule = str(config.get("r2c_v3_guard_rule", "minimax"))
    guard_per_fold = int(config.get("r2c_v3_guard_per_fold", 32))
    selection_per_fold = int(config.get("r2c_v2_anchor_per_fold", 16))
    relative_tolerance = float(config.get("r2c_v3_guard_relative_tolerance", 0.0))
    minimum_alpha = float(config.get("r2c_v3_min_server_alpha", 0.0))
    guard_wall_s = 0.0
    guard_gpu_s = 0.0
    fold0_mean: np.ndarray | None = None
    fold1_mean: np.ndarray | None = None
    guard_relative: np.ndarray | None = None

    guard_start = time.perf_counter()
    if fixed_alpha is not None:
        alpha = float(fixed_alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Fixed server alpha must lie in [0, 1]")
        alphas = np.asarray([alpha], dtype=np.float64)
        chosen = 0
        rule = "fixed"
    else:
        alphas = _validated_alphas(
            config.get("r2c_v3_server_alphas", [1.0, 0.75, 0.5, 0.25, 0.0])
        )
        candidate_params = {
            name: torch.stack(
                [before[name] + float(alpha_value) * (after[name] - before[name]) for alpha_value in alphas],
                dim=0,
            )
            for name in before
        }
        fold0, wall0, gpu0 = trainer.r2c_guard_losses(
            candidate_params, 0, guard_per_fold, selection_per_fold, len(alphas)
        )
        fold1, wall1, gpu1 = trainer.r2c_guard_losses(
            candidate_params, 1, guard_per_fold, selection_per_fold, len(alphas)
        )
        guard_wall_s += wall0 + wall1
        guard_gpu_s += gpu0 + gpu1
        fold0_mean = fold0.mean(axis=1)
        fold1_mean = fold1.mean(axis=1)
        chosen, guard_relative = _choose_alpha(
            alphas,
            fold0_mean,
            fold1_mean,
            rule,
            relative_tolerance,
            minimum_alpha,
        )
        alpha = float(alphas[chosen])
        del candidate_params, fold0, fold1

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(before[name] + alpha * (after[name] - before[name]))
    guard_wall_s += time.perf_counter() - guard_start - guard_wall_s

    result.aggregate_wall_s += guard_wall_s
    result.algorithm_gpu_s += guard_gpu_s
    for row in result.checkpoint_rows:
        row["protocol_version"] = PROTOCOL_VERSION
        row["selection_protocol_version"] = V2_PROTOCOL_VERSION
    certificate = result.certificate_row
    certificate["protocol_version"] = PROTOCOL_VERSION
    certificate["selection_protocol_version"] = V2_PROTOCOL_VERSION
    certificate["server_step_alpha"] = alpha
    certificate["server_step_rule"] = rule
    certificate["server_step_noop"] = bool(alpha == 0.0)
    certificate["guard_disjoint_from_selection_anchor"] = not bool(
        config.get("r2c_ablation_no_valid_crossfit", False)
    )
    certificate["guard_per_fold"] = 0 if fixed_alpha is not None else guard_per_fold
    certificate["guard_relative_tolerance"] = relative_tolerance
    certificate["guard_min_server_alpha"] = minimum_alpha
    certificate["guard_candidate_alphas_json"] = canonical_json(alphas.tolist())
    certificate["guard_fold0_losses_json"] = (
        None if fold0_mean is None else canonical_json(fold0_mean.tolist())
    )
    certificate["guard_fold1_losses_json"] = (
        None if fold1_mean is None else canonical_json(fold1_mean.tolist())
    )
    certificate["guard_worst_relative_json"] = (
        None if guard_relative is None else canonical_json(guard_relative.tolist())
    )
    certificate["guard_wall_s"] = guard_wall_s
    certificate["guard_gpu_s"] = guard_gpu_s
    certificate.pop("certificate_record_hash", None)
    certificate["certificate_record_hash"] = sha256_text(canonical_json(certificate))
    return result
