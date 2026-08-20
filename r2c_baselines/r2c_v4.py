from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from .data import FederatedData
from .r2c import R2CRoundResult
from .r2c_v2 import PROTOCOL_VERSION as V2_PROTOCOL_VERSION
from .r2c_v3 import PROTOCOL_VERSION as V3_PROTOCOL_VERSION, run_r2c_v3_round
from .traces import Trace
from .training import LocalTrainer
from .utils import canonical_json, sha256_text


PROTOCOL_VERSION = "dual-timescale-deployment-v4"


def validated_deployment_betas(values: Any) -> tuple[float, ...]:
    betas = np.asarray(values, dtype=np.float64)
    if betas.ndim != 1 or len(betas) == 0 or not np.isfinite(betas).all():
        raise ValueError("Deployment EMA betas must be a finite nonempty vector")
    if np.any(betas < 0.0) or np.any(betas >= 1.0):
        raise ValueError("Deployment EMA betas must lie in [0, 1)")
    if len(np.unique(betas)) != len(betas):
        raise ValueError("Deployment EMA betas must be unique")
    return tuple(float(value) for value in np.sort(betas))


def build_deployment_models(
    model: torch.nn.Module, betas: tuple[float, ...]
) -> dict[float, torch.nn.Module]:
    validated = validated_deployment_betas(betas)
    return {beta: copy.deepcopy(model).eval() for beta in validated}


@torch.no_grad()
def update_deployment_model(
    deployment_model: torch.nn.Module,
    fast_model: torch.nn.Module,
    beta: float,
) -> None:
    beta = float(beta)
    if not 0.0 <= beta < 1.0:
        raise ValueError("Deployment EMA beta must lie in [0, 1)")
    deployment_parameters = dict(deployment_model.named_parameters())
    fast_parameters = dict(fast_model.named_parameters())
    if deployment_parameters.keys() != fast_parameters.keys():
        raise ValueError("Fast and deployment parameter structures differ")
    for name, value in deployment_parameters.items():
        value.mul_(beta).add_(fast_parameters[name], alpha=1.0 - beta)

    deployment_buffers = dict(deployment_model.named_buffers())
    fast_buffers = dict(fast_model.named_buffers())
    if deployment_buffers.keys() != fast_buffers.keys():
        raise ValueError("Fast and deployment buffer structures differ")
    for name, value in deployment_buffers.items():
        source = fast_buffers[name]
        if value.is_floating_point():
            value.mul_(beta).add_(source, alpha=1.0 - beta)
        else:
            value.copy_(source)


def run_r2c_v4_round(
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
    """Run the frozen v3 fast path and relabel it for dual-state deployment."""

    result = run_r2c_v3_round(
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
    betas = validated_deployment_betas(
        config.get("r2c_v4_deployment_ema_betas", [0.8, 0.9, 0.95])
    )
    primary_beta = float(config.get("r2c_v4_primary_deployment_beta", betas[0]))
    if primary_beta not in betas:
        raise ValueError("Primary deployment beta must be one of the configured candidates")

    for row in result.checkpoint_rows:
        row["protocol_version"] = PROTOCOL_VERSION
        row["selection_protocol_version"] = V2_PROTOCOL_VERSION
        row["fast_update_protocol_version"] = V3_PROTOCOL_VERSION
    certificate = result.certificate_row
    certificate["protocol_version"] = PROTOCOL_VERSION
    certificate["selection_protocol_version"] = V2_PROTOCOL_VERSION
    certificate["fast_update_protocol_version"] = V3_PROTOCOL_VERSION
    certificate["deployment_rule"] = "server_only_parameter_ema"
    certificate["deployment_state_server_only"] = True
    certificate["deployment_ema_betas_json"] = canonical_json(list(betas))
    certificate["primary_deployment_beta"] = primary_beta
    certificate.pop("certificate_record_hash", None)
    certificate["certificate_record_hash"] = sha256_text(canonical_json(certificate))
    return result
