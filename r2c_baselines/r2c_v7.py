from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .data import FederatedData
from .r2c import R2CRoundResult
from .r2c_v3 import PROTOCOL_VERSION as V3_PROTOCOL_VERSION
from .r2c_v5 import PROTOCOL_VERSION as V5_PROTOCOL_VERSION, run_r2c_v5_round
from .r2c_v6 import (
    DEFAULT_COOLDOWN_ROUNDS,
    DEFAULT_FRACTION_THRESHOLD,
    DEFAULT_LOG_RATIO_THRESHOLD,
    DEFAULT_MIN_COMPARABLE_CLIENTS,
    TelemetryShiftDetector,
    TelemetryShiftObservation,
    attach_telemetry_observation,
)
from .traces import Trace
from .training import LocalTrainer
from .utils import canonical_json, sha256_text


PROTOCOL_VERSION = "telemetry-quarantine-deployment-v7"
DEPLOYMENT_RULE = "server_only_telemetry_triggered_one_round_deployment_quarantine"
DEFAULT_TRIGGER_DEPLOYMENT_BETA = 1.0


def validated_trigger_deployment_beta(config: dict[str, Any]) -> float:
    value = float(config.get("r2c_v7_trigger_deployment_beta", DEFAULT_TRIGGER_DEPLOYMENT_BETA))
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Trigger deployment beta must be finite and lie in [0, 1]")
    return value


def run_r2c_v7_round(
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
    selection_history_counts: np.ndarray,
) -> R2CRoundResult:
    """Run the frozen v5 learning path with a telemetry-gated deployment hold."""

    trigger_beta = validated_trigger_deployment_beta(config)
    result = run_r2c_v5_round(
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
    for row in result.checkpoint_rows:
        row["protocol_version"] = PROTOCOL_VERSION
        row["selection_protocol_version"] = V5_PROTOCOL_VERSION
        row["fast_update_protocol_version"] = V3_PROTOCOL_VERSION
        row["deployment_protocol_version"] = PROTOCOL_VERSION
        row["configured_trigger_deployment_beta"] = trigger_beta
    certificate = result.certificate_row
    certificate["protocol_version"] = PROTOCOL_VERSION
    certificate["selection_protocol_version"] = V5_PROTOCOL_VERSION
    certificate["fast_update_protocol_version"] = V3_PROTOCOL_VERSION
    certificate["deployment_protocol_version"] = PROTOCOL_VERSION
    certificate["deployment_rule"] = DEPLOYMENT_RULE
    certificate["configured_trigger_deployment_beta"] = trigger_beta
    certificate.pop("certificate_record_hash", None)
    certificate["certificate_record_hash"] = sha256_text(canonical_json(certificate))
    return result


__all__ = [
    "DEFAULT_COOLDOWN_ROUNDS",
    "DEFAULT_FRACTION_THRESHOLD",
    "DEFAULT_LOG_RATIO_THRESHOLD",
    "DEFAULT_MIN_COMPARABLE_CLIENTS",
    "DEFAULT_TRIGGER_DEPLOYMENT_BETA",
    "DEPLOYMENT_RULE",
    "PROTOCOL_VERSION",
    "TelemetryShiftDetector",
    "TelemetryShiftObservation",
    "attach_telemetry_observation",
    "run_r2c_v7_round",
    "validated_trigger_deployment_beta",
]
