from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .data import FederatedData
from .r2c import R2CRoundResult, history_balanced_conditional_targets
from .r2c_v3 import PROTOCOL_VERSION as V3_PROTOCOL_VERSION
from .r2c_v4 import PROTOCOL_VERSION as V4_PROTOCOL_VERSION, run_r2c_v4_round
from .traces import Trace
from .training import LocalTrainer
from .utils import canonical_json, sha256_text


PROTOCOL_VERSION = "history-balanced-exact-k-v5"
HISTORY_TARGET_RULE = "negative_log1p_cumulative_selection_count"


def validated_history_config(config: dict[str, Any]) -> tuple[float, float]:
    history_mix = float(config.get("r2c_v5_history_mix", 0.0))
    history_temperature = float(config.get("r2c_v5_history_temperature", 1.0))
    if not 0.0 <= history_mix <= 1.0:
        raise ValueError("R2C-v5 history-target mixture must lie in [0, 1]")
    if history_temperature <= 0.0 or not np.isfinite(history_temperature):
        raise ValueError("R2C-v5 history-target temperature must be finite and positive")
    return history_mix, history_temperature


def run_r2c_v5_round(
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
    """R2C-v4 with a label-free history-balanced exact-K target mixture."""

    history_mix, history_temperature = validated_history_config(config)
    counts = np.asarray(selection_history_counts, dtype=np.int64)
    if counts.ndim != 1 or np.any(counts < 0):
        raise ValueError("R2C-v5 selection-history counts must be a nonnegative vector")

    result = run_r2c_v4_round(
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
        selection_history_counts=counts,
    )

    for row in result.checkpoint_rows:
        row["protocol_version"] = PROTOCOL_VERSION
        row["selection_protocol_version"] = PROTOCOL_VERSION
        row["fast_update_protocol_version"] = V3_PROTOCOL_VERSION
        row["deployment_protocol_version"] = V4_PROTOCOL_VERSION
    certificate = result.certificate_row
    certificate["protocol_version"] = PROTOCOL_VERSION
    certificate["selection_protocol_version"] = PROTOCOL_VERSION
    certificate["fast_update_protocol_version"] = V3_PROTOCOL_VERSION
    certificate["deployment_protocol_version"] = V4_PROTOCOL_VERSION
    certificate["history_target_rule"] = HISTORY_TARGET_RULE
    certificate["history_target_mix"] = history_mix
    certificate["history_target_temperature"] = history_temperature
    certificate["selection_history_state_server_only"] = True
    certificate["selection_history_labels_used"] = False
    certificate.pop("certificate_record_hash", None)
    certificate["certificate_record_hash"] = sha256_text(canonical_json(certificate))
    return result


__all__ = [
    "HISTORY_TARGET_RULE",
    "PROTOCOL_VERSION",
    "history_balanced_conditional_targets",
    "run_r2c_v5_round",
    "validated_history_config",
]
