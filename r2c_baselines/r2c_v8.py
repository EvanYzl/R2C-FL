from __future__ import annotations

from dataclasses import asdict, dataclass
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


PROTOCOL_VERSION = "telemetry-recovery-pulse-deployment-v8"
DEPLOYMENT_RULE = "server_only_telemetry_triggered_hold_then_recovery_pulse"
DEFAULT_TRIGGER_DEPLOYMENT_BETA = 1.0
DEFAULT_RECOVERY_PULSE_BETA = 0.5
DEFAULT_RECOVERY_PULSE_ROUNDS = 5


def _validated_beta(config: dict[str, Any], key: str, default: float) -> float:
    value = float(config.get(key, default))
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be finite and lie in [0, 1]")
    return value


def validated_trigger_deployment_beta(config: dict[str, Any]) -> float:
    return _validated_beta(
        config,
        "r2c_v8_trigger_deployment_beta",
        DEFAULT_TRIGGER_DEPLOYMENT_BETA,
    )


def validated_recovery_pulse_beta(config: dict[str, Any]) -> float:
    return _validated_beta(
        config,
        "r2c_v8_recovery_pulse_beta",
        DEFAULT_RECOVERY_PULSE_BETA,
    )


def validated_recovery_pulse_rounds(config: dict[str, Any]) -> int:
    raw = config.get("r2c_v8_recovery_pulse_rounds", DEFAULT_RECOVERY_PULSE_ROUNDS)
    if isinstance(raw, bool):
        raise ValueError("r2c_v8_recovery_pulse_rounds must be a positive integer")
    value = int(raw)
    if float(raw) != float(value) or value <= 0:
        raise ValueError("r2c_v8_recovery_pulse_rounds must be a positive integer")
    return value


@dataclass(frozen=True)
class RecoveryPulseObservation:
    round_number: int
    telemetry_trigger: bool
    response_applied: bool
    hold_applied: bool
    recovery_applied: bool
    phase: str
    override_beta: float | None
    configured_trigger_beta: float
    configured_recovery_beta: float
    configured_recovery_rounds: int
    remaining_before: int
    remaining_after: int
    activation_count: int
    state_server_only: bool = True
    labels_used: bool = False
    scenario_metadata_used: bool = False

    def audit_fields(self) -> dict[str, Any]:
        fields = asdict(self)
        fields["round"] = fields.pop("round_number")
        return {f"deployment_pulse_{key}": value for key, value in fields.items()}


class DeploymentRecoveryPulse:
    """Auditable server-only hold followed by a fixed low-beta EMA pulse."""

    def __init__(
        self,
        *,
        trigger_beta: float = DEFAULT_TRIGGER_DEPLOYMENT_BETA,
        recovery_beta: float = DEFAULT_RECOVERY_PULSE_BETA,
        recovery_rounds: int = DEFAULT_RECOVERY_PULSE_ROUNDS,
    ) -> None:
        config = {
            "r2c_v8_trigger_deployment_beta": trigger_beta,
            "r2c_v8_recovery_pulse_beta": recovery_beta,
            "r2c_v8_recovery_pulse_rounds": recovery_rounds,
        }
        self.trigger_beta = validated_trigger_deployment_beta(config)
        self.recovery_beta = validated_recovery_pulse_beta(config)
        self.recovery_rounds = validated_recovery_pulse_rounds(config)
        self._remaining = 0
        self._activation_count = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DeploymentRecoveryPulse":
        return cls(
            trigger_beta=validated_trigger_deployment_beta(config),
            recovery_beta=validated_recovery_pulse_beta(config),
            recovery_rounds=validated_recovery_pulse_rounds(config),
        )

    @property
    def remaining(self) -> int:
        return self._remaining

    @property
    def activation_count(self) -> int:
        return self._activation_count

    def step(self, round_number: int, telemetry_trigger: bool) -> RecoveryPulseObservation:
        remaining_before = int(self._remaining)
        trigger = bool(telemetry_trigger)
        if trigger:
            self._activation_count += 1
            self._remaining = self.recovery_rounds
            phase = "trigger_hold"
            override_beta: float | None = self.trigger_beta
            hold_applied = True
            recovery_applied = False
        elif self._remaining > 0:
            phase = "recovery_pulse"
            override_beta = self.recovery_beta
            hold_applied = False
            recovery_applied = True
            self._remaining -= 1
        else:
            phase = "ordinary"
            override_beta = None
            hold_applied = False
            recovery_applied = False
        response_applied = bool(hold_applied or recovery_applied)
        return RecoveryPulseObservation(
            round_number=int(round_number),
            telemetry_trigger=trigger,
            response_applied=response_applied,
            hold_applied=hold_applied,
            recovery_applied=recovery_applied,
            phase=phase,
            override_beta=override_beta,
            configured_trigger_beta=self.trigger_beta,
            configured_recovery_beta=self.recovery_beta,
            configured_recovery_rounds=self.recovery_rounds,
            remaining_before=remaining_before,
            remaining_after=int(self._remaining),
            activation_count=int(self._activation_count),
        )


def run_r2c_v8_round(
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
    """Run the frozen v5 learning path and relabel only its deployment layer."""

    trigger_beta = validated_trigger_deployment_beta(config)
    recovery_beta = validated_recovery_pulse_beta(config)
    recovery_rounds = validated_recovery_pulse_rounds(config)
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
        row["configured_recovery_pulse_beta"] = recovery_beta
        row["configured_recovery_pulse_rounds"] = recovery_rounds
    certificate = result.certificate_row
    certificate["protocol_version"] = PROTOCOL_VERSION
    certificate["selection_protocol_version"] = V5_PROTOCOL_VERSION
    certificate["fast_update_protocol_version"] = V3_PROTOCOL_VERSION
    certificate["deployment_protocol_version"] = PROTOCOL_VERSION
    certificate["deployment_rule"] = DEPLOYMENT_RULE
    certificate["configured_trigger_deployment_beta"] = trigger_beta
    certificate["configured_recovery_pulse_beta"] = recovery_beta
    certificate["configured_recovery_pulse_rounds"] = recovery_rounds
    certificate.pop("certificate_record_hash", None)
    certificate["certificate_record_hash"] = sha256_text(canonical_json(certificate))
    return result


__all__ = [
    "DEFAULT_COOLDOWN_ROUNDS",
    "DEFAULT_FRACTION_THRESHOLD",
    "DEFAULT_LOG_RATIO_THRESHOLD",
    "DEFAULT_MIN_COMPARABLE_CLIENTS",
    "DEFAULT_RECOVERY_PULSE_BETA",
    "DEFAULT_RECOVERY_PULSE_ROUNDS",
    "DEFAULT_TRIGGER_DEPLOYMENT_BETA",
    "DEPLOYMENT_RULE",
    "DeploymentRecoveryPulse",
    "PROTOCOL_VERSION",
    "RecoveryPulseObservation",
    "TelemetryShiftDetector",
    "TelemetryShiftObservation",
    "attach_telemetry_observation",
    "run_r2c_v8_round",
    "validated_recovery_pulse_beta",
    "validated_recovery_pulse_rounds",
    "validated_trigger_deployment_beta",
]
