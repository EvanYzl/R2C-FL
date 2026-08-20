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


PROTOCOL_VERSION = "telemetry-multitimescale-router-deployment-v14"
DEPLOYMENT_RULE = (
    "server_only_fixed_warmup_then_telemetry_triggered_multitimescale_ema_router"
)
FAST_BETA = 0.90
WARMUP_ROUNDS = 200
TRIGGER_HOLD_BETA = 1.0
DEFAULT_CANDIDATE_ID = "CMTR-B975-R20"
CANDIDATES: dict[str, dict[str, float | int]] = {
    "CMTR-B950-R20": {"stable_beta": 0.950, "recovery_rounds": 20},
    "CMTR-B975-R10": {"stable_beta": 0.975, "recovery_rounds": 10},
    "CMTR-B975-R20": {"stable_beta": 0.975, "recovery_rounds": 20},
}


def validated_candidate_id(config: dict[str, Any]) -> str:
    value = str(config.get("r2c_v14_candidate_id", DEFAULT_CANDIDATE_ID))
    if value not in CANDIDATES:
        raise ValueError(
            "r2c_v14_candidate_id must be one of " + ", ".join(CANDIDATES)
        )
    return value


def candidate_stable_beta(candidate_id: str) -> float:
    if candidate_id not in CANDIDATES:
        raise ValueError(f"Unknown CMTR candidate: {candidate_id}")
    value = float(CANDIDATES[candidate_id]["stable_beta"])
    if not np.isfinite(value) or not FAST_BETA < value < 1.0:
        raise AssertionError("CMTR stable beta contract is invalid")
    return value


def candidate_recovery_rounds(candidate_id: str) -> int:
    if candidate_id not in CANDIDATES:
        raise ValueError(f"Unknown CMTR candidate: {candidate_id}")
    value = int(CANDIDATES[candidate_id]["recovery_rounds"])
    if value not in {10, 20}:
        raise AssertionError("CMTR recovery length contract is invalid")
    return value


def validated_warmup_rounds(config: dict[str, Any]) -> int:
    raw = config.get("r2c_v14_warmup_rounds", WARMUP_ROUNDS)
    if isinstance(raw, bool):
        raise ValueError("r2c_v14_warmup_rounds must equal the frozen integer 200")
    value = int(raw)
    if float(raw) != float(value) or value != WARMUP_ROUNDS:
        raise ValueError("r2c_v14_warmup_rounds must equal the frozen integer 200")
    return value


def validated_fast_beta(config: dict[str, Any]) -> float:
    value = float(config.get("r2c_v14_fast_beta", FAST_BETA))
    if not np.isfinite(value) or value != FAST_BETA:
        raise ValueError("r2c_v14_fast_beta must equal the frozen value 0.90")
    return value


@dataclass(frozen=True)
class CMTRObservation:
    round_number: int
    telemetry_trigger: bool
    response_applied: bool
    hold_applied: bool
    recovery_fast_applied: bool
    warmup_fast_applied: bool
    stable_route_applied: bool
    phase: str
    selected_role: str
    selected_beta: float
    fast_update_beta: float
    stable_update_beta: float
    configured_candidate_id: str
    configured_fast_beta: float
    configured_stable_beta: float
    configured_warmup_rounds: int
    configured_recovery_rounds: int
    remaining_before: int
    remaining_after: int
    activation_count: int
    deployment_state_count: int = 2
    state_server_only: bool = True
    labels_used: bool = False
    validation_predictions_used: bool = False
    test_predictions_used: bool = False
    scenario_metadata_used: bool = False
    event_round_used: bool = False
    future_trace_used: bool = False
    raw_global_deployment_used: bool = False

    def update_beta_for(self, deployment_beta: float) -> float:
        value = float(deployment_beta)
        if value == self.configured_fast_beta:
            return float(self.fast_update_beta)
        if value == self.configured_stable_beta:
            return float(self.stable_update_beta)
        raise ValueError(f"beta {value} is not a registered CMTR deployment state")

    def audit_fields(self) -> dict[str, Any]:
        fields = asdict(self)
        fields["round"] = fields.pop("round_number")
        return {f"deployment_cmtr_{key}": value for key, value in fields.items()}


class CausalMultiTimescaleRouter:
    """Route two persistent EMAs using only round number and causal telemetry."""

    STATE_SCHEMA_VERSION = "r2c-v14-cmtr-state-v1"

    def __init__(
        self,
        *,
        candidate_id: str = DEFAULT_CANDIDATE_ID,
        fast_beta: float = FAST_BETA,
        warmup_rounds: int = WARMUP_ROUNDS,
    ) -> None:
        config = {
            "r2c_v14_candidate_id": candidate_id,
            "r2c_v14_fast_beta": fast_beta,
            "r2c_v14_warmup_rounds": warmup_rounds,
        }
        self.candidate_id = validated_candidate_id(config)
        self.fast_beta = validated_fast_beta(config)
        self.stable_beta = candidate_stable_beta(self.candidate_id)
        self.warmup_rounds = validated_warmup_rounds(config)
        self.recovery_rounds = candidate_recovery_rounds(self.candidate_id)
        self._remaining = 0
        self._activation_count = 0
        self._last_round = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CausalMultiTimescaleRouter":
        return cls(
            candidate_id=validated_candidate_id(config),
            fast_beta=validated_fast_beta(config),
            warmup_rounds=validated_warmup_rounds(config),
        )

    @property
    def remaining(self) -> int:
        return int(self._remaining)

    @property
    def activation_count(self) -> int:
        return int(self._activation_count)

    def step(self, round_number: int, telemetry_trigger: bool) -> CMTRObservation:
        round_value = int(round_number)
        if isinstance(round_number, bool) or float(round_number) != float(round_value):
            raise ValueError("CMTR round_number must be an integer")
        if round_value <= self._last_round:
            raise ValueError("CMTR rounds must be strictly increasing")
        self._last_round = round_value

        trigger = bool(telemetry_trigger)
        remaining_before = int(self._remaining)
        if trigger:
            self._activation_count += 1
            self._remaining = self.recovery_rounds
            phase = "trigger_hold"
            selected_role = "stable"
            hold_applied = True
            recovery_fast_applied = False
            warmup_fast_applied = False
            stable_route_applied = True
            fast_update_beta = TRIGGER_HOLD_BETA
            stable_update_beta = TRIGGER_HOLD_BETA
        elif self._remaining > 0:
            phase = "recovery_fast"
            selected_role = "fast"
            hold_applied = False
            recovery_fast_applied = True
            warmup_fast_applied = False
            stable_route_applied = False
            fast_update_beta = self.fast_beta
            stable_update_beta = self.stable_beta
            self._remaining -= 1
        elif round_value <= self.warmup_rounds:
            phase = "warmup_fast"
            selected_role = "fast"
            hold_applied = False
            recovery_fast_applied = False
            warmup_fast_applied = True
            stable_route_applied = False
            fast_update_beta = self.fast_beta
            stable_update_beta = self.stable_beta
        else:
            phase = "stable"
            selected_role = "stable"
            hold_applied = False
            recovery_fast_applied = False
            warmup_fast_applied = False
            stable_route_applied = True
            fast_update_beta = self.fast_beta
            stable_update_beta = self.stable_beta

        selected_beta = self.fast_beta if selected_role == "fast" else self.stable_beta
        return CMTRObservation(
            round_number=round_value,
            telemetry_trigger=trigger,
            response_applied=bool(hold_applied or recovery_fast_applied),
            hold_applied=hold_applied,
            recovery_fast_applied=recovery_fast_applied,
            warmup_fast_applied=warmup_fast_applied,
            stable_route_applied=stable_route_applied,
            phase=phase,
            selected_role=selected_role,
            selected_beta=float(selected_beta),
            fast_update_beta=float(fast_update_beta),
            stable_update_beta=float(stable_update_beta),
            configured_candidate_id=self.candidate_id,
            configured_fast_beta=self.fast_beta,
            configured_stable_beta=self.stable_beta,
            configured_warmup_rounds=self.warmup_rounds,
            configured_recovery_rounds=self.recovery_rounds,
            remaining_before=remaining_before,
            remaining_after=int(self._remaining),
            activation_count=int(self._activation_count),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "fast_beta": self.fast_beta,
            "stable_beta": self.stable_beta,
            "warmup_rounds": self.warmup_rounds,
            "recovery_rounds": self.recovery_rounds,
            "remaining": int(self._remaining),
            "activation_count": int(self._activation_count),
            "last_round": int(self._last_round),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != self.STATE_SCHEMA_VERSION:
            raise ValueError("CMTR serialized-state schema mismatch")
        expected = {
            "candidate_id": self.candidate_id,
            "fast_beta": self.fast_beta,
            "stable_beta": self.stable_beta,
            "warmup_rounds": self.warmup_rounds,
            "recovery_rounds": self.recovery_rounds,
        }
        for key, value in expected.items():
            observed = state.get(key)
            if isinstance(value, float):
                if float(observed) != value:
                    raise ValueError(f"CMTR serialized-state {key} mismatch")
            elif observed != value:
                raise ValueError(f"CMTR serialized-state {key} mismatch")
        remaining = int(state.get("remaining", -1))
        activation_count = int(state.get("activation_count", -1))
        last_round = int(state.get("last_round", -1))
        if not 0 <= remaining <= self.recovery_rounds:
            raise ValueError("CMTR serialized remaining count is invalid")
        if activation_count < 0 or last_round < 0:
            raise ValueError("CMTR serialized counters are invalid")
        self._remaining = remaining
        self._activation_count = activation_count
        self._last_round = last_round


def run_r2c_v14_round(
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
    """Preserve the frozen v5 learning path and relabel deployment control."""

    candidate_id = validated_candidate_id(config)
    fast_beta = validated_fast_beta(config)
    stable_beta = candidate_stable_beta(candidate_id)
    warmup_rounds = validated_warmup_rounds(config)
    recovery_rounds = candidate_recovery_rounds(candidate_id)
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
        row["configured_cmtr_candidate_id"] = candidate_id
        row["configured_cmtr_fast_beta"] = fast_beta
        row["configured_cmtr_stable_beta"] = stable_beta
        row["configured_cmtr_warmup_rounds"] = warmup_rounds
        row["configured_cmtr_recovery_rounds"] = recovery_rounds
    certificate = result.certificate_row
    certificate["protocol_version"] = PROTOCOL_VERSION
    certificate["selection_protocol_version"] = V5_PROTOCOL_VERSION
    certificate["fast_update_protocol_version"] = V3_PROTOCOL_VERSION
    certificate["deployment_protocol_version"] = PROTOCOL_VERSION
    certificate["deployment_rule"] = DEPLOYMENT_RULE
    certificate["configured_cmtr_candidate_id"] = candidate_id
    certificate["configured_cmtr_fast_beta"] = fast_beta
    certificate["configured_cmtr_stable_beta"] = stable_beta
    certificate["configured_cmtr_warmup_rounds"] = warmup_rounds
    certificate["configured_cmtr_recovery_rounds"] = recovery_rounds
    certificate.pop("certificate_record_hash", None)
    certificate["certificate_record_hash"] = sha256_text(canonical_json(certificate))
    return result


__all__ = [
    "CANDIDATES",
    "CMTRObservation",
    "CausalMultiTimescaleRouter",
    "DEFAULT_CANDIDATE_ID",
    "DEFAULT_COOLDOWN_ROUNDS",
    "DEFAULT_FRACTION_THRESHOLD",
    "DEFAULT_LOG_RATIO_THRESHOLD",
    "DEFAULT_MIN_COMPARABLE_CLIENTS",
    "DEPLOYMENT_RULE",
    "FAST_BETA",
    "PROTOCOL_VERSION",
    "TRIGGER_HOLD_BETA",
    "TelemetryShiftDetector",
    "TelemetryShiftObservation",
    "WARMUP_ROUNDS",
    "attach_telemetry_observation",
    "candidate_recovery_rounds",
    "candidate_stable_beta",
    "run_r2c_v14_round",
    "validated_candidate_id",
    "validated_fast_beta",
    "validated_warmup_rounds",
]
