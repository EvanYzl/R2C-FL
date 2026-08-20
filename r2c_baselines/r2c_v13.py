from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import sqrt
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
from .training import LocalTrainer, model_state_hash
from .utils import canonical_json, sha256_text


PROTOCOL_VERSION = "telemetry-dual-anchor-recovery-envelope-deployment-v13"
DEPLOYMENT_RULE = "server_only_telemetry_triggered_dual_anchor_recovery_envelope"
DEFAULT_SCHEDULE_ID = "DARE-L5"
SCHEDULES: dict[str, tuple[str, int]] = {
    "DARE-L5": ("linear", 5),
    "DARE-C5": ("sqrt", 5),
    "DARE-L8": ("linear", 8),
}


def validated_schedule_id(config: dict[str, Any]) -> str:
    value = str(config.get("r2c_v13_schedule_id", DEFAULT_SCHEDULE_ID))
    if value not in SCHEDULES:
        raise ValueError(
            "r2c_v13_schedule_id must be one of " + ", ".join(sorted(SCHEDULES))
        )
    return value


def schedule_rounds(schedule_id: str) -> int:
    if schedule_id not in SCHEDULES:
        raise ValueError(f"Unknown DARE schedule: {schedule_id}")
    return int(SCHEDULES[schedule_id][1])


def schedule_lambda(schedule_id: str, recovery_index: int) -> float:
    if schedule_id not in SCHEDULES:
        raise ValueError(f"Unknown DARE schedule: {schedule_id}")
    kind, rounds = SCHEDULES[schedule_id]
    if isinstance(recovery_index, bool) or not 1 <= int(recovery_index) <= rounds:
        raise ValueError(f"recovery_index must lie in [1, {rounds}]")
    index = int(recovery_index)
    if float(recovery_index) != float(index):
        raise ValueError("recovery_index must be an integer")
    fraction = float(index / rounds)
    value = sqrt(fraction) if kind == "sqrt" else fraction
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise AssertionError("DARE schedule produced an invalid lambda")
    return float(value)


@dataclass(frozen=True)
class DAREObservation:
    round_number: int
    telemetry_trigger: bool
    response_applied: bool
    hold_applied: bool
    envelope_applied: bool
    tracking_applied: bool
    phase: str
    lambda_value: float | None
    equivalent_beta: float | None
    configured_schedule_id: str
    configured_recovery_rounds: int
    recovery_index: int
    remaining_before: int
    remaining_after: int
    activation_count: int
    pre_anchor_capture_requested: bool
    post_anchor_capture_requested: bool
    pre_anchor_captured: bool = False
    pre_anchor_released: bool = False
    pre_anchor_hash: str | None = None
    post_anchor_hash: str | None = None
    anchor_tensor_count_before: int = 0
    anchor_tensor_count_after: int = 0
    anchor_bytes_before: int = 0
    anchor_bytes_after: int = 0
    state_server_only: bool = True
    labels_used: bool = False
    scenario_metadata_used: bool = False
    future_trace_used: bool = False

    def audit_fields(self) -> dict[str, Any]:
        fields = asdict(self)
        fields["round"] = fields.pop("round_number")
        return {f"deployment_dare_{key}": value for key, value in fields.items()}


class DualAnchorRecoveryEnvelope:
    """Causal server-only pre-anchor interpolation with persistent post-shift tracking.

    The only additional full model state is the pre-event deployment anchor.  The
    trigger-round global model is the post anchor already resident in the server;
    its hash is retained for lineage, rather than copying a second model.
    """

    STATE_SCHEMA_VERSION = "r2c-v13-dare-state-v1"

    def __init__(self, *, schedule_id: str = DEFAULT_SCHEDULE_ID) -> None:
        self.schedule_id = validated_schedule_id(
            {"r2c_v13_schedule_id": schedule_id}
        )
        self.recovery_rounds = schedule_rounds(self.schedule_id)
        self._remaining = 0
        self._recovery_index = 0
        self._activation_count = 0
        self._activated = False
        self._last_round = 0
        self._pre_anchor_state: dict[str, torch.Tensor] | None = None
        self._pre_anchor_hash: str | None = None
        self._post_anchor_hash: str | None = None
        self._anchor_bytes = 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DualAnchorRecoveryEnvelope":
        return cls(schedule_id=validated_schedule_id(config))

    @property
    def remaining(self) -> int:
        return int(self._remaining)

    @property
    def recovery_index(self) -> int:
        return int(self._recovery_index)

    @property
    def activation_count(self) -> int:
        return int(self._activation_count)

    @property
    def activated(self) -> bool:
        return bool(self._activated)

    @property
    def anchor_tensor_count(self) -> int:
        return 0 if self._pre_anchor_state is None else len(self._pre_anchor_state)

    @property
    def anchor_bytes(self) -> int:
        return int(self._anchor_bytes if self._pre_anchor_state is not None else 0)

    def step(self, round_number: int, telemetry_trigger: bool) -> DAREObservation:
        round_value = int(round_number)
        if round_value <= self._last_round:
            raise ValueError("DARE rounds must be strictly increasing")
        self._last_round = round_value
        trigger = bool(telemetry_trigger)
        remaining_before = int(self._remaining)
        anchor_count_before = self.anchor_tensor_count
        anchor_bytes_before = self.anchor_bytes

        if trigger:
            self._activation_count += 1
            self._activated = True
            self._remaining = self.recovery_rounds
            self._recovery_index = 0
            phase = "trigger_anchor_hold"
            lambda_value: float | None = 0.0
            hold_applied = True
            envelope_applied = False
            tracking_applied = False
            capture_requested = True
        elif self._remaining > 0:
            self._recovery_index += 1
            lambda_value = schedule_lambda(self.schedule_id, self._recovery_index)
            self._remaining -= 1
            phase = "recovery_envelope"
            hold_applied = False
            envelope_applied = True
            tracking_applied = False
            capture_requested = False
        elif self._activated:
            lambda_value = 1.0
            phase = "post_shift_tracking"
            hold_applied = False
            envelope_applied = False
            tracking_applied = True
            capture_requested = False
        else:
            lambda_value = None
            phase = "ordinary"
            hold_applied = False
            envelope_applied = False
            tracking_applied = False
            capture_requested = False

        response_applied = bool(
            hold_applied or envelope_applied or tracking_applied
        )
        equivalent_beta = (
            None if lambda_value is None else float(1.0 - lambda_value)
        )
        return DAREObservation(
            round_number=round_value,
            telemetry_trigger=trigger,
            response_applied=response_applied,
            hold_applied=hold_applied,
            envelope_applied=envelope_applied,
            tracking_applied=tracking_applied,
            phase=phase,
            lambda_value=lambda_value,
            equivalent_beta=equivalent_beta,
            configured_schedule_id=self.schedule_id,
            configured_recovery_rounds=self.recovery_rounds,
            recovery_index=int(self._recovery_index),
            remaining_before=remaining_before,
            remaining_after=int(self._remaining),
            activation_count=int(self._activation_count),
            pre_anchor_capture_requested=capture_requested,
            post_anchor_capture_requested=capture_requested,
            pre_anchor_hash=self._pre_anchor_hash,
            post_anchor_hash=self._post_anchor_hash,
            anchor_tensor_count_before=anchor_count_before,
            anchor_tensor_count_after=self.anchor_tensor_count,
            anchor_bytes_before=anchor_bytes_before,
            anchor_bytes_after=self.anchor_bytes,
        )

    @staticmethod
    def _model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {name: value for name, value in model.state_dict().items()}

    def _capture_anchor(
        self,
        deployment_model: torch.nn.Module,
        current_model: torch.nn.Module,
    ) -> None:
        deployment_state = self._model_state(deployment_model)
        current_state = self._model_state(current_model)
        if deployment_state.keys() != current_state.keys():
            raise ValueError("DARE deployment/global model structures differ")
        self._pre_anchor_state = {
            name: value.detach().clone() for name, value in deployment_state.items()
        }
        self._anchor_bytes = int(
            sum(value.numel() * value.element_size() for value in self._pre_anchor_state.values())
        )
        self._pre_anchor_hash = model_state_hash(deployment_model)
        self._post_anchor_hash = model_state_hash(current_model)

    @torch.no_grad()
    def _interpolate(
        self,
        deployment_model: torch.nn.Module,
        current_model: torch.nn.Module,
        lambda_value: float,
    ) -> None:
        value = float(lambda_value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("DARE lambda must be finite and lie in [0, 1]")
        deployment_state = self._model_state(deployment_model)
        current_state = self._model_state(current_model)
        if deployment_state.keys() != current_state.keys():
            raise ValueError("DARE deployment/global model structures differ")
        if value < 1.0 and self._pre_anchor_state is None:
            raise RuntimeError("DARE pre anchor is unavailable during recovery")
        if self._pre_anchor_state is not None and (
            self._pre_anchor_state.keys() != deployment_state.keys()
        ):
            raise ValueError("DARE serialized anchor structure differs from the model")

        for name, target in deployment_state.items():
            source = current_state[name]
            if target.shape != source.shape or target.dtype != source.dtype:
                raise ValueError(f"DARE tensor mismatch for {name}")
            if value == 1.0:
                target.copy_(source)
                continue
            assert self._pre_anchor_state is not None
            anchor = self._pre_anchor_state[name].to(
                device=target.device, dtype=target.dtype
            )
            if target.is_floating_point() or target.is_complex():
                target.copy_(anchor).mul_(1.0 - value).add_(source, alpha=value)
            else:
                target.copy_(anchor if value == 0.0 else source)

    def apply(
        self,
        observation: DAREObservation,
        deployment_model: torch.nn.Module,
        current_model: torch.nn.Module,
    ) -> DAREObservation:
        if observation.round_number != self._last_round:
            raise ValueError("DARE observation does not match the current controller round")
        captured = False
        released = False
        if observation.pre_anchor_capture_requested:
            self._capture_anchor(deployment_model, current_model)
            captured = True
        if observation.envelope_applied or observation.tracking_applied:
            assert observation.lambda_value is not None
            self._interpolate(
                deployment_model,
                current_model,
                observation.lambda_value,
            )
        if observation.envelope_applied and observation.remaining_after == 0:
            self._pre_anchor_state = None
            self._anchor_bytes = 0
            released = True
        return replace(
            observation,
            pre_anchor_captured=captured,
            pre_anchor_released=released,
            pre_anchor_hash=self._pre_anchor_hash,
            post_anchor_hash=self._post_anchor_hash,
            anchor_tensor_count_after=self.anchor_tensor_count,
            anchor_bytes_after=self.anchor_bytes,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "schedule_id": self.schedule_id,
            "remaining": int(self._remaining),
            "recovery_index": int(self._recovery_index),
            "activation_count": int(self._activation_count),
            "activated": bool(self._activated),
            "last_round": int(self._last_round),
            "pre_anchor_hash": self._pre_anchor_hash,
            "post_anchor_hash": self._post_anchor_hash,
            "anchor_bytes": int(self._anchor_bytes),
            "pre_anchor_state": (
                None
                if self._pre_anchor_state is None
                else {
                    name: value.detach().clone()
                    for name, value in self._pre_anchor_state.items()
                }
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != self.STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported DARE state schema")
        if state.get("schedule_id") != self.schedule_id:
            raise ValueError("DARE state schedule differs from the controller")
        remaining = int(state.get("remaining", -1))
        recovery_index = int(state.get("recovery_index", -1))
        if not 0 <= remaining <= self.recovery_rounds:
            raise ValueError("Invalid DARE remaining-round state")
        if not 0 <= recovery_index <= self.recovery_rounds:
            raise ValueError("Invalid DARE recovery-index state")
        raw_anchor = state.get("pre_anchor_state")
        if raw_anchor is not None and not isinstance(raw_anchor, dict):
            raise ValueError("Invalid DARE anchor state")
        anchor_state = (
            None
            if raw_anchor is None
            else {
                str(name): value.detach().clone()
                for name, value in raw_anchor.items()
                if isinstance(value, torch.Tensor)
            }
        )
        if raw_anchor is not None and len(anchor_state or {}) != len(raw_anchor):
            raise ValueError("DARE anchor state contains a non-tensor value")
        if remaining > 0 and anchor_state is None:
            raise ValueError("DARE recovery state is missing its pre anchor")
        self._remaining = remaining
        self._recovery_index = recovery_index
        self._activation_count = int(state.get("activation_count", 0))
        self._activated = bool(state.get("activated", False))
        self._last_round = int(state.get("last_round", 0))
        self._pre_anchor_hash = state.get("pre_anchor_hash")
        self._post_anchor_hash = state.get("post_anchor_hash")
        self._pre_anchor_state = anchor_state
        self._anchor_bytes = int(state.get("anchor_bytes", 0)) if anchor_state else 0


def run_r2c_v13_round(
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
    """Preserve the frozen v5 learning path and relabel only deployment control."""

    schedule_id = validated_schedule_id(config)
    recovery_rounds = schedule_rounds(schedule_id)
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
        row["configured_dare_schedule_id"] = schedule_id
        row["configured_dare_recovery_rounds"] = recovery_rounds
    certificate = result.certificate_row
    certificate["protocol_version"] = PROTOCOL_VERSION
    certificate["selection_protocol_version"] = V5_PROTOCOL_VERSION
    certificate["fast_update_protocol_version"] = V3_PROTOCOL_VERSION
    certificate["deployment_protocol_version"] = PROTOCOL_VERSION
    certificate["deployment_rule"] = DEPLOYMENT_RULE
    certificate["configured_dare_schedule_id"] = schedule_id
    certificate["configured_dare_recovery_rounds"] = recovery_rounds
    certificate.pop("certificate_record_hash", None)
    certificate["certificate_record_hash"] = sha256_text(canonical_json(certificate))
    return result


__all__ = [
    "DAREObservation",
    "DEFAULT_COOLDOWN_ROUNDS",
    "DEFAULT_FRACTION_THRESHOLD",
    "DEFAULT_LOG_RATIO_THRESHOLD",
    "DEFAULT_MIN_COMPARABLE_CLIENTS",
    "DEFAULT_SCHEDULE_ID",
    "DEPLOYMENT_RULE",
    "DualAnchorRecoveryEnvelope",
    "PROTOCOL_VERSION",
    "SCHEDULES",
    "TelemetryShiftDetector",
    "TelemetryShiftObservation",
    "attach_telemetry_observation",
    "run_r2c_v13_round",
    "schedule_lambda",
    "schedule_rounds",
    "validated_schedule_id",
]
