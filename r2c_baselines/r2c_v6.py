from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from .data import FederatedData
from .r2c import R2CRoundResult
from .r2c_v3 import PROTOCOL_VERSION as V3_PROTOCOL_VERSION
from .r2c_v5 import PROTOCOL_VERSION as V5_PROTOCOL_VERSION, run_r2c_v5_round
from .traces import Trace
from .training import LocalTrainer
from .utils import canonical_json, sha256_text


PROTOCOL_VERSION = "telemetry-sync-deployment-v6"
DEPLOYMENT_RULE = "server_only_telemetry_triggered_parameter_ema"
DEFAULT_LOG_RATIO_THRESHOLD = math.log(1.5)
DEFAULT_FRACTION_THRESHOLD = 0.25
DEFAULT_MIN_COMPARABLE_CLIENTS = 10
DEFAULT_COOLDOWN_ROUNDS = 50


@dataclass(frozen=True)
class TelemetryShiftObservation:
    round_number: int
    comparable_clients: int
    changed_clients: int
    changed_fraction: float
    log_ratio_threshold: float
    fraction_threshold: float
    min_comparable_clients: int
    trigger: bool
    cooldown_before: int
    cooldown_after: int
    synchronization_count: int
    state_server_only: bool = True
    labels_used: bool = False
    scenario_metadata_used: bool = False

    def audit_fields(self) -> dict[str, Any]:
        fields = asdict(self)
        fields["round"] = fields.pop("round_number")
        return {f"telemetry_shift_{key}": value for key, value in fields.items()}


class TelemetryShiftDetector:
    """Detect broad duration changes using only server-side duration telemetry."""

    def __init__(
        self,
        num_clients: int,
        *,
        log_ratio_threshold: float = DEFAULT_LOG_RATIO_THRESHOLD,
        fraction_threshold: float = DEFAULT_FRACTION_THRESHOLD,
        min_comparable_clients: int = DEFAULT_MIN_COMPARABLE_CLIENTS,
        cooldown_rounds: int = DEFAULT_COOLDOWN_ROUNDS,
    ) -> None:
        if int(num_clients) <= 0:
            raise ValueError("Telemetry detector requires a positive client count")
        if not np.isfinite(log_ratio_threshold) or float(log_ratio_threshold) <= 0.0:
            raise ValueError("Telemetry log-ratio threshold must be finite and positive")
        if not 0.0 <= float(fraction_threshold) <= 1.0:
            raise ValueError("Telemetry changed-fraction threshold must lie in [0, 1]")
        if int(min_comparable_clients) <= 0:
            raise ValueError("Telemetry detector requires at least one comparable client")
        if int(cooldown_rounds) < 0:
            raise ValueError("Telemetry cooldown must be nonnegative")
        self.num_clients = int(num_clients)
        self.log_ratio_threshold = float(log_ratio_threshold)
        self.fraction_threshold = float(fraction_threshold)
        self.min_comparable_clients = int(min_comparable_clients)
        self.cooldown_rounds = int(cooldown_rounds)
        self._last_duration = np.full(self.num_clients, np.nan, dtype=np.float64)
        self._cooldown_remaining = 0
        self._synchronization_count = 0

    @classmethod
    def from_config(cls, num_clients: int, config: dict[str, Any]) -> "TelemetryShiftDetector":
        return cls(
            num_clients,
            log_ratio_threshold=float(
                config.get(
                    "r2c_v6_duration_log_ratio_threshold",
                    DEFAULT_LOG_RATIO_THRESHOLD,
                )
            ),
            fraction_threshold=float(
                config.get(
                    "r2c_v6_changed_fraction_threshold",
                    DEFAULT_FRACTION_THRESHOLD,
                )
            ),
            min_comparable_clients=int(
                config.get(
                    "r2c_v6_min_comparable_clients",
                    DEFAULT_MIN_COMPARABLE_CLIENTS,
                )
            ),
            cooldown_rounds=int(
                config.get("r2c_v6_cooldown_rounds", DEFAULT_COOLDOWN_ROUNDS)
            ),
        )

    @property
    def synchronization_count(self) -> int:
        return self._synchronization_count

    def observe(
        self,
        round_number: int,
        available_clients: np.ndarray,
        predicted_duration: np.ndarray,
    ) -> TelemetryShiftObservation:
        available = np.asarray(available_clients, dtype=np.int64)
        durations = np.asarray(predicted_duration, dtype=np.float64)
        if durations.shape != (self.num_clients,):
            raise ValueError("Telemetry duration vector has the wrong client dimension")
        if available.ndim != 1 or len(np.unique(available)) != len(available):
            raise ValueError("Available-client telemetry IDs must be a unique vector")
        if np.any(available < 0) or np.any(available >= self.num_clients):
            raise ValueError("Available-client telemetry ID is out of range")
        current = durations[available]
        if not np.isfinite(current).all() or np.any(current <= 0.0):
            raise ValueError("Available-client duration telemetry must be finite and positive")

        previous = self._last_duration[available]
        comparable_mask = np.isfinite(previous)
        comparable_clients = int(comparable_mask.sum())
        if comparable_clients:
            log_ratio = np.abs(np.log(current[comparable_mask] / previous[comparable_mask]))
            changed_clients = int((log_ratio >= self.log_ratio_threshold).sum())
            changed_fraction = float(changed_clients / comparable_clients)
        else:
            changed_clients = 0
            changed_fraction = 0.0

        cooldown_before = int(self._cooldown_remaining)
        trigger = bool(
            cooldown_before == 0
            and comparable_clients >= self.min_comparable_clients
            and changed_fraction >= self.fraction_threshold
        )
        if trigger:
            self._synchronization_count += 1
            cooldown_after = self.cooldown_rounds
        else:
            cooldown_after = max(0, cooldown_before - 1)
        self._cooldown_remaining = cooldown_after
        self._last_duration[available] = current

        return TelemetryShiftObservation(
            round_number=int(round_number),
            comparable_clients=comparable_clients,
            changed_clients=changed_clients,
            changed_fraction=changed_fraction,
            log_ratio_threshold=self.log_ratio_threshold,
            fraction_threshold=self.fraction_threshold,
            min_comparable_clients=self.min_comparable_clients,
            trigger=trigger,
            cooldown_before=cooldown_before,
            cooldown_after=cooldown_after,
            synchronization_count=self._synchronization_count,
        )


def attach_telemetry_observation(
    result: R2CRoundResult,
    observation: TelemetryShiftObservation,
) -> None:
    fields = observation.audit_fields()
    for row in result.checkpoint_rows:
        row.update(fields)
    certificate = result.certificate_row
    certificate.update(fields)
    certificate.pop("certificate_record_hash", None)
    certificate["certificate_record_hash"] = sha256_text(canonical_json(certificate))


def run_r2c_v6_round(
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
    """Run the frozen v5 learning path and relabel its adaptive deployment layer."""

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
    certificate = result.certificate_row
    certificate["protocol_version"] = PROTOCOL_VERSION
    certificate["selection_protocol_version"] = V5_PROTOCOL_VERSION
    certificate["fast_update_protocol_version"] = V3_PROTOCOL_VERSION
    certificate["deployment_protocol_version"] = PROTOCOL_VERSION
    certificate["deployment_rule"] = DEPLOYMENT_RULE
    certificate.pop("certificate_record_hash", None)
    certificate["certificate_record_hash"] = sha256_text(canonical_json(certificate))
    return result


__all__ = [
    "DEFAULT_COOLDOWN_ROUNDS",
    "DEFAULT_FRACTION_THRESHOLD",
    "DEFAULT_LOG_RATIO_THRESHOLD",
    "DEFAULT_MIN_COMPARABLE_CLIENTS",
    "DEPLOYMENT_RULE",
    "PROTOCOL_VERSION",
    "TelemetryShiftDetector",
    "TelemetryShiftObservation",
    "attach_telemetry_observation",
    "run_r2c_v6_round",
]
