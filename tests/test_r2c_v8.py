from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from r2c_baselines import r2c_v8


def test_recovery_pulse_configuration_validation() -> None:
    config = {
        "r2c_v8_trigger_deployment_beta": 1.0,
        "r2c_v8_recovery_pulse_beta": 0.5,
        "r2c_v8_recovery_pulse_rounds": 5,
    }
    pulse = r2c_v8.DeploymentRecoveryPulse.from_config(config)
    assert pulse.trigger_beta == 1.0
    assert pulse.recovery_beta == 0.5
    assert pulse.recovery_rounds == 5
    for key, value in (
        ("r2c_v8_trigger_deployment_beta", -0.01),
        ("r2c_v8_trigger_deployment_beta", 1.01),
        ("r2c_v8_recovery_pulse_beta", float("nan")),
        ("r2c_v8_recovery_pulse_beta", float("inf")),
    ):
        with pytest.raises(ValueError):
            r2c_v8.DeploymentRecoveryPulse.from_config({key: value})
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            r2c_v8.DeploymentRecoveryPulse.from_config(
                {"r2c_v8_recovery_pulse_rounds": value}
            )


def test_no_trigger_path_is_ordinary_and_state_free() -> None:
    pulse = r2c_v8.DeploymentRecoveryPulse(recovery_beta=0.5, recovery_rounds=3)
    observations = [pulse.step(round_number, False) for round_number in range(1, 8)]
    assert all(observation.phase == "ordinary" for observation in observations)
    assert all(observation.override_beta is None for observation in observations)
    assert not any(observation.response_applied for observation in observations)
    assert pulse.remaining == 0
    assert pulse.activation_count == 0


def test_trigger_hold_exact_pulse_duration_and_resume() -> None:
    pulse = r2c_v8.DeploymentRecoveryPulse(
        trigger_beta=1.0,
        recovery_beta=0.5,
        recovery_rounds=3,
    )
    before = pulse.step(9, False)
    trigger = pulse.step(10, True)
    recovery = [pulse.step(round_number, False) for round_number in range(11, 14)]
    resumed = pulse.step(14, False)

    assert before.phase == "ordinary"
    assert trigger.phase == "trigger_hold"
    assert trigger.override_beta == 1.0
    assert trigger.remaining_before == 0
    assert trigger.remaining_after == 3
    assert [observation.phase for observation in recovery] == ["recovery_pulse"] * 3
    assert [observation.override_beta for observation in recovery] == [0.5] * 3
    assert [observation.remaining_after for observation in recovery] == [2, 1, 0]
    assert resumed.phase == "ordinary"
    assert resumed.override_beta is None
    assert pulse.activation_count == 1


def test_retrigger_restarts_auditable_duration() -> None:
    pulse = r2c_v8.DeploymentRecoveryPulse(recovery_beta=0.25, recovery_rounds=2)
    pulse.step(20, True)
    first = pulse.step(21, False)
    retrigger = pulse.step(22, True)
    second = pulse.step(23, False)
    third = pulse.step(24, False)
    assert first.remaining_after == 1
    assert retrigger.phase == "trigger_hold"
    assert retrigger.remaining_before == 1
    assert retrigger.remaining_after == 2
    assert retrigger.activation_count == 2
    assert second.recovery_applied and third.recovery_applied
    assert third.remaining_after == 0


def test_pulse_audit_fields_disclose_forbidden_inputs() -> None:
    pulse = r2c_v8.DeploymentRecoveryPulse(recovery_beta=0.5, recovery_rounds=2)
    fields = pulse.step(500, True).audit_fields()
    assert fields["deployment_pulse_round"] == 500
    assert fields["deployment_pulse_phase"] == "trigger_hold"
    assert fields["deployment_pulse_state_server_only"] is True
    assert fields["deployment_pulse_labels_used"] is False
    assert fields["deployment_pulse_scenario_metadata_used"] is False


def test_v8_wrapper_preserves_v5_learning_path_and_relabels_protocol(monkeypatch) -> None:
    captured: dict[str, object] = {}
    result = SimpleNamespace(
        checkpoint_rows=[{"protocol_version": "old"}],
        certificate_row={"protocol_version": "old", "certificate_record_hash": "old"},
    )

    def fake_v5(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(r2c_v8, "run_r2c_v5_round", fake_v5)
    counts = np.arange(100, dtype=np.int64)
    returned = r2c_v8.run_r2c_v8_round(
        model=object(),
        trainer=object(),
        data=object(),
        trace=object(),
        round_number=3,
        available=np.arange(20),
        sample_counts=np.ones(100),
        learning_rate=0.1,
        payload_bytes=10,
        seed=7,
        config={
            "r2c_v5_history_mix": 0.45,
            "r2c_v8_trigger_deployment_beta": 1.0,
            "r2c_v8_recovery_pulse_beta": 0.5,
            "r2c_v8_recovery_pulse_rounds": 5,
        },
        run_id="test",
        round_start_model_hash="0" * 64,
        full_logging=True,
        selection_history_counts=counts,
    )
    assert returned is result
    assert np.array_equal(captured["selection_history_counts"], counts)
    assert result.checkpoint_rows[0]["protocol_version"] == r2c_v8.PROTOCOL_VERSION
    assert result.checkpoint_rows[0]["selection_protocol_version"] == r2c_v8.V5_PROTOCOL_VERSION
    assert result.certificate_row["deployment_rule"] == r2c_v8.DEPLOYMENT_RULE
    assert result.certificate_row["configured_recovery_pulse_beta"] == 0.5
    assert result.certificate_row["configured_recovery_pulse_rounds"] == 5
    assert result.certificate_row["certificate_record_hash"] != "old"
