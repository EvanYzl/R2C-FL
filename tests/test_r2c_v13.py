from __future__ import annotations

from copy import deepcopy
from math import sqrt
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from r2c_baselines import r2c_v13
from r2c_baselines.run import _apply_effective_deployment_update
from r2c_baselines.training import model_state_hash


class ToyModel(torch.nn.Module):
    def __init__(self, value: float, counter: int = 0) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((2, 2), float(value)))
        self.register_buffer("counter", torch.tensor(int(counter), dtype=torch.int64))


def set_toy(model: ToyModel, value: float, counter: int) -> None:
    with torch.no_grad():
        model.weight.fill_(float(value))
        model.counter.fill_(int(counter))


def test_schedule_set_is_closed_and_high_discrimination_grid_is_exact() -> None:
    assert set(r2c_v13.SCHEDULES) == {"DARE-L5", "DARE-C5", "DARE-L8"}
    assert [r2c_v13.schedule_lambda("DARE-L5", index) for index in range(1, 6)] == [
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    ]
    assert np.allclose(
        [r2c_v13.schedule_lambda("DARE-C5", index) for index in range(1, 6)],
        [sqrt(index / 5) for index in range(1, 6)],
        atol=0.0,
        rtol=0.0,
    )
    assert [r2c_v13.schedule_lambda("DARE-L8", index) for index in range(1, 9)] == [
        index / 8 for index in range(1, 9)
    ]
    for value in ("", "DARE-L6", "DARE-C8", "adaptive"):
        with pytest.raises(ValueError):
            r2c_v13.DualAnchorRecoveryEnvelope.from_config(
                {"r2c_v13_schedule_id": value}
            )
    for value in (0, -1, 1.5, True, 6):
        with pytest.raises(ValueError):
            r2c_v13.schedule_lambda("DARE-L5", value)


def test_no_trigger_is_state_free_and_does_not_touch_deployment() -> None:
    controller = r2c_v13.DualAnchorRecoveryEnvelope(schedule_id="DARE-L5")
    deployment = ToyModel(2.0, 2)
    current = ToyModel(9.0, 9)
    before = model_state_hash(deployment)
    observations = []
    for round_number in range(1, 8):
        observation = controller.step(round_number, False)
        observations.append(observation)
        if observation.response_applied:
            controller.apply(observation, deployment, current)
    assert model_state_hash(deployment) == before
    assert all(observation.phase == "ordinary" for observation in observations)
    assert all(observation.lambda_value is None for observation in observations)
    assert not any(observation.response_applied for observation in observations)
    assert not controller.activated
    assert controller.activation_count == 0
    assert controller.anchor_tensor_count == 0
    assert controller.anchor_bytes == 0


def test_no_trigger_runner_path_is_hash_identical_to_ordinary_ema() -> None:
    controller = r2c_v13.DualAnchorRecoveryEnvelope(schedule_id="DARE-L8")
    reference = ToyModel(1.0, 1)
    candidate = deepcopy(reference)
    current = ToyModel(2.0, 2)
    for round_number in range(1, 21):
        set_toy(current, 1.0 + round_number / 7.0, round_number)
        _apply_effective_deployment_update(reference, current, 0.95)
        observation = controller.step(round_number, False)
        assert not observation.response_applied
        _apply_effective_deployment_update(candidate, current, 0.95)
        assert model_state_hash(candidate) == model_state_hash(reference)
    assert controller.state_dict()["pre_anchor_state"] is None


def test_trigger_captures_one_anchor_then_interpolates_and_tracks_current_model() -> None:
    controller = r2c_v13.DualAnchorRecoveryEnvelope(schedule_id="DARE-L5")
    deployment = ToyModel(0.0, 0)
    current = ToyModel(10.0, 10)
    pre_hash = model_state_hash(deployment)
    post_hash = model_state_hash(current)

    trigger = controller.step(10, True)
    trigger = controller.apply(trigger, deployment, current)
    assert trigger.phase == "trigger_anchor_hold"
    assert trigger.lambda_value == 0.0
    assert trigger.equivalent_beta == 1.0
    assert trigger.pre_anchor_captured
    assert trigger.pre_anchor_hash == pre_hash
    assert trigger.post_anchor_hash == post_hash
    assert model_state_hash(deployment) == pre_hash
    assert controller.anchor_tensor_count == len(deployment.state_dict())
    assert controller.anchor_bytes > 0

    for index, round_number in enumerate(range(11, 16), start=1):
        set_toy(current, float(10 * (index + 1)), 10 * (index + 1))
        observation = controller.step(round_number, False)
        observation = controller.apply(observation, deployment, current)
        expected_lambda = index / 5
        assert observation.phase == "recovery_envelope"
        assert observation.envelope_applied
        assert observation.lambda_value == expected_lambda
        assert torch.allclose(
            deployment.weight,
            torch.full_like(deployment.weight, expected_lambda * 10 * (index + 1)),
            atol=0.0,
            rtol=0.0,
        )
        assert int(deployment.counter) == 10 * (index + 1)
    assert observation.pre_anchor_released
    assert controller.anchor_tensor_count == 0
    assert controller.anchor_bytes == 0

    set_toy(current, 77.0, 77)
    tracking = controller.step(16, False)
    tracking = controller.apply(tracking, deployment, current)
    assert tracking.phase == "post_shift_tracking"
    assert tracking.tracking_applied
    assert tracking.lambda_value == 1.0
    assert tracking.equivalent_beta == 0.0
    assert model_state_hash(deployment) == model_state_hash(current)


def test_mid_envelope_state_round_trip_is_exact() -> None:
    first = r2c_v13.DualAnchorRecoveryEnvelope(schedule_id="DARE-C5")
    deployment_a = ToyModel(1.0, 1)
    current = ToyModel(3.0, 3)
    first.apply(first.step(20, True), deployment_a, current)
    set_toy(current, 5.0, 5)
    first.apply(first.step(21, False), deployment_a, current)

    serialized = first.state_dict()
    second = r2c_v13.DualAnchorRecoveryEnvelope(schedule_id="DARE-C5")
    second.load_state_dict(serialized)
    deployment_b = deepcopy(deployment_a)
    set_toy(current, 7.0, 7)
    observation_a = first.apply(first.step(22, False), deployment_a, current)
    observation_b = second.apply(second.step(22, False), deployment_b, current)
    assert observation_a.audit_fields() == observation_b.audit_fields()
    assert model_state_hash(deployment_a) == model_state_hash(deployment_b)
    assert first.state_dict()["pre_anchor_hash"] == second.state_dict()["pre_anchor_hash"]
    with pytest.raises(ValueError):
        r2c_v13.DualAnchorRecoveryEnvelope(schedule_id="DARE-L8").load_state_dict(
            serialized
        )


def test_audit_fields_disclose_lineage_cost_and_forbidden_inputs() -> None:
    controller = r2c_v13.DualAnchorRecoveryEnvelope(schedule_id="DARE-L8")
    deployment = ToyModel(1.0, 1)
    current = ToyModel(2.0, 2)
    fields = controller.apply(
        controller.step(500, True), deployment, current
    ).audit_fields()
    assert fields["deployment_dare_round"] == 500
    assert fields["deployment_dare_phase"] == "trigger_anchor_hold"
    assert fields["deployment_dare_configured_schedule_id"] == "DARE-L8"
    assert fields["deployment_dare_anchor_tensor_count_after"] == len(
        deployment.state_dict()
    )
    assert fields["deployment_dare_anchor_bytes_after"] > 0
    assert fields["deployment_dare_state_server_only"] is True
    assert fields["deployment_dare_labels_used"] is False
    assert fields["deployment_dare_scenario_metadata_used"] is False
    assert fields["deployment_dare_future_trace_used"] is False


def test_v13_wrapper_preserves_v5_learning_path_and_relabels_protocol(monkeypatch) -> None:
    captured: dict[str, object] = {}
    result = SimpleNamespace(
        checkpoint_rows=[{"protocol_version": "old"}],
        certificate_row={"protocol_version": "old", "certificate_record_hash": "old"},
    )

    def fake_v5(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(r2c_v13, "run_r2c_v5_round", fake_v5)
    counts = np.arange(100, dtype=np.int64)
    returned = r2c_v13.run_r2c_v13_round(
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
            "r2c_v13_schedule_id": "DARE-C5",
        },
        run_id="test",
        round_start_model_hash="0" * 64,
        full_logging=True,
        selection_history_counts=counts,
    )
    assert returned is result
    assert np.array_equal(captured["selection_history_counts"], counts)
    assert result.checkpoint_rows[0]["protocol_version"] == r2c_v13.PROTOCOL_VERSION
    assert result.checkpoint_rows[0]["selection_protocol_version"] == r2c_v13.V5_PROTOCOL_VERSION
    assert result.certificate_row["deployment_rule"] == r2c_v13.DEPLOYMENT_RULE
    assert result.certificate_row["configured_dare_schedule_id"] == "DARE-C5"
    assert result.certificate_row["configured_dare_recovery_rounds"] == 5
    assert result.certificate_row["certificate_record_hash"] != "old"
