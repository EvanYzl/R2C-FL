from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from r2c_baselines import r2c_v14
from r2c_baselines.run_v14 import _apply_effective_deployment_update
from r2c_baselines.training import model_state_hash
from r2c_baselines.utils import sha256_file


class ToyModel(torch.nn.Module):
    def __init__(self, value: float, counter: int = 0) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((2, 2), float(value)))
        self.register_buffer("counter", torch.tensor(int(counter), dtype=torch.int64))


def set_toy(model: ToyModel, value: float, counter: int) -> None:
    with torch.no_grad():
        model.weight.fill_(float(value))
        model.counter.fill_(int(counter))


def test_candidate_grid_and_frozen_constants_are_closed() -> None:
    assert list(r2c_v14.CANDIDATES) == [
        "CMTR-B950-R20",
        "CMTR-B975-R10",
        "CMTR-B975-R20",
    ]
    assert r2c_v14.FAST_BETA == 0.9
    assert r2c_v14.WARMUP_ROUNDS == 200
    assert r2c_v14.CANDIDATES["CMTR-B950-R20"] == {
        "stable_beta": 0.95,
        "recovery_rounds": 20,
    }
    assert r2c_v14.CANDIDATES["CMTR-B975-R10"] == {
        "stable_beta": 0.975,
        "recovery_rounds": 10,
    }
    for value in ("", "CMTR-B900-R20", "adaptive", "CMTR-B975-R21"):
        with pytest.raises(ValueError):
            r2c_v14.CausalMultiTimescaleRouter.from_config(
                {"r2c_v14_candidate_id": value}
            )
    for value in (0.89, 0.91, float("nan")):
        with pytest.raises(ValueError):
            r2c_v14.CausalMultiTimescaleRouter.from_config(
                {"r2c_v14_fast_beta": value}
            )
    for value in (199, 201, True, 200.5):
        with pytest.raises(ValueError):
            r2c_v14.CausalMultiTimescaleRouter.from_config(
                {"r2c_v14_warmup_rounds": value}
            )


def test_warmup_boundary_trigger_recovery_and_persistent_stable_route() -> None:
    router = r2c_v14.CausalMultiTimescaleRouter(
        candidate_id="CMTR-B975-R20"
    )
    for round_number in range(1, 201):
        observation = router.step(round_number, False)
        assert observation.phase == "warmup_fast"
        assert observation.selected_role == "fast"
        assert observation.selected_beta == 0.9
        assert not observation.response_applied
    stable = router.step(201, False)
    assert stable.phase == "stable"
    assert stable.selected_beta == 0.975
    trigger = router.step(500, True)
    assert trigger.phase == "trigger_hold"
    assert trigger.hold_applied
    assert trigger.selected_role == "stable"
    assert trigger.fast_update_beta == 1.0
    assert trigger.stable_update_beta == 1.0
    assert trigger.remaining_after == 20
    for index, round_number in enumerate(range(501, 521), start=1):
        recovery = router.step(round_number, False)
        assert recovery.phase == "recovery_fast"
        assert recovery.recovery_fast_applied
        assert recovery.selected_beta == 0.9
        assert recovery.remaining_after == 20 - index
    terminal = router.step(521, False)
    assert terminal.phase == "stable"
    assert terminal.selected_beta == 0.975
    assert terminal.stable_route_applied
    assert not terminal.raw_global_deployment_used


def test_two_ema_states_update_independently_and_trigger_holds_both() -> None:
    router = r2c_v14.CausalMultiTimescaleRouter(
        candidate_id="CMTR-B975-R10"
    )
    fast = ToyModel(0.0, 0)
    stable = deepcopy(fast)
    current = ToyModel(1.0, 1)
    first = router.step(1, False)
    _apply_effective_deployment_update(
        fast, current, first.update_beta_for(0.9)
    )
    _apply_effective_deployment_update(
        stable, current, first.update_beta_for(0.975)
    )
    assert torch.allclose(fast.weight, torch.full_like(fast.weight, 0.1))
    assert torch.allclose(stable.weight, torch.full_like(stable.weight, 0.025))
    fast_before = model_state_hash(fast)
    stable_before = model_state_hash(stable)
    set_toy(current, 9.0, 9)
    trigger = router.step(2, True)
    _apply_effective_deployment_update(
        fast, current, trigger.update_beta_for(0.9)
    )
    _apply_effective_deployment_update(
        stable, current, trigger.update_beta_for(0.975)
    )
    assert model_state_hash(fast) == fast_before
    assert model_state_hash(stable) == stable_before
    recovery = router.step(3, False)
    _apply_effective_deployment_update(
        fast, current, recovery.update_beta_for(0.9)
    )
    _apply_effective_deployment_update(
        stable, current, recovery.update_beta_for(0.975)
    )
    assert model_state_hash(fast) != fast_before
    assert model_state_hash(stable) != stable_before
    assert recovery.selected_role == "fast"
    assert model_state_hash(fast) != model_state_hash(current)
    assert model_state_hash(stable) != model_state_hash(current)


def test_repeated_trigger_resets_recovery_without_future_inputs() -> None:
    router = r2c_v14.CausalMultiTimescaleRouter(
        candidate_id="CMTR-B975-R10"
    )
    router.step(300, True)
    for round_number in range(301, 305):
        router.step(round_number, False)
    retrigger = router.step(305, True)
    assert retrigger.remaining_before == 6
    assert retrigger.remaining_after == 10
    assert retrigger.activation_count == 2
    assert retrigger.labels_used is False
    assert retrigger.validation_predictions_used is False
    assert retrigger.test_predictions_used is False
    assert retrigger.scenario_metadata_used is False
    assert retrigger.event_round_used is False
    assert retrigger.future_trace_used is False


def test_mid_recovery_state_round_trip_is_exact() -> None:
    first = r2c_v14.CausalMultiTimescaleRouter(
        candidate_id="CMTR-B975-R20"
    )
    first.step(500, True)
    first.step(501, False)
    first.step(502, False)
    serialized = first.state_dict()
    second = r2c_v14.CausalMultiTimescaleRouter(
        candidate_id="CMTR-B975-R20"
    )
    second.load_state_dict(serialized)
    assert first.step(503, False).audit_fields() == second.step(
        503, False
    ).audit_fields()
    with pytest.raises(ValueError):
        r2c_v14.CausalMultiTimescaleRouter(
            candidate_id="CMTR-B975-R10"
        ).load_state_dict(serialized)


def test_audit_fields_disclose_route_cost_and_forbidden_inputs() -> None:
    router = r2c_v14.CausalMultiTimescaleRouter(
        candidate_id="CMTR-B950-R20"
    )
    fields = router.step(500, True).audit_fields()
    assert fields["deployment_cmtr_round"] == 500
    assert fields["deployment_cmtr_phase"] == "trigger_hold"
    assert fields["deployment_cmtr_configured_candidate_id"] == "CMTR-B950-R20"
    assert fields["deployment_cmtr_deployment_state_count"] == 2
    assert fields["deployment_cmtr_state_server_only"] is True
    assert fields["deployment_cmtr_labels_used"] is False
    assert fields["deployment_cmtr_validation_predictions_used"] is False
    assert fields["deployment_cmtr_test_predictions_used"] is False
    assert fields["deployment_cmtr_scenario_metadata_used"] is False
    assert fields["deployment_cmtr_event_round_used"] is False
    assert fields["deployment_cmtr_future_trace_used"] is False
    assert fields["deployment_cmtr_raw_global_deployment_used"] is False


def test_v14_wrapper_preserves_v5_learning_path_and_relabels_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    result = SimpleNamespace(
        checkpoint_rows=[{"protocol_version": "old"}],
        certificate_row={"protocol_version": "old", "certificate_record_hash": "old"},
    )

    def fake_v5(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(r2c_v14, "run_r2c_v5_round", fake_v5)
    counts = np.arange(100, dtype=np.int64)
    returned = r2c_v14.run_r2c_v14_round(
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
        config={"r2c_v14_candidate_id": "CMTR-B975-R10"},
        run_id="test",
        round_start_model_hash="0" * 64,
        full_logging=True,
        selection_history_counts=counts,
    )
    assert returned is result
    assert np.array_equal(captured["selection_history_counts"], counts)
    assert result.checkpoint_rows[0]["protocol_version"] == r2c_v14.PROTOCOL_VERSION
    assert result.checkpoint_rows[0]["selection_protocol_version"] == r2c_v14.V5_PROTOCOL_VERSION
    assert result.certificate_row["deployment_rule"] == r2c_v14.DEPLOYMENT_RULE
    assert result.certificate_row["configured_cmtr_fast_beta"] == 0.9
    assert result.certificate_row["configured_cmtr_stable_beta"] == 0.975
    assert result.certificate_row["configured_cmtr_recovery_rounds"] == 10
    assert result.certificate_row["certificate_record_hash"] != "old"


def test_frozen_v13_unified_runner_was_not_modified() -> None:
    path = Path(__file__).resolve().parents[1] / "r2c_baselines" / "run.py"
    assert sha256_file(path).upper() == (
        "6E534BE201B37CFCEC8A150250932481FC7EAA860E06D72CA090DE50D2E17B4D"
    )
