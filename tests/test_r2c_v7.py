from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from r2c_baselines import r2c_v7
from r2c_baselines.run import _apply_effective_deployment_update


def test_trigger_deployment_beta_validation() -> None:
    assert r2c_v7.validated_trigger_deployment_beta({}) == 1.0
    assert r2c_v7.validated_trigger_deployment_beta(
        {"r2c_v7_trigger_deployment_beta": 0.75}
    ) == 0.75
    for value in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            r2c_v7.validated_trigger_deployment_beta(
                {"r2c_v7_trigger_deployment_beta": value}
            )


def test_beta_one_quarantine_preserves_all_parameters_and_buffers() -> None:
    deployment = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.BatchNorm1d(4),
    )
    fast = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.BatchNorm1d(4),
    )
    with torch.no_grad():
        for value in deployment.parameters():
            value.fill_(2.0)
        for value in fast.parameters():
            value.fill_(7.0)
        deployment[1].num_batches_tracked.fill_(11)
        fast[1].num_batches_tracked.fill_(29)
    before = {name: value.detach().clone() for name, value in deployment.state_dict().items()}

    _apply_effective_deployment_update(deployment, fast, 1.0)

    assert all(
        torch.equal(value, deployment.state_dict()[name])
        for name, value in before.items()
    )


def test_regular_beta_still_updates_deployment_model() -> None:
    deployment = torch.nn.Linear(2, 1, bias=False)
    fast = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        deployment.weight.fill_(2.0)
        fast.weight.fill_(6.0)
    _apply_effective_deployment_update(deployment, fast, 0.75)
    assert torch.allclose(deployment.weight, torch.full_like(deployment.weight, 3.0))


def test_v7_wrapper_preserves_v5_learning_path_and_relabels_protocol(monkeypatch) -> None:
    captured: dict[str, object] = {}
    result = SimpleNamespace(
        checkpoint_rows=[{"protocol_version": "old"}],
        certificate_row={"protocol_version": "old", "certificate_record_hash": "old"},
    )

    def fake_v5(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(r2c_v7, "run_r2c_v5_round", fake_v5)
    counts = np.arange(100, dtype=np.int64)
    returned = r2c_v7.run_r2c_v7_round(
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
            "r2c_v7_trigger_deployment_beta": 1.0,
        },
        run_id="test",
        round_start_model_hash="0" * 64,
        full_logging=True,
        selection_history_counts=counts,
    )
    assert returned is result
    assert np.array_equal(captured["selection_history_counts"], counts)
    assert result.checkpoint_rows[0]["protocol_version"] == r2c_v7.PROTOCOL_VERSION
    assert result.checkpoint_rows[0]["selection_protocol_version"] == r2c_v7.V5_PROTOCOL_VERSION
    assert result.certificate_row["deployment_rule"] == r2c_v7.DEPLOYMENT_RULE
    assert result.certificate_row["configured_trigger_deployment_beta"] == 1.0
    assert result.certificate_row["certificate_record_hash"] != "old"
