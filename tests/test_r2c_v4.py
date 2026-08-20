from __future__ import annotations

import pytest
import torch

from r2c_baselines.r2c_v4 import (
    build_deployment_models,
    update_deployment_model,
    validated_deployment_betas,
)


def test_validated_deployment_betas_sorts_and_rejects_invalid_values() -> None:
    assert validated_deployment_betas([0.95, 0.8, 0.9]) == (0.8, 0.9, 0.95)
    with pytest.raises(ValueError):
        validated_deployment_betas([])
    with pytest.raises(ValueError):
        validated_deployment_betas([0.8, 0.8])
    with pytest.raises(ValueError):
        validated_deployment_betas([1.0])


def test_deployment_ema_is_server_only_and_exact() -> None:
    fast = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        fast.weight.fill_(2.0)
    deployment = build_deployment_models(fast, (0.9,))[0.9]
    with torch.no_grad():
        fast.weight.fill_(4.0)
    update_deployment_model(deployment, fast, 0.75)
    assert torch.equal(fast.weight, torch.full_like(fast.weight, 4.0))
    assert torch.allclose(deployment.weight, torch.full_like(deployment.weight, 2.5))
