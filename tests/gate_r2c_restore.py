from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from r2c_baselines.config import DATASETS, DEV_SEED
from r2c_baselines.data import FederatedData
from r2c_baselines.models import build_model
from r2c_baselines.run import _initialization_seed, configure_runtime
from r2c_baselines.training import LocalTrainer


def main() -> None:
    """Gate CPU checkpoint restoration and deterministic prefix reuse."""

    dataset_id = "D1"
    spec = DATASETS[dataset_id]
    configure_runtime(_initialization_seed(dataset_id, DEV_SEED))
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    data = FederatedData.load(dataset_id, DEV_SEED)
    model = build_model(spec.model_name, spec.num_classes).to(device).train()
    trainer = LocalTrainer(
        model,
        data,
        device,
        spec.batch_size,
        spec.local_steps,
        spec.weight_decay,
        max_parallel_clients=1,
    )
    client_ids = np.asarray([0, 1], dtype=np.int64)
    data_states = ["pre", "pre"]
    params = trainer.stacked_global(len(client_ids))
    trainer.r2c_train_step(
        params, client_ids, 1, data_states, 0, spec.base_lr
    )

    checkpoint = {name: value.detach().cpu().clone() for name, value in params.items()}
    restored_a = {name: value.to(device).clone() for name, value in checkpoint.items()}
    restored_b = {name: value.to(device).clone() for name, value in checkpoint.items()}
    for name in checkpoint:
        if not torch.equal(checkpoint[name], restored_a[name].cpu()):
            raise AssertionError(f"CPU round-trip mismatch before resume: {name}")

    trainer.r2c_train_step(
        restored_a, client_ids, 1, data_states, 1, spec.base_lr
    )
    trainer.r2c_train_step(
        restored_b, client_ids, 1, data_states, 1, spec.base_lr
    )
    mismatches = [
        name for name in restored_a if not torch.equal(restored_a[name], restored_b[name])
    ]
    if mismatches:
        raise AssertionError(f"Resumed tensors are not bitwise equal: {mismatches}")

    losses_a, _, _ = trainer.r2c_heldout_losses(
        restored_a, client_ids, data_states, 0, 16, parallel_clients=2
    )
    losses_b, _, _ = trainer.r2c_heldout_losses(
        restored_b, client_ids, data_states, 0, 16, parallel_clients=2
    )
    if not np.array_equal(losses_a, losses_b):
        raise AssertionError("Held-out losses differ after identical resume")

    print(
        json.dumps(
            {
                "status": "passed",
                "dataset_id": dataset_id,
                "clients": client_ids.tolist(),
                "checkpoint_tensors": len(checkpoint),
                "resume_step": 1,
                "bitwise_equal": True,
                "heldout_equal": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
