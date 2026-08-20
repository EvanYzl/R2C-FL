from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
# The public release is a standalone repository, so data, figures, manifests,
# queues, and run artifacts resolve beneath the repository root.
PROJECT_ROOT = EXPERIMENT_ROOT
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
PLOT_ROOT = PROJECT_ROOT / "figures" / "main_text_plot_data"
RUN_ROOT = EXPERIMENT_ROOT / "runs"
ASSET_ROOT = EXPERIMENT_ROOT / "frozen_assets"
QUEUE_ROOT = EXPERIMENT_ROOT / "queue"
LOG_ROOT = EXPERIMENT_ROOT / "logs"

FORMAL_SEED = 20260811
DEV_SEED = 20260810
SCHEMA_VERSION = "2.1.0"
NUM_CLIENTS = 100
SELECTED_K = 10
CANDIDATE_M = 20


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    name: str
    folder: str
    model_name: str
    num_classes: int
    batch_size: int
    local_steps: int
    round_budget: int
    base_lr: float
    weight_decay: float
    channels: int


DATASETS: dict[str, DatasetSpec] = {
    "D1": DatasetSpec("D1", "Fashion-MNIST", "fashion_mnist", "CNN2_GN", 10, 64, 10, 500, 0.03, 0.0, 1),
    "D2": DatasetSpec("D2", "CIFAR-10", "cifar10", "CNN4_GN", 10, 64, 10, 1000, 0.03, 5e-4, 3),
    "D3": DatasetSpec("D3", "SVHN", "svhn", "CNN4_GN", 10, 64, 10, 1000, 0.03, 5e-4, 3),
    "D4": DatasetSpec("D4", "CIFAR-100", "cifar100", "ResNet18_GN", 100, 32, 10, 1500, 0.03, 5e-4, 3),
}


SCENARIOS = ("S0", "S1", "S2", "S3", "S4")
BASELINES = ("FedAvg", "FedAU", "F3AST", "FedAWE", "PowerOfChoice", "Oort", "TiFL")
PROPOSED_METHOD = "R2C-FL"
METHODS = BASELINES + (PROPOSED_METHOD,)


# Exactly three preregistered choices per method and dataset.  Dataset-specific
# base learning rates are multiplied by lr_mult; method settings are never
# selected using formal test data.
SEARCH_VARIANTS: dict[str, tuple[dict[str, Any], ...]] = {
    "FedAvg": (
        {"lr_mult": 0.5},
        {"lr_mult": 1.0},
        {"lr_mult": 1.5},
    ),
    "FedAU": (
        {"lr_mult": 1.0, "fedau_k": 20},
        {"lr_mult": 1.0, "fedau_k": 50},
        {"lr_mult": 1.0, "fedau_k": 100},
    ),
    "F3AST": (
        {"lr_mult": 1.0, "f3ast_beta": 0.0005},
        {"lr_mult": 1.0, "f3ast_beta": 0.001},
        {"lr_mult": 1.0, "f3ast_beta": 0.005},
    ),
    "FedAWE": (
        {"lr_mult": 1.0, "server_lr": 0.5},
        {"lr_mult": 1.0, "server_lr": 1.0},
        {"lr_mult": 1.0, "server_lr": 1.5},
    ),
    "PowerOfChoice": (
        {"lr_mult": 1.0, "pow_d": 12},
        {"lr_mult": 1.0, "pow_d": 16},
        {"lr_mult": 1.0, "pow_d": 20},
    ),
    "Oort": (
        {"lr_mult": 1.0, "oort_exploration": 0.10},
        {"lr_mult": 1.0, "oort_exploration": 0.20},
        {"lr_mult": 1.0, "oort_exploration": 0.30},
    ),
    "TiFL": (
        {"lr_mult": 1.0, "tifl_temperature": 0.5},
        {"lr_mult": 1.0, "tifl_temperature": 1.0},
        {"lr_mult": 1.0, "tifl_temperature": 2.0},
    ),
}


# R2C-FL keeps the mechanism constants fixed and tunes only the local learning
# rate on the development seed.  The delta clip is frozen separately from the
# 95th percentile of complete local-update norms in disjoint norm-pilot runs.
R2C_COMMON_CONFIG: dict[str, Any] = {
    "r2c_lambda": 0.05,
    "r2c_delta": 0.05,
    "r2c_value_clip": 1.0,
    "r2c_scale_floor": 0.05,
    "r2c_temperature": 0.5,
    "r2c_bootstrap_replicates": 4096,
    "r2c_initial_drift_per_s": 0.01,
    "r2c_heldout_per_fold": 16,
    "r2c_eval_microbatch": 8,
    "r2c_completion_floor": 0.05,
    "r2c_checkpoint_backend": "cpu_ring_v1",
}

R2C_SEARCH_VARIANTS: tuple[dict[str, Any], ...] = tuple(
    {**R2C_COMMON_CONFIG, "lr_mult": lr_mult}
    for lr_mult in (0.5, 1.0, 1.5)
)


DEFAULT_METHOD_CONFIG: dict[str, dict[str, Any]] = {
    method: dict(variants[1]) for method, variants in SEARCH_VARIANTS.items()
}


def dataset_dict(dataset_id: str) -> dict[str, Any]:
    return asdict(DATASETS[dataset_id])
