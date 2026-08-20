from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import socket
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import psutil
import torch

from . import SCHEMA_VERSION
from .config import (
    CANDIDATE_M,
    DATASETS,
    EXPERIMENT_ROOT,
    NUM_CLIENTS,
    PROPOSED_METHOD,
    RUN_ROOT,
    SELECTED_K,
)
from .data import FederatedData, partition_asset_path, partition_meta_path
from .logging_io import ChunkedTableWriter, read_chunked_table
from .methods import BaselineAdapter
from .metrics import event_window_role, participation_jfi, recovery_auc20, worst10_participation
from .models import build_model, model_payload_bytes
from .r2c import run_r2c_round
from .r2c_v2 import PROTOCOL_VERSION as R2C_V2_PROTOCOL_VERSION, run_r2c_v2_round
from .r2c_v3 import PROTOCOL_VERSION as R2C_V3_PROTOCOL_VERSION, run_r2c_v3_round
from .r2c_v4 import (
    PROTOCOL_VERSION as R2C_V4_PROTOCOL_VERSION,
    build_deployment_models,
    run_r2c_v4_round,
    update_deployment_model,
    validated_deployment_betas,
)
from .r2c_v5 import PROTOCOL_VERSION as R2C_V5_PROTOCOL_VERSION, run_r2c_v5_round
from .r2c_v6 import (
    PROTOCOL_VERSION as R2C_V6_PROTOCOL_VERSION,
    TelemetryShiftDetector,
    attach_telemetry_observation,
    run_r2c_v6_round,
)
from .r2c_v7 import (
    PROTOCOL_VERSION as R2C_V7_PROTOCOL_VERSION,
    run_r2c_v7_round,
    validated_trigger_deployment_beta,
)
from .r2c_v8 import (
    PROTOCOL_VERSION as R2C_V8_PROTOCOL_VERSION,
    DeploymentRecoveryPulse,
    RecoveryPulseObservation,
    run_r2c_v8_round,
    validated_trigger_deployment_beta as validated_v8_trigger_deployment_beta,
)
from .r2c_v13 import (
    PROTOCOL_VERSION as R2C_V13_PROTOCOL_VERSION,
    DAREObservation,
    DualAnchorRecoveryEnvelope,
    run_r2c_v13_round,
)
from .traces import Trace, trace_asset_path, trace_meta_path
from .training import LocalTrainer, evaluate_model, model_state_hash
from .utils import (
    atomic_json,
    atomic_parquet,
    canonical_json,
    config_hash,
    cpu_model,
    environment_snapshot,
    git_commit,
    gpu_query,
    sha256_text,
    utc_now,
)


UPSTREAM_COMMITS = {
    "FedAvg": "unified-local-20260811",
    "FedAU": "612814c9791a1e41a8a1b123616af52377a224b9",
    "F3AST": "27b4e33adffea94a3fe53a5600fe5498f4cd3d5d",
    "FedAWE": "e0b8538adc95dcbcb63574729594ad4605df969b",
    "PowerOfChoice": "paper-adapter-v1",
    "Oort": "05a3aa1677a10f8e621055b1626ef82e73d09759",
    "TiFL": "paper-adapter-v1",
    "R2C-FL": "local-race-to-commit-v1",
}


def _apply_effective_deployment_update(
    deployment_model: torch.nn.Module,
    fast_model: torch.nn.Module,
    effective_beta: float,
) -> None:
    """Apply a regular EMA update or a protocol-requested beta=1 hold."""

    beta = float(effective_beta)
    if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("Effective deployment beta must be finite and lie in [0, 1]")
    if beta == 1.0:
        return
    update_deployment_model(deployment_model, fast_model, beta)


def configure_runtime(seed: int) -> None:
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)


def _initialization_seed(dataset_id: str, seed: int) -> int:
    return int(seed + int(dataset_id[1:]) * 100_003)


def _event_id(dataset_id: str, scenario_id: str, event_round: int | None) -> str | None:
    if event_round is None:
        return None
    event_type = {
        "S1": "availability_phase",
        "S2": "label_or_mixture_swap",
        "S3": "compute_reversal",
        "S4": "compound_boundary",
    }[scenario_id]
    return f"{dataset_id}-{scenario_id}-{event_type}-r{event_round}"


def _learning_rate(base_lr: float, multiplier: float, round_number: int, rounds: int) -> float:
    progress = (round_number - 1) / max(1, rounds - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr * multiplier * (0.10 + 0.90 * cosine))


def _aggregate_standard(
    model: torch.nn.Module,
    result: Any,
    selected: np.ndarray,
    eligible: np.ndarray,
    coefficients: np.ndarray,
) -> None:
    if len(eligible) == 0:
        return
    positions = {int(client): index for index, client in enumerate(selected)}
    indices = torch.tensor([positions[int(client)] for client in eligible], device=next(model.parameters()).device)
    coeff = torch.tensor(coefficients, device=indices.device, dtype=torch.float32)
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, global_parameter in named.items():
            final = result.final_params[name].index_select(0, indices)
            start = result.start_params[name].index_select(0, indices)
            view = coeff.view(len(coeff), *([1] * global_parameter.ndim))
            update = ((final - start) * view).sum(dim=0)
            global_parameter.add_(update)


def _aggregate_fedawe(
    model: torch.nn.Module,
    result: Any,
    selected: np.ndarray,
    eligible: np.ndarray,
    gaps: np.ndarray,
    server_lr: float,
) -> None:
    if len(eligible) == 0:
        return
    positions = {int(client): index for index, client in enumerate(selected)}
    indices = torch.tensor([positions[int(client)] for client in eligible], device=next(model.parameters()).device)
    gap = torch.tensor([float(gaps[int(client)]) for client in eligible], device=indices.device)
    base: dict[str, torch.Tensor] = {}
    echo: dict[str, torch.Tensor] = {}
    total_norm_sq = torch.zeros((), device=indices.device, dtype=torch.float64)
    for name in result.final_params:
        final = result.final_params[name].index_select(0, indices)
        start = result.start_params[name].index_select(0, indices)
        base[name] = start.mean(dim=0)
        view = gap.view(len(gap), *([1] * (start.ndim - 1)))
        echo[name] = ((final - start) * view).mean(dim=0)
        total_norm_sq += torch.square(echo[name].double()).sum()
    total_norm = float(torch.sqrt(total_norm_sq).item())
    scale = min(1.0, 0.5 / max(1e-12, total_norm))
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, parameter in named.items():
            parameter.copy_(base[name] + float(server_lr) * scale * echo[name])


def _make_local_bank(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().unsqueeze(0).repeat(NUM_CLIENTS, *([1] * parameter.ndim))
        for name, parameter in model.named_parameters()
    }


def _bank_fetch(
    bank: dict[str, torch.Tensor], client_ids: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    index = torch.from_numpy(np.asarray(client_ids, dtype=np.int64))
    return {name: value.index_select(0, index).to(device, non_blocking=True) for name, value in bank.items()}


def _bank_reset(
    bank: dict[str, torch.Tensor], client_ids: np.ndarray, model: torch.nn.Module
) -> None:
    if len(client_ids) == 0:
        return
    index = torch.from_numpy(np.asarray(client_ids, dtype=np.int64))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = parameter.detach().cpu().unsqueeze(0).expand(len(client_ids), *parameter.shape)
            bank[name].index_copy_(0, index, value)


def _per_class_rounds(rounds: int, event_round: int | None) -> set[int]:
    values = {1, rounds}
    if event_round is not None:
        values.update(
            event_round + offset
            for offset in (-1, 0, 10, 25, 50, 100)
            if 1 <= event_round + offset <= rounds
        )
    return values


def _run_manifest_base(
    job: dict[str, Any],
    data: FederatedData,
    trace: Trace,
    model: torch.nn.Module,
    environment: dict[str, Any],
    start_utc: str,
) -> dict[str, Any]:
    spec = DATASETS[job["dataset_id"]]
    gpu = gpu_query()
    method_config = dict(job["method_config"])
    full_config = dict(job)
    full_config["dataset_spec"] = spec.__dict__
    adapter_commit = git_commit(EXPERIMENT_ROOT / "r2c_baselines")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": job["run_id"],
        "block_id": job["block_id"],
        "source_kind": "REPRODUCED" if job["mode"] == "formal" else job["mode"].upper(),
        "method_id": job["method_id"],
        "dataset_id": job["dataset_id"],
        "scenario_id": job["scenario_id"],
        "severity": "base",
        "seed": int(job["seed"]),
        "partition_seed": int(job["partition_seed"]),
        "trace_seed": int(job["trace_seed"]),
        "git_commit": git_commit(EXPERIMENT_ROOT),
        "config_hash": config_hash(full_config),
        "environment_hash": config_hash(environment),
        "partition_hash": data.partition_hash,
        "heldout_hash": data.heldout_hash if job["method_id"] == PROPOSED_METHOD else None,
        "trace_hash": trace.trace_hash,
        "initial_model_hash": model_state_hash(model),
        "upstream_commit": (
            str(method_config.get("r2c_protocol_version"))
            if job["method_id"] == PROPOSED_METHOD
            and method_config.get("r2c_protocol_version")
            in {
                R2C_V2_PROTOCOL_VERSION,
                R2C_V3_PROTOCOL_VERSION,
                R2C_V4_PROTOCOL_VERSION,
                R2C_V5_PROTOCOL_VERSION,
                R2C_V6_PROTOCOL_VERSION,
                R2C_V7_PROTOCOL_VERSION,
                R2C_V8_PROTOCOL_VERSION,
                R2C_V13_PROTOCOL_VERSION,
            }
            else UPSTREAM_COMMITS[job["method_id"]]
        ),
        "adapter_commit": adapter_commit,
        "model_name": spec.model_name,
        "optimizer_name": "SGD(momentum=0)",
        "num_clients": NUM_CLIENTS,
        "candidate_m": CANDIDATE_M,
        "selected_k": SELECTED_K,
        "batch_size": spec.batch_size,
        "local_steps": spec.local_steps,
        "round_budget": int(job["rounds"]),
        "target_accuracy": job.get("target_accuracy"),
        "precision_mode": "fp32_tf32_disabled",
        "client_parallelism": int(job.get("client_microbatch", 1)),
        "host_id": socket.gethostname(),
        "gpu_index": 0,
        "gpu_uuid": gpu.get("uuid"),
        "gpu_model": gpu.get("name", environment.get("gpu_model")),
        "gpu_total_mib": gpu.get("memory_total_mib", environment.get("gpu_total_mib")),
        "cpu_model": cpu_model(),
        "ram_total_mib": int(psutil.virtual_memory().total / 2**20),
        "scratch_device": str(RUN_ROOT.drive),
        "queue_utc": job.get("queue_utc"),
        "start_utc": start_utc,
        "end_utc": None,
        "status": "running",
        "failure_reason": None,
        "retry_of_run_id": job.get("retry_of_run_id"),
        "method_parameters_json": canonical_json(method_config),
        "checkpoint_serialization_version": (
            "r2c-telemetry-dare-single-checkpoint-v13"
            if job["method_id"] == PROPOSED_METHOD
            and method_config.get("r2c_protocol_version") == R2C_V13_PROTOCOL_VERSION
            else
            "r2c-telemetry-recovery-pulse-single-checkpoint-v8"
            if job["method_id"] == PROPOSED_METHOD
            and method_config.get("r2c_protocol_version") == R2C_V8_PROTOCOL_VERSION
            else
            "r2c-telemetry-quarantine-single-checkpoint-v7"
            if job["method_id"] == PROPOSED_METHOD
            and method_config.get("r2c_protocol_version") == R2C_V7_PROTOCOL_VERSION
            else
            "r2c-telemetry-sync-single-checkpoint-v6"
            if job["method_id"] == PROPOSED_METHOD
            and method_config.get("r2c_protocol_version") == R2C_V6_PROTOCOL_VERSION
            else
            "r2c-history-balanced-single-checkpoint-v5"
            if job["method_id"] == PROPOSED_METHOD
            and method_config.get("r2c_protocol_version") == R2C_V5_PROTOCOL_VERSION
            else
            "r2c-anchor-single-checkpoint-v4"
            if job["method_id"] == PROPOSED_METHOD
            and method_config.get("r2c_protocol_version") == R2C_V4_PROTOCOL_VERSION
            else "r2c-anchor-single-checkpoint-v3"
            if job["method_id"] == PROPOSED_METHOD
            and method_config.get("r2c_protocol_version") == R2C_V3_PROTOCOL_VERSION
            else "r2c-anchor-single-checkpoint-v2"
            if job["method_id"] == PROPOSED_METHOD
            and method_config.get("r2c_protocol_version") == R2C_V2_PROTOCOL_VERSION
            else ("r2c-cpu-ring-v1" if job["method_id"] == PROPOSED_METHOD else None)
        ),
        "hardware_plan_deviation": "actual_single_RTX5080_vs_planned_7xRTX2080Ti",
    }


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    required = {
        "run_id",
        "block_id",
        "mode",
        "method_id",
        "dataset_id",
        "scenario_id",
        "seed",
        "partition_seed",
        "trace_seed",
        "rounds",
        "method_config",
        "evaluation_split",
    }
    missing = required - set(job)
    if missing:
        raise ValueError(f"Missing job fields: {sorted(missing)}")
    run_dir = RUN_ROOT / str(job["run_id"])
    success_path = run_dir / "_SUCCESS.json"
    if success_path.exists():
        return json.loads(success_path.read_text(encoding="utf-8"))
    if run_dir.exists():
        raise RuntimeError(f"Partial run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(run_dir / "job.json", job)

    start_utc = utc_now()
    spec = DATASETS[job["dataset_id"]]
    rounds = int(job["rounds"])
    seed = int(job["seed"])
    init_seed = _initialization_seed(job["dataset_id"], seed)
    configure_runtime(init_seed)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen execution protocol")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    data = FederatedData.load(job["dataset_id"], int(job["partition_seed"]))
    trace = Trace.load(job["dataset_id"], job["scenario_id"], int(job["trace_seed"]), rounds)
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        partition_asset_path(job["dataset_id"], int(job["partition_seed"])),
        input_dir / "partition.npz",
    )
    shutil.copy2(
        partition_meta_path(job["dataset_id"], int(job["partition_seed"])),
        input_dir / "partition.json",
    )
    shutil.copy2(
        trace_asset_path(job["dataset_id"], job["scenario_id"], int(job["trace_seed"]), rounds),
        input_dir / "trace.npz",
    )
    shutil.copy2(
        trace_meta_path(job["dataset_id"], job["scenario_id"], int(job["trace_seed"]), rounds),
        input_dir / "trace.json",
    )
    model = build_model(spec.model_name, spec.num_classes).to(device)
    model.train()
    environment = environment_snapshot()
    manifest = _run_manifest_base(job, data, trace, model, environment, start_utc)
    atomic_parquet(run_dir / "run_manifest.parquet", pd.DataFrame([manifest]))
    atomic_json(run_dir / "environment.json", environment)
    current_model_hash = str(manifest["initial_model_hash"])

    sample_counts = np.asarray([data.client_size(i, "pre") for i in range(NUM_CLIENTS)], dtype=np.int64)
    adapter = (
        None
        if job["method_id"] == PROPOSED_METHOD
        else BaselineAdapter(
            job["method_id"], sample_counts, seed, rounds, dict(job["method_config"])
        )
    )
    trainer = LocalTrainer(
        model,
        data,
        device,
        spec.batch_size,
        spec.local_steps,
        spec.weight_decay,
        max_parallel_clients=int(job.get("client_microbatch", 1)),
    )
    payload_bytes = model_payload_bytes(model)
    full_logging = bool(job.get("full_logging", job["mode"] == "formal"))
    method_config = dict(job["method_config"])
    protocol_version = method_config.get("r2c_protocol_version")
    no_drift_quarantine = bool(
        method_config.get("r2c_ablation_no_drift_quarantine", False)
    )
    no_reusable_prefix = bool(
        method_config.get("r2c_ablation_no_reusable_prefix", False)
    )
    deployment_betas: tuple[float, ...] = ()
    primary_deployment_beta: float | None = None
    deployment_models: dict[float, torch.nn.Module] = {}
    telemetry_shift_detector: TelemetryShiftDetector | None = None
    deployment_recovery_pulse: DeploymentRecoveryPulse | None = None
    deployment_dare: DualAnchorRecoveryEnvelope | None = None
    if job["method_id"] == PROPOSED_METHOD and protocol_version in {
        R2C_V4_PROTOCOL_VERSION,
        R2C_V5_PROTOCOL_VERSION,
        R2C_V6_PROTOCOL_VERSION,
        R2C_V7_PROTOCOL_VERSION,
        R2C_V8_PROTOCOL_VERSION,
        R2C_V13_PROTOCOL_VERSION,
    }:
        deployment_betas = validated_deployment_betas(
            method_config.get("r2c_v4_deployment_ema_betas", [0.8, 0.9, 0.95])
        )
        primary_deployment_beta = float(
            method_config.get("r2c_v4_primary_deployment_beta", deployment_betas[0])
        )
        if primary_deployment_beta not in deployment_betas:
            raise ValueError("Primary deployment beta must be one of the configured candidates")
        if protocol_version == R2C_V13_PROTOCOL_VERSION and len(deployment_betas) != 1:
            raise ValueError("R2C-v13 requires exactly one deployment state")
        deployment_models = build_deployment_models(model, deployment_betas)
        if protocol_version in {
            R2C_V6_PROTOCOL_VERSION,
            R2C_V7_PROTOCOL_VERSION,
            R2C_V8_PROTOCOL_VERSION,
            R2C_V13_PROTOCOL_VERSION,
        }:
            telemetry_shift_detector = TelemetryShiftDetector.from_config(
                NUM_CLIENTS, method_config
            )
        if protocol_version == R2C_V8_PROTOCOL_VERSION:
            deployment_recovery_pulse = DeploymentRecoveryPulse.from_config(method_config)
        if protocol_version == R2C_V13_PROTOCOL_VERSION:
            deployment_dare = DualAnchorRecoveryEnvelope.from_config(method_config)

    round_writer = ChunkedTableWriter(run_dir, "round_metrics", flush_rows=50)
    client_writer = ChunkedTableWriter(run_dir, "client_round_metrics", flush_rows=5000)
    class_writer = ChunkedTableWriter(run_dir, "evaluation_by_class", flush_rows=1000)
    system_writer = ChunkedTableWriter(run_dir, "system_samples", flush_rows=250)
    stage_writer = ChunkedTableWriter(run_dir, "stage_timings", flush_rows=500)
    failure_writer = ChunkedTableWriter(run_dir, "failure_events", flush_rows=50)
    checkpoint_writer = ChunkedTableWriter(run_dir, "checkpoint_metrics", flush_rows=2000)
    certificate_writer = ChunkedTableWriter(run_dir, "certificate_audit", flush_rows=100)
    deployment_writer = ChunkedTableWriter(
        run_dir, "deployment_candidate_metrics", flush_rows=max(100, 25 * max(1, len(deployment_betas)))
    )

    selected_counts = np.zeros(NUM_CLIENTS, dtype=np.int64)
    total_selected = 0
    total_on_time = 0
    fedawe_bank = _make_local_bank(model) if job["method_id"] == "FedAWE" else None
    fedawe_gap = np.ones(NUM_CLIENTS, dtype=np.int64)
    run_wall_start = time.perf_counter()
    process = psutil.Process()
    per_class_rounds = _per_class_rounds(rounds, trace.event_round)
    accuracy_history: list[float] = []
    loss_history: list[float] = []
    round_values: list[int] = []
    algorithm_elapsed = 0.0
    peak_cpu_rss_mib = 0.0
    peak_disk_mib = 0.0

    try:
        for round_number in range(1, rounds + 1):
            round_start = time.perf_counter()
            cpu_start = time.process_time()
            idx = round_number - 1
            available = np.flatnonzero(trace.available[idx]).astype(np.int64)
            all_clients = np.arange(NUM_CLIENTS, dtype=np.int64)
            simulated_compute_all, simulated_upload_all, predicted_duration = trace.simulated_times(
                round_number, all_clients, spec.local_steps, payload_bytes
            )
            telemetry_shift_observation = (
                telemetry_shift_detector.observe(round_number, available, predicted_duration)
                if telemetry_shift_detector is not None
                else None
            )

            lr = _learning_rate(
                spec.base_lr,
                float(job["method_config"].get("lr_mult", 1.0)),
                round_number,
                rounds,
            )
            probe_losses: dict[int, float] = {}
            probe_wall_s = 0.0
            probe_gpu_s = 0.0
            r2c_result = None
            if job["method_id"] == PROPOSED_METHOD:
                method_start = time.perf_counter()
                protocol_version = job["method_config"].get("r2c_protocol_version")
                if protocol_version == R2C_V13_PROTOCOL_VERSION:
                    r2c_runner = run_r2c_v13_round
                elif protocol_version == R2C_V8_PROTOCOL_VERSION:
                    r2c_runner = run_r2c_v8_round
                elif protocol_version == R2C_V7_PROTOCOL_VERSION:
                    r2c_runner = run_r2c_v7_round
                elif protocol_version == R2C_V6_PROTOCOL_VERSION:
                    r2c_runner = run_r2c_v6_round
                elif protocol_version == R2C_V5_PROTOCOL_VERSION:
                    r2c_runner = run_r2c_v5_round
                elif protocol_version == R2C_V4_PROTOCOL_VERSION:
                    r2c_runner = run_r2c_v4_round
                elif protocol_version == R2C_V3_PROTOCOL_VERSION:
                    r2c_runner = run_r2c_v3_round
                elif protocol_version == R2C_V2_PROTOCOL_VERSION:
                    r2c_runner = run_r2c_v2_round
                else:
                    r2c_runner = run_r2c_round
                r2c_kwargs: dict[str, Any] = dict(
                    model=model,
                    trainer=trainer,
                    data=data,
                    trace=trace,
                    round_number=round_number,
                    available=available,
                    sample_counts=sample_counts,
                    learning_rate=lr,
                    payload_bytes=payload_bytes,
                    seed=seed,
                    config=dict(job["method_config"]),
                    run_id=str(job["run_id"]),
                    round_start_model_hash=current_model_hash,
                    full_logging=full_logging,
                )
                if protocol_version in {
                    R2C_V5_PROTOCOL_VERSION,
                    R2C_V6_PROTOCOL_VERSION,
                    R2C_V7_PROTOCOL_VERSION,
                    R2C_V8_PROTOCOL_VERSION,
                    R2C_V13_PROTOCOL_VERSION,
                }:
                    r2c_kwargs["selection_history_counts"] = selected_counts.copy()
                r2c_result = r2c_runner(**r2c_kwargs)
                if telemetry_shift_observation is not None:
                    attach_telemetry_observation(r2c_result, telemetry_shift_observation)
                sampling_duration = max(
                    0.0,
                    r2c_result.algorithm_train_wall_s
                    - r2c_result.checkpoint_write_wall_s
                    - r2c_result.checkpoint_read_wall_s,
                )
                selected = r2c_result.selected
                eligible = r2c_result.eligible
                aggregate_duration = r2c_result.aggregate_wall_s
                loss_before = {
                    int(client_id): float(r2c_result.local_loss_before[int(client_id)])
                    for client_id in selected
                }
                loss_after = {
                    int(client_id): float(r2c_result.local_loss_after[int(client_id)])
                    for client_id in selected
                }
                local_wall_s = r2c_result.algorithm_train_wall_s
                local_gpu_s = r2c_result.algorithm_gpu_s
                audit_wall_s = r2c_result.audit_wall_s
                audit_gpu_s = r2c_result.audit_gpu_s
            else:
                assert adapter is not None
                sampling_start = time.perf_counter()
                selection = adapter.admit(available, round_number, predicted_duration)
                sampling_duration = time.perf_counter() - sampling_start
                if job["method_id"] == "PowerOfChoice" and len(selection.admitted):
                    probe_states = [trace.data_state_id(round_number, int(c)) for c in selection.admitted]
                    probe_losses, probe_wall_s, probe_gpu_s = trainer.probe_losses(
                        selection.admitted, round_number, probe_states
                    )
                    selection = adapter.finish_power_of_choice(selection, probe_losses)

                selected = selection.selected.astype(np.int64)
                selected_states = [trace.data_state_id(round_number, int(c)) for c in selected]
                start_params = _bank_fetch(fedawe_bank, selected, device) if fedawe_bank is not None and len(selected) else None
                local_result = trainer.train(
                    selected,
                    round_number,
                    selected_states,
                    lr,
                    start_params=start_params,
                    compute_checksums=full_logging,
                )
                selected_total = predicted_duration[selected] if len(selected) else np.empty(0)
                on_time_mask = selected_total <= float(trace.deadline[idx])
                eligible = selected[on_time_mask]

                aggregate_start = time.perf_counter()
                if job["method_id"] == "FedAWE":
                    _aggregate_fedawe(
                        model,
                        local_result,
                        selected,
                        eligible,
                        fedawe_gap,
                        float(job["method_config"].get("server_lr", 1.0)),
                    )
                    assert fedawe_bank is not None
                    _bank_reset(fedawe_bank, eligible, model)
                    eligible_set = set(eligible.tolist())
                    for client_id in range(NUM_CLIENTS):
                        fedawe_gap[client_id] = 1 if client_id in eligible_set else fedawe_gap[client_id] + 1
                else:
                    coefficients = adapter.aggregation_coefficients(eligible)
                    _aggregate_standard(model, local_result, selected, eligible, coefficients)
                aggregate_duration = time.perf_counter() - aggregate_start

                loss_before = {int(c): float(v) for c, v in zip(selected, local_result.loss_before)}
                loss_after = {int(c): float(v) for c, v in zip(selected, local_result.loss_after)}
                duration_map = {int(c): float(predicted_duration[int(c)]) for c in selected}
                adapter.observe(
                    round_number,
                    selected,
                    eligible,
                    loss_before,
                    loss_after,
                    duration_map,
                    selection.tier_ids,
                )
                local_wall_s = local_result.wall_s
                local_gpu_s = local_result.gpu_s
                audit_wall_s = 0.0
                audit_gpu_s = 0.0

            selected_counts[selected] += 1
            total_selected += len(selected)
            total_on_time += len(eligible)

            eval_start = time.perf_counter()
            eval_gpu_start = torch.cuda.Event(enable_timing=True)
            eval_gpu_end = torch.cuda.Event(enable_timing=True)
            eval_gpu_start.record()
            evaluation_model = model
            evaluation_model_hash = None
            deployment_hashes_before: dict[str, str] = {}
            deployment_hashes_after: dict[str, str] = {}
            telemetry_trigger_detected = bool(
                telemetry_shift_observation is not None
                and telemetry_shift_observation.trigger
            )
            deployment_pulse_observation: RecoveryPulseObservation | None = (
                deployment_recovery_pulse.step(round_number, telemetry_trigger_detected)
                if deployment_recovery_pulse is not None
                else None
            )
            deployment_dare_observation: DAREObservation | None = (
                deployment_dare.step(round_number, telemetry_trigger_detected)
                if deployment_dare is not None
                else None
            )
            if deployment_dare_observation is not None:
                telemetry_response_applied = bool(
                    deployment_dare_observation.response_applied
                    and not no_drift_quarantine
                )
            elif deployment_pulse_observation is not None:
                telemetry_response_applied = bool(
                    deployment_pulse_observation.response_applied
                    and not no_drift_quarantine
                )
            else:
                telemetry_response_applied = bool(
                    telemetry_trigger_detected and not no_drift_quarantine
                )
            deployment_sync_applied = bool(
                telemetry_response_applied and protocol_version == R2C_V6_PROTOCOL_VERSION
            )
            deployment_quarantine_applied = bool(
                telemetry_response_applied
                and (
                    protocol_version == R2C_V7_PROTOCOL_VERSION
                    or (
                        protocol_version == R2C_V8_PROTOCOL_VERSION
                        and deployment_pulse_observation is not None
                        and deployment_pulse_observation.hold_applied
                    )
                    or (
                        protocol_version == R2C_V13_PROTOCOL_VERSION
                        and deployment_dare_observation is not None
                        and deployment_dare_observation.hold_applied
                    )
                )
            )
            deployment_recovery_pulse_applied = bool(
                telemetry_response_applied
                and protocol_version == R2C_V8_PROTOCOL_VERSION
                and deployment_pulse_observation is not None
                and deployment_pulse_observation.recovery_applied
            )
            deployment_recovery_envelope_applied = bool(
                telemetry_response_applied
                and protocol_version == R2C_V13_PROTOCOL_VERSION
                and deployment_dare_observation is not None
                and deployment_dare_observation.envelope_applied
            )
            deployment_post_shift_tracking_applied = bool(
                telemetry_response_applied
                and protocol_version == R2C_V13_PROTOCOL_VERSION
                and deployment_dare_observation is not None
                and deployment_dare_observation.tracking_applied
            )
            configured_trigger_deployment_beta: float | None = None
            if protocol_version == R2C_V6_PROTOCOL_VERSION:
                configured_trigger_deployment_beta = 0.0
            elif protocol_version == R2C_V7_PROTOCOL_VERSION:
                configured_trigger_deployment_beta = validated_trigger_deployment_beta(method_config)
            elif protocol_version == R2C_V8_PROTOCOL_VERSION:
                configured_trigger_deployment_beta = validated_v8_trigger_deployment_beta(
                    method_config
                )
            elif protocol_version == R2C_V13_PROTOCOL_VERSION:
                configured_trigger_deployment_beta = 1.0
            if deployment_sync_applied:
                deployment_trigger_action = "hard_sync"
            elif deployment_quarantine_applied:
                deployment_trigger_action = "hold"
            elif deployment_recovery_pulse_applied:
                deployment_trigger_action = "recovery_pulse"
            elif deployment_recovery_envelope_applied:
                deployment_trigger_action = "recovery_envelope"
            elif deployment_post_shift_tracking_applied:
                deployment_trigger_action = "post_shift_tracking"
            else:
                deployment_trigger_action = "none"
            deployment_beta_override: float | None = None
            if telemetry_response_applied:
                if deployment_dare_observation is not None:
                    deployment_beta_override = (
                        deployment_dare_observation.equivalent_beta
                    )
                elif deployment_pulse_observation is not None:
                    deployment_beta_override = deployment_pulse_observation.override_beta
                else:
                    deployment_beta_override = configured_trigger_deployment_beta
            effective_primary_deployment_beta = primary_deployment_beta
            if deployment_models:
                for beta, deployment_model in deployment_models.items():
                    beta_key = format(float(beta), ".17g")
                    deployment_hashes_before[beta_key] = model_state_hash(deployment_model)
                    effective_beta = (
                        float(deployment_beta_override)
                        if deployment_beta_override is not None
                        else float(beta)
                    )
                    if (
                        telemetry_response_applied
                        and protocol_version == R2C_V13_PROTOCOL_VERSION
                    ):
                        if deployment_dare is None or deployment_dare_observation is None:
                            raise AssertionError("R2C-v13 deployment controller is unavailable")
                        deployment_dare_observation = deployment_dare.apply(
                            deployment_dare_observation,
                            deployment_model,
                            model,
                        )
                    else:
                        _apply_effective_deployment_update(
                            deployment_model, model, effective_beta
                        )
                    deployment_hashes_after[beta_key] = model_state_hash(deployment_model)
                effective_primary_deployment_beta = (
                    float(deployment_beta_override)
                    if deployment_beta_override is not None
                    else float(primary_deployment_beta)
                )
                evaluation_model = deployment_models[float(primary_deployment_beta)]
            accuracy, eval_loss, class_rows = evaluate_model(
                evaluation_model,
                data,
                job["evaluation_split"],
                device,
                batch_size=2048 if job["dataset_id"] == "D1" else (512 if job["dataset_id"] == "D4" else 1024),
                per_class=full_logging and round_number in per_class_rounds,
            )
            if deployment_models:
                evaluation_model_hash = model_state_hash(evaluation_model)
                for beta, deployment_model in deployment_models.items():
                    if beta == primary_deployment_beta:
                        candidate_accuracy = accuracy
                        candidate_loss = eval_loss
                        candidate_hash = evaluation_model_hash
                    else:
                        candidate_accuracy, candidate_loss, _ = evaluate_model(
                            deployment_model,
                            data,
                            job["evaluation_split"],
                            device,
                            batch_size=(
                                2048
                                if job["dataset_id"] == "D1"
                                else (512 if job["dataset_id"] == "D4" else 1024)
                            ),
                            per_class=False,
                        )
                        candidate_hash = model_state_hash(deployment_model)
                    deployment_writer.append(
                        {
                            "run_id": job["run_id"],
                            "round": round_number,
                            "deployment_beta": beta,
                            "effective_deployment_beta": (
                                float(deployment_beta_override)
                                if deployment_beta_override is not None
                                else float(beta)
                            ),
                            "deployment_synchronization_applied": deployment_sync_applied,
                            "deployment_quarantine_applied": deployment_quarantine_applied,
                            "deployment_recovery_pulse_applied": deployment_recovery_pulse_applied,
                            "deployment_recovery_envelope_applied": deployment_recovery_envelope_applied,
                            "deployment_post_shift_tracking_applied": deployment_post_shift_tracking_applied,
                            "deployment_shift_response_applied": telemetry_response_applied,
                            "deployment_trigger_action": deployment_trigger_action,
                            "configured_trigger_deployment_beta": configured_trigger_deployment_beta,
                            "deployment_pulse_phase": (
                                deployment_pulse_observation.phase
                                if deployment_pulse_observation is not None
                                else None
                            ),
                            "deployment_dare_phase": (
                                deployment_dare_observation.phase
                                if deployment_dare_observation is not None
                                else None
                            ),
                            "deployment_dare_lambda": (
                                deployment_dare_observation.lambda_value
                                if deployment_dare_observation is not None
                                else None
                            ),
                            "deployment_dare_schedule_id": (
                                deployment_dare_observation.configured_schedule_id
                                if deployment_dare_observation is not None
                                else None
                            ),
                            "deployment_dare_pre_anchor_hash": (
                                deployment_dare_observation.pre_anchor_hash
                                if deployment_dare_observation is not None
                                else None
                            ),
                            "deployment_dare_post_anchor_hash": (
                                deployment_dare_observation.post_anchor_hash
                                if deployment_dare_observation is not None
                                else None
                            ),
                            "deployment_model_hash_before": deployment_hashes_before[
                                format(float(beta), ".17g")
                            ],
                            "deployment_model_hash_after": deployment_hashes_after[
                                format(float(beta), ".17g")
                            ],
                            "is_primary": bool(beta == primary_deployment_beta),
                            "test_accuracy": float(candidate_accuracy),
                            "test_loss": float(candidate_loss),
                            "evaluation_model_hash": candidate_hash,
                        }
                    )
            if r2c_result is not None and full_logging:
                if telemetry_shift_observation is not None:
                    certificate = r2c_result.certificate_row
                    certificate["ablation_variant"] = str(
                        method_config.get("r2c_ablation_variant", "full")
                    )
                    certificate["drift_quarantine_enabled"] = not no_drift_quarantine
                    certificate["telemetry_trigger_detected"] = telemetry_trigger_detected
                    certificate["deployment_synchronization_applied"] = deployment_sync_applied
                    certificate["deployment_quarantine_applied"] = deployment_quarantine_applied
                    certificate["deployment_recovery_pulse_applied"] = (
                        deployment_recovery_pulse_applied
                    )
                    certificate["deployment_recovery_envelope_applied"] = (
                        deployment_recovery_envelope_applied
                    )
                    certificate["deployment_post_shift_tracking_applied"] = (
                        deployment_post_shift_tracking_applied
                    )
                    certificate["deployment_shift_response_applied"] = telemetry_response_applied
                    certificate["deployment_trigger_action"] = deployment_trigger_action
                    certificate["configured_trigger_deployment_beta"] = (
                        configured_trigger_deployment_beta
                    )
                    certificate["effective_primary_deployment_beta"] = float(
                        effective_primary_deployment_beta
                    )
                    certificate["deployment_model_hashes_before_json"] = canonical_json(
                        deployment_hashes_before
                    )
                    certificate["deployment_model_hashes_after_json"] = canonical_json(
                        deployment_hashes_after
                    )
                    certificate["deployment_hash_lineage_recorded"] = True
                    if deployment_pulse_observation is not None:
                        certificate.update(deployment_pulse_observation.audit_fields())
                    if deployment_dare_observation is not None:
                        certificate.update(deployment_dare_observation.audit_fields())
                    certificate.pop("certificate_record_hash", None)
                    certificate["certificate_record_hash"] = sha256_text(
                        canonical_json(certificate)
                    )
                    for row in r2c_result.checkpoint_rows:
                        row["ablation_variant"] = str(
                            method_config.get("r2c_ablation_variant", "full")
                        )
                        row["drift_quarantine_enabled"] = not no_drift_quarantine
                        row["telemetry_trigger_detected"] = telemetry_trigger_detected
                        row["deployment_synchronization_applied"] = deployment_sync_applied
                        row["deployment_quarantine_applied"] = deployment_quarantine_applied
                        row["deployment_recovery_pulse_applied"] = (
                            deployment_recovery_pulse_applied
                        )
                        row["deployment_recovery_envelope_applied"] = (
                            deployment_recovery_envelope_applied
                        )
                        row["deployment_post_shift_tracking_applied"] = (
                            deployment_post_shift_tracking_applied
                        )
                        row["deployment_shift_response_applied"] = telemetry_response_applied
                        row["deployment_trigger_action"] = deployment_trigger_action
                        row["configured_trigger_deployment_beta"] = (
                            configured_trigger_deployment_beta
                        )
                        row["effective_primary_deployment_beta"] = float(
                            effective_primary_deployment_beta
                        )
                        if deployment_pulse_observation is not None:
                            row.update(deployment_pulse_observation.audit_fields())
                        if deployment_dare_observation is not None:
                            row.update(deployment_dare_observation.audit_fields())
                checkpoint_writer.extend(r2c_result.checkpoint_rows)
                certificate_writer.append(r2c_result.certificate_row)
            eval_gpu_end.record()
            torch.cuda.synchronize(device)
            eval_gpu_s = eval_gpu_start.elapsed_time(eval_gpu_end) / 1000.0
            eval_duration = time.perf_counter() - eval_start

            global_hash = model_state_hash(model)
            current_model_hash = global_hash
            physical_round_wall = time.perf_counter() - round_start
            round_wall = max(0.0, physical_round_wall - audit_wall_s)
            algorithm_elapsed += round_wall
            cpu_time = time.process_time() - cpu_start
            accuracy_history.append(float(accuracy))
            loss_history.append(float(eval_loss))
            round_values.append(round_number)
            role, offset = event_window_role(round_number, trace.event_round)
            event_id = _event_id(job["dataset_id"], job["scenario_id"], trace.event_round)
            train_loss = (
                float(
                    np.average(
                        np.asarray([loss_after[int(client_id)] for client_id in selected]),
                        weights=sample_counts[selected],
                    )
                )
                if len(selected)
                else None
            )
            participation = participation_jfi(selected_counts)
            worst10 = worst10_participation(selected_counts)
            completion_rate = total_on_time / total_selected if total_selected else 0.0
            if r2c_result is not None:
                downloads = len(r2c_result.admitted) * payload_bytes
                uploads = len(selected) * payload_bytes + r2c_result.telemetry_upload_bytes
                payload_equiv_upload = len(selected) * payload_bytes
                admitted_clients = len(r2c_result.admitted)
            else:
                downloads = (
                    len(selection.admitted) * payload_bytes
                    if job["method_id"] == "PowerOfChoice"
                    else len(selected) * payload_bytes
                )
                uploads = len(selected) * payload_bytes
                payload_equiv_upload = uploads
                admitted_clients = len(selection.admitted)
            round_row = {
                    "run_id": job["run_id"],
                    "round": round_number,
                    "elapsed_wall_s": time.perf_counter() - run_wall_start,
                    "algorithm_elapsed_s": algorithm_elapsed,
                    "round_wall_s": round_wall,
                    "audit_wall_s": audit_wall_s,
                    "test_accuracy": float(accuracy),
                    "test_loss": float(eval_loss),
                    "train_loss_weighted": train_loss,
                    "available_clients": len(available),
                    "admitted_clients": admitted_clients,
                    "selected_clients": len(selected),
                    "completed_clients": len(selected),
                    "on_time_clients": len(eligible),
                    "bytes_upload": int(uploads),
                    "bytes_download": int(downloads),
                    "payload_equiv_upload": int(payload_equiv_upload),
                    "client_compute_s": local_wall_s + probe_wall_s,
                    "server_compute_s": aggregate_duration + sampling_duration,
                    "gpu_time_s": local_gpu_s + probe_gpu_s + eval_gpu_s,
                    "cpu_time_s": cpu_time,
                    "participation_jfi": participation,
                    "worst10_participation": worst10,
                    "deadline_completion_rate": completion_rate,
                    "event_id": event_id,
                    "event_offset_round": offset,
                    "auc20_window_role": role,
                    "dynamic_state_id": f"{trace.trace_hash}:r{round_number}",
                    "global_model_hash": global_hash,
                    "evaluation_model_hash": evaluation_model_hash or global_hash,
                    "evaluation_model_role": (
                        "server_only_dare_envelope"
                        if protocol_version == R2C_V13_PROTOCOL_VERSION
                        else (
                            "server_only_parameter_ema"
                            if deployment_models
                            else "global_training_model"
                        )
                    ),
                }
            if telemetry_shift_observation is not None:
                round_row.update(telemetry_shift_observation.audit_fields())
                if deployment_pulse_observation is not None:
                    round_row.update(deployment_pulse_observation.audit_fields())
                if deployment_dare_observation is not None:
                    round_row.update(deployment_dare_observation.audit_fields())
                round_row.update(
                    {
                        "ablation_variant": str(
                            method_config.get("r2c_ablation_variant", "full")
                        ),
                        "drift_quarantine_enabled": not no_drift_quarantine,
                        "telemetry_trigger_detected": telemetry_trigger_detected,
                        "deployment_synchronization_applied": deployment_sync_applied,
                        "deployment_quarantine_applied": deployment_quarantine_applied,
                        "deployment_recovery_pulse_applied": deployment_recovery_pulse_applied,
                        "deployment_recovery_envelope_applied": deployment_recovery_envelope_applied,
                        "deployment_post_shift_tracking_applied": deployment_post_shift_tracking_applied,
                        "deployment_shift_response_applied": telemetry_response_applied,
                        "deployment_trigger_action": deployment_trigger_action,
                        "configured_trigger_deployment_beta": configured_trigger_deployment_beta,
                        "effective_primary_deployment_beta": float(
                            effective_primary_deployment_beta
                        ),
                        "primary_deployment_model_hash_before": deployment_hashes_before[
                            format(float(primary_deployment_beta), ".17g")
                        ],
                        "primary_deployment_model_hash_after": deployment_hashes_after[
                            format(float(primary_deployment_beta), ".17g")
                        ],
                    }
                )
            round_writer.append(round_row)

            if full_logging:
                admitted_values = r2c_result.admitted if r2c_result is not None else selection.admitted
                admitted_set = set(int(v) for v in admitted_values)
                admission_draw_position = {
                    int(client_id): position
                    for position, client_id in enumerate(admitted_values)
                }
                available_set = set(int(v) for v in available)
                selected_position = {int(c): pos for pos, c in enumerate(selected)}
                eligible_set = set(int(v) for v in eligible)
                for client_id in range(NUM_CLIENTS):
                    available_flag = client_id in available_set
                    admitted_flag = client_id in admitted_set
                    selected_flag = client_id in selected_position
                    position = selected_position.get(client_id)
                    if r2c_result is not None:
                        checksum = r2c_result.checkpoint_checksum[client_id]
                        upload_checksum = r2c_result.upload_checksum[client_id]
                        probe_share = 0.0
                        if admitted_flag:
                            steps_used = spec.local_steps if selected_flag else r2c_result.prefix_steps
                            before_steps = min(steps_used, max(1, int(math.ceil(0.4 * spec.local_steps))))
                            after_steps = max(0, steps_used - before_steps)
                            local_share = float(
                                before_steps * trace.step_before[idx, client_id]
                                + after_steps * trace.step_after[idx, client_id]
                            )
                            if selected_flag and no_reusable_prefix:
                                prefix_before = min(
                                    r2c_result.prefix_steps,
                                    max(1, int(math.ceil(0.4 * spec.local_steps))),
                                )
                                prefix_after = max(
                                    0,
                                    r2c_result.prefix_steps - prefix_before,
                                )
                                local_share += float(
                                    prefix_before * trace.step_before[idx, client_id]
                                    + prefix_after * trace.step_after[idx, client_id]
                                )
                        else:
                            local_share = 0.0
                    else:
                        checksum = local_result.checksums[position] if position is not None else None
                        upload_checksum = checksum
                        probe_share = probe_wall_s / max(1, len(selection.admitted)) if admitted_flag and probe_wall_s else 0.0
                        local_share = local_result.wall_s / max(1, len(selected)) if selected_flag else 0.0
                    observed_step = (
                        float(0.4 * trace.step_before[idx, client_id] + 0.6 * trace.step_after[idx, client_id])
                        if admitted_flag
                        else None
                    )
                    observed_bandwidth = (
                        float(0.4 * trace.bandwidth_before[idx, client_id] + 0.6 * trace.bandwidth_after[idx, client_id])
                        if admitted_flag
                        else None
                    )
                    before_value = loss_before.get(client_id, probe_losses.get(client_id))
                    if r2c_result is not None:
                        utility_value = r2c_result.utility_scores[client_id]
                        admission_value = r2c_result.admission_prob[client_id]
                        conditional_target = r2c_result.conditional_targets[client_id]
                        inclusion_value = r2c_result.inclusion_prob[client_id]
                        finish_value = r2c_result.finish_prob[client_id]
                        effective_value = r2c_result.effective_probability[client_id]
                        hajek_raw = r2c_result.hajek_weight_raw[client_id]
                        hajek_normalized = r2c_result.hajek_weight_normalized[client_id]
                        delta_raw = r2c_result.delta_norm_raw[client_id]
                        delta_clipped = r2c_result.delta_norm_clipped[client_id]
                        tier_id = -1
                        prefix_steps = r2c_result.prefix_steps if admitted_flag else 0
                        final_steps = spec.local_steps if selected_flag else prefix_steps
                        telemetry_bytes = (
                            r2c_result.prefix_steps * 512 if admitted_flag else 0
                        )
                        client_payload = telemetry_bytes + (payload_bytes if selected_flag else 0)
                    else:
                        utility_value = selection.utility_scores[client_id]
                        admission_value = selection.admission_prob[client_id]
                        conditional_target = float(trace.deadline[idx]) if admitted_flag else np.nan
                        inclusion_value = selection.inclusion_prob[client_id]
                        finish_value = np.nan
                        effective_value = np.nan
                        hajek_raw = np.nan
                        hajek_normalized = np.nan
                        delta_raw = float(local_result.delta_norm[position]) if position is not None else np.nan
                        delta_clipped = delta_raw
                        tier_id = int(selection.tier_ids[client_id])
                        prefix_steps = 0
                        final_steps = spec.local_steps if selected_flag else 0
                        client_payload = payload_bytes if selected_flag else 0
                    client_writer.append(
                        {
                            "run_id": job["run_id"],
                            "round": round_number,
                            "client_id": client_id,
                            "client_cluster": int(trace.clusters[client_id]),
                            "n_samples": int(sample_counts[client_id]),
                            "available": bool(available_flag),
                            "admitted": bool(admitted_flag),
                            "admission_draw_position": admission_draw_position.get(client_id),
                            "selected": bool(selected_flag),
                            "stop_sent": bool(r2c_result is not None and admitted_flag and not selected_flag),
                            "stop_ack_s": 0.0 if r2c_result is not None and admitted_flag and not selected_flag else None,
                            "resumed": bool(
                                r2c_result is not None
                                and selected_flag
                                and r2c_result.prefix_steps < spec.local_steps
                                and not no_reusable_prefix
                            ),
                            "completed_local_work": bool(selected_flag),
                            "arrived_on_time": bool(client_id in eligible_set),
                            "aggregation_eligible": bool(client_id in eligible_set),
                            "prefix_steps": prefix_steps,
                            "final_steps": final_steps,
                            "compute_s": local_share + probe_share if admitted_flag else None,
                            "upload_s": float(simulated_upload_all[client_id]) if selected_flag else None,
                            "observed_step_s": observed_step,
                            "observed_bandwidth_bps": observed_bandwidth,
                            "deadline_s": float(trace.deadline[idx]) if admitted_flag else None,
                            "payload_bytes": int(client_payload),
                            "local_loss_before": before_value,
                            "local_loss_after": loss_after.get(client_id),
                            "delta_norm_raw": float(delta_raw) if np.isfinite(delta_raw) else None,
                            "delta_norm_clipped": float(delta_clipped) if np.isfinite(delta_clipped) else None,
                            "utility_score": float(utility_value) if np.isfinite(utility_value) else None,
                            "admission_prob": float(admission_value),
                            "conditional_target_s": float(conditional_target) if np.isfinite(conditional_target) else None,
                            "inclusion_prob_pi": float(inclusion_value) if np.isfinite(inclusion_value) else None,
                            "finish_prob_at_commit": float(finish_value) if np.isfinite(finish_value) else None,
                            "effective_update_probability": float(effective_value) if np.isfinite(effective_value) else None,
                            "hajek_weight_raw": float(hajek_raw) if np.isfinite(hajek_raw) else None,
                            "hajek_weight_normalized": float(hajek_normalized) if np.isfinite(hajek_normalized) else None,
                            "weight_cap_triggered": False,
                            "completion_prob_floor_triggered": bool(
                                r2c_result is not None
                                and admitted_flag
                                and np.isfinite(finish_value)
                                and finish_value
                                < float(job["method_config"].get("r2c_completion_floor", 1.0e-4))
                            ),
                            "selection_count_cumulative": int(selected_counts[client_id]),
                            "reusable_prefix_enabled": (
                                not no_reusable_prefix if r2c_result is not None else None
                            ),
                            "ablation_variant": (
                                str(method_config.get("r2c_ablation_variant", "full"))
                                if r2c_result is not None
                                else None
                            ),
                            "checkpoint_checksum": checksum,
                            "upload_checksum": upload_checksum,
                        }
                    )
                for value in class_rows:
                    value.update({"run_id": job["run_id"], "round": round_number})
                    class_writer.append(value)

            if r2c_result is not None:
                algorithm_start = algorithm_elapsed - round_wall
                stage_writer.extend(
                    [
                        {
                            "run_id": job["run_id"],
                            "round": round_number,
                            "stage": "race_and_resume",
                            "start_elapsed_s": algorithm_start,
                            "end_elapsed_s": algorithm_start + r2c_result.algorithm_train_wall_s,
                            "duration_s": r2c_result.algorithm_train_wall_s,
                            "gpu_s": r2c_result.algorithm_gpu_s,
                            "cpu_s": max(0.0, r2c_result.algorithm_train_wall_s - r2c_result.algorithm_gpu_s),
                            "io_read_bytes": int(
                                r2c_result.checkpoint_read_wall_s > 0
                            ) * payload_bytes * len(selected),
                            "io_write_bytes": payload_bytes * len(r2c_result.admitted) * r2c_result.prefix_steps,
                        },
                        {
                            "run_id": job["run_id"],
                            "round": round_number,
                            "stage": "aggregate",
                            "start_elapsed_s": algorithm_elapsed - eval_duration - aggregate_duration,
                            "end_elapsed_s": algorithm_elapsed - eval_duration,
                            "duration_s": aggregate_duration,
                            "gpu_s": None,
                            "cpu_s": aggregate_duration,
                            "io_read_bytes": 0,
                            "io_write_bytes": 0,
                        },
                        {
                            "run_id": job["run_id"],
                            "round": round_number,
                            "stage": "evaluation",
                            "start_elapsed_s": algorithm_elapsed - eval_duration,
                            "end_elapsed_s": algorithm_elapsed,
                            "duration_s": eval_duration,
                            "gpu_s": eval_gpu_s,
                            "cpu_s": None,
                            "io_read_bytes": 0,
                            "io_write_bytes": 0,
                        },
                        {
                            "run_id": job["run_id"],
                            "round": round_number,
                            "stage": "audit_replay",
                            "start_elapsed_s": time.perf_counter() - run_wall_start - audit_wall_s,
                            "end_elapsed_s": time.perf_counter() - run_wall_start,
                            "duration_s": audit_wall_s,
                            "gpu_s": audit_gpu_s,
                            "cpu_s": max(0.0, audit_wall_s - audit_gpu_s),
                            "io_read_bytes": int(r2c_result.certificate_row["replay_io_bytes"]),
                            "io_write_bytes": 0,
                        },
                    ]
                )
            else:
                stage_writer.extend(
                    [
                        {
                            "run_id": job["run_id"],
                            "round": round_number,
                            "stage": "sampling",
                            "start_elapsed_s": algorithm_elapsed - round_wall,
                            "end_elapsed_s": algorithm_elapsed - round_wall + sampling_duration + probe_wall_s,
                            "duration_s": sampling_duration + probe_wall_s,
                            "gpu_s": probe_gpu_s,
                            "cpu_s": sampling_duration,
                            "io_read_bytes": 0,
                            "io_write_bytes": 0,
                        },
                        {
                            "run_id": job["run_id"],
                            "round": round_number,
                            "stage": "resume",
                            "start_elapsed_s": algorithm_elapsed - round_wall + sampling_duration + probe_wall_s,
                            "end_elapsed_s": algorithm_elapsed - round_wall + sampling_duration + probe_wall_s + local_result.wall_s,
                            "duration_s": local_result.wall_s,
                            "gpu_s": local_result.gpu_s,
                            "cpu_s": None,
                            "io_read_bytes": 0,
                            "io_write_bytes": 0,
                        },
                        {
                            "run_id": job["run_id"],
                            "round": round_number,
                            "stage": "aggregate",
                            "start_elapsed_s": algorithm_elapsed - eval_duration - aggregate_duration,
                            "end_elapsed_s": algorithm_elapsed - eval_duration,
                            "duration_s": aggregate_duration,
                            "gpu_s": None,
                            "cpu_s": aggregate_duration,
                            "io_read_bytes": 0,
                            "io_write_bytes": 0,
                        },
                        {
                            "run_id": job["run_id"],
                            "round": round_number,
                            "stage": "evaluation",
                            "start_elapsed_s": algorithm_elapsed - eval_duration,
                            "end_elapsed_s": algorithm_elapsed,
                            "duration_s": eval_duration,
                            "gpu_s": eval_gpu_s,
                            "cpu_s": None,
                            "io_read_bytes": 0,
                            "io_write_bytes": 0,
                        },
                    ]
                )

            sample_period = max(1, rounds // 100)
            if round_number == 1 or round_number == rounds or round_number % sample_period == 0:
                gpu = gpu_query()
                memory_allocated = torch.cuda.memory_allocated(device) / 2**20
                memory_reserved = torch.cuda.memory_reserved(device) / 2**20
                rss = process.memory_info().rss / 2**20
                disk = psutil.disk_usage(str(RUN_ROOT)).used / 2**20
                peak_cpu_rss_mib = max(peak_cpu_rss_mib, rss)
                peak_disk_mib = max(peak_disk_mib, disk)
                io = process.io_counters()
                system_writer.append(
                    {
                        "run_id": job["run_id"],
                        "timestamp_utc": utc_now(),
                        "elapsed_s": time.perf_counter() - run_wall_start,
                        "round": round_number,
                        "stage": "evaluation",
                        "host_id": socket.gethostname(),
                        "gpu_index": 0,
                        "gpu_uuid": gpu.get("uuid"),
                        "gpu_util_pct": gpu.get("utilization_gpu_pct"),
                        "gpu_power_w": gpu.get("power_draw_w"),
                        "memory_allocated_mib": memory_allocated,
                        "memory_reserved_mib": memory_reserved,
                        "cpu_rss_mib": rss,
                        "cpu_util_pct": process.cpu_percent(interval=None),
                        "disk_used_mib": disk,
                        "io_read_bytes_cumulative": int(io.read_bytes),
                        "io_write_bytes_cumulative": int(io.write_bytes),
                    }
                )

            if round_number % max(1, rounds // 20) == 0 or round_number == rounds:
                progress = {
                    "run_id": job["run_id"],
                    "round": round_number,
                    "round_budget": rounds,
                    "last_accuracy": accuracy,
                    "algorithm_elapsed_s": algorithm_elapsed,
                    "updated_utc": utc_now(),
                }
                atomic_json(run_dir / "progress.json", progress)

            if r2c_result is None:
                del local_result
            if round_number % 25 == 0:
                round_writer.flush()
                if full_logging:
                    client_writer.flush()
                    checkpoint_writer.flush()
                    certificate_writer.flush()
                stage_writer.flush()
                deployment_writer.flush()

        indices = {
            "round_metrics": round_writer.finalize(),
            "client_round_metrics": client_writer.finalize(),
            "evaluation_by_class": class_writer.finalize(),
            "system_samples": system_writer.finalize(),
            "stage_timings": stage_writer.finalize(),
            "failure_events": failure_writer.finalize(),
            "checkpoint_metrics": checkpoint_writer.finalize(),
            "certificate_audit": certificate_writer.finalize(),
            "deployment_candidate_metrics": deployment_writer.finalize(),
        }
        if indices["round_metrics"]["rows"] != rounds:
            raise RuntimeError(
                f"round row count {indices['round_metrics']['rows']} != budget {rounds}"
            )
        if full_logging and indices["client_round_metrics"]["rows"] != rounds * NUM_CLIENTS:
            raise RuntimeError(
                f"client row count {indices['client_round_metrics']['rows']} != {rounds * NUM_CLIENTS}"
            )
        if deployment_models and indices["deployment_candidate_metrics"]["rows"] != rounds * len(deployment_betas):
            raise RuntimeError(
                "deployment candidate row count "
                f"{indices['deployment_candidate_metrics']['rows']} != {rounds * len(deployment_betas)}"
            )
        recovery = recovery_auc20(round_values, accuracy_history, trace.event_round)
        end_utc = utc_now()
        final_manifest = dict(manifest)
        final_manifest.update({"end_utc": end_utc, "status": "completed"})
        atomic_parquet(run_dir / "run_manifest.parquet", pd.DataFrame([final_manifest]))
        result = {
            "run_id": job["run_id"],
            "status": "completed",
            "mode": job["mode"],
            "dataset_id": job["dataset_id"],
            "method_id": job["method_id"],
            "scenario_id": job["scenario_id"],
            "rounds": rounds,
            "final_accuracy": accuracy_history[-1],
            "last50_accuracy": float(np.mean(accuracy_history[-50:])),
            "final_loss": loss_history[-1],
            "validation_objective": float(np.mean(accuracy_history[-max(10, min(50, rounds // 5)) :]))
            - float(recovery["recovery_deficit_auc20"] or 0.0),
            "recovery": recovery,
            "algorithm_elapsed_s": algorithm_elapsed,
            "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / 2**20),
            "peak_cpu_rss_mib": peak_cpu_rss_mib,
            "peak_disk_mib": peak_disk_mib,
            "table_indices": indices,
            "completed_utc": end_utc,
        }
        atomic_json(run_dir / "result.json", result)
        atomic_json(success_path, result)
        return result
    except Exception as exc:
        infra_error = isinstance(exc, (OSError, torch.cuda.OutOfMemoryError)) or "deterministic implementation" in str(exc)
        failure = {
            "run_id": job["run_id"],
            "round": len(round_values) + 1,
            "stage": "run",
            "timestamp_utc": utc_now(),
            "failure_class": "infra_failed" if infra_error else "method_failed",
            "exception_type": type(exc).__name__,
            "message_hash": sha256_text(str(exc)),
            "recoverable": infra_error,
            "action": "preserve_partial_and_retry_with_new_run_id",
            "retry_run_id": None,
        }
        failure_writer.append(failure)
        failure_writer.finalize()
        failed_manifest = dict(manifest)
        failed_manifest.update(
            {
                "end_utc": utc_now(),
                "status": failure["failure_class"],
                "failure_reason": f"{type(exc).__name__}:{failure['message_hash']}",
            }
        )
        atomic_parquet(run_dir / "run_manifest.parquet", pd.DataFrame([failed_manifest]))
        (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    result = run_job(job)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
