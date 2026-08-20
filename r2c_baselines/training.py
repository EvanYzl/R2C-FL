from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, grad_and_value, vmap

from .data import FederatedData, NORMALIZATION


@dataclass
class LocalTrainResult:
    client_ids: np.ndarray
    start_params: dict[str, torch.Tensor]
    final_params: dict[str, torch.Tensor]
    loss_before: np.ndarray
    loss_after: np.ndarray
    delta_norm: np.ndarray
    checksums: list[str]
    wall_s: float
    gpu_s: float


@dataclass
class R2CStepResult:
    loss: np.ndarray
    wall_s: float
    gpu_s: float


class LocalTrainer:
    def __init__(
        self,
        model: nn.Module,
        data: FederatedData,
        device: torch.device,
        batch_size: int,
        local_steps: int,
        weight_decay: float,
        max_parallel_clients: int = 1,
    ) -> None:
        self.model = model
        self.data = data
        self.device = device
        self.batch_size = int(batch_size)
        self.local_steps = int(local_steps)
        self.weight_decay = float(weight_decay)
        self.max_parallel_clients = max(1, int(max_parallel_clients))
        self._grad_and_value = grad_and_value(self._loss)
        self._batched_grad_and_value = vmap(self._grad_and_value, in_dims=(0, 0, 0))
        self._batched_loss = vmap(self._loss, in_dims=(0, 0, 0))
        self._batched_per_example_loss = vmap(self._per_example_loss, in_dims=(0, 0, 0))

    def _loss(self, params: dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = functional_call(self.model, params, (x,), strict=True)
        return F.cross_entropy(logits, y)

    def _per_example_loss(
        self, params: dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        logits = functional_call(self.model, params, (x,), strict=True)
        return F.cross_entropy(logits, y, reduction="none")

    def global_params(self) -> dict[str, torch.Tensor]:
        return {name: parameter for name, parameter in self.model.named_parameters()}

    def stacked_global(self, count: int) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().unsqueeze(0).expand(count, *parameter.shape).clone()
            for name, parameter in self.model.named_parameters()
        }

    def r2c_train_step(
        self,
        params: dict[str, torch.Tensor],
        client_ids: np.ndarray,
        round_number: int,
        data_states: list[str],
        local_step: int,
        learning_rate: float,
    ) -> R2CStepResult:
        """Advance a persistent candidate-state stack by exactly one SGD step."""

        client_ids = np.asarray(client_ids, dtype=np.int64)
        if len(client_ids) == 0:
            return R2CStepResult(np.empty(0), 0.0, 0.0)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)
        wall_start = time.perf_counter()
        start_event.record()
        losses_out: list[np.ndarray] = []
        for start in range(0, len(client_ids), self.max_parallel_clients):
            stop = min(len(client_ids), start + self.max_parallel_clients)
            chunk = {
                name: value[start:stop].detach().clone()
                for name, value in params.items()
            }
            x, y = self.data.batch(
                client_ids[start:stop],
                round_number,
                int(local_step),
                self.batch_size,
                data_states[start:stop],
                self.device,
            )
            grads, losses = self._batched_grad_and_value(chunk, x, y)
            with torch.no_grad():
                for name, value in chunk.items():
                    gradient = grads[name]
                    if self.weight_decay:
                        gradient = gradient + self.weight_decay * value
                    params[name][start:stop].copy_(
                        value - float(learning_rate) * gradient
                    )
            losses_out.append(losses.detach().cpu().numpy())
            del chunk, x, y, grads, losses
        end_event.record()
        torch.cuda.synchronize(self.device)
        return R2CStepResult(
            loss=np.concatenate(losses_out),
            wall_s=time.perf_counter() - wall_start,
            gpu_s=start_event.elapsed_time(end_event) / 1000.0,
        )

    def r2c_heldout_losses(
        self,
        params: dict[str, torch.Tensor],
        client_ids: np.ndarray,
        data_states: list[str],
        fold: int,
        limit: int,
        parallel_clients: int | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Return per-example held-out losses for one cross-fitting fold."""

        client_ids = np.asarray(client_ids, dtype=np.int64)
        if len(client_ids) == 0:
            return np.empty((0, 0)), 0.0, 0.0
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)
        wall_start = time.perf_counter()
        start_event.record()
        values: list[np.ndarray] = []
        eval_parallel = max(
            self.max_parallel_clients,
            int(parallel_clients or self.max_parallel_clients),
        )
        for start in range(0, len(client_ids), eval_parallel):
            stop = min(len(client_ids), start + eval_parallel)
            chunk = {name: value[start:stop] for name, value in params.items()}
            x, y = self.data.heldout_batch(
                client_ids[start:stop],
                data_states[start:stop],
                fold,
                self.device,
                limit=limit,
            )
            with torch.no_grad():
                losses = self._batched_per_example_loss(chunk, x, y)
            values.append(losses.detach().cpu().numpy())
            del chunk, x, y, losses
        end_event.record()
        torch.cuda.synchronize(self.device)
        return (
            np.concatenate(values, axis=0),
            time.perf_counter() - wall_start,
            start_event.elapsed_time(end_event) / 1000.0,
        )

    def r2c_anchor_losses(
        self,
        params: dict[str, torch.Tensor],
        fold: int,
        limit: int,
        parallel_clients: int | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Evaluate every candidate on the same frozen validation anchor."""

        count = next(iter(params.values())).shape[0]
        if count == 0:
            return np.empty((0, 0)), 0.0, 0.0
        anchor_x, anchor_y = self.data.anchor_batch(fold, self.device, int(limit))
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)
        wall_start = time.perf_counter()
        start_event.record()
        values: list[np.ndarray] = []
        eval_parallel = max(
            self.max_parallel_clients,
            int(parallel_clients or self.max_parallel_clients),
        )
        for start in range(0, count, eval_parallel):
            stop = min(count, start + eval_parallel)
            chunk_count = stop - start
            chunk = {name: value[start:stop] for name, value in params.items()}
            x = anchor_x.unsqueeze(0).expand(chunk_count, *anchor_x.shape)
            y = anchor_y.unsqueeze(0).expand(chunk_count, *anchor_y.shape)
            with torch.no_grad():
                losses = self._batched_per_example_loss(chunk, x, y)
            values.append(losses.detach().cpu().numpy())
            del chunk, x, y, losses
        end_event.record()
        torch.cuda.synchronize(self.device)
        del anchor_x, anchor_y
        return (
            np.concatenate(values, axis=0),
            time.perf_counter() - wall_start,
            start_event.elapsed_time(end_event) / 1000.0,
        )

    def r2c_guard_losses(
        self,
        params: dict[str, torch.Tensor],
        fold: int,
        limit: int,
        selection_limit: int,
        parallel_clients: int | None = None,
    ) -> tuple[np.ndarray, float, float]:
        """Evaluate server-step candidates on a disjoint shared guard anchor."""

        count = next(iter(params.values())).shape[0]
        if count == 0:
            return np.empty((0, 0)), 0.0, 0.0
        guard_x, guard_y = self.data.guard_anchor_batch(
            fold, self.device, int(limit), int(selection_limit)
        )
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)
        wall_start = time.perf_counter()
        start_event.record()
        values: list[np.ndarray] = []
        eval_parallel = max(
            self.max_parallel_clients,
            int(parallel_clients or self.max_parallel_clients),
        )
        for start in range(0, count, eval_parallel):
            stop = min(count, start + eval_parallel)
            chunk_count = stop - start
            chunk = {name: value[start:stop] for name, value in params.items()}
            x = guard_x.unsqueeze(0).expand(chunk_count, *guard_x.shape)
            y = guard_y.unsqueeze(0).expand(chunk_count, *guard_y.shape)
            with torch.no_grad():
                losses = self._batched_per_example_loss(chunk, x, y)
            values.append(losses.detach().cpu().numpy())
            del chunk, x, y, losses
        end_event.record()
        torch.cuda.synchronize(self.device)
        del guard_x, guard_y
        return (
            np.concatenate(values, axis=0),
            time.perf_counter() - wall_start,
            start_event.elapsed_time(end_event) / 1000.0,
        )

    def probe_losses(
        self,
        client_ids: np.ndarray,
        round_number: int,
        data_states: list[str],
    ) -> tuple[dict[int, float], float, float]:
        if len(client_ids) > self.max_parallel_clients:
            values: dict[int, float] = {}
            wall_s = 0.0
            gpu_s = 0.0
            for start in range(0, len(client_ids), self.max_parallel_clients):
                stop = min(len(client_ids), start + self.max_parallel_clients)
                partial, partial_wall, partial_gpu = self.probe_losses(
                    client_ids[start:stop], round_number, data_states[start:stop]
                )
                values.update(partial)
                wall_s += partial_wall
                gpu_s += partial_gpu
            return values, wall_s, gpu_s
        if len(client_ids) == 0:
            return {}, 0.0, 0.0
        params = self.stacked_global(len(client_ids))
        x, y = self.data.batch(
            client_ids,
            round_number,
            9001,
            self.batch_size,
            data_states,
            self.device,
        )
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)
        wall_start = time.perf_counter()
        start_event.record()
        with torch.no_grad():
            losses = self._batched_loss(params, x, y)
        end_event.record()
        torch.cuda.synchronize(self.device)
        wall_s = time.perf_counter() - wall_start
        gpu_s = start_event.elapsed_time(end_event) / 1000.0
        values = losses.detach().cpu().numpy()
        del params, x, y, losses
        return {int(c): float(v) for c, v in zip(client_ids, values)}, wall_s, gpu_s

    def train(
        self,
        client_ids: np.ndarray,
        round_number: int,
        data_states: list[str],
        learning_rate: float,
        start_params: dict[str, torch.Tensor] | None = None,
        compute_checksums: bool = True,
    ) -> LocalTrainResult:
        client_ids = np.asarray(client_ids, dtype=np.int64)
        if len(client_ids) > self.max_parallel_clients:
            partial_results: list[LocalTrainResult] = []
            for start in range(0, len(client_ids), self.max_parallel_clients):
                stop = min(len(client_ids), start + self.max_parallel_clients)
                partial_start = (
                    {name: value[start:stop] for name, value in start_params.items()}
                    if start_params is not None
                    else None
                )
                partial_results.append(
                    self.train(
                        client_ids[start:stop],
                        round_number,
                        data_states[start:stop],
                        learning_rate,
                        start_params=partial_start,
                        compute_checksums=compute_checksums,
                    )
                )
            return LocalTrainResult(
                client_ids=np.concatenate([value.client_ids for value in partial_results]),
                start_params={
                    name: torch.cat([value.start_params[name] for value in partial_results], dim=0)
                    for name in partial_results[0].start_params
                },
                final_params={
                    name: torch.cat([value.final_params[name] for value in partial_results], dim=0)
                    for name in partial_results[0].final_params
                },
                loss_before=np.concatenate([value.loss_before for value in partial_results]),
                loss_after=np.concatenate([value.loss_after for value in partial_results]),
                delta_norm=np.concatenate([value.delta_norm for value in partial_results]),
                checksums=[item for value in partial_results for item in value.checksums],
                wall_s=float(sum(value.wall_s for value in partial_results)),
                gpu_s=float(sum(value.gpu_s for value in partial_results)),
            )
        client_ids = np.asarray(client_ids, dtype=np.int64)
        if len(client_ids) == 0:
            return LocalTrainResult(client_ids, {}, {}, np.empty(0), np.empty(0), np.empty(0), [], 0.0, 0.0)
        params = start_params if start_params is not None else self.stacked_global(len(client_ids))
        original = {name: value.detach().clone() for name, value in params.items()}
        first_x: torch.Tensor | None = None
        first_y: torch.Tensor | None = None
        loss_before: torch.Tensor | None = None
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)
        wall_start = time.perf_counter()
        start_event.record()
        for local_step in range(self.local_steps):
            x, y = self.data.batch(
                client_ids,
                round_number,
                local_step,
                self.batch_size,
                data_states,
                self.device,
            )
            if local_step == 0:
                first_x, first_y = x, y
            grads, losses = self._batched_grad_and_value(params, x, y)
            if local_step == 0:
                loss_before = losses.detach()
            updated: dict[str, torch.Tensor] = {}
            for name, value in params.items():
                gradient = grads[name]
                if self.weight_decay:
                    gradient = gradient + self.weight_decay * value
                updated[name] = value - float(learning_rate) * gradient
            params = updated
            if local_step != 0:
                del x, y
            del grads, losses
        assert first_x is not None and first_y is not None and loss_before is not None
        with torch.no_grad():
            loss_after = self._batched_loss(params, first_x, first_y)
        end_event.record()
        torch.cuda.synchronize(self.device)
        wall_s = time.perf_counter() - wall_start
        gpu_s = start_event.elapsed_time(end_event) / 1000.0

        norms_sq = torch.zeros(len(client_ids), device=self.device, dtype=torch.float64)
        for name in params:
            delta = (params[name] - original[name]).reshape(len(client_ids), -1).double()
            norms_sq += torch.square(delta).sum(dim=1)
        delta_norm = torch.sqrt(norms_sq).detach().cpu().numpy()
        checksums: list[str] = []
        if compute_checksums:
            digests = [hashlib.sha256() for _ in client_ids]
            for name in sorted(params):
                cpu = params[name].detach().cpu().contiguous().numpy()
                for index, digest in enumerate(digests):
                    digest.update(cpu[index].tobytes())
            checksums = [digest.hexdigest() for digest in digests]
        else:
            checksums = [""] * len(client_ids)
        return LocalTrainResult(
            client_ids=client_ids,
            start_params=original,
            final_params=params,
            loss_before=loss_before.detach().cpu().numpy(),
            loss_after=loss_after.detach().cpu().numpy(),
            delta_norm=delta_norm,
            checksums=checksums,
            wall_s=wall_s,
            gpu_s=gpu_s,
        )


def model_state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_model_from_params(model: nn.Module, params: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        named = dict(model.named_parameters())
        for name, value in params.items():
            named[name].copy_(value)


def evaluate_model(
    model: nn.Module,
    data: FederatedData,
    split: str,
    device: torch.device,
    batch_size: int,
    per_class: bool = False,
) -> tuple[float, float, list[dict[str, float | int]]]:
    images, labels = data.evaluation_tensors(split)
    mean, std = NORMALIZATION[data.dataset_id]
    mean_tensor = torch.tensor(mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=torch.float32).view(1, -1, 1, 1)
    total_loss = 0.0
    total_correct = 0
    total = 0
    class_support = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    class_correct = np.zeros_like(class_support)
    class_loss = np.zeros_like(class_support, dtype=np.float64)
    class_entropy = np.zeros_like(class_support, dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            stop = min(len(labels), start + batch_size)
            x = torch.from_numpy(images[start:stop]).to(device=device, dtype=torch.float32).div_(255.0)
            y = torch.from_numpy(labels[start:stop]).to(device=device, dtype=torch.long)
            x = (x - mean_tensor) / std_tensor
            logits = model(x)
            losses = F.cross_entropy(logits, y, reduction="none")
            prediction = logits.argmax(dim=1)
            total_loss += float(losses.sum().item())
            total_correct += int((prediction == y).sum().item())
            total += len(y)
            if per_class:
                probs = torch.softmax(logits, dim=1)
                entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=1)
                y_cpu = y.cpu().numpy()
                correct_cpu = (prediction == y).cpu().numpy()
                loss_cpu = losses.cpu().numpy()
                entropy_cpu = entropy.cpu().numpy()
                for class_id in np.unique(y_cpu):
                    mask = y_cpu == class_id
                    class_support[class_id] += int(mask.sum())
                    class_correct[class_id] += int(correct_cpu[mask].sum())
                    class_loss[class_id] += float(loss_cpu[mask].sum())
                    class_entropy[class_id] += float(entropy_cpu[mask].sum())
    rows: list[dict[str, float | int]] = []
    if per_class:
        for class_id in range(len(class_support)):
            support = int(class_support[class_id])
            rows.append(
                {
                    "class_id": class_id,
                    "support": support,
                    "correct": int(class_correct[class_id]),
                    "accuracy": float(class_correct[class_id] / support) if support else float("nan"),
                    "loss_sum": float(class_loss[class_id]),
                    "prediction_entropy": float(class_entropy[class_id] / support) if support else float("nan"),
                }
            )
    model.train()
    return total_correct / total, total_loss / total, rows
