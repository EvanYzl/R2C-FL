from __future__ import annotations

import gzip
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets

from .config import ASSET_ROOT, DATASETS, DATA_ROOT, NUM_CLIENTS
from .utils import atomic_json, atomic_parquet, hash_arrays, hash_files, utc_now


NORMALIZATION = {
    "D1": ((0.2860,), (0.3530,)),
    "D2": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "D3": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "D4": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


def _read_idx(path: Path) -> np.ndarray:
    handle_context = gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
    with handle_context as handle:
        magic, = struct.unpack(">I", handle.read(4))
        ndim = magic & 0xFF
        shape = tuple(struct.unpack(">I", handle.read(4))[0] for _ in range(ndim))
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(shape)


def _fashion_files(folder: Path, train: bool) -> tuple[Path, Path]:
    prefix = "train" if train else "t10k"
    extracted = folder / "extracted"
    image = extracted / f"{prefix}-images-idx3-ubyte"
    label = extracted / f"{prefix}-labels-idx1-ubyte"
    if not image.exists():
        image = folder / f"{prefix}-images-idx3-ubyte.gz"
        label = folder / f"{prefix}-labels-idx1-ubyte.gz"
    return image, label


def load_raw_arrays(dataset_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    folder = DATA_ROOT / DATASETS[dataset_id].folder
    if dataset_id == "D1":
        train_image, train_label = _fashion_files(folder, True)
        test_image, test_label = _fashion_files(folder, False)
        x_train = _read_idx(train_image)[:, None, :, :]
        y_train = _read_idx(train_label).astype(np.int64)
        x_test = _read_idx(test_image)[:, None, :, :]
        y_test = _read_idx(test_label).astype(np.int64)
    elif dataset_id == "D2":
        train = datasets.CIFAR10(str(folder), train=True, download=False)
        test = datasets.CIFAR10(str(folder), train=False, download=False)
        x_train = np.asarray(train.data).transpose(0, 3, 1, 2)
        y_train = np.asarray(train.targets, dtype=np.int64)
        x_test = np.asarray(test.data).transpose(0, 3, 1, 2)
        y_test = np.asarray(test.targets, dtype=np.int64)
    elif dataset_id == "D3":
        train = datasets.SVHN(str(folder), split="train", download=False)
        test = datasets.SVHN(str(folder), split="test", download=False)
        x_train = np.asarray(train.data)
        y_train = np.asarray(train.labels, dtype=np.int64) % 10
        x_test = np.asarray(test.data)
        y_test = np.asarray(test.labels, dtype=np.int64) % 10
    elif dataset_id == "D4":
        train = datasets.CIFAR100(str(folder), train=True, download=False)
        test = datasets.CIFAR100(str(folder), train=False, download=False)
        x_train = np.asarray(train.data).transpose(0, 3, 1, 2)
        y_train = np.asarray(train.targets, dtype=np.int64)
        x_test = np.asarray(test.data).transpose(0, 3, 1, 2)
        y_test = np.asarray(test.targets, dtype=np.int64)
    else:
        raise ValueError(dataset_id)
    return x_train, y_train, x_test, y_test


def raw_data_checksum(dataset_id: str) -> str:
    folder = DATA_ROOT / DATASETS[dataset_id].folder
    files = [path for path in folder.rglob("*") if path.is_file()]
    return hash_files(files, folder)


def _dirichlet_partition(labels: np.ndarray, indices: np.ndarray, seed: int, alpha: float = 0.1) -> list[np.ndarray]:
    n_clients = NUM_CLIENTS
    target = len(indices) / n_clients
    for attempt in range(200):
        rng = np.random.default_rng(seed + 104729 * attempt)
        clients: list[list[int]] = [[] for _ in range(n_clients)]
        for class_id in np.unique(labels[indices]):
            class_indices = indices[labels[indices] == class_id].copy()
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(n_clients, alpha))
            capacity_mask = np.asarray([len(value) < target for value in clients], dtype=np.float64)
            proportions *= capacity_mask
            if proportions.sum() == 0:
                proportions = np.ones(n_clients, dtype=np.float64)
            proportions /= proportions.sum()
            cut = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            for client_id, piece in enumerate(np.split(class_indices, cut)):
                clients[client_id].extend(piece.tolist())
        sizes = np.asarray([len(value) for value in clients])
        if sizes.min() >= 10:
            result = []
            for values in clients:
                array = np.asarray(values, dtype=np.int64)
                rng.shuffle(array)
                result.append(array)
            return result
    raise RuntimeError(f"Unable to create nonempty alpha={alpha} partition after 200 attempts")


def _post_drift_mapping(client_indices: list[np.ndarray], labels: np.ndarray, seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(seed + 880301)
    affected = np.sort(rng.choice(NUM_CLIENTS, size=NUM_CLIENTS // 2, replace=False))
    dominant = []
    for client_id in affected:
        values = labels[client_indices[int(client_id)]]
        counts = np.bincount(values, minlength=int(labels.max()) + 1)
        dominant.append(int(np.argmax(counts)))
    ordered = affected[np.argsort(np.asarray(dominant), kind="stable")]
    shift = max(1, len(ordered) // 2)
    source = np.roll(ordered, shift)
    mapping = np.arange(NUM_CLIENTS, dtype=np.int64)
    mapping[ordered] = source
    post = [client_indices[int(mapping[i])].copy() for i in range(NUM_CLIENTS)]
    return mapping, post


def _client_heldout_folds(
    validation_indices: np.ndarray, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Assign the frozen validation pool to disjoint, deterministic client folds.

    The official training split's preregistered 10% validation subset never
    enters local optimization.  It is first assigned without replacement to
    the 100 simulated endpoints and then split by a deterministic example-hash
    ordering.  Post-drift endpoints reuse the frozen ownership mapping.
    """

    rng = np.random.default_rng(int(seed) + 0x524243)
    shuffled = rng.permutation(np.asarray(validation_indices, dtype=np.int64))
    groups = np.array_split(shuffled, NUM_CLIENTS)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    mask = np.uint64((1 << 64) - 1)
    salt = np.uint64(int(seed) & ((1 << 64) - 1))
    for client_id, group in enumerate(groups):
        if len(group) < 2:
            raise RuntimeError("Each client needs at least two held-out examples")
        values = np.asarray(group, dtype=np.uint64)
        keys = (values ^ salt ^ np.uint64(client_id + 1)) * np.uint64(0x9E3779B185EBCA87)
        keys &= mask
        ordered = np.asarray(group, dtype=np.int64)[np.argsort(keys, kind="stable")]
        fold0 = np.sort(ordered[0::2])
        fold1 = np.sort(ordered[1::2])
        folds.append((fold0, fold1))
    flattened = np.concatenate([np.concatenate(value) for value in folds])
    if len(flattened) != len(validation_indices) or len(np.unique(flattened)) != len(flattened):
        raise RuntimeError("Held-out client folds must be a disjoint cover of validation_indices")
    return folds


def partition_asset_path(dataset_id: str, seed: int) -> Path:
    return ASSET_ROOT / "partitions" / f"{dataset_id}_alpha0p1_s{seed}.npz"


def partition_meta_path(dataset_id: str, seed: int) -> Path:
    return ASSET_ROOT / "partitions" / f"{dataset_id}_alpha0p1_s{seed}.json"


def prepare_partition(dataset_id: str, seed: int, force: bool = False) -> dict[str, object]:
    output = partition_asset_path(dataset_id, seed)
    meta_path = partition_meta_path(dataset_id, seed)
    if output.exists() and meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    _, labels, _, _ = load_raw_arrays(dataset_id)
    all_indices = np.arange(len(labels), dtype=np.int64)
    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=0.10,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    train_indices = np.sort(np.asarray(train_indices, dtype=np.int64))
    validation_indices = np.sort(np.asarray(validation_indices, dtype=np.int64))
    clients = _dirichlet_partition(labels, train_indices, seed)
    drift_mapping, post_clients = _post_drift_mapping(clients, labels, seed)
    payload: dict[str, np.ndarray] = {
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "drift_mapping": drift_mapping,
    }
    for client_id in range(NUM_CLIENTS):
        payload[f"pre_{client_id:03d}"] = clients[client_id]
        payload[f"post_{client_id:03d}"] = post_clients[client_id]
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(temp, **payload)
    temp.replace(output)
    partition_hash = hash_arrays(
        [train_indices, validation_indices, drift_mapping]
        + clients
        + post_clients
    )
    meta = {
        "dataset_id": dataset_id,
        "partition_seed": seed,
        "partition_type": "dirichlet_label",
        "dirichlet_alpha": 0.1,
        "validation_rule": "fixed_stratified_10pct_of_official_train",
        "num_clients": NUM_CLIENTS,
        "partition_hash": partition_hash,
        "asset_path": str(output),
        "generated_utc": utc_now(),
        "minimum_client_samples": min(len(v) for v in clients),
        "maximum_client_samples": max(len(v) for v in clients),
    }
    atomic_json(meta_path, meta)
    return meta


@dataclass
class FederatedData:
    dataset_id: str
    seed: int
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    clients_pre: list[np.ndarray]
    clients_post: list[np.ndarray]
    heldout_folds_pre: list[tuple[np.ndarray, np.ndarray]]
    drift_mapping: np.ndarray
    partition_hash: str
    heldout_hash: str

    @classmethod
    def load(cls, dataset_id: str, seed: int) -> "FederatedData":
        meta = prepare_partition(dataset_id, seed)
        x_train, y_train, x_test, y_test = load_raw_arrays(dataset_id)
        with np.load(partition_asset_path(dataset_id, seed), allow_pickle=False) as archive:
            validation_indices = archive["validation_indices"].copy()
            heldout_folds = _client_heldout_folds(validation_indices, seed)
            return cls(
                dataset_id=dataset_id,
                seed=seed,
                train_images=x_train,
                train_labels=y_train,
                test_images=x_test,
                test_labels=y_test,
                train_indices=archive["train_indices"].copy(),
                validation_indices=validation_indices,
                clients_pre=[archive[f"pre_{i:03d}"].copy() for i in range(NUM_CLIENTS)],
                clients_post=[archive[f"post_{i:03d}"].copy() for i in range(NUM_CLIENTS)],
                heldout_folds_pre=heldout_folds,
                drift_mapping=archive["drift_mapping"].copy(),
                partition_hash=str(meta["partition_hash"]),
                heldout_hash=hash_arrays(
                    [value for pair in heldout_folds for value in pair]
                ),
            )

    def indices_for(self, client_id: int, data_state_id: str) -> np.ndarray:
        if data_state_id == "post_swap":
            return self.clients_post[int(client_id)]
        return self.clients_pre[int(client_id)]

    def client_size(self, client_id: int, data_state_id: str = "pre") -> int:
        return int(len(self.indices_for(client_id, data_state_id)))

    def heldout_indices_for(
        self, client_id: int, data_state_id: str, fold: int
    ) -> np.ndarray:
        if fold not in (0, 1):
            raise ValueError("fold must be 0 or 1")
        owner = int(self.drift_mapping[int(client_id)]) if data_state_id == "post_swap" else int(client_id)
        return self.heldout_folds_pre[owner][fold]

    def batch(
        self,
        client_ids: Iterable[int],
        round_number: int,
        local_step: int,
        batch_size: int,
        data_state_id: str | Iterable[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        client_ids = list(client_ids)
        if isinstance(data_state_id, str):
            states = [data_state_id] * len(client_ids)
        else:
            states = list(data_state_id)
        if len(states) != len(client_ids):
            raise ValueError("data_state_id length must match client_ids")
        for client_id, state in zip(client_ids, states):
            state_code = 1 if state == "post_swap" else 0
            indices = self.indices_for(int(client_id), state)
            local_seed = (
                int(self.seed) * 1_000_003
                + int(client_id) * 10_007
                + int(round_number) * 101
                + int(local_step) * 17
                + state_code
            ) % (2**63 - 1)
            rng = np.random.default_rng(local_seed)
            chosen = rng.choice(indices, size=batch_size, replace=len(indices) < batch_size)
            images.append(self.train_images[chosen])
            labels.append(self.train_labels[chosen])
        x = torch.from_numpy(np.stack(images)).to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
        y = torch.from_numpy(np.stack(labels)).to(device=device, dtype=torch.long, non_blocking=True)
        mean, std = NORMALIZATION[self.dataset_id]
        mean_tensor = torch.tensor(mean, device=device, dtype=x.dtype).view(1, 1, -1, 1, 1)
        std_tensor = torch.tensor(std, device=device, dtype=x.dtype).view(1, 1, -1, 1, 1)
        x = (x - mean_tensor) / std_tensor
        return x, y

    def heldout_batch(
        self,
        client_ids: Iterable[int],
        data_state_id: str | Iterable[str],
        fold: int,
        device: torch.device,
        limit: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        client_ids = list(client_ids)
        if isinstance(data_state_id, str):
            states = [data_state_id] * len(client_ids)
        else:
            states = list(data_state_id)
        if len(states) != len(client_ids):
            raise ValueError("data_state_id length must match client_ids")
        index_sets = [
            self.heldout_indices_for(int(client_id), state, fold)
            for client_id, state in zip(client_ids, states)
        ]
        if not index_sets:
            raise ValueError("heldout_batch requires at least one client")
        count = min(len(value) for value in index_sets)
        if limit is not None:
            count = min(count, int(limit))
        if count <= 0:
            raise RuntimeError("Empty held-out fold")
        images = [self.train_images[value[:count]] for value in index_sets]
        labels = [self.train_labels[value[:count]] for value in index_sets]
        x = torch.from_numpy(np.stack(images)).to(
            device=device, dtype=torch.float32, non_blocking=True
        ).div_(255.0)
        y = torch.from_numpy(np.stack(labels)).to(
            device=device, dtype=torch.long, non_blocking=True
        )
        mean, std = NORMALIZATION[self.dataset_id]
        mean_tensor = torch.tensor(mean, device=device, dtype=x.dtype).view(1, 1, -1, 1, 1)
        std_tensor = torch.tensor(std, device=device, dtype=x.dtype).view(1, 1, -1, 1, 1)
        return (x - mean_tensor) / std_tensor, y

    def anchor_indices(self, fold: int, limit: int) -> np.ndarray:
        """Return deterministic class-balanced indices from one frozen fold."""
        if fold not in (0, 1):
            raise ValueError("fold must be 0 or 1")
        if int(limit) <= 0:
            raise ValueError("anchor limit must be positive")
        pool = np.concatenate([pair[fold] for pair in self.heldout_folds_pre])
        labels = self.train_labels[pool]
        selected: list[int] = []
        per_class: list[np.ndarray] = []
        for class_id in range(DATASETS[self.dataset_id].num_classes):
            values = np.asarray(pool[labels == class_id], dtype=np.int64)
            keys = (
                values.astype(np.uint64)
                ^ np.uint64(self.seed)
                ^ np.uint64((fold + 1) * 0x9E3779B1)
            ) * np.uint64(0x9E3779B185EBCA87)
            per_class.append(values[np.argsort(keys, kind="stable")])
        depth = 0
        while len(selected) < min(int(limit), len(pool)):
            added = False
            for values in per_class:
                if depth < len(values):
                    selected.append(int(values[depth]))
                    added = True
                    if len(selected) >= int(limit):
                        break
            if not added:
                break
            depth += 1
        if not selected:
            raise RuntimeError("Empty global validation anchor")
        return np.asarray(selected, dtype=np.int64)

    def anchor_batch(
        self,
        fold: int,
        device: torch.device,
        limit: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a deterministic class-balanced global validation anchor.

        The two anchor folds are disjoint because they are assembled from the
        already frozen per-endpoint held-out folds.  The same examples are used
        for every candidate in a round, eliminating client-specific evaluation
        noise while keeping all anchor examples outside local optimization.
        """

        indices = self.anchor_indices(fold, limit)
        x = torch.from_numpy(self.train_images[indices]).to(
            device=device, dtype=torch.float32, non_blocking=True
        ).div_(255.0)
        y = torch.from_numpy(self.train_labels[indices]).to(
            device=device, dtype=torch.long, non_blocking=True
        )
        mean, std = NORMALIZATION[self.dataset_id]
        mean_tensor = torch.tensor(mean, device=device, dtype=x.dtype).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std, device=device, dtype=x.dtype).view(1, -1, 1, 1)
        return (x - mean_tensor) / std_tensor, y

    def guard_anchor_indices(
        self,
        fold: int,
        limit: int,
        selection_limit: int,
    ) -> np.ndarray:
        """Return a class-balanced guard anchor disjoint from selection anchors."""

        if fold not in (0, 1):
            raise ValueError("fold must be 0 or 1")
        if int(limit) <= 0 or int(selection_limit) <= 0:
            raise ValueError("guard and selection anchor limits must be positive")
        pool = np.concatenate([pair[fold] for pair in self.heldout_folds_pre])
        selection = self.anchor_indices(fold, int(selection_limit))
        pool = pool[~np.isin(pool, selection)]
        labels = self.train_labels[pool]
        selected: list[int] = []
        per_class: list[np.ndarray] = []
        for class_id in range(DATASETS[self.dataset_id].num_classes):
            values = np.asarray(pool[labels == class_id], dtype=np.int64)
            keys = (
                values.astype(np.uint64)
                ^ np.uint64(self.seed)
                ^ np.uint64((fold + 3) * 0x85EBCA77)
            ) * np.uint64(0xC2B2AE3D27D4EB4F)
            per_class.append(values[np.argsort(keys, kind="stable")])
        depth = 0
        while len(selected) < min(int(limit), len(pool)):
            added = False
            for values in per_class:
                if depth < len(values):
                    selected.append(int(values[depth]))
                    added = True
                    if len(selected) >= int(limit):
                        break
            if not added:
                break
            depth += 1
        if not selected:
            raise RuntimeError("Empty global guard anchor")
        result = np.asarray(selected, dtype=np.int64)
        if np.intersect1d(result, selection).size:
            raise RuntimeError("Guard anchor overlaps selection anchor")
        return result

    def guard_anchor_batch(
        self,
        fold: int,
        device: torch.device,
        limit: int,
        selection_limit: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a deterministic validation-only server trust-region batch."""

        indices = self.guard_anchor_indices(fold, limit, selection_limit)
        x = torch.from_numpy(self.train_images[indices]).to(
            device=device, dtype=torch.float32, non_blocking=True
        ).div_(255.0)
        y = torch.from_numpy(self.train_labels[indices]).to(
            device=device, dtype=torch.long, non_blocking=True
        )
        mean, std = NORMALIZATION[self.dataset_id]
        mean_tensor = torch.tensor(mean, device=device, dtype=x.dtype).view(1, -1, 1, 1)
        std_tensor = torch.tensor(std, device=device, dtype=x.dtype).view(1, -1, 1, 1)
        return (x - mean_tensor) / std_tensor, y

    def evaluation_tensors(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        if split == "test":
            return self.test_images, self.test_labels
        if split == "validation":
            return self.train_images[self.validation_indices], self.train_labels[self.validation_indices]
        raise ValueError(split)


def partition_stats(dataset_id: str, seed: int) -> pd.DataFrame:
    data = FederatedData.load(dataset_id, seed)
    spec = DATASETS[dataset_id]
    rows: list[dict[str, object]] = []
    global_counts = np.bincount(data.train_labels[data.train_indices], minlength=spec.num_classes).astype(np.float64)
    global_prob = global_counts / global_counts.sum()
    event_round = spec.round_budget // 2
    for state, clients, valid_from, valid_to in (
        ("pre", data.clients_pre, 1, event_round),
        ("post_swap", data.clients_post, event_round + 1, spec.round_budget),
    ):
        for client_id, indices in enumerate(clients):
            counts = np.bincount(data.train_labels[indices], minlength=spec.num_classes).astype(np.int64)
            prob = counts.astype(np.float64) / max(1, counts.sum())
            nz = prob > 0
            entropy = float(-(prob[nz] * np.log(prob[nz])).sum())
            mixture = 0.5 * (prob + global_prob)
            p_mask = prob > 0
            g_mask = global_prob > 0
            js = 0.5 * float((prob[p_mask] * np.log(prob[p_mask] / mixture[p_mask])).sum())
            js += 0.5 * float((global_prob[g_mask] * np.log(global_prob[g_mask] / mixture[g_mask])).sum())
            for class_id, count in enumerate(counts):
                rows.append(
                    {
                        "source_kind": "TRACE",
                        "partition_hash": data.partition_hash,
                        "dataset_id": dataset_id,
                        "partition_seed": seed,
                        "client_id": client_id,
                        "n_samples": int(len(indices)),
                        "class_id": class_id,
                        "class_count": int(count),
                        "client_class_entropy": entropy,
                        "dominant_class_fraction": float(prob.max()),
                        "js_to_global": js,
                        "data_state_id": state,
                        "valid_from_round": valid_from,
                        "valid_to_round": valid_to,
                    }
                )
    return pd.DataFrame(rows)


def write_partition_stats(dataset_ids: Iterable[str], seed: int, output: Path) -> None:
    frames = [partition_stats(dataset_id, seed) for dataset_id in dataset_ids]
    atomic_parquet(output, pd.concat(frames, ignore_index=True))
