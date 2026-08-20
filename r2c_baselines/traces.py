from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import ASSET_ROOT, DATASETS, NUM_CLIENTS
from .models import build_model, model_payload_bytes
from .utils import atomic_csv, atomic_json, atomic_parquet, canonical_json, hash_arrays, sha256_text, utc_now


TRACE_GENERATOR_VERSION = "r2c-trace-v1"


def event_round_for(scenario_id: str, rounds: int) -> int | None:
    if scenario_id == "S0":
        return None
    if scenario_id == "S1":
        return max(1, int(round(0.25 * rounds)))
    return max(1, int(round(0.50 * rounds)))


def trace_asset_path(dataset_id: str, scenario_id: str, seed: int, rounds: int) -> Path:
    return ASSET_ROOT / "traces" / f"{dataset_id}_{scenario_id}_base_s{seed}_T{rounds}.npz"


def trace_meta_path(dataset_id: str, scenario_id: str, seed: int, rounds: int) -> Path:
    return ASSET_ROOT / "traces" / f"{dataset_id}_{scenario_id}_base_s{seed}_T{rounds}.json"


def _base_profiles(dataset_id: str, seed: int) -> dict[str, np.ndarray]:
    dataset_number = int(dataset_id[1:])
    rng = np.random.default_rng(seed + dataset_number * 1_000_033)
    clusters = np.arange(NUM_CLIENTS, dtype=np.int16) % 4
    rng.shuffle(clusters)
    base_online = rng.uniform(0.58, 0.94, size=NUM_CLIENTS)
    compute_order = rng.permutation(NUM_CLIENTS)
    network_order = rng.permutation(NUM_CLIENTS)
    compute_rank = np.empty(NUM_CLIENTS, dtype=np.float64)
    network_rank = np.empty(NUM_CLIENTS, dtype=np.float64)
    compute_rank[compute_order] = np.linspace(0.0, 1.0, NUM_CLIENTS)
    network_rank[network_order] = np.linspace(0.0, 1.0, NUM_CLIENTS)
    fastest = {"D1": 0.0025, "D2": 0.0040, "D3": 0.0040, "D4": 0.0120}[dataset_id]
    step_time = fastest * np.power(4.0, compute_rank)
    bandwidth = 80_000_000.0 * np.power(8.0, network_rank)
    # This stream is intentionally identical to data._post_drift_mapping so
    # scenario data_state_id and the frozen post-swap ownership map address the
    # same 50 clients.  Reusing the mask for resource reversal makes S4's
    # affected population explicit and auditable.
    affected_rng = np.random.default_rng(seed + 880301)
    affected = np.zeros(NUM_CLIENTS, dtype=bool)
    affected[affected_rng.choice(NUM_CLIENTS, size=NUM_CLIENTS // 2, replace=False)] = True
    return {
        "clusters": clusters,
        "base_online": base_online,
        "step_time": step_time,
        "bandwidth": bandwidth,
        "affected": affected,
    }


def _reverse_within(values: np.ndarray, affected: np.ndarray) -> np.ndarray:
    result = values.copy()
    ids = np.flatnonzero(affected)
    order = ids[np.argsort(values[ids], kind="stable")]
    result[order] = values[order[::-1]]
    return result


def prepare_trace(
    dataset_id: str,
    scenario_id: str,
    seed: int,
    rounds: int | None = None,
    force: bool = False,
) -> dict[str, object]:
    spec = DATASETS[dataset_id]
    rounds = int(rounds or spec.round_budget)
    output = trace_asset_path(dataset_id, scenario_id, seed, rounds)
    meta_output = trace_meta_path(dataset_id, scenario_id, seed, rounds)
    if output.exists() and meta_output.exists() and not force:
        return json.loads(meta_output.read_text(encoding="utf-8"))

    profile = _base_profiles(dataset_id, seed)
    clusters = profile["clusters"]
    base_online = profile["base_online"]
    base_step = profile["step_time"]
    base_bandwidth = profile["bandwidth"]
    affected = profile["affected"]
    event_round = event_round_for(scenario_id, rounds)

    round_index = np.arange(1, rounds + 1, dtype=np.int32)
    online_probability = np.repeat(base_online[None, :], rounds, axis=0)
    if scenario_id in {"S1", "S4"}:
        period = max(4, int(round(0.25 * rounds)))
        phases = clusters.astype(np.float64) * (np.pi / 2.0)
        wave = np.sin(2.0 * np.pi * (round_index[:, None] - 1) / period + phases[None, :])
        online_probability = np.clip(base_online[None, :] + 0.35 * wave, 0.05, 0.995)

    dataset_number = int(dataset_id[1:])
    availability_rng = np.random.default_rng(seed * 13 + dataset_number * 9973)
    uniforms = availability_rng.random((rounds, NUM_CLIENTS))
    available = uniforms < online_probability

    step_before = np.repeat(base_step[None, :], rounds, axis=0)
    step_after = step_before.copy()
    bandwidth_before = np.repeat(base_bandwidth[None, :], rounds, axis=0)
    bandwidth_after = bandwidth_before.copy()
    if scenario_id in {"S3", "S4"} and event_round is not None:
        reversed_step = _reverse_within(base_step, affected)
        reversed_bandwidth = _reverse_within(base_bandwidth, affected)
        start = max(0, event_round - 1)
        step_after[start:, affected] = reversed_step[affected]
        bandwidth_after[start:, affected] = reversed_bandwidth[affected]

    data_state = np.zeros((rounds, NUM_CLIENTS), dtype=np.int8)
    if scenario_id in {"S2", "S4"} and event_round is not None:
        start = max(0, event_round - 1)
        data_state[start:, affected] = 1

    model = build_model(spec.model_name, spec.num_classes)
    payload = model_payload_bytes(model)
    del model
    integrated_step = spec.local_steps * (0.4 * step_before + 0.6 * step_after)
    integrated_upload = payload * 8.0 / np.maximum(1.0, 0.4 * bandwidth_before + 0.6 * bandwidth_after)
    predicted = integrated_step + integrated_upload
    deadline = np.empty(rounds, dtype=np.float64)
    for idx in range(rounds):
        roster = predicted[idx, available[idx]]
        if len(roster) == 0:
            roster = predicted[idx]
        deadline[idx] = float(np.quantile(roster, 0.75) + 0.05)

    trace_hash = hash_arrays(
        [
            online_probability.astype(np.float32),
            available.astype(np.uint8),
            step_before.astype(np.float32),
            step_after.astype(np.float32),
            bandwidth_before.astype(np.float64),
            bandwidth_after.astype(np.float64),
            deadline.astype(np.float32),
            data_state,
            clusters,
            affected.astype(np.uint8),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(
        temp,
        online_probability=online_probability.astype(np.float32),
        available=available.astype(np.uint8),
        step_before=step_before.astype(np.float32),
        step_after=step_after.astype(np.float32),
        bandwidth_before=bandwidth_before.astype(np.float64),
        bandwidth_after=bandwidth_after.astype(np.float64),
        deadline=deadline.astype(np.float32),
        data_state=data_state,
        clusters=clusters,
        affected=affected.astype(np.uint8),
    )
    temp.replace(output)
    meta: dict[str, object] = {
        "source_kind": "TRACE",
        "generator_version": TRACE_GENERATOR_VERSION,
        "dataset_id": dataset_id,
        "scenario_id": scenario_id,
        "severity": "base",
        "trace_seed": seed,
        "round_budget": rounds,
        "event_round": event_round,
        "within_round_checkpoint_fraction": 0.4 if scenario_id in {"S3", "S4"} else None,
        "affected_fraction": float(affected.mean()) if scenario_id in {"S2", "S3", "S4"} else (1.0 if scenario_id == "S1" else 0.0),
        "trace_hash": trace_hash,
        "asset_path": str(output),
        "payload_bytes_for_deadline": payload,
        "generated_utc": utc_now(),
    }
    atomic_json(meta_output, meta)
    return meta


@dataclass
class Trace:
    dataset_id: str
    scenario_id: str
    seed: int
    rounds: int
    trace_hash: str
    event_round: int | None
    online_probability: np.ndarray
    available: np.ndarray
    step_before: np.ndarray
    step_after: np.ndarray
    bandwidth_before: np.ndarray
    bandwidth_after: np.ndarray
    deadline: np.ndarray
    data_state: np.ndarray
    clusters: np.ndarray
    affected: np.ndarray

    @classmethod
    def load(cls, dataset_id: str, scenario_id: str, seed: int, rounds: int | None = None) -> "Trace":
        rounds = int(rounds or DATASETS[dataset_id].round_budget)
        meta = prepare_trace(dataset_id, scenario_id, seed, rounds)
        with np.load(trace_asset_path(dataset_id, scenario_id, seed, rounds), allow_pickle=False) as archive:
            return cls(
                dataset_id=dataset_id,
                scenario_id=scenario_id,
                seed=seed,
                rounds=rounds,
                trace_hash=str(meta["trace_hash"]),
                event_round=meta["event_round"],
                online_probability=archive["online_probability"].copy(),
                available=archive["available"].astype(bool),
                step_before=archive["step_before"].copy(),
                step_after=archive["step_after"].copy(),
                bandwidth_before=archive["bandwidth_before"].copy(),
                bandwidth_after=archive["bandwidth_after"].copy(),
                deadline=archive["deadline"].copy(),
                data_state=archive["data_state"].copy(),
                clusters=archive["clusters"].copy(),
                affected=archive["affected"].astype(bool),
            )

    def data_state_id(self, round_number: int, client_id: int) -> str:
        return "post_swap" if self.data_state[round_number - 1, client_id] else "pre"

    def simulated_times(self, round_number: int, client_ids: np.ndarray, local_steps: int, payload_bytes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = round_number - 1
        step = local_steps * (0.4 * self.step_before[idx, client_ids] + 0.6 * self.step_after[idx, client_ids])
        bandwidth = 0.4 * self.bandwidth_before[idx, client_ids] + 0.6 * self.bandwidth_after[idx, client_ids]
        upload = payload_bytes * 8.0 / np.maximum(1.0, bandwidth)
        total = step + upload
        return step.astype(np.float64), upload.astype(np.float64), total.astype(np.float64)


def trace_frame(trace: Trace) -> pd.DataFrame:
    fractions = (0.0, 0.4, 1.0) if trace.scenario_id in {"S3", "S4"} else (0.0,)
    frames: list[pd.DataFrame] = []
    rounds = np.repeat(np.arange(1, trace.rounds + 1, dtype=np.int32), NUM_CLIENTS)
    clients = np.tile(np.arange(NUM_CLIENTS, dtype=np.int16), trace.rounds)
    online = trace.online_probability.reshape(-1)
    available = trace.available.reshape(-1)
    deadline = np.repeat(trace.deadline, NUM_CLIENTS)
    state = np.where(trace.data_state.reshape(-1) == 1, "post_swap", "pre")
    cluster = np.tile(trace.clusters, trace.rounds)
    for fraction in fractions:
        after = fraction >= 0.4
        step_values = trace.step_after.reshape(-1) if after else trace.step_before.reshape(-1)
        bandwidth_values = trace.bandwidth_after.reshape(-1) if after else trace.bandwidth_before.reshape(-1)
        size = trace.rounds * NUM_CLIENTS
        frames.append(
            pd.DataFrame(
                {
                    "schema_version": np.repeat("2.1.0", size),
                    "source_kind": np.repeat("TRACE", size),
                    "trace_hash": np.repeat(trace.trace_hash, size),
                    "dataset_id": np.repeat(trace.dataset_id, size),
                    "scenario_id": np.repeat(trace.scenario_id, size),
                    "severity": np.repeat("base", size),
                    "trace_seed": np.repeat(trace.seed, size),
                    "round": rounds,
                    "within_round_fraction": np.repeat(fraction, size),
                    "client_id": clients,
                    "online_probability": online,
                    "available_realization": available,
                    "step_time_s": step_values,
                    "bandwidth_bps": bandwidth_values,
                    "deadline_s": deadline,
                    "data_state_id": state,
                    "client_cluster": cluster,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def dynamic_event_rows(trace: Trace) -> list[dict[str, object]]:
    if trace.event_round is None:
        return []
    event_time = float(trace.deadline[: max(0, trace.event_round - 1)].sum())
    base = {
        "source_kind": "TRACE",
        "dataset_id": trace.dataset_id,
        "scenario_id": trace.scenario_id,
        "severity": "base",
        "trace_seed": trace.seed,
        "event_round": int(trace.event_round),
        "event_time_s": event_time,
        "affected_fraction": float(trace.affected.mean()) if trace.scenario_id in {"S2", "S3", "S4"} else 1.0,
        "trace_hash": trace.trace_hash,
    }
    if trace.scenario_id == "S1":
        events = [("availability_phase", None, True, {"period_fraction": 0.25, "amplitude": 0.35})]
    elif trace.scenario_id == "S2":
        events = [("label_or_mixture_swap", None, True, {"swap_fraction": 0.5})]
    elif trace.scenario_id == "S3":
        events = [
            ("compute_reversal", 0.4, True, {"compute_ratio": 4.0}),
            ("bandwidth_reversal", 0.4, False, {"bandwidth_ratio": 8.0}),
        ]
    else:
        events = [("compound_boundary", 0.4, True, {"availability_amplitude": 0.35, "swap_fraction": 0.5, "compute_ratio": 4.0, "bandwidth_ratio": 8.0})]
    rows = []
    for event_type, fraction, anchor, parameters in events:
        value = dict(base)
        value.update(
            {
                "event_id": f"{trace.dataset_id}-{trace.scenario_id}-{event_type}-r{trace.event_round}",
                "within_round_fraction": fraction,
                "event_type": event_type,
                "auc20_anchor": bool(anchor),
                "parameters_json": canonical_json(parameters),
            }
        )
        rows.append(value)
    return rows


def write_formal_trace_tables(dataset_ids: Iterable[str], scenario_ids: Iterable[str], seed: int, output_root: Path) -> None:
    events: list[dict[str, object]] = []
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "scenario_trace.parquet"
    temp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    writer: pq.ParquetWriter | None = None
    for dataset_id in dataset_ids:
        for scenario_id in scenario_ids:
            trace = Trace.load(dataset_id, scenario_id, seed)
            table = pa.Table.from_pandas(trace_frame(trace), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp, table.schema, compression="snappy")
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)
            events.extend(dynamic_event_rows(trace))
    if writer is None:
        raise RuntimeError("No scenario traces were requested")
    writer.close()
    os.replace(temp, output)
    event_frame = pd.DataFrame(events)
    if "schema_version" not in event_frame.columns:
        event_frame.insert(0, "schema_version", "2.1.0")
    atomic_csv(output_root / "dynamic_events.csv", event_frame)
