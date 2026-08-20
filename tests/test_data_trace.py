import numpy as np

from r2c_baselines.config import DEV_SEED
from r2c_baselines.data import FederatedData, prepare_partition
from r2c_baselines.traces import Trace, prepare_trace


def test_fashion_partition_is_disjoint_and_validation_is_held_out():
    prepare_partition("D1", DEV_SEED)
    data = FederatedData.load("D1", DEV_SEED)
    combined = np.concatenate(data.clients_pre)
    assert len(combined) == len(np.unique(combined))
    assert len(set(combined).intersection(set(data.validation_indices))) == 0
    assert set(combined) == set(data.train_indices)
    assert min(map(len, data.clients_pre)) >= 10


def test_resource_reversal_starts_at_half_budget_and_preserves_rank_multiset():
    prepare_trace("D1", "S3", DEV_SEED, rounds=100)
    trace = Trace.load("D1", "S3", DEV_SEED, rounds=100)
    assert trace.event_round == 50
    before = trace.step_before[49, trace.affected]
    after = trace.step_after[49, trace.affected]
    assert np.allclose(np.sort(before), np.sort(after))
    assert not np.allclose(before, after)
    assert np.allclose(trace.step_before[48], trace.step_after[48])
    assert np.isclose(trace.online_probability.min(), trace.online_probability.min())


def test_trace_hash_is_reused_for_same_frozen_asset():
    first = prepare_trace("D1", "S4", DEV_SEED, rounds=100)
    second = prepare_trace("D1", "S4", DEV_SEED, rounds=100)
    assert first["trace_hash"] == second["trace_hash"]


def test_data_swap_and_trace_use_identical_affected_clients():
    data = FederatedData.load("D1", DEV_SEED)
    prepare_trace("D1", "S4", DEV_SEED, rounds=100, force=True)
    trace = Trace.load("D1", "S4", DEV_SEED, rounds=100)
    mapping_affected = data.drift_mapping != np.arange(100)
    assert np.array_equal(mapping_affected, trace.affected)
