import numpy as np

from r2c_baselines.config import BASELINES, DEFAULT_METHOD_CONFIG
from r2c_baselines.methods import BaselineAdapter


def test_all_adapters_respect_current_roster_and_k():
    counts = np.arange(1, 101)
    available = np.arange(17, 77)
    duration = np.linspace(0.1, 2.0, 100)
    for method in BASELINES:
        adapter = BaselineAdapter(method, counts, 123, 100, DEFAULT_METHOD_CONFIG[method])
        selection = adapter.admit(available, 1, duration)
        if method == "PowerOfChoice":
            losses = {int(client): float(client) for client in selection.admitted}
            selection = adapter.finish_power_of_choice(selection, losses)
        assert set(selection.admitted).issubset(set(available))
        assert set(selection.selected).issubset(set(available))
        assert len(selection.selected) <= 10
        assert len(selection.admitted) <= 20


def test_f3ast_frequency_update_and_importance_coefficients():
    counts = np.ones(100)
    adapter = BaselineAdapter("F3AST", counts, 123, 100, {"f3ast_beta": 0.001})
    before = adapter.r.copy()
    selection = adapter.admit(np.arange(100), 1, np.ones(100))
    assert len(selection.selected) == 10
    assert not np.array_equal(before, adapter.r)
    coefficients = adapter.aggregation_coefficients(selection.selected)
    assert np.all(np.isfinite(coefficients))
    assert np.all(coefficients > 0)


def test_shared_uniform_selection_for_fedavg_fedau_fedawe():
    counts = np.arange(1, 101)
    available = np.arange(100)
    selected = []
    for method in ("FedAvg", "FedAU", "FedAWE"):
        adapter = BaselineAdapter(method, counts, 42, 100, DEFAULT_METHOD_CONFIG[method])
        selected.append(adapter.admit(available, 7, np.ones(100)).selected.tolist())
    assert selected[0] == selected[1] == selected[2]

