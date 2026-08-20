from __future__ import annotations

import numpy as np

from r2c_baselines.data import FederatedData
from r2c_baselines.r2c_v3 import _choose_alpha, _validated_alphas


def test_server_alphas_are_validated_and_sorted() -> None:
    values = _validated_alphas([0.0, 0.5, 1.0, 0.25])
    assert np.array_equal(values, np.asarray([1.0, 0.5, 0.25, 0.0]))


def test_guard_rules_are_deterministic_and_prefer_larger_ties() -> None:
    alphas = np.asarray([1.0, 0.75, 0.5, 0.25, 0.0])
    fold0 = np.asarray([1.02, 1.00, 0.98, 0.98, 1.00])
    fold1 = np.asarray([1.01, 1.00, 0.99, 0.98, 1.00])
    chosen, _ = _choose_alpha(alphas, fold0, fold1, "minimax", 0.0)
    assert alphas[chosen] == 0.25
    chosen, _ = _choose_alpha(alphas, fold0, fold1, "mean_loss", 0.0)
    assert alphas[chosen] == 0.25
    chosen, _ = _choose_alpha(alphas, fold0, fold1, "largest_noninferior", 0.0)
    assert alphas[chosen] == 0.75
    worse = np.asarray([1.04, 1.03, 1.02, 1.01, 1.00])
    chosen, _ = _choose_alpha(
        alphas, worse, worse, "minimax", 0.0, minimum_alpha=0.25
    )
    assert alphas[chosen] == 0.25


def test_guard_anchors_are_disjoint_balanced_and_deterministic() -> None:
    data = FederatedData.load("D2", 20260810)
    for fold in (0, 1):
        selection = data.anchor_indices(fold, 16)
        first = data.guard_anchor_indices(fold, 40, 16)
        second = data.guard_anchor_indices(fold, 40, 16)
        assert np.array_equal(first, second)
        assert np.intersect1d(first, selection).size == 0
        counts = np.bincount(data.train_labels[first], minlength=10)
        assert counts.min() == counts.max() == 4
