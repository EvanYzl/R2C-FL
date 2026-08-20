import numpy as np

from r2c_baselines.data import FederatedData
from r2c_baselines.r2c import (
    conditional_inclusion_targets,
    pivotal_sample,
    project_capped_simplex,
)
from r2c_baselines.r2c_v2 import _score_from_anchor


def test_r2c_capped_simplex_and_pivotal_sample_are_exact_size():
    scores = np.linspace(-1.0, 1.0, 20)
    floor = 0.05 * 10 / 20
    targets = conditional_inclusion_targets(scores, 10, 0.5, floor)
    assert np.isfinite(targets).all()
    assert np.all(targets >= floor - 1e-12)
    assert np.all(targets <= 1.0 + 1e-12)
    assert abs(float(targets.sum()) - 10.0) <= 1e-12
    selected = pivotal_sample(targets, 10, np.random.default_rng(1234))
    assert len(selected) == 10
    assert len(np.unique(selected)) == 10


def test_r2c_projection_repairs_caps_without_changing_total():
    values = np.asarray([10.0, 9.0, 8.0, 0.1, 0.1, 0.1])
    result = project_capped_simplex(values, total=3.0, lower=0.05, upper=1.0)
    assert abs(float(result.sum()) - 3.0) <= 1e-12
    assert np.all(result >= 0.05 - 1e-12)
    assert np.all(result <= 1.0 + 1e-12)


def test_r2c_heldout_folds_are_disjoint_from_local_training_and_each_other():
    data = FederatedData.load("D1", 20260811)
    validation = set(data.validation_indices.tolist())
    training = set(data.train_indices.tolist())
    assert not validation & training
    observed = []
    for client_id in range(100):
        fold0 = data.heldout_indices_for(client_id, "pre", 0)
        fold1 = data.heldout_indices_for(client_id, "pre", 1)
        assert not set(fold0.tolist()) & set(fold1.tolist())
        observed.extend(fold0.tolist())
        observed.extend(fold1.tolist())
    assert len(observed) == len(set(observed)) == len(validation)
    assert set(observed) == validation


def test_r2c_v2_shared_anchors_are_disjoint_balanced_and_deterministic():
    data = FederatedData.load("D1", 20260811)
    first = data.anchor_indices(0, 20)
    repeated = data.anchor_indices(0, 20)
    second = data.anchor_indices(1, 20)
    assert np.array_equal(first, repeated)
    assert not set(first.tolist()) & set(second.tolist())
    assert set(first.tolist()).issubset(set(data.validation_indices.tolist()))
    assert set(second.tolist()).issubset(set(data.validation_indices.tolist()))
    for indices in (first, second):
        counts = np.bincount(data.train_labels[indices], minlength=10)
        assert int(counts.max() - counts.min()) <= 1


def test_r2c_v2_anchor_score_is_finite_and_fold_agreement_is_bounded():
    base0 = np.asarray([1.0, 1.2, 0.8, 1.1])
    base1 = np.asarray([0.9, 1.1, 0.7, 1.0])
    current0 = np.stack([base0 - value for value in np.linspace(-0.1, 0.2, 20)])
    current1 = np.stack([base1 - value for value in np.linspace(-0.05, 0.18, 20)])
    finish = np.linspace(0.2, 0.99, 20)
    result = _score_from_anchor(
        base0,
        base1,
        current0,
        current1,
        finish,
        scale_floor=1.0e-4,
        value_clip=4.0,
        finish_weight=0.75,
        radius_multiplier=1.0,
    )
    for name in ("raw_gain", "value_hat", "score", "score_variance", "radius"):
        assert np.isfinite(np.asarray(result[name])).all()
    assert 0.0 <= float(result["agreement"]) <= 1.0
