from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from r2c_baselines import r2c_v5
from r2c_baselines.r2c import conditional_inclusion_targets, history_balanced_conditional_targets


def test_history_balanced_targets_preserve_exact_k_and_bounds() -> None:
    scores = np.linspace(-2.0, 2.0, 20)
    counts = np.asarray([0, 1, 2, 3, 5] * 4, dtype=np.int64)
    anchor, history, final, history_scores = history_balanced_conditional_targets(
        scores, counts, 10, 0.35, 1.0, 0.1, 0.45
    )
    assert np.isclose(anchor.sum(), 10.0, atol=1.0e-12)
    assert np.isclose(history.sum(), 10.0, atol=1.0e-12)
    assert np.isclose(final.sum(), 10.0, atol=1.0e-12)
    assert np.all((final >= 0.1 - 1.0e-12) & (final <= 1.0 + 1.0e-12))
    assert np.allclose(final, 0.55 * anchor + 0.45 * history, atol=1.0e-12, rtol=0)
    assert np.allclose(history_scores, -np.log1p(counts), atol=0, rtol=0)


def test_zero_mix_is_exact_anchor_control() -> None:
    scores = np.linspace(-1.0, 1.0, 20)
    counts = np.arange(20, dtype=np.int64)
    expected = conditional_inclusion_targets(scores, 10, 0.35, 0.1)
    anchor, _, final, _ = history_balanced_conditional_targets(
        scores, counts, 10, 0.35, 1.0, 0.1, 0.0
    )
    assert np.array_equal(anchor, expected)
    assert np.array_equal(final, expected)


def test_history_target_prefers_less_selected_client_at_equal_utility() -> None:
    scores = np.zeros(20, dtype=np.float64)
    counts = np.asarray([0, 10] + [5] * 18, dtype=np.int64)
    _, history, final, _ = history_balanced_conditional_targets(
        scores, counts, 10, 0.35, 1.0, 0.1, 0.60
    )
    assert history[0] > history[1]
    assert final[0] > final[1]


def test_history_config_validation() -> None:
    assert r2c_v5.validated_history_config(
        {"r2c_v5_history_mix": 0.3, "r2c_v5_history_temperature": 1.0}
    ) == (0.3, 1.0)
    for config in (
        {"r2c_v5_history_mix": -0.1},
        {"r2c_v5_history_mix": 1.1},
        {"r2c_v5_history_temperature": 0.0},
    ):
        try:
            r2c_v5.validated_history_config(config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid config to fail: {config}")


def test_v5_wrapper_passes_history_and_relabels_protocol(monkeypatch) -> None:
    captured: dict[str, object] = {}
    result = SimpleNamespace(
        checkpoint_rows=[{"protocol_version": "old"}],
        certificate_row={"protocol_version": "old", "certificate_record_hash": "old"},
    )

    def fake_v4(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(r2c_v5, "run_r2c_v4_round", fake_v4)
    counts = np.arange(100, dtype=np.int64)
    returned = r2c_v5.run_r2c_v5_round(
        model=object(),
        trainer=object(),
        data=object(),
        trace=object(),
        round_number=3,
        available=np.arange(20),
        sample_counts=np.ones(100),
        learning_rate=0.1,
        payload_bytes=10,
        seed=7,
        config={"r2c_v5_history_mix": 0.3, "r2c_v5_history_temperature": 1.0},
        run_id="test",
        round_start_model_hash="0" * 64,
        full_logging=True,
        selection_history_counts=counts,
    )
    assert returned is result
    assert np.array_equal(captured["selection_history_counts"], counts)
    assert result.checkpoint_rows[0]["protocol_version"] == r2c_v5.PROTOCOL_VERSION
    assert result.checkpoint_rows[0]["selection_protocol_version"] == r2c_v5.PROTOCOL_VERSION
    assert result.certificate_row["protocol_version"] == r2c_v5.PROTOCOL_VERSION
    assert result.certificate_row["selection_history_labels_used"] is False
    assert result.certificate_row["certificate_record_hash"] != "old"
