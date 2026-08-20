import numpy as np

from r2c_baselines.metrics import participation_jfi, recovery_auc20, worst10_participation


def test_recovery_auc20_uses_exact_twenty_rounds():
    rounds = np.arange(1, 61)
    accuracy = np.full(60, 0.8)
    accuracy[30:50] = 0.7  # event=30; post rounds 31..50
    result = recovery_auc20(rounds, accuracy, 30)
    assert result["recovery_auc20_complete"] is True
    assert np.isclose(result["pre_event_accuracy"], 0.8)
    assert np.isclose(result["recovery_deficit_auc20"], 0.1)
    assert np.isclose(result["post_event_round20_accuracy"], 0.7)


def test_recovery_auc20_missing_window_is_null_not_zero():
    result = recovery_auc20(range(1, 31), np.full(30, 0.5), 20)
    assert result["recovery_auc20_complete"] is False
    assert result["recovery_deficit_auc20"] is None
    assert "missing_pre_rounds" in result["recovery_missing_reason"]
    assert "missing_post_rounds" in result["recovery_missing_reason"]


def test_fairness_metrics():
    assert participation_jfi([1, 1, 1, 1]) == 1.0
    assert participation_jfi([0, 0]) == 0.0
    assert worst10_participation(range(100)) == 4.5

