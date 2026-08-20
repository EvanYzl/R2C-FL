from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from r2c_baselines.r2c_tail_recovery_margin20 import (
    derive_window_metrics,
    exact_event_windows,
)


def _frame(pre: np.ndarray, post: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_offset_round": np.r_[np.arange(-20, 0), np.arange(1, 21)],
            "auc20_window_role": ["pre20"] * 20 + ["post20"] * 20,
            "test_accuracy": np.r_[pre, post],
        }
    )


def test_trm20_uses_five_lowest_post_rounds_without_clipping() -> None:
    pre = np.full(20, 0.50)
    post = np.linspace(0.40, 0.59, 20)
    metrics = derive_window_metrics(_frame(pre, post))
    assert metrics["trm20_pp"] == pytest.approx(-8.0)
    assert metrics["signed_delta20_pp"] == pytest.approx(-0.5)


def test_trm20_preserves_positive_margin() -> None:
    pre = np.full(20, 0.50)
    post = np.full(20, 0.53)
    metrics = derive_window_metrics(_frame(pre, post))
    assert metrics["trm20_pp"] == pytest.approx(3.0)


def test_exact_windows_reject_missing_offset() -> None:
    frame = _frame(np.full(20, 0.50), np.full(20, 0.53)).iloc[:-1]
    with pytest.raises(ValueError, match="post20"):
        exact_event_windows(frame)

