from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from r2c_baselines.r2c_post_event_tail_accuracy20 import derive_pta20_percent


def _frame(pre: np.ndarray, post: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_offset_round": np.r_[np.arange(-20, 0), np.arange(1, 21)],
            "auc20_window_role": ["pre20"] * 20 + ["post20"] * 20,
            "test_accuracy": np.r_[pre, post],
        }
    )


def test_pta20_uses_five_lowest_post_rounds_as_percent() -> None:
    pre = np.full(20, 0.90)
    post = np.linspace(0.40, 0.59, 20)
    assert derive_pta20_percent(_frame(pre, post)) == pytest.approx(42.0)


def test_pta20_is_independent_of_pre_window_level() -> None:
    post = np.full(20, 0.73)
    low_pre = _frame(np.full(20, 0.20), post)
    high_pre = _frame(np.full(20, 0.95), post)
    assert derive_pta20_percent(low_pre) == pytest.approx(73.0)
    assert derive_pta20_percent(high_pre) == pytest.approx(73.0)


def test_pta20_rejects_invalid_tail_size() -> None:
    frame = _frame(np.full(20, 0.50), np.full(20, 0.53))
    with pytest.raises(ValueError, match="tail_k"):
        derive_pta20_percent(frame, tail_k=0)
