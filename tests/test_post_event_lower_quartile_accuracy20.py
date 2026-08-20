from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from r2c_baselines.r2c_post_event_lower_quartile_accuracy20 import (
    build_report,
    derive_lqa20_percent,
)


def _frame(pre: np.ndarray, post: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_offset_round": np.r_[np.arange(-20, 0), np.arange(1, 21)],
            "auc20_window_role": ["pre20"] * 20 + ["post20"] * 20,
            "test_accuracy": np.r_[pre, post],
        }
    )


def test_lqa20_uses_linear_lower_quartile_as_percent() -> None:
    pre = np.full(20, 0.90)
    post = np.linspace(0.40, 0.59, 20)
    assert derive_lqa20_percent(_frame(pre, post)) == pytest.approx(44.75)


def test_lqa20_is_independent_of_pre_window_level() -> None:
    post = np.linspace(0.70, 0.89, 20)
    low_pre = _frame(np.full(20, 0.20), post)
    high_pre = _frame(np.full(20, 0.95), post)
    assert derive_lqa20_percent(low_pre) == pytest.approx(74.75)
    assert derive_lqa20_percent(high_pre) == pytest.approx(74.75)


def test_lqa20_requires_the_exact_complete_window() -> None:
    frame = _frame(np.full(20, 0.50), np.full(20, 0.53)).iloc[:-1]
    with pytest.raises(ValueError, match="post20 must contain exact offsets"):
        derive_lqa20_percent(frame)


def test_real_report_has_full_lineage_and_32_unique_table2_cells() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    report = build_report(repo_root)
    assert report["formal_run_mutations"] is False
    assert report["formal_gate_mutations"] is False
    assert report["audit"]["all_windows_complete"] is True
    assert report["audit"]["derived_records"] == 74
    assert report["audit"]["table2_lqa_unique_at_0.01"] == 32
    stats = report["selection_comparison_on_table2_32_cells"]["lqa20_percent"]
    assert stats["minimum_exact_pair_gap"] == pytest.approx(0.03)
