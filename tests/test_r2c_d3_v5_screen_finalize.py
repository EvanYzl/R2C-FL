from __future__ import annotations

import pandas as pd

from r2c_baselines import r2c_d3_v5_screen_finalize as finalize


def test_rank_with_overall_preserves_frozen_order_and_adds_serialization_rank() -> None:
    frame = pd.DataFrame(
        [
            {"alpha": 1.0, "beta": 0.0, "last50_validation_accuracy": 0.90, "recovery_deficit_auc20": 0.002, "target_hit_round": 80, "complete": True},
            {"alpha": 0.875, "beta": 0.8, "last50_validation_accuracy": 0.89, "recovery_deficit_auc20": 0.000, "target_hit_round": 70, "complete": True},
            {"alpha": 0.75, "beta": 0.95, "last50_validation_accuracy": 0.88, "recovery_deficit_auc20": 0.001, "target_hit_round": 90, "complete": True},
        ]
    )
    ranked = finalize.rank_with_overall(frame)
    assert ranked[["alpha", "beta"]].to_records(index=False).tolist() == [
        (0.875, 0.8),
        (1.0, 0.0),
        (0.75, 0.95),
    ]
    assert ranked["overall_rank"].tolist() == [1, 2, 3]
