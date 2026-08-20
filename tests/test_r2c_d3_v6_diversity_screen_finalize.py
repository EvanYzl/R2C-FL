from __future__ import annotations

import pandas as pd

from r2c_baselines import r2c_d3_v6_diversity_screen_finalize as finalize


def test_rank_with_overall_preserves_frozen_order_and_adds_serialization_rank() -> None:
    frame = pd.DataFrame(
        [
            {
                "history_mix": 0.15,
                "last50_validation_accuracy": 0.90,
                "recovery_deficit_auc20": 0.002,
                "target_hit_round": 80,
                "final_participation_jfi": 0.90,
                "final_worst10_participation": 31.0,
                "complete": True,
            },
            {
                "history_mix": 0.30,
                "last50_validation_accuracy": 0.89,
                "recovery_deficit_auc20": 0.000,
                "target_hit_round": 70,
                "final_participation_jfi": 0.91,
                "final_worst10_participation": 32.0,
                "complete": True,
            },
            {
                "history_mix": 0.45,
                "last50_validation_accuracy": 0.88,
                "recovery_deficit_auc20": 0.001,
                "target_hit_round": 90,
                "final_participation_jfi": 0.92,
                "final_worst10_participation": 33.0,
                "complete": True,
            },
            {
                "history_mix": 0.60,
                "last50_validation_accuracy": 0.95,
                "recovery_deficit_auc20": 0.000,
                "target_hit_round": 50,
                "final_participation_jfi": 0.80,
                "final_worst10_participation": 10.0,
                "complete": True,
            },
        ]
    )
    expected = finalize.FROZEN_RANKER(frame)
    ranked = finalize.rank_with_overall(frame)
    assert ranked["history_mix"].tolist() == expected["history_mix"].tolist()
    assert ranked["rank_sum"].tolist() == expected["rank_sum"].tolist()
    assert ranked["overall_rank"].tolist() == list(range(1, len(ranked) + 1))
    assert 0.60 not in set(ranked["history_mix"])


def test_rank_with_overall_handles_no_eligible_candidates() -> None:
    frame = pd.DataFrame(
        [
            {
                "history_mix": 0.15,
                "last50_validation_accuracy": 0.90,
                "recovery_deficit_auc20": 0.0,
                "target_hit_round": 80,
                "final_participation_jfi": 0.50,
                "final_worst10_participation": 5.0,
                "complete": True,
            }
        ]
    )
    ranked = finalize.rank_with_overall(frame)
    assert ranked.empty
    assert "overall_rank" in ranked.columns
