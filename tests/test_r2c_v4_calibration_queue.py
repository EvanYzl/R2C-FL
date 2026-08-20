from __future__ import annotations

import pandas as pd

from r2c_baselines.r2c_d2_v4_calibration_queue import _select_candidate_frame


def test_v4_candidate_order_requires_accuracy_then_auc_and_tiebreaks() -> None:
    frame = pd.DataFrame(
        [
            {"deployment_beta": 0.80, "last50_validation_accuracy": 0.670, "recovery_deficit_auc20": 0.0020},
            {"deployment_beta": 0.90, "last50_validation_accuracy": 0.671, "recovery_deficit_auc20": 0.0010},
            {"deployment_beta": 0.95, "last50_validation_accuracy": 0.650, "recovery_deficit_auc20": 0.0000},
        ]
    )
    ranked = _select_candidate_frame(frame)
    assert ranked["deployment_beta"].tolist() == [0.90, 0.80]
