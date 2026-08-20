from __future__ import annotations

from r2c_baselines.r2c_d2_v3_screen_queue import build_manifest
from r2c_baselines.r2c_v3 import PROTOCOL_VERSION


def test_v3_screen_is_eight_job_validation_only_queue() -> None:
    manifest = build_manifest(persist=False)
    jobs = manifest["jobs"]
    assert len(jobs) == 8
    assert len({job["job_id"] for job in jobs}) == 8
    assert manifest["formal_test_access"] is False
    assert manifest["selection_count"] == 2
    assert manifest["eligibility"] == {
        "recovery_deficit_auc20_lte": 0.002562,
        "last50_validation_accuracy_gte": 0.485,
    }
    for job in jobs:
        assert job["dataset_id"] == "D2"
        assert job["scenario_id"] == "S4"
        assert job["rounds"] == 200
        assert job["mode"] == "calibration"
        assert job["evaluation_split"] == "validation"
        assert job["seed"] == 20260810
        assert job["full_logging"] is True
        assert job["method_config"]["r2c_protocol_version"] == PROTOCOL_VERSION
        assert job["method_config"]["r2c_v2_audit_replay"] is False
        assert (
            "r2c_v3_fixed_server_alpha" in job["method_config"]
            or job["method_config"]["r2c_v3_min_server_alpha"] in {0.25, 0.50}
        )
