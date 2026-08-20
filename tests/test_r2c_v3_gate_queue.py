from __future__ import annotations

from r2c_baselines.r2c_d2_v3_gate_queue import TARGET_ACCURACY, build_manifest
from r2c_baselines.r2c_v3 import PROTOCOL_VERSION


def test_v3_gate_is_two_candidate_paired_validation_only() -> None:
    manifest = build_manifest(persist=False)
    jobs = manifest["jobs"]
    assert len(jobs) == 4
    assert len({job["job_id"] for job in jobs}) == 4
    assert manifest["formal_test_access"] is False
    assert manifest["target_accuracy"] == TARGET_ACCURACY
    assert {job["variant_label"] for job in jobs} == {"V5F085", "F075"}
    for label in {job["variant_label"] for job in jobs}:
        pair = [job for job in jobs if job["variant_label"] == label]
        assert {job["scenario_id"] for job in pair} == {"S0", "S4"}
    for job in jobs:
        assert job["dataset_id"] == "D2"
        assert job["rounds"] == 400
        assert job["mode"] == "calibration"
        assert job["evaluation_split"] == "validation"
        assert job["full_logging"] is True
        assert job["method_config"]["r2c_protocol_version"] == PROTOCOL_VERSION
        assert job["method_config"]["r2c_v2_audit_replay"] is False
