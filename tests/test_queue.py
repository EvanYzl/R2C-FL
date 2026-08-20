from r2c_baselines.queue import build_manifest
from r2c_baselines.r2c_table1_queue import build_manifest as build_r2c_manifest
from r2c_baselines.r2c_d2_opt_queue import build_manifest as build_d2_opt_manifest
from r2c_baselines.r2c_d2_gate_queue import build_manifest as build_d2_gate_manifest
from r2c_baselines.r2c_d2_fullval_queue import build_manifest as build_d2_fullval_manifest
from r2c_baselines.r2c_d2_formal_queue import build_manifest as build_d2_formal_manifest


def test_baseline_queue_cardinality():
    manifest = build_manifest(persist=False)
    formal = [job for job in manifest["jobs"] if job["stage"] == "formal"]
    calibration = [job for job in manifest["jobs"] if job["stage"] == "calibration"]
    assert len(formal) == 77
    assert len(calibration) == 84
    assert len({job["job_id"] for job in manifest["jobs"]}) == len(manifest["jobs"])


def test_r2c_table1_queue_cardinality_and_label_isolation():
    manifest = build_r2c_manifest(persist=False, force=True)
    counts = {
        stage: sum(job["stage"] == stage for job in manifest["jobs"])
        for stage in ("norm_pilot", "calibration", "formal")
    }
    assert counts == {"norm_pilot": 4, "calibration": 12, "formal": 8}
    assert len({job["job_id"] for job in manifest["jobs"]}) == 24
    for job in manifest["jobs"]:
        expected = "test" if job["stage"] == "formal" else "validation"
        assert job["evaluation_split"] == expected
        assert job["seed"] == (20260811 if job["stage"] == "formal" else 20260810)


def test_r2c_d2_optimization_screen_is_validation_only():
    manifest = build_d2_opt_manifest(persist=False, force=True)
    assert len(manifest["jobs"]) == 6
    assert len({job["job_id"] for job in manifest["jobs"]}) == 6
    assert manifest["formal_test_access"] is False
    for job in manifest["jobs"]:
        assert job["dataset_id"] == "D2"
        assert job["scenario_id"] == "S4"
        assert job["evaluation_split"] == "validation"
        assert job["seed"] == 20260810
        assert job["method_config"]["r2c_protocol_version"] == "anchor-bounded-overhead-v2"


def test_r2c_d2_paired_gate_uses_only_frozen_validation_winners():
    manifest = build_d2_gate_manifest(persist=False, force=True)
    assert len(manifest["jobs"]) == 4
    assert len({job["job_id"] for job in manifest["jobs"]}) == 4
    assert manifest["formal_test_access"] is False
    assert {job["scenario_id"] for job in manifest["jobs"]} == {"S0", "S4"}
    assert len({job["variant_index"] for job in manifest["jobs"]}) == 2
    for job in manifest["jobs"]:
        assert job["dataset_id"] == "D2"
        assert job["evaluation_split"] == "validation"
        assert job["rounds"] == 400
        assert job["seed"] == 20260810


def test_r2c_d2_full_validation_is_a_single_frozen_pair():
    manifest = build_d2_fullval_manifest(persist=False, force=True)
    assert len(manifest["jobs"]) == 2
    assert {job["scenario_id"] for job in manifest["jobs"]} == {"S0", "S4"}
    assert len({job["variant_index"] for job in manifest["jobs"]}) == 1
    assert manifest["formal_test_access"] is False
    for job in manifest["jobs"]:
        assert job["rounds"] == 1000
        assert job["evaluation_split"] == "validation"
        assert job["seed"] == 20260810


def test_r2c_d2_formal_is_one_test_pair_with_frozen_training_config():
    manifest = build_d2_formal_manifest(persist=False, force=True)
    assert manifest["scope"] == "D2_only_single_formal_pair"
    assert manifest["formal_jobs"] == 2
    assert manifest["required_wins"] == 3
    assert manifest["test_labels_for_selection"] is False
    assert len(manifest["baseline_thresholds"]) == 4
    assert {job["scenario_id"] for job in manifest["jobs"]} == {"S0", "S4"}
    assert len({job["variant_index"] for job in manifest["jobs"]}) == 1
    for job in manifest["jobs"]:
        assert job["dataset_id"] == "D2"
        assert job["rounds"] == 1000
        assert job["evaluation_split"] == "test"
        assert job["mode"] == "formal"
        assert job["seed"] == 20260811
        assert job["partition_seed"] == 20260811
        assert job["trace_seed"] == 20260811
        assert job["full_logging"] is True
        assert job["method_config"]["r2c_protocol_version"] == "anchor-bounded-overhead-v2"
        assert job["method_config"]["r2c_v2_audit_replay"] is True
