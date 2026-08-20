from __future__ import annotations

import pytest

import r2c_baselines.r2c_d3_v12_validation_pair_queue as pair
from r2c_baselines.r2c_d3_v12_validation_pair_queue import (
    BASELINE_ENVELOPE,
    GATE_MARGINS,
    evaluate_gate,
)


def _values() -> dict[str, float]:
    return {
        "s0_last50_accuracy": BASELINE_ENVELOPE["s0_last50_accuracy"] + 0.01,
        "s4_last50_accuracy": BASELINE_ENVELOPE["s4_last50_accuracy"] + 0.01,
        "s4_trm20_pp": BASELINE_ENVELOPE["s4_trm20_pp"] + 0.5,
        "s4_algorithm_tta_s": BASELINE_ENVELOPE["s4_algorithm_tta_s"] * 0.9,
    }


def test_four_strict_wins_pass() -> None:
    result = evaluate_gate(**_values())
    assert result["strict_win_count"] == 4
    assert result["misses"] == []
    assert result["gate_passed"] is True


@pytest.mark.parametrize(
    ("metric", "value"),
    (
        (
            "s0_last50_accuracy",
            BASELINE_ENVELOPE["s0_last50_accuracy"] - GATE_MARGINS["accuracy_fraction"],
        ),
        (
            "s4_last50_accuracy",
            BASELINE_ENVELOPE["s4_last50_accuracy"] - GATE_MARGINS["accuracy_fraction"],
        ),
        (
            "s4_trm20_pp",
            BASELINE_ENVELOPE["s4_trm20_pp"] - GATE_MARGINS["trm20_pp"],
        ),
        (
            "s4_algorithm_tta_s",
            BASELINE_ENVELOPE["s4_algorithm_tta_s"] * (1.0 + GATE_MARGINS["tta_relative"]),
        ),
    ),
)
def test_three_strict_and_one_boundary_close_pass(metric: str, value: float) -> None:
    values = _values()
    values[metric] = value
    result = evaluate_gate(**values)
    assert result["strict_win_count"] == 3
    assert result["misses"] == [metric]
    assert result["sole_miss_close"] is True
    assert result["gate_passed"] is True


def test_three_strict_and_one_outside_close_fails() -> None:
    values = _values()
    values["s4_trm20_pp"] = (
        BASELINE_ENVELOPE["s4_trm20_pp"] - GATE_MARGINS["trm20_pp"] - 1e-6
    )
    result = evaluate_gate(**values)
    assert result["strict_win_count"] == 3
    assert result["sole_miss_close"] is False
    assert result["gate_passed"] is False


def test_two_misses_fail_even_if_both_are_close() -> None:
    values = _values()
    values["s0_last50_accuracy"] = BASELINE_ENVELOPE["s0_last50_accuracy"]
    values["s4_last50_accuracy"] = BASELINE_ENVELOPE["s4_last50_accuracy"]
    result = evaluate_gate(**values)
    assert result["strict_win_count"] == 2
    assert result["gate_passed"] is False


def test_missing_tta_is_not_close() -> None:
    values = _values()
    values["s4_algorithm_tta_s"] = None
    result = evaluate_gate(**values)
    assert result["strict_win_count"] == 3
    assert result["within_close_margin"]["s4_algorithm_tta_s"] is False
    assert result["gate_passed"] is False


def _selected_context() -> dict[str, object]:
    config = {
        "r2c_protocol_version": pair.PROTOCOL_VERSION,
        "r2c_v4_deployment_ema_betas": [pair.ORDINARY_BETA],
        "r2c_v4_primary_deployment_beta": pair.ORDINARY_BETA,
        "r2c_v8_trigger_deployment_beta": pair.TRIGGER_BETA,
        "r2c_v8_recovery_pulse_beta": 0.5,
        "r2c_v8_recovery_pulse_rounds": 20,
        "r2c_v12_plan_id": pair.PLAN_ID,
    }
    return {
        "selected": {
            "candidate_id": "P4-B050-D20",
            "pulse_beta": 0.5,
            "pulse_rounds": 20,
        },
        "selected_job": {
            "candidate_id": "P4-B050-D20",
            "method_config": config,
        },
        "selected_run_id": "SCREEN-RUN",
    }


def _patch_manifest_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pair, "_screen_context", _selected_context)
    monkeypatch.setattr(pair, "_source_lineage", lambda context: {"source": "hash"})
    monkeypatch.setattr(pair, "_ensure_assets", lambda seed, rounds: {"asset": "hash"})
    monkeypatch.setattr(pair, "_implementation_hashes", lambda: {"implementation": "hash"})


def test_manifest_locks_selected_candidate_and_s0_s4_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_manifest_inputs(monkeypatch)
    manifest = pair.build_manifest(persist=False)
    assert manifest["candidate_locked_before_pair"] is True
    assert manifest["formal_test_access"] is False
    assert manifest["test_labels_used_for_selection"] is False
    assert manifest["job_order"] == [
        "A-R2C-D3-S0-V12PAIR-P4-B050-D20-s20260810",
        "A-R2C-D3-S4-V12PAIR-P4-B050-D20-s20260810",
    ]
    assert [job["scenario_id"] for job in manifest["jobs"]] == ["S0", "S4"]
    assert all(job["candidate_id"] == "P4-B050-D20" for job in manifest["jobs"])
    assert all(job["seed"] == 20260810 for job in manifest["jobs"])
    assert all(job["rounds"] == 1000 for job in manifest["jobs"])
    assert manifest["frozen_spec_hash"] == pair.config_hash(pair._frozen_spec(manifest))


def test_frozen_manifest_assertion_rejects_candidate_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_manifest_inputs(monkeypatch)
    manifest = pair.build_manifest(persist=False)
    pair._assert_frozen_manifest(manifest)
    manifest["jobs"][1]["candidate_id"] = "P1-B000-D05"
    with pytest.raises(RuntimeError, match="drift"):
        pair._assert_frozen_manifest(manifest)
