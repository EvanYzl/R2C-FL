from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from r2c_baselines import r2c_v6


def test_detector_requires_broad_change_and_honors_cooldown() -> None:
    detector = r2c_v6.TelemetryShiftDetector(100)
    available = np.arange(20, dtype=np.int64)
    duration = np.ones(100, dtype=np.float64)

    first = detector.observe(1, available, duration)
    assert first.comparable_clients == 0
    assert first.trigger is False

    second = detector.observe(2, available, duration)
    assert second.comparable_clients == 20
    assert second.changed_clients == 0
    assert second.trigger is False

    duration[available[:6]] = 2.0
    third = detector.observe(3, available, duration)
    assert third.changed_clients == 6
    assert math.isclose(third.changed_fraction, 0.30, abs_tol=0.0)
    assert third.trigger is True
    assert third.cooldown_before == 0
    assert third.cooldown_after == 50
    assert third.synchronization_count == 1

    duration[available] = 4.0
    fourth = detector.observe(4, available, duration)
    assert fourth.changed_clients == 20
    assert fourth.trigger is False
    assert fourth.cooldown_before == 50
    assert fourth.cooldown_after == 49
    assert fourth.synchronization_count == 1


def test_detector_compares_only_previous_observation_of_available_client() -> None:
    detector = r2c_v6.TelemetryShiftDetector(100)
    duration = np.ones(100, dtype=np.float64)
    detector.observe(1, np.arange(20), duration)

    duration[20:40] = 100.0
    unseen = detector.observe(2, np.arange(20, 40), duration)
    assert unseen.comparable_clients == 0
    assert unseen.trigger is False

    duration[:20] *= 1.49
    below = detector.observe(3, np.arange(20), duration)
    assert below.comparable_clients == 20
    assert below.changed_clients == 0
    assert below.trigger is False


def test_detector_configuration_validation() -> None:
    detector = r2c_v6.TelemetryShiftDetector.from_config(
        100,
        {
            "r2c_v6_duration_log_ratio_threshold": math.log(1.5),
            "r2c_v6_changed_fraction_threshold": 0.25,
            "r2c_v6_min_comparable_clients": 10,
            "r2c_v6_cooldown_rounds": 50,
        },
    )
    assert math.isclose(detector.log_ratio_threshold, math.log(1.5), abs_tol=0.0)
    assert detector.fraction_threshold == 0.25
    assert detector.min_comparable_clients == 10
    assert detector.cooldown_rounds == 50

    for kwargs in (
        {"log_ratio_threshold": 0.0},
        {"fraction_threshold": 1.1},
        {"min_comparable_clients": 0},
        {"cooldown_rounds": -1},
    ):
        try:
            r2c_v6.TelemetryShiftDetector(100, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid detector configuration: {kwargs}")


def test_v6_wrapper_preserves_v5_learning_path_and_relabels_protocol(monkeypatch) -> None:
    captured: dict[str, object] = {}
    result = SimpleNamespace(
        checkpoint_rows=[{"protocol_version": "old"}],
        certificate_row={"protocol_version": "old", "certificate_record_hash": "old"},
    )

    def fake_v5(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(r2c_v6, "run_r2c_v5_round", fake_v5)
    counts = np.arange(100, dtype=np.int64)
    returned = r2c_v6.run_r2c_v6_round(
        model=object(),
        trainer=object(),
        data=object(),
        trace=object(),
        round_number=3,
        available=np.arange(20),
        sample_counts=np.ones(100),
        learning_rate=0.1,
        payload_bytes=10,
        seed=7,
        config={"r2c_v5_history_mix": 0.45},
        run_id="test",
        round_start_model_hash="0" * 64,
        full_logging=True,
        selection_history_counts=counts,
    )
    assert returned is result
    assert np.array_equal(captured["selection_history_counts"], counts)
    assert result.checkpoint_rows[0]["protocol_version"] == r2c_v6.PROTOCOL_VERSION
    assert result.checkpoint_rows[0]["selection_protocol_version"] == r2c_v6.V5_PROTOCOL_VERSION
    assert result.certificate_row["deployment_rule"] == r2c_v6.DEPLOYMENT_RULE
    assert result.certificate_row["certificate_record_hash"] != "old"


def test_telemetry_observation_is_attached_to_checkpoint_and_certificate() -> None:
    result = SimpleNamespace(
        checkpoint_rows=[{}],
        certificate_row={"certificate_record_hash": "old"},
    )
    observation = r2c_v6.TelemetryShiftObservation(
        round_number=9,
        comparable_clients=20,
        changed_clients=7,
        changed_fraction=0.35,
        log_ratio_threshold=math.log(1.5),
        fraction_threshold=0.25,
        min_comparable_clients=10,
        trigger=True,
        cooldown_before=0,
        cooldown_after=50,
        synchronization_count=1,
    )
    r2c_v6.attach_telemetry_observation(result, observation)
    assert result.checkpoint_rows[0]["telemetry_shift_trigger"] is True
    assert result.certificate_row["telemetry_shift_round"] == 9
    assert result.certificate_row["telemetry_shift_labels_used"] is False
    assert result.certificate_row["telemetry_shift_scenario_metadata_used"] is False
    assert result.certificate_row["certificate_record_hash"] != "old"
