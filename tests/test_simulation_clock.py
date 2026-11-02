"""Unit tests for genevac.simulation.clock."""

import pytest

from genevac.simulation.clock import SimulationClock


def test_clock_default_timing_and_stride():
    """Verify default timing parameters: dt=0.1s, observation_interval=0.4s, stride=4."""
    clock = SimulationClock()
    assert clock.dt == pytest.approx(0.1)
    assert clock.observation_interval == pytest.approx(0.4)
    assert clock.observation_stride == 4
    assert clock.step_count == 0
    assert clock.timestamp == 0.0
    assert clock.is_observation_step() is True


def test_clock_rejects_invalid_dt():
    """Verify clock rejects zero, negative, non-finite, and non-numeric dt."""
    with pytest.raises(ValueError, match="dt must be strictly positive"):
        SimulationClock(dt=0.0)
    with pytest.raises(ValueError, match="dt must be strictly positive"):
        SimulationClock(dt=-0.1)
    with pytest.raises(ValueError, match="dt must be strictly positive"):
        SimulationClock(dt=float("nan"))
    with pytest.raises(ValueError, match="dt must be strictly positive"):
        SimulationClock(dt=float("inf"))
    with pytest.raises(TypeError, match="dt and observation_interval must be numeric"):
        SimulationClock(dt="invalid")  # type: ignore[arg-type]


def test_clock_rejects_invalid_observation_interval():
    """Verify clock rejects zero, negative, non-finite, and non-numeric observation intervals."""
    with pytest.raises(ValueError, match="observation_interval must be strictly positive"):
        SimulationClock(observation_interval=0.0)
    with pytest.raises(ValueError, match="observation_interval must be strictly positive"):
        SimulationClock(observation_interval=-0.4)
    with pytest.raises(ValueError, match="observation_interval must be strictly positive"):
        SimulationClock(observation_interval=float("nan"))
    with pytest.raises(TypeError, match="dt and observation_interval must be numeric"):
        SimulationClock(observation_interval="slow")  # type: ignore[arg-type]


def test_clock_rejects_non_integral_ratio():
    """Verify clock rejects observation intervals that are not exact integer multiples of dt."""
    # 0.35 / 0.1 = 3.5 (not integer)
    with pytest.raises(ValueError, match="must be an exact integer multiple of dt"):
        SimulationClock(dt=0.1, observation_interval=0.35)

    # 0.05 / 0.1 = 0.5 (less than dt)
    with pytest.raises(ValueError, match="must be greater than or equal to dt"):
        SimulationClock(dt=0.1, observation_interval=0.05)


def test_clock_accepts_valid_custom_timings():
    """Verify clock accepts valid integer-multiple timing ratios."""
    # 1.0s / 0.05s = 20 steps
    clock = SimulationClock(dt=0.05, observation_interval=1.0)
    assert clock.dt == pytest.approx(0.05)
    assert clock.observation_stride == 20

    # 0.1s / 0.1s = 1 step
    clock_equal = SimulationClock(dt=0.1, observation_interval=0.1)
    assert clock_equal.observation_stride == 1


def test_clock_timestamp_progression_and_no_drift():
    """Verify step advancement and exact timestamp derivation over 1000 steps."""
    clock = SimulationClock(dt=0.1, observation_interval=0.4)

    for expected_step in range(1, 1001):
        step, ts = clock.tick()
        assert step == expected_step
        assert clock.step_count == expected_step
        expected_ts = round(expected_step * 0.1, 9)
        assert ts == pytest.approx(expected_ts)
        assert clock.timestamp == pytest.approx(expected_ts)


def test_clock_observation_cadence_cycle():
    """Verify observation steps trigger exactly every 4 steps (0.4s apart) under default config."""
    clock = SimulationClock(dt=0.1, observation_interval=0.4)
    # Step 0 is an observation step
    assert clock.is_observation_step() is True

    observation_steps = [0]
    observation_times = [0.0]

    for _ in range(12):
        clock.tick()
        if clock.is_observation_step():
            observation_steps.append(clock.step_count)
            observation_times.append(clock.timestamp)

    assert observation_steps == [0, 4, 8, 12]
    assert observation_times == [0.0, 0.4, 0.8, 1.2]


def test_clock_reset():
    """Verify reset restores step count and timestamp back to zero."""
    clock = SimulationClock(dt=0.1, observation_interval=0.4)
    for _ in range(10):
        clock.tick()
    assert clock.step_count == 10
    assert clock.timestamp == pytest.approx(1.0)

    clock.reset()
    assert clock.step_count == 0
    assert clock.timestamp == 0.0
    assert clock.is_observation_step() is True
