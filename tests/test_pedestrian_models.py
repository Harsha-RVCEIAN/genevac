"""Unit tests for genevac.pedestrian.models."""

import math
import pytest

from genevac.environment.geometry import Point
from genevac.pedestrian.models import (
    PedestrianState,
    PedestrianStatus,
    TimestampedTrajectoryPoint,
    Velocity2D,
)


# --- Velocity2D Tests ---


def test_velocity2d_valid():
    """Verify Velocity2D construction, speed calculation, and tuple conversion."""
    v = Velocity2D(3.0, 4.0)
    assert v.vx == pytest.approx(3.0)
    assert v.vy == pytest.approx(4.0)
    assert v.speed == pytest.approx(5.0)
    assert v.as_tuple() == (3.0, 4.0)


def test_velocity2d_zero():
    """Verify zero velocity produces zero speed."""
    v = Velocity2D(0.0, 0.0)
    assert v.speed == 0.0
    assert v.as_tuple() == (0.0, 0.0)


def test_velocity2d_rejects_nan_and_inf():
    """Verify Velocity2D rejects NaN and infinite values."""
    with pytest.raises(ValueError, match="Velocity components must be finite"):
        Velocity2D(float("nan"), 1.0)
    with pytest.raises(ValueError, match="Velocity components must be finite"):
        Velocity2D(1.0, float("nan"))
    with pytest.raises(ValueError, match="Velocity components must be finite"):
        Velocity2D(float("inf"), 1.0)
    with pytest.raises(ValueError, match="Velocity components must be finite"):
        Velocity2D(1.0, float("-inf"))


def test_velocity2d_rejects_non_numeric():
    """Verify Velocity2D rejects non-numeric types."""
    with pytest.raises(TypeError, match="Velocity components must be numeric"):
        Velocity2D("invalid", 1.0)  # type: ignore[arg-type]


# --- TimestampedTrajectoryPoint Tests ---


def test_trajectory_point_valid():
    """Verify TimestampedTrajectoryPoint construction and tuple conversion."""
    pt = TimestampedTrajectoryPoint(step=10, timestamp=1.0, x=15.5, y=20.25)
    assert pt.step == 10
    assert pt.timestamp == pytest.approx(1.0)
    assert pt.x == pytest.approx(15.5)
    assert pt.y == pytest.approx(20.25)
    assert pt.as_tuple() == (10, 1.0, 15.5, 20.25)


def test_trajectory_point_rejects_negative_step():
    """Verify negative step is rejected."""
    with pytest.raises(ValueError, match="Step must be non-negative"):
        TimestampedTrajectoryPoint(step=-1, timestamp=0.0, x=1.0, y=1.0)


def test_trajectory_point_rejects_invalid_step_types():
    """Verify non-integer and bool step types are rejected."""
    with pytest.raises(TypeError, match="Step must be an integer"):
        TimestampedTrajectoryPoint(step=1.5, timestamp=0.0, x=1.0, y=1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Step must be an integer"):
        TimestampedTrajectoryPoint(step=True, timestamp=0.0, x=1.0, y=1.0)  # type: ignore[arg-type]


def test_trajectory_point_rejects_negative_or_non_finite_timestamp():
    """Verify negative, NaN, and infinite timestamps are rejected."""
    with pytest.raises(ValueError, match="Timestamp must be non-negative and finite"):
        TimestampedTrajectoryPoint(step=0, timestamp=-0.1, x=1.0, y=1.0)
    with pytest.raises(ValueError, match="Timestamp must be non-negative and finite"):
        TimestampedTrajectoryPoint(step=0, timestamp=float("nan"), x=1.0, y=1.0)
    with pytest.raises(ValueError, match="Timestamp must be non-negative and finite"):
        TimestampedTrajectoryPoint(step=0, timestamp=float("inf"), x=1.0, y=1.0)


def test_trajectory_point_rejects_non_finite_coordinates():
    """Verify NaN and infinite coordinates are rejected."""
    with pytest.raises(ValueError, match="Coordinates must be finite"):
        TimestampedTrajectoryPoint(step=0, timestamp=0.0, x=float("nan"), y=1.0)
    with pytest.raises(ValueError, match="Coordinates must be finite"):
        TimestampedTrajectoryPoint(step=0, timestamp=0.0, x=1.0, y=float("inf"))


# --- PedestrianState Tests ---


def test_pedestrian_state_valid_defaults():
    """Verify PedestrianState construction with default parameters."""
    ped = PedestrianState(id="ped_001", position=Point(5.0, 10.0))
    assert ped.id == "ped_001"
    assert ped.position == Point(5.0, 10.0)
    assert ped.velocity == Velocity2D(0.0, 0.0)
    assert ped.velocity.speed == 0.0
    assert ped.status == PedestrianStatus.ACTIVE
    assert ped.target_exit_id is None
    assert ped.current_area_id is None
    assert ped.waypoints == []
    assert ped.radius == pytest.approx(0.25)
    assert ped.preferred_speed == pytest.approx(1.3)
    assert ped.trajectory_history == []
    assert ped.max_history_len == 100


def test_pedestrian_state_tuple_coercion():
    """Verify position and velocity can be initialized from 2D tuples/lists."""
    ped = PedestrianState(
        id="ped_002",
        position=(12.0, 14.0),  # type: ignore[arg-type]
        velocity=(1.0, -0.5),  # type: ignore[arg-type]
    )
    assert isinstance(ped.position, Point)
    assert ped.position.x == 12.0
    assert ped.position.y == 14.0
    assert isinstance(ped.velocity, Velocity2D)
    assert ped.velocity.vx == 1.0
    assert ped.velocity.vy == -0.5


def test_pedestrian_state_rejects_invalid_ids():
    """Verify empty, non-string, or invalid IDs are rejected."""
    with pytest.raises(ValueError, match="Pedestrian ID must be a non-empty string"):
        PedestrianState(id="", position=Point(0.0, 0.0))
    with pytest.raises(ValueError, match="Pedestrian ID must be a non-empty string"):
        PedestrianState(id=123, position=Point(0.0, 0.0))  # type: ignore[arg-type]


def test_pedestrian_state_rejects_non_finite_position():
    """Verify NaN or infinite position coordinates are rejected."""
    with pytest.raises(ValueError, match="coordinates must be finite"):
        PedestrianState(id="ped_bad_pos", position=Point(float("nan"), 10.0))
    with pytest.raises(ValueError, match="coordinates must be finite"):
        PedestrianState(id="ped_bad_pos", position=Point(10.0, float("inf")))


def test_pedestrian_state_rejects_invalid_position_type():
    """Verify position type checking."""
    with pytest.raises(TypeError, match="position must be a Point"):
        PedestrianState(id="ped_bad_type", position="invalid")  # type: ignore[arg-type]


def test_pedestrian_state_status_validation():
    """Verify string coercion and rejection of invalid status values."""
    ped = PedestrianState(id="p1", position=Point(1, 1), status="evacuated")  # type: ignore[arg-type]
    assert ped.status == PedestrianStatus.EVACUATED

    with pytest.raises(ValueError, match="Invalid pedestrian status"):
        PedestrianState(id="p2", position=Point(1, 1), status="flying")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="status must be a PedestrianStatus"):
        PedestrianState(id="p3", position=Point(1, 1), status=123)  # type: ignore[arg-type]


def test_pedestrian_state_radius_validation():
    """Verify radius must be strictly positive and finite."""
    with pytest.raises(ValueError, match="radius must be strictly positive and finite"):
        PedestrianState(id="p", position=Point(1, 1), radius=0.0)
    with pytest.raises(ValueError, match="radius must be strictly positive and finite"):
        PedestrianState(id="p", position=Point(1, 1), radius=-0.25)
    with pytest.raises(ValueError, match="radius must be strictly positive and finite"):
        PedestrianState(id="p", position=Point(1, 1), radius=float("nan"))
    with pytest.raises(TypeError, match="radius must be numeric"):
        PedestrianState(id="p", position=Point(1, 1), radius="small")  # type: ignore[arg-type]


def test_pedestrian_state_preferred_speed_validation():
    """Verify preferred_speed must be non-negative and finite."""
    ped_stationary = PedestrianState(id="p", position=Point(1, 1), preferred_speed=0.0)
    assert ped_stationary.preferred_speed == 0.0

    with pytest.raises(ValueError, match="preferred_speed must be non-negative and finite"):
        PedestrianState(id="p", position=Point(1, 1), preferred_speed=-1.0)
    with pytest.raises(ValueError, match="preferred_speed must be non-negative and finite"):
        PedestrianState(id="p", position=Point(1, 1), preferred_speed=float("nan"))
    with pytest.raises(TypeError, match="preferred_speed must be numeric"):
        PedestrianState(id="p", position=Point(1, 1), preferred_speed="fast")  # type: ignore[arg-type]


def test_pedestrian_state_max_history_len_validation():
    """Verify max_history_len must be a strictly positive integer."""
    with pytest.raises(ValueError, match="max_history_len must be strictly positive"):
        PedestrianState(id="p", position=Point(1, 1), max_history_len=0)
    with pytest.raises(ValueError, match="max_history_len must be strictly positive"):
        PedestrianState(id="p", position=Point(1, 1), max_history_len=-10)
    with pytest.raises(TypeError, match="max_history_len must be an integer"):
        PedestrianState(id="p", position=Point(1, 1), max_history_len=10.5)  # type: ignore[arg-type]


def test_pedestrian_state_waypoints_validation():
    """Verify waypoints list validation and point coercion."""
    wps = [Point(1.0, 2.0), (3.0, 4.0)]
    ped = PedestrianState(id="p", position=Point(0, 0), waypoints=wps)  # type: ignore[arg-type]
    assert len(ped.waypoints) == 2
    assert isinstance(ped.waypoints[0], Point)
    assert isinstance(ped.waypoints[1], Point)
    assert ped.waypoints[1] == Point(3.0, 4.0)

    with pytest.raises(TypeError, match="waypoints must be a list"):
        PedestrianState(id="p", position=Point(0, 0), waypoints="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Waypoint 0 coordinates must be finite"):
        PedestrianState(id="p", position=Point(0, 0), waypoints=[Point(float("nan"), 1.0)])


def test_pedestrian_state_target_exit_and_area_validation():
    """Verify optional target_exit_id and current_area_id non-empty string checks."""
    ped = PedestrianState(
        id="p",
        position=Point(1, 1),
        target_exit_id="exit_left",
        current_area_id="room_1",
    )
    assert ped.target_exit_id == "exit_left"
    assert ped.current_area_id == "room_1"

    with pytest.raises(ValueError, match="target_exit_id must be a non-empty string"):
        PedestrianState(id="p", position=Point(1, 1), target_exit_id="")
    with pytest.raises(ValueError, match="current_area_id must be a non-empty string"):
        PedestrianState(id="p", position=Point(1, 1), current_area_id="")


# --- Trajectory History & Evacuation Transition Tests ---


def test_pedestrian_record_history_sample():
    """Verify record_history_sample creates and appends timestamped points."""
    ped = PedestrianState(id="ped_hist", position=Point(10.0, 15.0), max_history_len=3)
    assert len(ped.trajectory_history) == 0

    # Step 0
    ped.record_history_sample(step=0, timestamp=0.0)
    assert len(ped.trajectory_history) == 1
    assert ped.trajectory_history[0] == TimestampedTrajectoryPoint(step=0, timestamp=0.0, x=10.0, y=15.0)

    # Step 4 (position moved)
    ped.position = Point(10.5, 15.0)
    ped.record_history_sample(step=4, timestamp=0.4)
    assert len(ped.trajectory_history) == 2
    assert ped.trajectory_history[1] == TimestampedTrajectoryPoint(step=4, timestamp=0.4, x=10.5, y=15.0)

    # Step 8
    ped.position = Point(11.0, 15.0)
    ped.record_history_sample(step=8, timestamp=0.8)
    assert len(ped.trajectory_history) == 3

    # Step 12 (FIFO eviction of step 0)
    ped.position = Point(11.5, 15.0)
    ped.record_history_sample(step=12, timestamp=1.2)
    assert len(ped.trajectory_history) == 3
    assert ped.trajectory_history[0].step == 4
    assert ped.trajectory_history[1].step == 8
    assert ped.trajectory_history[2].step == 12


def test_pedestrian_freeze_at_evacuation():
    """Verify evacuation freezes velocity to 0, sets EVACUATED status, and clears waypoints."""
    ped = PedestrianState(
        id="ped_evac",
        position=Point(0.0, 15.0),
        velocity=Velocity2D(-1.3, 0.0),
        status=PedestrianStatus.ACTIVE,
        waypoints=[Point(0.0, 15.0)],
    )
    assert ped.velocity.speed == pytest.approx(1.3)
    assert len(ped.waypoints) == 1

    ped.freeze_at_evacuation()

    assert ped.status == PedestrianStatus.EVACUATED
    assert ped.velocity == Velocity2D(0.0, 0.0)
    assert ped.velocity.speed == 0.0
    assert ped.waypoints == []
