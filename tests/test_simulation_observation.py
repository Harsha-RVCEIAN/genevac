"""Unit tests for genevac.simulation.observation."""

import pytest

from genevac.environment.geometry import Point
from genevac.pedestrian.models import PedestrianState, PedestrianStatus, Velocity2D
from genevac.pedestrian.registry import PedestrianRegistry
from genevac.simulation.observation import (
    PedestrianObservation,
    SimulationObservationSnapshot,
    create_observation_snapshot,
)


def test_observation_snapshot_fields_and_counts():
    """Verify all snapshot fields, status accounting, and total count invariant."""
    registry = PedestrianRegistry()
    p_act = PedestrianState(id="p_act", position=Point(1.0, 2.0), status=PedestrianStatus.ACTIVE)
    p_evac = PedestrianState(id="p_evac", position=Point(0.0, 15.0), status=PedestrianStatus.EVACUATED)
    p_blk = PedestrianState(id="p_blk", position=Point(5.0, 5.0), status=PedestrianStatus.BLOCKED)
    p_inact = PedestrianState(id="p_inact", position=Point(20.0, 20.0), status=PedestrianStatus.INACTIVE)

    for p in [p_act, p_evac, p_blk, p_inact]:
        p.record_history_sample(step=0, timestamp=0.0)
        registry.register(p)

    snapshot = create_observation_snapshot(
        registry=registry,
        step_count=4,
        timestamp=0.4,
        dt=0.1,
        observation_interval=0.4,
    )

    assert snapshot.snapshot_id == "snap_000004"
    assert snapshot.step_count == 4
    assert snapshot.simulation_step == 4
    assert snapshot.timestamp == pytest.approx(0.4)
    assert snapshot.dt == pytest.approx(0.1)
    assert snapshot.observation_interval == pytest.approx(0.4)
    assert snapshot.total_count == 4
    assert snapshot.active_count == 1
    assert snapshot.evacuated_count == 1
    assert snapshot.blocked_count == 1
    assert snapshot.inactive_count == 1
    assert len(snapshot.pedestrians) == 4

    # Verify counts sum up exactly to total_count
    assert (
        snapshot.total_count
        == snapshot.active_count
        + snapshot.evacuated_count
        + snapshot.blocked_count
        + snapshot.inactive_count
    )


def test_observation_snapshot_retains_evacuated_pedestrians():
    """Verify evacuated pedestrians are included in snapshots with status and coordinates."""
    registry = PedestrianRegistry()
    p = PedestrianState(
        id="ped_evacuated",
        position=Point(0.0, 15.0),
        status=PedestrianStatus.EVACUATED,
        target_exit_id="exit_left",
    )
    p.record_history_sample(step=0, timestamp=0.0)
    registry.register(p)

    snapshot = create_observation_snapshot(
        registry=registry,
        step_count=8,
        timestamp=0.8,
        dt=0.1,
        observation_interval=0.4,
    )

    assert snapshot.evacuated_count == 1
    assert len(snapshot.pedestrians) == 1
    obs = snapshot.pedestrians[0]
    assert obs.pedestrian_id == "ped_evacuated"
    assert obs.status == "evacuated"
    assert obs.x == 0.0
    assert obs.y == 15.0
    assert obs.speed == 0.0
    assert obs.target_exit_id == "exit_left"


def test_snapshot_immutability_against_live_state_mutation():
    """Verify modifying live pedestrian objects does not alter previously emitted snapshots."""
    registry = PedestrianRegistry()
    p = PedestrianState(id="ped_live", position=Point(10.0, 10.0), status=PedestrianStatus.ACTIVE)
    p.record_history_sample(step=0, timestamp=0.0)
    registry.register(p)

    snapshot = create_observation_snapshot(
        registry=registry,
        step_count=0,
        timestamp=0.0,
        dt=0.1,
        observation_interval=0.4,
    )

    # Mutate live pedestrian
    p.position = Point(99.0, 99.0)
    p.status = PedestrianStatus.EVACUATED
    p.velocity = Velocity2D(5.0, 5.0)
    p.record_history_sample(step=4, timestamp=0.4)

    # Prior snapshot remains completely unchanged
    obs = snapshot.pedestrians[0]
    assert obs.x == 10.0
    assert obs.y == 10.0
    assert obs.status == "active"
    assert len(obs.trajectory_history) == 1
    assert obs.trajectory_history[0].step == 0


def test_snapshot_empty_registry():
    """Verify snapshot generation handles an empty population registry gracefully."""
    registry = PedestrianRegistry()
    snapshot = create_observation_snapshot(
        registry=registry,
        step_count=0,
        timestamp=0.0,
        dt=0.1,
        observation_interval=0.4,
    )

    assert snapshot.total_count == 0
    assert snapshot.active_count == 0
    assert snapshot.evacuated_count == 0
    assert snapshot.pedestrians == ()
    data = snapshot.to_dict()
    assert data["total_count"] == 0
    assert data["pedestrians"] == []


def test_snapshot_to_dict_serialization():
    """Verify serialization to dictionary preserves data types and precision."""
    registry = PedestrianRegistry()
    p = PedestrianState(
        id="ped_01",
        position=Point(12.34567, 23.45678),
        velocity=Velocity2D(1.11111, -0.22222),
        status=PedestrianStatus.ACTIVE,
        target_exit_id="exit_left",
    )
    p.record_history_sample(step=0, timestamp=0.0)
    registry.register(p)

    snapshot = create_observation_snapshot(
        registry=registry,
        step_count=0,
        timestamp=0.0,
        dt=0.1,
        observation_interval=0.4,
    )
    data = snapshot.to_dict()

    assert data["snapshot_id"] == "snap_000000"
    assert data["total_count"] == 1
    assert data["active_count"] == 1
    assert data["pedestrians"][0]["pedestrian_id"] == "ped_01"
    assert data["pedestrians"][0]["x"] == pytest.approx(12.34567)
    assert data["pedestrians"][0]["vx"] == pytest.approx(1.11111)
    assert len(data["pedestrians"][0]["trajectory_history"]) == 1
    assert data["pedestrians"][0]["trajectory_history"][0]["step"] == 0
