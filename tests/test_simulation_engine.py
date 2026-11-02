"""Unit tests for genevac.simulation.engine: lifecycle, exits, pauses, and isolation."""

import os
from pathlib import Path
import subprocess
import sys
import pytest

from genevac.environment.builder import create_initial_building
from genevac.environment.geometry import Point
from genevac.environment.models import Building
from genevac.pedestrian.models import PedestrianState, PedestrianStatus, Velocity2D
from genevac.pedestrian.registry import PedestrianRegistry
from genevac.simulation.engine import SimulationEngine


@pytest.fixture
def initial_building() -> Building:
    """Fixture providing the canonical 40x30m initial building."""
    return create_initial_building()


def test_engine_initialization_defaults(initial_building: Building):
    """Verify engine initializes with valid building, empty registry, and step 0 snapshot."""
    engine = SimulationEngine(initial_building)
    assert engine.building is initial_building
    assert engine.registry.total_count == 0
    assert engine.clock.step_count == 0
    assert engine.clock.timestamp == 0.0
    assert engine.is_paused is False
    assert engine.latest_snapshot is not None
    assert engine.latest_snapshot.step_count == 0
    assert engine.latest_snapshot.timestamp == 0.0
    assert engine.latest_snapshot.total_count == 0


def test_engine_rejects_invalid_building():
    """Verify engine rejects non-Building objects."""
    with pytest.raises(TypeError, match="building must be a Building instance"):
        SimulationEngine("not_a_building")  # type: ignore[arg-type]


def test_engine_validates_initial_pedestrian_positions(initial_building: Building):
    """Verify engine rejects initial pedestrian coordinates outside building or declared areas."""
    # Outside building bounds
    reg_oob = PedestrianRegistry()
    reg_oob.register(PedestrianState(id="p_oob", position=Point(45.0, 15.0)))
    with pytest.raises(ValueError, match="is outside building bounds"):
        SimulationEngine(initial_building, reg_oob)

    # In room_2 coordinates but declared in room_1
    reg_wrong_area = PedestrianRegistry()
    reg_wrong_area.register(
        PedestrianState(id="p_wrong", position=Point(25.0, 5.0), current_area_id="room_1")
    )
    with pytest.raises(ValueError, match="is not within declared area 'room_1' bounds"):
        SimulationEngine(initial_building, reg_wrong_area)


def test_engine_validates_initial_target_exit(initial_building: Building):
    """Verify engine rejects non-existent target exit references."""
    reg = PedestrianRegistry()
    reg.register(PedestrianState(id="p1", position=Point(5.0, 5.0), target_exit_id="exit_nonexistent"))
    with pytest.raises(ValueError, match="references non-existent target exit"):
        SimulationEngine(initial_building, reg)


def test_engine_automatic_route_initialization(initial_building: Building):
    """Verify engine automatically computes route waypoints if target_exit_id is assigned."""
    reg = PedestrianRegistry()
    ped = PedestrianState(
        id="p_auto",
        position=Point(5.0, 5.0),
        current_area_id="room_1",
        target_exit_id="exit_left",
    )
    assert ped.waypoints == []
    reg.register(ped)

    engine = SimulationEngine(initial_building, reg)
    assert len(ped.waypoints) > 0
    # Final waypoint is exit_left midpoint (0.0, 15.0)
    assert ped.waypoints[-1] == Point(0.0, 15.0)


def test_engine_step_and_observation_cadence(initial_building: Building):
    """Verify step() returns None on intermediate steps and snapshot on observation steps."""
    reg = PedestrianRegistry()
    reg.register(PedestrianState(id="p1", position=Point(10.0, 15.0)))
    engine = SimulationEngine(initial_building, reg, dt=0.1, observation_interval=0.4)

    # Intermediate steps 1, 2, 3 return None
    assert engine.step() is None
    assert engine.clock.step_count == 1
    assert engine.step() is None
    assert engine.clock.step_count == 2
    assert engine.step() is None
    assert engine.clock.step_count == 3

    # Observation step 4 returns a new snapshot
    snap4 = engine.step()
    assert snap4 is not None
    assert snap4.step_count == 4
    assert snap4.timestamp == pytest.approx(0.4)
    assert engine.latest_snapshot is snap4


def test_engine_pause_and_resume(initial_building: Building):
    """Verify pausing halts simulation advancement and movement."""
    reg = PedestrianRegistry()
    ped = PedestrianState(
        id="p_pause",
        position=Point(10.0, 15.0),
        preferred_speed=1.3,
        waypoints=[Point(20.0, 15.0)],
    )
    reg.register(ped)
    engine = SimulationEngine(initial_building, reg)

    engine.pause()
    assert engine.is_paused is True

    # Stepping while paused does nothing
    result = engine.step()
    assert result is None
    assert engine.clock.step_count == 0
    assert ped.position == Point(10.0, 15.0)

    # Resume and step
    engine.resume()
    assert engine.is_paused is False
    engine.step()
    assert engine.clock.step_count == 1
    assert ped.position.x > 10.0


def test_engine_run_steps_and_step_until(initial_building: Building):
    """Verify run_steps and step_until multi-step helper methods."""
    reg = PedestrianRegistry()
    reg.register(PedestrianState(id="p1", position=Point(10.0, 15.0)))
    engine = SimulationEngine(initial_building, reg, dt=0.1, observation_interval=0.4)

    # Run 8 steps -> produces snapshots at step 4 and step 8
    snapshots = engine.run_steps(8)
    assert len(snapshots) == 2
    assert snapshots[0].step_count == 4
    assert snapshots[1].step_count == 8
    assert engine.clock.step_count == 8

    # Step until step 16 -> produces snapshots at step 12 and step 16
    more_snaps = engine.step_until(16)
    assert len(more_snaps) == 2
    assert more_snaps[0].step_count == 12
    assert more_snaps[1].step_count == 16
    assert engine.clock.step_count == 16


# --- Exit Handling Tests ---


def test_correct_assigned_exit_crossing_preserves_position(initial_building: Building):
    """Verify crossing the assigned exit opening freezes pedestrian at exact crossing position."""
    # exit_left is at x=0.0, y in [14.0, 16.0].
    # Pedestrian at x=0.05, y=15.0 moving left toward (-1.0, 15.0)
    reg = PedestrianRegistry()
    ped = PedestrianState(
        id="p_evac_assigned",
        position=Point(0.05, 15.0),
        radius=0.25,
        preferred_speed=1.0,
        target_exit_id="exit_left",
        waypoints=[Point(-1.0, 15.0)],
    )
    reg.register(ped)
    engine = SimulationEngine(initial_building, reg, dt=0.1)

    assert engine.registry.active_count == 1
    assert engine.registry.evacuated_count == 0

    engine.step()

    # Evacuation triggered
    assert ped.status == PedestrianStatus.EVACUATED
    # Exact crossing point is x=0.0, y=15.0 (not teleported to exit midpoint)
    assert ped.position.x == pytest.approx(0.0)
    assert ped.position.y == pytest.approx(15.0)
    assert ped.velocity == Velocity2D(0.0, 0.0)
    assert ped.waypoints == []

    # Registry counts updated
    assert engine.registry.active_count == 0
    assert engine.registry.evacuated_count == 1


def test_crossing_outside_usable_opening_is_blocked(initial_building: Building):
    """Verify pedestrian crossing boundary outside usable opening (door jamb collision) is blocked."""
    # exit_left opening is y in [14.0, 16.0].
    # With radius 0.25, usable opening is y in [14.25, 15.75].
    # Attempting to cross at y = 14.10 (only 0.10m from north jamb) must collide with wall and be blocked.
    reg = PedestrianRegistry()
    ped = PedestrianState(
        id="p_jamb_col",
        position=Point(0.05, 14.10),
        radius=0.25,
        preferred_speed=1.0,
        target_exit_id="exit_left",
        waypoints=[Point(-1.0, 14.10)],
    )
    reg.register(ped)
    engine = SimulationEngine(initial_building, reg, dt=0.1)

    engine.step()

    assert ped.status == PedestrianStatus.BLOCKED
    assert ped.velocity == Velocity2D(0.0, 0.0)
    assert engine.registry.evacuated_count == 0


def test_approaching_exit_without_crossing_remains_active(initial_building: Building):
    """Verify pedestrian moving toward exit but not yet crossing remains ACTIVE."""
    reg = PedestrianRegistry()
    ped = PedestrianState(
        id="p_approach",
        position=Point(2.0, 15.0),
        radius=0.25,
        preferred_speed=1.0,
        target_exit_id="exit_left",
        waypoints=[Point(0.0, 15.0)],
    )
    reg.register(ped)
    engine = SimulationEngine(initial_building, reg, dt=0.1)

    engine.step()

    assert ped.status == PedestrianStatus.ACTIVE
    assert ped.position.x == pytest.approx(1.9)
    assert engine.registry.evacuated_count == 0


def test_crossing_unassigned_exit_is_blocked(initial_building: Building):
    """Verify pedestrian attempting to cross an exit not matching target_exit_id is blocked."""
    # Moving toward exit_left at x=0.0, but assigned target is exit_right
    reg = PedestrianRegistry()
    ped = PedestrianState(
        id="p_wrong_exit",
        position=Point(0.05, 15.0),
        radius=0.25,
        preferred_speed=1.0,
        target_exit_id="exit_right",
        waypoints=[Point(-1.0, 15.0)],
    )
    reg.register(ped)
    engine = SimulationEngine(initial_building, reg, dt=0.1)

    engine.step()

    # Blocked, not evacuated
    assert ped.status == PedestrianStatus.BLOCKED
    assert ped.velocity == Velocity2D(0.0, 0.0)
    assert engine.registry.evacuated_count == 0


def test_cohort_evacuation_end_to_end(initial_building: Building):
    """Verify complete deterministic evacuation of pedestrians spawned in different rooms."""
    reg = PedestrianRegistry()
    # Pedestrian in Room 1 targeting exit_left
    p1 = PedestrianState(id="p_r1", position=Point(10.0, 5.0), target_exit_id="exit_left")
    # Pedestrian in Corridor targeting exit_left
    p2 = PedestrianState(id="p_corr", position=Point(5.0, 15.0), target_exit_id="exit_left")
    reg.register(p1)
    reg.register(p2)

    engine = SimulationEngine(initial_building, reg, dt=0.1)
    assert engine.registry.active_count == 2
    assert engine.registry.evacuated_count == 0

    # Advance simulation until p2 evacuates (distance 5.0m at 1.3m/s takes ~39 steps)
    engine.run_steps(40)

    assert p2.status == PedestrianStatus.EVACUATED
    assert p2.position.x == pytest.approx(0.0)
    assert engine.registry.evacuated_count >= 1

    # Advance until p1 also evacuates (travels room1 -> doorway -> corridor -> exit)
    engine.run_steps(160)
    assert p1.status == PedestrianStatus.EVACUATED
    assert engine.registry.active_count == 0
    assert engine.registry.evacuated_count == 2


def test_simulation_architectural_isolation():
    """Verify that importing genevac.simulation in a clean process does not import pygame, torch, mid, or numpy."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import genevac.simulation, sys; "
            "assert 'pygame' not in sys.modules, 'pygame was imported by simulation!'; "
            "assert 'torch' not in sys.modules, 'torch was imported!'; "
            "assert 'mid' not in sys.modules, 'mid was imported!'; "
            "assert 'numpy' not in sys.modules, 'numpy was imported by simulation!'; "
            "assert 'genevac.visualization' not in sys.modules, 'visualization was imported by simulation!'"
        ),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Simulation architectural isolation violation: {res.stderr}"
