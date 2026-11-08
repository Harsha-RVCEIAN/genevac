"""Unit tests for genevac.simulation.engine: lifecycle, exits, pauses, and isolation."""

import os
from pathlib import Path
import random
import subprocess
import sys
import pytest

from genevac.environment.builder import create_initial_building
from genevac.environment.geometry import Point
from genevac.environment.models import Building
from genevac.pedestrian.models import PedestrianState, PedestrianStatus, Velocity2D
from genevac.pedestrian.registry import PedestrianRegistry
from genevac.simulation.engine import SimulationEngine
from genevac.simulation.events import SimulationEventType


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
    reg.register(PedestrianState(id="p1", position=Point(2.0, 5.0), target_exit_id="exit_nonexistent"))
    with pytest.raises(ValueError, match="references non-existent target exit"):
        SimulationEngine(initial_building, reg)


def test_engine_automatic_route_initialization(initial_building: Building):
    """Verify engine automatically computes route waypoints if target_exit_id is assigned."""
    reg = PedestrianRegistry()
    ped = PedestrianState(
        id="p_auto",
        position=Point(2.0, 5.0),
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
    engine = SimulationEngine(initial_building, reg, block_unassigned_exits=False)

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
    # Starts safely inside corridor at (0.5, 14.10) with full wall/boundary clearance,
    # then attempts to cross outward at y = 14.10 (only 0.10m from north jamb) -> must collide with wall and be blocked.
    reg = PedestrianRegistry()
    ped = PedestrianState(
        id="p_jamb_col",
        position=Point(0.5, 14.10),
        radius=0.25,
        preferred_speed=7.0,
        target_exit_id="exit_left",
        waypoints=[Point(-0.2, 14.10)],
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
    # Moving toward exit_left at x=0.0, but assigned target is exit_right.
    # Starts safely inside corridor at (0.5, 15.0) with full wall/boundary clearance.
    reg = PedestrianRegistry()
    ped = PedestrianState(
        id="p_wrong_exit",
        position=Point(0.5, 15.0),
        radius=0.25,
        preferred_speed=7.0,
        target_exit_id="exit_right",
        waypoints=[Point(-0.2, 15.0)],
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


def test_engine_unassigned_exit_default_blocks_and_records_event(initial_building: Building):
    """Verify that by default, an unassigned-exit pedestrian transitions to BLOCKED and records NO_ROUTE."""
    reg = PedestrianRegistry()
    ped_unassigned = PedestrianState(
        id="p_unassigned",
        position=Point(10.0, 15.0),
        current_area_id="main_corridor",
        target_exit_id=None,
        waypoints=[Point(12.0, 15.0)],  # Invalid waypoints without an assigned exit
    )
    ped_valid = PedestrianState(
        id="p_valid",
        position=Point(2.0, 15.0),
        current_area_id="main_corridor",
        target_exit_id="exit_left",
        preferred_speed=1.0,
    )
    reg.register(ped_unassigned)
    reg.register(ped_valid)

    # Default constructor has block_unassigned_exits=True
    engine = SimulationEngine(initial_building, reg)

    # Unassigned pedestrian is immediately BLOCKED with zero velocity and cleared waypoints
    assert ped_unassigned.status == PedestrianStatus.BLOCKED
    assert ped_unassigned.velocity == Velocity2D(0.0, 0.0)
    assert len(ped_unassigned.waypoints) == 0

    # Valid pedestrian remains ACTIVE
    assert ped_valid.status == PedestrianStatus.ACTIVE

    # Exactly one NO_ROUTE event recorded for the unassigned pedestrian
    unassigned_events = [e for e in engine.events if e.pedestrian_id == "p_unassigned"]
    assert len(unassigned_events) == 1
    assert unassigned_events[0].event_type == SimulationEventType.NO_ROUTE
    assert unassigned_events[0].details["reason"] == "unassigned_exit"

    # Other pedestrians continue simulating normally
    engine.step()
    assert ped_valid.status == PedestrianStatus.ACTIVE
    assert ped_unassigned.status == PedestrianStatus.BLOCKED


def test_engine_get_snapshot_repeated_calls_are_side_effect_free(initial_building: Building):
    """Verify that calling get_snapshot() repeatedly has zero side effects on clock, pedestrians, cadence, history, or randomness."""
    reg = PedestrianRegistry()
    ped = PedestrianState(
        id="ped_snap_invariance",
        position=Point(5.0, 15.0),
        current_area_id="main_corridor",
        target_exit_id="exit_left",
        preferred_speed=1.0,
    )
    reg.register(ped)
    engine = SimulationEngine(initial_building, reg, dt=0.1, observation_interval=0.4)

    # Step to step 2 (intermediate non-observation step)
    engine.step()
    engine.step()

    step_before = engine.clock.step_count
    time_before = engine.clock.timestamp
    ped_pos_before = Point(ped.position.x, ped.position.y)
    ped_vel_before = Velocity2D(ped.velocity.vx, ped.velocity.vy)
    ped_status_before = ped.status
    ped_hist_len_before = len(ped.trajectory_history)
    events_len_before = len(engine.events)
    latest_snapshot_before = engine.latest_snapshot
    rng_state_before = random.getstate()

    # Call get_snapshot() repeatedly
    snapshots = [engine.get_snapshot() for _ in range(5)]

    # 1. Clock step and timestamp unchanged
    assert engine.clock.step_count == step_before
    assert engine.clock.timestamp == time_before

    # 2. Pedestrian positions, velocities, and statuses unchanged
    assert ped.position == ped_pos_before
    assert ped.velocity == ped_vel_before
    assert ped.status == ped_status_before
    assert len(ped.trajectory_history) == ped_hist_len_before

    # 3. Events unchanged
    assert len(engine.events) == events_len_before

    # 4. Simulation randomness not consumed
    assert random.getstate() == rng_state_before

    # 5. Existing snapshot history / latest_snapshot unchanged
    assert engine.latest_snapshot is latest_snapshot_before

    # 6. Snapshots returned are identical in content
    for s in snapshots:
        assert s.step_count == step_before
        assert s.timestamp == time_before
        assert s.total_count == 1
        assert s.pedestrians[0].pedestrian_id == "ped_snap_invariance"

    # 7. Observation cadence unchanged
    step3 = engine.step()
    assert step3 is None
    step4 = engine.step()
    assert step4 is not None
    assert step4.step_count == 4


def test_engine_open_exit_updates_state_and_tracking(initial_building: Building):
    """Verify open_exit() updates exit model state and engine tracking consistently."""
    engine = SimulationEngine(initial_building)
    ex = initial_building.get_exit("exit_left")
    assert ex is not None
    assert ex.is_open is True
    assert "exit_left" in engine._open_exit_ids

    # Close exit
    assert engine.close_exit("exit_left") is True
    assert ex.is_open is False
    assert "exit_left" not in engine._open_exit_ids

    # Reopen exit
    assert engine.open_exit("exit_left") is True
    assert ex.is_open is True
    assert "exit_left" in engine._open_exit_ids

    # Reopening already open exit returns True safely
    assert engine.open_exit("exit_left") is True

    # Non-existent exit returns False
    assert engine.open_exit("non_existent_exit") is False


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


# =============================================================================
# Week 3 Final Corrections — Task 1: Radius-Aware Initial Pedestrian Validation
# =============================================================================


def test_task1_valid_initial_position_open_room(initial_building: Building):
    """1. Valid initial position in an open room initializes successfully."""
    reg = PedestrianRegistry()
    reg.register(
        PedestrianState(
            id="p_valid_room",
            position=Point(2.0, 5.0),
            radius=0.25,
            current_area_id="room_1",
            target_exit_id="exit_left",
        )
    )
    engine = SimulationEngine(initial_building, reg)
    assert engine.registry.total_count == 1
    assert engine.registry.get("p_valid_room").status == PedestrianStatus.ACTIVE


def test_task1_initial_position_inside_obstacle(initial_building: Building):
    """2. Initial position inside an obstacle raises ValueError identifying pedestrian."""
    # obstacle_room1 is [4.0, 6.0] x [4.0, 5.5]
    reg = PedestrianRegistry()
    reg.register(
        PedestrianState(
            id="p_in_obs",
            position=Point(5.0, 4.75),
            radius=0.25,
            current_area_id="room_1",
            target_exit_id="exit_left",
        )
    )
    with pytest.raises(ValueError, match=r"Pedestrian 'p_in_obs'.*violates walkable space clearance"):
        SimulationEngine(initial_building, reg)


def test_task1_initial_position_intersecting_wall(initial_building: Building):
    """3. Initial position intersecting a wall raises ValueError identifying pedestrian."""
    # Dividing wall wall_int_div_north is [19.8, 20.2] x [0.0, 10.0]
    reg = PedestrianRegistry()
    reg.register(
        PedestrianState(
            id="p_in_wall",
            position=Point(20.0, 5.0),
            radius=0.25,
            current_area_id="room_1",
            target_exit_id="exit_left",
        )
    )
    with pytest.raises(ValueError, match=r"Pedestrian 'p_in_wall'.*violates walkable space clearance"):
        SimulationEngine(initial_building, reg)


def test_task1_initial_position_violating_radius_clearance(initial_building: Building):
    """4. Initial position violating pedestrian-radius clearance to a wall raises ValueError."""
    # Room 1 left wall is at x=0.0. A pedestrian with radius=0.25 at x=0.1 penetrates the wall boundary.
    reg = PedestrianRegistry()
    reg.register(
        PedestrianState(
            id="p_wall_clearance",
            position=Point(0.1, 5.0),
            radius=0.25,
            current_area_id="room_1",
            target_exit_id="exit_left",
        )
    )
    with pytest.raises(ValueError, match=r"Pedestrian 'p_wall_clearance'.*violates walkable space clearance"):
        SimulationEngine(initial_building, reg)


def test_task1_valid_initial_position_near_doorway(initial_building: Building):
    """5. Valid initial position near/in a doorway initializes successfully."""
    # doorway door_room1_corridor is at y=13.5, spanning x in [9.0, 11.0]
    reg = PedestrianRegistry()
    reg.register(
        PedestrianState(
            id="p_doorway",
            position=Point(10.0, 13.5),
            radius=0.25,
            current_area_id="room_1",
            target_exit_id="exit_left",
        )
    )
    engine = SimulationEngine(initial_building, reg)
    assert engine.registry.get("p_doorway").status == PedestrianStatus.ACTIVE


def test_task1_valid_initial_position_in_corridor(initial_building: Building):
    """6. Valid initial position in a corridor initializes successfully."""
    reg = PedestrianRegistry()
    reg.register(
        PedestrianState(
            id="p_corridor",
            position=Point(15.0, 15.0),
            radius=0.25,
            current_area_id="main_corridor",
            target_exit_id="exit_left",
        )
    )
    engine = SimulationEngine(initial_building, reg)
    assert engine.registry.get("p_corridor").status == PedestrianStatus.ACTIVE


def test_task1_invalid_position_outside_walkable_space(initial_building: Building):
    """7. Invalid position outside walkable space raises ValueError."""
    reg = PedestrianRegistry()
    reg.register(
        PedestrianState(
            id="p_out_of_bounds",
            position=Point(45.0, 15.0),
            radius=0.25,
        )
    )
    with pytest.raises(ValueError, match="is outside building bounds"):
        SimulationEngine(initial_building, reg)


def test_task1_multiple_pedestrians_one_invalid_fails_safely(initial_building: Building):
    """8. Multiple pedestrians where one initial position is invalid fails safely without partial state."""
    reg = PedestrianRegistry()
    reg.register(PedestrianState(id="p1_valid", position=Point(2.0, 5.0), radius=0.25, target_exit_id="exit_left"))
    reg.register(PedestrianState(id="p2_valid", position=Point(10.0, 15.0), radius=0.25, target_exit_id="exit_left"))
    reg.register(PedestrianState(id="p3_bad", position=Point(5.0, 4.75), radius=0.25, target_exit_id="exit_left"))  # inside obstacle

    with pytest.raises(ValueError, match=r"Pedestrian 'p3_bad'.*violates walkable space clearance"):
        SimulationEngine(initial_building, reg)


# =============================================================================
# Week 3 Final Corrections — Task 2: Initial Overlap Population Validation
# =============================================================================


def test_task2_identical_starting_positions_rejected(initial_building: Building):
    """1. Two pedestrians starting with identical coordinates are rejected with both IDs."""
    reg = PedestrianRegistry()
    reg.register(PedestrianState(id="ped_alpha", position=Point(10.0, 15.0), radius=0.25, target_exit_id="exit_left"))
    reg.register(PedestrianState(id="ped_beta", position=Point(10.0, 15.0), radius=0.25, target_exit_id="exit_left"))

    with pytest.raises(ValueError) as excinfo:
        SimulationEngine(initial_building, reg)
    assert "Initial pedestrian overlap detected" in str(excinfo.value)
    assert "ped_alpha" in str(excinfo.value)
    assert "ped_beta" in str(excinfo.value)


def test_task2_overlapping_starting_positions_rejected(initial_building: Building):
    """2. Two pedestrians starting in physical overlap (distance < r1 + r2) are rejected."""
    reg = PedestrianRegistry()
    # Distance = 0.3m, sum of radii = 0.5m -> overlap
    reg.register(PedestrianState(id="ped_1", position=Point(10.0, 15.0), radius=0.25, target_exit_id="exit_left"))
    reg.register(PedestrianState(id="ped_2", position=Point(10.3, 15.0), radius=0.25, target_exit_id="exit_left"))

    with pytest.raises(ValueError) as excinfo:
        SimulationEngine(initial_building, reg)
    assert "Initial pedestrian overlap detected" in str(excinfo.value)
    assert "ped_1" in str(excinfo.value)
    assert "ped_2" in str(excinfo.value)


def test_task2_contact_threshold_starting_positions_allowed(initial_building: Building):
    """3. Two pedestrians starting exactly at contact threshold (dist == r1 + r2) are permitted."""
    reg = PedestrianRegistry()
    # Distance = 0.5m, sum of radii = 0.5m
    reg.register(PedestrianState(id="ped_touch1", position=Point(10.0, 15.0), radius=0.25, target_exit_id="exit_left"))
    reg.register(PedestrianState(id="ped_touch2", position=Point(10.5, 15.0), radius=0.25, target_exit_id="exit_left"))

    engine = SimulationEngine(initial_building, reg)
    assert engine.registry.total_count == 2


def test_task2_separated_valid_distance_allowed(initial_building: Building):
    """4. Two pedestrians separated by valid clearance are permitted."""
    reg = PedestrianRegistry()
    reg.register(PedestrianState(id="ped_sep1", position=Point(10.0, 15.0), radius=0.25, target_exit_id="exit_left"))
    reg.register(PedestrianState(id="ped_sep2", position=Point(12.0, 15.0), radius=0.25, target_exit_id="exit_left"))

    engine = SimulationEngine(initial_building, reg)
    assert engine.registry.total_count == 2


def test_task2_multiple_overlapping_pedestrians_identified(initial_building: Building):
    """5. Multiple overlapping pedestrians clearly identify the offending IDs."""
    reg = PedestrianRegistry()
    reg.register(PedestrianState(id="ped_safe", position=Point(2.0, 5.0), radius=0.25, target_exit_id="exit_left"))
    reg.register(PedestrianState(id="ped_clash_a", position=Point(10.0, 15.0), radius=0.25, target_exit_id="exit_left"))
    reg.register(PedestrianState(id="ped_clash_b", position=Point(10.2, 15.0), radius=0.25, target_exit_id="exit_left"))

    with pytest.raises(ValueError) as excinfo:
        SimulationEngine(initial_building, reg)
    assert "ped_clash_a" in str(excinfo.value)
    assert "ped_clash_b" in str(excinfo.value)


def test_task2_initial_separation_mandatory(initial_building: Building):
    """6. Initial separation validation is mandatory in SimulationEngine with no bypass mechanism."""
    reg = PedestrianRegistry()
    reg.register(PedestrianState(id="ped_m1", position=Point(10.0, 15.0), radius=0.25, target_exit_id="exit_left"))
    reg.register(PedestrianState(id="ped_m2", position=Point(10.0, 15.0), radius=0.25, target_exit_id="exit_left"))

    with pytest.raises(ValueError, match="Initial pedestrian overlap detected"):
        SimulationEngine(initial_building, reg)


# =============================================================================
# Additional Boundary Validation Tests (Room boundaries, doorway edges, exits)
# =============================================================================


def test_task1_initial_position_near_room_boundary_edges(initial_building: Building):
    """Verify valid clearance vs invalid intrusion near room boundaries."""
    # Left wall footprint is [0.0, 0.2] x [0.0, 14.0]. Radius = 0.25m.
    # Valid: Point(0.5, 5.0) -> left edge at 0.25m > 0.2m (clearance = 0.05m)
    reg_valid = PedestrianRegistry()
    reg_valid.register(PedestrianState(id="p_valid_edge", position=Point(0.5, 5.0), radius=0.25, current_area_id="room_1", target_exit_id="exit_left"))
    engine = SimulationEngine(initial_building, reg_valid)
    assert engine.registry.get("p_valid_edge").status == PedestrianStatus.ACTIVE

    # Invalid: Point(0.35, 5.0) -> left edge at 0.10m < 0.2m (penetrates left exterior wall [0, 0.2])
    reg_invalid = PedestrianRegistry()
    reg_invalid.register(PedestrianState(id="p_invalid_edge", position=Point(0.35, 5.0), radius=0.25, current_area_id="room_1", target_exit_id="exit_left"))
    with pytest.raises(ValueError, match="violates walkable space clearance"):
        SimulationEngine(initial_building, reg_invalid)


def test_task1_initial_position_near_doorway_edges(initial_building: Building):
    """Verify valid opening clearance vs invalid wall-jamb intrusion near doorways."""
    # door_room1_corridor is at y=13.5 with opening x in [9.0, 11.0]
    # Valid: centered at x=10.0, y=13.5 (clears jambs by 0.75m on both sides)
    reg_valid = PedestrianRegistry()
    reg_valid.register(PedestrianState(id="p_door_center", position=Point(10.0, 13.5), radius=0.25, current_area_id="room_1", target_exit_id="exit_left"))
    engine = SimulationEngine(initial_building, reg_valid)
    assert engine.registry.get("p_door_center").status == PedestrianStatus.ACTIVE

    # Invalid: at x=9.1, y=13.5 with radius 0.25, left edge reaches x=8.85 which penetrates solid wall [0, 9.0]
    reg_invalid = PedestrianRegistry()
    reg_invalid.register(PedestrianState(id="p_door_jamb", position=Point(9.1, 13.5), radius=0.25, current_area_id="room_1", target_exit_id="exit_left"))
    with pytest.raises(ValueError, match="violates walkable space clearance"):
        SimulationEngine(initial_building, reg_invalid)


def test_task1_initial_position_near_exit_boundaries(initial_building: Building):
    """Verify exit boundary clearance rules for assigned vs unassigned exits."""
    # exit_left is at x=0.0, opening y in [13.5, 16.5]
    # Valid for assigned exit: Point(0.3, 15.0) heading to exit_left
    reg_assigned = PedestrianRegistry()
    reg_assigned.register(PedestrianState(id="p_exit_assigned", position=Point(0.3, 15.0), radius=0.25, current_area_id="main_corridor", target_exit_id="exit_left"))
    engine = SimulationEngine(initial_building, reg_assigned)
    assert engine.registry.get("p_exit_assigned").status == PedestrianStatus.ACTIVE

    # Invalid for unassigned exit: Point(0.2, 15.0) with target_exit_id="exit_right"
    # For exit_right, the boundary at x=0 is a solid exterior wall!
    reg_unassigned = PedestrianRegistry()
    reg_unassigned.register(PedestrianState(id="p_exit_unassigned", position=Point(0.2, 15.0), radius=0.25, current_area_id="main_corridor", target_exit_id="exit_right"))
    with pytest.raises(ValueError, match="violates walkable space clearance"):
        SimulationEngine(initial_building, reg_unassigned)


