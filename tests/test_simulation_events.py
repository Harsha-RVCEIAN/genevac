"""Tests for GenEvac-Sentinel Week 3 Phase 1: Exit State, Structured Events, Safe No-Route Handling."""

import random
import pytest
from genevac.environment.builder import create_initial_building
from genevac.environment.geometry import LineSegment, Point, Rectangle, Wall, WallAlignment, WallType
from genevac.environment.models import Building, Exit, WalkableArea
from genevac.environment.serialization import exit_from_dict, exit_to_dict
from genevac.pedestrian.models import PedestrianState, PedestrianStatus, Velocity2D
from genevac.simulation.engine import SimulationEngine
from genevac.simulation.events import SimulationEvent, SimulationEventType
from genevac.simulation.exceptions import NoRouteError
from genevac.simulation.movement import (
    advance_pedestrian_step,
    check_collision,
    detect_exit_crossing,
    generate_deterministic_route,
)


# ---------------------------------------------------------------------------
# Part A: Exit State Tests
# ---------------------------------------------------------------------------

def test_exit_defaults_to_open():
    """Verify that an Exit defaults to is_open=True and can be explicitly initialized."""
    ex_default = Exit(
        id="ex_1",
        segment=LineSegment(Point(0.0, 10.0), Point(0.0, 14.0)),
        width=4.0,
        boundary="left",
        connected_area_id="area_main",
    )
    assert ex_default.is_open is True

    ex_closed = Exit(
        id="ex_2",
        segment=LineSegment(Point(40.0, 10.0), Point(40.0, 14.0)),
        width=4.0,
        boundary="right",
        connected_area_id="area_main",
        is_open=False,
    )
    assert ex_closed.is_open is False


def test_exit_close_open_state_and_serialization_round_trip():
    """Verify open/close operations and JSON serialization round-trip preserving is_open."""
    ex = Exit(
        id="ex_test",
        segment=LineSegment(Point(0.0, 10.0), Point(0.0, 14.0)),
        width=4.0,
        boundary="left",
        connected_area_id="area_1",
        is_open=True,
    )
    # Test close/open operations
    ex.close()
    assert ex.is_open is False
    ex.open()
    assert ex.is_open is True
    ex.close()
    assert ex.is_open is False

    # Serialization round-trip with is_open=False
    data = exit_to_dict(ex)
    assert data["is_open"] is False
    restored = exit_from_dict(data)
    assert restored.is_open is False
    assert restored.id == ex.id
    assert restored.boundary == ex.boundary
    assert restored.width == ex.width

    # Serialization round-trip with is_open=True
    ex.open()
    data_open = exit_to_dict(ex)
    assert data_open["is_open"] is True
    restored_open = exit_from_dict(data_open)
    assert restored_open.is_open is True

    # Backward compatibility: data missing "is_open" defaults to True
    del data["is_open"]
    legacy_restored = exit_from_dict(data)
    assert legacy_restored.is_open is True


# ---------------------------------------------------------------------------
# Part B & D: Closed Exit Collision & Crossing Tests
# ---------------------------------------------------------------------------

def test_closed_exit_cannot_be_crossed():
    """Verify that a closed exit rejects evacuation crossing."""
    building = create_initial_building()
    ex = building.get_exit("exit_left")
    assert ex is not None
    assert ex.is_open is True

    # Moving from inside (x=0.5) to outside (x=-0.5) across open exit (y in [14.0, 16.0])
    start_pos = Point(0.5, 15.0)
    end_pos = Point(-0.5, 15.0)
    crossed, crossed_pt = detect_exit_crossing(start_pos, end_pos, radius=0.3, exit_obj=ex)
    assert crossed is True
    assert crossed_pt is not None
    assert crossed_pt.x == 0.0

    # Close the exit
    ex.close()
    assert ex.is_open is False

    # Crossing must now be rejected
    crossed, crossed_pt = detect_exit_crossing(start_pos, end_pos, radius=0.3, exit_obj=ex)
    assert crossed is False
    assert crossed_pt is None


def test_closed_exit_behaves_as_solid_boundary_with_radius_clearance():
    """Verify that a closed exit behaves as a solid boundary enforcing pedestrian radius clearance."""
    building = create_initial_building()
    ex = building.get_exit("exit_left")
    assert ex is not None

    # When exit is open, entering the boundary opening (x=0.2, crossing x=0 exterior with radius 0.3) is allowed
    collides_open = check_collision(
        start=Point(0.5, 15.0),
        end=Point(0.2, 15.0),
        radius=0.3,
        building=building,
        target_exit=ex,
    )
    assert collides_open is False

    # Close the exit
    ex.close()

    # Now that the exit is closed, entering boundary opening violates radius clearance (< 0.3 from boundary)
    collides_closed = check_collision(
        start=Point(0.5, 15.0),
        end=Point(0.2, 15.0),
        radius=0.3,
        building=building,
        target_exit=ex,
    )
    assert collides_closed is True

    # Safe clearance: x = 0.35 (> radius 0.3) does not penetrate boundary and is safe
    collides_safe = check_collision(
        start=Point(0.5, 15.0),
        end=Point(0.35, 15.0),
        radius=0.3,
        building=building,
        target_exit=ex,
    )
    assert collides_safe is False


def test_exit_closed_during_active_simulation():
    """Verify exit closure during active simulation: pedestrian is blocked, EXIT_CLOSED emitted, no duplicate events."""
    building = create_initial_building()
    # Pedestrian starting in main_corridor heading to exit_left
    ped = PedestrianState(
        id="ped_approaching",
        position=Point(5.0, 15.0),
        current_area_id="main_corridor",
        target_exit_id="exit_left",
        preferred_speed=1.0,
    )
    engine = SimulationEngine(building=building, registry=[ped])
    assert ped.status == PedestrianStatus.ACTIVE

    # Step twice while exit is open
    engine.step()
    engine.step()
    assert ped.status == PedestrianStatus.ACTIVE
    initial_event_count = len(engine.events)

    # Close exit_left mid-simulation
    success = engine.close_exit("exit_left")
    assert success is True

    # Events recorded immediately on closure: EXIT_CLOSED and NO_ROUTE
    assert len(engine.events) == initial_event_count + 2
    evt_exit = engine.events[-2]
    evt_noroute = engine.events[-1]

    assert evt_exit.event_type == SimulationEventType.EXIT_CLOSED
    assert evt_exit.details["exit_id"] == "exit_left"

    assert evt_noroute.event_type == SimulationEventType.NO_ROUTE
    assert evt_noroute.pedestrian_id == "ped_approaching"
    assert ped.status == PedestrianStatus.BLOCKED
    assert ped.velocity == Velocity2D(0.0, 0.0)
    assert len(ped.waypoints) == 0

    # Advance several more steps - verify NO duplicate events are generated every frame
    events_after_close = len(engine.events)
    for _ in range(5):
        engine.step()

    # Event count must remain unchanged
    assert len(engine.events) == events_after_close
    assert ped.status == PedestrianStatus.BLOCKED


def test_open_exit_updates_state_and_tracking_consistently():
    """Verify that open_exit() updates exit model state and engine tracking consistently."""
    building = create_initial_building()
    engine = SimulationEngine(building=building)

    ex = building.get_exit("exit_left")
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

    # Reopening an already open exit returns True safely without error
    assert engine.open_exit("exit_left") is True

    # Non-existent exit returns False
    assert engine.open_exit("non_existent_exit") is False


# ---------------------------------------------------------------------------
# Part C & Task 1: Safe No-Route Handling & Unassigned Exit Tests
# ---------------------------------------------------------------------------

def test_unassigned_exit_default_blocks_and_records_event():
    """Verify that by default, an unassigned-exit pedestrian safely transitions to BLOCKED, sets zero velocity, clears waypoints, and records NO_ROUTE."""
    building = create_initial_building()
    ped_unassigned = PedestrianState(
        id="ped_unassigned",
        position=Point(10.0, 15.0),
        current_area_id="main_corridor",
        target_exit_id=None,
        waypoints=[Point(12.0, 15.0)],  # Invalid waypoints without an assigned exit
    )
    ped_valid = PedestrianState(
        id="ped_valid",
        position=Point(2.0, 15.0),
        current_area_id="main_corridor",
        target_exit_id="exit_left",
        preferred_speed=1.0,
    )
    # Default constructor has block_unassigned_exits=True
    engine = SimulationEngine(building=building, registry=[ped_unassigned, ped_valid])

    # Unassigned pedestrian is immediately BLOCKED with zero velocity and cleared waypoints
    assert ped_unassigned.status == PedestrianStatus.BLOCKED
    assert ped_unassigned.velocity == Velocity2D(0.0, 0.0)
    assert len(ped_unassigned.waypoints) == 0

    # Valid pedestrian remains ACTIVE
    assert ped_valid.status == PedestrianStatus.ACTIVE

    # Exactly one NO_ROUTE event recorded for the unassigned pedestrian
    unassigned_events = [e for e in engine.events if e.pedestrian_id == "ped_unassigned"]
    assert len(unassigned_events) == 1
    assert unassigned_events[0].event_type == SimulationEventType.NO_ROUTE
    assert unassigned_events[0].details["reason"] == "unassigned_exit"

    # Valid pedestrian continues simulating and evacuates normally
    for _ in range(30):
        engine.step()
        if ped_valid.status == PedestrianStatus.EVACUATED:
            break

    assert ped_valid.status == PedestrianStatus.EVACUATED
    assert ped_unassigned.status == PedestrianStatus.BLOCKED


def test_unassigned_exit_legacy_mode_allowed():
    """Verify that passing block_unassigned_exits=False permits unassigned pedestrians to remain ACTIVE."""
    building = create_initial_building()
    ped = PedestrianState(
        id="ped_legacy_unassigned",
        position=Point(10.0, 15.0),
        current_area_id="main_corridor",
        target_exit_id=None,
    )
    engine = SimulationEngine(building=building, registry=[ped], block_unassigned_exits=False)

    assert ped.status == PedestrianStatus.ACTIVE
    assert len(engine.events) == 0


def test_unreachable_exit_does_not_crash_engine():
    """Verify that a genuine unreachable route raises NoRouteError internally, does not crash engine, and sets BLOCKED."""
    # Standard building with room_1 disconnected from corridor
    building = create_initial_building()
    building.doorways = [d for d in building.doorways if not d.connects("room_1", "main_corridor")]

    # Direct check that generate_deterministic_route raises NoRouteError
    with pytest.raises(NoRouteError):
        generate_deterministic_route(
            building=building,
            start_pos=Point(2.0, 5.0),
            start_area_id="room_1",
            target_exit_id="exit_left",
        )

    # Engine initialization must safely catch NoRouteError and not crash
    ped = PedestrianState(
        id="ped_trapped",
        position=Point(2.0, 5.0),
        current_area_id="room_1",
        target_exit_id="exit_left",
    )
    engine = SimulationEngine(building=building, registry=[ped])

    assert ped.status == PedestrianStatus.BLOCKED
    assert ped.velocity == Velocity2D(0.0, 0.0)
    assert len(ped.waypoints) == 0

    events = engine.events
    assert len(events) == 1
    assert events[0].event_type == SimulationEventType.NO_ROUTE
    assert events[0].pedestrian_id == "ped_trapped"
    assert events[0].details["reason"] == "unreachable_exit"


def test_unknown_start_area_raises_value_error_directly():
    """Verify that an unknown start area raises ValueError directly and is NOT converted to NO_ROUTE."""
    building = create_initial_building()
    ped = PedestrianState(
        id="ped_bad_area",
        position=Point(2.0, 5.0),
        current_area_id="non_existent_area",
        target_exit_id="exit_left",
    )
    with pytest.raises(ValueError, match="non-existent walkable area") as excinfo:
        SimulationEngine(building=building, registry=[ped])
    assert type(excinfo.value) is ValueError
    assert not isinstance(excinfo.value, NoRouteError)

    # Calling generate_deterministic_route directly must also raise ValueError, NOT NoRouteError
    with pytest.raises(ValueError, match="Unknown start walkable area") as excinfo_route:
        generate_deterministic_route(
            building=building,
            start_pos=Point(2.0, 5.0),
            start_area_id="non_existent_area",
            target_exit_id="exit_left",
        )
    assert type(excinfo_route.value) is ValueError
    assert not isinstance(excinfo_route.value, NoRouteError)


def test_unknown_target_exit_raises_value_error_directly():
    """Verify that an unknown target exit raises ValueError directly and is NOT converted to NO_ROUTE."""
    building = create_initial_building()
    ped = PedestrianState(
        id="ped_bad_exit",
        position=Point(2.0, 5.0),
        current_area_id="room_1",
        target_exit_id="non_existent_exit",
    )
    with pytest.raises(ValueError, match="non-existent target exit") as excinfo:
        SimulationEngine(building=building, registry=[ped])
    assert type(excinfo.value) is ValueError
    assert not isinstance(excinfo.value, NoRouteError)

    # Calling generate_deterministic_route directly must also raise ValueError, NOT NoRouteError
    with pytest.raises(ValueError, match="Unknown target exit") as excinfo_route:
        generate_deterministic_route(
            building=building,
            start_pos=Point(2.0, 5.0),
            start_area_id="room_1",
            target_exit_id="non_existent_exit",
        )
    assert type(excinfo_route.value) is ValueError
    assert not isinstance(excinfo_route.value, NoRouteError)


def test_one_unreachable_pedestrian_does_not_prevent_other_valid_pedestrians_from_simulating():
    """Verify that one unreachable pedestrian does not prevent other valid pedestrians from simulating."""
    building = create_initial_building()
    building.doorways = [d for d in building.doorways if not d.connects("room_1", "main_corridor")]

    ped_invalid = PedestrianState(
        id="ped_invalid",
        position=Point(2.0, 5.0),
        current_area_id="room_1",
        target_exit_id="exit_left",
    )
    ped_valid = PedestrianState(
        id="ped_valid",
        position=Point(2.0, 15.0),
        current_area_id="main_corridor",
        target_exit_id="exit_left",
        preferred_speed=1.0,
    )

    engine = SimulationEngine(building=building, registry=[ped_invalid, ped_valid])

    assert ped_invalid.status == PedestrianStatus.BLOCKED
    assert ped_valid.status == PedestrianStatus.ACTIVE

    # Step simulation until ped_valid evacuates
    for _ in range(30):
        engine.step()
        if ped_valid.status == PedestrianStatus.EVACUATED:
            break

    assert ped_valid.status == PedestrianStatus.EVACUATED
    assert ped_invalid.status == PedestrianStatus.BLOCKED


# ---------------------------------------------------------------------------
# Task 2: Deep Event Immutability Tests
# ---------------------------------------------------------------------------

def test_event_immutability_top_level_rejected():
    """Verify that attempting to mutate top-level event details raises an error."""
    evt = SimulationEvent(
        event_id="evt_001",
        step=1,
        timestamp=0.1,
        event_type="test_event",
        details={"key": "initial_value"},
    )
    with pytest.raises(TypeError):
        evt.details["key"] = "modified"  # type: ignore[index]

    with pytest.raises(TypeError):
        evt.details["new_key"] = "added"  # type: ignore[index]

    with pytest.raises(TypeError):
        del evt.details["key"]  # type: ignore[misc]

    with pytest.raises(AttributeError):
        evt.details.clear()  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        evt.details.pop("key")  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        evt.details.update({"key": "val"})  # type: ignore[attr-defined]

    assert evt.details["key"] == "initial_value"
    assert "new_key" not in evt.details


def test_event_immutability_nested_dict_rejected():
    """Verify that attempting to mutate a nested dictionary in event details raises an error."""
    evt = SimulationEvent(
        event_id="evt_002",
        step=1,
        timestamp=0.1,
        event_type="test_event",
        details={"nested": {"inner_key": 42, "level2": {"level3": "deep"}}},
    )
    assert evt.details["nested"]["inner_key"] == 42
    with pytest.raises(TypeError):
        evt.details["nested"]["inner_key"] = 999  # type: ignore[index]

    with pytest.raises(TypeError):
        evt.details["nested"]["new_inner"] = "bad"  # type: ignore[index]

    with pytest.raises(TypeError):
        del evt.details["nested"]["inner_key"]  # type: ignore[misc]

    with pytest.raises(TypeError):
        evt.details["nested"]["level2"]["level3"] = "changed"  # type: ignore[index]

    assert evt.details["nested"]["inner_key"] == 42
    assert evt.details["nested"]["level2"]["level3"] == "deep"


def test_event_immutability_nested_list_rejected():
    """Verify that attempting to mutate a nested list/sequence in event details raises an error."""
    evt = SimulationEvent(
        event_id="evt_003",
        step=1,
        timestamp=0.1,
        event_type="test_event",
        details={"items": [1, 2, 3]},
    )
    # The list is recursively frozen to an immutable tuple
    assert isinstance(evt.details["items"], tuple)
    assert evt.details["items"] == (1, 2, 3)

    with pytest.raises(AttributeError):
        evt.details["items"].append(4)  # type: ignore[attr-defined]

    with pytest.raises(TypeError):
        evt.details["items"][0] = 99  # type: ignore[index]

    with pytest.raises(AttributeError):
        evt.details["items"].extend([4, 5])  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        evt.details["items"].pop()  # type: ignore[attr-defined]

    assert evt.details["items"] == (1, 2, 3)


def test_event_immutability_nested_set_rejected():
    """Verify that attempting to mutate a nested set in event details raises an error."""
    evt = SimulationEvent(
        event_id="evt_003_set",
        step=1,
        timestamp=0.1,
        event_type="test_event",
        details={"tags": {"door", "exit"}},
    )
    # The set is recursively frozen to a frozenset
    assert isinstance(evt.details["tags"], frozenset)
    assert evt.details["tags"] == frozenset({"door", "exit"})

    with pytest.raises(AttributeError):
        evt.details["tags"].add("corridor")  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        evt.details["tags"].remove("door")  # type: ignore[attr-defined]

    assert evt.details["tags"] == frozenset({"door", "exit"})


def test_event_immutability_input_mutation_isolation():
    """Verify that mutating input dictionaries, nested structures, or sets after creation does NOT affect the stored event."""
    raw_nested = {"status": "ok"}
    raw_list = [10, 20]
    raw_set = {"a", "b"}
    raw_details = {
        "nested": raw_nested,
        "list": raw_list,
        "set": raw_set,
    }

    evt = SimulationEvent(
        event_id="evt_004",
        step=1,
        timestamp=0.1,
        event_type="test_event",
        details=raw_details,
    )

    # Mutate the original external input structures
    raw_nested["status"] = "tampered"
    raw_list.append(30)
    raw_set.add("c")
    raw_details["new_root_entry"] = "injected"

    # Stored event details must remain strictly intact
    assert evt.details["nested"]["status"] == "ok"
    assert evt.details["list"] == (10, 20)
    assert evt.details["set"] == frozenset({"a", "b"})
    assert "new_root_entry" not in evt.details

    # to_dict roundtrip is JSON-compatible
    serialized = evt.to_dict()
    assert serialized["details"]["nested"]["status"] == "ok"
    assert serialized["details"]["list"] == [10, 20]


# ---------------------------------------------------------------------------
# Part B: Event Integrity, Determinism & Clock Synchronization Tests
# ---------------------------------------------------------------------------

def test_event_timestamps_and_step_indices_match_engine_clock():
    """Verify that recorded event timestamps and step indices match the engine clock precisely."""
    building = create_initial_building()
    ped = PedestrianState(
        id="ped_clock_test",
        position=Point(5.0, 15.0),
        current_area_id="main_corridor",
        target_exit_id="exit_left",
    )
    engine = SimulationEngine(building=building, registry=[ped], dt=0.1)

    # Run 4 steps (t = 0.4, step = 4)
    engine.run_steps(4)
    assert engine.clock.step_count == 4
    assert round(engine.clock.timestamp, 2) == 0.4

    # Close exit at step 4
    engine.close_exit("exit_left")

    latest = engine.latest_events
    assert len(latest) >= 1
    for evt in latest:
        assert evt.step == 4
        assert round(evt.timestamp, 2) == 0.4


def test_event_ids_are_deterministic_and_unique_within_a_run():
    """Verify that event IDs are deterministic across runs and unique within each run."""
    def run_sim():
        building = create_initial_building()
        ped = PedestrianState(
            id="ped_det",
            position=Point(5.0, 15.0),
            current_area_id="main_corridor",
            target_exit_id="exit_left",
        )
        eng = SimulationEngine(building=building, registry=[ped])
        eng.step()
        eng.close_exit("exit_left")
        eng.step()
        return eng.events

    events_run1 = run_sim()
    events_run2 = run_sim()

    # Determinism across identical runs
    assert len(events_run1) == len(events_run2)
    for e1, e2 in zip(events_run1, events_run2):
        assert e1.event_id == e2.event_id
        assert e1.step == e2.step
        assert e1.timestamp == e2.timestamp
        assert e1.event_type == e2.event_type
        assert e1.pedestrian_id == e2.pedestrian_id
        assert e1.details == e2.details

    # Uniqueness within a single run
    ids = [e.event_id for e1 in events_run1 for e in [e1]]
    assert len(ids) == len(set(ids))


def test_existing_snapshot_immutability_remains_intact():
    """Verify snapshot immutability: subsequent events or pedestrian changes do not alter earlier snapshots."""
    building = create_initial_building()
    ped = PedestrianState(
        id="ped_snap",
        position=Point(5.0, 15.0),
        current_area_id="main_corridor",
        target_exit_id="exit_left",
    )
    engine = SimulationEngine(building=building, registry=[ped])

    # Initial snapshot at step 0
    snap_t0 = engine.latest_snapshot
    assert snap_t0.pedestrians[0].status == PedestrianStatus.ACTIVE

    # Advance steps and close exit
    engine.step()
    engine.close_exit("exit_left")
    assert ped.status == PedestrianStatus.BLOCKED

    # Historical snapshot t0 must remain ACTIVE and unchanged
    assert snap_t0.pedestrians[0].status == PedestrianStatus.ACTIVE
    assert snap_t0.step_count == 0
    assert snap_t0.timestamp == 0.0

    # Snapshot to_dict roundtrip remains valid
    d = snap_t0.to_dict()
    assert d["step_count"] == 0
    assert len(d["pedestrians"]) == 1
    assert d["pedestrians"][0]["status"] == "active"


# ---------------------------------------------------------------------------
# Task 4: get_snapshot() Invariance Tests
# ---------------------------------------------------------------------------

def test_get_snapshot_repeated_calls_are_strictly_side_effect_free():
    """Verify that calling get_snapshot() repeatedly has zero side effects on clock, pedestrians, cadence, or history."""
    building = create_initial_building()
    ped = PedestrianState(
        id="ped_snap_invariance",
        position=Point(5.0, 15.0),
        current_area_id="main_corridor",
        target_exit_id="exit_left",
        preferred_speed=1.0,
    )
    engine = SimulationEngine(building=building, registry=[ped], dt=0.1, observation_interval=0.4)

    # Step simulation to step 2 (intermediate step, non-observation tick)
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

    # 2. Pedestrian state completely unchanged
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

    # 7. Subsequent steps advance cadence as normal (observation step occurs at step 4)
    step3_result = engine.step()  # step 3
    assert step3_result is None
    step4_result = engine.step()  # step 4 (observation step)
    assert step4_result is not None
    assert step4_result.step_count == 4
