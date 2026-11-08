"""End-to-end integration tests for GenEvac-Sentinel Phase 5 final sign-off."""

import os
from pathlib import Path
import subprocess
import sys
import pytest

# Ensure headless operation for Pygame testing
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

from genevac.environment.builder import create_initial_building
from genevac.environment.geometry import (
    LineSegment,
    Point,
    Rectangle,
    Wall,
    WallAlignment,
    WallType,
)
from genevac.environment.models import (
    Building,
    Doorway,
    Exit,
    Obstacle,
    WalkableArea,
)
from genevac.environment.serialization import (
    SCHEMA_VERSION,
    building_from_dict,
    building_to_dict,
    load_building,
    save_building,
)
from genevac.visualization.coordinates import Viewport
from genevac.visualization.renderer import EnvironmentRenderer, draw_building


@pytest.fixture(scope="module", autouse=True)
def initialize_pygame():
    """Ensure Pygame is initialized in headless mode for integration tests."""
    pygame.init()
    yield
    pygame.quit()


def test_end_to_end_initial_building_lifecycle(tmp_path: Path):
    """Verify complete 10-step lifecycle of the approved initial building:

    1. Create initial building.
    2. Validate it.
    3. Serialize to dictionary.
    4. Save to temporary JSON file.
    5. Load from JSON file.
    6. Validate reconstructed building.
    7. Create Pygame surface.
    8. Render reconstructed building.
    9. Confirm rendering completed without errors.
    10. Confirm rendering did not mutate the model.
    """
    # 1. Create initial building
    building = create_initial_building()
    assert building.id == "building_main"
    assert building.width == 40.0
    assert building.height == 30.0

    # 2. Validate
    building.validate()

    # 3. Serialize to dictionary
    data = building_to_dict(building)
    assert data["schema_version"] == SCHEMA_VERSION
    assert len(data["walkable_areas"]) == 5
    assert len(data["doorways"]) == 4
    assert len(data["exits"]) == 3
    assert len(data["walls"]) == 17
    assert len(data["obstacles"]) == 4

    # 4. Save to temporary file
    json_file = tmp_path / "lifecycle_building.json"
    save_building(building, json_file)
    assert json_file.is_file()

    # 5. Load from JSON file
    reconstructed = load_building(json_file)
    assert isinstance(reconstructed, Building)

    # 6. Validate reconstructed building
    reconstructed.validate()
    assert reconstructed.id == building.id
    assert reconstructed.width == pytest.approx(40.0)
    assert reconstructed.height == pytest.approx(30.0)

    # 7. Create Pygame surface & Viewport
    surface = pygame.Surface((1200, 800))
    viewport = Viewport(
        world_width=reconstructed.width,
        world_height=reconstructed.height,
        screen_width=1200,
        screen_height=800,
    )

    # Snapshot state prior to rendering
    area_ids_pre = [a.id for a in reconstructed.walkable_areas]
    wall_count_pre = len(reconstructed.walls)
    door_count_pre = len(reconstructed.doorways)
    exit_count_pre = len(reconstructed.exits)
    obs_count_pre = len(reconstructed.obstacles)

    # 8. Render reconstructed building
    renderer = EnvironmentRenderer()
    renderer.draw(surface, reconstructed, viewport)

    # 9. Confirm rendering completed without errors
    assert surface.get_size() == (1200, 800)
    # Check that pixels were drawn and not blank
    assert surface.get_at((0, 0)) == renderer.COLOR_BG
    sample_pt = viewport.world_to_screen_point(Point(5.0, 15.0))
    assert surface.get_at((int(sample_pt[0]), int(sample_pt[1])))[:3] == renderer.COLOR_CORRIDOR_FLOOR

    # 10. Confirm rendering did not mutate the model
    assert [a.id for a in reconstructed.walkable_areas] == area_ids_pre
    assert len(reconstructed.walls) == wall_count_pre
    assert len(reconstructed.doorways) == door_count_pre
    assert len(reconstructed.exits) == exit_count_pre
    assert len(reconstructed.obstacles) == obs_count_pre


def test_end_to_end_custom_building_lifecycle(tmp_path: Path):
    """Verify complete lifecycle for a distinct custom geometry building."""
    custom_areas = [
        WalkableArea(
            id="hall_alpha",
            bounds=Rectangle(0.0, 0.0, 15.0, 15.0),
            connected_area_ids=["hall_beta"],
        ),
        WalkableArea(
            id="hall_beta",
            bounds=Rectangle(15.0, 0.0, 15.0, 15.0),
            connected_area_ids=["hall_alpha"],
        ),
    ]
    custom_doors = [
        Doorway(
            id="portal_alpha_beta",
            connected_area_ids=("hall_alpha", "hall_beta"),
            segment=LineSegment(Point(15.0, 6.0), Point(15.0, 9.0)),
            width=3.0,
        )
    ]
    custom_exits = [
        Exit(
            id="exit_alpha_west",
            segment=LineSegment(Point(0.0, 6.0), Point(0.0, 9.0)),
            width=3.0,
            boundary="left",
            connected_area_id="hall_alpha",
        )
    ]
    custom_walls = [
        Wall(
            id="w_ext_top",
            start=Point(0.0, 0.0),
            end=Point(30.0, 0.0),
            thickness=0.2,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.POSITIVE,
        ),
        Wall(
            id="w_ext_bottom",
            start=Point(0.0, 15.0),
            end=Point(30.0, 15.0),
            thickness=0.2,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.NEGATIVE,
        ),
        Wall(
            id="w_ext_right",
            start=Point(30.0, 0.0),
            end=Point(30.0, 15.0),
            thickness=0.2,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.NEGATIVE,
        ),
        Wall(
            id="w_ext_left_n",
            start=Point(0.0, 0.0),
            end=Point(0.0, 6.0),
            thickness=0.2,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.POSITIVE,
        ),
        Wall(
            id="w_ext_left_s",
            start=Point(0.0, 9.0),
            end=Point(0.0, 15.0),
            thickness=0.2,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.POSITIVE,
        ),
        Wall(
            id="w_mid_n",
            start=Point(15.0, 0.0),
            end=Point(15.0, 6.0),
            thickness=0.2,
            wall_type=WallType.INTERIOR,
            alignment=WallAlignment.CENTER,
        ),
        Wall(
            id="w_mid_s",
            start=Point(15.0, 9.0),
            end=Point(15.0, 15.0),
            thickness=0.2,
            wall_type=WallType.INTERIOR,
            alignment=WallAlignment.CENTER,
        ),
    ]

    custom_b = Building(
        id="custom_facility",
        width=30.0,
        height=15.0,
        walkable_areas=custom_areas,
        walls=custom_walls,
        doorways=custom_doors,
        exits=custom_exits,
    )

    # Save, reload, render
    custom_json = tmp_path / "custom_facility.json"
    save_building(custom_b, custom_json)
    reloaded_b = load_building(custom_json)
    assert reloaded_b.id == "custom_facility"

    surface = pygame.Surface((900, 450))
    vp = Viewport(
        world_width=reloaded_b.width,
        world_height=reloaded_b.height,
        screen_width=900,
        screen_height=450,
    )
    draw_building(surface, reloaded_b, vp)
    assert surface.get_size() == (900, 450)


def test_complete_architectural_isolation():
    """Verify that importing genevac.environment does not import pygame, torch, or MID in clean process."""
    cmd = [
        sys.executable,
        "-c",
        (
            "import genevac.environment, sys; "
            "assert 'pygame' not in sys.modules, 'pygame was imported by environment!'; "
            "assert 'torch' not in sys.modules, 'torch was imported!'; "
            "assert 'mid' not in sys.modules, 'mid was imported!'; "
            "assert 'genevac.visualization' not in sys.modules, 'visualization was imported by environment!'"
        ),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Architectural isolation violation: {res.stderr}"


# ---------------------------------------------------------------------------
# Phase 5: Group 1 Integration Test
# ---------------------------------------------------------------------------

from dataclasses import FrozenInstanceError
import json
import math
from genevac.pedestrian.models import PedestrianState, PedestrianStatus, Velocity2D
from genevac.pedestrian.registry import PedestrianRegistry
from genevac.simulation.engine import SimulationEngine
from genevac.simulation.movement import check_collision, detect_exit_crossing
from genevac.simulation.observation import (
    PedestrianObservation,
    SimulationObservationSnapshot,
)



def test_json_reloaded_building_simulation_lifecycle(tmp_path: Path):
    """Verify complete end-to-end integration across JSON serialization and simulation:

    1. Create canonical initial building.
    2. Save building to temporary JSON file via save_building().
    3. Reload building via load_building() and validate reconstructed model.
    4. Register a deterministic cohort targeting distinct boundary exits (exit_left, exit_right, exit_bottom).
    5. Initialize SimulationEngine and verify initial t=0 state and step-zero observation snapshot.
    6. Verify automatic deterministic route generation for all cohort pedestrians.
    7. Step simulation until all pedestrians evacuate (with a safe timeout).
    8. Verify every pedestrian reaches their assigned exit within the usable opening.
    9. Verify final statuses are EVACUATED and velocity is zero.
    10. Verify dynamic population counts throughout and upon completion.
    11. Verify kinematic continuity (no teleportation) along trajectory history.
    12. Verify reloaded building geometry remains valid and unmutated.
    """
    # 1. Canonical building
    canonical = create_initial_building()
    canonical.validate()
    initial_area_count = len(canonical.walkable_areas)
    initial_wall_count = len(canonical.walls)
    initial_door_count = len(canonical.doorways)
    initial_exit_count = len(canonical.exits)
    initial_obs_count = len(canonical.obstacles)

    # 2. Save to JSON
    json_path = tmp_path / "canonical_sim_lifecycle.json"
    save_building(canonical, json_path)
    assert json_path.is_file()

    # 3. Reload from JSON and validate
    reloaded = load_building(json_path)
    assert isinstance(reloaded, Building)
    reloaded.validate()
    assert reloaded.id == canonical.id
    assert reloaded.width == pytest.approx(40.0)
    assert reloaded.height == pytest.approx(30.0)

    # 4. Create fresh pedestrian cohort targeting distinct boundary exits
    p1 = PedestrianState(
        id="ped_room1_west",
        position=Point(10.0, 5.0),
        preferred_speed=1.3,
        radius=0.25,
        target_exit_id="exit_left",
        current_area_id="room_1",
    )
    p2 = PedestrianState(
        id="ped_corridor_east",
        position=Point(20.0, 15.0),
        preferred_speed=1.4,
        radius=0.25,
        target_exit_id="exit_right",
        current_area_id="main_corridor",
    )
    p3 = PedestrianState(
        id="ped_room3_south",
        position=Point(10.0, 25.0),
        preferred_speed=1.2,
        radius=0.25,
        target_exit_id="exit_bottom",
        current_area_id="room_3",
    )

    registry = PedestrianRegistry()
    registry.register(p1)
    registry.register(p2)
    registry.register(p3)
    assert registry.total_count == 3
    assert registry.active_count == 3
    assert registry.evacuated_count == 0

    # 5. Initialize SimulationEngine and verify initial state
    engine = SimulationEngine(building=reloaded, registry=registry, dt=0.1, observation_interval=0.4)
    assert engine.building is reloaded
    assert engine.clock.step_count == 0
    assert engine.clock.timestamp == 0.0
    assert engine.is_paused is False

    # Verify step-zero observation snapshot
    snapshot_0 = engine.latest_snapshot
    assert snapshot_0 is not None
    assert snapshot_0.step_count == 0
    assert snapshot_0.timestamp == 0.0
    assert snapshot_0.total_count == 3
    assert snapshot_0.active_count == 3
    assert snapshot_0.evacuated_count == 0
    assert len(snapshot_0.pedestrians) == 3

    # 6. Verify automatic route initialization
    assert len(p1.waypoints) > 0
    assert p1.waypoints[-1] == Point(0.0, 15.0)  # exit_left midpoint
    assert len(p2.waypoints) > 0
    assert p2.waypoints[-1] == Point(40.0, 15.0)  # exit_right midpoint
    assert len(p3.waypoints) > 0
    assert p3.waypoints[-1] == Point(10.0, 30.0)  # exit_bottom midpoint

    # 7. Execute simulation until all pedestrians evacuate or timeout
    max_steps = 250  # 25.0 seconds simulation timeout
    observation_snapshots = [snapshot_0]

    while registry.active_count > 0 and engine.clock.step_count < max_steps:
        snap = engine.step()
        if snap is not None:
            observation_snapshots.append(snap)

    # 8. Assert all evacuated before timeout
    assert engine.clock.step_count < max_steps, f"Simulation timed out with active count {registry.active_count}"
    assert registry.active_count == 0
    assert registry.evacuated_count == 3
    assert registry.blocked_count == 0
    assert registry.total_count == 3

    # 9. Verify every pedestrian reached assigned exit with status EVACUATED and zero velocity
    assert p1.status == PedestrianStatus.EVACUATED
    assert p2.status == PedestrianStatus.EVACUATED
    assert p3.status == PedestrianStatus.EVACUATED
    assert p1.velocity.vx == 0.0 and p1.velocity.vy == 0.0
    assert p2.velocity.vx == 0.0 and p2.velocity.vy == 0.0
    assert p3.velocity.vx == 0.0 and p3.velocity.vy == 0.0

    # Verify exact boundary and usable opening coordinates:
    # p1 at exit_left: x=0.0, y in [14.0, 16.0]
    assert p1.position.x == pytest.approx(0.0, abs=1e-6)
    assert 14.0 <= p1.position.y <= 16.0

    # p2 at exit_right: x=40.0, y in [14.0, 16.0]
    assert p2.position.x == pytest.approx(40.0, abs=1e-6)
    assert 14.0 <= p2.position.y <= 16.0

    # p3 at exit_bottom: y=30.0, x in [9.0, 11.0]
    assert p3.position.y == pytest.approx(30.0, abs=1e-6)
    assert 9.0 <= p3.position.x <= 11.0

    # 10. Verify kinematic continuity (no teleportation) along trajectory history
    for ped in (p1, p2, p3):
        assert len(ped.trajectory_history) >= 2
        for i in range(1, len(ped.trajectory_history)):
            prev_sample = ped.trajectory_history[i - 1]
            curr_sample = ped.trajectory_history[i]
            assert curr_sample.step > prev_sample.step
            assert curr_sample.timestamp > prev_sample.timestamp
            dt_elapsed = curr_sample.timestamp - prev_sample.timestamp
            dist_moved = math.hypot(
                curr_sample.x - prev_sample.x,
                curr_sample.y - prev_sample.y,
            )
            # Max possible travel between recorded observation samples is preferred_speed * dt_elapsed
            max_expected_dist = ped.preferred_speed * dt_elapsed + 1e-4
            assert dist_moved <= max_expected_dist, (
                f"{ped.id} teleported at step {curr_sample.step}: moved {dist_moved} > {max_expected_dist}"
            )

    # 11. Verify reloaded building was not mutated by simulation
    reloaded.validate()
    assert len(reloaded.walkable_areas) == initial_area_count
    assert len(reloaded.walls) == initial_wall_count
    assert len(reloaded.doorways) == initial_door_count
    assert len(reloaded.exits) == initial_exit_count
    assert len(reloaded.obstacles) == initial_obs_count
    assert reloaded.width == pytest.approx(40.0)
    assert reloaded.height == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Phase 5: Group 2 Integration Tests
# ---------------------------------------------------------------------------

def test_simultaneous_three_exit_cohort_evacuation():
    """Verify simultaneous multi-exit evacuation of six pedestrians across 3 distinct boundary exits:

    1. Cohort: 2 assigned to exit_left (West), 2 to exit_right (East), 2 to exit_bottom (South).
    2. Spaced across Room 1, Room 2, Room 3, and Corridor to avoid overlap.
    3. Step simulation until all pedestrians evacuate (timeout bound: 250 steps / 25.0 s).
    4. Verify:
       - Every pedestrian is retained in registry.
       - Every pedestrian reaches only their assigned exit boundary segment and within usable opening.
       - All 6 have status EVACUATED and velocity (0, 0).
       - Final counts: total 6, active 0, evacuated 6, blocked 0, inactive 0.
       - Kinematic continuity (no teleportation) across all trajectory histories.
       - Building geometry remains valid and unchanged.
    """
    building = create_initial_building()
    building.validate()
    initial_area_count = len(building.walkable_areas)
    initial_wall_count = len(building.walls)
    initial_door_count = len(building.doorways)
    initial_exit_count = len(building.exits)
    initial_obs_count = len(building.obstacles)

    # 1. Setup 6 pedestrians assigned to the 3 boundary exits
    p_left_1 = PedestrianState(
        id="p_left_room1",
        position=Point(10.0, 5.0),
        preferred_speed=1.3,
        radius=0.25,
        target_exit_id="exit_left",
        current_area_id="room_1",
    )
    p_left_2 = PedestrianState(
        id="p_left_corr",
        position=Point(8.0, 15.0),
        preferred_speed=1.2,
        radius=0.25,
        target_exit_id="exit_left",
        current_area_id="main_corridor",
    )
    p_right_1 = PedestrianState(
        id="p_right_room2",
        position=Point(30.0, 5.0),
        preferred_speed=1.3,
        radius=0.25,
        target_exit_id="exit_right",
        current_area_id="room_2",
    )
    p_right_2 = PedestrianState(
        id="p_right_corr",
        position=Point(32.0, 15.0),
        preferred_speed=1.4,
        radius=0.25,
        target_exit_id="exit_right",
        current_area_id="main_corridor",
    )
    p_bottom_1 = PedestrianState(
        id="p_bottom_room3_w",
        position=Point(8.0, 24.0),
        preferred_speed=1.3,
        radius=0.25,
        target_exit_id="exit_bottom",
        current_area_id="room_3",
    )
    p_bottom_2 = PedestrianState(
        id="p_bottom_room3_e",
        position=Point(12.0, 22.0),
        preferred_speed=1.4,
        radius=0.25,
        target_exit_id="exit_bottom",
        current_area_id="room_3",
    )

    cohort = [p_left_1, p_left_2, p_right_1, p_right_2, p_bottom_1, p_bottom_2]
    registry = PedestrianRegistry()
    for p in cohort:
        registry.register(p)

    assert registry.total_count == 6
    assert registry.active_count == 6
    assert registry.evacuated_count == 0

    engine = SimulationEngine(
        building=building,
        registry=registry,
        dt=0.1,
        observation_interval=0.4,
        block_unassigned_exits=False,
    )

    # 2. Advance simulation until all pedestrians evacuate
    max_steps = 250
    while registry.active_count > 0 and engine.clock.step_count < max_steps:
        engine.step()

    # 3. Assert bounded evacuation
    assert engine.clock.step_count < max_steps, f"Evacuation timed out at step {engine.clock.step_count}"
    assert registry.active_count == 0
    assert registry.evacuated_count == 6
    assert registry.blocked_count == 0
    assert registry.inactive_count == 0
    assert registry.total_count == 6

    # 4. Verify all retained and reached assigned exits with zero velocity
    for p in cohort:
        assert registry.get(p.id) is p
        assert p.status == PedestrianStatus.EVACUATED
        assert p.velocity == Velocity2D(0.0, 0.0)

    # Exit left: x = 0.0, y in [14.0, 16.0]
    assert p_left_1.position.x == pytest.approx(0.0, abs=1e-6)
    assert 14.0 <= p_left_1.position.y <= 16.0
    assert p_left_2.position.x == pytest.approx(0.0, abs=1e-6)
    assert 14.0 <= p_left_2.position.y <= 16.0

    # Exit right: x = 40.0, y in [14.0, 16.0]
    assert p_right_1.position.x == pytest.approx(40.0, abs=1e-6)
    assert 14.0 <= p_right_1.position.y <= 16.0
    assert p_right_2.position.x == pytest.approx(40.0, abs=1e-6)
    assert 14.0 <= p_right_2.position.y <= 16.0

    # Exit bottom: y = 30.0, x in [9.0, 11.0]
    assert p_bottom_1.position.y == pytest.approx(30.0, abs=1e-6)
    assert 9.0 <= p_bottom_1.position.x <= 11.0
    assert p_bottom_2.position.y == pytest.approx(30.0, abs=1e-6)
    assert 9.0 <= p_bottom_2.position.x <= 11.0

    # 5. Kinematic continuity along trajectory histories
    for p in cohort:
        assert len(p.trajectory_history) >= 2
        for i in range(1, len(p.trajectory_history)):
            prev_s = p.trajectory_history[i - 1]
            curr_s = p.trajectory_history[i]
            assert curr_s.step > prev_s.step
            assert curr_s.timestamp > prev_s.timestamp
            dt_el = curr_s.timestamp - prev_s.timestamp
            dist = math.hypot(curr_s.x - prev_s.x, curr_s.y - prev_s.y)
            max_dist = p.preferred_speed * dt_el + 1e-4
            assert dist <= max_dist, f"{p.id} moved {dist} > {max_dist} between steps {prev_s.step}->{curr_s.step}"

    # 6. Building geometry unchanged
    building.validate()
    assert len(building.walkable_areas) == initial_area_count
    assert len(building.walls) == initial_wall_count
    assert len(building.doorways) == initial_door_count
    assert len(building.exits) == initial_exit_count
    assert len(building.obstacles) == initial_obs_count


def test_mixed_status_population_dynamics():
    """Verify simulation dynamics with a controlled population containing mixed statuses:

    1. Active pedestrian (p_active) who progresses and evacuates cleanly.
    2. Unassigned-exit pedestrian (p_no_exit) with target_exit_id=None: remains in room with zero velocity.
    3. Unassigned-exit crossing pedestrian (p_wrong_exit): attempts crossing unassigned exit and becomes BLOCKED.
    4. Controlled collision pedestrian (p_collision): moves into static obstacle and becomes BLOCKED.
    5. Inactive pedestrian (p_inactive): initially marked INACTIVE, remains stationary.
    6. Verify:
       - Active pedestrian completes evacuation.
       - Blocked pedestrians transition to BLOCKED and remain stationary with zero velocity.
       - Inactive pedestrian remains stationary and INACTIVE.
       - Unassigned exit pedestrian remains stationary with zero velocity.
       - Registry dynamic counts and snapshot counts match ground truth throughout.
       - Neither BLOCKED nor INACTIVE pedestrians are advanced or reactivated.
    """
    building = create_initial_building()
    registry = PedestrianRegistry()

    # 1. Active pedestrian: progresses along corridor to exit_left
    p_active = PedestrianState(
        id="p_act",
        position=Point(5.0, 15.0),
        preferred_speed=1.0,
        radius=0.25,
        target_exit_id="exit_left",
        current_area_id="main_corridor",
    )

    # 2. Unassigned exit pedestrian: target_exit_id is None, no waypoints
    p_no_exit = PedestrianState(
        id="p_no_exit",
        position=Point(15.0, 8.0),
        preferred_speed=1.0,
        radius=0.25,
        target_exit_id=None,
        current_area_id="room_1",
    )

    # 3. Controlled obstacle collision: moves into obstacle_room1 [4.0, 4.0, 6.0, 5.5]
    p_collision = PedestrianState(
        id="p_collision",
        position=Point(3.6, 4.75),
        preferred_speed=1.0,
        radius=0.25,
        target_exit_id="exit_left",
        current_area_id="room_1",
        waypoints=[Point(5.0, 4.75)],
    )

    # 4. Inactive pedestrian: initially INACTIVE
    p_inactive = PedestrianState(
        id="p_inactive",
        position=Point(30.0, 8.0),
        status=PedestrianStatus.INACTIVE,
        target_exit_id="exit_right",
        current_area_id="room_2",
    )

    # 5. Unassigned exit crossing: starts fully inside the corridor with clearance from exterior wall,
    # assigned target is exit_right, but given deterministic waypoint crossing exit_left (boundary X=0).
    p_wrong_exit = PedestrianState(
        id="p_wrong_exit",
        position=Point(0.5, 15.0),
        preferred_speed=7.0,
        radius=0.25,
        target_exit_id="exit_right",
        current_area_id="main_corridor",
        waypoints=[Point(-0.2, 15.0)],
    )

    for p in (p_active, p_no_exit, p_collision, p_inactive, p_wrong_exit):
        registry.register(p)

    # Verify p_wrong_exit starts fully inside the building with clearance from exterior wall
    assert p_wrong_exit.position.x - p_wrong_exit.radius >= 0.25
    assert not check_collision(p_wrong_exit.position, p_wrong_exit.position, p_wrong_exit.radius, building)

    # Verify the movement segment (0.5, 15.0) -> (-0.2, 15.0) intersects exit_left within usable opening
    exit_left = building.get_exit("exit_left")
    crossed, cross_pt = detect_exit_crossing(
        p_wrong_exit.position, Point(-0.2, 15.0), p_wrong_exit.radius, exit_left
    )
    assert crossed is True
    assert cross_pt is not None
    assert cross_pt.x == pytest.approx(0.0)
    assert cross_pt.y == pytest.approx(15.0)
    assert (exit_left.segment.start.y + p_wrong_exit.radius) <= cross_pt.y <= (exit_left.segment.end.y - p_wrong_exit.radius)

    engine = SimulationEngine(
        building=building,
        registry=registry,
        dt=0.1,
        observation_interval=0.4,
        block_unassigned_exits=False,
    )

    # Verify initial population counts
    assert registry.total_count == 5
    assert registry.active_count == 4
    assert registry.evacuated_count == 0
    assert registry.blocked_count == 0
    assert registry.inactive_count == 1

    # Step 1: p_wrong_exit attempts crossing unassigned exit -> BLOCKED
    # Specifically blocked because target_exit_id ("exit_right") != crossed exit ("exit_left")
    snap_1 = engine.step()
    assert p_wrong_exit.status == PedestrianStatus.BLOCKED
    assert p_wrong_exit.velocity == Velocity2D(0.0, 0.0)
    assert p_wrong_exit.position == Point(0.5, 15.0)
    p_wrong_exit_frozen_pos = Point(0.5, 15.0)

    # Step 2: p_collision hits obstacle clearance -> BLOCKED
    snap_2 = engine.step()
    assert p_collision.status == PedestrianStatus.BLOCKED
    assert p_collision.velocity == Velocity2D(0.0, 0.0)
    p_collision_frozen_pos = Point(p_collision.position.x, p_collision.position.y)

    # Advance until p_active evacuates
    max_steps = 100
    snapshots = [s for s in (snap_1, snap_2) if s is not None]

    while p_active.status == PedestrianStatus.ACTIVE and engine.clock.step_count < max_steps:
        snap = engine.step()
        if snap is not None:
            snapshots.append(snap)
            # Verify snapshot counts match registry ground truth at every observation tick
            assert snap.total_count == registry.total_count
            assert snap.active_count == registry.active_count
            assert snap.evacuated_count == registry.evacuated_count
            assert snap.blocked_count == registry.blocked_count
            assert snap.inactive_count == registry.inactive_count

    # Verify p_active reached assigned exit
    assert p_active.status == PedestrianStatus.EVACUATED
    assert p_active.position.x == pytest.approx(0.0, abs=1e-6)
    assert 14.0 <= p_active.position.y <= 16.0
    assert p_active.velocity == Velocity2D(0.0, 0.0)

    # Run 10 additional steps to verify BLOCKED, INACTIVE, and NO_EXIT pedestrians remain completely stationary
    steps_post_evac = 10
    for _ in range(steps_post_evac):
        engine.step()

    # Verify BLOCKED pedestrians remained frozen at exact collision positions
    assert p_collision.status == PedestrianStatus.BLOCKED
    assert p_collision.position == p_collision_frozen_pos
    assert p_collision.velocity == Velocity2D(0.0, 0.0)

    assert p_wrong_exit.status == PedestrianStatus.BLOCKED
    assert p_wrong_exit.position == p_wrong_exit_frozen_pos
    assert p_wrong_exit.velocity == Velocity2D(0.0, 0.0)

    # Verify INACTIVE pedestrian remained frozen
    assert p_inactive.status == PedestrianStatus.INACTIVE
    assert p_inactive.position == Point(30.0, 8.0)
    assert p_inactive.velocity == Velocity2D(0.0, 0.0)

    # Verify pedestrian with no assigned exit remained stationary with zero velocity
    assert p_no_exit.status == PedestrianStatus.ACTIVE
    assert p_no_exit.position == Point(15.0, 8.0)
    assert p_no_exit.velocity == Velocity2D(0.0, 0.0)

    # Final counts verification
    assert registry.total_count == 5
    assert registry.active_count == 1  # p_no_exit
    assert registry.evacuated_count == 1  # p_active
    assert registry.blocked_count == 2  # p_collision, p_wrong_exit
    assert registry.inactive_count == 1  # p_inactive


# ---------------------------------------------------------------------------
# Phase 5: Group 3 Integration Tests
# ---------------------------------------------------------------------------

def test_observation_snapshot_cadence_and_immutability():
    """Verify observation snapshot cadence, initial state, immutable contract, and serialization:

    1. Timing setup: dt=0.1s, observation_interval=0.4s (stride = 4 steps).
    2. Step-0 snapshot: verified at engine initialization (latest_snapshot at step 0, timestamp 0.0).
    3. Cadence progression: engine.step() emits None on intermediate steps (1, 2, 3) and emits
       SimulationObservationSnapshot on exact cadence steps (4, 8, 12, 16, 20).
    4. Step numbers and timestamps: exactly match engine clock (step * dt, rounded drift-free).
    5. Immutability contract:
       - SimulationObservationSnapshot raises FrozenInstanceError on attempted attribute mutation.
       - PedestrianObservation raises FrozenInstanceError on attempted attribute mutation.
       - Snapshot pedestrians tuple is immutable.
       - Live simulation advancement does not alter previously emitted snapshot objects.
    6. to_dict() serialization: accurately preserves all fields, matches snapshot data, and is JSON-serializable.
    7. Status counts: match registry ground truth at each observation tick.
    """
    building = create_initial_building()
    registry = PedestrianRegistry()
    p1 = PedestrianState(
        id="ped_corr",
        position=Point(10.0, 15.0),
        preferred_speed=1.0,
        radius=0.25,
        target_exit_id="exit_left",
        current_area_id="main_corridor",
    )
    p2 = PedestrianState(
        id="ped_inactive",
        position=Point(30.0, 8.0),
        status=PedestrianStatus.INACTIVE,
        target_exit_id="exit_right",
        current_area_id="room_2",
    )
    registry.register(p1)
    registry.register(p2)

    engine = SimulationEngine(
        building=building,
        registry=registry,
        dt=0.1,
        observation_interval=0.4,
        block_unassigned_exits=False,
    )
    stride = engine.clock.observation_stride
    assert stride == 4

    # 1. Initial snapshot behavior at step 0
    snap_0 = engine.latest_snapshot
    assert snap_0 is not None
    assert isinstance(snap_0, SimulationObservationSnapshot)
    assert snap_0.step_count == 0
    assert snap_0.simulation_step == 0
    assert snap_0.timestamp == 0.0
    assert snap_0.dt == 0.1
    assert snap_0.observation_interval == 0.4
    assert snap_0.total_count == 2
    assert snap_0.active_count == 1
    assert snap_0.inactive_count == 1
    assert snap_0.evacuated_count == 0
    assert snap_0.blocked_count == 0
    assert len(snap_0.pedestrians) == 2

    # 2. Immutability of snapshot and observation instances
    with pytest.raises(FrozenInstanceError):
        snap_0.step_count = 99  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        snap_0.total_count = 99  # type: ignore[misc]

    obs_0_p1 = snap_0.pedestrians[0]
    with pytest.raises(FrozenInstanceError):
        obs_0_p1.x = 999.0  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        obs_0_p1.status = "EVACUATED"  # type: ignore[misc]

    assert isinstance(snap_0.pedestrians, tuple)
    assert isinstance(obs_0_p1.trajectory_history, tuple)

    # 3. to_dict() serialization validation
    d0 = snap_0.to_dict()
    assert d0["snapshot_id"] == snap_0.snapshot_id
    assert d0["step_count"] == 0
    assert d0["simulation_step"] == 0
    assert d0["timestamp"] == 0.0
    assert d0["dt"] == 0.1
    assert d0["observation_interval"] == 0.4
    assert d0["total_count"] == 2
    assert d0["active_count"] == 1
    assert d0["inactive_count"] == 1
    assert len(d0["pedestrians"]) == 2
    json_str = json.dumps(d0)
    assert isinstance(json_str, str) and len(json_str) > 0

    # 4. Cadence verification over 20 steps
    # Derive expected cadence directly from engine.clock
    total_steps = 20
    emitted_snapshots: list[SimulationObservationSnapshot] = []

    for expected_step in range(1, total_steps + 1):
        step_snap = engine.step()
        assert engine.clock.step_count == expected_step
        assert engine.clock.timestamp == pytest.approx(round(expected_step * 0.1, 9))

        if expected_step % stride == 0:
            assert step_snap is not None, f"Expected snapshot at step {expected_step}"
            assert step_snap.step_count == expected_step
            assert step_snap.timestamp == pytest.approx(engine.clock.timestamp)
            assert step_snap.snapshot_id == f"snap_{expected_step:06d}"
            # Verify status counts agree with registry ground truth
            assert step_snap.total_count == registry.total_count
            assert step_snap.active_count == registry.active_count
            assert step_snap.evacuated_count == registry.evacuated_count
            assert step_snap.blocked_count == registry.blocked_count
            assert step_snap.inactive_count == registry.inactive_count
            emitted_snapshots.append(step_snap)
        else:
            assert step_snap is None, f"Expected None on non-cadence step {expected_step}"

    # Total emitted via step() should be total_steps // stride
    expected_emitted_count = total_steps // stride
    assert len(emitted_snapshots) == expected_emitted_count
    assert [s.step_count for s in emitted_snapshots] == [4, 8, 12, 16, 20]

    # Verify snap_0 remained unmutated despite 20 steps of live simulation
    assert snap_0.step_count == 0
    assert snap_0.timestamp == 0.0
    assert snap_0.pedestrians[0].x == pytest.approx(10.0)


def test_trajectory_history_over_complete_evacuation():
    """Verify trajectory history properties along complete evacuation:

    1. History samples contain timestamped fields: step, timestamp, x, y.
    2. Step numbers and timestamps are strictly monotonic.
    3. Consecutive positions obey distance bound: dist <= preferred_speed * dt_elapsed + epsilon.
    4. The final history sample matches the actual final evacuation crossing position.
    5. Final pedestrian status is EVACUATED and velocity is zero.
    6. History capacity FIFO behavior matches documented max_history_len contract.
    """
    building = create_initial_building()
    registry = PedestrianRegistry()

    # Pedestrian with default history capacity (max_history_len=100)
    # Starts at x=4.0, y=15.0, preferred_speed=1.0. Distance=4.0m to exit_left.
    # At dt=0.1, preferred_speed=1.0: 40 steps = 4.0s to evacuate (exactly on stride tick 40).
    p_full = PedestrianState(
        id="p_full_history",
        position=Point(4.0, 15.0),
        preferred_speed=1.0,
        radius=0.25,
        target_exit_id="exit_left",
        current_area_id="main_corridor",
        max_history_len=100,
    )

    # Pedestrian with small bounded history capacity (max_history_len=5)
    p_bounded = PedestrianState(
        id="p_bounded_history",
        position=Point(4.0, 15.0),
        preferred_speed=1.0,
        radius=0.25,
        target_exit_id="exit_left",
        current_area_id="main_corridor",
        max_history_len=5,
    )

    reg_full = PedestrianRegistry()
    reg_full.register(p_full)
    engine_full = SimulationEngine(
        building=create_initial_building(),
        registry=reg_full,
        dt=0.1,
        observation_interval=0.4,
        block_unassigned_exits=False,
    )

    reg_bounded = PedestrianRegistry()
    reg_bounded.register(p_bounded)
    engine_bounded = SimulationEngine(
        building=create_initial_building(),
        registry=reg_bounded,
        dt=0.1,
        observation_interval=0.4,
        block_unassigned_exits=False,
    )

    # Initially step 0 sample is recorded
    assert len(p_full.trajectory_history) == 1
    assert p_full.trajectory_history[0].step == 0
    assert p_full.trajectory_history[0].timestamp == 0.0
    assert p_full.trajectory_history[0].x == pytest.approx(4.0)
    assert p_full.trajectory_history[0].y == pytest.approx(15.0)

    # Run 40 steps to complete evacuation
    engine_full.run_steps(40)
    engine_bounded.run_steps(40)

    # Verify both evacuated
    assert p_full.status == PedestrianStatus.EVACUATED
    assert p_full.velocity == Velocity2D(0.0, 0.0)
    assert p_full.position.x == pytest.approx(0.0, abs=1e-6)
    assert p_full.position.y == pytest.approx(15.0, abs=1e-6)

    assert p_bounded.status == PedestrianStatus.EVACUATED
    assert p_bounded.velocity == Velocity2D(0.0, 0.0)
    assert p_bounded.position.x == pytest.approx(0.0, abs=1e-6)
    assert p_bounded.position.y == pytest.approx(15.0, abs=1e-6)

    # 1. Full history verification (11 samples: steps 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40)
    assert len(p_full.trajectory_history) == 11
    history_steps = [s.step for s in p_full.trajectory_history]
    assert history_steps == [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]

    for i in range(len(p_full.trajectory_history)):
        sample = p_full.trajectory_history[i]
        # Verify required fields
        assert hasattr(sample, "step")
        assert hasattr(sample, "timestamp")
        assert hasattr(sample, "x")
        assert hasattr(sample, "y")
        assert isinstance(sample.step, int)
        assert isinstance(sample.timestamp, float)
        assert isinstance(sample.x, float)
        assert isinstance(sample.y, float)

        if i > 0:
            prev = p_full.trajectory_history[i - 1]
            # Strictly increasing step and timestamp
            assert sample.step > prev.step
            assert sample.timestamp > prev.timestamp

            dt_el = sample.timestamp - prev.timestamp
            dist = math.hypot(sample.x - prev.x, sample.y - prev.y)
            max_dist = p_full.preferred_speed * dt_el + 1e-4
            assert dist <= max_dist, f"Distance {dist} exceeded maximum {max_dist} between steps {prev.step}->{sample.step}"

    # Verify final sample corresponds to actual final evacuation crossing position
    final_sample = p_full.trajectory_history[-1]
    assert final_sample.step == 40
    assert final_sample.timestamp == pytest.approx(4.0)
    assert final_sample.x == pytest.approx(0.0, abs=1e-6)
    assert final_sample.y == pytest.approx(15.0, abs=1e-6)

    # 2. Bounded capacity verification (FIFO retention: only 5 most recent samples)
    assert len(p_bounded.trajectory_history) == 5
    bounded_steps = [s.step for s in p_bounded.trajectory_history]
    assert bounded_steps == [24, 28, 32, 36, 40]
    assert p_bounded.trajectory_history[-1].x == pytest.approx(0.0, abs=1e-6)
    assert p_bounded.trajectory_history[-1].y == pytest.approx(15.0, abs=1e-6)


def test_evacuated_pedestrians_retained_in_later_snapshots():
    """Verify that evacuated pedestrians remain permanently in observation snapshots:

    1. Cohort: p_early (distance 2.0m, evacuates at step 20) and p_late (distance 6.0m, evacuates at step 60).
    2. At step 20: p_early reaches EVACUATED while p_late remains ACTIVE.
    3. Intermediate snapshots (steps 24, 28, 32, 36, 40, 44, 48, 52, 56):
       - p_early is retained in snapshot.pedestrians.
       - Status remains 'EVACUATED', position is frozen at crossing point (0.0, 15.0), velocity is 0.
       - Snapshot counts correctly report active_count=1, evacuated_count=1, total_count=2.
    4. At step 60: both pedestrians are EVACUATED in snapshot and registry.
    5. Registry ground truth retains all pedestrians throughout.
    """
    building = create_initial_building()
    registry = PedestrianRegistry()

    p_early = PedestrianState(
        id="ped_early",
        position=Point(2.0, 15.0),
        preferred_speed=1.0,
        radius=0.25,
        target_exit_id="exit_left",
        current_area_id="main_corridor",
    )
    p_late = PedestrianState(
        id="ped_late",
        position=Point(6.0, 15.0),
        preferred_speed=1.0,
        radius=0.25,
        target_exit_id="exit_left",
        current_area_id="main_corridor",
    )

    registry.register(p_early)
    registry.register(p_late)

    engine = SimulationEngine(
        building=building,
        registry=registry,
        dt=0.1,
        observation_interval=0.4,
        block_unassigned_exits=False,
    )

    # 1. Step to step 20 where p_early evacuates
    snaps_phase1 = engine.run_steps(20)
    assert p_early.status == PedestrianStatus.EVACUATED
    assert p_early.position.x == pytest.approx(0.0, abs=1e-6)
    assert p_early.position.y == pytest.approx(15.0, abs=1e-6)
    assert p_early.velocity == Velocity2D(0.0, 0.0)

    assert p_late.status == PedestrianStatus.ACTIVE
    assert p_late.position.x == pytest.approx(4.0, abs=1e-4)

    assert registry.active_count == 1
    assert registry.evacuated_count == 1
    assert registry.total_count == 2

    # Verify step-20 snapshot
    snap_20 = snaps_phase1[-1]
    assert snap_20.step_count == 20
    assert snap_20.active_count == 1
    assert snap_20.evacuated_count == 1
    assert snap_20.total_count == 2

    # 2. Step from step 21 to 59: collect intermediate snapshots where p_early is evacuated and p_late is active
    snaps_phase2: list[SimulationObservationSnapshot] = []
    for _ in range(39):  # steps 21 to 59
        s = engine.step()
        if s is not None:
            snaps_phase2.append(s)

    # Intermediate observation ticks: 24, 28, 32, 36, 40, 44, 48, 52, 56 (9 snapshots)
    assert len(snaps_phase2) == 9
    for snap in snaps_phase2:
        assert snap.total_count == 2
        assert snap.active_count == 1
        assert snap.evacuated_count == 1
        assert snap.blocked_count == 0
        assert snap.inactive_count == 0

        # Find early pedestrian in snapshot
        early_obs = next((p for p in snap.pedestrians if p.pedestrian_id == "ped_early"), None)
        assert early_obs is not None, f"ped_early missing from snapshot at step {snap.step_count}"
        assert early_obs.status == PedestrianStatus.EVACUATED.value
        assert early_obs.x == pytest.approx(0.0, abs=1e-6)
        assert early_obs.y == pytest.approx(15.0, abs=1e-6)
        assert early_obs.vx == 0.0
        assert early_obs.vy == 0.0
        assert early_obs.speed == 0.0

        # Find late pedestrian in snapshot
        late_obs = next((p for p in snap.pedestrians if p.pedestrian_id == "ped_late"), None)
        assert late_obs is not None, f"ped_late missing from snapshot at step {snap.step_count}"
        assert late_obs.status == PedestrianStatus.ACTIVE.value
        assert late_obs.speed > 0.0
        assert late_obs.x > 0.0

    # 3. Advance step 60: p_late evacuates
    snap_60 = engine.step()
    assert snap_60 is not None
    assert snap_60.step_count == 60
    assert snap_60.total_count == 2
    assert snap_60.active_count == 0
    assert snap_60.evacuated_count == 2

    # Verify both pedestrians have EVACUATED status in step-60 snapshot
    obs_60_early = next((p for p in snap_60.pedestrians if p.pedestrian_id == "ped_early"), None)
    assert obs_60_early is not None
    assert obs_60_early.status == PedestrianStatus.EVACUATED.value

    obs_60_late = next((p for p in snap_60.pedestrians if p.pedestrian_id == "ped_late"), None)
    assert obs_60_late is not None
    assert obs_60_late.status == PedestrianStatus.EVACUATED.value

    assert p_late.status == PedestrianStatus.EVACUATED
    assert p_late.position.x == pytest.approx(0.0, abs=1e-6)
    assert p_late.position.y == pytest.approx(15.0, abs=1e-6)
    assert p_late.velocity == Velocity2D(0.0, 0.0)

    # 4. Final registry verification
    assert registry.total_count == 2
    assert registry.active_count == 0
    assert registry.evacuated_count == 2
    assert registry.blocked_count == 0
    assert registry.inactive_count == 0
    assert "ped_early" in registry
    assert "ped_late" in registry
    assert registry.get("ped_early") is p_early
    assert registry.get("ped_late") is p_late


# ---------------------------------------------------------------------------
# Phase 5: Group 4 Integration Tests
# ---------------------------------------------------------------------------

from genevac.visualization.app import SimulationApp


def test_simulation_app_headless_full_evacuation():
    """Verify complete end-to-end evacuation driven through the interactive SimulationApp in headless mode:

    1. Initialize SimulationApp with a deterministic population factory in headless mode.
    2. Advance application through controlled frame deltas, decoupling render frames from simulation steps.
    3. Call app.render(surface) on each frame to verify read-only rendering without state mutation.
    4. Verify all pedestrians evacuate cleanly within a bounded frame count.
    5. Verify simulation time advances according to engine dt (0.1s), not wall-clock frame count.
    6. Verify app accumulator, clamping, and catch-up ceiling behavior.
    7. Verify HUD metadata matches authoritative engine state (time, step, population counts).
    8. Verify reset restores initial state and cleans up resources without errors.
    """
    def make_cohort() -> list[PedestrianState]:
        return [
            PedestrianState(
                id="app_ped_west",
                position=Point(4.0, 15.0),
                preferred_speed=1.0,
                radius=0.25,
                target_exit_id="exit_left",
                current_area_id="main_corridor",
            ),
            PedestrianState(
                id="app_ped_east",
                position=Point(36.0, 15.0),
                preferred_speed=1.0,
                radius=0.25,
                target_exit_id="exit_right",
                current_area_id="main_corridor",
            ),
            PedestrianState(
                id="app_ped_south",
                position=Point(10.0, 27.0),
                preferred_speed=1.0,
                radius=0.25,
                target_exit_id="exit_bottom",
                current_area_id="room_3",
            ),
        ]

    app = SimulationApp(
        population_factory=make_cohort,
        dt=0.1,
        observation_interval=0.4,
        width=1200,
        height=800,
    )
    surface = pygame.Surface((1200, 800))

    # Initial state checks
    assert app.engine.clock.step_count == 0
    assert app.engine.clock.timestamp == 0.0
    assert app.sim_accumulator == 0.0
    assert app.registry.total_count == 3
    assert app.registry.active_count == 3
    assert app.registry.evacuated_count == 0

    # Verify rendering step 0 does not mutate initial simulation state
    app.render(surface)
    assert app.engine.clock.step_count == 0
    assert app.engine.clock.timestamp == 0.0
    assert app.registry.active_count == 3

    # Drive simulation with controlled frame deltas (0.05s per frame, requiring 2 frames per 0.1s sim step)
    # 40 sim steps needed for complete evacuation -> exactly 80 frames at 0.05s
    max_frames = 120
    frame_count = 0
    total_steps_executed = 0

    while app.registry.active_count > 0 and frame_count < max_frames:
        steps_this_frame = app.update(0.05)
        total_steps_executed += steps_this_frame
        app.render(surface)
        frame_count += 1

    # Verify bounded evacuation: 80 frames executed, all evacuated before frame ceiling
    assert frame_count < max_frames, f"App timed out after {frame_count} frames with active={app.registry.active_count}"
    assert frame_count == 80
    assert total_steps_executed == 40
    assert app.engine.clock.step_count == 40
    assert app.engine.clock.timestamp == pytest.approx(4.0)
    assert app.sim_accumulator == pytest.approx(0.0)

    # Verify all pedestrians evacuated
    assert app.registry.active_count == 0
    assert app.registry.evacuated_count == 3
    assert app.registry.blocked_count == 0
    assert app.registry.total_count == 3

    p_west = app.registry["app_ped_west"]
    p_east = app.registry["app_ped_east"]
    p_south = app.registry["app_ped_south"]

    assert p_west.status == PedestrianStatus.EVACUATED
    assert p_west.position.x == pytest.approx(0.0, abs=1e-6)
    assert p_west.velocity == Velocity2D(0.0, 0.0)

    assert p_east.status == PedestrianStatus.EVACUATED
    assert p_east.position.x == pytest.approx(40.0, abs=1e-6)
    assert p_east.velocity == Velocity2D(0.0, 0.0)

    assert p_south.status == PedestrianStatus.EVACUATED
    assert p_south.position.y == pytest.approx(30.0, abs=1e-6)
    assert p_south.velocity == Velocity2D(0.0, 0.0)

    # Verify rendering post-evacuation does not mutate finished state
    app.render(surface)
    assert app.engine.clock.step_count == 40
    assert app.registry.evacuated_count == 3

    # Verify accumulator clamping and catch-up ceiling on the app
    # Large dt_wall (1.0s) is clamped to 0.25s (executing 2 steps of 0.1s dt, remainder 0.05s)
    steps_large = app.update(1.0)
    assert steps_large == 2
    assert app.sim_accumulator == pytest.approx(0.05)
    assert app.engine.clock.step_count == 42

    # Verify reset clears simulation and resets accumulator cleanly
    app.reset()
    assert app.engine.clock.step_count == 0
    assert app.engine.clock.timestamp == 0.0
    assert app.sim_accumulator == 0.0
    assert app.registry.active_count == 3
    assert app.registry.evacuated_count == 0


def test_deterministic_resimulation_after_reset():
    """Verify that resetting the SimulationApp produces 100% bitwise deterministic re-simulation:

    1. Configure SimulationApp with deterministic initial population.
    2. Run Run 1 over a fixed irregular sequence of frame deltas, calling update and render on each frame.
    3. Capture complete state at the end of Run 1: clock step/timestamp, accumulator, population counts,
       pedestrian positions, velocities, statuses, and full trajectory histories.
    4. Call app.reset().
    5. Verify reset completely restores initial state, creates fresh pedestrian objects (no reuse),
       and clears the accumulator and clock.
    6. Run Run 2 over the exact same sequence of frame deltas, calling update and render on each frame.
    7. Compare Run 2 against Run 1:
       - Exact match of final step_count and timestamp.
       - Exact match of final accumulator remainder.
       - Exact match of all population counts (total, active, evacuated, blocked, inactive).
       - Exact match of pedestrian positions, velocities, and statuses.
       - Exact sample-for-sample match of all trajectory histories across all pedestrians.
    """
    def make_cohort() -> list[PedestrianState]:
        return [
            PedestrianState(
                id="resim_p1",
                position=Point(6.0, 15.0),
                preferred_speed=1.2,
                radius=0.25,
                target_exit_id="exit_left",
                current_area_id="main_corridor",
            ),
            PedestrianState(
                id="resim_p2",
                position=Point(34.0, 15.0),
                preferred_speed=1.3,
                radius=0.25,
                target_exit_id="exit_right",
                current_area_id="main_corridor",
            ),
            PedestrianState(
                id="resim_p3",
                position=Point(10.0, 26.0),
                preferred_speed=1.1,
                radius=0.25,
                target_exit_id="exit_bottom",
                current_area_id="room_3",
            ),
        ]

    app = SimulationApp(
        population_factory=make_cohort,
        dt=0.1,
        observation_interval=0.4,
        width=1200,
        height=800,
    )
    surface = pygame.Surface((1200, 800))

    # Fixed sequence of irregular frame deltas exercising accumulator accumulation and carryover
    frame_deltas = [0.033, 0.05, 0.04, 0.08, 0.12, 0.05, 0.02, 0.06, 0.15, 0.05] * 3

    # --- RUN 1 ---
    for dt_wall in frame_deltas:
        app.update(dt_wall)
        app.render(surface)

    run1_step = app.engine.clock.step_count
    run1_time = app.engine.clock.timestamp
    run1_acc = app.sim_accumulator
    run1_counts = {
        "total": app.registry.total_count,
        "active": app.registry.active_count,
        "evacuated": app.registry.evacuated_count,
        "blocked": app.registry.blocked_count,
        "inactive": app.registry.inactive_count,
    }

    run1_peds_state: dict[str, dict] = {}
    for p in app.registry:
        run1_peds_state[p.id] = {
            "x": p.position.x,
            "y": p.position.y,
            "vx": p.velocity.vx,
            "vy": p.velocity.vy,
            "status": p.status,
            "history": [(pt.step, pt.timestamp, pt.x, pt.y) for pt in p.trajectory_history],
        }

    old_ped_instances = list(app.registry)

    # --- RESET ---
    app.reset()

    # Verify reset invariants
    assert app.engine.clock.step_count == 0
    assert app.engine.clock.timestamp == 0.0
    assert app.sim_accumulator == 0.0
    assert app.registry.total_count == 3
    assert app.registry.active_count == 3
    assert app.registry.evacuated_count == 0
    assert app.registry.blocked_count == 0
    assert app.registry.inactive_count == 0

    new_ped_instances = list(app.registry)
    assert len(new_ped_instances) == len(old_ped_instances)
    for old_p, new_p in zip(old_ped_instances, new_ped_instances):
        assert old_p is not new_p, f"Pedestrian {old_p.id} was reused rather than re-instantiated!"
        assert old_p.id == new_p.id

    # --- RUN 2 ---
    for dt_wall in frame_deltas:
        app.update(dt_wall)
        app.render(surface)

    # --- COMPARISON ---
    # 1. Clock and timing
    assert app.engine.clock.step_count == run1_step
    assert app.engine.clock.timestamp == pytest.approx(run1_time)
    assert abs(app.sim_accumulator - run1_acc) < 1e-9

    # 2. Population counts
    assert app.registry.total_count == run1_counts["total"]
    assert app.registry.active_count == run1_counts["active"]
    assert app.registry.evacuated_count == run1_counts["evacuated"]
    assert app.registry.blocked_count == run1_counts["blocked"]
    assert app.registry.inactive_count == run1_counts["inactive"]

    # 3. Pedestrian positions, velocities, statuses, and histories
    for p in app.registry:
        expected = run1_peds_state[p.id]
        assert p.position.x == pytest.approx(expected["x"], abs=1e-6)
        assert p.position.y == pytest.approx(expected["y"], abs=1e-6)
        assert p.velocity.vx == pytest.approx(expected["vx"], abs=1e-6)
        assert p.velocity.vy == pytest.approx(expected["vy"], abs=1e-6)
        assert p.status == expected["status"]

        run2_history = [(pt.step, pt.timestamp, pt.x, pt.y) for pt in p.trajectory_history]
        assert len(run2_history) == len(expected["history"])
        for s1, s2 in zip(expected["history"], run2_history):
            assert s1[0] == s2[0]  # step
            assert s1[1] == pytest.approx(s2[1])  # timestamp
            assert s1[2] == pytest.approx(s2[2], abs=1e-6)  # x
            assert s1[3] == pytest.approx(s2[3], abs=1e-6)  # y




