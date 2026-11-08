"""Tests for GenEvac-Sentinel Phase 4 Static Pygame Visualization."""

import copy
import os
from pathlib import Path
import pytest

# Ensure headless operation for CI and test execution
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

from genevac.environment.builder import create_initial_building
from genevac.environment.geometry import Point, Rectangle
from genevac.environment.models import Building, WalkableArea
from genevac.environment.serialization import load_building, save_building
from genevac.visualization.coordinates import Viewport
from genevac.visualization.renderer import EnvironmentRenderer, draw_building


# ---------------------------------------------------------------------------
# Coordinate Conversion & Viewport Tests
# ---------------------------------------------------------------------------

def test_world_to_screen_origin_mapping():
    """Verify that world origin (0.0, 0.0) correctly maps to the building's screen offsets."""
    vp = Viewport(
        world_width=40.0,
        world_height=30.0,
        screen_width=1200,
        screen_height=800,
        margin=40.0,
    )
    origin_screen = vp.world_to_screen_point(Point(0.0, 0.0))
    assert origin_screen == (vp.offset_x, vp.offset_y)


def test_building_dimensions_preserve_aspect_ratio():
    """Verify that scale factor is uniform and strictly preserves world aspect ratio."""
    vp = Viewport(
        world_width=40.0,
        world_height=30.0,
        screen_width=1200,
        screen_height=800,
    )
    world_ratio = 40.0 / 30.0
    screen_ratio = vp.building_px_w / vp.building_px_h
    assert screen_ratio == pytest.approx(world_ratio)


@pytest.mark.parametrize(
    "screen_w,screen_h",
    [
        (800, 600),
        (1024, 768),
        (1920, 1080),
        (600, 1200),  # Tall window (letterboxing)
        (1600, 400),  # Ultra-wide window (pillarboxing)
    ],
)
def test_window_resize_preserves_aspect_ratio(screen_w, screen_h):
    """Verify aspect ratio remains strictly 4:3 across arbitrary window resizes."""
    vp = Viewport(
        world_width=40.0,
        world_height=30.0,
        screen_width=1200,
        screen_height=800,
    )
    vp.update_screen_size(screen_w, screen_h)

    world_ratio = 40.0 / 30.0
    rendered_ratio = vp.building_px_w / vp.building_px_h
    assert rendered_ratio == pytest.approx(world_ratio)

    # Entire building must fit within screen bounds
    assert vp.offset_x >= 0
    assert vp.offset_y >= 0
    assert vp.offset_x + vp.building_px_w <= screen_w
    assert vp.offset_y + vp.building_px_h <= screen_h


def test_building_is_centered_in_viewport():
    """Verify the building is horizontally and vertically centered in the drawable area."""
    screen_w = 1200
    screen_h = 800
    margin = 40.0
    top_off = 60.0
    bot_off = 50.0

    vp = Viewport(
        world_width=40.0,
        world_height=30.0,
        screen_width=screen_w,
        screen_height=screen_h,
        margin=margin,
        top_offset=top_off,
        bottom_offset=bot_off,
    )

    avail_w = screen_w - 2.0 * margin
    avail_h = screen_h - margin - top_off - bot_off

    # Margins on both sides of the rendered building should be equal
    left_slack = vp.offset_x - margin
    right_slack = (margin + avail_w) - (vp.offset_x + vp.building_px_w)
    assert left_slack == pytest.approx(right_slack)

    top_slack = vp.offset_y - top_off
    bottom_slack = (top_off + avail_h) - (vp.offset_y + vp.building_px_h)
    assert top_slack == pytest.approx(bottom_slack)


def test_coordinate_conversion_corners_and_round_trip():
    """Verify building corner coordinates and screen-to-world round-trip conversions."""
    vp = Viewport(
        world_width=40.0,
        world_height=30.0,
        screen_width=1200,
        screen_height=800,
    )

    top_left = vp.world_to_screen_point(Point(0.0, 0.0))
    top_right = vp.world_to_screen_point(Point(40.0, 0.0))
    bottom_left = vp.world_to_screen_point(Point(0.0, 30.0))
    bottom_right = vp.world_to_screen_point(Point(40.0, 30.0))

    assert top_left == (vp.offset_x, vp.offset_y)
    assert top_right == (vp.offset_x + vp.building_px_w, vp.offset_y)
    assert bottom_left == (vp.offset_x, vp.offset_y + vp.building_px_h)
    assert bottom_right == (vp.offset_x + vp.building_px_w, vp.offset_y + vp.building_px_h)

    # Test round-trip conversion for internal test points
    test_points = [
        Point(0.0, 0.0),
        Point(40.0, 30.0),
        Point(20.0, 15.0),
        Point(9.0, 13.5),
        Point(24.0, 4.0),
    ]
    for pt in test_points:
        screen_pt = vp.world_to_screen_point(pt)
        recovered_pt = vp.screen_to_world_point(screen_pt)
        assert recovered_pt.x == pytest.approx(pt.x)
        assert recovered_pt.y == pytest.approx(pt.y)


def test_invalid_viewport_dimensions_rejected():
    """Verify rejection of zero or negative world dimensions."""
    with pytest.raises(ValueError, match="strictly positive"):
        Viewport(world_width=-10.0, world_height=30.0, screen_width=800, screen_height=600)


# ---------------------------------------------------------------------------
# Pygame Headless Rendering Tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def setup_headless_pygame():
    """Initialize Pygame once for testing."""
    pygame.init()
    yield
    pygame.quit()


def test_renderer_accepts_building_model():
    """Verify EnvironmentRenderer renders the default initial Building without error."""
    building = create_initial_building()
    surface = pygame.Surface((1200, 800))
    vp = Viewport(
        world_width=building.width,
        world_height=building.height,
        screen_width=1200,
        screen_height=800,
    )

    renderer = EnvironmentRenderer()
    renderer.draw(surface, building, vp)

    # Verify pixels were drawn (surface is not pure black)
    assert surface.get_at((0, 0)) == renderer.COLOR_BG
    # Check that a point in the corridor floor (away from the centered text label) has corridor floor color
    corridor_sample = vp.world_to_screen_point(Point(5.0, 15.0))
    px_color = surface.get_at((int(corridor_sample[0]), int(corridor_sample[1])))[:3]
    assert px_color == renderer.COLOR_CORRIDOR_FLOOR


def test_renderer_does_not_modify_building_object():
    """Verify rendering is purely a read-only operation and does not mutate Building state."""
    building = create_initial_building()
    surface = pygame.Surface((1200, 800))
    vp = Viewport(
        world_width=building.width,
        world_height=building.height,
        screen_width=1200,
        screen_height=800,
    )

    # Snapshot before render
    area_ids_before = [a.id for a in building.walkable_areas]
    wall_count_before = len(building.walls)
    door_count_before = len(building.doorways)
    exit_count_before = len(building.exits)
    obs_count_before = len(building.obstacles)

    draw_building(surface, building, vp)

    # Assert completely unchanged after render
    assert [a.id for a in building.walkable_areas] == area_ids_before
    assert len(building.walls) == wall_count_before
    assert len(building.doorways) == door_count_before
    assert len(building.exits) == exit_count_before
    assert len(building.obstacles) == obs_count_before


def test_rendering_works_with_json_loaded_building(tmp_path: Path):
    """Verify that a Building loaded from JSON via load_building() renders cleanly."""
    json_path = tmp_path / "test_building_render.json"
    original = create_initial_building()
    save_building(original, json_path)

    loaded = load_building(json_path)

    surface = pygame.Surface((1024, 768))
    vp = Viewport(
        world_width=loaded.width,
        world_height=loaded.height,
        screen_width=1024,
        screen_height=768,
    )

    renderer = EnvironmentRenderer()
    renderer.draw(surface, loaded, vp)
    assert surface.get_size() == (1024, 768)


def test_rendering_custom_building_dimensions():
    """Verify renderer dynamically handles custom building dimensions (20m x 15m)."""
    custom_b = Building(
        id="custom_small",
        width=20.0,
        height=15.0,
        walkable_areas=[
            WalkableArea(
                id="main_hall",
                bounds=Rectangle(0.0, 0.0, 20.0, 15.0),
                connected_area_ids=[],
            )
        ],
    )
    surface = pygame.Surface((800, 600))
    vp = Viewport(
        world_width=custom_b.width,
        world_height=custom_b.height,
        screen_width=800,
        screen_height=600,
    )

    draw_building(surface, custom_b, vp)
    assert surface.get_size() == (800, 600)


# ---------------------------------------------------------------------------
# Phase 4: Simulation Visualization Integration Tests
# ---------------------------------------------------------------------------

from genevac.pedestrian.models import PedestrianState, PedestrianStatus
from genevac.pedestrian.registry import PedestrianRegistry
from genevac.simulation.engine import SimulationEngine
from genevac.visualization.app import SimulationApp, create_default_pedestrian_cohort


def test_renderer_with_simulation_engine_draws_pedestrians():
    """Verify EnvironmentRenderer renders active simulation pedestrians and HUD."""
    building = create_initial_building()
    ped = PedestrianState(
        id="ped_test_1",
        position=Point(10.0, 15.0),
        preferred_speed=1.2,
        radius=0.4,
    )
    registry = PedestrianRegistry()
    registry.register(ped)
    engine = SimulationEngine(building=building, registry=registry, dt=0.1, block_unassigned_exits=False)

    surface = pygame.Surface((1200, 800))
    vp = Viewport(
        world_width=building.width,
        world_height=building.height,
        screen_width=1200,
        screen_height=800,
    )
    renderer = EnvironmentRenderer()
    renderer.draw(surface, building, vp, engine=engine)

    # Verify pedestrian position pixel matches pedestrian active color or core color
    screen_pos = vp.world_to_screen_point(ped.position)
    ped_px = surface.get_at((int(screen_pos[0]), int(screen_pos[1])))[:3]
    # The pixel at the center should be either the core (255, 255, 255) or active color
    assert ped_px in (renderer.COLOR_PED_ACTIVE, renderer.COLOR_PED_ACTIVE_CORE)


def test_renderer_draws_all_pedestrian_statuses():
    """Verify distinct visual representation of ACTIVE, EVACUATED, BLOCKED, and INACTIVE states."""
    building = create_initial_building()
    p_active = PedestrianState(id="p_act", position=Point(5.0, 5.0), status=PedestrianStatus.ACTIVE)
    p_evac = PedestrianState(id="p_eva", position=Point(10.0, 5.0), status=PedestrianStatus.EVACUATED)
    p_blocked = PedestrianState(id="p_blk", position=Point(15.0, 5.0), status=PedestrianStatus.BLOCKED)
    p_inactive = PedestrianState(id="p_ina", position=Point(20.0, 5.0), status=PedestrianStatus.INACTIVE)

    registry = PedestrianRegistry()
    for p in (p_active, p_evac, p_blocked, p_inactive):
        registry.register(p)
    surface = pygame.Surface((1200, 800))
    vp = Viewport(
        world_width=building.width,
        world_height=building.height,
        screen_width=1200,
        screen_height=800,
    )
    renderer = EnvironmentRenderer()

    # Draw using registry directly
    renderer.draw(surface, building, vp, registry=registry)

    # Check colors at centers
    pos_active = vp.world_to_screen_point(p_active.position)
    px_active = surface.get_at((int(pos_active[0]), int(pos_active[1])))[:3]
    assert px_active in (renderer.COLOR_PED_ACTIVE, renderer.COLOR_PED_ACTIVE_CORE)

    pos_blocked = vp.world_to_screen_point(p_blocked.position)
    px_blocked = surface.get_at((int(pos_blocked[0]), int(pos_blocked[1])))[:3]
    assert px_blocked == renderer.COLOR_PED_BLOCKED


def test_renderer_does_not_mutate_simulation_state():
    """Verify that renderer is strictly read-only and never mutates pedestrian or engine state."""
    building = create_initial_building()
    ped = PedestrianState(
        id="ped_immutable",
        position=Point(12.0, 15.0),
        preferred_speed=1.5,
        target_exit_id="exit_left",
        current_area_id="main_corridor",
    )
    registry = PedestrianRegistry()
    registry.register(ped)
    engine = SimulationEngine(building=building, registry=registry, dt=0.1)

    initial_pos = copy.deepcopy(ped.position)
    initial_vel = copy.deepcopy(ped.velocity)
    initial_status = ped.status
    initial_waypoints = copy.deepcopy(ped.waypoints)
    initial_history_len = len(ped.trajectory_history)
    initial_step = engine.clock.step_count
    initial_time = engine.clock.timestamp

    surface = pygame.Surface((1200, 800))
    vp = Viewport(
        world_width=building.width,
        world_height=building.height,
        screen_width=1200,
        screen_height=800,
    )
    renderer = EnvironmentRenderer()

    # Draw 5 consecutive frames
    for _ in range(5):
        renderer.draw(surface, building, vp, engine=engine)

    # Assert zero mutation
    assert ped.position == initial_pos
    assert ped.velocity == initial_vel
    assert ped.status == initial_status
    assert ped.waypoints == initial_waypoints
    assert len(ped.trajectory_history) == initial_history_len
    assert engine.clock.step_count == initial_step
    assert engine.clock.timestamp == initial_time


def test_simulation_app_timing_and_accumulator():
    """Verify decoupled fixed-timestep accumulator behavior without time.sleep()."""
    app = SimulationApp(dt=0.1)

    # 1. Zero delta executes 0 steps
    steps = app.update(0.0)
    assert steps == 0
    assert app.sim_accumulator == 0.0
    assert app.engine.clock.step_count == 0

    # 2. Accumulate delta less than dt: no step taken, remainder retained
    steps = app.update(0.04)
    assert steps == 0
    assert app.sim_accumulator == pytest.approx(0.04)
    assert app.engine.clock.step_count == 0

    # 3. Add more delta to cross dt boundary: 1 step taken, remainder retained
    steps = app.update(0.06)  # accumulator reaches 0.10 -> 1 step taken
    assert steps == 1
    assert app.sim_accumulator == pytest.approx(0.0)
    assert app.engine.clock.step_count == 1
    assert app.engine.clock.timestamp == pytest.approx(0.1)

    # 4. Multi-step delta with non-zero remainder
    steps = app.update(0.24)  # 2 steps taken (0.2s), 0.04s remainder
    assert steps == 2
    assert app.sim_accumulator == pytest.approx(0.04)
    assert app.engine.clock.step_count == 3
    assert app.engine.clock.timestamp == pytest.approx(0.3)


def test_simulation_app_catchup_ceiling_and_delta_clamp():
    """Verify maximum 5-step catch-up ceiling and 0.25s wall-clock clamping."""
    # With dt=0.01s, 0.25s corresponds to 25 steps, but catch-up limit is 5 steps
    app = SimulationApp(dt=0.01)

    # Large frame delta (1.0s) should be clamped to 0.25s
    # Max 5 steps executed, remainder (0.20s) retained
    steps = app.update(1.0)
    assert steps == 5
    assert app.engine.clock.step_count == 5
    assert app.engine.clock.timestamp == pytest.approx(0.05)
    assert app.sim_accumulator == pytest.approx(0.20)


def test_simulation_app_pause_resume():
    """Verify pause stops simulation advancement and resume restores it."""
    app = SimulationApp(dt=0.1)
    assert not app.is_paused

    # Pause
    app.pause()
    assert app.is_paused

    # Updates while paused do not step simulation or accumulate time
    steps = app.update(0.5)
    assert steps == 0
    assert app.engine.clock.step_count == 0
    assert app.sim_accumulator == 0.0

    # Resume
    app.resume()
    assert not app.is_paused

    steps = app.update(0.1)
    assert steps == 1
    assert app.engine.clock.step_count == 1

    # Toggle pause
    app.toggle_pause()
    assert app.is_paused
    app.toggle_pause()
    assert not app.is_paused


def test_simulation_app_deterministic_reset():
    """Verify deterministic reset creates fresh unmutated instances at initial state."""
    app = SimulationApp(dt=0.1)
    orig_peds = list(app.registry)
    orig_positions = [Point(p.position.x, p.position.y) for p in orig_peds]

    # Advance simulation by several steps to mutate states
    for _ in range(5):
        app.update(0.2)

    assert app.engine.clock.step_count > 0
    assert app.engine.clock.timestamp > 0.0

    # Call reset
    app.reset()

    # Clock and accumulator must be zero
    assert app.engine.clock.step_count == 0
    assert app.engine.clock.timestamp == 0.0
    assert app.sim_accumulator == 0.0

    new_peds = list(app.registry)
    assert len(new_peds) == len(orig_peds)

    # Crucial: verify fresh objects were instantiated, not reused
    for old_p, new_p in zip(orig_peds, new_peds):
        assert old_p is not new_p

    # Verify positions match initial factory coordinates
    new_positions = [p.position for p in new_peds]
    for orig_pt, new_pt in zip(orig_positions, new_positions):
        assert orig_pt.x == pytest.approx(new_pt.x)
        assert orig_pt.y == pytest.approx(new_pt.y)


def test_simulation_app_event_handling():
    """Verify interactive event handling for quit, pause, reset, and resize."""
    app = SimulationApp(dt=0.1)

    # QUIT event
    assert not app.handle_event(pygame.event.Event(pygame.QUIT))

    # ESCAPE key
    assert not app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))

    # 'Q' key
    assert not app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q))

    # SPACE key (pause toggle)
    assert not app.is_paused
    assert app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    assert app.is_paused
    assert app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))
    assert not app.is_paused

    # 'R' key (reset)
    app.update(0.2)
    assert app.engine.clock.step_count == 2
    assert app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
    assert app.engine.clock.step_count == 0

    # VIDEORESIZE event
    assert app.handle_event(pygame.event.Event(pygame.VIDEORESIZE, w=1400, h=900))
    assert app.viewport.screen_width == 1400
    assert app.viewport.screen_height == 900


def test_simulation_app_headless_run():
    """Verify headless execution of SimulationApp.run(max_frames=5) without hanging."""
    app = SimulationApp(dt=0.1)
    app.run(max_frames=5)
    # Renders 5 frames successfully
    assert app.engine.clock.step_count >= 0


def test_default_pedestrian_cohort_factory_creates_fresh_instances():
    """Verify create_default_pedestrian_cohort produces distinct objects on each call."""
    cohort_1 = create_default_pedestrian_cohort()
    cohort_2 = create_default_pedestrian_cohort()

    assert len(cohort_1) == 8
    assert len(cohort_2) == 8

    for p1, p2 in zip(cohort_1, cohort_2):
        assert p1.id == p2.id
        assert p1 is not p2
        assert p1.position.x == p2.position.x
        assert p1.position.y == p2.position.y

