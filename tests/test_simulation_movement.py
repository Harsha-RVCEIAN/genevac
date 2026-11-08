"""Unit tests for genevac.simulation.movement: continuous clearance, routing, and exit crossing."""

import pytest

from genevac.environment.builder import create_initial_building
from genevac.environment.geometry import LineSegment, Point, Rectangle, Wall, WallAlignment, WallType
from genevac.environment.models import Building, Doorway, Exit, Obstacle, WalkableArea
from genevac.pedestrian.models import PedestrianState, PedestrianStatus, Velocity2D
from genevac.simulation.exceptions import NoRouteError
from genevac.simulation.movement import (
    EPSILON,
    advance_pedestrian_step,
    advance_population_step,
    check_collision,
    check_pedestrian_collision,
    check_pedestrian_overlap,
    continuous_pedestrian_distance,
    detect_exit_crossing,
    distance_point_to_rect,
    distance_segment_to_rect,
    distance_segment_to_segment,
    find_walkable_area_for_point,
    generate_deterministic_route,
    is_point_clear_of_obstacles,
    is_position_in_walkable_space,
)


@pytest.fixture
def initial_building() -> Building:
    """Fixture providing the canonical 40x30m initial building."""
    return create_initial_building()


# --- Continuous Analytical Distance Tests ---


def test_distance_point_to_rect():
    """Verify Euclidean distance calculations from points to axis-aligned rectangles."""
    rect = Rectangle(x=10.0, y=10.0, width=5.0, height=5.0)

    # Point inside rectangle
    assert distance_point_to_rect(Point(12.0, 12.0), rect) == 0.0

    # Point on boundary
    assert distance_point_to_rect(Point(10.0, 12.0), rect) == 0.0

    # Orthogonal outside points
    assert distance_point_to_rect(Point(8.0, 12.0), rect) == pytest.approx(2.0)
    assert distance_point_to_rect(Point(17.0, 12.0), rect) == pytest.approx(2.0)
    assert distance_point_to_rect(Point(12.0, 7.0), rect) == pytest.approx(3.0)
    assert distance_point_to_rect(Point(12.0, 19.0), rect) == pytest.approx(4.0)

    # Diagonal outside point: (7, 6) relative to (10, 10) -> dx=3, dy=4 -> dist=5
    assert distance_point_to_rect(Point(7.0, 6.0), rect) == pytest.approx(5.0)


def test_distance_segment_to_segment_basic():
    """Verify basic orthogonal and intersecting segment distances."""
    s1 = LineSegment(Point(0.0, 0.0), Point(5.0, 0.0))
    s2 = LineSegment(Point(0.0, 2.0), Point(5.0, 2.0))
    assert distance_segment_to_segment(s1.start, s1.end, s2.start, s2.end) == pytest.approx(2.0)

    s3 = LineSegment(Point(2.5, -1.0), Point(2.5, 3.0))
    assert distance_segment_to_segment(s1.start, s1.end, s3.start, s3.end) == 0.0


def test_distance_parallel_overlapping_segments():
    """1. Verify parallel overlapping segments return exact perpendicular distance (3.0 m)."""
    p0, p1 = Point(0.0, 0.0), Point(4.0, 0.0)
    q0, q1 = Point(2.0, 3.0), Point(6.0, 3.0)
    assert distance_segment_to_segment(p0, p1, q0, q1) == pytest.approx(3.0)


def test_distance_parallel_separated_segments():
    """2. Verify parallel segments with disjoint projected intervals."""
    p0, p1 = Point(0.0, 0.0), Point(2.0, 0.0)
    q0, q1 = Point(4.0, 3.0), Point(6.0, 3.0)
    assert distance_segment_to_segment(p0, p1, q0, q1) == pytest.approx(3.605551275463989)


def test_distance_collinear_overlapping_segments():
    """3. Verify collinear overlapping segments return 0.0 m."""
    c0, c1 = Point(0.0, 0.0), Point(5.0, 0.0)
    c2, c3 = Point(3.0, 0.0), Point(8.0, 0.0)
    assert distance_segment_to_segment(c0, c1, c2, c3) == pytest.approx(0.0)


def test_distance_collinear_disjoint_segments():
    """4. Verify collinear disjoint segments return gap distance (3.0 m)."""
    c0, c1 = Point(0.0, 0.0), Point(2.0, 0.0)
    c2, c3 = Point(5.0, 0.0), Point(8.0, 0.0)
    assert distance_segment_to_segment(c0, c1, c2, c3) == pytest.approx(3.0)


def test_distance_reversed_endpoint_order():
    """5. Verify distance calculation is invariant to reversed endpoint order."""
    p0, p1 = Point(4.0, 0.0), Point(0.0, 0.0)
    q0, q1 = Point(6.0, 3.0), Point(2.0, 3.0)
    assert distance_segment_to_segment(p0, p1, q0, q1) == pytest.approx(3.0)


def test_distance_zero_length_first_segment():
    """6. Verify degenerate zero-length first segment returns point-to-segment distance (4.0 m)."""
    p0, p1 = Point(1.0, 1.0), Point(1.0, 1.0)
    q0, q1 = Point(1.0, 5.0), Point(4.0, 5.0)
    assert distance_segment_to_segment(p0, p1, q0, q1) == pytest.approx(4.0)


def test_distance_zero_length_second_segment():
    """7. Verify degenerate zero-length second segment returns point-to-segment distance (4.0 m)."""
    p0, p1 = Point(1.0, 5.0), Point(4.0, 5.0)
    q0, q1 = Point(1.0, 1.0), Point(1.0, 1.0)
    assert distance_segment_to_segment(p0, p1, q0, q1) == pytest.approx(4.0)


def test_distance_both_segments_zero_length():
    """8. Verify both segments zero-length returns point-to-point Euclidean distance (5.0 m)."""
    p0, p1 = Point(1.0, 1.0), Point(1.0, 1.0)
    q0, q1 = Point(4.0, 5.0), Point(4.0, 5.0)
    assert distance_segment_to_segment(p0, p1, q0, q1) == pytest.approx(5.0)


def test_distance_near_parallel_intersecting_internally():
    """9. Verify near-parallel segments that cross internally return 0.0 m."""
    p0, p1 = Point(-10.0, -1e-7), Point(10.0, 1e-7)
    q0, q1 = Point(-10.0, 1e-7), Point(10.0, -1e-7)
    assert distance_segment_to_segment(p0, p1, q0, q1) == pytest.approx(0.0)


def test_distance_near_parallel_non_intersecting():
    """10. Verify near-parallel non-intersecting segments return exact minimum distance (2.0 m)."""
    p0, p1 = Point(-10.0, 0.0), Point(10.0, 0.0)
    q0, q1 = Point(-10.0, 2.0), Point(10.0, 2.0 + 1e-7)
    assert distance_segment_to_segment(p0, p1, q0, q1) == pytest.approx(2.0)


def test_distance_short_segments_intersecting_internally():
    """Verify two short segments (< 1e-6 m, length 2e-7 m) intersecting internally return 0.0 m."""
    # Length = 2e-7 m (0.2 micrometres), strictly shorter than 1e-6 m
    p0 = Point(-1e-7, 0.0)
    p1 = Point(1e-7, 0.0)
    q0 = Point(0.0, -1e-7)
    q1 = Point(0.0, 1e-7)
    assert distance_segment_to_segment(p0, p1, q0, q1) == pytest.approx(0.0)


def test_distance_short_segment_and_point_segment():
    """Verify distance between a short segment (< 1e-6 m) and a point segment."""
    # Short segment along x from 0 to 2e-7 at y=0.0
    p0 = Point(0.0, 0.0)
    p1 = Point(2e-7, 0.0)
    # Point segment at (1e-7, 3.0)
    q0 = Point(1e-7, 3.0)
    q1 = Point(1e-7, 3.0)
    # Orthogonal projection onto midpoint of segment: distance is exactly 3.0 m
    assert distance_segment_to_segment(p0, p1, q0, q1) == pytest.approx(3.0)


def test_distance_two_distinct_zero_length_segments():
    """Verify Euclidean distance between two distinct zero-length point segments."""
    p0 = Point(1.0, 2.0)
    p1 = Point(1.0, 2.0)
    q0 = Point(4.0, 6.0)
    q1 = Point(4.0, 6.0)
    # hypot(4 - 1, 6 - 2) = hypot(3, 4) = 5.0 m
    assert distance_segment_to_segment(p0, p1, q0, q1) == pytest.approx(5.0)


def test_distance_short_segments_non_intersecting_analytical():
    """Verify analytically calculated distance between two short segments (< 1e-6 m) that do not intersect."""
    # Short segment P along y=0 from x=0 to x=2e-7
    p0 = Point(0.0, 0.0)
    p1 = Point(2e-7, 0.0)
    # Parallel short segment Q along y=4.0 from x=0 to x=2e-7
    q0 = Point(0.0, 4.0)
    q1 = Point(2e-7, 4.0)
    # Projected intervals coincide: distance is exactly 4.0 m
    assert distance_segment_to_segment(p0, p1, q0, q1) == pytest.approx(4.0)


# --- Adversarial Collision Tests ---


def test_adversarial_collision_between_fixed_samples():
    """Verify collision is detected even when obstacle is located between previous 5-sample points.

    Segment runs from x=0.0 to x=10.0 at y=5.0.
    A 5-sample check evaluates t = [0.0, 0.25, 0.5, 0.75, 1.0], which samples x = [0.0, 2.5, 5.0, 7.5, 10.0].
    We place a thin obstacle at x in [1.20, 1.30] (t = 0.125), which falls directly between sample 0 and 1.
    """
    thin_obs = Rectangle(x=1.20, y=4.0, width=0.10, height=2.0)
    seg = LineSegment(Point(0.0, 5.0), Point(10.0, 5.0))

    dist = distance_segment_to_rect(seg, thin_obs)
    assert dist == 0.0  # Continuous analytical check detects exact intersection!


def test_segment_passing_just_outside_radius_expanded_geometry():
    """Verify segment passing just outside obstacle radius clearance is accepted."""
    obs = Rectangle(x=10.0, y=10.0, width=4.0, height=4.0)
    radius = 0.25

    # Passing horizontally with clearance = 0.26m (> radius 0.25m)
    # y = 10.0 - 0.26 = 9.74
    seg_clear = LineSegment(Point(5.0, 9.74), Point(15.0, 9.74))
    dist = distance_segment_to_rect(seg_clear, obs)
    assert dist == pytest.approx(0.26)
    assert dist >= radius - 1e-6  # Permitted by contact policy!


def test_obstacle_tangency_at_exact_pedestrian_radius():
    """11. Verify segment tangent at exactly the pedestrian radius (dist == r) is permitted."""
    obs = Rectangle(x=10.0, y=10.0, width=4.0, height=4.0)
    radius = 0.25

    # Exactly tangent segment (dist = 0.25m == radius)
    # y = 10.0 - 0.25 = 9.75
    seg_tangent = LineSegment(Point(5.0, 9.75), Point(15.0, 9.75))
    dist_tangent = distance_segment_to_rect(seg_tangent, obs)
    assert dist_tangent == pytest.approx(0.25)
    assert dist_tangent >= radius - 1e-6  # Permitted by contact policy!


def test_obstacle_penetration_just_inside_pedestrian_radius():
    """12. Verify segment penetrating just inside the pedestrian radius (dist < r) is rejected."""
    obs = Rectangle(x=10.0, y=10.0, width=4.0, height=4.0)
    radius = 0.25

    # Grazing penetrating segment (dist = 0.249m < radius)
    seg_penetrating = LineSegment(Point(5.0, 9.751), Point(15.0, 9.751))
    dist_penetrating = distance_segment_to_rect(seg_penetrating, obs)
    assert dist_penetrating == pytest.approx(0.249)
    assert dist_penetrating < radius - 1e-6  # Rejected by contact policy!


def test_obstacle_tangency_contact_policy():
    """Contact policy verification covering both tangent clearance and penetration rejection."""
    test_obstacle_tangency_at_exact_pedestrian_radius()
    test_obstacle_penetration_just_inside_pedestrian_radius()


def test_zero_length_segment_inside_obstacle_rejected():
    """Verify zero-length segment positioned inside an obstacle has distance 0.0 and is rejected."""
    obs = Rectangle(x=4.0, y=4.0, width=2.0, height=1.5)
    seg_zero = LineSegment(Point(5.0, 4.5), Point(5.0, 4.5))

    dist = distance_segment_to_rect(seg_zero, obs)
    assert dist == 0.0


def test_cannot_pass_through_paper_thin_wall():
    """Verify a pedestrian cannot tunnel through a wall merely because the wall is thin."""
    # A wall with 0.01m thickness
    paper_thin_wall = Rectangle(x=5.0, y=0.0, width=0.01, height=10.0)
    seg = LineSegment(Point(4.5, 5.0), Point(5.5, 5.0))

    dist = distance_segment_to_rect(seg, paper_thin_wall)
    assert dist == 0.0


def test_valid_doorway_crossing_remains_possible(initial_building: Building):
    """Verify valid doorway crossing with radius 0.25 passes through 2.0m doorway without collision."""
    # Doorway door_room1_corridor at (10.0, 13.5). Moving across from y=13.0 to y=14.0
    seg = LineSegment(Point(10.0, 13.0), Point(10.0, 14.0))
    radius = 0.25

    # Check against all walls
    for wall in initial_building.walls:
        d = distance_segment_to_rect(seg, wall.footprint)
        assert d >= radius - 1e-6


# --- Route Validity Audit Tests ---


def test_deterministic_route_generation_identical_inputs(initial_building: Building):
    """Verify route generation produces identical waypoint lists for identical inputs."""
    start_pos = Point(5.0, 5.0)
    route1 = generate_deterministic_route(initial_building, start_pos, "room_1", "exit_left")
    route2 = generate_deterministic_route(initial_building, start_pos, "room_1", "exit_left")
    assert route1 == route2


def test_complete_route_audit_all_combinations_and_segments(initial_building: Building):
    """Audit all 15 room-to-exit combinations: verify every route segment has full clearance."""
    areas = ["room_1", "room_2", "main_corridor", "room_3", "room_4"]
    exits = ["exit_left", "exit_right", "exit_bottom"]
    radius = 0.25

    for area_id in areas:
        for exit_id in exits:
            area = initial_building.get_walkable_area(area_id)
            assert area is not None
            # Test multiple distinct spawn points in the room (including occluded zones)
            test_points = [
                Point(area.bounds.left + 2.0, area.bounds.top + 2.0),
                Point(area.bounds.left + 2.0, area.bounds.bottom - 2.0),
                Point(area.bounds.right - 2.0, area.bounds.top + 2.0),
                Point(area.bounds.right - 2.0, area.bounds.bottom - 2.0),
            ]

            for start_pt in test_points:
                route = generate_deterministic_route(initial_building, start_pt, area_id, exit_id, radius=radius)
                assert len(route) >= 1

                # Verify full segment-by-segment clearance along entire route
                waypoints = [start_pt] + route
                for i in range(len(waypoints) - 1):
                    seg_start = waypoints[i]
                    seg_end = waypoints[i + 1]

                    # Final segment reaches exit opening on exterior boundary
                    is_final_seg = (i == len(waypoints) - 2)
                    target_exit = initial_building.get_exit(exit_id) if is_final_seg else None

                    assert not check_collision(
                        seg_start, seg_end, radius, initial_building, target_exit=target_exit
                    ), f"Route collision in {area_id}->{exit_id} segment {seg_start.as_tuple()}->{seg_end.as_tuple()}"


def test_route_generation_rejects_unknown_areas_and_exits(initial_building: Building):
    """Verify rejection of non-existent area IDs and exit IDs raises ValueError directly (not NoRouteError)."""
    pt = Point(5.0, 5.0)
    with pytest.raises(ValueError, match="Unknown start walkable area") as excinfo:
        generate_deterministic_route(initial_building, pt, "room_ghost", "exit_left")
    assert type(excinfo.value) is ValueError
    assert not isinstance(excinfo.value, NoRouteError)

    with pytest.raises(ValueError, match="Unknown target exit") as excinfo2:
        generate_deterministic_route(initial_building, pt, "room_1", "exit_secret")
    assert type(excinfo2.value) is ValueError
    assert not isinstance(excinfo2.value, NoRouteError)


def test_route_generation_raises_no_route_error_for_disconnection(initial_building: Building):
    """Verify that topological disconnection raises NoRouteError distinguishable from ValueError."""
    pt = Point(5.0, 5.0)
    initial_building.doorways = [d for d in initial_building.doorways if not d.connects("room_1", "main_corridor")]
    with pytest.raises(NoRouteError, match="No doorway connecting room_1 to main_corridor") as excinfo:
        generate_deterministic_route(initial_building, pt, "room_1", "exit_left")
    assert isinstance(excinfo.value, NoRouteError)


# --- Exit Crossing Edge Case Tests ---


def test_exit_crossing_assigned_exit_detected_and_position_preserved(initial_building: Building):
    """Verify assigned exit crossing detects intersection and preserves exact crossing coordinate."""
    ex_left = initial_building.get_exit("exit_left")
    assert ex_left is not None
    radius = 0.25

    crossed, cross_pt = detect_exit_crossing(Point(0.1, 15.2), Point(-0.1, 15.2), radius, ex_left)
    assert crossed is True
    assert cross_pt is not None
    assert cross_pt.x == pytest.approx(0.0)
    assert cross_pt.y == pytest.approx(15.2)  # Exact crossing Y preserved, not teleported


def test_exit_crossing_exact_usable_opening_boundary(initial_building: Building):
    """Verify crossing exactly at the usable opening boundary (min + r and max - r) is accepted."""
    ex_left = initial_building.get_exit("exit_left")  # y in [14.0, 16.0]
    assert ex_left is not None
    radius = 0.25

    # North usable boundary: y = 14.0 + 0.25 = 14.25
    c_north, pt_north = detect_exit_crossing(Point(0.1, 14.25), Point(-0.1, 14.25), radius, ex_left)
    assert c_north is True
    assert pt_north == Point(0.0, 14.25)

    # South usable boundary: y = 16.0 - 0.25 = 15.75
    c_south, pt_south = detect_exit_crossing(Point(0.1, 15.75), Point(-0.1, 15.75), radius, ex_left)
    assert c_south is True
    assert pt_south == Point(0.0, 15.75)


def test_exit_crossing_outside_usable_opening_rejected(initial_building: Building):
    """Verify crossing outside usable opening (door jamb collision) is rejected."""
    ex_left = initial_building.get_exit("exit_left")
    assert ex_left is not None
    radius = 0.25

    # 1. Just outside north boundary: y = 14.24 (< 14.25)
    c_out_n, _ = detect_exit_crossing(Point(0.1, 14.24), Point(-0.1, 14.24), radius, ex_left)
    assert c_out_n is False

    # 2. Just outside south boundary: y = 15.76 (> 15.75)
    c_out_s, _ = detect_exit_crossing(Point(0.1, 15.76), Point(-0.1, 15.76), radius, ex_left)
    assert c_out_s is False


def test_exit_crossing_zero_length_movement_does_not_evacuate(initial_building: Building):
    """Verify zero-length movement on or near the exit does not trigger evacuation."""
    ex_left = initial_building.get_exit("exit_left")
    assert ex_left is not None
    radius = 0.25

    crossed, _ = detect_exit_crossing(Point(0.0, 15.0), Point(0.0, 15.0), radius, ex_left)
    assert crossed is False


def test_exit_crossing_inward_movement_from_boundary_does_not_evacuate(initial_building: Building):
    """Verify movement starting on the boundary and moving inwards into building does not evacuate."""
    ex_left = initial_building.get_exit("exit_left")
    assert ex_left is not None
    radius = 0.25

    # Moving right (+X into corridor) from boundary x=0.0
    crossed, _ = detect_exit_crossing(Point(0.0, 15.0), Point(0.1, 15.0), radius, ex_left)
    assert crossed is False


def test_exit_crossing_starting_on_boundary_moving_outwards_evacuates(initial_building: Building):
    """Verify movement starting on boundary and moving outwards across it evacuates."""
    ex_left = initial_building.get_exit("exit_left")
    assert ex_left is not None
    radius = 0.25

    crossed, cross_pt = detect_exit_crossing(Point(0.0, 15.0), Point(-0.1, 15.0), radius, ex_left)
    assert crossed is True
    assert cross_pt == Point(0.0, 15.0)


# --- Kinematic & State Update Tests ---


def test_movement_in_open_space(initial_building: Building):
    """Verify preferred speed and dt displacement consistency in open corridor."""
    ped = PedestrianState(
        id="ped_corr",
        position=Point(10.0, 15.0),
        preferred_speed=1.3,
        waypoints=[Point(20.0, 15.0)],
    )
    dt = 0.1

    advance_pedestrian_step(ped, initial_building, dt)

    expected_dx = 1.3 * 0.1  # 0.13m
    assert ped.position.x == pytest.approx(10.0 + expected_dx)
    assert ped.position.y == pytest.approx(15.0)
    assert ped.velocity.vx == pytest.approx(1.3)
    assert ped.velocity.vy == pytest.approx(0.0)
    assert ped.status == PedestrianStatus.ACTIVE


def test_non_active_pedestrians_do_not_move(initial_building: Building):
    """Verify BLOCKED, INACTIVE, and EVACUATED pedestrians do not advance."""
    for st in [PedestrianStatus.BLOCKED, PedestrianStatus.INACTIVE, PedestrianStatus.EVACUATED]:
        ped = PedestrianState(
            id=f"ped_{st.value}",
            position=Point(10.0, 15.0),
            status=st,
            preferred_speed=1.3,
            waypoints=[Point(20.0, 15.0)],
        )
        advance_pedestrian_step(ped, initial_building, dt=0.1)

        assert ped.position == Point(10.0, 15.0)
        assert ped.status == st


# ---------------------------------------------------------------------------
# Phase 2 Tests: Runtime Walkable-Space Validation & Area Tracking
# ---------------------------------------------------------------------------


def test_movement_inside_open_room(initial_building: Building):
    """Verify pedestrian moves safely inside an open room away from obstacles."""
    ped = PedestrianState(
        id="p_room",
        position=Point(2.0, 2.0),
        preferred_speed=1.0,
        waypoints=[Point(4.0, 2.0)],
        current_area_id="room_1",
    )
    advance_pedestrian_step(ped, initial_building, dt=0.5)
    assert ped.position == Point(2.5, 2.0)
    assert ped.current_area_id == "room_1"
    assert ped.status == PedestrianStatus.ACTIVE


def test_movement_through_valid_doorway(initial_building: Building):
    """Verify pedestrian moves smoothly across doorway portal into corridor."""
    ped = PedestrianState(
        id="p_door",
        position=Point(10.0, 13.0),
        preferred_speed=1.0,
        waypoints=[Point(10.0, 14.0)],
        current_area_id="room_1",
    )
    # Step moves through doorway center y=13.5 to y=14.0 in main_corridor
    advance_pedestrian_step(ped, initial_building, dt=1.0)
    assert ped.position == Point(10.0, 14.0)
    assert ped.current_area_id == "main_corridor"
    assert ped.status == PedestrianStatus.ACTIVE


def test_movement_outside_walkable_space_rejected(initial_building: Building):
    """Verify moving outside building bounds or into non-walkable wall zone is rejected."""
    # Attempting to move outside building bounds at y < 0
    ped = PedestrianState(
        id="p_oob",
        position=Point(5.0, 0.2),
        preferred_speed=1.0,
        waypoints=[Point(5.0, -0.5)],
        current_area_id="room_1",
    )
    advance_pedestrian_step(ped, initial_building, dt=0.5)
    assert ped.status == PedestrianStatus.BLOCKED
    assert ped.velocity == Velocity2D(0.0, 0.0)


def test_attempted_movement_through_walls_blocked(initial_building: Building):
    """Verify advancing into a solid interior wall transitions pedestrian to BLOCKED."""
    ped = PedestrianState(
        id="p_wall",
        position=Point(5.0, 13.0),
        preferred_speed=1.0,
        waypoints=[Point(5.0, 14.0)],  # Interior horizontal wall between room_1 and corridor at y=13.5
        current_area_id="room_1",
    )
    advance_pedestrian_step(ped, initial_building, dt=1.0)
    assert ped.status == PedestrianStatus.BLOCKED
    assert ped.velocity == Velocity2D(0.0, 0.0)


def test_attempted_movement_into_obstacle_blocked(initial_building: Building):
    """Verify advancing into a static obstacle transitions pedestrian to BLOCKED."""
    ped = PedestrianState(
        id="p_obs",
        position=Point(3.5, 4.75),
        preferred_speed=1.0,
        waypoints=[Point(4.5, 4.75)],  # obstacle_room1 is [4.0, 4.0, 2.0, 1.5]
        current_area_id="room_1",
    )
    advance_pedestrian_step(ped, initial_building, dt=1.0)
    assert ped.status == PedestrianStatus.BLOCKED
    assert ped.velocity == Velocity2D(0.0, 0.0)


def test_is_position_in_walkable_space(initial_building: Building):
    """Verify is_position_in_walkable_space handles interior, boundary, and obstacle positions."""
    # Inside corridor
    assert is_position_in_walkable_space(Point(20.0, 15.0), radius=0.25, building=initial_building) is True
    # Inside obstacle_room1
    assert is_position_in_walkable_space(Point(5.0, 4.75), radius=0.25, building=initial_building) is False
    # Outside building
    assert is_position_in_walkable_space(Point(-1.0, 15.0), radius=0.25, building=initial_building) is False
    # Inside wall footprint
    assert is_position_in_walkable_space(Point(20.0, 5.0), radius=0.25, building=initial_building) is False


def test_find_walkable_area_disambiguation(initial_building: Building):
    """Verify find_walkable_area_for_point disambiguates shared boundaries using velocity."""
    # Boundary between room_1 and corridor is at y=13.5, door x in [9.0, 11.0]
    door_mid = Point(10.0, 13.5)
    # Moving downward (+y) into corridor
    area_down = find_walkable_area_for_point(initial_building, door_mid, velocity=Velocity2D(0.0, 1.2))
    assert area_down is not None
    assert area_down.id == "main_corridor"

    # Moving upward (-y) into room_1
    area_up = find_walkable_area_for_point(initial_building, door_mid, velocity=Velocity2D(0.0, -1.2))
    assert area_up is not None
    assert area_up.id == "room_1"


# ---------------------------------------------------------------------------
# Phase 3 Tests: Deterministic Pedestrian Separation
# ---------------------------------------------------------------------------


def test_pedestrian_overlap_detection():
    """Verify check_pedestrian_overlap detects disk penetration and allows tangent contact."""
    # Overlapping disks
    assert check_pedestrian_overlap(Point(0.0, 0.0), 0.25, Point(0.4, 0.0), 0.25) is True
    # Tangent contact (dist == r1 + r2)
    assert check_pedestrian_overlap(Point(0.0, 0.0), 0.25, Point(0.5, 0.0), 0.25) is False
    # Separated disks
    assert check_pedestrian_overlap(Point(0.0, 0.0), 0.25, Point(0.6, 0.0), 0.25) is False


def test_continuous_pedestrian_distance_and_collision():
    """Verify continuous pedestrian collision detection detects head-on and crossing paths."""
    # Head-on conflict during step
    # P1 moves (0, 0) -> (1, 0), P2 moves (1, 0) -> (0, 0)
    conflict = check_pedestrian_collision(
        start_a=Point(0.0, 0.0), end_a=Point(1.0, 0.0), radius_a=0.25,
        start_b=Point(1.0, 0.0), end_b=Point(0.0, 0.0), radius_b=0.25,
    )
    assert conflict is True

    # Parallel same-direction movement with adequate lateral separation
    no_conflict = check_pedestrian_collision(
        start_a=Point(0.0, 0.0), end_a=Point(1.0, 0.0), radius_a=0.25,
        start_b=Point(0.0, 1.0), end_b=Point(1.0, 1.0), radius_b=0.25,
    )
    assert no_conflict is False


def test_two_pedestrians_approaching_each_other(initial_building: Building):
    """Verify two pedestrians approaching head-on in corridor separate deterministically."""
    p1 = PedestrianState(
        id="ped_01",
        position=Point(10.0, 15.0),
        preferred_speed=1.0,
        waypoints=[Point(20.0, 15.0)],
        current_area_id="main_corridor",
    )
    p2 = PedestrianState(
        id="ped_02",
        position=Point(11.0, 15.0),
        preferred_speed=1.0,
        waypoints=[Point(5.0, 15.0)],
        current_area_id="main_corridor",
    )
    # Step population
    advance_population_step([p1, p2], initial_building, dt=0.5)
    # Check that minimum separation is strictly preserved
    dist = p1.position.distance_to(p2.position)
    assert dist >= p1.radius + p2.radius - EPSILON
    assert p1.status == PedestrianStatus.ACTIVE
    assert p2.status == PedestrianStatus.ACTIVE


def test_stationary_pedestrian_blocks_route_deterministically(initial_building: Building):
    """Verify a stationary pedestrian blocking a narrow space stops the approaching pedestrian safely."""
    # Single-file corridor with a blocked pedestrian right in front
    p_blocker = PedestrianState(
        id="ped_blocker",
        position=Point(10.5, 15.0),
        status=PedestrianStatus.BLOCKED,
        waypoints=[],
        current_area_id="main_corridor",
    )
    p_walker = PedestrianState(
        id="ped_walker",
        position=Point(10.0, 15.0),
        preferred_speed=1.0,
        waypoints=[Point(11.0, 15.0)],
        current_area_id="main_corridor",
    )
    # Advance single pedestrian aware of others
    advance_pedestrian_step(p_walker, initial_building, dt=0.5, other_pedestrians=[p_blocker])
    # Separation must be preserved and no wall clipping occurred
    assert p_walker.position.distance_to(p_blocker.position) >= p_walker.radius + p_blocker.radius - EPSILON


# ---------------------------------------------------------------------------
# Phase 4 Tests: General Obstacle Navigation
# ---------------------------------------------------------------------------


def test_route_generation_around_custom_obstacle(initial_building: Building):
    """Verify visibility-graph route generation routes around an added obstacle."""
    # Place an obstacle directly on the corridor centerline between x=18 and x=22
    custom_obs = Obstacle(
        id="obs_corridor_blocker",
        bounds=Rectangle(x=18.0, y=14.2, width=4.0, height=1.6),
    )
    custom_building = Building(
        id="bldg_custom_obs",
        width=initial_building.width,
        height=initial_building.height,
        walkable_areas=initial_building.walkable_areas,
        obstacles=list(initial_building.obstacles) + [custom_obs],
        doorways=initial_building.doorways,
        walls=initial_building.walls,
        exits=initial_building.exits,
    )

    # Route from (15.0, 15.0) to exit_right
    route = generate_deterministic_route(
        building=custom_building,
        start_pos=Point(15.0, 15.0),
        start_area_id="main_corridor",
        target_exit_id="exit_right",
        radius=0.25,
    )
    assert len(route) >= 1
    # Check that every segment of the generated route clears all obstacles and walls
    curr = Point(15.0, 15.0)
    for wp in route:
        assert not check_collision(curr, wp, radius=0.25, building=custom_building, target_exit=custom_building.get_exit("exit_right"))
        curr = wp


def test_route_generation_raises_no_route_when_completely_blocked(initial_building: Building):
    """Verify NoRouteError is raised when static obstacles completely seal the corridor path to an exit."""
    # Place an obstacle spanning the full height of the corridor (y in [13.5, 16.5]) at x in [30.0, 32.0]
    # Without overlapping the doorway at x in [29.0, 31.0], place it at x in [32.0, 35.0]
    block_obs = Obstacle(
        id="obs_blocking_corridor",
        bounds=Rectangle(x=32.0, y=13.5, width=3.0, height=3.0),
    )
    custom_building = Building(
        id="bldg_blocked_corridor",
        width=initial_building.width,
        height=initial_building.height,
        walkable_areas=initial_building.walkable_areas,
        obstacles=list(initial_building.obstacles) + [block_obs],
        doorways=initial_building.doorways,
        walls=initial_building.walls,
        exits=initial_building.exits,
    )

    with pytest.raises(NoRouteError):
        generate_deterministic_route(
            building=custom_building,
            start_pos=Point(20.0, 15.0),
            start_area_id="main_corridor",
            target_exit_id="exit_right",
            radius=0.25,
        )


# ---------------------------------------------------------------------------
# Section 5: Specific Edge Cases 1-12
# ---------------------------------------------------------------------------


def test_edge_case_1_two_pedestrians_perpendicular_collision(initial_building: Building):
    """Edge Case 1: Two pedestrians reach doorway portal at the same instant from perpendicular angles."""
    # Ped 1 coming north from corridor (10.0, 14.5) to room_1 doorway (10.0, 13.5)
    # Ped 2 coming east along corridor (9.0, 13.8) crossing across doorway approach
    p1 = PedestrianState(
        id="ped_perp_1",
        position=Point(10.0, 14.5),
        preferred_speed=1.0,
        waypoints=[Point(10.0, 13.0)],
        current_area_id="main_corridor",
    )
    p2 = PedestrianState(
        id="ped_perp_2",
        position=Point(9.2, 14.0),
        preferred_speed=1.0,
        waypoints=[Point(11.0, 14.0)],
        current_area_id="main_corridor",
    )
    advance_population_step([p1, p2], initial_building, dt=0.5)
    assert p1.position.distance_to(p2.position) >= p1.radius + p2.radius - EPSILON
    assert is_position_in_walkable_space(p1.position, p1.radius, initial_building)
    assert is_position_in_walkable_space(p2.position, p2.radius, initial_building)


def test_edge_case_2_multiple_pedestrians_narrow_doorway_bottleneck(initial_building: Building):
    """Edge Case 2: Three pedestrians arrive at 2.0m doorway simultaneously, bottleneck queues without clipping."""
    # Place three pedestrians with valid initial separation >= 0.5m
    peds = [
        PedestrianState(id="p_queue_1", position=Point(9.4, 11.4), preferred_speed=1.2, waypoints=[Point(10.0, 15.0)], current_area_id="room_1"),
        PedestrianState(id="p_queue_2", position=Point(10.0, 12.0), preferred_speed=1.2, waypoints=[Point(10.0, 15.0)], current_area_id="room_1"),
        PedestrianState(id="p_queue_3", position=Point(10.6, 11.4), preferred_speed=1.2, waypoints=[Point(10.0, 15.0)], current_area_id="room_1"),
    ]
    for _ in range(5):
        advance_population_step(peds, initial_building, dt=0.2)
        # Verify mutual separation
        for i in range(len(peds)):
            for j in range(i + 1, len(peds)):
                assert peds[i].position.distance_to(peds[j].position) >= peds[i].radius + peds[j].radius - EPSILON
            assert is_position_in_walkable_space(peds[i].position, peds[i].radius, initial_building)


def test_edge_case_3_pedestrian_deflection_does_not_clip_wall(initial_building: Building):
    """Edge Case 3: Pedestrian pushed sideways near wall/obstacle remains strictly in walkable space."""
    # Near top wall of corridor (y=13.5), with valid clearance (y=13.8, radius=0.25 -> y-r = 13.55 > 13.5)
    # Pedestrian p_block is stationary at (15.2, 14.0). Pedestrian p_mover approaches from (14.5, 14.0).
    p_block = PedestrianState(id="p_b", position=Point(15.2, 14.0), status=PedestrianStatus.BLOCKED, waypoints=[], current_area_id="main_corridor")
    p_mover = PedestrianState(id="p_m", position=Point(14.5, 14.0), preferred_speed=1.0, waypoints=[Point(16.0, 14.0)], current_area_id="main_corridor")
    advance_population_step([p_block, p_mover], initial_building, dt=0.3)
    assert is_position_in_walkable_space(p_mover.position, p_mover.radius, initial_building)
    assert not check_collision(Point(14.5, 14.0), p_mover.position, p_mover.radius, initial_building)


def test_edge_case_4_zero_velocity_tangent_to_doorway_edge(initial_building: Building):
    """Edge Case 4: Stationary pedestrian resting tangent to a doorway post/edge does not cause false collision."""
    # Doorway at x in [9.0, 11.0], y=13.5. Left door post is at (9.0, 13.5).
    # Pedestrian with center at (9.25, 13.5) touches post tangentially with radius 0.25
    ped_pos = Point(9.25, 13.5)
    assert is_position_in_walkable_space(ped_pos, radius=0.25, building=initial_building)
    area = find_walkable_area_for_point(initial_building, ped_pos, velocity=Velocity2D(0.0, 0.0))
    assert area is not None


def test_edge_case_5_pathfinding_with_no_obstacles(initial_building: Building):
    """Edge Case 5: Pathfinding in a completely empty building with only walls and doorways."""
    empty_building = Building(
        id="bldg_empty",
        width=initial_building.width,
        height=initial_building.height,
        walkable_areas=initial_building.walkable_areas,
        obstacles=[],
        doorways=initial_building.doorways,
        walls=initial_building.walls,
        exits=initial_building.exits,
    )
    route = generate_deterministic_route(
        building=empty_building,
        start_pos=Point(5.0, 5.0),
        start_area_id="room_1",
        target_exit_id="exit_left",
        radius=0.25,
    )
    assert len(route) >= 1
    assert route[-1] == Point(0.0, 15.0)


def test_edge_case_6_high_density_corridor_deadlock_detection(initial_building: Building):
    """Edge Case 6: High-density corridor head-on deadlock is handled deterministically without crashing."""
    peds = [
        PedestrianState(id="p_east", position=Point(10.0, 15.0), preferred_speed=1.0, waypoints=[Point(20.0, 15.0)], current_area_id="main_corridor"),
        PedestrianState(id="p_west", position=Point(10.6, 15.0), preferred_speed=1.0, waypoints=[Point(5.0, 15.0)], current_area_id="main_corridor"),
    ]
    for _ in range(5):
        advance_population_step(peds, initial_building, dt=0.2)
        assert peds[0].position.distance_to(peds[1].position) >= peds[0].radius + peds[1].radius - EPSILON


def test_edge_case_7_pedestrian_exits_while_another_enters_vicinity(initial_building: Building):
    """Edge Case 7: Evacuated pedestrian does not block trailing pedestrian from evacuating."""
    p_lead = PedestrianState(
        id="p_lead",
        position=Point(0.2, 15.0),
        preferred_speed=1.0,
        waypoints=[Point(0.0, 15.0)],
        target_exit_id="exit_left",
        current_area_id="main_corridor",
    )
    p_trail = PedestrianState(
        id="p_trail",
        position=Point(0.8, 15.0),
        preferred_speed=1.0,
        waypoints=[Point(0.0, 15.0)],
        target_exit_id="exit_left",
        current_area_id="main_corridor",
    )
    # Step 1: p_lead evacuates
    advance_population_step([p_lead, p_trail], initial_building, dt=0.5)
    assert p_lead.status == PedestrianStatus.EVACUATED
    # Step 2: p_trail advances freely without being blocked by evacuated p_lead
    advance_population_step([p_lead, p_trail], initial_building, dt=0.5)
    assert p_trail.position.x < 0.8
    assert p_trail.status in (PedestrianStatus.ACTIVE, PedestrianStatus.EVACUATED)


def test_edge_case_8_exit_closes_while_pedestrian_in_exit_approach(initial_building: Building):
    """Edge Case 8: Exit closing while pedestrian crosses turns pedestrian BLOCKED immediately."""
    p = PedestrianState(
        id="p_closing",
        position=Point(0.3, 15.0),
        preferred_speed=1.0,
        waypoints=[Point(-0.2, 15.0)],
        target_exit_id="exit_left",
        current_area_id="main_corridor",
    )
    ex = initial_building.get_exit("exit_left")
    assert ex is not None
    ex.close()

    # Step should detect closed exit collision or unassigned crossing and block pedestrian
    advance_pedestrian_step(p, initial_building, dt=0.5)
    assert p.status == PedestrianStatus.BLOCKED
    assert p.velocity == Velocity2D(0.0, 0.0)


def test_edge_case_9_sub_millimeter_contact_tangency():
    """Edge Case 9: Separation exactly at r1 + r2 - 1e-7 is detected as overlap, r1 + r2 + 1e-7 is clear."""
    r1, r2 = 0.25, 0.25
    pos_a = Point(0.0, 0.0)
    # Overlap
    pos_b_overlap = Point(0.5 - 1e-5, 0.0)
    assert check_pedestrian_overlap(pos_a, r1, pos_b_overlap, r2) is True
    # Clear
    pos_b_clear = Point(0.5 + 1e-5, 0.0)
    assert check_pedestrian_overlap(pos_a, r1, pos_b_clear, r2) is False


def test_edge_case_10_large_pedestrian_radius_in_standard_doorway(initial_building: Building):
    """Edge Case 10: Pedestrian with radius=1.1m attempting to pass through 2.0m doorway is blocked."""
    # Doorway width is 2.0m; pedestrian with radius=1.1m (diameter 2.2m) cannot fit
    large_ped = PedestrianState(
        id="p_large",
        position=Point(10.0, 12.0),
        radius=1.1,
        preferred_speed=1.0,
        waypoints=[Point(10.0, 15.0)],
        current_area_id="room_1",
    )
    advance_pedestrian_step(large_ped, initial_building, dt=1.0)
    assert large_ped.status == PedestrianStatus.BLOCKED
    assert large_ped.velocity == Velocity2D(0.0, 0.0)


def test_edge_case_11_route_generation_around_narrow_passage_obstacles(initial_building: Building):
    """Edge Case 11: Route generation finds path through 2.5m gap between two interior obstacles."""
    obs1 = Obstacle(id="obs_gap_1", bounds=Rectangle(x=15.0, y=13.5, width=4.0, height=0.5))
    obs2 = Obstacle(id="obs_gap_2", bounds=Rectangle(x=15.0, y=15.5, width=4.0, height=0.5))
    # Gap between obs1 and obs2 is y in [14.0, 15.5], height 1.5m, plenty for radius 0.25m
    bldg = Building(
        id="bldg_narrow_gap",
        width=initial_building.width,
        height=initial_building.height,
        walkable_areas=initial_building.walkable_areas,
        obstacles=[obs1, obs2],
        doorways=initial_building.doorways,
        walls=initial_building.walls,
        exits=initial_building.exits,
    )
    route = generate_deterministic_route(
        building=bldg,
        start_pos=Point(10.0, 14.75),
        start_area_id="main_corridor",
        target_exit_id="exit_right",
        radius=0.25,
    )
    assert len(route) >= 1
    # Verify every leg clears obstacles
    curr = Point(10.0, 14.75)
    for wp in route:
        assert not check_collision(curr, wp, 0.25, bldg, target_exit=bldg.get_exit("exit_right"))
        curr = wp


def test_edge_case_12_repeated_simulations_identical_trajectories(initial_building: Building):
    """Edge Case 12: Repeated simulation runs with identical initial conditions produce bit-identical positions."""
    def run_sim():
        p1 = PedestrianState(id="p1", position=Point(5.0, 5.0), preferred_speed=1.2, waypoints=[Point(10.0, 5.0)], current_area_id="room_1")
        p2 = PedestrianState(id="p2", position=Point(10.0, 15.0), preferred_speed=1.4, waypoints=[Point(20.0, 15.0)], current_area_id="main_corridor")
        positions = []
        for _ in range(10):
            advance_population_step([p1, p2], initial_building, dt=0.1)
            positions.append(((p1.position.x, p1.position.y), (p2.position.x, p2.position.y)))
        return positions

    run1 = run_sim()
    run2 = run_sim()
    assert run1 == run2


# =============================================================================
# Week 3 Final Corrections — Task 2: Continuous Collision Unit Tests
# =============================================================================


def test_task2_check_pedestrian_collision_identical_coordinates():
    """1. Pedestrians starting at identical coordinates collide immediately."""
    start_a = Point(5.0, 5.0)
    end_a = Point(6.0, 5.0)
    start_b = Point(5.0, 5.0)
    end_b = Point(6.0, 5.0)
    assert check_pedestrian_collision(start_a, end_a, 0.25, start_b, end_b, 0.25) is True


def test_task2_check_pedestrian_collision_stationary_overlapping():
    """2. Two stationary pedestrians starting in physical contact/overlap collide."""
    # Distance = 0.3m, sum of radii = 0.5m
    start_a = Point(5.0, 5.0)
    end_a = Point(5.0, 5.0)
    start_b = Point(5.3, 5.0)
    end_b = Point(5.3, 5.0)
    assert check_pedestrian_collision(start_a, end_a, 0.25, start_b, end_b, 0.25) is True


def test_task2_check_pedestrian_collision_contact_threshold():
    """3. Two stationary pedestrians exactly at contact threshold (dist == r1 + r2) do not collide."""
    # Distance = 0.5m, sum of radii = 0.5m
    start_a = Point(5.0, 5.0)
    end_a = Point(5.0, 5.0)
    start_b = Point(5.5, 5.0)
    end_b = Point(5.5, 5.0)
    assert check_pedestrian_collision(start_a, end_a, 0.25, start_b, end_b, 0.25) is False


def test_task2_check_pedestrian_collision_separated_valid_distance():
    """4. Two pedestrians separated by valid distance do not collide."""
    start_a = Point(5.0, 5.0)
    end_a = Point(5.0, 5.0)
    start_b = Point(6.0, 5.0)
    end_b = Point(6.0, 5.0)
    assert check_pedestrian_collision(start_a, end_a, 0.25, start_b, end_b, 0.25) is False


def test_task2_check_pedestrian_collision_identical_trajectories():
    """5. Two pedestrians moving along identical trajectories collide."""
    start_a = Point(5.0, 5.0)
    end_a = Point(7.0, 5.0)
    start_b = Point(5.0, 5.0)
    end_b = Point(7.0, 5.0)
    assert check_pedestrian_collision(start_a, end_a, 0.25, start_b, end_b, 0.25) is True


def test_task2_check_pedestrian_collision_crossing_paths():
    """6. Crossing paths intersecting during one timestep collide."""
    # Path A: (4.0, 5.0) -> (6.0, 5.0)
    # Path B: (5.0, 4.0) -> (5.0, 6.0)
    # They cross at (5.0, 5.0) around t=0.5
    start_a = Point(4.0, 5.0)
    end_a = Point(6.0, 5.0)
    start_b = Point(5.0, 4.0)
    end_b = Point(5.0, 6.0)
    assert check_pedestrian_collision(start_a, end_a, 0.25, start_b, end_b, 0.25) is True


def test_task2_check_pedestrian_collision_head_on():
    """7. Head-on approaching paths collide during the timestep."""
    start_a = Point(4.0, 5.0)
    end_a = Point(6.0, 5.0)
    start_b = Point(6.0, 5.0)
    end_b = Point(4.0, 5.0)
    assert check_pedestrian_collision(start_a, end_a, 0.25, start_b, end_b, 0.25) is True


def test_task2_check_pedestrian_collision_different_radii():
    """8. Collision detection handles pedestrians of different radii correctly."""
    # r_a = 0.2, r_b = 0.5 -> contact dist = 0.7m. Distance = 0.6m -> collide.
    start_a = Point(5.0, 5.0)
    end_a = Point(5.0, 5.0)
    start_b = Point(5.6, 5.0)
    end_b = Point(5.6, 5.0)
    assert check_pedestrian_collision(start_a, end_a, 0.2, start_b, end_b, 0.5) is True

    # Distance = 0.8m > 0.7m -> no collision
    start_c = Point(5.8, 5.0)
    end_c = Point(5.8, 5.0)
    assert check_pedestrian_collision(start_a, end_a, 0.2, start_c, end_c, 0.5) is False


def test_task2_check_pedestrian_collision_parallel_clearance():
    """9. Parallel moving paths with sufficient lateral clearance do not collide."""
    start_a = Point(0.0, 0.0)
    end_a = Point(10.0, 0.0)
    start_b = Point(0.0, 1.0)
    end_b = Point(10.0, 1.0)
    # Lateral distance = 1.0m, sum of radii = 0.5m -> no collision
    assert check_pedestrian_collision(start_a, end_a, 0.25, start_b, end_b, 0.25) is False


# =============================================================================
# Week 3 Final Corrections — Task 3: Population Collision Resolution & Stress
# =============================================================================


def test_task3_stress_narrow_passage_queue(initial_building: Building):
    """1. Multiple pedestrians in narrow passage queue without penetrating each other or walls."""
    # Place 3 pedestrians in line in main_corridor at y=15.0 heading west to exit_left
    p1 = PedestrianState(id="p_lead", position=Point(3.0, 15.0), radius=0.25, preferred_speed=1.0, waypoints=[Point(0.0, 15.0)], target_exit_id="exit_left", current_area_id="main_corridor")
    p2 = PedestrianState(id="p_mid", position=Point(3.7, 15.0), radius=0.25, preferred_speed=1.0, waypoints=[Point(0.0, 15.0)], target_exit_id="exit_left", current_area_id="main_corridor")
    p3 = PedestrianState(id="p_tail", position=Point(4.4, 15.0), radius=0.25, preferred_speed=1.0, waypoints=[Point(0.0, 15.0)], target_exit_id="exit_left", current_area_id="main_corridor")

    for _ in range(10):
        advance_population_step([p1, p2, p3], initial_building, dt=0.1)
        # Verify mutual separation
        assert not check_pedestrian_overlap(p1.position, p1.radius, p2.position, p2.radius)
        assert not check_pedestrian_overlap(p2.position, p2.radius, p3.position, p3.radius)
        # Verify corridor containment
        for p in [p1, p2, p3]:
            assert 10.0 <= p.position.y <= 20.0
            assert is_position_in_walkable_space(p.position, p.radius, initial_building, target_exit=initial_building.get_exit("exit_left"))


def test_task3_stress_doorway_convergence(initial_building: Building):
    """2. Several pedestrians approaching the same doorway resolve safely without wall clipping."""
    # doorway_room1_corridor at (10.0, 10.0)
    p1 = PedestrianState(id="p_door1", position=Point(9.5, 9.0), radius=0.25, preferred_speed=1.2, waypoints=[Point(10.0, 12.0)], current_area_id="room_1")
    p2 = PedestrianState(id="p_door2", position=Point(10.5, 9.0), radius=0.25, preferred_speed=1.2, waypoints=[Point(10.0, 12.0)], current_area_id="room_1")

    for _ in range(5):
        advance_population_step([p1, p2], initial_building, dt=0.1)
        assert not check_pedestrian_overlap(p1.position, p1.radius, p2.position, p2.radius)
        for p in [p1, p2]:
            assert is_position_in_walkable_space(p.position, p.radius, initial_building)


def test_task3_stress_three_simultaneous_conflicts(initial_building: Building):
    """3. Three simultaneous conflicts resolve deterministically without collision or teleportation."""
    # 4 pedestrians in corridor converging towards a central point (15.0, 15.0)
    p_n = PedestrianState(id="p_north", position=Point(15.0, 16.0), radius=0.25, preferred_speed=1.0, waypoints=[Point(15.0, 14.0)], current_area_id="main_corridor")
    p_s = PedestrianState(id="p_south", position=Point(15.0, 14.0), radius=0.25, preferred_speed=1.0, waypoints=[Point(15.0, 16.0)], current_area_id="main_corridor")
    p_w = PedestrianState(id="p_west", position=Point(14.0, 15.0), radius=0.25, preferred_speed=1.0, waypoints=[Point(16.0, 15.0)], current_area_id="main_corridor")
    p_e = PedestrianState(id="p_east", position=Point(16.0, 15.0), radius=0.25, preferred_speed=1.0, waypoints=[Point(14.0, 15.0)], current_area_id="main_corridor")
    peds = [p_n, p_s, p_w, p_e]

    for _ in range(8):
        advance_population_step(peds, initial_building, dt=0.1)
        # Verify no pairwise overlap
        for i in range(len(peds)):
            for j in range(i + 1, len(peds)):
                assert not check_pedestrian_overlap(peds[i].position, peds[i].radius, peds[j].position, peds[j].radius)
        # Verify no teleportation (max distance per step <= speed * dt + epsilon)
        for p in peds:
            assert is_position_in_walkable_space(p.position, p.radius, initial_building)


def test_task3_stationary_pedestrian_obstruction(initial_building: Building):
    """4. Stationary pedestrian obstructs moving pedestrian; moving pedestrian halts or deflects."""
    p_stat = PedestrianState(id="p_stat", position=Point(12.0, 15.0), radius=0.25, preferred_speed=0.0, current_area_id="main_corridor")
    p_move = PedestrianState(id="p_move", position=Point(11.0, 15.0), radius=0.25, preferred_speed=1.5, waypoints=[Point(15.0, 15.0)], current_area_id="main_corridor")

    orig_stat_pos = Point(p_stat.position.x, p_stat.position.y)
    for _ in range(5):
        advance_population_step([p_stat, p_move], initial_building, dt=0.1)
        # Stationary pedestrian remains at its location
        assert p_stat.position.x == pytest.approx(orig_stat_pos.x)
        assert p_stat.position.y == pytest.approx(orig_stat_pos.y)
        # Moving pedestrian never penetrates stationary pedestrian
        assert not check_pedestrian_overlap(p_stat.position, p_stat.radius, p_move.position, p_move.radius)


def test_task3_conflicts_near_walls_and_obstacles(initial_building: Building):
    """5. Conflicts near walls/obstacles never push pedestrians into solid geometry."""
    # Room 1 dividing wall is at x=20.0, obstacle_room1 is at [4.0, 6.0] x [4.0, 5.5]
    # Place two pedestrians near the obstacle facing each other
    p_a = PedestrianState(id="p_a", position=Point(3.5, 4.75), radius=0.25, preferred_speed=1.0, waypoints=[Point(7.0, 4.75)], current_area_id="room_1")
    p_b = PedestrianState(id="p_b", position=Point(3.5, 3.5), radius=0.25, preferred_speed=1.0, waypoints=[Point(3.5, 6.0)], current_area_id="room_1")

    for _ in range(5):
        advance_population_step([p_a, p_b], initial_building, dt=0.1)
        assert not check_pedestrian_overlap(p_a.position, p_a.radius, p_b.position, p_b.radius)
        assert is_position_in_walkable_space(p_a.position, p_a.radius, initial_building)
        assert is_position_in_walkable_space(p_b.position, p_b.radius, initial_building)


def test_task3_stress_repeated_deterministic_simulation(initial_building: Building):
    """6. Multi-step multi-pedestrian simulations yield bit-identical trajectories."""
    def run():
        p1 = PedestrianState(id="p1", position=Point(2.0, 5.0), radius=0.25, preferred_speed=1.2, waypoints=[Point(10.0, 5.0)], current_area_id="room_1")
        p2 = PedestrianState(id="p2", position=Point(10.0, 15.0), radius=0.25, preferred_speed=1.4, waypoints=[Point(20.0, 15.0)], current_area_id="main_corridor")
        p3 = PedestrianState(id="p3", position=Point(15.0, 15.0), radius=0.25, preferred_speed=1.0, waypoints=[Point(10.0, 15.0)], current_area_id="main_corridor")
        history = []
        for _ in range(20):
            advance_population_step([p1, p2, p3], initial_building, dt=0.1)
            history.append((
                (p1.position.x, p1.position.y),
                (p2.position.x, p2.position.y),
                (p3.position.x, p3.position.y),
            ))
        return history

    h1 = run()
    h2 = run()
    assert h1 == h2


def test_task3_deadlock_freeze_safe_halt(initial_building: Building):
    """7. Completely trapped pedestrian halts safely with velocity zero without crashing."""
    # A pedestrian with no way forward because surrounding positions are blocked
    # surrounded by stationary pedestrians
    center = PedestrianState(id="p_trapped", position=Point(15.0, 15.0), radius=0.25, preferred_speed=1.0, waypoints=[Point(16.0, 15.0)], current_area_id="main_corridor")
    block_e = PedestrianState(id="p_be", position=Point(15.5, 15.0), radius=0.25, preferred_speed=0.0, current_area_id="main_corridor")
    block_w = PedestrianState(id="p_bw", position=Point(14.5, 15.0), radius=0.25, preferred_speed=0.0, current_area_id="main_corridor")
    block_n = PedestrianState(id="p_bn", position=Point(15.0, 15.5), radius=0.25, preferred_speed=0.0, current_area_id="main_corridor")
    block_s = PedestrianState(id="p_bs", position=Point(15.0, 14.5), radius=0.25, preferred_speed=0.0, current_area_id="main_corridor")
    all_peds = [center, block_e, block_w, block_n, block_s]

    # Step simulation
    advance_population_step(all_peds, initial_building, dt=0.1)

    # p_trapped should safely halt without crashing
    assert center.position.x == pytest.approx(15.0)
    assert center.position.y == pytest.approx(15.0)
    assert center.velocity.vx == pytest.approx(0.0)
    assert center.velocity.vy == pytest.approx(0.0)


def test_task3_cascading_three_pedestrian_conflict_safety(initial_building: Building):
    """8. Three simultaneous conflicting pedestrians: resolving one never introduces secondary collision."""
    # Ped A moves east, Ped B moves west (head-on), Ped C sits in the right-hand deflection path
    p_a = PedestrianState(id="p_headon_a", position=Point(10.0, 15.0), radius=0.25, preferred_speed=1.0, waypoints=[Point(12.0, 15.0)], current_area_id="main_corridor")
    p_b = PedestrianState(id="p_headon_b", position=Point(11.2, 15.0), radius=0.25, preferred_speed=1.0, waypoints=[Point(9.0, 15.0)], current_area_id="main_corridor")
    p_c = PedestrianState(id="p_deflect_c", position=Point(10.6, 15.6), radius=0.25, preferred_speed=1.0, waypoints=[Point(10.6, 14.0)], current_area_id="main_corridor")
    peds = [p_a, p_b, p_c]

    for _ in range(10):
        advance_population_step(peds, initial_building, dt=0.1)

        # 1. No pair overlaps beyond contact tolerance
        for i in range(len(peds)):
            for j in range(i + 1, len(peds)):
                assert not check_pedestrian_overlap(peds[i].position, peds[i].radius, peds[j].position, peds[j].radius), (
                    f"Secondary overlap between {peds[i].id} and {peds[j].id} at step {_}"
                )

        # 2. No pedestrian pushed into walls, obstacles, or outside walkable space
        for p in peds:
            assert is_position_in_walkable_space(p.position, p.radius, initial_building)


def test_task3_multi_agent_secondary_collision_guard_stops_safely(initial_building: Building):
    """9. Constrained lateral space: pedestrians halt safely rather than committing secondary collision."""
    # Place 3 pedestrians in tight column facing opposite directions in corridor
    p1 = PedestrianState(id="p_col1", position=Point(5.0, 14.8), radius=0.25, preferred_speed=1.2, waypoints=[Point(7.0, 14.8)], current_area_id="main_corridor")
    p2 = PedestrianState(id="p_col2", position=Point(5.8, 14.8), radius=0.25, preferred_speed=1.2, waypoints=[Point(4.0, 14.8)], current_area_id="main_corridor")
    p3 = PedestrianState(id="p_col3", position=Point(5.4, 15.3), radius=0.25, preferred_speed=1.0, waypoints=[Point(5.4, 14.0)], current_area_id="main_corridor")
    peds = [p1, p2, p3]

    for step in range(8):
        advance_population_step(peds, initial_building, dt=0.1)
        for i in range(len(peds)):
            for j in range(i + 1, len(peds)):
                assert not check_pedestrian_overlap(peds[i].position, peds[i].radius, peds[j].position, peds[j].radius)
            assert is_position_in_walkable_space(peds[i].position, peds[i].radius, initial_building)


