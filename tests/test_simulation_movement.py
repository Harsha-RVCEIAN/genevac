"""Unit tests for genevac.simulation.movement: continuous clearance, routing, and exit crossing."""

import pytest

from genevac.environment.builder import create_initial_building
from genevac.environment.geometry import LineSegment, Point, Rectangle, Wall, WallAlignment, WallType
from genevac.environment.models import Building, Doorway, Exit, WalkableArea
from genevac.pedestrian.models import PedestrianState, PedestrianStatus, Velocity2D
from genevac.simulation.movement import (
    advance_pedestrian_step,
    check_collision,
    detect_exit_crossing,
    distance_point_to_rect,
    distance_segment_to_rect,
    distance_segment_to_segment,
    generate_deterministic_route,
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
    """Verify rejection of non-existent area IDs and exit IDs."""
    pt = Point(5.0, 5.0)
    with pytest.raises(ValueError, match="Unknown start walkable area"):
        generate_deterministic_route(initial_building, pt, "room_ghost", "exit_left")

    with pytest.raises(ValueError, match="Unknown target exit"):
        generate_deterministic_route(initial_building, pt, "room_1", "exit_secret")


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
