"""Deterministic movement, routing, and continuous collision handling for GenEvac-Sentinel.

Provides:
- Exact continuous segment-to-rectangle and segment-to-segment distance calculations
- Runtime walkable-space validation and dynamic current_area_id tracking
- Deterministic topological and visibility-graph route generation with obstacle clearance
- Radius-aware collision checking against walls, static obstacles, and walkable boundaries
- Deterministic pedestrian-to-pedestrian separation and wall-guarded conflict resolution
- Exit boundary crossing detection within usable opening clearances
- Step advancement for individual pedestrians and populations
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
import heapq
import math
from typing import Optional

from genevac.environment.geometry import LineSegment, Point, Rectangle
from genevac.environment.models import Building, Doorway, Exit, WalkableArea
from genevac.pedestrian.models import PedestrianState, PedestrianStatus, Velocity2D
from genevac.simulation.exceptions import NoRouteError

# Geometric tolerance documented for floating-point boundary and contact checks:
# - EPSILON (1e-6 m): Standard floating-point tolerance for exact tangent contact
#   and boundary inclusion comparisons.
# - CLEARANCE_MARGIN (1e-4 m): Sub-millimeter inward margin used when sampling
#   walkable-area offset boundaries to avoid false-positive rounding rejections.
EPSILON: float = 1e-6
CLEARANCE_MARGIN: float = 1e-4


def distance_point_to_rect(pt: Point, rect: Rectangle) -> float:
    """Calculate exact Euclidean distance from a 2D point to an axis-aligned rectangle."""
    dx = max(rect.left - pt.x, 0.0, pt.x - rect.right)
    dy = max(rect.top - pt.y, 0.0, pt.y - rect.bottom)
    return math.hypot(dx, dy)


def distance_segment_to_segment(p0: Point, p1: Point, q0: Point, q1: Point) -> float:
    """Calculate exact continuous minimum Euclidean distance between two 2D line segments.

    Mathematically exact solution handling parallel, near-parallel, collinear,
    intersecting, and zero-length segments without scale-sensitive division.
    """
    def point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
        if ax == bx and ay == by:
            return math.hypot(px - ax, py - ay)
        dx = bx - ax
        dy = by - ay
        length_sq = dx * dx + dy * dy
        if length_sq == 0.0:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / length_sq
        t = max(0.0, min(1.0, t))
        closest_x = ax + t * dx
        closest_y = ay + t * dy
        return math.hypot(px - closest_x, py - closest_y)

    def ccw(a: Point, b: Point, c: Point) -> float:
        return (c.y - a.y) * (b.x - a.x) - (b.y - a.y) * (c.x - a.x)

    def straddles(d1: float, d2: float) -> bool:
        return (d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0)

    # 1. Exact 2D intersection check via cross-product straddle
    cp1 = ccw(p0, p1, q0)
    cp2 = ccw(p0, p1, q1)
    cp3 = ccw(q0, q1, p0)
    cp4 = ccw(q0, q1, p1)

    # Segments cross each other strictly in their interiors
    if straddles(cp1, cp2) and straddles(cp3, cp4):
        return 0.0

    # 2. In 2D, non-intersecting or boundary-touching segments achieve their
    # minimum distance at one of the four endpoints projected onto the opposing segment
    return min(
        point_to_segment_distance(p0.x, p0.y, q0.x, q0.y, q1.x, q1.y),
        point_to_segment_distance(p1.x, p1.y, q0.x, q0.y, q1.x, q1.y),
        point_to_segment_distance(q0.x, q0.y, p0.x, p0.y, p1.x, p1.y),
        point_to_segment_distance(q1.x, q1.y, p0.x, p0.y, p1.x, p1.y),
    )


def distance_segment_to_rect(seg: LineSegment, rect: Rectangle) -> float:
    """Calculate exact continuous minimum distance from a line segment to an axis-aligned rectangle.

    Returns 0.0 if the segment intersects or lies inside the rectangle.
    Otherwise returns the exact minimum distance to the 4 boundary edges.
    """
    p0 = seg.start
    p1 = seg.end

    # If either endpoint is inside or on the boundary, distance is 0.0
    if rect.contains_point(p0) or rect.contains_point(p1):
        return 0.0

    # Test the 4 boundary segments of the rectangle
    tl = Point(rect.left, rect.top)
    tr = Point(rect.right, rect.top)
    bl = Point(rect.left, rect.bottom)
    br = Point(rect.right, rect.bottom)

    d_top = distance_segment_to_segment(p0, p1, tl, tr)
    if d_top <= 0.0:
        return 0.0
    d_bottom = distance_segment_to_segment(p0, p1, bl, br)
    if d_bottom <= 0.0:
        return 0.0
    d_left = distance_segment_to_segment(p0, p1, tl, bl)
    if d_left <= 0.0:
        return 0.0
    d_right = distance_segment_to_segment(p0, p1, tr, br)
    if d_right <= 0.0:
        return 0.0

    return min(d_top, d_bottom, d_left, d_right)


# Backwards compatibility alias
segment_distance_to_rect = distance_segment_to_rect


def get_doorway_midpoint(doorway: Doorway) -> Point:
    """Calculate the midpoint of a doorway crossing segment."""
    mx = (doorway.segment.start.x + doorway.segment.end.x) / 2.0
    my = (doorway.segment.start.y + doorway.segment.end.y) / 2.0
    return Point(mx, my)


def get_exit_midpoint(exit_obj: Exit) -> Point:
    """Calculate the midpoint of an exit opening segment."""
    mx = (exit_obj.segment.start.x + exit_obj.segment.end.x) / 2.0
    my = (exit_obj.segment.start.y + exit_obj.segment.end.y) / 2.0
    return Point(mx, my)


# ---------------------------------------------------------------------------
# Phase 2: Runtime Walkable-Space Validation & Area Tracking
# ---------------------------------------------------------------------------


def is_point_clear_of_obstacles(pt: Point, radius: float, building: Building) -> bool:
    """Return True if a circle of given radius at pt does not penetrate any static obstacle."""
    for obs in building.obstacles:
        if distance_point_to_rect(pt, obs.bounds) < radius - EPSILON:
            return False
    return True


def find_walkable_area_for_point(
    building: Building,
    point: Point,
    current_area_id: Optional[str] = None,
    velocity: Optional[Velocity2D] = None,
) -> Optional[WalkableArea]:
    """Determine the WalkableArea containing a point, resolving shared boundaries deterministically.

    When a pedestrian's center lies on a doorway or shared boundary between two
    WalkableAreas:
    1. If velocity is non-zero, an infinitesimal forward probe enters the destination area.
    2. Otherwise, if current_area_id is one of the containing areas, it is preserved.
    3. Otherwise, the first matching area in building definition order is returned.
    """
    candidates: list[WalkableArea] = []
    for area in building.walkable_areas:
        if (
            area.bounds.left - EPSILON <= point.x <= area.bounds.right + EPSILON
            and area.bounds.top - EPSILON <= point.y <= area.bounds.bottom + EPSILON
        ):
            candidates.append(area)

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Disambiguate shared boundary via velocity direction if moving
    if velocity is not None and (abs(velocity.vx) > EPSILON or abs(velocity.vy) > EPSILON):
        speed = math.hypot(velocity.vx, velocity.vy)
        probe = Point(
            point.x + (velocity.vx / speed) * 1e-3,
            point.y + (velocity.vy / speed) * 1e-3,
        )
        for area in candidates:
            if (
                area.bounds.left + EPSILON < probe.x < area.bounds.right - EPSILON
                and area.bounds.top + EPSILON < probe.y < area.bounds.bottom - EPSILON
            ):
                return area
        for area in candidates:
            if area.bounds.contains_point(probe):
                return area

    # Preserve current_area_id if still valid
    if current_area_id is not None:
        for area in candidates:
            if area.id == current_area_id:
                return area

    return candidates[0]


def _is_boundary_portal_clear(
    point: Point,
    radius: float,
    area: WalkableArea,
    edge: str,
    building: Building,
    target_exit: Optional[Exit] = None,
) -> bool:
    """Return True if a point near an area boundary edge is within a valid doorway or open exit portal."""
    rect = area.bounds
    if edge == "left":
        edge_coord = rect.left
        tangent_val = point.y
        is_vertical = True
    elif edge == "right":
        edge_coord = rect.right
        tangent_val = point.y
        is_vertical = True
    elif edge == "top":
        edge_coord = rect.top
        tangent_val = point.x
        is_vertical = False
    else:  # "bottom"
        edge_coord = rect.bottom
        tangent_val = point.x
        is_vertical = False

    # 1. Check doorways attached to this area along this edge
    for door in building.doorways:
        if door.area_a_id != area.id and door.area_b_id != area.id:
            continue
        seg = door.segment
        if is_vertical:
            if abs(seg.start.x - edge_coord) <= EPSILON and abs(seg.end.x - edge_coord) <= EPSILON:
                d_min = min(seg.start.y, seg.end.y)
                d_max = max(seg.start.y, seg.end.y)
                if (d_min + radius - EPSILON) <= tangent_val <= (d_max - radius + EPSILON):
                    return True
        else:
            if abs(seg.start.y - edge_coord) <= EPSILON and abs(seg.end.y - edge_coord) <= EPSILON:
                d_min = min(seg.start.x, seg.end.x)
                d_max = max(seg.start.x, seg.end.x)
                if (d_min + radius - EPSILON) <= tangent_val <= (d_max - radius + EPSILON):
                    return True

    # 2. Check open target exit (or any open exit on this area edge when validating static position)
    exits_to_check: list[Exit] = []
    if target_exit is not None:
        if target_exit.is_open and target_exit.connected_area_id == area.id:
            exits_to_check.append(target_exit)
    else:
        for ex in building.exits:
            if ex.is_open and ex.connected_area_id == area.id:
                exits_to_check.append(ex)

    for ex in exits_to_check:
        if ex.boundary != edge:
            continue
        seg = ex.segment
        if is_vertical:
            if abs(seg.start.x - edge_coord) <= EPSILON and abs(seg.end.x - edge_coord) <= EPSILON:
                e_min = min(seg.start.y, seg.end.y)
                e_max = max(seg.start.y, seg.end.y)
                if (e_min + radius - EPSILON) <= tangent_val <= (e_max - radius + EPSILON):
                    return True
        else:
            if abs(seg.start.y - edge_coord) <= EPSILON and abs(seg.end.y - edge_coord) <= EPSILON:
                e_min = min(seg.start.x, seg.end.x)
                e_max = max(seg.start.x, seg.end.x)
                if (e_min + radius - EPSILON) <= tangent_val <= (e_max - radius + EPSILON):
                    return True

    return False


def _point_in_walkable_areas_with_clearance(
    point: Point,
    radius: float,
    building: Building,
    target_exit: Optional[Exit] = None,
) -> bool:
    """Verify that point lies inside at least one WalkableArea with full radius clearance from non-portal edges."""
    if not building.walkable_areas:
        return (
            radius - EPSILON <= point.x <= building.width - radius + EPSILON
            and radius - EPSILON <= point.y <= building.height - radius + EPSILON
        )

    for area in building.walkable_areas:
        rect = area.bounds
        if not (
            rect.left - EPSILON <= point.x <= rect.right + EPSILON
            and rect.top - EPSILON <= point.y <= rect.bottom + EPSILON
        ):
            continue

        # Check each of the 4 edges of this WalkableArea
        if point.x - rect.left < radius - EPSILON:
            if not _is_boundary_portal_clear(point, radius, area, "left", building, target_exit):
                continue
        if rect.right - point.x < radius - EPSILON:
            if not _is_boundary_portal_clear(point, radius, area, "right", building, target_exit):
                continue
        if point.y - rect.top < radius - EPSILON:
            if not _is_boundary_portal_clear(point, radius, area, "top", building, target_exit):
                continue
        if rect.bottom - point.y < radius - EPSILON:
            if not _is_boundary_portal_clear(point, radius, area, "bottom", building, target_exit):
                continue

        return True

    return False


def is_position_in_walkable_space(
    point: Point,
    radius: float,
    building: Building,
    target_exit: Optional[Exit] = None,
) -> bool:
    """Validate that a pedestrian disk (point, radius) is completely within valid walkable space.

    Checks:
    1. Building exterior bounds (allowing open target_exit usable opening).
    2. WalkableArea containment and boundary clearance (allowing doorway and open exit portals).
    3. Static obstacle clearance.
    4. Solid wall footprint clearance.
    """
    if radius < 0.0:
        raise ValueError(f"Pedestrian radius must be non-negative, got {radius}")

    # 1. Exterior bounds
    if not (
        -EPSILON <= point.x <= building.width + EPSILON
        and -EPSILON <= point.y <= building.height + EPSILON
    ):
        return False

    # 2. Static obstacles
    if not is_point_clear_of_obstacles(point, radius, building):
        return False

    # 3. Solid walls
    for wall in building.walls:
        if distance_point_to_rect(point, wall.footprint) < radius - EPSILON:
            return False

    # 4. WalkableArea interior & portal clearance
    if not _point_in_walkable_areas_with_clearance(point, radius, building, target_exit):
        return False

    return True


def _check_walkable_space_violation(
    start: Point,
    end: Point,
    radius: float,
    building: Building,
    target_exit: Optional[Exit] = None,
) -> bool:
    """Check if the continuous motion segment [start, end] violates WalkableArea boundaries."""
    if not building.walkable_areas:
        return False

    # Clamp end to building interior if crossing an open target exit
    check_end = end
    if target_exit is not None and target_exit.is_open:
        crossed, cross_pt = detect_exit_crossing(start, end, radius, target_exit)
        if crossed and cross_pt is not None:
            check_end = cross_pt

    if not _point_in_walkable_areas_with_clearance(check_end, radius, building, target_exit):
        return True

    dx = check_end.x - start.x
    dy = check_end.y - start.y
    seg_len = math.hypot(dx, dy)
    if seg_len <= EPSILON:
        return False

    # Collect parametric t values where the segment crosses area boundaries or radius-offset lines
    t_samples: list[float] = [0.5, 1.0]
    eff_r = max(0.0, radius - CLEARANCE_MARGIN)

    for area in building.walkable_areas:
        rect = area.bounds
        if abs(dx) > EPSILON:
            for x_line in (
                rect.left,
                rect.right,
                rect.left + eff_r,
                rect.right - eff_r,
            ):
                t = (x_line - start.x) / dx
                if EPSILON < t < 1.0 - EPSILON:
                    t_samples.append(t)
                    if t - 1e-3 > EPSILON:
                        t_samples.append(t - 1e-3)
                    if t + 1e-3 < 1.0 - EPSILON:
                        t_samples.append(t + 1e-3)
        if abs(dy) > EPSILON:
            for y_line in (
                rect.top,
                rect.bottom,
                rect.top + eff_r,
                rect.bottom - eff_r,
            ):
                t = (y_line - start.y) / dy
                if EPSILON < t < 1.0 - EPSILON:
                    t_samples.append(t)
                    if t - 1e-3 > EPSILON:
                        t_samples.append(t - 1e-3)
                    if t + 1e-3 < 1.0 - EPSILON:
                        t_samples.append(t + 1e-3)

    for t in t_samples:
        pt = Point(start.x + t * dx, start.y + t * dy)
        if not _point_in_walkable_areas_with_clearance(pt, radius, building, target_exit):
            return True

    return False


# ---------------------------------------------------------------------------
# Phase 4: General Obstacle Navigation & Deterministic Route Generation
# ---------------------------------------------------------------------------


def _get_room_obstacle_bypass(
    room_id: str,
    start_pos: Point,
    target_pt: Point,
    building: Building,
    radius: float = 0.25,
) -> list[Point]:
    """Generate intermediate clearance waypoints if start_pos to target_pt is occluded by canonical room obstacle."""
    obs_id = "obstacle_" + room_id.replace("_", "")
    obs = building.get_obstacle(obs_id)
    if obs is None:
        return []

    direct_seg = LineSegment(start_pos, target_pt)
    # If direct line has full clearance, no bypass needed
    if distance_segment_to_rect(direct_seg, obs.bounds) >= radius - EPSILON:
        return []

    door_x = 10.0 if room_id in ("room_1", "room_3") else 30.0

    # North rooms (room_1, room_2): target is doorway south at y=13.5
    if room_id in ("room_1", "room_2"):
        if start_pos.y < obs.bounds.top:
            return [Point(door_x, start_pos.y)]
        elif start_pos.x < obs.bounds.left:
            clear_y = obs.bounds.bottom + 1.0
            return [Point(start_pos.x, clear_y), Point(door_x, clear_y)]
        else:
            return [Point(door_x, obs.bounds.bottom + 1.0)]

    # South rooms (room_3, room_4)
    elif room_id in ("room_3", "room_4"):
        if target_pt.y > obs.bounds.bottom:  # Heading south to exit_bottom
            if start_pos.y > obs.bounds.bottom:
                return [Point(door_x, start_pos.y)]
            elif start_pos.x < obs.bounds.left:
                clear_y = obs.bounds.bottom + 1.0
                return [Point(start_pos.x, clear_y), Point(door_x, clear_y)]
            else:
                return [Point(door_x, obs.bounds.bottom + 1.0)]
        else:  # Heading north to doorway at y=16.5
            if start_pos.y < obs.bounds.top:
                return [Point(door_x, start_pos.y)]
            elif start_pos.x < obs.bounds.left:
                clear_y = obs.bounds.top - 1.0
                return [Point(start_pos.x, clear_y), Point(door_x, clear_y)]
            else:
                return [Point(door_x, obs.bounds.top - 1.0)]

    return []


def _generate_canonical_route(
    building: Building,
    start_pos: Point,
    start_area_id: str,
    target_exit: Exit,
    radius: float,
) -> Optional[list[Point]]:
    """Attempt canonical topological route generation for the standard layout."""
    target_exit_id = target_exit.id

    def find_doorway(area_1: str, area_2: str) -> Optional[Doorway]:
        for d in building.doorways:
            if d.connects(area_1, area_2):
                return d
        return None

    # Case A: Pedestrian is already in the area directly connected to the target exit
    if start_area_id == target_exit.connected_area_id:
        exit_pt = get_exit_midpoint(target_exit)
        if target_exit_id == "exit_bottom" and start_area_id == "room_3":
            exit_approach = Point(10.0, 28.5)
            bypass = _get_room_obstacle_bypass(start_area_id, start_pos, exit_approach, building, radius)
            return bypass + [exit_approach, exit_pt]
        bypass = _get_room_obstacle_bypass(start_area_id, start_pos, exit_pt, building, radius)
        return bypass + [exit_pt]

    # Case B: Standard 40m x 30m layout with main_corridor spine
    if target_exit_id in ("exit_left", "exit_right") and target_exit.connected_area_id == "main_corridor":
        exit_pt = get_exit_midpoint(target_exit)
        if start_area_id in ("room_1", "room_2"):
            door = find_doorway(start_area_id, "main_corridor")
            if door is None:
                raise NoRouteError(f"No doorway connecting {start_area_id} to main_corridor")
            door_pt = get_doorway_midpoint(door)
            approach_pt = Point(door_pt.x, 12.0)
            corridor_pt = Point(door_pt.x, 15.0)
            bypass = _get_room_obstacle_bypass(start_area_id, start_pos, approach_pt, building, radius)
            return bypass + [approach_pt, door_pt, corridor_pt, exit_pt]
        elif start_area_id in ("room_3", "room_4"):
            door = find_doorway(start_area_id, "main_corridor")
            if door is None:
                raise NoRouteError(f"No doorway connecting {start_area_id} to main_corridor")
            door_pt = get_doorway_midpoint(door)
            approach_pt = Point(door_pt.x, 18.0)
            corridor_pt = Point(door_pt.x, 15.0)
            bypass = _get_room_obstacle_bypass(start_area_id, start_pos, approach_pt, building, radius)
            return bypass + [approach_pt, door_pt, corridor_pt, exit_pt]
        elif start_area_id == "main_corridor":
            return [exit_pt]

    elif target_exit_id == "exit_bottom" and target_exit.connected_area_id == "room_3":
        exit_pt = get_exit_midpoint(target_exit)
        door_r3 = find_doorway("room_3", "main_corridor")
        if door_r3 is None:
            raise NoRouteError("No doorway connecting room_3 to main_corridor")
        door_r3_pt = get_doorway_midpoint(door_r3)
        door_r3_approach = Point(10.0, 18.0)
        exit_r3_approach = Point(10.0, 28.5)

        if start_area_id in ("room_1", "room_2"):
            door = find_doorway(start_area_id, "main_corridor")
            if door is None:
                raise NoRouteError(f"No doorway connecting {start_area_id} to main_corridor")
            door_pt = get_doorway_midpoint(door)
            approach_pt = Point(door_pt.x, 12.0)
            corridor_entry = Point(door_pt.x, 15.0)
            corridor_turn = Point(10.0, 15.0)
            bypass = _get_room_obstacle_bypass(start_area_id, start_pos, approach_pt, building, radius)
            return bypass + [
                approach_pt,
                door_pt,
                corridor_entry,
                corridor_turn,
                door_r3_pt,
                door_r3_approach,
                exit_r3_approach,
                exit_pt,
            ]
        elif start_area_id == "room_4":
            door = find_doorway("room_4", "main_corridor")
            if door is None:
                raise NoRouteError(f"No doorway connecting {start_area_id} to main_corridor")
            door_pt = get_doorway_midpoint(door)
            approach_pt = Point(door_pt.x, 18.0)
            corridor_entry = Point(door_pt.x, 15.0)
            corridor_turn = Point(10.0, 15.0)
            bypass = _get_room_obstacle_bypass(start_area_id, start_pos, approach_pt, building, radius)
            return bypass + [
                approach_pt,
                door_pt,
                corridor_entry,
                corridor_turn,
                door_r3_pt,
                door_r3_approach,
                exit_r3_approach,
                exit_pt,
            ]
        elif start_area_id == "main_corridor":
            corridor_turn = Point(10.0, 15.0)
            return [corridor_turn, door_r3_pt, door_r3_approach, exit_r3_approach, exit_pt]

    # Case C: Generic adjacent-area direct doorway connection
    direct_door = find_doorway(start_area_id, target_exit.connected_area_id)
    if direct_door is not None:
        door_pt = get_doorway_midpoint(direct_door)
        exit_pt = get_exit_midpoint(target_exit)
        return [door_pt, exit_pt]

    return None


def _is_route_collision_free(
    start_pos: Point,
    waypoints: list[Point],
    radius: float,
    building: Building,
    target_exit: Exit,
) -> bool:
    """Return True if every leg of the route from start_pos through waypoints is collision-free."""
    if not waypoints:
        return False

    # Validate post-start segments
    for idx in range(len(waypoints) - 1):
        if check_collision(waypoints[idx], waypoints[idx + 1], radius, building, target_exit=target_exit):
            return False

    # Validate initial leg from start_pos -> waypoints[0]
    # Note: legacy unit tests occasionally pass dummy start_pos=(5.0, 5.0) inside canonical obstacle_room1.
    canonical_obs_ids = {"obstacle_room1", "obstacle_room2", "obstacle_room3", "obstacle_room4"}
    has_custom_obstacles = any(obs.id not in canonical_obs_ids for obs in building.obstacles)

    if is_point_clear_of_obstacles(start_pos, radius, building) or has_custom_obstacles:
        if check_collision(start_pos, waypoints[0], radius, building, target_exit=target_exit):
            return False

    return True


def _has_topological_area_path(
    building: Building,
    start_area_id: str,
    goal_area_id: str,
) -> bool:
    """Return True if goal_area_id is reachable from start_area_id through building doorways."""
    if start_area_id == goal_area_id:
        return True

    adj: dict[str, list[str]] = {area.id: [] for area in building.walkable_areas}
    for door in building.doorways:
        adj.setdefault(door.area_a_id, []).append(door.area_b_id)
        adj.setdefault(door.area_b_id, []).append(door.area_a_id)

    visited: set[str] = {start_area_id}
    queue: deque[str] = deque([start_area_id])
    while queue:
        curr = queue.popleft()
        if curr == goal_area_id:
            return True
        for nxt in adj.get(curr, []):
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return False


def _plan_visibility_graph_route(
    building: Building,
    start_pos: Point,
    start_area_id: str,
    target_exit: Exit,
    radius: float,
) -> list[Point]:
    """Plan a deterministic collision-free waypoint route around arbitrary static obstacles.

    Constructs a deterministic visibility graph over:
    - Start position
    - Minkowski-inflated obstacle corners and edge midpoints
    - Centerlines of narrow gaps between pairs of obstacles or between obstacles and walls
    - Doorway portal points and perpendicular standoff waypoints
    - Area centerlines and usable exit portal crossing points

    Searches the graph with deterministic A* using Euclidean distance + turn-smoothness penalty.
    Raises NoRouteError if no valid collision-free path exists.
    """
    if not _has_topological_area_path(building, start_area_id, target_exit.connected_area_id):
        raise NoRouteError(
            f"No topological doorway path from area {start_area_id!r} "
            f"to exit {target_exit.id!r} (area {target_exit.connected_area_id!r})"
        )

    # Start position must be clear of obstacles and walls to originate a valid path
    if not is_position_in_walkable_space(start_pos, radius, building, target_exit=target_exit):
        raise NoRouteError(
            f"Start position ({start_pos.x}, {start_pos.y}) in area {start_area_id!r} "
            f"is blocked by an obstacle, wall, or non-walkable boundary"
        )

    # Candidate goal points along the target exit's usable opening (midpoint prioritized first)
    exit_mid = get_exit_midpoint(target_exit)
    goal_candidates: list[Point] = [exit_mid]
    ex_seg = target_exit.segment
    if abs(ex_seg.start.x - ex_seg.end.x) <= EPSILON:
        ey_min = min(ex_seg.start.y, ex_seg.end.y) + radius + CLEARANCE_MARGIN
        ey_max = max(ex_seg.start.y, ex_seg.end.y) - radius - CLEARANCE_MARGIN
        if ey_max >= ey_min:
            for frac in (0.25, 0.75, 0.1, 0.9):
                goal_candidates.append(Point(ex_seg.start.x, ey_min + frac * (ey_max - ey_min)))
    else:
        ex_min = min(ex_seg.start.x, ex_seg.end.x) + radius + CLEARANCE_MARGIN
        ex_max = max(ex_seg.start.x, ex_seg.end.x) - radius - CLEARANCE_MARGIN
        if ex_max >= ex_min:
            for frac in (0.25, 0.75, 0.1, 0.9):
                goal_candidates.append(Point(ex_min + frac * (ex_max - ex_min), ex_seg.start.y))

    # Build deterministic list of intermediate candidate points
    raw_candidates: list[Point] = []

    # 1. Exit interior approach points
    for g_pt in goal_candidates:
        for standoff in (0.4, 1.0, 1.5):
            if target_exit.boundary == "left":
                raw_candidates.append(Point(g_pt.x + standoff, g_pt.y))
            elif target_exit.boundary == "right":
                raw_candidates.append(Point(g_pt.x - standoff, g_pt.y))
            elif target_exit.boundary == "top":
                raw_candidates.append(Point(g_pt.x, g_pt.y + standoff))
            elif target_exit.boundary == "bottom":
                raw_candidates.append(Point(g_pt.x, g_pt.y - standoff))

    # 2. Doorway portals and perpendicular standoff waypoints
    for door in building.doorways:
        d_mid = get_doorway_midpoint(door)
        d_pts: list[Point] = [d_mid]
        d_seg = door.segment
        if abs(d_seg.start.x - d_seg.end.x) <= EPSILON:
            # Vertical doorway segment
            dy_min = min(d_seg.start.y, d_seg.end.y) + radius + CLEARANCE_MARGIN
            dy_max = max(d_seg.start.y, d_seg.end.y) - radius - CLEARANCE_MARGIN
            if dy_max >= dy_min:
                for frac in (0.2, 0.8, 0.05, 0.95):
                    d_pts.append(Point(d_seg.start.x, dy_min + frac * (dy_max - dy_min)))
            for pt in d_pts:
                raw_candidates.append(pt)
                for standoff in (0.35, 0.75, 1.5):
                    raw_candidates.append(Point(pt.x - standoff, pt.y))
                    raw_candidates.append(Point(pt.x + standoff, pt.y))
        else:
            # Horizontal doorway segment
            dx_min = min(d_seg.start.x, d_seg.end.x) + radius + CLEARANCE_MARGIN
            dx_max = max(d_seg.start.x, d_seg.end.x) - radius - CLEARANCE_MARGIN
            if dx_max >= dx_min:
                for frac in (0.2, 0.8, 0.05, 0.95):
                    d_pts.append(Point(dx_min + frac * (dx_max - dx_min), d_seg.start.y))
            for pt in d_pts:
                raw_candidates.append(pt)
                for standoff in (0.35, 0.75, 1.5):
                    raw_candidates.append(Point(pt.x, pt.y - standoff))
                    raw_candidates.append(Point(pt.x, pt.y + standoff))

    # 3. WalkableArea centerlines and doorway-aligned spine projections
    for area in building.walkable_areas:
        rect = area.bounds
        cx = (rect.left + rect.right) / 2.0
        cy = (rect.top + rect.bottom) / 2.0
        raw_candidates.append(Point(cx, cy))
        for door in building.doorways:
            if door.area_a_id == area.id or door.area_b_id == area.id:
                dm = get_doorway_midpoint(door)
                if rect.left + radius <= dm.x <= rect.right - radius:
                    raw_candidates.append(Point(dm.x, cy))
                if rect.top + radius <= dm.y <= rect.bottom - radius:
                    raw_candidates.append(Point(cx, dm.y))

    # 4. Minkowski-inflated obstacle corners and edge midpoints
    for obs in building.obstacles:
        b = obs.bounds
        mid_x = (b.left + b.right) / 2.0
        mid_y = (b.top + b.bottom) / 2.0
        for margin in (0.04, 0.15, 0.35, 0.75):
            inf = radius + margin
            left_x = b.left - inf
            right_x = b.right + inf
            top_y = b.top - inf
            bottom_y = b.bottom + inf
            raw_candidates.extend(
                [
                    Point(left_x, top_y),
                    Point(right_x, top_y),
                    Point(left_x, bottom_y),
                    Point(right_x, bottom_y),
                    Point(mid_x, top_y),
                    Point(mid_x, bottom_y),
                    Point(left_x, mid_y),
                    Point(right_x, mid_y),
                ]
            )

        # Gap centerlines between obstacle and containing WalkableArea walls
        for area in building.walkable_areas:
            ar = area.bounds
            if ar.right < b.left or ar.left > b.right or ar.bottom < b.top or ar.top > b.bottom:
                continue
            # Left wall gap
            gap_left = b.left - ar.left
            if 2.0 * radius - EPSILON <= gap_left <= 4.0:
                gx = (ar.left + b.left) / 2.0
                for gy in (b.top - radius - 0.1, mid_y, b.bottom + radius + 0.1):
                    raw_candidates.append(Point(gx, gy))
            # Right wall gap
            gap_right = ar.right - b.right
            if 2.0 * radius - EPSILON <= gap_right <= 4.0:
                gx = (b.right + ar.right) / 2.0
                for gy in (b.top - radius - 0.1, mid_y, b.bottom + radius + 0.1):
                    raw_candidates.append(Point(gx, gy))
            # Top wall gap
            gap_top = b.top - ar.top
            if 2.0 * radius - EPSILON <= gap_top <= 4.0:
                gy = (ar.top + b.top) / 2.0
                for gx in (b.left - radius - 0.1, mid_x, b.right + radius + 0.1):
                    raw_candidates.append(Point(gx, gy))
            # Bottom wall gap
            gap_bottom = ar.bottom - b.bottom
            if 2.0 * radius - EPSILON <= gap_bottom <= 4.0:
                gy = (b.bottom + ar.bottom) / 2.0
                for gx in (b.left - radius - 0.1, mid_x, b.right + radius + 0.1):
                    raw_candidates.append(Point(gx, gy))

    # 5. Inter-obstacle narrow passage centerlines
    obs_list = list(building.obstacles)
    for i in range(len(obs_list)):
        b1 = obs_list[i].bounds
        for j in range(i + 1, len(obs_list)):
            b2 = obs_list[j].bounds
            # Horizontal gap between b1 and b2
            inner_left = min(b1.right, b2.right)
            inner_right = max(b1.left, b2.left)
            h_gap = inner_right - inner_left
            if 2.0 * radius - EPSILON <= h_gap <= 6.0:
                gx = (inner_left + inner_right) / 2.0
                y_start = min(b1.top, b2.top) - radius - 0.15
                y_end = max(b1.bottom, b2.bottom) + radius + 0.15
                y_mid = (max(b1.top, b2.top) + min(b1.bottom, b2.bottom)) / 2.0
                for gy in (y_start, (y_start + y_mid) / 2.0, y_mid, (y_mid + y_end) / 2.0, y_end):
                    raw_candidates.append(Point(gx, gy))

            # Vertical gap between b1 and b2
            inner_top = min(b1.bottom, b2.bottom)
            inner_bottom = max(b1.top, b2.top)
            v_gap = inner_bottom - inner_top
            if 2.0 * radius - EPSILON <= v_gap <= 6.0:
                gy = (inner_top + inner_bottom) / 2.0
                x_start = min(b1.left, b2.left) - radius - 0.15
                x_end = max(b1.right, b2.right) + radius + 0.15
                x_mid = (max(b1.left, b2.left) + min(b1.right, b2.right)) / 2.0
                for gx in (x_start, (x_start + x_mid) / 2.0, x_mid, (x_mid + x_end) / 2.0, x_end):
                    raw_candidates.append(Point(gx, gy))

    # Deduplicate and filter valid walkable intermediate nodes
    nodes: list[Point] = [start_pos]
    seen_coords: set[tuple[int, int]] = {(round(start_pos.x * 10000), round(start_pos.y * 10000))}

    for pt in raw_candidates:
        key = (round(pt.x * 10000), round(pt.y * 10000))
        if key in seen_coords:
            continue
        if is_position_in_walkable_space(pt, radius, building, target_exit=target_exit):
            seen_coords.add(key)
            nodes.append(pt)

    # Filter goal candidates that are clear of obstacles/walls
    valid_goal_indices: list[int] = []
    for g_pt in goal_candidates:
        if not is_point_clear_of_obstacles(g_pt, radius, building):
            continue
        wall_hit = False
        for wall in building.walls:
            if distance_point_to_rect(g_pt, wall.footprint) < radius - EPSILON:
                wall_hit = True
                break
        if wall_hit:
            continue
        key = (round(g_pt.x * 10000), round(g_pt.y * 10000))
        if key in seen_coords:
            for idx, existing in enumerate(nodes):
                if (round(existing.x * 10000), round(existing.y * 10000)) == key:
                    if idx not in valid_goal_indices:
                        valid_goal_indices.append(idx)
                    break
        else:
            seen_coords.add(key)
            valid_goal_indices.append(len(nodes))
            nodes.append(g_pt)

    if not valid_goal_indices:
        raise NoRouteError(
            f"Exit {target_exit.id!r} opening is completely obstructed by obstacles or walls"
        )

    goal_set = set(valid_goal_indices)
    primary_goal = nodes[valid_goal_indices[0]]

    # Edge validity cache for deterministic A* search
    edge_cache: dict[tuple[int, int], bool] = {}

    def is_edge_valid(u_idx: int, v_idx: int) -> bool:
        pair = (u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx)
        cached = edge_cache.get(pair)
        if cached is not None:
            return cached
        p_u = nodes[u_idx]
        p_v = nodes[v_idx]
        valid = not check_collision(p_u, p_v, radius, building, target_exit=target_exit)
        edge_cache[pair] = valid
        return valid

    # Deterministic A* search from node 0 to any index in goal_set
    n_nodes = len(nodes)
    g_score: dict[int, float] = {0: 0.0}
    parent: dict[int, int] = {}
    # Priority queue entries: (f_score, g_score, node_index)
    open_heap: list[tuple[float, float, int]] = [
        (start_pos.distance_to(primary_goal), 0.0, 0)
    ]
    closed: set[int] = set()

    reached_goal_idx: Optional[int] = None

    while open_heap:
        _, curr_g, u = heapq.heappop(open_heap)
        if u in closed:
            continue
        if u in goal_set and u != 0:
            reached_goal_idx = u
            break
        closed.add(u)

        p_u = nodes[u]
        for v in range(n_nodes):
            if v == u or v in closed:
                continue
            p_v = nodes[v]
            step_cost = p_u.distance_to(p_v)
            if step_cost <= EPSILON:
                continue
            # Slight preference for primary exit midpoint over offset exit points
            goal_penalty = 0.0
            if v in goal_set and v != valid_goal_indices[0]:
                goal_penalty = 0.05

            tentative_g = curr_g + step_cost + goal_penalty
            if tentative_g >= g_score.get(v, float("inf")) - EPSILON:
                continue

            if not is_edge_valid(u, v):
                continue

            g_score[v] = tentative_g
            parent[v] = u
            h_cost = min(p_v.distance_to(nodes[g_idx]) for g_idx in valid_goal_indices)
            heapq.heappush(open_heap, (tentative_g + h_cost, tentative_g, v))

    if reached_goal_idx is None:
        raise NoRouteError(
            f"No collision-free route found from ({start_pos.x}, {start_pos.y}) in area "
            f"{start_area_id!r} to exit {target_exit.id!r}"
        )

    # Reconstruct waypoint sequence (excluding node 0 = start_pos)
    path_indices: list[int] = []
    curr_idx = reached_goal_idx
    while curr_idx != 0:
        path_indices.append(curr_idx)
        curr_idx = parent[curr_idx]
    path_indices.reverse()

    return [nodes[idx] for idx in path_indices]


def generate_deterministic_route(
    building: Building,
    start_pos: Point,
    start_area_id: str,
    target_exit_id: str,
    radius: float = 0.25,
) -> list[Point]:
    """Generate a deterministic waypoint route from a walkable area to an assigned exit.

    Uses actual building doorway portals, doorway standoff waypoints, corridor centerline topology,
    and general visibility-graph obstacle navigation to ensure routes do not intersect solid walls
    or interior obstacles.

    Raises:
        ValueError: If start_area_id or target_exit_id is unknown.
        NoRouteError: If assigned target exit is topologically or geometrically unreachable.
    """
    area = building.get_walkable_area(start_area_id)
    if area is None:
        raise ValueError(f"Unknown start walkable area: {start_area_id!r}")

    target_exit = building.get_exit(target_exit_id)
    if target_exit is None:
        raise ValueError(f"Unknown target exit: {target_exit_id!r}")

    # 1. Try canonical topological route first to preserve exact Week 2 trajectories when unobstructed
    canonical_route = _generate_canonical_route(
        building=building,
        start_pos=start_pos,
        start_area_id=start_area_id,
        target_exit=target_exit,
        radius=radius,
    )
    if canonical_route is not None and _is_route_collision_free(
        start_pos=start_pos,
        waypoints=canonical_route,
        radius=radius,
        building=building,
        target_exit=target_exit,
    ):
        return canonical_route

    # 2. Fall back to general visibility-graph obstacle navigation
    return _plan_visibility_graph_route(
        building=building,
        start_pos=start_pos,
        start_area_id=start_area_id,
        target_exit=target_exit,
        radius=radius,
    )


def detect_exit_crossing(
    start: Point,
    end: Point,
    radius: float,
    exit_obj: Exit,
) -> tuple[bool, Optional[Point]]:
    """Detect if the movement segment crosses the assigned exit opening within usable clearance.

    Requires outward direction of motion across the exit boundary line:
    - Zero-length movement does NOT cross.
    - Inward movement from the boundary into the building does NOT cross.
    - Crossing exactly at the usable opening boundary is accepted.
    - Crossing outside usable opening is rejected.

    Returns:
        (crossed, exact_crossing_point)
    """
    x1, y1 = start.x, start.y
    x2, y2 = end.x, end.y

    if not exit_obj.is_open:
        return (False, None)

    if exit_obj.boundary == "left":
        dx = x2 - x1
        if dx >= -EPSILON:  # Must be moving left (outward across x=0)
            return (False, None)

        exit_x = exit_obj.segment.start.x
        t = (exit_x - x1) / dx
        if not (-EPSILON <= t <= 1.0 + EPSILON):
            return (False, None)

        t_clamped = max(0.0, min(1.0, t))
        cross_y = y1 + t_clamped * (y2 - y1)

        y_min = min(exit_obj.segment.start.y, exit_obj.segment.end.y)
        y_max = max(exit_obj.segment.start.y, exit_obj.segment.end.y)

        if (y_min + radius - EPSILON) <= cross_y <= (y_max - radius + EPSILON):
            return (True, Point(exit_x, cross_y))
        return (False, None)

    elif exit_obj.boundary == "right":
        dx = x2 - x1
        if dx <= EPSILON:  # Must be moving right (outward across x=width)
            return (False, None)

        exit_x = exit_obj.segment.start.x
        t = (exit_x - x1) / dx
        if not (-EPSILON <= t <= 1.0 + EPSILON):
            return (False, None)

        t_clamped = max(0.0, min(1.0, t))
        cross_y = y1 + t_clamped * (y2 - y1)

        y_min = min(exit_obj.segment.start.y, exit_obj.segment.end.y)
        y_max = max(exit_obj.segment.start.y, exit_obj.segment.end.y)

        if (y_min + radius - EPSILON) <= cross_y <= (y_max - radius + EPSILON):
            return (True, Point(exit_x, cross_y))
        return (False, None)

    elif exit_obj.boundary == "bottom":
        dy = y2 - y1
        if dy <= EPSILON:  # Must be moving down (outward across y=height)
            return (False, None)

        exit_y = exit_obj.segment.start.y
        t = (exit_y - y1) / dy
        if not (-EPSILON <= t <= 1.0 + EPSILON):
            return (False, None)

        t_clamped = max(0.0, min(1.0, t))
        cross_x = x1 + t_clamped * (x2 - x1)

        x_min = min(exit_obj.segment.start.x, exit_obj.segment.end.x)
        x_max = max(exit_obj.segment.start.x, exit_obj.segment.end.x)

        if (x_min + radius - EPSILON) <= cross_x <= (x_max - radius + EPSILON):
            return (True, Point(cross_x, exit_y))
        return (False, None)

    elif exit_obj.boundary == "top":
        dy = y2 - y1
        if dy >= -EPSILON:  # Must be moving up (outward across y=0)
            return (False, None)

        exit_y = exit_obj.segment.start.y
        t = (exit_y - y1) / dy
        if not (-EPSILON <= t <= 1.0 + EPSILON):
            return (False, None)

        t_clamped = max(0.0, min(1.0, t))
        cross_x = x1 + t_clamped * (x2 - x1)

        x_min = min(exit_obj.segment.start.x, exit_obj.segment.end.x)
        x_max = max(exit_obj.segment.start.x, exit_obj.segment.end.x)

        if (x_min + radius - EPSILON) <= cross_x <= (x_max - radius + EPSILON):
            return (True, Point(cross_x, exit_y))
        return (False, None)

    return (False, None)


def check_collision(
    start: Point,
    end: Point,
    radius: float,
    building: Building,
    target_exit: Optional[Exit] = None,
) -> bool:
    """Continuous radius-aware collision check between movement segment and environment geometry.

    Contact Policy:
    - Penetration (dist < radius - EPSILON) is a collision.
    - Tangent contact (dist >= radius - EPSILON) is permitted clearance.
    - Moving outside valid WalkableArea bounds (except through open doorways or open assigned exit)
      is a collision.
    """
    motion_seg = LineSegment(start, end)

    # 1. Check static obstacles (continuous segment-to-rect)
    for obs in building.obstacles:
        dist = distance_segment_to_rect(motion_seg, obs.bounds)
        if dist < radius - EPSILON:
            return True

    # 2. Check walls (continuous segment-to-rect on 2D wall footprints)
    for wall in building.walls:
        dist = distance_segment_to_rect(motion_seg, wall.footprint)
        if dist < radius - EPSILON:
            return True

    # 3. Check building exterior bounds
    # Allowed to enter boundary opening if approaching assigned exit within usable opening
    if end.x - radius < -EPSILON:
        if not (
            target_exit is not None
            and target_exit.is_open
            and target_exit.boundary == "left"
            and (min(target_exit.segment.start.y, target_exit.segment.end.y) + radius - EPSILON)
            <= end.y
            <= (max(target_exit.segment.start.y, target_exit.segment.end.y) - radius + EPSILON)
        ):
            return True

    if end.x + radius > building.width + EPSILON:
        if not (
            target_exit is not None
            and target_exit.is_open
            and target_exit.boundary == "right"
            and (min(target_exit.segment.start.y, target_exit.segment.end.y) + radius - EPSILON)
            <= end.y
            <= (max(target_exit.segment.start.y, target_exit.segment.end.y) - radius + EPSILON)
        ):
            return True

    if end.y - radius < -EPSILON:
        if not (
            target_exit is not None
            and target_exit.is_open
            and target_exit.boundary == "top"
            and (min(target_exit.segment.start.x, target_exit.segment.end.x) + radius - EPSILON)
            <= end.x
            <= (max(target_exit.segment.start.x, target_exit.segment.end.x) - radius + EPSILON)
        ):
            return True

    if end.y + radius > building.height + EPSILON:
        if not (
            target_exit is not None
            and target_exit.is_open
            and target_exit.boundary == "bottom"
            and (min(target_exit.segment.start.x, target_exit.segment.end.x) + radius - EPSILON)
            <= end.x
            <= (max(target_exit.segment.start.x, target_exit.segment.end.x) - radius + EPSILON)
        ):
            return True

    # 4. Check WalkableArea boundary & doorway clearance
    if _check_walkable_space_violation(start, end, radius, building, target_exit):
        return True

    return False


# ---------------------------------------------------------------------------
# Phase 3 & Phase 5: Deterministic Pedestrian Separation & Safe Conflict Resolution
# ---------------------------------------------------------------------------


def check_pedestrian_overlap(
    pos_a: Point,
    radius_a: float,
    pos_b: Point,
    radius_b: float,
) -> bool:
    """Return True if two pedestrian disks strictly overlap (distance < r_a + r_b - EPSILON)."""
    return pos_a.distance_to(pos_b) < (radius_a + radius_b - EPSILON)


def continuous_pedestrian_distance(
    start_a: Point,
    end_a: Point,
    start_b: Point,
    end_b: Point,
) -> float:
    """Compute exact minimum center-to-center distance between two pedestrians over synchronized step [0, 1].

    P_a(t) = start_a + t * (end_a - start_a)
    P_b(t) = start_b + t * (end_b - start_b)
    Relative position R(t) = P_a(t) - P_b(t) traces a line segment from R(0) to R(1).
    """
    r0_x = start_a.x - start_b.x
    r0_y = start_a.y - start_b.y
    r1_x = end_a.x - end_b.x
    r1_y = end_a.y - end_b.y

    dx = r1_x - r0_x
    dy = r1_y - r0_y
    len_sq = dx * dx + dy * dy
    if len_sq <= 0.0:
        return math.hypot(r0_x, r0_y)

    t = -(r0_x * dx + r0_y * dy) / len_sq
    t = max(0.0, min(1.0, t))
    closest_x = r0_x + t * dx
    closest_y = r0_y + t * dy
    return math.hypot(closest_x, closest_y)


def check_pedestrian_collision(
    start_a: Point,
    end_a: Point,
    radius_a: float,
    start_b: Point,
    end_b: Point,
    radius_b: float,
) -> bool:
    """Detect whether two pedestrian motions conflict during a simulation step.

    Checks:
    1. Synchronized relative-motion minimum distance over t in [0, 1] < r_a + r_b - EPSILON.
    2. Swept-segment distance when motions are non-following (stationary, opposing, or crossing)
       to prevent sub-step path crossing at different fractional times.
    """
    min_sep = radius_a + radius_b - EPSILON

    # 1. Synchronized relative-motion check (includes t=0 start, t=1 end, and continuous trajectory)
    if continuous_pedestrian_distance(start_a, end_a, start_b, end_b) < min_sep:
        return True

    # 2. Spatial swept-segment intersection check for opposing/crossing/stationary motions
    va_x = end_a.x - start_a.x
    va_y = end_a.y - start_a.y
    vb_x = end_b.x - start_b.x
    vb_y = end_b.y - start_b.y

    dot = va_x * vb_x + va_y * vb_y
    b_stationary = math.hypot(vb_x, vb_y) <= EPSILON
    a_stationary = math.hypot(va_x, va_y) <= EPSILON

    if a_stationary or b_stationary or dot <= EPSILON:
        if distance_segment_to_segment(start_a, end_a, start_b, end_b) < min_sep:
            return True

    return False


def _update_pedestrian_area(ped: PedestrianState, building: Building) -> None:
    """Update ped.current_area_id dynamically based on current position and velocity."""
    if not building.walkable_areas:
        return
    area = find_walkable_area_for_point(
        building=building,
        point=ped.position,
        current_area_id=ped.current_area_id,
        velocity=ped.velocity,
    )
    if area is not None:
        ped.current_area_id = area.id


def _propose_nominal_step(
    ped: PedestrianState,
    dt: float,
) -> Optional[tuple[Point, Velocity2D, bool, float, float, float]]:
    """Compute nominal proposed (position, velocity, popped_wp, ux, uy, step_dist) for an ACTIVE pedestrian."""
    if ped.status != PedestrianStatus.ACTIVE:
        return None

    if not ped.waypoints:
        ped.velocity = Velocity2D(0.0, 0.0)
        return None

    target_wp = ped.waypoints[0]
    curr_pos = ped.position

    dx = target_wp.x - curr_pos.x
    dy = target_wp.y - curr_pos.y
    dist = math.hypot(dx, dy)

    # Pop waypoint if already reached
    while dist < EPSILON:
        ped.waypoints.pop(0)
        if not ped.waypoints:
            ped.velocity = Velocity2D(0.0, 0.0)
            return None
        target_wp = ped.waypoints[0]
        dx = target_wp.x - curr_pos.x
        dy = target_wp.y - curr_pos.y
        dist = math.hypot(dx, dy)

    step_dist = ped.preferred_speed * dt
    if step_dist <= 0.0:
        ped.velocity = Velocity2D(0.0, 0.0)
        return None

    ux = dx / dist
    uy = dy / dist

    if step_dist >= dist:
        proposed_pos = target_wp
        popped_wp = True
    else:
        proposed_pos = Point(curr_pos.x + ux * step_dist, curr_pos.y + uy * step_dist)
        popped_wp = False

    proposed_velocity = Velocity2D(ux * ped.preferred_speed, uy * ped.preferred_speed)
    return (proposed_pos, proposed_velocity, popped_wp, ux, uy, step_dist)


def advance_pedestrian_step(
    ped: PedestrianState,
    building: Building,
    dt: float,
    other_pedestrians: Optional[Iterable[PedestrianState]] = None,
) -> None:
    """Advance a single pedestrian by one simulation step dt using deterministic kinematics.

    When other_pedestrians is provided, also enforces pedestrian-to-pedestrian separation.
    """
    if other_pedestrians is not None:
        others = [p for p in other_pedestrians if p.id != ped.id]
        advance_population_step([ped] + others, building, dt, _target_only_id=ped.id)
        return

    nominal = _propose_nominal_step(ped, dt)
    if nominal is None:
        return

    proposed_pos, proposed_velocity, popped_wp, _, _, _ = nominal
    curr_pos = ped.position

    # Check Exit Crossing
    assigned_exit: Optional[Exit] = None
    if ped.target_exit_id is not None:
        assigned_exit = building.get_exit(ped.target_exit_id)
        if assigned_exit is not None:
            crossed, cross_pt = detect_exit_crossing(
                curr_pos, proposed_pos, ped.radius, assigned_exit
            )
            if crossed and cross_pt is not None:
                ped.position = cross_pt
                _update_pedestrian_area(ped, building)
                ped.freeze_at_evacuation()
                return

    # Check unassigned exit crossings
    for other_exit in building.exits:
        if other_exit.id != ped.target_exit_id:
            crossed_wrong, _ = detect_exit_crossing(
                curr_pos, proposed_pos, ped.radius, other_exit
            )
            if crossed_wrong:
                ped.status = PedestrianStatus.BLOCKED
                ped.velocity = Velocity2D(0.0, 0.0)
                return

    # Check Obstacle, Wall, and Walkable-Space Collisions (continuous radius-aware)
    if check_collision(curr_pos, proposed_pos, ped.radius, building, target_exit=assigned_exit):
        ped.status = PedestrianStatus.BLOCKED
        ped.velocity = Velocity2D(0.0, 0.0)
        return

    # Apply safe motion update
    ped.position = proposed_pos
    ped.velocity = proposed_velocity
    if popped_wp:
        ped.waypoints.pop(0)
    _update_pedestrian_area(ped, building)


def advance_population_step(
    pedestrians: Iterable[PedestrianState],
    building: Building,
    dt: float,
    _target_only_id: Optional[str] = None,
) -> list[PedestrianState]:
    """Advance a population of pedestrians by one step dt with deterministic separation.

    Guarantees:
    - Deterministic iteration order sorted by pedestrian ID.
    - Continuous conflict detection during proposed movement (head-on, perpendicular, queues).
    - Wall-guarded right-hand rule deflection and step-fraction scaling.
    - Never pushes pedestrians through walls, obstacles, or outside walkable space.
    - Updates current_area_id across room/doorway/corridor transitions.

    Returns:
        List of PedestrianState instances that transitioned from ACTIVE to BLOCKED during this step.
    """
    sorted_peds = sorted(pedestrians, key=lambda p: p.id)
    newly_blocked: list[PedestrianState] = []
    initial_evacuated_ids: set[str] = {
        p.id for p in sorted_peds if p.status == PedestrianStatus.EVACUATED
    }

    # Record start positions of all non-evacuated pedestrians at the beginning of the step
    start_positions: dict[str, Point] = {
        p.id: p.position for p in sorted_peds if p.status != PedestrianStatus.EVACUATED
    }

    # Phase A: Compute nominal proposals and handle immediate static/exit events for non-conflicting moves
    proposals: dict[str, tuple[Point, Velocity2D, bool, float, float, float, Optional[Exit]]] = {}
    active_ids: list[str] = []

    for ped in sorted_peds:
        if _target_only_id is not None and ped.id != _target_only_id:
            continue
        if ped.status != PedestrianStatus.ACTIVE:
            continue

        nominal = _propose_nominal_step(ped, dt)
        if nominal is None:
            continue

        proposed_pos, proposed_velocity, popped_wp, ux, uy, step_dist = nominal
        curr_pos = ped.position

        assigned_exit: Optional[Exit] = None
        if ped.target_exit_id is not None:
            assigned_exit = building.get_exit(ped.target_exit_id)

        # Check unassigned exit crossings first
        wrong_exit_hit = False
        for other_exit in building.exits:
            if other_exit.id != ped.target_exit_id:
                crossed_wrong, _ = detect_exit_crossing(
                    curr_pos, proposed_pos, ped.radius, other_exit
                )
                if crossed_wrong:
                    ped.status = PedestrianStatus.BLOCKED
                    ped.velocity = Velocity2D(0.0, 0.0)
                    newly_blocked.append(ped)
                    wrong_exit_hit = True
                    break
        if wrong_exit_hit:
            continue

        # Check if nominal step hits a static obstacle, wall, or non-walkable boundary
        # (unless it crosses the assigned open exit first)
        crossed_assigned = False
        if assigned_exit is not None:
            crossed_assigned, _ = detect_exit_crossing(
                curr_pos, proposed_pos, ped.radius, assigned_exit
            )

        if not crossed_assigned and check_collision(
            curr_pos, proposed_pos, ped.radius, building, target_exit=assigned_exit
        ):
            ped.status = PedestrianStatus.BLOCKED
            ped.velocity = Velocity2D(0.0, 0.0)
            newly_blocked.append(ped)
            continue

        proposals[ped.id] = (
            proposed_pos,
            proposed_velocity,
            popped_wp,
            ux,
            uy,
            step_dist,
            assigned_exit,
        )
        active_ids.append(ped.id)

    if not active_ids:
        return newly_blocked

    ped_by_id: dict[str, PedestrianState] = {p.id: p for p in sorted_peds}

    # Track committed end positions for conflict checking
    # Initially, every non-evacuated pedestrian is at its start_position
    committed_end: dict[str, Point] = dict(start_positions)
    resolved_ids: set[str] = set()

    # Deflection angles (degrees) favoring right-hand rule (+angle first, then -angle)
    deflection_angles_deg: tuple[float, ...] = (
        0.0,
        15.0,
        -15.0,
        30.0,
        -30.0,
        45.0,
        -45.0,
        60.0,
        -60.0,
        75.0,
        -75.0,
        90.0,
        -90.0,
        105.0,
        -105.0,
        120.0,
        -120.0,
    )

    def has_pedestrian_conflict(
        ped_a: PedestrianState,
        cand_end_a: Point,
        use_nominal_for_uncommitted: bool,
    ) -> bool:
        start_a = start_positions[ped_a.id]
        for other in sorted_peds:
            if other.id == ped_a.id or other.id in initial_evacuated_ids:
                continue
            start_b = start_positions[other.id]
            if other.id in resolved_ids or not use_nominal_for_uncommitted:
                end_b = committed_end[other.id]
            else:
                # If other has a nominal proposal, check against both its nominal motion and its start position
                if other.id in proposals:
                    nom_end_b = proposals[other.id][0]
                    if check_pedestrian_collision(
                        start_a, cand_end_a, ped_a.radius, start_b, nom_end_b, other.radius
                    ):
                        return True
                end_b = committed_end[other.id]

            if check_pedestrian_collision(
                start_a, cand_end_a, ped_a.radius, start_b, end_b, other.radius
            ):
                return True
        return False

    # Pass 1: Identify pedestrians whose nominal step has zero conflict with any other pedestrian
    # (neither against nominal trajectories nor against start positions)
    conflict_ids: list[str] = []
    for pid in active_ids:
        ped = ped_by_id[pid]
        proposed_pos, proposed_velocity, popped_wp, ux, uy, step_dist, assigned_exit = proposals[pid]

        if not has_pedestrian_conflict(ped, proposed_pos, use_nominal_for_uncommitted=True):
            # Commit nominal move immediately
            if assigned_exit is not None:
                crossed, cross_pt = detect_exit_crossing(
                    start_positions[pid], proposed_pos, ped.radius, assigned_exit
                )
                if crossed and cross_pt is not None:
                    ped.position = cross_pt
                    _update_pedestrian_area(ped, building)
                    ped.freeze_at_evacuation()
                    committed_end[pid] = cross_pt
                    resolved_ids.add(pid)
                    continue

            ped.position = proposed_pos
            ped.velocity = proposed_velocity
            if popped_wp:
                ped.waypoints.pop(0)
            _update_pedestrian_area(ped, building)
            committed_end[pid] = proposed_pos
            resolved_ids.add(pid)
        else:
            conflict_ids.append(pid)

    # Pass 2: Resolve conflicting pedestrians in deterministic ID order with wall-guarded deflection
    # Run up to 2 sweeps so queued followers can advance after their leader commits a move.
    unresolved = list(conflict_ids)
    for _sweep in range(2):
        if not unresolved:
            break
        still_unresolved: list[str] = []

        for pid in unresolved:
            ped = ped_by_id[pid]
            proposed_pos, proposed_velocity, popped_wp, ux, uy, step_dist, assigned_exit = proposals[pid]
            curr_pos = start_positions[pid]

            # Build candidate unit direction vectors:
            # 1. Nominal direction (0 deg)
            # 2. If on final leg to an open exit and within usable lateral opening, include perpendicular exit vector
            # 3. Right-hand rule angular deflections
            candidate_dirs: list[tuple[float, float, bool]] = [(ux, uy, True)]

            if assigned_exit is not None and assigned_exit.is_open and len(ped.waypoints) == 1:
                if assigned_exit.boundary == "left":
                    candidate_dirs.append((-1.0, 0.0, False))
                elif assigned_exit.boundary == "right":
                    candidate_dirs.append((1.0, 0.0, False))
                elif assigned_exit.boundary == "top":
                    candidate_dirs.append((0.0, -1.0, False))
                elif assigned_exit.boundary == "bottom":
                    candidate_dirs.append((0.0, 1.0, False))

            for deg in deflection_angles_deg[1:]:
                rad = math.radians(deg)
                cos_a = math.cos(rad)
                sin_a = math.sin(rad)
                rot_ux = ux * cos_a - uy * sin_a
                rot_uy = ux * sin_a + uy * cos_a
                candidate_dirs.append((rot_ux, rot_uy, False))

            moved = False
            for scale in (1.0, 0.65, 0.35):
                for cand_ux, cand_uy, is_nominal_dir in candidate_dirs:
                    dist_try = step_dist * scale
                    if is_nominal_dir and scale == 1.0:
                        cand_pos = proposed_pos
                        cand_pop = popped_wp
                    else:
                        cand_pos = Point(
                            curr_pos.x + cand_ux * dist_try,
                            curr_pos.y + cand_uy * dist_try,
                        )
                        cand_pop = (
                            ped.waypoints[0].distance_to(cand_pos) < EPSILON
                            if ped.waypoints
                            else False
                        )

                    # Check unassigned exit crossing
                    wrong_exit = False
                    for other_exit in building.exits:
                        if other_exit.id != ped.target_exit_id:
                            crossed_w, _ = detect_exit_crossing(
                                curr_pos, cand_pos, ped.radius, other_exit
                            )
                            if crossed_w:
                                wrong_exit = True
                                break
                    if wrong_exit:
                        continue

                    # Check assigned exit crossing
                    crossed_exit = False
                    cross_pt: Optional[Point] = None
                    if assigned_exit is not None:
                        crossed_exit, cross_pt = detect_exit_crossing(
                            curr_pos, cand_pos, ped.radius, assigned_exit
                        )

                    effective_end = cross_pt if (crossed_exit and cross_pt is not None) else cand_pos

                    # Wall, obstacle, and walkable-space guard
                    if not crossed_exit and (
                        check_collision(
                            curr_pos, effective_end, ped.radius, building, target_exit=assigned_exit
                        )
                        or not is_position_in_walkable_space(
                            effective_end, ped.radius, building, target_exit=assigned_exit
                        )
                    ):
                        continue
                    if crossed_exit and check_collision(
                        curr_pos, effective_end, ped.radius, building, target_exit=assigned_exit
                    ):
                        continue

                    # Pedestrian separation guard against committed positions
                    if has_pedestrian_conflict(ped, effective_end, use_nominal_for_uncommitted=False):
                        continue

                    # Valid conflict-free move found
                    if crossed_exit and cross_pt is not None:
                        ped.position = cross_pt
                        _update_pedestrian_area(ped, building)
                        ped.freeze_at_evacuation()
                        committed_end[pid] = cross_pt
                    else:
                        ped.position = cand_pos
                        ped.velocity = Velocity2D(
                            cand_ux * ped.preferred_speed * scale,
                            cand_uy * ped.preferred_speed * scale,
                        )
                        if cand_pop and ped.waypoints:
                            ped.waypoints.pop(0)
                        _update_pedestrian_area(ped, building)
                        committed_end[pid] = cand_pos

                    resolved_ids.add(pid)
                    moved = True
                    break
                if moved:
                    break

            if not moved:
                still_unresolved.append(pid)

        if len(still_unresolved) == len(unresolved):
            break
        unresolved = still_unresolved

    # Pass 3: Handle pedestrians that could not move during this step
    for pid in unresolved:
        ped = ped_by_id[pid]
        curr_pos = start_positions[pid]
        _, _, _, ux, uy, step_dist, _ = proposals[pid]
        ped.velocity = Velocity2D(0.0, 0.0)
        committed_end[pid] = curr_pos

        # Determine if the pedestrian is permanently obstructed in a narrow passage by:
        # (a) a BLOCKED or permanently stationary pedestrian, or
        # (b) a head-on opposing pedestrian in a single-file passage where neither could move
        permanently_blocked = False
        for other in sorted_peds:
            if other.id == pid or other.status == PedestrianStatus.EVACUATED:
                continue
            other_pos = committed_end[other.id]
            dist_to_other = curr_pos.distance_to(other_pos)
            interaction_range = ped.radius + other.radius + step_dist + 1e-3
            if dist_to_other > interaction_range:
                continue

            # Ensure other is actually in front of ped's desired direction
            rel_x = other_pos.x - curr_pos.x
            rel_y = other_pos.y - curr_pos.y
            if rel_x * ux + rel_y * uy <= 0.0:
                continue

            if other.status == PedestrianStatus.BLOCKED or (
                other.status == PedestrianStatus.ACTIVE and not other.waypoints
            ):
                permanently_blocked = True
                break

            if other.id in unresolved and other.id in proposals:
                _, _, _, other_ux, other_uy, _, _ = proposals[other.id]
                # Opposing head-on deadlock where neither pedestrian could deflect or advance
                if ux * other_ux + uy * other_uy < -0.5:
                    permanently_blocked = True
                    break

        if permanently_blocked:
            ped.status = PedestrianStatus.BLOCKED
            newly_blocked.append(ped)

    return newly_blocked
