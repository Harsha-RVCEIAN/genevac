"""Deterministic movement, routing, and continuous collision handling for GenEvac-Sentinel.

Provides:
- Exact continuous segment-to-rectangle and segment-to-segment distance calculations
- Deterministic topological route generation with obstacle-clearance waypoints
- Radius-aware collision checking against walls, static obstacles, and building bounds
- Exit boundary crossing detection within usable opening clearances
- Step advancement for active pedestrians
"""

from __future__ import annotations

import math
from typing import Optional, Union

from genevac.environment.geometry import LineSegment, Point, Rectangle
from genevac.environment.models import Building, Doorway, Exit, WalkableArea
from genevac.pedestrian.models import PedestrianState, PedestrianStatus, Velocity2D

EPSILON: float = 1e-6


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


def _get_room_obstacle_bypass(
    room_id: str,
    start_pos: Point,
    target_pt: Point,
    building: Building,
    radius: float = 0.25,
) -> list[Point]:
    """Generate intermediate clearance waypoints if start_pos to target_pt is occluded by room obstacle."""
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


def generate_deterministic_route(
    building: Building,
    start_pos: Point,
    start_area_id: str,
    target_exit_id: str,
    radius: float = 0.25,
) -> list[Point]:
    """Generate a deterministic waypoint route from a walkable area to an assigned exit.

    Uses actual building doorway portals, doorway standoff waypoints, corridor centerline topology,
    and obstacle clearance to ensure routes do not intersect solid walls or interior obstacles.

    Raises:
        ValueError: If start_area_id or target_exit_id is unknown, or if unreachable.
    """
    area = building.get_walkable_area(start_area_id)
    if area is None:
        raise ValueError(f"Unknown start walkable area: {start_area_id!r}")

    target_exit = building.get_exit(target_exit_id)
    if target_exit is None:
        raise ValueError(f"Unknown target exit: {target_exit_id!r}")

    # Case A: Pedestrian is already in the area directly connected to the target exit
    if start_area_id == target_exit.connected_area_id:
        exit_pt = get_exit_midpoint(target_exit)
        if target_exit_id == "exit_bottom":
            exit_approach = Point(10.0, 28.5)
            bypass = _get_room_obstacle_bypass(start_area_id, start_pos, exit_approach, building, radius)
            return bypass + [exit_approach, exit_pt]
        bypass = _get_room_obstacle_bypass(start_area_id, start_pos, exit_pt, building, radius)
        return bypass + [exit_pt]

    # Doorway lookup helper
    def find_doorway(area_1: str, area_2: str) -> Optional[Doorway]:
        for d in building.doorways:
            if d.connects(area_1, area_2):
                return d
        return None

    # Case B: Standard 40m x 30m layout with main_corridor spine
    if target_exit_id in ("exit_left", "exit_right"):
        exit_pt = get_exit_midpoint(target_exit)
        if start_area_id in ("room_1", "room_2"):
            door = find_doorway(start_area_id, "main_corridor")
            if door is None:
                raise ValueError(f"No doorway connecting {start_area_id} to main_corridor")
            door_pt = get_doorway_midpoint(door)
            approach_pt = Point(door_pt.x, 12.0)
            corridor_pt = Point(door_pt.x, 15.0)
            bypass = _get_room_obstacle_bypass(start_area_id, start_pos, approach_pt, building, radius)
            return bypass + [approach_pt, door_pt, corridor_pt, exit_pt]
        elif start_area_id in ("room_3", "room_4"):
            door = find_doorway(start_area_id, "main_corridor")
            if door is None:
                raise ValueError(f"No doorway connecting {start_area_id} to main_corridor")
            door_pt = get_doorway_midpoint(door)
            approach_pt = Point(door_pt.x, 18.0)
            corridor_pt = Point(door_pt.x, 15.0)
            bypass = _get_room_obstacle_bypass(start_area_id, start_pos, approach_pt, building, radius)
            return bypass + [approach_pt, door_pt, corridor_pt, exit_pt]
        elif start_area_id == "main_corridor":
            return [exit_pt]

    elif target_exit_id == "exit_bottom":
        exit_pt = get_exit_midpoint(target_exit)  # (10.0, 30.0)
        door_r3 = find_doorway("room_3", "main_corridor")
        if door_r3 is None:
            raise ValueError("No doorway connecting room_3 to main_corridor")
        door_r3_pt = get_doorway_midpoint(door_r3)  # (10.0, 16.5)
        door_r3_approach = Point(10.0, 18.0)
        exit_r3_approach = Point(10.0, 28.5)

        if start_area_id == "room_3":
            bypass = _get_room_obstacle_bypass("room_3", start_pos, exit_r3_approach, building, radius)
            return bypass + [exit_r3_approach, exit_pt]
        elif start_area_id in ("room_1", "room_2"):
            door = find_doorway(start_area_id, "main_corridor")
            if door is None:
                raise ValueError(f"No doorway connecting {start_area_id} to main_corridor")
            door_pt = get_doorway_midpoint(door)
            approach_pt = Point(door_pt.x, 12.0)
            corridor_entry = Point(door_pt.x, 15.0)
            corridor_turn = Point(10.0, 15.0)
            bypass = _get_room_obstacle_bypass(start_area_id, start_pos, approach_pt, building, radius)
            return bypass + [approach_pt, door_pt, corridor_entry, corridor_turn, door_r3_pt, door_r3_approach, exit_r3_approach, exit_pt]
        elif start_area_id == "room_4":
            door = find_doorway("room_4", "main_corridor")
            if door is None:
                raise ValueError(f"No doorway connecting {start_area_id} to main_corridor")
            door_pt = get_doorway_midpoint(door)
            approach_pt = Point(door_pt.x, 18.0)
            corridor_entry = Point(door_pt.x, 15.0)
            corridor_turn = Point(10.0, 15.0)
            bypass = _get_room_obstacle_bypass(start_area_id, start_pos, approach_pt, building, radius)
            return bypass + [approach_pt, door_pt, corridor_entry, corridor_turn, door_r3_pt, door_r3_approach, exit_r3_approach, exit_pt]
        elif start_area_id == "main_corridor":
            corridor_turn = Point(10.0, 15.0)
            return [corridor_turn, door_r3_pt, door_r3_approach, exit_r3_approach, exit_pt]

    # Case C: Generic adjacent-area connection (e.g. 2-room custom test building)
    direct_door = find_doorway(start_area_id, target_exit.connected_area_id)
    if direct_door is not None:
        door_pt = get_doorway_midpoint(direct_door)
        exit_pt = get_exit_midpoint(target_exit)
        return [door_pt, exit_pt]

    raise ValueError(
        f"Cannot generate route: area {start_area_id!r} cannot reach exit {target_exit_id!r} "
        f"in building {building.id!r}"
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
            and target_exit.boundary == "left"
            and (min(target_exit.segment.start.y, target_exit.segment.end.y) + radius - EPSILON)
            <= end.y
            <= (max(target_exit.segment.start.y, target_exit.segment.end.y) - radius + EPSILON)
        ):
            return True

    if end.x + radius > building.width + EPSILON:
        if not (
            target_exit is not None
            and target_exit.boundary == "right"
            and (min(target_exit.segment.start.y, target_exit.segment.end.y) + radius - EPSILON)
            <= end.y
            <= (max(target_exit.segment.start.y, target_exit.segment.end.y) - radius + EPSILON)
        ):
            return True

    if end.y - radius < -EPSILON:
        if not (
            target_exit is not None
            and target_exit.boundary == "top"
            and (min(target_exit.segment.start.x, target_exit.segment.end.x) + radius - EPSILON)
            <= end.x
            <= (max(target_exit.segment.start.x, target_exit.segment.end.x) - radius + EPSILON)
        ):
            return True

    if end.y + radius > building.height + EPSILON:
        if not (
            target_exit is not None
            and target_exit.boundary == "bottom"
            and (min(target_exit.segment.start.x, target_exit.segment.end.x) + radius - EPSILON)
            <= end.x
            <= (max(target_exit.segment.start.x, target_exit.segment.end.x) - radius + EPSILON)
        ):
            return True

    return False


def advance_pedestrian_step(
    ped: PedestrianState,
    building: Building,
    dt: float,
) -> None:
    """Advance a pedestrian by one simulation step dt using deterministic kinematics."""
    if ped.status != PedestrianStatus.ACTIVE:
        return

    if not ped.waypoints:
        ped.velocity = Velocity2D(0.0, 0.0)
        return

    target_wp = ped.waypoints[0]
    curr_pos = ped.position

    dx = target_wp.x - curr_pos.x
    dy = target_wp.y - curr_pos.y
    dist = math.hypot(dx, dy)

    # Pop waypoint if already reached
    if dist < EPSILON:
        ped.waypoints.pop(0)
        if not ped.waypoints:
            ped.velocity = Velocity2D(0.0, 0.0)
            return
        target_wp = ped.waypoints[0]
        dx = target_wp.x - curr_pos.x
        dy = target_wp.y - curr_pos.y
        dist = math.hypot(dx, dy)

    step_dist = ped.preferred_speed * dt
    if step_dist <= 0.0:
        ped.velocity = Velocity2D(0.0, 0.0)
        return

    ux = dx / dist
    uy = dy / dist

    if step_dist >= dist:
        proposed_pos = target_wp
        popped_wp = True
    else:
        proposed_pos = Point(curr_pos.x + ux * step_dist, curr_pos.y + uy * step_dist)
        popped_wp = False

    proposed_velocity = Velocity2D(ux * ped.preferred_speed, uy * ped.preferred_speed)

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

    # Check Obstacle and Wall Collisions (continuous radius-aware)
    if check_collision(curr_pos, proposed_pos, ped.radius, building, target_exit=assigned_exit):
        ped.status = PedestrianStatus.BLOCKED
        ped.velocity = Velocity2D(0.0, 0.0)
        return

    # Apply safe motion update
    ped.position = proposed_pos
    ped.velocity = proposed_velocity
    if popped_wp:
        ped.waypoints.pop(0)
