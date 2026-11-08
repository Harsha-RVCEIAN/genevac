"""Environment domain models for GenEvac-Sentinel.

Includes:
- WalkableArea
- Doorway
- Obstacle
- Exit
- Building (with comprehensive topological, geometric, and footprint validation)
- Re-exports of Wall, WallType, WallAlignment from geometry
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Optional

from genevac.environment.geometry import (
    LineSegment,
    Point,
    Rectangle,
    Wall,
    WallAlignment,
    WallType,
    segment_intersects_rect,
    segments_overlap,
)

EPSILON = 1e-6

__all__ = [
    "Wall",
    "WallType",
    "WallAlignment",
    "WalkableArea",
    "Doorway",
    "Obstacle",
    "Exit",
    "Building",
]


@dataclass
class WalkableArea:
    """A walkable rectangular zone (room or corridor) in the building."""

    id: str
    bounds: Rectangle
    connected_area_ids: list[str] = field(default_factory=list)
    area_type: str = "room"

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"WalkableArea ID must be a non-empty string, got {self.id!r}")
        if not isinstance(self.bounds, Rectangle):
            raise TypeError(f"bounds must be a Rectangle, got {type(self.bounds)}")
        self.connected_area_ids = list(self.connected_area_ids)


@dataclass
class Doorway:
    """An explicit portal/doorway connecting two walkable areas."""

    id: str
    connected_area_ids: tuple[str, str]
    segment: LineSegment
    width: float

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Doorway ID must be a non-empty string, got {self.id!r}")
        if not isinstance(self.connected_area_ids, (tuple, list)) or len(self.connected_area_ids) != 2:
            raise ValueError(
                f"Doorway connected_area_ids must contain exactly 2 area IDs, got {self.connected_area_ids!r}"
            )
        self.connected_area_ids = (str(self.connected_area_ids[0]), str(self.connected_area_ids[1]))
        if self.connected_area_ids[0] == self.connected_area_ids[1]:
            raise ValueError(f"Doorway '{self.id}' cannot connect an area to itself: {self.connected_area_ids[0]}")
        if not isinstance(self.segment, LineSegment):
            raise TypeError(f"Doorway segment must be a LineSegment, got {type(self.segment)}")
        self.width = float(self.width)
        if self.width <= 0:
            raise ValueError(f"Doorway width must be strictly positive (> 0), got {self.width}")

    @property
    def area_a_id(self) -> str:
        """First connected walkable area ID."""
        return self.connected_area_ids[0]

    @property
    def area_b_id(self) -> str:
        """Second connected walkable area ID."""
        return self.connected_area_ids[1]

    def connects(self, area_1: str, area_2: str) -> bool:
        """Check if this doorway connects the specified two areas in either direction."""
        return (self.connected_area_ids == (area_1, area_2)) or (self.connected_area_ids == (area_2, area_1))


@dataclass
class Obstacle:
    """A static rectangular obstacle within the building."""

    id: str
    bounds: Rectangle

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Obstacle ID must be a non-empty string, got {self.id!r}")
        if not isinstance(self.bounds, Rectangle):
            raise TypeError(f"bounds must be a Rectangle, got {type(self.bounds)}")

    @property
    def x(self) -> float:
        """Top-left X coordinate in meters."""
        return self.bounds.x

    @property
    def y(self) -> float:
        """Top-left Y coordinate in meters."""
        return self.bounds.y

    @property
    def width(self) -> float:
        """Width in meters."""
        return self.bounds.width

    @property
    def height(self) -> float:
        """Height in meters."""
        return self.bounds.height


@dataclass
class Exit:
    """An exterior boundary exit opening connected to an accessible walkable area."""

    id: str
    segment: LineSegment
    width: float
    boundary: str  # "left", "right", "bottom", "top"
    connected_area_id: str
    is_open: bool = True

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Exit ID must be a non-empty string, got {self.id!r}")
        if not isinstance(self.segment, LineSegment):
            raise TypeError(f"segment must be a LineSegment, got {type(self.segment)}")
        self.width = float(self.width)
        if self.width <= 0:
            raise ValueError(f"Exit width must be strictly positive (> 0), got {self.width}")
        if not self.connected_area_id or not isinstance(self.connected_area_id, str):
            raise ValueError(f"connected_area_id must be a non-empty string, got {self.connected_area_id!r}")
        if not self.boundary or not isinstance(self.boundary, str):
            raise ValueError(f"boundary must be a non-empty string, got {self.boundary!r}")
        self.is_open = bool(self.is_open)

    def close(self) -> None:
        """Mark this exit as closed."""
        self.is_open = False

    def open(self) -> None:
        """Mark this exit as open."""
        self.is_open = True


@dataclass
class Building:
    """Top-level environment representation containing all building components."""

    id: str
    width: float
    height: float
    walkable_areas: list[WalkableArea] = field(default_factory=list)
    walls: list[Wall] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    exits: list[Exit] = field(default_factory=list)
    doorways: list[Doorway] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Building ID must be a non-empty string, got {self.id!r}")
        self.width = float(self.width)
        self.height = float(self.height)
        if self.width <= 0:
            raise ValueError(f"Building width must be strictly positive (> 0), got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Building height must be strictly positive (> 0), got {self.height}")

        self.validate()

    @property
    def exterior_boundary(self) -> Rectangle:
        """Bounding rectangle of the entire building."""
        return Rectangle(0.0, 0.0, self.width, self.height)

    def get_walkable_area(self, area_id: str) -> Optional[WalkableArea]:
        """Look up a walkable area by its ID."""
        for area in self.walkable_areas:
            if area.id == area_id:
                return area
        return None

    def get_doorway(self, doorway_id: str) -> Optional[Doorway]:
        """Look up a doorway by its ID."""
        for door in self.doorways:
            if door.id == doorway_id:
                return door
        return None

    def get_exit(self, exit_id: str) -> Optional[Exit]:
        """Look up an exit by its ID."""
        for ex in self.exits:
            if ex.id == exit_id:
                return ex
        return None

    def get_wall(self, wall_id: str) -> Optional[Wall]:
        """Look up a wall by its ID."""
        for wall in self.walls:
            if wall.id == wall_id:
                return wall
        return None

    def get_obstacle(self, obstacle_id: str) -> Optional[Obstacle]:
        """Look up an obstacle by its ID."""
        for obs in self.obstacles:
            if obs.id == obstacle_id:
                return obs
        return None

    def get_exterior_walls(self) -> list[Wall]:
        """Return all exterior walls."""
        return [w for w in self.walls if w.wall_type == WallType.EXTERIOR]

    def get_interior_walls(self) -> list[Wall]:
        """Return all interior walls."""
        return [w for w in self.walls if w.wall_type == WallType.INTERIOR]

    def validate(self) -> None:
        """Perform comprehensive validation of all building components and relationships."""
        # 1. Unique IDs check across all component categories
        all_ids: set[str] = set()
        for category_name, collection in [
            ("walkable_areas", self.walkable_areas),
            ("walls", self.walls),
            ("obstacles", self.obstacles),
            ("exits", self.exits),
            ("doorways", self.doorways),
        ]:
            category_ids: set[str] = set()
            for item in collection:
                if item.id in category_ids:
                    raise ValueError(f"Duplicate ID '{item.id}' found in {category_name}")
                if item.id in all_ids:
                    raise ValueError(f"Duplicate ID '{item.id}' reused across building components")
                category_ids.add(item.id)
                all_ids.add(item.id)

        # 2. Walkable areas bounds and reciprocal connectivity check
        known_area_ids = {a.id for a in self.walkable_areas}
        area_map = {a.id: a for a in self.walkable_areas}

        for area in self.walkable_areas:
            if (
                area.bounds.left < -EPSILON
                or area.bounds.right > self.width + EPSILON
                or area.bounds.top < -EPSILON
                or area.bounds.bottom > self.height + EPSILON
            ):
                raise ValueError(
                    f"Walkable area '{area.id}' extends outside building bounds "
                    f"[0.0, {self.width}] x [0.0, {self.height}]"
                )

            # Check connectivity references
            for target_id in area.connected_area_ids:
                if target_id == area.id:
                    raise ValueError(f"Walkable area '{area.id}' cannot reference itself in connected_area_ids")
                if target_id not in known_area_ids:
                    raise ValueError(
                        f"Walkable area '{area.id}' references non-existent connected area '{target_id}'"
                    )
                # Reciprocal connectivity verification
                target_area = area_map[target_id]
                if area.id not in target_area.connected_area_ids:
                    raise ValueError(
                        f"Connectivity between '{area.id}' and '{target_id}' is not reciprocal: "
                        f"'{target_id}' does not list '{area.id}'"
                    )

        # 3. Complete walkable-area graph connectivity check (BFS)
        if self.walkable_areas:
            start_id = self.walkable_areas[0].id
            visited: set[str] = {start_id}
            queue = deque([start_id])
            while queue:
                curr_id = queue.popleft()
                for neighbor_id in area_map[curr_id].connected_area_ids:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)

            if len(visited) < len(self.walkable_areas):
                unreachable = sorted(list(known_area_ids - visited))
                raise ValueError(
                    f"Walkable area network is disconnected: unreachable areas {unreachable}"
                )

        # 4. Walls bounds and physical footprint check
        for wall in self.walls:
            for pt_name, pt in [("start", wall.start), ("end", wall.end)]:
                if (
                    pt.x < -EPSILON
                    or pt.x > self.width + EPSILON
                    or pt.y < -EPSILON
                    or pt.y > self.height + EPSILON
                ):
                    raise ValueError(
                        f"Wall '{wall.id}' {pt_name} point ({pt.x}, {pt.y}) "
                        f"extends outside building bounds [0.0, {self.width}] x [0.0, {self.height}]"
                    )

            # Verify physical footprint enclosed within building boundaries
            fp = wall.footprint
            if (
                fp.left < -EPSILON
                or fp.right > self.width + EPSILON
                or fp.top < -EPSILON
                or fp.bottom > self.height + EPSILON
            ):
                raise ValueError(
                    f"Wall '{wall.id}' physical footprint [{fp.left:.2f}, {fp.right:.2f}] x "
                    f"[{fp.top:.2f}, {fp.bottom:.2f}] extends outside building bounds [0.0, {self.width}] x [0.0, {self.height}]"
                )

        # 5. Obstacles bounds check
        for obs in self.obstacles:
            if (
                obs.bounds.left < -EPSILON
                or obs.bounds.right > self.width + EPSILON
                or obs.bounds.top < -EPSILON
                or obs.bounds.bottom > self.height + EPSILON
            ):
                raise ValueError(
                    f"Obstacle '{obs.id}' extends outside building bounds "
                    f"[0.0, {self.width}] x [0.0, {self.height}]"
                )

        # Helper for point in rectangle with tolerance
        def _rect_contains_point(rect: Rectangle, pt: Point) -> bool:
            return (
                rect.left - EPSILON <= pt.x <= rect.right + EPSILON
                and rect.top - EPSILON <= pt.y <= rect.bottom + EPSILON
            )

        # 6. Doorways validation
        for door in self.doorways:
            for pt_name, pt in [("start", door.segment.start), ("end", door.segment.end)]:
                if (
                    pt.x < -EPSILON
                    or pt.x > self.width + EPSILON
                    or pt.y < -EPSILON
                    or pt.y > self.height + EPSILON
                ):
                    raise ValueError(
                        f"Doorway '{door.id}' {pt_name} point ({pt.x}, {pt.y}) extends outside building bounds"
                    )

            # Check connected areas exist
            if door.area_a_id not in known_area_ids:
                raise ValueError(f"Doorway '{door.id}' references non-existent area '{door.area_a_id}'")
            if door.area_b_id not in known_area_ids:
                raise ValueError(f"Doorway '{door.id}' references non-existent area '{door.area_b_id}'")

            # Verify declared width matches segment length
            if not math.isclose(door.segment.length, door.width, abs_tol=EPSILON):
                raise ValueError(
                    f"Doorway '{door.id}' segment length ({door.segment.length:.3f} m) "
                    f"does not match declared width ({door.width:.3f} m)"
                )

            # Check doorway crossing segment is geometrically on the boundary of both connected areas
            area_a = area_map[door.area_a_id]
            area_b = area_map[door.area_b_id]
            if not (
                _rect_contains_point(area_a.bounds, door.segment.start)
                and _rect_contains_point(area_a.bounds, door.segment.end)
            ):
                raise ValueError(
                    f"Doorway '{door.id}' crossing segment is not on the boundary of '{door.area_a_id}'"
                )
            if not (
                _rect_contains_point(area_b.bounds, door.segment.start)
                and _rect_contains_point(area_b.bounds, door.segment.end)
            ):
                raise ValueError(
                    f"Doorway '{door.id}' crossing segment is not on the boundary of '{door.area_b_id}'"
                )

            # Check doorway areas are mutually declared in adjacency relationships
            if door.area_b_id not in area_map[door.area_a_id].connected_area_ids:
                raise ValueError(
                    f"Doorway '{door.id}' connects '{door.area_a_id}' and '{door.area_b_id}', "
                    f"but '{door.area_b_id}' is not in '{door.area_a_id}'.connected_area_ids"
                )

            # Ensure doorway is not blocked by a solid wall (checks both centerline and 2D footprint thickness)
            for wall in self.walls:
                if segments_overlap(wall.segment, door.segment) or segment_intersects_rect(
                    door.segment, wall.footprint
                ):
                    raise ValueError(
                        f"Doorway '{door.id}' opening is covered by solid wall '{wall.id}'"
                    )

            # Ensure doorway is not blocked by an obstacle
            for obs in self.obstacles:
                if segment_intersects_rect(door.segment, obs.bounds):
                    raise ValueError(
                        f"Doorway '{door.id}' opening is blocked by obstacle '{obs.id}'"
                    )

        # 7. Exits validation
        for ex in self.exits:
            for pt_name, pt in [("start", ex.segment.start), ("end", ex.segment.end)]:
                if (
                    pt.x < -EPSILON
                    or pt.x > self.width + EPSILON
                    or pt.y < -EPSILON
                    or pt.y > self.height + EPSILON
                ):
                    raise ValueError(
                        f"Exit '{ex.id}' {pt_name} point ({pt.x}, {pt.y}) extends outside building bounds"
                    )

            # Check connected area exists
            if ex.connected_area_id not in known_area_ids:
                raise ValueError(
                    f"Exit '{ex.id}' references non-existent walkable area '{ex.connected_area_id}'"
                )

            # Verify declared width matches segment length
            if not math.isclose(ex.segment.length, ex.width, abs_tol=EPSILON):
                raise ValueError(
                    f"Exit '{ex.id}' segment length ({ex.segment.length:.3f} m) "
                    f"does not match declared width ({ex.width:.3f} m)"
                )

            # Verify exit opening is on the boundary of its connected walkable area
            conn_area = area_map[ex.connected_area_id]
            if not (
                _rect_contains_point(conn_area.bounds, ex.segment.start)
                and _rect_contains_point(conn_area.bounds, ex.segment.end)
            ):
                raise ValueError(
                    f"Exit '{ex.id}' opening is not along the boundary of associated walkable area '{ex.connected_area_id}'"
                )

            # Check boundary alignment
            if ex.boundary == "left":
                if abs(ex.segment.start.x) > EPSILON or abs(ex.segment.end.x) > EPSILON:
                    raise ValueError(f"Exit '{ex.id}' specified on 'left' boundary but x != 0.0")
            elif ex.boundary == "right":
                if abs(ex.segment.start.x - self.width) > EPSILON or abs(ex.segment.end.x - self.width) > EPSILON:
                    raise ValueError(f"Exit '{ex.id}' specified on 'right' boundary but x != {self.width}")
            elif ex.boundary == "top":
                if abs(ex.segment.start.y) > EPSILON or abs(ex.segment.end.y) > EPSILON:
                    raise ValueError(f"Exit '{ex.id}' specified on 'top' boundary but y != 0.0")
            elif ex.boundary == "bottom":
                if abs(ex.segment.start.y - self.height) > EPSILON or abs(ex.segment.end.y - self.height) > EPSILON:
                    raise ValueError(f"Exit '{ex.id}' specified on 'bottom' boundary but y != {self.height}")

            # Ensure exit is not blocked by a solid wall (checks both centerline and 2D footprint thickness)
            for wall in self.walls:
                if segments_overlap(wall.segment, ex.segment) or segment_intersects_rect(
                    ex.segment, wall.footprint
                ):
                    raise ValueError(
                        f"Exit '{ex.id}' opening is covered by solid wall '{wall.id}'"
                    )

            # Ensure exit is not blocked by an obstacle
            for obs in self.obstacles:
                if segment_intersects_rect(ex.segment, obs.bounds):
                    raise ValueError(
                        f"Exit '{ex.id}' opening is blocked by obstacle '{obs.id}'"
                    )
