"""Geometry primitives for the GenEvac-Sentinel environment model.

Coordinate convention:
- Top-left origin: (0.0, 0.0)
- X increases to the right
- Y increases downward
- Units: meters
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class WallType(str, Enum):
    """Classification of building walls."""

    EXTERIOR = "exterior"
    INTERIOR = "interior"


class WallAlignment(str, Enum):
    """Alignment/extrusion direction of wall thickness relative to its centerline segment.

    - CENTER: thickness expands symmetrically by thickness/2 on both sides.
    - POSITIVE: thickness expands in positive coordinate direction (+X for vertical, +Y for horizontal).
    - NEGATIVE: thickness expands in negative coordinate direction (-X for vertical, -Y for horizontal).
    """

    CENTER = "center"
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True)
class Point:
    """A 2D point with floating-point coordinates in meters."""

    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))

    def distance_to(self, other: Point) -> float:
        """Euclidean distance to another point in meters."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_tuple(self) -> tuple[float, float]:
        """Return (x, y) coordinates as a tuple."""
        return (self.x, self.y)


@dataclass(frozen=True)
class Rectangle:
    """An axis-aligned rectangle defined by its top-left corner and dimensions in meters."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))
        object.__setattr__(self, "width", float(self.width))
        object.__setattr__(self, "height", float(self.height))

        if self.width <= 0:
            raise ValueError(f"Rectangle width must be strictly positive (> 0), got {self.width}")
        if self.height <= 0:
            raise ValueError(f"Rectangle height must be strictly positive (> 0), got {self.height}")

    @property
    def left(self) -> float:
        """Left edge X coordinate."""
        return self.x

    @property
    def right(self) -> float:
        """Right edge X coordinate."""
        return self.x + self.width

    @property
    def top(self) -> float:
        """Top edge Y coordinate."""
        return self.y

    @property
    def bottom(self) -> float:
        """Bottom edge Y coordinate."""
        return self.y + self.height

    def contains_point(self, point: Point) -> bool:
        """Check if a point lies within or on the rectangle boundary."""
        return self.left <= point.x <= self.right and self.top <= point.y <= self.bottom

    def contains_rect(self, other: Rectangle) -> bool:
        """Check if another rectangle is fully enclosed by this rectangle."""
        return (
            self.left <= other.left
            and other.right <= self.right
            and self.top <= other.top
            and other.bottom <= self.bottom
        )

    def intersects(self, other: Rectangle) -> bool:
        """Check if this rectangle overlaps with another rectangle."""
        return not (
            self.right <= other.left
            or self.left >= other.right
            or self.bottom <= other.top
            or self.top >= other.bottom
        )


@dataclass(frozen=True)
class LineSegment:
    """A line segment defined by two endpoints in 2D space."""

    start: Point
    end: Point

    def __post_init__(self) -> None:
        if not isinstance(self.start, Point):
            raise TypeError(f"LineSegment start must be a Point, got {type(self.start)}")
        if not isinstance(self.end, Point):
            raise TypeError(f"LineSegment end must be a Point, got {type(self.end)}")

    @property
    def length(self) -> float:
        """Length of the segment in meters."""
        return self.start.distance_to(self.end)

    def as_tuple(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ((x1, y1), (x2, y2)) coordinates."""
        return (self.start.as_tuple(), self.end.as_tuple())


@dataclass
class Wall:
    """A physical wall with linear geometry, thickness in meters, classification, and alignment."""

    id: str
    start: Point
    end: Point
    thickness: float = 0.2
    wall_type: WallType = WallType.INTERIOR
    alignment: WallAlignment = WallAlignment.CENTER

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Wall ID must be a non-empty string, got {self.id!r}")

        if isinstance(self.start, (tuple, list)):
            self.start = Point(self.start[0], self.start[1])
        elif not isinstance(self.start, Point):
            raise TypeError(f"Wall start must be a Point, got {type(self.start)}")

        if isinstance(self.end, (tuple, list)):
            self.end = Point(self.end[0], self.end[1])
        elif not isinstance(self.end, Point):
            raise TypeError(f"Wall end must be a Point, got {type(self.end)}")

        self.thickness = float(self.thickness)
        if self.thickness <= 0:
            raise ValueError(f"Wall thickness must be strictly positive (> 0), got {self.thickness}")

        if isinstance(self.wall_type, str) and not isinstance(self.wall_type, WallType):
            try:
                self.wall_type = WallType(self.wall_type)
            except ValueError:
                raise ValueError(
                    f"Invalid wall type: {self.wall_type!r}. Must be 'exterior' or 'interior'"
                )

        if isinstance(self.alignment, str) and not isinstance(self.alignment, WallAlignment):
            try:
                self.alignment = WallAlignment(self.alignment)
            except ValueError:
                raise ValueError(
                    f"Invalid wall alignment: {self.alignment!r}. Must be 'center', 'positive', or 'negative'"
                )

    @property
    def segment(self) -> LineSegment:
        """Return the underlying line segment."""
        return LineSegment(self.start, self.end)

    @property
    def length(self) -> float:
        """Length of the wall in meters."""
        return self.start.distance_to(self.end)

    @property
    def footprint(self) -> Rectangle:
        """Calculate the 2D bounding rectangle footprint of the physical wall."""
        t = self.thickness
        x_min = min(self.start.x, self.end.x)
        x_max = max(self.start.x, self.end.x)
        y_min = min(self.start.y, self.end.y)
        y_max = max(self.start.y, self.end.y)

        # Horizontal wall
        if math.isclose(y_min, y_max, abs_tol=1e-6):
            width = max(x_max - x_min, 1e-6)
            if self.alignment == WallAlignment.POSITIVE:
                rect_y = y_min
            elif self.alignment == WallAlignment.NEGATIVE:
                rect_y = y_min - t
            else:  # CENTER
                rect_y = y_min - t / 2.0
            return Rectangle(x=x_min, y=rect_y, width=width, height=t)

        # Vertical wall
        elif math.isclose(x_min, x_max, abs_tol=1e-6):
            height = max(y_max - y_min, 1e-6)
            if self.alignment == WallAlignment.POSITIVE:
                rect_x = x_min
            elif self.alignment == WallAlignment.NEGATIVE:
                rect_x = x_min - t
            else:  # CENTER
                rect_x = x_min - t / 2.0
            return Rectangle(x=rect_x, y=y_min, width=t, height=height)

        # General diagonal wall bounding box
        else:
            dx = x_max - x_min
            dy = y_max - y_min
            pad = t / 2.0
            return Rectangle(x=x_min - pad, y=y_min - pad, width=dx + t, height=dy + t)


def segments_overlap(s1: LineSegment, s2: LineSegment, eps: float = 1e-6) -> bool:
    """Check if two collinear axis-aligned line segments strictly overlap (more than point contact)."""
    # Horizontal check
    if (
        math.isclose(s1.start.y, s1.end.y, abs_tol=eps)
        and math.isclose(s2.start.y, s2.end.y, abs_tol=eps)
        and math.isclose(s1.start.y, s2.start.y, abs_tol=eps)
    ):
        s1_min_x = min(s1.start.x, s1.end.x)
        s1_max_x = max(s1.start.x, s1.end.x)
        s2_min_x = min(s2.start.x, s2.end.x)
        s2_max_x = max(s2.start.x, s2.end.x)
        return max(s1_min_x, s2_min_x) < min(s1_max_x, s2_max_x) - eps

    # Vertical check
    if (
        math.isclose(s1.start.x, s1.end.x, abs_tol=eps)
        and math.isclose(s2.start.x, s2.end.x, abs_tol=eps)
        and math.isclose(s1.start.x, s2.start.x, abs_tol=eps)
    ):
        s1_min_y = min(s1.start.y, s1.end.y)
        s1_max_y = max(s1.start.y, s1.end.y)
        s2_min_y = min(s2.start.y, s2.end.y)
        s2_max_y = max(s2.start.y, s2.end.y)
        return max(s1_min_y, s2_min_y) < min(s1_max_y, s2_max_y) - eps

    return False


def segment_intersects_rect(seg: LineSegment, rect: Rectangle, eps: float = 1e-6) -> bool:
    """Check if an axis-aligned line segment intersects or blocks a rectangle (including on edges)."""
    # Horizontal segment
    if math.isclose(seg.start.y, seg.end.y, abs_tol=eps):
        y = seg.start.y
        if rect.top - eps <= y <= rect.bottom + eps:
            seg_min_x = min(seg.start.x, seg.end.x)
            seg_max_x = max(seg.start.x, seg.end.x)
            return max(rect.left, seg_min_x) < min(rect.right, seg_max_x) - eps

    # Vertical segment
    if math.isclose(seg.start.x, seg.end.x, abs_tol=eps):
        x = seg.start.x
        if rect.left - eps <= x <= rect.right + eps:
            seg_min_y = min(seg.start.y, seg.end.y)
            seg_max_y = max(seg.start.y, seg.end.y)
            return max(rect.top, seg_min_y) < min(rect.bottom, seg_max_y) - eps

    return False
