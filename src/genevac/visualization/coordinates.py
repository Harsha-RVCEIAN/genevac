"""Coordinate conversion module for GenEvac-Sentinel visualization.

Converts 2D world coordinates (metres, top-left origin, X right, Y down)
to screen pixel coordinates while strictly preserving aspect ratio, centering
the building within the available viewport, and applying uniform margins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from genevac.environment.geometry import Point, Rectangle


@dataclass
class Viewport:
    """Manages world-to-screen coordinate mapping with aspect ratio preservation and centering."""

    world_width: float
    world_height: float
    screen_width: int
    screen_height: int
    margin: float = 50.0
    top_offset: float = 60.0  # Room for title bar / header info
    bottom_offset: float = 50.0  # Room for bottom legend

    def __post_init__(self) -> None:
        if self.world_width <= 0 or self.world_height <= 0:
            raise ValueError("World dimensions must be strictly positive")
        self.recalculate()

    def recalculate(self) -> None:
        """Recalculate scale factor and centering offsets based on screen dimensions."""
        avail_w = max(10.0, float(self.screen_width) - 2.0 * self.margin)
        avail_h = max(
            10.0,
            float(self.screen_height) - self.margin - self.top_offset - self.bottom_offset,
        )

        scale_x = avail_w / self.world_width
        scale_y = avail_h / self.world_height

        # Strict uniform scale factor preserves world aspect ratio without distortion
        self.scale = min(scale_x, scale_y)

        # Rendered building dimensions in screen pixels
        self.building_px_w = self.world_width * self.scale
        self.building_px_h = self.world_height * self.scale

        # Center building within the drawable area
        self.offset_x = self.margin + (avail_w - self.building_px_w) / 2.0
        self.offset_y = self.top_offset + (avail_h - self.building_px_h) / 2.0

    def update_screen_size(self, width: int, height: int) -> None:
        """Update screen size and recompute scale and offsets."""
        self.screen_width = max(100, int(width))
        self.screen_height = max(100, int(height))
        self.recalculate()

    def world_to_screen_x(self, world_x: float) -> float:
        """Convert a world X coordinate (metres) to screen X (pixels)."""
        return self.offset_x + float(world_x) * self.scale

    def world_to_screen_y(self, world_y: float) -> float:
        """Convert a world Y coordinate (metres) to screen Y (pixels)."""
        return self.offset_y + float(world_y) * self.scale

    def world_to_screen_point(
        self, point: Union[Point, tuple[float, float]]
    ) -> tuple[float, float]:
        """Convert a world point to screen pixel coordinates."""
        if isinstance(point, Point):
            wx, wy = point.x, point.y
        else:
            wx, wy = point[0], point[1]
        return (self.world_to_screen_x(wx), self.world_to_screen_y(wy))

    def world_to_screen_dim(self, meters: float) -> float:
        """Convert a scalar world distance/dimension in metres to screen pixels."""
        return float(meters) * self.scale

    def world_to_screen_rect(
        self, rect: Rectangle
    ) -> tuple[float, float, float, float]:
        """Convert a world Rectangle to screen tuple (screen_x, screen_y, screen_w, screen_h)."""
        sx = self.world_to_screen_x(rect.x)
        sy = self.world_to_screen_y(rect.y)
        sw = self.world_to_screen_dim(rect.width)
        sh = self.world_to_screen_dim(rect.height)
        return (sx, sy, sw, sh)

    def screen_to_world_point(
        self, screen_point: tuple[float, float]
    ) -> Point:
        """Convert screen pixel coordinates back to a world Point in metres."""
        sx, sy = screen_point
        wx = (sx - self.offset_x) / self.scale
        wy = (sy - self.offset_y) / self.scale
        return Point(x=wx, y=wy)

    @property
    def building_screen_bounds(self) -> tuple[float, float, float, float]:
        """Return screen bounding box (x, y, w, h) of the entire building."""
        return (self.offset_x, self.offset_y, self.building_px_w, self.building_px_h)
