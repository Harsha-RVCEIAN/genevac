"""Pygame renderer for GenEvac building geometry and pedestrian state.

This module is deliberately duck-typed: it accepts the public building,
viewport, engine, and registry objects used by the GenEvac/Pavan simulation
without importing or modifying the simulation engine itself.

Expected integration contract:
    EnvironmentRenderer().draw(screen, building, viewport, engine=engine)
    EnvironmentRenderer()._draw_pedestrians(screen, registry, viewport)

Coordinates are world-space metres. Conversion to pixels is delegated to the
provided viewport.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

try:
    import pygame
except ImportError:  # Allow importing this module in non-visual test environments.
    pygame = None  # type: ignore[assignment]


Color = Tuple[int, int, int]


class EnvironmentRenderer:
    """Render a building plan and its pedestrians onto a Pygame surface."""

    BACKGROUND: Color = (17, 24, 39)
    FLOOR: Color = (235, 239, 243)
    FLOOR_BORDER: Color = (100, 116, 139)
    OBSTACLE: Color = (71, 85, 105)
    WALL: Color = (51, 65, 85)
    EXIT_OPEN: Color = (34, 197, 94)
    EXIT_CLOSED: Color = (239, 68, 68)
    PEDESTRIAN: Color = (37, 99, 235)
    PEDESTRIAN_BLOCKED: Color = (245, 158, 11)

    def __init__(self, *, draw_labels: bool = True, pedestrian_outline: bool = True) -> None:
        self.draw_labels = bool(draw_labels)
        self.pedestrian_outline = bool(pedestrian_outline)
        self._font: Any = None
        self._small_font: Any = None

    @staticmethod
    def _require_pygame() -> Any:
        if pygame is None:
            raise RuntimeError(
                "pygame is required for rendering. Install it with `pip install pygame`."
            )
        return pygame

    @staticmethod
    def _value(obj: Any, *names: str, default: Any = None) -> Any:
        """Return the first available attribute/key from an object."""
        for name in names:
            if isinstance(obj, dict) and name in obj:
                return obj[name]
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    @classmethod
    def _xy(cls, point: Any) -> Tuple[float, float]:
        if point is None:
            raise ValueError("A world-space point cannot be None")
        if isinstance(point, (tuple, list)) and len(point) >= 2:
            return float(point[0]), float(point[1])
        return (
            float(cls._value(point, "x", default=0.0)),
            float(cls._value(point, "y", default=0.0)),
        )

    @classmethod
    def _rect_values(cls, bounds: Any) -> Tuple[float, float, float, float]:
        """Normalize Rect-like objects and (x, y, width, height) sequences."""
        if isinstance(bounds, (tuple, list)) and len(bounds) >= 4:
            x, y, a, b = map(float, bounds[:4])
            # A sequence is interpreted as x, y, width, height.
            return x, y, max(0.0, a), max(0.0, b)
        x = float(cls._value(bounds, "x", "left", default=0.0))
        y = float(cls._value(bounds, "y", "top", default=0.0))
        width = float(cls._value(bounds, "width", default=0.0))
        height = float(cls._value(bounds, "height", default=0.0))
        return x, y, max(0.0, width), max(0.0, height)

    @staticmethod
    def _iter(value: Any) -> Iterable[Any]:
        if value is None:
            return ()
        if isinstance(value, dict):
            return value.values()
        try:
            return iter(value)
        except TypeError:
            return ()

    @staticmethod
    def _screen_point(viewport: Any, point: Any) -> Tuple[int, int]:
        converted = viewport.world_to_screen_point(point)
        return int(round(converted[0])), int(round(converted[1]))

    @staticmethod
    def _screen_dim(viewport: Any, value: float) -> int:
        converter = getattr(viewport, "world_to_screen_dim", None)
        if callable(converter):
            res: Any = converter(float(value))
            return max(1, int(round(abs(float(res)))))
        return max(1, int(round(abs(float(value)))))

    def _fonts(self) -> None:
        pg = self._require_pygame()
        if self._font is None:
            self._font = pg.font.Font(None, 20)
        if self._small_font is None:
            self._small_font = pg.font.Font(None, 16)

    def draw(
        self,
        screen: Any,
        building: Any,
        viewport: Any,
        *,
        engine: Optional[Any] = None,
    ) -> None:
        """Draw floor, obstacles, walls, exits, and (when supplied) pedestrians."""
        pg = self._require_pygame()
        if screen is None or building is None or viewport is None:
            raise ValueError("screen, building, and viewport are required")

        screen.fill(self.BACKGROUND)
        self._draw_floor(screen, building, viewport)
        self._draw_obstacles(screen, building, viewport)
        self._draw_walls(screen, building, viewport)
        self._draw_exits(screen, building, viewport)

        if engine is not None:
            registry = self._value(engine, "registry")
            if registry is not None:
                self._draw_pedestrians(screen, registry, viewport)

    def _draw_floor(self, screen: Any, building: Any, viewport: Any) -> None:
        pg = self._require_pygame()
        bounds = getattr(viewport, "building_screen_bounds", None)
        if bounds is not None and len(bounds) >= 4:
            rect = pg.Rect(*(int(round(v)) for v in bounds))
            pg.draw.rect(screen, self.FLOOR, rect)
            pg.draw.rect(screen, self.FLOOR_BORDER, rect, width=2)
            return

        width = float(self._value(building, "width", default=0.0))
        height = float(self._value(building, "height", default=0.0))
        top_left = self._screen_point(viewport, (0.0, 0.0))
        bottom_right = self._screen_point(viewport, (width, height))
        rect = pg.Rect(
            min(top_left[0], bottom_right[0]),
            min(top_left[1], bottom_right[1]),
            abs(bottom_right[0] - top_left[0]),
            abs(bottom_right[1] - top_left[1]),
        )
        pg.draw.rect(screen, self.FLOOR, rect)
        pg.draw.rect(screen, self.FLOOR_BORDER, rect, width=2)

    def _draw_obstacles(self, screen: Any, building: Any, viewport: Any) -> None:
        pg = self._require_pygame()
        for obstacle in self._iter(self._value(building, "obstacles", default=())):
            bounds = self._value(obstacle, "bounds")
            if bounds is None:
                continue
            x, y, width, height = self._rect_values(bounds)
            p1 = self._screen_point(viewport, (x, y))
            p2 = self._screen_point(viewport, (x + width, y + height))
            rect = pg.Rect(
                min(p1[0], p2[0]),
                min(p1[1], p2[1]),
                max(1, abs(p2[0] - p1[0])),
                max(1, abs(p2[1] - p1[1])),
            )
            pg.draw.rect(screen, self.OBSTACLE, rect, border_radius=2)
            if self.draw_labels:
                self._fonts()
                label = self._value(obstacle, "id", "name")
                if label:
                    text = self._small_font.render(str(label), True, (248, 250, 252))
                    screen.blit(text, (rect.x + 3, rect.y + 2))

    def _draw_walls(self, screen: Any, building: Any, viewport: Any) -> None:
        pg = self._require_pygame()
        for wall in self._iter(self._value(building, "walls", default=())):
            segment = self._value(wall, "segment", default=wall)
            start = self._value(segment, "start", "p1", "a")
            end = self._value(segment, "end", "p2", "b")
            if start is None or end is None:
                continue
            pg.draw.line(
                screen,
                self.WALL,
                self._screen_point(viewport, self._xy(start)),
                self._screen_point(viewport, self._xy(end)),
                width=4,
            )

    def _draw_exits(self, screen: Any, building: Any, viewport: Any) -> None:
        pg = self._require_pygame()
        for exit_obj in self._iter(self._value(building, "exits", default=())):
            segment = self._value(exit_obj, "segment")
            if segment is None:
                continue
            start = self._value(segment, "start")
            end = self._value(segment, "end")
            if start is None or end is None:
                continue
            is_open = self._value(exit_obj, "is_open", "opened", default=True)
            color = self.EXIT_OPEN if bool(is_open) else self.EXIT_CLOSED
            p1 = self._screen_point(viewport, self._xy(start))
            p2 = self._screen_point(viewport, self._xy(end))
            pg.draw.line(screen, color, p1, p2, width=7)
            if self.draw_labels:
                self._fonts()
                label = self._value(exit_obj, "id", "name", default="EXIT")
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                text = self._small_font.render(str(label), True, color)
                screen.blit(text, (mid[0] + 4, mid[1] - 16))

    @classmethod
    def _registry_pedestrians(cls, registry: Any) -> Iterable[Any]:
        """Support common registry APIs without assuming one implementation."""
        for name in ("all", "get_all", "get_pedestrians", "values"):
            method = getattr(registry, name, None)
            if callable(method):
                try:
                    return cls._iter(method())
                except TypeError:
                    continue
        pedestrians = cls._value(registry, "pedestrians", "_pedestrians", "agents")
        if pedestrians is not None:
            return cls._iter(pedestrians)
        if isinstance(registry, (dict, list, tuple, set)):
            return cls._iter(registry)
        return ()

    def _draw_pedestrians(self, screen: Any, registry: Any, viewport: Any) -> None:
        """Draw registered pedestrians; kept public-compatible with integrated_app."""
        pg = self._require_pygame()
        for pedestrian in self._registry_pedestrians(registry):
            position = self._value(pedestrian, "position")
            if position is None:
                continue
            try:
                center = self._screen_point(viewport, self._xy(position))
            except (TypeError, ValueError, AttributeError):
                continue

            radius_world = float(self._value(pedestrian, "radius", default=0.22))
            radius_px = self._screen_dim(viewport, radius_world)
            status = self._value(pedestrian, "status", default="")
            status_text = str(getattr(status, "value", status)).lower()
            if "evac" in status_text:
                color = self.EXIT_OPEN
            elif "block" in status_text or "inactive" in status_text:
                color = (245, 158, 11)
            else:
                color = self.PEDESTRIAN

            pg.draw.circle(screen, color, center, radius_px)
            if self.pedestrian_outline:
                pg.draw.circle(screen, (15, 23, 42), center, radius_px, width=1)

            if self.draw_labels:
                pedestrian_id = self._value(pedestrian, "id", "agent_id")
                if pedestrian_id is not None:
                    self._fonts()
                    label = self._small_font.render(str(pedestrian_id), True, (15, 23, 42))
                    screen.blit(label, (center[0] + radius_px + 2, center[1] - radius_px - 2))


__all__ = ["EnvironmentRenderer"]
