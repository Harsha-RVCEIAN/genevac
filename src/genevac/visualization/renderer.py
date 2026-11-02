"""Environment renderer module for GenEvac-Sentinel.

Renders 2D top-down visualization of a Building model using Pygame:
- Walkable areas (rooms and corridor)
- Physical 2D wall footprints with 0.2m thickness
- Openings for doorways and exits
- Static obstacles
- High-visibility safety green exits
- Clear labels, scale indicator, and legend
"""

from __future__ import annotations

import math
from typing import Optional
import pygame

from genevac.environment.geometry import Point, Rectangle, WallType
from genevac.environment.models import Building, Doorway, Exit, Obstacle, WalkableArea
from genevac.pedestrian.models import PedestrianState, PedestrianStatus
from genevac.pedestrian.registry import PedestrianRegistry
from genevac.simulation.engine import SimulationEngine
from genevac.visualization.coordinates import Viewport


class EnvironmentRenderer:
    """Renders a Building environment model and active simulation state onto a Pygame surface."""

    # Modern curated architectural color palette
    COLOR_BG = (22, 27, 34)  # Deep slate backdrop
    COLOR_CANVAS = (13, 17, 23)  # Surrounding building apron
    COLOR_ROOM_FLOOR = (235, 240, 246)  # Clean cool light floor
    COLOR_CORRIDOR_FLOOR = (248, 250, 252)  # Brighter warm floor for corridor
    COLOR_FLOOR_BORDER = (203, 213, 225)  # Seam border
    COLOR_WALL_EXTERIOR = (30, 41, 59)  # Dark dense slate for exterior walls
    COLOR_WALL_INTERIOR = (51, 65, 85)  # Medium slate for interior partitions
    COLOR_WALL_OUTLINE = (15, 23, 42)
    COLOR_OBSTACLE = (217, 119, 6)  # Rich warm amber for obstacles
    COLOR_OBSTACLE_BORDER = (146, 64, 14)
    COLOR_DOORWAY = (14, 165, 233)  # Vibrant cyan portal threshold
    COLOR_EXIT = (34, 197, 94)  # Vibrant safety emerald green
    COLOR_TEXT_TITLE = (241, 245, 249)
    COLOR_TEXT_MUTED = (148, 163, 184)
    COLOR_TEXT_FLOOR = (71, 85, 105)
    COLOR_BORDER_BUILDING = (74, 85, 104)

    # Pedestrian state colors (read-only visual representation)
    COLOR_PED_ACTIVE = (56, 189, 248)  # Electric cyan/blue
    COLOR_PED_ACTIVE_CORE = (255, 255, 255)  # White center dot
    COLOR_PED_EVACUATED = (34, 197, 94)  # Emerald green
    COLOR_PED_BLOCKED = (239, 68, 68)  # Coral red
    COLOR_PED_BLOCKED_BORDER = (153, 27, 27)  # Deep red border
    COLOR_PED_INACTIVE = (148, 163, 184)  # Slate grey
    COLOR_PED_BADGE_BG = (15, 23, 42)  # Dark badge pill
    COLOR_PED_BADGE_TEXT = (241, 245, 249)  # Light badge text
    COLOR_HUD_RUNNING = (34, 197, 94)  # Status green
    COLOR_HUD_PAUSED = (234, 179, 8)  # Status amber/yellow

    def __init__(self) -> None:
        if not pygame.font.get_init():
            pygame.font.init()

        # Cached fonts using system or default font
        self.font_title = pygame.font.Font(None, 24)
        self.font_body = pygame.font.Font(None, 18)
        self.font_small = pygame.font.Font(None, 14)
        self.font_badge = pygame.font.Font(None, 16)

    def draw(
        self,
        surface: pygame.Surface,
        building: Building,
        viewport: Viewport,
        engine: Optional[SimulationEngine] = None,
        registry: Optional[PedestrianRegistry] = None,
    ) -> None:
        """Draw the complete building environment and optional simulation state onto the surface."""
        # 1. Fill window background
        surface.fill(self.COLOR_BG)

        # 2. Draw building canvas area
        bx, by, bw, bh = viewport.building_screen_bounds
        canvas_rect = pygame.Rect(int(bx), int(by), int(bw), int(bh))
        pygame.draw.rect(surface, self.COLOR_CANVAS, canvas_rect)

        # 3. Draw walkable areas (floors)
        self._draw_walkable_areas(surface, building, viewport)

        # 4. Draw doorways (portal indicators and open clearances)
        self._draw_doorways(surface, building, viewport)

        # 5. Draw exits (safety green thresholds and badges)
        self._draw_exits(surface, building, viewport)

        # 6. Draw walls (using physical 2D footprints)
        self._draw_walls(surface, building, viewport)

        # 7. Draw obstacles
        self._draw_obstacles(surface, building, viewport)

        # 8. Draw building exterior outline
        pygame.draw.rect(surface, self.COLOR_BORDER_BUILDING, canvas_rect, width=1)

        # 9. Draw pedestrians if simulation state or registry is available
        # Ensure single authoritative source of truth: engine.registry takes precedence
        effective_registry = engine.registry if engine is not None else registry
        if effective_registry is not None:
            self._draw_pedestrians(surface, effective_registry, viewport)

        # 10. Draw HUD: Title header, scale bar, and legend
        self._draw_header(surface, building, viewport, engine, effective_registry)
        self._draw_scale_bar(surface, viewport)
        self._draw_legend(surface, viewport, show_pedestrians=(effective_registry is not None))

    def _draw_walkable_areas(
        self, surface: pygame.Surface, building: Building, viewport: Viewport
    ) -> None:
        """Render walkable rooms and corridors with distinct tones and labels."""
        for area in building.walkable_areas:
            sx, sy, sw, sh = viewport.world_to_screen_rect(area.bounds)
            rect = pygame.Rect(int(sx), int(sy), int(sw), int(sh))

            # Floor color distinction
            fill_color = (
                self.COLOR_CORRIDOR_FLOOR
                if area.area_type == "corridor"
                else self.COLOR_ROOM_FLOOR
            )
            pygame.draw.rect(surface, fill_color, rect)
            pygame.draw.rect(surface, self.COLOR_FLOOR_BORDER, rect, width=1)

            # Center text label
            label_text = self._format_area_label(area)
            text_surf = self.font_body.render(label_text, True, self.COLOR_TEXT_FLOOR)
            text_rect = text_surf.get_rect(center=rect.center)
            surface.blit(text_surf, text_rect)

    def _format_area_label(self, area: WalkableArea) -> str:
        """Format a user-friendly label for a walkable zone."""
        name_map = {
            "room_1": "Room 1 (NW)",
            "room_2": "Room 2 (NE)",
            "main_corridor": "Main Corridor (3.0 m)",
            "room_3": "Room 3 (SW)",
            "room_4": "Room 4 (SE)",
        }
        return name_map.get(area.id, area.id.replace("_", " ").title())

    def _draw_walls(
        self, surface: pygame.Surface, building: Building, viewport: Viewport
    ) -> None:
        """Render walls as solid filled rectangles using their actual physical 2D footprints."""
        for wall in building.walls:
            sx, sy, sw, sh = viewport.world_to_screen_rect(wall.footprint)
            # Ensure minimum pixel dimension of 2px so thin walls remain visible on screen
            sw = max(2.0, sw)
            sh = max(2.0, sh)
            rect = pygame.Rect(int(sx), int(sy), int(sw), int(sh))

            color = (
                self.COLOR_WALL_EXTERIOR
                if wall.wall_type == WallType.EXTERIOR
                else self.COLOR_WALL_INTERIOR
            )
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, self.COLOR_WALL_OUTLINE, rect, width=1)

    def _draw_obstacles(
        self, surface: pygame.Surface, building: Building, viewport: Viewport
    ) -> None:
        """Render static obstacles with distinct borders and identifiers."""
        for obs in building.obstacles:
            sx, sy, sw, sh = viewport.world_to_screen_rect(obs.bounds)
            rect = pygame.Rect(int(sx), int(sy), int(sw), int(sh))

            pygame.draw.rect(surface, self.COLOR_OBSTACLE, rect)
            pygame.draw.rect(surface, self.COLOR_OBSTACLE_BORDER, rect, width=2)

            # Small obstacle label
            label = self.font_small.render("Obstacle", True, (255, 255, 255))
            label_rect = label.get_rect(center=rect.center)
            surface.blit(label, label_rect)

    def _draw_doorways(
        self, surface: pygame.Surface, building: Building, viewport: Viewport
    ) -> None:
        """Render doorway portal thresholds with clear cyan accent lines and jamb markers."""
        for door in building.doorways:
            p1 = viewport.world_to_screen_point(door.segment.start)
            p2 = viewport.world_to_screen_point(door.segment.end)

            # Threshold line
            pygame.draw.line(surface, self.COLOR_DOORWAY, p1, p2, width=4)

            # Jamb markers at endpoints
            pygame.draw.circle(surface, self.COLOR_DOORWAY, (int(p1[0]), int(p1[1])), 3)
            pygame.draw.circle(surface, self.COLOR_DOORWAY, (int(p2[0]), int(p2[1])), 3)

    def _draw_exits(
        self, surface: pygame.Surface, building: Building, viewport: Viewport
    ) -> None:
        """Render exterior exits with prominent emerald green threshold bands and badges."""
        for ex in building.exits:
            p1 = viewport.world_to_screen_point(ex.segment.start)
            p2 = viewport.world_to_screen_point(ex.segment.end)

            # Thick safety green exit band
            pygame.draw.line(surface, self.COLOR_EXIT, p1, p2, width=8)

            # Draw "EXIT" badge alongside the opening
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0

            badge_text = self.font_badge.render("EXIT", True, (255, 255, 255))
            badge_bg = pygame.Rect(0, 0, badge_text.get_width() + 8, badge_text.get_height() + 4)

            # Offset badge slightly based on boundary orientation
            if ex.boundary == "left":
                badge_bg.center = (int(mid_x - 24), int(mid_y))
            elif ex.boundary == "right":
                badge_bg.center = (int(mid_x + 24), int(mid_y))
            elif ex.boundary == "bottom":
                badge_bg.center = (int(mid_x), int(mid_y + 16))
            else:  # top
                badge_bg.center = (int(mid_x), int(mid_y - 16))

            pygame.draw.rect(surface, self.COLOR_EXIT, badge_bg, border_radius=3)
            surface.blit(
                badge_text,
                (
                    badge_bg.x + 4,
                    badge_bg.y + 2,
                ),
            )

    def _draw_pedestrians(
        self,
        surface: pygame.Surface,
        registry: PedestrianRegistry,
        viewport: Viewport,
    ) -> None:
        """Render pedestrians at their world coordinates with status-coded markers and ID badges."""
        for ped in registry:
            sx, sy = viewport.world_to_screen_point(ped.position)
            # Physical pedestrian radius converted to screen pixels with minimum visibility clamp
            r_px = max(4.0, viewport.world_to_screen_dim(ped.radius))
            ix, iy = int(round(sx)), int(round(sy))
            ir = max(3, int(round(r_px)))

            if ped.status == PedestrianStatus.ACTIVE:
                # Active: Electric cyan/blue with crisp core
                pygame.draw.circle(surface, self.COLOR_PED_ACTIVE, (ix, iy), ir)
                pygame.draw.circle(surface, (14, 116, 144), (ix, iy), ir, width=1)
                pygame.draw.circle(surface, self.COLOR_PED_ACTIVE_CORE, (ix, iy), max(1, int(ir * 0.35)))
            elif ped.status == PedestrianStatus.EVACUATED:
                # Evacuated: Emerald green ring with center point
                pygame.draw.circle(surface, self.COLOR_PED_EVACUATED, (ix, iy), ir, width=2)
                pygame.draw.circle(surface, self.COLOR_PED_EVACUATED, (ix, iy), max(1, int(ir * 0.4)))
            elif ped.status == PedestrianStatus.BLOCKED:
                # Blocked: Coral red filled circle with dark red border
                pygame.draw.circle(surface, self.COLOR_PED_BLOCKED, (ix, iy), ir)
                pygame.draw.circle(surface, self.COLOR_PED_BLOCKED_BORDER, (ix, iy), ir, width=2)
            else:  # INACTIVE
                # Inactive: Muted slate grey
                pygame.draw.circle(surface, self.COLOR_PED_INACTIVE, (ix, iy), ir)

            # ID Badge above pedestrian marker
            badge_surf = self.font_badge.render(ped.id, True, self.COLOR_PED_BADGE_TEXT)
            bw = badge_surf.get_width() + 6
            bh = badge_surf.get_height() + 2
            bx = int(ix - bw / 2.0)
            by = int(iy - ir - bh - 2)

            badge_rect = pygame.Rect(bx, by, bw, bh)
            pygame.draw.rect(surface, self.COLOR_PED_BADGE_BG, badge_rect, border_radius=3)
            pygame.draw.rect(surface, self.COLOR_BORDER_BUILDING, badge_rect, width=1, border_radius=3)
            surface.blit(badge_surf, (bx + 3, by + 1))

    def _draw_header(
        self,
        surface: pygame.Surface,
        building: Building,
        viewport: Viewport,
        engine: Optional[SimulationEngine] = None,
        registry: Optional[PedestrianRegistry] = None,
    ) -> None:
        """Render header title bar with environment and simulation metadata."""
        # Main Title (Left)
        title = self.font_title.render(
            f"GenEvac-Sentinel: {building.id} ({building.width:.1f} m x {building.height:.1f} m)",
            True,
            self.COLOR_TEXT_TITLE,
        )
        surface.blit(title, (24, 14))

        if engine is not None:
            # Simulation State Pill (Top Right)
            status_text = "[PAUSED]" if engine.is_paused else "[RUNNING]"
            status_color = self.COLOR_HUD_PAUSED if engine.is_paused else self.COLOR_HUD_RUNNING
            status_surf = self.font_title.render(status_text, True, status_color)

            # Sim Time and Step
            time_text = f"Time: {engine.clock.timestamp:.2f} s | Step: {engine.clock.step_count}"
            time_surf = self.font_title.render(time_text, True, self.COLOR_TEXT_TITLE)

            total_right_w = status_surf.get_width() + 16 + time_surf.get_width()
            right_x = surface.get_width() - total_right_w - 24
            surface.blit(time_surf, (right_x, 14))
            surface.blit(status_surf, (right_x + time_surf.get_width() + 16, 14))

            # Row 2 (Subheader): Pedestrian Counts and Control Hints
            reg = registry if registry is not None else engine.registry
            ped_counts = (
                f"Pedestrians: {reg.total_count} Total | {reg.active_count} Active | "
                f"{reg.evacuated_count} Evacuated | {reg.blocked_count} Blocked"
            )
            sub_left = self.font_small.render(ped_counts, True, self.COLOR_TEXT_MUTED)
            surface.blit(sub_left, (24, 38))

            controls_text = "[SPACE] Pause/Resume  |  [R] Reset Sim  |  [Q / ESC] Quit"
            sub_right = self.font_small.render(controls_text, True, self.COLOR_TEXT_MUTED)
            surface.blit(sub_right, (surface.get_width() - sub_right.get_width() - 24, 38))
        else:
            sub = self.font_small.render(
                f"4 Rooms | 1 Central Corridor (3.0 m) | 3 Exits | 4 Doorways | 4 Obstacles | Scale: 1 m = {viewport.scale:.1f} px",
                True,
                self.COLOR_TEXT_MUTED,
            )
            surface.blit(sub, (24, 38))

    def _draw_scale_bar(self, surface: pygame.Surface, viewport: Viewport) -> None:
        """Render a 10-meter physical scale bar in the lower left."""
        bar_m = 10.0
        bar_px = viewport.world_to_screen_dim(bar_m)
        x = 24
        y = surface.get_height() - 32

        # Draw ruler bar
        pygame.draw.line(surface, self.COLOR_TEXT_TITLE, (x, y), (x + bar_px, y), width=3)
        pygame.draw.line(surface, self.COLOR_TEXT_TITLE, (x, y - 4), (x, y + 4), width=2)
        pygame.draw.line(
            surface, self.COLOR_TEXT_TITLE, (x + bar_px, y - 4), (x + bar_px, y + 4), width=2
        )

        label = self.font_small.render(f"{bar_m:.0f} m", True, self.COLOR_TEXT_TITLE)
        surface.blit(label, (x + bar_px / 2.0 - label.get_width() / 2.0, y - 16))

    def _draw_legend(
        self,
        surface: pygame.Surface,
        viewport: Viewport,
        show_pedestrians: bool = False,
    ) -> None:
        """Render color-coded visual legend along the bottom edge."""
        legend_items: list[tuple[str, tuple[int, int, int], Optional[tuple[int, int, int]]]] = [
            ("Floor", self.COLOR_ROOM_FLOOR, self.COLOR_FLOOR_BORDER),
            ("Wall", self.COLOR_WALL_EXTERIOR, self.COLOR_WALL_OUTLINE),
            ("Doorway", self.COLOR_DOORWAY, None),
            ("Obstacle", self.COLOR_OBSTACLE, self.COLOR_OBSTACLE_BORDER),
            ("Exit", self.COLOR_EXIT, None),
        ]
        if show_pedestrians:
            legend_items.extend([
                ("Active", self.COLOR_PED_ACTIVE, None),
                ("Evacuated", self.COLOR_PED_EVACUATED, None),
                ("Blocked", self.COLOR_PED_BLOCKED, None),
            ])

        # Render items from right margin
        total_w = 0
        rendered_items = []
        for label, color, border in legend_items:
            text_surf = self.font_small.render(label, True, self.COLOR_TEXT_MUTED)
            item_w = 14 + 6 + text_surf.get_width() + 16
            total_w += item_w
            rendered_items.append((label, color, border, text_surf))

        curr_x = max(24, surface.get_width() - total_w - 12)
        y = surface.get_height() - 28

        for label, color, border, text_surf in rendered_items:
            # Swatch
            swatch_rect = pygame.Rect(curr_x, y + 2, 14, 10)
            pygame.draw.rect(surface, color, swatch_rect)
            if border:
                pygame.draw.rect(surface, border, swatch_rect, width=1)

            surface.blit(text_surf, (curr_x + 18, y))
            curr_x += 18 + text_surf.get_width() + 14


def draw_building(
    surface: pygame.Surface,
    building: Building,
    viewport: Viewport,
    renderer: Optional[EnvironmentRenderer] = None,
    engine: Optional[SimulationEngine] = None,
    registry: Optional[PedestrianRegistry] = None,
) -> None:
    """Convenience functional API to render a building using an EnvironmentRenderer."""
    if renderer is None:
        renderer = EnvironmentRenderer()
    renderer.draw(surface, building, viewport, engine=engine, registry=registry)
