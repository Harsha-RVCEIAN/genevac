"""Unit and regression tests for visualization refinement enhancements in integrated_app.py."""

import os
import pytest
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"

from genevac.visualization.integrated_app import (
    IntegratedSimulationApp,
    PROJECT_ROOT,
)

@pytest.fixture
def refined_app():
    map_path = PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json"
    return IntegratedSimulationApp(map_path=str(map_path), width=1440, height=860, fps=30)

class TestVisualizationRefinements:
    def test_side_panel_constant(self, refined_app: IntegratedSimulationApp):
        """Verify dedicated side panel width is defined and respected in viewport."""
        assert refined_app.SIDE_PANEL_WIDTH == 380
        # Viewport width must not exceed width - SIDE_PANEL_WIDTH
        assert refined_app.viewport.screen_width == refined_app.width - refined_app.SIDE_PANEL_WIDTH

    def test_mid_sample_predictions_rendering(self, refined_app: IntegratedSimulationApp):
        """Verify MID samples rendering parses sample.predictions correctly."""
        for _ in range(25):
            refined_app.update_physics(0.1)

        import pygame
        pygame.init()
        refined_app.screen = pygame.display.set_mode((refined_app.width, refined_app.height))
        refined_app.font_title = pygame.font.Font(None, 22)
        refined_app.font_section = pygame.font.Font(None, 20)
        refined_app.font_hud = pygame.font.Font(None, 18)
        refined_app.font_small = pygame.font.Font(None, 16)
        refined_app.font_legend = pygame.font.Font(None, 16)

        # Call _render_predictions - must execute without AttributeError on sample.predictions
        refined_app._render_predictions(refined_app.screen)
        # Should record walkability metrics
        assert hasattr(refined_app, "last_mid_walkability")
        assert hasattr(refined_app, "last_mid_collision_free")

    def test_transparent_heatmap_rendering(self, refined_app: IntegratedSimulationApp):
        """Verify transparent heatmap with legend renders cleanly."""
        for _ in range(25):
            refined_app.update_physics(0.1)

        import pygame
        pygame.init()
        refined_app.screen = pygame.display.set_mode((refined_app.width, refined_app.height))
        refined_app.font_small = pygame.font.Font(None, 16)

        # Render heatmap
        refined_app._render_spatial_heatmap(refined_app.screen)

    def test_dedicated_hud_rendering(self, refined_app: IntegratedSimulationApp):
        """Verify dedicated right-hand telemetry panel renders cleanly."""
        for _ in range(25):
            refined_app.update_physics(0.1)

        import pygame
        pygame.init()
        refined_app.screen = pygame.display.set_mode((refined_app.width, refined_app.height))
        refined_app.font_title = pygame.font.Font(None, 22)
        refined_app.font_section = pygame.font.Font(None, 20)
        refined_app.font_hud = pygame.font.Font(None, 18)
        refined_app.font_small = pygame.font.Font(None, 16)
        refined_app.font_legend = pygame.font.Font(None, 16)

        refined_app._render_hud_and_telemetry(refined_app.screen)

