"""Unit and regression tests for integrated visualization refinements.

Run from the repository root:
    pytest -q tests/test_visualization_refinement.py

These tests exercise the integrated application's actual rendering methods.
They require the Part 1 reference map and the project's visualization
dependencies (pygame, matplotlib, numpy).
"""

from __future__ import annotations

import os
from pathlib import Path

# Must be set before importing pygame or integrated_app.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

pygame = pytest.importorskip("pygame")

from genevac.visualization.integrated_app import (
    IntegratedSimulationApp,
    PROJECT_ROOT,
)


@pytest.fixture
def refined_app():
    """Create an integrated app using the project's canonical reference map."""
    map_path = (
        PROJECT_ROOT
        / "pavan-part1-reference"
        / "maps"
        / "building_main.json"
    )
    if not map_path.is_file():
        pytest.skip(f"Required Part 1 reference map is missing: {map_path}")

    pygame.init()
    pygame.font.init()

    app = IntegratedSimulationApp(
        map_path=str(map_path),
        width=1440,
        height=860,
        fps=30,
    )

    # Rendering helpers use these fonts. Initialize them explicitly so tests
    # do not depend on the application's interactive run() startup sequence.
    app.screen = pygame.Surface((app.width, app.height))
    app.font_title = pygame.font.Font(None, 22)
    app.font_section = pygame.font.Font(None, 20)
    app.font_hud = pygame.font.Font(None, 18)
    app.font_small = pygame.font.Font(None, 16)
    app.font_legend = pygame.font.Font(None, 16)

    try:
        yield app
    finally:
        pygame.quit()


def advance_simulation(app: IntegratedSimulationApp, steps: int = 25) -> None:
    """Advance the real engine enough to populate observations and heatmap data."""
    for _ in range(steps):
        app.update_physics(0.1)


class TestVisualizationRefinements:
    def test_side_panel_constant_and_viewport_width(self, refined_app):
        """The simulation viewport must leave room for the dedicated HUD panel."""
        app = refined_app

        assert app.SIDE_PANEL_WIDTH == 380
        assert app.viewport.screen_width == app.width - app.SIDE_PANEL_WIDTH

    def test_mid_sample_predictions_render_without_attribute_error(self, refined_app):
        """MID sample paths must be read through each sample's predictions field."""
        app = refined_app
        advance_simulation(app)

        # This is an execution regression test: with sufficient observation
        # history, rendering must not fail while traversing MID samples.
        app._render_predictions(app.screen)

        assert hasattr(app, "last_mid_walkability")
        assert hasattr(app, "last_mid_collision_free")

    def test_transparent_heatmap_rendering(self, refined_app):
        """The spatial heatmap and its legend must render onto a Pygame surface."""
        app = refined_app
        advance_simulation(app)

        # Ensure the test exercises the heatmap path even if the default
        # simulation state has no active pedestrians after the warm-up.
        active_positions = [
            (ped.position.x, ped.position.y)
            for ped in app.engine.registry
            if str(ped.status).lower().endswith("active")
        ]
        if not active_positions:
            pytest.skip("No active pedestrians available to populate the heatmap")

        app._render_spatial_heatmap(app.screen)

    def test_dedicated_hud_and_telemetry_panel_render(self, refined_app):
        """The right-side telemetry panel must render without missing fonts/state."""
        app = refined_app
        advance_simulation(app)

        app._render_hud_and_telemetry(app.screen)

    def test_full_visualization_frame_composes(self, refined_app):
        """The main rendering layers must compose on the configured screen."""
        app = refined_app
        advance_simulation(app)

        # Exercise the same major layers used by the integrated application.
        app.renderer.draw(app.screen, app.building, app.viewport, engine=app.engine)
        app._render_spatial_heatmap(app.screen)
        app._render_predictions(app.screen)
        app._render_hud_and_telemetry(app.screen)

        assert app.screen.get_size() == (app.width, app.height)
