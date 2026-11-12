"""Script to regenerate distinct, authentic screenshots with refined visualization.

Generates:
1. presentation_assets/integrated_simulation_live_pavan.png:
   Simulation running at step ~30, showing refined transparent spatial heatmap with
   density scale legend, multiple active pedestrians, Pavan building walls, exits,
   obstacles, and non-overlapping dedicated right-hand telemetry panel.

2. presentation_assets/selected_pedestrian_mid_predictions.png:
   Simulation stepped further (~45 steps), with a selected pedestrian displaying
   amber observation history trail, dashed Constant Velocity baseline, and all 6
   distinct high-contrast Evacuation MID stochastic prediction paths fanning around
   obstacles towards exits with clearly marked endpoints.
"""

import os
from pathlib import Path

# Headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from genevac.visualization.integrated_app import (
    IntegratedSimulationApp,
    PROJECT_ROOT,
)

def main():
    map_path = PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json"
    assets_dir = PROJECT_ROOT / "presentation_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Screenshot 1: Live Simulation + Heatmap (No prediction overlay)
    # -------------------------------------------------------------
    app1 = IntegratedSimulationApp(map_path=str(map_path), width=1440, height=860, fps=30)
    
    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()
    if not pygame.font.get_init():
        pygame.font.init()

    app1.screen = pygame.display.set_mode((app1.width, app1.height))
    app1.font_title = pygame.font.Font(None, 22)
    app1.font_section = pygame.font.Font(None, 20)
    app1.font_hud = pygame.font.Font(None, 18)
    app1.font_small = pygame.font.Font(None, 16)
    app1.font_legend = pygame.font.Font(None, 16)

    # Step simulation 25 times to get pedestrians moving and form density heatmap
    for _ in range(25):
        app1.update_physics(0.1)

    app1.show_heatmap = True
    app1.show_mid_predictions = False
    app1.show_baseline_predictions = False
    app1.selected_agent_id = None

    # Render frame
    app1.renderer.draw(app1.screen, app1.building, app1.viewport, engine=app1.engine)
    app1._render_spatial_heatmap(app1.screen)
    app1.renderer._draw_pedestrians(app1.screen, app1.engine.registry, app1.viewport)
    app1._render_hud_and_telemetry(app1.screen)

    out_path1 = assets_dir / "integrated_simulation_live_pavan.png"
    pygame.image.save(app1.screen, str(out_path1))
    print(f"Saved Screenshot 1: {out_path1} ({out_path1.stat().st_size} bytes)")

    # -------------------------------------------------------------
    # Screenshot 2: Target Pedestrian with MID Fan & Baseline
    # -------------------------------------------------------------
    app2 = IntegratedSimulationApp(map_path=str(map_path), width=1440, height=860, fps=30)
    app2.screen = pygame.display.set_mode((app2.width, app2.height))
    app2.font_title = pygame.font.Font(None, 22)
    app2.font_section = pygame.font.Font(None, 20)
    app2.font_hud = pygame.font.Font(None, 18)
    app2.font_small = pygame.font.Font(None, 16)
    app2.font_legend = pygame.font.Font(None, 16)

    # Step simulation 40 times to establish rich trajectory history
    for _ in range(40):
        app2.update_physics(0.1)

    app2.show_heatmap = True
    app2.show_mid_predictions = True
    app2.show_baseline_predictions = True
    active_peds = [p for p in app2.engine.registry if str(p.status).lower().endswith("active")]
    if active_peds:
        app2.selected_agent_id = active_peds[0].id

    # Render frame
    app2.renderer.draw(app2.screen, app2.building, app2.viewport, engine=app2.engine)
    app2._render_spatial_heatmap(app2.screen)
    app2.renderer._draw_pedestrians(app2.screen, app2.engine.registry, app2.viewport)
    app2._render_predictions(app2.screen)
    app2._render_hud_and_telemetry(app2.screen)

    out_path2 = assets_dir / "selected_pedestrian_mid_predictions.png"
    pygame.image.save(app2.screen, str(out_path2))
    print(f"Saved Screenshot 2: {out_path2} ({out_path2.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
