"""Script to generate verified screenshots for all 3 Operating Modes and all 5 Presentation Stages."""

import os
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
from genevac.visualization.integrated_app import (
    IntegratedSimulationApp,
    OperatingMode,
    PresentationStage,
    PROJECT_ROOT,
)


def render_and_save(app: IntegratedSimulationApp, out_path: Path) -> None:
    app.renderer.draw(app.screen, app.building, app.viewport, engine=app.engine)
    app._render_spatial_heatmap(app.screen)
    app.renderer._draw_pedestrians(app.screen, app.engine.registry, app.viewport)
    app._render_predictions(app.screen)
    app._render_hud_and_telemetry(app.screen)
    pygame.image.save(app.screen, str(out_path))
    print(f"Saved: {out_path.name} ({out_path.stat().st_size} bytes)")


def main() -> None:
    map_path = PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json"
    assets_dir = PROJECT_ROOT / "presentation_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()
    if not pygame.font.get_init():
        pygame.font.init()

    base_app = IntegratedSimulationApp(map_path=str(map_path), width=1440, height=860, fps=30)
    shared_mid = base_app.mid_predictor

    # 1. Generate screenshots for each of the 5 Presentation Stages
    stages = [
        (PresentationStage.STAGE_1_LIVE_EVACUATION, "presentation_stage_1_stage_1_live_evacuation.png"),
        (PresentationStage.STAGE_2_SPATIAL_HEATMAP, "presentation_stage_2_stage_2_spatial_heatmap.png"),
        (PresentationStage.STAGE_3_CONSTANT_VELOCITY, "presentation_stage_3_stage_3_constant_velocity.png"),
        (PresentationStage.STAGE_4_MID_PREDICTION, "presentation_stage_4_stage_4_mid_prediction.png"),
        (PresentationStage.STAGE_5_MID_ASSISTED_DEMONSTRATION, "presentation_stage_5_stage_5_mid_assisted_demonstration.png"),
    ]
    for stage, fname in stages:
        app = IntegratedSimulationApp(map_path=str(map_path), width=1440, height=860, fps=30)
        app.mid_predictor = shared_mid
        app.screen = pygame.display.set_mode((app.width, app.height))
        app.font_title = pygame.font.Font(None, 22)
        app.font_section = pygame.font.Font(None, 20)
        app.font_hud = pygame.font.Font(None, 18)
        app.font_small = pygame.font.Font(None, 16)
        app.font_legend = pygame.font.Font(None, 16)

        if stage == PresentationStage.STAGE_5_MID_ASSISTED_DEMONSTRATION:
            app.operating_mode = OperatingMode.MID_ASSISTED_EVACUATION

        for _ in range(28):
            app.update_physics(0.1)

        app.presentation_mode = True
        app.set_presentation_stage(stage)
        render_and_save(app, assets_dir / fname)

    # 2. Generate screenshots for the 3 Operating Modes at Step 28
    mode_configs = [
        (OperatingMode.ORIGINAL_SIMULATION, "mode1_original_simulation.png", False, False),
        (OperatingMode.MID_PREDICTION_ONLY, "mode2_mid_prediction_only.png", True, True),
        (OperatingMode.MID_ASSISTED_EVACUATION, "mode3_mid_assisted_evacuation.png", True, True),
    ]
    for mode, fname, show_mid, show_cv in mode_configs:
        app = IntegratedSimulationApp(map_path=str(map_path), width=1440, height=860, fps=30)
        app.mid_predictor = shared_mid
        app.screen = pygame.display.set_mode((app.width, app.height))
        app.font_title = pygame.font.Font(None, 22)
        app.font_section = pygame.font.Font(None, 20)
        app.font_hud = pygame.font.Font(None, 18)
        app.font_small = pygame.font.Font(None, 16)
        app.font_legend = pygame.font.Font(None, 16)

        app.operating_mode = mode
        app.show_mid_predictions = show_mid
        app.show_baseline_predictions = show_cv
        app.selected_agent_id = "ped_1"

        for _ in range(28):
            app.update_physics(0.1)

        render_and_save(app, assets_dir / fname)


if __name__ == "__main__":
    main()
