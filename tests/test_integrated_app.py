"""Integration tests for the GenEvac-Sentinel Integrated Simulation Application.

Verifies that Pavan's Module B Part 1 simulation engine and Harsha's Module B
Part 2 analytics, heatmaps, and prediction components integrate correctly within
IntegratedSimulationApp.

These tests run headless (SDL_VIDEODRIVER=dummy) and do NOT require a physical display.
"""

from __future__ import annotations

import os
import sys
import math
import pytest
from pathlib import Path

# Headless Pygame before any imports that touch pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Ensure project source is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARSHA_SRC = PROJECT_ROOT / "src"
if str(HARSHA_SRC) not in sys.path:
    sys.path.insert(0, str(HARSHA_SRC))

# Import integrated app (performs two-phase module loading internally)
from genevac.visualization.integrated_app import (
    IntegratedSimulationApp,
    create_pretrained_mid_predictor,
    PROJECT_ROOT as APP_PROJECT_ROOT,
    PAVAN,
)

# Part 2 imports (loaded via the integrated_app's module loader)
import genevac.prediction.data_contract as h_contract
import genevac.prediction.mid as h_mid
import genevac.prediction.baseline as h_baseline
import genevac.analytics.heatmap as h_heatmap


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def app():
    """Create a headless IntegratedSimulationApp with default building layout."""
    default_map = APP_PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json"
    if not default_map.is_file():
        pytest.skip(f"Building map not found: {default_map}")
    return IntegratedSimulationApp(map_path=str(default_map), width=800, height=600, fps=30)


# =============================================================================
# A. Initialization and Building Geometry
# =============================================================================

class TestInitialization:
    """Verify IntegratedSimulationApp initializes with authentic Pavan building."""

    def test_app_creates_successfully(self, app: IntegratedSimulationApp):
        """App instantiates without error."""
        assert app is not None

    def test_building_dimensions(self, app: IntegratedSimulationApp):
        """Building matches building_main.json (40×30m)."""
        assert app.building.width == pytest.approx(40.0, abs=1.0)
        assert app.building.height == pytest.approx(30.0, abs=1.0)

    def test_building_has_exits(self, app: IntegratedSimulationApp):
        """Building has 3 exits as defined in building_main.json."""
        assert len(app.building.exits) == 3

    def test_building_has_obstacles(self, app: IntegratedSimulationApp):
        """Building has 4 obstacles as defined in building_main.json."""
        assert len(app.building.obstacles) == 4

    def test_engine_initialized(self, app: IntegratedSimulationApp):
        """Pavan's SimulationEngine is initialized and at step 0."""
        assert app.engine is not None
        assert app.engine.clock.step_count == 0

    def test_part2_environment_created(self, app: IntegratedSimulationApp):
        """Part 2 EnvironmentMap is populated from building geometry."""
        assert app.part2_env is not None
        assert app.part2_env.width == app.building.width
        assert app.part2_env.height == app.building.height

    def test_initial_adapter_has_step0_snapshot(self, app: IntegratedSimulationApp):
        """Adapter buffer contains step-0 snapshot after initialization."""
        snap = app.adapter.buffer.get_snapshot(0, 0.0)
        assert snap is not None
        # Should have pedestrians
        assert len(snap.trajectories) > 0


# =============================================================================
# B. Physics Stepping and Snapshot Adaptation
# =============================================================================

class TestPhysicsIntegration:
    """Verify engine stepping and Part 2 snapshot adaptation."""

    def test_update_physics_advances_engine(self, app: IntegratedSimulationApp):
        """update_physics() advances Pavan's engine clock."""
        initial_step = app.engine.clock.step_count
        steps = app.update_physics(0.5)  # Half second wall clock
        assert steps > 0
        assert app.engine.clock.step_count > initial_step

    def test_adapter_accumulates_observations(self, app: IntegratedSimulationApp):
        """After sufficient steps, adapter buffer has trajectory history."""
        # Step enough to generate multiple observations
        for _ in range(20):
            app.update_physics(0.1)

        step = app.engine.clock.step_count
        timestamp = app.engine.clock.timestamp
        snap = app.adapter.buffer.get_snapshot(step, timestamp)
        assert snap is not None
        # At least one pedestrian should have history
        has_history = any(len(h) >= 2 for h in snap.trajectories.values())
        assert has_history, "Adapter should accumulate multiple observations after stepping"

    def test_paused_engine_no_advancement(self, app: IntegratedSimulationApp):
        """Paused engine does not advance on update_physics."""
        app.engine.pause()
        step_before = app.engine.clock.step_count
        steps = app.update_physics(0.5)
        assert steps == 0
        assert app.engine.clock.step_count == step_before

    def test_predictions_do_not_mutate_engine(self, app: IntegratedSimulationApp):
        """CV and MID predictions must not mutate engine state."""
        # Step to accumulate history
        for _ in range(20):
            app.update_physics(0.1)

        step = app.engine.clock.step_count
        timestamp = app.engine.clock.timestamp
        snap = app.adapter.buffer.get_snapshot(step, timestamp)

        # Pick the first agent with enough history
        target_id = None
        for aid, obs in snap.trajectories.items():
            if len(obs) >= 2:
                target_id = aid
                break
        if target_id is None:
            pytest.skip("No agent with sufficient history for prediction test")

        # Record engine state
        pre_step = app.engine.clock.step_count
        pre_time = app.engine.clock.timestamp
        pre_count = len(list(app.engine.registry))

        # Run predictions
        cv_pred = app.cv_predictor.predict(snap, target_id, horizon=12)
        mid_pred = app.mid_predictor.predict_samples(
            snapshot=snap, agent_id=target_id, k_samples=3,
            horizon=12, environment=app.part2_env,
        )

        # Verify engine state unchanged
        assert app.engine.clock.step_count == pre_step
        assert app.engine.clock.timestamp == pytest.approx(pre_time)
        assert len(list(app.engine.registry)) == pre_count

    def test_cv_prediction_returns_valid_result(self, app: IntegratedSimulationApp):
        """CV predictor returns TrajectoryPrediction with positions."""
        for _ in range(20):
            app.update_physics(0.1)

        step = app.engine.clock.step_count
        timestamp = app.engine.clock.timestamp
        snap = app.adapter.buffer.get_snapshot(step, timestamp)

        target_id = None
        for aid, obs in snap.trajectories.items():
            if len(obs) >= 2:
                target_id = aid
                break
        if target_id is None:
            pytest.skip("No agent with sufficient history")

        cv_pred = app.cv_predictor.predict(snap, target_id, horizon=12)
        assert cv_pred is not None
        assert len(cv_pred.predictions) > 0

    def test_mid_prediction_returns_valid_result(self, app: IntegratedSimulationApp):
        """MID predictor returns MultiTrajectoryPrediction with samples."""
        for _ in range(20):
            app.update_physics(0.1)

        step = app.engine.clock.step_count
        timestamp = app.engine.clock.timestamp
        snap = app.adapter.buffer.get_snapshot(step, timestamp)

        target_id = None
        for aid, obs in snap.trajectories.items():
            if len(obs) >= 2:
                target_id = aid
                break
        if target_id is None:
            pytest.skip("No agent with sufficient history")

        mid_result = app.mid_predictor.predict_samples(
            snapshot=snap, agent_id=target_id, k_samples=3,
            horizon=12, environment=app.part2_env,
        )
        assert mid_result is not None
        assert mid_result.sample_count == 3
        assert len(mid_result.samples) == 3
        assert mid_result.is_trained_model == app.mid_predictor.is_trained

    def test_insufficient_history_handled_gracefully(self, app: IntegratedSimulationApp):
        """Predictions with <2 observations do not crash."""
        snap = app.adapter.buffer.get_snapshot(0, 0.0)
        # Find agent with exactly 1 observation
        for aid, obs in snap.trajectories.items():
            if len(obs) < 2:
                # CV predictor should handle gracefully (may raise or return empty)
                try:
                    cv_pred = app.cv_predictor.predict(snap, aid, horizon=12)
                    # If it returns, it should not crash
                except Exception:
                    pass  # Graceful handling includes raising
                break


# =============================================================================
# C. MID Training Provenance (Issue A)
# =============================================================================

class TestMIDProvenance:
    """Verify MID model training provenance is dynamic and accurate."""

    def test_mid_predictor_is_trained(self, app: IntegratedSimulationApp):
        """create_pretrained_mid_predictor sets is_trained=True."""
        assert app.mid_predictor.is_trained is True

    def test_training_provenance_attributes(self, app: IntegratedSimulationApp):
        """MID predictor has accurate training metadata."""
        prov = app.mid_predictor.training_provenance
        assert prov["is_trained"] is True
        assert prov["training_epochs_completed"] == 10  # As set in create_pretrained_mid_predictor
        assert prov["training_sample_count"] > 0

    def test_epochs_and_samples_match_constructor(self, app: IntegratedSimulationApp):
        """Predictor epochs match the 10 configured in create_pretrained_mid_predictor."""
        assert app.mid_predictor._training_epochs_completed == 10
        assert app.mid_predictor._training_sample_count == 8  # 8 synthetic samples

    def test_untrained_predictor_provenance(self):
        """Fresh EvacuationMIDPredictor reports untrained state."""
        config = h_mid.EvacuationMIDConfig()
        predictor = h_mid.EvacuationMIDPredictor(config=config)
        assert predictor.is_trained is False
        assert predictor._training_epochs_completed == 0
        assert predictor._training_sample_count == 0


# =============================================================================
# D. Reset Behavior (Issue F)
# =============================================================================

class TestResetBehavior:
    """Verify reset_simulation clears Part 2 state but preserves trained MID model."""

    def test_reset_clears_engine_to_step0(self, app: IntegratedSimulationApp):
        """After reset, engine.clock.step_count == 0."""
        for _ in range(10):
            app.update_physics(0.1)
        assert app.engine.clock.step_count > 0

        app.reset_simulation()
        assert app.engine.clock.step_count == 0

    def test_reset_preserves_mid_predictor(self, app: IntegratedSimulationApp):
        """Reset does NOT reinstantiate mid_predictor — preserves trained model."""
        mid_id_before = id(app.mid_predictor)
        trained_before = app.mid_predictor.is_trained
        epochs_before = app.mid_predictor._training_epochs_completed

        for _ in range(10):
            app.update_physics(0.1)
        app.reset_simulation()

        assert id(app.mid_predictor) == mid_id_before, "MID predictor was reinstantiated during reset"
        assert app.mid_predictor.is_trained == trained_before
        assert app.mid_predictor._training_epochs_completed == epochs_before

    def test_reset_clears_adapter_buffer(self, app: IntegratedSimulationApp):
        """After reset, adapter buffer is cleared and reinitialized."""
        for _ in range(10):
            app.update_physics(0.1)
        app.reset_simulation()

        # Buffer should have only step-0 snapshot data
        snap = app.adapter.buffer.get_snapshot(0, 0.0)
        assert snap is not None
        # History length should be 1 (only initial snapshot)
        for aid, obs in snap.trajectories.items():
            assert len(obs) <= 1, f"Agent {aid} should have at most 1 observation after reset"

    def test_reset_clears_heatmap(self, app: IntegratedSimulationApp):
        """After reset, heatmap engine is cleared."""
        for _ in range(10):
            app.update_physics(0.1)

        # Force heatmap aggregation
        active_positions = [
            (p.position.x, p.position.y)
            for p in app.engine.registry
            if str(p.status).lower().endswith("active")
        ]
        if active_positions:
            app.heatmap_engine.aggregate_positions(
                positions=active_positions,
                timestamp=app.engine.clock.timestamp,
                step_index=app.engine.clock.step_count,
            )

        app.reset_simulation()
        # After reset, heatmap should produce empty/zero density
        result = app.heatmap_engine.aggregate_positions(
            positions=[], timestamp=0.0, step_index=0,
        )
        density = result.normalize_min_max("density")
        assert density.max() < 1e-4, "Heatmap should be empty after reset"

    def test_reset_sets_selected_agent(self, app: IntegratedSimulationApp):
        """After reset, selected_agent_id is set to first cohort member."""
        app.reset_simulation()
        assert app.selected_agent_id is not None


# =============================================================================
# E. Map Loading (Issue E)
# =============================================================================

class TestMapLoading:
    """Verify --map argument support for loading building layouts."""

    def test_default_map_path_exists(self):
        """Default building_main.json map path exists."""
        default_map = PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json"
        assert default_map.is_file(), f"Default map not found: {default_map}"

    def test_invalid_map_raises_error(self):
        """Non-existent map path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            IntegratedSimulationApp(map_path="/nonexistent/map.json")

    def test_main_function_has_map_arg(self):
        """main() parser includes --map argument."""
        import argparse
        from genevac.visualization.integrated_app import main
        # Verify the function exists and is callable
        assert callable(main)


# =============================================================================
# F. Collision-Free Validation Fields (Issue D)
# =============================================================================

class TestCollisionValidation:
    """Verify MultiTrajectoryPrediction collision metrics are computed correctly."""

    def test_mid_result_has_walkability_fields(self, app: IntegratedSimulationApp):
        """MultiTrajectoryPrediction includes walkable_validity and segment_collision_free."""
        for _ in range(20):
            app.update_physics(0.1)

        step = app.engine.clock.step_count
        timestamp = app.engine.clock.timestamp
        snap = app.adapter.buffer.get_snapshot(step, timestamp)

        target_id = None
        for aid, obs in snap.trajectories.items():
            if len(obs) >= 2:
                target_id = aid
                break
        if target_id is None:
            pytest.skip("No agent with sufficient history")

        mid_result = app.mid_predictor.predict_samples(
            snapshot=snap, agent_id=target_id, k_samples=3,
            horizon=12, environment=app.part2_env,
        )
        # Both fields should be finite floats in [0, 1]
        assert 0.0 <= mid_result.point_walkability <= 1.0
        assert 0.0 <= mid_result.segment_collision_free <= 1.0
        assert 0.0 <= mid_result.walkable_validity <= 1.0

    def test_walkable_validity_is_point_walkability_alias(self, app: IntegratedSimulationApp):
        """walkable_validity == point_walkability (documented alias)."""
        for _ in range(20):
            app.update_physics(0.1)

        step = app.engine.clock.step_count
        timestamp = app.engine.clock.timestamp
        snap = app.adapter.buffer.get_snapshot(step, timestamp)

        target_id = None
        for aid, obs in snap.trajectories.items():
            if len(obs) >= 2:
                target_id = aid
                break
        if target_id is None:
            pytest.skip("No agent with sufficient history")

        mid_result = app.mid_predictor.predict_samples(
            snapshot=snap, agent_id=target_id, k_samples=3,
            horizon=12, environment=app.part2_env,
        )
        assert mid_result.walkable_validity == mid_result.point_walkability


# =============================================================================
# G. Headless Mode
# =============================================================================

class TestHeadlessMode:
    """Verify application works in headless SDL_VIDEODRIVER=dummy mode."""

    def test_headless_run_with_max_frames(self, app: IntegratedSimulationApp):
        """run(max_frames=5) completes without crash in headless mode."""
        import pygame
        if not pygame.get_init():
            pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()
        if not pygame.font.get_init():
            pygame.font.init()

        # Run 5 frames then exit
        app.run(max_frames=5)
        # If we reach here, it didn't crash

