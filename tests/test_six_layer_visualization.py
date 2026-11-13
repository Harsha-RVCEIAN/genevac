"""Unit, integration, and causal traceability tests for Six-Layer Visualization.

Verifies:
1. All six layers source from their authentic Part 1 / Part 2 components without fabrication.
2. Stable candidate IDs (C0..C{K-1}), rejection reasons, and controller-selected candidate binding.
3. Explicit empty states ("No MID candidate selected", "NOT_ADOPTED") when no candidate or route is adopted.
4. Rejected/proposed routes are never rendered as adopted green engine routes.
5. Non-finite coordinates, missing data, and visual toggles never crash or mutate simulation state.
6. End-to-end causal chain from MID candidate generation through engine adoption and subsequent step movement.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import pytest

from genevac.prediction.data_contract import (
    PredictedPosition,
    TrajectoryPrediction,
    TrajectorySnapshot,
)
from genevac.prediction.mid import MultiTrajectoryPrediction
from genevac.visualization.integrated_app import (
    IntegratedSimulationApp,
    OperatingMode,
    PAVAN,
    PROJECT_ROOT,
)


@pytest.fixture(scope="module")
def shared_mid():
    """Share one pre-trained MID predictor across this test module for fast deterministic execution."""
    map_path = PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json"
    if not map_path.is_file():
        pytest.skip(f"Required Part 1 reference map missing: {map_path}")
    base_app = IntegratedSimulationApp(map_path=str(map_path), width=1440, height=860, fps=30)
    return base_app.mid_predictor


@pytest.fixture
def app(shared_mid):
    """Create a headless IntegratedSimulationApp with surface and fonts initialized."""
    map_path = PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json"
    if not map_path.is_file():
        pytest.skip(f"Required Part 1 reference map missing: {map_path}")

    pygame.init()
    pygame.font.init()

    inst = IntegratedSimulationApp(map_path=str(map_path), width=1440, height=860, fps=30)
    inst.mid_predictor = shared_mid
    inst.screen = pygame.Surface((inst.width, inst.height))
    inst._ensure_fonts()
    return inst


def _advance(app: IntegratedSimulationApp, steps: int = 10) -> None:
    for _ in range(steps):
        app.update_physics(0.1)


def _render_all(app: IntegratedSimulationApp) -> None:
    assert app.screen is not None
    app.renderer.draw(app.screen, app.building, app.viewport, engine=app.engine)
    app._render_spatial_heatmap(app.screen)
    app.renderer._draw_pedestrians(app.screen, app.engine.registry, app.viewport)
    app._render_predictions(app.screen)
    app._render_hud_and_telemetry(app.screen)


class TestSixLayerUnitContracts:
    def test_01_actual_path_uses_engine_recorded_positions(self, app: IntegratedSimulationApp):
        assert app.screen is not None
        app.show_mid_predictions = False
        app.show_baseline_predictions = False
        _advance(app, steps=10)

        ped = app.engine.registry.get("ped_1")
        assert ped is not None
        recorded = app.actual_traversed_paths["ped_1"]
        assert len(recorded) == 11  # steps 0..10
        assert [pt[0] for pt in recorded] == list(range(11))
        assert recorded[-1][2] == pytest.approx(ped.position.x)
        assert recorded[-1][3] == pytest.approx(ped.position.y)

        _render_all(app)
        l1 = app.latest_layer_states["actual_movement"]
        assert l1["rendered"] is True
        assert l1["path_drawn"] is True
        assert l1["marker_drawn"] is True
        assert l1["path_points"] == 11
        assert l1["pedestrian_id"] == "ped_1"

    def test_02_observed_history_uses_adapter_no_current_duplication(self, app: IntegratedSimulationApp):
        assert app.screen is not None
        # 9 steps -> t=0.9s; adapter recorded at 0.0, 0.4, 0.8 (3 observations)
        _advance(app, steps=9)
        assert app.engine.clock.step_count == 9

        snap = app.adapter.buffer.get_snapshot(app.engine.clock.step_count, app.engine.clock.timestamp)
        clean_hist = app._prepare_clean_observation_history(snap, "ped_1")
        assert len(clean_hist) == 3
        assert [o.step_index for o in clean_hist] == [0, 4, 8]
        assert [round(o.timestamp, 4) for o in clean_hist] == [0.0, 0.4, 0.8]

        ped = app.engine.registry.get("ped_1")
        assert ped is not None
        # Step 9 live position must NOT be artificially inserted into adapter history
        assert clean_hist[-1].step_index == 8
        assert math.hypot(ped.position.x - clean_hist[-1].x, ped.position.y - clean_hist[-1].y) > 0.01

        _render_all(app)
        l2 = app.latest_layer_states["observed_history"]
        assert l2["rendered"] is True
        assert l2["count"] == 3
        assert [round(t, 4) for t in l2["timestamps"]] == [0.0, 0.4, 0.8]

    def test_03_cv_layer_uses_baseline_prediction_output(self, app: IntegratedSimulationApp):
        assert app.screen is not None
        _advance(app, steps=8)
        snap = app.adapter.buffer.get_snapshot(8, 0.8)
        expected_cv = app.cv_predictor.predict(snap, "ped_1", horizon=app.prediction_horizon)

        app.show_baseline_predictions = False
        app.renderer.draw(app.screen, app.building, app.viewport, engine=app.engine)
        app._render_predictions(app.screen)
        without_cv = pygame.surfarray.array3d(app.screen).copy()

        app.show_baseline_predictions = True
        app.renderer.draw(app.screen, app.building, app.viewport, engine=app.engine)
        app._render_predictions(app.screen)
        with_cv = pygame.surfarray.array3d(app.screen).copy()

        assert (without_cv != with_cv).any()
        assert app._cached_cv_result is not None
        assert len(app._cached_cv_result.predictions) == len(expected_cv.predictions)
        for p_cached, p_exp in zip(app._cached_cv_result.predictions, expected_cv.predictions):
            assert p_cached.x == pytest.approx(p_exp.x)
            assert p_cached.y == pytest.approx(p_exp.y)
        assert app.latest_layer_states["cv_prediction"]["rendered"] is True

    def test_04_mid_layer_renders_all_k_candidates(self, app: IntegratedSimulationApp):
        assert app.screen is not None
        app.operating_mode = OperatingMode.MID_PREDICTION_ONLY
        _advance(app, steps=8)

        _render_all(app)
        l4_6 = app.latest_layer_states["mid_candidates"]
        assert l4_6["rendered"] is True
        assert l4_6["count"] == 6
        assert l4_6["candidate_ids"] == ("C0", "C1", "C2", "C3", "C4", "C5")

        # Reduce K via [DOWN] to 5 and confirm K=5 is faithfully represented
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN))
        _render_all(app)
        l4_5 = app.latest_layer_states["mid_candidates"]
        assert l4_5["count"] == 5
        assert l4_5["candidate_ids"] == ("C0", "C1", "C2", "C3", "C4")

    def test_05_06_candidate_identity_and_non_zero_controller_selection(self, app: IntegratedSimulationApp):
        """Controller selection of non-zero sample index (C1) is rendered, never defaulting to C0."""
        assert app.screen is not None
        app.operating_mode = OperatingMode.MID_ASSISTED_EVACUATION
        app.k_samples = 2
        _advance(app, steps=8)

        ped = app.engine.registry.get("ped_1")
        assert ped is not None
        cx, cy = ped.position.x, ped.position.y

        bad_preds = tuple(PredictedPosition(timestamp=0.8 + (i + 1) * 0.1, x=-5.0, y=-5.0) for i in range(12))
        good_preds = tuple(
            PredictedPosition(
                timestamp=0.8 + (i + 1) * 0.1,
                x=cx + (10.0 - cx) * ((i + 1) / 14.0),
                y=cy + (12.0 - cy) * ((i + 1) / 14.0) + 0.1,
            )
            for i in range(12)
        )
        s0 = TrajectoryPrediction("ped_1", 8, 0.8, bad_preds)
        s1 = TrajectoryPrediction("ped_1", 8, 0.8, good_preds)
        multi = MultiTrajectoryPrediction(
            agent_id="ped_1",
            source_step_index=8,
            source_timestamp=0.8,
            samples=(s0, s1),
            mean_prediction=s1,
            sample_diversity=4.0,
            point_walkability=0.5,
            segment_collision_free=0.5,
            walkable_validity=0.5,
            is_trained_model=True,
            sample_count=2,
        )

        # Clear any rate limit from step 8 auto-eval to evaluate this exact 2-sample batch at step 8
        app.controller.last_decision_time.clear()
        rec = app.controller.evaluate_and_recommend(
            agent_id="ped_1",
            current_pedestrian=ped,
            mid_prediction=multi,
            open_exit_ids=[ex.id for ex in app.building.exits if ex.is_open],
            current_time=0.8,
            step_index=8,
            generate_route_fn=PAVAN["generate_deterministic_route"],
            is_walkable_fn=PAVAN["is_position_in_walkable_space"],
            check_collision_fn=PAVAN["check_collision"],
            check_overlap_fn=PAVAN["check_pedestrian_overlap"],
            other_pedestrians=list(app.engine.registry),
        )
        assert rec.candidates[0].candidate_id == "C0"
        assert rec.candidates[1].candidate_id == "C1"
        assert rec.selected_sample_idx == 1
        assert rec.selected_candidate_id == "C1"
        assert "C0" in rec.rejection_summary()

        app._mid_cache_map[("ped_1", 8, 2, app.prediction_horizon)] = multi
        _render_all(app)

        l4 = app.latest_layer_states["mid_candidates"]
        l5 = app.latest_layer_states["selected_mid_candidate"]
        assert l4["candidate_ids"] == ("C0", "C1")
        assert l4["rejected"][0][0] == "C0"
        assert l5["status"] == "SELECTED"
        assert l5["selected_candidate_id"] == "C1"
        assert l5["sample_idx"] == 1

    def test_07_08_10_all_rejected_empty_selection_and_never_adopted(self, app: IntegratedSimulationApp):
        """When controller rejects all candidates: explicit 'No MID candidate selected', reasons exposed, and NOT rendered as adopted green route."""
        assert app.screen is not None
        app.operating_mode = OperatingMode.MID_ASSISTED_EVACUATION
        app.k_samples = 2
        _advance(app, steps=8)

        ped = app.engine.registry.get("ped_1")
        assert ped is not None
        # Deliberately feed all-invalid predictions through real predictor mock helper ONLY on app
        bad_preds = tuple(PredictedPosition(timestamp=0.8 + (i + 1) * 0.1, x=-10.0, y=-10.0) for i in range(12))
        s0 = TrajectoryPrediction("ped_1", 8, 0.8, bad_preds)
        s1 = TrajectoryPrediction("ped_1", 8, 0.8, bad_preds)
        all_bad_multi = MultiTrajectoryPrediction(
            agent_id="ped_1",
            source_step_index=8,
            source_timestamp=0.8,
            samples=(s0, s1),
            mean_prediction=s0,
            sample_diversity=0.0,
            point_walkability=0.0,
            segment_collision_free=0.0,
            walkable_validity=0.0,
            is_trained_model=True,
            sample_count=2,
        )

        # Force controller evaluation on this rejected batch at step 12
        app.controller.last_decision_time.clear()
        app._mid_cache_map[("ped_1", 12, 2, app.prediction_horizon)] = MultiTrajectoryPrediction(
            agent_id="ped_1",
            source_step_index=12,
            source_timestamp=1.2,
            samples=(
                TrajectoryPrediction("ped_1", 12, 1.2, tuple(PredictedPosition(timestamp=1.2 + (i + 1) * 0.1, x=-10.0, y=-10.0) for i in range(12))),
                TrajectoryPrediction("ped_1", 12, 1.2, tuple(PredictedPosition(timestamp=1.2 + (i + 1) * 0.1, x=-11.0, y=-11.0) for i in range(12))),
            ),
            mean_prediction=s0,
            sample_diversity=0.0,
            point_walkability=0.0,
            segment_collision_free=0.0,
            walkable_validity=0.0,
            is_trained_model=True,
            sample_count=2,
        )
        # Step to 12 so _apply_mid_assisted_guidance consumes the cached all-bad candidates cleanly
        _advance(app, steps=4)

        rec = app.controller.latest_recommendations_by_agent.get("ped_1")
        assert rec is not None
        assert rec.is_accepted is False
        assert rec.selected_sample_idx == -1
        assert rec.selected_candidate_id is None
        assert len(rec.rejected_candidates) == 2
        assert len(rec.proposed_waypoints) > 0  # proposed_waypoints contains baseline fallback, NOT accepted!
        assert len(ped.waypoints) > 0  # baseline engine waypoints still exist!

        adopted_rec = app.adopted_route_records_by_agent.get("ped_1")
        assert adopted_rec is not None
        assert adopted_rec.adoption_succeeded is False
        assert adopted_rec.adoption_status == "NOT_ADOPTED_REJECTED"

        # Toggle Layer 6 and ensure pixel output does NOT draw an adopted green route when rejected
        app.show_adopted_route = False
        _render_all(app)
        px_off = pygame.surfarray.array3d(app.screen).copy()

        app.show_adopted_route = True
        _render_all(app)
        px_on = pygame.surfarray.array3d(app.screen).copy()

        assert not (px_off != px_on).any()
        assert app.latest_layer_states["selected_mid_candidate"]["status"] == "NO_CANDIDATE_SELECTED"
        expected_rejected = tuple((cid, rsn) for cid, rsn in rec.rejection_summary().items())
        assert len(expected_rejected) == 2
        assert app.latest_layer_states["mid_candidates"]["rejected"] == expected_rejected
        assert app.latest_layer_states["adopted_route"]["rendered"] is False
        assert app.latest_layer_states["adopted_route"]["status"] == "NOT_ADOPTED"

        del all_bad_multi

    def test_09_adopted_route_reads_engine_waypoints_after_modification(self, app: IntegratedSimulationApp):
        assert app.screen is not None
        app.operating_mode = OperatingMode.MID_ASSISTED_EVACUATION
        _advance(app, steps=8)

        ped = app.engine.registry.get("ped_1")
        assert ped is not None
        adopted_rec = app.adopted_route_records_by_agent.get("ped_1")
        assert adopted_rec is not None and adopted_rec.adoption_succeeded is True

        # Engine advances and modifies/truncates ped.waypoints (simulating popped front waypoint)
        assert len(ped.waypoints) >= 2
        ped.waypoints.pop(0)
        expected_live_wps = tuple((wp.x, wp.y) for wp in ped.waypoints)

        _render_all(app)
        l6 = app.latest_layer_states["adopted_route"]
        assert l6["rendered"] is True
        assert l6["waypoints"] == expected_live_wps
        assert l6["waypoints"] != adopted_rec.proposed_waypoints

    def test_11_12_missing_and_non_finite_coordinates_handled_safely(self, app: IntegratedSimulationApp):
        assert app.screen is not None
        # Empty / <2 obs step 0
        _render_all(app)
        assert app.latest_layer_states["cv_prediction"]["rendered"] is False

        # Missing selected agent & None selected agent
        app.selected_agent_id = None
        _render_all(app)
        assert app.latest_layer_states == {}

        app.selected_agent_id = "missing_ped_xyz"
        _render_all(app)
        assert app.latest_layer_states["actual_movement"]["status"] == "MISSING"
        assert app.latest_layer_states["adopted_route"]["status"] == "UNAVAILABLE"

        # Non-finite coordinates safety
        assert app._safe_screen_point((float("nan"), 5.0)) is None
        assert app._safe_screen_point((10.0, float("inf"))) is None
        app.selected_agent_id = "ped_1"
        _advance(app, steps=8)
        app.actual_traversed_paths["ped_1"].append((99, 9.9, float("nan"), float("inf")))
        bad_obs = [
            SimpleNamespace(step_index=0, timestamp=0.0, x=8.0, y=6.0),
            SimpleNamespace(step_index=4, timestamp=0.4, x=float("nan"), y=6.0),
            SimpleNamespace(step_index=8, timestamp=0.8, x=7.2, y=6.0),
        ]
        fake_snap = TrajectorySnapshot(
            current_step=8,
            current_timestamp=0.8,
            timestep=0.1,
            trajectories={},
        )
        app._render_observed_history_layer(app.screen, bad_obs)  # type: ignore[arg-type]
        app._render_actual_movement_layer(app.screen, app.engine.registry.get("ped_1"), "ped_1")
        assert app.latest_layer_states["actual_movement"]["marker_drawn"] is True
        del fake_snap

    def test_13_rendering_toggles_do_not_alter_simulation_state(self, shared_mid):
        map_path = str(PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json")
        app_ref = IntegratedSimulationApp(map_path=map_path, width=1440, height=860, fps=30)
        app_ref.mid_predictor = shared_mid
        app_ref.operating_mode = OperatingMode.MID_ASSISTED_EVACUATION

        app_toggled = IntegratedSimulationApp(map_path=map_path, width=1440, height=860, fps=30)
        app_toggled.mid_predictor = shared_mid
        app_toggled.operating_mode = OperatingMode.MID_ASSISTED_EVACUATION
        pygame.init()
        pygame.font.init()
        app_toggled.screen = pygame.Surface((app_toggled.width, app_toggled.height))
        app_toggled._ensure_fonts()

        toggle_keys = [pygame.K_t, pygame.K_o, pygame.K_a, pygame.K_l, pygame.K_p, pygame.K_b, pygame.K_h]
        for idx in range(20):
            app_ref.update_physics(0.1)
            app_toggled.update_physics(0.1)
            key = toggle_keys[idx % len(toggle_keys)]
            app_toggled.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
            _render_all(app_toggled)
            app_toggled.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
            _render_all(app_toggled)

        assert app_ref.engine.clock.step_count == app_toggled.engine.clock.step_count
        assert app_ref.engine.clock.timestamp == pytest.approx(app_toggled.engine.clock.timestamp)
        assert app_ref.controller.accepted_interventions == app_toggled.controller.accepted_interventions
        for p_ref in app_ref.engine.registry:
            p_tog = app_toggled.engine.registry.get(p_ref.id)
            assert p_tog is not None
            assert math.hypot(p_ref.position.x - p_tog.position.x, p_ref.position.y - p_tog.position.y) == 0.0
            assert [(w.x, w.y) for w in p_ref.waypoints] == [(w.x, w.y) for w in p_tog.waypoints]


class TestSixLayerIntegrationAndCausalChain:
    def test_14_all_six_layers_simultaneous_in_assisted_mode(self, app: IntegratedSimulationApp):
        assert app.screen is not None
        app.operating_mode = OperatingMode.MID_ASSISTED_EVACUATION
        _advance(app, steps=16)

        _render_all(app)
        states = app.latest_layer_states
        assert states["actual_movement"]["rendered"] is True
        assert states["observed_history"]["rendered"] is True
        assert states["cv_prediction"]["rendered"] is True
        assert states["mid_candidates"]["rendered"] is True
        assert states["mid_candidates"]["count"] == 6
        assert states["selected_mid_candidate"]["rendered"] is True
        assert states["selected_mid_candidate"]["status"] == "SELECTED"
        assert states["adopted_route"]["rendered"] is True
        assert states["adopted_route"]["status"].startswith("ADOPTED")

    def test_15_prediction_only_shows_history_cv_mid_without_selection_or_adoption(self, shared_mid):
        map_path = str(PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json")
        app_orig = IntegratedSimulationApp(map_path=map_path, width=1440, height=860, fps=30)
        app_orig.mid_predictor = shared_mid
        app_orig.operating_mode = OperatingMode.ORIGINAL_SIMULATION

        app_pred = IntegratedSimulationApp(map_path=map_path, width=1440, height=860, fps=30)
        app_pred.mid_predictor = shared_mid
        app_pred.operating_mode = OperatingMode.MID_PREDICTION_ONLY
        pygame.init()
        pygame.font.init()
        app_pred.screen = pygame.Surface((app_pred.width, app_pred.height))
        app_pred._ensure_fonts()

        _advance(app_orig, steps=16)
        _advance(app_pred, steps=16)
        _render_all(app_pred)

        # Movement matches baseline 100% identically
        for po in app_orig.engine.registry:
            pp = app_pred.engine.registry.get(po.id)
            assert pp is not None
            assert math.hypot(po.position.x - pp.position.x, po.position.y - pp.position.y) == 0.0

        states = app_pred.latest_layer_states
        assert states["actual_movement"]["rendered"] is True
        assert states["observed_history"]["rendered"] is True
        assert states["cv_prediction"]["rendered"] is True
        assert states["mid_candidates"]["rendered"] is True
        assert states["selected_mid_candidate"]["status"] == "NO_CANDIDATE_SELECTED"
        assert states["adopted_route"]["status"] == "NOT_ADOPTED"
        assert states["adopted_route"]["rendered"] is False

    def test_16_17_end_to_end_causal_chain_trace_and_subsequent_engine_movement(self, app: IntegratedSimulationApp):
        """Verify complete causal link: MID -> selection -> proposed -> adopted -> subsequent engine step movement."""
        assert app.screen is not None
        app.operating_mode = OperatingMode.MID_ASSISTED_EVACUATION
        # Advance 14 steps: observation ticks occur at steps 4, 8, 12 -> followed by engine steps 9..14
        _advance(app, steps=14)
        _render_all(app)

        p1_traces = [e for e in app.causal_trace_log if e["agent_id"] == "ped_1"]
        assert len(p1_traces) >= 1
        latest_trace = p1_traces[-1]

        assert latest_trace["mid_candidate_count"] == 6
        assert latest_trace["selected_candidate_id"] == f"C{latest_trace['selected_sample_idx']}"
        assert latest_trace["selected_endpoint"] is not None
        assert len(latest_trace["proposed_waypoints"]) >= 1
        assert latest_trace["adoption_succeeded"] is True
        assert latest_trace["adopted_waypoints"] == latest_trace["proposed_waypoints"]
        assert latest_trace["post_step_index"] > latest_trace["decision_step"]
        assert latest_trace["moved_distance_m"] > 0.0

        # Ensure step distance respected speed limits in post-adoption step
        ped = app.engine.registry.get("ped_1")
        assert ped is not None
        assert latest_trace["moved_distance_m"] <= ped.preferred_speed * app.dt + 1e-5

        # Layer 5 & Layer 6 are semantically distinct: selected MID sample has 12 horizon steps while adopted route is engine waypoints
        sel_idx = latest_trace["selected_sample_idx"]
        mid_key = ("ped_1", latest_trace["decision_step"], app.k_samples, app.prediction_horizon)
        if mid_key in app._mid_cache_map:
            sel_sample = app._mid_cache_map[mid_key].samples[sel_idx]
            sel_coords = tuple((p.x, p.y) for p in sel_sample.predictions)
            assert sel_coords != latest_trace["adopted_waypoints"]
