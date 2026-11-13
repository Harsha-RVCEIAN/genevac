"""Comprehensive verification suite for Social MID Learning, MID-DRIVEN Mode, and Safety Layer.

Covers:
1. Social conditioning & summary extraction across interaction regimes + disjoint corpus.
2. Learned social interaction sensitivity & ablation (`use_social_encoding` + social_encoder).
3. Genuine `MID-DRIVEN` divergence from `ORIGINAL` starting from identical initial states.
4. `ORIGINAL` deterministic preservation and `[M]` mode round-trip route cleanup.
5. Static wall/obstacle collision blocking via `CollisionAvoidanceSafetyLayer`.
6. Swept-segment moving neighbor collision & near-collision avoidance.
7. Anti-deadlock stall detection and lateral escape recovery.
8. Keyboard `[M]` mode cycling + `[I]`/`[C]`/`[G]` read-only overlay toggle invariance.
9. Live HUD, social link/halo overlay, safety overlay, and Original ghost comparison rendering.
10. End-to-end deterministic repeatability across identical seeds in `MID-DRIVEN` mode.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Dict, List, Tuple

import pygame
import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"

from genevac.prediction.controller import (
    CollisionAvoidanceSafetyLayer,
    MIDDrivenMovementController,
    _continuous_segment_min_distance,
)
from genevac.prediction.data_contract import TrajectoryObservation, TrajectorySnapshot
from genevac.prediction.mid import (
    SOCIAL_FEATURE_DIM,
    EvacuationConditioning,
    EvacuationMIDConfig,
    EvacuationMIDPredictor,
)
from genevac.prediction.social_training import (
    TRAINING_INTERACTION_SCENARIOS,
    TRAINING_SOURCE_NAME,
    generate_social_interaction_training_samples,
)
from genevac.visualization.integrated_app import (
    IntegratedSimulationApp,
    OperatingMode,
    PROJECT_ROOT,
    PAVAN,
)


@pytest.fixture(scope="module")
def shared_mid() -> EvacuationMIDPredictor:
    """One shared pretrained social MID predictor to keep the suite fast and deterministic."""
    map_path = PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json"
    app = IntegratedSimulationApp(map_path=str(map_path), width=1440, height=860, fps=30)
    assert app.mid_predictor is not None
    return app.mid_predictor


@pytest.fixture
def make_app(shared_mid: EvacuationMIDPredictor):
    map_path = str(PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json")

    def _factory(seed: int = 42, mode: OperatingMode = OperatingMode.ORIGINAL_SIMULATION) -> IntegratedSimulationApp:
        app = IntegratedSimulationApp(
            map_path=map_path,
            width=1440,
            height=860,
            fps=30,
            mid_predictor=shared_mid,
        )
        app.base_seed = seed
        app.operating_mode = mode
        return app

    return _factory


def _make_obs_seq(agent_id: str, xs: List[float], ys: List[float], dt: float = 0.1) -> Tuple[TrajectoryObservation, ...]:
    return tuple(
        TrajectoryObservation(
            agent_id=agent_id,
            step_index=i + 1,
            timestamp=round((i + 1) * dt, 4),
            x=float(x),
            y=float(y),
        )
        for i, (x, y) in enumerate(zip(xs, ys))
    )


def _snapshot_from_agents(trajs: Dict[str, Tuple[TrajectoryObservation, ...]]) -> TrajectorySnapshot:
    max_t = max(obs.timestamp for seq in trajs.values() for obs in seq)
    max_s = max(obs.step_index for seq in trajs.values() for obs in seq)
    return TrajectorySnapshot(current_step=max_s, current_timestamp=max_t, timestep=0.1, trajectories=trajs)


class TestSocialConditioningAndTrainingCorpus:
    def test_01_social_conditioning_regimes_and_disjoint_corpus(self):
        """Social feature extraction classifies NONE, FOLLOWING, AVOIDANCE, CONGESTION, MERGING, OBSTRUCTED."""
        ego_seq = _make_obs_seq("ego", [10.0, 10.14, 10.28, 10.42], [10.0, 10.0, 10.0, 10.0])

        # Clear corridor
        snap_clear = _snapshot_from_agents({"ego": ego_seq})
        cond_clear = EvacuationConditioning.from_snapshot_and_env(snap_clear, "ego", environment=None)
        assert len(cond_clear.vector) == 16
        assert len(cond_clear.social_vector) == SOCIAL_FEATURE_DIM
        assert len(cond_clear.normalized_social_vector) == SOCIAL_FEATURE_DIM
        assert cond_clear.social_summary is not None
        assert cond_clear.social_summary.dominant_relation == "NONE"
        assert cond_clear.social_summary.nearby_count == 0

        # Slower leader directly ahead -> FOLLOWING
        leader_seq = _make_obs_seq("leader", [11.3, 11.35, 11.40, 11.45], [10.02, 10.02, 10.02, 10.02])
        snap_follow = _snapshot_from_agents({"ego": ego_seq, "leader": leader_seq})
        cond_follow = EvacuationConditioning.from_snapshot_and_env(snap_follow, "ego", environment=None)
        # Static ego/obstacle/wall features (0..11, 14..15) stay identical; density (12) rises, nearest dist (13) drops
        assert cond_follow.vector[:12] == cond_clear.vector[:12]
        assert cond_follow.vector[14:] == cond_clear.vector[14:]
        assert cond_follow.vector[12] > cond_clear.vector[12]
        assert cond_follow.vector[13] < cond_clear.vector[13]
        assert cond_follow.social_summary is not None
        assert cond_follow.social_summary.dominant_relation == "FOLLOWING"
        assert cond_follow.social_summary.nearby_count == 1
        assert cond_follow.social_summary.nearest_neighbor_id == "leader"

        # Opposing head-on pedestrian -> AVOIDANCE
        opp_seq = _make_obs_seq("opp", [12.1, 11.96, 11.82, 11.68], [10.05, 10.05, 10.05, 10.05])
        snap_avoid = _snapshot_from_agents({"ego": ego_seq, "opp": opp_seq})
        cond_avoid = EvacuationConditioning.from_snapshot_and_env(snap_avoid, "ego", environment=None)
        assert cond_avoid.social_summary is not None
        assert cond_avoid.social_summary.dominant_relation == "AVOIDANCE"

        # Stationary obstacle-pedestrian ahead -> OBSTRUCTED
        obs_seq = _make_obs_seq("blk", [11.2, 11.2, 11.2, 11.2], [10.0, 10.0, 10.0, 10.0])
        snap_obs = _snapshot_from_agents({"ego": ego_seq, "blk": obs_seq})
        cond_obs = EvacuationConditioning.from_snapshot_and_env(snap_obs, "ego", environment=None)
        assert cond_obs.social_summary is not None
        assert cond_obs.social_summary.dominant_relation == "OBSTRUCTED"

        # Oblique converging neighbor -> MERGING (same general forward hemisphere, non-trivial crossing angle)
        merge_seq = _make_obs_seq("mrg", [10.80, 10.90, 11.00, 11.10], [10.75, 10.65, 10.55, 10.45])
        snap_merge = _snapshot_from_agents({"ego": ego_seq, "mrg": merge_seq})
        cond_merge = EvacuationConditioning.from_snapshot_and_env(snap_merge, "ego", environment=None)
        assert cond_merge.social_summary is not None
        assert cond_merge.social_summary.dominant_relation == "MERGING"

        # Crowd cluster (>=4 nearby agents) -> CONGESTION
        crowd_trajs = {"ego": ego_seq}
        for idx, (ox, oy) in enumerate([(11.5, 10.5), (11.5, 9.5), (9.8, 10.8), (9.8, 9.2), (12.0, 10.0)]):
            crowd_trajs[f"c_{idx}"] = _make_obs_seq(f"c_{idx}", [ox, ox + 0.04, ox + 0.08, ox + 0.12], [oy] * 4)
        cond_cong = EvacuationConditioning.from_snapshot_and_env(_snapshot_from_agents(crowd_trajs), "ego", environment=None)
        assert cond_cong.social_summary is not None
        assert cond_cong.social_summary.dominant_relation == "CONGESTION"
        assert cond_cong.social_summary.nearby_count == 5

        # Training corpus check: non-empty multi-agent neighbors & full scenario provenance
        samples, scenarios = generate_social_interaction_training_samples(horizon=12, history_length=4, dt=0.1)
        assert len(samples) >= 40
        assert tuple(scenarios) == TRAINING_INTERACTION_SCENARIOS
        assert any(len(s.neighbor_trajectories) > 0 for s in samples)

    def test_02_learned_social_sensitivity_vs_ablation(self, shared_mid: EvacuationMIDPredictor):
        """Trained social encoder alters predicted trajectories under social interactions without rule guidance."""
        prov = shared_mid.training_provenance
        assert prov["has_social_interactions"] is True
        assert prov["training_source"] == TRAINING_SOURCE_NAME
        assert len(prov["interaction_scenarios"]) >= 5

        ego_seq = _make_obs_seq("ego", [10.0, 10.14, 10.28, 10.42], [10.0, 10.0, 10.0, 10.0])
        slow_leader = _make_obs_seq("leader", [11.2, 11.24, 11.28, 11.32], [10.0, 10.0, 10.0, 10.0])
        # Same radial distance (0.90m) behind ego -> identical isotropic local_density in base 16-dim vector!
        behind_walker = _make_obs_seq("behind", [9.10, 9.24, 9.38, 9.52], [10.0, 10.0, 10.0, 10.0])
        opp_walker = _make_obs_seq("opp", [12.0, 11.86, 11.72, 11.58], [10.04, 10.04, 10.04, 10.04])

        snap_clear = _snapshot_from_agents({"ego": ego_seq})
        snap_follow = _snapshot_from_agents({"ego": ego_seq, "leader": slow_leader})
        snap_behind = _snapshot_from_agents({"ego": ego_seq, "behind": behind_walker})
        snap_avoid = _snapshot_from_agents({"ego": ego_seq, "opp": opp_walker})

        # Pure neural prediction with environment=None (zero wall/exit/safety rule bias)
        pred_clear = shared_mid.predict_samples(snap_clear, "ego", k_samples=24, horizon=12, environment=None, seed=99)
        pred_follow = shared_mid.predict_samples(snap_follow, "ego", k_samples=24, horizon=12, environment=None, seed=99)
        pred_behind = shared_mid.predict_samples(snap_behind, "ego", k_samples=24, horizon=12, environment=None, seed=99)
        pred_avoid = shared_mid.predict_samples(snap_avoid, "ego", k_samples=24, horizon=12, environment=None, seed=99)

        def _mean_endpoint(mp) -> Tuple[float, float]:
            mx = sum(s.predictions[-1].x for s in mp.samples) / len(mp.samples)
            my = sum(s.predictions[-1].y for s in mp.samples) / len(mp.samples)
            return mx, my

        ec = _mean_endpoint(pred_clear)
        ef = _mean_endpoint(pred_follow)
        eb = _mean_endpoint(pred_behind)
        ea = _mean_endpoint(pred_avoid)
        assert math.hypot(ef[0] - ec[0], ef[1] - ec[1]) > 0.05
        assert math.hypot(ea[0] - ec[0], ea[1] - ec[1]) > 0.05
        # Holding isotropic local_density constant (0.90m ahead vs 0.90m behind), social_encoder drives braking
        assert math.hypot(ef[0] - eb[0], ef[1] - eb[1]) > 0.025
        assert ef[0] < eb[0]

        # Ablation: disabling social_encoder yields 0.0 directional difference between leader ahead vs behind
        ablation = copy.deepcopy(shared_mid)
        ablation.config = EvacuationMIDConfig(
            diffusion_steps=shared_mid.config.diffusion_steps,
            beta_start=shared_mid.config.beta_start,
            beta_end=shared_mid.config.beta_end,
            horizon=shared_mid.config.horizon,
            prediction_interval=shared_mid.config.prediction_interval,
            conditioning_dim=shared_mid.config.conditioning_dim,
            social_dim=shared_mid.config.social_dim,
            use_social_encoding=False,
            d_model=shared_mid.config.d_model,
            hidden_dim=shared_mid.config.hidden_dim,
            k_samples=shared_mid.config.k_samples,
        )
        p_ab_behind = ablation.predict_samples(snap_behind, "ego", k_samples=24, horizon=12, environment=None, seed=99)
        p_ab_follow = ablation.predict_samples(snap_follow, "ego", k_samples=24, horizon=12, environment=None, seed=99)
        ab_delta = math.hypot(
            _mean_endpoint(p_ab_follow)[0] - _mean_endpoint(p_ab_behind)[0],
            _mean_endpoint(p_ab_follow)[1] - _mean_endpoint(p_ab_behind)[1],
        )
        assert ab_delta < 1e-5


class TestMIDDrivenMovementAndOriginalPreservation:
    def test_03_mid_driven_diverges_from_original_safely(self, make_app):
        """MID-DRIVEN moves pedestrians genuinely differently from ORIGINAL while preserving all physics bounds."""
        app_orig = make_app(seed=42, mode=OperatingMode.ORIGINAL_SIMULATION)
        app_drv = make_app(seed=42, mode=OperatingMode.MID_DRIVEN)

        is_walkable_fn = PAVAN["is_position_in_walkable_space"]
        check_overlap_fn = PAVAN["check_pedestrian_overlap"]
        prev_pos = {p.id: (p.position.x, p.position.y) for p in app_drv.engine.registry}

        for _ in range(36):
            app_orig.update_physics(0.1)
            app_drv.update_physics(0.1)

            active_peds = []
            for p in app_drv.engine.registry:
                step_d = math.hypot(p.position.x - prev_pos[p.id][0], p.position.y - prev_pos[p.id][1])
                assert step_d <= p.preferred_speed * app_drv.dt + 1e-5
                prev_pos[p.id] = (p.position.x, p.position.y)
                if str(getattr(p.status, "value", p.status)).lower() == "active":
                    active_peds.append(p)
                    ex = app_drv.building.get_exit(p.target_exit_id) if p.target_exit_id else None
                    assert is_walkable_fn(p.position, p.radius, app_drv.building, ex)

            for i in range(len(active_peds)):
                for j in range(i + 1, len(active_peds)):
                    assert not check_overlap_fn(
                        active_peds[i].position, active_peds[i].radius,
                        active_peds[j].position, active_peds[j].radius,
                    )

        assert app_drv.driven_controller.mid_driven_steps > 0
        assert any(rec.adoption_succeeded for rec in app_drv.adopted_route_records_by_agent.values())

        divs = [
            math.hypot(
                app_orig.engine.registry.get(p.id).position.x - p.position.x,
                app_orig.engine.registry.get(p.id).position.y - p.position.y,
            )
            for p in app_drv.engine.registry
        ]
        assert max(divs) > 0.05
        assert sum(1 for d in divs if d > 0.01) >= 4

    def test_04_original_determinism_and_mode_round_trip_cleanup(self, make_app):
        """ORIGINAL mode is strictly deterministic; cycling out of MID-DRIVEN strips transient step waypoints."""
        app1 = make_app(seed=42, mode=OperatingMode.ORIGINAL_SIMULATION)
        app2 = make_app(seed=99, mode=OperatingMode.ORIGINAL_SIMULATION)
        for _ in range(25):
            app1.update_physics(0.1)
            app2.update_physics(0.1)

        for p1 in app1.engine.registry:
            p2 = app2.engine.registry.get(p1.id)
            assert p1.position.x == p2.position.x
            assert p1.position.y == p2.position.y

        # Enter MID_DRIVEN for 8 steps, then press [M] back to ORIGINAL
        app_cycle = make_app(seed=42, mode=OperatingMode.MID_DRIVEN)
        for _ in range(8):
            app_cycle.update_physics(0.1)
        assert len(app_cycle.driven_controller._active_step_waypoint) > 0

        app_cycle.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
        assert app_cycle.operating_mode == OperatingMode.ORIGINAL_SIMULATION
        assert len(app_cycle.driven_controller._active_step_waypoint) == 0


class TestSafetyCollisionAndDeadlockLayer:
    def test_05_static_wall_and_obstacle_blocking(self, make_app):
        """CollisionAvoidanceSafetyLayer blocks steps intersecting obs_room1 or walls and finds safe fallback."""
        app = make_app(seed=42, mode=OperatingMode.ORIGINAL_SIMULATION)
        ped = app.engine.registry.get("ped_1")
        point_cls = type(ped.position)
        safety = CollisionAvoidanceSafetyLayer(app.building)

        # Place pedestrian directly left of obstacle_room1 (bounds: x=4..6, y=4..5.5)
        ped.position = point_cls(3.65, 4.75)
        ped.waypoints = [point_cls(10.0, 4.75)]
        target_ex = app.building.get_exit(ped.target_exit_id)

        # Propose unsafe step straight into obstacle_room1 interior (x=3.85 + r=0.30 -> 4.15 > 4.0)
        res = safety.evaluate_and_enforce(
            agent_id=ped.id,
            current_pedestrian=ped,
            proposed_target=(3.85, 4.75),
            dt=0.1,
            step_index=5,
            current_time=0.5,
            target_exit_obj=target_ex,
            goal_anchor=(10.0, 4.75),
            other_pedestrians=(),
            is_walkable_fn=PAVAN["is_position_in_walkable_space"],
            check_collision_fn=PAVAN["check_collision"],
        )
        assert res.is_proposed_safe is False
        assert res.intervention_applied is True
        assert res.intervention_reason is not None
        assert PAVAN["is_position_in_walkable_space"](
            point_cls(*res.safe_target), ped.radius, app.building, target_ex
        )
        assert len(safety.events) >= 1

    def test_06_swept_segment_moving_neighbor_collision_avoidance(self, make_app):
        """Swept-segment continuous check detects head-on moving neighbor conflict and picks safer deflection."""
        app = make_app(seed=42, mode=OperatingMode.ORIGINAL_SIMULATION)
        p1 = app.engine.registry.get("ped_1")
        p2 = app.engine.registry.get("ped_2")
        point_cls = type(p1.position)
        vel_cls = type(p1.velocity)

        p1.position = point_cls(20.0, 15.0)
        p1.velocity = vel_cls(1.34, 0.0)
        p2.position = point_cls(20.72, 15.0)
        p2.velocity = vel_cls(-1.34, 0.0)
        p2.status = PAVAN["PedestrianStatus"].ACTIVE
        target_ex = app.building.get_exit(p1.target_exit_id)

        d_min, _, _ = _continuous_segment_min_distance(20.0, 15.0, 20.14, 15.0, 20.72, 15.0, 20.58, 15.0)
        assert d_min < p1.radius + p2.radius + 0.05

        safety = CollisionAvoidanceSafetyLayer(app.building)
        res = safety.evaluate_and_enforce(
            agent_id=p1.id,
            current_pedestrian=p1,
            proposed_target=(20.14, 15.0),
            dt=0.1,
            step_index=10,
            current_time=1.0,
            target_exit_obj=target_ex,
            goal_anchor=(28.0, 15.0),
            other_pedestrians=(p1, p2),
            neighbor_velocities={p2.id: (-1.34, 0.0)},
            is_walkable_fn=PAVAN["is_position_in_walkable_space"],
            check_collision_fn=PAVAN["check_collision"],
        )
        assert res.intervention_applied is True
        assert res.conflict_agent_id == p2.id
        assert (res.safe_target[0], res.safe_target[1]) != (20.14, 15.0) or res.safe_speed_mps < 1.34
        # Resulting safe target must stay laterally or temporally clearer of moving neighbor
        d_safe, _, _ = _continuous_segment_min_distance(
            20.0, 15.0, res.safe_target[0], res.safe_target[1], 20.72, 15.0, 20.586, 15.0
        )
        assert d_safe >= p1.radius + p2.radius

    def test_07_anti_deadlock_stall_recovery(self, make_app):
        """Repeated stall steps trigger deadlock detection and lateral escape recovery."""
        app = make_app(seed=42, mode=OperatingMode.ORIGINAL_SIMULATION)
        ped = app.engine.registry.get("ped_1")
        point_cls = type(ped.position)
        ped.position = point_cls(20.0, 15.0)
        target_ex = app.building.get_exit(ped.target_exit_id)

        safety = CollisionAvoidanceSafetyLayer(app.building, stall_threshold_steps=3)
        res = None
        for step_idx in range(1, 5):
            # Repeatedly remain stalled at (20.0, 15.0) while proposing zero progress
            res = safety.evaluate_and_enforce(
                agent_id=ped.id,
                current_pedestrian=ped,
                proposed_target=(20.002, 15.0),
                dt=0.1,
                step_index=step_idx,
                current_time=step_idx * 0.1,
                target_exit_obj=target_ex,
                goal_anchor=(25.0, 15.0),
                other_pedestrians=(),
                is_walkable_fn=PAVAN["is_position_in_walkable_space"],
                check_collision_fn=PAVAN["check_collision"],
            )
        assert res is not None
        assert res.deadlock_recovery_applied is True
        assert safety.deadlock_recoveries_total >= 1
        assert res.safe_speed_mps > 0.2


class TestControlsRenderingAndRepeatability:
    def test_08_keyboard_mode_cycling_and_toggle_invariance(self, make_app):
        """[M] cycles all 4 operating modes cleanly; toggling [I]/[C]/[G] never changes simulation physics."""
        app_a = make_app(seed=42, mode=OperatingMode.ORIGINAL_SIMULATION)
        app_b = make_app(seed=42, mode=OperatingMode.ORIGINAL_SIMULATION)

        expected_cycle = [
            OperatingMode.MID_PREDICTION_ONLY,
            OperatingMode.MID_ASSISTED_EVACUATION,
            OperatingMode.MID_DRIVEN,
            OperatingMode.ORIGINAL_SIMULATION,
        ]
        for expected in expected_cycle:
            app_a.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_m))
            assert app_a.operating_mode == expected

        # Switch both to MID_DRIVEN and flip all visual toggles on app_b
        app_a.operating_mode = OperatingMode.MID_DRIVEN
        app_b.operating_mode = OperatingMode.MID_DRIVEN
        for k in (pygame.K_i, pygame.K_c, pygame.K_g):
            app_b.handle_event(pygame.event.Event(pygame.KEYDOWN, key=k))

        for _ in range(16):
            app_a.update_physics(0.1)
            app_b.update_physics(0.1)

        for pa in app_a.engine.registry:
            pb = app_b.engine.registry.get(pa.id)
            assert abs(pa.position.x - pb.position.x) < 1e-9
            assert abs(pa.position.y - pb.position.y) < 1e-9

    def test_09_hud_social_safety_and_ghost_overlay_rendering(self, make_app):
        """Headless rendering records social overlay, safety overlay, and ghost baseline state without errors."""
        if not pygame.get_init():
            pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()
        if not pygame.font.get_init():
            pygame.font.init()

        app = make_app(seed=42, mode=OperatingMode.MID_DRIVEN)
        for _ in range(12):
            app.update_physics(0.1)

        ped = app.engine.registry.get("ped_1")
        assert ped is not None
        # Move ped_2 directly ahead of ped_1 toward exit_left (-x) and step past obs tick 16 so MID refreshes
        p2 = app.engine.registry.get("ped_2")
        p2.position = type(ped.position)(ped.position.x - 1.0, ped.position.y + 0.02)
        app._latest_driven_mid.pop("ped_1", None)
        for _ in range(5):
            app.update_physics(0.1)

        # Record a recent safety intervention for visual safety overlay verification
        step_now = int(app.engine.clock.step_count)
        app.driven_controller.safety_layer.evaluate_and_enforce(
            agent_id="ped_1",
            current_pedestrian=ped,
            proposed_target=(-5.0, -5.0),
            dt=0.1,
            step_index=step_now,
            current_time=app.engine.clock.timestamp,
            target_exit_obj=app.building.get_exit(ped.target_exit_id),
            goal_anchor=(ped.position.x + 2.0, ped.position.y),
            other_pedestrians=list(app.engine.registry),
            is_walkable_fn=PAVAN["is_position_in_walkable_space"],
            check_collision_fn=PAVAN["check_collision"],
        )

        app.show_social_interactions = True
        app.show_safety_overlays = True
        app.show_baseline_ghost = True
        app.screen = pygame.display.set_mode((app.width, app.height))
        app._ensure_fonts()
        app.renderer.draw(app.screen, app.building, app.viewport, engine=app.engine)
        app._render_spatial_heatmap(app.screen)
        app.renderer._draw_pedestrians(app.screen, app.engine.registry, app.viewport)
        app._render_predictions(app.screen)
        app._render_hud_and_telemetry(app.screen)

        assert "social_overlay" in app.latest_layer_states
        assert app.latest_layer_states["social_overlay"].get("rendered") is True
        assert app.latest_layer_states["safety_overlay"].get("rendered") is True
        assert app.latest_layer_states["safety_overlay"].get("active_events", 0) >= 1
        assert app.latest_layer_states["baseline_ghost"].get("rendered") is True

    def test_10_mid_driven_deterministic_repeatability(self, make_app):
        """Identical seeds in MID-DRIVEN mode produce identical agent trajectories, decisions, and safety totals."""
        a1 = make_app(seed=42, mode=OperatingMode.MID_DRIVEN)
        a2 = make_app(seed=42, mode=OperatingMode.MID_DRIVEN)

        for _ in range(20):
            a1.update_physics(0.1)
            a2.update_physics(0.1)

        assert a1.driven_controller.mid_driven_steps == a2.driven_controller.mid_driven_steps
        assert (
            a1.driven_controller.safety_layer.safety_interventions_total
            == a2.driven_controller.safety_layer.safety_interventions_total
        )
        for p1 in a1.engine.registry:
            p2 = a2.engine.registry.get(p1.id)
            assert p1.position.x == pytest.approx(p2.position.x, abs=1e-9)
            assert p1.position.y == pytest.approx(p2.position.y, abs=1e-9)
