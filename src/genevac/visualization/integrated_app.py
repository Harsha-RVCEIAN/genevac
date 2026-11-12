"""Integrated 2D Pygame Evacuation Simulation for GenEvac-Sentinel.

Seamlessly bridges Pavan's Module B Part 1 simulation engine, authoritative
building models, physical movement, and Pygame environment renderer with
Harsha's Module B Part 2 analytics, spatial heatmaps, Original Evacuation
Motion Indeterminacy Diffusion (MID) trajectory predictions, Five-Stage
Live Presentation Mode, and MID-Assisted Evacuation Decision Controller.

Modes:
- Operating Modes:
  1. ORIGINAL SIMULATION (Default baseline)
  2. MID PREDICTION ONLY (Stochastic diffusion overlays, read-only)
  3. MID-ASSISTED EVACUATION (Safe route recommendation layer)
- Presentation Mode:
  Dedicated 5-stage interactive sequence (toggled with [D])
"""

from __future__ import annotations

import argparse
from enum import Enum
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pygame

# =============================================================================
# 1. Clean Two-Phase Module Loader for Part 1 & Part 2 Integration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
HARSHA_SRC = PROJECT_ROOT / "src"
PAVAN_SRC = PROJECT_ROOT / "pavan-part1-reference" / "src"

# Step A: Load Harsha's Part 2 modules
if str(HARSHA_SRC) not in sys.path:
    sys.path.insert(0, str(HARSHA_SRC))

import genevac.environment as h_env
import genevac.prediction.adapter as h_adapter
import genevac.prediction.baseline as h_baseline
import genevac.prediction.controller as h_controller
import genevac.prediction.data_contract as h_contract
import genevac.prediction.mid as h_mid
import genevac.analytics.heatmap as h_heatmap

# Snapshot Harsha's genevac modules
harsha_modules = {k: v for k, v in sys.modules.items() if k == "genevac" or k.startswith("genevac.")}

# Step B: Temporarily swap sys.path to PAVAN_SRC to load Pavan's Part 1 modules
for k in list(harsha_modules.keys()):
    del sys.modules[k]

if str(HARSHA_SRC) in sys.path:
    sys.path.remove(str(HARSHA_SRC))
sys.path.insert(0, str(PAVAN_SRC))

import genevac.environment.serialization as p_serial
import genevac.simulation.engine as p_engine
import genevac.pedestrian.registry as p_reg
import genevac.pedestrian.models as p_models
import genevac.simulation.movement as p_movement
import genevac.visualization.app as p_app
import genevac.visualization.renderer as p_renderer
import genevac.visualization.coordinates as p_coords

# Package Pavan's components into an explicit dictionary
PAVAN = {
    "load_building": p_serial.load_building,
    "SimulationEngine": p_engine.SimulationEngine,
    "PedestrianRegistry": p_reg.PedestrianRegistry,
    "PedestrianStatus": p_models.PedestrianStatus,
    "generate_deterministic_route": p_movement.generate_deterministic_route,
    "is_position_in_walkable_space": p_movement.is_position_in_walkable_space,
    "check_collision": p_movement.check_collision,
    "check_pedestrian_overlap": p_movement.check_pedestrian_overlap,
    "create_default_pedestrian_cohort": p_app.create_default_pedestrian_cohort,
    "EnvironmentRenderer": p_renderer.EnvironmentRenderer,
    "Viewport": p_coords.Viewport,
}

# Step C: Restore Harsha's modules into sys.modules & sys.path
sys.path.remove(str(PAVAN_SRC))
sys.path.insert(0, str(HARSHA_SRC))
for k, v in harsha_modules.items():
    sys.modules[k] = v


# =============================================================================
# 2. Operating & Presentation Enums
# =============================================================================

class OperatingMode(str, Enum):
    """Authoritative operating modes for the integrated application."""
    ORIGINAL_SIMULATION = "ORIGINAL SIMULATION"
    MID_PREDICTION_ONLY = "MID PREDICTION ONLY"
    MID_ASSISTED_EVACUATION = "MID-ASSISTED EVACUATION"


class PresentationStage(int, Enum):
    """The 5 distinct presentation stages for technical defense."""
    STAGE_1_LIVE_EVACUATION = 1
    STAGE_2_SPATIAL_HEATMAP = 2
    STAGE_3_CONSTANT_VELOCITY = 3
    STAGE_4_MID_PREDICTION = 4
    STAGE_5_MID_ASSISTED_DEMONSTRATION = 5
    STAGE_5_INDEPENDENT_VERIFICATION = 5


class PredictionStatus(str, Enum):
    """Discrete observable status of the prediction pipeline."""
    WAITING_FOR_HISTORY = "WAITING FOR HISTORY"
    PREDICTING = "PREDICTING"
    ACTIVE = "ACTIVE"
    UNTRAINED = "UNTRAINED"
    ERROR = "ERROR"


# =============================================================================
# 3. Helper: Pre-Train Evacuation MID Model
# =============================================================================

def create_pretrained_mid_predictor(k_samples: int = 6, horizon: int = 12) -> h_mid.EvacuationMIDPredictor:
    """Instantiate and pre-train Evacuation MID on canonical evacuation samples."""
    mid_config = h_mid.EvacuationMIDConfig(
        horizon=horizon,
        prediction_interval=0.1,
        k_samples=k_samples,
        diffusion_steps=15,
        hidden_dim=128,
        w_exit=0.25,
        w_obs=0.45,
        w_kin=0.30,
    )
    predictor = h_mid.EvacuationMIDPredictor(config=mid_config)

    try:
        from genevac.prediction.evaluation import EvaluationSample

        train_samples: List[EvaluationSample] = []
        for y_start in np.linspace(6.0, 24.0, 8):
            hist = tuple(
                h_contract.TrajectoryObservation(
                    agent_id="train_agent",
                    timestamp=float(i * 0.1),
                    x=float(5.0 + i * 1.34 * 0.1),
                    y=float(y_start),
                    step_index=int(i),
                )
                for i in range(4)
            )
            fut = tuple(
                h_contract.TrajectoryObservation(
                    agent_id="train_agent",
                    timestamp=float((4 + j) * 0.1),
                    x=float(5.0 + (4 + j) * 1.34 * 0.1),
                    y=float(y_start + math.sin(j * 0.2) * 0.3),
                    step_index=int(4 + j),
                )
                for j in range(12)
            )
            train_samples.append(
                EvaluationSample(
                    agent_id="train_agent",
                    reference_step=3,
                    reference_timestamp=0.3,
                    history=hist,
                    ground_truth_future=fut,
                    cohort="active",
                )
            )
        predictor.fit(train_samples, epochs=10, batch_size=4)
    except Exception as e:
        print(f"[Warning] MID pre-training encountered: {e}")

    return predictor


# =============================================================================
# 4. Integrated Simulation Application Class
# =============================================================================

class IntegratedSimulationApp:
    """Seamlessly runs Pavan's SimulationEngine rendered via Pygame with Harsha's Part 2 features."""

    SIDE_PANEL_WIDTH = 380

    def __init__(
        self,
        map_path: Optional[str | Path] = None,
        width: int = 1440,
        height: int = 860,
        fps: int = 30,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.dt = 0.1
        self.observation_interval = 0.4

        # Resolve building map
        default_map = PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json"
        self.map_path = Path(map_path) if map_path is not None else default_map
        if not self.map_path.is_file():
            raise FileNotFoundError(f"Building map not found: {self.map_path}")

        # Initialize Pavan's building and simulation engine
        self.building = PAVAN["load_building"](self.map_path)
        self.engine: Any = None

        # Viewport maps world to screen area excluding the right-hand panel
        sim_screen_w = max(400, self.width - self.SIDE_PANEL_WIDTH)
        self.viewport = PAVAN["Viewport"](
            world_width=self.building.width,
            world_height=self.building.height,
            screen_width=sim_screen_w,
            screen_height=self.height,
            margin=45.0,
            top_offset=55.0,
            bottom_offset=45.0,
        )
        self.renderer = PAVAN["EnvironmentRenderer"]()

        # Build Harsha's Part 2 EnvironmentMap for physical collision checking in MID
        self.part2_env = self._build_part2_environment(self.building)

        # Part 2 Ingestion Adapter & Trajectory Buffer
        self.adapter = h_adapter.SimulationTrajectoryAdapter(
            observation_interval=self.observation_interval,
            min_history_required=2,
            window_size=30,
        )

        # Part 2 Spatial Heatmap Engine
        self.grid_spec = h_heatmap.GridSpecification(
            min_x=0.0,
            min_y=0.0,
            max_x=self.building.width,
            max_y=self.building.height,
            cell_size_x=1.0,
            cell_size_y=1.0,
        )
        self.heatmap_engine = h_heatmap.SpatialHeatmapEngine(self.grid_spec)

        # Part 2 Predictors
        self.cv_predictor = h_baseline.ConstantVelocityPredictor()
        self.k_samples = 6
        self.prediction_horizon = 12
        self.mid_predictor = create_pretrained_mid_predictor(
            k_samples=self.k_samples, horizon=self.prediction_horizon
        )

        # Part D: MID-Assisted Decision Controller
        self.controller = h_controller.MIDAssistedDecisionAdapter(
            building=self.building,
            min_decision_interval=0.4,
        )
        self.latest_recommendation: Optional[h_controller.MIDDecisionRecommendation] = None
        self.base_seed: int = 42

        # Operating Modes
        self.operating_mode: OperatingMode = OperatingMode.ORIGINAL_SIMULATION
        self.presentation_mode: bool = False
        self.presentation_stage: PresentationStage = PresentationStage.STAGE_1_LIVE_EVACUATION

        # State toggles
        self.show_heatmap = True
        self.show_mid_predictions = True
        self.show_baseline_predictions = True
        self.show_ids = True
        self.selected_agent_id: Optional[str] = "ped_1"

        # Cached latest metrics and telemetry
        self.last_mid_walkability: float = 1.0
        self.last_mid_collision_free: float = 1.0
        self.mid_status: PredictionStatus = PredictionStatus.WAITING_FOR_HISTORY
        self.cv_status: PredictionStatus = PredictionStatus.WAITING_FOR_HISTORY
        self.last_prediction_time: float = 0.0
        self.latest_prediction_error: Optional[str] = None
        self.actual_returned_k: int = 0
        self.available_observations_count: int = 0

        # Prediction Caches: (agent_id, step, K, horizon) -> Result
        self._cached_mid_key: Optional[Tuple[str, int, int, int]] = None
        self._cached_mid_result: Optional[h_mid.MultiTrajectoryPrediction] = None
        self._mid_cache_map: Dict[Tuple[str, int, int, int], h_mid.MultiTrajectoryPrediction] = {}
        self._cached_cv_key: Optional[Tuple[str, int, int]] = None
        self._cached_cv_result: Optional[h_contract.TrajectoryPrediction] = None

        # Timing accumulator
        self.sim_accumulator = 0.0
        self.max_catchup_steps = 5
        self.max_dt_wall = 0.25

        # High-contrast color palette for K individual MID stochastic paths
        self.mid_colors = [
            (56, 189, 248),    # Electric Cyan
            (244, 63, 94),     # Neon Rose / Coral Red
            (168, 85, 247),    # Vivid Purple
            (34, 197, 94),     # Bright Emerald
            (249, 115, 22),    # Amber Orange
            (236, 72, 153),    # Hot Pink
            (234, 179, 8),     # Bright Yellow
            (45, 212, 191),    # Mint Teal
            (99, 102, 241),    # Indigo
            (217, 70, 239),    # Fuchsia
        ]

        # Pygame surfaces & fonts
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font_title: Optional[pygame.font.Font] = None
        self.font_section: Optional[pygame.font.Font] = None
        self.font_hud: Optional[pygame.font.Font] = None
        self.font_small: Optional[pygame.font.Font] = None
        self.font_legend: Optional[pygame.font.Font] = None

        # Build initial simulation state
        self.reset_simulation()

    def _build_part2_environment(self, building: Any) -> h_env.EnvironmentMap:
        """Convert Pavan's Building geometry to Part 2 EnvironmentMap for MID physical collision detection."""
        env = h_env.EnvironmentMap(width=building.width, height=building.height)
        for obs in building.obstacles:
            env.add_obstacle(
                h_env.Obstacle(
                    id=obs.id,
                    bounds=(obs.bounds.x, obs.bounds.y, obs.bounds.x + obs.bounds.width, obs.bounds.y + obs.bounds.height),
                )
            )
        for ex in building.exits:
            mid_x = (ex.segment.start.x + ex.segment.end.x) / 2.0
            mid_y = (ex.segment.start.y + ex.segment.end.y) / 2.0
            env.add_exit(h_env.Exit(id=ex.id, position=(mid_x, mid_y), width=ex.width))
        return env

    def reset_simulation(self) -> None:
        """Reset Pavan's SimulationEngine, Part 2 analytics buffers, and prediction caches."""
        for ex in self.building.exits:
            ex.open()
        cohort = PAVAN["create_default_pedestrian_cohort"]()
        registry = PAVAN["PedestrianRegistry"]()
        for ped in cohort:
            registry.register(ped)

        self.engine = PAVAN["SimulationEngine"](
            building=self.building,
            registry=registry,
            dt=self.dt,
            observation_interval=self.observation_interval,
        )

        # Reset Part 2 adapter & heatmap engine
        self.adapter.reset()
        self.heatmap_engine.reset()
        self.controller.reset()

        # Clear prediction caches
        self._cached_mid_key = None
        self._cached_mid_result = None
        self._mid_cache_map.clear()
        self._cached_cv_key = None
        self._cached_cv_result = None
        self.latest_recommendation = None
        self.latest_prediction_error = None
        self.mid_status = PredictionStatus.WAITING_FOR_HISTORY
        self.cv_status = PredictionStatus.WAITING_FOR_HISTORY

        # Ingest initial snapshot at step 0
        self.adapter.adapt_snapshot(self.engine.latest_snapshot)
        self.sim_accumulator = 0.0

        if cohort:
            self.selected_agent_id = cohort[0].id
        else:
            self.selected_agent_id = None

    def update_physics(self, dt_wall: float) -> int:
        """Advance Pavan's SimulationEngine, adapt snapshots into Part 2 buffers, and execute controller if active."""
        if self.engine.is_paused:
            return 0

        clamped_dt = max(0.0, min(dt_wall, self.max_dt_wall))
        self.sim_accumulator += clamped_dt

        steps_taken = 0
        while self.sim_accumulator >= self.dt and steps_taken < self.max_catchup_steps:
            # Advance Pavan's authoritative physics
            self.engine.step()
            self.sim_accumulator -= self.dt
            steps_taken += 1

            # Ingest snapshot at observation cadence
            is_obs = (
                self.engine.clock.is_observation_step()
                if callable(self.engine.clock.is_observation_step)
                else bool(self.engine.clock.is_observation_step)
            )
            if is_obs:
                self.adapter.adapt_snapshot(self.engine.latest_snapshot)

                # Part D: In MID-Assisted Mode, safely evaluate recommendations on observation ticks
                if self.operating_mode == OperatingMode.MID_ASSISTED_EVACUATION:
                    self._apply_mid_assisted_guidance()

        return steps_taken

    def _apply_mid_assisted_guidance(self) -> None:
        """Safely execute MID decision adapter recommendations for all eligible pedestrians."""
        step = self.engine.clock.step_count
        timestamp = self.engine.clock.timestamp
        snapshot = self.adapter.buffer.get_snapshot(step, timestamp)
        open_exits = [ex.id for ex in self.building.exits if ex.is_open]
        all_peds = list(self.engine.registry)

        for ped in all_peds:
            aid = ped.id
            status_val = str(getattr(ped.status, "value", ped.status)).lower()
            if status_val not in ("active", "blocked"):
                continue

            if aid not in snapshot.trajectories or len(snapshot.trajectories[aid]) < 2:
                continue

            mid_res = self._get_or_compute_mid_prediction(snapshot, aid, step)
            if not mid_res or not mid_res.samples:
                continue

            rec = self.controller.evaluate_and_recommend(
                agent_id=aid,
                current_pedestrian=ped,
                mid_prediction=mid_res,
                open_exit_ids=open_exits,
                current_time=timestamp,
                step_index=step,
                generate_route_fn=PAVAN["generate_deterministic_route"],
                is_walkable_fn=PAVAN["is_position_in_walkable_space"],
                check_collision_fn=PAVAN["check_collision"],
                check_overlap_fn=PAVAN["check_pedestrian_overlap"],
                other_pedestrians=all_peds,
            )

            # Safe movement integration: Update target_exit_id and waypoints via engine public interface
            if rec.is_accepted and rec.proposed_waypoints:
                ped.target_exit_id = rec.target_exit_id
                ped.waypoints = list(rec.proposed_waypoints)
                if status_val == "blocked":
                    ped.status = PAVAN["PedestrianStatus"].ACTIVE

        if self.selected_agent_id:
            self.latest_recommendation = self.controller.latest_recommendations_by_agent.get(self.selected_agent_id)
            sel_key = (self.selected_agent_id, step, self.k_samples, self.prediction_horizon)
            if sel_key in self._mid_cache_map:
                self._cached_mid_key = sel_key
                self._cached_mid_result = self._mid_cache_map[sel_key]

    # =========================================================================
    # Prediction Caching & Execution Engine
    # =========================================================================

    def _get_or_compute_cv_prediction(
        self, snapshot: h_contract.TrajectorySnapshot, target_id: str, step: int
    ) -> Optional[h_contract.TrajectoryPrediction]:
        """Retrieve cached Constant Velocity prediction or compute cleanly with visible error handling."""
        cache_key = (target_id, step, self.prediction_horizon)
        if self._cached_cv_key == cache_key and self._cached_cv_result is not None:
            return self._cached_cv_result

        try:
            cv_pred = self.cv_predictor.predict(
                snapshot, target_id, horizon=self.prediction_horizon
            )
            self._cached_cv_key = cache_key
            self._cached_cv_result = cv_pred
            self.cv_status = PredictionStatus.ACTIVE
            return cv_pred
        except Exception as e:
            self.cv_status = PredictionStatus.ERROR
            self.latest_prediction_error = f"CV error on {target_id}: {e}"
            return None

    def _resolve_conditioning_exit_id(self, target_id: str) -> Optional[str]:
        """Determine the open exit ID to condition MID trajectory generation for target_id."""
        open_exits = [ex for ex in self.building.exits if ex.is_open]
        if not open_exits:
            return None
        open_ids = {ex.id for ex in open_exits}
        ped = self.engine.registry.get(target_id) if self.engine else None
        if ped and getattr(ped, "target_exit_id", None) in open_ids:
            return ped.target_exit_id
        if ped:
            best_ex = min(
                open_exits,
                key=lambda ex: math.hypot(
                    (ex.segment.start.x + ex.segment.end.x) / 2.0 - ped.position.x,
                    (ex.segment.start.y + ex.segment.end.y) / 2.0 - ped.position.y,
                ),
            )
            return best_ex.id
        return None

    def _get_or_compute_mid_prediction(
        self, snapshot: h_contract.TrajectorySnapshot, target_id: str, step: int
    ) -> Optional[h_mid.MultiTrajectoryPrediction]:
        """Retrieve cached MID prediction or execute diffusion with accurate state transitions."""
        cache_key = (target_id, step, self.k_samples, self.prediction_horizon)
        if self._cached_mid_key == cache_key and self._cached_mid_result is not None:
            return self._cached_mid_result
        if cache_key in self._mid_cache_map and self._cached_mid_key is not None:
            res = self._mid_cache_map[cache_key]
            if target_id == self.selected_agent_id:
                self._cached_mid_key = cache_key
                self._cached_mid_result = res
            return res

        t_start = time.perf_counter()
        try:
            self.mid_status = PredictionStatus.PREDICTING
            try:
                agent_num = int(str(target_id).split("_")[-1])
            except ValueError:
                agent_num = sum(ord(c) for c in str(target_id))
            det_seed = self.base_seed + step * 31 + agent_num * 7
            cond_exit_id = self._resolve_conditioning_exit_id(target_id)

            mid_result = self.mid_predictor.predict_samples(
                snapshot=snapshot,
                agent_id=target_id,
                k_samples=self.k_samples,
                horizon=self.prediction_horizon,
                environment=self.part2_env,
                target_exit_id=cond_exit_id,
                seed=det_seed,
            )
            self.last_prediction_time = time.perf_counter() - t_start
            self._mid_cache_map[cache_key] = mid_result
            self._cached_mid_key = cache_key
            self._cached_mid_result = mid_result
            if target_id == self.selected_agent_id:
                self.actual_returned_k = len(mid_result.samples)
                self.last_mid_walkability = mid_result.walkable_validity
                self.last_mid_collision_free = mid_result.segment_collision_free
            elif self.actual_returned_k == 0:
                self.actual_returned_k = len(mid_result.samples)
            self.mid_status = PredictionStatus.ACTIVE if self.mid_predictor.is_trained else PredictionStatus.UNTRAINED
            self.latest_prediction_error = None
            return mid_result
        except Exception as e:
            self.last_prediction_time = time.perf_counter() - t_start
            self.mid_status = PredictionStatus.ERROR
            self.latest_prediction_error = f"MID error on {target_id}: {e}"
            self.actual_returned_k = 0
            # Visible error logging without crashing
            print(f"[MID Pipeline Error] Agent {target_id} at step {step}: {e}")
            traceback.print_exc()
            return None

    # =========================================================================
    # Part 2 Rendering Layers (Heatmaps, Predictions, Telemetry HUD)
    # =========================================================================

    def _render_spatial_heatmap(self, surface: pygame.Surface) -> None:
        """Render transparent Part 2 spatial density heatmap with physical gradient."""
        if not self.show_heatmap or self.engine is None:
            return

        active_positions = [
            (p.position.x, p.position.y)
            for p in self.engine.registry
            if str(p.status).lower().endswith("active")
        ]
        if not active_positions:
            return

        step = self.engine.clock.step_count
        timestamp = self.engine.clock.timestamp

        grid_res = self.heatmap_engine.aggregate_positions(
            positions=active_positions,
            timestamp=timestamp,
            step_index=step,
        )

        raw_density = grid_res.density_grid
        max_density = float(np.max(raw_density))
        if max_density < 1e-4:
            return

        # Reference scale: 1.0 ped/m² is standard high local density in evacuation corridors
        density_norm = np.clip(raw_density / max(1.0, max_density), 0.0, 1.0)

        # Colormap: 'plasma' (smooth purple -> reddish-orange -> yellow)
        cmap = plt.get_cmap("plasma")
        rgba = (cmap(density_norm) * 255).astype(np.uint8)

        # Dynamic per-cell alpha: 0 for zero density so floor is 100% visible,
        # ramping up to ~150 (60% opacity) for high concentration clusters
        alpha_channel = (density_norm * 150).astype(np.uint8)
        rgba[:, :, 3] = np.where(density_norm > 0.01, alpha_channel, 0)

        # Transpose for Pygame: (x, y) = (col, row)
        rgba_t = np.transpose(rgba, (1, 0, 2))

        cols = self.grid_spec.cols
        rows = self.grid_spec.rows
        temp_surf = pygame.Surface((cols, rows), pygame.SRCALPHA)
        pygame.surfarray.blit_array(temp_surf, rgba_t[:, :, :3])

        # Apply custom alpha mask per pixel
        alpha_surf = pygame.surfarray.pixels_alpha(temp_surf)
        alpha_surf[:] = rgba_t[:, :, 3]
        del alpha_surf

        # Scale to building canvas screen bounds and blit
        bx, by, bw, bh = self.viewport.building_screen_bounds
        scaled_heatmap = pygame.transform.smoothscale(temp_surf, (int(bw), int(bh)))
        surface.blit(scaled_heatmap, (int(bx), int(by)))

        # Draw compact physical density legend in lower-left of the building area
        self._render_heatmap_legend(surface, max_density)

    def _render_heatmap_legend(self, surface: pygame.Surface, max_density: float) -> None:
        """Render compact density colorbar with physical units (peds/m²)."""
        bx, by, bw, bh = self.viewport.building_screen_bounds
        leg_w, leg_h = 160, 12
        leg_x = int(bx + 16)
        leg_y = int(by + bh - 42)

        # Gradient bar
        grad_surf = pygame.Surface((leg_w, leg_h))
        cmap = plt.get_cmap("plasma")
        for x in range(leg_w):
            norm_val = x / float(leg_w)
            color_rgb = [int(c * 255) for c in cmap(norm_val)[:3]]
            pygame.draw.line(grad_surf, color_rgb, (x, 0), (x, leg_h))

        # Background pill
        bg_rect = pygame.Rect(leg_x - 6, leg_y - 18, leg_w + 12, leg_h + 32)
        bg = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        bg.fill((15, 23, 42, 210))
        surface.blit(bg, bg_rect)
        pygame.draw.rect(surface, (51, 65, 85), bg_rect, width=1, border_radius=4)

        surface.blit(grad_surf, (leg_x, leg_y))
        pygame.draw.rect(surface, (148, 163, 184), (leg_x, leg_y, leg_w, leg_h), width=1)

        # Title & values
        title = self.font_small.render("Spatial Density (peds/m²)", True, (241, 245, 249))
        surface.blit(title, (leg_x, leg_y - 15))

        lbl_min = self.font_small.render("0.0", True, (203, 213, 225))
        lbl_max = self.font_small.render(f"{max_density:.1f}", True, (203, 213, 225))
        surface.blit(lbl_min, (leg_x, leg_y + leg_h + 2))
        surface.blit(lbl_max, (leg_x + leg_w - lbl_max.get_width(), leg_y + leg_h + 2))

    def _render_predictions(self, surface: pygame.Surface) -> None:
        """Render MID stochastic trajectory predictions, CV baseline, observation history, and controller recommendations."""
        target_id = self.selected_agent_id
        if not target_id:
            return

        # Selection ring around the targeted pedestrian (visible even if overlays are off)
        selected_ped = self.engine.registry.get(target_id)
        if selected_ped:
            sel_sx, sel_sy = self.viewport.world_to_screen_point((selected_ped.position.x, selected_ped.position.y))
            r_px = max(6, int(self.viewport.world_to_screen_dim(selected_ped.radius) + 9))
            pygame.draw.circle(surface, (250, 204, 21), (int(sel_sx), int(sel_sy)), r_px, width=2)
            pygame.draw.circle(surface, (250, 204, 21), (int(sel_sx), int(sel_sy)), r_px + 3, width=1)
            pygame.draw.circle(surface, (250, 204, 21), (int(sel_sx), int(sel_sy)), 2)

        if not (self.show_mid_predictions or self.show_baseline_predictions or self.operating_mode == OperatingMode.MID_ASSISTED_EVACUATION):
            return

        step = self.engine.clock.step_count
        timestamp = self.engine.clock.timestamp
        snapshot = self.adapter.buffer.get_snapshot(step, timestamp)

        if target_id not in snapshot.trajectories:
            self.mid_status = PredictionStatus.WAITING_FOR_HISTORY
            self.cv_status = PredictionStatus.WAITING_FOR_HISTORY
            self.available_observations_count = 0
            return

        history = snapshot.trajectories[target_id]
        self.available_observations_count = len(history)
        if len(history) < 2:
            self.mid_status = PredictionStatus.WAITING_FOR_HISTORY
            self.cv_status = PredictionStatus.WAITING_FOR_HISTORY
            return

        latest_obs = history[-1]
        start_sx, start_sy = self.viewport.world_to_screen_point((latest_obs.x, latest_obs.y))

        # A. Observation History (Amber breadcrumb trail)
        hist_pts = [self.viewport.world_to_screen_point((obs.x, obs.y)) for obs in history]
        if len(hist_pts) >= 2:
            for i in range(len(hist_pts) - 1):
                p1 = (int(hist_pts[i][0]), int(hist_pts[i][1]))
                p2 = (int(hist_pts[i+1][0]), int(hist_pts[i+1][1]))
                pygame.draw.line(surface, (245, 158, 11), p1, p2, width=2)
                pygame.draw.circle(surface, (245, 158, 11), p1, 2)

        # B. Constant Velocity Baseline Prediction (Dashed White Line)
        if self.show_baseline_predictions:
            cv_pred = self._get_or_compute_cv_prediction(snapshot, target_id, step)
            if cv_pred and cv_pred.predictions:
                pts = [(start_sx, start_sy)]
                for p in cv_pred.predictions:
                    sx, sy = self.viewport.world_to_screen_point((p.x, p.y))
                    pts.append((sx, sy))

                if len(pts) >= 2:
                    for i in range(len(pts) - 1):
                        p1 = (int(pts[i][0]), int(pts[i][1]))
                        p2 = (int(pts[i+1][0]), int(pts[i+1][1]))
                        if i % 2 == 0:  # dashed pattern
                            pygame.draw.line(surface, (241, 245, 249), p1, p2, width=3)
                    end_pt = (int(pts[-1][0]), int(pts[-1][1]))
                    pygame.draw.circle(surface, (241, 245, 249), end_pt, 5)
                    pygame.draw.circle(surface, (15, 23, 42), end_pt, 2)

        # C. Original Evacuation MID Stochastic Predictions (High-contrast K-sample fan)
        if self.show_mid_predictions or self.operating_mode == OperatingMode.MID_ASSISTED_EVACUATION:
            mid_result = self._get_or_compute_mid_prediction(snapshot, target_id, step)
            rec_for_target = (
                self.controller.latest_recommendations_by_agent.get(target_id)
                if self.operating_mode == OperatingMode.MID_ASSISTED_EVACUATION
                else None
            )
            cand_map = (
                {c.sample_idx: c for c in rec_for_target.candidates}
                if rec_for_target is not None
                else {}
            )
            selected_idx = (
                rec_for_target.selected_sample_idx
                if (rec_for_target is not None and rec_for_target.is_accepted)
                else -1
            )

            if mid_result and mid_result.samples:
                # Pass 1: Draw rejected and unselected valid samples first
                for idx, sample in enumerate(mid_result.samples):
                    if idx == selected_idx:
                        continue
                    pts = [(start_sx, start_sy)]
                    for p in sample.predictions:
                        sx, sy = self.viewport.world_to_screen_point((p.x, p.y))
                        pts.append((sx, sy))
                    if len(pts) < 2:
                        continue

                    cand_eval = cand_map.get(idx)
                    if cand_eval is not None and not cand_eval.is_valid:
                        # Rejected candidate: muted coral-red dashed line + 'x' marker
                        rej_color = (248, 113, 113)
                        for i in range(len(pts) - 1):
                            if i % 2 == 0:
                                p1 = (int(pts[i][0]), int(pts[i][1]))
                                p2 = (int(pts[i + 1][0]), int(pts[i + 1][1]))
                                pygame.draw.line(surface, rej_color, p1, p2, width=1)
                        end_x, end_y = int(pts[-1][0]), int(pts[-1][1])
                        pygame.draw.line(surface, rej_color, (end_x - 3, end_y - 3), (end_x + 3, end_y + 3), width=2)
                        pygame.draw.line(surface, rej_color, (end_x - 3, end_y + 3), (end_x + 3, end_y - 3), width=2)
                    else:
                        # Unselected feasible sample (or Prediction-Only sample)
                        color = self.mid_colors[idx % len(self.mid_colors)]
                        for i in range(len(pts) - 1):
                            p1 = (int(pts[i][0]), int(pts[i][1]))
                            p2 = (int(pts[i + 1][0]), int(pts[i + 1][1]))
                            pygame.draw.line(surface, color, p1, p2, width=2)
                        end_pt = (int(pts[-1][0]), int(pts[-1][1]))
                        pygame.draw.circle(surface, color, end_pt, 4)
                        pygame.draw.circle(surface, (15, 23, 42), end_pt, 2)

                # Pass 2: Draw the SELECTED MID candidate prominently on top
                if 0 <= selected_idx < len(mid_result.samples):
                    sel_sample = mid_result.samples[selected_idx]
                    pts = [(start_sx, start_sy)]
                    for p in sel_sample.predictions:
                        sx, sy = self.viewport.world_to_screen_point((p.x, p.y))
                        pts.append((sx, sy))
                    if len(pts) >= 2:
                        for i in range(len(pts) - 1):
                            p1 = (int(pts[i][0]), int(pts[i + 1][1]))
                            p1 = (int(pts[i][0]), int(pts[i][1]))
                            p2 = (int(pts[i + 1][0]), int(pts[i + 1][1]))
                            # Outer dark emerald-gold halo
                            pygame.draw.line(surface, (6, 95, 70), p1, p2, width=6)
                            # Bright gold-emerald selected trajectory core
                            pygame.draw.line(surface, (250, 204, 21), p1, p2, width=3)
                        end_pt = (int(pts[-1][0]), int(pts[-1][1]))
                        pygame.draw.circle(surface, (250, 204, 21), end_pt, 6)
                        pygame.draw.circle(surface, (15, 23, 42), end_pt, 3)

        # D. MID-Assisted Controller Recommendation & Resulting Route Overlay
        if self.operating_mode == OperatingMode.MID_ASSISTED_EVACUATION:
            # Draw subtle guide waypoint markers for all guided pedestrians
            for aid, rec_item in self.controller.latest_recommendations_by_agent.items():
                if rec_item.is_accepted and rec_item.guide_waypoint is not None:
                    ped_obj = self.engine.registry.get(aid)
                    if ped_obj and str(getattr(ped_obj.status, "value", ped_obj.status)).lower() == "active":
                        psx, psy = self.viewport.world_to_screen_point((ped_obj.position.x, ped_obj.position.y))
                        gsx, gsy = self.viewport.world_to_screen_point(
                            (rec_item.guide_waypoint.x, rec_item.guide_waypoint.y)
                        )
                        if aid != target_id:
                            pygame.draw.line(surface, (16, 185, 129), (int(psx), int(psy)), (int(gsx), int(gsy)), width=2)
                            pygame.draw.circle(surface, (52, 211, 153), (int(gsx), int(gsy)), 4)

            rec = self.controller.latest_recommendations_by_agent.get(target_id) or self.latest_recommendation
            if rec and rec.agent_id == target_id and rec.is_accepted and rec.proposed_waypoints:
                self.latest_recommendation = rec
                rec_pts = [(start_sx, start_sy)]
                for wp in rec.proposed_waypoints:
                    sx, sy = self.viewport.world_to_screen_point((wp.x, wp.y))
                    rec_pts.append((sx, sy))

                if len(rec_pts) >= 2:
                    for i in range(len(rec_pts) - 1):
                        p1 = (int(rec_pts[i][0]), int(rec_pts[i][1]))
                        p2 = (int(rec_pts[i + 1][0]), int(rec_pts[i + 1][1]))
                        # Glow outline
                        pygame.draw.line(surface, (6, 95, 70), p1, p2, width=5)
                        # Main route line
                        pygame.draw.line(surface, (52, 211, 153), p1, p2, width=2)
                        pygame.draw.circle(surface, (52, 211, 153), p2, 4)

                # Highlight the exact MID guide waypoint with a diamond marker
                if rec.guide_waypoint is not None:
                    gx, gy = self.viewport.world_to_screen_point((rec.guide_waypoint.x, rec.guide_waypoint.y))
                    igx, igy = int(gx), int(gy)
                    diamond = [(igx, igy - 7), (igx + 7, igy), (igx, igy + 7), (igx - 7, igy)]
                    pygame.draw.polygon(surface, (52, 211, 153), diamond)
                    pygame.draw.polygon(surface, (250, 204, 21), diamond, width=2)

    def _render_hud_and_telemetry(self, surface: pygame.Surface) -> None:
        """Render dedicated right-hand telemetry panel and bottom controls bar."""
        step = self.engine.clock.step_count
        sim_time = self.engine.clock.timestamp

        total_peds = len(self.engine.registry)
        active_count = len([p for p in self.engine.registry if str(p.status).lower().endswith("active")])
        evac_count = len([p for p in self.engine.registry if str(p.status).lower().endswith("evacuated")])
        blocked_count = len([p for p in self.engine.registry if str(p.status).lower().endswith("blocked")])

        speeds = [
            math.hypot(getattr(p.velocity, "vx", getattr(p.velocity, "x", 0.0)),
                       getattr(p.velocity, "vy", getattr(p.velocity, "y", 0.0)))
            for p in self.engine.registry
            if str(p.status).lower().endswith("active") and hasattr(p, "velocity")
        ]
        mean_speed = (sum(speeds) / len(speeds)) if speeds else 0.0

        panel_w = self.SIDE_PANEL_WIDTH
        panel_x = self.width - panel_w
        panel_y = 0
        panel_h = self.height

        panel_bg = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        pygame.draw.rect(surface, (15, 23, 42), panel_bg)
        pygame.draw.line(surface, (51, 65, 85), (panel_x, 0), (panel_x, panel_h), width=2)

        cur_y = 12
        margin_x = panel_x + 18
        content_w = panel_w - 36

        # --- SECTION 0: OPERATING MODE & PRESENTATION BANNER ---
        if self.presentation_mode:
            stage_names = {
                1: "STAGE 1: LIVE EVACUATION",
                2: "STAGE 2: SPATIAL HEATMAP",
                3: "STAGE 3: CONSTANT VELOCITY",
                4: "STAGE 4: MID MULTI-PATH PREDICTION",
                5: "STAGE 5: MID-ASSISTED MOVEMENT",
            }
            stage_title = stage_names.get(self.presentation_stage.value, f"STAGE {self.presentation_stage.value}")
            p_surf = self.font_title.render("PRESENTATION MODE [D to exit]", True, (244, 63, 94))
            s_surf = self.font_section.render(stage_title, True, (250, 204, 21))
            surface.blit(p_surf, (margin_x, cur_y))
            surface.blit(s_surf, (margin_x, cur_y + 19))
            cur_y += 40
        else:
            mode_color = (56, 189, 248) if self.operating_mode == OperatingMode.ORIGINAL_SIMULATION else (
                (192, 132, 252) if self.operating_mode == OperatingMode.MID_PREDICTION_ONLY else (52, 211, 153)
            )
            mode_label = f"MODE: {self.operating_mode.value}"
            m_surf = self.font_section.render(mode_label, True, mode_color)
            surface.blit(m_surf, (margin_x, cur_y))
            cur_y += 22

        # --- SECTION 1: SIMULATION STATISTICS ---
        title_surf = self.font_title.render("SIMULATION TELEMETRY", True, (56, 189, 248))
        surface.blit(title_surf, (margin_x, cur_y))
        cur_y += 20

        status_str = "PAUSED" if self.engine.is_paused else "RUNNING"
        status_col = (251, 191, 36) if self.engine.is_paused else (74, 222, 128)

        stats_rows = [
            ("Clock Time / Step:", f"{sim_time:5.1f}s  |  Step {step}", (241, 245, 249)),
            ("Simulation State:", status_str, status_col),
            ("Active Pedestrians:", f"{active_count} of {total_peds}", (241, 245, 249)),
            ("Evacuated / Blocked:", f"{evac_count} evac  |  {blocked_count} blocked", (148, 163, 184)),
            ("Mean Walking Speed:", f"{mean_speed:.2f} m/s", (241, 245, 249)),
            ("Density Heatmap:", "ACTIVE (plasma)" if self.show_heatmap else "OFF", (203, 213, 225)),
        ]
        for label, val, val_col in stats_rows:
            lbl_s = self.font_hud.render(label, True, (148, 163, 184))
            val_s = self.font_hud.render(val, True, val_col)
            surface.blit(lbl_s, (margin_x, cur_y))
            surface.blit(val_s, (panel_x + panel_w - margin_x + 18 - val_s.get_width() + 140, cur_y))
            cur_y += 16

        cur_y += 6
        pygame.draw.line(surface, (30, 41, 59), (margin_x, cur_y), (margin_x + content_w, cur_y), width=1)
        cur_y += 8

        # --- SECTION 2: SELECTED PEDESTRIAN DETAILS ---
        sec2_surf = self.font_section.render("SELECTED AGENT", True, (250, 204, 21))
        surface.blit(sec2_surf, (margin_x, cur_y))
        cur_y += 18

        target_id = self.selected_agent_id
        target_ped = self.engine.registry.get(target_id) if target_id else None

        if target_ped:
            p_status = str(target_ped.status).split(".")[-1].upper()
            p_pos = f"({target_ped.position.x:.1f}m, {target_ped.position.y:.1f}m)"
            p_spd = math.hypot(
                getattr(target_ped.velocity, "vx", getattr(target_ped.velocity, "x", 0.0)),
                getattr(target_ped.velocity, "vy", getattr(target_ped.velocity, "y", 0.0)),
            )
            ped_rows = [
                ("Agent ID:", str(target_id), (250, 204, 21)),
                ("Status / Position:", f"{p_status} @ {p_pos}", (241, 245, 249)),
                ("Target Exit:", str(target_ped.target_exit_id or "None"), (203, 213, 225)),
                ("Current Speed:", f"{p_spd:.2f} m/s", (241, 245, 249)),
                ("Available Obs:", f"{self.available_observations_count} points (0.4s dt)", (203, 213, 225)),
            ]
        else:
            ped_rows = [
                ("Agent ID:", target_id or "None Selected", (148, 163, 184)),
                ("Status:", "Agent evacuated or inactive", (239, 68, 68)),
            ]

        for label, val, val_col in ped_rows:
            lbl_s = self.font_hud.render(label, True, (148, 163, 184))
            val_s = self.font_hud.render(val, True, val_col)
            surface.blit(lbl_s, (margin_x, cur_y))
            surface.blit(val_s, (margin_x + 130, cur_y))
            cur_y += 16

        cur_y += 6
        pygame.draw.line(surface, (30, 41, 59), (margin_x, cur_y), (margin_x + content_w, cur_y), width=1)
        cur_y += 8

        # --- SECTION 3: PREDICTION PIPELINE STATUS & PROVENANCE ---
        sec3_surf = self.font_section.render("PREDICTION STATUS & PROVENANCE", True, (192, 132, 252))
        surface.blit(sec3_surf, (margin_x, cur_y))
        cur_y += 18

        mid_trained = self.mid_predictor.is_trained
        if mid_trained:
            _epochs = self.mid_predictor._training_epochs_completed
            _samples = self.mid_predictor._training_sample_count
            mid_prov_txt = f"TRAINED ({_epochs} ep, {_samples} smp)"
            mid_prov_col = (74, 222, 128)
        else:
            mid_prov_txt = "UNTRAINED (Random Init)"
            mid_prov_col = (251, 191, 36)

        status_colors = {
            PredictionStatus.WAITING_FOR_HISTORY: (251, 191, 36),
            PredictionStatus.PREDICTING: (56, 189, 248),
            PredictionStatus.ACTIVE: (74, 222, 128),
            PredictionStatus.UNTRAINED: (251, 191, 36),
            PredictionStatus.ERROR: (239, 68, 68),
        }

        pred_rows = [
            ("MID Pipeline State:", self.mid_status.value, status_colors.get(self.mid_status, (241, 245, 249))),
            ("CV Pipeline State:", self.cv_status.value, status_colors.get(self.cv_status, (241, 245, 249))),
            ("Model Provenance:", mid_prov_txt, mid_prov_col),
            ("Generated Trajs:", f"K={self.k_samples} req | {self.actual_returned_k} gen", (241, 245, 249)),
            ("Prediction Horizon:", f"H={self.prediction_horizon} (1.2s future)", (241, 245, 249)),
            ("Inference Time:", f"{self.last_prediction_time * 1000.0:.1f} ms", (203, 213, 225)),
            ("Point Walkability:", f"{self.last_mid_walkability * 100:.1f}% in bounds", (74, 222, 128)),
            ("Segment Collision-Free:", f"{self.last_mid_collision_free * 100:.1f}% clear", (74, 222, 128)),
        ]
        if self.latest_prediction_error:
            pred_rows.append(("Latest Error:", str(self.latest_prediction_error)[:24] + "...", (239, 68, 68)))

        for label, val, val_col in pred_rows:
            lbl_s = self.font_small.render(label, True, (148, 163, 184))
            val_s = self.font_small.render(val, True, val_col)
            surface.blit(lbl_s, (margin_x, cur_y))
            surface.blit(val_s, (margin_x + 130, cur_y))
            cur_y += 15

        cur_y += 6
        pygame.draw.line(surface, (30, 41, 59), (margin_x, cur_y), (margin_x + content_w, cur_y), width=1)
        cur_y += 8

        # --- SECTION 4: MID-ASSISTED DECISION CONTROLLER ---
        sec4_surf = self.font_section.render("MID DECISION CONTROLLER", True, (52, 211, 153))
        surface.blit(sec4_surf, (margin_x, cur_y))
        cur_y += 18

        if self.operating_mode == OperatingMode.MID_ASSISTED_EVACUATION:
            rec = (
                self.controller.latest_recommendations_by_agent.get(target_id)
                if target_id
                else None
            ) or self.latest_recommendation
            if rec:
                self.latest_recommendation = rec
                applied_str = "YES (Waypoint Updated)" if rec.guidance_applied else "NO (Original Route)"
                applied_col = (74, 222, 128) if rec.guidance_applied else (251, 191, 36)
                fb_str = "ACTIVE (Fallback)" if rec.fallback_to_original else "NONE (MID Active)"
                fb_col = (251, 191, 36) if rec.fallback_to_original else (74, 222, 128)
                sel_cand_str = (
                    f"Sample #{rec.selected_sample_idx}"
                    if rec.selected_sample_idx >= 0
                    else "None (Fallback)"
                )
                ctrl_rows = [
                    ("Valid Candidates:", f"{rec.valid_candidates_count} of {rec.evaluated_samples_count} feasible", (241, 245, 249)),
                    ("Selected Candidate:", sel_cand_str, (250, 204, 21)),
                    ("Guidance Applied:", applied_str, applied_col),
                    ("Fallback Status:", fb_str, fb_col),
                    ("Decision Reason:", str(rec.selection_reason)[:28], (203, 213, 225)),
                    ("Control Update Time:", f"{rec.control_compute_time_s * 1000.0:.2f} ms", (203, 213, 225)),
                    ("Cohort Interventions:", f"{self.controller.accepted_interventions} ok | {self.controller.fallback_count} fb", (203, 213, 225)),
                ]
            else:
                ctrl_rows = [
                    ("Guidance Applied:", "WAITING FOR OBS TICK", (251, 191, 36)),
                    ("Fallback Status:", "Original Route Active", (203, 213, 225)),
                ]
        else:
            ctrl_rows = [
                ("Guidance Applied:", "NO (Mode is Read-Only)", (148, 163, 184)),
                ("Fallback Status:", "Baseline Engine Active", (148, 163, 184)),
                ("Activate Controller:", "Press [M] -> MID-ASSISTED", (52, 211, 153)),
            ]

        for label, val, val_col in ctrl_rows:
            lbl_s = self.font_small.render(label, True, (148, 163, 184))
            val_s = self.font_small.render(val, True, val_col)
            surface.blit(lbl_s, (margin_x, cur_y))
            surface.blit(val_s, (margin_x + 130, cur_y))
            cur_y += 15

        cur_y += 6
        pygame.draw.line(surface, (30, 41, 59), (margin_x, cur_y), (margin_x + content_w, cur_y), width=1)
        cur_y += 8

        # --- SECTION 5: TRAJECTORY VISUAL LEGEND ---
        sec5_surf = self.font_section.render("TRAJECTORY LEGEND", True, (241, 245, 249))
        surface.blit(sec5_surf, (margin_x, cur_y))
        cur_y += 18

        legend_entries = [
            ("Observed History", (245, 158, 11), "dotted"),
            ("CV Baseline", (241, 245, 249), "dashed"),
            ("MID Candidate Paths (K)", self.mid_colors[0], "solid"),
            ("Selected MID Candidate", (250, 204, 21), "selected"),
            ("MID Guide Route", (52, 211, 153), "solid"),
            ("Rejected Candidate", (248, 113, 113), "rejected"),
        ]

        for name, col, style in legend_entries:
            line_y = cur_y + 6
            if style == "dotted":
                pygame.draw.circle(surface, col, (margin_x + 6, line_y), 2)
                pygame.draw.circle(surface, col, (margin_x + 14, line_y), 2)
                pygame.draw.circle(surface, col, (margin_x + 22, line_y), 2)
            elif style == "dashed":
                pygame.draw.line(surface, col, (margin_x, line_y), (margin_x + 10, line_y), width=2)
                pygame.draw.line(surface, col, (margin_x + 16, line_y), (margin_x + 26, line_y), width=2)
            elif style == "selected":
                pygame.draw.line(surface, (6, 95, 70), (margin_x, line_y), (margin_x + 26, line_y), width=5)
                pygame.draw.line(surface, col, (margin_x, line_y), (margin_x + 26, line_y), width=3)
                pygame.draw.circle(surface, col, (margin_x + 26, line_y), 4)
            elif style == "rejected":
                pygame.draw.line(surface, col, (margin_x, line_y), (margin_x + 18, line_y), width=1)
                pygame.draw.line(surface, col, (margin_x + 21, line_y - 3), (margin_x + 27, line_y + 3), width=2)
                pygame.draw.line(surface, col, (margin_x + 21, line_y + 3), (margin_x + 27, line_y - 3), width=2)
            else:
                pygame.draw.line(surface, col, (margin_x, line_y), (margin_x + 26, line_y), width=2)
                pygame.draw.circle(surface, col, (margin_x + 26, line_y), 3)

            lbl = self.font_small.render(name, True, (203, 213, 225))
            surface.blit(lbl, (margin_x + 36, cur_y))
            cur_y += 15

        # --- BOTTOM INTERACTIVE CONTROLS BAR ---
        ctrl_h = 36
        ctrl_y = self.height - ctrl_h
        ctrl_bg = pygame.Rect(0, ctrl_y, self.width, ctrl_h)
        pygame.draw.rect(surface, (15, 23, 42), ctrl_bg)
        pygame.draw.line(surface, (51, 65, 85), (0, ctrl_y), (self.width, ctrl_y), width=1)

        if self.presentation_mode:
            ctrl_text = (
                "[LEFT / RIGHT] Prev/Next Stage   [ENTER] Confirm Stage   "
                "[SPACE] Pause/Resume   [R] Reset Sim   [D] Exit Presentation Mode   [Q] Quit"
            )
        else:
            ctrl_text = (
                "[M] Mode Toggle   [D] Presentation Mode   [SPACE] Pause/Resume   [R] Reset   "
                "[H] Heatmap   [P] MID Fan   [B] Baseline   [I] IDs   [UP/DOWN] K   [CLICK] Select"
            )
        ctrl_surf = self.font_legend.render(ctrl_text, True, (203, 213, 225))
        surface.blit(ctrl_surf, (24, ctrl_y + 10))

    # =========================================================================
    # Event Pump & User Input Handling
    # =========================================================================

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle mouse clicks, keyboard shortcuts, and window resize."""
        if event.type == pygame.QUIT:
            return False

        elif event.type == pygame.VIDEORESIZE:
            self.width = max(800, event.w)
            self.height = max(500, event.h)
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            sim_screen_w = max(400, self.width - self.SIDE_PANEL_WIDTH)
            self.viewport.update_screen_size(sim_screen_w, self.height)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if mx >= (self.width - self.SIDE_PANEL_WIDTH):
                return True

            best_id = None
            min_dist = float("inf")
            for ped in self.engine.registry:
                sx, sy = self.viewport.world_to_screen_point(ped.position)
                dist = math.hypot(sx - mx, sy - my)
                if dist < min_dist and dist < 32.0:
                    min_dist = dist
                    best_id = ped.id
            if best_id and best_id != self.selected_agent_id:
                self.selected_agent_id = best_id
                # Invalidate prediction cache on selection change
                self._cached_mid_key = None
                self._cached_cv_key = None

        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False

            elif event.key == pygame.K_SPACE:
                if self.engine.is_paused:
                    self.engine.resume()
                else:
                    self.engine.pause()

            elif event.key == pygame.K_r:
                self.reset_simulation()

            elif event.key == pygame.K_d:
                # Toggle 5-Stage Live Presentation Mode
                self.presentation_mode = not self.presentation_mode
                if self.presentation_mode:
                    self.set_presentation_stage(PresentationStage.STAGE_1_LIVE_EVACUATION)

            elif self.presentation_mode:
                # Presentation Mode specific controls
                if event.key == pygame.K_RIGHT:
                    next_val = min(5, self.presentation_stage.value + 1)
                    self.set_presentation_stage(PresentationStage(next_val))
                elif event.key == pygame.K_LEFT:
                    prev_val = max(1, self.presentation_stage.value - 1)
                    self.set_presentation_stage(PresentationStage(prev_val))
                elif event.key == pygame.K_RETURN:
                    self.set_presentation_stage(self.presentation_stage)

            elif not self.presentation_mode:
                # Regular Mode toggles
                if event.key == pygame.K_m:
                    # Cycle through Operating Modes
                    modes = list(OperatingMode)
                    idx = modes.index(self.operating_mode)
                    self.operating_mode = modes[(idx + 1) % len(modes)]
                    if self.operating_mode == OperatingMode.MID_ASSISTED_EVACUATION:
                        self._apply_mid_assisted_guidance()
                elif event.key == pygame.K_h:
                    self.show_heatmap = not self.show_heatmap
                elif event.key == pygame.K_p:
                    self.show_mid_predictions = not self.show_mid_predictions
                elif event.key == pygame.K_b:
                    self.show_baseline_predictions = not self.show_baseline_predictions
                elif event.key == pygame.K_i:
                    self.show_ids = not self.show_ids
                elif event.key == pygame.K_UP:
                    self.k_samples = min(10, self.k_samples + 1)
                    self._cached_mid_key = None
                    self._mid_cache_map.clear()
                elif event.key == pygame.K_DOWN:
                    self.k_samples = max(1, self.k_samples - 1)
                    self._cached_mid_key = None
                    self._mid_cache_map.clear()

        return True

    def set_presentation_stage(self, stage: PresentationStage) -> None:
        """Apply deterministic configuration presets for the 5 presentation stages."""
        self.presentation_stage = stage
        if stage == PresentationStage.STAGE_1_LIVE_EVACUATION:
            self.operating_mode = OperatingMode.ORIGINAL_SIMULATION
            self.show_heatmap = False
            self.show_mid_predictions = False
            self.show_baseline_predictions = False
            if self.engine.is_paused:
                self.engine.resume()

        elif stage == PresentationStage.STAGE_2_SPATIAL_HEATMAP:
            self.operating_mode = OperatingMode.ORIGINAL_SIMULATION
            self.show_heatmap = True
            self.show_mid_predictions = False
            self.show_baseline_predictions = False

        elif stage == PresentationStage.STAGE_3_CONSTANT_VELOCITY:
            self.operating_mode = OperatingMode.MID_PREDICTION_ONLY
            self.show_heatmap = False
            self.show_mid_predictions = False
            self.show_baseline_predictions = True
            if not self.selected_agent_id:
                self.selected_agent_id = "ped_1"

        elif stage == PresentationStage.STAGE_4_MID_PREDICTION:
            self.operating_mode = OperatingMode.MID_PREDICTION_ONLY
            self.show_heatmap = False
            self.show_mid_predictions = True
            self.show_baseline_predictions = True
            self.k_samples = 6
            if not self.selected_agent_id:
                self.selected_agent_id = "ped_1"

        elif stage == PresentationStage.STAGE_5_MID_ASSISTED_DEMONSTRATION:
            self.operating_mode = OperatingMode.MID_ASSISTED_EVACUATION
            self.show_heatmap = False
            self.show_mid_predictions = True
            self.show_baseline_predictions = True
            self.k_samples = 6
            if not self.selected_agent_id:
                self.selected_agent_id = "ped_1"
            if self.engine.is_paused:
                self.engine.resume()
            self._apply_mid_assisted_guidance()

    # =========================================================================
    # Application Main Loop
    # =========================================================================

    def run(self, max_frames: Optional[int] = None) -> None:
        """Run the interactive Pygame loop with full integration."""
        if not pygame.get_init():
            pygame.init()
        if not pygame.display.get_init():
            pygame.display.init()
        if not pygame.font.get_init():
            pygame.font.init()

        pygame.display.set_caption(
            f"GenEvac-Sentinel — {self.building.id} ({self.building.width:.0f}m x {self.building.height:.0f}m) [Module B Part 1 + Part 2]"
        )

        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.Font(None, 22)
        self.font_section = pygame.font.Font(None, 20)
        self.font_hud = pygame.font.Font(None, 18)
        self.font_small = pygame.font.Font(None, 16)
        self.font_legend = pygame.font.Font(None, 16)

        running = True
        frames = 0

        while running:
            dt_millis = self.clock.tick(self.fps)
            dt_wall = dt_millis / 1000.0

            for event in pygame.event.get():
                if not self.handle_event(event):
                    running = False
                    break

            if not running:
                break

            # 1. Advance simulation physics (Module B Part 1)
            self.update_physics(dt_wall)

            # 2. Draw building floor, walls, obstacles, exits (Module B Part 1)
            self.renderer.draw(self.screen, self.building, self.viewport, engine=self.engine)

            # 3. Draw spatial density heatmap under pedestrians (Module B Part 2)
            self._render_spatial_heatmap(self.screen)

            # 4. Redraw pedestrians with custom ID toggle
            self.renderer._draw_pedestrians(self.screen, self.engine.registry, self.viewport)

            # 5. Draw MID trajectory predictions & baseline (Module B Part 2)
            self._render_predictions(self.screen)

            # 6. Draw telemetry HUD & controls (Module B Part 2)
            self._render_hud_and_telemetry(self.screen)

            pygame.display.flip()
            frames += 1

            if max_frames is not None and frames >= max_frames:
                break

        pygame.quit()


# =============================================================================
# 5. Launch Entrypoint
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GenEvac-Sentinel: Integrated 2D Simulation (Module B Part 1 + Part 2)"
    )
    parser.add_argument(
        "--map",
        type=str,
        default=str(PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json"),
        help="Path to building JSON layout file",
    )
    parser.add_argument("--width", type=int, default=1440, help="Window width (default: 1440)")
    parser.add_argument("--height", type=int, default=860, help="Window height (default: 860)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["original", "prediction", "assisted"],
        default="original",
        help="Initial operating mode: original, prediction, or assisted",
    )
    parser.add_argument(
        "--presentation",
        action="store_true",
        help="Launch directly into 5-stage Presentation Mode",
    )
    args = parser.parse_args()

    app = IntegratedSimulationApp(
        map_path=args.map, width=args.width, height=args.height, fps=args.fps
    )
    if args.mode == "prediction":
        app.operating_mode = OperatingMode.MID_PREDICTION_ONLY
    elif args.mode == "assisted":
        app.operating_mode = OperatingMode.MID_ASSISTED_EVACUATION

    if args.presentation:
        app.presentation_mode = True
        app.set_presentation_stage(PresentationStage.STAGE_1_LIVE_EVACUATION)

    app.run()


if __name__ == "__main__":
    main()
