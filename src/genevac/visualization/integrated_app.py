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
from collections import deque
from dataclasses import dataclass
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
import torch

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
import genevac.prediction.social_training as h_social_train
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
    "check_pedestrian_collision": p_movement.check_pedestrian_collision,
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
    MID_DRIVEN = "MID-DRIVEN"


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


@dataclass(frozen=True)
class DecisionTraceRecord:
    """Structured controller traceability record for one evaluation cycle."""
    agent_id: str
    step_index: int
    timestamp: float
    candidate_count_requested: int
    candidate_count_generated: int
    candidate_ids: Tuple[str, ...]
    valid_candidate_ids: Tuple[str, ...]
    rejected_candidates: Tuple[Tuple[str, str], ...]
    selected_sample_idx: int
    selected_candidate_id: Optional[str]
    selected_endpoint: Optional[Tuple[float, float]]
    is_accepted: bool
    guidance_applied: bool
    fallback_to_original: bool
    decision_reason: str
    rejection_reason: Optional[str]
    proposed_exit_id: Optional[str]
    proposed_waypoints: Tuple[Tuple[float, float], ...]
    guide_waypoint: Optional[Tuple[float, float]]


@dataclass(frozen=True)
class AdoptedRouteRecord:
    """Verified engine-side route adoption status for one pedestrian."""
    agent_id: str
    decision_step_index: int
    decision_timestamp: float
    target_exit_id: Optional[str]
    proposed_waypoints: Tuple[Tuple[float, float], ...]
    waypoints_after_adoption: Tuple[Tuple[float, float], ...]
    exit_after_adoption: Optional[str]
    adoption_attempted: bool
    adoption_succeeded: bool
    adoption_status: str
    reason: str


# =============================================================================
# 3. Helper: Pre-Train Evacuation MID Model
# =============================================================================

def create_pretrained_mid_predictor(k_samples: int = 6, horizon: int = 12) -> h_mid.EvacuationMIDPredictor:
    """Instantiate and deterministically pre-train Evacuation MID on multi-agent social rollouts."""
    mid_config = h_mid.EvacuationMIDConfig(
        horizon=horizon,
        prediction_interval=0.1,
        k_samples=k_samples,
        diffusion_steps=15,
        beta_end=0.22,
        hidden_dim=128,
        w_exit=0.25,
        w_obs=0.45,
        w_kin=0.30,
    )
    train_samples, scenarios = h_social_train.generate_social_interaction_training_samples(
        horizon=horizon,
        history_length=4,
        dt=0.1,
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        predictor = h_mid.EvacuationMIDPredictor(config=mid_config)
        try:
            predictor.fit(
                train_samples,
                epochs=10,
                batch_size=8,
                training_source=h_social_train.TRAINING_SOURCE_NAME,
                interaction_scenarios=scenarios,
            )
        except Exception as e:
            predictor._training_error = f"{type(e).__name__}: {e}"
            print(f"[Error] MID social pre-training failed (model remains untrained): {e}")

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
        mid_predictor: Optional[h_mid.EvacuationMIDPredictor] = None,
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
        self.renderer.draw_labels = False
        self.renderer._draw_pedestrians = self._draw_pedestrians

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
        self.mid_predictor = (
            mid_predictor
            if mid_predictor is not None
            else create_pretrained_mid_predictor(
                k_samples=self.k_samples, horizon=self.prediction_horizon
            )
        )

        # Part D: MID-Assisted & MID-Driven Decision Controllers (strictly decoupled safety layer)
        self.controller = h_controller.MIDAssistedDecisionAdapter(
            building=self.building,
            min_decision_interval=0.4,
        )
        self.driven_controller = h_controller.MIDDrivenMovementController(
            building=self.building,
        )
        self._driven_history_buffer = h_adapter.SimulationTrajectoryAdapter(
            observation_interval=self.dt,
            min_history_required=2,
            window_size=30,
        )
        self._latest_driven_mid: Dict[str, h_mid.MultiTrajectoryPrediction] = {}
        self.latest_recommendation: Optional[h_controller.MIDDecisionRecommendation] = None
        self.base_seed: int = 42
        self._mid_training_error: Optional[str] = getattr(self.mid_predictor, "_training_error", None)

        # Parallel read-only ORIGINAL reference engine for live visual & HUD movement comparison
        self._baseline_ref_building: Any = None
        self._baseline_ref_engine: Any = None
        self._baseline_ref_paths: Dict[str, List[Tuple[int, float, float, float]]] = {}

        # Operating Modes
        self.operating_mode: OperatingMode = OperatingMode.ORIGINAL_SIMULATION
        self.presentation_mode: bool = False
        self.presentation_stage: PresentationStage = PresentationStage.STAGE_1_LIVE_EVACUATION

        # State toggles (purely visual; never alter physics or controller policy)
        self.show_heatmap = True
        self.show_mid_predictions = True
        self.show_baseline_predictions = True
        self.show_actual_path = True
        self.show_observed_history = True
        self.show_adopted_route = True
        self.show_developer_overlay = False
        self.show_layer_labels = False
        self.show_social_interactions = True
        self.show_safety_overlays = True
        self.show_baseline_ghost = True
        self.selected_agent_id: Optional[str] = "ped_1"

        # Authoritative movement, decision, and adoption traceability stores
        self.actual_traversed_paths: Dict[str, List[Tuple[int, float, float, float]]] = {}
        self.decision_traces_by_agent: Dict[str, DecisionTraceRecord] = {}
        self.adopted_route_records_by_agent: Dict[str, AdoptedRouteRecord] = {}
        self.causal_trace_log: deque[Dict[str, Any]] = deque(maxlen=200)
        self._pending_causal_steps: Dict[str, Dict[str, Any]] = {}
        self.latest_layer_states: Dict[str, Dict[str, Any]] = {}
        self._placed_label_rects: List[pygame.Rect] = []

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
        # (Green is reserved exclusively for Exits and Layer 6 Actual Adopted Route)
        self.mid_colors = [
            (56, 189, 248),    # Electric Cyan
            (244, 63, 94),     # Neon Rose / Coral Red
            (168, 85, 247),    # Vivid Purple
            (232, 121, 249),   # Fuchsia Orchid
            (249, 115, 22),    # Amber Orange
            (236, 72, 153),    # Hot Pink
            (20, 184, 166),    # Deep Teal
            (99, 102, 241),    # Indigo
            (234, 179, 8),     # Bright Yellow
            (139, 92, 246),    # Soft Violet
        ]

        # Pygame surfaces & fonts
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font_title: Any = None
        self.font_section: Any = None
        self.font_hud: Any = None
        self.font_small: Any = None
        self.font_legend: Any = None

        # Build initial simulation state
        self.reset_simulation()
        if pygame.get_init() and pygame.font.get_init():
            self._ensure_fonts()

    def _ensure_fonts(self) -> None:
        """Ensure pygame fonts are initialized before rendering."""
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        if self.font_title is None:
            self.font_title = pygame.font.Font(None, 22)
        if self.font_section is None:
            self.font_section = pygame.font.Font(None, 20)
        if self.font_hud is None:
            self.font_hud = pygame.font.Font(None, 18)
        if self.font_small is None:
            self.font_small = pygame.font.Font(None, 16)
        if self.font_legend is None:
            self.font_legend = pygame.font.Font(None, 16)

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

        # Reset Part 2 adapter, step history buffer, heatmap engine & controllers
        self.adapter.reset()
        self._driven_history_buffer.reset()
        self.heatmap_engine.reset()
        self.controller.reset()
        self.driven_controller.reset()
        self._latest_driven_mid.clear()

        # Parallel isolated ORIGINAL baseline engine for live movement delta comparison
        self._baseline_ref_building = PAVAN["load_building"](self.map_path)
        for ex in self._baseline_ref_building.exits:
            ex.open()
        ref_cohort = PAVAN["create_default_pedestrian_cohort"]()
        ref_registry = PAVAN["PedestrianRegistry"]()
        for rped in ref_cohort:
            ref_registry.register(rped)
        self._baseline_ref_engine = PAVAN["SimulationEngine"](
            building=self._baseline_ref_building,
            registry=ref_registry,
            dt=self.dt,
            observation_interval=self.observation_interval,
        )
        self._baseline_ref_paths.clear()

        # Clear prediction caches & traceability histories
        self._cached_mid_key = None
        self._cached_mid_result = None
        self._mid_cache_map.clear()
        self._cached_cv_key = None
        self._cached_cv_result = None
        self.latest_recommendation = None
        self.actual_returned_k = 0
        self.available_observations_count = 0
        self.latest_prediction_error = self._mid_training_error
        if self._mid_training_error:
            self.mid_status = PredictionStatus.ERROR
        elif not self.mid_predictor.is_trained:
            self.mid_status = PredictionStatus.UNTRAINED
        else:
            self.mid_status = PredictionStatus.WAITING_FOR_HISTORY
        self.cv_status = PredictionStatus.WAITING_FOR_HISTORY
        self.actual_traversed_paths.clear()
        self.decision_traces_by_agent.clear()
        self.adopted_route_records_by_agent.clear()
        self.causal_trace_log.clear()
        self._pending_causal_steps.clear()
        self.latest_layer_states.clear()

        # Ingest initial snapshot at step 0 and record initial engine & baseline coordinates
        self.adapter.adapt_snapshot(self.engine.latest_snapshot)
        self._driven_history_buffer.adapt_snapshot(self.engine.latest_snapshot)
        self._record_engine_positions()
        self._record_baseline_positions()
        self.sim_accumulator = 0.0

        if cohort:
            self.selected_agent_id = cohort[0].id
        else:
            self.selected_agent_id = None

    def _record_baseline_positions(self) -> None:
        """Read-only recording of parallel unmodified ORIGINAL baseline positions."""
        if self._baseline_ref_engine is None:
            return
        step = int(self._baseline_ref_engine.clock.step_count)
        ts = float(self._baseline_ref_engine.clock.timestamp)
        for ped in self._baseline_ref_engine.registry:
            pos = getattr(ped, "position", None)
            if pos is None:
                continue
            px, py = getattr(pos, "x", None), getattr(pos, "y", None)
            if not self._is_finite_xy(px, py):
                continue
            assert px is not None and py is not None
            pts = self._baseline_ref_paths.setdefault(ped.id, [])
            if not pts or pts[-1][0] < step:
                pts.append((step, ts, float(px), float(py)))

    def _compute_baseline_divergence(self, target_id: Optional[str]) -> Tuple[float, float, float]:
        """Return (selected_agent_delta_m, cohort_mean_delta_m, cohort_max_delta_m) vs ORIGINAL."""
        if self.engine is None or self._baseline_ref_engine is None:
            return 0.0, 0.0, 0.0
        deltas: List[float] = []
        sel_delta = 0.0
        for ped in self.engine.registry:
            ref_ped = self._baseline_ref_engine.registry.get(ped.id)
            if ref_ped is None:
                continue
            d = math.hypot(
                float(ped.position.x) - float(ref_ped.position.x),
                float(ped.position.y) - float(ref_ped.position.y),
            )
            deltas.append(d)
            if target_id is not None and ped.id == target_id:
                sel_delta = d
        if not deltas:
            return 0.0, 0.0, 0.0
        return sel_delta, float(sum(deltas) / len(deltas)), float(max(deltas))

    def _restore_baseline_routes_if_needed(self) -> None:
        """Strip transient MID-injected waypoints cleanly when switching back to ORIGINAL/PREDICTION."""
        if self.engine is None:
            return
        for ped in self.engine.registry:
            aid = ped.id
            clean_wps = self.driven_controller.get_base_waypoints(aid, ped)
            if len(clean_wps) != len(getattr(ped, "waypoints", [])):
                ped.waypoints = list(clean_wps)
            clean_wps2 = self.controller.get_base_waypoints(aid, ped)
            if len(clean_wps2) != len(getattr(ped, "waypoints", [])):
                ped.waypoints = list(clean_wps2)
        self.driven_controller._active_step_waypoint.clear()
        self.controller._active_mid_waypoint.clear()

    @staticmethod
    def _is_finite_xy(x: Any, y: Any) -> bool:
        return (
            isinstance(x, (int, float))
            and not isinstance(x, bool)
            and isinstance(y, (int, float))
            and not isinstance(y, bool)
            and math.isfinite(float(x))
            and math.isfinite(float(y))
        )

    def _record_engine_positions(self) -> None:
        """Read-only recording of actual per-step pedestrian coordinates from SimulationEngine.registry."""
        if self.engine is None:
            return
        step = int(self.engine.clock.step_count)
        ts = float(self.engine.clock.timestamp)
        for ped in self.engine.registry:
            pos = getattr(ped, "position", None)
            if pos is None:
                continue
            px = getattr(pos, "x", None)
            py = getattr(pos, "y", None)
            if not self._is_finite_xy(px, py):
                continue
            assert px is not None and py is not None
            fx, fy = float(px), float(py)
            pts = self.actual_traversed_paths.setdefault(ped.id, [])
            if not pts or pts[-1][0] < step:
                pts.append((step, ts, fx, fy))

    def _complete_pending_causal_traces(self) -> None:
        """Finalize pending causal traces after the subsequent engine physics step completes."""
        if self.engine is None or not self._pending_causal_steps:
            return
        step = int(self.engine.clock.step_count)
        ts = float(self.engine.clock.timestamp)
        for aid in list(self._pending_causal_steps.keys()):
            pending = self._pending_causal_steps[aid]
            if step <= pending["decision_step"]:
                continue
            ped = self.engine.registry.get(aid)
            if ped is None:
                self._pending_causal_steps.pop(aid, None)
                continue
            post_pos = (float(ped.position.x), float(ped.position.y))
            pre_pos = pending["pre_step_position"]
            moved_dist = math.hypot(post_pos[0] - pre_pos[0], post_pos[1] - pre_pos[1])
            post_wps = tuple(
                (float(wp.x), float(wp.y))
                for wp in (getattr(ped, "waypoints", []) or [])
                if self._is_finite_xy(getattr(wp, "x", None), getattr(wp, "y", None))
            )
            entry = dict(pending)
            entry.update(
                {
                    "post_step_index": step,
                    "post_timestamp": ts,
                    "post_step_position": post_pos,
                    "post_step_waypoints": post_wps,
                    "post_step_status": str(getattr(ped.status, "value", ped.status)),
                    "moved_distance_m": moved_dist,
                }
            )
            self.causal_trace_log.append(entry)
            self._pending_causal_steps.pop(aid, None)

    def update_physics(self, dt_wall: float) -> int:
        """Advance Pavan's SimulationEngine, adapt snapshots into Part 2 buffers, and execute active controllers."""
        if self.engine.is_paused:
            return 0

        clamped_dt = max(0.0, min(dt_wall, self.max_dt_wall))
        self.sim_accumulator += clamped_dt

        steps_taken = 0
        while self.sim_accumulator >= self.dt and steps_taken < self.max_catchup_steps:
            # In MID-DRIVEN mode, MID generates primary movement & safety checks BEFORE Pavan executes step()
            if self.operating_mode == OperatingMode.MID_DRIVEN:
                self._apply_mid_driven_step()

            # Advance Pavan's authoritative physics
            self.engine.step()
            if self._baseline_ref_engine is not None and not self._baseline_ref_engine.is_paused:
                self._baseline_ref_engine.step()

            self.sim_accumulator -= self.dt
            steps_taken += 1

            # Record actual executed movement from authoritative engine & baseline reference
            self._record_engine_positions()
            self._record_baseline_positions()
            self._complete_pending_causal_traces()

            # Always maintain step-resolution read-only history buffer alongside 0.4s prediction adapter
            self._driven_history_buffer.adapt_snapshot(self.engine.latest_snapshot)

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

    def _apply_mid_driven_step(self) -> None:
        """Run closed-loop MID-driven trajectory generation and safety enforcement per physics step."""
        step = int(self.engine.clock.step_count)
        timestamp = float(self.engine.clock.timestamp)
        driven_snap = self._driven_history_buffer.buffer.get_snapshot(step, timestamp)
        open_exits = [ex.id for ex in self.building.exits if ex.is_open]
        all_peds = list(self.engine.registry)

        # Compute actual finite-diff neighbor velocities from recent observations (avoids raw preferred_speed bias)
        neighbor_vels: Dict[str, Tuple[float, float]] = {}
        for aid_n, obs_seq in driven_snap.trajectories.items():
            if len(obs_seq) >= 2:
                o_prev, o_curr = obs_seq[-2], obs_seq[-1]
                ndt = float(o_curr.timestamp) - float(o_prev.timestamp)
                if ndt > 1e-6:
                    neighbor_vels[aid_n] = (
                        (float(o_curr.x) - float(o_prev.x)) / ndt,
                        (float(o_curr.y) - float(o_prev.y)) / ndt,
                    )

        for ped in all_peds:
            aid = ped.id
            status_val = str(getattr(ped.status, "value", ped.status)).lower()
            if status_val not in ("active", "blocked"):
                continue
            if aid not in driven_snap.trajectories or len(driven_snap.trajectories[aid]) < 2:
                continue

            # Refresh MID predictions on cadence (0.4s normal, 0.2s during close social proximity)
            min_nbr_dist = min(
                (
                    math.hypot(other.position.x - ped.position.x, other.position.y - ped.position.y)
                    for other in all_peds
                    if other.id != aid and str(getattr(other.status, "value", other.status)).lower() == "active"
                ),
                default=50.0,
            )
            cadence = 2 if min_nbr_dist < 2.4 else 4
            cached_mid = self._latest_driven_mid.get(aid)
            if cached_mid is None or (step % cadence == 0):
                mid_res = self._get_or_compute_mid_prediction(driven_snap, aid, step)
                if mid_res is not None:
                    self._latest_driven_mid[aid] = mid_res
            else:
                mid_res = cached_mid

            if mid_res is not None:
                # Ensure aligned key at exact current step so visual Layer 4/5 stays consistent
                self._mid_cache_map[(aid, step, self.k_samples, self.prediction_horizon)] = mid_res
                if aid == self.selected_agent_id:
                    self._cached_mid_key = (aid, step, self.k_samples, self.prediction_horizon)
                    self._cached_mid_result = mid_res

            cmd_wps, rec, safety_res = self.driven_controller.compute_driven_command(
                agent_id=aid,
                ped=ped,
                mid_prediction=mid_res,
                open_exit_ids=open_exits,
                dt=self.dt,
                current_time=timestamp,
                step_index=step,
                generate_route_fn=PAVAN["generate_deterministic_route"],
                is_walkable_fn=PAVAN["is_position_in_walkable_space"],
                check_collision_fn=PAVAN["check_collision"],
                other_pedestrians=all_peds,
                neighbor_velocities=neighbor_vels,
            )

            sample_cnt = len(mid_res.samples) if mid_res else 0
            candidate_ids = tuple(f"C{i}" for i in range(sample_cnt))
            valid_ids = tuple(c.candidate_id for c in rec.candidates if c.is_valid)
            rejected_pairs = tuple(
                (c.candidate_id, c.rejection_reason or "rejected")
                for c in rec.candidates
                if not c.is_valid
            )
            selected_ep: Optional[Tuple[float, float]] = None
            if mid_res and 0 <= rec.selected_sample_idx < len(mid_res.samples):
                sel_sample = mid_res.samples[rec.selected_sample_idx]
                if sel_sample.predictions:
                    last_p = sel_sample.predictions[-1]
                    if self._is_finite_xy(last_p.x, last_p.y):
                        selected_ep = (float(last_p.x), float(last_p.y))

            proposed_wps_xy = tuple(
                (float(wp.x), float(wp.y))
                for wp in cmd_wps
                if self._is_finite_xy(getattr(wp, "x", None), getattr(wp, "y", None))
            )
            gw_xy = (
                (float(rec.guide_waypoint.x), float(rec.guide_waypoint.y))
                if (rec.guide_waypoint is not None and self._is_finite_xy(getattr(rec.guide_waypoint, "x", None), getattr(rec.guide_waypoint, "y", None)))
                else None
            )

            trace_rec = DecisionTraceRecord(
                agent_id=aid,
                step_index=step,
                timestamp=float(timestamp),
                candidate_count_requested=self.k_samples,
                candidate_count_generated=sample_cnt,
                candidate_ids=candidate_ids,
                valid_candidate_ids=valid_ids,
                rejected_candidates=rejected_pairs,
                selected_sample_idx=rec.selected_sample_idx,
                selected_candidate_id=rec.selected_candidate_id,
                selected_endpoint=selected_ep,
                is_accepted=rec.is_accepted,
                guidance_applied=rec.guidance_applied,
                fallback_to_original=rec.fallback_to_original,
                decision_reason=rec.selection_reason,
                rejection_reason=rec.rejection_reason,
                proposed_exit_id=rec.target_exit_id,
                proposed_waypoints=proposed_wps_xy,
                guide_waypoint=gw_xy,
            )
            self.decision_traces_by_agent[aid] = trace_rec

            if cmd_wps:
                pre_pos = (float(ped.position.x), float(ped.position.y))
                if rec.target_exit_id:
                    ped.target_exit_id = rec.target_exit_id
                ped.waypoints = list(cmd_wps)
                if status_val == "blocked":
                    ped.status = PAVAN["PedestrianStatus"].ACTIVE

                verified_wps = tuple(
                    (float(wp.x), float(wp.y))
                    for wp in (getattr(ped, "waypoints", []) or [])
                    if self._is_finite_xy(getattr(wp, "x", None), getattr(wp, "y", None))
                )
                self.adopted_route_records_by_agent[aid] = AdoptedRouteRecord(
                    agent_id=aid,
                    decision_step_index=step,
                    decision_timestamp=float(timestamp),
                    target_exit_id=rec.target_exit_id,
                    proposed_waypoints=proposed_wps_xy,
                    waypoints_after_adoption=verified_wps,
                    exit_after_adoption=getattr(ped, "target_exit_id", None),
                    adoption_attempted=True,
                    adoption_succeeded=bool(verified_wps),
                    adoption_status="ADOPTED" if verified_wps else "NOT_ADOPTED_BY_ENGINE",
                    reason=(
                        f"MID-DRIVEN step active ({safety_res.intervention_reason})"
                        if safety_res.intervention_applied
                        else "MID-DRIVEN trajectory step adopted"
                    ),
                )
                self._pending_causal_steps[aid] = {
                    "decision_step": step,
                    "decision_timestamp": float(timestamp),
                    "agent_id": aid,
                    "mid_candidate_count": sample_cnt,
                    "selected_candidate_id": trace_rec.selected_candidate_id,
                    "selected_sample_idx": rec.selected_sample_idx,
                    "selected_endpoint": selected_ep,
                    "guide_waypoint": gw_xy,
                    "proposed_waypoints": proposed_wps_xy,
                    "adoption_status": "ADOPTED",
                    "adoption_succeeded": bool(verified_wps),
                    "adopted_waypoints": verified_wps,
                    "pre_step_position": pre_pos,
                }

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

            candidate_ids = tuple(f"C{i}" for i in range(len(mid_res.samples)))
            valid_ids = tuple(c.candidate_id for c in rec.candidates if c.is_valid)
            rejected_pairs = tuple(
                (c.candidate_id, c.rejection_reason or "rejected")
                for c in rec.candidates
                if not c.is_valid
            )
            selected_ep: Optional[Tuple[float, float]] = None
            if rec.is_accepted and 0 <= rec.selected_sample_idx < len(mid_res.samples):
                sel_sample = mid_res.samples[rec.selected_sample_idx]
                if sel_sample.predictions:
                    last_p = sel_sample.predictions[-1]
                    if self._is_finite_xy(last_p.x, last_p.y):
                        selected_ep = (float(last_p.x), float(last_p.y))

            proposed_wps_xy = tuple(
                (float(wp.x), float(wp.y))
                for wp in rec.proposed_waypoints
                if self._is_finite_xy(getattr(wp, "x", None), getattr(wp, "y", None))
            )
            gw_xy = (
                (float(rec.guide_waypoint.x), float(rec.guide_waypoint.y))
                if (rec.guide_waypoint is not None and self._is_finite_xy(getattr(rec.guide_waypoint, "x", None), getattr(rec.guide_waypoint, "y", None)))
                else None
            )

            trace_rec = DecisionTraceRecord(
                agent_id=aid,
                step_index=step,
                timestamp=float(timestamp),
                candidate_count_requested=self.k_samples,
                candidate_count_generated=len(mid_res.samples),
                candidate_ids=candidate_ids,
                valid_candidate_ids=valid_ids,
                rejected_candidates=rejected_pairs,
                selected_sample_idx=rec.selected_sample_idx if rec.is_accepted else -1,
                selected_candidate_id=rec.selected_candidate_id,
                selected_endpoint=selected_ep,
                is_accepted=rec.is_accepted,
                guidance_applied=rec.guidance_applied,
                fallback_to_original=rec.fallback_to_original,
                decision_reason=rec.selection_reason,
                rejection_reason=rec.rejection_reason,
                proposed_exit_id=rec.target_exit_id,
                proposed_waypoints=proposed_wps_xy,
                guide_waypoint=gw_xy,
            )
            self.decision_traces_by_agent[aid] = trace_rec

            pre_pos = (float(ped.position.x), float(ped.position.y))

            # Safe movement integration: Update target_exit_id and waypoints via engine public interface
            needs_emergency_reroute = (
                (status_val == "blocked" or getattr(ped, "target_exit_id", None) not in open_exits)
                and bool(rec.proposed_waypoints)
                and rec.target_exit_id in open_exits
            )
            if (rec.is_accepted and rec.proposed_waypoints) or needs_emergency_reroute:
                ped.target_exit_id = rec.target_exit_id
                ped.waypoints = list(rec.proposed_waypoints)
                if status_val == "blocked":
                    ped.status = PAVAN["PedestrianStatus"].ACTIVE

                verified_wps = tuple(
                    (float(wp.x), float(wp.y))
                    for wp in (getattr(ped, "waypoints", []) or [])
                    if self._is_finite_xy(getattr(wp, "x", None), getattr(wp, "y", None))
                )
                verified_exit = getattr(ped, "target_exit_id", None)
                exact_match = (
                    len(verified_wps) == len(proposed_wps_xy)
                    and all(
                        abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6
                        for a, b in zip(verified_wps, proposed_wps_xy)
                    )
                    and verified_exit == rec.target_exit_id
                )
                adoption_status = (
                    ("ADOPTED" if rec.is_accepted else "ADOPTED_REROUTE_FALLBACK")
                    if exact_match
                    else ("MODIFIED_BY_ENGINE" if verified_wps else "NOT_ADOPTED_BY_ENGINE")
                )
                self.adopted_route_records_by_agent[aid] = AdoptedRouteRecord(
                    agent_id=aid,
                    decision_step_index=step,
                    decision_timestamp=float(timestamp),
                    target_exit_id=rec.target_exit_id,
                    proposed_waypoints=proposed_wps_xy,
                    waypoints_after_adoption=verified_wps,
                    exit_after_adoption=verified_exit,
                    adoption_attempted=True,
                    adoption_succeeded=bool(verified_wps),
                    adoption_status=adoption_status,
                    reason=(
                        (
                            "Engine accepted proposed waypoints"
                            if rec.is_accepted
                            else "Emergency open-exit reroute adopted after MID fallback"
                        )
                        if exact_match
                        else (
                            "Engine waypoints differ from proposed route"
                            if verified_wps
                            else "Engine cleared waypoints after assignment"
                        )
                    ),
                )
                self._pending_causal_steps[aid] = {
                    "decision_step": step,
                    "decision_timestamp": float(timestamp),
                    "agent_id": aid,
                    "mid_candidate_count": len(mid_res.samples),
                    "selected_candidate_id": trace_rec.selected_candidate_id,
                    "selected_sample_idx": rec.selected_sample_idx,
                    "selected_endpoint": selected_ep,
                    "guide_waypoint": gw_xy,
                    "proposed_waypoints": proposed_wps_xy,
                    "adoption_status": adoption_status,
                    "adoption_succeeded": bool(verified_wps),
                    "adopted_waypoints": verified_wps,
                    "pre_step_position": pre_pos,
                }
            else:
                verified_wps = tuple(
                    (float(wp.x), float(wp.y))
                    for wp in (getattr(ped, "waypoints", []) or [])
                    if self._is_finite_xy(getattr(wp, "x", None), getattr(wp, "y", None))
                )
                if not (rec.rejection_reason == "cadence_guard" and aid in self.adopted_route_records_by_agent and self.adopted_route_records_by_agent[aid].adoption_succeeded):
                    self.adopted_route_records_by_agent[aid] = AdoptedRouteRecord(
                        agent_id=aid,
                        decision_step_index=step,
                        decision_timestamp=float(timestamp),
                        target_exit_id=rec.target_exit_id,
                        proposed_waypoints=proposed_wps_xy,
                        waypoints_after_adoption=verified_wps,
                        exit_after_adoption=getattr(ped, "target_exit_id", None),
                        adoption_attempted=False,
                        adoption_succeeded=False,
                        adoption_status="NOT_ADOPTED_REJECTED",
                        reason=rec.rejection_reason or rec.selection_reason,
                    )

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
        # ramping gently to ~45 (~18% opacity) for a soft, subtle background ambient glow
        alpha_channel = (density_norm * 45).astype(np.uint8)
        rgba[:, :, 3] = np.where(density_norm > 0.01, alpha_channel, 0)

        # Display-only: hide the density glow where MID candidates are drawn so the
        # candidate fan stays readable. The density grid itself is not modified.
        clear_mask = self._heatmap_candidate_clear_mask()
        if clear_mask is not None:
            rgba[:, :, 3] = np.where(clear_mask, 0, rgba[:, :, 3])

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

        # Scale to building canvas screen bounds and blit onto the floor
        bx, by, bw, bh = self.viewport.building_screen_bounds
        scaled_heatmap = pygame.transform.smoothscale(temp_surf, (int(bw), int(bh)))
        surface.blit(scaled_heatmap, (int(bx), int(by)))

        # Ensure static building obstacles, walls, and exits stay strictly in front of the heatmap
        self.renderer._draw_obstacles(surface, self.building, self.viewport)
        self.renderer._draw_walls(surface, self.building, self.viewport)
        self.renderer._draw_exits(surface, self.building, self.viewport)

        # Draw compact physical density legend in lower-left of the building area
        self._render_heatmap_legend(surface, max_density)

    HEATMAP_CANDIDATE_CLEAR_RADIUS_M = 3.0

    def _heatmap_candidate_clear_mask(self) -> Optional[np.ndarray]:
        """Boolean (rows, cols) mask of heatmap cells around the selected agent whose
        MID candidates are being drawn; None when no candidates are rendered."""
        target_id = self.selected_agent_id
        if not target_id or self.engine is None:
            return None
        candidates_visible = self.show_mid_predictions or self.operating_mode in (
            OperatingMode.MID_ASSISTED_EVACUATION,
            OperatingMode.MID_DRIVEN,
        )
        if not candidates_visible:
            return None
        ped = self.engine.registry.get(target_id)
        if ped is None:
            return None
        px = getattr(ped.position, "x", None)
        py = getattr(ped.position, "y", None)
        if px is None or py is None or not self._is_finite_xy(px, py):
            return None
        gs = self.grid_spec
        xs = gs.min_x + (np.arange(gs.cols) + 0.5) * gs.cell_size_x
        ys = gs.min_y + (np.arange(gs.rows) + 0.5) * gs.cell_size_y
        dx = xs[np.newaxis, :] - float(px)
        dy = ys[:, np.newaxis] - float(py)
        r = self.HEATMAP_CANDIDATE_CLEAR_RADIUS_M
        return (dx * dx + dy * dy) <= r * r

    def _render_heatmap_legend(self, surface: pygame.Surface, max_density: float) -> None:
        """Render compact density colorbar with physical units (peds/m²)."""
        self._ensure_fonts()
        bx, by, _bw, bh = self.viewport.building_screen_bounds
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

    def _safe_screen_point(self, world_xy: Tuple[Any, Any]) -> Optional[Tuple[int, int]]:
        """Safely map finite world coordinates to finite screen pixel coordinates."""
        if not isinstance(world_xy, (tuple, list)) or len(world_xy) < 2:
            return None
        wx, wy = world_xy[0], world_xy[1]
        if not self._is_finite_xy(wx, wy):
            return None
        sx, sy = self.viewport.world_to_screen_point((float(wx), float(wy)))
        if not (math.isfinite(sx) and math.isfinite(sy)):
            return None
        return (int(round(sx)), int(round(sy)))

    @staticmethod
    def _draw_dashed_polyline(
        surface: pygame.Surface,
        pts: Sequence[Tuple[int, int]],
        color: Tuple[int, int, int],
        width: int = 2,
        dash_len: float = 9.0,
        gap_len: float = 6.0,
        casing_color: Optional[Tuple[int, int, int]] = None,
        casing_width: int = 0,
    ) -> None:
        """Draw a uniform pixel-length dashed polyline with optional high-contrast casing."""
        if len(pts) < 2:
            return
        period = dash_len + gap_len
        if period <= 0.0:
            return
        acc = 0.0
        for idx in range(len(pts) - 1):
            x1, y1 = pts[idx]
            x2, y2 = pts[idx + 1]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            seg_len = math.hypot(dx, dy)
            if seg_len < 1e-6:
                continue
            ux, uy = dx / seg_len, dy / seg_len
            pos = 0.0
            while pos < seg_len:
                mod = acc % period
                if mod < dash_len:
                    draw_rem = min(dash_len - mod, seg_len - pos)
                    p_a = (int(round(x1 + ux * pos)), int(round(y1 + uy * pos)))
                    p_b = (int(round(x1 + ux * (pos + draw_rem))), int(round(y1 + uy * (pos + draw_rem))))
                    if casing_color is not None and casing_width > width:
                        pygame.draw.line(surface, casing_color, p_a, p_b, width=casing_width)
                    pygame.draw.line(surface, color, p_a, p_b, width=width)
                    pos += draw_rem
                    acc += draw_rem
                else:
                    skip_rem = min(period - mod, seg_len - pos)
                    pos += skip_rem
                    acc += skip_rem

    @staticmethod
    def _draw_dotted_trail(
        surface: pygame.Surface,
        pts: Sequence[Tuple[int, int]],
        dot_color: Tuple[int, int, int],
        radius: int = 2,
        spacing: float = 8.0,
        outline_color: Tuple[int, int, int] = (120, 53, 15),
    ) -> None:
        """Draw an amber dotted breadcrumb trail preserving distinct observation markers."""
        if not pts:
            return
        if len(pts) >= 2 and spacing > 0.0:
            next_dist = 0.0
            for idx in range(len(pts) - 1):
                x1, y1 = pts[idx]
                x2, y2 = pts[idx + 1]
                dx = float(x2 - x1)
                dy = float(y2 - y1)
                seg_len = math.hypot(dx, dy)
                if seg_len < 1e-6:
                    continue
                ux, uy = dx / seg_len, dy / seg_len
                d = next_dist
                while d <= seg_len:
                    px = int(round(x1 + ux * d))
                    py = int(round(y1 + uy * d))
                    pygame.draw.circle(surface, outline_color, (px, py), max(2, radius + 1))
                    pygame.draw.circle(surface, dot_color, (px, py), max(1, radius))
                    d += spacing
                next_dist = d - seg_len
        for pt in pts:
            pygame.draw.circle(surface, outline_color, pt, max(3, radius + 2))
            pygame.draw.circle(surface, dot_color, pt, max(2, radius + 1))

    def _draw_label_pill(
        self,
        surface: pygame.Surface,
        text: str,
        pos: Tuple[int, int],
        text_color: Tuple[int, int, int],
        bg_color: Tuple[int, int, int] = (15, 23, 42),
        border_color: Optional[Tuple[int, int, int]] = None,
        anchor: Optional[Tuple[int, int]] = None,
        avoid_overlap: bool = True,
    ) -> None:
        """Render a readable text badge clamped inside viewport with overlap avoidance."""
        self._ensure_fonts()
        txt_surf = self.font_small.render(text, True, text_color)
        pw = txt_surf.get_width() + 8
        ph = txt_surf.get_height() + 4
        min_x = 8
        min_y = 26
        max_x = max(min_x, self.width - self.SIDE_PANEL_WIDTH - pw - 6)
        max_y = max(min_y, self.height - 42 - ph)
        base_rx = max(min_x, min(int(pos[0]), max_x))
        base_ry = max(min_y, min(int(pos[1]), max_y))
        rect = pygame.Rect(base_rx, base_ry, pw, ph)

        if avoid_overlap and self._placed_label_rects:
            def _overlaps(cand: pygame.Rect) -> bool:
                padded = cand.inflate(4, 3)
                return any(padded.colliderect(existing) for existing in self._placed_label_rects)

            if _overlaps(rect):
                found: Optional[pygame.Rect] = None
                for radius in (16, 28, 42, 58, 76, 96, 118):
                    for step_angle in range(12):
                        ang = (step_angle * math.pi) / 6.0
                        cx = max(min_x, min(int(round(base_rx + math.cos(ang) * radius)), max_x))
                        cy = max(min_y, min(int(round(base_ry + math.sin(ang) * radius * 0.78)), max_y))
                        cand_rect = pygame.Rect(cx, cy, pw, ph)
                        if not _overlaps(cand_rect):
                            found = cand_rect
                            break
                    if found is not None:
                        break
                if found is not None:
                    rect = found

        if avoid_overlap:
            self._placed_label_rects.append(rect)

        if anchor is not None:
            near_x = max(rect.left, min(int(anchor[0]), rect.right))
            near_y = max(rect.top, min(int(anchor[1]), rect.bottom))
            if math.hypot(near_x - anchor[0], near_y - anchor[1]) > 7.0:
                leader_col = border_color if border_color is not None else text_color
                pygame.draw.line(surface, (15, 23, 42), anchor, (near_x, near_y), width=3)
                pygame.draw.line(surface, leader_col, anchor, (near_x, near_y), width=1)

        pygame.draw.rect(surface, bg_color, rect, border_radius=3)
        if border_color is not None:
            pygame.draw.rect(surface, border_color, rect, width=1, border_radius=3)
        surface.blit(txt_surf, (rect.x + 4, rect.y + 2))

    def _prepare_clean_observation_history(
        self,
        snapshot: h_contract.TrajectorySnapshot,
        target_id: str,
    ) -> List[h_contract.TrajectoryObservation]:
        """Return chronologically sorted, deduplicated finite observations from adapter snapshot."""
        raw_history = snapshot.trajectories.get(target_id, ())
        seen_keys = set()
        clean_history: List[h_contract.TrajectoryObservation] = []
        for obs in sorted(raw_history, key=lambda o: (float(o.timestamp), int(o.step_index))):
            if not self._is_finite_xy(obs.x, obs.y):
                continue
            key = (int(obs.step_index), round(float(obs.timestamp), 9))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            clean_history.append(obs)
        return clean_history

    def _render_observed_history_layer(
        self,
        surface: pygame.Surface,
        clean_history: Sequence[h_contract.TrajectoryObservation],
    ) -> None:
        """Layer 2: Render adapter-recorded historical positions as amber dotted breadcrumbs."""
        pts: List[Tuple[int, int]] = []
        for obs in clean_history:
            sp = self._safe_screen_point((obs.x, obs.y))
            if sp is not None:
                pts.append(sp)

        if not self.show_observed_history:
            self.latest_layer_states["observed_history"] = {
                "rendered": False,
                "reason": "toggled_off",
                "count": len(clean_history),
                "timestamps": [float(o.timestamp) for o in clean_history],
            }
            return

        if pts:
            self._draw_dotted_trail(
                surface,
                pts,
                dot_color=(245, 158, 11),
                radius=2,
                spacing=8.0,
                outline_color=(120, 53, 15),
            )
            if self.show_layer_labels and len(pts) >= 2:
                self._draw_label_pill(
                    surface,
                    f"OBS({len(pts)})",
                    (pts[0][0] + 8, pts[0][1] - 18),
                    (251, 191, 36),
                    border_color=(245, 158, 11),
                    anchor=pts[0],
                )

        self.latest_layer_states["observed_history"] = {
            "rendered": bool(pts),
            "reason": "ok" if pts else "empty_history",
            "count": len(clean_history),
            "timestamps": [float(o.timestamp) for o in clean_history],
        }

    def _render_cv_prediction_layer(
        self,
        surface: pygame.Surface,
        obs_anchor_snap: Optional[h_contract.TrajectorySnapshot],
        target_id: str,
        clean_history: Sequence[h_contract.TrajectoryObservation],
    ) -> Optional[h_contract.TrajectoryPrediction]:
        """Layer 3: Render baseline Constant Velocity prediction (white dashed line + CV label)."""
        if not self.show_baseline_predictions:
            self.latest_layer_states["cv_prediction"] = {
                "rendered": False,
                "reason": "toggled_off",
                "count": 0,
            }
            return None

        if len(clean_history) < 2 or obs_anchor_snap is None:
            self.cv_status = PredictionStatus.WAITING_FOR_HISTORY
            self.latest_layer_states["cv_prediction"] = {
                "rendered": False,
                "reason": "insufficient_history",
                "count": 0,
            }
            return None

        latest_obs = clean_history[-1]
        cv_pred = self._get_or_compute_cv_prediction(
            obs_anchor_snap, target_id, int(latest_obs.step_index)
        )
        if not cv_pred or not cv_pred.predictions:
            self.latest_layer_states["cv_prediction"] = {
                "rendered": False,
                "reason": "empty_prediction",
                "count": 0,
            }
            return cv_pred

        anchor_pt = self._safe_screen_point((latest_obs.x, latest_obs.y))
        pts: List[Tuple[int, int]] = [anchor_pt] if anchor_pt is not None else []
        for p in cv_pred.predictions:
            sp = self._safe_screen_point((p.x, p.y))
            if sp is not None:
                pts.append(sp)

        if len(pts) >= 2:
            self._draw_dashed_polyline(
                surface,
                pts,
                color=(255, 255, 255),
                width=3,
                dash_len=9.0,
                gap_len=6.0,
                casing_color=(15, 23, 42),
                casing_width=5,
            )
            end_pt = pts[-1]
            pygame.draw.circle(surface, (15, 23, 42), end_pt, 6)
            pygame.draw.circle(surface, (255, 255, 255), end_pt, 5)
            pygame.draw.circle(surface, (15, 23, 42), end_pt, 2)
            if self.show_layer_labels:
                self._draw_label_pill(
                    surface,
                    "CV",
                    (end_pt[0] - 42, end_pt[1] - 10),
                    (255, 255, 255),
                    border_color=(148, 163, 184),
                    anchor=end_pt,
                )

        self.latest_layer_states["cv_prediction"] = {
            "rendered": len(pts) >= 2,
            "reason": "ok" if len(pts) >= 2 else "no_finite_points",
            "count": len(cv_pred.predictions),
            "source_step": cv_pred.source_step_index,
        }
        return cv_pred

    def _render_mid_candidates_layer(
        self,
        surface: pygame.Surface,
        obs_anchor_snap: Optional[h_contract.TrajectorySnapshot],
        snapshot: h_contract.TrajectorySnapshot,
        target_id: str,
        clean_history: Sequence[h_contract.TrajectoryObservation],
    ) -> Tuple[
        Optional[h_mid.MultiTrajectoryPrediction],
        Optional[h_controller.MIDDecisionRecommendation],
        bool,
        int,
        Optional[Tuple[int, int]],
    ]:
        """Layer 4: Render all K MID candidates preserving exact sample IDs and rejection states."""
        should_query_mid = (
            self.show_mid_predictions
            or self.operating_mode in (OperatingMode.MID_ASSISTED_EVACUATION, OperatingMode.MID_DRIVEN)
        )
        if self.operating_mode == OperatingMode.MID_ASSISTED_EVACUATION:
            rec_for_target = self.controller.latest_recommendations_by_agent.get(target_id)
        elif self.operating_mode == OperatingMode.MID_DRIVEN:
            rec_for_target = self.driven_controller.latest_decisions.get(target_id)
        else:
            rec_for_target = None

        if not should_query_mid:
            self.latest_layer_states["mid_candidates"] = {
                "rendered": False,
                "reason": "toggled_off",
                "count": 0,
                "candidate_ids": (),
                "rejected": (),
            }
            return None, rec_for_target, False, -1, None

        if len(clean_history) < 2 or obs_anchor_snap is None:
            self.mid_status = PredictionStatus.WAITING_FOR_HISTORY
            self.latest_layer_states["mid_candidates"] = {
                "rendered": False,
                "reason": "insufficient_history",
                "count": 0,
                "candidate_ids": (),
                "rejected": (),
            }
            return None, rec_for_target, False, -1, None

        latest_obs = clean_history[-1]
        mid_result: Optional[h_mid.MultiTrajectoryPrediction] = None

        # Prefer exact cached batch already evaluated by the controller so Layer 4 & Layer 5 stay aligned
        loaded_from_rec_cache = False
        if rec_for_target is not None and rec_for_target.candidates:
            rec_key = (target_id, int(rec_for_target.step_index), self.k_samples, self.prediction_horizon)
            if rec_key in self._mid_cache_map:
                mid_result = self._mid_cache_map[rec_key]
                loaded_from_rec_cache = True

        if mid_result is None:
            mid_step = int(latest_obs.step_index)
            mid_result = self._get_or_compute_mid_prediction(obs_anchor_snap, target_id, mid_step)

        anchor_pt = self._safe_screen_point((latest_obs.x, latest_obs.y))
        if not mid_result or not mid_result.samples:
            self.latest_layer_states["mid_candidates"] = {
                "rendered": False,
                "reason": "empty_candidates",
                "count": 0,
                "candidate_ids": (),
                "rejected": (),
            }
            return mid_result, rec_for_target, False, -1, anchor_pt

        # Anchor origin uses the snapshot observation recorded at mid_result.source_step_index when available
        src_obs = next(
            (o for o in reversed(clean_history) if int(o.step_index) == int(mid_result.source_step_index)),
            latest_obs,
        )
        anchor_pt = self._safe_screen_point((src_obs.x, src_obs.y))

        candidates_aligned = (
            rec_for_target is not None
            and len(rec_for_target.candidates) == len(mid_result.samples)
            and (
                int(rec_for_target.step_index) == int(mid_result.source_step_index)
                or (
                    self.operating_mode == OperatingMode.MID_DRIVEN
                    and loaded_from_rec_cache
                    and int(rec_for_target.step_index) >= int(mid_result.source_step_index)
                )
            )
        )
        cand_map = {c.sample_idx: c for c in rec_for_target.candidates} if (candidates_aligned and rec_for_target is not None) else {}
        selected_idx = (
            int(rec_for_target.selected_sample_idx)
            if (
                candidates_aligned
                and rec_for_target is not None
                and rec_for_target.is_accepted
                and 0 <= rec_for_target.selected_sample_idx < len(mid_result.samples)
            )
            else -1
        )

        candidate_ids: List[str] = []
        rejected_info: List[Tuple[str, str]] = []
        rendered_any = False

        for idx, sample in enumerate(mid_result.samples):
            cid = f"C{idx}"
            candidate_ids.append(cid)
            cand_eval = cand_map.get(idx)
            if cand_eval is not None and not cand_eval.is_valid:
                rejected_info.append((cid, cand_eval.rejection_reason or "rejected"))

            if not self.show_mid_predictions:
                continue
            if idx == selected_idx:
                # Layer 5 draws selected MID candidate prominently on top
                continue

            pts: List[Tuple[int, int]] = [anchor_pt] if anchor_pt is not None else []
            for p in sample.predictions:
                sp = self._safe_screen_point((p.x, p.y))
                if sp is not None:
                    pts.append(sp)
            if len(pts) < 2:
                continue

            rendered_any = True
            cand_angle = ((idx * 2.0 * math.pi) / max(1, len(mid_result.samples))) - (math.pi / 4.0)
            label_offset_x = int(round(math.cos(cand_angle) * 26.0))
            label_offset_y = int(round(math.sin(cand_angle) * 20.0))

            if cand_eval is not None and not cand_eval.is_valid:
                rej_color = (239, 68, 68)
                self._draw_dashed_polyline(
                    surface,
                    pts,
                    color=rej_color,
                    width=2,
                    dash_len=6.0,
                    gap_len=5.0,
                    casing_color=(15, 23, 42),
                    casing_width=3,
                )
                end_x, end_y = pts[-1]
                pygame.draw.line(surface, (15, 23, 42), (end_x - 4, end_y - 4), (end_x + 4, end_y + 4), width=4)
                pygame.draw.line(surface, (15, 23, 42), (end_x - 4, end_y + 4), (end_x + 4, end_y - 4), width=4)
                pygame.draw.line(surface, rej_color, (end_x - 4, end_y - 4), (end_x + 4, end_y + 4), width=2)
                pygame.draw.line(surface, rej_color, (end_x - 4, end_y + 4), (end_x + 4, end_y - 4), width=2)
                if self.show_layer_labels:
                    self._draw_label_pill(
                        surface,
                        f"{cid}[X]",
                        (end_x + label_offset_x, end_y + label_offset_y),
                        (254, 202, 202),
                        border_color=rej_color,
                        anchor=(end_x, end_y),
                    )
            else:
                color = self.mid_colors[idx % len(self.mid_colors)]
                self._draw_dashed_polyline(
                    surface,
                    pts,
                    color=color,
                    width=2,
                    dash_len=7.0,
                    gap_len=4.0,
                    casing_color=(15, 23, 42),
                    casing_width=3,
                )
                end_pt = pts[-1]
                pygame.draw.circle(surface, (15, 23, 42), end_pt, 4)
                pygame.draw.circle(surface, color, end_pt, 3)
                if self.show_layer_labels:
                    self._draw_label_pill(
                        surface,
                        cid,
                        (end_pt[0] + label_offset_x, end_pt[1] + label_offset_y),
                        color,
                        border_color=color,
                        anchor=end_pt,
                    )

        self.latest_layer_states["mid_candidates"] = {
            "rendered": rendered_any,
            "reason": "ok" if rendered_any else ("hidden_by_toggle" if not self.show_mid_predictions else "no_finite_points"),
            "count": len(mid_result.samples),
            "requested_k": self.k_samples,
            "candidate_ids": tuple(candidate_ids),
            "rejected": tuple(rejected_info),
            "source_step": int(mid_result.source_step_index),
        }
        return mid_result, rec_for_target, candidates_aligned, selected_idx, anchor_pt

    def _render_selected_mid_candidate_layer(
        self,
        surface: pygame.Surface,
        mid_result: Optional[h_mid.MultiTrajectoryPrediction],
        rec_for_target: Optional[h_controller.MIDDecisionRecommendation],
        candidates_aligned: bool,
        selected_idx: int,
        anchor_pt: Optional[Tuple[int, int]],
    ) -> None:
        """Layer 5: Highlight controller-selected MID candidate or explicitly show no selection."""
        bx, by, bw, bh = self.viewport.building_screen_bounds

        if (
            mid_result is not None
            and candidates_aligned
            and rec_for_target is not None
            and 0 <= selected_idx < len(mid_result.samples)
        ):
            sel_sample = mid_result.samples[selected_idx]
            pts: List[Tuple[int, int]] = [anchor_pt] if anchor_pt is not None else []
            for p in sel_sample.predictions:
                sp = self._safe_screen_point((p.x, p.y))
                if sp is not None:
                    pts.append(sp)

            if len(pts) >= 2:
                pygame.draw.lines(surface, (120, 53, 15), False, pts, width=8)
                pygame.draw.lines(surface, (250, 204, 21), False, pts, width=4)
                end_pt = pts[-1]
                pygame.draw.circle(surface, (120, 53, 15), end_pt, 8)
                pygame.draw.circle(surface, (250, 204, 21), end_pt, 6)
                pygame.draw.circle(surface, (15, 23, 42), end_pt, 3)
                if self.show_layer_labels:
                    self._draw_label_pill(
                        surface,
                        f"SEL C{selected_idx}",
                        (end_pt[0] + 18, end_pt[1] - 18),
                        (254, 240, 138),
                        bg_color=(66, 32, 6),
                        border_color=(250, 204, 21),
                        anchor=end_pt,
                    )
                self.latest_layer_states["selected_mid_candidate"] = {
                    "rendered": True,
                    "status": "SELECTED",
                    "selected_candidate_id": f"C{selected_idx}",
                    "sample_idx": selected_idx,
                    "reason": rec_for_target.selection_reason,
                }
                return

        if self.operating_mode not in (OperatingMode.MID_ASSISTED_EVACUATION, OperatingMode.MID_DRIVEN):
            empty_reason = f"Controller inactive ({self.operating_mode.value})"
        elif rec_for_target is None:
            empty_reason = "No controller decision yet"
        elif rec_for_target.rejection_reason:
            empty_reason = f"Rejected ({rec_for_target.rejection_reason})"
        elif not candidates_aligned:
            empty_reason = "Waiting for aligned K evaluation"
        else:
            empty_reason = "No feasible candidate selected"

        if self.show_layer_labels and (
            self.show_mid_predictions
            or self.operating_mode in (OperatingMode.MID_ASSISTED_EVACUATION, OperatingMode.MID_DRIVEN)
        ):
            pill_pos = (
                (anchor_pt[0] + 22, anchor_pt[1] + 24)
                if anchor_pt is not None
                else (int(bx + max(180, bw - 310)), int(by + max(8, bh - 28)))
            )
            self._draw_label_pill(
                surface,
                "No MID candidate selected",
                pill_pos,
                (251, 191, 36),
                border_color=(148, 163, 184),
                anchor=anchor_pt,
            )

        self.latest_layer_states["selected_mid_candidate"] = {
            "rendered": False,
            "status": "NO_CANDIDATE_SELECTED",
            "selected_candidate_id": None,
            "sample_idx": -1,
            "reason": empty_reason,
        }

    def _render_adopted_route_layer(
        self,
        surface: pygame.Surface,
        selected_ped: Any,
        target_id: str,
    ) -> None:
        """Layer 6: Render actual route adopted on SimulationEngine pedestrian (green + wp markers)."""
        adopted_rec = self.adopted_route_records_by_agent.get(target_id)
        if selected_ped is None:
            self.latest_layer_states["adopted_route"] = {
                "rendered": False,
                "status": "UNAVAILABLE",
                "reason": "Selected pedestrian not in registry",
                "waypoints": (),
            }
            return

        live_wps = tuple(
            (float(wp.x), float(wp.y))
            for wp in (getattr(selected_ped, "waypoints", []) or [])
            if self._is_finite_xy(getattr(wp, "x", None), getattr(wp, "y", None))
        )
        if (
            self.operating_mode not in (OperatingMode.MID_ASSISTED_EVACUATION, OperatingMode.MID_DRIVEN)
            or adopted_rec is None
            or not adopted_rec.adoption_succeeded
        ):
            if self.operating_mode not in (OperatingMode.MID_ASSISTED_EVACUATION, OperatingMode.MID_DRIVEN):
                reason = f"No adopted MID route ({self.operating_mode.value})"
            elif adopted_rec is None:
                reason = "No MID route adopted yet"
            else:
                reason = f"Not adopted ({adopted_rec.reason})"
            self.latest_layer_states["adopted_route"] = {
                "rendered": False,
                "status": "NOT_ADOPTED",
                "reason": reason,
                "waypoints": (),
                "engine_baseline_waypoints_count": len(live_wps),
            }
            return

        if not live_wps:
            ped_status = str(getattr(selected_ped.status, "value", selected_ped.status)).upper()
            self.latest_layer_states["adopted_route"] = {
                "rendered": False,
                "status": "ENGINE_CLEARED",
                "reason": f"Engine cleared waypoints ({ped_status})",
                "waypoints": (),
            }
            return

        if live_wps == adopted_rec.waypoints_after_adoption:
            live_status = "ADOPTED"
        elif (
            len(live_wps) < len(adopted_rec.waypoints_after_adoption)
            and adopted_rec.waypoints_after_adoption[-len(live_wps):] == live_wps
        ):
            live_status = f"ADOPTED (progress {len(live_wps)}/{len(adopted_rec.waypoints_after_adoption)})"
        else:
            live_status = "MODIFIED_BY_ENGINE"

        if not self.show_adopted_route:
            self.latest_layer_states["adopted_route"] = {
                "rendered": False,
                "status": live_status,
                "reason": "toggled_off",
                "waypoints": live_wps,
            }
            return

        pos_pt = self._safe_screen_point((selected_ped.position.x, selected_ped.position.y))
        pts: List[Tuple[int, int]] = [pos_pt] if pos_pt is not None else []
        wp_pts: List[Tuple[int, int]] = []
        for wp in live_wps:
            sp = self._safe_screen_point(wp)
            if sp is not None:
                pts.append(sp)
                wp_pts.append(sp)

        if len(pts) >= 2:
            pygame.draw.lines(surface, (6, 78, 59), False, pts, width=6)
            pygame.draw.lines(surface, (22, 163, 74), False, pts, width=3)
        for wp_pt in wp_pts:
            sq = pygame.Rect(wp_pt[0] - 4, wp_pt[1] - 4, 8, 8)
            pygame.draw.rect(surface, (34, 197, 94), sq)
            pygame.draw.rect(surface, (255, 255, 255), sq, width=1)

        if self.show_layer_labels and wp_pts:
            anchor_lbl = wp_pts[min(1, len(wp_pts) - 1)]
            self._draw_label_pill(
                surface,
                f"ADOPTED ({len(wp_pts)} wp)",
                (anchor_lbl[0] + 12, anchor_lbl[1] + 6),
                (187, 247, 208),
                bg_color=(6, 78, 59),
                border_color=(34, 197, 94),
                anchor=anchor_lbl,
            )

        self.latest_layer_states["adopted_route"] = {
            "rendered": bool(wp_pts),
            "status": live_status,
            "reason": adopted_rec.reason,
            "waypoints": live_wps,
            "target_exit_id": getattr(selected_ped, "target_exit_id", None),
        }

    def _draw_pedestrians(self, surface: pygame.Surface, registry: Any, viewport: Any) -> None:
        """Render pedestrians cleanly without clunky overlapping ID badges in minimal presentation mode."""
        for ped in registry:
            pos = getattr(ped, "position", None)
            if pos is None:
                continue
            sx, sy = viewport.world_to_screen_point(pos)
            r_px = max(4.0, viewport.world_to_screen_dim(float(getattr(ped, "radius", 0.22))))
            ix, iy = int(round(sx)), int(round(sy))
            ir = max(3, int(round(r_px)))

            status_str = str(getattr(getattr(ped, "status", ""), "value", getattr(ped, "status", ""))).upper()
            if "ACTIVE" in status_str:
                pygame.draw.circle(surface, (56, 189, 248), (ix, iy), ir)
                pygame.draw.circle(surface, (14, 116, 144), (ix, iy), ir, width=1)
                pygame.draw.circle(surface, (255, 255, 255), (ix, iy), max(1, int(ir * 0.35)))
            elif "EVAC" in status_str:
                pygame.draw.circle(surface, (34, 197, 94), (ix, iy), ir, width=2)
                pygame.draw.circle(surface, (34, 197, 94), (ix, iy), max(1, int(ir * 0.4)))
            elif "BLOCK" in status_str:
                pygame.draw.circle(surface, (239, 68, 68), (ix, iy), ir)
                pygame.draw.circle(surface, (153, 27, 27), (ix, iy), ir, width=2)
            else:
                pygame.draw.circle(surface, (148, 163, 184), (ix, iy), ir)

            # In Developer Overlay mode, render ID badges for debugging
            if self.show_developer_overlay and hasattr(ped, "id"):
                font_badge = getattr(self.renderer, "font_badge", None) or self.font_small
                badge_surf = font_badge.render(str(ped.id), True, (241, 245, 249))
                bw = badge_surf.get_width() + 6
                bh = badge_surf.get_height() + 2
                bx = int(ix - bw / 2.0)
                by = int(iy - ir - bh - 2)
                badge_rect = pygame.Rect(bx, by, bw, bh)
                pygame.draw.rect(surface, (15, 23, 42), badge_rect, border_radius=3)
                pygame.draw.rect(surface, (51, 65, 85), badge_rect, width=1, border_radius=3)
                surface.blit(badge_surf, (bx + 3, by + 1))

    def _render_actual_movement_layer(
        self,
        surface: pygame.Surface,
        selected_ped: Any,
        target_id: str,
    ) -> None:
        """Layer 1: Render actual traversed engine path (solid line) and top pedestrian marker."""
        raw_path = self.actual_traversed_paths.get(target_id, [])
        screen_pts: List[Tuple[int, int]] = []
        for pt in raw_path:
            sp = self._safe_screen_point((pt[2], pt[3]))
            if sp is not None:
                screen_pts.append(sp)

        path_drawn = False
        if self.show_actual_path and len(screen_pts) >= 2:
            pygame.draw.lines(surface, (15, 23, 42), False, screen_pts, width=5)
            pygame.draw.lines(surface, (37, 99, 235), False, screen_pts, width=3)
            pygame.draw.circle(surface, (37, 99, 235), screen_pts[0], 3)
            path_drawn = True

        marker_drawn = False
        ped_status_str = "MISSING"
        if selected_ped is not None and self._is_finite_xy(selected_ped.position.x, selected_ped.position.y):
            ped_status_str = str(getattr(selected_ped.status, "value", selected_ped.status)).upper()
            sp = self._safe_screen_point((selected_ped.position.x, selected_ped.position.y))
            if sp is not None:
                r_px = max(4, int(round(self.viewport.world_to_screen_dim(float(selected_ped.radius)))))
                if ped_status_str == "EVACUATED":
                    col = (34, 197, 94)
                elif ped_status_str == "BLOCKED":
                    col = (239, 68, 68)
                else:
                    col = (56, 189, 248)

                pygame.draw.circle(surface, col, sp, r_px)
                pygame.draw.circle(surface, (15, 23, 42), sp, r_px, width=1)
                pygame.draw.circle(surface, (255, 255, 255), sp, max(1, int(r_px * 0.35)))
                sel_r = r_px + 3
                pygame.draw.circle(surface, (250, 204, 21), sp, sel_r, width=2)
                marker_drawn = True

                if self.show_layer_labels:
                    self._draw_label_pill(
                        surface,
                        f"{selected_ped.id} [{ped_status_str}]",
                        (sp[0] + sel_r + 14, sp[1] - sel_r - 10),
                        (254, 240, 138),
                        border_color=(250, 204, 21),
                        anchor=sp,
                    )

        self.latest_layer_states["actual_movement"] = {
            "rendered": path_drawn or marker_drawn,
            "path_points": len(screen_pts),
            "path_drawn": path_drawn,
            "marker_drawn": marker_drawn,
            "pedestrian_id": target_id,
            "status": ped_status_str,
        }

    def _render_baseline_ghost_overlay(
        self,
        surface: pygame.Surface,
        selected_ped: Any,
        target_id: str,
    ) -> None:
        """Render parallel unmodified ORIGINAL baseline ghost trail and marker for direct comparison."""
        if (
            not self.show_baseline_ghost
            or self.operating_mode not in (OperatingMode.MID_ASSISTED_EVACUATION, OperatingMode.MID_DRIVEN)
            or self._baseline_ref_engine is None
        ):
            self.latest_layer_states["baseline_ghost"] = {"rendered": False}
            return

        ref_ped = self._baseline_ref_engine.registry.get(target_id)
        raw_ref = self._baseline_ref_paths.get(target_id, [])
        pts: List[Tuple[int, int]] = []
        for entry in raw_ref[-35:]:
            sp = self._safe_screen_point((entry[2], entry[3]))
            if sp is not None:
                pts.append(sp)

        drawn = False
        if len(pts) >= 2:
            self._draw_dashed_polyline(
                surface,
                pts,
                color=(148, 163, 184),
                width=2,
                dash_len=5.0,
                gap_len=5.0,
                casing_color=(15, 23, 42),
                casing_width=3,
            )
            drawn = True

        ref_sp: Optional[Tuple[int, int]] = None
        if ref_ped is not None and self._is_finite_xy(ref_ped.position.x, ref_ped.position.y):
            ref_sp = self._safe_screen_point((ref_ped.position.x, ref_ped.position.y))
            if ref_sp is not None:
                r_px = max(4, int(round(self.viewport.world_to_screen_dim(float(ref_ped.radius)))))
                pygame.draw.circle(surface, (148, 163, 184), ref_sp, r_px + 2, width=2)
                pygame.draw.circle(surface, (203, 213, 225), ref_sp, 3)
                drawn = True

        if ref_sp is not None and ref_ped is not None and selected_ped is not None and self._is_finite_xy(selected_ped.position.x, selected_ped.position.y):
            act_sp = self._safe_screen_point((selected_ped.position.x, selected_ped.position.y))
            delta_m = math.hypot(
                float(selected_ped.position.x) - float(ref_ped.position.x),
                float(selected_ped.position.y) - float(ref_ped.position.y),
            )
            if act_sp is not None and delta_m >= 0.12:
                self._draw_dashed_polyline(
                    surface, [ref_sp, act_sp], color=(244, 114, 182), width=2, dash_len=4.0, gap_len=4.0
                )
            if self.show_layer_labels:
                self._draw_label_pill(
                    surface,
                    f"ORIG ({delta_m:.2f}m)" if delta_m >= 0.05 else "ORIG",
                    (ref_sp[0] - 24, ref_sp[1] + 18),
                    (226, 232, 240),
                    bg_color=(30, 41, 59),
                    border_color=(148, 163, 184),
                    anchor=ref_sp,
                )

        self.latest_layer_states["baseline_ghost"] = {"rendered": drawn}

    def _render_social_interaction_overlay(
        self,
        surface: pygame.Surface,
        selected_ped: Any,
        target_id: str,
        mid_result: Optional[h_mid.MultiTrajectoryPrediction],
    ) -> None:
        """Visualize detected multi-agent social relations (links, halo, relation tag) cleanly."""
        if not self.show_social_interactions or selected_ped is None or mid_result is None:
            self.latest_layer_states["social_overlay"] = {"rendered": False}
            return

        summary = getattr(mid_result, "social_summary", None)
        if summary is None or summary.dominant_relation == "NONE" or summary.nearby_count <= 0:
            self.latest_layer_states["social_overlay"] = {"rendered": False, "relation": "NONE"}
            return

        sel_sp = self._safe_screen_point((selected_ped.position.x, selected_ped.position.y))
        if sel_sp is None:
            self.latest_layer_states["social_overlay"] = {"rendered": False}
            return

        rel_colors = {
            "FOLLOWING": (56, 189, 248),
            "AVOIDANCE": (251, 146, 60),
            "CONGESTION": (244, 63, 94),
            "MERGING": (192, 132, 252),
            "OBSTRUCTED": (239, 68, 68),
        }
        ring_col = rel_colors.get(summary.dominant_relation, (103, 232, 249))

        links_drawn = 0
        for nid in summary.interacting_neighbor_ids[:3]:
            nped = self.engine.registry.get(nid)
            if nped is None:
                continue
            d_nbr = math.hypot(
                float(selected_ped.position.x) - float(nped.position.x),
                float(selected_ped.position.y) - float(nped.position.y),
            )
            if d_nbr > 3.0:
                continue
            nsp = self._safe_screen_point((nped.position.x, nped.position.y))
            if nsp is None:
                continue

            if self.show_developer_overlay:
                pygame.draw.line(surface, (15, 23, 42), sel_sp, nsp, width=4)
                pygame.draw.line(surface, ring_col, sel_sp, nsp, width=2)
                pygame.draw.circle(surface, ring_col, nsp, 7, width=2)
            else:
                self._draw_dashed_polyline(
                    surface, [sel_sp, nsp], color=ring_col, width=1, dash_len=4.0, gap_len=4.0
                )
                pygame.draw.circle(surface, ring_col, nsp, 3)
            links_drawn += 1

        if self.show_developer_overlay and summary.dominant_relation != "NONE":
            pygame.draw.circle(surface, ring_col, sel_sp, 16, width=2)
            if self.show_layer_labels:
                self._draw_label_pill(
                    surface,
                    f"SOC:{summary.dominant_relation}",
                    (sel_sp[0] - 32, sel_sp[1] - 28),
                    (248, 250, 252),
                    bg_color=(30, 41, 59),
                    border_color=ring_col,
                    anchor=sel_sp,
                )

        self.latest_layer_states["social_overlay"] = {
            "rendered": bool(links_drawn > 0 or summary.dominant_relation != "NONE"),
            "relation": summary.dominant_relation,
            "nearby_count": summary.nearby_count,
            "links_drawn": links_drawn,
        }

    def _render_safety_avoidance_overlay(
        self,
        surface: pygame.Surface,
        selected_ped: Any,
        target_id: str,
    ) -> None:
        """Render active collision hazards, rejected unsafe stubs, and safe fallback vectors."""
        if not self.show_safety_overlays:
            self.latest_layer_states["safety_overlay"] = {"rendered": False}
            return

        curr_step = int(self.engine.clock.step_count)
        drawn_count = 0

        # Show safety intervention only when relevant to selected agent (or all in dev overlay)
        recent_events = [
            ev
            for ev in self.driven_controller.safety_layer.events[-12:]
            if (curr_step - ev.step_index <= 5) and (self.show_developer_overlay or ev.agent_id == target_id)
        ]
        seen_agents = set()
        for ev in reversed(recent_events):
            if ev.agent_id in seen_agents:
                continue
            seen_agents.add(ev.agent_id)
            ped_e = self.engine.registry.get(ev.agent_id)
            if ped_e is None:
                continue
            psp = self._safe_screen_point((ped_e.position.x, ped_e.position.y))
            prop_sp = self._safe_screen_point(ev.proposed_target)
            safe_sp = self._safe_screen_point(ev.safe_target)
            if psp is None:
                continue

            if self.show_developer_overlay:
                if prop_sp is not None:
                    self._draw_dashed_polyline(
                        surface, [psp, prop_sp], color=(239, 68, 68), width=2, dash_len=4.0, gap_len=4.0
                    )
                    pygame.draw.circle(surface, (239, 68, 68), prop_sp, 4, width=2)
                if safe_sp is not None:
                    pygame.draw.line(surface, (6, 78, 59), psp, safe_sp, width=5)
                    pygame.draw.line(surface, (52, 211, 153), psp, safe_sp, width=3)
                    pygame.draw.circle(surface, (52, 211, 153), safe_sp, 5)
                if ev.hazard_location is not None:
                    hsp = self._safe_screen_point(ev.hazard_location)
                    if hsp is not None:
                        pygame.draw.circle(surface, (251, 146, 60), hsp, 8, width=2)
                        pygame.draw.circle(surface, (239, 68, 68), hsp, 4, width=1)
            else:
                if safe_sp is not None:
                    self._draw_dashed_polyline(
                        surface, [psp, safe_sp], color=(52, 211, 153), width=2, dash_len=4.0, gap_len=4.0
                    )
                    pygame.draw.circle(surface, (52, 211, 153), safe_sp, 3)
            drawn_count += 1

        if selected_ped is not None and self.operating_mode == OperatingMode.MID_DRIVEN:
            latest_s = self.driven_controller.latest_safety.get(target_id)
            if latest_s is not None and latest_s.intervention_applied and self.show_layer_labels:
                sp = self._safe_screen_point((selected_ped.position.x, selected_ped.position.y))
                if sp is not None:
                    short_tag = {
                        "near_collision_risk": "NEAR_COL",
                        "dynamic_collision_avoided": "DYN_AVOID",
                        "static_obstacle_or_wall": "WALL_BLK",
                        "static_unwalkable_or_wall": "WALL_BLK",
                        "static_segment_collision": "WALL_BLK",
                        "forward_wall_clearance": "WALL_CLR",
                        "deadlock_escape": "DEADLOCK",
                    }.get(
                        str(latest_s.intervention_reason or ""),
                        str(latest_s.intervention_reason or "OVRD")[:10].upper(),
                    )
                    self._draw_label_pill(
                        surface,
                        f"SAFE:{short_tag}",
                        (sp[0] + 18, sp[1] + 22),
                        (167, 243, 208),
                        bg_color=(6, 78, 59),
                        border_color=(52, 211, 153),
                        anchor=sp,
                    )

        self.latest_layer_states["safety_overlay"] = {
            "rendered": drawn_count > 0,
            "active_events": drawn_count,
        }

    def _render_predictions(self, surface: pygame.Surface) -> None:
        """Render traceable layers, social relations, safety interventions, and baseline ghost comparison."""
        target_id = self.selected_agent_id
        if not target_id:
            self.latest_layer_states.clear()
            self._placed_label_rects = []
            return

        self._placed_label_rects = []
        selected_ped = self.engine.registry.get(target_id)
        for ped in self.engine.registry.all():
            if self._is_finite_xy(getattr(ped.position, "x", None), getattr(ped.position, "y", None)):
                psp = self._safe_screen_point((ped.position.x, ped.position.y))
                if psp is not None:
                    self._placed_label_rects.append(
                        pygame.Rect(psp[0] - 26, psp[1] - 26, 52, 36)
                    )
        if self.show_heatmap:
            bx, by, bw, bh = self.viewport.building_screen_bounds
            self._placed_label_rects.append(
                pygame.Rect(int(bx + 4), int(by + bh - 46), 192, 42)
            )
        step = int(self.engine.clock.step_count)
        timestamp = float(self.engine.clock.timestamp)
        if self.operating_mode == OperatingMode.MID_DRIVEN:
            snapshot = self._driven_history_buffer.buffer.get_snapshot(step, timestamp)
        else:
            snapshot = self.adapter.buffer.get_snapshot(step, timestamp)
        clean_history = self._prepare_clean_observation_history(snapshot, target_id)
        self.available_observations_count = len(clean_history)

        obs_anchor_snap: Optional[h_contract.TrajectorySnapshot] = None
        if len(clean_history) >= 2:
            latest_obs = clean_history[-1]
            obs_anchor_snap = h_contract.TrajectorySnapshot(
                current_step=int(latest_obs.step_index),
                current_timestamp=float(latest_obs.timestamp),
                timestep=snapshot.timestep,
                trajectories=snapshot.trajectories,
            )

        # Render semantic hierarchy: baseline ghost -> past -> CV -> MID -> selected -> adopted -> social/safety -> top ped
        self._render_baseline_ghost_overlay(surface, selected_ped, target_id)
        self._render_observed_history_layer(surface, clean_history)
        self._render_cv_prediction_layer(surface, obs_anchor_snap, target_id, clean_history)
        mid_result, rec_for_target, candidates_aligned, selected_idx, anchor_pt = (
            self._render_mid_candidates_layer(
                surface, obs_anchor_snap, snapshot, target_id, clean_history
            )
        )
        self._render_selected_mid_candidate_layer(
            surface, mid_result, rec_for_target, candidates_aligned, selected_idx, anchor_pt
        )
        self._render_adopted_route_layer(surface, selected_ped, target_id)
        self._render_safety_avoidance_overlay(surface, selected_ped, target_id)
        self._render_social_interaction_overlay(surface, selected_ped, target_id, mid_result)
        self._render_actual_movement_layer(surface, selected_ped, target_id)

    def _render_hud_and_telemetry(self, surface: pygame.Surface) -> None:
        """Render dedicated right-hand telemetry panel, 6-layer decision traceability, and controls bar."""
        self._ensure_fonts()
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
        ctrl_h = 36
        ctrl_y = self.height - ctrl_h
        panel_h = self.height

        panel_bg = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        pygame.draw.rect(surface, (15, 23, 42), panel_bg)
        pygame.draw.line(surface, (51, 65, 85), (panel_x, 0), (panel_x, panel_h), width=2)

        cur_y = 10
        margin_x = panel_x + 16
        content_w = panel_w - 32
        max_panel_y = ctrl_y - 6

        def _fit_val(text: str, max_px: int) -> pygame.Surface:
            s = self.font_small.render(text, True, (241, 245, 249))
            if s.get_width() <= max_px:
                return s
            for i in range(len(text) - 1, 0, -1):
                cand = text[:i] + "..."
                s = self.font_small.render(cand, True, (241, 245, 249))
                if s.get_width() <= max_px:
                    return s
            return s

        def _render_row(label: str, val: str, val_col: Tuple[int, int, int]) -> None:
            nonlocal cur_y
            if cur_y + 15 >= max_panel_y:
                return
            lbl_s = self.font_small.render(label, True, (148, 163, 184))
            surface.blit(lbl_s, (margin_x, cur_y))
            val_x = margin_x + 112
            val_max_w = content_w - 114
            val_s = self.font_small.render(val, True, val_col)
            if val_s.get_width() > val_max_w:
                for i in range(len(val) - 1, 0, -1):
                    cand = val[:i] + "..."
                    val_s = self.font_small.render(cand, True, val_col)
                    if val_s.get_width() <= val_max_w:
                        break
            surface.blit(val_s, (val_x, cur_y))
            cur_y += 15

        def _render_section_hdr(title: str, col: Tuple[int, int, int]) -> None:
            nonlocal cur_y
            if cur_y + 20 >= max_panel_y:
                return
            pygame.draw.line(surface, (30, 41, 59), (margin_x, cur_y), (margin_x + content_w, cur_y), width=1)
            cur_y += 5
            hdr = self.font_section.render(title, True, col)
            surface.blit(hdr, (margin_x, cur_y))
            cur_y += 18

        # --- SECTION 0: PROMINENT OPERATING MODE BANNER ---
        if self.presentation_mode:
            stage_names = {
                1: "STAGE 1: LIVE EVACUATION",
                2: "STAGE 2: SPATIAL HEATMAP",
                3: "STAGE 3: CONSTANT VELOCITY",
                4: "STAGE 4: MID MULTI-PATH PREDICTION",
                5: "STAGE 5: MID-ASSISTED MOVEMENT",
            }
            stage_title = stage_names.get(self.presentation_stage.value, f"STAGE {self.presentation_stage.value}")
            p_rect = pygame.Rect(margin_x, cur_y, content_w, 44)
            pygame.draw.rect(surface, (66, 32, 6), p_rect, border_radius=6)
            pygame.draw.rect(surface, (244, 63, 94), p_rect, width=2, border_radius=6)
            p_surf = self.font_section.render("PRESENTATION DEMO [D to exit]", True, (244, 63, 94))
            s_surf = self.font_small.render(stage_title, True, (250, 204, 21))
            surface.blit(p_surf, (margin_x + 10, cur_y + 5))
            surface.blit(s_surf, (margin_x + 10, cur_y + 23))
            cur_y += 50
        else:
            mode_meta = {
                OperatingMode.ORIGINAL_SIMULATION: (
                    "ORIGINAL SIMULATION",
                    "Pavan Physics Baseline (Unmodified)",
                    (56, 189, 248),
                    (12, 74, 110),
                ),
                OperatingMode.MID_PREDICTION_ONLY: (
                    "MID PREDICTION ONLY",
                    "Multi-Path Stochastic Inference (Read-Only)",
                    (192, 132, 252),
                    (59, 7, 100),
                ),
                OperatingMode.MID_ASSISTED_EVACUATION: (
                    "MID-ASSISTED EVACUATION",
                    "Dynamic Waypoint Guidance Active",
                    (52, 211, 153),
                    (6, 78, 59),
                ),
                OperatingMode.MID_DRIVEN: (
                    "MID-DRIVEN MOVEMENT",
                    "Neural MID Policy + Safety Shield",
                    (250, 204, 21),
                    (113, 63, 18),
                ),
            }
            m_title, m_sub, m_border, m_bg = mode_meta.get(
                self.operating_mode,
                (self.operating_mode.value, "", (148, 163, 184), (30, 41, 59)),
            )
            card_rect = pygame.Rect(margin_x, cur_y, content_w, 44)
            pygame.draw.rect(surface, m_bg, card_rect, border_radius=6)
            pygame.draw.rect(surface, m_border, card_rect, width=2, border_radius=6)

            m_title_surf = self.font_section.render(m_title, True, (255, 255, 255))
            m_sub_surf = self.font_small.render(m_sub, True, m_border)
            surface.blit(m_title_surf, (margin_x + 12, cur_y + 5))
            surface.blit(m_sub_surf, (margin_x + 12, cur_y + 23))
            cur_y += 52

        # --- SECTION 1: SIMULATION STATISTICS ---
        title_surf = self.font_title.render("SIMULATION TELEMETRY", True, (56, 189, 248))
        surface.blit(title_surf, (margin_x, cur_y))
        cur_y += 20

        status_str = "PAUSED" if self.engine.is_paused else "RUNNING"
        status_col = (251, 191, 36) if self.engine.is_paused else (74, 222, 128)

        _render_row("Clock / Step:", f"{sim_time:5.1f}s  (Step {step})", (241, 245, 249))
        _render_row("Engine State:", status_str, status_col)
        _render_row("Evacuation:", f"{evac_count}/{total_peds} Evacuated ({active_count} Active)", (241, 245, 249))
        _render_row("Mean Speed:", f"{mean_speed:.2f} m/s  |  Heatmap: {'ON' if self.show_heatmap else 'OFF'}", (203, 213, 225))

        # --- SECTION 2: SELECTED PEDESTRIAN & MOVEMENT DETAILS (LAYER 1 & 2) ---
        _render_section_hdr("SELECTED AGENT FOCUS", (250, 204, 21))

        target_id = self.selected_agent_id
        target_ped = self.engine.registry.get(target_id) if target_id else None
        path_pts_count = len(self.actual_traversed_paths.get(target_id, [])) if target_id else 0

        if target_ped:
            p_status = str(getattr(target_ped.status, "value", target_ped.status)).upper()
            p_pos = f"({target_ped.position.x:.1f}m, {target_ped.position.y:.1f}m)"
            p_spd = math.hypot(
                getattr(target_ped.velocity, "vx", getattr(target_ped.velocity, "x", 0.0)),
                getattr(target_ped.velocity, "vy", getattr(target_ped.velocity, "y", 0.0)),
            )
            _render_row("Agent Focus:", f"{target_id} -> {target_ped.target_exit_id or 'None'}", (250, 204, 21))
            _render_row("Status / Pos:", f"{p_status} @ {p_pos}", (241, 245, 249))
            _render_row("Traversed:", f"{path_pts_count} recorded pts | {self.available_observations_count} obs", (203, 213, 225))
            _render_row("Velocity:", f"{p_spd:.2f} m/s", (241, 245, 249))
        else:
            _render_row("Agent ID:", str(target_id or "None Selected"), (148, 163, 184))
            _render_row("Status:", "Not in registry", (239, 68, 68))

        # --- SECTION 3: PREDICTION PIPELINE STATUS & PROVENANCE (LAYER 3 & 4) ---
        _render_section_hdr("PREDICTIONS & PROVENANCE", (192, 132, 252))

        mid_prov = self.mid_predictor.training_provenance
        if self._mid_training_error:
            mid_prov_txt = f"TRAIN ERR: {self._mid_training_error[:20]}"
            mid_prov_col = (239, 68, 68)
        elif self.mid_predictor.is_trained:
            _epochs = int(mid_prov.get("training_epochs_completed", 10))
            _samples = int(mid_prov.get("training_sample_count", 54))
            _soc_tag = "|Social" if mid_prov.get("has_social_interactions", True) else ""
            mid_prov_txt = f"TRAINED ({_epochs}ep, {_samples}smp{_soc_tag})"
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

        mid_layer_state = self.latest_layer_states.get("mid_candidates", {})
        soc_layer_state = self.latest_layer_states.get("social_overlay", {})
        soc_rel = str(soc_layer_state.get("relation", "NONE"))
        soc_cnt = int(soc_layer_state.get("nearby_count", 0))
        c_gen = mid_layer_state.get("count", self.actual_returned_k)

        _render_row("MID / CV State:", f"{self.mid_status.value} | {self.cv_status.value}", status_colors.get(self.mid_status, (241, 245, 249)))
        _render_row("Provenance:", mid_prov_txt, mid_prov_col)
        _render_row("Social Context:", f"{soc_rel} ({soc_cnt} nearby)" if soc_cnt > 0 else "CLEAR (No nearby peds)", (56, 189, 248) if soc_rel == "NONE" else (251, 146, 60))
        _render_row("MID Candidates:", f"K={self.k_samples} gen (H={self.prediction_horizon * 0.1:.1f}s horizon)", (241, 245, 249))
        _render_row("Feasibility:", f"{self.last_mid_walkability * 100:.0f}% walk | {self.last_mid_collision_free * 100:.0f}% seg free", (74, 222, 128))

        # --- SECTION 4: MID DECISION, SAFETY & ADOPTION TRACE (LAYER 5 & 6) ---
        _render_section_hdr("DECISION, SAFETY & ADOPTION", (52, 211, 153))

        adopted_state = self.latest_layer_states.get("adopted_route", {})
        sel_state = self.latest_layer_states.get("selected_mid_candidate", {})
        adopted_rec = self.adopted_route_records_by_agent.get(target_id) if target_id else None
        sel_delta_m, cohort_mean_delta, cohort_max_delta = self._compute_baseline_divergence(target_id)

        if self.operating_mode == OperatingMode.MID_DRIVEN:
            drec = self.driven_controller.latest_decisions.get(target_id) if target_id else None
            dsafe = self.driven_controller.latest_safety.get(target_id) if target_id else None
            sl = self.driven_controller.safety_layer
            if drec is not None and drec.selected_sample_idx >= 0 and getattr(drec, "is_accepted", True):
                sel_cand_str = f"C{drec.selected_sample_idx} (Sample #{drec.selected_sample_idx})"
                sel_col = (250, 204, 21)
            elif drec is not None:
                sel_cand_str = "None (Safety/Fallback)"
                sel_col = (251, 146, 60)
            else:
                sel_cand_str = "None (Awaiting Evaluation)"
                sel_col = (148, 163, 184)

            safety_str = (
                f"{dsafe.intervention_reason} (v={dsafe.safe_speed_mps:.2f}m/s)"
                if (dsafe is not None and dsafe.intervention_applied)
                else f"CLEAR (clr={dsafe.min_clearance_m:.2f}m)" if dsafe is not None else "CLEAR"
            )
            safety_col = (52, 211, 153) if (dsafe and dsafe.intervention_applied) else (203, 213, 225)
            l6_hud_status = adopted_rec.adoption_status if adopted_rec is not None else adopted_state.get("status", "NOT_ADOPTED")
            l6_wps = adopted_state.get("waypoints", ())

            _render_row("Driver Mode:", f"MID-DRIVEN ({self.driven_controller.mid_driven_steps} control steps)", (250, 204, 21))
            _render_row("Selected MID:", sel_cand_str, sel_col)
            _render_row("Safety Layer:", safety_str, safety_col)
            _render_row("vs Original:", f"sel {sel_delta_m:.2f}m | avg {cohort_mean_delta:.2f}m", (244, 114, 182))
            _render_row("Adopted Route:", f"{l6_hud_status} ({len(l6_wps)} wp -> {getattr(target_ped, 'target_exit_id', '') or '?'})" if l6_wps else "NONE (Engine wps)", (74, 222, 128) if l6_wps else (251, 191, 36))

        elif self.operating_mode == OperatingMode.MID_ASSISTED_EVACUATION:
            rec = (
                self.controller.latest_recommendations_by_agent.get(target_id)
                if target_id
                else None
            ) or self.latest_recommendation
            if rec and rec.is_accepted and rec.selected_sample_idx >= 0:
                sel_cand_str = f"C{rec.selected_sample_idx} (Sample #{rec.selected_sample_idx})"
                sel_col = (250, 204, 21)
                decision_rsn = str(rec.selection_reason)
            elif rec:
                sel_cand_str = "None (Safety/Fallback)"
                sel_col = (251, 146, 60)
                decision_rsn = str(rec.rejection_reason or rec.selection_reason)
            else:
                sel_cand_str = "None (Awaiting Evaluation)"
                sel_col = (148, 163, 184)
                decision_rsn = "Awaiting observation interval"

            l6_hud_status = adopted_rec.adoption_status if adopted_rec is not None else adopted_state.get("status", "NOT_ADOPTED")
            l6_wps = adopted_state.get("waypoints", ())

            _render_row("Controller:", "MID-ASSISTED (Waypoint Steering)", (52, 211, 153))
            _render_row("Selected MID:", sel_cand_str, sel_col)
            _render_row("Decision:", decision_rsn, (203, 213, 225))
            _render_row("vs Original:", f"sel {sel_delta_m:.2f}m | avg {cohort_mean_delta:.2f}m", (244, 114, 182))
            _render_row("Adopted Route:", f"{l6_hud_status} ({len(l6_wps)} wp -> {getattr(target_ped, 'target_exit_id', '') or '?'})" if l6_wps else "NONE (Baseline active)", (74, 222, 128) if l6_wps else (251, 191, 36))

        else:
            base_wp_cnt = len(getattr(target_ped, "waypoints", []) or []) if target_ped else 0
            _render_row("Controller:", f"INACTIVE ({self.operating_mode.value})", (148, 163, 184))
            _render_row("Selected MID:", "None (Controller Inactive)", (148, 163, 184))
            _render_row("Adopted Route:", f"NONE ({base_wp_cnt} baseline engine wp)", (148, 163, 184))
            _render_row("Activate Ctrl:", "Press [M] -> ASSISTED / DRIVEN", (52, 211, 153))

        # --- SECTION 5: PRESENTATION & DIAGNOSTICS ---
        _render_section_hdr("PRESENTATION & DIAGNOSTICS", (241, 245, 249))

        dev_status = "ENABLED [Press L to hide]" if self.show_developer_overlay else "MINIMAL [Press L for details]"
        dev_col = (250, 204, 21) if self.show_developer_overlay else (148, 163, 184)
        _render_row("Overlay Mode:", dev_status, dev_col)

        if not self.show_developer_overlay:
            # Minimal Presentation Legend (Clean, 4 core elements)
            pres_legend = [
                ("Actual Path (Traversed)", (37, 99, 235), "actual"),
                ("MID Stochastic Candidates", (168, 85, 247), "candidates"),
                ("Selected MID Choice", (250, 204, 21), "selected"),
                ("Adopted Route (Engine wps)", (22, 163, 74), "adopted"),
            ]
            for name, col, style in pres_legend:
                if cur_y + 15 >= max_panel_y:
                    break
                line_y = cur_y + 6
                if style == "actual":
                    pygame.draw.line(surface, col, (margin_x, line_y), (margin_x + 18, line_y), width=2)
                    pygame.draw.circle(surface, (56, 189, 248), (margin_x + 18, line_y), 3)
                elif style == "candidates":
                    pygame.draw.line(surface, col, (margin_x, line_y), (margin_x + 18, line_y), width=2)
                    pygame.draw.circle(surface, col, (margin_x + 18, line_y), 3)
                elif style == "selected":
                    pygame.draw.line(surface, (120, 53, 15), (margin_x, line_y), (margin_x + 18, line_y), width=4)
                    pygame.draw.line(surface, col, (margin_x, line_y), (margin_x + 18, line_y), width=2)
                    pygame.draw.circle(surface, col, (margin_x + 18, line_y), 4)
                elif style == "adopted":
                    pygame.draw.line(surface, col, (margin_x, line_y), (margin_x + 18, line_y), width=2)
                    sq = pygame.Rect(margin_x + 15, line_y - 3, 6, 6)
                    pygame.draw.rect(surface, col, sq)

                lbl = self.font_small.render(name, True, (203, 213, 225))
                surface.blit(lbl, (margin_x + 28, cur_y))
                cur_y += 15
        else:
            # Full Developer Diagnostics Legend (All 6 layers + Ghost + Social/Safety)
            dev_legend = [
                ("L1 Actual ped path", (37, 99, 235), "actual"),
                ("L2 Observed history", (245, 158, 11), "dotted"),
                ("L3 CV baseline forecast", (255, 255, 255), "dashed"),
                ("L4 MID candidates (C0..Ck)", (168, 85, 247), "candidates"),
                ("L5 Selected candidate", (250, 204, 21), "selected"),
                ("L6 Adopted engine route", (22, 163, 74), "adopted"),
                ("Orig ghost baseline [G]", (148, 163, 184), "dashed"),
                ("Social link [I] / Safety [C]", (52, 211, 153), "adopted"),
            ]
            for name, col, style in dev_legend:
                if cur_y + 15 >= max_panel_y:
                    break
                line_y = cur_y + 6
                if style == "actual":
                    pygame.draw.line(surface, col, (margin_x, line_y), (margin_x + 18, line_y), width=2)
                    pygame.draw.circle(surface, (56, 189, 248), (margin_x + 18, line_y), 3)
                elif style == "dotted":
                    pygame.draw.circle(surface, col, (margin_x + 4, line_y), 2)
                    pygame.draw.circle(surface, col, (margin_x + 10, line_y), 2)
                    pygame.draw.circle(surface, col, (margin_x + 16, line_y), 2)
                elif style == "dashed":
                    pygame.draw.line(surface, col, (margin_x, line_y), (margin_x + 8, line_y), width=2)
                    pygame.draw.line(surface, col, (margin_x + 12, line_y), (margin_x + 18, line_y), width=2)
                elif style == "candidates":
                    pygame.draw.line(surface, col, (margin_x, line_y), (margin_x + 18, line_y), width=2)
                elif style == "selected":
                    pygame.draw.line(surface, (120, 53, 15), (margin_x, line_y), (margin_x + 18, line_y), width=4)
                    pygame.draw.line(surface, col, (margin_x, line_y), (margin_x + 18, line_y), width=2)
                    pygame.draw.circle(surface, col, (margin_x + 18, line_y), 4)
                elif style == "adopted":
                    pygame.draw.line(surface, col, (margin_x, line_y), (margin_x + 18, line_y), width=2)
                    sq = pygame.Rect(margin_x + 15, line_y - 3, 6, 6)
                    pygame.draw.rect(surface, col, sq)

                lbl = self.font_small.render(name, True, (203, 213, 225))
                surface.blit(lbl, (margin_x + 28, cur_y))
                cur_y += 15

        # --- BOTTOM INTERACTIVE CONTROLS BAR ---
        ctrl_bg = pygame.Rect(0, ctrl_y, self.width, ctrl_h)
        pygame.draw.rect(surface, (15, 23, 42), ctrl_bg)
        pygame.draw.line(surface, (51, 65, 85), (0, ctrl_y), (self.width, ctrl_y), width=1)

        ctrl_text = (
            "[M] Cycle Mode  [L] Developer Overlay  [H] Heatmap  [G] Ghost Baseline  "
            "[SPACE] Pause/Resume  [R] Reset  [CLICK] Select Pedestrian"
        )
        ctrl_surf = self.font_legend.render(ctrl_text, True, (203, 213, 225))
        surface.blit(ctrl_surf, (16, ctrl_y + 10))

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
                # Invalidate prediction cache selector pointer on selection change
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

            elif event.key == pygame.K_t:
                self.show_actual_path = not self.show_actual_path

            elif event.key == pygame.K_o:
                self.show_observed_history = not self.show_observed_history

            elif event.key == pygame.K_a:
                self.show_adopted_route = not self.show_adopted_route

            elif event.key in (pygame.K_l, pygame.K_v):
                self.show_developer_overlay = not self.show_developer_overlay
                self.show_layer_labels = self.show_developer_overlay
                self.renderer.draw_labels = self.show_developer_overlay

            elif event.key == pygame.K_i:
                self.show_social_interactions = not self.show_social_interactions

            elif event.key == pygame.K_c:
                self.show_safety_overlays = not self.show_safety_overlays

            elif event.key == pygame.K_g:
                self.show_baseline_ghost = not self.show_baseline_ghost

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
                    # Cycle through Operating Modes cleanly
                    modes = list(OperatingMode)
                    idx = modes.index(self.operating_mode)
                    self.operating_mode = modes[(idx + 1) % len(modes)]
                    if self.operating_mode == OperatingMode.MID_ASSISTED_EVACUATION:
                        self._apply_mid_assisted_guidance()
                    elif self.operating_mode == OperatingMode.MID_DRIVEN:
                        self._apply_mid_driven_step()
                    else:
                        self._restore_baseline_routes_if_needed()
                elif event.key == pygame.K_h:
                    self.show_heatmap = not self.show_heatmap
                elif event.key == pygame.K_p:
                    self.show_mid_predictions = not self.show_mid_predictions
                elif event.key == pygame.K_b:
                    self.show_baseline_predictions = not self.show_baseline_predictions
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
        clock = self.clock

        self._ensure_fonts()

        running = True
        frames = 0

        while running:
            dt_millis = clock.tick(self.fps)
            dt_wall = dt_millis / 1000.0

            for event in pygame.event.get():
                if not self.handle_event(event):
                    running = False
                    break

            if not running:
                break

            # 1. Advance simulation physics (Module B Part 1)
            self.update_physics(dt_wall)

            # 2. Draw building floor, obstacles, walls, exits (Module B Part 1)
            self.renderer.draw(self.screen, self.building, self.viewport, engine=None)

            # 3. Draw subtle spatial density heatmap on the floor (Module B Part 2)
            self._render_spatial_heatmap(self.screen)

            # 4. Draw MID trajectory predictions & baseline (Module B Part 2)
            self._render_predictions(self.screen)

            # 5. Draw pedestrians on top of heatmap and predictions
            self.renderer._draw_pedestrians(self.screen, self.engine.registry, self.viewport)

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
        choices=["original", "prediction", "assisted", "driven"],
        default="original",
        help="Initial operating mode: original, prediction, assisted, or driven",
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
    elif args.mode == "driven":
        app.operating_mode = OperatingMode.MID_DRIVEN

    if args.presentation:
        app.presentation_mode = True
        app.set_presentation_stage(PresentationStage.STAGE_1_LIVE_EVACUATION)

    app.run()


if __name__ == "__main__":
    main()