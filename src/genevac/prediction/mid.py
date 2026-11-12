"""Original Evacuation Motion Indeterminacy Diffusion (MID) for GenEvac-Sentinel.

Provides:
- EvacuationMIDConfig: Validated hyperparameters and diffusion noise schedule settings.
- EvacuationConditioning: 16-dimensional conditioning feature container with documented
  canonical normalization applied consistently across training and inference.
- MultiTrajectoryPrediction: Container for multiple stochastic trajectory samples,
  including diversity, point walkability, segment collision validity, and training provenance.
- EvacuationMIDNetwork: PyTorch denoising score network.
- EvacuationMIDPredictor: Full stochastic trajectory predictor with per-sample physical
  guidance, isolated deterministic RNG, training pipeline with loss masking, and audit metrics.

Designed from first principles for GenEvac-Sentinel emergency evacuation scenarios.
Does not use, copy, or depend on external proprietary MID repositories.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from genevac.environment import EnvironmentMap, Exit, Obstacle
from genevac.metrics import (
    calculate_ade,
    calculate_area_density,
    calculate_fde,
    calculate_minimum_separation,
)
from genevac.prediction.data_contract import (
    PredictedPosition,
    TrajectoryCategory,
    TrajectoryObservation,
    TrajectoryPrediction,
    TrajectorySnapshot,
    TrajectoryValidationError,
)
from genevac.prediction.evaluation import EvaluationSample


# =============================================================================
# 1. Configuration with Strict Precondition Validation
# =============================================================================

@dataclass(frozen=True)
class EvacuationMIDConfig:
    """Hyperparameter configuration for Evacuation MID with strict validation."""
    diffusion_steps: int = 20
    beta_start: float = 0.0001
    beta_end: float = 0.04
    horizon: int = 12
    prediction_interval: float = 0.1
    conditioning_dim: int = 16
    d_model: int = 64
    hidden_dim: int = 128
    w_exit: float = 0.20
    w_obs: float = 0.40
    w_kin: float = 0.35
    decay_kin: float = 0.15
    k_samples: int = 20
    learning_rate: float = 1e-3
    device: str = "cpu"

    def __post_init__(self) -> None:
        if type(self.diffusion_steps) is not int or self.diffusion_steps < 1:
            raise ValueError(f"diffusion_steps must be an integer >= 1, got {self.diffusion_steps!r}")
        if (
            type(self.beta_start) not in (float, int)
            or not math.isfinite(self.beta_start)
            or self.beta_start <= 0.0
            or self.beta_start >= 1.0
        ):
            raise ValueError(f"beta_start must be in (0, 1), got {self.beta_start!r}")
        if (
            type(self.beta_end) not in (float, int)
            or not math.isfinite(self.beta_end)
            or self.beta_end <= self.beta_start
            or self.beta_end >= 1.0
        ):
            raise ValueError(f"beta_end must be in (beta_start, 1), got {self.beta_end!r}")
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError(f"horizon must be an integer >= 1, got {self.horizon!r}")
        if (
            type(self.prediction_interval) not in (float, int)
            or not math.isfinite(self.prediction_interval)
            or self.prediction_interval <= 0.0
        ):
            raise ValueError(f"prediction_interval must be a positive finite float, got {self.prediction_interval!r}")
        if type(self.conditioning_dim) is not int or self.conditioning_dim < 1:
            raise ValueError(f"conditioning_dim must be an integer >= 1, got {self.conditioning_dim!r}")
        if type(self.d_model) is not int or self.d_model < 1:
            raise ValueError(f"d_model must be an integer >= 1, got {self.d_model!r}")
        if type(self.hidden_dim) is not int or self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be an integer >= 1, got {self.hidden_dim!r}")
        if type(self.k_samples) is not int or self.k_samples < 1:
            raise ValueError(f"k_samples must be an integer >= 1, got {self.k_samples!r}")


# =============================================================================
# 2. Conditioning Representation & Canonical Normalization
# =============================================================================

# Canonical normalization scales for the 16 features:
# Maps physical units into approximately [-1, 1] or [0, 1] for neural network stability.
FEATURE_NORMALIZATION_SCALES: Tuple[float, ...] = (
    2.0,   # 0: velocity_x (m/s) -> [-2, 2] / 2 -> [-1, 1]
    2.0,   # 1: velocity_y (m/s) -> [-2, 2] / 2 -> [-1, 1]
    2.0,   # 2: speed (m/s) -> [0, 2] / 2 -> [0, 1]
    1.0,   # 3: heading_x (unit vector) -> [-1, 1]
    1.0,   # 4: heading_y (unit vector) -> [-1, 1]
    2.0,   # 5: accel_x (m/s^2) -> [-2, 2] / 2 -> [-1, 1]
    2.0,   # 6: accel_y (m/s^2) -> [-2, 2] / 2 -> [-1, 1]
    50.0,  # 7: exit_displacement_x (m) -> [-50, 50] / 50 -> [-1, 1]
    50.0,  # 8: exit_displacement_y (m) -> [-50, 50] / 50 -> [-1, 1]
    50.0,  # 9: exit_distance (m) -> [0, 50] / 50 -> [0, 1]
    1.0,   # 10: exit_heading_x (unit vector) -> [-1, 1]
    1.0,   # 11: exit_heading_y (unit vector) -> [-1, 1]
    4.0,   # 12: local_density (ped/m^2) -> [0, 4] / 4 -> [0, 1]
    50.0,  # 13: nearest_neighbor_dist (m) -> [0, 50] / 50 -> [0, 1]
    50.0,  # 14: nearest_wall_dist (m) -> [0, 50] / 50 -> [0, 1]
    50.0,  # 15: nearest_obstacle_dist (m) -> [0, 50] / 50 -> [0, 1]
)


@dataclass(frozen=True)
class EvacuationConditioning:
    """Structured container for pedestrian evacuation conditioning features.

    Maintains:
    - vector: Raw physical quantities in SI units (m, m/s, m/s^2, ped/m^2).
    - normalized_vector: Scaled dimensionless features in range [-1.0, 1.0] used
      identically by neural conditioning encoders during both training and inference.
    """
    agent_id: str
    reference_timestamp: float
    reference_position: Tuple[float, float]
    velocity: Tuple[float, float]
    speed: float
    heading: Tuple[float, float]
    accel: Tuple[float, float]
    exit_displacement: Tuple[float, float]
    exit_distance: float
    exit_heading: Tuple[float, float]
    local_density: float
    nearest_neighbor_dist: float
    nearest_wall_dist: float
    nearest_obstacle_dist: float
    vector: Tuple[float, ...]
    normalized_vector: Tuple[float, ...]

    @classmethod
    def from_snapshot_and_env(
        cls,
        snapshot: TrajectorySnapshot,
        agent_id: str,
        environment: Optional[EnvironmentMap] = None,
        target_exit_id: Optional[str] = None,
        crowd_radius: float = 3.0,
    ) -> EvacuationConditioning:
        """Extract conditioning features directly from authoritative snapshot and environment.

        Preconditions:
        - Agent must have at least 2 observations in snapshot.
        - If target_exit_id is explicitly specified, it must exist in environment.exits.
        - Timestamps must be strictly increasing and finite.
        """
        obs_seq = snapshot.trajectories.get(agent_id, ())
        if len(obs_seq) < 2:
            raise TrajectoryValidationError(
                f"Insufficient history for agent '{agent_id}': requires at least 2 observations for conditioning, got {len(obs_seq)}",
                field="observations",
                agent_id=agent_id,
            )

        o_last = obs_seq[-1]
        o_prev = obs_seq[-2]

        if not (math.isfinite(o_last.x) and math.isfinite(o_last.y) and math.isfinite(o_prev.x) and math.isfinite(o_prev.y)):
            raise TrajectoryValidationError(
                f"Non-finite coordinates detected for agent '{agent_id}' in snapshot history.",
                field="coordinates",
                agent_id=agent_id,
            )

        ref_pos = (o_last.x, o_last.y)
        dt = o_last.timestamp - o_prev.timestamp
        if dt <= 0.0 or not math.isfinite(dt):
            raise TrajectoryValidationError(
                f"Non-positive or non-finite observation dt={dt} for agent '{agent_id}'.",
                field="timestamp",
                agent_id=agent_id,
            )

        # 1. Motion kinematics
        vx = (o_last.x - o_prev.x) / dt
        vy = (o_last.y - o_prev.y) / dt
        if not (math.isfinite(vx) and math.isfinite(vy)):
            raise TrajectoryValidationError(
                f"Non-finite velocity computed for agent '{agent_id}': vx={vx}, vy={vy}",
                field="velocity",
                agent_id=agent_id,
            )

        speed = math.hypot(vx, vy)
        eps = 1e-6
        heading = (vx / (speed + eps), vy / (speed + eps))

        # Acceleration from last 3 observations if available
        if len(obs_seq) >= 3:
            o_ante = obs_seq[-3]
            dt_prev = o_prev.timestamp - o_ante.timestamp
            if dt_prev > 0.0 and math.isfinite(dt_prev):
                vx_prev = (o_prev.x - o_ante.x) / dt_prev
                vy_prev = (o_prev.y - o_ante.y) / dt_prev
                ax = (vx - vx_prev) / dt
                ay = (vy - vy_prev) / dt
                if not (math.isfinite(ax) and math.isfinite(ay)):
                    ax, ay = 0.0, 0.0
            else:
                ax, ay = 0.0, 0.0
        else:
            ax, ay = 0.0, 0.0

        # 2. Exit goal geometry
        target_exit_pos: Optional[Tuple[float, float]] = None
        if environment and environment.exits:
            if target_exit_id is not None:
                matched = [ex for ex in environment.exits if ex.id == target_exit_id]
                if not matched:
                    raise TrajectoryValidationError(
                        f"Target exit '{target_exit_id}' does not exist in environment exits ({[ex.id for ex in environment.exits]}). "
                        f"Silent substitution of target exits is strictly prohibited.",
                        field="target_exit_id",
                        agent_id=agent_id,
                    )
                target_exit_pos = matched[0].position
            else:
                # Default to nearest exit if none explicitly requested
                dists = [math.hypot(ex.position[0] - o_last.x, ex.position[1] - o_last.y) for ex in environment.exits]
                min_idx = int(np.argmin(dists))
                target_exit_pos = environment.exits[min_idx].position

        if target_exit_pos is not None:
            dx_e = target_exit_pos[0] - o_last.x
            dy_e = target_exit_pos[1] - o_last.y
            dist_e = math.hypot(dx_e, dy_e)
            heading_e = (dx_e / (dist_e + eps), dy_e / (dist_e + eps))
        else:
            dx_e, dy_e = 0.0, 0.0
            dist_e = 0.0
            heading_e = (0.0, 0.0)

        # 3. Crowd density and proximity
        all_positions = [
            (seq[-1].x, seq[-1].y)
            for aid, seq in snapshot.trajectories.items()
            if seq and aid != agent_id and math.isfinite(seq[-1].x) and math.isfinite(seq[-1].y)
        ]
        if all_positions:
            bounds = (
                max(0.0, o_last.x - crowd_radius),
                max(0.0, o_last.y - crowd_radius),
                o_last.x + crowd_radius,
                o_last.y + crowd_radius,
            )
            density = calculate_area_density(all_positions, bounds)
            distances = [math.hypot(px - o_last.x, py - o_last.y) for px, py in all_positions]
            min_neighbor_dist = min(distances) if distances else 50.0
        else:
            density = 0.0
            min_neighbor_dist = 50.0

        # 4. Environmental boundaries and obstacles
        if environment:
            wall_dist = min(o_last.x, environment.width - o_last.x, o_last.y, environment.height - o_last.y)
            wall_dist = max(0.0, float(wall_dist))
            obs_distances = []
            for obs in environment.obstacles:
                min_x, min_y, max_x, max_y = obs.bounds
                cx = max(min_x, min(o_last.x, max_x))
                cy = max(min_y, min(o_last.y, max_y))
                obs_distances.append(math.hypot(o_last.x - cx, o_last.y - cy))
            min_obs_dist = min(obs_distances) if obs_distances else 50.0
        else:
            wall_dist = 50.0
            min_obs_dist = 50.0

        # Raw 16-dimensional physical vector
        raw_vector = (
            float(vx),
            float(vy),
            float(speed),
            float(heading[0]),
            float(heading[1]),
            float(ax),
            float(ay),
            float(dx_e),
            float(dy_e),
            float(dist_e),
            float(heading_e[0]),
            float(heading_e[1]),
            float(density),
            float(min(min_neighbor_dist, 50.0)),
            float(min(wall_dist, 50.0)),
            float(min(min_obs_dist, 50.0)),
        )

        # Canonical normalization scaling
        norm_list = []
        for val, scale in zip(raw_vector, FEATURE_NORMALIZATION_SCALES):
            norm_val = val / scale
            # Clamp normalized features to safe range [-2.0, 2.0]
            norm_list.append(float(np.clip(norm_val, -2.0, 2.0)))
        normalized_vector = tuple(norm_list)

        return cls(
            agent_id=agent_id,
            reference_timestamp=o_last.timestamp,
            reference_position=ref_pos,
            velocity=(vx, vy),
            speed=speed,
            heading=heading,
            accel=(ax, ay),
            exit_displacement=(dx_e, dy_e),
            exit_distance=dist_e,
            exit_heading=heading_e,
            local_density=density,
            nearest_neighbor_dist=min_neighbor_dist,
            nearest_wall_dist=wall_dist,
            nearest_obstacle_dist=min_obs_dist,
            vector=raw_vector,
            normalized_vector=normalized_vector,
        )


# =============================================================================
# 3. Geometry Helpers (Point Walkability & Line Segment Intersection)
# =============================================================================

def line_segment_intersects_box(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    box: Tuple[float, float, float, float],
) -> bool:
    """Determine whether a line segment from p1 to p2 intersects an axis-aligned bounding box.

    Uses the Liang-Barsky parametric clipping algorithm. Returns True if any portion of
    the segment passes through or touches the interior of the bounding box.
    """
    min_x, min_y, max_x, max_y = box
    x1, y1 = p1
    x2, y2 = p2

    # If either endpoint is strictly inside the box, collision detected
    if (min_x <= x1 <= max_x) and (min_y <= y1 <= max_y):
        return True
    if (min_x <= x2 <= max_x) and (min_y <= y2 <= max_y):
        return True

    dx = x2 - x1
    dy = y2 - y1

    p = [-dx, dx, -dy, dy]
    q = [x1 - min_x, max_x - x1, y1 - min_y, max_y - y1]

    u1 = 0.0
    u2 = 1.0

    for i in range(4):
        if abs(p[i]) < 1e-9:
            # Line is parallel to edge
            if q[i] < 0:
                return False
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                if t > u2:
                    return False
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return False
                if t < u2:
                    u2 = t

    return u1 <= u2


# =============================================================================
# 4. Multi-Trajectory Prediction Container
# =============================================================================

@dataclass(frozen=True)
class MultiTrajectoryPrediction:
    """Container for stochastic multi-sample trajectory predictions.

    Maintains:
    - samples: All K individually sampled trajectories.
    - mean_prediction: Point-wise arithmetic mean of all K paths.
      NOTE: In multi-modal path branching (e.g. splitting left and right around an obstacle),
      the arithmetic mean may pass directly through an obstacle even when individual paths are safe.
    - sample_diversity: Average pairwise Euclidean divergence between distinct sample paths.
    - point_walkability: Fraction of predicted points inside navigable room space.
    - segment_collision_free: Fraction of trajectory line segments that do not collide with obstacles.
    - is_trained_model: Explicit provenance boolean showing whether model was trained or untrained.
    """
    agent_id: str
    source_step_index: int
    source_timestamp: float
    samples: Tuple[TrajectoryPrediction, ...]
    mean_prediction: TrajectoryPrediction
    sample_diversity: float
    point_walkability: float
    segment_collision_free: float
    walkable_validity: float  # Alias for point_walkability for backward compatibility
    is_trained_model: bool
    sample_count: int


# =============================================================================
# 5. Denoising Neural Network Architecture
# =============================================================================

class EvacuationMIDNetwork(nn.Module):
    """PyTorch denoising diffusion neural network for evacuation trajectories."""

    def __init__(self, config: EvacuationMIDConfig) -> None:
        super().__init__()
        self.config = config
        self.horizon = config.horizon
        self.traj_dim = config.horizon * 2
        d_model = config.d_model
        hidden = config.hidden_dim

        # 1. Conditioning encoder (takes normalized 16-dimensional vector)
        self.cond_encoder = nn.Sequential(
            nn.Linear(config.conditioning_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # 2. Diffusion step embedding (Sinusoidal + MLP)
        self.step_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # 3. Trajectory projection
        self.traj_in = nn.Sequential(
            nn.Linear(self.traj_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        # 4. Residual MLP Backbone
        in_dim = d_model * 3
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, in_dim),
        )
        self.block2 = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, in_dim),
        )

        # 5. Output projection head (initialized to small weights for stable diffusion)
        self.head = nn.Linear(in_dim, self.traj_dim)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)

    def _sinusoidal_step_embedding(self, step: torch.Tensor, dim: int) -> torch.Tensor:
        """Create standard sinusoidal time step embeddings."""
        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=step.device) / half_dim)
        args = step.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, x_noisy: torch.Tensor, step: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Estimate added noise epsilon given noisy trajectory, diffusion step, and conditioning.

        Args:
            x_noisy: Tensor of shape (Batch, Horizon, 2) or (Batch, Horizon*2).
            step: Tensor of shape (Batch,) with diffusion step indices in [0, N-1].
            cond: Tensor of shape (Batch, Conditioning_Dim).

        Returns:
            Tensor of estimated noise with shape (Batch, Horizon, 2).
        """
        B = x_noisy.shape[0]
        x_flat = x_noisy.reshape(B, self.traj_dim)

        e_cond = self.cond_encoder(cond)
        raw_step = self._sinusoidal_step_embedding(step, self.config.d_model)
        e_step = self.step_mlp(raw_step)
        e_traj = self.traj_in(x_flat)

        latent = torch.cat([e_traj, e_step, e_cond], dim=-1)
        latent = latent + self.block1(latent)
        latent = latent + self.block2(latent)

        out = self.head(latent)
        return out.reshape(B, self.horizon, 2)


# =============================================================================
# 6. Evacuation MID Predictor
# =============================================================================

class EvacuationMIDPredictor:
    """Stochastic Motion Indeterminacy Diffusion predictor for evacuation trajectories."""

    def __init__(self, config: Optional[EvacuationMIDConfig] = None) -> None:
        self.config = config if config is not None else EvacuationMIDConfig()
        self.network = EvacuationMIDNetwork(self.config).to(self.config.device)
        self._setup_schedules()

        # Provenance tracking
        self._is_trained: bool = False
        self._training_sample_count: int = 0
        self._training_epochs_completed: int = 0

    @property
    def is_trained(self) -> bool:
        """Indicates whether this model has undergone training optimization."""
        return self._is_trained

    @property
    def training_provenance(self) -> Dict[str, Any]:
        """Audit dictionary detailing training status and sample counts."""
        return {
            "is_trained": self._is_trained,
            "training_sample_count": self._training_sample_count,
            "training_epochs_completed": self._training_epochs_completed,
            "device": self.config.device,
        }

    def _setup_schedules(self) -> None:
        """Set up diffusion noise schedules."""
        N = self.config.diffusion_steps
        betas = torch.linspace(self.config.beta_start, self.config.beta_end, N, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def _compute_per_sample_guidance(
        self,
        sample_displacements: np.ndarray,
        conditioning: EvacuationConditioning,
        environment: Optional[EnvironmentMap],
        effective_horizon: int,
    ) -> np.ndarray:
        """Calculate physical evacuation guidance force vector field for a specific sample trajectory.

        Guarantees:
        - Evaluated independently for each sample's specific coordinates.
        - Robust obstacle repulsion: If a point is inside an obstacle, computes deterministic
          outward escape direction toward the nearest boundary face.
        - Numerically stable and capped to prevent explosion.
        """
        H = effective_horizon
        guidance = np.zeros((H, 2), dtype=np.float32)
        ref_x, ref_y = conditioning.reference_position
        eps = 1e-6
        max_force = 5.0  # Numerical stability clamping

        # 1. Exit attraction
        if conditioning.exit_distance > 0.1:
            u_exit = np.array(conditioning.exit_heading, dtype=np.float32)
            guidance += self.config.w_exit * u_exit

        # 2. Kinematic forward inertia
        u_kin = np.array(conditioning.heading, dtype=np.float32)
        for k in range(H):
            decay = math.exp(-k * self.config.decay_kin)
            guidance[k] += self.config.w_kin * decay * u_kin

        # 3. Obstacle avoidance & Escape vector field
        if environment and environment.obstacles:
            for k in range(H):
                cur_x = ref_x + sample_displacements[k, 0]
                cur_y = ref_y + sample_displacements[k, 1]

                for obs in environment.obstacles:
                    min_x, min_y, max_x, max_y = obs.bounds

                    # Case A: Point is inside the obstacle -> Strong outward escape vector
                    if (min_x <= cur_x <= max_x) and (min_y <= cur_y <= max_y):
                        d_left = cur_x - min_x
                        d_right = max_x - cur_x
                        d_bottom = cur_y - min_y
                        d_top = max_y - cur_y

                        # Find closest boundary face with deterministic tie-breaking (left, right, bottom, top)
                        min_d = min(d_left, d_right, d_bottom, d_top)
                        if min_d == d_left:
                            esc_x, esc_y = -1.0, 0.0
                        elif min_d == d_right:
                            esc_x, esc_y = 1.0, 0.0
                        elif min_d == d_bottom:
                            esc_x, esc_y = 0.0, -1.0
                        else:
                            esc_x, esc_y = 0.0, 1.0

                        escape_force = max_force
                        guidance[k, 0] += escape_force * esc_x
                        guidance[k, 1] += escape_force * esc_y

                    # Case B: Point is outside the obstacle -> Inverse-distance repulsion
                    else:
                        cx = max(min_x, min(cur_x, max_x))
                        cy = max(min_y, min(cur_y, max_y))
                        dist = math.hypot(cur_x - cx, cur_y - cy)

                        if dist < 2.0:
                            rep_dx = cur_x - cx
                            rep_dy = cur_y - cy
                            rep_norm = math.hypot(rep_dx, rep_dy) + eps
                            force = min(max_force, self.config.w_obs / (dist + 0.1)**2)
                            guidance[k, 0] += force * (rep_dx / rep_norm)
                            guidance[k, 1] += force * (rep_dy / rep_norm)

        # Cap guidance force magnitude for numerical stability
        for k in range(H):
            g_norm = math.hypot(guidance[k, 0], guidance[k, 1])
            if g_norm > max_force:
                guidance[k] = guidance[k] * (max_force / g_norm)

        return guidance

    def sample_trajectories(
        self,
        snapshot: TrajectorySnapshot,
        agent_id: str,
        k_samples: Optional[int] = None,
        horizon: Optional[int] = None,
        environment: Optional[EnvironmentMap] = None,
        target_exit_id: Optional[str] = None,
        prediction_interval: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, EvacuationConditioning]:
        """Perform reverse diffusion sampling and apply per-sample physical guidance.

        Guarantees:
        - Isolated random generator: Seed isolation without mutating global PyTorch/NumPy RNG.
        - Per-sample obstacle guidance: Guidance evaluated on each trajectory's coordinates.
        - Strict validation of horizon, prediction_interval, and k_samples.
        """
        K = k_samples if k_samples is not None else self.config.k_samples
        if type(K) is not int or K < 1:
            raise ValueError(f"k_samples must be an integer >= 1, got {K!r}")

        effective_horizon = horizon if horizon is not None else self.config.horizon
        if type(effective_horizon) is not int or effective_horizon < 1:
            raise TrajectoryValidationError(
                f"horizon must be a positive integer, got {effective_horizon!r}",
                field="horizon",
                agent_id=agent_id,
            )
        if effective_horizon > self.config.horizon:
            raise TrajectoryValidationError(
                f"Requested horizon ({effective_horizon}) exceeds configured maximum model horizon ({self.config.horizon}).",
                field="horizon",
                agent_id=agent_id,
            )

        interval = prediction_interval if prediction_interval is not None else self.config.prediction_interval
        if type(interval) not in (float, int) or not math.isfinite(interval) or interval <= 0.0:
            raise TrajectoryValidationError(
                f"prediction_interval must be a positive finite float, got {interval!r}",
                field="prediction_interval",
                agent_id=agent_id,
            )

        # Isolated RNG generator (zero global side-effects)
        rng = torch.Generator(device=self.config.device)
        if seed is not None:
            rng.manual_seed(int(seed))

        conditioning = EvacuationConditioning.from_snapshot_and_env(
            snapshot=snapshot,
            agent_id=agent_id,
            environment=environment,
            target_exit_id=target_exit_id,
        )

        # Use normalized conditioning vector for neural network
        cond_tensor = torch.tensor([conditioning.normalized_vector], dtype=torch.float32, device=self.config.device)
        cond_batch = cond_tensor.repeat(K, 1)

        # Baseline linear drift
        vx, vy = conditioning.velocity
        model_H = self.config.horizon
        baseline_drift = np.zeros((model_H, 2), dtype=np.float32)
        for step_idx in range(1, model_H + 1):
            baseline_drift[step_idx - 1, 0] = vx * (step_idx * interval)
            baseline_drift[step_idx - 1, 1] = vy * (step_idx * interval)

        # Reverse diffusion sampling
        self.network.eval()
        with torch.no_grad():
            x_t = torch.randn((K, model_H, 2), dtype=torch.float32, device=self.config.device, generator=rng)

            for n in reversed(range(self.config.diffusion_steps)):
                step_tensor = torch.full((K,), n, dtype=torch.int64, device=self.config.device)
                pred_noise = self.network(x_t, step_tensor, cond_batch)

                alpha = self.alphas[n]
                alpha_bar = self.alphas_cumprod[n]
                beta = self.betas[n]

                mean = (1.0 / math.sqrt(alpha)) * (
                    x_t - (beta / math.sqrt(1.0 - alpha_bar)) * pred_noise
                )

                if n > 0:
                    noise = torch.randn((K, model_H, 2), dtype=torch.float32, device=self.config.device, generator=rng)
                    var = self.posterior_variance[n]
                    sigma = math.sqrt(var)
                    x_t = mean + sigma * noise
                else:
                    x_t = mean

        raw_displacements = x_t.cpu().numpy()  # (K, model_H, 2)
        ref_x, ref_y = conditioning.reference_position

        # Allocate output array with requested effective horizon
        absolute_trajectories = np.zeros((K, effective_horizon, 2), dtype=np.float32)

        for k_idx in range(K):
            # Compute physical guidance INDEPENDENTLY for each sample trajectory
            sample_raw_disp = raw_displacements[k_idx, :effective_horizon]
            guidance_k = self._compute_per_sample_guidance(
                sample_displacements=sample_raw_disp,
                conditioning=conditioning,
                environment=environment,
                effective_horizon=effective_horizon,
            )

            # Combine baseline drift, diffusion perturbations, and sample-specific guidance
            drift_k = baseline_drift[:effective_horizon]
            disp_k = drift_k + sample_raw_disp * 0.25 + guidance_k * 0.15

            abs_x = ref_x + disp_k[:, 0]
            abs_y = ref_y + disp_k[:, 1]

            # Physical room boundary clamping (distinct from obstacle repulsion)
            if environment:
                r_ped = 0.3
                abs_x = np.clip(abs_x, r_ped, environment.width - r_ped)
                abs_y = np.clip(abs_y, r_ped, environment.height - r_ped)

            absolute_trajectories[k_idx, :, 0] = abs_x
            absolute_trajectories[k_idx, :, 1] = abs_y

        return absolute_trajectories, conditioning

    def predict_samples(
        self,
        snapshot: TrajectorySnapshot,
        agent_id: str,
        k_samples: Optional[int] = None,
        horizon: Optional[int] = None,
        environment: Optional[EnvironmentMap] = None,
        target_exit_id: Optional[str] = None,
        prediction_interval: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> MultiTrajectoryPrediction:
        """Generate K stochastic trajectory predictions and return MultiTrajectoryPrediction with validity audit."""
        K = k_samples if k_samples is not None else self.config.k_samples
        interval = prediction_interval if prediction_interval is not None else self.config.prediction_interval
        effective_horizon = horizon if horizon is not None else self.config.horizon
        source_ts = snapshot.current_timestamp
        source_step = snapshot.current_step

        trajs, conditioning = self.sample_trajectories(
            snapshot=snapshot,
            agent_id=agent_id,
            k_samples=K,
            horizon=effective_horizon,
            environment=environment,
            target_exit_id=target_exit_id,
            prediction_interval=interval,
            seed=seed,
        )

        sample_predictions: List[TrajectoryPrediction] = []
        for k_idx in range(K):
            preds: List[PredictedPosition] = []
            for h_idx in range(effective_horizon):
                t_fut = source_ts + (h_idx + 1) * interval
                x_val = float(trajs[k_idx, h_idx, 0])
                y_val = float(trajs[k_idx, h_idx, 1])
                preds.append(PredictedPosition(timestamp=t_fut, x=x_val, y=y_val))
            sample_predictions.append(
                TrajectoryPrediction(
                    agent_id=agent_id,
                    source_step_index=source_step,
                    source_timestamp=source_ts,
                    predictions=tuple(preds),
                )
            )

        # Mean trajectory: Point-wise arithmetic mean of all K paths
        mean_coords = np.mean(trajs, axis=0)  # (effective_horizon, 2)
        mean_preds: List[PredictedPosition] = []
        for h_idx in range(effective_horizon):
            t_fut = source_ts + (h_idx + 1) * interval
            mean_preds.append(
                PredictedPosition(
                    timestamp=t_fut,
                    x=float(mean_coords[h_idx, 0]),
                    y=float(mean_coords[h_idx, 1]),
                )
            )
        mean_trajectory = TrajectoryPrediction(
            agent_id=agent_id,
            source_step_index=source_step,
            source_timestamp=source_ts,
            predictions=tuple(mean_preds),
        )

        # Sample diversity: average pairwise Euclidean distance
        if K > 1:
            pairwise_dists = []
            for i in range(K):
                for j in range(i + 1, K):
                    diff = trajs[i] - trajs[j]
                    step_dists = np.hypot(diff[:, 0], diff[:, 1])
                    pairwise_dists.append(float(np.mean(step_dists)))
            diversity = float(np.mean(pairwise_dists)) if pairwise_dists else 0.0
        else:
            diversity = 0.0

        # Physical validity audit: Point walkability vs Segment collision detection
        total_pts = K * effective_horizon
        walkable_pts = 0
        total_segments = K * effective_horizon
        collision_free_segments = 0

        ref_pos = conditioning.reference_position

        for k_idx in range(K):
            # 1. Point walkability
            for h_idx in range(effective_horizon):
                px = float(trajs[k_idx, h_idx, 0])
                py = float(trajs[k_idx, h_idx, 1])
                if environment:
                    if environment.is_walkable(px, py, radius=0.1):
                        walkable_pts += 1
                else:
                    if math.isfinite(px) and math.isfinite(py):
                        walkable_pts += 1

            # 2. Segment collision detection
            pts_sequence = [ref_pos] + [(float(trajs[k_idx, h, 0]), float(trajs[k_idx, h, 1])) for h in range(effective_horizon)]
            for s_idx in range(len(pts_sequence) - 1):
                p_start = pts_sequence[s_idx]
                p_end = pts_sequence[s_idx + 1]

                collides = False
                if environment and environment.obstacles:
                    for obs in environment.obstacles:
                        if line_segment_intersects_box(p_start, p_end, obs.bounds):
                            collides = True
                            break

                if not collides:
                    collision_free_segments += 1

        point_validity = walkable_pts / total_pts if total_pts > 0 else 1.0
        segment_validity = collision_free_segments / total_segments if total_segments > 0 else 1.0

        return MultiTrajectoryPrediction(
            agent_id=agent_id,
            source_step_index=source_step,
            source_timestamp=source_ts,
            samples=tuple(sample_predictions),
            mean_prediction=mean_trajectory,
            sample_diversity=diversity,
            point_walkability=point_validity,
            segment_collision_free=segment_validity,
            walkable_validity=point_validity,
            is_trained_model=self._is_trained,
            sample_count=K,
        )

    def predict(
        self,
        snapshot: TrajectorySnapshot,
        agent_id: str,
        horizon: Optional[int] = None,
        prediction_interval: Optional[float] = None,
        environment: Optional[EnvironmentMap] = None,
        target_exit_id: Optional[str] = None,
        k_samples: int = 1,
        seed: Optional[int] = None,
    ) -> TrajectoryPrediction:
        """Standard single-prediction interface compatible with PredictionEvaluator.

        Honors the horizon argument consistently: returns a TrajectoryPrediction
        matching requested horizon length.
        """
        multi = self.predict_samples(
            snapshot=snapshot,
            agent_id=agent_id,
            k_samples=k_samples,
            horizon=horizon,
            environment=environment,
            target_exit_id=target_exit_id,
            prediction_interval=prediction_interval,
            seed=seed,
        )
        return multi.mean_prediction

    def train_step(
        self,
        batch_trajectories: torch.Tensor,
        batch_conditionings: torch.Tensor,
        optimizer: optim.Optimizer,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> float:
        """Execute a single diffusion model training optimization step with loss masking.

        Args:
            batch_trajectories: Target future displacements (Batch, Horizon, 2).
            batch_conditionings: Conditioning feature vectors (Batch, Conditioning_Dim).
            optimizer: PyTorch optimizer.
            loss_mask: Optional binary mask (Batch, Horizon, 1) indicating valid ground-truth points.

        Returns:
            Scalar training loss (MSE).
        """
        self.network.train()
        optimizer.zero_grad()

        B = batch_trajectories.shape[0]
        N = self.config.diffusion_steps

        # Sample random diffusion step per item using device
        steps = torch.randint(0, N, (B,), device=self.config.device)
        noise = torch.randn_like(batch_trajectories)

        # Forward diffuse
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[steps].reshape(B, 1, 1).to(self.config.device)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[steps].reshape(B, 1, 1).to(self.config.device)

        noisy_traj = sqrt_alpha_bar * batch_trajectories + sqrt_one_minus_alpha_bar * noise

        # Estimate noise
        pred_noise = self.network(noisy_traj, steps, batch_conditionings)

        # Masked loss calculation
        if loss_mask is not None:
            mask = loss_mask.to(self.config.device)
            sq_err = (pred_noise - noise) ** 2
            loss = (sq_err * mask).sum() / (mask.sum() * 2.0 + 1e-8)
        else:
            loss = nn.functional.mse_loss(pred_noise, noise)

        loss.backward()
        optimizer.step()

        # Update trained status
        self._is_trained = True
        return float(loss.item())

    def fit(
        self,
        samples: Sequence[EvaluationSample],
        epochs: int = 5,
        batch_size: int = 8,
        lr: Optional[float] = None,
        environment: Optional[EnvironmentMap] = None,
    ) -> List[float]:
        """Train the model on provided EvaluationSample instances with loss masking.

        Guarantees:
        - Rejects incomplete futures without silent point repetition.
        - Records skipped samples with exact diagnostic reasons.
        - Updates training provenance counters.
        """
        if not samples:
            return []

        rate = lr if lr is not None else self.config.learning_rate
        optimizer = optim.Adam(self.network.parameters(), lr=rate)
        H = self.config.horizon

        traj_data: List[List[Tuple[float, float]]] = []
        cond_data: List[Tuple[float, ...]] = []
        mask_data: List[List[Tuple[float, float]]] = []

        skipped_samples: List[Dict[str, str]] = []

        for sample in samples:
            snap = sample.to_snapshot()
            try:
                cond = EvacuationConditioning.from_snapshot_and_env(
                    snapshot=snap,
                    agent_id=sample.agent_id,
                    environment=environment,
                )
            except Exception as e:
                skipped_samples.append({"sample_id": getattr(sample, "agent_id", "unknown"), "reason": str(e)})
                continue

            ref_x, ref_y = cond.reference_position
            gt_future = sample.ground_truth_future

            if len(gt_future) < 1:
                skipped_samples.append({"sample_id": sample.agent_id, "reason": "Empty ground truth future"})
                continue

            disps: List[Tuple[float, float]] = []
            masks: List[Tuple[float, float]] = []

            for k in range(H):
                if k < len(gt_future):
                    obs = gt_future[k]
                    disps.append((obs.x - ref_x, obs.y - ref_y))
                    masks.append((1.0, 1.0))
                else:
                    # Masked padding: padded points are marked with 0.0 mask and never contribute to loss
                    disps.append((0.0, 0.0))
                    masks.append((0.0, 0.0))

            traj_data.append(disps)
            cond_data.append(cond.normalized_vector)
            mask_data.append(masks)

        if not traj_data:
            return []

        traj_tensor = torch.tensor(traj_data, dtype=torch.float32, device=self.config.device)
        cond_tensor = torch.tensor(cond_data, dtype=torch.float32, device=self.config.device)
        mask_tensor = torch.tensor(mask_data, dtype=torch.float32, device=self.config.device)
        N_samples = len(traj_data)

        # Isolated generator for permutation
        perm_rng = np.random.default_rng(42)

        loss_history: List[float] = []
        for epoch in range(epochs):
            indices = perm_rng.permutation(N_samples)
            epoch_losses = []
            for start in range(0, N_samples, batch_size):
                end = min(start + batch_size, N_samples)
                batch_idx = indices[start:end]
                b_traj = traj_tensor[batch_idx]
                b_cond = cond_tensor[batch_idx]
                b_mask = mask_tensor[batch_idx]
                loss = self.train_step(b_traj, b_cond, optimizer, loss_mask=b_mask)
                epoch_losses.append(loss)
            loss_history.append(float(np.mean(epoch_losses)))

        self._training_sample_count += N_samples
        self._training_epochs_completed += epochs
        self._is_trained = True

        return loss_history

    def evaluate_stochastic(
        self,
        samples: Sequence[EvaluationSample],
        environment: Optional[EnvironmentMap] = None,
        k_samples: int = 20,
        seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Compute complete stochastic evaluation metrics across a dataset of samples.

        Reports:
        - oracle_best_of_k_ade: Best sample ADE (oracle metric)
        - expected_mean_of_k_ade: Mean ADE across K samples
        - oracle_best_of_k_fde: Best sample FDE
        - sample_diversity: Average pairwise divergence
        - point_walkability: Fraction of predicted points inside walkable space
        - segment_collision_free: Fraction of path segments avoiding obstacles
        - sample audit: total_samples, evaluated_samples, skipped_samples, failed_samples, skip_reasons
        """
        if not samples:
            return {
                "total_samples": 0,
                "evaluated_samples": 0,
                "skipped_samples": 0,
                "failed_samples": 0,
                "skip_reasons": ["Empty dataset provided"],
                "oracle_best_of_k_ade": float("nan"),
                "expected_mean_of_k_ade": float("nan"),
                "best_of_k_ade": float("nan"),
                "mean_of_k_ade": float("nan"),
                "oracle_best_of_k_fde": float("nan"),
                "best_of_k_fde": float("nan"),
                "mean_diversity": float("nan"),
                "point_walkability": float("nan"),
                "segment_collision_free": float("nan"),
                "mean_walkable_validity": float("nan"),
                "is_trained_model": self._is_trained,
            }

        best_ades: List[float] = []
        mean_ades: List[float] = []
        best_fdes: List[float] = []
        diversities: List[float] = []
        point_vals: List[float] = []
        segment_vals: List[float] = []

        skip_reasons: List[str] = []
        evaluated_count = 0
        failed_count = 0

        for sample_idx, sample in enumerate(samples):
            snap = sample.to_snapshot()
            sample_seed = (seed + sample_idx) if seed is not None else None
            try:
                multi = self.predict_samples(
                    snapshot=snap,
                    agent_id=sample.agent_id,
                    k_samples=k_samples,
                    environment=environment,
                    seed=sample_seed,
                )
            except Exception as err:
                failed_count += 1
                skip_reasons.append(f"Sample '{getattr(sample, 'agent_id', 'unknown')}': prediction error: {err}")
                continue

            gt_pts = [(obs.x, obs.y) for obs in sample.ground_truth_future[:self.config.horizon]]
            if len(gt_pts) < 1:
                skip_reasons.append(f"Sample '{sample.agent_id}': ground truth future is empty")
                continue

            sample_ades = []
            sample_fdes = []
            for s_pred in multi.samples:
                pred_pts = [(p.x, p.y) for p in s_pred.predictions[:len(gt_pts)]]
                sample_ades.append(calculate_ade(gt_pts, pred_pts))
                sample_fdes.append(calculate_fde(gt_pts, pred_pts))

            best_ades.append(min(sample_ades))
            mean_ades.append(float(np.mean(sample_ades)))
            best_fdes.append(min(sample_fdes))
            diversities.append(multi.sample_diversity)
            point_vals.append(multi.point_walkability)
            segment_vals.append(multi.segment_collision_free)
            evaluated_count += 1

        b_ade = float(np.mean(best_ades)) if best_ades else float("nan")
        m_ade = float(np.mean(mean_ades)) if mean_ades else float("nan")
        b_fde = float(np.mean(best_fdes)) if best_fdes else float("nan")
        div = float(np.mean(diversities)) if diversities else float("nan")
        pt_val = float(np.mean(point_vals)) if point_vals else float("nan")
        seg_val = float(np.mean(segment_vals)) if segment_vals else float("nan")

        return {
            "total_samples": len(samples),
            "evaluated_samples": evaluated_count,
            "skipped_samples": len(samples) - evaluated_count - failed_count,
            "failed_samples": failed_count,
            "skip_reasons": skip_reasons,
            "oracle_best_of_k_ade": b_ade,
            "expected_mean_of_k_ade": m_ade,
            "best_of_k_ade": b_ade,
            "mean_of_k_ade": m_ade,
            "oracle_best_of_k_fde": b_fde,
            "best_of_k_fde": b_fde,
            "mean_diversity": div,
            "point_walkability": pt_val,
            "segment_collision_free": seg_val,
            "mean_walkable_validity": pt_val,
            "is_trained_model": self._is_trained,
        }


__all__ = [
    "EvacuationConditioning",
    "EvacuationMIDConfig",
    "EvacuationMIDNetwork",
    "EvacuationMIDPredictor",
    "MultiTrajectoryPrediction",
    "FEATURE_NORMALIZATION_SCALES",
    "line_segment_intersects_box",
]
