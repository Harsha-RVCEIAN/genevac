"""Multi-agent social interaction training corpus for Evacuation MID.

Generates physical multi-pedestrian evacuation training rollouts on dedicated
synthetic training geometries strictly disjoint from ``building_main.json``
and comparative benchmark scenes so evaluation remains completely unseen.

Phenomena covered (coupled with multi-agent neighbor histories):
1. ``unobstructed_cruise``: Unobstructed baseline motion along diverse headings.
2. ``following_slow_leader``: Longitudinal headway keeping and smooth braking behind slower leaders.
3. ``crowd_avoidance_headon``: Anticipatory lateral swerve and speed easing around opposing pedestrians.
4. ``bottleneck_congestion``: Speed compression and flow spacing within multi-agent crowd pockets.
5. ``corridor_merging``: Mutual speed yielding and lateral merge gap formation at stream junctions.
6. ``pedestrian_obstruction``: Deceleration and lateral bypass around stationary/stalled blockers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

from genevac.prediction.data_contract import TrajectoryObservation
from genevac.prediction.evaluation import EvaluationSample, SampleGenerator

TRAINING_SOURCE_NAME: str = "multi_agent_social_evacuation_rollouts"
TRAINING_INTERACTION_SCENARIOS: Tuple[str, ...] = (
    "unobstructed_cruise",
    "following_slow_leader",
    "crowd_avoidance_headon",
    "bottleneck_congestion",
    "corridor_merging",
    "pedestrian_obstruction",
)


@dataclass(frozen=True)
class _AgentInputSpec:
    agent_id: str
    x: float
    y: float
    gx: float
    gy: float
    speed: float
    is_ego: bool = False
    freeze: bool = False


@dataclass
class _RolloutAgentState:
    agent_id: str
    x: float
    y: float
    gx: float
    gy: float
    vx: float
    vy: float
    v_pref: float
    freeze: bool = False


def _rollout_social_interaction_scenario(
    scenario_name: str,
    agent_specs: Sequence[_AgentInputSpec],
    dt: float = 0.1,
    total_steps: int = 24,
    horizon: int = 12,
    history_length: int = 4,
    sample_stride: int = 2,
    max_samples: int = 16,
) -> List[EvaluationSample]:
    """Simulate continuous social-force pedestrian rollouts and extract windowed samples."""
    del scenario_name
    states: List[_RolloutAgentState] = []
    trajectories: Dict[str, List[TrajectoryObservation]] = {}
    ego_ids: List[str] = []

    for spec in agent_specs:
        dx = spec.gx - spec.x
        dy = spec.gy - spec.y
        norm = math.hypot(dx, dy)
        vx = (dx / norm) * spec.speed if norm > 1e-6 else 0.0
        vy = (dy / norm) * spec.speed if norm > 1e-6 else 0.0
        states.append(
            _RolloutAgentState(
                agent_id=spec.agent_id,
                x=spec.x,
                y=spec.y,
                gx=spec.gx,
                gy=spec.gy,
                vx=vx,
                vy=vy,
                v_pref=spec.speed,
                freeze=spec.freeze,
            )
        )
        trajectories[spec.agent_id] = [
            TrajectoryObservation(agent_id=spec.agent_id, step_index=0, timestamp=0.0, x=spec.x, y=spec.y)
        ]
        if spec.is_ego:
            ego_ids.append(spec.agent_id)

    for step in range(1, total_steps):
        ts = round(step * dt, 6)
        next_states: List[Tuple[float, float, float, float]] = []
        for i, st in enumerate(states):
            x, y, vx, vy, v_pref = st.x, st.y, st.vx, st.vy, st.v_pref
            if st.freeze or v_pref < 1e-6:
                next_states.append((x, y, 0.0, 0.0))
                continue

            dx_g, dy_g = st.gx - x, st.gy - y
            dist_g = math.hypot(dx_g, dy_g)
            if dist_g < 1e-4:
                next_states.append((x, y, 0.0, 0.0))
                continue

            ux, uy = dx_g / dist_g, dy_g / dist_g
            target_speed = v_pref
            lat_push_x, lat_push_y = 0.0, 0.0

            # Compute physical social response from current instantaneous neighbors
            for j, other in enumerate(states):
                if i == j:
                    continue
                rx, ry = other.x - x, other.y - y
                dist = math.hypot(rx, ry)
                if dist > 4.2:
                    continue

                fwd = rx * ux + ry * uy
                lat = -rx * uy + ry * ux
                ospeed = math.hypot(other.vx, other.vy)
                align = ((ux * other.vx + uy * other.vy) / (ospeed + 1e-6)) if ospeed > 0.05 else 0.0

                # Following slower leader ahead in lane
                if 0.2 < fwd < 2.6 and abs(lat) < 0.75 and align > 0.35:
                    gap_factor = max(0.25, min(1.0, (fwd - 0.65) / 1.85))
                    cap_speed = ospeed + 0.35 * (v_pref - ospeed) * gap_factor
                    target_speed = min(target_speed, max(0.18, cap_speed))

                # Opposing / crossing / obstructed pedestrian anticipation
                elif fwd > -0.15 and dist < 3.6:
                    rvx, rvy = other.vx - vx, other.vy - vy
                    rv_sq = rvx * rvx + rvy * rvy
                    t_cpa = max(0.0, min(2.2, -(rx * rvx + ry * rvy) / rv_sq)) if rv_sq > 1e-4 else 0.0
                    d_cpa = math.hypot(rx + rvx * t_cpa, ry + rvy * t_cpa)

                    if d_cpa < 1.35 or (fwd > 0.0 and abs(lat) < 0.85 and ospeed < 0.25):
                        weight = (3.6 - dist) / 3.6
                        dodge_sign = -1.0 if lat >= 0.0 else 1.0
                        lat_push_x += dodge_sign * (-uy) * (0.95 * weight)
                        lat_push_y += dodge_sign * ux * (0.95 * weight)
                        if ospeed < 0.25 and abs(lat) < 0.55 and fwd < 1.8:
                            target_speed = min(target_speed, max(0.28, v_pref * 0.45))
                        elif align < -0.25 and dist < 2.2:
                            target_speed = min(target_speed, max(0.45, v_pref * 0.72))

            # Congestion damping when multiple neighbors populate front sector
            front_crowd = sum(
                1
                for j, other in enumerate(states)
                if i != j
                and 0.0 < ((other.x - x) * ux + (other.y - y) * uy) < 3.2
                and math.hypot(other.x - x, other.y - y) < 3.2
            )
            if front_crowd >= 2:
                target_speed = min(target_speed, v_pref * max(0.38, 1.0 - 0.18 * front_crowd))

            des_vx = ux * target_speed + lat_push_x
            des_vy = uy * target_speed + lat_push_y
            des_norm = math.hypot(des_vx, des_vy)
            if des_norm > v_pref * 1.1:
                scale = (v_pref * 1.1) / des_norm
                des_vx *= scale
                des_vy *= scale

            # Smooth velocity relaxation (human relaxation time ~ 0.35s)
            alpha = min(1.0, dt / 0.35)
            nvx = vx + alpha * (des_vx - vx)
            nvy = vy + alpha * (des_vy - vy)
            nx = x + nvx * dt
            ny = y + nvy * dt
            next_states.append((nx, ny, nvx, nvy))

        for i, st in enumerate(states):
            nx, ny, nvx, nvy = next_states[i]
            st.x, st.y, st.vx, st.vy = nx, ny, nvx, nvy
            trajectories[st.agent_id].append(
                TrajectoryObservation(agent_id=st.agent_id, step_index=step, timestamp=ts, x=nx, y=ny)
            )

    all_trajs: Dict[str, Sequence[TrajectoryObservation]] = {
        aid: tuple(obs) for aid, obs in trajectories.items()
    }
    generator = SampleGenerator(
        history_length=history_length,
        prediction_horizon=horizon,
        stride=sample_stride,
    )
    raw_samples = generator.generate_samples(all_trajs, include_neighbors=True)
    ego_set = set(ego_ids)
    filtered = [s for s in raw_samples if s.agent_id in ego_set]
    return filtered[:max_samples]


def generate_social_interaction_training_samples(
    horizon: int = 12,
    history_length: int = 4,
    dt: float = 0.1,
) -> Tuple[List[EvaluationSample], Tuple[str, ...]]:
    """Generate deterministic multi-agent training samples covering all social interaction types.

    The returned ``EvaluationSample`` instances use dedicated training corridors
    disjoint from ``building_main.json`` (no evaluation leakage) and include full
    synchronized historical neighbor trajectories in ``neighbor_trajectories``.
    """
    samples: List[EvaluationSample] = []

    # 1. Unobstructed cruise across multiple headings (no directional bias)
    cruise_headings = [
        ("u_east", 2.0, 10.0, 22.0, 10.0, 1.35),
        ("u_west", 22.0, 14.0, 2.0, 14.0, 1.30),
        ("u_diag_up", 4.0, 4.0, 20.0, 18.0, 1.40),
        ("u_diag_dn", 4.0, 18.0, 20.0, 4.0, 1.25),
    ]
    for cid, sx, sy, gx, gy, spd in cruise_headings:
        samples.extend(
            _rollout_social_interaction_scenario(
                scenario_name="unobstructed_cruise",
                agent_specs=[_AgentInputSpec(cid, sx, sy, gx, gy, spd, is_ego=True)],
                dt=dt,
                total_steps=20,
                horizon=horizon,
                history_length=history_length,
                sample_stride=2,
                max_samples=3,
            )
        )

    # 2. Following slow leader (Eastbound & Westbound variants)
    samples.extend(
        _rollout_social_interaction_scenario(
            scenario_name="following_slow_leader",
            agent_specs=[
                _AgentInputSpec("f_ego_1", 3.0, 8.0, 22.0, 8.0, 1.45, is_ego=True),
                _AgentInputSpec("f_lead_1", 4.8, 8.0, 22.0, 8.0, 0.62, is_ego=False),
            ],
            dt=dt,
            total_steps=26,
            horizon=horizon,
            history_length=history_length,
            sample_stride=2,
            max_samples=6,
        )
    )
    samples.extend(
        _rollout_social_interaction_scenario(
            scenario_name="following_slow_leader",
            agent_specs=[
                _AgentInputSpec("f_ego_2", 18.0, 16.0, 2.0, 16.0, 1.38, is_ego=True),
                _AgentInputSpec("f_lead_2", 16.2, 16.05, 2.0, 16.0, 0.55, is_ego=False),
            ],
            dt=dt,
            total_steps=24,
            horizon=horizon,
            history_length=history_length,
            sample_stride=2,
            max_samples=5,
        )
    )

    # 3. Head-on and diagonal crowd avoidance
    samples.extend(
        _rollout_social_interaction_scenario(
            scenario_name="crowd_avoidance_headon",
            agent_specs=[
                _AgentInputSpec("a_ego_1", 3.0, 11.0, 21.0, 11.0, 1.35, is_ego=True),
                _AgentInputSpec("a_opp_1", 8.0, 11.05, 1.0, 11.05, 1.25, is_ego=True),
                _AgentInputSpec("a_flank_1", 6.5, 12.2, 1.5, 12.2, 1.15, is_ego=False),
            ],
            dt=dt,
            total_steps=26,
            horizon=horizon,
            history_length=history_length,
            sample_stride=2,
            max_samples=10,
        )
    )

    # 4. Bottleneck congestion (dense front pocket of 4 slower movers)
    samples.extend(
        _rollout_social_interaction_scenario(
            scenario_name="bottleneck_congestion",
            agent_specs=[
                _AgentInputSpec("c_ego_1", 2.5, 9.0, 20.0, 9.0, 1.40, is_ego=True),
                _AgentInputSpec("c_nbr_1", 4.2, 9.0, 20.0, 9.0, 0.55, is_ego=False),
                _AgentInputSpec("c_nbr_2", 4.6, 9.8, 20.0, 9.3, 0.60, is_ego=False),
                _AgentInputSpec("c_nbr_3", 4.5, 8.2, 20.0, 8.8, 0.58, is_ego=False),
                _AgentInputSpec("c_nbr_4", 5.6, 9.1, 20.0, 9.0, 0.50, is_ego=False),
            ],
            dt=dt,
            total_steps=26,
            horizon=horizon,
            history_length=history_length,
            sample_stride=2,
            max_samples=8,
        )
    )

    # 5. Corridor merging (~55 deg diagonal convergence)
    samples.extend(
        _rollout_social_interaction_scenario(
            scenario_name="corridor_merging",
            agent_specs=[
                _AgentInputSpec("m_ego_1", 3.0, 7.0, 21.0, 7.0, 1.32, is_ego=True),
                _AgentInputSpec("m_cross_1", 4.5, 10.2, 18.0, 6.5, 1.28, is_ego=True),
                _AgentInputSpec("m_cross_2", 3.6, 11.4, 18.0, 6.8, 1.22, is_ego=False),
            ],
            dt=dt,
            total_steps=26,
            horizon=horizon,
            history_length=history_length,
            sample_stride=2,
            max_samples=10,
        )
    )

    # 6. Pedestrian obstruction (stationary / stalled pedestrian blocking forward lane)
    samples.extend(
        _rollout_social_interaction_scenario(
            scenario_name="pedestrian_obstruction",
            agent_specs=[
                _AgentInputSpec("o_ego_1", 3.0, 6.0, 20.0, 6.0, 1.35, is_ego=True),
                _AgentInputSpec("o_block_1", 5.2, 6.02, 5.2, 6.02, 0.0, is_ego=False, freeze=True),
            ],
            dt=dt,
            total_steps=24,
            horizon=horizon,
            history_length=history_length,
            sample_stride=2,
            max_samples=6,
        )
    )

    return samples, TRAINING_INTERACTION_SCENARIOS


__all__ = [
    "TRAINING_INTERACTION_SCENARIOS",
    "TRAINING_SOURCE_NAME",
    "generate_social_interaction_training_samples",
]
