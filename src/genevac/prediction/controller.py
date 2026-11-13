"""MID-Assisted Decision Adapter for GenEvac-Sentinel Evacuation Scenarios.

Architecture:
    Actual SimulationEngine
            |
            v
    Observation Adapter (SimulationTrajectoryAdapter)
            |
            v
    MID Predictor (EvacuationMIDPredictor)
            |
            v
    MID Trajectory Evaluation & Feasibility Filter
            |
            v
    MIDAssistedDecisionAdapter
            |
            v
    Validated Movement / Route Guidance Command (guide_waypoint + topological tail)
            |
            v
    Existing SimulationEngine public interface (target_exit_id, waypoints, status)

Selection Rule & Safety Guarantees:
1. Hard Feasibility Filter:
   Every candidate MID trajectory is tested against the authoritative building
   geometry using Pavan's continuous collision and walkability checks:
   - Point walkability & clearance (`is_position_in_walkable_space`)
   - Continuous segment collision check from current position to guide waypoint (`check_collision`)
   - Continuous segment collision check from guide waypoint to next topological goal (`check_collision`)
   - Strict positive forward progress toward the topological goal / exit (prevents oscillation)
   - Pedestrian-to-pedestrian overlap check (`check_pedestrian_overlap`)
   Every candidate must also remain walkable and collision-free across its complete
   predicted horizon. Missing authoritative geometry callbacks fail closed.
   Any candidate failing any of these checks is marked invalid (`is_valid=False`) and rejected.
2. Lexicographic Ranking of Feasible Candidates:
   Valid candidates are ranked deterministically without arbitrary weighted formulas:
   (a) Full-horizon sample collision-free status (`full_sample_collision_free`)
   (b) Local pedestrian congestion count within 3.0m of the guide waypoint (`local_congestion_count`)
   (c) Obstacle clearance in metres (binned to 0.5m increments up to 2.0m)
   (d) Total remaining polyline route distance to the target exit (`route_distance_m`)
3. Fallback Guarantee:
   If zero candidates are feasible, the controller records a fallback (`fallback_to_original=True`)
   and preserves the original simulation engine route unchanged.
4. No Teleportation or Physics Bypass:
   The controller never overwrites `ped.position` or `ped.velocity`. The authoritative
   `SimulationEngine.step()` executes all movement, speed limits, and collision resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from genevac.prediction.mid import MultiTrajectoryPrediction


@dataclass(frozen=True)
class CandidateEvaluation:
    """Detailed feasibility and physical evaluation of a single MID trajectory sample."""
    sample_idx: int
    is_valid: bool
    rejection_reason: Optional[str]
    guide_waypoint: Optional[Any]
    guide_step_idx: int
    target_exit_id: Optional[str]
    full_sample_collision_free: bool
    local_congestion_count: int
    min_obstacle_clearance_m: float
    route_distance_m: float

    @property
    def candidate_id(self) -> str:
        """Stable human-readable identifier linked 1:1 with sample_idx."""
        return f"C{self.sample_idx}"


@dataclass(frozen=True)
class MIDDecisionRecommendation:
    """Immutable recommendation and audit record from the MID-assisted decision adapter."""
    agent_id: str
    target_exit_id: Optional[str]
    proposed_waypoints: Tuple[Any, ...]
    guide_waypoint: Optional[Any]
    evaluated_samples_count: int
    valid_candidates_count: int
    invalid_candidates_count: int
    selected_sample_idx: int
    candidates: Tuple[CandidateEvaluation, ...]
    selection_reason: str
    is_accepted: bool
    guidance_applied: bool
    fallback_to_original: bool
    rejection_reason: Optional[str] = None
    step_index: int = 0
    timestamp: float = 0.0
    control_compute_time_s: float = 0.0

    @property
    def selected_candidate_id(self) -> Optional[str]:
        """Stable ID (C{idx}) of the controller-selected candidate, or None if unselected."""
        if not self.is_accepted or self.selected_sample_idx < 0:
            return None
        return f"C{self.selected_sample_idx}"

    @property
    def rejected_candidates(self) -> Tuple[CandidateEvaluation, ...]:
        """Tuple of rejected CandidateEvaluation records preserving sample order."""
        return tuple(c for c in self.candidates if not c.is_valid)

    @property
    def accepted_candidate(self) -> Optional[CandidateEvaluation]:
        """Return the CandidateEvaluation matching selected_sample_idx, if accepted."""
        if not self.is_accepted or self.selected_sample_idx < 0:
            return None
        for cand in self.candidates:
            if cand.sample_idx == self.selected_sample_idx:
                return cand
        return None

    def rejection_summary(self) -> Dict[str, str]:
        """Return {candidate_id: rejection_reason} for every rejected candidate."""
        return {
            c.candidate_id: (c.rejection_reason or "rejected")
            for c in self.candidates
            if not c.is_valid
        }


@dataclass(frozen=True)
class SafetyInterventionEvent:
    """Explicit structured record of a collision-avoidance or deadlock intervention."""
    agent_id: str
    step_index: int
    timestamp: float
    reason: str
    proposed_target: Tuple[float, float]
    safe_target: Tuple[float, float]
    hazard_location: Optional[Tuple[float, float]] = None
    conflict_agent_id: Optional[str] = None
    proposed_speed_mps: float = 0.0
    safe_speed_mps: float = 0.0
    min_clearance_m: float = 0.0


@dataclass(frozen=True)
class SafetyCheckResult:
    """Output of evaluating a proposed step against the explicit collision avoidance layer."""
    is_proposed_safe: bool
    safe_target: Tuple[float, float]
    safe_speed_mps: float
    intervention_applied: bool
    intervention_reason: Optional[str]
    hazard_points: Tuple[Tuple[float, float], ...]
    rejected_targets: Tuple[Tuple[float, float], ...]
    conflict_agent_id: Optional[str]
    min_clearance_m: float
    near_collision_detected: bool
    deadlock_recovery_applied: bool


def _continuous_segment_min_distance(
    ax0: float,
    ay0: float,
    ax1: float,
    ay1: float,
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
) -> Tuple[float, float, Tuple[float, float]]:
    """Compute minimum Euclidean distance between two linearly moving disks over t in [0, 1].

    Returns:
        (min_distance_m, t_closest, midpoint_at_closest)
    """
    rx0 = bx0 - ax0
    ry0 = by0 - ay0
    drx = (bx1 - bx0) - (ax1 - ax0)
    dry = (by1 - by0) - (ay1 - ay0)
    dr_sq = drx * drx + dry * dry
    if dr_sq > 1e-12:
        t_star = max(0.0, min(1.0, -(rx0 * drx + ry0 * dry) / dr_sq))
    else:
        t_star = 0.0
    cx = rx0 + drx * t_star
    cy = ry0 + dry * t_star
    min_dist = math.hypot(cx, cy)
    pa_x = ax0 + (ax1 - ax0) * t_star
    pa_y = ay0 + (ay1 - ay0) * t_star
    pb_x = bx0 + (bx1 - bx0) * t_star
    pb_y = by0 + (by1 - by0) * t_star
    return min_dist, t_star, (0.5 * (pa_x + pb_x), 0.5 * (pa_y + pb_y))


class MIDAssistedDecisionAdapter:
    """Evaluates MID trajectory candidates and safely guides pedestrian movement via engine waypoints."""

    def __init__(self, building: Any, min_decision_interval: float = 0.4) -> None:
        self.building = building
        self.min_decision_interval = float(min_decision_interval)
        self.last_decision_time: Dict[str, float] = {}
        self._active_mid_waypoint: Dict[str, Any] = {}
        self.latest_recommendations_by_agent: Dict[str, MIDDecisionRecommendation] = {}
        self.recommendation_history: List[MIDDecisionRecommendation] = []
        self.total_evaluations: int = 0
        self.accepted_interventions: int = 0
        self.invalid_candidates_total: int = 0
        self.fallback_count: int = 0

    def reset(self) -> None:
        """Clear adapter state, active guide waypoints, and decision history."""
        self.last_decision_time.clear()
        self._active_mid_waypoint.clear()
        self.latest_recommendations_by_agent.clear()
        self.recommendation_history.clear()
        self.total_evaluations = 0
        self.accepted_interventions = 0
        self.invalid_candidates_total = 0
        self.fallback_count = 0

    def get_base_waypoints(self, agent_id: str, current_pedestrian: Any) -> List[Any]:
        """Return the pedestrian's canonical topological waypoints, stripping any unpopped MID guide point."""
        current_wps = list(getattr(current_pedestrian, "waypoints", []))
        active_gw = self._active_mid_waypoint.get(agent_id)
        if active_gw is not None and current_wps:
            first_wp = current_wps[0]
            if (
                abs(first_wp.x - active_gw.x) < 1e-5
                and abs(first_wp.y - active_gw.y) < 1e-5
                and len(current_wps) > 1
            ):
                return current_wps[1:]
        return current_wps

    def _compute_min_obstacle_clearance(self, pt_x: float, pt_y: float) -> float:
        """Compute minimum Euclidean distance from (pt_x, pt_y) to any static obstacle in building."""
        obstacles = getattr(self.building, "obstacles", ())
        if not obstacles:
            return 50.0
        min_d = float("inf")
        for obs in obstacles:
            b = obs.bounds
            dx = max(b.left - pt_x, 0.0, pt_x - b.right)
            dy = max(b.top - pt_y, 0.0, pt_y - b.bottom)
            d = math.hypot(dx, dy)
            if d < min_d:
                min_d = d
        return min_d

    @staticmethod
    def _polyline_length(points: Sequence[Any]) -> float:
        if len(points) < 2:
            return 0.0
        total = 0.0
        for i in range(len(points) - 1):
            total += math.hypot(points[i + 1].x - points[i].x, points[i + 1].y - points[i].y)
        return total

    def evaluate_and_recommend(
        self,
        agent_id: str,
        current_pedestrian: Any,
        mid_prediction: MultiTrajectoryPrediction,
        open_exit_ids: Sequence[str],
        current_time: float,
        step_index: int,
        generate_route_fn: Callable[..., List[Any]],
        is_walkable_fn: Optional[Callable[..., bool]] = None,
        check_collision_fn: Optional[Callable[..., bool]] = None,
        check_overlap_fn: Optional[Callable[..., bool]] = None,
        other_pedestrians: Optional[Sequence[Any]] = None,
    ) -> MIDDecisionRecommendation:
        """Evaluate K stochastic MID samples and produce a validated movement guidance recommendation."""
        t_start = time.perf_counter()
        self.total_evaluations += 1

        base_wps = self.get_base_waypoints(agent_id, current_pedestrian)
        curr_pos = current_pedestrian.position
        point_cls = type(curr_pos)
        ped_radius = float(getattr(current_pedestrian, "radius", 0.3))
        curr_exit_id = getattr(current_pedestrian, "target_exit_id", None)

        # 1. Check cadence rate-limiting
        last_time = self.last_decision_time.get(agent_id, -100.0)
        if current_time - last_time < self.min_decision_interval - 1e-6:
            rec = MIDDecisionRecommendation(
                agent_id=agent_id,
                target_exit_id=curr_exit_id or "",
                proposed_waypoints=tuple(getattr(current_pedestrian, "waypoints", [])),
                guide_waypoint=self._active_mid_waypoint.get(agent_id),
                evaluated_samples_count=len(mid_prediction.samples),
                valid_candidates_count=0,
                invalid_candidates_count=0,
                selected_sample_idx=-1,
                candidates=(),
                selection_reason="Rate-limited cadence guard active; retaining current guidance",
                is_accepted=False,
                guidance_applied=False,
                fallback_to_original=True,
                rejection_reason="cadence_guard",
                step_index=step_index,
                timestamp=current_time,
                control_compute_time_s=time.perf_counter() - t_start,
            )
            self.fallback_count += 1
            self.latest_recommendations_by_agent[agent_id] = rec
            return rec

        # 2. Check open exits availability
        open_exits_map: Dict[str, Tuple[float, float]] = {}
        for ex in self.building.exits:
            if ex.id in open_exit_ids and ex.is_open:
                mid_x = (ex.segment.start.x + ex.segment.end.x) / 2.0
                mid_y = (ex.segment.start.y + ex.segment.end.y) / 2.0
                open_exits_map[ex.id] = (mid_x, mid_y)

        if not open_exits_map:
            rec = MIDDecisionRecommendation(
                agent_id=agent_id,
                target_exit_id=curr_exit_id or "",
                proposed_waypoints=tuple(base_wps),
                guide_waypoint=None,
                evaluated_samples_count=len(mid_prediction.samples),
                valid_candidates_count=0,
                invalid_candidates_count=len(mid_prediction.samples),
                selected_sample_idx=-1,
                candidates=(),
                selection_reason="No open exits available in building; fallback to baseline",
                is_accepted=False,
                guidance_applied=False,
                fallback_to_original=True,
                rejection_reason="no_open_exits",
                step_index=step_index,
                timestamp=current_time,
                control_compute_time_s=time.perf_counter() - t_start,
            )
            self.fallback_count += 1
            self.invalid_candidates_total += len(mid_prediction.samples)
            self.latest_recommendations_by_agent[agent_id] = rec
            return rec

        # 3. Determine active target exit & topological base route
        # If current target exit is closed or pedestrian has no route, select nearest reachable open exit
        needs_exit_reroute = (
            curr_exit_id is None
            or curr_exit_id not in open_exits_map
            or not base_wps
            or str(getattr(current_pedestrian.status, "value", current_pedestrian.status)).lower() == "blocked"
        )

        start_area = getattr(current_pedestrian, "current_area_id", None)
        if needs_exit_reroute:
            if not start_area:
                rec = MIDDecisionRecommendation(
                    agent_id=agent_id,
                    target_exit_id=curr_exit_id or "",
                    proposed_waypoints=tuple(base_wps),
                    guide_waypoint=None,
                    evaluated_samples_count=len(mid_prediction.samples),
                    valid_candidates_count=0,
                    invalid_candidates_count=len(mid_prediction.samples),
                    selected_sample_idx=-1,
                    candidates=(),
                    selection_reason="Undefined current_area_id for reroute; fallback",
                    is_accepted=False,
                    guidance_applied=False,
                    fallback_to_original=True,
                    rejection_reason="undefined_area",
                    step_index=step_index,
                    timestamp=current_time,
                    control_compute_time_s=time.perf_counter() - t_start,
                )
                self.fallback_count += 1
                self.invalid_candidates_total += len(mid_prediction.samples)
                self.latest_recommendations_by_agent[agent_id] = rec
                return rec

            # Find which open exit the MID samples approach most closely that has a valid route
            best_reroute_exit: Optional[str] = None
            best_reroute_wps: List[Any] = []
            best_reroute_dist = float("inf")

            for s_idx, sample in enumerate(mid_prediction.samples):
                if not sample.predictions:
                    continue
                term_pt = sample.predictions[-1]
                for ex_id, (ex_x, ex_y) in open_exits_map.items():
                    d_ex = math.hypot(term_pt.x - ex_x, term_pt.y - ex_y)
                    if d_ex < best_reroute_dist:
                        try:
                            cand_route = generate_route_fn(
                                building=self.building,
                                start_pos=curr_pos,
                                start_area_id=start_area,
                                target_exit_id=ex_id,
                                radius=ped_radius,
                            )
                            if cand_route:
                                best_reroute_dist = d_ex
                                best_reroute_exit = ex_id
                                best_reroute_wps = list(cand_route)
                        except Exception:
                            continue

            if best_reroute_exit is None or not best_reroute_wps:
                rec = MIDDecisionRecommendation(
                    agent_id=agent_id,
                    target_exit_id=curr_exit_id or "",
                    proposed_waypoints=tuple(base_wps),
                    guide_waypoint=None,
                    evaluated_samples_count=len(mid_prediction.samples),
                    valid_candidates_count=0,
                    invalid_candidates_count=len(mid_prediction.samples),
                    selected_sample_idx=-1,
                    candidates=(),
                    selection_reason="No reachable open exit route found; fallback",
                    is_accepted=False,
                    guidance_applied=False,
                    fallback_to_original=True,
                    rejection_reason="no_reachable_exit",
                    step_index=step_index,
                    timestamp=current_time,
                    control_compute_time_s=time.perf_counter() - t_start,
                )
                self.fallback_count += 1
                self.invalid_candidates_total += len(mid_prediction.samples)
                self.latest_recommendations_by_agent[agent_id] = rec
                return rec

            target_exit_id = best_reroute_exit
            base_wps = best_reroute_wps
        else:
            target_exit_id = curr_exit_id

        target_exit_obj = self.building.get_exit(target_exit_id)

        # Advance past any canonical waypoint already reached or safely bypassed within 0.45m
        while len(base_wps) > 1:
            d0 = math.hypot(base_wps[0].x - curr_pos.x, base_wps[0].y - curr_pos.y)
            if d0 < 0.45 and (
                check_collision_fn is None
                or not check_collision_fn(curr_pos, base_wps[1], ped_radius, self.building, target_exit_obj)
            ):
                base_wps.pop(0)
            else:
                break

        goal_wp = base_wps[0]
        d_curr_to_goal = math.hypot(goal_wp.x - curr_pos.x, goal_wp.y - curr_pos.y)

        other_active = [
            p for p in (other_pedestrians or ())
            if getattr(p, "id", None) != agent_id
            and str(getattr(p.status, "value", p.status)).lower() == "active"
        ]

        # 4. Evaluate each of the K MID trajectory samples
        candidate_evals: List[CandidateEvaluation] = []
        horizon_probe_indices = (7, 5, 3, 9, 11, 2)

        for s_idx, sample in enumerate(mid_prediction.samples):
            preds = sample.predictions
            if not preds:
                candidate_evals.append(
                    CandidateEvaluation(
                        sample_idx=s_idx,
                        is_valid=False,
                        rejection_reason="empty_predictions",
                        guide_waypoint=None,
                        guide_step_idx=-1,
                        target_exit_id=target_exit_id,
                        full_sample_collision_free=False,
                        local_congestion_count=0,
                        min_obstacle_clearance_m=0.0,
                        route_distance_m=float("inf"),
                    )
                )
                continue

            # Check non-finite coordinates across sample
            if not all(math.isfinite(p.x) and math.isfinite(p.y) for p in preds):
                candidate_evals.append(
                    CandidateEvaluation(
                        sample_idx=s_idx,
                        is_valid=False,
                        rejection_reason="non_finite_coordinates",
                        guide_waypoint=None,
                        guide_step_idx=-1,
                        target_exit_id=target_exit_id,
                        full_sample_collision_free=False,
                        local_congestion_count=0,
                        min_obstacle_clearance_m=0.0,
                        route_distance_m=float("inf"),
                    )
                )
                continue

            # Fail closed: the controller must never accept a candidate without the
            # authoritative geometry checks supplied by the simulation integration.
            if is_walkable_fn is None or check_collision_fn is None or check_overlap_fn is None:
                candidate_evals.append(
                    CandidateEvaluation(
                        sample_idx=s_idx,
                        is_valid=False,
                        rejection_reason="missing_authoritative_safety_checks",
                        guide_waypoint=None,
                        guide_step_idx=-1,
                        target_exit_id=target_exit_id,
                        full_sample_collision_free=False,
                        local_congestion_count=0,
                        min_obstacle_clearance_m=0.0,
                        route_distance_m=float("inf"),
                    )
                )
                continue

            # Hard full-horizon feasibility filter. Every predicted point and every
            # continuous segment from the current position through the sample must
            # remain walkable and collision-free. A locally safe waypoint cannot
            # rescue a trajectory that later passes through a wall/obstacle.
            full_clean = True
            prev_pt = curr_pos
            for pred_pt in preds:
                pt_obj = point_cls(pred_pt.x, pred_pt.y)
                if not is_walkable_fn(pt_obj, ped_radius, self.building, target_exit_obj):
                    full_clean = False
                    break
                if check_collision_fn(prev_pt, pt_obj, ped_radius, self.building, target_exit_obj):
                    full_clean = False
                    break
                prev_pt = pt_obj

            if not full_clean:
                candidate_evals.append(
                    CandidateEvaluation(
                        sample_idx=s_idx,
                        is_valid=False,
                        rejection_reason="full_horizon_collision_or_unwalkable",
                        guide_waypoint=None,
                        guide_step_idx=-1,
                        target_exit_id=target_exit_id,
                        full_sample_collision_free=False,
                        local_congestion_count=0,
                        min_obstacle_clearance_m=0.0,
                        route_distance_m=float("inf"),
                    )
                )
                continue

            # Find a feasible intermediate guide waypoint along sample S_k without overshooting goal_wp
            selected_gw: Optional[Any] = None
            selected_h_idx: int = -1
            bypasses_goal_wp: bool = False
            fail_reason = "no_forward_progress"

            if d_curr_to_goal >= 1.5:
                gx_u = (goal_wp.x - curr_pos.x) / d_curr_to_goal
                gy_u = (goal_wp.y - curr_pos.y) / d_curr_to_goal

                for h_idx in horizon_probe_indices:
                    if h_idx >= len(preds):
                        continue
                    cand_p = preds[h_idx]
                    wk = point_cls(cand_p.x, cand_p.y)

                    rel_x = wk.x - curr_pos.x
                    rel_y = wk.y - curr_pos.y
                    proj = rel_x * gx_u + rel_y * gy_u
                    lat = abs(-rel_x * gy_u + rel_y * gx_u)
                    d_step = math.hypot(rel_x, rel_y)
                    d_to_goal = math.hypot(goal_wp.x - wk.x, goal_wp.y - wk.y)

                    # Strict non-overshoot & forward progress:
                    # Must advance at least 0.18m along goal vector, stop at least 0.55m before goal_wp,
                    # and keep lateral deviation bounded so trajectories never oscillate backward.
                    if proj < 0.18 or proj > d_curr_to_goal - 0.55:
                        continue
                    if lat > min(0.85, 0.45 * d_curr_to_goal):
                        continue
                    if d_to_goal >= d_curr_to_goal - 0.12:
                        continue

                    # Validate walkable space & clearance
                    if is_walkable_fn is not None:
                        if not is_walkable_fn(wk, ped_radius, self.building, target_exit_obj):
                            fail_reason = "boundary_or_obstacle_violation"
                            continue

                    # Validate continuous segment collision: curr_pos -> wk AND wk -> goal_wp
                    if check_collision_fn is not None:
                        if check_collision_fn(curr_pos, wk, ped_radius, self.building, target_exit_obj):
                            fail_reason = "obstacle_or_wall_collision"
                            continue
                        if check_collision_fn(wk, goal_wp, ped_radius, self.building, target_exit_obj):
                            fail_reason = "continuation_collision"
                            continue

                    # Validate pedestrian overlap
                    if check_overlap_fn is not None and other_active:
                        if any(check_overlap_fn(wk, ped_radius, other.position, other.radius) for other in other_active):
                            fail_reason = "pedestrian_overlap"
                            continue

                    selected_gw = wk
                    selected_h_idx = h_idx
                    break
            else:
                # Within 1.5m of goal_wp (doorway portal or final exit approach):
                # Check if sample S_k can smoothly round the corner directly to base_wps[1]
                if len(base_wps) > 1 and check_collision_fn is not None and is_walkable_fn is not None:
                    next_wp = base_wps[1]
                    d_curr_next = math.hypot(next_wp.x - curr_pos.x, next_wp.y - curr_pos.y)
                    for h_idx in (5, 3, 7, 2):
                        if h_idx >= len(preds):
                            continue
                        cand_p = preds[h_idx]
                        wk = point_cls(cand_p.x, cand_p.y)
                        d_wk_next = math.hypot(next_wp.x - wk.x, next_wp.y - wk.y)
                        d_step = math.hypot(wk.x - curr_pos.x, wk.y - curr_pos.y)
                        if d_step < 0.15 or d_wk_next >= d_curr_next - 0.15:
                            continue
                        if not is_walkable_fn(wk, ped_radius, self.building, target_exit_obj):
                            continue
                        if check_collision_fn(curr_pos, wk, ped_radius, self.building, target_exit_obj):
                            continue
                        if check_collision_fn(wk, next_wp, ped_radius, self.building, target_exit_obj):
                            continue
                        if check_overlap_fn is not None and other_active:
                            if any(check_overlap_fn(wk, ped_radius, other.position, other.radius) for other in other_active):
                                continue
                        selected_gw = wk
                        selected_h_idx = h_idx
                        bypasses_goal_wp = True
                        break

                # Otherwise, verify the sample progresses in the half-space of goal_wp and align to goal_wp
                if selected_gw is None and d_curr_to_goal > 1e-5:
                    gx_u = (goal_wp.x - curr_pos.x) / d_curr_to_goal
                    gy_u = (goal_wp.y - curr_pos.y) / d_curr_to_goal
                    end_p = preds[min(3, len(preds) - 1)]
                    proj_short = (end_p.x - curr_pos.x) * gx_u + (end_p.y - curr_pos.y) * gy_u
                    if proj_short > 0.05 or needs_exit_reroute:
                        can_direct = True
                        if check_collision_fn is not None:
                            if check_collision_fn(curr_pos, goal_wp, ped_radius, self.building, target_exit_obj):
                                can_direct = False
                        if can_direct:
                            selected_gw = goal_wp
                            selected_h_idx = min(3, len(preds) - 1)

            if selected_gw is None:
                candidate_evals.append(
                    CandidateEvaluation(
                        sample_idx=s_idx,
                        is_valid=False,
                        rejection_reason=fail_reason,
                        guide_waypoint=None,
                        guide_step_idx=-1,
                        target_exit_id=target_exit_id,
                        full_sample_collision_free=full_clean,
                        local_congestion_count=0,
                        min_obstacle_clearance_m=0.0,
                        route_distance_m=float("inf"),
                    )
                )
            else:
                cong_count = sum(
                    1 for other in other_active
                    if math.hypot(other.position.x - selected_gw.x, other.position.y - selected_gw.y) <= 3.0
                )
                obs_clear = self._compute_min_obstacle_clearance(selected_gw.x, selected_gw.y)
                if bypasses_goal_wp and len(base_wps) > 1:
                    tail_for_cand = base_wps[1:]
                else:
                    tail_for_cand = base_wps
                anchor_wp = tail_for_cand[0]
                route_dist = (
                    math.hypot(selected_gw.x - curr_pos.x, selected_gw.y - curr_pos.y)
                    + math.hypot(anchor_wp.x - selected_gw.x, anchor_wp.y - selected_gw.y)
                    + self._polyline_length(tail_for_cand)
                )
                candidate_evals.append(
                    CandidateEvaluation(
                        sample_idx=s_idx,
                        is_valid=True,
                        rejection_reason="bypass_corner" if bypasses_goal_wp else None,
                        guide_waypoint=selected_gw,
                        guide_step_idx=selected_h_idx,
                        target_exit_id=target_exit_id,
                        full_sample_collision_free=full_clean,
                        local_congestion_count=cong_count,
                        min_obstacle_clearance_m=obs_clear,
                        route_distance_m=route_dist,
                    )
                )

        valid_cands = [c for c in candidate_evals if c.is_valid]
        invalid_count = len(candidate_evals) - len(valid_cands)
        self.invalid_candidates_total += invalid_count

        if not valid_cands:
            # All candidates infeasible -> Fallback to original baseline route
            self._active_mid_waypoint.pop(agent_id, None)
            rec = MIDDecisionRecommendation(
                agent_id=agent_id,
                target_exit_id=target_exit_id,
                proposed_waypoints=tuple(base_wps),
                guide_waypoint=None,
                evaluated_samples_count=len(candidate_evals),
                valid_candidates_count=0,
                invalid_candidates_count=invalid_count,
                selected_sample_idx=-1,
                candidates=tuple(candidate_evals),
                selection_reason="All MID candidates failed geometric/progress checks; fallback to baseline",
                is_accepted=False,
                guidance_applied=False,
                fallback_to_original=True,
                rejection_reason="all_candidates_infeasible",
                step_index=step_index,
                timestamp=current_time,
                control_compute_time_s=time.perf_counter() - t_start,
            )
            self.fallback_count += 1
            self.latest_recommendations_by_agent[agent_id] = rec
            self.recommendation_history.append(rec)
            return rec

        # 5. Deterministic Lexicographic Selection among Valid Candidates
        valid_cands.sort(
            key=lambda c: (
                not c.full_sample_collision_free,
                c.local_congestion_count,
                -round(min(c.min_obstacle_clearance_m, 2.0), 1),
                round(c.route_distance_m, 3),
                c.sample_idx,
            )
        )
        best_cand = valid_cands[0]
        gw = best_cand.guide_waypoint
        did_bypass = best_cand.rejection_reason == "bypass_corner"

        # Clean up internal marker on candidate evaluation so rejection_reason is None for valid candidates
        cleaned_evals = tuple(
            CandidateEvaluation(
                sample_idx=c.sample_idx,
                is_valid=c.is_valid,
                rejection_reason=None if c.is_valid else c.rejection_reason,
                guide_waypoint=c.guide_waypoint,
                guide_step_idx=c.guide_step_idx,
                target_exit_id=c.target_exit_id,
                full_sample_collision_free=c.full_sample_collision_free,
                local_congestion_count=c.local_congestion_count,
                min_obstacle_clearance_m=c.min_obstacle_clearance_m,
                route_distance_m=c.route_distance_m,
            )
            for c in candidate_evals
        )

        # Construct proposed waypoints: prepend guide_waypoint if distinct from goal_wp
        effective_tail = base_wps[1:] if (did_bypass and len(base_wps) > 1) else base_wps
        assert gw is not None
        if abs(gw.x - effective_tail[0].x) < 1e-5 and abs(gw.y - effective_tail[0].y) < 1e-5:
            proposed_wps = tuple(effective_tail)
            self._active_mid_waypoint.pop(agent_id, None)
        else:
            proposed_wps = (gw, *effective_tail)
            self._active_mid_waypoint[agent_id] = gw

        self.accepted_interventions += 1
        self.last_decision_time[agent_id] = current_time

        reason_str = (
            f"Selected sample #{best_cand.sample_idx} -> wp({gw.x:.1f},{gw.y:.1f}) "
            f"[valid={len(valid_cands)}/{len(cleaned_evals)}, dist={best_cand.route_distance_m:.1f}m]"
        )

        rec = MIDDecisionRecommendation(
            agent_id=agent_id,
            target_exit_id=best_cand.target_exit_id,
            proposed_waypoints=proposed_wps,
            guide_waypoint=gw,
            evaluated_samples_count=len(cleaned_evals),
            valid_candidates_count=len(valid_cands),
            invalid_candidates_count=invalid_count,
            selected_sample_idx=best_cand.sample_idx,
            candidates=cleaned_evals,
            selection_reason=reason_str,
            is_accepted=True,
            guidance_applied=True,
            fallback_to_original=False,
            rejection_reason=None,
            step_index=step_index,
            timestamp=current_time,
            control_compute_time_s=time.perf_counter() - t_start,
        )
        self.latest_recommendations_by_agent[agent_id] = rec
        self.recommendation_history.append(rec)
        return rec


class CollisionAvoidanceSafetyLayer:
    """Explicit rule-based continuous safety guard separating hard safety from neural MID.

    Enforces:
    - Static wall, obstacle, and walkable-area checks across full segment and intermediate sub-steps.
    - Continuous swept-segment conflict & near-collision checks against moving neighbors.
    - Angular & speed-modulated safe maneuver fallback selection on conflict.
    - Anti-deadlock stall detection and lateral escape recovery.
    - Explicit structured safety intervention logging without mutating Pavan's engine internals.
    """

    def __init__(
        self,
        building: Any,
        hard_margin_m: float = 0.05,
        near_collision_margin_m: float = 0.28,
        stall_threshold_steps: int = 7,
        max_events: int = 240,
    ) -> None:
        self.building = building
        self.hard_margin_m = float(hard_margin_m)
        self.near_collision_margin_m = float(near_collision_margin_m)
        self.stall_threshold_steps = int(stall_threshold_steps)
        self.max_events = int(max_events)

        self.safety_interventions_total: int = 0
        self.static_interventions_total: int = 0
        self.dynamic_interventions_total: int = 0
        self.near_collision_avoided_total: int = 0
        self.deadlock_recoveries_total: int = 0

        self.events: List[SafetyInterventionEvent] = []
        self.latest_by_agent: Dict[str, SafetyCheckResult] = {}
        self._last_positions: Dict[str, Tuple[float, float]] = {}
        self._stall_counts: Dict[str, int] = {}

    def reset(self) -> None:
        """Reset intervention counters, stall history, and per-agent cache."""
        self.safety_interventions_total = 0
        self.static_interventions_total = 0
        self.dynamic_interventions_total = 0
        self.near_collision_avoided_total = 0
        self.deadlock_recoveries_total = 0
        self.events.clear()
        self.latest_by_agent.clear()
        self._last_positions.clear()
        self._stall_counts.clear()

    def _record_event(self, event: SafetyInterventionEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            del self.events[: len(self.events) - self.max_events]

    @staticmethod
    def _is_active_neighbor(agent_id: str, ped: Any) -> bool:
        if getattr(ped, "id", None) == agent_id:
            return False
        status = getattr(ped, "status", None)
        status_str = str(getattr(status, "value", status)).lower()
        return status_str == "active"

    def check_static_segment(
        self,
        start_pos: Any,
        end_pos: Any,
        radius: float,
        target_exit_obj: Any,
        is_walkable_fn: Optional[Callable[..., bool]],
        check_collision_fn: Optional[Callable[..., bool]],
    ) -> Tuple[bool, Optional[str]]:
        """Verify continuous segment safety across endpoint, sub-steps, and obstacle/wall checks."""
        if not (math.isfinite(end_pos.x) and math.isfinite(end_pos.y)):
            return False, "non_finite_target"
        if is_walkable_fn is None or check_collision_fn is None:
            return False, "missing_authoritative_geometry_checks"

        point_cls = type(start_pos)
        for frac in (0.34, 0.67, 1.0):
            px = start_pos.x + (end_pos.x - start_pos.x) * frac
            py = start_pos.y + (end_pos.y - start_pos.y) * frac
            if not is_walkable_fn(point_cls(px, py), radius, self.building, target_exit_obj):
                return False, "static_unwalkable_or_wall"

        if check_collision_fn(start_pos, end_pos, radius, self.building, target_exit_obj):
            return False, "static_segment_collision"

        return True, None

    def check_dynamic_neighbors(
        self,
        agent_id: str,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        radius: float,
        dt: float,
        other_pedestrians: Optional[Sequence[Any]],
        neighbor_velocities: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Tuple[bool, bool, float, Optional[str], Optional[Tuple[float, float]]]:
        """Compute continuous swept-segment conflict & near-collision risk against moving neighbors."""
        if not other_pedestrians:
            return False, False, 50.0, None, None

        min_clearance = 50.0
        has_hard = False
        has_near = False
        conflict_id: Optional[str] = None
        hazard_pt: Optional[Tuple[float, float]] = None

        look_dt = max(float(dt), 0.1)
        for other in other_pedestrians:
            if not self._is_active_neighbor(agent_id, other):
                continue
            oid = str(getattr(other, "id", ""))
            opos = getattr(other, "position", None)
            if opos is None:
                continue
            ox, oy = float(opos.x), float(opos.y)
            if math.hypot(ox - start_x, oy - start_y) > 4.5:
                continue
            orad = float(getattr(other, "radius", 0.3))

            if neighbor_velocities and oid in neighbor_velocities:
                ovx, ovy = neighbor_velocities[oid]
            else:
                ovel = getattr(other, "velocity", None)
                ovx = float(getattr(ovel, "vx", 0.0)) if ovel is not None else 0.0
                ovy = float(getattr(ovel, "vy", 0.0)) if ovel is not None else 0.0

            o_end_x = ox + ovx * look_dt
            o_end_y = oy + ovy * look_dt

            d_min, _, mid_pt = _continuous_segment_min_distance(
                start_x, start_y, end_x, end_y, ox, oy, o_end_x, o_end_y
            )
            combined_r = radius + orad
            clearance = d_min - combined_r
            curr_dist = math.hypot(ox - start_x, oy - start_y)

            if clearance < min_clearance:
                min_clearance = clearance

            # Hard physical conflict or overlap-tightening step
            if d_min < combined_r + self.hard_margin_m:
                if math.hypot(end_x - start_x, end_y - start_y) > 1e-4 and (
                    d_min <= curr_dist + 1e-4 or curr_dist < combined_r + 0.08
                ):
                    has_hard = True
                    conflict_id = oid
                    hazard_pt = mid_pt
            elif d_min < combined_r + self.near_collision_margin_m and d_min < curr_dist - 0.005:
                has_near = True
                if conflict_id is None:
                    conflict_id = oid
                    hazard_pt = mid_pt

        return has_hard, has_near, min_clearance, conflict_id, hazard_pt

    def evaluate_and_enforce(
        self,
        agent_id: str,
        current_pedestrian: Any,
        proposed_target: Tuple[float, float],
        dt: float,
        step_index: int,
        current_time: float,
        target_exit_obj: Any,
        goal_anchor: Optional[Tuple[float, float]] = None,
        other_pedestrians: Optional[Sequence[Any]] = None,
        neighbor_velocities: Optional[Dict[str, Tuple[float, float]]] = None,
        is_walkable_fn: Optional[Callable[..., bool]] = None,
        check_collision_fn: Optional[Callable[..., bool]] = None,
    ) -> SafetyCheckResult:
        """Validate a proposed step target and substitute a safe collision-free command if unsafe."""
        curr_pos = current_pedestrian.position
        point_cls = type(curr_pos)
        curr_x, curr_y = float(curr_pos.x), float(curr_pos.y)
        radius = float(getattr(current_pedestrian, "radius", 0.3))
        pref_speed = float(getattr(current_pedestrian, "preferred_speed", 1.34))
        eff_dt = max(float(dt), 1e-4)

        # Update deadlock / stall detector cleanly
        prev_pos = self._last_positions.get(agent_id)
        if prev_pos is not None:
            moved = math.hypot(curr_x - prev_pos[0], curr_y - prev_pos[1])
            if moved < 0.02:
                self._stall_counts[agent_id] = self._stall_counts.get(agent_id, 0) + 1
            else:
                self._stall_counts[agent_id] = 0
        else:
            self._stall_counts[agent_id] = 0
        self._last_positions[agent_id] = (curr_x, curr_y)

        prop_x, prop_y = float(proposed_target[0]), float(proposed_target[1])
        prop_dx = prop_x - curr_x
        prop_dy = prop_y - curr_y
        prop_dist = math.hypot(prop_dx, prop_dy)
        max_step = max(1e-4, pref_speed * eff_dt)

        # Normalize maximum step to preferred speed physically
        if prop_dist > max_step * 1.001 and prop_dist > 1e-9:
            scale = max_step / prop_dist
            prop_x = curr_x + prop_dx * scale
            prop_y = curr_y + prop_dy * scale
            prop_dx = prop_x - curr_x
            prop_dy = prop_y - curr_y
            prop_dist = max_step

        prop_speed = prop_dist / eff_dt
        goal_dist = math.hypot(goal_anchor[0] - curr_x, goal_anchor[1] - curr_y) if goal_anchor else 10.0
        stalled = self._stall_counts.get(agent_id, 0) >= self.stall_threshold_steps and goal_dist > 0.8

        # Evaluate nominal proposal
        prop_pt = point_cls(prop_x, prop_y)
        static_ok, static_reason = self.check_static_segment(
            curr_pos, prop_pt, radius, target_exit_obj, is_walkable_fn, check_collision_fn
        )
        hard_dyn, near_dyn, clearance_m, conflict_id, dyn_hazard = self.check_dynamic_neighbors(
            agent_id,
            curr_x,
            curr_y,
            prop_x,
            prop_y,
            radius,
            eff_dt,
            other_pedestrians,
            neighbor_velocities,
        )

        # Also validate a forward probe (next ~0.45s) against corners/walls when moving at speed
        forward_wall_risk = False
        if static_ok and prop_dist > 1e-4 and is_walkable_fn is not None and check_collision_fn is not None:
            ux, uy = prop_dx / prop_dist, prop_dy / prop_dist
            probe_len = min(max_step * 3.0, 0.55)
            probe_pt = point_cls(curr_x + ux * probe_len, curr_y + uy * probe_len)
            if target_exit_obj is not None:
                ex_mid_x = (target_exit_obj.segment.start.x + target_exit_obj.segment.end.x) * 0.5
                ex_mid_y = (target_exit_obj.segment.start.y + target_exit_obj.segment.end.y) * 0.5
                dist_ex = math.hypot(ex_mid_x - curr_x, ex_mid_y - curr_y)
            else:
                dist_ex = 10.0
            if dist_ex > probe_len + 0.15:
                probe_ok, _ = self.check_static_segment(
                    curr_pos, probe_pt, radius, target_exit_obj, is_walkable_fn, check_collision_fn
                )
                if not probe_ok:
                    forward_wall_risk = True

        hazard_points: List[Tuple[float, float]] = []
        rejected_targets: List[Tuple[float, float]] = []
        if not static_ok or forward_wall_risk:
            hazard_points.append((prop_x, prop_y))
            rejected_targets.append((prop_x, prop_y))
        if (hard_dyn or near_dyn) and dyn_hazard is not None:
            hazard_points.append(dyn_hazard)
            if (prop_x, prop_y) not in rejected_targets:
                rejected_targets.append((prop_x, prop_y))

        # Fast-path: nominal proposal is completely clean and not deadlocked
        if static_ok and not forward_wall_risk and not hard_dyn and not near_dyn and not stalled:
            res = SafetyCheckResult(
                is_proposed_safe=True,
                safe_target=(prop_x, prop_y),
                safe_speed_mps=prop_speed,
                intervention_applied=False,
                intervention_reason=None,
                hazard_points=(),
                rejected_targets=(),
                conflict_agent_id=None,
                min_clearance_m=clearance_m,
                near_collision_detected=False,
                deadlock_recovery_applied=False,
            )
            self.latest_by_agent[agent_id] = res
            return res

        # Candidate fallback maneuver search
        if prop_dist > 1e-5:
            base_ang = math.atan2(prop_dy, prop_dx)
        elif goal_anchor is not None:
            base_ang = math.atan2(goal_anchor[1] - curr_y, goal_anchor[0] - curr_x)
        else:
            base_ang = 0.0

        goal_ang = (
            math.atan2(goal_anchor[1] - curr_y, goal_anchor[0] - curr_x)
            if goal_anchor is not None
            else base_ang
        )

        # Deterministic probe ordering: deadlock prioritizes lateral escape; otherwise gradual deflection + slowdown
        if stalled:
            angle_offsets_deg = (45.0, -45.0, 75.0, -75.0, 105.0, -105.0, 22.0, -22.0, 135.0, -135.0)
            speed_scales = (0.85, 0.55)
        else:
            angle_offsets_deg = (0.0, 22.0, -22.0, 42.0, -42.0, 68.0, -68.0, 95.0, -95.0, 125.0, -125.0)
            speed_scales = (1.0, 0.70, 0.40)

        best_candidate: Optional[Tuple[Tuple[float, float, float, float], float, float, float, bool]] = None
        for deg in angle_offsets_deg:
            rad = base_ang + math.radians(deg)
            ux = math.cos(rad)
            uy = math.sin(rad)
            for s_frac in speed_scales:
                cand_step = max_step * s_frac
                cx = curr_x + ux * cand_step
                cy = curr_y + uy * cand_step
                c_pt = point_cls(cx, cy)

                c_stat_ok, _ = self.check_static_segment(
                    curr_pos, c_pt, radius, target_exit_obj, is_walkable_fn, check_collision_fn
                )
                if not c_stat_ok:
                    continue

                c_hard, c_near, c_clear, _, _ = self.check_dynamic_neighbors(
                    agent_id,
                    curr_x,
                    curr_y,
                    cx,
                    cy,
                    radius,
                    eff_dt,
                    other_pedestrians,
                    neighbor_velocities,
                )
                if c_hard:
                    continue

                # Score cleanly: prefer non-near-collision, goal progress, low angular deviation, higher speed
                goal_cos = math.cos(rad - goal_ang)
                dev_norm = abs(deg) / 180.0
                score_tuple = (
                    1.0 if c_near else 0.0,
                    -round(min(c_clear, 0.65), 2),
                    -round(goal_cos * s_frac, 3),
                    round(dev_norm, 3),
                )
                if best_candidate is None or score_tuple < best_candidate[0]:
                    best_candidate = (score_tuple, cx, cy, cand_step / eff_dt, c_near)

        if best_candidate is not None:
            _, safe_x, safe_y, safe_spd, _ = best_candidate
            changed_meaningfully = math.hypot(safe_x - prop_x, safe_y - prop_y) > 0.008 or stalled
            # If only near_collision triggered AND nominal still offered nearly identical clearance without improving separation
            if not changed_meaningfully and static_ok and not hard_dyn and not forward_wall_risk:
                res = SafetyCheckResult(
                    is_proposed_safe=True,
                    safe_target=(prop_x, prop_y),
                    safe_speed_mps=prop_speed,
                    intervention_applied=False,
                    intervention_reason=None,
                    hazard_points=tuple(hazard_points),
                    rejected_targets=(),
                    conflict_agent_id=conflict_id,
                    min_clearance_m=clearance_m,
                    near_collision_detected=near_dyn,
                    deadlock_recovery_applied=False,
                )
                self.latest_by_agent[agent_id] = res
                return res
        else:
            # Boxed-in safe yield: minimal non-zero hold vector aligned slightly within walkable space
            safe_x, safe_y, safe_spd = curr_x, curr_y, 0.0
            for tiny_deg in (0.0, 90.0, -90.0, 180.0):
                ang = base_ang + math.radians(tiny_deg)
                tx = curr_x + math.cos(ang) * 0.0025
                ty = curr_y + math.sin(ang) * 0.0025
                t_ok, _ = self.check_static_segment(
                    curr_pos, point_cls(tx, ty), radius, target_exit_obj, is_walkable_fn, check_collision_fn
                )
                if t_ok:
                    safe_x, safe_y, safe_spd = tx, ty, 0.0025 / eff_dt
                    break

        if not static_ok or forward_wall_risk:
            reason = static_reason or "static_wall_or_obstacle_risk"
            self.static_interventions_total += 1
        elif hard_dyn:
            reason = "moving_pedestrian_conflict"
            self.dynamic_interventions_total += 1
        elif near_dyn:
            reason = "near_collision_risk"
            self.near_collision_avoided_total += 1
        elif stalled:
            reason = "deadlock_stall_recovery"
            self.deadlock_recoveries_total += 1
            self._stall_counts[agent_id] = 0
        else:
            reason = "safety_adjustment"

        self.safety_interventions_total += 1
        self._record_event(
            SafetyInterventionEvent(
                agent_id=agent_id,
                step_index=step_index,
                timestamp=current_time,
                reason=reason,
                proposed_target=(prop_x, prop_y),
                safe_target=(safe_x, safe_y),
                hazard_location=hazard_points[0] if hazard_points else None,
                conflict_agent_id=conflict_id,
                proposed_speed_mps=prop_speed,
                safe_speed_mps=safe_spd,
                min_clearance_m=clearance_m,
            )
        )

        res = SafetyCheckResult(
            is_proposed_safe=(static_ok and not hard_dyn and not forward_wall_risk),
            safe_target=(safe_x, safe_y),
            safe_speed_mps=safe_spd,
            intervention_applied=True,
            intervention_reason=reason,
            hazard_points=tuple(hazard_points),
            rejected_targets=tuple(rejected_targets),
            conflict_agent_id=conflict_id,
            min_clearance_m=clearance_m,
            near_collision_detected=near_dyn,
            deadlock_recovery_applied=stalled and reason == "deadlock_stall_recovery",
        )
        self.latest_by_agent[agent_id] = res
        return res


class MIDDrivenMovementController:
    """Primary MID-driven closed-loop movement controller with decoupled collision safety layer.

    Unlike baseline reactive routing or weak MID-assisted waypoint nudges, this controller
    uses the learned multi-agent EvacuationMIDPredictor as the primary generator of
    per-step pedestrian velocity and direction, then filters and stabilizes those commands
    through ``CollisionAvoidanceSafetyLayer`` before actuating them cleanly via Pavan's
    authoritative ``ped.waypoints`` + ``SimulationEngine.step()`` interface.
    """

    def __init__(self, building: Any, safety_layer: Optional[CollisionAvoidanceSafetyLayer] = None) -> None:
        self.building = building
        self.safety_layer = safety_layer if safety_layer is not None else CollisionAvoidanceSafetyLayer(building)
        self._active_step_waypoint: Dict[str, Any] = {}
        self.latest_decisions: Dict[str, MIDDecisionRecommendation] = {}
        self.latest_safety: Dict[str, SafetyCheckResult] = {}
        self.total_control_steps: int = 0
        self.mid_driven_steps: int = 0
        self.route_reroutes: int = 0

    def reset(self) -> None:
        """Clear controller state and underlying safety layer counters."""
        self._active_step_waypoint.clear()
        self.latest_decisions.clear()
        self.latest_safety.clear()
        self.total_control_steps = 0
        self.mid_driven_steps = 0
        self.route_reroutes = 0
        self.safety_layer.reset()

    def get_base_waypoints(self, agent_id: str, ped: Any) -> List[Any]:
        """Strip any previously injected short-horizon MID-DRIVEN step waypoint, keeping exit anchors."""
        current_wps = list(getattr(ped, "waypoints", []))
        active_swp = self._active_step_waypoint.get(agent_id)
        if active_swp is not None and current_wps:
            first_wp = current_wps[0]
            if (
                abs(first_wp.x - active_swp.x) < 1e-5
                and abs(first_wp.y - active_swp.y) < 1e-5
                and len(current_wps) > 1
            ):
                return current_wps[1:]
        return current_wps

    def compute_driven_command(
        self,
        agent_id: str,
        ped: Any,
        mid_prediction: Optional[MultiTrajectoryPrediction],
        open_exit_ids: Sequence[str],
        dt: float,
        current_time: float,
        step_index: int,
        generate_route_fn: Callable[..., List[Any]],
        is_walkable_fn: Optional[Callable[..., bool]] = None,
        check_collision_fn: Optional[Callable[..., bool]] = None,
        other_pedestrians: Optional[Sequence[Any]] = None,
        neighbor_velocities: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Tuple[List[Any], MIDDecisionRecommendation, SafetyCheckResult]:
        """Generate a safe, MID-driven waypoint command ``[safe_step_wp, *tail]`` for one physics step."""
        t_start = time.perf_counter()
        self.total_control_steps += 1

        curr_pos = ped.position
        point_cls = type(curr_pos)
        curr_x, curr_y = float(curr_pos.x), float(curr_pos.y)
        ped_radius = float(getattr(ped, "radius", 0.3))
        pref_speed = float(getattr(ped, "preferred_speed", 1.34))
        eff_dt = max(float(dt), 1e-4)

        base_wps = self.get_base_waypoints(agent_id, ped)
        curr_exit_id = getattr(ped, "target_exit_id", None)

        # Resolve reachable open exit anchor
        open_exits = [ex for ex in self.building.exits if ex.id in open_exit_ids and ex.is_open]
        open_exit_map = {ex.id: ex for ex in open_exits}
        target_exit_id = curr_exit_id if (curr_exit_id in open_exit_map) else None
        target_exit_obj = open_exit_map.get(target_exit_id) if target_exit_id else None

        start_area = getattr(ped, "current_area_id", None)
        needs_reroute = target_exit_obj is None or not base_wps
        if needs_reroute and open_exits and start_area:
            best_route: Optional[List[Any]] = None
            best_len = float("inf")
            best_ex_id = target_exit_id
            for ex in open_exits:
                try:
                    cand_route = generate_route_fn(
                        building=self.building,
                        start_pos=curr_pos,
                        start_area_id=start_area,
                        target_exit_id=ex.id,
                        radius=ped_radius,
                    )
                    if not cand_route:
                        continue
                    r_len = math.hypot(cand_route[0].x - curr_x, cand_route[0].y - curr_y)
                    for i in range(len(cand_route) - 1):
                        r_len += math.hypot(
                            cand_route[i + 1].x - cand_route[i].x,
                            cand_route[i + 1].y - cand_route[i].y,
                        )
                    if r_len < best_len:
                        best_len = r_len
                        best_route = cand_route
                        best_ex_id = ex.id
                except Exception:
                    continue
            if best_route and best_ex_id:
                base_wps = best_route
                target_exit_id = best_ex_id
                target_exit_obj = open_exit_map.get(best_ex_id)
                self.route_reroutes += 1

        if not base_wps:
            # Hold safely if no topological exit remains reachable
            res_safe = SafetyCheckResult(
                is_proposed_safe=True,
                safe_target=(curr_x, curr_y),
                safe_speed_mps=0.0,
                intervention_applied=False,
                intervention_reason="no_reachable_exit",
                hazard_points=(),
                rejected_targets=(),
                conflict_agent_id=None,
                min_clearance_m=50.0,
                near_collision_detected=False,
                deadlock_recovery_applied=False,
            )
            rec = MIDDecisionRecommendation(
                agent_id=agent_id,
                target_exit_id=target_exit_id,
                proposed_waypoints=(),
                guide_waypoint=None,
                evaluated_samples_count=0,
                valid_candidates_count=0,
                invalid_candidates_count=0,
                selected_sample_idx=-1,
                candidates=(),
                selection_reason="MID-DRIVEN: No open exit route available",
                is_accepted=False,
                guidance_applied=False,
                fallback_to_original=True,
                rejection_reason="no_reachable_exit",
                step_index=step_index,
                timestamp=current_time,
                control_compute_time_s=time.perf_counter() - t_start,
            )
            self.latest_decisions[agent_id] = rec
            self.latest_safety[agent_id] = res_safe
            return [], rec, res_safe

        goal_wp = base_wps[0]
        goal_dist = math.hypot(goal_wp.x - curr_x, goal_wp.y - curr_y)

        # Final exit portal crossing (< 0.45m directly to final exit endpoint): let Pavan cross cleanly
        if len(base_wps) == 1 and goal_dist <= max(0.45, pref_speed * eff_dt * 2.5):
            goal_ok, _ = self.safety_layer.check_static_segment(
                curr_pos, goal_wp, ped_radius, target_exit_obj, is_walkable_fn, check_collision_fn
            )
            if goal_ok:
                safety_res = self.safety_layer.evaluate_and_enforce(
                    agent_id=agent_id,
                    current_pedestrian=ped,
                    proposed_target=(float(goal_wp.x), float(goal_wp.y)),
                    dt=eff_dt,
                    step_index=step_index,
                    current_time=current_time,
                    target_exit_obj=target_exit_obj,
                    goal_anchor=(float(goal_wp.x), float(goal_wp.y)),
                    other_pedestrians=other_pedestrians,
                    neighbor_velocities=neighbor_velocities,
                    is_walkable_fn=is_walkable_fn,
                    check_collision_fn=check_collision_fn,
                )
                sx, sy = safety_res.safe_target
                if math.hypot(sx - goal_wp.x, sy - goal_wp.y) < 0.05 and not safety_res.intervention_applied:
                    self._active_step_waypoint.pop(agent_id, None)
                    cmd_wps = list(base_wps)
                else:
                    step_wp = point_cls(sx, sy)
                    self._active_step_waypoint[agent_id] = step_wp
                    cmd_wps = [step_wp, *base_wps]
                rec = MIDDecisionRecommendation(
                    agent_id=agent_id,
                    target_exit_id=target_exit_id,
                    proposed_waypoints=tuple(cmd_wps),
                    guide_waypoint=cmd_wps[0],
                    evaluated_samples_count=len(mid_prediction.samples) if mid_prediction else 0,
                    valid_candidates_count=0,
                    invalid_candidates_count=0,
                    selected_sample_idx=0,
                    candidates=(),
                    selection_reason="MID-DRIVEN: Final exit crossing phase",
                    is_accepted=True,
                    guidance_applied=True,
                    fallback_to_original=False,
                    step_index=step_index,
                    timestamp=current_time,
                    control_compute_time_s=time.perf_counter() - t_start,
                )
                self.latest_decisions[agent_id] = rec
                self.latest_safety[agent_id] = safety_res
                return cmd_wps, rec, safety_res

        # Evaluate MID candidate trajectories as primary velocity/direction proposals
        cand_evals: List[CandidateEvaluation] = []
        scored_valid: List[Tuple[Tuple[float, float, float, float, int], int, Tuple[float, float], Tuple[float, float], int]] = []
        invalid_cnt = 0

        samples = mid_prediction.samples if mid_prediction is not None else ()
        goal_ux = (goal_wp.x - curr_x) / (goal_dist + 1e-8)
        goal_uy = (goal_wp.y - curr_y) / (goal_dist + 1e-8)

        for s_idx, sample in enumerate(samples):
            preds = sample.predictions
            if len(preds) == 0 or not all(math.isfinite(p.x) and math.isfinite(p.y) for p in preds):
                invalid_cnt += 1
                cand_evals.append(
                    CandidateEvaluation(
                        sample_idx=s_idx,
                        is_valid=False,
                        rejection_reason="empty_or_non_finite",
                        guide_waypoint=None,
                        guide_step_idx=-1,
                        target_exit_id=target_exit_id,
                        full_sample_collision_free=False,
                        local_congestion_count=0,
                        min_obstacle_clearance_m=0.0,
                        route_distance_m=float("inf"),
                    )
                )
                continue

            first_pred = preds[0]
            # Extract immediate MID velocity over first 0.2s (or step 0..1)
            dt_0 = max(1e-3, first_pred.timestamp - sample.source_timestamp)
            if len(preds) >= 2:
                p_look = preds[1]
                dt_look = max(1e-3, p_look.timestamp - sample.source_timestamp)
            else:
                p_look = first_pred
                dt_look = dt_0

            dx_m = p_look.x - curr_x
            dy_m = p_look.y - curr_y
            dist_m = math.hypot(dx_m, dy_m)
            mid_speed = min(pref_speed, dist_m / dt_look)

            if dist_m > 1e-5:
                mux, muy = dx_m / dist_m, dy_m / dist_m
                step_dx = mux * mid_speed * eff_dt
                step_dy = muy * mid_speed * eff_dt
            else:
                # Slow yield predicted by MID
                step_dx = goal_ux * (0.04 * pref_speed * eff_dt)
                step_dy = goal_uy * (0.04 * pref_speed * eff_dt)

            prop_tx = curr_x + step_dx
            prop_ty = curr_y + step_dy

            # Require MID candidate not to reverse > 70 degrees backward away from tactical goal
            # unless responding strongly to a close social obstruction directly ahead
            social_active = (
                mid_prediction is not None
                and mid_prediction.social_summary is not None
                and mid_prediction.social_summary.nearby_count > 0
            )
            forward_align = (step_dx * goal_ux + step_dy * goal_uy) / (math.hypot(step_dx, step_dy) + 1e-8)
            min_cos = -0.35 if social_active else 0.10
            if forward_align < min_cos:
                invalid_cnt += 1
                cand_evals.append(
                    CandidateEvaluation(
                        sample_idx=s_idx,
                        is_valid=False,
                        rejection_reason="backward_heading",
                        guide_waypoint=None,
                        guide_step_idx=-1,
                        target_exit_id=target_exit_id,
                        full_sample_collision_free=False,
                        local_congestion_count=0,
                        min_obstacle_clearance_m=0.0,
                        route_distance_m=float("inf"),
                    )
                )
                continue

            # Verify short horizon walkability & segment safety over first 5 points (~0.5s)
            short_clean = True
            prev_pt = curr_pos
            horizon_check = min(5, len(preds))
            for h_i in range(horizon_check):
                pt_obj = point_cls(preds[h_i].x, preds[h_i].y)
                s_ok, _ = self.safety_layer.check_static_segment(
                    prev_pt, pt_obj, ped_radius, target_exit_obj, is_walkable_fn, check_collision_fn
                )
                if not s_ok:
                    short_clean = False
                    break
                prev_pt = pt_obj

            if not short_clean:
                invalid_cnt += 1
                cand_evals.append(
                    CandidateEvaluation(
                        sample_idx=s_idx,
                        is_valid=False,
                        rejection_reason="static_obstacle_or_wall",
                        guide_waypoint=None,
                        guide_step_idx=-1,
                        target_exit_id=target_exit_id,
                        full_sample_collision_free=False,
                        local_congestion_count=0,
                        min_obstacle_clearance_m=0.0,
                        route_distance_m=float("inf"),
                    )
                )
                continue

            c_hard, c_near, c_clear, _, _ = self.safety_layer.check_dynamic_neighbors(
                agent_id,
                curr_x,
                curr_y,
                prop_tx,
                prop_ty,
                ped_radius,
                eff_dt,
                other_pedestrians,
                neighbor_velocities,
            )

            gw_pt = point_cls(preds[min(3, len(preds) - 1)].x, preds[min(3, len(preds) - 1)].y)
            tail_dist = math.hypot(goal_wp.x - prop_tx, goal_wp.y - prop_ty)
            # Rank clean MID trajectories preferring dynamic safety, progress, & preserved MID speed
            score_key = (
                2.0 if c_hard else (1.0 if c_near else 0.0),
                -round(min(c_clear, 0.75), 2),
                round(tail_dist, 3),
                -round(forward_align, 2),
                s_idx,
            )
            cand_eval = CandidateEvaluation(
                sample_idx=s_idx,
                is_valid=True,
                rejection_reason=None,
                guide_waypoint=gw_pt,
                guide_step_idx=min(3, len(preds) - 1),
                target_exit_id=target_exit_id,
                full_sample_collision_free=not c_hard,
                local_congestion_count=1 if c_near else 0,
                min_obstacle_clearance_m=max(0.0, c_clear),
                route_distance_m=tail_dist,
            )
            cand_evals.append(cand_eval)
            scored_valid.append((score_key, s_idx, (prop_tx, prop_ty), (gw_pt.x, gw_pt.y), len(preds)))

        if scored_valid:
            scored_valid.sort(key=lambda item: item[0])
            _, best_idx, best_prop_target, best_gw_xy, _ = scored_valid[0]
            mid_active = True
            source_desc = f"MID-DRIVEN sample C{best_idx}"
        else:
            # No statically valid MID sample -> propose tactical step toward goal anchor, guarded by safety layer
            best_idx = -1
            step_len = min(goal_dist, pref_speed * eff_dt * 0.85)
            best_prop_target = (curr_x + goal_ux * step_len, curr_y + goal_uy * step_len)
            best_gw_xy = (float(goal_wp.x), float(goal_wp.y))
            mid_active = False
            source_desc = "MID-DRIVEN safety-fallback step"

        # Enforce explicit collision avoidance layer on EVERY step before Pavan executes it
        safety_res = self.safety_layer.evaluate_and_enforce(
            agent_id=agent_id,
            current_pedestrian=ped,
            proposed_target=best_prop_target,
            dt=eff_dt,
            step_index=step_index,
            current_time=current_time,
            target_exit_obj=target_exit_obj,
            goal_anchor=(float(goal_wp.x), float(goal_wp.y)),
            other_pedestrians=other_pedestrians,
            neighbor_velocities=neighbor_velocities,
            is_walkable_fn=is_walkable_fn,
            check_collision_fn=check_collision_fn,
        )

        sx, sy = safety_res.safe_target
        # Keep non-trivial distance >= 0.002m from current position so Pavan does not discard waypoints
        if math.hypot(sx - curr_x, sy - curr_y) < 1e-3:
            sx = curr_x + goal_ux * 0.0025
            sy = curr_y + goal_uy * 0.0025
            t_ok, _ = self.safety_layer.check_static_segment(
                curr_pos, point_cls(sx, sy), ped_radius, target_exit_obj, is_walkable_fn, check_collision_fn
            )
            if not t_ok:
                sx, sy = curr_x, curr_y

        safe_step_wp = point_cls(sx, sy)
        self._active_step_waypoint[agent_id] = safe_step_wp
        cmd_wps = [safe_step_wp, *base_wps]

        if mid_active:
            self.mid_driven_steps += 1

        safety_tag = f" [SAFETY: {safety_res.intervention_reason}]" if safety_res.intervention_applied else ""
        rec = MIDDecisionRecommendation(
            agent_id=agent_id,
            target_exit_id=target_exit_id,
            proposed_waypoints=tuple(cmd_wps),
            guide_waypoint=point_cls(best_gw_xy[0], best_gw_xy[1]),
            evaluated_samples_count=len(cand_evals),
            valid_candidates_count=len(scored_valid),
            invalid_candidates_count=invalid_cnt,
            selected_sample_idx=best_idx,
            candidates=tuple(cand_evals),
            selection_reason=f"{source_desc} (v={safety_res.safe_speed_mps:.2f}m/s){safety_tag}",
            is_accepted=mid_active or not safety_res.intervention_applied,
            guidance_applied=True,
            fallback_to_original=not mid_active,
            rejection_reason=safety_res.intervention_reason,
            step_index=step_index,
            timestamp=current_time,
            control_compute_time_s=time.perf_counter() - t_start,
        )
        self.latest_decisions[agent_id] = rec
        self.latest_safety[agent_id] = safety_res
        return cmd_wps, rec, safety_res


__all__ = [
    "CandidateEvaluation",
    "CollisionAvoidanceSafetyLayer",
    "MIDAssistedDecisionAdapter",
    "MIDDecisionRecommendation",
    "MIDDrivenMovementController",
    "SafetyCheckResult",
    "SafetyInterventionEvent",
]
