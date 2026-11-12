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

from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from genevac.prediction.data_contract import TrajectorySnapshot
from genevac.prediction.mid import MultiTrajectoryPrediction


@dataclass(frozen=True)
class CandidateEvaluation:
    """Detailed feasibility and physical evaluation of a single MID trajectory sample."""
    sample_idx: int
    is_valid: bool
    rejection_reason: Optional[str]
    guide_waypoint: Optional[Any]
    guide_step_idx: int
    target_exit_id: str
    full_sample_collision_free: bool
    local_congestion_count: int
    min_obstacle_clearance_m: float
    route_distance_m: float


@dataclass(frozen=True)
class MIDDecisionRecommendation:
    """Immutable recommendation and audit record from the MID-assisted decision adapter."""
    agent_id: str
    target_exit_id: str
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
        remaining_tail_len = self._polyline_length(base_wps)

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

            # Check whether full 12-step sample is collision-free in building
            full_clean = True
            if check_collision_fn is not None and is_walkable_fn is not None:
                prev_pt = curr_pos
                for p in preds:
                    pt_obj = point_cls(p.x, p.y)
                    if not is_walkable_fn(pt_obj, ped_radius, self.building, target_exit_obj):
                        full_clean = False
                        break
                    if check_collision_fn(prev_pt, pt_obj, ped_radius, self.building, target_exit_obj):
                        full_clean = False
                        break
                    prev_pt = pt_obj

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
