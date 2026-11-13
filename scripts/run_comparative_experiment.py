"""Controlled Comparative Experiment & Three Proofs Generator for GenEvac-Sentinel.

Executes and verifies the three operating modes on identical initial states:
1. ORIGINAL SIMULATION (deterministic Pavan baseline)
2. MID PREDICTION ONLY (read-only stochastic diffusion overlays)
3. MID-ASSISTED EVACUATION (closed-loop MID trajectory evaluation & waypoint steering)

Generates and verifies the Three Required Proofs:
- Proof 1: Same starting conditions, different actual pedestrian movement in engine state
  (and 0.0m divergence between ORIGINAL and MID PREDICTION ONLY).
- Proof 2: Actual MID causality trace (generated K=6 candidates -> feasibility filter &
  ranking -> selected candidate & guide_waypoint -> updated ped.waypoints -> divergent
  next-step ped.position from Pavan's SimulationEngine.step()).
- Proof 3: Safety & physics preservation (0 wall/obstacle penetrations, 0 speed limit
  violations, 0 teleports, 0 pedestrian overlaps across all steps).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ["SDL_VIDEODRIVER"] = "dummy"

from genevac.prediction.data_contract import TrajectoryObservation
from genevac.prediction.evaluation import EvaluationSample, PredictionEvaluator, SampleGenerator
from genevac.visualization.integrated_app import (
    IntegratedSimulationApp,
    OperatingMode,
    PROJECT_ROOT,
    PAVAN,
)


def _create_interactive_crowd_cohort() -> List[Any]:
    """Deterministic dense multi-pedestrian cohort in building_main for interaction benchmarking."""
    cohort = list(PAVAN["create_default_pedestrian_cohort"]())
    # Clone additional pedestrians with safe initial spacing across central hall & doorways
    template = cohort[0]
    ped_cls = type(template)
    point_cls = type(template.position)
    # Infer Velocity2D type from template.velocity and PedestrianStatus from template.status
    vel_cls = type(template.velocity)
    extras = [
        # Follower faster behind ped_1 in room_1 toward exit_left
        ("ped_int_follow_1", point_cls(9.6, 6.2), 1.48, "exit_left"),
        # Oncoming walkers approaching across main_corridor against ped_5
        ("ped_int_opp_1", point_cls(11.0, 15.0), 1.32, "exit_right"),
        ("ped_int_opp_2", point_cls(27.0, 15.05), 1.28, "exit_left"),
        # Merging pair & slow leader ahead in room_3 toward exit_bottom
        ("ped_int_slow_1", point_cls(10.5, 26.5), 0.58, "exit_bottom"),
        ("ped_int_merge_1", point_cls(13.0, 20.0), 1.35, "exit_bottom"),
        ("ped_int_merge_2", point_cls(8.2, 22.0), 1.28, "exit_bottom"),
    ]
    building_tmp = PAVAN["load_building"](PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json")
    for pid, pos, spd, exit_id in extras:
        area = "main_corridor"
        for room in building_tmp.walkable_areas:
            b = room.bounds
            if b.x <= pos.x <= b.x + b.width and b.y <= pos.y <= b.y + b.height:
                area = room.id
                break
        wps = PAVAN["generate_deterministic_route"](
            building=building_tmp,
            start_pos=pos,
            start_area_id=area,
            target_exit_id=exit_id,
            radius=float(template.radius),
        )
        cohort.append(
            ped_cls(
                id=pid,
                position=pos,
                velocity=vel_cls(0.0, 0.0),
                preferred_speed=spd,
                radius=float(template.radius),
                target_exit_id=exit_id,
                waypoints=wps,
                status=template.status,
                current_area_id=area,
            )
        )
    return cohort


def run_mode_simulation(
    mode: OperatingMode,
    seed: int = 42,
    max_steps: int = 350,
    close_exit_left_at_step: Optional[int] = None,
    shared_mid_predictor: Any = None,
    use_interactive_crowd: bool = False,
) -> Dict[str, Any]:
    """Run a complete evacuation simulation under the specified operating mode and seed."""
    map_path = PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json"
    app = IntegratedSimulationApp(
        map_path=str(map_path),
        width=1440,
        height=860,
        fps=30,
        mid_predictor=shared_mid_predictor,
    )
    if use_interactive_crowd:
        crowd = _create_interactive_crowd_cohort()
        reg = PAVAN["PedestrianRegistry"]()
        for ped in crowd:
            reg.register(ped)
        app.engine = PAVAN["SimulationEngine"](
            building=app.building,
            registry=reg,
            dt=app.dt,
            observation_interval=app.observation_interval,
        )
        app.adapter.reset()
        app._driven_history_buffer.reset()
        app.heatmap_engine.reset()
        app.controller.reset()
        app.driven_controller.reset()
        app._latest_driven_mid.clear()
        app.actual_traversed_paths.clear()
        app.adapter.adapt_snapshot(app.engine.latest_snapshot)
        app._driven_history_buffer.adapt_snapshot(app.engine.latest_snapshot)
        app._record_engine_positions()
    app.operating_mode = mode
    app.base_seed = seed
    app.selected_agent_id = "ped_1"

    is_walkable_fn = PAVAN["is_position_in_walkable_space"]
    check_overlap_fn = PAVAN["check_pedestrian_overlap"]

    initial_count = len(app.engine.registry)
    trajectories: Dict[str, List[Tuple[int, float, float, str]]] = {
        p.id: [(0, float(p.position.x), float(p.position.y), str(getattr(p.status, "value", p.status)).lower())]
        for p in app.engine.registry
    }
    prev_positions: Dict[str, Tuple[float, float]] = {
        p.id: (float(p.position.x), float(p.position.y)) for p in app.engine.registry
    }
    evac_step_by_agent: Dict[str, Optional[int]] = {p.id: None for p in app.engine.registry}

    speed_violations = 0
    walkability_violations = 0
    overlap_violations = 0
    near_collision_events = 0
    teleport_violations = 0
    max_step_displacement = 0.0
    step_speeds_sum = 0.0
    step_speed_samples = 0
    peak_nearby_density = 0.0
    causality_traces: List[Dict[str, Any]] = []

    for s in range(1, max_steps + 1):
        if close_exit_left_at_step is not None and s == close_exit_left_at_step:
            app.engine.close_exit("exit_left")

        app.update_physics(0.1)
        if mode == OperatingMode.MID_PREDICTION_ONLY and app.engine.clock.is_observation_step():
            snap = app.adapter.buffer.get_snapshot(app.engine.clock.step_count, app.engine.clock.timestamp)
            if "ped_1" in snap.trajectories and len(snap.trajectories["ped_1"]) >= 2:
                app._get_or_compute_mid_prediction(snap, "ped_1", app.engine.clock.step_count)

        # Capture causality trace at step 4 and step 8
        if s in (4, 8):
            ped1 = app.engine.registry.get("ped_1")
            rec1 = (
                app.driven_controller.latest_decisions.get("ped_1")
                if mode == OperatingMode.MID_DRIVEN
                else app.controller.latest_recommendations_by_agent.get("ped_1")
            )
            trace_entry: Dict[str, Any] = {
                "step": s,
                "timestamp_s": round(app.engine.clock.timestamp, 2),
                "mode": mode.value,
                "ped_1_position": (round(ped1.position.x, 4), round(ped1.position.y, 4)),
                "ped_1_velocity": (round(ped1.velocity.vx, 4), round(ped1.velocity.vy, 4)),
                "ped_1_target_exit": ped1.target_exit_id,
                "ped_1_waypoints": [(round(w.x, 4), round(w.y, 4)) for w in ped1.waypoints],
            }
            if rec1 is not None:
                trace_entry["controller_decision"] = {
                    "evaluated_samples": rec1.evaluated_samples_count,
                    "valid_candidates": rec1.valid_candidates_count,
                    "invalid_candidates": rec1.invalid_candidates_count,
                    "selected_sample_idx": rec1.selected_sample_idx,
                    "guide_waypoint": (
                        (round(rec1.guide_waypoint.x, 4), round(rec1.guide_waypoint.y, 4))
                        if rec1.guide_waypoint is not None
                        else None
                    ),
                    "selection_reason": rec1.selection_reason,
                    "guidance_applied": rec1.guidance_applied,
                    "fallback_to_original": rec1.fallback_to_original,
                    "candidates": [
                        {
                            "sample_idx": c.sample_idx,
                            "is_valid": c.is_valid,
                            "rejection_reason": c.rejection_reason,
                            "guide_waypoint": (
                                (round(c.guide_waypoint.x, 4), round(c.guide_waypoint.y, 4))
                                if c.guide_waypoint is not None
                                else None
                            ),
                            "full_sample_collision_free": c.full_sample_collision_free,
                            "local_congestion_count": c.local_congestion_count,
                            "min_obstacle_clearance_m": round(c.min_obstacle_clearance_m, 3),
                            "route_distance_m": round(c.route_distance_m, 3),
                        }
                        for c in rec1.candidates
                    ],
                }
            causality_traces.append(trace_entry)

        # Record positions and verify physical safety constraints
        active_peds = []
        for p in app.engine.registry:
            st = str(getattr(p.status, "value", p.status)).lower()
            curr_xy = (float(p.position.x), float(p.position.y))
            prev_xy = prev_positions[p.id]
            step_dist = math.hypot(curr_xy[0] - prev_xy[0], curr_xy[1] - prev_xy[1])
            max_allowed = p.preferred_speed * app.dt + 1e-5
            if step_dist > max_allowed:
                speed_violations += 1
            if step_dist > 0.5:  # Any jump > 0.5m in 0.1s is a teleport
                teleport_violations += 1
            if step_dist > max_step_displacement:
                max_step_displacement = step_dist

            prev_positions[p.id] = curr_xy
            trajectories[p.id].append((s, curr_xy[0], curr_xy[1], st))

            if st == "evacuated" and evac_step_by_agent[p.id] is None:
                evac_step_by_agent[p.id] = s

            if st == "active":
                active_peds.append(p)
                step_speeds_sum += step_dist / app.dt
                step_speed_samples += 1
                target_ex = app.building.get_exit(p.target_exit_id) if p.target_exit_id else None
                if not is_walkable_fn(p.position, p.radius, app.building, target_ex):
                    walkability_violations += 1

        # Pairwise overlap & near-collision density check among active pedestrians
        for i in range(len(active_peds)):
            pi = active_peds[i]
            close_3m = 1
            for j in range(len(active_peds)):
                if i == j:
                    continue
                pj = active_peds[j]
                d_ij = math.hypot(pi.position.x - pj.position.x, pi.position.y - pj.position.y)
                if d_ij <= 3.0:
                    close_3m += 1
                if j > i:
                    if check_overlap_fn(pi.position, pi.radius, pj.position, pj.radius):
                        overlap_violations += 1
                    elif d_ij < (pi.radius + pj.radius + 0.14):
                        near_collision_events += 1
            local_den = close_3m / (math.pi * 3.0 * 3.0)
            if local_den > peak_nearby_density:
                peak_nearby_density = local_den

        evac_count = sum(
            1 for p in app.engine.registry if str(getattr(p.status, "value", p.status)).lower() == "evacuated"
        )
        blocked_count = sum(
            1 for p in app.engine.registry if str(getattr(p.status, "value", p.status)).lower() == "blocked"
        )
        if evac_count == initial_count or (len(active_peds) == 0 and blocked_count > 0):
            break

    total_time = app.engine.clock.timestamp
    final_evac = sum(
        1 for p in app.engine.registry if str(getattr(p.status, "value", p.status)).lower() == "evacuated"
    )
    final_blocked = sum(
        1 for p in app.engine.registry if str(getattr(p.status, "value", p.status)).lower() == "blocked"
    )
    final_active = sum(
        1 for p in app.engine.registry if str(getattr(p.status, "value", p.status)).lower() == "active"
    )

    total_dist_by_agent: Dict[str, float] = {}
    for aid, pts in trajectories.items():
        dist = 0.0
        for k in range(len(pts) - 1):
            dist += math.hypot(pts[k + 1][1] - pts[k][1], pts[k + 1][2] - pts[k][2])
        total_dist_by_agent[aid] = round(dist, 3)

    if mode == OperatingMode.MID_ASSISTED_EVACUATION:
        ctrl_interventions = app.controller.accepted_interventions
        ctrl_rejected = app.controller.invalid_candidates_total
        ctrl_fallbacks = app.controller.fallback_count
        safety_interventions = 0
        safety_near_avoided = 0
        safety_deadlocks = 0
    elif mode == OperatingMode.MID_DRIVEN:
        sl = app.driven_controller.safety_layer
        ctrl_interventions = app.driven_controller.mid_driven_steps
        ctrl_rejected = 0
        ctrl_fallbacks = app.driven_controller.route_reroutes
        safety_interventions = sl.safety_interventions_total
        safety_near_avoided = sl.near_collision_avoided_total
        safety_deadlocks = sl.deadlock_recoveries_total
    else:
        ctrl_interventions = 0
        ctrl_rejected = 0
        ctrl_fallbacks = 0
        safety_interventions = 0
        safety_near_avoided = 0
        safety_deadlocks = 0

    return {
        "mode": mode.value,
        "seed": seed,
        "total_steps": app.engine.clock.step_count,
        "elapsed_time_s": round(total_time, 2),
        "initial_pedestrians": initial_count,
        "evacuated": final_evac,
        "blocked": final_blocked,
        "active": final_active,
        "throughput_peds_per_sec": round(final_evac / total_time, 3) if total_time > 0 else 0.0,
        "mean_speed_mps": round(step_speeds_sum / step_speed_samples, 4) if step_speed_samples > 0 else 0.0,
        "peak_local_density_ped_m2": round(peak_nearby_density, 4),
        "near_collision_proximity_steps": near_collision_events,
        "mean_path_length_m": round(sum(total_dist_by_agent.values()) / len(total_dist_by_agent), 3),
        "interventions_count": ctrl_interventions,
        "safety_interventions_count": safety_interventions,
        "safety_near_collisions_avoided": safety_near_avoided,
        "deadlock_recoveries": safety_deadlocks,
        "invalid_candidates_rejected": ctrl_rejected,
        "fallbacks_count": ctrl_fallbacks,
        "safety_audit": {
            "speed_limit_violations": speed_violations,
            "walkability_or_obstacle_penetrations": walkability_violations,
            "pedestrian_overlap_violations": overlap_violations,
            "teleport_violations": teleport_violations,
            "max_single_step_displacement_m": round(max_step_displacement, 4),
        },
        "evac_step_by_agent": evac_step_by_agent,
        "total_dist_by_agent": total_dist_by_agent,
        "causality_traces": causality_traces,
        "_raw_trajectories": trajectories,
    }


def compute_trajectory_divergence(
    traj_a: Dict[str, List[Tuple[int, float, float, str]]],
    traj_b: Dict[str, List[Tuple[int, float, float, str]]],
) -> Dict[str, Dict[str, float]]:
    """Compute per-agent maximum and mean Euclidean position divergence across common steps."""
    report: Dict[str, Dict[str, float]] = {}
    for aid in sorted(traj_a.keys()):
        pts_a = traj_a[aid]
        pts_b = traj_b[aid]
        common_len = min(len(pts_a), len(pts_b))
        dists = [
            math.hypot(pts_a[i][1] - pts_b[i][1], pts_a[i][2] - pts_b[i][2])
            for i in range(common_len)
        ]
        report[aid] = {
            "max_divergence_m": round(max(dists) if dists else 0.0, 4),
            "mean_divergence_m": round(sum(dists) / len(dists) if dists else 0.0, 4),
            "steps_compared": common_len,
        }
    return report


def evaluate_heldout_mid_vs_cv(shared_mid: Any, orig_raw_trajectories: Dict[str, List[Tuple[int, float, float, str]]]) -> Dict[str, Any]:
    """Compare trained MID (stochastic & mean) vs Constant Velocity on unseen ORIGINAL rollout."""
    from genevac.prediction.baseline import ConstantVelocityPredictor

    cv = ConstantVelocityPredictor()
    traj_obs: Dict[str, Sequence[TrajectoryObservation]] = {}
    for aid, pts in orig_raw_trajectories.items():
        if len(pts) >= 18:
            traj_obs[aid] = tuple(
                TrajectoryObservation(
                    agent_id=aid,
                    step_index=int(st),
                    timestamp=round(st * 0.1, 4),
                    x=float(x),
                    y=float(y),
                )
                for st, x, y, _ in pts
            )

    samples = SampleGenerator(history_length=4, prediction_horizon=12, stride=6).generate_samples(
        traj_obs, include_neighbors=True
    )[:24]
    if not samples:
        return {"evaluated_samples": 0}

    cv_rep = PredictionEvaluator(cv).evaluate_samples(samples)
    mid_stoch = shared_mid.evaluate_stochastic(samples, k_samples=6, seed=42)
    return {
        "evaluated_samples": len(samples),
        "cv_baseline_mean_ade_m": round(cv_rep.mean_ade, 4),
        "cv_baseline_mean_fde_m": round(cv_rep.mean_fde, 4),
        "mid_expected_mean_ade_m": round(mid_stoch["expected_mean_of_k_ade"], 4),
        "mid_oracle_best_of_k_ade_m": round(mid_stoch["oracle_best_of_k_ade"], 4),
        "mid_oracle_best_of_k_fde_m": round(mid_stoch["oracle_best_of_k_fde"], 4),
        "mid_mean_diversity_m": round(mid_stoch["mean_diversity"], 4),
        "mid_point_walkability": round(mid_stoch["point_walkability"], 4),
        "mid_training_provenance": shared_mid.training_provenance,
    }


def main() -> None:
    print("=" * 80)
    print("GENEVAC-SENTINEL: FULL MID INTEGRATION BENCHMARK (ORIGINAL vs ASSISTED vs MID-DRIVEN)")
    print("=" * 80)

    map_path = PROJECT_ROOT / "pavan-part1-reference" / "maps" / "building_main.json"
    base_app = IntegratedSimulationApp(map_path=str(map_path), width=1440, height=860, fps=30)
    shared_mid = base_app.mid_predictor

    # 1. Run Normal Evacuation (All Exits Open) across ALL modes
    print("-> Running Normal Evacuation modes (Original / Pred / Assisted / Driven)...", flush=True)
    res_orig = run_mode_simulation(OperatingMode.ORIGINAL_SIMULATION, seed=42, shared_mid_predictor=shared_mid)
    print(f"   done ORIGINAL: evac={res_orig['evacuated']}/{res_orig['initial_pedestrians']} steps={res_orig['total_steps']}", flush=True)
    res_pred = run_mode_simulation(OperatingMode.MID_PREDICTION_ONLY, seed=42, shared_mid_predictor=shared_mid)
    res_asst_42 = run_mode_simulation(OperatingMode.MID_ASSISTED_EVACUATION, seed=42, shared_mid_predictor=shared_mid)
    print(f"   done MID-ASSISTED: evac={res_asst_42['evacuated']}/{res_asst_42['initial_pedestrians']} steps={res_asst_42['total_steps']}", flush=True)
    res_driven_42 = run_mode_simulation(OperatingMode.MID_DRIVEN, seed=42, shared_mid_predictor=shared_mid)
    print(f"   done MID-DRIVEN (seed=42): evac={res_driven_42['evacuated']}/{res_driven_42['initial_pedestrians']} steps={res_driven_42['total_steps']}", flush=True)
    res_driven_43 = run_mode_simulation(OperatingMode.MID_DRIVEN, seed=43, shared_mid_predictor=shared_mid)
    res_driven_44 = run_mode_simulation(OperatingMode.MID_DRIVEN, seed=44, shared_mid_predictor=shared_mid)
    print("   done MID-DRIVEN (seeds=43,44)", flush=True)

    div_orig_vs_pred = compute_trajectory_divergence(res_orig["_raw_trajectories"], res_pred["_raw_trajectories"])
    div_orig_vs_asst = compute_trajectory_divergence(res_orig["_raw_trajectories"], res_asst_42["_raw_trajectories"])
    div_orig_vs_driven = compute_trajectory_divergence(res_orig["_raw_trajectories"], res_driven_42["_raw_trajectories"])
    heldout_eval = evaluate_heldout_mid_vs_cv(shared_mid, res_orig["_raw_trajectories"])

    print("\n[PROOF 1] SAME STARTING CONDITIONS, DIFFERENT MOVEMENT (Normal Evacuation, Seed=42):")
    print(f"  - ORIGINAL vs MID PREDICTION ONLY max divergence: "
          f"{max(v['max_divergence_m'] for v in div_orig_vs_pred.values()):.6f} m (Expected: 0.000000 m)")
    print(f"  - ORIGINAL vs MID-ASSISTED max divergence: "
          f"{max(v['max_divergence_m'] for v in div_orig_vs_asst.values()):.4f} m")
    print(f"  - ORIGINAL vs MID-DRIVEN per-agent trajectory divergence:")
    for aid, stats in div_orig_vs_driven.items():
        print(
            f"      {aid}: max_div={stats['max_divergence_m']:.4f}m | mean_div={stats['mean_divergence_m']:.4f}m | "
            f"evac_step: orig={res_orig['evac_step_by_agent'][aid]} vs driven={res_driven_42['evac_step_by_agent'][aid]}"
        )

    print("\n[PROOF 2] ACTUAL MID CAUSALITY TRACE (Step 4 & Step 8 for ped_1):")
    for idx in range(min(len(res_orig["causality_traces"]), len(res_driven_42["causality_traces"]))):
        t_orig = res_orig["causality_traces"][idx]
        t_drv = res_driven_42["causality_traces"][idx]
        s = t_drv["step"]
        next_pos_orig = res_orig["_raw_trajectories"]["ped_1"][s + 1][1:3]
        next_pos_drv = res_driven_42["_raw_trajectories"]["ped_1"][s + 1][1:3]
        print(f"  * Step {s} (t={t_drv['timestamp_s']}s):")
        print(f"      Current pos: orig={t_orig['ped_1_position']} | mid_driven={t_drv['ped_1_position']}")
        cd = t_drv.get("controller_decision", {})
        print(f"      MID-DRIVEN Decision: {cd.get('selection_reason')}")
        print(f"      Next engine step (Step {s+1}): orig=({next_pos_orig[0]:.4f}, {next_pos_orig[1]:.4f}) "
              f"vs mid_driven=({next_pos_drv[0]:.4f}, {next_pos_drv[1]:.4f})")

    print("\n[PROOF 3] SAFETY, EFFICIENCY & INTERVENTION VERIFICATION:")
    for label, r in [
        ("ORIGINAL (seed=42)", res_orig),
        ("MID PRED ONLY (seed=42)", res_pred),
        ("MID-ASSISTED (seed=42)", res_asst_42),
        ("MID-DRIVEN (seed=42)", res_driven_42),
        ("MID-DRIVEN (seed=43)", res_driven_43),
        ("MID-DRIVEN (seed=44)", res_driven_44),
    ]:
        sa = r["safety_audit"]
        print(
            f"  - {label:25s}: evac={r['evacuated']}/{r['initial_pedestrians']} in {r['elapsed_time_s']}s | "
            f"v_avg={r['mean_speed_mps']:.2f}m/s | near_col={r['near_collision_proximity_steps']} | "
            f"safety_ovrd={r['safety_interventions_count']} | obs_viol={sa['walkability_or_obstacle_penetrations']} | "
            f"ovlp_viol={sa['pedestrian_overlap_violations']} | teleports={sa['teleport_violations']}"
        )

    # 2. Interactive Multi-Agent Crowd Scenario Comparison
    print("\n[INTERACTIVE CROWD SCENARIO: Dense Merging + Following + Opposing Flow]:", flush=True)
    int_orig = run_mode_simulation(OperatingMode.ORIGINAL_SIMULATION, seed=42, use_interactive_crowd=True, shared_mid_predictor=shared_mid)
    int_asst = run_mode_simulation(OperatingMode.MID_ASSISTED_EVACUATION, seed=42, use_interactive_crowd=True, shared_mid_predictor=shared_mid)
    print("   done Crowd ORIGINAL & MID-ASSISTED", flush=True)
    int_drv = run_mode_simulation(OperatingMode.MID_DRIVEN, seed=42, use_interactive_crowd=True, shared_mid_predictor=shared_mid)
    print("   done Crowd MID-DRIVEN", flush=True)
    div_int = compute_trajectory_divergence(int_orig["_raw_trajectories"], int_drv["_raw_trajectories"])
    for label, r in [("ORIGINAL (Crowd)", int_orig), ("MID-ASSISTED (Crowd)", int_asst), ("MID-DRIVEN (Crowd)", int_drv)]:
        sa = r["safety_audit"]
        print(
            f"  - {label:22s}: evac={r['evacuated']}/{r['initial_pedestrians']} in {r['elapsed_time_s']}s | "
            f"near_col={r['near_collision_proximity_steps']} | safety_ovrd={r['safety_interventions_count']} | "
            f"ovlp_viol={sa['pedestrian_overlap_violations']} | obs_viol={sa['walkability_or_obstacle_penetrations']}"
        )
    print(f"  - Interactive crowd max divergence ORIGINAL vs MID-DRIVEN: "
          f"{max(v['max_divergence_m'] for v in div_int.values()):.4f} m")

    # 3. Disrupted Evacuation (Exit Closure at Step 25)
    print("\n[DISRUPTED SCENARIO: exit_left closed at Step 25 (t=2.5s)]:", flush=True)
    dis_orig = run_mode_simulation(OperatingMode.ORIGINAL_SIMULATION, seed=42, max_steps=350, close_exit_left_at_step=25, shared_mid_predictor=shared_mid)
    dis_asst = run_mode_simulation(OperatingMode.MID_ASSISTED_EVACUATION, seed=42, max_steps=350, close_exit_left_at_step=25, shared_mid_predictor=shared_mid)
    print("   done Disrupted ORIGINAL & MID-ASSISTED", flush=True)
    dis_drv = run_mode_simulation(OperatingMode.MID_DRIVEN, seed=42, max_steps=350, close_exit_left_at_step=25, shared_mid_predictor=shared_mid)
    print("   done Disrupted MID-DRIVEN", flush=True)
    for label, r in [("ORIGINAL (Disrupted)", dis_orig), ("MID-ASSISTED (Disrupted)", dis_asst), ("MID-DRIVEN (Disrupted)", dis_drv)]:
        sa = r["safety_audit"]
        print(
            f"  - {label:26s}: evac={r['evacuated']}/{r['initial_pedestrians']}, blocked={r['blocked']} in {r['elapsed_time_s']}s | "
            f"ovlp_viol={sa['pedestrian_overlap_violations']} | obs_viol={sa['walkability_or_obstacle_penetrations']}"
        )

    print(f"\n[HELD-OUT PREDICTION VS CV BASELINE] samples={heldout_eval.get('evaluated_samples')}: "
          f"CV_ADE={heldout_eval.get('cv_baseline_mean_ade_m')}m vs MID_BestOfK_ADE={heldout_eval.get('mid_oracle_best_of_k_ade_m')}m "
          f"(MID_Mean_ADE={heldout_eval.get('mid_expected_mean_ade_m')}m)")

    for r in (res_orig, res_pred, res_asst_42, res_driven_42, res_driven_43, res_driven_44, int_orig, int_asst, int_drv, dis_orig, dis_asst, dis_drv):
        r.pop("_raw_trajectories", None)

    output_payload = {
        "normal_evacuation_scenario": {
            "original_simulation": res_orig,
            "mid_prediction_only": res_pred,
            "mid_assisted_seed_42": res_asst_42,
            "mid_driven_seed_42": res_driven_42,
            "mid_driven_seed_43": res_driven_43,
            "mid_driven_seed_44": res_driven_44,
            "proof_1_divergence_orig_vs_pred_only": div_orig_vs_pred,
            "proof_1_divergence_orig_vs_mid_assisted": div_orig_vs_asst,
            "proof_1_divergence_orig_vs_mid_driven": div_orig_vs_driven,
        },
        "interactive_crowd_scenario": {
            "original_simulation": int_orig,
            "mid_assisted_evacuation": int_asst,
            "mid_driven": int_drv,
            "divergence_orig_vs_mid_driven": div_int,
        },
        "exit_closure_scenario_step_25": {
            "original_simulation": dis_orig,
            "mid_assisted_evacuation": dis_asst,
            "mid_driven": dis_drv,
        },
        "heldout_prediction_vs_cv": heldout_eval,
    }

    out_file = PROJECT_ROOT / "data" / "experiments" / "comparative_modes_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved full verification report to {out_file}")


if __name__ == "__main__":
    main()
