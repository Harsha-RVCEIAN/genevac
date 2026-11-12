"""GenEvac-Sentinel Phase 10: Master Experiment Execution and Delivery Script.

Executes:
1. Multi-scenario comparative experiments (Population scaling, Obstacle geometry, Exit bottlenecks).
2. Trajectory prediction benchmark (ConstantVelocity baseline vs. Untrained MID vs. Trained MID).
3. Spatial heatmap aggregation and live analytics verification.
4. Complete machine-readable export of all artifacts (CSV, JSON, Markdown).
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genevac.analytics import (
    GridSpecification,
    LiveAnalyticsPipeline,
    SpatialHeatmapEngine,
)
from genevac.environment import EnvironmentMap, Exit, Obstacle
from genevac.experiments import (
    ComparisonDefinition,
    ScenarioComparisonRunner,
    ScenarioDefinition,
    export_all_experiment_artifacts,
)
from genevac.metrics import (
    calculate_ade,
    calculate_area_density,
    calculate_fde,
    calculate_minimum_separation,
)
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.prediction import (
    ConstantVelocityPredictor,
    EvacuationMIDConfig,
    EvacuationMIDPredictor,
    PredictionEvaluator,
    SimulationTrajectoryAdapter,
    TrajectoryCategory,
    TrajectoryObservation,
    TrajectorySnapshot,
    line_segment_intersects_box,
)
from genevac.prediction.evaluation import EvaluationSample
from genevac.simulation import SimulationEngine


def run_comparative_studies(base_out_dir: Path) -> dict:
    """Execute 3 formal scenario comparison studies and export artifacts."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE 1: Multi-Scenario Comparative Evacuation Studies")
    print("=" * 70)

    results_summary = {}

    # -------------------------------------------------------------------------
    # Study 1: Population Density & Scaling
    # -------------------------------------------------------------------------
    print("\n[Study 1/3] Executing Population Scaling Study (5 vs 15 vs 30 agents)...")
    sc_pop5 = ScenarioDefinition(
        scenario_id="pop_05_baseline",
        name="Corridor Low Density (5 peds)",
        version="1.0.0",
        description="Corridor with 5 evacuees",
        agent_count=5,
        engine_config={
            "width": 50.0,
            "height": 20.0,
            "max_steps": 40,
            "timestep": 0.1,
            "exits": [{"id": "main_exit", "position": [50.0, 10.0], "width": 3.0}],
            "obstacles": [],
        },
        seed=101,
    )
    sc_pop15 = ScenarioDefinition(
        scenario_id="pop_15_medium",
        name="Corridor Medium Density (15 peds)",
        version="1.0.0",
        description="Corridor with 15 evacuees",
        agent_count=15,
        engine_config={
            "width": 50.0,
            "height": 20.0,
            "max_steps": 40,
            "timestep": 0.1,
            "exits": [{"id": "main_exit", "position": [50.0, 10.0], "width": 3.0}],
            "obstacles": [],
        },
        seed=101,
    )
    sc_pop30 = ScenarioDefinition(
        scenario_id="pop_30_high",
        name="Corridor High Density (30 peds)",
        version="1.0.0",
        description="Corridor with 30 evacuees",
        agent_count=30,
        engine_config={
            "width": 50.0,
            "height": 20.0,
            "max_steps": 40,
            "timestep": 0.1,
            "exits": [{"id": "main_exit", "position": [50.0, 10.0], "width": 3.0}],
            "obstacles": [],
        },
        seed=101,
    )
    comp_pop = ComparisonDefinition(
        comparison_id="study_pop_scaling",
        title="Pedestrian Population Scaling and Density Impact",
        research_question="Evaluate the effect of pedestrian population scaling on crowd density and minimum separation.",
        baseline_scenario_id="pop_05_baseline",
        scenario_ids=("pop_05_baseline", "pop_15_medium", "pop_30_high"),
        independent_variable="pedestrian_population_count",
        dependent_metrics=("total_elapsed_time", "mean_speed", "area_density", "minimum_separation", "exit_throughput"),
        controlled_factors={"width": 50.0, "height": 20.0, "timestep": 0.1, "max_steps": 40},
    )

    study_pop = ScenarioComparisonRunner.execute_comparison(comp_pop, [sc_pop5, sc_pop15, sc_pop30], repetitions=2)
    export_all_experiment_artifacts(study_pop, base_out_dir / "study_1_population_scaling")
    results_summary["population_scaling"] = {
        "pop_05_density": study_pop.aggregates[("pop_05_baseline", "area_density")].mean,
        "pop_15_density": study_pop.aggregates[("pop_15_medium", "area_density")].mean,
        "pop_30_density": study_pop.aggregates[("pop_30_high", "area_density")].mean,
        "pop_05_min_sep": study_pop.aggregates[("pop_05_baseline", "minimum_separation")].mean,
        "pop_15_min_sep": study_pop.aggregates[("pop_15_medium", "minimum_separation")].mean,
        "pop_30_min_sep": study_pop.aggregates[("pop_30_high", "minimum_separation")].mean,
    }

    # -------------------------------------------------------------------------
    # Study 2: Obstacle Detour & Geometry
    # -------------------------------------------------------------------------
    print("\n[Study 2/3] Executing Obstacle Geometry Study (Clear vs Central Pillar vs Dual Obstacles)...")
    sc_clear = ScenarioDefinition(
        scenario_id="obs_clear_baseline",
        name="Clear Hallway Baseline",
        version="1.0.0",
        description="Unobstructed hallway",
        agent_count=15,
        engine_config={
            "width": 50.0,
            "height": 20.0,
            "max_steps": 40,
            "timestep": 0.1,
            "exits": [{"id": "main_exit", "position": [50.0, 10.0], "width": 3.0}],
            "obstacles": [],
        },
        seed=202,
    )
    sc_pillar = ScenarioDefinition(
        scenario_id="obs_central_pillar",
        name="Hallway with Central Pillar",
        version="1.0.0",
        description="Hallway with large central obstacle pillar",
        agent_count=15,
        engine_config={
            "width": 50.0,
            "height": 20.0,
            "max_steps": 40,
            "timestep": 0.1,
            "exits": [{"id": "main_exit", "position": [50.0, 10.0], "width": 3.0}],
            "obstacles": [{"id": "pillar_center", "bounds": [20.0, 6.0, 30.0, 14.0]}],
        },
        seed=202,
    )
    sc_dual = ScenarioDefinition(
        scenario_id="obs_dual_pillars",
        name="Hallway with Staggered Dual Pillars",
        version="1.0.0",
        description="Hallway with two staggered pillars forming a chicane",
        agent_count=15,
        engine_config={
            "width": 50.0,
            "height": 20.0,
            "max_steps": 40,
            "timestep": 0.1,
            "exits": [{"id": "main_exit", "position": [50.0, 10.0], "width": 3.0}],
            "obstacles": [
                {"id": "pillar_north", "bounds": [15.0, 10.0, 22.0, 18.0]},
                {"id": "pillar_south", "bounds": [28.0, 2.0, 35.0, 10.0]},
            ],
        },
        seed=202,
    )
    comp_obs = ComparisonDefinition(
        comparison_id="study_obstacle_geometry",
        title="Obstacle Geometry and Spatial Constriction Analysis",
        research_question="Evaluate the effect of geometric obstacles on hallway pedestrian flow.",
        baseline_scenario_id="obs_clear_baseline",
        scenario_ids=("obs_clear_baseline", "obs_central_pillar", "obs_dual_pillars"),
        independent_variable="obstacle_configuration",
        dependent_metrics=("total_elapsed_time", "mean_speed", "area_density", "minimum_separation", "exit_throughput"),
        controlled_factors={"width": 50.0, "height": 20.0, "agent_count": 15, "timestep": 0.1, "max_steps": 40},
    )

    study_obs = ScenarioComparisonRunner.execute_comparison(comp_obs, [sc_clear, sc_pillar, sc_dual], repetitions=2)
    export_all_experiment_artifacts(study_obs, base_out_dir / "study_2_obstacle_geometry")
    results_summary["obstacle_geometry"] = {
        "clear_speed": study_obs.aggregates[("obs_clear_baseline", "mean_speed")].mean,
        "pillar_speed": study_obs.aggregates[("obs_central_pillar", "mean_speed")].mean,
        "dual_speed": study_obs.aggregates[("obs_dual_pillars", "mean_speed")].mean,
        "clear_min_sep": study_obs.aggregates[("obs_clear_baseline", "minimum_separation")].mean,
        "pillar_min_sep": study_obs.aggregates[("obs_central_pillar", "minimum_separation")].mean,
        "dual_min_sep": study_obs.aggregates[("obs_dual_pillars", "minimum_separation")].mean,
    }

    # -------------------------------------------------------------------------
    # Study 3: Exit Configuration & Bottlenecks
    # -------------------------------------------------------------------------
    print("\n[Study 3/3] Executing Exit Configuration Study (Standard 3m vs Bottleneck 1m vs Dual Exits)...")
    sc_exit_std = ScenarioDefinition(
        scenario_id="exit_std_baseline",
        name="Standard Exit Baseline (3.0m)",
        version="1.0.0",
        description="Single standard 3.0m exit door",
        agent_count=20,
        engine_config={
            "width": 50.0,
            "height": 20.0,
            "max_steps": 40,
            "timestep": 0.1,
            "exits": [{"id": "exit_east", "position": [50.0, 10.0], "width": 3.0}],
            "obstacles": [],
        },
        seed=303,
    )
    sc_exit_bn = ScenarioDefinition(
        scenario_id="exit_narrow_bottleneck",
        name="Narrow Exit Bottleneck (1.0m)",
        version="1.0.0",
        description="Single narrow bottleneck 1.0m exit door",
        agent_count=20,
        engine_config={
            "width": 50.0,
            "height": 20.0,
            "max_steps": 40,
            "timestep": 0.1,
            "exits": [{"id": "exit_east", "position": [50.0, 10.0], "width": 1.0}],
            "obstacles": [],
        },
        seed=303,
    )
    sc_exit_dual = ScenarioDefinition(
        scenario_id="exit_dual_portals",
        name="Dual Exit Portals (2x 2.5m)",
        version="1.0.0",
        description="Dual exit doors at north-east and south-east",
        agent_count=20,
        engine_config={
            "width": 50.0,
            "height": 20.0,
            "max_steps": 40,
            "timestep": 0.1,
            "exits": [
                {"id": "exit_ne", "position": [50.0, 5.0], "width": 2.5},
                {"id": "exit_se", "position": [50.0, 15.0], "width": 2.5},
            ],
            "obstacles": [],
        },
        seed=303,
    )
    comp_exit = ComparisonDefinition(
        comparison_id="study_exit_configuration",
        title="Exit Width and Dual Portal Bottleneck Evaluation",
        research_question="Evaluate the impact of exit width constriction and dual exit availability on evacuation flow.",
        baseline_scenario_id="exit_std_baseline",
        scenario_ids=("exit_std_baseline", "exit_narrow_bottleneck", "exit_dual_portals"),
        independent_variable="exit_portal_configuration",
        dependent_metrics=("total_elapsed_time", "mean_speed", "area_density", "minimum_separation", "exit_throughput"),
        controlled_factors={"width": 50.0, "height": 20.0, "agent_count": 20, "timestep": 0.1, "max_steps": 40},
    )

    study_exit = ScenarioComparisonRunner.execute_comparison(comp_exit, [sc_exit_std, sc_exit_bn, sc_exit_dual], repetitions=2)
    export_all_experiment_artifacts(study_exit, base_out_dir / "study_3_exit_bottlenecks")
    results_summary["exit_configuration"] = {
        "std_density": study_exit.aggregates[("exit_std_baseline", "area_density")].mean,
        "bottleneck_density": study_exit.aggregates[("exit_narrow_bottleneck", "area_density")].mean,
        "dual_density": study_exit.aggregates[("exit_dual_portals", "area_density")].mean,
        "std_min_sep": study_exit.aggregates[("exit_std_baseline", "minimum_separation")].mean,
        "bottleneck_min_sep": study_exit.aggregates[("exit_narrow_bottleneck", "minimum_separation")].mean,
        "dual_min_sep": study_exit.aggregates[("exit_dual_portals", "minimum_separation")].mean,
    }

    print("All 3 scenario comparison studies executed and exported successfully.")
    return results_summary


def run_prediction_benchmark(base_out_dir: Path) -> dict:
    """Execute trajectory prediction benchmark comparing Baseline vs Untrained MID vs Trained MID."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE 2: Trajectory Prediction Benchmark (Baseline vs MID)")
    print("=" * 70)

    # 1. Generate realistic multi-agent simulation trajectories
    env = EnvironmentMap(width=50.0, height=25.0)
    env.add_exit(Exit(id="exit_east", position=(50.0, 12.5), width=3.0))
    env.add_obstacle(Obstacle(id="obs_pillar", bounds=(20.0, 8.0, 30.0, 17.0)))

    engine = SimulationEngine(environment=env, max_steps=50, timestep=0.1, seed=42)

    # Add 12 pedestrians in various positions with tactical velocities
    ped_configs = [
        ("ped_01", (5.0, 5.0), (1.2, 0.2)),
        ("ped_02", (6.0, 8.0), (1.1, 0.3)),
        ("ped_03", (5.0, 12.0), (1.3, -0.1)),
        ("ped_04", (8.0, 15.0), (1.0, -0.3)),
        ("ped_05", (7.0, 20.0), (1.2, -0.4)),
        ("ped_06", (12.0, 6.0), (1.4, 0.1)),
        ("ped_07", (14.0, 11.0), (0.9, 0.4)),
        ("ped_08", (10.0, 18.0), (1.3, -0.2)),
        ("ped_09", (15.0, 4.0), (1.1, 0.2)),
        ("ped_10", (16.0, 21.0), (1.2, -0.3)),
        ("ped_11", (3.0, 10.0), (1.3, 0.0)),
        ("ped_12", (4.0, 16.0), (1.1, -0.1)),
    ]
    for pid, pos, vel in ped_configs:
        engine.add_pedestrian(Pedestrian(id=pid, position=pos, velocity=vel, status=PedestrianStatus.ACTIVE))

    adapter = SimulationTrajectoryAdapter(observation_interval=0.1, default_run_id="pred_bench_run")

    # Step simulation and record observations
    for _ in range(30):
        engine.step()
        adapter.update_from_simulation(engine, force=True)

    full_snap = adapter.buffer.get_snapshot(engine.current_step, engine.current_step * engine.timestep, timestep=engine.timestep)
    print(f"Generated simulation trajectory snapshot across {len(full_snap.trajectories)} agents.")

    # 2. Construct Evaluation Samples (T_obs = 6 steps, Horizon H = 10 steps)
    history_len = 6
    horizon_len = 10
    eval_samples: list[EvaluationSample] = []
    training_samples: list[EvaluationSample] = []

    for idx, (pid, traj) in enumerate(sorted(full_snap.trajectories.items())):
        if len(traj) >= history_len + horizon_len:
            # First split for training
            hist_train = tuple(traj[:history_len])
            gt_train = tuple(traj[history_len : history_len + horizon_len])
            sample_train = EvaluationSample(
                agent_id=pid,
                reference_step=hist_train[-1].step_index,
                reference_timestamp=hist_train[-1].timestamp,
                history=hist_train,
                ground_truth_future=gt_train,
            )
            training_samples.append(sample_train)

            # Shifted window for test evaluation (sliding window)
            offset = 4
            hist_test = tuple(traj[offset : offset + history_len])
            gt_test = tuple(traj[offset + history_len : offset + history_len + horizon_len])
            sample_test = EvaluationSample(
                agent_id=pid,
                reference_step=hist_test[-1].step_index,
                reference_timestamp=hist_test[-1].timestamp,
                history=hist_test,
                ground_truth_future=gt_test,
            )
            eval_samples.append(sample_test)

    print(f"Constructed {len(training_samples)} training samples and {len(eval_samples)} evaluation test samples.")

    # 3. Model 1: Constant Velocity Baseline
    print("\n--- Evaluating Model 1: ConstantVelocityPredictor (Kinematic Baseline) ---")
    base_pred = ConstantVelocityPredictor()
    evaluator_base = PredictionEvaluator(predictor=base_pred)
    base_report = evaluator_base.evaluate_samples(eval_samples)

    # Compute baseline physical walkability and segment collision
    base_point_walkable = 0
    base_total_points = 0
    base_coll_free_segments = 0
    base_total_segments = 0

    for s in eval_samples:
        snap_s = s.to_snapshot()
        p = base_pred.predict(snap_s, s.agent_id, horizon=horizon_len, prediction_interval=0.1)
        coords = [pred_pos.position for pred_pos in p.predictions]
        # Point walkability
        for pt in coords:
            base_total_points += 1
            if env.is_walkable(pt[0], pt[1]):
                base_point_walkable += 1
        # Segment collision
        traj_points = [s.history[-1].position] + coords
        collided = False
        for i in range(len(traj_points) - 1):
            base_total_segments += 1
            for obs in env.obstacles:
                if line_segment_intersects_box(traj_points[i], traj_points[i + 1], obs.bounds):
                    collided = True
                    break
        if not collided:
            base_coll_free_segments += 1

    base_walk_rate = base_point_walkable / max(base_total_points, 1)
    base_coll_rate = base_coll_free_segments / max(len(eval_samples), 1)

    print(f"Baseline Mean ADE:               {base_report.mean_ade:.4f} m")
    print(f"Baseline Mean FDE:               {base_report.mean_fde:.4f} m")
    print(f"Baseline Point Walkability:      {base_walk_rate * 100:.1f}%")
    print(f"Baseline Segment Collision-Free: {base_coll_rate * 100:.1f}%")

    # 4. Model 2: Untrained Evacuation MID (K=10 samples)
    print("\n--- Evaluating Model 2: EvacuationMIDPredictor (Untrained Prototype, K=10) ---")
    mid_cfg = EvacuationMIDConfig(diffusion_steps=10, horizon=horizon_len, k_samples=10, learning_rate=1e-3)
    untrained_mid = EvacuationMIDPredictor(config=mid_cfg)

    untrained_stoch = untrained_mid.evaluate_stochastic(
        samples=eval_samples,
        environment=env,
        k_samples=10,
        seed=42,
    )
    print(f"Untrained MID Oracle Best-of-K ADE: {untrained_stoch['oracle_best_of_k_ade']:.4f} m")
    print(f"Untrained MID Expected Mean ADE:    {untrained_stoch['expected_mean_of_k_ade']:.4f} m")
    print(f"Untrained MID Oracle Best-of-K FDE: {untrained_stoch['oracle_best_of_k_fde']:.4f} m")
    print(f"Untrained MID Sample Diversity:     {untrained_stoch['mean_diversity']:.4f} m")
    print(f"Untrained MID Point Walkability:    {untrained_stoch['point_walkability'] * 100:.1f}%")
    print(f"Untrained MID Collision-Free:       {untrained_stoch['segment_collision_free'] * 100:.1f}%")

    # 5. Model 3: Trained Evacuation MID (5 epochs with loss masking)
    print("\n--- Training Model 3: EvacuationMIDPredictor (5 Epochs with Loss Masking) ---")
    trained_mid = EvacuationMIDPredictor(config=mid_cfg)
    fit_loss = trained_mid.fit(training_samples, epochs=5, lr=2e-3, batch_size=4)
    final_loss = fit_loss[-1] if isinstance(fit_loss, list) else fit_loss
    print(f"Training completed. Final Epoch Loss: {final_loss:.6f}")
    print(f"Training provenance: {trained_mid.training_provenance}")

    print("\n--- Evaluating Model 3: EvacuationMIDPredictor (Trained Model, K=10) ---")
    trained_stoch = trained_mid.evaluate_stochastic(
        samples=eval_samples,
        environment=env,
        k_samples=10,
        seed=42,
    )
    print(f"Trained MID Oracle Best-of-K ADE:   {trained_stoch['oracle_best_of_k_ade']:.4f} m")
    print(f"Trained MID Expected Mean ADE:      {trained_stoch['expected_mean_of_k_ade']:.4f} m")
    print(f"Trained MID Oracle Best-of-K FDE:   {trained_stoch['oracle_best_of_k_fde']:.4f} m")
    print(f"Trained MID Sample Diversity:       {trained_stoch['mean_diversity']:.4f} m")
    print(f"Trained MID Point Walkability:      {trained_stoch['point_walkability'] * 100:.1f}%")
    print(f"Trained MID Collision-Free:         {trained_stoch['segment_collision_free'] * 100:.1f}%")

    # Compile benchmark data
    benchmark_data = {
        "evaluation_samples_count": len(eval_samples),
        "prediction_horizon": horizon_len,
        "history_steps": history_len,
        "baseline_constant_velocity": {
            "mean_ade_m": round(base_report.mean_ade, 6),
            "mean_fde_m": round(base_report.mean_fde, 6),
            "median_ade_m": round(base_report.median_ade, 6),
            "median_fde_m": round(base_report.median_fde, 6),
            "point_walkability_pct": round(base_walk_rate * 100.0, 2),
            "segment_collision_free_pct": round(base_coll_rate * 100.0, 2),
            "is_stochastic": False,
        },
        "evacuation_mid_untrained": {
            "is_trained": False,
            "oracle_best_of_k_ade_m": round(untrained_stoch["oracle_best_of_k_ade"], 6),
            "oracle_best_of_k_fde_m": round(untrained_stoch["oracle_best_of_k_fde"], 6),
            "expected_mean_of_k_ade_m": round(untrained_stoch["expected_mean_of_k_ade"], 6),
            "sample_diversity_m": round(untrained_stoch["mean_diversity"], 6),
            "point_walkability_pct": round(untrained_stoch["point_walkability"] * 100.0, 2),
            "segment_collision_free_pct": round(untrained_stoch["segment_collision_free"] * 100.0, 2),
            "k_samples": 10,
        },
        "evacuation_mid_trained": {
            "is_trained": True,
            "training_epochs": 5,
            "final_train_loss": round(final_loss, 6),
            "oracle_best_of_k_ade_m": round(trained_stoch["oracle_best_of_k_ade"], 6),
            "oracle_best_of_k_fde_m": round(trained_stoch["oracle_best_of_k_fde"], 6),
            "expected_mean_of_k_ade_m": round(trained_stoch["expected_mean_of_k_ade"], 6),
            "sample_diversity_m": round(trained_stoch["mean_diversity"], 6),
            "point_walkability_pct": round(trained_stoch["point_walkability"] * 100.0, 2),
            "segment_collision_free_pct": round(trained_stoch["segment_collision_free"] * 100.0, 2),
            "k_samples": 10,
        },
    }

    # Save benchmark JSON
    json_path = base_out_dir / "trajectory_prediction_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    # Save benchmark CSV table
    csv_path = base_out_dir / "trajectory_prediction_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model",
            "Trained",
            "Samples_K",
            "Oracle_Best_ADE_m",
            "Expected_Mean_ADE_m",
            "Oracle_Best_FDE_m",
            "Diversity_m",
            "Point_Walkability_Pct",
            "Segment_Collision_Free_Pct",
        ])
        writer.writerow([
            "ConstantVelocity (Baseline)",
            "N/A",
            1,
            f"{base_report.mean_ade:.4f}",
            f"{base_report.mean_ade:.4f}",
            f"{base_report.mean_fde:.4f}",
            "0.0000",
            f"{base_walk_rate * 100:.1f}",
            f"{base_coll_rate * 100:.1f}",
        ])
        writer.writerow([
            "Evacuation MID (Untrained)",
            "False",
            10,
            f"{untrained_stoch['oracle_best_of_k_ade']:.4f}",
            f"{untrained_stoch['expected_mean_of_k_ade']:.4f}",
            f"{untrained_stoch['oracle_best_of_k_fde']:.4f}",
            f"{untrained_stoch['mean_diversity']:.4f}",
            f"{untrained_stoch['point_walkability'] * 100:.1f}",
            f"{untrained_stoch['segment_collision_free'] * 100:.1f}",
        ])
        writer.writerow([
            "Evacuation MID (Trained)",
            "True",
            10,
            f"{trained_stoch['oracle_best_of_k_ade']:.4f}",
            f"{trained_stoch['expected_mean_of_k_ade']:.4f}",
            f"{trained_stoch['oracle_best_of_k_fde']:.4f}",
            f"{trained_stoch['mean_diversity']:.4f}",
            f"{trained_stoch['point_walkability'] * 100:.1f}",
            f"{trained_stoch['segment_collision_free'] * 100:.1f}",
        ])

    print(f"Saved benchmark outputs to {json_path} and {csv_path}")
    return benchmark_data


def run_analytics_profiling(base_out_dir: Path) -> dict:
    """Execute spatial heatmap generation and live analytics verification."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE 3: Spatial Heatmap & Live Analytics Profiling")
    print("=" * 70)

    # 1. Setup Environment and Simulation
    env = EnvironmentMap(width=50.0, height=20.0)
    env.add_exit(Exit("exit_1", (50.0, 10.0), 3.0))
    env.add_obstacle(Obstacle("obs_1", (20.0, 5.0, 28.0, 15.0)))

    engine = SimulationEngine(environment=env, max_steps=30, timestep=0.1, seed=555)
    for i in range(20):
        engine.add_pedestrian(
            Pedestrian(
                id=f"agent_{i+1:02d}",
                position=(5.0 + (i % 5) * 3.0, 3.0 + (i // 5) * 4.0),
                velocity=(1.0 + (i % 3) * 0.1, 0.0),
                status=PedestrianStatus.ACTIVE,
            )
        )

    # 2. Setup Spatial Grid Specification (25 cols x 10 rows, cell size 2m x 2m)
    grid_spec = GridSpecification(min_x=0.0, min_y=0.0, max_x=50.0, max_y=20.0, cell_size_x=2.0, cell_size_y=2.0)
    heatmap_engine = SpatialHeatmapEngine(grid_spec)

    adapter = SimulationTrajectoryAdapter(observation_interval=0.1, default_run_id="heatmap_run")
    pipeline = LiveAnalyticsPipeline(grid_spec=grid_spec, observation_interval=0.1, run_id="live_run_01")

    # Step and aggregate
    for step_idx in range(25):
        engine.step()
        adapter.update_from_simulation(engine, force=True)
        pipeline.poll_simulation(engine, force=True)

    snap = adapter.buffer.get_snapshot(engine.current_step, engine.current_step * engine.timestep, timestep=engine.timestep)
    grid_res = heatmap_engine.aggregate_snapshot(snap, run_id="heatmap_run")

    norm_density = grid_res.normalize_min_max("density")

    analytics_data = {
        "grid_specification": {
            "width_m": grid_spec.width,
            "height_m": grid_spec.height,
            "cols": grid_spec.cols,
            "rows": grid_spec.rows,
            "cell_size_m": [grid_spec.cell_size_x, grid_spec.cell_size_y],
            "cell_area_m2": grid_spec.cell_area,
        },
        "spatial_grid_results": {
            "total_agents_processed": grid_res.total_agents_processed,
            "accepted_agents": grid_res.accepted_agents,
            "rejected_agents": grid_res.rejected_agents,
            "max_occupancy_in_single_cell": int(np.max(grid_res.occupancy_grid)),
            "max_density_ped_per_m2": round(float(np.max(grid_res.density_grid)), 4),
            "mean_active_density_ped_per_m2": round(float(np.mean(grid_res.density_grid[grid_res.density_grid > 0])), 4),
            "min_max_normalized_max": round(float(np.max(norm_density)), 4),
        },
        "live_chart_series": {
            series_name: len(s.points) for series_name, s in pipeline.series.items()
        },
    }

    json_path = base_out_dir / "spatial_heatmap_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analytics_data, f, indent=2)

    print(f"Heatmap Grid shape: {grid_res.occupancy_grid.shape}")
    print(f"Max single cell occupancy: {analytics_data['spatial_grid_results']['max_occupancy_in_single_cell']} peds")
    print(f"Max density: {analytics_data['spatial_grid_results']['max_density_ped_per_m2']} ped/m²")
    print(f"Live chart series points: {analytics_data['live_chart_series']}")
    print(f"Saved analytics summary to {json_path}")
    return analytics_data


def main():
    print("=" * 70)
    print("GenEvac-Sentinel: Phase 10 Experiments and Delivery Execution")
    print("=" * 70)

    out_dir = Path("data/experiments/phase10")
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    comp_results = run_comparative_studies(out_dir)
    pred_results = run_prediction_benchmark(out_dir)
    analytics_results = run_analytics_profiling(out_dir)
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"PHASE 10 EXPERIMENTS COMPLETE (Total Duration: {elapsed:.2f}s)")
    print(f"All exported artifacts persisted to: {out_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
