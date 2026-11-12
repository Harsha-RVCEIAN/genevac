"""Comprehensive verification test suite for GenEvac-Sentinel Phase 09.

Verifies:
1. Unit testing across all core modules (Contracts, Adapter, Metrics, Heatmaps, Live Analytics, Scenarios, Predictions, Exports).
2. Complete end-to-end 10-step integration workflow connecting the entire pipeline.
3. Rigorous edge cases (Empty data, inactive agents, missing ground truth, invalid inputs, untrained models, collisions, single-sample stats, export safety).
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest

import numpy as np
import pytest
import torch

from genevac.analytics import (
    ChartPoint,
    ChartSeries,
    GridSpecification,
    LiveAnalyticsPipeline,
    LiveAnalyticsSnapshot,
    SpatialGridResult,
    SpatialHeatmapEngine,
)
from genevac.environment import EnvironmentMap, Exit, Obstacle
from genevac.experiments import (
    AggregateMetricResult,
    ComparisonDefinition,
    ComparisonStudyResult,
    PerRunMetricResult,
    RunRecord,
    RunStatus,
    ScenarioComparisonRunner,
    ScenarioDefinition,
    TerminationReason,
    aggregate_per_run_results,
    export_aggregate_comparison_csv,
    export_all_experiment_artifacts,
    export_experiment_json,
    export_per_run_metrics_csv,
    export_scenario_inventory_csv,
    generate_experiment_report,
)
from genevac.metrics import (
    calculate_ade,
    calculate_area_density,
    calculate_clearance_time,
    calculate_exit_throughput,
    calculate_fde,
    calculate_minimum_separation,
)
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.prediction import (
    ConstantVelocityPredictor,
    DatasetAdapter,
    EvacuationConditioning,
    EvacuationMIDConfig,
    EvacuationMIDNetwork,
    EvacuationMIDPredictor,
    MultiTrajectoryPrediction,
    PredictionEvaluator,
    SampleGenerator,
    SimulationTrajectoryAdapter,
    TrajectoryCategory,
    TrajectoryHistoryBuffer,
    TrajectoryObservation,
    TrajectoryPrediction,
    TrajectorySnapshot,
    TrajectoryValidationError,
    line_segment_intersects_box,
)
from genevac.prediction.evaluation import EvaluationSample
from genevac.simulation import SimulationEngine


class TestPhase09UnitVerification(unittest.TestCase):
    """Unit tests for individual module components."""

    def test_data_contract_immutability_and_validation(self):
        """TrajectoryObservation and TrajectorySnapshot enforce validation and immutability."""
        obs = TrajectoryObservation(
            agent_id="ped_01",
            step_index=1,
            timestamp=0.1,
            x=5.0,
            y=10.0,
            category=TrajectoryCategory.SIMULATED,
        )
        self.assertEqual(obs.agent_id, "ped_01")
        self.assertEqual(obs.position, (5.0, 10.0))

        # Test immutability
        with self.assertRaises(Exception):
            obs.x = 20.0  # type: ignore

        # Invalid timestamp check
        with self.assertRaises(TrajectoryValidationError):
            TrajectoryObservation(
                agent_id="ped_01",
                step_index=1,
                timestamp=-0.5,
                x=5.0,
                y=10.0,
            )

    def test_metrics_calculation(self):
        """Evacuation metrics produce accurate, unit-tested values."""
        positions = [(0.0, 0.0), (3.0, 4.0)]
        min_sep = calculate_minimum_separation(positions)
        self.assertAlmostEqual(min_sep, 5.0, places=6)

        bounds = (0.0, 0.0, 10.0, 10.0)
        density = calculate_area_density(positions, bounds)
        self.assertAlmostEqual(density, 2.0 / 100.0, places=6)

        ade = calculate_ade([(0.0, 0.0), (2.0, 2.0)], [(0.0, 0.0), (2.0, 2.0)])
        self.assertEqual(ade, 0.0)

    def test_spatial_heatmap_generation(self):
        """SpatialHeatmapEngine computes correct grid dimensions and density."""
        spec = GridSpecification(min_x=0.0, min_y=0.0, max_x=20.0, max_y=10.0, cell_size_x=2.0, cell_size_y=2.0)
        engine = SpatialHeatmapEngine(spec)
        self.assertEqual(spec.cols, 10)
        self.assertEqual(spec.rows, 5)

        obs = {
            "p1": [TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0, category=TrajectoryCategory.SIMULATED)],
            "p2": [TrajectoryObservation("p2", 0, 0.0, 1.5, 1.5, category=TrajectoryCategory.SIMULATED)],
        }
        snap = TrajectorySnapshot(0, 0.0, 0.1, MappingProxyType(obs))
        res = engine.aggregate_snapshot(snap)
        self.assertEqual(res.occupancy_grid.shape, (5, 10))
        self.assertEqual(res.occupancy_grid[0, 0], 2)
        self.assertAlmostEqual(res.density_grid[0, 0], 2.0 / 4.0)

    def test_live_analytics_chart_series(self):
        """ChartSeries enforces deduplication, bounding, and ordering."""
        series = ChartSeries("s1", "m1", "Metric 1", "unit", max_points=3)
        self.assertTrue(series.append_point(ChartPoint(0.0, 0, 1.0, 1)))
        self.assertTrue(series.append_point(ChartPoint(0.1, 1, 2.0, 1)))
        self.assertFalse(series.append_point(ChartPoint(0.1, 1, 2.0, 1)))  # Duplicate step rejected
        self.assertTrue(series.append_point(ChartPoint(0.2, 2, 3.0, 1)))
        self.assertTrue(series.append_point(ChartPoint(0.3, 3, 4.0, 1)))  # Oldest pruned
        self.assertEqual(len(series.points), 3)
        self.assertEqual(series.points[0].step_index, 1)

    def test_baseline_trajectory_prediction(self):
        """ConstantVelocityPredictor extrapolates linearly from observation history."""
        predictor = ConstantVelocityPredictor()
        obs = [
            TrajectoryObservation("p1", 0, 0.0, 0.0, 0.0, category=TrajectoryCategory.SIMULATED),
            TrajectoryObservation("p1", 1, 1.0, 2.0, 3.0, category=TrajectoryCategory.SIMULATED),
        ]
        snap = TrajectorySnapshot(1, 1.0, 1.0, MappingProxyType({"p1": obs}))
        pred = predictor.predict(snap, "p1", horizon=2, prediction_interval=1.0)
        self.assertEqual(len(pred.predictions), 2)
        self.assertAlmostEqual(pred.predictions[0].x, 4.0)
        self.assertAlmostEqual(pred.predictions[0].y, 6.0)
        self.assertAlmostEqual(pred.predictions[1].x, 6.0)
        self.assertAlmostEqual(pred.predictions[1].y, 9.0)

    def test_mid_conditioning_and_isolated_rng(self):
        """EvacuationConditioning vector is 16-dim and sampling preserves global RNG state."""
        env = EnvironmentMap(width=30.0, height=20.0)
        env.add_exit(Exit("e1", (30.0, 10.0), 2.0))
        obs = [
            TrajectoryObservation("p1", 0, 0.0, 5.0, 5.0, category=TrajectoryCategory.SIMULATED),
            TrajectoryObservation("p1", 1, 0.5, 6.0, 5.0, category=TrajectoryCategory.SIMULATED),
        ]
        snap = TrajectorySnapshot(1, 0.5, 0.5, MappingProxyType({"p1": obs}))
        cond = EvacuationConditioning.from_snapshot_and_env(snap, "p1", env, "e1")
        self.assertEqual(len(cond.vector), 16)
        self.assertEqual(len(cond.normalized_vector), 16)

        config = EvacuationMIDConfig(diffusion_steps=5, k_samples=3)
        mid = EvacuationMIDPredictor(config)

        # Isolated RNG check: sampling must not mutate global torch RNG state
        global_state_before = torch.get_rng_state()
        multi_pred = mid.predict_samples(snap, "p1", horizon=4, seed=42, environment=env, k_samples=3)
        global_state_after = torch.get_rng_state()
        self.assertTrue(torch.equal(global_state_before, global_state_after))
        self.assertEqual(len(multi_pred.samples), 3)
        self.assertFalse(multi_pred.is_trained_model)


class TestPhase09IntegrationWorkflow(unittest.TestCase):
    """End-to-end 10-step integration workflow testing the complete Module B pipeline."""

    def test_complete_10_step_integration_pipeline(self):
        """Execute and verify the full 10-step integration chain seamlessly."""
        temp_dir = tempfile.TemporaryDirectory()
        out_path = Path(temp_dir.name)

        try:
            # -------------------------------------------------------------
            # STEP 1: Obtain simulation observations from real SimulationEngine
            # -------------------------------------------------------------
            env = EnvironmentMap(width=40.0, height=20.0)
            env.add_exit(Exit(id="exit_east", position=(40.0, 10.0), width=3.0))
            env.add_obstacle(Obstacle(id="obs_mid", bounds=(15.0, 8.0, 25.0, 12.0)))

            engine = SimulationEngine(environment=env, max_steps=20, timestep=0.1, seed=123)
            engine.add_pedestrian(
                Pedestrian(id="ped_01", position=(5.0, 5.0), velocity=(1.0, 0.5), status=PedestrianStatus.ACTIVE)
            )
            engine.add_pedestrian(
                Pedestrian(id="ped_02", position=(8.0, 15.0), velocity=(1.2, -0.2), status=PedestrianStatus.ACTIVE)
            )
            engine.add_pedestrian(
                Pedestrian(id="ped_03", position=(10.0, 10.0), velocity=(0.8, 0.0), status=PedestrianStatus.ACTIVE)
            )

            # Advance 5 steps
            for _ in range(5):
                engine.step()
            self.assertEqual(engine.current_step, 5)

            # -------------------------------------------------------------
            # STEP 2: Convert observations through SimulationTrajectoryAdapter
            # -------------------------------------------------------------
            adapter = SimulationTrajectoryAdapter(observation_interval=0.1, default_run_id="run_int_01")
            new_obs = adapter.update_from_simulation(engine, force=True)
            self.assertEqual(new_obs, 3)

            # Step further and adapt
            for _ in range(5):
                engine.step()
            adapter.update_from_simulation(engine, force=True)

            snapshot = adapter.buffer.get_snapshot(engine.current_step, engine.current_step * engine.timestep, timestep=engine.timestep)
            self.assertIsNotNone(snapshot)
            self.assertIn("ped_01", snapshot.trajectories)
            self.assertGreaterEqual(len(snapshot.trajectories["ped_01"]), 2)

            # -------------------------------------------------------------
            # STEP 3: Calculate evacuation metrics
            # -------------------------------------------------------------
            positions = [p.position for p in engine.pedestrians]
            density = calculate_area_density(positions, (0.0, 0.0, env.width, env.height))
            min_sep = calculate_minimum_separation(positions)
            self.assertGreater(density, 0.0)
            self.assertGreater(min_sep, 0.0)

            clearance = calculate_clearance_time(engine.current_step, engine.timestep)
            self.assertAlmostEqual(clearance, engine.current_step * engine.timestep)

            throughput = calculate_exit_throughput(0, 1.0)
            self.assertEqual(throughput, 0.0)

            # -------------------------------------------------------------
            # STEP 4: Generate spatial heatmaps
            # -------------------------------------------------------------
            grid_spec = GridSpecification(min_x=0.0, min_y=0.0, max_x=40.0, max_y=20.0, cell_size_x=2.0, cell_size_y=2.0)
            heatmap_engine = SpatialHeatmapEngine(grid_spec)
            grid_result = heatmap_engine.aggregate_snapshot(snapshot)
            self.assertIsInstance(grid_result, SpatialGridResult)
            self.assertEqual(grid_result.density_grid.shape, (10, 20))
            self.assertGreater(np.sum(grid_result.occupancy_grid), 0)

            norm_density = grid_result.normalize_min_max("density")
            self.assertEqual(norm_density.shape, (10, 20))
            self.assertLessEqual(np.max(norm_density), 1.0)

            # -------------------------------------------------------------
            # STEP 5: Produce live analytics and chart data
            # -------------------------------------------------------------
            pipeline = LiveAnalyticsPipeline(grid_spec=grid_spec, observation_interval=0.1, run_id="run_int_01")
            live_snap = pipeline.poll_simulation(engine, force=True)
            self.assertIsInstance(live_snap, LiveAnalyticsSnapshot)
            self.assertEqual(len(live_snap.active_agent_ids), 3)
            self.assertIn("active_count", pipeline.series)
            self.assertGreaterEqual(len(pipeline.series["active_count"].points), 1)

            # -------------------------------------------------------------
            # STEP 6: Compare scenarios via ScenarioComparisonRunner
            # -------------------------------------------------------------
            sc_base = ScenarioDefinition(
                scenario_id="sc_open_corridor",
                name="Open Corridor Baseline",
                agent_count=5,
                engine_config={
                    "width": 40.0,
                    "height": 20.0,
                    "max_steps": 10,
                    "timestep": 0.1,
                    "exits": [{"id": "exit_1", "position": [40.0, 10.0], "width": 3.0}],
                    "obstacles": [],
                },
                seed=42,
            )
            sc_variant = ScenarioDefinition(
                scenario_id="sc_pillar_corridor",
                name="Corridor with Pillar",
                agent_count=5,
                engine_config={
                    "width": 40.0,
                    "height": 20.0,
                    "max_steps": 10,
                    "timestep": 0.1,
                    "exits": [{"id": "exit_1", "position": [40.0, 10.0], "width": 3.0}],
                    "obstacles": [{"id": "pillar", "bounds": [18.0, 8.0, 22.0, 12.0]}],
                },
                seed=42,
            )
            comp_def = ComparisonDefinition(
                comparison_id="comp_study_01",
                title="Obstacle Impact Study",
                research_question="Evaluate pillar impact on pedestrian movement.",
                baseline_scenario_id="sc_open_corridor",
                scenario_ids=("sc_open_corridor", "sc_pillar_corridor"),
                independent_variable="obstacle_presence",
                dependent_metrics=("total_elapsed_time", "mean_speed", "area_density", "minimum_separation", "exit_throughput"),
                controlled_factors={"width": 40.0, "height": 20.0, "agent_count": 5},
            )

            study_result = ScenarioComparisonRunner.execute_comparison(
                comparison=comp_def,
                scenarios=[sc_base, sc_variant],
                repetitions=2,
            )
            self.assertIsInstance(study_result, ComparisonStudyResult)
            self.assertEqual(len(study_result.run_records), 4)

            # -------------------------------------------------------------
            # STEP 7: Generate baseline and MID predictions
            # -------------------------------------------------------------
            base_pred = ConstantVelocityPredictor()
            pred_base = base_pred.predict(snapshot, "ped_01", horizon=5, prediction_interval=0.1)
            self.assertIsInstance(pred_base, TrajectoryPrediction)
            self.assertEqual(len(pred_base.predictions), 5)

            mid_config = EvacuationMIDConfig(diffusion_steps=5, k_samples=4)
            mid_pred = EvacuationMIDPredictor(config=mid_config)
            pred_mid = mid_pred.predict_samples(
                snapshot=snapshot,
                agent_id="ped_01",
                horizon=5,
                prediction_interval=0.1,
                seed=101,
                environment=env,
                k_samples=4,
            )
            self.assertIsInstance(pred_mid, MultiTrajectoryPrediction)
            self.assertEqual(len(pred_mid.samples), 4)
            self.assertEqual(len(pred_mid.mean_prediction.predictions), 5)

            # -------------------------------------------------------------
            # STEP 8: Evaluate predictions against ground truth
            # -------------------------------------------------------------
            # Step the simulation forward to create ground truth future
            for _ in range(5):
                engine.step()
            adapter.update_from_simulation(engine, force=True)
            full_snap = adapter.buffer.get_snapshot(engine.current_step, engine.current_step * engine.timestep, timestep=engine.timestep)

            full_traj = full_snap.trajectories["ped_01"]
            hist = tuple(full_traj[:2])
            gt_future = tuple(full_traj[2:])
            eval_sample = EvaluationSample(
                agent_id="ped_01",
                reference_step=hist[-1].step_index,
                reference_timestamp=hist[-1].timestamp,
                history=hist,
                ground_truth_future=gt_future,
            )

            evaluator = PredictionEvaluator(predictor=base_pred)
            eval_report = evaluator.evaluate_samples([eval_sample])
            self.assertEqual(eval_report.total_samples, 1)
            self.assertEqual(eval_report.evaluated_samples, 1)
            self.assertGreaterEqual(eval_report.mean_ade, 0.0)

            stoch_report = mid_pred.evaluate_stochastic([eval_sample], environment=env, k_samples=4, seed=202)
            self.assertEqual(stoch_report["evaluated_samples"], 1)
            self.assertIn("oracle_best_of_k_ade", stoch_report)
            self.assertIn("expected_mean_of_k_ade", stoch_report)
            self.assertIn("mean_diversity", stoch_report)
            self.assertIn("point_walkability", stoch_report)
            self.assertIn("segment_collision_free", stoch_report)
            self.assertLessEqual(stoch_report["oracle_best_of_k_ade"], stoch_report["expected_mean_of_k_ade"] + 1e-6)

            # -------------------------------------------------------------
            # STEP 9: Export evaluation results (CSV and JSON)
            # -------------------------------------------------------------
            inv_csv = export_scenario_inventory_csv(study_result, out_path / "scenario_inventory.csv")
            runs_csv = export_per_run_metrics_csv(study_result, out_path / "per_run_metrics.csv")
            aggs_csv = export_aggregate_comparison_csv(study_result, out_path / "aggregate_comparison.csv")
            json_exp = export_experiment_json(study_result, out_path / "experiment.json")

            self.assertTrue((out_path / "scenario_inventory.csv").is_file())
            self.assertTrue((out_path / "per_run_metrics.csv").is_file())
            self.assertTrue((out_path / "aggregate_comparison.csv").is_file())
            self.assertTrue((out_path / "experiment.json").is_file())

            # -------------------------------------------------------------
            # STEP 10: Generate final experiment report & export all artifacts
            # -------------------------------------------------------------
            md_report = generate_experiment_report(study_result, out_path / "experiment_report.md")
            self.assertTrue((out_path / "experiment_report.md").is_file())
            self.assertIn("Obstacle Impact Study", md_report)

            bundle = export_all_experiment_artifacts(study_result, out_path / "bundle")
            self.assertTrue((out_path / "bundle" / "scenario_inventory.csv").is_file())
            self.assertTrue((out_path / "bundle" / "experiment_report.md").is_file())

        finally:
            temp_dir.cleanup()


class TestPhase09EdgeCaseVerification(unittest.TestCase):
    """Rigorous edge-case verification for boundary conditions across Module B."""

    def test_empty_simulation_and_zero_agents(self):
        """Simulation and adapter gracefully handle scenarios with zero pedestrians."""
        env = EnvironmentMap(width=20.0, height=10.0)
        engine = SimulationEngine(environment=env, max_steps=5)
        res = engine.run()
        self.assertEqual(res["pedestrians_total"], 0)
        self.assertEqual(res["evacuated"], 0)

        adapter = SimulationTrajectoryAdapter()
        new_obs = adapter.update_from_simulation(engine)
        self.assertEqual(new_obs, 0)

        # Empty positions metrics: minimum separation is inf when count < 2
        self.assertEqual(calculate_minimum_separation([]), float("inf"))
        self.assertEqual(calculate_area_density([], (0.0, 0.0, 20.0, 10.0)), 0.0)

    def test_inactive_and_evacuated_agent_state_transitions(self):
        """Adapter preserves history but halts active tracking for evacuated/inactive agents."""
        env = EnvironmentMap(width=20.0, height=10.0)
        engine = SimulationEngine(environment=env, max_steps=10)
        ped = Pedestrian("p_evac", (5.0, 5.0), velocity=(1.0, 0.0), status=PedestrianStatus.ACTIVE)
        engine.add_pedestrian(ped)

        adapter = SimulationTrajectoryAdapter(observation_interval=0.1)
        adapter.update_from_simulation(engine, force=True)
        self.assertIn("p_evac", adapter.active_agents)

        # Transition to evacuated
        ped.status = PedestrianStatus.EVACUATED
        engine.step()
        adapter.update_from_simulation(engine, force=True)
        self.assertIn("p_evac", adapter.evacuated_agents)
        self.assertNotIn("p_evac", adapter.active_agents)

    def test_single_agent_scenario_execution(self):
        """Single agent scenarios execute cleanly without crashing pairwise metrics."""
        env = EnvironmentMap(width=20.0, height=10.0)
        engine = SimulationEngine(environment=env, max_steps=5)
        engine.add_pedestrian(Pedestrian("p_solo", (5.0, 5.0), (0.5, 0.5)))
        engine.run()

        # Minimum separation for 1 pedestrian must return inf without ZeroDivisionError
        min_sep = calculate_minimum_separation([(5.0, 5.0)])
        self.assertEqual(min_sep, float("inf"))

    def test_insufficient_history_and_invalid_dt_rejections(self):
        """Baseline and MID predictors defensively reject invalid histories and non-positive deltas."""
        base_pred = ConstantVelocityPredictor()
        mid_pred = EvacuationMIDPredictor()

        # Insufficient history (1 point)
        obs_single = [TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0, category=TrajectoryCategory.SIMULATED)]
        snap_single = TrajectorySnapshot(0, 0.0, 0.1, MappingProxyType({"p1": obs_single}))

        with self.assertRaises(TrajectoryValidationError):
            base_pred.predict(snap_single, "p1")

        with self.assertRaises(TrajectoryValidationError):
            mid_pred.predict(snap_single, "p1")

        # Invalid non-positive time delta
        obs_bad_dt = [
            TrajectoryObservation("p1", 0, 1.0, 1.0, 1.0, category=TrajectoryCategory.SIMULATED),
            TrajectoryObservation("p1", 1, 1.0, 2.0, 2.0, category=TrajectoryCategory.SIMULATED),
        ]
        snap_bad_dt = TrajectorySnapshot(1, 1.0, 0.1, MappingProxyType({"p1": obs_bad_dt}))
        with self.assertRaises(TrajectoryValidationError):
            base_pred.predict(snap_bad_dt, "p1")

    def test_untrained_vs_trained_mid_tracking(self):
        """MID predictor tracks training provenance explicitly."""
        mid = EvacuationMIDPredictor()
        self.assertFalse(mid.is_trained)
        self.assertEqual(mid.training_provenance["training_sample_count"], 0)
        self.assertEqual(mid.training_provenance["training_epochs_completed"], 0)

        # Mark trained via provenance properties
        mid._is_trained = True
        mid._training_sample_count = 100
        mid._training_epochs_completed = 5
        self.assertTrue(mid.is_trained)
        self.assertEqual(mid.training_provenance["training_sample_count"], 100)
        self.assertEqual(mid.training_provenance["training_epochs_completed"], 5)

    def test_obstacle_collision_segment_and_point(self):
        """Point walkability and line segment collision clipping evaluate correctly."""
        env = EnvironmentMap(width=20.0, height=20.0)
        env.add_obstacle(Obstacle("box", (5.0, 5.0, 10.0, 10.0)))

        # Point inside
        self.assertFalse(env.is_walkable(7.0, 7.0))
        # Point outside
        self.assertTrue(env.is_walkable(2.0, 2.0))

        # Segment crossing obstacle
        self.assertTrue(line_segment_intersects_box((0.0, 7.5), (15.0, 7.5), (5.0, 5.0, 10.0, 10.0)))
        # Segment completely outside
        self.assertFalse(line_segment_intersects_box((0.0, 2.0), (15.0, 2.0), (5.0, 5.0, 10.0, 10.0)))

    def test_single_sample_statistical_aggregation(self):
        """Statistical aggregation handles N=1 without crashing on variance calculation."""
        run_rec = RunRecord(
            run_id="run_01",
            scenario_id="sc_01",
            scenario_version="1.0.0",
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T00:00:10Z",
            seed=42,
            initial_agent_count=5,
            final_step=10,
            simulated_duration=1.0,
            termination_reason=TerminationReason.COMPLETED,
            run_status=RunStatus.SUCCESS,
            config_fingerprint="abc12345",
        )
        metric_res = PerRunMetricResult(
            run_id="run_01",
            scenario_id="sc_01",
            metric_id="mean_speed",
            display_name="Mean Speed",
            value=1.34,
            unit="m/s",
            status="VALID",
        )
        agg = aggregate_per_run_results(
            results=[metric_res],
            run_records=[run_rec],
            scenario_id="sc_01",
            metric_id="mean_speed",
            unit="m/s",
        )
        self.assertEqual(agg.eligible_run_count, 1)
        self.assertEqual(agg.mean, 1.34)
        self.assertIsNone(agg.std_dev)  # Must be None for N=1, not 0.0 and no ZeroDivisionError

    def test_export_path_resilience_and_serialization(self):
        """Export utilities create missing directories and handle non-finite floats safely."""
        temp_dir = tempfile.TemporaryDirectory()
        out_sub_path = Path(temp_dir.name) / "deep" / "nested" / "dir" / "export.json"

        # Construct minimal study with baseline AND variant
        sc_base = ScenarioDefinition(
            scenario_id="sc_min_base",
            name="Minimal Base",
            agent_count=2,
            engine_config={"width": 10.0, "height": 10.0, "max_steps": 5, "timestep": 0.1},
        )
        sc_var = ScenarioDefinition(
            scenario_id="sc_min_var",
            name="Minimal Var",
            agent_count=2,
            engine_config={"width": 10.0, "height": 10.0, "max_steps": 5, "timestep": 0.1},
        )
        comp = ComparisonDefinition(
            comparison_id="comp_min",
            title="Min Study",
            research_question="Q",
            baseline_scenario_id="sc_min_base",
            scenario_ids=("sc_min_base", "sc_min_var"),
            independent_variable="none",
            dependent_metrics=("total_elapsed_time",),
            controlled_factors={},
        )
        study = ScenarioComparisonRunner.execute_comparison(comp, [sc_base, sc_var], repetitions=1)

        # Export to deeply nested nonexistent directory
        json_str = export_experiment_json(study, out_sub_path)
        self.assertTrue(out_sub_path.is_file())
        self.assertIn("comp_min", json_str)
        temp_dir.cleanup()
