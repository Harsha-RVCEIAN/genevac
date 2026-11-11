import math
import unittest

from genevac.environment import EnvironmentMap
from genevac.metrics import (
    PathCheckStatus,
    aggregate_metric_series,
    calculate_ade,
    calculate_agent_evacuation_times,
    calculate_area_density,
    calculate_clearance_time,
    calculate_evacuation_rate_series,
    calculate_exit_throughput,
    calculate_fde,
    calculate_geometric_path_length,
    calculate_minimum_separation,
    calculate_prediction_coverage,
    validate_evacuation_path,
)
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.simulation import SimulationEngine


class TestMetricsPhase02(unittest.TestCase):
    def test_legacy_metrics_compatibility(self):
        gt = [(0.0, 0.0), (2.0, 2.0)]
        pred = [(0.0, 0.0), (2.0, 2.0)]
        self.assertEqual(calculate_ade(gt, pred), 0.0)
        self.assertEqual(calculate_fde(gt, pred), 0.0)

        gt2 = [(0.0, 0.0), (3.0, 4.0)]
        pred2 = [(0.0, 0.0), (0.0, 0.0)]
        self.assertEqual(calculate_fde(gt2, pred2), 5.0)
        self.assertEqual(calculate_ade(gt2, pred2), 2.5)
        self.assertEqual(calculate_clearance_time(100, 0.1), 10.0)

    def test_trajectory_metrics_validation(self):
        with self.assertRaises(ValueError):
            calculate_ade([], [])
        with self.assertRaises(ValueError):
            calculate_ade([(0.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)])
        with self.assertRaises(ValueError):
            calculate_ade([(float("nan"), 0.0)], [(0.0, 0.0)])

        self.assertEqual(calculate_prediction_coverage(8, 10), 0.8)
        self.assertEqual(calculate_prediction_coverage(0, 10), 0.0)
        self.assertEqual(calculate_prediction_coverage(12, 10), 1.0)

    def test_evacuation_and_crowd_metrics(self):
        # Per-agent evac times
        events = [("p1", 5.2), ("p2", 8.4), ("p1", 9.0)]
        times = calculate_agent_evacuation_times(events)
        self.assertEqual(times["p1"], 5.2)
        self.assertEqual(times["p2"], 8.4)

        # Throughput
        tp = calculate_exit_throughput(exit_counts=15, time_window=10.0)
        self.assertEqual(tp, 1.5)

        # Evacuation rate series
        rates = calculate_evacuation_rate_series([0, 5, 10], total_population=10)
        self.assertEqual(rates, [0.0, 0.5, 1.0])

        # Minimum separation
        pos = [(0.0, 0.0), (3.0, 4.0), (1.0, 0.0)]
        self.assertEqual(calculate_minimum_separation(pos), 1.0)
        self.assertEqual(calculate_minimum_separation([(0.0, 0.0)]), float("inf"))

        # Area density
        density = calculate_area_density(
            agent_positions=[(1.0, 1.0), (2.0, 2.0), (99.0, 99.0)],
            bounds=(0.0, 0.0, 10.0, 10.0),
        )
        self.assertAlmostEqual(density, 2 / 100.0)

    def test_path_length_and_validation(self):
        # Path length
        path = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]
        self.assertEqual(calculate_geometric_path_length(path), 7.0)

        # Full validation on valid path
        summary = validate_evacuation_path(
            path=path,
            world_bounds=(0.0, 0.0, 10.0, 10.0),
            target_exit=(3.0, 4.0),
            exit_tolerance=0.5,
        )
        self.assertTrue(summary.is_valid)
        self.assertEqual(summary.failed_checks, 0)
        self.assertEqual(summary.path_length, 7.0)

        # Path with duplicate points
        dup_path = [(0.0, 0.0), (0.0, 0.0), (1.0, 1.0)]
        dup_summary = validate_evacuation_path(dup_path)
        self.assertFalse(dup_summary.is_valid)
        self.assertGreater(dup_summary.failed_checks, 0)

        # Path crossing obstacle
        # Obstacle box from (1.0, -1.0) to (2.0, 1.0)
        obs_path = [(0.0, 0.0), (3.0, 0.0)]
        obs_summary = validate_evacuation_path(
            path=obs_path,
            obstacles=[(1.0, -1.0, 2.0, 1.0)],
        )
        self.assertFalse(obs_summary.is_valid)
        fail_reasons = [r.reason for r in obs_summary.results if r.status == PathCheckStatus.FAIL]
        self.assertTrue(any("intersects obstacle" in r for r in fail_reasons))

    def test_aggregation(self):
        summary = aggregate_metric_series([1.0, 2.0, 3.0, 4.0, 5.0], "test_metric", "m")
        self.assertEqual(summary.sample_count, 5)
        self.assertEqual(summary.mean, 3.0)
        self.assertEqual(summary.median, 3.0)
        self.assertEqual(summary.min_val, 1.0)
        self.assertEqual(summary.max_val, 5.0)

    def test_engine_non_mutation(self):
        """Proof that metric and route calculations do not mutate or advance simulation engine."""
        env = EnvironmentMap(width=50.0, height=50.0)
        engine = SimulationEngine(environment=env, max_steps=100, timestep=0.1)
        ped = Pedestrian(id="p1", position=(5.0, 5.0), velocity=(1.0, 0.0), status=PedestrianStatus.ACTIVE)
        engine.add_pedestrian(ped)

        initial_step = engine.current_step
        initial_pos = ped.position
        initial_vel = ped.velocity
        initial_status = ped.status
        initial_ped_count = len(engine.pedestrians)

        # Perform metrics calculations
        calculate_clearance_time(engine.current_step, engine.timestep)
        calculate_area_density([p.position for p in engine.pedestrians], (0.0, 0.0, 50.0, 50.0))
        calculate_minimum_separation([p.position for p in engine.pedestrians])
        validate_evacuation_path([(5.0, 5.0), (10.0, 10.0)], world_bounds=(0.0, 0.0, 50.0, 50.0))

        # Assert zero mutation
        self.assertEqual(engine.current_step, initial_step)
        self.assertEqual(ped.position, initial_pos)
        self.assertEqual(ped.velocity, initial_vel)
        self.assertEqual(ped.status, initial_status)
        self.assertEqual(len(engine.pedestrians), initial_ped_count)


if __name__ == "__main__":
    unittest.main()
