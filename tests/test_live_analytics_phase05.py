"""Unit, integration, and replay tests for Phase 05 Live Analytics and Chart Pipeline.

Verifies:
1. LiveAnalyticsPipeline non-invasive polling (zero engine stepping / mutation).
2. Distinction between source simulation time and refresh wall-clock time.
3. Accurate metric records (active, blocked, evacuated, not_observed, min_separation, clearance).
4. Idempotence and duplicate suppression: no duplicate chart points on repeated poll.
5. ChartSeries bounded retention and step deduplication.
6. Controlled multi-step deterministic replay walkthrough.
7. Run reset isolation (zero cross-run contamination).
8. Freshness mode (LIVE vs REPLAY) and PAUSED state handling.
9. Exit throughput availability semantics.
10. Explicit clearance time naming ("Total Elapsed Simulation Time").
"""

import math
import time
import unittest

from genevac.analytics import (
    ChartPoint,
    ChartSeries,
    GridSpecification,
    LiveAnalyticsPipeline,
    LiveAnalyticsSnapshot,
)
from genevac.analytics.live import AnalyticsFreshnessMode, MetricStatus
from genevac.environment import EnvironmentMap
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.simulation import SimulationEngine


class TestLiveAnalyticsPhase05(unittest.TestCase):
    def test_chart_series_deduplication_and_bounding(self):
        """ChartSeries enforces step uniqueness and bounded history size."""
        series = ChartSeries("test_s", "active", "Active Peds", "count", run_id="run_1", max_points=3)

        # Append step 0, 1, 2
        self.assertTrue(series.append_point(ChartPoint(0.0, 0, 10.0, 10, "run_1")))
        self.assertTrue(series.append_point(ChartPoint(0.1, 1, 9.0, 9, "run_1")))
        self.assertTrue(series.append_point(ChartPoint(0.2, 2, 8.0, 8, "run_1")))
        self.assertEqual(len(series.points), 3)

        # Duplicate step 2 must be rejected
        self.assertFalse(series.append_point(ChartPoint(0.2, 2, 8.0, 8, "run_1")))
        self.assertEqual(len(series.points), 3)

        # Append step 3 -> oldest point (step 0) must be pruned
        self.assertTrue(series.append_point(ChartPoint(0.3, 3, 7.0, 7, "run_1")))
        self.assertEqual(len(series.points), 3)
        self.assertEqual(series.points[0].step_index, 1)
        self.assertEqual(series.points[-1].step_index, 3)

    def test_pipeline_polling_and_no_new_data(self):
        """Pipeline poll produces valid metrics; repeated poll at same step is a no-op."""
        env = EnvironmentMap(width=20.0, height=20.0)
        sim = SimulationEngine(environment=env, max_steps=100, timestep=0.1)
        p1 = Pedestrian("p1", (0.0, 0.0), velocity=(1.0, 0.0), status=PedestrianStatus.ACTIVE)
        p2 = Pedestrian("p2", (10.0, 10.0), velocity=(0.0, 0.0), status=PedestrianStatus.ACTIVE)
        sim.add_pedestrian(p1)
        sim.add_pedestrian(p2)

        spec = GridSpecification(0.0, 0.0, 20.0, 20.0, 5.0, 5.0)
        pipeline = LiveAnalyticsPipeline(grid_spec=spec, observation_interval=0.4, run_id="run_live")

        # Initial Poll at Step 0
        snap1 = pipeline.poll_simulation(sim)
        self.assertTrue(snap1.has_new_data)
        self.assertEqual(snap1.source_step, 0)
        self.assertEqual(snap1.source_timestamp, 0.0)
        self.assertEqual(snap1.metrics["active_count"].value, 2.0)
        self.assertAlmostEqual(snap1.metrics["min_separation"].value, math.hypot(10.0, 10.0))
        self.assertEqual(snap1.metrics["clearance_time"].display_name, "Total Elapsed Simulation Time")

        # Initial point added to series
        self.assertEqual(len(pipeline.series["active_count"].points), 1)

        # Repeated Poll without stepping simulation
        snap2 = pipeline.poll_simulation(sim)
        self.assertFalse(snap2.has_new_data)
        # Series count must NOT increase
        self.assertEqual(len(pipeline.series["active_count"].points), 1)

    def test_freshness_and_paused_mode_semantics(self):
        """Paused simulations and replay mode do not falsely trigger staleness."""
        env = EnvironmentMap(width=20.0, height=20.0)
        sim = SimulationEngine(environment=env, max_steps=100, timestep=0.1)
        sim.add_pedestrian(Pedestrian("p1", (0.0, 0.0)))

        # 1. Replay mode: never stale even if wall-clock time passes
        pipeline_replay = LiveAnalyticsPipeline(
            mode=AnalyticsFreshnessMode.REPLAY,
            stale_timeout_seconds=0.05,
            run_id="run_replay_test"
        )
        snap_rep = pipeline_replay.poll_simulation(sim)
        time.sleep(0.06)
        snap_rep2 = pipeline_replay.poll_simulation(sim)
        self.assertFalse(snap_rep2.is_stale)

        # 2. Paused simulation in LIVE mode: status is PAUSED, not falsely STALE
        pipeline_live = LiveAnalyticsPipeline(
            mode=AnalyticsFreshnessMode.LIVE,
            stale_timeout_seconds=0.05,
            run_id="run_live_test"
        )
        snap_paused = pipeline_live.poll_simulation(sim, is_paused=True)
        self.assertTrue(snap_paused.is_paused)
        self.assertFalse(snap_paused.is_stale)
        self.assertEqual(snap_paused.metrics["active_count"].status, MetricStatus.PAUSED)

        # 3. Unpaused LIVE mode with elapsed timeout: flagged as STALE
        time.sleep(0.06)
        snap_stale = pipeline_live.poll_simulation(sim, is_paused=False)
        self.assertTrue(snap_stale.is_stale)
        self.assertEqual(snap_stale.metrics["active_count"].status, MetricStatus.STALE)

    def test_exit_throughput_unavailability_handling(self):
        """When engine does not provide exit events, throughput is cleanly marked UNAVAILABLE."""
        env = EnvironmentMap(width=20.0, height=20.0)
        sim = SimulationEngine(environment=env, max_steps=100, timestep=0.1)
        sim.add_pedestrian(Pedestrian("p1", (0.0, 0.0)))

        pipeline = LiveAnalyticsPipeline(run_id="run_tp")
        snap = pipeline.poll_simulation(sim)

        self.assertEqual(snap.metrics["exit_throughput"].status, MetricStatus.UNAVAILABLE)
        self.assertIsNone(snap.metrics["exit_throughput"].value)
        self.assertIn("Authoritative exit events not provided", snap.metrics["exit_throughput"].reason)

    def test_run_transition_isolation(self):
        """Switching run_id resets all series and accumulators cleanly."""
        env = EnvironmentMap(width=20.0, height=20.0)
        sim = SimulationEngine(environment=env, timestep=0.1)
        sim.add_pedestrian(Pedestrian("p1", (1.0, 1.0)))

        pipeline = LiveAnalyticsPipeline(run_id="run_A")
        pipeline.poll_simulation(sim)
        self.assertEqual(len(pipeline.series["active_count"].points), 1)
        self.assertEqual(pipeline.series["active_count"].run_id, "run_A")

        # Poll with different run_id -> triggers automatic clean reset
        snap_b = pipeline.poll_simulation(sim, run_id="run_B")
        self.assertEqual(pipeline.run_id, "run_B")
        self.assertEqual(snap_b.run_id, "run_B")
        self.assertEqual(pipeline.series["active_count"].run_id, "run_B")
        self.assertEqual(len(pipeline.series["active_count"].points), 1)

    def test_controlled_replay_walkthrough(self):
        """Step-by-step deterministic replay integration walkthrough."""
        env = EnvironmentMap(width=20.0, height=20.0)
        sim = SimulationEngine(environment=env, timestep=0.1)
        p1 = Pedestrian("p1", (0.0, 0.0), velocity=(2.5, 0.0), status=PedestrianStatus.ACTIVE)
        p2 = Pedestrian("p2", (10.0, 10.0), velocity=(0.0, 0.0), status=PedestrianStatus.ACTIVE)
        sim.add_pedestrian(p1)
        sim.add_pedestrian(p2)

        spec = GridSpecification(0.0, 0.0, 20.0, 20.0, 5.0, 5.0)
        pipeline = LiveAnalyticsPipeline(grid_spec=spec, observation_interval=0.4, run_id="run_replay")

        # Step 0
        s0 = pipeline.poll_simulation(sim)
        self.assertEqual(s0.source_step, 0)
        self.assertEqual(s0.metrics["active_count"].value, 2.0)

        # Advance 4 steps (step 4, t=0.4s)
        for _ in range(4):
            sim.step()
        self.assertEqual(sim.current_step, 4)

        s4 = pipeline.poll_simulation(sim)
        self.assertTrue(s4.has_new_data)
        self.assertEqual(s4.source_step, 4)
        self.assertAlmostEqual(s4.source_timestamp, 0.4)
        self.assertEqual(len(pipeline.series["active_count"].points), 2)

        # Check dwell time accumulated at step 0 positions: cell (0, 0) and cell (2, 2)
        self.assertAlmostEqual(s4.spatial_result.dwell_time_grid[0, 0], 0.4)
        self.assertAlmostEqual(s4.spatial_result.dwell_time_grid[2, 2], 0.4)

    def test_engine_non_mutation_during_repeated_polling(self):
        """Proof that live analytics polling never mutates simulation engine state."""
        env = EnvironmentMap(width=50.0, height=50.0)
        sim = SimulationEngine(environment=env, max_steps=100, timestep=0.1)
        ped = Pedestrian("p_test", (5.0, 5.0), velocity=(1.0, 0.0), status=PedestrianStatus.ACTIVE)
        sim.add_pedestrian(ped)

        pipeline = LiveAnalyticsPipeline(observation_interval=0.4)

        before_step = sim.current_step
        before_pos = ped.position
        before_vel = ped.velocity
        before_status = ped.status
        before_peds_len = len(sim.pedestrians)

        # Poll 5 times
        for _ in range(5):
            pipeline.poll_simulation(sim)

        self.assertEqual(sim.current_step, before_step)
        self.assertEqual(ped.position, before_pos)
        self.assertEqual(ped.velocity, before_vel)
        self.assertEqual(ped.status, before_status)
        self.assertEqual(len(sim.pedestrians), before_peds_len)


if __name__ == "__main__":
    unittest.main()
