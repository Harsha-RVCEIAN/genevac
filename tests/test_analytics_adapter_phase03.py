"""Tests for Phase 03 Analytics Data Adapter enhancements.

Verifies:
1. adapt_snapshot consuming pre-built SimulationObservationSnapshot objects.
2. SnapshotAdaptationReport auditing accepted, rejected, and duplicate records.
3. Resilience against malformed/non-finite observations without corrupting valid records.
4. Seamless pipeline flow from adapter output to Phase 02 metrics and predictors.
5. Absolute non-mutation of simulation engine and snapshot instances.
"""

from dataclasses import dataclass
import math
import unittest
from typing import Optional, Tuple

from genevac.environment import EnvironmentMap
from genevac.metrics import (
    calculate_ade,
    calculate_area_density,
    calculate_clearance_time,
    calculate_fde,
    calculate_minimum_separation,
    validate_evacuation_path,
)
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.prediction import (
    ConstantVelocityPredictor,
    SimulationTrajectoryAdapter,
    SnapshotAdaptationReport,
)
from genevac.simulation import SimulationEngine


@dataclass(frozen=True)
class MockPedObservation:
    pedestrian_id: str
    x: float
    y: float
    status: str = "active"
    vx: float = 0.0
    vy: float = 0.0
    target_exit_id: Optional[str] = None


@dataclass(frozen=True)
class MockSnapshot:
    snapshot_id: str
    step_count: int
    timestamp: float
    dt: float
    pedestrians: Tuple[MockPedObservation, ...]


class TestAnalyticsDataAdapterPhase03(unittest.TestCase):
    def test_direct_snapshot_adaptation(self):
        """Adapter correctly ingests immutable MockSnapshot and tracks cohorts."""
        snap = MockSnapshot(
            snapshot_id="snap_000004",
            step_count=4,
            timestamp=0.4,
            dt=0.1,
            pedestrians=(
                MockPedObservation(pedestrian_id="p1", x=1.0, y=2.0, status="active"),
                MockPedObservation(pedestrian_id="p2", x=3.0, y=4.0, status="blocked"),
            ),
        )

        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)
        report = adapter.adapt_snapshot(snap)

        self.assertIsInstance(report, SnapshotAdaptationReport)
        self.assertEqual(report.accepted_observations, 2)
        self.assertEqual(report.rejected_observations, 0)
        self.assertFalse(report.is_duplicate)
        self.assertIn("p1", adapter.active_agents)
        self.assertIn("p2", adapter.blocked_agents)

    def test_duplicate_snapshot_deduplication(self):
        """Repeated submission of identical snapshot is a no-op and reports duplicate."""
        snap = MockSnapshot(
            snapshot_id="snap_000000",
            step_count=0,
            timestamp=0.0,
            dt=0.1,
            pedestrians=(MockPedObservation("p1", 0.0, 0.0),),
        )
        adapter = SimulationTrajectoryAdapter()
        rep1 = adapter.adapt_snapshot(snap)
        self.assertEqual(rep1.accepted_observations, 1)

        rep2 = adapter.adapt_snapshot(snap)
        self.assertEqual(rep2.accepted_observations, 0)
        self.assertTrue(rep2.is_duplicate)
        self.assertEqual(len(adapter.buffer.get_observations("p1")), 1)

    def test_quarantine_malformed_observation_without_corrupting_valid(self):
        """Malformed record (NaN or empty ID) is rejected, while valid record is accepted."""
        snap = MockSnapshot(
            snapshot_id="snap_000004",
            step_count=4,
            timestamp=0.4,
            dt=0.1,
            pedestrians=(
                MockPedObservation(pedestrian_id="valid_p1", x=2.0, y=2.0),
                MockPedObservation(pedestrian_id="bad_p2", x=float("nan"), y=3.0),
                MockPedObservation(pedestrian_id="", x=4.0, y=4.0),
            ),
        )
        adapter = SimulationTrajectoryAdapter()
        report = adapter.adapt_snapshot(snap)

        self.assertEqual(report.accepted_observations, 1)
        self.assertEqual(report.rejected_observations, 2)
        self.assertIn("valid_p1", adapter.active_agents)
        self.assertNotIn("bad_p2", adapter.active_agents)
        self.assertEqual(len(report.rejection_reasons), 2)

    def test_pipeline_to_phase02_consumers(self):
        """Canonical snapshot feeds Phase 02 metrics and predictors seamlessly."""
        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)

        # Step 0
        adapter.adapt_snapshot(MockSnapshot("snap_0", 0, 0.0, 0.1, (MockPedObservation("p1", 0.0, 0.0), MockPedObservation("p2", 10.0, 10.0))))
        # Step 4
        adapter.adapt_snapshot(MockSnapshot("snap_4", 4, 0.4, 0.1, (MockPedObservation("p1", 2.0, 0.0), MockPedObservation("p2", 10.0, 10.0))))

        pred_snap = adapter.get_prediction_snapshot(current_step=4, timestep=0.1, cohort="active")

        # 1. Feed Predictor
        predictor = ConstantVelocityPredictor()
        pred = predictor.predict(pred_snap, agent_id="p1", horizon=2)
        self.assertEqual(len(pred.predictions), 2)
        # vx = 2.0 / 0.4 = 5.0 m/s -> at t=0.5: x=2.5
        self.assertAlmostEqual(pred.predictions[0].x, 2.5)

        # 2. Feed ADE / FDE
        gt_positions = [(2.5, 0.0), (3.0, 0.0)]
        pred_positions = [(p.x, p.y) for p in pred.predictions]
        ade = calculate_ade(gt_positions, pred_positions)
        fde = calculate_fde(gt_positions, pred_positions)
        self.assertAlmostEqual(ade, 0.0)
        self.assertAlmostEqual(fde, 0.0)

        # 3. Feed Crowd Separation and Density
        all_positions = [obs[-1].position for obs in pred_snap.trajectories.values()]
        min_sep = calculate_minimum_separation(all_positions)
        self.assertAlmostEqual(min_sep, math.hypot(10.0 - 2.0, 10.0 - 0.0))

        density = calculate_area_density(all_positions, (0.0, 0.0, 20.0, 20.0))
        self.assertAlmostEqual(density, 2 / 400.0)

        # 4. Feed Clearance Time
        clear_t = calculate_clearance_time(pred_snap.current_step, pred_snap.timestep)
        self.assertAlmostEqual(clear_t, 0.4)

        # 5. Feed Route Validation
        p1_traj = [obs.position for obs in pred_snap.trajectories["p1"]]
        summary = validate_evacuation_path(p1_traj, world_bounds=(0.0, 0.0, 50.0, 50.0))
        self.assertTrue(summary.is_valid)

    def test_missing_agent_marked_not_observed_not_evacuated(self):
        """An agent present at step 0 but absent at step 4 is NOT_OBSERVED, never assumed evacuated."""
        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)

        # Step 0: p1 and p2 present
        snap0 = MockSnapshot("snap_0", 0, 0.0, 0.1, (MockPedObservation("p1", 0.0, 0.0), MockPedObservation("p2", 1.0, 1.0)))
        adapter.adapt_snapshot(snap0)
        self.assertIn("p1", adapter.active_agents)
        self.assertIn("p2", adapter.active_agents)
        self.assertEqual(len(adapter.not_observed_agents), 0)
        self.assertEqual(len(adapter.evacuated_agents), 0)

        # Step 4: p2 present, p1 is absent from snapshot without status="evacuated"
        snap4 = MockSnapshot("snap_4", 4, 0.4, 0.1, (MockPedObservation("p2", 2.0, 2.0),))
        adapter.adapt_snapshot(snap4)

        # Invariant: p1 must be in not_observed_agents, NOT evacuated_agents
        self.assertNotIn("p1", adapter.active_agents)
        self.assertIn("p1", adapter.not_observed_agents)
        self.assertNotIn("p1", adapter.evacuated_agents)
        self.assertEqual(len(adapter.evacuated_agents), 0)

        # Invariant: Past observations of p1 are preserved in buffer, not cleared or modified
        self.assertEqual(len(adapter.buffer.get_observations("p1")), 1)
        self.assertEqual(adapter.buffer.get_observations("p1")[0].step_index, 0)

        # Invariant: Distinct cohorts
        self.assertTrue(adapter.active_agents.isdisjoint(adapter.not_observed_agents))
        self.assertTrue(adapter.evacuated_agents.isdisjoint(adapter.not_observed_agents))

    def test_real_engine_read_only_integration(self):
        """Full read-only proof using real SimulationEngine instance."""
        env = EnvironmentMap(width=50.0, height=50.0)
        engine = SimulationEngine(environment=env, max_steps=100, timestep=0.1)
        p1 = Pedestrian(id="p1", position=(2.0, 2.0), velocity=(1.0, 0.0), status=PedestrianStatus.ACTIVE)
        p2 = Pedestrian(id="p2", position=(5.0, 5.0), velocity=(0.0, 1.0), status=PedestrianStatus.ACTIVE)
        engine.add_pedestrian(p1)
        engine.add_pedestrian(p2)

        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)

        # Capture state before adaptation
        before_step = engine.current_step
        before_p1_pos = p1.position
        before_p1_vel = p1.velocity
        before_p1_status = p1.status
        before_p2_pos = p2.position
        before_peds_count = len(engine.pedestrians)

        # Ingest via update_from_simulation
        count = adapter.update_from_simulation(engine)
        self.assertEqual(count, 2)

        # Capture state after adaptation
        self.assertEqual(engine.current_step, before_step)
        self.assertEqual(p1.position, before_p1_pos)
        self.assertEqual(p1.velocity, before_p1_vel)
        self.assertEqual(p1.status, before_p1_status)
        self.assertEqual(p2.position, before_p2_pos)
        self.assertEqual(len(engine.pedestrians), before_peds_count)


if __name__ == "__main__":
    unittest.main()

