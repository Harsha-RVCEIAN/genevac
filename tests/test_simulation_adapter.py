"""Tests for SimulationTrajectoryAdapter (Stage 1).

Verifies:
1. Adapter initialization and custom buffer injection.
2. Cadence filtering (0.4s interval vs 0.1s simulation steps).
3. Robustness to duplicate updates, irregular steps, and skipped steps.
4. Reset detection via step rollback (monotonic step drop) without run_id.
5. Reset detection via engine instance replacement.
6. Cohort tracking: ACTIVE, BLOCKED, EVACUATED (terminal), INACTIVE.
7. Cohort separation in prediction snapshot (BLOCKED not mixed with ACTIVE).
8. Prediction history snapshot generation with configurable max_observation_age.
9. Immutability & non-mutation of SimulationEngine and Pedestrian instances.
10. ConstantVelocityPredictor execution using adapter prediction snapshots.
"""

import unittest
from typing import Any
from genevac.environment import EnvironmentMap
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.prediction import (
    ConstantVelocityPredictor,
    SimulationTrajectoryAdapter,
    TrajectoryHistoryBuffer,
    TrajectoryValidationError,
)
from genevac.simulation import SimulationEngine


class MockPedestrian:
    """Mock pedestrian to test custom statuses (e.g. BLOCKED, INACTIVE) safely."""
    def __init__(self, id: str, position: tuple[float, float], status: str = "active"):
        self.id = id
        self.position = position
        self.status = status


class MockEngine:
    """Mock engine simulating step progression and pedestrian collection."""
    def __init__(self, current_step: int = 0, timestep: float = 0.1):
        self.current_step = current_step
        self.timestep = timestep
        self.pedestrians: list[Any] = []


class TestSimulationTrajectoryAdapter(unittest.TestCase):
    """Test suite for SimulationTrajectoryAdapter."""

    def test_adapter_cadence_filtering(self):
        """Adapter records every 0.4s (4 steps) by default, skipping intermediate steps."""
        env = EnvironmentMap(width=50.0, height=50.0)
        engine = SimulationEngine(environment=env, timestep=0.1)
        ped = Pedestrian(id="p1", position=(0.0, 0.0), velocity=(1.0, 2.0))
        engine.add_pedestrian(ped)

        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)

        # Step 0: Initial observation recorded
        recorded_0 = adapter.update_from_simulation(engine)
        self.assertEqual(recorded_0, 1)
        self.assertEqual(len(adapter.buffer.get_observations("p1")), 1)

        # Step 1, 2, 3: Should NOT be recorded (cadence = 0.4s)
        for _ in range(3):
            engine.step()
            rec = adapter.update_from_simulation(engine)
            self.assertEqual(rec, 0)
        self.assertEqual(len(adapter.buffer.get_observations("p1")), 1)

        # Step 4: (t = 0.4s) -> Should be recorded!
        engine.step()
        self.assertEqual(engine.current_step, 4)
        rec_4 = adapter.update_from_simulation(engine)
        self.assertEqual(rec_4, 1)
        self.assertEqual(len(adapter.buffer.get_observations("p1")), 2)

        # Step 5, 6, 7: Skipped
        for _ in range(3):
            engine.step()
            adapter.update_from_simulation(engine)
        self.assertEqual(len(adapter.buffer.get_observations("p1")), 2)

        # Step 8: (t = 0.8s) -> Recorded
        engine.step()
        self.assertEqual(engine.current_step, 8)
        rec_8 = adapter.update_from_simulation(engine)
        self.assertEqual(rec_8, 1)
        self.assertEqual(len(adapter.buffer.get_observations("p1")), 3)

    def test_idempotence_on_duplicate_step_calls(self):
        """Calling update_from_simulation multiple times at the same step does not duplicate data."""
        engine = MockEngine(current_step=0, timestep=0.1)
        engine.pedestrians = [MockPedestrian("p1", (10.0, 10.0))]

        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)
        self.assertEqual(adapter.update_from_simulation(engine), 1)

        # Repeated calls without advancing step
        for _ in range(5):
            self.assertEqual(adapter.update_from_simulation(engine), 0)
        self.assertEqual(len(adapter.buffer.get_observations("p1")), 1)

    def test_reset_detection_on_step_rollback(self):
        """When current_step drops below last observed step, buffer and cohorts reset."""
        engine = MockEngine(current_step=0, timestep=0.1)
        engine.pedestrians = [MockPedestrian("p1", (0.0, 0.0))]
        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)

        adapter.update_from_simulation(engine)
        engine.current_step = 4
        engine.pedestrians = [MockPedestrian("p1", (4.0, 4.0))]
        adapter.update_from_simulation(engine)
        self.assertEqual(len(adapter.buffer.get_observations("p1")), 2)

        # Simulation resets: step drops to 0
        engine.current_step = 0
        engine.pedestrians = [MockPedestrian("p1", (0.0, 0.0))]
        rec = adapter.update_from_simulation(engine)

        self.assertEqual(rec, 1)
        # History must have been cleared, containing only the new step 0 observation
        self.assertEqual(len(adapter.buffer.get_observations("p1")), 1)
        self.assertEqual(adapter.buffer.get_observations("p1")[0].step_index, 0)

    def test_reset_detection_on_engine_instance_replacement(self):
        """Replacing the engine instance triggers automatic reset."""
        engine1 = MockEngine(current_step=4, timestep=0.1)
        engine1.pedestrians = [MockPedestrian("p1", (1.0, 1.0))]
        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)

        adapter.update_from_simulation(engine1)
        self.assertEqual(len(adapter.buffer.get_observations("p1")), 1)

        # Completely new engine instance created for another run
        engine2 = MockEngine(current_step=4, timestep=0.1)
        engine2.pedestrians = [MockPedestrian("p2", (5.0, 5.0))]

        adapter.update_from_simulation(engine2)
        # Prior engine's agents should not exist in buffer
        self.assertEqual(len(adapter.buffer.get_observations("p1")), 0)
        self.assertEqual(len(adapter.buffer.get_observations("p2")), 1)

    def test_cohort_separation_active_blocked_evacuated_inactive(self):
        """Adapter correctly partitions cohorts without mixing BLOCKED into ACTIVE snapshot."""
        engine = MockEngine(current_step=0, timestep=0.1)
        engine.pedestrians = [
            MockPedestrian("p_act", (0.0, 0.0), status="active"),
            MockPedestrian("p_blk", (1.0, 1.0), status="blocked"),
            MockPedestrian("p_inact", (2.0, 2.0), status="inactive"),
            MockPedestrian("p_evac", (3.0, 3.0), status="evacuated"),
        ]

        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)
        rec_count = adapter.update_from_simulation(engine)

        # Only ACTIVE and BLOCKED are recorded (2 pedestrians)
        self.assertEqual(rec_count, 2)
        self.assertIn("p_act", adapter.active_agents)
        self.assertIn("p_blk", adapter.blocked_agents)
        self.assertIn("p_inact", adapter.inactive_agents)
        self.assertIn("p_evac", adapter.evacuated_agents)

        # Check cohort queries
        snap_active = adapter.get_prediction_snapshot(current_step=0, cohort="active", max_observation_age=None)
        self.assertIn("p_act", snap_active.trajectories)
        self.assertNotIn("p_blk", snap_active.trajectories)
        self.assertNotIn("p_inact", snap_active.trajectories)
        self.assertNotIn("p_evac", snap_active.trajectories)

        snap_blocked = adapter.get_prediction_snapshot(current_step=0, cohort="blocked", max_observation_age=None)
        self.assertIn("p_blk", snap_blocked.trajectories)
        self.assertNotIn("p_act", snap_blocked.trajectories)

        snap_all = adapter.get_prediction_snapshot(current_step=0, cohort="all_tracked", max_observation_age=None)
        self.assertEqual(set(snap_all.trajectories.keys()), {"p_act", "p_blk"})

    def test_evacuated_is_terminal_for_current_run(self):
        """EVACUATED stops further recording while preserving past observations."""
        engine = MockEngine(current_step=0, timestep=0.1)
        p = MockPedestrian("p1", (0.0, 0.0), status="active")
        engine.pedestrians = [p]

        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)
        adapter.update_from_simulation(engine)

        # Step 4: still active
        engine.current_step = 4
        p.position = (1.0, 1.0)
        adapter.update_from_simulation(engine)
        self.assertEqual(len(adapter.buffer.get_observations("p1")), 2)

        # Step 8: Pedestrian evacuates!
        engine.current_step = 8
        p.status = "evacuated"
        p.position = (2.0, 2.0)
        rec_evac = adapter.update_from_simulation(engine)
        self.assertEqual(rec_evac, 0)
        # History is preserved (2 observations), not overwritten or added to
        self.assertEqual(len(adapter.buffer.get_observations("p1")), 2)
        self.assertEqual(adapter.buffer.get_observations("p1")[-1].step_index, 4)
        self.assertIn("p1", adapter.evacuated_agents)
        self.assertNotIn("p1", adapter.active_agents)

        # Step 12: Even if pedestrian somehow flips back to active in corrupted run,
        # adapter terminal rule blocks further recording for current run
        engine.current_step = 12
        p.status = "active"
        rec_after = adapter.update_from_simulation(engine)
        self.assertEqual(rec_after, 0)
        self.assertEqual(len(adapter.buffer.get_observations("p1")), 2)

    def test_zero_mutation_of_simulation_entities(self):
        """Adapter never modifies Pedestrian objects or SimulationEngine."""
        env = EnvironmentMap(width=50.0, height=50.0)
        engine = SimulationEngine(environment=env, timestep=0.1)
        ped = Pedestrian(id="ped_safe", position=(1.0, 2.0), velocity=(0.5, 0.5), status=PedestrianStatus.ACTIVE)
        engine.add_pedestrian(ped)

        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)
        adapter.update_from_simulation(engine)

        self.assertEqual(ped.position, (1.0, 2.0))
        self.assertEqual(ped.velocity, (0.5, 0.5))
        self.assertEqual(ped.status, PedestrianStatus.ACTIVE)
        self.assertEqual(engine.current_step, 0)

    def test_end_to_end_prediction_via_adapter_snapshot(self):
        """ConstantVelocityPredictor works seamlessly with get_prediction_snapshot()."""
        engine = MockEngine(current_step=0, timestep=0.1)
        engine.pedestrians = [MockPedestrian("p1", (0.0, 0.0), status="active")]
        adapter = SimulationTrajectoryAdapter(observation_interval=0.4, max_observation_age=0.5)

        adapter.update_from_simulation(engine)

        engine.current_step = 4
        engine.pedestrians = [MockPedestrian("p1", (2.0, 4.0), status="active")]
        adapter.update_from_simulation(engine)

        # Snapshot at step 4 (t = 0.4s)
        snapshot = adapter.get_prediction_snapshot(current_step=4, timestep=0.1, cohort="active")
        predictor = ConstantVelocityPredictor()
        prediction = predictor.predict(snapshot=snapshot, agent_id="p1", horizon=3)

        self.assertEqual(prediction.agent_id, "p1")
        self.assertEqual(len(prediction.predictions), 3)
        self.assertAlmostEqual(prediction.source_timestamp, 0.4)
        # vx = 2.0 / 0.4 = 5.0, vy = 4.0 / 0.4 = 10.0
        # t1 = 0.4 + 0.1 = 0.5 -> x = 2.0 + 5.0 * 0.1 = 2.5, y = 4.0 + 10.0 * 0.1 = 5.0
        self.assertAlmostEqual(prediction.predictions[0].timestamp, 0.5)
        self.assertAlmostEqual(prediction.predictions[0].x, 2.5)
        self.assertAlmostEqual(prediction.predictions[0].y, 5.0)

    def test_incapacitated_status_handling(self):
        """INCAPACITATED status is tracked, stops new observations, and remains separate from ACTIVE."""
        engine = MockEngine(current_step=0, timestep=0.1)
        p = MockPedestrian("p_inc", (0.0, 0.0), status="active")
        engine.pedestrians = [p]
        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)

        adapter.update_from_simulation(engine)
        self.assertEqual(len(adapter.buffer.get_observations("p_inc")), 1)

        # Transition to incapacitated
        engine.current_step = 4
        p.status = "incapacitated"
        p.position = (1.0, 1.0)
        rec = adapter.update_from_simulation(engine)

        self.assertEqual(rec, 0)
        self.assertIn("p_inc", adapter.incapacitated_agents)
        self.assertNotIn("p_inc", adapter.active_agents)
        self.assertEqual(len(adapter.buffer.get_observations("p_inc")), 1)

    def test_unknown_status_raises_validation_error(self):
        """Unknown status must NOT silently become ACTIVE; it must raise TrajectoryValidationError."""
        engine = MockEngine(current_step=0, timestep=0.1)
        engine.pedestrians = [MockPedestrian("p_unk", (0.0, 0.0), status="confused_status")]
        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)

        with self.assertRaises(TrajectoryValidationError) as ctx:
            adapter.update_from_simulation(engine)
        self.assertEqual(ctx.exception.field, "status")
        self.assertEqual(ctx.exception.agent_id, "p_unk")

    def test_missing_status_attribute_raises_validation_error(self):
        """A pedestrian with status=None must raise TrajectoryValidationError."""
        engine = MockEngine(current_step=0, timestep=0.1)
        p = MockPedestrian("p_none", (0.0, 0.0))
        p.status = None  # type: ignore
        engine.pedestrians = [p]
        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)

        with self.assertRaises(TrajectoryValidationError) as ctx:
            adapter.update_from_simulation(engine)
        self.assertEqual(ctx.exception.field, "status")

    def test_invalid_adapter_constructor_arguments(self):
        """Constructor rejects invalid observation_interval and max_observation_age."""
        with self.assertRaises(ValueError):
            SimulationTrajectoryAdapter(observation_interval=-0.1)
        with self.assertRaises(ValueError):
            SimulationTrajectoryAdapter(observation_interval=float("nan"))
        with self.assertRaises(ValueError):
            SimulationTrajectoryAdapter(observation_interval="0.4")  # type: ignore
        with self.assertRaises(ValueError):
            SimulationTrajectoryAdapter(max_observation_age=-1.0)
        with self.assertRaises(ValueError):
            SimulationTrajectoryAdapter(max_observation_age=float("inf"))

    def test_invalid_engine_timestep_and_current_step(self):
        """update_from_simulation validates engine.current_step and engine.timestep."""
        adapter = SimulationTrajectoryAdapter()

        bad_step_engine = MockEngine(current_step=-5, timestep=0.1)
        with self.assertRaises(TrajectoryValidationError) as ctx:
            adapter.update_from_simulation(bad_step_engine)
        self.assertEqual(ctx.exception.field, "current_step")

        bad_step_engine2 = MockEngine(current_step="5", timestep=0.1)  # type: ignore
        with self.assertRaises(TrajectoryValidationError) as ctx:
            adapter.update_from_simulation(bad_step_engine2)
        self.assertEqual(ctx.exception.field, "current_step")

        bad_timestep_engine = MockEngine(current_step=0, timestep=-0.1)
        with self.assertRaises(TrajectoryValidationError) as ctx:
            adapter.update_from_simulation(bad_timestep_engine)
        self.assertEqual(ctx.exception.field, "timestep")

        bad_timestep_engine2 = MockEngine(current_step=0, timestep=float("nan"))
        with self.assertRaises(TrajectoryValidationError) as ctx:
            adapter.update_from_simulation(bad_timestep_engine2)
        self.assertEqual(ctx.exception.field, "timestep")

    def test_reset_clears_all_adapter_state(self):
        """Explicit reset() clears last_engine_id, observed/recorded steps, and all cohorts."""
        engine = MockEngine(current_step=0, timestep=0.1)
        engine.pedestrians = [
            MockPedestrian("p_act", (0.0, 0.0), status="active"),
            MockPedestrian("p_blk", (1.0, 1.0), status="blocked"),
            MockPedestrian("p_evac", (2.0, 2.0), status="evacuated"),
            MockPedestrian("p_inc", (3.0, 3.0), status="incapacitated"),
            MockPedestrian("p_inact", (4.0, 4.0), status="inactive"),
        ]
        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)
        adapter.update_from_simulation(engine)

        self.assertIsNotNone(adapter._last_engine_id)
        self.assertEqual(adapter._last_observed_step, 0)
        self.assertEqual(adapter._last_recorded_step, 0)

        adapter.reset()

        self.assertIsNone(adapter._last_engine_id)
        self.assertEqual(adapter._last_observed_step, -1)
        self.assertEqual(adapter._last_recorded_step, -1)
        self.assertEqual(adapter._last_recorded_timestamp, -1.0)
        self.assertEqual(len(adapter.active_agents), 0)
        self.assertEqual(len(adapter.blocked_agents), 0)
        self.assertEqual(len(adapter.evacuated_agents), 0)
        self.assertEqual(len(adapter.incapacitated_agents), 0)
        self.assertEqual(len(adapter.inactive_agents), 0)
        self.assertEqual(len(adapter._agent_last_recorded_step), 0)
        self.assertEqual(len(adapter._agent_last_recorded_time), 0)
        self.assertEqual(len(adapter.buffer.get_agent_ids()), 0)


if __name__ == "__main__":
    unittest.main()

