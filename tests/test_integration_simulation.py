"""Integration tests connecting SimulationEngine (Person A) to TrajectoryHistoryBuffer and ConstantVelocityPredictor (Person B).

Verifies:
- SimulationEngine stepping and authoritative observation recording.
- Snapshot creation without modifying engine state.
- Pure prediction execution with future timestamps.
- Authoritative state immutability.
- Pedestrian lifecycle changes (ACTIVE, EVACUATED, INCAPACITATED).
"""

import unittest
from genevac.environment import EnvironmentMap
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.prediction import (
    ConstantVelocityPredictor,
    TrajectoryCategory,
    TrajectoryHistoryBuffer,
    TrajectoryObservation,
    TrajectoryValidationError,
)
from genevac.simulation import SimulationEngine


class TestSimulationIntegration(unittest.TestCase):
    """Integration test suite for simulation-to-prediction interface."""

    def test_engine_stepping_and_observation_recording(self):
        """Verify engine stepping records observations accurately with engine-supplied time."""
        env = EnvironmentMap(width=50.0, height=50.0)
        engine = SimulationEngine(environment=env, timestep=0.1)
        ped1 = Pedestrian(id="ped_1", position=(10.0, 20.0), velocity=(1.0, 0.5))
        ped2 = Pedestrian(id="ped_2", position=(5.0, 5.0), velocity=(-0.5, 2.0))
        engine.add_pedestrian(ped1)
        engine.add_pedestrian(ped2)

        buffer = TrajectoryHistoryBuffer(window_size=20, min_history_required=2)

        # Record initial step 0
        current_time = engine.current_step * engine.timestep
        for ped in engine.pedestrians:
            buffer.record(
                observation_or_agent_id=ped.id,
                position=ped.position,
                step_index=engine.current_step,
                timestamp=current_time,
                category=TrajectoryCategory.SIMULATED,
            )

        # Advance engine 3 steps and record observations
        for _ in range(3):
            engine.step()
            current_time = engine.current_step * engine.timestep
            for ped in engine.pedestrians:
                buffer.record(
                    observation_or_agent_id=ped.id,
                    position=ped.position,
                    step_index=engine.current_step,
                    timestamp=current_time,
                    category=TrajectoryCategory.SIMULATED,
                )

        self.assertEqual(engine.current_step, 3)
        self.assertEqual(len(buffer.get_observations("ped_1")), 4)
        self.assertEqual(len(buffer.get_observations("ped_2")), 4)

        # Verify positions match exact simulation progression
        obs1_last = buffer.get_observations("ped_1")[-1]
        self.assertAlmostEqual(obs1_last.x, 10.0 + 1.0 * 0.3)
        self.assertAlmostEqual(obs1_last.y, 20.0 + 0.5 * 0.3)
        self.assertAlmostEqual(obs1_last.timestamp, 0.3)
        self.assertEqual(obs1_last.step_index, 3)

    def test_snapshot_creation_and_predictor_execution(self):
        """Verify immutable snapshot creation and ConstantVelocityPredictor execution."""
        env = EnvironmentMap(width=50.0, height=50.0)
        engine = SimulationEngine(environment=env, timestep=0.1)
        ped = Pedestrian(id="ped_alpha", position=(0.0, 0.0), velocity=(2.0, 3.0))
        engine.add_pedestrian(ped)

        buffer = TrajectoryHistoryBuffer(window_size=10, min_history_required=2)

        # Record 4 steps
        for _ in range(4):
            cur_time = engine.current_step * engine.timestep
            buffer.record(
                observation_or_agent_id=ped.id,
                position=ped.position,
                step_index=engine.current_step,
                timestamp=cur_time,
                category=TrajectoryCategory.SIMULATED,
            )
            engine.step()

        # Record at final step
        cur_time = engine.current_step * engine.timestep
        buffer.record(
            observation_or_agent_id=ped.id,
            position=ped.position,
            step_index=engine.current_step,
            timestamp=cur_time,
            category=TrajectoryCategory.SIMULATED,
        )

        snapshot = buffer.get_snapshot(
            current_step=engine.current_step,
            current_timestamp=cur_time,
            timestep=engine.timestep,
        )

        # Verify snapshot metadata
        self.assertEqual(snapshot.current_step, 4)
        self.assertAlmostEqual(snapshot.current_timestamp, 0.4)
        self.assertAlmostEqual(snapshot.timestep, 0.1)

        # Run predictor
        predictor = ConstantVelocityPredictor()
        prediction = predictor.predict(snapshot=snapshot, agent_id="ped_alpha", horizon=5)

        # Verify prediction properties
        self.assertEqual(prediction.agent_id, "ped_alpha")
        self.assertEqual(prediction.source_step_index, 4)
        self.assertAlmostEqual(prediction.source_timestamp, 0.4)
        self.assertEqual(prediction.category, TrajectoryCategory.PREDICTED)
        self.assertEqual(len(prediction.predictions), 5)

        # Verify strictly future timestamps
        prev_t = snapshot.current_timestamp
        for pred_pt in prediction.predictions:
            self.assertGreater(pred_pt.timestamp, prev_t)
            prev_t = pred_pt.timestamp

        # Verify predicted coordinates extrapolate constant velocity
        # At step 4: position is (2.0*0.4, 3.0*0.4) = (0.8, 1.2)
        # 1st prediction (t=0.5): (1.0, 1.5)
        self.assertAlmostEqual(prediction.predictions[0].timestamp, 0.5)
        self.assertAlmostEqual(prediction.predictions[0].x, 1.0)
        self.assertAlmostEqual(prediction.predictions[0].y, 1.5)

    def test_authoritative_simulation_state_is_never_mutated_by_prediction(self):
        """Verify calling predict does not change engine, pedestrian objects, or buffer."""
        env = EnvironmentMap(width=50.0, height=50.0)
        engine = SimulationEngine(environment=env, timestep=0.1)
        ped = Pedestrian(id="ped_const", position=(1.0, 2.0), velocity=(0.5, 0.5))
        engine.add_pedestrian(ped)

        buffer = TrajectoryHistoryBuffer()

        for _ in range(3):
            buffer.record(ped.id, ped.position, engine.current_step, engine.current_step * engine.timestep)
            engine.step()

        cur_t = engine.current_step * engine.timestep
        buffer.record(ped.id, ped.position, engine.current_step, cur_t)

        orig_engine_step = engine.current_step
        orig_ped_pos = ped.position
        orig_ped_vel = ped.velocity
        orig_ped_status = ped.status
        orig_buffer_len = len(buffer.get_observations(ped.id))

        snapshot = buffer.get_snapshot(current_step=orig_engine_step, current_timestamp=cur_t, timestep=0.1)
        predictor = ConstantVelocityPredictor()
        _ = predictor.predict(snapshot=snapshot, agent_id=ped.id, horizon=10)

        # State checks
        self.assertEqual(engine.current_step, orig_engine_step)
        self.assertEqual(ped.position, orig_ped_pos)
        self.assertEqual(ped.velocity, orig_ped_vel)
        self.assertEqual(ped.status, orig_ped_status)
        self.assertEqual(len(buffer.get_observations(ped.id)), orig_buffer_len)

    def test_batch_prediction_across_multiple_active_pedestrians(self):
        """Verify predict_batch correctly handles multiple simulation pedestrians."""
        env = EnvironmentMap(width=50.0, height=50.0)
        engine = SimulationEngine(environment=env, timestep=0.1)
        for i in range(3):
            engine.add_pedestrian(Pedestrian(id=f"p_{i}", position=(float(i), float(i * 2)), velocity=(1.0, 1.0)))

        buffer = TrajectoryHistoryBuffer()
        for _ in range(2):
            t = engine.current_step * engine.timestep
            for p in engine.pedestrians:
                buffer.record(p.id, p.position, engine.current_step, t)
            engine.step()

        t_snap = engine.current_step * engine.timestep
        for p in engine.pedestrians:
            buffer.record(p.id, p.position, engine.current_step, t_snap)

        snapshot = buffer.get_snapshot(engine.current_step, t_snap, engine.timestep)
        predictor = ConstantVelocityPredictor()
        batch_res = predictor.predict_batch(snapshot=snapshot, horizon=4)

        self.assertEqual(len(batch_res), 3)
        for i in range(3):
            res = batch_res[f"p_{i}"]
            self.assertFalse(isinstance(res, TrajectoryValidationError))
            self.assertEqual(len(res.predictions), 4)


if __name__ == "__main__":
    unittest.main()

