"""Tests for timestamp consistency, irregular sampling intervals, and freshness validation."""

import unittest
from genevac.prediction import (
    ConstantVelocityPredictor,
    TrajectoryCategory,
    TrajectoryHistoryBuffer,
    TrajectoryObservation,
    TrajectorySnapshot,
    TrajectoryValidationError,
    TrajectoryValidator,
)


class TestTimestampConsistency(unittest.TestCase):
    """Test suite ensuring strict timestamp consistency and freshness semantics."""

    def test_snapshot_timestamp_matches_engine_capture_state(self):
        """Snapshot current_timestamp represents the simulation time at snapshot capture."""
        buffer = TrajectoryHistoryBuffer()
        buffer.record("a1", (0.0, 0.0), step_index=0, timestamp=0.0)
        buffer.record("a1", (1.0, 1.0), step_index=1, timestamp=0.1)

        snapshot = buffer.get_snapshot(current_step=5, current_timestamp=0.5, timestep=0.1)
        self.assertEqual(snapshot.current_step, 5)
        self.assertEqual(snapshot.current_timestamp, 0.5)
        self.assertEqual(snapshot.timestep, 0.1)

    def test_observations_retain_their_own_actual_timestamps(self):
        """Each observation keeps its individual timestamp, not overwritten by snapshot time."""
        buffer = TrajectoryHistoryBuffer()
        buffer.record("a1", (0.0, 0.0), step_index=2, timestamp=0.2)
        buffer.record("a1", (1.0, 0.5), step_index=4, timestamp=0.4)

        snapshot = buffer.get_snapshot(current_step=10, current_timestamp=1.0, timestep=0.1)
        obs_seq = snapshot.trajectories["a1"]

        self.assertEqual(len(obs_seq), 2)
        self.assertEqual(obs_seq[0].timestamp, 0.2)
        self.assertEqual(obs_seq[0].step_index, 2)
        self.assertEqual(obs_seq[1].timestamp, 0.4)
        self.assertEqual(obs_seq[1].step_index, 4)
        self.assertEqual(snapshot.current_timestamp, 1.0)

    def test_irregular_observation_intervals_prediction(self):
        """ConstantVelocityPredictor correctly computes velocity and future points across irregular dt."""
        buffer = TrajectoryHistoryBuffer()
        # dt = 0.5s between steps, non-uniform interval
        buffer.record("sensor_ped", (10.0, 20.0), step_index=0, timestamp=0.0, category=TrajectoryCategory.OBSERVED)
        buffer.record("sensor_ped", (15.0, 30.0), step_index=5, timestamp=0.5, category=TrajectoryCategory.OBSERVED)

        # Snapshot at t = 0.8
        snapshot = buffer.get_snapshot(current_step=8, current_timestamp=0.8, timestep=0.1)

        predictor = ConstantVelocityPredictor()
        pred = predictor.predict(snapshot=snapshot, agent_id="sensor_ped", horizon=3, prediction_interval=0.2)

        # vx = (15 - 10) / 0.5 = 10.0 m/s, vy = (30 - 20) / 0.5 = 20.0 m/s
        # t_future_1 = snapshot.current_timestamp + 0.2 = 1.0
        # dt_from_last = 1.0 - 0.5 = 0.5s
        # x_future_1 = 15.0 + 10.0 * 0.5 = 20.0
        # y_future_1 = 30.0 + 20.0 * 0.5 = 40.0
        self.assertEqual(pred.predictions[0].timestamp, 1.0)
        self.assertAlmostEqual(pred.predictions[0].x, 20.0)
        self.assertAlmostEqual(pred.predictions[0].y, 40.0)

    def test_configured_max_observation_age_validations(self):
        """Buffer and validator reject non-positive or invalid max_observation_age values."""
        with self.assertRaises(ValueError):
            TrajectoryHistoryBuffer(max_observation_age=-0.5)

        with self.assertRaises(ValueError):
            TrajectoryHistoryBuffer(max_observation_age=float("nan"))

        with self.assertRaises(ValueError):
            TrajectoryHistoryBuffer(max_observation_age="invalid")  # type: ignore

        # Valid zero and positive
        buf_zero = TrajectoryHistoryBuffer(max_observation_age=0.0)
        self.assertEqual(buf_zero.max_observation_age, 0.0)

        buf_pos = TrajectoryHistoryBuffer(max_observation_age=1.5)
        self.assertEqual(buf_pos.max_observation_age, 1.5)

    def test_stale_observation_rejected_by_buffer_snapshot(self):
        """Buffer snapshot creation fails with TrajectoryValidationError when latest observation exceeds max age."""
        buffer = TrajectoryHistoryBuffer(max_observation_age=0.2)
        buffer.record("ped_stale", (0.0, 0.0), step_index=1, timestamp=0.1)
        buffer.record("ped_stale", (1.0, 1.0), step_index=2, timestamp=0.2)

        # At timestamp 0.3, age is 0.3 - 0.2 = 0.1 <= 0.2 -> valid
        snap_fresh = buffer.get_snapshot(current_step=3, current_timestamp=0.3, timestep=0.1)
        self.assertIn("ped_stale", snap_fresh.trajectories)

        # At timestamp 0.6, age is 0.6 - 0.2 = 0.4 > 0.2 -> raises TrajectoryValidationError
        with self.assertRaises(TrajectoryValidationError) as ctx:
            buffer.get_snapshot(current_step=6, current_timestamp=0.6, timestep=0.1)
        self.assertEqual(ctx.exception.field, "observation_age")
        self.assertEqual(ctx.exception.agent_id, "ped_stale")

    def test_stale_observation_snapshot_override(self):
        """Snapshot creation allows overriding max_observation_age at validation time."""
        buffer = TrajectoryHistoryBuffer(max_observation_age=1.0)  # default lenient
        buffer.record("ped_1", (0.0, 0.0), step_index=1, timestamp=0.1)
        buffer.record("ped_1", (1.0, 1.0), step_index=2, timestamp=0.2)

        # Snapshot with strict override max_observation_age=0.1 at t=0.4 (age=0.2 > 0.1)
        with self.assertRaises(TrajectoryValidationError) as ctx:
            buffer.get_snapshot(
                current_step=4,
                current_timestamp=0.4,
                timestep=0.1,
                max_observation_age=0.1,
            )
        self.assertEqual(ctx.exception.field, "observation_age")

    def test_predictor_stale_observation_check(self):
        """ConstantVelocityPredictor rejects stale observation when max_observation_age is supplied."""
        buffer = TrajectoryHistoryBuffer()  # unconstrained buffer
        buffer.record("ped_1", (0.0, 0.0), step_index=1, timestamp=0.1)
        buffer.record("ped_1", (1.0, 1.0), step_index=2, timestamp=0.2)

        snapshot = buffer.get_snapshot(current_step=10, current_timestamp=1.0, timestep=0.1)
        predictor = ConstantVelocityPredictor()

        # At current_timestamp=1.0, observation at 0.2 has age = 0.8s
        # Predictor with max_observation_age=0.5 must reject it
        with self.assertRaises(TrajectoryValidationError) as ctx:
            predictor.predict(snapshot=snapshot, agent_id="ped_1", horizon=5, max_observation_age=0.5)
        self.assertEqual(ctx.exception.field, "observation_age")

        # Predictor with max_observation_age=1.0 allows it
        pred = predictor.predict(snapshot=snapshot, agent_id="ped_1", horizon=5, max_observation_age=1.0)
        self.assertEqual(len(pred.predictions), 5)


if __name__ == "__main__":
    unittest.main()

