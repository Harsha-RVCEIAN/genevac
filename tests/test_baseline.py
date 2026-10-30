"""Tests for the timestamp-aware Constant Velocity baseline predictor."""

import dataclasses
import math
import unittest

from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.prediction import (
    ConstantVelocityPredictor,
    TrajectoryCategory,
    TrajectoryHistoryBuffer,
    TrajectoryObservation,
    TrajectoryPrediction,
    TrajectorySnapshot,
    TrajectoryValidationError,
)


class TestConstantVelocityPredictor(unittest.TestCase):
    """Unit and contract tests for ConstantVelocityPredictor."""

    def _make_snapshot(
        self,
        observations_by_agent: dict[str, list[TrajectoryObservation]],
        current_step: int = 10,
        current_timestamp: float = 1.0,
        timestep: float = 0.1,
    ) -> TrajectorySnapshot:
        buf = TrajectoryHistoryBuffer(window_size=30)
        for obs_list in observations_by_agent.values():
            for obs in obs_list:
                buf.record(obs)
        return buf.get_snapshot(
            current_step=current_step,
            current_timestamp=current_timestamp,
            timestep=timestep,
        )

    def test_regular_timestep_positive_x_velocity(self):
        """Obs at t=0.0 (0,0) and t=0.2 (2,0). vx=10. At source_t=1.0, t_future=1.1: x=2+10*(1.1-0.2)=11.0."""
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("p1", 2, 0.2, 2.0, 0.0),
                ]
            },
            current_step=10,
            current_timestamp=1.0,
            timestep=0.1,
        )
        pred = ConstantVelocityPredictor().predict(snap, "p1", horizon=1, prediction_interval=0.1)
        self.assertEqual(pred.agent_id, "p1")
        self.assertEqual(pred.source_step_index, 10)
        self.assertAlmostEqual(pred.source_timestamp, 1.0)
        self.assertEqual(len(pred.predictions), 1)
        pos = pred.predictions[0]
        self.assertAlmostEqual(pos.timestamp, 1.1)
        # vx = (2 - 0) / (0.2 - 0.0) = 10.0; x_future = 2.0 + 10.0 * (1.1 - 0.2) = 11.0
        self.assertAlmostEqual(pos.x, 11.0, places=6)
        self.assertAlmostEqual(pos.y, 0.0, places=6)

    def test_displacement_vs_timestamp_aware_numerical_example(self):
        """Verify timestamp-aware produces (3,0) not (4,0) for the prompt's documented example.

        Observation 1: Position = (0, 0), Timestamp = 0.0
        Observation 2: Position = (2, 0), Timestamp = 0.2
        Velocity = (10, 0) m/s
        At source timestamp 0.2, future timestamp 0.3 (k=1, dt=0.1):
        Expected position = 2.0 + 10.0 * (0.3 - 0.2) = (3, 0)
        (An outdated displacement-per-step method would erroneously yield (4, 0)).
        """
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("p1", 2, 0.2, 2.0, 0.0),
                ]
            },
            current_step=2,
            current_timestamp=0.2,
            timestep=0.1,
        )
        pred = ConstantVelocityPredictor().predict(snap, "p1", horizon=1, prediction_interval=0.1)
        pos = pred.predictions[0]
        self.assertAlmostEqual(pos.timestamp, 0.3, places=6)
        self.assertAlmostEqual(pos.x, 3.0, places=6)
        self.assertAlmostEqual(pos.y, 0.0, places=6)

    def test_irregular_timestamp_intervals_skipped_steps(self):
        """Steps 5 and 7 (step 6 missed). dt=0.2s, displacement=(1.0, 0.0) -> vx=5.0 m/s.
        Source step 7, timestamp 0.7, t_future 0.8: x = 1.0 + 5.0 * (0.8 - 0.7) = 1.5.
        """
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 5, 0.5, 0.0, 0.0),
                    TrajectoryObservation("p1", 7, 0.7, 1.0, 0.0),
                ]
            },
            current_step=7,
            current_timestamp=0.7,
            timestep=0.1,
        )
        pred = ConstantVelocityPredictor().predict(snap, "p1", horizon=1, prediction_interval=0.1)
        self.assertAlmostEqual(pred.predictions[0].timestamp, 0.8, places=6)
        self.assertAlmostEqual(pred.predictions[0].x, 1.5, places=6)
        self.assertAlmostEqual(pred.predictions[0].y, 0.0, places=6)

    def test_stationary_pedestrian(self):
        """Stationary agent at (5.0, 5.0) -> vx=0, vy=0. All future predictions remain at (5.0, 5.0)."""
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 5.0, 5.0),
                    TrajectoryObservation("p1", 1, 0.1, 5.0, 5.0),
                ]
            },
            current_step=5,
            current_timestamp=0.5,
            timestep=0.1,
        )
        pred = ConstantVelocityPredictor().predict(snap, "p1", horizon=4)
        for pos in pred.predictions:
            self.assertAlmostEqual(pos.x, 5.0, places=6)
            self.assertAlmostEqual(pos.y, 5.0, places=6)

    def test_positive_x_velocity(self):
        """Obs at (0,0) t=0.0 and (1.5, 0) t=0.1 -> vx=15.0 m/s."""
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("p1", 1, 0.1, 1.5, 0.0),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
            timestep=0.1,
        )
        pred = ConstantVelocityPredictor().predict(snap, "p1", horizon=2, prediction_interval=0.1)
        self.assertAlmostEqual(pred.predictions[0].x, 3.0, places=6)
        self.assertAlmostEqual(pred.predictions[1].x, 4.5, places=6)

    def test_negative_x_velocity(self):
        """Obs at t=0.0 (4.0, 0) and t=0.2 (2.0, 0) -> vx = -10.0 m/s.
        At source t=0.2, t_future=0.3: x = 2.0 + (-10.0) * 0.1 = 1.0.
        """
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 4.0, 0.0),
                    TrajectoryObservation("p1", 2, 0.2, 2.0, 0.0),
                ]
            },
            current_step=2,
            current_timestamp=0.2,
            timestep=0.1,
        )
        pred = ConstantVelocityPredictor().predict(snap, "p1", horizon=1, prediction_interval=0.1)
        self.assertAlmostEqual(pred.predictions[0].x, 1.0, places=6)
        self.assertAlmostEqual(pred.predictions[0].y, 0.0, places=6)

    def test_positive_y_velocity(self):
        """Obs at t=0.0 (0, 0) and t=0.1 (0, 1.0) -> vy = 10.0 m/s.
        At source t=0.1, t_future=0.2: y = 1.0 + 10.0 * 0.1 = 2.0.
        """
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("p1", 1, 0.1, 0.0, 1.0),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
            timestep=0.1,
        )
        pred = ConstantVelocityPredictor().predict(snap, "p1", horizon=1, prediction_interval=0.1)
        self.assertAlmostEqual(pred.predictions[0].x, 0.0, places=6)
        self.assertAlmostEqual(pred.predictions[0].y, 2.0, places=6)

    def test_negative_y_velocity(self):
        """Obs at t=0.0 (0, 5.0) and t=0.1 (0, 3.0) -> vy = -20.0 m/s.
        At source t=0.1, t_future=0.2: y = 3.0 + (-20.0) * 0.1 = 1.0.
        """
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 0.0, 5.0),
                    TrajectoryObservation("p1", 1, 0.1, 0.0, 3.0),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
            timestep=0.1,
        )
        pred = ConstantVelocityPredictor().predict(snap, "p1", horizon=1, prediction_interval=0.1)
        self.assertAlmostEqual(pred.predictions[0].y, 1.0, places=6)

    def test_one_observation_history_raises_validation_error(self):
        """Predictor must not invent velocity when only 1 observation is available."""
        snap = self._make_snapshot(
            {"p1": [TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0)]},
            current_step=1,
            current_timestamp=0.1,
        )
        with self.assertRaises(TrajectoryValidationError) as ctx:
            ConstantVelocityPredictor().predict(snap, "p1")
        self.assertIn("Insufficient history", str(ctx.exception))

    def test_zero_observation_history_raises_validation_error(self):
        snap = self._make_snapshot({}, current_step=0, current_timestamp=0.0)
        with self.assertRaises(TrajectoryValidationError) as ctx:
            ConstantVelocityPredictor().predict(snap, "unknown_agent")
        self.assertIn("Insufficient history", str(ctx.exception))

    def test_zero_or_negative_horizon_raises_validation_error(self):
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("p1", 1, 0.1, 1.0, 0.0),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
        )
        predictor = ConstantVelocityPredictor()
        with self.assertRaises(TrajectoryValidationError):
            predictor.predict(snap, "p1", horizon=0)
        with self.assertRaises(TrajectoryValidationError):
            predictor.predict(snap, "p1", horizon=-5)

    def test_invalid_prediction_interval_raises_validation_error(self):
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("p1", 1, 0.1, 1.0, 0.0),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
        )
        predictor = ConstantVelocityPredictor()
        with self.assertRaises(TrajectoryValidationError):
            predictor.predict(snap, "p1", prediction_interval=0.0)
        with self.assertRaises(TrajectoryValidationError):
            predictor.predict(snap, "p1", prediction_interval=-0.1)
        with self.assertRaises(TrajectoryValidationError):
            predictor.predict(snap, "p1", prediction_interval=float("nan"))

    def test_multiple_pedestrians_independent_predictions(self):
        """Verify independent multi-pedestrian predictions."""
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("p1", 1, 0.1, 1.0, 0.0),
                ],
                "p2": [
                    TrajectoryObservation("p2", 0, 0.0, 10.0, 0.0),
                    TrajectoryObservation("p2", 1, 0.1, 10.0, 2.0),
                ],
            },
            current_step=5,
            current_timestamp=0.5,
            timestep=0.1,
        )
        predictor = ConstantVelocityPredictor()
        pred1 = predictor.predict(snap, "p1", horizon=1)
        pred2 = predictor.predict(snap, "p2", horizon=1)
        self.assertEqual(pred1.agent_id, "p1")
        self.assertEqual(pred2.agent_id, "p2")
        # p1: vx=10, vy=0. source_t=0.5, t_future=0.6. x=1.0 + 10.0 * (0.6 - 0.1) = 6.0
        self.assertAlmostEqual(pred1.predictions[0].x, 6.0, places=5)
        # p2: vx=0, vy=20. source_t=0.5, t_future=0.6. y=2.0 + 20.0 * (0.6 - 0.1) = 12.0
        self.assertAlmostEqual(pred2.predictions[0].y, 12.0, places=5)

    def test_correct_future_timestamps(self):
        """Verify predicted timestamps are strictly increasing and strictly greater than source timestamp."""
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("p1", 1, 0.1, 0.5, 0.0),
                ]
            },
            current_step=10,
            current_timestamp=1.0,
            timestep=0.1,
        )
        pred = ConstantVelocityPredictor().predict(snap, "p1", horizon=3, prediction_interval=0.1)
        self.assertAlmostEqual(pred.predictions[0].timestamp, 1.1, places=6)
        self.assertAlmostEqual(pred.predictions[1].timestamp, 1.2, places=6)
        self.assertAlmostEqual(pred.predictions[2].timestamp, 1.3, places=6)
        for p in pred.predictions:
            self.assertGreater(p.timestamp, pred.source_timestamp)

    def test_prediction_output_metadata(self):
        """Verify all metadata fields on TrajectoryPrediction."""
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 5, 0.5, 0.0, 0.0),
                    TrajectoryObservation("p1", 6, 0.6, 1.0, 0.0),
                ]
            },
            current_step=10,
            current_timestamp=1.0,
        )
        pred = ConstantVelocityPredictor().predict(snap, "p1", horizon=2)
        self.assertEqual(pred.agent_id, "p1")
        self.assertEqual(pred.source_step_index, 10)
        self.assertAlmostEqual(pred.source_timestamp, 1.0)
        self.assertEqual(len(pred.predictions), 2)
        self.assertEqual(pred.category, TrajectoryCategory.PREDICTED)

    def test_no_mutation_of_original_observations(self):
        """Prediction must never modify observations within the snapshot."""
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 1.0, 2.0),
                    TrajectoryObservation("p1", 1, 0.1, 2.0, 4.0),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
        )
        orig_obs = snap.trajectories["p1"][0]
        ConstantVelocityPredictor().predict(snap, "p1", horizon=5)
        self.assertEqual(snap.trajectories["p1"][0].x, orig_obs.x)
        self.assertEqual(snap.trajectories["p1"][0].y, orig_obs.y)
        self.assertEqual(snap.trajectories["p1"][0].timestamp, orig_obs.timestamp)

    def test_no_mutation_of_simulation_owned_pedestrian_state(self):
        """Prediction must never modify Person A's Pedestrian objects or Simulation state."""
        ped = Pedestrian(
            id="ped_99",
            position=(10.0, 20.0),
            velocity=(1.0, 0.5),
            status=PedestrianStatus.ACTIVE,
        )
        # Record observations from ped state
        buf = TrajectoryHistoryBuffer(window_size=10)
        buf.record(
            TrajectoryObservation(ped.id, 0, 0.0, ped.position[0], ped.position[1])
        )
        buf.record(
            TrajectoryObservation(ped.id, 1, 0.1, ped.position[0] + 0.1, ped.position[1] + 0.05)
        )
        snap = buf.get_snapshot(current_step=1, current_timestamp=0.1, timestep=0.1)

        # Run predictor
        pred = ConstantVelocityPredictor().predict(snap, ped.id, horizon=5)
        self.assertEqual(len(pred.predictions), 5)

        # Confirm pedestrian owned by Person A is completely unaltered
        self.assertEqual(ped.position, (10.0, 20.0))
        self.assertEqual(ped.velocity, (1.0, 0.5))
        self.assertEqual(ped.status, PedestrianStatus.ACTIVE)

    def test_predict_batch_returns_results_and_errors(self):
        """Batch predict returns dict with predictions for valid agents and errors for insufficient agents."""
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("p1", 1, 0.1, 1.0, 0.0),
                ],
                "p2": [
                    TrajectoryObservation("p2", 0, 0.0, 5.0, 5.0),
                ],
            },
            current_step=1,
            current_timestamp=0.1,
        )
        results = ConstantVelocityPredictor().predict_batch(snap, horizon=2)
        self.assertIsInstance(results["p1"], TrajectoryPrediction)
        self.assertIsInstance(results["p2"], TrajectoryValidationError)

    def test_all_predicted_coordinates_are_finite(self):
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 100.0, 200.0),
                    TrajectoryObservation("p1", 1, 0.1, 101.0, 202.0),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
        )
        pred = ConstantVelocityPredictor().predict(snap, "p1", horizon=10)
        for pos in pred.predictions:
            self.assertTrue(math.isfinite(pos.x))
            self.assertTrue(math.isfinite(pos.y))
            self.assertTrue(math.isfinite(pos.timestamp))

    def test_prediction_immutability(self):
        snap = self._make_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("p1", 1, 0.1, 1.0, 0.0),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
        )
        pred = ConstantVelocityPredictor().predict(snap, "p1", horizon=2)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            pred.agent_id = "mutated"  # type: ignore


if __name__ == "__main__":
    unittest.main()
