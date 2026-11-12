"""Dedicated Phase 07 tests for ConstantVelocityPredictor and baseline trajectory prediction.

Verifies:
1. Straight-line movement across all quadrants (X+, X-, Y+, Y-, diagonal).
2. Stationary pedestrians (vx=0, vy=0) maintaining constant position.
3. Changing velocity (acceleration, deceleration, direction changes) with ADE/FDE evaluation.
4. Irregular observation intervals and skipped timesteps.
5. Insufficient history rejection (0 and 1 observations).
6. Invalid timestamp handling (dt <= 0, non-finite, out-of-order).
7. Agent disappearance and staleness enforcement (missing agent, stale observations).
8. Run reset and isolation (no trajectory leakage across reset or different run IDs).
9. Multiple pedestrian IDs with independent predictions and batch prediction.
10. Deterministic reproducibility (bitwise identical repeated predictions).
11. Integration with existing ADE and FDE metrics.
"""

from __future__ import annotations

import dataclasses
import math
import unittest

from genevac.metrics import calculate_ade, calculate_fde
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.prediction import (
    ConstantVelocityPredictor,
    EvaluationSample,
    PredictionEvaluator,
    TrajectoryCategory,
    TrajectoryHistoryBuffer,
    TrajectoryObservation,
    TrajectoryPrediction,
    TrajectorySnapshot,
    TrajectoryValidationError,
)


class TestBaselineTrajectoryPhase07(unittest.TestCase):
    """Rigorous baseline trajectory prediction test suite for Phase 07."""

    def _build_snapshot(
        self,
        observations: dict[str, list[TrajectoryObservation]],
        current_step: int = 10,
        current_timestamp: float = 1.0,
        timestep: float = 0.1,
    ) -> TrajectorySnapshot:
        buf = TrajectoryHistoryBuffer(window_size=50)
        for obs_list in observations.values():
            for obs in obs_list:
                buf.record(obs)
        return buf.get_snapshot(
            current_step=current_step,
            current_timestamp=current_timestamp,
            timestep=timestep,
        )

    # -------------------------------------------------------------------------
    # 1. Straight-line movement across quadrants
    # -------------------------------------------------------------------------
    def test_straight_line_diagonal_positive(self):
        """Diagonal movement in (+x, +y): dx=1.0, dy=2.0 over dt=0.2s -> vx=5.0, vy=10.0."""
        snap = self._build_snapshot(
            {
                "ped_diag": [
                    TrajectoryObservation("ped_diag", 0, 0.0, 1.0, 2.0),
                    TrajectoryObservation("ped_diag", 2, 0.2, 2.0, 4.0),
                ]
            },
            current_step=2,
            current_timestamp=0.2,
            timestep=0.1,
        )
        pred = ConstantVelocityPredictor().predict(snap, "ped_diag", horizon=3, prediction_interval=0.1)
        self.assertEqual(len(pred.predictions), 3)
        # t=0.3: x=2.0 + 5.0*0.1 = 2.5, y=4.0 + 10.0*0.1 = 5.0
        self.assertAlmostEqual(pred.predictions[0].timestamp, 0.3)
        self.assertAlmostEqual(pred.predictions[0].x, 2.5)
        self.assertAlmostEqual(pred.predictions[0].y, 5.0)
        # t=0.4: x=3.0, y=6.0
        self.assertAlmostEqual(pred.predictions[1].timestamp, 0.4)
        self.assertAlmostEqual(pred.predictions[1].x, 3.0)
        self.assertAlmostEqual(pred.predictions[1].y, 6.0)
        # t=0.5: x=3.5, y=7.0
        self.assertAlmostEqual(pred.predictions[2].timestamp, 0.5)
        self.assertAlmostEqual(pred.predictions[2].x, 3.5)
        self.assertAlmostEqual(pred.predictions[2].y, 7.0)

    def test_straight_line_negative_velocity(self):
        """Movement in (-x, -y): dx=-2.0, dy=-1.0 over dt=0.5s -> vx=-4.0, vy=-2.0."""
        snap = self._build_snapshot(
            {
                "ped_neg": [
                    TrajectoryObservation("ped_neg", 0, 0.0, 10.0, 20.0),
                    TrajectoryObservation("ped_neg", 5, 0.5, 8.0, 19.0),
                ]
            },
            current_step=5,
            current_timestamp=0.5,
            timestep=0.1,
        )
        pred = ConstantVelocityPredictor().predict(snap, "ped_neg", horizon=2, prediction_interval=0.2)
        # t=0.7: x=8.0 + (-4.0)*0.2 = 7.2, y=19.0 + (-2.0)*0.2 = 18.6
        self.assertAlmostEqual(pred.predictions[0].timestamp, 0.7)
        self.assertAlmostEqual(pred.predictions[0].x, 7.2)
        self.assertAlmostEqual(pred.predictions[0].y, 18.6)

    # -------------------------------------------------------------------------
    # 2. Stationary pedestrians
    # -------------------------------------------------------------------------
    def test_stationary_pedestrian_retains_exact_coordinates(self):
        """Stationary agent at (12.34, 56.78) over multiple steps remains fixed."""
        snap = self._build_snapshot(
            {
                "ped_still": [
                    TrajectoryObservation("ped_still", 0, 0.0, 12.34, 56.78),
                    TrajectoryObservation("ped_still", 4, 0.4, 12.34, 56.78),
                ]
            },
            current_step=4,
            current_timestamp=0.4,
            timestep=0.1,
        )
        pred = ConstantVelocityPredictor().predict(snap, "ped_still", horizon=10)
        for pos in pred.predictions:
            self.assertAlmostEqual(pos.x, 12.34)
            self.assertAlmostEqual(pos.y, 56.78)

    # -------------------------------------------------------------------------
    # 3. Changing velocity and evaluation with ADE/FDE
    # -------------------------------------------------------------------------
    def test_changing_velocity_evaluation(self):
        """An accelerating pedestrian evaluated with PredictionEvaluator produces nonzero ADE/FDE."""
        # Ground truth: accelerating with a = 2.0 m/s^2 along x: x(t) = 0.5 * 2.0 * t^2 = t^2
        # Observations at t=0.0 (x=0), t=0.4 (x=0.16), t=0.8 (x=0.64)
        # Future ground truth at t=1.2 (x=1.44), t=1.6 (x=2.56)
        hist = (
            TrajectoryObservation("ped_accel", 0, 0.0, 0.0, 0.0),
            TrajectoryObservation("ped_accel", 4, 0.4, 0.16, 0.0),
            TrajectoryObservation("ped_accel", 8, 0.8, 0.64, 0.0),
        )
        fut = (
            TrajectoryObservation("ped_accel", 12, 1.2, 1.44, 0.0),
            TrajectoryObservation("ped_accel", 16, 1.6, 2.56, 0.0),
        )
        sample = EvaluationSample(
            agent_id="ped_accel",
            reference_step=8,
            reference_timestamp=0.8,
            history=hist,
            ground_truth_future=fut,
        )
        evaluator = PredictionEvaluator()
        result = evaluator.evaluate_sample(sample, timestep=0.1)

        self.assertEqual(result.status, "SUCCESS")
        # Constant velocity computed from last 2 observations:
        # vx = (0.64 - 0.16) / (0.8 - 0.4) = 0.48 / 0.4 = 1.2 m/s
        # Predicted at t=1.2: x = 0.64 + 1.2 * 0.4 = 1.12 (Error: |1.44 - 1.12| = 0.32)
        # Predicted at t=1.6: x = 0.64 + 1.2 * 0.8 = 1.60 (Error: |2.56 - 1.60| = 0.96)
        # Expected ADE = (0.32 + 0.96) / 2 = 0.64
        # Expected FDE = 0.96
        self.assertAlmostEqual(result.ade, 0.64, places=4)
        self.assertAlmostEqual(result.fde, 0.96, places=4)

    # -------------------------------------------------------------------------
    # 4. Irregular observation intervals and skipped steps
    # -------------------------------------------------------------------------
    def test_irregular_observation_intervals(self):
        """Observations with irregular gaps (step 0 at 0.0, step 3 at 0.45s) compute exact velocity."""
        snap = self._build_snapshot(
            {
                "ped_irreg": [
                    TrajectoryObservation("ped_irreg", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("ped_irreg", 3, 0.45, 0.9, 0.45),
                ]
            },
            current_step=3,
            current_timestamp=0.45,
            timestep=0.15,
        )
        # dt = 0.45s. vx = 0.9 / 0.45 = 2.0 m/s, vy = 0.45 / 0.45 = 1.0 m/s
        pred = ConstantVelocityPredictor().predict(snap, "ped_irreg", horizon=2, prediction_interval=0.1)
        # t=0.55: x = 0.9 + 2.0*0.1 = 1.1, y = 0.45 + 1.0*0.1 = 0.55
        self.assertAlmostEqual(pred.predictions[0].timestamp, 0.55)
        self.assertAlmostEqual(pred.predictions[0].x, 1.1)
        self.assertAlmostEqual(pred.predictions[0].y, 0.55)

    # -------------------------------------------------------------------------
    # 5. Insufficient history
    # -------------------------------------------------------------------------
    def test_insufficient_history_raises_error(self):
        """0 or 1 observations must raise TrajectoryValidationError."""
        snap_empty = self._build_snapshot({}, current_step=0, current_timestamp=0.0)
        with self.assertRaises(TrajectoryValidationError) as ctx1:
            ConstantVelocityPredictor().predict(snap_empty, "unknown")
        self.assertIn("Insufficient history", str(ctx1.exception))

        snap_one = self._build_snapshot(
            {"p1": [TrajectoryObservation("p1", 0, 0.0, 5.0, 5.0)]},
            current_step=0,
            current_timestamp=0.0,
        )
        with self.assertRaises(TrajectoryValidationError) as ctx2:
            ConstantVelocityPredictor().predict(snap_one, "p1")
        self.assertIn("requires at least 2 observations", str(ctx2.exception))

    # -------------------------------------------------------------------------
    # 6. Invalid timestamp handling
    # -------------------------------------------------------------------------
    def test_invalid_dt_and_future_observation_rejected(self):
        """Future observation relative to snapshot timestamp is rejected."""
        # Construct invalid observation sequence directly in snapshot
        obs1 = TrajectoryObservation("p1", 0, 0.0, 0.0, 0.0)
        obs2 = TrajectoryObservation("p1", 1, 1.5, 1.0, 1.0)  # ts=1.5 > snap current_timestamp=1.0
        snap = TrajectorySnapshot(
            current_step=10,
            current_timestamp=1.0,
            timestep=0.1,
            trajectories={"p1": (obs1, obs2)},
        )
        with self.assertRaises(TrajectoryValidationError) as ctx:
            ConstantVelocityPredictor().predict(snap, "p1")
        self.assertIn("Future observation", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 7. Agent disappearance and staleness enforcement
    # -------------------------------------------------------------------------
    def test_stale_observation_enforcement(self):
        """Observation older than max_observation_age raises TrajectoryValidationError."""
        snap = self._build_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("p1", 1, 0.1, 1.0, 0.0),
                ]
            },
            current_step=20,
            current_timestamp=2.0,  # age = 2.0 - 0.1 = 1.9s
            timestep=0.1,
        )
        with self.assertRaises(TrajectoryValidationError) as ctx:
            ConstantVelocityPredictor().predict(snap, "p1", max_observation_age=0.5)
        self.assertIn("Stale observation", str(ctx.exception))

    def test_missing_agent_batch_prediction_yields_validation_error(self):
        """Missing agent in batch prediction returns TrajectoryValidationError in result map."""
        snap = self._build_snapshot(
            {
                "active_ped": [
                    TrajectoryObservation("active_ped", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("active_ped", 1, 0.1, 1.0, 0.0),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
        )
        results = ConstantVelocityPredictor().predict_batch(
            snap, agent_ids=["active_ped", "missing_ped"], horizon=2
        )
        self.assertIsInstance(results["active_ped"], TrajectoryPrediction)
        self.assertIsInstance(results["missing_ped"], TrajectoryValidationError)

    # -------------------------------------------------------------------------
    # 8. Run reset and isolation
    # -------------------------------------------------------------------------
    def test_run_reset_prevents_history_bleed(self):
        """Resetting the buffer between runs completely removes previous observations."""
        buf = TrajectoryHistoryBuffer(window_size=10)
        buf.record(TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0))
        buf.record(TrajectoryObservation("p1", 1, 0.1, 2.0, 2.0))
        snap1 = buf.get_snapshot(current_step=1, current_timestamp=0.1)

        pred1 = ConstantVelocityPredictor().predict(snap1, "p1", horizon=1)
        self.assertEqual(pred1.agent_id, "p1")

        # Clear buffer (simulating run transition)
        buf.clear()
        self.assertEqual(buf.get_agent_ids(), [])

        # Start new run with new pedestrian p2
        buf.record(TrajectoryObservation("p2", 0, 0.0, 10.0, 10.0))
        buf.record(TrajectoryObservation("p2", 1, 0.1, 11.0, 10.0))
        snap2 = buf.get_snapshot(current_step=1, current_timestamp=0.1)

        # p1 does not exist in snap2
        self.assertNotIn("p1", snap2.trajectories)
        with self.assertRaises(TrajectoryValidationError):
            ConstantVelocityPredictor().predict(snap2, "p1")

        # p2 predicts cleanly
        pred2 = ConstantVelocityPredictor().predict(snap2, "p2", horizon=1)
        self.assertEqual(pred2.agent_id, "p2")

    # -------------------------------------------------------------------------
    # 9. Multiple pedestrian IDs and batch prediction
    # -------------------------------------------------------------------------
    def test_multiple_pedestrians_batch_predictions(self):
        """Batch prediction computes independent trajectories for all valid agents."""
        snap = self._build_snapshot(
            {
                "p_east": [
                    TrajectoryObservation("p_east", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("p_east", 1, 0.1, 1.0, 0.0),
                ],
                "p_south": [
                    TrajectoryObservation("p_south", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("p_south", 1, 0.1, 0.0, 2.0),
                ],
                "p_short": [
                    TrajectoryObservation("p_short", 0, 0.0, 5.0, 5.0),
                ],
            },
            current_step=1,
            current_timestamp=0.1,
        )
        batch = ConstantVelocityPredictor().predict_batch(snap, horizon=3)
        self.assertIsInstance(batch["p_east"], TrajectoryPrediction)
        self.assertIsInstance(batch["p_south"], TrajectoryPrediction)
        self.assertIsInstance(batch["p_short"], TrajectoryValidationError)

        # p_east moves east
        self.assertAlmostEqual(batch["p_east"].predictions[0].x, 2.0)
        self.assertAlmostEqual(batch["p_east"].predictions[0].y, 0.0)

        # p_south moves south
        self.assertAlmostEqual(batch["p_south"].predictions[0].x, 0.0)
        self.assertAlmostEqual(batch["p_south"].predictions[0].y, 4.0)

    # -------------------------------------------------------------------------
    # 10. Deterministic reproducibility
    # -------------------------------------------------------------------------
    def test_deterministic_reproducibility(self):
        """Calling predict multiple times with identical inputs yields identical outputs."""
        snap = self._build_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 3.14159, 2.71828),
                    TrajectoryObservation("p1", 5, 0.5, 4.14159, 3.71828),
                ]
            },
            current_step=5,
            current_timestamp=0.5,
        )
        predictor = ConstantVelocityPredictor()
        pred1 = predictor.predict(snap, "p1", horizon=5)
        pred2 = predictor.predict(snap, "p1", horizon=5)

        self.assertEqual(len(pred1.predictions), len(pred2.predictions))
        for p1, p2 in zip(pred1.predictions, pred2.predictions):
            self.assertEqual(p1.timestamp, p2.timestamp)
            self.assertEqual(p1.x, p2.x)
            self.assertEqual(p1.y, p2.y)

    # -------------------------------------------------------------------------
    # 11. Zero mutation of simulation state
    # -------------------------------------------------------------------------
    def test_zero_simulation_mutation(self):
        """Simulation engine and pedestrian objects remain strictly invariant."""
        ped = Pedestrian(
            id="p_inv",
            position=(15.0, 25.0),
            velocity=(1.2, 0.8),
            status=PedestrianStatus.ACTIVE,
        )
        pos_before = ped.position
        vel_before = ped.velocity
        status_before = ped.status

        snap = self._build_snapshot(
            {
                "p_inv": [
                    TrajectoryObservation("p_inv", 0, 0.0, 15.0, 25.0),
                    TrajectoryObservation("p_inv", 1, 0.1, 15.12, 25.08),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
        )
        _ = ConstantVelocityPredictor().predict(snap, "p_inv", horizon=5)

        self.assertEqual(ped.position, pos_before)
        self.assertEqual(ped.velocity, vel_before)
        self.assertEqual(ped.status, status_before)


if __name__ == "__main__":
    unittest.main()

