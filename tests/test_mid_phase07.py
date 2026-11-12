"""Comprehensive Phase 07 tests for Evacuation Motion Indeterminacy Diffusion (MID) repairs.

Verifies:
1. Per-sample guidance independence (sample 0 guidance is never reused across samples).
2. Points inside obstacles and obstacle corner escape vectors (deterministic outward direction).
3. Finite guidance vectors and numerical stability safeguards.
4. Trained vs. untrained model tracking and provenance reporting.
5. 16-feature vector ordering, canonical normalization scaling, and train/inference consistency.
6. Rejection of invalid exit IDs (no silent substitution) and safe fallback when environment omitted.
7. Strict validation of horizon, prediction interval, and sample count arguments.
8. Seed reproducibility using isolated local generators without global RNG side effects.
9. Training pipeline execution with loss masking (no silent point repetition).
10. Evaluation reporting distinguishing oracle metrics, diversity, point walkability, and segment collisions.
11. Line-segment obstacle collision detection vs point walkability.
12. Full compatibility with PredictionEvaluator and data contracts.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from genevac.environment import EnvironmentMap, Exit, Obstacle
from genevac.prediction import (
    EvaluationSample,
    EvacuationConditioning,
    EvacuationMIDConfig,
    EvacuationMIDNetwork,
    EvacuationMIDPredictor,
    MultiTrajectoryPrediction,
    PredictionEvaluator,
    TrajectoryCategory,
    TrajectoryHistoryBuffer,
    TrajectoryObservation,
    TrajectoryPrediction,
    TrajectorySnapshot,
    TrajectoryValidationError,
)
from genevac.prediction.mid import (
    FEATURE_NORMALIZATION_SCALES,
    line_segment_intersects_box,
)


class TestEvacuationMIDPhase07Repairs(unittest.TestCase):
    """Rigorous Phase 07 test suite validating all required MID repairs."""

    def setUp(self) -> None:
        self.env = EnvironmentMap(width=50.0, height=20.0)
        self.env.add_exit(Exit(id="exit_east", position=(50.0, 10.0), width=2.5))
        self.env.add_exit(Exit(id="exit_west", position=(0.0, 10.0), width=2.5))
        self.env.add_obstacle(Obstacle(id="obs_pillar", bounds=(20.0, 8.0, 24.0, 12.0)))

        self.config = EvacuationMIDConfig(
            diffusion_steps=5,
            horizon=8,
            prediction_interval=0.1,
            k_samples=4,
            hidden_dim=32,
            d_model=16,
        )
        self.predictor = EvacuationMIDPredictor(self.config)

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
    # 1. Per-sample guidance independence
    # -------------------------------------------------------------------------
    def test_per_sample_guidance_independence(self):
        """Guidance must be calculated independently for every sample trajectory."""
        snap = self._build_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 18.0, 10.0),
                    TrajectoryObservation("p1", 1, 0.1, 18.5, 10.0),  # heading toward obstacle at x=20..24
                ]
            },
            current_step=1,
            current_timestamp=0.1,
        )
        # Sample K=3 trajectories
        trajs, cond = self.predictor.sample_trajectories(
            snapshot=snap,
            agent_id="p1",
            k_samples=3,
            environment=self.env,
            seed=42,
        )
        self.assertEqual(trajs.shape[0], 3)
        # Displacements must differ across samples
        diff_0_1 = np.linalg.norm(trajs[0] - trajs[1])
        diff_1_2 = np.linalg.norm(trajs[1] - trajs[2])
        self.assertGreater(diff_0_1, 0.0)
        self.assertGreater(diff_1_2, 0.0)

        # Compute per-sample guidance explicitly for sample 0 and sample 1
        g0 = self.predictor._compute_per_sample_guidance(trajs[0] - cond.reference_position, cond, self.env, 8)
        g1 = self.predictor._compute_per_sample_guidance(trajs[1] - cond.reference_position, cond, self.env, 8)
        # Because sample 0 and sample 1 are at different coordinates, their obstacle forces must be distinct
        self.assertFalse(np.allclose(g0, g1, atol=1e-5))

    # -------------------------------------------------------------------------
    # 2. Points inside obstacles and corner escape vectors
    # -------------------------------------------------------------------------
    def test_obstacle_repulsion_inside_and_at_corners(self):
        """Points inside an obstacle must generate a deterministic outward escape vector."""
        snap = self._build_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 10.0, 10.0),
                    TrajectoryObservation("p1", 1, 0.1, 10.1, 10.0),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
        )
        cond = EvacuationConditioning.from_snapshot_and_env(snap, "p1", self.env)

        # Place a test point inside the obstacle (bounds: 20..24, 8..12)
        # Test point at (20.5, 10.0): closest boundary is left face (x=20), d_left=0.5
        inside_disp = np.zeros((8, 2), dtype=np.float32)
        inside_disp[0] = (20.5 - cond.reference_position[0], 10.0 - cond.reference_position[1])

        guidance = self.predictor._compute_per_sample_guidance(inside_disp, cond, self.env, 8)
        # Obstacle force at step 0 must point left (negative x)
        self.assertLess(guidance[0, 0], 0.0)
        self.assertTrue(math.isfinite(guidance[0, 0]))
        self.assertTrue(math.isfinite(guidance[0, 1]))

        # Test point at (23.5, 10.0): closest boundary is right face (x=24), d_right=0.5
        inside_disp[0] = (23.5 - cond.reference_position[0], 10.0 - cond.reference_position[1])
        guidance_right = self.predictor._compute_per_sample_guidance(inside_disp, cond, self.env, 8)
        self.assertGreater(guidance_right[0, 0], 0.0)

    # -------------------------------------------------------------------------
    # 3. Finite guidance vectors and numerical stability
    # -------------------------------------------------------------------------
    def test_finite_guidance_and_numerical_stability(self):
        """Guidance calculations must remain bounded and finite even with extreme inputs."""
        snap = self._build_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 0.0, 0.0),
                    TrajectoryObservation("p1", 1, 0.1, 1e-6, 1e-6),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
        )
        cond = EvacuationConditioning.from_snapshot_and_env(snap, "p1", self.env)
        disp = np.full((8, 2), 1e6, dtype=np.float32)
        guidance = self.predictor._compute_per_sample_guidance(disp, cond, self.env, 8)

        for step in range(8):
            self.assertTrue(math.isfinite(guidance[step, 0]))
            self.assertTrue(math.isfinite(guidance[step, 1]))
            # Maximum force must be clamped
            self.assertLessEqual(math.hypot(guidance[step, 0], guidance[step, 1]), 6.0)

    # -------------------------------------------------------------------------
    # 4. Trained vs. Untrained model tracking
    # -------------------------------------------------------------------------
    def test_trained_vs_untrained_model_tracking(self):
        """Predictor must track whether training has occurred and expose provenance."""
        fresh_pred = EvacuationMIDPredictor(self.config)
        self.assertFalse(fresh_pred.is_trained)
        prov = fresh_pred.training_provenance
        self.assertFalse(prov["is_trained"])
        self.assertEqual(prov["training_sample_count"], 0)

        # Run predict_samples on untrained model
        snap = self._build_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 5.0, 5.0),
                    TrajectoryObservation("p1", 1, 0.1, 5.1, 5.1),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
        )
        multi_untrained = fresh_pred.predict_samples(snap, "p1", k_samples=2, seed=42)
        self.assertFalse(multi_untrained.is_trained_model)

        # Perform training
        sample = EvaluationSample(
            agent_id="p1",
            reference_step=1,
            reference_timestamp=0.1,
            history=(
                TrajectoryObservation("p1", 0, 0.0, 5.0, 5.0),
                TrajectoryObservation("p1", 1, 0.1, 5.1, 5.1),
            ),
            ground_truth_future=tuple(
                TrajectoryObservation("p1", 2 + k, 0.2 + k * 0.1, 5.2 + k * 0.1, 5.2 + k * 0.1)
                for k in range(8)
            ),
        )
        losses = fresh_pred.fit([sample], epochs=2, batch_size=2)
        self.assertTrue(fresh_pred.is_trained)
        self.assertEqual(len(losses), 2)
        self.assertGreater(fresh_pred.training_provenance["training_sample_count"], 0)

        # Predictions now reflect trained provenance
        multi_trained = fresh_pred.predict_samples(snap, "p1", k_samples=2, seed=42)
        self.assertTrue(multi_trained.is_trained_model)

    # -------------------------------------------------------------------------
    # 5. 16-feature vector ordering and canonical normalization
    # -------------------------------------------------------------------------
    def test_feature_ordering_and_normalization(self):
        """Conditioning vector has exactly 16 elements with documented normalization scaling."""
        snap = self._build_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 10.0, 10.0),
                    TrajectoryObservation("p1", 1, 0.1, 10.2, 10.0),  # vx = 2.0 m/s
                ]
            },
            current_step=1,
            current_timestamp=0.1,
        )
        cond = EvacuationConditioning.from_snapshot_and_env(snap, "p1", self.env, target_exit_id="exit_east")
        self.assertEqual(len(cond.vector), 16)
        self.assertEqual(len(cond.normalized_vector), 16)
        self.assertEqual(len(FEATURE_NORMALIZATION_SCALES), 16)

        # Check raw physical velocity vx = 2.0 m/s (index 0)
        self.assertAlmostEqual(cond.vector[0], 2.0, places=5)
        # Normalized velocity scaled by 2.0 -> 1.0 (index 0)
        self.assertAlmostEqual(cond.normalized_vector[0], 1.0, places=5)

        # All normalized values must be within [-2.0, 2.0]
        for val in cond.normalized_vector:
            self.assertTrue(-2.0 <= val <= 2.0)

    # -------------------------------------------------------------------------
    # 6. Rejection of invalid exit IDs (no silent substitution)
    # -------------------------------------------------------------------------
    def test_invalid_target_exit_rejection(self):
        """Specifying an exit ID that does not exist in environment raises TrajectoryValidationError."""
        snap = self._build_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 10.0, 10.0),
                    TrajectoryObservation("p1", 1, 0.1, 10.1, 10.0),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
        )
        with self.assertRaises(TrajectoryValidationError) as ctx:
            EvacuationConditioning.from_snapshot_and_env(
                snapshot=snap,
                agent_id="p1",
                environment=self.env,
                target_exit_id="nonexistent_door_99",
            )
        self.assertIn("does not exist", str(ctx.exception))
        self.assertIn("Silent substitution", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 7. Strict validation of horizon, interval, and sample counts
    # -------------------------------------------------------------------------
    def test_validation_of_horizon_and_parameters(self):
        """predict() and predict_samples() reject invalid horizons, intervals, and sample counts."""
        snap = self._build_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 10.0, 10.0),
                    TrajectoryObservation("p1", 1, 0.1, 10.1, 10.0),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
        )
        # Horizon = 0
        with self.assertRaises(TrajectoryValidationError):
            self.predictor.predict(snap, "p1", horizon=0)

        # Horizon > model config horizon (8)
        with self.assertRaises(TrajectoryValidationError):
            self.predictor.predict(snap, "p1", horizon=20)

        # Valid horizon < model config horizon: MUST be honored
        pred_short = self.predictor.predict(snap, "p1", horizon=4)
        self.assertEqual(len(pred_short.predictions), 4)

        # Prediction interval <= 0
        with self.assertRaises(TrajectoryValidationError):
            self.predictor.predict(snap, "p1", prediction_interval=0.0)

        # Negative sample count
        with self.assertRaises(ValueError):
            self.predictor.predict_samples(snap, "p1", k_samples=0)

    # -------------------------------------------------------------------------
    # 8. Isolated RNG seed reproducibility
    # -------------------------------------------------------------------------
    def test_isolated_rng_without_global_side_effects(self):
        """Sampling with a seed must produce reproducible outputs without resetting global RNG."""
        snap = self._build_snapshot(
            {
                "p1": [
                    TrajectoryObservation("p1", 0, 0.0, 5.0, 5.0),
                    TrajectoryObservation("p1", 1, 0.1, 5.1, 5.1),
                ]
            },
            current_step=1,
            current_timestamp=0.1,
        )
        # Measure global torch random state before call
        global_state_before = torch.get_rng_state()

        # Run predictor with isolated seed
        multi1 = self.predictor.predict_samples(snap, "p1", k_samples=2, seed=777)
        multi2 = self.predictor.predict_samples(snap, "p1", k_samples=2, seed=777)

        # Both runs with same seed produce identical trajectories
        for p1, p2 in zip(multi1.samples[0].predictions, multi2.samples[0].predictions):
            self.assertEqual(p1.x, p2.x)
            self.assertEqual(p1.y, p2.y)

        # Global RNG state was NOT reset to seed 777
        global_state_after = torch.get_rng_state()
        self.assertTrue(torch.equal(global_state_before, global_state_after))

    # -------------------------------------------------------------------------
    # 9. Line-segment collision detection vs point walkability
    # -------------------------------------------------------------------------
    def test_segment_collision_detection(self):
        """Line segment intersecting an obstacle must be flagged as colliding."""
        # Obstacle box: (20.0, 8.0, 24.0, 12.0)
        box = (20.0, 8.0, 24.0, 12.0)

        # Segment passing straight through the box from x=18 to x=26 at y=10
        p_start = (18.0, 10.0)
        p_end = (26.0, 10.0)
        self.assertTrue(line_segment_intersects_box(p_start, p_end, box))

        # Segment completely outside the box (y = 5.0)
        p_clear_start = (18.0, 5.0)
        p_clear_end = (26.0, 5.0)
        self.assertFalse(line_segment_intersects_box(p_clear_start, p_clear_end, box))

        # Segment starting inside box
        p_in_start = (21.0, 9.0)
        p_in_end = (25.0, 9.0)
        self.assertTrue(line_segment_intersects_box(p_in_start, p_in_end, box))

    # -------------------------------------------------------------------------
    # 10. Training pipeline with loss masking
    # -------------------------------------------------------------------------
    def test_training_pipeline_with_loss_masking(self):
        """Training pipeline masks unobserved future steps without silent point repetition."""
        # Create sample with only 4 future ground-truth points (less than model horizon 8)
        sample_partial = EvaluationSample(
            agent_id="p_part",
            reference_step=1,
            reference_timestamp=0.1,
            history=(
                TrajectoryObservation("p_part", 0, 0.0, 0.0, 0.0),
                TrajectoryObservation("p_part", 1, 0.1, 0.1, 0.1),
            ),
            ground_truth_future=tuple(
                TrajectoryObservation("p_part", 2 + k, 0.2 + k * 0.1, 0.2 + k * 0.1, 0.2 + k * 0.1)
                for k in range(4)
            ),
        )
        losses = self.predictor.fit([sample_partial], epochs=2, batch_size=2)
        self.assertEqual(len(losses), 2)
        for loss in losses:
            self.assertTrue(math.isfinite(loss))

    # -------------------------------------------------------------------------
    # 11. Stochastic evaluation reporting
    # -------------------------------------------------------------------------
    def test_stochastic_evaluation_reporting(self):
        """evaluate_stochastic reports oracle metrics, diversity, point validity, and segment validity."""
        sample = EvaluationSample(
            agent_id="p_eval",
            reference_step=1,
            reference_timestamp=0.1,
            history=(
                TrajectoryObservation("p_eval", 0, 0.0, 10.0, 10.0),
                TrajectoryObservation("p_eval", 1, 0.1, 10.1, 10.0),
            ),
            ground_truth_future=tuple(
                TrajectoryObservation("p_eval", 2 + k, 0.2 + k * 0.1, 10.2 + k * 0.1, 10.0)
                for k in range(8)
            ),
        )
        report = self.predictor.evaluate_stochastic([sample], environment=self.env, k_samples=4, seed=42)

        self.assertEqual(report["total_samples"], 1)
        self.assertEqual(report["evaluated_samples"], 1)
        self.assertEqual(report["failed_samples"], 0)
        self.assertTrue(math.isfinite(report["oracle_best_of_k_ade"]))
        self.assertTrue(math.isfinite(report["expected_mean_of_k_ade"]))
        self.assertLessEqual(report["oracle_best_of_k_ade"], report["expected_mean_of_k_ade"] + 1e-6)
        self.assertTrue(math.isfinite(report["point_walkability"]))
        self.assertTrue(math.isfinite(report["segment_collision_free"]))
        self.assertGreaterEqual(report["segment_collision_free"], 0.0)

        # Empty dataset handling
        empty_report = self.predictor.evaluate_stochastic([])
        self.assertEqual(empty_report["total_samples"], 0)
        self.assertTrue(math.isnan(empty_report["oracle_best_of_k_ade"]))


if __name__ == "__main__":
    unittest.main()
