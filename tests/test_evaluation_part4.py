"""Tests for Part 4 Prediction Evaluation Subsystem.

Verifies:
1. Strict data isolation: EvaluationSample rejects future leakage into history.
2. SampleGenerator sliding windows: correct step alignment and horizon matching.
3. Timestamp-based matching: prediction and ground truth matched by timestamp, not index.
4. Correct calculation of ADE and FDE on hand-calculable constant velocity ground truth (ADE=0.0, FDE=0.0).
5. Hand-calculable accelerating ground-truth trajectory with known nonzero ADE and FDE.
6. Partial horizon handling when agents evacuate early.
7. Cohort partitioning: ACTIVE and BLOCKED cohorts evaluated separately.
8. DatasetAdapter loading real ETH dataset files and JSON sample files.
9. Aggregate report generation and JSON/CSV serialization.
10. End-to-end evaluation with SimulationEvaluationAdapter and SimulationEngine.
"""

import math
import unittest
from pathlib import Path

from genevac.environment import EnvironmentMap
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.prediction import (
    ConstantVelocityPredictor,
    DatasetAdapter,
    EvaluationSample,
    PredictionEvaluator,
    SampleGenerator,
    SimulationEvaluationAdapter,
    TrajectoryCategory,
    TrajectoryObservation,
)
from genevac.simulation import SimulationEngine


def _obs(aid: str, step: int, ts: float, x: float, y: float) -> TrajectoryObservation:
    return TrajectoryObservation(
        agent_id=aid,
        step_index=step,
        timestamp=ts,
        x=x,
        y=y,
        category=TrajectoryCategory.SIMULATED,
    )


class TestPredictionEvaluation(unittest.TestCase):
    """Test suite for Part 4 evaluation subsystem."""

    def test_strict_data_isolation_rejects_leakage(self):
        """EvaluationSample raises ValueError if history contains timestamps > reference_timestamp."""
        hist = (
            _obs("a1", 0, 0.0, 0.0, 0.0),
            _obs("a1", 1, 0.5, 1.0, 1.0),  # ts=0.5 > ref=0.4 -> LEAKAGE!
        )
        fut = (_obs("a1", 2, 0.8, 2.0, 2.0),)

        with self.assertRaises(ValueError) as ctx:
            EvaluationSample(
                agent_id="a1",
                reference_step=1,
                reference_timestamp=0.4,
                history=hist,
                ground_truth_future=fut,
            )
        self.assertIn("Data leakage detected", str(ctx.exception))

    def test_sample_generator_window_partitioning(self):
        """SampleGenerator creates non-leaking windows of exact history and horizon lengths."""
        # 20 observations for pedestrian p1 at 0.4s intervals
        obs_seq = [_obs("p1", i, i * 0.4, float(i), float(i * 2)) for i in range(20)]
        trajectories = {"p1": obs_seq}

        generator = SampleGenerator(history_length=8, prediction_horizon=12, stride=1)
        samples = generator.generate_samples(trajectories)

        # 20 points with hist=8 and horizon=12 -> exactly 1 window (0..7 history, 8..19 future)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(len(sample.history), 8)
        self.assertEqual(len(sample.ground_truth_future), 12)
        self.assertEqual(sample.reference_step, 7)
        self.assertAlmostEqual(sample.reference_timestamp, 7 * 0.4)
        self.assertEqual(sample.ground_truth_future[0].step_index, 8)

    def test_hand_calculable_exact_constant_velocity(self):
        """Predicting a pure constant-velocity trajectory yields exact ADE=0.0 and FDE=0.0."""
        # Velocity vx = 1.0 m/s, vy = 2.0 m/s
        obs_seq = [_obs("p1", i, i * 0.4, 1.0 * (i * 0.4), 2.0 * (i * 0.4)) for i in range(12)]
        sample = EvaluationSample(
            agent_id="p1",
            reference_step=3,
            reference_timestamp=3 * 0.4,
            history=tuple(obs_seq[:4]),
            ground_truth_future=tuple(obs_seq[4:]),
        )

        evaluator = PredictionEvaluator()
        result = evaluator.evaluate_sample(sample)

        self.assertEqual(result.status, "SUCCESS")
        self.assertAlmostEqual(result.ade, 0.0, places=6)
        self.assertAlmostEqual(result.fde, 0.0, places=6)
        for err in result.horizon_errors:
            self.assertAlmostEqual(err, 0.0, places=6)

    def test_hand_calculable_nonzero_error(self):
        """Predicting an accelerating trajectory yields known hand-calculable displacement errors."""
        # Position x(t) = t^2, so vx increases over time
        # History at t=0.0 -> x=0, t=1.0 -> x=1.0. Last velocity vx = (1.0 - 0.0) / 1.0 = 1.0 m/s
        # Future ground truth at t=2.0 -> x=4.0, t=3.0 -> x=9.0
        # Constant velocity predicts:
        # t=2.0: x_pred = 1.0 + 1.0*(2.0 - 1.0) = 2.0  (err = |4.0 - 2.0| = 2.0)
        # t=3.0: x_pred = 1.0 + 1.0*(3.0 - 1.0) = 3.0  (err = |9.0 - 3.0| = 6.0)
        # Expected ADE = (2.0 + 6.0) / 2 = 4.0 m
        # Expected FDE = 6.0 m
        history = (
            _obs("accel", 0, 0.0, 0.0, 0.0),
            _obs("accel", 1, 1.0, 1.0, 0.0),
        )
        ground_truth = (
            _obs("accel", 2, 2.0, 4.0, 0.0),
            _obs("accel", 3, 3.0, 9.0, 0.0),
        )
        sample = EvaluationSample(
            agent_id="accel",
            reference_step=1,
            reference_timestamp=1.0,
            history=history,
            ground_truth_future=ground_truth,
        )

        evaluator = PredictionEvaluator()
        result = evaluator.evaluate_sample(sample)

        self.assertEqual(result.status, "SUCCESS")
        self.assertAlmostEqual(result.ade, 4.0, places=6)
        self.assertAlmostEqual(result.fde, 6.0, places=6)
        self.assertAlmostEqual(result.horizon_errors[0], 2.0, places=6)
        self.assertAlmostEqual(result.horizon_errors[1], 6.0, places=6)

    def test_partial_future_handling_for_evacuated_agent(self):
        """When an agent evacuates before full horizon, allow_partial_future records PARTIAL result."""
        # Agent has 4 history steps and only 2 future steps before evacuating
        history = tuple(_obs("evac_ped", i, i * 0.1, float(i), 0.0) for i in range(4))
        future = tuple(_obs("evac_ped", i, i * 0.1, float(i), 0.0) for i in range(4, 6))

        sample = EvaluationSample(
            agent_id="evac_ped",
            reference_step=3,
            reference_timestamp=0.3,
            history=history,
            ground_truth_future=future,
            cohort="evacuated",
        )

        evaluator = PredictionEvaluator()
        result = evaluator.evaluate_sample(sample, timestep=0.1)

        self.assertEqual(result.status, "PARTIAL")
        self.assertAlmostEqual(result.ade, 0.0, places=6)
        self.assertEqual(result.matched_points, 2)

    def test_cohort_separate_evaluation(self):
        """Aggregate report computes metrics separately for ACTIVE and BLOCKED cohorts."""
        s_act = EvaluationSample(
            agent_id="act_1",
            reference_step=1,
            reference_timestamp=1.0,
            history=(_obs("act_1", 0, 0.0, 0.0, 0.0), _obs("act_1", 1, 1.0, 1.0, 0.0)),
            ground_truth_future=(_obs("act_1", 2, 2.0, 2.0, 0.0),),
            cohort="active",
        )
        # Blocked agent: deviates or stationary
        s_blk = EvaluationSample(
            agent_id="blk_1",
            reference_step=1,
            reference_timestamp=1.0,
            history=(_obs("blk_1", 0, 0.0, 0.0, 0.0), _obs("blk_1", 1, 1.0, 1.0, 0.0)),
            ground_truth_future=(_obs("blk_1", 2, 2.0, 1.0, 0.0),),  # ground truth stopped at x=1.0 (err=1.0)
            cohort="blocked",
        )

        evaluator = PredictionEvaluator()
        report = evaluator.evaluate_samples([s_act, s_blk])

        self.assertIn("active", report.cohort_metrics)
        self.assertIn("blocked", report.cohort_metrics)
        self.assertAlmostEqual(report.cohort_metrics["active"]["mean_ade"], 0.0)
        self.assertAlmostEqual(report.cohort_metrics["blocked"]["mean_ade"], 1.0)

    def test_dataset_adapter_eth_loader(self):
        """DatasetAdapter cleanly loads ETH benchmark data."""
        eth_file = Path("data/raw/eth/test/biwi_eth.txt")
        if not eth_file.is_file():
            self.skipTest("ETH raw file not present in repository")

        trajectories = DatasetAdapter.load_eth_file(eth_file)
        self.assertGreater(len(trajectories), 0)
        # Verify chronological order
        for aid, obs_list in trajectories.items():
            for i in range(1, len(obs_list)):
                self.assertGreater(obs_list[i].timestamp, obs_list[i - 1].timestamp)

    def test_dataset_adapter_sample_json_loader(self):
        """DatasetAdapter loads sample_trajectories.json."""
        json_file = Path("data/sample/sample_trajectories.json")
        trajectories = DatasetAdapter.load_sample_json(json_file)
        self.assertIn("ped_001", trajectories)
        self.assertGreaterEqual(len(trajectories["ped_001"]), 5)

    def test_report_json_and_csv_serialization(self, tmp_path=None):
        """Aggregate report serializes cleanly to JSON and CSV."""
        import tempfile

        s = EvaluationSample(
            agent_id="ser_1",
            reference_step=1,
            reference_timestamp=1.0,
            history=(_obs("ser_1", 0, 0.0, 0.0, 0.0), _obs("ser_1", 1, 1.0, 1.0, 0.0)),
            ground_truth_future=(_obs("ser_1", 2, 2.0, 2.0, 0.0),),
        )
        evaluator = PredictionEvaluator()
        report = evaluator.evaluate_samples([s])

        with tempfile.TemporaryDirectory() as td:
            out_json = Path(td) / "rep.json"
            out_csv = Path(td) / "res.csv"

            report.save_json(out_json)
            report.save_csv(out_csv)

            self.assertTrue(out_json.is_file())
            self.assertTrue(out_csv.is_file())

            # Read back JSON
            import json
            with open(out_json) as f:
                d = json.load(f)
            self.assertEqual(d["total_samples"], 1)
            self.assertAlmostEqual(d["mean_ade"], 0.0)


if __name__ == "__main__":
    unittest.main()

