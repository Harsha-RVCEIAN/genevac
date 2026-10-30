"""Contract tests for trajectory observations, history buffer, snapshots, and prediction results."""

import dataclasses
import math
import unittest

from genevac.prediction import (
    PredictedPosition,
    TrajectoryCategory,
    TrajectoryHistoryBuffer,
    TrajectoryObservation,
    TrajectoryPrediction,
    TrajectoryValidationError,
    TrajectoryValidator,
)


class TestTrajectoryObservation(unittest.TestCase):
    """Tests for TrajectoryObservation schema and validation."""

    def test_valid_observation_creation(self):
        obs = TrajectoryObservation(
            agent_id="ped_01",
            step_index=5,
            timestamp=0.5,
            x=12.4,
            y=8.2,
            category=TrajectoryCategory.SIMULATED,
        )
        self.assertEqual(obs.agent_id, "ped_01")
        self.assertEqual(obs.step_index, 5)
        self.assertEqual(obs.timestamp, 0.5)
        self.assertEqual(obs.x, 12.4)
        self.assertEqual(obs.y, 8.2)
        self.assertEqual(obs.position, (12.4, 8.2))
        self.assertEqual(obs.category, TrajectoryCategory.SIMULATED)

    def test_reject_empty_or_whitespace_agent_id(self):
        with self.assertRaises(TrajectoryValidationError):
            TrajectoryObservation(agent_id="", step_index=0, timestamp=0.0, x=1.0, y=1.0)
        with self.assertRaises(TrajectoryValidationError):
            TrajectoryObservation(agent_id="   ", step_index=0, timestamp=0.0, x=1.0, y=1.0)

    def test_reject_invalid_agent_id_type(self):
        with self.assertRaises(TrajectoryValidationError):
            TrajectoryObservation(agent_id=123, step_index=0, timestamp=0.0, x=1.0, y=1.0)  # type: ignore

    def test_reject_invalid_step_index(self):
        # Negative step index
        with self.assertRaises(TrajectoryValidationError):
            TrajectoryObservation(agent_id="ped_01", step_index=-1, timestamp=0.0, x=1.0, y=1.0)
        # Boolean used as step index (bool is a subclass of int in Python)
        with self.assertRaises(TrajectoryValidationError):
            TrajectoryObservation(agent_id="ped_01", step_index=True, timestamp=0.0, x=1.0, y=1.0)  # type: ignore
        # Float used as step index
        with self.assertRaises(TrajectoryValidationError):
            TrajectoryObservation(agent_id="ped_01", step_index=1.5, timestamp=0.0, x=1.0, y=1.0)  # type: ignore

    def test_reject_invalid_timestamp(self):
        # Negative timestamp
        with self.assertRaises(TrajectoryValidationError):
            TrajectoryObservation(agent_id="ped_01", step_index=0, timestamp=-0.1, x=1.0, y=1.0)
        # NaN timestamp
        with self.assertRaises(TrajectoryValidationError):
            TrajectoryObservation(agent_id="ped_01", step_index=0, timestamp=float("nan"), x=1.0, y=1.0)
        # Infinite timestamp
        with self.assertRaises(TrajectoryValidationError):
            TrajectoryObservation(agent_id="ped_01", step_index=0, timestamp=float("inf"), x=1.0, y=1.0)
        # Boolean timestamp
        with self.assertRaises(TrajectoryValidationError):
            TrajectoryObservation(agent_id="ped_01", step_index=0, timestamp=False, x=1.0, y=1.0)  # type: ignore

    def test_reject_non_finite_coordinates(self):
        # NaN x or y
        with self.assertRaises(TrajectoryValidationError):
            TrajectoryObservation(agent_id="ped_01", step_index=0, timestamp=0.0, x=float("nan"), y=1.0)
        with self.assertRaises(TrajectoryValidationError):
            TrajectoryObservation(agent_id="ped_01", step_index=0, timestamp=0.0, x=1.0, y=float("nan"))
        # Inf x or y
        with self.assertRaises(TrajectoryValidationError):
            TrajectoryObservation(agent_id="ped_01", step_index=0, timestamp=0.0, x=float("inf"), y=1.0)
        with self.assertRaises(TrajectoryValidationError):
            TrajectoryObservation(agent_id="ped_01", step_index=0, timestamp=0.0, x=1.0, y=float("-inf"))

    def test_observation_immutability(self):
        obs = TrajectoryObservation(agent_id="ped_01", step_index=0, timestamp=0.0, x=1.0, y=2.0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            obs.x = 5.0  # type: ignore
        with self.assertRaises(dataclasses.FrozenInstanceError):
            obs.timestamp = 1.0  # type: ignore


class TestTrajectoryHistoryBuffer(unittest.TestCase):
    """Tests for TrajectoryHistoryBuffer caching and rolling window."""

    def test_multiple_agents_and_chronological_retrieval(self):
        buf = TrajectoryHistoryBuffer(window_size=10)
        # Agent 1
        buf.record(TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0))
        buf.record(TrajectoryObservation("p1", 1, 0.1, 1.2, 1.1))
        # Agent 2
        buf.record(TrajectoryObservation("p2", 0, 0.0, 5.0, 5.0))
        buf.record(TrajectoryObservation("p2", 1, 0.1, 5.1, 5.2))

        p1_obs = buf.get_observations("p1")
        p2_obs = buf.get_observations("p2")

        self.assertEqual(len(p1_obs), 2)
        self.assertEqual(len(p2_obs), 2)
        self.assertEqual(p1_obs[0].step_index, 0)
        self.assertEqual(p1_obs[1].step_index, 1)
        self.assertEqual(p1_obs[1].timestamp, 0.1)

        self.assertEqual(buf.get_trajectory("p1"), [(1.0, 1.0), (1.2, 1.1)])
        self.assertEqual(buf.get_trajectory("p2"), [(5.0, 5.0), (5.1, 5.2)])

    def test_configurable_rolling_window(self):
        buf = TrajectoryHistoryBuffer(window_size=3)
        for i in range(5):
            buf.record(TrajectoryObservation("p1", i, round(i * 0.1, 2), float(i), float(i * 2)))

        obs = buf.get_observations("p1")
        self.assertEqual(len(obs), 3)
        self.assertEqual(obs[0].step_index, 2)
        self.assertEqual(obs[1].step_index, 3)
        self.assertEqual(obs[2].step_index, 4)
        self.assertEqual(buf.get_trajectory("p1"), [(2.0, 4.0), (3.0, 6.0), (4.0, 8.0)])

    def test_reject_duplicate_step_observation(self):
        buf = TrajectoryHistoryBuffer(window_size=5)
        buf.record(TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0))
        with self.assertRaises(TrajectoryValidationError) as ctx:
            buf.record(TrajectoryObservation("p1", 0, 0.0, 1.5, 1.5))
        self.assertIn("Duplicate observation", str(ctx.exception))

    def test_reject_out_of_order_step_observation(self):
        buf = TrajectoryHistoryBuffer(window_size=5)
        buf.record(TrajectoryObservation("p1", 2, 0.2, 1.0, 1.0))
        with self.assertRaises(TrajectoryValidationError) as ctx:
            buf.record(TrajectoryObservation("p1", 1, 0.3, 1.5, 1.5))
        self.assertIn("Out-of-order", str(ctx.exception))

    def test_reject_inconsistent_timestamp_ordering(self):
        buf = TrajectoryHistoryBuffer(window_size=5)
        buf.record(TrajectoryObservation("p1", 0, 0.2, 1.0, 1.0))
        with self.assertRaises(TrajectoryValidationError) as ctx:
            # Step increases, but timestamp is equal or decreasing
            buf.record(TrajectoryObservation("p1", 1, 0.2, 1.5, 1.5))
        self.assertIn("Inconsistent timestamp", str(ctx.exception))

    def test_insufficient_history_detection(self):
        buf = TrajectoryHistoryBuffer(window_size=5, min_history_required=3)
        buf.record(TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0))
        self.assertFalse(buf.has_sufficient_history("p1"))
        self.assertFalse(buf.has_sufficient_history("unknown_agent"))

        buf.record(TrajectoryObservation("p1", 1, 0.1, 1.1, 1.0))
        buf.record(TrajectoryObservation("p1", 2, 0.2, 1.2, 1.0))
        self.assertTrue(buf.has_sufficient_history("p1"))
        self.assertTrue(buf.has_sufficient_history("p1", min_required=2))
        self.assertFalse(buf.has_sufficient_history("p1", min_required=4))

    def test_preserves_supplied_timestamps_without_inferring(self):
        buf = TrajectoryHistoryBuffer(window_size=5)
        # Variable physical timestamps (e.g. from external sensor or non-trivial step)
        buf.record(TrajectoryObservation("p1", 10, 1.05, 1.0, 1.0))
        buf.record(TrajectoryObservation("p1", 15, 1.55, 2.0, 2.0))
        obs = buf.get_observations("p1")
        self.assertEqual(obs[0].timestamp, 1.05)
        self.assertEqual(obs[1].timestamp, 1.55)

    def test_buffer_accepts_skipped_step_observations(self):
        """Buffer must accept observations with non-consecutive step indices (missed simulation steps)."""
        buf = TrajectoryHistoryBuffer(window_size=5)
        buf.record(TrajectoryObservation("p1", 10, 1.0, 1.0, 1.0))
        buf.record(TrajectoryObservation("p1", 11, 1.1, 1.1, 1.1))
        buf.record(TrajectoryObservation("p1", 13, 1.3, 1.3, 1.3))  # step 12 skipped
        obs = buf.get_observations("p1")
        self.assertEqual(len(obs), 3)
        self.assertEqual(obs[2].step_index, 13)
        self.assertEqual(obs[2].timestamp, 1.3)

    def test_reject_predicted_category_observation_object(self):
        """Buffer must reject TrajectoryObservation with PREDICTED category."""
        buf = TrajectoryHistoryBuffer(window_size=5)
        obs = TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0, category=TrajectoryCategory.PREDICTED)
        with self.assertRaises(TrajectoryValidationError) as ctx:
            buf.record(obs)
        self.assertIn("Predicted observations must not be recorded", str(ctx.exception))
        # Buffer must remain unmodified
        self.assertEqual(len(buf.get_agent_ids()), 0)

    def test_reject_predicted_category_field_recording(self):
        """Buffer must reject field-based recording with PREDICTED category."""
        buf = TrajectoryHistoryBuffer(window_size=5)
        with self.assertRaises(TrajectoryValidationError) as ctx:
            buf.record("p1", (1.0, 1.0), step_index=0, timestamp=0.0, category=TrajectoryCategory.PREDICTED)
        self.assertIn("Predicted observations must not be recorded", str(ctx.exception))
        self.assertEqual(len(buf.get_agent_ids()), 0)


class TestTrajectorySnapshot(unittest.TestCase):
    """Tests for TrajectorySnapshot immutability and state representation."""

    def test_valid_snapshot_creation_and_metadata(self):
        buf = TrajectoryHistoryBuffer(window_size=5)
        buf.record(TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0))
        buf.record(TrajectoryObservation("p1", 1, 0.1, 1.2, 1.2))
        buf.record(TrajectoryObservation("p2", 1, 0.1, 4.0, 5.0))

        snapshot = buf.get_snapshot(current_step=1, current_timestamp=0.1, timestep=0.1)
        self.assertEqual(snapshot.current_step, 1)
        self.assertEqual(snapshot.current_timestamp, 0.1)
        self.assertEqual(snapshot.timestep, 0.1)
        self.assertEqual(set(snapshot.trajectories.keys()), {"p1", "p2"})
        self.assertEqual(len(snapshot.trajectories["p1"]), 2)
        self.assertEqual(len(snapshot.trajectories["p2"]), 1)

    def test_snapshot_immutability_and_cache_isolation(self):
        buf = TrajectoryHistoryBuffer(window_size=5)
        buf.record(TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0))
        snapshot = buf.get_snapshot(current_step=0, current_timestamp=0.0, timestep=0.1)

        # Modifying buffer afterwards does not affect the existing snapshot
        buf.record(TrajectoryObservation("p1", 1, 0.1, 2.0, 2.0))
        self.assertEqual(len(snapshot.trajectories["p1"]), 1)
        self.assertEqual(len(buf.get_observations("p1")), 2)

        # Modifying snapshot fields is prevented
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot.current_step = 2  # type: ignore

    def test_snapshot_with_target_agents_filter(self):
        buf = TrajectoryHistoryBuffer(window_size=5)
        buf.record(TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0))
        buf.record(TrajectoryObservation("p2", 0, 0.0, 2.0, 2.0))
        buf.record(TrajectoryObservation("p3", 0, 0.0, 3.0, 3.0))

        snapshot = buf.get_snapshot(current_step=0, current_timestamp=0.0, agent_ids=["p1", "p3"])
        self.assertEqual(set(snapshot.trajectories.keys()), {"p1", "p3"})
        self.assertNotIn("p2", snapshot.trajectories)

    def test_snapshot_mapping_is_immutable(self):
        """Snapshot trajectories mapping must not allow item assignment."""
        buf = TrajectoryHistoryBuffer(window_size=5)
        buf.record(TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0))
        snapshot = buf.get_snapshot(current_step=0, current_timestamp=0.0, timestep=0.1)
        with self.assertRaises(TypeError):
            snapshot.trajectories["p2"] = ()  # type: ignore

    def test_snapshot_tuple_observations_are_immutable(self):
        """Snapshot observation tuples must reject append/mutation."""
        buf = TrajectoryHistoryBuffer(window_size=5)
        buf.record(TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0))
        snapshot = buf.get_snapshot(current_step=0, current_timestamp=0.0, timestep=0.1)
        traj = snapshot.trajectories["p1"]
        self.assertIsInstance(traj, tuple)
        with self.assertRaises(AttributeError):
            traj.append(None)  # type: ignore

    def test_snapshot_isolates_from_buffer_mutation(self):
        """Mutating history buffer after snapshot creation must not change existing snapshot."""
        buf = TrajectoryHistoryBuffer(window_size=5)
        buf.record(TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0))
        snapshot = buf.get_snapshot(current_step=0, current_timestamp=0.0, timestep=0.1)
        # Add more observations after snapshot
        buf.record(TrajectoryObservation("p1", 1, 0.1, 2.0, 2.0))
        buf.record(TrajectoryObservation("p1", 2, 0.2, 3.0, 3.0))
        # Snapshot must still show only 1 observation
        self.assertEqual(len(snapshot.trajectories["p1"]), 1)
        self.assertEqual(snapshot.trajectories["p1"][0].x, 1.0)


class TestTrajectoryPredictionContract(unittest.TestCase):
    """Tests for TrajectoryPrediction result contracts."""

    def test_valid_prediction_result(self):
        predictions = (
            PredictedPosition(timestamp=1.1, x=10.5, y=5.2),
            PredictedPosition(timestamp=1.2, x=11.0, y=5.4),
            PredictedPosition(timestamp=1.3, x=11.5, y=5.6),
        )
        pred = TrajectoryPrediction(
            agent_id="ped_01",
            source_step_index=10,
            source_timestamp=1.0,
            predictions=predictions,
        )
        self.assertEqual(pred.agent_id, "ped_01")
        self.assertEqual(pred.source_step_index, 10)
        self.assertEqual(pred.source_timestamp, 1.0)
        self.assertEqual(len(pred.predictions), 3)
        self.assertEqual(pred.predictions[0].position, (10.5, 5.2))
        self.assertEqual(pred.category, TrajectoryCategory.PREDICTED)

    def test_reject_predicted_timestamp_not_strictly_later_than_source(self):
        # Predicted timestamp equals source timestamp
        with self.assertRaises(TrajectoryValidationError) as ctx:
            TrajectoryPrediction(
                agent_id="p1",
                source_step_index=5,
                source_timestamp=0.5,
                predictions=(PredictedPosition(timestamp=0.5, x=1.0, y=1.0),),
            )
        self.assertIn("strictly later than previous timestamp", str(ctx.exception))

        # Predicted timestamp before source timestamp
        with self.assertRaises(TrajectoryValidationError) as ctx:
            TrajectoryPrediction(
                agent_id="p1",
                source_step_index=5,
                source_timestamp=0.5,
                predictions=(PredictedPosition(timestamp=0.4, x=1.0, y=1.0),),
            )
        self.assertIn("strictly later than previous timestamp", str(ctx.exception))

    def test_reject_out_of_order_predicted_timestamps(self):
        with self.assertRaises(TrajectoryValidationError) as ctx:
            TrajectoryPrediction(
                agent_id="p1",
                source_step_index=5,
                source_timestamp=0.5,
                predictions=(
                    PredictedPosition(timestamp=0.7, x=1.0, y=1.0),
                    PredictedPosition(timestamp=0.6, x=2.0, y=2.0),
                ),
            )
        self.assertIn("strictly later than previous timestamp", str(ctx.exception))

    def test_reject_non_finite_predicted_position(self):
        with self.assertRaises(TrajectoryValidationError):
            PredictedPosition(timestamp=0.6, x=float("nan"), y=1.0)
        with self.assertRaises(TrajectoryValidationError):
            PredictedPosition(timestamp=float("inf"), x=1.0, y=1.0)

    def test_prediction_immutability(self):
        pred_pos = PredictedPosition(timestamp=1.1, x=10.0, y=5.0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            pred_pos.x = 20.0  # type: ignore

        pred = TrajectoryPrediction(
            agent_id="p1",
            source_step_index=1,
            source_timestamp=0.1,
            predictions=(pred_pos,),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            pred.agent_id = "p2"  # type: ignore


class TestTrajectoryValidator(unittest.TestCase):
    """Tests for TrajectoryValidator sequence and timestep verification."""

    def test_validate_observation_sequence_success(self):
        seq = [
            TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0),
            TrajectoryObservation("p1", 1, 0.1, 1.1, 1.1),
            TrajectoryObservation("p1", 2, 0.2, 1.2, 1.2),
        ]
        # Should pass without error
        TrajectoryValidator.validate_observation_sequence(seq, expected_agent_id="p1", min_history=2, expected_timestep=0.1)

    def test_validate_observation_sequence_insufficient_history(self):
        seq = [TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0)]
        with self.assertRaises(TrajectoryValidationError) as ctx:
            TrajectoryValidator.validate_observation_sequence(seq, min_history=3)
        self.assertIn("Insufficient history", str(ctx.exception))

    def test_validate_observation_sequence_timestep_discrepancy(self):
        seq = [
            TrajectoryObservation("p1", 0, 0.0, 1.0, 1.0),
            TrajectoryObservation("p1", 1, 0.25, 1.1, 1.1),  # Expected 0.1s, got 0.25s
        ]
        with self.assertRaises(TrajectoryValidationError) as ctx:
            TrajectoryValidator.validate_observation_sequence(seq, expected_timestep=0.1)
        self.assertIn("Timestamp step discrepancy", str(ctx.exception))

    def test_validate_observation_sequence_allows_skipped_steps(self):
        """Steps 10, 11, 13 (step 12 missed) should pass with expected_timestep validation."""
        seq = [
            TrajectoryObservation("p1", 10, 1.0, 1.0, 1.0),
            TrajectoryObservation("p1", 11, 1.1, 1.1, 1.1),
            TrajectoryObservation("p1", 13, 1.3, 1.3, 1.3),
        ]
        TrajectoryValidator.validate_observation_sequence(seq, expected_agent_id="p1", expected_timestep=0.1)


class TestSampleDataset(unittest.TestCase):
    """Tests loading and verifying sample_trajectories.json against the data contract."""

    def test_load_sample_trajectories_file(self):
        import json
        from pathlib import Path

        sample_file = Path("data/sample/sample_trajectories.json")
        self.assertTrue(sample_file.exists(), "Sample trajectory file data/sample/sample_trajectories.json must exist")

        with open(sample_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("observations", data)
        self.assertGreater(len(data["observations"]), 0)

        buf = TrajectoryHistoryBuffer(window_size=20)
        for row in data["observations"]:
            obs = TrajectoryObservation(
                agent_id=row["agent_id"],
                step_index=row["step_index"],
                timestamp=row["timestamp"],
                x=row["x"],
                y=row["y"],
            )
            buf.record(obs)

        # Check expected agents are loaded
        agent_ids = buf.get_agent_ids()
        self.assertIn("ped_001", agent_ids)
        self.assertIn("ped_002", agent_ids)
        self.assertIn("ped_003", agent_ids)

        # ped_001 and ped_002 have 11 observations (steps 0 to 10)
        self.assertEqual(len(buf.get_observations("ped_001")), 11)
        self.assertEqual(len(buf.get_observations("ped_002")), 11)

        # ped_003 has shorter history (4 observations, steps 7 to 10)
        self.assertEqual(len(buf.get_observations("ped_003")), 4)
        self.assertTrue(buf.has_sufficient_history("ped_003", min_required=3))
        self.assertFalse(buf.has_sufficient_history("ped_003", min_required=5))

        # Snapshot at step 10
        snapshot = buf.get_snapshot(current_step=10, current_timestamp=1.0, timestep=0.1)
        self.assertEqual(len(snapshot.trajectories), 3)
        TrajectoryValidator.validate_snapshot(snapshot)


class TestPredictionOutputContract(unittest.TestCase):
    """Verify TrajectoryPrediction always contains all required fields with valid values."""

    def _make_prediction(self, **kwargs):
        defaults = {
            "agent_id": "p1",
            "source_step_index": 5,
            "source_timestamp": 0.5,
            "predictions": (PredictedPosition(timestamp=0.6, x=1.0, y=1.0),),
        }
        defaults.update(kwargs)
        return TrajectoryPrediction(**defaults)

    def test_prediction_has_all_required_fields(self):
        pred = self._make_prediction()
        self.assertIsInstance(pred.agent_id, str)
        self.assertIsInstance(pred.source_step_index, int)
        self.assertIsInstance(pred.source_timestamp, float)
        self.assertIsInstance(pred.predictions, tuple)
        self.assertGreater(len(pred.predictions), 0)
        self.assertEqual(pred.category, TrajectoryCategory.PREDICTED)

    def test_all_predicted_timestamps_strictly_after_source(self):
        pred = self._make_prediction(
            source_timestamp=0.5,
            predictions=(
                PredictedPosition(timestamp=0.6, x=0.0, y=0.0),
                PredictedPosition(timestamp=0.7, x=1.0, y=0.0),
            ),
        )
        for pos in pred.predictions:
            self.assertGreater(pos.timestamp, pred.source_timestamp)

    def test_predicted_timestamps_strictly_increasing(self):
        pred = self._make_prediction(
            predictions=(
                PredictedPosition(timestamp=0.6, x=0.0, y=0.0),
                PredictedPosition(timestamp=0.7, x=1.0, y=0.0),
                PredictedPosition(timestamp=0.8, x=2.0, y=0.0),
            ),
        )
        ts = [p.timestamp for p in pred.predictions]
        for i in range(1, len(ts)):
            self.assertGreater(ts[i], ts[i - 1])

    def test_predicted_coordinates_finite(self):
        pred = self._make_prediction(
            predictions=(PredictedPosition(timestamp=0.6, x=1234.5, y=-999.9),),
        )
        self.assertTrue(math.isfinite(pred.predictions[0].x))
        self.assertTrue(math.isfinite(pred.predictions[0].y))

    def test_prediction_category_always_predicted(self):
        pred = self._make_prediction()
        self.assertEqual(pred.category, TrajectoryCategory.PREDICTED)

    def test_agent_identity_preserved(self):
        pred = self._make_prediction(agent_id="ped_42")
        self.assertEqual(pred.agent_id, "ped_42")

    def test_prediction_count_matches_tuple_length(self):
        preds = tuple(PredictedPosition(timestamp=0.5 + 0.1 * (k + 1), x=float(k), y=0.0) for k in range(5))
        pred = self._make_prediction(predictions=preds)
        self.assertEqual(len(pred.predictions), 5)


if __name__ == "__main__":
    unittest.main()
