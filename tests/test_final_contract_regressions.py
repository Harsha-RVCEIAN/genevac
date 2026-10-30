"""Regression tests for the final Part 2/3 contract hardening."""

import unittest

from genevac.prediction import (
    TrajectoryCategory,
    TrajectoryHistoryBuffer,
    TrajectoryObservation,
    TrajectorySnapshot,
    TrajectoryValidationError,
    TrajectoryValidator,
)


class TestSnapshotDefensiveImmutability(unittest.TestCase):
    def test_direct_snapshot_copies_mapping_and_freezes_sequences(self):
        observation = TrajectoryObservation("p1", 0, 0.0, 1.0, 2.0)
        source = {"p1": [observation]}
        snapshot = TrajectorySnapshot(0, 0.0, 0.1, source)

        source["p2"] = [TrajectoryObservation("p2", 0, 0.0, 3.0, 4.0)]
        source["p1"].append(TrajectoryObservation("p1", 1, 0.1, 2.0, 3.0))

        self.assertEqual(set(snapshot.trajectories), {"p1"})
        self.assertIsInstance(snapshot.trajectories["p1"], tuple)
        self.assertEqual(len(snapshot.trajectories["p1"]), 1)
        with self.assertRaises(TypeError):
            snapshot.trajectories["p2"] = ()  # type: ignore[index]
        with self.assertRaises(AttributeError):
            snapshot.trajectories["p1"].append(observation)  # type: ignore[attr-defined]

    def test_buffer_snapshot_has_same_immutability_guarantee(self):
        buffer = TrajectoryHistoryBuffer()
        buffer.record(TrajectoryObservation("p1", 0, 0.0, 1.0, 2.0))
        snapshot = buffer.get_snapshot(0, 0.0, 0.1)
        self.assertIsInstance(snapshot.trajectories["p1"], tuple)
        with self.assertRaises(TypeError):
            snapshot.trajectories["p2"] = ()  # type: ignore[index]


class TestSnapshotTimestampBoundary(unittest.TestCase):
    def test_future_observation_is_rejected(self):
        snapshot = TrajectorySnapshot(
            1,
            0.1,
            0.1,
            {"p1": [TrajectoryObservation("p1", 1, 0.2, 1.0, 2.0)]},
        )
        with self.assertRaises(TrajectoryValidationError) as context:
            TrajectoryValidator.validate_snapshot(snapshot)
        self.assertEqual(context.exception.field, "observation_timestamp")

    def test_small_floating_point_timestamp_noise_is_tolerated(self):
        snapshot = TrajectorySnapshot(
            1,
            0.3,
            0.1,
            {"p1": [TrajectoryObservation("p1", 1, 0.3000000005, 1.0, 2.0)]},
        )
        # Documented 1e-9 second tolerance accommodates representation noise.
        TrajectoryValidator.validate_snapshot(snapshot, max_observation_age=0.0)


class TestRecordCoordinateValidation(unittest.TestCase):
    def setUp(self):
        self.buffer = TrajectoryHistoryBuffer()

    def test_valid_numeric_coordinates_are_recorded(self):
        self.buffer.record("p1", (1, 2.5), step_index=0, timestamp=0.0)
        observation = self.buffer.get_observations("p1")[0]
        self.assertEqual(observation.position, (1, 2.5))
        self.assertEqual(observation.category, TrajectoryCategory.SIMULATED)

    def test_rejects_invalid_coordinate_structures(self):
        invalid_positions = (
            "12",
            b"12",
            (1.0,),
            (1.0, 2.0, 3.0),
            (True, 2.0),
            (1.0, False),
            ("1.0", 2.0),
            (1.0, "2.0"),
            (float("nan"), 2.0),
            (1.0, float("inf")),
            (1.0, float("-inf")),
            None,
            12.0,
        )
        for position in invalid_positions:
            with self.subTest(position=position):
                with self.assertRaises(TrajectoryValidationError):
                    self.buffer.record(
                        f"p{len(self.buffer.get_agent_ids())}",
                        position,  # type: ignore[arg-type]
                        step_index=0,
                        timestamp=0.0,
                    )


if __name__ == "__main__":
    unittest.main()
