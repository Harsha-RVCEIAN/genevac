"""Regression tests for Part 2 & Part 3 bug fixes.

Covers:
  1. Direct snapshot construction from a mutable dict: immutability guarantee.
  2. Mutation of the original dict after snapshot creation.
  3. Future-dated observations (observation.timestamp > snapshot.current_timestamp).
  4. Valid observations with tiny floating-point timestamp differences.
  5. Invalid coordinate types / structures passed to TrajectoryHistoryBuffer.record().
  6. Existing valid recording behaviour (sanity smoke tests).
"""

import math
import unittest
from types import MappingProxyType

from genevac.prediction import (
    TrajectoryCategory,
    TrajectoryHistoryBuffer,
    TrajectoryObservation,
    TrajectorySnapshot,
    TrajectoryValidationError,
    TrajectoryValidator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs(agent_id: str, step: int, ts: float, x: float = 0.0, y: float = 0.0) -> TrajectoryObservation:
    """Convenience constructor for a SIMULATED TrajectoryObservation."""
    return TrajectoryObservation(
        agent_id=agent_id,
        step_index=step,
        timestamp=ts,
        x=x,
        y=y,
        category=TrajectoryCategory.SIMULATED,
    )


# ---------------------------------------------------------------------------
# 1 & 2. Snapshot Immutability — direct construction from a mutable dict
# ---------------------------------------------------------------------------

class TestSnapshotImmutabilityDirect(unittest.TestCase):
    """TrajectorySnapshot must be immutable whether created from a plain dict or MappingProxyType."""

    def test_direct_dict_trajectories_coerced_to_mapping_proxy(self):
        """Passing a plain dict produces a MappingProxyType inside the snapshot."""
        obs = _obs("a1", 0, 0.0)
        snap = TrajectorySnapshot(
            current_step=0,
            current_timestamp=0.0,
            timestep=0.1,
            trajectories={"a1": (obs,)},
        )
        self.assertIsInstance(snap.trajectories, MappingProxyType)

    def test_direct_list_sequence_coerced_to_tuple(self):
        """An inner list sequence is coerced to a tuple during construction."""
        obs = _obs("a1", 0, 0.0)
        snap = TrajectorySnapshot(
            current_step=0,
            current_timestamp=0.0,
            timestep=0.1,
            trajectories={"a1": [obs]},  # list, not tuple
        )
        self.assertIsInstance(snap.trajectories["a1"], tuple)
        self.assertEqual(snap.trajectories["a1"], (obs,))

    def test_mutating_original_dict_does_not_affect_snapshot(self):
        """Mutating the original input dict after construction must leave the snapshot unchanged."""
        obs1 = _obs("a1", 0, 0.0, 1.0, 2.0)
        obs2 = _obs("a1", 1, 0.1, 2.0, 3.0)
        original: dict = {"a1": (obs1,)}

        snap = TrajectorySnapshot(
            current_step=1,
            current_timestamp=0.1,
            timestep=0.1,
            trajectories=original,
        )

        # Now mutate the original
        original["a1"] = (obs1, obs2)
        original["a2"] = (obs2,)

        # Snapshot must still reflect the state at construction time
        self.assertNotIn("a2", snap.trajectories)
        self.assertEqual(len(snap.trajectories["a1"]), 1)
        self.assertEqual(snap.trajectories["a1"][0], obs1)

    def test_snapshot_mapping_is_read_only(self):
        """Attempting to set a key on trajectories must raise TypeError."""
        obs = _obs("a1", 0, 0.0)
        snap = TrajectorySnapshot(
            current_step=0,
            current_timestamp=0.0,
            timestep=0.1,
            trajectories={"a1": (obs,)},
        )
        with self.assertRaises(TypeError):
            snap.trajectories["a1"] = ()  # type: ignore[index]

    def test_buffer_and_direct_construction_same_immutability(self):
        """Buffer-based and direct construction must produce equivalent immutability."""
        buf = TrajectoryHistoryBuffer()
        buf.record("a1", (1.0, 2.0), step_index=0, timestamp=0.0)
        buf.record("a1", (2.0, 3.0), step_index=1, timestamp=0.1)

        snap_buf = buf.get_snapshot(current_step=1, current_timestamp=0.1, timestep=0.1)

        obs1 = _obs("a1", 0, 0.0, 1.0, 2.0)
        obs2 = _obs("a1", 1, 0.1, 2.0, 3.0)
        snap_direct = TrajectorySnapshot(
            current_step=1,
            current_timestamp=0.1,
            timestep=0.1,
            trajectories={"a1": (obs1, obs2)},
        )

        self.assertIsInstance(snap_buf.trajectories, MappingProxyType)
        self.assertIsInstance(snap_direct.trajectories, MappingProxyType)
        self.assertIsInstance(snap_buf.trajectories["a1"], tuple)
        self.assertIsInstance(snap_direct.trajectories["a1"], tuple)


# ---------------------------------------------------------------------------
# 3. Future-dated observations
# ---------------------------------------------------------------------------

class TestFutureDatedObservations(unittest.TestCase):
    """validate_snapshot must reject any observation whose timestamp exceeds
    snapshot.current_timestamp (up to a 1e-9 s floating-point tolerance)."""

    def test_future_observation_raises_validation_error(self):
        """An observation timestamped strictly after the snapshot is rejected."""
        # Use OBSERVED category to avoid the simulated-timestep-drift check which
        # would fire first on the large dt between step 0 (t=0.0) and step 1 (t=1.0).
        def _obs_observed(agent_id, step, ts, x=0.0, y=0.0):
            return TrajectoryObservation(
                agent_id=agent_id, step_index=step, timestamp=ts,
                x=x, y=y, category=TrajectoryCategory.OBSERVED,
            )

        obs_past = _obs_observed("a1", 0, 0.0)
        obs_future = _obs_observed("a1", 1, 1.0)  # future relative to snapshot t=0.5

        snap = TrajectorySnapshot(
            current_step=5,
            current_timestamp=0.5,  # snapshot is at 0.5 s
            timestep=0.1,
            trajectories={"a1": (obs_past, obs_future)},  # obs_future.timestamp=1.0 > 0.5
        )

        with self.assertRaises(TrajectoryValidationError) as ctx:
            TrajectoryValidator.validate_snapshot(snap)

        self.assertEqual(ctx.exception.field, "observation_timestamp")
        self.assertEqual(ctx.exception.agent_id, "a1")

    def test_observation_at_exactly_snapshot_timestamp_is_allowed(self):
        """An observation whose timestamp equals snapshot.current_timestamp is valid."""
        obs1 = _obs("a1", 0, 0.0)
        obs2 = _obs("a1", 5, 0.5)  # exactly at snapshot time

        snap = TrajectorySnapshot(
            current_step=5,
            current_timestamp=0.5,
            timestep=0.1,
            trajectories={"a1": (obs1, obs2)},
        )
        # Must not raise
        TrajectoryValidator.validate_snapshot(snap)

    def test_observation_just_within_fp_tolerance_is_allowed(self):
        """An observation within 1e-9 s of the snapshot timestamp is treated as current."""
        ts_snap = 0.3
        obs1 = _obs("a1", 0, 0.0)
        # timestamp is 5e-10 s ahead — within the 1e-9 tolerance
        obs2 = _obs("a1", 3, ts_snap + 5e-10)

        snap = TrajectorySnapshot(
            current_step=3,
            current_timestamp=ts_snap,
            timestep=0.1,
            trajectories={"a1": (obs1, obs2)},
        )
        # Should not raise
        TrajectoryValidator.validate_snapshot(snap)

    def test_future_observation_field_label(self):
        """field must be 'observation_timestamp', not 'observation_age'."""
        def _obs_observed(agent_id, step, ts, x=0.0, y=0.0):
            return TrajectoryObservation(
                agent_id=agent_id, step_index=step, timestamp=ts,
                x=x, y=y, category=TrajectoryCategory.OBSERVED,
            )

        obs_future = _obs_observed("a1", 10, 2.0)  # way ahead of snapshot t=0.0

        snap = TrajectorySnapshot(
            current_step=0,
            current_timestamp=0.0,
            timestep=0.1,
            trajectories={"a1": (obs_future,)},
        )

        with self.assertRaises(TrajectoryValidationError) as ctx:
            TrajectoryValidator.validate_snapshot(snap)

        self.assertEqual(ctx.exception.field, "observation_timestamp")


# ---------------------------------------------------------------------------
# 4. Valid observations with tiny floating-point timestamp differences
# ---------------------------------------------------------------------------

class TestTinyTimestampDifferences(unittest.TestCase):
    """Observations that are chronologically valid but differ by only tiny amounts
    should be accepted without triggering the future-observation or stale-observation checks."""

    def test_epsilon_apart_timestamps_accepted(self):
        """Two observations separated by 1e-7 s (well above 1e-9 tolerance) are both valid."""
        obs1 = _obs("a1", 0, 0.0)
        obs2 = _obs("a1", 1, 1e-7)

        snap = TrajectorySnapshot(
            current_step=1,
            current_timestamp=1e-7,
            timestep=1e-7,
            trajectories={"a1": (obs1, obs2)},
        )
        # Must not raise
        TrajectoryValidator.validate_snapshot(snap)

    def test_snapshot_timestamp_exactly_at_derived_value(self):
        """Snapshot timestamp derived as current_step * timestep passes validation."""
        ts = 5 * 0.1  # = 0.5 (floating-point)
        obs1 = _obs("a1", 0, 0.0)
        obs2 = _obs("a1", 5, ts)

        snap = TrajectorySnapshot(
            current_step=5,
            current_timestamp=ts,
            timestep=0.1,
            trajectories={"a1": (obs1, obs2)},
        )
        TrajectoryValidator.validate_snapshot(snap)

    def test_buffer_record_and_snapshot_with_near_equal_timestamps(self):
        """Buffer-recorded observations with tiny timestamp differences roundtrip correctly."""
        buf = TrajectoryHistoryBuffer()
        buf.record("ped", (0.0, 0.0), step_index=0, timestamp=0.0)
        buf.record("ped", (1.0, 0.0), step_index=1, timestamp=1e-7)
        snap = buf.get_snapshot(current_step=1, current_timestamp=1e-7, timestep=1e-7)
        self.assertEqual(len(snap.trajectories["ped"]), 2)


# ---------------------------------------------------------------------------
# 5. Invalid coordinate types passed to record()
# ---------------------------------------------------------------------------

class TestRecordCoordinateValidation(unittest.TestCase):
    """TrajectoryHistoryBuffer.record() must reject invalid coordinate types/values
    with TrajectoryValidationError rather than leaking TypeError or IndexError."""

    def setUp(self):
        self.buf = TrajectoryHistoryBuffer()

    def test_string_x_coordinate_raises(self):
        """A string x-coordinate is rejected."""
        with self.assertRaises(TrajectoryValidationError) as ctx:
            self.buf.record("p1", ("bad", 1.0), step_index=0, timestamp=0.0)
        self.assertIn("position", ctx.exception.field)

    def test_string_y_coordinate_raises(self):
        """A string y-coordinate is rejected."""
        with self.assertRaises(TrajectoryValidationError) as ctx:
            self.buf.record("p1", (1.0, "bad"), step_index=0, timestamp=0.0)
        self.assertIn("position", ctx.exception.field)

    def test_boolean_x_coordinate_raises(self):
        """A boolean x-coordinate (True/False) is rejected."""
        with self.assertRaises(TrajectoryValidationError) as ctx:
            self.buf.record("p1", (True, 1.0), step_index=0, timestamp=0.0)
        self.assertIn("position", ctx.exception.field)

    def test_nan_coordinate_raises(self):
        """NaN x-coordinate is rejected."""
        with self.assertRaises(TrajectoryValidationError) as ctx:
            self.buf.record("p1", (float("nan"), 1.0), step_index=0, timestamp=0.0)
        self.assertIn("position", ctx.exception.field)

    def test_inf_coordinate_raises(self):
        """Inf y-coordinate is rejected."""
        with self.assertRaises(TrajectoryValidationError) as ctx:
            self.buf.record("p1", (1.0, float("inf")), step_index=0, timestamp=0.0)
        self.assertIn("position", ctx.exception.field)

    def test_position_not_a_sequence_raises(self):
        """A non-sequence position raises TrajectoryValidationError, not TypeError."""
        with self.assertRaises(TrajectoryValidationError) as ctx:
            self.buf.record("p1", 42, step_index=0, timestamp=0.0)  # type: ignore[arg-type]
        self.assertIn("position", ctx.exception.field)

    def test_position_wrong_length_raises(self):
        """A position with ≠ 2 elements is rejected."""
        with self.assertRaises(TrajectoryValidationError) as ctx:
            self.buf.record("p1", (1.0, 2.0, 3.0), step_index=0, timestamp=0.0)  # type: ignore[arg-type]
        self.assertIn("position", ctx.exception.field)

    def test_none_coordinate_raises(self):
        """None as a coordinate value is rejected."""
        with self.assertRaises(TrajectoryValidationError) as ctx:
            self.buf.record("p1", (None, 1.0), step_index=0, timestamp=0.0)  # type: ignore[arg-type]
        self.assertIn("position", ctx.exception.field)


# ---------------------------------------------------------------------------
# 6. Existing valid recording behaviour (smoke tests)
# ---------------------------------------------------------------------------

class TestValidRecordingBehaviour(unittest.TestCase):
    """Ensure existing valid paths still work after the coordinate validation additions."""

    def test_record_by_fields_valid_integers(self):
        """Integer coordinate values are accepted and stored correctly."""
        buf = TrajectoryHistoryBuffer()
        buf.record("p1", (10, 20), step_index=0, timestamp=0.0)
        obs = buf.get_observations("p1")[0]
        self.assertAlmostEqual(obs.x, 10.0)
        self.assertAlmostEqual(obs.y, 20.0)

    def test_record_by_fields_valid_floats(self):
        """Float coordinate values are accepted."""
        buf = TrajectoryHistoryBuffer()
        buf.record("p1", (1.5, -3.14), step_index=0, timestamp=0.0)
        obs = buf.get_observations("p1")[0]
        self.assertAlmostEqual(obs.x, 1.5)
        self.assertAlmostEqual(obs.y, -3.14)

    def test_record_by_observation_object_still_works(self):
        """Passing a pre-constructed TrajectoryObservation bypasses field validation."""
        buf = TrajectoryHistoryBuffer()
        obs = _obs("p1", 0, 0.0, 5.0, 6.0)
        buf.record(obs)
        stored = buf.get_observations("p1")[0]
        self.assertEqual(stored, obs)

    def test_multiple_steps_record_and_retrieve(self):
        """Recording multiple valid observations produces correct ordered history."""
        buf = TrajectoryHistoryBuffer()
        for i in range(5):
            buf.record("p1", (float(i), float(i * 2)), step_index=i, timestamp=i * 0.1)
        obs_seq = buf.get_observations("p1")
        self.assertEqual(len(obs_seq), 5)
        self.assertAlmostEqual(obs_seq[-1].x, 4.0)
        self.assertAlmostEqual(obs_seq[-1].timestamp, 0.4)

    def test_snapshot_from_buffer_still_immutable(self):
        """Buffer-created snapshot is still a MappingProxyType with tuple values."""
        buf = TrajectoryHistoryBuffer()
        buf.record("p1", (0.0, 0.0), step_index=0, timestamp=0.0)
        buf.record("p1", (1.0, 1.0), step_index=1, timestamp=0.1)
        snap = buf.get_snapshot(current_step=1, current_timestamp=0.1, timestep=0.1)
        self.assertIsInstance(snap.trajectories, MappingProxyType)
        self.assertIsInstance(snap.trajectories["p1"], tuple)


if __name__ == "__main__":
    unittest.main()
