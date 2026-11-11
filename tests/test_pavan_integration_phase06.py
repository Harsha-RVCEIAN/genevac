"""Comprehensive test suite for Phase 06 Pavan engine compatibility, event ingestion, and run isolation."""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
import pytest

from genevac.prediction.adapter import (
    EventAdaptationReport,
    SimulationTrajectoryAdapter,
    SnapshotAdaptationReport,
    verify_engine_read_only,
)
from genevac.prediction.data_contract import TrajectoryValidationError


# Mock fixtures mimicking Pavan's authoritative SimulationEvent and SimulationObservationSnapshot
@dataclass(frozen=True)
class MockTimestampedTrajectoryPoint:
    step: int
    timestamp: float
    x: float
    y: float


@dataclass(frozen=True)
class MockPedestrianObservation:
    pedestrian_id: str
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    status: str
    target_exit_id: Optional[str]
    trajectory_history: Tuple[MockTimestampedTrajectoryPoint, ...] = ()


@dataclass(frozen=True)
class MockSimulationObservationSnapshot:
    snapshot_id: str
    step_count: int
    timestamp: float
    dt: float
    observation_interval: float
    total_count: int
    active_count: int
    evacuated_count: int
    blocked_count: int
    inactive_count: int
    pedestrians: Tuple[MockPedestrianObservation, ...]


@dataclass(frozen=True)
class MockSimulationEvent:
    event_id: str
    step: int
    timestamp: float
    event_type: str
    pedestrian_id: Optional[str] = None
    details: Dict[str, Any] = None


class MockPavanEngine:
    """Mock simulating Pavan's public SimulationEngine interface."""
    def __init__(self, step: int = 10, ts: float = 1.0, dt: float = 0.1):
        self.step_count = step
        self.timestamp = ts
        self.dt = dt
        self.is_paused = False
        self.events: List[MockSimulationEvent] = [
            MockSimulationEvent("evt_001", 5, 0.5, "exit_closed", details={"exit_id": "exit_1"}),
            MockSimulationEvent("evt_002", 5, 0.5, "blocked", pedestrian_id="agent_1"),
        ]
        self.latest_events: List[MockSimulationEvent] = [self.events[-1]]
        self.pedestrians = [
            MockPedestrianObservation("agent_1", 10.0, 5.0, 0.0, 0.0, 0.0, "blocked", "exit_1"),
            MockPedestrianObservation("agent_2", 20.0, 5.0, 1.2, 0.0, 1.2, "active", "exit_2"),
        ]
        self.latest_snapshot = MockSimulationObservationSnapshot(
            snapshot_id="snap_000010",
            step_count=10,
            timestamp=1.0,
            dt=0.1,
            observation_interval=0.4,
            total_count=2,
            active_count=1,
            evacuated_count=0,
            blocked_count=1,
            inactive_count=0,
            pedestrians=tuple(self.pedestrians),
        )

    def get_snapshot(self) -> MockSimulationObservationSnapshot:
        return self.latest_snapshot

    def get_events(self, event_type=None, pedestrian_id=None):
        res = self.events
        if event_type:
            res = [e for e in res if e.event_type == event_type]
        if pedestrian_id:
            res = [e for e in res if e.pedestrian_id == pedestrian_id]
        return tuple(res)


class TestPavanIntegrationPhase06:
    def test_pavan_snapshot_all_statuses_and_trajectory_history(self):
        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)

        hist = (
            MockTimestampedTrajectoryPoint(0, 0.0, 5.0, 5.0),
            MockTimestampedTrajectoryPoint(4, 0.4, 5.4, 5.0),
        )
        peds = (
            MockPedestrianObservation("a_active", 5.8, 5.0, 1.0, 0.0, 1.0, "active", "exit_1", trajectory_history=hist),
            MockPedestrianObservation("a_blocked", 10.0, 5.0, 0.0, 0.0, 0.0, "blocked", "exit_1"),
            MockPedestrianObservation("a_evac", 50.0, 10.0, 0.0, 0.0, 0.0, "evacuated", "exit_1"),
            MockPedestrianObservation("a_inact", 0.0, 0.0, 0.0, 0.0, 0.0, "inactive", None),
        )
        snap = MockSimulationObservationSnapshot(
            snapshot_id="snap_000008",
            step_count=8,
            timestamp=0.8,
            dt=0.1,
            observation_interval=0.4,
            total_count=4,
            active_count=1,
            evacuated_count=1,
            blocked_count=1,
            inactive_count=1,
            pedestrians=peds,
        )

        rep = adapter.adapt_snapshot(snap, run_id="run_pavan_01")
        assert rep.accepted_observations == 2  # active and blocked recorded into buffer
        assert rep.is_duplicate is False
        assert "a_active" in adapter.active_agents
        assert "a_blocked" in adapter.blocked_agents
        assert "a_evac" in adapter.evacuated_agents
        assert "a_inact" in adapter.inactive_agents
        assert len(adapter.trajectory_histories["a_active"]) == 2

    def test_pavan_event_ingestion_and_identity_preservation(self):
        adapter = SimulationTrajectoryAdapter()

        evt = MockSimulationEvent(
            event_id="evt_0001",
            step=12,
            timestamp=1.2,
            event_type="blocked",
            pedestrian_id="agent_x",
            details={"reason": "exit_congestion"},
        )

        rep = adapter.adapt_event(evt, run_id="run_01")
        assert rep.event_id == "evt_0001"
        assert rep.run_id == "run_01"
        assert rep.step_index == 12
        assert math.isclose(rep.timestamp, 1.2)
        assert rep.event_type == "blocked"
        assert rep.pedestrian_id == "agent_x"
        assert rep.status == "ACCEPTED"
        assert rep.is_duplicate is False
        assert rep.details == {"reason": "exit_congestion"}

        # Agent should be in blocked cohort
        assert "agent_x" in adapter.blocked_agents

    def test_duplicate_event_handling_in_same_run(self):
        adapter = SimulationTrajectoryAdapter()
        evt = MockSimulationEvent("evt_dup", 10, 1.0, "no_route", pedestrian_id="agent_1")

        rep1 = adapter.adapt_event(evt, run_id="run_A")
        assert rep1.status == "ACCEPTED"
        assert rep1.is_duplicate is False

        rep2 = adapter.adapt_event(evt, run_id="run_A")
        assert rep2.status == "DUPLICATE"
        assert rep2.is_duplicate is True
        assert "already ingested" in rep2.rejection_reason

    def test_run_isolation_with_identical_snapshot_and_event_ids(self):
        adapter = SimulationTrajectoryAdapter()

        snap = MockSimulationObservationSnapshot(
            snapshot_id="snap_shared",
            step_count=4,
            timestamp=0.4,
            dt=0.1,
            observation_interval=0.4,
            total_count=1,
            active_count=1,
            evacuated_count=0,
            blocked_count=0,
            inactive_count=0,
            pedestrians=(MockPedestrianObservation("agent_1", 5.0, 5.0, 1.0, 0.0, 1.0, "active", "exit_1"),),
        )
        evt = MockSimulationEvent("evt_shared", 4, 0.4, "blocked", pedestrian_id="agent_1")

        # Ingest for Run A
        s_rep_a = adapter.adapt_snapshot(snap, run_id="run_A")
        e_rep_a = adapter.adapt_event(evt, run_id="run_A")
        assert s_rep_a.accepted_observations == 1
        assert e_rep_a.status == "ACCEPTED"

        # Ingest identical snapshot_id and event_id for Run B
        s_rep_b = adapter.adapt_snapshot(snap, run_id="run_B")
        e_rep_b = adapter.adapt_event(evt, run_id="run_B")

        # Because run_id differs, Run B MUST NOT be rejected as a duplicate
        assert s_rep_b.is_duplicate is False
        assert s_rep_b.accepted_observations == 1
        assert e_rep_b.is_duplicate is False
        assert e_rep_b.status == "ACCEPTED"

    def test_snapshot_event_double_count_prevention(self):
        adapter = SimulationTrajectoryAdapter()

        # Step 4: Event marks agent_1 as blocked
        evt = MockSimulationEvent("evt_1", 4, 0.4, "blocked", pedestrian_id="agent_1")
        adapter.adapt_event(evt, run_id="run_1")
        assert "agent_1" in adapter.blocked_agents

        # Step 4: Snapshot also reports agent_1 as blocked
        snap = MockSimulationObservationSnapshot(
            snapshot_id="snap_4",
            step_count=4,
            timestamp=0.4,
            dt=0.1,
            observation_interval=0.4,
            total_count=1,
            active_count=0,
            evacuated_count=0,
            blocked_count=1,
            inactive_count=0,
            pedestrians=(MockPedestrianObservation("agent_1", 5.0, 5.0, 0.0, 0.0, 0.0, "blocked", "exit_1"),),
        )
        adapter.adapt_snapshot(snap, run_id="run_1")

        # agent_1 is in blocked cohort exactly once (set cardinality = 1)
        assert len(adapter.blocked_agents) == 1
        assert "agent_1" in adapter.blocked_agents
        assert "agent_1" not in adapter.active_agents

    def test_missing_pedestrian_becomes_not_observed_never_evacuated(self):
        adapter = SimulationTrajectoryAdapter(observation_interval=0.4)

        # Snapshot 1: agent_1 and agent_2 are active
        snap1 = MockSimulationObservationSnapshot(
            snapshot_id="snap_01",
            step_count=4,
            timestamp=0.4,
            dt=0.1,
            observation_interval=0.4,
            total_count=2,
            active_count=2,
            evacuated_count=0,
            blocked_count=0,
            inactive_count=0,
            pedestrians=(
                MockPedestrianObservation("agent_1", 5.0, 5.0, 1.0, 0.0, 1.0, "active", "exit_1"),
                MockPedestrianObservation("agent_2", 8.0, 5.0, 1.0, 0.0, 1.0, "active", "exit_1"),
            ),
        )
        adapter.adapt_snapshot(snap1, run_id="run_01")
        assert adapter.active_agents == {"agent_1", "agent_2"}

        # Snapshot 2: agent_2 disappears from snapshot
        snap2 = MockSimulationObservationSnapshot(
            snapshot_id="snap_02",
            step_count=8,
            timestamp=0.8,
            dt=0.1,
            observation_interval=0.4,
            total_count=1,
            active_count=1,
            evacuated_count=0,
            blocked_count=0,
            inactive_count=0,
            pedestrians=(
                MockPedestrianObservation("agent_1", 5.4, 5.0, 1.0, 0.0, 1.0, "active", "exit_1"),
            ),
        )
        adapter.adapt_snapshot(snap2, run_id="run_01")

        assert "agent_1" in adapter.active_agents
        assert "agent_2" not in adapter.active_agents
        # Must be in not_observed_agents, NOT evacuated_agents
        assert "agent_2" in adapter.not_observed_agents
        assert "agent_2" not in adapter.evacuated_agents

    def test_read_only_engine_verification_passes(self):
        engine = MockPavanEngine()
        adapter = SimulationTrajectoryAdapter()

        is_unmutated, violations = verify_engine_read_only(
            engine,
            lambda: adapter.read_only_engine_ingest(engine, run_id="run_test")
        )
        assert is_unmutated is True
        assert len(violations) == 0

    def test_read_only_engine_verification_catches_mutation(self):
        engine = MockPavanEngine()

        def mutating_step():
            engine.step_count += 1

        is_unmutated, violations = verify_engine_read_only(engine, mutating_step)
        assert is_unmutated is False
        assert any("step count changed" in v for v in violations)

        def mutating_timestamp():
            engine.timestamp += 0.5

        is_unmutated, violations = verify_engine_read_only(engine, mutating_timestamp)
        assert is_unmutated is False
        assert any("timestamp changed" in v for v in violations)

        def mutating_dt():
            engine.dt = 0.05

        is_unmutated, violations = verify_engine_read_only(engine, mutating_dt)
        assert is_unmutated is False
        assert any("dt changed" in v for v in violations)

        def mutating_pause():
            engine.is_paused = True

        is_unmutated, violations = verify_engine_read_only(engine, mutating_pause)
        assert is_unmutated is False
        assert any("pause status changed" in v for v in violations)

        def mutating_events():
            engine.events.append(MockSimulationEvent("evt_mutated", 10, 1.0, "blocked"))

        is_unmutated, violations = verify_engine_read_only(engine, mutating_events)
        assert is_unmutated is False
        assert any("event collection or content changed" in v for v in violations)

        def mutating_pedestrians():
            engine.pedestrians[0] = MockPedestrianObservation("agent_1", 99.0, 5.0, 0.0, 0.0, 0.0, "blocked", "exit_1")

        is_unmutated, violations = verify_engine_read_only(engine, mutating_pedestrians)
        assert is_unmutated is False
        assert any("Pedestrian states, positions, velocities, or histories changed" in v for v in violations)

    def test_repeated_blocked_events_deduplication_and_cardinality(self):
        adapter = SimulationTrajectoryAdapter()
        evt = MockSimulationEvent("evt_block_1", 4, 0.4, "blocked", pedestrian_id="agent_1")

        rep1 = adapter.adapt_event(evt, run_id="run_1")
        assert rep1.status == "ACCEPTED"
        assert rep1.is_duplicate is False
        assert "agent_1" in adapter.blocked_agents
        assert len(adapter.blocked_agents) == 1

        # Re-send identical event
        rep2 = adapter.adapt_event(evt, run_id="run_1")
        assert rep2.status == "DUPLICATE"
        assert rep2.is_duplicate is True
        assert len(adapter.blocked_agents) == 1
        assert "agent_1" in adapter.blocked_agents

    def test_blocked_agent_becoming_active_precedence(self):
        adapter = SimulationTrajectoryAdapter()

        # Step 4: Event marks agent_1 as blocked
        evt = MockSimulationEvent("evt_block_4", 4, 0.4, "blocked", pedestrian_id="agent_1")
        adapter.adapt_event(evt, run_id="run_1")
        assert "agent_1" in adapter.blocked_agents
        assert "agent_1" not in adapter.active_agents

        # Step 8: Subsequent snapshot reports agent_1 as active (rerouted or unblocked)
        snap = MockSimulationObservationSnapshot(
            snapshot_id="snap_8",
            step_count=8,
            timestamp=0.8,
            dt=0.1,
            observation_interval=0.4,
            total_count=1,
            active_count=1,
            evacuated_count=0,
            blocked_count=0,
            inactive_count=0,
            pedestrians=(MockPedestrianObservation("agent_1", 6.0, 5.0, 1.0, 0.0, 1.0, "active", "exit_1"),),
        )
        adapter.adapt_snapshot(snap, run_id="run_1")

        # Newer snapshot takes precedence over older event
        assert "agent_1" in adapter.active_agents
        assert "agent_1" not in adapter.blocked_agents

    def test_blocked_agent_becoming_inactive(self):
        adapter = SimulationTrajectoryAdapter()

        # Step 4: Event marks agent_1 as blocked
        evt = MockSimulationEvent("evt_block_4", 4, 0.4, "blocked", pedestrian_id="agent_1")
        adapter.adapt_event(evt, run_id="run_1")
        assert "agent_1" in adapter.blocked_agents

        # Step 8: Snapshot reports agent_1 as inactive
        snap = MockSimulationObservationSnapshot(
            snapshot_id="snap_8",
            step_count=8,
            timestamp=0.8,
            dt=0.1,
            observation_interval=0.4,
            total_count=1,
            active_count=0,
            evacuated_count=0,
            blocked_count=0,
            inactive_count=1,
            pedestrians=(MockPedestrianObservation("agent_1", 6.0, 5.0, 0.0, 0.0, 0.0, "inactive", None),),
        )
        adapter.adapt_snapshot(snap, run_id="run_1")

        assert "agent_1" in adapter.inactive_agents
        assert "agent_1" not in adapter.blocked_agents
        assert "agent_1" not in adapter.active_agents

    def test_out_of_order_event_does_not_override_newer_snapshot(self):
        adapter = SimulationTrajectoryAdapter()

        # Step 8: Snapshot reports agent_1 as active
        snap = MockSimulationObservationSnapshot(
            snapshot_id="snap_8",
            step_count=8,
            timestamp=0.8,
            dt=0.1,
            observation_interval=0.4,
            total_count=1,
            active_count=1,
            evacuated_count=0,
            blocked_count=0,
            inactive_count=0,
            pedestrians=(MockPedestrianObservation("agent_1", 6.0, 5.0, 1.0, 0.0, 1.0, "active", "exit_1"),),
        )
        adapter.adapt_snapshot(snap, run_id="run_1")
        assert "agent_1" in adapter.active_agents

        # Older out-of-order event at step 4 arrives reporting agent_1 as blocked
        old_evt = MockSimulationEvent("evt_block_old", 4, 0.4, "blocked", pedestrian_id="agent_1")
        rep = adapter.adapt_event(old_evt, run_id="run_1")
        assert rep.status == "ACCEPTED"

        # Out-of-order check prevents step 4 event from overriding step 8 active observation
        assert "agent_1" in adapter.active_agents
        assert "agent_1" not in adapter.blocked_agents

    def test_repeated_earlier_event_rejection(self):
        adapter = SimulationTrajectoryAdapter()

        # Event at step 2
        evt = MockSimulationEvent("evt_early", 2, 0.2, "no_route", pedestrian_id="agent_1")
        rep1 = adapter.adapt_event(evt, run_id="run_1")
        assert rep1.status == "ACCEPTED"

        # Ingest snapshot at step 4
        snap = MockSimulationObservationSnapshot(
            snapshot_id="snap_4",
            step_count=4,
            timestamp=0.4,
            dt=0.1,
            observation_interval=0.4,
            total_count=1,
            active_count=0,
            evacuated_count=0,
            blocked_count=1,
            inactive_count=0,
            pedestrians=(MockPedestrianObservation("agent_1", 5.0, 5.0, 0.0, 0.0, 0.0, "blocked", "exit_1"),),
        )
        adapter.adapt_snapshot(snap, run_id="run_1")

        # Ingest earlier event again
        rep2 = adapter.adapt_event(evt, run_id="run_1")
        assert rep2.status == "DUPLICATE"
        assert rep2.is_duplicate is True

    def test_malformed_event_rejection(self):
        adapter = SimulationTrajectoryAdapter()

        bad_evt1 = MockSimulationEvent("", 10, 1.0, "blocked")  # empty event_id
        rep1 = adapter.adapt_event(bad_evt1)
        assert rep1.status == "REJECTED"
        assert "event_id must be a non-empty string" in rep1.rejection_reason

        bad_evt2 = MockSimulationEvent("evt_2", -5, 1.0, "blocked")  # negative step
        rep2 = adapter.adapt_event(bad_evt2)
        assert rep2.status == "REJECTED"
        assert "step must be non-negative" in rep2.rejection_reason

        bad_evt3 = MockSimulationEvent("evt_3", 10, -1.0, "blocked")  # negative timestamp
        rep3 = adapter.adapt_event(bad_evt3)
        assert rep3.status == "REJECTED"
        assert "timestamp must be finite non-negative" in rep3.rejection_reason

