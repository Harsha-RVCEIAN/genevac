"""Simulation-to-prediction adapter for GenEvac-Sentinel.

Bridges Person A's simulation engine and Person B's prediction/metrics subsystem:
- Non-invasively monitors simulation progression without mutating simulation state.
- Ingests both live SimulationEngine instances and immutable SimulationObservationSnapshot objects.
- Ingests Pavan's discrete SimulationEvent objects (BLOCKED, NO_ROUTE, EXIT_CLOSED).
- Converts authoritative simulation state to validated TrajectoryObservation records.
- Enforces cadence filtering (e.g. 0.4s observation interval on a 0.1s simulation engine)
  robust against duplicate steps, irregular invocation, and skipped steps.
- Enforces run-scoped deduplication keys (run_id, snapshot_id), (run_id, step), and (run_id, event_id).
- Tracks pedestrian cohorts (ACTIVE, BLOCKED, EVACUATED, INCAPACITATED, INACTIVE, NOT_OBSERVED)
  strictly within the adapter without polluting core TrajectoryObservation contracts.
- Treats EVACUATED as a terminal state for the current run (preserves history, blocks recording).
- Detects simulation resets via step rollbacks, engine instance replacements, or explicit run_id changes.
- Prevents double-counting between snapshot cohorts and event streams.
- Provides read-only integrity verification to guarantee zero simulation mutation.
"""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

if TYPE_CHECKING:
    from genevac.prediction import TrajectoryHistoryBuffer

from genevac.prediction.data_contract import (
    TrajectoryCategory,
    TrajectoryObservation,
    TrajectorySnapshot,
    TrajectoryValidationError,
)


@dataclass(frozen=True)
class SnapshotAdaptationReport:
    """Auditable summary of a snapshot ingestion execution."""
    snapshot_id: str
    step_index: int
    timestamp: float
    total_candidates: int
    accepted_observations: int
    rejected_observations: int
    is_duplicate: bool
    rejection_reasons: Tuple[str, ...] = field(default_factory=tuple)
    run_id: str = "default_run"


@dataclass(frozen=True)
class EventAdaptationReport:
    """Auditable summary of an event ingestion execution."""
    event_id: str
    run_id: str
    step_index: int
    timestamp: float
    event_type: str
    pedestrian_id: Optional[str]
    is_duplicate: bool
    status: str  # "ACCEPTED", "DUPLICATE", "REJECTED"
    details: Dict[str, Any] = field(default_factory=dict)
    rejection_reason: str = ""


class SimulationTrajectoryAdapter:
    """Adapter bridging authoritative SimulationEngine state to prediction/metrics buffers.

    Responsibilities:
    1. Reset & Run Isolation: Tracks active run_id, engine instance identity, and step monotonicity.
       Clears run-scoped state when engine replacement, rollback, or run transition is detected.
    2. Run-Scoped Deduplication:
       - Snapshots: (run_id, snapshot_id) and (run_id, step_index).
       - Events: (run_id, event_id).
    3. Observation Cadence: Configurable interval (e.g. 0.4s / 4 steps). Idempotently handles
       duplicate steps and irregular invocations.
    4. Snapshot Ingestion: Exposes ``adapt_snapshot(snapshot, force, run_id)``.
    5. Event Ingestion: Exposes ``adapt_event(event, run_id)`` and ``ingest_events(events, run_id)``.
    6. Double-Count Prevention: Harmonizes status updates across snapshots and event streams.
    7. Read-Only Verification: Verifies engine state invariance during ingestion.
    """

    def __init__(
        self,
        buffer: Optional[TrajectoryHistoryBuffer] = None,
        observation_interval: float = 0.4,
        max_observation_age: Optional[float] = 0.45,
        window_size: int = 20,
        min_history_required: int = 2,
        default_run_id: str = "default_run",
    ) -> None:
        """Initialize the adapter."""
        if (
            type(observation_interval) not in (float, int)
            or isinstance(observation_interval, bool)
            or not math.isfinite(observation_interval)
            or observation_interval <= 0.0
        ):
            raise ValueError(f"observation_interval must be a finite positive number, got {observation_interval!r}")

        if max_observation_age is not None:
            if (
                type(max_observation_age) not in (float, int)
                or isinstance(max_observation_age, bool)
                or not math.isfinite(max_observation_age)
                or max_observation_age < 0.0
            ):
                raise ValueError(f"max_observation_age must be a finite non-negative number or None, got {max_observation_age!r}")

        if buffer is not None:
            self.buffer = buffer
        else:
            from genevac.prediction import TrajectoryHistoryBuffer

            self.buffer = TrajectoryHistoryBuffer(
                window_size=window_size,
                min_history_required=min_history_required,
                max_observation_age=None,
            )
        self.observation_interval = float(observation_interval)
        self.max_observation_age = float(max_observation_age) if max_observation_age is not None else None
        self._current_run_id: str = str(default_run_id).strip() or "default_run"

        # Run tracking & reset detection
        self._last_engine_id: Optional[int] = None
        self._last_observed_step: int = -1
        self._last_recorded_step: int = -1
        self._last_recorded_timestamp: float = -1.0

        # Run-scoped deduplication sets
        self._seen_snapshot_keys: Set[Tuple[str, str]] = set()  # (run_id, snap_id)
        self._seen_snapshot_steps: Set[Tuple[str, int]] = set() # (run_id, step)
        self._seen_event_keys: Set[Tuple[str, str]] = set()     # (run_id, event_id)
        self._ingested_events_by_run: Dict[str, List[EventAdaptationReport]] = {}

        # Cohort tracking (strictly inside adapter)
        self._active_agents: Set[str] = set()
        self._blocked_agents: Set[str] = set()
        self._evacuated_agents: Set[str] = set()
        self._incapacitated_agents: Set[str] = set()
        self._inactive_agents: Set[str] = set()
        self._not_observed_agents: Set[str] = set()

        # Event-derived blocked tracking to prevent double-counting: run_id -> (ped_id -> event_step)
        self._blocked_from_events: Dict[str, Dict[str, int]] = {}

        # Authoritative trajectory history storage from snapshots (ped_id -> tuple of points)
        self._pedestrian_trajectory_histories: Dict[str, Tuple[Any, ...]] = {}

        # Per-agent last recorded step/time to guard irregular or out-of-order calls
        self._agent_last_recorded_step: Dict[str, int] = {}
        self._agent_last_recorded_time: Dict[str, float] = {}

    @property
    def current_run_id(self) -> str:
        return self._current_run_id

    @property
    def seen_snapshot_ids(self) -> Set[str]:
        """Backward compatibility view of snapshot IDs seen in current run."""
        return {snap_id for (rid, snap_id) in self._seen_snapshot_keys if rid == self._current_run_id}

    def reset(self, run_id: Optional[str] = None) -> None:
        """Explicitly reset adapter state for the specified run (or current run)."""
        if run_id is not None and str(run_id).strip():
            self._current_run_id = str(run_id).strip()
            self._seen_snapshot_keys = {k for k in self._seen_snapshot_keys if k[0] != self._current_run_id}
            self._seen_snapshot_steps = {k for k in self._seen_snapshot_steps if k[0] != self._current_run_id}
            self._seen_event_keys = {k for k in self._seen_event_keys if k[0] != self._current_run_id}
            self._blocked_from_events.pop(self._current_run_id, None)
        else:
            self._seen_snapshot_keys.clear()
            self._seen_snapshot_steps.clear()
            self._seen_event_keys.clear()
            self._blocked_from_events.clear()

        self.buffer.clear()
        self._last_engine_id = None
        self._last_observed_step = -1
        self._last_recorded_step = -1
        self._last_recorded_timestamp = -1.0

        self._active_agents.clear()
        self._blocked_agents.clear()
        self._evacuated_agents.clear()
        self._incapacitated_agents.clear()
        self._inactive_agents.clear()
        self._not_observed_agents.clear()
        self._pedestrian_trajectory_histories.clear()
        self._agent_last_recorded_step.clear()
        self._agent_last_recorded_time.clear()

    def adapt_snapshot(
        self,
        snapshot: Any,
        force: bool = False,
        run_id: Optional[str] = None,
    ) -> SnapshotAdaptationReport:
        """Adapt a verified SimulationObservationSnapshot object into canonical observations.

        Args:
            snapshot: Immutable SimulationObservationSnapshot instance or duck-typed snapshot.
            force: If True, forces recording at current step regardless of observation cadence.
            run_id: Explicit execution run ID. If different from current_run_id, triggers isolated reset.

        Returns:
            SnapshotAdaptationReport detailing accepted, rejected, and duplicate counts.
        """
        active_run_id = str(run_id).strip() if (run_id is not None and str(run_id).strip()) else self._current_run_id
        if active_run_id != self._current_run_id:
            self.reset(run_id=active_run_id)

        # Read step count
        step = getattr(snapshot, "step_count", None)
        if step is None:
            step = getattr(snapshot, "current_step", None)
        if type(step) is not int or step < 0:
            raise TrajectoryValidationError(
                f"snapshot step must be a non-negative integer, got {step!r}",
                field="step_count",
            )

        # Read timestamp
        ts = getattr(snapshot, "timestamp", None)
        if ts is None:
            dt = getattr(snapshot, "dt", getattr(snapshot, "timestep", 0.1))
            ts = step * dt
        ts = float(ts)
        if not math.isfinite(ts) or ts < 0.0:
            raise TrajectoryValidationError(
                f"snapshot timestamp must be a finite non-negative number, got {ts!r}",
                field="timestamp",
            )

        # Read snapshot_id
        snap_id = str(getattr(snapshot, "snapshot_id", f"snap_{step:06d}"))

        # Step rollback reset check
        if step < self._last_observed_step:
            self.reset(run_id=active_run_id)
        self._last_observed_step = step

        # Run-scoped duplicate check by (run_id, snap_id) or (run_id, step)
        snap_key = (active_run_id, snap_id)
        step_key = (active_run_id, step)

        if snap_key in self._seen_snapshot_keys or (step_key in self._seen_snapshot_steps and not force):
            return SnapshotAdaptationReport(
                snapshot_id=snap_id,
                step_index=step,
                timestamp=ts,
                total_candidates=0,
                accepted_observations=0,
                rejected_observations=0,
                is_duplicate=True,
                rejection_reasons=("Duplicate snapshot step or snapshot_id already processed for run.",),
                run_id=active_run_id,
            )

        # Cadence check
        if not force and self._last_recorded_timestamp >= 0.0:
            time_since_last = ts - self._last_recorded_timestamp
            if time_since_last < (self.observation_interval - 1e-6):
                return SnapshotAdaptationReport(
                    snapshot_id=snap_id,
                    step_index=step,
                    timestamp=ts,
                    total_candidates=0,
                    accepted_observations=0,
                    rejected_observations=0,
                    is_duplicate=False,
                    rejection_reasons=("Observation cadence interval not met.",),
                    run_id=active_run_id,
                )

        peds = getattr(snapshot, "pedestrians", ())
        accepted = 0
        rejected = 0
        rejection_reasons: List[str] = []

        for ped in peds:
            ped_id = getattr(ped, "pedestrian_id", getattr(ped, "id", None))
            if not ped_id or not str(ped_id).strip():
                rejected += 1
                rejection_reasons.append("Empty or missing pedestrian ID.")
                continue
            ped_id = str(ped_id).strip()

            raw_status = getattr(ped, "status", None)
            if raw_status is None:
                rejected += 1
                rejection_reasons.append(f"Pedestrian '{ped_id}' has no status attribute.")
                continue
            status_val = raw_status.value if hasattr(raw_status, "value") else raw_status
            status_str = str(status_val).strip().lower()

            # Extract coordinates
            if hasattr(ped, "x") and hasattr(ped, "y"):
                coord_x = float(ped.x)
                coord_y = float(ped.y)
            elif hasattr(ped, "position"):
                pos = ped.position
                coord_x = float(pos[0]) if hasattr(pos, "__getitem__") else float(pos.x)
                coord_y = float(pos[1]) if hasattr(pos, "__getitem__") else float(pos.y)
            else:
                rejected += 1
                rejection_reasons.append(f"Pedestrian '{ped_id}' missing coordinates.")
                continue

            if not (math.isfinite(coord_x) and math.isfinite(coord_y)):
                rejected += 1
                rejection_reasons.append(f"Non-finite coordinates ({coord_x}, {coord_y}) for agent '{ped_id}'.")
                continue

            # Preserve authoritative trajectory history if present
            if hasattr(ped, "trajectory_history") and getattr(ped, "trajectory_history"):
                self._pedestrian_trajectory_histories[ped_id] = tuple(ped.trajectory_history)

            # Route status after verifying coordinates are valid
            event_blocked_map = self._blocked_from_events.get(active_run_id, {})
            evt_block_step = event_blocked_map.get(ped_id, -1)

            if status_str == "evacuated" or ped_id in self._evacuated_agents:
                self._evacuated_agents.add(ped_id)
                self._active_agents.discard(ped_id)
                self._blocked_agents.discard(ped_id)
                self._incapacitated_agents.discard(ped_id)
                event_blocked_map.pop(ped_id, None)
                continue

            if status_str == "incapacitated" or ped_id in self._incapacitated_agents:
                self._incapacitated_agents.add(ped_id)
                self._active_agents.discard(ped_id)
                self._blocked_agents.discard(ped_id)
                self._evacuated_agents.discard(ped_id)
                event_blocked_map.pop(ped_id, None)
                continue

            if status_str == "inactive":
                self._inactive_agents.add(ped_id)
                self._active_agents.discard(ped_id)
                self._blocked_agents.discard(ped_id)
                self._incapacitated_agents.discard(ped_id)
                event_blocked_map.pop(ped_id, None)
                continue

            # Harmonize with any events that marked agent blocked
            if status_str == "blocked":
                self._blocked_agents.add(ped_id)
                self._active_agents.discard(ped_id)
            elif status_str == "active":
                # If authoritative snapshot step is >= event step, newer snapshot takes precedence
                if step >= evt_block_step:
                    self._active_agents.add(ped_id)
                    self._blocked_agents.discard(ped_id)
                    event_blocked_map.pop(ped_id, None)
                else:
                    self._blocked_agents.add(ped_id)
                    self._active_agents.discard(ped_id)
            else:
                rejected += 1
                rejection_reasons.append(f"Unknown pedestrian status '{raw_status!r}' for agent '{ped_id}'.")
                continue

            # Per-agent duplicate check
            last_agent_step = self._agent_last_recorded_step.get(ped_id, -1)
            if step <= last_agent_step:
                continue

            self.buffer.record(
                observation_or_agent_id=ped_id,
                position=(coord_x, coord_y),
                step_index=step,
                timestamp=ts,
                category=TrajectoryCategory.SIMULATED,
            )
            self._agent_last_recorded_step[ped_id] = step
            self._agent_last_recorded_time[ped_id] = ts
            accepted += 1

        # Check for missing agents: previously tracked but absent in this snapshot
        current_candidate_ids = {
            str(getattr(p, "pedestrian_id", getattr(p, "id", ""))).strip()
            for p in peds
            if getattr(p, "pedestrian_id", getattr(p, "id", None))
        }
        all_previously_tracked = (self._active_agents | self._blocked_agents) - current_candidate_ids
        for missing_id in all_previously_tracked:
            self._not_observed_agents.add(missing_id)
            self._active_agents.discard(missing_id)
            self._blocked_agents.discard(missing_id)

        self._last_recorded_step = step
        self._last_recorded_timestamp = ts
        self._seen_snapshot_keys.add(snap_key)
        self._seen_snapshot_steps.add(step_key)

        return SnapshotAdaptationReport(
            snapshot_id=snap_id,
            step_index=step,
            timestamp=ts,
            total_candidates=len(peds),
            accepted_observations=accepted,
            rejected_observations=rejected,
            is_duplicate=False,
            rejection_reasons=tuple(rejection_reasons),
            run_id=active_run_id,
        )

    def adapt_event(
        self,
        event: Any,
        run_id: Optional[str] = None,
    ) -> EventAdaptationReport:
        """Adapt a verified SimulationEvent record deterministically with run-scoped deduplication.

        Args:
            event: Pavan SimulationEvent or compatible duck-typed object.
            run_id: Explicit execution run ID.

        Returns:
            EventAdaptationReport documenting event acceptance, duplicate status, or rejection.
        """
        active_run_id = str(run_id).strip() if (run_id is not None and str(run_id).strip()) else self._current_run_id
        if active_run_id != self._current_run_id:
            self.reset(run_id=active_run_id)

        # 1. Extract and validate event_id
        evt_id = getattr(event, "event_id", None)
        if not evt_id or not isinstance(evt_id, str):
            return EventAdaptationReport(
                event_id=str(evt_id or "invalid_id"),
                run_id=active_run_id,
                step_index=0,
                timestamp=0.0,
                event_type="unknown",
                pedestrian_id=None,
                is_duplicate=False,
                status="REJECTED",
                rejection_reason=f"event_id must be a non-empty string, got {evt_id!r}",
            )

        # 2. Check run-scoped event deduplication
        evt_key = (active_run_id, evt_id)
        if evt_key in self._seen_event_keys:
            step = int(getattr(event, "step", 0))
            ts = float(getattr(event, "timestamp", 0.0))
            etype = str(getattr(event, "event_type", "unknown"))
            pid = getattr(event, "pedestrian_id", None)
            return EventAdaptationReport(
                event_id=evt_id,
                run_id=active_run_id,
                step_index=step,
                timestamp=ts,
                event_type=etype,
                pedestrian_id=pid,
                is_duplicate=True,
                status="DUPLICATE",
                rejection_reason="Event already ingested for this run.",
            )

        # 3. Extract and validate fields
        step = getattr(event, "step", None)
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            return EventAdaptationReport(
                event_id=evt_id,
                run_id=active_run_id,
                step_index=-1,
                timestamp=0.0,
                event_type="unknown",
                pedestrian_id=None,
                is_duplicate=False,
                status="REJECTED",
                rejection_reason=f"event step must be non-negative int, got {step!r}",
            )

        ts = getattr(event, "timestamp", None)
        try:
            ts_f = float(ts)
            if not (math.isfinite(ts_f) and ts_f >= 0.0):
                raise ValueError()
        except (TypeError, ValueError):
            return EventAdaptationReport(
                event_id=evt_id,
                run_id=active_run_id,
                step_index=step,
                timestamp=-1.0,
                event_type="unknown",
                pedestrian_id=None,
                is_duplicate=False,
                status="REJECTED",
                rejection_reason=f"event timestamp must be finite non-negative number, got {ts!r}",
            )

        raw_type = getattr(event, "event_type", None)
        if not raw_type or not isinstance(raw_type, (str, bytes)):
            return EventAdaptationReport(
                event_id=evt_id,
                run_id=active_run_id,
                step_index=step,
                timestamp=ts_f,
                event_type="unknown",
                pedestrian_id=None,
                is_duplicate=False,
                status="REJECTED",
                rejection_reason=f"event_type must be a non-empty string, got {raw_type!r}",
            )
        event_type_str = str(raw_type).strip().lower()

        ped_id = getattr(event, "pedestrian_id", None)
        if ped_id is not None:
            ped_id = str(ped_id).strip()

        details = dict(getattr(event, "details", {}) or {})

        # 4. Harmonize event state with adapter cohorts without double counting
        if event_type_str in ("blocked", "no_route") and ped_id:
            if active_run_id not in self._blocked_from_events:
                self._blocked_from_events[active_run_id] = {}

            # Out-of-order check: if agent already observed at step > this event step, do not override
            last_agent_step = self._agent_last_recorded_step.get(ped_id, -1)
            if step >= last_agent_step:
                self._blocked_from_events[active_run_id][ped_id] = step
                self._blocked_agents.add(ped_id)
                self._active_agents.discard(ped_id)

        # 5. Record event ingestion
        self._seen_event_keys.add(evt_key)
        rep = EventAdaptationReport(
            event_id=evt_id,
            run_id=active_run_id,
            step_index=step,
            timestamp=ts_f,
            event_type=event_type_str,
            pedestrian_id=ped_id,
            is_duplicate=False,
            status="ACCEPTED",
            details=details,
        )

        if active_run_id not in self._ingested_events_by_run:
            self._ingested_events_by_run[active_run_id] = []
        self._ingested_events_by_run[active_run_id].append(rep)

        return rep

    def ingest_events(
        self,
        events: Sequence[Any],
        run_id: Optional[str] = None,
    ) -> List[EventAdaptationReport]:
        """Ingest a batch of events with run-scoped deduplication."""
        reports: List[EventAdaptationReport] = []
        for evt in events:
            reports.append(self.adapt_event(evt, run_id=run_id))
        return reports

    def read_only_engine_ingest(
        self,
        engine: Any,
        run_id: Optional[str] = None,
        force_snapshot: bool = False,
    ) -> Tuple[Optional[SnapshotAdaptationReport], List[EventAdaptationReport]]:
        """Perform a certified read-only ingestion of snapshots and events from a running engine."""
        active_run_id = str(run_id).strip() if (run_id is not None and str(run_id).strip()) else self._current_run_id

        # 1. Snapshot engine properties before ingestion
        before_step = getattr(engine, "current_step", None)
        if before_step is None and hasattr(engine, "clock"):
            before_step = getattr(engine.clock, "step_count", None)

        before_ts = getattr(engine, "timestamp", None)
        if before_ts is None and hasattr(engine, "clock"):
            before_ts = getattr(engine.clock, "timestamp", None)

        before_events_len = len(getattr(engine, "events", ()))

        # 2. Ingest snapshot
        snapshot_report: Optional[SnapshotAdaptationReport] = None
        if hasattr(engine, "get_snapshot") and callable(engine.get_snapshot):
            snap = engine.get_snapshot()
            if snap is not None:
                snapshot_report = self.adapt_snapshot(snap, force=force_snapshot, run_id=active_run_id)
        elif hasattr(engine, "latest_snapshot") and engine.latest_snapshot is not None:
            snapshot_report = self.adapt_snapshot(engine.latest_snapshot, force=force_snapshot, run_id=active_run_id)
        else:
            # Fallback to direct engine inspection
            self.update_from_simulation(engine, force=force_snapshot, run_id=active_run_id)

        # 3. Ingest events
        event_reports: List[EventAdaptationReport] = []
        if hasattr(engine, "latest_events") and getattr(engine, "latest_events", None):
            event_reports.extend(self.ingest_events(engine.latest_events, run_id=active_run_id))
        elif hasattr(engine, "events") and getattr(engine, "events", None):
            event_reports.extend(self.ingest_events(engine.events, run_id=active_run_id))

        # 4. Verify engine was not mutated
        after_step = getattr(engine, "current_step", None)
        if after_step is None and hasattr(engine, "clock"):
            after_step = getattr(engine.clock, "step_count", None)

        after_ts = getattr(engine, "timestamp", None)
        if after_ts is None and hasattr(engine, "clock"):
            after_ts = getattr(engine.clock, "timestamp", None)

        after_events_len = len(getattr(engine, "events", ()))

        if before_step != after_step:
            raise RuntimeError(f"Engine step mutated during ingestion: {before_step} -> {after_step}")
        if before_ts != after_ts:
            raise RuntimeError(f"Engine timestamp mutated during ingestion: {before_ts} -> {after_ts}")
        if before_events_len != after_events_len:
            raise RuntimeError(f"Engine event count mutated during ingestion: {before_events_len} -> {after_events_len}")

        return snapshot_report, event_reports

    def update_from_simulation(self, engine: Any, force: bool = False, run_id: Optional[str] = None) -> int:
        """Inspect running simulation engine and record observations if cadence is met."""
        active_run_id = str(run_id).strip() if (run_id is not None and str(run_id).strip()) else self._current_run_id
        if active_run_id != self._current_run_id:
            self.reset(run_id=active_run_id)

        # Validate current_step
        current_step = getattr(engine, "current_step", None)
        if current_step is None and hasattr(engine, "clock"):
            current_step = getattr(engine.clock, "step_count", None)
        if current_step is None:
            current_step = getattr(engine, "step_count", None)
        if type(current_step) is not int or current_step < 0:
            raise TrajectoryValidationError(
                f"engine.current_step must be a non-negative integer, got {current_step!r}",
                field="current_step",
            )

        # Validate timestep
        timestep = getattr(engine, "timestep", None)
        if timestep is None and hasattr(engine, "clock"):
            timestep = getattr(engine.clock, "dt", None)
        if timestep is None:
            timestep = getattr(engine, "dt", 0.1)

        if (
            type(timestep) not in (float, int)
            or isinstance(timestep, bool)
            or not math.isfinite(timestep)
            or timestep <= 0.0
        ):
            raise TrajectoryValidationError(
                f"engine.timestep must be a finite positive number, got {timestep!r}",
                field="timestep",
            )
        timestep = float(timestep)

        ts_attr = getattr(engine, "timestamp", None)
        if ts_attr is None and hasattr(engine, "clock"):
            ts_attr = getattr(engine.clock, "timestamp", None)
        current_timestamp = float(ts_attr) if ts_attr is not None else (current_step * timestep)

        if not math.isfinite(current_timestamp) or current_timestamp < 0.0:
            raise TrajectoryValidationError(
                f"derived current_timestamp must be a finite non-negative number, got {current_timestamp!r}",
                field="current_timestamp",
            )

        # Reset signal detection
        current_engine_id = id(engine)
        engine_changed = self._last_engine_id is not None and self._last_engine_id != current_engine_id
        step_rollback = current_step < self._last_observed_step

        if engine_changed or step_rollback:
            self.reset(run_id=active_run_id)

        self._last_engine_id = current_engine_id
        self._last_observed_step = current_step

        step_key = (active_run_id, current_step)
        if step_key in self._seen_snapshot_steps and not force:
            return 0

        # Cadence check
        if current_step == self._last_recorded_step and not force:
            return 0

        if not force and self._last_recorded_timestamp >= 0.0:
            time_since_last = current_timestamp - self._last_recorded_timestamp
            if time_since_last < (self.observation_interval - 1e-6):
                return 0

        # Process pedestrians with strict TrajectoryValidationError raising
        pedestrians = getattr(engine, "pedestrians", ())
        recorded_count = 0
        current_candidate_ids = {
            str(getattr(p, "pedestrian_id", getattr(p, "id", ""))).strip()
            for p in pedestrians
            if getattr(p, "pedestrian_id", getattr(p, "id", None))
        }

        for ped in pedestrians:
            ped_id = getattr(ped, "pedestrian_id", getattr(ped, "id", None))
            if not ped_id or not str(ped_id).strip():
                raise TrajectoryValidationError(
                    "Missing or empty pedestrian ID.",
                    field="agent_id",
                    step_index=current_step,
                )
            ped_id = str(ped_id).strip()

            raw_status = getattr(ped, "status", None)
            if raw_status is None:
                raise TrajectoryValidationError(
                    f"Pedestrian '{ped_id}' has no status attribute.",
                    field="status",
                    agent_id=ped_id,
                    step_index=current_step,
                )
            status_val = raw_status.value if hasattr(raw_status, "value") else raw_status
            status_str = str(status_val).strip().lower()

            event_blocked_map = self._blocked_from_events.get(active_run_id, {})
            evt_block_step = event_blocked_map.get(ped_id, -1)

            if status_str == "evacuated" or ped_id in self._evacuated_agents:
                self._evacuated_agents.add(ped_id)
                self._active_agents.discard(ped_id)
                self._blocked_agents.discard(ped_id)
                self._incapacitated_agents.discard(ped_id)
                event_blocked_map.pop(ped_id, None)
                continue

            if status_str == "incapacitated" or ped_id in self._incapacitated_agents:
                self._incapacitated_agents.add(ped_id)
                self._active_agents.discard(ped_id)
                self._blocked_agents.discard(ped_id)
                self._evacuated_agents.discard(ped_id)
                event_blocked_map.pop(ped_id, None)
                continue

            if status_str == "inactive":
                self._inactive_agents.add(ped_id)
                self._active_agents.discard(ped_id)
                self._blocked_agents.discard(ped_id)
                self._incapacitated_agents.discard(ped_id)
                event_blocked_map.pop(ped_id, None)
                continue

            if status_str == "blocked":
                self._blocked_agents.add(ped_id)
                self._active_agents.discard(ped_id)
            elif status_str == "active":
                # If authoritative snapshot step is >= event step, newer snapshot takes precedence
                if current_step >= evt_block_step:
                    self._active_agents.add(ped_id)
                    self._blocked_agents.discard(ped_id)
                    event_blocked_map.pop(ped_id, None)
                else:
                    self._blocked_agents.add(ped_id)
                    self._active_agents.discard(ped_id)
            else:
                raise TrajectoryValidationError(
                    f"Unknown pedestrian status '{raw_status!r}' for agent '{ped_id}'. "
                    f"Allowed statuses: 'active', 'blocked', 'evacuated', 'incapacitated', 'inactive'.",
                    field="status",
                    agent_id=ped_id,
                    step_index=current_step,
                )

            # Extract coordinates
            if hasattr(ped, "position"):
                pos = ped.position
                coord_x = float(pos[0]) if hasattr(pos, "__getitem__") else float(pos.x)
                coord_y = float(pos[1]) if hasattr(pos, "__getitem__") else float(pos.y)
            elif hasattr(ped, "x") and hasattr(ped, "y"):
                coord_x = float(ped.x)
                coord_y = float(ped.y)
            else:
                raise TrajectoryValidationError(
                    f"Pedestrian '{ped_id}' missing coordinates.",
                    field="position",
                    agent_id=ped_id,
                    step_index=current_step,
                )

            last_agent_step = self._agent_last_recorded_step.get(ped_id, -1)
            if current_step <= last_agent_step:
                continue

            self.buffer.record(
                observation_or_agent_id=ped_id,
                position=(coord_x, coord_y),
                step_index=current_step,
                timestamp=current_timestamp,
                category=TrajectoryCategory.SIMULATED,
            )
            self._agent_last_recorded_step[ped_id] = current_step
            self._agent_last_recorded_time[ped_id] = current_timestamp
            recorded_count += 1

        all_previously_tracked = (self._active_agents | self._blocked_agents) - current_candidate_ids
        for missing_id in all_previously_tracked:
            self._not_observed_agents.add(missing_id)
            self._active_agents.discard(missing_id)
            self._blocked_agents.discard(missing_id)

        self._last_recorded_step = current_step
        self._last_recorded_timestamp = current_timestamp
        self._seen_snapshot_steps.add(step_key)
        return recorded_count

    def get_prediction_snapshot(
        self,
        current_step: int,
        timestep: float = 0.1,
        cohort: str = "active",
        agent_ids: Optional[Sequence[str]] = None,
        max_observation_age: Optional[float] = -1.0,
    ) -> TrajectorySnapshot:
        """Retrieve an immutable TrajectorySnapshot formatted for prediction/metrics models."""
        current_timestamp = current_step * timestep

        if agent_ids is not None:
            target_ids = list(agent_ids)
        elif cohort == "active":
            target_ids = sorted(self._active_agents)
        elif cohort == "blocked":
            target_ids = sorted(self._blocked_agents)
        elif cohort == "all_tracked":
            target_ids = sorted(self._active_agents | self._blocked_agents)
        else:
            raise ValueError(f"Unknown cohort '{cohort}'. Allowed: 'active', 'blocked', 'all_tracked'")

        effective_max_age = (
            self.max_observation_age
            if max_observation_age == -1.0
            else max_observation_age
        )

        return self.buffer.get_snapshot(
            current_step=current_step,
            current_timestamp=current_timestamp,
            timestep=timestep,
            agent_ids=target_ids,
            max_observation_age=effective_max_age,
        )

    @property
    def active_agents(self) -> frozenset[str]:
        return frozenset(self._active_agents)

    @property
    def blocked_agents(self) -> frozenset[str]:
        return frozenset(self._blocked_agents)

    @property
    def evacuated_agents(self) -> frozenset[str]:
        return frozenset(self._evacuated_agents)

    @property
    def incapacitated_agents(self) -> frozenset[str]:
        return frozenset(self._incapacitated_agents)

    @property
    def inactive_agents(self) -> frozenset[str]:
        return frozenset(self._inactive_agents)

    @property
    def not_observed_agents(self) -> frozenset[str]:
        return frozenset(self._not_observed_agents)

    @property
    def trajectory_histories(self) -> Dict[str, Tuple[Any, ...]]:
        return dict(self._pedestrian_trajectory_histories)


def _capture_engine_public_state(engine: Any) -> Dict[str, Any]:
    """Capture a comprehensive, immutable snapshot of the engine's public state."""
    state: Dict[str, Any] = {}

    # 1. Clock step, timestamp, dt
    step = getattr(engine, "current_step", None)
    if step is None and hasattr(engine, "clock"):
        step = getattr(engine.clock, "step_count", None)
    if step is None:
        step = getattr(engine, "step_count", None)
    state["step"] = step

    ts = getattr(engine, "timestamp", None)
    if ts is None and hasattr(engine, "clock"):
        ts = getattr(engine.clock, "timestamp", None)
    state["timestamp"] = ts

    dt = getattr(engine, "timestep", None)
    if dt is None and hasattr(engine, "clock"):
        dt = getattr(engine.clock, "dt", None)
    if dt is None:
        dt = getattr(engine, "dt", None)
    state["dt"] = dt

    # 2. Pause status
    state["is_paused"] = getattr(engine, "is_paused", None)

    # 3. Latest snapshot identity and content
    snap = getattr(engine, "latest_snapshot", None)
    if snap is not None:
        state["latest_snapshot"] = (
            getattr(snap, "snapshot_id", None),
            getattr(snap, "step_count", None),
            getattr(snap, "timestamp", None),
            getattr(snap, "total_count", None),
            getattr(snap, "active_count", None),
            getattr(snap, "evacuated_count", None),
            getattr(snap, "blocked_count", None),
            getattr(snap, "inactive_count", None),
            len(getattr(snap, "pedestrians", ())),
        )
    else:
        state["latest_snapshot"] = None

    # 4. Events deep comparison (identities, types, timestamps, steps, and details)
    events = getattr(engine, "events", ())
    state["events"] = tuple(
        (
            getattr(e, "event_id", None),
            getattr(e, "step", None),
            getattr(e, "timestamp", None),
            getattr(e, "event_type", None),
            getattr(e, "pedestrian_id", None),
            tuple(sorted((getattr(e, "details", {}) or {}).items())),
        )
        for e in events
    )

    # 5. Pedestrians (identities, positions, velocities, statuses, history)
    peds = getattr(engine, "pedestrians", None)
    if peds is None and hasattr(engine, "registry"):
        peds = list(engine.registry)
    if peds is not None:
        state["pedestrians"] = tuple(
            (
                getattr(p, "id", getattr(p, "pedestrian_id", "")),
                (
                    tuple(p.position) if hasattr(p, "position") and hasattr(p.position, "__iter__")
                    else (getattr(p.position, "x", 0.0), getattr(p.position, "y", 0.0)) if hasattr(p, "position")
                    else (getattr(p, "x", 0.0), getattr(p, "y", 0.0))
                ),
                (
                    tuple(p.velocity) if hasattr(p, "velocity") and hasattr(p.velocity, "__iter__")
                    else (getattr(p.velocity, "x", 0.0), getattr(p.velocity, "y", 0.0)) if hasattr(p, "velocity")
                    else (getattr(p, "vx", 0.0), getattr(p, "vy", 0.0))
                ),
                str(getattr(p, "status", "")),
                getattr(p, "target_exit_id", getattr(p, "target_exit", "")),
                tuple(
                    (getattr(h, "step", 0), getattr(h, "timestamp", 0.0), getattr(h, "x", 0.0), getattr(h, "y", 0.0))
                    for h in getattr(p, "trajectory_history", ())
                ),
            )
            for p in peds
        )
    else:
        state["pedestrians"] = ()

    return state


def verify_engine_read_only(engine: Any, action: Callable[[], Any]) -> Tuple[bool, List[str]]:
    """Execute an action while verifying that engine state remains strictly unmutated.

    Checks:
    - Step count / clock step
    - Timestamp and dt
    - Engine pause status
    - Latest snapshot identity and contents
    - Event identities and full details content
    - Pedestrian positions, statuses, velocities, and trajectory history

    Returns:
        (is_unmutated: bool, violation_messages: List[str])
    """
    violations: List[str] = []
    before = _capture_engine_public_state(engine)
    action()
    after = _capture_engine_public_state(engine)

    if before["step"] != after["step"]:
        violations.append(f"Engine step count changed: {before['step']} -> {after['step']}")
    if before["timestamp"] != after["timestamp"]:
        violations.append(f"Engine timestamp changed: {before['timestamp']} -> {after['timestamp']}")
    if before["dt"] != after["dt"]:
        violations.append(f"Engine dt changed: {before['dt']} -> {after['dt']}")
    if before["is_paused"] != after["is_paused"]:
        violations.append(f"Engine pause status changed: {before['is_paused']} -> {after['is_paused']}")
    if before["latest_snapshot"] != after["latest_snapshot"]:
        violations.append(f"Engine latest_snapshot changed: {before['latest_snapshot']} -> {after['latest_snapshot']}")
    if before["events"] != after["events"]:
        violations.append(f"Engine event collection or content changed: {before['events']} -> {after['events']}")
    if before["pedestrians"] != after["pedestrians"]:
        violations.append("Pedestrian states, positions, velocities, or histories changed during ingestion.")

    return (len(violations) == 0, violations)
