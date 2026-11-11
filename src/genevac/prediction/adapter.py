"""Simulation-to-prediction adapter for GenEvac-Sentinel.

Bridges Person A's simulation engine and Person B's prediction/metrics subsystem:
- Non-invasively monitors simulation progression without mutating simulation state.
- Ingests both live SimulationEngine instances and immutable SimulationObservationSnapshot objects.
- Converts authoritative simulation state to validated TrajectoryObservation records.
- Enforces cadence filtering (e.g. 0.4s observation interval on a 0.1s simulation engine)
  robust against duplicate steps, irregular invocation, and skipped steps.
- Tracks pedestrian cohorts (ACTIVE, BLOCKED, EVACUATED, INACTIVE) strictly within the
  adapter without polluting core TrajectoryObservation contracts.
- Treats EVACUATED as a terminal state for the current run (preserves history, blocks recording).
- Detects simulation resets via step rollbacks or engine instance replacements, clearing
  prior run history to prevent cross-run pollution.
- Returns structured SnapshotAdaptationReport for auditable ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set

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
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)


class SimulationTrajectoryAdapter:
    """Adapter bridging authoritative SimulationEngine state to prediction/metrics buffers.

    Responsibilities:
    1. Reset detection: Tracks engine instance identity and monotonic step progress.
       If engine instance changes or current_step < last_observed_step, resets tracking
       and clears the buffer.
    2. Observation cadence: Configurable interval (e.g. 0.4s / 4 steps). Only records
       when elapsed steps or time satisfy the observation cadence. Handles duplicate
       steps and irregular invocations idempotently.
    3. Direct Snapshot Ingestion: Exposes ``adapt_snapshot(snapshot)`` to consume pre-built
       immutable SimulationObservationSnapshot instances.
    4. Status & Cohort management:
       - ACTIVE: Recorded into history buffer.
       - BLOCKED: Recorded into history buffer, tracked in blocked cohort set.
       - EVACUATED: Terminal for current run. Past observations preserved; further recording rejected.
       - INCAPACITATED: Preserved, non-active, further recording rejected.
       - INACTIVE: Completely ignored.
    5. Prediction & Analytics history snapshot:
       Exposes ``get_prediction_snapshot()`` to create an immutable TrajectorySnapshot
       specifically for prediction and metric computation.
    """

    def __init__(
        self,
        buffer: Optional[TrajectoryHistoryBuffer] = None,
        observation_interval: float = 0.4,
        max_observation_age: Optional[float] = 0.45,
        window_size: int = 20,
        min_history_required: int = 2,
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

        # Reset & Run tracking
        self._last_engine_id: Optional[int] = None
        self._last_observed_step: int = -1
        self._last_recorded_step: int = -1
        self._last_recorded_timestamp: float = -1.0
        self._seen_snapshot_ids: Set[str] = set()

        # Cohort tracking (kept strictly inside adapter)
        self._active_agents: Set[str] = set()
        self._blocked_agents: Set[str] = set()
        self._evacuated_agents: Set[str] = set()
        self._incapacitated_agents: Set[str] = set()
        self._inactive_agents: Set[str] = set()
        self._not_observed_agents: Set[str] = set()

        # Per-agent last recorded step/time to guard irregular or out-of-order calls
        self._agent_last_recorded_step: Dict[str, int] = {}
        self._agent_last_recorded_time: Dict[str, float] = {}

    def reset(self) -> None:
        """Explicitly reset adapter state and clear the underlying prediction buffer."""
        self.buffer.clear()
        self._last_engine_id = None
        self._last_observed_step = -1
        self._last_recorded_step = -1
        self._last_recorded_timestamp = -1.0
        self._seen_snapshot_ids.clear()
        self._active_agents.clear()
        self._blocked_agents.clear()
        self._evacuated_agents.clear()
        self._incapacitated_agents.clear()
        self._inactive_agents.clear()
        self._not_observed_agents.clear()
        self._agent_last_recorded_step.clear()
        self._agent_last_recorded_time.clear()

    def adapt_snapshot(self, snapshot: Any, force: bool = False) -> SnapshotAdaptationReport:
        """Adapt a verified SimulationObservationSnapshot object into canonical observations.

        Args:
            snapshot: Immutable SimulationObservationSnapshot instance or object with step/timestamp/pedestrians.
            force: If True, forces recording at current step regardless of observation cadence.

        Returns:
            SnapshotAdaptationReport detailing accepted, rejected, and duplicate counts.
        """
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
            self.reset()
        self._last_observed_step = step

        # Duplicate check by snapshot_id or step
        if snap_id in self._seen_snapshot_ids or (step == self._last_recorded_step and not force):
            return SnapshotAdaptationReport(
                snapshot_id=snap_id,
                step_index=step,
                timestamp=ts,
                total_candidates=0,
                accepted_observations=0,
                rejected_observations=0,
                is_duplicate=True,
                rejection_reasons=("Duplicate snapshot step or snapshot_id already processed.",),
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
                )

        peds = getattr(snapshot, "pedestrians", ())
        accepted = 0
        rejected = 0
        rejection_reasons: List[str] = []

        for ped in peds:
            # Extract ped_id
            ped_id = getattr(ped, "pedestrian_id", getattr(ped, "id", None))
            if not ped_id or not str(ped_id).strip():
                rejected += 1
                rejection_reasons.append("Empty or missing pedestrian ID.")
                continue
            ped_id = str(ped_id).strip()

            # Extract status
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

            # Route status after verifying coordinates are valid
            if status_str == "evacuated" or ped_id in self._evacuated_agents:
                self._evacuated_agents.add(ped_id)
                self._active_agents.discard(ped_id)
                self._blocked_agents.discard(ped_id)
                self._incapacitated_agents.discard(ped_id)
                continue

            if status_str == "incapacitated" or ped_id in self._incapacitated_agents:
                self._incapacitated_agents.add(ped_id)
                self._active_agents.discard(ped_id)
                self._blocked_agents.discard(ped_id)
                continue

            if status_str == "inactive":
                self._inactive_agents.add(ped_id)
                self._active_agents.discard(ped_id)
                self._blocked_agents.discard(ped_id)
                self._incapacitated_agents.discard(ped_id)
                continue

            if status_str == "blocked":
                self._blocked_agents.add(ped_id)
                self._active_agents.discard(ped_id)
            elif status_str == "active":
                self._active_agents.add(ped_id)
                self._blocked_agents.discard(ped_id)
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

        # Check for previously tracked agents absent in this snapshot
        current_candidate_ids = {
            str(getattr(p, "pedestrian_id", getattr(p, "id", ""))).strip()
            for p in peds
            if getattr(p, "pedestrian_id", getattr(p, "id", None))
        }
        # Any agent that was previously active/blocked but is absent from this snapshot becomes NOT_OBSERVED (never assumed evacuated)
        all_previously_tracked = (self._active_agents | self._blocked_agents) - current_candidate_ids
        for missing_id in all_previously_tracked:
            self._not_observed_agents.add(missing_id)
            self._active_agents.discard(missing_id)
            self._blocked_agents.discard(missing_id)

        self._last_recorded_step = step
        self._last_recorded_timestamp = ts
        self._seen_snapshot_ids.add(snap_id)

        return SnapshotAdaptationReport(
            snapshot_id=snap_id,
            step_index=step,
            timestamp=ts,
            total_candidates=len(peds),
            accepted_observations=accepted,
            rejected_observations=rejected,
            is_duplicate=False,
            rejection_reasons=tuple(rejection_reasons),
        )

    def update_from_simulation(self, engine: Any, force: bool = False) -> int:
        """Inspect running simulation engine and record observations if cadence is met.

        Args:
            engine: Authoritative SimulationEngine instance.
            force: If True, forces recording at the current step regardless of observation cadence.

        Returns:
            Number of pedestrian observations recorded in this update.
        """
        # Validate current_step
        current_step = getattr(engine, "current_step", None)
        if current_step is None:
            current_step = getattr(engine, "step_count", None)
        if type(current_step) is not int or current_step < 0:
            raise TrajectoryValidationError(
                f"engine.current_step must be a non-negative integer, got {current_step!r}",
                field="current_step",
            )

        # Validate timestep
        timestep = getattr(engine, "timestep", getattr(engine, "dt", 0.1))
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
        current_timestamp: float = current_step * timestep
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
            self.reset()

        self._last_engine_id = current_engine_id
        self._last_observed_step = current_step

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

            if status_str == "evacuated" or ped_id in self._evacuated_agents:
                self._evacuated_agents.add(ped_id)
                self._active_agents.discard(ped_id)
                self._blocked_agents.discard(ped_id)
                self._incapacitated_agents.discard(ped_id)
                continue

            if status_str == "incapacitated" or ped_id in self._incapacitated_agents:
                self._incapacitated_agents.add(ped_id)
                self._active_agents.discard(ped_id)
                self._blocked_agents.discard(ped_id)
                continue

            if status_str == "inactive":
                self._inactive_agents.add(ped_id)
                self._active_agents.discard(ped_id)
                self._blocked_agents.discard(ped_id)
                self._incapacitated_agents.discard(ped_id)
                continue

            if status_str == "blocked":
                self._blocked_agents.add(ped_id)
                self._active_agents.discard(ped_id)
            elif status_str == "active":
                self._active_agents.add(ped_id)
                self._blocked_agents.discard(ped_id)
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
        """Set of previously observed agent IDs that are absent from the latest snapshot without explicit terminal status."""
        return frozenset(self._not_observed_agents)
