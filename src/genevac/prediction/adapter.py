"""Simulation-to-prediction adapter for GenEvac-Sentinel.

Bridges Person A's simulation engine and Person B's prediction subsystem:
- Non-invasively monitors simulation progression without mutating simulation state.
- Converts authoritative simulation state to validated TrajectoryObservation records.
- Enforces cadence filtering (e.g. 0.4s observation interval on a 0.1s simulation engine)
  robust against duplicate steps, irregular invocation, and skipped steps.
- Tracks pedestrian cohorts (ACTIVE, BLOCKED, EVACUATED, INACTIVE) strictly within the
  adapter without polluting core TrajectoryObservation contracts.
- Treats EVACUATED as a terminal state for the current run (preserves history, blocks recording).
- Detects simulation resets via step rollbacks or engine instance replacements, clearing
  prior run history to prevent cross-run pollution.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from genevac.prediction import TrajectoryHistoryBuffer

from genevac.prediction.data_contract import (
    TrajectoryCategory,
    TrajectoryObservation,
    TrajectorySnapshot,
)


class SimulationTrajectoryAdapter:
    """Adapter bridging authoritative SimulationEngine state to prediction buffers.

    Responsibilities:
    1. Reset detection: Tracks engine instance identity and monotonic step progress.
       If engine instance changes or current_step < last_observed_step, resets tracking
       and clears the buffer.
    2. Observation cadence: Configurable interval (e.g. 0.4s / 4 steps). Only records
       when elapsed steps or time satisfy the observation cadence. Handles duplicate
       steps and irregular invocations idempotently.
    3. Status & Cohort management:
       - ACTIVE: Recorded into history buffer.
       - BLOCKED: Recorded into history buffer, tracked in blocked cohort set.
       - EVACUATED: Terminal for current run. Past observations preserved; further recording rejected.
       - INACTIVE: Completely ignored.
    4. Prediction history snapshot:
       Exposes ``get_prediction_snapshot()`` to create an immutable TrajectorySnapshot
       specifically for prediction input, clearly distinct from simulation engine snapshots.
    """

    def __init__(
        self,
        buffer: TrajectoryHistoryBuffer | None = None,
        observation_interval: float = 0.4,
        max_observation_age: float | None = 0.45,
        window_size: int = 20,
        min_history_required: int = 2,
    ) -> None:
        """Initialize the adapter.

        Args:
            buffer: Optional existing TrajectoryHistoryBuffer. If None, a new one is created.
            observation_interval: Minimum time (seconds) between successive observations
                recorded for any agent (default 0.4s = 4 steps of 0.1s simulation).
            max_observation_age: Maximum age (seconds) for observation freshness when
                generating prediction snapshots. Defaults to 0.45s (covering 0.4s nominal
                cadence + 0.05s tolerance). None disables age validation.
            window_size: Buffer window size if creating a new buffer.
            min_history_required: Buffer min history if creating a new buffer.
        """
        if buffer is not None:
            self.buffer = buffer
        else:
            # Lazy import to avoid circular dependency
            from genevac.prediction import TrajectoryHistoryBuffer

            self.buffer = TrajectoryHistoryBuffer(
                window_size=window_size,
                min_history_required=min_history_required,
                max_observation_age=None,  # Handled with adapter's configurable max_observation_age
            )
        self.observation_interval = float(observation_interval)
        self.max_observation_age = float(max_observation_age) if max_observation_age is not None else None

        # Reset & Run tracking (without inventing synthetic run IDs)
        self._last_engine_ref: weakref.ref[Any] | None = None
        self._last_engine_id: int | None = None
        self._last_observed_step: int = -1
        self._last_recorded_step: int = -1
        self._last_recorded_timestamp: float = -1.0

        # Cohort tracking (kept strictly inside adapter)
        self._active_agents: set[str] = set()
        self._blocked_agents: set[str] = set()
        self._evacuated_agents: set[str] = set()
        self._inactive_agents: set[str] = set()

        # Per-agent last recorded step/time to guard irregular or out-of-order calls
        self._agent_last_recorded_step: dict[str, int] = {}
        self._agent_last_recorded_time: dict[str, float] = {}

    def reset(self) -> None:
        """Explicitly reset adapter state and clear the underlying prediction buffer."""
        self.buffer.clear()
        self._last_observed_step = -1
        self._last_recorded_step = -1
        self._last_recorded_timestamp = -1.0
        self._active_agents.clear()
        self._blocked_agents.clear()
        self._evacuated_agents.clear()
        self._inactive_agents.clear()
        self._agent_last_recorded_step.clear()
        self._agent_last_recorded_time.clear()

    def update_from_simulation(self, engine: Any, force: bool = False) -> int:
        """Inspect simulation engine and record observations if cadence conditions are met.

        Args:
            engine: Authoritative SimulationEngine instance.
            force: If True, forces recording at the current step regardless of observation cadence.

        Returns:
            Number of pedestrian observations recorded in this update (0 if cadence not met or duplicate).
        """
        current_step: int = engine.current_step
        timestep: float = getattr(engine, "timestep", 0.1)
        current_timestamp: float = current_step * timestep

        # --- Reset Signal Detection ---
        # 1. Engine instance identity changed
        current_engine_id = id(engine)
        engine_changed = self._last_engine_id is not None and self._last_engine_id != current_engine_id

        # 2. Step rollback (monotonic step drop)
        step_rollback = current_step < self._last_observed_step

        if engine_changed or step_rollback:
            self.reset()

        self._last_engine_id = current_engine_id
        self._last_engine_ref = weakref.ref(engine)
        self._last_observed_step = current_step

        # --- Cadence & Duplicate Step Check ---
        # If this exact step was already recorded, do not record again (idempotent for repeated calls)
        if current_step == self._last_recorded_step and not force:
            return 0

        # Check if enough time/steps have passed according to observation_interval
        if not force and self._last_recorded_timestamp >= 0.0:
            time_since_last = current_timestamp - self._last_recorded_timestamp
            # Allow 1e-6 floating-point tolerance on interval multiples
            if time_since_last < (self.observation_interval - 1e-6):
                return 0

        # --- Process Pedestrians ---
        pedestrians = getattr(engine, "pedestrians", ())
        recorded_count = 0

        current_step_active: set[str] = set()
        current_step_blocked: set[str] = set()

        for ped in pedestrians:
            ped_id = str(ped.id)
            raw_status = getattr(ped, "status", "active")
            status_str = raw_status.value if hasattr(raw_status, "value") else str(raw_status).lower()

            # Handle EVACUATED: terminal for remainder of run
            if status_str == "evacuated" or ped_id in self._evacuated_agents:
                self._evacuated_agents.add(ped_id)
                self._active_agents.discard(ped_id)
                self._blocked_agents.discard(ped_id)
                # Terminal: preserve historical observations, but reject further recording
                continue

            # Handle INACTIVE: ignore completely
            if status_str == "inactive":
                self._inactive_agents.add(ped_id)
                self._active_agents.discard(ped_id)
                self._blocked_agents.discard(ped_id)
                continue

            # Handle BLOCKED vs ACTIVE
            if status_str == "blocked":
                current_step_blocked.add(ped_id)
                self._blocked_agents.add(ped_id)
                self._active_agents.discard(ped_id)
            else:
                current_step_active.add(ped_id)
                self._active_agents.add(ped_id)
                self._blocked_agents.discard(ped_id)

            # Guard against per-agent duplicate/out-of-order steps
            last_agent_step = self._agent_last_recorded_step.get(ped_id, -1)
            if current_step <= last_agent_step:
                continue

            # Extract coordinates (preserving top-left frame, metres)
            pos = ped.position
            coord_x = float(pos[0])
            coord_y = float(pos[1])

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

        self._last_recorded_step = current_step
        self._last_recorded_timestamp = current_timestamp
        return recorded_count

    def get_prediction_snapshot(
        self,
        current_step: int,
        timestep: float = 0.1,
        cohort: str = "active",
        agent_ids: Sequence[str] | None = None,
        max_observation_age: float | None = -1.0,  # Sentinel for default
    ) -> TrajectorySnapshot:
        """Retrieve an immutable TrajectorySnapshot specifically formatted for prediction models.

        Note:
            This snapshot captures historical observations from the prediction cache and is
            strictly distinct from any authoritative simulation state snapshot produced by Person A.

        Args:
            current_step: Current simulation step.
            timestep: Authoritative simulation timestep.
            cohort: Target evaluation cohort:
                - 'active': only active agents (default).
                - 'blocked': only blocked agents.
                - 'all_tracked': both active and blocked agents.
            agent_ids: Optional explicit list of agent IDs. If provided, overrides cohort filtering.
            max_observation_age: Freshness age limit override. If -1.0 (sentinel), uses
                self.max_observation_age. None disables freshness check.

        Returns:
            Immutable TrajectorySnapshot validated and ready for ConstantVelocityPredictor.
        """
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

    # --- Cohort inspection accessors (read-only views) ---

    @property
    def active_agents(self) -> frozenset[str]:
        """Set of currently active agent IDs tracked by the adapter."""
        return frozenset(self._active_agents)

    @property
    def blocked_agents(self) -> frozenset[str]:
        """Set of currently blocked agent IDs tracked by the adapter."""
        return frozenset(self._blocked_agents)

    @property
    def evacuated_agents(self) -> frozenset[str]:
        """Set of evacuated agent IDs (terminal for the current run)."""
        return frozenset(self._evacuated_agents)

    @property
    def inactive_agents(self) -> frozenset[str]:
        """Set of inactive agent IDs excluded from prediction."""
        return frozenset(self._inactive_agents)
