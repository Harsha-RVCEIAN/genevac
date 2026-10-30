"""AI prediction models, trajectory data contracts, and history buffer."""

from collections import deque
from collections.abc import Sequence
from types import MappingProxyType

from genevac.prediction.baseline import ConstantVelocityPredictor
from genevac.prediction.data_contract import (
    PredictedPosition,
    TrajectoryCategory,
    TrajectoryObservation,
    TrajectoryPrediction,
    TrajectorySnapshot,
    TrajectoryValidationError,
    TrajectoryValidator,
)


class TrajectoryHistoryBuffer:
    """Prediction-history cache storing validated trajectory observations.
    
    Maintains a rolling window of TrajectoryObservation instances per agent.
    Consumes observations from the simulation engine without independently
    advancing time or altering pedestrian state.
    """

    def __init__(self, window_size: int = 20, min_history_required: int = 2):
        if type(window_size) is not int or window_size <= 0:
            raise ValueError(f"window_size must be a positive integer, got {window_size}")
        if type(min_history_required) is not int or min_history_required < 1:
            raise ValueError(f"min_history_required must be an integer >= 1, got {min_history_required}")

        self.window_size = window_size
        self.min_history_required = min_history_required
        self.history: dict[str, deque[TrajectoryObservation]] = {}

    def record(
        self,
        observation_or_agent_id: TrajectoryObservation | str,
        position: tuple[float, float] | None = None,
        step_index: int | None = None,
        timestamp: float | None = None,
        category: TrajectoryCategory = TrajectoryCategory.SIMULATED,
    ) -> None:
        """Record an observation in the cache.
        
        Can accept either a pre-constructed TrajectoryObservation or separate fields:
        - `buf.record(obs)`
        - `buf.record("ped_1", (10.0, 5.0), step_index=1, timestamp=0.1)`
        """
        if isinstance(observation_or_agent_id, TrajectoryObservation):
            obs = observation_or_agent_id
        elif isinstance(observation_or_agent_id, str):
            if position is None:
                raise TrajectoryValidationError(
                    "position tuple (x, y) must be provided when recording by agent_id",
                    field="position",
                    agent_id=observation_or_agent_id,
                )
            if step_index is None or timestamp is None:
                raise TrajectoryValidationError(
                    "Simulation engine must supply step_index and timestamp. Independent inference is prohibited.",
                    field="step_index/timestamp",
                    agent_id=observation_or_agent_id,
                )
            obs = TrajectoryObservation(
                agent_id=observation_or_agent_id,
                step_index=step_index,
                timestamp=timestamp,
                x=float(position[0]),
                y=float(position[1]),
                category=category,
            )
        else:
            raise TrajectoryValidationError(
                f"Expected TrajectoryObservation or str agent_id, got {type(observation_or_agent_id).__name__}",
                field="observation",
            )

        # Reject PREDICTED category — predicted observations must never enter the authoritative buffer
        if obs.category == TrajectoryCategory.PREDICTED:
            raise TrajectoryValidationError(
                f"Predicted observations must not be recorded in the authoritative history buffer. "
                f"Use TrajectoryCategory.SIMULATED or OBSERVED for agent '{obs.agent_id}'.",
                field="category",
                agent_id=obs.agent_id,
                step_index=obs.step_index,
            )

        # Validate chronological order and check duplicates against the agent's last observation
        agent_id = obs.agent_id
        if agent_id not in self.history:
            self.history[agent_id] = deque(maxlen=self.window_size)

        agent_deque = self.history[agent_id]
        if agent_deque:
            last_obs = agent_deque[-1]
            if obs.step_index == last_obs.step_index:
                raise TrajectoryValidationError(
                    f"Duplicate observation for agent '{agent_id}' at step {obs.step_index}",
                    field="step_index",
                    agent_id=agent_id,
                    step_index=obs.step_index,
                )
            if obs.step_index < last_obs.step_index:
                raise TrajectoryValidationError(
                    f"Out-of-order observation for agent '{agent_id}': step {obs.step_index} follows step {last_obs.step_index}",
                    field="step_index",
                    agent_id=agent_id,
                    step_index=obs.step_index,
                )
            if obs.timestamp <= last_obs.timestamp:
                raise TrajectoryValidationError(
                    f"Inconsistent timestamp for agent '{agent_id}': timestamp {obs.timestamp} is not greater than previous {last_obs.timestamp}",
                    field="timestamp",
                    agent_id=agent_id,
                    step_index=obs.step_index,
                )

        agent_deque.append(obs)

    def get_trajectory(self, agent_id: str, as_tuples: bool = True) -> list[tuple[float, float]] | list[TrajectoryObservation]:
        """Retrieve an agent's history.
        
        Args:
            agent_id: Identifier of the agent.
            as_tuples: If True (default), returns list of (x, y) tuples for backwards
                       compatibility with LinearTrajectoryPredictor.
                       If False, returns list of TrajectoryObservation instances.
        """
        agent_deque = self.history.get(agent_id, deque())
        if as_tuples:
            return [(obs.x, obs.y) for obs in agent_deque]
        return list(agent_deque)

    def get_observations(self, agent_id: str) -> tuple[TrajectoryObservation, ...]:
        """Retrieve an agent's observations as an immutable tuple."""
        return tuple(self.history.get(agent_id, ()))

    def has_sufficient_history(self, agent_id: str, min_required: int | None = None) -> bool:
        """Check whether the agent has accumulated at least min_required observations."""
        threshold = min_required if min_required is not None else self.min_history_required
        return len(self.history.get(agent_id, ())) >= threshold

    def get_agent_ids(self) -> list[str]:
        """Return list of all agent IDs currently tracked in the buffer."""
        return list(self.history.keys())

    def get_snapshot(
        self,
        current_step: int,
        current_timestamp: float,
        timestep: float = 0.1,
        agent_ids: Sequence[str] | None = None,
    ) -> TrajectorySnapshot:
        """Create a read-only, immutable TrajectorySnapshot representing simulation state."""
        target_ids = agent_ids if agent_ids is not None else list(self.history.keys())
        trajectories: dict[str, tuple[TrajectoryObservation, ...]] = {}
        for aid in target_ids:
            if aid in self.history:
                trajectories[aid] = tuple(self.history[aid])

        snapshot = TrajectorySnapshot(
            current_step=current_step,
            current_timestamp=current_timestamp,
            timestep=timestep,
            trajectories=MappingProxyType(trajectories),
        )
        TrajectoryValidator.validate_snapshot(snapshot)
        return snapshot

    def clear(self, agent_id: str | None = None) -> None:
        """Clear history for a specific agent or all agents."""
        if agent_id is not None:
            if agent_id in self.history:
                del self.history[agent_id]
        else:
            self.history.clear()


class LinearTrajectoryPredictor:
    """Baseline predictor assuming constant velocity extrapolation."""

    def predict(self, history: list[tuple[float, float]], horizon: int = 10) -> list[tuple[float, float]]:
        if len(history) < 2:
            return [history[-1]] * horizon if history else []
        dx = history[-1][0] - history[-2][0]
        dy = history[-1][1] - history[-2][1]

        predictions = []
        curr_x, curr_y = history[-1]
        for _ in range(horizon):
            curr_x += dx
            curr_y += dy
            predictions.append((curr_x, curr_y))
        return predictions


__all__ = [
    "ConstantVelocityPredictor",
    "LinearTrajectoryPredictor",
    "PredictedPosition",
    "TrajectoryCategory",
    "TrajectoryHistoryBuffer",
    "TrajectoryObservation",
    "TrajectoryPrediction",
    "TrajectorySnapshot",
    "TrajectoryValidationError",
    "TrajectoryValidator",
]
