"""AI prediction models, trajectory data contracts, and history buffer."""

import math
from collections import deque
from collections.abc import Sequence
from types import MappingProxyType

from genevac.prediction.adapter import SimulationTrajectoryAdapter, SnapshotAdaptationReport
from genevac.prediction.baseline import ConstantVelocityPredictor
from genevac.prediction.dataset_adapter import DatasetAdapter, SimulationEvaluationAdapter
from genevac.prediction.evaluation import (
    AggregateEvaluationReport,
    EvaluationSample,
    PredictionEvaluator,
    SampleGenerator,
    TrajectoryEvaluationResult,
)
from genevac.prediction.mid import (
    EvacuationConditioning,
    EvacuationMIDConfig,
    EvacuationMIDNetwork,
    EvacuationMIDPredictor,
    MultiTrajectoryPrediction,
    line_segment_intersects_box,
)
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

    def __init__(
        self,
        window_size: int = 20,
        min_history_required: int = 2,
        max_observation_age: float | None = None,
    ):
        if type(window_size) is not int or window_size <= 0:
            raise ValueError(f"window_size must be a positive integer, got {window_size}")
        if type(min_history_required) is not int or min_history_required < 1:
            raise ValueError(f"min_history_required must be an integer >= 1, got {min_history_required}")
        if max_observation_age is not None:
            if type(max_observation_age) not in (float, int) or isinstance(max_observation_age, bool):
                raise ValueError("max_observation_age must be a finite non-negative number or None")
            if max_observation_age < 0:
                raise ValueError("max_observation_age cannot be negative")
            if not math.isfinite(max_observation_age):
                raise ValueError("max_observation_age must be finite")
        self.window_size = window_size
        self.min_history_required = min_history_required
        self.max_observation_age = max_observation_age
        self.history: dict[str, deque[TrajectoryObservation]] = {}

    def record(
        self,
        observation_or_agent_id: TrajectoryObservation | str,
        position: tuple[float, float] | None = None,
        step_index: int | None = None,
        timestamp: float | None = None,
        category: TrajectoryCategory = TrajectoryCategory.SIMULATED,
    ) -> None:
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

            agent_id_str = observation_or_agent_id
            if not isinstance(position, (tuple, list)) or len(position) != 2:
                raise TrajectoryValidationError(
                    f"position must be a (x, y) tuple or list of length 2, got {position!r}",
                    field="position",
                    agent_id=agent_id_str,
                )
            for coord_name, coord_val in (("x", position[0]), ("y", position[1])):
                if isinstance(coord_val, bool) or type(coord_val) not in (float, int):
                    raise TrajectoryValidationError(
                        f"position {coord_name} coordinate must be a finite float or int, got {type(coord_val).__name__!r}",
                        field=f"position.{coord_name}",
                        agent_id=agent_id_str,
                    )
                if not math.isfinite(coord_val):
                    raise TrajectoryValidationError(
                        f"position {coord_name} coordinate must be finite, got {coord_val!r}",
                        field=f"position.{coord_name}",
                        agent_id=agent_id_str,
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

        if obs.category == TrajectoryCategory.PREDICTED:
            raise TrajectoryValidationError(
                f"Predicted observations must not be recorded in the authoritative history buffer. "
                f"Use TrajectoryCategory.SIMULATED or OBSERVED for agent '{obs.agent_id}'.",
                field="category",
                agent_id=obs.agent_id,
                step_index=obs.step_index,
            )

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
        agent_deque = self.history.get(agent_id, deque())
        if as_tuples:
            return [(obs.x, obs.y) for obs in agent_deque]
        return list(agent_deque)

    def get_observations(self, agent_id: str) -> tuple[TrajectoryObservation, ...]:
        return tuple(self.history.get(agent_id, ()))

    def has_sufficient_history(self, agent_id: str, min_required: int | None = None) -> bool:
        threshold = min_required if min_required is not None else self.min_history_required
        return len(self.history.get(agent_id, ())) >= threshold

    def get_agent_ids(self) -> list[str]:
        return list(self.history.keys())

    def is_latest_observation_fresh(
        self,
        agent_id: str,
        reference_timestamp: float,
        max_age: float | None = None,
    ) -> bool:
        allowed_age = max_age if max_age is not None else self.max_observation_age
        if allowed_age is None:
            return True

        observations = self.history.get(agent_id)
        if not observations:
            return False

        latest_obs = observations[-1]
        age = reference_timestamp - latest_obs.timestamp
        return age <= (allowed_age + 1e-6)

    def get_snapshot(
        self,
        current_step: int,
        current_timestamp: float,
        timestep: float = 0.1,
        agent_ids: Sequence[str] | None = None,
        max_observation_age: float | None = None,
    ) -> TrajectorySnapshot:
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
        effective_max_age = (
            max_observation_age if max_observation_age is not None else self.max_observation_age
        )
        TrajectoryValidator.validate_snapshot(snapshot, max_observation_age=effective_max_age)
        return snapshot

    def clear(self, agent_id: str | None = None) -> None:
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
    "AggregateEvaluationReport",
    "ConstantVelocityPredictor",
    "DatasetAdapter",
    "EvaluationSample",
    "EvacuationConditioning",
    "EvacuationMIDConfig",
    "EvacuationMIDNetwork",
    "EvacuationMIDPredictor",
    "LinearTrajectoryPredictor",
    "MultiTrajectoryPrediction",
    "line_segment_intersects_box",
    "PredictedPosition",
    "PredictionEvaluator",
    "SampleGenerator",
    "SimulationEvaluationAdapter",
    "SimulationTrajectoryAdapter",
    "SnapshotAdaptationReport",
    "TrajectoryCategory",
    "TrajectoryEvaluationResult",
    "TrajectoryHistoryBuffer",
    "TrajectoryObservation",
    "TrajectoryPrediction",
    "TrajectorySnapshot",
    "TrajectoryValidationError",
    "TrajectoryValidator",
]
