"""Trajectory data models, contracts, and validation."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


class TrajectoryCategory(str, Enum):
    """Category of trajectory data."""
    OBSERVED = "observed"    # External dataset or real-world recordings (e.g. ETH/UCY)
    SIMULATED = "simulated"  # Authoritative observations from GenEvac simulation engine
    PREDICTED = "predicted"  # Future trajectory estimates from AI / prediction models


class TrajectoryValidationError(ValueError):
    """Raised when trajectory data violates contract requirements."""
    def __init__(
        self,
        message: str,
        field: str | None = None,
        agent_id: str | None = None,
        step_index: int | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.field = field
        self.agent_id = agent_id
        self.step_index = step_index

    def __str__(self) -> str:
        details = []
        if self.field:
            details.append(f"field='{self.field}'")
        if self.agent_id is not None:
            details.append(f"agent_id='{self.agent_id}'")
        if self.step_index is not None:
            details.append(f"step_index={self.step_index}")
        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message


@dataclass(frozen=True)
class TrajectoryObservation:
    """Immutable single-step pedestrian trajectory observation.
    
    Coordinates follow the agreed GenEvac convention:
    - Origin: top-left (0, 0)
    - Positive X: right (east)
    - Positive Y: downward (south)
    - Spatial units: metres
    - Temporal units: seconds
    """
    agent_id: str
    step_index: int
    timestamp: float
    x: float
    y: float
    category: TrajectoryCategory = TrajectoryCategory.SIMULATED

    def __post_init__(self) -> None:
        # Validate agent_id
        if not isinstance(self.agent_id, str):
            raise TrajectoryValidationError(
                f"agent_id must be a string, got {type(self.agent_id).__name__}",
                field="agent_id",
                agent_id=str(self.agent_id),
            )
        if not self.agent_id.strip():
            raise TrajectoryValidationError(
                "agent_id cannot be empty or whitespace",
                field="agent_id",
                agent_id=self.agent_id,
            )

        # Validate step_index (must be int, not bool, >= 0)
        if type(self.step_index) is not int:
            raise TrajectoryValidationError(
                f"step_index must be an integer (bool not allowed), got {type(self.step_index).__name__}",
                field="step_index",
                agent_id=self.agent_id,
            )
        if self.step_index < 0:
            raise TrajectoryValidationError(
                f"step_index must be non-negative, got {self.step_index}",
                field="step_index",
                agent_id=self.agent_id,
                step_index=self.step_index,
            )

        # Validate timestamp (finite, float/int, >= 0)
        if type(self.timestamp) not in (float, int) or isinstance(self.timestamp, bool):
            raise TrajectoryValidationError(
                f"timestamp must be a float or int, got {type(self.timestamp).__name__}",
                field="timestamp",
                agent_id=self.agent_id,
                step_index=self.step_index,
            )
        if not math.isfinite(self.timestamp):
            raise TrajectoryValidationError(
                f"timestamp must be finite, got {self.timestamp}",
                field="timestamp",
                agent_id=self.agent_id,
                step_index=self.step_index,
            )
        if self.timestamp < 0.0:
            raise TrajectoryValidationError(
                f"timestamp must be non-negative, got {self.timestamp}",
                field="timestamp",
                agent_id=self.agent_id,
                step_index=self.step_index,
            )

        # Validate coordinates x and y (finite, numeric)
        for coord_name, coord_val in (("x", self.x), ("y", self.y)):
            if type(coord_val) not in (float, int) or isinstance(coord_val, bool):
                raise TrajectoryValidationError(
                    f"{coord_name} coordinate must be a float or int, got {type(coord_val).__name__}",
                    field=coord_name,
                    agent_id=self.agent_id,
                    step_index=self.step_index,
                )
            if not math.isfinite(coord_val):
                raise TrajectoryValidationError(
                    f"{coord_name} coordinate must be finite, got {coord_val}",
                    field=coord_name,
                    agent_id=self.agent_id,
                    step_index=self.step_index,
                )

        # Validate category
        if not isinstance(self.category, TrajectoryCategory):
            try:
                object.__setattr__(self, "category", TrajectoryCategory(self.category))
            except ValueError:
                raise TrajectoryValidationError(
                    f"Invalid trajectory category: {self.category}",
                    field="category",
                    agent_id=self.agent_id,
                    step_index=self.step_index,
                )

    @property
    def position(self) -> tuple[float, float]:
        """Convenience property returning (x, y) tuple."""
        return (self.x, self.y)


@dataclass(frozen=True)
class TrajectorySnapshot:
    """Immutable read-only view of trajectory histories at a specific simulation step."""
    current_step: int
    current_timestamp: float
    timestep: float
    trajectories: Mapping[str, tuple[TrajectoryObservation, ...]]

    def __post_init__(self) -> None:
        if type(self.current_step) is not int or self.current_step < 0:
            raise TrajectoryValidationError(
                f"current_step must be a non-negative integer, got {self.current_step}",
                field="current_step",
            )
        if type(self.current_timestamp) not in (float, int) or not math.isfinite(self.current_timestamp) or self.current_timestamp < 0.0:
            raise TrajectoryValidationError(
                f"current_timestamp must be a finite non-negative number, got {self.current_timestamp}",
                field="current_timestamp",
            )
        if type(self.timestep) not in (float, int) or not math.isfinite(self.timestep) or self.timestep <= 0.0:
            raise TrajectoryValidationError(
                f"timestep must be a finite positive number, got {self.timestep}",
                field="timestep",
            )

        # Freeze at the model boundary, not only in TrajectoryHistoryBuffer.
        # A defensive copy prevents later mutation through the caller's mapping.
        if not isinstance(self.trajectories, Mapping):
            raise TrajectoryValidationError(
                "trajectories must be a mapping of agent IDs to observation sequences",
                field="trajectories",
            )
        frozen_trajectories: dict[str, tuple[TrajectoryObservation, ...]] = {}
        for agent_id, observations in self.trajectories.items():
            if not isinstance(agent_id, str) or not agent_id.strip():
                raise TrajectoryValidationError(
                    "trajectory mapping keys must be non-empty agent IDs",
                    field="agent_id",
                )
            if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
                raise TrajectoryValidationError(
                    f"Observations for agent '{agent_id}' must be a sequence",
                    field="trajectories",
                    agent_id=agent_id,
                )
            frozen_sequence = tuple(observations)
            for observation in frozen_sequence:
                if not isinstance(observation, TrajectoryObservation):
                    raise TrajectoryValidationError(
                        f"All entries for agent '{agent_id}' must be TrajectoryObservation instances",
                        field="trajectories",
                        agent_id=agent_id,
                    )
            frozen_trajectories[agent_id] = frozen_sequence
        object.__setattr__(self, "trajectories", MappingProxyType(frozen_trajectories))


@dataclass(frozen=True)
class PredictedPosition:
    """Immutable single predicted future coordinate point with timestamp."""
    timestamp: float
    x: float
    y: float

    def __post_init__(self) -> None:
        if type(self.timestamp) not in (float, int) or isinstance(self.timestamp, bool):
            raise TrajectoryValidationError(
                f"timestamp must be a float or int, got {type(self.timestamp).__name__}",
                field="timestamp",
            )
        if not math.isfinite(self.timestamp) or self.timestamp < 0.0:
            raise TrajectoryValidationError(
                f"timestamp must be a finite non-negative number, got {self.timestamp}",
                field="timestamp",
            )
        for coord_name, coord_val in (("x", self.x), ("y", self.y)):
            if type(coord_val) not in (float, int) or isinstance(coord_val, bool) or not math.isfinite(coord_val):
                raise TrajectoryValidationError(
                    f"{coord_name} coordinate must be a finite number, got {coord_val}",
                    field=coord_name,
                )

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class TrajectoryPrediction:
    """Immutable future predicted trajectory for an agent.
    
    Predicted positions are estimates from AI or prediction models,
    and are distinct from actual observed or simulated state.
    """
    agent_id: str
    source_step_index: int
    source_timestamp: float
    predictions: tuple[PredictedPosition, ...]
    category: TrajectoryCategory = field(default=TrajectoryCategory.PREDICTED, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise TrajectoryValidationError(
                "agent_id must be a non-empty string",
                field="agent_id",
                agent_id=str(self.agent_id),
            )
        if type(self.source_step_index) is not int or self.source_step_index < 0:
            raise TrajectoryValidationError(
                f"source_step_index must be a non-negative integer, got {self.source_step_index}",
                field="source_step_index",
                agent_id=self.agent_id,
            )
        if type(self.source_timestamp) not in (float, int) or not math.isfinite(self.source_timestamp) or self.source_timestamp < 0.0:
            raise TrajectoryValidationError(
                f"source_timestamp must be a finite non-negative number, got {self.source_timestamp}",
                field="source_timestamp",
                agent_id=self.agent_id,
            )

        # Predictions sequence must be non-empty tuple of PredictedPosition
        if not isinstance(self.predictions, tuple):
            object.__setattr__(self, "predictions", tuple(self.predictions))

        prev_time = self.source_timestamp
        for idx, pred in enumerate(self.predictions):
            if not isinstance(pred, PredictedPosition):
                raise TrajectoryValidationError(
                    f"prediction element {idx} must be a PredictedPosition, got {type(pred).__name__}",
                    field="predictions",
                    agent_id=self.agent_id,
                )
            if pred.timestamp <= prev_time:
                raise TrajectoryValidationError(
                    f"Predicted timestamp at index {idx} ({pred.timestamp}) must be strictly later than previous timestamp ({prev_time})",
                    field="predictions.timestamp",
                    agent_id=self.agent_id,
                )
            prev_time = pred.timestamp


class TrajectoryValidator:
    """Validator for observations, sequences, snapshots, and prediction results."""

    @staticmethod
    def validate_observation(obs: TrajectoryObservation) -> None:
        """Validate a single trajectory observation."""
        if not isinstance(obs, TrajectoryObservation):
            raise TrajectoryValidationError(
                f"Expected TrajectoryObservation instance, got {type(obs).__name__}",
                field="observation",
            )
        # TrajectoryObservation.__post_init__ guarantees individual field validity

    @staticmethod
    def validate_observation_sequence(
        observations: Sequence[TrajectoryObservation],
        expected_agent_id: str | None = None,
        min_history: int | None = None,
        expected_timestep: float | None = None,
        tolerance: float = 1e-4,
    ) -> None:
        """Validate an ordered sequence of observations for an agent.
        
        Checks:
        - Consistent agent ID
        - Strictly increasing step_index
        - Strictly increasing timestamp
        - Consistency with expected_timestep if specified
        - Minimum history threshold check
        """
        if not observations:
            if min_history and min_history > 0:
                raise TrajectoryValidationError(
                    f"Observation sequence is empty (requires at least {min_history} observations)",
                    field="observations",
                    agent_id=expected_agent_id,
                )
            return

        if min_history is not None and len(observations) < min_history:
            raise TrajectoryValidationError(
                f"Insufficient history: got {len(observations)} observations, required at least {min_history}",
                field="observations",
                agent_id=expected_agent_id or observations[0].agent_id,
            )

        agent_id = expected_agent_id or observations[0].agent_id
        prev_step: int | None = None
        prev_time: float | None = None

        for idx, obs in enumerate(observations):
            TrajectoryValidator.validate_observation(obs)

            if obs.agent_id != agent_id:
                raise TrajectoryValidationError(
                    f"Mismatched agent_id at index {idx}: expected '{agent_id}', got '{obs.agent_id}'",
                    field="agent_id",
                    agent_id=obs.agent_id,
                    step_index=obs.step_index,
                )

            if prev_step is not None:
                if obs.step_index == prev_step:
                    raise TrajectoryValidationError(
                        f"Duplicate observation for agent '{agent_id}' at step {obs.step_index}",
                        field="step_index",
                        agent_id=agent_id,
                        step_index=obs.step_index,
                    )
                if obs.step_index < prev_step:
                    raise TrajectoryValidationError(
                        f"Out-of-order step_index for agent '{agent_id}': step {obs.step_index} follows step {prev_step}",
                        field="step_index",
                        agent_id=agent_id,
                        step_index=obs.step_index,
                    )

            if prev_time is not None:
                if obs.timestamp <= prev_time:
                    raise TrajectoryValidationError(
                        f"Inconsistent timestamp for agent '{agent_id}': timestamp {obs.timestamp} is not greater than previous {prev_time}",
                        field="timestamp",
                        agent_id=agent_id,
                        step_index=obs.step_index,
                    )

                if expected_timestep is not None and obs.category == TrajectoryCategory.SIMULATED:
                    dt = obs.timestamp - prev_time
                    step_diff = obs.step_index - (prev_step if prev_step is not None else obs.step_index - 1)
                    expected_dt = expected_timestep * step_diff
                    if abs(dt - expected_dt) > tolerance:
                        raise TrajectoryValidationError(
                            f"Timestamp step discrepancy for agent '{agent_id}': observed dt={dt:.6f}s does not match expected {expected_dt:.6f}s",
                            field="timestamp",
                            agent_id=agent_id,
                            step_index=obs.step_index,
                        )

            prev_step = obs.step_index
            prev_time = obs.timestamp

    @staticmethod
    def validate_snapshot(
        snapshot: TrajectorySnapshot,
        max_observation_age: float | None = None,
    ) -> None:
        """Validate a TrajectorySnapshot and all contained observation sequences.
        
        Args:
            snapshot: TrajectorySnapshot instance to validate.
            max_observation_age: Optional maximum permitted age (in seconds) between
                snapshot.current_timestamp and the latest observation timestamp for each agent.
                If any agent's latest observation is older than max_observation_age,
                TrajectoryValidationError is raised.
        """
        if not isinstance(snapshot, TrajectorySnapshot):
            raise TrajectoryValidationError(
                f"Expected TrajectorySnapshot instance, got {type(snapshot).__name__}",
                field="snapshot",
            )

        if max_observation_age is not None:
            if (
                type(max_observation_age) not in (float, int)
                or isinstance(max_observation_age, bool)
                or not math.isfinite(max_observation_age)
                or max_observation_age < 0.0
            ):
                raise TrajectoryValidationError(
                    f"max_observation_age must be a finite non-negative number, got {max_observation_age!r}",
                    field="max_observation_age",
                )

        for agent_id, obs_seq in snapshot.trajectories.items():
            TrajectoryValidator.validate_observation_sequence(
                obs_seq,
                expected_agent_id=agent_id,
                expected_timestep=snapshot.timestep,
            )

            # A snapshot cannot contain measurements from its future. Allow
            # only 1e-9 seconds of representation noise from floating-point math.
            timestamp_tolerance = 1e-9
            for observation in obs_seq:
                if observation.timestamp > snapshot.current_timestamp + timestamp_tolerance:
                    raise TrajectoryValidationError(
                        f"Future observation for agent '{agent_id}': observation timestamp "
                        f"{observation.timestamp:.12g}s exceeds snapshot timestamp "
                        f"{snapshot.current_timestamp:.12g}s",
                        field="observation_timestamp",
                        agent_id=agent_id,
                        step_index=observation.step_index,
                    )

            if max_observation_age is not None and obs_seq:
                latest_obs_time = obs_seq[-1].timestamp
                # Clamp tiny tolerated negative age to zero.
                age = max(0.0, snapshot.current_timestamp - latest_obs_time)
                if age > max_observation_age + 1e-6:
                    raise TrajectoryValidationError(
                        f"Stale observation for agent '{agent_id}': latest observation timestamp is {latest_obs_time:.6f}s, "
                        f"which is {age:.6f}s older than snapshot timestamp {snapshot.current_timestamp:.6f}s "
                        f"(max allowed: {max_observation_age:.6f}s)",
                        field="observation_age",
                        agent_id=agent_id,
                        step_index=obs_seq[-1].step_index,
                    )

    @staticmethod
    def validate_prediction(prediction: TrajectoryPrediction) -> None:
        """Validate a TrajectoryPrediction."""
        if not isinstance(prediction, TrajectoryPrediction):
            raise TrajectoryValidationError(
                f"Expected TrajectoryPrediction instance, got {type(prediction).__name__}",
                field="prediction",
            )
        # TrajectoryPrediction.__post_init__ guarantees internal validity

