"""Pure kinematic Constant Velocity baseline predictor."""

from __future__ import annotations

import math
from collections.abc import Sequence

from genevac.prediction.data_contract import (
    PredictedPosition,
    TrajectoryObservation,
    TrajectoryPrediction,
    TrajectorySnapshot,
    TrajectoryValidationError,
)


class ConstantVelocityPredictor:
    """Timestamp-aware constant-velocity trajectory predictor.

    Computes velocity from the last two valid observations using their
    actual timestamps:
        vx = (x2 - x1) / (t2 - t1)
        vy = (y2 - y1) / (t2 - t1)

    For future timestamp t_future:
        p_future = p2 + v * (t_future - t2)

    This baseline NEVER:
    - Uses wall-clock time.
    - Assumes uniform timestep spacing.
    - Modifies simulation state or pedestrian objects.
    - Appends predictions to the authoritative history buffer.
    """

    def predict(
        self,
        snapshot: TrajectorySnapshot,
        agent_id: str,
        horizon: int = 12,
        prediction_interval: float | None = None,
        max_observation_age: float | None = None,
    ) -> TrajectoryPrediction:
        """Predict future positions for an agent using constant velocity extrapolation.

        Args:
            snapshot: Read-only TrajectorySnapshot of the simulation state.
            agent_id: Identifier of the target pedestrian.
            horizon: Number of future timesteps to predict (must be an integer > 0).
            prediction_interval: Time interval between predicted steps in seconds.
                If None, defaults to snapshot.timestep. Must be > 0.
            max_observation_age: Optional maximum permitted age (in seconds) between
                snapshot.current_timestamp and the agent's latest observation timestamp.
                If the latest observation is older than this limit, TrajectoryValidationError
                is raised.

        Returns:
            An immutable TrajectoryPrediction instance containing PredictedPosition points
            with timestamps strictly greater than snapshot.current_timestamp.

        Raises:
            TrajectoryValidationError: If the agent has fewer than 2 observations,
                horizon <= 0, prediction_interval <= 0, timestamps are invalid, or
                latest observation exceeds max_observation_age.
        """
        if type(horizon) is not int or horizon <= 0:
            raise TrajectoryValidationError(
                f"horizon must be a positive integer, got {horizon!r}",
                field="horizon",
                agent_id=agent_id,
            )

        interval = prediction_interval if prediction_interval is not None else snapshot.timestep
        if (
            type(interval) not in (float, int)
            or isinstance(interval, bool)
            or not math.isfinite(interval)
            or interval <= 0.0
        ):
            raise TrajectoryValidationError(
                f"prediction_interval must be a finite positive number, got {interval!r}",
                field="prediction_interval",
                agent_id=agent_id,
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
                    agent_id=agent_id,
                )

        observations: Sequence[TrajectoryObservation] = snapshot.trajectories.get(agent_id, ())
        if len(observations) < 2:
            raise TrajectoryValidationError(
                f"Insufficient history for agent '{agent_id}': requires at least 2 observations to compute velocity, got {len(observations)}.",
                field="observations",
                agent_id=agent_id,
            )

        obs_prev = observations[-2]
        obs_last = observations[-1]

        if max_observation_age is not None:
            age = snapshot.current_timestamp - obs_last.timestamp
            if age > max_observation_age + 1e-6:
                raise TrajectoryValidationError(
                    f"Stale observation for agent '{agent_id}': latest observation timestamp is {obs_last.timestamp:.6f}s, "
                    f"which is {age:.6f}s older than snapshot timestamp {snapshot.current_timestamp:.6f}s "
                    f"(max allowed: {max_observation_age:.6f}s)",
                    field="observation_age",
                    agent_id=agent_id,
                    step_index=obs_last.step_index,
                )

        dt = obs_last.timestamp - obs_prev.timestamp
        if dt <= 0.0 or not math.isfinite(dt):
            raise TrajectoryValidationError(
                f"Non-positive or non-finite time delta for agent '{agent_id}': dt={dt:.6f}s between steps {obs_prev.step_index} and {obs_last.step_index}.",
                field="timestamp",
                agent_id=agent_id,
            )

        vx = (obs_last.x - obs_prev.x) / dt
        vy = (obs_last.y - obs_prev.y) / dt

        source_timestamp = snapshot.current_timestamp
        predicted_positions: list[PredictedPosition] = []

        for k in range(1, horizon + 1):
            t_future = source_timestamp + k * interval
            x_future = obs_last.x + vx * (t_future - obs_last.timestamp)
            y_future = obs_last.y + vy * (t_future - obs_last.timestamp)
            predicted_positions.append(
                PredictedPosition(timestamp=t_future, x=x_future, y=y_future)
            )

        return TrajectoryPrediction(
            agent_id=agent_id,
            source_step_index=snapshot.current_step,
            source_timestamp=source_timestamp,
            predictions=tuple(predicted_positions),
        )

    def predict_batch(
        self,
        snapshot: TrajectorySnapshot,
        agent_ids: Sequence[str] | None = None,
        horizon: int = 12,
        prediction_interval: float | None = None,
        max_observation_age: float | None = None,
    ) -> dict[str, TrajectoryPrediction | TrajectoryValidationError]:
        """Predict future trajectories for multiple agents from a snapshot.

        Args:
            snapshot: Read-only simulation state snapshot.
            agent_ids: List/Sequence of agent IDs to predict for. If None,
                predicts for all agents present in snapshot.
            horizon: Number of future timesteps to predict.
            prediction_interval: Time interval between predicted steps.
            max_observation_age: Optional maximum permitted observation age.

        Returns:
            Dictionary mapping agent_id to TrajectoryPrediction (if successful)
            or TrajectoryValidationError (if agent has insufficient history or invalid data).
        """
        targets = agent_ids if agent_ids is not None else list(snapshot.trajectories.keys())
        results: dict[str, TrajectoryPrediction | TrajectoryValidationError] = {}
        for aid in targets:
            try:
                results[aid] = self.predict(
                    snapshot=snapshot,
                    agent_id=aid,
                    horizon=horizon,
                    prediction_interval=prediction_interval,
                    max_observation_age=max_observation_age,
                )
            except TrajectoryValidationError as err:
                results[aid] = err
        return results


__all__ = ["ConstantVelocityPredictor"]

