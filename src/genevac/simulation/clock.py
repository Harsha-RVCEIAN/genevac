"""Simulation clock module for GenEvac-Sentinel.

Provides an authoritative deterministic clock owned by the SimulationEngine:
- Step-based progression: integer step count
- Derived timestamp: timestamp = step_count * dt (eliminates cumulative drift)
- Observation cadence synchronization: exact integer stride validation
"""

from __future__ import annotations

import math


class SimulationClock:
    """Authoritative clock for the simulation engine.

    Guarantees:
    - Single source of temporal truth
    - Deterministic progression
    - Zero cumulative floating-point drift
    - Strict validation of dt and observation_interval
    """

    def __init__(self, dt: float = 0.1, observation_interval: float = 0.4) -> None:
        try:
            dt_f = float(dt)
            obs_f = float(observation_interval)
        except (TypeError, ValueError) as err:
            raise TypeError(
                f"dt and observation_interval must be numeric: dt={dt!r}, obs={observation_interval!r}"
            ) from err

        if not (math.isfinite(dt_f) and dt_f > 0.0):
            raise ValueError(f"dt must be strictly positive and finite (> 0), got {dt}")
        if not (math.isfinite(obs_f) and obs_f > 0.0):
            raise ValueError(
                f"observation_interval must be strictly positive and finite (> 0), got {observation_interval}"
            )

        if obs_f < dt_f:
            raise ValueError(
                f"observation_interval ({obs_f} s) must be greater than or equal to dt ({dt_f} s)"
            )

        ratio = obs_f / dt_f
        nearest_int = round(ratio)
        if abs(ratio - nearest_int) > 1e-6:
            raise ValueError(
                f"observation_interval ({obs_f} s) must be an exact integer multiple of dt ({dt_f} s). "
                f"Ratio {ratio} is not an integer."
            )

        self._dt: float = dt_f
        self._observation_interval: float = obs_f
        self._observation_stride: int = int(nearest_int)
        self._step_count: int = 0

    @property
    def dt(self) -> float:
        """Fixed simulation timestep in seconds."""
        return self._dt

    @property
    def observation_interval(self) -> float:
        """Observation/snapshot sampling interval in seconds."""
        return self._observation_interval

    @property
    def observation_stride(self) -> int:
        """Number of simulation steps between observation snapshots."""
        return self._observation_stride

    @property
    def step_count(self) -> int:
        """Current simulation step index (starts at 0)."""
        return self._step_count

    @property
    def timestamp(self) -> float:
        """Current simulation time in seconds derived from step_count * dt."""
        return round(self._step_count * self._dt, 9)

    def is_observation_step(self) -> bool:
        """Return True if the current step corresponds to an observation cadence tick."""
        return (self._step_count % self._observation_stride) == 0

    def tick(self) -> tuple[int, float]:
        """Advance the clock by one simulation step.

        Returns:
            Tuple of (new_step_count, new_timestamp).
        """
        self._step_count += 1
        return (self._step_count, self.timestamp)

    def reset(self) -> None:
        """Reset the clock back to initial state (step 0, t=0.0)."""
        self._step_count = 0
