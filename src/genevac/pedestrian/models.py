"""Pedestrian domain models for GenEvac-Sentinel.

Provides data primitives and state representation for simulated pedestrians:
- PedestrianStatus: Lifecycle state enum (ACTIVE, EVACUATED, BLOCKED, INACTIVE)
- Velocity2D: 2D velocity vector in m/s with speed and validation
- TimestampedTrajectoryPoint: Historical coordinate sample with external step and timestamp
- PedestrianState: Authoritative state of a pedestrian (position, velocity, status, history)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Optional, Union

from genevac.environment.geometry import Point


class PedestrianStatus(str, Enum):
    """Lifecycle status of a simulated pedestrian."""

    ACTIVE = "active"
    EVACUATED = "evacuated"
    BLOCKED = "blocked"
    INACTIVE = "inactive"


@dataclass(frozen=True)
class Velocity2D:
    """2D velocity vector in metres per second.

    Components must be finite numerical values.
    """

    vx: float
    vy: float

    def __post_init__(self) -> None:
        try:
            vx_f = float(self.vx)
            vy_f = float(self.vy)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Velocity components must be numeric, got vx={self.vx!r}, vy={self.vy!r}") from err

        if not (math.isfinite(vx_f) and math.isfinite(vy_f)):
            raise ValueError(f"Velocity components must be finite: vx={self.vx}, vy={self.vy}")

        object.__setattr__(self, "vx", vx_f)
        object.__setattr__(self, "vy", vy_f)

    @property
    def speed(self) -> float:
        """Scalar speed in metres per second."""
        return math.hypot(self.vx, self.vy)

    def as_tuple(self) -> tuple[float, float]:
        """Return (vx, vy) as a tuple of floats."""
        return (self.vx, self.vy)


@dataclass(frozen=True)
class TimestampedTrajectoryPoint:
    """A timestamped historical coordinate sample of a pedestrian.

    The simulation engine supplies step and timestamp; this class does not maintain
    an independent simulation clock.
    """

    step: int
    timestamp: float
    x: float
    y: float

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool):
            raise TypeError(f"Step must be an integer, got {type(self.step).__name__}")
        if self.step < 0:
            raise ValueError(f"Step must be non-negative, got {self.step}")

        try:
            ts_f = float(self.timestamp)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Timestamp must be numeric, got {self.timestamp!r}") from err
        if not (math.isfinite(ts_f) and ts_f >= 0.0):
            raise ValueError(f"Timestamp must be non-negative and finite, got {self.timestamp}")

        try:
            x_f = float(self.x)
            y_f = float(self.y)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Coordinates must be numeric, got x={self.x!r}, y={self.y!r}") from err
        if not (math.isfinite(x_f) and math.isfinite(y_f)):
            raise ValueError(f"Coordinates must be finite: ({self.x}, {self.y})")

        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "timestamp", ts_f)
        object.__setattr__(self, "x", x_f)
        object.__setattr__(self, "y", y_f)

    def as_tuple(self) -> tuple[int, float, float, float]:
        """Return (step, timestamp, x, y) as a tuple."""
        return (self.step, self.timestamp, self.x, self.y)


@dataclass
class PedestrianState:
    """Authoritative state of an individual pedestrian.

    Does not maintain an internal clock; step and timestamp for history recording
    are provided externally by the simulation engine.
    """

    id: str
    position: Point
    velocity: Velocity2D = field(default_factory=lambda: Velocity2D(0.0, 0.0))
    status: PedestrianStatus = PedestrianStatus.ACTIVE
    target_exit_id: Optional[str] = None
    current_area_id: Optional[str] = None
    waypoints: list[Point] = field(default_factory=list)
    radius: float = 0.25
    preferred_speed: float = 1.3
    trajectory_history: list[TimestampedTrajectoryPoint] = field(default_factory=list)
    max_history_len: int = 100

    def __post_init__(self) -> None:
        self._coerce_and_validate()

    def _coerce_and_validate(self) -> None:
        """Validate and coerce attributes to maintain strong invariant guarantees."""
        # 1. ID validation
        if not self.id or not isinstance(self.id, str):
            raise ValueError(f"Pedestrian ID must be a non-empty string, got {self.id!r}")

        # 2. Position coercion & validation
        if isinstance(self.position, (tuple, list)):
            if len(self.position) != 2:
                raise ValueError(f"Position coordinate sequence must have length 2, got {self.position!r}")
            self.position = Point(self.position[0], self.position[1])
        elif not isinstance(self.position, Point):
            raise TypeError(f"position must be a Point, got {type(self.position).__name__}")

        if not (math.isfinite(self.position.x) and math.isfinite(self.position.y)):
            raise ValueError(
                f"Pedestrian '{self.id}' coordinates must be finite: ({self.position.x}, {self.position.y})"
            )

        # 3. Velocity coercion & validation
        if isinstance(self.velocity, (tuple, list)):
            if len(self.velocity) != 2:
                raise ValueError(f"Velocity sequence must have length 2, got {self.velocity!r}")
            self.velocity = Velocity2D(self.velocity[0], self.velocity[1])
        elif not isinstance(self.velocity, Velocity2D):
            raise TypeError(f"velocity must be a Velocity2D, got {type(self.velocity).__name__}")

        # 4. Status coercion & validation
        if isinstance(self.status, str) and not isinstance(self.status, PedestrianStatus):
            try:
                self.status = PedestrianStatus(self.status)
            except ValueError as err:
                raise ValueError(
                    f"Invalid pedestrian status {self.status!r}. "
                    f"Must be one of {[s.value for s in PedestrianStatus]}"
                ) from err
        elif not isinstance(self.status, PedestrianStatus):
            raise TypeError(f"status must be a PedestrianStatus, got {type(self.status).__name__}")

        # 5. Optional references
        if self.target_exit_id is not None:
            if not isinstance(self.target_exit_id, str) or not self.target_exit_id:
                raise ValueError(f"target_exit_id must be a non-empty string if provided, got {self.target_exit_id!r}")

        if self.current_area_id is not None:
            if not isinstance(self.current_area_id, str) or not self.current_area_id:
                raise ValueError(f"current_area_id must be a non-empty string if provided, got {self.current_area_id!r}")

        # 6. Waypoints validation
        if not isinstance(self.waypoints, list):
            raise TypeError(f"waypoints must be a list, got {type(self.waypoints).__name__}")
        validated_waypoints: list[Point] = []
        for idx, wp in enumerate(self.waypoints):
            if isinstance(wp, (tuple, list)):
                if len(wp) != 2:
                    raise ValueError(f"Waypoint {idx} sequence must have length 2, got {wp!r}")
                wp = Point(wp[0], wp[1])
            elif not isinstance(wp, Point):
                raise TypeError(f"Waypoint {idx} must be a Point, got {type(wp).__name__}")
            if not (math.isfinite(wp.x) and math.isfinite(wp.y)):
                raise ValueError(f"Waypoint {idx} coordinates must be finite: ({wp.x}, {wp.y})")
            validated_waypoints.append(wp)
        self.waypoints = validated_waypoints

        # 7. Physical parameters validation
        try:
            r_f = float(self.radius)
        except (TypeError, ValueError) as err:
            raise TypeError(f"radius must be numeric, got {self.radius!r}") from err
        if not (math.isfinite(r_f) and r_f > 0.0):
            raise ValueError(f"Pedestrian radius must be strictly positive and finite, got {self.radius}")
        self.radius = r_f

        try:
            spd_f = float(self.preferred_speed)
        except (TypeError, ValueError) as err:
            raise TypeError(f"preferred_speed must be numeric, got {self.preferred_speed!r}") from err
        if not (math.isfinite(spd_f) and spd_f >= 0.0):
            raise ValueError(f"preferred_speed must be non-negative and finite, got {self.preferred_speed}")
        self.preferred_speed = spd_f

        # 8. History capacity validation
        if not isinstance(self.max_history_len, int) or isinstance(self.max_history_len, bool):
            raise TypeError(f"max_history_len must be an integer, got {type(self.max_history_len).__name__}")
        if self.max_history_len <= 0:
            raise ValueError(f"max_history_len must be strictly positive (> 0), got {self.max_history_len}")

        # 9. Trajectory history validation
        if not isinstance(self.trajectory_history, list):
            raise TypeError(f"trajectory_history must be a list, got {type(self.trajectory_history).__name__}")
        for pt in self.trajectory_history:
            if not isinstance(pt, TimestampedTrajectoryPoint):
                raise TypeError(
                    f"trajectory_history entries must be TimestampedTrajectoryPoint, got {type(pt).__name__}"
                )
        if len(self.trajectory_history) > self.max_history_len:
            self.trajectory_history = self.trajectory_history[-self.max_history_len :]

    def record_history_sample(self, step: int, timestamp: float) -> None:
        """Record current position as a timestamped trajectory sample.

        The step and timestamp are provided directly by the simulation clock.
        Enforces bounded FIFO capacity: drops oldest sample if history exceeds max_history_len.
        """
        sample = TimestampedTrajectoryPoint(
            step=step,
            timestamp=timestamp,
            x=self.position.x,
            y=self.position.y,
        )
        self.trajectory_history.append(sample)
        if len(self.trajectory_history) > self.max_history_len:
            self.trajectory_history.pop(0)

    def freeze_at_evacuation(self) -> None:
        """Transition pedestrian into permanently frozen evacuated status.

        Sets status to EVACUATED, freezes velocity to (0, 0), and clears waypoints.
        """
        self.status = PedestrianStatus.EVACUATED
        self.velocity = Velocity2D(0.0, 0.0)
        self.waypoints.clear()
