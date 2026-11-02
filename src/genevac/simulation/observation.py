"""Observation contract and immutable snapshot models for GenEvac-Sentinel.

Provides read-only observation data structures emitted by the simulation engine:
- PedestrianObservation: Immutable external view of an individual pedestrian
- SimulationObservationSnapshot: System-wide observation snapshot at observation ticks

Note on trajectory history capacity:
A default max_history_len of 100 samples recorded at the 2.5 Hz observation interval
(0.4 s) represents approximately 40.0 seconds of rolling trajectory history, not
the pedestrian's lifetime trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from genevac.pedestrian.models import (
    PedestrianState,
    PedestrianStatus,
    TimestampedTrajectoryPoint,
)
from genevac.pedestrian.registry import PedestrianRegistry


@dataclass(frozen=True)
class PedestrianObservation:
    """Immutable external observation of an individual pedestrian.

    Retained in all subsequent snapshots even after status becomes EVACUATED.
    Maintains high numerical precision for downstream evaluation.
    """

    pedestrian_id: str
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    status: str
    target_exit_id: Optional[str]
    trajectory_history: tuple[TimestampedTrajectoryPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize pedestrian observation to a JSON-compatible dictionary."""
        return {
            "pedestrian_id": self.pedestrian_id,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "speed": self.speed,
            "status": self.status,
            "target_exit_id": self.target_exit_id,
            "trajectory_history": [
                {
                    "step": pt.step,
                    "timestamp": pt.timestamp,
                    "x": pt.x,
                    "y": pt.y,
                }
                for pt in self.trajectory_history
            ],
        }


@dataclass(frozen=True)
class SimulationObservationSnapshot:
    """Immutable, authoritative snapshot of the simulation at an observation tick.

    Guarantees:
    - Counts of all four statuses add up to total_count.
    - Snapshot contents are decoupled from live simulation state mutations.
    - Independent of Pygame, PyTorch, MID, and NumPy.
    """

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
    pedestrians: tuple[PedestrianObservation, ...]

    def __post_init__(self) -> None:
        expected_total = (
            self.active_count + self.evacuated_count + self.blocked_count + self.inactive_count
        )
        if self.total_count != expected_total:
            raise ValueError(
                f"total_count ({self.total_count}) must equal sum of status counts "
                f"({self.active_count} active + {self.evacuated_count} evacuated + "
                f"{self.blocked_count} blocked + {self.inactive_count} inactive = {expected_total})"
            )
        if len(self.pedestrians) != self.total_count:
            raise ValueError(
                f"Number of pedestrian observations ({len(self.pedestrians)}) does not match "
                f"total_count ({self.total_count})"
            )

    @property
    def simulation_step(self) -> int:
        """Alias for step_count."""
        return self.step_count

    def to_dict(self) -> dict[str, Any]:
        """Serialize simulation snapshot to a JSON-compatible dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "step_count": self.step_count,
            "simulation_step": self.step_count,
            "timestamp": self.timestamp,
            "dt": self.dt,
            "observation_interval": self.observation_interval,
            "total_count": self.total_count,
            "active_count": self.active_count,
            "evacuated_count": self.evacuated_count,
            "blocked_count": self.blocked_count,
            "inactive_count": self.inactive_count,
            "pedestrians": [p.to_dict() for p in self.pedestrians],
        }


def create_observation_snapshot(
    registry: PedestrianRegistry,
    step_count: int,
    timestamp: float,
    dt: float,
    observation_interval: float,
    snapshot_prefix: str = "snap",
) -> SimulationObservationSnapshot:
    """Create an immutable snapshot from the current state of a pedestrian registry."""
    ped_obs_list: list[PedestrianObservation] = []

    for ped in registry:
        obs = PedestrianObservation(
            pedestrian_id=ped.id,
            x=ped.position.x,
            y=ped.position.y,
            vx=ped.velocity.vx,
            vy=ped.velocity.vy,
            speed=ped.velocity.speed,
            status=ped.status.value,
            target_exit_id=ped.target_exit_id,
            # Immutable copy of trajectory history
            trajectory_history=tuple(ped.trajectory_history),
        )
        ped_obs_list.append(obs)

    snapshot_id = f"{snapshot_prefix}_{step_count:06d}"

    return SimulationObservationSnapshot(
        snapshot_id=snapshot_id,
        step_count=step_count,
        timestamp=timestamp,
        dt=dt,
        observation_interval=observation_interval,
        total_count=registry.total_count,
        active_count=registry.active_count,
        evacuated_count=registry.evacuated_count,
        blocked_count=registry.blocked_count,
        inactive_count=registry.inactive_count,
        pedestrians=tuple(ped_obs_list),
    )
