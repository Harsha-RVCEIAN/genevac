"""Pedestrian package for GenEvac-Sentinel.

Provides domain models and a population registry for simulated pedestrians.
"""

from genevac.pedestrian.models import (
    PedestrianState,
    PedestrianStatus,
    TimestampedTrajectoryPoint,
    Velocity2D,
)
from genevac.pedestrian.registry import PedestrianRegistry

__all__ = [
    "PedestrianStatus",
    "Velocity2D",
    "TimestampedTrajectoryPoint",
    "PedestrianState",
    "PedestrianRegistry",
]
