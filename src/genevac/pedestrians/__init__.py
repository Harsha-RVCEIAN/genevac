"""Pedestrian state, attributes, and behavior models."""

from dataclasses import dataclass
from typing import Tuple
from enum import Enum


class PedestrianStatus(str, Enum):
    ACTIVE = "active"
    EVACUATED = "evacuated"
    INCAPACITATED = "incapacitated"


@dataclass
class Pedestrian:
    id: str
    position: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    desired_speed: float = 1.34
    radius: float = 0.3
    target_exit: str = ""
    status: PedestrianStatus = PedestrianStatus.ACTIVE

    def update_position(self, dt: float) -> None:
        if self.status != PedestrianStatus.ACTIVE:
            return
        new_x = self.position[0] + self.velocity[0] * dt
        new_y = self.position[1] + self.velocity[1] * dt
        self.position = (new_x, new_y)
