"""Environment, map, obstacles, and exits."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Exit:
    id: str
    position: Tuple[float, float]
    width: float = 2.0


@dataclass
class Obstacle:
    id: str
    bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)


class EnvironmentMap:
    def __init__(self, width: float = 50.0, height: float = 20.0):
        self.width = width
        self.height = height
        self.exits: List[Exit] = []
        self.obstacles: List[Obstacle] = []

    def add_exit(self, exit_obj: Exit) -> None:
        self.exits.append(exit_obj)

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)

    def is_walkable(self, x: float, y: float, radius: float = 0.0) -> bool:
        if x - radius < 0 or x + radius > self.width or y - radius < 0 or y + radius > self.height:
            return False
        for obs in self.obstacles:
            min_x, min_y, max_x, max_y = obs.bounds
            if (min_x - radius <= x <= max_x + radius) and (min_y - radius <= y <= max_y + radius):
                return False
        return True
