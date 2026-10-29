"""Simulation loop, state updates, and orchestration."""

from typing import List, Dict, Any
from genevac.environment import EnvironmentMap
from genevac.pedestrians import Pedestrian, PedestrianStatus


class SimulationEngine:
    def __init__(self, environment: EnvironmentMap, max_steps: int = 1000, timestep: float = 0.1, seed: int = 42):
        self.environment = environment
        self.max_steps = max_steps
        self.timestep = timestep
        self.seed = seed
        self.pedestrians: List[Pedestrian] = []
        self.current_step = 0

    def add_pedestrian(self, ped: Pedestrian) -> None:
        self.pedestrians.append(ped)

    def step(self) -> None:
        self.current_step += 1
        for ped in self.pedestrians:
            ped.update_position(self.timestep)

    def is_finished(self) -> bool:
        if self.current_step >= self.max_steps:
            return True
        active_peds = [p for p in self.pedestrians if p.status == PedestrianStatus.ACTIVE]
        return len(active_peds) == 0 and len(self.pedestrians) > 0

    def run(self) -> Dict[str, Any]:
        while not self.is_finished():
            self.step()
        return {
            "total_steps": self.current_step,
            "pedestrians_total": len(self.pedestrians),
            "evacuated": len([p for p in self.pedestrians if p.status == PedestrianStatus.EVACUATED]),
        }
