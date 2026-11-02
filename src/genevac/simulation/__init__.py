"""Simulation package for GenEvac-Sentinel.

Provides:
- SimulationClock: Authoritative deterministic clock
- SimulationEngine: Authoritative engine coordinating kinematics and observations
- Waypoint and route generation functions
- Observation snapshot contracts
"""

from genevac.simulation.clock import SimulationClock
from genevac.simulation.engine import SimulationEngine
from genevac.simulation.movement import (
    advance_pedestrian_step,
    check_collision,
    detect_exit_crossing,
    distance_point_to_rect,
    generate_deterministic_route,
)
from genevac.simulation.observation import (
    PedestrianObservation,
    SimulationObservationSnapshot,
    create_observation_snapshot,
)

__all__ = [
    "SimulationClock",
    "SimulationEngine",
    "generate_deterministic_route",
    "advance_pedestrian_step",
    "check_collision",
    "detect_exit_crossing",
    "distance_point_to_rect",
    "PedestrianObservation",
    "SimulationObservationSnapshot",
    "create_observation_snapshot",
]
