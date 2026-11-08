"""Simulation package for GenEvac-Sentinel.

Provides:
- SimulationClock: Authoritative deterministic clock
- SimulationEngine: Authoritative engine coordinating kinematics and observations
- Waypoint and route generation functions
- Observation snapshot contracts
"""

from genevac.simulation.clock import SimulationClock
from genevac.simulation.engine import SimulationEngine
from genevac.simulation.events import SimulationEvent, SimulationEventType
from genevac.simulation.exceptions import NoRouteError
from genevac.simulation.movement import (
    advance_pedestrian_step,
    advance_population_step,
    check_collision,
    check_pedestrian_collision,
    check_pedestrian_overlap,
    continuous_pedestrian_distance,
    detect_exit_crossing,
    distance_point_to_rect,
    find_walkable_area_for_point,
    generate_deterministic_route,
    is_point_clear_of_obstacles,
    is_position_in_walkable_space,
)
from genevac.simulation.observation import (
    PedestrianObservation,
    SimulationObservationSnapshot,
    create_observation_snapshot,
)

__all__ = [
    "SimulationClock",
    "SimulationEngine",
    "SimulationEvent",
    "SimulationEventType",
    "NoRouteError",
    "generate_deterministic_route",
    "advance_pedestrian_step",
    "advance_population_step",
    "check_collision",
    "check_pedestrian_collision",
    "check_pedestrian_overlap",
    "continuous_pedestrian_distance",
    "detect_exit_crossing",
    "distance_point_to_rect",
    "find_walkable_area_for_point",
    "is_point_clear_of_obstacles",
    "is_position_in_walkable_space",
    "PedestrianObservation",
    "SimulationObservationSnapshot",
    "create_observation_snapshot",
]
