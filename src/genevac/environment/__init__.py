"""Environment module for GenEvac-Sentinel.

Defines geometry primitives, environment domain models, floor-plan builders,
and JSON serialization / deserialization routines.
"""

from genevac.environment.builder import create_initial_building
from genevac.environment.geometry import (
    LineSegment,
    Point,
    Rectangle,
    Wall,
    WallAlignment,
    WallType,
)
from genevac.environment.models import (
    Building,
    Doorway,
    Exit,
    Obstacle,
    WalkableArea,
)
from genevac.environment.serialization import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    building_from_dict,
    building_to_dict,
    load_building,
    save_building,
)

__all__ = [
    "Point",
    "Rectangle",
    "LineSegment",
    "Wall",
    "WallType",
    "WallAlignment",
    "WalkableArea",
    "Doorway",
    "Obstacle",
    "Exit",
    "Building",
    "create_initial_building",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "building_to_dict",
    "building_from_dict",
    "save_building",
    "load_building",
]
