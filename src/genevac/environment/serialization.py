"""JSON serialization and deserialization for the GenEvac-Sentinel environment model.

Provides functions to serialize a Building model to a JSON-compatible dictionary or file,
and to deserialize, validate, and reconstruct the Building instance using the existing
Phase 2 domain models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

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

SCHEMA_VERSION: str = "1.0"
SUPPORTED_SCHEMA_VERSIONS: set[str] = {"1.0"}


def point_to_dict(point: Point) -> dict[str, float]:
    """Serialize a Point to a dictionary."""
    return {"x": point.x, "y": point.y}


def point_from_dict(data: Any, context: str = "point") -> Point:
    """Deserialize a Point from a dictionary."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for {context}, got {type(data).__name__}")
    if "x" not in data or "y" not in data:
        raise ValueError(f"Missing required coordinate 'x' or 'y' in {context}: {data!r}")
    try:
        x = float(data["x"])
        y = float(data["y"])
    except (ValueError, TypeError) as err:
        raise ValueError(f"Coordinates in {context} must be numeric: {data!r}") from err
    return Point(x=x, y=y)


def rectangle_to_dict(rect: Rectangle) -> dict[str, float]:
    """Serialize a Rectangle to a dictionary."""
    return {
        "x": rect.x,
        "y": rect.y,
        "width": rect.width,
        "height": rect.height,
    }


def rectangle_from_dict(data: Any, context: str = "rectangle") -> Rectangle:
    """Deserialize a Rectangle from a dictionary."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for {context}, got {type(data).__name__}")
    for key in ("x", "y", "width", "height"):
        if key not in data:
            raise ValueError(f"Missing required field '{key}' in {context}: {data!r}")
    try:
        x = float(data["x"])
        y = float(data["y"])
        width = float(data["width"])
        height = float(data["height"])
    except (ValueError, TypeError) as err:
        raise ValueError(f"Dimensions and coordinates in {context} must be numeric: {data!r}") from err
    return Rectangle(x=x, y=y, width=width, height=height)


def line_segment_to_dict(segment: LineSegment) -> dict[str, Any]:
    """Serialize a LineSegment to a dictionary."""
    return {
        "start": point_to_dict(segment.start),
        "end": point_to_dict(segment.end),
    }


def line_segment_from_dict(data: Any, context: str = "segment") -> LineSegment:
    """Deserialize a LineSegment from a dictionary."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for {context}, got {type(data).__name__}")
    if "start" not in data or "end" not in data:
        raise ValueError(f"Missing 'start' or 'end' in {context}: {data!r}")
    start = point_from_dict(data["start"], context=f"{context}.start")
    end = point_from_dict(data["end"], context=f"{context}.end")
    return LineSegment(start=start, end=end)


def wall_to_dict(wall: Wall) -> dict[str, Any]:
    """Serialize a Wall to a dictionary."""
    return {
        "id": wall.id,
        "start": point_to_dict(wall.start),
        "end": point_to_dict(wall.end),
        "thickness": wall.thickness,
        "wall_type": wall.wall_type.value,
        "alignment": wall.alignment.value,
    }


def wall_from_dict(data: Any) -> Wall:
    """Deserialize a Wall from a dictionary."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for wall, got {type(data).__name__}")
    if "id" not in data:
        raise ValueError(f"Missing required field 'id' in wall: {data!r}")
    wall_id = str(data["id"])
    if not wall_id:
        raise ValueError(f"Wall ID must not be empty: {data!r}")

    if "start" not in data or "end" not in data:
        raise ValueError(f"Wall '{wall_id}' missing 'start' or 'end' coordinates")

    start = point_from_dict(data["start"], context=f"wall '{wall_id}'.start")
    end = point_from_dict(data["end"], context=f"wall '{wall_id}'.end")

    thickness = float(data.get("thickness", 0.2))

    wall_type_raw = data.get("wall_type", WallType.INTERIOR.value)
    try:
        wall_type = WallType(wall_type_raw)
    except ValueError as err:
        raise ValueError(
            f"Invalid wall_type {wall_type_raw!r} for wall '{wall_id}'. Must be 'exterior' or 'interior'"
        ) from err

    alignment_raw = data.get("alignment", WallAlignment.CENTER.value)
    try:
        alignment = WallAlignment(alignment_raw)
    except ValueError as err:
        raise ValueError(
            f"Invalid alignment {alignment_raw!r} for wall '{wall_id}'. Must be 'center', 'positive', or 'negative'"
        ) from err

    return Wall(
        id=wall_id,
        start=start,
        end=end,
        thickness=thickness,
        wall_type=wall_type,
        alignment=alignment,
    )


def walkable_area_to_dict(area: WalkableArea) -> dict[str, Any]:
    """Serialize a WalkableArea to a dictionary."""
    return {
        "id": area.id,
        "bounds": rectangle_to_dict(area.bounds),
        "connected_area_ids": list(area.connected_area_ids),
        "area_type": area.area_type,
    }


def walkable_area_from_dict(data: Any) -> WalkableArea:
    """Deserialize a WalkableArea from a dictionary."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for walkable_area, got {type(data).__name__}")
    if "id" not in data:
        raise ValueError(f"Missing required field 'id' in walkable area: {data!r}")
    area_id = str(data["id"])
    if not area_id:
        raise ValueError(f"Walkable area ID must not be empty: {data!r}")

    if "bounds" not in data:
        raise ValueError(f"Walkable area '{area_id}' missing 'bounds'")
    bounds = rectangle_from_dict(data["bounds"], context=f"walkable_area '{area_id}'.bounds")

    connected_ids = data.get("connected_area_ids", [])
    if not isinstance(connected_ids, list):
        raise TypeError(f"connected_area_ids in walkable area '{area_id}' must be a list")

    area_type = str(data.get("area_type", "room"))

    return WalkableArea(
        id=area_id,
        bounds=bounds,
        connected_area_ids=[str(cid) for cid in connected_ids],
        area_type=area_type,
    )


def doorway_to_dict(door: Doorway) -> dict[str, Any]:
    """Serialize a Doorway to a dictionary."""
    return {
        "id": door.id,
        "connected_area_ids": list(door.connected_area_ids),
        "segment": line_segment_to_dict(door.segment),
        "width": door.width,
    }


def doorway_from_dict(data: Any) -> Doorway:
    """Deserialize a Doorway from a dictionary."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for doorway, got {type(data).__name__}")
    if "id" not in data:
        raise ValueError(f"Missing required field 'id' in doorway: {data!r}")
    door_id = str(data["id"])

    if "connected_area_ids" not in data:
        raise ValueError(f"Doorway '{door_id}' missing 'connected_area_ids'")
    conn_ids = data["connected_area_ids"]
    if not isinstance(conn_ids, (list, tuple)) or len(conn_ids) != 2:
        raise ValueError(
            f"Doorway '{door_id}' connected_area_ids must contain exactly 2 area IDs, got {conn_ids!r}"
        )

    if "segment" not in data:
        raise ValueError(f"Doorway '{door_id}' missing 'segment'")
    segment = line_segment_from_dict(data["segment"], context=f"doorway '{door_id}'.segment")

    if "width" not in data:
        raise ValueError(f"Doorway '{door_id}' missing 'width'")
    try:
        width = float(data["width"])
    except (ValueError, TypeError) as err:
        raise ValueError(f"Doorway '{door_id}' width must be numeric: {data['width']!r}") from err

    return Doorway(
        id=door_id,
        connected_area_ids=(str(conn_ids[0]), str(conn_ids[1])),
        segment=segment,
        width=width,
    )


def obstacle_to_dict(obs: Obstacle) -> dict[str, Any]:
    """Serialize an Obstacle to a dictionary."""
    return {
        "id": obs.id,
        "bounds": rectangle_to_dict(obs.bounds),
    }


def obstacle_from_dict(data: Any) -> Obstacle:
    """Deserialize an Obstacle from a dictionary."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for obstacle, got {type(data).__name__}")
    if "id" not in data:
        raise ValueError(f"Missing required field 'id' in obstacle: {data!r}")
    obs_id = str(data["id"])
    if not obs_id:
        raise ValueError(f"Obstacle ID must not be empty: {data!r}")

    if "bounds" not in data:
        raise ValueError(f"Obstacle '{obs_id}' missing 'bounds'")
    bounds = rectangle_from_dict(data["bounds"], context=f"obstacle '{obs_id}'.bounds")

    return Obstacle(id=obs_id, bounds=bounds)


def exit_to_dict(ex: Exit) -> dict[str, Any]:
    """Serialize an Exit to a dictionary."""
    return {
        "id": ex.id,
        "segment": line_segment_to_dict(ex.segment),
        "width": ex.width,
        "boundary": ex.boundary,
        "connected_area_id": ex.connected_area_id,
        "is_open": ex.is_open,
    }


def exit_from_dict(data: Any) -> Exit:
    """Deserialize an Exit from a dictionary."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for exit, got {type(data).__name__}")
    if "id" not in data:
        raise ValueError(f"Missing required field 'id' in exit: {data!r}")
    exit_id = str(data["id"])
    if not exit_id:
        raise ValueError(f"Exit ID must not be empty: {data!r}")

    for key in ("segment", "width", "boundary", "connected_area_id"):
        if key not in data:
            raise ValueError(f"Exit '{exit_id}' missing required field '{key}'")

    segment = line_segment_from_dict(data["segment"], context=f"exit '{exit_id}'.segment")
    try:
        width = float(data["width"])
    except (ValueError, TypeError) as err:
        raise ValueError(f"Exit '{exit_id}' width must be numeric: {data['width']!r}") from err

    boundary = str(data["boundary"])
    connected_area_id = str(data["connected_area_id"])
    is_open = bool(data.get("is_open", True))

    return Exit(
        id=exit_id,
        segment=segment,
        width=width,
        boundary=boundary,
        connected_area_id=connected_area_id,
        is_open=is_open,
    )


def building_to_dict(building: Building) -> dict[str, Any]:
    """Serialize an entire Building instance to a JSON-compatible dictionary."""
    if not isinstance(building, Building):
        raise TypeError(f"Expected Building instance, got {type(building).__name__}")

    return {
        "schema_version": SCHEMA_VERSION,
        "id": building.id,
        "width": building.width,
        "height": building.height,
        "walkable_areas": [walkable_area_to_dict(a) for a in building.walkable_areas],
        "doorways": [doorway_to_dict(d) for d in building.doorways],
        "exits": [exit_to_dict(e) for e in building.exits],
        "walls": [wall_to_dict(w) for w in building.walls],
        "obstacles": [obstacle_to_dict(o) for o in building.obstacles],
    }


def building_from_dict(data: Any) -> Building:
    """Deserialize, validate, and reconstruct a Building instance from a dictionary.

    Enforces schema version checking, structure validation, and delegates comprehensive
    spatial and connectivity validation to Building.validate().
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict for building data, got {type(data).__name__}")

    # Check schema version
    if "schema_version" not in data:
        raise ValueError("Missing required field 'schema_version' in configuration")
    schema_ver = str(data["schema_version"])
    if schema_ver not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema version: {schema_ver!r}. "
            f"Supported versions: {sorted(list(SUPPORTED_SCHEMA_VERSIONS))}"
        )

    # Support nested "building" key if present
    b_data = data["building"] if "building" in data and isinstance(data["building"], dict) else data

    # Validate building top-level fields
    for field_name in ("id", "width", "height"):
        if field_name not in b_data:
            raise ValueError(f"Missing required building field: '{field_name}'")

    b_id = str(b_data["id"])
    if not b_id:
        raise ValueError("Building ID must not be empty")

    try:
        width = float(b_data["width"])
        height = float(b_data["height"])
    except (ValueError, TypeError) as err:
        raise ValueError("Building width and height must be numeric") from err

    # Deserialize collections
    walkable_areas_data = b_data.get("walkable_areas", [])
    if not isinstance(walkable_areas_data, list):
        raise TypeError("Field 'walkable_areas' must be a list")
    walkable_areas = [walkable_area_from_dict(a) for a in walkable_areas_data]

    doorways_data = b_data.get("doorways", [])
    if not isinstance(doorways_data, list):
        raise TypeError("Field 'doorways' must be a list")
    doorways = [doorway_from_dict(d) for d in doorways_data]

    exits_data = b_data.get("exits", [])
    if not isinstance(exits_data, list):
        raise TypeError("Field 'exits' must be a list")
    exits = [exit_from_dict(e) for e in exits_data]

    walls_data = b_data.get("walls", [])
    if not isinstance(walls_data, list):
        raise TypeError("Field 'walls' must be a list")
    walls = [wall_from_dict(w) for w in walls_data]

    obstacles_data = b_data.get("obstacles", [])
    if not isinstance(obstacles_data, list):
        raise TypeError("Field 'obstacles' must be a list")
    obstacles = [obstacle_from_dict(o) for o in obstacles_data]

    # Construct and validate Building instance
    # Note: Building.__post_init__() calls self.validate() automatically.
    return Building(
        id=b_id,
        width=width,
        height=height,
        walkable_areas=walkable_areas,
        walls=walls,
        obstacles=obstacles,
        exits=exits,
        doorways=doorways,
    )


def save_building(building: Building, path: Union[str, Path]) -> None:
    """Serialize a Building instance and save it to a formatted JSON file."""
    if not isinstance(building, Building):
        raise TypeError(f"Expected Building instance, got {type(building).__name__}")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = building_to_dict(building)

    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_building(path: Union[str, Path]) -> Building:
    """Load and reconstruct a validated Building instance from a JSON file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Environment configuration file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    return building_from_dict(data)
