"""Tests for GenEvac-Sentinel Phase 3 Environment JSON Serialization and Deserialization."""

import json
from pathlib import Path
import pytest

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
    building_from_dict,
    building_to_dict,
    load_building,
    save_building,
)


# ---------------------------------------------------------------------------
# Round-trip Serialization & Deserialization Tests
# ---------------------------------------------------------------------------

def test_serialize_initial_building_to_dict():
    """Verify serialization of the initial building yields expected top-level schema and counts."""
    building = create_initial_building()
    data = building_to_dict(building)

    assert data["schema_version"] == SCHEMA_VERSION
    assert data["id"] == "building_main"
    assert data["width"] == 40.0
    assert data["height"] == 30.0

    assert len(data["walkable_areas"]) == 5
    assert len(data["doorways"]) == 4
    assert len(data["exits"]) == 3
    assert len(data["walls"]) == 17
    assert len(data["obstacles"]) == 4


def test_round_trip_dict_preserves_all_attributes():
    """Verify that serializing to dict and deserializing reconstructs an identical, validated Building."""
    original = create_initial_building()
    data = building_to_dict(original)
    reconstructed = building_from_dict(data)

    # Building level
    assert reconstructed.id == original.id
    assert reconstructed.width == pytest.approx(original.width)
    assert reconstructed.height == pytest.approx(original.height)

    # Walkable Areas
    assert len(reconstructed.walkable_areas) == len(original.walkable_areas)
    for orig_area in original.walkable_areas:
        recon_area = reconstructed.get_walkable_area(orig_area.id)
        assert recon_area is not None
        assert recon_area.bounds.x == pytest.approx(orig_area.bounds.x)
        assert recon_area.bounds.y == pytest.approx(orig_area.bounds.y)
        assert recon_area.bounds.width == pytest.approx(orig_area.bounds.width)
        assert recon_area.bounds.height == pytest.approx(orig_area.bounds.height)
        assert recon_area.connected_area_ids == orig_area.connected_area_ids
        assert recon_area.area_type == orig_area.area_type

    # Doorways
    assert len(reconstructed.doorways) == len(original.doorways)
    for orig_door in original.doorways:
        recon_door = reconstructed.get_doorway(orig_door.id)
        assert recon_door is not None
        assert recon_door.connected_area_ids == orig_door.connected_area_ids
        assert recon_door.width == pytest.approx(orig_door.width)
        assert recon_door.segment.start.x == pytest.approx(orig_door.segment.start.x)
        assert recon_door.segment.start.y == pytest.approx(orig_door.segment.start.y)
        assert recon_door.segment.end.x == pytest.approx(orig_door.segment.end.x)
        assert recon_door.segment.end.y == pytest.approx(orig_door.segment.end.y)

    # Exits
    assert len(reconstructed.exits) == len(original.exits)
    for orig_exit in original.exits:
        recon_exit = reconstructed.get_exit(orig_exit.id)
        assert recon_exit is not None
        assert recon_exit.width == pytest.approx(orig_exit.width)
        assert recon_exit.boundary == orig_exit.boundary
        assert recon_exit.connected_area_id == orig_exit.connected_area_id
        assert recon_exit.segment.start.x == pytest.approx(orig_exit.segment.start.x)
        assert recon_exit.segment.start.y == pytest.approx(orig_exit.segment.start.y)
        assert recon_exit.segment.end.x == pytest.approx(orig_exit.segment.end.x)
        assert recon_exit.segment.end.y == pytest.approx(orig_exit.segment.end.y)

    # Walls & Enums
    assert len(reconstructed.walls) == len(original.walls)
    for orig_wall in original.walls:
        recon_wall = reconstructed.get_wall(orig_wall.id)
        assert recon_wall is not None
        assert recon_wall.thickness == pytest.approx(orig_wall.thickness)
        assert isinstance(recon_wall.wall_type, WallType)
        assert recon_wall.wall_type == orig_wall.wall_type
        assert isinstance(recon_wall.alignment, WallAlignment)
        assert recon_wall.alignment == orig_wall.alignment
        assert recon_wall.start.x == pytest.approx(orig_wall.start.x)
        assert recon_wall.start.y == pytest.approx(orig_wall.start.y)
        assert recon_wall.end.x == pytest.approx(orig_wall.end.x)
        assert recon_wall.end.y == pytest.approx(orig_wall.end.y)

    # Obstacles
    assert len(reconstructed.obstacles) == len(original.obstacles)
    for orig_obs in original.obstacles:
        recon_obs = reconstructed.get_obstacle(orig_obs.id)
        assert recon_obs is not None
        assert recon_obs.bounds.x == pytest.approx(orig_obs.bounds.x)
        assert recon_obs.bounds.y == pytest.approx(orig_obs.bounds.y)
        assert recon_obs.bounds.width == pytest.approx(orig_obs.bounds.width)
        assert recon_obs.bounds.height == pytest.approx(orig_obs.bounds.height)


def test_save_and_load_file_round_trip(tmp_path: Path):
    """Verify saving to and loading from a JSON file preserves the entire environment."""
    file_path = tmp_path / "building_test.json"
    original = create_initial_building()

    save_building(original, file_path)
    assert file_path.is_file()

    # Verify JSON content is valid and readable
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["id"] == "building_main"

    loaded = load_building(file_path)
    assert loaded.id == original.id
    assert loaded.width == pytest.approx(original.width)
    assert len(loaded.walkable_areas) == len(original.walkable_areas)
    assert len(loaded.walls) == len(original.walls)


def test_arbitrary_custom_building_serialization(tmp_path: Path):
    """Verify that serialization works with an arbitrary custom valid building."""
    custom_areas = [
        WalkableArea(
            id="room_west",
            bounds=Rectangle(0.0, 0.0, 10.0, 10.0),
            connected_area_ids=["room_east"],
        ),
        WalkableArea(
            id="room_east",
            bounds=Rectangle(10.0, 0.0, 10.0, 10.0),
            connected_area_ids=["room_west"],
        ),
    ]
    custom_doors = [
        Doorway(
            id="door_w_e",
            connected_area_ids=("room_west", "room_east"),
            segment=LineSegment(Point(10.0, 4.0), Point(10.0, 6.0)),
            width=2.0,
        )
    ]
    custom_exits = [
        Exit(
            id="exit_w",
            segment=LineSegment(Point(0.0, 4.0), Point(0.0, 6.0)),
            width=2.0,
            boundary="left",
            connected_area_id="room_west",
        )
    ]
    custom_walls = [
        Wall(
            id="w_top",
            start=Point(0.0, 0.0),
            end=Point(20.0, 0.0),
            thickness=0.2,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.POSITIVE,
        ),
        Wall(
            id="w_bot",
            start=Point(0.0, 10.0),
            end=Point(20.0, 10.0),
            thickness=0.2,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.NEGATIVE,
        ),
        Wall(
            id="w_right",
            start=Point(20.0, 0.0),
            end=Point(20.0, 10.0),
            thickness=0.2,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.NEGATIVE,
        ),
        Wall(
            id="w_left_top",
            start=Point(0.0, 0.0),
            end=Point(0.0, 4.0),
            thickness=0.2,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.POSITIVE,
        ),
        Wall(
            id="w_left_bot",
            start=Point(0.0, 6.0),
            end=Point(0.0, 10.0),
            thickness=0.2,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.POSITIVE,
        ),
        Wall(
            id="w_mid_top",
            start=Point(10.0, 0.0),
            end=Point(10.0, 4.0),
            thickness=0.2,
            wall_type=WallType.INTERIOR,
            alignment=WallAlignment.CENTER,
        ),
        Wall(
            id="w_mid_bot",
            start=Point(10.0, 6.0),
            end=Point(10.0, 10.0),
            thickness=0.2,
            wall_type=WallType.INTERIOR,
            alignment=WallAlignment.CENTER,
        ),
    ]

    custom_b = Building(
        id="custom_b",
        width=20.0,
        height=10.0,
        walkable_areas=custom_areas,
        walls=custom_walls,
        doorways=custom_doors,
        exits=custom_exits,
    )

    file_path = tmp_path / "custom.json"
    save_building(custom_b, file_path)
    loaded_b = load_building(file_path)

    assert loaded_b.id == "custom_b"
    assert loaded_b.width == 20.0
    assert len(loaded_b.walkable_areas) == 2
    assert len(loaded_b.doorways) == 1
    assert len(loaded_b.exits) == 1


# ---------------------------------------------------------------------------
# Error Handling & Validation Tests
# ---------------------------------------------------------------------------

def test_load_missing_file_raises():
    """Verify load_building raises FileNotFoundError when file does not exist."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_building("non_existent_building_file.json")


def test_load_invalid_json_syntax_raises(tmp_path: Path):
    """Verify load_building raises JSONDecodeError when reading malformed JSON."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{malformed: json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_building(bad_json)


def test_save_building_type_error():
    """Verify save_building rejects non-Building objects."""
    with pytest.raises(TypeError, match="Expected Building"):
        save_building("not_a_building", "path.json")  # type: ignore


def test_building_from_dict_type_error():
    """Verify building_from_dict rejects non-dict inputs."""
    with pytest.raises(TypeError, match="Expected dict"):
        building_from_dict(["not", "a", "dict"])


@pytest.mark.parametrize(
    "invalid_schema_ver,match_str",
    [
        (None, "Missing required field 'schema_version'"),
        ("2.0", "Unsupported schema version"),
        ("0.9", "Unsupported schema version"),
        ("alpha", "Unsupported schema version"),
    ],
)
def test_reject_unsupported_or_missing_schema_version(invalid_schema_ver, match_str):
    """Verify deserialization rejects missing or unsupported schema versions."""
    data = building_to_dict(create_initial_building())
    if invalid_schema_ver is None:
        del data["schema_version"]
    else:
        data["schema_version"] = invalid_schema_ver

    with pytest.raises(ValueError, match=match_str):
        building_from_dict(data)


@pytest.mark.parametrize("missing_field", ["id", "width", "height"])
def test_reject_missing_required_building_fields(missing_field):
    """Verify deserialization rejects missing top-level building fields."""
    data = building_to_dict(create_initial_building())
    del data[missing_field]

    with pytest.raises(ValueError, match=f"Missing required building field: '{missing_field}'"):
        building_from_dict(data)


def test_reject_malformed_geometry_coordinates():
    """Verify deserialization rejects non-numeric geometry coordinates."""
    data = building_to_dict(create_initial_building())
    data["walls"][0]["start"]["x"] = "invalid_x_coordinate"

    with pytest.raises(ValueError, match="must be numeric"):
        building_from_dict(data)


def test_reject_negative_rectangle_dimensions():
    """Verify deserialization rejects invalid rectangle bounds."""
    data = building_to_dict(create_initial_building())
    data["walkable_areas"][0]["bounds"]["width"] = -10.0

    with pytest.raises(ValueError, match="width must be strictly positive"):
        building_from_dict(data)


def test_reject_invalid_wall_type_enum():
    """Verify deserialization rejects unrecognized wall type strings."""
    data = building_to_dict(create_initial_building())
    data["walls"][0]["wall_type"] = "reinforced_glass"

    with pytest.raises(ValueError, match="Invalid wall_type.*reinforced_glass"):
        building_from_dict(data)


def test_reject_invalid_wall_alignment_enum():
    """Verify deserialization rejects unrecognized wall alignment strings."""
    data = building_to_dict(create_initial_building())
    data["walls"][0]["alignment"] = "diagonal"

    with pytest.raises(ValueError, match="Invalid alignment.*diagonal"):
        building_from_dict(data)


def test_reject_invalid_connectivity_reference():
    """Verify deserialization rejects area connectivity pointing to non-existent areas."""
    data = building_to_dict(create_initial_building())
    data["walkable_areas"][0]["connected_area_ids"].append("phantom_zone")

    with pytest.raises(ValueError, match="references non-existent connected area"):
        building_from_dict(data)


def test_reject_non_reciprocal_connectivity():
    """Verify deserialization rejects non-reciprocal connectivity."""
    data = building_to_dict(create_initial_building())
    # Remove room_1 from main_corridor's connected_area_ids
    corridor_data = next(a for a in data["walkable_areas"] if a["id"] == "main_corridor")
    corridor_data["connected_area_ids"].remove("room_1")

    with pytest.raises(ValueError, match="is not reciprocal"):
        building_from_dict(data)


def test_reject_doorway_with_mismatched_width():
    """Verify deserialization rejects doorway where segment length != width."""
    data = building_to_dict(create_initial_building())
    data["doorways"][0]["width"] = 5.0  # Segment length is 2.0m

    with pytest.raises(ValueError, match="does not match declared width"):
        building_from_dict(data)


def test_reject_exit_on_wrong_boundary():
    """Verify deserialization rejects an exit with coordinates mismatched to its boundary."""
    data = building_to_dict(create_initial_building())
    # exit_left is declared on 'left' (x=0), but we move x to 5.0
    data["exits"][0]["segment"]["start"]["x"] = 5.0
    data["exits"][0]["segment"]["end"]["x"] = 5.0

    with pytest.raises(ValueError, match="specified on 'left' boundary but x != 0.0"):
        building_from_dict(data)


def test_building_validation_invoked_on_deserialized_wall_blocking_exit():
    """Verify that Building.validate() is invoked and catches a solid wall placed over an exit."""
    data = building_to_dict(create_initial_building())
    # Add a wall covering exit_left
    covering_wall = {
        "id": "wall_cover_exit",
        "start": {"x": 0.0, "y": 14.0},
        "end": {"x": 0.0, "y": 16.0},
        "thickness": 0.2,
        "wall_type": "exterior",
        "alignment": "positive",
    }
    data["walls"].append(covering_wall)

    with pytest.raises(ValueError, match="opening is covered by solid wall"):
        building_from_dict(data)
