"""Tests for GenEvac-Sentinel Phase 2 Environment Data Model and Architecture Audit."""

import sys
import pytest

from genevac.environment.builder import create_initial_building
from genevac.environment.geometry import (
    LineSegment,
    Point,
    Rectangle,
    Wall,
    WallAlignment,
    WallType,
    segment_intersects_rect,
    segments_overlap,
)
from genevac.environment.models import (
    Building,
    Doorway,
    Exit,
    Obstacle,
    WalkableArea,
)


# ---------------------------------------------------------------------------
# Geometry Primitives Tests
# ---------------------------------------------------------------------------

def test_point_construction():
    """Verify point construction, float conversion, distance, and tuple representation."""
    p1 = Point(0, 0)
    p2 = Point(3.0, 4.0)

    assert p1.x == 0.0
    assert p1.y == 0.0
    assert p2.x == 3.0
    assert p2.y == 4.0
    assert p1.distance_to(p2) == pytest.approx(5.0)
    assert p1.as_tuple() == (0.0, 0.0)


def test_rectangle_construction_and_properties():
    """Verify rectangle coordinates, properties, and point/rect containment."""
    rect = Rectangle(x=5.0, y=10.0, width=20.0, height=15.0)

    assert rect.left == 5.0
    assert rect.top == 10.0
    assert rect.right == 25.0
    assert rect.bottom == 25.0

    # Containment
    assert rect.contains_point(Point(5.0, 10.0))
    assert rect.contains_point(Point(15.0, 15.0))
    assert rect.contains_point(Point(25.0, 25.0))
    assert not rect.contains_point(Point(4.9, 15.0))
    assert not rect.contains_point(Point(15.0, 25.1))

    # Inner rectangle containment
    inner = Rectangle(x=6.0, y=11.0, width=5.0, height=5.0)
    assert rect.contains_rect(inner)

    # Overlapping and disjoint rectangles
    overlapping = Rectangle(x=20.0, y=20.0, width=10.0, height=10.0)
    assert rect.intersects(overlapping)
    disjoint = Rectangle(x=30.0, y=30.0, width=5.0, height=5.0)
    assert not rect.intersects(disjoint)


@pytest.mark.parametrize(
    "width,height,expected_err",
    [
        (-1.0, 10.0, "width must be strictly positive"),
        (0.0, 10.0, "width must be strictly positive"),
        (10.0, -2.0, "height must be strictly positive"),
        (10.0, 0.0, "height must be strictly positive"),
    ],
)
def test_rectangle_invalid_dimensions(width, height, expected_err):
    """Verify rejection of zero and negative rectangle width/height."""
    with pytest.raises(ValueError, match=expected_err):
        Rectangle(x=0.0, y=0.0, width=width, height=height)


def test_line_segment_construction():
    """Verify line segment endpoints and length calculation."""
    seg = LineSegment(start=Point(0.0, 0.0), end=Point(6.0, 8.0))
    assert seg.length == pytest.approx(10.0)
    assert seg.as_tuple() == ((0.0, 0.0), (6.0, 8.0))

    with pytest.raises(TypeError):
        LineSegment(start=(0, 0), end=Point(1, 1))  # type: ignore


# ---------------------------------------------------------------------------
# Wall Construction, Thickness & Footprint Tests
# ---------------------------------------------------------------------------

def test_wall_construction_and_thickness():
    """Verify wall properties, line segment conversion, and default thickness."""
    wall = Wall(
        id="wall_test",
        start=Point(0.0, 0.0),
        end=Point(10.0, 0.0),
        thickness=0.2,
        wall_type=WallType.EXTERIOR,
    )
    assert wall.id == "wall_test"
    assert wall.thickness == 0.2
    assert wall.wall_type == WallType.EXTERIOR
    assert wall.alignment == WallAlignment.CENTER
    assert wall.length == pytest.approx(10.0)
    assert wall.segment.start == Point(0.0, 0.0)
    assert wall.segment.end == Point(10.0, 0.0)


@pytest.mark.parametrize("invalid_thickness", [0.0, -0.2, -5.0])
def test_wall_invalid_thickness(invalid_thickness):
    """Verify rejection of non-positive wall thickness."""
    with pytest.raises(ValueError, match="Wall thickness must be strictly positive"):
        Wall(
            id="wall_bad",
            start=Point(0.0, 0.0),
            end=Point(1.0, 0.0),
            thickness=invalid_thickness,
        )


def test_wall_invalid_type():
    """Verify rejection of unrecognized wall classifications."""
    with pytest.raises(ValueError, match="Invalid wall type"):
        Wall(
            id="wall_bad_type",
            start=Point(0.0, 0.0),
            end=Point(1.0, 0.0),
            thickness=0.2,
            wall_type="floating",  # type: ignore
        )


def test_wall_footprint_alignments():
    """Verify wall physical footprint bounds under center, positive, and negative alignments."""
    # Horizontal wall at y=0, length 10
    w_pos = Wall(
        id="w_pos",
        start=Point(0.0, 0.0),
        end=Point(10.0, 0.0),
        thickness=0.2,
        alignment=WallAlignment.POSITIVE,
    )
    fp_pos = w_pos.footprint
    assert fp_pos.top == pytest.approx(0.0)
    assert fp_pos.bottom == pytest.approx(0.2)
    assert fp_pos.left == pytest.approx(0.0)
    assert fp_pos.right == pytest.approx(10.0)

    # Horizontal wall at y=30, negative alignment
    w_neg = Wall(
        id="w_neg",
        start=Point(0.0, 30.0),
        end=Point(10.0, 30.0),
        thickness=0.2,
        alignment=WallAlignment.NEGATIVE,
    )
    fp_neg = w_neg.footprint
    assert fp_neg.top == pytest.approx(29.8)
    assert fp_neg.bottom == pytest.approx(30.0)

    # Vertical wall at x=0, positive alignment
    w_vert_pos = Wall(
        id="w_vpos",
        start=Point(0.0, 0.0),
        end=Point(0.0, 10.0),
        thickness=0.2,
        alignment=WallAlignment.POSITIVE,
    )
    fp_vpos = w_vert_pos.footprint
    assert fp_vpos.left == pytest.approx(0.0)
    assert fp_vpos.right == pytest.approx(0.2)

    # Vertical wall at x=40, negative alignment
    w_vert_neg = Wall(
        id="w_vneg",
        start=Point(40.0, 0.0),
        end=Point(40.0, 10.0),
        thickness=0.2,
        alignment=WallAlignment.NEGATIVE,
    )
    fp_vneg = w_vert_neg.footprint
    assert fp_vneg.left == pytest.approx(39.8)
    assert fp_vneg.right == pytest.approx(40.0)


def test_initial_building_all_wall_footprints_within_boundary():
    """Verify every wall footprint in the initial building is completely within [0, 40] x [0, 30]."""
    b = create_initial_building()
    b_boundary = b.exterior_boundary

    for wall in b.walls:
        fp = wall.footprint
        assert b_boundary.contains_rect(fp), (
            f"Wall '{wall.id}' footprint [{fp.left}, {fp.right}] x [{fp.top}, {fp.bottom}] "
            f"extends outside building boundary"
        )


def test_exterior_wall_centered_alignment_rejected_by_footprint_validation():
    """Verify an exterior wall placed on building edge with centered thickness is rejected."""
    # A wall at x=0 with CENTER alignment will have x in [-0.1, +0.1], which extends outside the building
    bad_wall = Wall(
        id="wall_out",
        start=Point(0.0, 0.0),
        end=Point(0.0, 14.0),
        thickness=0.2,
        alignment=WallAlignment.CENTER,
    )
    with pytest.raises(ValueError, match="physical footprint.*extends outside building bounds"):
        Building(id="bad_b", width=40.0, height=30.0, walls=[bad_wall])


# ---------------------------------------------------------------------------
# Initial Building Layout & Approved Specification Tests
# ---------------------------------------------------------------------------

def test_building_dimensions():
    """Verify building dimensions match the approved 40 m x 30 m specification."""
    b = create_initial_building()
    assert b.width == 40.0
    assert b.height == 30.0
    assert b.exterior_boundary == Rectangle(0.0, 0.0, 40.0, 30.0)


def test_four_rooms_and_one_corridor_structure():
    """Verify initial building has exactly 4 rooms and 1 horizontal corridor."""
    b = create_initial_building()

    area_ids = {a.id for a in b.walkable_areas}
    expected_ids = {"room_1", "room_2", "room_3", "room_4", "main_corridor"}
    assert area_ids == expected_ids
    assert len(b.walkable_areas) == 5

    rooms = [a for a in b.walkable_areas if a.area_type == "room"]
    corridors = [a for a in b.walkable_areas if a.area_type == "corridor"]
    assert len(rooms) == 4
    assert len(corridors) == 1
    assert corridors[0].id == "main_corridor"


def test_corridor_width_exact_3m():
    """Verify the horizontal corridor width (breadth) is exactly 3.0 m."""
    b = create_initial_building()
    corridor = b.get_walkable_area("main_corridor")
    assert corridor is not None
    assert corridor.bounds.height == 3.0
    assert corridor.bounds.width == 40.0


def test_rooms_above_and_below_corridor():
    """Verify two rooms are positioned strictly above and two below the corridor."""
    b = create_initial_building()
    corridor = b.get_walkable_area("main_corridor")
    assert corridor is not None

    corr_top = corridor.bounds.top
    corr_bottom = corridor.bounds.bottom

    r1 = b.get_walkable_area("room_1")
    r2 = b.get_walkable_area("room_2")
    r3 = b.get_walkable_area("room_3")
    r4 = b.get_walkable_area("room_4")

    assert r1 is not None and r2 is not None and r3 is not None and r4 is not None

    # Room 1 and 2 are above corridor
    assert r1.bounds.bottom <= corr_top
    assert r2.bounds.bottom <= corr_top
    assert r1.bounds.top == 0.0
    assert r2.bounds.top == 0.0

    # Room 3 and 4 are below corridor
    assert r3.bounds.top >= corr_bottom
    assert r4.bounds.top >= corr_bottom
    assert r3.bounds.bottom == 30.0
    assert r4.bounds.bottom == 30.0


# ---------------------------------------------------------------------------
# Explicit Doorway Representation Tests
# ---------------------------------------------------------------------------

def test_doorway_construction_and_properties():
    """Verify doorway creation, area connectivity check, and properties."""
    door = Doorway(
        id="door_test",
        connected_area_ids=("room_1", "main_corridor"),
        segment=LineSegment(Point(9.0, 13.5), Point(11.0, 13.5)),
        width=2.0,
    )
    assert door.id == "door_test"
    assert door.area_a_id == "room_1"
    assert door.area_b_id == "main_corridor"
    assert door.width == 2.0
    assert door.connects("room_1", "main_corridor")
    assert door.connects("main_corridor", "room_1")
    assert not door.connects("room_1", "room_2")


def test_doorway_invalid_construction():
    """Verify rejection of invalid doorway parameters."""
    # Self-connecting doorway
    with pytest.raises(ValueError, match="cannot connect an area to itself"):
        Doorway(
            id="door_loop",
            connected_area_ids=("room_1", "room_1"),
            segment=LineSegment(Point(0.0, 0.0), Point(1.0, 0.0)),
            width=2.0,
        )

    # Invalid width
    with pytest.raises(ValueError, match="width must be strictly positive"):
        Doorway(
            id="door_bad_w",
            connected_area_ids=("room_1", "room_2"),
            segment=LineSegment(Point(0.0, 0.0), Point(1.0, 0.0)),
            width=0.0,
        )


def test_initial_building_four_room_doorways():
    """Verify initial building contains exactly four doorways for the four room-to-corridor connections."""
    b = create_initial_building()
    assert len(b.doorways) == 4

    expected_doors = {
        "door_room1_corridor": ("room_1", "main_corridor", 9.0, 11.0, 13.5),
        "door_room2_corridor": ("room_2", "main_corridor", 29.0, 31.0, 13.5),
        "door_room3_corridor": ("room_3", "main_corridor", 9.0, 11.0, 16.5),
        "door_room4_corridor": ("room_4", "main_corridor", 29.0, 31.0, 16.5),
    }

    for door_id, (area_a, area_b, x1, x2, y) in expected_doors.items():
        door = b.get_doorway(door_id)
        assert door is not None, f"Missing doorway {door_id}"
        assert door.connects(area_a, area_b)
        assert door.width == 2.0
        assert door.segment.start == Point(x1, y)
        assert door.segment.end == Point(x2, y)


def test_doorway_adjacency_mismatch_rejected():
    """Verify error when a doorway connects areas that do not declare mutual adjacency."""
    c = WalkableArea(id="c", bounds=Rectangle(0.0, 10.0, 20.0, 5.0), connected_area_ids=["a1", "a2"])
    a1 = WalkableArea(id="a1", bounds=Rectangle(0.0, 0.0, 10.0, 10.0), connected_area_ids=["c"])
    a2 = WalkableArea(id="a2", bounds=Rectangle(10.0, 0.0, 10.0, 10.0), connected_area_ids=["c"])
    door = Doorway(
        id="door_a1_a2",
        connected_area_ids=("a1", "a2"),
        segment=LineSegment(Point(10.0, 4.0), Point(10.0, 6.0)),
        width=2.0,
    )
    with pytest.raises(ValueError, match="is not in 'a1'.connected_area_ids"):
        Building(id="bad_b", width=40.0, height=30.0, walkable_areas=[a1, a2, c], doorways=[door])


# ---------------------------------------------------------------------------
# Wall and Walkable-Area Consistency Tests
# ---------------------------------------------------------------------------

def test_doorway_opening_not_covered_by_wall():
    """Verify validation detects and rejects a wall that covers a doorway opening."""
    a1 = WalkableArea(id="a1", bounds=Rectangle(0.0, 0.0, 20.0, 10.0), connected_area_ids=["a2"])
    a2 = WalkableArea(id="a2", bounds=Rectangle(0.0, 10.0, 20.0, 10.0), connected_area_ids=["a1"])
    door = Doorway(
        id="door_1_2",
        connected_area_ids=("a1", "a2"),
        segment=LineSegment(Point(9.0, 10.0), Point(11.0, 10.0)),
        width=2.0,
    )
    # Solid wall right across the doorway from x=8.0 to x=12.0
    blocking_wall = Wall(
        id="block_wall",
        start=Point(8.0, 10.0),
        end=Point(12.0, 10.0),
        thickness=0.2,
    )
    with pytest.raises(ValueError, match="opening is covered by solid wall"):
        Building(
            id="bad_b",
            width=40.0,
            height=30.0,
            walkable_areas=[a1, a2],
            walls=[blocking_wall],
            doorways=[door],
        )


def test_exit_opening_not_covered_by_wall():
    """Verify validation detects and rejects a wall that covers an exit opening."""
    a1 = WalkableArea(id="a1", bounds=Rectangle(0.0, 0.0, 40.0, 30.0), connected_area_ids=[])
    ex = Exit(
        id="exit_left",
        segment=LineSegment(Point(0.0, 14.0), Point(0.0, 16.0)),
        width=2.0,
        boundary="left",
        connected_area_id="a1",
    )
    # Wall covering the exit from y=13 to y=17
    covering_wall = Wall(
        id="wall_block_exit",
        start=Point(0.0, 13.0),
        end=Point(0.0, 17.0),
        thickness=0.2,
        alignment=WallAlignment.POSITIVE,
    )
    with pytest.raises(ValueError, match="opening is covered by solid wall"):
        Building(
            id="bad_b",
            width=40.0,
            height=30.0,
            walkable_areas=[a1],
            walls=[covering_wall],
            exits=[ex],
        )


def test_obstacle_overlapping_doorway_rejected():
    """Verify validation detects and rejects an obstacle that intersects a doorway opening."""
    a1 = WalkableArea(id="a1", bounds=Rectangle(0.0, 0.0, 20.0, 10.0), connected_area_ids=["a2"])
    a2 = WalkableArea(id="a2", bounds=Rectangle(0.0, 10.0, 20.0, 10.0), connected_area_ids=["a1"])
    door = Doorway(
        id="door_1_2",
        connected_area_ids=("a1", "a2"),
        segment=LineSegment(Point(9.0, 10.0), Point(11.0, 10.0)),
        width=2.0,
    )
    # Obstacle overlapping doorway at y=10, x in [9, 11]
    blocking_obs = Obstacle(
        id="blocking_desk",
        bounds=Rectangle(x=9.5, y=9.0, width=1.0, height=2.0),
    )
    with pytest.raises(ValueError, match="opening is blocked by obstacle"):
        Building(
            id="bad_b",
            width=40.0,
            height=30.0,
            walkable_areas=[a1, a2],
            obstacles=[blocking_obs],
            doorways=[door],
        )


def test_obstacle_overlapping_exit_rejected():
    """Verify validation detects and rejects an obstacle that intersects an exit opening."""
    a1 = WalkableArea(id="a1", bounds=Rectangle(0.0, 0.0, 40.0, 30.0), connected_area_ids=[])
    ex = Exit(
        id="exit_left",
        segment=LineSegment(Point(0.0, 14.0), Point(0.0, 16.0)),
        width=2.0,
        boundary="left",
        connected_area_id="a1",
    )
    # Obstacle placed right across the exit opening
    blocking_obs = Obstacle(
        id="blocking_kiosk",
        bounds=Rectangle(x=0.0, y=14.5, width=1.0, height=1.0),
    )
    with pytest.raises(ValueError, match="opening is blocked by obstacle"):
        Building(
            id="bad_b",
            width=40.0,
            height=30.0,
            walkable_areas=[a1],
            obstacles=[blocking_obs],
            exits=[ex],
        )


# ---------------------------------------------------------------------------
# Reciprocal & Complete Graph Connectivity Tests
# ---------------------------------------------------------------------------

def test_room_to_corridor_connectivity():
    """Verify room-to-corridor connectivity is explicitly and reciprocally defined."""
    b = create_initial_building()
    corridor = b.get_walkable_area("main_corridor")
    assert corridor is not None

    for room_id in ["room_1", "room_2", "room_3", "room_4"]:
        room = b.get_walkable_area(room_id)
        assert room is not None
        assert "main_corridor" in room.connected_area_ids
        assert room_id in corridor.connected_area_ids


def test_non_reciprocal_connectivity_rejected():
    """Verify validation raises error if area connectivity is not reciprocal."""
    a1 = WalkableArea(id="a1", bounds=Rectangle(0.0, 0.0, 10.0, 10.0), connected_area_ids=["a2"])
    # a2 does not list a1
    a2 = WalkableArea(id="a2", bounds=Rectangle(10.0, 0.0, 10.0, 10.0), connected_area_ids=[])
    with pytest.raises(ValueError, match="not reciprocal"):
        Building(id="bad_b", width=40.0, height=30.0, walkable_areas=[a1, a2])


def test_disconnected_walkable_graph_rejected():
    """Verify validation raises error if walkable areas form disconnected components."""
    # a1 and a2 connected to each other
    a1 = WalkableArea(id="a1", bounds=Rectangle(0.0, 0.0, 10.0, 10.0), connected_area_ids=["a2"])
    a2 = WalkableArea(id="a2", bounds=Rectangle(10.0, 0.0, 10.0, 10.0), connected_area_ids=["a1"])
    # a3 is completely isolated
    a3 = WalkableArea(id="a3_isolated", bounds=Rectangle(20.0, 0.0, 10.0, 10.0), connected_area_ids=[])
    with pytest.raises(ValueError, match="Walkable area network is disconnected"):
        Building(id="bad_b", width=40.0, height=30.0, walkable_areas=[a1, a2, a3])


def test_invalid_connected_area_reference_rejected():
    """Verify validation raises error if an area references a non-existent area."""
    bad_area = WalkableArea(
        id="orphan_room",
        bounds=Rectangle(0.0, 0.0, 10.0, 10.0),
        connected_area_ids=["non_existent_corridor"],
    )
    with pytest.raises(ValueError, match="references non-existent connected area"):
        Building(id="bad_b", width=40.0, height=30.0, walkable_areas=[bad_area])


def test_self_referencing_connection_rejected():
    """Verify validation raises error if an area references itself."""
    loop_area = WalkableArea(
        id="loop_room",
        bounds=Rectangle(0.0, 0.0, 10.0, 10.0),
        connected_area_ids=["loop_room"],
    )
    with pytest.raises(ValueError, match="cannot reference itself in connected_area_ids"):
        Building(id="bad_b", width=40.0, height=30.0, walkable_areas=[loop_area])


# ---------------------------------------------------------------------------
# Exits Specification & Placement Tests
# ---------------------------------------------------------------------------

def test_three_exits_and_boundary_placement():
    """Verify three exits exist on left, right, and bottom outer boundaries."""
    b = create_initial_building()
    assert len(b.exits) == 3

    exit_left = b.get_exit("exit_left")
    exit_right = b.get_exit("exit_right")
    exit_bottom = b.get_exit("exit_bottom")

    assert exit_left is not None
    assert exit_right is not None
    assert exit_bottom is not None

    # Left boundary: X = 0.0
    assert exit_left.boundary == "left"
    assert exit_left.segment.start.x == pytest.approx(0.0)
    assert exit_left.segment.end.x == pytest.approx(0.0)
    assert exit_left.width == pytest.approx(2.0)

    # Right boundary: X = 40.0
    assert exit_right.boundary == "right"
    assert exit_right.segment.start.x == pytest.approx(40.0)
    assert exit_right.segment.end.x == pytest.approx(40.0)
    assert exit_right.width == pytest.approx(2.0)

    # Bottom boundary: Y = 30.0
    assert exit_bottom.boundary == "bottom"
    assert exit_bottom.segment.start.y == pytest.approx(30.0)
    assert exit_bottom.segment.end.y == pytest.approx(30.0)
    assert exit_bottom.width == pytest.approx(2.0)


def test_exit_connectivity():
    """Verify all exits connect to valid, accessible walkable areas."""
    b = create_initial_building()
    for ex in b.exits:
        area = b.get_walkable_area(ex.connected_area_id)
        assert area is not None, f"Exit {ex.id} must connect to an existing walkable area"


def test_invalid_exit_reference_rejected():
    """Verify validation raises error if an exit references a non-existent area."""
    bad_exit = Exit(
        id="exit_left",
        segment=LineSegment(Point(0.0, 10.0), Point(0.0, 12.0)),
        width=2.0,
        boundary="left",
        connected_area_id="phantom_corridor",
    )
    with pytest.raises(ValueError, match="references non-existent walkable area"):
        Building(id="bad_b", width=40.0, height=30.0, exits=[bad_exit])


def test_exit_boundary_mismatch_rejected():
    """Verify validation raises error if an exit's coordinates do not match its boundary declaration."""
    bad_exit = Exit(
        id="exit_left",
        segment=LineSegment(Point(5.0, 10.0), Point(5.0, 12.0)),
        width=2.0,
        boundary="left",
        connected_area_id="area_1",
    )
    area = WalkableArea(id="area_1", bounds=Rectangle(0.0, 0.0, 20.0, 20.0))
    with pytest.raises(ValueError, match="specified on 'left' boundary but x != 0.0"):
        Building(id="bad_b", width=40.0, height=30.0, walkable_areas=[area], exits=[bad_exit])


# ---------------------------------------------------------------------------
# Obstacles Tests
# ---------------------------------------------------------------------------

def test_obstacle_bounds_in_building():
    """Verify all obstacles are strictly within the building and room boundaries."""
    b = create_initial_building()
    assert len(b.obstacles) == 4

    for obs in b.obstacles:
        # Check inside building
        assert b.exterior_boundary.contains_rect(obs.bounds)
        # Check obstacle is within at least one walkable room
        enclosing_areas = [a for a in b.walkable_areas if a.bounds.contains_rect(obs.bounds)]
        assert len(enclosing_areas) == 1, f"Obstacle {obs.id} must be cleanly within a walkable room"


# ---------------------------------------------------------------------------
# Out-of-Bounds & Unique ID Validation Tests
# ---------------------------------------------------------------------------

def test_unique_id_enforcement_in_category():
    """Verify duplicate IDs in the same collection are rejected."""
    a1 = WalkableArea(id="dup", bounds=Rectangle(0.0, 0.0, 10.0, 10.0), connected_area_ids=["other"])
    other = WalkableArea(id="other", bounds=Rectangle(10.0, 0.0, 10.0, 10.0), connected_area_ids=["dup"])
    a2 = WalkableArea(id="dup", bounds=Rectangle(20.0, 0.0, 10.0, 10.0))
    with pytest.raises(ValueError, match="Duplicate ID 'dup'"):
        Building(id="bad_b", width=40.0, height=30.0, walkable_areas=[a1, other, a2])


def test_unique_id_enforcement_across_categories():
    """Verify duplicate IDs across different component types are rejected."""
    a1 = WalkableArea(id="shared_id", bounds=Rectangle(0.0, 0.0, 10.0, 10.0))
    obs = Obstacle(id="shared_id", bounds=Rectangle(2.0, 2.0, 2.0, 2.0))
    with pytest.raises(ValueError, match="reused across building components"):
        Building(id="bad_b", width=40.0, height=30.0, walkable_areas=[a1], obstacles=[obs])


def test_out_of_bounds_walkable_area_rejected():
    """Verify walkable area extending beyond building bounds is rejected."""
    out_area = WalkableArea(
        id="overflow_room",
        bounds=Rectangle(30.0, 0.0, 20.0, 10.0),  # extends to x=50 > 40
    )
    with pytest.raises(ValueError, match="extends outside building bounds"):
        Building(id="bad_b", width=40.0, height=30.0, walkable_areas=[out_area])


def test_out_of_bounds_obstacle_rejected():
    """Verify obstacle extending beyond building bounds is rejected."""
    out_obs = Obstacle(
        id="overflow_obs",
        bounds=Rectangle(35.0, 25.0, 10.0, 10.0),  # extends to x=45, y=35
    )
    with pytest.raises(ValueError, match="extends outside building bounds"):
        Building(id="bad_b", width=40.0, height=30.0, obstacles=[out_obs])


def test_out_of_bounds_wall_rejected():
    """Verify wall with coordinates outside building is rejected."""
    bad_wall = Wall(
        id="bad_wall",
        start=Point(0.0, 0.0),
        end=Point(45.0, 0.0),  # x=45 > 40
        thickness=0.2,
    )
    with pytest.raises(ValueError, match="extends outside building bounds"):
        Building(id="bad_b", width=40.0, height=30.0, walls=[bad_wall])


def test_out_of_bounds_exit_rejected():
    """Verify exit with coordinates outside building is rejected."""
    area = WalkableArea(id="room_1", bounds=Rectangle(0.0, 0.0, 40.0, 30.0))
    bad_exit = Exit(
        id="bad_exit",
        segment=LineSegment(Point(0.0, 28.0), Point(0.0, 35.0)),  # y=35 > 30
        width=2.0,
        boundary="left",
        connected_area_id="room_1",
    )
    with pytest.raises(ValueError, match="extends outside building bounds"):
        Building(id="bad_b", width=40.0, height=30.0, walkable_areas=[area], exits=[bad_exit])


# ---------------------------------------------------------------------------
# Strict Architectural Scope Isolation Tests
# ---------------------------------------------------------------------------

def test_no_disallowed_imports():
    """Verify the environment module in isolation does NOT import pygame, torch, or MID."""
    import os
    import subprocess
    from pathlib import Path

    cmd = [
        sys.executable,
        "-c",
        "import genevac.environment, sys; assert 'pygame' not in sys.modules and 'torch' not in sys.modules and 'mid' not in sys.modules",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Disallowed modules imported: {res.stderr}"


# ---------------------------------------------------------------------------
# Targeted Verification Audit Tests
# ---------------------------------------------------------------------------

def test_doorway_length_width_mismatch_rejected():
    """Verify rejection when doorway declared width does not match segment length."""
    a1 = WalkableArea(id="a1", bounds=Rectangle(0.0, 0.0, 10.0, 10.0), connected_area_ids=["a2"])
    a2 = WalkableArea(id="a2", bounds=Rectangle(10.0, 0.0, 10.0, 10.0), connected_area_ids=["a1"])
    # Segment is 2m long, but width declared as 3m
    door = Doorway(
        id="door_mismatch",
        connected_area_ids=("a1", "a2"),
        segment=LineSegment(Point(10.0, 4.0), Point(10.0, 6.0)),
        width=3.0,
    )
    with pytest.raises(ValueError, match="does not match declared width"):
        Building(id="b", width=40.0, height=30.0, walkable_areas=[a1, a2], doorways=[door])


def test_doorway_not_on_area_boundary_rejected():
    """Verify rejection when doorway crossing segment is not on the boundary of its connected areas."""
    a1 = WalkableArea(id="a1", bounds=Rectangle(0.0, 0.0, 10.0, 10.0), connected_area_ids=["a2"])
    a2 = WalkableArea(id="a2", bounds=Rectangle(10.0, 0.0, 10.0, 10.0), connected_area_ids=["a1"])
    # Doorway placed at x=5.0 (inside a1, not on boundary with a2)
    door = Doorway(
        id="door_misplaced",
        connected_area_ids=("a1", "a2"),
        segment=LineSegment(Point(5.0, 4.0), Point(5.0, 6.0)),
        width=2.0,
    )
    with pytest.raises(ValueError, match="crossing segment is not on the boundary"):
        Building(id="b", width=40.0, height=30.0, walkable_areas=[a1, a2], doorways=[door])


def test_exit_length_width_mismatch_rejected():
    """Verify rejection when exit declared width does not match segment length."""
    a1 = WalkableArea(id="a1", bounds=Rectangle(0.0, 0.0, 40.0, 30.0), connected_area_ids=[])
    # Segment is 2m long, but width is declared as 2.5m
    ex = Exit(
        id="exit_mismatch",
        segment=LineSegment(Point(0.0, 10.0), Point(0.0, 12.0)),
        width=2.5,
        boundary="left",
        connected_area_id="a1",
    )
    with pytest.raises(ValueError, match="does not match declared width"):
        Building(id="b", width=40.0, height=30.0, walkable_areas=[a1], exits=[ex])


def test_exit_not_on_connected_area_boundary_rejected():
    """Verify rejection when exit is not on the boundary of its connected area."""
    # a1 only covers x in [10, 40], so x=0 is not on its boundary
    a1 = WalkableArea(id="a1", bounds=Rectangle(10.0, 0.0, 30.0, 30.0), connected_area_ids=[])
    ex = Exit(
        id="exit_detached",
        segment=LineSegment(Point(0.0, 10.0), Point(0.0, 12.0)),
        width=2.0,
        boundary="left",
        connected_area_id="a1",
    )
    with pytest.raises(ValueError, match="opening is not along the boundary"):
        Building(id="b", width=40.0, height=30.0, walkable_areas=[a1], exits=[ex])


def test_wall_footprint_thickness_intrusion_into_doorway_rejected():
    """Verify that a perpendicular wall whose thickness intrudes into a doorway is rejected."""
    a1 = WalkableArea(id="a1", bounds=Rectangle(0.0, 0.0, 20.0, 10.0), connected_area_ids=["a2"])
    a2 = WalkableArea(id="a2", bounds=Rectangle(0.0, 10.0, 20.0, 10.0), connected_area_ids=["a1"])
    # Doorway along horizontal seam y=10.0, x in [9.0, 11.0]
    door = Doorway(
        id="door_test",
        connected_area_ids=("a1", "a2"),
        segment=LineSegment(Point(9.0, 10.0), Point(11.0, 10.0)),
        width=2.0,
    )
    # Perpendicular vertical wall at x=10.0, ending at y=10.0, with thickness 0.2m (CENTER alignment: x in [9.9, 10.1])
    # The wall centerline touches at (10, 10), but its physical footprint penetrates the doorway opening
    intruding_wall = Wall(
        id="perp_wall",
        start=Point(10.0, 5.0),
        end=Point(10.0, 10.0),
        thickness=0.2,
        alignment=WallAlignment.CENTER,
    )
    with pytest.raises(ValueError, match="opening is covered by solid wall"):
        Building(
            id="bad_b",
            width=40.0,
            height=30.0,
            walkable_areas=[a1, a2],
            walls=[intruding_wall],
            doorways=[door],
        )


def test_initial_building_graph_connectivity_and_exit_access():
    """Verify that all four rooms and corridor form one connected walkable graph and all exits connect to it."""
    b = create_initial_building()

    # 1. Verify graph is connected: all 5 areas reachable from room_1
    visited = set()
    queue = ["room_1"]
    while queue:
        curr = queue.pop(0)
        if curr not in visited:
            visited.add(curr)
            for neighbor in b.get_walkable_area(curr).connected_area_ids:
                if neighbor not in visited:
                    queue.append(neighbor)

    assert visited == {"room_1", "room_2", "room_3", "room_4", "main_corridor"}

    # 2. Verify all three exits connect to areas within this connected component
    for ex in b.exits:
        assert ex.connected_area_id in visited
