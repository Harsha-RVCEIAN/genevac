"""Initial floor-plan environment builder for GenEvac-Sentinel.

Constructs the approved initial 40 m x 30 m building with:
- 4 rooms (two above, two below horizontal corridor)
- 1 horizontal corridor of exactly 3 m width
- 4 explicit doorways representing room-to-corridor portals
- 3 exterior exits (left, right, bottom)
- Static interior obstacles
- Exterior walls with inward alignment and gaps for exits
- Interior walls with doorway openings for room-to-corridor connectivity
"""

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


def create_initial_building() -> Building:
    """Build and return the validated initial floor plan for GenEvac-Sentinel.

    Approved Specification:
    - Dimensions: 40.0 m wide x 30.0 m high.
    - Top-left origin: (0.0, 0.0).
    - Corridor: 3.0 m wide, centered at Y in [13.5, 16.5].
    - Rooms:
        * room_1: [0.0, 20.0] x [0.0, 13.5] (Top-Left / Northwest)
        * room_2: [20.0, 40.0] x [0.0, 13.5] (Top-Right / Northeast)
        * room_3: [0.0, 20.0] x [16.5, 30.0] (Bottom-Left / Southwest)
        * room_4: [20.0, 40.0] x [16.5, 30.0] (Bottom-Right / Southeast)
    - Doorways (4 portals connecting rooms to corridor):
        * door_room1_corridor: (9.0, 13.5) to (11.0, 13.5), width = 2.0 m
        * door_room2_corridor: (29.0, 13.5) to (31.0, 13.5), width = 2.0 m
        * door_room3_corridor: (9.0, 16.5) to (11.0, 16.5), width = 2.0 m
        * door_room4_corridor: (29.0, 16.5) to (31.0, 16.5), width = 2.0 m
    - Exits:
        * exit_left: Left boundary X = 0.0, Y in [14.0, 16.0], width = 2.0 m
        * exit_right: Right boundary X = 40.0, Y in [14.0, 16.0], width = 2.0 m
        * exit_bottom: Bottom boundary Y = 30.0, X in [9.0, 11.0], width = 2.0 m
    - Wall thickness: 0.2 m with inward alignment on exterior boundaries.
    - Obstacles: 4 static rectangular furniture/pillar obstacles, one per room.
    """
    # 1. Walkable Areas
    walkable_areas = [
        WalkableArea(
            id="room_1",
            bounds=Rectangle(x=0.0, y=0.0, width=20.0, height=13.5),
            connected_area_ids=["main_corridor"],
            area_type="room",
        ),
        WalkableArea(
            id="room_2",
            bounds=Rectangle(x=20.0, y=0.0, width=20.0, height=13.5),
            connected_area_ids=["main_corridor"],
            area_type="room",
        ),
        WalkableArea(
            id="main_corridor",
            bounds=Rectangle(x=0.0, y=13.5, width=40.0, height=3.0),
            connected_area_ids=["room_1", "room_2", "room_3", "room_4"],
            area_type="corridor",
        ),
        WalkableArea(
            id="room_3",
            bounds=Rectangle(x=0.0, y=16.5, width=20.0, height=13.5),
            connected_area_ids=["main_corridor"],
            area_type="room",
        ),
        WalkableArea(
            id="room_4",
            bounds=Rectangle(x=20.0, y=16.5, width=20.0, height=13.5),
            connected_area_ids=["main_corridor"],
            area_type="room",
        ),
    ]

    # 2. Doorways (explicit portals connecting rooms to corridor)
    doorways = [
        Doorway(
            id="door_room1_corridor",
            connected_area_ids=("room_1", "main_corridor"),
            segment=LineSegment(Point(9.0, 13.5), Point(11.0, 13.5)),
            width=2.0,
        ),
        Doorway(
            id="door_room2_corridor",
            connected_area_ids=("room_2", "main_corridor"),
            segment=LineSegment(Point(29.0, 13.5), Point(31.0, 13.5)),
            width=2.0,
        ),
        Doorway(
            id="door_room3_corridor",
            connected_area_ids=("room_3", "main_corridor"),
            segment=LineSegment(Point(9.0, 16.5), Point(11.0, 16.5)),
            width=2.0,
        ),
        Doorway(
            id="door_room4_corridor",
            connected_area_ids=("room_4", "main_corridor"),
            segment=LineSegment(Point(29.0, 16.5), Point(31.0, 16.5)),
            width=2.0,
        ),
    ]

    # 3. Exits (openings on exterior boundaries, not solid walls)
    exits = [
        Exit(
            id="exit_left",
            segment=LineSegment(Point(0.0, 14.0), Point(0.0, 16.0)),
            width=2.0,
            boundary="left",
            connected_area_id="main_corridor",
        ),
        Exit(
            id="exit_right",
            segment=LineSegment(Point(40.0, 14.0), Point(40.0, 16.0)),
            width=2.0,
            boundary="right",
            connected_area_id="main_corridor",
        ),
        Exit(
            id="exit_bottom",
            segment=LineSegment(Point(9.0, 30.0), Point(11.0, 30.0)),
            width=2.0,
            boundary="bottom",
            connected_area_id="room_3",
        ),
    ]

    # 4. Walls
    wall_thickness = 0.2
    walls = [
        # Exterior walls (with gaps accounting for exit openings and inward thickness alignment)
        # Top boundary (no exits, thickness expands inward into building: +Y)
        Wall(
            id="wall_ext_top",
            start=Point(0.0, 0.0),
            end=Point(40.0, 0.0),
            thickness=wall_thickness,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.POSITIVE,
        ),
        # Left boundary (gap at Y in [14.0, 16.0] for exit_left, thickness expands inward: +X)
        Wall(
            id="wall_ext_left_north",
            start=Point(0.0, 0.0),
            end=Point(0.0, 14.0),
            thickness=wall_thickness,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.POSITIVE,
        ),
        Wall(
            id="wall_ext_left_south",
            start=Point(0.0, 16.0),
            end=Point(0.0, 30.0),
            thickness=wall_thickness,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.POSITIVE,
        ),
        # Right boundary (gap at Y in [14.0, 16.0] for exit_right, thickness expands inward: -X)
        Wall(
            id="wall_ext_right_north",
            start=Point(40.0, 0.0),
            end=Point(40.0, 14.0),
            thickness=wall_thickness,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.NEGATIVE,
        ),
        Wall(
            id="wall_ext_right_south",
            start=Point(40.0, 16.0),
            end=Point(40.0, 30.0),
            thickness=wall_thickness,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.NEGATIVE,
        ),
        # Bottom boundary (gap at X in [9.0, 11.0] for exit_bottom, thickness expands inward: -Y)
        Wall(
            id="wall_ext_bottom_west",
            start=Point(0.0, 30.0),
            end=Point(9.0, 30.0),
            thickness=wall_thickness,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.NEGATIVE,
        ),
        Wall(
            id="wall_ext_bottom_east",
            start=Point(11.0, 30.0),
            end=Point(40.0, 30.0),
            thickness=wall_thickness,
            wall_type=WallType.EXTERIOR,
            alignment=WallAlignment.NEGATIVE,
        ),
        # Interior dividing walls (centered alignment)
        # Vertical divider between room_1 and room_2 (north)
        Wall(
            id="wall_int_div_north",
            start=Point(20.0, 0.0),
            end=Point(20.0, 13.5),
            thickness=wall_thickness,
            wall_type=WallType.INTERIOR,
            alignment=WallAlignment.CENTER,
        ),
        # Vertical divider between room_3 and room_4 (south)
        Wall(
            id="wall_int_div_south",
            start=Point(20.0, 16.5),
            end=Point(20.0, 30.0),
            thickness=wall_thickness,
            wall_type=WallType.INTERIOR,
            alignment=WallAlignment.CENTER,
        ),
        # North corridor wall: between room_1 & corridor (doorway at X in [9.0, 11.0])
        Wall(
            id="wall_int_room1_corr_w",
            start=Point(0.0, 13.5),
            end=Point(9.0, 13.5),
            thickness=wall_thickness,
            wall_type=WallType.INTERIOR,
            alignment=WallAlignment.CENTER,
        ),
        Wall(
            id="wall_int_room1_corr_e",
            start=Point(11.0, 13.5),
            end=Point(20.0, 13.5),
            thickness=wall_thickness,
            wall_type=WallType.INTERIOR,
            alignment=WallAlignment.CENTER,
        ),
        # North corridor wall: between room_2 & corridor (doorway at X in [29.0, 31.0])
        Wall(
            id="wall_int_room2_corr_w",
            start=Point(20.0, 13.5),
            end=Point(29.0, 13.5),
            thickness=wall_thickness,
            wall_type=WallType.INTERIOR,
            alignment=WallAlignment.CENTER,
        ),
        Wall(
            id="wall_int_room2_corr_e",
            start=Point(31.0, 13.5),
            end=Point(40.0, 13.5),
            thickness=wall_thickness,
            wall_type=WallType.INTERIOR,
            alignment=WallAlignment.CENTER,
        ),
        # South corridor wall: between room_3 & corridor (doorway at X in [9.0, 11.0])
        Wall(
            id="wall_int_room3_corr_w",
            start=Point(0.0, 16.5),
            end=Point(9.0, 16.5),
            thickness=wall_thickness,
            wall_type=WallType.INTERIOR,
            alignment=WallAlignment.CENTER,
        ),
        Wall(
            id="wall_int_room3_corr_e",
            start=Point(11.0, 16.5),
            end=Point(20.0, 16.5),
            thickness=wall_thickness,
            wall_type=WallType.INTERIOR,
            alignment=WallAlignment.CENTER,
        ),
        # South corridor wall: between room_4 & corridor (doorway at X in [29.0, 31.0])
        Wall(
            id="wall_int_room4_corr_w",
            start=Point(20.0, 16.5),
            end=Point(29.0, 16.5),
            thickness=wall_thickness,
            wall_type=WallType.INTERIOR,
            alignment=WallAlignment.CENTER,
        ),
        Wall(
            id="wall_int_room4_corr_e",
            start=Point(31.0, 16.5),
            end=Point(40.0, 16.5),
            thickness=wall_thickness,
            wall_type=WallType.INTERIOR,
            alignment=WallAlignment.CENTER,
        ),
    ]

    # 5. Obstacles (static interior obstacles placed within rooms, clear of paths/exits/doorways)
    obstacles = [
        Obstacle(
            id="obstacle_room1",
            bounds=Rectangle(x=4.0, y=4.0, width=2.0, height=1.5),
        ),
        Obstacle(
            id="obstacle_room2",
            bounds=Rectangle(x=24.0, y=4.0, width=2.0, height=1.5),
        ),
        Obstacle(
            id="obstacle_room3",
            bounds=Rectangle(x=4.0, y=20.0, width=2.0, height=1.5),
        ),
        Obstacle(
            id="obstacle_room4",
            bounds=Rectangle(x=24.0, y=20.0, width=2.0, height=1.5),
        ),
    ]

    return Building(
        id="building_main",
        width=40.0,
        height=30.0,
        walkable_areas=walkable_areas,
        walls=walls,
        obstacles=obstacles,
        exits=exits,
        doorways=doorways,
    )
