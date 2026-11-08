# GenEvac-Sentinel — Building & Environment Specification

## 1. Coordinate Convention & Global Units

The simulation environment strictly adheres to the following coordinate convention:

| Attribute | Specification |
| :--- | :--- |
| **Origin $(0.0, 0.0)$** | Top-left corner of the building boundary |
| **Horizontal Axis ($+X$)** | Extends eastward (to the right) from $0.0$ to $40.0\text{ m}$ |
| **Vertical Axis ($+Y$)** | Extends southward (downward) from $0.0$ to $30.0\text{ m}$ |
| **Coordinate Units** | SI Metres ($\text{m}$) represented as continuous floating-point values |
| **Global Building Size** | Width: $40.0\text{ m}$, Height: $30.0\text{ m}$ (Area: $1,200\text{ m}^2$) |

---

## 2. Floor Plan Architectural Layout

```
(0,0)═══════════════════════(20,0)═══════════════════════(40,0)
║                           ║                           ║
║         ROOM 1            ║         ROOM 2            ║
║    [4,6]x[4,5.5] (obs)    ║   [24,26]x[4,5.5] (obs)   ║
║                           ║                           ║
║         ═══[DOOR 1]═══    ║         ═══[DOOR 2]═══    ║
║            (9..11, 13.5)  ║           (29..31, 13.5)  ║
╠═══════════════════════════╩═══════════════════════════╣ (y=13.5)
║ [EXIT LEFT]                               [EXIT RIGHT]║
║ (0, 14..16)         MAIN CORRIDOR         (40, 14..16)║ (width=3.0m)
╠═══════════════════════════╦═══════════════════════════╣ (y=16.5)
║            (9..11, 16.5)  ║           (29..31, 16.5)  ║
║         ═══[DOOR 3]═══    ║         ═══[DOOR 4]═══    ║
║                           ║                           ║
║         ROOM 3            ║         ROOM 4            ║
║    [4,6]x[20,21.5] (obs)  ║   [24,26]x[20,21.5] (obs) ║
║         [EXIT BOTTOM]     ║                           ║
║         (9..11, 30.0)     ║                           ║
(0,30)══════════════════════(20,30)══════════════════════(40,30)
```

---

## 3. Walkable Areas & Topological Graph

The building floor plan contains 5 primary walkable areas:

| Area ID | Area Type | X Range ($\text{m}$) | Y Range ($\text{m}$) | Dimensions ($W \times H$) | Connected Area IDs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `room_1` | Room (NW) | $[0.0, 20.0]$ | $[0.0, 13.5]$ | $20.0\text{ m} \times 13.5\text{ m}$ | `["main_corridor"]` |
| `room_2` | Room (NE) | $[20.0, 40.0]$ | $[0.0, 13.5]$ | $20.0\text{ m} \times 13.5\text{ m}$ | `["main_corridor"]` |
| `main_corridor` | Corridor | $[0.0, 40.0]$ | $[13.5, 16.5]$ | $40.0\text{ m} \times 3.0\text{ m}$ | `["room_1", "room_2", "room_3", "room_4"]` |
| `room_3` | Room (SW) | $[0.0, 20.0]$ | $[16.5, 30.0]$ | $20.0\text{ m} \times 13.5\text{ m}$ | `["main_corridor"]` |
| `room_4` | Room (SE) | $[20.0, 40.0]$ | $[16.5, 30.0]$ | $20.0\text{ m} \times 13.5\text{ m}$ | `["main_corridor"]` |

### Topological Adjacency
- The central corridor (`main_corridor`) serves as the central evacuation spine.
- All four rooms connect directly to `main_corridor` via dedicated doorways.
- There are no direct doors between `room_1` and `room_2` or between `room_3` and `room_4`.

---

## 4. Doorway Portals

Four 2.0 m wide doorways connect the rooms to the central corridor:

| Doorway ID | Connects | Line Segment Start | Line Segment End | Portal Width | Midpoint Portal |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `door_room1_corridor` | `room_1` $\leftrightarrow$ `main_corridor` | $(9.0, 13.5)$ | $(11.0, 13.5)$ | $2.0\text{ m}$ | $(10.0, 13.5)$ |
| `door_room2_corridor` | `room_2` $\leftrightarrow$ `main_corridor` | $(29.0, 13.5)$ | $(31.0, 13.5)$ | $2.0\text{ m}$ | $(30.0, 13.5)$ |
| `door_room3_corridor` | `room_3` $\leftrightarrow$ `main_corridor` | $(9.0, 16.5)$ | $(11.0, 16.5)$ | $2.0\text{ m}$ | $(10.0, 16.5)$ |
| `door_room4_corridor` | `room_4` $\leftrightarrow$ `main_corridor` | $(29.0, 16.5)$ | $(31.0, 16.5)$ | $2.0\text{ m}$ | $(30.0, 16.5)$ |

---

## 5. Exterior Exits

Three 2.0 m wide evacuation exits are positioned on outer building boundaries:

| Exit ID | Outer Boundary | Line Segment Start | Line Segment End | Width | Connected Area | Initial State |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `exit_left` | Left ($X=0.0$) | $(0.0, 14.0)$ | $(0.0, 16.0)$ | $2.0\text{ m}$ | `main_corridor` | Open (`is_open=True`) |
| `exit_right` | Right ($X=40.0$) | $(40.0, 14.0)$ | $(40.0, 16.0)$ | $2.0\text{ m}$ | `main_corridor` | Open (`is_open=True`) |
| `exit_bottom` | Bottom ($Y=30.0$) | $(9.0, 30.0)$ | $(11.0, 30.0)$ | $2.0\text{ m}$ | `room_3` | Open (`is_open=True`) |

> [!NOTE]
> `exit_bottom` connects directly to `room_3`. Pedestrians from `room_1`, `room_2`, or `room_4` targeting `exit_bottom` must traverse `main_corridor`, pass through `door_room3_corridor` into `room_3`, and exit through `exit_bottom`.

---

## 6. Static Interior Obstacles

Each room contains one static rectangular furniture obstacle of identical dimensions ($2.0\text{ m}$ width $\times 1.5\text{ m}$ height):

| Obstacle ID | Located In | Bounding Box $[X_{\min}, X_{\max}] \times [Y_{\min}, Y_{\max}]$ | Width $\times$ Height | Description |
| :--- | :--- | :--- | :--- | :--- |
| `obstacle_room1` | `room_1` | $[4.0, 6.0] \times [4.0, 5.5]$ | $2.0\text{ m} \times 1.5\text{ m}$ | Conference table / pillar |
| `obstacle_room2` | `room_2` | $[24.0, 26.0] \times [4.0, 5.5]$ | $2.0\text{ m} \times 1.5\text{ m}$ | Office cluster |
| `obstacle_room3` | `room_3` | $[4.0, 6.0] \times [20.0, 21.5]$ | $2.0\text{ m} \times 1.5\text{ m}$ | Storage obstacle |
| `obstacle_room4` | `room_4` | $[24.0, 26.0] \times [20.0, 21.5]$ | $2.0\text{ m} \times 1.5\text{ m}$ | Workspace obstacle |

---

## 7. Walls & Physical Thickness Footprints

All walls possess a standard physical thickness of $0.2\text{ m}$.

### Exterior Walls (Inward Thickness Alignment)
- Exterior walls expand inward into the building boundary so the outer facade strictly aligns with $[0, 40] \times [0, 30]$.
- **Top Wall** (`wall_ext_top`): $(0.0, 0.0) \to (40.0, 0.0)$, thickness $0.2$, alignment `POSITIVE` ($+Y$, footprint $[0, 40] \times [0.0, 0.2]$).
- **Left Wall** (divided by `exit_left`):
  - `wall_ext_left_north`: $(0.0, 0.0) \to (0.0, 14.0)$, alignment `POSITIVE` ($+X$, footprint $[0.0, 0.2] \times [0.0, 14.0]$).
  - `wall_ext_left_south`: $(0.0, 16.0) \to (0.0, 30.0)$, alignment `POSITIVE` ($+X$, footprint $[0.0, 0.2] \times [16.0, 30.0]$).
- **Right Wall** (divided by `exit_right`):
  - `wall_ext_right_north`: $(40.0, 0.0) \to (40.0, 14.0)$, alignment `NEGATIVE` ($-X$, footprint $[39.8, 40.0] \times [0.0, 14.0]$).
  - `wall_ext_right_south`: $(40.0, 16.0) \to (40.0, 30.0)$, alignment `NEGATIVE` ($-X$, footprint $[39.8, 40.0] \times [16.0, 30.0]$).
- **Bottom Wall** (divided by `exit_bottom`):
  - `wall_ext_bottom_west`: $(0.0, 30.0) \to (9.0, 30.0)$, alignment `NEGATIVE` ($-Y$, footprint $[0.0, 9.0] \times [29.8, 30.0]$).
  - `wall_ext_bottom_east`: $(11.0, 30.0) \to (40.0, 30.0)$, alignment `NEGATIVE` ($-Y$, footprint $[11.0, 40.0] \times [29.8, 30.0]$).

### Interior Dividing Walls (Centered Alignment)
- Centered alignment expands $\pm 0.1\text{ m}$ on either side of the centerline.
- `wall_int_div_north`: $(20.0, 0.0) \to (20.0, 13.5)$ (footprint $[19.9, 20.1] \times [0.0, 13.5]$).
- `wall_int_div_south`: $(20.0, 16.5) \to (20.0, 30.0)$ (footprint $[19.9, 20.1] \times [16.5, 30.0]$).
- North corridor wall (at $Y=13.5$, interrupted by doorways):
  - `wall_int_room1_corr_w`: $(0.0, 13.5) \to (9.0, 13.5)$
  - `wall_int_room1_corr_e`: $(11.0, 13.5) \to (20.0, 13.5)$
  - `wall_int_room2_corr_w`: $(20.0, 13.5) \to (29.0, 13.5)$
  - `wall_int_room2_corr_e`: $(31.0, 13.5) \to (40.0, 13.5)$
- South corridor wall (at $Y=16.5$, interrupted by doorways):
  - `wall_int_room3_corr_w`: $(0.0, 16.5) \to (9.0, 16.5)$
  - `wall_int_room3_corr_e`: $(11.0, 16.5) \to (20.0, 16.5)$
  - `wall_int_room4_corr_w`: $(20.0, 16.5) \to (29.0, 16.5)$
  - `wall_int_room4_corr_e`: $(31.0, 16.5) \to (40.0, 16.5)$

---

## 8. JSON Serialization Format

Buildings are serialized using `genevac.environment.serialization`:

```json
{
  "schema_version": "1.0",
  "building": {
    "id": "building_main",
    "width": 40.0,
    "height": 30.0,
    "walkable_areas": [ ... ],
    "doorways": [ ... ],
    "exits": [
      {
        "id": "exit_left",
        "segment": {
          "start": {"x": 0.0, "y": 14.0},
          "end": {"x": 0.0, "y": 16.0}
        },
        "width": 2.0,
        "boundary": "left",
        "connected_area_id": "main_corridor",
        "is_open": true
      }
    ],
    "obstacles": [ ... ],
    "walls": [ ... ]
  }
}
```
If `"is_open"` is omitted from an exit record in older maps, the deserializer defaults to `True` for backward compatibility.

