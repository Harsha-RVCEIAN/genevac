# GenEvac-Sentinel: Week 1 Design Specification & Roadmap

## 1. Project Overview & Strategy
GenEvac-Sentinel is an educational 2D evacuation simulation. The development strategy prioritizes building and thoroughly verifying a reliable 2D Minimum Viable Product (MVP) before introducing pedestrian dynamics, evaluation metrics, AI prediction, or webcam integration.

Week 1 is dedicated exclusively to the building map and static environment.

---

## 2. Approved Design Decisions

| Parameter | Approved Decision | Notes / Constraints |
|---|---|---|
| **Map Dimensions** | 40 m wide × 30 m high | Metric units (meters) |
| **Coordinate Origin** | Top-left corner `(0, 0)` | Standard 2D graphics screen-space convention |
| **Coordinate Directions** | X increases right; Y increases downward | $(x, y) \in [0.0, 40.0] \times [0.0, 30.0]$ |
| **Geometry Representation** | Rectangles and line segments | Axis-aligned bounding primitives and linear segments |
| **Building Layout** | 1 horizontal main corridor (3 m wide) with 4 rooms | 2 rooms positioned above corridor, 2 rooms below corridor |
| **Exits** | 3 exits | Located on left, right, and bottom outer walls |
| **Obstacles** | Static obstacles | Will be placed within the sample environment |
| **Map Configuration** | JSON format | Human-readable and declarative configuration |
| **Visualization** | Pygame | 2D rendering engine |
| **Display Scaling** | Aspect-ratio preserving scaling | Fits map into window without distortion (letterboxing/pillarboxing) |
| **Geometry Validation** | Strict boundary checking | Reject geometry outside map $[0, 40] \times [0, 30]$; permit intentional shared edges / connected boundaries |
| **Testing Framework** | pytest | Automated test suite execution |
| **Future Planned Population**| 100 pedestrians | **Deferred:** Not implemented in Week 1 |

---

## 3. Five-Phase Week 1 Roadmap

```
Phase 1: Design Finalization & Setup (CURRENT)
   │
   ▼
Phase 2: Environment Data Model (Represent boundaries, walls, walkable areas, obstacles, exits)
   │
   ▼
Phase 3: JSON Loading & Validation (Parser, schema validation, boundary enforcement)
   │
   ▼
Phase 4: Static Pygame Visualization (Proportional scaling, color-coded rendering)
   │
   ▼
Phase 5: Testing & Sign-off (Test coverage, invalid config rejection, final verification)
```

### Phase Details:
- **Phase 1 — Design finalization and setup (Current Phase)**
  - Repository inspection.
  - Environment and dependency setup (`Python 3.11`, `pygame`, `pytest`).
  - Approved design and coordinate conventions documented.
- **Phase 2 — Environment data model**
  - Represent building boundaries, walls, walkable areas, obstacles, and uniquely identified exits in Python data structures.
- **Phase 3 — JSON loading and validation**
  - Load map configuration from JSON files.
  - Validate coordinates, geometry bounds, and exit placements.
  - Reject malformed or out-of-bounds configurations with informative errors.
- **Phase 4 — Static Pygame visualization**
  - Render the building map statically.
  - Visually distinguish walls, walkable areas, obstacles, and exits.
  - Preserve map proportions when scaling to window dimensions.
- **Phase 5 — Testing and sign-off**
  - Verify valid and invalid configurations with automated pytest suites.
  - Inspect visual rendering and confirm acceptance criteria.

---

## 4. Unresolved Items & Implementation Planning Points
The following items are intentionally recorded for resolution in subsequent phases, without making contradictory assumptions:

1. **Exact Room Coordinates and Dimensions:**
   - The exact $(x, y, w, h)$ bounds for the two rooms above the corridor and two rooms below the corridor will be defined during Phase 2/3.
2. **Corridor Vertical Placement:**
   - The vertical offset of the 3 m horizontal corridor within the 30 m height (e.g., $y \in [13.5, 16.5]$ or similar) will be finalized during floor plan modeling.
3. **Exit Dimensions and Exact Segment Positions:**
   - The exact widths (in meters) and segment coordinates for the left, right, and bottom exits will be specified.
4. **Static Obstacle Locations:**
   - The exact types, quantities, and coordinates of static obstacles in the sample map will be established in Phase 3.
5. **JSON Schema Key Conventions:**
   - Formal naming standards (e.g., `snake_case` keys: `walkable_areas`, `walls`, `exits`, `obstacles`, `metadata`) will be finalized during Phase 3 schema design.
6. **Default Pygame Window Resolution:**
   - Default window size (e.g., 1024×768 or 1280×960) and background canvas colors to be approved in Phase 4.
