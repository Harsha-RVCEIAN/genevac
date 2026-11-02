# GenEvac-Sentinel

An educational 2D evacuation simulation platform.

## Current Status: Week 1 Completed & Signed Off (Phases 1–5 Complete)

GenEvac-Sentinel follows a strict phased MVP development strategy. Week 1 is focused solely on the building map and static environment.

- **Phase 1: Design Finalization and Setup** — Completed & Approved
- **Phase 2: Environment Data Model** — Completed & Approved
- **Phase 3: JSON Loading and Validation** — Completed & Approved
- **Phase 4: Static Pygame Visualization** — Completed & Approved
- **Phase 5: Integration Testing and Sign-off** — Completed (92/92 tests passing)

## Documentation
- [Design Specification & Roadmap](docs/design_spec_week1.md)
- [Setup Guide](docs/setup_guide.md)

## Environment Model & JSON Serialization

The environment model provides domain primitives (`Point`, `Rectangle`, `LineSegment`, `Wall`, `Doorway`, `Obstacle`, `Exit`, `Building`) and standard-library JSON serialization routines.

### Quick Usage Example

```python
from genevac.environment import (
    create_initial_building,
    save_building,
    load_building,
    SCHEMA_VERSION,
)

# 1. Create the validated initial 40m x 30m building layout
building = create_initial_building()

# 2. Save the building to a formatted JSON file
save_building(building, "maps/building_main.json")

# 3. Load and reconstruct the validated Building instance from JSON
reconstructed = load_building("maps/building_main.json")
assert reconstructed.id == building.id
```

### Schema Versioning
- The JSON configuration format embeds a top-level `"schema_version"` string (current version: `"1.0"`).
- The version constant is defined as `SCHEMA_VERSION` in `genevac.environment.serialization` and re-exported by `genevac.environment`.
- Unsupported or missing schema versions are strictly rejected during deserialization.

## Static 2D Visualization (Pygame)

Phase 4 implements a clean, top-down 2D Pygame visualization that renders actual `Building` instances with aspect-ratio preserving scaling, centered viewports, and clear color-coding.

### Launching the Visualization

Activate your virtual environment, then run the visualizer:

```bash
# Visualize the default 40m x 30m initial building layout
python -m genevac.visualization.app

# Visualize a building loaded from a custom JSON configuration
python -m genevac.visualization.app --json maps/building_main.json

# Specify custom initial window dimensions (default: 1200 x 800)
python -m genevac.visualization.app --width 1024 --height 768
```

### Interactive Controls
- **Window Resizing**: Drag window borders to resize. The renderer dynamically recomputes viewport scaling, preserving the 4:3 building proportions and keeping the layout centered.
- **`R`**: Reset window to default dimensions.
- **`ESC` / `Q`**: Cleanly close the window and exit.

### Visual Elements
- **Walkable Floor**: Soft neutral tone for rooms, distinct lighter tone for the central corridor.
- **Walls**: Solid dark slate rectangles reflecting actual 2D physical footprints with 0.2 m thickness.
- **Doorways**: Vivid cyan portal threshold lines with jamb markers.
- **Exits**: High-visibility safety green threshold bands with `EXIT` badges.
- **Obstacles**: Warm amber filled rectangles with contrasting borders.
- **HUD & Legend**: Header status with metric dimensions, a 10-meter physical scale bar, and an element legend.

## Quick Start
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -v
```