# Building Environment Maps

This directory contains JSON configurations representing floor plans and environments for GenEvac-Sentinel.

## Included Map Files
- **`building_main.json`**: The canonical $40.0\text{ m} \times 30.0\text{ m}$ educational building layout featuring:
  - 4 rooms (`room_1`, `room_2`, `room_3`, `room_4`)
  - 1 central horizontal corridor (`main_corridor`, $3.0\text{ m}$ width)
  - 4 doorway portals connecting each room to the corridor ($2.0\text{ m}$ width)
  - 3 exterior boundary exits (`exit_left`, `exit_right`, `exit_bottom`)
  - 4 static interior furniture obstacles ($2.0\text{ m} \times 1.5\text{ m}$)
  - Full $0.2\text{ m}$ wall footprints

## Schema Specification
All JSON map files conform to schema version `"1.0"`. Deserialization verifies:
- Top-level `"schema_version": "1.0"`
- Geometry coordinates in continuous metres
- Non-overlapping boundary and doorway constraints

## Usage Examples

### Loading a Map
```python
from genevac.environment.serialization import load_building

building = load_building("maps/building_main.json")
print(f"Loaded {building.id}: {building.width}m x {building.height}m")
```

### Visualizing a Map
```bash
PYTHONPATH=src .venv/bin/python3 -m genevac.visualization.app --json maps/building_main.json
```

