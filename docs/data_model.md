# Data Models & Contracts

This document outlines the primary data structures used across Genevac.

## 1. Pedestrian State
- `id`: `int` or `str` - Unique agent identifier.
- `position`: `Tuple[float, float]` - Cartesian coordinates `(x, y)` in meters.
- `velocity`: `Tuple[float, float]` - Velocity vector `(vx, vy)` in m/s.
- `target_exit`: `Optional[str]` - Target exit ID.
- `desired_speed`: `float` - Baseline walking speed (default ~1.2 - 1.5 m/s).
- `radius`: `float` - Physical personal space boundary (default ~0.25 - 0.3 m).
- `status`: `str` - One of `ACTIVE`, `EVACUATED`, `INCAPACITATED`.

## 2. Trajectory History Contract
- `history_window`: Sliding window of past `T_obs` timesteps.
- Each observation record contains:
  - `timestamp`: Float timestamp or integer step index.
  - `agent_id`: Agent identifier.
  - `position`: `(x, y)` coordinates.
  - `velocity`: `(vx, vy)`.

## 3. Scenario Definition
Scenarios are serialized as JSON:
```json
{
  "name": "hallway_evacuation",
  "map_dimensions": [50.0, 20.0],
  "exits": [
    {"id": "exit_east", "position": [50.0, 10.0], "width": 2.5}
  ],
  "obstacles": [
    {"id": "pillar_1", "shape": "box", "bounds": [20.0, 8.0, 24.0, 12.0]}
  ],
  "pedestrian_count": 50,
  "timestep": 0.1,
  "max_steps": 1000
}
```
