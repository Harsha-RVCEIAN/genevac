# Path and Route Check Catalogue

**Project:** GenEvac-Sentinel  
**Module:** `src/genevac/metrics/route_checks.py`  
**Author:** Harsha (Person B / Module B Part 2)  
**Date:** 2026-11-09  

---

## 1. Overview
The route and path validation engine performs 13 deterministic geometric, topological, and spatial safety checks on pedestrian evacuation routes. Each check produces a structured `PathCheckResult` containing:
- `check_id`: Canonical identifier (e.g., `RC01_START_VALIDITY`)
- `status`: `PASS`, `FAIL`, or `UNKNOWN`
- `reason`: Descriptive explanation
- `details`: Diagnostic coordinates, distances, or element IDs
- `is_static`: True for fixed geometry; False for time-varying dynamic data

---

## 2. Specification of 13 Canonical Path Checks

| Check ID | Check Name | Description | Failure Condition | Output Details |
|---|---|---|---|---|
| `RC01_NON_EMPTY` | Non-Empty Path | Verifies path has at least 2 points ($N \ge 2$). | $N < 2$ | `{"point_count": N}` |
| `RC02_FINITE_COORDS` | Finite Coordinates | All $(x, y)$ coordinate values must be finite numbers. | `NaN`, `Inf`, or non-numeric | `{"first_invalid_index": idx}` |
| `RC03_WORLD_BOUNDS` | World Bounds Containment | All path points lie inside `[0, width] x [0, height]`. | Any point outside bounds | `{"point_index": idx, "pos": (x,y)}` |
| `RC04_NO_CONSECUTIVE_DUPLICATES` | No Zero-Length Segments | Consecutive points must have non-zero displacement ($\|q_{i+1} - q_i\| > \epsilon$). | $\|q_{i+1} - q_i\| \le \epsilon$ | `{"segment_index": i}` |
| `RC05_STEP_DISTANCE_LIMIT` | Step Distance Continuity | Segments must not exceed maximum plausible step distance (teleportation guard). | $\|q_{i+1} - q_i\| > \text{max\_step}$ | `{"segment_index": i, "length": L}` |
| `RC06_START_VALIDITY` | Start Point Validity | Start point $q_0$ must lie within declared start region / walkable space. | Outside valid starting region | `{"start_pos": (x, y)}` |
| `RC07_DESTINATION_VALIDITY` | Destination Validity | Final point $q_n$ must reach the target exit zone within tolerance. | Distance to exit $> \text{exit\_tol}$ | `{"exit_id": id, "dist_to_exit": d}` |
| `RC08_OBSTACLE_INTERSECTION` | Static Obstacle Intersection | Path segments must not intersect any static obstacle polygons or walls. | Segment intersects boundary | `{"obstacle_id": id, "segment": i}` |
| `RC09_WALKABLE_SPACE` | Walkable Space Clearance | All waypoints must have valid clearance ($\ge \text{ped\_radius}$) from obstacles. | Clearance $< \text{radius}$ | `{"min_clearance": d, "pos": (x,y)}` |
| `RC10_ROUTE_LENGTH` | Route Length & Efficiency | Computes geometric length $L = \sum \|q_{i+1} - q_i\|_2$ and detour ratio $L / L_{\text{euclid}}$. | Detour ratio exceeds max limit | `{"length_m": L, "detour_ratio": r}` |
| `RC11_SELF_INTERSECTION` | Loop & Self-Intersection Check | Path should not loop back on itself or produce redundant cycles. | Non-adjacent segments intersect | `{"intersecting_segments": (i, j)}` |
| `RC12_STATIC_CLEARANCE` | Minimum Static Clearance | Measures closest approach to any wall or obstacle along entire path. | Distance $< \text{min\_clearance}$ | `{"min_clearance_m": d}` |
| `RC13_DYNAMIC_HAZARD_CLEARANCE` | Dynamic Hazard / Crowd Clearance | Evaluates distance to dynamic hazard zones or high-density cells if dynamic data is supplied. | Route enters active hazard | `{"hazard_id": id, "dist": d}` |

