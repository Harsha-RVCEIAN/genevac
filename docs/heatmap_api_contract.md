# Heatmap API Contract and Phase 05 Rendering Interface

**Project:** GenEvac-Sentinel  
**Document:** `docs/heatmap_api_contract.md`  
**Date:** 2026-11-09  

---

## 1. Public API Interface
- Module: `src/genevac/analytics/heatmap.py` (re-exported via `src/genevac/analytics`)
- Classes:
  - `GridSpecification(min_x, min_y, max_x, max_y, cell_size_x, cell_size_y)`: Immutable grid metadata.
  - `SpatialGridResult`: Container for 2D numpy arrays (`occupancy`, `density`, `dwell_time`), grid geometry, timestamp, sample counts, and diagnostics.
  - `SpatialHeatmapEngine(grid_spec)`: Calculation engine providing:
    - `map_point_to_cell(x, y) -> Optional[Tuple[int, int]]`
    - `aggregate_snapshot(snapshot, cohort="active") -> SpatialGridResult`
    - `accumulate_dwell_time(dt, snapshot) -> SpatialGridResult`
    - `reset()`: Run-scoped state reset.

---

## 2. Phase 05 Rendering Contract
1. Phase 05 visualizer consumes `SpatialGridResult` directly as numeric 2D arrays.
2. The renderer maps numeric values to colors without altering the underlying physical quantities.
3. The renderer never executes simulation steps or alters engine objects.

