# Phase 04 Heatmap Engine Source Inventory

**Project:** GenEvac-Sentinel  
**Document:** `docs/phase04_heatmap_inventory.md`  
**Author:** Harsha (Person B / Module B Part 2)  
**Date:** 2026-11-09  

---

## 1. Existing Heatmap Codebase Audit

| Source File | Existing Symbols | Signatures & Types | Evaluation & Limitations | Proposed Action |
|---|---|---|---|---|
| `src/genevac/analytics/__init__.py` | `HeatmapCalculator` | `__init__(grid_width, grid_height, resolution=1.0)`<br>`record_positions(positions: List[Tuple[float, float]])`<br>`get_density_matrix() -> np.ndarray` | Basic matrix binning. Lacks: coordinate bounds `(min_x, min_y, max_x, max_y)`, floor-based index calculation with max-edge handling, out-of-bounds rejection diagnostics, distinct occupancy count vs. physical density layers ($peds/m^2$), and dwell-time accumulation. | **REFACTOR & EXPAND** in `src/genevac/analytics/heatmap.py` while preserving backwards-compatible wrapper in `src/genevac/analytics/__init__.py`. |
| `src/genevac/visualization/__init__.py` | `plot_heatmap` | `plot_heatmap(density_matrix: np.ndarray, title: str) -> plt.Figure` | Pure Matplotlib visualizer. Correctly separated from calculation logic. | **REUSE** in Phase 05 visualization. |
| `tests/test_heatmaps.py` | `TestHeatmaps` | `test_heatmap_density` | Verifies basic cell indexing on 10x10 grid with 2.0 resolution. | **EXTEND** with full Phase 04 grid and numerical test suite. |

---

## 2. Reusability and Strategy
1. Keep the numerical grid calculation completely independent of Matplotlib / Pygame.
2. Build an immutable, validated `GridSpecification` and `SpatialGridResult` with explicit layers:
   - `occupancy_count`: integer matrix $[row, col]$
   - `physical_density`: floating-point matrix $[row, col]$ ($peds / m^2$)
   - `dwell_time`: floating-point matrix $[row, col]$ ($agent \cdot s$)
3. Integrate with Phase 03 canonical `TrajectorySnapshot` and `SimulationTrajectoryAdapter`.

