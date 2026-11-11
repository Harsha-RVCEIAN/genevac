# Phase 05 Source Inventory: Live Analytics & Charts Pipeline

**Project:** GenEvac-Sentinel  
**Document:** `docs/phase05_source_inventory.md`  
**Author:** Harsha (Person B / Module B Part 2)  
**Date:** 2026-11-09  

---

## 1. Components Inspected & Audit

| Component | Source Path | Existing Symbols | Phase 05 Role | Action |
|---|---|---|---|---|
| **Simulation Ingestion Adapter** | `src/genevac/prediction/adapter.py` | `SimulationTrajectoryAdapter`, `SnapshotAdaptationReport` | Provides read-only canonical `TrajectorySnapshot` without engine mutation. | **REUSE** |
| **Metrics Engine** | `src/genevac/metrics/` | `calculate_ade`, `calculate_fde`, `calculate_clearance_time`, `calculate_minimum_separation`, `calculate_area_density`, `calculate_exit_throughput` | Pure calculation functions providing core metrics. | **REUSE** |
| **Spatial Heatmap Engine** | `src/genevac/analytics/heatmap.py` | `SpatialHeatmapEngine`, `GridSpecification`, `SpatialGridResult` | Provides validated occupancy, physical density ($peds/m^2$), and dwell-time arrays. | **REUSE** |
| **Visualizer** | `src/genevac/visualization/__init__.py` | `plot_heatmap`, `plot_trajectories` | Matplotlib renderers for figures. | **REUSE** |
| **Live Pipeline & Series Builder** | Not present | None | Converts live snapshots into structured `LiveAnalyticsSnapshot` and `ChartSeries` collections with ordering and deduplication. | **IMPLEMENT** in `src/genevac/analytics/live.py` |

---

## 2. Target Public Symbols to Add
- `LiveAnalyticsSnapshot`: Immutable snapshot of metrics, heatmap results, run/step/timestamp metadata.
- `MetricRecord`: Individual metric value with unit, status, sample count, and timestamps.
- `ChartPoint` & `ChartSeries`: Renderer-neutral time-series data with duplicate suppression and bounded history.
- `LiveAnalyticsPipeline`: Read-only pipeline orchestrating adapter, metrics, heatmap engine, and series updates.

