# Phase 05 Metric Selection & Live Display Matrix

**Project:** GenEvac-Sentinel  
**Document:** `docs/phase05_metric_selection.md`  
**Date:** 2026-11-09  

---

## 1. Core Live Analytics Metrics

| Metric ID | Display Name | Unit | Source Module / Function | Inclusion / Cohort Rule | Status |
|---|---|---|---|---|---|
| `active_count` | Active Pedestrians | Count | `adapter.active_agents` | Length of active cohort set | **INCLUDED** |
| `blocked_count` | Blocked Pedestrians | Count | `adapter.blocked_agents` | Length of blocked cohort set | **INCLUDED** |
| `evacuated_count` | Evacuated Pedestrians | Count | `adapter.evacuated_agents` | Length of evacuated cohort set (explicit only) | **INCLUDED** |
| `not_observed_count` | Absent / Not Observed | Count | `adapter.not_observed_agents` | Missing from snapshot without explicit evacuation | **INCLUDED** |
| `clearance_time` | Total Elapsed Time | Seconds ($s$) | `metrics.calculate_clearance_time` | Simulation time derived as `step * dt` | **INCLUDED** |
| `min_separation` | Minimum Pedestrian Separation | Meters ($m$) | `metrics.calculate_minimum_separation` | Pairwise distance across active agents | **INCLUDED** |
| `exit_throughput` | Instantaneous Exit Flow | $peds / s$ | `metrics.calculate_exit_throughput` | Rate of exit events over sliding window | **INCLUDED** |
| `occupancy_grid` | Spatial Cell Occupancy | Count / cell | `SpatialHeatmapEngine.occupancy_grid` | Raw integer count per cell | **INCLUDED** |
| `density_grid` | Physical Crowd Density | $peds / m^2$ | `SpatialHeatmapEngine.density_grid` | $\text{Occupancy} / \text{CellArea}$ | **INCLUDED** |
| `dwell_time_grid` | Accumulated Dwell Time | $agent \cdot s$ | `SpatialHeatmapEngine.dwell_time_grid` | Sample-hold interval accumulation | **INCLUDED** |

---

## 2. Blocked / Conditional Metrics

| Metric | Reason for Block / Conditional Status |
|---|---|
| **ERSI (Route Safety Index)** | Formula, factor weights, and normalization undefined in project specification. Marked **BLOCKED**. |
| **Delta Score** | Mathematical formulation undefined in project specification. Marked **BLOCKED**. |
| **Prediction ADE / FDE** | Displayed only when future ground truth is available; marked conditional. |

