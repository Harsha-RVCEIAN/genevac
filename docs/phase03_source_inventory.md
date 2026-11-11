# Phase 03 Source Inventory: Simulation Engine vs. Analytics Data Adapter

**Project:** GenEvac-Sentinel  
**Workstream:** Module B Part 2 (Person B / Harsha)  
**Date:** 2026-11-09  

---

## 1. Source Inventory Matrix

| Concern | Actual Source Path / Symbol | Actual Field / API | Existing Adapter Behavior | Gap | Status |
|---|---|---|---|---|---|
| **Snapshot Retrieval** | `SimulationEngine.get_snapshot()` or `engine.latest_snapshot` | Returns `SimulationObservationSnapshot` (or `None`) | Calls `update_from_simulation(engine)` polling live engine attributes | Does not directly accept pre-built `SimulationObservationSnapshot` objects | **IMPLEMENT** |
| **Snapshot Identity** | `SimulationObservationSnapshot.snapshot_id` | `snapshot_id: str` (e.g. `"snap_000004"`) | Uses `(current_step, current_timestamp)` internally | `snapshot_id` not recorded on canonical snapshot | **IMPLEMENT** |
| **Simulation Step** | `SimulationObservationSnapshot.step_count` / `engine.current_step` | `step_count: int` / `current_step: int` | Maps to `step_index` in `TrajectoryObservation` | None, fully supported | **REUSE** |
| **Simulation Timestamp** | `SimulationObservationSnapshot.timestamp` / `clock.timestamp` | `timestamp: float` (seconds, simulation-relative) | Maps to `timestamp` in `TrajectoryObservation` | None, fully supported | **REUSE** |
| **Simulation Timestep** | `SimulationObservationSnapshot.dt` / `engine.timestep` | `dt: float` (seconds, default 0.1s) | Maps to `snapshot.timestep` | None, fully supported | **REUSE** |
| **Observation Cadence** | `SimulationObservationSnapshot.observation_interval` | `observation_interval: float` (default 0.4s) | Configured in `SimulationTrajectoryAdapter(observation_interval=0.4)` | None, fully supported | **REUSE** |
| **Pedestrian Observation Collection** | `SimulationObservationSnapshot.pedestrians` / `engine.pedestrians` | `tuple[PedestrianObservation, ...]` / `List[Pedestrian]` | Iterates `pedestrians` | Does not support reading from `PedestrianObservation` named attributes (`pedestrian_id`, `x`, `y`) | **IMPLEMENT** |
| **Pedestrian Coordinates** | `PedestrianObservation.x, y` / `ped.position` | `x: float, y: float` / `position: Tuple[float, float]` | Maps to `obs.x, obs.y` in meters (top-left origin) | None, coordinates aligned | **REUSE** |
| **Pedestrian Velocity** | `PedestrianObservation.vx, vy, speed` | `vx: float, vy: float, speed: float` | `TrajectoryObservation` only stores `(x, y)` | Optional velocity not forwarded to canonical observation | **EXTEND (Backward-Compatible)** |
| **Pedestrian Status** | `PedestrianObservation.status` / `ped.status` | `status: str` (`"active"`, `"evacuated"`, `"blocked"`, `"inactive"`) | Adapter partitions into `active_agents`, `blocked_agents`, `evacuated_agents`, `inactive_agents` | Status preserved in adapter cohorts, but missing structured conversion reports | **EXTEND** |
| **Reset Detection** | `SimulationEngine` instance `id(engine)` & step rollback | Monotonic check | Detects `id(engine)` change and `current_step < last_step`, resetting buffer | None, tested and verified | **REUSE** |
| **Deduplication** | Repeated step / timestamp poll | No step advancement | Skips duplicate step idempotently | Support explicit snapshot-id deduplication | **EXTEND** |

