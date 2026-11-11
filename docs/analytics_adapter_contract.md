# Analytics Adapter Contract & Field Mapping Specification

**Project:** GenEvac-Sentinel  
**Document:** `docs/analytics_adapter_contract.md`  
**Author:** Harsha (Person B / Module B Part 2)  
**Date:** 2026-11-09  

---

## 1. Concrete Source-to-Canonical Field Mapping Table

| Engine Source Field | Actual Source Type | Canonical Field | Canonical Type | Conversion / Normalization | Missing-Field Policy | Verified By |
|---|---|---|---|---|---|---|
| `snapshot.snapshot_id` | `str` (e.g. `"snap_000004"`) | `snapshot_id` (metadata) | `str` | Direct copy if string; otherwise `f"snap_{step:06d}"` | Default generated from `step_count` | `test_simulation_adapter.py` |
| `snapshot.step_count` / `engine.current_step` | `int` ($\ge 0$) | `step_index` / `current_step` | `int` | Direct integer validation | Required; raise `TrajectoryValidationError` if missing | `test_data_contract.py` |
| `snapshot.timestamp` | `float` (seconds $\ge 0$) | `timestamp` / `current_timestamp` | `float` | Finite float validation; derived as `step * dt` if omitted | Required; derived from `step * dt` | `test_timestamp_consistency.py` |
| `snapshot.dt` / `engine.timestep` | `float` ($> 0$, seconds) | `timestep` | `float` | Finite positive float | Default to `0.1` s if absent | `test_simulation_adapter.py` |
| `ped.pedestrian_id` / `ped.id` | `str` | `agent_id` | `str` | `str(id).strip()`, non-empty | Required; raise `TrajectoryValidationError` if empty | `test_data_contract.py` |
| `ped.x`, `ped.y` / `ped.position` | `float` or `(float, float)` | `x`, `y` | `float`, `float` | Finite float validation; meters, top-left frame | Required; reject malformed record | `test_data_contract.py` |
| `ped.vx`, `ped.vy` | `float`, `float` (m/s) | `vx`, `vy` (optional) | `Optional[float]` | Finite float if present | Optional; defaults to `None` | `test_simulation_adapter.py` |
| `ped.status` | `str` or `PedestrianStatus` enum | `status` (cohort partition) | `str` (`"active"`, etc.) | `.strip().lower()` mapped to adapter cohort sets | Required; unknown status raises `TrajectoryValidationError` | `test_simulation_adapter.py` |
| `ped.target_exit_id` | `Optional[str]` | `target_exit_id` | `Optional[str]` | Direct string if present | Optional; `None` if unassigned | `test_simulation_adapter.py` |

---

## 2. Ingestion & Validation Policy
1. **Zero Silent Coercion**: Coordinates, steps, and timestamps with `NaN`, `Inf`, or invalid types are never converted to zero; they are quarantined and logged in `SnapshotAdaptationReport`.
2. **Deduplication**: Re-submitting an identical snapshot produces `0` new observations and does not duplicate history samples.
3. **Run Isolation**: Step rollbacks or engine instance replacements automatically invoke `adapter.reset()` to clear all cohorts and history buffers.

