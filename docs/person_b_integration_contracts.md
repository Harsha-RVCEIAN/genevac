# GenEvac-Sentinel: Person B Integration Contracts

**Document:** `docs/person_b_integration_contracts.md`  
**Status:** Source-Verified Audit (Phase 1)  
**Author:** Harsha (Person B / Module B Part 2)  
**Upstream Integration:** Pavan (Module B Part 1)  

---

## 1. Engine Interface Evidence (Pavan Reference vs. Local Engine)

### 1.1 Local Engine Interface (`src/genevac/simulation/__init__.py`)
- **Engine class:** `SimulationEngine(environment, max_steps, timestep, seed)`
- **Step method:** `step() -> None` (increments `current_step` by 1)
- **Pedestrian collection:** `pedestrians: List[Pedestrian]`
- **Pedestrian attributes:**
  - `id: str` (e.g. `"ped_1"`)
  - `position: Tuple[float, float]` (meters, top-left origin)
  - `velocity: Tuple[float, float]` (m/s)
  - `status: PedestrianStatus` (`ACTIVE`, `EVACUATED`, `INCAPACITATED`)
  - `update_position(dt: float)`

### 1.2 Pavan Full Reference Engine (`pavan-part1-reference/src/genevac/simulation/engine.py`)
- **Engine class:** `SimulationEngine(building, registry, dt=0.1, observation_interval=0.4, block_unassigned_exits=True)`
- **Clock:** `SimulationClock(dt=0.1, observation_interval=0.4)`:
  - `dt`: 0.1 s
  - `observation_interval`: 0.4 s (stride = 4)
  - `step_count`: monotonic integer
  - `timestamp`: derived as `step_count * dt`
- **Snapshot interface:**
  - `engine.get_snapshot() -> SimulationObservationSnapshot`
  - `engine.latest_snapshot: Optional[SimulationObservationSnapshot]`
  - `engine.step() -> Optional[SimulationObservationSnapshot]` (returns snapshot every 4 steps)
  - `SimulationObservationSnapshot` fields:
    - `snapshot_id: str`
    - `step_count: int`
    - `timestamp: float`
    - `dt: float`
    - `observation_interval: float`
    - `total_count: int`, `active_count: int`, `evacuated_count: int`, `blocked_count: int`, `inactive_count: int`
    - `pedestrians: tuple[PedestrianObservation, ...]`
- **Pedestrian observation shape:**
  - `pedestrian_id: str`
  - `x: float`, `y: float` (meters)
  - `vx: float`, `vy: float`, `speed: float` (m/s)
  - `status: str` (`"active"`, `"evacuated"`, `"blocked"`, `"inactive"`)
  - `target_exit_id: Optional[str]`
  - `trajectory_history: tuple[TimestampedTrajectoryPoint, ...]`

---

## 2. Contract Compatibility Matrix

| Contract Item | Engine Source Behavior | Person B Adapter Expectation | Compatible? | Action Required |
|---|---|---|---|---|
| **Position Coordinates** | `(x, y)` floats in meters | `x: float, y: float` in meters | **Yes** | Fully compatible; direct mapping. |
| **Coordinate Frame** | Top-left origin `(0, 0)`, $+X$ right (east), $+Y$ down (south) | Same convention defined in `TrajectoryObservation` docstring | **Yes** | Fully aligned. |
| **Step Counter** | `current_step: int` or `step_count: int` ($\ge 0$) | `step_index: int` ($\ge 0$, monotonic) | **Yes** | Checked and validated. |
| **Simulation Time** | `current_step * timestep` (or `clock.timestamp`) | `timestamp: float` ($\ge 0$, monotonic) | **Yes** | Derived directly to avoid drift. |
| **Observation Cadence** | Simulator updates at 10 Hz ($dt=0.1$ s); snapshots at 2.5 Hz ($0.4$ s) | Configurable `observation_interval=0.4` | **Yes** | Cadence filter verified by tests. |
| **Pedestrian Identity** | String ID (`ped.id` or `ped.pedestrian_id`) | Non-empty `str` (`agent_id`) | **Yes** | Preserves ID across history. |
| **Pedestrian Status** | `ACTIVE`, `BLOCKED`, `EVACUATED`, `INCAPACITATED`, `INACTIVE` | Explicit cohort partitioning in adapter | **Yes** | EVACUATED is terminal; INACTIVE ignored. |
| **Run Identity** | Engine does not expose persistent `run_id` | Adapter detects reset on instance change or step rollback | **Compatible with Adapter Guarantee** | Propose standard `run_id` in Phase 6/8. |
| **Scenario Metadata** | Building / EnvironmentMap parameters | Immutably captured in experiment runner | **Compatible** | Formalize schema in Phase 6. |

---

## 3. Contract Decisions & Resolutions

| Decision | Source Evidence | Resolution Status |
|---|---|---|
| **Spatial Units** | `Point` and `EnvironmentMap` define coordinates in meters. | **RESOLVED:** Meters ($m$). |
| **Coordinate Origin** | Documented top-left origin: $+X$ East, $+Y$ South. | **RESOLVED:** Top-left origin. |
| **Temporal Units** | `dt = 0.1` s, `observation_interval = 0.4` s. | **RESOLVED:** Seconds ($s$), simulation time. |
| **Step Cadence** | Fixed discrete steps of $0.1$ s. | **RESOLVED:** Fixed $dt = 0.1$ s. |
| **Identity Stability** | Pedestrian IDs persist across steps within a run. | **RESOLVED:** Scoped to single simulation run. |
| **Lifecycle** | Transitions: `ACTIVE -> EVACUATED` (terminal), `ACTIVE -> BLOCKED`. | **RESOLVED:** Handled by adapter cohort logic. |
| **MID Status** | `research/MID` requires missing checkpoints and CUDA. | **RESOLVED / BLOCKED:** Quarantined, gated. |

