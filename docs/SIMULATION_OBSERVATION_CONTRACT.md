# GenEvac-Sentinel: Simulation-to-Prediction Observation Contract

## 1. Overview and Purpose

This document defines the strict, authoritative data contract between Person A's simulation engine and Person B's downstream prediction and evaluation pipeline.

The contract ensures that:
1. Person B receives complete, high-fidelity physical state data for all agents.
2. The simulation engine remains the authoritative source of ground truth.
3. Person B cannot directly mutate or corrupt the running simulation state through the observation objects.
4. Downstream consumers have zero mandatory dependencies on Pygame, PyTorch, MID, or NumPy.

---

## 2. Emission Cadence and Lifecycle

- **Physics Timestep ($\Delta t$)**: $0.1\text{ s}$ ($10\text{ Hz}$).
- **Observation Interval**: $0.4\text{ s}$ ($2.5\text{ Hz}$), corresponding to every 4 physics steps.
- **Emission Method**:
  - `SimulationEngine.step() -> Optional[SimulationObservationSnapshot]`:
    - Returns a new `SimulationObservationSnapshot` if and only if the current step falls on an observation tick (`step % 4 == 0`).
    - Returns `None` on intermediate physics substeps.
  - `SimulationEngine.get_snapshot() -> SimulationObservationSnapshot`:
    - Generates an on-demand immutable snapshot at any arbitrary point in time.

---

## 3. Schema and Field Specifications

All observation classes are defined in `src/genevac/simulation/observation.py`.

### 3.1 SimulationObservationSnapshot

The top-level container emitted at each observation tick:

```python
@dataclass(frozen=True)
class SimulationObservationSnapshot:
    snapshot_id: str
    step_count: int
    timestamp: float
    dt: float
    observation_interval: float
    total_count: int
    active_count: int
    evacuated_count: int
    blocked_count: int
    inactive_count: int
    pedestrians: tuple[PedestrianObservation, ...]
```

| Field Name | Type | SI Units | Description & Invariants |
| :--- | :--- | :--- | :--- |
| `snapshot_id` | `str` | N/A | Step-derived identifier string: `f"{snapshot_prefix}_{step_count:06d}"` (e.g. `"snap_000004"`). Monotonic and unique within a continuous run; repeats from step zero upon application reset. Not globally unique across separate runs. |
| `step_count` | `int` | dimensionless | Simulation physics step index ($\ge 0$). |
| `simulation_step`| `int` | dimensionless | Property alias for `step_count`. |
| `timestamp` | `float` | seconds ($\text{s}$) | Elapsed simulation time ($\ge 0.0\text{ s}$). |
| `dt` | `float` | seconds ($\text{s}$) | Physics integration timestep ($0.1\text{ s}$). |
| `observation_interval` | `float` | seconds ($\text{s}$) | Scheduled cadence between observations ($0.4\text{ s}$). |
| `total_count` | `int` | dimensionless | Total registered population ($N$). Must equal the length of `pedestrians`. |
| `active_count` | `int` | dimensionless | Number of agents in `active` state. |
| `evacuated_count`| `int` | dimensionless | Number of agents in `evacuated` state. |
| `blocked_count` | `int` | dimensionless | Number of agents in `blocked` state. |
| `inactive_count` | `int` | dimensionless | Number of agents in `inactive` state. |
| `pedestrians` | `tuple[PedestrianObservation, ...]` | N/A | Immutable tuple of individual agent observations. |

> **Invariant Check**:
> $$\text{total\_count} = \text{active\_count} + \text{evacuated\_count} + \text{blocked\_count} + \text{inactive\_count} = \text{len}(\text{pedestrians})$$
> Any violation of this equality raises a `ValueError` during snapshot creation.

### 3.2 PedestrianObservation

Individual agent observation record:

```python
@dataclass(frozen=True)
class PedestrianObservation:
    pedestrian_id: str
    x: float
    y: float
    vx: float
    vy: float
    speed: float
    status: str
    target_exit_id: Optional[str]
    trajectory_history: tuple[TimestampedTrajectoryPoint, ...]
```

| Field Name | Type | SI Units | Description & Invariants |
| :--- | :--- | :--- | :--- |
| `pedestrian_id`| `str` | N/A | Stable unique agent string identifier (e.g. `"ped_1"`). |
| `x` | `float` | metres ($\text{m}$) | Current X coordinate in continuous world coordinates. |
| `y` | `float` | metres ($\text{m}$) | Current Y coordinate in continuous world coordinates. |
| `vx` | `float` | $\text{m/s}$ | Instantaneous velocity component along X-axis. |
| `vy` | `float` | $\text{m/s}$ | Instantaneous velocity component along Y-axis. |
| `speed` | `float` | $\text{m/s}$ | Instantaneous scalar speed ($\sqrt{v_x^2 + v_y^2}$). |
| `status` | `str` | N/A | Lowercase string: `"active"`, `"evacuated"`, `"blocked"`, `"inactive"`. |
| `target_exit_id`| `Optional[str]`| N/A | Designated exit identifier (or `None`). |
| `trajectory_history` | `tuple[TimestampedTrajectoryPoint, ...]` | N/A | Immutable tuple of recent trajectory samples. |

### 3.3 TimestampedTrajectoryPoint

Individual trajectory historical record:

```python
@dataclass(frozen=True)
class TimestampedTrajectoryPoint:
    step: int
    timestamp: float
    x: float
    y: float
```

| Field Name | Type | SI Units | Description |
| :--- | :--- | :--- | :--- |
| `step` | `int` | dimensionless | Physics step index at time of recording. |
| `timestamp` | `float` | seconds ($\text{s}$) | Elapsed simulation time at time of recording. |
| `x` | `float` | metres ($\text{m}$) | Physical X coordinate. |
| `y` | `float` | metres ($\text{m}$) | Physical Y coordinate. |

---

## 4. Immutability and Safety Guarantees

1. **Deep Frozen Dataclasses**: All classes (`SimulationObservationSnapshot`, `PedestrianObservation`, `TimestampedTrajectoryPoint`) are decorated with `@dataclass(frozen=True)`. Attempting to modify any field raises `FrozenInstanceError`.
2. **Immutable Collections**: All collection attributes (`pedestrians`, `trajectory_history`) are concrete immutable `tuple` instances, preventing append, remove, or in-place sorting mutations.
3. **Decoupled Memory**: Creating a snapshot copies live agent coordinates into new value objects. Subsequent movements of simulation agents do not modify previously emitted snapshots.

---

## 5. Agent Retention Policy

A critical contract guarantee:
- **Evacuated agents are NEVER removed from `pedestrians` in subsequent snapshots.**
- When an agent reaches an exit, their status changes to `"evacuated"`, velocity drops to `0.0`, and they continue to appear in every subsequent snapshot.
- This allows Person B's prediction evaluation pipeline to calculate total evacuation times, final trajectory lengths, and survival metrics without losing agent history.

---

## 6. Documented Limitations for Downstream Integration

### 6.1 Snapshot ID Semantics & Lack of `run_id`
- `snapshot_id` is derived from the configured snapshot prefix (default `"snap"`) and the simulation step count: `f"{snapshot_prefix}_{step_count:06d}"`.
- It is strictly monotonic and unique within a continuous simulation run when the step count advances.
- `SimulationEngine` itself does not expose a public reset method. Application reset is implemented via `SimulationApp.reset()`, which creates a new `SimulationEngine` instance with a fresh population.
- Because a newly created engine instance starts from step zero, snapshot IDs can repeat from step zero (`"snap_000000"`, `"snap_000004"`) across runs.
- There is no independent `run_id` or `session_id` in the observation schema.
- **Integration Requirement**: If Person B runs multiple simulation experiments or Monte Carlo trials, Person B must manage the run identifier externally (e.g., in their own wrapper class, file naming, or directory structure).

### 6.2 Rolling Trajectory Window vs Lifetime History
- Samples are appended to `trajectory_history` only on observation ticks ($0.4\text{ s}$ cadence) and at step 0 ($t=0.0\text{ s}$), NOT on intermediate $0.1\text{ s}$ physics steps.
- `max_history_len` defaults to `100`. At $2.5\text{ Hz}$ ($0.4\text{ s}$ stride), a full buffer of 100 samples spans 99 observation steps ($99 \times 0.4\text{ s} = 39.6\text{ s}$ between oldest and newest point, approximately $40.0\text{ seconds}$ of recent elapsed simulation time).
- If a simulation runs for $> 40.0\text{ seconds}$, older trajectory samples are evicted via FIFO `pop(0)`.
- **Integration Requirement**: If Person B needs complete lifetime trajectories for long runs, Person B should record incoming snapshots sequentially.

---

## 7. JSON Serialization

Both snapshot and observation models provide `.to_dict()` methods producing plain Python dictionaries composed strictly of JSON-serializable primitives (`str`, `int`, `float`, `None`, `list`, `dict`):

```python
import json

snapshot = engine.get_snapshot()
data_dict = snapshot.to_dict()
json_string = json.dumps(data_dict, indent=2)
```

