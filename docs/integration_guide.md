# GenEvac-Sentinel Simulation-to-Prediction Integration Guide

This guide specifies the operational handshake between the **Simulation Engine (Person A)** and the **Prediction Subsystem (Person B)**.

---

## 1. Adapter Sequence: Step-by-Step

The integration between `SimulationEngine` and `TrajectoryHistoryBuffer` follows an explicit feed pattern:

```python
from genevac.simulation import SimulationEngine
from genevac.environment import EnvironmentMap
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.prediction import (
    TrajectoryHistoryBuffer,
    TrajectoryCategory,
    ConstantVelocityPredictor,
)

# 1. Initialize simulation engine and trajectory buffer
env = EnvironmentMap(width=50.0, height=50.0)
engine = SimulationEngine(environment=env, timestep=0.1)
buffer = TrajectoryHistoryBuffer(
    window_size=20,
    min_history_required=2,
    max_observation_age=0.2,  # Configurable freshness window (seconds)
)

# 2. Add pedestrians to the simulation engine
ped1 = Pedestrian(id="ped_01", position=(5.0, 10.0), velocity=(1.0, 0.0))
engine.add_pedestrian(ped1)

# 3. Simulation Loop with Authoritative Observation Feed
for step_idx in range(10):
    # a. Advance engine state (Person A authority)
    engine.step()

    # b. Derive simulation time: this engine currently exposes current_step and
    #    timestep, not a separate current_timestamp property.
    sim_time = engine.current_step * engine.timestep

    # c. Feed active pedestrian observations into the history buffer
    for ped in engine.pedestrians:
        if ped.status == PedestrianStatus.ACTIVE:
            buffer.record(
                observation_or_agent_id=ped.id,
                position=ped.position,
                step_index=engine.current_step,
                timestamp=sim_time,
                category=TrajectoryCategory.SIMULATED,
            )

# 4. Snapshot Generation (Read-only view)
# The snapshot timestamp is derived from the current fixed-timestep engine state.
# The snapshot timestamp is derived from the current fixed-timestep engine state.
cur_sim_time = engine.current_step * engine.timestep
snapshot = buffer.get_snapshot(
    current_step=engine.current_step,
    current_timestamp=cur_sim_time,
    timestep=engine.timestep,
    agent_ids=["ped_01"],  # Or None for all agents in buffer
)

# 5. Prediction Execution
predictor = ConstantVelocityPredictor()
prediction = predictor.predict(snapshot=snapshot, agent_id="ped_01", horizon=10)
```

---

## 2. Timestamp Semantics & Irregular Intervals

1. **Snapshot Timestamp (`snapshot.current_timestamp`)**:
   - The current `SimulationEngine` exposes `current_step` and `timestep`, but does not expose an independent `current_timestamp` property.
   - Under the current fixed-timestep engine, derive the snapshot time as $t_{\text{current}} = \text{current\_step} \times \text{timestep}$.
   - Pass that derived value to `TrajectoryHistoryBuffer.get_snapshot(current_timestamp=...)`.
   - This derivation is valid for the current fixed-timestep engine; if variable timesteps are introduced, the authoritative clock contract must be revised.
   - It is independent of individual pedestrian update times.
2. **Observation Timestamp (`obs.timestamp`)**:
   - Each observation preserves the exact timestamp when it was observed or recorded.
   - Observations for a specific pedestrian need not equal `snapshot.current_timestamp` if the pedestrian skipped steps or was observed asynchronously.
3. **Irregular Observation Intervals**:
   - Observations can arrive at irregular intervals (e.g., step 0 at $t=0.0\text{ s}$, step 5 at $t=0.5\text{ s}$).
   - `ConstantVelocityPredictor` computes velocity based on actual observation timestamps:
     $$v_x = \frac{x_2 - x_1}{t_2 - t_1}, \quad v_y = \frac{y_2 - y_1}{t_2 - t_1}$$
   - Predictions project forward starting from the snapshot reference timestamp:
     $$t_{\text{future}} = t_{\text{snapshot}} + k \times \Delta t_{\text{pred}}$$
     $$p_{\text{future}} = p_{\text{latest}} + \vec{v} \cdot (t_{\text{future}} - t_{\text{latest}})$$

---

## 3. Freshness Validation (`max_observation_age`)

To prevent stale observations from silently producing seemingly valid predictions:

1. **Configuration**:
   - `TrajectoryHistoryBuffer(max_observation_age: float | None = None)` accepts a maximum permitted elapsed time $\ge 0.0\text{ s}$.
   - Can also be overridden at validation/snapshot time: `buffer.get_snapshot(..., max_observation_age=0.2)`.
   - Can be enforced directly in predictors: `predictor.predict(..., max_observation_age=0.2)`.
2. **Freshness Rule**:
   - For an observation to be fresh at snapshot reference timestamp $t_{\text{snapshot}}$:
     $$\text{age} = t_{\text{snapshot}} - t_{\text{latest\_obs}} \le \text{max\_observation\_age}$$
   - If $\text{age} > \text{max\_observation\_age}$, a `TrajectoryValidationError` is raised with `field="observation_age"`.
3. **Freshness vs. History Length**:
   - `has_sufficient_history(agent_id)` checks *only* whether $\text{count}(\text{observations}) \ge \text{min\_history\_required}$.
   - Freshness is checked independently via `is_latest_observation_fresh(agent_id, reference_timestamp)` and snapshot/prediction validation.

---

## 4. Lifecycle Handling (Disappearance, Evacuation, Incapacitation)

1. **Preserving History**:
   - When a pedestrian evacuates (`PedestrianStatus.EVACUATED`), becomes incapacitated (`PedestrianStatus.INCAPACITATED`), or stops appearing, their past observations remain in `TrajectoryHistoryBuffer`.
   - The buffer **never** automatically purges history upon agent disappearance.
2. **Terminal States vs. Active Agents**:
   - The simulation loop should only record active agents (`ped.status == PedestrianStatus.ACTIVE`).
   - Inactive agents naturally stop receiving new observations.
3. **Preventing Stale Prediction**:
   - If snapshot generation or batch prediction is run across all agents in the buffer, stale agents (whose latest observation age exceeds `max_observation_age`) are rejected with `TrajectoryValidationError`.
   - Callers should pass active `agent_ids` to `get_snapshot()` or filter predictions.
4. **Explicit Retirement**:
   - An explicit call to `buffer.clear(agent_id)` can be made if and only if the caller explicitly decides to purge memory for that pedestrian.

---

## 5. Non-Mutation Guarantee

- `ConstantVelocityPredictor` takes an immutable `TrajectorySnapshot`.
- It never accesses or mutates `SimulationEngine`, `Pedestrian`, or `EnvironmentMap` instances.
- Prediction results are tagged with `TrajectoryCategory.PREDICTED` and are strictly forbidden from being fed back into the authoritative history buffer.

