# Phase 06 — Scenario and Simulation Engine Interface Inventory

**Document Version:** 1.0.0  
**Phase:** 06 — Scenario Comparison  
**Owner:** Module B Part 2 (Person B / Harsha)  
**Upstream Engine Owner:** Module B Part 1 (Pavan)  

---

## 1. Upstream Interface & Construction Paths

### 1.1 Local Checked-Out Engine (`src/genevac/simulation/__init__.py`)
The checked-out repository provides a clean, minimal, deterministic simulation engine:
```python
class SimulationEngine:
    def __init__(
        self,
        environment: EnvironmentMap,
        max_steps: int = 1000,
        timestep: float = 0.1,
        seed: int = 42,
    ):
        ...
```
* **Supported Construction Path:**
  1. Instantiate an `EnvironmentMap(width=..., height=...)`.
  2. Populate `EnvironmentMap.exits` via `env.add_exit(Exit(id=..., position=..., width=...))`.
  3. Populate `EnvironmentMap.obstacles` via `env.add_obstacle(Obstacle(id=..., bounds=...))`.
  4. Instantiate `SimulationEngine(environment=env, max_steps=..., timestep=..., seed=...)`.
  5. Populate `SimulationEngine.pedestrians` via `engine.add_pedestrian(Pedestrian(id=..., position=..., velocity=..., desired_speed=..., radius=..., status=...))`.
* **Execution & Stepping:**
  - `engine.step()`: Increments `current_step` by 1 and updates each active pedestrian position:
    $$x_{t+1} = x_t + v_x \cdot \Delta t, \quad y_{t+1} = y_t + v_y \cdot \Delta t$$
  - `engine.is_finished()`: Returns `True` if `current_step >= max_steps` OR if all pedestrians are non-active (`len(active_peds) == 0 and len(pedestrians) > 0`).
  - `engine.run()`: Steps until `is_finished()` and returns a summary dict: `{"total_steps": int, "pedestrians_total": int, "evacuated": int}`.

### 1.2 Pavan's Reference Engine (`pavan-part1-reference/src/genevac/simulation/engine.py`)
In the reference repository, the engine constructor is:
```python
class SimulationEngine:
    def __init__(
        self,
        building: Building,
        registry: Sequence[PedestrianState],
        dt: float = 0.1,
        observation_interval: float = 0.4,
    ):
        ...
```
* Read-only observation snapshots emitted at cadence `observation_interval`.
* Stepping driven by `run_steps(num_steps)` or `step_until(target_step)`.

---

## 2. Supported Configuration Fields & Defaults

### 2.1 Engine Constructor Parameters

| Parameter | Type | Default | Accepted Range | Where Consumed | Effect | Evidence Location |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `environment` | `EnvironmentMap` | *required* | Instance of `EnvironmentMap` | `self.environment` | Boundaries & obstacles for layout | `src/genevac/simulation/__init__.py:10` |
| `max_steps` | `int` | `1000` | $\ge 1$ | `self.max_steps` in `is_finished()` | Run duration limit | `src/genevac/simulation/__init__.py:11,26` |
| `timestep` | `float` | `0.1` | $> 0.0$ | `ped.update_position(self.timestep)` | Integration delta ($\text{s}$) | `src/genevac/simulation/__init__.py:12,23` |
| `seed` | `int` | `42` | Any int | `self.seed = seed` | Stored attribute (no RNG used) | `src/genevac/simulation/__init__.py:13` |

### 2.2 Environment Parameters

| Parameter | Type | Default | Accepted Range | Where Consumed | Effect | Evidence Location |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `width` | `float` | `50.0` | $> 0.0$ | `is_walkable()` bounds | Horizontal room dimension | `src/genevac/environment/__init__.py:22` |
| `height` | `float` | `20.0` | $> 0.0$ | `is_walkable()` bounds | Vertical room dimension | `src/genevac/environment/__init__.py:23` |
| `Exit.id` | `str` | *required* | Non-empty str | `add_exit()` | Exit identification | `src/genevac/environment/__init__.py:9` |
| `Exit.position`| `Tuple[float, float]` | *required* | Finite $(x, y)$ | `Exit` position | Geometric location | `src/genevac/environment/__init__.py:10` |
| `Exit.width` | `float` | `2.0` | $> 0.0$ | Door aperture | Door width | `src/genevac/environment/__init__.py:11` |
| `Obstacle.bounds` | `Tuple[4 floats]` | *required* | `(min_x, min_y, max_x, max_y)` | `is_walkable()` collision | Impassable geometric box | `src/genevac/environment/__init__.py:17` |

### 2.3 Pedestrian Parameters

| Parameter | Type | Default | Accepted Range | Where Consumed | Effect | Evidence Location |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | `str` | *required* | Non-empty str | Agent identity | Downstream tracking key | `src/genevac/pedestrians/__init__.py:16` |
| `position` | `Tuple[float, float]`| *required* | Finite $(x, y)$ | Initial coordinates | Starting location | `src/genevac/pedestrians/__init__.py:17` |
| `velocity` | `Tuple[float, float]`| `(0.0, 0.0)` | Finite $(v_x, v_y)$ | Movement update | Per-step kinematic displacement | `src/genevac/pedestrians/__init__.py:18` |
| `desired_speed` | `float` | `1.34` | $\ge 0.0$ | Target speed | Speed magnitude | `src/genevac/pedestrians/__init__.py:19` |
| `radius` | `float` | `0.3` | $> 0.0$ | Collision radius | Physical body size | `src/genevac/pedestrians/__init__.py:20` |
| `target_exit` | `str` | `""` | Any str | Routing destination | Navigation target | `src/genevac/pedestrians/__init__.py:21` |
| `status` | `PedestrianStatus` | `ACTIVE` | `ACTIVE`, `EVACUATED`, `INCAPACITATED` | State gate in step | Movement enablement | `src/genevac/pedestrians/__init__.py:22` |

---

## 3. Seed Support and Repeatability Audit

1. **Local Simulation Engine:**
   - Parameter `seed: int = 42` is accepted by `SimulationEngine.__init__` and assigned to `self.seed`.
   - **Critical Finding:** Neither `SimulationEngine.step()` nor `Pedestrian.update_position()` invokes Python's `random` or `numpy.random`. The kinematics are purely deterministic linear updates.
   - **Verdict:** Reproducibility is guaranteed by deterministic initial states. Changing `seed` without external stochastic generation does not alter trajectory outcomes. The system must report `seed_effect="deterministic_constant"` rather than claiming pseudo-random Monte Carlo variation.
2. **Reference Simulation Engine:**
   - Parameter `seed` is not present in constructor. Movement uses deterministic social force / rule-based step advance (`advance_population_step`).
   - **Verdict:** `seed` is unsupported upstream; marked `UNSUPPORTED`.

---

## 4. Run Duration and Stopping Conditions

1. **Local Engine Stopping Rule:**
   - Condition A: `self.current_step >= self.max_steps`.
   - Condition B: `len(active_peds) == 0 and len(self.pedestrians) > 0`.
   - In either case, `is_finished()` returns `True`, terminating `run()`.
2. **Termination Reasons Captured:**
   - `COMPLETED`: All pedestrians reached inactive/evacuated status before `max_steps`.
   - `MAX_STEPS_REACHED`: `current_step >= max_steps` while pedestrians remained active.
   - `STOPPED_EARLY`: External stop signal.
   - `ERROR`: Unhandled exception during stepping.

---

## 5. Existing Reusable Phase 02, 03, and 05 Components

* **Phase 02 Metrics (`src/genevac/metrics/`):**
  - `calculate_ade`, `calculate_fde`, `calculate_prediction_coverage`: Trajectory prediction accuracy.
  - `calculate_clearance_time`, `calculate_agent_evacuation_times`, `calculate_exit_throughput`: Evacuation performance.
  - `calculate_area_density`, `calculate_minimum_separation`: Crowd density and safety spacing.
  - `calculate_geometric_path_length`, `validate_evacuation_path`: Route verification.
  - `MetricSummary`, `aggregate_metric_series`: Basic statistical moments.
* **Phase 03 Canonical Observation Adapter (`src/genevac/prediction/adapter.py`):**
  - `SimulationTrajectoryAdapter`: Ingests simulation instances or snapshots without mutation.
  - Enforces cadence filtering, tracks cohorts, and outputs canonical `TrajectoryObservation` and `TrajectorySnapshot`.
* **Phase 05 Live Analytics Contracts (`src/genevac/analytics/live.py`):**
  - `MetricRecord`, `MetricStatus` (`VALID`, `UNAVAILABLE`, `STALE`, `NOT_COMPUTED`, `PAUSED`).
  - `ChartPoint`, `ChartSeries` for bounded, deduplicated time-series.

---

## 6. Unsupported or Ambiguous Capabilities

1. **Stochastic Seed Variation:**
   - Because the simulation engine uses deterministic kinematics without an internal PRNG, runs with identical initial conditions produce identical results regardless of seed. Multi-run repetitions with different seeds only produce variation if an external stochastic initial placement generator is employed.
2. **Authoritative Exit Events:**
   - The engine does not emit discrete `EXIT` event streams with sub-step timestamps. Evacuation status is recorded per snapshot.
3. **Composite Ranking / ERSI / Delta Score:**
   - Undefined by upstream specification; remains **BLOCKED**. No composite scenario rank or score will be synthesized.

---

## 7. Scope Protection Commitment

- **Pavan's Simulation Engine:** Untouched (zero modifications to `src/genevac/simulation/__init__.py` or `pavan-part1-reference/`).
- **Module A Density Prediction:** Untouched and out of scope.
- **New Scenario Knobs:** No unsupported parameters will be injected into the engine. Only parameters natively supported by `EnvironmentMap`, `Exit`, `Obstacle`, `Pedestrian`, and `SimulationEngine` will be used.

---

## 8. Proposed Files for Phase 06 Implementation

1. `src/genevac/experiments/scenario.py` — Scenario definition, run records, per-run metric results, comparison definition, aggregate metric results, and configuration fingerprinting.
2. `src/genevac/experiments/comparability.py` — Metric comparability gate, reason codes, and validation rules.
3. `src/genevac/experiments/runner.py` — Non-invasive scenario runner and orchestrator connecting simulation runs to Phase 02/03/05 analytics.
4. `src/genevac/experiments/presentation.py` — Markdown tables, difference-from-baseline summaries, and interpretation notes generator.
5. `src/genevac/experiments/__init__.py` — Clean re-exports of Phase 06 symbols.
6. `docs/phase06_comparison_schema.md` — Schema documentation.
7. `docs/phase06_metric_comparability.md` — Comparability policy documentation.
8. `docs/phase06_validation_examples.md` — Concrete comparison examples and reproduction instructions.
9. `docs/phase06_interpretation_notes.md` — Scientific interpretation and limitations notes.
10. `tests/test_scenario_metadata_phase06.py` — Scenario metadata and fingerprint tests.
11. `tests/test_comparability_phase06.py` — Comparability gate tests.
12. `tests/test_aggregation_phase06.py` — Statistical aggregation and determinism tests.
13. `tests/test_scenario_comparison_phase06.py` — End-to-end integration test of actual scenario comparison.
