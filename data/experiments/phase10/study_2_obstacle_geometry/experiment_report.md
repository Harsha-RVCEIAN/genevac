# Experiment Evaluation Report: Obstacle Geometry and Spatial Constriction Analysis

- **Experiment Identifier:** `study_obstacle_geometry`
- **Schema Version:** `1.0.0`
- **Execution Timestamp:** `2026-11-12T04:54:06.206045+00:00`
- **Baseline Scenario:** `obs_clear_baseline`

---

## 1. Experiment Overview

- **Research Question:** Evaluate the effect of geometric obstacles on hallway pedestrian flow.
- **Independent Variable:** `obstacle_configuration`
- **Dependent Metrics:** `total_elapsed_time`, `mean_speed`, `area_density`, `minimum_separation`, `exit_throughput`
- **Controlled Factors:** `width=50.0`, `height=20.0`, `agent_count=15`, `timestep=0.1`, `max_steps=40`
- **Repetition Policy:** deterministic_duplicate

---

## 2. Scenario Inventory

| Scenario ID | Name (Ver) | Role | Agents | Environment | Key Configuration | Fingerprint | Runs (Succ/Att) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `obs_central_pillar` | Hallway with Central Pillar (1.0.0) | VARIANT | 15 | 50.0m × 20.0m | 1 obstacles, 1 exits | `7e016f7819` | 2/2 |
| `obs_clear_baseline` | Clear Hallway Baseline (1.0.0) | **BASELINE** | 15 | 50.0m × 20.0m | 1 exits | `c74f914951` | 2/2 |
| `obs_dual_pillars` | Hallway with Staggered Dual Pillars (1.0.0) | VARIANT | 15 | 50.0m × 20.0m | 2 obstacles, 1 exits | `a5b281b878` | 2/2 |

---

## 3. Execution Summary

- **Total Runs Attempted:** 6
- **Successful Runs:** 6
- **Partial Runs:** 0
- **Failed Runs:** 0

| Run ID | Scenario ID | Final Step | Simulated Duration | Run Status | Termination Reason | Failure Details |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `obs_central_pillar_run_01` | `obs_central_pillar` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |
| `obs_central_pillar_run_02` | `obs_central_pillar` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |
| `obs_clear_baseline_run_01` | `obs_clear_baseline` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |
| `obs_clear_baseline_run_02` | `obs_clear_baseline` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |
| `obs_dual_pillars_run_01` | `obs_dual_pillars` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |
| `obs_dual_pillars_run_02` | `obs_dual_pillars` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |

---

## 4. Per-Run Results

| Run ID | Scenario | Metric | Value | Unit | Status | Sample Count (Valid/Missing) | Diagnostics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `obs_central_pillar_run_01` | `obs_central_pillar` | `area_density` | 0.0150 | ped/m² | VALID | 1/0 | None |
| `obs_central_pillar_run_01` | `obs_central_pillar` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/15 evacuated before simulation end. |
| `obs_central_pillar_run_01` | `obs_central_pillar` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `obs_central_pillar_run_01` | `obs_central_pillar` | `mean_speed` | 1.0000 | m/s | VALID | 15/0 | None |
| `obs_central_pillar_run_01` | `obs_central_pillar` | `minimum_separation` | 2.0000 | metres | VALID | 15/0 | None |
| `obs_central_pillar_run_01` | `obs_central_pillar` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |
| `obs_central_pillar_run_02` | `obs_central_pillar` | `area_density` | 0.0150 | ped/m² | VALID | 1/0 | None |
| `obs_central_pillar_run_02` | `obs_central_pillar` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/15 evacuated before simulation end. |
| `obs_central_pillar_run_02` | `obs_central_pillar` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `obs_central_pillar_run_02` | `obs_central_pillar` | `mean_speed` | 1.0000 | m/s | VALID | 15/0 | None |
| `obs_central_pillar_run_02` | `obs_central_pillar` | `minimum_separation` | 2.0000 | metres | VALID | 15/0 | None |
| `obs_central_pillar_run_02` | `obs_central_pillar` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |
| `obs_clear_baseline_run_01` | `obs_clear_baseline` | `area_density` | 0.0150 | ped/m² | VALID | 1/0 | None |
| `obs_clear_baseline_run_01` | `obs_clear_baseline` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/15 evacuated before simulation end. |
| `obs_clear_baseline_run_01` | `obs_clear_baseline` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `obs_clear_baseline_run_01` | `obs_clear_baseline` | `mean_speed` | 1.0000 | m/s | VALID | 15/0 | None |
| `obs_clear_baseline_run_01` | `obs_clear_baseline` | `minimum_separation` | 2.0000 | metres | VALID | 15/0 | None |
| `obs_clear_baseline_run_01` | `obs_clear_baseline` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |
| `obs_clear_baseline_run_02` | `obs_clear_baseline` | `area_density` | 0.0150 | ped/m² | VALID | 1/0 | None |
| `obs_clear_baseline_run_02` | `obs_clear_baseline` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/15 evacuated before simulation end. |
| `obs_clear_baseline_run_02` | `obs_clear_baseline` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `obs_clear_baseline_run_02` | `obs_clear_baseline` | `mean_speed` | 1.0000 | m/s | VALID | 15/0 | None |
| `obs_clear_baseline_run_02` | `obs_clear_baseline` | `minimum_separation` | 2.0000 | metres | VALID | 15/0 | None |
| `obs_clear_baseline_run_02` | `obs_clear_baseline` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |
| `obs_dual_pillars_run_01` | `obs_dual_pillars` | `area_density` | 0.0150 | ped/m² | VALID | 1/0 | None |
| `obs_dual_pillars_run_01` | `obs_dual_pillars` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/15 evacuated before simulation end. |
| `obs_dual_pillars_run_01` | `obs_dual_pillars` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `obs_dual_pillars_run_01` | `obs_dual_pillars` | `mean_speed` | 1.0000 | m/s | VALID | 15/0 | None |
| `obs_dual_pillars_run_01` | `obs_dual_pillars` | `minimum_separation` | 2.0000 | metres | VALID | 15/0 | None |
| `obs_dual_pillars_run_01` | `obs_dual_pillars` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |
| `obs_dual_pillars_run_02` | `obs_dual_pillars` | `area_density` | 0.0150 | ped/m² | VALID | 1/0 | None |
| `obs_dual_pillars_run_02` | `obs_dual_pillars` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/15 evacuated before simulation end. |
| `obs_dual_pillars_run_02` | `obs_dual_pillars` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `obs_dual_pillars_run_02` | `obs_dual_pillars` | `mean_speed` | 1.0000 | m/s | VALID | 15/0 | None |
| `obs_dual_pillars_run_02` | `obs_dual_pillars` | `minimum_separation` | 2.0000 | metres | VALID | 15/0 | None |
| `obs_dual_pillars_run_02` | `obs_dual_pillars` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |

---

## 5. Aggregate Metric Comparisons

| Metric | Unit | Baseline Mean | Variant Scenario | Variant Mean | Absolute Delta (Δ) | Relative Delta (%Δ) | Comparability | Caveats / Diagnostics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `total_elapsed_time` | seconds | 4.000 ± 0.000 | `obs_central_pillar` | 4.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `total_elapsed_time` | seconds | 4.000 ± 0.000 | `obs_dual_pillars` | 4.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `mean_speed` | m/s | 1.000 ± 0.000 | `obs_central_pillar` | 1.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `mean_speed` | m/s | 1.000 ± 0.000 | `obs_dual_pillars` | 1.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `area_density` | ped/m² | 0.015 ± 0.000 | `obs_central_pillar` | 0.015 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `area_density` | ped/m² | 0.015 ± 0.000 | `obs_dual_pillars` | 0.015 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `minimum_separation` | metres | 2.000 ± 0.000 | `obs_central_pillar` | 2.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `minimum_separation` | metres | 2.000 ± 0.000 | `obs_dual_pillars` | 2.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `exit_throughput` | ped/s | UNAVAILABLE | `obs_central_pillar` | UNAVAILABLE | N/A | N/A | **NOT_COMPARABLE** | Authoritative measurements or exit events absent; cross-scenario comparison cannot be made. |
| `exit_throughput` | ped/s | UNAVAILABLE | `obs_dual_pillars` | UNAVAILABLE | N/A | N/A | **NOT_COMPARABLE** | Authoritative measurements or exit events absent; cross-scenario comparison cannot be made. |

---

## 6. Comparability Decisions

| Variant Scenario | Metric | Decision | Comparable? | Reason Codes | Caveats & Explanations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `obs_central_pillar` | `area_density` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |
| `obs_central_pillar` | `exit_throughput` | **NOT_COMPARABLE** | NO | `DATA_UNAVAILABLE` | Authoritative measurements or exit events absent; cross-scenario comparison cannot be made. |
| `obs_central_pillar` | `mean_speed` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |
| `obs_central_pillar` | `minimum_separation` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |
| `obs_central_pillar` | `total_elapsed_time` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |
| `obs_dual_pillars` | `area_density` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |
| `obs_dual_pillars` | `exit_throughput` | **NOT_COMPARABLE** | NO | `DATA_UNAVAILABLE` | Authoritative measurements or exit events absent; cross-scenario comparison cannot be made. |
| `obs_dual_pillars` | `mean_speed` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |
| `obs_dual_pillars` | `minimum_separation` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |
| `obs_dual_pillars` | `total_elapsed_time` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |

---

## 7. Scientific Interpretation

### Scientific Interpretation & Study Limitations
- **Research Question:** Evaluate the effect of geometric obstacles on hallway pedestrian flow.
- **Independent Variable:** `obstacle_configuration`
- **Controlled Factors:** `width=50.0`, `height=20.0`, `agent_count=15`, `timestep=0.1`, `max_steps=40`

#### 1. Observed Differences:
- For scenario `obs_central_pillar`, `area_density` decreased by 0.000 ped/m² (+0.0%) compared to baseline `obs_clear_baseline`.
- For scenario `obs_central_pillar`, `exit_throughput` delta could not be computed (decision: NOT_COMPARABLE).
- For scenario `obs_central_pillar`, `mean_speed` decreased by 0.000 m/s (+0.0%) compared to baseline `obs_clear_baseline`.
- For scenario `obs_central_pillar`, `minimum_separation` decreased by 0.000 metres (+0.0%) compared to baseline `obs_clear_baseline`.
- For scenario `obs_central_pillar`, `total_elapsed_time` decreased by 0.000 seconds (+0.0%) compared to baseline `obs_clear_baseline`.
- For scenario `obs_dual_pillars`, `area_density` decreased by 0.000 ped/m² (+0.0%) compared to baseline `obs_clear_baseline`.
- For scenario `obs_dual_pillars`, `exit_throughput` delta could not be computed (decision: NOT_COMPARABLE).
- For scenario `obs_dual_pillars`, `mean_speed` decreased by 0.000 m/s (+0.0%) compared to baseline `obs_clear_baseline`.
- For scenario `obs_dual_pillars`, `minimum_separation` decreased by 0.000 metres (+0.0%) compared to baseline `obs_clear_baseline`.
- For scenario `obs_dual_pillars`, `total_elapsed_time` decreased by 0.000 seconds (+0.0%) compared to baseline `obs_clear_baseline`.

#### 2. Plausible Explanations & Confounders:
- Differences in clearance time and crowd density reflect physical changes in layout geometry and agent kinematic bottlenecks.
- Because the simulation dynamics are deterministic, variation across runs with identical initial conditions is 0.0 unless stochastic initial agent generators are used.

#### 3. What Cannot Be Concluded:
- **No Composite Ranking:** Scenarios are not ranked using composite weighted scores or artificial indexes.
- **No Unverified Causal Attribution:** In multi-factor exploratory scenarios, observed changes cannot be attributed solely to a single parameter without isolating each factor independently.
- **ERSI / Delta Score:** Intentionally excluded due to absence of authoritative mathematical specifications in repository scope.

---

## 8. Unavailable Metrics and Reasons

- **Metric `exit_throughput` (Scenario `obs_central_pillar`):** UNAVAILABLE. Reason: Run obs_central_pillar_run_01: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).; Run obs_central_pillar_run_02: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).
- **Metric `exit_throughput` (Scenario `obs_clear_baseline`):** UNAVAILABLE. Reason: Run obs_clear_baseline_run_01: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).; Run obs_clear_baseline_run_02: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).
- **Metric `exit_throughput` (Scenario `obs_dual_pillars`):** UNAVAILABLE. Reason: Run obs_dual_pillars_run_01: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).; Run obs_dual_pillars_run_02: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).

---

## 9. Failed Runs and Reasons

*All attempted simulation runs completed successfully with zero runtime failures.*

---

## 10. Reproducibility Metadata

- **Repetition Policy & Sample Type:** `deterministic_duplicate`
- **Seed Engine Usage:** `seed_used_by_engine=False`. *As inspected in the current Pavan SimulationEngine (genevac.simulation.engine), the engine's kinematics are purely deterministic and do not consume PRNG seeds during physics stepping. Supplied seeds are recorded as scenario intent only.*
- **Scenario Configuration Fingerprints (SHA-256):**
  - `obs_central_pillar`: `7e016f781948a753ead0e76ccc3dbe6c796b20dd3d5af5e74074f47ec0bfff49`
  - `obs_clear_baseline`: `c74f914951895946759c3948112a9a84ba233f09590134fca115dec7d0068187`
  - `obs_dual_pillars`: `a5b281b87875f47447a028a5a8585ddbea1fd42e8e171ad3e3dcabc16a50e736`
- **Notes:** Executed 2 repetitions per scenario. All multi-run metrics produced zero variance (std_dev = 0.0) due to deterministic simulation kinematics. These are deterministic duplicates and must not be treated as independent stochastic samples.

---

## 11. Limitations & Boundaries

- **Directly Measured vs. Derived:** Physical trajectories, step durations, and boundary collisions are measured directly; densities, clearance times, and mean velocities are mathematically derived.
- **No Composite Scoring:** GenEvac-Sentinel strictly rejects artificial composite scoring, weighting schemes, or opaque ranking formulas.
- **Zero Inferred Evacuations:** Agents missing from snapshot observations are marked `not_observed` and are never assumed evacuated.
- **Simulation Engine Independence:** Person B analytics maintain a strictly read-only relationship with Pavan's simulation engine.
