# Experiment Evaluation Report: Exit Width and Dual Portal Bottleneck Evaluation

- **Experiment Identifier:** `study_exit_configuration`
- **Schema Version:** `1.0.0`
- **Execution Timestamp:** `2026-11-12T04:54:06.225303+00:00`
- **Baseline Scenario:** `exit_std_baseline`

---

## 1. Experiment Overview

- **Research Question:** Evaluate the impact of exit width constriction and dual exit availability on evacuation flow.
- **Independent Variable:** `exit_portal_configuration`
- **Dependent Metrics:** `total_elapsed_time`, `mean_speed`, `area_density`, `minimum_separation`, `exit_throughput`
- **Controlled Factors:** `width=50.0`, `height=20.0`, `agent_count=20`, `timestep=0.1`, `max_steps=40`
- **Repetition Policy:** deterministic_duplicate

---

## 2. Scenario Inventory

| Scenario ID | Name (Ver) | Role | Agents | Environment | Key Configuration | Fingerprint | Runs (Succ/Att) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `exit_dual_portals` | Dual Exit Portals (2x 2.5m) (1.0.0) | VARIANT | 20 | 50.0m × 20.0m | 2 exits | `60977ae660` | 2/2 |
| `exit_narrow_bottleneck` | Narrow Exit Bottleneck (1.0m) (1.0.0) | VARIANT | 20 | 50.0m × 20.0m | 1 exits | `a88261ebee` | 2/2 |
| `exit_std_baseline` | Standard Exit Baseline (3.0m) (1.0.0) | **BASELINE** | 20 | 50.0m × 20.0m | 1 exits | `26e91f29df` | 2/2 |

---

## 3. Execution Summary

- **Total Runs Attempted:** 6
- **Successful Runs:** 6
- **Partial Runs:** 0
- **Failed Runs:** 0

| Run ID | Scenario ID | Final Step | Simulated Duration | Run Status | Termination Reason | Failure Details |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `exit_dual_portals_run_01` | `exit_dual_portals` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |
| `exit_dual_portals_run_02` | `exit_dual_portals` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |
| `exit_narrow_bottleneck_run_01` | `exit_narrow_bottleneck` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |
| `exit_narrow_bottleneck_run_02` | `exit_narrow_bottleneck` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |
| `exit_std_baseline_run_01` | `exit_std_baseline` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |
| `exit_std_baseline_run_02` | `exit_std_baseline` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |

---

## 4. Per-Run Results

| Run ID | Scenario | Metric | Value | Unit | Status | Sample Count (Valid/Missing) | Diagnostics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `exit_dual_portals_run_01` | `exit_dual_portals` | `area_density` | 0.0200 | ped/m² | VALID | 1/0 | None |
| `exit_dual_portals_run_01` | `exit_dual_portals` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/20 evacuated before simulation end. |
| `exit_dual_portals_run_01` | `exit_dual_portals` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `exit_dual_portals_run_01` | `exit_dual_portals` | `mean_speed` | 1.0000 | m/s | VALID | 20/0 | None |
| `exit_dual_portals_run_01` | `exit_dual_portals` | `minimum_separation` | 2.0000 | metres | VALID | 20/0 | None |
| `exit_dual_portals_run_01` | `exit_dual_portals` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |
| `exit_dual_portals_run_02` | `exit_dual_portals` | `area_density` | 0.0200 | ped/m² | VALID | 1/0 | None |
| `exit_dual_portals_run_02` | `exit_dual_portals` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/20 evacuated before simulation end. |
| `exit_dual_portals_run_02` | `exit_dual_portals` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `exit_dual_portals_run_02` | `exit_dual_portals` | `mean_speed` | 1.0000 | m/s | VALID | 20/0 | None |
| `exit_dual_portals_run_02` | `exit_dual_portals` | `minimum_separation` | 2.0000 | metres | VALID | 20/0 | None |
| `exit_dual_portals_run_02` | `exit_dual_portals` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |
| `exit_narrow_bottleneck_run_01` | `exit_narrow_bottleneck` | `area_density` | 0.0200 | ped/m² | VALID | 1/0 | None |
| `exit_narrow_bottleneck_run_01` | `exit_narrow_bottleneck` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/20 evacuated before simulation end. |
| `exit_narrow_bottleneck_run_01` | `exit_narrow_bottleneck` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `exit_narrow_bottleneck_run_01` | `exit_narrow_bottleneck` | `mean_speed` | 1.0000 | m/s | VALID | 20/0 | None |
| `exit_narrow_bottleneck_run_01` | `exit_narrow_bottleneck` | `minimum_separation` | 2.0000 | metres | VALID | 20/0 | None |
| `exit_narrow_bottleneck_run_01` | `exit_narrow_bottleneck` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |
| `exit_narrow_bottleneck_run_02` | `exit_narrow_bottleneck` | `area_density` | 0.0200 | ped/m² | VALID | 1/0 | None |
| `exit_narrow_bottleneck_run_02` | `exit_narrow_bottleneck` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/20 evacuated before simulation end. |
| `exit_narrow_bottleneck_run_02` | `exit_narrow_bottleneck` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `exit_narrow_bottleneck_run_02` | `exit_narrow_bottleneck` | `mean_speed` | 1.0000 | m/s | VALID | 20/0 | None |
| `exit_narrow_bottleneck_run_02` | `exit_narrow_bottleneck` | `minimum_separation` | 2.0000 | metres | VALID | 20/0 | None |
| `exit_narrow_bottleneck_run_02` | `exit_narrow_bottleneck` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |
| `exit_std_baseline_run_01` | `exit_std_baseline` | `area_density` | 0.0200 | ped/m² | VALID | 1/0 | None |
| `exit_std_baseline_run_01` | `exit_std_baseline` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/20 evacuated before simulation end. |
| `exit_std_baseline_run_01` | `exit_std_baseline` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `exit_std_baseline_run_01` | `exit_std_baseline` | `mean_speed` | 1.0000 | m/s | VALID | 20/0 | None |
| `exit_std_baseline_run_01` | `exit_std_baseline` | `minimum_separation` | 2.0000 | metres | VALID | 20/0 | None |
| `exit_std_baseline_run_01` | `exit_std_baseline` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |
| `exit_std_baseline_run_02` | `exit_std_baseline` | `area_density` | 0.0200 | ped/m² | VALID | 1/0 | None |
| `exit_std_baseline_run_02` | `exit_std_baseline` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/20 evacuated before simulation end. |
| `exit_std_baseline_run_02` | `exit_std_baseline` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `exit_std_baseline_run_02` | `exit_std_baseline` | `mean_speed` | 1.0000 | m/s | VALID | 20/0 | None |
| `exit_std_baseline_run_02` | `exit_std_baseline` | `minimum_separation` | 2.0000 | metres | VALID | 20/0 | None |
| `exit_std_baseline_run_02` | `exit_std_baseline` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |

---

## 5. Aggregate Metric Comparisons

| Metric | Unit | Baseline Mean | Variant Scenario | Variant Mean | Absolute Delta (Δ) | Relative Delta (%Δ) | Comparability | Caveats / Diagnostics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `total_elapsed_time` | seconds | 4.000 ± 0.000 | `exit_narrow_bottleneck` | 4.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `total_elapsed_time` | seconds | 4.000 ± 0.000 | `exit_dual_portals` | 4.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `mean_speed` | m/s | 1.000 ± 0.000 | `exit_narrow_bottleneck` | 1.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `mean_speed` | m/s | 1.000 ± 0.000 | `exit_dual_portals` | 1.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `area_density` | ped/m² | 0.020 ± 0.000 | `exit_narrow_bottleneck` | 0.020 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `area_density` | ped/m² | 0.020 ± 0.000 | `exit_dual_portals` | 0.020 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `minimum_separation` | metres | 2.000 ± 0.000 | `exit_narrow_bottleneck` | 2.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `minimum_separation` | metres | 2.000 ± 0.000 | `exit_dual_portals` | 2.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE** | Metric definitions, units, windows, spatial frames, and populations match. |
| `exit_throughput` | ped/s | UNAVAILABLE | `exit_narrow_bottleneck` | UNAVAILABLE | N/A | N/A | **NOT_COMPARABLE** | Authoritative measurements or exit events absent; cross-scenario comparison cannot be made. |
| `exit_throughput` | ped/s | UNAVAILABLE | `exit_dual_portals` | UNAVAILABLE | N/A | N/A | **NOT_COMPARABLE** | Authoritative measurements or exit events absent; cross-scenario comparison cannot be made. |

---

## 6. Comparability Decisions

| Variant Scenario | Metric | Decision | Comparable? | Reason Codes | Caveats & Explanations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `exit_dual_portals` | `area_density` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |
| `exit_dual_portals` | `exit_throughput` | **NOT_COMPARABLE** | NO | `DATA_UNAVAILABLE` | Authoritative measurements or exit events absent; cross-scenario comparison cannot be made. |
| `exit_dual_portals` | `mean_speed` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |
| `exit_dual_portals` | `minimum_separation` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |
| `exit_dual_portals` | `total_elapsed_time` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |
| `exit_narrow_bottleneck` | `area_density` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |
| `exit_narrow_bottleneck` | `exit_throughput` | **NOT_COMPARABLE** | NO | `DATA_UNAVAILABLE` | Authoritative measurements or exit events absent; cross-scenario comparison cannot be made. |
| `exit_narrow_bottleneck` | `mean_speed` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |
| `exit_narrow_bottleneck` | `minimum_separation` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |
| `exit_narrow_bottleneck` | `total_elapsed_time` | **COMPARABLE** | YES | `FULLY_COMPARABLE` | Metric definitions, units, windows, spatial frames, and populations match. |

---

## 7. Scientific Interpretation

### Scientific Interpretation & Study Limitations
- **Research Question:** Evaluate the impact of exit width constriction and dual exit availability on evacuation flow.
- **Independent Variable:** `exit_portal_configuration`
- **Controlled Factors:** `width=50.0`, `height=20.0`, `agent_count=20`, `timestep=0.1`, `max_steps=40`

#### 1. Observed Differences:
- For scenario `exit_dual_portals`, `area_density` decreased by 0.000 ped/m² (+0.0%) compared to baseline `exit_std_baseline`.
- For scenario `exit_dual_portals`, `exit_throughput` delta could not be computed (decision: NOT_COMPARABLE).
- For scenario `exit_dual_portals`, `mean_speed` decreased by 0.000 m/s (+0.0%) compared to baseline `exit_std_baseline`.
- For scenario `exit_dual_portals`, `minimum_separation` decreased by 0.000 metres (+0.0%) compared to baseline `exit_std_baseline`.
- For scenario `exit_dual_portals`, `total_elapsed_time` decreased by 0.000 seconds (+0.0%) compared to baseline `exit_std_baseline`.
- For scenario `exit_narrow_bottleneck`, `area_density` decreased by 0.000 ped/m² (+0.0%) compared to baseline `exit_std_baseline`.
- For scenario `exit_narrow_bottleneck`, `exit_throughput` delta could not be computed (decision: NOT_COMPARABLE).
- For scenario `exit_narrow_bottleneck`, `mean_speed` decreased by 0.000 m/s (+0.0%) compared to baseline `exit_std_baseline`.
- For scenario `exit_narrow_bottleneck`, `minimum_separation` decreased by 0.000 metres (+0.0%) compared to baseline `exit_std_baseline`.
- For scenario `exit_narrow_bottleneck`, `total_elapsed_time` decreased by 0.000 seconds (+0.0%) compared to baseline `exit_std_baseline`.

#### 2. Plausible Explanations & Confounders:
- Differences in clearance time and crowd density reflect physical changes in layout geometry and agent kinematic bottlenecks.
- Because the simulation dynamics are deterministic, variation across runs with identical initial conditions is 0.0 unless stochastic initial agent generators are used.

#### 3. What Cannot Be Concluded:
- **No Composite Ranking:** Scenarios are not ranked using composite weighted scores or artificial indexes.
- **No Unverified Causal Attribution:** In multi-factor exploratory scenarios, observed changes cannot be attributed solely to a single parameter without isolating each factor independently.
- **ERSI / Delta Score:** Intentionally excluded due to absence of authoritative mathematical specifications in repository scope.

---

## 8. Unavailable Metrics and Reasons

- **Metric `exit_throughput` (Scenario `exit_dual_portals`):** UNAVAILABLE. Reason: Run exit_dual_portals_run_01: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).; Run exit_dual_portals_run_02: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).
- **Metric `exit_throughput` (Scenario `exit_narrow_bottleneck`):** UNAVAILABLE. Reason: Run exit_narrow_bottleneck_run_01: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).; Run exit_narrow_bottleneck_run_02: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).
- **Metric `exit_throughput` (Scenario `exit_std_baseline`):** UNAVAILABLE. Reason: Run exit_std_baseline_run_01: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).; Run exit_std_baseline_run_02: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).

---

## 9. Failed Runs and Reasons

*All attempted simulation runs completed successfully with zero runtime failures.*

---

## 10. Reproducibility Metadata

- **Repetition Policy & Sample Type:** `deterministic_duplicate`
- **Seed Engine Usage:** `seed_used_by_engine=False`. *As inspected in the current Pavan SimulationEngine (genevac.simulation.engine), the engine's kinematics are purely deterministic and do not consume PRNG seeds during physics stepping. Supplied seeds are recorded as scenario intent only.*
- **Scenario Configuration Fingerprints (SHA-256):**
  - `exit_dual_portals`: `60977ae66027e6e0fcab80d668d59dc8836e91f73d56e721e60b11620d84e9f5`
  - `exit_narrow_bottleneck`: `a88261ebee45cdb082cfbf7df975af0d3a9a02ceea2a804dfe3a6e5f36fb67bf`
  - `exit_std_baseline`: `26e91f29dff98d9723cfd2798efcb7f614d5d02b5c2f61254b4ea3f75d62fc00`
- **Notes:** Executed 2 repetitions per scenario. All multi-run metrics produced zero variance (std_dev = 0.0) due to deterministic simulation kinematics. These are deterministic duplicates and must not be treated as independent stochastic samples.

---

## 11. Limitations & Boundaries

- **Directly Measured vs. Derived:** Physical trajectories, step durations, and boundary collisions are measured directly; densities, clearance times, and mean velocities are mathematically derived.
- **No Composite Scoring:** GenEvac-Sentinel strictly rejects artificial composite scoring, weighting schemes, or opaque ranking formulas.
- **Zero Inferred Evacuations:** Agents missing from snapshot observations are marked `not_observed` and are never assumed evacuated.
- **Simulation Engine Independence:** Person B analytics maintain a strictly read-only relationship with Pavan's simulation engine.
