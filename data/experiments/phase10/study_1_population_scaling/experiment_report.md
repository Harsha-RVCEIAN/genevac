# Experiment Evaluation Report: Pedestrian Population Scaling and Density Impact

- **Experiment Identifier:** `study_pop_scaling`
- **Schema Version:** `1.0.0`
- **Execution Timestamp:** `2026-11-12T04:54:06.189942+00:00`
- **Baseline Scenario:** `pop_05_baseline`

---

## 1. Experiment Overview

- **Research Question:** Evaluate the effect of pedestrian population scaling on crowd density and minimum separation.
- **Independent Variable:** `pedestrian_population_count`
- **Dependent Metrics:** `total_elapsed_time`, `mean_speed`, `area_density`, `minimum_separation`, `exit_throughput`
- **Controlled Factors:** `width=50.0`, `height=20.0`, `timestep=0.1`, `max_steps=40`
- **Repetition Policy:** deterministic_duplicate

---

## 2. Scenario Inventory

| Scenario ID | Name (Ver) | Role | Agents | Environment | Key Configuration | Fingerprint | Runs (Succ/Att) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `pop_05_baseline` | Corridor Low Density (5 peds) (1.0.0) | **BASELINE** | 5 | 50.0m × 20.0m | 1 exits | `b2b8fed82b` | 2/2 |
| `pop_15_medium` | Corridor Medium Density (15 peds) (1.0.0) | VARIANT | 15 | 50.0m × 20.0m | 1 exits | `1a4087494b` | 2/2 |
| `pop_30_high` | Corridor High Density (30 peds) (1.0.0) | VARIANT | 30 | 50.0m × 20.0m | 1 exits | `aa18520b36` | 2/2 |

---

## 3. Execution Summary

- **Total Runs Attempted:** 6
- **Successful Runs:** 6
- **Partial Runs:** 0
- **Failed Runs:** 0

| Run ID | Scenario ID | Final Step | Simulated Duration | Run Status | Termination Reason | Failure Details |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `pop_05_baseline_run_01` | `pop_05_baseline` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |
| `pop_05_baseline_run_02` | `pop_05_baseline` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |
| `pop_15_medium_run_01` | `pop_15_medium` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |
| `pop_15_medium_run_02` | `pop_15_medium` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |
| `pop_30_high_run_01` | `pop_30_high` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |
| `pop_30_high_run_02` | `pop_30_high` | 40 | 4.0s | SUCCESS | MAX_STEPS_REACHED | None |

---

## 4. Per-Run Results

| Run ID | Scenario | Metric | Value | Unit | Status | Sample Count (Valid/Missing) | Diagnostics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `pop_05_baseline_run_01` | `pop_05_baseline` | `area_density` | 0.0050 | ped/m² | VALID | 1/0 | None |
| `pop_05_baseline_run_01` | `pop_05_baseline` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/5 evacuated before simulation end. |
| `pop_05_baseline_run_01` | `pop_05_baseline` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `pop_05_baseline_run_01` | `pop_05_baseline` | `mean_speed` | 1.0000 | m/s | VALID | 5/0 | None |
| `pop_05_baseline_run_01` | `pop_05_baseline` | `minimum_separation` | 2.0000 | metres | VALID | 5/0 | None |
| `pop_05_baseline_run_01` | `pop_05_baseline` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |
| `pop_05_baseline_run_02` | `pop_05_baseline` | `area_density` | 0.0050 | ped/m² | VALID | 1/0 | None |
| `pop_05_baseline_run_02` | `pop_05_baseline` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/5 evacuated before simulation end. |
| `pop_05_baseline_run_02` | `pop_05_baseline` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `pop_05_baseline_run_02` | `pop_05_baseline` | `mean_speed` | 1.0000 | m/s | VALID | 5/0 | None |
| `pop_05_baseline_run_02` | `pop_05_baseline` | `minimum_separation` | 2.0000 | metres | VALID | 5/0 | None |
| `pop_05_baseline_run_02` | `pop_05_baseline` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |
| `pop_15_medium_run_01` | `pop_15_medium` | `area_density` | 0.0150 | ped/m² | VALID | 1/0 | None |
| `pop_15_medium_run_01` | `pop_15_medium` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/15 evacuated before simulation end. |
| `pop_15_medium_run_01` | `pop_15_medium` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `pop_15_medium_run_01` | `pop_15_medium` | `mean_speed` | 1.0000 | m/s | VALID | 15/0 | None |
| `pop_15_medium_run_01` | `pop_15_medium` | `minimum_separation` | 2.0000 | metres | VALID | 15/0 | None |
| `pop_15_medium_run_01` | `pop_15_medium` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |
| `pop_15_medium_run_02` | `pop_15_medium` | `area_density` | 0.0150 | ped/m² | VALID | 1/0 | None |
| `pop_15_medium_run_02` | `pop_15_medium` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/15 evacuated before simulation end. |
| `pop_15_medium_run_02` | `pop_15_medium` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `pop_15_medium_run_02` | `pop_15_medium` | `mean_speed` | 1.0000 | m/s | VALID | 15/0 | None |
| `pop_15_medium_run_02` | `pop_15_medium` | `minimum_separation` | 2.0000 | metres | VALID | 15/0 | None |
| `pop_15_medium_run_02` | `pop_15_medium` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |
| `pop_30_high_run_01` | `pop_30_high` | `area_density` | 0.0300 | ped/m² | VALID | 1/0 | None |
| `pop_30_high_run_01` | `pop_30_high` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/30 evacuated before simulation end. |
| `pop_30_high_run_01` | `pop_30_high` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `pop_30_high_run_01` | `pop_30_high` | `mean_speed` | 1.0000 | m/s | VALID | 30/0 | None |
| `pop_30_high_run_01` | `pop_30_high` | `minimum_separation` | 2.0000 | metres | VALID | 30/0 | None |
| `pop_30_high_run_01` | `pop_30_high` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |
| `pop_30_high_run_02` | `pop_30_high` | `area_density` | 0.0300 | ped/m² | VALID | 1/0 | None |
| `pop_30_high_run_02` | `pop_30_high` | `clearance_time` | UNAVAILABLE | seconds | UNAVAILABLE | 0/0 | Evacuation incomplete: 0/30 evacuated before simulation end. |
| `pop_30_high_run_02` | `pop_30_high` | `exit_throughput` | UNAVAILABLE | ped/s | UNAVAILABLE | 0/0 | Authoritative discrete exit events not emitted by engine. |
| `pop_30_high_run_02` | `pop_30_high` | `mean_speed` | 1.0000 | m/s | VALID | 30/0 | None |
| `pop_30_high_run_02` | `pop_30_high` | `minimum_separation` | 2.0000 | metres | VALID | 30/0 | None |
| `pop_30_high_run_02` | `pop_30_high` | `total_elapsed_time` | 4.0000 | seconds | VALID | 1/0 | None |

---

## 5. Aggregate Metric Comparisons

| Metric | Unit | Baseline Mean | Variant Scenario | Variant Mean | Absolute Delta (Δ) | Relative Delta (%Δ) | Comparability | Caveats / Diagnostics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `total_elapsed_time` | seconds | 4.000 ± 0.000 | `pop_15_medium` | 4.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE_WITH_CAVEATS** | Agent counts differ (5 vs 15). Raw counts reflect population size differences. |
| `total_elapsed_time` | seconds | 4.000 ± 0.000 | `pop_30_high` | 4.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE_WITH_CAVEATS** | Agent counts differ (5 vs 30). Raw counts reflect population size differences. |
| `mean_speed` | m/s | 1.000 ± 0.000 | `pop_15_medium` | 1.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE_WITH_CAVEATS** | Agent counts differ (5 vs 15). Raw counts reflect population size differences. |
| `mean_speed` | m/s | 1.000 ± 0.000 | `pop_30_high` | 1.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE_WITH_CAVEATS** | Agent counts differ (5 vs 30). Raw counts reflect population size differences. |
| `area_density` | ped/m² | 0.005 ± 0.000 | `pop_15_medium` | 0.015 ± 0.000 | +0.010 | +200.0% | **COMPARABLE_WITH_CAVEATS** | Agent counts differ (5 vs 15). Raw counts reflect population size differences. |
| `area_density` | ped/m² | 0.005 ± 0.000 | `pop_30_high` | 0.030 ± 0.000 | +0.025 | +500.0% | **COMPARABLE_WITH_CAVEATS** | Agent counts differ (5 vs 30). Raw counts reflect population size differences. |
| `minimum_separation` | metres | 2.000 ± 0.000 | `pop_15_medium` | 2.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE_WITH_CAVEATS** | Agent counts differ (5 vs 15). Raw counts reflect population size differences. |
| `minimum_separation` | metres | 2.000 ± 0.000 | `pop_30_high` | 2.000 ± 0.000 | +0.000 | +0.0% | **COMPARABLE_WITH_CAVEATS** | Agent counts differ (5 vs 30). Raw counts reflect population size differences. |
| `exit_throughput` | ped/s | UNAVAILABLE | `pop_15_medium` | UNAVAILABLE | N/A | N/A | **NOT_COMPARABLE** | Authoritative measurements or exit events absent; cross-scenario comparison cannot be made. |
| `exit_throughput` | ped/s | UNAVAILABLE | `pop_30_high` | UNAVAILABLE | N/A | N/A | **NOT_COMPARABLE** | Authoritative measurements or exit events absent; cross-scenario comparison cannot be made. |

---

## 6. Comparability Decisions

| Variant Scenario | Metric | Decision | Comparable? | Reason Codes | Caveats & Explanations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `pop_15_medium` | `area_density` | **COMPARABLE_WITH_CAVEATS** | YES | `POPULATION_MISMATCH` | Agent counts differ (5 vs 15). Raw counts reflect population size differences. |
| `pop_15_medium` | `exit_throughput` | **NOT_COMPARABLE** | NO | `DATA_UNAVAILABLE` | Authoritative measurements or exit events absent; cross-scenario comparison cannot be made. |
| `pop_15_medium` | `mean_speed` | **COMPARABLE_WITH_CAVEATS** | YES | `POPULATION_MISMATCH` | Agent counts differ (5 vs 15). Raw counts reflect population size differences. |
| `pop_15_medium` | `minimum_separation` | **COMPARABLE_WITH_CAVEATS** | YES | `POPULATION_MISMATCH` | Agent counts differ (5 vs 15). Raw counts reflect population size differences. |
| `pop_15_medium` | `total_elapsed_time` | **COMPARABLE_WITH_CAVEATS** | YES | `POPULATION_MISMATCH` | Agent counts differ (5 vs 15). Raw counts reflect population size differences. |
| `pop_30_high` | `area_density` | **COMPARABLE_WITH_CAVEATS** | YES | `POPULATION_MISMATCH` | Agent counts differ (5 vs 30). Raw counts reflect population size differences. |
| `pop_30_high` | `exit_throughput` | **NOT_COMPARABLE** | NO | `DATA_UNAVAILABLE` | Authoritative measurements or exit events absent; cross-scenario comparison cannot be made. |
| `pop_30_high` | `mean_speed` | **COMPARABLE_WITH_CAVEATS** | YES | `POPULATION_MISMATCH` | Agent counts differ (5 vs 30). Raw counts reflect population size differences. |
| `pop_30_high` | `minimum_separation` | **COMPARABLE_WITH_CAVEATS** | YES | `POPULATION_MISMATCH` | Agent counts differ (5 vs 30). Raw counts reflect population size differences. |
| `pop_30_high` | `total_elapsed_time` | **COMPARABLE_WITH_CAVEATS** | YES | `POPULATION_MISMATCH` | Agent counts differ (5 vs 30). Raw counts reflect population size differences. |

---

## 7. Scientific Interpretation

### Scientific Interpretation & Study Limitations
- **Research Question:** Evaluate the effect of pedestrian population scaling on crowd density and minimum separation.
- **Independent Variable:** `pedestrian_population_count`
- **Controlled Factors:** `width=50.0`, `height=20.0`, `timestep=0.1`, `max_steps=40`

#### 1. Observed Differences:
- For scenario `pop_15_medium`, `area_density` increased by 0.010 ped/m² (+200.0%) compared to baseline `pop_05_baseline`.
- For scenario `pop_15_medium`, `exit_throughput` delta could not be computed (decision: NOT_COMPARABLE).
- For scenario `pop_15_medium`, `mean_speed` decreased by 0.000 m/s (+0.0%) compared to baseline `pop_05_baseline`.
- For scenario `pop_15_medium`, `minimum_separation` decreased by 0.000 metres (+0.0%) compared to baseline `pop_05_baseline`.
- For scenario `pop_15_medium`, `total_elapsed_time` decreased by 0.000 seconds (+0.0%) compared to baseline `pop_05_baseline`.
- For scenario `pop_30_high`, `area_density` increased by 0.025 ped/m² (+500.0%) compared to baseline `pop_05_baseline`.
- For scenario `pop_30_high`, `exit_throughput` delta could not be computed (decision: NOT_COMPARABLE).
- For scenario `pop_30_high`, `mean_speed` decreased by 0.000 m/s (+0.0%) compared to baseline `pop_05_baseline`.
- For scenario `pop_30_high`, `minimum_separation` decreased by 0.000 metres (+0.0%) compared to baseline `pop_05_baseline`.
- For scenario `pop_30_high`, `total_elapsed_time` decreased by 0.000 seconds (+0.0%) compared to baseline `pop_05_baseline`.

#### 2. Plausible Explanations & Confounders:
- Differences in clearance time and crowd density reflect physical changes in layout geometry and agent kinematic bottlenecks.
- Because the simulation dynamics are deterministic, variation across runs with identical initial conditions is 0.0 unless stochastic initial agent generators are used.

#### 3. What Cannot Be Concluded:
- **No Composite Ranking:** Scenarios are not ranked using composite weighted scores or artificial indexes.
- **No Unverified Causal Attribution:** In multi-factor exploratory scenarios, observed changes cannot be attributed solely to a single parameter without isolating each factor independently.
- **ERSI / Delta Score:** Intentionally excluded due to absence of authoritative mathematical specifications in repository scope.

---

## 8. Unavailable Metrics and Reasons

- **Metric `exit_throughput` (Scenario `pop_05_baseline`):** UNAVAILABLE. Reason: Run pop_05_baseline_run_01: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).; Run pop_05_baseline_run_02: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).
- **Metric `exit_throughput` (Scenario `pop_15_medium`):** UNAVAILABLE. Reason: Run pop_15_medium_run_01: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).; Run pop_15_medium_run_02: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).
- **Metric `exit_throughput` (Scenario `pop_30_high`):** UNAVAILABLE. Reason: Run pop_30_high_run_01: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).; Run pop_30_high_run_02: Metric status is UNAVAILABLE (Authoritative discrete exit events not emitted by engine.).

---

## 9. Failed Runs and Reasons

*All attempted simulation runs completed successfully with zero runtime failures.*

---

## 10. Reproducibility Metadata

- **Repetition Policy & Sample Type:** `deterministic_duplicate`
- **Seed Engine Usage:** `seed_used_by_engine=False`. *As inspected in the current Pavan SimulationEngine (genevac.simulation.engine), the engine's kinematics are purely deterministic and do not consume PRNG seeds during physics stepping. Supplied seeds are recorded as scenario intent only.*
- **Scenario Configuration Fingerprints (SHA-256):**
  - `pop_05_baseline`: `b2b8fed82b721a1e7924cbbf8885e88c2a7435fc070d61c929ececbb475f2e20`
  - `pop_15_medium`: `1a4087494b6939b6ee51ac85082cc1e33ee31055e38df63a0f003aedf1716156`
  - `pop_30_high`: `aa18520b368d445c1310a58284d60a25fd14d97d5478e83e6efb56d717f7feda`
- **Notes:** Executed 2 repetitions per scenario. All multi-run metrics produced zero variance (std_dev = 0.0) due to deterministic simulation kinematics. These are deterministic duplicates and must not be treated as independent stochastic samples.

---

## 11. Limitations & Boundaries

- **Directly Measured vs. Derived:** Physical trajectories, step durations, and boundary collisions are measured directly; densities, clearance times, and mean velocities are mathematically derived.
- **No Composite Scoring:** GenEvac-Sentinel strictly rejects artificial composite scoring, weighting schemes, or opaque ranking formulas.
- **Zero Inferred Evacuations:** Agents missing from snapshot observations are marked `not_observed` and are never assumed evacuated.
- **Simulation Engine Independence:** Person B analytics maintain a strictly read-only relationship with Pavan's simulation engine.
