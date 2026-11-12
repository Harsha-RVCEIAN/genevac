# Phase 07 Technical Reference: Baseline Trajectory Prediction

**Document Version:** 1.0.0  
**Phase:** 07 — Baseline Trajectory Prediction  
**Subsystem:** Module B Part 2 (Person B / Harsha)  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. Executive Summary

The baseline trajectory prediction subsystem provides a transparent, deterministic, timestamp-aware benchmark model based on pure kinematic constant-velocity extrapolation (`ConstantVelocityPredictor`). It consumes immutable simulation observation histories from `TrajectorySnapshot` and produces structured, validated future trajectory estimates (`TrajectoryPrediction`) without mutating simulation state or relying on future ground-truth data.

This model serves as the foundational comparative baseline against which advanced stochastic and learning-based trajectory models (such as the GenEvac-Sentinel Evacuation MID) are evaluated.

---

## 2. Mathematical Formulation & Kinematic Derivation

### 2.1 Velocity Estimation
Given an agent's chronological observation sequence $S = (o_1, o_2, \dots, o_m)$ with $m \ge 2$, where each observation $o_i$ provides spatial position $p_i = (x_i, y_i) \in \mathbb{R}^2$ and authoritative physical timestamp $t_i \in \mathbb{R}_{\ge 0}$:

1. Let $o_{\text{prev}} = o_{m-1}$ and $o_{\text{last}} = o_m$ be the two most recent valid observations.
2. The authoritative physical time interval is:
   $$\Delta t_{\text{obs}} = t_{\text{last}} - t_{\text{prev}}$$
3. The instantaneous velocity vector $\mathbf{v} = (v_x, v_y)$ is estimated as:
   $$v_x = \frac{x_{\text{last}} - x_{\text{prev}}}{\Delta t_{\text{obs}}}, \quad v_y = \frac{y_{\text{last}} - y_{\text{prev}}}{\Delta t_{\text{obs}}}$$

### 2.2 Future Trajectory Extrapolation
For a prediction horizon $H \in \mathbb{N}^+$ and prediction interval $\delta > 0$ (defaulting to the snapshot timestep $\Delta t$):
1. The future evaluation timestamps are defined relative to the snapshot reference timestamp $t_{\text{ref}}$:
   $$t_{\text{future}}(k) = t_{\text{ref}} + k \cdot \delta, \quad \text{for } k \in \{1, 2, \dots, H\}$$
2. The future position coordinates are evaluated directly from $o_{\text{last}}$:
   $$x_{\text{future}}(k) = x_{\text{last}} + v_x \cdot (t_{\text{future}}(k) - t_{\text{last}})$$
   $$y_{\text{future}}(k) = y_{\text{last}} + v_y \cdot (t_{\text{future}}(k) - t_{\text{last}})$$

### 2.3 Drift Avoidance & Numerical Stability
- **Direct vs. Cumulative Extrapolation:** Rather than iteratively updating $p_{k} = p_{k-1} + \mathbf{v} \cdot \delta$, the baseline directly projects from the authoritative last position $p_{\text{last}}$. This eliminates cumulative floating-point roundoff errors over extended prediction horizons.
- **Timestamp Awareness vs. Displacement Assumption:** Outdated models that assume uniform timestep spacing calculate displacement per step ($\Delta x = x_2 - x_1$) and multiply by future steps. When observation intervals vary (e.g. cadence filtering at 0.4s on a 0.1s simulation engine, or skipped steps due to network latency), step-displacement models produce catastrophic errors. The timestamp-aware formula rigorously handles arbitrary positive $\Delta t_{\text{obs}}$ and $\delta$.

---

## 3. Contract Enforcement & Error Handling

`ConstantVelocityPredictor` enforces strict preconditions defined in `src/genevac/prediction/data_contract.py`:

| Condition | Verification Rule | System Behavior |
| :--- | :--- | :--- |
| **Insufficient History** | $\text{len}(S) < 2$ | Raises `TrajectoryValidationError("Insufficient history...")` |
| **Invalid Horizon** | $H \le 0$ or $H \notin \mathbb{Z}$ | Raises `TrajectoryValidationError("horizon must be a positive integer")` |
| **Invalid Interval** | $\delta \le 0$ or not finite | Raises `TrajectoryValidationError("prediction_interval must be a finite positive number")` |
| **Non-positive Interval** | $t_{\text{last}} \le t_{\text{prev}}$ | Raises `TrajectoryValidationError("Non-positive or non-finite time delta")` |
| **Non-finite Output** | Non-finite velocity or coordinate | Raises `TrajectoryValidationError("Non-finite velocity...")` |
| **Stale Observation** | $t_{\text{ref}} - t_{\text{last}} > \tau_{\max}$ | Raises `TrajectoryValidationError("Stale observation...")` |
| **Future Observation** | $t_{\text{obs}} > t_{\text{ref}} + 10^{-9}$ | Raises `TrajectoryValidationError("Future observation...")` |

In batch execution (`predict_batch`), errors for individual pedestrians are captured and returned in the result dictionary without halting predictions for valid agents.

---

## 4. Assumptions & Fundamental Limitations

While computationally lightweight ($\approx 0.14\text{ ms}$ for 50 agents on CPU) and deterministic, constant-velocity prediction has significant scientific limitations in emergency evacuation contexts:

1. **Zero Acceleration ($\mathbf{a} = 0$):** Cannot anticipate changes in pedestrian velocity, deceleration approaching bottlenecks, or sudden acceleration when an exit opens.
2. **Ignorance of Geometry:** Trajectory extrapolation is unconstrained by walls, partitions, or obstacles. An agent heading toward a closed wall will be predicted to pass straight through it.
3. **Ignorance of Social Repulsion & Congestion:** Does not account for crowd density, interpersonal separation, or queue formation at exit portals.
4. **Deterministic Single-Path Output:** Generates exactly one deterministic future trajectory. It cannot represent the multi-modal indeterminacy of tactical human choices (e.g., choosing between left and right exits or detouring around congestion).
5. **High Sensitivity to Observation Noise:** Because velocity is derived from only the final two observations, localized sensor or stepping jitter translates directly into heading and velocity errors across the entire prediction horizon.

---

## 5. Verification Evidence

The baseline implementation is validated by a dedicated test suite (`tests/test_baseline_trajectory_phase07.py`) comprising 13 tests:

```text
tests/test_baseline_trajectory_phase07.py:
- test_straight_line_diagonal_positive ................... PASSED
- test_straight_line_negative_velocity .................. PASSED
- test_stationary_pedestrian_retains_exact_coordinates ... PASSED
- test_changing_velocity_evaluation ...................... PASSED
- test_irregular_observation_intervals .................. PASSED
- test_insufficient_history_raises_error ................ PASSED
- test_invalid_dt_and_future_observation_rejected ....... PASSED
- test_stale_observation_enforcement .................... PASSED
- test_missing_agent_batch_prediction_yields_validation_error PASSED
- test_run_reset_prevents_history_bleed ................. PASSED
- test_multiple_pedestrians_batch_predictions ........... PASSED
- test_deterministic_reproducibility ..................... PASSED
- test_zero_simulation_mutation .......................... PASSED
```

All 13 tests pass with zero failures and confirm complete compliance with GenEvac-Sentinel contracts.

