# Phase 02 Requirement-to-Code Inventory

**Project:** GenEvac-Sentinel  
**Workstream:** Module B Part 2 (Harsha)  
**Date:** 2026-11-09  

| Requirement | Existing Path / Symbol | Input / Output | Formula / Behavior | Tests | Status | Action |
|---|---|---|---|---|---|---|
| **ADE (Trajectory)** | `src/genevac/metrics/__init__.py:calculate_ade` & `src/genevac/prediction/evaluation.py:PredictionEvaluator` | Inputs: ground truth & predicted positions; Output: float (meters) | $\frac{1}{T}\sum \|\hat{p}_t - p_t\|_2$ | `tests/test_metrics.py`, `tests/test_evaluation_part4.py` | **REUSE** | Standardize in `src/genevac/metrics/trajectory.py` with timestamp/identity alignment & sample validation. |
| **FDE (Trajectory)** | `src/genevac/metrics/__init__.py:calculate_fde` & `src/genevac/prediction/evaluation.py:PredictionEvaluator` | Inputs: ground truth & predicted positions; Output: float (meters) | $\|\hat{p}_{t_f} - p_{t_f}\|_2$ | `tests/test_metrics.py`, `tests/test_evaluation_part4.py` | **REUSE** | Standardize in `src/genevac/metrics/trajectory.py` with final valid timestamp alignment. |
| **Prediction Coverage** | `src/genevac/prediction/evaluation.py:PredictionEvaluator` | Inputs: samples; Output: float ratio $[0, 1]$ | $\frac{N_{\text{matched}}}{N_{\text{expected}}}$ | `tests/test_evaluation_part4.py` | **REUSE** | Expose in `metrics/trajectory.py`. |
| **Clearance / Evac Time** | `src/genevac/metrics/__init__.py:calculate_clearance_time` | Inputs: step count, dt; Output: float seconds | $T = \text{steps} \times dt$ | `tests/test_metrics.py` | **REUSE** | Retain in `metrics/evacuation.py`. |
| **Per-Agent Evacuation Time** | Not present | Inputs: agent trajectories or engine events; Output: dict of agent_id -> float (s) | Arrival time at exit or terminal state timestamp | None | **IMPLEMENT** | Implement in `metrics/evacuation.py`. |
| **Evacuated Count & Rate** | Not present | Inputs: sequence of snapshots / events; Output: time series of counts / rates | Aggregated evacuated status count per snapshot | None | **IMPLEMENT** | Implement in `metrics/evacuation.py`. |
| **Exit Throughput** | Not present | Inputs: exit events, interval $\Delta t$; Output: agents/second | $\frac{\Delta N_{\text{evacuated}}}{\Delta t}$ per exit | None | **IMPLEMENT** | Implement in `metrics/evacuation.py`. |
| **Bottleneck Congestion / Density** | Not present | Inputs: agent positions, area geometry; Output: agents $/ m^2$ | $\rho = \frac{N}{A}$ | None | **IMPLEMENT** | Implement in `metrics/crowd.py`. |
| **Minimum Pedestrian Separation** | Not present | Inputs: snapshot pedestrian positions; Output: float (meters) | $\min_{i \ne j} \|p_i - p_j\|_2$ | None | **IMPLEMENT** | Implement in `metrics/crowd.py`. |
| **Route & Path Checks (1-13)** | Not present | Inputs: Path points, bounds, obstacles; Output: structured `PathValidationResult` | Geometric & boundary checks (length, bounds, crossings, clearance) | None | **IMPLEMENT** | Implement comprehensive deterministic checks in `metrics/route_checks.py`. |
| **Geometric Route Length** | Not present | Inputs: path points; Output: float meters | $L = \sum \|q_{i+1} - q_i\|_2$ | None | **IMPLEMENT** | Implement in `metrics/route_checks.py`. |
| **ERSI (Evacuation Route Safety Index)** | None in repo | Inputs: route / crowd / risk parameters; Output: safety index | Not defined in repo or specs | None | **BLOCKED** | Formula, weights, and criteria unstated. Marked BLOCKED; spec documented. |
| **Delta Score** | None in repo | Inputs: baseline vs. alternative; Output: comparison score | Not defined in repo or specs | None | **BLOCKED** | Mathematical definition unstated. Marked BLOCKED; spec documented. |
| **Metric Aggregation** | `src/genevac/prediction/evaluation.py:AggregateEvaluationReport` | Inputs: sequence of results; Output: aggregate metrics | Mean, median, sample counts, exclusions | `tests/test_evaluation_part4.py` | **REUSE / EXTEND** | Provide general `MetricSummary` in `metrics/aggregation.py`. |
| **Experiment Configuration Validation** | `src/genevac/experiments/__init__.py:ExperimentRunner` | Inputs: scenario config dict; Output: validation result | Schema checks (id, repeats, seed, scenarios) | `tests/test_experiments.py` | **EXTEND** | Add `validate_experiment_config` in `src/genevac/experiments/validation.py`. |

