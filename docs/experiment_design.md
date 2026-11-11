# GenEvac-Sentinel: Experiment Design Matrix

**Project:** GenEvac-Sentinel  
**Document:** `docs/experiment_design.md`  
**Author:** Harsha (Person B / Module B Part 2)  
**Date:** 2026-11-09  

---

## 1. Experiment Matrix

| Experiment ID | Research Question | Independent Variable | Controlled Variables | Primary Outcomes | Baseline Condition | Repetitions / Seeds | Status |
|---|---|---|---|---|---|---|---|
| **EXP-01-CADENCE** | Impact of observation sampling interval on trajectory prediction ADE/FDE | Observation cadence $\Delta t_{\text{obs}} \in \{0.1, 0.2, 0.4, 0.8\}\text{ s}$ | Constant Velocity predictor, $H=12$, corridor scenario | ADE ($m$), FDE ($m$), Latency ($\mu s$) | 0.1 s baseline | 5 seeds (42..46) | **DESIGNED** |
| **EXP-02-BOTTLENECK** | Evacuation clearance time sensitivity to door bottleneck width | Exit width $w \in \{1.0, 1.5, 2.0, 3.0\}\text{ m}$ | Building layout 50x20m, 50 agents, desired speed 1.34 m/s | Clearance time ($s$), Mean density ($peds/m^2$), Min separation ($m$) | Exit width 2.0m | 5 seeds (42..46) | **DESIGNED** |
| **EXP-03-CLOSURE** | Delay and rerouting penalty under unexpected exit closure | Exit closure step $S_{\text{close}} \in \{20, 50, \infty\}$ | 2 exits, 60 agents, layout 50x20m | $\Delta \text{Clearance Time}$, blocked agent count, throughput | No closure ($S = \infty$) | 5 seeds (42..46) | **DESIGNED** |
| **EXP-04-PRED-HORIZON**| Degradation of Constant Velocity prediction error over future horizons | Prediction horizon $H \in \{3, 6, 9, 12, 20\}$ | Standard 0.4s observation cadence, 10 Hz engine | Horizon-wise displacement error $\text{Err}(h)$ | $H=3$ steps | 5 seeds (42..46) | **DESIGNED** |

---

## 2. Configuration Validation Specification
The experiment runner enforces schema validation before execution:
1. `experiment_id`: non-empty string.
2. `repetitions`: integer $\ge 1$.
3. `scenarios`: non-empty list of valid scenario configuration dicts.
4. `base_seed`: valid non-negative integer.
5. `metrics`: list of registered metric names.

