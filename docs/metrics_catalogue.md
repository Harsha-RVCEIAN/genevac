# GenEvac-Sentinel: Metrics Catalogue

**Project:** GenEvac-Sentinel  
**Module:** `src/genevac/metrics/`  
**Author:** Harsha (Person B / Module B Part 2)  
**Date:** 2026-11-09  

---

## 1. Trajectory Prediction Metrics

### 1.1 Average Displacement Error (ADE)
- **Formula:** $\text{ADE} = \frac{1}{T}\sum_{t=1}^{T} \|\hat{p}_t - p_t\|_2$
- **Units:** Meters ($m$)
- **Population:** Agents with valid trajectory predictions aligned with future ground-truth within timestamp tolerance ($\le 10^{-4}$ s).
- **Missing Data Policy:** Excluded from average if zero points match; reported in exclusion count.
- **Reference Implementation:** `src/genevac/metrics/trajectory.py:compute_ade`

### 1.2 Final Displacement Error (FDE)
- **Formula:** $\text{FDE} = \|\hat{p}_{t_f} - p_{t_f}\|_2$
- **Units:** Meters ($m$)
- **Population:** Agents with valid match at final target timestamp $t_f$.
- **Missing Data Policy:** Marked invalid / `NaN` if final timestamp is missing.
- **Reference Implementation:** `src/genevac/metrics/trajectory.py:compute_fde`

### 1.3 Prediction Coverage
- **Formula:** $\text{Coverage} = \frac{N_{\text{matched}}}{N_{\text{expected}}}$
- **Units:** Ratio $[0, 1]$
- **Population:** All evaluation windows generated across agents.

---

## 2. Evacuation & Crowd Flow Metrics

### 2.1 Total Clearance Time
- **Formula:** $T_{\text{clear}} = N_{\text{steps}} \times dt$
- **Units:** Seconds ($s$)
- **Population:** Run-level metric, evaluated at point when all active pedestrians reach an exit or simulation terminates.

### 2.2 Per-Agent Evacuation Time
- **Formula:** $t_{\text{evac}}(i) = t_{\text{exit}}(i) - t_{\text{start}}(i)$
- **Units:** Seconds ($s$)
- **Population:** Pedestrians with status `EVACUATED`.

### 2.3 Exit Throughput
- **Formula:** $\Phi_{\text{exit}}(t, \Delta t) = \frac{\Delta N_{\text{evacuated}}(t, t+\Delta t)}{\Delta t}$
- **Units:** Pedestrians per second ($peds/s$)
- **Population:** Per exit per time window $\Delta t$.

### 2.4 Minimum Pedestrian Separation
- **Formula:** $d_{\min}(t) = \min_{i \ne j} \|p_i(t) - p_j(t)\|_2$
- **Units:** Meters ($m$)
- **Population:** All active pedestrians at time $t$.

### 2.5 Bottleneck / Area Density
- **Formula:** $\rho(A, t) = \frac{N(A, t)}{\text{Area}(A)}$
- **Units:** Pedestrians per square meter ($peds/m^2$)
- **Population:** Active pedestrians inside designated boundary $A$.

