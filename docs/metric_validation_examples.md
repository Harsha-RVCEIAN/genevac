# Metric Validation Fixtures and Numerical Examples

**Project:** GenEvac-Sentinel  
**Document:** `docs/metric_validation_examples.md`  
**Date:** 2026-11-09  

---

## 1. Hand-Calculated Fixtures

### 1.1 Trajectory Prediction ADE / FDE
- **Ground Truth**: $[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]$
- **Prediction**: $[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]$
  - Error at $t=1$: $0.0$
  - Error at $t=2$: $0.0$
  - Error at $t=3$: $0.0$
  - $\text{ADE} = 0.0\text{ m}$, $\text{FDE} = 0.0\text{ m}$.

- **Ground Truth**: $[(0.0, 0.0), (3.0, 4.0)]$
- **Prediction**: $[(0.0, 0.0), (0.0, 0.0)]$
  - Error at $t=1$: $0.0$
  - Error at $t=2$: $\sqrt{(3-0)^2 + (4-0)^2} = 5.0\text{ m}$
  - $\text{ADE} = \frac{0 + 5}{2} = 2.5\text{ m}$, $\text{FDE} = 5.0\text{ m}$.

### 1.2 Path Length
- **Path Points**: $[(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]$
  - Segment 1: $\|(3, 0) - (0, 0)\| = 3.0\text{ m}$
  - Segment 2: $\|(3, 4) - (3, 0)\| = 4.0\text{ m}$
  - Geometric Length: $L = 3.0 + 4.0 = 7.0\text{ m}$.
  - Euclidean distance: $d(q_0, q_2) = \sqrt{3^2 + 4^2} = 5.0\text{ m}$.
  - Detour ratio: $7.0 / 5.0 = 1.4$.

### 1.3 Exit Throughput
- In a 10.0 second window, 15 agents cross exit `"exit_main"`:
  - $\Phi = \frac{15\text{ agents}}{10.0\text{ s}} = 1.5\text{ agents/s}$.

### 1.4 Area Density
- Area $A = [0, 10] \times [0, 5]$, $\text{Area} = 50\text{ m}^2$.
- 25 pedestrians present inside $A$:
  - $\rho = \frac{25}{50} = 0.5\text{ peds/m}^2$.

### 1.5 Minimum Pedestrian Separation
- Pedestrians at $(0, 0)$, $(1, 0)$, and $(5, 5)$:
  - Pairwise distances: $d(p_1, p_2) = 1.0$, $d(p_1, p_3) = \sqrt{50} \approx 7.07$, $d(p_2, p_3) = \sqrt{41} \approx 6.40$.
  - $d_{\min} = 1.0\text{ m}$.

