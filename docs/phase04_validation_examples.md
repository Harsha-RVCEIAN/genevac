# Phase 04 Validation Examples & Hand-Calculated Fixtures

**Project:** GenEvac-Sentinel  
**Document:** `docs/phase04_validation_examples.md`  
**Date:** 2026-11-09  

---

## 1. Hand-Calculated Numerical Fixture

### Grid Configuration:
- Bounds: $\min_x = 0.0, \min_y = 0.0, \max_x = 10.0, \max_y = 10.0$
- Cell size: $w = 2.0\text{ m}, h = 2.0\text{ m}$
- Grid dimensions: $\text{rows} = 5, \text{cols} = 5$
- Cell area: $A = 2.0 \times 2.0 = 4.0\text{ m}^2$

### Input Agents at $t = 0.0$:
- Agent A: $(1.0, 1.0) \to \text{col} = \lfloor 1.0 / 2 \rfloor = 0, \text{row} = \lfloor 1.0 / 2 \rfloor = 0$
- Agent B: $(1.5, 1.5) \to \text{col} = \lfloor 1.5 / 2 \rfloor = 0, \text{row} = \lfloor 1.5 / 2 \rfloor = 0$
- Agent C: $(5.0, 5.0) \to \text{col} = \lfloor 5.0 / 2 \rfloor = 2, \text{row} = \lfloor 5.0 / 2 \rfloor = 2$
- Agent D: $(10.0, 10.0) \to \text{boundary max edge mapped to final valid cell}: \text{col} = 4, \text{row} = 4$
- Agent E: $(15.0, 5.0) \to \text{out of bounds (rejected)}$

### Expected Occupancy Grid $[5, 5]$:
- Cell $[0, 0] = 2$ (Agents A, B)
- Cell $[2, 2] = 1$ (Agent C)
- Cell $[4, 4] = 1$ (Agent D)
- All other cells $= 0$

### Expected Physical Density Grid $[5, 5]$ ($peds / m^2$):
- Cell $[0, 0] = 2 / 4.0 = 0.5\text{ peds/m}^2$
- Cell $[2, 2] = 1 / 4.0 = 0.25\text{ peds/m}^2$
- Cell $[4, 4] = 1 / 4.0 = 0.25\text{ peds/m}^2$
- All other cells $= 0.0\text{ peds/m}^2$

### Dwell Time over 2 consecutive snapshots ($\Delta t = 0.4\text{ s}$):
- If Agents A and B remain in $[0, 0]$ for both snapshots:
  - Dwell time at $[0, 0] = 2\text{ agents} \times 0.4\text{ s} = 0.8\text{ agent}\cdot\text{s}$.

