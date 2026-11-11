# Phase 05 Validation Examples & Controlled Replay Fixture

**Project:** GenEvac-Sentinel  
**Document:** `docs/phase05_validation_examples.md`  
**Date:** 2026-11-09  

---

## 1. Hand-Calculated Replay Walkthrough

### Scenario Setup:
- 2 Pedestrians:
  - Agent A (`p1`): starts at $(0.0, 0.0)$, moves $+1.0$ m in X each 0.4s observation step.
  - Agent B (`p2`): starts at $(10.0, 10.0)$, stays stationary.
- Grid: $20\text{m} \times 20\text{m}$, cell size $5.0\text{m} \times 5.0\text{m}$ ($4 \times 4$ cells).
- Cadence: $0.4$ s.

### Step 0 ($t = 0.0$ s):
- Active count: 2
- Blocked count: 0
- Evacuated count: 0
- Min separation: $\sqrt{10^2 + 10^2} \approx 14.142\text{ m}$
- Occupancy: cell $[0, 0] = 1$, cell $[2, 2] = 1$
- Dwell time: 0.0 (initial snapshot)

### Step 4 ($t = 0.4$ s):
- Agent A at $(1.0, 0.0)$, Agent B at $(10.0, 10.0)$
- Min separation: $\sqrt{9^2 + 10^2} = \sqrt{181} \approx 13.454\text{ m}$
- Dwell time: cell $[0, 0] = 0.4$ agent-s, cell $[2, 2] = 0.4$ agent-s
- Repeated poll without advancing step: `has_new_data = False`, chart points not appended.

