# ERSI (Evacuation Route Safety Index) and Delta Score Specification & Blocker Analysis

**Project:** GenEvac-Sentinel  
**Workstream:** Module B Part 2 (Harsha)  
**Status:** BLOCKED / Gated Pending Authoritative Project Brief  
**Date:** 2026-11-09  

---

## 1. Evacuation Route Safety Index (ERSI) Investigation

### 1.1 Status
**BLOCKED**

### 1.2 Investigation Details
A repository-wide search (`git grep -i "ERSI"` and inspection of all documentation under `docs/` and `pavan-part1-reference/`) reveals zero occurrences of the term **ERSI** or **Evacuation Route Safety Index**.

### 1.3 Missing Specifications Required Before Implementation
To implement ERSI without speculative invention, the following authoritative specifications must be provided by Harsha or project stakeholders:
1. **Mathematical Formula**: Is ERSI a linear weighted sum, geometric mean, penalty-based score, or probabilistic risk model?
2. **Included Factors**: Does it evaluate:
   - Geometric path length / detour ratio?
   - Static clearance from walls and obstacles?
   - Dynamic crowd density / bottleneck exposure along the path?
   - Exit congestion / queue length?
   - Visibility or smoke/hazard exposure?
3. **Weights and Scaling**: Exact coefficients $w_i$, normalization denominators, and score boundaries (e.g. $[0, 1]$ or $[0, 100]$).
4. **Directionality**: Does a higher score mean *safer* (safety index) or *more dangerous* (risk index)?
5. **Handling of Blocked / Infeasible Routes**: Explicit output when a route intersects an obstacle or closed exit (e.g., $0.0$, $\infty$, or `NaN`/`None`).

### 1.4 Phase 02 Action
Per Global Execution Rule 5 ("No guessed contracts") and Phase 02 guidelines:
- We **do not invent** an arbitrary formula or weights.
- ERSI is declared **BLOCKED**.
- In place of an ungrounded synthetic score, we provide 13 rigorous, deterministic geometric and spatial path validation checks in `src/genevac/metrics/route_checks.py`.

---

## 2. Delta Score Investigation

### 2.1 Status
**BLOCKED**

### 2.2 Investigation Details
A search for `Delta Score`, `DeltaScore`, or comparative differential indices yielded zero definitions in project documentation and code.

### 2.3 Missing Specifications Required Before Implementation
1. **Compared Quantities**: Does Delta Score compare:
   - Evacuation clearance time under Scenario A vs. Scenario B ($\Delta T = T_B - T_A$)?
   - Prediction accuracy between baseline Constant Velocity and neural/MID models ($\Delta \text{ADE}$)?
   - Route safety score between primary and alternative evacuation paths?
   - Crowd congestion before and after an exit closure?
2. **Formulation**: Signed difference ($v_{\text{alt}} - v_{\text{base}}$), absolute difference ($|v_{\text{alt}} - v_{\text{base}}|$), relative ratio ($\frac{v_{\text{alt}} - v_{\text{base}}}{v_{\text{base}}}$), or normalized z-score?
3. **Reference Baseline**: What is the canonical reference baseline across runs?

### 2.4 Phase 02 Action
- Delta Score is declared **BLOCKED**.
- Comparative scenario metrics will compute transparent, unadulterated metric differences $(\Delta \text{metric} = \text{val}_2 - \text{val}_1)$ with units preserved in Phase 06 rather than synthesizing an unverified single-number score.

