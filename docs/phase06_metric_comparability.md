# Phase 06 — Metric Comparability Policy and Audit Gate

**Document Version:** 1.0.0  
**Phase:** 06 — Scenario Comparison  
**Owner:** Module B Part 2 (Person B / Harsha)  

---

## 1. Objective
Cross-scenario comparisons are invalid if underlying metrics differ in definition, measurement units, evaluation horizons, spatial domain boundaries, or population sizes. The `ComparabilityGate` enforces verification before any mathematical delta or cross-scenario chart is produced.

---

## 2. Comparability Evaluation Dimensions

1. **Dimensional Consistency (Units):**
   - Direct mathematical differences ($\Delta = \mu_v - \mu_b$) require exact unit equality.
   - If unit strings differ (`seconds` vs `hours`), decision is `NOT_COMPARABLE` (`UNIT_MISMATCH`).
2. **Metric Definition Version:**
   - Implementations with mismatched version tags are rejected (`DEFINITION_VERSION_MISMATCH`).
3. **Trajectory Prediction Horizons (ADE/FDE):**
   - Multi-step forecasts must be evaluated across identical horizons (e.g., both 10 steps). Mismatches yield `NOT_COMPARABLE` (`HORIZON_MISMATCH`).
4. **Spatial Grids & Heatmaps:**
   - Spatial density matrices require identical coordinate bounding boxes and grid cell sizes. Cell-by-cell comparisons on mismatched resolutions are rejected (`GRID_RESOLUTION_MISMATCH`).
5. **Route and Navigation Destinations:**
   - Path length comparisons require identical navigation destinations. Comparing a path targeting `exit_west` with one targeting `exit_east` is rejected (`DESTINATION_MISMATCH`).
6. **Simulated Duration & Cumulative Quantities:**
   - Cumulative totals across unequal time durations are rejected unless an explicit rate normalization policy is declared (`DURATION_MISMATCH`).
7. **Cohort & Population Counts:**
   - Unequal agent counts introduce crowd scaling effects. The gate marks these as `COMPARABLE_WITH_CAVEATS` (`POPULATION_MISMATCH`) and documents that raw totals reflect population size.
8. **Blocked Metrics (ERSI, Delta Score):**
   - Metrics with no authoritative specification are unconditionally rejected with `NOT_COMPARABLE` (`METRIC_BLOCKED`).

---

## 3. Decision Matrix

| Audit Condition | Decision Code | Cross-Scenario Aggregation Allowed? | Reporting Action |
| :--- | :--- | :--- | :--- |
| All dimensions match | `COMPARABLE` | **Yes** | Compute $\Delta$ and $\%\Delta$ |
| Minor population or duration nuance | `COMPARABLE_WITH_CAVEATS` | **Yes** | Compute $\Delta$, append caveat footnotes |
| Unit, definition, or grid mismatch | `NOT_COMPARABLE` | **No** | Mark $\Delta = \text{N/A}$, log rejection reason |
| Undefined metric (ERSI / Delta Score) | `NOT_COMPARABLE` | **No** | Flag `METRIC_BLOCKED`, exclude from analysis |
