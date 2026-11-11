---
trigger: always_on
---

# Coding and Testing Rules

## Implementation

- Inspect relevant source and tests before editing.
- Prefer small, additive, scoped changes.
- Reuse existing classes/functions and public APIs.
- Keep calculations separate from rendering and UI.
- Do not duplicate metric formulas in charts or callbacks.
- Do not invent metric formulas, scenario definitions, or MID algorithms.
- Keep configuration, units, assumptions, and data provenance explicit.
- Do not add dependencies without approval.

## Metrics and heatmaps

For each metric document formula, unit, population, inclusion/exclusion rules, time window, aggregation, missing-data policy, and authoritative source.

For heatmaps document coordinate bounds, grid resolution/cell size, accumulated quantity, time window, normalization, and boundary behavior. Missing data must not silently become zero.

## Prediction and evaluation

- Reuse the existing baseline predictor if it meets the approved requirement.
- Use chronological observations and actual time deltas where required.
- Align predictions and ground truth by pedestrian/run identity and timestamp/horizon, not array position alone.
- Retain per-run results as well as aggregates.
- Do not claim accuracy or validation without appropriate ground truth and evidence.

## MID

The project schedule mentions “MID inference” without defining it. Find an authoritative project definition or ask Harsha. Do not guess what MID stands for or implement a guessed method.

## Testing

- Inspect existing tests first; never weaken/delete tests to make the suite pass.
- Add focused tests for every behavior change.
- Cover boundary, invalid, empty, identity, ordering, timestamp, coordinate, lifecycle, and repeatability cases as applicable.
- Run relevant narrow tests, then integration/regression tests, then broader tests if feasible.
- Report exact commands and actual pass/fail/skip counts.
- Distinguish unit tests, mocked integration, and real Part 1 engine integration.
