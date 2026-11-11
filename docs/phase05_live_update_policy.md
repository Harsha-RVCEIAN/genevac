# Phase 05 Live Update & Polling Policy

**Project:** GenEvac-Sentinel  
**Document:** `docs/phase05_live_update_policy.md`  
**Date:** 2026-11-09  

---

## 1. Polling vs Event Architecture
- **Non-Invasive Polling:** The pipeline periodically calls `pipeline.poll_simulation(engine)`.
- **Zero Stepping:** Polling strictly inspects engine state; it never calls `engine.step()` or `engine.run()`.
- **Clock Independence:** The UI polling cadence (e.g. 250 ms) is independent of the simulation timestep ($dt = 0.1$ s).

## 2. No-New-Data and Stale Policy
- If `engine.current_step == last_observed_step`:
  - `has_new_data = False`
  - Existing chart points and dwell time are **not** updated or duplicated.
  - Stale status is flagged if wall-clock time exceeds `stale_timeout_seconds` (default: 5.0s).

## 3. Run Transition Policy
- If a step rollback or new `run_id` is detected:
  - `pipeline.reset(new_run_id)` is invoked.
  - Adapter, heatmap accumulators, and chart series are cleared to guarantee zero cross-run leakage.

