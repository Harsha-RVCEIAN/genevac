# Phase 04 Closure Fixes Report

**Project:** GenEvac-Sentinel  
**Document:** `docs/phase04_closure_fixes.md`  
**Author:** Harsha (Person B / Module B Part 2)  
**Date:** 2026-11-09  

---

## 1. Summary of Hardening Fixes

### A1. Dwell-Time Interval Attribution
- **Previous limitation:** Dwell time added instantaneous `occupancy * dt` without tracking per-agent intervals.
- **Implemented Fix:**
  - Implemented per-agent state tracking: `_agent_prev_state: Dict[str, Tuple[float, int, int]]` storing `(prev_timestamp, prev_row, prev_col)`.
  - **Policy:** Sample-hold / left-endpoint attribution. Elapsed interval $\Delta t = t_{\text{curr}} - t_{\text{prev}}$ is attributed to the previous cell where the agent was observed.
  - **First Observation:** Yields $0.0$ elapsed dwell time (initial observation only records initial state).
  - **Maximum Gap Protection:** Intervals where $\Delta t > \text{max\_observation\_gap}$ (default 2.0s) or $\le 0.0$ are skipped without accumulation.
  - **Out-of-Order Rejection:** Non-monotonic timestamps raise explicit `ValueError`.

### A2. Run Identity & Duplicate Protection Across All Heatmap Layers
- **Implemented Fix:**
  - Added `run_id` parameter to `aggregate_snapshot(snapshot, run_id="default_run", ...)`.
  - Enforced run isolation: Attempting to process snapshots from different `run_id`s in the same accumulator raises `ValueError`. Callers must invoke `reset(new_run_id)`.
  - Composite deduplication key: `(run_id, step_index)` tracked in `_seen_snapshot_keys`.
  - Re-submitting a previously processed snapshot returns a result marked `is_duplicate=True` with zero new occupancy or dwell accumulation.

