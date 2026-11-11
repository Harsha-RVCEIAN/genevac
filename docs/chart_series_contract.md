# Chart Series Contract & Retention Specification

**Project:** GenEvac-Sentinel  
**Document:** `docs/chart_series_contract.md`  
**Date:** 2026-11-09  

---

## 1. Schema Definitions

### 1.1 ChartPoint
- `timestamp: float`: Simulation timestamp (seconds).
- `step_index: int`: Simulation step.
- `value: Optional[float]`: Numeric value.
- `sample_count: int`: Supporting sample count.
- `run_id: str`: Run identity.

### 1.2 ChartSeries
- `series_id: str`: Unique identifier.
- `metric_id: str`: Associated metric name.
- `display_label: str`: User-facing chart title.
- `unit: str`: Physical unit.
- `run_id: str`: Run identity.
- `points: Tuple[ChartPoint, ...]`: Chronologically sorted points.
- `max_points: int`: Rolling window limit (default: 500).

### Invariants:
1. **Deduplication:** A point with the same `(run_id, step_index)` is never appended twice.
2. **Chronological Sorting:** Enforces strictly increasing timestamp order.
3. **Bounded Memory:** History is truncated from the left to preserve the latest `max_points`.

