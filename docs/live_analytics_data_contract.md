# Live Analytics Data Contract

**Project:** GenEvac-Sentinel  
**Document:** `docs/live_analytics_data_contract.md`  
**Date:** 2026-11-09  

---

## 1. Schema Definitions

### 1.1 MetricRecord
- `metric_id: str`: Unique identifier (e.g. `"active_count"`, `"min_separation"`).
- `value: Optional[float]`: Numeric value, or `None` if unavailable.
- `unit: str`: Physical unit (e.g. `"m"`, `"s"`, `"peds/s"`).
- `status: str`: `"VALID"`, `"UNAVAILABLE"`, `"STALE"`, or `"NOT_COMPUTED"`.
- `sample_count: int`: Number of contributing agents.
- `source_step: int`: Step index at which value was computed.
- `source_timestamp: float`: Simulation timestamp (seconds).
- `reason: str`: Error or unavailability explanation.

### 1.2 LiveAnalyticsSnapshot
- `run_id: str`: Stable simulation run identifier.
- `source_step: int`: Current engine step.
- `source_timestamp: float`: Simulation clock time ($s$).
- `refresh_timestamp: float`: Wall-clock time when analytics last updated.
- `has_new_data: bool`: False if engine has not advanced.
- `metrics: Dict[str, MetricRecord]`: Mapping of metric_id -> record.
- `spatial_result: Optional[SpatialGridResult]`: Phase 04 heatmap layers.

