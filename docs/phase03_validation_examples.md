# Phase 03 Validation and Field Conversion Examples

**Project:** GenEvac-Sentinel  
**Document:** `docs/phase03_validation_examples.md`  
**Date:** 2026-11-09  

---

## 1. Concrete Adaptation Fixtures

### 1.1 Snapshot Object Adaptation
```python
# Pavan's SimulationObservationSnapshot
snapshot = SimulationObservationSnapshot(
    snapshot_id="snap_000004",
    step_count=4,
    timestamp=0.4,
    dt=0.1,
    observation_interval=0.4,
    total_count=2,
    active_count=1,
    evacuated_count=0,
    blocked_count=1,
    inactive_count=0,
    pedestrians=(
        PedestrianObservation(
            pedestrian_id="p1", x=2.0, y=3.0, vx=0.5, vy=0.0, speed=0.5,
            status="active", target_exit_id="ex1", trajectory_history=()
        ),
        PedestrianObservation(
            pedestrian_id="p2", x=5.0, y=5.0, vx=0.0, vy=0.0, speed=0.0,
            status="blocked", target_exit_id=None, trajectory_history=()
        ),
    ),
)

# Adapted into Person B's SimulationTrajectoryAdapter
adapter = SimulationTrajectoryAdapter()
report = adapter.adapt_snapshot(snapshot)

# Results:
# report.accepted_observations == 2
# report.rejected_observations == 0
# "p1" in adapter.active_agents
# "p2" in adapter.blocked_agents
```

### 1.2 Rejected Record Handling
If an observation contains `x = float('nan')` or an empty `pedestrian_id = ""`:
- `report.rejected_observations == 1`
- `report.rejection_reasons` contains descriptive entry with agent ID and reason.
- Valid agents within the same snapshot are successfully recorded.

