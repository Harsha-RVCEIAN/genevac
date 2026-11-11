# Phase 06 Technical Reference: Complete Integration, Repair & Verification

**Document Version:** 1.0.0  
**Phase:** 06 — Scenario Comparison  
**Author:** Module B Part 2 (Person B / Harsha)  
**Integration Target:** Module B Part 1 Simulation Engine (Pavan)  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary & Integration Architecture

Phase 06 establishes an auditable, scientifically disciplined comparison layer for pedestrian evacuation scenarios. It consumes simulation outputs from Pavan's Module B Part 1 simulation engine through non-invasive public interfaces, ingests discrete event streams, enforces run-scoped deduplication, evaluates metric comparability, and produces deterministic aggregate summaries.

```
       [ Upstream Simulation Engine (Pavan / Module B Part 1) ]
                 │                                │
                 │ engine.get_snapshot()          │ engine.events / latest_events
                 ▼                                ▼
       [ Snapshot Ingestion ]           [ Event Ingestion ]
      (PedestrianObservation)           (SimulationEvent: BLOCKED, etc.)
                 │                                │
                 └────────────────┬───────────────┘
                                  ▼
                 [ SimulationTrajectoryAdapter ]
        (Cadence, Cohorts, Run-Scoped Keys: (run_id, snap_id/step/evt_id))
                                  │
                                  ▼
                 [ Person B Analytics Subsystem ]
             ├── Phase 02 Metric Evaluators (Speed, Density, Separation)
             ├── Phase 04 Spatial Heatmap Engine (Occupancy, Dwell Time)
             └── Phase 05 Live Analytics & Chart Series (FIFO Bounded)
                                  │
                                  ▼
                 [ ScenarioComparisonRunner ]
             ├── Pre-execution Configuration Validation
             ├── Comparability Gate (Units, Windows, Populations, Availability)
             ├── Deterministic Aggregation (Per-run Preservation)
             └── Markdown Tables & Scientific Interpretation Reporting
```

---

## 2. Pavan Simulation Engine Compatibility Contract

### 2.1 Verified Public Interfaces
The adapter interacts strictly with documented public engine symbols and does not mutate engine state:
- `engine.get_snapshot()`: Returns an immutable `SimulationObservationSnapshot`.
- `engine.latest_snapshot`: Accesses the most recent snapshot without stepping.
- `engine.clock.step_count` / `engine.current_step`: Current integer step index.
- `engine.clock.timestamp` / `engine.timestamp`: Current authoritative simulation time (seconds).
- `engine.clock.dt` / `engine.timestep`: Integration delta (seconds).
- `engine.events`: Full tuple of all `SimulationEvent` records emitted during the run.
- `engine.latest_events`: Tuple of `SimulationEvent` records emitted during the latest step.
- `engine.get_events(event_type=..., pedestrian_id=...)`: Filtered query for simulation events.

### 2.2 Observation Snapshot Mapping
| Pavan Snapshot Field | Type | Person B Mapping / Usage |
| :--- | :--- | :--- |
| `snapshot_id` | `str` | Run-scoped deduplication key `(run_id, snapshot_id)` |
| `step_count` | `int` | Monotonic step counter and `(run_id, step_count)` duplicate gate |
| `timestamp` | `float` | Authoritative simulation time in seconds |
| `dt` | `float` | Simulation integration time step |
| `observation_interval` | `float` | Cadence filtering interval (default $0.4\text{ s}$) |
| `pedestrians` | `tuple` | Sequence of `PedestrianObservation` records |

### 2.3 Pedestrian Observation Mapping & Cohort State
Each pedestrian observation is mapped into canonical `TrajectoryObservation`:
- `pedestrian_id` $\to$ `agent_id`
- `(x, y)` $\to$ `position` (verified finite coordinates)
- `status`:
  - `ACTIVE`: Ingested into `TrajectoryHistoryBuffer`, assigned to `active_agents` cohort.
  - `BLOCKED`: Ingested into `TrajectoryHistoryBuffer`, assigned to `blocked_agents` cohort.
  - `EVACUATED`: Terminal state for current run. Recording ceases, prior history retained, assigned to `evacuated_agents`.
  - `INCAPACITATED`: Terminal state for current run. Recording ceases, assigned to `incapacitated_agents`.
  - `INACTIVE`: Completely ignored, assigned to `inactive_agents`.
- **Missing Agents:** Any agent previously active/blocked that is absent from a subsequent snapshot is placed into `not_observed_agents`. Missing agents are **never** assumed evacuated.
- `trajectory_history`: Preserved directly in `adapter.trajectory_histories[pedestrian_id]`.

### 2.4 Discrete Simulation Event Mapping
Pavan's engine emits structured, immutable `SimulationEvent` objects:
- `event_id`: Unique identifier within the run $\to$ deduplicated via `(run_id, event_id)`.
- `step`: Integer step at which event occurred.
- `timestamp`: Simulation timestamp.
- `event_type`: Standard types: `no_route`, `exit_closed`, `blocked`.
- `pedestrian_id`: Associated agent ID.
- `details`: Mapping with diagnostic metadata (e.g. `{"reason": "target_exit_closed", "exit_id": "exit_1"}`).
- **Double-Counting Prevention:** When an event indicates an agent is blocked, the agent is placed into `_blocked_from_events[run_id]`. Cohorts are maintained as sets, ensuring that an agent reported as blocked in both an event and a subsequent snapshot is counted exactly once.

---

## 3. Adapter Public Methods & Verification

| Method | Signature | Responsibility |
| :--- | :--- | :--- |
| `adapt_snapshot` | `(snapshot, force=False, run_id=None) -> SnapshotAdaptationReport` | Ingests snapshot with cadence filtering and `(run_id, snap_id)` deduplication |
| `adapt_event` | `(event, run_id=None) -> EventAdaptationReport` | Ingests single discrete event with `(run_id, event_id)` deduplication |
| `ingest_events` | `(events, run_id=None) -> List[EventAdaptationReport]` | Batch ingestion of simulation event records |
| `read_only_engine_ingest` | `(engine, run_id=None, force=False) -> (snap_rep, evt_reps)` | Ingests engine snapshot and events while executing read-only mutation checks |
| `update_from_simulation` | `(engine, force=False, run_id=None) -> int` | Direct read-only inspection of running simulation engine |
| `reset` | `(run_id=None) -> None` | Resets buffer and tracking, isolating run-scoped deduplication |
| `verify_engine_read_only` | `(engine, action) -> (is_unmutated, violations)` | Standalone test utility verifying zero engine mutation |

---

## 4. Metric Availability & Naming Rules

1. **Total Elapsed Simulation Time vs. Evacuation Clearance Time:**
   - `total_elapsed_time`: The simulation clock duration ($\text{step} \times \Delta t$). Always reported as `"Total Elapsed Simulation Time"`.
   - `clearance_time`: Reported as `"Evacuation Clearance Time"`. It is **VALID only when all initial agents have evacuated**. If any agent remains active, blocked, or incapacitated at the end of the run, `clearance_time` is reported as `UNAVAILABLE` with reason `"Evacuation incomplete: X/Y evacuated"`.
2. **Exit Throughput Rate:**
   - Requires authoritative exit events with discrete timestamps.
   - Because Pavan's engine does not emit discrete exit event logs, `exit_throughput` is truthfully reported as `UNAVAILABLE` (`"Authoritative discrete exit events not emitted by engine"`). It is never synthesized from agent disappearance.
3. **Data Availability in Comparability Gate:**
   - A metric must not be marked comparable when underlying data does not exist.
   - When a metric is `UNAVAILABLE` or has 0 valid samples in either scenario, `ComparabilityGate` decides `NOT_COMPARABLE` with reason code `DATA_UNAVAILABLE`.
4. **Blocked Metrics (ERSI, Delta Score):**
   - The project documentation contains no mathematical formulation, coordinate system, normalization, or baseline semantics for ERSI or Delta Score.
   - These metrics remain unconditionally **BLOCKED** and are rejected by the comparability gate. No synthetic formulas have been invented.

---

## 5. Scenario Comparison & Aggregation Discipline

1. **Pre-Execution Configuration Validation:**
   `ScenarioDefinition.validate_configuration()` validates room boundaries ($W>0, H>0$), steps ($\ge 1$), timestep ($>0$), exit definitions, and obstacle bounds prior to execution. Malformed configurations are rejected early, recording `RunStatus.FAILED` in the audit log without crashing the runner.
2. **Deterministic Repeatability vs. Stochastic Variability:**
   Because the simulation engine uses deterministic kinematics without an internal PRNG, running identical scenario definitions repeatedly yields zero variance ($\sigma = 0.0$). The system marks these runs with `repetition_sample_type="deterministic_duplicate"` to prevent deterministic duplicates from being misrepresented as independent stochastic Monte Carlo samples.
3. **Traceability to Contributing Run IDs:**
   Every aggregated metric summary preserves `contributing_run_ids`, `attempted_run_count`, `successful_run_count`, `failed_run_count`, `eligible_run_count`, `excluded_run_count`, and exact exclusion reasons.

---

## 6. Test Verification Evidence

```text
Full Test Suite: python -m pytest tests/ -v
Result: 196 passed, 0 failed, 1 warning in 0.53s

Dedicated Phase 06 Suites:
- tests/test_scenario_metadata_phase06.py (5 tests) ........... PASSED
- tests/test_comparability_phase06.py (8 tests) ............... PASSED
- tests/test_aggregation_phase06.py (5 tests) ................. PASSED
- tests/test_scenario_comparison_phase06.py (5 tests) ......... PASSED
- tests/test_pavan_integration_phase06.py (14 tests) .......... PASSED
Total Phase 06 dedicated tests: 37 passed in 0.22s
```

All 196 tests pass across the entire repository with zero regressions across prediction, baseline, evaluation, metrics, heatmaps, live charts, adapter, and scenario comparison subsystems.

