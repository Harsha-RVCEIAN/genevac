# Phase 03 Canonical Data Flow & Architecture

**Project:** GenEvac-Sentinel  
**Document:** `docs/phase03_data_flow.md`  
**Author:** Harsha (Person B / Module B Part 2)  
**Date:** 2026-11-09  

---

## 1. End-to-End Canonical Ingestion Data Flow

```mermaid
flowchart TD
    subgraph Part1_Engine ["Pavan's Simulation Engine (Module B Part 1)"]
        Eng["SimulationEngine (step, registry, clock)"]
        Snap["SimulationObservationSnapshot (step_count, timestamp, dt, pedestrians)"]
        Eng -->|emits periodic snapshot (0.4s)| Snap
    end

    subgraph Adapter_Layer ["SimulationTrajectoryAdapter (src/genevac/prediction/adapter.py)"]
        Ingest["adapt_snapshot(snapshot) / update_from_simulation(engine)"]
        Dedupe{"Deduplication & Cadence Check"}
        CohortPartition["Cohort Separation (ACTIVE, BLOCKED, EVACUATED, INACTIVE)"]
        Buffer["TrajectoryHistoryBuffer (FIFO deque per agent)"]

        Snap --> Ingest
        Ingest --> Dedupe
        Dedupe -->|new valid cadence tick| CohortPartition
        CohortPartition -->|simulated observation| Buffer
    end

    subgraph Canonical_Outputs ["Canonical Analytics Data Layer"]
        CanonicalSnap["TrajectorySnapshot (MappingProxyType, immutable observations)"]
        Report["SnapshotAdaptationReport (accepted, rejected, duplicates)"]
        Buffer --> CanonicalSnap
        Ingest --> Report
    end

    subgraph Phase02_Consumers ["Phase 02 Metric & Validation Consumers"]
        Pred["ConstantVelocityPredictor (ADE/FDE evaluation)"]
        Crowd["Crowd Metrics (min separation, area density)"]
        Evac["Evacuation Metrics (throughput, clearance)"]
        Route["Route Validation (RC01 - RC13 checks)"]

        CanonicalSnap --> Pred
        CanonicalSnap --> Crowd
        CanonicalSnap --> Evac
        CanonicalSnap --> Route
    end
```

---

## 2. Ingestion Invariants
1. **Read-Only**: The adapter never calls mutating methods (`engine.step()`, `engine.run()`, etc.).
2. **Single History Buffer**: Canonical observations are persisted in `TrajectoryHistoryBuffer`; no secondary store is created.
3. **Immutability Guarantee**: `TrajectorySnapshot` holds observations wrapped in `MappingProxyType`, guaranteeing zero post-construction mutation.

