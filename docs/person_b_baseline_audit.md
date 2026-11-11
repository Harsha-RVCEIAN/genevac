# GenEvac-Sentinel: Person B Baseline Audit & Deliverables Traceability

**Document:** `docs/person_b_baseline_audit.md`  
**Status:** Source-Verified Audit (Phase 1)  
**Author:** Harsha (Person B / Module B Part 2)  
**Upstream Integration:** Pavan (Module B Part 1)  

---

## 1. Deliverables Traceability Matrix

| Person B Deliverable | Source Implementation | Existing Tests | Status | Action Needed |
|---|---|---|---|---|
| **1. Metric Definitions & Design** | `docs/prediction.md`, `src/genevac/metrics/__init__.py` | `tests/test_metrics.py`, `tests/test_evaluation_part4.py` | **PARTIAL** | Complete catalog (clearance, flow rate, individual delay) in Phase 2. |
| **2. Metric Implementation** | `src/genevac/metrics/__init__.py` | `tests/test_metrics.py` | **PARTIAL** | Implement remaining approved formulas in Phase 2. |
| **3. Heatmap Calculations** | `src/genevac/visualization/__init__.py` | `tests/test_heatmaps.py` | **PARTIAL** | Build numerical grid engine (bounds, cell size, binning, normalization) in Phase 4. |
| **4. Live Analytics Ingestion** | `src/genevac/prediction/adapter.py` | `tests/test_simulation_adapter.py` | **EXISTS / VERIFIED** | Non-invasive, cadence-filtered, zero-mutation. Ready for Phase 3/5. |
| **5. Charts & Data Export** | `src/genevac/export/__init__.py`, `evaluation.py` | `tests/test_export.py`, `tests/test_evaluation_part4.py` | **EXISTS / VERIFIED** | Enhance with run metadata standards in Phase 5/8. |
| **6. Scenario Comparisons** | `src/genevac/experiments/__init__.py` | `tests/test_experiments.py` | **PARTIAL** | Build multi-run aggregation and variance analysis in Phase 6. |
| **7. Baseline Trajectory Prediction**| `src/genevac/prediction/baseline.py` | `tests/test_baseline.py`, `tests/test_regression_part2_part3.py` | **EXISTS / VERIFIED** | Timestamp-aware Constant Velocity baseline fully tested. |
| **8. MID Investigation** | `docs/mid_feasibility.md`, `docs/ai/mid_reference_recovery_report.md` | `research/evidence/mid_reference/test_stages.py` | **UNRESOLVED / BLOCKED** | Gated; definition documented (Motion Indeterminacy Diffusion); code quarantined. |
| **9. Experiments & Evaluation** | `src/genevac/prediction/evaluation.py` | `tests/test_evaluation_part4.py` | **EXISTS / VERIFIED** | Evaluator computes ADE/FDE, macro vs. point-weighted statistics cleanly. |

---

## 2. Risk Register

| Risk ID | Description | Severity | Impact | Mitigation Strategy | Owner | Status |
|---|---|---|---|---|---|---|
| **R-01** | Absence of engine-level `run_id` | Medium | Risk of mixing trajectory histories across runs if simulator resets | Adapter checks instance `id(engine)` and step rollback to reset history buffer cleanly | Harsha | **Mitigated** |
| **R-02** | Mutation of simulation engine state | High | Ingestion mutating pedestrian coordinates or step progression | Read-only adapter copies scalar primitives into immutable frozen dataclasses | Harsha | **Mitigated** |
| **R-03** | MID execution block | High | Research repo lacks weights, hardcodes CUDA, has script bugs | Quarantined under `research/MID/`; Phase 7 focuses on baseline; MID remains gated | Harsha | **Gated** |
| **R-04** | Cumulative clock drift | Medium | Repeated floating-point `timestamp += dt` accumulation | Derived strictly via `step_count * dt` with 10 Hz / 2.5 Hz synchronization | Harsha / Pavan | **Mitigated** |
| **R-05** | Metric / Heatmap UI coupling | Medium | Calculation logic entangled with matplotlib or UI rendering | Strict separation: computation modules output numerical matrices/dataclasses; visualizers consume them | Harsha | **Mitigated** |

