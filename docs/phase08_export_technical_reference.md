# Phase 08 Technical Reference: Experiment Export, Reporting & Reproducibility

**Document Version:** 1.0.0  
**Phase:** 08 — Experiment Export, Reporting & Reproducibility (formerly referenced as Export Subsystem)  
**Author:** Module B Part 2 (Person B / Harsha)  
**Integration Target:** Module B Part 1 Simulation Engine (Pavan) & Phase 06 Scenario Comparison  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary & Architecture

This module establishes an auditable, scientifically disciplined export and reporting layer on top of the Phase 06 scenario-comparison subsystem. It enables complete export of experimental results into portable, structured formats (CSV and JSON) and generates comprehensive 11-section evaluation reports in Markdown, accompanied by strict reproducibility metadata.

```
       [ Upstream Simulation Engine (Pavan / Module B Part 1) ]
                                 │
                                 ▼
                     [ Phase 06 Runner & Models ]
            (ScenarioDefinition, RunRecord, PerRunMetricResult,
             AggregateMetricResult, ComparabilityResult, MetricDelta)
                                 │
                                 ▼
                 [ Export & Reporting Layer ]
                    (src/genevac/experiments/export.py)
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
  [ CSV Exporters ]       [ JSON Exporter ]      [ Report Generator ]
  - Scenario Inventory    - Full Result Payload  - 11 Structured Sections
  - Per-Run Metrics       - Explicit Null-Safety - Diagnostic Tables
  - Aggregate Comparisons - Schema Version 1.0.0 - Scientific Limitations
                                 │
                                 ▼
                    [ Reproducibility Metadata ]
        - Truthful PRNG Seed Engine Usage (Deterministic Kinematics)
        - Deterministic Duplicate Repetition Classification
        - Deterministic SHA-256 Configuration Fingerprints
```

---

## 2. Part A — CSV Export Specifications

All CSV files are encoded in UTF-8 with standard comma separators, newline line terminators (`\n`), and robust quote-escaping for multi-line or special-character strings.

### 2.1 Scenario Inventory CSV (`export_scenario_inventory_csv`)
Captures all scenario configurations participating in the study.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `experiment_id` | `string` | Unique comparison experiment identifier |
| `scenario_id` | `string` | Unique identifier for scenario definition |
| `scenario_name` | `string` | Human-readable scenario name |
| `scenario_version` | `string` | Semantic scenario version (e.g. `1.0.0`) |
| `scenario_description` | `string` | Qualitative narrative of scenario design intent |
| `role` | `string` | `BASELINE` or `VARIANT` |
| `agent_count` | `integer` | Configured initial pedestrian population |
| `environment_id` | `string` | Layout geometry identifier |
| `room_width` | `float (2 dec)` | Physical width in metres |
| `room_height` | `float (2 dec)` | Physical height in metres |
| `obstacle_count` | `integer` | Count of obstacle bounding boxes |
| `exit_count` | `integer` | Count of defined exit portals |
| `max_steps` | `integer` | Configured simulation step horizon |
| `timestep` | `float (4 dec)` | Physical integration timestep $\Delta t$ (seconds) |
| `seed` | `string` | Configured random seed (if provided) |
| `stopping_condition` | `string` | Termination policy (e.g. `max_steps_or_empty`) |
| `config_fingerprint` | `string` | Canonical SHA-256 configuration hash (64 hex characters) |

### 2.2 Per-Run Metrics CSV (`export_per_run_metrics_csv`)
Emits exactly one row per simulation run and evaluated metric.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `experiment_id` | `string` | Comparison experiment identifier |
| `scenario_id` | `string` | Scenario definition identifier |
| `run_id` | `string` | Unique execution run identifier (e.g. `sc_base_run_01`) |
| `run_status` | `string` | `SUCCESS`, `PARTIAL`, or `FAILED` |
| `metric_name` | `string` | Canonical metric symbol (e.g. `mean_speed`) |
| `metric_value` | `float / empty` | Measured value (`.6f`) or empty string if unavailable (never coerced to `0.0`) |
| `metric_unit` | `string` | Measurement unit (e.g. `m/s`, `seconds`, `ped/m²`) |
| `availability_status` | `string` | `VALID` or `UNAVAILABLE` |
| `unavailable_reason` | `string` | Diagnostic reason if unavailable or failed |
| `repetition_sample_type` | `string` | `deterministic_duplicate` or `single_or_stochastic` |

### 2.3 Aggregate Comparison CSV (`export_aggregate_comparison_csv`)
Provides cross-scenario summary statistics per metric.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `experiment_id` | `string` | Comparison experiment identifier |
| `scenario_id` | `string` | Scenario identifier |
| `metric_name` | `string` | Metric identifier |
| `mean` | `float / empty` | Sample mean across eligible runs |
| `standard_deviation` | `float / empty` | Sample standard deviation; `None` (empty) when fewer than 2 eligible runs, `0.0` when all eligible values are identical (deterministic duplicates) |
| `minimum` | `float / empty` | Minimum observation |
| `maximum` | `float / empty` | Maximum observation |
| `sample_count` | `integer` | Eligible contributing run count |
| `comparability_decision` | `string` | `BASELINE`, `COMPARABLE`, or `NOT_COMPARABLE` |
| `availability_status` | `string` | `VALID` or `UNAVAILABLE` |
| `unavailable_reason` | `string` | Exclusion explanations if data unavailable |

---

## 3. Part B — JSON Export Specification

`export_experiment_json` exports the entire `ComparisonStudyResult` structure as portable, valid JSON conforming to `schema_version: "1.0.0"`.

### Null-Safety and Type Handling
- **Missing / Non-finite Values:** Any missing, unavailable, or non-finite metric value serializes explicitly to `null`, ensuring spreadsheet and JSON parsers do not treat missing measurements as zero.
- **Pure Primitives:** All enums (`RunStatus`, `TerminationReason`, `ComparabilityDecision`) serialize to strings; tuples serialize to lists; dataclasses serialize to nested JSON objects. No Python-specific pickle or custom class tokens are emitted.

---

## 4. Part C — Reproducibility Metadata Specification

`extract_reproducibility_metadata` generates an auditable record of simulation execution parameters:

1. **Truthful PRNG Seed Disclosure:**
   - `seed_used_by_engine`: Explicitly set to `false`.
   - `seed_usage_note`: Documents that, as inspected in the current Pavan `SimulationEngine` (`genevac.simulation.engine`), the engine's kinematics are purely deterministic and do not consume PRNG seeds during physics stepping. Configured seeds are recorded strictly as scenario design intent.
2. **Deterministic Duplicate Classification:**
   - Multi-run repetitions under identical scenario definitions yield zero standard deviation ($\sigma = 0.0$).
   - Repetitions are explicitly tagged as `deterministic_duplicate` in both CSV and JSON to prevent them from being mistaken for independent stochastic Monte Carlo trials.
3. **Deterministic Configuration Fingerprinting:**
   - Every scenario computes an invariant SHA-256 fingerprint over its sorted canonical JSON dictionary.

---

## 5. Part D — Experiment Report Generation

`generate_experiment_report` produces a complete, standalone Markdown evaluation report structured into 11 sections:

1. **Experiment Overview:** Research question, independent variable, dependent metrics, and controlled factors.
2. **Scenario Inventory:** Tabular summary of baseline and variant configurations (reusing `generate_scenario_inventory_table`).
3. **Execution Summary:** Counts of attempted, successful, partial, and failed runs, with per-run execution detail table (reusing `generate_per_run_detail_table`).
4. **Per-Run Results:** Tabular breakdown of per-run metric observations and sample counts.
5. **Aggregate Metric Comparisons:** Cross-scenario comparative table with baseline deltas, relative percentages, and comparability statuses (reusing `generate_metric_comparison_table`).
6. **Comparability Decisions:** Audit table of comparability gates, reason codes, and caveats.
7. **Scientific Interpretation:** Objective narrative of observed differences and confounders (reusing `generate_scientific_interpretation`).
8. **Unavailable Metrics and Reasons:** Explicit audit of metrics that could not be computed (e.g. exit throughput).
9. **Failed Runs and Reasons:** Detailed error messages and step counts for any failed or partial runs.
10. **Reproducibility Metadata:** Seed usage disclosures, configuration fingerprints, and sample classifications.
11. **Limitations & Boundaries:** Statement of simulation kinematic assumptions, zero inferred evacuations, and exclusion of undefined composite scoring.

---

## 6. Metric and Simulation Boundaries Preserved

- **ERSI & Delta Score:** Remain strictly **BLOCKED** and are rejected by comparability audits (`NOT_COMPARABLE`).
- **Exit Throughput:** Truthfully reported as `UNAVAILABLE` (`DATA_UNAVAILABLE`) because Pavan's engine does not emit discrete exit events.
- **Clearance Time:** Truthfully reported as `UNAVAILABLE` when active pedestrians remain in the environment at run termination; never conflated with `total_elapsed_time`.
- **Zero Value Coercion:** Missing or unavailable metrics serialize to `""` in CSV and `null` in JSON; never coerced to `0.0`.
- **Zero Inferred Evacuations:** Disappearing pedestrians become `not_observed`, never `evacuated`.

---

## 7. Test Verification Evidence

```text
Full Test Suite: python -m pytest tests/ -v
Result: 208 passed, 0 failed, 1 warning in 1.31s

Phase 07 Dedicated Suite:
- tests/test_experiment_export_phase07.py (12 tests) ........... PASSED

Phase 06 Scenario Comparison Suites:
- tests/test_scenario_metadata_phase06.py (5 tests) ........... PASSED
- tests/test_comparability_phase06.py (8 tests) ............... PASSED
- tests/test_aggregation_phase06.py (5 tests) ................. PASSED
- tests/test_scenario_comparison_phase06.py (5 tests) ......... PASSED
- tests/test_pavan_integration_phase06.py (14 tests) .......... PASSED

Regression Baseline:
All 196 prior tests across Phases 01-06 continue to pass with zero regressions.
```

