Phase: 08 — Experiment Export, Reporting & Reproducibility (Export Subsystem)
Status: COMPLETE
Files and symbols inspected:
  - src/genevac/experiments/scenario.py (ScenarioDefinition, RunRecord, ComparisonDefinition, AggregateMetricResult, aggregate_per_run_results)
  - src/genevac/experiments/runner.py (ScenarioComparisonRunner, ComparisonStudyResult, MetricDelta)
  - src/genevac/experiments/comparability.py (ComparabilityGate, ComparabilityEvaluation, ComparabilityDecision, ComparabilityResult)
  - src/genevac/experiments/presentation.py (generate_scenario_inventory_table, generate_metric_comparison_table, generate_per_run_detail_table, generate_scientific_interpretation)
  - src/genevac/export/__init__.py (export_to_json, export_to_csv)
  - tests/test_export.py
  - tests/test_scenario_comparison_phase06.py
  - tests/test_pavan_integration_phase06.py
Existing implementation reused:
  - ScenarioDefinition, RunRecord, PerRunMetricResult, AggregateMetricResult, ComparisonStudyResult data structures
  - Presentation table generators (generate_scenario_inventory_table, generate_metric_comparison_table, generate_per_run_detail_table, generate_scientific_interpretation)
  - SHA-256 fingerprinting from ScenarioDefinition
Proposed files to change:
  - src/genevac/experiments/export.py (new module implementing Parts A, B, C, D)
  - src/genevac/experiments/__init__.py (expose export functions)
  - src/genevac/export/__init__.py (re-export Phase 07/08 functions alongside legacy exports)
  - tests/test_experiment_export_phase07.py (new comprehensive export test suite)
  - docs/phase08_export_technical_reference.md (technical documentation)
  - prompt_outputs/08_export.md (completion status report)
Files actually created:
  - src/genevac/experiments/export.py
  - tests/test_experiment_export_phase07.py
  - docs/phase08_export_technical_reference.md
  - prompt_outputs/08_export.md
Files actually modified:
  - src/genevac/experiments/__init__.py
  - src/genevac/export/__init__.py
Commands actually executed:
  - git status --short --branch; git log -n 3 --oneline; git stash list
  - python -m pytest tests/test_export.py -v
  - python -m pytest tests/test_experiment_export_phase07.py -v
  - python -m pytest tests/ -v
  - git status --short --branch; git diff --stat
Tests/experiments actually run:
  - 12 dedicated export tests in tests/test_experiment_export_phase07.py
  - Full regression suite of 208 tests across the entire repository
Actual results:
  - 208 passed, 0 failed, 1 warning (deprecation warning in anyio pytest plugin) in 1.31s
Assumptions and unresolved questions:
  - As inspected in the current Pavan SimulationEngine (genevac.simulation.engine), the engine's kinematics are deterministic and do not consume PRNG seeds; duplicate repetitions are explicitly classified as deterministic duplicates.
  - ERSI and Delta Score remain blocked pending authoritative definition.
  - Exit throughput remains UNAVAILABLE pending discrete exit event emission by the engine.
Risks/limitations:
  - Repetitions without stochastic initial agent positioning produce zero variance.
Approval needed:
  - None for export.

