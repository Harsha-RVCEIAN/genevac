# Phase 06 — Scenario Comparison Data Schema Specification

**Document Version:** 1.0.0  
**Phase:** 06 — Scenario Comparison  
**Owner:** Module B Part 2 (Person B / Harsha)  

---

## 1. Overview
The comparison layer separates:
1. **Scenario Definition:** The intended simulation setup (parameters, layout, agent population).
2. **Run Record:** The execution metadata (run ID, duration, exit status, config fingerprint).
3. **Per-Run Metric Result:** The raw computed metric for a single run, retaining full audit lineage.
4. **Comparison Definition:** The declared research question, baseline, independent variable, controls, and inclusion rules.
5. **Aggregate Metric Result:** The transparent summary across eligible runs, retaining contributing run IDs and exclusion reasons.

---

## 2. Entity Schemas

### 2.1 `ScenarioDefinition`
```python
@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_id: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    environment_id: str = "default_env"
    environment_version: str = "1.0.0"
    environment_provenance: str = ""
    agent_count: int = 0
    initial_conditions_ref: str = ""
    engine_config: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = 42
    stopping_condition: str = "max_steps_or_empty"
    max_duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```
- **Config Fingerprint:** `compute_fingerprint()` computes a SHA-256 hash of canonicalized JSON fields, ensuring deterministic versioning independent of memory addresses or timestamp variations.

### 2.2 `RunRecord`
```python
@dataclass(frozen=True)
class RunRecord:
    run_id: str
    scenario_id: str
    scenario_version: str
    started_at: str
    ended_at: str
    seed: Optional[int]
    initial_agent_count: int
    final_step: int
    simulated_duration: float
    termination_reason: TerminationReason  # COMPLETED, MAX_STEPS_REACHED, STOPPED_EARLY, ERROR, UNKNOWN
    run_status: RunStatus                  # SUCCESS, PARTIAL, FAILED
    failure_reason: str = ""
    config_fingerprint: str = ""
    source_provenance: Dict[str, str] = field(default_factory=dict)
```

### 2.3 `PerRunMetricResult`
```python
@dataclass(frozen=True)
class PerRunMetricResult:
    run_id: str
    scenario_id: str
    metric_id: str
    display_name: str
    value: Optional[float]
    unit: str
    status: str = "VALID"  # VALID, UNAVAILABLE, NOT_COMPUTED, PARTIAL, INVALID, FAILED
    reason: str = ""
    time_window: Tuple[float, float] = (0.0, 0.0)
    population: str = "all"
    spatial_domain: str = "all"
    metric_definition_version: str = "1.0.0"
    source_step_range: Tuple[int, int] = (0, 0)
    valid_sample_count: int = 0
    missing_sample_count: int = 0
```

### 2.4 `ComparisonDefinition`
```python
@dataclass(frozen=True)
class ComparisonDefinition:
    comparison_id: str
    title: str
    research_question: str
    baseline_scenario_id: str
    scenario_ids: Tuple[str, ...]
    independent_variable: str
    dependent_metrics: Tuple[str, ...]
    controlled_factors: Dict[str, Any] = field(default_factory=dict)
    repetition_policy: Dict[str, Any] = field(default_factory=dict)
    seed_policy: str = "deterministic_constant"
    inclusion_rules: Dict[str, Any] = field(default_factory=dict)
    exclusion_rules: Dict[str, Any] = field(default_factory=dict)
    comparability_policy: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
```

### 2.5 `AggregateMetricResult`
```python
@dataclass(frozen=True)
class AggregateMetricResult:
    scenario_id: str
    metric_id: str
    unit: str
    attempted_run_count: int
    successful_run_count: int
    partial_run_count: int
    failed_run_count: int
    eligible_run_count: int
    excluded_run_count: int
    unavailable_count: int
    mean: Optional[float]
    median: Optional[float]
    std_dev: Optional[float]  # None if eligible_run_count < 2
    min_val: Optional[float]
    max_val: Optional[float]
    aggregation_method: str = "sample_mean_and_stdev"
    inclusion_reasons: Tuple[str, ...] = field(default_factory=tuple)
    exclusion_reasons: Tuple[str, ...] = field(default_factory=tuple)
    contributing_run_ids: Tuple[str, ...] = field(default_factory=tuple)
    is_single_run: bool = False
```
