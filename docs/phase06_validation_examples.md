# Phase 06 — Scenario Comparison Validation Examples & Reproduction Guide

**Document Version:** 1.0.0  
**Phase:** 06 — Scenario Comparison  
**Owner:** Module B Part 2 (Person B / Harsha)  

---

## 1. Minimal Demonstrable Comparison Study

### 1.1 Research Question
> *"What is the impact of introducing a central geometric obstacle on pedestrian evacuation clearance time, mean walking speed, and minimum spatial separation in a 50m × 20m hallway?"*

### 1.2 Comparison Definition
* **Baseline Scenario (`scenario_hallway_open`):**
  - Dimensions: $50.0\text{ m} \times 20.0\text{ m}$.
  - Exits: 1 exit at $(50.0, 10.0)$, width $2.5\text{ m}$.
  - Obstacles: None.
  - Population: 20 agents uniformly spaced along $x \in [5.0, 23.0], y \in [5.0, 7.0]$, velocity $(1.0, 0.0)\text{ m/s}$.
  - Max steps: 100 ($10.0\text{ s}$ duration at $\Delta t = 0.1\text{ s}$).
* **Variant Scenario (`scenario_hallway_obstacle`):**
  - Identical dimensions, exits, population, and initial velocities.
  - Exactly one factor changed: addition of central pillar obstacle at bounds `(24.0, 8.0, 26.0, 12.0)`.
* **Controls:**
  - Geometry: $50\text{ m} \times 20\text{ m}$.
  - Agent count: $N = 20$.
  - Kinematic velocity: $(1.0, 0.0)\text{ m/s}$.
  - Step size: $\Delta t = 0.1\text{ s}$, max steps = 100.
* **Repetitions:** 3 runs per scenario.

---

## 2. Programmatic Execution Script
```python
from genevac.experiments import (
    ComparisonDefinition,
    ScenarioDefinition,
    ScenarioComparisonRunner,
    generate_scenario_inventory_table,
    generate_metric_comparison_table,
    generate_scientific_interpretation,
)

# 1. Define Scenarios
baseline = ScenarioDefinition(
    scenario_id="scenario_hallway_open",
    name="Open Hallway Baseline",
    agent_count=20,
    engine_config={
        "width": 50.0,
        "height": 20.0,
        "max_steps": 100,
        "timestep": 0.1,
        "exits": [{"id": "exit_east", "position": [50.0, 10.0], "width": 2.5}],
        "obstacles": [],
    },
)

variant = ScenarioDefinition(
    scenario_id="scenario_hallway_obstacle",
    name="Obstacle Hallway Variant",
    agent_count=20,
    engine_config={
        "width": 50.0,
        "height": 20.0,
        "max_steps": 100,
        "timestep": 0.1,
        "exits": [{"id": "exit_east", "position": [50.0, 10.0], "width": 2.5}],
        "obstacles": [{"id": "pillar_central", "bounds": [24.0, 8.0, 26.0, 12.0]}],
    },
)

# 2. Define Comparison Study
comp_def = ComparisonDefinition(
    comparison_id="comp_obstacle_impact",
    title="Central Obstacle Impact Study",
    research_question="Measure impact of central obstacle on clearance time and spatial separation.",
    baseline_scenario_id="scenario_hallway_open",
    scenario_ids=("scenario_hallway_open", "scenario_hallway_obstacle"),
    independent_variable="presence_of_central_pillar",
    dependent_metrics=("total_elapsed_time", "mean_speed", "area_density", "minimum_separation", "exit_throughput"),
    controlled_factors={"width": 50.0, "height": 20.0, "agent_count": 20, "timestep": 0.1},
)

# 3. Run Comparison Study
study_result = ScenarioComparisonRunner.execute_comparison(
    comparison=comp_def,
    scenarios=[baseline, variant],
    repetitions=3,
)

# 4. Format Tables
print(generate_scenario_inventory_table(study_result))
print(generate_metric_comparison_table(study_result))
print(generate_scientific_interpretation(study_result))
```

---

## 3. Reproduction Instructions
1. Run test suite:
   ```powershell
   python -m pytest tests/test_scenario_comparison_phase06.py -v
   ```
2. Verify that all 3 runs per scenario execute deterministically, per-run records are recorded, and metrics are compared against baseline.
