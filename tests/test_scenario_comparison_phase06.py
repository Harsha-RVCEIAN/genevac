"""Integration test for Phase 06 Scenario Comparison using real simulation execution."""

import math
import pytest

from genevac.experiments import (
    ComparisonDefinition,
    ScenarioDefinition,
    ScenarioComparisonRunner,
    generate_metric_comparison_table,
    generate_per_run_detail_table,
    generate_scenario_inventory_table,
    generate_scientific_interpretation,
)
from genevac.experiments.comparability import ComparabilityDecision
from genevac.simulation import SimulationEngine


class TestScenarioComparisonPhase06:
    def test_end_to_end_scenario_comparison_execution(self):
        # 1. Define real baseline scenario (open corridor)
        baseline = ScenarioDefinition(
            scenario_id="corridor_open",
            name="Open Hallway Corridor",
            agent_count=10,
            engine_config={
                "width": 40.0,
                "height": 15.0,
                "max_steps": 20,
                "timestep": 0.1,
                "exits": [{"id": "exit_1", "position": [40.0, 7.5], "width": 2.0}],
                "obstacles": [],
            },
        )

        # 2. Define real variant scenario (same hallway with a central pillar obstacle)
        variant = ScenarioDefinition(
            scenario_id="corridor_with_pillar",
            name="Hallway with Central Pillar",
            agent_count=10,
            engine_config={
                "width": 40.0,
                "height": 15.0,
                "max_steps": 20,
                "timestep": 0.1,
                "exits": [{"id": "exit_1", "position": [40.0, 7.5], "width": 2.0}],
                "obstacles": [{"id": "pillar_1", "bounds": [18.0, 6.0, 22.0, 9.0]}],
            },
        )

        # 3. Formal comparison design declaration
        comp_def = ComparisonDefinition(
            comparison_id="study_pillar_impact",
            title="Geometric Pillar Impact Analysis",
            research_question="Evaluate the impact of a central pillar on corridor evacuation speed and separation.",
            baseline_scenario_id="corridor_open",
            scenario_ids=("corridor_open", "corridor_with_pillar"),
            independent_variable="presence_of_pillar_obstacle",
            dependent_metrics=("total_elapsed_time", "mean_speed", "area_density", "minimum_separation", "exit_throughput"),
            controlled_factors={"width": 40.0, "height": 15.0, "agent_count": 10, "timestep": 0.1, "max_steps": 20},
        )

        # 4. Execute multi-run comparison study
        study = ScenarioComparisonRunner.execute_comparison(
            comparison=comp_def,
            scenarios=[baseline, variant],
            repetitions=2,
        )

        # 5. Verify run records and execution integrity
        assert len(study.run_records) == 4  # 2 scenarios * 2 repetitions
        for r in study.run_records:
            assert r.run_status.value == "SUCCESS"
            assert r.final_step == 20
            assert math.isclose(r.simulated_duration, 2.0)
            assert len(r.config_fingerprint) == 64  # SHA-256 length

        # 6. Verify comparability evaluations
        # total_elapsed_time, mean_speed, area_density, minimum_separation should be COMPARABLE
        assert study.comparability_results[("corridor_with_pillar", "total_elapsed_time")].decision == ComparabilityDecision.COMPARABLE
        assert study.comparability_results[("corridor_with_pillar", "mean_speed")].decision == ComparabilityDecision.COMPARABLE
        assert study.comparability_results[("corridor_with_pillar", "area_density")].decision == ComparabilityDecision.COMPARABLE

        # 7. Verify aggregates and baseline deltas
        base_density = study.aggregates[("corridor_open", "area_density")].mean
        var_density = study.aggregates[("corridor_with_pillar", "area_density")].mean
        assert base_density is not None and var_density is not None
        assert math.isclose(base_density, 10 / (40.0 * 15.0))
        assert math.isclose(var_density, 10 / (40.0 * 15.0))

        delta_density = study.metric_deltas[("corridor_with_pillar", "area_density")]
        assert math.isclose(delta_density.absolute_delta, 0.0)

        # exit_throughput is UNAVAILABLE because engine doesn't emit discrete exit event log
        var_tp = study.aggregates[("corridor_with_pillar", "exit_throughput")]
        assert var_tp.eligible_run_count == 0
        assert var_tp.unavailable_count == 2
        assert var_tp.mean is None

        # 8. Verify formatted output tables render without error
        inventory_tbl = generate_scenario_inventory_table(study)
        assert "| `corridor_open` |" in inventory_tbl
        assert "| `corridor_with_pillar` |" in inventory_tbl

        comparison_tbl = generate_metric_comparison_table(study)
        assert "| `total_elapsed_time` |" in comparison_tbl
        assert "**COMPARABLE**" in comparison_tbl

        per_run_tbl = generate_per_run_detail_table(study)
        assert "| `corridor_open_run_01` |" in per_run_tbl
        assert "| `corridor_with_pillar_run_02` |" in per_run_tbl

        interpretation = generate_scientific_interpretation(study)
        assert "Scientific Interpretation & Study Limitations" in interpretation
        assert "No Composite Ranking" in interpretation
