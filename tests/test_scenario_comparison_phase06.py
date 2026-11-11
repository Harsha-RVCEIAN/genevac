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
        assert study.comparability_results[("corridor_with_pillar", "exit_throughput")].decision == ComparabilityDecision.NOT_COMPARABLE
        assert "DATA_UNAVAILABLE" in study.comparability_results[("corridor_with_pillar", "exit_throughput")].reason_codes

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
        assert study.repetition_sample_type == "deterministic_duplicate"

    def test_invalid_scenario_configuration_fails_early_and_records_audit(self):
        bad_scenario = ScenarioDefinition(
            scenario_id="sc_invalid_geo",
            name="Invalid Geometry Scenario",
            agent_count=10,
            engine_config={
                "width": -50.0,  # Invalid negative width!
                "height": 20.0,
                "max_steps": 100,
            },
        )
        base = ScenarioDefinition(
            scenario_id="sc_base",
            name="Base",
            agent_count=10,
            engine_config={"width": 50.0, "height": 20.0, "max_steps": 100},
        )
        comp = ComparisonDefinition(
            comparison_id="comp_test_fail",
            title="Fail Test",
            research_question="Verify early rejection of malformed config.",
            baseline_scenario_id="sc_base",
            scenario_ids=("sc_base", "sc_invalid_geo"),
            independent_variable="width",
            dependent_metrics=("total_elapsed_time",),
        )

        study = ScenarioComparisonRunner.execute_comparison(comp, [base, bad_scenario], repetitions=1)
        bad_run = [r for r in study.run_records if r.scenario_id == "sc_invalid_geo"][0]
        assert bad_run.run_status.value == "FAILED"
        assert bad_run.final_step == 0
        assert "width must be a finite positive number" in bad_run.failure_reason

    def test_clearance_time_unavailable_when_agents_remain_active(self):
        # Scenario where agents cannot reach exit within max_steps (10 steps)
        sc = ScenarioDefinition(
            scenario_id="sc_slow",
            name="Slow Incomplete Evacuation",
            agent_count=5,
            engine_config={
                "width": 100.0,
                "height": 20.0,
                "max_steps": 5,
                "timestep": 0.1,
                "exits": [{"id": "exit_far", "position": [100.0, 10.0], "width": 2.0}],
            },
        )
        base = ScenarioDefinition(
            scenario_id="sc_base2",
            name="Base 2",
            agent_count=5,
            engine_config={"width": 100.0, "height": 20.0, "max_steps": 5, "timestep": 0.1},
        )
        comp = ComparisonDefinition(
            comparison_id="comp_clearance_test",
            title="Clearance Availability Test",
            research_question="Verify clearance_time is UNAVAILABLE when evacuation is incomplete.",
            baseline_scenario_id="sc_base2",
            scenario_ids=("sc_base2", "sc_slow"),
            independent_variable="none",
            dependent_metrics=("clearance_time", "total_elapsed_time"),
        )

        study = ScenarioComparisonRunner.execute_comparison(comp, [base, sc], repetitions=1)

        # total_elapsed_time should be VALID (0.5s)
        elapsed_agg = study.aggregates[("sc_slow", "total_elapsed_time")]
        assert elapsed_agg.eligible_run_count == 1
        assert math.isclose(elapsed_agg.mean, 0.5)

        # clearance_time must be UNAVAILABLE because agents are still in corridor
        clearance_agg = study.aggregates[("sc_slow", "clearance_time")]
        assert clearance_agg.eligible_run_count == 0
        assert clearance_agg.unavailable_count == 1
        assert clearance_agg.mean is None

    def test_scenario_validation_boundary_conditions(self):
        # 1. Negative / zero room dimensions
        sc_neg_h = ScenarioDefinition(
            scenario_id="sc_neg_h",
            name="Negative Height",
            agent_count=5,
            engine_config={"width": 50.0, "height": 0.0, "max_steps": 10},
        )
        valid, errors = sc_neg_h.validate_configuration()
        assert valid is False
        assert any("height must be a finite positive number" in e for e in errors)

        # 2. Invalid max_steps and timestep
        sc_bad_steps = ScenarioDefinition(
            scenario_id="sc_bad_steps",
            name="Bad Steps",
            agent_count=5,
            engine_config={"width": 50.0, "height": 20.0, "max_steps": 0, "timestep": -0.1},
        )
        valid, errors = sc_bad_steps.validate_configuration()
        assert valid is False
        assert any("max_steps must be a positive integer" in e for e in errors)
        assert any("timestep must be a finite positive number" in e for e in errors)

        # 3. Out-of-bounds exit
        sc_bad_exit = ScenarioDefinition(
            scenario_id="sc_bad_exit",
            name="Out of bounds exit",
            agent_count=5,
            engine_config={
                "width": 50.0,
                "height": 20.0,
                "exits": [{"id": "ex_oob", "position": [55.0, 10.0]}],
            },
        )
        valid, errors = sc_bad_exit.validate_configuration()
        assert valid is False
        assert any("exit[0] out-of-bounds" in e for e in errors)

        # 4. Out-of-bounds obstacle
        sc_bad_obs = ScenarioDefinition(
            scenario_id="sc_bad_obs",
            name="Out of bounds obstacle",
            agent_count=5,
            engine_config={
                "width": 50.0,
                "height": 20.0,
                "obstacles": [{"id": "obs_oob", "bounds": [40.0, 5.0, 55.0, 15.0]}],
            },
        )
        valid, errors = sc_bad_obs.validate_configuration()
        assert valid is False
        assert any("obstacle[0] out-of-bounds" in e for e in errors)

        # 5. Degenerate obstacle geometry (min_x >= max_x or min_y >= max_y)
        sc_degen_x = ScenarioDefinition(
            scenario_id="sc_degen_x",
            name="Degenerate X",
            agent_count=5,
            engine_config={
                "width": 50.0,
                "height": 20.0,
                "obstacles": [{"id": "obs_degen_x", "bounds": [30.0, 5.0, 20.0, 15.0]}],
            },
        )
        valid, errors = sc_degen_x.validate_configuration()
        assert valid is False
        assert any("invalid degenerate geometry" in e for e in errors)

        sc_degen_y = ScenarioDefinition(
            scenario_id="sc_degen_y",
            name="Degenerate Y",
            agent_count=5,
            engine_config={
                "width": 50.0,
                "height": 20.0,
                "obstacles": [{"id": "obs_degen_y", "bounds": [10.0, 15.0, 20.0, 15.0]}],
            },
        )
        valid, errors = sc_degen_y.validate_configuration()
        assert valid is False
        assert any("invalid degenerate geometry" in e for e in errors)

    def test_missing_values_never_coerced_to_zero(self):
        base = ScenarioDefinition(
            scenario_id="sc_base_na",
            name="Base",
            agent_count=5,
            engine_config={"width": 40.0, "height": 10.0, "max_steps": 5, "timestep": 0.1},
        )
        variant = ScenarioDefinition(
            scenario_id="sc_variant_na",
            name="Variant",
            agent_count=5,
            engine_config={"width": 40.0, "height": 10.0, "max_steps": 5, "timestep": 0.1},
        )
        comp = ComparisonDefinition(
            comparison_id="comp_na_test",
            title="Non-coercion Test",
            research_question="Verify missing values are not silently zeroed.",
            baseline_scenario_id="sc_base_na",
            scenario_ids=("sc_base_na", "sc_variant_na"),
            independent_variable="none",
            dependent_metrics=("exit_throughput", "clearance_time"),
        )
        study = ScenarioComparisonRunner.execute_comparison(comp, [base, variant], repetitions=1)

        # Throughput aggregate
        tp_agg = study.aggregates[("sc_base_na", "exit_throughput")]
        assert tp_agg.mean is None
        assert tp_agg.std_dev is None
        assert tp_agg.min_val is None
        assert tp_agg.max_val is None
        assert tp_agg.unavailable_count == 1
        assert tp_agg.eligible_run_count == 0

        # Rendering check: tables must not display 0.000 for unavailable metrics
        tbl = generate_metric_comparison_table(study)
        assert "0.000" not in tbl or "Total Elapsed" in tbl  # Only valid metrics could be 0, not unavailable
        assert "N/A" in tbl or "UNAVAILABLE" in tbl or "DATA_UNAVAILABLE" in tbl
