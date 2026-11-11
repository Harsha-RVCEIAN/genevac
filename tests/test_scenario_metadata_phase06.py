"""Tests for Phase 06 Scenario Metadata and Configuration Fingerprinting."""

import pytest

from genevac.experiments.scenario import (
    RunRecord,
    RunStatus,
    ScenarioDefinition,
    TerminationReason,
)


class TestScenarioMetadataPhase06:
    def test_scenario_definition_preserves_identity_and_provenance(self):
        sc = ScenarioDefinition(
            scenario_id="sc_baseline_01",
            name="Baseline Open Corridor",
            version="1.2.0",
            description="Testing standard evacuation dynamics.",
            environment_id="hallway_main",
            environment_version="2.0.1",
            environment_provenance="maps/building_main.json",
            agent_count=50,
            initial_conditions_ref="fixtures/initial_cohort_50.json",
            engine_config={"width": 60.0, "height": 15.0, "timestep": 0.1},
            seed=101,
            stopping_condition="max_steps_or_empty",
            max_duration=100.0,
        )

        assert sc.scenario_id == "sc_baseline_01"
        assert sc.name == "Baseline Open Corridor"
        assert sc.version == "1.2.0"
        assert sc.environment_id == "hallway_main"
        assert sc.environment_provenance == "maps/building_main.json"
        assert sc.agent_count == 50
        assert sc.seed == 101
        assert sc.stopping_condition == "max_steps_or_empty"

    def test_scenario_definition_validation(self):
        with pytest.raises(ValueError, match="scenario_id must be a non-empty string"):
            ScenarioDefinition(scenario_id="", name="Valid Name")

        with pytest.raises(ValueError, match="name must be a non-empty string"):
            ScenarioDefinition(scenario_id="valid_id", name="")

        with pytest.raises(ValueError, match="agent_count cannot be negative"):
            ScenarioDefinition(scenario_id="valid_id", name="Valid Name", agent_count=-5)

    def test_configuration_fingerprint_determinism_and_sensitivity(self):
        sc1 = ScenarioDefinition(
            scenario_id="sc_01",
            name="Corridor Test",
            agent_count=25,
            engine_config={"width": 50.0, "height": 20.0, "timestep": 0.1},
            seed=42,
        )
        sc2 = ScenarioDefinition(
            scenario_id="sc_01",
            name="Corridor Test",
            agent_count=25,
            engine_config={"timestep": 0.1, "width": 50.0, "height": 20.0},  # reordered keys
            seed=42,
        )
        sc_diff = ScenarioDefinition(
            scenario_id="sc_01",
            name="Corridor Test",
            agent_count=25,
            engine_config={"width": 50.0, "height": 20.0, "timestep": 0.2},  # different timestep
            seed=42,
        )

        fp1 = sc1.compute_fingerprint()
        fp2 = sc2.compute_fingerprint()
        fp_diff = sc_diff.compute_fingerprint()

        assert fp1 == fp2, "Fingerprint must be invariant to dictionary key insertion order."
        assert fp1 != fp_diff, "Fingerprint must change when a simulation parameter changes."

    def test_run_record_preservation(self):
        record = RunRecord(
            run_id="run_001",
            scenario_id="sc_01",
            scenario_version="1.0.0",
            started_at="2026-11-09T20:00:00Z",
            ended_at="2026-11-09T20:00:10Z",
            seed=42,
            initial_agent_count=20,
            final_step=100,
            simulated_duration=10.0,
            termination_reason=TerminationReason.COMPLETED,
            run_status=RunStatus.SUCCESS,
            config_fingerprint="abc123hash",
        )

        assert record.run_id == "run_001"
        assert record.termination_reason == TerminationReason.COMPLETED
        assert record.run_status == RunStatus.SUCCESS
        assert record.simulated_duration == 10.0

    def test_run_record_validation(self):
        with pytest.raises(ValueError, match="run_id must be a non-empty string"):
            RunRecord(
                run_id="",
                scenario_id="sc_01",
                scenario_version="1.0.0",
                started_at="",
                ended_at="",
                seed=42,
                initial_agent_count=10,
                final_step=10,
                simulated_duration=1.0,
                termination_reason=TerminationReason.COMPLETED,
                run_status=RunStatus.SUCCESS,
            )

        with pytest.raises(ValueError, match="simulated_duration must be a finite non-negative number"):
            RunRecord(
                run_id="run_01",
                scenario_id="sc_01",
                scenario_version="1.0.0",
                started_at="",
                ended_at="",
                seed=42,
                initial_agent_count=10,
                final_step=10,
                simulated_duration=-5.0,
                termination_reason=TerminationReason.COMPLETED,
                run_status=RunStatus.SUCCESS,
            )
