"""Comprehensive unit and integration test suite for Phase 07 Experiment Export & Reporting."""

import csv
import json
import math
from pathlib import Path
import tempfile
import pytest

from genevac.experiments import (
    ComparisonDefinition,
    ComparisonStudyResult,
    ScenarioComparisonRunner,
    ScenarioDefinition,
)
from genevac.experiments.export import (
    SCHEMA_VERSION,
    SOFTWARE_VERSION,
    export_aggregate_comparison_csv,
    export_all_experiment_artifacts,
    export_experiment_json,
    export_per_run_metrics_csv,
    export_scenario_inventory_csv,
    extract_reproducibility_metadata,
    generate_experiment_report,
)
from genevac.experiments.scenario import RunStatus


@pytest.fixture
def executed_study() -> ComparisonStudyResult:
    """Fixture producing a multi-scenario, multi-run executed study with valid and unavailable metrics."""
    baseline = ScenarioDefinition(
        scenario_id="sc_base_hall",
        name="Baseline Corridor",
        version="1.0.0",
        description="Standard open hallway without obstacles",
        agent_count=6,
        engine_config={
            "width": 30.0,
            "height": 10.0,
            "max_steps": 15,
            "timestep": 0.1,
            "exits": [{"id": "ex_1", "position": [30.0, 5.0], "width": 2.0}],
            "obstacles": [],
        },
        seed=100,
    )
    variant = ScenarioDefinition(
        scenario_id="sc_var_pillar",
        name="Variant Corridor With Pillar",
        version="1.1.0",
        description="Hallway with central pillar obstacle, causing detour",
        agent_count=6,
        engine_config={
            "width": 30.0,
            "height": 10.0,
            "max_steps": 15,
            "timestep": 0.1,
            "exits": [{"id": "ex_1", "position": [30.0, 5.0], "width": 2.0}],
            "obstacles": [{"id": "col_1", "bounds": [14.0, 4.0, 16.0, 6.0]}],
        },
        seed=101,
    )
    comp = ComparisonDefinition(
        comparison_id="study_pillar_eval",
        title="Pillar Bottleneck Evaluation",
        research_question="Does a central pillar reduce mean evacuation speed and increase density?",
        baseline_scenario_id="sc_base_hall",
        scenario_ids=("sc_base_hall", "sc_var_pillar"),
        independent_variable="pillar_presence",
        dependent_metrics=("total_elapsed_time", "mean_speed", "area_density", "exit_throughput", "clearance_time"),
        controlled_factors={"width": 30.0, "height": 10.0, "agent_count": 6, "timestep": 0.1},
    )

    return ScenarioComparisonRunner.execute_comparison(
        comparison=comp,
        scenarios=[baseline, variant],
        repetitions=2,
    )


@pytest.fixture
def executed_study_with_failure() -> ComparisonStudyResult:
    """Fixture producing a study containing an early-failing malformed scenario."""
    base = ScenarioDefinition(
        scenario_id="sc_good",
        name="Valid Scenario",
        agent_count=5,
        engine_config={"width": 40.0, "height": 15.0, "max_steps": 10, "timestep": 0.1},
    )
    bad = ScenarioDefinition(
        scenario_id="sc_broken_geo",
        name="Broken Geometry Scenario",
        agent_count=5,
        engine_config={"width": -20.0, "height": 15.0, "max_steps": 10, "timestep": 0.1},  # negative width
    )
    comp = ComparisonDefinition(
        comparison_id="study_with_failure",
        title="Failure Audit Study",
        research_question="Verify export handling of failed runs.",
        baseline_scenario_id="sc_good",
        scenario_ids=("sc_good", "sc_broken_geo"),
        independent_variable="validity",
        dependent_metrics=("total_elapsed_time", "mean_speed"),
    )
    return ScenarioComparisonRunner.execute_comparison(comp, [base, bad], repetitions=1)


class TestExperimentExportCSV:
    def test_csv_export_scenario_inventory(self, executed_study: ComparisonStudyResult):
        csv_text = export_scenario_inventory_csv(executed_study)
        reader = list(csv.DictReader(csv_text.splitlines()))

        assert len(reader) == 2
        s_map = {row["scenario_id"]: row for row in reader}

        # Check headers
        expected_headers = [
            "experiment_id", "scenario_id", "scenario_name", "scenario_version",
            "scenario_description", "role", "agent_count", "environment_id",
            "room_width", "room_height", "obstacle_count", "exit_count",
            "max_steps", "timestep", "seed", "stopping_condition", "config_fingerprint",
        ]
        assert list(reader[0].keys()) == expected_headers

        # Baseline checks
        base_row = s_map["sc_base_hall"]
        assert base_row["experiment_id"] == "study_pillar_eval"
        assert base_row["role"] == "BASELINE"
        assert base_row["scenario_name"] == "Baseline Corridor"
        assert base_row["agent_count"] == "6"
        assert base_row["room_width"] == "30.00"
        assert base_row["obstacle_count"] == "0"
        assert base_row["exit_count"] == "1"
        assert base_row["seed"] == "100"
        assert len(base_row["config_fingerprint"]) == 64

        # Variant checks
        var_row = s_map["sc_var_pillar"]
        assert var_row["role"] == "VARIANT"
        assert var_row["obstacle_count"] == "1"
        assert var_row["seed"] == "101"

    def test_csv_export_per_run_metrics(self, executed_study: ComparisonStudyResult):
        csv_text = export_per_run_metrics_csv(executed_study)
        reader = list(csv.DictReader(csv_text.splitlines()))

        # Rows match total per_run_results generated
        assert len(reader) == len(executed_study.per_run_results)
        assert len(reader) == 24  # 4 runs * 6 metrics

        expected_headers = [
            "experiment_id", "scenario_id", "run_id", "run_status",
            "metric_name", "metric_value", "metric_unit",
            "availability_status", "unavailable_reason", "repetition_sample_type",
        ]
        assert list(reader[0].keys()) == expected_headers

        # Verify valid metric representation (e.g. total_elapsed_time)
        elapsed_rows = [r for r in reader if r["metric_name"] == "total_elapsed_time"]
        assert len(elapsed_rows) == 4
        for r in elapsed_rows:
            assert r["run_status"] == "SUCCESS"
            assert r["availability_status"] == "VALID"
            assert r["metric_value"] != ""
            assert math.isclose(float(r["metric_value"]), 1.5, rel_tol=1e-3)
            assert r["metric_unit"] == "seconds"
            assert r["unavailable_reason"] == ""
            assert r["repetition_sample_type"] == "deterministic_duplicate"

        # Verify unavailable metric representation (exit_throughput: data unavailable)
        tp_rows = [r for r in reader if r["metric_name"] == "exit_throughput"]
        assert len(tp_rows) == 4
        for r in tp_rows:
            assert r["availability_status"] == "UNAVAILABLE"
            assert r["metric_value"] == "", "Unavailable values must not be coerced to 0.0"
            assert "Authoritative discrete exit events not emitted" in r["unavailable_reason"]

    def test_csv_export_aggregate_comparison(self, executed_study: ComparisonStudyResult):
        csv_text = export_aggregate_comparison_csv(executed_study)
        reader = list(csv.DictReader(csv_text.splitlines()))

        # 2 scenarios * 5 metrics = 10 rows
        assert len(reader) == 10

        expected_headers = [
            "experiment_id", "scenario_id", "metric_name", "mean",
            "standard_deviation", "minimum", "maximum", "sample_count",
            "comparability_decision", "availability_status", "unavailable_reason",
        ]
        assert list(reader[0].keys()) == expected_headers

        # Check baseline row
        base_elapsed = [r for r in reader if r["scenario_id"] == "sc_base_hall" and r["metric_name"] == "total_elapsed_time"][0]
        assert base_elapsed["comparability_decision"] == "BASELINE"
        assert base_elapsed["availability_status"] == "VALID"
        assert math.isclose(float(base_elapsed["mean"]), 1.5, rel_tol=1e-3)
        assert math.isclose(float(base_elapsed["standard_deviation"]), 0.0, abs_tol=1e-5)
        assert base_elapsed["sample_count"] == "2"

        # Check variant row
        var_elapsed = [r for r in reader if r["scenario_id"] == "sc_var_pillar" and r["metric_name"] == "total_elapsed_time"][0]
        assert var_elapsed["comparability_decision"] == "COMPARABLE"
        assert var_elapsed["availability_status"] == "VALID"

        # Check unavailable metric in aggregates (exit_throughput)
        var_tp = [r for r in reader if r["scenario_id"] == "sc_var_pillar" and r["metric_name"] == "exit_throughput"][0]
        assert var_tp["availability_status"] == "UNAVAILABLE"
        assert var_tp["mean"] == ""
        assert var_tp["standard_deviation"] == ""
        assert var_tp["sample_count"] == "0"
        assert var_tp["comparability_decision"] == "NOT_COMPARABLE"
        assert var_tp["unavailable_reason"] != ""

    def test_csv_export_failed_run_representation(self, executed_study_with_failure: ComparisonStudyResult):
        csv_text = export_per_run_metrics_csv(executed_study_with_failure)
        reader = list(csv.DictReader(csv_text.splitlines()))

        failed_rows = [r for r in reader if r["scenario_id"] == "sc_broken_geo"]
        assert len(failed_rows) > 0
        for r in failed_rows:
            assert r["run_status"] == "FAILED"
            assert r["availability_status"] == "UNAVAILABLE"
            assert r["metric_value"] == ""
            assert "width must be a finite positive number" in r["unavailable_reason"]

    def test_csv_escaping_with_commas_and_quotes(self):
        sc = ScenarioDefinition(
            scenario_id="sc_special",
            name='Corridor with "Special, Quote" & Comma',
            description="Testing CSV escaping: line1\nline2, with commas",
            agent_count=5,
            engine_config={"width": 40.0, "height": 10.0, "max_steps": 5, "timestep": 0.1},
        )
        base = ScenarioDefinition(
            scenario_id="sc_base",
            name="Base",
            agent_count=5,
            engine_config={"width": 40.0, "height": 10.0, "max_steps": 5, "timestep": 0.1},
        )
        comp = ComparisonDefinition(
            comparison_id="comp_esc",
            title="Escaping Test",
            research_question="Verify CSV quotation and multiline escaping.",
            baseline_scenario_id="sc_base",
            scenario_ids=("sc_base", "sc_special"),
            independent_variable="none",
            dependent_metrics=("total_elapsed_time",),
        )
        study = ScenarioComparisonRunner.execute_comparison(comp, [base, sc], repetitions=1)

        csv_text = export_scenario_inventory_csv(study)
        reader = list(csv.DictReader(csv_text.splitlines()))
        assert len(reader) == 2
        special_row = [r for r in reader if r["scenario_id"] == "sc_special"][0]
        assert special_row["scenario_name"] == 'Corridor with "Special, Quote" & Comma'

    def test_empty_dataset_csv_exports(self):
        empty_comp = ComparisonDefinition(
            comparison_id="comp_empty",
            title="Empty",
            research_question="Empty",
            baseline_scenario_id="sc_1",
            scenario_ids=("sc_1", "sc_2"),
            independent_variable="none",
            dependent_metrics=(),
        )
        empty_study = ComparisonStudyResult(
            comparison=empty_comp,
            scenarios={},
            run_records=(),
            per_run_results=(),
            aggregates={},
            comparability_results={},
            metric_deltas={},
            executed_at="2026-11-11T00:00:00Z",
            reproducibility_notes="Empty",
        )

        s_csv = export_scenario_inventory_csv(empty_study)
        assert len(s_csv.strip().splitlines()) == 1  # header only

        m_csv = export_per_run_metrics_csv(empty_study)
        assert len(m_csv.strip().splitlines()) == 1  # header only

        a_csv = export_aggregate_comparison_csv(empty_study)
        assert len(a_csv.strip().splitlines()) == 1  # header only


class TestExperimentExportJSON:
    def test_json_export_structure_and_roundtrip(self, executed_study: ComparisonStudyResult):
        json_str = export_experiment_json(executed_study)
        data = json.loads(json_str)

        assert data["schema_version"] == SCHEMA_VERSION
        assert data["software_version"] == SOFTWARE_VERSION
        assert data["experiment_identifier"] == "study_pillar_eval"

        # Metadata
        meta = data["comparison_metadata"]
        assert meta["title"] == "Pillar Bottleneck Evaluation"
        assert meta["baseline_scenario_id"] == "sc_base_hall"
        assert meta["independent_variable"] == "pillar_presence"

        # Scenarios
        assert len(data["scenarios"]) == 2
        assert "sc_base_hall" in data["scenarios"]
        assert "sc_var_pillar" in data["scenarios"]

        # Run records
        assert len(data["run_records"]) == 4
        for r in data["run_records"]:
            assert r["run_status"] == "SUCCESS"
            assert len(r["config_fingerprint"]) == 64

        # Per run metrics
        assert len(data["per_run_metrics"]) == len(executed_study.per_run_results)
        assert len(data["per_run_metrics"]) == 24
        tp_metrics = [m for m in data["per_run_metrics"] if m["metric_id"] == "exit_throughput"]
        assert len(tp_metrics) == 4
        for m in tp_metrics:
            assert m["value"] is None, "JSON must serialize unavailable value as null, not 0.0"
            assert m["status"] == "UNAVAILABLE"

        # Aggregates
        assert len(data["aggregate_metrics"]) == 10
        tp_aggs = [a for a in data["aggregate_metrics"] if a["metric_id"] == "exit_throughput"]
        for a in tp_aggs:
            assert a["mean"] is None
            assert a["std_dev"] is None
            assert a["eligible_run_count"] == 0

        # Comparability
        assert len(data["comparability_evaluations"]) == 5
        tp_comp = [c for c in data["comparability_evaluations"] if c["metric_id"] == "exit_throughput"][0]
        assert tp_comp["decision"] == "NOT_COMPARABLE"
        assert tp_comp["is_comparable"] is False
        assert "DATA_UNAVAILABLE" in tp_comp["reason_codes"]

        # Reproducibility metadata
        repro = data["reproducibility_metadata"]
        assert repro["seed_used_by_engine"] is False
        assert repro["total_runs_attempted"] == 4
        assert repro["repetition_sample_type"] == "deterministic_duplicate"

    def test_json_export_preserves_failed_run(self, executed_study_with_failure: ComparisonStudyResult):
        json_str = export_experiment_json(executed_study_with_failure)
        data = json.loads(json_str)

        failed_runs = [r for r in data["run_records"] if r["run_status"] == "FAILED"]
        assert len(failed_runs) == 1
        fr = failed_runs[0]
        assert fr["scenario_id"] == "sc_broken_geo"
        assert "width must be a finite positive number" in fr["failure_reason"]


class TestReproducibilityMetadata:
    def test_reproducibility_metadata_truthful_seed_and_sample_type(self, executed_study: ComparisonStudyResult):
        meta = extract_reproducibility_metadata(executed_study)

        assert meta["schema_version"] == "1.0.0"
        assert meta["seed_used_by_engine"] is False
        assert "kinematics are purely deterministic" in meta["seed_usage_note"]
        assert meta["configured_seeds"]["sc_base_hall"] == 100
        assert meta["configured_seeds"]["sc_var_pillar"] == 101
        assert meta["repetition_sample_type"] == "deterministic_duplicate"
        assert "deterministic duplicates and must not be treated as independent stochastic samples" in meta["reproducibility_notes"]
        assert len(meta["scenario_config_fingerprints"]["sc_base_hall"]) == 64
        assert meta["runs_by_status"]["SUCCESS"] == 4
        assert meta["runs_by_status"]["FAILED"] == 0


class TestExperimentReportGeneration:
    def test_experiment_report_has_all_11_sections(self, executed_study: ComparisonStudyResult):
        report = generate_experiment_report(executed_study)

        expected_sections = [
            "# Experiment Evaluation Report",
            "## 1. Experiment Overview",
            "## 2. Scenario Inventory",
            "## 3. Execution Summary",
            "## 4. Per-Run Results",
            "## 5. Aggregate Metric Comparisons",
            "## 6. Comparability Decisions",
            "## 7. Scientific Interpretation",
            "## 8. Unavailable Metrics and Reasons",
            "## 9. Failed Runs and Reasons",
            "## 10. Reproducibility Metadata",
            "## 11. Limitations & Boundaries",
        ]

        for sec in expected_sections:
            assert sec in report, f"Section '{sec}' missing from report!"

        # Section 8 check
        assert "Metric `exit_throughput`" in report
        assert "UNAVAILABLE" in report

        # Section 9 check (no failures)
        assert "zero runtime failures" in report

        # Section 10 check
        assert "seed_used_by_engine=False" in report
        assert "deterministic_duplicate" in report

        # Section 11 check
        assert "No Composite Scoring" in report
        assert "Zero Inferred Evacuations" in report

    def test_experiment_report_with_failed_run(self, executed_study_with_failure: ComparisonStudyResult):
        report = generate_experiment_report(executed_study_with_failure)

        assert "## 9. Failed Runs and Reasons" in report
        assert "Run `sc_broken_geo_run_01`" in report
        assert "Status `FAILED`" in report
        assert "width must be a finite positive number" in report

    def test_export_all_artifacts_to_disk(self, executed_study: ComparisonStudyResult):
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = export_all_experiment_artifacts(executed_study, tmp_dir)

            assert len(artifacts) == 5
            for key, path in artifacts.items():
                assert path.exists(), f"Artifact '{key}' at {path} was not created!"
                assert path.stat().st_size > 0, f"Artifact '{key}' at {path} is empty!"

            # Verify JSON round-trip from written file
            with open(artifacts["experiment_json"], encoding="utf-8") as f:
                loaded_json = json.load(f)
            assert loaded_json["experiment_identifier"] == "study_pillar_eval"

            # Verify CSV readable
            with open(artifacts["scenario_inventory_csv"], encoding="utf-8") as f:
                inv_rows = list(csv.reader(f))
            assert len(inv_rows) == 3  # header + 2 scenarios

            # Verify Report content
            report_text = artifacts["experiment_report_md"].read_text(encoding="utf-8")
            assert "# Experiment Evaluation Report" in report_text
