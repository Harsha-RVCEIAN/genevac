"""Tests for Phase 06 Statistical Aggregation and Determinism."""

import math
import pytest

from genevac.experiments.scenario import (
    PerRunMetricResult,
    RunRecord,
    RunStatus,
    TerminationReason,
    aggregate_per_run_results,
)


class TestAggregationPhase06:
    def _create_sample_run(self, run_id: str, scenario_id: str, status: RunStatus = RunStatus.SUCCESS) -> RunRecord:
        return RunRecord(
            run_id=run_id,
            scenario_id=scenario_id,
            scenario_version="1.0.0",
            started_at="2026-11-09T20:00:00Z",
            ended_at="2026-11-09T20:00:10Z",
            seed=42,
            initial_agent_count=20,
            final_step=100,
            simulated_duration=10.0,
            termination_reason=TerminationReason.COMPLETED if status == RunStatus.SUCCESS else TerminationReason.ERROR,
            run_status=status,
            failure_reason="Crash in engine" if status == RunStatus.FAILED else "",
        )

    def test_multi_run_mean_and_stdev(self):
        scenario_id = "sc_01"
        metric_id = "clearance_time"

        runs = [
            self._create_sample_run("run_1", scenario_id),
            self._create_sample_run("run_2", scenario_id),
            self._create_sample_run("run_3", scenario_id),
        ]

        # Values: 10.0, 12.0, 14.0 -> mean 12.0, median 12.0, sample stdev 2.0
        results = [
            PerRunMetricResult("run_1", scenario_id, metric_id, "Clearance Time", 10.0, "s"),
            PerRunMetricResult("run_2", scenario_id, metric_id, "Clearance Time", 12.0, "s"),
            PerRunMetricResult("run_3", scenario_id, metric_id, "Clearance Time", 14.0, "s"),
        ]

        agg = aggregate_per_run_results(results, runs, scenario_id, metric_id, unit="s")
        assert agg.eligible_run_count == 3
        assert agg.attempted_run_count == 3
        assert agg.successful_run_count == 3
        assert agg.failed_run_count == 0
        assert math.isclose(agg.mean, 12.0)
        assert math.isclose(agg.median, 12.0)
        assert math.isclose(agg.std_dev, 2.0)
        assert agg.is_single_run is False
        assert agg.contributing_run_ids == ("run_1", "run_2", "run_3")

    def test_single_run_variability_is_unavailable(self):
        scenario_id = "sc_01"
        metric_id = "clearance_time"

        runs = [self._create_sample_run("run_1", scenario_id)]
        results = [PerRunMetricResult("run_1", scenario_id, metric_id, "Clearance Time", 15.5, "s")]

        agg = aggregate_per_run_results(results, runs, scenario_id, metric_id, unit="s")
        assert agg.eligible_run_count == 1
        assert math.isclose(agg.mean, 15.5)
        assert math.isclose(agg.median, 15.5)
        assert agg.std_dev is None, "Variability for single run must be None (not 0.0)."
        assert agg.is_single_run is True

    def test_failed_runs_retained_in_audit_not_coerced_to_zero(self):
        scenario_id = "sc_01"
        metric_id = "mean_speed"

        runs = [
            self._create_sample_run("run_1", scenario_id, RunStatus.SUCCESS),
            self._create_sample_run("run_2", scenario_id, RunStatus.FAILED),
        ]
        results = [
            PerRunMetricResult("run_1", scenario_id, metric_id, "Speed", 1.4, "m/s", status="VALID"),
            PerRunMetricResult("run_2", scenario_id, metric_id, "Speed", None, "m/s", status="FAILED", reason="Crash"),
        ]

        agg = aggregate_per_run_results(results, runs, scenario_id, metric_id, unit="m/s")
        assert agg.attempted_run_count == 2
        assert agg.successful_run_count == 1
        assert agg.failed_run_count == 1
        assert agg.eligible_run_count == 1
        assert agg.excluded_run_count == 1
        assert math.isclose(agg.mean, 1.4)
        assert len(agg.exclusion_reasons) > 0
        assert "Run run_2: Run status is FAILED" in agg.exclusion_reasons[0]

    def test_unavailable_metric_not_coerced_to_zero(self):
        scenario_id = "sc_01"
        metric_id = "exit_throughput"

        runs = [self._create_sample_run("run_1", scenario_id, RunStatus.SUCCESS)]
        results = [
            PerRunMetricResult("run_1", scenario_id, metric_id, "Throughput", None, "ped/s", status="UNAVAILABLE", reason="No events")
        ]

        agg = aggregate_per_run_results(results, runs, scenario_id, metric_id, unit="ped/s")
        assert agg.eligible_run_count == 0
        assert agg.unavailable_count == 1
        assert agg.mean is None
        assert agg.median is None
        assert agg.std_dev is None

    def test_deterministic_output_under_input_reordering(self):
        scenario_id = "sc_01"
        metric_id = "metric_a"

        runs = [
            self._create_sample_run("run_c", scenario_id),
            self._create_sample_run("run_a", scenario_id),
            self._create_sample_run("run_b", scenario_id),
        ]
        results = [
            PerRunMetricResult("run_c", scenario_id, metric_id, "A", 30.0, "m"),
            PerRunMetricResult("run_a", scenario_id, metric_id, "A", 10.0, "m"),
            PerRunMetricResult("run_b", scenario_id, metric_id, "A", 20.0, "m"),
        ]

        # Aggregate forward
        agg1 = aggregate_per_run_results(results, runs, scenario_id, metric_id, unit="m")

        # Aggregate reversed
        agg2 = aggregate_per_run_results(list(reversed(results)), list(reversed(runs)), scenario_id, metric_id, unit="m")

        assert agg1.mean == agg2.mean
        assert agg1.std_dev == agg2.std_dev
        assert agg1.contributing_run_ids == ("run_a", "run_b", "run_c")
        assert agg2.contributing_run_ids == ("run_a", "run_b", "run_c")
