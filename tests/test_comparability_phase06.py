"""Tests for Phase 06 Metric Comparability Gate."""

import pytest

from genevac.experiments.comparability import (
    ComparabilityDecision,
    ComparabilityGate,
)


class TestComparabilityPhase06:
    def test_identical_metrics_are_comparable(self):
        base_meta = {
            "unit": "seconds",
            "metric_definition_version": "1.0.0",
            "simulated_duration": 50.0,
            "agent_count": 30,
        }
        var_meta = {
            "unit": "seconds",
            "metric_definition_version": "1.0.0",
            "simulated_duration": 50.0,
            "agent_count": 30,
        }

        res = ComparabilityGate.evaluate_metric_comparability("clearance_time", base_meta, var_meta)
        assert res.decision == ComparabilityDecision.COMPARABLE
        assert res.is_comparable is True
        assert "FULLY_COMPARABLE" in res.reason_codes

    def test_unit_mismatch_rejected(self):
        base_meta = {"unit": "seconds"}
        var_meta = {"unit": "minutes"}

        res = ComparabilityGate.evaluate_metric_comparability("clearance_time", base_meta, var_meta)
        assert res.decision == ComparabilityDecision.NOT_COMPARABLE
        assert "UNIT_MISMATCH" in res.reason_codes
        assert res.is_comparable is False

    def test_prediction_horizon_mismatch_rejected(self):
        base_meta = {"unit": "metres", "prediction_horizon": 10}
        var_meta = {"unit": "metres", "prediction_horizon": 20}

        res = ComparabilityGate.evaluate_metric_comparability("ade", base_meta, var_meta)
        assert res.decision == ComparabilityDecision.NOT_COMPARABLE
        assert "HORIZON_MISMATCH" in res.reason_codes

    def test_spatial_grid_mismatch_rejected(self):
        base_meta = {"unit": "ped/m²", "grid_resolution": 1.0, "environment_bounds": (0, 0, 50, 20)}
        var_meta = {"unit": "ped/m²", "grid_resolution": 0.5, "environment_bounds": (0, 0, 50, 20)}

        res = ComparabilityGate.evaluate_metric_comparability("spatial_density_grid", base_meta, var_meta)
        assert res.decision == ComparabilityDecision.NOT_COMPARABLE
        assert "GRID_RESOLUTION_MISMATCH" in res.reason_codes

    def test_blocked_metrics_rejected(self):
        base_meta = {"unit": "score"}
        var_meta = {"unit": "score"}

        for blocked in ("ersi_score", "delta_score", "risk_index"):
            res = ComparabilityGate.evaluate_metric_comparability(blocked, base_meta, var_meta)
            assert res.decision == ComparabilityDecision.NOT_COMPARABLE
            assert "METRIC_BLOCKED" in res.reason_codes
            assert "lacks an authoritative specification" in res.explanation

    def test_route_destination_mismatch_rejected(self):
        base_meta = {"unit": "metres", "target_destination_id": "exit_east"}
        var_meta = {"unit": "metres", "target_destination_id": "exit_west"}

        res = ComparabilityGate.evaluate_metric_comparability("path_length", base_meta, var_meta)
        assert res.decision == ComparabilityDecision.NOT_COMPARABLE
        assert "DESTINATION_MISMATCH" in res.reason_codes

    def test_unequal_agent_counts_trigger_caveats(self):
        base_meta = {"unit": "m/s", "agent_count": 20}
        var_meta = {"unit": "m/s", "agent_count": 40}

        res = ComparabilityGate.evaluate_metric_comparability("mean_speed", base_meta, var_meta)
        assert res.decision == ComparabilityDecision.COMPARABLE_WITH_CAVEATS
        assert res.is_comparable is True
        assert "POPULATION_MISMATCH" in res.reason_codes
        assert len(res.caveats) > 0

    def test_duration_mismatch_policy(self):
        base_meta = {"unit": "ped", "simulated_duration": 10.0}
        var_meta = {"unit": "ped", "simulated_duration": 20.0}

        # By default without policy -> rejected
        res_default = ComparabilityGate.evaluate_metric_comparability("evacuated_total", base_meta, var_meta)
        assert res_default.decision == ComparabilityDecision.NOT_COMPARABLE
        assert "DURATION_MISMATCH" in res_default.reason_codes

        # With explicit policy allowing unequal duration with caveats
        res_allowed = ComparabilityGate.evaluate_metric_comparability(
            "evacuated_total",
            base_meta,
            var_meta,
            comparability_policy={"allow_unequal_duration": True},
        )
        assert res_allowed.decision == ComparabilityDecision.COMPARABLE_WITH_CAVEATS
        assert len(res_allowed.caveats) > 0
