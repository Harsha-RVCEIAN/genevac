"""Metric and scenario comparability validation gate for GenEvac-Sentinel Phase 06.

Enforces scientific comparability rules prior to cross-scenario aggregation or plotting:
- Verifies dimensional consistency, time horizons, agent cohort populations, and spatial bounds.
- Rejects comparisons across incompatible grid geometries, units, or definitions.
- Explicitly flags unapproved metrics (e.g. ERSI, Delta Score) as BLOCKED / NOT_COMPARABLE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


class ComparabilityDecision(str, Enum):
    COMPARABLE = "COMPARABLE"
    COMPARABLE_WITH_CAVEATS = "COMPARABLE_WITH_CAVEATS"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    UNDETERMINED = "UNDETERMINED"


@dataclass(frozen=True)
class ComparabilityResult:
    """Outcome of a comparability audit between two scenarios for a specific metric."""
    metric_id: str
    decision: ComparabilityDecision
    reason_codes: Tuple[str, ...]
    explanation: str
    caveats: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_comparable(self) -> bool:
        return self.decision in (ComparabilityDecision.COMPARABLE, ComparabilityDecision.COMPARABLE_WITH_CAVEATS)


class ComparabilityGate:
    """Audit gate enforcing scientific comparability rules across scenario configurations."""

    BLOCKED_METRIC_PREFIXES = ("ersi", "delta_score", "risk_index", "composite_rank")

    @classmethod
    def evaluate_metric_comparability(
        cls,
        metric_id: str,
        baseline_meta: Dict[str, Any],
        variant_meta: Dict[str, Any],
        comparability_policy: Optional[Dict[str, Any]] = None,
    ) -> ComparabilityResult:
        """Evaluate if metric_id can be validly compared between baseline and variant.

        Args:
            metric_id: Identifier of the metric being compared.
            baseline_meta: Context metadata for baseline scenario (unit, window, population, grid, etc.).
            variant_meta: Context metadata for variant scenario.
            comparability_policy: Optional policy rules (e.g. allow_unequal_agent_counts, shared_window).

        Returns:
            ComparabilityResult with explicit decision, reason codes, explanation, and caveats.
        """
        policy = comparability_policy or {}
        reasons: List[str] = []
        caveats: List[str] = []

        # 1. Check for blocked / undefined metrics (ERSI, Delta Score)
        metric_lower = metric_id.lower()
        for blocked in cls.BLOCKED_METRIC_PREFIXES:
            if blocked in metric_lower:
                return ComparabilityResult(
                    metric_id=metric_id,
                    decision=ComparabilityDecision.NOT_COMPARABLE,
                    reason_codes=("METRIC_BLOCKED",),
                    explanation=f"Metric '{metric_id}' lacks an authoritative specification and is blocked from comparison.",
                )

        # 2. Check Unit Compatibility
        base_unit = baseline_meta.get("unit", "").strip()
        var_unit = variant_meta.get("unit", "").strip()
        if base_unit != var_unit:
            return ComparabilityResult(
                metric_id=metric_id,
                decision=ComparabilityDecision.NOT_COMPARABLE,
                reason_codes=("UNIT_MISMATCH",),
                explanation=f"Unit mismatch: baseline has '{base_unit}' but variant has '{var_unit}'.",
            )

        # 3. Check Metric Definition Version
        base_ver = baseline_meta.get("metric_definition_version", "1.0.0")
        var_ver = variant_meta.get("metric_definition_version", "1.0.0")
        if base_ver != var_ver:
            return ComparabilityResult(
                metric_id=metric_id,
                decision=ComparabilityDecision.NOT_COMPARABLE,
                reason_codes=("DEFINITION_VERSION_MISMATCH",),
                explanation=f"Implementation version mismatch: '{base_ver}' vs '{var_ver}'.",
            )

        # 4. Check Trajectory Prediction Horizons (for ADE/FDE)
        if any(term in metric_lower for term in ("ade", "fde", "displacement_error")):
            base_horizon = baseline_meta.get("prediction_horizon")
            var_horizon = variant_meta.get("prediction_horizon")
            if base_horizon != var_horizon and (base_horizon is not None or var_horizon is not None):
                return ComparabilityResult(
                    metric_id=metric_id,
                    decision=ComparabilityDecision.NOT_COMPARABLE,
                    reason_codes=("HORIZON_MISMATCH",),
                    explanation=f"Prediction horizon mismatch: baseline {base_horizon} steps vs variant {var_horizon} steps.",
                )

        # 5. Check Spatial Coordinate Grid Compatibility (for heatmaps / spatial density)
        if any(term in metric_lower for term in ("heatmap", "spatial_density", "grid_occupancy")):
            base_grid = baseline_meta.get("grid_resolution")
            var_grid = variant_meta.get("grid_resolution")
            base_bounds = baseline_meta.get("environment_bounds")
            var_bounds = variant_meta.get("environment_bounds")

            if base_grid != var_grid:
                return ComparabilityResult(
                    metric_id=metric_id,
                    decision=ComparabilityDecision.NOT_COMPARABLE,
                    reason_codes=("GRID_RESOLUTION_MISMATCH",),
                    explanation=f"Grid resolution mismatch: {base_grid} vs {var_grid}. Cell-by-cell comparison invalid without regridding.",
                )
            if base_bounds != var_bounds:
                return ComparabilityResult(
                    metric_id=metric_id,
                    decision=ComparabilityDecision.NOT_COMPARABLE,
                    reason_codes=("SPATIAL_BOUNDS_MISMATCH",),
                    explanation=f"Spatial bounding box mismatch: {base_bounds} vs {var_bounds}.",
                )

        # 6. Check Route / Destination Compatibility (for route/path metrics)
        if any(term in metric_lower for term in ("route", "path_length", "path_validation")):
            base_dest = baseline_meta.get("target_destination_id")
            var_dest = variant_meta.get("target_destination_id")
            if base_dest != var_dest and (base_dest is not None or var_dest is not None):
                return ComparabilityResult(
                    metric_id=metric_id,
                    decision=ComparabilityDecision.NOT_COMPARABLE,
                    reason_codes=("DESTINATION_MISMATCH",),
                    explanation=f"Route target destination mismatch: '{base_dest}' vs '{var_dest}'. Path lengths cannot be compared.",
                )

        # 7. Check Duration & Time Windows
        base_duration = baseline_meta.get("simulated_duration")
        var_duration = variant_meta.get("simulated_duration")
        allow_unequal_duration = policy.get("allow_unequal_duration", False)

        if base_duration is not None and var_duration is not None:
            if not math.isclose(base_duration, var_duration, rel_tol=1e-3, abs_tol=1e-3):
                if not allow_unequal_duration:
                    reasons.append("DURATION_MISMATCH")
                else:
                    reasons.append("DURATION_MISMATCH_ALLOWED")
                caveats.append(f"Simulated duration differs ({base_duration:.1f}s vs {var_duration:.1f}s). Cumulative totals must not be compared without rate normalization.")

        # 8. Check Population / Agent Counts
        base_agents = baseline_meta.get("agent_count")
        var_agents = variant_meta.get("agent_count")
        allow_unequal_agents = policy.get("allow_unequal_agents", False)

        if base_agents is not None and var_agents is not None:
            if base_agents != var_agents:
                reasons.append("POPULATION_MISMATCH")
                caveats.append(f"Agent counts differ ({base_agents} vs {var_agents}). Raw counts reflect population size differences.")

        # Final Decision Synthesis
        if "DURATION_MISMATCH" in reasons and not allow_unequal_duration:
            return ComparabilityResult(
                metric_id=metric_id,
                decision=ComparabilityDecision.NOT_COMPARABLE,
                reason_codes=tuple(reasons),
                explanation=f"Unequal durations ({base_duration}s vs {var_duration}s) without a shared window policy.",
                caveats=tuple(caveats),
            )

        if caveats:
            return ComparabilityResult(
                metric_id=metric_id,
                decision=ComparabilityDecision.COMPARABLE_WITH_CAVEATS,
                reason_codes=tuple(reasons) if reasons else ("CAVEATS_APPLIED",),
                explanation="Metrics are comparable subject to noted population or duration qualifications.",
                caveats=tuple(caveats),
            )

        return ComparabilityResult(
            metric_id=metric_id,
            decision=ComparabilityDecision.COMPARABLE,
            reason_codes=("FULLY_COMPARABLE",),
            explanation="Metric definitions, units, windows, spatial frames, and populations match.",
            caveats=(),
        )
