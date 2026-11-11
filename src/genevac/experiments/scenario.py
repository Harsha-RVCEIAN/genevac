"""Scenario and execution data models for GenEvac-Sentinel Phase 06.

Provides immutable data contracts preserving:
1. Scenario definition and provenance.
2. Individual run execution records and termination status.
3. Per-run metric results with lineage.
4. Formal comparison study definitions.
5. Statistically rigorous, deterministic aggregated summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple


class RunStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class TerminationReason(str, Enum):
    COMPLETED = "COMPLETED"
    MAX_STEPS_REACHED = "MAX_STEPS_REACHED"
    STOPPED_EARLY = "STOPPED_EARLY"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ScenarioDefinition:
    """Named, versioned scenario configuration preserving design intent and provenance."""
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

    def __post_init__(self) -> None:
        if not self.scenario_id or not str(self.scenario_id).strip():
            raise ValueError("scenario_id must be a non-empty string.")
        if not self.name or not str(self.name).strip():
            raise ValueError("name must be a non-empty string.")
        if self.agent_count < 0:
            raise ValueError(f"agent_count cannot be negative, got {self.agent_count}")

    def compute_fingerprint(self) -> str:
        """Compute a deterministic SHA-256 fingerprint from canonical scenario configuration."""
        canonical_dict = {
            "scenario_id": self.scenario_id,
            "version": self.version,
            "environment_id": self.environment_id,
            "environment_version": self.environment_version,
            "environment_provenance": self.environment_provenance,
            "agent_count": self.agent_count,
            "initial_conditions_ref": self.initial_conditions_ref,
            "engine_config": self.engine_config,
            "seed": self.seed,
            "stopping_condition": self.stopping_condition,
            "max_duration": self.max_duration,
        }
        serialized = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate engine configuration parameters against supported ranges and geometries."""
        errors: List[str] = []
        cfg = self.engine_config or {}

        width = cfg.get("width", 50.0)
        if not (isinstance(width, (int, float)) and math.isfinite(width) and width > 0):
            errors.append(f"width must be a finite positive number, got {width!r}")

        height = cfg.get("height", 20.0)
        if not (isinstance(height, (int, float)) and math.isfinite(height) and height > 0):
            errors.append(f"height must be a finite positive number, got {height!r}")

        max_steps = cfg.get("max_steps", 100)
        if not (isinstance(max_steps, int) and not isinstance(max_steps, bool) and max_steps >= 1):
            errors.append(f"max_steps must be a positive integer, got {max_steps!r}")

        timestep = cfg.get("timestep", 0.1)
        if not (isinstance(timestep, (int, float)) and math.isfinite(timestep) and timestep > 0):
            errors.append(f"timestep must be a finite positive number, got {timestep!r}")

        if self.agent_count < 0:
            errors.append(f"agent_count cannot be negative, got {self.agent_count}")

        # Check exits
        exits = cfg.get("exits", [])
        if not isinstance(exits, list):
            errors.append(f"exits must be a list of exit specifications, got {type(exits).__name__}")
        else:
            for i, ex in enumerate(exits):
                if not isinstance(ex, dict) or not ex.get("id"):
                    errors.append(f"exit[{i}] missing required 'id'")
                pos = ex.get("position")
                if not (isinstance(pos, (list, tuple)) and len(pos) == 2):
                    errors.append(f"exit[{i}] position must be a 2-tuple (x, y)")
                else:
                    ex_x, ex_y = pos
                    if not (math.isfinite(ex_x) and math.isfinite(ex_y)):
                        errors.append(f"exit[{i}] position coordinates must be finite, got {pos!r}")
                    elif not (0.0 <= ex_x <= width and 0.0 <= ex_y <= height):
                        errors.append(f"exit[{i}] out-of-bounds: ({ex_x}, {ex_y}) outside room dimensions ({width}, {height})")

        # Check obstacles
        obstacles = cfg.get("obstacles", [])
        if not isinstance(obstacles, list):
            errors.append(f"obstacles must be a list of obstacle specifications, got {type(obstacles).__name__}")
        else:
            for i, obs in enumerate(obstacles):
                if not isinstance(obs, dict) or not obs.get("id"):
                    errors.append(f"obstacle[{i}] missing required 'id'")
                bounds = obs.get("bounds")
                if not (isinstance(bounds, (list, tuple)) and len(bounds) == 4):
                    errors.append(f"obstacle[{i}] bounds must be a 4-tuple (min_x, min_y, max_x, max_y)")
                else:
                    min_x, min_y, max_x, max_y = bounds
                    if not (math.isfinite(min_x) and math.isfinite(min_y) and math.isfinite(max_x) and math.isfinite(max_y)):
                        errors.append(f"obstacle[{i}] coordinates must be finite, got {bounds!r}")
                    elif min_x >= max_x or min_y >= max_y:
                        errors.append(f"obstacle[{i}] has invalid degenerate geometry: min bounds must be strictly less than max bounds ({bounds!r})")
                    elif min_x < 0.0 or max_x > width or min_y < 0.0 or max_y > height:
                        errors.append(f"obstacle[{i}] out-of-bounds: bounds {bounds!r} exceed room dimensions ({width}, {height})")

        return (len(errors) == 0, errors)


@dataclass(frozen=True)
class RunRecord:
    """Execution record for an attempted simulation run."""
    run_id: str
    scenario_id: str
    scenario_version: str
    started_at: str
    ended_at: str
    seed: Optional[int]
    initial_agent_count: int
    final_step: int
    simulated_duration: float
    termination_reason: TerminationReason
    run_status: RunStatus
    failure_reason: str = ""
    config_fingerprint: str = ""
    source_provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id or not str(self.run_id).strip():
            raise ValueError("run_id must be a non-empty string.")
        if not self.scenario_id or not str(self.scenario_id).strip():
            raise ValueError("scenario_id must be a non-empty string.")
        if self.simulated_duration < 0.0 or not math.isfinite(self.simulated_duration):
            raise ValueError(f"simulated_duration must be a finite non-negative number, got {self.simulated_duration!r}")


@dataclass(frozen=True)
class PerRunMetricResult:
    """Individual metric calculation record for a specific run."""
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


@dataclass(frozen=True)
class ComparisonDefinition:
    """Structured declaration of a scientific scenario comparison."""
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

    def __post_init__(self) -> None:
        if not self.comparison_id or not str(self.comparison_id).strip():
            raise ValueError("comparison_id must be a non-empty string.")
        if not self.baseline_scenario_id:
            raise ValueError("baseline_scenario_id must be specified.")
        if self.baseline_scenario_id not in self.scenario_ids:
            raise ValueError(f"baseline_scenario_id '{self.baseline_scenario_id}' must be in scenario_ids {self.scenario_ids}")
        if len(self.scenario_ids) < 2:
            raise ValueError("Comparison requires at least two scenario IDs (baseline and variant).")


@dataclass(frozen=True)
class AggregateMetricResult:
    """Summary of eligible per-run metric results preserving full audit metadata."""
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
    std_dev: Optional[float]
    min_val: Optional[float]
    max_val: Optional[float]
    aggregation_method: str = "sample_mean_and_stdev"
    inclusion_reasons: Tuple[str, ...] = field(default_factory=tuple)
    exclusion_reasons: Tuple[str, ...] = field(default_factory=tuple)
    contributing_run_ids: Tuple[str, ...] = field(default_factory=tuple)
    is_single_run: bool = False
    is_deterministic_repetition: bool = False


def aggregate_per_run_results(
    results: Sequence[PerRunMetricResult],
    run_records: Sequence[RunRecord],
    scenario_id: str,
    metric_id: str,
    unit: str = "",
    min_samples_for_std: int = 2,
) -> AggregateMetricResult:
    """Deterministically aggregate valid eligible per-run results for a given scenario and metric.

    Rules:
    1. Only runs associated with scenario_id and matching metric_id are processed.
    2. Runs with status != SUCCESS or metric status != VALID are excluded from numeric moments.
    3. Unavailable / failed runs are tracked in count tallies and reasons, never coerced to 0.0.
    4. For a single eligible run, std_dev is None and is_single_run is True.
    5. Contributing run IDs and output ordering are deterministically sorted.
    """
    scenario_runs = {r.run_id: r for r in run_records if r.scenario_id == scenario_id}
    relevant_results = [
        r for r in results if r.scenario_id == scenario_id and r.metric_id == metric_id
    ]

    attempted_count = len(relevant_results)
    successful_runs = sum(1 for r in relevant_results if scenario_runs.get(r.run_id) and scenario_runs[r.run_id].run_status == RunStatus.SUCCESS)
    partial_runs = sum(1 for r in relevant_results if scenario_runs.get(r.run_id) and scenario_runs[r.run_id].run_status == RunStatus.PARTIAL)
    failed_runs = sum(1 for r in relevant_results if scenario_runs.get(r.run_id) and scenario_runs[r.run_id].run_status == RunStatus.FAILED)

    eligible_values: List[float] = []
    contributing_ids: List[str] = []
    exclusion_reasons: List[str] = []
    inclusion_reasons: List[str] = []
    unavailable_count = 0

    for res in sorted(relevant_results, key=lambda x: x.run_id):
        run_rec = scenario_runs.get(res.run_id)
        if run_rec is None:
            exclusion_reasons.append(f"Run {res.run_id}: No matching RunRecord found.")
            continue

        if run_rec.run_status != RunStatus.SUCCESS:
            exclusion_reasons.append(f"Run {res.run_id}: Run status is {run_rec.run_status.value} ({run_rec.failure_reason}).")
            continue

        if res.status != "VALID":
            unavailable_count += 1
            exclusion_reasons.append(f"Run {res.run_id}: Metric status is {res.status} ({res.reason}).")
            continue

        if res.value is None or not math.isfinite(res.value):
            unavailable_count += 1
            exclusion_reasons.append(f"Run {res.run_id}: Metric value is non-finite or None ({res.value}).")
            continue

        eligible_values.append(float(res.value))
        contributing_ids.append(res.run_id)
        inclusion_reasons.append(f"Run {res.run_id}: Valid observation {res.value:.4f} {unit}")

    eligible_count = len(eligible_values)
    excluded_count = attempted_count - eligible_count

    if eligible_count == 0:
        return AggregateMetricResult(
            scenario_id=scenario_id,
            metric_id=metric_id,
            unit=unit,
            attempted_run_count=attempted_count,
            successful_run_count=successful_runs,
            partial_run_count=partial_runs,
            failed_run_count=failed_runs,
            eligible_run_count=0,
            excluded_run_count=excluded_count,
            unavailable_count=unavailable_count,
            mean=None,
            median=None,
            std_dev=None,
            min_val=None,
            max_val=None,
            inclusion_reasons=tuple(inclusion_reasons),
            exclusion_reasons=tuple(exclusion_reasons),
            contributing_run_ids=(),
            is_single_run=False,
        )

    mean_val = statistics.fmean(eligible_values)
    median_val = statistics.median(eligible_values)
    min_val = min(eligible_values)
    max_val = max(eligible_values)

    if eligible_count >= min_samples_for_std:
        std_val = statistics.stdev(eligible_values)
        is_single = False
        is_deterministic = math.isclose(std_val, 0.0)
    else:
        std_val = None
        is_single = True
        is_deterministic = False

    return AggregateMetricResult(
        scenario_id=scenario_id,
        metric_id=metric_id,
        unit=unit,
        attempted_run_count=attempted_count,
        successful_run_count=successful_runs,
        partial_run_count=partial_runs,
        failed_run_count=failed_runs,
        eligible_run_count=eligible_count,
        excluded_run_count=excluded_count,
        unavailable_count=unavailable_count,
        mean=mean_val,
        median=median_val,
        std_dev=std_val,
        min_val=min_val,
        max_val=max_val,
        inclusion_reasons=tuple(inclusion_reasons),
        exclusion_reasons=tuple(exclusion_reasons),
        contributing_run_ids=tuple(contributing_ids),
        is_single_run=is_single,
        is_deterministic_repetition=is_deterministic,
    )
