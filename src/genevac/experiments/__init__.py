"""Scenario definitions, execution, comparisons, and validation."""

from typing import Dict, Any, List
from genevac.environment import EnvironmentMap, Exit
from genevac.simulation import SimulationEngine
from genevac.experiments.validation import validate_experiment_config
from genevac.experiments.scenario import (
    AggregateMetricResult,
    ComparisonDefinition,
    PerRunMetricResult,
    RunRecord,
    RunStatus,
    ScenarioDefinition,
    TerminationReason,
    aggregate_per_run_results,
)
from genevac.experiments.comparability import (
    ComparabilityDecision,
    ComparabilityGate,
    ComparabilityResult,
)
from genevac.experiments.runner import (
    ComparisonStudyResult,
    MetricDelta,
    ScenarioComparisonRunner,
)
from genevac.experiments.presentation import (
    generate_metric_comparison_table,
    generate_per_run_detail_table,
    generate_scenario_inventory_table,
    generate_scientific_interpretation,
)


class ExperimentRunner:
    """Legacy scenario runner maintained for backward compatibility."""
    def __init__(self, scenarios: List[Dict[str, Any]]):
        self.scenarios = scenarios

    def run_all(self) -> List[Dict[str, Any]]:
        results = []
        for sc in self.scenarios:
            env = EnvironmentMap(width=sc.get("width", 50.0), height=sc.get("height", 20.0))
            for ex in sc.get("exits", []):
                env.add_exit(Exit(id=ex["id"], position=tuple(ex["position"]), width=ex.get("width", 2.0)))
            engine = SimulationEngine(environment=env, max_steps=sc.get("max_steps", 100))
            summary = engine.run()
            results.append({"name": sc.get("name", "unnamed"), "summary": summary})
        return results


__all__ = [
    "ExperimentRunner",
    "validate_experiment_config",
    "ScenarioDefinition",
    "RunRecord",
    "PerRunMetricResult",
    "ComparisonDefinition",
    "AggregateMetricResult",
    "RunStatus",
    "TerminationReason",
    "aggregate_per_run_results",
    "ComparabilityDecision",
    "ComparabilityGate",
    "ComparabilityResult",
    "ScenarioComparisonRunner",
    "ComparisonStudyResult",
    "MetricDelta",
    "generate_scenario_inventory_table",
    "generate_metric_comparison_table",
    "generate_per_run_detail_table",
    "generate_scientific_interpretation",
]
