"""Scenario definitions, execution, and comparisons."""

from typing import Dict, Any, List
from genevac.environment import EnvironmentMap, Exit
from genevac.simulation import SimulationEngine


class ExperimentRunner:
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
