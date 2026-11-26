"""Batch simulation generation workflow for GenEvac-Sentinel.

Runs multiple simulation scenarios back-to-back using the existing
``SimulationEngine`` / ``ScenarioDefinition`` interfaces, records each run
with ``SimulationRecorder``, validates it, and writes a master batch manifest.

Usage
-----
::

    from genevac.prediction.simulation_batch import BatchGenerator

    gen = BatchGenerator(
        output_dir=Path("data/simulation/raw/v1"),
        scenarios=[my_scenario],
        runs_per_scenario=5,
    )
    summary = gen.run()
    print(summary)
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from genevac.data.simulation.schema import generate_run_id, iso_timestamp
from genevac.environment import EnvironmentMap, Exit, Obstacle
from genevac.experiments.scenario import ScenarioDefinition, RunStatus
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.prediction.simulation_recorder import record_full_run
from genevac.prediction.simulation_validator import validate_dataset
from genevac.simulation import SimulationEngine


class BatchGenerator:
    """Generate, record, and validate multiple simulation runs.

    Parameters
    ----------
    output_dir:
        Root directory for raw output (sub-dirs ``trajectories/`` and
        ``manifests/`` are created automatically).
    scenarios:
        Sequence of :class:`ScenarioDefinition` objects to run.
    runs_per_scenario:
        How many independent runs to execute per scenario.
    base_seed:
        Starting seed; each run uses ``base_seed + run_index`` so that runs
        are reproducible without sharing a seed.
    """

    def __init__(
        self,
        output_dir: Path,
        scenarios: Sequence[ScenarioDefinition],
        *,
        runs_per_scenario: int = 1,
        base_seed: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.scenarios = list(scenarios)
        self.runs_per_scenario = runs_per_scenario
        self.base_seed = base_seed

        self._traj_dir = self.output_dir / "trajectories"
        self._manifests_dir = self.output_dir / "manifests"
        self._traj_dir.mkdir(parents=True, exist_ok=True)
        self._manifests_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute all runs and return a batch summary dict."""
        completed: List[str] = []
        interrupted: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        global_index = 0

        for scenario in self.scenarios:
            for rep in range(self.runs_per_scenario):
                seed = self.base_seed + global_index
                global_index += 1
                run_id = generate_run_id()
                try:
                    engine = self._build_engine(scenario, seed)
                    metadata, traj_path = record_full_run(
                        engine,
                        scenario_id=scenario.scenario_id,
                        layout_id=scenario.environment_id,
                        seed=seed,
                        run_id=run_id,
                        trajectories_dir=self._traj_dir,
                        manifests_dir=self._manifests_dir,
                    )
                    if metadata.status == "completed":
                        completed.append(run_id)
                    elif metadata.status == "interrupted":
                        interrupted.append({
                            "scenario_id": scenario.scenario_id,
                            "layout_id": scenario.environment_id,
                            "seed": seed,
                            "run_id": run_id,
                            "status": metadata.status,
                            "reason": metadata.failure_reason or "run interrupted",
                        })
                    else:
                        failed.append({
                            "scenario_id": scenario.scenario_id,
                            "layout_id": scenario.environment_id,
                            "seed": seed,
                            "run_id": run_id,
                            "status": metadata.status,
                            "error": metadata.failure_reason or "run did not complete",
                        })
                except Exception as exc:  # noqa: BLE001
                    failed.append({
                        "scenario_id": scenario.scenario_id,
                        "layout_id": scenario.environment_id,
                        "seed": seed,
                        "run_id": run_id,
                        "status": "failed",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    })

        # Validate the whole batch
        validation_report = validate_dataset(self._manifests_dir)

        summary = {
            "schema_version": "v1",
            "generated_at": iso_timestamp(),
            "completed_runs": completed,
            "interrupted_runs": interrupted,
            "failed_runs": failed,
            "total_completed": len(completed),
            "total_interrupted": len(interrupted),
            "total_failed": len(failed),
            "validation": validation_report,
        }

        # Write master manifest
        master_path = self._manifests_dir / "batch_manifest.json"
        master_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_engine(self, scenario: ScenarioDefinition, seed: int) -> SimulationEngine:
        """Build a ``SimulationEngine`` from a ``ScenarioDefinition`` config."""
        cfg = scenario.engine_config or {}
        width = float(cfg.get("width", 50.0))
        height = float(cfg.get("height", 20.0))
        max_steps = int(cfg.get("max_steps", 200))
        timestep = float(cfg.get("timestep", 0.1))

        env = EnvironmentMap(width=width, height=height)
        for ex in cfg.get("exits", []):
            env.add_exit(Exit(
                id=ex["id"],
                position=tuple(ex["position"]),
                width=float(ex.get("width", 2.0)),
            ))
        for obs in cfg.get("obstacles", []):
            env.add_obstacle(Obstacle(id=obs["id"], bounds=tuple(obs["bounds"])))

        engine = SimulationEngine(
            environment=env,
            max_steps=max_steps,
            timestep=timestep,
            seed=seed,
        )

        peds_cfg = cfg.get("pedestrians", [])
        if peds_cfg:
            for p_data in peds_cfg:
                engine.add_pedestrian(Pedestrian(
                    id=p_data["id"],
                    position=tuple(p_data["position"]),
                    velocity=tuple(p_data.get("velocity", (1.0, 0.0))),
                    desired_speed=float(p_data.get("desired_speed", 1.34)),
                    radius=float(p_data.get("radius", 0.3)),
                    target_exit=p_data.get("target_exit", ""),
                    status=PedestrianStatus(p_data.get("status", "active")),
                ))
        else:
            # Default placement grid
            for i in range(scenario.agent_count):
                engine.add_pedestrian(Pedestrian(
                    id=f"agent_{i + 1:03d}",
                    position=(5.0 + (i % 8) * 2.0, 5.0 + (i // 8) * 2.0),
                    velocity=(1.0, 0.0),
                ))
        return engine
