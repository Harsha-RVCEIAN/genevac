# -*- coding: utf-8 -*-
"""Simulation Recorder

Records trajectories from a running :class:`genevac.simulation.SimulationEngine`
using the :class:`genevac.prediction.adapter.SimulationTrajectoryAdapter`.
The recorder writes:
- A JSONL file with one :class:`genevac.data.simulation.TrajectoryRecord`
  per pedestrian observation.
- A run‑level metadata JSON file.
- A small manifest JSON linking the two.

All files are placed under the versioned ``data/simulation/raw/v1`` layout
defined in :mod:`genevac.data.simulation.paths`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ..prediction.adapter import SimulationTrajectoryAdapter
from .schema import RunMetadata, TrajectoryRecord, generate_run_id, iso_timestamp
from .paths import RUNS_ROOT, TRAJECTORIES_ROOT, MANIFESTS_ROOT


class SimulationRecorder:
    """Utility to record a simulation run.

    Typical usage::

        adapter = SimulationTrajectoryAdapter()
        recorder = SimulationRecorder(
            scenario_id="demo_scenario",
            layout_id="layout_a",
            seed=42,
            population=30,
            env_width=20.0,
            env_height=15.0,
            timestep=0.04,
            max_steps=5000,
        )
        # feed snapshots produced by the engine
        for snapshot in engine.run():
            for ped in snapshot.pedestrians:
                recorder.record_snapshot(
                    pedestrian_id=ped.id,
                    step_index=snapshot.step_index,
                    timestamp=snapshot.timestamp,
                    x=ped.position.x,
                    y=ped.position.y,
                    vx=getattr(ped, "velocity", None).x if getattr(ped, "velocity", None) else None,
                    vy=getattr(ped, "velocity", None).y if getattr(ped, "velocity", None) else None,
                    status=getattr(ped, "status", None),
                )
        recorder.finalize(evacuated=engine.evacuated_count)
    """

    def __init__(
        self,
        scenario_id: str,
        layout_id: str,
        seed: int,
        population: int,
        env_width: float,
        env_height: float,
        timestep: float,
        max_steps: int,
    ) -> None:
        self.run_id: str = generate_run_id()
        self.scenario_id = scenario_id
        self.layout_id = layout_id
        self.seed = seed
        self.population = population
        self.env_width = env_width
        self.env_height = env_height
        self.timestep = timestep
        self.max_steps = max_steps
        self.start_time = iso_timestamp()
        self._steps_recorded: int = 0
        self._evacuated: Optional[int] = None

        # Prepare file handles
        self._traj_path: Path = TRAJECTORIES_ROOT / f"{self.run_id}.jsonl"
        self._traj_file = open(self._traj_path, "w", encoding="utf-8")

        # Metadata will be written on finalize
        self._metadata_written = False

    # ---------------------------------------------------------------------
    # Recording API
    # ---------------------------------------------------------------------
    def record_snapshot(
        self,
        pedestrian_id: str,
        step_index: int,
        timestamp: float,
        x: float,
        y: float,
        vx: Optional[float] = None,
        vy: Optional[float] = None,
        status: Optional[str] = None,
    ) -> None:
        """Append a single observation to the trajectory JSONL file.

        The caller is responsible for providing ``vx``/``vy`` only if the
        simulation exposes authoritative velocity values. If ``None`` the
        fields will be stored as ``null`` in JSON.
        """
        speed: Optional[float] = None
        if vx is not None and vy is not None:
            speed = (vx ** 2 + vy ** 2) ** 0.5

        record = TrajectoryRecord(
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            layout_id=self.layout_id,
            pedestrian_id=pedestrian_id,
            step_index=step_index,
            timestamp=timestamp,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            speed=speed,
            status=status,
            category="simulated",
        )
        self._traj_file.write(record.to_json() + "\n")
        self._steps_recorded = max(self._steps_recorded, step_index + 1)

    # ---------------------------------------------------------------------
    # Finalisation
    # ---------------------------------------------------------------------
    def finalize(self, evacuated: int, status: str = "completed", failure_reason: Optional[str] = None) -> None:
        """Close the recorder, write metadata and a manifest.

        Parameters
        ----------
        evacuated: int
            Number of pedestrians that reached an exit.
        status: str, optional
            ``"completed"``, ``"interrupted"`` or ``"failed"``.
        failure_reason: str, optional
            Reason string when ``status`` is ``"failed"``.
        """
        # Close trajectory file first
        self._traj_file.close()

        end_time = iso_timestamp()
        metadata = RunMetadata(
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            layout_id=self.layout_id,
            seed=self.seed,
            population=self.population,
            environment_width=self.env_width,
            environment_height=self.env_height,
            timestep=self.timestep,
            max_steps=self.max_steps,
            total_steps=self._steps_recorded,
            evacuated=evacuated,
            start_time=self.start_time,
            end_time=end_time,
            status=status,
            failure_reason=failure_reason,
        )

        # Write metadata JSON
        meta_path = RUNS_ROOT / f"{self.run_id}.json"
        meta_path.write_text(metadata.to_json(), encoding="utf-8")

        # Write manifest linking the two files
        manifest = {
            "run_id": self.run_id,
            "metadata_path": str(meta_path.relative_to(meta_path.parents[3])),  # relative to project root
            "trajectory_path": str(self._traj_path.relative_to(self._traj_path.parents[3])),
            "status": status,
        }
        manifest_path = MANIFESTS_ROOT / f"{self.run_id}.json"
        manifest_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")

        self._metadata_written = True

    # ---------------------------------------------------------------------
    # Safety helpers
    # ---------------------------------------------------------------------
    def __del__(self):
        # Guard against forgetting to close the file – best‑effort cleanup.
        if not self._traj_file.closed:
            try:
                self._traj_file.close()
            except Exception:
                pass
        if not self._metadata_written:
            # Write a minimal failed metadata file so downstream tools have something.
            try:
                self.finalize(evacuated=0, status="failed", failure_reason="recorder not finalised")
            except Exception:
                pass
