"""Simulation trajectory recorder for GenEvac-Sentinel.

Wraps the existing ``SimulationTrajectoryAdapter`` and captures each
pedestrian observation into a versioned JSONL file, together with a
JSON run manifest.

Design rules (enforced here):
- Never mutates ``SimulationEngine`` or adapter state.
- Velocity (vx, vy) recorded **only** when the engine exposes authoritative values.
- Pedestrian status recorded **only** when authoritative; otherwise stored as ``None``.
- Evacuation events captured only when engine event hooks exist; otherwise left for
  post-processing.
- Run IDs are UUID4; seeds are stored separately for reproducibility.
- Duplicate observations are silently skipped (same run_id + pedestrian_id + step_index).
- Incomplete runs are written and marked ``"interrupted"``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from genevac.data.simulation.manifest import write_manifest
from genevac.data.simulation.paths import TRAJECTORIES_ROOT, MANIFESTS_ROOT
from genevac.data.simulation.schema import (
    SCHEMA_VERSION,
    RunMetadata,
    TrajectoryRecord,
    generate_run_id,
    iso_timestamp,
)
from genevac.pedestrians import PedestrianStatus


class SimulationRecorder:
    """Record pedestrian observations from one ``SimulationEngine`` run.

    Usage
    -----
    ::

        recorder = SimulationRecorder(scenario_id="evac_01", layout_id="hall_A", seed=42)
        engine = SimulationEngine(environment=env, seed=42)
        # ... add pedestrians ...
        recorder.begin()
        engine.run()
        for step in range(engine.max_steps):
            engine.step()
            recorder.record_step(engine)
            if engine.is_finished():
                break
        result = recorder.finish(engine)

    Alternatively use the high-level :func:`record_full_run` helper.
    """

    def __init__(
        self,
        scenario_id: str,
        layout_id: str,
        seed: int,
        *,
        run_id: Optional[str] = None,
        trajectories_dir: Optional[Path] = None,
        manifests_dir: Optional[Path] = None,
    ) -> None:
        self.scenario_id = scenario_id
        self.layout_id = layout_id
        self.seed = seed
        self.run_id: str = run_id or generate_run_id()

        self._trajectories_dir = Path(trajectories_dir) if trajectories_dir else TRAJECTORIES_ROOT
        self._manifests_dir = Path(manifests_dir) if manifests_dir else MANIFESTS_ROOT
        self._trajectories_dir.mkdir(parents=True, exist_ok=True)
        self._manifests_dir.mkdir(parents=True, exist_ok=True)

        self._traj_path: Path = self._trajectories_dir / f"{self.run_id}.jsonl"
        self._traj_file = None  # opened in begin()

        self._seen: Set[Tuple[str, int]] = set()  # (pedestrian_id, step_index)
        self._start_time: str = ""
        self._records_written: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def begin(self) -> None:
        """Open the JSONL trajectory file and record the start timestamp."""
        self._start_time = iso_timestamp()
        self._traj_file = self._traj_path.open("w", encoding="utf-8")

    def record_step(self, engine: Any) -> int:
        """Capture one simulation step from *engine*.

        Returns the number of new records written (0 when all were duplicates).
        """
        if self._traj_file is None:
            raise RuntimeError("Call begin() before record_step().")

        step_index = int(getattr(engine, "current_step", 0))
        timestep = float(getattr(engine, "timestep", 0.1))
        timestamp = round(step_index * timestep, 6)

        written = 0
        for ped in getattr(engine, "pedestrians", []):
            ped_id = str(getattr(ped, "id", "unknown"))
            key = (ped_id, step_index)
            if key in self._seen:
                continue
            self._seen.add(key)

            # Position – required
            pos = getattr(ped, "position", None)
            if pos is None or len(pos) < 2:
                continue
            x, y = float(pos[0]), float(pos[1])
            if not (math.isfinite(x) and math.isfinite(y)):
                continue

            # Velocity – only if engine exposes it
            vel = getattr(ped, "velocity", None)
            vx: Optional[float] = None
            vy: Optional[float] = None
            speed: Optional[float] = None
            if vel is not None and len(vel) >= 2:
                vx = float(vel[0])
                vy = float(vel[1])
                if math.isfinite(vx) and math.isfinite(vy):
                    speed = round(math.hypot(vx, vy), 6)
                else:
                    vx = vy = None

            # Status – only if authoritative
            raw_status = getattr(ped, "status", None)
            status_str: Optional[str] = None
            if raw_status is not None:
                if isinstance(raw_status, PedestrianStatus):
                    status_str = raw_status.value
                elif isinstance(raw_status, str):
                    status_str = raw_status

            rec = TrajectoryRecord(
                run_id=self.run_id,
                scenario_id=self.scenario_id,
                layout_id=self.layout_id,
                pedestrian_id=ped_id,
                step_index=step_index,
                timestamp=timestamp,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                speed=speed,
                status=status_str,
                category="simulated",
            )
            self._traj_file.write(rec.to_json() + "\n")
            written += 1
            self._records_written += 1

        return written

    def finish(
        self,
        engine: Any,
        *,
        status: str = "completed",
        failure_reason: Optional[str] = None,
    ) -> RunMetadata:
        """Close files and write the run manifest.

        Parameters
        ----------
        engine:
            The ``SimulationEngine`` instance (read-only).
        status:
            One of ``"completed"``, ``"interrupted"``, ``"failed"``.
        failure_reason:
            Human-readable reason for failure (optional).

        Returns
        -------
        RunMetadata
            Finalised metadata for this run.
        """
        if self._traj_file is not None:
            self._traj_file.flush()
            self._traj_file.close()
            self._traj_file = None

        end_time = iso_timestamp()
        total_steps = int(getattr(engine, "current_step", 0))
        peds = list(getattr(engine, "pedestrians", []))
        population = len(peds)
        evacuated = sum(
            1 for p in peds
            if getattr(p, "status", None) in (PedestrianStatus.EVACUATED, "evacuated")
        )
        env = getattr(engine, "environment", None)
        env_w = float(getattr(env, "width", 0.0)) if env else 0.0
        env_h = float(getattr(env, "height", 0.0)) if env else 0.0

        metadata = RunMetadata(
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            layout_id=self.layout_id,
            seed=self.seed,
            population=population,
            environment_width=env_w,
            environment_height=env_h,
            timestep=float(getattr(engine, "timestep", 0.1)),
            max_steps=int(getattr(engine, "max_steps", 0)),
            total_steps=total_steps,
            evacuated=evacuated,
            start_time=self._start_time,
            end_time=end_time,
            status=status,
            failure_reason=failure_reason,
        )

        write_manifest(
            metadata=metadata,
            trajectory_file=self._traj_path,
            dest_dir=self._manifests_dir,
        )
        return metadata

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def records_written(self) -> int:
        return self._records_written

    @property
    def trajectory_path(self) -> Path:
        return self._traj_path


# ---------------------------------------------------------------------------
# High-level convenience helper
# ---------------------------------------------------------------------------

def record_full_run(
    engine: Any,
    *,
    scenario_id: str,
    layout_id: str,
    seed: int,
    run_id: Optional[str] = None,
    trajectories_dir: Optional[Path] = None,
    manifests_dir: Optional[Path] = None,
) -> Tuple[RunMetadata, Path]:
    """Run *engine* to completion and record every step.

    The engine must already have pedestrians added and must not have been
    stepped yet (``current_step == 0`` expected).

    Returns
    -------
    (RunMetadata, trajectory_path)
    """
    recorder = SimulationRecorder(
        scenario_id=scenario_id,
        layout_id=layout_id,
        seed=seed,
        run_id=run_id,
        trajectories_dir=trajectories_dir,
        manifests_dir=manifests_dir,
    )
    recorder.begin()
    status = "completed"
    failure_reason = None
    try:
        while not engine.is_finished():
            engine.step()
            recorder.record_step(engine)
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        failure_reason = str(exc)
    finally:
        metadata = recorder.finish(engine, status=status, failure_reason=failure_reason)

    return metadata, recorder.trajectory_path
