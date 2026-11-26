"""Simulation trajectory recorder for GenEvac-Sentinel.

Records pedestrian observations from a ``SimulationEngine`` run into a
versioned JSONL file, together with a JSON run manifest and optional
authoritative simulation events JSONL.

Supports both:
- The authoritative ``genevac.simulation.engine.SimulationEngine`` (``clock``,
  ``registry``, ``building``, ``events``).
- The legacy ``genevac.simulation.SimulationEngine`` (``current_step``,
  ``pedestrians``, ``environment``, ``is_finished()``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from genevac.data.simulation.manifest import write_manifest
from genevac.data.simulation.paths import EVENTS_ROOT, MANIFESTS_ROOT, TRAJECTORIES_ROOT
from genevac.data.simulation.schema import (
    RunMetadata,
    TrajectoryRecord,
    generate_run_id,
    iso_timestamp,
)


def _extract_xy(pos: Any) -> Optional[Tuple[float, float]]:
    if pos is None:
        return None
    if hasattr(pos, "x") and hasattr(pos, "y"):
        return float(pos.x), float(pos.y)
    if isinstance(pos, (tuple, list)) and len(pos) >= 2:
        return float(pos[0]), float(pos[1])
    return None


def _extract_vx_vy(vel: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if vel is None:
        return None, None, None
    if hasattr(vel, "vx") and hasattr(vel, "vy"):
        vx, vy = float(vel.vx), float(vel.vy)
    elif isinstance(vel, (tuple, list)) and len(vel) >= 2:
        vx, vy = float(vel[0]), float(vel[1])
    else:
        return None, None, None
    if math.isfinite(vx) and math.isfinite(vy):
        return vx, vy, round(math.hypot(vx, vy), 6)
    return None, None, None


def _extract_status(ped: Any) -> Optional[str]:
    raw_status = getattr(ped, "status", None)
    if raw_status is None:
        return None
    if hasattr(raw_status, "value"):
        return str(raw_status.value)
    if isinstance(raw_status, str):
        return raw_status
    return None


def _iter_pedestrians(engine: Any) -> List[Any]:
    registry = getattr(engine, "registry", None)
    if registry is not None:
        return list(registry)
    return list(getattr(engine, "pedestrians", []))


def _engine_step_and_time(engine: Any) -> Tuple[int, float, float]:
    clock = getattr(engine, "clock", None)
    if clock is not None:
        return (
            int(clock.step_count),
            float(clock.dt),
            round(float(clock.timestamp), 6),
        )
    step_index = int(getattr(engine, "current_step", 0))
    timestep = float(getattr(engine, "timestep", 0.1))
    return step_index, timestep, round(step_index * timestep, 6)


class SimulationRecorder:
    """Record pedestrian observations from one ``SimulationEngine`` run."""

    def __init__(
        self,
        scenario_id: str,
        layout_id: str,
        seed: int,
        *,
        run_id: Optional[str] = None,
        trajectories_dir: Optional[Path] = None,
        manifests_dir: Optional[Path] = None,
        events_dir: Optional[Path] = None,
    ) -> None:
        self.scenario_id = scenario_id
        self.layout_id = layout_id
        self.seed = seed
        self.run_id: str = run_id or generate_run_id()

        self._trajectories_dir = Path(trajectories_dir) if trajectories_dir else TRAJECTORIES_ROOT
        self._manifests_dir = Path(manifests_dir) if manifests_dir else MANIFESTS_ROOT
        self._events_dir = Path(events_dir) if events_dir else EVENTS_ROOT
        self._trajectories_dir.mkdir(parents=True, exist_ok=True)
        self._manifests_dir.mkdir(parents=True, exist_ok=True)

        self._traj_path: Path = self._trajectories_dir / f"{self.run_id}.jsonl"
        self._events_path: Path = self._events_dir / f"{self.run_id}.jsonl"
        self._traj_file = None

        self._seen: Set[Tuple[str, int]] = set()
        self._start_time: str = ""
        self._records_written: int = 0

    def begin(self) -> None:
        """Open the JSONL trajectory file and record the start timestamp."""
        self._start_time = iso_timestamp()
        self._traj_file = self._traj_path.open("w", encoding="utf-8")

    def record_step(self, engine: Any) -> int:
        """Capture one simulation step from *engine*."""
        if self._traj_file is None:
            raise RuntimeError("Call begin() before record_step().")

        step_index, _, timestamp = _engine_step_and_time(engine)
        written = 0
        for ped in _iter_pedestrians(engine):
            ped_id = str(getattr(ped, "id", "unknown"))
            key = (ped_id, step_index)
            if key in self._seen:
                continue
            self._seen.add(key)

            xy = _extract_xy(getattr(ped, "position", None))
            if xy is None:
                continue
            x, y = xy
            if not (math.isfinite(x) and math.isfinite(y)):
                continue

            vx, vy, speed = _extract_vx_vy(getattr(ped, "velocity", None))
            status_str = _extract_status(ped)

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
        max_steps: Optional[int] = None,
    ) -> RunMetadata:
        """Close files and write the run manifest (plus events if exposed)."""
        if self._traj_file is not None:
            self._traj_file.flush()
            self._traj_file.close()
            self._traj_file = None

        end_time = iso_timestamp()
        total_steps, timestep, _ = _engine_step_and_time(engine)
        peds = _iter_pedestrians(engine)
        population = len(peds)
        evacuated = sum(1 for p in peds if _extract_status(p) == "evacuated")
        active_remaining = sum(1 for p in peds if _extract_status(p) == "active")

        env = getattr(engine, "building", None) or getattr(engine, "environment", None)
        env_w = float(getattr(env, "width", 0.0)) if env else 0.0
        env_h = float(getattr(env, "height", 0.0)) if env else 0.0
        resolved_max_steps = int(max_steps) if max_steps is not None else int(getattr(engine, "max_steps", 0))

        if status == "failed":
            termination_reason = "error"
        elif status == "interrupted":
            termination_reason = "interrupted"
        elif population > 0 and active_remaining == 0:
            termination_reason = "evacuation_completed"
        elif resolved_max_steps > 0 and total_steps >= resolved_max_steps:
            termination_reason = "max_steps_reached"
        else:
            termination_reason = "completed"

        events_file: Optional[Path] = None
        engine_events = getattr(engine, "events", None)
        if engine_events:
            self._events_dir.mkdir(parents=True, exist_ok=True)
            with self._events_path.open("w", encoding="utf-8") as ef:
                for evt in engine_events:
                    payload = {
                        "run_id": self.run_id,
                        "event_id": getattr(evt, "event_id", None),
                        "event_type": getattr(getattr(evt, "event_type", None), "value", str(getattr(evt, "event_type", ""))),
                        "step": getattr(evt, "step", None),
                        "timestamp": getattr(evt, "timestamp", None),
                        "pedestrian_id": getattr(evt, "pedestrian_id", None),
                        "details": getattr(evt, "details", {}),
                    }
                    ef.write(json.dumps(payload, separators=(",", ":")) + "\n")
            events_file = self._events_path

        metadata = RunMetadata(
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            layout_id=self.layout_id,
            seed=self.seed,
            population=population,
            environment_width=env_w,
            environment_height=env_h,
            timestep=timestep,
            max_steps=resolved_max_steps,
            total_steps=total_steps,
            evacuated=evacuated,
            start_time=self._start_time,
            end_time=end_time,
            status=status,
            failure_reason=failure_reason,
            termination_reason=termination_reason,
        )

        write_manifest(
            metadata=metadata,
            trajectory_file=self._traj_path,
            events_file=events_file,
            dest_dir=self._manifests_dir,
        )
        return metadata

    @property
    def records_written(self) -> int:
        return self._records_written

    @property
    def trajectory_path(self) -> Path:
        return self._traj_path


def _engine_finished(engine: Any, max_steps: Optional[int]) -> bool:
    if hasattr(engine, "is_finished"):
        return bool(engine.is_finished())
    step_count, _, _ = _engine_step_and_time(engine)
    if max_steps is not None and step_count >= max_steps:
        return True
    peds = _iter_pedestrians(engine)
    if not peds:
        return True
    return all(_extract_status(p) != "active" for p in peds)


def record_full_run(
    engine: Any,
    *,
    scenario_id: str,
    layout_id: str,
    seed: int,
    run_id: Optional[str] = None,
    trajectories_dir: Optional[Path] = None,
    manifests_dir: Optional[Path] = None,
    events_dir: Optional[Path] = None,
    max_steps: Optional[int] = None,
) -> Tuple[RunMetadata, Path]:
    """Run *engine* to completion and record every step."""
    recorder = SimulationRecorder(
        scenario_id=scenario_id,
        layout_id=layout_id,
        seed=seed,
        run_id=run_id,
        trajectories_dir=trajectories_dir,
        manifests_dir=manifests_dir,
        events_dir=events_dir,
    )
    recorder.begin()
    status = "completed"
    failure_reason = None
    is_authoritative = not hasattr(engine, "is_finished")
    try:
        if is_authoritative:
            recorder.record_step(engine)
        while not _engine_finished(engine, max_steps):
            engine.step()
            recorder.record_step(engine)
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        failure_reason = str(exc)
    finally:
        metadata = recorder.finish(
            engine,
            status=status,
            failure_reason=failure_reason,
            max_steps=max_steps,
        )

    return metadata, recorder.trajectory_path
