"""Batch simulation generation workflow for GenEvac-Sentinel.

Runs multiple simulation scenarios using the authoritative
``genevac.simulation.engine.SimulationEngine`` (with exit-crossing detection
and evacuation status transitions), records each run with
``SimulationRecorder``, validates it, and writes a master batch manifest.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from genevac.data.simulation.schema import generate_run_id, iso_timestamp
from genevac.environment.geometry import (
    LineSegment,
    Point,
    Rectangle,
    Wall,
    WallAlignment,
    WallType,
)
from genevac.environment.models import Building, Exit, Obstacle, WalkableArea
from genevac.experiments.scenario import ScenarioDefinition
from genevac.pedestrian.models import PedestrianState, PedestrianStatus
from genevac.pedestrian.registry import PedestrianRegistry
from genevac.prediction.simulation_recorder import record_full_run
from genevac.prediction.simulation_validator import validate_dataset
from genevac.simulation.engine import SimulationEngine


def _build_boundary_walls(
    width: float,
    height: float,
    exits: Sequence[Exit],
    thickness: float = 0.2,
) -> List[Wall]:
    """Build exterior walls on all four boundaries with gaps for exits."""
    walls: List[Wall] = []

    def _split_segments(
        total_len: float,
        gaps: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        ordered = sorted(gaps)
        segs: List[Tuple[float, float]] = []
        cursor = 0.0
        for g_start, g_end in ordered:
            g_start = max(0.0, g_start)
            g_end = min(total_len, g_end)
            if g_start > cursor + 1e-6:
                segs.append((cursor, g_start))
            cursor = max(cursor, g_end)
        if cursor < total_len - 1e-6:
            segs.append((cursor, total_len))
        return segs

    by_side: Dict[str, List[Tuple[float, float]]] = {
        "top": [],
        "bottom": [],
        "left": [],
        "right": [],
    }
    for ex in exits:
        if ex.boundary in ("top", "bottom"):
            a, b = sorted((ex.segment.start.x, ex.segment.end.x))
            by_side[ex.boundary].append((a, b))
        elif ex.boundary in ("left", "right"):
            a, b = sorted((ex.segment.start.y, ex.segment.end.y))
            by_side[ex.boundary].append((a, b))

    for idx, (x0, x1) in enumerate(_split_segments(width, by_side["top"])):
        walls.append(
            Wall(
                id=f"wall_ext_top_{idx}",
                start=Point(x0, 0.0),
                end=Point(x1, 0.0),
                thickness=thickness,
                wall_type=WallType.EXTERIOR,
                alignment=WallAlignment.POSITIVE,
            )
        )
    for idx, (x0, x1) in enumerate(_split_segments(width, by_side["bottom"])):
        walls.append(
            Wall(
                id=f"wall_ext_bottom_{idx}",
                start=Point(x0, height),
                end=Point(x1, height),
                thickness=thickness,
                wall_type=WallType.EXTERIOR,
                alignment=WallAlignment.NEGATIVE,
            )
        )
    for idx, (y0, y1) in enumerate(_split_segments(height, by_side["left"])):
        walls.append(
            Wall(
                id=f"wall_ext_left_{idx}",
                start=Point(0.0, y0),
                end=Point(0.0, y1),
                thickness=thickness,
                wall_type=WallType.EXTERIOR,
                alignment=WallAlignment.POSITIVE,
            )
        )
    for idx, (y0, y1) in enumerate(_split_segments(height, by_side["right"])):
        walls.append(
            Wall(
                id=f"wall_ext_right_{idx}",
                start=Point(width, y0),
                end=Point(width, y1),
                thickness=thickness,
                wall_type=WallType.EXTERIOR,
                alignment=WallAlignment.NEGATIVE,
            )
        )
    return walls


def _convert_exit(ex_cfg: Dict[str, Any], width: float, height: float) -> Exit:
    ex_id = str(ex_cfg["id"])
    pos = ex_cfg["position"]
    ex_x, ex_y = float(pos[0]), float(pos[1])
    ex_w = float(ex_cfg.get("width", 2.0))
    half = ex_w / 2.0

    d_left = abs(ex_x)
    d_right = abs(width - ex_x)
    d_top = abs(ex_y)
    d_bottom = abs(height - ex_y)
    nearest = min(
        (d_left, "left"),
        (d_right, "right"),
        (d_top, "top"),
        (d_bottom, "bottom"),
        key=lambda item: item[0],
    )[1]

    if nearest == "right":
        y0 = max(0.0, min(height - ex_w, ex_y - half))
        seg = LineSegment(Point(width, y0), Point(width, y0 + ex_w))
    elif nearest == "left":
        y0 = max(0.0, min(height - ex_w, ex_y - half))
        seg = LineSegment(Point(0.0, y0), Point(0.0, y0 + ex_w))
    elif nearest == "bottom":
        x0 = max(0.0, min(width - ex_w, ex_x - half))
        seg = LineSegment(Point(x0, height), Point(x0 + ex_w, height))
    else:
        x0 = max(0.0, min(width - ex_w, ex_x - half))
        seg = LineSegment(Point(x0, 0.0), Point(x0 + ex_w, 0.0))

    return Exit(
        id=ex_id,
        segment=seg,
        width=ex_w,
        boundary=nearest,
        connected_area_id="main_area",
    )


class BatchGenerator:
    """Generate, record, and validate multiple simulation runs."""

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
        self._events_dir = self.output_dir / "events"
        self._traj_dir.mkdir(parents=True, exist_ok=True)
        self._manifests_dir.mkdir(parents=True, exist_ok=True)
        self._events_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Dict[str, Any]:
        """Execute all runs and return a batch summary dict."""
        completed: List[str] = []
        interrupted: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        global_index = 0

        for scenario in self.scenarios:
            for _ in range(self.runs_per_scenario):
                seed = self.base_seed + global_index
                global_index += 1
                run_id = generate_run_id()
                max_steps = int((scenario.engine_config or {}).get("max_steps", 200))
                try:
                    engine = self._build_engine(scenario, seed)
                    metadata, _ = record_full_run(
                        engine,
                        scenario_id=scenario.scenario_id,
                        layout_id=scenario.environment_id,
                        seed=seed,
                        run_id=run_id,
                        trajectories_dir=self._traj_dir,
                        manifests_dir=self._manifests_dir,
                        events_dir=self._events_dir,
                        max_steps=max_steps,
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

        master_path = self._manifests_dir / "batch_manifest.json"
        master_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _build_engine(self, scenario: ScenarioDefinition, seed: int) -> SimulationEngine:
        """Build the authoritative ``SimulationEngine`` from a ``ScenarioDefinition``."""
        cfg = scenario.engine_config or {}
        width = float(cfg.get("width", 50.0))
        height = float(cfg.get("height", 20.0))
        timestep = float(cfg.get("timestep", 0.1))
        obs_interval = float(cfg.get("observation_interval", timestep))

        exits_cfg = cfg.get("exits", [])
        if not exits_cfg:
            exits_cfg = [{"id": "exit_default", "position": [width, height / 2.0], "width": 2.0}]
        exits = [_convert_exit(ex, width, height) for ex in exits_cfg]

        obstacles: List[Obstacle] = []
        for obs in cfg.get("obstacles", []):
            min_x, min_y, max_x, max_y = map(float, obs["bounds"])
            obstacles.append(
                Obstacle(
                    id=str(obs["id"]),
                    bounds=Rectangle(min_x, min_y, max_x - min_x, max_y - min_y),
                )
            )

        walkable_areas = [
            WalkableArea(
                id="main_area",
                bounds=Rectangle(0.0, 0.0, width, height),
                connected_area_ids=[],
                area_type="room",
            )
        ]
        walls = _build_boundary_walls(width, height, exits)
        building = Building(
            id=scenario.environment_id,
            width=width,
            height=height,
            walkable_areas=walkable_areas,
            walls=walls,
            obstacles=obstacles,
            exits=exits,
        )

        default_exit_id = exits[0].id
        registry = PedestrianRegistry()
        peds_cfg = cfg.get("pedestrians", [])
        if peds_cfg:
            for p_data in peds_cfg:
                pos = p_data["position"]
                target_exit = p_data.get("target_exit") or default_exit_id
                registry.register(
                    PedestrianState(
                        id=str(p_data["id"]),
                        position=Point(float(pos[0]), float(pos[1])),
                        status=PedestrianStatus(p_data.get("status", "active")),
                        target_exit_id=target_exit,
                        current_area_id="main_area",
                        radius=float(p_data.get("radius", 0.25)),
                        preferred_speed=float(p_data.get("desired_speed", 1.34)),
                    )
                )
        else:
            for i in range(scenario.agent_count):
                registry.register(
                    PedestrianState(
                        id=f"agent_{i + 1:03d}",
                        position=Point(3.0 + (i % 6) * 2.0, 2.0 + (i // 6) * 2.0),
                        status=PedestrianStatus.ACTIVE,
                        target_exit_id=default_exit_id,
                        current_area_id="main_area",
                    )
                )

        return SimulationEngine(
            building=building,
            registry=registry,
            dt=timestep,
            observation_interval=obs_interval,
        )
