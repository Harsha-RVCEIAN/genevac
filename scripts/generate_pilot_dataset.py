#!/usr/bin/env python3
"""Generate a pilot simulation dataset from multiple GenEvac scenarios & layouts.

This script:
1. Creates a fresh, isolated output directory (no stale-output contamination).
2. Defines 3 distinct scenarios across 3 distinct layouts (3 runs each = 9 runs).
3. Runs simulations and records every step, failing loudly on any error.
4. Validates recorded data and writes a quality report.
5. Pre-processes history/future windows.
6. Generates both run-level and layout-level splits with non-empty train/val/test
   and verifies zero run/window/layout leakage.
7. Generates dataset statistics (including evacuation & termination outcomes).

Usage::

    python scripts/generate_pilot_dataset.py
    python scripts/generate_pilot_dataset.py --output-dir data/simulation/pilots/custom --clean
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genevac.data.simulation.paths import SIMULATION_ROOT
from genevac.experiments.scenario import ScenarioDefinition
from genevac.prediction.simulation_batch import BatchGenerator
from genevac.prediction.simulation_preprocess import SimulationWindow, generate_windows
from genevac.prediction.simulation_split import (
    SPLIT_NAMES,
    check_leakage,
    split_by_layout,
    split_run_ids,
    write_split_manifest,
)
from genevac.prediction.simulation_stats import generate_stats
from genevac.prediction.simulation_validator import validate_dataset, write_quality_report


PILOT_MARKER = ".genevac_pilot_output"


def build_pilot_scenarios() -> List[ScenarioDefinition]:
    """Return 3 distinct scenarios across 3 distinct layouts."""
    return [
        ScenarioDefinition(
            scenario_id="pilot_hall_evac_01",
            name="Pilot: Hall A single-exit evacuation",
            description="20x10m hall with 1 exit and 8 pedestrians",
            environment_id="pilot_hall_A",
            agent_count=8,
            seed=42,
            engine_config={
                "width": 20.0,
                "height": 10.0,
                "max_steps": 40,
                "timestep": 0.1,
                "exits": [{"id": "exit_A", "position": [19.5, 5.0], "width": 2.0}],
                "obstacles": [],
                "pedestrians": [
                    {"id": f"hall_p{i+1}", "position": [2.0 + i * 1.5, 5.0], "velocity": [1.0, 0.0]}
                    for i in range(8)
                ],
            },
        ),
        ScenarioDefinition(
            scenario_id="pilot_corridor_evac_02",
            name="Pilot: Corridor B obstacle evacuation",
            description="30x8m corridor with 1 obstacle, 1 exit, and 6 pedestrians",
            environment_id="pilot_corridor_B",
            agent_count=6,
            seed=100,
            engine_config={
                "width": 30.0,
                "height": 8.0,
                "max_steps": 40,
                "timestep": 0.1,
                "exits": [{"id": "exit_B", "position": [29.5, 4.0], "width": 2.0}],
                "obstacles": [{"id": "pillar_1", "bounds": [12.0, 3.0, 14.0, 5.0]}],
                "pedestrians": [
                    {"id": f"corr_p{i+1}", "position": [2.0 + i * 1.4, 2.0 + (i % 2) * 4.0], "velocity": [1.2, 0.0]}
                    for i in range(6)
                ],
            },
        ),
        ScenarioDefinition(
            scenario_id="pilot_atrium_evac_03",
            name="Pilot: Atrium C dual-exit evacuation",
            description="25x15m atrium with 2 exits, 1 obstacle, and 8 pedestrians",
            environment_id="pilot_atrium_C",
            agent_count=8,
            seed=200,
            engine_config={
                "width": 25.0,
                "height": 15.0,
                "max_steps": 40,
                "timestep": 0.1,
                "exits": [
                    {"id": "exit_C1", "position": [24.5, 4.0], "width": 2.0},
                    {"id": "exit_C2", "position": [24.5, 11.0], "width": 2.0},
                ],
                "obstacles": [{"id": "kiosk_1", "bounds": [10.0, 6.0, 13.0, 9.0]}],
                "pedestrians": [
                    {"id": f"atr_p{i+1}", "position": [2.0 + (i % 4) * 1.8, 3.0 + (i // 4) * 8.0], "velocity": [1.1, 0.0]}
                    for i in range(8)
                ],
            },
        ),
    ]


def _resolve_output_dir(output_dir: str | None, clean: bool) -> Path:
    pilots_root = SIMULATION_ROOT / "pilots"
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = pilots_root / f"{stamp}_{uuid.uuid4().hex[:6]}"
    else:
        target = Path(output_dir).resolve()
        if target.exists() and clean:
            if not (target / PILOT_MARKER).is_file():
                raise RuntimeError(
                    f"Refusing to clean {target}: missing {PILOT_MARKER} safety marker"
                )
            shutil.rmtree(target)
        elif target.exists() and any(target.iterdir()):
            raise RuntimeError(
                f"Output directory {target} is not empty. Pass --clean or omit --output-dir for a fresh directory."
            )
    target.mkdir(parents=True, exist_ok=True)
    (target / PILOT_MARKER).write_text("pilot_output\n", encoding="utf-8")
    return target


def run_pilot(
    *,
    output_dir: str | None = None,
    clean: bool = False,
    runs_per_scenario: int = 3,
    base_seed: int = 42,
) -> Path:
    root = _resolve_output_dir(output_dir, clean)
    raw_dir = root / "raw" / "v1"
    processed_dir = root / "processed" / "v1"

    print("=" * 60)
    print("GenEvac Pilot Dataset Generation")
    print("=" * 60)
    print(f"Output root: {root}")

    scenarios = build_pilot_scenarios()
    print(
        f"\n[1/5] Running batch simulation ({len(scenarios)} scenarios x {runs_per_scenario} runs)..."
    )
    gen = BatchGenerator(
        output_dir=raw_dir,
        scenarios=scenarios,
        runs_per_scenario=runs_per_scenario,
        base_seed=base_seed,
    )
    summary = gen.run()
    completed = summary["completed_runs"]
    print(f"      Completed runs   : {summary['total_completed']}")
    print(f"      Interrupted runs : {summary['total_interrupted']}")
    print(f"      Failed runs      : {summary['total_failed']}")

    if summary["total_failed"] > 0 or summary["total_interrupted"] > 0 or not completed:
        print("ERROR: Batch simulation reported failed/interrupted runs or 0 completed runs.")
        sys.exit(1)

    print("\n[2/5] Validating recorded data...")
    manifests_dir = raw_dir / "manifests"
    report = validate_dataset(manifests_dir)
    qr_path = processed_dir / "quality_report.json"
    write_quality_report(report, qr_path)
    print(f"      Total records  : {report['total_records']}")
    print(f"      Valid records  : {report['valid_records']}")
    print(f"      Errors         : {len(report['errors'])}")
    print(f"      Quality report : {qr_path}")

    if report["errors"] or report["invalid_runs"] > 0 or report["invalid_records"] > 0:
        print("ERROR: Dataset validation failed.")
        sys.exit(1)

    print("\n[3/5] Pre-processing windows...")
    windows_by_run: Dict[str, List[SimulationWindow]] = {}
    run_to_layout: Dict[str, str] = {}
    layout_to_runs: Dict[str, List[str]] = {}
    total_windows = 0

    for run_id in completed:
        mpath = manifests_dir / f"{run_id}_manifest.json"
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        layout_id = manifest["layout_id"]
        run_to_layout[run_id] = layout_id
        layout_to_runs.setdefault(layout_id, []).append(run_id)
        traj_file = Path(manifest["trajectory_file"])
        wins = generate_windows(
            traj_file,
            run_id=run_id,
            scenario_id=manifest["scenario_id"],
            layout_id=layout_id,
        )
        windows_by_run[run_id] = wins
        total_windows += len(wins)
    print(f"      Total windows generated: {total_windows}")
    if total_windows == 0:
        print("ERROR: No preprocessing windows generated.")
        sys.exit(1)

    print("\n[4/5] Splitting runs (run-level and layout-level) & checking leakage...")
    run_assignments = split_run_ids(completed, seed=base_seed)
    run_windows_by_split: Dict[str, List[SimulationWindow]] = {s: [] for s in SPLIT_NAMES}
    for rid, sp in run_assignments.items():
        run_windows_by_split[sp].extend(windows_by_run[rid])
    run_leakage = check_leakage(run_assignments, windows_by_split=run_windows_by_split)
    if run_leakage:
        print(f"ERROR: Run-level split validation failed: {run_leakage}")
        sys.exit(1)
    write_split_manifest(run_assignments, processed_dir / "splits" / "run_split.json", seed=base_seed)

    layout_assignments = split_by_layout(layout_to_runs, seed=base_seed)
    layout_windows_by_split: Dict[str, List[SimulationWindow]] = {s: [] for s in SPLIT_NAMES}
    for rid, sp in layout_assignments.items():
        layout_windows_by_split[sp].extend(windows_by_run[rid])
    layout_leakage = check_leakage(
        layout_assignments,
        run_to_layout=run_to_layout,
        windows_by_split=layout_windows_by_split,
    )
    if layout_leakage:
        print(f"ERROR: Layout-level split validation failed: {layout_leakage}")
        sys.exit(1)
    write_split_manifest(
        layout_assignments, processed_dir / "splits" / "layout_split.json", seed=base_seed
    )

    for label, assign in (("Run-level", run_assignments), ("Layout-level", layout_assignments)):
        counts = {s: sum(1 for v in assign.values() if v == s) for s in SPLIT_NAMES}
        print(f"      {label:12s}: {counts} (no leakage)")

    print("\n[5/5] Generating statistics...")
    stats = generate_stats(
        manifests_dir,
        output_json=processed_dir / "reports" / "stats.json",
        output_csv=processed_dir / "reports" / "stats.csv",
    )
    print(f"      Scenarios          : {stats['scenarios']}")
    print(f"      Layouts            : {stats['layouts']}")
    print(f"      Total pedestrians  : {stats['total_pedestrians']}")
    print(f"      Total observations : {stats['total_observations']}")
    print(f"      Total evacuated    : {stats['total_evacuated']}")
    print(f"      Termination reasons: {stats['termination_reason_distribution']}")

    print("\n" + "=" * 60)
    print("Pilot dataset generation complete.")
    print(f"  Raw data  : {raw_dir}")
    print(f"  Processed : {processed_dir}")
    print("=" * 60)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GenEvac pilot simulation dataset.")
    parser.add_argument("--output-dir", default=None, help="Explicit output directory.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Safely clean --output-dir if it carries the pilot safety marker.",
    )
    args = parser.parse_args()
    try:
        run_pilot(output_dir=args.output_dir, clean=args.clean)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
