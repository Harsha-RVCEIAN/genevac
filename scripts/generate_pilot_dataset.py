#!/usr/bin/env python3
"""Generate a small pilot dataset from a default GenEvac scenario.

This script:
1. Defines a minimal evacuation scenario.
2. Runs a simulation and records every step.
3. Validates the recorded data.
4. Pre-processes windows.
5. Splits into train/val/test.
6. Generates statistics.

Usage::

    python scripts/generate_pilot_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project src is on PYTHONPATH when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genevac.data.simulation.paths import RAW_V1, PROCESSED_V1
from genevac.environment import EnvironmentMap, Exit, Obstacle
from genevac.experiments.scenario import ScenarioDefinition
from genevac.pedestrians import Pedestrian, PedestrianStatus
from genevac.prediction.simulation_batch import BatchGenerator
from genevac.prediction.simulation_preprocess import generate_windows
from genevac.prediction.simulation_split import (
    split_run_ids,
    check_leakage,
    write_split_manifest,
)
from genevac.prediction.simulation_stats import generate_stats
from genevac.prediction.simulation_validator import validate_dataset, write_quality_report
from genevac.simulation import SimulationEngine


def main() -> None:
    print("=" * 60)
    print("GenEvac Pilot Dataset Generation")
    print("=" * 60)

    # -------------------------------------------------------------------
    # 1. Define a minimal scenario
    # -------------------------------------------------------------------
    scenario = ScenarioDefinition(
        scenario_id="pilot_evac_01",
        name="Pilot: single-room evacuation",
        description="Small 20×10m room with 1 exit and 10 pedestrians",
        environment_id="pilot_hall_A",
        agent_count=10,
        seed=42,
        engine_config={
            "width": 20.0,
            "height": 10.0,
            "max_steps": 150,
            "timestep": 0.1,
            "exits": [{"id": "exit_A", "position": [19.5, 5.0], "width": 2.0}],
            "obstacles": [],
            "pedestrians": [
                {"id": f"p{i+1}", "position": [2.0 + i * 1.5, 5.0],
                 "velocity": [1.0, 0.0]}
                for i in range(10)
            ],
        },
    )

    # -------------------------------------------------------------------
    # 2. Run & record
    # -------------------------------------------------------------------
    print("\n[1/5] Running batch simulation (1 scenario × 3 runs)…")
    gen = BatchGenerator(
        output_dir=RAW_V1,
        scenarios=[scenario],
        runs_per_scenario=3,
        base_seed=42,
    )
    summary = gen.run()
    completed = summary["completed_runs"]
    print(f"      Completed runs : {len(completed)}")
    print(f"      Failed runs    : {len(summary['failed_runs'])}")

    if not completed:
        print("ERROR: no runs completed. Aborting.")
        sys.exit(1)

    # -------------------------------------------------------------------
    # 3. Validate
    # -------------------------------------------------------------------
    print("\n[2/5] Validating recorded data…")
    report = validate_dataset(RAW_V1 / "manifests")
    qr_path = PROCESSED_V1 / "quality_report.json"
    write_quality_report(report, qr_path)
    print(f"      Total records  : {report['total_records']}")
    print(f"      Valid records  : {report['valid_records']}")
    print(f"      Errors         : {len(report['errors'])}")
    print(f"      Quality report : {qr_path}")

    # -------------------------------------------------------------------
    # 4. Pre-process windows from first completed run
    # -------------------------------------------------------------------
    print("\n[3/5] Pre-processing windows…")
    manifests_dir = RAW_V1 / "manifests"
    windows_dir = PROCESSED_V1 / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    total_windows = 0
    for run_id in completed:
        mpath = manifests_dir / f"{run_id}_manifest.json"
        if not mpath.is_file():
            continue
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        traj_file = Path(manifest["trajectory_file"])
        if not traj_file.is_file():
            continue
        wins = generate_windows(
            traj_file,
            run_id=run_id,
            scenario_id=manifest.get("scenario_id", "unknown"),
            layout_id=manifest.get("layout_id", "unknown"),
        )
        total_windows += len(wins)
    print(f"      Total windows generated: {total_windows}")

    # -------------------------------------------------------------------
    # 5. Split + leakage check
    # -------------------------------------------------------------------
    print("\n[4/5] Splitting runs…")
    assignments = split_run_ids(completed, seed=42)
    leakage_errors = check_leakage(assignments)
    split_path = PROCESSED_V1 / "splits" / "run_split.json"
    write_split_manifest(assignments, split_path)
    print(f"      Assignments: {dict.fromkeys(assignments.values(), 0)}")
    for split in ("train", "val", "test"):
        count = sum(1 for v in assignments.values() if v == split)
        print(f"        {split:5s}: {count} runs")
    if leakage_errors:
        print(f"      LEAKAGE DETECTED: {leakage_errors}")
    else:
        print("      No leakage detected ✓")

    # -------------------------------------------------------------------
    # 6. Statistics
    # -------------------------------------------------------------------
    print("\n[5/5] Generating statistics…")
    stats = generate_stats(
        manifests_dir,
        output_json=PROCESSED_V1 / "reports" / "stats.json",
        output_csv=PROCESSED_V1 / "reports" / "stats.csv",
    )
    print(f"      Total pedestrians  : {stats['total_pedestrians']}")
    print(f"      Total observations : {stats['total_observations']}")
    print(f"      Mean traj length   : {stats['trajectory_length']['mean']}")

    print("\n" + "=" * 60)
    print("Pilot dataset generation complete.")
    print(f"  Raw data  : {RAW_V1}")
    print(f"  Processed : {PROCESSED_V1}")
    print("=" * 60)


if __name__ == "__main__":
    main()
