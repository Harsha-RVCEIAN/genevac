"""Dataset statistics and coverage reports for GenEvac simulation recordings.

Generates summary statistics from a directory of validated manifests, without
fabricating unavailable coverage information.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from genevac.data.simulation.manifest import list_manifests, load_manifest
from genevac.prediction.simulation_preprocess import load_run_records, filter_valid_records, group_by_pedestrian


def generate_stats(
    manifests_dir: Path,
    *,
    output_json: Optional[Path] = None,
    output_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compute dataset statistics from all manifests in *manifests_dir*.

    Parameters
    ----------
    manifests_dir:
        Directory containing ``*_manifest.json`` files.
    output_json:
        If provided, write the report to this JSON file.
    output_csv:
        If provided, write a CSV summary to this path.

    Returns
    -------
    dict
        Statistics report.
    """
    manifests = list_manifests(manifests_dir)
    if not manifests:
        return {"error": "No manifests found", "total_runs": 0}

    total_runs = len(manifests)
    scenario_ids: set = set()
    layout_ids: set = set()
    total_peds = 0
    total_obs = 0
    traj_lengths: List[int] = []
    populations: List[int] = []
    evacuation_counts: List[int] = []
    status_counter: Counter = Counter()
    termination_counter: Counter = Counter()
    run_errors: List[str] = []

    for mpath in manifests:
        try:
            manifest = load_manifest(mpath)
        except Exception as exc:
            run_errors.append(f"{mpath.name}: {exc}")
            continue

        scenario_ids.add(manifest.get("scenario_id", "UNKNOWN"))
        layout_ids.add(manifest.get("layout_id", "UNKNOWN"))
        status_counter[manifest.get("status", "unknown")] += 1
        termination_counter[manifest.get("termination_reason") or "unknown"] += 1
        populations.append(int(manifest.get("population", 0)))
        evacuation_counts.append(int(manifest.get("evacuated", 0)))

        traj_file_str = manifest.get("trajectory_file")
        if traj_file_str and Path(traj_file_str).is_file():
            records = filter_valid_records(load_run_records(Path(traj_file_str)))
            groups = group_by_pedestrian(records)
            total_peds += len(groups)
            for obs_list in groups.values():
                n = len(obs_list)
                traj_lengths.append(n)
                total_obs += n

    def _safe_stats(vals: List[int]):
        if not vals:
            return {"min": None, "max": None, "mean": None, "median": None}
        return {
            "min": min(vals),
            "max": max(vals),
            "mean": round(statistics.mean(vals), 3),
            "median": round(statistics.median(vals), 3),
        }

    report: Dict[str, Any] = {
        "schema_version": "v1",
        "total_runs": total_runs,
        "scenarios": sorted(scenario_ids),
        "layouts": sorted(layout_ids),
        "total_pedestrians": total_peds,
        "total_observations": total_obs,
        "trajectory_length": _safe_stats(traj_lengths),
        "population": _safe_stats(populations),
        "evacuation_counts": _safe_stats(evacuation_counts),
        "total_evacuated": sum(evacuation_counts),
        "run_status_distribution": dict(status_counter),
        "termination_reason_distribution": dict(termination_counter),
        "load_errors": run_errors,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            ["metric", "value"],
            ["total_runs", str(total_runs)],
            ["total_pedestrians", str(total_peds)],
            ["total_observations", str(total_obs)],
            ["mean_trajectory_length", str(report["trajectory_length"]["mean"])],
            ["median_trajectory_length", str(report["trajectory_length"]["median"])],
            ["mean_population", str(report["population"]["mean"])],
        ]
        output_csv.write_text(
            "\n".join(",".join(r) for r in rows), encoding="utf-8"
        )

    return report
