"""Run manifest generation for GenEvac simulation datasets.

Each simulation run produces:
  - A JSONL trajectory file (one ``TrajectoryRecord`` per line).
  - A JSON manifest binding the run metadata to that file.

Manifests are written atomically (write to a ``.tmp`` file, then rename).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from genevac.data.simulation.schema import RunMetadata, SCHEMA_VERSION


def write_manifest(
    metadata: RunMetadata,
    trajectory_file: Path,
    events_file: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
    dest_dir: Optional[Path] = None,
) -> Path:
    """Write a JSON manifest for a completed (or interrupted) run.

    Parameters
    ----------
    metadata:
        The ``RunMetadata`` for this run.
    trajectory_file:
        Path to the JSONL file containing trajectory records.
    events_file:
        Optional path to an events JSONL file (may be ``None``).
    extra:
        Any additional key/value pairs to embed in the manifest.
    dest_dir:
        Directory in which to write the manifest.  Defaults to the
        parent directory of ``trajectory_file``.

    Returns
    -------
    Path
        The path of the written manifest file.
    """
    if dest_dir is None:
        dest_dir = trajectory_file.parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": metadata.run_id,
        "scenario_id": metadata.scenario_id,
        "layout_id": metadata.layout_id,
        "seed": metadata.seed,
        "population": metadata.population,
        "status": metadata.status,
        "total_steps": metadata.total_steps,
        "evacuated": metadata.evacuated,
        "start_time": metadata.start_time,
        "end_time": metadata.end_time,
        "trajectory_file": str(trajectory_file),
        "events_file": str(events_file) if events_file is not None else None,
        "failure_reason": metadata.failure_reason,
        "termination_reason": metadata.termination_reason,
    }
    if extra:
        manifest.update(extra)

    out_path = dest_dir / f"{metadata.run_id}_manifest.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)
    return out_path


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load and return a manifest JSON as a plain dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_manifests(manifests_dir: Path) -> list[Path]:
    """Return sorted list of all run ``*_manifest.json`` files in *manifests_dir*."""
    return sorted(
        p for p in Path(manifests_dir).glob("*_manifest.json")
        if p.name != "batch_manifest.json"
    )
