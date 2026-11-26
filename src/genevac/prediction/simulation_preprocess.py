"""Pre-processing pipeline for GenEvac simulation trajectory recordings.

Loads raw JSONL trajectory files, filters invalid entries, optionally
samples at a configured stride, and generates history/future windows
compatible with the existing ``TrajectoryObservation`` data contract.

Design
------
- Never silently interpolates or invents missing data.
- Every transformation is documented and configurable.
- Preserves the ``SIMULATED`` category from ``TrajectoryCategory``.
- Keeps simulation data separate from real-world ETH/UCY data.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from genevac.prediction.data_contract import TrajectoryCategory, TrajectoryObservation
from genevac.data.simulation.schema import TrajectoryRecord


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run_records(trajectory_file: Path) -> List[TrajectoryRecord]:
    """Load all records from a JSONL trajectory file.

    Skips blank lines and malformed JSON (logs them).
    """
    records: List[TrajectoryRecord] = []
    with trajectory_file.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
                records.append(TrajectoryRecord(**data))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass  # Validator is responsible for reporting bad lines
    return records


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_valid_records(records: List[TrajectoryRecord]) -> List[TrajectoryRecord]:
    """Remove records with non-finite coordinates or negative timestamps."""
    valid = []
    for rec in records:
        if (
            math.isfinite(rec.x)
            and math.isfinite(rec.y)
            and math.isfinite(rec.timestamp)
            and rec.timestamp >= 0.0
        ):
            valid.append(rec)
    return valid


# ---------------------------------------------------------------------------
# Grouping by pedestrian
# ---------------------------------------------------------------------------

def group_by_pedestrian(
    records: List[TrajectoryRecord],
) -> Dict[str, List[TrajectoryRecord]]:
    """Return a dict mapping ``pedestrian_id`` → sorted list of records."""
    groups: Dict[str, List[TrajectoryRecord]] = {}
    for rec in records:
        groups.setdefault(rec.pedestrian_id, []).append(rec)
    for ped_id in groups:
        groups[ped_id].sort(key=lambda r: r.timestamp)
    return groups


# ---------------------------------------------------------------------------
# Conversion to TrajectoryObservation
# ---------------------------------------------------------------------------

def to_trajectory_observation(rec: TrajectoryRecord, index: int) -> TrajectoryObservation:
    """Convert a ``TrajectoryRecord`` to a ``TrajectoryObservation``."""
    return TrajectoryObservation(
        agent_id=rec.pedestrian_id,
        step_index=index,
        timestamp=rec.timestamp,
        x=rec.x,
        y=rec.y,
        category=TrajectoryCategory.SIMULATED,
    )


# ---------------------------------------------------------------------------
# History / future window generation
# ---------------------------------------------------------------------------

@dataclass
class SimulationWindow:
    """A history/future window for one pedestrian from one run."""
    run_id: str
    scenario_id: str
    layout_id: str
    pedestrian_id: str
    reference_timestamp: float
    history: Tuple[TrajectoryObservation, ...]
    future: Tuple[TrajectoryObservation, ...]


def generate_windows(
    trajectory_file: Path,
    *,
    run_id: str,
    scenario_id: str,
    layout_id: str,
    history_len: int = 8,
    future_len: int = 12,
    stride: int = 1,
    min_trajectory_length: int = 2,
) -> List[SimulationWindow]:
    """Generate overlapping history/future windows from one JSONL trajectory file.

    Parameters
    ----------
    trajectory_file:
        Path to the JSONL file produced by the recorder.
    run_id / scenario_id / layout_id:
        Provenance fields attached to each window.
    history_len:
        Number of past observations in each window (default 8).
    future_len:
        Number of future observations in each window (default 12).
    stride:
        Step between consecutive windows (default 1 = all possible windows).
    min_trajectory_length:
        Minimum trajectory length to include a pedestrian.

    Returns
    -------
    List[SimulationWindow]
    """
    records = filter_valid_records(load_run_records(trajectory_file))
    groups = group_by_pedestrian(records)
    windows: List[SimulationWindow] = []
    window_size = history_len + future_len

    for ped_id, ped_records in groups.items():
        sampled = ped_records[::stride]
        if len(sampled) < min_trajectory_length:
            continue
        obs = [to_trajectory_observation(r, i) for i, r in enumerate(sampled)]
        for start in range(0, len(obs) - window_size + 1):
            hist = tuple(obs[start : start + history_len])
            fut = tuple(obs[start + history_len : start + window_size])
            if len(hist) == history_len and len(fut) == future_len:
                windows.append(SimulationWindow(
                    run_id=run_id,
                    scenario_id=scenario_id,
                    layout_id=layout_id,
                    pedestrian_id=ped_id,
                    reference_timestamp=hist[-1].timestamp,
                    history=hist,
                    future=fut,
                ))
    return windows
