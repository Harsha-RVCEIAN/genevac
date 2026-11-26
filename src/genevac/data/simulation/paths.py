"""Path constants for the simulation dataset storage.
All paths are expressed relative to the repository root (``PROJECT_ROOT``).
The directory hierarchy matches the specification in the task description.
"""

from pathlib import Path

# Repository root – this file lives under ``src/genevac/data/simulation/paths.py``;
# ``PROJECT_ROOT`` is computed by moving five levels up (parents[4]).
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Base directory for simulation recordings
SIMULATION_ROOT = PROJECT_ROOT / "data" / "simulation"

# Raw (unprocessed) recordings – versioned sub‑folders (e.g. v1)
RAW_ROOT = SIMULATION_ROOT / "raw"

# Processed data (windows, splits, etc.) – versioned as well
PROCESSED_ROOT = SIMULATION_ROOT / "processed"

# Helper to obtain a versioned sub‑directory, creating it lazily.
def versioned_path(base: Path, version: str) -> Path:
    """Return ``base / version`` and ensure the directory exists.

    Parameters
    ----------
    base: Path
        Either ``RAW_ROOT`` or ``PROCESSED_ROOT``.
    version: str
        Dataset version identifier, e.g. ``"v1"``.
    """
    p = base / version
    p.mkdir(parents=True, exist_ok=True)
    return p

# Standard version used by the implementation.
DEFAULT_VERSION = "v1"

# Exported convenience paths
RAW_V1 = versioned_path(RAW_ROOT, DEFAULT_VERSION)
PROCESSED_V1 = versioned_path(PROCESSED_ROOT, DEFAULT_VERSION)

# Sub‑folders inside the versioned raw directory
RUNS_ROOT = RAW_V1 / "runs"
TRAJECTORIES_ROOT = RAW_V1 / "trajectories"
MANIFESTS_ROOT = RAW_V1 / "manifests"
EVENTS_ROOT = RAW_V1 / "events"

# Ensure the common sub‑folders exist.
for _p in (RUNS_ROOT, TRAJECTORIES_ROOT, MANIFESTS_ROOT, EVENTS_ROOT):
    _p.mkdir(parents=True, exist_ok=True)
