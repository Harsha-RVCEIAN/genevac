"""Run-level and layout-level splitting for GenEvac simulation datasets.

Rules
-----
- Splits are done at the **run level** (never window/agent level alone).
- All windows from the same run stay in the same split.
- Layout-level splitting is applied when ``layout_id`` metadata is available.
- Deterministic: given the same list of run IDs and ``seed``, the split
  assignment is always identical.
- A leakage checker verifies that no run appears in more than one split.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Deterministic grouping helper
# ---------------------------------------------------------------------------

def _stable_hash(value: str, seed: int = 0) -> int:
    """Return a stable integer hash of *value* (independent of Python hash seed)."""
    digest = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    return int(digest, 16)


# ---------------------------------------------------------------------------
# Run-level split
# ---------------------------------------------------------------------------

def split_run_ids(
    run_ids: List[str],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, str]:
    """Assign each run ID to ``"train"``, ``"val"``, or ``"test"``.

    Parameters
    ----------
    run_ids:
        List of unique run identifiers.
    train_ratio / val_ratio:
        Fractions of runs allocated to train and validation sets.
        Test = 1 - train_ratio - val_ratio.
    seed:
        Integer seed for deterministic shuffling.

    Returns
    -------
    dict
        ``{run_id: split_name}`` mapping.
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1")

    # Sort then assign deterministically via hash bucket
    sorted_ids = sorted(run_ids, key=lambda r: _stable_hash(r, seed))
    n = len(sorted_ids)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    # Remaining go to test
    assignments: Dict[str, str] = {}
    for i, run_id in enumerate(sorted_ids):
        if i < n_train:
            assignments[run_id] = "train"
        elif i < n_train + n_val:
            assignments[run_id] = "val"
        else:
            assignments[run_id] = "test"
    return assignments


# ---------------------------------------------------------------------------
# Layout-level split (when available)
# ---------------------------------------------------------------------------

def split_by_layout(
    layout_to_run_ids: Dict[str, List[str]],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, str]:
    """Assign runs by layout group so all runs from the same layout share a split.

    Parameters
    ----------
    layout_to_run_ids:
        Dict mapping ``layout_id`` to the list of run IDs using that layout.

    Returns
    -------
    dict
        ``{run_id: split_name}`` mapping.
    """
    layout_ids = sorted(layout_to_run_ids.keys(), key=lambda l: _stable_hash(l, seed))
    n = len(layout_ids)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))

    assignments: Dict[str, str] = {}
    for i, layout_id in enumerate(layout_ids):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        for run_id in layout_to_run_ids[layout_id]:
            assignments[run_id] = split
    return assignments


# ---------------------------------------------------------------------------
# Leakage checker
# ---------------------------------------------------------------------------

def check_leakage(assignments: Dict[str, str]) -> List[str]:
    """Return a list of error messages for any run assigned to multiple splits.

    In a correct split each run maps to exactly one split, so this function
    should always return an empty list.  It is exposed for testing.
    """
    errors: List[str] = []
    split_to_runs: Dict[str, List[str]] = {}
    for run_id, split in assignments.items():
        split_to_runs.setdefault(split, []).append(run_id)

    # Count runs per split – check that the union equals the full set
    all_assigned = set(assignments.keys())
    seen: set = set()
    for split, runs in split_to_runs.items():
        for run_id in runs:
            if run_id in seen:
                errors.append(f"run_id={run_id!r} appears in multiple splits")
            seen.add(run_id)

    missing = all_assigned - seen
    for run_id in missing:
        errors.append(f"run_id={run_id!r} has no split assignment")
    return errors


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def write_split_manifest(
    assignments: Dict[str, str],
    dest: Path,
    *,
    seed: int = 42,
    extra_meta: Optional[Dict] = None,
) -> None:
    """Write split assignments to a JSON manifest."""
    payload = {
        "schema_version": "v1",
        "split_seed": seed,
        "assignments": assignments,
    }
    if extra_meta:
        payload.update(extra_meta)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_manifest(path: Path) -> Dict[str, str]:
    """Load and return the ``{run_id: split}`` dict from a manifest."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["assignments"]
