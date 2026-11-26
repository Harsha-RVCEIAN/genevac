"""Run-level and layout-level splitting for GenEvac simulation datasets.

Rules
-----
- Splits are done at the **run level** (never window/agent level alone).
- All windows from the same run stay in the same split.
- Layout-level splitting is applied when ``layout_id`` metadata is available.
- Deterministic: given the same list of run IDs and ``seed``, the split
  assignment is always identical.
- Validation enforces non-empty train, val, and test splits by default.
- Leakage checks inspect actual split membership, window-level run_id
  membership, and held-out layout isolation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set


SPLIT_NAMES = ("train", "val", "test")


def _stable_hash(value: str, seed: int = 0) -> int:
    """Return a stable integer hash of *value* (independent of Python hash seed)."""
    digest = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    return int(digest, 16)


def _compute_split_counts(
    n: int,
    train_ratio: float,
    val_ratio: float,
    allow_empty_splits: bool,
) -> tuple[int, int]:
    """Validate ratios and compute (n_train, n_val) leaving n_test = n - n_train - n_val."""
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.0")
    if n == 0:
        raise ValueError("Cannot split an empty collection")
    if not allow_empty_splits and n < 3:
        raise ValueError(
            f"At least 3 items are required for non-empty train/val/test splits, got {n}"
        )

    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio)) if n >= 2 else 0
    if not allow_empty_splits:
        while n_train + n_val >= n:
            if n_train > n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                n_train -= 1
    return n_train, n_val


def split_run_ids(
    run_ids: Sequence[str],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    allow_empty_splits: bool = False,
) -> Dict[str, str]:
    """Assign each unique run ID to ``"train"``, ``"val"``, or ``"test"``."""
    unique_ids = list(dict.fromkeys(run_ids))
    if len(unique_ids) != len(run_ids):
        raise ValueError("Duplicate run_ids provided to split_run_ids")

    n_train, n_val = _compute_split_counts(
        len(unique_ids), train_ratio, val_ratio, allow_empty_splits
    )
    sorted_ids = sorted(unique_ids, key=lambda r: _stable_hash(r, seed))

    assignments: Dict[str, str] = {}
    for i, run_id in enumerate(sorted_ids):
        if i < n_train:
            assignments[run_id] = "train"
        elif i < n_train + n_val:
            assignments[run_id] = "val"
        else:
            assignments[run_id] = "test"
    return assignments


def split_by_layout(
    layout_to_run_ids: Dict[str, List[str]],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    allow_empty_splits: bool = False,
) -> Dict[str, str]:
    """Assign runs by layout group so all runs from the same layout share a split."""
    n_train, n_val = _compute_split_counts(
        len(layout_to_run_ids), train_ratio, val_ratio, allow_empty_splits
    )
    layout_ids = sorted(layout_to_run_ids.keys(), key=lambda lid: _stable_hash(lid, seed))

    assignments: Dict[str, str] = {}
    for i, layout_id in enumerate(layout_ids):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        for run_id in layout_to_run_ids[layout_id]:
            if run_id in assignments:
                raise ValueError(f"run_id={run_id!r} appears under multiple layouts")
            assignments[run_id] = split
    return assignments


def check_split_leakage(
    split_to_run_ids: Dict[str, Sequence[str]],
    *,
    run_to_layout: Optional[Dict[str, str]] = None,
    windows_by_split: Optional[Dict[str, Sequence[Any]]] = None,
    require_non_empty: Sequence[str] = SPLIT_NAMES,
) -> List[str]:
    """Verify actual split membership has no cross-split run, window, or layout leakage."""
    errors: List[str] = []

    for required in require_non_empty:
        if len(split_to_run_ids.get(required, ())) == 0:
            errors.append(f"split '{required}' is empty")

    seen_runs: Dict[str, str] = {}
    for split, runs in split_to_run_ids.items():
        for run_id in runs:
            prev_split = seen_runs.get(run_id)
            if prev_split is not None and prev_split != split:
                errors.append(
                    f"run_id={run_id!r} leaked across splits '{prev_split}' and '{split}'"
                )
            seen_runs[run_id] = split

    if run_to_layout is not None:
        seen_layouts: Dict[str, str] = {}
        for split, runs in split_to_run_ids.items():
            for run_id in runs:
                layout_id = run_to_layout.get(run_id)
                if layout_id is None:
                    errors.append(f"run_id={run_id!r} missing layout_id for layout leakage check")
                    continue
                prev_split = seen_layouts.get(layout_id)
                if prev_split is not None and prev_split != split:
                    errors.append(
                        f"layout_id={layout_id!r} leaked across splits '{prev_split}' and '{split}'"
                    )
                seen_layouts[layout_id] = split

    if windows_by_split is not None:
        window_run_splits: Dict[str, str] = {}
        for split, windows in windows_by_split.items():
            expected_runs: Set[str] = set(split_to_run_ids.get(split, ()))
            for win in windows:
                w_run_id = getattr(win, "run_id", None)
                if w_run_id is None:
                    errors.append(f"window in split '{split}' missing run_id")
                    continue
                if expected_runs and w_run_id not in expected_runs:
                    errors.append(
                        f"window with run_id={w_run_id!r} found in split '{split}' "
                        f"but run belongs to '{seen_runs.get(w_run_id, 'unassigned')}'"
                    )
                prev_w_split = window_run_splits.get(w_run_id)
                if prev_w_split is not None and prev_w_split != split:
                    errors.append(
                        f"windows for run_id={w_run_id!r} crossed splits '{prev_w_split}' and '{split}'"
                    )
                window_run_splits[w_run_id] = split

    return errors


def check_leakage(
    assignments: Dict[str, str],
    *,
    run_to_layout: Optional[Dict[str, str]] = None,
    windows_by_split: Optional[Dict[str, Sequence[Any]]] = None,
    require_non_empty: Sequence[str] = SPLIT_NAMES,
) -> List[str]:
    """Build per-split membership from *assignments* and run :func:`check_split_leakage`."""
    split_to_runs: Dict[str, List[str]] = {s: [] for s in SPLIT_NAMES}
    for run_id, split in assignments.items():
        split_to_runs.setdefault(split, []).append(run_id)
    return check_split_leakage(
        split_to_runs,
        run_to_layout=run_to_layout,
        windows_by_split=windows_by_split,
        require_non_empty=require_non_empty,
    )


def write_split_manifest(
    assignments: Dict[str, str],
    dest: Path,
    *,
    seed: int = 42,
    extra_meta: Optional[Dict] = None,
) -> None:
    """Write split assignments and per-split counts to a JSON manifest."""
    counts = {s: sum(1 for v in assignments.values() if v == s) for s in SPLIT_NAMES}
    payload: Dict[str, Any] = {
        "schema_version": "v1",
        "split_seed": seed,
        "counts": counts,
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
