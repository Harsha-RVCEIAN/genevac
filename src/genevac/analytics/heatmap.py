"""Deterministic numeric spatial aggregation engine for GenEvac-Sentinel.

Produces validated, unit-aware spatial grids:
- Occupancy count matrix [row, col] (integer count per snapshot)
- Physical density matrix [row, col] (peds / m^2)
- Dwell time matrix [row, col] (agent-seconds per cell)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from genevac.prediction.data_contract import TrajectorySnapshot


@dataclass(frozen=True)
class GridSpecification:
    """Immutable specification of 2D spatial grid bounds and resolution."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    cell_size_x: float
    cell_size_y: float

    def __post_init__(self) -> None:
        for name, val in (
            ("min_x", self.min_x),
            ("min_y", self.min_y),
            ("max_x", self.max_x),
            ("max_y", self.max_y),
            ("cell_size_x", self.cell_size_x),
            ("cell_size_y", self.cell_size_y),
        ):
            if type(val) not in (float, int) or isinstance(val, bool) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite numeric value, got {val!r}")

        if self.max_x <= self.min_x:
            raise ValueError(f"max_x ({self.max_x}) must be strictly greater than min_x ({self.min_x})")
        if self.max_y <= self.min_y:
            raise ValueError(f"max_y ({self.max_y}) must be strictly greater than min_y ({self.min_y})")
        if self.cell_size_x <= 0.0:
            raise ValueError(f"cell_size_x must be positive (> 0), got {self.cell_size_x}")
        if self.cell_size_y <= 0.0:
            raise ValueError(f"cell_size_y must be positive (> 0), got {self.cell_size_y}")

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def cols(self) -> int:
        return int(math.ceil(self.width / self.cell_size_x))

    @property
    def rows(self) -> int:
        return int(math.ceil(self.height / self.cell_size_y))

    @property
    def cell_area(self) -> float:
        """Nominal area of a full grid cell in m^2."""
        return self.cell_size_x * self.cell_size_y


@dataclass(frozen=True)
class SpatialGridResult:
    """Immutable container for computed spatial aggregation layers."""
    grid_spec: GridSpecification
    occupancy_grid: np.ndarray  # int64, shape (rows, cols)
    density_grid: np.ndarray    # float64, shape (rows, cols), unit: peds / m^2
    dwell_time_grid: Optional[np.ndarray] = None  # float64, unit: agent-seconds
    run_id: str = "default_run"
    cohort: str = "all"
    timestamp: float = 0.0
    step_index: int = 0
    total_agents_processed: int = 0
    accepted_agents: int = 0
    rejected_agents: int = 0
    rejection_reasons: Tuple[str, ...] = field(default_factory=tuple)
    is_duplicate: bool = False
    attribution_policy: str = "sample_hold_left_endpoint"

    def normalize_min_max(self, layer: str = "density") -> np.ndarray:
        """Return derived min-max normalized array scaled to [0.0, 1.0]."""
        arr = getattr(self, f"{layer}_grid")
        if arr is None:
            raise ValueError(f"Layer '{layer}' is not available.")
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        if max_v - min_v < 1e-9:
            return np.zeros_like(arr, dtype=float)
        return (arr - min_v) / (max_v - min_v)


class SpatialHeatmapEngine:
    """State-isolated numerical spatial grid and accumulation engine."""

    def __init__(
        self,
        grid_spec: GridSpecification,
        max_observation_gap: float = 2.0,
    ) -> None:
        if not isinstance(grid_spec, GridSpecification):
            raise TypeError(f"grid_spec must be a GridSpecification, got {type(grid_spec).__name__}")
        self.grid_spec = grid_spec
        self.max_observation_gap = float(max_observation_gap)
        self.dwell_time_accumulator = np.zeros((self.grid_spec.rows, self.grid_spec.cols), dtype=float)
        self._current_run_id: Optional[str] = None
        self._last_processed_step: int = -1
        self._last_processed_timestamp: float = -1.0
        self._seen_snapshot_keys: Set[Tuple[str, int]] = set()
        # Per-agent previous state for interval dwell attribution: agent_id -> (timestamp, row, col)
        self._agent_prev_state: Dict[str, Tuple[float, int, int]] = {}

    def reset(self, new_run_id: Optional[str] = None) -> None:
        """Reset internal run-scoped state and accumulators."""
        self.dwell_time_accumulator.fill(0.0)
        self._current_run_id = new_run_id
        self._last_processed_step = -1
        self._last_processed_timestamp = -1.0
        self._seen_snapshot_keys.clear()
        self._agent_prev_state.clear()

    def map_point_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Map a continuous world coordinate (x, y) to (row, col) grid indices.

        Boundary convention:
        - Half-open: [min_x, max_x) and [min_y, max_y)
        - Exact maximum boundary (x == max_x or y == max_y) maps to the final cell
        - Strictly out-of-bounds coordinates return None
        """
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        if x < self.grid_spec.min_x or x > self.grid_spec.max_x:
            return None
        if y < self.grid_spec.min_y or y > self.grid_spec.max_y:
            return None

        # Floor-based index
        col = int(math.floor((x - self.grid_spec.min_x) / self.grid_spec.cell_size_x))
        row = int(math.floor((y - self.grid_spec.min_y) / self.grid_spec.cell_size_y))

        # Handle boundary max edge inclusion
        if col >= self.grid_spec.cols:
            col = self.grid_spec.cols - 1
        if row >= self.grid_spec.rows:
            row = self.grid_spec.rows - 1

        return (row, col)

    def aggregate_positions(
        self,
        positions: Sequence[Tuple[float, float]],
        timestamp: float = 0.0,
        step_index: int = 0,
        run_id: str = "default_run",
        cohort: str = "all",
    ) -> SpatialGridResult:
        """Compute instantaneous occupancy and physical density for a collection of positions."""
        occupancy = np.zeros((self.grid_spec.rows, self.grid_spec.cols), dtype=np.int64)
        accepted = 0
        rejected = 0
        rejections: List[str] = []

        for idx, pos in enumerate(positions):
            if len(pos) != 2:
                rejected += 1
                rejections.append(f"Position {idx} does not have length 2.")
                continue
            x, y = float(pos[0]), float(pos[1])
            cell = self.map_point_to_cell(x, y)
            if cell is None:
                rejected += 1
                rejections.append(f"Position {idx} ({x}, {y}) out of grid bounds.")
                continue

            row, col = cell
            occupancy[row, col] += 1
            accepted += 1

        density = occupancy.astype(float) / self.grid_spec.cell_area

        return SpatialGridResult(
            grid_spec=self.grid_spec,
            occupancy_grid=occupancy,
            density_grid=density,
            dwell_time_grid=self.dwell_time_accumulator.copy(),
            run_id=run_id,
            cohort=cohort,
            timestamp=timestamp,
            step_index=step_index,
            total_agents_processed=len(positions),
            accepted_agents=accepted,
            rejected_agents=rejected,
            rejection_reasons=tuple(rejections),
            is_duplicate=False,
        )

    def aggregate_snapshot(
        self,
        snapshot: TrajectorySnapshot,
        run_id: str = "default_run",
        cohort: str = "all",
        accumulate_dwell: bool = True,
    ) -> SpatialGridResult:
        """Aggregate observations from a canonical TrajectorySnapshot into spatial grids.

        Guarantees:
        - Run isolation: Rejects mixing runs into the same accumulator.
        - Idempotence: Duplicate snapshot keys (run_id, step) do not increase counts or dwell.
        - Conservative Sample-Hold Dwell Attribution: delta_t = t_curr - t_prev attributed to previous cell.
        """
        if not isinstance(snapshot, TrajectorySnapshot):
            raise TypeError(f"Expected TrajectorySnapshot, got {type(snapshot).__name__}")

        # Run isolation check
        if self._current_run_id is None:
            self._current_run_id = run_id
        elif self._current_run_id != run_id:
            raise ValueError(
                f"Cannot process snapshot for run_id '{run_id}' in accumulator initialized for '{self._current_run_id}'. "
                f"Call reset(new_run_id) explicitly before aggregating a new run."
            )

        step = snapshot.current_step
        ts = snapshot.current_timestamp
        snapshot_key = (run_id, step)

        # Duplicate protection check
        if snapshot_key in self._seen_snapshot_keys:
            # Return last valid state marked as duplicate
            occ = np.zeros((self.grid_spec.rows, self.grid_spec.cols), dtype=np.int64)
            dens = np.zeros((self.grid_spec.rows, self.grid_spec.cols), dtype=float)
            return SpatialGridResult(
                grid_spec=self.grid_spec,
                occupancy_grid=occ,
                density_grid=dens,
                dwell_time_grid=self.dwell_time_accumulator.copy(),
                run_id=run_id,
                cohort=cohort,
                timestamp=ts,
                step_index=step,
                total_agents_processed=0,
                accepted_agents=0,
                rejected_agents=0,
                rejection_reasons=(f"Duplicate snapshot key (run_id='{run_id}', step={step}) already processed.",),
                is_duplicate=True,
            )

        # Out-of-order check
        if ts < self._last_processed_timestamp:
            raise ValueError(
                f"Out-of-order snapshot timestamp ({ts:.4f}s < previous {self._last_processed_timestamp:.4f}s) "
                f"detected for run '{run_id}' at step {step}."
            )

        # Compute instantaneous occupancy
        occupancy = np.zeros((self.grid_spec.rows, self.grid_spec.cols), dtype=np.int64)
        accepted = 0
        rejected = 0
        rejections: List[str] = []

        current_agent_cells: Dict[str, Tuple[int, int]] = {}

        for aid, obs_seq in snapshot.trajectories.items():
            if not obs_seq:
                continue
            latest_obs = obs_seq[-1]
            cell = self.map_point_to_cell(latest_obs.x, latest_obs.y)
            if cell is None:
                rejected += 1
                rejections.append(f"Agent '{aid}' position ({latest_obs.x}, {latest_obs.y}) out of grid bounds.")
                continue

            row, col = cell
            occupancy[row, col] += 1
            accepted += 1
            current_agent_cells[aid] = cell

        density = occupancy.astype(float) / self.grid_spec.cell_area

        # Per-agent dwell time interval accumulation
        if accumulate_dwell:
            for aid, (curr_row, curr_col) in current_agent_cells.items():
                if aid in self._agent_prev_state:
                    prev_time, prev_row, prev_col = self._agent_prev_state[aid]
                    dt = ts - prev_time
                    # Only accumulate if strictly positive, finite, and within max_observation_gap
                    if 0.0 < dt <= self.max_observation_gap:
                        # Sample-hold: elapsed duration attributed to previous cell
                        self.dwell_time_accumulator[prev_row, prev_col] += dt

                # Update agent previous state
                self._agent_prev_state[aid] = (ts, curr_row, curr_col)

        self._seen_snapshot_keys.add(snapshot_key)
        self._last_processed_step = step
        self._last_processed_timestamp = ts

        return SpatialGridResult(
            grid_spec=self.grid_spec,
            occupancy_grid=occupancy,
            density_grid=density,
            dwell_time_grid=self.dwell_time_accumulator.copy(),
            run_id=run_id,
            cohort=cohort,
            timestamp=ts,
            step_index=step,
            total_agents_processed=accepted + rejected,
            accepted_agents=accepted,
            rejected_agents=rejected,
            rejection_reasons=tuple(rejections),
            is_duplicate=False,
            attribution_policy="sample_hold_left_endpoint",
        )
