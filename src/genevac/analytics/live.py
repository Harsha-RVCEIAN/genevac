"""Live analytics pipeline, metric snapshot builder, and chart series generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from genevac.analytics.heatmap import GridSpecification, SpatialGridResult, SpatialHeatmapEngine
from genevac.metrics import (
    calculate_area_density,
    calculate_clearance_time,
    calculate_exit_throughput,
    calculate_minimum_separation,
)
from genevac.prediction.adapter import SimulationTrajectoryAdapter
from genevac.prediction.data_contract import TrajectorySnapshot


class AnalyticsFreshnessMode(str, Enum):
    LIVE = "LIVE"
    REPLAY = "REPLAY"


class MetricStatus(str, Enum):
    VALID = "VALID"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"
    NOT_COMPUTED = "NOT_COMPUTED"
    PAUSED = "PAUSED"


@dataclass(frozen=True)
class MetricRecord:
    """Individual metric calculation record."""
    metric_id: str
    display_name: str
    value: Optional[float]
    unit: str
    status: str = "VALID"  # VALID, UNAVAILABLE, STALE, NOT_COMPUTED, PAUSED
    sample_count: int = 0
    source_step: int = 0
    source_timestamp: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class ChartPoint:
    """A single data point within a time series."""
    timestamp: float
    step_index: int
    value: Optional[float]
    sample_count: int = 0
    run_id: str = "default_run"


@dataclass
class ChartSeries:
    """Renderer-neutral time-series with deduplication and bounded memory."""
    series_id: str
    metric_id: str
    display_label: str
    unit: str
    run_id: str = "default_run"
    max_points: int = 500
    _points: List[ChartPoint] = field(default_factory=list)
    _seen_steps: Set[int] = field(default_factory=set)

    @property
    def points(self) -> Tuple[ChartPoint, ...]:
        return tuple(self._points)

    def append_point(self, point: ChartPoint) -> bool:
        """Append point idempotently with step deduplication and size bounding."""
        if point.step_index in self._seen_steps:
            return False

        self._points.append(point)
        self._seen_steps.add(point.step_index)

        # Bounded retention
        if len(self._points) > self.max_points:
            removed = self._points.pop(0)
            self._seen_steps.discard(removed.step_index)

        return True

    def clear(self) -> None:
        self._points.clear()
        self._seen_steps.clear()


@dataclass(frozen=True)
class LiveAnalyticsSnapshot:
    """Consolidated analytics snapshot representing system state at a specific step."""
    run_id: str
    source_step: int
    source_timestamp: float
    refresh_timestamp: float
    has_new_data: bool
    is_paused: bool
    is_stale: bool
    metrics: Dict[str, MetricRecord]
    spatial_result: Optional[SpatialGridResult]
    active_agent_ids: Tuple[str, ...]
    blocked_agent_ids: Tuple[str, ...]
    evacuated_agent_ids: Tuple[str, ...]
    not_observed_agent_ids: Tuple[str, ...]


class LiveAnalyticsPipeline:
    """Read-only orchestration pipeline for live analytics and chart series."""

    def __init__(
        self,
        grid_spec: Optional[GridSpecification] = None,
        observation_interval: float = 0.4,
        run_id: str = "run_1",
        max_series_points: int = 500,
        mode: AnalyticsFreshnessMode = AnalyticsFreshnessMode.LIVE,
        stale_timeout_seconds: float = 5.0,
    ) -> None:
        self.run_id = str(run_id)
        self.observation_interval = float(observation_interval)
        self.mode = mode
        self.stale_timeout_seconds = float(stale_timeout_seconds)
        self.adapter = SimulationTrajectoryAdapter(observation_interval=observation_interval)

        self.grid_spec = grid_spec or GridSpecification(
            min_x=0.0, min_y=0.0, max_x=50.0, max_y=50.0, cell_size_x=2.0, cell_size_y=2.0
        )
        self.heatmap_engine = SpatialHeatmapEngine(self.grid_spec)

        self._last_processed_step: int = -1
        self._last_processed_timestamp: float = -1.0
        self._last_data_received_wall_time: float = time.time()
        self._last_refresh_wall_time: float = 0.0
        self._latest_snapshot: Optional[LiveAnalyticsSnapshot] = None

        # Chart series containers
        self.series: Dict[str, ChartSeries] = {
            "active_count": ChartSeries("series_active", "active_count", "Active Pedestrians", "count", self.run_id, max_series_points),
            "blocked_count": ChartSeries("series_blocked", "blocked_count", "Blocked Pedestrians", "count", self.run_id, max_series_points),
            "evacuated_count": ChartSeries("series_evacuated", "evacuated_count", "Evacuated Pedestrians", "count", self.run_id, max_series_points),
            "min_separation": ChartSeries("series_min_sep", "min_separation", "Minimum Separation", "m", self.run_id, max_series_points),
            "exit_throughput": ChartSeries("series_throughput", "exit_throughput", "Exit Throughput", "peds/s", self.run_id, max_series_points),
        }

    def reset(self, new_run_id: Optional[str] = None) -> None:
        """Reset all pipeline state, adapter, heatmap engine, and series."""
        if new_run_id is not None:
            self.run_id = str(new_run_id)
        self.adapter.reset()
        self.heatmap_engine.reset(new_run_id=self.run_id)
        self._last_processed_step = -1
        self._last_processed_timestamp = -1.0
        self._last_data_received_wall_time = time.time()
        self._latest_snapshot = None
        for s in self.series.values():
            s.run_id = self.run_id
            s.clear()

    def poll_simulation(
        self,
        engine: Any,
        force: bool = False,
        run_id: Optional[str] = None,
        is_paused: bool = False,
    ) -> LiveAnalyticsSnapshot:
        """Non-invasively poll running simulation engine and update live metrics and series.

        DOES NOT call step(), run(), or advance() on the engine.
        """
        current_wall_time = time.time()
        current_run_id = str(run_id) if run_id is not None else self.run_id

        # Detect run transition
        if current_run_id != self.run_id:
            self.reset(new_run_id=current_run_id)

        step = getattr(engine, "current_step", getattr(engine, "step_count", 0))

        # Check for step rollback / reset signal
        if step < self._last_processed_step:
            self.reset(new_run_id=current_run_id)

        # Update adapter (returns number of newly accepted observations)
        new_obs_count = self.adapter.update_from_simulation(engine, force=force)
        has_new_data = (step != self._last_processed_step) and (new_obs_count > 0 or force)

        if has_new_data:
            self._last_data_received_wall_time = current_wall_time

        timestep = getattr(engine, "timestep", getattr(engine, "dt", 0.1))
        sim_time = step * timestep

        # Produce canonical trajectory snapshot
        traj_snap = self.adapter.get_prediction_snapshot(
            current_step=step, timestep=timestep, cohort="all_tracked", max_observation_age=None
        )

        # Update heatmap
        spatial_res = self.heatmap_engine.aggregate_snapshot(
            traj_snap, run_id=self.run_id, accumulate_dwell=has_new_data
        )

        # Detect staleness: only in LIVE mode when not paused and timeout elapsed
        is_stale = False
        if self.mode == AnalyticsFreshnessMode.LIVE and not is_paused:
            if (current_wall_time - self._last_data_received_wall_time) > self.stale_timeout_seconds:
                is_stale = True

        # Compute core metrics
        active_ids = sorted(self.adapter.active_agents)
        blocked_ids = sorted(self.adapter.blocked_agents)
        evac_ids = sorted(self.adapter.evacuated_agents)
        not_obs_ids = sorted(self.adapter.not_observed_agents)

        # Active positions for crowd metrics
        active_positions = [
            traj_snap.trajectories[aid][-1].position
            for aid in active_ids
            if aid in traj_snap.trajectories and traj_snap.trajectories[aid]
        ]

        min_sep = calculate_minimum_separation(active_positions)
        min_sep_val = min_sep if math.isfinite(min_sep) else None

        clear_time = calculate_clearance_time(step, timestep)

        # Exit throughput: Check if engine exposes explicit exit events
        throughput_val: Optional[float] = None
        throughput_status = MetricStatus.UNAVAILABLE
        throughput_reason = "Authoritative exit events not provided by engine"

        engine_events = getattr(engine, "events", None)
        if engine_events is not None and callable(getattr(engine, "get_events", None)):
            try:
                exit_evts = engine.get_events(event_type="exit")
                # Calculate throughput over recent window (default 5.0s)
                window_duration = 5.0
                recent_exits = [
                    e for e in exit_evts
                    if (sim_time - getattr(e, "timestamp", 0.0)) <= window_duration
                ]
                throughput_val = calculate_exit_throughput(len(recent_exits), window_duration)
                throughput_status = MetricStatus.VALID
                throughput_reason = ""
            except Exception as err:
                throughput_status = MetricStatus.UNAVAILABLE
                throughput_reason = f"Error querying engine exit events: {err}"

        # Status resolution
        base_status = MetricStatus.PAUSED if is_paused else (MetricStatus.STALE if is_stale else MetricStatus.VALID)

        metrics: Dict[str, MetricRecord] = {
            "active_count": MetricRecord(
                "active_count", "Active Pedestrians", float(len(active_ids)), "count",
                base_status, len(active_ids), step, sim_time
            ),
            "blocked_count": MetricRecord(
                "blocked_count", "Blocked Pedestrians", float(len(blocked_ids)), "count",
                base_status, len(blocked_ids), step, sim_time
            ),
            "evacuated_count": MetricRecord(
                "evacuated_count", "Evacuated Pedestrians", float(len(evac_ids)), "count",
                base_status, len(evac_ids), step, sim_time
            ),
            "not_observed_count": MetricRecord(
                "not_observed_count", "Not Observed Pedestrians", float(len(not_obs_ids)), "count",
                base_status, len(not_obs_ids), step, sim_time
            ),
            "clearance_time": MetricRecord(
                "clearance_time", "Total Elapsed Simulation Time", clear_time, "s",
                base_status, 1, step, sim_time
            ),
            "min_separation": MetricRecord(
                "min_separation", "Minimum Separation", min_sep_val, "m",
                base_status if min_sep_val is not None else MetricStatus.UNAVAILABLE,
                len(active_positions), step, sim_time,
                reason="Requires at least 2 active pedestrians" if min_sep_val is None else ""
            ),
            "exit_throughput": MetricRecord(
                "exit_throughput", "Exit Throughput", throughput_val, "peds/s",
                throughput_status if not is_stale else MetricStatus.STALE,
                len(evac_ids), step, sim_time,
                reason=throughput_reason
            ),
        }

        # Update series if new data arrived
        if has_new_data:
            self.series["active_count"].append_point(ChartPoint(sim_time, step, float(len(active_ids)), len(active_ids), self.run_id))
            self.series["blocked_count"].append_point(ChartPoint(sim_time, step, float(len(blocked_ids)), len(blocked_ids), self.run_id))
            self.series["evacuated_count"].append_point(ChartPoint(sim_time, step, float(len(evac_ids)), len(evac_ids), self.run_id))
            if min_sep_val is not None:
                self.series["min_separation"].append_point(ChartPoint(sim_time, step, min_sep_val, len(active_positions), self.run_id))
            if throughput_val is not None:
                self.series["exit_throughput"].append_point(ChartPoint(sim_time, step, throughput_val, len(evac_ids), self.run_id))
            self._last_processed_step = step
            self._last_processed_timestamp = sim_time

        self._last_refresh_wall_time = current_wall_time

        snapshot = LiveAnalyticsSnapshot(
            run_id=self.run_id,
            source_step=step,
            source_timestamp=sim_time,
            refresh_timestamp=current_wall_time,
            has_new_data=has_new_data,
            is_paused=is_paused,
            is_stale=is_stale,
            metrics=metrics,
            spatial_result=spatial_res,
            active_agent_ids=tuple(active_ids),
            blocked_agent_ids=tuple(blocked_ids),
            evacuated_agent_ids=tuple(evac_ids),
            not_observed_agent_ids=tuple(not_obs_ids),
        )
        self._latest_snapshot = snapshot
        return snapshot
