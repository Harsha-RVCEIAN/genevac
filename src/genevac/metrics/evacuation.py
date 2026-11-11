"""Evacuation performance and exit dynamics metrics."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Tuple


def calculate_clearance_time(step_count: int, dt: float) -> float:
    """Compute total clearance / evacuation time in seconds.

    Args:
        step_count: Number of simulation steps executed until clearance.
        dt: Authoritative simulation timestep in seconds.

    Returns:
        Clearance time in seconds.
    """
    if step_count < 0:
        raise ValueError(f"step_count must be non-negative, got {step_count}")
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError(f"dt must be positive finite float, got {dt}")
    return step_count * dt


def calculate_agent_evacuation_times(
    exit_events: Sequence[Tuple[str, float]],
) -> Dict[str, float]:
    """Compute per-agent evacuation times from exit events.

    Args:
        exit_events: Sequence of (agent_id, exit_timestamp).

    Returns:
        Mapping of agent_id -> evacuation timestamp in seconds.
    """
    evac_times: Dict[str, float] = {}
    for aid, ts in exit_events:
        if not aid or not str(aid).strip():
            continue
        ts_f = float(ts)
        if not math.isfinite(ts_f) or ts_f < 0.0:
            raise ValueError(f"Invalid timestamp {ts!r} for agent '{aid}'")
        # Record first exit event for each agent
        if aid not in evac_times:
            evac_times[aid] = ts_f
    return evac_times


def calculate_exit_throughput(
    exit_counts: int,
    time_window: float,
) -> float:
    """Compute throughput at an exit in pedestrians per second.

    Args:
        exit_counts: Number of agents crossing the exit during the window.
        time_window: Window duration in seconds.

    Returns:
        Throughput in agents/second.
    """
    if exit_counts < 0:
        raise ValueError(f"exit_counts must be non-negative, got {exit_counts}")
    if time_window <= 0.0 or not math.isfinite(time_window):
        raise ValueError(f"time_window must be positive finite number, got {time_window}")
    return exit_counts / time_window


def calculate_evacuation_rate_series(
    evacuated_counts_by_step: Sequence[int],
    total_population: int,
) -> list[float]:
    """Compute cumulative evacuation percentage series over time.

    Args:
        evacuated_counts_by_step: Count of evacuated agents at each step.
        total_population: Total agent population count.

    Returns:
        List of proportions [0.0, 1.0] for each step.
    """
    if total_population <= 0:
        raise ValueError(f"total_population must be positive, got {total_population}")
    series: list[float] = []
    for count in evacuated_counts_by_step:
        if count < 0:
            raise ValueError(f"Evacuated count cannot be negative, got {count}")
        ratio = min(1.0, count / total_population)
        series.append(ratio)
    return series

