"""Crowd spatial metrics, separation, and local density."""

from __future__ import annotations

import math
from typing import Sequence, Tuple


def calculate_minimum_separation(
    positions: Sequence[Tuple[float, float]],
) -> float:
    """Compute the minimum Euclidean separation between any pair of agents.

    Args:
        positions: Sequence of (x, y) agent positions at a given timestamp.

    Returns:
        Minimum distance in meters, or float('inf') if fewer than 2 agents exist.
    """
    if len(positions) < 2:
        return float("inf")

    min_dist = float("inf")
    n = len(positions)
    for i in range(n):
        x1, y1 = positions[i]
        for j in range(i + 1, n):
            x2, y2 = positions[j]
            d = math.hypot(x1 - x2, y1 - y2)
            if d < min_dist:
                min_dist = d
    return min_dist


def calculate_area_density(
    agent_positions: Sequence[Tuple[float, float]],
    bounds: Tuple[float, float, float, float],
) -> float:
    """Compute crowd density inside a rectangular area in agents / m^2.

    Args:
        agent_positions: Sequence of (x, y) coordinates.
        bounds: (min_x, min_y, max_x, max_y) in meters.

    Returns:
        Density in agents / m^2.
    """
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        raise ValueError(f"Area bounds must have positive dimensions: width={width}, height={height}")

    area = width * height
    count = 0
    for x, y in agent_positions:
        if min_x <= x <= max_x and min_y <= y <= max_y:
            count += 1
    return count / area

