"""Trajectory prediction error metrics and validation."""

from __future__ import annotations

import math
from typing import Sequence, Tuple


def calculate_ade(
    ground_truth: Sequence[Tuple[float, float]],
    predicted: Sequence[Tuple[float, float]],
) -> float:
    """Compute Average Displacement Error (ADE) in meters.

    Args:
        ground_truth: Sequence of (x, y) ground truth positions.
        predicted: Sequence of (x, y) predicted positions.

    Returns:
        Average Euclidean displacement error.

    Raises:
        ValueError: If either sequence is empty, lengths differ, or coordinates are non-finite.
    """
    if not ground_truth or not predicted:
        raise ValueError("Ground truth and predicted sequences must not be empty.")
    if len(ground_truth) != len(predicted):
        raise ValueError(
            f"Ground truth length ({len(ground_truth)}) and predicted length ({len(predicted)}) must match."
        )

    total_error = 0.0
    for idx, (gt, pr) in enumerate(zip(ground_truth, predicted)):
        if len(gt) != 2 or len(pr) != 2:
            raise ValueError(f"Points at index {idx} must be (x, y) tuples.")
        gx, gy = float(gt[0]), float(gt[1])
        px, py = float(pr[0]), float(pr[1])
        if not (math.isfinite(gx) and math.isfinite(gy) and math.isfinite(px) and math.isfinite(py)):
            raise ValueError(f"Non-finite coordinates detected at index {idx}.")
        total_error += math.hypot(gx - px, gy - py)

    return total_error / len(ground_truth)


def calculate_fde(
    ground_truth: Sequence[Tuple[float, float]],
    predicted: Sequence[Tuple[float, float]],
) -> float:
    """Compute Final Displacement Error (FDE) in meters.

    Args:
        ground_truth: Sequence of (x, y) ground truth positions.
        predicted: Sequence of (x, y) predicted positions.

    Returns:
        Euclidean error at the final timestep.

    Raises:
        ValueError: If either sequence is empty, lengths differ, or final coordinates are non-finite.
    """
    if not ground_truth or not predicted:
        raise ValueError("Ground truth and predicted sequences must not be empty.")
    if len(ground_truth) != len(predicted):
        raise ValueError(
            f"Ground truth length ({len(ground_truth)}) and predicted length ({len(predicted)}) must match."
        )

    gt_final = ground_truth[-1]
    pr_final = predicted[-1]
    gx, gy = float(gt_final[0]), float(gt_final[1])
    px, py = float(pr_final[0]), float(pr_final[1])
    if not (math.isfinite(gx) and math.isfinite(gy) and math.isfinite(px) and math.isfinite(py)):
        raise ValueError("Non-finite coordinates detected in final position.")

    return math.hypot(gx - px, gy - py)


def calculate_prediction_coverage(matched_count: int, expected_count: int) -> float:
    """Compute prediction coverage ratio in range [0.0, 1.0]."""
    if expected_count <= 0:
        return 0.0
    if matched_count < 0:
        raise ValueError("matched_count cannot be negative.")
    return min(1.0, max(0.0, matched_count / expected_count))

