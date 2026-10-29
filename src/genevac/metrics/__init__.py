"""Prediction accuracy and evacuation performance metrics."""

import math
from typing import List, Tuple


def calculate_ade(ground_truth: List[Tuple[float, float]], predicted: List[Tuple[float, float]]) -> float:
    """Average Displacement Error (ADE)."""
    if not ground_truth or not predicted or len(ground_truth) != len(predicted):
        raise ValueError("Ground truth and prediction sequences must be non-empty and of equal length.")
    total_disp = sum(math.hypot(gt[0] - pred[0], gt[1] - pred[1]) for gt, pred in zip(ground_truth, predicted))
    return total_disp / len(ground_truth)


def calculate_fde(ground_truth: List[Tuple[float, float]], predicted: List[Tuple[float, float]]) -> float:
    """Final Displacement Error (FDE)."""
    if not ground_truth or not predicted or len(ground_truth) != len(predicted):
        raise ValueError("Ground truth and prediction sequences must be non-empty and of equal length.")
    return math.hypot(ground_truth[-1][0] - predicted[-1][0], ground_truth[-1][1] - predicted[-1][1])


def calculate_clearance_time(step_count: int, dt: float) -> float:
    """Total clearance time in seconds."""
    return step_count * dt
