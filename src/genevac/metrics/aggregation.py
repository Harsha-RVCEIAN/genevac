"""Statistical aggregation utilities for multi-sample and multi-run metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class MetricSummary:
    """Summary statistics for a numeric metric series."""
    metric_name: str
    unit: str
    sample_count: int
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float


def aggregate_metric_series(
    values: Sequence[float],
    metric_name: str = "metric",
    unit: str = "",
) -> MetricSummary:
    """Compute summary statistics for a numeric series.

    Args:
        values: Sequence of numeric values.
        metric_name: Name of the metric.
        unit: Physical unit.

    Returns:
        MetricSummary containing count, mean, median, std_dev, min, max.
    """
    valid = [float(v) for v in values if math.isfinite(v)]
    if not valid:
        return MetricSummary(
            metric_name=metric_name,
            unit=unit,
            sample_count=0,
            mean=float("nan"),
            median=float("nan"),
            std_dev=float("nan"),
            min_val=float("nan"),
            max_val=float("nan"),
        )

    mean_val = statistics.fmean(valid)
    median_val = statistics.median(valid)
    std_val = statistics.stdev(valid) if len(valid) > 1 else 0.0

    return MetricSummary(
        metric_name=metric_name,
        unit=unit,
        sample_count=len(valid),
        mean=mean_val,
        median=median_val,
        std_dev=std_val,
        min_val=min(valid),
        max_val=max(valid),
    )

