"""Statistics, congestion, spatial heatmap, and live analytics."""

from typing import List, Tuple
import numpy as np

from genevac.analytics.heatmap import (
    GridSpecification,
    SpatialGridResult,
    SpatialHeatmapEngine,
)
from genevac.analytics.live import (
    ChartPoint,
    ChartSeries,
    LiveAnalyticsPipeline,
    LiveAnalyticsSnapshot,
    MetricRecord,
)


class HeatmapCalculator:
    """Backwards-compatible wrapper delegating to SpatialHeatmapEngine."""

    def __init__(self, grid_width: float, grid_height: float, resolution: float = 1.0):
        self.grid_width = float(grid_width)
        self.grid_height = float(grid_height)
        self.resolution = float(resolution)
        self.grid_spec = GridSpecification(
            min_x=0.0,
            min_y=0.0,
            max_x=self.grid_width,
            max_y=self.grid_height,
            cell_size_x=self.resolution,
            cell_size_y=self.resolution,
        )
        self.engine = SpatialHeatmapEngine(self.grid_spec)
        self.cols = self.grid_spec.cols
        self.rows = self.grid_spec.rows
        self.density_grid = np.zeros((self.rows, self.cols), dtype=float)

    def record_positions(self, positions: List[Tuple[float, float]]) -> None:
        result = self.engine.aggregate_positions(positions)
        self.density_grid += result.occupancy_grid.astype(float)

    def get_density_matrix(self) -> np.ndarray:
        return self.density_grid


__all__ = [
    "ChartPoint",
    "ChartSeries",
    "GridSpecification",
    "HeatmapCalculator",
    "LiveAnalyticsPipeline",
    "LiveAnalyticsSnapshot",
    "MetricRecord",
    "SpatialGridResult",
    "SpatialHeatmapEngine",
]
