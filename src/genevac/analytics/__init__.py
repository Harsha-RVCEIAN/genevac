"""Statistics, congestion, and spatial heatmap calculations."""

from typing import List, Tuple
import numpy as np


class HeatmapCalculator:
    def __init__(self, grid_width: float, grid_height: float, resolution: float = 1.0):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.resolution = resolution
        self.cols = int(np.ceil(grid_width / resolution))
        self.rows = int(np.ceil(grid_height / resolution))
        self.density_grid = np.zeros((self.rows, self.cols), dtype=float)

    def record_positions(self, positions: List[Tuple[float, float]]) -> None:
        for x, y in positions:
            col = int(x / self.resolution)
            row = int(y / self.resolution)
            if 0 <= col < self.cols and 0 <= row < self.rows:
                self.density_grid[row, col] += 1.0

    def get_density_matrix(self) -> np.ndarray:
        return self.density_grid
