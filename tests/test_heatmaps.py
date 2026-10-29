import unittest
from genevac.analytics import HeatmapCalculator


class TestHeatmaps(unittest.TestCase):
    def test_heatmap_density(self):
        calc = HeatmapCalculator(grid_width=10.0, grid_height=10.0, resolution=2.0)
        calc.record_positions([(1.0, 1.0), (1.5, 1.5), (5.0, 5.0)])
        grid = calc.get_density_matrix()
        self.assertEqual(grid[0, 0], 2.0)
        self.assertEqual(grid[2, 2], 1.0)


if __name__ == "__main__":
    unittest.main()
