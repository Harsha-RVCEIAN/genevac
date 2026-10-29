import unittest
from genevac.metrics import calculate_ade, calculate_fde, calculate_clearance_time


class TestMetrics(unittest.TestCase):
    def test_metrics_calculation(self):
        gt = [(0.0, 0.0), (2.0, 2.0)]
        pred = [(0.0, 0.0), (2.0, 2.0)]
        self.assertEqual(calculate_ade(gt, pred), 0.0)
        self.assertEqual(calculate_fde(gt, pred), 0.0)

        gt2 = [(0.0, 0.0), (3.0, 4.0)]
        pred2 = [(0.0, 0.0), (0.0, 0.0)]
        self.assertEqual(calculate_fde(gt2, pred2), 5.0)
        self.assertEqual(calculate_clearance_time(100, 0.1), 10.0)


if __name__ == "__main__":
    unittest.main()
