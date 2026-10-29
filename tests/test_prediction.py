import unittest
from genevac.prediction import TrajectoryHistoryBuffer, LinearTrajectoryPredictor


class TestPrediction(unittest.TestCase):
    def test_history_buffer(self):
        buf = TrajectoryHistoryBuffer(window_size=3)
        buf.record("a1", (0.0, 0.0))
        buf.record("a1", (1.0, 1.0))
        buf.record("a1", (2.0, 2.0))
        buf.record("a1", (3.0, 3.0))
        self.assertEqual(len(buf.get_trajectory("a1")), 3)
        self.assertEqual(buf.get_trajectory("a1")[-1], (3.0, 3.0))

    def test_linear_predictor(self):
        predictor = LinearTrajectoryPredictor()
        history = [(0.0, 0.0), (1.0, 2.0)]
        preds = predictor.predict(history, horizon=2)
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0], (2.0, 4.0))
        self.assertEqual(preds[1], (3.0, 6.0))


if __name__ == "__main__":
    unittest.main()
