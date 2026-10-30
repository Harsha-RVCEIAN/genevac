import unittest

from genevac.prediction import (
    LinearTrajectoryPredictor,
    TrajectoryHistoryBuffer,
    TrajectoryObservation,
)


class TestPrediction(unittest.TestCase):
    def test_history_buffer(self):
        buf = TrajectoryHistoryBuffer(window_size=3)
        buf.record("a1", (0.0, 0.0), step_index=0, timestamp=0.0)
        buf.record("a1", (1.0, 1.0), step_index=1, timestamp=0.1)
        buf.record("a1", (2.0, 2.0), step_index=2, timestamp=0.2)
        buf.record("a1", (3.0, 3.0), step_index=3, timestamp=0.3)
        self.assertEqual(len(buf.get_trajectory("a1")), 3)
        self.assertEqual(buf.get_trajectory("a1")[-1], (3.0, 3.0))
        obs = buf.get_observations("a1")
        self.assertEqual(len(obs), 3)
        self.assertEqual(obs[-1].step_index, 3)
        self.assertEqual(obs[-1].timestamp, 0.3)

    def test_history_buffer_with_observation_object(self):
        buf = TrajectoryHistoryBuffer(window_size=2)
        obs1 = TrajectoryObservation(agent_id="ped_01", step_index=10, timestamp=1.0, x=5.0, y=10.0)
        obs2 = TrajectoryObservation(agent_id="ped_01", step_index=11, timestamp=1.1, x=5.5, y=10.2)
        buf.record(obs1)
        buf.record(obs2)
        self.assertEqual(len(buf.get_observations("ped_01")), 2)
        self.assertEqual(buf.get_trajectory("ped_01")[0], (5.0, 10.0))
        self.assertEqual(buf.get_trajectory("ped_01")[1], (5.5, 10.2))

    def test_linear_predictor(self):
        predictor = LinearTrajectoryPredictor()
        history = [(0.0, 0.0), (1.0, 2.0)]
        preds = predictor.predict(history, horizon=2)
        self.assertEqual(len(preds), 2)
        self.assertEqual(preds[0], (2.0, 4.0))
        self.assertEqual(preds[1], (3.0, 6.0))


if __name__ == "__main__":
    unittest.main()
