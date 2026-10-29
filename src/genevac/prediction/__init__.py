"""AI prediction models and trajectory history buffer."""

from typing import Dict, List, Tuple
from collections import deque


class TrajectoryHistoryBuffer:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.history: Dict[str, deque] = {}

    def record(self, agent_id: str, position: Tuple[float, float]) -> None:
        if agent_id not in self.history:
            self.history[agent_id] = deque(maxlen=self.window_size)
        self.history[agent_id].append(position)

    def get_trajectory(self, agent_id: str) -> List[Tuple[float, float]]:
        return list(self.history.get(agent_id, []))


class LinearTrajectoryPredictor:
    """Baseline predictor assuming constant velocity extrapolation."""
    def predict(self, history: List[Tuple[float, float]], horizon: int = 10) -> List[Tuple[float, float]]:
        if len(history) < 2:
            return [history[-1]] * horizon if history else []
        dx = history[-1][0] - history[-2][0]
        dy = history[-1][1] - history[-2][1]
        
        predictions = []
        curr_x, curr_y = history[-1]
        for _ in range(horizon):
            curr_x += dx
            curr_y += dy
            predictions.append((curr_x, curr_y))
        return predictions
