"""Evaluate AI prediction performance on ground truth trajectories."""

from genevac.prediction import LinearTrajectoryPredictor
from genevac.metrics import calculate_ade, calculate_fde


def main():
    print("Evaluating AI trajectory predictions...")
    ground_truth = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    history = ground_truth[:2]
    future_gt = ground_truth[2:]

    predictor = LinearTrajectoryPredictor()
    predicted = predictor.predict(history, horizon=len(future_gt))

    ade = calculate_ade(future_gt, predicted)
    fde = calculate_fde(future_gt, predicted)

    print(f"Ground Truth Future: {future_gt}")
    print(f"Predicted Future:    {predicted}")
    print(f"ADE: {ade:.4f} m | FDE: {fde:.4f} m")


if __name__ == "__main__":
    main()
