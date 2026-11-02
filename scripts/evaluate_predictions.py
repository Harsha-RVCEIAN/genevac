"""Evaluate AI prediction performance on ground truth trajectories using ConstantVelocityPredictor."""

import argparse
from pathlib import Path

from genevac.prediction import (
    ConstantVelocityPredictor,
    DatasetAdapter,
    PredictionEvaluator,
    SampleGenerator,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectory prediction models on datasets.")
    parser.add_argument("--data", type=str, default="data/raw/eth/test/biwi_eth.txt", help="Path to evaluation dataset")
    parser.add_argument("--format", type=str, choices=["eth", "json"], default="eth", help="Dataset format ('eth' or 'json')")
    parser.add_argument("--history", type=int, default=8, help="Number of historical observation steps")
    parser.add_argument("--horizon", type=int, default=12, help="Number of future prediction steps")
    parser.add_argument("--stride", type=int, default=1, help="Sliding window stride")
    parser.add_argument("--out_json", type=str, default="data/processed/evaluation_report.json", help="Output path for JSON report")
    parser.add_argument("--out_csv", type=str, default="data/processed/evaluation_results.csv", help="Output path for CSV per-trajectory metrics")
    args = parser.parse_args()

    data_path = Path(args.data)
    print(f"=== GenEvac-Sentinel Part 4: Trajectory Prediction Evaluation ===")
    print(f"Dataset:            {data_path}")
    print(f"History Length:     {args.history} observations")
    print(f"Prediction Horizon: {args.horizon} observations")
    print(f"Evaluation Model:   ConstantVelocityPredictor")

    # Load trajectories
    if args.format == "eth" and data_path.is_file():
        trajectories = DatasetAdapter.load_eth_file(data_path)
    elif data_path.is_file() and data_path.suffix == ".json":
        trajectories = DatasetAdapter.load_sample_json(data_path)
    else:
        # Fallback to sample JSON
        sample_path = Path("data/sample/sample_trajectories.json")
        print(f"Note: '{data_path}' not found or specified as json. Falling back to {sample_path}")
        trajectories = DatasetAdapter.load_sample_json(sample_path)

    print(f"Loaded {len(trajectories)} distinct pedestrian tracks.")

    # Generate sliding window evaluation samples
    generator = SampleGenerator(
        history_length=args.history,
        prediction_horizon=args.horizon,
        stride=args.stride,
        allow_partial_future=False,
    )
    samples = generator.generate_samples(trajectories)
    print(f"Generated {len(samples)} evaluation windows (zero data leakage enforced).")

    if not samples:
        print("No evaluation windows met the length requirement. Lower --history or --horizon.")
        return

    # Run evaluation
    evaluator = PredictionEvaluator(predictor=ConstantVelocityPredictor())
    report = evaluator.evaluate_samples(samples)

    # Print summary
    print("\n--- Evaluation Results ---")
    print(f"Total Samples:       {report.total_samples}")
    print(f"Evaluated Samples:   {report.evaluated_samples}")
    print(f"Successful:          {report.successful_samples}")
    print(f"Partial:             {report.partial_samples}")
    print(f"Failed:              {report.failed_samples}")
    print(f"Mean ADE:            {report.mean_ade:.4f} m")
    print(f"Median ADE:          {report.median_ade:.4f} m")
    print(f"Mean FDE:            {report.mean_fde:.4f} m")
    print(f"Median FDE:          {report.median_fde:.4f} m")

    if report.horizon_mean_errors:
        print(f"Horizon Errors (m):  {[round(e, 4) for e in report.horizon_mean_errors]}")

    # Save outputs
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    report.save_json(out_json)
    report.save_csv(out_csv)
    print(f"\nSaved aggregate JSON report: {out_json}")
    print(f"Saved trajectory CSV report: {out_csv}")


if __name__ == "__main__":
    main()
